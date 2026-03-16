"""Agent API endpoints for LLM-powered ML configuration suggestions.

This module provides two sets of endpoints:

1. Legacy single-step endpoints (suggest-config, suggest-dataset-spec, suggest-experiment-plan):
   - These call the LLM directly for a single suggestion
   - They are marked as deprecated and will be removed in a future version
   - Use the new pipeline endpoints instead for better observability

2. New multi-step pipeline endpoints:
   - POST /run-setup-pipeline: Runs the full 5-step setup pipeline
   - GET /runs: List agent runs for a project
   - GET /runs/{run_id}: Get a specific run with its steps and logs
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.api_key import LLMProvider
from app.models.project import Project, TaskType
from app.models.data_source import DataSource
from app.models.user import User
from app.models.dataset_spec import DatasetSpec
from app.models.experiment import Experiment
from app.models import AgentRun, AgentStep, AgentStepLog, AgentRunStatus
from app.services import api_key_service
from app.services.llm_client import get_llm_client, GeminiClient, OpenAIClient
from app.services.agent_service import (
    build_schema_summary,
    generate_project_config,
    generate_dataset_spec,
    generate_experiment_plan,
)
from app.core.exceptions import (
    LLMError,
    LLMTimeoutError,
    LLMParsingError,
    DataError,
    DatasetBuildError,
    StorageError,
    CeleryTaskError,
    AgentPipelineError,
)
from app.services.agent_executor import (
    create_setup_pipeline,
    run_setup_pipeline_for_project,
    run_agent_pipeline,
    get_agent_run_with_steps,
    list_agent_runs_for_project,
    create_results_pipeline,
    run_results_pipeline_for_experiment,
    create_data_architect_pipeline,
    run_data_architect_pipeline,
)
from app.schemas.agent import (
    SchemaSummary,
    ProjectConfigRequest,
    ProjectConfigResponse,
    DatasetSpecRequest,
    DatasetSpecResponse as AgentDatasetSpecResponse,
    ExperimentPlanRequest,
    ExperimentPlanResponse,
    AgentRunRead,
    AgentRunWithSteps,
    AgentRunWithStepsAndLogs,
    AgentRunList,
    AgentStepRead,
    AgentStepWithLogs,
    AgentStepLogRead,
    AgentStepLogList,
)
from app.schemas.dataset_spec import DatasetSpecResponse
from app.schemas.experiment import ExperimentResponse, ExperimentRunResponse
from app.services.agents.orchestration import (
    AVAILABLE_JUDGE_MODELS,
    DEFAULT_JUDGE_MODEL,
    get_available_judge_models,
)
from app.services.dataset_validator import DatasetValidator
from app.services.user_holdout_service import create_user_holdout_set

logger = logging.getLogger(__name__)


def autofix_experiment_plan_for_time_series(
    experiment_plan: Dict[str, Any],
    dataset_spec: Optional[DatasetSpec],
) -> tuple[Dict[str, Any], Optional[str]]:
    """Auto-fix experiment plan if time-based data is using a random split strategy.

    This prevents data leakage warnings by automatically correcting split strategy
    before the experiment is created.

    Args:
        experiment_plan: The experiment plan dictionary
        dataset_spec: The DatasetSpec for this experiment (may be None)

    Returns:
        Tuple of (fixed_plan, fix_message) where fix_message describes what was fixed,
        or None if no fix was needed.
    """
    if not dataset_spec or not dataset_spec.is_time_based:
        return experiment_plan, None

    # Check if using a random split strategy
    validation_strategy = experiment_plan.get("validation_strategy") or {}
    split_strategy = validation_strategy.get("split_strategy", "").lower()

    random_splits = ["random", "stratified", "group_random"]
    if split_strategy not in random_splits:
        return experiment_plan, None

    # Determine the correct time-based split
    if dataset_spec.entity_id_column:
        correct_split = "group_time"
    else:
        correct_split = "time"

    # Apply the fix
    fixed_plan = experiment_plan.copy()
    if "validation_strategy" not in fixed_plan or fixed_plan["validation_strategy"] is None:
        fixed_plan["validation_strategy"] = {}
    else:
        fixed_plan["validation_strategy"] = fixed_plan["validation_strategy"].copy()

    fixed_plan["validation_strategy"]["split_strategy"] = correct_split

    # If there's a time_column in dataset_spec, ensure it's used
    if dataset_spec.time_column:
        fixed_plan["validation_strategy"]["time_column"] = dataset_spec.time_column
    if dataset_spec.entity_id_column and correct_split == "group_time":
        fixed_plan["validation_strategy"]["group_column"] = dataset_spec.entity_id_column

    fix_message = (
        f"Auto-fixed split strategy: '{split_strategy}' -> '{correct_split}' "
        f"(time-based dataset requires time-ordered splits to prevent data leakage)"
    )
    logger.info(f"Auto-fix applied: {fix_message}")

    return fixed_plan, fix_message


# Router for project-scoped endpoints
router = APIRouter(prefix="/projects/{project_id}/agent", tags=["Agent"])

# Router for top-level agent run/step endpoints
agent_runs_router = APIRouter(tags=["Agent Runs"])

# Router for orchestration-related endpoints
orchestration_router = APIRouter(prefix="/orchestration", tags=["Orchestration"])


def get_llm_provider_and_key(db: Session, requested_provider: Optional[LLMProvider] = None):
    """Get the LLM provider and API key to use."""
    key_status = api_key_service.get_api_key_status(db)

    provider = requested_provider
    if provider:
        provider_key = "openai" if provider == LLMProvider.OPENAI else "gemini"
        if not key_status.get(provider_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No API key configured for {provider.value}",
            )
    else:
        if key_status.get("openai"):
            provider = LLMProvider.OPENAI
        elif key_status.get("gemini"):
            provider = LLMProvider.GEMINI
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No API keys configured. Please add an OpenAI or Gemini API key in Settings.",
            )

    api_key = api_key_service.get_api_key(db, provider)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"API key for {provider.value} not found",
        )

    return provider, api_key.api_key


def get_debate_clients(db: Session) -> tuple[Optional[GeminiClient], Optional[OpenAIClient]]:
    """Get clients for debate mode (Gemini for critique, OpenAI for judge).

    Returns:
        Tuple of (gemini_client, openai_client). Either may be None if not configured.
    """
    key_status = api_key_service.get_api_key_status(db)

    gemini_client = None
    openai_client = None

    # Get Gemini client for critique agent
    if key_status.get("gemini"):
        gemini_key = api_key_service.get_api_key(db, LLMProvider.GEMINI)
        if gemini_key:
            gemini_client = GeminiClient(gemini_key.api_key)
            logger.info("Debate mode: Gemini client created for critique agent")
    else:
        logger.warning("Debate mode: No Gemini API key configured - critique agent will use main LLM")

    # Get OpenAI client for judge
    if key_status.get("openai"):
        openai_key = api_key_service.get_api_key(db, LLMProvider.OPENAI)
        if openai_key:
            openai_client = OpenAIClient(openai_key.api_key, model="gpt-5.1")
            logger.info("Debate mode: OpenAI client created for judge (gpt-5.1)")
    else:
        logger.warning("Debate mode: No OpenAI API key configured - judge will use main LLM")

    return gemini_client, openai_client


def get_project_or_404(project_id: UUID, db: Session, current_user: Optional[User] = None) -> Project:
    """Get project by ID or raise 404."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    # Check ownership if user is authenticated
    if current_user and project.owner_id and project.owner_id != current_user.id:
        # Check if shared (simplified - in production would check shares table)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )
    return project


def get_data_source_or_404(data_source_id: UUID, db: Session) -> DataSource:
    """Get data source by ID or raise 404."""
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {data_source_id} not found",
        )
    return data_source


@router.get("/schema-summary/{data_source_id}", response_model=SchemaSummary)
def get_schema_summary(
    project_id: UUID,
    data_source_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get schema summary for a data source.

    Returns column names, types, and basic statistics without raw PII.
    This is used as context for LLM-powered configuration suggestions.
    """
    # Verify project exists and user has access
    project = get_project_or_404(project_id, db, current_user)

    # Get data source
    data_source = get_data_source_or_404(data_source_id, db)

    # Verify data source belongs to project
    if data_source.project_id != project.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source does not belong to this project",
        )

    # Check if schema_summary exists
    if not data_source.schema_summary:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source has no schema summary. Please re-upload or analyze the file.",
        )

    # Build and return schema summary
    return build_schema_summary(
        data_source_id=str(data_source.id),
        data_source_name=data_source.name,
        analysis_result=data_source.schema_summary,
    )


@router.post("/suggest-config", response_model=ProjectConfigResponse, deprecated=True)
async def suggest_project_config(
    project_id: UUID,
    request: ProjectConfigRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Suggest project configuration using LLM.

    **DEPRECATED**: Use POST /run-setup-pipeline instead for better observability.
    This endpoint will be removed in a future version.

    Based on user's description and data source schema, the LLM will suggest:
    - task_type (binary, multiclass, regression, etc.)
    - target_column (the column to predict)
    - primary_metric (metric to optimize)

    Returns the suggestion along with the schema summary used for context.
    """
    # Verify project and get data source
    project = get_project_or_404(project_id, db, current_user)
    data_source = get_data_source_or_404(request.data_source_id, db)

    if data_source.project_id != project.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source does not belong to this project",
        )

    if not data_source.schema_summary:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source has no schema summary",
        )

    # Get LLM client
    provider, api_key_str = get_llm_provider_and_key(db)
    client = get_llm_client(provider, api_key_str)

    # Build schema summary
    schema_summary = build_schema_summary(
        data_source_id=str(data_source.id),
        data_source_name=data_source.name,
        analysis_result=data_source.schema_summary,
    )

    try:
        # Generate suggestion
        suggestion = await generate_project_config(
            client=client,
            description=request.description,
            schema_summary=schema_summary,
        )

        return ProjectConfigResponse(
            suggestion=suggestion,
            schema_summary=schema_summary,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate configuration: {str(e)}",
        )
    except (LLMError, LLMTimeoutError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected error generating config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.post("/suggest-dataset-spec", response_model=AgentDatasetSpecResponse, deprecated=True)
async def suggest_dataset_spec(
    project_id: UUID,
    request: DatasetSpecRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Suggest dataset specification using LLM.

    **DEPRECATED**: Use POST /run-setup-pipeline instead for better observability.
    This endpoint will be removed in a future version.

    Based on task type and target column, the LLM will suggest:
    - feature_columns (columns to use as features)
    - excluded_columns (columns to exclude, with reasons)
    - suggested_filters (optional data filtering)

    Returns the suggestion along with the schema summary used for context.
    """
    # Verify project and get data source
    project = get_project_or_404(project_id, db, current_user)
    data_source = get_data_source_or_404(request.data_source_id, db)

    if data_source.project_id != project.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source does not belong to this project",
        )

    if not data_source.schema_summary:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source has no schema summary",
        )

    # Get LLM client
    provider, api_key_str = get_llm_provider_and_key(db)
    client = get_llm_client(provider, api_key_str)

    # Build schema summary
    schema_summary = build_schema_summary(
        data_source_id=str(data_source.id),
        data_source_name=data_source.name,
        analysis_result=data_source.schema_summary,
    )

    try:
        # Generate suggestion
        suggestion = await generate_dataset_spec(
            client=client,
            schema_summary=schema_summary,
            task_type=request.task_type,
            target_column=request.target_column,
            description=request.description,
        )

        return AgentDatasetSpecResponse(
            suggestion=suggestion,
            schema_summary=schema_summary,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate dataset spec: {str(e)}",
        )
    except (LLMError, LLMTimeoutError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected error generating dataset spec: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.post("/suggest-experiment-plan", response_model=ExperimentPlanResponse, deprecated=True)
async def suggest_experiment_plan(
    project_id: UUID,
    request: ExperimentPlanRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Suggest experiment plan with multiple variants using LLM.

    **DEPRECATED**: Use POST /run-setup-pipeline instead for better observability.
    This endpoint will be removed in a future version.

    Based on task configuration and dataset characteristics, the LLM will suggest:
    - Multiple experiment variants (quick, balanced, high_quality)
    - AutoML configuration for each variant
    - Recommended variant to start with

    Returns the experiment plan suggestion.
    """
    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Get LLM client
    provider, api_key_str = get_llm_provider_and_key(db)
    client = get_llm_client(provider, api_key_str)

    try:
        # Generate suggestion
        suggestion = await generate_experiment_plan(
            client=client,
            task_type=request.task_type,
            target_column=request.target_column,
            primary_metric=request.primary_metric,
            feature_columns=request.feature_columns,
            row_count=request.row_count,
            time_budget_minutes=request.time_budget_minutes,
            description=request.description,
        )

        return ExperimentPlanResponse(suggestion=suggestion)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate experiment plan: {str(e)}",
        )
    except (LLMError, LLMTimeoutError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected error generating experiment plan: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


# ============================================
# New Multi-Step Pipeline Endpoints
# ============================================

class SetupPipelineRequest(BaseModel):
    """Request to start a setup pipeline."""

    data_source_id: UUID = Field(
        ...,
        description="UUID of the data source to analyze"
    )
    description: str = Field(
        ...,
        min_length=10,
        description="User's description of what they want to predict/achieve"
    )
    time_budget_minutes: Optional[int] = Field(
        None,
        ge=1,
        le=1440,
        description="Optional time budget constraint in minutes"
    )
    run_async: bool = Field(
        False,
        description="If true, return immediately with run_id. If false, wait for completion."
    )
    # Orchestration options
    orchestration_mode: str = Field(
        "sequential",
        description="Pipeline orchestration mode: 'sequential' (default) or 'project_manager'"
    )
    debate_mode: str = Field(
        "disabled",
        description="Debate system: 'disabled' (default) or 'enabled'"
    )
    judge_model: Optional[str] = Field(
        None,
        description="OpenAI model to use as judge when debate doesn't reach consensus (e.g., 'gpt-4o', 'gpt-4-turbo')"
    )
    debate_partner: Optional[str] = Field(
        "gemini-2.0-flash",
        description="LLM model to use as debate partner for critique (e.g., 'gemini-2.0-flash', 'claude-sonnet-4', 'gpt-4o')"
    )
    max_debate_rounds: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum number of debate rounds before calling judge (default: 3, min: 1, max: 10)"
    )
    # Holdout validation options
    holdout_enabled: bool = Field(
        False,
        description="If true, hold out 5% of data before pipeline for user validation"
    )
    holdout_percentage: float = Field(
        5.0,
        ge=1.0,
        le=20.0,
        description="Percentage of data to hold out for validation (default: 5%, range: 1-20%)"
    )
    # Context documents options
    use_context_documents: bool = Field(
        True,
        description="If true (default), use uploaded context documents in AI prompts. If false, ignore all context."
    )
    context_ab_testing: bool = Field(
        False,
        description="If true, create experiments both with and without context documents for A/B comparison."
    )


class SetupPipelineResponse(BaseModel):
    """Response from starting a setup pipeline."""

    run_id: UUID
    status: str
    message: str


@router.post("/run-setup-pipeline", response_model=SetupPipelineResponse)
async def run_setup_pipeline(
    project_id: UUID,
    request: SetupPipelineRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Start the full AI-powered setup pipeline.

    This creates an agent run with 5 sequential steps:
    1. PROBLEM_UNDERSTANDING - Determine task type, target column, metric
    2. DATA_AUDIT - Analyze data quality and potential issues
    3. DATASET_DESIGN - Select features and exclusions
    4. EXPERIMENT_DESIGN - Create experiment variants
    5. PLAN_CRITIC - Review and validate the plan

    If run_async=False (default), waits for the pipeline to complete.
    If run_async=True, returns immediately with the run_id for polling.

    Returns:
        run_id: UUID of the created agent run
        status: Current status of the run
        message: Human-readable status message
    """
    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Verify data source exists
    data_source = get_data_source_or_404(request.data_source_id, db)
    if data_source.project_id != project.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source does not belong to this project",
        )

    if not data_source.schema_summary:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source has no schema summary. Please upload and analyze the data first.",
        )

    # Create user holdout set if enabled
    holdout_set = None
    if request.holdout_enabled:
        try:
            holdout_set, _ = create_user_holdout_set(
                db=db,
                project_id=project_id,
                data_source_id=request.data_source_id,
                holdout_percentage=request.holdout_percentage,
            )
            logger.info(
                f"Created user holdout set with {holdout_set.holdout_row_count} rows "
                f"({request.holdout_percentage}% of {holdout_set.total_rows_original})"
            )
        except Exception as e:
            logger.warning(f"Failed to create holdout set: {e}")
            # Don't fail the pipeline if holdout creation fails
            # Just log and continue

    # Get LLM client
    provider, api_key_str = get_llm_provider_and_key(db)
    llm_client = get_llm_client(provider, api_key_str)

    # Get debate clients (for debate mode)
    gemini_client, openai_client = get_debate_clients(db)

    # Log debate mode status
    if request.debate_mode == "enabled":
        logger.info(f"Setup pipeline has debate_mode={request.debate_mode}")
        logger.info(f"  - gemini_client available: {gemini_client is not None}")
        logger.info(f"  - openai_client available: {openai_client is not None}")

    try:
        if request.run_async:
            # Create the pipeline but don't run it yet
            agent_run = create_setup_pipeline(
                db=db,
                project_id=project_id,
                data_source_id=request.data_source_id,
                description=request.description,
                time_budget_minutes=request.time_budget_minutes,
                orchestration_mode=request.orchestration_mode,
                debate_mode=request.debate_mode,
                judge_model=request.judge_model,
                debate_partner=request.debate_partner,
                max_debate_rounds=request.max_debate_rounds,
                use_context_documents=request.use_context_documents,
                context_ab_testing=request.context_ab_testing,
            )

            # In a real implementation, you'd queue this for background processing
            # For now, we just return the pending run
            return SetupPipelineResponse(
                run_id=agent_run.id,
                status=agent_run.status.value,
                message="Pipeline created. Poll GET /runs/{run_id} for status.",
            )
        else:
            # Run the full pipeline synchronously
            agent_run = await run_setup_pipeline_for_project(
                db=db,
                project_id=project_id,
                data_source_id=request.data_source_id,
                description=request.description,
                time_budget_minutes=request.time_budget_minutes,
                orchestration_mode=request.orchestration_mode,
                debate_mode=request.debate_mode,
                judge_model=request.judge_model,
                debate_partner=request.debate_partner,
                max_debate_rounds=request.max_debate_rounds,
                llm_client=llm_client,
                gemini_client=gemini_client,
                openai_client=openai_client,
                use_context_documents=request.use_context_documents,
                context_ab_testing=request.context_ab_testing,
            )

            return SetupPipelineResponse(
                run_id=agent_run.id,
                status=agent_run.status.value,
                message=f"Pipeline completed with {len(agent_run.steps)} steps.",
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except (LLMError, LLMTimeoutError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except AgentPipelineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline step failed: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}",
        )


@router.get("/runs", response_model=AgentRunList)
def list_runs(
    project_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List agent runs for this project.

    Returns paginated list of agent runs, ordered by creation time (newest first).
    """
    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    runs, total = list_agent_runs_for_project(db, project_id, skip, limit)

    return AgentRunList(
        items=[AgentRunRead.model_validate(run) for run in runs],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/runs/{run_id}", response_model=AgentRunWithStepsAndLogs)
def get_run(
    project_id: UUID,
    run_id: UUID,
    include_logs: bool = Query(True, description="Include step logs in response"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a specific agent run with its steps and logs.

    Returns the full run details including all steps and their execution logs.
    Use this to monitor pipeline progress or review completed runs.
    """
    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    agent_run = get_agent_run_with_steps(db, run_id, include_logs=include_logs)
    if not agent_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run {run_id} not found",
        )

    # Verify run belongs to this project
    if agent_run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent run does not belong to this project",
        )

    return AgentRunWithStepsAndLogs.model_validate(agent_run)


@router.post("/runs/{run_id}/resume")
async def resume_run(
    project_id: UUID,
    run_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Resume a pending or failed agent run.

    This will continue executing any pending steps in the run.
    Useful for resuming after an async creation or retrying after a failure.
    """
    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    agent_run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if not agent_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run {run_id} not found",
        )

    if agent_run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent run does not belong to this project",
        )

    # Get LLM client
    provider, api_key_str = get_llm_provider_and_key(db)
    llm_client = get_llm_client(provider, api_key_str)

    # Get debate clients (for debate mode)
    gemini_client, openai_client = get_debate_clients(db)

    # Log debate mode status
    if agent_run.debate_mode:
        logger.info(f"Pipeline {run_id} has debate_mode={agent_run.debate_mode}")
        logger.info(f"  - gemini_client available: {gemini_client is not None}")
        logger.info(f"  - openai_client available: {openai_client is not None}")

    try:
        completed_run = await run_agent_pipeline(
            db,
            run_id,
            llm_client,
            gemini_client=gemini_client,
            openai_client=openai_client,
        )

        return SetupPipelineResponse(
            run_id=completed_run.id,
            status=completed_run.status.value,
            message=f"Pipeline resumed and completed.",
        )
    except (LLMError, LLMTimeoutError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except AgentPipelineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline step failed: {str(e)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected error resuming pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}",
        )


# ============================================
# Top-Level Agent Run/Step Endpoints
# ============================================

@agent_runs_router.get("/agent-runs/{run_id}", response_model=AgentRunWithSteps)
def get_agent_run(
    run_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get an agent run with its steps.

    Returns:
        run metadata,
        a list of associated steps with: id, step_type, status, timestamps.
    """
    agent_run = get_agent_run_with_steps(db, run_id, include_logs=False)
    if not agent_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run {run_id} not found",
        )

    return AgentRunWithSteps.model_validate(agent_run)


@agent_runs_router.delete("/agent-runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent_run(
    run_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Delete an agent run and all associated steps, logs, and created data sources.

    This will:
    - Delete all agent step logs
    - Delete all agent steps
    - Delete any training datasets created by this run
    - Delete the agent run itself
    """
    agent_run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if not agent_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run {run_id} not found",
        )

    # Don't allow deleting running pipelines
    if agent_run.status == AgentRunStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a running agent run. Wait for it to complete or fail.",
        )

    # If this run created a training dataset, delete it too
    if agent_run.result_json:
        training_data_source_id = agent_run.result_json.get("training_data_source_id")
        if training_data_source_id:
            try:
                data_source = db.query(DataSource).filter(
                    DataSource.id == training_data_source_id
                ).first()
                if data_source:
                    # Delete the file if it exists
                    if data_source.config_json and data_source.config_json.get("file_path"):
                        import os
                        file_path = data_source.config_json["file_path"]
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    db.delete(data_source)
                    logger.info(f"Deleted training data source {training_data_source_id}")
            except Exception as e:
                logger.warning(f"Failed to delete training data source: {e}")

    # Delete all steps and their logs (cascade should handle logs, but be explicit)
    for step in agent_run.steps:
        db.query(AgentStepLog).filter(AgentStepLog.agent_step_id == step.id).delete()
        db.delete(step)

    # Delete the run
    db.delete(agent_run)
    db.commit()

    logger.info(f"Deleted agent run {run_id}")
    return None


class CancelResponse(BaseModel):
    """Response from cancel endpoint."""
    run_id: UUID
    status: str
    message: str


@agent_runs_router.post("/agent-runs/{run_id}/cancel", response_model=CancelResponse)
def cancel_agent_run(
    run_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Cancel a running agent pipeline.

    This marks the run as CANCELLED. The pipeline execution loop will detect
    this status change and stop processing before the next step.

    Only running pipelines can be cancelled.
    """
    agent_run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if not agent_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run {run_id} not found",
        )

    # Only allow cancelling running pipelines
    if agent_run.status != AgentRunStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel agent run with status '{agent_run.status.value}'. Only running pipelines can be cancelled.",
        )

    # Mark as cancelled
    agent_run.status = AgentRunStatus.CANCELLED
    agent_run.error_message = "Cancelled by user"
    db.commit()

    logger.info(f"Cancelled agent run {run_id}")
    return CancelResponse(
        run_id=run_id,
        status="cancelled",
        message="Pipeline cancellation requested. It will stop before the next step."
    )


@agent_runs_router.get("/agent-steps/{step_id}", response_model=AgentStepRead)
def get_agent_step(
    step_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a specific agent step with its metadata and I/O.

    Returns:
        step metadata,
        input_json,
        output_json.
    """
    step = db.query(AgentStep).filter(AgentStep.id == step_id).first()
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent step {step_id} not found",
        )

    return AgentStepRead.model_validate(step)


@agent_runs_router.get("/agent-steps/{step_id}/logs", response_model=AgentStepLogList)
def get_step_logs(
    step_id: UUID,
    since_sequence: int = Query(0, ge=0, description="Return logs with sequence > this value"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get logs for an agent step, supporting streaming/polling.

    Returns logs with sequence > since_sequence in order.
    Use last_sequence from the response as since_sequence in subsequent polls.

    Returns:
        logs: List of log entries
        last_sequence: Highest sequence in this batch (use for next poll)
        has_more: Whether the step is still running (more logs may come)
    """
    # Verify step exists
    step = db.query(AgentStep).filter(AgentStep.id == step_id).first()
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent step {step_id} not found",
        )

    # Query logs with sequence > since_sequence
    logs = (
        db.query(AgentStepLog)
        .filter(AgentStepLog.agent_step_id == step_id)
        .filter(AgentStepLog.sequence > since_sequence)
        .order_by(AgentStepLog.sequence)
        .limit(limit)
        .all()
    )

    # Determine last_sequence
    if logs:
        last_sequence = max(log.sequence for log in logs)
    else:
        last_sequence = since_sequence

    # Check if step is still running (more logs may come)
    has_more = step.status in ("pending", "running")

    return AgentStepLogList(
        logs=[AgentStepLogRead.model_validate(log) for log in logs],
        last_sequence=last_sequence,
        has_more=has_more,
    )


# ============================================
# Apply Agent Step Outputs to Create Resources
# ============================================

class ApplyDatasetSpecRequest(BaseModel):
    """Request body for applying dataset spec with optional user modifications."""
    target_column: Optional[str] = Field(
        None,
        description="Override target column from agent suggestion"
    )
    feature_columns: Optional[List[str]] = Field(
        None,
        description="Override feature columns from agent suggestion"
    )
    name: Optional[str] = Field(
        None,
        max_length=255,
        description="Custom name for the DatasetSpec"
    )


class ApplyDatasetSpecResponse(BaseModel):
    """Response from applying dataset spec from an agent step."""
    dataset_spec_id: UUID
    message: str


class ApplyExperimentPlanResponse(BaseModel):
    """Response from applying experiment plan from an agent step."""
    experiment_id: UUID
    message: str


@router.post("/apply-dataset-spec-from-step/{step_id}", response_model=ApplyDatasetSpecResponse)
def apply_dataset_spec_from_step(
    project_id: UUID,
    step_id: UUID,
    request: Optional[ApplyDatasetSpecRequest] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a DatasetSpec from a dataset_design agent step's output.

    This endpoint reads the feature selection and exclusion recommendations
    from a completed dataset_design step and creates a new DatasetSpec row.

    Users can optionally override the AI suggestions by passing modified values
    in the request body.

    Args:
        project_id: The project UUID
        step_id: The agent step UUID (must be a completed dataset_design step)
        request: Optional request body with user modifications

    Returns:
        dataset_spec_id: UUID of the created DatasetSpec
        message: Success message
    """
    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Get the agent step
    step = db.query(AgentStep).filter(AgentStep.id == step_id).first()
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent step {step_id} not found",
        )

    # Verify step belongs to this project (through its run)
    agent_run = db.query(AgentRun).filter(AgentRun.id == step.agent_run_id).first()
    if not agent_run or agent_run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent step does not belong to this project",
        )

    # Verify step type
    if step.step_type.value != "dataset_design":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step type must be 'dataset_design', got '{step.step_type.value}'",
        )

    # Verify step is completed
    if step.status.value != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step must be completed, current status is '{step.status.value}'",
        )

    # Extract output data
    output = step.output_json
    if not output:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Step has no output data",
        )

    # Get required fields from output (can be overridden by user)
    feature_columns = output.get("feature_columns", [])
    target_column = None

    # Get target column from input (passed from previous step)
    input_data = step.input_json or {}
    target_column = input_data.get("target_column")

    # Also check output for target_column (might be included there)
    if not target_column:
        target_column = output.get("target_column")

    # Apply user overrides if provided
    if request:
        if request.target_column:
            target_column = request.target_column
        if request.feature_columns:
            feature_columns = request.feature_columns

    if not target_column:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not determine target column from step data",
        )

    if not feature_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No feature columns found in step output",
        )

    # Get data source from the run's config
    run_config = agent_run.config_json or {}
    data_source_id = run_config.get("data_source_id")
    if not data_source_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not determine data source from agent run",
        )

    # Build optional fields
    excluded_columns = output.get("excluded_columns", [])
    exclusion_reasons = output.get("exclusion_reasons", {})
    natural_language_summary = output.get("natural_language_summary", "")

    # Get time-based metadata from problem_understanding step
    is_time_based = False
    time_column = None
    entity_id_column = None
    prediction_horizon = None
    target_positive_class = None
    target_creation = None
    target_exists = True  # Default: target exists in raw data

    problem_step = (
        db.query(AgentStep)
        .filter(
            AgentStep.agent_run_id == agent_run.id,
            AgentStep.step_type == "problem_understanding"
        )
        .first()
    )
    if problem_step and problem_step.output_json:
        problem_output = problem_step.output_json
        is_time_based = problem_output.get("is_time_based", False)
        time_column = problem_output.get("time_column")
        entity_id_column = problem_output.get("entity_id_column")
        prediction_horizon = problem_output.get("prediction_horizon")
        target_positive_class = problem_output.get("target_positive_class")
        target_creation = problem_output.get("target_creation")
        target_exists = problem_output.get("target_exists", True)

    # Create the DatasetSpec
    spec_name = "Agent-Generated Dataset Spec"
    if request and request.name:
        spec_name = request.name

    dataset_spec = DatasetSpec(
        project_id=project_id,
        name=spec_name,
        description=natural_language_summary[:500] if natural_language_summary else "Created from AI pipeline recommendation",
        data_sources_json=[data_source_id],
        target_column=target_column,
        feature_columns=feature_columns,
        filters_json=None,  # Could add suggested_filters here if present
        spec_json={
            "source": "agent_pipeline",
            "agent_step_id": str(step_id),
            "excluded_columns": excluded_columns,
            "exclusion_reasons": exclusion_reasons,
            "target_creation": target_creation,
            "target_exists": target_exists,
        },
        # Time-based task metadata
        is_time_based=is_time_based,
        time_column=time_column,
        entity_id_column=entity_id_column,
        prediction_horizon=prediction_horizon,
        target_positive_class=target_positive_class,
    )

    db.add(dataset_spec)
    db.commit()
    db.refresh(dataset_spec)

    return ApplyDatasetSpecResponse(
        dataset_spec_id=dataset_spec.id,
        message=f"Created DatasetSpec with {len(feature_columns)} features, targeting '{target_column}'",
    )


class BatchApplyDatasetSpecRequest(BaseModel):
    """Request to create multiple DatasetSpecs from selected variants."""
    variant_names: List[str] = Field(
        ...,
        description="List of variant names to create DatasetSpecs for"
    )


class BatchApplyDatasetSpecResponse(BaseModel):
    """Response from batch creating DatasetSpecs."""
    dataset_specs: List[dict] = Field(
        ...,
        description="List of created DatasetSpecs with their IDs and variant names"
    )
    message: str
    created_count: int = 0
    skipped_variants: List[str] = Field(default_factory=list, description="Variants skipped due to missing features")


@router.post("/apply-dataset-specs-batch/{step_id}", response_model=BatchApplyDatasetSpecResponse)
def apply_dataset_specs_batch(
    project_id: UUID,
    step_id: UUID,
    request: BatchApplyDatasetSpecRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create multiple DatasetSpecs from selected dataset variants.

    This endpoint reads the dataset design variants from a completed dataset_design
    step and creates DatasetSpec rows for each selected variant.

    Args:
        project_id: The project UUID
        step_id: The agent step UUID (must be a completed dataset_design step)
        request: List of variant names to create DatasetSpecs for

    Returns:
        dataset_specs: List of created DatasetSpecs with IDs
        message: Success message
    """
    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Get the agent step
    step = db.query(AgentStep).filter(AgentStep.id == step_id).first()
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent step {step_id} not found",
        )

    # Verify step belongs to this project
    agent_run = db.query(AgentRun).filter(AgentRun.id == step.agent_run_id).first()
    if not agent_run or agent_run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent step does not belong to this project",
        )

    # Verify step type
    if step.step_type.value != "dataset_design":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step type must be 'dataset_design', got '{step.step_type.value}'",
        )

    # Verify step is completed
    if step.status.value != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step must be completed, current status is '{step.status.value}'",
        )

    # Extract output data
    output = step.output_json
    if not output:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Step has no output data",
        )

    # Get variants from output
    variants = output.get("variants", [])
    if not variants:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No dataset variants found in step output",
        )

    # Get target column from input
    input_data = step.input_json or {}
    target_column = input_data.get("target_column") or output.get("target_column")
    if not target_column:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not determine target column from step data",
        )

    # Get data source from the run's config
    run_config = agent_run.config_json or {}
    data_source_id = run_config.get("data_source_id")
    if not data_source_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not determine data source from agent run",
        )

    # Build variant lookup
    variant_lookup = {v.get("name"): v for v in variants}
    available_names = list(variant_lookup.keys())

    # Validate requested variant_names is not empty
    if not request.variant_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"variant_names cannot be empty. Available variants: {available_names}",
        )

    # Validate requested variants exist
    invalid_names = [n for n in request.variant_names if n not in variant_lookup]
    if invalid_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid variant names: {invalid_names}. Available: {available_names}",
        )

    # Create DatasetSpecs for each selected variant
    created_specs = []
    skipped_variants = []
    for variant_name in request.variant_names:
        variant = variant_lookup[variant_name]

        feature_columns = variant.get("feature_columns", [])
        if not feature_columns:
            skipped_variants.append(variant_name)
            logger.warning(f"Skipping variant '{variant_name}' - no feature columns defined")
            continue

        excluded_columns = variant.get("excluded_columns", [])
        exclusion_reasons = variant.get("exclusion_reasons", {})
        train_test_split = variant.get("train_test_split", "80_20")
        preprocessing_strategy = variant.get("preprocessing_strategy", "auto")
        engineered_features = variant.get("engineered_features", [])

        # Get target_creation and time-based metadata from problem_understanding step
        target_creation = None
        target_exists = True  # Default: target exists in raw data
        is_time_based = False
        time_column = None
        entity_id_column = None
        prediction_horizon = None
        target_positive_class = None

        problem_step = (
            db.query(AgentStep)
            .filter(
                AgentStep.agent_run_id == agent_run.id,
                AgentStep.step_type == "problem_understanding"
            )
            .first()
        )
        if problem_step and problem_step.output_json:
            problem_output = problem_step.output_json
            target_creation = problem_output.get("target_creation")
            target_exists = problem_output.get("target_exists", True)
            is_time_based = problem_output.get("is_time_based", False)
            time_column = problem_output.get("time_column")
            entity_id_column = problem_output.get("entity_id_column")
            prediction_horizon = problem_output.get("prediction_horizon")
            target_positive_class = problem_output.get("target_positive_class")

        dataset_spec = DatasetSpec(
            project_id=project_id,
            name=f"Dataset - {variant_name}",
            description=variant.get("description", f"Created from {variant_name} variant")[:500],
            data_sources_json=[data_source_id],
            target_column=target_column,
            feature_columns=feature_columns,
            filters_json=variant.get("suggested_filters"),
            spec_json={
                "source": "agent_pipeline",
                "agent_step_id": str(step_id),
                "variant_name": variant_name,
                "excluded_columns": excluded_columns,
                "exclusion_reasons": exclusion_reasons,
                "train_test_split": train_test_split,
                "preprocessing_strategy": preprocessing_strategy,
                "expected_tradeoff": variant.get("expected_tradeoff", ""),
                "engineered_features": engineered_features,
                "target_creation": target_creation,
                "target_exists": target_exists,
            },
            # Time-based task metadata
            is_time_based=is_time_based,
            time_column=time_column,
            entity_id_column=entity_id_column,
            prediction_horizon=prediction_horizon,
            target_positive_class=target_positive_class,
        )

        db.add(dataset_spec)
        db.flush()  # Get the ID without committing

        created_specs.append({
            "dataset_spec_id": str(dataset_spec.id),
            "variant_name": variant_name,
            "feature_count": len(feature_columns),
            "train_test_split": train_test_split,
        })

    db.commit()

    # Build message with warnings if any variants were skipped
    message = f"Created {len(created_specs)} DatasetSpecs from selected variants"
    if skipped_variants:
        message += f" (skipped {len(skipped_variants)} variants with no features: {skipped_variants})"

    return BatchApplyDatasetSpecResponse(
        dataset_specs=created_specs,
        message=message,
        created_count=len(created_specs),
        skipped_variants=skipped_variants,
    )


@router.post("/apply-experiment-plan-from-step/{step_id}", response_model=ApplyExperimentPlanResponse)
def apply_experiment_plan_from_step(
    project_id: UUID,
    step_id: UUID,
    dataset_spec_id: Optional[UUID] = None,
    variant: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create an Experiment from an experiment_design agent step's output.

    This endpoint reads the experiment plan recommendations from a completed
    experiment_design step and creates a new Experiment row.

    Args:
        project_id: The project UUID
        step_id: The agent step UUID (must be a completed experiment_design step)
        dataset_spec_id: Optional DatasetSpec UUID to use. If not provided,
                        will attempt to use the most recently created spec for this project.
        variant: Optional variant name to use (e.g., 'quick', 'balanced', 'high_quality').
                If not provided, uses the recommended_variant from the step output.

    Returns:
        experiment_id: UUID of the created Experiment
        message: Success message
    """
    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Get the agent step
    step = db.query(AgentStep).filter(AgentStep.id == step_id).first()
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent step {step_id} not found",
        )

    # Verify step belongs to this project (through its run)
    agent_run = db.query(AgentRun).filter(AgentRun.id == step.agent_run_id).first()
    if not agent_run or agent_run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent step does not belong to this project",
        )

    # Verify step type
    if step.step_type.value != "experiment_design":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step type must be 'experiment_design', got '{step.step_type.value}'",
        )

    # Verify step is completed
    if step.status.value != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step must be completed, current status is '{step.status.value}'",
        )

    # Extract output data
    output = step.output_json
    if not output:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Step has no output data",
        )

    # Get variants from output
    variants = output.get("variants", [])
    recommended_variant = output.get("recommended_variant")

    if not variants:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No experiment variants found in step output",
        )

    # Select which variant to use
    variant_name = variant or recommended_variant
    if not variant_name:
        # Default to first variant
        variant_name = variants[0].get("name") if variants else None

    # Find the selected variant
    selected_variant = None
    for v in variants:
        if v.get("name") == variant_name:
            selected_variant = v
            break

    if not selected_variant:
        available = [v.get("name") for v in variants]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Variant '{variant_name}' not found. Available: {available}",
        )

    # Get or find dataset_spec_id
    if not dataset_spec_id:
        # Try to find the most recent dataset spec for this project
        latest_spec = (
            db.query(DatasetSpec)
            .filter(DatasetSpec.project_id == project_id)
            .order_by(DatasetSpec.created_at.desc())
            .first()
        )
        if not latest_spec:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No DatasetSpec found for this project. Create one first using apply-dataset-spec-from-step.",
            )
        dataset_spec_id = latest_spec.id
        dataset_spec = latest_spec
    else:
        # Verify the dataset spec exists and belongs to project
        dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
        if not dataset_spec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DatasetSpec {dataset_spec_id} not found",
            )
        if dataset_spec.project_id != project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="DatasetSpec does not belong to this project",
            )

    # Get primary metric from input (passed from previous steps)
    input_data = step.input_json or {}
    primary_metric = input_data.get("primary_metric", "accuracy")

    # Build experiment plan from selected variant
    automl_config = selected_variant.get("automl_config", {})
    natural_language_summary = output.get("natural_language_summary", "")

    # Extract validation_strategy from selected variant if present
    validation_strategy = selected_variant.get("validation_strategy")

    experiment_plan = {
        "source": "agent_pipeline",
        "agent_step_id": str(step_id),
        "variant_name": variant_name,
        "variant_description": selected_variant.get("description", ""),
        "automl_config": automl_config,
        "validation_strategy": validation_strategy,  # Include agent-specified split strategy
        "all_variants": variants,  # Store all variants for reference
    }

    # Auto-fix split strategy for time-based datasets
    experiment_plan, fix_message = autofix_experiment_plan_for_time_series(
        experiment_plan, dataset_spec
    )

    # Create the Experiment
    experiment = Experiment(
        project_id=project_id,
        dataset_spec_id=dataset_spec_id,
        name=f"Agent Experiment - {variant_name}",
        description=natural_language_summary[:500] if natural_language_summary else f"Created from AI pipeline ({variant_name} variant)",
        primary_metric=primary_metric,
        metric_direction="maximize",  # Default, could be inferred from metric type
        experiment_plan_json=experiment_plan,
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    return ApplyExperimentPlanResponse(
        experiment_id=experiment.id,
        message=f"Created Experiment using '{variant_name}' variant with metric '{primary_metric}'",
    )


class BatchApplyExperimentPlanRequest(BaseModel):
    """Request to create experiments for all dataset specs in the project."""
    variant: Optional[str] = Field(
        None,
        description="Experiment variant to use (e.g., 'quick', 'balanced', 'high_quality'). Uses recommended if not specified."
    )
    create_all_variants: bool = Field(
        False,
        description="If true, create one experiment PER VARIANT for each dataset spec (ignores 'variant' field). "
                    "This follows the agent's full recommendation to try multiple experiment configurations."
    )
    run_immediately: bool = Field(
        True,
        description="If true, queue all created experiments for immediate execution"
    )
    use_modal: Optional[bool] = Field(
        None,
        description="If true, use Modal cloud for training. If None (default), use Modal if configured, else local."
    )


class BatchApplyExperimentPlanResponse(BaseModel):
    """Response from batch creating experiments for all datasets."""
    experiments: List[dict] = Field(
        ...,
        description="List of created experiments with their IDs and dataset info"
    )
    message: str
    created_count: int
    queued_count: int = 0
    failed_queue_count: int = 0
    queue_errors: List[str] = Field(default_factory=list, description="List of queue error messages")


@router.post("/apply-experiments-batch/{step_id}", response_model=BatchApplyExperimentPlanResponse)
def apply_experiments_batch(
    project_id: UUID,
    step_id: UUID,
    request: Optional[BatchApplyExperimentPlanRequest] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create experiments for ALL dataset specs in the project from an experiment_design step.

    This endpoint:
    1. Reads the experiment plan from a completed experiment_design step
    2. Gets all dataset specs for the project
    3. Creates an experiment for each dataset spec using the specified variant
    4. Optionally queues all experiments for immediate execution

    Args:
        project_id: The project UUID
        step_id: The agent step UUID (must be a completed experiment_design step)
        request: Optional configuration for variant and immediate execution

    Returns:
        List of created experiments with their IDs and status
    """
    from app.tasks.automl import run_automl_experiment_task

    if request is None:
        request = BatchApplyExperimentPlanRequest()

    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Get the agent step
    step = db.query(AgentStep).filter(AgentStep.id == step_id).first()
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent step {step_id} not found",
        )

    # Verify step belongs to this project (through its run)
    agent_run = db.query(AgentRun).filter(AgentRun.id == step.agent_run_id).first()
    if not agent_run or agent_run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent step does not belong to this project",
        )

    # Verify step type
    if step.step_type.value != "experiment_design":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step type must be 'experiment_design', got '{step.step_type.value}'",
        )

    # Verify step is completed
    if step.status.value != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step must be completed, current status is '{step.status.value}'",
        )

    # Get problem_understanding step to extract task_type and set it on project if not already set
    if not project.task_type:
        problem_step = (
            db.query(AgentStep)
            .filter(
                AgentStep.agent_run_id == agent_run.id,
                AgentStep.step_type == "problem_understanding"
            )
            .first()
        )
        if problem_step and problem_step.output_json:
            pipeline_task_type = problem_step.output_json.get("task_type")
            if pipeline_task_type:
                # Map pipeline task_type string to TaskType enum
                task_type_map = {
                    "binary": TaskType.BINARY,
                    "multiclass": TaskType.MULTICLASS,
                    "regression": TaskType.REGRESSION,
                    "quantile": TaskType.QUANTILE,
                    "timeseries_forecast": TaskType.TIMESERIES_FORECAST,
                    "classification": TaskType.BINARY,  # Legacy alias
                }
                mapped_type = task_type_map.get(pipeline_task_type.lower())
                if mapped_type:
                    project.task_type = mapped_type
                    db.flush()

    # Extract output data
    output = step.output_json
    if not output:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Step has no output data",
        )

    # Get variants from output
    variants = output.get("variants", [])
    recommended_variant = output.get("recommended_variant")

    if not variants:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No experiment variants found in step output",
        )

    # Get ALL dataset specs for this project
    dataset_specs = (
        db.query(DatasetSpec)
        .filter(DatasetSpec.project_id == project_id)
        .order_by(DatasetSpec.created_at.desc())
        .all()
    )

    if not dataset_specs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No DatasetSpecs found for this project. Create dataset specs first using the Dataset Design step.",
        )

    # Get primary metric from input (passed from previous steps)
    input_data = step.input_json or {}
    primary_metric = input_data.get("primary_metric", "accuracy")
    natural_language_summary = output.get("natural_language_summary", "")

    # Determine which variants to create experiments for
    if request.create_all_variants:
        # Create experiments for ALL variants (following agent's full recommendation)
        variants_to_use = variants
    else:
        # Select which variant to use
        variant_name = request.variant or recommended_variant
        if not variant_name:
            variant_name = variants[0].get("name") if variants else None

        # Find the selected variant
        selected_variant = None
        for v in variants:
            if v.get("name") == variant_name:
                selected_variant = v
                break

        if not selected_variant:
            available = [v.get("name") for v in variants]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Variant '{variant_name}' not found. Available: {available}",
            )
        variants_to_use = [selected_variant]

    # Store the experiment design config in each dataset spec for later use
    # This ensures the "Run Experiments" button works even after new agent runs
    # IMPORTANT: Only store if no config exists yet - never overwrite from iteration runs
    # This preserves the INITIAL experiment design config for each dataset
    experiment_design_config = {
        "step_id": str(step_id),
        "agent_run_id": str(agent_run.id),
        "variants": variants,
        "recommended_variant": recommended_variant,
        "primary_metric": primary_metric,
        "natural_language_summary": natural_language_summary,
        "stored_at": datetime.utcnow().isoformat(),
        "source_type": "initial",  # "initial" = first run, "iteration" = auto-improve
        "parent_experiment_id": None,  # No parent for initial runs
    }
    for spec in dataset_specs:
        # Only store if no config exists - protect existing configs from being overwritten
        if not spec.agent_experiment_design_json:
            spec.agent_experiment_design_json = experiment_design_config
    db.flush()

    # Create experiments for each dataset spec AND each variant
    created_experiments = []
    for spec in dataset_specs:
        for variant in variants_to_use:
            variant_name = variant.get("name", "unnamed")
            automl_config = variant.get("automl_config", {})
            validation_strategy = variant.get("validation_strategy")
            variant_description = variant.get("description", "")
            is_recommended = (variant_name == recommended_variant)

            # Build comprehensive experiment plan with agent's full recommendation
            experiment_plan = {
                "source": "agent_pipeline",
                "agent_step_id": str(step_id),
                "variant_name": variant_name,
                "variant_description": variant_description,
                "is_recommended_variant": is_recommended,
                "automl_config": automl_config,
                "validation_strategy": validation_strategy,
                "dataset_spec_name": spec.name,
                "all_variants_summary": [
                    {"name": v.get("name"), "description": v.get("description", "")[:100]}
                    for v in variants
                ],
                "agent_reasoning": natural_language_summary,
            }

            # Build rich description including agent's explanation
            description_parts = []
            if is_recommended:
                description_parts.append(f"🌟 RECOMMENDED by AI Agent")
            description_parts.append(f"Variant: {variant_name}")
            if variant_description:
                description_parts.append(f"Strategy: {variant_description}")
            if automl_config:
                time_limit = automl_config.get("time_limit", "auto")
                presets = automl_config.get("presets", "default")
                description_parts.append(f"Config: {presets} preset, {time_limit}s time limit")
            if validation_strategy:
                if isinstance(validation_strategy, dict):
                    split_strategy = validation_strategy.get("split_strategy", "unknown")
                    description_parts.append(f"Validation: {split_strategy} split")
                else:
                    description_parts.append(f"Validation: {validation_strategy}")
            if natural_language_summary:
                description_parts.append(f"\nAgent Analysis: {natural_language_summary[:400]}")

            full_description = "\n".join(description_parts)

            # Auto-fix split strategy for time-based datasets
            experiment_plan, _ = autofix_experiment_plan_for_time_series(
                experiment_plan, spec
            )

            experiment = Experiment(
                project_id=project_id,
                dataset_spec_id=spec.id,
                name=f"{spec.name} - {variant_name}" + (" ⭐" if is_recommended else ""),
                description=full_description[:2000],  # Ensure within DB limit
                primary_metric=primary_metric,
                metric_direction="maximize",
                experiment_plan_json=experiment_plan,
            )

            db.add(experiment)
            db.flush()  # Get the ID without committing

            created_experiments.append({
                "experiment_id": str(experiment.id),
                "dataset_spec_id": str(spec.id),
                "dataset_spec_name": spec.name,
                "variant_name": variant_name,
                "is_recommended": is_recommended,
                "name": experiment.name,
                "status": "pending",
            })

    db.commit()

    # Optionally queue all experiments for immediate execution
    queued_count = 0
    failed_queue_count = 0
    queue_errors = []
    backend_used = "modal"
    if request.run_immediately:
        from app.tasks import run_experiment_modal

        for exp_info in created_experiments:
            try:
                experiment = db.query(Experiment).filter(
                    Experiment.id == UUID(exp_info["experiment_id"])
                ).first()
                if experiment:
                    task = run_experiment_modal.delay(exp_info["experiment_id"])
                    experiment.celery_task_id = task.id
                    exp_info["status"] = "queued"
                    exp_info["task_id"] = task.id
                    exp_info["backend"] = backend_used
                    queued_count += 1
            except Exception as e:
                exp_info["status"] = "queue_failed"
                exp_info["error"] = str(e)
                failed_queue_count += 1
                queue_errors.append(f"Experiment {exp_info.get('experiment_id', 'unknown')}: {str(e)}")
                logger.error(f"Failed to queue experiment {exp_info.get('experiment_id')}: {e}")

        db.commit()

    # Build descriptive message
    variant_names_used = list(set(e.get("variant_name", "unknown") for e in created_experiments))
    if request.create_all_variants:
        message = f"Created {len(created_experiments)} experiments ({len(dataset_specs)} datasets × {len(variants_to_use)} variants per agent's recommendation)"
    else:
        message = f"Created {len(created_experiments)} experiments using '{variant_names_used[0]}' variant"

    if recommended_variant:
        recommended_count = sum(1 for e in created_experiments if e.get("is_recommended"))
        if recommended_count > 0:
            message += f" ({recommended_count} recommended by agent)"

    if queued_count > 0:
        message += f" - {queued_count} queued on {backend_used}"

    if failed_queue_count > 0:
        message += f" - WARNING: {failed_queue_count} failed to queue"

    return BatchApplyExperimentPlanResponse(
        experiments=created_experiments,
        message=message,
        created_count=len(created_experiments),
        queued_count=queued_count,
        failed_queue_count=failed_queue_count,
        queue_errors=queue_errors,
    )


# ============================================
# Create Experiments for a Single Dataset Spec
# ============================================

class CreateExperimentsForDatasetRequest(BaseModel):
    """Request to create experiments for a specific dataset spec."""
    dataset_spec_id: UUID = Field(..., description="The dataset spec to create experiments for")
    create_all_variants: bool = Field(
        True,
        description="If true, create one experiment per variant. If false, create only the recommended variant."
    )
    run_immediately: bool = Field(True, description="If true, queue experiments for immediate execution")
    use_modal: Optional[bool] = Field(None, description="If true, use Modal cloud. If None, auto-detect.")


class CreateExperimentsForDatasetResponse(BaseModel):
    """Response from creating experiments for a dataset spec."""
    experiments: List[dict]
    message: str
    created_count: int
    queued_count: int


@router.post("/create-experiments-for-dataset/{step_id}", response_model=CreateExperimentsForDatasetResponse)
def create_experiments_for_dataset(
    project_id: UUID,
    step_id: UUID,
    request: CreateExperimentsForDatasetRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create experiments for a SINGLE dataset spec from an experiment_design step.

    This endpoint allows creating experiments for a specific dataset spec, useful when
    the user wants to run experiments for just one dataset from the dataset detail modal.

    Creates experiments following the agent's recommendations (all variants or recommended only).
    """
    from app.tasks.automl import run_automl_experiment_task

    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Get the agent step
    step = db.query(AgentStep).filter(AgentStep.id == step_id).first()
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent step {step_id} not found",
        )

    # Verify step is experiment_design and completed
    if step.step_type.value != "experiment_design":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step type must be 'experiment_design', got '{step.step_type.value}'",
        )

    if step.status.value != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Step must be completed, current status is '{step.status.value}'",
        )

    # Verify dataset spec exists and belongs to project
    spec = db.query(DatasetSpec).filter(
        DatasetSpec.id == request.dataset_spec_id,
        DatasetSpec.project_id == project_id,
    ).first()
    if not spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {request.dataset_spec_id} not found in this project",
        )

    # Pre-flight validation: verify columns exist in data source
    validation_result = DatasetValidator(db).validate_dataset_spec(spec.id)
    if not validation_result.is_valid:
        logger.warning(
            f"Dataset spec {spec.id} validation failed: "
            f"missing_target={validation_result.missing_target}, "
            f"missing_features={len(validation_result.missing_features)}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Dataset validation failed - some columns don't exist in the data source",
                "missing_target": validation_result.missing_target,
                "missing_features": validation_result.missing_features[:20],
                "available_columns": sorted(list(validation_result.available_columns))[:50],
                "feedback": validation_result.to_feedback(),
            }
        )

    # Extract output data
    output = step.output_json
    if not output:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Step has no output data",
        )

    variants = output.get("variants", [])
    recommended_variant = output.get("recommended_variant")
    natural_language_summary = output.get("natural_language_summary", "")

    if not variants:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No experiment variants found in step output",
        )

    # Get primary metric
    input_data = step.input_json or {}
    primary_metric = input_data.get("primary_metric", "accuracy")

    # Store the experiment design config in the dataset spec for later use
    # This ensures the "Run Experiments" button works even after new agent runs
    # IMPORTANT: Only store if no config exists yet - never overwrite from iteration runs
    if not spec.agent_experiment_design_json:
        # Get agent run for tracking
        agent_run = db.query(AgentRun).filter(AgentRun.id == step.agent_run_id).first()
        spec.agent_experiment_design_json = {
            "step_id": str(step_id),
            "agent_run_id": str(agent_run.id) if agent_run else None,
            "variants": variants,
            "recommended_variant": recommended_variant,
            "primary_metric": primary_metric,
            "natural_language_summary": natural_language_summary,
            "stored_at": datetime.utcnow().isoformat(),
            "source_type": "initial",
            "parent_experiment_id": None,
        }
        db.flush()

    # Determine which variants to use
    if request.create_all_variants:
        variants_to_use = variants
    else:
        # Only use recommended variant
        variants_to_use = [v for v in variants if v.get("name") == recommended_variant]
        if not variants_to_use:
            variants_to_use = [variants[0]]  # Fallback to first

    # Create experiments for this dataset spec
    created_experiments = []
    for variant in variants_to_use:
        variant_name = variant.get("name", "unnamed")
        automl_config = variant.get("automl_config", {})
        validation_strategy = variant.get("validation_strategy")
        variant_description = variant.get("description", "")
        is_recommended = (variant_name == recommended_variant)

        # Build experiment plan with agent's recommendation
        experiment_plan = {
            "source": "agent_pipeline",
            "agent_step_id": str(step_id),
            "variant_name": variant_name,
            "variant_description": variant_description,
            "is_recommended_variant": is_recommended,
            "automl_config": automl_config,
            "validation_strategy": validation_strategy,
            "dataset_spec_name": spec.name,
            "agent_reasoning": natural_language_summary,
        }

        # Build rich description
        description_parts = []
        if is_recommended:
            description_parts.append(f"🌟 RECOMMENDED by AI Agent")
        description_parts.append(f"Variant: {variant_name}")
        if variant_description:
            description_parts.append(f"Strategy: {variant_description}")
        if automl_config:
            time_limit = automl_config.get("time_limit", "auto")
            presets = automl_config.get("presets", "default")
            description_parts.append(f"Config: {presets} preset, {time_limit}s time limit")
        if natural_language_summary:
            description_parts.append(f"\nAgent Analysis: {natural_language_summary[:400]}")

        # Auto-fix split strategy for time-based datasets
        experiment_plan, _ = autofix_experiment_plan_for_time_series(
            experiment_plan, spec
        )

        experiment = Experiment(
            project_id=project_id,
            dataset_spec_id=spec.id,
            name=f"{spec.name} - {variant_name}" + (" ⭐" if is_recommended else ""),
            description="\n".join(description_parts)[:2000],
            primary_metric=primary_metric,
            metric_direction="maximize",
            experiment_plan_json=experiment_plan,
        )

        db.add(experiment)
        db.flush()

        created_experiments.append({
            "experiment_id": str(experiment.id),
            "dataset_spec_id": str(spec.id),
            "dataset_spec_name": spec.name,
            "variant_name": variant_name,
            "is_recommended": is_recommended,
            "name": experiment.name,
            "status": "pending",
        })

    db.commit()

    # Optionally queue experiments for execution
    queued_count = 0
    if request.run_immediately:
        from app.tasks import run_experiment_modal

        for exp_info in created_experiments:
            try:
                experiment = db.query(Experiment).filter(
                    Experiment.id == UUID(exp_info["experiment_id"])
                ).first()
                if experiment:
                    task = run_experiment_modal.delay(exp_info["experiment_id"])
                    experiment.celery_task_id = task.id
                    exp_info["status"] = "queued"
                    exp_info["task_id"] = task.id
                    queued_count += 1
            except Exception as e:
                exp_info["status"] = "queue_failed"
                exp_info["error"] = str(e)

        db.commit()

    message = f"Created {len(created_experiments)} experiment(s) for '{spec.name}'"
    if queued_count > 0:
        message += f" ({queued_count} queued)"

    return CreateExperimentsForDatasetResponse(
        experiments=created_experiments,
        message=message,
        created_count=len(created_experiments),
        queued_count=queued_count,
    )


# ============================================
# Create Experiments from Stored Agent Config
# ============================================

class CreateExperimentsFromStoredConfigRequest(BaseModel):
    """Request to create experiments using stored agent config in the dataset spec."""
    dataset_spec_id: UUID = Field(..., description="The dataset spec to create experiments for")
    create_all_variants: bool = Field(
        True,
        description="If true, create one experiment per variant. If false, create only the recommended variant."
    )
    run_immediately: bool = Field(True, description="If true, queue experiments for immediate execution")
    use_modal: Optional[bool] = Field(None, description="If true, use Modal cloud. If None, auto-detect.")


@router.post("/create-experiments-from-stored-config", response_model=CreateExperimentsForDatasetResponse)
def create_experiments_from_stored_config(
    project_id: UUID,
    request: CreateExperimentsFromStoredConfigRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create experiments for a dataset spec using its stored agent experiment design config.

    This endpoint allows creating experiments without requiring the original agent step,
    using the experiment design configuration stored in the dataset spec's
    agent_experiment_design_json field. This is useful when:
    - The original agent run has been superseded by newer runs
    - Users want to re-run experiments with the same configuration
    - Auto-improve iterations need to reference the original experiment design

    The dataset spec must have agent_experiment_design_json populated (happens when
    experiments are first created via batch_apply_experiment_plan or create_experiments_for_dataset).
    """
    from app.tasks.automl import run_automl_experiment_task

    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Verify dataset spec exists and belongs to project
    spec = db.query(DatasetSpec).filter(
        DatasetSpec.id == request.dataset_spec_id,
        DatasetSpec.project_id == project_id,
    ).first()
    if not spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {request.dataset_spec_id} not found in this project",
        )

    # Pre-flight validation: verify columns exist in data source
    validation_result = DatasetValidator(db).validate_dataset_spec(spec.id)
    if not validation_result.is_valid:
        logger.warning(
            f"Dataset spec {spec.id} validation failed: "
            f"missing_target={validation_result.missing_target}, "
            f"missing_features={len(validation_result.missing_features)}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Dataset validation failed - some columns don't exist in the data source",
                "missing_target": validation_result.missing_target,
                "missing_features": validation_result.missing_features[:20],
                "available_columns": sorted(list(validation_result.available_columns))[:50],
                "feedback": validation_result.to_feedback(),
            }
        )

    # Check for stored agent experiment design config
    agent_config = spec.agent_experiment_design_json
    if not agent_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset spec has no stored agent experiment design configuration. "
                   "Run the AI Agent pipeline first, or use the step-based endpoint.",
        )

    # Extract config
    variants = agent_config.get("variants", [])
    recommended_variant = agent_config.get("recommended_variant")
    primary_metric = agent_config.get("primary_metric", "accuracy")
    natural_language_summary = agent_config.get("natural_language_summary", "")
    step_id = agent_config.get("step_id")

    if not variants:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No experiment variants found in stored agent configuration",
        )

    # Determine which variants to use
    if request.create_all_variants:
        variants_to_use = variants
    else:
        variants_to_use = [v for v in variants if v.get("name") == recommended_variant]
        if not variants_to_use:
            variants_to_use = [variants[0]]

    # Create experiments
    created_experiments = []
    for variant in variants_to_use:
        variant_name = variant.get("name", "unnamed")
        automl_config = variant.get("automl_config", {})
        validation_strategy = variant.get("validation_strategy")
        variant_description = variant.get("description", "")
        is_recommended = (variant_name == recommended_variant)

        # Build experiment plan
        experiment_plan = {
            "source": "agent_pipeline_stored",
            "agent_step_id": step_id,
            "variant_name": variant_name,
            "variant_description": variant_description,
            "is_recommended_variant": is_recommended,
            "automl_config": automl_config,
            "validation_strategy": validation_strategy,
            "dataset_spec_name": spec.name,
            "agent_reasoning": natural_language_summary,
        }

        # Build description
        description_parts = []
        if is_recommended:
            description_parts.append(f"🌟 RECOMMENDED by AI Agent")
        description_parts.append(f"Variant: {variant_name}")
        if variant_description:
            description_parts.append(f"Strategy: {variant_description}")
        if automl_config:
            time_limit = automl_config.get("time_limit", "auto")
            presets = automl_config.get("presets", "default")
            description_parts.append(f"Config: {presets} preset, {time_limit}s time limit")
        if natural_language_summary:
            description_parts.append(f"\nAgent Analysis: {natural_language_summary[:400]}")

        # Auto-fix split strategy for time-based datasets
        experiment_plan, _ = autofix_experiment_plan_for_time_series(
            experiment_plan, spec
        )

        experiment = Experiment(
            project_id=project_id,
            dataset_spec_id=spec.id,
            name=f"{spec.name} - {variant_name}" + (" ⭐" if is_recommended else ""),
            description="\n".join(description_parts)[:2000],
            primary_metric=primary_metric,
            metric_direction="maximize",
            experiment_plan_json=experiment_plan,
        )

        db.add(experiment)
        db.flush()

        created_experiments.append({
            "experiment_id": str(experiment.id),
            "dataset_spec_id": str(spec.id),
            "dataset_spec_name": spec.name,
            "variant_name": variant_name,
            "is_recommended": is_recommended,
            "name": experiment.name,
            "status": "pending",
        })

    db.commit()

    # Optionally queue experiments
    queued_count = 0
    if request.run_immediately:
        from app.tasks import run_experiment_modal

        for exp_info in created_experiments:
            try:
                experiment = db.query(Experiment).filter(
                    Experiment.id == UUID(exp_info["experiment_id"])
                ).first()
                if experiment:
                    task = run_experiment_modal.delay(exp_info["experiment_id"])
                    experiment.celery_task_id = task.id
                    exp_info["status"] = "queued"
                    exp_info["task_id"] = task.id
                    queued_count += 1
            except Exception as e:
                exp_info["status"] = "queue_failed"
                exp_info["error"] = str(e)

        db.commit()

    message = f"Created {len(created_experiments)} experiment(s) for '{spec.name}' from stored config"
    if queued_count > 0:
        message += f" ({queued_count} queued)"

    return CreateExperimentsForDatasetResponse(
        experiments=created_experiments,
        message=message,
        created_count=len(created_experiments),
        queued_count=queued_count,
    )


# ============================================
# Experiment Results Pipeline
# ============================================

class ResultsPipelineRequest(BaseModel):
    """Request to start a results analysis pipeline."""

    run_async: bool = Field(
        False,
        description="If true, return immediately with run_id. If false, wait for completion."
    )


class ResultsPipelineResponse(BaseModel):
    """Response from starting a results analysis pipeline."""

    run_id: UUID
    status: str
    message: str


# Create a separate router for experiment-scoped agent endpoints
experiment_agent_router = APIRouter(prefix="/experiments/{experiment_id}/agent", tags=["Experiment Agent"])


@experiment_agent_router.post("/run-results-pipeline", response_model=ResultsPipelineResponse)
async def run_results_pipeline(
    experiment_id: UUID,
    request: Optional[ResultsPipelineRequest] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Start the AI-powered results analysis pipeline for an experiment.

    This creates an agent run with 2 sequential steps:
    1. RESULTS_INTERPRETATION - Analyze experiment results and provide recommendations
    2. RESULTS_CRITIC - Review results for potential issues (overfitting, data leakage, etc.)

    The experiment must be in 'completed' status to run this pipeline.

    If run_async=False (default), waits for the pipeline to complete.
    If run_async=True, returns immediately with the run_id for polling.

    Returns:
        run_id: UUID of the created agent run
        status: Current status of the run
        message: Human-readable status message
    """
    # Verify experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Verify experiment is completed
    if experiment.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Experiment must be completed to run results pipeline. Current status: {experiment.status}",
        )

    # Get LLM client
    provider, api_key_str = get_llm_provider_and_key(db)
    llm_client = get_llm_client(provider, api_key_str)

    run_async = request.run_async if request else False

    try:
        if run_async:
            # Create the pipeline but don't run it yet
            agent_run = create_results_pipeline(
                db=db,
                experiment_id=experiment_id,
            )

            return ResultsPipelineResponse(
                run_id=agent_run.id,
                status=agent_run.status.value,
                message="Results pipeline created. Poll GET /agent-runs/{run_id} for status.",
            )
        else:
            # Run the full pipeline synchronously
            agent_run = await run_results_pipeline_for_experiment(
                db=db,
                experiment_id=experiment_id,
                llm_client=llm_client,
            )

            return ResultsPipelineResponse(
                run_id=agent_run.id,
                status=agent_run.status.value,
                message=f"Results pipeline completed with {len(agent_run.steps)} steps.",
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Results pipeline failed: {str(e)}",
        )


@experiment_agent_router.get("/runs", response_model=AgentRunList)
def list_experiment_agent_runs(
    experiment_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List agent runs for an experiment.

    Returns paginated list of agent runs (results pipelines), ordered by creation time (newest first).
    """
    # Verify experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Query agent runs for this experiment
    query = db.query(AgentRun).filter(AgentRun.experiment_id == experiment_id)
    total = query.count()
    runs = query.order_by(AgentRun.created_at.desc()).offset(skip).limit(limit).all()

    return AgentRunList(
        items=[AgentRunRead.model_validate(run) for run in runs],
        total=total,
        skip=skip,
        limit=limit,
    )


@experiment_agent_router.get("/runs/{run_id}", response_model=AgentRunWithStepsAndLogs)
def get_experiment_agent_run(
    experiment_id: UUID,
    run_id: UUID,
    include_logs: bool = Query(True, description="Include step logs in response"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a specific agent run for an experiment with its steps and logs.

    Returns the full run details including all steps (results_interpretation, results_critic)
    and their execution logs.
    """
    # Verify experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    agent_run = get_agent_run_with_steps(db, run_id, include_logs=include_logs)
    if not agent_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run {run_id} not found",
        )

    # Verify run belongs to this experiment
    if agent_run.experiment_id != experiment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent run does not belong to this experiment",
        )

    return AgentRunWithStepsAndLogs.model_validate(agent_run)


# ============================================
# Dataset Discovery Endpoints
# ============================================

class DatasetDiscoveryRequest(BaseModel):
    """Request to start a dataset discovery pipeline."""

    project_description: str = Field(
        ...,
        min_length=10,
        description="Description of what the user wants to predict or achieve"
    )
    constraints: Optional[dict] = Field(
        None,
        description="Optional constraints: geography, allow_public_data, licensing requirements"
    )
    run_async: bool = Field(
        False,
        description="If true, return immediately with run_id. If false, wait for completion."
    )


class DatasetDiscoveryResponse(BaseModel):
    """Response from starting a dataset discovery pipeline."""

    run_id: UUID
    status: str
    message: str


class ApplyDiscoveredDatasetsRequest(BaseModel):
    """Request to apply discovered datasets as data sources."""

    dataset_indices: List[int] = Field(
        ...,
        description="Indices of discovered datasets to apply (from output_json.discovered_datasets)"
    )


class ApplyDiscoveredDatasetsResponse(BaseModel):
    """Response from applying discovered datasets."""

    data_sources: List[dict] = Field(
        ...,
        description="List of created data sources with their IDs and details"
    )
    message: str


@router.post("/run-dataset-discovery", response_model=DatasetDiscoveryResponse)
async def run_dataset_discovery(
    project_id: UUID,
    request: DatasetDiscoveryRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Start the AI-powered dataset discovery pipeline.

    This endpoint helps users find relevant public datasets BEFORE they have any
    data sources configured. It uses an LLM to search for and recommend datasets
    based on the user's description.

    This is designed to run as a pre-step before the main setup pipeline when
    the user doesn't have their own data yet.

    Args:
        project_id: UUID of the project
        request: Description and optional constraints for dataset search

    Returns:
        run_id: UUID of the created agent run
        status: Current status of the run
        message: Human-readable status message

    The discovered datasets will be available in the agent run's result_json
    under 'discovered_datasets', each containing:
    - name: Dataset name
    - source_url: URL to access the dataset
    - schema_summary: Estimated schema info (rows, columns, target candidate)
    - licensing: License information
    - fit_for_purpose: Why this dataset is relevant
    """
    from app.services.agent_executor import (
        create_dataset_discovery_pipeline,
        run_dataset_discovery_pipeline,
    )

    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Get LLM client
    provider, api_key_str = get_llm_provider_and_key(db)
    llm_client = get_llm_client(provider, api_key_str)

    try:
        if request.run_async:
            # Create the pipeline but don't run it yet
            agent_run = create_dataset_discovery_pipeline(
                db=db,
                project_id=project_id,
                project_description=request.project_description,
                constraints=request.constraints,
            )

            return DatasetDiscoveryResponse(
                run_id=agent_run.id,
                status=agent_run.status.value,
                message="Dataset discovery pipeline created. Poll GET /runs/{run_id} for status.",
            )
        else:
            # Run the full pipeline synchronously
            agent_run = await run_dataset_discovery_pipeline(
                db=db,
                project_id=project_id,
                project_description=request.project_description,
                constraints=request.constraints,
                llm_client=llm_client,
            )

            # Count discovered datasets
            discovered_count = 0
            if agent_run.result_json:
                discovered_count = len(agent_run.result_json.get("discovered_datasets", []))

            return DatasetDiscoveryResponse(
                run_id=agent_run.id,
                status=agent_run.status.value,
                message=f"Dataset discovery completed. Found {discovered_count} relevant datasets.",
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset discovery failed: {str(e)}",
        )


@router.post("/apply-discovered-datasets/{run_id}", response_model=ApplyDiscoveredDatasetsResponse)
def apply_discovered_datasets(
    project_id: UUID,
    run_id: UUID,
    request: ApplyDiscoveredDatasetsRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Apply selected discovered datasets as data sources.

    After running dataset discovery, use this endpoint to download the selected
    datasets and create DataSource records with the actual data files.

    The endpoint will:
    1. Download each selected dataset from its source URL
    2. Analyze the schema of the downloaded file
    3. Store the file locally
    4. Create a DataSource record with type='file_upload'

    Args:
        project_id: UUID of the project
        run_id: UUID of the dataset discovery agent run
        request: List of dataset indices to apply

    Returns:
        data_sources: List of created data sources with their IDs and download status
        message: Success message

    Note: Downloads are performed synchronously. For large datasets, this may take
    some time. Failed downloads will be reported but won't stop other downloads.
    """
    from app.models.data_source import DataSource, DataSourceType
    from app.services.dataset_downloader import (
        DatasetDownloader,
        DatasetDownloadError,
    )

    # Verify project exists
    project = get_project_or_404(project_id, db, current_user)

    # Get the agent run
    agent_run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if not agent_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run {run_id} not found",
        )

    # Verify run belongs to this project
    if agent_run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent run does not belong to this project",
        )

    # Verify it's a completed discovery run
    if agent_run.status.value != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent run must be completed. Current status: {agent_run.status.value}",
        )

    # Get discovered datasets from result
    result = agent_run.result_json or {}
    discovered_datasets = result.get("discovered_datasets", [])

    if not discovered_datasets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No discovered datasets found in agent run results",
        )

    # Validate indices
    invalid_indices = [i for i in request.dataset_indices if i < 0 or i >= len(discovered_datasets)]
    if invalid_indices:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dataset indices: {invalid_indices}. Valid range: 0-{len(discovered_datasets)-1}",
        )

    # Initialize downloader
    downloader = DatasetDownloader()

    # Download and create DataSource for each selected dataset
    created_sources = []
    download_errors = []

    for idx in request.dataset_indices:
        dataset = discovered_datasets[idx]
        dataset_name = dataset.get("name", f"Discovered Dataset {idx + 1}")
        source_url = dataset.get("source_url", "")

        if not source_url:
            download_errors.append({
                "index": idx,
                "name": dataset_name,
                "error": "No source URL provided",
            })
            continue

        try:
            # Extract expected schema info for validation
            expected_columns = None
            expected_row_count = None
            schema_summary_hint = dataset.get("schema_summary")
            if schema_summary_hint:
                # Extract column names from the schema hint
                if isinstance(schema_summary_hint, dict):
                    expected_columns = schema_summary_hint.get("columns", [])
                    expected_row_count = schema_summary_hint.get("estimated_rows")
                elif isinstance(schema_summary_hint, str):
                    # Parse column names from string like "col1, col2, col3"
                    expected_columns = [c.strip() for c in schema_summary_hint.split(",") if c.strip()]

            # Download and analyze the dataset (with Selenium fallback)
            file_path, config_json, schema_summary = downloader.download_dataset(
                source_url=source_url,
                project_id=str(project_id),
                dataset_name=dataset_name,
                timeout=180,  # 3 minute timeout per dataset
                use_selenium=True,  # Enable Selenium fallback for complex sites
                expected_columns=expected_columns,
                expected_row_count=expected_row_count,
            )

            # Add discovery metadata to config
            config_json["licensing"] = dataset.get("licensing", "Unknown")
            config_json["fit_for_purpose"] = dataset.get("fit_for_purpose", "")
            config_json["discovered_from"] = {
                "agent_run_id": str(run_id),
                "dataset_index": idx,
            }

            # Create DataSource with actual downloaded file
            data_source = DataSource(
                project_id=project_id,
                name=dataset_name,
                type=DataSourceType.FILE_UPLOAD,  # Use FILE_UPLOAD since we have the actual file
                config_json=config_json,
                schema_summary=schema_summary,
            )

            db.add(data_source)
            db.flush()

            created_sources.append({
                "data_source_id": str(data_source.id),
                "name": data_source.name,
                "source_url": source_url,
                "licensing": config_json.get("licensing", "Unknown"),
                "downloaded": True,
                "file_size_bytes": config_json.get("file_size_bytes"),
                "row_count": schema_summary.get("row_count") if schema_summary else None,
                "column_count": len(schema_summary.get("columns", [])) if schema_summary else None,
            })

        except DatasetDownloadError as e:
            download_errors.append({
                "index": idx,
                "name": dataset_name,
                "error": str(e),
            })
        except Exception as e:
            download_errors.append({
                "index": idx,
                "name": dataset_name,
                "error": f"Unexpected error: {str(e)}",
            })

    # Commit successful downloads
    if created_sources:
        db.commit()

    # Build response message
    if download_errors and not created_sources:
        # All downloads failed
        error_details = "; ".join([f"{e['name']}: {e['error']}" for e in download_errors])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"All downloads failed: {error_details}",
        )

    message_parts = [f"Successfully downloaded {len(created_sources)} dataset(s)"]
    if download_errors:
        message_parts.append(f"{len(download_errors)} failed")

    return ApplyDiscoveredDatasetsResponse(
        data_sources=created_sources,
        message=". ".join(message_parts),
    )


# ============================================
# Data Architect Pipeline Endpoints
# ============================================

class DataArchitectRequest(BaseModel):
    """Request to start a Data Architect pipeline."""

    target_hint: Optional[str] = Field(
        None,
        description="Optional hint about which column is the target to predict"
    )
    run_async: bool = Field(
        False,
        description="If True, just create the run and return immediately. If False (default), run synchronously."
    )
    auto_run_setup: bool = Field(
        True,
        description="If True (default), automatically run AI Setup pipeline on the newly created training dataset."
    )


class DataArchitectResponse(BaseModel):
    """Response from starting a Data Architect pipeline."""

    agent_run_id: UUID
    status: str
    message: str
    setup_run_id: Optional[UUID] = Field(
        None,
        description="Agent run ID for the auto-started AI Setup pipeline (if auto_run_setup was True)"
    )


@router.post("/run-data-architect", response_model=DataArchitectResponse)
async def run_data_architect(
    project_id: UUID,
    request: DataArchitectRequest = DataArchitectRequest(),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Start the Data Architect pipeline to build a training dataset.

    This endpoint runs the full Data Architect pipeline:
    1. DATASET_INVENTORY - Profile all data sources
    2. RELATIONSHIP_DISCOVERY - Discover relationships between tables
    3. TRAINING_DATASET_PLANNING - Use LLM to plan the training dataset
    4. TRAINING_DATASET_BUILD - Build and register the training dataset

    The pipeline runs in the background and returns immediately with the agent_run_id.
    Use the agent run status endpoints to check progress.

    Args:
        project_id: UUID of the project
        request: Optional target_hint to guide the LLM

    Returns:
        agent_run_id: UUID of the created agent run
    """
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    # Verify project has data sources
    from app.models import DataSource
    data_source_count = db.query(DataSource).filter(
        DataSource.project_id == project_id
    ).count()

    if data_source_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project has no data sources. Please add data sources first.",
        )

    try:
        if request.run_async:
            # Just create the pipeline and return immediately
            agent_run = create_data_architect_pipeline(
                db=db,
                project_id=project_id,
                target_hint=request.target_hint,
            )

            return DataArchitectResponse(
                agent_run_id=agent_run.id,
                status="pending",
                message=(
                    f"Data Architect pipeline created with {len(agent_run.steps)} steps. "
                    "Use the agent run endpoints to check progress."
                ),
            )
        else:
            # Run the full pipeline synchronously (default)
            # Get LLM client
            provider, api_key_str = get_llm_provider_and_key(db)
            llm_client = get_llm_client(provider, api_key_str)

            agent_run = await run_data_architect_pipeline(
                db=db,
                project_id=project_id,
                target_hint=request.target_hint,
                llm_client=llm_client,
            )

            # Auto-run AI Setup pipeline on the new training dataset if requested
            setup_run_id = None
            if request.auto_run_setup and agent_run.status.value == "completed":
                result = agent_run.result_json or {}
                training_data_source_id = result.get("training_data_source_id")

                if training_data_source_id:
                    logger.info(f"Auto-running AI Setup pipeline on training dataset {training_data_source_id}")
                    try:
                        setup_run = await run_setup_pipeline_for_project(
                            db=db,
                            project_id=project_id,
                            data_source_id=UUID(training_data_source_id) if isinstance(training_data_source_id, str) else training_data_source_id,
                            description=project.description or f"Analyze training dataset for {project.name}",
                            llm_client=llm_client,
                        )
                        setup_run_id = setup_run.id
                        logger.info(f"AI Setup pipeline completed: {setup_run_id}")
                    except Exception as setup_err:
                        logger.warning(f"Auto AI Setup failed (continuing): {setup_err}")

            message = f"Data Architect pipeline completed with {len(agent_run.steps)} steps."
            if setup_run_id:
                message += " AI Setup pipeline also completed."

            return DataArchitectResponse(
                agent_run_id=agent_run.id,
                status=agent_run.status.value,
                message=message,
                setup_run_id=setup_run_id,
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Data Architect pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}",
        )


# ============================================
# Orchestration Endpoints
# ============================================


class DebatePartnerOption(BaseModel):
    """A debate partner option with display info."""
    model: str = Field(description="Model identifier")
    display_name: str = Field(description="Human-readable name")
    provider: str = Field(description="LLM provider (gemini, openai, anthropic)")
    description: str = Field(description="Brief description")


class OrchestrationOptionsResponse(BaseModel):
    """Available orchestration options."""

    orchestration_modes: List[str] = Field(
        description="Available orchestration modes"
    )
    debate_modes: List[str] = Field(
        description="Available debate modes"
    )
    judge_models: List[str] = Field(
        description="Available OpenAI models for judging"
    )
    default_judge_model: str = Field(
        description="Default judge model"
    )
    debate_partners: List[DebatePartnerOption] = Field(
        description="Available LLM models for debate critique"
    )
    default_debate_partner: str = Field(
        description="Default debate partner model"
    )
    default_max_debate_rounds: int = Field(
        default=3,
        description="Default max debate rounds before calling judge"
    )


# Available debate partner models
AVAILABLE_DEBATE_PARTNERS = [
    DebatePartnerOption(
        model="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        provider="gemini",
        description="Fast and efficient. Good for quick debates."
    ),
    DebatePartnerOption(
        model="gemini-2.0-pro",
        display_name="Gemini 2.0 Pro",
        provider="gemini",
        description="More capable Gemini model. Better for complex critiques."
    ),
    DebatePartnerOption(
        model="claude-sonnet-4",
        display_name="Claude Sonnet 4",
        provider="anthropic",
        description="Anthropic's balanced model. Thoughtful analysis."
    ),
    DebatePartnerOption(
        model="gpt-4o",
        display_name="GPT-4o",
        provider="openai",
        description="OpenAI's multimodal model. Strong reasoning."
    ),
    DebatePartnerOption(
        model="gpt-5.1",
        display_name="GPT-5.1",
        provider="openai",
        description="Latest OpenAI model. Most capable critic."
    ),
]

DEFAULT_DEBATE_PARTNER = "gemini-2.0-flash"


@orchestration_router.get("/options", response_model=OrchestrationOptionsResponse)
async def get_orchestration_options():
    """Get available orchestration options for pipelines.

    Returns:
        - Available orchestration modes (sequential, project_manager)
        - Available debate modes (disabled, enabled)
        - Available OpenAI judge models
        - Available debate partner LLMs
        - Default settings
    """
    return OrchestrationOptionsResponse(
        orchestration_modes=["sequential", "project_manager"],
        debate_modes=["disabled", "enabled"],
        judge_models=get_available_judge_models(),
        default_judge_model=DEFAULT_JUDGE_MODEL,
        debate_partners=AVAILABLE_DEBATE_PARTNERS,
        default_debate_partner=DEFAULT_DEBATE_PARTNER,
        default_max_debate_rounds=3,
    )


class JudgeModelsResponse(BaseModel):
    """Available judge models."""

    models: List[str] = Field(description="List of available OpenAI models for judging")
    default: str = Field(description="Default judge model")


@orchestration_router.get("/judge-models", response_model=JudgeModelsResponse)
async def get_judge_models():
    """Get available OpenAI models that can be used as judge.

    Returns list of model names that can be passed to the judge_model parameter
    when starting a pipeline with debate mode enabled.
    """
    return JudgeModelsResponse(
        models=get_available_judge_models(),
        default=DEFAULT_JUDGE_MODEL,
    )


# =============================================================================
# User Holdout Set Endpoints
# =============================================================================


class HoldoutSetResponse(BaseModel):
    """Response for holdout set information."""

    id: UUID
    project_id: UUID
    data_source_id: UUID
    holdout_percentage: float
    total_rows_original: int
    holdout_row_count: int
    training_row_count: int
    target_column: Optional[str]
    feature_columns: Optional[List[str]]
    created_at: str


class HoldoutRowResponse(BaseModel):
    """Response for a single holdout row."""

    index: int
    total_rows: int
    data: Dict[str, Any]
    target_column: Optional[str] = None
    target_value: Optional[Any] = None


class HoldoutRowsResponse(BaseModel):
    """Response for multiple holdout rows."""

    holdout_set_id: UUID
    total_rows: int
    rows: List[Dict[str, Any]]
    target_column: Optional[str]


@router.get("/holdout-set", response_model=Optional[HoldoutSetResponse])
async def get_project_holdout_set(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get the holdout set for a project.

    Returns information about the user-controlled holdout set, if one exists.
    This set was created before the pipeline ran and can be used for manual
    model validation.
    """
    from app.services.user_holdout_service import get_holdout_set_for_project

    # Verify project exists and user has access
    project = get_project_or_404(project_id, db, current_user)

    holdout_set = get_holdout_set_for_project(db, project_id)
    if not holdout_set:
        return None

    return HoldoutSetResponse(
        id=holdout_set.id,
        project_id=holdout_set.project_id,
        data_source_id=holdout_set.data_source_id,
        holdout_percentage=holdout_set.holdout_percentage,
        total_rows_original=holdout_set.total_rows_original,
        holdout_row_count=holdout_set.holdout_row_count,
        training_row_count=holdout_set.training_row_count,
        target_column=holdout_set.target_column,
        feature_columns=holdout_set.feature_columns_json,
        created_at=holdout_set.created_at.isoformat(),
    )


@router.get("/holdout-set/row/{row_index}", response_model=HoldoutRowResponse)
async def get_holdout_row(
    project_id: UUID,
    row_index: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a specific row from the holdout set.

    Args:
        project_id: Project ID
        row_index: 0-based index of the row to retrieve

    Returns:
        The row data with feature values and the target value (if target column is set)
    """
    from app.services.user_holdout_service import get_holdout_set_for_project

    # Verify project exists and user has access
    project = get_project_or_404(project_id, db, current_user)

    holdout_set = get_holdout_set_for_project(db, project_id)
    if not holdout_set:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No holdout set found for this project"
        )

    row_data = holdout_set.get_row(row_index)
    if row_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Row index {row_index} not found in holdout set"
        )

    # Extract target value if target column is set
    target_value = None
    target_column = holdout_set.target_column
    if target_column and target_column in row_data:
        target_value = row_data[target_column]

    return HoldoutRowResponse(
        index=row_index,
        total_rows=holdout_set.holdout_row_count,
        data=row_data,
        target_column=target_column,
        target_value=target_value,
    )


@router.get("/holdout-set/rows", response_model=HoldoutRowsResponse)
async def get_all_holdout_rows(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get all rows from the holdout set.

    Returns all holdout rows for use in batch validation testing.
    """
    from app.services.user_holdout_service import get_holdout_set_for_project

    # Verify project exists and user has access
    project = get_project_or_404(project_id, db, current_user)

    holdout_set = get_holdout_set_for_project(db, project_id)
    if not holdout_set:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No holdout set found for this project"
        )

    return HoldoutRowsResponse(
        holdout_set_id=holdout_set.id,
        total_rows=holdout_set.holdout_row_count,
        rows=holdout_set.get_holdout_rows(),
        target_column=holdout_set.target_column,
    )


@router.put("/holdout-set/target-column")
async def update_holdout_target_column(
    project_id: UUID,
    target_column: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update the target column for the holdout set.

    This is typically called after the pipeline determines the target column,
    so the holdout set knows which column contains the "answer" for validation.
    """
    from app.services.user_holdout_service import (
        get_holdout_set_for_project,
        update_holdout_target_column as update_target,
    )

    # Verify project exists and user has access
    project = get_project_or_404(project_id, db, current_user)

    holdout_set = get_holdout_set_for_project(db, project_id)
    if not holdout_set:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No holdout set found for this project"
        )

    updated = update_target(db, holdout_set.id, target_column)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update target column"
        )

    return {"message": "Target column updated", "target_column": target_column}

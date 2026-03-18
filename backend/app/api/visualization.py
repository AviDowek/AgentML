"""Visualization API endpoints."""
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.api.dependencies import check_project_access
from app.models.api_key import LLMProvider
from app.models.data_source import DataSource
from app.models.project import Project
from app.models.user import User
from app.models.visualization import Visualization
from app.services import api_key_service
from app.services.llm_client import get_llm_client
from app.services.visualization_service import (
    get_data_summary_for_llm,
    generate_visualization_code,
    generate_default_visualizations,
    explain_visualization,
    execute_visualization_code,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/projects/{project_id}/visualize", tags=["Visualization"])


# ==================== Request/Response Schemas ====================


class GenerateVisualizationRequest(BaseModel):
    """Request to generate a visualization."""
    data_source_id: UUID = Field(..., description="ID of the data source to visualize")
    request: str = Field(..., min_length=1, description="User's visualization request")
    previous_visualizations: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Previous visualizations for context"
    )


class VisualizationResponse(BaseModel):
    """Response containing generated visualization."""
    code: str = Field(..., description="Generated Python code")
    title: str = Field(..., description="Visualization title")
    description: str = Field(..., description="What the visualization shows")
    chart_type: str = Field(..., description="Type of chart")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image if executed")
    error: Optional[str] = Field(None, description="Error message if execution failed")


class ExecuteCodeRequest(BaseModel):
    """Request to execute visualization code."""
    code: str = Field(..., description="Python visualization code to execute")


class ExecuteCodeResponse(BaseModel):
    """Response from code execution."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    error: Optional[str] = Field(None, description="Error message if failed")


class ExplainVisualizationRequest(BaseModel):
    """Request to explain a visualization."""
    data_source_id: UUID = Field(..., description="ID of the data source")
    visualization_info: Dict[str, Any] = Field(..., description="Info about the visualization")


class ExplainVisualizationResponse(BaseModel):
    """Response with visualization explanation."""
    explanation: str = Field(..., description="Explanation of what the visualization shows")


class SuggestVisualizationsRequest(BaseModel):
    """Request for visualization suggestions."""
    data_source_id: UUID = Field(..., description="ID of the data source")


class VisualizationSuggestion(BaseModel):
    """A single visualization suggestion."""
    title: str
    description: str
    chart_type: str
    request: str


class SuggestVisualizationsResponse(BaseModel):
    """Response with visualization suggestions."""
    suggestions: List[VisualizationSuggestion]
    data_summary: Dict[str, Any] = Field(..., description="Summary of the data structure")


# ==================== Helper Functions ====================


def get_llm_provider_and_key(db: Session):
    """Get the LLM provider and API key to use."""
    key_status = api_key_service.get_api_key_status(db)

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


def get_data_source_file_path(db: Session, project_id: UUID, data_source_id: UUID) -> str:
    """Get the file path for a data source."""
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Get data source
    data_source = db.query(DataSource).filter(
        DataSource.id == data_source_id,
        DataSource.project_id == project_id
    ).first()
    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {data_source_id} not found in project {project_id}",
        )

    # Ensure file exists on disk (restore from DB if needed)
    from app.services.file_storage import ensure_file_on_disk
    try:
        return str(ensure_file_on_disk(data_source))
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


def _build_project_context(db: Session, project: Project) -> Dict[str, Any]:
    """Build context from project info and agent run results for better visualization suggestions.

    Gathers:
    - Project name, description, task type
    - Problem understanding from agent runs (target variable, metrics)
    - Data audit findings (issues, recommendations)
    - Any identified relationships or patterns

    Args:
        db: Database session
        project: Project model instance

    Returns:
        Dictionary with project context for visualization suggestions
    """
    from app.models.agent_run import AgentRun, AgentStep, AgentRunStatus, AgentStepType, AgentStepStatus

    context = {
        "project_name": project.name,
        "project_description": project.description or "",
        "task_type": project.task_type.value if project.task_type else None,
        "target_column": None,
        "problem_summary": None,
        "data_quality_issues": [],
        "recommended_analyses": [],
        "relationships": [],
    }

    # Get recent completed agent runs for this project
    recent_runs = (
        db.query(AgentRun)
        .filter(
            AgentRun.project_id == project.id,
            AgentRun.status == AgentRunStatus.COMPLETED
        )
        .order_by(AgentRun.updated_at.desc())
        .limit(5)
        .all()
    )

    for run in recent_runs:
        # Check each step for useful context
        for step in run.steps:
            if step.status != AgentStepStatus.COMPLETED or not step.output_json:
                continue

            output = step.output_json

            # Extract from problem understanding
            if step.step_type == AgentStepType.PROBLEM_UNDERSTANDING:
                if output.get("target_column"):
                    context["target_column"] = output["target_column"]
                if output.get("reasoning"):
                    context["problem_summary"] = output["reasoning"][:500]
                if output.get("primary_metric"):
                    context["primary_metric"] = output["primary_metric"]

            # Extract from data audit
            elif step.step_type == AgentStepType.DATA_AUDIT:
                if output.get("issues"):
                    # Get top issues that might affect visualization
                    issues = output["issues"][:5] if isinstance(output["issues"], list) else []
                    context["data_quality_issues"] = issues
                if output.get("recommendations"):
                    recs = output["recommendations"][:3] if isinstance(output["recommendations"], list) else []
                    context["recommended_analyses"] = recs
                if output.get("correlations"):
                    context["correlations"] = output["correlations"]

            # Extract from relationship discovery
            elif step.step_type == AgentStepType.RELATIONSHIP_DISCOVERY:
                if output.get("relationships"):
                    rels = output["relationships"][:5] if isinstance(output["relationships"], list) else []
                    context["relationships"] = rels

            # Extract from dataset design
            elif step.step_type == AgentStepType.DATASET_DESIGN:
                if output.get("target_column"):
                    context["target_column"] = output["target_column"]
                if output.get("features_to_include"):
                    context["key_features"] = output["features_to_include"][:10]

    return context


# ==================== Endpoints ====================


@router.post("/generate", response_model=VisualizationResponse)
async def generate_visualization(
    project_id: UUID,
    request: GenerateVisualizationRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Generate visualization code for a data source.

    Uses GPT-4.1 to generate Python code based on the data structure
    (column names, types, statistics) without sending raw data.

    If the generated code fails to execute, the error is fed back to the LLM
    to retry with corrections (up to 3 attempts).
    """
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    # Get file path
    file_path = get_data_source_file_path(db, project_id, request.data_source_id)

    # Get data summary (no raw data, just metadata)
    try:
        data_summary = get_data_summary_for_llm(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read data file: {str(e)}",
        )

    # Get LLM client - use gpt-5.1 for visualization
    provider, api_key_str = get_llm_provider_and_key(db)
    llm_client = get_llm_client(provider, api_key_str, model="gpt-5.1")

    max_retries = 3
    last_error = None
    last_code = None
    viz_result = None

    for attempt in range(max_retries):
        # Generate visualization code (with error feedback on retries)
        try:
            viz_result = await generate_visualization_code(
                llm_client=llm_client,
                data_summary=data_summary,
                user_request=request.request,
                file_path=file_path,
                previous_visualizations=request.previous_visualizations,
                error_feedback=last_error,
                failed_code=last_code,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate visualization code: {str(e)}",
            )

        # Execute the code to get the image
        code = viz_result.get("code", "")
        logger.info(f"Attempt {attempt + 1}/{max_retries}: Generated visualization code ({len(code)} chars), executing...")
        exec_result = execute_visualization_code(code)

        image_base64 = exec_result.get("image_base64")
        error = exec_result.get("error")

        if image_base64 and not error:
            # Success! Return the result
            logger.info(f"Visualization generated successfully on attempt {attempt + 1}")
            return VisualizationResponse(
                code=code,
                title=viz_result.get("title", "Visualization"),
                description=viz_result.get("description", ""),
                chart_type=viz_result.get("chart_type", "unknown"),
                image_base64=image_base64,
                error=None,
            )

        # Failed - save error for next retry
        last_error = error
        last_code = code
        logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error}")

    # All retries exhausted - return last result with error
    logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
    return VisualizationResponse(
        code=last_code or "",
        title=viz_result.get("title", "Visualization") if viz_result else "Visualization",
        description=viz_result.get("description", "") if viz_result else "",
        chart_type=viz_result.get("chart_type", "unknown") if viz_result else "unknown",
        image_base64=None,
        error=f"Failed after {max_retries} attempts. Last error: {last_error}",
    )


@router.post("/execute", response_model=ExecuteCodeResponse)
async def execute_code(
    project_id: UUID,
    request: ExecuteCodeRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Execute Python visualization code and return the result.

    The code should produce a matplotlib/seaborn visualization and
    save it as a base64-encoded PNG.
    """
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    # Execute the code
    result = execute_visualization_code(request.code)

    return ExecuteCodeResponse(
        image_base64=result.get("image_base64"),
        error=result.get("error"),
    )


@router.post("/explain", response_model=ExplainVisualizationResponse)
async def explain_viz(
    project_id: UUID,
    request: ExplainVisualizationRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get an AI explanation of what a visualization shows.

    Helps users understand charts and their implications.
    """
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    # Get file path
    file_path = get_data_source_file_path(db, project_id, request.data_source_id)

    # Get data summary
    try:
        data_summary = get_data_summary_for_llm(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read data file: {str(e)}",
        )

    # Get LLM client - use gpt-5.1 for explanation
    provider, api_key_str = get_llm_provider_and_key(db)
    llm_client = get_llm_client(provider, api_key_str, model="gpt-5.1")

    # Get explanation
    try:
        explanation = await explain_visualization(
            llm_client=llm_client,
            visualization_info=request.visualization_info,
            data_summary=data_summary,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation: {str(e)}",
        )

    return ExplainVisualizationResponse(explanation=explanation)


@router.post("/suggestions", response_model=SuggestVisualizationsResponse)
async def get_suggestions(
    project_id: UUID,
    request: SuggestVisualizationsRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get AI-suggested visualizations for initial data exploration.

    Returns 3-4 recommended visualizations based on the data structure,
    project context, and any agent run insights.
    """
    from app.models.agent_run import AgentRun, AgentRunStatus, AgentStepType

    # Get project for context
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    # Get file path
    file_path = get_data_source_file_path(db, project_id, request.data_source_id)

    # Get data summary
    try:
        data_summary = get_data_summary_for_llm(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read data file: {str(e)}",
        )

    # Build project context from project info and agent run results
    project_context = _build_project_context(db, project)

    # Get LLM client - use gpt-5.1 for suggestions
    provider, api_key_str = get_llm_provider_and_key(db)
    llm_client = get_llm_client(provider, api_key_str, model="gpt-5.1")

    # Get suggestions with project context
    try:
        suggestions = await generate_default_visualizations(
            llm_client=llm_client,
            data_summary=data_summary,
            file_path=file_path,
            project_context=project_context,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate suggestions: {str(e)}",
        )

    # Parse suggestions - handle both dict and Pydantic model objects
    parsed_suggestions = []
    logger.info(f"Raw suggestions from LLM: {suggestions}")
    for i, s in enumerate(suggestions):
        logger.info(f"Processing suggestion {i}: type={type(s)}, value={s}")
        if isinstance(s, dict):
            parsed_suggestions.append(VisualizationSuggestion(
                title=s.get("title", ""),
                description=s.get("description", ""),
                chart_type=s.get("chart_type", ""),
                request=s.get("request", ""),
            ))
        elif hasattr(s, 'title'):  # Pydantic model
            parsed_suggestions.append(VisualizationSuggestion(
                title=getattr(s, 'title', ''),
                description=getattr(s, 'description', ''),
                chart_type=getattr(s, 'chart_type', ''),
                request=getattr(s, 'request', ''),
            ))
        else:
            logger.warning(f"Unknown suggestion format: {type(s)} - {s}")

    logger.info(f"Parsed {len(parsed_suggestions)} suggestions")
    for i, ps in enumerate(parsed_suggestions):
        logger.info(f"Parsed suggestion {i}: title='{ps.title}', request='{ps.request}'")

    return SuggestVisualizationsResponse(
        suggestions=parsed_suggestions,
        data_summary=data_summary,
    )


@router.get("/data-summary/{data_source_id}")
async def get_data_summary(
    project_id: UUID,
    data_source_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get the data summary for a data source (no raw data).

    Returns column information, types, statistics for visualization planning.
    """
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    # Get file path
    file_path = get_data_source_file_path(db, project_id, data_source_id)

    # Get data summary
    try:
        data_summary = get_data_summary_for_llm(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read data file: {str(e)}",
        )

    return data_summary


# ==================== Saved Visualizations CRUD ====================


class SaveVisualizationRequest(BaseModel):
    """Request to save a visualization."""
    data_source_id: Optional[UUID] = Field(None, description="ID of the data source")
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    chart_type: Optional[str] = Field(None, max_length=50)
    request: Optional[str] = None
    code: str = Field(..., description="Generated Python code")
    image_base64: Optional[str] = None
    explanation: Optional[str] = None
    is_ai_suggested: Optional[str] = Field(default="false", max_length=10)
    display_order: Optional[str] = Field(default="0", max_length=20)


class SavedVisualizationResponse(BaseModel):
    """Response for a saved visualization."""
    id: UUID
    project_id: UUID
    data_source_id: Optional[UUID] = None
    owner_id: Optional[UUID] = None
    title: str
    description: Optional[str] = None
    chart_type: Optional[str] = None
    request: Optional[str] = None
    code: str
    image_base64: Optional[str] = None
    explanation: Optional[str] = None
    is_ai_suggested: Optional[str] = None
    display_order: Optional[str] = None
    created_at: Any
    updated_at: Any

    class Config:
        from_attributes = True


class UpdateVisualizationRequest(BaseModel):
    """Request to update a visualization."""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    display_order: Optional[str] = Field(None, max_length=20)


@router.post("/saved", response_model=SavedVisualizationResponse, status_code=status.HTTP_201_CREATED)
async def save_visualization(
    project_id: UUID,
    request: SaveVisualizationRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Save a visualization to the database."""
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    user_id = current_user.id if current_user else None

    db_visualization = Visualization(
        project_id=project_id,
        owner_id=user_id,
        data_source_id=request.data_source_id,
        title=request.title,
        description=request.description,
        chart_type=request.chart_type,
        request=request.request,
        code=request.code,
        image_base64=request.image_base64,
        explanation=request.explanation,
        is_ai_suggested=request.is_ai_suggested,
        display_order=request.display_order,
    )
    db.add(db_visualization)
    db.commit()
    db.refresh(db_visualization)
    return db_visualization


@router.get("/saved", response_model=List[SavedVisualizationResponse])
async def list_saved_visualizations(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List all saved visualizations for a project belonging to the current user."""
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    user_id = current_user.id if current_user else None

    query = db.query(Visualization).filter(Visualization.project_id == project_id)
    if user_id:
        query = query.filter(Visualization.owner_id == user_id)

    return query.order_by(Visualization.display_order, Visualization.created_at.desc()).all()


@router.get("/saved/{visualization_id}", response_model=SavedVisualizationResponse)
async def get_saved_visualization(
    project_id: UUID,
    visualization_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a saved visualization by ID."""
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    user_id = current_user.id if current_user else None

    query = db.query(Visualization).filter(
        Visualization.id == visualization_id,
        Visualization.project_id == project_id,
    )
    if user_id:
        query = query.filter(Visualization.owner_id == user_id)

    visualization = query.first()
    if not visualization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visualization {visualization_id} not found",
        )
    return visualization


@router.put("/saved/{visualization_id}", response_model=SavedVisualizationResponse)
async def update_saved_visualization(
    project_id: UUID,
    visualization_id: UUID,
    request: UpdateVisualizationRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update a saved visualization."""
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    user_id = current_user.id if current_user else None

    query = db.query(Visualization).filter(
        Visualization.id == visualization_id,
        Visualization.project_id == project_id,
    )
    if user_id:
        query = query.filter(Visualization.owner_id == user_id)

    visualization = query.first()
    if not visualization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visualization {visualization_id} not found",
        )

    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(visualization, field, value)

    db.commit()
    db.refresh(visualization)
    return visualization


@router.delete("/saved/{visualization_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_saved_visualization(
    project_id: UUID,
    visualization_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Delete a saved visualization."""
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    user_id = current_user.id if current_user else None

    query = db.query(Visualization).filter(
        Visualization.id == visualization_id,
        Visualization.project_id == project_id,
    )
    if user_id:
        query = query.filter(Visualization.owner_id == user_id)

    visualization = query.first()
    if not visualization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visualization {visualization_id} not found",
        )

    db.delete(visualization)
    db.commit()
    return None


# ==================== Dataset Spec Visualization ====================


class GenerateDatasetSpecVisualizationRequest(BaseModel):
    """Request to generate a visualization for a dataset spec."""
    dataset_spec_id: UUID = Field(..., description="ID of the dataset spec to visualize")
    request: str = Field(..., min_length=1, description="User's visualization request")
    previous_visualizations: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Previous visualizations for context"
    )


class DatasetSpecSuggestionsRequest(BaseModel):
    """Request for visualization suggestions for a dataset spec."""
    dataset_spec_id: UUID = Field(..., description="ID of the dataset spec")


def _materialize_dataset_spec(db: Session, dataset_spec) -> str:
    """Materialize a dataset spec to a temporary file and return the path.

    This applies feature engineering and target creation to produce the final dataset.
    """
    import tempfile
    import pandas as pd
    from app.models.data_source import DataSource
    from app.services.feature_engineering import apply_feature_engineering, apply_target_creation

    # Get the data source
    data_source_ids = dataset_spec.data_sources_json
    if isinstance(data_source_ids, list) and len(data_source_ids) > 0:
        data_source_id = data_source_ids[0]
    elif isinstance(data_source_ids, dict):
        data_source_id = (
            data_source_ids.get("source_ids", [None])[0] or
            data_source_ids.get("sources", [None])[0] or
            data_source_ids.get("source_id") or
            data_source_ids.get("primary")
        )
    else:
        raise ValueError("No data source configured in dataset spec")

    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not data_source:
        raise ValueError(f"Data source {data_source_id} not found")

    # Ensure file exists on disk (restore from DB if needed)
    from app.services.file_storage import ensure_file_on_disk
    file_path = str(ensure_file_on_disk(data_source))

    # Read the data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)  # Default to CSV

    # Apply feature engineering if specified
    spec_json = dataset_spec.spec_json or {}
    engineered_features = spec_json.get("engineered_features", [])
    if engineered_features:
        logger.info(f"Applying {len(engineered_features)} feature engineering steps for visualization")
        df = apply_feature_engineering(df, engineered_features, inplace=False)

    # Create target column if specified
    target_creation = spec_json.get("target_creation")
    target_column = dataset_spec.target_column or spec_json.get("target_column")

    if target_creation and target_column and target_column not in df.columns:
        logger.info(f"Creating target column '{target_column}' for visualization")
        if "column_name" not in target_creation:
            target_creation = {**target_creation, "column_name": target_column}
        df = apply_target_creation(df, target_creation, inplace=False)

    # Select only the relevant columns if specified
    if dataset_spec.feature_columns:
        columns_to_keep = list(dataset_spec.feature_columns)
        if target_column and target_column not in columns_to_keep:
            columns_to_keep.append(target_column)
        available_cols = [c for c in columns_to_keep if c in df.columns]
        if available_cols:
            df = df[available_cols]

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    logger.info(f"Materialized dataset spec to {temp_file.name}: {len(df)} rows, {len(df.columns)} columns")
    return temp_file.name


@router.post("/dataset-spec/generate", response_model=VisualizationResponse)
async def generate_dataset_spec_visualization(
    project_id: UUID,
    request: GenerateDatasetSpecVisualizationRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Generate visualization for a dataset spec (with feature engineering applied).

    This endpoint materializes the dataset spec (applying feature engineering and
    target creation) and then generates visualizations on the processed data.
    """
    from app.models.dataset_spec import DatasetSpec

    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    # Get dataset spec
    dataset_spec = db.query(DatasetSpec).filter(
        DatasetSpec.id == request.dataset_spec_id,
        DatasetSpec.project_id == project_id
    ).first()
    if not dataset_spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {request.dataset_spec_id} not found in project {project_id}",
        )

    # Materialize the dataset spec to a temp file
    try:
        temp_file_path = _materialize_dataset_spec(db, dataset_spec)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to materialize dataset spec: {str(e)}",
        )

    # Get data summary
    try:
        data_summary = get_data_summary_for_llm(temp_file_path)
    except Exception as e:
        import os
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read materialized data: {str(e)}",
        )

    # Get LLM client
    provider, api_key_str = get_llm_provider_and_key(db)
    llm_client = get_llm_client(provider, api_key_str, model="gpt-5.1")

    max_retries = 3
    last_error = None
    last_code = None
    viz_result = None

    for attempt in range(max_retries):
        try:
            viz_result = await generate_visualization_code(
                llm_client=llm_client,
                data_summary=data_summary,
                user_request=request.request,
                file_path=temp_file_path,
                previous_visualizations=request.previous_visualizations,
                error_feedback=last_error,
                failed_code=last_code,
            )
        except Exception as e:
            import os
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate visualization code: {str(e)}",
            )

        code = viz_result.get("code", "")
        logger.info(f"Dataset spec viz attempt {attempt + 1}/{max_retries}: Generated code ({len(code)} chars)")
        exec_result = execute_visualization_code(code)

        image_base64 = exec_result.get("image_base64")
        error = exec_result.get("error")

        if image_base64 and not error:
            logger.info(f"Dataset spec visualization generated successfully on attempt {attempt + 1}")
            import os
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return VisualizationResponse(
                code=code,
                title=viz_result.get("title", "Visualization"),
                description=viz_result.get("description", ""),
                chart_type=viz_result.get("chart_type", "unknown"),
                image_base64=image_base64,
                error=None,
            )

        last_error = error
        last_code = code
        logger.warning(f"Dataset spec viz attempt {attempt + 1}/{max_retries} failed: {error}")

    # Cleanup
    import os
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    logger.error(f"All {max_retries} attempts failed for dataset spec visualization")
    return VisualizationResponse(
        code=last_code or "",
        title=viz_result.get("title", "Visualization") if viz_result else "Visualization",
        description=viz_result.get("description", "") if viz_result else "",
        chart_type=viz_result.get("chart_type", "unknown") if viz_result else "unknown",
        image_base64=None,
        error=f"Failed after {max_retries} attempts. Last error: {last_error}",
    )


@router.post("/dataset-spec/suggestions", response_model=SuggestVisualizationsResponse)
async def get_dataset_spec_suggestions(
    project_id: UUID,
    request: DatasetSpecSuggestionsRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get AI-suggested visualizations for a dataset spec.

    Returns visualization suggestions based on the processed dataset
    (with feature engineering and target creation applied).
    """
    from app.models.dataset_spec import DatasetSpec

    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this project",
        )

    # Get dataset spec
    dataset_spec = db.query(DatasetSpec).filter(
        DatasetSpec.id == request.dataset_spec_id,
        DatasetSpec.project_id == project_id
    ).first()
    if not dataset_spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {request.dataset_spec_id} not found in project {project_id}",
        )

    # Materialize the dataset spec
    try:
        temp_file_path = _materialize_dataset_spec(db, dataset_spec)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to materialize dataset spec: {str(e)}",
        )

    # Get data summary
    try:
        data_summary = get_data_summary_for_llm(temp_file_path)
    except Exception as e:
        import os
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read materialized data: {str(e)}",
        )

    # Build project context
    project_context = _build_project_context(db, project)

    # Add dataset spec context
    project_context["dataset_spec_name"] = dataset_spec.name
    project_context["target_column"] = dataset_spec.target_column
    if dataset_spec.feature_columns:
        project_context["feature_columns"] = dataset_spec.feature_columns

    # Get LLM client
    provider, api_key_str = get_llm_provider_and_key(db)
    llm_client = get_llm_client(provider, api_key_str, model="gpt-5.1")

    # Get suggestions
    try:
        suggestions = await generate_default_visualizations(
            llm_client=llm_client,
            data_summary=data_summary,
            file_path=temp_file_path,
            project_context=project_context,
        )
    except Exception as e:
        import os
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate suggestions: {str(e)}",
        )

    # Cleanup temp file
    import os
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    # Parse suggestions
    parsed_suggestions = []
    for s in suggestions:
        if isinstance(s, dict):
            parsed_suggestions.append(VisualizationSuggestion(
                title=s.get("title", ""),
                description=s.get("description", ""),
                chart_type=s.get("chart_type", ""),
                request=s.get("request", ""),
            ))
        elif hasattr(s, 'title'):
            parsed_suggestions.append(VisualizationSuggestion(
                title=getattr(s, 'title', ''),
                description=getattr(s, 'description', ''),
                chart_type=getattr(s, 'chart_type', ''),
                request=getattr(s, 'request', ''),
            ))

    return SuggestVisualizationsResponse(
        suggestions=parsed_suggestions,
        data_summary=data_summary,
    )

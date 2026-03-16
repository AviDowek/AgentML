"""Agent step executor for running multi-step agent pipelines."""
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.models import (
    AgentStep,
    AgentStepStatus,
    AgentStepType,
    AgentStepLog,
    AgentRun,
    AgentRunStatus,
    LogMessageType,
    DataSource,
    Project,
    Experiment,
    Trial,
    ModelVersion,
    PipelineOrchestrationMode,
    DebateMode,
)
from app.models.research_cycle import (
    ResearchCycle,
    CycleExperiment,
    LabNotebookEntry,
    LabNotebookAuthorType,
)
from app.models.dataset_spec import DatasetSpec
from app.models.auto_ds_session import AutoDSSession, AutoDSSessionStatus
from app.schemas.agent import (
    SchemaSummary,
    TrainingDatasetSpec,
    TrainingDatasetPlanningOutput,
    BaseFilter,
    TargetDefinition,
    JoinPlanItem,
    JoinAggregation,
    AggregationFeature,
    DatasetDesignSuggestion,
    ExperimentPlanSuggestion,
)
from app.services.agent_service import (
    build_schema_summary,
    generate_project_config,
    generate_dataset_spec,
    generate_dataset_design,
    generate_experiment_plan,
    expand_user_goal,
    _format_schema_for_prompt,
    execute_with_tools,
)
from app.services.agent_tools import AgentToolExecutor
from app.services.llm_client import BaseLLMClient, get_llm_client
from app.services.training_dataset_builder import materialize_training_dataset, MaterializationResult
from app.services.data_profiler import profile_data_source, profile_all_data_sources
from app.core.exceptions import (
    LLMError,
    LLMTimeoutError,
    LLMParsingError,
    DataError,
    DatasetBuildError,
    AgentPipelineError,
    AgentStepError,
    PipelineCancelledError,
)
from app.services.relationship_discovery import discover_relationships_for_project
from app.services.agents import get_agent_class, is_agent_registered
from app.services.task_context import (
    build_task_context,
    get_task_type_hints,
    format_context_for_prompt,
)
from app.services.context_builder import ContextBuilder
from app.services.prompts import (
    SYSTEM_ROLE_DATA_SCIENTIST,
    SYSTEM_ROLE_ML_ANALYST,
    SYSTEM_ROLE_ML_CRITIC,
    SYSTEM_ROLE_DATA_EVALUATOR,
    SYSTEM_ROLE_DATASET_EXPERT,
    SYSTEM_ROLE_MODEL_REVIEWER,
    SYSTEM_ROLE_DATASET_DESIGNER,
    SYSTEM_ROLE_LAB_NOTEBOOK_AGENT,
    SYSTEM_ROLE_ROBUSTNESS_AUDITOR,
    SYSTEM_ROLE_DATASET_DESIGN_WITH_TOOLS,
    SYSTEM_ROLE_EXPERIMENT_DESIGN_WITH_TOOLS,
    get_data_analysis_prompt,
    get_data_audit_prompt,
    get_results_interpretation_prompt,
    get_results_critic_prompt,
    get_dataset_discovery_prompt,
    get_training_dataset_planning_prompt,
    get_lab_notebook_summary_prompt,
    get_robustness_audit_prompt,
    get_dataset_design_prompt,
    get_experiment_plan_prompt,
    format_leaderboard_for_prompt,
)
from app.services.agents.orchestration import (
    ProjectManagerAgent,
    DebateManager,
    create_debate_manager,
    ORCHESTRABLE_AGENTS,
)
from app.services.llm_client import GeminiClient, OpenAIClient, create_critique_client

logger = logging.getLogger(__name__)

T = TypeVar('T')


def _check_cancellation(db: Session, run_id: UUID, step_type: str = None) -> None:
    """Check if the pipeline run has been cancelled and raise if so.

    This function should be called at the start of each iteration in pipeline loops
    to allow user-requested cancellations to take effect between steps.

    Args:
        db: Database session
        run_id: UUID of the agent run to check
        step_type: Optional step type for error message context

    Raises:
        PipelineCancelledError: If the run status is CANCELLED
    """
    # Refresh the run from DB to get latest status
    agent_run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if agent_run and agent_run.status == AgentRunStatus.CANCELLED:
        logger.info(f"Pipeline {run_id} cancellation detected - stopping execution")
        raise PipelineCancelledError(str(run_id), step_type)


def _build_task_context_for_step(
    db: Session,
    project_id: Optional[str],
    research_cycle_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build task_context dict for inclusion in step input_json (Prompt 7 Step 2).

    This helper function safely builds a task_context that can be serialized
    and stored in the step's input_json for debugging and context persistence.

    Args:
        db: Database session
        project_id: UUID string of the project
        research_cycle_id: Optional UUID string of the research cycle
        experiment_id: Optional UUID string of the experiment

    Returns:
        Task context dict or None if building fails
    """
    if not project_id:
        return None

    try:
        task_context = build_task_context(
            db=db,
            project_id=project_id,
            research_cycle_id=research_cycle_id,
            experiment_id=experiment_id,
            include_leakage_candidates=True,
            include_past_cycles=True,
            max_experiments=5,
        )
        return task_context
    except Exception as e:
        logger.warning(f"Could not build task_context for step input: {e}")
        return None


def _infer_metric_from_task(task_type: str) -> str:
    """Infer the primary metric from task type.

    Args:
        task_type: The ML task type

    Returns:
        The inferred primary metric name
    """
    metric_map = {
        "regression": "root_mean_squared_error",
        "binary": "roc_auc",
        "multiclass": "accuracy",
        "quantile": "pinball_loss",
    }
    return metric_map.get(task_type, "accuracy")


def _fill_experiment_design_defaults(result: Dict[str, Any], step_logger: Any = None) -> Dict[str, Any]:
    """Fill in missing fields in experiment design result with sensible defaults.

    This handles cases where the LLM omits optional-but-expected fields like
    validation_strategy.reasoning.

    Args:
        result: The raw result dict from LLM
        step_logger: Optional logger for warnings

    Returns:
        The result dict with missing fields filled in
    """
    if not isinstance(result, dict):
        return result

    if "variants" in result and isinstance(result["variants"], list):
        for variant in result["variants"]:
            if not isinstance(variant, dict):
                continue

            # Fill in validation_strategy.reasoning if missing
            if "validation_strategy" in variant and isinstance(variant["validation_strategy"], dict):
                vs = variant["validation_strategy"]
                if "reasoning" not in vs or not vs["reasoning"]:
                    # Generate a default reasoning based on split_strategy
                    split_strategy = vs.get("split_strategy", "random")
                    default_reasoning = f"Using {split_strategy} split to ensure proper evaluation."
                    vs["reasoning"] = default_reasoning
                    if step_logger:
                        step_logger.warning(f"Added default reasoning for validation_strategy in variant '{variant.get('name', 'unknown')}'")

    return result


def _fill_dataset_design_defaults(result: Dict[str, Any], step_logger: Any = None) -> Dict[str, Any]:
    """Fill in missing fields in dataset design result with sensible defaults.

    Args:
        result: The raw result dict from LLM
        step_logger: Optional logger for warnings

    Returns:
        The result dict with missing fields filled in
    """
    if not isinstance(result, dict):
        return result

    if "variants" in result and isinstance(result["variants"], list):
        for variant in result["variants"]:
            if not isinstance(variant, dict):
                continue

            # Ensure required fields have defaults
            if "name" not in variant:
                variant["name"] = f"variant_{result['variants'].index(variant) + 1}"
            if "description" not in variant:
                variant["description"] = f"Dataset variant {variant['name']}"
            if "feature_columns" not in variant:
                variant["feature_columns"] = []
            if "expected_tradeoff" not in variant:
                variant["expected_tradeoff"] = "Trade-off not specified"

            # Fix engineered_features if it contains strings instead of objects
            if "engineered_features" in variant and isinstance(variant["engineered_features"], list):
                fixed_features = []
                for feat in variant["engineered_features"]:
                    if isinstance(feat, str):
                        # LLM returned just a string name - skip it
                        if step_logger:
                            step_logger.warning(f"Skipping invalid engineered_feature '{feat}' (string instead of object)")
                        continue
                    elif isinstance(feat, dict):
                        # Ensure required fields exist
                        if "output_column" in feat and "formula" in feat:
                            if "source_columns" not in feat:
                                feat["source_columns"] = []
                            if "description" not in feat:
                                feat["description"] = f"Engineered feature: {feat['output_column']}"
                            fixed_features.append(feat)
                variant["engineered_features"] = fixed_features

    return result


def _build_revision_feedback_message(
    issues: List[str],
    revision_feedback: List[Dict[str, Any]],
    revision_count: int,
) -> str:
    """Build a feedback message for the Experiment Planner to address Critic concerns.

    Args:
        issues: List of issue descriptions from the Critic
        revision_feedback: Detailed feedback items with specific suggestions
        revision_count: Current revision attempt number

    Returns:
        A formatted message explaining what needs to be revised
    """
    lines = [
        f"## 🔄 Plan Revision Request (Attempt {revision_count})",
        "",
        "The Plan Critic has reviewed your experiment design and identified concerns that must be addressed.",
        "",
        "### Issues to Address:",
    ]

    for issue in issues:
        lines.append(f"- {issue}")

    if revision_feedback:
        lines.append("")
        lines.append("### Specific Feedback:")
        for fb in revision_feedback:
            variant_name = fb.get("variant", "Unknown")
            issue_type = fb.get("issue", "")
            lines.append(f"\n**Variant: {variant_name}**")

            if issue_type == "random_split_on_time_data":
                lines.append(
                    "  - You proposed a random/stratified split for time-based data without justification."
                )
                lines.append(
                    "  - This typically causes data leakage in time-series problems."
                )
                lines.append(
                    "  - **Action Required**: Either (1) switch to 'time' or 'group_time' split, OR "
                    "(2) provide a detailed, convincing explanation in `validation_strategy.reasoning` "
                    "for why random split is appropriate in this specific case."
                )
            elif issue_type == "weak_justification":
                eval_explanation = fb.get("evaluation", "")
                suggested_action = fb.get("suggested_action", "")
                lines.append(f"  - Your justification was evaluated as WEAK: {eval_explanation}")
                if suggested_action:
                    lines.append(f"  - **Suggested Action**: {suggested_action}")
            elif issue_type == "invalid_justification":
                eval_explanation = fb.get("evaluation", "")
                suggested_action = fb.get("suggested_action", "")
                lines.append(f"  - Your justification was evaluated as INVALID: {eval_explanation}")
                if suggested_action:
                    lines.append(f"  - **Required Action**: {suggested_action}")
            else:
                # Generic feedback
                lines.append(f"  - Issue: {fb.get('issue', 'See above')}")
                if fb.get("suggested_action"):
                    lines.append(f"  - Suggestion: {fb['suggested_action']}")

    lines.extend([
        "",
        "### How to Respond:",
        "1. Review the feedback above carefully.",
        "2. Either modify your experiment design to address the concerns, OR",
        "3. Provide a strong, detailed justification in `validation_strategy.reasoning` explaining "
        "why your original approach is correct despite the Critic's concerns.",
        "",
        "**Note**: Generic justifications like 'standard practice' or 'commonly used' will be rejected. "
        "You must explain *specifically* why data leakage is not a concern for THIS dataset and task.",
    ])

    return "\n".join(lines)


def safe_get(data: Any, key: str, default: T = None, expected_type: type = None) -> T:
    """Safely get a value from a dict, handling cases where data might not be a dict.

    This is useful when parsing LLM responses that might return strings instead of
    proper nested objects.

    Args:
        data: The data to extract from (should be a dict, but handles other types)
        key: The key to look up
        default: Default value if key not found or data is not a dict
        expected_type: Optional type to validate against. If provided and value doesn't
                      match, returns default instead.

    Returns:
        The value at key if data is a dict and key exists, otherwise default
    """
    if not isinstance(data, dict):
        return default

    value = data.get(key, default)

    # If expected_type is provided, validate the value's type
    if expected_type is not None and value is not None:
        if not isinstance(value, expected_type):
            logger.warning(f"Expected {expected_type.__name__} for key '{key}', got {type(value).__name__}")
            return default

    return value


class StepLogger:
    """Helper class to append logs to an agent step."""

    def __init__(self, db: Session, step_id: UUID):
        self.db = db
        self.step_id = step_id
        self._sequence = 0

    def _get_next_sequence(self) -> int:
        """Get the next sequence number for this step."""
        # Get max sequence from existing logs
        max_seq = (
            self.db.query(AgentStepLog)
            .filter(AgentStepLog.agent_step_id == self.step_id)
            .count()
        )
        self._sequence = max_seq + 1
        return self._sequence

    def log(
        self,
        message: str,
        message_type: LogMessageType = LogMessageType.INFO,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentStepLog:
        """Append a log entry to the step."""
        log_entry = AgentStepLog(
            agent_step_id=self.step_id,
            sequence=self._get_next_sequence(),
            timestamp=datetime.utcnow(),
            message_type=message_type,
            message=message,
            metadata_json=metadata,
        )
        self.db.add(log_entry)
        self.db.commit()
        return log_entry

    def info(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> AgentStepLog:
        """Log an info message."""
        return self.log(message, LogMessageType.INFO, metadata)

    def warning(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> AgentStepLog:
        """Log a warning message."""
        return self.log(message, LogMessageType.WARNING, metadata)

    def error(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> AgentStepLog:
        """Log an error message."""
        return self.log(message, LogMessageType.ERROR, metadata)

    def thought(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> AgentStepLog:
        """Log a thought/reasoning message (deprecated, use thinking())."""
        return self.log(message, LogMessageType.THOUGHT, metadata)

    def summary(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> AgentStepLog:
        """Log a summary message - final narrative summary for the step."""
        return self.log(message, LogMessageType.SUMMARY, metadata)

    def thinking(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> AgentStepLog:
        """Log a thinking message - internal reasoning, step-by-step analysis."""
        return self.log(message, LogMessageType.THINKING, metadata)

    def hypothesis(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> AgentStepLog:
        """Log a hypothesis message - candidate explanations or theories."""
        return self.log(message, LogMessageType.HYPOTHESIS, metadata)

    def action(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> AgentStepLog:
        """Log an action message - specific actions/commands the agent is taking."""
        return self.log(message, LogMessageType.ACTION, metadata)


def append_step_log(
    db: Session,
    step_id: UUID,
    message_type: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentStepLog:
    """Convenience function to append a log entry to a step.

    Args:
        db: Database session
        step_id: UUID of the agent step
        message_type: Type of message (info, warning, error, thought, summary, thinking, hypothesis, action)
        message: The log message content
        metadata: Optional structured metadata

    Returns:
        The created AgentStepLog entry
    """
    step_logger = StepLogger(db, step_id)
    msg_type = LogMessageType(message_type) if isinstance(message_type, str) else message_type
    return step_logger.log(message, msg_type, metadata)


# ============================================
# Typed Step Logging Helper Functions
# ============================================

def log_step_thinking(
    db: Session,
    step_id: UUID,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentStepLog:
    """Log internal reasoning or step-by-step analysis.

    Use this for intermediate thoughts, deliberations, and analysis
    that shows how the agent is working through a problem.
    """
    return StepLogger(db, step_id).thinking(message, metadata)


def log_step_hypothesis(
    db: Session,
    step_id: UUID,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentStepLog:
    """Log a candidate explanation or theory.

    Use this when the agent proposes potential solutions,
    theories, or explanations that it will evaluate.
    """
    return StepLogger(db, step_id).hypothesis(message, metadata)


def log_step_action(
    db: Session,
    step_id: UUID,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentStepLog:
    """Log a specific action or command the agent is taking.

    Use this to announce what the agent is about to do,
    e.g., "Now I will design features X, Y, Z".
    """
    return StepLogger(db, step_id).action(message, metadata)


def log_step_summary(
    db: Session,
    step_id: UUID,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentStepLog:
    """Log a final narrative summary for the step.

    Use this at the end of a step to summarize what was accomplished.
    """
    return StepLogger(db, step_id).summary(message, metadata)


def log_step_warning(
    db: Session,
    step_id: UUID,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentStepLog:
    """Log a potential issue or warning.

    Use this for concerning situations that don't block progress
    but should be noted.
    """
    return StepLogger(db, step_id).warning(message, metadata)


def log_step_error(
    db: Session,
    step_id: UUID,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentStepLog:
    """Log an error that occurred during step execution.

    Use this when something went wrong during the step.
    """
    return StepLogger(db, step_id).error(message, metadata)


def log_step_info(
    db: Session,
    step_id: UUID,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentStepLog:
    """Log general informational message.

    Use this for neutral status updates and informational messages.
    """
    return StepLogger(db, step_id).info(message, metadata)


# ============================================
# Project History Context Builder
# ============================================

def build_agent_context_for_project(
    db: Session,
    project_id: UUID,
    research_cycle_id: Optional[UUID] = None,
    max_cycles: int = 3,
    max_notebook_entries: int = 5,
    max_experiments: int = 5,
) -> Dict[str, Any]:
    """Build a summarized context of project history for agent planning steps.

    This function gathers relevant historical information from the project to help
    agents make better decisions by learning from past experiments, critiques,
    robustness audits, and lab notebook entries.

    Args:
        db: Database session
        project_id: UUID of the project
        research_cycle_id: Optional UUID of current research cycle (excludes it from history)
        max_cycles: Maximum number of recent cycles to include (default 3)
        max_notebook_entries: Maximum notebook entries to include (default 5)
        max_experiments: Maximum experiments per cycle to include (default 5)

    Returns:
        Dict containing summarized project history:
        - problem_description: Project problem description
        - task_type: ML task type
        - primary_metric: Primary optimization metric
        - research_cycles: List of recent cycle summaries
        - best_models: List of best performing models with metrics
        - robustness_findings: Latest robustness audit findings
        - notebook_highlights: Recent lab notebook entry summaries
        - dataset_specs: Recent dataset configurations
    """
    context: Dict[str, Any] = {
        "problem_description": None,
        "task_type": None,
        "primary_metric": None,
        "research_cycles": [],
        "best_models": [],
        "robustness_findings": None,
        "notebook_highlights": [],
        "dataset_specs": [],
    }

    # Load project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return context

    context["problem_description"] = project.description or "No description provided"
    context["task_type"] = project.task_type
    # Infer primary_metric from task_type since Project doesn't have primary_metric field
    task_type_str = project.task_type.value if hasattr(project.task_type, 'value') else (project.task_type or "classification")
    context["primary_metric"] = _infer_metric_from_task(task_type_str)

    # Get recent research cycles (excluding current if provided)
    cycles_query = (
        db.query(ResearchCycle)
        .filter(ResearchCycle.project_id == project_id)
        .order_by(ResearchCycle.sequence_number.desc())
    )
    if research_cycle_id:
        cycles_query = cycles_query.filter(ResearchCycle.id != research_cycle_id)

    recent_cycles = cycles_query.limit(max_cycles).all()

    # Build cycle summaries
    for cycle in recent_cycles:
        cycle_summary = {
            "sequence_number": cycle.sequence_number,
            "status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
            "title": cycle.summary_title or f"Cycle {cycle.sequence_number}",
            "experiments": [],
            "key_findings": [],
        }

        # Get experiments for this cycle
        cycle_experiments = (
            db.query(CycleExperiment)
            .filter(CycleExperiment.research_cycle_id == cycle.id)
            .all()
        )
        exp_ids = [ce.experiment_id for ce in cycle_experiments]

        if exp_ids:
            experiments = (
                db.query(Experiment)
                .filter(Experiment.id.in_(exp_ids))
                .order_by(Experiment.created_at.desc())
                .limit(max_experiments)
                .all()
            )

            for exp in experiments:
                exp_summary = {
                    "name": exp.name,
                    "status": exp.status.value if hasattr(exp.status, 'value') else str(exp.status),
                    "primary_metric": exp.primary_metric,
                    "best_metric_value": None,
                    "trial_count": len(exp.trials) if exp.trials else 0,
                }

                # Get best metric from trials
                if exp.trials:
                    for trial in exp.trials:
                        if trial.metrics_json and exp.primary_metric:
                            metric_val = trial.metrics_json.get(exp.primary_metric)
                            if metric_val is not None:
                                if exp_summary["best_metric_value"] is None:
                                    exp_summary["best_metric_value"] = metric_val
                                elif exp.metric_direction == "maximize":
                                    exp_summary["best_metric_value"] = max(
                                        exp_summary["best_metric_value"], metric_val
                                    )
                                else:
                                    exp_summary["best_metric_value"] = min(
                                        exp_summary["best_metric_value"], metric_val
                                    )

                cycle_summary["experiments"].append(exp_summary)

        # Get lab notebook entries for this cycle (for key findings)
        cycle_entries = (
            db.query(LabNotebookEntry)
            .filter(LabNotebookEntry.research_cycle_id == cycle.id)
            .order_by(LabNotebookEntry.created_at.desc())
            .limit(3)
            .all()
        )
        for entry in cycle_entries:
            # Truncate body to first 200 chars for summary
            body_snippet = (entry.body_markdown or "")[:200]
            if len(entry.body_markdown or "") > 200:
                body_snippet += "..."
            cycle_summary["key_findings"].append({
                "title": entry.title,
                "author": entry.author_type.value if hasattr(entry.author_type, 'value') else str(entry.author_type),
                "snippet": body_snippet,
            })

        context["research_cycles"].append(cycle_summary)

    # Get best models across all experiments
    all_experiments = (
        db.query(Experiment)
        .filter(Experiment.project_id == project_id)
        .filter(Experiment.status == "completed")
        .order_by(Experiment.created_at.desc())
        .limit(10)
        .all()
    )

    best_models = []
    for exp in all_experiments:
        if exp.trials:
            for trial in exp.trials:
                if trial.status.value == "completed" and trial.metrics_json:
                    metric_val = trial.metrics_json.get(exp.primary_metric or context["primary_metric"])
                    if metric_val is not None:
                        best_models.append({
                            "experiment": exp.name,
                            "trial": trial.variant_name,
                            "metric": exp.primary_metric or context["primary_metric"],
                            "value": metric_val,
                            "data_split": trial.data_split_strategy,
                        })

    # Sort by metric value (assume higher is better for now)
    best_models.sort(key=lambda x: x["value"], reverse=True)
    context["best_models"] = best_models[:5]

    # Get latest robustness audit findings
    robustness_step = (
        db.query(AgentStep)
        .join(AgentRun, AgentStep.agent_run_id == AgentRun.id)
        .filter(AgentRun.project_id == project_id)
        .filter(AgentStep.step_type == AgentStepType.ROBUSTNESS_AUDIT)
        .filter(AgentStep.status == AgentStepStatus.COMPLETED)
        .order_by(AgentStep.finished_at.desc())
        .first()
    )

    if robustness_step and robustness_step.output_json:
        audit_output = robustness_step.output_json.get("robustness_audit", {})
        context["robustness_findings"] = {
            "overfitting_risk": audit_output.get("overfitting_risk", "unknown"),
            "suspicious_patterns": [
                p.get("description", "") for p in audit_output.get("suspicious_patterns", [])[:3]
            ],
            "recommendations": audit_output.get("recommendations", [])[:3],
            "summary": audit_output.get("natural_language_summary", ""),
        }

    # Get recent lab notebook highlights (project-wide, most recent)
    notebook_entries = (
        db.query(LabNotebookEntry)
        .filter(LabNotebookEntry.project_id == project_id)
        .order_by(LabNotebookEntry.created_at.desc())
        .limit(max_notebook_entries)
        .all()
    )

    for entry in notebook_entries:
        body_snippet = (entry.body_markdown or "")[:300]
        if len(entry.body_markdown or "") > 300:
            body_snippet += "..."
        context["notebook_highlights"].append({
            "title": entry.title,
            "author": entry.author_type.value if hasattr(entry.author_type, 'value') else str(entry.author_type),
            "cycle": entry.research_cycle.sequence_number if entry.research_cycle else None,
            "snippet": body_snippet,
        })

    # Get recent dataset specs
    dataset_specs = (
        db.query(DatasetSpec)
        .filter(DatasetSpec.project_id == project_id)
        .order_by(DatasetSpec.created_at.desc())
        .limit(3)
        .all()
    )

    for spec in dataset_specs:
        spec_summary = {
            "name": spec.name,
            "target_column": spec.target_column,
            "feature_count": len(spec.feature_columns) if spec.feature_columns else 0,
            "description": (spec.description or "")[:150],
        }
        context["dataset_specs"].append(spec_summary)

    return context


def format_project_context_for_prompt(context: Dict[str, Any]) -> str:
    """Format project context into a readable string for LLM prompts.

    Args:
        context: Context dict from build_agent_context_for_project

    Returns:
        Formatted string suitable for inclusion in prompts
    """
    lines = []

    # Problem description
    lines.append("## Project History & Context")
    lines.append("")
    lines.append(f"**Problem**: {context.get('problem_description', 'Not specified')}")
    lines.append(f"**Task Type**: {context.get('task_type', 'Not specified')}")
    lines.append(f"**Primary Metric**: {context.get('primary_metric', 'Not specified')}")
    lines.append("")

    # Research cycles
    cycles = context.get("research_cycles", [])
    if cycles:
        lines.append("### Previous Research Cycles")
        for cycle in cycles:
            lines.append(f"\n**Cycle {cycle['sequence_number']}**: {cycle['title']} (Status: {cycle['status']})")

            # Experiments in this cycle
            if cycle.get("experiments"):
                for exp in cycle["experiments"]:
                    metric_str = f"{exp['best_metric_value']:.4f}" if exp.get("best_metric_value") is not None else "N/A"
                    lines.append(f"  - {exp['name']}: {exp['primary_metric']}={metric_str} ({exp['trial_count']} trials)")

            # Key findings from notebook
            if cycle.get("key_findings"):
                lines.append("  Key findings:")
                for finding in cycle["key_findings"]:
                    lines.append(f"    - [{finding['author']}] {finding['title']}")
        lines.append("")

    # Best models
    best_models = context.get("best_models", [])
    if best_models:
        lines.append("### Best Performing Models")
        for model in best_models[:3]:
            lines.append(f"  - {model['experiment']} / {model['trial']}: {model['metric']}={model['value']:.4f}")
        lines.append("")

    # Robustness findings
    robustness = context.get("robustness_findings")
    if robustness:
        lines.append("### Latest Robustness Audit")
        risk_emoji = {"low": "✅", "medium": "⚠️", "high": "🚨"}.get(robustness["overfitting_risk"], "❓")
        lines.append(f"**Risk Level**: {risk_emoji} {robustness['overfitting_risk'].upper()}")

        if robustness.get("summary"):
            lines.append(f"**Summary**: {robustness['summary']}")

        if robustness.get("suspicious_patterns"):
            lines.append("**Issues Found**:")
            for pattern in robustness["suspicious_patterns"]:
                lines.append(f"  - {pattern}")

        if robustness.get("recommendations"):
            lines.append("**Recommendations**:")
            for rec in robustness["recommendations"]:
                lines.append(f"  - {rec}")
        lines.append("")

    # Dataset specs
    specs = context.get("dataset_specs", [])
    if specs:
        lines.append("### Previous Dataset Configurations")
        for spec in specs:
            lines.append(f"  - {spec['name']}: target='{spec['target_column']}', {spec['feature_count']} features")
        lines.append("")

    # Notebook highlights
    highlights = context.get("notebook_highlights", [])
    if highlights:
        lines.append("### Recent Lab Notebook Entries")
        for entry in highlights[:3]:
            cycle_str = f"Cycle {entry['cycle']}" if entry.get("cycle") else "General"
            lines.append(f"  - [{entry['author']}] ({cycle_str}) {entry['title']}")
        lines.append("")

    if not any([cycles, best_models, robustness, specs, highlights]):
        lines.append("*No previous history available - this is a new project.*")
        lines.append("")

    return "\n".join(lines)


# ============================================
# LEGACY Step Handler Functions - BACKUP ONLY
# ============================================
# WARNING: These handler functions are NO LONGER CALLED.
# They have been replaced by agent classes in app/services/agents/
# Kept here as reference/backup. To restore, uncomment STEP_HANDLERS
# registry above and add fallback code in run_agent_step.
# ============================================

async def handle_data_analysis_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the data analysis step - the first interactive step in the pipeline.

    This step analyzes the uploaded data and provides:
    - Assessment of data quality and suitability for the ML task
    - Recommendations for data preparation
    - Option to search for additional datasets if the current data is insufficient

    Input JSON should contain:
    - description: User's goal description
    - data_source_id: UUID of the data source
    - schema_summary: Pre-built schema summary (optional, will be built if missing)

    Returns:
        Dict with data_assessment, suitability_score, recommendations, issues,
        can_proceed (bool), suggest_more_data (bool), natural_language_summary
    """
    from pydantic import BaseModel, Field
    from typing import List as ListType

    input_data = step.input_json or {}

    description = input_data.get("description", "")
    if not description:
        raise ValueError("Missing 'description' in input_json")

    step_logger.action(f"Analyzing data for: {description[:100]}...")

    # Get or build schema summary
    schema_data = input_data.get("schema_summary")
    if schema_data:
        schema_summary = SchemaSummary(**schema_data)
        step_logger.thinking(f"Using provided schema: {schema_summary.data_source_name}")
    else:
        # Need to load from data source
        data_source_id = input_data.get("data_source_id")
        if not data_source_id:
            raise ValueError("Missing 'data_source_id' or 'schema_summary' in input_json")

        step_logger.action("Loading data source schema...")
        data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if not data_source:
            raise ValueError(f"Data source not found: {data_source_id}")

        if not data_source.schema_summary:
            raise ValueError(f"Data source {data_source_id} has no schema analysis")

        schema_summary = build_schema_summary(
            data_source_id=str(data_source.id),
            data_source_name=data_source.name,
            analysis_result=data_source.schema_summary,
        )

    step_logger.thinking(f"Examining dataset: {schema_summary.data_source_name}")
    step_logger.thinking(f"Dataset size: {schema_summary.row_count} rows, {schema_summary.column_count} columns")

    # Analyze data quality issues
    issues = []
    recommendations = []

    # Check for high null percentages
    high_null_cols = []
    for col in schema_summary.columns:
        if col.null_percentage > 30:
            high_null_cols.append({"name": col.name, "null_pct": col.null_percentage})
            issues.append(f"Column '{col.name}' has {col.null_percentage:.1f}% missing values")

    # Check for potential ID columns
    id_cols = []
    for col in schema_summary.columns:
        if col.unique_count == schema_summary.row_count:
            id_cols.append(col.name)

    # Check for low-variance columns
    constant_cols = []
    for col in schema_summary.columns:
        if col.unique_count == 1:
            constant_cols.append(col.name)
            issues.append(f"Column '{col.name}' has only one unique value (constant)")

    # Check dataset size
    if schema_summary.row_count < 100:
        issues.append(f"Dataset is very small ({schema_summary.row_count} rows) - may not be enough for reliable ML")

    if schema_summary.row_count < 1000:
        recommendations.append("Consider collecting more data if possible - larger datasets typically give better results")

    step_logger.thinking(f"Found {len(issues)} potential issues in the data")

    # Define response schema
    class DataAnalysisResponse(BaseModel):
        suitability_score: float = Field(
            description="Score from 0.0 to 1.0 indicating how suitable the data is for the ML task"
        )
        can_proceed: bool = Field(
            description="Whether the data is sufficient to proceed with ML experiments"
        )
        suggest_more_data: bool = Field(
            description="Whether to suggest searching for additional datasets"
        )
        target_column_suggestion: str = Field(
            description="Best guess for the target column based on description and schema"
        )
        task_type_suggestion: str = Field(
            description="Suggested task type: binary_classification, multiclass_classification, or regression"
        )
        key_observations: ListType[str] = Field(
            description="Key observations about the data quality and structure"
        )
        data_preparation_recommendations: ListType[str] = Field(
            description="Specific recommendations for preparing this data for ML"
        )
        limitations: ListType[str] = Field(
            description="Limitations or concerns about using this data"
        )
        natural_language_summary: str = Field(
            description="A comprehensive summary explaining the analysis in plain language for the user"
        )

    # Format schema for prompt
    schema_str = _format_schema_for_prompt(schema_summary)

    # Use centralized prompt from prompts.py
    prompt = get_data_analysis_prompt(
        description=description,
        data_source_name=schema_summary.data_source_name,
        row_count=schema_summary.row_count,
        column_count=schema_summary.column_count,
        schema_str=schema_str,
        issues=issues,
        id_cols=id_cols,
        constant_cols=constant_cols,
        high_null_cols=high_null_cols,
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_DATA_EVALUATOR},
        {"role": "user", "content": prompt},
    ]

    step_logger.action("Consulting AI for comprehensive data analysis...")
    response = await llm_client.chat_json(messages, DataAnalysisResponse)

    suitability_score = response.get("suitability_score", 0.5)
    can_proceed = response.get("can_proceed", True)
    suggest_more_data = response.get("suggest_more_data", False)

    step_logger.hypothesis(f"Suitability assessment: {suitability_score:.0%} suitable for ML task")

    if response.get("target_column_suggestion"):
        step_logger.hypothesis(f"Suggested target column: {response.get('target_column_suggestion')}")

    if response.get("task_type_suggestion"):
        step_logger.hypothesis(f"Suggested task type: {response.get('task_type_suggestion')}")

    if can_proceed:
        step_logger.summary(f"Data analysis complete. Score: {suitability_score:.0%}. Ready to proceed with ML experiments.")
    else:
        step_logger.warning(f"Data analysis complete. Score: {suitability_score:.0%}. Data may need improvement before proceeding.")

    return {
        "suitability_score": suitability_score,
        "can_proceed": can_proceed,
        "suggest_more_data": suggest_more_data,
        "target_column_suggestion": response.get("target_column_suggestion", ""),
        "task_type_suggestion": response.get("task_type_suggestion", ""),
        "key_observations": response.get("key_observations", []),
        "data_preparation_recommendations": response.get("data_preparation_recommendations", []),
        "limitations": response.get("limitations", []),
        "natural_language_summary": response.get("natural_language_summary", ""),
        # Pass through useful data for next steps
        "schema_summary": schema_summary.model_dump(mode="json"),
        "issues": issues,
        "high_null_columns": [c["name"] for c in high_null_cols],
        "potential_id_columns": id_cols,
        "constant_columns": constant_cols,
    }


def _log_context_usage(
    step_logger: StepLogger,
    task_context: Optional[Dict[str, Any]],
    task_hints: Optional[Dict[str, Any]] = None,
    step_name: str = "agent",
) -> Dict[str, Any]:
    """Log which contextual factors an agent is using (Prompt 7 Step 7).

    This helper provides consistent logging across all agents about the
    contextual information they're relying on. It also returns a summary
    dict that can be included in output_json.

    Args:
        step_logger: The step logger for output
        task_context: The TaskContext dict
        task_hints: Optional task type hints
        step_name: Name of the current step for logging

    Returns:
        Dict summarizing which context factors were used
    """
    context_factors_used = {
        "is_time_based": False,
        "has_baselines": False,
        "has_label_shuffle": False,
        "has_robustness_info": False,
        "has_leakage_candidates": False,
        "has_expected_metric_range": False,
        "factors_summary": [],
    }

    if not task_context:
        step_logger.thought(f"[{step_name}] No TaskContext available - using defaults")
        return context_factors_used

    # Check is_time_based
    dataset_spec = task_context.get("dataset_spec") or {}
    is_time_based = dataset_spec.get("is_time_based", False)
    time_column = dataset_spec.get("time_column")

    if task_hints and task_hints.get("is_time_based"):
        is_time_based = True

    if is_time_based:
        context_factors_used["is_time_based"] = True
        context_factors_used["time_column"] = time_column
        factor_msg = f"📅 Time-based task (time_column: {time_column or 'unspecified'})"
        context_factors_used["factors_summary"].append(factor_msg)
        step_logger.info(f"[{step_name}] {factor_msg}")

    # Check baselines
    baselines = task_context.get("baselines", {})
    if baselines.get("available"):
        context_factors_used["has_baselines"] = True
        baseline_types = []
        if baselines.get("majority_class"):
            baseline_types.append("majority_class")
        if baselines.get("simple_model"):
            baseline_types.append("simple_model")
        if baselines.get("mean_predictor"):
            baseline_types.append("mean_predictor")
        if baselines.get("regression_baseline"):
            baseline_types.append("regression_baseline")

        factor_msg = f"📊 Using baseline metrics: {', '.join(baseline_types)}"
        context_factors_used["factors_summary"].append(factor_msg)
        context_factors_used["baseline_types"] = baseline_types
        step_logger.info(f"[{step_name}] {factor_msg}")

    # Check label shuffle results
    label_shuffle = task_context.get("label_shuffle", {})
    if label_shuffle.get("available"):
        context_factors_used["has_label_shuffle"] = True
        leakage_detected = label_shuffle.get("leakage_detected", False)
        factor_msg = f"🔀 Label shuffle test available (leakage_detected: {leakage_detected})"
        context_factors_used["factors_summary"].append(factor_msg)
        context_factors_used["label_shuffle_leakage"] = leakage_detected
        if leakage_detected:
            step_logger.warning(f"[{step_name}] {factor_msg}")
        else:
            step_logger.info(f"[{step_name}] {factor_msg}")

    # Check robustness info
    robustness = task_context.get("robustness")
    if robustness:
        context_factors_used["has_robustness_info"] = True
        overfitting_risk = robustness.get("overfitting_risk", "unknown")
        leakage_suspected = robustness.get("leakage_suspected", False)
        risk_level = robustness.get("risk_level", "unknown")

        factor_msg = f"🛡️ Robustness info: overfitting_risk={overfitting_risk}, risk_level={risk_level}"
        context_factors_used["factors_summary"].append(factor_msg)
        context_factors_used["overfitting_risk"] = overfitting_risk
        context_factors_used["risk_level"] = risk_level

        if overfitting_risk == "high" or leakage_suspected or risk_level in ("high", "critical"):
            step_logger.warning(f"[{step_name}] {factor_msg}")
        else:
            step_logger.info(f"[{step_name}] {factor_msg}")

        if leakage_suspected:
            step_logger.warning(f"[{step_name}] ⚠️ Leakage suspected from prior robustness analysis")
            context_factors_used["leakage_suspected"] = True

    # Check leakage candidates
    leakage_candidates = task_context.get("leakage_candidates", [])
    if leakage_candidates:
        context_factors_used["has_leakage_candidates"] = True
        high_count = len([lc for lc in leakage_candidates if lc.get("severity") == "high"])
        medium_count = len([lc for lc in leakage_candidates if lc.get("severity") == "medium"])
        total_count = len(leakage_candidates)

        factor_msg = f"🔓 Leakage candidates: {total_count} total ({high_count} high, {medium_count} medium)"
        context_factors_used["factors_summary"].append(factor_msg)
        context_factors_used["leakage_high_count"] = high_count
        context_factors_used["leakage_total_count"] = total_count

        if high_count > 0:
            step_logger.warning(f"[{step_name}] {factor_msg}")
        else:
            step_logger.info(f"[{step_name}] {factor_msg}")

    # Check expected metric range (may come from context_analysis)
    # This is typically set by Problem Framer and passed through Experiment Planner
    if task_hints and task_hints.get("expected_metric_range"):
        context_factors_used["has_expected_metric_range"] = True
        expected_range = task_hints["expected_metric_range"]
        metric = expected_range.get("metric", "primary")
        lower = expected_range.get("lower_bound")
        upper = expected_range.get("upper_bound")
        if lower is not None and upper is not None:
            factor_msg = f"📈 Expected {metric} range: [{lower:.3f} - {upper:.3f}]"
            context_factors_used["factors_summary"].append(factor_msg)
            step_logger.info(f"[{step_name}] {factor_msg}")

    # Log summary
    factor_count = len(context_factors_used["factors_summary"])
    if factor_count > 0:
        step_logger.thought(f"[{step_name}] Using {factor_count} contextual factor(s) from TaskContext")
    else:
        step_logger.thought(f"[{step_name}] TaskContext available but no specific factors applied")

    return context_factors_used


def _calculate_expected_metric_range(
    task_type: str,
    primary_metric: str,
    task_context: Optional[Dict[str, Any]] = None,
    class_balance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate expected metric range based on task type and baseline info (Prompt 7 Step 3).

    This function uses available baseline metrics, class balance, and robustness info
    to generate realistic expectations for model performance.

    Args:
        task_type: The ML task type (binary, multiclass, regression)
        primary_metric: The primary evaluation metric
        task_context: Optional task context with baselines, robustness info
        class_balance: Optional class distribution info

    Returns:
        Dict with expected_metric_range containing lower_bound, upper_bound, reasoning
    """
    result: Dict[str, Any] = {
        "primary_metric": {
            "lower_bound": None,
            "upper_bound": None,
            "reasoning": "No baseline information available; using default expectations.",
        }
    }

    # Extract baseline info from task_context
    baselines = {}
    robustness = {}
    if task_context:
        baselines = task_context.get("baselines", {})
        robustness = task_context.get("robustness", {}) or {}

    # Classification metrics (AUC, accuracy, f1, etc.)
    if task_type in ("binary", "multiclass"):
        # Get baseline performance values
        majority_acc = None
        simple_model_metric = None

        if baselines.get("available"):
            # Get majority class baseline
            majority_class = baselines.get("majority_class", {})
            majority_acc = majority_class.get("accuracy")

            # Get simple model baseline (logistic for classification)
            simple_model = baselines.get("simple_model", {})
            if primary_metric in ("roc_auc", "auc"):
                simple_model_metric = simple_model.get("roc_auc")
            elif primary_metric == "accuracy":
                simple_model_metric = simple_model.get("accuracy")
            elif primary_metric in ("f1", "f1_score"):
                simple_model_metric = simple_model.get("f1")

        # Calculate expected range based on available info
        reasoning_parts = []

        if primary_metric in ("roc_auc", "auc"):
            # AUC-based expectations
            if simple_model_metric is not None:
                # Expect 5-15 points above simple model
                lower = min(simple_model_metric + 0.03, 0.95)
                upper = min(simple_model_metric + 0.15, 0.95)
                result["primary_metric"]["lower_bound"] = round(lower, 3)
                result["primary_metric"]["upper_bound"] = round(upper, 3)
                reasoning_parts.append(f"Simple model AUC = {simple_model_metric:.3f}")
                reasoning_parts.append(f"Expecting improvement of 0.03-0.15 above baseline")
            else:
                # Default expectations for AUC
                result["primary_metric"]["lower_bound"] = 0.55
                result["primary_metric"]["upper_bound"] = 0.75
                reasoning_parts.append("No baseline AUC available; using conservative defaults")

            # Adjust for class imbalance
            if class_balance:
                minority_pct = None
                if isinstance(class_balance, dict):
                    values = list(class_balance.values())
                    if len(values) >= 2:
                        total = sum(v for v in values if isinstance(v, (int, float)))
                        if total > 0:
                            minority_pct = min(values) / total if all(isinstance(v, (int, float)) for v in values) else None

                if minority_pct is not None and minority_pct < 0.1:
                    # Highly imbalanced - be more conservative
                    result["primary_metric"]["upper_bound"] = min(
                        result["primary_metric"]["upper_bound"] or 0.75,
                        0.80
                    )
                    reasoning_parts.append(f"Class imbalance detected (minority ~{minority_pct:.1%})")

            # AUC > 0.85 is suspicious for typical business problems
            if (result["primary_metric"]["upper_bound"] or 0) > 0.85:
                result["primary_metric"]["upper_bound"] = 0.85
                reasoning_parts.append("AUC > 0.85 is often suspicious; capping at 0.85")

        elif primary_metric == "accuracy":
            # Accuracy-based expectations
            if majority_acc is not None:
                lower = majority_acc + 0.03
                upper = min(majority_acc + 0.20, 0.95)
                result["primary_metric"]["lower_bound"] = round(lower, 3)
                result["primary_metric"]["upper_bound"] = round(upper, 3)
                reasoning_parts.append(f"Majority-class accuracy = {majority_acc:.3f}")
                reasoning_parts.append(f"Expecting improvement of 3-20 points above baseline")
            else:
                result["primary_metric"]["lower_bound"] = 0.55
                result["primary_metric"]["upper_bound"] = 0.85
                reasoning_parts.append("No baseline accuracy available; using conservative defaults")

        else:
            # Other classification metrics (f1, precision, recall)
            if simple_model_metric is not None:
                lower = simple_model_metric + 0.03
                upper = min(simple_model_metric + 0.20, 0.95)
                result["primary_metric"]["lower_bound"] = round(lower, 3)
                result["primary_metric"]["upper_bound"] = round(upper, 3)
                reasoning_parts.append(f"Simple model {primary_metric} = {simple_model_metric:.3f}")
            else:
                result["primary_metric"]["lower_bound"] = 0.50
                result["primary_metric"]["upper_bound"] = 0.80
                reasoning_parts.append(f"No baseline {primary_metric} available; using conservative defaults")

        result["primary_metric"]["reasoning"] = "; ".join(reasoning_parts) if reasoning_parts else result["primary_metric"]["reasoning"]

    elif task_type == "regression":
        # Regression metrics (RMSE, MAE, R2)
        mean_predictor = baselines.get("mean_predictor", {}) if baselines.get("available") else {}
        regression_baseline = baselines.get("regression_baseline", {}) if baselines.get("available") else {}

        reasoning_parts = []

        if primary_metric in ("rmse", "mse", "root_mean_squared_error"):
            baseline_rmse = mean_predictor.get("rmse") or regression_baseline.get("rmse")
            if baseline_rmse is not None:
                # Expect 10-40% improvement over baseline
                upper = baseline_rmse * 0.90  # At least 10% better
                lower = baseline_rmse * 0.60  # Up to 40% better
                result["primary_metric"]["lower_bound"] = round(lower, 4)
                result["primary_metric"]["upper_bound"] = round(upper, 4)
                reasoning_parts.append(f"Baseline RMSE = {baseline_rmse:.4f}")
                reasoning_parts.append("Expecting 10-40% improvement over baseline")
            else:
                result["primary_metric"]["reasoning"] = "No baseline RMSE available; targets should be set after seeing initial results"

        elif primary_metric in ("mae", "mean_absolute_error"):
            baseline_mae = mean_predictor.get("mae") or regression_baseline.get("mae")
            if baseline_mae is not None:
                upper = baseline_mae * 0.90
                lower = baseline_mae * 0.60
                result["primary_metric"]["lower_bound"] = round(lower, 4)
                result["primary_metric"]["upper_bound"] = round(upper, 4)
                reasoning_parts.append(f"Baseline MAE = {baseline_mae:.4f}")
                reasoning_parts.append("Expecting 10-40% improvement over baseline")
            else:
                result["primary_metric"]["reasoning"] = "No baseline MAE available; targets should be set after seeing initial results"

        elif primary_metric in ("r2", "r2_score", "r_squared"):
            baseline_r2 = mean_predictor.get("r2") or regression_baseline.get("r2")
            if baseline_r2 is not None:
                # R2 ranges from 0 to 1 (typically), expect improvement
                lower = max(baseline_r2 + 0.10, 0.20)
                upper = min(baseline_r2 + 0.40, 0.90)
                result["primary_metric"]["lower_bound"] = round(lower, 3)
                result["primary_metric"]["upper_bound"] = round(upper, 3)
                reasoning_parts.append(f"Baseline R² = {baseline_r2:.3f}")
                reasoning_parts.append("Expecting 0.10-0.40 improvement over baseline")
            else:
                result["primary_metric"]["lower_bound"] = 0.20
                result["primary_metric"]["upper_bound"] = 0.70
                reasoning_parts.append("No baseline R² available; using conservative defaults")

        if reasoning_parts:
            result["primary_metric"]["reasoning"] = "; ".join(reasoning_parts)

    # Add warning if robustness shows potential issues
    if robustness.get("leakage_suspected"):
        result["primary_metric"]["reasoning"] += " WARNING: Leakage suspected - actual performance may be lower in production."

    return result


async def handle_problem_understanding_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the problem understanding step (Prompt 7 Step 3 - upgraded).

    This step analyzes the user's goal and data schema to determine:
    - Task type (binary, multiclass, regression, etc.)
    - Target column
    - Primary metric
    - Confidence and reasoning
    - Expected metric range (based on baselines and task context)

    Input JSON should contain:
    - description: User's goal description
    - data_source_id: UUID of the data source
    - schema_summary: Pre-built schema summary (optional, will be built if missing)
    - task_context: Optional task context with baselines, robustness info (Prompt 7)

    Returns:
        Dict with task_type, target_column, primary_metric, reasoning, confidence,
        expected_metric_range, and time-based metadata
    """
    input_data = step.input_json or {}

    description = input_data.get("description", "")
    if not description:
        raise ValueError("Missing 'description' in input_json")

    step_logger.info(f"Analyzing problem: {description[:100]}...")

    # Extract task_context if available (Prompt 7 Step 3)
    task_context = input_data.get("task_context")

    # Log context usage with consistent helper (Prompt 7 Step 7)
    context_factors = _log_context_usage(
        step_logger=step_logger,
        task_context=task_context,
        task_hints=None,  # No hints available yet at this step
        step_name="Problem Understanding",
    )

    # Additional logging for data profile (not covered by standard helper)
    if task_context:
        data_profile = task_context.get("data_profile_summary", {}) or {}
        if data_profile.get("row_count"):
            step_logger.thought(f"Data profile: {data_profile['row_count']} rows, {data_profile.get('feature_count', 'unknown')} features")

    # Get or build schema summary
    schema_data = input_data.get("schema_summary")
    if schema_data:
        schema_summary = SchemaSummary(**schema_data)
        step_logger.info(f"Using provided schema: {schema_summary.data_source_name}")
    else:
        # Need to load from data source
        data_source_id = input_data.get("data_source_id")
        if not data_source_id:
            raise ValueError("Missing 'data_source_id' or 'schema_summary' in input_json")

        step_logger.info("Loading data source schema...")
        data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if not data_source:
            raise ValueError(f"Data source not found: {data_source_id}")

        if not data_source.schema_summary:
            raise ValueError(f"Data source {data_source_id} has no schema analysis")

        schema_summary = build_schema_summary(
            data_source_id=str(data_source.id),
            data_source_name=data_source.name,
            analysis_result=data_source.schema_summary,
        )

    step_logger.thought(f"Dataset has {schema_summary.row_count} rows and {schema_summary.column_count} columns")

    # Analyze columns
    numeric_cols = [c.name for c in schema_summary.columns if c.inferred_type == "numeric"]
    categorical_cols = [c.name for c in schema_summary.columns if c.inferred_type == "categorical"]
    step_logger.thought(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")

    # Call LLM to generate config
    step_logger.info("Consulting LLM to determine task configuration...")

    suggestion = await generate_project_config(
        client=llm_client,
        description=description,
        schema_summary=schema_summary,
    )

    step_logger.summary(
        f"Identified {suggestion.task_type} task targeting '{suggestion.target_column}' "
        f"with {suggestion.primary_metric} metric (confidence: {suggestion.confidence:.0%})"
    )

    # Log target creation info if the target needs to be created
    if suggestion.target_creation:
        step_logger.info(
            f"Target column '{suggestion.target_column}' will be created using formula: "
            f"{suggestion.target_creation.formula}"
        )

    # Log suggested feature engineering
    if suggestion.suggested_feature_engineering:
        step_logger.info(
            f"Suggested {len(suggestion.suggested_feature_engineering)} feature engineering steps"
        )

    # Log time-based task detection (Prompt 7 Step 3)
    if suggestion.is_time_based:
        step_logger.info(
            f"Detected time-based task: time_column='{suggestion.time_column}', "
            f"horizon='{suggestion.prediction_horizon}'"
        )
        if suggestion.entity_id_column:
            step_logger.info(f"Entity ID column: '{suggestion.entity_id_column}'")
        if suggestion.target_positive_class:
            step_logger.info(f"Target positive class: '{suggestion.target_positive_class}'")
        step_logger.thought("Time-based task detected - time-aware splits are required to prevent look-ahead bias")
    else:
        step_logger.thought("Task is not time-based; no need for time-aware splits")

    # Calculate expected metric range based on baselines and task context (Prompt 7 Step 3)
    # Extract class balance from schema if available
    class_balance = None
    if schema_summary.columns:
        # Try to get class balance from target column stats
        for col in schema_summary.columns:
            if col.name == suggestion.target_column and hasattr(col, 'value_counts') and col.value_counts:
                class_balance = col.value_counts
                break

    # Also check data_profile_summary from task_context
    if not class_balance and task_context:
        data_profile = task_context.get("data_profile_summary", {}) or {}
        class_balance = data_profile.get("class_balance")

    expected_metric_range = _calculate_expected_metric_range(
        task_type=suggestion.task_type,
        primary_metric=suggestion.primary_metric,
        task_context=task_context,
        class_balance=class_balance,
    )

    # Log the expected metric range
    primary_range = expected_metric_range.get("primary_metric", {})
    if primary_range.get("lower_bound") is not None and primary_range.get("upper_bound") is not None:
        step_logger.thought(
            f"Expected {suggestion.primary_metric} range: "
            f"{primary_range['lower_bound']:.3f} - {primary_range['upper_bound']:.3f}"
        )
        step_logger.thought(f"Reasoning: {primary_range.get('reasoning', 'N/A')}")
    else:
        step_logger.thought(f"Expected metric range: {primary_range.get('reasoning', 'Unable to determine range')}")

    # Add secondary metrics list based on task type
    secondary_metrics = []
    if suggestion.task_type == "binary":
        secondary_metrics = ["accuracy", "f1", "precision", "recall"]
        if suggestion.primary_metric != "roc_auc":
            secondary_metrics.insert(0, "roc_auc")
    elif suggestion.task_type == "multiclass":
        secondary_metrics = ["accuracy", "f1_macro", "f1_weighted"]
    elif suggestion.task_type == "regression":
        secondary_metrics = ["rmse", "mae", "r2"]
        if suggestion.primary_metric in secondary_metrics:
            secondary_metrics.remove(suggestion.primary_metric)

    return {
        "task_type": suggestion.task_type,
        "target_column": suggestion.target_column,
        "target_exists": suggestion.target_exists,
        "target_creation": suggestion.target_creation.model_dump(mode="json") if suggestion.target_creation else None,
        "primary_metric": suggestion.primary_metric,
        "secondary_metrics": secondary_metrics,
        "reasoning": suggestion.reasoning,
        "confidence": suggestion.confidence,
        "suggested_name": suggestion.suggested_name,
        "suggested_feature_engineering": [
            fe.model_dump(mode="json") for fe in suggestion.suggested_feature_engineering
        ],
        "schema_summary": schema_summary.model_dump(mode="json"),
        # Time-based task metadata
        "is_time_based": suggestion.is_time_based,
        "time_column": suggestion.time_column,
        "entity_id_column": suggestion.entity_id_column,
        "prediction_horizon": suggestion.prediction_horizon,
        "target_positive_class": suggestion.target_positive_class,
        # Expected metric range (Prompt 7 Step 3)
        "expected_metric_range": expected_metric_range,
        # Context factors used (Prompt 7 Step 7)
        "context_factors_used": context_factors,
    }


async def handle_data_audit_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the data audit step.

    This step performs a COMPREHENSIVE analysis of data quality and characteristics,
    checking for issues that could impact model training and validity.

    Input JSON should contain:
    - schema_summary: Schema summary from previous step
    - target_column: The target column (optional)

    Returns:
        Dict with data quality metrics, issues, warnings, and recommendations
    """
    input_data = step.input_json or {}

    schema_data = input_data.get("schema_summary")
    if not schema_data:
        raise ValueError("Missing 'schema_summary' in input_json")

    schema_summary = SchemaSummary(**schema_data)
    target_column = input_data.get("target_column")

    step_logger.info(f"🔍 Starting comprehensive data audit for: {schema_summary.data_source_name}")
    step_logger.info(f"   Dataset: {schema_summary.row_count:,} rows × {schema_summary.column_count} columns")

    # Categorize issues by severity
    critical_issues = []  # Must fix before training
    warnings = []  # Should investigate
    info_notes = []  # Good to know
    recommendations = []

    # Track specific problematic columns
    audit_details = {
        "high_null_columns": [],
        "moderate_null_columns": [],
        "potential_id_columns": [],
        "constant_columns": [],
        "low_variance_columns": [],
        "high_cardinality_columns": [],
        "potential_leakage_columns": [],
        "class_imbalance": None,
        "small_dataset_warning": False,
        "duplicate_risk": False,
    }

    # =========================================================================
    # 1. DATASET SIZE CHECK
    # =========================================================================
    step_logger.thinking("Checking dataset size adequacy...")

    if schema_summary.row_count < 100:
        critical_issues.append(f"⚠️ CRITICAL: Very small dataset ({schema_summary.row_count} rows) - insufficient for reliable ML training")
        recommendations.append("Collect more data or use simpler models with strong regularization")
        audit_details["small_dataset_warning"] = True
    elif schema_summary.row_count < 500:
        warnings.append(f"Small dataset ({schema_summary.row_count} rows) - results may have high variance")
        recommendations.append("Consider using cross-validation and simpler models")
        audit_details["small_dataset_warning"] = True
    elif schema_summary.row_count < 1000:
        info_notes.append(f"Moderate dataset size ({schema_summary.row_count} rows) - watch for overfitting")

    step_logger.info(f"   Dataset size: {schema_summary.row_count:,} rows {'⚠️' if audit_details['small_dataset_warning'] else '✓'}")

    # =========================================================================
    # 2. NULL VALUE ANALYSIS (more granular thresholds)
    # =========================================================================
    step_logger.thinking("Analyzing missing values across all columns...")

    for col in schema_summary.columns:
        if col.null_percentage > 50:
            audit_details["high_null_columns"].append(col.name)
            critical_issues.append(f"Column '{col.name}' has {col.null_percentage:.1f}% missing values - consider dropping")
        elif col.null_percentage > 20:
            audit_details["moderate_null_columns"].append(col.name)
            warnings.append(f"Column '{col.name}' has {col.null_percentage:.1f}% missing values")
        elif col.null_percentage > 5:
            info_notes.append(f"Column '{col.name}' has {col.null_percentage:.1f}% missing values")

    if audit_details["high_null_columns"]:
        step_logger.warning(f"   Found {len(audit_details['high_null_columns'])} columns with >50% nulls: {audit_details['high_null_columns']}")
        recommendations.append("Drop columns with >50% missing values or use advanced imputation")
    if audit_details["moderate_null_columns"]:
        step_logger.warning(f"   Found {len(audit_details['moderate_null_columns'])} columns with 20-50% nulls")
        recommendations.append("Consider imputation strategies (mean/median/mode or ML-based)")

    total_null_cols = len(audit_details["high_null_columns"]) + len(audit_details["moderate_null_columns"])
    step_logger.info(f"   Missing value analysis: {total_null_cols} columns with significant nulls {'⚠️' if total_null_cols > 0 else '✓'}")

    # =========================================================================
    # 3. ID COLUMN DETECTION
    # =========================================================================
    step_logger.thinking("Detecting potential ID columns...")

    for col in schema_summary.columns:
        if col.name == target_column:
            continue
        # ID columns: all unique values or names suggesting IDs
        is_all_unique = col.unique_count == schema_summary.row_count
        name_suggests_id = any(pattern in col.name.lower() for pattern in ['_id', 'id_', 'uuid', 'guid', 'key', 'index', 'row_num'])

        if is_all_unique or (name_suggests_id and col.unique_count > schema_summary.row_count * 0.9):
            audit_details["potential_id_columns"].append(col.name)
            warnings.append(f"Column '{col.name}' appears to be an ID column (should be excluded from features)")

    if audit_details["potential_id_columns"]:
        step_logger.warning(f"   Potential ID columns (exclude from features): {audit_details['potential_id_columns']}")
        recommendations.append("Remove ID columns from feature set - they cause overfitting")
    step_logger.info(f"   ID column detection: {len(audit_details['potential_id_columns'])} found {'⚠️' if audit_details['potential_id_columns'] else '✓'}")

    # =========================================================================
    # 4. CONSTANT AND LOW-VARIANCE COLUMNS
    # =========================================================================
    step_logger.thinking("Checking for constant and low-variance columns...")

    for col in schema_summary.columns:
        if col.unique_count == 1:
            audit_details["constant_columns"].append(col.name)
            warnings.append(f"Column '{col.name}' is constant (single value) - provides no information")
        elif col.unique_count == 2 and col.null_percentage > 90:
            # Almost constant (one value + nulls)
            audit_details["low_variance_columns"].append(col.name)
            warnings.append(f"Column '{col.name}' has near-zero variance")
        elif schema_summary.row_count > 100 and col.unique_count < 3:
            audit_details["low_variance_columns"].append(col.name)
            info_notes.append(f"Column '{col.name}' has very low variance ({col.unique_count} unique values)")

    if audit_details["constant_columns"]:
        step_logger.warning(f"   Constant columns (remove): {audit_details['constant_columns']}")
        recommendations.append("Remove constant columns - they provide no predictive value")
    step_logger.info(f"   Variance check: {len(audit_details['constant_columns'])} constant, {len(audit_details['low_variance_columns'])} low-variance {'⚠️' if audit_details['constant_columns'] else '✓'}")

    # =========================================================================
    # 5. HIGH CARDINALITY CATEGORICAL COLUMNS
    # =========================================================================
    step_logger.thinking("Checking categorical column cardinality...")

    for col in schema_summary.columns:
        if col.inferred_type == "categorical" or (col.inferred_type == "text" and col.unique_count < schema_summary.row_count * 0.5):
            cardinality_ratio = col.unique_count / schema_summary.row_count if schema_summary.row_count > 0 else 0

            if col.unique_count > 100 and cardinality_ratio > 0.1:
                audit_details["high_cardinality_columns"].append({
                    "name": col.name,
                    "unique_count": col.unique_count,
                    "cardinality_ratio": cardinality_ratio
                })
                warnings.append(f"Column '{col.name}' has high cardinality ({col.unique_count} categories) - may cause issues")

    if audit_details["high_cardinality_columns"]:
        step_logger.warning(f"   High cardinality columns: {[c['name'] for c in audit_details['high_cardinality_columns']]}")
        recommendations.append("Consider encoding strategies for high-cardinality categoricals (target encoding, frequency encoding)")
    step_logger.info(f"   Cardinality check: {len(audit_details['high_cardinality_columns'])} high-cardinality columns {'⚠️' if audit_details['high_cardinality_columns'] else '✓'}")

    # =========================================================================
    # 6. POTENTIAL DATA LEAKAGE DETECTION
    # =========================================================================
    step_logger.thinking("Scanning for potential data leakage indicators...")

    leakage_patterns = [
        'target', 'label', 'outcome', 'result', 'prediction', 'pred_',
        'future_', 'next_', 'will_', 'actual_', 'true_', 'y_'
    ]

    for col in schema_summary.columns:
        if col.name == target_column:
            continue

        col_lower = col.name.lower()

        # Check for suspicious column names
        for pattern in leakage_patterns:
            if pattern in col_lower:
                audit_details["potential_leakage_columns"].append({
                    "name": col.name,
                    "reason": f"Name contains '{pattern}' - may leak target information"
                })
                critical_issues.append(f"⚠️ POTENTIAL LEAKAGE: Column '{col.name}' may contain target information")
                break

        # Check for suspiciously perfect correlation indicators (if numeric)
        if col.inferred_type == "numeric" and target_column:
            target_col = next((c for c in schema_summary.columns if c.name == target_column), None)
            if target_col and target_col.inferred_type == "numeric":
                # If a column has similar range to target, flag for investigation
                if (col.min is not None and col.max is not None and
                    target_col.min is not None and target_col.max is not None):
                    col_range = col.max - col.min if col.max != col.min else 1
                    target_range = target_col.max - target_col.min if target_col.max != target_col.min else 1
                    if 0.9 < col_range / target_range < 1.1 and col.name not in audit_details["potential_id_columns"]:
                        # Similar range - could be derived from target
                        info_notes.append(f"Column '{col.name}' has similar range to target - verify it's not derived from target")

    if audit_details["potential_leakage_columns"]:
        step_logger.error(f"   ⚠️ POTENTIAL DATA LEAKAGE detected in: {[c['name'] for c in audit_details['potential_leakage_columns']]}")
        recommendations.append("CRITICAL: Investigate potential leakage columns - they can cause unrealistically good results")
    step_logger.info(f"   Leakage scan: {len(audit_details['potential_leakage_columns'])} suspicious columns {'🚨' if audit_details['potential_leakage_columns'] else '✓'}")

    # =========================================================================
    # 6b. ENHANCED LEAKAGE DETECTION (Prompt 6)
    # =========================================================================
    leakage_candidates: List[Dict[str, Any]] = []

    # Try to load actual data for correlation-based detection
    try:
        from app.services.leakage_detector import (
            detect_potential_leakage_features,
            get_leakage_summary,
        )

        # Get time column from input if available
        time_column = input_data.get("time_column")
        is_time_based = input_data.get("is_time_based", False)

        # Get data source to load actual data
        data_source_id = input_data.get("data_source_id")
        project_id = input_data.get("project_id")

        df_for_leakage: Optional[pd.DataFrame] = None

        if data_source_id:
            step_logger.thinking("Loading data for enhanced leakage detection...")
            try:
                data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
                if data_source and data_source.file_path:
                    import os
                    if os.path.exists(data_source.file_path):
                        # Load a sample for analysis
                        df_for_leakage = pd.read_csv(data_source.file_path, nrows=10000)
            except Exception as e:
                step_logger.info(f"   Could not load data for correlation analysis: {e}")

        # If we couldn't get the data from file, try from schema columns
        if df_for_leakage is None:
            # Create a minimal DataFrame from schema info for name-based detection
            column_names = [col.name for col in schema_summary.columns]
            df_for_leakage = pd.DataFrame(columns=column_names)

        # Run enhanced leakage detection
        if target_column and df_for_leakage is not None:
            step_logger.thinking("Running enhanced leakage detection heuristics...")

            leakage_candidates = detect_potential_leakage_features(
                df=df_for_leakage,
                target_column=target_column,
                time_column=time_column,
                correlation_threshold=0.9,
            )

            # Log findings
            if leakage_candidates:
                leakage_summary = get_leakage_summary(leakage_candidates)
                step_logger.warning(f"   🔍 Enhanced leakage detection found {leakage_summary['total_count']} suspicious features:")

                for candidate in leakage_candidates:
                    severity_emoji = "🚨" if candidate["severity"] == "high" else "⚠️" if candidate["severity"] == "medium" else "ℹ️"
                    step_logger.warning(f"      {severity_emoji} {candidate['column']}: {candidate['reason']}")

                    # Add high-severity findings to critical issues
                    if candidate["severity"] == "high":
                        critical_issues.append(
                            f"⚠️ LEAKAGE CANDIDATE: '{candidate['column']}' - {candidate['reason']}"
                        )
                    elif candidate["severity"] == "medium":
                        warnings.append(
                            f"Potential leakage: '{candidate['column']}' - {candidate['reason']}"
                        )

                # Add recommendations based on findings
                if leakage_summary["high_severity_count"] > 0:
                    recommendations.append(
                        f"CRITICAL: Review {leakage_summary['high_severity_count']} high-severity leakage candidates before training"
                    )

                if is_time_based and leakage_summary["total_count"] > 0:
                    recommendations.append(
                        "For time-based predictions, verify all features are computed from past data only (no look-ahead bias)"
                    )
            else:
                step_logger.info("   ✓ Enhanced leakage detection: No suspicious features found")

        # Store in audit details
        audit_details["leakage_candidates"] = leakage_candidates

    except ImportError:
        step_logger.info("   Enhanced leakage detection not available")
    except Exception as e:
        step_logger.warning(f"   Enhanced leakage detection failed: {e}")

    # =========================================================================
    # 7. TARGET COLUMN ANALYSIS
    # =========================================================================
    target_info = None
    target_stats = None

    if target_column:
        step_logger.thinking(f"Analyzing target column '{target_column}'...")
        target_col = next((c for c in schema_summary.columns if c.name == target_column), None)

        if target_col:
            target_info = {
                "name": target_col.name,
                "unique_count": target_col.unique_count,
                "null_percentage": target_col.null_percentage,
                "inferred_type": target_col.inferred_type,
            }

            # Check for target issues
            if target_col.null_percentage > 0:
                critical_issues.append(f"Target column has {target_col.null_percentage:.1f}% null values - must handle before training")
                recommendations.append("Remove rows with null target or investigate data collection issues")

            step_logger.info(f"   Target '{target_column}': {target_col.unique_count} unique values, {target_col.null_percentage:.1f}% nulls")

            # Classification target analysis
            if target_col.inferred_type == "categorical" or (target_col.unique_count and target_col.unique_count <= 20):
                target_stats = {"class_counts": target_col.top_values or {}}

                if target_col.top_values and len(target_col.top_values) >= 2:
                    total = sum(target_col.top_values.values())
                    class_counts = sorted(target_col.top_values.values(), reverse=True)
                    majority_count = class_counts[0]
                    minority_count = class_counts[-1]
                    majority_pct = majority_count / total * 100 if total > 0 else 0
                    imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')

                    target_stats["majority_class_pct"] = majority_pct
                    target_stats["imbalance_ratio"] = imbalance_ratio
                    target_stats["num_classes"] = len(target_col.top_values)

                    step_logger.info(f"   Target distribution: {len(target_col.top_values)} classes, majority class: {majority_pct:.1f}%")

                    # Check for class imbalance
                    if imbalance_ratio > 10:
                        audit_details["class_imbalance"] = {
                            "ratio": imbalance_ratio,
                            "severity": "severe"
                        }
                        critical_issues.append(f"⚠️ SEVERE CLASS IMBALANCE: {imbalance_ratio:.1f}:1 ratio - standard metrics will be misleading")
                        recommendations.append("Use class weights, SMOTE, or evaluation metrics like F1/AUC instead of accuracy")
                        step_logger.error(f"   🚨 Severe class imbalance detected: {imbalance_ratio:.1f}:1 ratio")
                    elif imbalance_ratio > 3:
                        audit_details["class_imbalance"] = {
                            "ratio": imbalance_ratio,
                            "severity": "moderate"
                        }
                        warnings.append(f"Moderate class imbalance ({imbalance_ratio:.1f}:1 ratio) - consider balanced metrics")
                        recommendations.append("Consider using balanced accuracy or F1 score as primary metric")
                        step_logger.warning(f"   ⚠️ Moderate class imbalance: {imbalance_ratio:.1f}:1 ratio")
                    else:
                        step_logger.info(f"   Class balance: Good ({imbalance_ratio:.1f}:1 ratio) ✓")

            # Regression target analysis
            elif target_col.inferred_type == "numeric":
                target_stats = {
                    "min": target_col.min,
                    "max": target_col.max,
                    "mean": target_col.mean,
                    "std": None,
                }

                if target_col.min is not None and target_col.max is not None:
                    col_range = target_col.max - target_col.min
                    target_stats["std"] = col_range / 4.0  # Rough estimate
                    target_stats["baseline_rmse"] = target_stats["std"]

                    step_logger.info(f"   Target range: {target_col.min:.4f} to {target_col.max:.4f}")
                    step_logger.info(f"   Baseline RMSE (predicting mean): ~{target_stats['std']:.4f}")

                    # Check for extreme outliers in target
                    if target_col.mean is not None:
                        mean_to_max = abs(target_col.max - target_col.mean)
                        mean_to_min = abs(target_col.mean - target_col.min)
                        if mean_to_max > 5 * mean_to_min or mean_to_min > 5 * mean_to_max:
                            warnings.append("Target column may have extreme outliers - consider robust scaling or outlier removal")
                            step_logger.warning(f"   ⚠️ Target may have outliers (asymmetric distribution)")
        else:
            critical_issues.append(f"Target column '{target_column}' not found in dataset!")
            step_logger.error(f"   ❌ Target column '{target_column}' not found!")

    # =========================================================================
    # 8. DUPLICATE RISK ASSESSMENT
    # =========================================================================
    step_logger.thinking("Assessing duplicate row risk...")

    # If we have ID columns and they're not unique, there might be duplicates
    non_id_unique_cols = [col for col in schema_summary.columns
                         if col.name not in audit_details["potential_id_columns"]
                         and col.unique_count == schema_summary.row_count]

    if len(non_id_unique_cols) == 0 and schema_summary.row_count > 100:
        # No column has all unique values (besides IDs) - duplicates are possible
        info_notes.append("No unique identifier column found - verify dataset has no duplicate rows")
        audit_details["duplicate_risk"] = True
        step_logger.info(f"   Duplicate risk: Possible (no unique column) - verify manually")
    else:
        step_logger.info(f"   Duplicate risk: Low ✓")

    # =========================================================================
    # 9. GENERATE SUMMARY
    # =========================================================================
    total_issues = len(critical_issues) + len(warnings)

    step_logger.info("")
    step_logger.info("=" * 60)
    step_logger.info("📋 DATA AUDIT SUMMARY")
    step_logger.info("=" * 60)

    if critical_issues:
        step_logger.error(f"🚨 CRITICAL ISSUES ({len(critical_issues)}):")
        for issue in critical_issues:
            step_logger.error(f"   • {issue}")

    if warnings:
        step_logger.warning(f"⚠️ WARNINGS ({len(warnings)}):")
        for warn in warnings[:10]:  # Limit to first 10
            step_logger.warning(f"   • {warn}")
        if len(warnings) > 10:
            step_logger.warning(f"   ... and {len(warnings) - 10} more warnings")

    if info_notes:
        step_logger.info(f"ℹ️ NOTES ({len(info_notes)}):")
        for note in info_notes[:5]:  # Limit to first 5
            step_logger.info(f"   • {note}")

    if recommendations:
        step_logger.info(f"💡 RECOMMENDATIONS ({len(recommendations)}):")
        for rec in recommendations[:8]:  # Limit to first 8
            step_logger.info(f"   • {rec}")

    # Overall assessment
    if critical_issues:
        step_logger.error(f"\n🔴 AUDIT RESULT: {len(critical_issues)} critical issues require attention before training")
    elif warnings:
        step_logger.warning(f"\n🟡 AUDIT RESULT: {len(warnings)} warnings - proceed with caution")
    else:
        step_logger.info(f"\n🟢 AUDIT RESULT: Data looks good for training!")

    step_logger.info("=" * 60)

    return {
        "data_source_name": schema_summary.data_source_name,
        "row_count": schema_summary.row_count,
        "column_count": schema_summary.column_count,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "info_notes": info_notes,
        "recommendations": recommendations,
        "audit_details": audit_details,
        "target_info": target_info,
        "target_stats": target_stats,
        # Prompt 6: Enhanced leakage detection results
        "leakage_candidates": leakage_candidates,
        # Legacy fields for backward compatibility
        "issues": critical_issues + warnings,
        "high_null_columns": audit_details["high_null_columns"],
        "potential_id_columns": audit_details["potential_id_columns"],
        "constant_columns": audit_details["constant_columns"],
    }


async def handle_dataset_design_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the dataset design step.

    This step determines which columns to use as features with multiple variants.
    It uses project history context to learn from previous experiments and findings.

    Input JSON should contain:
    - schema_summary: Schema summary
    - task_type: The ML task type
    - target_column: The target column
    - description: Optional additional context
    - max_variants: Optional maximum variants to generate (default 10)

    Returns:
        Dict with variants (list), recommended_variant, reasoning, warnings
    """
    input_data = step.input_json or {}

    schema_data = input_data.get("schema_summary")
    if not schema_data:
        raise ValueError("Missing 'schema_summary' in input_json")

    task_type = input_data.get("task_type")
    target_column = input_data.get("target_column")
    description = input_data.get("description")
    max_variants = input_data.get("max_variants", 10)

    if not task_type or not target_column:
        raise ValueError("Missing 'task_type' or 'target_column' in input_json")

    schema_summary = SchemaSummary(**schema_data)

    # Extract audit findings from previous steps (if available)
    audit_critical_issues = input_data.get("critical_issues", [])
    audit_warnings = input_data.get("warnings", [])
    audit_details = input_data.get("audit_details", {})
    audit_recommendations = input_data.get("recommendations", [])

    # Build audit context string for the LLM
    audit_context = ""
    if audit_critical_issues or audit_warnings or audit_recommendations:
        audit_context = "\n\n## 🔍 DATA AUDIT FINDINGS (from previous analysis)\n"

        if audit_critical_issues:
            audit_context += "\n### ⚠️ CRITICAL ISSUES - MUST ADDRESS:\n"
            for issue in audit_critical_issues:
                audit_context += f"- {issue}\n"

        if audit_warnings:
            audit_context += "\n### ⚡ WARNINGS - Consider carefully:\n"
            for warning in audit_warnings:
                audit_context += f"- {warning}\n"

        if audit_details:
            if audit_details.get("class_imbalance"):
                imb = audit_details["class_imbalance"]
                audit_context += f"\n### Class Imbalance: {imb.get('severity', 'unknown')} ({imb.get('ratio', 'N/A')}:1 ratio)\n"
            if audit_details.get("leakage_risk"):
                leak = audit_details["leakage_risk"]
                audit_context += f"\n### ⚠️ POTENTIAL DATA LEAKAGE: {', '.join(leak.get('columns', []))}\n"
                audit_context += "These columns may contain future information - consider excluding them.\n"
            if audit_details.get("high_null_columns"):
                audit_context += f"\n### High Null Columns: {', '.join(audit_details['high_null_columns'])}\n"
            if audit_details.get("potential_id_columns"):
                audit_context += f"\n### Likely ID Columns (exclude from features): {', '.join(audit_details['potential_id_columns'])}\n"

        if audit_recommendations:
            audit_context += "\n### 💡 RECOMMENDATIONS:\n"
            for rec in audit_recommendations[:5]:  # Limit to top 5
                audit_context += f"- {rec}\n"

        step_logger.info("📋 Including Data Audit findings in dataset design context")

    step_logger.info(f"Designing dataset variants for {task_type} task targeting '{target_column}'")
    step_logger.thinking(f"Analyzing {schema_summary.column_count} columns to generate up to {max_variants} variants...")

    # Get project ID and research cycle ID for tool executor
    project_id = step.agent_run.project_id if step.agent_run else None
    research_cycle_id = getattr(step.agent_run, 'research_cycle_id', None) if step.agent_run else None

    # Fetch context documents for the project (if available and enabled)
    context_documents = ""
    use_context_docs = True  # Default to True
    context_ab_testing = False  # Default to False

    # Debug logging for context document config
    print(f"📚 Dataset Design: project_id={project_id}")
    print(f"📚 Dataset Design: step.agent_run={step.agent_run}")
    if step.agent_run:
        print(f"📚 Dataset Design: config_json={step.agent_run.config_json}")
    else:
        print("📚 Dataset Design: WARNING - step.agent_run is None!")

    if step.agent_run and step.agent_run.config_json:
        use_context_docs = step.agent_run.config_json.get("use_context_documents", True)
        context_ab_testing = step.agent_run.config_json.get("context_ab_testing", False)
        print(f"📚 Dataset Design: use_context_docs={use_context_docs}, context_ab_testing={context_ab_testing}")
    else:
        print("📚 Dataset Design: No config_json found, using defaults")

    if project_id and use_context_docs:
        context_builder = ContextBuilder(db)
        context_documents = context_builder.build_context_section(project_id)
        if context_documents:
            if context_ab_testing:
                step_logger.info("📚 A/B Testing enabled: Will create variants WITH and WITHOUT context documents")
            else:
                step_logger.info("📚 Including context documents in dataset design")
        else:
            step_logger.info("📚 No active context documents found for this project")
            # Disable A/B testing if no context documents exist
            context_ab_testing = False
    elif project_id and not use_context_docs:
        step_logger.info("📚 Context documents disabled for this pipeline")
        context_ab_testing = False  # Can't A/B test if context is disabled

    # Format schema for the prompt
    schema_text = _format_schema_for_prompt(schema_summary)

    # Build the user prompt using centralized prompt generator
    user_prompt = get_dataset_design_prompt(
        schema_text=schema_text,
        task_type=task_type,
        target_column=target_column,
        description=description,
        max_variants=max_variants,
        project_history_context=None,  # Agent will query via tools
        context_documents=context_documents,
    )

    # Append audit context to prompt if available
    if audit_context:
        user_prompt += audit_context
        user_prompt += "\n\n**IMPORTANT**: Your dataset variants MUST account for the audit findings above. "
        user_prompt += "Exclude problematic columns, handle class imbalance appropriately, and avoid data leakage.\n"

    # Build messages with tool-enabled system prompt
    messages = [
        {"role": "system", "content": SYSTEM_ROLE_DATASET_DESIGN_WITH_TOOLS},
        {"role": "user", "content": user_prompt},
    ]

    # Create tool executor if we have a project context
    if project_id:
        step_logger.thinking("Agent will query project history via tools to inform dataset design...")
        tool_executor = AgentToolExecutor(
            db=db,
            project_id=project_id,
            current_cycle_id=research_cycle_id,
        )

        # Call LLM with tool support - agent will query history as needed
        # Retry loop for validation errors
        max_validation_retries = 2
        last_error = None

        for attempt in range(max_validation_retries + 1):
            step_logger.action(f"Consulting LLM for dataset design variants (with tool access)... {'(retry)' if attempt > 0 else ''}")
            result = await execute_with_tools(
                client=llm_client,
                messages=messages,
                tool_executor=tool_executor,
                response_schema=DatasetDesignSuggestion,
                step_logger=step_logger,
            )

            # Truncate variants if LLM returned more than allowed
            if "variants" in result and len(result["variants"]) > max_variants:
                step_logger.warning(f"LLM returned {len(result['variants'])} variants, truncating to {max_variants}")
                result["variants"] = result["variants"][:max_variants]
                # Ensure recommended_variant is still in the list
                variant_names = [v.get("name") for v in result["variants"]]
                if result.get("recommended_variant") not in variant_names:
                    result["recommended_variant"] = variant_names[0] if variant_names else "variant_1"

            # Fill in missing fields with sensible defaults
            result = _fill_dataset_design_defaults(result, step_logger)

            try:
                suggestion = DatasetDesignSuggestion(**result)
                break  # Success!
            except Exception as e:
                last_error = e
                step_logger.warning(f"Validation error (attempt {attempt + 1}): {str(e)[:500]}")

                if attempt < max_validation_retries:
                    # Add error feedback to messages for retry
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(result, default=str)[:2000]  # Truncate to save tokens
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Your JSON response had validation errors:\n{str(e)[:1000]}\n\n"
                                  "Please fix these errors and provide a corrected JSON response. "
                                  "Make sure all required fields are present (name, description, feature_columns, expected_tradeoff for each variant)."
                    })
                else:
                    raise ValueError(f"Failed to generate valid dataset design after {max_validation_retries + 1} attempts: {e}")
    else:
        # Fallback to direct generation without tools
        step_logger.action("Consulting LLM for dataset design variants...")
        suggestion = await generate_dataset_design(
            client=llm_client,
            schema_summary=schema_summary,
            task_type=task_type,
            target_column=target_column,
            description=description,
            max_variants=max_variants,
            project_history_context=None,
        )

    step_logger.info(f"Generated {len(suggestion.variants)} dataset variants")

    for variant in suggestion.variants:
        step_logger.thought(
            f"Variant '{variant.name}': {len(variant.feature_columns)} features, "
            f"{variant.train_test_split} split - {variant.description[:100]}..."
        )

    if suggestion.warnings:
        for warning in suggestion.warnings:
            step_logger.warning(warning)

    # Handle A/B Testing: if enabled, create variants both WITH and WITHOUT context
    all_variants = []
    all_warnings = list(suggestion.warnings) if suggestion.warnings else []

    if context_ab_testing and context_documents:
        # Current variants were generated WITH context - add suffix
        step_logger.info("📚 A/B Testing: Adding [WITH CONTEXT] variants...")
        for variant in suggestion.variants:
            variant_dict = variant.model_dump()
            variant_dict["name"] = f"{variant.name} [WITH CONTEXT]"
            variant_dict["description"] = f"[WITH CONTEXT] {variant.description}"
            all_variants.append(variant_dict)

        # Now generate variants WITHOUT context
        step_logger.info("📚 A/B Testing: Generating [NO CONTEXT] variants...")
        no_context_prompt = get_dataset_design_prompt(
            schema_text=schema_text,
            task_type=task_type,
            target_column=target_column,
            description=description,
            max_variants=max_variants,
            project_history_context=None,
            context_documents="",  # No context
        )
        if audit_context:
            no_context_prompt += audit_context
            no_context_prompt += "\n\n**IMPORTANT**: Your dataset variants MUST account for the audit findings above. "
            no_context_prompt += "Exclude problematic columns, handle class imbalance appropriately, and avoid data leakage.\n"

        no_context_messages = [
            {"role": "system", "content": SYSTEM_ROLE_DATASET_DESIGN_WITH_TOOLS},
            {"role": "user", "content": no_context_prompt},
        ]

        if project_id:
            no_context_tool_executor = AgentToolExecutor(
                db=db,
                project_id=project_id,
                current_cycle_id=research_cycle_id,
            )
            step_logger.action("Generating [NO CONTEXT] dataset variants...")
            no_context_result = await execute_with_tools(
                client=llm_client,
                messages=no_context_messages,
                tool_executor=no_context_tool_executor,
                response_schema=DatasetDesignSuggestion,
                step_logger=step_logger,
            )
            if "variants" in no_context_result and len(no_context_result["variants"]) > max_variants:
                no_context_result["variants"] = no_context_result["variants"][:max_variants]
            no_context_result = _fill_dataset_design_defaults(no_context_result, step_logger)
            no_context_suggestion = DatasetDesignSuggestion(**no_context_result)
        else:
            no_context_suggestion = await generate_dataset_design(
                client=llm_client,
                schema_summary=schema_summary,
                task_type=task_type,
                target_column=target_column,
                description=description,
                max_variants=max_variants,
                project_history_context=None,
            )

        # Add [NO CONTEXT] suffix to these variants
        for variant in no_context_suggestion.variants:
            variant_dict = variant.model_dump()
            variant_dict["name"] = f"{variant.name} [NO CONTEXT]"
            variant_dict["description"] = f"[NO CONTEXT] {variant.description}"
            all_variants.append(variant_dict)

        if no_context_suggestion.warnings:
            all_warnings.extend(no_context_suggestion.warnings)

        step_logger.info(f"📚 A/B Testing complete: {len(all_variants)} total variants ({len(suggestion.variants)} WITH + {len(no_context_suggestion.variants)} WITHOUT context)")

        # Update recommended variant to include suffix
        recommended_variant_name = f"{suggestion.recommended_variant} [WITH CONTEXT]"
    else:
        # No A/B testing - just use the variants as-is
        all_variants = [v.model_dump() for v in suggestion.variants]
        recommended_variant_name = suggestion.recommended_variant

    step_logger.summary(
        f"Dataset design complete: {len(all_variants)} variants generated, "
        f"recommended: '{recommended_variant_name}'"
    )

    # Return both the new multi-variant format AND legacy single-variant format for compatibility
    recommended = next(
        (v for v in suggestion.variants if v.name == suggestion.recommended_variant),
        suggestion.variants[0]
    )

    return {
        # New multi-variant format
        "variants": all_variants,
        "recommended_variant": recommended_variant_name,
        "reasoning": suggestion.reasoning,
        "warnings": all_warnings,
        # Legacy single-variant format for backward compatibility
        "feature_columns": recommended.feature_columns,
        "excluded_columns": recommended.excluded_columns,
        "exclusion_reasons": recommended.exclusion_reasons,
        "suggested_filters": recommended.suggested_filters,
        # A/B testing metadata
        "context_ab_testing": context_ab_testing,
        "had_context_documents": bool(context_documents),
    }


def _analyze_task_context_for_experiment_design(
    task_context: Optional[Dict[str, Any]],
    step_logger: StepLogger,
) -> Dict[str, Any]:
    """Analyze task_context to inform experiment design decisions (Prompt 7 Step 4).

    Extracts and analyzes:
    - Time-based task info (is_time_based, time_column, entity_id_column)
    - Robustness warnings from prior runs
    - Leakage candidates to potentially ablate
    - Past cycles to avoid repeating failed strategies
    - Expected metric ranges from Problem Framer

    Args:
        task_context: The task context dict or None
        step_logger: Logger for thinking/info messages

    Returns:
        Dict with analysis results for experiment design:
        - is_time_based: bool
        - time_column: str or None
        - entity_id_column: str or None
        - recommended_split_type: str
        - split_type_reasoning: str
        - leakage_features_to_drop: list of column names
        - prior_warnings: list of warning strings
        - should_run_sanity_check: bool
        - expected_metric_range: dict or None
        - failed_strategies: list of strategies to avoid
    """
    analysis = {
        "is_time_based": False,
        "time_column": None,
        "entity_id_column": None,
        "recommended_split_type": "stratified",
        "split_type_reasoning": "Default stratified split for classification tasks.",
        "leakage_features_to_drop": [],
        "prior_warnings": [],
        "should_run_sanity_check": False,
        "expected_metric_range": None,
        "failed_strategies": [],
    }

    if not task_context:
        step_logger.thought("No task context available - using default experiment design strategy")
        return analysis

    step_logger.thought("Analyzing task context for experiment design decisions...")

    # 1. Extract time-based info from dataset_spec
    dataset_spec = task_context.get("dataset_spec", {}) or {}
    analysis["is_time_based"] = dataset_spec.get("is_time_based", False)
    analysis["time_column"] = dataset_spec.get("time_column")
    analysis["entity_id_column"] = dataset_spec.get("entity_id_column")

    if analysis["is_time_based"]:
        step_logger.thought(f"Time-based task detected: time_column='{analysis['time_column']}'")
        if analysis["entity_id_column"]:
            analysis["recommended_split_type"] = "group_time"
            analysis["split_type_reasoning"] = (
                f"Time-based split with entity grouping (entity_id_column='{analysis['entity_id_column']}') "
                "to prevent look-ahead bias and entity leakage."
            )
        else:
            analysis["recommended_split_type"] = "time"
            analysis["split_type_reasoning"] = (
                f"Time-based split using '{analysis['time_column']}' to prevent look-ahead bias."
            )
        step_logger.thought(f"Recommended split: {analysis['recommended_split_type']}")
    else:
        step_logger.thought("Task is not time-based; standard stratified splits are appropriate")

    # 2. Extract robustness warnings from prior experiments
    robustness = task_context.get("robustness", {}) or {}

    if robustness.get("leakage_suspected"):
        analysis["prior_warnings"].append("Prior experiments suggest potential data leakage")
        analysis["should_run_sanity_check"] = True
        step_logger.thought("WARNING: Prior cycle flagged leakage_suspected - will include sanity check experiments")

    if robustness.get("overfitting_risk") == "high":
        analysis["prior_warnings"].append(f"High overfitting risk detected in prior experiments")
        analysis["should_run_sanity_check"] = True
        step_logger.thought("WARNING: High overfitting risk in prior cycles - will include regularization experiments")

    if robustness.get("time_split_suspicious"):
        analysis["prior_warnings"].append("Prior time-based split produced suspicious results")
        analysis["failed_strategies"].append("time_split_with_current_features")
        step_logger.thought("WARNING: Prior cycle time_split_suspicious - may need stricter temporal separation")

    # 3. Extract leakage candidates to potentially drop
    leakage_candidates = task_context.get("leakage_candidates", []) or []
    high_severity_leakage = [
        c["column"] for c in leakage_candidates
        if c.get("severity") == "high"
    ]
    if high_severity_leakage:
        analysis["leakage_features_to_drop"] = high_severity_leakage
        analysis["should_run_sanity_check"] = True
        step_logger.thought(
            f"Identified {len(high_severity_leakage)} high-severity leakage feature(s) to consider dropping: "
            f"{', '.join(high_severity_leakage[:5])}"
        )

    # 4. Extract expected metric range from project context
    project = task_context.get("project", {}) or {}
    # Check if there's expected_metric_range from a previous problem understanding step
    # This would be in past_cycles_summary or latest_experiments
    latest_experiments = task_context.get("latest_experiments", []) or []
    if latest_experiments:
        # Check for expected_metric_range in experiment metadata
        for exp in latest_experiments:
            if exp.get("expected_metric_range"):
                analysis["expected_metric_range"] = exp["expected_metric_range"]
                break

    # Also check baselines for setting expectations
    baselines = task_context.get("baselines", {}) or {}
    if baselines.get("available") and not analysis["expected_metric_range"]:
        # Build expected range from baselines
        majority_class = baselines.get("majority_class", {})
        simple_model = baselines.get("simple_model", {})
        if majority_class.get("accuracy") or simple_model.get("roc_auc"):
            analysis["expected_metric_range"] = {
                "baseline_accuracy": majority_class.get("accuracy"),
                "baseline_auc": simple_model.get("roc_auc"),
            }
            step_logger.thought(
                f"Baseline metrics: accuracy={majority_class.get('accuracy')}, "
                f"AUC={simple_model.get('roc_auc')}"
            )

    # 5. Check past cycles for failed strategies
    past_cycles = task_context.get("past_cycles_summary", []) or []
    for cycle in past_cycles:
        if cycle.get("status") == "failed":
            # Extract what went wrong
            failure_reason = cycle.get("failure_reason", "unknown")
            analysis["failed_strategies"].append(f"Cycle {cycle.get('cycle_number', '?')}: {failure_reason}")

    # Log summary
    if analysis["prior_warnings"]:
        step_logger.thought(f"Prior run warnings to address: {len(analysis['prior_warnings'])}")
    if analysis["failed_strategies"]:
        step_logger.thought(f"Failed strategies to avoid: {len(analysis['failed_strategies'])}")

    return analysis


def _generate_experiment_family_goals(
    task_type: str,
    primary_metric: str,
    context_analysis: Dict[str, Any],
    time_budget_minutes: Optional[int],
) -> List[Dict[str, Any]]:
    """Generate experiment family goals based on context analysis (Prompt 7 Step 4).

    Creates a structured list of experiments with goals that address:
    - Baseline/sanity checks
    - Main quality runs
    - Ablation experiments for suspected leakage features
    - Time-budget appropriate configurations

    Args:
        task_type: ML task type (binary, regression, etc.)
        primary_metric: Primary evaluation metric
        context_analysis: Output from _analyze_task_context_for_experiment_design
        time_budget_minutes: Optional total time budget

    Returns:
        List of experiment goal dicts with name, goal, time_budget_minutes, preset
    """
    experiments = []

    # Default time allocation
    total_budget = time_budget_minutes or 120
    is_time_based = context_analysis.get("is_time_based", False)
    should_sanity_check = context_analysis.get("should_run_sanity_check", False)
    leakage_features = context_analysis.get("leakage_features_to_drop", [])

    # 1. Always start with a quick baseline/sanity experiment
    if should_sanity_check or is_time_based:
        experiments.append({
            "name": "Baseline Sanity Check",
            "goal": (
                f"Validate that {context_analysis['recommended_split_type']} split "
                f"{primary_metric} is above baseline but not unrealistically high. "
                "Catches obvious data leakage or setup errors early."
            ),
            "time_budget_minutes": min(15, total_budget // 6),
            "preset": "good_quality",
            "split_type": context_analysis["recommended_split_type"],
        })

    # 2. Main experiment - good quality with proper split
    main_budget = total_budget // 2 if should_sanity_check else total_budget * 2 // 3
    experiments.append({
        "name": "Primary Experiment",
        "goal": (
            f"Train models optimizing {primary_metric} with {context_analysis['recommended_split_type']} split. "
            f"Target: achieve realistic improvement over baseline."
        ),
        "time_budget_minutes": main_budget,
        "preset": "best_quality" if total_budget >= 60 else "good_quality",
        "split_type": context_analysis["recommended_split_type"],
    })

    # 3. If leakage features detected, add ablation experiment
    if leakage_features:
        experiments.append({
            "name": "Leakage Feature Ablation",
            "goal": (
                f"Test model stability by dropping suspected leakage features: "
                f"{', '.join(leakage_features[:3])}. "
                "If performance drops significantly, those features may be legitimate. "
                "If similar, confirms they were causing leakage."
            ),
            "time_budget_minutes": min(30, total_budget // 4),
            "preset": "good_quality",
            "split_type": context_analysis["recommended_split_type"],
            "features_to_drop": leakage_features,
        })

    # 4. If prior cycle had issues, add diagnostic experiment
    if context_analysis.get("prior_warnings"):
        experiments.append({
            "name": "Diagnostic Experiment",
            "goal": (
                "Address prior cycle warnings: " +
                "; ".join(context_analysis["prior_warnings"][:2]) +
                ". Uses simpler models as sanity checks."
            ),
            "time_budget_minutes": min(20, total_budget // 5),
            "preset": "medium_quality",
            "split_type": context_analysis["recommended_split_type"],
            "use_simple_models": True,
        })

    return experiments


async def handle_experiment_design_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the experiment design step.

    This step creates experiment variants with different configurations.
    It uses project history context to learn from previous experiments and findings.

    Input JSON should contain:
    - task_type: The ML task type
    - target_column: The target column
    - primary_metric: The metric to optimize
    - feature_columns: Selected feature columns
    - row_count: Number of rows
    - time_budget_minutes: Optional time constraint
    - description: Optional additional context
    - target_stats: Optional target variable statistics for baseline context

    Returns:
        Dict with variants, recommended_variant, reasoning
    """
    input_data = step.input_json or {}

    task_type = input_data.get("task_type")
    target_column = input_data.get("target_column")
    primary_metric = input_data.get("primary_metric")
    feature_columns = input_data.get("feature_columns", [])
    row_count = input_data.get("row_count", 0)
    time_budget_minutes = input_data.get("time_budget_minutes")
    description = input_data.get("description")
    target_stats = input_data.get("target_stats")  # Target statistics for baseline context

    if not all([task_type, target_column, primary_metric]):
        raise ValueError("Missing required fields: task_type, target_column, primary_metric")

    # Extract audit findings from previous steps (if available)
    audit_critical_issues = input_data.get("critical_issues", [])
    audit_warnings = input_data.get("warnings", [])
    audit_details = input_data.get("audit_details", {})

    # Build audit context string for the LLM
    audit_context_exp = ""
    if audit_critical_issues or audit_warnings:
        audit_context_exp = "\n\n## 🔍 DATA AUDIT FINDINGS (from previous analysis)\n"

        if audit_critical_issues:
            audit_context_exp += "\n### ⚠️ CRITICAL ISSUES:\n"
            for issue in audit_critical_issues:
                audit_context_exp += f"- {issue}\n"

        if audit_warnings:
            audit_context_exp += "\n### ⚡ WARNINGS:\n"
            for warning in audit_warnings[:5]:  # Limit
                audit_context_exp += f"- {warning}\n"

        if audit_details:
            if audit_details.get("class_imbalance"):
                imb = audit_details["class_imbalance"]
                severity = imb.get('severity', 'unknown')
                ratio = imb.get('ratio', 'N/A')
                audit_context_exp += f"\n### Class Imbalance: {severity} ({ratio}:1 ratio)\n"
                audit_context_exp += "Consider: class weights, oversampling, or stratified validation.\n"

        step_logger.info("📋 Including Data Audit findings in experiment design context")

    step_logger.info(f"Designing experiment for {task_type} task with {len(feature_columns)} features")
    step_logger.thinking(f"Dataset has {row_count:,} rows, optimizing for {primary_metric}")

    if time_budget_minutes:
        step_logger.info(f"Time budget constraint: {time_budget_minutes} minutes")

    # Get project ID and research cycle ID for tool executor
    project_id = step.agent_run.project_id if step.agent_run else None
    research_cycle_id = getattr(step.agent_run, 'research_cycle_id', None) if step.agent_run else None

    # Build unified TaskContext for the experiment planner (Prompt 7)
    task_context_str = ""
    task_hints = {}
    task_context = None
    context_analysis = None
    context_factors = {}  # Initialized for Prompt 7 Step 7 logging
    if project_id:
        try:
            task_context = build_task_context(
                db=db,
                project_id=str(project_id),
                research_cycle_id=str(research_cycle_id) if research_cycle_id else None,
                include_leakage_candidates=True,
                include_past_cycles=True,
                max_experiments=5,
            )
            task_hints = get_task_type_hints(task_context)
            task_context_str = format_context_for_prompt(
                task_context,
                include_sections=["project", "dataset_spec", "baselines", "robustness", "leakage_candidates"],
                max_length=3000,
            )
            if task_context_str:
                step_logger.info("📊 Built unified TaskContext for experiment design")

            # Log context usage with consistent helper (Prompt 7 Step 7)
            context_factors = _log_context_usage(
                step_logger=step_logger,
                task_context=task_context,
                task_hints=task_hints,
                step_name="Experiment Design",
            )

            # Analyze task_context for experiment design decisions (Prompt 7 Step 4)
            context_analysis = _analyze_task_context_for_experiment_design(task_context, step_logger)

        except Exception as e:
            step_logger.warning(f"Could not build TaskContext: {e}")

    # Also extract task_context from input_data if passed in (Prompt 7 Step 2)
    input_task_context = input_data.get("task_context")
    if input_task_context and not context_analysis:
        try:
            context_analysis = _analyze_task_context_for_experiment_design(input_task_context, step_logger)
        except Exception as e:
            step_logger.warning(f"Could not analyze input task_context: {e}")

    # Default context_analysis if none available
    if not context_analysis:
        context_analysis = {
            "is_time_based": False,
            "recommended_split_type": "stratified",
            "split_type_reasoning": "Default stratified split for classification tasks.",
            "leakage_features_to_drop": [],
            "prior_warnings": [],
            "should_run_sanity_check": False,
        }

    # Build the user prompt using centralized prompt generator
    user_prompt = get_experiment_plan_prompt(
        task_type=task_type,
        target_column=target_column,
        primary_metric=primary_metric,
        feature_count=len(feature_columns),
        row_count=row_count,
        time_budget_minutes=time_budget_minutes,
        description=description,
        feature_columns=feature_columns,
        target_stats=target_stats,
        project_history_context=None,  # Agent will query via tools
    )

    # Append audit context to prompt if available
    if audit_context_exp:
        user_prompt += audit_context_exp
        user_prompt += "\n\n**IMPORTANT**: Your experiment design MUST account for the audit findings above. "
        user_prompt += "Consider appropriate validation strategies for any class imbalance or data issues.\n"

    # Append unified TaskContext (Prompt 7) - provides consistent view of project state
    if task_context_str:
        user_prompt += "\n\n## 📊 PROJECT CONTEXT\n"
        user_prompt += task_context_str
        # Add task-specific guidance based on hints
        if task_hints.get("is_time_based"):
            user_prompt += "\n\n**TIME-BASED TASK**: Use time-based validation strategy (e.g., walk-forward). "
            user_prompt += "DO NOT use random shuffle for validation - this would leak future information.\n"
        if task_hints.get("leakage_warnings"):
            user_prompt += "\n**LEAKAGE WARNINGS**:\n"
            for warning in task_hints["leakage_warnings"][:3]:
                user_prompt += f"- {warning}\n"

    # Add context_analysis guidance to prompt (Prompt 7 Step 4)
    if context_analysis:
        user_prompt += "\n\n## 🎯 EXPERIMENT DESIGN GUIDANCE (from TaskContext analysis)\n"
        user_prompt += f"**Recommended Split Type**: `{context_analysis['recommended_split_type']}`\n"
        user_prompt += f"**Reasoning**: {context_analysis.get('split_type_reasoning', 'N/A')}\n"

        if context_analysis.get("is_time_based"):
            user_prompt += "\n**CRITICAL**: This is a TIME-BASED task. You MUST use time-based splits. "
            user_prompt += "Do NOT propose random splits unless you provide explicit justification in your reasoning.\n"

        if context_analysis.get("leakage_features_to_drop"):
            user_prompt += f"\n**Leakage Features to Consider Dropping**: {', '.join(context_analysis['leakage_features_to_drop'][:5])}\n"
            user_prompt += "Consider creating an ablation experiment that drops these features to test model stability.\n"

        if context_analysis.get("prior_warnings"):
            user_prompt += "\n**Prior Cycle Warnings to Address**:\n"
            for warning in context_analysis["prior_warnings"][:3]:
                user_prompt += f"- {warning}\n"

        if context_analysis.get("failed_strategies"):
            user_prompt += "\n**Failed Strategies to Avoid**:\n"
            for strategy in context_analysis["failed_strategies"][:3]:
                user_prompt += f"- {strategy}\n"

    # Handle revision requests from Plan Critic
    is_revision = input_data.get("revision_request", False)
    critic_feedback = input_data.get("critic_feedback", "")
    revision_number = input_data.get("revision_number", 0)

    if is_revision and critic_feedback:
        step_logger.info(f"🔄 This is a plan revision (attempt {revision_number}) based on Critic feedback")
        user_prompt += f"\n\n{critic_feedback}\n"
        user_prompt += "\n**CRITICAL**: This is a revision request. You MUST address the Critic's concerns above. "
        user_prompt += "Either fix the identified issues OR provide a detailed, convincing justification "
        user_prompt += "in `validation_strategy.reasoning` explaining why your approach is correct.\n"

    # Build messages with tool-enabled system prompt
    messages = [
        {"role": "system", "content": SYSTEM_ROLE_EXPERIMENT_DESIGN_WITH_TOOLS},
        {"role": "user", "content": user_prompt},
    ]

    # Create tool executor if we have a project context
    if project_id:
        step_logger.thinking("Agent will query project history via tools to inform experiment design...")
        tool_executor = AgentToolExecutor(
            db=db,
            project_id=project_id,
            current_cycle_id=research_cycle_id,
        )

        # Call LLM with tool support - agent will query history as needed
        # Retry loop for validation errors
        max_validation_retries = 2
        last_error = None

        for attempt in range(max_validation_retries + 1):
            step_logger.action(f"Consulting LLM for experiment variants (with tool access)... {'(retry)' if attempt > 0 else ''}")
            result = await execute_with_tools(
                client=llm_client,
                messages=messages,
                tool_executor=tool_executor,
                response_schema=ExperimentPlanSuggestion,
                step_logger=step_logger,
            )

            # Fill in missing fields with sensible defaults
            result = _fill_experiment_design_defaults(result, step_logger)

            try:
                suggestion = ExperimentPlanSuggestion(**result)
                break  # Success!
            except Exception as e:
                last_error = e
                step_logger.warning(f"Validation error (attempt {attempt + 1}): {str(e)[:500]}")

                if attempt < max_validation_retries:
                    # Add error feedback to messages for retry
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(result, default=str)[:2000]  # Truncate to save tokens
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Your JSON response had validation errors:\n{str(e)[:1000]}\n\n"
                                  "Please fix these errors and provide a corrected JSON response. "
                                  "Make sure all required fields are present."
                    })
                else:
                    raise ValueError(f"Failed to generate valid experiment design after {max_validation_retries + 1} attempts: {e}")
    else:
        # Fallback to direct generation without tools
        step_logger.action("Consulting LLM for experiment variants...")
        suggestion = await generate_experiment_plan(
            client=llm_client,
            task_type=task_type,
            target_column=target_column,
            primary_metric=primary_metric,
            feature_columns=feature_columns,
            row_count=row_count,
            time_budget_minutes=time_budget_minutes,
            description=description,
            target_stats=target_stats,
            project_history_context=None,
        )

    step_logger.info(f"Generated {len(suggestion.variants)} experiment variants")

    for variant in suggestion.variants:
        time_limit = variant.automl_config.get("time_limit", 300)
        presets = variant.automl_config.get("presets", "medium_quality")
        step_logger.thought(f"Variant '{variant.name}': {presets} preset, {time_limit}s time limit")

    step_logger.summary(
        f"Experiment design complete. Recommended: '{suggestion.recommended_variant}'. "
        f"Estimated total time: {suggestion.estimated_total_time_minutes} minutes."
    )

    # Store the experiment design config for ALL dataset specs in the project
    # This ensures the "Run Experiments" button works for all datasets, even if
    # a new agent run (e.g., iteration) happens before experiments are created
    try:
        # Get the agent run to get the project_id
        agent_run = db.query(AgentRun).filter(AgentRun.id == step.agent_run_id).first()
        if agent_run and agent_run.project_id:
            # Get all dataset specs for this project
            dataset_specs = (
                db.query(DatasetSpec)
                .filter(DatasetSpec.project_id == agent_run.project_id)
                .all()
            )

            # Build the config to store
            experiment_design_config = {
                "step_id": str(step.id),
                "agent_run_id": str(agent_run.id),
                "variants": [v.model_dump() for v in suggestion.variants],
                "recommended_variant": suggestion.recommended_variant,
                "primary_metric": primary_metric,
                "natural_language_summary": suggestion.reasoning or "",
                "stored_at": datetime.utcnow().isoformat(),
                "source_type": "initial",
                "parent_experiment_id": None,
            }

            # Store config for each dataset spec that doesn't already have one
            stored_count = 0
            for spec in dataset_specs:
                if not spec.agent_experiment_design_json:
                    spec.agent_experiment_design_json = experiment_design_config
                    stored_count += 1

            if stored_count > 0:
                db.commit()
                step_logger.info(f"Stored experiment design config for {stored_count} dataset spec(s)")
    except Exception as e:
        # Log but don't fail the step if config storage fails
        logger.warning(f"Failed to store experiment design config for dataset specs: {e}")

    # Generate experiment_family_goals based on context analysis (Prompt 7 Step 4)
    experiment_family_goals = _generate_experiment_family_goals(
        task_type=task_type,
        primary_metric=primary_metric,
        context_analysis=context_analysis,
        time_budget_minutes=time_budget_minutes,
    )

    # Log the experiment goals
    step_logger.thought(f"Generated {len(experiment_family_goals)} experiment family goal(s)")
    for goal in experiment_family_goals:
        step_logger.thought(f"  - {goal['name']}: {goal['goal'][:100]}...")

    return {
        "variants": [v.model_dump() for v in suggestion.variants],
        "recommended_variant": suggestion.recommended_variant,
        "reasoning": suggestion.reasoning,
        "estimated_total_time_minutes": suggestion.estimated_total_time_minutes,
        # Prompt 7 Step 4: Split strategy and experiment goals
        "split_strategy": {
            "type": context_analysis.get("recommended_split_type", "stratified"),
            "reasoning": context_analysis.get("split_type_reasoning", ""),
            "is_time_based": context_analysis.get("is_time_based", False),
            "time_column": context_analysis.get("time_column"),
            "entity_id_column": context_analysis.get("entity_id_column"),
        },
        "experiment_family_goals": experiment_family_goals,
        "context_analysis": {
            "leakage_features_to_drop": context_analysis.get("leakage_features_to_drop", []),
            "prior_warnings": context_analysis.get("prior_warnings", []),
            "should_run_sanity_check": context_analysis.get("should_run_sanity_check", False),
            "expected_metric_range": context_analysis.get("expected_metric_range"),
        },
        # Context factors used (Prompt 7 Step 7)
        "context_factors_used": context_factors,
    }


def _validate_plan_against_context(
    input_data: Dict[str, Any],
    task_context: Optional[Dict[str, Any]],
    task_hints: Dict[str, Any],
    step_logger: StepLogger,
) -> Dict[str, Any]:
    """Validate plan against task context for context-aware critique (Prompt 7 Step 5).

    Args:
        input_data: The plan input data including variants, features, etc.
        task_context: The built TaskContext from the database
        task_hints: The task type hints derived from context
        step_logger: Logger for step output

    Returns:
        Dict with validation results:
        - split_validation: {valid, warnings, required_changes}
        - metric_validation: {valid, warnings, required_changes}
        - leakage_validation: {valid, warnings, required_changes}
        - overall_valid: bool
    """
    validation_result = {
        "split_validation": {"valid": True, "warnings": [], "required_changes": []},
        "metric_validation": {"valid": True, "warnings": [], "required_changes": []},
        "leakage_validation": {"valid": True, "warnings": [], "required_changes": []},
        "overall_valid": True,
    }

    # Extract plan components
    feature_columns = set(input_data.get("feature_columns", []))
    variants = input_data.get("variants", [])
    is_time_based = input_data.get("is_time_based", False)
    time_column = input_data.get("time_column")

    # Override is_time_based from task_context if available
    if task_context:
        dataset_spec = task_context.get("dataset_spec") or {}
        if dataset_spec.get("is_time_based"):
            is_time_based = True
            time_column = time_column or dataset_spec.get("time_column")
            step_logger.thought("TaskContext confirms this is a time-based task")

    # Also check task_hints
    if task_hints.get("is_time_based"):
        is_time_based = True
        step_logger.thought("TaskHints confirms this is a time-based task")

    # -------------------------------------------------------------------
    # 1. SPLIT STRATEGY VALIDATION
    # For time-based tasks, require time-based split or explicit override
    # -------------------------------------------------------------------
    if is_time_based:
        for variant in variants:
            val_strategy = variant.get("validation_strategy", {})
            split_type = val_strategy.get("split_strategy", "random")
            reasoning = val_strategy.get("reasoning", "")

            if split_type in ("random", "stratified", "group_random"):
                # Check if there's an explicit override flag
                override_flag = val_strategy.get("time_split_override", False)
                has_justification = reasoning and len(reasoning.strip()) > 20

                if override_flag and has_justification:
                    # Explicit override with justification - accept with warning
                    validation_result["split_validation"]["warnings"].append(
                        f"Variant '{variant.get('name')}' uses '{split_type}' split on time-based data "
                        f"with explicit override. Justification: {reasoning[:100]}..."
                    )
                elif not has_justification:
                    # No override and no justification - require change
                    validation_result["split_validation"]["valid"] = False
                    validation_result["split_validation"]["required_changes"].append({
                        "variant": variant.get("name"),
                        "issue": "random_split_on_time_data",
                        "current_value": split_type,
                        "required_action": f"Use 'time' or 'group_time' split for time-based task with time_column='{time_column}'",
                        "alternative": "Provide 'time_split_override: true' with detailed justification",
                    })
                    validation_result["overall_valid"] = False

    # -------------------------------------------------------------------
    # 2. EXPECTED METRIC RANGE VALIDATION
    # Check if plan targets metrics outside expected range
    # -------------------------------------------------------------------
    expected_range = None
    if task_context:
        # Check for expected metric range from Problem Framer output
        # It may be in context_analysis from Experiment Planner or directly in task_context
        context_analysis = input_data.get("context_analysis", {})
        expected_range = context_analysis.get("expected_metric_range")

        # Also check baselines for comparison
        baselines = task_context.get("baselines", {})
        if baselines.get("available") and not expected_range:
            # Construct expected range from baselines
            task_type = input_data.get("task_type", "binary")
            primary_metric = input_data.get("primary_metric", "")

            if task_type in ("binary", "multiclass"):
                # For classification, expect to beat simple model but not by crazy amounts
                simple_auc = baselines.get("simple_model", {}).get("roc_auc", 0.5)
                majority_auc = baselines.get("majority_class", {}).get("roc_auc", 0.5)
                baseline_auc = max(simple_auc or 0.5, majority_auc or 0.5)

                expected_range = {
                    "lower_bound": baseline_auc,
                    "upper_bound": min(0.95, baseline_auc + 0.20),  # Cap at 0.95, max 0.20 improvement
                    "metric": "roc_auc",
                    "reasoning": "Based on baseline models",
                }
            elif task_type == "regression":
                # For regression, expect to improve on mean predictor
                mean_rmse = baselines.get("mean_predictor", {}).get("rmse")
                ridge_rmse = baselines.get("regression_baseline", {}).get("rmse")

                if mean_rmse or ridge_rmse:
                    baseline_rmse = ridge_rmse or mean_rmse
                    expected_range = {
                        "lower_bound": baseline_rmse * 0.30,  # Best case: 70% reduction
                        "upper_bound": baseline_rmse * 0.90,  # Should at least beat baseline by 10%
                        "metric": "rmse",
                        "reasoning": "Based on baseline models",
                    }

    if expected_range:
        step_logger.thought(
            f"Expected metric range: {expected_range.get('metric')} "
            f"[{expected_range.get('lower_bound'):.3f} - {expected_range.get('upper_bound'):.3f}]"
        )

        # Check if plan mentions target metrics that are unrealistic
        for variant in variants:
            target_metrics = variant.get("target_metrics", {})
            for metric_name, target_value in target_metrics.items():
                if isinstance(target_value, (int, float)):
                    # Check if metric is the same as expected range metric
                    if expected_range.get("metric") and metric_name.lower() == expected_range["metric"].lower():
                        lower = expected_range.get("lower_bound", 0)
                        upper = expected_range.get("upper_bound", 1)

                        # For classification metrics (higher is better)
                        if metric_name.lower() in ("roc_auc", "auc", "accuracy", "f1", "precision", "recall"):
                            if target_value > upper + 0.05:  # 5% tolerance
                                validation_result["metric_validation"]["warnings"].append(
                                    f"Variant '{variant.get('name')}' targets {metric_name}={target_value:.3f}, "
                                    f"which exceeds expected upper bound of {upper:.3f}. "
                                    f"This may be unrealistic given baseline performance."
                                )
                        # For error metrics (lower is better)
                        elif metric_name.lower() in ("rmse", "mae", "mse"):
                            if target_value < lower * 0.8:  # 20% below lower bound
                                validation_result["metric_validation"]["warnings"].append(
                                    f"Variant '{variant.get('name')}' targets {metric_name}={target_value:.3f}, "
                                    f"which is below expected lower bound of {lower:.3f}. "
                                    f"This may be unrealistic given baseline performance."
                                )

    # -------------------------------------------------------------------
    # 3. LEAKAGE FEATURE VALIDATION
    # Check if plan includes features flagged as potential leakage
    # -------------------------------------------------------------------
    if task_context:
        leakage_candidates = task_context.get("leakage_candidates", [])
        if leakage_candidates:
            high_severity = [lc for lc in leakage_candidates if lc.get("severity") == "high"]
            medium_severity = [lc for lc in leakage_candidates if lc.get("severity") == "medium"]

            # Check high-severity leakage features
            for lc in high_severity:
                col = lc.get("column")
                if col in feature_columns:
                    validation_result["leakage_validation"]["valid"] = False
                    validation_result["leakage_validation"]["required_changes"].append({
                        "column": col,
                        "severity": "high",
                        "reason": lc.get("reason", "Potential data leakage"),
                        "required_action": f"Remove '{col}' from features or run ablation experiment",
                        "detection_method": lc.get("detection_method", "unknown"),
                    })
                    validation_result["overall_valid"] = False

            # Check medium-severity leakage features (warning only)
            for lc in medium_severity:
                col = lc.get("column")
                if col in feature_columns:
                    validation_result["leakage_validation"]["warnings"].append(
                        f"Feature '{col}' flagged as potential leakage (medium severity): {lc.get('reason')}. "
                        f"Consider running an ablation experiment to verify."
                    )

    # Log validation summary
    split_issues = len(validation_result["split_validation"]["required_changes"])
    leakage_issues = len(validation_result["leakage_validation"]["required_changes"])
    total_warnings = (
        len(validation_result["split_validation"]["warnings"]) +
        len(validation_result["metric_validation"]["warnings"]) +
        len(validation_result["leakage_validation"]["warnings"])
    )

    if split_issues or leakage_issues:
        step_logger.warning(
            f"Context validation found {split_issues} split issue(s), {leakage_issues} leakage issue(s)"
        )
    if total_warnings:
        step_logger.thought(f"Context validation raised {total_warnings} warning(s)")

    return validation_result


def _generate_plan_summary(
    approved: bool,
    issues: List[str],
    warnings: List[str],
    required_changes: List[Dict[str, Any]],
    feature_count: int,
    variant_count: int,
    context_validation: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a natural language summary of plan review (Prompt 7 Step 5).

    Returns a human-readable summary of the plan review outcome.
    """
    summary_parts = []

    if approved:
        summary_parts.append(
            f"✅ Plan APPROVED with {variant_count} experiment variant(s) and {feature_count} features."
        )
    else:
        summary_parts.append(
            f"❌ Plan REQUIRES REVISION. {len(required_changes)} issue(s) must be addressed before proceeding."
        )

    # Summarize required changes
    if required_changes:
        change_types = {}
        for change in required_changes:
            issue_type = change.get("issue", "unknown")
            change_types[issue_type] = change_types.get(issue_type, 0) + 1

        change_summary = ", ".join(
            f"{count} {issue_type.replace('_', ' ')}" for issue_type, count in change_types.items()
        )
        summary_parts.append(f"Required changes: {change_summary}.")

    # Summarize warnings
    if warnings:
        summary_parts.append(f"{len(warnings)} warning(s) to consider:")
        for warning in warnings[:3]:  # First 3 warnings
            # Truncate long warnings
            if len(warning) > 100:
                warning = warning[:100] + "..."
            summary_parts.append(f"  • {warning}")
        if len(warnings) > 3:
            summary_parts.append(f"  • ... and {len(warnings) - 3} more")

    # Add context-specific notes
    if context_validation:
        if not context_validation.get("split_validation", {}).get("valid", True):
            summary_parts.append(
                "⚠️ Split strategy requires attention: Time-based data detected but random split proposed."
            )
        if not context_validation.get("leakage_validation", {}).get("valid", True):
            summary_parts.append(
                "🚨 Leakage risk: High-severity leakage candidates found in feature set."
            )

    return "\n".join(summary_parts)


async def handle_plan_critic_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the plan critic step.

    This step reviews the overall plan and provides feedback.

    Input JSON should contain the full plan from previous steps.

    Returns:
        Dict with review results, potential issues, and approval status
    """
    input_data = step.input_json or {}

    step_logger.info("Reviewing the experiment plan...")

    # Extract plan components
    task_type = input_data.get("task_type")
    target_column = input_data.get("target_column")
    feature_columns = input_data.get("feature_columns", [])
    variants = input_data.get("variants", [])

    # Build unified TaskContext for the plan critic (Prompt 7)
    project_id = step.agent_run.project_id if step.agent_run else None
    research_cycle_id = getattr(step.agent_run, 'research_cycle_id', None) if step.agent_run else None
    task_context = None
    task_hints = {}
    context_factors = {}  # Initialized for Prompt 7 Step 7 logging
    if project_id:
        try:
            task_context = build_task_context(
                db=db,
                project_id=str(project_id),
                research_cycle_id=str(research_cycle_id) if research_cycle_id else None,
                include_leakage_candidates=True,
                include_past_cycles=True,
            )
            task_hints = get_task_type_hints(task_context)
            step_logger.info("📊 Built unified TaskContext for plan review")

            # Log context usage with consistent helper (Prompt 7 Step 7)
            context_factors = _log_context_usage(
                step_logger=step_logger,
                task_context=task_context,
                task_hints=task_hints,
                step_name="Plan Critic",
            )
        except Exception as e:
            step_logger.warning(f"Could not build TaskContext: {e}")

    # Run context-aware validation against plan (Prompt 7 Step 5)
    context_validation = _validate_plan_against_context(
        input_data=input_data,
        task_context=task_context,
        task_hints=task_hints,
        step_logger=step_logger,
    )

    # Extract audit findings from accumulated context (flat keys from Data Audit step)
    audit_critical_issues = input_data.get("critical_issues", [])
    audit_warnings_from_audit = input_data.get("warnings", [])
    audit_details = input_data.get("audit_details", {})
    high_null_columns = input_data.get("high_null_columns", [])
    constant_columns = input_data.get("constant_columns", [])
    potential_id_columns = input_data.get("potential_id_columns", [])

    issues = []
    warnings = []
    required_changes = []  # Prompt 7 Step 5: Structured required changes list
    approved = True

    # Integrate context validation results (Prompt 7 Step 5)
    if not context_validation.get("overall_valid", True):
        approved = False

    # Add required changes from context validation
    for validation_type in ["split_validation", "leakage_validation", "metric_validation"]:
        val_result = context_validation.get(validation_type, {})
        for change in val_result.get("required_changes", []):
            required_changes.append(change)
            # Also add to issues for backward compatibility
            change_desc = change.get("required_action") or change.get("issue", "Unknown issue")
            issues.append(f"[{validation_type}] {change_desc}")

        for warning in val_result.get("warnings", []):
            if warning not in warnings:
                warnings.append(warning)

    # Log that we're using audit findings
    if audit_critical_issues or audit_warnings_from_audit:
        step_logger.info(f"📋 Reviewing plan against {len(audit_critical_issues)} critical issues and {len(audit_warnings_from_audit)} audit warnings")

    # Check for critical issues from data audit
    if constant_columns:
        # Check if any constant columns are in the feature set
        included_constants = [c for c in constant_columns if c in feature_columns]
        if included_constants:
            issues.append(f"Plan includes constant columns that should be removed: {included_constants}")
            approved = False
            step_logger.error(f"⚠️ Constant columns in features: {included_constants}")

    if high_null_columns:
        included_high_null = [c for c in high_null_columns if c in feature_columns]
        if included_high_null:
            warnings.append(f"Some features have high null rates (>30%): {included_high_null}")
            step_logger.warning(f"⚡ High-null columns in features: {included_high_null}")

    if potential_id_columns:
        included_ids = [c for c in potential_id_columns if c in feature_columns]
        if included_ids:
            warnings.append(f"Potential ID columns included as features (may cause leakage): {included_ids}")
            step_logger.warning(f"⚡ Potential ID columns in features: {included_ids}")

    # Check for class imbalance from audit
    if audit_details.get("class_imbalance"):
        imbalance = audit_details["class_imbalance"]
        if imbalance.get("severity") == "severe":
            warnings.append(f"Severe class imbalance ({imbalance.get('ratio', 'N/A')}:1) - ensure appropriate handling")
            step_logger.warning(f"⚡ Severe class imbalance detected: {imbalance.get('ratio')}:1")

    # Check for data leakage risk
    if audit_details.get("leakage_risk"):
        leakage = audit_details["leakage_risk"]
        leaky_cols = leakage.get("columns", [])
        included_leaky = [c for c in leaky_cols if c in feature_columns]
        if included_leaky:
            issues.append(f"POTENTIAL DATA LEAKAGE: {included_leaky} may contain future information")
            approved = False
            step_logger.error(f"🚨 Data leakage risk: {included_leaky}")

    # Use TaskContext hints to add additional checks (Prompt 7)
    if task_hints:
        # Check for leakage warnings from TaskContext
        if task_hints.get("leakage_warnings"):
            for warning in task_hints["leakage_warnings"][:3]:
                if warning not in warnings:
                    warnings.append(f"TaskContext: {warning}")
                    step_logger.warning(f"⚠️ {warning}")

        # Override is_time_based if TaskContext says so
        if task_hints.get("is_time_based") and not input_data.get("is_time_based"):
            step_logger.warning("⚠️ TaskContext indicates time-based task but plan doesn't reflect this")
            warnings.append("TaskContext indicates this is a time-based task - ensure proper time-aware validation")

        # Check for data quality warnings
        if task_hints.get("data_quality_warnings"):
            for dq_warning in task_hints["data_quality_warnings"][:2]:
                if dq_warning not in warnings:
                    warnings.append(f"Data Quality: {dq_warning}")

    # Check for leakage candidates from TaskContext
    if task_context and task_context.get("leakage_candidates"):
        leakage_candidates = task_context["leakage_candidates"]
        high_severity = [lc for lc in leakage_candidates if lc.get("severity") == "high"]
        if high_severity:
            leaky_in_features = [lc["column"] for lc in high_severity if lc["column"] in feature_columns]
            if leaky_in_features:
                issues.append(
                    f"HIGH-SEVERITY LEAKAGE CANDIDATES in features: {leaky_in_features}. "
                    f"Review these columns before training."
                )
                approved = False
                step_logger.error(f"🚨 High-severity leakage candidates in features: {leaky_in_features}")

    # Include any critical issues from audit that weren't addressed
    if audit_critical_issues:
        for critical in audit_critical_issues:
            if "small dataset" in critical.lower():
                warnings.append("Small dataset warning from audit - results may be unreliable")
            elif "imbalance" in critical.lower() and "class_imbalance" not in str(audit_details):
                warnings.append(critical)

    step_logger.thought(f"Checking {len(feature_columns)} features and {len(variants)} variants...")

    # Check for basic validity
    if not feature_columns:
        issues.append("No features selected for training")
        approved = False
        step_logger.error("Critical: No features selected!")

    if not variants:
        issues.append("No experiment variants defined")
        approved = False
        step_logger.error("Critical: No experiment variants!")

    if len(feature_columns) < 2:
        warnings.append("Very few features selected - model may have limited predictive power")
        step_logger.warning("Only 1-2 features selected")

    # Check for split strategy concerns on time-based tasks
    is_time_based = input_data.get("is_time_based", False)
    time_column = input_data.get("time_column")
    entity_id_column = input_data.get("entity_id_column")

    # Track variants that need justification review
    variants_needing_justification = []
    revision_feedback = []

    if is_time_based:
        step_logger.info(f"📅 Time-based task detected (time_column: {time_column}, entity_id: {entity_id_column})")

        for variant in variants:
            val_strategy = variant.get("validation_strategy", {})
            split_type = val_strategy.get("split_strategy", "random")
            reasoning = val_strategy.get("reasoning", "")

            # Check for non-time-based splits on time-based data
            if split_type in ("random", "stratified", "group_random"):
                step_logger.info(
                    f"Variant '{variant.get('name')}' uses '{split_type}' split on time-based task. "
                    f"Evaluating justification..."
                )

                # Check if the Planner provided a justification
                if not reasoning or len(reasoning.strip()) < 20:
                    # No meaningful justification provided - require revision
                    issues.append(
                        f"⚠️ Variant '{variant.get('name')}' uses '{split_type}' split on time-based data "
                        f"but provides no justification. Random splits on time-series data typically cause "
                        f"data leakage. Please either: (1) switch to 'time' or 'group_time' split, or "
                        f"(2) provide a detailed explanation for why random split is appropriate here."
                    )
                    revision_feedback.append({
                        "variant": variant.get("name"),
                        "issue": "random_split_on_time_data",
                        "current_split": split_type,
                        "suggestion": "Use 'time' or 'group_time' split, or provide strong justification",
                        "requires_response": True
                    })
                    approved = False
                    step_logger.warning(
                        f"❌ Variant '{variant.get('name')}': No justification for '{split_type}' split on time-based data"
                    )
                else:
                    # Justification provided - evaluate its quality using LLM
                    variants_needing_justification.append({
                        "variant_name": variant.get("name"),
                        "split_type": split_type,
                        "reasoning": reasoning,
                        "time_column": time_column,
                        "entity_id_column": entity_id_column,
                    })

            # Check that time_column is specified for time-based splits
            elif split_type in ("time", "group_time", "temporal"):
                vs_time_col = val_strategy.get("time_column")
                if not vs_time_col and not time_column:
                    warnings.append(
                        f"Variant '{variant.get('name')}' uses '{split_type}' split but no time_column specified"
                    )
                    step_logger.warning(f"⚡ Missing time_column for '{split_type}' split in '{variant.get('name')}'")

                # Check entity_id_column for group_time splits
                if split_type == "group_time":
                    vs_entity_col = val_strategy.get("entity_id_column") or val_strategy.get("group_column")
                    if not vs_entity_col and not entity_id_column:
                        warnings.append(
                            f"Variant '{variant.get('name')}' uses 'group_time' split but no entity_id_column specified"
                        )
                        step_logger.warning(f"⚡ Missing entity_id_column for 'group_time' split in '{variant.get('name')}'")

    # Use LLM to evaluate any justifications that were provided
    if variants_needing_justification and llm_client:
        step_logger.info(f"📝 Evaluating {len(variants_needing_justification)} justification(s) for non-standard splits...")

        for var_info in variants_needing_justification:
            eval_prompt = f"""You are a senior ML engineer reviewing an experiment plan.

The data scientist proposes using a '{var_info['split_type']}' split strategy for a TIME-BASED prediction task.

Time-based context:
- Time column: {var_info['time_column'] or 'not specified'}
- Entity ID column: {var_info['entity_id_column'] or 'not specified (single time series)'}

Their justification:
"{var_info['reasoning']}"

IMPORTANT: Random/stratified splits on time-series data typically cause DATA LEAKAGE because:
- Future data points may end up in the training set
- The model learns patterns from the future to predict the past
- This produces artificially inflated metrics that won't generalize

However, there ARE valid exceptions where random splits might be acceptable:
1. Cross-sectional data at a single point in time (not truly temporal)
2. When time independence can be verified (no autocorrelation)
3. Specific experimental designs (e.g., comparing models, not production deployment)

Evaluate the justification. Is it:
A) VALID - The justification provides a legitimate reason why random splits won't cause leakage
B) WEAK - The justification doesn't address the leakage concern adequately
C) INVALID - The justification is wrong or irrelevant

Respond with a JSON object:
{{"verdict": "VALID" | "WEAK" | "INVALID", "explanation": "1-2 sentence explanation", "suggested_action": "what the data scientist should do"}}"""

            try:
                from pydantic import BaseModel, Field

                class JustificationEval(BaseModel):
                    verdict: str = Field(description="VALID, WEAK, or INVALID")
                    explanation: str = Field(description="1-2 sentence explanation")
                    suggested_action: str = Field(description="What the data scientist should do")

                eval_result = await llm_client.structured_completion(
                    messages=[{"role": "user", "content": eval_prompt}],
                    response_model=JustificationEval,
                    temperature=0.3,
                )

                verdict = eval_result.verdict.upper()
                step_logger.info(
                    f"Justification evaluation for '{var_info['variant_name']}': {verdict} - {eval_result.explanation}"
                )

                if verdict == "VALID":
                    # Justification accepted - add as warning but don't block
                    warnings.append(
                        f"✓ Variant '{var_info['variant_name']}' uses '{var_info['split_type']}' split on time-based data. "
                        f"Justification accepted: {eval_result.explanation}"
                    )
                elif verdict == "WEAK":
                    # Weak justification - require stronger reasoning or plan change
                    issues.append(
                        f"⚠️ Variant '{var_info['variant_name']}': Justification for '{var_info['split_type']}' split is weak. "
                        f"{eval_result.explanation} {eval_result.suggested_action}"
                    )
                    revision_feedback.append({
                        "variant": var_info["variant_name"],
                        "issue": "weak_justification",
                        "evaluation": eval_result.explanation,
                        "suggested_action": eval_result.suggested_action,
                        "requires_response": True
                    })
                    approved = False
                else:  # INVALID
                    # Invalid justification - reject
                    issues.append(
                        f"❌ Variant '{var_info['variant_name']}': Justification for '{var_info['split_type']}' split is invalid. "
                        f"{eval_result.explanation} {eval_result.suggested_action}"
                    )
                    revision_feedback.append({
                        "variant": var_info["variant_name"],
                        "issue": "invalid_justification",
                        "evaluation": eval_result.explanation,
                        "suggested_action": eval_result.suggested_action,
                        "requires_response": True
                    })
                    approved = False

            except Exception as e:
                logger.warning(f"Failed to evaluate justification for '{var_info['variant_name']}': {e}")
                # On eval failure, be conservative - require time-based split
                issues.append(
                    f"⚠️ Could not evaluate justification for '{var_info['variant_name']}'. "
                    f"Please use 'time' or 'group_time' split for time-based data to prevent data leakage."
                )
                approved = False

    # Check variant configurations
    for variant in variants:
        config = variant.get("automl_config", {})
        time_limit = config.get("time_limit", 0)
        if time_limit < 30:
            warnings.append(f"Variant '{variant.get('name')}' has very short time limit ({time_limit}s)")

    if warnings:
        for warning in warnings:
            step_logger.warning(warning)

    # Merge revision_feedback into required_changes (Prompt 7 Step 5)
    # revision_feedback comes from split justification evaluations
    for feedback in revision_feedback:
        required_changes.append({
            "variant": feedback.get("variant"),
            "issue": feedback.get("issue"),
            "current_value": feedback.get("current_split"),
            "required_action": feedback.get("suggested_action") or feedback.get("suggestion"),
            "evaluation": feedback.get("evaluation"),
        })

    status = "approved" if approved else "needs_revision"
    step_logger.summary(f"Plan review complete. Status: {status}. {len(issues)} issues, {len(warnings)} warnings.")

    # Generate natural language summary (Prompt 7 Step 5)
    natural_language_summary = _generate_plan_summary(
        approved=approved,
        issues=issues,
        warnings=warnings,
        required_changes=required_changes,
        feature_count=len(feature_columns),
        variant_count=len(variants),
        context_validation=context_validation,
    )

    result = {
        # Core approval status (Prompt 7 Step 5 structured format)
        "approved": approved,
        "warnings": warnings,
        "required_changes": required_changes,
        "natural_language_summary": natural_language_summary,
        # Additional details
        "status": status,
        "issues": issues,  # Keep for backward compatibility
        "feature_count": len(feature_columns),
        "variant_count": len(variants),
        "context_validation": {
            "split_valid": context_validation.get("split_validation", {}).get("valid", True),
            "metric_valid": context_validation.get("metric_validation", {}).get("valid", True),
            "leakage_valid": context_validation.get("leakage_validation", {}).get("valid", True),
        },
        # Context factors used (Prompt 7 Step 7)
        "context_factors_used": context_factors,
    }

    # Include revision feedback for backward compatibility
    if not approved and revision_feedback:
        result["revision_feedback"] = revision_feedback
        step_logger.info(f"📝 Plan requires revision. {len(required_changes)} issue(s) need to be addressed.")

    return result


async def handle_results_interpretation_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the results interpretation step.

    This step analyzes experiment results and provides a summary with recommendations.

    Input JSON should contain:
    - experiment_id: UUID of the experiment to interpret

    Returns:
        Dict with results_summary, recommendation (recommended_model_id, reason), natural_language_summary
    """
    from pydantic import BaseModel, Field
    from typing import Optional as Opt

    input_data = step.input_json or {}
    experiment_id = input_data.get("experiment_id")

    if not experiment_id:
        raise ValueError("Missing 'experiment_id' in input_json")

    step_logger.info(f"Loading experiment {experiment_id} for interpretation...")

    # Load experiment with trials and model versions
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")

    step_logger.info(f"Experiment: {experiment.name or 'Unnamed'} (status: {experiment.status})")

    # Load trials
    trials = db.query(Trial).filter(Trial.experiment_id == experiment_id).all()
    step_logger.thought(f"Found {len(trials)} trial(s)")

    # Load model versions
    model_versions = db.query(ModelVersion).filter(ModelVersion.experiment_id == experiment_id).all()
    step_logger.thought(f"Found {len(model_versions)} model version(s)")

    # Build leaderboard data for LLM
    # Note: AutoGluon returns negative values for error metrics (like RMSE) internally
    # so that "higher is better" works consistently. We normalize these for display.
    error_metrics = {"root_mean_squared_error", "mean_squared_error", "mean_absolute_error", "rmse", "mse", "mae"}

    leaderboard_data = []
    for mv in model_versions:
        raw_metrics = mv.metrics_json or {}
        # Normalize negative error metrics to positive for display
        metrics = {}
        for key, value in raw_metrics.items():
            if isinstance(value, (int, float)):
                # Convert negative error metrics to positive
                if key.lower() in error_metrics and value < 0:
                    metrics[key] = abs(value)
                else:
                    metrics[key] = value
            else:
                metrics[key] = value

        leaderboard_data.append({
            "model_id": str(mv.id),
            "model_name": mv.name,
            "model_type": mv.model_type,
            "metrics": metrics,
            "trial_id": str(mv.trial_id) if mv.trial_id else None,
        })

    # Sort by primary metric if available
    primary_metric = experiment.primary_metric
    if primary_metric:
        # Try to sort by primary metric (higher is better for most metrics)
        def get_metric_value(item):
            return item["metrics"].get(primary_metric, 0) or 0
        leaderboard_data.sort(key=get_metric_value, reverse=True)

    step_logger.info(f"Built leaderboard with {len(leaderboard_data)} models")

    if leaderboard_data:
        top_model = leaderboard_data[0]
        step_logger.thought(f"Top model: {top_model['model_name']} with {primary_metric}={top_model['metrics'].get(primary_metric, 'N/A')}")

    # Build trial summaries
    trial_summaries = []
    for trial in trials:
        trial_summaries.append({
            "trial_id": str(trial.id),
            "variant_name": trial.variant_name,
            "status": trial.status,
            "metrics": trial.metrics_json or {},
            "best_model_ref": trial.best_model_ref,
        })

    # Define response schema
    class ResultsRecommendation(BaseModel):
        recommended_model_id: str = Field(description="UUID of the recommended model")
        reason: str = Field(description="Why this model is recommended")

    class ResultsInterpretationResponse(BaseModel):
        results_summary: str = Field(description="Summary of the experiment results and leaderboard")
        recommendation: ResultsRecommendation = Field(description="Model recommendation")
        natural_language_summary: str = Field(description="A comprehensive natural language summary for end users")

    # Get dataset info from dataset spec if available
    dataset_info = ""
    if experiment.dataset_spec:
        ds = experiment.dataset_spec
        feature_count = len(ds.feature_columns) if ds.feature_columns else 0
        dataset_info = f"""
Dataset Configuration:
- Target Column: {ds.target_column or 'unknown'}
- Feature Columns: {feature_count} features
- Features: {', '.join(ds.feature_columns[:10]) if ds.feature_columns else 'N/A'}{'...' if ds.feature_columns and len(ds.feature_columns) > 10 else ''}
"""

    # Build prompt for LLM using centralized prompts
    task_type = experiment.experiment_plan_json.get('task_type', 'unknown') if experiment.experiment_plan_json else 'unknown'
    prompt = get_results_interpretation_prompt(
        experiment_name=experiment.name or 'Unnamed',
        task_type=task_type,
        primary_metric=primary_metric or _infer_metric_from_task(task_type),
        status=str(experiment.status),
        dataset_info=dataset_info,
        trial_summaries=str(trial_summaries),
        leaderboard_data=str(leaderboard_data),
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_ML_ANALYST},
        {"role": "user", "content": prompt},
    ]

    step_logger.info("Consulting LLM for results interpretation...")
    response = await llm_client.chat_json(messages, ResultsInterpretationResponse)

    step_logger.summary(f"Analysis complete. Recommended model: {response.get('recommendation', {}).get('recommended_model_id', 'N/A')}")

    return {
        "results_summary": response.get("results_summary", ""),
        "recommendation": response.get("recommendation", {}),
        "natural_language_summary": response.get("natural_language_summary", ""),
        "leaderboard": leaderboard_data,
        "trial_summaries": trial_summaries,
    }


async def handle_results_critic_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the results critic step.

    This step reviews experiment results for potential issues like overfitting,
    data leakage, or other problems.

    Input JSON should contain:
    - experiment_id: UUID of the experiment to critique
    - results_interpretation: Output from results interpretation step (optional)

    Returns:
        Dict with critic_findings (severity, issues, approved), natural_language_summary
    """
    from pydantic import BaseModel, Field
    from typing import List as ListType

    input_data = step.input_json or {}
    experiment_id = input_data.get("experiment_id")
    results_interpretation = input_data.get("results_interpretation", {})

    if not experiment_id:
        raise ValueError("Missing 'experiment_id' in input_json")

    step_logger.info(f"Loading experiment {experiment_id} for critique...")

    # Load experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")

    # Load dataset spec for row count
    dataset_spec = experiment.dataset_spec
    row_count = None
    feature_count = None
    if dataset_spec:
        # Try to get row count from original data source or stored info
        if hasattr(dataset_spec, 'row_count'):
            row_count = dataset_spec.row_count
        feature_count = len(dataset_spec.feature_columns) if dataset_spec.feature_columns else None

    step_logger.thought(f"Dataset: {row_count or 'unknown'} rows, {feature_count or 'unknown'} features")

    # Load model versions with their metrics
    model_versions = db.query(ModelVersion).filter(ModelVersion.experiment_id == experiment_id).all()
    step_logger.info(f"Analyzing {len(model_versions)} model(s) for potential issues...")

    # Build model details for LLM
    model_details = []
    for mv in model_versions:
        metrics = mv.metrics_json or {}
        feature_importances = mv.feature_importances_json or {}
        model_details.append({
            "model_id": str(mv.id),
            "model_name": mv.name,
            "model_type": mv.model_type,
            "metrics": metrics,
            "feature_importances": feature_importances,
        })

    # Check for obvious issues programmatically
    issues_found = []
    warnings_found = []

    # Check for suspiciously perfect metrics
    for md in model_details:
        metrics = md["metrics"]
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                if metric_value == 1.0 and metric_name in ["accuracy", "auc", "f1", "r2"]:
                    issues_found.append(f"Model '{md['model_name']}' has perfect {metric_name}=1.0 - possible data leakage or overfitting")
                    step_logger.warning(f"Suspicious: {md['model_name']} has {metric_name}=1.0")

    # Check for data size issues
    if row_count and row_count < 100:
        warnings_found.append(f"Small dataset ({row_count} rows) - results may not generalize well")
        step_logger.warning(f"Small dataset: {row_count} rows")

    if feature_count and row_count and feature_count > row_count / 10:
        warnings_found.append(f"High feature-to-sample ratio ({feature_count} features, {row_count} rows) - risk of overfitting")
        step_logger.warning(f"High feature ratio: {feature_count} features for {row_count} rows")

    # Define response schema
    class CriticIssue(BaseModel):
        issue: str = Field(description="Description of the issue")
        severity: str = Field(description="Severity: critical, warning, or info")
        recommendation: str = Field(description="Recommended action to address the issue")

    class CriticFindings(BaseModel):
        severity: str = Field(description="Overall severity: critical, warning, or ok")
        issues: ListType[CriticIssue] = Field(description="List of identified issues")
        approved: bool = Field(description="Whether the results are approved for production use")

    class ResultsCriticResponse(BaseModel):
        critic_findings: CriticFindings = Field(description="Detailed findings from the critique")
        natural_language_summary: str = Field(description="Summary of the critique for end users")

    # Build prompt for LLM using centralized prompt
    prompt = get_results_critic_prompt(
        experiment_name=experiment.name or 'Unnamed',
        primary_metric=experiment.primary_metric or 'unknown',
        status=experiment.status,
        row_count=row_count,
        feature_count=feature_count,
        model_details=model_details,
        issues_found=issues_found,
        warnings_found=warnings_found,
        results_summary=results_interpretation.get('results_summary', 'Not available'),
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_MODEL_REVIEWER},
        {"role": "user", "content": prompt},
    ]

    step_logger.info("Consulting LLM for results critique...")
    response = await llm_client.chat_json(messages, ResultsCriticResponse)

    findings = response.get("critic_findings", {})
    severity = findings.get("severity", "ok")
    approved = findings.get("approved", True)
    issue_count = len(findings.get("issues", []))

    if severity == "critical":
        step_logger.error(f"Critical issues found: {issue_count} issues, NOT approved")
    elif severity == "warning":
        step_logger.warning(f"Warnings found: {issue_count} issues, approved={approved}")
    else:
        step_logger.summary(f"Critique complete: {severity} severity, {issue_count} issues, approved={approved}")

    return {
        "critic_findings": findings,
        "natural_language_summary": response.get("natural_language_summary", ""),
    }


async def handle_dataset_discovery_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the dataset discovery step.

    This step searches for relevant public datasets based on a problem description
    and presents them as options for the user.

    Input JSON should contain:
    - project_description: Text description of the ML problem (e.g., "I want to predict used car prices in the US")
    - constraints: Optional dict with:
        - geography: Geographic region constraint (e.g., "US", "Europe")
        - allow_public_data: Whether to search public data sources (default: True)

    Returns:
        Dict with:
        - discovered_datasets: List of dataset candidates with metadata
        - natural_language_summary: Summary of findings for the user
    """
    from pydantic import BaseModel, Field
    from typing import List as ListType

    input_data = step.input_json or {}

    project_description = input_data.get("project_description", "")
    if not project_description:
        raise ValueError("Missing 'project_description' in input_json")

    constraints = input_data.get("constraints", {})
    geography = constraints.get("geography", "")
    allow_public_data = constraints.get("allow_public_data", True)

    step_logger.info(f"Searching for datasets for: {project_description[:100]}...")

    if geography:
        step_logger.thought(f"Geographic constraint: {geography}")

    if not allow_public_data:
        step_logger.warning("Public data sources are disabled - search may be limited")

    # Define the schema for dataset discovery
    class DatasetSchemaInfo(BaseModel):
        rows_estimate: int = Field(description="Estimated number of rows in the dataset")
        columns: ListType[str] = Field(description="List of column names")
        target_candidate: str = Field(description="Best guess for which column could be the prediction target")

    class DiscoveredDataset(BaseModel):
        name: str = Field(description="Name of the dataset")
        source_url: str = Field(description="URL where the dataset can be found")
        schema_summary: DatasetSchemaInfo = Field(description="Summary of the dataset schema")
        licensing: str = Field(description="License type (e.g., CC BY 4.0, MIT, Proprietary, Unknown)")
        fit_for_purpose: str = Field(description="Assessment of how well this dataset fits the user's needs")

    class DatasetDiscoveryResponse(BaseModel):
        discovered_datasets: ListType[DiscoveredDataset] = Field(
            description="List of discovered datasets with metadata"
        )
        natural_language_summary: str = Field(
            description="Summary of findings for the user"
        )

    # Build prompt for LLM to search and recommend datasets using centralized prompt
    geography_constraint = f"\n- Geographic focus: {geography}" if geography else ""
    public_data_note = "" if allow_public_data else "\n- Note: User prefers private/proprietary data sources only"

    step_logger.info("Consulting LLM to search for relevant datasets...")

    prompt = get_dataset_discovery_prompt(
        project_description=project_description,
        geography_constraint=geography_constraint,
        public_data_note=public_data_note,
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_DATASET_EXPERT},
        {"role": "user", "content": prompt},
    ]

    step_logger.thought("Searching dataset repositories for verified, high-quality datasets...")

    response = await llm_client.chat_json(messages, DatasetDiscoveryResponse)

    discovered_datasets = response.get("discovered_datasets", [])
    natural_language_summary = response.get("natural_language_summary", "")

    # Log what was found
    if discovered_datasets:
        step_logger.info(f"Found {len(discovered_datasets)} potential dataset(s)")
        for i, ds in enumerate(discovered_datasets, 1):
            name = ds.get("name", "Unknown")
            licensing = ds.get("licensing", "Unknown")
            schema = ds.get("schema_summary", {})
            rows = schema.get("rows_estimate", 0)
            target = schema.get("target_candidate", "N/A")
            step_logger.thought(
                f"Dataset {i}: {name} (~{rows:,} rows, target: {target}, license: {licensing})"
            )
    else:
        step_logger.warning("No suitable datasets found")

    step_logger.summary(
        f"Dataset discovery complete. Found {len(discovered_datasets)} dataset(s). "
        f"Summary: {natural_language_summary[:100]}..."
    )

    return {
        "discovered_datasets": discovered_datasets,
        "natural_language_summary": natural_language_summary,
    }


async def handle_training_dataset_planning_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the training dataset planning step.

    This step uses relationship discovery results and data profiles to propose
    a TrainingDatasetSpec - a complete plan for building a training dataset.

    Input JSON should contain:
    - project_description: User's description of the ML goal
    - target_hint: Optional hint about which column is the target
    - data_source_profiles: List of data source profile summaries
    - relationships_summary: Output from relationship discovery service

    Returns:
        Dict with training_dataset_spec and natural_language_summary
    """
    from pydantic import BaseModel, Field
    from typing import List as ListType, Optional as Opt

    input_data = step.input_json or {}

    project_description = input_data.get("project_description", "")
    if not project_description:
        raise ValueError("Missing 'project_description' in input_json")

    target_hint = input_data.get("target_hint")
    data_source_profiles = input_data.get("data_source_profiles", [])
    relationships_summary = input_data.get("relationships_summary", {})

    if not data_source_profiles:
        raise ValueError("Missing 'data_source_profiles' in input_json")

    if not relationships_summary:
        raise ValueError("Missing 'relationships_summary' in input_json")

    step_logger.info(f"Planning training dataset for: {project_description[:100]}...")

    # Extract key information
    tables = relationships_summary.get("tables", [])
    relationships = relationships_summary.get("relationships", [])
    base_table_candidates = relationships_summary.get("base_table_candidates", [])

    step_logger.info(f"Evaluating {len(tables)} tables as potential base tables...")

    # Log base table candidates
    for candidate in base_table_candidates[:5]:
        table_name = candidate.get("table", "unknown")
        score = candidate.get("score", 0)
        reasons = candidate.get("reasons", [])
        step_logger.thought(
            f"Considering {table_name} as base table (score: {score:.2f}): "
            f"{', '.join(reasons[:2]) if reasons else 'no specific reasons'}"
        )

    # Build table summaries for the prompt
    table_summaries = []
    for table in tables:
        summary = {
            "name": table.get("table_name", ""),
            "source_name": table.get("source_name", ""),
            "row_count": table.get("row_count", 0),
            "column_count": table.get("column_count", 0),
            "id_columns": [c.get("name") for c in table.get("id_columns", [])],
            "target_columns": [c.get("name") for c in table.get("target_columns", [])],
            "has_obvious_id": table.get("has_obvious_id", False),
            "has_potential_target": table.get("has_potential_target", False),
        }
        table_summaries.append(summary)

    # Build relationship summaries for the prompt
    relationship_summaries = []
    for rel in relationships:
        rel_summary = {
            "from_table": rel.get("from_table", ""),
            "to_table": rel.get("to_table", ""),
            "from_column": rel.get("from_column", ""),
            "to_column": rel.get("to_column", ""),
            "cardinality": rel.get("cardinality", "unknown"),
        }
        relationship_summaries.append(rel_summary)

    step_logger.info(f"Found {len(relationships)} relationship(s) between tables")

    # Define response schema for LLM
    class LLMBaseFilter(BaseModel):
        column: str = Field(..., description="Column to filter on")
        operator: str = Field(..., description="Filter operator")
        value: Any = Field(None, description="Value to compare against")

    class LLMTargetDefinition(BaseModel):
        table: str = Field(..., description="Table containing the target")
        column: str = Field(..., description="Column to predict")
        join_key: Opt[str] = Field(None, description="Join key if target is in different table")
        label_window_days: Opt[int] = Field(None, description="Days forward for time-based targets")

    class LLMAggFeature(BaseModel):
        name: str = Field(..., description="Feature name")
        agg: str = Field(..., description="Aggregation: sum, count, avg, min, max")
        column: str = Field(..., description="Column to aggregate")

    class LLMJoinAggregation(BaseModel):
        window_days: Opt[int] = Field(None, description="Time window in days")
        features: ListType[LLMAggFeature] = Field(default_factory=list)

    class LLMJoinPlanItem(BaseModel):
        from_table: str = Field(..., description="Source table")
        to_table: str = Field(..., description="Target table")
        left_key: str = Field(..., description="Key in source table")
        right_key: str = Field(..., description="Key in target table")
        relationship: str = Field(..., description="one_to_one, one_to_many, many_to_one")
        aggregation: Opt[LLMJoinAggregation] = Field(None)

    class LLMTrainingDatasetSpec(BaseModel):
        base_table: str = Field(..., description="Base table name")
        base_filters: ListType[LLMBaseFilter] = Field(default_factory=list)
        target_definition: LLMTargetDefinition
        join_plan: ListType[LLMJoinPlanItem] = Field(default_factory=list)
        excluded_tables: ListType[str] = Field(default_factory=list)
        excluded_columns: ListType[str] = Field(default_factory=list)

    class LLMPlanningResponse(BaseModel):
        training_dataset_spec: LLMTrainingDatasetSpec
        natural_language_summary: str = Field(
            ...,
            description="Explanation of the plan for the user"
        )

    # Build prompt using centralized prompt
    base_table_candidates_formatted = str([{
        "table": c.get("table"),
        "score": c.get("score"),
        "reasons": c.get("reasons", []),
        "target_columns": c.get("target_columns", []),
    } for c in base_table_candidates[:5]])

    prompt = get_training_dataset_planning_prompt(
        project_description=project_description,
        target_hint=target_hint or "",
        table_summaries=str(table_summaries),
        base_table_candidates=base_table_candidates_formatted,
        relationship_summaries=str(relationship_summaries),
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_DATASET_DESIGNER},
        {"role": "user", "content": prompt},
    ]

    # Retry logic with error feedback (similar to visualization generation)
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            step_logger.info(f"Consulting LLM for training dataset plan (attempt {attempt + 1}/{max_retries})...")

            # Add error feedback to prompt on retries
            if last_error and attempt > 0:
                retry_messages = messages.copy()
                retry_messages.append({
                    "role": "assistant",
                    "content": "I'll generate the training dataset specification."
                })
                retry_messages.append({
                    "role": "user",
                    "content": f"Your previous response caused an error: {last_error}\n\nPlease fix the issue and provide a valid response. Make sure all nested objects are proper dictionaries, not strings."
                })
                response = await llm_client.chat_json(retry_messages, LLMPlanningResponse)
            else:
                response = await llm_client.chat_json(messages, LLMPlanningResponse)

            # Extract and validate the response - with safety checks for string values
            spec_data = response.get("training_dataset_spec", {})
            if isinstance(spec_data, str):
                raise ValueError(f"training_dataset_spec is a string, expected dict: {spec_data[:100]}")

            natural_language_summary = response.get("natural_language_summary", "")
            if not isinstance(natural_language_summary, str):
                natural_language_summary = str(natural_language_summary) if natural_language_summary else ""

            # Log what was decided - with safety checks
            base_table = spec_data.get("base_table", "unknown") if isinstance(spec_data, dict) else "unknown"
            target_def = spec_data.get("target_definition", {}) if isinstance(spec_data, dict) else {}
            if isinstance(target_def, str):
                raise ValueError(f"target_definition is a string, expected dict: {target_def[:100]}")

            join_plan = spec_data.get("join_plan", []) if isinstance(spec_data, dict) else []
            if isinstance(join_plan, str):
                raise ValueError(f"join_plan is a string, expected list: {join_plan[:100]}")

            step_logger.thought(f"Selected base table: {base_table}")

            # Safely access target_def fields
            target_table = target_def.get('table', '') if isinstance(target_def, dict) else ''
            target_column = target_def.get('column', '') if isinstance(target_def, dict) else ''
            step_logger.thought(f"Target: {target_table}.{target_column}")

            if join_plan and isinstance(join_plan, list):
                step_logger.info(f"Join plan includes {len(join_plan)} table(s)")
                for join in join_plan:
                    if not isinstance(join, dict):
                        step_logger.thought(f"  Skipping invalid join entry: {type(join)}")
                        continue
                    agg = join.get("aggregation")
                    # Handle case where LLM returns aggregation as a string instead of dict
                    if agg and isinstance(agg, dict) and agg.get("features"):
                        features = agg.get("features", [])
                        feature_count = len(features) if isinstance(features, list) else 0
                        step_logger.thought(
                            f"  {join.get('from_table', '?')} -> {join.get('to_table', '?')}: "
                            f"{join.get('relationship', '?')} ({feature_count} aggregated features)"
                        )
                    else:
                        step_logger.thought(
                            f"  {join.get('from_table', '?')} -> {join.get('to_table', '?')}: "
                            f"{join.get('relationship', '?')} (direct join)"
                        )

            excluded_tables = spec_data.get("excluded_tables", []) if isinstance(spec_data, dict) else []
            if excluded_tables and isinstance(excluded_tables, list):
                step_logger.info(f"Excluding tables: {', '.join(str(t) for t in excluded_tables)}")

            step_logger.summary(
                f"Training dataset plan complete. Base table: {base_table}. "
                f"Target: {target_column or 'unknown'}. "
                f"Joins: {len(join_plan) if isinstance(join_plan, list) else 0}."
            )

            return {
                "training_dataset_spec": spec_data,
                "natural_language_summary": natural_language_summary,
            }

        except Exception as e:
            last_error = str(e)
            step_logger.thought(f"Attempt {attempt + 1} failed: {last_error}")
            if attempt == max_retries - 1:
                step_logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
                raise ValueError(f"Training dataset planning failed after {max_retries} attempts: {last_error}")


async def handle_training_dataset_build_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the training dataset build step.

    This step materializes a training dataset from a TrainingDatasetSpec.

    Input JSON should contain:
    - project_id: UUID of the project
    - training_dataset_spec: The TrainingDatasetSpec dictionary
    - max_rows: Optional maximum rows (default 1_000_000)
    - output_format: Optional output format - "parquet" or "csv" (default "parquet")

    Returns:
        Dict with data_source_id, row_count, column_count, target_column, feature_columns
    """
    input_data = step.input_json or {}

    project_id = input_data.get("project_id")
    if not project_id:
        raise ValueError("Missing 'project_id' in input_json")

    training_dataset_spec = input_data.get("training_dataset_spec")
    if not training_dataset_spec:
        raise ValueError("Missing 'training_dataset_spec' in input_json")

    max_rows = input_data.get("max_rows", 1_000_000)
    output_format = input_data.get("output_format", "parquet")

    step_logger.info("Starting training dataset materialization...")

    # Extract key info for logging
    base_table = training_dataset_spec.get("base_table", "unknown")
    target_def = training_dataset_spec.get("target_definition", {})
    target_column = target_def.get("column", "unknown")
    join_plan = training_dataset_spec.get("join_plan", [])

    step_logger.thought(f"Base table: {base_table}")
    step_logger.thought(f"Target column: {target_column}")
    step_logger.thought(f"Join plan: {len(join_plan)} table(s)")

    # Get time-based metadata from problem_understanding step in same agent run
    is_time_based = False
    time_column = None
    entity_id_column = None
    prediction_horizon = None
    target_positive_class = None

    if step.agent_run_id:
        problem_step = db.query(AgentStep).filter(
            AgentStep.agent_run_id == step.agent_run_id,
            AgentStep.step_type == AgentStepType.PROBLEM_UNDERSTANDING,
            AgentStep.status == AgentStepStatus.COMPLETED,
        ).first()

        if problem_step and problem_step.output_json:
            problem_output = problem_step.output_json
            is_time_based = problem_output.get("is_time_based", False)
            time_column = problem_output.get("time_column")
            entity_id_column = problem_output.get("entity_id_column")
            prediction_horizon = problem_output.get("prediction_horizon")
            target_positive_class = problem_output.get("target_positive_class")

            if is_time_based:
                step_logger.thought(f"Time-based task detected: time_column='{time_column}', horizon='{prediction_horizon}'")

    step_logger.info(f"Materializing dataset from '{base_table}' with {len(join_plan)} joins...")

    try:
        # Materialize the dataset (pass step_logger for sampling logs)
        result: MaterializationResult = materialize_training_dataset(
            db=db,
            project_id=UUID(project_id) if isinstance(project_id, str) else project_id,
            training_dataset_spec=training_dataset_spec,
            max_rows=max_rows,
            output_format=output_format,
            step_logger=step_logger,
            # Time-based task metadata
            is_time_based=is_time_based,
            time_column=time_column,
            entity_id_column=entity_id_column,
            prediction_horizon=prediction_horizon,
            target_positive_class=target_positive_class,
        )

        # Load the created data source to get details
        data_source = db.query(DataSource).filter(DataSource.id == result.data_source_id).first()

        if data_source and data_source.schema_summary:
            columns = [c.get("name") for c in data_source.schema_summary.get("columns", [])]
            feature_columns = [c for c in columns if c != target_column]

            # Log sampling information if dataset was sampled
            if result.was_sampled and result.sampling_message:
                step_logger.warning(result.sampling_message)

            step_logger.info(f"Dataset created: {result.row_count:,} rows, {result.column_count} columns")

            # Build summary message
            summary_parts = [
                f"Training dataset materialized successfully.",
                f"DataSource ID: {result.data_source_id}.",
                f"Size: {result.row_count:,} rows, {len(feature_columns)} features, target: {target_column}",
            ]
            if result.was_sampled:
                summary_parts.append(f"(Sampled from {result.original_row_count:,} total rows)")

            step_logger.summary(" ".join(summary_parts))

            return {
                "data_source_id": str(result.data_source_id),
                "row_count": result.row_count,
                "column_count": result.column_count,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "output_format": output_format,
                "was_sampled": result.was_sampled,
                "original_row_count": result.original_row_count,
                "sampling_message": result.sampling_message,
            }
        else:
            step_logger.warning("Data source created but schema_summary not available")
            return {
                "data_source_id": str(result.data_source_id),
                "target_column": target_column,
                "output_format": output_format,
            }

    except Exception as e:
        step_logger.error(f"Failed to materialize training dataset: {str(e)}")
        raise


async def handle_dataset_inventory_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the dataset inventory step.

    This step profiles all data sources in a project to create an inventory
    of available datasets for the Data Architect pipeline.

    Input JSON should contain:
    - project_id: UUID of the project

    Returns:
        Dict with data_source_profiles list
    """
    input_data = step.input_json or {}

    project_id = input_data.get("project_id")
    if not project_id:
        raise ValueError("Missing 'project_id' in input_json")

    step_logger.info("Starting dataset inventory...")

    # Get the project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    # Get all data sources for the project
    data_sources = db.query(DataSource).filter(
        DataSource.project_id == project_id
    ).all()

    if not data_sources:
        step_logger.warning("No data sources found in project")
        return {
            "data_source_profiles": [],
            "total_sources": 0,
            "profiled_count": 0,
        }

    step_logger.info(f"Found {len(data_sources)} data source(s) to profile")

    # Profile all data sources
    profiles = []
    errors = []

    for ds in data_sources:
        step_logger.thought(f"Profiling data source: {ds.name}")
        try:
            profile = profile_data_source(db, ds.id)
            profiles.append(profile)

            # Also update the data source's profile_json for future use
            ds.profile_json = profile
            db.commit()

            step_logger.info(
                f"Profiled '{ds.name}': {profile.get('estimated_row_count', 0):,} rows, "
                f"{profile.get('column_count', 0)} columns"
            )
        except Exception as e:
            error_msg = f"Failed to profile '{ds.name}': {str(e)}"
            step_logger.warning(error_msg)
            errors.append({
                "source_id": str(ds.id),
                "source_name": ds.name,
                "error": str(e),
            })

    step_logger.summary(
        f"Dataset inventory complete. Profiled {len(profiles)}/{len(data_sources)} data sources."
    )

    return {
        "data_source_profiles": profiles,
        "total_sources": len(data_sources),
        "profiled_count": len(profiles),
        "error_count": len(errors),
        "errors": errors if errors else None,
    }


async def handle_relationship_discovery_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the relationship discovery step.

    This step discovers relationships between data sources in a project.
    The profiles should already be saved to each DataSource's profile_json
    from the previous DATASET_INVENTORY step.

    Input JSON should contain:
    - project_id: UUID of the project
    - data_source_profiles: List of profiles from dataset_inventory step (for reference)

    Returns:
        Dict with tables, relationships, base_table_candidates
    """
    input_data = step.input_json or {}

    project_id = input_data.get("project_id")
    if not project_id:
        raise ValueError("Missing 'project_id' in input_json")

    data_source_profiles = input_data.get("data_source_profiles", [])
    profile_count = len(data_source_profiles) if data_source_profiles else 0

    step_logger.info("Starting relationship discovery...")
    step_logger.thought(f"Analyzing {profile_count} data source profile(s)")

    # Run relationship discovery (uses profile_json from DataSources)
    try:
        project_uuid = UUID(project_id) if isinstance(project_id, str) else project_id
        relationships_result = discover_relationships_for_project(db, project_uuid)

        tables = relationships_result.get("tables", [])
        relationships = relationships_result.get("relationships", [])
        base_candidates = relationships_result.get("base_table_candidates", [])

        step_logger.info(f"Found {len(tables)} table(s)")
        step_logger.info(f"Discovered {len(relationships)} relationship(s)")

        # Log some details about relationships
        for rel in relationships[:5]:  # Log first 5 relationships
            step_logger.thought(
                f"  {rel.get('from_table')}.{rel.get('from_column')} -> "
                f"{rel.get('to_table')}.{rel.get('to_column')} "
                f"({rel.get('relationship_type', 'unknown')})"
            )
        if len(relationships) > 5:
            step_logger.thought(f"  ... and {len(relationships) - 5} more")

        # Log base table candidates
        step_logger.info(f"Identified {len(base_candidates)} base table candidate(s)")
        for candidate in base_candidates[:3]:
            step_logger.thought(
                f"  {candidate.get('table')}: score={candidate.get('score', 0):.2f} "
                f"- {', '.join(candidate.get('reasons', ['unknown']))}"
            )

        step_logger.summary(
            f"Relationship discovery complete. "
            f"{len(tables)} tables, {len(relationships)} relationships, "
            f"{len(base_candidates)} base table candidates."
        )

        return {
            "tables": tables,
            "relationships": relationships,
            "base_table_candidates": base_candidates,
            "relationships_summary": relationships_result,
        }

    except Exception as e:
        step_logger.error(f"Relationship discovery failed: {str(e)}")
        raise


# ============================================
# Lab Notebook Summary Handler
# ============================================

async def handle_lab_notebook_summary_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the lab notebook summary step.

    This step creates a comprehensive summary of a research cycle, documenting
    what was attempted, the results, and proposed next directions.

    Input JSON should contain:
    - research_cycle_id: UUID of the research cycle to summarize
    - project_id: UUID of the project (fallback if cycle not found)

    Returns:
        Dict with:
        - lab_note: Dict containing title and body_markdown
        - lab_notebook_entry_id: ID of the created notebook entry
    """
    from pydantic import BaseModel, Field

    input_data = step.input_json or {}
    research_cycle_id = input_data.get("research_cycle_id")
    project_id = input_data.get("project_id")

    # Validate required input
    if not research_cycle_id:
        # Try to get cycle from the agent run
        agent_run = step.agent_run
        if agent_run and agent_run.research_cycle_id:
            research_cycle_id = str(agent_run.research_cycle_id)
        else:
            raise ValueError("Missing 'research_cycle_id' in step input")

    step_logger.action("Starting lab notebook summary generation...")

    # Load the research cycle
    cycle = db.query(ResearchCycle).filter(ResearchCycle.id == research_cycle_id).first()

    if not cycle:
        raise ValueError(f"Research cycle not found: {research_cycle_id}")

    step_logger.thinking(f"Summarizing research cycle #{cycle.sequence_number}")

    # Load the project
    project = db.query(Project).filter(Project.id == cycle.project_id).first()
    if not project:
        raise ValueError(f"Project not found for cycle {cycle.id}")

    project_id = project.id
    step_logger.info(f"Project: {project.name}")

    # Load experiments linked to this cycle
    cycle_experiments = (
        db.query(CycleExperiment)
        .filter(CycleExperiment.research_cycle_id == cycle.id)
        .all()
    )

    experiment_ids = [ce.experiment_id for ce in cycle_experiments]
    experiments = (
        db.query(Experiment)
        .filter(Experiment.id.in_(experiment_ids))
        .all()
    ) if experiment_ids else []

    step_logger.info(f"Found {len(experiments)} experiments in this cycle")

    # Build experiments summary
    experiments_summary_lines = []
    best_model_info_lines = []
    best_score = None
    best_metric_name = None
    best_model_name = None

    for exp in experiments:
        # Calculate best metric from trials
        exp_best_metric = None
        exp_best_model = None
        if exp.trials:
            for trial in exp.trials:
                if trial.metrics_json and exp.primary_metric:
                    metric_val = trial.metrics_json.get(exp.primary_metric)
                    if metric_val is not None:
                        if exp_best_metric is None:
                            exp_best_metric = metric_val
                            exp_best_model = trial.best_model_ref
                        else:
                            # Assume maximize for now (can be improved with metric_direction)
                            if metric_val > exp_best_metric:
                                exp_best_metric = metric_val
                                exp_best_model = trial.best_model_ref

        exp_line = f"- **{exp.name}** (Status: {exp.status.value if hasattr(exp.status, 'value') else exp.status})"
        if exp_best_metric is not None:
            exp_line += f" - Best {exp.primary_metric}: {exp_best_metric:.4f}"
        experiments_summary_lines.append(exp_line)

        # Track best model across experiments
        if exp_best_metric is not None:
            if best_score is None or (
                # For error metrics (lower is better), compare appropriately
                "error" in (exp.primary_metric or "").lower() or
                "loss" in (exp.primary_metric or "").lower()
            ):
                if best_score is None or exp_best_metric < best_score:
                    best_score = exp_best_metric
                    best_metric_name = exp.primary_metric
                    best_model_name = exp_best_model
            else:
                if exp_best_metric > best_score:
                    best_score = exp_best_metric
                    best_metric_name = exp.primary_metric
                    best_model_name = exp_best_model

    if not experiments_summary_lines:
        experiments_summary_lines.append("No experiments completed in this cycle yet.")

    experiments_summary = "\n".join(experiments_summary_lines)

    # Build best model info
    if best_score is not None:
        best_model_info_lines.append(f"- **Best Model**: {best_model_name or 'Unknown'}")
        best_model_info_lines.append(f"- **{best_metric_name}**: {best_score:.4f}")
    else:
        best_model_info_lines.append("No model results available yet.")

    best_model_info = "\n".join(best_model_info_lines)

    # Collect outputs from other agent steps in this cycle's runs
    step_logger.thinking("Gathering insights from agent steps in this cycle...")

    agent_runs = (
        db.query(AgentRun)
        .filter(AgentRun.research_cycle_id == cycle.id)
        .all()
    )

    step_outputs_lines = []
    for run in agent_runs:
        if run.steps:
            for s in run.steps:
                if s.output_json and s.status == AgentStepStatus.COMPLETED:
                    # Extract key summaries from different step types
                    output = s.output_json
                    step_type = s.step_type

                    if step_type == AgentStepType.DATA_ANALYSIS:
                        summary = output.get("natural_language_summary", "")
                        if summary:
                            step_outputs_lines.append(f"**Data Analysis**: {summary[:500]}...")

                    elif step_type == AgentStepType.DATASET_DESIGN:
                        summary = output.get("natural_language_summary", "")
                        if summary:
                            step_outputs_lines.append(f"**Dataset Design**: {summary[:500]}...")

                    elif step_type == AgentStepType.EXPERIMENT_DESIGN:
                        summary = output.get("natural_language_summary", "")
                        if summary:
                            step_outputs_lines.append(f"**Experiment Design**: {summary[:500]}...")

                    elif step_type == AgentStepType.PLAN_CRITIC:
                        summary = output.get("natural_language_summary", "")
                        if summary:
                            step_outputs_lines.append(f"**Plan Critic**: {summary[:500]}...")

                    elif step_type == AgentStepType.RESULTS_INTERPRETATION:
                        summary = output.get("natural_language_summary", "")
                        if summary:
                            step_outputs_lines.append(f"**Results Interpretation**: {summary[:500]}...")

                    elif step_type == AgentStepType.RESULTS_CRITIC:
                        findings = output.get("critic_findings", {})
                        severity = findings.get("severity", "unknown")
                        approved = findings.get("approved", False)
                        step_outputs_lines.append(
                            f"**Results Critic**: Severity={severity}, Approved={approved}"
                        )

    if not step_outputs_lines:
        step_outputs_lines.append("No agent insights available for this cycle.")

    step_outputs_summary = "\n\n".join(step_outputs_lines)

    # Get previous cycles context
    previous_cycles_context = None
    if cycle.sequence_number > 1:
        previous_entries = (
            db.query(LabNotebookEntry)
            .filter(LabNotebookEntry.project_id == project_id)
            .filter(LabNotebookEntry.author_type == LabNotebookAuthorType.AGENT)
            .order_by(LabNotebookEntry.created_at.desc())
            .limit(3)
            .all()
        )
        if previous_entries:
            context_lines = []
            for entry in reversed(previous_entries):
                context_lines.append(f"### {entry.title}\n{entry.body_markdown[:1000]}...")
            previous_cycles_context = "\n\n".join(context_lines)

    # Get problem description from project or agent run config
    problem_description = project.description or "No problem description available."
    for run in agent_runs:
        if run.config_json and run.config_json.get("description"):
            problem_description = run.config_json["description"]
            break

    # Define response schema
    class LabNotebookSummaryResponse(BaseModel):
        title: str = Field(description="Title for this cycle summary")
        body_markdown: str = Field(description="Full Markdown content of the lab notebook entry")

    # Build prompt
    prompt = get_lab_notebook_summary_prompt(
        cycle_number=cycle.sequence_number,
        cycle_title=cycle.summary_title,
        project_name=project.name,
        problem_description=problem_description,
        experiments_summary=experiments_summary,
        best_model_info=best_model_info,
        step_outputs_summary=step_outputs_summary,
        previous_cycles_context=previous_cycles_context,
    )

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_LAB_NOTEBOOK_AGENT},
        {"role": "user", "content": prompt},
    ]

    step_logger.action("Generating lab notebook summary with LLM...")
    response = await llm_client.chat_json(messages, LabNotebookSummaryResponse)

    title = response.get("title", f"Cycle {cycle.sequence_number} Summary")
    body_markdown = response.get("body_markdown", "")

    step_logger.thinking(f"Generated summary: {title}")

    # Create the lab notebook entry
    entry = LabNotebookEntry(
        project_id=project_id,
        research_cycle_id=cycle.id,
        agent_step_id=step.id,
        author_type=LabNotebookAuthorType.AGENT,
        title=title,
        body_markdown=body_markdown,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    step_logger.info(f"Created lab notebook entry: {entry.id}")

    # Update cycle summary title if not set
    if not cycle.summary_title:
        cycle.summary_title = title
        db.commit()

    step_logger.summary(f"Lab notebook entry created: '{title}'")

    return {
        "lab_note": {
            "title": title,
            "body_markdown": body_markdown,
        },
        "lab_notebook_entry_id": str(entry.id),
    }


# ============================================
# Robustness & Overfitting Audit Handler
# ============================================

async def handle_robustness_audit_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the robustness and overfitting audit step.

    This step analyzes experiments for overfitting, suspicious patterns,
    data leakage, and compares against baselines. It produces a structured report.

    Input JSON should contain:
    - project_id: UUID of the project
    - experiment_id: Optional single experiment ID to audit
    - experiment_ids: List of experiment IDs to audit (or all if not provided)
    - research_cycle_id: Optional UUID of the research cycle
    - task_type: Type of ML task (binary, multiclass, regression)
    - is_time_based: Whether this is a time-based prediction task
    - primary_metric: The metric to focus on (e.g., "roc_auc", "rmse")

    Returns:
        Dict with:
        - robustness_audit: The full audit results including:
          - overfitting_risk: "low" | "medium" | "high"
          - leakage_suspected: bool
          - time_split_suspicious: bool
          - metrics_summary: {...}
          - warnings: [...]
          - recommendations: [...]
          - natural_language_summary: str
    """
    from pydantic import BaseModel, Field

    class RobustnessAuditResponse(BaseModel):
        overfitting_risk: str = Field(..., description="Risk level: low, medium, or high")
        train_val_analysis: Dict[str, Any] = Field(default_factory=dict)
        suspicious_patterns: List[Dict[str, Any]] = Field(default_factory=list)
        baseline_comparison: Dict[str, Any] = Field(default_factory=dict)
        cv_analysis: Dict[str, Any] = Field(default_factory=dict)
        recommendations: List[str] = Field(default_factory=list)
        natural_language_summary: str = Field(default="")

    input_data = step.input_json or {}
    project_id = input_data.get("project_id")
    research_cycle_id = input_data.get("research_cycle_id")
    experiment_ids = input_data.get("experiment_ids", [])
    # Support single experiment_id as well
    single_experiment_id = input_data.get("experiment_id")
    if single_experiment_id and single_experiment_id not in experiment_ids:
        experiment_ids.append(single_experiment_id)
    primary_metric = input_data.get("primary_metric")
    task_type = input_data.get("task_type")
    is_time_based_input = input_data.get("is_time_based", False)

    step_logger.action("Starting robustness and overfitting audit...")

    # Load project
    if not project_id:
        # Try to get from agent run
        agent_run = step.agent_run
        if agent_run and agent_run.project_id:
            project_id = str(agent_run.project_id)
        else:
            raise ValueError("Missing 'project_id' in step input")

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project not found: {project_id}")

    # Use provided task_type or fall back to project's task_type
    if not task_type:
        task_type = project.task_type or "unknown"

    step_logger.info(f"Project: {project.name}")
    step_logger.info(f"Task type: {task_type}")

    # Build unified TaskContext for the robustness auditor (Prompt 7)
    task_context = None
    task_hints = {}
    context_factors = {}  # Initialized for Prompt 7 Step 7 logging
    try:
        task_context = build_task_context(
            db=db,
            project_id=str(project_id),
            research_cycle_id=str(research_cycle_id) if research_cycle_id else None,
            include_leakage_candidates=True,
            include_past_cycles=True,
            max_experiments=10,
        )
        task_hints = get_task_type_hints(task_context)
        step_logger.info("📊 Built unified TaskContext for robustness audit")

        # Log context usage with consistent helper (Prompt 7 Step 7)
        context_factors = _log_context_usage(
            step_logger=step_logger,
            task_context=task_context,
            task_hints=task_hints,
            step_name="Robustness Audit",
        )
    except Exception as e:
        step_logger.warning(f"Could not build TaskContext: {e}")

    # Determine if time-based from input or dataset spec or TaskContext
    is_time_based = is_time_based_input or task_hints.get("is_time_based", False)
    is_classification = task_type in ("binary", "multiclass", "classification")

    # Get experiments to audit
    if experiment_ids:
        # Specific experiments provided
        experiments = (
            db.query(Experiment)
            .filter(Experiment.id.in_(experiment_ids))
            .all()
        )
    elif research_cycle_id:
        # Get experiments from research cycle
        cycle_experiments = (
            db.query(CycleExperiment)
            .filter(CycleExperiment.research_cycle_id == research_cycle_id)
            .all()
        )
        exp_ids = [ce.experiment_id for ce in cycle_experiments]
        experiments = (
            db.query(Experiment)
            .filter(Experiment.id.in_(exp_ids))
            .all()
        ) if exp_ids else []
    else:
        # Get all experiments for the project
        experiments = (
            db.query(Experiment)
            .filter(Experiment.project_id == project_id)
            .all()
        )

    if not experiments:
        step_logger.warning("No experiments found to audit")
        return {
            "robustness_audit": {
                "overfitting_risk": "unknown",
                "train_val_analysis": {},
                "suspicious_patterns": [],
                "baseline_comparison": {},
                "cv_analysis": {},
                "recommendations": ["Run experiments first before auditing"],
                "natural_language_summary": "No experiments available for audit.",
            }
        }

    step_logger.info(f"Auditing {len(experiments)} experiment(s)")

    # Determine primary metric if not provided
    if not primary_metric:
        for exp in experiments:
            if exp.primary_metric:
                primary_metric = exp.primary_metric
                break
        if not primary_metric:
            primary_metric = _infer_metric_from_task(project.task_type or "classification")

    step_logger.info(f"Primary metric: {primary_metric}")

    # Collect trial data with train vs validation metrics
    trials_data_lines = []
    all_train_metrics = []
    all_val_metrics = []
    all_gaps = []
    cv_fold_data = []
    best_val_metric = None

    # Track baseline metrics and leakage/split concerns (Prompt 4 requirements)
    all_baseline_metrics: List[Dict[str, Any]] = []
    leakage_suspected = False
    leakage_warnings: List[str] = []
    time_split_suspicious = False
    time_split_warnings: List[str] = []
    warnings_list: List[str] = []

    for exp in experiments:
        step_logger.thinking(f"Analyzing experiment: {exp.name}")

        if not exp.trials:
            trials_data_lines.append(f"**{exp.name}**: No trials found")
            continue

        for trial in exp.trials:
            trial_line = f"**{exp.name} - {trial.variant_name}** (Status: {trial.status.value if hasattr(trial.status, 'value') else trial.status})"

            # Get metrics from trial
            metrics = trial.metrics_json or {}

            # Look for training metrics (common patterns)
            train_metric = None
            val_metric = None

            # Try various common metric key patterns
            metric_lower = primary_metric.lower()

            # Check for explicit train/val metrics
            for key, value in metrics.items():
                key_lower = key.lower()
                if metric_lower in key_lower:
                    if 'train' in key_lower:
                        train_metric = value
                    elif 'val' in key_lower or 'test' in key_lower:
                        val_metric = value
                    elif train_metric is None and val_metric is None:
                        # If no train/val prefix, assume it's validation
                        val_metric = value

            # If only one metric found, it's likely the validation metric
            if val_metric is None and train_metric is not None:
                val_metric = train_metric
                train_metric = None
            elif train_metric is None and val_metric is None:
                # Try to get the primary metric directly
                val_metric = metrics.get(primary_metric) or metrics.get(metric_lower)

            # Build trial data line
            if train_metric is not None and val_metric is not None:
                gap = abs(train_metric - val_metric)
                trial_line += f"\n  - Training {primary_metric}: {train_metric:.4f}"
                trial_line += f"\n  - Validation {primary_metric}: {val_metric:.4f}"
                trial_line += f"\n  - Gap: {gap:.4f}"

                all_train_metrics.append(train_metric)
                all_val_metrics.append(val_metric)
                all_gaps.append(gap)

                # Log hypothesis about this trial
                if gap > 0.15:
                    step_logger.hypothesis(
                        f"Large train-val gap ({gap:.3f}) in {trial.variant_name} suggests overfitting"
                    )
                elif gap > 0.08:
                    step_logger.hypothesis(
                        f"Moderate train-val gap ({gap:.3f}) in {trial.variant_name} - may need attention"
                    )
            elif val_metric is not None:
                trial_line += f"\n  - Validation {primary_metric}: {val_metric:.4f}"
                trial_line += "\n  - Training metrics not available"
                all_val_metrics.append(val_metric)
            else:
                trial_line += "\n  - Metrics not available"

            # Track best validation metric
            if val_metric is not None:
                if best_val_metric is None:
                    best_val_metric = val_metric
                else:
                    # Assume higher is better for now (can be improved with metric_direction)
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric

            # Check for CV fold data in metrics
            fold_metrics = []
            for key, value in metrics.items():
                if 'fold' in key.lower() and isinstance(value, (int, float)):
                    fold_metrics.append(value)

            if fold_metrics:
                cv_fold_data.extend(fold_metrics)
                fold_variance = max(fold_metrics) - min(fold_metrics) if len(fold_metrics) > 1 else 0
                trial_line += f"\n  - CV Folds: {len(fold_metrics)}, Range: {fold_variance:.4f}"

                if fold_variance > 0.1:
                    step_logger.hypothesis(
                        f"High CV variance ({fold_variance:.3f}) suggests unstable model"
                    )

            # === Prompt 4: Extract real baseline metrics from trial ===
            if trial.baseline_metrics_json:
                baseline_data = trial.baseline_metrics_json
                all_baseline_metrics.append(baseline_data)

                # Check for label-shuffle leakage
                label_shuffle = baseline_data.get("label_shuffle", {})
                if label_shuffle.get("leakage_detected") is True:
                    leakage_suspected = True
                    warning_msg = label_shuffle.get("warning") or "Label-shuffle test indicates potential data leakage"
                    if warning_msg not in leakage_warnings:
                        leakage_warnings.append(warning_msg)
                    step_logger.hypothesis(f"LEAKAGE DETECTED in {trial.variant_name}: {warning_msg}")
                    trial_line += f"\n  - ⚠️ LEAKAGE WARNING: {warning_msg}"

                # Add baseline metrics to trial line
                majority_class = baseline_data.get("majority_class", {})
                mean_predictor = baseline_data.get("mean_predictor", {})
                simple_model = baseline_data.get("simple_logistic", {}) or baseline_data.get("simple_ridge", {})

                if majority_class:
                    trial_line += f"\n  - Majority class baseline: acc={majority_class.get('accuracy', 'N/A')}"
                if mean_predictor:
                    trial_line += f"\n  - Mean predictor baseline: rmse={mean_predictor.get('rmse', 'N/A')}"
                if simple_model:
                    if "accuracy" in simple_model:
                        trial_line += f"\n  - Simple logistic baseline: acc={simple_model.get('accuracy', 'N/A')}"
                    elif "rmse" in simple_model:
                        trial_line += f"\n  - Simple ridge baseline: rmse={simple_model.get('rmse', 'N/A')}"

            # === Prompt 4: Check for time-split issues ===
            # === Prompt 7 Step 6: Enhanced split strategy validation with context ===
            split_strategy = trial.data_split_strategy
            if split_strategy:
                # Check if is_time_based but using random/stratified split
                if is_time_based and split_strategy in ("random", "stratified", "group_random"):
                    time_split_suspicious = True
                    # Build context-aware warning message
                    recommended_split = task_hints.get("recommended_split", "time")
                    warning_msg = (
                        f"Time-based data using '{split_strategy}' split may cause temporal leakage. "
                        f"Recommended split: '{recommended_split}'."
                    )
                    if warning_msg not in time_split_warnings:
                        time_split_warnings.append(warning_msg)
                    step_logger.hypothesis(f"TIME SPLIT ISSUE in {trial.variant_name}: {warning_msg}")
                    # Log with context from task_hints
                    if task_hints.get("is_time_based"):
                        step_logger.warning(
                            f"📊 Split validation: TaskContext confirms time-based task, "
                            f"but trial uses '{split_strategy}' split. "
                            f"Recommended: '{recommended_split}'"
                        )
                    trial_line += f"\n  - ⚠️ SPLIT WARNING: {warning_msg}"

            # Also check dataset spec for is_time_based if not provided in input
            if not is_time_based and exp.dataset_spec_id:
                dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == exp.dataset_spec_id).first()
                if dataset_spec and dataset_spec.is_time_based:
                    is_time_based = True
                    if split_strategy and split_strategy in ("random", "stratified", "group_random"):
                        time_split_suspicious = True
                        warning_msg = (
                            f"Dataset is time-based but using '{split_strategy}' split. "
                            f"This may cause temporal leakage."
                        )
                        if warning_msg not in time_split_warnings:
                            time_split_warnings.append(warning_msg)
                        step_logger.hypothesis(f"TIME SPLIT ISSUE: {warning_msg}")

            trials_data_lines.append(trial_line)

    trials_data = "\n\n".join(trials_data_lines) if trials_data_lines else "No trial data available"

    # Build baseline information - prefer real baselines from trials
    step_logger.thinking("Computing baseline comparisons...")
    baseline_lines = []
    baseline_value = None
    baseline_type = None

    # Use real baseline metrics if available from any trial
    if all_baseline_metrics:
        step_logger.info(f"Using real baseline metrics from {len(all_baseline_metrics)} trial(s)")

        # Aggregate baseline metrics (use first available)
        real_baselines = all_baseline_metrics[0]

        if is_classification:
            # Classification baselines
            majority = real_baselines.get("majority_class", {})
            simple = real_baselines.get("simple_logistic", {})
            shuffle = real_baselines.get("label_shuffle", {})

            if majority:
                maj_acc = majority.get("accuracy")
                maj_auc = majority.get("roc_auc")
                baseline_lines.append(f"- Majority class baseline: accuracy={maj_acc:.4f}" if maj_acc else "")
                if maj_auc is not None:
                    baseline_lines.append(f"- Majority class ROC AUC: {maj_auc:.4f}")
                    if "auc" in primary_metric.lower() or "roc" in primary_metric.lower():
                        baseline_value = maj_auc
                        baseline_type = "majority_class"
                elif maj_acc is not None and "accuracy" in primary_metric.lower():
                    baseline_value = maj_acc
                    baseline_type = "majority_class"

            if simple:
                simple_acc = simple.get("accuracy")
                simple_auc = simple.get("roc_auc")
                simple_f1 = simple.get("f1")
                if simple_acc:
                    baseline_lines.append(f"- Simple logistic baseline: accuracy={simple_acc:.4f}")
                if simple_auc:
                    baseline_lines.append(f"- Simple logistic ROC AUC: {simple_auc:.4f}")
                if simple_f1:
                    baseline_lines.append(f"- Simple logistic F1: {simple_f1:.4f}")

            if shuffle:
                shuffle_acc = shuffle.get("shuffled_accuracy")
                shuffle_auc = shuffle.get("shuffled_roc_auc")
                if shuffle_acc:
                    baseline_lines.append(f"- Label-shuffle accuracy: {shuffle_acc:.4f} (expected: ~{shuffle.get('expected_random_accuracy', 0.5):.2f})")
                if shuffle_auc:
                    baseline_lines.append(f"- Label-shuffle ROC AUC: {shuffle_auc:.4f} (expected: ~0.50)")
                if shuffle.get("leakage_detected"):
                    baseline_lines.append(f"- ⚠️ LEAKAGE DETECTED: {shuffle.get('warning', 'See label-shuffle results')}")
        else:
            # Regression baselines
            mean_pred = real_baselines.get("mean_predictor", {})
            simple = real_baselines.get("simple_ridge", {})
            shuffle = real_baselines.get("label_shuffle", {})

            if mean_pred:
                mean_rmse = mean_pred.get("rmse")
                mean_mae = mean_pred.get("mae")
                mean_r2 = mean_pred.get("r2")
                if mean_rmse:
                    baseline_lines.append(f"- Mean predictor RMSE: {mean_rmse:.4f}")
                    if "rmse" in primary_metric.lower():
                        baseline_value = mean_rmse
                        baseline_type = "mean_predictor"
                if mean_mae:
                    baseline_lines.append(f"- Mean predictor MAE: {mean_mae:.4f}")
                if mean_r2 is not None:
                    baseline_lines.append(f"- Mean predictor R²: {mean_r2:.4f}")

            if simple:
                ridge_rmse = simple.get("rmse")
                ridge_mae = simple.get("mae")
                ridge_r2 = simple.get("r2")
                if ridge_rmse:
                    baseline_lines.append(f"- Simple ridge RMSE: {ridge_rmse:.4f}")
                if ridge_mae:
                    baseline_lines.append(f"- Simple ridge MAE: {ridge_mae:.4f}")
                if ridge_r2 is not None:
                    baseline_lines.append(f"- Simple ridge R²: {ridge_r2:.4f}")

            if shuffle:
                shuffle_r2 = shuffle.get("shuffled_r2")
                shuffle_rmse = shuffle.get("shuffled_rmse")
                if shuffle_r2 is not None:
                    baseline_lines.append(f"- Label-shuffle R²: {shuffle_r2:.4f} (expected: ~0 or negative)")
                if shuffle_rmse:
                    baseline_lines.append(f"- Label-shuffle RMSE: {shuffle_rmse:.4f}")
                if shuffle.get("leakage_detected"):
                    baseline_lines.append(f"- ⚠️ LEAKAGE DETECTED: {shuffle.get('warning', 'See label-shuffle results')}")

        # Filter out empty strings
        baseline_lines = [line for line in baseline_lines if line]
    else:
        # Fallback to estimated baselines if no real ones available
        step_logger.info("No real baseline metrics found, using estimated baselines")

        if is_classification:
            if 'auc' in primary_metric.lower() or 'roc' in primary_metric.lower():
                baseline_value = 0.5
                baseline_type = "random_classifier"
                baseline_lines.append(f"- Random classifier AUC baseline (estimated): {baseline_value:.4f}")
            elif 'accuracy' in primary_metric.lower():
                baseline_value = 0.5
                baseline_type = "majority_class"
                baseline_lines.append(f"- Majority class accuracy baseline (estimated): {baseline_value:.4f}")
            else:
                baseline_value = 0.5
                baseline_type = "random"
                baseline_lines.append(f"- Random baseline (estimated): {baseline_value:.4f}")
        else:
            baseline_type = "mean_predictor"
            baseline_value = None
            baseline_lines.append("- Mean predictor baseline: Not computed (no baseline data available)")

    baseline_info = "\n".join(baseline_lines) if baseline_lines else "Baseline comparison not available"

    # Build CV data summary
    cv_data = None
    if cv_fold_data:
        cv_variance = max(cv_fold_data) - min(cv_fold_data) if len(cv_fold_data) > 1 else 0
        cv_mean = sum(cv_fold_data) / len(cv_fold_data)
        cv_data = f"""
- Number of folds observed: {len(cv_fold_data)}
- Mean across folds: {cv_mean:.4f}
- Range (max - min): {cv_variance:.4f}
- Min fold value: {min(cv_fold_data):.4f}
- Max fold value: {max(cv_fold_data):.4f}
"""

    # Compute summary statistics for logging
    if all_gaps:
        worst_gap = max(all_gaps)
        avg_gap = sum(all_gaps) / len(all_gaps)
        step_logger.thinking(f"Train-val gap analysis: worst={worst_gap:.4f}, avg={avg_gap:.4f}")

        if worst_gap > 0.2:
            step_logger.hypothesis("SEVERE overfitting detected - worst gap exceeds 0.20")
        elif worst_gap > 0.1:
            step_logger.hypothesis("MODERATE overfitting risk - worst gap exceeds 0.10")

    # Check baseline comparison
    if best_val_metric is not None and baseline_value is not None:
        relative_improvement = (best_val_metric - baseline_value) / baseline_value if baseline_value > 0 else 0
        step_logger.thinking(f"Best model: {best_val_metric:.4f}, Baseline: {baseline_value:.4f}, Improvement: {relative_improvement:.1%}")

        if relative_improvement < 0.05:
            step_logger.hypothesis("Model barely improves over trivial baseline (<5% improvement)")
        elif best_val_metric > 0.98:
            step_logger.hypothesis(f"Suspiciously high performance ({best_val_metric:.4f}) - check for data leakage")

    # Generate LLM prompt
    prompt = get_robustness_audit_prompt(
        project_name=project.name,
        problem_description=project.description or "No description provided",
        task_type=project.task_type or "unknown",
        primary_metric=primary_metric,
        trials_data=trials_data,
        baseline_info=baseline_info,
        cv_data=cv_data,
    )

    # Append unified TaskContext (Prompt 7)
    if task_context:
        task_context_str = format_context_for_prompt(
            task_context,
            include_sections=["baselines", "robustness", "leakage_candidates"],
            max_length=2000,
        )
        if task_context_str:
            prompt += "\n\n## 📊 ADDITIONAL PROJECT CONTEXT\n"
            prompt += task_context_str
            # Add leakage context from TaskContext
            if task_hints.get("leakage_warnings"):
                prompt += "\n\n**LEAKAGE WARNINGS TO CONSIDER**:\n"
                for warning in task_hints["leakage_warnings"][:3]:
                    prompt += f"- {warning}\n"

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_ROBUSTNESS_AUDITOR},
        {"role": "user", "content": prompt},
    ]

    step_logger.action("Generating robustness audit with LLM...")
    response = await llm_client.chat_json(messages, RobustnessAuditResponse)

    # Extract results
    overfitting_risk = response.get("overfitting_risk", "unknown")
    suspicious_patterns = response.get("suspicious_patterns", [])
    recommendations = response.get("recommendations", [])
    summary = response.get("natural_language_summary", "")

    # Log the results
    risk_emoji = {"low": "✅", "medium": "⚠️", "high": "🚨"}.get(overfitting_risk, "❓")
    step_logger.summary(f"Robustness Audit Complete - Risk: {risk_emoji} {overfitting_risk.upper()}")

    if suspicious_patterns:
        step_logger.info(f"Found {len(suspicious_patterns)} suspicious pattern(s)")
        for pattern in suspicious_patterns:
            step_logger.warning(f"[{pattern.get('type', 'unknown')}] {pattern.get('description', '')}")

    if recommendations:
        step_logger.info(f"Generated {len(recommendations)} recommendation(s)")

    # Compile all warnings (Prompt 4 requirement)
    all_warnings = []
    if leakage_suspected:
        step_logger.warning("⚠️ DATA LEAKAGE SUSPECTED - Label-shuffle test detected issues")
        all_warnings.extend(leakage_warnings)
    if time_split_suspicious:
        step_logger.warning("⚠️ TIME-SPLIT SUSPICIOUS - Time-based data using inappropriate split")
        all_warnings.extend(time_split_warnings)

    # Build metrics summary (Prompt 4 requirement)
    metrics_summary = {
        "best_val_metric": best_val_metric,
        "primary_metric": primary_metric,
        "train_val_gap_worst": max(all_gaps) if all_gaps else None,
        "train_val_gap_avg": sum(all_gaps) / len(all_gaps) if all_gaps else None,
        "cv_variance": max(cv_fold_data) - min(cv_fold_data) if len(cv_fold_data) > 1 else None,
        "baseline_value": baseline_value,
        "baseline_type": baseline_type,
    }

    # Build baseline metrics summary for response
    baseline_metrics_summary = {}
    if all_baseline_metrics:
        bm = all_baseline_metrics[0]  # Use first trial's baselines
        baseline_metrics_summary = {
            "majority_class": bm.get("majority_class", {}),
            "mean_predictor": bm.get("mean_predictor", {}),
            "simple_logistic": bm.get("simple_logistic", {}),
            "simple_ridge": bm.get("simple_ridge", {}),
            "label_shuffle": bm.get("label_shuffle", {}),
        }

    # === Prompt 5: "Too Good To Be True" Detection ===
    from app.services.risk_scoring import (
        compute_risk_adjusted_score,
        check_too_good_to_be_true,
        get_model_risk_status,
    )

    # Collect additional metrics from all experiments for TGTBT check
    additional_metrics: Dict[str, float] = {}
    for exp in experiments:
        for trial in exp.trials:
            if trial.metrics_json:
                for key, value in trial.metrics_json.items():
                    if isinstance(value, (int, float)) and key not in additional_metrics:
                        additional_metrics[key] = value

    # === Prompt 7 Step 6: Extract expected_metric_range from TaskContext or input_data ===
    expected_metric_range: Optional[Dict[str, Any]] = None

    # Try to get from input_data.context_analysis first (from Problem Framer)
    context_analysis = input_data.get("context_analysis", {})
    if context_analysis and isinstance(context_analysis, dict):
        expected_metric_range = context_analysis.get("expected_metric_range")

    # Fallback: Try to get from task_context
    if not expected_metric_range and task_context:
        # Check in data_profile_summary
        profile_summary = task_context.get("data_profile_summary", {})
        if profile_summary and isinstance(profile_summary, dict):
            expected_metric_range = profile_summary.get("expected_metric_range")
        # Also check direct context_analysis in task_context
        if not expected_metric_range:
            ctx_analysis = task_context.get("context_analysis", {})
            if ctx_analysis:
                expected_metric_range = ctx_analysis.get("expected_metric_range")

    # === Prompt 7 Step 6: Explicit baseline vs achieved metrics logging ===
    context_reasoning_parts: List[str] = []

    if expected_metric_range:
        step_logger.thinking(
            f"Expected metric range from Problem Framer: "
            f"{expected_metric_range.get('metric', 'unknown')}: "
            f"[{expected_metric_range.get('lower_bound', '?')}-{expected_metric_range.get('upper_bound', '?')}]"
        )
        context_factors["expected_metric_range"] = {
            "metric": expected_metric_range.get("metric"),
            "lower_bound": expected_metric_range.get("lower_bound"),
            "upper_bound": expected_metric_range.get("upper_bound"),
        }

    # Build context-aware reasoning string
    if baseline_value is not None:
        context_reasoning_parts.append(f"Baseline {baseline_type or 'value'} ~{baseline_value:.3f}")
    if expected_metric_range:
        lb = expected_metric_range.get("lower_bound", "?")
        ub = expected_metric_range.get("upper_bound", "?")
        context_reasoning_parts.append(f"expected realistic range [{lb}-{ub}]")
    if best_val_metric is not None:
        context_reasoning_parts.append(f"best model {primary_metric} {best_val_metric:.3f}")

    # Get split strategy from context for reasoning
    split_strategy_used = None
    if task_context:
        ds_spec = task_context.get("dataset_spec", {})
        if ds_spec:
            split_config = ds_spec.get("split_strategy", {})
            if isinstance(split_config, dict):
                split_strategy_used = split_config.get("type")
    if split_strategy_used:
        context_reasoning_parts.append(f"using {split_strategy_used} split")

    context_reasoning = ", ".join(context_reasoning_parts) if context_reasoning_parts else None

    # Check for "too good to be true" pattern (Prompt 7: now with expected_metric_range)
    too_good_to_be_true, tgtbt_warning = check_too_good_to_be_true(
        is_time_based=is_time_based,
        task_type=task_type,
        best_val_metric=best_val_metric,
        primary_metric=primary_metric,
        additional_metrics=additional_metrics,
        expected_metric_range=expected_metric_range,
    )

    if too_good_to_be_true and tgtbt_warning:
        step_logger.warning(f"⚠️ TOO GOOD TO BE TRUE: {tgtbt_warning}")
        all_warnings.append(tgtbt_warning)
        # Add as a suspicious pattern
        suspicious_patterns.append({
            "type": "too_good_to_be_true",
            "severity": "high",
            "description": tgtbt_warning,
        })
        # Upgrade overfitting risk if TGTBT detected
        if overfitting_risk == "low":
            overfitting_risk = "medium"

    # === Prompt 7 Step 6: Log context-aware reasoning for risk assessment ===
    if context_reasoning:
        step_logger.info(f"📊 Context for audit: {context_reasoning}")

        # Log detailed reasoning if there are concerns
        if best_val_metric is not None and expected_metric_range:
            upper_bound = expected_metric_range.get("upper_bound")
            if upper_bound is not None and best_val_metric > upper_bound:
                step_logger.warning(
                    f"🚨 Risk assessment reasoning: {context_reasoning} - "
                    f"actual exceeds expected upper bound by {best_val_metric - upper_bound:.3f}, "
                    f"marking risk HIGH"
                )
                # Upgrade overfitting risk when actual significantly exceeds expected
                if overfitting_risk == "low":
                    overfitting_risk = "medium"
                elif overfitting_risk == "medium" and (best_val_metric - upper_bound) > 0.10:
                    overfitting_risk = "high"

    # === Prompt 6: Check if leakage candidates are among top important features ===
    leakage_in_important_features = False
    concerning_leakage_features: List[Dict[str, Any]] = []
    leakage_importance_warning = ""
    leakage_candidates_from_audit: List[Dict[str, Any]] = input_data.get("leakage_candidates", [])

    if leakage_candidates_from_audit:
        # Import leakage checker
        try:
            from app.services.leakage_detector import check_leakage_in_important_features as check_leakage_imp
        except ImportError:
            check_leakage_imp = None
            logger.warning("Could not import leakage detector for importance check")

        if check_leakage_imp:
            # Collect feature importances from all trials
            all_feature_importances: Dict[str, float] = {}

            for exp in experiments:
                for trial in exp.trials:
                    # Check for feature_importances in metrics_json
                    if trial.metrics_json and "feature_importances" in trial.metrics_json:
                        fi = trial.metrics_json["feature_importances"]
                        if isinstance(fi, dict):
                            all_feature_importances.update(fi)

                    # Also check related model versions for feature importances
                    if hasattr(trial, "model_versions"):
                        for mv in trial.model_versions:
                            if mv.feature_importances_json:
                                fi = mv.feature_importances_json
                                if isinstance(fi, dict):
                                    all_feature_importances.update(fi)

            if all_feature_importances:
                step_logger.thinking(f"Checking {len(leakage_candidates_from_audit)} leakage candidates against {len(all_feature_importances)} feature importances")

                leakage_in_important_features, concerning_leakage_features, leakage_importance_warning = check_leakage_imp(
                    leakage_candidates=leakage_candidates_from_audit,
                    feature_importances=all_feature_importances,
                    top_n=10,
                    importance_threshold=0.05,
                )

                if leakage_in_important_features:
                    step_logger.warning(f"⚠️ LEAKAGE IN IMPORTANT FEATURES: {leakage_importance_warning}")
                    all_warnings.append(leakage_importance_warning)

                    # Increase leakage_suspected flag
                    leakage_suspected = True

                    # Add as suspicious pattern
                    for feat in concerning_leakage_features:
                        suspicious_patterns.append({
                            "type": "leakage_in_important_feature",
                            "severity": "high",
                            "description": f"Feature '{feat['column']}' (importance rank: {feat.get('importance_rank', 'N/A')}) "
                                           f"flagged for potential leakage: {feat.get('reason', 'Unknown reason')}",
                        })

                    # Upgrade overfitting risk if needed
                    if overfitting_risk == "low":
                        overfitting_risk = "medium"
                    elif overfitting_risk == "medium":
                        overfitting_risk = "high"

                    step_logger.hypothesis(
                        f"Model relies on {len(concerning_leakage_features)} suspicious feature(s) - "
                        f"high risk of data leakage affecting performance"
                    )

    # === Prompt 5: Compute risk-adjusted score ===
    risk_adjusted_score = None
    if best_val_metric is not None:
        risk_adjusted_score = compute_risk_adjusted_score(
            primary_metric=best_val_metric,
            overfitting_risk=overfitting_risk,
            leakage_suspected=leakage_suspected,
            time_split_suspicious=time_split_suspicious,
        )
        step_logger.info(f"Risk-adjusted score: {risk_adjusted_score:.4f}")

        # Store risk_adjusted_score in all trials' metrics_json
        for exp in experiments:
            for trial in exp.trials:
                if trial.metrics_json is None:
                    trial.metrics_json = {}
                trial.metrics_json["risk_adjusted_score"] = risk_adjusted_score
        try:
            db.commit()
            step_logger.info("Saved risk_adjusted_score to trial metrics")
        except Exception as e:
            logger.warning(f"Failed to save risk_adjusted_score to trials: {e}")
            db.rollback()

    # Get model risk status for promotion gating
    risk_level, requires_override, risk_reason = get_model_risk_status(
        overfitting_risk=overfitting_risk,
        leakage_suspected=leakage_suspected,
        time_split_suspicious=time_split_suspicious,
        too_good_to_be_true=too_good_to_be_true,
    )

    return {
        "robustness_audit": {
            # Prompt 4 required output fields
            "overfitting_risk": overfitting_risk,
            "leakage_suspected": leakage_suspected,
            "time_split_suspicious": time_split_suspicious,
            "metrics_summary": metrics_summary,
            "warnings": all_warnings,
            "recommendations": recommendations,
            "natural_language_summary": summary,
            # Prompt 5 additions
            "too_good_to_be_true": too_good_to_be_true,
            "risk_adjusted_score": risk_adjusted_score,
            "risk_level": risk_level,
            "requires_override": requires_override,
            "risk_reason": risk_reason,
            # Additional detailed analysis
            "train_val_analysis": response.get("train_val_analysis", {}),
            "suspicious_patterns": suspicious_patterns,
            "baseline_comparison": response.get("baseline_comparison", {}),
            "baseline_metrics": baseline_metrics_summary,
            "cv_analysis": response.get("cv_analysis", {}),
            "is_time_based": is_time_based,
            "task_type": task_type,
            # Prompt 6: Leakage in important features
            "leakage_in_important_features": leakage_in_important_features,
            "concerning_leakage_features": concerning_leakage_features,
            "leakage_candidates_count": len(leakage_candidates_from_audit),
            # Prompt 7 Step 6: Context-aware reasoning
            "context_reasoning": context_reasoning,
            # Context factors used (Prompt 7 Step 7)
            "context_factors_used": context_factors,
        }
    }


# ============================================
# LEGACY Step Handler Registry - BACKUP ONLY
# ============================================
# WARNING: This registry is NO LONGER USED.
# All agents now use the class-based system in app/services/agents/
# These handlers are kept here as reference/backup only.
# DO NOT add new handlers here - create agent classes instead.
# To restore: rename back to STEP_HANDLERS and uncomment fallback in run_agent_step
#
# _LEGACY_STEP_HANDLERS_BACKUP: Dict[AgentStepType, Callable] = {
#     AgentStepType.DATASET_DISCOVERY: handle_dataset_discovery_step,
#     AgentStepType.DATA_ANALYSIS: handle_data_analysis_step,
#     AgentStepType.PROBLEM_UNDERSTANDING: handle_problem_understanding_step,
#     AgentStepType.DATA_AUDIT: handle_data_audit_step,
#     AgentStepType.DATASET_DESIGN: handle_dataset_design_step,
#     AgentStepType.EXPERIMENT_DESIGN: handle_experiment_design_step,
#     AgentStepType.PLAN_CRITIC: handle_plan_critic_step,
#     AgentStepType.RESULTS_INTERPRETATION: handle_results_interpretation_step,
#     AgentStepType.RESULTS_CRITIC: handle_results_critic_step,
#     AgentStepType.TRAINING_DATASET_PLANNING: handle_training_dataset_planning_step,
#     AgentStepType.TRAINING_DATASET_BUILD: handle_training_dataset_build_step,
#     AgentStepType.DATASET_INVENTORY: handle_dataset_inventory_step,
#     AgentStepType.RELATIONSHIP_DISCOVERY: handle_relationship_discovery_step,
#     AgentStepType.LAB_NOTEBOOK_SUMMARY: handle_lab_notebook_summary_step,
#     AgentStepType.ROBUSTNESS_AUDIT: handle_robustness_audit_step,
# }


# ============================================
# Main Step Executor
# ============================================

async def run_agent_step(
    db: Session,
    step_id: UUID,
    llm_client: Optional[BaseLLMClient] = None,
) -> AgentStep:
    """Execute an agent step.

    This function:
    1. Loads the agent step
    2. Marks it as running
    3. Dispatches to the appropriate handler based on step_type
    4. Saves the output and marks as completed (or failed)

    Args:
        db: Database session
        step_id: UUID of the agent step to run
        llm_client: Optional LLM client (will be created if not provided)

    Returns:
        The updated AgentStep
    """
    # Load the step
    step = db.query(AgentStep).filter(AgentStep.id == step_id).first()
    if not step:
        raise ValueError(f"Agent step not found: {step_id}")

    # Create step logger
    step_logger = StepLogger(db, step_id)

    # Mark as running
    step.status = AgentStepStatus.RUNNING
    step.started_at = datetime.utcnow()
    db.commit()

    step_logger.info(f"Starting step: {step.step_type.value}")

    try:
        # Get or create LLM client
        if llm_client is None:
            llm_client = get_llm_client(db)

        # Get the agent class for this step type from the registry
        if not is_agent_registered(step.step_type):
            raise ValueError(f"No agent registered for step type: {step.step_type}")

        agent_class = get_agent_class(step.step_type)
        agent = agent_class(db, step, step_logger, llm_client)
        output = await agent.execute()

        # Mark as completed
        step.output_json = output
        step.status = AgentStepStatus.COMPLETED
        step.finished_at = datetime.utcnow()
        db.commit()

        step_logger.info(f"Step completed successfully")

        return step

    except (LLMError, LLMTimeoutError) as e:
        # LLM-specific failures
        error_message = f"LLM service error: {str(e)}"
        logger.error(f"Step {step_id} LLM failure: {error_message}")
        step_logger.error(f"Step failed (LLM): {error_message}")

        step.status = AgentStepStatus.FAILED
        step.error_message = error_message
        step.finished_at = datetime.utcnow()
        step.retry_count += 1
        db.commit()

        raise AgentStepError(step.step_type.value, str(step_id), error_message)

    except (DataError, DatasetBuildError) as e:
        # Data-related failures
        error_message = f"Data error: {str(e)}"
        logger.error(f"Step {step_id} data failure: {error_message}")
        step_logger.error(f"Step failed (Data): {error_message}")

        step.status = AgentStepStatus.FAILED
        step.error_message = error_message
        step.finished_at = datetime.utcnow()
        step.retry_count += 1
        db.commit()

        raise AgentStepError(step.step_type.value, str(step_id), error_message)

    except ValueError as e:
        # Configuration/validation errors
        error_message = str(e)
        logger.error(f"Step {step_id} validation error: {error_message}")
        step_logger.error(f"Step failed (Validation): {error_message}")

        step.status = AgentStepStatus.FAILED
        step.error_message = error_message
        step.finished_at = datetime.utcnow()
        step.retry_count += 1
        db.commit()

        raise

    except Exception as e:
        # Catch-all for unexpected errors
        error_message = str(e)
        logger.exception(f"Step {step_id} unexpected failure: {error_message}")
        step_logger.error(f"Step failed (Unexpected): {error_message}")

        step.status = AgentStepStatus.FAILED
        step.error_message = error_message
        step.finished_at = datetime.utcnow()
        step.retry_count += 1
        db.commit()

        raise


async def run_agent_pipeline(
    db: Session,
    run_id: UUID,
    llm_client: Optional[BaseLLMClient] = None,
    gemini_client: Optional[GeminiClient] = None,
    openai_client: Optional[OpenAIClient] = None,
) -> AgentRun:
    """Execute all pending steps in an agent run.

    Supports multiple orchestration modes:
    - SEQUENTIAL: Traditional sequential execution (default)
    - PROJECT_MANAGER: Dynamic orchestration with PM agent deciding next steps

    Also supports debate mode where Gemini critiques each step.

    Args:
        db: Database session
        run_id: UUID of the agent run
        llm_client: Optional main LLM client
        gemini_client: Optional Gemini client for debates
        openai_client: Optional OpenAI client for judge

    Returns:
        The updated AgentRun
    """
    # Load the run
    agent_run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if not agent_run:
        raise ValueError(f"Agent run not found: {run_id}")

    # Mark run as running
    agent_run.status = AgentRunStatus.RUNNING
    db.commit()

    # Check orchestration mode and dispatch
    orchestration_mode = getattr(agent_run, 'orchestration_mode', None)
    debate_mode = getattr(agent_run, 'debate_mode', None)

    try:
        # Project Manager mode - dynamic orchestration
        if orchestration_mode == PipelineOrchestrationMode.PROJECT_MANAGER:
            return await _run_project_manager_pipeline(
                db=db,
                agent_run=agent_run,
                llm_client=llm_client,
                gemini_client=gemini_client,
                openai_client=openai_client,
                debate_enabled=(debate_mode == DebateMode.ENABLED),
            )

        # Sequential mode (default) - with optional debate
        return await _run_sequential_pipeline(
            db=db,
            agent_run=agent_run,
            llm_client=llm_client,
            gemini_client=gemini_client,
            openai_client=openai_client,
            debate_enabled=(debate_mode == DebateMode.ENABLED),
        )

    except PipelineCancelledError as e:
        # Pipeline was cancelled by user - status is already CANCELLED
        # Just log and return the run (don't override status)
        logger.info(f"Pipeline {run_id} was cancelled by user")
        db.refresh(agent_run)  # Refresh to get latest state
        return agent_run

    except AgentStepError as e:
        agent_run.status = AgentRunStatus.FAILED
        agent_run.error_message = str(e)
        db.commit()
        raise AgentPipelineError(str(e))

    except (LLMError, LLMTimeoutError) as e:
        agent_run.status = AgentRunStatus.FAILED
        agent_run.error_message = f"LLM service error: {str(e)}"
        db.commit()
        raise AgentPipelineError(f"LLM service unavailable: {e}")

    except Exception as e:
        logger.exception(f"Pipeline {run_id} unexpected failure: {e}")
        agent_run.status = AgentRunStatus.FAILED
        agent_run.error_message = str(e)
        db.commit()
        raise


async def _run_sequential_pipeline(
    db: Session,
    agent_run: AgentRun,
    llm_client: Optional[BaseLLMClient] = None,
    gemini_client: Optional[GeminiClient] = None,
    openai_client: Optional[OpenAIClient] = None,
    debate_enabled: bool = False,
) -> AgentRun:
    """Execute pipeline steps sequentially (traditional mode).

    Args:
        db: Database session
        agent_run: The agent run to execute
        llm_client: Optional main LLM client
        gemini_client: Optional Gemini client for debates
        openai_client: Optional OpenAI client for judge
        debate_enabled: Whether to run debates on each step

    Returns:
        The updated AgentRun
    """
    run_id = agent_run.id

    try:
        # Define step ordering for all pipelines (lower = runs first)
        # This ensures proper step order even when created_at has identical timestamps
        ALL_STEP_ORDER: Dict[AgentStepType, int] = {
            # Setup pipeline
            AgentStepType.DATA_ANALYSIS: 1,
            AgentStepType.PROBLEM_UNDERSTANDING: 2,
            AgentStepType.DATA_AUDIT: 3,
            AgentStepType.DATASET_DESIGN: 4,
            AgentStepType.DATASET_VALIDATION: 5,  # Validates columns + feature performance
            AgentStepType.EXPERIMENT_DESIGN: 6,
            AgentStepType.PLAN_CRITIC: 7,
            # Results pipeline
            AgentStepType.RESULTS_INTERPRETATION: 10,
            AgentStepType.RESULTS_CRITIC: 11,
            # Improvement pipeline (must run in order!)
            AgentStepType.ITERATION_CONTEXT: 15,
            AgentStepType.IMPROVEMENT_DATA_ANALYSIS: 16,
            AgentStepType.IMPROVEMENT_DATASET_DESIGN: 17,
            AgentStepType.IMPROVEMENT_EXPERIMENT_DESIGN: 18,
            AgentStepType.IMPROVEMENT_ANALYSIS: 19,
            AgentStepType.IMPROVEMENT_PLAN: 19,
            # Data architect pipeline
            AgentStepType.DATASET_INVENTORY: 20,
            AgentStepType.RELATIONSHIP_DISCOVERY: 21,
            AgentStepType.TRAINING_DATASET_PLANNING: 22,
            AgentStepType.TRAINING_DATASET_BUILD: 23,
            # Discovery & misc
            AgentStepType.DATASET_DISCOVERY: 30,
            AgentStepType.LAB_NOTEBOOK_SUMMARY: 100,
        }

        # Get pending steps, ordered by step type priority then created_at
        pending_steps = (
            db.query(AgentStep)
            .filter(AgentStep.agent_run_id == run_id)
            .filter(AgentStep.status == AgentStepStatus.PENDING)
            .all()
        )
        # Sort by pipeline order, fallback to created_at for unknown types
        pending_steps = sorted(
            pending_steps,
            key=lambda s: (ALL_STEP_ORDER.get(s.step_type, 999), s.created_at)
        )

        # Track outputs to pass between steps
        accumulated_output = {}

        # Initialize debate manager if enabled
        debate_manager = None
        debate_partner_model = getattr(agent_run, 'debate_partner', None) or "gemini-2.0-flash"
        logger.info(f"Sequential Pipeline debate setup: debate_enabled={debate_enabled}, debate_partner={debate_partner_model}")
        if debate_enabled:
            # Create critique client based on the selected debate partner
            critique_client = gemini_client  # Use passed client if available
            if not critique_client:
                critique_client = create_critique_client(debate_partner_model)

            if critique_client:
                logger.info(f"Sequential Pipeline: Creating debate manager with debate partner: {debate_partner_model}")
                from app.services.agents.utils.step_logger import StepLogger
                seq_step_logger = StepLogger(db=db, step_id=None, agent_run_id=run_id)
                debate_manager = create_debate_manager(
                    db=db,
                    agent_run=agent_run,
                    step_logger=seq_step_logger,
                    main_llm_client=llm_client,
                    gemini_client=critique_client,
                    openai_client=openai_client,
                )
                logger.info("Sequential Pipeline: Debate manager created successfully")
            else:
                logger.warning(f"Sequential Pipeline: Debate enabled but no API key for {debate_partner_model} - debates will be SKIPPED")

        # Planner-Critic revision loop configuration
        MAX_PLAN_REVISIONS = 2
        plan_revision_count = 0

        for step in pending_steps:
            # Check for cancellation before each step
            _check_cancellation(db, run_id, step.step_type.value if step.step_type else None)

            # Merge previous outputs into this step's input if needed
            if step.input_json is None:
                step.input_json = {}

            # Add accumulated outputs that aren't already in input
            modified = False
            for key, value in accumulated_output.items():
                if key not in step.input_json:
                    step.input_json[key] = value
                    modified = True

            # Flag the JSON column as modified so SQLAlchemy persists the changes
            if modified:
                flag_modified(step, "input_json")

            db.commit()

            # Run the step (with optional debate)
            if debate_manager:
                completed_step = await _run_step_with_debate(
                    db=db,
                    step=step,
                    llm_client=llm_client,
                    debate_manager=debate_manager,
                    accumulated_output=accumulated_output,
                )
            else:
                completed_step = await run_agent_step(db, step.id, llm_client)

            # Accumulate output for next steps
            if completed_step.output_json:
                accumulated_output.update(completed_step.output_json)

            # ============================================
            # Dataset Validation Feedback Loop
            # ============================================
            # If Dataset Validation found invalid columns, re-run the appropriate agent
            # Keep retrying until valid - columns MUST be valid
            MAX_VALIDATION_RETRIES = 10  # Safety limit to prevent infinite loops
            validation_retry_count = accumulated_output.get("_validation_retry_count", 0)

            # Use a while loop to keep retrying until validation passes
            while (
                completed_step.step_type == AgentStepType.DATASET_VALIDATION
                and completed_step.output_json
                and not completed_step.output_json.get("is_valid", True)
                and validation_retry_count < MAX_VALIDATION_RETRIES
            ):
                missing_target = completed_step.output_json.get("missing_target")
                missing_features = completed_step.output_json.get("missing_features", [])
                available_columns = completed_step.output_json.get("available_columns", [])
                validation_feedback = completed_step.output_json.get("feedback", "")

                validation_retry_count += 1
                accumulated_output["_validation_retry_count"] = validation_retry_count

                if missing_target:
                    # Target column is invalid - re-run problem_understanding
                    logger.info(
                        f"Dataset Validation failed: missing target '{missing_target}' "
                        f"(retry {validation_retry_count}/{MAX_VALIDATION_RETRIES}). "
                        f"Re-running problem_understanding."
                    )

                    # Get description from config_json (required by problem_understanding)
                    config = agent_run.config_json or {}
                    description = config.get("description", "")
                    if not description:
                        # Try to get from accumulated_output
                        description = accumulated_output.get("description", "")
                    if not description:
                        # Fallback to project description
                        project = db.query(Project).filter(Project.id == agent_run.project_id).first()
                        description = project.description if project else "ML task"

                    # Create problem_understanding step with feedback
                    fix_input = {
                        **{k: v for k, v in accumulated_output.items()
                           if not k.startswith("_") and k not in ("is_valid", "missing_target", "missing_features")},
                        "description": description,  # Required by problem_understanding
                        "validation_failed": True,
                        "missing_target": missing_target,
                        "available_columns": available_columns,
                        "validation_feedback": f"Target column '{missing_target}' does not exist. Choose from: {', '.join(available_columns[:30])}",
                    }

                    fix_step = AgentStep(
                        agent_run_id=agent_run.id,
                        step_type=AgentStepType.PROBLEM_UNDERSTANDING,
                        status=AgentStepStatus.PENDING,
                        input_json=fix_input,
                    )
                    db.add(fix_step)
                    db.commit()

                    # Run problem_understanding to fix target
                    logger.info(f"Running problem_understanding fix step {fix_step.id}")
                    completed_fix = await run_agent_step(db, fix_step.id, llm_client)
                    if completed_fix.output_json:
                        accumulated_output.update(completed_fix.output_json)

                    # Now re-run dataset_design with the fixed target
                    redesign_input = {
                        **{k: v for k, v in accumulated_output.items() if not k.startswith("_")},
                        "validation_failed": True,
                        "available_columns": available_columns,
                    }
                    redesign_step = AgentStep(
                        agent_run_id=agent_run.id,
                        step_type=AgentStepType.DATASET_DESIGN,
                        status=AgentStepStatus.PENDING,
                        input_json=redesign_input,
                    )
                    db.add(redesign_step)
                    db.commit()

                    logger.info(f"Running dataset_design fix step {redesign_step.id}")
                    completed_redesign = await run_agent_step(db, redesign_step.id, llm_client)
                    if completed_redesign.output_json:
                        accumulated_output.update(completed_redesign.output_json)

                    # Re-run validation
                    revalidate_input = {
                        **{k: v for k, v in accumulated_output.items() if not k.startswith("_")},
                    }
                    revalidate_step = AgentStep(
                        agent_run_id=agent_run.id,
                        step_type=AgentStepType.DATASET_VALIDATION,
                        status=AgentStepStatus.PENDING,
                        input_json=revalidate_input,
                    )
                    db.add(revalidate_step)
                    db.commit()

                    logger.info(f"Running dataset_validation re-check step {revalidate_step.id}")
                    completed_revalidate = await run_agent_step(db, revalidate_step.id, llm_client)
                    if completed_revalidate.output_json:
                        accumulated_output.update(completed_revalidate.output_json)
                    # Update completed_step so while loop re-checks this result
                    completed_step = completed_revalidate

                elif missing_features:
                    # Features are invalid but target is OK - re-run dataset_design only
                    logger.info(
                        f"Dataset Validation failed: {len(missing_features)} missing features "
                        f"(retry {validation_retry_count}/{MAX_VALIDATION_RETRIES}). "
                        f"Re-running dataset_design."
                    )

                    # Create dataset_design step with feedback
                    fix_input = {
                        **{k: v for k, v in accumulated_output.items()
                           if not k.startswith("_") and k not in ("is_valid", "missing_features")},
                        "validation_failed": True,
                        "is_valid": False,
                        "missing_features": missing_features,
                        "available_columns": available_columns,
                        "feedback": validation_feedback,
                    }

                    fix_step = AgentStep(
                        agent_run_id=agent_run.id,
                        step_type=AgentStepType.DATASET_DESIGN,
                        status=AgentStepStatus.PENDING,
                        input_json=fix_input,
                    )
                    db.add(fix_step)
                    db.commit()

                    # Run dataset_design to fix features
                    logger.info(f"Running dataset_design fix step {fix_step.id}")
                    completed_fix = await run_agent_step(db, fix_step.id, llm_client)
                    if completed_fix.output_json:
                        accumulated_output.update(completed_fix.output_json)

                    # Re-run validation
                    revalidate_input = {
                        **{k: v for k, v in accumulated_output.items() if not k.startswith("_")},
                    }
                    revalidate_step = AgentStep(
                        agent_run_id=agent_run.id,
                        step_type=AgentStepType.DATASET_VALIDATION,
                        status=AgentStepStatus.PENDING,
                        input_json=revalidate_input,
                    )
                    db.add(revalidate_step)
                    db.commit()

                    logger.info(f"Running dataset_validation re-check step {revalidate_step.id}")
                    completed_revalidate = await run_agent_step(db, revalidate_step.id, llm_client)
                    if completed_revalidate.output_json:
                        accumulated_output.update(completed_revalidate.output_json)
                    # Update completed_step so while loop re-checks this result
                    completed_step = completed_revalidate

            # If we exhausted retries, fail hard
            if (
                completed_step.step_type == AgentStepType.DATASET_VALIDATION
                and completed_step.output_json
                and not completed_step.output_json.get("is_valid", True)
            ):
                missing_features = completed_step.output_json.get("missing_features", [])
                raise ValueError(
                    f"Dataset Validation failed after {MAX_VALIDATION_RETRIES} retries. "
                    f"Columns MUST be valid. Missing: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
                )

            # ============================================
            # Feature Performance Validation Feedback Loop
            # ============================================
            # If columns are valid but engineered features failed performance testing,
            # re-run dataset_design to generate better features
            MAX_FEATURE_RETRIES = 1
            if (
                completed_step.step_type == AgentStepType.DATASET_VALIDATION
                and completed_step.output_json
                and completed_step.output_json.get("is_valid", False)  # Columns are valid
            ):
                feature_validation = completed_step.output_json.get("feature_performance_validation", {})
                if feature_validation.get("validated", False):
                    features_before = feature_validation.get("features_before_validation", 0)
                    features_after = feature_validation.get("features_after_validation", 0)
                    validation_results = feature_validation.get("results", [])

                    # Check if too many features failed (>50% failure rate with at least 2 features proposed)
                    if features_before >= 2 and features_after < features_before * 0.5:
                        feature_retry_count = accumulated_output.get("_feature_retry_count", 0)

                        if feature_retry_count < MAX_FEATURE_RETRIES:
                            feature_retry_count += 1
                            accumulated_output["_feature_retry_count"] = feature_retry_count

                            # Build feedback about which features failed and why
                            failed_features = [
                                r for r in validation_results if not r.get("passed", False)
                            ]
                            failed_summary = "\n".join([
                                f"- {r.get('feature_name', '?')}: {r.get('reason', 'Unknown reason')}"
                                for r in failed_features[:10]
                            ])

                            logger.info(
                                f"Feature performance validation: {features_after}/{features_before} passed "
                                f"(retry {feature_retry_count}/{MAX_FEATURE_RETRIES}). "
                                f"Re-running dataset_design for better features."
                            )

                            # Create dataset_design step with feature feedback
                            feature_fix_input = {
                                **{k: v for k, v in accumulated_output.items()
                                   if not k.startswith("_")},
                                "feature_validation_failed": True,
                                "failed_features": [r.get("feature_name") for r in failed_features],
                                "failed_features_feedback": (
                                    f"## Feature Validation Results\n"
                                    f"Only {features_after}/{features_before} proposed features improved model performance.\n\n"
                                    f"### Failed Features (DO NOT recreate these):\n{failed_summary}\n\n"
                                    f"### Instructions:\n"
                                    f"1. Do NOT recreate any of the failed features above\n"
                                    f"2. Suggest DIFFERENT engineered features\n"
                                    f"3. Focus on features that are more likely to improve prediction\n"
                                    f"4. Consider simpler transformations that may be more robust"
                                ),
                            }

                            feature_fix_step = AgentStep(
                                agent_run_id=agent_run.id,
                                step_type=AgentStepType.DATASET_DESIGN,
                                status=AgentStepStatus.PENDING,
                                input_json=feature_fix_input,
                            )
                            db.add(feature_fix_step)
                            db.commit()

                            # Run dataset_design to generate better features
                            logger.info(f"Running dataset_design feature improvement step {feature_fix_step.id}")
                            completed_feature_fix = await run_agent_step(db, feature_fix_step.id, llm_client)
                            if completed_feature_fix.output_json:
                                accumulated_output.update(completed_feature_fix.output_json)

                            # Re-run validation
                            feature_revalidate_input = {
                                **{k: v for k, v in accumulated_output.items() if not k.startswith("_")},
                            }
                            feature_revalidate_step = AgentStep(
                                agent_run_id=agent_run.id,
                                step_type=AgentStepType.DATASET_VALIDATION,
                                status=AgentStepStatus.PENDING,
                                input_json=feature_revalidate_input,
                            )
                            db.add(feature_revalidate_step)
                            db.commit()

                            logger.info(f"Running dataset_validation feature re-check step {feature_revalidate_step.id}")
                            completed_feature_revalidate = await run_agent_step(db, feature_revalidate_step.id, llm_client)
                            if completed_feature_revalidate.output_json:
                                accumulated_output.update(completed_feature_revalidate.output_json)
                        else:
                            logger.warning(
                                f"Feature validation: Only {features_after}/{features_before} passed after "
                                f"{MAX_FEATURE_RETRIES} retry. Continuing with validated features only."
                            )

            # ============================================
            # Planner-Critic Feedback Loop
            # ============================================
            # If Plan Critic rejected the plan, loop back to Experiment Design
            if (
                completed_step.step_type == AgentStepType.PLAN_CRITIC
                and completed_step.output_json
                and not completed_step.output_json.get("approved", True)
                and plan_revision_count < MAX_PLAN_REVISIONS
            ):
                plan_revision_count += 1
                revision_feedback = completed_step.output_json.get("revision_feedback", [])
                issues = completed_step.output_json.get("issues", [])

                logger.info(
                    f"Plan Critic rejected plan (revision {plan_revision_count}/{MAX_PLAN_REVISIONS}). "
                    f"Requesting revision from Experiment Planner."
                )

                # Build feedback message for the Planner
                feedback_message = _build_revision_feedback_message(
                    issues=issues,
                    revision_feedback=revision_feedback,
                    revision_count=plan_revision_count,
                )

                # Build task_context for revision steps (Prompt 7 Step 2)
                revision_task_context = _build_task_context_for_step(
                    db=db,
                    project_id=str(agent_run.project_id) if agent_run.project_id else None,
                    research_cycle_id=str(getattr(agent_run, 'research_cycle_id', None)) if getattr(agent_run, 'research_cycle_id', None) else None,
                )

                # Create a new Experiment Design step with revision feedback
                revision_input = {
                    **{k: v for k, v in accumulated_output.items()
                       if k not in ("approved", "status", "issues", "warnings", "revision_feedback")},
                    "revision_request": True,
                    "revision_number": plan_revision_count,
                    "critic_feedback": feedback_message,
                    "previous_issues": issues,
                }
                if revision_task_context:
                    revision_input["task_context"] = revision_task_context

                revision_step = AgentStep(
                    agent_run_id=agent_run.id,
                    step_type=AgentStepType.EXPERIMENT_DESIGN,
                    status=AgentStepStatus.PENDING,
                    input_json=revision_input,
                )
                db.add(revision_step)
                db.commit()

                # Run the revision step
                logger.info(f"Running experiment design revision step {revision_step.id}")
                completed_revision = await run_agent_step(db, revision_step.id, llm_client)

                if completed_revision.output_json:
                    # Update accumulated output with revised plan, but keep other context
                    for key in ["variants", "recommended_variant", "reasoning", "natural_language_summary"]:
                        if key in completed_revision.output_json:
                            accumulated_output[key] = completed_revision.output_json[key]

                # Create a new Plan Critic step to review the revised plan
                review_input = {
                    **accumulated_output,
                    "revision_number": plan_revision_count,
                    "is_revision_review": True,
                }
                if revision_task_context:
                    review_input["task_context"] = revision_task_context

                review_step = AgentStep(
                    agent_run_id=agent_run.id,
                    step_type=AgentStepType.PLAN_CRITIC,
                    status=AgentStepStatus.PENDING,
                    input_json=review_input,
                )
                db.add(review_step)
                db.commit()

                # Run the review step
                logger.info(f"Running plan critic review step {review_step.id}")
                completed_review = await run_agent_step(db, review_step.id, llm_client)

                if completed_review.output_json:
                    accumulated_output.update(completed_review.output_json)

                # If still not approved and we have revisions left, loop will continue
                # (next iteration will be the remaining steps, critic check won't trigger
                # because we already updated accumulated_output with the new review)

        # Mark run as completed
        agent_run.status = AgentRunStatus.COMPLETED
        agent_run.result_json = accumulated_output
        db.commit()

        return agent_run

    except Exception:
        # Let exceptions bubble up to the main run_agent_pipeline function
        # which handles error status and logging
        raise


async def _run_project_manager_pipeline(
    db: Session,
    agent_run: AgentRun,
    llm_client: Optional[BaseLLMClient] = None,
    gemini_client: Optional[GeminiClient] = None,
    openai_client: Optional[OpenAIClient] = None,
    debate_enabled: bool = False,
) -> AgentRun:
    """Execute pipeline with Project Manager dynamic orchestration.

    In this mode, the Project Manager agent decides which agent runs next
    based on context and previous outputs. Agents can run multiple times
    and the PM decides when the pipeline is complete.

    Args:
        db: Database session
        agent_run: The agent run to execute
        llm_client: Optional main LLM client
        gemini_client: Optional Gemini client for debates
        openai_client: Optional OpenAI client for judge
        debate_enabled: Whether to run debates on each step

    Returns:
        The updated AgentRun
    """
    from app.services.agents.utils.step_logger import StepLogger

    run_id = agent_run.id
    MAX_PM_ITERATIONS = 20  # Prevent infinite loops

    try:
        # Initialize step logger for PM (step_id will be set when first PM step is created)
        pm_step_logger = StepLogger(db=db, step_id=None)
        # Note: Can't log until we have a valid step_id

        # Get base inputs from config_json (set by create_setup_pipeline)
        config = agent_run.config_json or {}
        description = config.get("description", "")
        data_source_id = config.get("data_source_id")

        # Log configuration for debugging
        logger.info(f"PM Pipeline starting - config_json keys: {list(config.keys())}")
        logger.info(f"PM Pipeline - description present: {bool(description)}, data_source_id: {data_source_id}")

        # Fallback: If description is missing from config, try to get it from the project
        if not description:
            project = db.query(Project).filter(Project.id == agent_run.project_id).first()
            if project and project.description:
                description = project.description
                logger.info(f"PM Pipeline - Using project description as fallback")

        # Fallback: Try to get from first pre-created step's input_json
        if not description:
            first_step = db.query(AgentStep).filter(
                AgentStep.agent_run_id == agent_run.id,
                AgentStep.step_type == AgentStepType.DATA_ANALYSIS,
            ).first()
            if first_step and first_step.input_json and first_step.input_json.get("description"):
                description = first_step.input_json.get("description")
                logger.info(f"PM Pipeline - Using first step's description as fallback")

        # Validate required inputs
        if not description:
            raise ValueError(
                "PM Pipeline requires a 'description' in config_json. "
                f"config_json: {config}. Please ensure the pipeline was created correctly."
            )

        # Build schema summary from data source (needed by agents)
        schema_dict = None
        if data_source_id:
            data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
            if data_source and data_source.schema_summary:
                schema_summary = build_schema_summary(
                    data_source_id=str(data_source.id),
                    data_source_name=data_source.name,
                    analysis_result=data_source.schema_summary,
                )
                schema_dict = schema_summary.model_dump(mode="json")

        # Initialize accumulated_output with base inputs that agents need
        accumulated_output = {
            "description": description,
            "data_source_id": data_source_id,
        }
        if schema_dict:
            accumulated_output["schema_summary"] = schema_dict

        # Build initial context for PM
        project = db.query(Project).filter(Project.id == agent_run.project_id).first()
        data_sources = db.query(DataSource).filter(
            DataSource.project_id == agent_run.project_id
        ).all() if agent_run.project_id else []

        pm_context = {
            "project_name": project.name if project else "Unknown",
            "project_description": project.description if project else "",
            "goal_description": description,  # The user's ML goal
            "data_source_count": len(data_sources),
            "available_agents": list(ORCHESTRABLE_AGENTS.keys()),
        }

        # Initialize debate manager if enabled
        debate_manager = None
        debate_partner_model = getattr(agent_run, 'debate_partner', None) or "gemini-2.0-flash"
        logger.info(f"PM Pipeline debate setup: debate_enabled={debate_enabled}, debate_partner={debate_partner_model}")
        if debate_enabled:
            # Create critique client based on the selected debate partner
            critique_client = gemini_client  # Use passed client if available
            if not critique_client:
                critique_client = create_critique_client(debate_partner_model)

            if critique_client:
                logger.info(f"PM Pipeline: Creating debate manager with debate partner: {debate_partner_model}")
                debate_manager = create_debate_manager(
                    db=db,
                    agent_run=agent_run,
                    step_logger=pm_step_logger,
                    main_llm_client=llm_client,
                    gemini_client=critique_client,
                    openai_client=openai_client,
                )
                logger.info(f"PM Pipeline: Debate manager created successfully")
            else:
                logger.warning(f"PM Pipeline: Debate enabled but no API key for {debate_partner_model} - debates will be SKIPPED")

        iteration = 0
        while iteration < MAX_PM_ITERATIONS:
            iteration += 1

            # Check for cancellation before each PM iteration
            _check_cancellation(db, run_id, "project_manager")

            # Get completed steps for PM execution history
            completed_step_types = [
                s.step_type.value for s in db.query(AgentStep).filter(
                    AgentStep.agent_run_id == run_id,
                    AgentStep.status == AgentStepStatus.COMPLETED,
                    AgentStep.step_type != AgentStepType.PROJECT_MANAGER,  # Exclude PM steps
                ).all()
            ]

            # Get last agent output for PM context
            last_completed = db.query(AgentStep).filter(
                AgentStep.agent_run_id == run_id,
                AgentStep.status == AgentStepStatus.COMPLETED,
                AgentStep.step_type != AgentStepType.PROJECT_MANAGER,
            ).order_by(AgentStep.finished_at.desc()).first()

            last_agent_output = last_completed.output_json if last_completed else {}
            last_agent_type = last_completed.step_type.value if last_completed else None

            # Get last PIPELINE agent (excluding debate steps) for loop prevention
            DEBATE_STEP_TYPES = {"gemini_critique", "openai_critique", "claude_critique", "debate_synthesis"}
            pipeline_steps_only = [s for s in completed_step_types if s not in DEBATE_STEP_TYPES]
            last_pipeline_agent = pipeline_steps_only[-1] if pipeline_steps_only else None

            logger.info(f"PM iteration {iteration}: execution_history={completed_step_types}, last_agent={last_agent_type}")

            # Create PM step to decide next action - use keys that PM agent expects
            pm_step = AgentStep(
                agent_run_id=run_id,
                step_type=AgentStepType.PROJECT_MANAGER,
                status=AgentStepStatus.PENDING,
                input_json={
                    # Keys that PM agent reads:
                    "pipeline_state": pm_context,
                    "last_agent": last_agent_type,
                    "last_agent_output": last_agent_output,
                    "execution_history": completed_step_types,
                    "goal": pm_context.get("goal_description", ""),
                    "accumulated_outputs": accumulated_output,
                    # Also keep these for debugging/backwards compat:
                    "iteration": iteration,
                },
            )
            db.add(pm_step)
            db.commit()

            # Run PM step
            pm_step_logger.step_id = pm_step.id
            if iteration == 1:
                pm_step_logger.thinking("Project Manager starting dynamic orchestration")
            pm_agent = ProjectManagerAgent(
                db=db,
                step=pm_step,
                step_logger=pm_step_logger,
                llm_client=llm_client,
            )

            pm_decision = await pm_agent.execute()
            pm_step.status = AgentStepStatus.COMPLETED
            pm_step.output_json = pm_decision
            db.commit()

            # Check if PM declares completion
            if pm_decision.get("is_complete", False):
                pm_step_logger.summary(f"Project Manager declared pipeline complete: {pm_decision.get('completion_reason', '')}")
                break

            # Get next agent to run
            next_agent_type = pm_decision.get("next_agent")
            if not next_agent_type:
                pm_step_logger.warning("PM did not specify next agent - ending pipeline")
                break

            # Convert to AgentStepType
            try:
                next_step_type = AgentStepType(next_agent_type)
            except ValueError:
                pm_step_logger.warning(f"Invalid agent type from PM: {next_agent_type}")
                continue

            # Check if agent is orchestrable
            if next_step_type not in ORCHESTRABLE_AGENTS:
                pm_step_logger.warning(f"Agent {next_agent_type} is not orchestrable")
                continue

            # GUARDRAIL: Limit how many times each agent can run (max 2 = original + 1 retry)
            MAX_RUNS_PER_AGENT = 2
            PIPELINE_SEQUENCE = [
                "data_analysis", "problem_understanding", "dataset_design",
                "data_audit", "dataset_validation", "experiment_design", "plan_critic"
            ]

            # Count how many times this agent has already run
            agent_run_count = pipeline_steps_only.count(next_agent_type)

            if agent_run_count >= MAX_RUNS_PER_AGENT:
                pm_step_logger.warning(f"Agent {next_agent_type} has already run {agent_run_count} times (max {MAX_RUNS_PER_AGENT})")
                # Force progression to next agent in pipeline sequence
                if next_agent_type in PIPELINE_SEQUENCE:
                    current_idx = PIPELINE_SEQUENCE.index(next_agent_type)
                    # Find the next agent that hasn't exceeded its limit
                    for next_idx in range(current_idx + 1, len(PIPELINE_SEQUENCE)):
                        candidate = PIPELINE_SEQUENCE[next_idx]
                        candidate_count = pipeline_steps_only.count(candidate)
                        if candidate_count < MAX_RUNS_PER_AGENT:
                            pm_step_logger.action(f"Forcing progression to {candidate} (skipping over-run agents)")
                            next_agent_type = candidate
                            next_step_type = AgentStepType(next_agent_type)
                            break
                    else:
                        # All agents at/past limit - pipeline is complete
                        pm_step_logger.summary("All pipeline agents have run - declaring complete")
                        break

            pm_step_logger.thinking(f"PM selected: {next_agent_type} - {pm_decision.get('reasoning', '')[:200]}")

            # Log what we're passing to the agent for debugging
            logger.info(f"PM Pipeline creating step for {next_agent_type}")
            logger.info(f"PM Pipeline accumulated_output keys: {list(accumulated_output.keys())}")
            logger.info(f"PM Pipeline description in accumulated_output: {'description' in accumulated_output}")

            # Create and run the selected agent step
            step_input_json = {
                **accumulated_output,
                "pm_instructions": pm_decision.get("agent_instructions", ""),
                "pm_focus_areas": pm_decision.get("focus_areas", []),
            }
            logger.info(f"PM Pipeline step_input_json keys: {list(step_input_json.keys())}")

            agent_step = AgentStep(
                agent_run_id=run_id,
                step_type=next_step_type,
                status=AgentStepStatus.PENDING,
                input_json=step_input_json,
            )
            db.add(agent_step)
            db.commit()

            # Verify the step was saved correctly
            db.refresh(agent_step)
            logger.info(f"PM Pipeline step saved - input_json keys: {list(agent_step.input_json.keys()) if agent_step.input_json else 'None'}")

            # Run the agent step
            completed_step = await run_agent_step(db, agent_step.id, llm_client)

            # Run debate if enabled
            logger.info(f"Debate check for {next_agent_type}: enabled={debate_enabled}, manager={debate_manager is not None}, has_output={completed_step.output_json is not None}")
            if debate_enabled and debate_manager and completed_step.output_json:
                logger.info(f"STARTING DEBATE for {next_agent_type}")
                pm_step_logger.thinking(f"Running debate for {next_agent_type}")
                debate_result = await debate_manager.run_debate(
                    agent_type=next_step_type,
                    main_agent_output=completed_step.output_json,
                    context=accumulated_output,
                )

                # Store full debate transcript with all messages
                if agent_run.debate_transcript_json is None:
                    agent_run.debate_transcript_json = []

                # Serialize transcript messages for storage
                transcript_messages = []
                for msg in debate_result.transcript:
                    transcript_messages.append({
                        "role": msg.role,
                        "content": msg.content,
                        "round": msg.round,
                        "agrees": msg.agrees,
                        "confidence": msg.confidence,
                        "metadata": msg.metadata,
                    })

                agent_run.debate_transcript_json.append({
                    "step_type": next_agent_type,
                    "iteration": iteration,
                    "consensus_reached": debate_result.consensus_reached,
                    "winner": debate_result.winner,
                    "total_rounds": debate_result.total_rounds,
                    "summary": debate_result.summary,
                    "judge_decision": debate_result.judge_decision,
                    "messages": transcript_messages,  # Full back-and-forth
                })
                flag_modified(agent_run, "debate_transcript_json")

                # Create AgentStep records for each debate round so users can see them
                for msg in debate_result.transcript:
                    if msg.role == "critique_agent":
                        step_type = AgentStepType.GEMINI_CRITIQUE
                    elif msg.role == "judge":
                        step_type = AgentStepType.OPENAI_JUDGE
                    else:
                        step_type = AgentStepType.DEBATE_ROUND

                    debate_step = AgentStep(
                        agent_run_id=run_id,
                        step_type=step_type,
                        status=AgentStepStatus.COMPLETED,
                        input_json={
                            "debate_for": next_agent_type,
                            "round": msg.round,
                        },
                        output_json={
                            "role": msg.role,
                            "content": msg.content,
                            "round": msg.round,
                            "agrees": msg.agrees,
                            "confidence": msg.confidence,
                        },
                    )
                    db.add(debate_step)

                # Add judge decision step if there was one
                if debate_result.judge_decision:
                    judge_step = AgentStep(
                        agent_run_id=run_id,
                        step_type=AgentStepType.OPENAI_JUDGE,
                        status=AgentStepStatus.COMPLETED,
                        input_json={
                            "debate_for": next_agent_type,
                            "total_rounds": debate_result.total_rounds,
                        },
                        output_json={
                            "winner": debate_result.winner,
                            "reasoning": debate_result.judge_decision.get("reasoning", ""),
                            "final_decision": debate_result.judge_decision,
                        },
                    )
                    db.add(judge_step)

                db.commit()

                # Use debate result as the step output
                completed_step.output_json = debate_result.final_output
                flag_modified(completed_step, "output_json")
                db.commit()

            # Accumulate output
            if completed_step.output_json:
                accumulated_output.update(completed_step.output_json)

            # Update PM context with results
            pm_context["last_agent"] = next_agent_type
            pm_context["last_output_keys"] = list(completed_step.output_json.keys()) if completed_step.output_json else []

        if iteration >= MAX_PM_ITERATIONS:
            pm_step_logger.warning(f"PM pipeline reached max iterations ({MAX_PM_ITERATIONS})")

        # Mark run as completed
        agent_run.status = AgentRunStatus.COMPLETED
        agent_run.result_json = accumulated_output
        db.commit()

        return agent_run

    except Exception:
        # Let exceptions bubble up to the main run_agent_pipeline function
        raise


async def _run_step_with_debate(
    db: Session,
    step: AgentStep,
    llm_client: Optional[BaseLLMClient],
    debate_manager: DebateManager,
    accumulated_output: Dict[str, Any],
) -> AgentStep:
    """Run a single step with optional debate.

    Args:
        db: Database session
        step: The step to run
        llm_client: LLM client
        debate_manager: The debate manager
        accumulated_output: Accumulated outputs from previous steps

    Returns:
        The completed step (potentially with modified output from debate)
    """
    # Run the step normally
    completed_step = await run_agent_step(db, step.id, llm_client)

    # Skip debate for certain step types (critique agents, PM, judge)
    skip_debate_types = {
        AgentStepType.PLAN_CRITIC,
        AgentStepType.RESULTS_CRITIC,
        AgentStepType.PROJECT_MANAGER,
        AgentStepType.GEMINI_CRITIQUE,
        AgentStepType.OPENAI_JUDGE,
        AgentStepType.DEBATE_ROUND,
        AgentStepType.LAB_NOTEBOOK_SUMMARY,
    }

    if completed_step.step_type in skip_debate_types:
        logger.info(f"Sequential debate: Skipping debate for {completed_step.step_type} (excluded type)")
        return completed_step

    # Run debate on the step output
    if completed_step.output_json:
        logger.info(f"STARTING DEBATE for {completed_step.step_type} (Sequential Pipeline)")
        debate_result = await debate_manager.run_debate(
            agent_type=completed_step.step_type,
            main_agent_output=completed_step.output_json,
            context=accumulated_output,
        )
        logger.info(f"Debate completed for {completed_step.step_type}: consensus={debate_result.consensus_reached}")

        # Update step output with debate result
        completed_step.output_json = debate_result.final_output
        flag_modified(completed_step, "output_json")
        db.commit()
    else:
        logger.info(f"Sequential debate: No output from {completed_step.step_type}, skipping debate")

    return completed_step


# ============================================
# Project Setup Pipeline
# ============================================


def create_setup_pipeline_notebook_entry(
    db: Session,
    agent_run: AgentRun,
    project_id: UUID,
) -> LabNotebookEntry:
    """Create a lab notebook entry summarizing the setup pipeline results.

    This captures the agent's problem analysis, data audit findings, and
    experiment design in a human-readable format for the lab notebook.

    Note: The result_json contains flattened outputs from all steps merged together,
    not nested by step name. Keys like 'task_type', 'target_column', 'variants', etc.
    are at the top level.

    Args:
        db: Database session
        agent_run: The completed agent run
        project_id: UUID of the project

    Returns:
        The created LabNotebookEntry
    """
    result = agent_run.result_json or {}

    # Build title from problem understanding keys (flattened in result)
    task_type = result.get("task_type", "ML Task")
    target_col = result.get("target_column", "target")
    title = f"Setup Pipeline Complete: {task_type} for {target_col}"

    # Build markdown body with sections
    sections = []

    # Problem Understanding section (from flattened keys)
    if result.get("task_type") or result.get("target_column"):
        sections.append("## Problem Understanding\n")
        if result.get("task_type"):
            sections.append(f"**Task Type:** {result['task_type']}\n")
        if result.get("target_column"):
            sections.append(f"**Target Column:** {result['target_column']}\n")
        if result.get("primary_metric"):
            sections.append(f"**Primary Metric:** {result['primary_metric']}\n")
        if result.get("reasoning"):
            sections.append(f"\n{result['reasoning']}\n")
        sections.append("\n")

    # Data Audit section (from flattened keys)
    if result.get("row_count") or result.get("column_count") or result.get("issues"):
        sections.append("## Data Audit Findings\n")
        if result.get("row_count"):
            sections.append(f"**Rows:** {result['row_count']:,}\n")
        if result.get("column_count"):
            sections.append(f"**Columns:** {result['column_count']}\n")
        if result.get("feature_count"):
            sections.append(f"**Features:** {result['feature_count']}\n")
        if result.get("suitability_score"):
            sections.append(f"**Suitability Score:** {result['suitability_score']}/100\n")

        issues = result.get("issues", [])
        if isinstance(issues, list) and issues:
            sections.append("\n**Issues Found:**\n")
            for issue in issues[:5]:
                if isinstance(issue, dict):
                    sections.append(f"- {issue.get('description', str(issue))}\n")
                else:
                    sections.append(f"- {issue}\n")

        if result.get("key_observations"):
            obs = result["key_observations"]
            if isinstance(obs, list) and obs:
                sections.append("\n**Key Observations:**\n")
                for o in obs[:5]:
                    sections.append(f"- {o}\n")

        if result.get("recommendations"):
            recs = result["recommendations"]
            if isinstance(recs, list) and recs:
                sections.append("\n**Recommendations:**\n")
                for rec in recs[:5]:
                    sections.append(f"- {rec}\n")
        sections.append("\n")

    # Dataset Design section (from flattened keys)
    variants = result.get("variants", [])
    if isinstance(variants, list) and variants:
        sections.append("## Dataset Design\n")
        sections.append(f"**{len(variants)} dataset variation(s) designed**\n\n")
        for i, variant in enumerate(variants[:3], 1):
            if isinstance(variant, dict):
                name = variant.get("name", f"Variation {i}")
                rationale = variant.get("rationale", "")
                sections.append(f"### {name}\n")
                if rationale:
                    sections.append(f"{rationale}\n")
                features = variant.get("engineered_features", [])
                if features:
                    sections.append(f"- {len(features)} engineered features\n")
            else:
                sections.append(f"### Variation {i}\n")
            sections.append("\n")

    # Plan Critic section (from flattened keys)
    if result.get("approved") is not None or result.get("status"):
        sections.append("## Plan Review\n")
        if result.get("approved"):
            sections.append("**Status:** Approved\n")
        elif result.get("status"):
            sections.append(f"**Status:** {result['status']}\n")

        if result.get("can_proceed") is not None:
            sections.append(f"**Can Proceed:** {'Yes' if result['can_proceed'] else 'No'}\n")

        if result.get("limitations"):
            lims = result["limitations"]
            if isinstance(lims, list) and lims:
                sections.append("\n**Limitations:**\n")
                for lim in lims[:3]:
                    sections.append(f"- {lim}\n")

    # Natural language summary if available
    if result.get("natural_language_summary"):
        sections.append("\n## Summary\n")
        sections.append(f"{result['natural_language_summary']}\n")

    body_markdown = "".join(sections) if sections else "Setup pipeline completed successfully."

    # Create the notebook entry
    entry = LabNotebookEntry(
        project_id=project_id,
        research_cycle_id=None,
        agent_step_id=None,
        author_type=LabNotebookAuthorType.AGENT,
        title=title,
        body_markdown=body_markdown,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    logger.info(f"Created lab notebook entry {entry.id} for setup pipeline")
    return entry


def create_results_pipeline_notebook_entry(
    db: Session,
    agent_run: AgentRun,
    experiment_id: UUID,
) -> LabNotebookEntry:
    """Create a lab notebook entry summarizing the results pipeline analysis.

    This captures the agent's interpretation of experiment results and
    critical analysis in a human-readable format for the lab notebook.

    Note: The result_json contains flattened outputs with keys like:
    'leaderboard', 'recommendation', 'critic_findings', 'results_summary',
    'trial_summaries', 'natural_language_summary'

    Args:
        db: Database session
        agent_run: The completed agent run
        experiment_id: UUID of the experiment analyzed

    Returns:
        The created LabNotebookEntry
    """
    # Get the experiment to find project_id
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")

    result = agent_run.result_json or {}

    # Build title
    exp_name = experiment.name or "Experiment"
    title = f"Results Analysis: {exp_name}"

    # Build markdown body
    sections = []

    # Results Summary section
    if result.get("results_summary") or result.get("natural_language_summary"):
        sections.append("## Results Summary\n")
        if result.get("natural_language_summary"):
            sections.append(f"{result['natural_language_summary']}\n\n")
        elif result.get("results_summary"):
            summary = result["results_summary"]
            if isinstance(summary, str):
                sections.append(f"{summary}\n\n")
            elif isinstance(summary, dict):
                if summary.get("summary"):
                    sections.append(f"{summary['summary']}\n\n")
                if summary.get("key_findings"):
                    findings = summary["key_findings"]
                    if isinstance(findings, list) and findings:
                        sections.append("**Key Findings:**\n")
                        for f in findings[:5]:
                            sections.append(f"- {f}\n")
                        sections.append("\n")

    # Leaderboard section
    leaderboard = result.get("leaderboard", [])
    if isinstance(leaderboard, list) and leaderboard:
        sections.append("## Model Leaderboard\n")
        sections.append("| Rank | Model | Score |\n")
        sections.append("|------|-------|-------|\n")
        for i, entry in enumerate(leaderboard[:5], 1):
            if isinstance(entry, dict):
                model = entry.get("model", entry.get("name", f"Model {i}"))
                score = entry.get("score", entry.get("metric_value", "N/A"))
                if isinstance(score, float):
                    score = f"{score:.4f}"
                sections.append(f"| {i} | {model} | {score} |\n")
        sections.append("\n")

    # Trial Summaries section
    trial_summaries = result.get("trial_summaries", [])
    if isinstance(trial_summaries, list) and trial_summaries:
        sections.append("## Trial Summaries\n")
        for i, trial in enumerate(trial_summaries[:5], 1):
            if isinstance(trial, dict):
                name = trial.get("name", trial.get("trial_name", f"Trial {i}"))
                status = trial.get("status", "completed")
                score = trial.get("score", trial.get("primary_metric_value"))
                sections.append(f"### {name}\n")
                sections.append(f"- **Status:** {status}\n")
                if score is not None:
                    if isinstance(score, float):
                        sections.append(f"- **Score:** {score:.4f}\n")
                    else:
                        sections.append(f"- **Score:** {score}\n")
                if trial.get("insights"):
                    sections.append(f"- **Insights:** {trial['insights']}\n")
                sections.append("\n")

    # Critic Findings section
    critic_findings = result.get("critic_findings", {})
    if critic_findings:
        sections.append("## Critical Analysis\n")
        if isinstance(critic_findings, dict):
            if critic_findings.get("overall_assessment"):
                sections.append(f"{critic_findings['overall_assessment']}\n\n")
            if critic_findings.get("concerns"):
                concerns = critic_findings["concerns"]
                if isinstance(concerns, list) and concerns:
                    sections.append("**Concerns:**\n")
                    for c in concerns[:5]:
                        sections.append(f"- {c}\n")
                    sections.append("\n")
            if critic_findings.get("suggestions"):
                suggestions = critic_findings["suggestions"]
                if isinstance(suggestions, list) and suggestions:
                    sections.append("**Suggestions for Improvement:**\n")
                    for s in suggestions[:5]:
                        sections.append(f"- {s}\n")
                    sections.append("\n")
        elif isinstance(critic_findings, str):
            sections.append(f"{critic_findings}\n\n")

    # Recommendation section
    recommendation = result.get("recommendation")
    if recommendation:
        sections.append("## Recommendation\n")
        if isinstance(recommendation, dict):
            if recommendation.get("summary"):
                sections.append(f"{recommendation['summary']}\n\n")
            if recommendation.get("next_steps"):
                next_steps = recommendation["next_steps"]
                if isinstance(next_steps, list) and next_steps:
                    sections.append("**Next Steps:**\n")
                    for ns in next_steps[:5]:
                        sections.append(f"- {ns}\n")
        elif isinstance(recommendation, str):
            sections.append(f"{recommendation}\n")

    body_markdown = "".join(sections) if sections else "Results analysis completed successfully."

    # Create the notebook entry
    entry = LabNotebookEntry(
        project_id=experiment.project_id,
        research_cycle_id=None,
        agent_step_id=None,
        author_type=LabNotebookAuthorType.AGENT,
        title=title,
        body_markdown=body_markdown,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    logger.info(f"Created lab notebook entry {entry.id} for results pipeline (experiment {experiment_id})")
    return entry


# Define the standard pipeline steps for project setup
SETUP_PIPELINE_STEPS: List[AgentStepType] = [
    AgentStepType.DATA_ANALYSIS,
    AgentStepType.PROBLEM_UNDERSTANDING,
    AgentStepType.DATA_AUDIT,
    AgentStepType.DATASET_DESIGN,
    AgentStepType.DATASET_VALIDATION,  # Validates columns + feature performance
    AgentStepType.EXPERIMENT_DESIGN,
    AgentStepType.PLAN_CRITIC,
]


def create_setup_pipeline(
    db: Session,
    project_id: UUID,
    data_source_id: UUID,
    description: str,
    time_budget_minutes: Optional[int] = None,
    orchestration_mode: str = "sequential",
    debate_mode: str = "disabled",
    judge_model: Optional[str] = None,
    max_debate_rounds: int = 3,
    debate_partner: Optional[str] = "gemini-2.0-flash",
    use_context_documents: bool = True,
    context_ab_testing: bool = False,
) -> AgentRun:
    """Create an agent run with all setup pipeline steps.

    This creates the agent_run and all agent_steps in pending state,
    ready to be executed.

    Args:
        db: Database session
        project_id: UUID of the project
        data_source_id: UUID of the data source to analyze
        description: User's description of the ML goal
        time_budget_minutes: Optional time budget constraint
        orchestration_mode: 'sequential' or 'project_manager'
        debate_mode: 'disabled' or 'enabled'
        judge_model: OpenAI model for judge (when debate enabled)
        max_debate_rounds: Max debate rounds before judge decides (default: 3)
        use_context_documents: If True, include context documents in AI prompts
        context_ab_testing: If True, create experiments with and without context for A/B comparison

    Returns:
        The created AgentRun with steps
    """
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project not found: {project_id}")

    # Verify data source exists and belongs to project
    data_source = db.query(DataSource).filter(
        DataSource.id == data_source_id,
        DataSource.project_id == project_id,
    ).first()
    if not data_source:
        raise ValueError(f"Data source not found or doesn't belong to project: {data_source_id}")

    if not data_source.schema_summary:
        raise ValueError(f"Data source {data_source_id} has no schema analysis. Upload and analyze the data first.")

    # Build schema summary
    schema_summary = build_schema_summary(
        data_source_id=str(data_source.id),
        data_source_name=data_source.name,
        analysis_result=data_source.schema_summary,
    )

    # Convert orchestration mode strings to enums
    from app.models import PipelineOrchestrationMode, DebateMode
    try:
        orch_mode = PipelineOrchestrationMode(orchestration_mode)
    except ValueError:
        orch_mode = PipelineOrchestrationMode.SEQUENTIAL

    try:
        deb_mode = DebateMode(debate_mode)
    except ValueError:
        deb_mode = DebateMode.DISABLED

    # Create the agent run with orchestration settings
    agent_run = AgentRun(
        project_id=project_id,
        name=f"Setup Pipeline for {project.name}",
        description=f"AI-powered project setup: {description[:100]}...",
        status=AgentRunStatus.PENDING,
        config_json={
            "description": description,
            "data_source_id": str(data_source_id),
            "time_budget_minutes": time_budget_minutes,
            "use_context_documents": use_context_documents,
            "context_ab_testing": context_ab_testing,
        },
        orchestration_mode=orch_mode,
        debate_mode=deb_mode,
        judge_model=judge_model,
        max_debate_rounds=max_debate_rounds,
        debate_partner=debate_partner,
    )
    db.add(agent_run)
    db.commit()
    db.refresh(agent_run)

    # Create steps with appropriate input for each
    # Use mode="json" to ensure UUID is serialized as string
    schema_dict = schema_summary.model_dump(mode="json")

    # Build task_context for later steps (Prompt 7 Step 2)
    # Note: At setup time, project exists but may not have experiments/specs yet
    setup_task_context = _build_task_context_for_step(
        db=db,
        project_id=str(project_id),
    )

    for step_type in SETUP_PIPELINE_STEPS:
        # Build input_json based on step type
        if step_type == AgentStepType.DATA_ANALYSIS:
            input_json = {
                "description": description,
                "data_source_id": str(data_source_id),
                "schema_summary": schema_dict,
            }
        elif step_type == AgentStepType.PROBLEM_UNDERSTANDING:
            # Will receive schema_summary from data_analysis step
            input_json = {
                "description": description,
                "data_source_id": str(data_source_id),
            }
            # Include task_context for context persistence (Prompt 7 Step 2)
            if setup_task_context:
                input_json["task_context"] = setup_task_context
        elif step_type == AgentStepType.DATA_AUDIT:
            # Will receive schema_summary and target_column from previous step
            input_json = {
                "schema_summary": schema_dict,
            }
        elif step_type == AgentStepType.DATASET_DESIGN:
            # Will receive task_type, target_column, schema_summary from accumulated outputs
            input_json = {
                "description": description,
            }
        elif step_type == AgentStepType.EXPERIMENT_DESIGN:
            # Will receive feature_columns, task_type, etc. from accumulated outputs
            input_json = {
                "time_budget_minutes": time_budget_minutes,
                "description": description,
            }
            # Include task_context for context persistence (Prompt 7 Step 2)
            if setup_task_context:
                input_json["task_context"] = setup_task_context
        elif step_type == AgentStepType.PLAN_CRITIC:
            # Will receive all accumulated outputs
            input_json = {}
            # Include task_context for context persistence (Prompt 7 Step 2)
            if setup_task_context:
                input_json["task_context"] = setup_task_context
        else:
            input_json = {}

        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=step_type,
            status=AgentStepStatus.PENDING,
            input_json=input_json,
        )
        db.add(step)

    db.commit()
    db.refresh(agent_run)

    return agent_run


async def run_setup_pipeline_for_project(
    db: Session,
    project_id: UUID,
    data_source_id: UUID,
    description: str,
    time_budget_minutes: Optional[int] = None,
    orchestration_mode: str = "sequential",
    debate_mode: str = "disabled",
    judge_model: Optional[str] = None,
    max_debate_rounds: int = 3,
    debate_partner: Optional[str] = "gemini-2.0-flash",
    llm_client: Optional[BaseLLMClient] = None,
    gemini_client: Optional[GeminiClient] = None,
    openai_client: Optional[OpenAIClient] = None,
    use_context_documents: bool = True,
    context_ab_testing: bool = False,
) -> AgentRun:
    """Create and run the full setup pipeline for a project.

    This function:
    1. Expands the user's brief description using GPT-5.1 thinking
    2. Creates an agent_run with project_id, status="pending"
    3. Creates 5 agent_steps in order (PROBLEM_UNDERSTANDING → PLAN_CRITIC)
    4. Executes each step sequentially, passing outputs between steps
    5. Returns the completed agent_run

    Args:
        db: Database session
        project_id: UUID of the project
        data_source_id: UUID of the data source to analyze
        description: User's description of the ML goal
        time_budget_minutes: Optional time budget constraint
        orchestration_mode: 'sequential' or 'project_manager'
        debate_mode: 'disabled' or 'enabled'
        judge_model: OpenAI model for judge (when debate enabled)
        llm_client: Optional LLM client
        gemini_client: Optional Gemini client for debates
        openai_client: Optional OpenAI client for judge
        use_context_documents: If True, include context documents in AI prompts
        context_ab_testing: If True, create experiments with and without context for A/B comparison

    Returns:
        The completed AgentRun with all step outputs

    Raises:
        ValueError: If project or data source not found
        Exception: If any step fails
    """
    # Get or create the LLM client if not provided
    if not llm_client:
        llm_client = get_llm_client()

    # Get data source for schema context
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    schema_summary = None
    if data_source and data_source.schema_summary:
        try:
            schema_summary = build_schema_summary(
                data_source_id=str(data_source.id),
                data_source_name=data_source.name,
                analysis_result=data_source.schema_summary,
            )
        except Exception as e:
            logger.warning(f"Could not build schema summary for goal expansion: {e}")

    # Expand the user's description using GPT-5.1 thinking
    logger.info(f"Expanding user goal description using GPT-5.1 thinking...")
    try:
        expanded_description = await expand_user_goal(
            client=llm_client,
            user_description=description,
            schema_summary=schema_summary,
        )
        logger.info(f"Goal expansion complete: {len(description)} -> {len(expanded_description)} chars")
    except Exception as e:
        logger.warning(f"Goal expansion failed, using original description: {e}")
        expanded_description = description

    # Create the pipeline with the expanded description
    agent_run = create_setup_pipeline(
        db=db,
        project_id=project_id,
        data_source_id=data_source_id,
        description=expanded_description,
        time_budget_minutes=time_budget_minutes,
        orchestration_mode=orchestration_mode,
        debate_mode=debate_mode,
        judge_model=judge_model,
        max_debate_rounds=max_debate_rounds,
        debate_partner=debate_partner,
        use_context_documents=use_context_documents,
        context_ab_testing=context_ab_testing,
    )

    # Run the pipeline (with optional orchestration mode and debate)
    completed_run = await run_agent_pipeline(
        db=db,
        run_id=agent_run.id,
        llm_client=llm_client,
        gemini_client=gemini_client,
        openai_client=openai_client,
    )

    # Create a lab notebook entry if pipeline completed successfully
    if completed_run.status == AgentRunStatus.COMPLETED:
        try:
            create_setup_pipeline_notebook_entry(db, completed_run, project_id)
        except Exception as e:
            logger.warning(f"Failed to create lab notebook entry for setup pipeline: {e}")

        # Trigger Auto DS if enabled on the project
        try:
            _trigger_auto_ds_if_enabled(db, project_id)
        except Exception as e:
            logger.warning(f"Failed to trigger Auto DS after pipeline completion: {e}")

    return completed_run


def _trigger_auto_ds_if_enabled(db: Session, project_id: UUID) -> None:
    """Trigger Auto DS session if enabled on the project.

    This is called after the setup pipeline completes successfully.
    It checks if:
    1. auto_ds_enabled is True on the project
    2. start_on_pipeline_complete is True in the config (or config is None)
    3. There's no already active Auto DS session
    4. There's at least one dataset in the project

    If all conditions are met, it creates a new Auto DS session and starts it.
    """
    from app.tasks.auto_ds_tasks import run_auto_ds_session

    # Get the project with fresh data
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        logger.warning(f"Project {project_id} not found for Auto DS trigger")
        return

    # Check if Auto DS is enabled on the project
    if not project.auto_ds_enabled:
        logger.debug(f"Auto DS not enabled for project {project_id}")
        return

    # Check if start_on_pipeline_complete is enabled (default True if not set)
    config = project.auto_ds_config_json or {}
    start_on_complete = config.get("start_on_pipeline_complete", True)
    if not start_on_complete:
        logger.debug(f"Auto DS start_on_pipeline_complete disabled for project {project_id}")
        return

    # Check if there's already an active Auto DS session
    if project.active_auto_ds_session_id:
        active_session = db.query(AutoDSSession).filter(
            AutoDSSession.id == project.active_auto_ds_session_id
        ).first()
        if active_session and active_session.status in [
            AutoDSSessionStatus.PENDING,
            AutoDSSessionStatus.RUNNING,
            AutoDSSessionStatus.PAUSED,
        ]:
            logger.info(f"Project {project_id} already has an active Auto DS session")
            return

    # Check if there's at least one dataset in the project
    dataset_count = db.query(DatasetSpec).filter(
        DatasetSpec.project_id == project_id
    ).count()
    if dataset_count == 0:
        logger.debug(f"No datasets in project {project_id}, skipping Auto DS trigger")
        return

    # Create new Auto DS session
    logger.info(f"Triggering Auto DS session for project {project_id} after pipeline completion")

    session = AutoDSSession(
        project_id=project_id,
        name=f"Auto DS - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        status=AutoDSSessionStatus.PENDING,
        max_iterations=config.get("max_iterations", 10),
        accuracy_threshold=config.get("accuracy_threshold"),
        time_budget_minutes=config.get("time_budget_minutes"),
        current_iteration=0,
        total_experiments_run=0,
    )
    db.add(session)
    db.flush()

    # Update project with active session
    project.active_auto_ds_session_id = session.id
    db.commit()

    # Start the Celery task
    task = run_auto_ds_session.delay(str(session.id))

    # Update session with task ID
    session.celery_task_id = task.id
    db.commit()

    logger.info(f"Auto DS session {session.id} started with task {task.id}")


def get_agent_run_with_steps(
    db: Session,
    run_id: UUID,
    include_logs: bool = False,
) -> Optional[AgentRun]:
    """Get an agent run with its steps (and optionally logs).

    Args:
        db: Database session
        run_id: UUID of the agent run
        include_logs: Whether to include step logs

    Returns:
        AgentRun with steps loaded, or None if not found
    """
    agent_run = db.query(AgentRun).filter(AgentRun.id == run_id).first()
    if not agent_run:
        return None

    # Force load steps
    _ = agent_run.steps

    if include_logs:
        for step in agent_run.steps:
            _ = step.logs

    return agent_run


def list_agent_runs_for_project(
    db: Session,
    project_id: UUID,
    skip: int = 0,
    limit: int = 20,
) -> tuple[List[AgentRun], int]:
    """List agent runs for a project.

    Args:
        db: Database session
        project_id: UUID of the project
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        Tuple of (list of runs, total count)
    """
    query = db.query(AgentRun).filter(AgentRun.project_id == project_id)
    total = query.count()
    runs = query.order_by(AgentRun.created_at.desc()).offset(skip).limit(limit).all()
    return runs, total


# ============================================
# Experiment Results Pipeline
# ============================================

# Define the pipeline steps for experiment results analysis
RESULTS_PIPELINE_STEPS: List[AgentStepType] = [
    AgentStepType.RESULTS_INTERPRETATION,
    AgentStepType.RESULTS_CRITIC,
]


def create_results_pipeline(
    db: Session,
    experiment_id: UUID,
) -> AgentRun:
    """Create an agent run with results pipeline steps for an experiment.

    This creates the agent_run and all agent_steps in pending state,
    ready to be executed.

    Args:
        db: Database session
        experiment_id: UUID of the experiment to analyze

    Returns:
        The created AgentRun with steps
    """
    # Verify experiment exists and is completed
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")

    if experiment.status != "completed":
        raise ValueError(f"Experiment must be completed to run results pipeline. Current status: {experiment.status}")

    # Create the agent run
    agent_run = AgentRun(
        project_id=experiment.project_id,
        experiment_id=experiment_id,
        name=f"Results Analysis for {experiment.name or 'Experiment'}",
        description=f"AI-powered analysis of experiment results",
        status=AgentRunStatus.PENDING,
        config_json={
            "experiment_id": str(experiment_id),
        },
    )
    db.add(agent_run)
    db.commit()
    db.refresh(agent_run)

    # Build task_context for results steps (Prompt 7 Step 2)
    results_task_context = _build_task_context_for_step(
        db=db,
        project_id=str(experiment.project_id),
        experiment_id=str(experiment_id),
    )

    # Create steps
    for step_type in RESULTS_PIPELINE_STEPS:
        if step_type == AgentStepType.RESULTS_INTERPRETATION:
            input_json = {
                "experiment_id": str(experiment_id),
            }
        elif step_type == AgentStepType.RESULTS_CRITIC:
            # Will receive results_interpretation from accumulated outputs
            input_json = {
                "experiment_id": str(experiment_id),
            }
        else:
            input_json = {}

        # Include task_context for context persistence (Prompt 7 Step 2)
        if results_task_context:
            input_json["task_context"] = results_task_context

        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=step_type,
            status=AgentStepStatus.PENDING,
            input_json=input_json,
        )
        db.add(step)

    db.commit()
    db.refresh(agent_run)

    return agent_run


async def run_results_pipeline_for_experiment(
    db: Session,
    experiment_id: UUID,
    llm_client: Optional[BaseLLMClient] = None,
) -> AgentRun:
    """Create and run the results analysis pipeline for an experiment.

    This function:
    1. Creates an agent_run with experiment_id, status="pending"
    2. Creates 2 agent_steps (RESULTS_INTERPRETATION → RESULTS_CRITIC)
    3. Executes each step sequentially, passing outputs between steps
    4. Returns the completed agent_run

    Args:
        db: Database session
        experiment_id: UUID of the experiment to analyze
        llm_client: Optional LLM client

    Returns:
        The completed AgentRun with all step outputs

    Raises:
        ValueError: If experiment not found or not completed
        Exception: If any step fails
    """
    # Create the pipeline
    agent_run = create_results_pipeline(
        db=db,
        experiment_id=experiment_id,
    )

    # Run the pipeline
    completed_run = await run_agent_pipeline(db, agent_run.id, llm_client)

    # Create a lab notebook entry if pipeline completed successfully
    if completed_run.status == AgentRunStatus.COMPLETED:
        try:
            create_results_pipeline_notebook_entry(db, completed_run, experiment_id)
        except Exception as e:
            logger.warning(f"Failed to create lab notebook entry for results pipeline: {e}")

    return completed_run


# ============================================
# Dataset Discovery Pipeline
# ============================================

def create_dataset_discovery_pipeline(
    db: Session,
    project_id: UUID,
    project_description: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> AgentRun:
    """Create an agent run with a single DATASET_DISCOVERY step.

    This pipeline runs before any data sources exist, to help users
    find relevant public datasets for their ML task.

    Args:
        db: Database session
        project_id: UUID of the project
        project_description: User's description of what they want to predict
        constraints: Optional constraints like geography, licensing, allow_public_data

    Returns:
        The created AgentRun with the discovery step
    """
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project not found: {project_id}")

    # Create the agent run
    agent_run = AgentRun(
        project_id=project_id,
        name=f"Dataset Discovery for {project.name}",
        description=f"Finding relevant public datasets for: {project_description[:100]}...",
        status=AgentRunStatus.PENDING,
        config_json={
            "project_description": project_description,
            "constraints": constraints or {},
        },
    )
    db.add(agent_run)
    db.commit()
    db.refresh(agent_run)

    # Build task_context for dataset discovery (Prompt 7 Step 2)
    discovery_task_context = _build_task_context_for_step(
        db=db,
        project_id=str(project_id),
    )

    # Create the single DATASET_DISCOVERY step
    discovery_input = {
        "project_description": project_description,
        "constraints": constraints or {},
    }
    if discovery_task_context:
        discovery_input["task_context"] = discovery_task_context

    step = AgentStep(
        agent_run_id=agent_run.id,
        step_type=AgentStepType.DATASET_DISCOVERY,
        status=AgentStepStatus.PENDING,
        input_json=discovery_input,
    )
    db.add(step)
    db.commit()
    db.refresh(agent_run)

    return agent_run


async def run_dataset_discovery_pipeline(
    db: Session,
    project_id: UUID,
    project_description: str,
    constraints: Optional[Dict[str, Any]] = None,
    llm_client: Optional[BaseLLMClient] = None,
) -> AgentRun:
    """Create and run the dataset discovery pipeline for a project.

    This function:
    1. Creates an agent_run with project_id, status="pending"
    2. Creates a single DATASET_DISCOVERY step
    3. Executes the step to find relevant public datasets
    4. Returns the completed agent_run with discovered datasets in output

    Args:
        db: Database session
        project_id: UUID of the project
        project_description: User's description of the ML goal
        constraints: Optional constraints (geography, licensing, etc.)
        llm_client: Optional LLM client

    Returns:
        The completed AgentRun with discovered datasets in result_json

    Raises:
        ValueError: If project not found
        Exception: If the step fails
    """
    # Create the pipeline
    agent_run = create_dataset_discovery_pipeline(
        db=db,
        project_id=project_id,
        project_description=project_description,
        constraints=constraints,
    )

    # Run the pipeline
    return await run_agent_pipeline(db, agent_run.id, llm_client)


# ============================================
# Data Architect Pipeline
# ============================================

# Define the pipeline steps for Data Architect
DATA_ARCHITECT_PIPELINE_STEPS: List[AgentStepType] = [
    AgentStepType.DATASET_INVENTORY,
    AgentStepType.RELATIONSHIP_DISCOVERY,
    AgentStepType.TRAINING_DATASET_PLANNING,
    AgentStepType.TRAINING_DATASET_BUILD,
]


def create_data_architect_pipeline(
    db: Session,
    project_id: UUID,
    target_hint: Optional[str] = None,
) -> AgentRun:
    """Create an agent run with Data Architect pipeline steps.

    This creates the agent_run and all agent_steps in pending state,
    ready to be executed. The pipeline has 4 steps:
    1. DATASET_INVENTORY - Profile all data sources
    2. RELATIONSHIP_DISCOVERY - Discover relationships between tables
    3. TRAINING_DATASET_PLANNING - Plan the training dataset with LLM
    4. TRAINING_DATASET_BUILD - Build and register the training dataset

    Args:
        db: Database session
        project_id: UUID of the project
        target_hint: Optional hint about which column is the target

    Returns:
        The created AgentRun with steps
    """
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project not found: {project_id}")

    # Verify project has data sources
    data_source_count = db.query(DataSource).filter(
        DataSource.project_id == project_id
    ).count()
    if data_source_count == 0:
        raise ValueError("Project has no data sources. Please add data sources first.")

    # Create the agent run
    agent_run = AgentRun(
        project_id=project_id,
        name=f"Data Architect - {project.name}",
        description="Automated training dataset construction pipeline",
        status=AgentRunStatus.PENDING,
        config_json={
            "target_hint": target_hint,
            "project_description": project.description or f"ML project: {project.name}",
        },
    )
    db.add(agent_run)
    db.commit()
    db.refresh(agent_run)

    # Build task_context for data architect steps (Prompt 7 Step 2)
    architect_task_context = _build_task_context_for_step(
        db=db,
        project_id=str(project_id),
    )

    # Create each step with initial input
    for i, step_type in enumerate(DATA_ARCHITECT_PIPELINE_STEPS):
        input_json = {
            "project_id": str(project_id),
        }

        # First step gets the target_hint
        if step_type == AgentStepType.DATASET_INVENTORY:
            pass  # Just needs project_id

        # Subsequent steps get inputs from previous step outputs (handled at runtime)

        # Include task_context for context persistence (Prompt 7 Step 2)
        if architect_task_context:
            input_json["task_context"] = architect_task_context

        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=step_type,
            status=AgentStepStatus.PENDING,
            input_json=input_json,
        )
        db.add(step)

    db.commit()
    db.refresh(agent_run)

    return agent_run


async def run_data_architect_pipeline(
    db: Session,
    project_id: UUID,
    target_hint: Optional[str] = None,
    llm_client: Optional[BaseLLMClient] = None,
) -> AgentRun:
    """Create and run the Data Architect pipeline for a project.

    This function:
    1. Creates an agent_run for the Data Architect pipeline
    2. Executes 4 steps sequentially:
       - DATASET_INVENTORY: Profile all data sources
       - RELATIONSHIP_DISCOVERY: Discover relationships
       - TRAINING_DATASET_PLANNING: Plan with LLM
       - TRAINING_DATASET_BUILD: Build the dataset
    3. Passes outputs between steps as inputs
    4. Returns the completed agent_run

    Args:
        db: Database session
        project_id: UUID of the project
        target_hint: Optional hint about which column is the target
        llm_client: Optional LLM client

    Returns:
        The completed AgentRun with training_data_source_id in result_json

    Raises:
        ValueError: If project not found or has no data sources
        Exception: If any step fails
    """
    # Create the pipeline
    agent_run = create_data_architect_pipeline(
        db=db,
        project_id=project_id,
        target_hint=target_hint,
    )

    # Get the project for description
    project = db.query(Project).filter(Project.id == project_id).first()
    project_description = project.description or f"ML project: {project.name}"

    # Mark run as running
    agent_run.status = AgentRunStatus.RUNNING
    db.commit()

    try:
        # Get steps in the correct pipeline order (by step_type, not created_at)
        # created_at can have identical timestamps causing random order
        step_order = {step_type: idx for idx, step_type in enumerate(DATA_ARCHITECT_PIPELINE_STEPS)}
        steps = sorted(
            agent_run.steps,
            key=lambda s: step_order.get(s.step_type, 999)
        )

        # Track outputs from each step
        data_source_profiles = None
        relationships_summary = None
        training_dataset_spec = None

        for step in steps:
            step_logger = StepLogger(db, step.id)

            # Update input based on previous step outputs
            if step.step_type == AgentStepType.RELATIONSHIP_DISCOVERY:
                if data_source_profiles:
                    step.input_json = {
                        "project_id": str(project_id),
                        "data_source_profiles": data_source_profiles,
                    }
                    db.commit()

            elif step.step_type == AgentStepType.TRAINING_DATASET_PLANNING:
                if data_source_profiles and relationships_summary:
                    step.input_json = {
                        "project_id": str(project_id),
                        "project_description": project_description,
                        "target_hint": target_hint,
                        "data_source_profiles": data_source_profiles,
                        "relationships_summary": relationships_summary,
                    }
                    db.commit()

            elif step.step_type == AgentStepType.TRAINING_DATASET_BUILD:
                if training_dataset_spec:
                    step.input_json = {
                        "project_id": str(project_id),
                        "training_dataset_spec": training_dataset_spec,
                    }
                    db.commit()

            # Run the step
            step = await run_agent_step(db, step.id, llm_client)

            if step.status != AgentStepStatus.COMPLETED:
                raise Exception(f"Step {step.step_type.value} failed: {step.error_message}")

            # Extract outputs for next step
            output = step.output_json or {}

            if step.step_type == AgentStepType.DATASET_INVENTORY:
                data_source_profiles = output.get("data_source_profiles", [])

            elif step.step_type == AgentStepType.RELATIONSHIP_DISCOVERY:
                relationships_summary = output.get("relationships_summary", {})

            elif step.step_type == AgentStepType.TRAINING_DATASET_PLANNING:
                training_dataset_spec = output.get("training_dataset_spec")

        # Mark run as completed
        agent_run.status = AgentRunStatus.COMPLETED

        # Set result_json with final outputs
        last_step = steps[-1] if steps else None
        if last_step and last_step.output_json:
            agent_run.result_json = {
                "training_data_source_id": last_step.output_json.get("data_source_id"),
                "row_count": last_step.output_json.get("row_count"),
                "column_count": last_step.output_json.get("column_count"),
                "target_column": last_step.output_json.get("target_column"),
                "feature_columns": last_step.output_json.get("feature_columns"),
            }

        db.commit()
        db.refresh(agent_run)

        return agent_run

    except Exception as e:
        # Mark run as failed
        agent_run.status = AgentRunStatus.FAILED
        agent_run.error_message = str(e)
        db.commit()
        raise


# ============================================
# Lab Notebook Summary Pipeline
# ============================================

async def trigger_lab_notebook_summary(
    db: Session,
    research_cycle_id: UUID,
    project_id: UUID,
    llm_client: Optional[BaseLLMClient] = None,
) -> AgentRun:
    """Trigger lab notebook summary generation for a research cycle.

    This is the orchestrator hook that should be called after a research cycle
    completes (i.e., after experiments have run and results critiques are done).

    It creates an agent run with a single LAB_NOTEBOOK_SUMMARY step that:
    - Gathers all experiment results from the cycle
    - Collects step outputs from the setup and results pipelines
    - Uses the LLM to generate a comprehensive Markdown summary
    - Creates a lab notebook entry with author_type='agent'

    Args:
        db: Database session
        research_cycle_id: UUID of the research cycle to summarize
        project_id: UUID of the project
        llm_client: Optional LLM client (created if not provided)

    Returns:
        The completed AgentRun with lab_note in result_json

    Raises:
        ValueError: If research cycle not found
        Exception: If the step fails
    """
    # Verify research cycle exists
    research_cycle = db.query(ResearchCycle).filter(
        ResearchCycle.id == research_cycle_id
    ).first()
    if not research_cycle:
        raise ValueError(f"Research cycle not found: {research_cycle_id}")

    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project not found: {project_id}")

    # Create the agent run
    agent_run = AgentRun(
        project_id=project_id,
        research_cycle_id=research_cycle_id,
        name=f"Lab Notebook Summary - Cycle #{research_cycle.sequence_number}",
        description="AI-generated summary of research cycle findings",
        status=AgentRunStatus.PENDING,
        config_json={
            "research_cycle_id": str(research_cycle_id),
            "project_id": str(project_id),
        },
    )
    db.add(agent_run)
    db.commit()
    db.refresh(agent_run)

    # Build task_context for lab notebook summary (Prompt 7 Step 2)
    notebook_task_context = _build_task_context_for_step(
        db=db,
        project_id=str(project_id),
        research_cycle_id=str(research_cycle_id),
    )

    # Create the single summary step
    summary_input = {
        "research_cycle_id": str(research_cycle_id),
        "project_id": str(project_id),
    }
    if notebook_task_context:
        summary_input["task_context"] = notebook_task_context

    step = AgentStep(
        agent_run_id=agent_run.id,
        step_type=AgentStepType.LAB_NOTEBOOK_SUMMARY,
        status=AgentStepStatus.PENDING,
        input_json=summary_input,
    )
    db.add(step)
    db.commit()

    # Run the pipeline
    return await run_agent_pipeline(db, agent_run.id, llm_client)
"""Project History API - Unified view of all project history data.

This module provides a single endpoint that aggregates:
- Research cycles with their experiments
- Agent runs with their steps and tool calls
- Lab notebook entries
- Best models and experiment results

This enables the frontend to display a comprehensive history view
showing all previous iterations, agent reasoning, and tool calls.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import (
    AgentRun,
    AgentStep,
    AgentStepLog,
    AgentStepType,
    AgentStepStatus,
    Project,
    Experiment,
    Trial,
    User,
    LogMessageType,
)
from app.models.research_cycle import (
    ResearchCycle,
    ResearchCycleStatus,
    CycleExperiment,
    LabNotebookEntry,
    LabNotebookAuthorType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}/history", tags=["Project History"])


# =============================================================================
# Response Schemas
# =============================================================================

class ToolCallDetail(BaseModel):
    """Details of a single tool call made by an agent."""
    name: str
    arguments: Dict[str, Any]
    result_preview: Optional[str] = None
    timestamp: Optional[datetime] = None


class AgentThinkingDetail(BaseModel):
    """Agent thinking/reasoning from a step."""
    step_id: UUID
    step_type: str
    step_name: Optional[str] = None
    status: str
    tool_calls: List[ToolCallDetail] = []
    thinking_log: List[str] = []  # Filtered for THINKING log type
    observation_log: List[str] = []  # Filtered for OBSERVATION log type
    action_log: List[str] = []  # Filtered for ACTION log type
    summary: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class AgentRunDetail(BaseModel):
    """Details of an agent run with all its steps and reasoning."""
    id: UUID
    run_type: Optional[str] = None
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_steps: int
    completed_steps: int
    steps: List[AgentThinkingDetail] = []


class ExperimentResultDetail(BaseModel):
    """Details of an experiment result."""
    id: UUID
    name: str
    status: str
    primary_metric: Optional[str] = None
    best_score: Optional[float] = None
    trial_count: int
    best_trial_name: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class ResearchCycleDetail(BaseModel):
    """Details of a research cycle with experiments."""
    id: UUID
    sequence_number: int
    status: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    experiments: List[ExperimentResultDetail] = []
    agent_runs: List[AgentRunDetail] = []


class NotebookEntryDetail(BaseModel):
    """Details of a lab notebook entry."""
    id: UUID
    title: str
    body_markdown: Optional[str] = None
    author_type: str
    research_cycle_id: Optional[UUID] = None
    agent_step_id: Optional[UUID] = None
    created_at: datetime


class BestModelDetail(BaseModel):
    """Details of a best-performing model."""
    experiment_id: UUID
    experiment_name: str
    trial_id: UUID
    trial_name: str
    metric_name: str
    metric_value: float
    model_path: Optional[str] = None
    research_cycle_id: Optional[UUID] = None


class ProjectHistoryResponse(BaseModel):
    """Complete project history response."""
    project_id: UUID
    project_name: str
    total_cycles: int
    total_experiments: int
    total_notebook_entries: int

    # All research cycles with their experiments and agent runs
    research_cycles: List[ResearchCycleDetail] = []

    # All notebook entries (human and agent)
    notebook_entries: List[NotebookEntryDetail] = []

    # Best models across all experiments
    best_models: List[BestModelDetail] = []

    # Recent activity timestamps
    last_experiment_at: Optional[datetime] = None
    last_agent_run_at: Optional[datetime] = None


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_tool_calls_from_logs(logs: List[AgentStepLog]) -> List[ToolCallDetail]:
    """Extract tool call details from agent step logs."""
    tool_calls = []
    for log in logs:
        if log.message_type == LogMessageType.OBSERVATION:
            message = log.message or ""
            # Tool calls are logged with "Calling tool:" prefix
            if message.startswith("Calling tool:"):
                # Parse tool name and args from message
                try:
                    # Format: "Calling tool: tool_name({args...})"
                    tool_part = message.replace("Calling tool:", "").strip()
                    if "(" in tool_part:
                        name = tool_part.split("(")[0].strip()
                        tool_calls.append(ToolCallDetail(
                            name=name,
                            arguments={},  # Args are truncated in logs
                            timestamp=log.created_at,
                        ))
                except Exception:
                    pass
            # Tool results are logged with "Tool result:" prefix
            elif message.startswith("Tool result:") and tool_calls:
                # Add result preview to the most recent tool call
                tool_calls[-1].result_preview = message[:500]
    return tool_calls


def _build_agent_thinking_detail(step: AgentStep, db: Session) -> AgentThinkingDetail:
    """Build agent thinking detail from a step."""
    # Get all logs for this step
    logs = db.query(AgentStepLog).filter(
        AgentStepLog.step_id == step.id
    ).order_by(AgentStepLog.created_at).all()

    # Categorize logs by type
    thinking_log = []
    observation_log = []
    action_log = []
    summary = None

    for log in logs:
        if log.message_type == LogMessageType.THINKING:
            thinking_log.append(log.message or "")
        elif log.message_type == LogMessageType.OBSERVATION:
            observation_log.append(log.message or "")
        elif log.message_type == LogMessageType.ACTION:
            action_log.append(log.message or "")
        elif log.message_type == LogMessageType.SUMMARY:
            summary = log.message

    # Extract tool calls
    tool_calls = _extract_tool_calls_from_logs(logs)

    status_value = step.status.value if hasattr(step.status, 'value') else str(step.status)
    step_type_value = step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type)

    return AgentThinkingDetail(
        step_id=step.id,
        step_type=step_type_value,
        step_name=step.name,
        status=status_value,
        tool_calls=tool_calls,
        thinking_log=thinking_log,
        observation_log=observation_log,
        action_log=action_log,
        summary=summary,
        created_at=step.created_at,
        completed_at=step.completed_at,
    )


def _build_agent_run_detail(run: AgentRun, db: Session) -> AgentRunDetail:
    """Build agent run detail with all steps."""
    steps = db.query(AgentStep).filter(
        AgentStep.agent_run_id == run.id
    ).order_by(AgentStep.order).all()

    step_details = [_build_agent_thinking_detail(step, db) for step in steps]

    completed = sum(1 for s in steps if s.status == AgentStepStatus.COMPLETED)

    status_value = run.status.value if hasattr(run.status, 'value') else str(run.status)

    return AgentRunDetail(
        id=run.id,
        run_type=getattr(run, 'run_type', None),
        status=status_value,
        created_at=run.created_at,
        completed_at=run.completed_at,
        total_steps=len(steps),
        completed_steps=completed,
        steps=step_details,
    )


def _get_best_score_for_experiment(experiment: Experiment, db: Session) -> tuple:
    """Get the best score and trial for an experiment.

    Returns: (best_score, best_trial_name)
    """
    if not experiment.primary_metric:
        return None, None

    trials = db.query(Trial).filter(Trial.experiment_id == experiment.id).all()

    best_score = None
    best_trial_name = None

    for trial in trials:
        if trial.metrics_json and experiment.primary_metric in trial.metrics_json:
            score = trial.metrics_json[experiment.primary_metric]
            if score is not None:
                if best_score is None or score > best_score:
                    best_score = score
                    best_trial_name = trial.name

    return best_score, best_trial_name


# =============================================================================
# Endpoint
# =============================================================================

@router.get("", response_model=ProjectHistoryResponse)
def get_project_history(
    project_id: UUID,
    include_logs: bool = Query(True, description="Include detailed agent logs"),
    limit_cycles: int = Query(20, description="Max number of cycles to return"),
    limit_entries: int = Query(50, description="Max number of notebook entries"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get complete project history for the History View UI.

    Returns all research cycles, experiments, agent runs with their
    reasoning/tool calls, notebook entries, and best models.

    This endpoint aggregates data from multiple sources to provide
    a comprehensive view of what has been tried and learned.
    """
    # Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Get research cycles with experiments
    cycles = db.query(ResearchCycle).filter(
        ResearchCycle.project_id == project_id
    ).order_by(ResearchCycle.sequence_number.desc()).limit(limit_cycles).all()

    research_cycle_details = []
    best_models = []
    all_experiments_count = 0
    last_experiment_at = None

    for cycle in cycles:
        # Get experiments for this cycle
        cycle_exp_links = db.query(CycleExperiment).filter(
            CycleExperiment.research_cycle_id == cycle.id
        ).all()

        experiment_details = []
        for link in cycle_exp_links:
            exp = db.query(Experiment).filter(Experiment.id == link.experiment_id).first()
            if exp:
                all_experiments_count += 1
                trial_count = db.query(Trial).filter(Trial.experiment_id == exp.id).count()
                best_score, best_trial_name = _get_best_score_for_experiment(exp, db)

                status_value = exp.status.value if hasattr(exp.status, 'value') else str(exp.status)

                experiment_details.append(ExperimentResultDetail(
                    id=exp.id,
                    name=exp.name,
                    status=status_value,
                    primary_metric=exp.primary_metric,
                    best_score=best_score,
                    trial_count=trial_count,
                    best_trial_name=best_trial_name,
                    created_at=exp.created_at,
                    completed_at=exp.completed_at,
                ))

                # Track latest experiment
                if exp.created_at and (not last_experiment_at or exp.created_at > last_experiment_at):
                    last_experiment_at = exp.created_at

                # Add to best models if it has a best score
                if best_score is not None and exp.primary_metric:
                    # Find the actual trial
                    best_trial = db.query(Trial).filter(
                        Trial.experiment_id == exp.id,
                        Trial.name == best_trial_name
                    ).first()

                    best_models.append(BestModelDetail(
                        experiment_id=exp.id,
                        experiment_name=exp.name,
                        trial_id=best_trial.id if best_trial else exp.id,
                        trial_name=best_trial_name or "Unknown",
                        metric_name=exp.primary_metric,
                        metric_value=best_score,
                        model_path=best_trial.model_path if best_trial else None,
                        research_cycle_id=cycle.id,
                    ))

        # Get agent runs for this cycle
        agent_runs = db.query(AgentRun).filter(
            AgentRun.project_id == project_id,
            AgentRun.research_cycle_id == cycle.id
        ).order_by(AgentRun.created_at.desc()).all()

        agent_run_details = []
        if include_logs:
            agent_run_details = [_build_agent_run_detail(run, db) for run in agent_runs]
        else:
            # Just basic info without logs
            for run in agent_runs:
                status_value = run.status.value if hasattr(run.status, 'value') else str(run.status)
                agent_run_details.append(AgentRunDetail(
                    id=run.id,
                    status=status_value,
                    created_at=run.created_at,
                    completed_at=run.completed_at,
                    total_steps=0,
                    completed_steps=0,
                ))

        status_value = cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)

        research_cycle_details.append(ResearchCycleDetail(
            id=cycle.id,
            sequence_number=cycle.sequence_number,
            status=status_value,
            title=cycle.summary_title,
            created_at=cycle.created_at,
            updated_at=cycle.updated_at,
            experiments=experiment_details,
            agent_runs=agent_run_details,
        ))

    # Get notebook entries
    entries = db.query(LabNotebookEntry).filter(
        LabNotebookEntry.project_id == project_id
    ).order_by(LabNotebookEntry.created_at.desc()).limit(limit_entries).all()

    notebook_details = [
        NotebookEntryDetail(
            id=entry.id,
            title=entry.title,
            body_markdown=entry.body_markdown,
            author_type=entry.author_type.value if hasattr(entry.author_type, 'value') else str(entry.author_type),
            research_cycle_id=entry.research_cycle_id,
            agent_step_id=entry.agent_step_id,
            created_at=entry.created_at,
        )
        for entry in entries
    ]

    # Sort best models by metric value (descending)
    best_models.sort(key=lambda x: x.metric_value, reverse=True)
    best_models = best_models[:10]  # Top 10

    # Get last agent run timestamp
    last_agent_run = db.query(AgentRun).filter(
        AgentRun.project_id == project_id
    ).order_by(AgentRun.created_at.desc()).first()

    return ProjectHistoryResponse(
        project_id=project_id,
        project_name=project.name,
        total_cycles=len(cycles),
        total_experiments=all_experiments_count,
        total_notebook_entries=len(entries),
        research_cycles=research_cycle_details,
        notebook_entries=notebook_details,
        best_models=best_models,
        last_experiment_at=last_experiment_at,
        last_agent_run_at=last_agent_run.created_at if last_agent_run else None,
    )


@router.get("/agent-step/{step_id}/thinking", response_model=AgentThinkingDetail)
def get_agent_step_thinking(
    project_id: UUID,
    step_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get detailed thinking/reasoning for a specific agent step.

    Returns all logs categorized by type (thinking, observation, action),
    tool calls made, and the final summary.
    """
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Get step
    step = db.query(AgentStep).filter(AgentStep.id == step_id).first()
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent step {step_id} not found",
        )

    # Verify step belongs to this project
    run = db.query(AgentRun).filter(AgentRun.id == step.agent_run_id).first()
    if not run or run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent step {step_id} not found in project {project_id}",
        )

    return _build_agent_thinking_detail(step, db)

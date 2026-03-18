"""Auto DS session API endpoints."""
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.api.dependencies import check_project_access
from app.models.user import User
from app.models.project import Project
from app.models.auto_ds_session import (
    AutoDSSession,
    AutoDSSessionStatus,
    AutoDSIteration,
    AutoDSIterationStatus,
    AutoDSIterationExperiment,
    ResearchInsight,
    GlobalInsight,
)
from app.models.experiment import Experiment
from app.models.dataset_spec import DatasetSpec
from app.schemas.auto_ds import (
    AutoDSSessionCreate,
    AutoDSSessionUpdate,
    AutoDSSessionResponse,
    AutoDSSessionSummary,
    AutoDSSessionListResponse,
    AutoDSSessionStartRequest,
    AutoDSSessionStartResponse,
    AutoDSSessionProgress,
    AutoDSIterationResponse,
    AutoDSIterationSummary,
    IterationExperimentInfo,
    ResearchInsightResponse,
    ResearchInsightListResponse,
    GlobalInsightResponse,
    GlobalInsightListResponse,
)
from app.services.auto_ds_orchestrator import AutoDSOrchestrator, get_experiment_all_scores
from app.core.task_dispatch import dispatch_task

router = APIRouter(tags=["auto_ds"])


# ============================================
# Auto DS Session Endpoints
# ============================================

@router.post(
    "/projects/{project_id}/auto-ds-sessions",
    response_model=AutoDSSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_auto_ds_session(
    project_id: UUID,
    session_data: AutoDSSessionCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new Auto DS session for a project."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    orchestrator = AutoDSOrchestrator(db)
    session = orchestrator.create_session(
        project_id=project_id,
        name=session_data.name,
        description=session_data.description,
        max_iterations=session_data.max_iterations,
        accuracy_threshold=session_data.accuracy_threshold,
        time_budget_minutes=session_data.time_budget_minutes,
        min_improvement_threshold=session_data.min_improvement_threshold,
        plateau_iterations=session_data.plateau_iterations,
        max_experiments_per_dataset=session_data.max_experiments_per_dataset,
        max_active_datasets=session_data.max_active_datasets,
        config=session_data.config_json,
    )

    return _build_session_response(session)


@router.get(
    "/projects/{project_id}/auto-ds-sessions",
    response_model=AutoDSSessionListResponse,
)
def list_auto_ds_sessions(
    project_id: UUID,
    status_filter: Optional[AutoDSSessionStatus] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List Auto DS sessions for a project."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    query = db.query(AutoDSSession).filter(AutoDSSession.project_id == project_id)

    if status_filter:
        query = query.filter(AutoDSSession.status == status_filter)

    total = query.count()
    sessions = query.order_by(AutoDSSession.created_at.desc()).offset(skip).limit(limit).all()

    return AutoDSSessionListResponse(
        sessions=[AutoDSSessionSummary.model_validate(s) for s in sessions],
        total=total,
    )


@router.get(
    "/projects/{project_id}/auto-ds-sessions/{session_id}",
    response_model=AutoDSSessionResponse,
)
def get_auto_ds_session(
    project_id: UUID,
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get details of an Auto DS session."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == session_id,
        AutoDSSession.project_id == project_id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auto DS session {session_id} not found",
        )

    return _build_session_response(session)


@router.patch(
    "/projects/{project_id}/auto-ds-sessions/{session_id}",
    response_model=AutoDSSessionResponse,
)
def update_auto_ds_session(
    project_id: UUID,
    session_id: UUID,
    session_data: AutoDSSessionUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update an Auto DS session."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == session_id,
        AutoDSSession.project_id == project_id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auto DS session {session_id} not found",
        )

    # Only allow updates when session is pending or paused
    if session.status not in [AutoDSSessionStatus.PENDING, AutoDSSessionStatus.PAUSED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update session in {session.status.value} status",
        )

    # Apply updates
    update_data = session_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(session, key, value)

    session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(session)

    return _build_session_response(session)


@router.post(
    "/projects/{project_id}/auto-ds-sessions/{session_id}/start",
    response_model=AutoDSSessionStartResponse,
)
def start_auto_ds_session(
    project_id: UUID,
    session_id: UUID,
    request: Optional[AutoDSSessionStartRequest] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Start an Auto DS session."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == session_id,
        AutoDSSession.project_id == project_id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auto DS session {session_id} not found",
        )

    # Only allow starting from pending or paused status
    if session.status not in [AutoDSSessionStatus.PENDING, AutoDSSessionStatus.PAUSED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot start session in {session.status.value} status",
        )

    # Get initial dataset spec IDs if provided
    initial_dataset_spec_ids = None
    if request and request.initial_dataset_spec_ids:
        initial_dataset_spec_ids = [str(id) for id in request.initial_dataset_spec_ids]

    # Launch the background task
    print(f"🚀 API: Queueing task for session {session_id}")
    task = dispatch_task(
        "run_auto_ds_session",
        session_id=str(session_id),
        initial_dataset_spec_ids=initial_dataset_spec_ids,
    )
    print(f"🚀 API: Task queued with ID: {task.id}")

    return AutoDSSessionStartResponse(
        session_id=session_id,
        status=AutoDSSessionStatus.RUNNING,
        celery_task_id=task.id,
        message="Auto DS session started successfully",
    )


@router.post(
    "/projects/{project_id}/auto-ds-sessions/{session_id}/pause",
    response_model=AutoDSSessionResponse,
)
def pause_auto_ds_session(
    project_id: UUID,
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Pause a running Auto DS session."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == session_id,
        AutoDSSession.project_id == project_id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auto DS session {session_id} not found",
        )

    if session.status != AutoDSSessionStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot pause session in {session.status.value} status",
        )

    orchestrator = AutoDSOrchestrator(db)
    orchestrator.pause_session(session_id)

    db.refresh(session)
    return _build_session_response(session)


@router.post(
    "/projects/{project_id}/auto-ds-sessions/{session_id}/stop",
    response_model=AutoDSSessionResponse,
)
def stop_auto_ds_session(
    project_id: UUID,
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Stop an Auto DS session."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == session_id,
        AutoDSSession.project_id == project_id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auto DS session {session_id} not found",
        )

    if session.status not in [AutoDSSessionStatus.RUNNING, AutoDSSessionStatus.PAUSED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot stop session in {session.status.value} status",
        )

    orchestrator = AutoDSOrchestrator(db)
    orchestrator.stop_session(session_id)

    db.refresh(session)
    return _build_session_response(session)


@router.get(
    "/projects/{project_id}/auto-ds-sessions/{session_id}/progress",
    response_model=AutoDSSessionProgress,
)
def get_auto_ds_session_progress(
    project_id: UUID,
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get detailed progress of an Auto DS session."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == session_id,
        AutoDSSession.project_id == project_id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auto DS session {session_id} not found",
        )

    # Calculate elapsed time
    elapsed_minutes = None
    if session.started_at:
        now = datetime.utcnow()
        elapsed_minutes = (now - session.started_at).total_seconds() / 60

    # Get current iteration details
    current_iter_status = None
    current_iter_completed = 0
    current_iter_planned = 0

    if session.iterations:
        current_iter = max(session.iterations, key=lambda i: i.iteration_number)
        current_iter_status = current_iter.status
        current_iter_completed = current_iter.experiments_completed
        current_iter_planned = current_iter.experiments_planned

    # Check stopping conditions
    threshold_reached = False
    if session.accuracy_threshold and session.best_score:
        threshold_reached = session.best_score >= session.accuracy_threshold

    plateau_detected = session.iterations_without_improvement >= session.plateau_iterations

    return AutoDSSessionProgress(
        session_id=session.id,
        status=session.status,
        current_iteration=session.current_iteration,
        max_iterations=session.max_iterations,
        best_score=session.best_score,
        total_experiments_run=session.total_experiments_run,
        iterations_without_improvement=session.iterations_without_improvement,
        current_iteration_status=current_iter_status,
        current_iteration_experiments_completed=current_iter_completed,
        current_iteration_experiments_planned=current_iter_planned,
        started_at=session.started_at,
        elapsed_minutes=elapsed_minutes,
        time_budget_minutes=session.time_budget_minutes,
        accuracy_threshold=session.accuracy_threshold,
        threshold_reached=threshold_reached,
        plateau_detected=plateau_detected,
    )


@router.delete(
    "/projects/{project_id}/auto-ds-sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_auto_ds_session(
    project_id: UUID,
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Delete an Auto DS session."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == session_id,
        AutoDSSession.project_id == project_id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auto DS session {session_id} not found",
        )

    # Don't allow deleting running sessions
    if session.status == AutoDSSessionStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a running session. Stop it first.",
        )

    db.delete(session)
    db.commit()


# ============================================
# Iteration Endpoints
# ============================================

@router.get(
    "/projects/{project_id}/auto-ds-sessions/{session_id}/iterations",
    response_model=List[AutoDSIterationSummary],
)
def list_session_iterations(
    project_id: UUID,
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List iterations for an Auto DS session with experiments."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == session_id,
        AutoDSSession.project_id == project_id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auto DS session {session_id} not found",
        )

    iterations = db.query(AutoDSIteration).filter(
        AutoDSIteration.session_id == session_id
    ).order_by(AutoDSIteration.iteration_number).all()

    result = []
    for iteration in iterations:
        # Build experiment list for this iteration
        experiments = _get_iteration_experiments(db, iteration.id)

        result.append(AutoDSIterationSummary(
            id=iteration.id,
            iteration_number=iteration.iteration_number,
            status=iteration.status,
            experiments_planned=iteration.experiments_planned,
            experiments_completed=iteration.experiments_completed,
            experiments_failed=iteration.experiments_failed,
            best_score_this_iteration=iteration.best_score_this_iteration,
            best_train_score_this_iteration=iteration.best_train_score_this_iteration,
            best_val_score_this_iteration=iteration.best_val_score_this_iteration,
            best_holdout_score_this_iteration=iteration.best_holdout_score_this_iteration,
            best_experiment_id_this_iteration=iteration.best_experiment_id_this_iteration,
            analysis_summary_json=iteration.analysis_summary_json,
            error_message=iteration.error_message,
            created_at=iteration.created_at,
            experiments=experiments,
        ))

    return result


def _get_iteration_experiments(db: Session, iteration_id: UUID) -> List[IterationExperimentInfo]:
    """Get experiments for an iteration."""
    # Get iteration experiments with related experiment and dataset
    iter_exps = db.query(AutoDSIterationExperiment).filter(
        AutoDSIterationExperiment.iteration_id == iteration_id
    ).all()

    experiments = []
    for ie in iter_exps:
        # Get the experiment
        exp = db.query(Experiment).filter(Experiment.id == ie.experiment_id).first()
        if not exp:
            continue

        # Get dataset name if available
        dataset_name = None
        if ie.dataset_spec_id:
            ds = db.query(DatasetSpec).filter(DatasetSpec.id == ie.dataset_spec_id).first()
            if ds:
                dataset_name = ds.name

        # Get all 3 scores for this experiment
        all_scores = get_experiment_all_scores(exp)
        
        experiments.append(IterationExperimentInfo(
            experiment_id=exp.id,
            experiment_name=exp.name,
            experiment_status=exp.status.value,
            dataset_spec_id=ie.dataset_spec_id,
            dataset_name=dataset_name,
            experiment_variant=ie.experiment_variant,
            hypothesis=ie.experiment_hypothesis,
            score=ie.score,
            train_score=all_scores["train_score"],
            val_score=all_scores["val_score"],
            holdout_score=all_scores["holdout_score"],
            rank_in_iteration=ie.rank_in_iteration,
            created_at=exp.created_at,
        ))

    return experiments


@router.get(
    "/projects/{project_id}/auto-ds-sessions/{session_id}/iterations/{iteration_id}",
    response_model=AutoDSIterationResponse,
)
def get_session_iteration(
    project_id: UUID,
    session_id: UUID,
    iteration_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get details of a specific iteration."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    iteration = db.query(AutoDSIteration).filter(
        AutoDSIteration.id == iteration_id,
        AutoDSIteration.session_id == session_id,
    ).first()

    if not iteration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Iteration {iteration_id} not found",
        )

    # Verify session belongs to project
    session = iteration.session
    if session.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Iteration {iteration_id} not found",
        )

    return AutoDSIterationResponse.model_validate(iteration)


# ============================================
# Research Insight Endpoints
# ============================================

@router.get(
    "/projects/{project_id}/auto-ds-sessions/{session_id}/insights",
    response_model=ResearchInsightListResponse,
)
def list_session_insights(
    project_id: UUID,
    session_id: UUID,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List research insights from an Auto DS session."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == session_id,
        AutoDSSession.project_id == project_id,
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Auto DS session {session_id} not found",
        )

    query = db.query(ResearchInsight).filter(ResearchInsight.session_id == session_id)
    total = query.count()
    insights = query.order_by(ResearchInsight.created_at.desc()).offset(skip).limit(limit).all()

    return ResearchInsightListResponse(
        insights=[ResearchInsightResponse.model_validate(i) for i in insights],
        total=total,
    )


@router.get(
    "/projects/{project_id}/insights",
    response_model=ResearchInsightListResponse,
)
def list_project_insights(
    project_id: UUID,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List all research insights for a project."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    query = db.query(ResearchInsight).filter(ResearchInsight.project_id == project_id)
    total = query.count()
    insights = query.order_by(ResearchInsight.created_at.desc()).offset(skip).limit(limit).all()

    return ResearchInsightListResponse(
        insights=[ResearchInsightResponse.model_validate(i) for i in insights],
        total=total,
    )


# ============================================
# Global Insight Endpoints
# ============================================

@router.get(
    "/global-insights",
    response_model=GlobalInsightListResponse,
)
def list_global_insights(
    skip: int = 0,
    limit: int = 50,
    active_only: bool = True,
    min_confidence: float = 0.0,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List global insights across all projects."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    query = db.query(GlobalInsight)

    if active_only:
        query = query.filter(GlobalInsight.is_active == True)

    if min_confidence > 0:
        query = query.filter(GlobalInsight.confidence_score >= min_confidence)

    total = query.count()
    insights = query.order_by(GlobalInsight.confidence_score.desc()).offset(skip).limit(limit).all()

    return GlobalInsightListResponse(
        insights=[GlobalInsightResponse.model_validate(i) for i in insights],
        total=total,
    )


@router.get(
    "/global-insights/{insight_id}",
    response_model=GlobalInsightResponse,
)
def get_global_insight(
    insight_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get details of a global insight."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    insight = db.query(GlobalInsight).filter(GlobalInsight.id == insight_id).first()

    if not insight:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Global insight {insight_id} not found",
        )

    return GlobalInsightResponse.model_validate(insight)


# ============================================
# Helper Functions
# ============================================

def _build_session_response(session: AutoDSSession) -> AutoDSSessionResponse:
    """Build a full session response with nested data."""
    iterations = [
        AutoDSIterationSummary(
            id=i.id,
            iteration_number=i.iteration_number,
            status=i.status,
            experiments_planned=i.experiments_planned,
            experiments_completed=i.experiments_completed,
            experiments_failed=i.experiments_failed,
            best_score_this_iteration=i.best_score_this_iteration,
            best_train_score_this_iteration=i.best_train_score_this_iteration,
            best_val_score_this_iteration=i.best_val_score_this_iteration,
            best_holdout_score_this_iteration=i.best_holdout_score_this_iteration,
            created_at=i.created_at,
        )
        for i in sorted(session.iterations, key=lambda x: x.iteration_number)
    ]

    return AutoDSSessionResponse(
        id=session.id,
        project_id=session.project_id,
        name=session.name,
        description=session.description,
        status=session.status,
        max_iterations=session.max_iterations,
        accuracy_threshold=session.accuracy_threshold,
        time_budget_minutes=session.time_budget_minutes,
        min_improvement_threshold=session.min_improvement_threshold,
        plateau_iterations=session.plateau_iterations,
        max_experiments_per_dataset=session.max_experiments_per_dataset,
        max_active_datasets=session.max_active_datasets,
        execution_mode=session.execution_mode,
        adaptive_decline_threshold=session.adaptive_decline_threshold,
        phased_min_baseline_improvement=session.phased_min_baseline_improvement,
        dynamic_experiments_per_cycle=session.dynamic_experiments_per_cycle,
        validation_strategy=session.validation_strategy,
        validation_num_seeds=session.validation_num_seeds,
        validation_cv_folds=session.validation_cv_folds,
        enable_feature_engineering=session.enable_feature_engineering,
        enable_ensemble=session.enable_ensemble,
        enable_ablation=session.enable_ablation,
        enable_diverse_configs=session.enable_diverse_configs,
        current_iteration=session.current_iteration,
        best_score=session.best_score,
        best_train_score=session.best_train_score,
        best_val_score=session.best_val_score,
        best_holdout_score=session.best_holdout_score,
        best_experiment_id=session.best_experiment_id,
        total_experiments_run=session.total_experiments_run,
        iterations_without_improvement=session.iterations_without_improvement,
        started_at=session.started_at,
        completed_at=session.completed_at,
        research_cycle_id=session.research_cycle_id,
        celery_task_id=session.celery_task_id,
        config_json=session.config_json,
        created_at=session.created_at,
        updated_at=session.updated_at,
        iterations=iterations,
    )

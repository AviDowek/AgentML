"""Project API endpoints."""
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.project import Project
from app.models.user import User
from app.models.sharing import ProjectShare, InviteStatus, ShareRole
from app.models.auto_ds_session import AutoDSSession, AutoDSSessionStatus
from app.models.dataset_spec import DatasetSpec
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
    AutoDSConfig,
)
from app.tasks.auto_ds_tasks import run_auto_ds_session

router = APIRouter(prefix="/projects", tags=["projects"])


def get_accessible_projects_query(db: Session, user: Optional[User]):
    """Build query for projects accessible to a user (owned or shared)."""
    if not user:
        # Return empty query if no user - require authentication
        return db.query(Project).filter(False)  # Always returns empty

    # Projects owned by user or shared with user (accepted)
    shared_project_ids = db.query(ProjectShare.project_id).filter(
        ProjectShare.user_id == user.id,
        ProjectShare.status == InviteStatus.ACCEPTED,
    ).subquery()

    return db.query(Project).filter(
        or_(
            Project.owner_id == user.id,
            Project.owner_id.is_(None),  # Legacy projects without owner
            Project.id.in_(shared_project_ids),
        )
    )


def check_project_access(
    db: Session, project: Project, user: Optional[User], require_write: bool = False
) -> bool:
    """Check if user has access to project."""
    if not user:
        return False  # Require authentication

    # Owner has full access
    if project.owner_id == user.id or project.owner_id is None:
        return True

    # Check sharing
    share = db.query(ProjectShare).filter(
        ProjectShare.project_id == project.id,
        ProjectShare.user_id == user.id,
        ProjectShare.status == InviteStatus.ACCEPTED,
    ).first()

    if not share:
        return False

    if require_write and share.role == ShareRole.VIEWER:
        return False

    return True


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new project."""
    db_project = Project(
        name=project.name,
        description=project.description,
        task_type=project.task_type,
        owner_id=current_user.id if current_user else None,
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


@router.get("", response_model=ProjectListResponse)
def list_projects(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List all projects accessible to the current user."""
    query = get_accessible_projects_query(db, current_user)
    total = query.count()
    projects = query.offset(skip).limit(limit).all()
    return ProjectListResponse(items=projects, total=total)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a project by ID."""
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

    return project


@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: UUID,
    project_update: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to modify this project",
        )

    update_data = project_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)

    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Delete a project (owner only)."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Only owner can delete
    if current_user and project.owner_id and project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the project owner can delete this project",
        )

    db.delete(project)
    db.commit()
    return None


# ============= Auto DS Endpoints =============


@router.post("/{project_id}/auto-ds/start")
def start_auto_ds_session(
    project_id: UUID,
    dataset_spec_ids: Optional[List[UUID]] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Start an Auto DS session for the project.

    If dataset_spec_ids is provided, only those datasets will be used.
    Otherwise, all datasets in the project will be used.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to start Auto DS on this project",
        )

    # Check if there's already an active session
    if project.active_auto_ds_session_id:
        active_session = db.query(AutoDSSession).filter(
            AutoDSSession.id == project.active_auto_ds_session_id
        ).first()
        if active_session and active_session.status in [
            AutoDSSessionStatus.PENDING,
            AutoDSSessionStatus.RUNNING,
            AutoDSSessionStatus.PAUSED,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Project already has an active Auto DS session: {active_session.name}",
            )

    # Get config from project or use defaults
    config = project.auto_ds_config_json or {}

    # Create new Auto DS session
    session = AutoDSSession(
        project_id=project_id,
        name=f"Auto DS - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        status=AutoDSSessionStatus.PENDING,
        max_iterations=config.get("max_iterations", 10),
        accuracy_threshold=config.get("accuracy_threshold"),
        time_budget_minutes=config.get("time_budget_minutes"),
        max_experiments_per_dataset=config.get("max_experiments_per_dataset", 3),
        max_active_datasets=config.get("max_active_datasets", 5),
        current_iteration=0,
        total_experiments_run=0,
    )
    db.add(session)
    db.flush()

    # Update project with active session
    project.active_auto_ds_session_id = session.id
    db.commit()

    # Start the Celery task
    dataset_ids = [str(id) for id in dataset_spec_ids] if dataset_spec_ids else None
    task = run_auto_ds_session.delay(str(session.id), initial_dataset_spec_ids=dataset_ids)

    # Update session with task ID
    session.celery_task_id = task.id
    db.commit()

    return {
        "session_id": str(session.id),
        "task_id": task.id,
        "status": session.status.value,
        "message": "Auto DS session started",
    }


@router.post("/{project_id}/auto-ds/stop")
def stop_auto_ds_session(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Stop the current Auto DS session for the project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to stop Auto DS on this project",
        )

    if not project.active_auto_ds_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active Auto DS session to stop",
        )

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == project.active_auto_ds_session_id
    ).first()

    if not session:
        # Clear the stale reference
        project.active_auto_ds_session_id = None
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Auto DS session not found",
        )

    # Mark session as stopped
    session.status = AutoDSSessionStatus.STOPPED
    session.completed_at = datetime.utcnow()
    project.active_auto_ds_session_id = None
    db.commit()

    return {
        "session_id": str(session.id),
        "status": session.status.value,
        "message": "Auto DS session stopped",
    }


@router.get("/{project_id}/auto-ds/status")
def get_auto_ds_status(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get the status of the current Auto DS session for the project."""
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

    if not project.active_auto_ds_session_id:
        return {
            "active": False,
            "message": "No active Auto DS session",
        }

    session = db.query(AutoDSSession).filter(
        AutoDSSession.id == project.active_auto_ds_session_id
    ).first()

    if not session:
        project.active_auto_ds_session_id = None
        db.commit()
        return {
            "active": False,
            "message": "No active Auto DS session",
        }

    return {
        "active": session.status in [
            AutoDSSessionStatus.PENDING,
            AutoDSSessionStatus.RUNNING,
            AutoDSSessionStatus.PAUSED,
        ],
        "session_id": str(session.id),
        "name": session.name,
        "status": session.status.value,
        "current_iteration": session.current_iteration,
        "max_iterations": session.max_iterations,
        "total_experiments_run": session.total_experiments_run,
        "best_score": session.best_score,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
    }


@router.get("/{project_id}/auto-ds/sessions")
def list_auto_ds_sessions(
    project_id: UUID,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List all Auto DS sessions for the project."""
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

    query = db.query(AutoDSSession).filter(
        AutoDSSession.project_id == project_id
    ).order_by(AutoDSSession.created_at.desc())

    total = query.count()
    sessions = query.offset(skip).limit(limit).all()

    return {
        "items": [
            {
                "id": str(s.id),
                "name": s.name,
                "status": s.status.value,
                "current_iteration": s.current_iteration,
                "max_iterations": s.max_iterations,
                "total_experiments_run": s.total_experiments_run,
                "best_score": s.best_score,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ],
        "total": total,
    }

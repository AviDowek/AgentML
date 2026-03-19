"""Admin API endpoints — platform analytics, user management, and logs."""
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, desc, text
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_admin
from app.models.user import User
from app.models.project import Project
from app.models.experiment import Experiment, ExperimentStatus
from app.models.agent_run import AgentRun, AgentRunStatus
from app.models.auto_ds_session import AutoDSSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class AdminUserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    is_admin: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    project_count: int = 0
    experiment_count: int = 0
    last_activity: Optional[datetime] = None

    class Config:
        from_attributes = True


class PlatformStatsResponse(BaseModel):
    total_users: int
    active_users_7d: int
    active_users_30d: int
    total_projects: int
    total_experiments: int
    experiments_by_status: dict
    total_agent_runs: int
    total_auto_ds_sessions: int
    new_users_7d: int
    new_users_30d: int
    experiments_last_7d: int
    experiments_last_30d: int


class ActivityLogEntry(BaseModel):
    id: str
    timestamp: datetime
    user_email: Optional[str]
    action: str
    details: Optional[str]
    status: Optional[str]
    project_name: Optional[str]
    experiment_name: Optional[str]


class DailyStatEntry(BaseModel):
    date: str
    count: int


# ── Platform Stats ───────────────────────────────────────────────────────────

@router.get("/stats", response_model=PlatformStatsResponse)
def get_platform_stats(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """Get platform-wide statistics."""
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    total_users = db.query(func.count(User.id)).scalar() or 0
    total_projects = db.query(func.count(Project.id)).scalar() or 0
    total_experiments = db.query(func.count(Experiment.id)).scalar() or 0
    total_agent_runs = db.query(func.count(AgentRun.id)).scalar() or 0
    total_auto_ds = db.query(func.count(AutoDSSession.id)).scalar() or 0

    # New users
    new_users_7d = db.query(func.count(User.id)).filter(User.created_at >= seven_days_ago).scalar() or 0
    new_users_30d = db.query(func.count(User.id)).filter(User.created_at >= thirty_days_ago).scalar() or 0

    # Active users (created experiments or agent runs recently)
    active_7d = db.query(func.count(func.distinct(Project.owner_id))).filter(
        Project.updated_at >= seven_days_ago
    ).scalar() or 0
    active_30d = db.query(func.count(func.distinct(Project.owner_id))).filter(
        Project.updated_at >= thirty_days_ago
    ).scalar() or 0

    # Experiments by status
    status_counts = db.query(
        Experiment.status, func.count(Experiment.id)
    ).group_by(Experiment.status).all()
    experiments_by_status = {s.value if hasattr(s, 'value') else str(s): c for s, c in status_counts}

    # Recent experiments
    experiments_7d = db.query(func.count(Experiment.id)).filter(
        Experiment.created_at >= seven_days_ago
    ).scalar() or 0
    experiments_30d = db.query(func.count(Experiment.id)).filter(
        Experiment.created_at >= thirty_days_ago
    ).scalar() or 0

    return PlatformStatsResponse(
        total_users=total_users,
        active_users_7d=active_7d,
        active_users_30d=active_30d,
        total_projects=total_projects,
        total_experiments=total_experiments,
        experiments_by_status=experiments_by_status,
        total_agent_runs=total_agent_runs,
        total_auto_ds_sessions=total_auto_ds,
        new_users_7d=new_users_7d,
        new_users_30d=new_users_30d,
        experiments_last_7d=experiments_7d,
        experiments_last_30d=experiments_30d,
    )


# ── Users ────────────────────────────────────────────────────────────────────

@router.get("/users", response_model=List[AdminUserResponse])
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    search: Optional[str] = Query(None),
):
    """List all users with analytics."""
    query = db.query(User)

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (User.email.ilike(search_term)) | (User.full_name.ilike(search_term))
        )

    users = query.order_by(desc(User.created_at)).offset(skip).limit(limit).all()

    result = []
    for u in users:
        project_count = db.query(func.count(Project.id)).filter(Project.owner_id == u.id).scalar() or 0
        experiment_count = db.query(func.count(Experiment.id)).join(Project).filter(
            Project.owner_id == u.id
        ).scalar() or 0

        # Last activity: most recent project or experiment update
        last_project = db.query(func.max(Project.updated_at)).filter(Project.owner_id == u.id).scalar()
        last_experiment = db.query(func.max(Experiment.updated_at)).join(Project).filter(
            Project.owner_id == u.id
        ).scalar()
        last_activity = max(filter(None, [last_project, last_experiment, u.updated_at]), default=None)

        result.append(AdminUserResponse(
            id=str(u.id),
            email=u.email,
            full_name=u.full_name,
            is_active=u.is_active,
            is_verified=u.is_verified,
            is_admin=u.is_admin,
            created_at=u.created_at,
            updated_at=u.updated_at,
            project_count=project_count,
            experiment_count=experiment_count,
            last_activity=last_activity,
        ))

    return result


@router.get("/users/count")
def get_user_count(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """Get total user count."""
    return {"count": db.query(func.count(User.id)).scalar() or 0}


# ── Activity Logs ────────────────────────────────────────────────────────────

@router.get("/logs", response_model=List[ActivityLogEntry])
def get_activity_logs(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    days: int = Query(7, ge=1, le=90),
):
    """Get platform activity logs (experiments, agent runs)."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    logs = []

    # Experiments as activity
    experiments = (
        db.query(Experiment, Project, User)
        .join(Project, Experiment.project_id == Project.id)
        .outerjoin(User, Project.owner_id == User.id)
        .filter(Experiment.created_at >= cutoff)
        .order_by(desc(Experiment.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )

    for exp, proj, user in experiments:
        status_str = exp.status.value if hasattr(exp.status, 'value') else str(exp.status)
        logs.append(ActivityLogEntry(
            id=str(exp.id),
            timestamp=exp.created_at,
            user_email=user.email if user else None,
            action=f"Experiment: {exp.name}",
            details=f"Metric: {exp.primary_metric or 'N/A'}" + (f" | Error: {exp.error_message[:100]}" if exp.error_message else ""),
            status=status_str,
            project_name=proj.name if proj else None,
            experiment_name=exp.name,
        ))

    # Sort by timestamp descending
    logs.sort(key=lambda x: x.timestamp, reverse=True)
    return logs[:limit]


# ── Agent Run Logs ───────────────────────────────────────────────────────────

@router.get("/agent-runs", response_model=List[ActivityLogEntry])
def get_agent_run_logs(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    days: int = Query(7, ge=1, le=90),
):
    """Get agent run logs."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    runs = (
        db.query(AgentRun, Project)
        .outerjoin(Project, AgentRun.project_id == Project.id)
        .filter(AgentRun.created_at >= cutoff)
        .order_by(desc(AgentRun.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )

    logs = []
    for run, proj in runs:
        status_str = run.status.value if hasattr(run.status, 'value') else str(run.status)
        logs.append(ActivityLogEntry(
            id=str(run.id),
            timestamp=run.created_at,
            user_email=None,
            action=f"Agent Run: {run.name or 'unnamed'}",
            details=run.error_message[:200] if run.error_message else None,
            status=status_str,
            project_name=proj.name if proj else None,
            experiment_name=None,
        ))

    return logs


# ── Daily Trends ─────────────────────────────────────────────────────────────

@router.get("/trends/users", response_model=List[DailyStatEntry])
def get_user_signup_trends(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    days: int = Query(30, ge=1, le=365),
):
    """Get daily user signup counts."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    results = (
        db.query(
            func.date(User.created_at).label("date"),
            func.count(User.id).label("count"),
        )
        .filter(User.created_at >= cutoff)
        .group_by(func.date(User.created_at))
        .order_by(func.date(User.created_at))
        .all()
    )
    return [DailyStatEntry(date=str(r.date), count=r.count) for r in results]


@router.get("/trends/experiments", response_model=List[DailyStatEntry])
def get_experiment_trends(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    days: int = Query(30, ge=1, le=365),
):
    """Get daily experiment counts."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    results = (
        db.query(
            func.date(Experiment.created_at).label("date"),
            func.count(Experiment.id).label("count"),
        )
        .filter(Experiment.created_at >= cutoff)
        .group_by(func.date(Experiment.created_at))
        .order_by(func.date(Experiment.created_at))
        .all()
    )
    return [DailyStatEntry(date=str(r.date), count=r.count) for r in results]

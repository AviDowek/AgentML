"""Shared API dependencies for project access control."""
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.project import Project
from app.models.user import User
from app.models.sharing import ProjectShare, InviteStatus, ShareRole


def check_project_access(
    db: Session, project: Project, user: Optional[User], require_write: bool = False
) -> bool:
    """Check if user has access to project."""
    if not user:
        return False

    if project.owner_id == user.id or project.owner_id is None:
        return True

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


def get_project_with_access(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Project:
    """Dependency that validates project exists and user has access.

    Use as: project: Project = Depends(get_project_with_access)
    """
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


def get_project_with_write_access(
    project_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Project:
    """Dependency that validates project exists and user has write access."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have write access to this project",
        )
    return project

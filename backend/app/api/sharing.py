"""Sharing API endpoints for projects."""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user_required, get_current_user, generate_invite_token
from app.models.user import User
from app.models.project import Project
from app.models.sharing import ProjectShare, ShareRole, InviteStatus
from app.schemas.sharing import (
    ShareRequest,
    ShareResponse,
    ShareListResponse,
    AcceptInviteRequest,
    AcceptInviteResponse,
    UpdateShareRequest,
)
from app.services.email_service import send_project_invite_email

router = APIRouter(prefix="/api/v1/sharing", tags=["Sharing"])


# ==================== Project Sharing ====================


@router.get("/projects/{project_id}/shares", response_model=ShareListResponse)
def list_project_shares(
    project_id: UUID,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """List all shares for a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check ownership or admin share (legacy projects without owner are accessible)
    is_owner = project.owner_id == user.id or project.owner_id is None
    if not is_owner:
        share = db.query(ProjectShare).filter(
            ProjectShare.project_id == project_id,
            ProjectShare.user_id == user.id,
            ProjectShare.role == ShareRole.ADMIN,
            ProjectShare.status == InviteStatus.ACCEPTED,
        ).first()
        if not share:
            raise HTTPException(status_code=403, detail="Not authorized to view shares")

    shares = db.query(ProjectShare).filter(ProjectShare.project_id == project_id).all()

    items = []
    for share in shares:
        item = ShareResponse(
            id=share.id,
            user_id=share.user_id,
            invited_email=share.invited_email,
            role=share.role,
            status=share.status,
            created_at=share.created_at,
            user_name=share.user.full_name if share.user else None,
        )
        items.append(item)

    return ShareListResponse(items=items, total=len(items))


@router.post("/projects/{project_id}/shares", response_model=ShareResponse)
async def share_project(
    project_id: UUID,
    data: ShareRequest,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Share a project with another user by email."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check ownership or admin share (legacy projects without owner are accessible)
    is_owner = project.owner_id == user.id or project.owner_id is None
    if not is_owner:
        share = db.query(ProjectShare).filter(
            ProjectShare.project_id == project_id,
            ProjectShare.user_id == user.id,
            ProjectShare.role == ShareRole.ADMIN,
            ProjectShare.status == InviteStatus.ACCEPTED,
        ).first()
        if not share:
            raise HTTPException(status_code=403, detail="Not authorized to share this project")

    # Can't share with yourself
    if data.email == user.email:
        raise HTTPException(status_code=400, detail="Cannot share with yourself")

    # Check if already shared with this email
    existing_share = db.query(ProjectShare).filter(
        ProjectShare.project_id == project_id,
        (ProjectShare.invited_email == data.email) |
        (ProjectShare.user_id.in_(
            db.query(User.id).filter(User.email == data.email)
        ))
    ).first()

    if existing_share:
        raise HTTPException(status_code=400, detail="Already shared with this user")

    # Check if user exists
    target_user = db.query(User).filter(User.email == data.email).first()

    # Create share
    invite_token = generate_invite_token()
    new_share = ProjectShare(
        project_id=project_id,
        user_id=target_user.id if target_user else None,
        invited_email=data.email if not target_user else None,
        role=data.role,
        status=InviteStatus.PENDING,
        invite_token=invite_token,
    )
    db.add(new_share)
    db.commit()
    db.refresh(new_share)

    # Send invitation email
    await send_project_invite_email(
        to_email=data.email,
        inviter_name=user.full_name or user.email,
        project_name=project.name,
        invite_token=invite_token,
        role=data.role.value,
    )

    return ShareResponse(
        id=new_share.id,
        user_id=new_share.user_id,
        invited_email=new_share.invited_email,
        role=new_share.role,
        status=new_share.status,
        created_at=new_share.created_at,
        user_name=target_user.full_name if target_user else None,
    )


@router.put("/projects/{project_id}/shares/{share_id}", response_model=ShareResponse)
def update_project_share(
    project_id: UUID,
    share_id: UUID,
    data: UpdateShareRequest,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Update a project share's role."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check ownership (legacy projects without owner are accessible)
    is_owner = project.owner_id == user.id or project.owner_id is None
    if not is_owner:
        raise HTTPException(status_code=403, detail="Only the owner can modify shares")

    share = db.query(ProjectShare).filter(
        ProjectShare.id == share_id,
        ProjectShare.project_id == project_id,
    ).first()

    if not share:
        raise HTTPException(status_code=404, detail="Share not found")

    share.role = data.role
    db.commit()
    db.refresh(share)

    return ShareResponse(
        id=share.id,
        user_id=share.user_id,
        invited_email=share.invited_email,
        role=share.role,
        status=share.status,
        created_at=share.created_at,
        user_name=share.user.full_name if share.user else None,
    )


@router.delete("/projects/{project_id}/shares/{share_id}")
def remove_project_share(
    project_id: UUID,
    share_id: UUID,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Remove a share from a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check ownership (legacy projects without owner are accessible)
    is_owner = project.owner_id == user.id or project.owner_id is None
    if not is_owner:
        raise HTTPException(status_code=403, detail="Only the owner can remove shares")

    share = db.query(ProjectShare).filter(
        ProjectShare.id == share_id,
        ProjectShare.project_id == project_id,
    ).first()

    if not share:
        raise HTTPException(status_code=404, detail="Share not found")

    db.delete(share)
    db.commit()

    return {"message": "Share removed successfully"}


# ==================== Accept Invitations ====================


@router.post("/accept-invite", response_model=AcceptInviteResponse)
def accept_invite(
    data: AcceptInviteRequest,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Accept a sharing invitation."""
    # Check project shares
    project_share = db.query(ProjectShare).filter(
        ProjectShare.invite_token == data.token,
        ProjectShare.status == InviteStatus.PENDING,
    ).first()

    if project_share:
        # Verify the email matches (if it was an email invite to non-user)
        if project_share.invited_email and project_share.invited_email != user.email:
            raise HTTPException(
                status_code=403,
                detail="This invitation was sent to a different email address",
            )

        project_share.user_id = user.id
        project_share.invited_email = None
        project_share.status = InviteStatus.ACCEPTED
        project_share.invite_token = None  # Clear token after use
        db.commit()

        return AcceptInviteResponse(
            message="Invitation accepted successfully",
            resource_type="project",
            resource_id=project_share.project_id,
        )

    raise HTTPException(
        status_code=404,
        detail="Invalid or expired invitation token",
    )


@router.get("/my-shares")
def get_my_shares(
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Get all resources shared with the current user."""
    # Get shared projects
    project_shares = db.query(ProjectShare).filter(
        ProjectShare.user_id == user.id,
        ProjectShare.status == InviteStatus.ACCEPTED,
    ).all()

    shared_projects = []
    for share in project_shares:
        shared_projects.append({
            "id": str(share.project.id),
            "name": share.project.name,
            "role": share.role.value,
            "owner": share.project.owner.email if share.project.owner else None,
            "shared_at": share.created_at.isoformat(),
        })

    # Get pending invitations
    pending_project_invites = db.query(ProjectShare).filter(
        ProjectShare.user_id == user.id,
        ProjectShare.status == InviteStatus.PENDING,
    ).all()

    pending_invites = []
    for share in pending_project_invites:
        pending_invites.append({
            "type": "project",
            "id": str(share.project.id),
            "name": share.project.name,
            "role": share.role.value,
            "token": share.invite_token,
        })

    return {
        "shared_projects": shared_projects,
        "pending_invites": pending_invites,
    }

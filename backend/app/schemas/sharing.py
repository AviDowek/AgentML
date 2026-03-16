"""Sharing schemas for projects."""
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field

from app.models.sharing import ShareRole, InviteStatus


class ShareRequest(BaseModel):
    """Request to share a resource."""
    email: EmailStr = Field(..., description="Email of user to share with")
    role: ShareRole = Field(default=ShareRole.VIEWER, description="Role for the shared user")


class ShareResponse(BaseModel):
    """Response for a share entry."""
    id: UUID
    user_id: Optional[UUID] = None
    invited_email: Optional[str] = None
    role: ShareRole
    status: InviteStatus
    created_at: datetime
    user_name: Optional[str] = None  # Filled in from user relationship

    class Config:
        from_attributes = True


class ProjectShareResponse(ShareResponse):
    """Response for project share."""
    project_id: UUID


class ShareListResponse(BaseModel):
    """List of shares for a resource."""
    items: List[ShareResponse]
    total: int


class AcceptInviteRequest(BaseModel):
    """Request to accept a sharing invitation."""
    token: str = Field(..., description="Invitation token from email")


class AcceptInviteResponse(BaseModel):
    """Response after accepting invitation."""
    message: str
    resource_type: str  # 'project'
    resource_id: UUID


class UpdateShareRequest(BaseModel):
    """Request to update a share role."""
    role: ShareRole

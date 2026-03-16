"""Sharing models for projects."""
import uuid
import enum
from sqlalchemy import Column, String, ForeignKey, Enum as SQLEnum, UniqueConstraint
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.models.base import TimestampMixin, GUID


class ShareRole(str, enum.Enum):
    """Role for shared access."""
    VIEWER = "viewer"  # Can view but not modify
    EDITOR = "editor"  # Can modify
    ADMIN = "admin"    # Can modify and manage sharing


class InviteStatus(str, enum.Enum):
    """Status of a sharing invitation."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"


class ProjectShare(Base, TimestampMixin):
    """Share a project with another user."""

    __tablename__ = "project_shares"
    __table_args__ = (
        UniqueConstraint('project_id', 'user_id', name='uq_project_user'),
    )

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)  # Nullable for pending invites
    invited_email = Column(String(255), nullable=True)  # For invites to non-registered users
    role = Column(
        SQLEnum(ShareRole, values_callable=lambda x: [e.value for e in x]),
        default=ShareRole.VIEWER,
        nullable=False
    )
    status = Column(
        SQLEnum(InviteStatus, values_callable=lambda x: [e.value for e in x]),
        default=InviteStatus.PENDING,
        nullable=False
    )
    invite_token = Column(String(255), unique=True, nullable=True)  # Token for accepting invite

    # Relationships
    project = relationship("Project", back_populates="shares")
    user = relationship("User", back_populates="project_shares", foreign_keys=[user_id])

    def __repr__(self):
        return f"<ProjectShare project={self.project_id} user={self.user_id or self.invited_email}>"

"""User model for authentication."""
import uuid
import enum
from sqlalchemy import Column, String, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.models.base import TimestampMixin, GUID


class User(Base, TimestampMixin):
    """User account for authentication."""

    __tablename__ = "users"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)  # Nullable for OAuth-only users
    full_name = Column(String(255), nullable=True)

    # OAuth
    google_id = Column(String(255), unique=True, nullable=True, index=True)

    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False, server_default="false")

    # Relationships
    owned_projects = relationship("Project", back_populates="owner", foreign_keys="Project.owner_id")
    project_shares = relationship("ProjectShare", back_populates="user", foreign_keys="ProjectShare.user_id")
    conversations = relationship("Conversation", back_populates="owner")

    def __repr__(self):
        return f"<User {self.email}>"

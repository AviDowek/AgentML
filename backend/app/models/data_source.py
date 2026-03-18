"""Data source model."""
from sqlalchemy import Column, String, ForeignKey, Enum as SQLEnum, LargeBinary
from sqlalchemy.orm import relationship
import uuid
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class DataSourceType(str, enum.Enum):
    """Data source types."""
    FILE_UPLOAD = "file_upload"
    DATABASE = "database"
    S3 = "s3"
    API = "api"
    EXTERNAL_DATASET = "external_dataset"  # Discovered public dataset


class DataSource(Base, TimestampMixin):
    """Data source model - represents a data connection."""

    __tablename__ = "data_sources"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    type = Column(
        SQLEnum(DataSourceType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    config_json = Column(JSONType(), nullable=True)  # Connection config (credentials, paths, etc.)
    schema_summary = Column(JSONType(), nullable=True)  # Analyzed schema info
    profile_json = Column(JSONType(), nullable=True)  # Detailed data profile for AI agents
    file_data = Column(LargeBinary, nullable=True)  # Raw file bytes for persistent storage

    # Relationships
    project = relationship("Project", back_populates="data_sources")
    holdout_sets = relationship("HoldoutSet", back_populates="data_source", cascade="all, delete-orphan")

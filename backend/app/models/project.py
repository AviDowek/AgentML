"""Project model."""
from sqlalchemy import Column, String, Text, Enum as SQLEnum, ForeignKey, Integer, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
import uuid
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, GUID


# Default limits for large dataset safeguards
DEFAULT_MAX_TRAINING_ROWS = 1_000_000
DEFAULT_PROFILING_SAMPLE_ROWS = 50_000
DEFAULT_MAX_AGGREGATION_WINDOW_DAYS = 365


class TaskType(str, enum.Enum):
    """ML task types supported by AutoGluon."""
    # Tabular tasks
    BINARY = "binary"  # Binary classification
    MULTICLASS = "multiclass"  # Multi-class classification
    REGRESSION = "regression"  # Standard regression
    QUANTILE = "quantile"  # Quantile regression (predict percentiles)

    # Time series tasks
    TIMESERIES_FORECAST = "timeseries_forecast"  # Time series forecasting

    # Multimodal tasks
    MULTIMODAL_CLASSIFICATION = "multimodal_classification"  # Text + tabular + images
    MULTIMODAL_REGRESSION = "multimodal_regression"  # Text + tabular + images

    # Legacy aliases (for backward compatibility)
    CLASSIFICATION = "classification"  # Maps to binary


class ProjectStatus(str, enum.Enum):
    """Project status."""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class Project(Base, TimestampMixin):
    """Project model - represents an ML project."""

    __tablename__ = "projects"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(
        SQLEnum(TaskType, values_callable=lambda x: [e.value for e in x]),
        nullable=True
    )
    status = Column(
        SQLEnum(ProjectStatus, values_callable=lambda x: [e.value for e in x]),
        default=ProjectStatus.DRAFT,
        nullable=False
    )

    # Large dataset safeguard settings
    max_training_rows = Column(
        Integer,
        default=DEFAULT_MAX_TRAINING_ROWS,
        nullable=False,
        doc="Maximum rows in materialized training dataset (default 1M)"
    )
    profiling_sample_rows = Column(
        Integer,
        default=DEFAULT_PROFILING_SAMPLE_ROWS,
        nullable=False,
        doc="Sample size for data profiling (default 50K)"
    )
    max_aggregation_window_days = Column(
        Integer,
        default=DEFAULT_MAX_AGGREGATION_WINDOW_DAYS,
        nullable=False,
        doc="Maximum days for aggregation window in joins (default 365)"
    )

    # Owner relationship
    owner_id = Column(GUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    # Auto DS settings
    auto_ds_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Enable automatic Auto DS sessions after pipeline completion"
    )
    auto_ds_config_json = Column(
        JSONB,
        nullable=True,
        doc="Auto DS configuration: max_iterations, accuracy_threshold, time_budget_minutes, etc."
    )
    active_auto_ds_session_id = Column(
        GUID(),
        ForeignKey("auto_ds_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Currently active Auto DS session for this project"
    )

    # Context document settings
    context_ab_testing_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Enable A/B testing: create half experiments WITH context, half WITHOUT"
    )

    # Relationships
    owner = relationship("User", back_populates="owned_projects", foreign_keys=[owner_id])
    shares = relationship("ProjectShare", back_populates="project", cascade="all, delete-orphan")
    data_sources = relationship("DataSource", back_populates="project", cascade="all, delete-orphan")
    dataset_specs = relationship("DatasetSpec", back_populates="project", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="project", cascade="all, delete-orphan")
    model_versions = relationship("ModelVersion", back_populates="project", cascade="all, delete-orphan")
    retraining_policies = relationship("RetrainingPolicy", back_populates="project", cascade="all, delete-orphan")
    agent_runs = relationship("AgentRun", back_populates="project", cascade="all, delete-orphan")
    visualizations = relationship("Visualization", back_populates="project", cascade="all, delete-orphan")
    research_cycles = relationship("ResearchCycle", back_populates="project", cascade="all, delete-orphan", order_by="ResearchCycle.sequence_number")
    lab_notebook_entries = relationship("LabNotebookEntry", back_populates="project", cascade="all, delete-orphan", order_by="LabNotebookEntry.created_at.desc()")
    holdout_sets = relationship("HoldoutSet", back_populates="project", cascade="all, delete-orphan")

    # Auto DS Team relationships
    auto_ds_sessions = relationship(
        "AutoDSSession",
        back_populates="project",
        cascade="all, delete-orphan",
        foreign_keys="AutoDSSession.project_id"
    )
    active_auto_ds_session = relationship(
        "AutoDSSession",
        foreign_keys=[active_auto_ds_session_id],
        post_update=True
    )
    research_insights = relationship("ResearchInsight", back_populates="project", cascade="all, delete-orphan")
    context_documents = relationship("ContextDocument", back_populates="project", cascade="all, delete-orphan")

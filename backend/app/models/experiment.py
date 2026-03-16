"""Experiment and Trial models."""
from sqlalchemy import Column, String, Text, ForeignKey, Enum as SQLEnum, Integer
from sqlalchemy.orm import relationship
import uuid
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class ExperimentStatus(str, enum.Enum):
    """Experiment status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricDirection(str, enum.Enum):
    """Metric optimization direction."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Experiment(Base, TimestampMixin):
    """Experiment model - represents an ML experiment."""

    __tablename__ = "experiments"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    dataset_spec_id = Column(GUID(), ForeignKey("dataset_specs.id", ondelete="SET NULL"), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(
        SQLEnum(ExperimentStatus, values_callable=lambda x: [e.value for e in x]),
        default=ExperimentStatus.PENDING,
        nullable=False
    )

    # Metrics configuration
    primary_metric = Column(String(100), nullable=True)  # e.g., "accuracy", "rmse"
    metric_direction = Column(
        SQLEnum(MetricDirection, values_callable=lambda x: [e.value for e in x]),
        nullable=True
    )

    # Full experiment plan (designed by LLM or user)
    experiment_plan_json = Column(JSONType(), nullable=True)

    # Plan versioning
    plan_version = Column(Integer, default=1, nullable=False)
    plan_history_json = Column(JSONType(), nullable=True)  # List of {version, plan, changed_at, changed_by}

    # Celery task tracking
    celery_task_id = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)

    # Auto-improve iteration tracking
    iteration_number = Column(Integer, default=1, nullable=False)  # Which iteration this is (1, 2, 3...)
    parent_experiment_id = Column(GUID(), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True)
    improvement_context_json = Column(JSONType(), nullable=True)  # What improvements were applied

    # Training results (populated after experiment completes)
    # Stores metrics, scores, model info from training
    results_json = Column(JSONType(), nullable=True)

    # Auto-iterate settings (automatically run AI feedback and iterate after each experiment)
    auto_iterate_enabled = Column(Integer, default=0, nullable=False)  # 0=disabled, 1=enabled (SQLite friendly)
    auto_iterate_max = Column(Integer, default=5, nullable=False)  # Max iterations before stopping (default 5)

    # Relationships
    project = relationship("Project", back_populates="experiments")
    dataset_spec = relationship("DatasetSpec", back_populates="experiments")
    trials = relationship("Trial", back_populates="experiment", cascade="all, delete-orphan")
    model_versions = relationship("ModelVersion", back_populates="experiment")
    agent_runs = relationship("AgentRun", back_populates="experiment")
    cycle_experiments = relationship("CycleExperiment", back_populates="experiment", cascade="all, delete-orphan")

    # Auto DS iteration experiments link
    auto_ds_iteration_experiments = relationship(
        "AutoDSIterationExperiment",
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    # Self-referential relationships for improvement iterations
    parent_experiment = relationship(
        "Experiment",
        remote_side=[id],
        foreign_keys=[parent_experiment_id],
        backref="child_experiments"
    )


class TrialStatus(str, enum.Enum):
    """Trial status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Trial(Base, TimestampMixin):
    """Trial model - represents a single experiment variant/run."""

    __tablename__ = "trials"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(GUID(), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    variant_name = Column(String(255), nullable=False)

    # Trial configuration
    data_split_strategy = Column(String(100), nullable=True)  # e.g., "random_80_20", "time_based"
    automl_config = Column(JSONType(), nullable=True)  # AutoML library configuration

    # Status and results
    status = Column(
        SQLEnum(TrialStatus, values_callable=lambda x: [e.value for e in x]),
        default=TrialStatus.PENDING,
        nullable=False
    )
    metrics_json = Column(JSONType(), nullable=True)  # Final metrics
    best_model_ref = Column(String(500), nullable=True)  # Reference to best model artifact

    # Logging
    logs_location = Column(String(500), nullable=True)
    training_logs = Column(Text, nullable=True)  # Captured training output for AI analysis
    leaderboard_json = Column(JSONType(), nullable=True)  # Model leaderboard from AutoGluon

    # AI critique feedback
    critique_json = Column(JSONType(), nullable=True)  # AI-generated feedback on training results

    # Baseline and sanity check metrics
    # Stores baseline model metrics and label-shuffle test results
    # Example: {
    #   "majority_class": {"accuracy": 0.52, "roc_auc": 0.5},
    #   "simple_logistic": {"accuracy": 0.56, "roc_auc": 0.58},
    #   "label_shuffle": {"accuracy": 0.52, "roc_auc": 0.51}
    # }
    baseline_metrics_json = Column(JSONType(), nullable=True)

    # Relationships
    experiment = relationship("Experiment", back_populates="trials")
    model_versions = relationship("ModelVersion", back_populates="trial")

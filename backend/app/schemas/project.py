"""Project schemas."""
from datetime import datetime
from typing import Optional, Any, Dict
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.project import (
    TaskType,
    ProjectStatus,
    DEFAULT_MAX_TRAINING_ROWS,
    DEFAULT_PROFILING_SAMPLE_ROWS,
    DEFAULT_MAX_AGGREGATION_WINDOW_DAYS,
)


class AutoDSConfig(BaseModel):
    """Configuration for Auto DS autonomous sessions."""

    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of research iterations"
    )
    accuracy_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Stop when accuracy reaches this threshold"
    )
    time_budget_minutes: Optional[int] = Field(
        default=None,
        ge=10,
        le=1440,
        description="Maximum time budget in minutes"
    )
    parallel_experiments: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of experiments to run in parallel"
    )
    start_on_pipeline_complete: bool = Field(
        default=True,
        description="Automatically start Auto DS when pipeline completes"
    )


class ProjectBase(BaseModel):
    """Base project schema."""

    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    task_type: Optional[TaskType] = Field(
        None,
        description="""ML task type. Determines which AutoML runner is used:

**Tabular (autogluon.tabular):**
- `binary` - Binary classification (2 classes)
- `multiclass` - Multi-class classification (3+ classes)
- `regression` - Predict continuous values
- `quantile` - Predict value percentiles (e.g., 10th, 50th, 90th)

**Time Series (autogluon.timeseries):**
- `timeseries_forecast` - Forecast future values. Requires `prediction_length` and `time_column` in experiment config.

**Multimodal (autogluon.multimodal):**
- `multimodal_classification` - Classify using text + tabular + images
- `multimodal_regression` - Predict values using text + tabular + images

**Legacy:**
- `classification` - Alias for `binary`
"""
    )


class ProjectCreate(ProjectBase):
    """Schema for creating a project.

    The task_type determines which AutoML runner will be used for experiments.
    """

    pass


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    task_type: Optional[TaskType] = None
    status: Optional[ProjectStatus] = None
    # Large dataset safeguard settings
    max_training_rows: Optional[int] = Field(
        None,
        ge=1000,
        le=100_000_000,
        description="Maximum rows in materialized training dataset"
    )
    profiling_sample_rows: Optional[int] = Field(
        None,
        ge=100,
        le=1_000_000,
        description="Sample size for data profiling"
    )
    max_aggregation_window_days: Optional[int] = Field(
        None,
        ge=1,
        le=3650,
        description="Maximum days for aggregation window in joins"
    )
    # Auto DS settings
    auto_ds_enabled: Optional[bool] = Field(
        None,
        description="Enable automatic Auto DS sessions after pipeline completion"
    )
    auto_ds_config_json: Optional[Dict[str, Any]] = Field(
        None,
        description="Auto DS configuration settings"
    )


class ProjectResponse(ProjectBase):
    """Schema for project response."""

    id: UUID
    status: ProjectStatus
    # Large dataset safeguard settings
    max_training_rows: int = Field(
        default=DEFAULT_MAX_TRAINING_ROWS,
        description="Maximum rows in materialized training dataset"
    )
    profiling_sample_rows: int = Field(
        default=DEFAULT_PROFILING_SAMPLE_ROWS,
        description="Sample size for data profiling"
    )
    max_aggregation_window_days: int = Field(
        default=DEFAULT_MAX_AGGREGATION_WINDOW_DAYS,
        description="Maximum days for aggregation window in joins"
    )
    # Auto DS settings
    auto_ds_enabled: bool = Field(
        default=False,
        description="Enable automatic Auto DS sessions after pipeline completion"
    )
    auto_ds_config_json: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Auto DS configuration settings"
    )
    active_auto_ds_session_id: Optional[UUID] = Field(
        default=None,
        description="ID of the currently active Auto DS session"
    )
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    """Schema for list of projects response."""

    items: list[ProjectResponse]
    total: int

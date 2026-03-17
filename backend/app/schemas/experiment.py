"""Experiment and trial schemas."""
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.experiment import ExperimentStatus, MetricDirection, TrialStatus


class TrainingBackend(str, Enum):
    """Training backend options."""
    MODAL = "modal"  # Run on Modal.com cloud


class TrainingOptions(BaseModel):
    """Options for how to run training."""
    backend: TrainingBackend = Field(
        default=TrainingBackend.MODAL,
        description="Training runs on Modal.com cloud"
    )


class ExperimentBase(BaseModel):
    """Base experiment schema."""

    name: str = Field(..., min_length=1, max_length=255, description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    dataset_spec_id: Optional[UUID] = Field(
        None,
        description="ID of the DatasetSpec to use. Required to run the experiment."
    )
    primary_metric: Optional[str] = Field(
        None,
        description="""Metric to optimize. If not set, defaults based on task type:
- binary: roc_auc
- multiclass: accuracy
- regression: rmse
- quantile: pinball_loss
- timeseries_forecast: MASE
- multimodal_*: accuracy (classification) or rmse (regression)"""
    )
    metric_direction: Optional[MetricDirection] = Field(
        None,
        description="Whether to maximize or minimize the metric"
    )
    experiment_plan_json: Optional[dict[str, Any]] = Field(
        None,
        description="""Configuration for the experiment. Structure:
```json
{
  "automl_config": {
    // Common options (all task types):
    "time_limit": 300,          // Training time in seconds
    "presets": "medium_quality", // Quality preset

    // Tabular options:
    "num_bag_folds": 5,
    "num_stack_levels": 0,
    "excluded_model_types": [],

    // Quantile options:
    "quantile_levels": [0.1, 0.5, 0.9],

    // Time series options (REQUIRED for timeseries_forecast):
    "prediction_length": 7,     // Steps to forecast
    "time_column": "date",      // Timestamp column name
    "id_column": "series_id",   // Optional: for multiple series
    "freq": "D"                 // Optional: D=daily, H=hourly, W=weekly
  }
}
```"""
    )


class ExperimentCreate(ExperimentBase):
    """Schema for creating an experiment.

    Set dataset_spec_id to link to your data, then use POST /experiments/{id}/run to execute.
    """

    pass


class ExperimentUpdate(BaseModel):
    """Schema for updating an experiment."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    dataset_spec_id: Optional[UUID] = None
    primary_metric: Optional[str] = None
    metric_direction: Optional[MetricDirection] = None
    experiment_plan_json: Optional[dict[str, Any]] = None
    status: Optional[ExperimentStatus] = None


class ExperimentResponse(ExperimentBase):
    """Schema for experiment response."""

    id: UUID
    project_id: UUID
    status: ExperimentStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExperimentDetailResponse(ExperimentResponse):
    """Schema for detailed experiment response with summary info."""

    trial_count: int = 0
    best_model: Optional[dict[str, Any]] = None
    best_metrics: Optional[dict[str, Any]] = None
    # Auto-improve iteration fields
    iteration_number: int = 1
    parent_experiment_id: Optional[UUID] = None
    improvement_context_json: Optional[dict[str, Any]] = None

    # Auto-iterate settings
    auto_iterate_enabled: bool = Field(
        False,
        description="When enabled, automatically runs AI feedback and creates next iteration after training completes"
    )
    auto_iterate_max: int = Field(
        5,
        description="Maximum number of iterations before auto-iterate stops (default: 5)"
    )

    # Holdout-based scoring (Prompt: Make Holdout Score the Real Score)
    # final_score = holdout score (authoritative), val_score = CV/validation score
    final_score: Optional[float] = Field(
        None,
        description="Final authoritative score from holdout evaluation (use this for comparisons)"
    )
    val_score: Optional[float] = Field(
        None,
        description="Validation/CV score from training (NOT the final score)"
    )
    train_score: Optional[float] = Field(
        None,
        description="Training score (for overfitting analysis)"
    )
    has_holdout: bool = Field(
        False,
        description="Whether holdout evaluation was performed"
    )
    holdout_samples: Optional[int] = Field(
        None,
        description="Number of samples in holdout set"
    )
    overfitting_gap: Optional[float] = Field(
        None,
        description="Gap between validation and holdout scores (positive = overfitting)"
    )
    score_source: str = Field(
        "validation",
        description="Source of final_score: 'holdout' (preferred) or 'validation' (fallback)"
    )


class ExperimentRunRequest(BaseModel):
    """Schema for running an experiment with options."""
    training_options: Optional[TrainingOptions] = Field(
        default=None,
        description="Training configuration options. If None, uses defaults."
    )


class ExperimentRunResponse(BaseModel):
    """Schema for experiment run/cancel response."""

    experiment_id: UUID
    status: str
    task_id: Optional[str] = None
    message: Optional[str] = None
    backend: Optional[str] = None  # 'local' or 'modal'


class ExperimentProgressResponse(BaseModel):
    """Schema for experiment progress response."""

    experiment_id: UUID
    status: str  # pending, queued, running, completed, failed, cancelled
    progress: int = 0  # 0-100 percentage
    message: Optional[str] = None
    stage: Optional[str] = None  # Current stage (e.g., "loading_data", "training")


class AutoIterateSettingsRequest(BaseModel):
    """Schema for updating auto-iterate settings."""
    enabled: bool = Field(
        ...,
        description="Enable or disable auto-iterate mode"
    )
    max_iterations: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of iterations before auto-iterate stops (1-20, default: 5)"
    )


class AutoIterateSettingsResponse(BaseModel):
    """Schema for auto-iterate settings response."""
    experiment_id: UUID
    auto_iterate_enabled: bool
    auto_iterate_max: int
    current_iteration: int
    can_continue: bool = Field(
        ...,
        description="Whether more iterations are allowed (current_iteration < max)"
    )
    message: Optional[str] = None


class TrialBase(BaseModel):
    """Base trial schema."""

    variant_name: str = Field(..., min_length=1, max_length=255)
    data_split_strategy: Optional[str] = None
    automl_config: Optional[dict[str, Any]] = None


class TrialCreate(TrialBase):
    """Schema for creating a trial."""

    pass


class TrialResponse(TrialBase):
    """Schema for trial response."""

    id: UUID
    experiment_id: UUID
    status: TrialStatus
    metrics_json: Optional[dict[str, Any]] = None
    baseline_metrics_json: Optional[dict[str, Any]] = None
    best_model_ref: Optional[str] = None
    logs_location: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

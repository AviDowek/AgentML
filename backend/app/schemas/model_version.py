"""Model version schemas."""
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.model_version import ModelStatus


class ModelVersionBase(BaseModel):
    """Base model version schema."""

    name: str = Field(..., min_length=1, max_length=255)
    model_type: Optional[str] = None
    artifact_location: Optional[str] = None
    metrics_json: Optional[dict[str, Any]] = None
    feature_importances_json: Optional[dict[str, Any]] = None
    serving_config_json: Optional[dict[str, Any]] = None


class ModelVersionCreate(ModelVersionBase):
    """Schema for creating a model version."""

    experiment_id: Optional[UUID] = None
    trial_id: Optional[UUID] = None


class ModelVersionResponse(ModelVersionBase):
    """Schema for model version response."""

    id: UUID
    project_id: UUID
    experiment_id: Optional[UUID] = None
    trial_id: Optional[UUID] = None
    status: ModelStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ModelPromoteRequest(BaseModel):
    """Schema for promoting a model."""

    status: ModelStatus = Field(..., description="New status: candidate, shadow, or production")
    override_reason: Optional[str] = Field(
        None,
        min_length=10,
        max_length=2000,
        description=(
            "Required when promoting a high-risk model. Explain why you believe "
            "this model is safe to promote despite identified risks (e.g., overfitting, "
            "data leakage). Your reason will be logged in the lab notebook."
        ),
    )


class PredictionRequest(BaseModel):
    """Schema for prediction request."""

    features: dict[str, Any] = Field(..., description="Feature values for prediction")


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    prediction: Any = Field(..., description="Model prediction")
    probabilities: Optional[dict[str, float]] = Field(None, description="Class probabilities (for classification)")
    model_id: UUID
    model_name: str


class ModelExplainRequest(BaseModel):
    """Schema for model explanation request."""

    question: str = Field(..., min_length=1, max_length=2000, description="Question about the model")


class ModelExplainResponse(BaseModel):
    """Schema for model explanation response."""

    answer: str = Field(..., description="Answer to the question (markdown formatted)")
    model_id: UUID


# Validation Sample Schemas

class ValidationSampleResponse(BaseModel):
    """Schema for a single validation sample."""

    id: UUID
    model_version_id: UUID
    row_index: int
    features: dict[str, Any] = Field(..., description="Feature values for this sample")
    target_value: str = Field(..., description="Actual target value")
    predicted_value: str = Field(..., description="Model's predicted value")
    error_value: Optional[float] = Field(None, description="Signed error (prediction - target)")
    absolute_error: Optional[float] = Field(None, description="Absolute error")
    prediction_probabilities: Optional[dict[str, float]] = Field(
        None, description="Class probabilities (classification only)"
    )

    class Config:
        from_attributes = True

    @classmethod
    def from_db_model(cls, sample) -> "ValidationSampleResponse":
        """Convert from SQLAlchemy model to Pydantic schema."""
        return cls(
            id=sample.id,
            model_version_id=sample.model_version_id,
            row_index=sample.row_index,
            features=sample.features_json,
            target_value=sample.target_value,
            predicted_value=sample.predicted_value,
            error_value=sample.error_value,
            absolute_error=sample.absolute_error,
            prediction_probabilities=sample.prediction_probabilities_json,
        )


class ValidationSamplesListResponse(BaseModel):
    """Schema for paginated validation samples list."""

    model_id: UUID
    total: int = Field(..., description="Total number of validation samples")
    limit: int = Field(..., description="Number of samples per page")
    offset: int = Field(..., description="Current offset")
    samples: list[ValidationSampleResponse] = Field(..., description="List of validation samples")


class WhatIfRequest(BaseModel):
    """Schema for what-if prediction request."""

    sample_id: UUID = Field(..., description="ID of the validation sample to modify")
    modified_features: dict[str, Any] = Field(
        ..., description="Features to override (partial dict, only changed features)"
    )


class WhatIfResponse(BaseModel):
    """Schema for what-if prediction response."""

    original_sample: ValidationSampleResponse
    modified_features: dict[str, Any] = Field(..., description="Features that were modified")
    original_prediction: Any = Field(..., description="Original model prediction")
    modified_prediction: Any = Field(..., description="Prediction with modified features")
    prediction_delta: Optional[float] = Field(
        None, description="Change in prediction (for regression/numeric)"
    )
    original_probabilities: Optional[dict[str, float]] = Field(
        None, description="Original class probabilities"
    )
    modified_probabilities: Optional[dict[str, float]] = Field(
        None, description="Modified class probabilities"
    )


# Model Testing Data Schemas

class FeatureStatistics(BaseModel):
    """Statistics for a single feature."""

    name: str
    type: str  # "numeric", "categorical", "boolean"
    importance: Optional[float] = None  # Feature importance score
    # Numeric stats
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    # Categorical stats
    categories: Optional[list[str]] = None
    most_common: Optional[str] = None


class ModelTestingDataResponse(BaseModel):
    """Response with feature statistics and sample data for model testing."""

    model_id: UUID
    model_name: str
    task_type: str
    target_column: Optional[str] = None

    # Feature info with stats
    features: list[FeatureStatistics] = Field(
        ..., description="Feature statistics including importance, min/max, categories"
    )

    # Top important features (sorted by importance)
    top_features: list[str] = Field(
        ..., description="Names of top features by importance"
    )

    # A random sample to start with
    sample_data: Optional[dict[str, Any]] = Field(
        None, description="A sample row from validation data to use as starting point"
    )

    # Whether we have validation samples available
    has_validation_samples: bool = False
    validation_sample_count: int = 0


# Raw Prediction Schemas

class RawPredictionRequest(BaseModel):
    """Schema for raw data prediction request.

    Use this endpoint when you have raw, untransformed data.
    The feature pipeline will automatically apply any necessary
    transformations before making predictions.
    """

    data: dict[str, Any] = Field(
        ...,
        description="Raw feature values (before transformation). "
        "Can include original columns that will be transformed into model features.",
    )


class RawPredictionResponse(BaseModel):
    """Schema for raw data prediction response."""

    prediction: Any = Field(..., description="Model prediction")
    probabilities: Optional[dict[str, float]] = Field(
        None, description="Class probabilities (for classification)"
    )
    model_id: UUID
    model_name: str
    transformations_applied: bool = Field(
        ..., description="Whether feature transformations were applied"
    )
    transformed_features: Optional[dict[str, Any]] = Field(
        None, description="Feature values after transformation (for debugging)"
    )


class BatchPredictionRequest(BaseModel):
    """Schema for batch raw data prediction request."""

    data: list[dict[str, Any]] = Field(
        ...,
        description="List of raw data records for batch prediction",
        min_length=1,
        max_length=1000,
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    predictions: list[Any] = Field(..., description="List of predictions")
    probabilities: Optional[list[dict[str, float]]] = Field(
        None, description="List of class probabilities (for classification)"
    )
    model_id: UUID
    model_name: str
    count: int = Field(..., description="Number of predictions made")


# Model Export Schemas

class ModelExportInfoResponse(BaseModel):
    """Information about an exportable model."""

    model_id: UUID
    model_name: str
    model_type: str
    task_type: str
    can_export: bool = Field(..., description="Whether the model can be exported")
    export_size_mb: Optional[float] = Field(
        None, description="Estimated export size in MB"
    )
    has_pipeline: bool = Field(
        ..., description="Whether feature pipeline is included"
    )
    required_packages: list[str] = Field(
        default_factory=list,
        description="Python packages required to use the model",
    )


class FeaturePipelineInfoResponse(BaseModel):
    """Information about the feature pipeline for a model."""

    model_id: UUID
    has_transformations: bool
    transformation_count: int
    required_input_columns: list[str] = Field(
        ..., description="Columns required in raw input data"
    )
    output_columns: list[str] = Field(
        ..., description="Columns after transformation"
    )
    target_column: Optional[str] = None
    pipeline_config: Optional[dict[str, Any]] = Field(
        None, description="Full pipeline configuration (for debugging)"
    )


# Remote Prediction Schemas

class RemotePredictionRequest(BaseModel):
    """Schema for remote prediction request (Modal cloud)."""

    features: dict[str, Any] = Field(..., description="Feature values for prediction")


class RemotePredictionResponse(BaseModel):
    """Schema for remote prediction response from Modal."""

    prediction: Any = Field(..., description="Model prediction")
    probabilities: Optional[list[dict[str, float]]] = Field(
        None, description="Class probabilities (for classification)"
    )
    model_id: UUID
    model_name: str
    is_remote: bool = Field(True, description="Whether prediction was made remotely")
    experiment_id: Optional[str] = Field(None, description="Experiment ID used for remote lookup")


class RemoteModelStatusResponse(BaseModel):
    """Schema for remote model status check."""

    model_id: UUID
    experiment_id: Optional[str] = None
    exists_on_volume: bool = Field(..., description="Whether model is stored on Modal Volume")
    model_size_mb: Optional[float] = Field(None, description="Model size on volume in MB")
    can_predict_locally: bool = Field(..., description="Whether local prediction is available")
    can_predict_remotely: bool = Field(..., description="Whether remote prediction is available")

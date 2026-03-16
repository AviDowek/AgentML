"""Dataset specification schemas."""
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DatasetSpecBase(BaseModel):
    """Base dataset spec schema."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    data_sources_json: Optional[Any] = None  # Can be list of UUIDs or dict with join config
    target_column: Optional[str] = None
    feature_columns: Optional[list[str]] = None
    filters_json: Optional[dict[str, Any]] = None
    spec_json: Optional[dict[str, Any]] = None
    agent_experiment_design_json: Optional[dict[str, Any]] = None

    # Time-based task metadata
    is_time_based: bool = Field(
        default=False,
        description="Whether this is a time-series/temporal prediction task"
    )
    time_column: Optional[str] = Field(
        None,
        description="The datetime column used for temporal ordering (e.g., 'date', 'timestamp')"
    )
    entity_id_column: Optional[str] = Field(
        None,
        description="Column identifying unique entities for panel data (e.g., 'ticker', 'user_id')"
    )
    prediction_horizon: Optional[str] = Field(
        None,
        description="Prediction horizon in human-readable format (e.g., '1d', '5d', '1w', 'next_bar')"
    )
    target_positive_class: Optional[str] = Field(
        None,
        description="For classification: the positive class value (e.g., 'up', '1', 'True')"
    )


class DatasetSpecCreate(DatasetSpecBase):
    """Schema for creating a dataset spec."""

    pass


class DatasetSpecUpdate(BaseModel):
    """Schema for updating a dataset spec."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    data_sources_json: Optional[Any] = None  # Can be list of UUIDs or dict with join config
    target_column: Optional[str] = None
    feature_columns: Optional[list[str]] = None
    filters_json: Optional[dict[str, Any]] = None
    spec_json: Optional[dict[str, Any]] = None
    agent_experiment_design_json: Optional[dict[str, Any]] = None

    # Time-based task metadata
    is_time_based: Optional[bool] = Field(
        None,
        description="Whether this is a time-series/temporal prediction task"
    )
    time_column: Optional[str] = Field(
        None,
        description="The datetime column used for temporal ordering"
    )
    entity_id_column: Optional[str] = Field(
        None,
        description="Column identifying unique entities for panel data"
    )
    prediction_horizon: Optional[str] = Field(
        None,
        description="Prediction horizon in human-readable format"
    )
    target_positive_class: Optional[str] = Field(
        None,
        description="For classification: the positive class value"
    )


class DatasetSpecResponse(DatasetSpecBase):
    """Schema for dataset spec response."""

    id: UUID
    project_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

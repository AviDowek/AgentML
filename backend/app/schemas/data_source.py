"""Data source schemas."""
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.data_source import DataSourceType


class DataSourceBase(BaseModel):
    """Base data source schema."""

    name: str = Field(..., min_length=1, max_length=255)
    type: DataSourceType
    config_json: Optional[dict[str, Any]] = None


class DataSourceCreate(DataSourceBase):
    """Schema for creating a data source."""

    pass


class DataSourceUpdate(BaseModel):
    """Schema for updating a data source."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    config_json: Optional[dict[str, Any]] = None
    schema_summary: Optional[dict[str, Any]] = None
    profile_json: Optional[dict[str, Any]] = None


class DataSourceResponse(DataSourceBase):
    """Schema for data source response."""

    id: UUID
    project_id: UUID
    schema_summary: Optional[dict[str, Any]] = None
    profile_json: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

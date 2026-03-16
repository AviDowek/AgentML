"""Visualization schemas."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class VisualizationBase(BaseModel):
    """Base visualization schema."""

    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    chart_type: Optional[str] = Field(None, max_length=50)
    request: Optional[str] = None
    code: str
    image_base64: Optional[str] = None
    explanation: Optional[str] = None
    is_ai_suggested: Optional[str] = Field(default="false", max_length=10)
    display_order: Optional[str] = Field(default="0", max_length=20)


class VisualizationCreate(VisualizationBase):
    """Schema for creating a visualization."""

    data_source_id: Optional[UUID] = None


class VisualizationUpdate(BaseModel):
    """Schema for updating a visualization."""

    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    display_order: Optional[str] = Field(None, max_length=20)


class VisualizationResponse(VisualizationBase):
    """Schema for visualization response."""

    id: UUID
    project_id: UUID
    data_source_id: Optional[UUID] = None
    owner_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

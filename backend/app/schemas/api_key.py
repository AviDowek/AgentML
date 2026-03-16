"""API key schemas."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.api_key import LLMProvider


class ApiKeyCreate(BaseModel):
    """Schema for creating/updating an API key."""
    provider: LLMProvider = Field(..., description="LLM provider (openai or gemini)")
    api_key: str = Field(..., min_length=1, description="The API key")
    name: Optional[str] = Field(None, description="Display name for this key")


class ApiKeyResponse(BaseModel):
    """Schema for API key response (key is masked)."""
    id: UUID
    provider: LLMProvider
    name: Optional[str]
    key_preview: str  # Masked version of the key
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ApiKeyStatusResponse(BaseModel):
    """Schema for checking which providers have keys configured."""
    openai: bool
    gemini: bool

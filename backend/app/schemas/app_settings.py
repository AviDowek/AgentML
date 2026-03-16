"""App settings schemas."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AIModelOption(BaseModel):
    """Information about an AI model option."""
    value: str
    display_name: str
    description: str


class AppSettingsResponse(BaseModel):
    """Current app settings."""
    ai_model: str
    ai_model_display_name: str
    updated_at: datetime

    class Config:
        from_attributes = True


class AppSettingsUpdate(BaseModel):
    """Schema for updating app settings."""
    ai_model: Optional[str] = None

"""Conversation schemas."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ConversationMessageBase(BaseModel):
    """Base schema for conversation message."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ConversationMessageCreate(ConversationMessageBase):
    """Schema for creating a message."""
    pass


class ConversationMessageResponse(ConversationMessageBase):
    """Response schema for a message."""
    id: UUID
    conversation_id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationBase(BaseModel):
    """Base schema for conversation."""
    title: str = Field(default="New Conversation", max_length=255)
    context_type: Optional[str] = None
    context_id: Optional[str] = None


class ConversationCreate(ConversationBase):
    """Schema for creating a conversation."""
    context_data: Optional[dict] = None


class ConversationUpdate(BaseModel):
    """Schema for updating a conversation."""
    title: Optional[str] = Field(None, max_length=255)


class ConversationSummary(BaseModel):
    """Summary schema for conversation list."""
    id: UUID
    title: str
    context_type: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    last_message_preview: Optional[str] = None

    model_config = {"from_attributes": True}


class ConversationResponse(ConversationBase):
    """Full response schema for a conversation."""
    id: UUID
    context_data: Optional[dict]
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessageResponse] = []

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""
    items: list[ConversationSummary]
    total: int


class VisualizationImage(BaseModel):
    """Visualization image data for AI context."""
    title: str = Field(..., description="Chart title")
    description: Optional[str] = Field(None, description="Chart description")
    chart_type: Optional[str] = Field(None, description="Type of chart")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    message: str = Field(..., min_length=1, description="User's message")
    current_visualizations: Optional[list[VisualizationImage]] = Field(
        None,
        description="Current visualizations to share with AI (images included)"
    )


class SendMessageResponse(BaseModel):
    """Response after sending a message."""
    user_message: ConversationMessageResponse
    assistant_message: ConversationMessageResponse
    provider: str

"""Context Document schemas for API requests/responses."""
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


class ContextDocumentCreate(BaseModel):
    """Schema for creating a context document (via form data, not JSON body)."""
    name: str = Field(..., min_length=1, max_length=255, description="User-provided name for the document")
    explanation: str = Field(..., min_length=10, max_length=5000, description="Explanation of what this document contains and how it should be used")


class ContextDocumentUpdate(BaseModel):
    """Schema for updating a context document."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    explanation: Optional[str] = Field(None, min_length=10, max_length=5000)
    is_active: Optional[bool] = None


class ContextDocumentResponse(BaseModel):
    """Response schema for a context document."""
    id: UUID
    project_id: UUID
    name: str
    original_filename: str
    file_type: str
    file_size_bytes: int
    explanation: str
    extraction_status: str
    extraction_error: Optional[str] = None
    is_active: bool
    has_content: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ContextDocumentDetailResponse(ContextDocumentResponse):
    """Detailed response including extracted text (for preview)."""
    extracted_text: Optional[str] = None
    content_preview: Optional[str] = Field(None, description="First 500 chars of content for preview")

    @classmethod
    def from_orm_with_preview(cls, obj, preview_length: int = 500):
        """Create response with content preview."""
        content_preview = None
        if obj.extracted_text:
            content_preview = obj.extracted_text[:preview_length]
            if len(obj.extracted_text) > preview_length:
                content_preview += "..."

        return cls(
            id=obj.id,
            project_id=obj.project_id,
            name=obj.name,
            original_filename=obj.original_filename,
            file_type=obj.file_type,
            file_size_bytes=obj.file_size_bytes,
            explanation=obj.explanation,
            extraction_status=obj.extraction_status,
            extraction_error=obj.extraction_error,
            is_active=obj.is_active,
            has_content=obj.has_content,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            extracted_text=obj.extracted_text,
            content_preview=content_preview,
        )


class ContextDocumentListResponse(BaseModel):
    """Response for listing context documents."""
    documents: List[ContextDocumentResponse]
    total: int
    total_active: int


class ContextDocumentSummary(BaseModel):
    """Summary for display in project context."""
    id: UUID
    name: str
    file_type: str
    explanation: str
    is_active: bool
    has_content: bool

    class Config:
        from_attributes = True


class SupportedExtensionsResponse(BaseModel):
    """Response listing supported file extensions."""
    extensions: dict[str, str]
    max_file_size_mb: Optional[float] = None  # None means no limit

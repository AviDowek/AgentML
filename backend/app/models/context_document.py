"""Context Document model for supplementary project knowledge.

Context documents allow users to upload supporting documentation (PDFs, Word docs,
text files, images) with explanations. The AI agents use this context to make
better-informed decisions about dataset design and experiments.
"""
from sqlalchemy import Column, String, Text, ForeignKey, Integer, Boolean
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base
from app.models.base import TimestampMixin, GUID


class ContextDocument(Base, TimestampMixin):
    """Context document model - stores supplementary project documentation.

    Users can upload documents with required explanations. The system extracts
    text content and injects it into AI agent prompts for better decision-making.

    Supports A/B testing: experiments can be run with and without context
    to measure the impact of the documentation on model performance.
    """

    __tablename__ = "context_documents"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        GUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Document metadata
    name = Column(String(255), nullable=False, doc="User-provided name for the document")
    original_filename = Column(String(500), nullable=False, doc="Original uploaded filename")
    file_path = Column(String(1000), nullable=False, doc="Path to stored file")
    file_type = Column(String(50), nullable=False, doc="File type: pdf, docx, txt, md, html, image")
    file_size_bytes = Column(Integer, nullable=False, doc="File size in bytes")

    # REQUIRED user explanation - this is critical for context
    explanation = Column(
        Text,
        nullable=False,
        doc="User's explanation of what this document contains and how it should be used"
    )

    # Extracted text content for prompt injection
    extracted_text = Column(
        Text,
        nullable=True,
        doc="Text content extracted from the document"
    )
    extraction_status = Column(
        String(50),
        default="pending",
        nullable=False,
        doc="Status of text extraction: pending, completed, failed"
    )
    extraction_error = Column(
        Text,
        nullable=True,
        doc="Error message if extraction failed"
    )

    # Usage control
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether this document should be included in agent prompts"
    )

    # Relationships
    project = relationship("Project", back_populates="context_documents")

    def __repr__(self):
        return f"<ContextDocument {self.name} ({self.file_type})>"

    @property
    def has_content(self) -> bool:
        """Check if document has extracted content available."""
        return (
            self.extraction_status == "completed"
            and self.extracted_text is not None
            and len(self.extracted_text.strip()) > 0
        )

    @property
    def content_for_prompt(self) -> str:
        """Get content suitable for prompt injection.

        Returns the extracted text if available, otherwise just the explanation.
        For images, only the explanation is used (no OCR).
        """
        if self.file_type == "image":
            return f"[Image document - see explanation for details]"
        if self.has_content:
            return self.extracted_text
        return "[Content extraction pending or failed]"

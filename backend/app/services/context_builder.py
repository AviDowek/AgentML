"""Context Builder Service - builds formatted context for prompt injection.

This service fetches active context documents from a project and formats them
for injection into AI agent prompts. It handles:
- Fetching active documents from the database
- Formatting content with explanations
- Token limit management (truncation if needed)
- Summary generation for logging/debugging
"""
import logging
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.context_document import ContextDocument

logger = logging.getLogger(__name__)


# Approximate characters per token (rough estimate for planning)
CHARS_PER_TOKEN = 4

# Default max tokens for context section
DEFAULT_MAX_TOKENS = 4000


class ContextBuilder:
    """Builds formatted context sections from project documents."""

    def __init__(self, db: Session):
        self.db = db

    def get_active_documents(self, project_id: UUID) -> List[ContextDocument]:
        """Get all active context documents for a project."""
        # Debug logging
        logger.info(f"📚 ContextBuilder: Fetching documents for project {project_id}")
        print(f"📚 ContextBuilder: Fetching documents for project {project_id}")

        docs = (
            self.db.query(ContextDocument)
            .filter(
                ContextDocument.project_id == project_id,
                ContextDocument.is_active == True
            )
            .order_by(ContextDocument.created_at)
            .all()
        )

        logger.info(f"📚 ContextBuilder: Found {len(docs)} active documents")
        print(f"📚 ContextBuilder: Found {len(docs)} active documents")
        for doc in docs:
            logger.info(f"📚   - {doc.name}: {doc.file_type}, active={doc.is_active}")
            print(f"📚   - {doc.name}: {doc.file_type}, active={doc.is_active}")

        return docs

    def build_context_section(
        self,
        project_id: UUID,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        include_usage_guidance: bool = True,
    ) -> str:
        """Build formatted context section from active documents.

        Args:
            project_id: ID of the project to get documents for
            max_tokens: Maximum tokens for the context section
            include_usage_guidance: Whether to include usage guidance text

        Returns:
            Formatted context section string, or empty string if no documents
        """
        documents = self.get_active_documents(project_id)

        if not documents:
            return ""

        max_chars = max_tokens * CHARS_PER_TOKEN
        sections = []
        total_chars = 0

        # Header
        header = "## Project Context Documents\n\n"
        header += "The following supplementary documents have been provided to help understand this project:\n\n"
        sections.append(header)
        total_chars += len(header)

        for doc in documents:
            doc_section = self._format_document(doc, max_chars - total_chars - 500)  # Reserve 500 for guidance

            if doc_section:
                sections.append(doc_section)
                total_chars += len(doc_section)

                # Check if we're approaching the limit
                if total_chars >= max_chars - 500:
                    sections.append("\n*[Additional context documents truncated due to length limits]*\n")
                    break

        # Add usage guidance
        if include_usage_guidance and sections:
            guidance = self._get_usage_guidance()
            sections.append(guidance)

        return "".join(sections)

    def _format_document(self, doc: ContextDocument, max_chars: int) -> str:
        """Format a single document for context injection.

        Args:
            doc: The context document to format
            max_chars: Maximum characters for this document's section

        Returns:
            Formatted document section
        """
        parts = []

        # Document header
        parts.append(f"### {doc.name}\n")
        parts.append(f"**Type:** {doc.file_type.upper()} | **File:** {doc.original_filename}\n\n")

        # User's explanation (always included - this is critical context)
        parts.append(f"**Purpose:** {doc.explanation}\n\n")

        # Content section
        content = doc.content_for_prompt
        if content and content not in ["[Image document - see explanation for details]", "[Content extraction pending or failed]"]:
            # Calculate available space for content
            header_len = sum(len(p) for p in parts)
            available_chars = max_chars - header_len - 100  # Reserve 100 for truncation notice

            if len(content) > available_chars and available_chars > 100:
                content = content[:available_chars] + "\n\n*[Content truncated...]*"

            parts.append("**Content:**\n")
            parts.append(f"```\n{content}\n```\n")
        elif doc.file_type == 'image':
            parts.append("*[Image document - refer to purpose above for details]*\n")
        else:
            parts.append("*[Content extraction pending or failed - refer to purpose above]*\n")

        parts.append("\n---\n\n")

        return "".join(parts)

    def _get_usage_guidance(self) -> str:
        """Get guidance text for how to use context documents."""
        return """
**How to Use This Context:**

Use the above context documents to inform your decisions about:
- **Feature Selection:** Consider domain-specific features mentioned in the documentation
- **Feature Engineering:** Apply domain knowledge for meaningful transformations
- **Model Configuration:** Adjust settings based on problem characteristics
- **Result Interpretation:** Understand what metrics and outcomes mean in context
- **Data Quality:** Watch for issues specific to this domain

The user's explanations for each document are particularly important - they indicate
what the user considers relevant and how the information should be applied.

---

"""

    def get_context_summary(self, project_id: UUID) -> Dict[str, Any]:
        """Get summary of available context for logging.

        Args:
            project_id: ID of the project

        Returns:
            Dict with context summary information
        """
        documents = self.get_active_documents(project_id)

        if not documents:
            return {
                "has_context": False,
                "document_count": 0,
                "documents": [],
            }

        doc_summaries = []
        total_chars = 0

        for doc in documents:
            content_length = len(doc.extracted_text) if doc.extracted_text else 0
            doc_summaries.append({
                "id": str(doc.id),
                "name": doc.name,
                "file_type": doc.file_type,
                "has_content": doc.has_content,
                "content_length": content_length,
                "explanation_preview": doc.explanation[:100] + "..." if len(doc.explanation) > 100 else doc.explanation,
            })
            total_chars += content_length

        return {
            "has_context": True,
            "document_count": len(documents),
            "total_content_chars": total_chars,
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "documents": doc_summaries,
        }

    def has_context_documents(self, project_id: UUID) -> bool:
        """Check if project has any active context documents.

        Args:
            project_id: ID of the project

        Returns:
            True if there are active context documents
        """
        count = (
            self.db.query(ContextDocument)
            .filter(
                ContextDocument.project_id == project_id,
                ContextDocument.is_active == True
            )
            .count()
        )
        return count > 0


def build_context_for_prompt(
    db: Session,
    project_id: UUID,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Convenience function to build context section.

    Args:
        db: Database session
        project_id: ID of the project
        max_tokens: Maximum tokens for context section

    Returns:
        Formatted context section string
    """
    builder = ContextBuilder(db)
    return builder.build_context_section(project_id, max_tokens)


def get_context_summary(db: Session, project_id: UUID) -> Dict[str, Any]:
    """Convenience function to get context summary.

    Args:
        db: Database session
        project_id: ID of the project

    Returns:
        Context summary dict
    """
    builder = ContextBuilder(db)
    return builder.get_context_summary(project_id)

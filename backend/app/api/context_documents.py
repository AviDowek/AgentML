"""Context Documents API endpoints.

Provides endpoints for uploading, managing, and retrieving context documents
that supplement project knowledge for AI agents.
"""
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.api.dependencies import check_project_access
from app.models.user import User
from app.models.project import Project
from app.models.context_document import ContextDocument
from app.schemas.context_document import (
    ContextDocumentUpdate,
    ContextDocumentResponse,
    ContextDocumentDetailResponse,
    ContextDocumentListResponse,
    SupportedExtensionsResponse,
)
from app.services.context_document_handler import (
    ContextDocumentHandler,
    get_supported_extensions,
)


logger = logging.getLogger(__name__)

router = APIRouter(tags=["context-documents"])


@router.get(
    "/context-documents/supported-extensions",
    response_model=SupportedExtensionsResponse,
)
async def get_supported_file_extensions(current_user: Optional[User] = Depends(get_current_user)):
    """Get list of supported file extensions for context documents."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return SupportedExtensionsResponse(
        extensions=get_supported_extensions(),
        max_file_size_mb=None,  # No file size limit
    )


@router.post(
    "/projects/{project_id}/context-documents/upload",
    response_model=ContextDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_context_document(
    project_id: UUID,
    file: UploadFile = File(..., description="The document file to upload"),
    name: str = Form(..., min_length=1, max_length=255, description="Name for this document"),
    explanation: str = Form(..., min_length=10, max_length=5000, description="Explanation of what this document contains and how it should be used"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Upload a context document to a project.

    Context documents provide supplementary information that AI agents use
    when designing datasets and experiments. Users must provide an explanation
    of what the document contains and how it should be used.

    Supported file types:
    - Documents: PDF, Word (.docx, .doc), Text (.txt), Markdown (.md), HTML
    - Images: PNG, JPG, JPEG, GIF, WebP, BMP, TIFF (explanation only, no OCR)

    Args:
        project_id: UUID of the project
        file: The file to upload
        name: User-provided name for the document
        explanation: Explanation of what this document contains (REQUIRED)
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    # Validate file has a name
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename",
        )

    try:
        handler = ContextDocumentHandler(db)
        context_doc = handler.save_and_create(
            file=file.file,
            original_filename=file.filename,
            project_id=project_id,
            name=name,
            explanation=explanation,
        )

        logger.info(f"Uploaded context document {context_doc.id} for project {project_id}")
        return context_doc

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Failed to upload context document for project {project_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}",
        )


@router.get(
    "/projects/{project_id}/context-documents",
    response_model=ContextDocumentListResponse,
)
async def list_context_documents(
    project_id: UUID,
    include_inactive: bool = False,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List all context documents for a project.

    Args:
        project_id: UUID of the project
        include_inactive: Whether to include inactive documents (default: False)
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this project")

    query = db.query(ContextDocument).filter(ContextDocument.project_id == project_id)

    if not include_inactive:
        query = query.filter(ContextDocument.is_active == True)

    documents = query.order_by(ContextDocument.created_at.desc()).all()

    # Count active documents
    total_active = db.query(ContextDocument).filter(
        ContextDocument.project_id == project_id,
        ContextDocument.is_active == True
    ).count()

    return ContextDocumentListResponse(
        documents=documents,
        total=len(documents),
        total_active=total_active,
    )


@router.get(
    "/context-documents/{document_id}",
    response_model=ContextDocumentDetailResponse,
)
async def get_context_document(
    document_id: UUID,
    include_content: bool = False,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a single context document by ID.

    Args:
        document_id: UUID of the context document
        include_content: Whether to include full extracted text (default: False)
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    context_doc = db.query(ContextDocument).filter(ContextDocument.id == document_id).first()

    if not context_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context document {document_id} not found",
        )
    project = db.query(Project).filter(Project.id == context_doc.project_id).first()
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    # Use custom method to include preview
    return ContextDocumentDetailResponse.from_orm_with_preview(
        context_doc,
        preview_length=2000 if include_content else 500
    )


@router.put(
    "/context-documents/{document_id}",
    response_model=ContextDocumentResponse,
)
async def update_context_document(
    document_id: UUID,
    update: ContextDocumentUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update a context document's metadata.

    Can update name, explanation, and active status.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    context_doc = db.query(ContextDocument).filter(ContextDocument.id == document_id).first()

    if not context_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context document {document_id} not found",
        )
    project = db.query(Project).filter(Project.id == context_doc.project_id).first()
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    update_data = update.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(context_doc, field, value)

    db.commit()
    db.refresh(context_doc)

    logger.info(f"Updated context document {document_id}")
    return context_doc


@router.delete(
    "/context-documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_context_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Delete a context document and its file."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    context_doc = db.query(ContextDocument).filter(ContextDocument.id == document_id).first()

    if not context_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context document {document_id} not found",
        )
    project = db.query(Project).filter(Project.id == context_doc.project_id).first()
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    try:
        handler = ContextDocumentHandler(db)
        handler.delete_document(context_doc)
        logger.info(f"Deleted context document {document_id}")
    except Exception as e:
        logger.exception(f"Failed to delete context document {document_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )


@router.post(
    "/context-documents/{document_id}/reextract",
    response_model=ContextDocumentResponse,
)
async def reextract_text(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Re-run text extraction for a document.

    Useful if extraction initially failed or to refresh content.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    context_doc = db.query(ContextDocument).filter(ContextDocument.id == document_id).first()

    if not context_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context document {document_id} not found",
        )
    project = db.query(Project).filter(Project.id == context_doc.project_id).first()
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    try:
        handler = ContextDocumentHandler(db)
        handler.re_extract_text(context_doc)
        db.refresh(context_doc)

        logger.info(f"Re-extracted text for context document {document_id}")
        return context_doc

    except Exception as e:
        logger.exception(f"Failed to re-extract text for document {document_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to re-extract text: {str(e)}",
        )


@router.post(
    "/context-documents/{document_id}/toggle-active",
    response_model=ContextDocumentResponse,
)
async def toggle_document_active(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Toggle a document's active status."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    context_doc = db.query(ContextDocument).filter(ContextDocument.id == document_id).first()

    if not context_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Context document {document_id} not found",
        )
    project = db.query(Project).filter(Project.id == context_doc.project_id).first()
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    context_doc.is_active = not context_doc.is_active
    db.commit()
    db.refresh(context_doc)

    logger.info(f"Toggled context document {document_id} active status to {context_doc.is_active}")
    return context_doc

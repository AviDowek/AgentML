"""Data source API endpoints."""
import os
import uuid as uuid_module
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.project import Project
from app.models.data_source import DataSource
from app.schemas.data_source import (
    DataSourceCreate,
    DataSourceUpdate,
    DataSourceResponse,
)
from app.services.schema_analyzer import SchemaAnalyzer, SUPPORTED_EXTENSIONS, get_file_type
from app.services.file_handlers import read_file, get_handler
from app.services.data_profiler import profile_data_source, profile_all_data_sources
from app.core.security import get_current_user
from app.models.user import User
from app.api.dependencies import get_project_with_access, get_project_with_write_access


class DataPreviewResponse(BaseModel):
    """Response model for data preview."""
    columns: list[str]
    rows: list[list[Any]]
    total_rows: int
    total_columns: int
    page: int
    page_size: int
    has_more: bool

router = APIRouter(tags=["data-sources"])


@router.post(
    "/projects/{project_id}/data-sources/upload",
    response_model=DataSourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_data_source(
    project_id: UUID,
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    delimiter: str = Form(","),
    sheet_name: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Upload a data file as a new data source.

    Supported file types:
    - Tabular: CSV, TSV, Excel (.xlsx, .xls, .xlsm), JSON, JSON Lines
    - Columnar: Parquet, Feather/Arrow, ORC
    - Databases: SQLite (.db, .sqlite, .sqlite3)
    - Statistical: SAS (.sas7bdat, .xpt), Stata (.dta), SPSS (.sav)
    - Structured: XML, HTML (tables), HDF5 (.h5)
    - Other: Pickle (.pkl), Fixed-width (.fwf), Text, Word (.docx)

    Creates a data source record with type='file_upload' and runs
    schema analysis to populate schema_summary.

    Args:
        project_id: UUID of the project
        file: The file to upload
        name: Optional custom name for the data source
        delimiter: CSV delimiter (default: ",")
        sheet_name: Excel sheet name to analyze (default: first sheet)
        table_name: SQLite table name to analyze (default: first table)
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    settings = get_settings()

    # Verify project exists and user has write access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Validate file has a name
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename",
        )

    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_ext}. Supported types: {supported}",
        )

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum allowed ({settings.max_upload_size_mb}MB)",
        )

    # Create upload directory if it doesn't exist
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    file_id = uuid_module.uuid4()
    safe_filename = f"{file_id}_{file.filename}"
    file_path = upload_dir / safe_filename

    # Save file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    # Analyze schema based on file type
    try:
        analyzer = SchemaAnalyzer()
        file_type = get_file_type(file_path)

        # Build kwargs for analysis based on file type
        analyze_kwargs = {}
        if file_type == "csv":
            analyze_kwargs["delimiter"] = delimiter
        elif file_type == "excel" and sheet_name:
            analyze_kwargs["sheet_name"] = sheet_name
        elif file_type == "sqlite" and table_name:
            analyze_kwargs["table_name"] = table_name

        schema_summary = analyzer.analyze_file(file_path, **analyze_kwargs)
    except Exception as e:
        # Clean up file on analysis failure
        os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to analyze file: {str(e)}",
        )

    # Use provided name or original filename
    source_name = name or file.filename

    # Build config based on file type
    config_json = {
        "file_path": str(file_path.absolute()),
        "original_filename": file.filename,
        "file_type": file_type,
        "file_size_bytes": file_size,
    }

    # Add type-specific config
    if file_type == "csv":
        config_json["delimiter"] = delimiter
    elif file_type == "excel":
        config_json["sheet_name"] = sheet_name or schema_summary.get("analyzed_sheet", "Sheet1")
        config_json["sheet_names"] = schema_summary.get("sheet_names", [])
    elif file_type == "sqlite":
        config_json["table_name"] = table_name or schema_summary.get("analyzed_table")
        config_json["tables"] = schema_summary.get("tables", [])
    elif file_type == "hdf5":
        config_json["key"] = schema_summary.get("analyzed_key")
        config_json["keys"] = schema_summary.get("keys", [])
    elif file_type == "html":
        config_json["table_index"] = schema_summary.get("analyzed_table_index", 0)
        config_json["table_count"] = schema_summary.get("table_count", 1)

    # Create data source record
    db_data_source = DataSource(
        project_id=project_id,
        name=source_name,
        type="file_upload",
        config_json=config_json,
        schema_summary=schema_summary,
    )
    db.add(db_data_source)
    db.commit()
    db.refresh(db_data_source)
    return db_data_source


@router.post(
    "/projects/{project_id}/data-sources",
    response_model=DataSourceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_data_source(
    project_id: UUID,
    data_source: DataSourceCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new data source for a project."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    db_data_source = DataSource(
        project_id=project_id,
        name=data_source.name,
        type=data_source.type,
        config_json=data_source.config_json,
    )
    db.add(db_data_source)
    db.commit()
    db.refresh(db_data_source)
    return db_data_source


@router.get(
    "/projects/{project_id}/data-sources",
    response_model=list[DataSourceResponse],
)
def list_data_sources(project_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """List all data sources for a project."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    return db.query(DataSource).filter(DataSource.project_id == project_id).all()


@router.get(
    "/data-sources/{data_source_id}",
    response_model=DataSourceResponse,
)
def get_data_source(data_source_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Get a data source by ID."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {data_source_id} not found",
        )
    return data_source


@router.put(
    "/data-sources/{data_source_id}",
    response_model=DataSourceResponse,
)
def update_data_source(
    data_source_id: UUID,
    data_source_update: DataSourceUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update a data source."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {data_source_id} not found",
        )

    update_data = data_source_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(data_source, field, value)

    db.commit()
    db.refresh(data_source)
    return data_source


@router.delete("/data-sources/{data_source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_data_source(data_source_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Delete a data source."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {data_source_id} not found",
        )

    db.delete(data_source)
    db.commit()
    return None


@router.get(
    "/data-sources/{data_source_id}/data",
    response_model=DataPreviewResponse,
)
def get_data_source_data(
    data_source_id: UUID,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(100, ge=1, le=1000, description="Rows per page"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get the actual data from a data source with pagination.

    Returns the raw data from the file in a tabular format suitable for display.
    Supports pagination to handle large files efficiently.

    Args:
        data_source_id: UUID of the data source
        page: Page number (1-indexed, default 1)
        page_size: Number of rows per page (default 100, max 1000)
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {data_source_id} not found",
        )

    # Get file path from config
    config = data_source.config_json or {}
    file_path = config.get("file_path")

    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data source does not have a file path",
        )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data file not found on disk",
        )

    # Determine file type and read data using the file handler system
    try:
        # Build kwargs from config for handler-specific options
        read_kwargs = {}
        if config.get("delimiter"):
            read_kwargs["delimiter"] = config["delimiter"]
        if config.get("sheet_name"):
            read_kwargs["sheet_name"] = config["sheet_name"]
        if config.get("table_name"):
            read_kwargs["table_name"] = config["table_name"]
        if config.get("key"):
            read_kwargs["key"] = config["key"]
        if config.get("table_index"):
            read_kwargs["table_index"] = config["table_index"]

        # Use the unified file handler system
        handler = get_handler(file_path)
        if not handler:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type for data preview",
            )

        # Read without sample limit for preview (we paginate instead)
        df, _ = handler.read(Path(file_path), sample_rows=None, **read_kwargs)

        total_rows = len(df)
        total_columns = len(df.columns)

        # Calculate pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        # Get the slice of data
        df_slice = df.iloc[start_idx:end_idx]

        # Convert to list of lists, handling NaN values
        rows = []
        for _, row in df_slice.iterrows():
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append(None)
                elif isinstance(val, (pd.Timestamp,)):
                    row_data.append(str(val))
                else:
                    row_data.append(val)
            rows.append(row_data)

        return DataPreviewResponse(
            columns=list(df.columns.astype(str)),
            rows=rows,
            total_rows=total_rows,
            total_columns=total_columns,
            page=page,
            page_size=page_size,
            has_more=end_idx < total_rows,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read data file: {str(e)}",
        )


class DataProfileResponse(BaseModel):
    """Response model for a single data source profile."""
    source_id: str
    source_name: str
    source_type: str
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    table_name: Optional[str] = None
    source_url: Optional[str] = None
    estimated_row_count: int
    column_count: int
    columns: list[dict[str, Any]]
    profiled_at: str
    sample_size: int
    warnings: list[str]
    is_estimate: Optional[bool] = None


class ProfileAllResponse(BaseModel):
    """Response model for profiling all data sources in a project."""
    project_id: str
    profiles: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    total_sources: int
    profiled_count: int
    error_count: int


@router.post(
    "/projects/{project_id}/data-sources/profile-all",
    response_model=ProfileAllResponse,
)
def profile_all_project_data_sources(
    project_id: UUID,
    sample_rows: int = Query(50000, ge=1000, le=100000, description="Max rows to sample per source"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Profile all data sources in a project.

    This endpoint runs data profiling on all data sources in the project,
    generating detailed statistics and quality assessments that the AI
    Data Architect agent can use for analysis.

    The profiling is idempotent - running multiple times will update the
    profiles with fresh data.

    Args:
        project_id: UUID of the project
        sample_rows: Maximum rows to sample per data source (default 50000, max 100000)

    Returns:
        ProfileAllResponse with profiles for all data sources
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    try:
        result = profile_all_data_sources(db, project_id, sample_rows=sample_rows)

        # Store profiles in each data source's profile_json field
        for profile in result["profiles"]:
            source_id = profile.get("source_id")
            if source_id:
                data_source = db.query(DataSource).filter(
                    DataSource.id == UUID(source_id)
                ).first()
                if data_source:
                    data_source.profile_json = profile

        db.commit()

        return ProfileAllResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to profile data sources: {str(e)}",
        )


@router.post(
    "/data-sources/{data_source_id}/profile",
    response_model=DataProfileResponse,
)
def profile_single_data_source(
    data_source_id: UUID,
    sample_rows: int = Query(50000, ge=1000, le=100000, description="Max rows to sample"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Profile a single data source.

    Runs data profiling on a specific data source, generating detailed
    statistics and quality assessments.

    Args:
        data_source_id: UUID of the data source to profile
        sample_rows: Maximum rows to sample (default 50000, max 100000)

    Returns:
        DataProfileResponse with the profile data
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source {data_source_id} not found",
        )

    try:
        profile = profile_data_source(db, data_source_id, sample_rows=sample_rows)

        # Store the profile in the data source's profile_json field
        data_source.profile_json = profile
        db.commit()

        return DataProfileResponse(**profile)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to profile data source: {str(e)}",
        )

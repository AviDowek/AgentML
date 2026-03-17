"""Dataset specification API endpoints."""
import io
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.api.dependencies import check_project_access
from app.models.user import User
from app.models.project import Project
from app.models.dataset_spec import DatasetSpec
from app.models.experiment import Experiment, ExperimentStatus
from app.models.experiment import Trial
from app.models.model_version import ModelVersion
from app.schemas.dataset_spec import (
    DatasetSpecCreate,
    DatasetSpecUpdate,
    DatasetSpecResponse,
)
from app.services.dataset_builder import DatasetBuilder


class DatasetPreviewResponse(BaseModel):
    """Response model for dataset preview."""
    columns: list[str]
    rows: list[list[Any]]
    total_rows: int
    total_columns: int
    page: int
    page_size: int
    has_more: bool


class DatasetExperimentResponse(BaseModel):
    """Response for an experiment associated with a dataset."""
    id: UUID
    name: str
    description: Optional[str] = None
    status: str
    primary_metric: Optional[str] = None
    metric_direction: Optional[str] = None
    best_score: Optional[float] = None
    best_metrics: Optional[Dict[str, Any]] = None
    best_model_type: Optional[str] = None
    iteration_number: int = 1
    parent_experiment_id: Optional[UUID] = None
    improvement_summary: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    training_time_seconds: Optional[float] = None


class DatasetResultsResponse(BaseModel):
    """Response for dataset results with all experiments."""
    dataset_spec_id: UUID
    dataset_name: str
    dataset_description: Optional[str] = None
    target_column: Optional[str] = None
    total_experiments: int
    completed_experiments: int
    best_experiment_id: Optional[UUID] = None
    best_score: Optional[float] = None
    primary_metric: Optional[str] = None
    experiments: List[DatasetExperimentResponse]


router = APIRouter(tags=["dataset-specs"])


@router.post(
    "/projects/{project_id}/dataset-specs",
    response_model=DatasetSpecResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_dataset_spec(
    project_id: UUID,
    dataset_spec: DatasetSpecCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new dataset specification for a project."""
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

    db_dataset_spec = DatasetSpec(
        project_id=project_id,
        name=dataset_spec.name,
        description=dataset_spec.description,
        data_sources_json=dataset_spec.data_sources_json,
        target_column=dataset_spec.target_column,
        feature_columns=dataset_spec.feature_columns,
        filters_json=dataset_spec.filters_json,
        spec_json=dataset_spec.spec_json,
        # Time-based task metadata
        is_time_based=dataset_spec.is_time_based,
        time_column=dataset_spec.time_column,
        entity_id_column=dataset_spec.entity_id_column,
        prediction_horizon=dataset_spec.prediction_horizon,
        target_positive_class=dataset_spec.target_positive_class,
    )
    db.add(db_dataset_spec)
    db.commit()
    db.refresh(db_dataset_spec)
    return db_dataset_spec


@router.get(
    "/projects/{project_id}/dataset-specs",
    response_model=list[DatasetSpecResponse],
)
def list_dataset_specs(project_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """List all dataset specifications for a project."""
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

    return db.query(DatasetSpec).filter(DatasetSpec.project_id == project_id).all()


@router.get(
    "/dataset-specs/{dataset_spec_id}",
    response_model=DatasetSpecResponse,
)
def get_dataset_spec(dataset_spec_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Get a dataset specification by ID."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
    if not dataset_spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {dataset_spec_id} not found",
        )
    project = db.query(Project).filter(Project.id == dataset_spec.project_id).first()
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")
    return dataset_spec


@router.put(
    "/dataset-specs/{dataset_spec_id}",
    response_model=DatasetSpecResponse,
)
def update_dataset_spec(
    dataset_spec_id: UUID,
    dataset_spec_update: DatasetSpecUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update a dataset specification."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
    if not dataset_spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {dataset_spec_id} not found",
        )
    project = db.query(Project).filter(Project.id == dataset_spec.project_id).first()
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    update_data = dataset_spec_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(dataset_spec, field, value)

    db.commit()
    db.refresh(dataset_spec)
    return dataset_spec


@router.delete("/dataset-specs/{dataset_spec_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset_spec(dataset_spec_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Delete a dataset specification."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
    if not dataset_spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {dataset_spec_id} not found",
        )
    project = db.query(Project).filter(Project.id == dataset_spec.project_id).first()
    if not check_project_access(db, project, current_user, require_write=True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    db.delete(dataset_spec)
    db.commit()
    return None


@router.get(
    "/dataset-specs/{dataset_spec_id}/data",
    response_model=DatasetPreviewResponse,
)
def get_dataset_spec_data(
    dataset_spec_id: UUID,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(100, ge=1, le=1000, description="Rows per page"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get data preview for a dataset specification.

    This endpoint builds the dataset from the configured data sources
    and returns a paginated preview of the data.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
    if not dataset_spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {dataset_spec_id} not found",
        )
    project = db.query(Project).filter(Project.id == dataset_spec.project_id).first()
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    try:
        # Build the dataset from sources
        builder = DatasetBuilder(db)
        df = builder.build_dataset_from_spec(dataset_spec_id)

        total_rows = len(df)
        total_columns = len(df.columns)

        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        df_page = df.iloc[start_idx:end_idx]

        # Convert to list of lists, handling various types
        rows = []
        for _, row in df_page.iterrows():
            row_data = []
            for val in row:
                if val is None or (hasattr(val, '__class__') and val.__class__.__name__ == 'NaT'):
                    row_data.append(None)
                elif hasattr(val, 'item'):  # numpy types
                    row_data.append(val.item())
                elif isinstance(val, (int, float, bool, str)):
                    row_data.append(val)
                else:
                    row_data.append(str(val))
            rows.append(row_data)

        return DatasetPreviewResponse(
            columns=list(df.columns.astype(str)),
            rows=rows,
            total_rows=total_rows,
            total_columns=total_columns,
            page=page,
            page_size=page_size,
            has_more=end_idx < total_rows,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build dataset: {str(e)}",
        )


# Error metrics for normalizing display values
ERROR_METRICS = {
    "root_mean_squared_error", "mean_squared_error", "mean_absolute_error",
    "rmse", "mse", "mae", "neg_root_mean_squared_error", "neg_mean_squared_error",
    "neg_mean_absolute_error"
}


def _normalize_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize metrics for display by converting negative error metrics to positive."""
    if not metrics:
        return {}

    normalized = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if key.lower() in ERROR_METRICS and value < 0:
                normalized[key] = abs(value)
            else:
                normalized[key] = value
        else:
            normalized[key] = value
    return normalized


@router.get(
    "/dataset-specs/{dataset_spec_id}/experiments",
    response_model=DatasetResultsResponse,
    summary="Get all experiments for a dataset",
    description="""Get all experiments that use this dataset specification.

Returns a comparison view of all experiments including:
- Status and metrics for each experiment
- Best performing experiment
- Auto-improvement iteration chains
- Training time and model type information

This is useful for comparing multiple experiment runs on the same dataset.
""",
)
def get_dataset_experiments(
    dataset_spec_id: UUID,
    include_iterations: bool = Query(True, description="Include auto-improve iterations"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get all experiments for a dataset specification."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Get dataset spec
    dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
    if not dataset_spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {dataset_spec_id} not found",
        )
    project = db.query(Project).filter(Project.id == dataset_spec.project_id).first()
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    # Get all experiments using this dataset spec
    experiments_query = db.query(Experiment).filter(
        Experiment.dataset_spec_id == dataset_spec_id
    )

    if not include_iterations:
        # Only get root experiments (no parent)
        experiments_query = experiments_query.filter(Experiment.parent_experiment_id.is_(None))

    experiments = experiments_query.order_by(Experiment.created_at.desc()).all()

    # Build response for each experiment
    experiment_responses = []
    best_experiment_id = None
    best_score = None
    primary_metric = None

    for exp in experiments:
        # Get trial metrics
        trial = (
            db.query(Trial)
            .filter(Trial.experiment_id == exp.id)
            .order_by(Trial.created_at.desc())
            .first()
        )

        exp_best_score = None
        best_metrics = None
        training_time = None

        if trial and trial.metrics_json:
            best_metrics = _normalize_metrics(trial.metrics_json)
            metric_key = exp.primary_metric or "score_val"
            exp_best_score = best_metrics.get(metric_key, best_metrics.get("score_val"))
            training_time = best_metrics.get("training_time_seconds")

        # Get best model type
        best_model_type = None
        if exp.status == ExperimentStatus.COMPLETED:
            model = (
                db.query(ModelVersion)
                .filter(ModelVersion.experiment_id == exp.id)
                .order_by(ModelVersion.created_at.desc())
                .first()
            )
            if model:
                best_model_type = model.model_type

        # Get improvement summary for iterations
        improvement_summary = None
        if exp.improvement_context_json:
            raw_summary = exp.improvement_context_json.get("summary")
            if not raw_summary and exp.improvement_context_json.get("improvement_plan"):
                plan = exp.improvement_context_json["improvement_plan"]
                raw_summary = plan.get("plan_summary") or plan.get("iteration_description")
            if isinstance(raw_summary, list):
                improvement_summary = " ".join(str(item) for item in raw_summary)
            elif raw_summary:
                improvement_summary = str(raw_summary)

        experiment_responses.append(DatasetExperimentResponse(
            id=exp.id,
            name=exp.name,
            description=exp.description,
            status=exp.status.value if exp.status else "pending",
            primary_metric=exp.primary_metric,
            metric_direction=exp.metric_direction,
            best_score=exp_best_score,
            best_metrics=best_metrics,
            best_model_type=best_model_type,
            iteration_number=exp.iteration_number or 1,
            parent_experiment_id=exp.parent_experiment_id,
            improvement_summary=improvement_summary,
            created_at=exp.created_at,
            completed_at=exp.updated_at if exp.status == ExperimentStatus.COMPLETED else None,
            training_time_seconds=training_time,
        ))

        # Track best experiment
        if exp.status == ExperimentStatus.COMPLETED and exp_best_score is not None:
            if not primary_metric:
                primary_metric = exp.primary_metric
            direction = exp.metric_direction or "maximize"

            if best_score is None:
                best_score = exp_best_score
                best_experiment_id = exp.id
            elif direction == "maximize" and exp_best_score > best_score:
                best_score = exp_best_score
                best_experiment_id = exp.id
            elif direction == "minimize" and exp_best_score < best_score:
                best_score = exp_best_score
                best_experiment_id = exp.id

    # Count completed experiments
    completed_count = sum(1 for e in experiment_responses if e.status == "completed")

    return DatasetResultsResponse(
        dataset_spec_id=dataset_spec_id,
        dataset_name=dataset_spec.name,
        dataset_description=dataset_spec.description,
        target_column=dataset_spec.target_column,
        total_experiments=len(experiment_responses),
        completed_experiments=completed_count,
        best_experiment_id=best_experiment_id,
        best_score=best_score,
        primary_metric=primary_metric,
        experiments=experiment_responses,
    )


@router.get(
    "/dataset-specs/{dataset_spec_id}/download",
    summary="Download dataset as CSV",
    description="Build the dataset from spec and download as a CSV file.",
)
def download_dataset_spec(
    dataset_spec_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Download the built dataset as a CSV file."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
    if not dataset_spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset spec {dataset_spec_id} not found",
        )
    project = db.query(Project).filter(Project.id == dataset_spec.project_id).first()
    if not check_project_access(db, project, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this resource")

    try:
        # Build the dataset from sources
        builder = DatasetBuilder(db)
        df = builder.build_dataset_from_spec(dataset_spec_id)

        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Create safe filename from dataset name
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in dataset_spec.name)
        safe_name = safe_name.strip().replace(" ", "_")[:100]
        filename = f"{safe_name}.csv"

        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build dataset: {str(e)}",
        )

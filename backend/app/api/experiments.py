"""Experiment and trial API endpoints."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.celery_app import celery_app
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.project import Project
from app.models.experiment import Experiment, Trial, ExperimentStatus
from app.models.model_version import ModelVersion
from app.models.dataset_spec import DatasetSpec
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
    ExperimentRunRequest,
    ExperimentRunResponse,
    ExperimentDetailResponse,
    ExperimentProgressResponse,
    TrainingBackend,
    TrainingOptions,
    TrialCreate,
    TrialResponse,
    AutoIterateSettingsRequest,
    AutoIterateSettingsResponse,
)
from app.tasks import run_automl_experiment_task, run_experiment_modal


# Error metrics that AutoGluon stores as negative values (higher is better internally)
# These need to be converted to positive for display (lower is better for users)
ERROR_METRICS = {
    "root_mean_squared_error", "mean_squared_error", "mean_absolute_error",
    "rmse", "mse", "mae", "neg_root_mean_squared_error", "neg_mean_squared_error",
    "neg_mean_absolute_error"
}


def normalize_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize metrics for display by converting negative error metrics to positive.

    AutoGluon internally stores error metrics (RMSE, MSE, MAE) as negative values
    so that 'higher is better' works consistently across all metrics. When displaying
    to users, we convert these back to positive values since lower error is better.

    Args:
        metrics: Raw metrics dictionary from the database

    Returns:
        Normalized metrics dictionary with positive error values
    """
    if not metrics:
        return {}

    normalized = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Convert negative error metrics to positive for display
            if key.lower() in ERROR_METRICS and value < 0:
                normalized[key] = abs(value)
            else:
                normalized[key] = value
        else:
            normalized[key] = value

    return normalized

router = APIRouter(tags=["experiments"])


# Experiment endpoints


@router.post(
    "/projects/{project_id}/experiments",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_experiment(
    project_id: UUID,
    experiment: ExperimentCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new experiment for a project."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    db_experiment = Experiment(
        project_id=project_id,
        name=experiment.name,
        description=experiment.description,
        dataset_spec_id=experiment.dataset_spec_id,
        primary_metric=experiment.primary_metric,
        metric_direction=experiment.metric_direction,
        experiment_plan_json=experiment.experiment_plan_json,
    )
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    return db_experiment


@router.get(
    "/projects/{project_id}/experiments",
    response_model=list[ExperimentResponse],
)
def list_experiments(project_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """List all experiments for a project."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    return db.query(Experiment).filter(Experiment.project_id == project_id).all()


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentDetailResponse,
)
def get_experiment(experiment_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Get an experiment by ID with summary info."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Get trials for this experiment
    trials = db.query(Trial).filter(Trial.experiment_id == experiment_id).all()

    # Get best model version if experiment completed
    best_model = None
    best_metrics = None
    final_score = None
    val_score = None
    train_score = None
    has_holdout = False
    holdout_samples = None
    overfitting_gap = None
    score_source = "validation"

    if experiment.status == ExperimentStatus.COMPLETED:
        model = (
            db.query(ModelVersion)
            .filter(ModelVersion.experiment_id == experiment_id)
            .order_by(ModelVersion.created_at.desc())
            .first()
        )
        if model:
            best_model = {
                "id": str(model.id),
                "name": model.name,
                "model_type": model.model_type,
                "status": model.status.value,
            }
            # Normalize metrics to convert negative error metrics to positive for display
            best_metrics = normalize_metrics(model.metrics_json)

            # === Extract holdout-based scores (Make Holdout Score the Real Score) ===
            primary_metric = experiment.primary_metric or "score_val"
            raw_metrics = model.metrics_json or {}

            # Extract validation score (from CV/training evaluation)
            val_score = raw_metrics.get(primary_metric) or raw_metrics.get("score_val")
            if val_score is not None and isinstance(val_score, (int, float)):
                # Normalize if error metric
                if primary_metric.lower() in ERROR_METRICS and val_score < 0:
                    val_score = abs(val_score)

            # Extract training score
            train_key = f"train_{primary_metric}"
            train_score = raw_metrics.get(train_key)
            if train_score is None:
                # Try common train metric keys
                for key in ["train_accuracy", "train_roc_auc", "train_rmse", "train_r2"]:
                    if key in raw_metrics:
                        train_score = raw_metrics[key]
                        break
            if train_score is not None and isinstance(train_score, (int, float)):
                if primary_metric.lower() in ERROR_METRICS and train_score < 0:
                    train_score = abs(train_score)

            # Extract holdout score (CANONICAL FINAL SCORE)
            holdout_key = f"holdout_{primary_metric}"
            holdout_score = raw_metrics.get(holdout_key)
            if holdout_score is None:
                # Try alternate holdout keys
                for key in raw_metrics:
                    if key.startswith("holdout_") and (
                        primary_metric.lower() in key.lower() or
                        "auc" in key.lower() or
                        "accuracy" in key.lower() or
                        "rmse" in key.lower() or
                        "root_mean_squared_error" in key.lower() or  # RMSE full name
                        "mean_squared_error" in key.lower() or  # MSE
                        "r2" in key.lower()  # R-squared
                    ):
                        holdout_score = raw_metrics[key]
                        break

            # Also check improvement_context_json for holdout score
            if holdout_score is None and experiment.improvement_context_json:
                holdout_score = experiment.improvement_context_json.get("holdout_score")

            if holdout_score is not None and isinstance(holdout_score, (int, float)):
                # Normalize if error metric
                if primary_metric.lower() in ERROR_METRICS and holdout_score < 0:
                    holdout_score = abs(holdout_score)
                final_score = holdout_score
                has_holdout = True
                score_source = "holdout"
            else:
                # Fallback to validation score
                final_score = val_score
                score_source = "validation"

            # Get holdout sample count
            holdout_samples = raw_metrics.get("holdout_num_samples") or raw_metrics.get("holdout_size")

            # Calculate overfitting gap (how much better validation is than holdout)
            # Positive gap = overfitting (model performed better on validation than holdout)
            # Need to account for metric direction:
            # - "maximize" metrics (accuracy, AUC): val > holdout means overfitting
            # - "minimize" metrics (RMSE, MAE): val < holdout means overfitting
            if val_score is not None and final_score is not None and has_holdout:
                if experiment.metric_direction == "minimize":
                    # For error metrics: lower is better
                    # If val_score < holdout_score, validation did better = overfitting
                    overfitting_gap = final_score - val_score
                else:
                    # For maximize metrics: higher is better
                    # If val_score > holdout_score, validation did better = overfitting
                    overfitting_gap = val_score - final_score

    return ExperimentDetailResponse(
        id=experiment.id,
        project_id=experiment.project_id,
        name=experiment.name,
        description=experiment.description,
        dataset_spec_id=experiment.dataset_spec_id,
        primary_metric=experiment.primary_metric,
        metric_direction=experiment.metric_direction,
        experiment_plan_json=experiment.experiment_plan_json,
        status=experiment.status,
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        trial_count=len(trials),
        best_model=best_model,
        best_metrics=best_metrics,
        # Auto-improve iteration fields
        iteration_number=experiment.iteration_number,
        parent_experiment_id=experiment.parent_experiment_id,
        improvement_context_json=experiment.improvement_context_json,
        # Auto-iterate settings
        auto_iterate_enabled=bool(experiment.auto_iterate_enabled),
        auto_iterate_max=experiment.auto_iterate_max,
        # Holdout-based scoring (Make Holdout Score the Real Score)
        final_score=final_score,
        val_score=val_score,
        train_score=train_score,
        has_holdout=has_holdout,
        holdout_samples=holdout_samples,
        overfitting_gap=overfitting_gap,
        score_source=score_source,
    )


@router.put(
    "/experiments/{experiment_id}",
    response_model=ExperimentResponse,
)
def update_experiment(
    experiment_id: UUID,
    experiment_update: ExperimentUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update an experiment."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    update_data = experiment_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(experiment, field, value)

    db.commit()
    db.refresh(experiment)
    return experiment


@router.delete("/experiments/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(experiment_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Delete an experiment.

    If the experiment has a running or pending task, it will be cancelled first.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # If experiment has a task, revoke it before deleting
    if experiment.celery_task_id and experiment.status in [
        ExperimentStatus.PENDING,
        ExperimentStatus.RUNNING
    ]:
        _revoke_celery_task(experiment.celery_task_id)

    db.delete(experiment)
    db.commit()
    return None


class BulkDeleteRequest(BaseModel):
    """Request body for bulk delete operations."""
    experiment_ids: List[UUID]


class BulkDeleteResponse(BaseModel):
    """Response for bulk delete operations."""
    deleted_count: int
    failed_ids: List[str]
    errors: List[str]


@router.post("/experiments/bulk-delete", response_model=BulkDeleteResponse)
def bulk_delete_experiments(
    request: BulkDeleteRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Delete multiple experiments at once.

    Skips experiments that are currently running and returns info about failures.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    deleted_count = 0
    failed_ids = []
    errors = []

    for exp_id in request.experiment_ids:
        try:
            experiment = db.query(Experiment).filter(Experiment.id == exp_id).first()
            if not experiment:
                failed_ids.append(str(exp_id))
                errors.append(f"Experiment {exp_id} not found")
                continue

            # Skip running experiments
            if experiment.status == ExperimentStatus.RUNNING:
                failed_ids.append(str(exp_id))
                errors.append(f"Cannot delete running experiment {exp_id}")
                continue

            # Revoke any pending tasks
            if experiment.celery_task_id and experiment.status == ExperimentStatus.PENDING:
                _revoke_celery_task(experiment.celery_task_id)

            db.delete(experiment)
            deleted_count += 1

        except Exception as e:
            failed_ids.append(str(exp_id))
            errors.append(f"Error deleting {exp_id}: {str(e)}")

    db.commit()

    return BulkDeleteResponse(
        deleted_count=deleted_count,
        failed_ids=failed_ids,
        errors=errors,
    )


@router.put(
    "/experiments/{experiment_id}/auto-iterate",
    response_model=AutoIterateSettingsResponse,
    summary="Update auto-iterate settings",
    description="""Enable or disable automatic iteration after experiment completion.

When enabled, the system will automatically:
1. Run AI analysis on completed experiments
2. Generate improvement suggestions
3. Create and run a new iteration

**Settings:**
- `enabled`: Turn auto-iterate on/off
- `max_iterations`: Maximum number of iterations before stopping (1-20, default: 5)

**Note:** Auto-iterate only triggers when training completes successfully. Settings are propagated
to new iterations so the chain continues until max_iterations is reached.
""",
)
def update_auto_iterate_settings(
    experiment_id: UUID,
    request: AutoIterateSettingsRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update auto-iterate settings for an experiment."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Update settings
    experiment.auto_iterate_enabled = 1 if request.enabled else 0
    experiment.auto_iterate_max = request.max_iterations

    db.commit()
    db.refresh(experiment)

    can_continue = experiment.iteration_number < experiment.auto_iterate_max

    return AutoIterateSettingsResponse(
        experiment_id=experiment.id,
        auto_iterate_enabled=bool(experiment.auto_iterate_enabled),
        auto_iterate_max=experiment.auto_iterate_max,
        current_iteration=experiment.iteration_number,
        can_continue=can_continue,
        message=f"Auto-iterate {'enabled' if request.enabled else 'disabled'}. "
                f"Current iteration: {experiment.iteration_number}/{experiment.auto_iterate_max}"
    )


@router.get(
    "/experiments/{experiment_id}/auto-iterate",
    response_model=AutoIterateSettingsResponse,
    summary="Get auto-iterate settings",
)
def get_auto_iterate_settings(
    experiment_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get current auto-iterate settings for an experiment."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    can_continue = experiment.iteration_number < experiment.auto_iterate_max

    return AutoIterateSettingsResponse(
        experiment_id=experiment.id,
        auto_iterate_enabled=bool(experiment.auto_iterate_enabled),
        auto_iterate_max=experiment.auto_iterate_max,
        current_iteration=experiment.iteration_number,
        can_continue=can_continue,
    )


class ApplyFixRequest(BaseModel):
    """Request to apply a fix to an experiment based on a detected issue."""
    issue_type: str  # e.g., "split_strategy", "data_leakage", "overfitting", "class_imbalance"
    issue_description: str  # The warning/issue message shown to the user
    recommended_fix: Optional[str] = None  # Optional: specific fix recommendation


class ApplyFixResponse(BaseModel):
    """Response after applying a fix to an experiment."""
    experiment_id: UUID
    issue_type: str
    changes_applied: Dict[str, Any]
    message: str


async def _apply_fix_with_llm(
    experiment: Experiment,
    dataset_spec: Optional[DatasetSpec],
    issue_type: str,
    issue_description: str,
    recommended_fix: Optional[str] = None,
) -> Dict[str, Any]:
    """Use LLM to intelligently fix the experiment plan based on the detected issue."""
    import json
    import logging
    from app.services.llm_client import get_llm_client

    logger = logging.getLogger(__name__)

    # Get the current experiment plan
    current_plan = experiment.experiment_plan_json or {}

    # Build context about the dataset
    dataset_context = ""
    if dataset_spec:
        # Get task type from experiment plan if available
        task_type = "unknown"
        if current_plan:
            automl_config = current_plan.get("automl_config", {})
            task_type = automl_config.get("problem_type", "unknown")

        dataset_context = f"""
Dataset Information:
- Task Type: {task_type}
- Target Column: {dataset_spec.target_column or 'unknown'}
- Is Time-Based: {dataset_spec.is_time_based}
- Time Column: {dataset_spec.time_column or 'None'}
- Entity ID Column: {dataset_spec.entity_id_column or 'None'}
"""

    # Build the prompt for the LLM
    system_prompt = """You are an ML experiment design expert. Your task is to fix an experiment configuration based on a detected issue.

You will receive:
1. The current experiment plan (JSON)
2. Information about the detected issue
3. Dataset context

You must output ONLY a valid JSON object with the updated experiment plan. The JSON should have the same structure as the input but with the necessary fixes applied.

Important guidelines:
- For split_strategy issues on time-based data: Change split_strategy to "time" or "group_time" (if entity_id_column exists)
- For overfitting issues: Reduce model complexity, add regularization, use simpler presets like "good_quality"
- For class_imbalance issues: Enable sample weighting, consider using balanced_accuracy metric
- For data_leakage issues: Identify and exclude suspicious features from feature_columns

Always preserve existing valid configuration and only modify what's necessary to fix the issue."""

    user_prompt = f"""## Current Experiment Plan
```json
{json.dumps(current_plan, indent=2, default=str)}
```

## Detected Issue
**Issue Type:** {issue_type}
**Description:** {issue_description}
{f"**Recommended Fix:** {recommended_fix}" if recommended_fix else ""}

{dataset_context}

## Your Task
Update the experiment plan to fix the detected issue. Return ONLY the updated JSON object (no markdown, no explanation, just the JSON).
"""

    try:
        llm_client = get_llm_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"Calling LLM to fix experiment plan for issue: {issue_type}")
        response = await llm_client.chat(messages)

        # Parse the response as JSON
        response_text = response.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last line (code block markers)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)

        updated_plan = json.loads(response_text)

        # Calculate what changed
        changes = {
            "issue_type": issue_type,
            "llm_applied": True,
        }

        # Track key changes
        old_validation = current_plan.get("validation_strategy", {})
        new_validation = updated_plan.get("validation_strategy", {})
        if old_validation.get("split_strategy") != new_validation.get("split_strategy"):
            changes["split_strategy"] = {
                "previous": old_validation.get("split_strategy"),
                "new": new_validation.get("split_strategy"),
            }

        old_automl = current_plan.get("automl_config", {})
        new_automl = updated_plan.get("automl_config", {})
        if old_automl.get("presets") != new_automl.get("presets"):
            changes["presets"] = {
                "previous": old_automl.get("presets"),
                "new": new_automl.get("presets"),
            }

        # Update the experiment
        experiment.experiment_plan_json = updated_plan
        logger.info(f"Successfully updated experiment plan via LLM for issue: {issue_type}")

        return changes

    except Exception as e:
        logger.error(f"LLM fix failed: {e}, falling back to rule-based fix")
        # Fall back to simple rule-based fixes
        return _apply_rule_based_fix(experiment, dataset_spec, issue_type, issue_description)


def _apply_rule_based_fix(
    experiment: Experiment,
    dataset_spec: Optional[DatasetSpec],
    issue_type: str,
    issue_description: str,
) -> Dict[str, Any]:
    """Fallback rule-based fix if LLM fails."""
    plan = experiment.experiment_plan_json or {}

    if issue_type == "split_strategy" and dataset_spec and dataset_spec.is_time_based:
        new_strategy = "group_time" if dataset_spec.entity_id_column else "time"
        current_validation = plan.get("validation_strategy", {})
        previous_strategy = current_validation.get("split_strategy", "unknown")

        plan["validation_strategy"] = {
            **current_validation,
            "split_strategy": new_strategy,
            "time_column": dataset_spec.time_column,
            "reasoning": f"Updated to {new_strategy} split to prevent data leakage",
        }
        if dataset_spec.entity_id_column:
            plan["validation_strategy"]["entity_id_column"] = dataset_spec.entity_id_column

        experiment.experiment_plan_json = plan
        return {"split_strategy": {"previous": previous_strategy, "new": new_strategy}}

    elif issue_type == "overfitting":
        automl_config = plan.get("automl_config", {})
        previous_presets = automl_config.get("presets", "best_quality")
        automl_config["presets"] = "good_quality"
        plan["automl_config"] = automl_config
        experiment.experiment_plan_json = plan
        return {"presets": {"previous": previous_presets, "new": "good_quality"}}

    elif issue_type == "class_imbalance":
        automl_config = plan.get("automl_config", {})
        automl_config["sample_weight"] = "balance"
        plan["automl_config"] = automl_config
        experiment.experiment_plan_json = plan
        return {"sample_weight": "balance"}

    elif issue_type == "data_leakage":
        plan["leakage_warning_applied"] = True
        experiment.experiment_plan_json = plan
        return {"leakage_warning_applied": True}

    return {"fallback": True}


@router.post(
    "/experiments/{experiment_id}/apply-fix",
    response_model=ApplyFixResponse,
    summary="Apply a fix based on detected issue",
    description="""Apply an automated fix to the experiment based on a detected warning or issue.

This endpoint uses an LLM agent to intelligently analyze the issue and update the experiment configuration.

**Supported Issue Types:**
- `split_strategy`: Fix split strategy for time-based data (changes to time/group_time split)
- `overfitting`: Apply regularization and simplify model configuration
- `class_imbalance`: Enable sample weighting and use balanced metrics
- `data_leakage`: Flag suspicious features for exclusion

The endpoint sends the issue to the experiment design agent which determines the appropriate fix.
After applying the fix, re-run the experiment to see the changes take effect.
""",
)
async def apply_fix(
    experiment_id: UUID,
    request: ApplyFixRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Apply an automated fix to the experiment using the LLM agent."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Get linked dataset spec
    dataset_spec = None
    if experiment.dataset_spec_id:
        dataset_spec = db.query(DatasetSpec).filter(
            DatasetSpec.id == experiment.dataset_spec_id
        ).first()

    # Apply the fix using LLM
    issue_type = request.issue_type.lower()
    changes = await _apply_fix_with_llm(
        experiment=experiment,
        dataset_spec=dataset_spec,
        issue_type=issue_type,
        issue_description=request.issue_description,
        recommended_fix=request.recommended_fix,
    )

    db.commit()
    db.refresh(experiment)

    # Generate message based on what changed
    if "split_strategy" in changes:
        message = f"Split strategy updated to '{changes['split_strategy']['new']}' by AI agent. Re-run the experiment to apply."
    elif "presets" in changes:
        message = f"Model configuration updated to '{changes['presets']['new']}' with more regularization. Re-run to apply."
    elif "sample_weight" in changes:
        message = "Sample weighting enabled for class imbalance. Re-run the experiment to apply."
    elif "leakage_warning_applied" in changes:
        message = "Data leakage warning applied. Review and re-run the experiment."
    else:
        message = f"Experiment plan updated by AI agent for {issue_type} issue. Re-run to apply changes."

    return ApplyFixResponse(
        experiment_id=experiment.id,
        issue_type=request.issue_type,
        changes_applied=changes,
        message=message,
    )


@router.post(
    "/experiments/{experiment_id}/run",
    response_model=ExperimentRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run an experiment",
    description="""Queue an experiment for background execution using AutoML.

**Prerequisites:**
- Experiment must have a `dataset_spec_id` configured
- Experiment status must be `pending` or `failed`
- Celery worker must be running to process the task

**Training Options:**
- `backend`: "local" (Celery worker) or "modal" (Modal.com cloud)
- `resource_limits_enabled`: Enable CPU/memory limits for local training (default: true)
- `num_cpus`, `num_gpus`, `memory_limit_gb`: Override resource limits

**Runner Selection:**
The runner is automatically selected based on the project's `task_type`:
- `binary`, `multiclass`, `regression`, `quantile` → TabularRunner (autogluon.tabular)
- `timeseries_forecast` → TimeSeriesRunner (autogluon.timeseries)
- `multimodal_classification`, `multimodal_regression` → MultiModalRunner (autogluon.multimodal)

**Time Series Configuration:**
For `timeseries_forecast`, set `experiment_plan_json` with required config:
```json
{"automl_config": {"prediction_length": 7, "time_column": "date"}}
```

**Returns:**
- `task_id`: Celery task ID for tracking
- `status`: "queued"
- `backend`: "local" or "modal"
""",
)
def run_experiment(
    experiment_id: UUID,
    request: ExperimentRunRequest = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Start running an experiment (asynchronous)."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Check if experiment is in a runnable state
    if experiment.status not in [ExperimentStatus.PENDING, ExperimentStatus.FAILED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Experiment is {experiment.status.value}, cannot run. "
            f"Only pending or failed experiments can be run.",
        )

    # Validate experiment has dataset_spec
    if not experiment.dataset_spec_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Experiment has no dataset_spec_id configured",
        )

    # Get training options (use defaults if not provided)
    training_options = request.training_options if request else None
    if training_options is None:
        training_options = TrainingOptions()

    # Reset experiment status immediately so UI shows correct state
    experiment.status = ExperimentStatus.PENDING
    experiment.error_message = None  # Clear any previous error
    db.commit()

    # Handle Modal backend
    if training_options.backend == TrainingBackend.MODAL:
        from app.services.modal_runner import is_modal_configured, get_modal_status

        if not is_modal_configured():
            status_info = get_modal_status()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Modal is not configured. Status: {status_info}. "
                f"Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env",
            )

        # Queue the Modal training task via Celery
        task = run_experiment_modal.delay(str(experiment_id))

        # Store the celery task ID for tracking and cancellation
        experiment.celery_task_id = task.id
        db.commit()

        return ExperimentRunResponse(
            experiment_id=experiment_id,
            status="queued",
            task_id=task.id,
            message=f"Experiment queued for Modal cloud execution. Task ID: {task.id}",
            backend="modal",
        )

    # Queue the local experiment task with resource options
    task = run_automl_experiment_task.delay(
        str(experiment_id),
        resource_limits_enabled=training_options.resource_limits_enabled,
        num_cpus=training_options.num_cpus,
        num_gpus=training_options.num_gpus,
        memory_limit_gb=training_options.memory_limit_gb,
    )

    # Store the celery task ID for tracking and cancellation
    experiment.celery_task_id = task.id
    db.commit()

    limits_msg = ""
    if training_options.resource_limits_enabled:
        limits_msg = " with resource limits"
    else:
        limits_msg = " without resource limits (full power)"

    return ExperimentRunResponse(
        experiment_id=experiment_id,
        status="queued",
        task_id=task.id,
        message=f"Experiment queued for local execution{limits_msg}. Task ID: {task.id}",
        backend="local",
    )


@router.get(
    "/training-options",
    summary="Get available training options",
    description="Get the current training configuration and available backends.",
)
def get_training_options(current_user: Optional[User] = Depends(get_current_user)):
    """Get available training options and their current configuration."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.core.config import get_settings
    from app.services.modal_runner import get_modal_status

    settings = get_settings()
    modal_status = get_modal_status()

    return {
        "backends": {
            "local": {
                "available": True,
                "description": "Run on local Celery worker",
            },
            "modal": {
                "available": modal_status["configured"],
                "description": "Run on Modal.com cloud (faster, no limits)",
                "status": modal_status,
            },
        },
        "resource_limits": {
            "enabled_by_default": settings.resource_limits_enabled,
            "defaults": {
                "num_cpus": settings.autogluon_num_cpus,
                "num_gpus": settings.autogluon_num_gpus,
                "memory_limit_gb": settings.max_memory_gb,
            },
            "description": "Resource limits help prevent system freezes during training",
        },
        "automl_defaults": {
            "time_limit": settings.automl_time_limit,
            "presets": settings.automl_presets,
        },
    }


def _revoke_celery_task(task_id: str) -> bool:
    """Revoke a Celery task and remove it from the queue.

    Args:
        task_id: The Celery task ID to revoke

    Returns:
        True if task was revoked, False otherwise
    """
    if not task_id:
        return False

    try:
        # Terminate the task forcefully (SIGTERM) if it's running
        celery_app.control.revoke(
            task_id,
            terminate=True,
            signal="SIGTERM"
        )

        # Also try to remove from result backend to clean up
        from celery.result import AsyncResult
        result = AsyncResult(task_id, app=celery_app)
        result.forget()

        return True
    except Exception:
        # Task may have already completed or not exist
        return False


@router.post(
    "/experiments/{experiment_id}/cancel",
    response_model=ExperimentRunResponse,
)
def cancel_experiment(experiment_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Cancel a running or pending experiment.

    This will:
    1. Revoke the Celery task (stops it if running, removes from queue if pending)
    2. Update experiment status to CANCELLED
    3. For Modal tasks, the next poll will detect cancellation and abort
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    if experiment.status not in [ExperimentStatus.PENDING, ExperimentStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Experiment is {experiment.status.value}, cannot cancel",
        )

    # Revoke the Celery task if it exists
    task_revoked = _revoke_celery_task(experiment.celery_task_id)

    # Update status and clear the task ID
    experiment.status = ExperimentStatus.CANCELLED
    old_task_id = experiment.celery_task_id
    experiment.celery_task_id = None
    db.commit()

    message = "Experiment cancelled"
    if task_revoked:
        message += f" and task {old_task_id} terminated"

    return ExperimentRunResponse(
        experiment_id=experiment_id,
        status="cancelled",
        message=message,
    )


from pydantic import BaseModel, Field
from typing import List, Optional


class BatchRunExperimentsRequest(BaseModel):
    """Request to run multiple experiments in parallel."""
    experiment_ids: List[UUID] = Field(
        ...,
        description="List of experiment IDs to run in parallel"
    )


class BatchExperimentStatus(BaseModel):
    """Status of a single experiment in a batch."""
    experiment_id: UUID
    status: str
    task_id: Optional[str] = None
    message: str


class BatchRunExperimentsResponse(BaseModel):
    """Response from batch running experiments."""
    experiments: List[BatchExperimentStatus]
    message: str
    queued_count: int
    failed_count: int


@router.post(
    "/experiments/run-batch",
    response_model=BatchRunExperimentsResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run multiple experiments in parallel",
    description="""Queue multiple experiments for parallel execution using AutoML.

**Prerequisites for each experiment:**
- Experiment must have a `dataset_spec_id` configured
- Experiment status must be `pending` or `failed`
- Celery worker must be running to process the tasks

All valid experiments will be queued immediately to run in parallel.
Experiments that fail validation will be reported but won't stop others from running.

**Returns:**
- List of experiment statuses with their task IDs
- Summary counts of queued vs failed experiments
""",
)
def run_experiments_batch(
    request: BatchRunExperimentsRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Start running multiple experiments in parallel (asynchronous)."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    results = []
    queued_count = 0
    failed_count = 0

    for experiment_id in request.experiment_ids:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

        if not experiment:
            results.append(BatchExperimentStatus(
                experiment_id=experiment_id,
                status="error",
                message=f"Experiment not found",
            ))
            failed_count += 1
            continue

        # Check if experiment is in a runnable state
        if experiment.status not in [ExperimentStatus.PENDING, ExperimentStatus.FAILED]:
            results.append(BatchExperimentStatus(
                experiment_id=experiment_id,
                status="error",
                message=f"Experiment is {experiment.status.value}, cannot run",
            ))
            failed_count += 1
            continue

        # Validate experiment has dataset_spec
        if not experiment.dataset_spec_id:
            results.append(BatchExperimentStatus(
                experiment_id=experiment_id,
                status="error",
                message="Experiment has no dataset_spec_id configured",
            ))
            failed_count += 1
            continue

        # Queue the experiment task (always use Modal)
        try:
            task = run_experiment_modal.delay(str(experiment_id))

            # Store the celery task ID for tracking and cancellation
            experiment.celery_task_id = task.id

            results.append(BatchExperimentStatus(
                experiment_id=experiment_id,
                status="queued",
                task_id=task.id,
                message="Experiment queued for execution",
            ))
            queued_count += 1
        except Exception as e:
            results.append(BatchExperimentStatus(
                experiment_id=experiment_id,
                status="error",
                message=f"Failed to queue: {str(e)}",
            ))
            failed_count += 1

    # Commit all task ID updates
    db.commit()

    return BatchRunExperimentsResponse(
        experiments=results,
        message=f"Queued {queued_count} experiments for parallel execution, {failed_count} failed",
        queued_count=queued_count,
        failed_count=failed_count,
    )


@router.get(
    "/experiments/{experiment_id}/progress",
    response_model=ExperimentProgressResponse,
)
def get_experiment_progress(experiment_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Get the progress of a running experiment.

    Returns real-time progress information from the Celery task.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Default response for non-running states
    if experiment.status == ExperimentStatus.PENDING:
        return ExperimentProgressResponse(
            experiment_id=experiment_id,
            status="pending",
            progress=0,
            message="Waiting to start...",
        )

    if experiment.status == ExperimentStatus.COMPLETED:
        return ExperimentProgressResponse(
            experiment_id=experiment_id,
            status="completed",
            progress=100,
            message="Experiment completed successfully",
        )

    if experiment.status == ExperimentStatus.FAILED:
        return ExperimentProgressResponse(
            experiment_id=experiment_id,
            status="failed",
            progress=0,
            message=experiment.error_message or "Experiment failed",
        )

    if experiment.status == ExperimentStatus.CANCELLED:
        return ExperimentProgressResponse(
            experiment_id=experiment_id,
            status="cancelled",
            progress=0,
            message="Experiment was cancelled",
        )

    # For running experiments, try to get progress from Celery
    if experiment.celery_task_id:
        try:
            result = AsyncResult(experiment.celery_task_id, app=celery_app)
            if result.state == "RUNNING":
                meta = result.info or {}
                return ExperimentProgressResponse(
                    experiment_id=experiment_id,
                    status="running",
                    progress=meta.get("progress", 0),
                    message=meta.get("message", "Training in progress..."),
                    stage=meta.get("stage"),
                )
            elif result.state == "PENDING":
                return ExperimentProgressResponse(
                    experiment_id=experiment_id,
                    status="queued",
                    progress=0,
                    message="Task queued, waiting for worker...",
                )
        except Exception:
            pass

    # Fallback for running without task info
    return ExperimentProgressResponse(
        experiment_id=experiment_id,
        status="running",
        progress=0,
        message="Training in progress...",
    )


# Trial endpoints


@router.post(
    "/experiments/{experiment_id}/trials",
    response_model=TrialResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_trial(
    experiment_id: UUID,
    trial: TrialCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new trial for an experiment."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    db_trial = Trial(
        experiment_id=experiment_id,
        variant_name=trial.variant_name,
        data_split_strategy=trial.data_split_strategy,
        automl_config=trial.automl_config,
    )
    db.add(db_trial)
    db.commit()
    db.refresh(db_trial)
    return db_trial


@router.get(
    "/experiments/{experiment_id}/trials",
    response_model=list[TrialResponse],
)
def list_trials(experiment_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """List all trials for an experiment."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Verify experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    return db.query(Trial).filter(Trial.experiment_id == experiment_id).all()


@router.get(
    "/trials/{trial_id}",
    response_model=TrialResponse,
)
def get_trial(trial_id: UUID, db: Session = Depends(get_db), current_user: Optional[User] = Depends(get_current_user)):
    """Get a trial by ID."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    trial = db.query(Trial).filter(Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trial {trial_id} not found",
        )
    return trial


# Training Critique endpoints


@router.post(
    "/experiments/{experiment_id}/critique",
    response_model=dict,
)
async def generate_training_critique(
    experiment_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Generate AI critique of training results with improvement suggestions.

    Analyzes training logs, leaderboard, and metrics to provide actionable
    feedback for improving model performance.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.services.prompts import get_training_critique_prompt
    from app.services.llm_client import get_llm_client
    from app.schemas.agent import TrainingCritiqueResult

    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    if experiment.status != ExperimentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Experiment must be completed to generate critique. Current status: {experiment.status.value}",
        )

    # Get the latest trial
    trial = (
        db.query(Trial)
        .filter(Trial.experiment_id == experiment_id)
        .order_by(Trial.created_at.desc())
        .first()
    )
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trials found for this experiment",
        )

    # Get model version for feature importances
    model_version = (
        db.query(ModelVersion)
        .filter(ModelVersion.experiment_id == experiment_id)
        .order_by(ModelVersion.created_at.desc())
        .first()
    )

    # Build leaderboard summary
    leaderboard = trial.leaderboard_json or []
    leaderboard_lines = []
    for model in leaderboard[:10]:  # Top 10 models
        name = model.get("model", "Unknown")
        score = model.get("score_val", 0)
        fit_time = model.get("fit_time", 0)
        leaderboard_lines.append(f"  - {name}: score={score:.4f}, fit_time={fit_time:.1f}s")
    leaderboard_summary = "\n".join(leaderboard_lines) if leaderboard_lines else "(no leaderboard data)"

    # Get dataset info - try to load actual data if not in spec
    dataset_spec = experiment.dataset_spec
    feature_columns = dataset_spec.feature_columns if dataset_spec else []
    target_column = dataset_spec.target_column if dataset_spec else experiment.experiment_plan_json.get("target_column", "unknown")
    row_count = 0

    # First try to get from spec_json
    if dataset_spec and dataset_spec.spec_json:
        row_count = dataset_spec.spec_json.get("row_count", 0)

    # If row_count is 0 or feature_columns is empty, try loading actual dataset
    if (row_count == 0 or not feature_columns) and dataset_spec:
        try:
            from app.tasks.automl import load_dataset_from_spec
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Loading actual dataset for critique context...")
            df = load_dataset_from_spec(db, dataset_spec)
            row_count = len(df)
            if not feature_columns:
                feature_columns = [c for c in df.columns if c != target_column]
            logger.info(f"Loaded dataset: {row_count} rows, {len(feature_columns)} features")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not load dataset for critique context: {e}")

    dataset_shape = f"{row_count} rows x {len(feature_columns)} features"

    # Get metrics
    metrics = trial.metrics_json or {}
    primary_metric = experiment.primary_metric or "score_val"
    best_score = metrics.get(primary_metric, metrics.get("score_val", 0))

    # Get feature importances
    feature_importances = {}
    if model_version and model_version.feature_importances_json:
        feature_importances = model_version.feature_importances_json

    # Get training logs
    training_logs = trial.training_logs or "(no logs captured)"

    # Get task type
    task_type = "binary"
    if experiment.experiment_plan_json:
        task_type = experiment.experiment_plan_json.get("task_type", "binary")

    # Get training time
    training_time = metrics.get("training_time_seconds", 0)
    num_models = metrics.get("num_models_trained", len(leaderboard))

    # Generate prompt
    prompt = get_training_critique_prompt(
        experiment_name=experiment.name,
        task_type=task_type,
        target_column=target_column,
        primary_metric=primary_metric,
        best_score=float(best_score) if best_score else 0.0,
        training_time_seconds=float(training_time) if training_time else 0.0,
        num_models_trained=int(num_models) if num_models else 0,
        dataset_shape=dataset_shape,
        feature_columns=feature_columns or [],
        leaderboard_summary=leaderboard_summary,
        training_logs=training_logs,
        feature_importances=feature_importances,
    )

    # Call LLM
    llm_client = get_llm_client()
    messages = [
        {"role": "system", "content": "You are an expert ML engineer providing actionable feedback on AutoML training results. Always respond with valid JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = await llm_client.chat_json(messages, TrainingCritiqueResult)

        # Validate and fix result
        if isinstance(result, dict):
            # Ensure all required fields
            critique = TrainingCritiqueResult(**result)
        else:
            critique = result

        critique_dict = critique.model_dump()

        # Save critique to trial
        trial.critique_json = critique_dict
        db.commit()

        return {
            "experiment_id": str(experiment_id),
            "trial_id": str(trial.id),
            "critique": critique_dict,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate critique: {str(e)}",
        )


@router.get(
    "/experiments/{experiment_id}/critique",
    response_model=dict,
)
def get_training_critique(
    experiment_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get the cached training critique for an experiment.

    Returns the previously generated critique if available.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Get the latest trial with critique
    trial = (
        db.query(Trial)
        .filter(Trial.experiment_id == experiment_id)
        .filter(Trial.critique_json.isnot(None))
        .order_by(Trial.created_at.desc())
        .first()
    )

    if not trial or not trial.critique_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No critique found for this experiment. Use POST to generate one.",
        )

    return {
        "experiment_id": str(experiment_id),
        "trial_id": str(trial.id),
        "critique": trial.critique_json,
    }


# =============================================================================
# Auto-Improve Pipeline endpoints
# =============================================================================


class ImproveExperimentResponse(BaseModel):
    """Response from triggering improvement pipeline."""
    experiment_id: UUID
    task_id: str
    message: str
    status: str = "queued"


class ExperimentIterationResponse(BaseModel):
    """Response for an experiment iteration."""
    id: UUID
    name: str
    iteration_number: int
    status: str
    best_score: Optional[float] = None  # DEPRECATED: Use final_score instead
    primary_metric: Optional[str] = None
    improvement_summary: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None  # Full metrics for comparison
    created_at: datetime
    # Holdout-based scoring (Make Holdout Score the Real Score)
    final_score: Optional[float] = None  # Canonical score from holdout evaluation
    val_score: Optional[float] = None  # Validation/CV score (NOT the final score)
    has_holdout: bool = False  # Whether holdout evaluation was performed
    overfitting_gap: Optional[float] = None  # Gap between val and holdout (positive = overfitting)
    score_source: str = "validation"  # 'holdout' (preferred) or 'validation' (fallback)


class ExperimentIterationsResponse(BaseModel):
    """Response for experiment iterations list."""
    root_experiment_id: UUID
    total_iterations: int
    iterations: List[ExperimentIterationResponse]


@router.post(
    "/experiments/{experiment_id}/improve",
    response_model=ImproveExperimentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger auto-improvement pipeline",
    description="""Trigger the auto-improvement pipeline for a completed experiment.

This will:
1. Analyze the training results and critique
2. Generate an improvement plan
3. Create a new DatasetSpec with improvements
4. Create a new Experiment (iteration N+1)
5. Automatically queue the new experiment for training

**Prerequisites:**
- Experiment must be in `completed` status
- Experiment must have at least one completed trial

**Returns:**
- The new experiment ID that was created
- Task ID for tracking the improvement pipeline
""",
)
async def trigger_improvement(
    experiment_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Trigger auto-improvement pipeline for a completed experiment."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.tasks.experiment_tasks import run_auto_improve_pipeline

    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Validate experiment is completed
    if experiment.status != ExperimentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Experiment must be completed to improve. Current status: {experiment.status.value}",
        )

    # Check for existing trials
    trial = (
        db.query(Trial)
        .filter(Trial.experiment_id == experiment_id)
        .first()
    )
    if not trial:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No trials found for this experiment",
        )

    # Queue the auto-improve pipeline
    task = run_auto_improve_pipeline.delay(str(experiment_id))

    return ImproveExperimentResponse(
        experiment_id=experiment_id,
        task_id=task.id,
        message=f"Improvement pipeline queued. Task ID: {task.id}",
        status="queued",
    )


@router.get(
    "/experiments/{experiment_id}/iterations",
    response_model=ExperimentIterationsResponse,
    summary="Get all iterations of an experiment",
    description="""Get all iterations in an experiment's improvement chain.

Returns the root experiment and all its descendants, with comparison data
for metrics progression across iterations.

**Navigation:**
- Use `parent_experiment_id` to find the parent
- Use `child_experiments` to find children (next iterations)
""",
)
def get_experiment_iterations(
    experiment_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get all iterations of an experiment (parent + children chain)."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    # Get the requested experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Find the root experiment (no parent)
    root = experiment
    while root.parent_experiment_id:
        parent = db.query(Experiment).filter(Experiment.id == root.parent_experiment_id).first()
        if not parent:
            break
        root = parent

    # Collect all iterations starting from root
    iterations = []

    def collect_iterations(exp: Experiment):
        # Get trial metrics (for validation/CV scores)
        trial = (
            db.query(Trial)
            .filter(Trial.experiment_id == exp.id)
            .order_by(Trial.created_at.desc())
            .first()
        )

        # Get ModelVersion for holdout scores (Make Holdout Score the Real Score)
        model = (
            db.query(ModelVersion)
            .filter(ModelVersion.experiment_id == exp.id)
            .order_by(ModelVersion.created_at.desc())
            .first()
        )

        # Initialize score variables
        best_score = None
        final_score = None
        val_score = None
        has_holdout = False
        overfitting_gap = None
        score_source = "validation"
        metrics = None
        primary_metric = exp.primary_metric or "score_val"

        # Get validation score from trial metrics (CV/validation scores)
        if trial and trial.metrics_json:
            metrics = normalize_metrics(trial.metrics_json)
            val_score = metrics.get(primary_metric, metrics.get("score_val"))

        # Get holdout score from ModelVersion if available
        if model and model.metrics_json:
            raw_metrics = model.metrics_json
            # Merge model metrics into metrics dict for display
            if metrics is None:
                metrics = normalize_metrics(raw_metrics)
            else:
                metrics.update(normalize_metrics(raw_metrics))

            # Extract holdout score (CANONICAL FINAL SCORE)
            holdout_key = f"holdout_{primary_metric}"
            holdout_score = raw_metrics.get(holdout_key)
            if holdout_score is None:
                # Try alternate holdout keys
                for key in raw_metrics:
                    if key.startswith("holdout_") and (
                        primary_metric.lower() in key.lower() or
                        "auc" in key.lower() or
                        "accuracy" in key.lower() or
                        "rmse" in key.lower() or
                        "root_mean_squared_error" in key.lower() or  # RMSE full name
                        "mean_squared_error" in key.lower() or  # MSE
                        "r2" in key.lower()  # R-squared
                    ):
                        holdout_score = raw_metrics[key]
                        break

            # Also check improvement_context_json for holdout score
            if holdout_score is None and exp.improvement_context_json:
                holdout_score = exp.improvement_context_json.get("holdout_score")

            if holdout_score is not None and isinstance(holdout_score, (int, float)):
                # Normalize if error metric
                if primary_metric.lower() in ERROR_METRICS and holdout_score < 0:
                    holdout_score = abs(holdout_score)
                final_score = holdout_score
                has_holdout = True
                score_source = "holdout"

        # Fallback to validation score if no holdout
        if final_score is None:
            final_score = val_score

        # Calculate overfitting gap (positive = overfitting)
        # Account for metric direction: minimize vs maximize
        if val_score is not None and final_score is not None and has_holdout:
            if exp.metric_direction == "minimize":
                # For error metrics: lower is better
                # If val_score < holdout_score, validation did better = overfitting
                overfitting_gap = final_score - val_score
            else:
                # For maximize metrics: higher is better
                # If val_score > holdout_score, validation did better = overfitting
                overfitting_gap = val_score - final_score

        # best_score for backward compatibility (deprecated, use final_score)
        best_score = final_score

        improvement_summary = None
        if exp.improvement_context_json:
            # Get summary from improvement context
            raw_summary = exp.improvement_context_json.get("summary")
            # If no summary, try to build one from improvement_plan
            if not raw_summary and exp.improvement_context_json.get("improvement_plan"):
                plan = exp.improvement_context_json["improvement_plan"]
                raw_summary = plan.get("plan_summary") or plan.get("iteration_description")

            # Handle case where summary is a list (convert to string)
            if isinstance(raw_summary, list):
                improvement_summary = " ".join(str(item) for item in raw_summary)
            elif raw_summary:
                improvement_summary = str(raw_summary)

        iterations.append(ExperimentIterationResponse(
            id=exp.id,
            name=exp.name,
            iteration_number=exp.iteration_number or 1,
            status=exp.status.value if exp.status else "pending",
            best_score=best_score,  # Deprecated, use final_score
            primary_metric=exp.primary_metric,
            improvement_summary=improvement_summary,
            metrics=metrics,
            created_at=exp.created_at,
            # Holdout-based scoring (Make Holdout Score the Real Score)
            final_score=final_score,
            val_score=val_score,
            has_holdout=has_holdout,
            overfitting_gap=overfitting_gap,
            score_source=score_source,
        ))

        # Get children
        children = (
            db.query(Experiment)
            .filter(Experiment.parent_experiment_id == exp.id)
            .order_by(Experiment.iteration_number)
            .all()
        )
        for child in children:
            collect_iterations(child)

    collect_iterations(root)

    # Sort by iteration number
    iterations.sort(key=lambda x: x.iteration_number)

    return ExperimentIterationsResponse(
        root_experiment_id=root.id,
        total_iterations=len(iterations),
        iterations=iterations,
    )


@router.get(
    "/experiments/{experiment_id}/improvement-status",
    summary="Get improvement pipeline status",
    description="Get the status of an ongoing improvement pipeline.",
)
def get_improvement_status(
    experiment_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get the status of the improvement pipeline for an experiment."""
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.models.agent_run import AgentRun

    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Find the latest improvement agent run for this experiment
    agent_run = (
        db.query(AgentRun)
        .filter(AgentRun.experiment_id == experiment_id)
        .filter(AgentRun.name.like("Improve%"))
        .order_by(AgentRun.created_at.desc())
        .first()
    )

    if not agent_run:
        return {
            "experiment_id": str(experiment_id),
            "has_improvement_run": False,
            "message": "No improvement pipeline has been run for this experiment",
        }

    # Get steps
    steps = []
    for step in agent_run.steps:
        steps.append({
            "step_type": step.step_type.value,
            "status": step.status.value,
            "started_at": step.started_at.isoformat() if step.started_at else None,
            "finished_at": step.finished_at.isoformat() if step.finished_at else None,
            "error_message": step.error_message,
        })

    # Get result if completed
    result = None
    if agent_run.result_json:
        result = agent_run.result_json

    return {
        "experiment_id": str(experiment_id),
        "has_improvement_run": True,
        "agent_run_id": str(agent_run.id),
        "status": agent_run.status.value,
        "steps": steps,
        "result": result,
        "error_message": agent_run.error_message,
    }


@router.get(
    "/experiments/{experiment_id}/overfitting-report",
    summary="Get overfitting analysis report",
    description="Get a detailed overfitting analysis for an experiment and all its iterations.",
)
def get_overfitting_analysis(
    experiment_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get overfitting analysis for an experiment chain.

    This endpoint returns:
    - Holdout validation scores for each iteration
    - Overfitting risk percentage for each iteration (0-100)
    - Overall trend (improving, stable, degrading)
    - Recommendation (continue, warning, stop)
    - Best iteration to use if overfitting is detected
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.services.holdout_validator import get_overfitting_report

    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Get overfitting report
    report = get_overfitting_report(db, experiment)

    return {
        "experiment_id": str(experiment_id),
        "iteration_number": experiment.iteration_number,
        **report,
    }


@router.get(
    "/projects/{project_id}/experiments/{experiment_id}/iterations/overfitting",
    summary="Get overfitting status for all iterations",
    description="Get overfitting analysis for an entire experiment iteration chain.",
)
def get_iterations_overfitting(
    project_id: UUID,
    experiment_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get overfitting status for all iterations in a chain.

    This provides a summary view suitable for displaying in a UI,
    showing the holdout score and risk level for each iteration.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from app.services.holdout_validator import get_overfitting_report

    # Verify project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Get experiment
    experiment = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.project_id == project_id,
    ).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found in project {project_id}",
        )

    # Get overfitting report
    report = get_overfitting_report(db, experiment)

    # Build summary for each iteration
    iterations_summary = []
    for entry in report.get("score_history", []):
        iterations_summary.append({
            "experiment_id": entry.get("experiment_id"),
            "iteration": entry.get("iteration"),
            "holdout_score": entry.get("score"),
            "metric": entry.get("metric"),
            "overfitting_risk": entry.get("overfitting_risk", 0),
            "status": entry.get("status", "unknown"),
            "is_best": entry.get("is_best", False),
        })

    return {
        "project_id": str(project_id),
        "experiment_id": str(experiment_id),
        "total_iterations": len(iterations_summary),
        "overall_risk": report.get("overall_risk", 0),
        "risk_level": report.get("risk_level", "low"),
        "trend": report.get("trend", "unknown"),
        "best_iteration": report.get("best_iteration"),
        "best_score": report.get("best_score"),
        "current_score": report.get("current_score"),
        "recommendation": report.get("recommendation"),
        "message": report.get("message"),
        "iterations": iterations_summary,
    }


# ============================================================================
# Visualization Endpoints
# ============================================================================


@router.get(
    "/experiments/{experiment_id}/visualization",
    summary="Get visualization data for experiment results",
    description="Returns data for visualizing predictions vs actuals, suitable for charts.",
)
def get_experiment_visualization(
    experiment_id: UUID,
    max_points: int = 500,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get visualization data for experiment results.

    Returns data formatted for different visualization types based on the task:
    - Classification: Confusion matrix data, class distribution
    - Regression: Predicted vs actual scatter/line data
    - Time Series: Time-based line charts with predictions

    The response adapts to what data is available and provides
    fallback visualizations if primary data is missing.
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    import logging
    logger = logging.getLogger(__name__)

    try:
        from app.models.validation_sample import ValidationSample
        from app.models.dataset_spec import DatasetSpec

        # Get experiment
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found",
            )

        if experiment.status != ExperimentStatus.COMPLETED:
            return {
                "visualization_type": "pending",
                "message": "Experiment not yet completed",
                "data": None,
            }

        # Get the best model version
        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.experiment_id == experiment_id)
            .order_by(ModelVersion.created_at.desc())
            .first()
        )

        if not model_version:
            return {
                "visualization_type": "no_model",
                "message": "No model found for this experiment",
                "data": None,
            }

        # Get task type
        task_type = "binary"
        if experiment.experiment_plan_json:
            task_type = experiment.experiment_plan_json.get("task_type", "binary")

        # Get target column name
        target_column = "target"
        dataset_spec = experiment.dataset_spec
        if dataset_spec:
            target_column = dataset_spec.target_column or "target"

        # Try to get validation samples
        validation_samples = (
            db.query(ValidationSample)
            .filter(ValidationSample.model_version_id == model_version.id)
            .order_by(ValidationSample.row_index)
            .limit(max_points)
            .all()
        )

        # Build response based on task type and available data
        # Safely get metrics, handling None, and normalize negative error metrics
        metrics = {}
        if model_version.metrics_json:
            metrics = normalize_metrics(model_version.metrics_json)
        elif experiment.best_metrics:
            metrics = normalize_metrics(experiment.best_metrics)

        # Get dataset context for contextual metric explanations
        dataset_context = _get_dataset_context(db, dataset_spec, target_column)

        # Extract validation strategy from experiment plan
        validation_strategy = None
        if experiment.experiment_plan_json:
            validation_strategy = experiment.experiment_plan_json.get("validation_strategy")

        # Compute baseline comparison for regression tasks
        baseline_comparison = None
        if task_type in ["regression", "quantile"] and dataset_context:
            target_std = dataset_context.get("target_std")
            rmse = metrics.get("root_mean_squared_error") or metrics.get("rmse")
            if target_std and rmse:
                # Baseline RMSE (predicting mean) equals target standard deviation
                improvement_vs_baseline = ((target_std - rmse) / target_std * 100) if target_std > 0 else 0
                baseline_comparison = {
                    "baseline_rmse": target_std,
                    "model_rmse": rmse,
                    "improvement_pct": improvement_vs_baseline,
                    "interpretation": (
                        f"Model RMSE ({rmse:.4f}) vs baseline ({target_std:.4f}): "
                        f"{abs(improvement_vs_baseline):.1f}% {'better' if improvement_vs_baseline > 0 else 'worse'} than predicting the mean"
                    ),
                }

        # Detect potential overfitting from train vs validation scores
        overfitting_warning = None
        score_train = metrics.get("score_train")
        score_val = metrics.get("score_val")
        if score_train is not None and score_val is not None:
            gap = abs(score_train - score_val)
            # For most metrics, a gap > 10% suggests overfitting
            if gap > 0.1 * abs(score_train):
                overfitting_warning = {
                    "train_score": score_train,
                    "val_score": score_val,
                    "gap": gap,
                    "message": f"Train/validation gap of {gap:.4f} may indicate overfitting",
                }

        result = {
            "experiment_id": str(experiment_id),
            "model_version_id": str(model_version.id),
            "task_type": task_type,
            "target_column": target_column,
            "primary_metric": experiment.primary_metric or "score_val",
            "metrics": metrics,
            "dataset_context": dataset_context,
            "validation_strategy": validation_strategy,
            "baseline_comparison": baseline_comparison,
            "overfitting_warning": overfitting_warning,
        }

        if validation_samples:
            # We have validation samples - build visualization from them
            if task_type in ["binary", "multiclass"]:
                result.update(_build_classification_viz(validation_samples, task_type))
            elif task_type in ["regression", "quantile"]:
                result.update(_build_regression_viz(validation_samples))
            elif task_type == "time_series":
                result.update(_build_timeseries_viz(validation_samples, dataset_spec))
            else:
                # Default to regression-style viz
                result.update(_build_regression_viz(validation_samples))
        else:
            # No validation samples - provide summary visualization from metrics
            result.update(_build_metrics_summary_viz(
                metrics,
                task_type,
                experiment.primary_metric
            ))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating visualization for experiment {experiment_id}")
        # Return a graceful error response instead of 500
        return {
            "visualization_type": "error",
            "message": f"Could not generate visualization: {str(e)}",
            "data": None,
        }


def _build_classification_viz(samples, task_type: str):
    """Build visualization data for classification tasks."""
    from collections import Counter

    # Collect predictions and actuals
    predictions = []
    actuals = []
    probabilities = []

    for sample in samples:
        actuals.append(sample.target_value)
        predictions.append(sample.predicted_value)
        if sample.prediction_probabilities_json:
            probabilities.append(sample.prediction_probabilities_json)

    # Get unique classes
    all_classes = sorted(set(actuals) | set(predictions))

    # Build confusion matrix
    confusion_matrix = {}
    for actual in all_classes:
        confusion_matrix[str(actual)] = {}
        for pred in all_classes:
            confusion_matrix[str(actual)][str(pred)] = 0

    for actual, pred in zip(actuals, predictions):
        confusion_matrix[str(actual)][str(pred)] += 1

    # Class distribution
    actual_dist = Counter(actuals)
    pred_dist = Counter(predictions)

    # Accuracy per class
    class_accuracy = {}
    for cls in all_classes:
        correct = sum(1 for a, p in zip(actuals, predictions) if a == cls and p == cls)
        total = sum(1 for a in actuals if a == cls)
        class_accuracy[str(cls)] = correct / total if total > 0 else 0

    # Sample predictions for display (first 50)
    sample_predictions = []
    for i, sample in enumerate(samples[:50]):
        entry = {
            "index": i,
            "actual": sample.target_value,
            "predicted": sample.predicted_value,
            "correct": sample.target_value == sample.predicted_value,
        }
        if sample.prediction_probabilities_json:
            entry["probabilities"] = sample.prediction_probabilities_json
        sample_predictions.append(entry)

    return {
        "visualization_type": "classification",
        "data": {
            "classes": [str(c) for c in all_classes],
            "confusion_matrix": confusion_matrix,
            "class_distribution": {
                "actual": {str(k): v for k, v in actual_dist.items()},
                "predicted": {str(k): v for k, v in pred_dist.items()},
            },
            "class_accuracy": class_accuracy,
            "total_samples": len(samples),
            "correct_predictions": sum(1 for a, p in zip(actuals, predictions) if a == p),
            "sample_predictions": sample_predictions,
        },
        "chart_config": {
            "recommended_charts": ["confusion_matrix", "class_distribution", "accuracy_by_class"],
            "title": f"Classification Results: {len(all_classes)} Classes",
            "description": "How well the model predicts each category",
        },
    }


def _build_regression_viz(samples):
    """Build visualization data for regression tasks."""
    import statistics

    # Collect predictions and actuals
    points = []
    errors = []

    for sample in samples:
        try:
            actual = float(sample.target_value)
            predicted = float(sample.predicted_value)
            error = predicted - actual
            abs_error = abs(error)
            pct_error = abs(error / actual * 100) if actual != 0 else 0

            points.append({
                "index": sample.row_index,
                "actual": actual,
                "predicted": predicted,
                "error": error,
                "abs_error": abs_error,
                "pct_error": pct_error,
            })
            errors.append(abs_error)
        except (ValueError, TypeError):
            continue

    if not points:
        return {
            "visualization_type": "no_data",
            "message": "Could not parse numeric values from validation samples",
            "data": None,
        }

    # Sort by actual value for line chart
    points_sorted = sorted(points, key=lambda x: x["actual"])

    # Calculate summary stats
    actuals = [p["actual"] for p in points]
    predictions = [p["predicted"] for p in points]

    mean_error = statistics.mean(errors) if errors else 0
    median_error = statistics.median(errors) if errors else 0

    # Calculate R² approximation
    mean_actual = statistics.mean(actuals) if actuals else 0
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
    ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Find worst predictions (largest errors)
    worst_predictions = sorted(points, key=lambda x: x["abs_error"], reverse=True)[:10]

    # Error distribution buckets
    error_buckets = {"< 1%": 0, "1-5%": 0, "5-10%": 0, "10-25%": 0, "> 25%": 0}
    for p in points:
        pct = p["pct_error"]
        if pct < 1:
            error_buckets["< 1%"] += 1
        elif pct < 5:
            error_buckets["1-5%"] += 1
        elif pct < 10:
            error_buckets["5-10%"] += 1
        elif pct < 25:
            error_buckets["10-25%"] += 1
        else:
            error_buckets["> 25%"] += 1

    return {
        "visualization_type": "regression",
        "data": {
            "points": points_sorted[:200],  # Limit for performance
            "total_samples": len(points),
            "summary": {
                "min_actual": min(actuals),
                "max_actual": max(actuals),
                "min_predicted": min(predictions),
                "max_predicted": max(predictions),
                "mean_error": mean_error,
                "median_error": median_error,
                "r_squared": r_squared,
            },
            "worst_predictions": worst_predictions,
            "error_distribution": error_buckets,
        },
        "chart_config": {
            "recommended_charts": ["scatter_actual_vs_predicted", "error_distribution", "residuals"],
            "title": "Regression Results: Predicted vs Actual",
            "description": "How close are the model's predictions to the real values?",
            "x_axis": "Actual Value",
            "y_axis": "Predicted Value",
        },
    }


def _build_timeseries_viz(samples, dataset_spec):
    """Build visualization data for time series tasks."""
    # Try to detect time/date column from dataset spec
    time_column = None
    if dataset_spec and dataset_spec.spec_json:
        # Look for datetime columns
        for col, info in dataset_spec.spec_json.get("columns", {}).items():
            if info.get("dtype") in ["datetime64", "datetime", "date"] or "date" in col.lower() or "time" in col.lower():
                time_column = col
                break

    # Build time series points
    points = []
    for sample in samples:
        try:
            actual = float(sample.target_value)
            predicted = float(sample.predicted_value)

            point = {
                "index": sample.row_index,
                "actual": actual,
                "predicted": predicted,
                "error": predicted - actual,
            }

            # Try to extract time value from features
            if time_column and sample.features_json:
                time_val = sample.features_json.get(time_column)
                if time_val:
                    point["time"] = str(time_val)

            points.append(point)
        except (ValueError, TypeError):
            continue

    if not points:
        return _build_regression_viz(samples)  # Fallback

    # Sort by index (assumed to be chronological)
    points_sorted = sorted(points, key=lambda x: x["index"])

    # Calculate trend accuracy (direction prediction)
    direction_correct = 0
    for i in range(1, len(points_sorted)):
        actual_dir = points_sorted[i]["actual"] - points_sorted[i-1]["actual"]
        pred_dir = points_sorted[i]["predicted"] - points_sorted[i-1]["predicted"]
        if (actual_dir >= 0 and pred_dir >= 0) or (actual_dir < 0 and pred_dir < 0):
            direction_correct += 1

    direction_accuracy = direction_correct / (len(points_sorted) - 1) if len(points_sorted) > 1 else 0

    return {
        "visualization_type": "timeseries",
        "data": {
            "points": points_sorted[:300],  # Limit for performance
            "total_samples": len(points),
            "time_column": time_column,
            "direction_accuracy": direction_accuracy,
            "summary": {
                "trend_prediction_accuracy": f"{direction_accuracy * 100:.1f}%",
            },
        },
        "chart_config": {
            "recommended_charts": ["line_chart", "direction_accuracy"],
            "title": "Time Series Forecast vs Actual",
            "description": "How well does the model predict future values?",
            "x_axis": time_column or "Time Index",
            "y_axis": "Value",
        },
    }


def _build_metrics_summary_viz(metrics: dict, task_type: str, primary_metric: str = None):
    """Build a summary visualization from metrics when no validation samples exist."""
    # Filter to numeric metrics only
    numeric_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not key.startswith("_"):
            numeric_metrics[key] = float(value)

    if not numeric_metrics:
        return {
            "visualization_type": "no_data",
            "message": "No metrics available for visualization",
            "data": None,
        }

    # Determine metric quality for display
    metric_quality = {}
    for key, value in numeric_metrics.items():
        quality = _assess_metric_quality(key, value, task_type)
        metric_quality[key] = quality

    # Highlight primary metric
    primary = primary_metric or "score_val"
    primary_value = numeric_metrics.get(primary, numeric_metrics.get("score_val", 0))
    primary_quality = metric_quality.get(primary, "unknown")

    return {
        "visualization_type": "metrics_summary",
        "data": {
            "metrics": numeric_metrics,
            "metric_quality": metric_quality,
            "primary_metric": {
                "name": primary,
                "value": primary_value,
                "quality": primary_quality,
            },
        },
        "chart_config": {
            "recommended_charts": ["metric_bars", "quality_gauge"],
            "title": "Model Performance Summary",
            "description": "Key metrics from training and validation",
        },
    }


def _assess_metric_quality(metric_key: str, value: float, task_type: str) -> str:
    """Assess the quality of a metric value."""
    key = metric_key.lower().replace("-", "_")

    # Classification metrics
    if key in ["roc_auc", "auc"]:
        if value >= 0.9:
            return "excellent"
        if value >= 0.8:
            return "good"
        if value >= 0.7:
            return "fair"
        if value >= 0.6:
            return "poor"
        return "very_poor"

    if key in ["accuracy", "balanced_accuracy"]:
        if value >= 0.95:
            return "excellent"
        if value >= 0.85:
            return "good"
        if value >= 0.75:
            return "fair"
        if value >= 0.65:
            return "poor"
        return "very_poor"

    if key in ["f1", "precision", "recall"]:
        if value >= 0.9:
            return "excellent"
        if value >= 0.7:
            return "good"
        if value >= 0.5:
            return "fair"
        return "poor"

    if key == "mcc":
        if value >= 0.7:
            return "excellent"
        if value >= 0.5:
            return "good"
        if value >= 0.3:
            return "fair"
        if value >= 0.1:
            return "poor"
        return "very_poor"

    if key == "r2":
        if value >= 0.9:
            return "excellent"
        if value >= 0.7:
            return "good"
        if value >= 0.5:
            return "fair"
        if value >= 0:
            return "poor"
        return "very_poor"

    return "unknown"


def _get_dataset_context(db, dataset_spec, target_column: str) -> dict:
    """Get dataset context for contextual metric explanations.

    Returns statistics about the target variable to help explain metrics
    in the context of the actual data scale.
    """
    import pandas as pd
    import os

    context = {
        "target_column": target_column,
        "target_min": None,
        "target_max": None,
        "target_mean": None,
        "target_std": None,
        "row_count": None,
        "feature_count": None,
    }

    if not dataset_spec:
        return context

    # Try to get row/feature count from spec
    if dataset_spec.spec_json:
        context["row_count"] = dataset_spec.spec_json.get("row_count")
        context["feature_count"] = len(dataset_spec.feature_columns or [])

    # Try to load actual data to compute target statistics
    try:
        # DatasetSpec uses data_sources_json, not data_source relationship
        from app.models.data_source import DataSource

        data_source_id = None
        if dataset_spec.data_sources_json:
            sources = dataset_spec.data_sources_json
            if isinstance(sources, list) and len(sources) > 0:
                first_source = sources[0]
                if isinstance(first_source, dict):
                    data_source_id = first_source.get("data_source_id") or first_source.get("id")
                elif isinstance(first_source, str):
                    data_source_id = first_source
            elif isinstance(sources, dict):
                data_source_id = sources.get("data_source_id") or sources.get("id")

        if not data_source_id:
            return context

        data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if not data_source:
            return context

        file_path = data_source.file_path
        if not file_path or not os.path.exists(file_path):
            return context

        # Load data and compute target statistics
        if file_path.endswith('.csv'):
            # First get the actual row count efficiently
            with open(file_path, 'r') as f:
                actual_row_count = sum(1 for _ in f) - 1  # Subtract header
            context["row_count"] = actual_row_count

            # Then load sample for statistics
            df = pd.read_csv(file_path, nrows=10000)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, nrows=10000)
            # For Excel, we can't easily get full row count without loading all
            context["row_count"] = len(df)  # Will be limited to 10000
        else:
            return context

        if target_column not in df.columns:
            return context

        # Compute target statistics for numeric targets
        target_series = pd.to_numeric(df[target_column], errors='coerce')
        target_series = target_series.dropna()

        if len(target_series) > 0:
            context["target_min"] = float(target_series.min())
            context["target_max"] = float(target_series.max())
            context["target_mean"] = float(target_series.mean())
            context["target_std"] = float(target_series.std())
            # Don't overwrite actual row count for CSV
            if not file_path.endswith('.csv'):
                context["row_count"] = len(df)

    except Exception as e:
        # Silently fail - context is optional
        import logging
        logging.getLogger(__name__).debug(f"Could not compute dataset context: {e}")

    return context


# =============================================================================
# NOTEBOOK GENERATION ENDPOINTS
# =============================================================================

@router.get(
    "/experiments/{experiment_id}/notebook",
    summary="Get experiment notebook",
    description="Generate a Jupyter notebook documenting the experiment pipeline.",
    response_class=None,  # Will return JSON or file depending on format
)
def get_experiment_notebook(
    experiment_id: UUID,
    format: str = "json",
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a reproducible Jupyter notebook for an experiment.

    Args:
        experiment_id: ID of the experiment
        format: 'json' for notebook JSON content, 'download' for .ipynb file

    Returns:
        Notebook content as JSON or downloadable file
    """
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    from fastapi.responses import Response, JSONResponse
    from app.services.notebook_generator import generate_experiment_notebook

    # Check experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    # Generate notebook
    notebook_json = generate_experiment_notebook(str(experiment_id), db)

    if not notebook_json:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate notebook",
        )

    if format == "download":
        # Return as downloadable .ipynb file
        filename = f"experiment_{experiment.name.replace(' ', '_')}_{experiment_id}.ipynb"
        return Response(
            content=notebook_json,
            media_type="application/x-ipynb+json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )
    else:
        # Return as JSON for rendering in UI
        import json
        return JSONResponse(content=json.loads(notebook_json))

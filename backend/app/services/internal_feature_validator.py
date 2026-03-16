"""Internal feature validation for improvement pipeline.

This module provides functionality to validate new engineered features
INTERNALLY before creating user-facing experiments. When new features
are created during improvement iterations, we run a quick training
to verify they actually help before committing to a full training run.

Flow:
1. New features proposed by improvement pipeline
2. Run quick internal validation training (60-90 seconds)
3. Compare score to parent experiment
4. If improved: proceed to full production training
5. If not improved: record failure, optionally retry with different features
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Experiment, Trial, DatasetSpec
from app.models.experiment import ExperimentStatus

logger = logging.getLogger(__name__)


def get_experiment_final_score(experiment: Experiment) -> Optional[float]:
    """Get the final score for an experiment from its best model version.

    The final score is computed from the model version's metrics_json, preferring
    holdout_score over validation score. This mirrors the logic in the API layer.

    Args:
        experiment: The experiment to get the score from

    Returns:
        The final score (holdout or validation), or None if not available
    """
    if not experiment.model_versions:
        return None

    # Find the best model version (highest validation score)
    best_version = None
    best_val_score = None

    for mv in experiment.model_versions:
        metrics = mv.metrics_json or {}
        val_score = metrics.get("validation_score") or metrics.get("val_score")
        if val_score is not None:
            if best_val_score is None or val_score > best_val_score:
                best_val_score = val_score
                best_version = mv

    if not best_version:
        # Fallback to first model version if none have validation scores
        best_version = experiment.model_versions[0] if experiment.model_versions else None

    if not best_version:
        return None

    metrics = best_version.metrics_json or {}

    # Prefer holdout score over validation score
    holdout_score = metrics.get("holdout_score")
    if holdout_score is not None:
        return holdout_score

    # Fallback to validation score
    val_score = metrics.get("validation_score") or metrics.get("val_score")
    return val_score


# Quick validation training config
QUICK_VALIDATION_CONFIG = {
    "time_limit": 90,  # 90 seconds for quick validation
    "presets": "medium_quality",  # Faster preset for validation
    "num_bag_folds": 0,  # No bagging for speed
    "num_stack_levels": 0,  # No stacking for speed
}

# Minimum improvement threshold to consider features successful
MIN_IMPROVEMENT_THRESHOLD = 0.001  # 0.1% improvement required


def needs_feature_validation(
    new_engineered_features: list,
    parent_experiment: Experiment,
) -> bool:
    """Determine if we need to run internal feature validation.

    We validate when:
    - There are new engineered features (not just dropped features)
    - The parent experiment has a valid score to compare against

    Args:
        new_engineered_features: List of new features from dataset design
        parent_experiment: The parent experiment we're improving

    Returns:
        True if validation is needed
    """
    if not new_engineered_features:
        logger.info("No new features - skipping internal validation")
        return False

    # Need a baseline score to compare against
    parent_score = get_experiment_final_score(parent_experiment)
    if parent_score is None:
        logger.warning("Parent has no score - skipping validation (no baseline)")
        return False

    logger.info(f"Will validate {len(new_engineered_features)} new features")
    return True


def run_quick_validation_training(
    db: Session,
    dataset_spec: DatasetSpec,
    parent_experiment: Experiment,
    validation_experiment_name: str,
) -> Tuple[Optional[float], Optional[str]]:
    """Run a quick internal training to validate new features.

    This runs synchronously (blocking) and creates a hidden/internal
    experiment that is not shown prominently to users.

    Args:
        db: Database session
        dataset_spec: The new dataset spec with engineered features
        parent_experiment: Parent experiment for comparison
        validation_experiment_name: Name for the validation experiment

    Returns:
        Tuple of (validation_score, error_message)
        - If successful: (score, None)
        - If failed: (None, error_message)
    """
    from app.tasks.automl import load_dataset_from_spec
    from app.services.automl_runner import TabularRunner

    logger.info(f"Starting quick internal validation training for {validation_experiment_name}")

    try:
        # Create a minimal experiment for validation
        # Mark it as internal so it's not prominently displayed
        validation_experiment = Experiment(
            project_id=parent_experiment.project_id,
            dataset_spec_id=dataset_spec.id,
            name=f"[Internal Validation] {validation_experiment_name}",
            description="Internal feature validation - auto-generated",
            status=ExperimentStatus.RUNNING,
            primary_metric=parent_experiment.primary_metric,
            metric_direction=parent_experiment.metric_direction,
            experiment_plan_json={
                "automl_config": QUICK_VALIDATION_CONFIG,
                "internal_validation": True,  # Flag as internal
            },
            iteration_number=parent_experiment.iteration_number,  # Same iteration (not a real iteration)
            parent_experiment_id=parent_experiment.id,
            improvement_context_json={
                "type": "internal_feature_validation",
                "parent_id": str(parent_experiment.id),
            },
        )
        db.add(validation_experiment)
        db.commit()
        db.refresh(validation_experiment)

        logger.info(f"Created internal validation experiment {validation_experiment.id}")

        # Load the dataset
        try:
            df = load_dataset_from_spec(db, dataset_spec)
            logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            validation_experiment.status = ExperimentStatus.FAILED
            validation_experiment.error_message = f"Failed to load dataset: {e}"
            db.commit()
            return None, f"Failed to load dataset: {e}"

        # Determine task type
        project = parent_experiment.project
        task_type = project.task_type.value if project and project.task_type else "binary"

        # Create temporary output path
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "validation_model"
            output_path.mkdir(parents=True, exist_ok=True)

            try:
                # Run quick training using TabularRunner
                runner = TabularRunner(
                    task_type=task_type,
                    label_column=dataset_spec.target_column,
                    output_path=str(output_path),
                    time_limit=QUICK_VALIDATION_CONFIG["time_limit"],
                    presets=QUICK_VALIDATION_CONFIG["presets"],
                    num_bag_folds=QUICK_VALIDATION_CONFIG["num_bag_folds"],
                    num_stack_levels=QUICK_VALIDATION_CONFIG["num_stack_levels"],
                )

                # Split data for validation
                from sklearn.model_selection import train_test_split
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

                logger.info(f"Running quick validation training: {len(train_df)} train, {len(val_df)} val")

                # Run training
                result = runner.run(train_df=train_df, val_df=val_df)

                # Get the validation score
                val_score = result.metrics.get("score_val") or result.metrics.get(parent_experiment.primary_metric)

                # Update experiment status
                validation_experiment.status = ExperimentStatus.COMPLETED
                validation_experiment.val_score = val_score
                db.commit()

                logger.info(f"Internal validation complete: score = {val_score}")
                return val_score, None

            except Exception as e:
                validation_experiment.status = ExperimentStatus.FAILED
                validation_experiment.error_message = str(e)
                db.commit()
                logger.error(f"Internal validation training failed: {e}")
                return None, str(e)

    except Exception as e:
        logger.error(f"Failed to create validation experiment: {e}")
        return None, str(e)


def evaluate_validation_result(
    validation_score: Optional[float],
    parent_score: float,
    metric_direction: str = "maximize",
    threshold: float = MIN_IMPROVEMENT_THRESHOLD,
) -> Tuple[bool, str]:
    """Evaluate if the validation result indicates features are helpful.

    Args:
        validation_score: Score from quick validation training
        parent_score: Score from parent experiment
        metric_direction: "maximize" or "minimize"
        threshold: Minimum improvement required

    Returns:
        Tuple of (features_are_good, explanation)
    """
    if validation_score is None:
        return False, "Validation training failed - cannot evaluate features"

    if metric_direction == "maximize":
        improvement = validation_score - parent_score
        improved = improvement > threshold
    else:
        improvement = parent_score - validation_score
        improved = improvement > threshold

    if improved:
        explanation = (
            f"Features validated successfully. "
            f"Validation score: {validation_score:.4f} vs parent: {parent_score:.4f} "
            f"(improvement: {improvement:+.4f})"
        )
        logger.info(explanation)
        return True, explanation
    else:
        explanation = (
            f"Features did not improve performance. "
            f"Validation score: {validation_score:.4f} vs parent: {parent_score:.4f} "
            f"(change: {improvement:+.4f}, threshold: {threshold})"
        )
        logger.warning(explanation)
        return False, explanation


def create_production_experiment_config(
    experiment_design: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a production-quality experiment config.

    This overrides any "quick check" settings with production settings.

    Args:
        experiment_design: Original experiment design from LLM

    Returns:
        Modified config suitable for production training
    """
    automl_config = dict(experiment_design.get("automl_config", {}))

    # Ensure production-quality settings
    if automl_config.get("time_limit", 0) < 300:
        automl_config["time_limit"] = 300  # Minimum 5 minutes for production

    if automl_config.get("presets") in ["medium_quality", "good_quality"]:
        automl_config["presets"] = "high_quality"  # Upgrade to high quality

    # Ensure reasonable bagging/stacking
    if automl_config.get("num_bag_folds", 0) < 5:
        automl_config["num_bag_folds"] = 5

    return {
        **experiment_design,
        "automl_config": automl_config,
        "validated_features": True,  # Mark that features were validated
    }

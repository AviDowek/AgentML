"""Celery tasks for running ML experiments."""
import logging
import os
from datetime import datetime
from typing import Optional
from uuid import UUID

import pandas as pd

from app.tasks.celery_app import celery_app
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.experiment import Experiment, Trial, ExperimentStatus, TrialStatus
from app.models.model_version import ModelVersion
from app.models.validation_sample import ValidationSample
from app.services.dataset_builder import DatasetBuilder
from app.services.automl_runner import get_runner_for_task, ValidationPrediction
from app.services.holdout_validator import (
    get_or_create_holdout_indices,
    evaluate_on_holdout,
    record_holdout_score,
    get_overfitting_report,
)
from app.services.baseline_models import compute_all_baselines
from app.services.robust_validation import (
    run_robust_validation,
    format_robust_validation_summary,
    RobustValidationResult,
)
from app.services.smart_metrics import (
    SmartMetricSelector,
    get_smart_metric_recommendations,
    format_metric_recommendations,
)
from app.core.exceptions import (
    LLMError,
    LLMTimeoutError,
    DataError,
    DatasetBuildError,
    TrainingError,
    AutoMLError,
    CeleryTaskError,
    ImprovementPipelineError,
)
from app.core.task_dispatch import dispatch_task

logger = logging.getLogger(__name__)


def sanitize_json_values(obj):
    """Recursively sanitize JSON data by replacing NaN, Infinity with None.

    PostgreSQL's JSON type doesn't accept NaN or Infinity values, so we need
    to replace them with null before saving to the database.

    Args:
        obj: Any JSON-serializable object (dict, list, or primitive)

    Returns:
        The sanitized object with NaN/Infinity replaced by None
    """
    import math

    if isinstance(obj, dict):
        return {k: sanitize_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj


def _detect_task_type(target_series: pd.Series) -> str:
    """Auto-detect ML task type from target column data.

    Args:
        target_series: The target column as a pandas Series

    Returns:
        One of: 'regression', 'binary', 'multiclass'
    """
    # Drop NA values for analysis
    non_null = target_series.dropna()

    if len(non_null) == 0:
        return "binary"  # Default fallback

    # Check if numeric
    is_numeric = pd.api.types.is_numeric_dtype(non_null)
    unique_count = non_null.nunique()

    if is_numeric:
        # If numeric with many unique values (>20 or >10% of data), it's regression
        unique_ratio = unique_count / len(non_null)
        if unique_count > 20 or unique_ratio > 0.1:
            # Check if values look like floats (not integers masquerading as floats)
            if non_null.dtype in ['float64', 'float32']:
                # Check if any values have decimal parts
                has_decimals = (non_null % 1 != 0).any()
                if has_decimals:
                    return "regression"

            # Even integers with many unique values are likely regression
            if unique_count > 50:
                return "regression"

        # Few unique numeric values - likely classification
        if unique_count == 2:
            return "binary"
        elif unique_count <= 20:
            return "multiclass"
        else:
            return "regression"
    else:
        # Categorical/string target
        if unique_count == 2:
            return "binary"
        else:
            return "multiclass"


def _get_default_metric(task_type: str) -> str:
    """Get the default primary metric for a task type.

    DEPRECATED: Use get_smart_metric() instead which analyzes data characteristics.
    This function is kept for backward compatibility.

    Args:
        task_type: The ML task type (regression, binary, multiclass, etc.)

    Returns:
        The default metric name for that task type
    """
    metric_map = {
        "regression": "root_mean_squared_error",
        "binary": "roc_auc",  # Better than accuracy for any imbalance
        "multiclass": "f1_macro",  # Better than accuracy for imbalanced classes
        "quantile": "pinball_loss",
    }
    return metric_map.get(task_type, "f1_macro")


def get_smart_metric(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> tuple[str, dict]:
    """Get the optimal metric based on data characteristics.

    Args:
        df: Training DataFrame
        target_column: Name of target column
        task_type: ML task type (regression, binary, multiclass)

    Returns:
        Tuple of (recommended_metric, data_characteristics_dict)
    """
    try:
        characteristics = get_smart_metric_recommendations(
            df=df,
            target_column=target_column,
            task_type=task_type,
        )

        # Log the recommendations
        logger.info(format_metric_recommendations(characteristics))

        # Return as dict for storage/passing
        char_dict = {
            "task_type": characteristics.task_type,
            "num_classes": characteristics.num_classes,
            "imbalance_ratio": characteristics.imbalance_ratio,
            "imbalance_severity": characteristics.imbalance_severity.value,
            "recommended_metric": characteristics.recommended_metric,
            "secondary_metrics": characteristics.recommended_secondary_metrics,
            "should_use_sample_weights": characteristics.should_use_sample_weights,
            "recommended_presets": characteristics.recommended_presets,
            "total_samples": characteristics.total_samples,
        }

        return characteristics.recommended_metric, char_dict

    except Exception as e:
        logger.warning(f"Smart metric selection failed: {e}. Falling back to defaults.")
        return _get_default_metric(task_type), {}


def apply_resource_limits(
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    memory_limit_gb: Optional[int] = None,
) -> dict:
    """Apply resource limits via environment variables for AutoGluon/NumPy/OpenBLAS.

    Returns a dict with the applied limits for logging.
    """
    settings = get_settings()

    # Use provided values or fall back to config defaults
    cpus = num_cpus if num_cpus is not None else settings.autogluon_num_cpus
    gpus = num_gpus if num_gpus is not None else settings.autogluon_num_gpus
    memory = memory_limit_gb if memory_limit_gb is not None else settings.max_memory_gb

    # Set environment variables that limit parallelism
    # These must be set BEFORE importing numpy/autogluon in the worker
    os.environ["OMP_NUM_THREADS"] = str(cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
    os.environ["MKL_NUM_THREADS"] = str(cpus)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpus)

    # AutoGluon-specific resource controls
    os.environ["AG_NUM_CPUS"] = str(cpus)
    os.environ["AG_NUM_GPUS"] = str(gpus)

    # Ray (used by AutoGluon) resource limits
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # Disable memory monitor overhead

    logger.info(f"Applied resource limits: CPUs={cpus}, GPUs={gpus}, Memory={memory}GB")

    return {"num_cpus": cpus, "num_gpus": gpus, "memory_limit_gb": memory}


def save_validation_samples(
    db,
    model_version_id,
    validation_predictions: list[ValidationPrediction],
    max_samples: int = 1000,
) -> int:
    """Save validation predictions to the database.

    Args:
        db: Database session
        model_version_id: UUID of the model version
        validation_predictions: List of validation predictions
        max_samples: Maximum number of samples to store (to avoid huge DB entries)

    Returns:
        Number of samples saved
    """
    if not validation_predictions:
        return 0

    # Limit samples if too many (sample evenly from the set)
    predictions_to_save = validation_predictions
    if len(validation_predictions) > max_samples:
        # Sample evenly across the validation set
        step = len(validation_predictions) / max_samples
        indices = [int(i * step) for i in range(max_samples)]
        predictions_to_save = [validation_predictions[i] for i in indices]
        logger.info(
            f"Limiting validation samples from {len(validation_predictions)} to {max_samples}"
        )

    # Create ValidationSample records
    samples = []
    for pred in predictions_to_save:
        sample = ValidationSample(
            model_version_id=model_version_id,
            row_index=pred.row_index,
            features_json=pred.features,
            target_value=str(pred.target_value),
            predicted_value=str(pred.predicted_value),
            error_value=pred.error_value,
            absolute_error=pred.absolute_error,
            prediction_probabilities_json=pred.prediction_probabilities,
        )
        samples.append(sample)

    db.bulk_save_objects(samples)
    db.commit()

    return len(samples)


@celery_app.task(bind=True, name="app.tasks.run_experiment")
def run_experiment(
    self,
    experiment_id: str,
    resource_limits_enabled: bool = True,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    memory_limit_gb: Optional[int] = None,
) -> dict:
    """Run an ML experiment.

    This task:
    1. Applies resource limits (if enabled)
    2. Loads the experiment and dataset spec
    3. Builds the dataset from data sources
    4. Runs AutoML training
    5. Creates a Trial and ModelVersion with results

    Args:
        experiment_id: UUID of the experiment to run
        resource_limits_enabled: Whether to apply CPU/memory limits (default True)
        num_cpus: Override number of CPUs (None = use config default)
        num_gpus: Override number of GPUs (None = use config default)
        memory_limit_gb: Override memory limit in GB (None = use config default)

    Returns:
        Dictionary with experiment results
    """
    settings = get_settings()

    # Apply resource limits FIRST before any heavy imports
    applied_limits = None
    if resource_limits_enabled and settings.resource_limits_enabled:
        applied_limits = apply_resource_limits(num_cpus, num_gpus, memory_limit_gb)
    else:
        logger.info("Resource limits disabled - using all available resources")

    db = SessionLocal()

    try:
        experiment_uuid = UUID(experiment_id)

        # Load experiment
        experiment = db.query(Experiment).filter(Experiment.id == experiment_uuid).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        logger.info(f"Starting experiment: {experiment.name} ({experiment_id})")

        # Update experiment status to running
        experiment.status = ExperimentStatus.RUNNING
        db.commit()

        # Validate experiment has dataset_spec
        if not experiment.dataset_spec_id:
            raise ValueError(f"Experiment {experiment_id} has no dataset_spec_id configured")

        # Build the dataset
        builder = DatasetBuilder(db)
        dataset = builder.build_dataset_from_spec(experiment.dataset_spec_id)

        # =====================================================================
        # FEATURE ENGINEERING WARNING STORAGE: Persist any failures for UI visibility
        # =====================================================================
        fe_failures = dataset.attrs.get('_feature_engineering_failures', [])
        if fe_failures:
            logger.warning(f"⚠️ {len(fe_failures)} feature engineering step(s) FAILED!")
            for failure in fe_failures:
                logger.warning(f"   - {failure.get('feature', 'unknown')}: {failure.get('error', 'unknown error')}")

            # Store in experiment for UI visibility
            if experiment.experiment_plan_json is None:
                experiment.experiment_plan_json = {}
            experiment.experiment_plan_json['feature_engineering_warnings'] = [
                f"⚠️ Feature '{f.get('feature', 'unknown')}' FAILED: {f.get('error', 'unknown error')}"
                for f in fe_failures
            ]
            experiment.experiment_plan_json['feature_engineering_failure_count'] = len(fe_failures)
            experiment.experiment_plan_json['feature_engineering_success_count'] = dataset.attrs.get('_feature_engineering_success_count', 0)
            db.commit()
            logger.info(f"Stored {len(fe_failures)} feature engineering warnings in experiment record")

        # Get dataset spec for target column
        dataset_spec = experiment.dataset_spec
        if not dataset_spec:
            raise ValueError(f"DatasetSpec {experiment.dataset_spec_id} not found")

        target_column = dataset_spec.target_column
        if not target_column:
            raise ValueError(f"DatasetSpec {dataset_spec.id} has no target_column configured")

        # Update dataset spec with actual row count and feature columns if not set
        # This ensures future critiques have this info without reloading the dataset
        spec_updated = False
        spec_json = dataset_spec.spec_json or {}
        if not spec_json.get("row_count"):
            spec_json["row_count"] = len(dataset)
            spec_updated = True
        if not dataset_spec.feature_columns:
            feature_cols = [c for c in dataset.columns if c != target_column]
            dataset_spec.feature_columns = feature_cols
            spec_updated = True
        if spec_updated:
            dataset_spec.spec_json = spec_json
            db.commit()
            logger.info(f"Updated dataset spec with row_count={len(dataset)}, features={len(dataset_spec.feature_columns or [])}")

        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Handle rows with NaN/Inf in target column
        # Note: ML training REQUIRES known target values - rows with NaN targets cannot be used
        import numpy as np
        original_len = len(dataset)
        target_invalid = dataset[target_column].isna() | dataset[target_column].isin([np.inf, -np.inf])
        invalid_count = int(target_invalid.sum())

        if invalid_count > 0:
            invalid_pct = (invalid_count / original_len) * 100

            # Check if this looks like a shifted/lagged target (NaN at end of groups or series)
            # vs a data quality problem
            if invalid_pct > 20:
                # More than 20% invalid - this is likely a data problem, not just shifted targets
                raise ValueError(
                    f"Target column '{target_column}' has {invalid_count:,} invalid values ({invalid_pct:.1f}% of data). "
                    f"This is too high and indicates a data quality issue. "
                    f"Please check your feature engineering or data preparation. "
                    f"If this target is intentionally computed (e.g., future returns), ensure the formula is correct."
                )
            elif invalid_pct > 5:
                # 5-20% invalid - warn prominently but proceed
                logger.warning(f"⚠️ HIGH NaN RATE: {invalid_count:,} rows ({invalid_pct:.1f}%) have invalid target values")
                logger.warning(f"   This may indicate a data quality issue with '{target_column}'")
                logger.warning(f"   Common causes: incorrect feature engineering, missing data, or division by zero")
                logger.warning(f"   Proceeding with {original_len - invalid_count:,} valid rows...")
            else:
                # <5% invalid - likely just shifted targets (expected)
                logger.info(f"Removed {invalid_count:,} rows ({invalid_pct:.1f}%) with invalid target values (expected for shifted/lagged targets)")

            dataset = dataset[~target_invalid].copy()

        # Determine task type - try project setting first, then auto-detect from data
        project = experiment.project
        task_type = project.task_type.value if project.task_type else None

        # Auto-detect task type from target column if not explicitly set
        if not task_type or task_type == "classification":
            target_series = dataset[target_column]
            automl_task_type = _detect_task_type(target_series)
            logger.info(f"Auto-detected task type: {automl_task_type} from target column '{target_column}'")
        else:
            # Map task types to runner-compatible types
            task_type_map = {
                "classification": "binary",
                "binary": "binary",
                "multiclass": "multiclass",
                "regression": "regression",
                "quantile": "quantile",
                "timeseries_forecast": "timeseries_forecast",
                "multimodal_classification": "multimodal_classification",
                "multimodal_regression": "multimodal_regression",
            }
            automl_task_type = task_type_map.get(task_type, "binary")

        # =====================================================================
        # HOLDOUT VALIDATION: Create persistent holdout set for overfitting detection
        # =====================================================================
        holdout_df = None
        training_df = dataset  # Default to full dataset if holdout creation fails
        validation_strategy = spec_json.get("validation_strategy", {})

        try:
            logger.info("=" * 60)
            logger.info("🔒 HOLDOUT VALIDATION SETUP")
            logger.info("=" * 60)

            training_df, holdout_df, holdout_indices = get_or_create_holdout_indices(
                df=dataset,
                dataset_spec=dataset_spec,
                db=db,
                target_column=target_column,
                task_type=automl_task_type,
                validation_strategy=validation_strategy,
            )

            holdout_pct = len(holdout_df) / len(dataset) * 100
            logger.info(f"   ✓ Created holdout set: {len(holdout_df):,} samples ({holdout_pct:.1f}%)")
            logger.info(f"   ✓ Training set: {len(training_df):,} samples ({100-holdout_pct:.1f}%)")
            logger.info(f"   ✓ Split strategy: {validation_strategy.get('split_strategy', 'default')}")
            logger.info("   📌 This holdout is NEVER used during training - only for final evaluation")
            logger.info("   📌 It detects overfitting and unrealistic results across iterations")
            logger.info("=" * 60)

        except Exception as holdout_err:
            logger.warning(f"⚠️ Could not create holdout set: {holdout_err}")
            logger.warning("   Proceeding with full dataset for training (no overfitting detection)")
            training_df = dataset
            holdout_df = None

        # =====================================================================
        # ABLATION STUDY: Drop specified columns if ablation experiment
        # =====================================================================
        if experiment.experiment_plan_json:
            drop_columns = experiment.experiment_plan_json.get("drop_columns", [])
            ablation_target = experiment.experiment_plan_json.get("ablation_target", "")
            if drop_columns:
                # Filter to columns that exist in the dataset (excluding target)
                valid_drops = [c for c in drop_columns if c in training_df.columns and c != target_column]
                if valid_drops:
                    logger.info("=" * 60)
                    logger.info("🔬 ABLATION STUDY")
                    logger.info(f"   Target: {ablation_target or 'custom'}")
                    logger.info(f"   Dropping columns: {valid_drops}")
                    logger.info("=" * 60)
                    training_df = training_df.drop(columns=valid_drops)
                    if holdout_df is not None:
                        holdout_df = holdout_df.drop(columns=valid_drops)
                    logger.info(f"   ✓ Dataset now has {len(training_df.columns)} columns (was {len(training_df.columns) + len(valid_drops)})")
                else:
                    logger.warning(f"Ablation: No valid columns to drop (requested: {drop_columns})")

        # =====================================================================
        # SMART METRIC SELECTION: Analyze data to pick optimal metric
        # =====================================================================
        data_characteristics = {}
        primary_metric = experiment.primary_metric or None

        if not primary_metric:
            # No metric specified - use smart selection based on data analysis
            logger.info("=" * 60)
            logger.info("🎯 SMART METRIC SELECTION")
            logger.info("=" * 60)
            primary_metric, data_characteristics = get_smart_metric(
                df=training_df,
                target_column=target_column,
                task_type=automl_task_type,
            )
            logger.info(f"   ✓ Selected metric: {primary_metric}")
            if data_characteristics.get("imbalance_severity"):
                logger.info(f"   ✓ Class imbalance: {data_characteristics.get('imbalance_severity')}")
                logger.info(f"   ✓ Imbalance ratio: {data_characteristics.get('imbalance_ratio', 1):.2f}:1")
            if data_characteristics.get("should_use_sample_weights"):
                logger.info("   ✓ Sample weights: ENABLED (will be passed to AutoGluon)")
            logger.info("=" * 60)
        else:
            logger.info(f"Using user-specified metric: {primary_metric}")

        # Get AutoML config
        automl_config = {
            "time_limit": settings.automl_time_limit,
            "presets": settings.automl_presets,
        }

        # Override with experiment plan config if present
        if experiment.experiment_plan_json:
            plan_config = experiment.experiment_plan_json.get("automl_config", {})
            automl_config.update(plan_config)

        # Apply data characteristics to config (sample weights, presets)
        if data_characteristics:
            if data_characteristics.get("should_use_sample_weights"):
                automl_config["use_sample_weights"] = True
                automl_config["imbalance_severity"] = data_characteristics.get("imbalance_severity")
            if data_characteristics.get("recommended_presets") and "presets" not in (experiment.experiment_plan_json or {}).get("automl_config", {}):
                # Only apply recommended presets if not explicitly set
                automl_config["presets"] = data_characteristics.get("recommended_presets")
                logger.info(f"Applied recommended presets: {automl_config['presets']}")
            # Store data characteristics for later analysis
            automl_config["data_characteristics"] = data_characteristics

        # Normalize: convert max_runtime_seconds to time_limit if needed
        if "max_runtime_seconds" in automl_config:
            automl_config["time_limit"] = automl_config.pop("max_runtime_seconds")
            logger.info(f"Converted max_runtime_seconds to time_limit: {automl_config['time_limit']}s")

        logger.info(f"AutoML config for experiment: time_limit={automl_config.get('time_limit')}s, presets={automl_config.get('presets')}")

        # Create trial record
        trial = Trial(
            experiment_id=experiment.id,
            variant_name="AutoML_MVP",
            data_split_strategy="holdout_15pct" if holdout_df is not None else "random_80_20",
            automl_config=automl_config,
            status=TrialStatus.RUNNING,
        )
        db.add(trial)
        db.commit()
        db.refresh(trial)

        logger.info(f"Created trial: {trial.id}")

        # Set up log capture for real-time updates via Redis polling
        from app.services.training_logs import TrainingLogContext, TrainingLogStore

        # Add initial logs with interpretations before starting training
        pre_log_store = TrainingLogStore(str(experiment.id))
        pre_log_store.clear()  # Clear any stale logs
        pre_log_store.add_milestone("Starting local training...")
        pre_log_store.add_log(
            f"Dataset: {len(dataset)} rows, {len(dataset.columns)} columns",
            log_type="info",
            interpreted=f"Your dataset has {len(dataset):,} rows and {len(dataset.columns)} columns."
        )
        if holdout_df is not None:
            pre_log_store.add_log(
                f"Holdout: {len(holdout_df)} samples reserved for final validation",
                log_type="info",
                interpreted=f"🔒 {len(holdout_df):,} samples ({len(holdout_df)/len(dataset)*100:.1f}%) held out for overfitting detection."
            )
            pre_log_store.add_log(
                f"Training on: {len(training_df)} samples",
                log_type="info",
                interpreted=f"Training will use {len(training_df):,} samples (holdout is NEVER used during training)."
            )
        pre_log_store.add_log(
            f"Target: {target_column}, Task: {automl_task_type}",
            log_type="info",
            interpreted=f"Predicting '{target_column}' using {automl_task_type} approach."
        )
        pre_log_store.add_log(
            f"Time limit: {automl_config.get('time_limit', 300)}s",
            log_type="info",
            interpreted=f"Training will run for up to {automl_config.get('time_limit', 300) // 60} minutes."
        )

        # Check for robust/strict validation settings in experiment plan
        robust_validation_config = {}
        if experiment.experiment_plan_json:
            robust_validation_config = experiment.experiment_plan_json.get('robust_validation', {})
        
        validation_strategy_mode = robust_validation_config.get('validation_strategy', 'STANDARD')
        num_seeds = robust_validation_config.get('num_seeds', 3)
        cv_folds = robust_validation_config.get('cv_folds', 5)

        with TrainingLogContext(str(experiment.id)) as log_context:
            # Check if robust/strict validation is requested
            if validation_strategy_mode in ('ROBUST', 'STRICT'):
                logger.info(f'Using {validation_strategy_mode} validation with {num_seeds} seeds')
                
                robust_result = run_robust_validation(
                    dataset=training_df,
                    target_column=target_column,
                    task_type=automl_task_type,
                    primary_metric=primary_metric,
                    config=automl_config,
                    experiment_id=str(experiment.id),
                    artifacts_dir=settings.artifacts_dir,
                    validation_strategy=validation_strategy_mode,
                    num_seeds=num_seeds,
                    cv_folds=cv_folds,
                )
                
                # Log summary of robust validation
                summary = format_robust_validation_summary(robust_result, primary_metric)
                logger.info(summary)
                
                # Use primary result as the main result
                result = robust_result.primary_result
                
                # Store robust validation metrics in automl_config for later use
                automl_config['robust_validation_results'] = {
                    'mean_metrics': robust_result.mean_metrics,
                    'std_metrics': robust_result.std_metrics,
                    'num_seeds': robust_result.num_seeds,
                    'cv_folds': robust_result.cv_folds,
                    'is_stable': robust_result.is_stable,
                    'stability_warning': robust_result.stability_warning,
                    'coefficient_of_variation': robust_result.coefficient_of_variation,
                }
                if robust_result.confidence_interval_95:
                    automl_config['robust_validation_results']['confidence_interval_95'] = list(robust_result.confidence_interval_95)
            else:
                # Standard validation - single run
                runner = get_runner_for_task(
                    task_type=automl_task_type,
                    artifacts_dir=settings.artifacts_dir
                )

                # Train on training_df (excludes holdout)
                result = runner.run_experiment(
                    dataset=training_df,  # Use training set only (excludes holdout)
                    target_column=target_column,
                    task_type=automl_task_type,
                    primary_metric=primary_metric,
                    config=automl_config,
                    experiment_id=str(experiment.id),
                )

        # Get captured logs after context exits
        training_logs = log_context.get_all_logs_text()

        # Update trial with results - sanitize JSON to handle NaN/Infinity values
        trial.status = TrialStatus.COMPLETED
        trial.metrics_json = sanitize_json_values(result.metrics)
        trial.best_model_ref = result.best_model_name
        trial.logs_location = result.artifact_path
        trial.leaderboard_json = sanitize_json_values(result.leaderboard)  # Model leaderboard for critique
        trial.training_logs = training_logs  # Captured training output for AI analysis
        # Store baseline metrics for sanity checking
        if result.baseline_metrics:
            trial.baseline_metrics_json = sanitize_json_values(result.baseline_metrics)
            logger.info(f"Stored baseline metrics: {list(result.baseline_metrics.keys())}")
        db.commit()

        logger.info(f"Trial completed: best_model={result.best_model_name}")

        # Build serving config with feature information
        feature_columns = list(dataset.columns)
        if target_column in feature_columns:
            feature_columns.remove(target_column)

        # Determine feature types
        features = []
        for col in feature_columns:
            col_dtype = str(dataset[col].dtype)
            if col_dtype in ['int64', 'int32', 'float64', 'float32']:
                feat_type = 'numeric'
            elif col_dtype == 'bool':
                feat_type = 'boolean'
            elif col_dtype == 'datetime64[ns]':
                feat_type = 'datetime'
            else:
                feat_type = 'categorical'
            features.append({'name': col, 'type': feat_type})

        serving_config = {
            'features': features,
            'target_column': target_column,
            'task_type': automl_task_type,
        }

        # =========================================================================
        # TRAIN METRICS: Evaluate on training data for overfitting detection (local)
        # =========================================================================
        train_metrics = {}
        try:
            from autogluon.tabular import TabularPredictor
            predictor = TabularPredictor.load(result.artifact_path)

            logger.info("Evaluating on training data (for overfitting detection)...")
            train_eval = predictor.evaluate(training_df, silent=True)
            logger.info(f"Train evaluation result: {train_eval}")

            if isinstance(train_eval, dict):
                for k, v in train_eval.items():
                    import math
                    if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                        train_metrics[f"train_{k}"] = float(v)
            elif train_eval is not None:
                primary = experiment.primary_metric or _get_default_metric(automl_task_type)
                train_metrics[f"train_{primary}"] = float(train_eval)

            logger.info(f"Train metrics: {train_metrics}")
        except Exception as train_err:
            logger.warning(f"Could not compute train metrics: {train_err}")

        # Create ModelVersion for the best model - sanitize JSON to handle NaN/Infinity values
        # Include train metrics and dataset size for overfitting detection
        model_version = ModelVersion(
            project_id=experiment.project_id,
            experiment_id=experiment.id,
            trial_id=trial.id,
            name=f"{experiment.name} - {result.best_model_name}",
            model_type=result.best_model_name,
            artifact_location=result.artifact_path,
            metrics_json=sanitize_json_values({
                **result.metrics,
                **train_metrics,  # Train metrics for overfitting detection
                "training_time_seconds": result.training_time_seconds,
                "num_models_trained": result.num_models_trained,
                "validation_samples_count": len(result.validation_predictions),
                "dataset_size": len(training_df),  # Dataset size for reliability
                "holdout_size": len(holdout_df) if holdout_df is not None else 0,
            }),
            feature_importances_json=sanitize_json_values(result.feature_importances),
            serving_config_json=serving_config,
        )
        db.add(model_version)
        db.commit()
        db.refresh(model_version)

        logger.info(f"Created model version: {model_version.id}")

        # Save validation samples for error analysis
        num_samples_saved = 0
        if result.validation_predictions:
            try:
                num_samples_saved = save_validation_samples(
                    db=db,
                    model_version_id=model_version.id,
                    validation_predictions=result.validation_predictions,
                    max_samples=1000,  # Limit to 1000 samples to avoid DB bloat
                )
                logger.info(f"Saved {num_samples_saved} validation samples")
            except Exception as e:
                logger.warning(f"Could not save validation samples: {e}")

        # =====================================================================
        # HOLDOUT EVALUATION: Evaluate on held-out data for true performance
        # =====================================================================
        holdout_score = None
        holdout_validation_result = None

        if holdout_df is not None and len(holdout_df) > 0:
            try:
                logger.info("=" * 60)
                logger.info("🔍 HOLDOUT VALIDATION EVALUATION")
                logger.info("=" * 60)
                logger.info("   Evaluating model on data it has NEVER seen during training...")

                # Load the trained predictor
                from autogluon.tabular import TabularPredictor
                predictor = TabularPredictor.load(result.artifact_path)

                # Evaluate on holdout
                eval_metric = experiment.primary_metric or _get_default_metric(automl_task_type)
                holdout_result = evaluate_on_holdout(
                    predictor=predictor,
                    holdout_df=holdout_df,
                    target_column=target_column,
                    metric_name=eval_metric,
                    task_type=automl_task_type,
                )
                holdout_score = holdout_result["score"]

                # Record score for overfitting tracking across iterations
                holdout_validation_result = record_holdout_score(
                    db=db,
                    experiment=experiment,
                    holdout_score=holdout_score,
                    metric_name=eval_metric,
                )

                # Display results to user
                logger.info(f"   ✓ Holdout {eval_metric}: {holdout_score:.4f}")
                logger.info(f"   ✓ Evaluated on {holdout_result['num_samples']} samples")

                # Compare to CV score
                cv_score = result.metrics.get(eval_metric)
                if cv_score is not None:
                    # Normalize signs for comparison (some metrics are negative)
                    cv_val = abs(cv_score) if cv_score < 0 else cv_score
                    holdout_val = abs(holdout_score) if holdout_score < 0 else holdout_score

                    diff = holdout_val - cv_val
                    diff_pct = (diff / cv_val * 100) if cv_val != 0 else 0

                    logger.info(f"   📊 CV {eval_metric}: {cv_score:.4f}")
                    logger.info(f"   📊 Holdout {eval_metric}: {holdout_score:.4f}")

                    if abs(diff_pct) > 10:
                        logger.warning(f"   ⚠️ SIGNIFICANT GAP: {diff_pct:+.1f}% difference between CV and holdout")
                        logger.warning("      This may indicate overfitting or data leakage!")
                    elif abs(diff_pct) > 5:
                        logger.info(f"   ℹ️ Moderate gap: {diff_pct:+.1f}% difference (within acceptable range)")
                    else:
                        logger.info(f"   ✅ Scores are consistent: {diff_pct:+.1f}% difference (excellent)")

                # Show train vs val comparison for overfitting detection
                train_score_key = f"train_{eval_metric}"
                train_score = train_metrics.get(train_score_key)
                if train_score is not None and cv_score is not None:
                    train_val = abs(train_score) if train_score < 0 else train_score
                    cv_v = abs(cv_score) if cv_score < 0 else cv_score
                    train_diff_pct = ((train_val - cv_v) / cv_v * 100) if cv_v != 0 else 0

                    logger.info(f"   📊 Train {eval_metric}: {train_score:.4f}")
                    logger.info(f"   📊 Val {eval_metric}: {cv_score:.4f}")

                    if train_diff_pct > 15:
                        logger.warning(f"   ⚠️ OVERFITTING WARNING: Train score {train_diff_pct:+.1f}% better than val!")
                    else:
                        logger.info(f"   ✅ Train/Val gap: {train_diff_pct:+.1f}% (acceptable)")

                # Check overfitting status
                if holdout_validation_result:
                    if holdout_validation_result.recommendation == "stop":
                        logger.error(f"   🚨 OVERFITTING DETECTED: Holdout score is degrading!")
                        logger.error(f"      Best holdout score was {holdout_validation_result.best_score:.4f}")
                        logger.error(f"      Current holdout score is {holdout_score:.4f}")
                        logger.error("      Consider stopping iterations and using an earlier model.")
                    elif holdout_validation_result.recommendation == "warning":
                        logger.warning(f"   ⚠️ OVERFITTING WARNING: Holdout performance may be declining")
                        logger.warning(f"      Best: {holdout_validation_result.best_score:.4f}, Current: {holdout_score:.4f}")

                # Update metrics with holdout score
                if model_version.metrics_json:
                    model_version.metrics_json[f"holdout_{eval_metric}"] = holdout_score
                    model_version.metrics_json["holdout_num_samples"] = holdout_result["num_samples"]
                    db.commit()

                logger.info("=" * 60)

            except Exception as holdout_eval_err:
                logger.warning(f"⚠️ Holdout evaluation failed: {holdout_eval_err}")
                logger.warning("   Training completed but true performance unknown")

        # Update experiment status and metadata
        experiment.status = ExperimentStatus.COMPLETED
        # Store detected primary metric if not already set
        if not experiment.primary_metric:
            experiment.primary_metric = _get_default_metric(automl_task_type)
        # Store task type in experiment plan if not present
        if experiment.experiment_plan_json is None:
            experiment.experiment_plan_json = {}
        if "task_type" not in experiment.experiment_plan_json:
            experiment.experiment_plan_json["task_type"] = automl_task_type
        db.commit()

        logger.info(f"Experiment completed: {experiment.name}")

        # Queue critique generation as a follow-up task
        try:
            dispatch_task("generate_training_critique", str(experiment.id), str(trial.id))
            logger.info(f"Queued training critique for experiment {experiment.id}")
        except Exception as critique_error:
            logger.warning(f"Failed to queue training critique: {critique_error}")

        # Queue robustness audit as a follow-up task (Prompt 4 requirement)
        try:
            dispatch_task("run_robustness_audit", str(experiment.id))
            logger.info(f"Queued robustness audit for experiment {experiment.id}")
        except Exception as audit_error:
            logger.warning(f"Failed to queue robustness audit: {audit_error}")

        return {
            "experiment_id": str(experiment.id),
            "trial_id": str(trial.id),
            "model_version_id": str(model_version.id),
            "best_model_name": result.best_model_name,
            "metrics": result.metrics,
            "num_models_trained": result.num_models_trained,
            "training_time_seconds": result.training_time_seconds,
            "resource_limits": applied_limits,
            "backend": "local",
            "validation_samples_saved": num_samples_saved,
            "task_type": automl_task_type,
            "primary_metric": experiment.primary_metric,
            # Holdout validation results
            "holdout_score": holdout_score,
            "holdout_samples": len(holdout_df) if holdout_df is not None else 0,
            "holdout_recommendation": holdout_validation_result.recommendation if holdout_validation_result else None,
        }

    except (DataError, DatasetBuildError) as e:
        logger.error(f"Experiment {experiment_id} data error: {e}")
        try:
            experiment = db.query(Experiment).filter(
                Experiment.id == UUID(experiment_id)
            ).first()
            if experiment:
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = f"Data error: {str(e)}"
                db.commit()
        except Exception as db_err:
            logger.warning(f"Failed to update experiment status: {db_err}")
        raise AutoMLError(experiment_id, f"Dataset build failed: {e}")

    except TrainingError as e:
        logger.error(f"Experiment {experiment_id} training error: {e}")
        try:
            experiment = db.query(Experiment).filter(
                Experiment.id == UUID(experiment_id)
            ).first()
            if experiment:
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = f"Training error: {str(e)}"
                db.commit()
        except Exception as db_err:
            logger.warning(f"Failed to update experiment status: {db_err}")
        raise

    except Exception as e:
        logger.exception(f"Experiment {experiment_id} unexpected failure: {e}")
        try:
            experiment = db.query(Experiment).filter(
                Experiment.id == UUID(experiment_id)
            ).first()
            if experiment:
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = str(e)
                db.commit()
        except Exception as db_err:
            logger.warning(f"Failed to update experiment status: {db_err}")
        raise

    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.run_experiment_modal")
def run_experiment_modal(
    self,
    experiment_id: str,
) -> dict:
    """Run an ML experiment on Modal.com cloud.

    This task:
    1. Loads the experiment and dataset spec
    2. Builds the dataset from data sources
    3. Submits to Modal for cloud training (no resource limits)
    4. Creates a Trial and ModelVersion with results

    Args:
        experiment_id: UUID of the experiment to run

    Returns:
        Dictionary with experiment results
    """
    print(f"🌩️ run_experiment_modal STARTED: experiment_id={experiment_id}")
    logger.info(f"🌩️ run_experiment_modal STARTED: experiment_id={experiment_id}")

    import asyncio
    from app.services.modal_runner import run_experiment_on_modal, is_modal_configured

    print(f"🌩️ Checking Modal configuration...")
    if not is_modal_configured():
        print(f"❌ Modal is NOT configured!")
        raise ValueError("Modal is not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env")
    print(f"✅ Modal is configured")

    settings = get_settings()
    db = SessionLocal()

    try:
        experiment_uuid = UUID(experiment_id)

        # Load experiment
        experiment = db.query(Experiment).filter(Experiment.id == experiment_uuid).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        logger.info(f"Starting Modal experiment: {experiment.name} ({experiment_id})")

        # Update experiment status to running
        experiment.status = ExperimentStatus.RUNNING
        db.commit()

        # Validate experiment has dataset_spec
        if not experiment.dataset_spec_id:
            raise ValueError(f"Experiment {experiment_id} has no dataset_spec_id configured")

        # Build the dataset
        builder = DatasetBuilder(db)
        dataset = builder.build_dataset_from_spec(experiment.dataset_spec_id)

        # =====================================================================
        # FEATURE ENGINEERING WARNING STORAGE: Persist any failures for UI visibility
        # =====================================================================
        fe_failures = dataset.attrs.get('_feature_engineering_failures', [])
        if fe_failures:
            logger.warning(f"⚠️ {len(fe_failures)} feature engineering step(s) FAILED!")
            for failure in fe_failures:
                logger.warning(f"   - {failure.get('feature', 'unknown')}: {failure.get('error', 'unknown error')}")

            # Store in experiment for UI visibility
            if experiment.experiment_plan_json is None:
                experiment.experiment_plan_json = {}
            experiment.experiment_plan_json['feature_engineering_warnings'] = [
                f"⚠️ Feature '{f.get('feature', 'unknown')}' FAILED: {f.get('error', 'unknown error')}"
                for f in fe_failures
            ]
            experiment.experiment_plan_json['feature_engineering_failure_count'] = len(fe_failures)
            experiment.experiment_plan_json['feature_engineering_success_count'] = dataset.attrs.get('_feature_engineering_success_count', 0)
            db.commit()
            logger.info(f"Stored {len(fe_failures)} feature engineering warnings in experiment record")

        # Get dataset spec for target column
        dataset_spec = experiment.dataset_spec
        if not dataset_spec:
            raise ValueError(f"DatasetSpec {experiment.dataset_spec_id} not found")

        target_column = dataset_spec.target_column
        if not target_column:
            raise ValueError(f"DatasetSpec {dataset_spec.id} has no target_column configured")

        # Update dataset spec with actual row count and feature columns if not set
        # This ensures future critiques have this info without reloading the dataset
        spec_updated = False
        spec_json = dataset_spec.spec_json or {}
        if not spec_json.get("row_count"):
            spec_json["row_count"] = len(dataset)
            spec_updated = True
        if not dataset_spec.feature_columns:
            feature_cols = [c for c in dataset.columns if c != target_column]
            dataset_spec.feature_columns = feature_cols
            spec_updated = True
        if spec_updated:
            dataset_spec.spec_json = spec_json
            db.commit()
            logger.info(f"Updated dataset spec with row_count={len(dataset)}, features={len(dataset_spec.feature_columns or [])}")

        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Handle rows with NaN/Inf in target column
        # Note: ML training REQUIRES known target values - rows with NaN targets cannot be used
        import numpy as np
        original_len = len(dataset)
        target_invalid = dataset[target_column].isna() | dataset[target_column].isin([np.inf, -np.inf])
        invalid_count = int(target_invalid.sum())

        if invalid_count > 0:
            invalid_pct = (invalid_count / original_len) * 100

            # Check if this looks like a shifted/lagged target (NaN at end of groups or series)
            # vs a data quality problem
            if invalid_pct > 20:
                # More than 20% invalid - this is likely a data problem, not just shifted targets
                raise ValueError(
                    f"Target column '{target_column}' has {invalid_count:,} invalid values ({invalid_pct:.1f}% of data). "
                    f"This is too high and indicates a data quality issue. "
                    f"Please check your feature engineering or data preparation. "
                    f"If this target is intentionally computed (e.g., future returns), ensure the formula is correct."
                )
            elif invalid_pct > 5:
                # 5-20% invalid - warn prominently but proceed
                logger.warning(f"⚠️ HIGH NaN RATE: {invalid_count:,} rows ({invalid_pct:.1f}%) have invalid target values")
                logger.warning(f"   This may indicate a data quality issue with '{target_column}'")
                logger.warning(f"   Common causes: incorrect feature engineering, missing data, or division by zero")
                logger.warning(f"   Proceeding with {original_len - invalid_count:,} valid rows...")
            else:
                # <5% invalid - likely just shifted targets (expected)
                logger.info(f"Removed {invalid_count:,} rows ({invalid_pct:.1f}%) with invalid target values (expected for shifted/lagged targets)")

            dataset = dataset[~target_invalid].copy()

        # Determine task type - try project setting first, then auto-detect from data
        project = experiment.project
        task_type = project.task_type.value if project.task_type else None

        # Auto-detect task type from target column if not explicitly set
        if not task_type or task_type == "classification":
            target_series = dataset[target_column]
            automl_task_type = _detect_task_type(target_series)
            logger.info(f"Auto-detected task type: {automl_task_type} from target column '{target_column}'")
        else:
            # Map task types
            task_type_map = {
                "classification": "binary",
                "binary": "binary",
                "multiclass": "multiclass",
                "regression": "regression",
                "quantile": "quantile",
            }
            automl_task_type = task_type_map.get(task_type, "binary")

        # =====================================================================
        # HOLDOUT VALIDATION: Create persistent holdout set for overfitting detection
        # =====================================================================
        holdout_df = None
        training_df = dataset  # Default to full dataset if holdout creation fails
        validation_strategy = spec_json.get("validation_strategy", {})

        try:
            logger.info("=" * 60)
            logger.info("🔒 HOLDOUT VALIDATION SETUP (Modal)")
            logger.info("=" * 60)

            training_df, holdout_df, holdout_indices = get_or_create_holdout_indices(
                df=dataset,
                dataset_spec=dataset_spec,
                db=db,
                target_column=target_column,
                task_type=automl_task_type,
                validation_strategy=validation_strategy,
            )

            holdout_pct = len(holdout_df) / len(dataset) * 100
            logger.info(f"   ✓ Created holdout set: {len(holdout_df):,} samples ({holdout_pct:.1f}%)")
            logger.info(f"   ✓ Training set: {len(training_df):,} samples ({100-holdout_pct:.1f}%)")
            logger.info(f"   ✓ Split strategy: {validation_strategy.get('split_strategy', 'default')}")
            logger.info("   📌 Holdout data will be sent to Modal for evaluation after training")
            logger.info("=" * 60)

        except Exception as holdout_err:
            logger.warning(f"⚠️ Could not create holdout set: {holdout_err}")
            logger.warning("   Proceeding with full dataset for Modal training (no overfitting detection)")
            training_df = dataset
            holdout_df = None

        # Determine primary metric
        primary_metric = experiment.primary_metric or None

        # Get AutoML config - use settings as defaults (same as local execution)
        # This ensures agent-specified configs are respected regardless of execution environment
        automl_config = {
            "time_limit": settings.automl_time_limit,
            "presets": settings.automl_presets,
        }

        # Override with experiment plan config if present (agent config takes priority)
        if experiment.experiment_plan_json:
            plan_config = experiment.experiment_plan_json.get("automl_config", {})
            automl_config.update(plan_config)

        # Normalize: convert max_runtime_seconds to time_limit if needed
        if "max_runtime_seconds" in automl_config:
            automl_config["time_limit"] = automl_config.pop("max_runtime_seconds")
            logger.info(f"Converted max_runtime_seconds to time_limit: {automl_config['time_limit']}s")

        logger.info(f"AutoML config for Modal experiment: time_limit={automl_config.get('time_limit')}s, presets={automl_config.get('presets')}")

        # Create trial record
        trial = Trial(
            experiment_id=experiment.id,
            variant_name="AutoML_Modal_Cloud",
            data_split_strategy="holdout_15pct" if holdout_df is not None else "random_80_20",
            automl_config=automl_config,
            status=TrialStatus.RUNNING,
        )
        db.add(trial)
        db.commit()
        db.refresh(trial)

        logger.info(f"Created trial: {trial.id}, submitting to Modal cloud...")

        # Clear any old logs and send initial progress update via Redis log store
        from app.services.training_logs import TrainingLogStore
        log_store = TrainingLogStore(str(experiment.id))
        log_store.clear()  # Clear any stale logs from previous runs
        log_store.add_milestone("Submitting to Modal cloud...")
        log_store.add_log(
            f"Dataset: {len(dataset)} rows, {len(dataset.columns)} columns",
            log_type="info",
            interpreted=f"Your dataset has {len(dataset):,} rows and {len(dataset.columns)} columns."
        )
        if holdout_df is not None:
            log_store.add_log(
                f"Holdout: {len(holdout_df)} samples reserved for final validation",
                log_type="info",
                interpreted=f"🔒 {len(holdout_df):,} samples ({len(holdout_df)/len(dataset)*100:.1f}%) held out locally for overfitting detection."
            )
            log_store.add_log(
                f"Sending to Modal: {len(training_df)} samples for training",
                log_type="info",
                interpreted=f"Training on Modal will use {len(training_df):,} samples (holdout kept locally)."
            )
        log_store.add_log(
            f"Target: {target_column}, Task: {automl_task_type}",
            log_type="info",
            interpreted=f"Predicting '{target_column}' using {automl_task_type} approach."
        )
        log_store.add_log(
            f"Time limit: {automl_config.get('time_limit', 300)}s",
            log_type="info",
            interpreted=f"Training will run for up to {automl_config.get('time_limit', 300) // 60} minutes."
        )
        log_store.add_milestone("Waiting for Modal container to start...")
        log_store.add_log(
            "Training is running in the cloud. Logs will appear when complete.",
            log_type="progress",
            interpreted="Training is running in the cloud. Detailed logs will appear when training completes."
        )

        # Run on Modal - this is an async function, run it synchronously
        # Create a new event loop to avoid "Event loop is closed" errors
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            run_experiment_on_modal(
                experiment_id=str(experiment.id),
                dataset=training_df,  # Send only training data (excludes holdout)
                target_column=target_column,
                task_type=automl_task_type,
                primary_metric=primary_metric,
                config=automl_config,
                holdout_df=holdout_df,  # Send holdout for evaluation on Modal
                download_model=False,  # Keep model on Modal - predictions run via /predict-auto
            )
        )

        if not result.get("success"):
            log_store.add_log(
                "Modal training failed!",
                log_type="error",
                interpreted="Training failed in the cloud. Check the error details above."
            )
            raise ValueError(f"Modal training failed: {result}")

        # Stream the captured logs from Modal
        # These will be interpreted when the frontend polls for logs
        training_logs = result.get("training_logs", "")
        if training_logs:
            log_store.add_milestone("Training complete! Processing results...")
            for line in training_logs.split("\n"):
                if line.strip():
                    log_store.add_raw_log(line)

        best_model = result.get('best_model_name', 'Unknown')
        log_store.add_log(
            f"Best model: {best_model}",
            log_type="milestone",
            interpreted=f"Training complete! The best performing model is {best_model}."
        )

        # Update trial with results - sanitize JSON to handle NaN/Infinity values
        # Include ALL metrics (train, holdout, dataset_size) for critique analysis
        trial_metrics = {
            **result.get("metrics", {}),
            **result.get("train_metrics", {}),  # Train metrics for overfitting detection
            **result.get("holdout_metrics", {}),  # Holdout metrics
            "training_time_seconds": result.get("training_time_seconds"),
            "num_models_trained": result.get("num_models_trained"),
            "dataset_size": result.get("dataset_size", len(training_df)),
            "holdout_size": result.get("holdout_size", len(holdout_df) if holdout_df is not None else 0),
        }
        trial.status = TrialStatus.COMPLETED
        trial.metrics_json = sanitize_json_values(trial_metrics)
        trial.best_model_ref = result.get("best_model_name")
        trial.logs_location = result.get("artifact_path")
        trial.training_logs = result.get("training_logs", "")  # Captured training output
        trial.leaderboard_json = sanitize_json_values(result.get("leaderboard", []))  # Model leaderboard

        # Compute baseline metrics for Modal training (computed locally)
        try:
            from sklearn.model_selection import train_test_split
            is_classification = automl_task_type in ("binary", "multiclass")
            train_split, val_split = train_test_split(
                training_df,
                test_size=0.2,
                random_state=42,
                stratify=training_df[target_column] if is_classification else None
            )
            baseline_metrics = compute_all_baselines(
                train_data=train_split,
                val_data=val_split,
                target_column=target_column,
                task_type=automl_task_type,
                run_shuffle_test=True,
            )
            if baseline_metrics:
                trial.baseline_metrics_json = sanitize_json_values(baseline_metrics)
                logger.info(f"Stored baseline metrics for Modal: {list(baseline_metrics.keys())}")
        except Exception as e:
            logger.warning(f"Failed to compute baseline metrics for Modal training: {e}")

        db.commit()

        logger.info(f"Modal trial completed: best_model={result.get('best_model_name')}")

        # Build serving config
        feature_columns = list(dataset.columns)
        if target_column in feature_columns:
            feature_columns.remove(target_column)

        features = []
        for col in feature_columns:
            col_dtype = str(dataset[col].dtype)
            if col_dtype in ['int64', 'int32', 'float64', 'float32']:
                feat_type = 'numeric'
            elif col_dtype == 'bool':
                feat_type = 'boolean'
            else:
                feat_type = 'categorical'
            features.append({'name': col, 'type': feat_type})

        serving_config = {
            'features': features,
            'target_column': target_column,
            'task_type': automl_task_type,
        }

        # =====================================================================
        # EXTRACT MODEL ARCHIVE: Save the Modal-trained model locally
        # =====================================================================
        local_artifact_path = None
        model_archive_b64 = result.get("model_archive_b64")

        if model_archive_b64:
            try:
                import base64
                import tarfile
                from io import BytesIO
                from pathlib import Path

                # Create local artifacts directory
                backend_dir = Path(__file__).parent.parent.parent  # backend/
                local_artifact_dir = backend_dir / "artifacts" / str(experiment.id)
                local_artifact_dir.mkdir(parents=True, exist_ok=True)

                # Decode and extract the archive
                archive_bytes = base64.b64decode(model_archive_b64)
                archive_buffer = BytesIO(archive_bytes)
                archive_size_mb = len(archive_bytes) / (1024 * 1024)

                logger.info(f"Extracting model archive ({archive_size_mb:.2f} MB) to {local_artifact_dir}")
                log_store.add_log(
                    f"Downloading model from cloud ({archive_size_mb:.1f} MB)...",
                    log_type="progress",
                    interpreted=f"Saving trained model to local storage ({archive_size_mb:.1f} MB)."
                )

                with tarfile.open(fileobj=archive_buffer, mode="r:gz") as tar:
                    tar.extractall(path=str(local_artifact_dir))

                local_artifact_path = str(local_artifact_dir)
                logger.info(f"Model extracted successfully to {local_artifact_path}")
                log_store.add_log(
                    f"Model saved to {local_artifact_path}",
                    log_type="milestone",
                    interpreted="Model saved locally and ready for predictions!"
                )

            except Exception as extract_err:
                logger.error(f"Failed to extract model archive: {extract_err}")
                log_store.add_log(
                    f"Warning: Could not save model locally: {extract_err}",
                    log_type="warning",
                    interpreted="Model trained but could not be saved locally. Predictions may not work."
                )
        else:
            # Model download was skipped for faster training
            logger.info("Model download skipped (download_model=False) - use Download button to get model")
            log_store.add_log(
                "Model download skipped for faster results",
                log_type="info",
                interpreted="Training complete! Use the Download Model button on the model page to get the model for predictions."
            )

        # Create ModelVersion - sanitize JSON to handle NaN/Infinity values
        # Include train metrics, holdout metrics, and dataset size for overfitting detection
        model_downloaded = result.get("model_downloaded", local_artifact_path is not None)
        metrics_to_save = {
            **result.get("metrics", {}),
            **result.get("train_metrics", {}),  # Train metrics for overfitting detection
            **result.get("holdout_metrics", {}),  # Holdout metrics
            "training_time_seconds": result.get("training_time_seconds"),
            "num_models_trained": result.get("num_models_trained"),
            "dataset_size": result.get("dataset_size", len(training_df)),  # Dataset size for reliability
            "holdout_size": result.get("holdout_size", len(holdout_df) if holdout_df is not None else 0),
            "model_downloaded": model_downloaded,  # Whether model artifacts are available locally
        }

        # Use local artifact path if available, otherwise fall back to Modal path (won't work for predictions)
        artifact_location = local_artifact_path or result.get("artifact_path")

        model_version = ModelVersion(
            project_id=experiment.project_id,
            experiment_id=experiment.id,
            trial_id=trial.id,
            name=f"{experiment.name} - {result.get('best_model_name')} (Modal)",
            model_type=result.get("best_model_name"),
            artifact_location=artifact_location,
            metrics_json=sanitize_json_values(metrics_to_save),
            feature_importances_json=sanitize_json_values(result.get("feature_importances", {})),
            serving_config_json=serving_config,
        )
        db.add(model_version)
        db.commit()
        db.refresh(model_version)

        logger.info(f"Created model version: {model_version.id}")

        # =====================================================================
        # HOLDOUT EVALUATION: Use holdout score from Modal (evaluated on Modal)
        # =====================================================================
        holdout_score = result.get("holdout_score")  # Get holdout score evaluated on Modal
        holdout_validation_result = None

        if holdout_score is not None:
            try:
                logger.info("=" * 60)
                logger.info("🔍 HOLDOUT VALIDATION RESULTS (from Modal)")
                logger.info("=" * 60)

                eval_metric = experiment.primary_metric or _get_default_metric(automl_task_type)
                holdout_size = result.get("holdout_size", len(holdout_df) if holdout_df is not None else 0)

                # Display results to user
                logger.info(f"   ✓ Holdout {eval_metric}: {holdout_score:.4f}")
                logger.info(f"   ✓ Evaluated on {holdout_size} samples (on Modal)")

                # Record score for overfitting tracking across iterations
                holdout_validation_result = record_holdout_score(
                    db=db,
                    experiment=experiment,
                    holdout_score=holdout_score,
                    metric_name=eval_metric,
                )

                # Compare to CV/Val score
                cv_score = result.get("metrics", {}).get(eval_metric) or result.get("metrics", {}).get("score_val")
                if cv_score is not None:
                    cv_val = abs(cv_score) if cv_score < 0 else cv_score
                    holdout_val = abs(holdout_score) if holdout_score < 0 else holdout_score
                    diff_pct = ((holdout_val - cv_val) / cv_val * 100) if cv_val != 0 else 0

                    logger.info(f"   📊 CV/Val {eval_metric}: {cv_score:.4f}")
                    logger.info(f"   📊 Holdout {eval_metric}: {holdout_score:.4f}")

                    if abs(diff_pct) > 10:
                        logger.warning(f"   ⚠️ SIGNIFICANT GAP: {diff_pct:+.1f}% between CV and holdout!")
                    elif abs(diff_pct) > 5:
                        logger.info(f"   ℹ️ Moderate gap: {diff_pct:+.1f}% difference")
                    else:
                        logger.info(f"   ✅ Scores consistent: {diff_pct:+.1f}% difference")

                # Show train vs val comparison for overfitting detection
                train_metrics = result.get("train_metrics", {})
                train_score_key = f"train_{eval_metric}"
                train_score = train_metrics.get(train_score_key)
                if train_score is not None and cv_score is not None:
                    train_val = abs(train_score) if train_score < 0 else train_score
                    cv_val = abs(cv_score) if cv_score < 0 else cv_score
                    train_diff_pct = ((train_val - cv_val) / cv_val * 100) if cv_val != 0 else 0

                    logger.info(f"   📊 Train {eval_metric}: {train_score:.4f}")
                    logger.info(f"   📊 Val {eval_metric}: {cv_score:.4f}")

                    if train_diff_pct > 15:
                        logger.warning(f"   ⚠️ OVERFITTING WARNING: Train score {train_diff_pct:+.1f}% better than val!")
                    else:
                        logger.info(f"   ✅ Train/Val gap: {train_diff_pct:+.1f}% (acceptable)")

                # Check overfitting status from history
                if holdout_validation_result:
                    if holdout_validation_result.recommendation == "stop":
                        logger.error(f"   🚨 OVERFITTING DETECTED: Holdout score degrading!")
                        logger.error(f"      Best: {holdout_validation_result.best_score:.4f}, Current: {holdout_score:.4f}")
                    elif holdout_validation_result.recommendation == "warning":
                        logger.warning(f"   ⚠️ Overfitting warning: Performance may be declining")

                logger.info("=" * 60)

            except Exception as holdout_eval_err:
                logger.warning(f"⚠️ Error processing holdout results: {holdout_eval_err}")
        elif holdout_df is not None and len(holdout_df) > 0:
            logger.warning("⚠️ Holdout data was sent but no holdout score returned from Modal")
            logger.warning("   This may indicate an evaluation error on Modal")

        # Update experiment status and metadata
        experiment.status = ExperimentStatus.COMPLETED
        # Store detected primary metric if not already set
        if not experiment.primary_metric:
            experiment.primary_metric = _get_default_metric(automl_task_type)
        # Store task type in experiment plan if not present
        if experiment.experiment_plan_json is None:
            experiment.experiment_plan_json = {}
        if "task_type" not in experiment.experiment_plan_json:
            experiment.experiment_plan_json["task_type"] = automl_task_type

        # Store experiment results for ensemble and analysis
        experiment.results_json = {
            "holdout_score": holdout_score,
            "volume_model_path": result.get("volume_model_path"),
            "model_saved_to_volume": result.get("model_saved_to_volume", False),
            "best_model_name": result.get("best_model_name"),
            "metrics": result.get("metrics", {}),
            "train_metrics": result.get("train_metrics", {}),
            "training_time_seconds": result.get("training_time_seconds"),
            "task_type": automl_task_type,
            "backend": "modal",
        }

        db.commit()

        logger.info(f"Modal experiment completed: {experiment.name}")

        # Queue critique generation as a follow-up task
        try:
            dispatch_task("generate_training_critique", str(experiment.id), str(trial.id))
            logger.info(f"Queued training critique for Modal experiment {experiment.id}")
        except Exception as critique_error:
            logger.warning(f"Failed to queue training critique: {critique_error}")

        # Queue robustness audit as a follow-up task (Prompt 4 requirement)
        try:
            dispatch_task("run_robustness_audit", str(experiment.id))
            logger.info(f"Queued robustness audit for Modal experiment {experiment.id}")
        except Exception as audit_error:
            logger.warning(f"Failed to queue robustness audit: {audit_error}")

        # Auto-iterate: automatically trigger AI feedback and next iteration
        if experiment.auto_iterate_enabled and experiment.iteration_number < experiment.auto_iterate_max:
            logger.info("=" * 60)
            logger.info("🔄 AUTO-ITERATE ENABLED")
            logger.info(f"   Current iteration: {experiment.iteration_number}")
            logger.info(f"   Max iterations: {experiment.auto_iterate_max}")
            logger.info("   Automatically triggering improvement pipeline...")
            logger.info("=" * 60)
            try:
                # Queue the auto-improve pipeline task
                dispatch_task("run_auto_improve_pipeline", str(experiment.id), use_enhanced_pipeline=True)
                logger.info(f"Queued auto-improve pipeline for experiment {experiment.id}")
            except Exception as auto_iter_error:
                logger.warning(f"Failed to queue auto-improve pipeline: {auto_iter_error}")
        elif experiment.auto_iterate_enabled:
            logger.info("=" * 60)
            logger.info("🛑 AUTO-ITERATE LIMIT REACHED")
            logger.info(f"   Completed iteration: {experiment.iteration_number}")
            logger.info(f"   Max iterations: {experiment.auto_iterate_max}")
            logger.info("   No more auto-iterations will be triggered.")
            logger.info("=" * 60)

        return {
            "experiment_id": str(experiment.id),
            "trial_id": str(trial.id),
            "model_version_id": str(model_version.id),
            "best_model_name": result.get("best_model_name"),
            "metrics": result.get("metrics"),
            "num_models_trained": result.get("num_models_trained"),
            "training_time_seconds": result.get("training_time_seconds"),
            "backend": "modal",
            "task_type": automl_task_type,
            "primary_metric": experiment.primary_metric,
            # Holdout validation results
            "holdout_score": holdout_score,
            "holdout_samples": len(holdout_df) if holdout_df is not None else 0,
            "holdout_recommendation": holdout_validation_result.recommendation if holdout_validation_result else None,
        }

    except (DataError, DatasetBuildError) as e:
        logger.error(f"Modal experiment {experiment_id} data error: {e}")
        try:
            experiment = db.query(Experiment).filter(
                Experiment.id == UUID(experiment_id)
            ).first()
            if experiment:
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = f"Data error: {str(e)}"
                db.commit()
        except Exception as db_err:
            logger.warning(f"Failed to update experiment status: {db_err}")
        raise AutoMLError(experiment_id, f"Dataset build failed: {e}")

    except TrainingError as e:
        logger.error(f"Modal experiment {experiment_id} training error: {e}")
        try:
            experiment = db.query(Experiment).filter(
                Experiment.id == UUID(experiment_id)
            ).first()
            if experiment:
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = f"Training error: {str(e)}"
                db.commit()
        except Exception as db_err:
            logger.warning(f"Failed to update experiment status: {db_err}")
        raise

    except Exception as e:
        logger.exception(f"Modal experiment {experiment_id} unexpected failure: {e}")
        try:
            experiment = db.query(Experiment).filter(
                Experiment.id == UUID(experiment_id)
            ).first()
            if experiment:
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = str(e)
                db.commit()
        except Exception as db_err:
            logger.warning(f"Failed to update experiment status: {db_err}")
        raise

    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.generate_training_critique")
def generate_training_critique(
    self,
    experiment_id: str,
    trial_id: str,
) -> dict:
    """Generate AI critique of training results.

    This task runs after experiment completion to analyze training logs
    and provide improvement suggestions.

    Args:
        experiment_id: UUID of the experiment
        trial_id: UUID of the trial to critique

    Returns:
        Dictionary with critique results
    """
    import asyncio
    from uuid import UUID

    from app.services.prompts import get_training_critique_prompt
    from app.services.llm_client import get_llm_client
    from app.schemas.agent import TrainingCritiqueResult
    from app.models.model_version import ModelVersion

    db = SessionLocal()

    try:
        experiment_uuid = UUID(experiment_id)
        trial_uuid = UUID(trial_id)

        # Get experiment
        experiment = db.query(Experiment).filter(Experiment.id == experiment_uuid).first()
        if not experiment:
            logger.warning(f"Critique: Experiment {experiment_id} not found")
            return {"success": False, "error": "Experiment not found"}

        # Get trial
        trial = db.query(Trial).filter(Trial.id == trial_uuid).first()
        if not trial:
            logger.warning(f"Critique: Trial {trial_id} not found")
            return {"success": False, "error": "Trial not found"}

        logger.info(f"Generating training critique for experiment {experiment.name}")

        # Get model version for feature importances
        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.experiment_id == experiment_uuid)
            .order_by(ModelVersion.created_at.desc())
            .first()
        )

        # Build leaderboard summary
        leaderboard = trial.leaderboard_json or []
        leaderboard_lines = []
        for model in leaderboard[:10]:  # Top 10 models
            name = model.get("model", "Unknown")
            score = model.get("score_val")
            fit_time = model.get("fit_time")
            # Handle None values for score and fit_time
            score_str = f"{score:.4f}" if score is not None else "N/A"
            fit_time_str = f"{fit_time:.1f}s" if fit_time is not None else "N/A"
            leaderboard_lines.append(f"  - {name}: score={score_str}, fit_time={fit_time_str}")
        leaderboard_summary = "\n".join(leaderboard_lines) if leaderboard_lines else "(no leaderboard data)"

        # Get dataset info - try to load actual data if not in spec
        dataset_spec = experiment.dataset_spec
        feature_columns = dataset_spec.feature_columns if dataset_spec else []
        target_column = dataset_spec.target_column if dataset_spec else "unknown"
        row_count = 0

        # First try to get from spec_json
        if dataset_spec and dataset_spec.spec_json:
            row_count = dataset_spec.spec_json.get("row_count", 0)

        # If row_count is 0 or feature_columns is empty, try loading actual dataset
        if (row_count == 0 or not feature_columns) and dataset_spec:
            try:
                from app.tasks.automl import load_dataset_from_spec
                logger.info("Loading actual dataset for critique context...")
                df = load_dataset_from_spec(db, dataset_spec)
                row_count = len(df)
                if not feature_columns:
                    # Get feature columns from actual data (exclude target)
                    feature_columns = [c for c in df.columns if c != target_column]
                logger.info(f"Loaded dataset: {row_count} rows, {len(feature_columns)} features")
            except Exception as e:
                logger.warning(f"Could not load dataset for critique context: {e}")

        dataset_shape = f"{row_count} rows x {len(feature_columns)} features"

        # Get metrics
        metrics = trial.metrics_json or {}
        primary_metric = experiment.primary_metric or "score_val"
        best_score = metrics.get(primary_metric, metrics.get("score_val", 0))

        # Extract train metrics for overfitting analysis
        train_score = metrics.get(f"train_{primary_metric}")
        holdout_score = metrics.get(f"holdout_{primary_metric}")

        # Extract dataset size for reliability assessment
        dataset_size = metrics.get("dataset_size", row_count)
        holdout_size = metrics.get("holdout_size", 0)

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

        # Generate prompt with overfitting and reliability info
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
            train_score=float(train_score) if train_score is not None else None,
            holdout_score=float(holdout_score) if holdout_score is not None else None,
            dataset_size=int(dataset_size) if dataset_size else None,
            holdout_size=int(holdout_size) if holdout_size else None,
        )

        # Call LLM (run async in sync context)
        async def get_critique():
            llm_client = get_llm_client()
            messages = [
                {"role": "system", "content": "You are an expert ML engineer providing actionable feedback on AutoML training results. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ]
            return await llm_client.chat_json(messages, TrainingCritiqueResult)

        result = asyncio.get_event_loop().run_until_complete(get_critique())

        # Validate and fix result
        if isinstance(result, dict):
            critique = TrainingCritiqueResult(**result)
        else:
            critique = result

        critique_dict = critique.model_dump()

        # Save critique to trial
        trial.critique_json = critique_dict
        db.commit()

        logger.info(f"Critique generated for experiment {experiment.name}: {critique.performance_rating}")

        return {
            "success": True,
            "experiment_id": experiment_id,
            "trial_id": trial_id,
            "performance_rating": critique.performance_rating,
            "summary": critique.summary,
        }

    except Exception as e:
        logger.error(f"Failed to generate critique for experiment {experiment_id}: {e}")
        return {"success": False, "error": str(e)}

    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.run_robustness_audit")
def run_robustness_audit(
    self,
    experiment_id: str,
) -> dict:
    """Run robustness audit for a completed experiment.

    This task runs automatically after experiment completion to:
    1. Check for overfitting using train/val gap analysis
    2. Check for data leakage using label-shuffle test results
    3. Check for time-split issues if using time-based data
    4. Compare model performance against baselines
    5. Generate warnings and recommendations

    Args:
        experiment_id: UUID of the experiment to audit

    Returns:
        Dictionary with robustness audit results
    """
    import asyncio
    from datetime import datetime
    from uuid import UUID

    from app.models.agent_run import AgentRun, AgentStep, AgentRunStatus, AgentStepStatus, AgentStepType
    from app.services.agent_executor import handle_robustness_audit_step
    from app.services.llm_client import get_llm_client
    from app.services.agents.utils.step_logger import StepLogger
    from app.services.task_context import build_task_context

    logger.info(f"Starting robustness audit for experiment {experiment_id}")

    db = SessionLocal()

    try:
        experiment_uuid = UUID(experiment_id)

        # Get experiment
        experiment = db.query(Experiment).filter(Experiment.id == experiment_uuid).first()
        if not experiment:
            logger.warning(f"Robustness audit: Experiment {experiment_id} not found")
            return {"success": False, "error": "Experiment not found"}

        project = experiment.project
        if not project:
            logger.warning(f"Robustness audit: Project not found for experiment {experiment_id}")
            return {"success": False, "error": "Project not found"}

        # Get dataset spec for is_time_based
        is_time_based = False
        if experiment.dataset_spec_id:
            from app.models.dataset_spec import DatasetSpec
            dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == experiment.dataset_spec_id).first()
            if dataset_spec:
                is_time_based = dataset_spec.is_time_based or False

        logger.info(f"Running robustness audit for experiment '{experiment.name}' (time_based={is_time_based})")

        # Build task_context for the audit step (Prompt 7 Step 2)
        try:
            task_context = build_task_context(
                db=db,
                project_id=str(project.id),
                experiment_id=str(experiment.id),
                include_leakage_candidates=True,
                include_past_cycles=True,
                max_experiments=5,
            )
        except Exception as e:
            logger.warning(f"Could not build task_context for robustness audit: {e}")
            task_context = None

        # Create an agent run for the audit
        agent_run = AgentRun(
            project_id=project.id,
            name=f"Robustness Audit - {experiment.name}",
            description=f"Automated robustness audit for experiment {experiment.name}",
            status=AgentRunStatus.RUNNING,
        )
        db.add(agent_run)
        db.commit()

        # Create the robustness audit step input
        step_input_json = {
            "project_id": str(project.id),
            "experiment_id": str(experiment.id),
            "task_type": project.task_type or "binary",
            "is_time_based": is_time_based,
            "primary_metric": experiment.primary_metric,
        }
        # Include task_context for context persistence (Prompt 7 Step 2)
        if task_context:
            step_input_json["task_context"] = task_context

        # Create the robustness audit step
        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            status=AgentStepStatus.PENDING,
            input_json=step_input_json,
        )
        db.add(step)
        db.commit()

        # Run the audit
        step_logger = StepLogger(db, step.id)
        llm_client = get_llm_client()

        async def run_audit():
            return await handle_robustness_audit_step(db, step, step_logger, llm_client)

        result = asyncio.get_event_loop().run_until_complete(run_audit())

        # Update step status
        step.status = AgentStepStatus.COMPLETED
        step.finished_at = datetime.utcnow()
        step.output_json = result
        db.commit()

        # Update agent run status
        agent_run.status = AgentRunStatus.COMPLETED
        agent_run.finished_at = datetime.utcnow()
        agent_run.result_json = result
        db.commit()

        # Extract key results
        audit_result = result.get("robustness_audit", {})
        overfitting_risk = audit_result.get("overfitting_risk", "unknown")
        leakage_suspected = audit_result.get("leakage_suspected", False)
        time_split_suspicious = audit_result.get("time_split_suspicious", False)
        warnings = audit_result.get("warnings", [])

        # Log summary
        logger.info(f"Robustness audit completed for {experiment.name}:")
        logger.info(f"  - Overfitting risk: {overfitting_risk}")
        logger.info(f"  - Leakage suspected: {leakage_suspected}")
        logger.info(f"  - Time-split suspicious: {time_split_suspicious}")
        if warnings:
            logger.warning(f"  - Warnings: {len(warnings)}")
            for w in warnings[:3]:  # Log first 3 warnings
                logger.warning(f"    * {w[:100]}...")

        return {
            "success": True,
            "experiment_id": experiment_id,
            "agent_run_id": str(agent_run.id),
            "step_id": str(step.id),
            "overfitting_risk": overfitting_risk,
            "leakage_suspected": leakage_suspected,
            "time_split_suspicious": time_split_suspicious,
            "warnings_count": len(warnings),
        }

    except Exception as e:
        logger.error(f"Failed to run robustness audit for experiment {experiment_id}: {e}")
        return {"success": False, "error": str(e)}

    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.run_auto_improve_pipeline")
def run_auto_improve_pipeline(
    self,
    experiment_id: str,
    use_enhanced_pipeline: bool = True,
) -> dict:
    """Run the auto-improve pipeline for an experiment.

    This task runs the ENHANCED improvement pipeline which:
    1. Gathers complete iteration context (all history, errors, metrics)
    2. Re-analyzes data with full iteration feedback
    3. Redesigns dataset with knowledge of what worked/failed
    4. Designs new experiment configuration
    5. Creates new DatasetSpec with improved features
    6. Creates new Experiment (iteration N+1)
    7. Queues the new experiment for training on Modal

    The enhanced pipeline uses specialized agents that have access to
    the complete history of all iterations to make smarter decisions.

    Args:
        experiment_id: UUID of the completed experiment to improve
        use_enhanced_pipeline: If True, use the full agent pipeline (default)

    Returns:
        Dictionary with the new experiment ID and improvement details
    """
    import asyncio
    from datetime import datetime

    from app.models.agent_run import AgentRun, AgentStep, AgentRunStatus, AgentStepStatus, AgentStepType
    from app.models.dataset_spec import DatasetSpec
    from app.services.llm_client import get_llm_client
    from app.services.prompts import (
        SYSTEM_ROLE_IMPROVEMENT_ANALYST,
        get_improvement_analysis_prompt,
        get_improvement_plan_prompt,
    )

    logger.info(f"Starting {'enhanced' if use_enhanced_pipeline else 'simple'} auto-improve pipeline for experiment {experiment_id}")

    db = SessionLocal()

    try:
        experiment_uuid = UUID(experiment_id)

        # Load the parent experiment
        parent_experiment = db.query(Experiment).filter(Experiment.id == experiment_uuid).first()
        if not parent_experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if parent_experiment.status != ExperimentStatus.COMPLETED:
            raise ValueError(f"Experiment must be completed to improve. Current status: {parent_experiment.status.value}")

        # Get dataset spec
        dataset_spec = parent_experiment.dataset_spec
        if not dataset_spec:
            raise ValueError("Parent experiment has no dataset spec")

        # ============================================================
        # CHECK FOR OVERFITTING BEFORE PROCEEDING
        # ============================================================
        try:
            from app.services.holdout_validator import get_overfitting_report
            overfitting_report = get_overfitting_report(db, parent_experiment)

            if overfitting_report.get("recommendation") == "stop":
                logger.warning(f"OVERFITTING DETECTED: {overfitting_report.get('message')}")
                return {
                    "status": "stopped",
                    "reason": "overfitting_detected",
                    "message": overfitting_report.get("message"),
                    "best_iteration": overfitting_report.get("best_iteration"),
                    "recommendation": "Revert to iteration " + str(overfitting_report.get("best_iteration")),
                }
            elif overfitting_report.get("recommendation") == "warning":
                logger.warning(f"Overfitting warning: {overfitting_report.get('message')}")
                # Continue but log the warning
        except Exception as e:
            logger.warning(f"Could not check overfitting status: {e}")

        # ============================================================
        # ENHANCED PIPELINE: Use full agent pipeline with iteration context
        # ============================================================
        if use_enhanced_pipeline:
            logger.info("Running ENHANCED improvement pipeline with full agent analysis...")

            from app.services.improvement_pipeline import (
                run_full_improve_pipeline,
                gather_iteration_context,
            )

            # Determine primary metric
            primary_metric = parent_experiment.primary_metric or "roc_auc"

            # Run the enhanced pipeline (this runs the 4 agent steps)
            async def run_enhanced():
                return await run_full_improve_pipeline(
                    db=db,
                    experiment_id=experiment_uuid,
                    primary_metric=primary_metric,
                )

            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                agent_run = loop.run_until_complete(run_enhanced())
            finally:
                loop.close()

            logger.info(f"Enhanced pipeline completed: agent_run {agent_run.id}")

            # Extract results from the agent run
            result = agent_run.result_json or {}
            experiment_design = result.get("experiment_design", {})
            dataset_design = result.get("dataset_design", {})
            iteration_context = result.get("iteration_context", {})

            # Build new dataset spec from the design
            target_column = dataset_spec.target_column
            features_to_keep = dataset_design.get("features_to_keep", dataset_spec.feature_columns or [])
            new_engineered_features = dataset_design.get("new_engineered_features", [])
            features_to_drop = dataset_design.get("features_to_drop", [])

            # Get existing engineered features from parent
            parent_spec_json = dataset_spec.spec_json or {}
            parent_engineered = parent_spec_json.get("engineered_features", [])

            # Merge: parent features + new features
            all_engineered_features = list(parent_engineered) + list(new_engineered_features)

            # Build new feature list (remove dropped, add new engineered)
            new_feature_columns = [f for f in features_to_keep if f not in features_to_drop]
            for eng_feat in new_engineered_features:
                if isinstance(eng_feat, dict) and "output_column" in eng_feat:
                    new_feature_columns.append(eng_feat["output_column"])

            logger.info(f"Enhanced pipeline: {len(parent_engineered)} parent features + {len(new_engineered_features)} new = {len(all_engineered_features)} total")

            # Create new dataset spec
            new_spec_json = dict(parent_spec_json)
            new_spec_json["iteration"] = parent_experiment.iteration_number + 1
            new_spec_json["parent_spec_id"] = str(dataset_spec.id)
            new_spec_json["engineered_features"] = all_engineered_features
            new_spec_json["improvement_rationale"] = dataset_design.get("rationale", "")

            # Build agent experiment design config for this iteration
            # This allows the "Run Experiments" button to work for iteration dataset specs
            iteration_experiment_design_config = {
                "step_id": None,  # No step for iteration-created configs
                "agent_run_id": str(agent_run.id),
                "variants": [{
                    "name": "iteration_default",
                    "description": experiment_design.get("iteration_description", "Auto-improve iteration"),
                    "automl_config": experiment_design.get("automl_config", {"time_limit": 300, "presets": "best_quality"}),
                    "validation_strategy": experiment_design.get("validation_strategy"),
                }],
                "recommended_variant": "iteration_default",
                "primary_metric": parent_experiment.primary_metric,
                "natural_language_summary": experiment_design.get("rationale", ""),
                "stored_at": datetime.utcnow().isoformat(),
                "source_type": "iteration",  # Mark as iteration config
                "parent_experiment_id": str(parent_experiment.id),  # Link to parent
                "iteration_number": parent_experiment.iteration_number + 1,
            }

            new_dataset_spec = DatasetSpec(
                project_id=parent_experiment.project_id,
                data_sources_json=dataset_spec.data_sources_json,
                name=f"{dataset_spec.name} - Iteration {parent_experiment.iteration_number + 1}",
                target_column=target_column,
                feature_columns=new_feature_columns,
                spec_json=new_spec_json,
                filters_json=dataset_spec.filters_json,
                agent_experiment_design_json=iteration_experiment_design_config,
                # Copy time-based metadata from parent
                is_time_based=dataset_spec.is_time_based,
                time_column=dataset_spec.time_column,
                entity_id_column=dataset_spec.entity_id_column,
                prediction_horizon=dataset_spec.prediction_horizon,
                target_positive_class=dataset_spec.target_positive_class,
            )
            db.add(new_dataset_spec)
            db.commit()
            db.refresh(new_dataset_spec)

            logger.info(f"Created new dataset spec {new_dataset_spec.id}")

            # ============================================================
            # INTERNAL FEATURE VALIDATION (when new features are created)
            # ============================================================
            # If new features were engineered, run a quick internal validation
            # to verify they actually help before committing to full training
            feature_validation_result = None
            if new_engineered_features:
                from app.services.internal_feature_validator import (
                    needs_feature_validation,
                    run_quick_validation_training,
                    evaluate_validation_result,
                    create_production_experiment_config,
                    get_experiment_final_score,
                )

                if needs_feature_validation(new_engineered_features, parent_experiment):
                    logger.info(f"Running internal feature validation for {len(new_engineered_features)} new features...")

                    # Get parent score for comparison
                    parent_score = get_experiment_final_score(parent_experiment) or 0
                    metric_direction = parent_experiment.metric_direction or "maximize"

                    # Run quick internal validation training
                    validation_score, validation_error = run_quick_validation_training(
                        db=db,
                        dataset_spec=new_dataset_spec,
                        parent_experiment=parent_experiment,
                        validation_experiment_name=f"Validate {len(new_engineered_features)} features - Iter {parent_experiment.iteration_number + 1}",
                    )

                    # Evaluate the results
                    features_are_good, explanation = evaluate_validation_result(
                        validation_score=validation_score,
                        parent_score=parent_score,
                        metric_direction=metric_direction,
                    )

                    feature_validation_result = {
                        "validated": True,
                        "features_passed": features_are_good,
                        "validation_score": validation_score,
                        "parent_score": parent_score,
                        "explanation": explanation,
                        "validation_error": validation_error,
                    }

                    if features_are_good:
                        logger.info(f"Feature validation PASSED: {explanation}")
                        # Upgrade experiment design to production quality
                        experiment_design = create_production_experiment_config(experiment_design)
                    else:
                        logger.warning(f"Feature validation FAILED: {explanation}")
                        # Still proceed but log the warning
                        # TODO: In future, could loop back to redesign features here

            # Create new experiment with the designed config
            automl_config = experiment_design.get("automl_config", {"time_limit": 300, "presets": "best_quality"})
            new_experiment_plan = dict(parent_experiment.experiment_plan_json or {})
            new_experiment_plan["automl_config"] = automl_config

            # Pass validation_strategy from agent to new experiment
            validation_strategy = experiment_design.get("validation_strategy")
            if validation_strategy:
                new_experiment_plan["validation_strategy"] = validation_strategy
                # Handle both dict and string formats for validation_strategy
                if isinstance(validation_strategy, dict):
                    logger.info(f"Using agent-specified validation strategy: {validation_strategy.get('split_strategy', 'default')}")
                else:
                    logger.info(f"Using agent-specified validation strategy: {validation_strategy}")

            new_experiment = Experiment(
                project_id=parent_experiment.project_id,
                dataset_spec_id=new_dataset_spec.id,
                name=experiment_design.get("iteration_name", f"{parent_experiment.name} - Iteration {parent_experiment.iteration_number + 1}"),
                description=experiment_design.get("iteration_description", f"Enhanced auto-improve from iteration {parent_experiment.iteration_number}"),
                status=ExperimentStatus.PENDING,
                primary_metric=parent_experiment.primary_metric,
                metric_direction=parent_experiment.metric_direction,
                experiment_plan_json=new_experiment_plan,
                iteration_number=parent_experiment.iteration_number + 1,
                parent_experiment_id=parent_experiment.id,
                improvement_context_json={
                    "agent_run_id": str(agent_run.id),
                    "dataset_design": dataset_design,
                    "experiment_design": experiment_design,
                    "summary": experiment_design.get("expected_improvements", ["Enhanced improvement"]),
                    "rationale": dataset_design.get("rationale", ""),
                    "feature_validation": feature_validation_result,
                },
            )
            db.add(new_experiment)
            db.commit()
            db.refresh(new_experiment)

            logger.info(f"Created new experiment {new_experiment.id} (iteration {new_experiment.iteration_number})")

            # Queue for training on Modal
            task = dispatch_task("run_experiment_modal", str(new_experiment.id))

            new_experiment.celery_task_id = task.id
            db.commit()

            logger.info(f"Queued experiment {new_experiment.id} for training on modal")

            return {
                "success": True,
                "pipeline_type": "enhanced",
                "agent_run_id": str(agent_run.id),
                "new_experiment_id": str(new_experiment.id),
                "new_dataset_spec_id": str(new_dataset_spec.id),
                "iteration_number": new_experiment.iteration_number,
                "new_features": len(new_engineered_features),
                "dropped_features": len(features_to_drop),
                "training_task_id": task.id,
                "backend": backend_used,
                "feature_validation": feature_validation_result,
            }

        # ============================================================
        # SIMPLE PIPELINE: Original 2-step improvement (fallback)
        # ============================================================
        logger.info("Running simple 2-step improvement pipeline...")

        # Get the latest trial with critique
        trial = (
            db.query(Trial)
            .filter(Trial.experiment_id == experiment_uuid)
            .order_by(Trial.created_at.desc())
            .first()
        )
        if not trial:
            raise ValueError("No trials found for this experiment")

        # Get model version for feature importances
        model_version = (
            db.query(ModelVersion)
            .filter(ModelVersion.experiment_id == experiment_uuid)
            .order_by(ModelVersion.created_at.desc())
            .first()
        )

        # Create an AgentRun to track the improvement pipeline
        agent_run = AgentRun(
            project_id=parent_experiment.project_id,
            experiment_id=parent_experiment.id,
            name=f"Improve Iteration {parent_experiment.iteration_number}",
            description=f"Auto-improvement pipeline for {parent_experiment.name}",
            status=AgentRunStatus.RUNNING,
            config_json={"parent_experiment_id": str(parent_experiment.id)},
        )
        db.add(agent_run)
        db.commit()
        db.refresh(agent_run)

        logger.info(f"Created agent run {agent_run.id} for improvement pipeline")

        # Step 1: Improvement Analysis
        analysis_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.IMPROVEMENT_ANALYSIS,
            status=AgentStepStatus.RUNNING,
            started_at=datetime.utcnow(),
            input_json={
                "experiment_id": str(parent_experiment.id),
                "experiment_name": parent_experiment.name,
                "iteration_number": parent_experiment.iteration_number,
            },
        )
        db.add(analysis_step)
        db.commit()

        try:
            # Gather all context for analysis
            feature_columns = dataset_spec.feature_columns or []
            target_column = dataset_spec.target_column or ""
            task_type = parent_experiment.experiment_plan_json.get("task_type", "binary") if parent_experiment.experiment_plan_json else "binary"
            primary_metric = parent_experiment.primary_metric or "accuracy"

            # Get metrics
            metrics = trial.metrics_json or {}
            best_score = metrics.get(primary_metric, metrics.get("score_val", 0))
            training_time = metrics.get("training_time_seconds", 0)
            num_models = metrics.get("num_models_trained", 0)

            # Get leaderboard summary
            leaderboard = trial.leaderboard_json or []
            leaderboard_lines = []
            for model in leaderboard[:10]:
                name = model.get("model", "Unknown")
                score = model.get("score_val", 0)
                fit_time = model.get("fit_time", 0)
                leaderboard_lines.append(f"  - {name}: score={score:.4f}, fit_time={fit_time:.1f}s")
            leaderboard_summary = "\n".join(leaderboard_lines) if leaderboard_lines else "(no leaderboard data)"

            # Get feature importances
            feature_importances = {}
            if model_version and model_version.feature_importances_json:
                feature_importances = model_version.feature_importances_json

            # Get training logs
            training_logs = trial.training_logs or "(no logs captured)"

            # Get critique
            critique_json = trial.critique_json

            # Get previous improvement contexts
            previous_improvements = []
            if parent_experiment.parent_experiment_id:
                # Traverse the chain of parent experiments
                current = parent_experiment
                while current.parent_experiment_id:
                    parent = db.query(Experiment).filter(Experiment.id == current.parent_experiment_id).first()
                    if parent and parent.improvement_context_json:
                        previous_improvements.insert(0, parent.improvement_context_json)
                    current = parent
                    if not current:
                        break

            # ========== NEW: Load actual data and gather rich context ==========

            # Load actual dataset to get real column info
            data_statistics = None
            try:
                from app.tasks.automl import load_dataset_from_spec
                logger.info("Loading actual dataset for analysis...")
                df = load_dataset_from_spec(db, dataset_spec)

                # Build data statistics
                data_statistics = {
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "column_stats": {},
                    "sample_values": {},
                }

                # Get column statistics
                for col in df.columns[:30]:  # Limit to 30 columns
                    col_stats = {
                        "dtype": str(df[col].dtype),
                        "null_pct": float(df[col].isna().sum() / len(df) * 100),
                        "unique": int(df[col].nunique()),
                    }
                    data_statistics["column_stats"][col] = col_stats

                # Get sample values from first row
                if len(df) > 0:
                    first_row = df.iloc[0]
                    for col in df.columns[:15]:
                        data_statistics["sample_values"][col] = str(first_row[col])

                logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

            except Exception as e:
                logger.warning(f"Could not load dataset for analysis: {e}")
                data_statistics = None

            # ========== NEW: Gather complete iteration history ==========
            iteration_history = []
            error_history = []

            # Get ALL experiments in this chain (find the root and traverse forward)
            root_experiment = parent_experiment
            while root_experiment.parent_experiment_id:
                parent = db.query(Experiment).filter(Experiment.id == root_experiment.parent_experiment_id).first()
                if parent:
                    root_experiment = parent
                else:
                    break

            # Now traverse from root to current, gathering all history
            experiments_in_chain = [root_experiment]
            current_exp = root_experiment
            while True:
                # Find child experiment
                child = db.query(Experiment).filter(
                    Experiment.parent_experiment_id == current_exp.id
                ).first()
                if child:
                    experiments_in_chain.append(child)
                    current_exp = child
                else:
                    break

            # Build iteration history from all experiments
            for exp in experiments_in_chain:
                exp_trial = db.query(Trial).filter(Trial.experiment_id == exp.id).order_by(Trial.created_at.desc()).first()
                exp_metrics = exp_trial.metrics_json if exp_trial else {}
                exp_score = exp_metrics.get(primary_metric, exp_metrics.get("score_val", 0))

                history_entry = {
                    "iteration": exp.iteration_number,
                    "score": float(exp_score) if exp_score else 0.0,
                    "status": exp.status.value if exp.status else "unknown",
                    "changes_made": exp.improvement_context_json.get("summary", "") if exp.improvement_context_json else "",
                }

                # Capture errors
                if exp.status == ExperimentStatus.FAILED and exp.error_message:
                    history_entry["error"] = exp.error_message
                    error_history.append(f"Iteration {exp.iteration_number}: {exp.error_message}")

                iteration_history.append(history_entry)

            logger.info(f"Gathered history from {len(iteration_history)} iterations, {len(error_history)} errors")

            # Dataset shape (use actual data if available)
            if data_statistics:
                row_count = data_statistics["row_count"]
                col_count = len(data_statistics["columns"])
            else:
                row_count = dataset_spec.spec_json.get("row_count", 0) if dataset_spec.spec_json else 0
                col_count = len(feature_columns)
            dataset_shape = f"{row_count} rows x {col_count} features"

            # Generate improvement analysis prompt with all new context
            analysis_prompt = get_improvement_analysis_prompt(
                experiment_name=parent_experiment.name,
                iteration_number=parent_experiment.iteration_number,
                task_type=task_type,
                target_column=target_column,
                primary_metric=primary_metric,
                best_score=float(best_score) if best_score else 0.0,
                training_time_seconds=float(training_time) if training_time else 0.0,
                num_models_trained=int(num_models) if num_models else 0,
                dataset_shape=dataset_shape,
                feature_columns=feature_columns,
                leaderboard_summary=leaderboard_summary,
                training_logs=training_logs,
                feature_importances=feature_importances,
                critique_json=critique_json,
                previous_improvements=previous_improvements,
                # New rich context
                data_statistics=data_statistics,
                iteration_history=iteration_history if iteration_history else None,
                error_history=error_history if error_history else None,
            )

            # Call LLM for analysis
            async def run_analysis():
                llm_client = get_llm_client()
                messages = [
                    {"role": "system", "content": SYSTEM_ROLE_IMPROVEMENT_ANALYST},
                    {"role": "user", "content": analysis_prompt},
                ]
                return await llm_client.chat_json(messages, None)

            improvement_analysis = asyncio.get_event_loop().run_until_complete(run_analysis())

            analysis_step.status = AgentStepStatus.COMPLETED
            analysis_step.finished_at = datetime.utcnow()
            analysis_step.output_json = improvement_analysis
            db.commit()

            logger.info(f"Improvement analysis complete: {improvement_analysis.get('improvement_summary', '')[:100]}")

        except Exception as e:
            analysis_step.status = AgentStepStatus.FAILED
            analysis_step.error_message = str(e)
            analysis_step.finished_at = datetime.utcnow()
            db.commit()
            raise

        # Step 2: Improvement Plan
        plan_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.IMPROVEMENT_PLAN,
            status=AgentStepStatus.RUNNING,
            started_at=datetime.utcnow(),
            input_json={"improvement_analysis": improvement_analysis},
        )
        db.add(plan_step)
        db.commit()

        try:
            # Get raw columns - prefer actual data columns if we loaded them earlier
            raw_columns = []
            if data_statistics and data_statistics.get("columns"):
                # Use actual columns from loaded data (most accurate)
                raw_columns = data_statistics["columns"]
                logger.info(f"Using {len(raw_columns)} columns from actual loaded data")
            else:
                # Fallback to data source schema
                data_sources_json = dataset_spec.data_sources_json or []
                if data_sources_json:
                    from app.models.data_source import DataSource
                    for ds_info in data_sources_json:
                        # Handle both formats: string ID or dict with data_source_id
                        if isinstance(ds_info, str):
                            ds_id = ds_info
                        elif isinstance(ds_info, dict):
                            ds_id = ds_info.get("data_source_id")
                        else:
                            continue

                        if ds_id:
                            data_source = db.query(DataSource).filter(DataSource.id == ds_id).first()
                            if data_source and data_source.schema_summary:
                                cols = data_source.schema_summary.get("columns", [])
                                if isinstance(cols, list):
                                    raw_columns.extend([c.get("name") if isinstance(c, dict) else str(c) for c in cols])
                                elif isinstance(cols, dict):
                                    raw_columns.extend(cols.keys())
                logger.info(f"Using {len(raw_columns)} columns from data source schema")

            # Get existing engineered features from parent spec
            parent_spec_json = dataset_spec.spec_json or {}
            existing_engineered_features = parent_spec_json.get("engineered_features", [])

            # Generate improvement plan
            plan_prompt = get_improvement_plan_prompt(
                experiment_name=parent_experiment.name,
                iteration_number=parent_experiment.iteration_number,
                task_type=task_type,
                target_column=target_column,
                current_features=feature_columns,
                improvement_analysis=improvement_analysis,
                current_experiment_plan=parent_experiment.experiment_plan_json or {},
                raw_columns=raw_columns if raw_columns else None,
                existing_engineered_features=existing_engineered_features if existing_engineered_features else None,
            )

            async def run_plan():
                llm_client = get_llm_client()
                messages = [
                    {"role": "system", "content": SYSTEM_ROLE_IMPROVEMENT_ANALYST},
                    {"role": "user", "content": plan_prompt},
                ]
                return await llm_client.chat_json(messages, None)

            improvement_plan = asyncio.get_event_loop().run_until_complete(run_plan())

            plan_step.status = AgentStepStatus.COMPLETED
            plan_step.finished_at = datetime.utcnow()
            plan_step.output_json = improvement_plan
            db.commit()

            logger.info(f"Improvement plan created: {improvement_plan.get('plan_summary', '')[:100]}")

        except Exception as e:
            plan_step.status = AgentStepStatus.FAILED
            plan_step.error_message = str(e)
            plan_step.finished_at = datetime.utcnow()
            db.commit()
            raise

        # Step 3: Create new DatasetSpec with improvements
        feature_changes = improvement_plan.get("feature_changes", {})
        features_to_keep = feature_changes.get("features_to_keep", feature_columns)
        new_engineered_features = feature_changes.get("engineered_features", [])

        # Get EXISTING engineered features from parent dataset spec
        # These must be preserved because new features may depend on them
        parent_spec_json = dataset_spec.spec_json or {}
        parent_engineered_features = parent_spec_json.get("engineered_features", [])

        # Merge: parent features first (order matters for dependencies), then new features
        all_engineered_features = list(parent_engineered_features) + list(new_engineered_features)

        logger.info(f"Inheriting {len(parent_engineered_features)} features from parent, adding {len(new_engineered_features)} new features")

        # Build new feature list
        new_feature_columns = list(features_to_keep)
        for eng_feat in new_engineered_features:
            if isinstance(eng_feat, dict) and "output_column" in eng_feat:
                new_feature_columns.append(eng_feat["output_column"])

        # Create new dataset spec based on original
        new_spec_json = dict(dataset_spec.spec_json or {})
        new_spec_json["iteration"] = parent_experiment.iteration_number + 1
        new_spec_json["parent_spec_id"] = str(dataset_spec.id)
        new_spec_json["engineered_features"] = all_engineered_features  # Include ALL features (parent + new)

        # Build agent experiment design config for this iteration (simple pipeline)
        simple_iteration_config = {
            "step_id": None,
            "agent_run_id": None,  # Simple pipeline doesn't have agent run
            "variants": [{
                "name": "iteration_default",
                "description": improvement_plan.get("plan_summary", "Auto-improve iteration"),
                "automl_config": improvement_plan.get("automl_config", {}),
                "validation_strategy": improvement_plan.get("validation_strategy"),
            }],
            "recommended_variant": "iteration_default",
            "primary_metric": parent_experiment.primary_metric,
            "natural_language_summary": improvement_plan.get("plan_summary", ""),
            "stored_at": datetime.utcnow().isoformat(),
            "source_type": "iteration",
            "parent_experiment_id": str(parent_experiment.id),
            "iteration_number": parent_experiment.iteration_number + 1,
        }

        new_dataset_spec = DatasetSpec(
            project_id=parent_experiment.project_id,
            data_sources_json=dataset_spec.data_sources_json,  # Copy data sources
            name=f"{dataset_spec.name} - Iteration {parent_experiment.iteration_number + 1}",
            target_column=target_column,
            feature_columns=new_feature_columns,
            spec_json=new_spec_json,
            filters_json=dataset_spec.filters_json,  # Copy filters too
            agent_experiment_design_json=simple_iteration_config,
            # Copy time-based metadata from parent
            is_time_based=dataset_spec.is_time_based,
            time_column=dataset_spec.time_column,
            entity_id_column=dataset_spec.entity_id_column,
            prediction_horizon=dataset_spec.prediction_horizon,
            target_positive_class=dataset_spec.target_positive_class,
        )
        db.add(new_dataset_spec)
        db.commit()
        db.refresh(new_dataset_spec)

        logger.info(f"Created new dataset spec {new_dataset_spec.id}")

        # Step 4: Create new Experiment
        new_automl_config = improvement_plan.get("automl_config", {})
        new_experiment_plan = dict(parent_experiment.experiment_plan_json or {})
        new_experiment_plan["automl_config"] = new_automl_config

        # Pass validation_strategy from agent to new experiment
        validation_strategy = improvement_plan.get("validation_strategy")
        if validation_strategy:
            new_experiment_plan["validation_strategy"] = validation_strategy
            # Handle both dict and string formats for validation_strategy
            if isinstance(validation_strategy, dict):
                logger.info(f"Using agent-specified validation strategy: {validation_strategy.get('split_strategy', 'default')}")
            else:
                logger.info(f"Using agent-specified validation strategy: {validation_strategy}")

        new_experiment = Experiment(
            project_id=parent_experiment.project_id,
            dataset_spec_id=new_dataset_spec.id,
            name=improvement_plan.get("iteration_name", f"{parent_experiment.name} - Iteration {parent_experiment.iteration_number + 1}"),
            description=improvement_plan.get("iteration_description", f"Auto-improved from iteration {parent_experiment.iteration_number}"),
            status=ExperimentStatus.PENDING,
            primary_metric=parent_experiment.primary_metric,
            metric_direction=parent_experiment.metric_direction,
            experiment_plan_json=new_experiment_plan,
            iteration_number=parent_experiment.iteration_number + 1,
            parent_experiment_id=parent_experiment.id,
            improvement_context_json={
                "improvement_analysis": improvement_analysis,
                "improvement_plan": improvement_plan,
                "summary": improvement_plan.get("plan_summary", ""),
            },
        )
        db.add(new_experiment)
        db.commit()
        db.refresh(new_experiment)

        logger.info(f"Created new experiment {new_experiment.id} (iteration {new_experiment.iteration_number})")

        # Step 5: Queue the new experiment for training on Modal
        task = dispatch_task("run_experiment_modal", str(new_experiment.id))

        new_experiment.celery_task_id = task.id
        db.commit()

        logger.info(f"Queued experiment {new_experiment.id} for training on modal, task {task.id}")

        # Mark agent run as complete
        agent_run.status = AgentRunStatus.COMPLETED
        agent_run.result_json = {
            "new_experiment_id": str(new_experiment.id),
            "new_dataset_spec_id": str(new_dataset_spec.id),
            "iteration_number": new_experiment.iteration_number,
            "improvement_summary": improvement_plan.get("plan_summary", ""),
        }
        db.commit()

        return {
            "success": True,
            "new_experiment_id": str(new_experiment.id),
            "new_dataset_spec_id": str(new_dataset_spec.id),
            "iteration_number": new_experiment.iteration_number,
            "improvement_summary": improvement_plan.get("plan_summary", ""),
            "training_task_id": task.id,
        }

    except (LLMError, LLMTimeoutError) as e:
        logger.error(f"Auto-improve pipeline LLM failure for {experiment_id}: {e}")
        try:
            if 'agent_run' in locals():
                agent_run.status = AgentRunStatus.FAILED
                agent_run.error_message = f"LLM service error: {str(e)}"
                db.commit()
        except Exception as db_err:
            logger.warning(f"Failed to update agent run status: {db_err}")
        raise ImprovementPipelineError(experiment_id, "LLM call", str(e))

    except (DataError, DatasetBuildError) as e:
        logger.error(f"Auto-improve pipeline data error for {experiment_id}: {e}")
        try:
            if 'agent_run' in locals():
                agent_run.status = AgentRunStatus.FAILED
                agent_run.error_message = f"Data error: {str(e)}"
                db.commit()
        except Exception as db_err:
            logger.warning(f"Failed to update agent run status: {db_err}")
        raise ImprovementPipelineError(experiment_id, "data processing", str(e))

    except Exception as e:
        logger.exception(f"Auto-improve pipeline unexpected failure for {experiment_id}: {e}")
        try:
            if 'agent_run' in locals():
                agent_run.status = AgentRunStatus.FAILED
                agent_run.error_message = str(e)
                db.commit()
        except Exception:
            pass

        raise

    finally:
        db.close()


@celery_app.task(name="app.tasks.cancel_experiment")
def cancel_experiment(experiment_id: str) -> dict:
    """Cancel a running experiment.

    Args:
        experiment_id: UUID of the experiment to cancel

    Returns:
        Dictionary with cancellation status
    """
    db = SessionLocal()

    try:
        experiment_uuid = UUID(experiment_id)
        experiment = db.query(Experiment).filter(Experiment.id == experiment_uuid).first()

        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment.status not in [ExperimentStatus.PENDING, ExperimentStatus.RUNNING]:
            return {
                "experiment_id": experiment_id,
                "cancelled": False,
                "reason": f"Experiment is {experiment.status}, cannot cancel",
            }

        experiment.status = ExperimentStatus.CANCELLED
        db.commit()

        return {
            "experiment_id": experiment_id,
            "cancelled": True,
        }

    finally:
        db.close()

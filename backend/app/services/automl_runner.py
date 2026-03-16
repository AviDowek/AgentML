"""AutoML runner service for training ML models using AutoGluon."""
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import re
import pandas as pd

from app.services.smart_metrics import SmartMetricSelector, get_smart_metric_recommendations

logger = logging.getLogger(__name__)


# Time-series indicator patterns in feature names
TIME_SERIES_PATTERNS = [
    r'(?i)lag_?\d*',           # lag, lag_1, lag1, etc.
    r'(?i)_lag\d*$',           # feature_lag1, price_lag
    r'(?i)rolling_',           # rolling_mean, rolling_std
    r'(?i)moving_',            # moving_average
    r'(?i)_ma\d*$',            # feature_ma, price_ma7
    r'(?i)^ma_?\d*',           # ma_7, ma7
    r'(?i)ema_?\d*',           # ema, ema_12
    r'(?i)sma_?\d*',           # sma, sma_20
    r'(?i)volatility',         # volatility, volatility_21d
    r'(?i)return_?s?',         # return, returns, return_1d
    r'(?i)pct_change',         # pct_change
    r'(?i)diff_?\d*',          # diff, diff_1, diff1
    r'(?i)shift_?\d*',         # shift, shift_1
    r'(?i)previous_',          # previous_close, previous_price
    r'(?i)prior_',             # prior_value
    r'(?i)day_of_week',        # day_of_week
    r'(?i)dayofweek',          # dayofweek
    r'(?i)month$',             # month (as standalone or suffix)
    r'(?i)^month_',            # month_sin, month_cos
    r'(?i)quarter$',           # quarter
    r'(?i)year$',              # year
    r'(?i)weekday',            # weekday
    r'(?i)rsi_?\d*',           # rsi, rsi_14
    r'(?i)macd',               # macd, macd_signal
    r'(?i)bollinger',          # bollinger_upper
    r'(?i)momentum',           # momentum, momentum_10
]

TIME_SERIES_COMPILED = [re.compile(p) for p in TIME_SERIES_PATTERNS]


def detect_time_series_features(columns: list[str]) -> tuple[bool, list[str]]:
    """Detect if dataset has time-series features that require temporal validation.

    Args:
        columns: List of column names

    Returns:
        Tuple of (is_time_series, matched_features)
    """
    matched = []
    for col in columns:
        for pattern in TIME_SERIES_COMPILED:
            if pattern.search(col):
                matched.append(col)
                break

    # Consider it time-series if we have 2+ matching features
    is_time_series = len(matched) >= 2

    return is_time_series, matched


# Task type literals for type hints
TabularTaskType = Literal["regression", "binary", "multiclass", "quantile"]
TimeSeriesTaskType = Literal["timeseries_forecast"]
MultiModalTaskType = Literal["multimodal_classification", "multimodal_regression"]
AllTaskTypes = TabularTaskType | TimeSeriesTaskType | MultiModalTaskType


@dataclass
class ValidationPrediction:
    """A single validation prediction with all related data."""
    row_index: int
    features: dict[str, Any]
    target_value: Any
    predicted_value: Any
    error_value: float | None = None
    absolute_error: float | None = None
    prediction_probabilities: dict[str, float] | None = None


@dataclass
class AutoMLResult:
    """Result from an AutoML training run."""

    leaderboard: list[dict[str, Any]]
    best_model_name: str
    artifact_path: str
    feature_importances: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    num_models_trained: int = 0
    task_type: str = ""
    # Additional fields for specific task types
    quantile_levels: list[float] = field(default_factory=list)
    prediction_length: int = 0
    # Validation predictions for error analysis
    validation_predictions: list[ValidationPrediction] = field(default_factory=list)
    # Baseline metrics for sanity checking
    # Contains: majority_class/mean_predictor, simple_logistic/simple_ridge, label_shuffle
    baseline_metrics: dict[str, Any] = field(default_factory=dict)


class BaseRunner(ABC):
    """Base class for all AutoML runners."""

    def __init__(self, artifacts_dir: str = "./artifacts"):
        """Initialize the runner.

        Args:
            artifacts_dir: Base directory for storing model artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run_experiment(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        task_type: str,
        primary_metric: str | None = None,
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> AutoMLResult:
        """Run an AutoML experiment."""
        pass


class TabularRunner(BaseRunner):
    """Runner for tabular ML tasks using AutoGluon TabularPredictor.

    Supports: binary classification, multiclass classification,
    regression, and quantile regression.
    """

    def run_experiment(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        task_type: TabularTaskType,
        primary_metric: str | None = None,
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> AutoMLResult:
        """Run a tabular AutoML experiment.

        Args:
            dataset: Training DataFrame
            target_column: Name of the target column
            task_type: Type of ML task (regression, binary, multiclass, quantile)
            primary_metric: Metric to optimize (auto-detected if None)
            config: AutoML configuration overrides
            experiment_id: Unique ID for this experiment (for artifact path)

        Returns:
            AutoMLResult with leaderboard, best model, metrics, and validation predictions
        """
        from autogluon.tabular import TabularPredictor
        from sklearn.model_selection import train_test_split
        import time
        import numpy as np

        config = config or {}

        # Handle quantile regression special case
        quantile_levels = config.get("quantile_levels", [0.1, 0.5, 0.9])
        is_quantile = task_type == "quantile"
        is_classification = task_type in ("binary", "multiclass")

        # Get sample weights config (set by experiment_tasks based on data analysis)
        use_sample_weights = config.get("use_sample_weights", False)
        data_characteristics = config.get("data_characteristics", {})

        # Determine evaluation metric using smart selection if not provided
        if primary_metric is None:
            if task_type in ("binary", "multiclass"):
                # Use smart metric selection for classification
                try:
                    characteristics = get_smart_metric_recommendations(
                        df=dataset,
                        target_column=target_column,
                        task_type=task_type,
                    )
                    primary_metric = characteristics.recommended_metric
                    use_sample_weights = characteristics.should_use_sample_weights
                    logger.info(f"Smart metric selection: {primary_metric} (imbalance: {characteristics.imbalance_severity.value})")
                except Exception as e:
                    logger.warning(f"Smart metric selection failed: {e}")
                    # Fallback to safe defaults (NOT accuracy for classification)
                    primary_metric = "f1" if task_type == "binary" else "f1_macro"
            else:
                # Default metrics for non-classification tasks
                metric_map = {
                    "regression": "root_mean_squared_error",
                    "quantile": "pinball_loss",
                }
                primary_metric = metric_map.get(task_type, "root_mean_squared_error")

        # Map common metric names to AutoGluon metric names
        metric_mapping = {
            "rmse": "root_mean_squared_error",
            "mse": "mean_squared_error",
            "mae": "mean_absolute_error",
            "r2": "r2",
            "accuracy": "accuracy",
            "auc": "roc_auc",
            "roc_auc": "roc_auc",
            "f1": "f1",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
            "balanced_accuracy": "balanced_accuracy",
            "average_precision": "average_precision",
            "log_loss": "log_loss",
            "pinball": "pinball_loss",
            "pinball_loss": "pinball_loss",
        }
        eval_metric = metric_mapping.get(primary_metric.lower(), primary_metric)

        # Set up artifact path
        artifact_name = experiment_id or f"experiment_{int(time.time())}"
        artifact_path = self.artifacts_dir / artifact_name
        if artifact_path.exists():
            shutil.rmtree(artifact_path)

        # Configure AutoGluon
        time_limit = config.get("time_limit", 300)  # Default 5 minutes
        presets = config.get("presets", "medium_quality")
        num_bag_folds = config.get("num_bag_folds", 5)
        num_stack_levels = config.get("num_stack_levels", 0)
        excluded_model_types = config.get("excluded_model_types", [])

        # Validation split configuration
        validation_split = config.get("validation_split", 0.2)
        random_seed = config.get("random_seed", 42)

        logger.info(
            f"Starting Tabular AutoML experiment: {artifact_name}, "
            f"task={task_type}, metric={eval_metric}, time_limit={time_limit}s"
        )

        start_time = time.time()

        # Split data into train and validation sets
        # The split strategy is determined by the AI agents and passed in config
        # Supported strategies: "temporal", "random", "stratified", "group"
        is_classification = task_type in ("binary", "multiclass")

        # Get validation strategy from config (set by AI agents)
        validation_strategy = config.get("validation_strategy", {})
        split_strategy = validation_strategy.get("split_strategy", None)
        # Support both new and legacy field names
        time_col = validation_strategy.get("time_column")
        entity_id_col = validation_strategy.get("entity_id_column") or validation_strategy.get("group_column")

        # Normalize legacy split type names
        if split_strategy == "temporal":
            split_strategy = "time"
            logger.info("Normalized 'temporal' split type to 'time'")
        elif split_strategy == "group":
            # Legacy "group" becomes "group_random" (random split with groups kept together)
            split_strategy = "group_random"
            logger.info("Normalized 'group' split type to 'group_random'")

        # Auto-detect time-series features to enforce time-based split
        is_time_series, ts_features = detect_time_series_features(list(dataset.columns))
        if is_time_series:
            logger.info(f"Detected time-series features: {ts_features[:5]}{'...' if len(ts_features) > 5 else ''}")
            # Only override if using random/stratified (not if already using time-based)
            if split_strategy in (None, "random", "stratified", "group_random"):
                logger.warning(
                    f"TIME-SERIES DATA DETECTED but split_strategy='{split_strategy}' was specified. "
                    f"OVERRIDING to 'time' to prevent data leakage!"
                )
                split_strategy = "time" if not entity_id_col else "group_time"

        # If no strategy specified, use sensible defaults based on task type
        if not split_strategy:
            if is_classification:
                split_strategy = "stratified"
            else:
                split_strategy = "random"
            logger.info(f"No split_strategy specified, defaulting to '{split_strategy}' for {task_type} task")
        else:
            logger.info(f"Using split_strategy: '{split_strategy}'")
            if time_col:
                logger.info(f"Time column: '{time_col}'")
            if entity_id_col:
                logger.info(f"Entity ID column: '{entity_id_col}'")
            if validation_strategy.get("reasoning"):
                logger.info(f"Split reasoning: {validation_strategy.get('reasoning')}")

        # Apply the selected split strategy
        if split_strategy == "time":
            # Time-based split: sort by time_column, use oldest N% for training, newest N% for validation
            # This prevents data leakage in time-series data
            if time_col and time_col in dataset.columns:
                # Sort by time column to ensure proper temporal ordering
                dataset_sorted = dataset.sort_values(by=time_col, ascending=True).reset_index(drop=True)
                logger.info(f"Sorted dataset by time column '{time_col}' for time-based split")
            else:
                # Assume data is already in chronological order
                dataset_sorted = dataset.reset_index(drop=True)
                logger.warning(
                    f"No time_column specified for 'time' split - assuming data is already in chronological order"
                )

            split_idx = int(len(dataset_sorted) * (1 - validation_split))
            train_data = dataset_sorted.iloc[:split_idx].copy()
            val_data = dataset_sorted.iloc[split_idx:].copy()
            val_indices = val_data.index.tolist()
            logger.info(
                f"Using TIME split: train={len(train_data)} (oldest), validation={len(val_data)} (newest)"
            )

        elif split_strategy == "group_time":
            # Group-time split: sort by time, then split by entity to prevent data leakage
            # Each entity's full time series goes to either train or val
            if not entity_id_col or entity_id_col not in dataset.columns:
                # Fall back to regular time split
                logger.warning(
                    f"Entity column '{entity_id_col}' not found or not specified, falling back to 'time' split"
                )
                if time_col and time_col in dataset.columns:
                    dataset_sorted = dataset.sort_values(by=time_col, ascending=True).reset_index(drop=True)
                    logger.info(f"Sorted dataset by time column '{time_col}' for time-based split")
                else:
                    dataset_sorted = dataset.reset_index(drop=True)
                    logger.warning("No time_column specified - assuming data is already in chronological order")

                split_idx = int(len(dataset_sorted) * (1 - validation_split))
                train_data = dataset_sorted.iloc[:split_idx].copy()
                val_data = dataset_sorted.iloc[split_idx:].copy()
                val_indices = val_data.index.tolist()
                logger.info(
                    f"Using TIME split (fallback): train={len(train_data)} (oldest), validation={len(val_data)} (newest)"
                )
            else:
                # Sort by time column if available
                if time_col and time_col in dataset.columns:
                    dataset_sorted = dataset.sort_values(by=[entity_id_col, time_col], ascending=True)
                else:
                    dataset_sorted = dataset.sort_values(by=entity_id_col)

                # Get unique entities sorted by their last (most recent) observation time
                if time_col and time_col in dataset.columns:
                    entity_last_time = dataset_sorted.groupby(entity_id_col)[time_col].max().sort_values()
                    sorted_entities = entity_last_time.index.tolist()
                else:
                    # Just use entity order
                    sorted_entities = dataset_sorted[entity_id_col].unique().tolist()

                # Split entities: oldest entities go to train, newest to val
                n_entities = len(sorted_entities)
                n_train_entities = int(n_entities * (1 - validation_split))
                train_entities = set(sorted_entities[:n_train_entities])
                val_entities = set(sorted_entities[n_train_entities:])

                train_data = dataset_sorted[dataset_sorted[entity_id_col].isin(train_entities)].copy()
                val_data = dataset_sorted[dataset_sorted[entity_id_col].isin(val_entities)].copy()
                val_indices = val_data.index.tolist()
                logger.info(
                    f"Using GROUP_TIME split by '{entity_id_col}': "
                    f"train={len(train_data)} ({len(train_entities)} entities), "
                    f"validation={len(val_data)} ({len(val_entities)} entities)"
                )

        elif split_strategy == "group_random" and entity_id_col and entity_id_col in dataset.columns:
            # Group-random split: keep entity groups together but randomly assign to train/val
            from sklearn.model_selection import GroupShuffleSplit
            groups = dataset[entity_id_col]
            gss = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=random_seed)
            train_idx, val_idx = next(gss.split(dataset, groups=groups))
            train_data = dataset.iloc[train_idx].copy()
            val_data = dataset.iloc[val_idx].copy()
            val_indices = val_data.index.tolist()
            logger.info(
                f"Using GROUP_RANDOM split by '{entity_id_col}': train={len(train_data)}, validation={len(val_data)}"
            )

        elif split_strategy == "stratified" and is_classification:
            # Stratified split for classification - preserves class proportions
            stratify = dataset[target_column]
            try:
                train_data, val_data = train_test_split(
                    dataset,
                    test_size=validation_split,
                    random_state=random_seed,
                    stratify=stratify,
                )
                logger.info(
                    f"Using STRATIFIED split: train={len(train_data)}, validation={len(val_data)}"
                )
            except ValueError as e:
                # Stratification failed (e.g., too few samples in a class)
                logger.warning(f"Stratified split failed ({e}), falling back to random split")
                train_data, val_data = train_test_split(
                    dataset,
                    test_size=validation_split,
                    random_state=random_seed,
                )
            val_indices = val_data.index.tolist()
        else:
            # Random split (default for regression without temporal data)
            train_data, val_data = train_test_split(
                dataset,
                test_size=validation_split,
                random_state=random_seed,
            )
            val_indices = val_data.index.tolist()
            logger.info(
                f"Using RANDOM split: train={len(train_data)}, validation={len(val_data)}"
            )

        logger.info(
            f"Split data: train={len(train_data)}, validation={len(val_data)}"
        )

        # Determine problem type for AutoGluon
        # Quantile uses "quantile" problem type
        problem_type_map = {
            "regression": "regression",
            "binary": "binary",
            "multiclass": "multiclass",
            "quantile": "quantile",
        }
        problem_type = problem_type_map.get(task_type)

        # Build predictor kwargs
        predictor_kwargs = {
            "label": target_column,
            "path": str(artifact_path),
            "eval_metric": eval_metric,
            "problem_type": problem_type,
        }

        # Add quantile levels for quantile regression
        if is_quantile:
            predictor_kwargs["quantile_levels"] = quantile_levels

        # Train with AutoGluon on training data only
        predictor = TabularPredictor(**predictor_kwargs)

        # Determine if we're using bagged mode (which happens with good_quality or better presets)
        # In bagged mode, we need use_bag_holdout=True to use tuning_data
        use_bagging = presets in ["good_quality", "high_quality", "best_quality"] or num_bag_folds > 0

        # Data size warnings and auto-adjustments to prevent overfitting
        n_samples = len(train_data)
        n_features = len(train_data.columns) - 1  # exclude target

        # Small dataset thresholds
        SMALL_DATASET_THRESHOLD = 1000
        MEDIUM_DATASET_THRESHOLD = 5000

        if n_samples < SMALL_DATASET_THRESHOLD:
            logger.warning(
                f"SMALL DATASET DETECTED ({n_samples} samples, {n_features} features). "
                f"High risk of overfitting with complex ensembles."
            )
            # Reduce stacking for small datasets
            if num_stack_levels > 1:
                original_stack = num_stack_levels
                num_stack_levels = 1
                logger.warning(
                    f"Reducing num_stack_levels from {original_stack} to {num_stack_levels} "
                    f"to prevent overfitting on small dataset"
                )
            # Also reduce bag folds for very small datasets
            if num_bag_folds > 5 and n_samples < 500:
                original_folds = num_bag_folds
                num_bag_folds = 3
                logger.warning(
                    f"Reducing num_bag_folds from {original_folds} to {num_bag_folds} "
                    f"for very small dataset ({n_samples} samples)"
                )

        if n_samples < MEDIUM_DATASET_THRESHOLD and is_time_series:
            logger.warning(
                f"TIME-SERIES DATA with limited samples ({n_samples}). "
                f"Complex models may not generalize to future periods."
            )

        fit_kwargs = {
            "train_data": train_data,
            "time_limit": time_limit,
            "presets": presets,
            "excluded_model_types": excluded_model_types,
            "dynamic_stacking": False,  # Disable to avoid "Learner is already fit" edge case
            "verbosity": 2,
        }

        if use_bagging:
            # In bagged mode, use tuning_data as holdout validation
            fit_kwargs["tuning_data"] = val_data
            fit_kwargs["use_bag_holdout"] = True
            # Don't manually set bag folds - let preset handle it
        else:
            # In non-bagged mode, can use tuning_data directly
            fit_kwargs["tuning_data"] = val_data
            fit_kwargs["num_bag_folds"] = num_bag_folds
            fit_kwargs["num_stack_levels"] = num_stack_levels

        # Add sample weights for imbalanced classification
        if use_sample_weights and is_classification:
            try:
                selector = SmartMetricSelector()
                sample_weights = selector.get_sample_weights(train_data[target_column], method="balanced")
                if sample_weights is not None:
                    # Add sample weights as a column
                    weight_column = "__sample_weight__"
                    train_data = train_data.copy()
                    train_data[weight_column] = sample_weights.values
                    fit_kwargs["train_data"] = train_data
                    fit_kwargs["sample_weight"] = weight_column
                    logger.info(f"Applied sample weights for class imbalance (method: balanced)")
            except Exception as weight_err:
                logger.warning(f"Could not apply sample weights: {weight_err}")

        predictor.fit(**fit_kwargs)

        training_time = time.time() - start_time

        # Get leaderboard
        leaderboard_df = predictor.leaderboard(silent=True)
        leaderboard = leaderboard_df.to_dict(orient="records")

        # Get best model name
        best_model_name = predictor.model_best

        # Get feature importances (if available)
        feature_importances = {}
        try:
            importance_df = predictor.feature_importance(val_data, silent=True)
            if importance_df is not None and not importance_df.empty:
                feature_importances = importance_df["importance"].to_dict()
        except Exception as e:
            logger.warning(f"Could not compute feature importances: {e}")

        # Get metrics for best model
        metrics = {}
        try:
            import pandas as pd

            # Log leaderboard columns for debugging
            logger.info(f"Leaderboard columns: {list(leaderboard_df.columns)}")

            # Get predictor info for scores
            predictor_info = predictor.info()

            # Try to get best model score from predictor info
            best_score = predictor_info.get("best_model_score_val")
            if best_score is not None and not pd.isna(best_score):
                metrics[primary_metric] = float(best_score)
                metrics["score_val"] = float(best_score)

            # Get additional metrics from leaderboard
            best_model_row = leaderboard_df[
                leaderboard_df["model"] == best_model_name
            ].iloc[0]

            logger.info(f"Best model row: {best_model_row.to_dict()}")

            # Try different column names for the score (varies by AutoGluon version/mode)
            for score_col in ["score_val", "score_test", "score_holdout"]:
                if score_col in best_model_row:
                    val = best_model_row[score_col]
                    if val is not None and not pd.isna(val):
                        metrics["score_val"] = float(val)
                        if primary_metric not in metrics or pd.isna(metrics.get(primary_metric)):
                            metrics[primary_metric] = float(val)
                        break

            if "pred_time_val" in best_model_row and not pd.isna(best_model_row["pred_time_val"]):
                metrics["pred_time_val"] = float(best_model_row["pred_time_val"])
            if "fit_time" in best_model_row and not pd.isna(best_model_row["fit_time"]):
                metrics["fit_time"] = float(best_model_row["fit_time"])

            # Capture training score for overfitting detection
            # AutoGluon stores this as "score_train" in the leaderboard when available
            for train_col in ["score_train", "score_training"]:
                if train_col in best_model_row and not pd.isna(best_model_row[train_col]):
                    metrics["score_train"] = float(best_model_row[train_col])
                    break

            # Try to get training score from predictor info as fallback
            if "score_train" not in metrics:
                train_info_score = predictor_info.get("best_model_score_train")
                if train_info_score is not None and not pd.isna(train_info_score):
                    metrics["score_train"] = float(train_info_score)

            # If we still don't have metrics, evaluate on validation data directly
            if not metrics.get(primary_metric) or pd.isna(metrics.get(primary_metric)):
                logger.info("No metrics from leaderboard, evaluating on validation data...")
                eval_score = predictor.evaluate(val_data, silent=True)
                logger.info(f"Evaluation result: {eval_score}")
                if isinstance(eval_score, dict):
                    for k, v in eval_score.items():
                        if v is not None and not pd.isna(v):
                            metrics[k] = float(v)
                    if primary_metric not in metrics and eval_score:
                        first_val = list(eval_score.values())[0]
                        if first_val is not None and not pd.isna(first_val):
                            metrics[primary_metric] = float(first_val)
                else:
                    if eval_score is not None and not pd.isna(eval_score):
                        metrics[primary_metric] = float(eval_score)
                        metrics["score_val"] = float(eval_score)

            logger.info(f"Final extracted metrics: {metrics}")

            # Normalize metrics: AutoGluon internally uses negative error metrics for optimization
            # (since it always maximizes). Convert to positive for display.
            ERROR_METRICS = {
                "root_mean_squared_error", "rmse", "mean_squared_error", "mse",
                "mean_absolute_error", "mae", "median_absolute_error",
                "mean_squared_log_error", "log_loss", "pinball_loss"
            }
            for key in list(metrics.keys()):
                if any(err_metric in key.lower() for err_metric in ERROR_METRICS):
                    val = metrics[key]
                    if val is not None and val < 0:
                        metrics[key] = abs(val)
                        logger.info(f"Normalized {key}: {val} -> {metrics[key]} (converted to positive)")

        except Exception as e:
            logger.warning(f"Could not extract detailed metrics: {e}", exc_info=True)

        # Capture validation predictions
        validation_predictions = []
        try:
            validation_predictions = self._capture_validation_predictions(
                predictor=predictor,
                val_data=val_data,
                val_indices=val_indices,
                target_column=target_column,
                task_type=task_type,
                is_quantile=is_quantile,
            )
            logger.info(f"Captured {len(validation_predictions)} validation predictions")
        except Exception as e:
            logger.warning(f"Could not capture validation predictions: {e}")

        num_models = len(leaderboard)

        logger.info(
            f"Tabular AutoML completed: {num_models} models trained, "
            f"best={best_model_name}, time={training_time:.1f}s"
        )

        # Compute baseline metrics for sanity checking
        baseline_metrics = {}
        if not is_quantile:  # Baselines not implemented for quantile regression
            try:
                from app.services.baseline_models import compute_all_baselines
                baseline_metrics = compute_all_baselines(
                    train_data=train_data,
                    val_data=val_data,
                    target_column=target_column,
                    task_type=task_type,
                    primary_metric=primary_metric,
                    run_shuffle_test=True,
                )
                logger.info(f"Baseline metrics computed: {list(baseline_metrics.keys())}")
            except Exception as e:
                logger.warning(f"Failed to compute baseline metrics: {e}")

        return AutoMLResult(
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            artifact_path=str(artifact_path),
            feature_importances=feature_importances,
            metrics=metrics,
            training_time_seconds=training_time,
            num_models_trained=num_models,
            task_type=task_type,
            quantile_levels=quantile_levels if is_quantile else [],
            validation_predictions=validation_predictions,
            baseline_metrics=baseline_metrics,
        )

    def _capture_validation_predictions(
        self,
        predictor,
        val_data: pd.DataFrame,
        val_indices: list[int],
        target_column: str,
        task_type: str,
        is_quantile: bool,
    ) -> list[ValidationPrediction]:
        """Capture predictions on validation data for error analysis.

        Args:
            predictor: Trained AutoGluon predictor
            val_data: Validation DataFrame
            val_indices: Original indices of validation rows
            target_column: Name of target column
            task_type: ML task type
            is_quantile: Whether this is quantile regression

        Returns:
            List of ValidationPrediction objects
        """
        import numpy as np

        # Get feature columns (all columns except target)
        feature_columns = [c for c in val_data.columns if c != target_column]

        # Get predictions
        predictions = predictor.predict(val_data)

        # Get prediction probabilities for classification
        pred_probas = None
        is_classification = task_type in ("binary", "multiclass")
        if is_classification:
            try:
                pred_probas = predictor.predict_proba(val_data)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")

        # Build validation predictions list
        validation_predictions = []

        for i, (idx, row) in enumerate(val_data.iterrows()):
            # Get feature values as dict
            features = {col: self._serialize_value(row[col]) for col in feature_columns}

            # Get target and predicted values
            target_value = row[target_column]
            predicted_value = predictions.iloc[i] if hasattr(predictions, 'iloc') else predictions[i]

            # Calculate error metrics
            error_value = None
            absolute_error = None

            if is_quantile:
                # For quantile, use median prediction if available
                if hasattr(predicted_value, '__iter__'):
                    predicted_value = predicted_value[len(predicted_value) // 2]
                try:
                    error_value = float(predicted_value) - float(target_value)
                    absolute_error = abs(error_value)
                except (ValueError, TypeError):
                    pass
            elif task_type == "regression":
                try:
                    error_value = float(predicted_value) - float(target_value)
                    absolute_error = abs(error_value)
                except (ValueError, TypeError):
                    pass
            elif is_classification:
                # For classification, error is 0 if correct, 1 if incorrect
                is_correct = str(predicted_value) == str(target_value)
                error_value = 0.0 if is_correct else 1.0
                absolute_error = error_value

            # Get prediction probabilities for this row
            prediction_probabilities = None
            if pred_probas is not None:
                try:
                    row_probas = pred_probas.iloc[i]
                    if isinstance(row_probas, pd.Series):
                        prediction_probabilities = {
                            str(k): float(v) for k, v in row_probas.items()
                        }
                    else:
                        # For binary classification with single probability
                        prediction_probabilities = {"probability": float(row_probas)}
                except Exception:
                    pass

            validation_predictions.append(ValidationPrediction(
                row_index=i,  # Index within validation set
                features=features,
                target_value=self._serialize_value(target_value),
                predicted_value=self._serialize_value(predicted_value),
                error_value=error_value,
                absolute_error=absolute_error,
                prediction_probabilities=prediction_probabilities,
            ))

        return validation_predictions

    def _serialize_value(self, value: Any) -> Any:
        """Convert a value to JSON-serializable format."""
        import numpy as np

        # Handle numpy arrays first (before pd.isna which doesn't work on arrays)
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, pd.Timestamp):
            return value.isoformat()
        else:
            return value

    def load_predictor(self, artifact_path: str):
        """Load a trained predictor from disk.

        Handles cross-platform loading (models trained on Linux, loaded on Windows).
        """
        import platform
        import pathlib
        from autogluon.tabular import TabularPredictor

        # Fix for loading models trained on Linux on Windows
        # PosixPath objects can't be instantiated on Windows, so we patch it
        if platform.system() == "Windows":
            pathlib.PosixPath = pathlib.WindowsPath

        return TabularPredictor.load(artifact_path)

    def predict(self, artifact_path: str, data: pd.DataFrame) -> pd.Series:
        """Make predictions using a trained model."""
        predictor = self.load_predictor(artifact_path)
        return predictor.predict(data)

    def predict_proba(self, artifact_path: str, data: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities for classification models."""
        predictor = self.load_predictor(artifact_path)
        return predictor.predict_proba(data)


class TimeSeriesRunner(BaseRunner):
    """Runner for time series forecasting using AutoGluon TimeSeriesPredictor."""

    def run_experiment(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        task_type: TimeSeriesTaskType = "timeseries_forecast",
        primary_metric: str | None = None,
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> AutoMLResult:
        """Run a time series forecasting experiment.

        Args:
            dataset: Training DataFrame with time series data.
                     Must have columns: item_id (optional), timestamp, target
            target_column: Name of the target column to forecast
            task_type: Always "timeseries_forecast"
            primary_metric: Metric to optimize (default: MASE)
            config: Configuration including:
                - prediction_length: Number of steps to forecast (required)
                - time_column: Name of timestamp column (default: "timestamp")
                - id_column: Name of item/series ID column (default: None for single series)
                - freq: Time series frequency (default: auto-detected)
                - time_limit: Training time limit in seconds
                - presets: AutoGluon preset quality
            experiment_id: Unique ID for this experiment

        Returns:
            AutoMLResult with leaderboard, best model, and metrics
        """
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        import time

        config = config or {}

        # Required configuration
        prediction_length = config.get("prediction_length", 10)
        time_column = config.get("time_column", "timestamp")
        id_column = config.get("id_column", None)
        freq = config.get("freq", None)  # Auto-detect if None

        # Metric selection
        if primary_metric is None:
            primary_metric = "MASE"  # Mean Absolute Scaled Error

        metric_mapping = {
            "mase": "MASE",
            "mape": "MAPE",
            "smape": "sMAPE",
            "rmse": "RMSE",
            "mae": "MAE",
            "wql": "WQL",  # Weighted Quantile Loss
        }
        eval_metric = metric_mapping.get(primary_metric.lower(), primary_metric)

        # Set up artifact path
        artifact_name = experiment_id or f"ts_experiment_{int(time.time())}"
        artifact_path = self.artifacts_dir / artifact_name
        if artifact_path.exists():
            shutil.rmtree(artifact_path)

        # Configure training
        time_limit = config.get("time_limit", 300)  # Default 5 minutes
        presets = config.get("presets", "medium_quality")

        logger.info(
            f"Starting Time Series experiment: {artifact_name}, "
            f"prediction_length={prediction_length}, metric={eval_metric}, "
            f"time_limit={time_limit}s"
        )

        start_time = time.time()

        # Prepare data for AutoGluon TimeSeriesDataFrame
        # If no id_column, treat as single time series
        if id_column is None:
            dataset = dataset.copy()
            dataset["item_id"] = "series_0"
            id_column = "item_id"

        # Convert to TimeSeriesDataFrame
        ts_data = TimeSeriesDataFrame.from_data_frame(
            dataset,
            id_column=id_column,
            timestamp_column=time_column,
        )

        # Train predictor
        predictor = TimeSeriesPredictor(
            path=str(artifact_path),
            target=target_column,
            prediction_length=prediction_length,
            eval_metric=eval_metric,
            freq=freq,
        )

        predictor.fit(
            train_data=ts_data,
            time_limit=time_limit,
            presets=presets,
            verbosity=2,
        )

        training_time = time.time() - start_time

        # Get leaderboard
        leaderboard_df = predictor.leaderboard(silent=True)
        leaderboard = leaderboard_df.to_dict(orient="records")

        # Get best model
        best_model_name = predictor.model_best

        # Get metrics
        metrics = {}
        try:
            best_model_row = leaderboard_df[
                leaderboard_df["model"] == best_model_name
            ].iloc[0]
            metrics["score_val"] = best_model_row.get("score_val", 0)
            metrics[eval_metric] = best_model_row.get("score_val", 0)
            if "pred_time_val" in best_model_row:
                metrics["pred_time_val"] = best_model_row["pred_time_val"]
            if "fit_time" in best_model_row:
                metrics["fit_time"] = best_model_row["fit_time"]
        except Exception as e:
            logger.warning(f"Could not extract detailed metrics: {e}")

        num_models = len(leaderboard)

        logger.info(
            f"Time Series experiment completed: {num_models} models trained, "
            f"best={best_model_name}, time={training_time:.1f}s"
        )

        return AutoMLResult(
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            artifact_path=str(artifact_path),
            feature_importances={},  # Time series doesn't have feature importances
            metrics=metrics,
            training_time_seconds=training_time,
            num_models_trained=num_models,
            task_type=task_type,
            prediction_length=prediction_length,
        )

    def load_predictor(self, artifact_path: str):
        """Load a trained predictor from disk.

        Handles cross-platform loading (models trained on Linux, loaded on Windows).
        """
        import platform
        import pathlib
        from autogluon.timeseries import TimeSeriesPredictor

        if platform.system() == "Windows":
            pathlib.PosixPath = pathlib.WindowsPath

        return TimeSeriesPredictor.load(artifact_path)

    def predict(self, artifact_path: str, data: pd.DataFrame) -> pd.DataFrame:
        """Make forecasts using a trained model.

        Returns DataFrame with forecasts for each time series.
        """
        from autogluon.timeseries import TimeSeriesDataFrame
        predictor = self.load_predictor(artifact_path)

        # Convert to TimeSeriesDataFrame if needed
        if not isinstance(data, TimeSeriesDataFrame):
            data = TimeSeriesDataFrame(data)

        return predictor.predict(data)


class MultiModalRunner(BaseRunner):
    """Runner for multimodal ML tasks using AutoGluon MultiModalPredictor.

    Supports combined text, tabular, and image data for classification or regression.
    """

    def run_experiment(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        task_type: MultiModalTaskType,
        primary_metric: str | None = None,
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> AutoMLResult:
        """Run a multimodal AutoML experiment.

        Args:
            dataset: Training DataFrame with mixed data types.
                     Text columns: regular string columns
                     Image columns: paths to image files
                     Tabular columns: numeric/categorical features
            target_column: Name of the target column
            task_type: "multimodal_classification" or "multimodal_regression"
            primary_metric: Metric to optimize (auto-detected if None)
            config: Configuration including:
                - time_limit: Training time limit in seconds
                - presets: AutoGluon preset quality
                - image_columns: List of columns containing image paths
                - text_columns: List of columns containing text
            experiment_id: Unique ID for this experiment

        Returns:
            AutoMLResult with leaderboard, best model, and metrics
        """
        from autogluon.multimodal import MultiModalPredictor
        import time

        config = config or {}

        # Determine if classification or regression
        is_classification = task_type == "multimodal_classification"

        # Metric selection
        if primary_metric is None:
            if is_classification:
                primary_metric = "accuracy"
            else:
                primary_metric = "rmse"

        metric_mapping = {
            "accuracy": "accuracy",
            "auc": "roc_auc",
            "roc_auc": "roc_auc",
            "f1": "f1",
            "log_loss": "log_loss",
            "rmse": "rmse",
            "mse": "mse",
            "mae": "mae",
            "r2": "r2",
        }
        eval_metric = metric_mapping.get(primary_metric.lower(), primary_metric)

        # Set up artifact path
        artifact_name = experiment_id or f"mm_experiment_{int(time.time())}"
        artifact_path = self.artifacts_dir / artifact_name
        if artifact_path.exists():
            shutil.rmtree(artifact_path)

        # Configure training
        time_limit = config.get("time_limit", 300)  # Default 5 minutes
        presets = config.get("presets", "medium_quality")

        # Determine problem type
        problem_type = "multiclass" if is_classification else "regression"
        # Check if binary classification
        if is_classification and dataset[target_column].nunique() == 2:
            problem_type = "binary"

        logger.info(
            f"Starting Multimodal experiment: {artifact_name}, "
            f"task={task_type}, problem_type={problem_type}, "
            f"metric={eval_metric}, time_limit={time_limit}s"
        )

        start_time = time.time()

        # Train predictor
        predictor = MultiModalPredictor(
            label=target_column,
            path=str(artifact_path),
            eval_metric=eval_metric,
            problem_type=problem_type,
        )

        predictor.fit(
            train_data=dataset,
            time_limit=time_limit,
            presets=presets,
            verbosity=2,
        )

        training_time = time.time() - start_time

        # Get model info
        # MultiModalPredictor doesn't have a leaderboard like Tabular
        # but we can get some info
        best_model_name = "AutoMM"  # MultiModal uses a single ensemble model

        # Create a simple leaderboard entry
        leaderboard = [{
            "model": best_model_name,
            "fit_time": training_time,
        }]

        # Get metrics
        metrics = {}
        try:
            # Evaluate on training data (or validation if available)
            eval_results = predictor.evaluate(dataset)
            if isinstance(eval_results, dict):
                metrics = eval_results
            else:
                metrics[eval_metric] = eval_results
        except Exception as e:
            logger.warning(f"Could not evaluate model: {e}")

        logger.info(
            f"Multimodal experiment completed: best={best_model_name}, "
            f"time={training_time:.1f}s"
        )

        return AutoMLResult(
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            artifact_path=str(artifact_path),
            feature_importances={},  # MultiModal doesn't expose feature importances easily
            metrics=metrics,
            training_time_seconds=training_time,
            num_models_trained=1,  # AutoMM trains a single multi-modal ensemble
            task_type=task_type,
        )

    def load_predictor(self, artifact_path: str):
        """Load a trained predictor from disk.

        Handles cross-platform loading (models trained on Linux, loaded on Windows).
        """
        import platform
        import pathlib
        from autogluon.multimodal import MultiModalPredictor

        # Fix for loading models trained on Linux on Windows
        # PosixPath objects can't be instantiated on Windows, so we patch it
        if platform.system() == "Windows":
            pathlib.PosixPath = pathlib.WindowsPath

        return MultiModalPredictor.load(artifact_path)

    def predict(self, artifact_path: str, data: pd.DataFrame) -> pd.Series:
        """Make predictions using a trained model."""
        predictor = self.load_predictor(artifact_path)
        return predictor.predict(data)

    def predict_proba(self, artifact_path: str, data: pd.DataFrame) -> pd.DataFrame:
        """Get prediction probabilities for classification models."""
        predictor = self.load_predictor(artifact_path)
        return predictor.predict_proba(data)


# Legacy alias for backward compatibility
AutoMLRunner = TabularRunner


def get_runner_for_task(
    task_type: str,
    artifacts_dir: str = "./artifacts"
) -> BaseRunner:
    """Factory function to get the appropriate runner for a task type.

    Args:
        task_type: The ML task type
        artifacts_dir: Directory for model artifacts

    Returns:
        Appropriate runner instance for the task type
    """
    tabular_tasks = {"regression", "binary", "multiclass", "quantile", "classification"}
    timeseries_tasks = {"timeseries_forecast"}
    multimodal_tasks = {"multimodal_classification", "multimodal_regression"}

    if task_type in tabular_tasks:
        return TabularRunner(artifacts_dir)
    elif task_type in timeseries_tasks:
        return TimeSeriesRunner(artifacts_dir)
    elif task_type in multimodal_tasks:
        return MultiModalRunner(artifacts_dir)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

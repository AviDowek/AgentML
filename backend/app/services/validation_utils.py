"""Validation utilities for ML experiments."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Valid metrics by task type for AutoGluon
VALID_METRICS_BY_TASK = {
    "binary": {
        "roc_auc", "accuracy", "f1", "f1_macro", "f1_micro", "f1_weighted",
        "balanced_accuracy", "precision", "recall", "log_loss", "average_precision",
    },
    "multiclass": {
        "accuracy", "balanced_accuracy", "f1_macro", "f1_micro", "f1_weighted",
        "log_loss", "roc_auc_ovo_macro", "roc_auc_ovr_macro",
    },
    "regression": {
        "root_mean_squared_error", "mean_squared_error", "mean_absolute_error",
        "r2", "median_absolute_error", "rmse", "mse", "mae",
    },
    "quantile": {
        "pinball_loss",
    },
}

# Metric aliases (user-friendly names -> AutoGluon names)
METRIC_ALIASES = {
    "rmse": "root_mean_squared_error",
    "mse": "mean_squared_error",
    "mae": "mean_absolute_error",
    "auc": "roc_auc",
}

# Default metrics by task type
DEFAULT_METRICS = {
    "binary": "roc_auc",
    "multiclass": "accuracy",
    "regression": "root_mean_squared_error",
    "quantile": "pinball_loss",
}


def validate_metric_for_task(metric: str, task_type: str) -> tuple[bool, Optional[str]]:
    """Validate that a metric is appropriate for a task type.

    Args:
        metric: The metric name to validate
        task_type: The ML task type (binary, multiclass, regression, quantile)

    Returns:
        Tuple of (is_valid, normalized_metric_name or None)
    """
    if not metric:
        return True, DEFAULT_METRICS.get(task_type)

    # Normalize metric name
    metric_lower = metric.lower()

    # Check aliases
    normalized = METRIC_ALIASES.get(metric_lower, metric_lower)

    # Get valid metrics for task
    valid_metrics = VALID_METRICS_BY_TASK.get(task_type, set())

    if normalized in valid_metrics or metric_lower in valid_metrics:
        return True, normalized

    # Metric not valid for this task
    return False, None


def get_default_metric(task_type: str) -> str:
    """Get the default metric for a task type.

    Args:
        task_type: The ML task type

    Returns:
        Default metric name
    """
    return DEFAULT_METRICS.get(task_type, "accuracy")


def normalize_metric(metric: str, task_type: str) -> str:
    """Normalize a metric name, falling back to default if invalid.

    Args:
        metric: The metric name to normalize
        task_type: The ML task type

    Returns:
        Normalized metric name
    """
    is_valid, normalized = validate_metric_for_task(metric, task_type)

    if is_valid and normalized:
        return normalized

    # Log warning and use default
    default = get_default_metric(task_type)
    logger.warning(
        f"Invalid metric '{metric}' for task type '{task_type}'. "
        f"Valid options: {VALID_METRICS_BY_TASK.get(task_type, set())}. "
        f"Using default: {default}"
    )
    return default

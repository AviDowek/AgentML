"""Smart metric selection based on data characteristics.

This module provides intelligent metric selection that considers:
- Task type (binary, multiclass, regression)
- Class imbalance severity
- Data characteristics
- Business context

The goal is to select metrics that will produce actually usable models,
not just models that look good on misleading metrics like accuracy.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ImbalanceSeverity(str, Enum):
    """Class imbalance severity levels."""
    NONE = "none"           # Ratio < 1.5:1
    MILD = "mild"           # Ratio 1.5:1 - 3:1
    MODERATE = "moderate"   # Ratio 3:1 - 10:1
    SEVERE = "severe"       # Ratio > 10:1
    EXTREME = "extreme"     # Ratio > 50:1


@dataclass
class DataCharacteristics:
    """Analyzed data characteristics for metric selection."""
    task_type: str
    num_classes: int = 2
    class_distribution: Optional[dict[str, float]] = None
    imbalance_ratio: float = 1.0
    imbalance_severity: ImbalanceSeverity = ImbalanceSeverity.NONE
    minority_class_count: int = 0
    total_samples: int = 0
    has_temporal_component: bool = False
    is_ranking_task: bool = False

    # Recommended settings based on analysis
    recommended_metric: str = "accuracy"
    recommended_secondary_metrics: list[str] = None
    should_use_sample_weights: bool = False
    recommended_presets: str = "medium_quality"

    def __post_init__(self):
        if self.recommended_secondary_metrics is None:
            self.recommended_secondary_metrics = []


class SmartMetricSelector:
    """Intelligently selects metrics based on data characteristics."""

    # Metric mappings for different scenarios
    BINARY_METRICS = {
        ImbalanceSeverity.NONE: "accuracy",
        ImbalanceSeverity.MILD: "roc_auc",
        ImbalanceSeverity.MODERATE: "f1",
        ImbalanceSeverity.SEVERE: "f1",
        ImbalanceSeverity.EXTREME: "average_precision",  # PR-AUC, better for extreme imbalance
    }

    MULTICLASS_METRICS = {
        ImbalanceSeverity.NONE: "accuracy",
        ImbalanceSeverity.MILD: "balanced_accuracy",
        ImbalanceSeverity.MODERATE: "f1_macro",
        ImbalanceSeverity.SEVERE: "f1_macro",
        ImbalanceSeverity.EXTREME: "f1_weighted",
    }

    REGRESSION_METRICS = {
        "default": "root_mean_squared_error",
        "log_scale": "mean_absolute_percentage_error",
        "robust": "median_absolute_error",
    }

    # AutoGluon-compatible metric names
    AUTOGLUON_METRIC_MAP = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "balanced_accuracy": "balanced_accuracy",
        "average_precision": "average_precision",
        "log_loss": "log_loss",
        "root_mean_squared_error": "root_mean_squared_error",
        "mean_absolute_error": "mean_absolute_error",
        "mean_absolute_percentage_error": "mean_absolute_percentage_error",
        "median_absolute_error": "median_absolute_error",
        "r2": "r2",
        "pinball_loss": "pinball_loss",
    }

    def __init__(self):
        pass

    def analyze_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Optional[str] = None,
    ) -> DataCharacteristics:
        """Analyze data to determine optimal metric and training settings.

        Args:
            df: The training DataFrame
            target_column: Name of the target column
            task_type: Optional override for task type (binary, multiclass, regression)

        Returns:
            DataCharacteristics with recommended settings
        """
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not in DataFrame")
            return DataCharacteristics(
                task_type=task_type or "binary",
                recommended_metric="accuracy",
            )

        target = df[target_column]
        total_samples = len(target)

        # Determine task type if not provided
        if task_type is None:
            task_type = self._infer_task_type(target)

        logger.info(f"Analyzing data for task type: {task_type}")

        if task_type == "regression":
            return self._analyze_regression(target, total_samples)
        else:
            return self._analyze_classification(target, task_type, total_samples)

    def _infer_task_type(self, target: pd.Series) -> str:
        """Infer task type from target column."""
        # Check if numeric with many unique values → regression
        if pd.api.types.is_numeric_dtype(target):
            unique_ratio = target.nunique() / len(target)
            if unique_ratio > 0.1 and target.nunique() > 20:
                return "regression"

        # Check number of classes
        num_classes = target.nunique()
        if num_classes == 2:
            return "binary"
        elif num_classes > 2:
            return "multiclass"
        else:
            return "binary"  # Default

    def _analyze_classification(
        self,
        target: pd.Series,
        task_type: str,
        total_samples: int,
    ) -> DataCharacteristics:
        """Analyze classification task data."""
        # Get class distribution
        value_counts = target.value_counts()
        num_classes = len(value_counts)

        class_distribution = {
            str(cls): count / total_samples
            for cls, count in value_counts.items()
        }

        # Calculate imbalance
        majority_count = value_counts.iloc[0]
        minority_count = value_counts.iloc[-1]
        imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')

        # Determine severity
        imbalance_severity = self._get_imbalance_severity(imbalance_ratio)

        logger.info(f"Class imbalance analysis: ratio={imbalance_ratio:.2f}:1, severity={imbalance_severity.value}")
        logger.info(f"Class distribution: {class_distribution}")

        # Select metric based on task type and imbalance
        if task_type == "binary":
            recommended_metric = self.BINARY_METRICS.get(
                imbalance_severity, "roc_auc"
            )
            secondary_metrics = self._get_binary_secondary_metrics(imbalance_severity)
        else:
            recommended_metric = self.MULTICLASS_METRICS.get(
                imbalance_severity, "f1_macro"
            )
            secondary_metrics = self._get_multiclass_secondary_metrics(imbalance_severity)

        # Determine if sample weights should be used
        should_use_weights = imbalance_severity in [
            ImbalanceSeverity.MODERATE,
            ImbalanceSeverity.SEVERE,
            ImbalanceSeverity.EXTREME,
        ]

        # Determine recommended presets
        if imbalance_severity == ImbalanceSeverity.EXTREME:
            # For extreme imbalance, simpler models often work better
            recommended_presets = "good_quality"
        elif imbalance_severity == ImbalanceSeverity.SEVERE:
            recommended_presets = "high_quality"
        else:
            recommended_presets = "high_quality"

        return DataCharacteristics(
            task_type=task_type,
            num_classes=num_classes,
            class_distribution=class_distribution,
            imbalance_ratio=imbalance_ratio,
            imbalance_severity=imbalance_severity,
            minority_class_count=minority_count,
            total_samples=total_samples,
            recommended_metric=recommended_metric,
            recommended_secondary_metrics=secondary_metrics,
            should_use_sample_weights=should_use_weights,
            recommended_presets=recommended_presets,
        )

    def _analyze_regression(
        self,
        target: pd.Series,
        total_samples: int,
    ) -> DataCharacteristics:
        """Analyze regression task data."""
        # Check for log-scale data (large range with positive values)
        target_clean = target.dropna()
        if len(target_clean) == 0:
            return DataCharacteristics(
                task_type="regression",
                total_samples=total_samples,
                recommended_metric="root_mean_squared_error",
                recommended_secondary_metrics=["mean_absolute_error", "r2"],
            )

        min_val = target_clean.min()
        max_val = target_clean.max()
        range_ratio = max_val / min_val if min_val > 0 else float('inf')

        # Check for outliers
        q1, q3 = target_clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_threshold = 3 * iqr
        has_outliers = ((target_clean < q1 - outlier_threshold) | (target_clean > q3 + outlier_threshold)).any()

        if range_ratio > 1000 and min_val > 0:
            # Log-scale data - use MAPE
            recommended_metric = "mean_absolute_percentage_error"
            secondary_metrics = ["root_mean_squared_error", "r2"]
        elif has_outliers:
            # Robust metric for outliers
            recommended_metric = "median_absolute_error"
            secondary_metrics = ["root_mean_squared_error", "mean_absolute_error"]
        else:
            # Standard regression
            recommended_metric = "root_mean_squared_error"
            secondary_metrics = ["mean_absolute_error", "r2"]

        return DataCharacteristics(
            task_type="regression",
            total_samples=total_samples,
            recommended_metric=recommended_metric,
            recommended_secondary_metrics=secondary_metrics,
            recommended_presets="high_quality",
        )

    def _get_imbalance_severity(self, ratio: float) -> ImbalanceSeverity:
        """Determine imbalance severity from ratio."""
        if ratio < 1.5:
            return ImbalanceSeverity.NONE
        elif ratio < 3:
            return ImbalanceSeverity.MILD
        elif ratio < 10:
            return ImbalanceSeverity.MODERATE
        elif ratio < 50:
            return ImbalanceSeverity.SEVERE
        else:
            return ImbalanceSeverity.EXTREME

    def _get_binary_secondary_metrics(self, severity: ImbalanceSeverity) -> list[str]:
        """Get secondary metrics for binary classification."""
        if severity in [ImbalanceSeverity.SEVERE, ImbalanceSeverity.EXTREME]:
            return ["precision", "recall", "average_precision", "roc_auc"]
        elif severity == ImbalanceSeverity.MODERATE:
            return ["precision", "recall", "roc_auc"]
        else:
            return ["f1", "roc_auc"]

    def _get_multiclass_secondary_metrics(self, severity: ImbalanceSeverity) -> list[str]:
        """Get secondary metrics for multiclass classification."""
        if severity in [ImbalanceSeverity.SEVERE, ImbalanceSeverity.EXTREME]:
            return ["balanced_accuracy", "f1_weighted", "log_loss"]
        elif severity == ImbalanceSeverity.MODERATE:
            return ["balanced_accuracy", "accuracy"]
        else:
            return ["f1_macro", "balanced_accuracy"]

    def get_autogluon_metric(self, metric: str) -> str:
        """Convert metric name to AutoGluon-compatible format."""
        return self.AUTOGLUON_METRIC_MAP.get(metric, metric)

    def get_sample_weights(
        self,
        target: pd.Series,
        method: str = "balanced",
    ) -> Optional[pd.Series]:
        """Calculate sample weights for imbalanced data.

        Args:
            target: Target column
            method: Weighting method ('balanced', 'sqrt', or 'custom')

        Returns:
            Series of sample weights, or None if not applicable
        """
        value_counts = target.value_counts()
        n_samples = len(target)
        n_classes = len(value_counts)

        if method == "balanced":
            # sklearn-style balanced weights: n_samples / (n_classes * n_samples_per_class)
            weights = {}
            for cls, count in value_counts.items():
                weights[cls] = n_samples / (n_classes * count)
        elif method == "sqrt":
            # Square root weighting - less aggressive than balanced
            max_count = value_counts.max()
            weights = {}
            for cls, count in value_counts.items():
                weights[cls] = np.sqrt(max_count / count)
        else:
            return None

        # Normalize weights to have mean of 1
        mean_weight = sum(weights[cls] * value_counts[cls] / n_samples for cls in weights)
        weights = {cls: w / mean_weight for cls, w in weights.items()}

        # Map weights to samples
        sample_weights = target.map(weights)

        logger.info(f"Sample weights calculated using '{method}' method: {weights}")

        return sample_weights


def get_smart_metric_recommendations(
    df: pd.DataFrame,
    target_column: str,
    task_type: Optional[str] = None,
) -> DataCharacteristics:
    """Convenience function to get metric recommendations.

    Args:
        df: Training DataFrame
        target_column: Name of target column
        task_type: Optional task type override

    Returns:
        DataCharacteristics with all recommendations
    """
    selector = SmartMetricSelector()
    return selector.analyze_data(df, target_column, task_type)


def format_metric_recommendations(characteristics: DataCharacteristics) -> str:
    """Format recommendations as a human-readable string."""
    lines = [
        "=" * 60,
        "SMART METRIC RECOMMENDATIONS",
        "=" * 60,
        f"Task Type: {characteristics.task_type}",
        f"Total Samples: {characteristics.total_samples:,}",
    ]

    if characteristics.task_type in ["binary", "multiclass"]:
        lines.extend([
            f"Number of Classes: {characteristics.num_classes}",
            f"Class Imbalance Ratio: {characteristics.imbalance_ratio:.2f}:1",
            f"Imbalance Severity: {characteristics.imbalance_severity.value.upper()}",
            f"Minority Class Count: {characteristics.minority_class_count:,}",
        ])

    lines.extend([
        "",
        "RECOMMENDATIONS:",
        f"  Primary Metric: {characteristics.recommended_metric}",
        f"  Secondary Metrics: {', '.join(characteristics.recommended_secondary_metrics)}",
        f"  Use Sample Weights: {'Yes' if characteristics.should_use_sample_weights else 'No'}",
        f"  Recommended Presets: {characteristics.recommended_presets}",
        "=" * 60,
    ])

    return "\n".join(lines)

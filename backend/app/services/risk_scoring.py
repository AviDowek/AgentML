"""Risk-adjusted scoring and "too good to be true" detection.

This module implements:
1. Risk-adjusted score computation with penalties for overfitting/leakage risks
2. "Too good to be true" heuristic for time-based classification tasks
3. Promotion guardrails logic

Prompt 5 requirements implementation.
"""
import logging
from typing import Any, Dict, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_risk_adjusted_score(
    primary_metric: float,
    overfitting_risk: str,
    leakage_suspected: bool,
    time_split_suspicious: bool,
) -> float:
    """Compute a risk-adjusted score for a model.

    Applies penalties based on identified risks to discount overly optimistic
    performance metrics.

    Args:
        primary_metric: The raw primary metric value (e.g., accuracy, AUC, RMSE)
        overfitting_risk: Risk level: "low", "medium", or "high"
        leakage_suspected: Whether data leakage is suspected
        time_split_suspicious: Whether time-based split issues were detected

    Returns:
        Risk-adjusted score (primary_metric minus penalties)

    Penalty schedule:
        - overfitting_risk "medium": -0.05
        - overfitting_risk "high": -0.10
        - leakage_suspected: -0.15
        - time_split_suspicious: -0.05
    """
    penalty = 0.0

    # Overfitting risk penalty
    if overfitting_risk == "medium":
        penalty += 0.05
        logger.debug(f"Applied 0.05 penalty for medium overfitting risk")
    elif overfitting_risk == "high":
        penalty += 0.10
        logger.debug(f"Applied 0.10 penalty for high overfitting risk")

    # Leakage penalty (most severe)
    if leakage_suspected:
        penalty += 0.15
        logger.debug(f"Applied 0.15 penalty for suspected data leakage")

    # Time-split penalty
    if time_split_suspicious:
        penalty += 0.05
        logger.debug(f"Applied 0.05 penalty for suspicious time-based split")

    risk_adjusted = primary_metric - penalty

    logger.info(
        f"Risk-adjusted score: {primary_metric:.4f} - {penalty:.4f} penalty = {risk_adjusted:.4f} "
        f"(overfitting={overfitting_risk}, leakage={leakage_suspected}, time_split={time_split_suspicious})"
    )

    return risk_adjusted


def check_too_good_to_be_true(
    is_time_based: bool,
    task_type: str,
    best_val_metric: Optional[float],
    primary_metric: str,
    additional_metrics: Optional[Dict[str, float]] = None,
    expected_metric_range: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Check if model performance is "too good to be true" for time-based classification.

    For time-based prediction tasks (e.g., stock prediction, churn prediction),
    very high performance metrics often indicate data leakage or unrealistic
    evaluation rather than genuine predictive ability.

    Enhanced in Prompt 7 to use expected_metric_range from Problem Framer.

    Args:
        is_time_based: Whether this is a time-based prediction task
        task_type: The ML task type (binary, multiclass, regression)
        best_val_metric: The best validation metric value
        primary_metric: Name of the primary metric
        additional_metrics: Optional dict of additional metrics to check
        expected_metric_range: Optional dict with expected realistic performance range
                             from Problem Framer. Format:
                             {"metric": "roc_auc", "lower_bound": 0.60, "upper_bound": 0.75}

    Returns:
        Tuple of (is_too_good, warning_message)
    """
    additional_metrics = additional_metrics or {}
    warnings = []

    # === Prompt 7: Check against expected_metric_range from Problem Framer ===
    if expected_metric_range and best_val_metric is not None:
        expected_metric = expected_metric_range.get("metric", "").lower()
        upper_bound = expected_metric_range.get("upper_bound")
        lower_bound = expected_metric_range.get("lower_bound")

        # Check if the primary metric matches the expected metric
        metric_lower = primary_metric.lower()
        metric_matches = (
            expected_metric in metric_lower or
            metric_lower in expected_metric or
            (expected_metric == "roc_auc" and ("auc" in metric_lower or "roc" in metric_lower))
        )

        if metric_matches and upper_bound is not None:
            # If actual metric exceeds expected upper bound by significant margin
            if best_val_metric > upper_bound:
                margin = best_val_metric - upper_bound
                warnings.append(
                    f"Model achieves {primary_metric}={best_val_metric:.3f}, which exceeds the "
                    f"expected realistic upper bound of {upper_bound:.2f} (from Problem Framer analysis) "
                    f"by {margin:.3f}. Expected range was [{lower_bound:.2f}-{upper_bound:.2f}]. "
                    f"This warrants investigation for data leakage or evaluation issues."
                )
                logger.warning(
                    f"TGTBT: actual={best_val_metric:.3f} > expected_upper={upper_bound:.2f} "
                    f"for {primary_metric}"
                )
                # If exceeds by more than 0.10, definitely flag
                if margin > 0.10:
                    logger.warning("TGTBT: Exceeds expected range by >0.10 - high concern")

    # Only apply hardcoded heuristic for time-based classification tasks
    is_classification = task_type in ("binary", "multiclass", "classification")
    if not is_time_based or not is_classification:
        # Still return any expected_metric_range warnings
        if warnings:
            return True, " ".join(warnings)
        return False, None

    # Check AUC threshold
    metric_lower = primary_metric.lower()
    auc_value = None

    if "auc" in metric_lower or "roc" in metric_lower:
        auc_value = best_val_metric
    else:
        # Check additional metrics for AUC
        for key, value in additional_metrics.items():
            if "auc" in key.lower() or "roc" in key.lower():
                auc_value = value
                break

    if auc_value is not None and auc_value > 0.80:
        warnings.append(
            f"AUC of {auc_value:.3f} on time-based classification is suspiciously high. "
            f"In domains like stock prediction or churn forecasting, AUC > 0.80 often "
            f"indicates data leakage or look-ahead bias rather than genuine predictive power."
        )

    # Check MCC threshold
    mcc_value = None
    for key, value in additional_metrics.items():
        if "mcc" in key.lower() or "matthews" in key.lower():
            mcc_value = value
            break

    if mcc_value is not None and mcc_value > 0.50:
        warnings.append(
            f"MCC of {mcc_value:.3f} on time-based classification is suspiciously high. "
            f"MCC > 0.50 on forward-looking prediction tasks typically suggests "
            f"data contamination or improper temporal splits."
        )

    # Also check accuracy for time-based binary classification
    if task_type == "binary":
        acc_value = None
        if "accuracy" in metric_lower:
            acc_value = best_val_metric
        else:
            for key, value in additional_metrics.items():
                if "accuracy" in key.lower():
                    acc_value = value
                    break

        if acc_value is not None and acc_value > 0.85:
            warnings.append(
                f"Accuracy of {acc_value:.3f} on time-based binary classification "
                f"warrants careful review. Consider whether temporal information "
                f"could be leaking into features."
            )

    if warnings:
        combined_warning = " ".join(warnings)
        logger.warning(f"TOO GOOD TO BE TRUE: {combined_warning}")
        return True, combined_warning

    return False, None


def get_model_risk_status(
    overfitting_risk: str,
    leakage_suspected: bool,
    time_split_suspicious: bool,
    too_good_to_be_true: bool = False,
) -> Tuple[str, bool, str]:
    """Determine overall model risk status for promotion gating.

    Args:
        overfitting_risk: Risk level from robustness audit
        leakage_suspected: Whether leakage was detected
        time_split_suspicious: Whether time-split issues were detected
        too_good_to_be_true: Whether model triggered TGTBT heuristic

    Returns:
        Tuple of (risk_level, requires_override, reason)
        - risk_level: "low", "medium", "high", or "critical"
        - requires_override: Whether promotion requires explicit override
        - reason: Human-readable explanation
    """
    reasons = []

    # Determine if override is required
    requires_override = False

    if leakage_suspected:
        reasons.append("Data leakage suspected from label-shuffle test")
        requires_override = True

    if overfitting_risk == "high":
        reasons.append("High overfitting risk detected")
        requires_override = True

    if too_good_to_be_true:
        reasons.append("Performance appears 'too good to be true' for time-based prediction")
        requires_override = True

    if time_split_suspicious:
        reasons.append("Time-based data may have temporal leakage due to split strategy")

    # Determine overall risk level
    if leakage_suspected or too_good_to_be_true:
        risk_level = "critical"
    elif overfitting_risk == "high":
        risk_level = "high"
    elif overfitting_risk == "medium" or time_split_suspicious:
        risk_level = "medium"
    else:
        risk_level = "low"

    reason = "; ".join(reasons) if reasons else "No significant risks identified"

    return risk_level, requires_override, reason


def format_promotion_block_message(
    risk_level: str,
    reasons: str,
) -> str:
    """Format the error message when promotion is blocked.

    Args:
        risk_level: The risk level (high, critical)
        reasons: Human-readable reasons for the block

    Returns:
        Formatted error message
    """
    message = (
        f"Model promotion blocked due to {risk_level} risk.\n\n"
        f"Reason: {reasons}\n\n"
        f"To proceed with promotion, provide an override_reason explaining why "
        f"you believe this model is safe to promote despite the identified risks. "
        f"Your override reason will be logged in the lab notebook for audit purposes."
    )
    return message

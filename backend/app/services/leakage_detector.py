"""Feature Leakage Detection Service (Prompt 6).

This module provides heuristics to detect potential data leakage in features,
particularly for time-based prediction tasks.

Detection methods:
1. Name-based suspicion - keywords suggesting future/target information
2. Correlation-based suspicion - features highly correlated with target
3. Time-order check - features that may use future information
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Keywords that suggest a feature may contain target/future information
# Note: Using (?:^|_) and (?:$|_) to match word boundaries with underscores
SUSPICIOUS_NAME_PATTERNS = [
    # Target-related - match "label", "_label", "label_", etc.
    (r"(?i)(?:^|_)label(?:$|_)", "Feature name contains 'label', may be target-related"),
    (r"(?i)(?:^|_)target(?:$|_)", "Feature name contains 'target', may be the prediction target"),
    (r"(?i)^y_", "Feature name starts with 'y_', commonly used for targets"),
    (r"(?i)ground.?truth", "Feature name suggests ground truth/label"),
    (r"(?i)(?:^|_)outcome(?:$|_)", "Feature name contains 'outcome', may leak target"),
    (r"(?i)(?:^|_)result(?:$|_)", "Feature name contains 'result', may leak target"),
    (r"(?i)(?:^|_)actual(?:$|_)", "Feature name contains 'actual', may leak target"),

    # Future-looking - match "future", "_future", "future_", etc.
    (r"(?i)(?:^|_)future(?:$|_)", "Feature name suggests future information"),
    (r"(?i)(?:^|_)next(?:$|_)", "Feature name suggests next period data"),
    (r"(?i)(?:^|_)forward(?:$|_)", "Feature name suggests forward-looking data"),
    (r"(?i)t\+\d+", "Feature name suggests future time offset (t+N)"),
    (r"(?i)_\+\d+d", "Feature name suggests future day offset (+Nd)"),
    (r"(?i)lead_", "Feature name suggests lead/forward operation"),

    # Time-based potential leakage for financial data
    (r"(?i)ret_\d+d$", "Feature may be forward return (ret_Nd); verify it's computed from past data only"),
    (r"(?i)return_\d+d$", "Feature may be forward return; verify computation direction"),
    (r"(?i)price_change_\d+d$", "Feature may use future price changes"),

    # Cancellation/churn specific
    (r"(?i)cancellation.?date", "Cancellation date may leak churn outcome"),
    (r"(?i)churn.?date", "Churn date may leak churn outcome"),
    (r"(?i)end.?date", "End date may leak subscription/account outcome"),
    (r"(?i)termination", "Termination info may leak outcome"),

    # Post-event features
    (r"(?i)post_", "Feature name suggests post-event data"),
    (r"(?i)after_", "Feature name suggests after-event data"),
    (r"(?i)final_", "Feature name suggests final/concluded state"),
]

# Patterns for features that are likely safe despite suspicious-sounding names
SAFE_PATTERNS = [
    r"(?i)^target_encoding_",  # Target encoding is a valid technique
    r"(?i)lag_",  # Lag features are typically safe
    r"(?i)_lag\d+",
    r"(?i)rolling_",  # Rolling features from past are safe
    r"(?i)ma_\d+",  # Moving averages
    r"(?i)ema_\d+",  # Exponential moving averages
    r"(?i)past_",  # Explicitly past data
    r"(?i)historical_",
    r"(?i)prev_",  # Previous period
    r"(?i)last_",  # Last period
]


def detect_potential_leakage_features(
    df: pd.DataFrame,
    target_column: str,
    time_column: Optional[str] = None,
    feature_lineage: Optional[Dict[str, Dict[str, Any]]] = None,
    correlation_threshold: float = 0.9,
) -> List[Dict[str, Any]]:
    """Detect features that may cause data leakage.

    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        time_column: Optional name of the time/date column
        feature_lineage: Optional dict mapping feature names to their computation metadata
                        Expected format: {feature_name: {"window": "forward"/"backward", ...}}
        correlation_threshold: Threshold for correlation-based suspicion (default 0.9)

    Returns:
        List of suspicious feature dictionaries:
        [
            {
                "column": "feature_name",
                "reason": "Explanation of why it's suspicious",
                "severity": "low" | "medium" | "high",
                "detection_method": "name" | "correlation" | "lineage"
            },
            ...
        ]
    """
    suspects: List[Dict[str, Any]] = []
    feature_columns = [c for c in df.columns if c != target_column and c != time_column]

    if not feature_columns:
        logger.warning("No feature columns to analyze for leakage")
        return suspects

    # 1. Name-based detection
    name_suspects = _detect_by_name(feature_columns, target_column)
    suspects.extend(name_suspects)

    # 2. Correlation-based detection
    if target_column in df.columns:
        corr_suspects = _detect_by_correlation(
            df, feature_columns, target_column, correlation_threshold
        )
        # Don't duplicate if already flagged by name
        flagged_columns = {s["column"] for s in suspects}
        for suspect in corr_suspects:
            if suspect["column"] not in flagged_columns:
                suspects.append(suspect)
            else:
                # Upgrade severity if also has high correlation
                for s in suspects:
                    if s["column"] == suspect["column"]:
                        s["severity"] = "high"
                        s["reason"] += f" Additionally, {suspect['reason']}"
                        break

    # 3. Lineage-based detection (if provided)
    if feature_lineage:
        lineage_suspects = _detect_by_lineage(feature_columns, feature_lineage)
        flagged_columns = {s["column"] for s in suspects}
        for suspect in lineage_suspects:
            if suspect["column"] not in flagged_columns:
                suspects.append(suspect)
            else:
                # Upgrade to high severity
                for s in suspects:
                    if s["column"] == suspect["column"]:
                        s["severity"] = "high"
                        s["reason"] += f" {suspect['reason']}"
                        break

    # Sort by severity (high first)
    severity_order = {"high": 0, "medium": 1, "low": 2}
    suspects.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))

    logger.info(f"Detected {len(suspects)} potential leakage features")
    for suspect in suspects:
        logger.warning(
            f"Potential leakage: {suspect['column']} - {suspect['reason']} "
            f"(severity: {suspect['severity']})"
        )

    return suspects


def _detect_by_name(
    feature_columns: List[str],
    target_column: str,
) -> List[Dict[str, Any]]:
    """Detect suspicious features by name patterns."""
    suspects = []

    for col in feature_columns:
        # Skip if it matches safe patterns
        is_safe = any(re.search(pattern, col) for pattern, _ in [(p, "") for p in SAFE_PATTERNS])
        if is_safe:
            continue

        # Check exact match with target column
        if col.lower() == target_column.lower():
            suspects.append({
                "column": col,
                "reason": "Feature has the same name as target column",
                "severity": "high",
                "detection_method": "name",
            })
            continue

        # Check against suspicious patterns
        for pattern, reason in SUSPICIOUS_NAME_PATTERNS:
            if re.search(pattern, col):
                # Determine severity based on pattern type
                if any(kw in pattern.lower() for kw in ["target", "label", "ground", "y_"]):
                    severity = "high"
                elif any(kw in pattern.lower() for kw in ["future", "next", "forward", r"t\+"]):
                    severity = "high"
                elif any(kw in pattern.lower() for kw in ["cancellation", "churn", "termination"]):
                    severity = "high"
                else:
                    severity = "medium"

                suspects.append({
                    "column": col,
                    "reason": reason,
                    "severity": severity,
                    "detection_method": "name",
                })
                break  # Only flag once per column

    return suspects


def _detect_by_correlation(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    threshold: float = 0.9,
) -> List[Dict[str, Any]]:
    """Detect features with suspiciously high correlation to target."""
    suspects = []

    # Get target as numeric if possible
    target = df[target_column]
    if target.dtype == 'object' or target.dtype.name == 'category':
        try:
            # Try to encode categorical target
            target = pd.factorize(target)[0]
        except Exception:
            logger.debug(f"Could not encode target {target_column} for correlation analysis")
            return suspects

    # Compute correlations for numeric features
    for col in feature_columns:
        if col == target_column:
            continue

        try:
            feature = df[col]

            # Skip non-numeric features
            if not np.issubdtype(feature.dtype, np.number):
                continue

            # Handle missing values
            valid_mask = ~(feature.isna() | pd.isna(target))
            if valid_mask.sum() < 10:  # Need at least 10 valid pairs
                continue

            # Compute correlation
            corr = np.corrcoef(feature[valid_mask], target[valid_mask])[0, 1]

            if np.isnan(corr):
                continue

            if abs(corr) > threshold:
                suspects.append({
                    "column": col,
                    "reason": f"Very high correlation with target (r={corr:.3f}); "
                              f"may be a direct transformation of the target",
                    "severity": "high" if abs(corr) > 0.95 else "medium",
                    "detection_method": "correlation",
                    "correlation": float(corr),
                })
            elif abs(corr) > 0.8:
                # Flag high but not extreme correlations as informational
                suspects.append({
                    "column": col,
                    "reason": f"High correlation with target (r={corr:.3f}); "
                              f"verify this isn't leaking target information",
                    "severity": "low",
                    "detection_method": "correlation",
                    "correlation": float(corr),
                })

        except Exception as e:
            logger.debug(f"Could not compute correlation for {col}: {e}")

    return suspects


def _detect_by_lineage(
    feature_columns: List[str],
    feature_lineage: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect features that use forward-looking data based on lineage metadata."""
    suspects = []

    for col in feature_columns:
        if col not in feature_lineage:
            continue

        lineage = feature_lineage[col]

        # Check for forward-looking windows
        window_type = lineage.get("window", "").lower()
        if window_type in ("forward", "future", "lead"):
            suspects.append({
                "column": col,
                "reason": f"Feature lineage indicates forward-looking window ({window_type}); "
                          f"this uses future data that won't be available at prediction time",
                "severity": "high",
                "detection_method": "lineage",
            })
            continue

        # Check for time offset
        time_offset = lineage.get("time_offset", 0)
        if isinstance(time_offset, (int, float)) and time_offset > 0:
            suspects.append({
                "column": col,
                "reason": f"Feature lineage shows positive time offset (+{time_offset}); "
                          f"this looks into the future",
                "severity": "high",
                "detection_method": "lineage",
            })
            continue

        # Check for post-event flag
        if lineage.get("post_event", False):
            suspects.append({
                "column": col,
                "reason": "Feature lineage indicates it's computed after the event; "
                          "not available at prediction time",
                "severity": "high",
                "detection_method": "lineage",
            })

    return suspects


def check_leakage_in_important_features(
    leakage_candidates: List[Dict[str, Any]],
    feature_importances: Dict[str, float],
    top_n: int = 10,
    importance_threshold: float = 0.05,
) -> Tuple[bool, List[Dict[str, Any]], str]:
    """Check if any leakage candidates are among the most important features.

    Args:
        leakage_candidates: List of potential leakage features from detect_potential_leakage_features
        feature_importances: Dict mapping feature names to importance scores
        top_n: Number of top features to consider
        importance_threshold: Minimum importance to flag (as fraction of max importance)

    Returns:
        Tuple of:
        - has_concerning_leakage: bool indicating if concerning features found
        - concerning_features: List of leakage features that are also important
        - warning_message: Human-readable warning
    """
    if not leakage_candidates or not feature_importances:
        return False, [], ""

    leakage_columns = {c["column"] for c in leakage_candidates}
    leakage_lookup = {c["column"]: c for c in leakage_candidates}

    # Sort features by importance
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
        reverse=True,
    )

    # Get top N features
    top_features = sorted_features[:top_n]

    # Find max importance for threshold calculation
    max_importance = max(abs(v) for _, v in sorted_features) if sorted_features else 1

    concerning = []
    for feat_name, importance in top_features:
        if feat_name in leakage_columns:
            relative_importance = abs(importance) / max_importance if max_importance > 0 else 0
            if relative_importance >= importance_threshold:
                leakage_info = leakage_lookup[feat_name]
                concerning.append({
                    **leakage_info,
                    "importance": float(importance),
                    "importance_rank": top_features.index((feat_name, importance)) + 1,
                })

    if not concerning:
        return False, [], ""

    # Build warning message
    feature_list = ", ".join(f["column"] for f in concerning)
    warning = (
        f"Model relies heavily on {len(concerning)} suspicious feature(s): {feature_list}. "
        f"These features were flagged as potential data leakage. "
        f"The model's performance may not generalize to production."
    )

    return True, concerning, warning


def get_leakage_summary(
    leakage_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a summary of detected leakage candidates.

    Args:
        leakage_candidates: List of potential leakage features

    Returns:
        Summary dict with counts by severity and detection method
    """
    if not leakage_candidates:
        return {
            "total_count": 0,
            "high_severity_count": 0,
            "medium_severity_count": 0,
            "low_severity_count": 0,
            "by_method": {},
            "high_severity_features": [],
        }

    high = [c for c in leakage_candidates if c.get("severity") == "high"]
    medium = [c for c in leakage_candidates if c.get("severity") == "medium"]
    low = [c for c in leakage_candidates if c.get("severity") == "low"]

    by_method = {}
    for c in leakage_candidates:
        method = c.get("detection_method", "unknown")
        by_method[method] = by_method.get(method, 0) + 1

    return {
        "total_count": len(leakage_candidates),
        "high_severity_count": len(high),
        "medium_severity_count": len(medium),
        "low_severity_count": len(low),
        "by_method": by_method,
        "high_severity_features": [c["column"] for c in high],
    }

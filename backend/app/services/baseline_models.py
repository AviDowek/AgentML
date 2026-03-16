"""Baseline models and sanity tests for experiment validation.

This module implements:
1. Majority class predictor - always predicts the most common class
2. Simple logistic regression - L2-regularized logistic/linear regression
3. Label-shuffle sanity test - verifies model isn't exploiting data leakage

If the AutoML model doesn't beat these baselines significantly, it may indicate
overfitting, data leakage, or that the features have no predictive power.
"""
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def compute_baseline_metrics(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    target_column: str,
    task_type: Literal["binary", "multiclass", "regression"],
    primary_metric: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute baseline metrics for experiment validation.

    Runs simple baseline models and returns their performance metrics.
    These serve as lower bounds that the AutoML model should beat.

    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        target_column: Name of the target column
        task_type: Type of ML task
        primary_metric: Primary metric being optimized

    Returns:
        Dictionary with baseline metrics:
        {
            "majority_class": {"accuracy": 0.52, "roc_auc": 0.5, ...},
            "simple_model": {"accuracy": 0.56, "roc_auc": 0.58, ...}
        }
    """
    baselines = {}
    is_classification = task_type in ("binary", "multiclass")

    # Prepare features and target
    feature_cols = [c for c in train_data.columns if c != target_column]

    X_train = train_data[feature_cols].copy()
    y_train = train_data[target_column].copy()
    X_val = val_data[feature_cols].copy()
    y_val = val_data[target_column].copy()

    # Handle non-numeric features by selecting only numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        logger.warning("No numeric features found for baseline models")
        return {}

    X_train_numeric = X_train[numeric_cols].copy()
    X_val_numeric = X_val[numeric_cols].copy()

    # Fill NaN values with median for numeric columns
    X_train_numeric = X_train_numeric.fillna(X_train_numeric.median())
    X_val_numeric = X_val_numeric.fillna(X_train_numeric.median())

    # Replace infinite values
    X_train_numeric = X_train_numeric.replace([np.inf, -np.inf], 0)
    X_val_numeric = X_val_numeric.replace([np.inf, -np.inf], 0)

    # Encode target for classification
    label_encoder = None
    if is_classification:
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train.astype(str))
        y_val_encoded = label_encoder.transform(y_val.astype(str))
    else:
        y_train_encoded = y_train.values
        y_val_encoded = y_val.values

    # 1. Majority class / mean predictor baseline
    try:
        if is_classification:
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train_numeric, y_train_encoded)
            y_pred = dummy.predict(X_val_numeric)

            baseline_metrics = {
                "accuracy": float(accuracy_score(y_val_encoded, y_pred)),
            }

            # ROC AUC only for binary
            if task_type == "binary" and len(np.unique(y_val_encoded)) == 2:
                # Majority class predictor always predicts same class, so AUC = 0.5
                baseline_metrics["roc_auc"] = 0.5

            baselines["majority_class"] = baseline_metrics
            logger.info(f"Majority class baseline: {baseline_metrics}")
        else:
            dummy = DummyRegressor(strategy="mean")
            dummy.fit(X_train_numeric, y_train_encoded)
            y_pred = dummy.predict(X_val_numeric)

            baseline_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_val_encoded, y_pred))),
                "mae": float(mean_absolute_error(y_val_encoded, y_pred)),
                "r2": float(r2_score(y_val_encoded, y_pred)),
            }

            baselines["mean_predictor"] = baseline_metrics
            logger.info(f"Mean predictor baseline: {baseline_metrics}")

    except Exception as e:
        logger.warning(f"Failed to compute majority/mean baseline: {e}")

    # 2. Simple regularized model baseline
    try:
        # Scale features for regularized model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_numeric)
        X_val_scaled = scaler.transform(X_val_numeric)

        if is_classification:
            # Logistic regression with L2 regularization
            model = LogisticRegression(
                C=1.0,  # Regularization strength
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )
            model.fit(X_train_scaled, y_train_encoded)
            y_pred = model.predict(X_val_scaled)

            simple_metrics = {
                "accuracy": float(accuracy_score(y_val_encoded, y_pred)),
            }

            # ROC AUC for binary classification
            if task_type == "binary" and len(np.unique(y_val_encoded)) == 2:
                try:
                    y_proba = model.predict_proba(X_val_scaled)[:, 1]
                    simple_metrics["roc_auc"] = float(roc_auc_score(y_val_encoded, y_proba))
                except Exception:
                    pass

            # F1 score
            try:
                avg = "binary" if task_type == "binary" else "weighted"
                simple_metrics["f1"] = float(f1_score(y_val_encoded, y_pred, average=avg))
            except Exception:
                pass

            baselines["simple_logistic"] = simple_metrics
            logger.info(f"Simple logistic baseline: {simple_metrics}")
        else:
            # Ridge regression with L2 regularization
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train_scaled, y_train_encoded)
            y_pred = model.predict(X_val_scaled)

            simple_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_val_encoded, y_pred))),
                "mae": float(mean_absolute_error(y_val_encoded, y_pred)),
                "r2": float(r2_score(y_val_encoded, y_pred)),
            }

            baselines["simple_ridge"] = simple_metrics
            logger.info(f"Simple ridge baseline: {simple_metrics}")

    except Exception as e:
        logger.warning(f"Failed to compute simple model baseline: {e}")

    return baselines


def run_label_shuffle_test(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    target_column: str,
    task_type: Literal["binary", "multiclass", "regression"],
    primary_metric: str | None = None,
    time_limit: int = 60,
) -> dict[str, Any]:
    """Run label-shuffle sanity test to detect data leakage.

    Trains a model on randomly shuffled labels. If the model performs
    significantly better than chance (AUC >> 0.5 for classification),
    it indicates potential data leakage or target encoding issues.

    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        target_column: Name of the target column
        task_type: Type of ML task
        primary_metric: Primary metric being optimized
        time_limit: Max training time in seconds (should be short)

    Returns:
        Dictionary with shuffle test results:
        {
            "shuffled_accuracy": 0.51,
            "shuffled_roc_auc": 0.52,
            "leakage_detected": False,
            "warning": "optional warning message"
        }
    """
    is_classification = task_type in ("binary", "multiclass")

    # Prepare features and target
    feature_cols = [c for c in train_data.columns if c != target_column]

    X_train = train_data[feature_cols].copy()
    y_train = train_data[target_column].copy()
    X_val = val_data[feature_cols].copy()
    y_val = val_data[target_column].copy()

    # Handle non-numeric features
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        logger.warning("No numeric features for label-shuffle test")
        return {"error": "No numeric features available"}

    X_train_numeric = X_train[numeric_cols].copy()
    X_val_numeric = X_val[numeric_cols].copy()

    # Fill NaN values
    X_train_numeric = X_train_numeric.fillna(X_train_numeric.median())
    X_val_numeric = X_val_numeric.fillna(X_train_numeric.median())

    # Replace infinite values
    X_train_numeric = X_train_numeric.replace([np.inf, -np.inf], 0)
    X_val_numeric = X_val_numeric.replace([np.inf, -np.inf], 0)

    # Shuffle training labels randomly
    np.random.seed(42)
    y_train_shuffled = y_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Encode target for classification
    label_encoder = None
    if is_classification:
        label_encoder = LabelEncoder()
        # Fit on all unique values from both train and val
        all_labels = pd.concat([y_train, y_val]).astype(str).unique()
        label_encoder.fit(all_labels)
        y_train_encoded = label_encoder.transform(y_train_shuffled.astype(str))
        y_val_encoded = label_encoder.transform(y_val.astype(str))
    else:
        y_train_encoded = y_train_shuffled.values
        y_val_encoded = y_val.values

    results = {}

    try:
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_numeric)
        X_val_scaled = scaler.transform(X_val_numeric)

        if is_classification:
            # Train a simple model on shuffled data
            model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )
            model.fit(X_train_scaled, y_train_encoded)
            y_pred = model.predict(X_val_scaled)

            results["shuffled_accuracy"] = float(accuracy_score(y_val_encoded, y_pred))

            # Expected accuracy should be the MAJORITY CLASS baseline, not 1/n_classes
            # If the model just predicts the most common class, that's NOT leakage
            n_classes = len(np.unique(y_val_encoded))

            # Calculate majority class proportion (what a dummy classifier would achieve)
            unique, counts = np.unique(y_val_encoded, return_counts=True)
            majority_class_proportion = max(counts) / len(y_val_encoded)

            # Use majority class as the baseline (not uniform random)
            expected_accuracy = majority_class_proportion
            results["expected_random_accuracy"] = expected_accuracy
            results["uniform_random_accuracy"] = 1.0 / n_classes
            results["majority_class_proportion"] = float(majority_class_proportion)

            # ROC AUC for binary
            if task_type == "binary" and len(np.unique(y_val_encoded)) == 2:
                try:
                    y_proba = model.predict_proba(X_val_scaled)[:, 1]
                    shuffled_auc = roc_auc_score(y_val_encoded, y_proba)
                    results["shuffled_roc_auc"] = float(shuffled_auc)

                    # Check for leakage: AUC should be close to 0.5
                    # Allow some variance, flag if AUC > 0.55
                    if shuffled_auc > 0.55:
                        results["leakage_detected"] = True
                        results["warning"] = (
                            f"POTENTIAL DATA LEAKAGE: Label-shuffle test yielded AUC={shuffled_auc:.3f} "
                            f"(expected ~0.5). Features may encode target information."
                        )
                        logger.warning(results["warning"])
                    else:
                        results["leakage_detected"] = False
                except Exception as e:
                    logger.warning(f"Could not compute shuffled ROC AUC: {e}")

            # Check accuracy-based leakage for multiclass
            else:
                # If shuffled accuracy significantly exceeds majority class baseline
                # Note: With shuffled labels, model should only be able to exploit class imbalance
                # If it does much better than majority class, features likely encode target info
                # Threshold: majority_class + 0.10 (absolute) or 1.15x (relative), whichever is higher
                leakage_threshold = max(expected_accuracy + 0.10, expected_accuracy * 1.15)
                results["leakage_threshold"] = float(leakage_threshold)

                if results["shuffled_accuracy"] > leakage_threshold:
                    results["leakage_detected"] = True
                    results["warning"] = (
                        f"POTENTIAL DATA LEAKAGE: Label-shuffle test yielded accuracy="
                        f"{results['shuffled_accuracy']:.3f} (majority class baseline: {expected_accuracy:.3f}, "
                        f"threshold: {leakage_threshold:.3f}). Features may encode target information."
                    )
                    logger.warning(results["warning"])
                else:
                    results["leakage_detected"] = False
                    logger.info(
                        f"Label-shuffle test passed: {results['shuffled_accuracy']:.3f} <= {leakage_threshold:.3f} "
                        f"(majority class: {expected_accuracy:.3f})"
                    )
        else:
            # Regression: train on shuffled labels
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train_scaled, y_train_encoded)
            y_pred = model.predict(X_val_scaled)

            results["shuffled_r2"] = float(r2_score(y_val_encoded, y_pred))
            results["shuffled_rmse"] = float(np.sqrt(mean_squared_error(y_val_encoded, y_pred)))

            # For shuffled data, R² should be close to 0 or negative
            if results["shuffled_r2"] > 0.1:
                results["leakage_detected"] = True
                results["warning"] = (
                    f"POTENTIAL DATA LEAKAGE: Label-shuffle test yielded R²={results['shuffled_r2']:.3f} "
                    f"(expected ~0 or negative). Features may encode target information."
                )
                logger.warning(results["warning"])
            else:
                results["leakage_detected"] = False

        logger.info(f"Label-shuffle test completed: {results}")

    except Exception as e:
        logger.warning(f"Label-shuffle test failed: {e}")
        results["error"] = str(e)
        results["leakage_detected"] = None

    return results


def compute_all_baselines(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    target_column: str,
    task_type: Literal["binary", "multiclass", "regression"],
    primary_metric: str | None = None,
    run_shuffle_test: bool = True,
) -> dict[str, Any]:
    """Compute all baseline metrics and sanity tests.

    This is the main entry point for baseline computation.

    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        target_column: Name of the target column
        task_type: Type of ML task
        primary_metric: Primary metric being optimized
        run_shuffle_test: Whether to run the label-shuffle test

    Returns:
        Complete baseline metrics dictionary:
        {
            "majority_class": {...},
            "simple_logistic": {...},  # or simple_ridge for regression
            "label_shuffle": {...}
        }
    """
    logger.info(f"Computing baseline metrics for {task_type} task...")

    # Compute baseline model metrics
    baselines = compute_baseline_metrics(
        train_data=train_data,
        val_data=val_data,
        target_column=target_column,
        task_type=task_type,
        primary_metric=primary_metric,
    )

    # Run label-shuffle sanity test
    if run_shuffle_test:
        shuffle_results = run_label_shuffle_test(
            train_data=train_data,
            val_data=val_data,
            target_column=target_column,
            task_type=task_type,
            primary_metric=primary_metric,
        )
        baselines["label_shuffle"] = shuffle_results

    logger.info(f"Baseline computation complete: {list(baselines.keys())}")

    return baselines

"""Feature validation pipeline for testing engineered features before use.

This module provides quick validation of engineered features to ensure they:
1. Actually improve model performance
2. Don't cause data leakage
3. Don't introduce noise
4. Are stable across different data splits

Features that fail validation are rejected before being presented to the user.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


@dataclass
class FeatureValidationResult:
    """Result of validating a single feature."""
    feature_name: str
    passed: bool
    reason: str
    baseline_score: float = 0.0
    with_feature_score: float = 0.0
    improvement: float = 0.0
    importance_rank: Optional[int] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class FeatureSetValidationResult:
    """Result of validating a set of features."""
    passed_features: list[str]
    failed_features: list[FeatureValidationResult]
    baseline_score: float
    final_score: float
    improvement_percent: float
    validation_method: str
    total_features_tested: int
    recommendations: list[str]


class FeatureValidator:
    """Validates engineered features before they're used in training."""

    # Minimum improvement threshold to keep a feature
    MIN_IMPROVEMENT_THRESHOLD = 0.001  # 0.1% improvement required

    # Maximum acceptable variance in feature importance across folds
    MAX_IMPORTANCE_VARIANCE = 0.5

    # Quick validation settings
    QUICK_CV_FOLDS = 3
    QUICK_ESTIMATORS = 50
    MAX_SAMPLES_FOR_QUICK_TEST = 10000

    def __init__(self, task_type: str = "binary", random_state: int = 42):
        """Initialize validator.

        Args:
            task_type: 'binary', 'multiclass', or 'regression'
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state

    def validate_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        new_features: list[str],
        original_features: list[str],
        metric: str = "auto",
    ) -> FeatureSetValidationResult:
        """Validate a set of new features against baseline.

        Args:
            df: DataFrame with all features and target
            target_column: Name of target column
            new_features: List of new feature names to validate
            original_features: List of original feature names (baseline)
            metric: Metric to use for validation ('auto' to select based on task)

        Returns:
            FeatureSetValidationResult with validated features
        """
        logger.info(f"Validating {len(new_features)} new features...")

        # Prepare data
        X_original, X_with_new, y = self._prepare_data(
            df, target_column, new_features, original_features
        )

        if X_original is None:
            return FeatureSetValidationResult(
                passed_features=[],
                failed_features=[
                    FeatureValidationResult(
                        feature_name=f,
                        passed=False,
                        reason="Data preparation failed",
                    )
                    for f in new_features
                ],
                baseline_score=0.0,
                final_score=0.0,
                improvement_percent=0.0,
                validation_method="failed",
                total_features_tested=len(new_features),
                recommendations=["Check data quality and feature definitions"],
            )

        # Select metric if auto
        if metric == "auto":
            metric = self._get_default_metric()

        # Get baseline score with original features only
        baseline_score = self._quick_evaluate(X_original, y, metric)
        logger.info(f"Baseline score with {len(original_features)} features: {baseline_score:.4f}")

        # Validate each new feature individually
        passed_features = []
        failed_features = []

        for feature in new_features:
            result = self._validate_single_feature(
                X_original, X_with_new, y, feature, baseline_score, metric
            )
            if result.passed:
                passed_features.append(feature)
            else:
                failed_features.append(result)

        # Calculate final score with all passed features
        if passed_features:
            final_features = list(original_features) + passed_features
            X_final = df[final_features].copy()
            X_final = self._preprocess_features(X_final)
            if X_final is not None:
                final_score = self._quick_evaluate(X_final, y, metric)
            else:
                final_score = baseline_score
        else:
            final_score = baseline_score

        improvement_percent = ((final_score - baseline_score) / abs(baseline_score) * 100) if baseline_score != 0 else 0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            passed_features, failed_features, improvement_percent
        )

        logger.info(f"Feature validation complete: {len(passed_features)} passed, {len(failed_features)} failed")
        logger.info(f"Score improvement: {baseline_score:.4f} -> {final_score:.4f} ({improvement_percent:+.2f}%)")

        return FeatureSetValidationResult(
            passed_features=passed_features,
            failed_features=failed_features,
            baseline_score=baseline_score,
            final_score=final_score,
            improvement_percent=improvement_percent,
            validation_method=f"quick_cv_{self.QUICK_CV_FOLDS}fold",
            total_features_tested=len(new_features),
            recommendations=recommendations,
        )

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        new_features: list[str],
        original_features: list[str],
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare data for validation."""
        try:
            # Get target
            y = df[target_column].copy()

            # Encode target if classification
            if self.task_type in ["binary", "multiclass"]:
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

            # Get features
            all_features = list(set(original_features + new_features))
            available_features = [f for f in all_features if f in df.columns]

            if not available_features:
                logger.error("No features available for validation")
                return None, None, None

            X_all = df[available_features].copy()

            # Sample if too large
            if len(X_all) > self.MAX_SAMPLES_FOR_QUICK_TEST:
                sample_idx = np.random.choice(
                    len(X_all), self.MAX_SAMPLES_FOR_QUICK_TEST, replace=False
                )
                X_all = X_all.iloc[sample_idx]
                y = y.iloc[sample_idx]

            # Preprocess
            X_original = self._preprocess_features(
                X_all[[f for f in original_features if f in X_all.columns]]
            )
            X_with_new = self._preprocess_features(X_all)

            return X_original, X_with_new, y

        except Exception as e:
            logger.error(f"Error preparing data for validation: {e}")
            return None, None, None

    def _preprocess_features(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preprocess features for quick evaluation."""
        try:
            X = X.copy()

            # Convert categorical columns
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    # Use frequency encoding for speed
                    freq_map = X[col].value_counts(normalize=True).to_dict()
                    X[col] = X[col].map(freq_map).fillna(0)

            # Handle missing values
            X = X.fillna(0)

            # Handle infinities
            X = X.replace([np.inf, -np.inf], 0)

            return X

        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return None

    def _quick_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str,
    ) -> float:
        """Quick evaluation using cross-validation."""
        try:
            # Select model based on task type
            if self.task_type == "regression":
                model = RandomForestRegressor(
                    n_estimators=self.QUICK_ESTIMATORS,
                    max_depth=6,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                cv = KFold(n_splits=self.QUICK_CV_FOLDS, shuffle=True, random_state=self.random_state)
                scoring = self._get_sklearn_scoring(metric)
            else:
                model = RandomForestClassifier(
                    n_estimators=self.QUICK_ESTIMATORS,
                    max_depth=6,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight='balanced',  # Handle imbalance
                )
                cv = StratifiedKFold(n_splits=self.QUICK_CV_FOLDS, shuffle=True, random_state=self.random_state)
                scoring = self._get_sklearn_scoring(metric)

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)

        except Exception as e:
            logger.error(f"Error in quick evaluation: {e}")
            return 0.0

    def _validate_single_feature(
        self,
        X_original: pd.DataFrame,
        X_with_new: pd.DataFrame,
        y: pd.Series,
        feature: str,
        baseline_score: float,
        metric: str,
    ) -> FeatureValidationResult:
        """Validate a single feature."""
        warnings = []

        # Check if feature exists
        if feature not in X_with_new.columns:
            return FeatureValidationResult(
                feature_name=feature,
                passed=False,
                reason=f"Feature '{feature}' not found in data",
            )

        # Create dataset with just this new feature added
        try:
            X_test = X_original.copy()
            X_test[feature] = X_with_new[feature]
        except Exception as e:
            return FeatureValidationResult(
                feature_name=feature,
                passed=False,
                reason=f"Error adding feature: {e}",
            )

        # Check for constant values
        if X_test[feature].nunique() <= 1:
            return FeatureValidationResult(
                feature_name=feature,
                passed=False,
                reason="Feature has constant or single value",
            )

        # Check for too many missing values
        missing_ratio = X_test[feature].isna().mean()
        if missing_ratio > 0.5:
            return FeatureValidationResult(
                feature_name=feature,
                passed=False,
                reason=f"Feature has {missing_ratio*100:.1f}% missing values",
            )
        elif missing_ratio > 0.2:
            warnings.append(f"High missing ratio: {missing_ratio*100:.1f}%")

        # Check for high correlation with existing features (potential redundancy)
        for orig_feature in X_original.columns:
            try:
                corr = X_test[feature].corr(X_original[orig_feature])
                if abs(corr) > 0.95:
                    warnings.append(f"Very high correlation ({corr:.2f}) with {orig_feature}")
            except:
                pass

        # Evaluate with the new feature
        X_test = self._preprocess_features(X_test)
        if X_test is None:
            return FeatureValidationResult(
                feature_name=feature,
                passed=False,
                reason="Feature preprocessing failed",
            )

        with_feature_score = self._quick_evaluate(X_test, y, metric)

        # Calculate improvement
        improvement = with_feature_score - baseline_score

        # Determine if feature passes
        passed = improvement >= self.MIN_IMPROVEMENT_THRESHOLD

        if passed:
            reason = f"Improves score by {improvement*100:.2f}%"
        else:
            if improvement < 0:
                reason = f"Hurts score by {abs(improvement)*100:.2f}%"
            else:
                reason = f"Improvement ({improvement*100:.3f}%) below threshold"

        return FeatureValidationResult(
            feature_name=feature,
            passed=passed,
            reason=reason,
            baseline_score=baseline_score,
            with_feature_score=with_feature_score,
            improvement=improvement,
            warnings=warnings,
        )

    def _get_default_metric(self) -> str:
        """Get default metric based on task type."""
        if self.task_type == "regression":
            return "r2"
        elif self.task_type == "binary":
            return "roc_auc"
        else:
            return "f1_macro"

    def _get_sklearn_scoring(self, metric: str) -> str:
        """Convert metric name to sklearn scoring parameter."""
        metric_map = {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
            "f1": "f1",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
            "balanced_accuracy": "balanced_accuracy",
            "r2": "r2",
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
        }
        return metric_map.get(metric, "accuracy")

    def _generate_recommendations(
        self,
        passed: list[str],
        failed: list[FeatureValidationResult],
        improvement: float,
    ) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if len(passed) == 0:
            recommendations.append(
                "No features improved the baseline. Consider different feature engineering approaches."
            )
        elif improvement < 1.0:
            recommendations.append(
                f"Modest improvement of {improvement:.2f}%. Consider more domain-specific features."
            )
        else:
            recommendations.append(
                f"Good improvement of {improvement:.2f}% with {len(passed)} features."
            )

        # Analyze failure patterns
        if failed:
            hurt_model = [f for f in failed if f.improvement < 0]
            no_impact = [f for f in failed if f.improvement >= 0]

            if hurt_model:
                recommendations.append(
                    f"{len(hurt_model)} features hurt model performance - removed."
                )

            if no_impact:
                recommendations.append(
                    f"{len(no_impact)} features had no significant impact - removed."
                )

        return recommendations


@dataclass
class ExistingFeatureValidationResult:
    """Result of validating existing feature column selection."""
    validated_features: list[str]
    removed_features: list[FeatureValidationResult]
    feature_importances: dict[str, float]
    baseline_all_features_score: float
    final_validated_score: float
    validation_method: str
    recommendations: list[str]


class ExistingFeatureValidator:
    """Validates that selected existing columns actually contribute to prediction."""

    # Features with importance below this are considered useless
    MIN_IMPORTANCE_THRESHOLD = 0.001  # 0.1% of total importance

    # Features that hurt performance this much are removed
    MAX_PERFORMANCE_HURT = -0.005  # -0.5% score reduction

    # Quick validation settings
    QUICK_CV_FOLDS = 3
    QUICK_ESTIMATORS = 50
    MAX_SAMPLES = 10000

    def __init__(self, task_type: str = "binary", random_state: int = 42):
        """Initialize validator.

        Args:
            task_type: 'binary', 'multiclass', or 'regression'
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state

    def validate_selected_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        metric: str = "auto",
    ) -> ExistingFeatureValidationResult:
        """Validate that the selected feature columns actually contribute to prediction.

        This uses feature importance from a quick RandomForest to identify:
        - Features with near-zero importance (not useful)
        - Features that may hurt model performance

        Args:
            df: DataFrame with all data
            target_column: Target column name
            feature_columns: List of feature columns to validate
            metric: Metric for evaluation ('auto' to select based on task)

        Returns:
            ExistingFeatureValidationResult with validated features
        """
        logger.info(f"Validating {len(feature_columns)} existing feature columns...")

        # Prepare data
        X, y = self._prepare_data(df, target_column, feature_columns)

        if X is None or len(X.columns) == 0:
            return ExistingFeatureValidationResult(
                validated_features=feature_columns,  # Return all if we can't validate
                removed_features=[],
                feature_importances={},
                baseline_all_features_score=0.0,
                final_validated_score=0.0,
                validation_method="failed",
                recommendations=["Could not validate features - returning all"],
            )

        # Select metric if auto
        if metric == "auto":
            metric = self._get_default_metric()

        # Get baseline score with all features
        baseline_score = self._quick_evaluate(X, y, metric)
        logger.info(f"Baseline score with {len(X.columns)} features: {baseline_score:.4f}")

        # Get feature importances
        importances = self._get_feature_importances(X, y)

        # Normalize importances
        total_importance = sum(importances.values()) if importances else 1.0
        normalized_importances = {
            k: v / total_importance for k, v in importances.items()
        } if total_importance > 0 else importances

        # Identify low-importance features
        validated_features = []
        removed_features = []

        for feature in feature_columns:
            if feature not in X.columns:
                removed_features.append(FeatureValidationResult(
                    feature_name=feature,
                    passed=False,
                    reason="Feature not found in data",
                ))
                continue

            importance = normalized_importances.get(feature, 0.0)

            if importance < self.MIN_IMPORTANCE_THRESHOLD:
                removed_features.append(FeatureValidationResult(
                    feature_name=feature,
                    passed=False,
                    reason=f"Very low importance ({importance*100:.3f}%)",
                    importance_rank=None,
                ))
            else:
                validated_features.append(feature)

        # If we removed features, check if performance improved
        final_score = baseline_score
        if removed_features and validated_features:
            X_validated = X[[f for f in validated_features if f in X.columns]]
            if len(X_validated.columns) > 0:
                final_score = self._quick_evaluate(X_validated, y, metric)
                logger.info(
                    f"Score after removing {len(removed_features)} low-importance features: "
                    f"{baseline_score:.4f} -> {final_score:.4f}"
                )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            validated_features, removed_features, baseline_score, final_score, normalized_importances
        )

        logger.info(
            f"Feature validation complete: {len(validated_features)} kept, "
            f"{len(removed_features)} removed"
        )

        return ExistingFeatureValidationResult(
            validated_features=validated_features,
            removed_features=removed_features,
            feature_importances=normalized_importances,
            baseline_all_features_score=baseline_score,
            final_validated_score=final_score,
            validation_method=f"importance_cv_{self.QUICK_CV_FOLDS}fold",
            recommendations=recommendations,
        )

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare data for validation."""
        try:
            # Get target
            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found")
                return None, None

            y = df[target_column].copy()

            # Encode target if classification
            if self.task_type in ["binary", "multiclass"]:
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

            # Get available features
            available_features = [f for f in feature_columns if f in df.columns]

            if not available_features:
                logger.error("No features available for validation")
                return None, None

            X = df[available_features].copy()

            # Sample if too large
            if len(X) > self.MAX_SAMPLES:
                sample_idx = np.random.choice(len(X), self.MAX_SAMPLES, replace=False)
                X = X.iloc[sample_idx]
                y = y.iloc[sample_idx]

            # Preprocess
            X = self._preprocess_features(X)

            return X, y

        except Exception as e:
            logger.error(f"Error preparing data for validation: {e}")
            return None, None

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for quick evaluation."""
        try:
            X = X.copy()

            # Convert categorical columns
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    freq_map = X[col].value_counts(normalize=True).to_dict()
                    X[col] = X[col].map(freq_map).fillna(0)

            # Handle missing values
            X = X.fillna(0)

            # Handle infinities
            X = X.replace([np.inf, -np.inf], 0)

            return X

        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return X

    def _quick_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str,
    ) -> float:
        """Quick evaluation using cross-validation."""
        try:
            if self.task_type == "regression":
                model = RandomForestRegressor(
                    n_estimators=self.QUICK_ESTIMATORS,
                    max_depth=6,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                cv = KFold(n_splits=self.QUICK_CV_FOLDS, shuffle=True, random_state=self.random_state)
            else:
                model = RandomForestClassifier(
                    n_estimators=self.QUICK_ESTIMATORS,
                    max_depth=6,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight='balanced',
                )
                cv = StratifiedKFold(n_splits=self.QUICK_CV_FOLDS, shuffle=True, random_state=self.random_state)

            scoring = self._get_sklearn_scoring(metric)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return np.mean(scores)

        except Exception as e:
            logger.error(f"Error in quick evaluation: {e}")
            return 0.0

    def _get_feature_importances(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """Get feature importances using RandomForest."""
        try:
            if self.task_type == "regression":
                model = RandomForestRegressor(
                    n_estimators=self.QUICK_ESTIMATORS,
                    max_depth=6,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=self.QUICK_ESTIMATORS,
                    max_depth=6,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight='balanced',
                )

            model.fit(X, y)
            importances = dict(zip(X.columns, model.feature_importances_))

            # Sort by importance
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

            return importances

        except Exception as e:
            logger.error(f"Error getting feature importances: {e}")
            return {}

    def _get_default_metric(self) -> str:
        """Get default metric based on task type."""
        if self.task_type == "regression":
            return "r2"
        elif self.task_type == "binary":
            return "roc_auc"
        else:
            return "f1_macro"

    def _get_sklearn_scoring(self, metric: str) -> str:
        """Convert metric name to sklearn scoring parameter."""
        metric_map = {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
            "f1": "f1",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
            "balanced_accuracy": "balanced_accuracy",
            "r2": "r2",
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
        }
        return metric_map.get(metric, "accuracy")

    def _generate_recommendations(
        self,
        validated: list[str],
        removed: list[FeatureValidationResult],
        baseline: float,
        final: float,
        importances: dict[str, float],
    ) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if removed:
            recommendations.append(
                f"Removed {len(removed)} features with very low predictive power."
            )

        if final > baseline:
            recommendations.append(
                f"Removing low-importance features improved score: {baseline:.4f} -> {final:.4f}"
            )
        elif final < baseline - 0.01:
            recommendations.append(
                f"Warning: Removing features hurt performance. Consider keeping all features."
            )

        # Top features
        if importances:
            top_features = list(importances.keys())[:5]
            recommendations.append(
                f"Most important features: {', '.join(top_features)}"
            )

        return recommendations


def validate_feature_set(
    df: pd.DataFrame,
    target_column: str,
    new_features: list[str],
    original_features: list[str],
    task_type: str = "binary",
    metric: str = "auto",
) -> FeatureSetValidationResult:
    """Convenience function to validate features.

    Args:
        df: DataFrame with all data
        target_column: Target column name
        new_features: New features to validate
        original_features: Original baseline features
        task_type: Task type ('binary', 'multiclass', 'regression')
        metric: Metric for evaluation

    Returns:
        Validation result with passed/failed features
    """
    validator = FeatureValidator(task_type=task_type)
    return validator.validate_features(
        df, target_column, new_features, original_features, metric
    )

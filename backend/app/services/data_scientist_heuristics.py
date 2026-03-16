"""Data Scientist Heuristics - Smart recommendations based on ML patterns.

This module encapsulates the knowledge and decision-making patterns that
experienced data scientists use when iterating on ML models. It provides
actionable recommendations based on problem characteristics, historical
performance, and common ML patterns.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProblemPattern(Enum):
    """Common problem patterns that affect ML approach."""
    CLASS_IMBALANCE_MILD = "class_imbalance_mild"
    CLASS_IMBALANCE_SEVERE = "class_imbalance_severe"
    HIGH_CARDINALITY = "high_cardinality"
    MANY_FEATURES = "many_features"
    FEW_SAMPLES = "few_samples"
    TIME_SERIES = "time_series"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    SCORE_PLATEAU = "score_plateau"
    FEATURE_NOISE = "feature_noise"
    MISSING_DATA = "missing_data"
    SKEWED_DISTRIBUTION = "skewed_distribution"


class ImprovementStrategy(Enum):
    """Improvement strategies a data scientist might use."""
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    HANDLE_IMBALANCE = "handle_imbalance"
    REGULARIZATION = "regularization"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ENSEMBLE_METHODS = "ensemble_methods"
    DATA_AUGMENTATION = "data_augmentation"
    TARGET_ENCODING = "target_encoding"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    OUTLIER_HANDLING = "outlier_handling"


@dataclass
class Recommendation:
    """A specific recommendation for improving model performance."""
    strategy: ImprovementStrategy
    priority: int  # 1-10, higher = more important
    description: str
    rationale: str
    specific_actions: List[str]
    expected_impact: str
    risk_level: str  # "low", "medium", "high"
    applicable_patterns: List[ProblemPattern] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """Result of diagnosing model performance issues."""
    detected_patterns: List[ProblemPattern]
    primary_bottleneck: str
    recommendations: List[Recommendation]
    warnings: List[str]
    insights: List[str]
    confidence: float  # 0-1


class DataScientistHeuristics:
    """Encapsulates data scientist knowledge for ML improvement.

    This class acts like an experienced data scientist, analyzing
    experiment history and data characteristics to provide smart
    recommendations for improving model performance.
    """

    # Thresholds for pattern detection
    IMBALANCE_MILD_THRESHOLD = 0.3  # Minority class < 30%
    IMBALANCE_SEVERE_THRESHOLD = 0.1  # Minority class < 10%
    HIGH_CARDINALITY_THRESHOLD = 50  # More than 50 unique values
    MANY_FEATURES_THRESHOLD = 100
    FEW_SAMPLES_THRESHOLD = 1000
    MISSING_DATA_THRESHOLD = 0.1  # >10% missing
    PLATEAU_ITERATIONS = 3  # Iterations without improvement

    def __init__(self):
        self.heuristics = self._build_heuristics()

    def diagnose(
        self,
        iteration_context: Dict[str, Any],
        data_stats: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisResult:
        """Diagnose performance issues and recommend improvements.

        This is the main entry point - it analyzes the experiment history
        and current state to provide recommendations.

        Args:
            iteration_context: Context from IterationContextAgent
            data_stats: Optional data statistics

        Returns:
            DiagnosisResult with patterns, recommendations, and insights
        """
        patterns = []
        warnings = []
        insights = []

        # Detect patterns from data
        if data_stats:
            patterns.extend(self._detect_data_patterns(data_stats))

        # Detect patterns from iteration history
        patterns.extend(self._detect_iteration_patterns(iteration_context))

        # Remove duplicates
        patterns = list(set(patterns))

        # Identify primary bottleneck
        bottleneck = self._identify_bottleneck(patterns, iteration_context)

        # Get recommendations based on patterns
        recommendations = self._get_recommendations(patterns, iteration_context)

        # Generate insights from analysis
        insights = self._generate_insights(patterns, iteration_context)

        # Calculate confidence based on how much information we have
        confidence = self._calculate_confidence(iteration_context, data_stats)

        return DiagnosisResult(
            detected_patterns=patterns,
            primary_bottleneck=bottleneck,
            recommendations=recommendations,
            warnings=warnings,
            insights=insights,
            confidence=confidence,
        )

    def _detect_data_patterns(self, data_stats: Dict[str, Any]) -> List[ProblemPattern]:
        """Detect patterns from data statistics."""
        patterns = []
        column_stats = data_stats.get("column_stats", {})

        # Check for high cardinality
        for col, stats in column_stats.items():
            if stats.get("unique", 0) > self.HIGH_CARDINALITY_THRESHOLD:
                if stats.get("dtype", "").startswith(("object", "category")):
                    patterns.append(ProblemPattern.HIGH_CARDINALITY)
                    break

        # Check for missing data
        for col, stats in column_stats.items():
            null_pct = stats.get("null_pct", 0)
            if null_pct > self.MISSING_DATA_THRESHOLD * 100:
                patterns.append(ProblemPattern.MISSING_DATA)
                break

        # Check for skewed distributions
        for col, stats in column_stats.items():
            if "mean" in stats and "std" in stats:
                mean = stats.get("mean", 0)
                std = stats.get("std", 1)
                if mean and std:
                    max_val = stats.get("max", 0)
                    min_val = stats.get("min", 0)
                    if max_val and min_val:
                        # Check for large range relative to std
                        range_val = max_val - min_val
                        if range_val > 10 * std:
                            patterns.append(ProblemPattern.SKEWED_DISTRIBUTION)
                            break

        # Check for many features
        if data_stats.get("column_count", 0) > self.MANY_FEATURES_THRESHOLD:
            patterns.append(ProblemPattern.MANY_FEATURES)

        # Check for few samples
        if data_stats.get("row_count", 0) < self.FEW_SAMPLES_THRESHOLD:
            patterns.append(ProblemPattern.FEW_SAMPLES)

        return patterns

    def _detect_iteration_patterns(self, ctx: Dict[str, Any]) -> List[ProblemPattern]:
        """Detect patterns from iteration history."""
        patterns = []

        # Check for overfitting
        overfitting_report = ctx.get("overfitting_report", {})
        if overfitting_report:
            if overfitting_report.get("trend") == "worsening":
                patterns.append(ProblemPattern.OVERFITTING)
            elif overfitting_report.get("recommendation") == "simplify":
                patterns.append(ProblemPattern.OVERFITTING)

        # Check for score plateau
        history = ctx.get("iteration_history", [])
        if len(history) >= self.PLATEAU_ITERATIONS:
            recent_scores = [h.get("score", 0) for h in history[-self.PLATEAU_ITERATIONS:]]
            if all(s > 0 for s in recent_scores):
                max_score = max(recent_scores)
                min_score = min(recent_scores)
                # Less than 1% improvement over last N iterations
                if (max_score - min_score) / max_score < 0.01:
                    patterns.append(ProblemPattern.SCORE_PLATEAU)

        # Check for underfitting (consistently low scores)
        if history:
            best_score = ctx.get("best_score", 0)
            if best_score < 0.7 and len(history) >= 3:
                patterns.append(ProblemPattern.UNDERFITTING)

        # Check score trend
        score_trend = ctx.get("score_trend", "flat")
        if score_trend == "declining":
            patterns.append(ProblemPattern.OVERFITTING)

        return patterns

    def _identify_bottleneck(
        self,
        patterns: List[ProblemPattern],
        ctx: Dict[str, Any],
    ) -> str:
        """Identify the primary bottleneck limiting performance."""

        # Priority order for bottlenecks
        if ProblemPattern.OVERFITTING in patterns:
            return "Model is overfitting - validation performance is degrading while training performance remains high"

        if ProblemPattern.CLASS_IMBALANCE_SEVERE in patterns:
            return "Severe class imbalance is causing the model to favor the majority class"

        if ProblemPattern.SCORE_PLATEAU in patterns:
            return "Performance has plateaued - current feature set may have reached its limit"

        if ProblemPattern.UNDERFITTING in patterns:
            return "Model is underfitting - current features may not capture the signal well"

        if ProblemPattern.HIGH_CARDINALITY in patterns:
            return "High cardinality categorical features may be causing sparse representations"

        if ProblemPattern.MANY_FEATURES in patterns:
            return "Too many features may be introducing noise and hindering learning"

        if ProblemPattern.MISSING_DATA in patterns:
            return "High missing data rate may be limiting model performance"

        # Default bottleneck based on score
        best_score = ctx.get("best_score", 0)
        if best_score < 0.6:
            return "Low overall performance - may need better features or different approach"
        elif best_score < 0.8:
            return "Moderate performance - feature engineering or hyperparameter tuning may help"
        else:
            return "Good performance - marginal improvements possible with advanced techniques"

    def _get_recommendations(
        self,
        patterns: List[ProblemPattern],
        ctx: Dict[str, Any],
    ) -> List[Recommendation]:
        """Get ranked recommendations based on detected patterns."""
        recommendations = []

        for pattern in patterns:
            recs = self.heuristics.get(pattern, [])
            for rec in recs:
                # Adjust priority based on context
                adjusted_rec = self._adjust_recommendation(rec, ctx)
                if adjusted_rec:
                    recommendations.append(adjusted_rec)

        # Add general recommendations if score is low
        best_score = ctx.get("best_score", 0)
        if best_score < 0.7:
            recommendations.append(Recommendation(
                strategy=ImprovementStrategy.FEATURE_ENGINEERING,
                priority=8,
                description="Create domain-specific features",
                rationale="Low scores often indicate missing signal in features",
                specific_actions=[
                    "Analyze top feature importances for patterns",
                    "Create interaction features between top predictors",
                    "Add polynomial features for numeric columns",
                ],
                expected_impact="5-15% improvement possible",
                risk_level="low",
            ))

        # Add plateau-breaking recommendations
        if ProblemPattern.SCORE_PLATEAU in patterns:
            recommendations.append(Recommendation(
                strategy=ImprovementStrategy.ENSEMBLE_METHODS,
                priority=7,
                description="Try diverse model ensemble",
                rationale="Plateau may be broken by combining different model types",
                specific_actions=[
                    "Include models with different inductive biases",
                    "Try stacking with diverse base learners",
                    "Consider neural network ensemble",
                ],
                expected_impact="2-5% improvement possible",
                risk_level="medium",
            ))

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)

        # Return top recommendations
        return recommendations[:5]

    def _adjust_recommendation(
        self,
        rec: Recommendation,
        ctx: Dict[str, Any],
    ) -> Optional[Recommendation]:
        """Adjust recommendation based on context."""
        # Check if this strategy was already tried
        improvements = ctx.get("improvement_attempts", [])
        strategy_name = rec.strategy.value

        # Lower priority if strategy was tried and failed
        for imp in improvements:
            if strategy_name in imp.get("changes", "").lower():
                if not imp.get("success", True):
                    # Reduce priority significantly if failed before
                    rec.priority = max(1, rec.priority - 5)

        return rec

    def _generate_insights(
        self,
        patterns: List[ProblemPattern],
        ctx: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable insights from analysis."""
        insights = []

        # Score-based insights
        best_score = ctx.get("best_score", 0)
        current_score = ctx.get("current_score", 0)

        if current_score < best_score * 0.98:
            insights.append(
                f"Current score ({current_score:.4f}) is below best ({best_score:.4f}) - "
                "consider reverting recent changes"
            )

        # Iteration-based insights
        total_iterations = ctx.get("total_iterations", 0)
        if total_iterations > 10 and best_score < 0.8:
            insights.append(
                "Many iterations without reaching 0.8 score - "
                "consider trying a fundamentally different approach"
            )

        # Pattern-based insights
        if ProblemPattern.OVERFITTING in patterns:
            insights.append(
                "Overfitting detected - try regularization, fewer features, or simpler models"
            )

        if ProblemPattern.SCORE_PLATEAU in patterns:
            insights.append(
                "Score plateau detected - feature set may have reached its limit, "
                "try new feature types or external data"
            )

        # Feature engineering insights
        fe_feedback = ctx.get("feature_engineering_feedback", {})
        successful = fe_feedback.get("successful_features", [])
        failed = fe_feedback.get("failed_features", [])

        if len(failed) > len(successful) * 2:
            insights.append(
                f"High feature failure rate ({len(failed)} failed vs {len(successful)} successful) - "
                "focus on simpler, more reliable transformations"
            )

        if successful:
            # Analyze successful feature patterns
            formula_patterns = [s.get("formula", "") for s in successful]
            if any("rolling" in f.lower() for f in formula_patterns):
                insights.append("Rolling window features have worked - try more temporal features")
            if any("ratio" in f.lower() or "/" in f for f in formula_patterns):
                insights.append("Ratio features have worked - try more ratio combinations")

        return insights

    def _calculate_confidence(
        self,
        ctx: Dict[str, Any],
        data_stats: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate confidence in the diagnosis."""
        confidence = 0.5  # Base confidence

        # More data = higher confidence
        if data_stats:
            confidence += 0.1
            if data_stats.get("row_count", 0) > 1000:
                confidence += 0.1

        # More iterations = higher confidence
        total_iterations = ctx.get("total_iterations", 0)
        if total_iterations >= 3:
            confidence += 0.1
        if total_iterations >= 5:
            confidence += 0.1

        # Cap at 0.95
        return min(0.95, confidence)

    def _build_heuristics(self) -> Dict[ProblemPattern, List[Recommendation]]:
        """Build the heuristics knowledge base."""
        return {
            ProblemPattern.CLASS_IMBALANCE_MILD: [
                Recommendation(
                    strategy=ImprovementStrategy.HANDLE_IMBALANCE,
                    priority=7,
                    description="Apply class weights to handle mild imbalance",
                    rationale="Class weights help the model pay more attention to minority class",
                    specific_actions=[
                        "Use class_weight='balanced' in model training",
                        "Switch to balanced accuracy or F1 as primary metric",
                        "Consider using balanced sampling",
                    ],
                    expected_impact="3-8% improvement on minority class metrics",
                    risk_level="low",
                ),
            ],
            ProblemPattern.CLASS_IMBALANCE_SEVERE: [
                Recommendation(
                    strategy=ImprovementStrategy.HANDLE_IMBALANCE,
                    priority=9,
                    description="Apply aggressive resampling for severe imbalance",
                    rationale="Severe imbalance requires more aggressive techniques",
                    specific_actions=[
                        "Apply SMOTE or ADASYN oversampling",
                        "Use focal loss if available",
                        "Switch to precision-recall AUC as metric",
                        "Consider cost-sensitive learning",
                    ],
                    expected_impact="10-20% improvement on minority class",
                    risk_level="medium",
                ),
            ],
            ProblemPattern.HIGH_CARDINALITY: [
                Recommendation(
                    strategy=ImprovementStrategy.TARGET_ENCODING,
                    priority=8,
                    description="Use target encoding for high cardinality categoricals",
                    rationale="One-hot encoding creates too many sparse features",
                    specific_actions=[
                        "Apply target encoding with smoothing",
                        "Consider frequency encoding as fallback",
                        "Use leave-one-out encoding to prevent leakage",
                    ],
                    expected_impact="5-10% improvement possible",
                    risk_level="medium",
                ),
            ],
            ProblemPattern.MANY_FEATURES: [
                Recommendation(
                    strategy=ImprovementStrategy.FEATURE_SELECTION,
                    priority=7,
                    description="Apply feature selection to reduce dimensionality",
                    rationale="Too many features can introduce noise and slow training",
                    specific_actions=[
                        "Remove features with near-zero importance",
                        "Apply recursive feature elimination",
                        "Use correlation-based filtering",
                    ],
                    expected_impact="Faster training, 2-5% improvement possible",
                    risk_level="low",
                ),
                Recommendation(
                    strategy=ImprovementStrategy.DIMENSIONALITY_REDUCTION,
                    priority=5,
                    description="Consider PCA for feature compression",
                    rationale="PCA can capture most variance with fewer features",
                    specific_actions=[
                        "Apply PCA keeping 95% variance",
                        "Combine PCA features with top original features",
                    ],
                    expected_impact="Faster training, may slightly reduce accuracy",
                    risk_level="medium",
                ),
            ],
            ProblemPattern.OVERFITTING: [
                Recommendation(
                    strategy=ImprovementStrategy.REGULARIZATION,
                    priority=9,
                    description="Apply stronger regularization",
                    rationale="Overfitting indicates model is too complex for data",
                    specific_actions=[
                        "Reduce max_depth in tree models",
                        "Increase min_samples_leaf",
                        "Add L1/L2 regularization",
                        "Increase dropout if using neural networks",
                    ],
                    expected_impact="Better generalization, validation score should improve",
                    risk_level="low",
                ),
                Recommendation(
                    strategy=ImprovementStrategy.FEATURE_SELECTION,
                    priority=8,
                    description="Reduce feature count to prevent overfitting",
                    rationale="Fewer features = less opportunity to overfit",
                    specific_actions=[
                        "Keep only top 50% features by importance",
                        "Remove highly correlated features",
                        "Remove features with high variance across folds",
                    ],
                    expected_impact="Better generalization",
                    risk_level="low",
                ),
            ],
            ProblemPattern.UNDERFITTING: [
                Recommendation(
                    strategy=ImprovementStrategy.FEATURE_ENGINEERING,
                    priority=9,
                    description="Create more expressive features",
                    rationale="Underfitting suggests current features don't capture signal",
                    specific_actions=[
                        "Create interaction features between top predictors",
                        "Add polynomial features",
                        "Engineer domain-specific features",
                        "Try feature crosses",
                    ],
                    expected_impact="5-15% improvement possible",
                    risk_level="low",
                ),
                Recommendation(
                    strategy=ImprovementStrategy.ENSEMBLE_METHODS,
                    priority=7,
                    description="Use more complex models",
                    rationale="Simple models may not have capacity to learn patterns",
                    specific_actions=[
                        "Try gradient boosting (XGBoost, LightGBM)",
                        "Increase model complexity (more trees, deeper)",
                        "Consider neural network for complex patterns",
                    ],
                    expected_impact="10-20% improvement possible",
                    risk_level="medium",
                ),
            ],
            ProblemPattern.SCORE_PLATEAU: [
                Recommendation(
                    strategy=ImprovementStrategy.FEATURE_ENGINEERING,
                    priority=8,
                    description="Try completely new feature types",
                    rationale="Current feature set has reached its limit",
                    specific_actions=[
                        "Add time-based features if applicable",
                        "Try text embeddings if text data available",
                        "Create aggregate features at different levels",
                        "Add external data sources if possible",
                    ],
                    expected_impact="Break plateau with 2-10% improvement",
                    risk_level="medium",
                ),
                Recommendation(
                    strategy=ImprovementStrategy.HYPERPARAMETER_TUNING,
                    priority=6,
                    description="Intensive hyperparameter search",
                    rationale="Plateau may be broken with optimal hyperparameters",
                    specific_actions=[
                        "Run extensive hyperparameter optimization",
                        "Try different optimization algorithms",
                        "Consider Bayesian optimization",
                    ],
                    expected_impact="1-3% improvement possible",
                    risk_level="low",
                ),
            ],
            ProblemPattern.MISSING_DATA: [
                Recommendation(
                    strategy=ImprovementStrategy.FEATURE_ENGINEERING,
                    priority=6,
                    description="Handle missing data strategically",
                    rationale="Missing patterns may contain predictive signal",
                    specific_actions=[
                        "Create 'is_missing' indicator features",
                        "Try multiple imputation strategies",
                        "Consider dropping high-missing columns",
                    ],
                    expected_impact="2-5% improvement possible",
                    risk_level="low",
                ),
            ],
            ProblemPattern.SKEWED_DISTRIBUTION: [
                Recommendation(
                    strategy=ImprovementStrategy.FEATURE_ENGINEERING,
                    priority=5,
                    description="Apply transformations to skewed features",
                    rationale="Skewed features can hurt model performance",
                    specific_actions=[
                        "Apply log transform (np.log1p) to skewed numerics",
                        "Try Box-Cox or Yeo-Johnson transforms",
                        "Consider binning extremely skewed features",
                    ],
                    expected_impact="1-5% improvement possible",
                    risk_level="low",
                ),
            ],
        }

    def get_feature_suggestions(
        self,
        data_stats: Dict[str, Any],
        existing_features: List[str],
        failed_features: List[str],
        task_type: str = "binary",
    ) -> List[Dict[str, Any]]:
        """Suggest specific features based on data characteristics.

        This method acts like a data scientist brainstorming features.

        Args:
            data_stats: Statistics about the dataset
            existing_features: Features that already exist
            failed_features: Features that were tried and failed
            task_type: Type of ML task

        Returns:
            List of feature suggestions with formulas
        """
        suggestions = []
        columns = data_stats.get("columns", [])
        column_stats = data_stats.get("column_stats", {})

        # Identify numeric columns
        numeric_cols = [
            col for col, stats in column_stats.items()
            if stats.get("dtype", "").startswith(("int", "float"))
        ]

        # Identify date columns
        date_cols = [
            col for col in columns
            if any(kw in col.lower() for kw in ["date", "time", "day", "month", "year"])
        ]

        # Generate interaction features
        if len(numeric_cols) >= 2:
            top_numeric = numeric_cols[:5]
            for i, col1 in enumerate(top_numeric):
                for col2 in top_numeric[i+1:i+3]:
                    feature_name = f"{col1}_x_{col2}_ratio"
                    if feature_name not in existing_features and feature_name not in failed_features:
                        suggestions.append({
                            "output_column": feature_name,
                            "formula": f'df["{col1}"] / (df["{col2}"] + 1e-8)',
                            "description": f"Ratio of {col1} to {col2}",
                            "type": "ratio",
                        })

        # Generate log transforms for skewed features
        for col, stats in column_stats.items():
            if stats.get("dtype", "").startswith(("int", "float")):
                min_val = stats.get("min", 0)
                max_val = stats.get("max", 0)
                if min_val is not None and max_val is not None and min_val >= 0:
                    if max_val > 0 and (max_val / (min_val + 1)) > 100:
                        feature_name = f"{col}_log"
                        if feature_name not in existing_features and feature_name not in failed_features:
                            suggestions.append({
                                "output_column": feature_name,
                                "formula": f'np.log1p(df["{col}"])',
                                "description": f"Log transform of {col}",
                                "type": "transform",
                            })

        # Generate date features
        for col in date_cols:
            for suffix, formula_part in [
                ("_dayofweek", ".dt.dayofweek"),
                ("_month", ".dt.month"),
                ("_is_weekend", ".dt.dayofweek.isin([5,6]).astype(int)"),
            ]:
                feature_name = f"{col}{suffix}"
                if feature_name not in existing_features and feature_name not in failed_features:
                    suggestions.append({
                        "output_column": feature_name,
                        "formula": f'pd.to_datetime(df["{col}"], errors="coerce"){formula_part}',
                        "description": f"Extract {suffix[1:]} from {col}",
                        "type": "datetime",
                    })

        # Limit suggestions
        return suggestions[:10]


def get_smart_recommendations(
    iteration_context: Dict[str, Any],
    data_stats: Optional[Dict[str, Any]] = None,
) -> DiagnosisResult:
    """Get smart recommendations for improving an experiment.

    This is the main entry point for getting AI-powered recommendations.

    Args:
        iteration_context: Context from IterationContextAgent
        data_stats: Optional data statistics

    Returns:
        DiagnosisResult with recommendations
    """
    heuristics = DataScientistHeuristics()
    return heuristics.diagnose(iteration_context, data_stats)

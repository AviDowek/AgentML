"""Ablation study service for measuring feature importance.

This module provides functionality to systematically remove features or components
and measure their impact on model performance, helping identify which elements
contribute most to prediction accuracy.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result from a single ablation experiment."""
    ablated_feature: str
    original_score: float
    ablated_score: float
    score_delta: float
    relative_importance: float  # Percentage contribution
    experiment_id: Optional[str] = None


@dataclass
class AblationStudyResult:
    """Complete ablation study results."""

    # Original model performance (baseline)
    baseline_score: float
    baseline_experiment_id: str

    # All ablation results sorted by importance
    ablation_results: List[AblationResult] = field(default_factory=list)

    # Summary statistics
    total_features_tested: int = 0
    significant_features: int = 0

    # Feature rankings
    feature_importance_ranking: List[tuple[str, float]] = field(default_factory=list)

    # Top contributors
    top_positive_contributors: List[str] = field(default_factory=list)
    top_negative_contributors: List[str] = field(default_factory=list)  # Features that hurt

    # Recommendations
    recommended_features_to_keep: List[str] = field(default_factory=list)
    recommended_features_to_remove: List[str] = field(default_factory=list)


class AblationStudyRunner:
    """Runs ablation studies to identify feature importance."""

    def __init__(
        self,
        task_type: str,
        primary_metric: str,
        higher_is_better: bool = True,
    ):
        """Initialize ablation study runner.

        Args:
            task_type: ML task type (binary, multiclass, regression)
            primary_metric: Metric to track for importance
            higher_is_better: Whether higher metric values are better
        """
        self.task_type = task_type
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better

        # Detect metric direction
        error_metrics = {
            "rmse", "mse", "mae", "root_mean_squared_error",
            "mean_squared_error", "mean_absolute_error", "log_loss",
            "pinball_loss"
        }
        if any(e in primary_metric.lower() for e in error_metrics):
            self.higher_is_better = False

    def identify_feature_groups(
        self,
        data: pd.DataFrame,
        target_column: str,
    ) -> dict[str, List[str]]:
        """Identify logical feature groups for ablation.

        Args:
            data: DataFrame with features
            target_column: Name of target column

        Returns:
            Dict mapping group name to list of feature names
        """
        feature_cols = [c for c in data.columns if c != target_column]
        groups = {}

        # Group by prefix (e.g., "user_age", "user_income" -> "user_*")
        prefix_groups = {}
        for col in feature_cols:
            parts = col.split("_")
            if len(parts) > 1:
                prefix = parts[0]
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(col)

        # Only keep prefix groups with multiple features
        for prefix, cols in prefix_groups.items():
            if len(cols) > 1:
                groups[f"{prefix}_*"] = cols

        # Group by data type
        numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

        if numeric_cols:
            groups["all_numeric"] = numeric_cols
        if categorical_cols:
            groups["all_categorical"] = categorical_cols

        # Identify potential engineered features
        engineered = []
        for col in feature_cols:
            if any(indicator in col.lower() for indicator in
                   ["ratio", "log_", "sqrt_", "squared", "_x_", "_interaction", "_binned"]):
                engineered.append(col)

        if engineered:
            groups["engineered_features"] = engineered

        return groups

    def generate_ablation_experiments(
        self,
        data: pd.DataFrame,
        target_column: str,
        base_config: dict[str, Any],
        max_individual_features: int = 20,
        include_groups: bool = True,
    ) -> List[dict]:
        """Generate experiment configurations for ablation study.

        Args:
            data: Training DataFrame
            target_column: Name of target column
            base_config: Base AutoML configuration
            max_individual_features: Max number of individual features to ablate
            include_groups: Whether to include feature group ablations

        Returns:
            List of experiment configs, each with 'ablation_target' and 'drop_columns'
        """
        feature_cols = [c for c in data.columns if c != target_column]
        experiments = []

        # Prioritize features by variance (for numeric) or cardinality (for categorical)
        feature_scores = {}
        for col in feature_cols:
            if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Numeric: use coefficient of variation
                std = data[col].std()
                mean = abs(data[col].mean())
                feature_scores[col] = std / mean if mean > 0 else std
            else:
                # Categorical: use number of unique values
                feature_scores[col] = data[col].nunique() / len(data)

        # Sort features by score (higher score = potentially more important)
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:max_individual_features]]

        # Generate individual feature ablations
        for feature in top_features:
            config = base_config.copy()
            config["ablation_target"] = feature
            config["drop_columns"] = [feature]
            config["ablation_type"] = "single_feature"
            experiments.append(config)

        # Generate feature group ablations
        if include_groups:
            groups = self.identify_feature_groups(data, target_column)
            for group_name, group_cols in groups.items():
                config = base_config.copy()
                config["ablation_target"] = group_name
                config["drop_columns"] = group_cols
                config["ablation_type"] = "feature_group"
                experiments.append(config)

        logger.info(
            f"Generated {len(experiments)} ablation experiments "
            f"({len(top_features)} individual + {len(experiments) - len(top_features)} groups)"
        )

        return experiments

    def calculate_importance(
        self,
        baseline_score: float,
        ablation_results: List[tuple[str, float, str]],
    ) -> AblationStudyResult:
        """Calculate feature importance from ablation results.

        Args:
            baseline_score: Score with all features
            ablation_results: List of (feature_name, ablated_score, experiment_id)

        Returns:
            AblationStudyResult with importance rankings
        """
        results = []

        for feature, ablated_score, exp_id in ablation_results:
            if ablated_score is None:
                continue

            # Calculate score delta
            if self.higher_is_better:
                # Higher is better: positive delta means feature helps
                score_delta = baseline_score - ablated_score
            else:
                # Lower is better: negative delta means feature helps (removing it hurts)
                score_delta = ablated_score - baseline_score

            # Calculate relative importance (% contribution)
            if baseline_score != 0:
                relative_importance = abs(score_delta) / abs(baseline_score) * 100
            else:
                relative_importance = 0

            results.append(AblationResult(
                ablated_feature=feature,
                original_score=baseline_score,
                ablated_score=ablated_score,
                score_delta=score_delta,
                relative_importance=relative_importance,
                experiment_id=exp_id,
            ))

        # Sort by importance (absolute delta)
        results.sort(key=lambda x: abs(x.score_delta), reverse=True)

        # Create ranking
        feature_ranking = [(r.ablated_feature, r.score_delta) for r in results]

        # Identify top positive contributors (removing them hurts performance)
        positive_contributors = [
            r.ablated_feature for r in results
            if r.score_delta > 0  # Positive delta = feature helps
        ][:5]

        # Identify negative contributors (removing them helps performance)
        negative_contributors = [
            r.ablated_feature for r in results
            if r.score_delta < 0  # Negative delta = feature hurts
        ][:5]

        # Recommendations
        significance_threshold = 0.01  # 1% relative change
        significant = [r for r in results if r.relative_importance >= significance_threshold * 100]

        keep_features = [
            r.ablated_feature for r in results
            if r.score_delta > 0 and r.relative_importance >= 1.0
        ]

        remove_features = [
            r.ablated_feature for r in results
            if r.score_delta < -0.5  # Removing helps by more than 0.5%
        ]

        return AblationStudyResult(
            baseline_score=baseline_score,
            baseline_experiment_id="",
            ablation_results=results,
            total_features_tested=len(results),
            significant_features=len(significant),
            feature_importance_ranking=feature_ranking,
            top_positive_contributors=positive_contributors,
            top_negative_contributors=negative_contributors,
            recommended_features_to_keep=keep_features,
            recommended_features_to_remove=remove_features,
        )


def generate_ablation_prompt_section(
    dataset_info: dict[str, Any],
    previous_results: Optional[List[dict]] = None,
) -> str:
    """Generate prompt section for AI to design ablation experiments.

    Args:
        dataset_info: Information about the dataset
        previous_results: Results from previous experiments if available

    Returns:
        Prompt section for ablation study design
    """
    feature_list = dataset_info.get("feature_columns", [])

    prompt = """
## Ablation Study Design

Ablation studies help identify which features contribute most to model performance.
Your goal is to strategically select features to ablate (remove) to understand their importance.

### Guidelines for Ablation Target Selection:

1. **Start with Domain-Suspicious Features**
   - Features that might cause data leakage
   - Features that might not be available at prediction time
   - Highly correlated feature pairs

2. **Test Engineered Features**
   - Any features created through transformations
   - Interaction features
   - Aggregated features

3. **Group Related Features**
   - Test removing all features from a category together
   - Compare individual vs group removal

4. **Consider Feature Groups by Type**
   - Test all temporal features together
   - Test all categorical features together
   - Test all derived metrics together

### How to Specify Ablation:
In your experiment configuration, use:
- `ablation_target`: Name describing what you're removing
- `drop_columns`: List of column names to exclude from training

"""

    if feature_list:
        prompt += f"\n### Available Features ({len(feature_list)} total):\n"
        # Show first 30 features
        for feat in feature_list[:30]:
            prompt += f"- {feat}\n"
        if len(feature_list) > 30:
            prompt += f"- ... and {len(feature_list) - 30} more\n"

    if previous_results:
        prompt += "\n### Previous Ablation Results:\n"
        for result in previous_results[-5:]:  # Show last 5
            target = result.get("ablation_target", "unknown")
            score = result.get("score", "N/A")
            prompt += f"- Removed `{target}`: score = {score}\n"

    return prompt


def format_ablation_study_summary(result: AblationStudyResult) -> str:
    """Format ablation study results as a readable summary.

    Args:
        result: AblationStudyResult to format

    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 60,
        "ABLATION STUDY RESULTS",
        "=" * 60,
        f"Baseline Score: {result.baseline_score:.4f}",
        f"Features Tested: {result.total_features_tested}",
        f"Significant Features: {result.significant_features}",
        "",
        "Feature Importance Ranking (by impact when removed):",
        "-" * 40,
    ]

    # Show top 10 most important features
    for i, (feature, delta) in enumerate(result.feature_importance_ranking[:10], 1):
        direction = "+" if delta > 0 else ""
        lines.append(f"  {i}. {feature}: {direction}{delta:.4f}")

    lines.append("")

    if result.top_positive_contributors:
        lines.append("Top Contributing Features (removing hurts):")
        for feat in result.top_positive_contributors:
            lines.append(f"  + {feat}")
        lines.append("")

    if result.top_negative_contributors:
        lines.append("Potentially Harmful Features (removing helps):")
        for feat in result.top_negative_contributors:
            lines.append(f"  - {feat}")
        lines.append("")

    if result.recommended_features_to_remove:
        lines.append("RECOMMENDATION: Consider removing these features:")
        for feat in result.recommended_features_to_remove:
            lines.append(f"  * {feat}")

    lines.append("=" * 60)

    return "\n".join(lines)


def create_ablation_experiment_plan(
    base_experiment_config: dict[str, Any],
    features_to_ablate: List[str],
    ablation_type: str = "individual",
) -> List[dict]:
    """Create experiment plan entries for ablation study.

    Args:
        base_experiment_config: Base configuration to use
        features_to_ablate: Features to test removing
        ablation_type: "individual" for one at a time, "cumulative" for sequential

    Returns:
        List of experiment configurations
    """
    experiments = []

    if ablation_type == "individual":
        # Test each feature independently
        for feature in features_to_ablate:
            config = base_experiment_config.copy()
            config["ablation_target"] = feature
            config["drop_columns"] = [feature]
            config["description"] = f"Ablation: remove {feature}"
            experiments.append(config)

    elif ablation_type == "cumulative":
        # Sequentially remove features in order
        dropped = []
        for feature in features_to_ablate:
            dropped.append(feature)
            config = base_experiment_config.copy()
            config["ablation_target"] = f"cumulative_{len(dropped)}"
            config["drop_columns"] = dropped.copy()
            config["description"] = f"Ablation: remove {', '.join(dropped)}"
            experiments.append(config)

    elif ablation_type == "reverse":
        # Start with minimal features, add them back
        all_features = set(features_to_ablate)
        for i, feature in enumerate(features_to_ablate):
            # Keep only features up to and including this one
            keep = features_to_ablate[:i+1]
            drop = list(all_features - set(keep))
            config = base_experiment_config.copy()
            config["ablation_target"] = f"forward_{i+1}"
            config["drop_columns"] = drop
            config["description"] = f"Forward selection: keep {', '.join(keep)}"
            experiments.append(config)

    return experiments

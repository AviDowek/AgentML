"""Robust validation service for multi-seed and cross-validation experiments.

This module implements validation strategies for high-stakes ML applications:
- STANDARD: Single run with default validation
- ROBUST: Multiple random seeds + optional cross-validation
- STRICT: Maximum seeds, CV folds, and statistical significance tests
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from app.services.automl_runner import AutoMLResult, get_runner_for_task

logger = logging.getLogger(__name__)


@dataclass
class RobustValidationResult:
    """Result from robust validation with multiple runs."""

    # Primary result (best performing or most stable)
    primary_result: AutoMLResult

    # Aggregated metrics across all runs
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    min_metrics: dict[str, float] = field(default_factory=dict)
    max_metrics: dict[str, float] = field(default_factory=dict)

    # All individual run results
    all_results: list[AutoMLResult] = field(default_factory=list)

    # Validation info
    num_seeds: int = 1
    cv_folds: int = 0
    validation_strategy: str = "STANDARD"

    # Statistical tests (for STRICT mode)
    confidence_interval_95: Optional[tuple[float, float]] = None
    is_statistically_significant: Optional[bool] = None
    baseline_comparison_pvalue: Optional[float] = None

    # Stability metrics
    coefficient_of_variation: Optional[float] = None
    is_stable: bool = True
    stability_warning: Optional[str] = None


def run_robust_validation(
    dataset: pd.DataFrame,
    target_column: str,
    task_type: str,
    primary_metric: str,
    config: dict[str, Any],
    experiment_id: str,
    artifacts_dir: str,
    validation_strategy: str = "STANDARD",
    num_seeds: int = 3,
    cv_folds: int = 5,
) -> RobustValidationResult:
    """Run experiment with robust validation (multi-seed and/or CV).

    Args:
        dataset: Training DataFrame
        target_column: Name of target column
        task_type: ML task type (regression, binary, multiclass)
        primary_metric: Metric to optimize
        config: AutoML configuration
        experiment_id: Unique experiment ID
        artifacts_dir: Directory for model artifacts
        validation_strategy: One of STANDARD, ROBUST, STRICT
        num_seeds: Number of random seeds to run (for ROBUST/STRICT)
        cv_folds: Number of CV folds (for ROBUST/STRICT)

    Returns:
        RobustValidationResult with aggregated metrics
    """
    if validation_strategy == "STANDARD":
        # Single run - delegate directly to runner
        runner = get_runner_for_task(task_type=task_type, artifacts_dir=artifacts_dir)
        result = runner.run_experiment(
            dataset=dataset,
            target_column=target_column,
            task_type=task_type,
            primary_metric=primary_metric,
            config=config,
            experiment_id=experiment_id,
        )
        return RobustValidationResult(
            primary_result=result,
            mean_metrics=result.metrics.copy(),
            all_results=[result],
            num_seeds=1,
            validation_strategy="STANDARD",
        )

    # ROBUST or STRICT mode - run multiple times with different seeds
    seeds = _get_seeds_for_strategy(validation_strategy, num_seeds)

    logger.info(f"Running {validation_strategy} validation with {len(seeds)} seeds")

    all_results = []
    all_metrics = []

    for i, seed in enumerate(seeds):
        logger.info(f"Running seed {i+1}/{len(seeds)} (seed={seed})")

        # Create config with this seed
        seed_config = config.copy()
        seed_config["random_seed"] = seed

        # For STRICT mode with CV, increase bag folds
        if validation_strategy == "STRICT" and cv_folds > 0:
            seed_config["num_bag_folds"] = cv_folds
        elif validation_strategy == "ROBUST" and cv_folds > 0:
            seed_config["num_bag_folds"] = cv_folds

        # Run experiment
        runner = get_runner_for_task(task_type=task_type, artifacts_dir=artifacts_dir)

        # Use unique experiment ID for each seed
        seed_experiment_id = f"{experiment_id}_seed{seed}"

        try:
            result = runner.run_experiment(
                dataset=dataset,
                target_column=target_column,
                task_type=task_type,
                primary_metric=primary_metric,
                config=seed_config,
                experiment_id=seed_experiment_id,
            )
            all_results.append(result)
            all_metrics.append(result.metrics)
            logger.info(f"Seed {seed} completed: {primary_metric}={result.metrics.get(primary_metric, 'N/A')}")
        except Exception as e:
            logger.warning(f"Seed {seed} failed: {e}")
            continue

    if not all_results:
        raise ValueError("All validation runs failed")

    # Aggregate metrics
    aggregated = _aggregate_metrics(all_metrics, primary_metric)

    # Select primary result (best performing on primary metric)
    primary_result = _select_primary_result(all_results, primary_metric)

    # Calculate stability metrics
    stability = _calculate_stability(all_metrics, primary_metric)

    # For STRICT mode, run statistical tests
    statistical = {}
    if validation_strategy == "STRICT":
        statistical = _run_statistical_tests(all_metrics, primary_metric, primary_result)

    return RobustValidationResult(
        primary_result=primary_result,
        mean_metrics=aggregated["mean"],
        std_metrics=aggregated["std"],
        min_metrics=aggregated["min"],
        max_metrics=aggregated["max"],
        all_results=all_results,
        num_seeds=len(all_results),
        cv_folds=cv_folds if validation_strategy in ("ROBUST", "STRICT") else 0,
        validation_strategy=validation_strategy,
        confidence_interval_95=statistical.get("confidence_interval"),
        is_statistically_significant=statistical.get("is_significant"),
        baseline_comparison_pvalue=statistical.get("pvalue"),
        coefficient_of_variation=stability.get("cv"),
        is_stable=stability.get("is_stable", True),
        stability_warning=stability.get("warning"),
    )


def _get_seeds_for_strategy(strategy: str, num_seeds: int) -> list[int]:
    """Get list of random seeds based on validation strategy."""
    if strategy == "ROBUST":
        return [42, 123, 456][:num_seeds]
    elif strategy == "STRICT":
        # More seeds for STRICT mode
        return [42, 123, 456, 789, 1337][:max(num_seeds, 5)]
    else:
        return [42]


def _aggregate_metrics(all_metrics: list[dict], primary_metric: str) -> dict:
    """Aggregate metrics across all runs."""
    # Collect all metric keys
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    result = {"mean": {}, "std": {}, "min": {}, "max": {}}

    for key in all_keys:
        values = []
        for m in all_metrics:
            if key in m and m[key] is not None:
                try:
                    values.append(float(m[key]))
                except (ValueError, TypeError):
                    continue

        if values:
            result["mean"][key] = float(np.mean(values))
            result["std"][key] = float(np.std(values))
            result["min"][key] = float(np.min(values))
            result["max"][key] = float(np.max(values))

    return result


def _select_primary_result(
    all_results: list[AutoMLResult],
    primary_metric: str,
) -> AutoMLResult:
    """Select the primary result based on best performance on primary metric."""
    if len(all_results) == 1:
        return all_results[0]

    # Determine if higher is better
    error_metrics = {
        "rmse", "mse", "mae", "root_mean_squared_error",
        "mean_squared_error", "mean_absolute_error", "log_loss",
        "pinball_loss"
    }
    lower_is_better = any(e in primary_metric.lower() for e in error_metrics)

    best_result = all_results[0]
    best_score = best_result.metrics.get(primary_metric, float('inf') if lower_is_better else float('-inf'))

    for result in all_results[1:]:
        score = result.metrics.get(primary_metric, float('inf') if lower_is_better else float('-inf'))
        if lower_is_better:
            if score < best_score:
                best_score = score
                best_result = result
        else:
            if score > best_score:
                best_score = score
                best_result = result

    return best_result


def _calculate_stability(all_metrics: list[dict], primary_metric: str) -> dict:
    """Calculate stability metrics for the primary metric."""
    values = []
    for m in all_metrics:
        if primary_metric in m and m[primary_metric] is not None:
            try:
                values.append(float(m[primary_metric]))
            except (ValueError, TypeError):
                continue

    if len(values) < 2:
        return {"is_stable": True}

    mean_val = np.mean(values)
    std_val = np.std(values)

    # Coefficient of variation (relative standard deviation)
    if mean_val != 0:
        cv = abs(std_val / mean_val) * 100  # As percentage
    else:
        cv = 0 if std_val == 0 else float('inf')

    # Stability thresholds
    # CV > 10% is concerning, CV > 20% is unstable
    is_stable = cv <= 10
    warning = None

    if cv > 20:
        warning = f"HIGH VARIANCE: Results vary by {cv:.1f}% across seeds. Model may be unstable."
    elif cv > 10:
        warning = f"MODERATE VARIANCE: Results vary by {cv:.1f}% across seeds. Consider more data or simpler models."

    return {
        "cv": cv,
        "is_stable": is_stable,
        "warning": warning,
    }


def _run_statistical_tests(
    all_metrics: list[dict],
    primary_metric: str,
    primary_result: AutoMLResult,
) -> dict:
    """Run statistical significance tests for STRICT validation."""
    values = []
    for m in all_metrics:
        if primary_metric in m and m[primary_metric] is not None:
            try:
                values.append(float(m[primary_metric]))
            except (ValueError, TypeError):
                continue

    if len(values) < 3:
        return {}

    values = np.array(values)

    # 95% confidence interval using t-distribution
    n = len(values)
    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of the mean

    # t-value for 95% CI with n-1 degrees of freedom
    t_val = stats.t.ppf(0.975, n - 1)
    ci_low = mean - t_val * sem
    ci_high = mean + t_val * sem

    # Compare against baseline (if available)
    is_significant = None
    pvalue = None

    baseline_metrics = primary_result.baseline_metrics
    if baseline_metrics:
        # Get baseline score to compare against
        baseline_score = None
        for key in ["simple_model", "logistic", "ridge", "majority_class", "mean_predictor"]:
            if key in baseline_metrics:
                baseline_score = baseline_metrics[key].get(primary_metric)
                break

        if baseline_score is not None:
            # One-sample t-test: are our results significantly better than baseline?
            error_metrics = {
                "rmse", "mse", "mae", "root_mean_squared_error",
                "mean_squared_error", "mean_absolute_error", "log_loss"
            }
            lower_is_better = any(e in primary_metric.lower() for e in error_metrics)

            if lower_is_better:
                # Test if our values are significantly lower than baseline
                t_stat, pvalue = stats.ttest_1samp(values, baseline_score)
                is_significant = t_stat < 0 and pvalue / 2 < 0.05
            else:
                # Test if our values are significantly higher than baseline
                t_stat, pvalue = stats.ttest_1samp(values, baseline_score)
                is_significant = t_stat > 0 and pvalue / 2 < 0.05

    return {
        "confidence_interval": (ci_low, ci_high),
        "is_significant": is_significant,
        "pvalue": pvalue,
    }


def format_robust_validation_summary(result: RobustValidationResult, primary_metric: str) -> str:
    """Format a human-readable summary of robust validation results."""
    lines = [
        f"{'='*60}",
        f"ROBUST VALIDATION SUMMARY ({result.validation_strategy})",
        f"{'='*60}",
        f"Seeds run: {result.num_seeds}",
    ]

    if result.cv_folds > 0:
        lines.append(f"CV folds per seed: {result.cv_folds}")

    lines.append("")

    # Primary metric summary
    if primary_metric in result.mean_metrics:
        mean = result.mean_metrics[primary_metric]
        std = result.std_metrics.get(primary_metric, 0)
        min_val = result.min_metrics.get(primary_metric, mean)
        max_val = result.max_metrics.get(primary_metric, mean)

        lines.extend([
            f"Primary Metric: {primary_metric}",
            f"  Mean:   {mean:.4f}",
            f"  Std:    {std:.4f}",
            f"  Range:  [{min_val:.4f}, {max_val:.4f}]",
        ])

        if result.confidence_interval_95:
            ci_low, ci_high = result.confidence_interval_95
            lines.append(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    lines.append("")

    # Stability assessment
    if result.coefficient_of_variation is not None:
        lines.append(f"Stability (CV): {result.coefficient_of_variation:.1f}%")

    if not result.is_stable:
        lines.append("⚠️ WARNING: Results are unstable across seeds")
    if result.stability_warning:
        lines.append(f"  {result.stability_warning}")

    # Statistical significance
    if result.is_statistically_significant is not None:
        if result.is_statistically_significant:
            lines.append("✓ Results are statistically significant vs baseline")
        else:
            lines.append("⚠ Results are NOT statistically significant vs baseline")

    if result.baseline_comparison_pvalue is not None:
        lines.append(f"  p-value: {result.baseline_comparison_pvalue:.4f}")

    lines.append(f"{'='*60}")

    return "\n".join(lines)

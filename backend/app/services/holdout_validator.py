"""Holdout validation service for preventing overfitting across iterations.

This module manages a persistent holdout set that is:
1. Created once when the first experiment runs
2. Never used for training - only for final evaluation
3. Used to detect overfitting across improvement iterations

The key insight is that cross-validation scores within AutoGluon can improve
while actually overfitting to the training distribution. A truly held-out set
that is NEVER seen during any training phase catches this.
"""
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.dataset_spec import DatasetSpec
from app.models.experiment import Experiment

logger = logging.getLogger(__name__)


# Holdout fraction - this data is NEVER used for training
HOLDOUT_FRACTION = 0.15  # 15% permanently held out


@dataclass
class HoldoutValidationResult:
    """Result of evaluating on the holdout set."""
    holdout_score: float
    metric_name: str
    num_samples: int
    predictions: List[Dict[str, Any]]
    score_history: List[Dict[str, Any]]  # All historical holdout scores
    is_degrading: bool  # True if holdout score is worse than best
    best_score: float
    degradation_amount: float  # How much worse than best (0 if improving)
    recommendation: str  # "continue", "warning", or "stop"


def get_or_create_holdout_indices(
    df: pd.DataFrame,
    dataset_spec: DatasetSpec,
    db: Session,
    target_column: str,
    task_type: str = "binary",
    validation_strategy: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """Get or create persistent holdout indices for a dataset spec.

    On first call: Creates holdout indices and stores them in spec_json.
    On subsequent calls: Returns the same holdout indices for consistency.

    Args:
        df: Full dataset
        dataset_spec: DatasetSpec model instance
        db: Database session
        target_column: Name of target column (for stratification)
        task_type: ML task type for stratification decisions
        validation_strategy: Dict from AI agents with split_strategy, group_column, etc.

    Returns:
        Tuple of (training_df, holdout_df, holdout_indices)
    """
    spec_json = dataset_spec.spec_json or {}
    holdout_config = spec_json.get("holdout_validation", {})

    # Extract split strategy from validation_strategy (set by AI agents)
    validation_strategy = validation_strategy or {}
    split_strategy = validation_strategy.get("split_strategy")
    group_column = validation_strategy.get("group_column")

    # Check if holdout indices already exist
    existing_indices = holdout_config.get("indices")
    holdout_hash = holdout_config.get("data_hash")
    existing_strategy = holdout_config.get("split_strategy")

    # Create a hash of the data to detect if it changed
    current_hash = _hash_dataframe(df)

    # Re-create holdout if data changed OR if the split strategy changed
    strategy_changed = split_strategy and split_strategy != existing_strategy

    if existing_indices and holdout_hash == current_hash and not strategy_changed:
        # Use existing holdout indices
        logger.info(f"Using existing holdout set: {len(existing_indices)} samples (strategy: {existing_strategy or 'default'})")
        holdout_indices = existing_indices
    else:
        # Create new holdout indices
        if strategy_changed:
            logger.info(f"Split strategy changed from '{existing_strategy}' to '{split_strategy}', recreating holdout...")
        else:
            logger.info("Creating new persistent holdout set...")

        holdout_indices = _create_stratified_holdout_indices(
            df, target_column, task_type, HOLDOUT_FRACTION,
            split_strategy=split_strategy,
            group_column=group_column
        )

        # Store in spec_json
        spec_json["holdout_validation"] = {
            "indices": holdout_indices,
            "data_hash": current_hash,
            "holdout_fraction": HOLDOUT_FRACTION,
            "num_samples": len(holdout_indices),
            "split_strategy": split_strategy,
            "group_column": group_column,
            "created_at": pd.Timestamp.now().isoformat(),
        }
        dataset_spec.spec_json = spec_json
        db.commit()
        logger.info(f"Created and stored holdout set: {len(holdout_indices)} samples (strategy: {split_strategy or 'default'})")

    # Split dataframe
    holdout_mask = df.index.isin(holdout_indices)
    holdout_df = df[holdout_mask].copy()
    training_df = df[~holdout_mask].copy()

    logger.info(f"Split: {len(training_df)} training, {len(holdout_df)} holdout")

    return training_df, holdout_df, holdout_indices


def _hash_dataframe(df: pd.DataFrame, sample_size: int = 1000) -> str:
    """Create a hash to detect if data changed significantly."""
    # Sample for efficiency on large datasets
    if len(df) > sample_size:
        sample = df.sample(n=sample_size, random_state=42)
    else:
        sample = df

    # Hash based on shape and sample values
    hash_input = f"{df.shape}_{sample.values.tobytes()}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def _create_stratified_holdout_indices(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    holdout_fraction: float,
    split_strategy: str = None,
    group_column: str = None,
) -> List[int]:
    """Create holdout indices using the specified split strategy.

    The split strategy is determined by AI agents and should match the training split.

    Args:
        df: Input DataFrame
        target_column: Name of target column
        task_type: ML task type (binary, multiclass, regression, etc.)
        holdout_fraction: Fraction of data to hold out
        split_strategy: Strategy from AI agents - "temporal", "random", "stratified", "group"
        group_column: Column name for group-based splits
    """
    from sklearn.model_selection import train_test_split

    n_samples = len(df)
    indices = list(range(n_samples))

    # Use stratification for classification tasks as default
    is_classification = task_type in ("binary", "multiclass", "classification")

    # If no strategy specified, use sensible defaults
    if not split_strategy:
        if is_classification:
            split_strategy = "stratified"
        else:
            split_strategy = "random"
        logger.info(f"No split_strategy specified for holdout, defaulting to '{split_strategy}'")

    if split_strategy == "temporal":
        # Temporal split: holdout is the last N% of data (most recent)
        # This ensures holdout set represents "future" data the model hasn't seen
        split_idx = int(n_samples * (1 - holdout_fraction))
        holdout_indices = indices[split_idx:]
        logger.info(f"Using TEMPORAL holdout split: {len(holdout_indices)} samples from end of dataset")
    elif split_strategy == "group" and group_column and group_column in df.columns:
        # Group-based split: keep related rows together
        from sklearn.model_selection import GroupShuffleSplit
        groups = df[group_column]
        gss = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=42)
        _, holdout_idx = next(gss.split(df, groups=groups))
        holdout_indices = [indices[i] for i in holdout_idx]
        logger.info(f"Using GROUP holdout split by '{group_column}': {len(holdout_indices)} samples")
    elif split_strategy == "stratified" and is_classification and target_column in df.columns:
        try:
            _, holdout_indices = train_test_split(
                indices,
                test_size=holdout_fraction,
                random_state=42,
                stratify=df[target_column].values,
            )
            logger.info(f"Using STRATIFIED holdout split: {len(holdout_indices)} samples")
        except ValueError:
            # Stratification failed (e.g., too few samples in a class)
            logger.warning("Stratified split failed, using random split")
            _, holdout_indices = train_test_split(
                indices,
                test_size=holdout_fraction,
                random_state=42,
            )
    else:
        # Random split (default)
        _, holdout_indices = train_test_split(
            indices,
            test_size=holdout_fraction,
            random_state=42,
        )
        logger.info(f"Using RANDOM holdout split: {len(holdout_indices)} samples")

    return list(holdout_indices)


def evaluate_on_holdout(
    predictor,
    holdout_df: pd.DataFrame,
    target_column: str,
    metric_name: str,
    task_type: str = "binary",
) -> Dict[str, Any]:
    """Evaluate a trained model on the holdout set.

    Args:
        predictor: Trained AutoGluon predictor
        holdout_df: Holdout DataFrame
        target_column: Name of target column
        metric_name: Metric to compute
        task_type: ML task type

    Returns:
        Dict with score and predictions
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score
    )

    # Get predictions
    X_holdout = holdout_df.drop(columns=[target_column])
    y_holdout = holdout_df[target_column]

    predictions = predictor.predict(X_holdout)

    # Calculate metric
    is_classification = task_type in ("binary", "multiclass", "classification")

    if is_classification:
        try:
            # Try to get probability predictions for AUC
            proba = predictor.predict_proba(X_holdout)
            if task_type == "binary" and len(proba.columns) == 2:
                proba_positive = proba.iloc[:, 1].values
            else:
                proba_positive = None
        except Exception:
            proba_positive = None

        if metric_name in ("roc_auc", "auc") and proba_positive is not None:
            try:
                score = roc_auc_score(y_holdout, proba_positive)
            except ValueError:
                score = accuracy_score(y_holdout, predictions)
        elif metric_name == "f1":
            score = f1_score(y_holdout, predictions, average="weighted")
        else:
            score = accuracy_score(y_holdout, predictions)
    else:
        # Regression metrics
        if metric_name in ("rmse", "root_mean_squared_error"):
            score = -np.sqrt(mean_squared_error(y_holdout, predictions))  # Negative for consistency (higher is better)
        elif metric_name in ("mae", "mean_absolute_error"):
            score = -mean_absolute_error(y_holdout, predictions)
        elif metric_name == "r2":
            score = r2_score(y_holdout, predictions)
        else:
            score = -np.sqrt(mean_squared_error(y_holdout, predictions))

    return {
        "score": float(score),
        "metric": metric_name,
        "num_samples": len(holdout_df),
        "predictions": predictions.tolist()[:100],  # Limit for storage
    }


def record_holdout_score(
    db: Session,
    experiment: Experiment,
    holdout_score: float,
    metric_name: str,
) -> HoldoutValidationResult:
    """Record a holdout score and check for overfitting.

    Args:
        db: Database session
        experiment: Current experiment
        holdout_score: Score on holdout set
        metric_name: Name of the metric

    Returns:
        HoldoutValidationResult with analysis
    """
    # Get the root experiment to access all iteration scores
    root = _get_root_experiment(db, experiment)

    # Collect all holdout scores from the chain
    score_history = _collect_holdout_scores(db, root)

    # Add current score
    current_entry = {
        "experiment_id": str(experiment.id),
        "iteration": experiment.iteration_number,
        "score": holdout_score,
        "metric": metric_name,
    }
    score_history.append(current_entry)

    # Store in experiment's improvement context
    improvement_context = experiment.improvement_context_json or {}
    improvement_context["holdout_score"] = holdout_score
    improvement_context["holdout_metric"] = metric_name
    experiment.improvement_context_json = improvement_context
    db.commit()

    # Analyze for overfitting
    scores = [entry["score"] for entry in score_history]
    best_score = max(scores)  # Assuming higher is better
    best_iteration = scores.index(best_score) + 1

    is_degrading = holdout_score < best_score
    degradation = best_score - holdout_score if is_degrading else 0

    # Determine recommendation
    if len(scores) < 2:
        recommendation = "continue"
    elif degradation > 0.05:  # More than 5% degradation
        recommendation = "stop"
    elif degradation > 0.02:  # 2-5% degradation
        recommendation = "warning"
    elif len(scores) >= 3 and all(scores[-1] <= s for s in scores[-3:]):
        # Score hasn't improved in last 3 iterations
        recommendation = "warning"
    else:
        recommendation = "continue"

    logger.info(
        f"Holdout validation: score={holdout_score:.4f}, best={best_score:.4f}, "
        f"degradation={degradation:.4f}, recommendation={recommendation}"
    )

    return HoldoutValidationResult(
        holdout_score=holdout_score,
        metric_name=metric_name,
        num_samples=0,  # Will be filled by caller
        predictions=[],
        score_history=score_history,
        is_degrading=is_degrading,
        best_score=best_score,
        degradation_amount=degradation,
        recommendation=recommendation,
    )


def _get_root_experiment(db: Session, experiment: Experiment) -> Experiment:
    """Get the root experiment in an iteration chain."""
    current = experiment
    while current.parent_experiment_id:
        parent = db.query(Experiment).filter(
            Experiment.id == current.parent_experiment_id
        ).first()
        if parent:
            current = parent
        else:
            break
    return current


def _collect_holdout_scores(db: Session, root: Experiment) -> List[Dict[str, Any]]:
    """Collect all holdout scores from an experiment chain."""
    from app.models.experiment import Trial
    from app.models.model_version import ModelVersion

    scores = []

    # BFS through the chain
    to_visit = [root]
    visited = set()

    while to_visit:
        exp = to_visit.pop(0)
        if exp.id in visited:
            continue
        visited.add(exp.id)

        holdout_score = None
        holdout_metric = "unknown"

        # Check 1: improvement_context_json (legacy location)
        ctx = exp.improvement_context_json or {}
        if "holdout_score" in ctx:
            holdout_score = ctx["holdout_score"]
            holdout_metric = ctx.get("holdout_metric", "unknown")

        # Check 2: Trial metrics_json (Modal training stores here)
        if holdout_score is None:
            trials = db.query(Trial).filter(Trial.experiment_id == exp.id).all()
            for trial in trials:
                trial_metrics = trial.metrics_json or {}
                # Look for holdout_root_mean_squared_error, holdout_r2, etc.
                for key, value in trial_metrics.items():
                    if key.startswith("holdout_") and isinstance(value, (int, float)):
                        # Use the primary metric's holdout if we can find it
                        primary_metric = exp.primary_metric or "rmse"
                        holdout_key = f"holdout_{primary_metric}"
                        if holdout_key in trial_metrics:
                            holdout_score = abs(trial_metrics[holdout_key])  # abs() to handle sign-flipped metrics
                            holdout_metric = primary_metric
                            break
                        elif key == "holdout_root_mean_squared_error":
                            holdout_score = abs(value)
                            holdout_metric = "rmse"
                            break

        # Check 3: Model version metrics_json
        if holdout_score is None:
            models = db.query(ModelVersion).filter(ModelVersion.experiment_id == exp.id).all()
            for model in models:
                model_metrics = model.metrics_json or {}
                for key, value in model_metrics.items():
                    if key.startswith("holdout_") and isinstance(value, (int, float)):
                        primary_metric = exp.primary_metric or "rmse"
                        holdout_key = f"holdout_{primary_metric}"
                        if holdout_key in model_metrics:
                            holdout_score = abs(model_metrics[holdout_key])
                            holdout_metric = primary_metric
                            break
                        elif key == "holdout_root_mean_squared_error":
                            holdout_score = abs(value)
                            holdout_metric = "rmse"
                            break

        if holdout_score is not None:
            scores.append({
                "experiment_id": str(exp.id),
                "iteration": exp.iteration_number or 1,
                "score": holdout_score,
                "metric": holdout_metric,
            })

        # Add children
        children = db.query(Experiment).filter(
            Experiment.parent_experiment_id == exp.id
        ).all()
        to_visit.extend(children)

    # Sort by iteration
    scores.sort(key=lambda x: x["iteration"])
    return scores


def calculate_overfitting_risk(
    scores: List[float],
    current_idx: int,
    best_score: float,
    higher_is_better: bool = True,
) -> int:
    """Calculate overfitting risk percentage for a given iteration.

    Risk factors:
    - Distance from best score (higher = more risk)
    - Declining trend (increases risk)
    - Number of iterations since best (increases risk)

    Args:
        scores: List of holdout scores
        current_idx: Current iteration index
        best_score: Best score seen so far
        higher_is_better: True for accuracy/AUC, False for RMSE/MAE

    Returns:
        Integer 0-100 representing overfitting risk percentage
    """
    if len(scores) < 2 or current_idx == 0:
        return 0

    current_score = scores[current_idx]

    # Factor 1: Degradation from best (0-50 points)
    # For maximize metrics: degradation = best - current (positive if getting worse)
    # For minimize metrics: degradation = current - best (positive if getting worse)
    if abs(best_score) > 0.0001:
        if higher_is_better:
            degradation = best_score - current_score
        else:
            degradation = current_score - best_score
        degradation_pct = max(0, degradation / abs(best_score)) * 100
        degradation_risk = min(50, degradation_pct * 10)  # Max 50 points
    else:
        degradation_risk = 0

    # Factor 2: Recent trend (0-30 points)
    if current_idx >= 2:
        recent = scores[max(0, current_idx - 2):current_idx + 1]
        if len(recent) >= 2:
            # Detect degrading trend based on metric direction
            if higher_is_better:
                is_degrading = recent[-1] < recent[0]
                decline = recent[0] - recent[-1]
            else:
                is_degrading = recent[-1] > recent[0]
                decline = recent[-1] - recent[0]

            if is_degrading and abs(recent[0]) > 0.0001:
                decline_pct = abs(decline / recent[0]) * 100
                trend_risk = min(30, decline_pct * 6)
            else:
                trend_risk = 0
        else:
            trend_risk = 0
    else:
        trend_risk = 0

    # Factor 3: Iterations since best (0-20 points)
    best_idx = scores.index(best_score)
    iterations_since_best = current_idx - best_idx
    staleness_risk = min(20, iterations_since_best * 5)

    total_risk = int(degradation_risk + trend_risk + staleness_risk)
    return min(100, max(0, total_risk))


def get_overfitting_report(
    db: Session,
    experiment: Experiment,
) -> Dict[str, Any]:
    """Generate a report on overfitting status for an experiment chain.

    Returns a dict with:
    - score_history: List of all holdout scores with risk percentages
    - trend: "improving", "stable", "degrading"
    - best_iteration: Which iteration had best holdout score
    - recommendation: What to do next
    - overall_risk: Current overfitting risk percentage (0-100)
    """
    root = _get_root_experiment(db, experiment)
    score_history = _collect_holdout_scores(db, root)

    # Determine metric direction from experiment
    # "maximize" for accuracy, AUC, etc. - "minimize" for RMSE, MAE, etc.
    higher_is_better = experiment.metric_direction != "minimize"

    if len(score_history) < 1:
        return {
            "score_history": [],
            "trend": "unknown",
            "best_iteration": 1,
            "best_score": 0,
            "current_score": 0,
            "recommendation": "continue",
            "message": "No holdout scores recorded yet",
            "overall_risk": 0,
            "risk_level": "low",
        }

    scores = [s["score"] for s in score_history]
    # Use max for higher-is-better metrics, min for lower-is-better
    best_score = max(scores) if higher_is_better else min(scores)
    best_iteration = scores.index(best_score) + 1
    current_score = scores[-1] if scores else 0

    # Calculate risk for each iteration
    for i, entry in enumerate(score_history):
        entry["overfitting_risk"] = calculate_overfitting_risk(
            scores, i, best_score, higher_is_better=higher_is_better
        )
        entry["is_best"] = (i == scores.index(best_score))

        # Determine status for each iteration
        if entry["is_best"]:
            entry["status"] = "best"
        elif entry["overfitting_risk"] >= 60:
            entry["status"] = "high_risk"
        elif entry["overfitting_risk"] >= 30:
            entry["status"] = "warning"
        else:
            entry["status"] = "healthy"

    # Current overall risk
    overall_risk = score_history[-1]["overfitting_risk"] if score_history else 0

    # Risk level classification
    if overall_risk >= 60:
        risk_level = "high"
    elif overall_risk >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    if len(score_history) < 2:
        return {
            "score_history": score_history,
            "trend": "unknown",
            "best_iteration": best_iteration,
            "best_score": best_score,
            "current_score": current_score,
            "recommendation": "continue",
            "message": "Not enough iterations to determine trend",
            "overall_risk": overall_risk,
            "risk_level": risk_level,
        }

    # Determine trend (account for metric direction)
    if len(scores) >= 3:
        recent = scores[-3:]
        if higher_is_better:
            # Higher scores are better
            if recent[-1] > recent[0]:
                trend = "improving"
            elif recent[-1] < recent[0] - 0.02:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            # Lower scores are better (RMSE, MAE, etc.)
            if recent[-1] < recent[0]:
                trend = "improving"
            elif recent[-1] > recent[0] + 0.02:
                trend = "degrading"
            else:
                trend = "stable"
    else:
        if higher_is_better:
            trend = "improving" if scores[-1] > scores[0] else "degrading"
        else:
            trend = "improving" if scores[-1] < scores[0] else "degrading"

    # Recommendation based on risk
    if overall_risk >= 70:
        recommendation = "stop"
        message = f"High overfitting risk ({overall_risk}%). Holdout score degraded from {best_score:.4f} (iteration {best_iteration}) to {current_score:.4f}. Consider reverting."
    elif overall_risk >= 40:
        recommendation = "warning"
        message = f"Medium overfitting risk ({overall_risk}%). Best holdout score was {best_score:.4f} at iteration {best_iteration}. Current: {current_score:.4f}."
    elif trend == "stable" and len(scores) >= 5:
        recommendation = "warning"
        message = f"Holdout score has plateaued around {current_score:.4f}. Additional iterations may not help."
    else:
        recommendation = "continue"
        message = f"Low overfitting risk ({overall_risk}%). Safe to continue iterating."

    return {
        "score_history": score_history,
        "trend": trend,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "current_score": current_score,
        "recommendation": recommendation,
        "message": message,
        "overall_risk": overall_risk,
        "risk_level": risk_level,
    }

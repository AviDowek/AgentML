"""Ensemble building service for combining multiple trained models.

This module provides functionality to combine predictions from multiple
AutoML experiments to create a stronger ensemble model.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EnsembleMember:
    """A single model in the ensemble."""
    experiment_id: str
    artifact_path: str
    score: float
    weight: float = 1.0
    task_type: str = ""


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    predictions: pd.Series
    prediction_probas: Optional[pd.DataFrame] = None
    member_weights: dict[str, float] = field(default_factory=dict)
    ensemble_method: str = "weighted_average"
    individual_scores: dict[str, float] = field(default_factory=dict)
    ensemble_score: Optional[float] = None


class EnsembleBuilder:
    """Builder for creating ensembles from multiple trained models."""

    def __init__(self, task_type: str = "binary"):
        """Initialize the ensemble builder.

        Args:
            task_type: ML task type (binary, multiclass, regression)
        """
        self.task_type = task_type
        self.members: List[EnsembleMember] = []
        self._predictors = {}

    def add_member(
        self,
        experiment_id: str,
        artifact_path: str,
        score: float,
        weight: Optional[float] = None,
    ) -> None:
        """Add a model to the ensemble.

        Args:
            experiment_id: ID of the experiment
            artifact_path: Path to the trained model artifacts
            score: Validation score of the model
            weight: Optional custom weight (defaults to score-based)
        """
        if weight is None:
            # Default weight based on score
            # For error metrics (lower is better), invert
            weight = score

        member = EnsembleMember(
            experiment_id=experiment_id,
            artifact_path=artifact_path,
            score=score,
            weight=weight,
            task_type=self.task_type,
        )
        self.members.append(member)
        logger.info(f"Added ensemble member: {experiment_id} (score={score:.4f}, weight={weight:.4f})")

    def add_members_from_experiments(
        self,
        experiments: List[dict],
        top_k: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
    ) -> None:
        """Add members from a list of experiment results.

        Args:
            experiments: List of experiment dicts with 'id', 'artifact_path', 'score'
            top_k: Only use top K performing experiments
            min_score_threshold: Minimum score required to include
        """
        # Sort by score (assuming higher is better for most metrics)
        sorted_exps = sorted(
            experiments,
            key=lambda x: x.get("score", 0) or 0,
            reverse=True,
        )

        # Apply filters
        if min_score_threshold is not None:
            sorted_exps = [e for e in sorted_exps if (e.get("score", 0) or 0) >= min_score_threshold]

        if top_k is not None:
            sorted_exps = sorted_exps[:top_k]

        for exp in sorted_exps:
            self.add_member(
                experiment_id=str(exp.get("id", "")),
                artifact_path=exp.get("artifact_path", ""),
                score=exp.get("score", 0) or 0,
            )

    def _load_predictor(self, artifact_path: str):
        """Load a predictor from disk (cached)."""
        if artifact_path in self._predictors:
            return self._predictors[artifact_path]

        from app.services.automl_runner import TabularRunner
        runner = TabularRunner()

        if not Path(artifact_path).exists():
            raise ValueError(f"Artifact path does not exist: {artifact_path}")

        predictor = runner.load_predictor(artifact_path)
        self._predictors[artifact_path] = predictor
        return predictor

    def _normalize_weights(self) -> dict[str, float]:
        """Normalize member weights to sum to 1."""
        if not self.members:
            return {}

        total_weight = sum(m.weight for m in self.members)
        if total_weight == 0:
            # Equal weights if all zeros
            return {m.experiment_id: 1.0 / len(self.members) for m in self.members}

        return {m.experiment_id: m.weight / total_weight for m in self.members}

    def predict(
        self,
        data: pd.DataFrame,
        method: str = "weighted_average",
    ) -> EnsembleResult:
        """Generate ensemble predictions.

        Args:
            data: Input DataFrame for prediction
            method: Ensemble method:
                - "weighted_average": Weight predictions by model score
                - "simple_average": Equal weight averaging
                - "voting": Majority voting (classification only)
                - "median": Median of predictions (regression)

        Returns:
            EnsembleResult with combined predictions
        """
        if not self.members:
            raise ValueError("No ensemble members added")

        normalized_weights = self._normalize_weights()

        # Collect predictions from all members
        all_predictions = []
        all_probas = []
        is_classification = self.task_type in ("binary", "multiclass")

        for member in self.members:
            try:
                predictor = self._load_predictor(member.artifact_path)
                preds = predictor.predict(data)
                all_predictions.append({
                    "experiment_id": member.experiment_id,
                    "predictions": preds,
                    "weight": normalized_weights[member.experiment_id],
                })

                if is_classification:
                    try:
                        probas = predictor.predict_proba(data)
                        all_probas.append({
                            "experiment_id": member.experiment_id,
                            "probas": probas,
                            "weight": normalized_weights[member.experiment_id],
                        })
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"Failed to get predictions from {member.experiment_id}: {e}")
                continue

        if not all_predictions:
            raise ValueError("No valid predictions from any ensemble member")

        # Combine predictions based on method
        if method == "simple_average":
            final_predictions = self._simple_average(all_predictions, is_classification)
        elif method == "weighted_average":
            final_predictions = self._weighted_average(all_predictions, all_probas, is_classification)
        elif method == "voting":
            final_predictions = self._voting(all_predictions)
        elif method == "median":
            final_predictions = self._median(all_predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        # Combine probabilities if available
        combined_probas = None
        if all_probas and is_classification:
            combined_probas = self._combine_probabilities(all_probas, method)

        return EnsembleResult(
            predictions=final_predictions,
            prediction_probas=combined_probas,
            member_weights=normalized_weights,
            ensemble_method=method,
            individual_scores={m.experiment_id: m.score for m in self.members},
        )

    def _simple_average(
        self,
        predictions: List[dict],
        is_classification: bool,
    ) -> pd.Series:
        """Simple average of all predictions."""
        if is_classification:
            # For classification, use mode (most common prediction)
            pred_df = pd.DataFrame({p["experiment_id"]: p["predictions"] for p in predictions})
            return pred_df.mode(axis=1)[0]
        else:
            # For regression, use mean
            pred_values = np.stack([p["predictions"].values for p in predictions])
            return pd.Series(np.mean(pred_values, axis=0), index=predictions[0]["predictions"].index)

    def _weighted_average(
        self,
        predictions: List[dict],
        probas: List[dict],
        is_classification: bool,
    ) -> pd.Series:
        """Weighted average of predictions."""
        if is_classification and probas:
            # For classification, weight the probabilities then take argmax
            combined_proba = self._combine_probabilities(probas, "weighted_average")
            if combined_proba is not None:
                return combined_proba.idxmax(axis=1)

        # Fallback to weighted voting for classification without probas
        if is_classification:
            # Weight votes by model weight
            pred_df = pd.DataFrame({p["experiment_id"]: p["predictions"] for p in predictions})
            weights = {p["experiment_id"]: p["weight"] for p in predictions}

            def weighted_mode(row):
                vote_weights = {}
                for col in pred_df.columns:
                    val = row[col]
                    vote_weights[val] = vote_weights.get(val, 0) + weights[col]
                return max(vote_weights, key=vote_weights.get)

            return pred_df.apply(weighted_mode, axis=1)
        else:
            # For regression, weighted mean
            pred_values = np.stack([p["predictions"].values for p in predictions])
            weights = np.array([p["weight"] for p in predictions])
            weighted_sum = np.sum(pred_values * weights[:, np.newaxis], axis=0)
            return pd.Series(weighted_sum, index=predictions[0]["predictions"].index)

    def _voting(self, predictions: List[dict]) -> pd.Series:
        """Majority voting for classification."""
        pred_df = pd.DataFrame({p["experiment_id"]: p["predictions"] for p in predictions})
        return pred_df.mode(axis=1)[0]

    def _median(self, predictions: List[dict]) -> pd.Series:
        """Median of predictions for regression."""
        pred_values = np.stack([p["predictions"].values for p in predictions])
        return pd.Series(np.median(pred_values, axis=0), index=predictions[0]["predictions"].index)

    def _combine_probabilities(
        self,
        probas: List[dict],
        method: str,
    ) -> Optional[pd.DataFrame]:
        """Combine probability predictions from multiple models."""
        if not probas:
            return None

        # Get all class columns
        first_proba = probas[0]["probas"]
        if not isinstance(first_proba, pd.DataFrame):
            return None

        columns = first_proba.columns.tolist()

        if method == "weighted_average":
            # Weighted average of probabilities
            combined = pd.DataFrame(0.0, index=first_proba.index, columns=columns)
            for p in probas:
                proba_df = p["probas"]
                if isinstance(proba_df, pd.DataFrame):
                    # Align columns in case they differ
                    for col in columns:
                        if col in proba_df.columns:
                            combined[col] += proba_df[col] * p["weight"]
            return combined
        else:
            # Simple average
            combined = pd.DataFrame(0.0, index=first_proba.index, columns=columns)
            for p in probas:
                proba_df = p["probas"]
                if isinstance(proba_df, pd.DataFrame):
                    for col in columns:
                        if col in proba_df.columns:
                            combined[col] += proba_df[col]
            return combined / len(probas)

    def evaluate(
        self,
        data: pd.DataFrame,
        target_column: str,
        metric: str,
        method: str = "weighted_average",
    ) -> dict:
        """Evaluate ensemble performance on a dataset.

        Args:
            data: DataFrame with features and target
            target_column: Name of target column
            metric: Metric to evaluate
            method: Ensemble method to use

        Returns:
            Dict with ensemble_score and individual_scores
        """
        from sklearn import metrics as sklearn_metrics

        X = data.drop(columns=[target_column])
        y_true = data[target_column]

        result = self.predict(X, method=method)
        y_pred = result.predictions

        # Calculate metric
        is_classification = self.task_type in ("binary", "multiclass")

        if metric in ("accuracy", "accuracy_score"):
            score = sklearn_metrics.accuracy_score(y_true, y_pred)
        elif metric in ("f1", "f1_score"):
            score = sklearn_metrics.f1_score(y_true, y_pred, average="binary" if self.task_type == "binary" else "macro")
        elif metric in ("f1_macro", "f1_macro_score"):
            score = sklearn_metrics.f1_score(y_true, y_pred, average="macro")
        elif metric in ("roc_auc", "auc"):
            if result.prediction_probas is not None and self.task_type == "binary":
                # Use probabilities for AUC
                proba = result.prediction_probas.iloc[:, 1] if result.prediction_probas.shape[1] > 1 else result.prediction_probas.iloc[:, 0]
                score = sklearn_metrics.roc_auc_score(y_true, proba)
            else:
                score = sklearn_metrics.roc_auc_score(y_true, y_pred)
        elif metric in ("rmse", "root_mean_squared_error"):
            score = np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred))
        elif metric in ("mse", "mean_squared_error"):
            score = sklearn_metrics.mean_squared_error(y_true, y_pred)
        elif metric in ("mae", "mean_absolute_error"):
            score = sklearn_metrics.mean_absolute_error(y_true, y_pred)
        elif metric == "r2":
            score = sklearn_metrics.r2_score(y_true, y_pred)
        else:
            # Try to get metric from sklearn
            try:
                metric_fn = getattr(sklearn_metrics, metric)
                score = metric_fn(y_true, y_pred)
            except AttributeError:
                logger.warning(f"Unknown metric: {metric}")
                score = None

        result.ensemble_score = score
        return {
            "ensemble_score": score,
            "individual_scores": result.individual_scores,
            "member_weights": result.member_weights,
            "method": method,
        }


def build_ensemble_from_session_experiments(
    experiments: List[dict],
    task_type: str,
    top_k: int = 5,
    min_improvement_threshold: float = 0.01,
) -> EnsembleBuilder:
    """Build an ensemble from Auto DS session experiments.

    Args:
        experiments: List of experiment dicts with results
        task_type: ML task type
        top_k: Number of top experiments to include
        min_improvement_threshold: Minimum score to include

    Returns:
        Configured EnsembleBuilder
    """
    builder = EnsembleBuilder(task_type=task_type)

    # Filter to completed experiments with valid scores
    valid_experiments = []
    for exp in experiments:
        if exp.get("status") == "completed" and exp.get("artifact_path"):
            score = exp.get("score")
            if score is not None:
                valid_experiments.append(exp)

    if not valid_experiments:
        logger.warning("No valid experiments found for ensemble")
        return builder

    # Find best score
    best_score = max(e.get("score", 0) for e in valid_experiments)

    # Calculate relative improvement threshold
    threshold = best_score * (1 - min_improvement_threshold)

    builder.add_members_from_experiments(
        experiments=valid_experiments,
        top_k=top_k,
        min_score_threshold=threshold,
    )

    logger.info(f"Built ensemble with {len(builder.members)} members from {len(valid_experiments)} experiments")
    return builder

"""Iteration Context Agent - Gathers context from all iterations.

This agent traverses the experiment chain to gather metrics, errors,
improvement attempts, and data statistics for informing improvements.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from app.models import (
    AgentStepType,
    DatasetSpec,
    Experiment,
    ModelVersion,
    Trial,
)
from app.models.experiment import ExperimentStatus
from app.services.agents.base import BaseAgent


def _collect_feature_engineering_feedback(experiments: List[Experiment]) -> Dict[str, Any]:
    """Collect feature engineering feedback from all experiments in a chain."""
    successful = []
    failed = []
    available_columns = []

    for exp in experiments:
        ctx = exp.improvement_context_json or {}

        fe_results = ctx.get("feature_engineering_result", {})
        if fe_results:
            for feat in fe_results.get("successful_features", []):
                if feat not in successful:
                    successful.append({
                        **feat,
                        "iteration": exp.iteration_number,
                    })

            for feat in fe_results.get("failed_features", []):
                failed.append({
                    **feat,
                    "iteration": exp.iteration_number,
                })

            cols = fe_results.get("available_columns", [])
            if cols:
                available_columns = cols

        if exp.dataset_spec:
            spec_json = exp.dataset_spec.spec_json or {}
            eng_feats = spec_json.get("engineered_features", [])
            for feat in eng_feats:
                name = feat.get("output_column")
                if name and not any(s.get("feature") == name for s in successful):
                    successful.append({
                        "feature": name,
                        "formula": feat.get("formula", ""),
                        "description": feat.get("description", ""),
                        "iteration": exp.iteration_number,
                    })

    return {
        "successful_features": successful,
        "failed_features": failed,
        "available_columns": available_columns,
        "summary": f"{len(successful)} successful, {len(failed)} failed features across iterations"
    }


class IterationContextAgent(BaseAgent):
    """Gathers complete context from all iterations.

    Input JSON:
        - experiment_id: UUID of the experiment to improve
        - primary_metric: The metric to track (default: roc_auc)

    Output:
        - iteration_history: List of all iterations with metrics and status
        - error_history: List of errors encountered
        - improvement_attempts: List of changes tried and their outcomes
        - best_score: Best score achieved across all iterations
        - current_score: Current score
        - score_trend: Whether scores are improving, declining, or flat
        - data_statistics: Actual data statistics if loading succeeds
        - dataset_spec: Dataset specification info
        - project: Project info
    """

    name = "iteration_context"
    step_type = AgentStepType.ITERATION_CONTEXT

    async def execute(self) -> Dict[str, Any]:
        """Execute iteration context gathering."""
        experiment_id = self.require_input("experiment_id")
        primary_metric = self.get_input("primary_metric", "roc_auc")

        self.logger.info(f"Gathering iteration context for experiment {experiment_id}")

        # Load experiment
        experiment = self.db.query(Experiment).filter(
            Experiment.id == experiment_id
        ).first()
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Get dataset spec
        dataset_spec = experiment.dataset_spec
        if not dataset_spec:
            raise ValueError("Experiment has no dataset spec")

        self.logger.thinking("Traversing experiment chain to gather history...")

        # Gather iteration context
        context = self._gather_iteration_context(experiment, primary_metric)

        self.logger.info(f"Found {context['total_iterations']} iterations in chain")
        self.logger.info(f"Score trend: {context['score_trend']}")
        self.logger.info(f"Best score: {context['best_score']:.4f}")
        self.logger.info(f"Current score: {context['current_score']:.4f}")

        if context['error_history']:
            self.logger.warning(f"Found {len(context['error_history'])} errors in history")
            for err in context['error_history'][:3]:
                self.logger.thinking(f"  Iteration {err['iteration']}: {err['error'][:100]}...")

        # Load actual data statistics
        self.logger.info("Loading actual dataset for analysis...")
        data_statistics = self._load_data_statistics(dataset_spec)

        if data_statistics:
            self.logger.info(f"Dataset: {data_statistics['row_count']} rows, {data_statistics['column_count']} columns")
            context["data_statistics"] = data_statistics
        else:
            self.logger.warning("Could not load dataset - will proceed with metadata only")

        # Add dataset spec info
        context["dataset_spec"] = {
            "id": str(dataset_spec.id),
            "name": dataset_spec.name,
            "target_column": dataset_spec.target_column,
            "feature_columns": dataset_spec.feature_columns or [],
            "spec_json": dataset_spec.spec_json or {},
        }

        # Add project info
        project = experiment.project
        if project:
            context["project"] = {
                "id": str(project.id),
                "name": project.name,
                "task_type": project.task_type.value if project.task_type else "unknown",
                "description": project.description or "",
            }

        self.logger.summary(
            f"Gathered context from {context['total_iterations']} iterations. "
            f"Score: {context['current_score']:.4f} (trend: {context['score_trend']}). "
            f"{len(context['error_history'])} errors found."
        )

        return context

    def _gather_iteration_context(
        self,
        experiment: Experiment,
        primary_metric: str,
    ) -> Dict[str, Any]:
        """Gather complete context from all iterations in an experiment chain."""
        # Find root experiment
        root_experiment = experiment
        while root_experiment.parent_experiment_id:
            parent = self.db.query(Experiment).filter(
                Experiment.id == root_experiment.parent_experiment_id
            ).first()
            if parent:
                root_experiment = parent
            else:
                break

        # Traverse from root to current
        experiments_in_chain = [root_experiment]
        current_exp = root_experiment
        while True:
            child = self.db.query(Experiment).filter(
                Experiment.parent_experiment_id == current_exp.id
            ).first()
            if child:
                experiments_in_chain.append(child)
                current_exp = child
            else:
                break

        # Build context
        iteration_history = []
        error_history = []
        improvement_attempts = []
        all_scores = []
        total_training_time = 0.0
        all_logs = []
        all_critiques = []

        for exp in experiments_in_chain:
            trial = self.db.query(Trial).filter(
                Trial.experiment_id == exp.id
            ).order_by(Trial.created_at.desc()).first()

            metrics = (trial.metrics_json if trial and trial.metrics_json else {}) or {}
            score = metrics.get(primary_metric, metrics.get("score_val", 0))
            training_time = metrics.get("training_time_seconds", 0)

            if score:
                all_scores.append(float(score))
            total_training_time += float(training_time) if training_time else 0

            entry = {
                "iteration": exp.iteration_number,
                "experiment_id": str(exp.id),
                "name": exp.name,
                "score": float(score) if score else 0.0,
                "status": exp.status.value if exp.status else "unknown",
                "training_time_seconds": training_time,
                "num_models_trained": metrics.get("num_models_trained", 0),
            }

            if exp.improvement_context_json:
                changes = exp.improvement_context_json.get("summary", "")
                if changes:
                    entry["changes_made"] = changes
                    improvement_attempts.append({
                        "iteration": exp.iteration_number,
                        "changes": changes,
                        "result_score": float(score) if score else 0.0,
                        "success": exp.status == ExperimentStatus.COMPLETED,
                    })

            if exp.status == ExperimentStatus.FAILED and exp.error_message:
                entry["error"] = exp.error_message
                error_history.append({
                    "iteration": exp.iteration_number,
                    "error": exp.error_message,
                })

            if trial and trial.training_logs:
                logs = trial.training_logs
                log_lines = logs.split('\n')
                important_lines = [
                    line for line in log_lines
                    if any(kw in line.lower() for kw in ['error', 'warning', 'failed', 'skipped', 'best model'])
                ][:10]
                if important_lines:
                    all_logs.append({
                        "iteration": exp.iteration_number,
                        "key_logs": important_lines,
                    })

            if trial and trial.critique_json:
                all_critiques.append({
                    "iteration": exp.iteration_number,
                    "critique": trial.critique_json,
                })

            if trial and trial.leaderboard_json:
                leaderboard = trial.leaderboard_json[:5]
                entry["top_models"] = [
                    {"model": m.get("model", "?"), "score": m.get("score_val", 0)}
                    for m in leaderboard
                ]

            model_version = self.db.query(ModelVersion).filter(
                ModelVersion.experiment_id == exp.id
            ).order_by(ModelVersion.created_at.desc()).first()

            if model_version and model_version.feature_importances_json:
                importances = model_version.feature_importances_json
                sorted_feats = sorted(
                    importances.items(), key=lambda x: abs(x[1]), reverse=True
                )[:10]
                entry["top_features"] = [
                    {"feature": f, "importance": imp} for f, imp in sorted_feats
                ]

            iteration_history.append(entry)

        # Calculate score trend
        score_trend = "flat"
        if len(all_scores) >= 2:
            first_half_avg = sum(all_scores[:len(all_scores)//2]) / (len(all_scores)//2)
            second_half_avg = sum(all_scores[len(all_scores)//2:]) / (len(all_scores) - len(all_scores)//2)
            if second_half_avg > first_half_avg * 1.01:
                score_trend = "improving"
            elif second_half_avg < first_half_avg * 0.99:
                score_trend = "declining"

        # Collect feature engineering feedback
        feature_engineering_feedback = _collect_feature_engineering_feedback(experiments_in_chain)

        # Check for overfitting
        overfitting_report = None
        try:
            from app.services.holdout_validator import get_overfitting_report
            overfitting_report = get_overfitting_report(self.db, experiment)
        except Exception:
            pass

        return {
            "iteration_history": iteration_history,
            "error_history": error_history,
            "improvement_attempts": improvement_attempts,
            "best_score": max(all_scores) if all_scores else 0.0,
            "worst_score": min(all_scores) if all_scores else 0.0,
            "current_score": all_scores[-1] if all_scores else 0.0,
            "score_trend": score_trend,
            "total_iterations": len(experiments_in_chain),
            "total_training_time_seconds": total_training_time,
            "logs_summary": all_logs,
            "critiques": all_critiques,
            "feature_engineering_feedback": feature_engineering_feedback,
            "overfitting_report": overfitting_report,
        }

    def _load_data_statistics(
        self,
        dataset_spec: DatasetSpec,
    ) -> Optional[Dict[str, Any]]:
        """Load actual data and compute statistics."""
        try:
            from app.tasks.automl import load_dataset_from_spec
            self.logger.thinking("Loading actual dataset for analysis...")
            df = load_dataset_from_spec(self.db, dataset_spec)

            stats = {
                "columns": list(df.columns),
                "row_count": len(df),
                "column_count": len(df.columns),
                "column_stats": {},
                "sample_values": {},
                "dtypes": {},
            }

            for col in df.columns[:30]:
                col_stats = {
                    "dtype": str(df[col].dtype),
                    "null_pct": float(df[col].isna().sum() / len(df) * 100),
                    "unique": int(df[col].nunique()),
                }

                if pd.api.types.is_numeric_dtype(df[col]):
                    col_stats.update({
                        "min": float(df[col].min()) if not df[col].isna().all() else None,
                        "max": float(df[col].max()) if not df[col].isna().all() else None,
                        "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                        "std": float(df[col].std()) if not df[col].isna().all() else None,
                    })

                stats["column_stats"][col] = col_stats
                stats["dtypes"][col] = str(df[col].dtype)

            if len(df) > 0:
                first_row = df.iloc[0]
                for col in df.columns[:15]:
                    stats["sample_values"][col] = str(first_row[col])[:50]

            return stats

        except Exception as e:
            self.logger.warning(f"Could not load dataset for analysis: {e}")
            return None

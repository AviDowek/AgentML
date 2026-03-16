"""Improvement Analysis Agent - Analyzes what to improve based on results.

This agent is used in the simple improvement pipeline to analyze
experiment results and identify what should be improved.
"""

from typing import Any, Dict, List, Optional

from app.models import (
    AgentStepType,
    DataSource,
    DatasetSpec,
    Experiment,
    ModelVersion,
    Trial,
)
from app.services.agents.base import BaseAgent
from app.services.prompts import (
    SYSTEM_ROLE_IMPROVEMENT_ANALYST,
    get_improvement_analysis_prompt,
)


class ImprovementAnalysisAgent(BaseAgent):
    """Analyzes experiment results for the simple improvement pipeline.

    Input JSON:
        - experiment_id: UUID of the experiment to analyze
        - experiment_name: Name of the experiment
        - iteration_number: Current iteration number

    Output:
        - improvement_summary: Summary of what to improve
        - key_insights: List of key insights
        - recommended_changes: List of recommended changes
        - priority_areas: Priority areas for improvement
    """

    name = "improvement_analysis"
    step_type = AgentStepType.IMPROVEMENT_ANALYSIS

    async def execute(self) -> Dict[str, Any]:
        """Execute improvement analysis."""
        experiment_id = self.require_input("experiment_id")
        experiment_name = self.get_input("experiment_name", "Unknown")
        iteration_number = self.get_input("iteration_number", 0)

        self.logger.info(f"Analyzing experiment {experiment_name} for improvements...")

        # Load experiment
        experiment = self.db.query(Experiment).filter(
            Experiment.id == experiment_id
        ).first()
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Load dataset spec
        dataset_spec = experiment.dataset_spec
        if not dataset_spec:
            raise ValueError("Experiment has no dataset spec")

        # Get latest trial
        trial = self.db.query(Trial).filter(
            Trial.experiment_id == experiment.id
        ).order_by(Trial.created_at.desc()).first()

        if not trial:
            raise ValueError("No trial found for experiment")

        # Gather context
        feature_columns = dataset_spec.feature_columns or []
        target_column = dataset_spec.target_column or ""
        task_type = experiment.experiment_plan_json.get("task_type", "binary") if experiment.experiment_plan_json else "binary"
        primary_metric = experiment.primary_metric or "accuracy"

        # Get metrics
        metrics = trial.metrics_json or {}
        best_score = metrics.get(primary_metric, metrics.get("score_val", 0))
        training_time = metrics.get("training_time_seconds", 0)
        num_models = metrics.get("num_models_trained", 0)

        # Get leaderboard summary
        leaderboard = trial.leaderboard_json or []
        leaderboard_lines = []
        for model in leaderboard[:10]:
            name = model.get("model", "Unknown")
            score = model.get("score_val", 0)
            fit_time = model.get("fit_time", 0)
            leaderboard_lines.append(f"  - {name}: score={score:.4f}, fit_time={fit_time:.1f}s")
        leaderboard_summary = "\n".join(leaderboard_lines) if leaderboard_lines else "(no leaderboard data)"

        # Get feature importances
        feature_importances = {}
        model_version = self.db.query(ModelVersion).filter(
            ModelVersion.experiment_id == experiment.id
        ).order_by(ModelVersion.created_at.desc()).first()
        if model_version and model_version.feature_importances_json:
            feature_importances = model_version.feature_importances_json

        # Get training logs
        training_logs = trial.training_logs or "(no logs captured)"

        # Get critique
        critique_json = trial.critique_json

        # Get previous improvement contexts
        previous_improvements = []
        if experiment.parent_experiment_id:
            current = experiment
            while current.parent_experiment_id:
                parent = self.db.query(Experiment).filter(
                    Experiment.id == current.parent_experiment_id
                ).first()
                if parent and parent.improvement_context_json:
                    previous_improvements.insert(0, parent.improvement_context_json)
                current = parent
                if not current:
                    break

        # Load actual data statistics
        data_statistics = self._load_data_statistics(dataset_spec)

        # Gather iteration history
        iteration_history, error_history = self._gather_iteration_history(
            experiment, primary_metric
        )

        # Dataset shape
        if data_statistics:
            row_count = data_statistics["row_count"]
            col_count = len(data_statistics["columns"])
        else:
            row_count = dataset_spec.spec_json.get("row_count", 0) if dataset_spec.spec_json else 0
            col_count = len(feature_columns)
        dataset_shape = f"{row_count} rows x {col_count} features"

        # Generate analysis prompt
        analysis_prompt = get_improvement_analysis_prompt(
            experiment_name=experiment_name,
            iteration_number=iteration_number,
            task_type=task_type,
            target_column=target_column,
            primary_metric=primary_metric,
            best_score=float(best_score) if best_score else 0.0,
            training_time_seconds=float(training_time) if training_time else 0.0,
            num_models_trained=int(num_models) if num_models else 0,
            dataset_shape=dataset_shape,
            feature_columns=feature_columns,
            leaderboard_summary=leaderboard_summary,
            training_logs=training_logs,
            feature_importances=feature_importances,
            critique_json=critique_json,
            previous_improvements=previous_improvements,
            data_statistics=data_statistics,
            iteration_history=iteration_history if iteration_history else None,
            error_history=error_history if error_history else None,
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_IMPROVEMENT_ANALYST},
            {"role": "user", "content": analysis_prompt},
        ]

        self.logger.action("Consulting LLM for improvement analysis...")
        response = await self.llm.chat_json(messages, None)

        self.logger.summary(
            f"Improvement analysis complete: {response.get('improvement_summary', '')[:100]}"
        )

        return response

    def _load_data_statistics(self, dataset_spec: DatasetSpec) -> Optional[Dict[str, Any]]:
        """Load actual data and compute statistics."""
        try:
            from app.tasks.automl import load_dataset_from_spec
            self.logger.thinking("Loading actual dataset for analysis...")
            df = load_dataset_from_spec(self.db, dataset_spec)

            stats = {
                "columns": list(df.columns),
                "row_count": len(df),
                "column_stats": {},
                "sample_values": {},
            }

            for col in df.columns[:30]:
                col_stats = {
                    "dtype": str(df[col].dtype),
                    "null_pct": float(df[col].isna().sum() / len(df) * 100),
                    "unique": int(df[col].nunique()),
                }
                stats["column_stats"][col] = col_stats

            if len(df) > 0:
                first_row = df.iloc[0]
                for col in df.columns[:15]:
                    stats["sample_values"][col] = str(first_row[col])

            return stats
        except Exception as e:
            self.logger.warning(f"Could not load dataset for analysis: {e}")
            return None

    def _gather_iteration_history(
        self, experiment: Experiment, primary_metric: str
    ) -> tuple:
        """Gather iteration history from experiment chain."""
        iteration_history = []
        error_history = []

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

        # Build history
        from app.models.experiment import ExperimentStatus
        for exp in experiments_in_chain:
            exp_trial = self.db.query(Trial).filter(
                Trial.experiment_id == exp.id
            ).order_by(Trial.created_at.desc()).first()
            exp_metrics = exp_trial.metrics_json if exp_trial else {}
            exp_score = exp_metrics.get(primary_metric, exp_metrics.get("score_val", 0))

            history_entry = {
                "iteration": exp.iteration_number,
                "score": float(exp_score) if exp_score else 0.0,
                "status": exp.status.value if exp.status else "unknown",
                "changes_made": exp.improvement_context_json.get("summary", "") if exp.improvement_context_json else "",
            }

            if exp.status == ExperimentStatus.FAILED and exp.error_message:
                history_entry["error"] = exp.error_message
                error_history.append(f"Iteration {exp.iteration_number}: {exp.error_message}")

            iteration_history.append(history_entry)

        return iteration_history, error_history

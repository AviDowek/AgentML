"""Results Interpretation Agent - Analyzes experiment results.

This agent analyzes experiment results and provides a summary with recommendations
for which model to deploy.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models import Experiment, ModelVersion, Trial
from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.prompts import (
    SYSTEM_ROLE_ML_ANALYST,
    get_results_interpretation_prompt,
)


def _infer_metric_from_task(task_type: str) -> str:
    """Infer primary metric from task type."""
    if task_type == "binary":
        return "roc_auc"
    elif task_type == "multiclass":
        return "accuracy"
    elif task_type == "regression":
        return "rmse"
    return "unknown"


class ResultsRecommendation(BaseModel):
    """Recommendation for which model to use."""
    recommended_model_id: str = Field(description="UUID of the recommended model")
    reason: str = Field(description="Why this model is recommended")


class ResultsInterpretationResponse(BaseModel):
    """Response schema for results interpretation."""
    results_summary: str = Field(description="Summary of the experiment results")
    recommendation: ResultsRecommendation = Field(description="Model recommendation")
    natural_language_summary: str = Field(description="Comprehensive summary for end users")


class ResultsInterpretationAgent(BaseAgent):
    """Analyzes experiment results and provides recommendations.

    Input JSON:
        - experiment_id: UUID of the experiment to interpret

    Output:
        - results_summary: Summary of experiment results
        - recommendation: Model recommendation with reason
        - natural_language_summary: Summary for end users
        - leaderboard: Sorted list of models with metrics
        - trial_summaries: Summary of each trial
    """

    name = "results_interpretation"
    step_type = AgentStepType.RESULTS_INTERPRETATION

    async def execute(self) -> Dict[str, Any]:
        """Execute results interpretation."""
        experiment_id = self.require_input("experiment_id")

        self.logger.info(f"Loading experiment {experiment_id} for interpretation...")

        # Load experiment with trials and model versions
        experiment = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        self.logger.info(f"Experiment: {experiment.name or 'Unnamed'} (status: {experiment.status})")

        # Load trials
        trials = self.db.query(Trial).filter(Trial.experiment_id == experiment_id).all()
        self.logger.thought(f"Found {len(trials)} trial(s)")

        # Load model versions
        model_versions = self.db.query(ModelVersion).filter(ModelVersion.experiment_id == experiment_id).all()
        self.logger.thought(f"Found {len(model_versions)} model version(s)")

        # Build leaderboard data
        leaderboard_data = self._build_leaderboard(model_versions, experiment.primary_metric)

        self.logger.info(f"Built leaderboard with {len(leaderboard_data)} models")

        if leaderboard_data:
            top_model = leaderboard_data[0]
            metric_value = top_model['metrics'].get(experiment.primary_metric, 'N/A')
            self.logger.thought(f"Top model: {top_model['model_name']} with {experiment.primary_metric}={metric_value}")

        # Build trial summaries
        trial_summaries = self._build_trial_summaries(trials)

        # Get dataset info
        dataset_info = self._get_dataset_info(experiment)

        # Get task type
        task_type = 'unknown'
        if experiment.experiment_plan_json:
            task_type = experiment.experiment_plan_json.get('task_type', 'unknown')

        # Build prompt
        prompt = get_results_interpretation_prompt(
            experiment_name=experiment.name or 'Unnamed',
            task_type=task_type,
            primary_metric=experiment.primary_metric or _infer_metric_from_task(task_type),
            status=str(experiment.status),
            dataset_info=dataset_info,
            trial_summaries=str(trial_summaries),
            leaderboard_data=str(leaderboard_data),
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_ML_ANALYST},
            {"role": "user", "content": prompt},
        ]

        self.logger.info("Consulting LLM for results interpretation...")
        response = await self.llm.chat_json(messages, ResultsInterpretationResponse)

        recommended_model = response.get('recommendation', {}).get('recommended_model_id', 'N/A')
        self.logger.summary(f"Analysis complete. Recommended model: {recommended_model}")

        return {
            "results_summary": response.get("results_summary", ""),
            "recommendation": response.get("recommendation", {}),
            "natural_language_summary": response.get("natural_language_summary", ""),
            "leaderboard": leaderboard_data,
            "trial_summaries": trial_summaries,
        }

    def _build_leaderboard(self, model_versions: List[ModelVersion], primary_metric: Optional[str]) -> List[Dict]:
        """Build sorted leaderboard from model versions."""
        # Error metrics that are stored as negative values internally
        error_metrics = {"root_mean_squared_error", "mean_squared_error", "mean_absolute_error", "rmse", "mse", "mae"}

        leaderboard_data = []
        for mv in model_versions:
            raw_metrics = mv.metrics_json or {}

            # Normalize negative error metrics to positive
            metrics = {}
            for key, value in raw_metrics.items():
                if isinstance(value, (int, float)):
                    if key.lower() in error_metrics and value < 0:
                        metrics[key] = abs(value)
                    else:
                        metrics[key] = value
                else:
                    metrics[key] = value

            leaderboard_data.append({
                "model_id": str(mv.id),
                "model_name": mv.name,
                "model_type": mv.model_type,
                "metrics": metrics,
                "trial_id": str(mv.trial_id) if mv.trial_id else None,
            })

        # Sort by primary metric if available
        if primary_metric:
            def get_metric_value(item):
                return item["metrics"].get(primary_metric, 0) or 0
            leaderboard_data.sort(key=get_metric_value, reverse=True)

        return leaderboard_data

    def _build_trial_summaries(self, trials: List[Trial]) -> List[Dict]:
        """Build trial summaries."""
        summaries = []
        for trial in trials:
            summaries.append({
                "trial_id": str(trial.id),
                "variant_name": trial.variant_name,
                "status": trial.status,
                "metrics": trial.metrics_json or {},
                "best_model_ref": trial.best_model_ref,
            })
        return summaries

    def _get_dataset_info(self, experiment: Experiment) -> str:
        """Get dataset info from experiment."""
        if not experiment.dataset_spec:
            return ""

        ds = experiment.dataset_spec
        feature_count = len(ds.feature_columns) if ds.feature_columns else 0

        features_preview = ""
        if ds.feature_columns:
            features_preview = ', '.join(ds.feature_columns[:10])
            if len(ds.feature_columns) > 10:
                features_preview += '...'

        return f"""
Dataset Configuration:
- Target Column: {ds.target_column or 'unknown'}
- Feature Columns: {feature_count} features
- Features: {features_preview or 'N/A'}
"""

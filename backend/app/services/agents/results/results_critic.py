"""Results Critic Agent - Reviews results for issues.

This agent reviews experiment results for potential issues like overfitting,
data leakage, or other problems.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models import Experiment, ModelVersion
from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.prompts import (
    SYSTEM_ROLE_MODEL_REVIEWER,
    get_results_critic_prompt,
)


class CriticIssue(BaseModel):
    """An issue found during critique."""
    issue: str = Field(description="Description of the issue")
    severity: str = Field(description="Severity: critical, warning, or info")
    recommendation: str = Field(description="Recommended action")


class CriticFindings(BaseModel):
    """Overall findings from the critique."""
    severity: str = Field(description="Overall severity: critical, warning, or ok")
    issues: List[CriticIssue] = Field(description="List of identified issues")
    approved: bool = Field(description="Whether results are approved for production")


class ResultsCriticResponse(BaseModel):
    """Response schema for results critique."""
    critic_findings: CriticFindings = Field(description="Detailed findings")
    natural_language_summary: str = Field(description="Summary for end users")


class ResultsCriticAgent(BaseAgent):
    """Reviews experiment results for potential issues.

    Input JSON:
        - experiment_id: UUID of the experiment to critique
        - results_interpretation: Output from results interpretation step (optional)

    Output:
        - critic_findings: Detailed findings with severity and issues
        - natural_language_summary: Summary for end users
    """

    name = "results_critic"
    step_type = AgentStepType.RESULTS_CRITIC

    async def execute(self) -> Dict[str, Any]:
        """Execute results critique."""
        experiment_id = self.require_input("experiment_id")
        results_interpretation = self.get_input("results_interpretation", {})

        self.logger.info(f"Loading experiment {experiment_id} for critique...")

        # Load experiment
        experiment = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Load dataset spec for row count
        row_count, feature_count = self._get_dataset_info(experiment)
        self.logger.thought(f"Dataset: {row_count or 'unknown'} rows, {feature_count or 'unknown'} features")

        # Load model versions with metrics
        model_versions = self.db.query(ModelVersion).filter(ModelVersion.experiment_id == experiment_id).all()
        self.logger.info(f"Analyzing {len(model_versions)} model(s) for potential issues...")

        # Build model details
        model_details = self._build_model_details(model_versions)

        # Check for obvious issues programmatically
        issues_found, warnings_found = self._check_obvious_issues(
            model_details, row_count, feature_count
        )

        # Build prompt
        prompt = get_results_critic_prompt(
            experiment_name=experiment.name or 'Unnamed',
            primary_metric=experiment.primary_metric or 'unknown',
            status=experiment.status,
            row_count=row_count,
            feature_count=feature_count,
            model_details=model_details,
            issues_found=issues_found,
            warnings_found=warnings_found,
            results_summary=results_interpretation.get('results_summary', 'Not available'),
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_MODEL_REVIEWER},
            {"role": "user", "content": prompt},
        ]

        self.logger.info("Consulting LLM for results critique...")
        response = await self.llm.chat_json(messages, ResultsCriticResponse)

        findings = response.get("critic_findings", {})
        severity = findings.get("severity", "ok")
        approved = findings.get("approved", True)
        issue_count = len(findings.get("issues", []))

        if severity == "critical":
            self.logger.error(f"Critical issues found: {issue_count} issues, NOT approved")
        elif severity == "warning":
            self.logger.warning(f"Warnings found: {issue_count} issues, approved={approved}")
        else:
            self.logger.summary(f"Critique complete: {severity} severity, {issue_count} issues, approved={approved}")

        return {
            "critic_findings": findings,
            "natural_language_summary": response.get("natural_language_summary", ""),
        }

    def _get_dataset_info(self, experiment: Experiment) -> tuple:
        """Get row count and feature count from experiment."""
        row_count = None
        feature_count = None

        if experiment.dataset_spec:
            ds = experiment.dataset_spec
            if hasattr(ds, 'row_count'):
                row_count = ds.row_count
            feature_count = len(ds.feature_columns) if ds.feature_columns else None

        return row_count, feature_count

    def _build_model_details(self, model_versions: List[ModelVersion]) -> List[Dict]:
        """Build model details for LLM analysis."""
        model_details = []
        for mv in model_versions:
            metrics = mv.metrics_json or {}
            feature_importances = mv.feature_importances_json or {}
            model_details.append({
                "model_id": str(mv.id),
                "model_name": mv.name,
                "model_type": mv.model_type,
                "metrics": metrics,
                "feature_importances": feature_importances,
            })
        return model_details

    def _check_obvious_issues(
        self, model_details: List[Dict], row_count: Optional[int], feature_count: Optional[int]
    ) -> tuple:
        """Check for obvious issues programmatically."""
        issues_found = []
        warnings_found = []

        # Check for suspiciously perfect metrics
        for md in model_details:
            metrics = md["metrics"]
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    if metric_value == 1.0 and metric_name in ["accuracy", "auc", "f1", "r2"]:
                        issues_found.append(
                            f"Model '{md['model_name']}' has perfect {metric_name}=1.0 - "
                            "possible data leakage or overfitting"
                        )
                        self.logger.warning(f"Suspicious: {md['model_name']} has {metric_name}=1.0")

        # Check for data size issues
        if row_count and row_count < 100:
            warnings_found.append(
                f"Small dataset ({row_count} rows) - results may not generalize well"
            )
            self.logger.warning(f"Small dataset: {row_count} rows")

        if feature_count and row_count and feature_count > row_count / 10:
            warnings_found.append(
                f"High feature-to-sample ratio ({feature_count} features, {row_count} rows) - "
                "risk of overfitting"
            )
            self.logger.warning(f"High feature ratio: {feature_count} features for {row_count} rows")

        return issues_found, warnings_found

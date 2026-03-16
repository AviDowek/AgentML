"""Improvement Plan Agent - Creates actionable improvement plan.

This agent is used in the simple improvement pipeline to create
a concrete plan based on the improvement analysis.
"""

from typing import Any, Dict, List, Optional

from app.models import AgentStepType, DataSource, DatasetSpec
from app.services.agents.base import BaseAgent
from app.services.prompts import (
    SYSTEM_ROLE_IMPROVEMENT_ANALYST,
    get_improvement_plan_prompt,
)


class ImprovementPlanAgent(BaseAgent):
    """Creates improvement plan for the simple improvement pipeline.

    Input JSON:
        - improvement_analysis: Output from ImprovementAnalysisAgent
        - experiment_name: Name of the experiment
        - iteration_number: Current iteration number
        - task_type: ML task type
        - target_column: Target column name
        - current_features: Current feature columns
        - current_experiment_plan: Current experiment plan JSON
        - dataset_spec_id: Dataset spec ID for getting raw columns

    Output:
        - plan_summary: Summary of the improvement plan
        - feature_changes: Dict with features_to_keep and engineered_features
        - automl_config: Updated AutoML configuration
        - validation_strategy: Validation strategy
    """

    name = "improvement_plan"
    step_type = AgentStepType.IMPROVEMENT_PLAN

    async def execute(self) -> Dict[str, Any]:
        """Execute improvement plan generation."""
        improvement_analysis = self.require_input("improvement_analysis")
        experiment_name = self.get_input("experiment_name", "Unknown")
        iteration_number = self.get_input("iteration_number", 0)
        task_type = self.get_input("task_type", "binary")
        target_column = self.get_input("target_column", "")
        current_features = self.get_input("current_features", [])
        current_experiment_plan = self.get_input("current_experiment_plan", {})
        dataset_spec_id = self.get_input("dataset_spec_id")

        self.logger.info(f"Creating improvement plan for {experiment_name}...")

        # Get raw columns from dataset spec
        raw_columns = []
        existing_engineered_features = []

        if dataset_spec_id:
            dataset_spec = self.db.query(DatasetSpec).filter(
                DatasetSpec.id == dataset_spec_id
            ).first()

            if dataset_spec:
                # Try to get columns from data statistics or data sources
                raw_columns = self._get_raw_columns(dataset_spec)

                # Get existing engineered features
                spec_json = dataset_spec.spec_json or {}
                existing_engineered_features = spec_json.get("engineered_features", [])

        self.logger.thinking(f"Using {len(raw_columns)} raw columns for feature engineering")

        # Generate plan prompt
        plan_prompt = get_improvement_plan_prompt(
            experiment_name=experiment_name,
            iteration_number=iteration_number,
            task_type=task_type,
            target_column=target_column,
            current_features=current_features,
            improvement_analysis=improvement_analysis,
            current_experiment_plan=current_experiment_plan,
            raw_columns=raw_columns if raw_columns else None,
            existing_engineered_features=existing_engineered_features if existing_engineered_features else None,
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_IMPROVEMENT_ANALYST},
            {"role": "user", "content": plan_prompt},
        ]

        self.logger.action("Consulting LLM for improvement plan...")
        response = await self.llm.chat_json(messages, None)

        # Log plan summary
        plan_summary = response.get("plan_summary", "")
        self.logger.summary(f"Improvement plan created: {plan_summary[:100]}")

        feature_changes = response.get("feature_changes", {})
        new_features = feature_changes.get("engineered_features", [])
        if new_features:
            self.logger.info(f"Plan includes {len(new_features)} new engineered features")

        return response

    def _get_raw_columns(self, dataset_spec: DatasetSpec) -> List[str]:
        """Get raw columns from dataset spec or data sources."""
        raw_columns = []

        # First try to load actual data columns
        try:
            from app.tasks.automl import load_dataset_from_spec
            df = load_dataset_from_spec(self.db, dataset_spec)
            raw_columns = list(df.columns)
            self.logger.thinking(f"Loaded {len(raw_columns)} columns from actual data")
            return raw_columns
        except Exception:
            pass

        # Fallback to data source schema
        data_sources_json = dataset_spec.data_sources_json or []
        if data_sources_json:
            for ds_info in data_sources_json:
                if isinstance(ds_info, str):
                    ds_id = ds_info
                elif isinstance(ds_info, dict):
                    ds_id = ds_info.get("data_source_id")
                else:
                    continue

                if ds_id:
                    data_source = self.db.query(DataSource).filter(
                        DataSource.id == ds_id
                    ).first()
                    if data_source and data_source.schema_summary:
                        cols = data_source.schema_summary.get("columns", [])
                        if isinstance(cols, list):
                            raw_columns.extend([
                                c.get("name") if isinstance(c, dict) else str(c)
                                for c in cols
                            ])
                        elif isinstance(cols, dict):
                            raw_columns.extend(cols.keys())

        return raw_columns

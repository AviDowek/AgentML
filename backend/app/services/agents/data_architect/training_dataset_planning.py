"""Training Dataset Planning Agent - Plans how to build a training dataset.

This agent uses relationship discovery results to propose a TrainingDatasetSpec -
a complete plan for building a training dataset from multiple tables.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.prompts import (
    SYSTEM_ROLE_DATASET_DESIGNER,
    get_training_dataset_planning_prompt,
)


# LLM Response Schemas
class LLMBaseFilter(BaseModel):
    column: str = Field(..., description="Column to filter on")
    operator: str = Field(..., description="Filter operator")
    value: Any = Field(None, description="Value to compare against")


class LLMTargetDefinition(BaseModel):
    table: str = Field(..., description="Table containing the target")
    column: str = Field(..., description="Column to predict")
    join_key: Optional[str] = Field(None, description="Join key if target in different table")
    label_window_days: Optional[int] = Field(None, description="Days forward for time-based targets")


class LLMAggFeature(BaseModel):
    name: str = Field(..., description="Feature name")
    agg: str = Field(..., description="Aggregation: sum, count, avg, min, max")
    column: str = Field(..., description="Column to aggregate")


class LLMJoinAggregation(BaseModel):
    window_days: Optional[int] = Field(None, description="Time window in days")
    features: List[LLMAggFeature] = Field(default_factory=list)


class LLMJoinPlanItem(BaseModel):
    from_table: str = Field(..., description="Source table")
    to_table: str = Field(..., description="Target table")
    left_key: str = Field(..., description="Key in source table")
    right_key: str = Field(..., description="Key in target table")
    relationship: str = Field(..., description="one_to_one, one_to_many, many_to_one")
    aggregation: Optional[LLMJoinAggregation] = Field(None)


class LLMTrainingDatasetSpec(BaseModel):
    base_table: str = Field(..., description="Base table name")
    base_filters: List[LLMBaseFilter] = Field(default_factory=list)
    target_definition: LLMTargetDefinition
    join_plan: List[LLMJoinPlanItem] = Field(default_factory=list)
    excluded_tables: List[str] = Field(default_factory=list)
    excluded_columns: List[str] = Field(default_factory=list)


class LLMPlanningResponse(BaseModel):
    training_dataset_spec: LLMTrainingDatasetSpec
    natural_language_summary: str = Field(..., description="Explanation of the plan")


class TrainingDatasetPlanningAgent(BaseAgent):
    """Plans training dataset construction from multiple tables.

    Input JSON:
        - project_description: User's ML goal description
        - target_hint: Optional hint about target column
        - data_source_profiles: List of data source profiles
        - relationships_summary: Output from relationship discovery

    Output:
        - training_dataset_spec: The training dataset specification
        - natural_language_summary: Explanation for user
    """

    name = "training_dataset_planning"
    step_type = AgentStepType.TRAINING_DATASET_PLANNING

    async def execute(self) -> Dict[str, Any]:
        """Execute training dataset planning."""
        project_description = self.require_input("project_description")
        target_hint = self.get_input("target_hint")
        data_source_profiles = self.require_input("data_source_profiles")
        relationships_summary = self.require_input("relationships_summary")

        self.logger.info(f"Planning training dataset for: {project_description[:100]}...")

        # Extract key information
        tables = relationships_summary.get("tables", [])
        relationships = relationships_summary.get("relationships", [])
        base_table_candidates = relationships_summary.get("base_table_candidates", [])

        self.logger.info(f"Evaluating {len(tables)} tables as potential base tables...")

        # Log base table candidates
        for candidate in base_table_candidates[:5]:
            table_name = candidate.get("table", "unknown")
            score = candidate.get("score", 0)
            reasons = candidate.get("reasons", [])
            self.logger.thought(
                f"Considering {table_name} as base table (score: {score:.2f}): "
                f"{', '.join(reasons[:2]) if reasons else 'no specific reasons'}"
            )

        # Build summaries for prompt
        table_summaries = self._build_table_summaries(tables)
        relationship_summaries = self._build_relationship_summaries(relationships)

        self.logger.info(f"Found {len(relationships)} relationship(s) between tables")

        # Build prompt
        base_table_formatted = str([{
            "table": c.get("table"),
            "score": c.get("score"),
            "reasons": c.get("reasons", []),
            "target_columns": c.get("target_columns", []),
        } for c in base_table_candidates[:5]])

        prompt = get_training_dataset_planning_prompt(
            project_description=project_description,
            target_hint=target_hint or "",
            table_summaries=str(table_summaries),
            base_table_candidates=base_table_formatted,
            relationship_summaries=str(relationship_summaries),
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_DATASET_DESIGNER},
            {"role": "user", "content": prompt},
        ]

        # Retry logic
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Consulting LLM for training dataset plan (attempt {attempt + 1}/{max_retries})...")

                if last_error and attempt > 0:
                    retry_messages = messages.copy()
                    retry_messages.extend([
                        {"role": "assistant", "content": "I'll generate the training dataset specification."},
                        {"role": "user", "content": f"Your previous response caused an error: {last_error}\n\nPlease fix the issue."}
                    ])
                    response = await self.llm.chat_json(retry_messages, LLMPlanningResponse)
                else:
                    response = await self.llm.chat_json(messages, LLMPlanningResponse)

                spec_data = response.get("training_dataset_spec", {})
                if isinstance(spec_data, str):
                    raise ValueError(f"training_dataset_spec is string, expected dict: {spec_data[:100]}")

                natural_language_summary = response.get("natural_language_summary", "")

                # Log what was decided
                base_table = spec_data.get("base_table", "unknown") if isinstance(spec_data, dict) else "unknown"
                target_def = spec_data.get("target_definition", {}) if isinstance(spec_data, dict) else {}
                join_plan = spec_data.get("join_plan", []) if isinstance(spec_data, dict) else []

                self.logger.thought(f"Selected base table: {base_table}")
                if isinstance(target_def, dict):
                    self.logger.thought(f"Target: {target_def.get('table', '')}.{target_def.get('column', '')}")

                if join_plan and isinstance(join_plan, list):
                    self.logger.info(f"Join plan includes {len(join_plan)} table(s)")

                self.logger.summary(
                    f"Training dataset plan complete. Base table: {base_table}. "
                    f"Joins: {len(join_plan) if isinstance(join_plan, list) else 0}."
                )

                return {
                    "training_dataset_spec": spec_data,
                    "natural_language_summary": natural_language_summary,
                }

            except Exception as e:
                last_error = str(e)
                self.logger.thought(f"Attempt {attempt + 1} failed: {last_error}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed: {last_error}")
                    raise ValueError(f"Training dataset planning failed: {last_error}")

    def _build_table_summaries(self, tables: List[Dict]) -> List[Dict]:
        """Build table summaries for prompt."""
        summaries = []
        for table in tables:
            summaries.append({
                "name": table.get("table_name", ""),
                "source_name": table.get("source_name", ""),
                "row_count": table.get("row_count", 0),
                "column_count": table.get("column_count", 0),
                "id_columns": [c.get("name") for c in table.get("id_columns", [])],
                "target_columns": [c.get("name") for c in table.get("target_columns", [])],
                "has_obvious_id": table.get("has_obvious_id", False),
                "has_potential_target": table.get("has_potential_target", False),
            })
        return summaries

    def _build_relationship_summaries(self, relationships: List[Dict]) -> List[Dict]:
        """Build relationship summaries for prompt."""
        summaries = []
        for rel in relationships:
            summaries.append({
                "from_table": rel.get("from_table", ""),
                "to_table": rel.get("to_table", ""),
                "from_column": rel.get("from_column", ""),
                "to_column": rel.get("to_column", ""),
                "cardinality": rel.get("cardinality", "unknown"),
            })
        return summaries

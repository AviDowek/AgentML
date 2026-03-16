"""Data Analysis Agent - First step in the setup pipeline.

This agent analyzes uploaded data and provides:
- Assessment of data quality and suitability for the ML task
- Recommendations for data preparation
- Option to search for additional datasets if the current data is insufficient

The agent can use tools to fetch context it needs, making it self-sufficient
regardless of how it's orchestrated (PM mode or sequential mode).
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models import DataSource
from app.models import AgentStepType
from app.schemas.agent import SchemaSummary
from app.services.agents.base import BaseAgent
from app.services.agent_service import build_schema_summary, _format_schema_for_prompt
from app.services.prompts import (
    SYSTEM_ROLE_DATA_EVALUATOR,
    get_data_analysis_prompt,
)

logger = logging.getLogger(__name__)


class DataAnalysisResponse(BaseModel):
    """Response schema for data analysis."""
    suitability_score: float = Field(
        description="Score from 0.0 to 1.0 indicating how suitable the data is for the ML task"
    )
    can_proceed: bool = Field(
        description="Whether the data is sufficient to proceed with ML experiments"
    )
    suggest_more_data: bool = Field(
        description="Whether to suggest searching for additional datasets"
    )
    target_column_suggestion: str = Field(
        description="Best guess for the target column based on description and schema"
    )
    task_type_suggestion: str = Field(
        description="Suggested task type: binary_classification, multiclass_classification, or regression"
    )
    key_observations: List[str] = Field(
        description="Key observations about the data quality and structure"
    )
    data_preparation_recommendations: List[str] = Field(
        description="Specific recommendations for preparing this data for ML"
    )
    limitations: List[str] = Field(
        description="Limitations or concerns about using this data"
    )
    natural_language_summary: str = Field(
        description="A comprehensive summary explaining the analysis in plain language for the user"
    )


class DataAnalysisAgent(BaseAgent):
    """Analyzes uploaded data and assesses its suitability for ML tasks.

    This is the first interactive step in the pipeline. It examines the
    data quality, identifies potential issues, and provides recommendations.

    The agent uses tools to fetch context if not provided in input:
        - get_user_goal: Get the user's ML goal description
        - get_data_source_info: Get data source schema and details

    Input JSON (all optional - agent will fetch via tools if missing):
        - description: User's goal description
        - data_source_id: UUID of the data source
        - schema_summary: Pre-built schema summary

    Output:
        - suitability_score: 0.0-1.0 rating
        - can_proceed: Boolean approval
        - suggest_more_data: Whether to suggest finding more data
        - target_column_suggestion: Best target guess
        - task_type_suggestion: binary/multiclass/regression
        - key_observations: List of observations
        - data_preparation_recommendations: List of recommendations
        - limitations: List of concerns
        - natural_language_summary: User-friendly summary
        - schema_summary: Passed through for next steps
        - issues: Data quality issues found
        - high_null_columns: Columns with >30% nulls
        - potential_id_columns: Likely ID columns
        - constant_columns: Single-value columns
    """

    name = "data_analysis"
    step_type = AgentStepType.DATA_ANALYSIS
    uses_tools = True  # Enable tool calling for self-sufficient context fetching

    async def execute(self) -> Dict[str, Any]:
        """Execute data analysis on the provided dataset."""
        # Get description - try input first, then use tool
        description = await self._get_description()
        self.logger.action(f"Analyzing data for: {description[:100]}...")

        # Get or build schema summary
        schema_summary = await self._get_schema_summary()

        self.logger.thinking(f"Examining dataset: {schema_summary.data_source_name}")
        self.logger.thinking(f"Dataset size: {schema_summary.row_count} rows, {schema_summary.column_count} columns")

        # Analyze data quality
        issues, high_null_cols, id_cols, constant_cols, recommendations = self._analyze_data_quality(schema_summary)

        self.logger.thinking(f"Found {len(issues)} potential issues in the data")

        # Get LLM analysis
        response = await self._get_llm_analysis(
            description, schema_summary, issues, high_null_cols, id_cols, constant_cols
        )

        # Extract and log results
        suitability_score = response.get("suitability_score", 0.5)
        can_proceed = response.get("can_proceed", True)
        suggest_more_data = response.get("suggest_more_data", False)

        self.logger.hypothesis(f"Suitability assessment: {suitability_score:.0%} suitable for ML task")

        if response.get("target_column_suggestion"):
            self.logger.hypothesis(f"Suggested target column: {response.get('target_column_suggestion')}")

        if response.get("task_type_suggestion"):
            self.logger.hypothesis(f"Suggested task type: {response.get('task_type_suggestion')}")

        if can_proceed:
            self.logger.summary(f"Data analysis complete. Score: {suitability_score:.0%}. Ready to proceed with ML experiments.")
        else:
            self.logger.warning(f"Data analysis complete. Score: {suitability_score:.0%}. Data may need improvement before proceeding.")

        return {
            "suitability_score": suitability_score,
            "can_proceed": can_proceed,
            "suggest_more_data": suggest_more_data,
            "target_column_suggestion": response.get("target_column_suggestion", ""),
            "task_type_suggestion": response.get("task_type_suggestion", ""),
            "key_observations": response.get("key_observations", []),
            "data_preparation_recommendations": response.get("data_preparation_recommendations", []),
            "limitations": response.get("limitations", []),
            "natural_language_summary": response.get("natural_language_summary", ""),
            # Pass through useful data for next steps
            "schema_summary": schema_summary.model_dump(mode="json"),
            "issues": issues,
            "high_null_columns": [c["name"] for c in high_null_cols],
            "potential_id_columns": id_cols,
            "constant_columns": constant_cols,
        }

    async def _get_description(self) -> str:
        """Get the user's goal description.

        Tries in order:
        1. Input JSON (description key)
        2. Tool call to get_user_goal

        Returns:
            The user's goal description

        Raises:
            ValueError: If description cannot be found
        """
        # Try input first
        description = self.get_input("description")
        if description:
            logger.info("Using description from input_json")
            return description

        # Fall back to tool call
        self.logger.thinking("Description not in input, fetching via tool...")
        if self._tool_executor:
            result = self._tool_executor.execute_tool("get_user_goal", {})
            description = result.get("goal_description", "")
            if description:
                logger.info("Got description from tool call")
                return description

        raise ValueError(
            "Could not find description. "
            "Ensure the pipeline was created with a description or the project has a description."
        )

    async def _get_schema_summary(self) -> SchemaSummary:
        """Get or build the schema summary for the data source.

        Tries in order:
        1. Input JSON (schema_summary key)
        2. Input JSON (data_source_id key) -> load from DB
        3. Tool call to get_data_source_info
        """
        # Try schema from input
        schema_data = self.get_input("schema_summary")
        if schema_data:
            self.logger.thinking(f"Using provided schema: {schema_data.get('data_source_name', 'unknown')}")
            return SchemaSummary(**schema_data)

        # Try loading from data source ID in input
        data_source_id = self.get_input("data_source_id")
        if data_source_id:
            self.logger.action("Loading data source schema from ID...")
            data_source = self.db.query(DataSource).filter(DataSource.id == data_source_id).first()
            if data_source and data_source.schema_summary:
                return build_schema_summary(
                    data_source_id=str(data_source.id),
                    data_source_name=data_source.name,
                    analysis_result=data_source.schema_summary,
                )

        # Fall back to tool call
        self.logger.thinking("Schema not in input, fetching via tool...")
        if self._tool_executor:
            result = self._tool_executor.execute_tool("get_data_source_info", {})

            # Check if we got data sources
            data_sources = result.get("data_sources", [])
            if data_sources:
                # Use the first data source
                ds_info = data_sources[0]
                schema_data = ds_info.get("schema_summary")
                if schema_data:
                    logger.info(f"Got schema from tool call for: {ds_info.get('name')}")
                    return SchemaSummary(**schema_data)

            # Check if it's a single data source response
            schema_data = result.get("schema_summary")
            if schema_data:
                logger.info("Got schema from tool call")
                return SchemaSummary(**schema_data)

        raise ValueError(
            "Could not find data source schema. "
            "Ensure the pipeline has a data source with schema analysis."
        )

    def _analyze_data_quality(self, schema_summary: SchemaSummary) -> tuple:
        """Analyze the schema for data quality issues."""
        issues = []
        recommendations = []
        high_null_cols = []
        id_cols = []
        constant_cols = []

        # Check for high null percentages
        for col in schema_summary.columns:
            if col.null_percentage > 30:
                high_null_cols.append({"name": col.name, "null_pct": col.null_percentage})
                issues.append(f"Column '{col.name}' has {col.null_percentage:.1f}% missing values")

        # Check for potential ID columns
        for col in schema_summary.columns:
            if col.unique_count == schema_summary.row_count:
                id_cols.append(col.name)

        # Check for low-variance columns
        for col in schema_summary.columns:
            if col.unique_count == 1:
                constant_cols.append(col.name)
                issues.append(f"Column '{col.name}' has only one unique value (constant)")

        # Check dataset size
        if schema_summary.row_count < 100:
            issues.append(f"Dataset is very small ({schema_summary.row_count} rows) - may not be enough for reliable ML")

        if schema_summary.row_count < 1000:
            recommendations.append("Consider collecting more data if possible - larger datasets typically give better results")

        return issues, high_null_cols, id_cols, constant_cols, recommendations

    async def _get_llm_analysis(
        self,
        description: str,
        schema_summary: SchemaSummary,
        issues: List[str],
        high_null_cols: List[Dict],
        id_cols: List[str],
        constant_cols: List[str],
    ) -> Dict[str, Any]:
        """Get LLM analysis of the data."""
        schema_str = _format_schema_for_prompt(schema_summary)

        prompt = get_data_analysis_prompt(
            description=description,
            data_source_name=schema_summary.data_source_name,
            row_count=schema_summary.row_count,
            column_count=schema_summary.column_count,
            schema_str=schema_str,
            issues=issues,
            id_cols=id_cols,
            constant_cols=constant_cols,
            high_null_cols=high_null_cols,
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_DATA_EVALUATOR},
            {"role": "user", "content": prompt},
        ]

        self.logger.action("Consulting AI for comprehensive data analysis...")
        return await self.llm.chat_json(messages, DataAnalysisResponse)

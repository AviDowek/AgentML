"""Training Dataset Build Agent - Materializes the training dataset.

This agent materializes a training dataset from a TrainingDatasetSpec,
creating the actual file and DataSource.
"""

from typing import Any, Dict
from uuid import UUID

from app.models import DataSource
from app.models import AgentStep, AgentStepType, AgentStepStatus
from app.services.agents.base import BaseAgent
from app.services.training_dataset_builder import (
    materialize_training_dataset,
    MaterializationResult,
)


class TrainingDatasetBuildAgent(BaseAgent):
    """Materializes a training dataset from a spec.

    Input JSON:
        - project_id: UUID of the project
        - training_dataset_spec: The TrainingDatasetSpec dictionary
        - max_rows: Optional maximum rows (default 1,000,000)
        - output_format: "parquet" or "csv" (default "parquet")

    Output:
        - data_source_id: ID of created data source
        - row_count: Number of rows
        - column_count: Number of columns
        - target_column: The target column
        - feature_columns: List of feature columns
        - output_format: Output format used
        - was_sampled: Whether dataset was sampled
        - original_row_count: Original row count if sampled
        - sampling_message: Sampling message if sampled
    """

    name = "training_dataset_build"
    step_type = AgentStepType.TRAINING_DATASET_BUILD

    async def execute(self) -> Dict[str, Any]:
        """Execute training dataset materialization."""
        project_id = self.require_input("project_id")
        training_dataset_spec = self.require_input("training_dataset_spec")
        max_rows = self.get_input("max_rows", 1_000_000)
        output_format = self.get_input("output_format", "parquet")

        self.logger.info("Starting training dataset materialization...")

        # Extract key info
        base_table = training_dataset_spec.get("base_table", "unknown")
        target_def = training_dataset_spec.get("target_definition", {})
        target_column = target_def.get("column", "unknown")
        join_plan = training_dataset_spec.get("join_plan", [])

        self.logger.thought(f"Base table: {base_table}")
        self.logger.thought(f"Target column: {target_column}")
        self.logger.thought(f"Join plan: {len(join_plan)} table(s)")

        # Get time-based metadata from problem_understanding step
        time_metadata = self._get_time_metadata()

        self.logger.info(f"Materializing dataset from '{base_table}' with {len(join_plan)} joins...")

        try:
            result: MaterializationResult = materialize_training_dataset(
                db=self.db,
                project_id=UUID(project_id) if isinstance(project_id, str) else project_id,
                training_dataset_spec=training_dataset_spec,
                max_rows=max_rows,
                output_format=output_format,
                step_logger=self.logger,
                **time_metadata,
            )

            # Load created data source for details
            data_source = self.db.query(DataSource).filter(DataSource.id == result.data_source_id).first()

            if data_source and data_source.schema_summary:
                columns = [c.get("name") for c in data_source.schema_summary.get("columns", [])]
                feature_columns = [c for c in columns if c != target_column]

                if result.was_sampled and result.sampling_message:
                    self.logger.warning(result.sampling_message)

                self.logger.info(f"Dataset created: {result.row_count:,} rows, {result.column_count} columns")

                summary_parts = [
                    f"Training dataset materialized successfully.",
                    f"DataSource ID: {result.data_source_id}.",
                    f"Size: {result.row_count:,} rows, {len(feature_columns)} features, target: {target_column}",
                ]
                if result.was_sampled:
                    summary_parts.append(f"(Sampled from {result.original_row_count:,} total rows)")

                self.logger.summary(" ".join(summary_parts))

                return {
                    "data_source_id": str(result.data_source_id),
                    "row_count": result.row_count,
                    "column_count": result.column_count,
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                    "output_format": output_format,
                    "was_sampled": result.was_sampled,
                    "original_row_count": result.original_row_count,
                    "sampling_message": result.sampling_message,
                }
            else:
                self.logger.warning("Data source created but schema_summary not available")
                return {
                    "data_source_id": str(result.data_source_id),
                    "target_column": target_column,
                    "output_format": output_format,
                }

        except Exception as e:
            self.logger.error(f"Failed to materialize training dataset: {str(e)}")
            raise

    def _get_time_metadata(self) -> Dict[str, Any]:
        """Get time-based metadata from problem_understanding step."""
        metadata = {
            "is_time_based": False,
            "time_column": None,
            "entity_id_column": None,
            "prediction_horizon": None,
            "target_positive_class": None,
        }

        if not self.step.agent_run_id:
            return metadata

        problem_step = self.db.query(AgentStep).filter(
            AgentStep.agent_run_id == self.step.agent_run_id,
            AgentStep.step_type == AgentStepType.PROBLEM_UNDERSTANDING,
            AgentStep.status == AgentStepStatus.COMPLETED,
        ).first()

        if problem_step and problem_step.output_json:
            problem_output = problem_step.output_json
            metadata["is_time_based"] = problem_output.get("is_time_based", False)
            metadata["time_column"] = problem_output.get("time_column")
            metadata["entity_id_column"] = problem_output.get("entity_id_column")
            metadata["prediction_horizon"] = problem_output.get("prediction_horizon")
            metadata["target_positive_class"] = problem_output.get("target_positive_class")

            if metadata["is_time_based"]:
                self.logger.thought(
                    f"Time-based task detected: time_column='{metadata['time_column']}', "
                    f"horizon='{metadata['prediction_horizon']}'"
                )

        return metadata

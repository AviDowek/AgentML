"""Problem Understanding Agent - Determines task type, target, and metrics.

This agent analyzes the user's goal and data schema to determine:
- Task type (binary, multiclass, regression, etc.)
- Target column
- Primary metric
- Confidence and reasoning
- Expected metric range based on baselines and task context
"""

import re
from typing import Any, Dict, List, Optional

from app.models import DataSource
from app.models import AgentStepType
from app.schemas.agent import SchemaSummary
from app.services.agents.base import BaseAgent
from app.services.agent_service import (
    build_schema_summary,
    generate_project_config,
)


def _log_context_usage(
    step_logger,
    task_context: Optional[Dict[str, Any]],
    task_hints: Optional[Dict[str, Any]] = None,
    step_name: str = "agent",
) -> Dict[str, Any]:
    """Log which contextual factors an agent is using.

    This helper provides consistent logging across all agents about the
    contextual information they're relying on.
    """
    context_factors_used = {
        "is_time_based": False,
        "has_baselines": False,
        "has_label_shuffle": False,
        "has_robustness_info": False,
        "has_leakage_candidates": False,
        "has_expected_metric_range": False,
        "factors_summary": [],
    }

    if not task_context:
        step_logger.thought(f"[{step_name}] No TaskContext available - using defaults")
        return context_factors_used

    # Check is_time_based
    dataset_spec = task_context.get("dataset_spec") or {}
    is_time_based = dataset_spec.get("is_time_based", False)
    time_column = dataset_spec.get("time_column")

    if task_hints and task_hints.get("is_time_based"):
        is_time_based = True

    if is_time_based:
        context_factors_used["is_time_based"] = True
        context_factors_used["time_column"] = time_column
        factor_msg = f"📅 Time-based task (time_column: {time_column or 'unspecified'})"
        context_factors_used["factors_summary"].append(factor_msg)
        step_logger.info(f"[{step_name}] {factor_msg}")

    # Check baselines
    baselines = task_context.get("baselines", {})
    if baselines.get("available"):
        context_factors_used["has_baselines"] = True
        baseline_types = []
        if baselines.get("majority_class"):
            baseline_types.append("majority_class")
        if baselines.get("simple_model"):
            baseline_types.append("simple_model")
        if baselines.get("mean_predictor"):
            baseline_types.append("mean_predictor")
        if baselines.get("regression_baseline"):
            baseline_types.append("regression_baseline")

        factor_msg = f"📊 Using baseline metrics: {', '.join(baseline_types)}"
        context_factors_used["factors_summary"].append(factor_msg)
        context_factors_used["baseline_types"] = baseline_types
        step_logger.info(f"[{step_name}] {factor_msg}")

    # Check label shuffle results
    label_shuffle = task_context.get("label_shuffle", {})
    if label_shuffle.get("available"):
        context_factors_used["has_label_shuffle"] = True
        leakage_detected = label_shuffle.get("leakage_detected", False)
        factor_msg = f"🔀 Label shuffle test available (leakage_detected: {leakage_detected})"
        context_factors_used["factors_summary"].append(factor_msg)
        context_factors_used["label_shuffle_leakage"] = leakage_detected
        if leakage_detected:
            step_logger.warning(f"[{step_name}] {factor_msg}")
        else:
            step_logger.info(f"[{step_name}] {factor_msg}")

    # Check robustness info
    robustness = task_context.get("robustness")
    if robustness:
        context_factors_used["has_robustness_info"] = True
        overfitting_risk = robustness.get("overfitting_risk", "unknown")
        leakage_suspected = robustness.get("leakage_suspected", False)
        risk_level = robustness.get("risk_level", "unknown")

        factor_msg = f"🛡️ Robustness info: overfitting_risk={overfitting_risk}, risk_level={risk_level}"
        context_factors_used["factors_summary"].append(factor_msg)
        context_factors_used["overfitting_risk"] = overfitting_risk
        context_factors_used["risk_level"] = risk_level

        if overfitting_risk == "high" or leakage_suspected or risk_level in ("high", "critical"):
            step_logger.warning(f"[{step_name}] {factor_msg}")
        else:
            step_logger.info(f"[{step_name}] {factor_msg}")

        if leakage_suspected:
            step_logger.warning(f"[{step_name}] ⚠️ Leakage suspected from prior robustness analysis")
            context_factors_used["leakage_suspected"] = True

    # Check leakage candidates
    leakage_candidates = task_context.get("leakage_candidates", [])
    if leakage_candidates:
        context_factors_used["has_leakage_candidates"] = True
        high_count = len([lc for lc in leakage_candidates if lc.get("severity") == "high"])
        medium_count = len([lc for lc in leakage_candidates if lc.get("severity") == "medium"])
        total_count = len(leakage_candidates)

        factor_msg = f"🔓 Leakage candidates: {total_count} total ({high_count} high, {medium_count} medium)"
        context_factors_used["factors_summary"].append(factor_msg)
        context_factors_used["leakage_high_count"] = high_count
        context_factors_used["leakage_total_count"] = total_count

        if high_count > 0:
            step_logger.warning(f"[{step_name}] {factor_msg}")
        else:
            step_logger.info(f"[{step_name}] {factor_msg}")

    # Check expected metric range
    if task_hints and task_hints.get("expected_metric_range"):
        context_factors_used["has_expected_metric_range"] = True
        expected_range = task_hints["expected_metric_range"]
        metric = expected_range.get("metric", "primary")
        lower = expected_range.get("lower_bound")
        upper = expected_range.get("upper_bound")
        if lower is not None and upper is not None:
            factor_msg = f"📈 Expected {metric} range: [{lower:.3f} - {upper:.3f}]"
            context_factors_used["factors_summary"].append(factor_msg)
            step_logger.info(f"[{step_name}] {factor_msg}")

    # Log summary
    factor_count = len(context_factors_used["factors_summary"])
    if factor_count > 0:
        step_logger.thought(f"[{step_name}] Using {factor_count} contextual factor(s) from TaskContext")
    else:
        step_logger.thought(f"[{step_name}] TaskContext available but no specific factors applied")

    return context_factors_used


def _calculate_expected_metric_range(
    task_type: str,
    primary_metric: str,
    task_context: Optional[Dict[str, Any]] = None,
    class_balance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate expected metric range based on task type and baseline info."""
    result: Dict[str, Any] = {
        "primary_metric": {
            "lower_bound": None,
            "upper_bound": None,
            "reasoning": "No baseline information available; using default expectations.",
        }
    }

    baselines = {}
    robustness = {}
    if task_context:
        baselines = task_context.get("baselines", {})
        robustness = task_context.get("robustness", {}) or {}

    if task_type in ("binary", "multiclass"):
        majority_acc = None
        simple_model_metric = None

        if baselines.get("available"):
            majority_class = baselines.get("majority_class", {})
            majority_acc = majority_class.get("accuracy")

            simple_model = baselines.get("simple_model", {})
            if primary_metric in ("roc_auc", "auc"):
                simple_model_metric = simple_model.get("roc_auc")
            elif primary_metric == "accuracy":
                simple_model_metric = simple_model.get("accuracy")
            elif primary_metric in ("f1", "f1_score"):
                simple_model_metric = simple_model.get("f1")

        reasoning_parts = []

        if primary_metric in ("roc_auc", "auc"):
            if simple_model_metric is not None:
                lower = min(simple_model_metric + 0.03, 0.95)
                upper = min(simple_model_metric + 0.15, 0.95)
                result["primary_metric"]["lower_bound"] = round(lower, 3)
                result["primary_metric"]["upper_bound"] = round(upper, 3)
                reasoning_parts.append(f"Simple model AUC = {simple_model_metric:.3f}")
                reasoning_parts.append(f"Expecting improvement of 0.03-0.15 above baseline")
            else:
                result["primary_metric"]["lower_bound"] = 0.55
                result["primary_metric"]["upper_bound"] = 0.75
                reasoning_parts.append("No baseline AUC available; using conservative defaults")

            if class_balance:
                minority_pct = None
                if isinstance(class_balance, dict):
                    values = list(class_balance.values())
                    if len(values) >= 2:
                        total = sum(v for v in values if isinstance(v, (int, float)))
                        if total > 0:
                            minority_pct = min(values) / total if all(isinstance(v, (int, float)) for v in values) else None

                if minority_pct is not None and minority_pct < 0.1:
                    result["primary_metric"]["upper_bound"] = min(
                        result["primary_metric"]["upper_bound"] or 0.75,
                        0.80
                    )
                    reasoning_parts.append(f"Class imbalance detected (minority ~{minority_pct:.1%})")

            if (result["primary_metric"]["upper_bound"] or 0) > 0.85:
                result["primary_metric"]["upper_bound"] = 0.85
                reasoning_parts.append("AUC > 0.85 is often suspicious; capping at 0.85")

        elif primary_metric == "accuracy":
            if majority_acc is not None:
                lower = majority_acc + 0.03
                upper = min(majority_acc + 0.20, 0.95)
                result["primary_metric"]["lower_bound"] = round(lower, 3)
                result["primary_metric"]["upper_bound"] = round(upper, 3)
                reasoning_parts.append(f"Majority-class accuracy = {majority_acc:.3f}")
                reasoning_parts.append(f"Expecting improvement of 3-20 points above baseline")
            else:
                result["primary_metric"]["lower_bound"] = 0.55
                result["primary_metric"]["upper_bound"] = 0.85
                reasoning_parts.append("No baseline accuracy available; using conservative defaults")

        else:
            if simple_model_metric is not None:
                lower = simple_model_metric + 0.03
                upper = min(simple_model_metric + 0.20, 0.95)
                result["primary_metric"]["lower_bound"] = round(lower, 3)
                result["primary_metric"]["upper_bound"] = round(upper, 3)
                reasoning_parts.append(f"Simple model {primary_metric} = {simple_model_metric:.3f}")
            else:
                result["primary_metric"]["lower_bound"] = 0.50
                result["primary_metric"]["upper_bound"] = 0.80
                reasoning_parts.append(f"No baseline {primary_metric} available; using conservative defaults")

        result["primary_metric"]["reasoning"] = "; ".join(reasoning_parts) if reasoning_parts else result["primary_metric"]["reasoning"]

    elif task_type == "regression":
        mean_predictor = baselines.get("mean_predictor", {}) if baselines.get("available") else {}
        regression_baseline = baselines.get("regression_baseline", {}) if baselines.get("available") else {}

        reasoning_parts = []

        if primary_metric in ("rmse", "mse", "root_mean_squared_error"):
            baseline_rmse = mean_predictor.get("rmse") or regression_baseline.get("rmse")
            if baseline_rmse is not None:
                upper = baseline_rmse * 0.90
                lower = baseline_rmse * 0.60
                result["primary_metric"]["lower_bound"] = round(lower, 4)
                result["primary_metric"]["upper_bound"] = round(upper, 4)
                reasoning_parts.append(f"Baseline RMSE = {baseline_rmse:.4f}")
                reasoning_parts.append("Expecting 10-40% improvement over baseline")
            else:
                result["primary_metric"]["reasoning"] = "No baseline RMSE available; targets should be set after seeing initial results"

        elif primary_metric in ("mae", "mean_absolute_error"):
            baseline_mae = mean_predictor.get("mae") or regression_baseline.get("mae")
            if baseline_mae is not None:
                upper = baseline_mae * 0.90
                lower = baseline_mae * 0.60
                result["primary_metric"]["lower_bound"] = round(lower, 4)
                result["primary_metric"]["upper_bound"] = round(upper, 4)
                reasoning_parts.append(f"Baseline MAE = {baseline_mae:.4f}")
                reasoning_parts.append("Expecting 10-40% improvement over baseline")
            else:
                result["primary_metric"]["reasoning"] = "No baseline MAE available; targets should be set after seeing initial results"

        elif primary_metric in ("r2", "r2_score", "r_squared"):
            baseline_r2 = mean_predictor.get("r2") or regression_baseline.get("r2")
            if baseline_r2 is not None:
                lower = max(baseline_r2 + 0.10, 0.20)
                upper = min(baseline_r2 + 0.40, 0.90)
                result["primary_metric"]["lower_bound"] = round(lower, 3)
                result["primary_metric"]["upper_bound"] = round(upper, 3)
                reasoning_parts.append(f"Baseline R² = {baseline_r2:.3f}")
                reasoning_parts.append("Expecting 0.10-0.40 improvement over baseline")
            else:
                result["primary_metric"]["lower_bound"] = 0.20
                result["primary_metric"]["upper_bound"] = 0.70
                reasoning_parts.append("No baseline R² available; using conservative defaults")

        if reasoning_parts:
            result["primary_metric"]["reasoning"] = "; ".join(reasoning_parts)

    if robustness.get("leakage_suspected"):
        result["primary_metric"]["reasoning"] += " WARNING: Leakage suspected - actual performance may be lower in production."

    return result


class ProblemUnderstandingAgent(BaseAgent):
    """Analyzes user's goal and data to determine task configuration.

    Input JSON:
        - description: User's goal description
        - data_source_id: UUID of the data source
        - schema_summary: Pre-built schema summary (optional)
        - task_context: Optional task context with baselines, robustness info

    Output:
        - task_type: binary, multiclass, or regression
        - target_column: The target column name
        - target_exists: Whether target exists in data
        - target_creation: Target creation info if needed
        - primary_metric: The primary evaluation metric
        - secondary_metrics: List of secondary metrics
        - reasoning: Explanation of decisions
        - confidence: Confidence score (0-1)
        - suggested_name: Suggested project name
        - suggested_feature_engineering: Feature engineering suggestions
        - schema_summary: Schema for downstream use
        - is_time_based: Whether task is time-based
        - time_column: Time column if time-based
        - entity_id_column: Entity ID column if applicable
        - prediction_horizon: Prediction horizon if time-based
        - target_positive_class: Positive class for binary
        - expected_metric_range: Expected performance range
        - context_factors_used: What context was used
    """

    name = "problem_understanding"
    step_type = AgentStepType.PROBLEM_UNDERSTANDING

    async def execute(self) -> Dict[str, Any]:
        """Execute problem understanding analysis."""
        description = self.require_input("description")
        self.logger.info(f"Analyzing problem: {description[:100]}...")

        # Check for validation feedback about missing target column
        description = self._add_validation_feedback_to_description(description)

        # Extract task_context if available
        task_context = self.get_input("task_context")

        # Log context usage
        context_factors = _log_context_usage(
            step_logger=self.logger,
            task_context=task_context,
            task_hints=None,
            step_name="Problem Understanding",
        )

        # Additional logging for data profile
        if task_context:
            data_profile = task_context.get("data_profile_summary", {}) or {}
            if data_profile.get("row_count"):
                self.logger.thought(
                    f"Data profile: {data_profile['row_count']} rows, "
                    f"{data_profile.get('feature_count', 'unknown')} features"
                )

        # Get or build schema summary
        schema_summary = await self._get_schema_summary()

        self.logger.thought(
            f"Dataset has {schema_summary.row_count} rows and "
            f"{schema_summary.column_count} columns"
        )

        # Analyze columns
        numeric_cols = [c.name for c in schema_summary.columns if c.inferred_type == "numeric"]
        categorical_cols = [c.name for c in schema_summary.columns if c.inferred_type == "categorical"]
        self.logger.thought(
            f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns"
        )

        # Call LLM to generate config with target quality validation
        self.logger.info("Consulting LLM to determine task configuration...")

        suggestion = await self._get_suggestion_with_validation(
            description=description,
            schema_summary=schema_summary,
        )

        # Log target creation info
        if suggestion.target_creation:
            self.logger.info(
                f"Target column '{suggestion.target_column}' will be created using formula: "
                f"{suggestion.target_creation.formula}"
            )

        # Log suggested feature engineering
        if suggestion.suggested_feature_engineering:
            self.logger.info(
                f"Suggested {len(suggestion.suggested_feature_engineering)} feature engineering steps"
            )

        # Log time-based task detection
        if suggestion.is_time_based:
            self.logger.info(
                f"Detected time-based task: time_column='{suggestion.time_column}', "
                f"horizon='{suggestion.prediction_horizon}'"
            )
            if suggestion.entity_id_column:
                self.logger.info(f"Entity ID column: '{suggestion.entity_id_column}'")
            if suggestion.target_positive_class:
                self.logger.info(f"Target positive class: '{suggestion.target_positive_class}'")
            self.logger.thought(
                "Time-based task detected - time-aware splits are required to prevent look-ahead bias"
            )
        else:
            self.logger.thought("Task is not time-based; no need for time-aware splits")

        # Calculate expected metric range
        class_balance = self._get_class_balance(schema_summary, suggestion.target_column, task_context)

        expected_metric_range = _calculate_expected_metric_range(
            task_type=suggestion.task_type,
            primary_metric=suggestion.primary_metric,
            task_context=task_context,
            class_balance=class_balance,
        )

        # Log the expected metric range
        primary_range = expected_metric_range.get("primary_metric", {})
        if primary_range.get("lower_bound") is not None and primary_range.get("upper_bound") is not None:
            self.logger.thought(
                f"Expected {suggestion.primary_metric} range: "
                f"{primary_range['lower_bound']:.3f} - {primary_range['upper_bound']:.3f}"
            )
            self.logger.thought(f"Reasoning: {primary_range.get('reasoning', 'N/A')}")
        else:
            self.logger.thought(
                f"Expected metric range: {primary_range.get('reasoning', 'Unable to determine range')}"
            )

        # Add secondary metrics
        secondary_metrics = self._get_secondary_metrics(
            suggestion.task_type, suggestion.primary_metric
        )

        return {
            "task_type": suggestion.task_type,
            "target_column": suggestion.target_column,
            "target_exists": suggestion.target_exists,
            "target_creation": suggestion.target_creation.model_dump(mode="json") if suggestion.target_creation else None,
            "primary_metric": suggestion.primary_metric,
            "secondary_metrics": secondary_metrics,
            "reasoning": suggestion.reasoning,
            "confidence": suggestion.confidence,
            "suggested_name": suggestion.suggested_name,
            "suggested_feature_engineering": [
                fe.model_dump(mode="json") for fe in suggestion.suggested_feature_engineering
            ],
            "schema_summary": schema_summary.model_dump(mode="json"),
            "is_time_based": suggestion.is_time_based,
            "time_column": suggestion.time_column,
            "entity_id_column": suggestion.entity_id_column,
            "prediction_horizon": suggestion.prediction_horizon,
            "target_positive_class": suggestion.target_positive_class,
            "expected_metric_range": expected_metric_range,
            "context_factors_used": context_factors,
        }

    async def _get_schema_summary(self) -> SchemaSummary:
        """Get or build the schema summary."""
        schema_data = self.get_input("schema_summary")
        if schema_data:
            self.logger.info(f"Using provided schema: {schema_data.get('data_source_name', 'unknown')}")
            return SchemaSummary(**schema_data)

        data_source_id = self.get_input("data_source_id")
        if not data_source_id:
            raise ValueError("Missing 'data_source_id' or 'schema_summary' in input_json")

        self.logger.info("Loading data source schema...")
        data_source = self.db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if not data_source:
            raise ValueError(f"Data source not found: {data_source_id}")

        if not data_source.schema_summary:
            raise ValueError(f"Data source {data_source_id} has no schema analysis")

        return build_schema_summary(
            data_source_id=str(data_source.id),
            data_source_name=data_source.name,
            analysis_result=data_source.schema_summary,
        )

    def _get_class_balance(
        self,
        schema_summary: SchemaSummary,
        target_column: str,
        task_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extract class balance from schema or task context."""
        class_balance = None

        if schema_summary.columns:
            for col in schema_summary.columns:
                if col.name == target_column and hasattr(col, 'value_counts') and col.value_counts:
                    class_balance = col.value_counts
                    break

        if not class_balance and task_context:
            data_profile = task_context.get("data_profile_summary", {}) or {}
            class_balance = data_profile.get("class_balance")

        return class_balance

    def _get_secondary_metrics(self, task_type: str, primary_metric: str) -> List[str]:
        """Get appropriate secondary metrics for the task type."""
        if task_type == "binary":
            secondary = ["accuracy", "f1", "precision", "recall"]
            if primary_metric != "roc_auc":
                secondary.insert(0, "roc_auc")
        elif task_type == "multiclass":
            secondary = ["accuracy", "f1_macro", "f1_weighted"]
        elif task_type == "regression":
            secondary = ["rmse", "mae", "r2"]
            if primary_metric in secondary:
                secondary.remove(primary_metric)
        else:
            secondary = []

        return secondary

    def _add_validation_feedback_to_description(self, description: str) -> str:
        """Add validation feedback to description when target column was invalid.

        This handles feedback from two sources:
        1. dataset_validation: has 'missing_target' and 'available_columns'
        2. data_audit: has 'critical_issues' containing "Target column X not found"

        The PM re-runs problem_understanding with this feedback so the LLM
        can pick a valid target from the available columns.
        """
        # Check for feedback from dataset_validation
        missing_target = self.get_input("missing_target")
        available_columns = self.get_input("available_columns", [])

        # Check for feedback from data_audit (critical_issues)
        critical_issues = self.get_input("critical_issues", [])
        audit_missing_target = None
        for issue in critical_issues:
            if "Target column" in issue and "not found" in issue:
                # Extract target name from "Target column 'X' not found in dataset!"
                match = re.search(r"Target column ['\"]?([^'\"]+)['\"]? not found", issue)
                if match:
                    audit_missing_target = match.group(1)
                    break

        # Use whichever source provided the missing target info
        invalid_target = missing_target or audit_missing_target
        if not invalid_target:
            return description

        self.logger.warning(f"Received feedback - target '{invalid_target}' does not exist!")

        feedback = f"\n\n## ⚠️ CRITICAL: TARGET COLUMN DOES NOT EXIST\n"
        feedback += f"Your previous selection of target column '{invalid_target}' is INVALID.\n"
        feedback += f"This column does NOT exist in the data source.\n\n"

        # Get available columns from schema_summary if not provided directly
        if not available_columns:
            schema_summary = self.get_input("schema_summary")
            if schema_summary:
                columns = schema_summary.get("columns", [])
                available_columns = [
                    c.get("name") or c.get("column_name") or c
                    for c in columns
                    if isinstance(c, dict) or isinstance(c, str)
                ]

        feedback += "### You have TWO options:\n\n"

        feedback += "**Option 1: Pick an existing column** (RECOMMENDED)\n"
        if available_columns:
            feedback += "Choose a target from these available columns:\n"
            for col in sorted(available_columns)[:30]:  # Limit display
                feedback += f"- `{col}`\n"
            if len(available_columns) > 30:
                feedback += f"... and {len(available_columns) - 30} more\n"
            feedback += "\n"

        feedback += "**Option 2: Engineer the target** (ADVANCED)\n"
        feedback += "If you want to CREATE a target from existing columns, you must provide:\n"
        feedback += "- `target_exists: false`\n"
        feedback += "- `target_creation` with:\n"
        feedback += "  - `formula`: How to compute the target (e.g., 'max(revenue) grouped by seller')\n"
        feedback += "  - `source_columns`: List of existing columns needed (must exist!)\n"
        feedback += "  - `description`: What the engineered target represents\n\n"

        feedback += "**INSTRUCTION**: Choose Option 1 unless there's a clear reason to engineer a target.\n"
        feedback += "If engineering, ensure ALL source_columns exist in the available columns list.\n"

        return description + feedback

    def _validate_target_column_quality(
        self,
        target_column: str,
        schema_summary: "SchemaSummary",
        max_null_percentage: float = 20.0,
    ) -> tuple[bool, Optional[str]]:
        """Validate that the target column has acceptable data quality.

        Returns:
            (is_valid, feedback_message) - If invalid, feedback_message contains
            instructions for the LLM to pick a better target.
        """
        # Find the target column in schema
        target_col_info = None
        for col in schema_summary.columns:
            if col.name == target_column:
                target_col_info = col
                break

        if not target_col_info:
            # Column doesn't exist - handled by _add_validation_feedback_to_description
            return True, None

        null_pct = target_col_info.null_percentage

        if null_pct <= max_null_percentage:
            if null_pct > 10:
                self.logger.warning(
                    f"Target column '{target_column}' has {null_pct:.1f}% null values - "
                    f"this is acceptable but may reduce training data"
                )
            return True, None

        # Target has too many nulls - build feedback for LLM
        self.logger.error(
            f"Target column '{target_column}' has {null_pct:.1f}% null values - "
            f"this exceeds the maximum of {max_null_percentage}%"
        )

        # Find alternative columns with low null rates
        good_alternatives = []
        for col in schema_summary.columns:
            if col.name != target_column and col.null_percentage <= max_null_percentage:
                # Prefer categorical or numeric columns that could be targets
                if col.inferred_type in ("categorical", "numeric", "boolean"):
                    if col.unique_count >= 2:  # Must have at least 2 values
                        good_alternatives.append({
                            "name": col.name,
                            "type": col.inferred_type,
                            "null_pct": col.null_percentage,
                            "unique_count": col.unique_count,
                        })

        # Sort by null percentage (lower is better)
        good_alternatives.sort(key=lambda x: x["null_pct"])

        feedback = f"\n\n## ⚠️ CRITICAL: TARGET COLUMN HAS TOO MANY NULL VALUES\n"
        feedback += f"Your selected target column '{target_column}' has **{null_pct:.1f}% null/missing values**.\n"
        feedback += f"This is too high (maximum allowed: {max_null_percentage}%).\n\n"
        feedback += "A target column with this many missing values cannot be used for training.\n\n"

        feedback += "### You MUST do one of the following:\n\n"

        feedback += "**Option 1: Pick a different target column** (RECOMMENDED)\n"
        if good_alternatives:
            feedback += "Here are columns with acceptable null rates that could be targets:\n"
            for alt in good_alternatives[:10]:
                feedback += f"- `{alt['name']}` ({alt['type']}, {alt['null_pct']:.1f}% nulls, {alt['unique_count']} unique values)\n"
            feedback += "\n"
        else:
            feedback += "WARNING: No columns with acceptable null rates found. Consider Option 2.\n\n"

        feedback += "**Option 2: Engineer a target from other columns**\n"
        feedback += "Create a derived target using `target_exists: false` and `target_creation`:\n"
        feedback += "- `formula`: How to compute the target (e.g., 'Price > median(Price)')\n"
        feedback += "- `source_columns`: List of existing columns with low null rates\n"
        feedback += "- `description`: What the engineered target represents\n\n"

        feedback += f"**DO NOT** select '{target_column}' again. It has too many missing values.\n"

        return False, feedback

    def _normalize_metric_for_task_type(self, metric: str, task_type: str) -> str:
        """Normalize metric name to be compatible with the task type.

        AutoGluon requires specific metric variants for different problem types.
        """
        if task_type == "multiclass":
            # These metrics only work for binary classification
            multiclass_fixes = {
                "f1": "f1_macro",
                "roc_auc": "roc_auc_ovr_weighted",
                "precision": "precision_macro",
                "recall": "recall_macro",
            }
            if metric.lower() in multiclass_fixes:
                fixed = multiclass_fixes[metric.lower()]
                self.logger.warning(
                    f"Metric '{metric}' is invalid for multiclass - using '{fixed}' instead"
                )
                return fixed
        elif task_type == "regression":
            # These are classification metrics - not valid for regression
            regression_fixes = {
                "f1": "rmse",
                "f1_macro": "rmse",
                "accuracy": "rmse",
                "roc_auc": "rmse",
            }
            if metric.lower() in regression_fixes:
                fixed = regression_fixes[metric.lower()]
                self.logger.warning(
                    f"Metric '{metric}' is invalid for regression - using '{fixed}' instead"
                )
                return fixed
        return metric

    async def _get_suggestion_with_validation(
        self,
        description: str,
        schema_summary: "SchemaSummary",
        max_retries: int = 2,
    ) -> Any:
        """Get LLM suggestion with target quality validation and retry.

        If the LLM picks a target with too many nulls, we add feedback and retry.
        """
        current_description = description
        rejected_targets = []

        for attempt in range(max_retries + 1):
            suggestion = await generate_project_config(
                client=self.llm,
                description=current_description,
                schema_summary=schema_summary,
            )

            # Normalize the primary metric for the task type
            normalized_metric = self._normalize_metric_for_task_type(
                suggestion.primary_metric, suggestion.task_type
            )
            if normalized_metric != suggestion.primary_metric:
                suggestion.primary_metric = normalized_metric

            self.logger.summary(
                f"Identified {suggestion.task_type} task targeting '{suggestion.target_column}' "
                f"with {suggestion.primary_metric} metric (confidence: {suggestion.confidence:.0%})"
            )

            # If target is being created (doesn't exist), skip null validation
            if not suggestion.target_exists:
                self.logger.info(
                    f"Target '{suggestion.target_column}' will be created - skipping null validation"
                )
                return suggestion

            # Validate target quality
            is_valid, feedback = self._validate_target_column_quality(
                suggestion.target_column,
                schema_summary,
            )

            if is_valid:
                return suggestion

            # Target has quality issues
            rejected_targets.append(suggestion.target_column)

            if attempt >= max_retries:
                # Max retries reached - fail with clear error
                raise ValueError(
                    f"Cannot proceed: Target column '{suggestion.target_column}' has too many null values, "
                    f"and no valid alternative was found after {max_retries} attempts. "
                    f"Rejected targets: {rejected_targets}. "
                    f"Please clean your data or specify a valid target column."
                )

            # Add feedback for retry
            self.logger.warning(
                f"Retrying target selection (attempt {attempt + 2}/{max_retries + 1}) - "
                f"previous target '{suggestion.target_column}' rejected due to high null rate"
            )

            # Prevent LLM from picking the same bad targets again
            reject_list = ", ".join(f"'{t}'" for t in rejected_targets)
            current_description = description + feedback
            current_description += f"\n\n**PREVIOUSLY REJECTED TARGETS**: {reject_list}\n"
            current_description += "Do NOT select any of these columns as the target.\n"

        # Should not reach here
        return suggestion

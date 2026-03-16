"""Experiment Design Agent - Creates experiment variants with AutoML configs.

This agent creates experiment variants with different configurations.
It uses project history context to learn from previous experiments.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models import AgentRun, DatasetSpec
from app.models import AgentStepType
from app.schemas.agent import ExperimentPlanSuggestion
from app.services.agents.base import BaseAgent
from app.services.agent_service import generate_experiment_plan
from app.services.prompts import (
    SYSTEM_ROLE_EXPERIMENT_DESIGN_WITH_TOOLS,
    get_experiment_plan_prompt,
)
from app.services.agent_tools import AgentToolExecutor
from app.services.agent_service import execute_with_tools
from app.services.task_context import (
    build_task_context,
    get_task_type_hints,
    format_context_for_prompt,
)


def _fill_experiment_design_defaults(result: Dict[str, Any], step_logger) -> Dict[str, Any]:
    """Fill in missing fields with sensible defaults."""
    if "variants" not in result:
        result["variants"] = []
    if "recommended_variant" not in result:
        if result["variants"]:
            result["recommended_variant"] = result["variants"][0].get("name", "experiment_1")
        else:
            result["recommended_variant"] = "experiment_1"
    if "reasoning" not in result:
        result["reasoning"] = "No specific reasoning provided."
    if "estimated_total_time_minutes" not in result:
        result["estimated_total_time_minutes"] = 60

    # Fill defaults for each variant
    for i, variant in enumerate(result.get("variants", [])):
        if "name" not in variant:
            variant["name"] = f"experiment_{i + 1}"
        if "description" not in variant:
            variant["description"] = f"Experiment variant {i + 1}"
        if "automl_config" not in variant:
            variant["automl_config"] = {"time_limit": 300, "presets": "medium_quality"}
        if "validation_strategy" not in variant:
            variant["validation_strategy"] = {
                "split_strategy": "stratified",
                "test_size": 0.2,
            }

    return result


def _analyze_task_context(task_context: Optional[Dict], step_logger) -> Dict[str, Any]:
    """Analyze task context for experiment design decisions."""
    analysis = {
        "is_time_based": False,
        "time_column": None,
        "entity_id_column": None,
        "recommended_split_type": "stratified",
        "split_type_reasoning": "Default stratified split for classification tasks.",
        "leakage_features_to_drop": [],
        "prior_warnings": [],
        "should_run_sanity_check": False,
        "expected_metric_range": None,
        "failed_strategies": [],
    }

    if not task_context:
        step_logger.thought("No task context available - using default experiment design strategy")
        return analysis

    step_logger.thought("Analyzing task context for experiment design decisions...")

    # Extract time-based info
    dataset_spec = task_context.get("dataset_spec", {}) or {}
    analysis["is_time_based"] = dataset_spec.get("is_time_based", False)
    analysis["time_column"] = dataset_spec.get("time_column")
    analysis["entity_id_column"] = dataset_spec.get("entity_id_column")

    if analysis["is_time_based"]:
        step_logger.thought(f"Time-based task detected: time_column='{analysis['time_column']}'")
        if analysis["entity_id_column"]:
            analysis["recommended_split_type"] = "group_time"
            analysis["split_type_reasoning"] = (
                f"Time-based split with entity grouping (entity_id_column='{analysis['entity_id_column']}') "
                "to prevent look-ahead bias and entity leakage."
            )
        else:
            analysis["recommended_split_type"] = "time"
            analysis["split_type_reasoning"] = (
                f"Time-based split using '{analysis['time_column']}' to prevent look-ahead bias."
            )
        step_logger.thought(f"Recommended split: {analysis['recommended_split_type']}")

    # Extract robustness warnings
    robustness = task_context.get("robustness", {}) or {}
    if robustness.get("leakage_suspected"):
        analysis["prior_warnings"].append("Prior experiments suggest potential data leakage")
        analysis["should_run_sanity_check"] = True

    if robustness.get("overfitting_risk") == "high":
        analysis["prior_warnings"].append("High overfitting risk detected in prior experiments")
        analysis["should_run_sanity_check"] = True

    # Extract leakage candidates
    leakage_candidates = task_context.get("leakage_candidates", []) or []
    high_severity = [c["column"] for c in leakage_candidates if c.get("severity") == "high"]
    if high_severity:
        analysis["leakage_features_to_drop"] = high_severity
        analysis["should_run_sanity_check"] = True
        step_logger.thought(f"Identified {len(high_severity)} high-severity leakage feature(s)")

    return analysis


def _generate_experiment_family_goals(
    task_type: str,
    primary_metric: str,
    context_analysis: Dict[str, Any],
    time_budget_minutes: Optional[int],
) -> List[Dict[str, Any]]:
    """Generate experiment family goals based on context."""
    experiments = []
    total_budget = time_budget_minutes or 120
    is_time_based = context_analysis.get("is_time_based", False)
    should_sanity_check = context_analysis.get("should_run_sanity_check", False)
    leakage_features = context_analysis.get("leakage_features_to_drop", [])

    if should_sanity_check or is_time_based:
        experiments.append({
            "name": "Baseline Sanity Check",
            "goal": (
                f"Validate that {context_analysis['recommended_split_type']} split "
                f"{primary_metric} is above baseline but not unrealistically high."
            ),
            "time_budget_minutes": min(15, total_budget // 6),
            "preset": "good_quality",
            "split_type": context_analysis["recommended_split_type"],
        })

    main_budget = total_budget // 2 if should_sanity_check else total_budget * 2 // 3
    experiments.append({
        "name": "Primary Experiment",
        "goal": (
            f"Train models optimizing {primary_metric} with {context_analysis['recommended_split_type']} split."
        ),
        "time_budget_minutes": main_budget,
        "preset": "best_quality" if total_budget >= 60 else "good_quality",
        "split_type": context_analysis["recommended_split_type"],
    })

    if leakage_features:
        experiments.append({
            "name": "Leakage Feature Ablation",
            "goal": f"Test model stability by dropping suspected leakage features: {', '.join(leakage_features[:3])}",
            "time_budget_minutes": min(30, total_budget // 4),
            "preset": "good_quality",
            "split_type": context_analysis["recommended_split_type"],
            "features_to_drop": leakage_features,
        })

    return experiments


class ExperimentDesignAgent(BaseAgent):
    """Creates experiment variants with AutoML configurations.

    Input JSON:
        - task_type: The ML task type
        - target_column: The target column
        - primary_metric: The metric to optimize
        - feature_columns: Selected feature columns
        - row_count: Number of rows
        - time_budget_minutes: Optional time constraint
        - description: Optional additional context
        - target_stats: Optional target variable statistics
        - critical_issues: From audit (optional)
        - warnings: From audit (optional)
        - audit_details: Audit details (optional)
        - task_context: Task context (optional)
        - revision_request: Whether this is a revision (optional)
        - critic_feedback: Feedback from critic (optional)

    Output:
        - variants: List of experiment variant configurations
        - recommended_variant: Name of recommended variant
        - reasoning: Design rationale
        - estimated_total_time_minutes: Total estimated time
        - split_strategy: Recommended split strategy info
        - experiment_family_goals: Experiment goals
        - context_analysis: Context analysis results
        - context_factors_used: Context factors used
    """

    name = "experiment_design"
    step_type = AgentStepType.EXPERIMENT_DESIGN

    async def execute(self) -> Dict[str, Any]:
        """Execute experiment design generation."""
        task_type = self.require_input("task_type")
        target_column = self.require_input("target_column")
        primary_metric = self.require_input("primary_metric")
        feature_columns = self.get_input("feature_columns", [])
        row_count = self.get_input("row_count", 0)
        time_budget_minutes = self.get_input("time_budget_minutes")
        description = self.get_input("description")
        target_stats = self.get_input("target_stats")

        self.logger.info(f"Designing experiment for {task_type} task with {len(feature_columns)} features")
        self.logger.thinking(f"Dataset has {row_count:,} rows, optimizing for {primary_metric}")

        if time_budget_minutes:
            self.logger.info(f"Time budget constraint: {time_budget_minutes} minutes")

        # Get project context
        project_id = self.step.agent_run.project_id if self.step.agent_run else None
        research_cycle_id = getattr(self.step.agent_run, 'research_cycle_id', None) if self.step.agent_run else None

        # Build task context
        task_context, task_hints, task_context_str, context_factors = await self._build_context(
            project_id, research_cycle_id
        )

        # Analyze context for decisions
        context_analysis = _analyze_task_context(
            task_context or self.get_input("task_context"), self.logger
        )

        # Build prompt
        user_prompt = self._build_prompt(
            task_type, target_column, primary_metric, feature_columns,
            row_count, time_budget_minutes, description, target_stats,
            task_context_str, task_hints, context_analysis
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_EXPERIMENT_DESIGN_WITH_TOOLS},
            {"role": "user", "content": user_prompt},
        ]

        # Execute design
        if project_id:
            suggestion = await self._execute_with_tools(project_id, research_cycle_id, messages)
        else:
            self.logger.action("Consulting LLM for experiment variants...")
            suggestion = await generate_experiment_plan(
                client=self.llm,
                task_type=task_type,
                target_column=target_column,
                primary_metric=primary_metric,
                feature_columns=feature_columns,
                row_count=row_count,
                time_budget_minutes=time_budget_minutes,
                description=description,
                target_stats=target_stats,
                project_history_context=None,
            )

        self.logger.info(f"Generated {len(suggestion.variants)} experiment variants")

        for variant in suggestion.variants:
            time_limit = variant.automl_config.get("time_limit", 300)
            presets = variant.automl_config.get("presets", "medium_quality")
            self.logger.thought(f"Variant '{variant.name}': {presets} preset, {time_limit}s time limit")

        self.logger.summary(
            f"Experiment design complete. Recommended: '{suggestion.recommended_variant}'. "
            f"Estimated total time: {suggestion.estimated_total_time_minutes} minutes."
        )

        # Store config for dataset specs
        await self._store_design_config(suggestion, primary_metric)

        # Generate experiment goals
        experiment_family_goals = _generate_experiment_family_goals(
            task_type, primary_metric, context_analysis, time_budget_minutes
        )

        return {
            "variants": [v.model_dump() for v in suggestion.variants],
            "recommended_variant": suggestion.recommended_variant,
            "reasoning": suggestion.reasoning,
            "estimated_total_time_minutes": suggestion.estimated_total_time_minutes,
            "split_strategy": {
                "type": context_analysis.get("recommended_split_type", "stratified"),
                "reasoning": context_analysis.get("split_type_reasoning", ""),
                "is_time_based": context_analysis.get("is_time_based", False),
                "time_column": context_analysis.get("time_column"),
                "entity_id_column": context_analysis.get("entity_id_column"),
            },
            "experiment_family_goals": experiment_family_goals,
            "context_analysis": {
                "leakage_features_to_drop": context_analysis.get("leakage_features_to_drop", []),
                "prior_warnings": context_analysis.get("prior_warnings", []),
                "should_run_sanity_check": context_analysis.get("should_run_sanity_check", False),
                "expected_metric_range": context_analysis.get("expected_metric_range"),
            },
            "context_factors_used": context_factors,
        }

    async def _build_context(self, project_id, research_cycle_id):
        """Build task context for experiment design."""
        task_context = None
        task_hints = {}
        task_context_str = ""
        context_factors = {}

        if project_id:
            try:
                task_context = build_task_context(
                    db=self.db,
                    project_id=str(project_id),
                    research_cycle_id=str(research_cycle_id) if research_cycle_id else None,
                    include_leakage_candidates=True,
                    include_past_cycles=True,
                    max_experiments=5,
                )
                task_hints = get_task_type_hints(task_context)
                task_context_str = format_context_for_prompt(
                    task_context,
                    include_sections=["project", "dataset_spec", "baselines", "robustness", "leakage_candidates"],
                    max_length=3000,
                )
                if task_context_str:
                    self.logger.info("📊 Built unified TaskContext for experiment design")

                # Log context usage
                from app.services.agents.setup.problem_understanding import _log_context_usage
                context_factors = _log_context_usage(
                    step_logger=self.logger,
                    task_context=task_context,
                    task_hints=task_hints,
                    step_name="Experiment Design",
                )

            except Exception as e:
                self.logger.warning(f"Could not build TaskContext: {e}")

        return task_context, task_hints, task_context_str, context_factors

    def _build_prompt(
        self, task_type, target_column, primary_metric, feature_columns,
        row_count, time_budget_minutes, description, target_stats,
        task_context_str, task_hints, context_analysis
    ) -> str:
        """Build the user prompt for experiment design."""
        # Build audit context
        audit_context = ""
        audit_critical_issues = self.get_input("critical_issues", [])
        audit_warnings = self.get_input("warnings", [])
        audit_details = self.get_input("audit_details", {})

        if audit_critical_issues or audit_warnings:
            audit_context = "\n\n## 🔍 DATA AUDIT FINDINGS\n"
            if audit_critical_issues:
                audit_context += "\n### ⚠️ CRITICAL ISSUES:\n"
                for issue in audit_critical_issues:
                    audit_context += f"- {issue}\n"
            if audit_warnings:
                audit_context += "\n### ⚡ WARNINGS:\n"
                for warning in audit_warnings[:5]:
                    audit_context += f"- {warning}\n"

        # Base prompt
        user_prompt = get_experiment_plan_prompt(
            task_type=task_type,
            target_column=target_column,
            primary_metric=primary_metric,
            feature_count=len(feature_columns),
            row_count=row_count,
            time_budget_minutes=time_budget_minutes,
            description=description,
            feature_columns=feature_columns,
            target_stats=target_stats,
            project_history_context=None,
        )

        if audit_context:
            user_prompt += audit_context

        if task_context_str:
            user_prompt += "\n\n## 📊 PROJECT CONTEXT\n"
            user_prompt += task_context_str
            if task_hints.get("is_time_based"):
                user_prompt += "\n\n**TIME-BASED TASK**: Use time-based validation strategy.\n"

        # Add context analysis guidance
        if context_analysis:
            user_prompt += "\n\n## 🎯 EXPERIMENT DESIGN GUIDANCE\n"
            user_prompt += f"**Recommended Split Type**: `{context_analysis['recommended_split_type']}`\n"
            user_prompt += f"**Reasoning**: {context_analysis.get('split_type_reasoning', 'N/A')}\n"

            if context_analysis.get("is_time_based"):
                user_prompt += "\n**CRITICAL**: This is a TIME-BASED task. Use time-based splits.\n"

        # Handle revision requests
        is_revision = self.get_input("revision_request", False)
        critic_feedback = self.get_input("critic_feedback", "")

        if is_revision and critic_feedback:
            revision_number = self.get_input("revision_number", 0)
            self.logger.info(f"🔄 This is a plan revision (attempt {revision_number}) based on Critic feedback")
            user_prompt += f"\n\n{critic_feedback}\n"
            user_prompt += "\n**CRITICAL**: This is a revision request. Address the Critic's concerns.\n"

        return user_prompt

    async def _execute_with_tools(self, project_id, research_cycle_id, messages) -> ExperimentPlanSuggestion:
        """Execute with tool support."""
        self.logger.thinking("Agent will query project history via tools...")

        tool_executor = AgentToolExecutor(
            db=self.db,
            project_id=project_id,
            current_cycle_id=research_cycle_id,
        )

        max_validation_retries = 2

        for attempt in range(max_validation_retries + 1):
            retry_label = '(retry)' if attempt > 0 else ''
            self.logger.action(f"Consulting LLM for experiment variants (with tool access)... {retry_label}")

            result = await execute_with_tools(
                client=self.llm,
                messages=messages,
                tool_executor=tool_executor,
                response_schema=ExperimentPlanSuggestion,
                step_logger=self.logger,
            )

            result = _fill_experiment_design_defaults(result, self.logger)

            try:
                return ExperimentPlanSuggestion(**result)
            except Exception as e:
                self.logger.warning(f"Validation error (attempt {attempt + 1}): {str(e)[:500]}")

                if attempt < max_validation_retries:
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(result, default=str)[:2000]
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Validation errors:\n{str(e)[:1000]}\nPlease fix these errors."
                    })
                else:
                    raise ValueError(f"Failed to generate valid experiment design: {e}")

        raise ValueError("Failed to generate valid experiment design")

    async def _store_design_config(self, suggestion: ExperimentPlanSuggestion, primary_metric: str):
        """Store experiment design config for dataset specs."""
        try:
            agent_run = self.db.query(AgentRun).filter(AgentRun.id == self.step.agent_run_id).first()
            if not agent_run or not agent_run.project_id:
                return

            dataset_specs = (
                self.db.query(DatasetSpec)
                .filter(DatasetSpec.project_id == agent_run.project_id)
                .all()
            )

            experiment_design_config = {
                "step_id": str(self.step.id),
                "agent_run_id": str(agent_run.id),
                "variants": [v.model_dump() for v in suggestion.variants],
                "recommended_variant": suggestion.recommended_variant,
                "primary_metric": primary_metric,
                "natural_language_summary": suggestion.reasoning or "",
                "stored_at": datetime.utcnow().isoformat(),
                "source_type": "initial",
                "parent_experiment_id": None,
            }

            stored_count = 0
            for spec in dataset_specs:
                if not spec.agent_experiment_design_json:
                    spec.agent_experiment_design_json = experiment_design_config
                    stored_count += 1

            if stored_count > 0:
                self.db.commit()
                self.logger.info(f"Stored experiment design config for {stored_count} dataset spec(s)")

        except Exception as e:
            self.logger.warning(f"Failed to store experiment design config: {e}")

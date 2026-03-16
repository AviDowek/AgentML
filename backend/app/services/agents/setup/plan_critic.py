"""Plan Critic Agent - Reviews and validates experiment plans.

This agent reviews the overall plan and provides feedback,
checking for issues like data leakage, inappropriate splits,
and other problems.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.task_context import (
    build_task_context,
    get_task_type_hints,
)


def _validate_plan_against_context(
    input_data: Dict[str, Any],
    task_context: Optional[Dict[str, Any]],
    task_hints: Dict[str, Any],
    step_logger,
) -> Dict[str, Any]:
    """Validate plan against task context."""
    validation_result = {
        "split_validation": {"valid": True, "warnings": [], "required_changes": []},
        "metric_validation": {"valid": True, "warnings": [], "required_changes": []},
        "leakage_validation": {"valid": True, "warnings": [], "required_changes": []},
        "overall_valid": True,
    }

    feature_columns = set(input_data.get("feature_columns", []))
    variants = input_data.get("variants", [])
    is_time_based = input_data.get("is_time_based", False)
    time_column = input_data.get("time_column")

    # Override from task_context
    if task_context:
        dataset_spec = task_context.get("dataset_spec") or {}
        if dataset_spec.get("is_time_based"):
            is_time_based = True
            time_column = time_column or dataset_spec.get("time_column")

    if task_hints.get("is_time_based"):
        is_time_based = True

    # Split validation for time-based tasks
    if is_time_based:
        for variant in variants:
            val_strategy = variant.get("validation_strategy", {})
            split_type = val_strategy.get("split_strategy", "random")
            reasoning = val_strategy.get("reasoning", "")

            if split_type in ("random", "stratified", "group_random"):
                override_flag = val_strategy.get("time_split_override", False)
                has_justification = reasoning and len(reasoning.strip()) > 20

                if override_flag and has_justification:
                    validation_result["split_validation"]["warnings"].append(
                        f"Variant '{variant.get('name')}' uses '{split_type}' split on time-based data "
                        f"with explicit override."
                    )
                elif not has_justification:
                    validation_result["split_validation"]["valid"] = False
                    validation_result["split_validation"]["required_changes"].append({
                        "variant": variant.get("name"),
                        "issue": "random_split_on_time_data",
                        "current_value": split_type,
                        "required_action": f"Use 'time' or 'group_time' split for time-based task with time_column='{time_column}'",
                    })
                    validation_result["overall_valid"] = False

    # Leakage validation
    if task_context:
        leakage_candidates = task_context.get("leakage_candidates", [])
        if leakage_candidates:
            high_severity = [lc for lc in leakage_candidates if lc.get("severity") == "high"]
            for lc in high_severity:
                col = lc.get("column")
                if col in feature_columns:
                    validation_result["leakage_validation"]["valid"] = False
                    validation_result["leakage_validation"]["required_changes"].append({
                        "column": col,
                        "severity": "high",
                        "reason": lc.get("reason", "Potential data leakage"),
                        "required_action": f"Remove '{col}' from features or run ablation experiment",
                    })
                    validation_result["overall_valid"] = False

            medium_severity = [lc for lc in leakage_candidates if lc.get("severity") == "medium"]
            for lc in medium_severity:
                col = lc.get("column")
                if col in feature_columns:
                    validation_result["leakage_validation"]["warnings"].append(
                        f"Feature '{col}' flagged as potential leakage (medium severity)"
                    )

    return validation_result


def _generate_plan_summary(
    approved: bool,
    issues: List[str],
    warnings: List[str],
    required_changes: List[Dict],
    feature_count: int,
    variant_count: int,
    context_validation: Optional[Dict],
) -> str:
    """Generate natural language summary of plan review."""
    parts = []

    if approved:
        parts.append(f"✅ Plan APPROVED with {variant_count} experiment variant(s) and {feature_count} features.")
    else:
        parts.append(f"❌ Plan REQUIRES REVISION. {len(required_changes)} issue(s) must be addressed.")

    if required_changes:
        change_types = {}
        for change in required_changes:
            issue_type = change.get("issue", "unknown")
            change_types[issue_type] = change_types.get(issue_type, 0) + 1
        change_summary = ", ".join(f"{count} {issue_type.replace('_', ' ')}" for issue_type, count in change_types.items())
        parts.append(f"Required changes: {change_summary}.")

    if warnings:
        parts.append(f"{len(warnings)} warning(s) to consider:")
        for warning in warnings[:3]:
            if len(warning) > 100:
                warning = warning[:100] + "..."
            parts.append(f"  • {warning}")
        if len(warnings) > 3:
            parts.append(f"  • ... and {len(warnings) - 3} more")

    return "\n".join(parts)


class JustificationEval(BaseModel):
    """Evaluation result for split justification."""
    verdict: str = Field(description="VALID, WEAK, or INVALID")
    explanation: str = Field(description="1-2 sentence explanation")
    suggested_action: str = Field(description="What the data scientist should do")


class PlanCriticAgent(BaseAgent):
    """Reviews and validates experiment plans.

    Input JSON:
        - task_type: The ML task type
        - target_column: The target column
        - feature_columns: Selected features
        - variants: Experiment variants
        - is_time_based: Whether task is time-based
        - time_column: Time column if time-based
        - entity_id_column: Entity ID column if applicable
        - critical_issues: From audit (optional)
        - warnings: From audit (optional)
        - audit_details: Audit details (optional)
        - high_null_columns: High null columns (optional)
        - constant_columns: Constant columns (optional)
        - potential_id_columns: Potential ID columns (optional)

    Output:
        - approved: Whether plan is approved
        - warnings: List of warnings
        - required_changes: Required changes
        - natural_language_summary: Summary for user
        - status: "approved" or "needs_revision"
        - issues: List of issues
        - feature_count: Number of features
        - variant_count: Number of variants
        - context_validation: Validation results
        - context_factors_used: Context factors used
        - revision_feedback: Feedback for revision (if not approved)
    """

    name = "plan_critic"
    step_type = AgentStepType.PLAN_CRITIC

    async def execute(self) -> Dict[str, Any]:
        """Execute plan review."""
        self.logger.info("Reviewing the experiment plan...")

        feature_columns = self.get_input("feature_columns", [])
        variants = self.get_input("variants", [])

        # Build task context
        project_id = self.step.agent_run.project_id if self.step.agent_run else None
        research_cycle_id = getattr(self.step.agent_run, 'research_cycle_id', None) if self.step.agent_run else None

        task_context, task_hints, context_factors = self._build_context(project_id, research_cycle_id)

        # Run context validation
        context_validation = _validate_plan_against_context(
            input_data=self.input_data,
            task_context=task_context,
            task_hints=task_hints,
            step_logger=self.logger,
        )

        # Initialize tracking
        issues = []
        warnings = []
        required_changes = []
        revision_feedback = []
        approved = True

        # Process context validation
        if not context_validation.get("overall_valid", True):
            approved = False

        for validation_type in ["split_validation", "leakage_validation", "metric_validation"]:
            val_result = context_validation.get(validation_type, {})
            for change in val_result.get("required_changes", []):
                required_changes.append(change)
                change_desc = change.get("required_action") or change.get("issue", "Unknown issue")
                issues.append(f"[{validation_type}] {change_desc}")
            for warning in val_result.get("warnings", []):
                if warning not in warnings:
                    warnings.append(warning)

        # Check audit findings
        audit_critical_issues = self.get_input("critical_issues", [])
        audit_warnings = self.get_input("warnings", [])
        audit_details = self.get_input("audit_details", {})
        high_null_columns = self.get_input("high_null_columns", [])
        constant_columns = self.get_input("constant_columns", [])
        potential_id_columns = self.get_input("potential_id_columns", [])

        if audit_critical_issues or audit_warnings:
            self.logger.info(f"📋 Reviewing plan against {len(audit_critical_issues)} critical issues")

        # Check constant columns
        if constant_columns:
            included_constants = [c for c in constant_columns if c in feature_columns]
            if included_constants:
                issues.append(f"Plan includes constant columns that should be removed: {included_constants}")
                approved = False
                self.logger.error(f"⚠️ Constant columns in features: {included_constants}")

        # Check high null columns
        if high_null_columns:
            included_high_null = [c for c in high_null_columns if c in feature_columns]
            if included_high_null:
                warnings.append(f"Some features have high null rates (>30%): {included_high_null}")
                self.logger.warning(f"⚡ High-null columns in features: {included_high_null}")

        # Check ID columns
        if potential_id_columns:
            included_ids = [c for c in potential_id_columns if c in feature_columns]
            if included_ids:
                warnings.append(f"Potential ID columns included as features: {included_ids}")
                self.logger.warning(f"⚡ Potential ID columns in features: {included_ids}")

        # Check class imbalance
        if audit_details.get("class_imbalance"):
            imbalance = audit_details["class_imbalance"]
            # Handle both dict and string formats for class_imbalance
            if isinstance(imbalance, dict):
                if imbalance.get("severity") == "severe":
                    warnings.append(f"Severe class imbalance ({imbalance.get('ratio', 'N/A')}:1)")
                    self.logger.warning(f"⚡ Severe class imbalance: {imbalance.get('ratio')}:1")
            elif isinstance(imbalance, str) and "severe" in imbalance.lower():
                warnings.append(f"Severe class imbalance: {imbalance}")
                self.logger.warning(f"⚡ Severe class imbalance: {imbalance}")

        # Check leakage from task hints
        if task_hints:
            for warning in task_hints.get("leakage_warnings", [])[:3]:
                if warning not in warnings:
                    warnings.append(f"TaskContext: {warning}")

        # Check leakage candidates from context
        if task_context and task_context.get("leakage_candidates"):
            leakage_candidates = task_context["leakage_candidates"]
            high_severity = [lc for lc in leakage_candidates if lc.get("severity") == "high"]
            if high_severity:
                leaky_in_features = [lc["column"] for lc in high_severity if lc["column"] in feature_columns]
                if leaky_in_features:
                    issues.append(f"HIGH-SEVERITY LEAKAGE CANDIDATES in features: {leaky_in_features}")
                    approved = False
                    self.logger.error(f"🚨 High-severity leakage candidates: {leaky_in_features}")

        # Basic validity checks
        self.logger.thought(f"Checking {len(feature_columns)} features and {len(variants)} variants...")

        if not feature_columns:
            issues.append("No features selected for training")
            approved = False
            self.logger.error("Critical: No features selected!")

        if not variants:
            issues.append("No experiment variants defined")
            approved = False
            self.logger.error("Critical: No experiment variants!")

        if len(feature_columns) < 2:
            warnings.append("Very few features selected - model may have limited predictive power")
            self.logger.warning("Only 1-2 features selected")

        # COLUMN VALIDATION: Check all referenced columns exist
        column_validation = self._validate_column_references(feature_columns, variants)
        if not column_validation["valid"]:
            approved = False
            for error in column_validation["errors"]:
                issues.append(error)
                self.logger.error(f"Column validation: {error}")
            required_changes.append({
                "issue": "invalid_column_references",
                "missing_columns": column_validation["missing_columns"],
                "required_action": "Fix column references - use only available columns or define engineered features properly",
            })
        for warning in column_validation.get("warnings", []):
            warnings.append(warning)
            self.logger.warning(f"Column validation: {warning}")

        # Time-based split validation
        is_time_based = self.get_input("is_time_based", False)
        time_column = self.get_input("time_column")
        entity_id_column = self.get_input("entity_id_column")

        if is_time_based:
            self.logger.info(f"📅 Time-based task detected (time_column: {time_column})")
            approved, revision_feedback = await self._validate_time_splits(
                variants, time_column, entity_id_column, issues, warnings, approved, revision_feedback
            )

        # Check variant configurations
        for variant in variants:
            config = variant.get("automl_config", {})
            time_limit = config.get("time_limit", 0)
            if time_limit < 30:
                warnings.append(f"Variant '{variant.get('name')}' has very short time limit ({time_limit}s)")

        # Merge revision feedback
        for feedback in revision_feedback:
            required_changes.append({
                "variant": feedback.get("variant"),
                "issue": feedback.get("issue"),
                "current_value": feedback.get("current_split"),
                "required_action": feedback.get("suggested_action") or feedback.get("suggestion"),
                "evaluation": feedback.get("evaluation"),
            })

        status = "approved" if approved else "needs_revision"
        self.logger.summary(f"Plan review complete. Status: {status}. {len(issues)} issues, {len(warnings)} warnings.")

        # Generate summary
        natural_language_summary = _generate_plan_summary(
            approved=approved,
            issues=issues,
            warnings=warnings,
            required_changes=required_changes,
            feature_count=len(feature_columns),
            variant_count=len(variants),
            context_validation=context_validation,
        )

        result = {
            "approved": approved,
            "warnings": warnings,
            "required_changes": required_changes,
            "natural_language_summary": natural_language_summary,
            "status": status,
            "issues": issues,
            "feature_count": len(feature_columns),
            "variant_count": len(variants),
            "context_validation": {
                "split_valid": context_validation.get("split_validation", {}).get("valid", True),
                "metric_valid": context_validation.get("metric_validation", {}).get("valid", True),
                "leakage_valid": context_validation.get("leakage_validation", {}).get("valid", True),
            },
            "context_factors_used": context_factors,
        }

        if not approved and revision_feedback:
            result["revision_feedback"] = revision_feedback
            self.logger.info(f"📝 Plan requires revision. {len(required_changes)} issue(s) need to be addressed.")

        return result

    def _build_context(self, project_id, research_cycle_id):
        """Build task context for plan critique."""
        task_context = None
        task_hints = {}
        context_factors = {}

        if project_id:
            try:
                task_context = build_task_context(
                    db=self.db,
                    project_id=str(project_id),
                    research_cycle_id=str(research_cycle_id) if research_cycle_id else None,
                    include_leakage_candidates=True,
                    include_past_cycles=True,
                )
                task_hints = get_task_type_hints(task_context)
                self.logger.info("📊 Built unified TaskContext for plan review")

                # Log context usage
                from app.services.agents.setup.problem_understanding import _log_context_usage
                context_factors = _log_context_usage(
                    step_logger=self.logger,
                    task_context=task_context,
                    task_hints=task_hints,
                    step_name="Plan Critic",
                )
            except Exception as e:
                self.logger.warning(f"Could not build TaskContext: {e}")

        return task_context, task_hints, context_factors

    async def _validate_time_splits(
        self, variants, time_column, entity_id_column, issues, warnings, approved, revision_feedback
    ):
        """Validate time-based splits in variants."""
        variants_needing_justification = []

        for variant in variants:
            val_strategy = variant.get("validation_strategy", {})
            split_type = val_strategy.get("split_strategy", "random")
            reasoning = val_strategy.get("reasoning", "")

            if split_type in ("random", "stratified", "group_random"):
                self.logger.info(f"Variant '{variant.get('name')}' uses '{split_type}' split on time-based task")

                if not reasoning or len(reasoning.strip()) < 20:
                    issues.append(
                        f"⚠️ Variant '{variant.get('name')}' uses '{split_type}' split on time-based data "
                        f"but provides no justification."
                    )
                    revision_feedback.append({
                        "variant": variant.get("name"),
                        "issue": "random_split_on_time_data",
                        "current_split": split_type,
                        "suggestion": "Use 'time' or 'group_time' split, or provide justification",
                        "requires_response": True
                    })
                    approved = False
                    self.logger.warning(f"❌ Variant '{variant.get('name')}': No justification for '{split_type}'")
                else:
                    variants_needing_justification.append({
                        "variant_name": variant.get("name"),
                        "split_type": split_type,
                        "reasoning": reasoning,
                        "time_column": time_column,
                        "entity_id_column": entity_id_column,
                    })

            elif split_type in ("time", "group_time", "temporal"):
                vs_time_col = val_strategy.get("time_column")
                if not vs_time_col and not time_column:
                    warnings.append(f"Variant '{variant.get('name')}' uses '{split_type}' but no time_column specified")

                if split_type == "group_time":
                    vs_entity_col = val_strategy.get("entity_id_column") or val_strategy.get("group_column")
                    if not vs_entity_col and not entity_id_column:
                        warnings.append(f"Variant '{variant.get('name')}' uses 'group_time' but no entity_id_column")

        # Evaluate justifications with LLM
        if variants_needing_justification:
            self.logger.info(f"📝 Evaluating {len(variants_needing_justification)} justification(s)...")

            for var_info in variants_needing_justification:
                eval_result = await self._evaluate_justification(var_info)

                if eval_result.verdict == "VALID":
                    warnings.append(
                        f"✓ Variant '{var_info['variant_name']}' justification accepted: {eval_result.explanation}"
                    )
                elif eval_result.verdict == "WEAK":
                    issues.append(
                        f"⚠️ Variant '{var_info['variant_name']}': Weak justification. {eval_result.explanation}"
                    )
                    revision_feedback.append({
                        "variant": var_info["variant_name"],
                        "issue": "weak_justification",
                        "evaluation": eval_result.explanation,
                        "suggested_action": eval_result.suggested_action,
                        "requires_response": True
                    })
                    approved = False
                else:  # INVALID
                    issues.append(
                        f"❌ Variant '{var_info['variant_name']}': Invalid justification. {eval_result.explanation}"
                    )
                    revision_feedback.append({
                        "variant": var_info["variant_name"],
                        "issue": "invalid_justification",
                        "evaluation": eval_result.explanation,
                        "suggested_action": eval_result.suggested_action,
                        "requires_response": True
                    })
                    approved = False

        return approved, revision_feedback

    def _validate_column_references(
        self, feature_columns: List[str], variants: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate that all column references in the plan actually exist.

        Checks:
        1. Target column exists (raw or engineered)
        2. All feature columns exist (raw or engineered)
        3. Engineered features have valid source columns
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_columns": [],
        }

        # Get available columns from schema_summary or validation output
        available_columns = set(self.get_input("available_columns", []))
        schema_summary = self.get_input("schema_summary")

        if not available_columns and schema_summary:
            columns = schema_summary.get("columns", [])
            available_columns = set(
                c.get("name") or c.get("column_name") or c
                for c in columns
                if isinstance(c, dict) or isinstance(c, str)
            )

        if not available_columns:
            result["warnings"].append(
                "Could not determine available columns - skipping column validation"
            )
            return result

        self.logger.thinking(f"Validating columns against {len(available_columns)} available columns")

        # Get engineered features from variants
        all_engineered_outputs = set()
        all_engineered_sources = set()

        for variant in variants:
            engineered_features = variant.get("engineered_features", [])
            for feat in engineered_features:
                if isinstance(feat, dict):
                    output_col = feat.get("output_column")
                    if output_col:
                        all_engineered_outputs.add(output_col)
                    source_cols = feat.get("source_columns", [])
                    all_engineered_sources.update(source_cols)

        # Also check top-level engineered_features
        top_level_engineered = self.get_input("engineered_features", [])
        for feat in top_level_engineered:
            if isinstance(feat, dict):
                output_col = feat.get("output_column")
                if output_col:
                    all_engineered_outputs.add(output_col)
                source_cols = feat.get("source_columns", [])
                all_engineered_sources.update(source_cols)

        # Valid columns = raw columns + engineered outputs
        valid_columns = available_columns | all_engineered_outputs

        # Check target column
        target_column = self.get_input("target_column")
        if target_column and target_column not in valid_columns:
            result["valid"] = False
            result["errors"].append(
                f"Target column '{target_column}' does not exist in data and is not engineered"
            )
            result["missing_columns"].append(target_column)

        # Check feature columns
        missing_features = []
        for col in feature_columns:
            if col not in valid_columns:
                missing_features.append(col)

        if missing_features:
            result["valid"] = False
            result["errors"].append(
                f"{len(missing_features)} feature column(s) do not exist: {missing_features[:5]}"
                + ("..." if len(missing_features) > 5 else "")
            )
            result["missing_columns"].extend(missing_features)

        # Check source columns for engineered features
        missing_sources = []
        for src_col in all_engineered_sources:
            if src_col not in available_columns:
                missing_sources.append(src_col)

        if missing_sources:
            result["valid"] = False
            result["errors"].append(
                f"Engineered features reference missing source columns: {missing_sources[:5]}"
                + ("..." if len(missing_sources) > 5 else "")
            )
            result["missing_columns"].extend(missing_sources)

        if result["valid"]:
            self.logger.info(f"✅ All {len(feature_columns)} column references validated")
            if all_engineered_outputs:
                result["warnings"].append(
                    f"{len(all_engineered_outputs)} engineered columns will be created: "
                    f"{list(all_engineered_outputs)[:3]}{'...' if len(all_engineered_outputs) > 3 else ''}"
                )

        return result

    async def _evaluate_justification(self, var_info: Dict) -> JustificationEval:
        """Evaluate a justification for non-standard split."""
        eval_prompt = f"""You are a senior ML engineer reviewing an experiment plan.

The data scientist proposes using a '{var_info['split_type']}' split strategy for a TIME-BASED prediction task.

Time-based context:
- Time column: {var_info['time_column'] or 'not specified'}
- Entity ID column: {var_info['entity_id_column'] or 'not specified'}

Their justification:
"{var_info['reasoning']}"

IMPORTANT: Random/stratified splits on time-series data typically cause DATA LEAKAGE.

However, valid exceptions exist:
1. Cross-sectional data at a single point in time
2. When time independence can be verified
3. Specific experimental designs

Evaluate the justification:
A) VALID - Legitimate reason why random splits won't cause leakage
B) WEAK - Doesn't address the leakage concern adequately
C) INVALID - Wrong or irrelevant

Respond with JSON: {{"verdict": "VALID"|"WEAK"|"INVALID", "explanation": "...", "suggested_action": "..."}}"""

        try:
            eval_result = await self.llm.structured_completion(
                messages=[{"role": "user", "content": eval_prompt}],
                response_model=JustificationEval,
                temperature=0.3,
            )

            self.logger.info(f"Justification evaluation for '{var_info['variant_name']}': {eval_result.verdict}")
            return eval_result

        except Exception as e:
            self.logger.warning(f"Failed to evaluate justification: {e}")
            return JustificationEval(
                verdict="INVALID",
                explanation="Could not evaluate justification",
                suggested_action="Use 'time' or 'group_time' split for time-based data"
            )

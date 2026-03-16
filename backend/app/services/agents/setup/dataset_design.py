"""Dataset Design Agent - Generates feature column variants.

This agent determines which columns to use as features with multiple variants.
It uses project history context to learn from previous experiments and findings.
"""

import json
from typing import Any, Dict, List, Optional

from app.models import AgentRun
from app.models import AgentStepType
from app.schemas.agent import SchemaSummary, DatasetDesignSuggestion
from app.services.agents.base import BaseAgent
from app.services.agent_service import (
    generate_dataset_design,
    _format_schema_for_prompt,
)
from app.services.prompts import (
    SYSTEM_ROLE_DATASET_DESIGN_WITH_TOOLS,
    get_dataset_design_prompt,
)
from app.services.agent_tools import AgentToolExecutor
from app.services.agent_service import execute_with_tools
from app.services.context_builder import ContextBuilder


def _fill_dataset_design_defaults(result: Dict[str, Any], step_logger) -> Dict[str, Any]:
    """Fill in missing fields with sensible defaults."""
    if "variants" not in result:
        result["variants"] = []
    if "recommended_variant" not in result:
        if result["variants"]:
            result["recommended_variant"] = result["variants"][0].get("name", "variant_1")
        else:
            result["recommended_variant"] = "variant_1"
    if "reasoning" not in result:
        result["reasoning"] = "No specific reasoning provided."
    if "warnings" not in result:
        result["warnings"] = []

    # Fill defaults for each variant
    for i, variant in enumerate(result.get("variants", [])):
        if "name" not in variant:
            variant["name"] = f"variant_{i + 1}"
        if "description" not in variant:
            variant["description"] = f"Dataset variant {i + 1}"
        if "feature_columns" not in variant:
            variant["feature_columns"] = []
        if "excluded_columns" not in variant:
            variant["excluded_columns"] = []
        if "exclusion_reasons" not in variant:
            variant["exclusion_reasons"] = {}
        if "expected_tradeoff" not in variant:
            variant["expected_tradeoff"] = "Unknown tradeoff"
        if "train_test_split" not in variant:
            variant["train_test_split"] = 0.8
        if "suggested_filters" not in variant:
            variant["suggested_filters"] = []

    return result


class DatasetDesignAgent(BaseAgent):
    """Designs dataset variants with different feature columns.

    Input JSON:
        - schema_summary: Schema summary
        - task_type: The ML task type
        - target_column: The target column
        - description: Optional additional context
        - max_variants: Optional maximum variants to generate (default 10)
        - critical_issues: Critical issues from audit (optional)
        - warnings: Warnings from audit (optional)
        - audit_details: Audit details (optional)
        - recommendations: Audit recommendations (optional)

    Output:
        - variants: List of dataset variant configurations
        - recommended_variant: Name of recommended variant
        - reasoning: Design rationale
        - warnings: Any warnings
        - feature_columns: Legacy single-variant features
        - excluded_columns: Legacy excluded columns
        - exclusion_reasons: Legacy exclusion reasons
        - suggested_filters: Legacy suggested filters
    """

    name = "dataset_design"
    step_type = AgentStepType.DATASET_DESIGN

    async def execute(self) -> Dict[str, Any]:
        """Execute dataset design generation."""
        schema_data = self.require_input("schema_summary")
        task_type = self.require_input("task_type")
        target_column = self.require_input("target_column")
        description = self.get_input("description")
        max_variants = self.get_input("max_variants", 10)

        schema_summary = SchemaSummary(**schema_data)

        # Build audit context
        audit_context = self._build_audit_context()

        self.logger.info(f"Designing dataset variants for {task_type} task targeting '{target_column}'")
        self.logger.thinking(f"Analyzing {schema_summary.column_count} columns to generate up to {max_variants} variants...")

        # Get project context
        project_id = self.step.agent_run.project_id if self.step.agent_run else None
        research_cycle_id = getattr(self.step.agent_run, 'research_cycle_id', None) if self.step.agent_run else None

        # Fetch context documents for the project (if available and enabled)
        context_documents = ""
        use_context_docs = True  # Default to True
        context_ab_testing = False  # Default to False

        # Debug logging
        print(f"📚 DatasetDesignAgent: project_id={project_id}")
        if self.step.agent_run:
            config_json = self.step.agent_run.config_json or {}
            print(f"📚 DatasetDesignAgent: config_json={config_json}")
            use_context_docs = config_json.get("use_context_documents", True)
            context_ab_testing = config_json.get("context_ab_testing", False)
            print(f"📚 DatasetDesignAgent: use_context_docs={use_context_docs}, context_ab_testing={context_ab_testing}")
        else:
            print("📚 DatasetDesignAgent: WARNING - step.agent_run is None!")

        if project_id and use_context_docs:
            context_builder = ContextBuilder(self.db)
            context_documents = context_builder.build_context_section(project_id)
            if context_documents:
                if context_ab_testing:
                    self.logger.info("📚 A/B Testing enabled: Will create variants WITH and WITHOUT context documents")
                else:
                    self.logger.info("📚 Including context documents in dataset design")
            else:
                self.logger.info("📚 No active context documents found for this project")
                context_ab_testing = False  # Can't A/B test without context
        elif project_id and not use_context_docs:
            self.logger.info("📚 Context documents disabled for this pipeline")
            context_ab_testing = False

        # Format schema for prompt
        schema_text = _format_schema_for_prompt(schema_summary)

        # Build user prompt
        user_prompt = get_dataset_design_prompt(
            schema_text=schema_text,
            task_type=task_type,
            target_column=target_column,
            description=description,
            max_variants=max_variants,
            project_history_context=None,
            context_documents=context_documents,
        )

        if audit_context:
            user_prompt += audit_context
            user_prompt += "\n\n**IMPORTANT**: Your dataset variants MUST account for the audit findings above. "
            user_prompt += "Exclude problematic columns, handle class imbalance appropriately, and avoid data leakage.\n"

        # Check for validation feedback from a prior dataset_validation run
        validation_feedback = self._build_validation_feedback_context()
        if validation_feedback:
            user_prompt += validation_feedback
            self.logger.warning("Received validation feedback - fixing column references")

        # Check for feature performance validation feedback
        feature_feedback = self._build_feature_validation_feedback_context()
        if feature_feedback:
            user_prompt += feature_feedback
            self.logger.warning("Received feature validation feedback - generating better features")

        # Check for existing feature importance feedback
        existing_feedback = self._build_existing_feature_feedback_context()
        if existing_feedback:
            user_prompt += existing_feedback
            self.logger.info("Including feature importance feedback from validation")

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_DATASET_DESIGN_WITH_TOOLS},
            {"role": "user", "content": user_prompt},
        ]

        if project_id:
            suggestion = await self._execute_with_tools(
                project_id, research_cycle_id, messages, max_variants
            )
        else:
            self.logger.action("Consulting LLM for dataset design variants...")
            suggestion = await generate_dataset_design(
                client=self.llm,
                schema_summary=schema_summary,
                task_type=task_type,
                target_column=target_column,
                description=description,
                max_variants=max_variants,
                project_history_context=None,
            )

        self.logger.info(f"Generated {len(suggestion.variants)} dataset variants")

        for variant in suggestion.variants:
            self.logger.thought(
                f"Variant '{variant.name}': {len(variant.feature_columns)} features, "
                f"{variant.train_test_split} split - {variant.description[:100]}..."
            )

        if suggestion.warnings:
            for warning in suggestion.warnings:
                self.logger.warning(warning)

        # Handle A/B Testing: if enabled, create variants both WITH and WITHOUT context
        all_variants = []
        all_warnings = list(suggestion.warnings) if suggestion.warnings else []

        if context_ab_testing and context_documents:
            # Current variants were generated WITH context - add suffix
            self.logger.info("📚 A/B Testing: Adding [WITH CONTEXT] variants...")
            for variant in suggestion.variants:
                variant_dict = variant.model_dump()
                variant_dict["name"] = f"{variant.name} [WITH CONTEXT]"
                variant_dict["description"] = f"[WITH CONTEXT] {variant.description}"
                all_variants.append(variant_dict)

            # Now generate variants WITHOUT context
            self.logger.info("📚 A/B Testing: Generating [NO CONTEXT] variants...")
            no_context_prompt = get_dataset_design_prompt(
                schema_text=schema_text,
                task_type=task_type,
                target_column=target_column,
                description=description,
                max_variants=max_variants,
                project_history_context=None,
                context_documents="",  # No context
            )
            if audit_context:
                no_context_prompt += audit_context
                no_context_prompt += "\n\n**IMPORTANT**: Your dataset variants MUST account for the audit findings above. "
                no_context_prompt += "Exclude problematic columns, handle class imbalance appropriately, and avoid data leakage.\n"

            no_context_messages = [
                {"role": "system", "content": SYSTEM_ROLE_DATASET_DESIGN_WITH_TOOLS},
                {"role": "user", "content": no_context_prompt},
            ]

            if project_id:
                self.logger.action("Generating [NO CONTEXT] dataset variants...")
                no_context_suggestion = await self._execute_with_tools(
                    project_id, research_cycle_id, no_context_messages, max_variants
                )
            else:
                no_context_suggestion = await generate_dataset_design(
                    client=self.llm,
                    schema_summary=schema_summary,
                    task_type=task_type,
                    target_column=target_column,
                    description=description,
                    max_variants=max_variants,
                    project_history_context=None,
                )

            # Add NO CONTEXT variants with suffix
            for variant in no_context_suggestion.variants:
                variant_dict = variant.model_dump()
                variant_dict["name"] = f"{variant.name} [NO CONTEXT]"
                variant_dict["description"] = f"[NO CONTEXT] {variant.description}"
                all_variants.append(variant_dict)

            if no_context_suggestion.warnings:
                all_warnings.extend(no_context_suggestion.warnings)

            self.logger.info(f"📚 A/B Testing complete: {len(all_variants)} total variants ({len(suggestion.variants)} with context, {len(no_context_suggestion.variants)} without)")

            # Update recommendation to include suffix
            recommended_name = f"{suggestion.recommended_variant} [WITH CONTEXT]"

        else:
            # No A/B testing - use variants as-is
            all_variants = [v.model_dump() for v in suggestion.variants]
            recommended_name = suggestion.recommended_variant

        self.logger.summary(
            f"Dataset design complete: {len(all_variants)} variants generated, "
            f"recommended: '{recommended_name}'"
        )

        # Get recommended variant for legacy format
        recommended = next(
            (v for v in all_variants if v.get("name") == recommended_name),
            all_variants[0] if all_variants else {}
        )

        return {
            "variants": all_variants,
            "recommended_variant": recommended_name,
            "reasoning": suggestion.reasoning,
            "warnings": all_warnings,
            # Legacy format
            "feature_columns": recommended.get("feature_columns", []),
            "excluded_columns": recommended.get("excluded_columns", []),
            "exclusion_reasons": recommended.get("exclusion_reasons", {}),
            "suggested_filters": recommended.get("suggested_filters", []),
            # A/B testing metadata
            "had_context_documents": bool(context_documents),
            "context_ab_testing": context_ab_testing,
        }

    def _build_audit_context(self) -> str:
        """Build audit context string from previous step findings."""
        audit_critical_issues = self.get_input("critical_issues", [])
        audit_warnings = self.get_input("warnings", [])
        audit_details = self.get_input("audit_details", {})
        audit_recommendations = self.get_input("recommendations", [])

        if not (audit_critical_issues or audit_warnings or audit_recommendations):
            return ""

        audit_context = "\n\n## 🔍 DATA AUDIT FINDINGS (from previous analysis)\n"

        if audit_critical_issues:
            audit_context += "\n### ⚠️ CRITICAL ISSUES - MUST ADDRESS:\n"
            for issue in audit_critical_issues:
                audit_context += f"- {issue}\n"

        if audit_warnings:
            audit_context += "\n### ⚡ WARNINGS - Consider carefully:\n"
            for warning in audit_warnings:
                audit_context += f"- {warning}\n"

        if audit_details:
            if audit_details.get("class_imbalance"):
                imb = audit_details["class_imbalance"]
                if isinstance(imb, dict):
                    audit_context += f"\n### Class Imbalance: {imb.get('severity', 'unknown')} ({imb.get('ratio', 'N/A')}:1 ratio)\n"
                elif isinstance(imb, str):
                    audit_context += f"\n### Class Imbalance: {imb}\n"
            if audit_details.get("leakage_risk"):
                leak = audit_details["leakage_risk"]
                if isinstance(leak, dict):
                    audit_context += f"\n### ⚠️ POTENTIAL DATA LEAKAGE: {', '.join(leak.get('columns', []))}\n"
                    audit_context += "These columns may contain future information - consider excluding them.\n"
                elif isinstance(leak, str):
                    audit_context += f"\n### ⚠️ POTENTIAL DATA LEAKAGE: {leak}\n"
            if audit_details.get("high_null_columns"):
                # Handle both string and dict formats
                high_null_cols = audit_details['high_null_columns']
                col_names = []
                for col in high_null_cols:
                    if isinstance(col, dict):
                        col_names.append(col.get('column', col.get('name', str(col))))
                    else:
                        col_names.append(str(col))
                audit_context += f"\n### High Null Columns: {', '.join(col_names)}\n"
            if audit_details.get("potential_id_columns"):
                # Handle both string and dict formats
                id_cols = audit_details['potential_id_columns']
                col_names = []
                for col in id_cols:
                    if isinstance(col, dict):
                        col_names.append(col.get('column', col.get('name', str(col))))
                    else:
                        col_names.append(str(col))
                audit_context += f"\n### Likely ID Columns (exclude from features): {', '.join(col_names)}\n"

        if audit_recommendations:
            audit_context += "\n### 💡 RECOMMENDATIONS:\n"
            for rec in audit_recommendations[:5]:
                audit_context += f"- {rec}\n"

        self.logger.info("📋 Including Data Audit findings in dataset design context")
        return audit_context

    def _build_validation_feedback_context(self) -> str:
        """Build context from prior dataset_validation failures.

        When dataset_validation fails (columns don't exist), the PM will re-run
        this agent with the validation feedback so we can fix the column names.
        """
        # Check for validation failure feedback
        is_valid = self.get_input("is_valid")
        validation_feedback = self.get_input("feedback")
        missing_target = self.get_input("missing_target")
        missing_features = self.get_input("missing_features", [])
        available_columns = self.get_input("available_columns", [])

        # If validation passed or no feedback, nothing to add
        if is_valid is True or is_valid is None:
            return ""

        if not validation_feedback and not missing_target and not missing_features:
            return ""

        context = "\n\n## ⚠️ CRITICAL: VALIDATION FAILED - FIX REQUIRED\n"
        context += "**Your previous dataset design used column names that DON'T EXIST in the data source.**\n"
        context += "**You MUST fix this by only using columns from the AVAILABLE COLUMNS list below.**\n\n"

        if missing_target:
            context += f"### ❌ MISSING TARGET COLUMN: `{missing_target}`\n"
            context += "This target column does not exist. You must choose a target from the available columns.\n\n"

        if missing_features:
            context += f"### ❌ MISSING FEATURE COLUMNS ({len(missing_features)} total):\n"
            for col in missing_features[:20]:  # Limit to 20 for readability
                context += f"- `{col}`\n"
            if len(missing_features) > 20:
                context += f"- ... and {len(missing_features) - 20} more\n"
            context += "\nThese feature columns do not exist. Remove them or replace with valid columns.\n\n"

        if available_columns:
            context += "### ✅ AVAILABLE COLUMNS (use ONLY these):\n"
            for col in sorted(available_columns):
                context += f"- `{col}`\n"
            context += "\n"

        if validation_feedback:
            context += f"### Validation Feedback:\n{validation_feedback}\n\n"

        context += "**INSTRUCTION**: Regenerate all dataset variants using ONLY the available columns listed above.\n"
        context += "Do NOT hallucinate or invent column names. Check your column references carefully.\n"

        return context

    def _build_feature_validation_feedback_context(self) -> str:
        """Build context from feature performance validation failures.

        When engineered features fail performance validation (don't improve model),
        this provides feedback to generate better features.
        """
        # Check for feature validation failure feedback
        feature_validation_failed = self.get_input("feature_validation_failed")
        failed_features = self.get_input("failed_features", [])
        failed_features_feedback = self.get_input("failed_features_feedback")

        if not feature_validation_failed or not failed_features_feedback:
            return ""

        context = "\n\n## ⚠️ FEATURE PERFORMANCE VALIDATION FAILED\n"
        context += "**Your previous engineered features did NOT improve model performance.**\n"
        context += "**You MUST suggest DIFFERENT features that are more likely to help.**\n\n"

        context += failed_features_feedback
        context += "\n"

        return context

    def _build_existing_feature_feedback_context(self) -> str:
        """Build context from existing feature validation results.

        Provides feedback about which existing columns have high vs low importance,
        so the LLM can make better feature selections.
        """
        existing_validation = self.get_input("existing_feature_validation", {})
        if not existing_validation or not existing_validation.get("validated"):
            return ""

        importances = existing_validation.get("feature_importances", {})
        removed = existing_validation.get("removed_features", [])

        if not importances and not removed:
            return ""

        context = "\n\n## 📊 Feature Importance Feedback (from validation)\n"

        # Show top important features
        if importances:
            top_features = list(importances.items())[:10]
            context += "\n### Most Predictive Columns (PRIORITIZE these):\n"
            for feat, importance in top_features:
                context += f"- `{feat}`: {importance*100:.1f}% importance\n"

        # Show removed features
        if removed:
            context += f"\n### Low-Importance Columns ({len(removed)} removed):\n"
            for r in removed[:5]:
                context += f"- `{r.get('feature_name', '?')}`: {r.get('reason', 'low importance')}\n"
            if len(removed) > 5:
                context += f"- ... and {len(removed) - 5} more\n"
            context += "\nThese columns have very low predictive power and were filtered out.\n"

        return context

    async def _execute_with_tools(
        self,
        project_id,
        research_cycle_id,
        messages: List[Dict],
        max_variants: int,
    ) -> DatasetDesignSuggestion:
        """Execute with tool support for project context."""
        self.logger.thinking("Agent will query project history via tools to inform dataset design...")

        tool_executor = AgentToolExecutor(
            db=self.db,
            project_id=project_id,
            current_cycle_id=research_cycle_id,
        )

        max_validation_retries = 2

        for attempt in range(max_validation_retries + 1):
            retry_label = '(retry)' if attempt > 0 else ''
            self.logger.action(f"Consulting LLM for dataset design variants (with tool access)... {retry_label}")

            result = await execute_with_tools(
                client=self.llm,
                messages=messages,
                tool_executor=tool_executor,
                response_schema=DatasetDesignSuggestion,
                step_logger=self.logger,
            )

            # Truncate variants if needed
            if "variants" in result and len(result["variants"]) > max_variants:
                self.logger.warning(f"LLM returned {len(result['variants'])} variants, truncating to {max_variants}")
                result["variants"] = result["variants"][:max_variants]
                variant_names = [v.get("name") for v in result["variants"]]
                if result.get("recommended_variant") not in variant_names:
                    result["recommended_variant"] = variant_names[0] if variant_names else "variant_1"

            result = _fill_dataset_design_defaults(result, self.logger)

            try:
                return DatasetDesignSuggestion(**result)
            except Exception as e:
                self.logger.warning(f"Validation error (attempt {attempt + 1}): {str(e)[:500]}")

                if attempt < max_validation_retries:
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(result, default=str)[:2000]
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Your JSON response had validation errors:\n{str(e)[:1000]}\n\n"
                                  "Please fix these errors and provide a corrected JSON response. "
                                  "Make sure all required fields are present."
                    })
                else:
                    raise ValueError(f"Failed to generate valid dataset design after {max_validation_retries + 1} attempts: {e}")

        raise ValueError("Failed to generate valid dataset design")

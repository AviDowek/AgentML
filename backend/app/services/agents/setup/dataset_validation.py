"""Dataset Validation Agent - Validates dataset design against actual data.

This agent checks that the dataset design (target column, feature columns,
engineered features) actually matches the real columns in the data source.
This prevents AI hallucination of column names and provides feedback for
dataset design revision.

Additionally, it validates that engineered features actually improve model
performance using quick cross-validation tests.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.models import AgentStepType
from app.models.data_source import DataSource
from app.services.agents.base import BaseAgent
from app.services.dataset_validator import DatasetValidator, DatasetValidationResult
from app.services.feature_validation import (
    FeatureValidator,
    FeatureValidationResult,
    ExistingFeatureValidator,
)
from app.services.file_handlers import read_file

logger = logging.getLogger(__name__)

# Default maximum number of validated features to return
DEFAULT_MAX_SUGGESTED_FEATURES = 5


class DatasetValidationAgent(BaseAgent):
    """Validates dataset design against actual data source columns.

    Input JSON:
        - target_column: The target column name from dataset design
        - feature_columns: List of feature column names from dataset design
        - data_source_ids: List of data source UUIDs to check against (optional - will be looked up from project)
        - data_source_id: Single data source UUID (fallback)
        - schema_summary: Schema summary containing data_source_id (fallback)
        - engineered_features: Optional list of engineered feature definitions
        - variants: Optional list of dataset variants (if multiple)

    Output:
        - is_valid: Whether the dataset design is valid
        - validation_errors: List of validation errors
        - validation_warnings: List of validation warnings
        - missing_target: Target column if missing (or None)
        - missing_features: List of missing feature columns
        - missing_source_columns: List of missing source columns for engineering
        - available_columns: List of actually available columns
        - feedback: Human-readable feedback for the dataset designer
        - variants_validation: Validation results for each variant (if multiple)
    """

    name = "dataset_validation"
    step_type = AgentStepType.DATASET_VALIDATION

    def _load_sample_data(
        self,
        data_source_ids: List[str],
        max_rows: int = 10000
    ) -> Optional[pd.DataFrame]:
        """Load a sample of data for feature validation.

        Args:
            data_source_ids: List of data source UUIDs
            max_rows: Maximum rows to sample for quick validation

        Returns:
            DataFrame sample or None if loading fails
        """
        try:
            if not data_source_ids:
                return None

            # Get the first data source
            data_source = self.db.query(DataSource).filter(
                DataSource.id == data_source_ids[0]
            ).first()

            if not data_source:
                logger.warning("Data source not found")
                return None

            # Ensure file exists on disk (restore from DB if needed)
            from app.services.file_storage import ensure_file_on_disk
            try:
                file_path = str(ensure_file_on_disk(data_source))
            except ValueError:
                logger.warning("Could not restore file from DB")
                return None

            # Use file_handlers to support all file types
            df, metadata = read_file(file_path, sample_rows=max_rows)
            logger.info(f"Loaded {len(df)} rows from {file_path} (metadata: {metadata})")
            return df

        except Exception as e:
            logger.error(f"Failed to load data for validation: {e}")
            return None

    def _apply_feature_formula(
        self,
        df: pd.DataFrame,
        feature_def: Dict[str, Any]
    ) -> Optional[pd.Series]:
        """Apply a feature formula to create a new column.

        Args:
            df: Input DataFrame
            feature_def: Feature definition with formula

        Returns:
            Series with computed feature or None on error
        """
        try:
            formula = feature_def.get("formula", "")
            output_column = feature_def.get("output_column", "")

            if not formula or not output_column:
                return None

            # Create a safe evaluation context
            local_vars = {"df": df, "np": np, "pd": pd}

            # Execute the formula
            result = eval(formula, {"__builtins__": {}}, local_vars)

            if isinstance(result, pd.Series):
                return result
            elif isinstance(result, (np.ndarray, list)):
                return pd.Series(result, index=df.index)
            else:
                return None

        except Exception as e:
            logger.debug(f"Failed to apply formula for {feature_def.get('output_column', '?')}: {e}")
            return None

    def _validate_feature_performance(
        self,
        df: pd.DataFrame,
        target_column: str,
        original_features: List[str],
        engineered_features: List[Dict[str, Any]],
        task_type: str = "binary",
        max_features: int = DEFAULT_MAX_SUGGESTED_FEATURES
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate engineered features and return only those that improve performance.

        Args:
            df: DataFrame with data
            target_column: Name of target column
            original_features: List of existing feature names
            engineered_features: List of engineered feature definitions
            task_type: 'binary', 'multiclass', or 'regression'
            max_features: Maximum features to return

        Returns:
            Tuple of (validated features, validation results)
        """
        validated_features = []
        validation_results = []

        if df is None or len(engineered_features) == 0:
            return engineered_features, []

        self.logger.info(f"Validating {len(engineered_features)} engineered features...")

        # Apply each feature and test
        df_test = df.copy()
        applied_features = []

        for feat_def in engineered_features:
            output_col = feat_def.get("output_column", "")

            # Try to apply the formula
            feature_series = self._apply_feature_formula(df, feat_def)

            if feature_series is None:
                validation_results.append({
                    "feature_name": output_col,
                    "passed": False,
                    "reason": "Formula execution failed",
                    "improvement": 0.0,
                })
                continue

            # Add to test DataFrame
            df_test[output_col] = feature_series
            applied_features.append((feat_def, output_col))

        if not applied_features:
            self.logger.warning("No features could be applied for validation")
            return [], validation_results

        # Get available original features
        available_originals = [f for f in original_features if f in df_test.columns]
        if not available_originals:
            self.logger.warning("No original features available for baseline")
            return engineered_features[:max_features], validation_results

        # Run validation
        try:
            validator = FeatureValidator(task_type=task_type)
            new_feature_names = [name for _, name in applied_features]

            result = validator.validate_features(
                df=df_test,
                target_column=target_column,
                new_features=new_feature_names,
                original_features=available_originals
            )

            # Collect validated features
            for feat_def, feat_name in applied_features:
                if feat_name in result.passed_features:
                    validated_features.append(feat_def)
                    validation_results.append({
                        "feature_name": feat_name,
                        "passed": True,
                        "reason": "Improves model performance",
                        "improvement": result.improvement_percent / 100 if result.passed_features else 0,
                    })
                else:
                    # Find the failure reason
                    failed = next((f for f in result.failed_features if f.feature_name == feat_name), None)
                    validation_results.append({
                        "feature_name": feat_name,
                        "passed": False,
                        "reason": failed.reason if failed else "Did not improve performance",
                        "improvement": 0.0,
                    })

            self.logger.info(
                f"Validation complete: {len(validated_features)}/{len(engineered_features)} features passed"
            )

            if result.recommendations:
                for rec in result.recommendations[:3]:
                    self.logger.thinking(f"  {rec}")

        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            # Return features without validation on error
            return engineered_features[:max_features], validation_results

        # Limit to max_features
        if len(validated_features) > max_features:
            self.logger.info(f"Limiting to top {max_features} validated features")
            validated_features = validated_features[:max_features]

        return validated_features, validation_results

    def _get_data_source_ids(self) -> List[str]:
        """Get data source IDs from various input sources or project."""
        # Try direct input first
        data_source_ids = self.get_input("data_source_ids")
        if data_source_ids:
            return data_source_ids

        # Try single data_source_id
        data_source_id = self.get_input("data_source_id")
        if data_source_id:
            return [str(data_source_id)]

        # Try schema_summary
        schema_summary = self.get_input("schema_summary")
        if schema_summary and isinstance(schema_summary, dict):
            ds_id = schema_summary.get("data_source_id")
            if ds_id:
                return [str(ds_id)]

        # Look up from project
        project_id = None
        if self.step and self.step.agent_run:
            project_id = self.step.agent_run.project_id

        if project_id:
            self.logger.thinking("Looking up data sources from project...")
            data_sources = self.db.query(DataSource).filter(
                DataSource.project_id == project_id
            ).all()
            if data_sources:
                ids = [str(ds.id) for ds in data_sources]
                self.logger.info(f"Found {len(ids)} data source(s) from project")
                return ids

        raise ValueError(
            "Could not determine data source IDs. Please provide 'data_source_ids', "
            "'data_source_id', or 'schema_summary' with data_source_id."
        )

    async def execute(self) -> Dict[str, Any]:
        """Execute dataset validation."""
        self.logger.thinking("Validating dataset design against actual data sources...")

        # Get inputs
        target_column = self.get_input("target_column")
        feature_columns = self.get_input("feature_columns", [])
        data_source_ids = self._get_data_source_ids()
        engineered_features = self.get_input("engineered_features", [])
        variants = self.get_input("variants", [])

        # Get configurable max_suggested_features from input or use default
        max_suggested_features = self.get_input(
            "max_suggested_features",
            DEFAULT_MAX_SUGGESTED_FEATURES
        )

        # Get task type for feature validation
        task_type = self.get_input("task_type", "binary_classification")
        if "regression" in task_type.lower():
            validation_task_type = "regression"
        elif "multiclass" in task_type.lower() or "multi" in task_type.lower():
            validation_task_type = "multiclass"
        else:
            validation_task_type = "binary"

        # Check for engineered target (target_creation from problem_understanding)
        target_creation = self.get_input("target_creation")
        target_exists = self.get_input("target_exists", True)

        # If target_creation is provided, AUTOMATICALLY infer target doesn't exist
        # (the LLM may forget to set target_exists=False when providing a formula)
        if target_creation:
            target_exists = False
            self.logger.info(
                f"target_creation provided - automatically inferring target_exists=False"
            )

        # If target_creation is provided, treat target as an engineered column
        if target_creation and not target_exists and target_column:
            self.logger.info(f"Target '{target_column}' is engineered via target_creation")
            # Add target to engineered features so validator knows it will be created
            engineered_features = list(engineered_features)  # Make a copy
            engineered_features.append({
                "output_column": target_column,
                "source_columns": target_creation.get("source_columns", []),
                "formula": target_creation.get("formula", ""),
                "description": target_creation.get("description", "Engineered target"),
            })

        # Create validator
        validator = DatasetValidator(self.db)

        # If we have variants, validate each one
        if variants:
            # Pass engineered_features (which may now include target if target_creation was set)
            result = await self._validate_variants(
                validator, variants, data_source_ids, engineered_features, target_column
            )

            # Add feature performance validation for variants with engineered features
            if engineered_features and result.get("is_valid"):
                result = await self._add_feature_performance_validation(
                    result, data_source_ids, target_column, feature_columns,
                    engineered_features, validation_task_type, max_suggested_features
                )

            # Validate existing feature columns (always run if valid)
            # For variants, get feature_columns from the first valid variant
            variant_feature_columns = feature_columns
            logger.info(f"Feature validation check: is_valid={result.get('is_valid')}, "
                        f"top-level feature_columns={len(feature_columns)}, "
                        f"valid_variants={result.get('valid_variant_names', [])}")
            self.logger.info(f"Feature validation check: is_valid={result.get('is_valid')}, "
                           f"top-level feature_columns={len(feature_columns)}, "
                           f"valid_variants={result.get('valid_variant_names', [])}")

            if not variant_feature_columns and result.get("valid_variant_names"):
                # Find the first valid variant and get its feature columns
                first_valid_name = result["valid_variant_names"][0]
                for variant in variants:
                    if variant.get("name") == first_valid_name:
                        variant_feature_columns = variant.get("feature_columns", [])
                        logger.info(f"Got {len(variant_feature_columns)} feature columns from variant '{first_valid_name}'")
                        self.logger.info(f"Got {len(variant_feature_columns)} feature columns from variant '{first_valid_name}'")
                        break

            if result.get("is_valid") and variant_feature_columns:
                result = await self._validate_existing_feature_columns(
                    result, data_source_ids, target_column, variant_feature_columns,
                    validation_task_type
                )

            return result

        # Single variant validation
        self.logger.action(f"Checking {len(feature_columns)} feature columns and target '{target_column}'")
        self.logger.info(f"Validating against {len(data_source_ids)} data source(s)")

        result = validator.validate_dataset_design(
            target_column=target_column,
            feature_columns=feature_columns,
            data_source_ids=data_source_ids,
            engineered_features=engineered_features,
        )

        output = self._build_output(result, target_column, feature_columns)

        # Add feature performance validation if there are engineered features
        if engineered_features and output.get("is_valid"):
            output = await self._add_feature_performance_validation(
                output, data_source_ids, target_column, feature_columns,
                engineered_features, validation_task_type, max_suggested_features
            )

        # Validate existing feature columns (always run if valid)
        if output.get("is_valid") and feature_columns:
            output = await self._validate_existing_feature_columns(
                output, data_source_ids, target_column, feature_columns,
                validation_task_type
            )

        return output

    async def _add_feature_performance_validation(
        self,
        output: Dict[str, Any],
        data_source_ids: List[str],
        target_column: str,
        feature_columns: List[str],
        engineered_features: List[Dict[str, Any]],
        task_type: str,
        max_features: int,
    ) -> Dict[str, Any]:
        """Add feature performance validation to the output.

        Tests if engineered features actually improve model performance.
        """
        # Load sample data for validation
        sample_df = self._load_sample_data(data_source_ids)

        if sample_df is None:
            self.logger.warning("Could not load data for feature performance validation")
            output["feature_performance_validation"] = {
                "validated": False,
                "reason": "Could not load data",
                "results": [],
            }
            return output

        if target_column not in sample_df.columns:
            self.logger.warning(f"Target column '{target_column}' not in data for performance validation")
            output["feature_performance_validation"] = {
                "validated": False,
                "reason": f"Target column '{target_column}' not found in data",
                "results": [],
            }
            return output

        # Run performance validation
        self.logger.action("Validating engineered features improve model performance...")
        validated_features, validation_results = self._validate_feature_performance(
            df=sample_df,
            target_column=target_column,
            original_features=feature_columns,
            engineered_features=engineered_features,
            task_type=task_type,
            max_features=max_features,
        )

        # Log results
        passed_count = len(validated_features)
        failed_count = len(engineered_features) - passed_count

        if failed_count > 0:
            self.logger.info(
                f"Feature validation: {passed_count} passed, {failed_count} failed/rejected"
            )
            for result in validation_results:
                if not result.get("passed"):
                    self.logger.thinking(f"  Rejected {result.get('feature_name')}: {result.get('reason')}")

        # Add validation results to output
        output["feature_performance_validation"] = {
            "validated": True,
            "features_before_validation": len(engineered_features),
            "features_after_validation": len(validated_features),
            "max_suggested_features": max_features,
            "results": validation_results,
        }
        output["validated_engineered_features"] = validated_features

        return output

    async def _validate_existing_feature_columns(
        self,
        output: Dict[str, Any],
        data_source_ids: List[str],
        target_column: str,
        feature_columns: List[str],
        task_type: str,
    ) -> Dict[str, Any]:
        """Validate that selected existing feature columns actually contribute to prediction.

        Uses feature importance to identify columns that don't help the model.
        """
        if not feature_columns:
            return output

        # Load sample data
        sample_df = self._load_sample_data(data_source_ids)

        if sample_df is None:
            self.logger.warning("Could not load data for existing feature validation")
            output["existing_feature_validation"] = {
                "validated": False,
                "reason": "Could not load data",
            }
            return output

        if target_column not in sample_df.columns:
            self.logger.warning(f"Target column '{target_column}' not in data for feature validation")
            output["existing_feature_validation"] = {
                "validated": False,
                "reason": f"Target column '{target_column}' not found",
            }
            return output

        # Run existing feature validation
        self.logger.action("Validating selected feature columns have predictive power...")

        validator = ExistingFeatureValidator(task_type=task_type)
        result = validator.validate_selected_features(
            df=sample_df,
            target_column=target_column,
            feature_columns=feature_columns,
        )

        # Log results
        if result.removed_features:
            self.logger.info(
                f"Feature importance validation: {len(result.validated_features)} kept, "
                f"{len(result.removed_features)} low-importance features identified"
            )
            for removed in result.removed_features[:5]:
                self.logger.thinking(f"  Low importance: {removed.feature_name} - {removed.reason}")
        else:
            self.logger.info(f"All {len(result.validated_features)} features have predictive value")

        # Log top features
        if result.feature_importances:
            top_5 = list(result.feature_importances.items())[:5]
            top_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in top_5])
            self.logger.thinking(f"Top features by importance: {top_str}")

        # Add validation results to output
        output["existing_feature_validation"] = {
            "validated": True,
            "features_before": len(feature_columns),
            "features_after": len(result.validated_features),
            "removed_features": [
                {"feature_name": r.feature_name, "reason": r.reason}
                for r in result.removed_features
            ],
            "feature_importances": result.feature_importances,
            "baseline_score": result.baseline_all_features_score,
            "final_score": result.final_validated_score,
            "recommendations": result.recommendations,
        }
        output["validated_feature_columns"] = result.validated_features

        return output

    async def _validate_variants(
        self,
        validator: DatasetValidator,
        variants: List[Dict[str, Any]],
        data_source_ids: List[str],
        engineered_features: List[Dict[str, Any]],
        default_target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate multiple dataset variants."""
        self.logger.info(f"Validating {len(variants)} dataset variants")

        # Get available columns once
        available_columns = validator.get_available_columns(data_source_ids)

        variants_validation = []
        any_valid = False
        all_feedback = []

        for variant in variants:
            variant_name = variant.get("name", "unnamed")
            # Use variant's target_column if specified, otherwise use the default from problem_understanding
            target_column = variant.get("target_column") or default_target_column
            feature_columns = variant.get("feature_columns", [])
            # Merge variant's engineered features with top-level ones (which may include target_creation)
            variant_engineered = list(engineered_features) + variant.get("engineered_features", [])

            self.logger.thinking(f"Validating variant '{variant_name}': target='{target_column}', {len(feature_columns)} features")

            result = validator.validate_columns(
                target_column=target_column,
                feature_columns=feature_columns,
                available_columns=available_columns,
                engineered_features=variant_engineered,
            )

            variant_result = {
                "variant_name": variant_name,
                "is_valid": result.is_valid,
                "missing_target": result.missing_target,
                "missing_features": result.missing_features,
                "missing_source_columns": result.missing_source_columns,
                "presumed_engineered": result.presumed_engineered,
                "errors": result.errors,
                "warnings": result.warnings,
            }
            variants_validation.append(variant_result)

            if result.is_valid:
                any_valid = True
                if result.presumed_engineered:
                    self.logger.info(
                        f"Variant '{variant_name}': VALID "
                        f"({len(result.presumed_engineered)} columns assumed to be engineered)"
                    )
                else:
                    self.logger.info(f"Variant '{variant_name}': VALID")
            else:
                self.logger.warning(f"Variant '{variant_name}': INVALID - {len(result.missing_features)} missing features")
                all_feedback.append(f"### Variant: {variant_name}\n{result.to_feedback()}")

        # Overall result
        if any_valid:
            valid_variants = [v["variant_name"] for v in variants_validation if v["is_valid"]]
            self.logger.summary(f"Validation complete: {len(valid_variants)}/{len(variants)} variants are valid")
        else:
            self.logger.error("All dataset variants have validation errors")

        # Bubble up missing_target to top level for PM routing
        # All variants use the same target, so check if any has a missing target
        missing_target = None
        for v in variants_validation:
            if v.get("missing_target"):
                missing_target = v["missing_target"]
                break

        # Aggregate missing features across all variants
        all_missing_features = set()
        for v in variants_validation:
            all_missing_features.update(v.get("missing_features", []))

        return {
            "is_valid": any_valid,
            "all_valid": all(v["is_valid"] for v in variants_validation),
            # CRITICAL: These top-level fields trigger PM routing
            "missing_target": missing_target,  # If set, PM routes to problem_understanding
            "missing_features": list(all_missing_features),  # If set (no missing_target), PM routes to dataset_design
            "validation_errors": [
                e for v in variants_validation for e in v.get("errors", [])
            ],
            "validation_warnings": [
                w for v in variants_validation for w in v.get("warnings", [])
            ],
            "available_columns": list(available_columns),
            "variants_validation": variants_validation,
            "feedback": "\n\n".join(all_feedback) if all_feedback else "All variants are valid.",
            "valid_variant_names": [v["variant_name"] for v in variants_validation if v["is_valid"]],
            "invalid_variant_names": [v["variant_name"] for v in variants_validation if not v["is_valid"]],
        }

    def _build_output(
        self,
        result: DatasetValidationResult,
        target_column: str,
        feature_columns: List[str],
    ) -> Dict[str, Any]:
        """Build output dictionary from validation result."""
        if result.is_valid:
            msg = f"Dataset design is VALID. Target '{target_column}' and {len(feature_columns)} features verified."
            if result.presumed_engineered:
                msg += f" ({len(result.presumed_engineered)} columns assumed to be engineered)"
            self.logger.summary(msg)
        else:
            error_count = len(result.errors)
            self.logger.error(f"Dataset design is INVALID. {error_count} error(s) found.")

            if result.missing_target:
                self.logger.warning(f"Target column '{result.missing_target}' not found in data source")

            if result.missing_features:
                self.logger.warning(
                    f"{len(result.missing_features)} feature columns not found: "
                    f"{', '.join(result.missing_features[:5])}{'...' if len(result.missing_features) > 5 else ''}"
                )

            if result.missing_source_columns:
                self.logger.warning(
                    f"{len(result.missing_source_columns)} source columns for engineered features not found"
                )

            # Log available columns for debugging
            self.logger.thinking(
                f"Available columns ({len(result.available_columns)}): "
                f"{', '.join(sorted(list(result.available_columns))[:10])}..."
            )

        # Log presumed engineered features (even on success)
        if result.presumed_engineered:
            self.logger.info(
                f"Presumed engineered features: {', '.join(result.presumed_engineered[:5])}"
                + ("..." if len(result.presumed_engineered) > 5 else "")
            )

        return {
            "is_valid": result.is_valid,
            "validation_errors": result.errors,
            "validation_warnings": result.warnings,
            "missing_target": result.missing_target,
            "missing_features": result.missing_features,
            "missing_source_columns": result.missing_source_columns,
            "presumed_engineered": result.presumed_engineered,
            "available_columns": list(result.available_columns),
            "feedback": result.to_feedback(),
        }

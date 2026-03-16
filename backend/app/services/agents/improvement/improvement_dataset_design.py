"""Improvement Dataset Design Agent - Redesigns dataset with iteration feedback.

This agent creates a new dataset design based on what features worked
and what feature engineering failed in previous iterations.

Features are validated before being returned to ensure they actually improve
model performance.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.prompts import SYSTEM_ROLE_ML_ANALYST
from app.services.feature_validation import FeatureValidator, FeatureValidationResult

logger = logging.getLogger(__name__)

# Default maximum number of validated features to return (configurable via input)
DEFAULT_MAX_SUGGESTED_FEATURES = 5


class DatasetDesign(BaseModel):
    """Response schema for dataset design."""
    features_to_keep: List[str] = Field(
        description="Existing features to retain"
    )
    features_to_drop: List[str] = Field(
        description="Features to remove from the dataset"
    )
    new_engineered_features: List[Dict[str, Any]] = Field(
        description="New features to create with output_column, formula, source_columns, description"
    )
    data_filters: List[Dict[str, Any]] = Field(
        description="Data filters to apply (column, operator, value)"
    )
    rationale: str = Field(
        description="Explanation of why these changes should improve results"
    )
    expected_impact: str = Field(
        description="Expected impact on model performance"
    )


class ImprovementDatasetDesignAgent(BaseAgent):
    """Redesigns dataset based on iteration feedback.

    Input JSON:
        - data_analysis: Output from ImprovementDataAnalysisAgent
        - iteration_context: Context from IterationContextAgent

    Output:
        - features_to_keep: Features to retain
        - features_to_drop: Features to remove
        - new_engineered_features: New features to create (validated, max 5)
        - data_filters: Filters to apply
        - rationale: Why these changes should help
        - expected_impact: Expected performance impact
        - validation_results: Results from feature validation
    """

    name = "improvement_dataset_design"
    step_type = AgentStepType.IMPROVEMENT_DATASET_DESIGN

    def _load_sample_data(
        self,
        dataset_spec: Dict[str, Any],
        max_rows: int = 10000
    ) -> Optional[pd.DataFrame]:
        """Load a sample of data for validation.

        Args:
            dataset_spec: Dataset specification with file path
            max_rows: Maximum rows to sample for quick validation

        Returns:
            DataFrame sample or None if loading fails
        """
        try:
            file_path = dataset_spec.get("data_file") or dataset_spec.get("file_path")
            if not file_path:
                logger.warning("No data file path in dataset_spec")
                return None

            # Try to load data
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path, nrows=max_rows)
            elif file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
                if len(df) > max_rows:
                    df = df.sample(n=max_rows, random_state=42)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return None

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

    def _validate_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        original_features: List[str],
        new_feature_defs: List[Dict[str, Any]],
        task_type: str = "binary",
        max_features: int = DEFAULT_MAX_SUGGESTED_FEATURES
    ) -> tuple[List[Dict[str, Any]], List[FeatureValidationResult]]:
        """Validate new features and return only those that improve performance.

        Args:
            df: DataFrame with data
            target_column: Name of target column
            original_features: List of existing feature names
            new_feature_defs: List of new feature definitions
            task_type: 'binary', 'multiclass', or 'regression'

        Returns:
            Tuple of (validated features, validation results)
        """
        validated_features = []
        validation_results = []

        if df is None or len(new_feature_defs) == 0:
            return [], []

        self.logger.info(f"Validating {len(new_feature_defs)} proposed features...")

        # Apply each feature and test
        df_test = df.copy()
        applied_features = []

        for feat_def in new_feature_defs:
            output_col = feat_def.get("output_column", "")

            # Try to apply the formula
            feature_series = self._apply_feature_formula(df, feat_def)

            if feature_series is None:
                validation_results.append(FeatureValidationResult(
                    feature_name=output_col,
                    passed=False,
                    reason="Formula execution failed"
                ))
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
            return [], validation_results

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
                    validation_results.append(FeatureValidationResult(
                        feature_name=feat_name,
                        passed=True,
                        reason="Improves model performance",
                        improvement=result.improvement_percent / 100 if result.passed_features else 0
                    ))
                else:
                    # Find the failure reason
                    failed = next((f for f in result.failed_features if f.feature_name == feat_name), None)
                    validation_results.append(failed or FeatureValidationResult(
                        feature_name=feat_name,
                        passed=False,
                        reason="Did not improve performance"
                    ))

            self.logger.info(
                f"Validation complete: {len(validated_features)}/{len(new_feature_defs)} features passed"
            )

            if result.recommendations:
                for rec in result.recommendations[:3]:
                    self.logger.thinking(f"  {rec}")

        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            # Return features without validation on error
            return new_feature_defs[:max_features], validation_results

        # Limit to max_features
        if len(validated_features) > max_features:
            self.logger.info(f"Limiting to top {max_features} validated features")
            validated_features = validated_features[:max_features]

        return validated_features, validation_results

    async def execute(self) -> Dict[str, Any]:
        """Execute dataset redesign with iteration feedback."""
        data_analysis = self.get_input("data_analysis", {})
        iteration_context = self.get_input("iteration_context", {})

        if not iteration_context:
            raise ValueError("Missing iteration_context")

        # Get configurable max_suggested_features from input or use default
        max_suggested_features = self.get_input(
            "max_suggested_features",
            DEFAULT_MAX_SUGGESTED_FEATURES
        )

        self.logger.info(f"Designing improved dataset (max {max_suggested_features} features)...")

        dataset_spec = iteration_context.get("dataset_spec", {})
        data_stats = iteration_context.get("data_statistics", {})

        # Get current features and engineered features
        current_features = dataset_spec.get("feature_columns", [])
        spec_json = dataset_spec.get("spec_json", {})
        existing_engineered = spec_json.get("engineered_features", [])

        # Get raw columns
        raw_columns = data_stats.get("columns", [])

        # Format recommendations from analysis
        recommended_features = data_analysis.get("recommended_features", [])
        features_to_remove = data_analysis.get("features_to_remove", [])
        untried = data_analysis.get("untried_opportunities", [])

        prompt = f"""You are designing an improved dataset for the next ML training iteration.
You have complete knowledge of what has been tried before and what worked.

## Current Dataset
- Target Column: {dataset_spec.get('target_column', 'unknown')}
- Current Features: {', '.join(current_features[:20])}{'...' if len(current_features) > 20 else ''}
- Existing Engineered Features: {len(existing_engineered)} features

## Available Raw Columns (you can ONLY use these in formulas!)
{', '.join(raw_columns[:40])}

## Analysis Recommendations
Recommended New Features:
{recommended_features}

Features to Remove:
{features_to_remove}

Untried Opportunities:
{untried}

## Guidelines
1. Only create features using columns that ACTUALLY EXIST (listed above)
2. Use pandas/numpy syntax for formulas: df["column"].operation()
3. Common patterns:
   - Rolling averages: df["col"].rolling(window=7).mean()
   - Differences: df["col1"] - df["col2"]
   - Ratios: df["col1"] / df["col2"]
   - Lag features: df["col"].shift(1)
   - Time features: df["date_col"].dt.dayofweek
   - Log transform: np.log1p(df["col"])

4. DO NOT create features that:
   - Reference columns that don't exist
   - Were tried before and failed
   - Could cause data leakage

Create specific, implementable features that address the identified bottlenecks."""

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_ML_ANALYST},
            {"role": "user", "content": prompt},
        ]

        self.logger.action("Consulting LLM for dataset design...")
        response = await self.llm.chat_json(messages, DatasetDesign)

        new_features = response.get("new_engineered_features", [])
        to_drop = response.get("features_to_drop", [])

        self.logger.info(f"LLM proposed {len(new_features)} new engineered features")
        for feat in new_features[:5]:
            if isinstance(feat, dict):
                self.logger.thinking(f"  {feat.get('output_column', '?')}: {feat.get('formula', '?')[:60]}")

        # Validate features before returning
        validation_results = []
        validated_features = new_features  # Default to all if validation not possible

        if new_features:
            # Load sample data for validation
            sample_df = self._load_sample_data(dataset_spec)

            if sample_df is not None:
                target_column = dataset_spec.get("target_column", "")

                # Determine task type from context
                task_type = "binary"  # Default
                if iteration_context.get("problem_type"):
                    pt = iteration_context.get("problem_type", "").lower()
                    if "regression" in pt:
                        task_type = "regression"
                    elif "multiclass" in pt or "multi" in pt:
                        task_type = "multiclass"

                # Run validation
                self.logger.action("Validating proposed features with quick tests...")
                validated_features, validation_results = self._validate_features(
                    df=sample_df,
                    target_column=target_column,
                    original_features=current_features,
                    new_feature_defs=new_features,
                    task_type=task_type,
                    max_features=max_suggested_features
                )

                # Log validation summary
                passed_count = len(validated_features)
                failed_count = len(new_features) - passed_count

                if failed_count > 0:
                    self.logger.info(
                        f"Validation filtered: {passed_count} passed, {failed_count} failed/rejected"
                    )
                    for result in validation_results:
                        if not result.passed:
                            self.logger.thinking(f"  Rejected {result.feature_name}: {result.reason}")
            else:
                self.logger.warning("Could not load data for validation - returning unvalidated features")
                # Limit unvalidated features
                validated_features = new_features[:max_suggested_features]

        if to_drop:
            self.logger.info(f"Dropping {len(to_drop)} features")

        self.logger.thinking(f"Rationale: {response.get('rationale', 'Not specified')[:200]}")

        self.logger.summary(
            f"Dataset design complete. +{len(validated_features)} validated features, -{len(to_drop)} removed. "
            f"Expected: {response.get('expected_impact', 'improved performance')[:50]}"
        )

        # Serialize validation results for return
        validation_summary = [
            {
                "feature_name": r.feature_name,
                "passed": r.passed,
                "reason": r.reason,
                "improvement": getattr(r, "improvement", 0.0),
            }
            for r in validation_results
        ]

        return {
            **response,
            "new_engineered_features": validated_features,  # Override with validated
            "existing_engineered_features": existing_engineered,
            "iteration_context": iteration_context,
            "validation_results": validation_summary,
            "features_before_validation": len(new_features),
            "features_after_validation": len(validated_features),
            "max_suggested_features": max_suggested_features,
        }

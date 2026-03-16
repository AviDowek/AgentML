"""Feature Pipeline Service for transforming raw data to model-ready features.

This module provides a reusable pipeline for applying feature engineering
transformations to raw data, enabling:
1. Raw data predictions via API
2. Model export with bundled transformations
3. Testing UI with raw data input
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.feature_engineering import (
    apply_feature_engineering,
    apply_target_creation,
    _get_safe_builtins,  # FIX: Import unified builtins for consistency
)

logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineConfig:
    """Configuration for a feature transformation pipeline.

    This config can be serialized to JSON and bundled with exported models.
    """

    # Feature engineering steps (from spec_json["feature_engineering"])
    feature_engineering: List[Dict[str, Any]] = field(default_factory=list)

    # Target creation (if any - from spec_json["target_definition"]["target_creation"])
    target_creation: Optional[Dict[str, Any]] = None

    # Expected input columns (raw data columns)
    input_columns: List[str] = field(default_factory=list)

    # Expected output columns (after transformation)
    output_columns: List[str] = field(default_factory=list)

    # Target column name
    target_column: Optional[str] = None

    # Task type for context
    task_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_engineering": self.feature_engineering,
            "target_creation": self.target_creation,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "target_column": self.target_column,
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeaturePipelineConfig":
        """Create from dictionary."""
        return cls(
            feature_engineering=data.get("feature_engineering", []),
            target_creation=data.get("target_creation"),
            input_columns=data.get("input_columns", []),
            output_columns=data.get("output_columns", []),
            target_column=data.get("target_column"),
            task_type=data.get("task_type"),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "FeaturePipelineConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class FeaturePipeline:
    """Feature transformation pipeline for converting raw data to model features.

    Usage:
        # Create from dataset spec
        pipeline = FeaturePipeline.from_dataset_spec(spec_json, serving_config)

        # Transform raw data
        transformed_df = pipeline.transform(raw_df)

        # Export for bundling with model
        config_json = pipeline.export_config()
    """

    def __init__(self, config: FeaturePipelineConfig):
        """Initialize with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config

    @classmethod
    def from_dataset_spec(
        cls,
        spec_json: Optional[Dict[str, Any]],
        serving_config: Optional[Dict[str, Any]] = None,
    ) -> "FeaturePipeline":
        """Create a pipeline from a dataset specification.

        Args:
            spec_json: The dataset_spec.spec_json containing transformations
            serving_config: Optional serving config from model_version

        Returns:
            FeaturePipeline instance
        """
        if not spec_json:
            spec_json = {}

        # Extract feature engineering steps
        feature_engineering = spec_json.get("feature_engineering", [])

        # Extract target creation if defined
        target_definition = spec_json.get("target_definition", {})
        target_creation = None
        if isinstance(target_definition, dict):
            target_creation = target_definition.get("target_creation")

        # Get column info from serving config
        input_columns = []
        output_columns = []
        target_column = None
        task_type = None

        if serving_config:
            features = serving_config.get("features", [])
            output_columns = [f["name"] for f in features if isinstance(f, dict)]
            target_column = serving_config.get("target_column")
            task_type = serving_config.get("task_type")

        # Determine input columns from feature engineering sources
        for step in feature_engineering:
            source_cols = step.get("source_columns", [])
            for col in source_cols:
                if col not in input_columns:
                    input_columns.append(col)

        config = FeaturePipelineConfig(
            feature_engineering=feature_engineering,
            target_creation=target_creation,
            input_columns=input_columns,
            output_columns=output_columns,
            target_column=target_column,
            task_type=task_type,
        )

        return cls(config)

    @classmethod
    def from_model_version(cls, model_version) -> "FeaturePipeline":
        """Create a pipeline from a ModelVersion database object.

        Args:
            model_version: ModelVersion database object

        Returns:
            FeaturePipeline instance
        """
        # Get spec_json from the associated experiment's dataset_spec
        spec_json = None

        if model_version.experiment:
            experiment = model_version.experiment
            if experiment.dataset_spec:
                spec_json = experiment.dataset_spec.spec_json

        serving_config = model_version.serving_config_json

        return cls.from_dataset_spec(spec_json, serving_config)

    def transform(
        self,
        raw_df: pd.DataFrame,
        create_target: bool = False,
        strict: bool = False,
    ) -> pd.DataFrame:
        """Transform raw data to model-ready features.

        Args:
            raw_df: Raw input DataFrame with original columns
            create_target: If True, also create the target column (for training)
            strict: If True, raise on transformation errors

        Returns:
            Transformed DataFrame with engineered features
        """
        df = raw_df.copy()

        # Apply feature engineering
        if self.config.feature_engineering:
            logger.info(
                f"Applying {len(self.config.feature_engineering)} feature engineering steps"
            )
            df = apply_feature_engineering(
                df,
                self.config.feature_engineering,
                inplace=True,
                strict=strict,
            )

        # Optionally create target column
        if create_target and self.config.target_creation:
            logger.info("Creating target column")
            df = apply_target_creation(
                df,
                self.config.target_creation,
                inplace=True,
            )

        return df

    def get_required_columns(self) -> List[str]:
        """Get the list of columns required in raw input data.

        Returns:
            List of column names that must be present in raw data
        """
        required = set()

        # Add source columns from feature engineering
        for step in self.config.feature_engineering:
            source_cols = step.get("source_columns", [])
            required.update(source_cols)

        # Add target source columns if target creation is defined
        if self.config.target_creation:
            target_sources = self.config.target_creation.get("source_columns", [])
            required.update(target_sources)

        return list(required)

    def get_output_columns(self) -> List[str]:
        """Get the list of columns produced after transformation.

        Returns:
            List of output column names
        """
        return self.config.output_columns.copy()

    def validate_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that input DataFrame has required columns.

        Args:
            df: Input DataFrame to validate

        Returns:
            Dict with validation result:
            - valid: bool
            - missing_columns: list of missing column names
            - extra_columns: list of extra column names
        """
        required = set(self.get_required_columns())
        available = set(df.columns)

        missing = required - available
        extra = available - required

        return {
            "valid": len(missing) == 0,
            "missing_columns": list(missing),
            "extra_columns": list(extra),
            "required_columns": list(required),
        }

    def export_config(self) -> str:
        """Export pipeline configuration as JSON.

        Returns:
            JSON string of pipeline configuration
        """
        return self.config.to_json()

    def has_transformations(self) -> bool:
        """Check if this pipeline has any transformations defined.

        Returns:
            True if there are feature engineering steps or target creation
        """
        return bool(self.config.feature_engineering) or bool(
            self.config.target_creation
        )


def get_pipeline_for_model(
    db,
    model_id: str,
) -> Optional[FeaturePipeline]:
    """Get the feature pipeline for a model.

    Args:
        db: Database session
        model_id: Model version UUID

    Returns:
        FeaturePipeline or None if model not found
    """
    from uuid import UUID

    from app.models.model_version import ModelVersion
    from app.models.experiment import Experiment
    from app.models.dataset_spec import DatasetSpec

    # Get model
    model = db.query(ModelVersion).filter(ModelVersion.id == UUID(model_id)).first()
    if not model:
        return None

    # Try to get spec_json from experiment's dataset_spec
    spec_json = None
    if model.experiment_id:
        experiment = (
            db.query(Experiment).filter(Experiment.id == model.experiment_id).first()
        )
        if experiment and experiment.dataset_spec_id:
            dataset_spec = (
                db.query(DatasetSpec)
                .filter(DatasetSpec.id == experiment.dataset_spec_id)
                .first()
            )
            if dataset_spec:
                spec_json = dataset_spec.spec_json

    return FeaturePipeline.from_dataset_spec(spec_json, model.serving_config_json)


def generate_predict_script(
    pipeline_config: Dict[str, Any],
    model_type: str = "autogluon",
) -> str:
    """Generate a standalone predict.py script for exported models.

    Args:
        pipeline_config: Pipeline configuration dict
        model_type: Type of model (autogluon, sklearn, etc.)

    Returns:
        Python script content as string
    """
    script = '''"""Prediction script for exported model.

This script loads the trained model and applies any necessary
feature transformations before making predictions.

Usage:
    python predict.py input.csv output.csv
    python predict.py --json '{"col1": val1, "col2": val2}'
"""
import json
import sys
from pathlib import Path

import pandas as pd


# Feature engineering configuration (from training pipeline)
PIPELINE_CONFIG = '''

    script += json.dumps(pipeline_config, indent=4)

    script += '''


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations."""
    import numpy as np

    feature_steps = PIPELINE_CONFIG.get("feature_engineering", [])

    for step in feature_steps:
        output_column = step.get("output_column")
        formula = step.get("formula")
        source_columns = step.get("source_columns", [])

        if not output_column or not formula:
            continue

        # Check required columns exist
        missing = [c for c in source_columns if c not in df.columns]
        if missing:
            print(f"Warning: Missing columns {missing} for feature {output_column}, skipping")
            continue

        try:
            # Build execution context
            exec_context = {"df": df, "pd": pd, "np": np}
            exec_context.update({col: df[col] for col in df.columns})

            # Execute formula with FULL builtins for consistency
            # FIX: Previous version was missing: bool, list, dict, tuple, set, sum, min, max, abs, round, range, any, all, isinstance, type
            safe_builtins = {
                # Types
                "int": int, "float": float, "str": str, "bool": bool,
                "list": list, "dict": dict, "tuple": tuple, "set": set, "object": object,
                # Functions
                "len": len, "sum": sum, "min": min, "max": max, "abs": abs,
                "round": round, "range": range, "enumerate": enumerate, "zip": zip,
                "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
                "any": any, "all": all, "isinstance": isinstance, "type": type,
                # Libraries
                "np": np, "pd": pd,
            }
            result = eval(formula, {"__builtins__": safe_builtins}, exec_context)

            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index)

            df[output_column] = result
            print(f"Created feature: {output_column}")

        except Exception as e:
            print(f"Warning: Failed to create {output_column}: {e}")

    return df


def load_model(model_path: str):
    """Load the trained model."""
'''

    if model_type == "autogluon":
        script += '''    from autogluon.tabular import TabularPredictor
    return TabularPredictor.load(model_path)
'''
    else:
        script += '''    import joblib
    return joblib.load(model_path)
'''

    script += '''

def predict(model, df: pd.DataFrame):
    """Make predictions with the model."""
    # Apply feature transformations
    transformed_df = apply_feature_engineering(df)

    # Get expected features from config
    output_columns = PIPELINE_CONFIG.get("output_columns", [])
    target_column = PIPELINE_CONFIG.get("target_column")

    # Filter to expected columns if specified
    if output_columns:
        available = [c for c in output_columns if c in transformed_df.columns]
        if available:
            transformed_df = transformed_df[available]

    # Make predictions
'''

    if model_type == "autogluon":
        script += '''    predictions = model.predict(transformed_df)
'''
    else:
        script += '''    predictions = model.predict(transformed_df)
'''

    script += '''    return predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Make predictions with exported model")
    parser.add_argument("input", nargs="?", help="Input CSV file path")
    parser.add_argument("output", nargs="?", help="Output CSV file path")
    parser.add_argument("--json", help="JSON input for single prediction")
    parser.add_argument("--model-path", default=".", help="Path to model directory")

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path)
    print(f"Model loaded from {args.model_path}")

    if args.json:
        # Single prediction from JSON
        data = json.loads(args.json)
        df = pd.DataFrame([data])
        predictions = predict(model, df)
        print(f"Prediction: {predictions.iloc[0]}")

    elif args.input:
        # Batch prediction from CSV
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows from {args.input}")

        predictions = predict(model, df)

        if args.output:
            # Save to output file
            result_df = df.copy()
            result_df["prediction"] = predictions
            result_df.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            # Print predictions
            print("Predictions:")
            print(predictions)

    else:
        print("Error: Provide either --json for single prediction or input CSV file")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

    return script

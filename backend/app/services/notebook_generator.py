"""Jupyter Notebook Generator for ML Pipeline Reproducibility.

Generates complete, standalone Jupyter notebooks that document and reproduce
everything the agents did during the ML pipeline, including:
- Data loading and exploration
- Data audit results
- Feature engineering code
- AutoGluon configuration
- Training and evaluation
- Model predictions
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

logger = logging.getLogger(__name__)


class NotebookGenerator:
    """Generates reproducible Jupyter notebooks from ML pipeline runs."""

    def __init__(self):
        self.notebook = new_notebook()
        self.cells = []

    def generate_experiment_notebook(
        self,
        experiment_data: dict[str, Any],
        dataset_spec_data: dict[str, Any],
        feature_engineering_code: list[dict[str, str]],
        automl_config: dict[str, Any],
        metrics: dict[str, Any],
        data_audit_results: Optional[dict[str, Any]] = None,
        agent_analysis: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate a complete experiment notebook.

        Args:
            experiment_data: Experiment metadata and configuration
            dataset_spec_data: Dataset specification details
            feature_engineering_code: List of feature engineering operations
            automl_config: AutoGluon configuration used
            metrics: Training and evaluation metrics
            data_audit_results: Optional data audit findings
            agent_analysis: Optional agent analysis and recommendations

        Returns:
            Notebook as a JSON string
        """
        self.cells = []

        # Title and metadata
        self._add_title_section(experiment_data)

        # Table of contents
        self._add_toc()

        # Environment setup
        self._add_setup_section()

        # Data loading
        self._add_data_loading_section(dataset_spec_data)

        # Data audit
        if data_audit_results:
            self._add_data_audit_section(data_audit_results)

        # Exploratory data analysis
        self._add_eda_section(dataset_spec_data)

        # Feature engineering
        self._add_feature_engineering_section(feature_engineering_code, dataset_spec_data)

        # AutoGluon configuration
        self._add_automl_config_section(automl_config, experiment_data)

        # Training
        self._add_training_section(automl_config, dataset_spec_data)

        # Evaluation
        self._add_evaluation_section(metrics)

        # Predictions
        self._add_prediction_section()

        # Agent analysis (if available)
        if agent_analysis:
            self._add_agent_analysis_section(agent_analysis)

        # Conclusion
        self._add_conclusion_section(experiment_data, metrics)

        # Build notebook
        self.notebook.cells = self.cells
        return nbformat.writes(self.notebook)

    def _add_markdown(self, content: str):
        """Add a markdown cell."""
        self.cells.append(new_markdown_cell(content))

    def _add_code(self, code: str):
        """Add a code cell."""
        self.cells.append(new_code_cell(code))

    def _add_title_section(self, experiment_data: dict):
        """Add title and metadata section."""
        exp_name = experiment_data.get("name", "Unnamed Experiment")
        created_at = experiment_data.get("created_at", datetime.now().isoformat())
        iteration = experiment_data.get("iteration_number", 1)

        self._add_markdown(f"""# {exp_name}

## ML Pipeline Reproducibility Notebook

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Experiment ID**: `{experiment_data.get("id", "N/A")}`
**Iteration**: {iteration}
**Status**: {experiment_data.get("status", "unknown")}

---

This notebook contains the complete, reproducible code for this ML experiment.
You can run this notebook standalone to replicate the entire pipeline.
""")

    def _add_toc(self):
        """Add table of contents."""
        self._add_markdown("""## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Loading](#2-data-loading)
3. [Data Audit](#3-data-audit)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Feature Engineering](#5-feature-engineering)
6. [AutoGluon Configuration](#6-autogluon-configuration)
7. [Model Training](#7-model-training)
8. [Evaluation](#8-evaluation)
9. [Making Predictions](#9-making-predictions)
10. [Conclusion](#10-conclusion)

---
""")

    def _add_setup_section(self):
        """Add environment setup section."""
        self._add_markdown("""## 1. Environment Setup

First, let's install and import the required packages.
""")

        self._add_code("""# Install required packages (uncomment if needed)
# !pip install autogluon pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# AutoGluon
from autogluon.tabular import TabularPredictor

# Display settings
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.4f}'.format)

print("Environment ready!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
""")

    def _add_data_loading_section(self, dataset_spec: dict):
        """Add data loading section."""
        self._add_markdown("""## 2. Data Loading

Load the dataset from the specified source.
""")

        # Generate data loading code based on dataset spec
        target_column = dataset_spec.get("target_column", "target")
        file_path = dataset_spec.get("file_path", "data.csv")

        self._add_code(f'''# Dataset configuration
TARGET_COLUMN = "{target_column}"
DATA_PATH = "{file_path}"  # Update this path as needed

# Load data
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully!")
    print(f"Shape: {{df.shape[0]:,}} rows x {{df.shape[1]}} columns")
except FileNotFoundError:
    print(f"File not found at {{DATA_PATH}}")
    print("Please update DATA_PATH to point to your data file")
    df = None
''')

        self._add_code("""# Quick data overview
if df is not None:
    print("\\nColumn Types:")
    print(df.dtypes)
    print("\\nFirst 5 rows:")
    display(df.head())
    print("\\nBasic Statistics:")
    display(df.describe())
""")

    def _add_data_audit_section(self, audit_results: dict):
        """Add data audit section."""
        self._add_markdown("""## 3. Data Audit

Automated data quality analysis results.
""")

        # Display audit results
        audit_json = json.dumps(audit_results, indent=2, default=str)
        self._add_markdown(f"""### Audit Findings

```json
{audit_json[:3000]}{"..." if len(audit_json) > 3000 else ""}
```
""")

        self._add_code("""# Manual data quality checks
if df is not None:
    print("=== Data Quality Summary ===")
    print(f"\\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
    print(missing_df[missing_df['Missing'] > 0])

    print(f"\\nDuplicate Rows: {df.duplicated().sum():,}")

    print(f"\\nTarget Distribution:")
    if TARGET_COLUMN in df.columns:
        print(df[TARGET_COLUMN].value_counts())
""")

    def _add_eda_section(self, dataset_spec: dict):
        """Add exploratory data analysis section."""
        self._add_markdown("""## 4. Exploratory Data Analysis

Visual exploration of the data.
""")

        self._add_code("""# Target distribution
if df is not None and TARGET_COLUMN in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Value counts
    df[TARGET_COLUMN].value_counts().plot(kind='bar', ax=axes[0])
    axes[0].set_title('Target Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')

    # Pie chart
    df[TARGET_COLUMN].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
    axes[1].set_title('Target Proportions')

    plt.tight_layout()
    plt.show()
""")

        self._add_code("""# Numeric feature distributions
if df is not None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)

    if numeric_cols:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, col in enumerate(numeric_cols[:16]):  # Limit to 16 features
            df[col].hist(bins=30, ax=axes[i], edgecolor='black')
            axes[i].set_title(col[:20])

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
""")

    def _add_feature_engineering_section(
        self,
        feature_engineering_code: list[dict],
        dataset_spec: dict,
    ):
        """Add feature engineering section."""
        self._add_markdown("""## 5. Feature Engineering

Engineered features to improve model performance.
""")

        if not feature_engineering_code:
            self._add_markdown("*No custom feature engineering was applied in this experiment.*")
            return

        # Generate feature engineering code
        self._add_code("""# Feature engineering functions
def engineer_features(df):
    \"\"\"Apply all feature engineering transformations.\"\"\"
    df = df.copy()

    # Track created features
    created_features = []
""")

        for i, fe in enumerate(feature_engineering_code):
            name = fe.get("name", f"feature_{i}")
            formula = fe.get("formula", "")
            description = fe.get("description", "")

            # Clean up the formula for code generation
            formula_code = formula.strip()
            if not formula_code.startswith("df["):
                formula_code = f"df['{name}'] = {formula_code}"

            self._add_code(f'''    # Feature: {name}
    # {description}
    try:
        {formula_code}
        created_features.append("{name}")
        print(f"Created feature: {name}")
    except Exception as e:
        print(f"Failed to create {name}: {{e}}")
''')

        self._add_code("""
    print(f"\\nTotal features created: {len(created_features)}")
    return df, created_features

# Apply feature engineering
if df is not None:
    df, created_features = engineer_features(df)
    print(f"\\nDataset shape after feature engineering: {df.shape}")
""")

    def _add_automl_config_section(self, automl_config: dict, experiment_data: dict):
        """Add AutoGluon configuration section."""
        self._add_markdown("""## 6. AutoGluon Configuration

The AutoML configuration used for training.
""")

        config_json = json.dumps(automl_config, indent=2, default=str)
        self._add_markdown(f"""### Configuration

```python
{config_json}
```
""")

        # Extract key settings
        time_limit = automl_config.get("time_limit", 300)
        presets = automl_config.get("presets", "medium_quality")
        eval_metric = experiment_data.get("primary_metric", "auto")

        self._add_code(f'''# AutoGluon configuration
AUTOML_CONFIG = {{
    "time_limit": {time_limit},
    "presets": "{presets}",
    "eval_metric": "{eval_metric}",
    "verbosity": 2,
}}

print("AutoGluon Configuration:")
for key, value in AUTOML_CONFIG.items():
    print(f"  {{key}}: {{value}}")
''')

    def _add_training_section(self, automl_config: dict, dataset_spec: dict):
        """Add model training section."""
        self._add_markdown("""## 7. Model Training

Train the model using AutoGluon.
""")

        time_limit = automl_config.get("time_limit", 300)
        presets = automl_config.get("presets", "medium_quality")

        self._add_code(f'''# Prepare data for training
if df is not None:
    # Remove any problematic columns
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]

    # Create train/test split
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN] if df[TARGET_COLUMN].dtype == 'object' or df[TARGET_COLUMN].nunique() < 20 else None
    )

    print(f"Training set: {{len(train_df):,}} samples")
    print(f"Test set: {{len(test_df):,}} samples")
''')

        self._add_code(f'''# Train AutoGluon model
if df is not None:
    # Create predictor
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        eval_metric=AUTOML_CONFIG.get("eval_metric", "auto"),
        path="AutogluonModels/experiment"
    )

    # Train
    print("Starting training...")
    predictor.fit(
        train_data=train_df,
        time_limit={time_limit},
        presets="{presets}",
        verbosity=2,
    )

    print("\\nTraining complete!")
''')

    def _add_evaluation_section(self, metrics: dict):
        """Add evaluation section."""
        self._add_markdown("""## 8. Evaluation

Model performance evaluation.
""")

        metrics_json = json.dumps(metrics, indent=2, default=str)
        self._add_markdown(f"""### Training Metrics

```json
{metrics_json[:2000]}{"..." if len(metrics_json) > 2000 else ""}
```
""")

        self._add_code("""# Evaluate on test set
if df is not None and 'predictor' in dir():
    # Get predictions
    y_pred = predictor.predict(test_df)
    y_true = test_df[TARGET_COLUMN]

    # Evaluate
    print("=== Test Set Performance ===")
    performance = predictor.evaluate(test_df)
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")

    # Leaderboard
    print("\\n=== Model Leaderboard ===")
    display(predictor.leaderboard(test_df))
""")

        self._add_code("""# Confusion matrix for classification
if df is not None and 'predictor' in dir():
    from sklearn.metrics import confusion_matrix, classification_report

    if y_true.dtype == 'object' or y_true.nunique() < 20:
        print("\\n=== Classification Report ===")
        print(classification_report(y_true, y_pred))

        print("\\n=== Confusion Matrix ===")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
""")

        self._add_code("""# Feature importance
if df is not None and 'predictor' in dir():
    print("\\n=== Feature Importance ===")
    importance = predictor.feature_importance(test_df)

    # Plot top 20 features
    top_features = importance.head(20)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
""")

    def _add_prediction_section(self):
        """Add prediction section."""
        self._add_markdown("""## 9. Making Predictions

How to use the trained model for predictions.
""")

        self._add_code("""# Example: Make predictions on new data
def predict_new_data(new_df, predictor):
    \"\"\"Make predictions on new data.

    Args:
        new_df: DataFrame with same features as training data
        predictor: Trained AutoGluon predictor

    Returns:
        predictions: Predicted values
        probabilities: Class probabilities (for classification)
    \"\"\"
    predictions = predictor.predict(new_df)

    # Try to get probabilities for classification
    try:
        probabilities = predictor.predict_proba(new_df)
    except:
        probabilities = None

    return predictions, probabilities

# Example usage
if df is not None and 'predictor' in dir():
    # Take a few samples from test set
    sample_df = test_df.head(5).drop(columns=[TARGET_COLUMN])

    print("Sample predictions:")
    preds, probs = predict_new_data(sample_df, predictor)

    results = pd.DataFrame({
        'Prediction': preds,
        'Actual': test_df[TARGET_COLUMN].head(5).values
    })
    display(results)

    if probs is not None:
        print("\\nClass Probabilities:")
        display(probs.head())
""")

        self._add_code("""# Save model for later use
if 'predictor' in dir():
    model_path = "saved_model"
    predictor.save(model_path)
    print(f"Model saved to: {model_path}")

    # Load model example
    # loaded_predictor = TabularPredictor.load(model_path)
""")

    def _add_agent_analysis_section(self, agent_analysis: dict):
        """Add agent analysis section."""
        self._add_markdown("""## Agent Analysis

AI agent analysis and recommendations from the pipeline.
""")

        analysis_text = json.dumps(agent_analysis, indent=2, default=str)
        self._add_markdown(f"""### Analysis Results

```json
{analysis_text[:5000]}{"..." if len(analysis_text) > 5000 else ""}
```
""")

    def _add_conclusion_section(self, experiment_data: dict, metrics: dict):
        """Add conclusion section."""
        self._add_markdown(f"""## 10. Conclusion

### Experiment Summary

- **Experiment**: {experiment_data.get("name", "N/A")}
- **Status**: {experiment_data.get("status", "N/A")}
- **Primary Metric**: {experiment_data.get("primary_metric", "N/A")}

### Key Metrics

""")

        # Extract and display key metrics
        if metrics:
            metrics_list = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_list.append(f"- **{key}**: {value:.4f}")
            if metrics_list:
                self._add_markdown("\n".join(metrics_list[:10]))

        self._add_markdown("""
### Next Steps

1. Review feature importance and consider domain-specific features
2. Try different presets or time limits if performance is insufficient
3. Consider hyperparameter tuning for top models
4. Deploy the model using the saved artifacts

---

*This notebook was automatically generated by AgentML*
""")


def generate_experiment_notebook(
    experiment_id: str,
    db,  # SQLAlchemy session
) -> Optional[str]:
    """Generate a notebook for a specific experiment.

    Args:
        experiment_id: UUID of the experiment
        db: Database session

    Returns:
        Notebook JSON string, or None if generation fails
    """
    from app.models.experiment import Experiment
    from app.models.dataset_spec import DatasetSpec
    from app.models.model_version import ModelVersion

    try:
        # Load experiment
        exp = db.query(Experiment).filter(Experiment.id == UUID(experiment_id)).first()
        if not exp:
            logger.error(f"Experiment {experiment_id} not found")
            return None

        # Load dataset spec
        dataset_spec = db.query(DatasetSpec).filter(
            DatasetSpec.id == exp.dataset_spec_id
        ).first()

        # Load model version for metrics
        model_version = db.query(ModelVersion).filter(
            ModelVersion.experiment_id == exp.id
        ).first()

        # Build experiment data
        experiment_data = {
            "id": str(exp.id),
            "name": exp.name,
            "status": exp.status.value if exp.status else "unknown",
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "iteration_number": exp.iteration_number,
            "primary_metric": exp.primary_metric,
        }

        # Build dataset spec data
        dataset_spec_data = {
            "target_column": dataset_spec.target_column if dataset_spec else "target",
            "file_path": "data.csv",  # Placeholder
        }

        # Get feature engineering from dataset spec's spec_json
        feature_engineering_code = []
        if dataset_spec and dataset_spec.spec_json:
            engineered_features = dataset_spec.spec_json.get("engineered_features", [])
            for feat in engineered_features:
                if isinstance(feat, dict):
                    feature_engineering_code.append({
                        "name": feat.get("output_column", feat.get("name", "")),
                        "formula": feat.get("expression", feat.get("formula", "")),
                        "description": feat.get("description", ""),
                    })

        # Get AutoML config
        automl_config = exp.experiment_plan_json.get("automl_config", {}) if exp.experiment_plan_json else {}

        # Get metrics
        metrics = model_version.metrics_json if model_version and model_version.metrics_json else {}

        # Get data audit results from spec_json if available
        data_audit_results = None
        if dataset_spec and dataset_spec.spec_json:
            data_audit_results = dataset_spec.spec_json.get("audit_results")

        # Generate notebook
        generator = NotebookGenerator()
        notebook_json = generator.generate_experiment_notebook(
            experiment_data=experiment_data,
            dataset_spec_data=dataset_spec_data,
            feature_engineering_code=feature_engineering_code,
            automl_config=automl_config,
            metrics=metrics,
            data_audit_results=data_audit_results,
            agent_analysis=exp.improvement_context_json,
        )

        return notebook_json

    except Exception as e:
        logger.error(f"Error generating notebook for experiment {experiment_id}: {e}")
        return None

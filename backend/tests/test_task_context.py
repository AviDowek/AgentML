"""Tests for the task context builder service (Prompt 7)."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.services.task_context import (
    build_task_context,
    get_task_type_hints,
    format_context_for_prompt,
    _build_project_context,
    _build_dataset_spec_context,
    _build_data_profile_summary,
    _extract_baselines_and_shuffle,
    _build_robustness_summary,
)


class TestBuildProjectContext:
    """Tests for _build_project_context function."""

    def test_builds_basic_project_context(self):
        """Test building context from a project."""
        project = MagicMock()
        project.id = uuid4()
        project.name = "Test Project"
        project.description = "A test project"
        project.task_type = "binary"
        project.created_at = datetime(2024, 1, 1)

        context = _build_project_context(project)

        assert context["id"] == str(project.id)
        assert context["name"] == "Test Project"
        assert context["description"] == "A test project"
        assert context["task_type"] == "binary"
        assert context["created_at"] == "2024-01-01T00:00:00"


class TestBuildDatasetSpecContext:
    """Tests for _build_dataset_spec_context function."""

    def test_builds_non_time_based_spec(self):
        """Test building context for non-time-based dataset spec."""
        spec = MagicMock()
        spec.id = uuid4()
        spec.name = "Customer Churn Dataset"
        spec.target_column = "churned"
        spec.feature_columns = ["age", "income", "tenure"]
        spec.is_time_based = False
        spec.time_column = None
        spec.entity_id_column = None
        spec.split_strategy_json = {"type": "stratified", "test_size": 0.2}
        spec.preprocessing_strategy = "standard"
        spec.filters_json = None

        context = _build_dataset_spec_context(spec)

        assert context["name"] == "Customer Churn Dataset"
        assert context["target_column"] == "churned"
        assert context["feature_columns"] == ["age", "income", "tenure"]
        assert context["is_time_based"] is False
        assert context["time_column"] is None
        assert context["split_strategy"]["type"] == "stratified"

    def test_builds_time_based_spec(self):
        """Test building context for time-based dataset spec."""
        spec = MagicMock()
        spec.id = uuid4()
        spec.name = "Stock Prediction Dataset"
        spec.target_column = "ret_5d"
        spec.feature_columns_json = ["ma_20", "vol_10d", "momentum"]
        spec.is_time_based = True
        spec.time_column = "date"
        spec.entity_id_column = "ticker"
        spec.split_strategy_json = {"type": "time", "test_size": 0.2}
        spec.preprocessing_strategy = "robust"
        spec.filters_json = {"min_date": "2020-01-01"}

        context = _build_dataset_spec_context(spec)

        assert context["is_time_based"] is True
        assert context["time_column"] == "date"
        assert context["entity_id_column"] == "ticker"
        assert context["split_strategy"]["type"] == "time"


class TestExtractBaselinesAndShuffle:
    """Tests for _extract_baselines_and_shuffle function."""

    def test_extracts_classification_baselines(self):
        """Test extracting classification baselines."""
        trial = MagicMock()
        trial.baseline_metrics_json = {
            "majority_class": {"accuracy": 0.65, "roc_auc": 0.5},
            "simple_logistic": {"accuracy": 0.72, "roc_auc": 0.78},
            "label_shuffle": {
                "shuffled_accuracy": 0.50,
                "shuffled_roc_auc": 0.51,
                "leakage_detected": False,
            },
        }

        experiment = MagicMock()
        experiment.trials = [trial]

        baselines, label_shuffle = _extract_baselines_and_shuffle([experiment])

        assert baselines["available"] is True
        assert baselines["majority_class"]["accuracy"] == 0.65
        assert baselines["simple_model"]["roc_auc"] == 0.78
        assert label_shuffle["available"] is True
        assert label_shuffle["leakage_detected"] is False

    def test_extracts_regression_baselines(self):
        """Test extracting regression baselines."""
        trial = MagicMock()
        trial.baseline_metrics_json = {
            "mean_predictor": {"rmse": 15.5, "mae": 12.0, "r2": 0.0},
            "simple_ridge": {"rmse": 10.2, "mae": 8.5, "r2": 0.45},
        }

        experiment = MagicMock()
        experiment.trials = [trial]

        baselines, label_shuffle = _extract_baselines_and_shuffle([experiment])

        assert baselines["available"] is True
        assert baselines["mean_predictor"]["rmse"] == 15.5
        assert baselines["regression_baseline"]["r2"] == 0.45

    def test_detects_leakage_from_shuffle(self):
        """Test detecting leakage from label shuffle results."""
        trial = MagicMock()
        trial.baseline_metrics_json = {
            "label_shuffle": {
                "shuffled_roc_auc": 0.72,  # Too high!
                "leakage_detected": True,
                "warning": "AUC too high after label shuffle",
            },
        }

        experiment = MagicMock()
        experiment.trials = [trial]

        baselines, label_shuffle = _extract_baselines_and_shuffle([experiment])

        assert label_shuffle["available"] is True
        assert label_shuffle["leakage_detected"] is True
        assert "too high" in label_shuffle["warning"].lower()

    def test_handles_no_baselines(self):
        """Test handling experiments without baselines."""
        trial = MagicMock()
        trial.baseline_metrics_json = None

        experiment = MagicMock()
        experiment.trials = [trial]

        baselines, label_shuffle = _extract_baselines_and_shuffle([experiment])

        assert baselines["available"] is False
        assert label_shuffle["available"] is False


class TestBuildRobustnessSummary:
    """Tests for _build_robustness_summary function."""

    def test_detects_high_overfitting_risk(self):
        """Test detecting high overfitting risk from train-val gap."""
        trial = MagicMock()
        trial.metrics_json = {
            "train_accuracy": 0.95,
            "val_accuracy": 0.72,  # Gap of 0.23 > 0.15
        }
        trial.baseline_metrics_json = None
        trial.data_split_strategy = "random"

        experiment = MagicMock()
        experiment.trials = [trial]

        robustness = _build_robustness_summary([experiment])

        assert robustness["overfitting_risk"] == "high"
        assert len(robustness["warnings"]) > 0

    def test_detects_medium_overfitting_risk(self):
        """Test detecting medium overfitting risk."""
        trial = MagicMock()
        trial.metrics_json = {
            "train_accuracy": 0.88,
            "val_accuracy": 0.78,  # Gap of 0.10 > 0.08
        }
        trial.baseline_metrics_json = None
        trial.data_split_strategy = "random"

        experiment = MagicMock()
        experiment.trials = [trial]

        robustness = _build_robustness_summary([experiment])

        assert robustness["overfitting_risk"] == "medium"

    def test_detects_low_overfitting_risk(self):
        """Test detecting low overfitting risk."""
        trial = MagicMock()
        trial.metrics_json = {
            "train_accuracy": 0.82,
            "val_accuracy": 0.80,  # Gap of 0.02 < 0.08
        }
        trial.baseline_metrics_json = None
        trial.data_split_strategy = "random"

        experiment = MagicMock()
        experiment.trials = [trial]

        robustness = _build_robustness_summary([experiment])

        assert robustness["overfitting_risk"] == "low"

    def test_detects_leakage_suspected(self):
        """Test detecting leakage from baseline metrics."""
        trial = MagicMock()
        trial.metrics_json = {}
        trial.baseline_metrics_json = {
            "label_shuffle": {
                "leakage_detected": True,
                "warning": "Potential data leakage",
            }
        }
        trial.data_split_strategy = "random"

        experiment = MagicMock()
        experiment.trials = [trial]

        robustness = _build_robustness_summary([experiment])

        assert robustness["leakage_suspected"] is True
        assert "leakage" in robustness["warnings"][0].lower()

    def test_includes_risk_adjusted_score(self):
        """Test including risk-adjusted score."""
        trial = MagicMock()
        trial.metrics_json = {
            "risk_adjusted_score": 0.72,
        }
        trial.baseline_metrics_json = None
        trial.data_split_strategy = "time"

        experiment = MagicMock()
        experiment.trials = [trial]

        robustness = _build_robustness_summary([experiment])

        assert robustness["risk_adjusted_score_best_model"] == 0.72


class TestGetTaskTypeHints:
    """Tests for get_task_type_hints function."""

    def test_binary_classification_hints(self):
        """Test hints for binary classification."""
        context = {
            "project": {"task_type": "binary"},
            "dataset_spec": {"is_time_based": False},
            "leakage_candidates": [],
            "data_profile_summary": {},
        }

        hints = get_task_type_hints(context)

        assert hints["is_classification"] is True
        assert hints["is_binary"] is True
        assert hints["is_regression"] is False
        assert "roc_auc" in hints["recommended_metrics"]
        assert hints["recommended_split"] == "stratified"

    def test_regression_hints(self):
        """Test hints for regression."""
        context = {
            "project": {"task_type": "regression"},
            "dataset_spec": {"is_time_based": False},
            "leakage_candidates": [],
            "data_profile_summary": {},
        }

        hints = get_task_type_hints(context)

        assert hints["is_regression"] is True
        assert hints["is_classification"] is False
        assert "rmse" in hints["recommended_metrics"]

    def test_time_based_hints(self):
        """Test hints for time-based tasks."""
        context = {
            "project": {"task_type": "binary"},
            "dataset_spec": {"is_time_based": True},
            "leakage_candidates": [],
            "data_profile_summary": {},
        }

        hints = get_task_type_hints(context)

        assert hints["is_time_based"] is True
        assert hints["recommended_split"] == "time"
        assert len(hints["leakage_warnings"]) > 0

    def test_leakage_warnings_in_hints(self):
        """Test leakage warnings are included in hints."""
        context = {
            "project": {"task_type": "binary"},
            "dataset_spec": {"is_time_based": False},
            "leakage_candidates": [
                {"column": "target_encoded", "severity": "high", "reason": "Contains target"},
                {"column": "future_val", "severity": "high", "reason": "Future data"},
            ],
            "data_profile_summary": {},
        }

        hints = get_task_type_hints(context)

        assert len(hints["leakage_warnings"]) > 0
        assert "2 high-severity" in hints["leakage_warnings"][0]

    def test_data_quality_warnings(self):
        """Test data quality warnings are included."""
        context = {
            "project": {"task_type": "binary"},
            "dataset_spec": {"is_time_based": False},
            "leakage_candidates": [],
            "data_profile_summary": {
                "missingness_summary": {"columns_with_nulls": 5}
            },
        }

        hints = get_task_type_hints(context)

        assert len(hints["data_quality_warnings"]) > 0
        assert "5 column(s)" in hints["data_quality_warnings"][0]


class TestFormatContextForPrompt:
    """Tests for format_context_for_prompt function."""

    def test_formats_basic_context(self):
        """Test formatting a basic context."""
        context = {
            "project": {
                "name": "Churn Prediction",
                "description": "Predict customer churn",
                "task_type": "binary",
            },
            "dataset_spec": {
                "target_column": "churned",
                "feature_columns": ["a", "b", "c"],
                "is_time_based": False,
            },
            "data_profile_summary": {
                "row_count": 10000,
                "column_count": 20,
            },
            "latest_experiments": [],
            "baselines": {"available": False},
            "robustness": None,
            "leakage_candidates": [],
        }

        result = format_context_for_prompt(context)

        assert "Churn Prediction" in result
        assert "binary" in result
        assert "churned" in result
        assert "10,000" in result

    def test_includes_baselines_when_available(self):
        """Test including baselines in formatted output."""
        context = {
            "project": {"name": "Test", "task_type": "binary"},
            "dataset_spec": None,
            "data_profile_summary": None,
            "latest_experiments": [],
            "baselines": {
                "available": True,
                "majority_class": {"accuracy": 0.65, "roc_auc": 0.5},
            },
            "robustness": None,
            "leakage_candidates": [],
        }

        result = format_context_for_prompt(context)

        assert "Baselines" in result
        assert "Majority Class" in result

    def test_includes_robustness_warnings(self):
        """Test including robustness warnings."""
        context = {
            "project": {"name": "Test", "task_type": "binary"},
            "dataset_spec": None,
            "data_profile_summary": None,
            "latest_experiments": [],
            "baselines": {"available": False},
            "robustness": {
                "overfitting_risk": "high",
                "leakage_suspected": True,
                "warnings": ["Large train-val gap"],
            },
            "leakage_candidates": [],
        }

        result = format_context_for_prompt(context)

        assert "Robustness" in result
        assert "high" in result
        assert "LEAKAGE SUSPECTED" in result

    def test_truncates_long_output(self):
        """Test truncation of long output."""
        context = {
            "project": {"name": "Test" * 100, "description": "X" * 1000, "task_type": "binary"},
            "dataset_spec": {"target_column": "y", "feature_columns": [f"f{i}" for i in range(100)], "is_time_based": False},
            "data_profile_summary": {},
            "latest_experiments": [],
            "baselines": {"available": False},
            "robustness": None,
            "leakage_candidates": [{"column": f"col{i}", "severity": "high", "reason": "test"} for i in range(50)],
        }

        result = format_context_for_prompt(context, max_length=500)

        assert len(result) <= 520  # Some buffer for truncation message
        assert "truncated" in result

    def test_filters_sections(self):
        """Test filtering specific sections."""
        context = {
            "project": {"name": "Test", "task_type": "binary"},
            "dataset_spec": {"target_column": "y", "feature_columns": [], "is_time_based": False},
            "data_profile_summary": {"row_count": 100},
            "latest_experiments": [],
            "baselines": {"available": True, "majority_class": {"accuracy": 0.5}},
            "robustness": None,
            "leakage_candidates": [],
        }

        result = format_context_for_prompt(context, include_sections=["project"])

        assert "Project" in result
        assert "Dataset Configuration" not in result
        assert "Baselines" not in result

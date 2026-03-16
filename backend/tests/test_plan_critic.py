"""Tests for Plan Critic context validation (Prompt 7 Step 5)."""
import pytest
from unittest.mock import MagicMock

from app.services.agent_executor import (
    _validate_plan_against_context,
    _generate_plan_summary,
)


class TestValidatePlanAgainstContext:
    """Tests for _validate_plan_against_context function."""

    def _make_step_logger(self):
        """Create a mock step logger."""
        logger = MagicMock()
        logger.thought = MagicMock()
        logger.warning = MagicMock()
        logger.info = MagicMock()
        return logger

    def test_valid_plan_no_context(self):
        """Test validation passes when no context available."""
        input_data = {
            "feature_columns": ["age", "income"],
            "variants": [{"name": "v1", "validation_strategy": {"split_strategy": "stratified"}}],
            "is_time_based": False,
        }
        result = _validate_plan_against_context(
            input_data=input_data,
            task_context=None,
            task_hints={},
            step_logger=self._make_step_logger(),
        )
        assert result["overall_valid"] is True
        assert result["split_validation"]["valid"] is True
        assert result["leakage_validation"]["valid"] is True

    def test_rejects_random_split_on_time_based_task(self):
        """Test that random split is rejected for time-based tasks."""
        input_data = {
            "feature_columns": ["age", "income"],
            "variants": [
                {"name": "main", "validation_strategy": {"split_strategy": "random"}}
            ],
            "is_time_based": True,
            "time_column": "date",
        }
        result = _validate_plan_against_context(
            input_data=input_data,
            task_context=None,
            task_hints={"is_time_based": True},
            step_logger=self._make_step_logger(),
        )
        assert result["overall_valid"] is False
        assert result["split_validation"]["valid"] is False
        assert len(result["split_validation"]["required_changes"]) == 1
        change = result["split_validation"]["required_changes"][0]
        assert change["variant"] == "main"
        assert change["issue"] == "random_split_on_time_data"

    def test_accepts_time_split_on_time_based_task(self):
        """Test that time split is accepted for time-based tasks."""
        input_data = {
            "feature_columns": ["age", "income"],
            "variants": [
                {"name": "main", "validation_strategy": {"split_strategy": "time", "time_column": "date"}}
            ],
            "is_time_based": True,
            "time_column": "date",
        }
        result = _validate_plan_against_context(
            input_data=input_data,
            task_context=None,
            task_hints={"is_time_based": True},
            step_logger=self._make_step_logger(),
        )
        assert result["split_validation"]["valid"] is True
        assert len(result["split_validation"]["required_changes"]) == 0

    def test_accepts_override_with_justification(self):
        """Test that random split with override flag and justification is accepted."""
        input_data = {
            "feature_columns": ["age", "income"],
            "variants": [
                {
                    "name": "cross_sectional",
                    "validation_strategy": {
                        "split_strategy": "random",
                        "time_split_override": True,
                        "reasoning": "This is cross-sectional data at a single point in time, not truly temporal",
                    },
                }
            ],
            "is_time_based": True,
            "time_column": "date",
        }
        result = _validate_plan_against_context(
            input_data=input_data,
            task_context=None,
            task_hints={"is_time_based": True},
            step_logger=self._make_step_logger(),
        )
        assert result["split_validation"]["valid"] is True
        # Should have a warning but no required change
        assert len(result["split_validation"]["warnings"]) == 1
        assert len(result["split_validation"]["required_changes"]) == 0

    def test_detects_high_severity_leakage_features(self):
        """Test detection of high-severity leakage features in plan."""
        input_data = {
            "feature_columns": ["age", "income", "future_value", "cancellation_date"],
            "variants": [],
        }
        task_context = {
            "leakage_candidates": [
                {"column": "future_value", "severity": "high", "reason": "Future data", "detection_method": "name"},
                {"column": "cancellation_date", "severity": "high", "reason": "Leaks target", "detection_method": "name"},
            ]
        }
        result = _validate_plan_against_context(
            input_data=input_data,
            task_context=task_context,
            task_hints={},
            step_logger=self._make_step_logger(),
        )
        assert result["overall_valid"] is False
        assert result["leakage_validation"]["valid"] is False
        assert len(result["leakage_validation"]["required_changes"]) == 2

    def test_warns_on_medium_severity_leakage_features(self):
        """Test warning for medium-severity leakage features."""
        input_data = {
            "feature_columns": ["age", "suspicious_feature"],
            "variants": [],
        }
        task_context = {
            "leakage_candidates": [
                {"column": "suspicious_feature", "severity": "medium", "reason": "High correlation"},
            ]
        }
        result = _validate_plan_against_context(
            input_data=input_data,
            task_context=task_context,
            task_hints={},
            step_logger=self._make_step_logger(),
        )
        assert result["overall_valid"] is True  # Medium doesn't block
        assert result["leakage_validation"]["valid"] is True
        assert len(result["leakage_validation"]["warnings"]) == 1

    def test_context_overrides_is_time_based(self):
        """Test that TaskContext can override is_time_based."""
        input_data = {
            "feature_columns": ["age", "income"],
            "variants": [
                {"name": "main", "validation_strategy": {"split_strategy": "random"}}
            ],
            "is_time_based": False,  # Plan says not time-based
        }
        task_context = {
            "dataset_spec": {"is_time_based": True, "time_column": "date"}  # But context says it is
        }
        result = _validate_plan_against_context(
            input_data=input_data,
            task_context=task_context,
            task_hints={},
            step_logger=self._make_step_logger(),
        )
        # Should fail because context says time-based
        assert result["split_validation"]["valid"] is False

    def test_metric_validation_warns_on_unrealistic_target(self):
        """Test warning when plan targets unrealistic metrics."""
        input_data = {
            "feature_columns": ["age", "income"],
            "variants": [
                {
                    "name": "main",
                    "validation_strategy": {},
                    "target_metrics": {"roc_auc": 0.99},  # Very high target
                }
            ],
            "context_analysis": {
                "expected_metric_range": {
                    "metric": "roc_auc",
                    "lower_bound": 0.65,
                    "upper_bound": 0.80,
                }
            },
        }
        # Need a non-empty task_context for the metric validation check to run
        task_context = {"project": {"name": "test"}}
        result = _validate_plan_against_context(
            input_data=input_data,
            task_context=task_context,
            task_hints={},
            step_logger=self._make_step_logger(),
        )
        assert len(result["metric_validation"]["warnings"]) == 1
        assert "0.99" in result["metric_validation"]["warnings"][0]


class TestGeneratePlanSummary:
    """Tests for _generate_plan_summary function."""

    def test_approved_plan_summary(self):
        """Test summary for approved plan."""
        summary = _generate_plan_summary(
            approved=True,
            issues=[],
            warnings=[],
            required_changes=[],
            feature_count=20,
            variant_count=3,
        )
        assert "✅ Plan APPROVED" in summary
        assert "3 experiment variant(s)" in summary
        assert "20 features" in summary

    def test_rejected_plan_summary(self):
        """Test summary for rejected plan."""
        summary = _generate_plan_summary(
            approved=False,
            issues=["Random split on time data"],
            warnings=["High null columns"],
            required_changes=[{"issue": "random_split_on_time_data", "variant": "main"}],
            feature_count=15,
            variant_count=2,
        )
        assert "❌ Plan REQUIRES REVISION" in summary
        assert "1 issue(s)" in summary

    def test_summary_includes_warnings(self):
        """Test that summary includes warnings."""
        summary = _generate_plan_summary(
            approved=True,
            issues=[],
            warnings=["Warning 1", "Warning 2", "Warning 3", "Warning 4"],
            required_changes=[],
            feature_count=10,
            variant_count=1,
        )
        assert "4 warning(s)" in summary
        assert "Warning 1" in summary
        assert "... and 1 more" in summary  # 4th warning should be truncated

    def test_summary_includes_context_notes(self):
        """Test that summary includes context-specific notes."""
        summary = _generate_plan_summary(
            approved=False,
            issues=[],
            warnings=[],
            required_changes=[],
            feature_count=10,
            variant_count=1,
            context_validation={
                "split_validation": {"valid": False},
                "leakage_validation": {"valid": True},
            },
        )
        assert "Split strategy requires attention" in summary

    def test_summary_leakage_warning(self):
        """Test that summary includes leakage warning."""
        summary = _generate_plan_summary(
            approved=False,
            issues=[],
            warnings=[],
            required_changes=[],
            feature_count=10,
            variant_count=1,
            context_validation={
                "split_validation": {"valid": True},
                "leakage_validation": {"valid": False},
            },
        )
        assert "Leakage risk" in summary

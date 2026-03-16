"""Tests for the leakage detector module (Prompt 6)."""
import pytest
import pandas as pd
import numpy as np

from app.services.leakage_detector import (
    detect_potential_leakage_features,
    check_leakage_in_important_features,
    get_leakage_summary,
    SUSPICIOUS_NAME_PATTERNS,
    SAFE_PATTERNS,
)


class TestDetectPotentialLeakageFeatures:
    """Tests for detect_potential_leakage_features function."""

    def test_no_leakage_clean_data(self):
        """Test that clean data returns no suspects."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, 70000, 80000],
            "target": [0, 1, 0, 1],
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
        )
        assert len(suspects) == 0

    def test_detects_label_keyword(self):
        """Test detection of 'label' keyword in feature name."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "encoded_label": [0, 1, 0, 1],
            "target": [0, 1, 0, 1],
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
        )
        assert len(suspects) == 1
        assert suspects[0]["column"] == "encoded_label"
        assert suspects[0]["severity"] == "high"
        assert suspects[0]["detection_method"] == "name"

    def test_detects_target_keyword(self):
        """Test detection of 'target' keyword in feature name."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "target_encoded": [0, 1, 0, 1],
            "churn": [0, 1, 0, 1],
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="churn",
        )
        assert len(suspects) == 1
        assert suspects[0]["column"] == "target_encoded"

    def test_detects_future_keyword(self):
        """Test detection of 'future' keyword in feature name."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "future_value": [100, 200, 300, 400],
            "target": [0, 1, 0, 1],
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
        )
        assert len(suspects) == 1
        assert suspects[0]["column"] == "future_value"
        assert suspects[0]["severity"] == "high"

    def test_detects_next_keyword(self):
        """Test detection of 'next' keyword in feature name."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "next_month_sales": [100, 200, 300, 400],
            "target": [0, 1, 0, 1],
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
        )
        assert len(suspects) == 1
        assert suspects[0]["column"] == "next_month_sales"

    def test_detects_t_plus_n_pattern(self):
        """Test detection of t+N pattern in feature name."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "price_t+1": [100, 200, 300, 400],
            "price_t+3": [100, 200, 300, 400],
            "target": [0, 1, 0, 1],
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
        )
        assert len(suspects) == 2
        columns = [s["column"] for s in suspects]
        assert "price_t+1" in columns
        assert "price_t+3" in columns

    def test_detects_ret_nd_pattern(self):
        """Test detection of ret_Nd return pattern."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "ret_1d": [0.01, 0.02, 0.03, 0.04],
            "ret_5d": [0.05, 0.06, 0.07, 0.08],
            "target": [0, 1, 0, 1],
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
        )
        assert len(suspects) == 2
        columns = [s["column"] for s in suspects]
        assert "ret_1d" in columns
        assert "ret_5d" in columns

    def test_safe_patterns_not_flagged(self):
        """Test that safe patterns like lag_ are not flagged."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "lag_1_price": [100, 200, 300, 400],
            "rolling_mean_7d": [100, 200, 300, 400],
            "ma_20": [100, 200, 300, 400],
            "prev_month_sales": [100, 200, 300, 400],
            "target": [0, 1, 0, 1],
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
        )
        # All are safe patterns
        assert len(suspects) == 0

    def test_detects_high_correlation(self):
        """Test detection of highly correlated features."""
        # Create feature that is perfectly correlated with target
        target = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "suspicious_feature": target,  # Perfect correlation
            "target": target,
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
            correlation_threshold=0.9,
        )
        assert len(suspects) >= 1
        columns = [s["column"] for s in suspects]
        assert "suspicious_feature" in columns
        # Should be detected by correlation
        corr_suspects = [s for s in suspects if s["detection_method"] == "correlation"]
        assert len(corr_suspects) >= 1

    def test_detects_cancellation_date(self):
        """Test detection of cancellation_date which may leak churn."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "cancellation_date": ["2024-01-01", "2024-02-01", None, None],
            "churned": [1, 1, 0, 0],
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="churned",
        )
        assert len(suspects) == 1
        assert suspects[0]["column"] == "cancellation_date"
        assert suspects[0]["severity"] == "high"

    def test_lineage_forward_window(self):
        """Test detection via lineage metadata with forward window."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "avg_future_price": [100, 200, 300, 400],
            "target": [0, 1, 0, 1],
        })
        lineage = {
            "avg_future_price": {"window": "forward"},
        }
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
            feature_lineage=lineage,
        )
        # Should be flagged both by name (future) and lineage
        assert len(suspects) >= 1
        columns = [s["column"] for s in suspects]
        assert "avg_future_price" in columns

    def test_lineage_positive_time_offset(self):
        """Test detection via positive time offset in lineage."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "shifted_value": [100, 200, 300, 400],
            "target": [0, 1, 0, 1],
        })
        lineage = {
            "shifted_value": {"time_offset": 5},  # +5 days into future
        }
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
            feature_lineage=lineage,
        )
        assert len(suspects) == 1
        assert suspects[0]["column"] == "shifted_value"
        assert suspects[0]["detection_method"] == "lineage"

    def test_combined_detection_upgrades_severity(self):
        """Test that multiple detection methods upgrade severity."""
        # Create a feature that is both name-suspicious AND highly correlated
        target = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "future_indicator": target,  # Name-suspicious AND correlated
            "target": target,
        })
        suspects = detect_potential_leakage_features(
            df=df,
            target_column="target",
            correlation_threshold=0.9,
        )
        assert len(suspects) >= 1
        future_suspect = [s for s in suspects if s["column"] == "future_indicator"][0]
        # Should be upgraded to high severity
        assert future_suspect["severity"] == "high"


class TestCheckLeakageInImportantFeatures:
    """Tests for check_leakage_in_important_features function."""

    def test_no_leakage_candidates(self):
        """Test with no leakage candidates."""
        has_concerning, features, warning = check_leakage_in_important_features(
            leakage_candidates=[],
            feature_importances={"age": 0.5, "income": 0.3},
        )
        assert has_concerning is False
        assert len(features) == 0
        assert warning == ""

    def test_no_feature_importances(self):
        """Test with no feature importances."""
        has_concerning, features, warning = check_leakage_in_important_features(
            leakage_candidates=[{"column": "leaky", "reason": "test", "severity": "high", "detection_method": "name"}],
            feature_importances={},
        )
        assert has_concerning is False
        assert len(features) == 0

    def test_leakage_not_important(self):
        """Test when leakage candidates are not among top features."""
        leakage_candidates = [
            {"column": "leaky_feature", "reason": "test", "severity": "high", "detection_method": "name"},
        ]
        feature_importances = {
            "age": 0.5,
            "income": 0.3,
            "education": 0.15,
            "leaky_feature": 0.01,  # Very low importance
        }
        has_concerning, features, warning = check_leakage_in_important_features(
            leakage_candidates=leakage_candidates,
            feature_importances=feature_importances,
            top_n=3,
        )
        assert has_concerning is False

    def test_leakage_is_top_feature(self):
        """Test when leakage candidate is among top features."""
        leakage_candidates = [
            {"column": "leaky_feature", "reason": "Future data", "severity": "high", "detection_method": "name"},
        ]
        feature_importances = {
            "leaky_feature": 0.5,  # Top feature!
            "age": 0.3,
            "income": 0.15,
            "education": 0.05,
        }
        has_concerning, features, warning = check_leakage_in_important_features(
            leakage_candidates=leakage_candidates,
            feature_importances=feature_importances,
            top_n=10,
        )
        assert has_concerning is True
        assert len(features) == 1
        assert features[0]["column"] == "leaky_feature"
        assert features[0]["importance_rank"] == 1
        assert "suspicious" in warning.lower()

    def test_multiple_leakage_in_top_features(self):
        """Test when multiple leakage candidates are in top features."""
        leakage_candidates = [
            {"column": "leaky_1", "reason": "Future data", "severity": "high", "detection_method": "name"},
            {"column": "leaky_2", "reason": "Target correlation", "severity": "medium", "detection_method": "correlation"},
        ]
        feature_importances = {
            "age": 0.3,
            "leaky_1": 0.25,  # Second most important
            "leaky_2": 0.20,  # Third most important
            "income": 0.15,
            "education": 0.10,
        }
        has_concerning, features, warning = check_leakage_in_important_features(
            leakage_candidates=leakage_candidates,
            feature_importances=feature_importances,
            top_n=5,
        )
        assert has_concerning is True
        assert len(features) == 2
        assert "2 suspicious" in warning

    def test_importance_threshold(self):
        """Test importance threshold filtering."""
        leakage_candidates = [
            {"column": "leaky_feature", "reason": "test", "severity": "high", "detection_method": "name"},
        ]
        feature_importances = {
            "age": 0.5,
            "leaky_feature": 0.02,  # Below 5% of max
        }
        has_concerning, features, warning = check_leakage_in_important_features(
            leakage_candidates=leakage_candidates,
            feature_importances=feature_importances,
            top_n=10,
            importance_threshold=0.05,  # 5% threshold
        )
        # 0.02 is 4% of 0.5, so below threshold
        assert has_concerning is False


class TestGetLeakageSummary:
    """Tests for get_leakage_summary function."""

    def test_empty_candidates(self):
        """Test summary with no candidates."""
        summary = get_leakage_summary([])
        assert summary["total_count"] == 0
        assert summary["high_severity_count"] == 0
        assert summary["medium_severity_count"] == 0
        assert summary["low_severity_count"] == 0

    def test_mixed_severity_candidates(self):
        """Test summary with mixed severity candidates."""
        candidates = [
            {"column": "col1", "reason": "test", "severity": "high", "detection_method": "name"},
            {"column": "col2", "reason": "test", "severity": "high", "detection_method": "correlation"},
            {"column": "col3", "reason": "test", "severity": "medium", "detection_method": "name"},
            {"column": "col4", "reason": "test", "severity": "low", "detection_method": "correlation"},
        ]
        summary = get_leakage_summary(candidates)
        assert summary["total_count"] == 4
        assert summary["high_severity_count"] == 2
        assert summary["medium_severity_count"] == 1
        assert summary["low_severity_count"] == 1
        assert "col1" in summary["high_severity_features"]
        assert "col2" in summary["high_severity_features"]
        assert summary["by_method"]["name"] == 2
        assert summary["by_method"]["correlation"] == 2


class TestPatternCompleteness:
    """Tests to verify suspicious and safe patterns are working."""

    def test_suspicious_patterns_exist(self):
        """Verify suspicious patterns list is populated."""
        assert len(SUSPICIOUS_NAME_PATTERNS) > 0
        # Check some expected patterns
        pattern_strings = [p[0] for p in SUSPICIOUS_NAME_PATTERNS]
        patterns_text = " ".join(pattern_strings)
        assert "label" in patterns_text.lower()
        assert "target" in patterns_text.lower()
        assert "future" in patterns_text.lower()

    def test_safe_patterns_exist(self):
        """Verify safe patterns list is populated."""
        assert len(SAFE_PATTERNS) > 0
        patterns_text = " ".join(SAFE_PATTERNS)
        assert "lag" in patterns_text.lower()
        assert "rolling" in patterns_text.lower()

"""Tests for the risk scoring module (Prompt 5)."""
import pytest

from app.services.risk_scoring import (
    compute_risk_adjusted_score,
    check_too_good_to_be_true,
    get_model_risk_status,
    format_promotion_block_message,
)


class TestComputeRiskAdjustedScore:
    """Tests for compute_risk_adjusted_score function."""

    def test_no_penalties_low_risk(self):
        """Test that no penalties are applied for low-risk models."""
        score = compute_risk_adjusted_score(
            primary_metric=0.85,
            overfitting_risk="low",
            leakage_suspected=False,
            time_split_suspicious=False,
        )
        assert score == 0.85

    def test_medium_overfitting_penalty(self):
        """Test 0.05 penalty for medium overfitting risk."""
        score = compute_risk_adjusted_score(
            primary_metric=0.85,
            overfitting_risk="medium",
            leakage_suspected=False,
            time_split_suspicious=False,
        )
        assert score == pytest.approx(0.80, abs=0.001)

    def test_high_overfitting_penalty(self):
        """Test 0.10 penalty for high overfitting risk."""
        score = compute_risk_adjusted_score(
            primary_metric=0.85,
            overfitting_risk="high",
            leakage_suspected=False,
            time_split_suspicious=False,
        )
        assert score == pytest.approx(0.75, abs=0.001)

    def test_leakage_penalty(self):
        """Test 0.15 penalty for suspected leakage."""
        score = compute_risk_adjusted_score(
            primary_metric=0.85,
            overfitting_risk="low",
            leakage_suspected=True,
            time_split_suspicious=False,
        )
        assert score == pytest.approx(0.70, abs=0.001)

    def test_time_split_penalty(self):
        """Test 0.05 penalty for time-split issues."""
        score = compute_risk_adjusted_score(
            primary_metric=0.85,
            overfitting_risk="low",
            leakage_suspected=False,
            time_split_suspicious=True,
        )
        assert score == pytest.approx(0.80, abs=0.001)

    def test_combined_penalties(self):
        """Test that penalties are cumulative."""
        score = compute_risk_adjusted_score(
            primary_metric=0.85,
            overfitting_risk="high",  # -0.10
            leakage_suspected=True,    # -0.15
            time_split_suspicious=True,  # -0.05
        )
        # 0.85 - 0.10 - 0.15 - 0.05 = 0.55
        assert score == pytest.approx(0.55, abs=0.001)


class TestCheckTooGoodToBeTrue:
    """Tests for check_too_good_to_be_true function."""

    def test_non_time_based_not_flagged(self):
        """Test that non-time-based tasks are not flagged."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=False,
            task_type="binary",
            best_val_metric=0.95,
            primary_metric="roc_auc",
        )
        assert is_tgtbt is False
        assert warning is None

    def test_regression_not_flagged(self):
        """Test that regression tasks are not flagged."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=True,
            task_type="regression",
            best_val_metric=0.95,
            primary_metric="r2",
        )
        assert is_tgtbt is False
        assert warning is None

    def test_high_auc_flagged(self):
        """Test that high AUC on time-based classification is flagged."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=True,
            task_type="binary",
            best_val_metric=0.85,
            primary_metric="roc_auc",
        )
        assert is_tgtbt is True
        assert warning is not None
        assert "AUC" in warning
        assert "0.850" in warning

    def test_moderate_auc_not_flagged(self):
        """Test that moderate AUC is not flagged."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=True,
            task_type="binary",
            best_val_metric=0.75,
            primary_metric="roc_auc",
        )
        assert is_tgtbt is False
        assert warning is None

    def test_high_mcc_flagged(self):
        """Test that high MCC on time-based classification is flagged."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=True,
            task_type="binary",
            best_val_metric=0.65,
            primary_metric="accuracy",
            additional_metrics={"mcc": 0.55},
        )
        assert is_tgtbt is True
        assert warning is not None
        assert "MCC" in warning

    def test_high_accuracy_on_binary_flagged(self):
        """Test that very high accuracy on time-based binary is flagged."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=True,
            task_type="binary",
            best_val_metric=0.90,
            primary_metric="accuracy",
        )
        assert is_tgtbt is True
        assert warning is not None
        assert "Accuracy" in warning

    # Prompt 7 Step 6: expected_metric_range tests
    def test_exceeds_expected_metric_range_flagged(self):
        """Test that exceeding expected_metric_range is flagged (Prompt 7 Step 6)."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=False,  # Non-time-based to isolate expected_metric_range
            task_type="binary",
            best_val_metric=0.89,
            primary_metric="roc_auc",
            expected_metric_range={
                "metric": "roc_auc",
                "lower_bound": 0.60,
                "upper_bound": 0.75,
            },
        )
        assert is_tgtbt is True
        assert warning is not None
        assert "0.89" in warning or "0.890" in warning
        assert "0.75" in warning
        assert "expected" in warning.lower()

    def test_within_expected_metric_range_not_flagged(self):
        """Test that metrics within expected_metric_range are not flagged (Prompt 7 Step 6)."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=False,
            task_type="binary",
            best_val_metric=0.72,
            primary_metric="roc_auc",
            expected_metric_range={
                "metric": "roc_auc",
                "lower_bound": 0.60,
                "upper_bound": 0.75,
            },
        )
        assert is_tgtbt is False
        assert warning is None

    def test_expected_metric_range_plus_time_based(self):
        """Test that expected_metric_range combines with time-based checks (Prompt 7 Step 6)."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=True,  # Time-based adds additional heuristics
            task_type="binary",
            best_val_metric=0.85,  # Exceeds expected upper 0.75 AND triggers AUC > 0.80
            primary_metric="roc_auc",
            expected_metric_range={
                "metric": "roc_auc",
                "lower_bound": 0.60,
                "upper_bound": 0.75,
            },
        )
        assert is_tgtbt is True
        assert warning is not None
        # Should mention both expected range AND suspiciously high AUC
        assert "expected" in warning.lower() or "realistic" in warning.lower()

    def test_expected_metric_range_different_metric_not_matched(self):
        """Test that expected_metric_range for different metric is not matched (Prompt 7 Step 6)."""
        is_tgtbt, warning = check_too_good_to_be_true(
            is_time_based=False,
            task_type="regression",
            best_val_metric=0.95,  # High value but for RMSE not AUC
            primary_metric="rmse",
            expected_metric_range={
                "metric": "roc_auc",  # Different metric
                "lower_bound": 0.60,
                "upper_bound": 0.75,
            },
        )
        # No match because metric is different
        assert is_tgtbt is False


class TestGetModelRiskStatus:
    """Tests for get_model_risk_status function."""

    def test_low_risk_no_override(self):
        """Test that low risk does not require override."""
        risk_level, requires_override, reason = get_model_risk_status(
            overfitting_risk="low",
            leakage_suspected=False,
            time_split_suspicious=False,
        )
        assert risk_level == "low"
        assert requires_override is False
        assert "No significant risks" in reason

    def test_high_overfitting_requires_override(self):
        """Test that high overfitting requires override."""
        risk_level, requires_override, reason = get_model_risk_status(
            overfitting_risk="high",
            leakage_suspected=False,
            time_split_suspicious=False,
        )
        assert risk_level == "high"
        assert requires_override is True
        assert "High overfitting" in reason

    def test_leakage_requires_override(self):
        """Test that suspected leakage requires override."""
        risk_level, requires_override, reason = get_model_risk_status(
            overfitting_risk="low",
            leakage_suspected=True,
            time_split_suspicious=False,
        )
        assert risk_level == "critical"
        assert requires_override is True
        assert "leakage" in reason.lower()

    def test_tgtbt_requires_override(self):
        """Test that TGTBT requires override."""
        risk_level, requires_override, reason = get_model_risk_status(
            overfitting_risk="low",
            leakage_suspected=False,
            time_split_suspicious=False,
            too_good_to_be_true=True,
        )
        assert risk_level == "critical"
        assert requires_override is True
        assert "too good to be true" in reason.lower()

    def test_time_split_medium_risk(self):
        """Test that time-split issues result in medium risk."""
        risk_level, requires_override, reason = get_model_risk_status(
            overfitting_risk="low",
            leakage_suspected=False,
            time_split_suspicious=True,
        )
        assert risk_level == "medium"
        # Time-split alone doesn't require override
        assert requires_override is False


class TestFormatPromotionBlockMessage:
    """Tests for format_promotion_block_message function."""

    def test_message_contains_risk_level(self):
        """Test that message contains the risk level."""
        message = format_promotion_block_message(
            risk_level="high",
            reasons="Test reason",
        )
        assert "high" in message
        assert "Test reason" in message

    def test_message_mentions_override(self):
        """Test that message mentions override requirement."""
        message = format_promotion_block_message(
            risk_level="critical",
            reasons="Leakage detected",
        )
        assert "override_reason" in message
        assert "lab notebook" in message.lower()

"""Tests for validation samples functionality."""
import pytest
import uuid
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from app.models.validation_sample import ValidationSample
from app.models.model_version import ModelVersion
from app.services.automl_runner import (
    ValidationPrediction,
    AutoMLResult,
    TabularRunner,
)
from app.tasks.experiment_tasks import save_validation_samples


class TestValidationPrediction:
    """Tests for ValidationPrediction dataclass."""

    def test_create_validation_prediction_regression(self):
        """Test creating a validation prediction for regression."""
        pred = ValidationPrediction(
            row_index=0,
            features={"feature1": 1.5, "feature2": "category_a"},
            target_value=10.0,
            predicted_value=9.5,
            error_value=-0.5,
            absolute_error=0.5,
        )
        assert pred.row_index == 0
        assert pred.features == {"feature1": 1.5, "feature2": "category_a"}
        assert pred.target_value == 10.0
        assert pred.predicted_value == 9.5
        assert pred.error_value == -0.5
        assert pred.absolute_error == 0.5
        assert pred.prediction_probabilities is None

    def test_create_validation_prediction_classification(self):
        """Test creating a validation prediction for classification."""
        pred = ValidationPrediction(
            row_index=5,
            features={"feature1": 2.0},
            target_value="class_a",
            predicted_value="class_b",
            error_value=1.0,
            absolute_error=1.0,
            prediction_probabilities={"class_a": 0.3, "class_b": 0.7},
        )
        assert pred.target_value == "class_a"
        assert pred.predicted_value == "class_b"
        assert pred.prediction_probabilities == {"class_a": 0.3, "class_b": 0.7}


class TestAutoMLResultWithValidation:
    """Tests for AutoMLResult with validation predictions."""

    def test_automl_result_includes_validation_predictions(self):
        """Test that AutoMLResult can include validation predictions."""
        predictions = [
            ValidationPrediction(
                row_index=i,
                features={"x": float(i)},
                target_value=float(i * 2),
                predicted_value=float(i * 2 + 0.1),
                error_value=0.1,
                absolute_error=0.1,
            )
            for i in range(5)
        ]

        result = AutoMLResult(
            leaderboard=[{"model": "TestModel", "score_val": 0.9}],
            best_model_name="TestModel",
            artifact_path="/tmp/model",
            validation_predictions=predictions,
        )

        assert len(result.validation_predictions) == 5
        assert result.validation_predictions[0].row_index == 0
        assert result.validation_predictions[4].row_index == 4

    def test_automl_result_empty_validation_predictions(self):
        """Test that AutoMLResult defaults to empty validation predictions."""
        result = AutoMLResult(
            leaderboard=[],
            best_model_name="TestModel",
            artifact_path="/tmp/model",
        )
        assert result.validation_predictions == []


class TestSaveValidationSamples:
    """Tests for save_validation_samples function."""

    def test_save_empty_predictions(self):
        """Test saving empty predictions list."""
        mock_db = MagicMock()
        model_version_id = uuid.uuid4()

        count = save_validation_samples(
            db=mock_db,
            model_version_id=model_version_id,
            validation_predictions=[],
        )

        assert count == 0
        mock_db.bulk_save_objects.assert_not_called()

    def test_save_predictions_basic(self):
        """Test saving basic predictions."""
        mock_db = MagicMock()
        model_version_id = uuid.uuid4()

        predictions = [
            ValidationPrediction(
                row_index=i,
                features={"x": float(i)},
                target_value=float(i),
                predicted_value=float(i + 0.1),
                error_value=0.1,
                absolute_error=0.1,
            )
            for i in range(5)
        ]

        count = save_validation_samples(
            db=mock_db,
            model_version_id=model_version_id,
            validation_predictions=predictions,
        )

        assert count == 5
        mock_db.bulk_save_objects.assert_called_once()
        mock_db.commit.assert_called_once()

        # Verify the samples passed to bulk_save_objects
        saved_samples = mock_db.bulk_save_objects.call_args[0][0]
        assert len(saved_samples) == 5
        assert all(isinstance(s, ValidationSample) for s in saved_samples)
        assert saved_samples[0].row_index == 0
        assert saved_samples[0].target_value == "0.0"
        assert saved_samples[0].predicted_value == "0.1"

    def test_save_predictions_with_limit(self):
        """Test saving predictions with max_samples limit."""
        mock_db = MagicMock()
        model_version_id = uuid.uuid4()

        # Create 100 predictions
        predictions = [
            ValidationPrediction(
                row_index=i,
                features={"x": float(i)},
                target_value=float(i),
                predicted_value=float(i),
            )
            for i in range(100)
        ]

        # Limit to 10 samples
        count = save_validation_samples(
            db=mock_db,
            model_version_id=model_version_id,
            validation_predictions=predictions,
            max_samples=10,
        )

        assert count == 10
        saved_samples = mock_db.bulk_save_objects.call_args[0][0]
        assert len(saved_samples) == 10

    def test_save_predictions_with_probabilities(self):
        """Test saving classification predictions with probabilities."""
        mock_db = MagicMock()
        model_version_id = uuid.uuid4()

        predictions = [
            ValidationPrediction(
                row_index=0,
                features={"x": 1.0},
                target_value="class_a",
                predicted_value="class_a",
                error_value=0.0,
                absolute_error=0.0,
                prediction_probabilities={"class_a": 0.9, "class_b": 0.1},
            )
        ]

        count = save_validation_samples(
            db=mock_db,
            model_version_id=model_version_id,
            validation_predictions=predictions,
        )

        assert count == 1
        saved_sample = mock_db.bulk_save_objects.call_args[0][0][0]
        assert saved_sample.prediction_probabilities_json == {"class_a": 0.9, "class_b": 0.1}


class TestTabularRunnerCaptureValidation:
    """Tests for TabularRunner validation capture methods."""

    def test_serialize_value_basic_types(self):
        """Test value serialization for basic types."""
        runner = TabularRunner()

        assert runner._serialize_value(1) == 1
        assert runner._serialize_value(1.5) == 1.5
        assert runner._serialize_value("test") == "test"
        assert runner._serialize_value(None) is None

    def test_serialize_value_numpy_types(self):
        """Test value serialization for numpy types."""
        runner = TabularRunner()

        assert runner._serialize_value(np.int64(42)) == 42
        assert runner._serialize_value(np.float64(3.14)) == 3.14
        assert runner._serialize_value(np.array([1, 2, 3])) == [1, 2, 3]

    def test_serialize_value_pandas_na(self):
        """Test value serialization for pandas NA."""
        runner = TabularRunner()

        assert runner._serialize_value(pd.NA) is None
        assert runner._serialize_value(np.nan) is None

    def test_serialize_value_timestamp(self):
        """Test value serialization for pandas Timestamp."""
        runner = TabularRunner()

        ts = pd.Timestamp("2024-01-15 10:30:00")
        result = runner._serialize_value(ts)
        assert "2024-01-15" in result
        assert "10:30:00" in result


class TestValidationSampleModel:
    """Tests for ValidationSample SQLAlchemy model."""

    def test_validation_sample_creation(self):
        """Test creating a ValidationSample instance."""
        model_version_id = uuid.uuid4()

        sample = ValidationSample(
            model_version_id=model_version_id,
            row_index=0,
            features_json={"feature1": 1.0, "feature2": "category"},
            target_value="10.5",
            predicted_value="10.2",
            error_value=-0.3,
            absolute_error=0.3,
            prediction_probabilities_json=None,
        )

        assert sample.model_version_id == model_version_id
        assert sample.row_index == 0
        assert sample.features_json == {"feature1": 1.0, "feature2": "category"}
        assert sample.target_value == "10.5"
        assert sample.predicted_value == "10.2"
        assert sample.error_value == -0.3
        assert sample.absolute_error == 0.3

    def test_validation_sample_with_probabilities(self):
        """Test ValidationSample with prediction probabilities."""
        sample = ValidationSample(
            model_version_id=uuid.uuid4(),
            row_index=1,
            features_json={"x": 5.0},
            target_value="class_a",
            predicted_value="class_b",
            error_value=1.0,
            absolute_error=1.0,
            prediction_probabilities_json={"class_a": 0.4, "class_b": 0.6},
        )

        assert sample.prediction_probabilities_json == {"class_a": 0.4, "class_b": 0.6}


class TestIntegrationValidationCapture:
    """Integration tests for validation prediction capture (requires mocking AutoGluon)."""

    def test_tabular_runner_captures_validation_predictions(self):
        """Test that TabularRunner captures validation predictions during training."""
        with patch('autogluon.tabular.TabularPredictor') as mock_predictor_class:
            # Create mock predictor
            mock_predictor = MagicMock()
            mock_predictor_class.return_value = mock_predictor

            # Mock predictor methods
            mock_predictor.model_best = "LightGBM"
            mock_predictor.leaderboard.return_value = pd.DataFrame([
                {"model": "LightGBM", "score_val": 0.95, "pred_time_val": 0.01, "fit_time": 10.0}
            ])
            mock_predictor.info.return_value = {"best_model_score_val": 0.95}
            mock_predictor.feature_importance.return_value = pd.DataFrame(
                {"importance": [0.5, 0.3, 0.2]},
                index=["feature1", "feature2", "feature3"]
            )

            # Mock predictions - return predictions matching the number of validation rows
            mock_predictor.predict.return_value = pd.Series([1.1, 2.1, 3.1, 4.1])
            mock_predictor.predict_proba.side_effect = Exception("Not classification")

            # Create test dataset
            dataset = pd.DataFrame({
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "feature3": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
                "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            })

            # Run experiment
            runner = TabularRunner(artifacts_dir="/tmp/test_artifacts")
            result = runner.run_experiment(
                dataset=dataset,
                target_column="target",
                task_type="regression",
                config={"time_limit": 1, "validation_split": 0.4},  # 40% = 4 rows
            )

            # Verify validation predictions were captured
            assert len(result.validation_predictions) > 0
            # With 10 rows and 0.4 split, we expect 4 validation samples
            assert len(result.validation_predictions) == 4

            # Verify prediction structure
            pred = result.validation_predictions[0]
            assert isinstance(pred, ValidationPrediction)
            assert "feature1" in pred.features
            assert "feature2" in pred.features
            assert "feature3" in pred.features
            assert pred.target_value is not None
            assert pred.predicted_value is not None


class TestErrorDistribution:
    """Tests for error distribution analysis."""

    def test_regression_error_distribution(self):
        """Test that regression errors look reasonable (not all zeros, not NaN)."""
        # Simulate validation predictions from a regression model
        predictions = []
        np.random.seed(42)

        for i in range(100):
            target = float(i) + np.random.normal(0, 5)
            predicted = target + np.random.normal(0, 2)  # Add some prediction error
            error = predicted - target

            predictions.append(ValidationPrediction(
                row_index=i,
                features={"x": float(i)},
                target_value=target,
                predicted_value=predicted,
                error_value=error,
                absolute_error=abs(error),
            ))

        # Verify error distribution properties
        errors = [p.error_value for p in predictions]
        abs_errors = [p.absolute_error for p in predictions]

        # Errors should not all be zero
        assert not all(e == 0 for e in errors)

        # Errors should not be NaN
        assert not any(np.isnan(e) for e in errors if e is not None)

        # Absolute errors should all be non-negative
        assert all(e >= 0 for e in abs_errors if e is not None)

        # Mean absolute error should be reasonable (not zero, not huge)
        mean_abs_error = np.mean(abs_errors)
        assert 0 < mean_abs_error < 10  # Given our simulation parameters

        # Errors should have some positive and some negative values
        positive_errors = sum(1 for e in errors if e > 0)
        negative_errors = sum(1 for e in errors if e < 0)
        assert positive_errors > 10
        assert negative_errors > 10

    def test_classification_error_distribution(self):
        """Test that classification errors are binary (0 or 1)."""
        predictions = []

        for i in range(100):
            is_correct = i % 3 != 0  # 2/3 correct predictions
            error = 0.0 if is_correct else 1.0

            predictions.append(ValidationPrediction(
                row_index=i,
                features={"x": float(i)},
                target_value="class_a" if is_correct else "class_b",
                predicted_value="class_a",
                error_value=error,
                absolute_error=error,
            ))

        errors = [p.error_value for p in predictions]

        # All errors should be 0 or 1
        assert all(e in (0.0, 1.0) for e in errors)

        # Should have both correct and incorrect predictions
        correct_count = sum(1 for e in errors if e == 0.0)
        incorrect_count = sum(1 for e in errors if e == 1.0)

        assert correct_count > 0
        assert incorrect_count > 0
        assert correct_count + incorrect_count == 100

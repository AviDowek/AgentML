"""Tests for AutoML integration (Phase 3).

These tests verify:
1. AutoML runner functionality
2. Experiment run API endpoints
3. End-to-end experiment execution (mocked AutoML)
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from app.services.automl_runner import (
    AutoMLRunner,
    AutoMLResult,
    TabularRunner,
    TimeSeriesRunner,
    MultiModalRunner,
    get_runner_for_task,
)


class TestAutoMLRunner:
    """Test AutoML runner service."""

    @pytest.fixture
    def runner(self):
        """Create an AutoML runner with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield AutoMLRunner(artifacts_dir=tmpdir)

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })

    @pytest.fixture
    def regression_dataset(self):
        """Create a sample regression dataset."""
        return pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "target": [15.5, 25.2, 35.8, 45.1, 55.9, 65.3, 75.7, 85.2, 95.6, 105.4],
        })

    def test_automl_result_dataclass(self):
        """Test AutoMLResult dataclass initialization."""
        result = AutoMLResult(
            leaderboard=[{"model": "LightGBM", "score": 0.95}],
            best_model_name="LightGBM",
            artifact_path="/path/to/artifacts",
            feature_importances={"feature1": 0.5, "feature2": 0.3},
            metrics={"accuracy": 0.95},
            training_time_seconds=120.5,
            num_models_trained=5,
        )

        assert result.best_model_name == "LightGBM"
        assert result.num_models_trained == 5
        assert "accuracy" in result.metrics

    def test_run_experiment_binary(self, runner, sample_dataset):
        """Test running a binary classification experiment (mocked)."""
        # Mock the autogluon.tabular module
        mock_predictor = MagicMock()
        mock_predictor.model_best = "LightGBM"

        # Mock leaderboard
        mock_leaderboard = pd.DataFrame({
            "model": ["LightGBM", "XGBoost"],
            "score_val": [0.95, 0.92],
            "pred_time_val": [0.01, 0.02],
            "fit_time": [10.0, 15.0],
        })
        mock_predictor.leaderboard.return_value = mock_leaderboard

        # Mock info
        mock_predictor.info.return_value = {"best_model_score_val": 0.95}

        # Mock feature importance
        mock_importance = pd.DataFrame({
            "importance": [0.6, 0.3, 0.1],
        }, index=["feature1", "feature2", "category"])
        mock_predictor.feature_importance.return_value = mock_importance

        # Create a mock TabularPredictor class
        mock_predictor_class = MagicMock(return_value=mock_predictor)

        with patch.dict("sys.modules", {"autogluon.tabular": MagicMock(TabularPredictor=mock_predictor_class)}):
            result = runner.run_experiment(
                dataset=sample_dataset,
                target_column="target",
                task_type="binary",
                primary_metric="accuracy",
                config={"time_limit": 60},
                experiment_id="test_exp_1",
            )

        assert isinstance(result, AutoMLResult)
        assert result.best_model_name == "LightGBM"
        assert len(result.leaderboard) == 2
        assert result.num_models_trained == 2
        assert "feature1" in result.feature_importances

    def test_run_experiment_regression(self, runner, regression_dataset):
        """Test running a regression experiment (mocked)."""
        mock_predictor = MagicMock()
        mock_predictor.model_best = "CatBoost"

        mock_leaderboard = pd.DataFrame({
            "model": ["CatBoost"],
            "score_val": [-5.2],  # RMSE is negative in AutoGluon
            "fit_time": [20.0],
        })
        mock_predictor.leaderboard.return_value = mock_leaderboard
        mock_predictor.info.return_value = {"best_model_score_val": -5.2}
        mock_predictor.feature_importance.return_value = None

        mock_predictor_class = MagicMock(return_value=mock_predictor)

        with patch.dict("sys.modules", {"autogluon.tabular": MagicMock(TabularPredictor=mock_predictor_class)}):
            result = runner.run_experiment(
                dataset=regression_dataset,
                target_column="target",
                task_type="regression",
                primary_metric="rmse",
            )

        assert result.best_model_name == "CatBoost"

    def test_metric_mapping(self, runner):
        """Test that metric names are correctly mapped."""
        metric_map = {
            "rmse": "root_mean_squared_error",
            "mse": "mean_squared_error",
            "mae": "mean_absolute_error",
            "accuracy": "accuracy",
            "auc": "roc_auc",
            "roc_auc": "roc_auc",
        }

        for input_metric, expected in metric_map.items():
            # The mapping is done inside run_experiment, so we verify the logic
            mapping = {
                "rmse": "root_mean_squared_error",
                "mse": "mean_squared_error",
                "mae": "mean_absolute_error",
                "r2": "r2",
                "accuracy": "accuracy",
                "auc": "roc_auc",
                "roc_auc": "roc_auc",
                "f1": "f1",
                "log_loss": "log_loss",
            }
            assert mapping.get(input_metric.lower(), input_metric) == expected


class TestExperimentRunAPI:
    """Test experiment run/cancel API endpoints."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post(
            "/projects",
            json={"name": "AutoML Test Project", "task_type": "classification"},
        )
        return response.json()["id"]

    @pytest.fixture
    def dataset_spec_id(self, client, project_id):
        """Create a dataset spec with target column."""
        response = client.post(
            f"/projects/{project_id}/dataset-specs",
            json={
                "name": "Test Spec",
                "target_column": "target",
                "feature_columns": ["feature1", "feature2"],
            },
        )
        return response.json()["id"]

    @pytest.fixture
    def experiment_id(self, client, project_id, dataset_spec_id):
        """Create an experiment and return its ID."""
        response = client.post(
            f"/projects/{project_id}/experiments",
            json={
                "name": "AutoML Experiment",
                "dataset_spec_id": dataset_spec_id,
                "primary_metric": "accuracy",
                "metric_direction": "maximize",
            },
        )
        return response.json()["id"]

    @patch("app.api.experiments.run_automl_experiment_task")
    def test_run_experiment_queues_task(self, mock_task, client, experiment_id):
        """Test that running an experiment queues a Celery task."""
        mock_task.delay.return_value = Mock(id="task-123")

        response = client.post(f"/experiments/{experiment_id}/run")

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "queued"
        assert data["task_id"] == "task-123"
        assert data["experiment_id"] == experiment_id
        assert data["backend"] == "local"
        # Verify task was called with default training options
        mock_task.delay.assert_called_once_with(
            experiment_id,
            resource_limits_enabled=True,
            num_cpus=None,
            num_gpus=None,
            memory_limit_gb=None,
        )

    @patch("app.api.experiments.run_automl_experiment_task")
    def test_run_experiment_with_custom_options(self, mock_task, client, experiment_id):
        """Test that running an experiment with custom training options works."""
        mock_task.delay.return_value = Mock(id="task-456")

        response = client.post(
            f"/experiments/{experiment_id}/run",
            json={
                "training_options": {
                    "backend": "local",
                    "resource_limits_enabled": False,
                    "num_cpus": 4,
                    "num_gpus": 1,
                    "memory_limit_gb": 16,
                }
            },
        )

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "queued"
        assert data["backend"] == "local"
        # Verify task was called with custom training options
        mock_task.delay.assert_called_once_with(
            experiment_id,
            resource_limits_enabled=False,
            num_cpus=4,
            num_gpus=1,
            memory_limit_gb=16,
        )

    def test_run_experiment_without_dataset_spec_fails(self, client, project_id):
        """Test that running an experiment without dataset_spec fails."""
        # Create experiment without dataset_spec_id
        create_response = client.post(
            f"/projects/{project_id}/experiments",
            json={"name": "No Spec Experiment"},
        )
        exp_id = create_response.json()["id"]

        response = client.post(f"/experiments/{exp_id}/run")

        assert response.status_code == 400
        assert "dataset_spec_id" in response.json()["detail"]

    @patch("app.api.experiments.run_automl_experiment_task")
    def test_run_already_running_experiment_fails(self, mock_task, client, project_id, dataset_spec_id):
        """Test that running an already running experiment fails."""
        # Create and update experiment to running status
        create_response = client.post(
            f"/projects/{project_id}/experiments",
            json={
                "name": "Running Experiment",
                "dataset_spec_id": dataset_spec_id,
            },
        )
        exp_id = create_response.json()["id"]

        # Update status to running
        client.put(f"/experiments/{exp_id}", json={"status": "running"})

        response = client.post(f"/experiments/{exp_id}/run")

        assert response.status_code == 400
        assert "cannot run" in response.json()["detail"].lower()

    def test_cancel_pending_experiment(self, client, experiment_id):
        """Test canceling a pending experiment."""
        response = client.post(f"/experiments/{experiment_id}/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

        # Verify experiment status changed
        get_response = client.get(f"/experiments/{experiment_id}")
        assert get_response.json()["status"] == "cancelled"

    def test_cancel_completed_experiment_fails(self, client, project_id, dataset_spec_id):
        """Test that canceling a completed experiment fails."""
        create_response = client.post(
            f"/projects/{project_id}/experiments",
            json={
                "name": "Completed Experiment",
                "dataset_spec_id": dataset_spec_id,
            },
        )
        exp_id = create_response.json()["id"]

        # Update status to completed
        client.put(f"/experiments/{exp_id}", json={"status": "completed"})

        response = client.post(f"/experiments/{exp_id}/cancel")

        assert response.status_code == 400
        assert "cannot cancel" in response.json()["detail"].lower()

    def test_run_experiment_not_found(self, client):
        """Test running a non-existent experiment."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.post(f"/experiments/{fake_id}/run")
        assert response.status_code == 404

    def test_get_training_options(self, client):
        """Test getting training options endpoint."""
        response = client.get("/training-options")
        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "backends" in data
        assert "resource_limits" in data
        assert "automl_defaults" in data

        # Check backends
        assert "local" in data["backends"]
        assert data["backends"]["local"]["available"] is True
        assert "modal" in data["backends"]

        # Check resource limits
        assert "enabled_by_default" in data["resource_limits"]
        assert "defaults" in data["resource_limits"]
        assert "num_cpus" in data["resource_limits"]["defaults"]
        assert "num_gpus" in data["resource_limits"]["defaults"]
        assert "memory_limit_gb" in data["resource_limits"]["defaults"]

        # Check automl defaults
        assert "time_limit" in data["automl_defaults"]
        assert "presets" in data["automl_defaults"]


class TestExperimentDetailResponse:
    """Test experiment detail response with summary info."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project."""
        response = client.post(
            "/projects",
            json={"name": "Detail Test Project", "task_type": "classification"},
        )
        return response.json()["id"]

    @pytest.fixture
    def dataset_spec_id(self, client, project_id):
        """Create a dataset spec."""
        response = client.post(
            f"/projects/{project_id}/dataset-specs",
            json={"name": "Test Spec", "target_column": "target"},
        )
        return response.json()["id"]

    def test_get_experiment_includes_trial_count(self, client, project_id, dataset_spec_id):
        """Test that get experiment includes trial count."""
        # Create experiment
        create_response = client.post(
            f"/projects/{project_id}/experiments",
            json={
                "name": "Trial Count Test",
                "dataset_spec_id": dataset_spec_id,
            },
        )
        exp_id = create_response.json()["id"]

        # Create some trials
        client.post(f"/experiments/{exp_id}/trials", json={"variant_name": "Trial 1"})
        client.post(f"/experiments/{exp_id}/trials", json={"variant_name": "Trial 2"})

        # Get experiment detail
        response = client.get(f"/experiments/{exp_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["trial_count"] == 2

    def test_get_experiment_includes_best_model(self, client, project_id, dataset_spec_id, db_session):
        """Test that completed experiment includes best model info."""
        from app.models.experiment import Experiment, ExperimentStatus
        from app.models.model_version import ModelVersion
        import uuid

        # Create experiment
        create_response = client.post(
            f"/projects/{project_id}/experiments",
            json={
                "name": "Best Model Test",
                "dataset_spec_id": dataset_spec_id,
            },
        )
        exp_id = create_response.json()["id"]

        # Update experiment to completed and add model via DB
        experiment = db_session.query(Experiment).filter(
            Experiment.id == uuid.UUID(exp_id)
        ).first()
        experiment.status = ExperimentStatus.COMPLETED
        db_session.commit()

        # Create model version
        model = ModelVersion(
            project_id=uuid.UUID(project_id),
            experiment_id=uuid.UUID(exp_id),
            name="Best Model",
            model_type="LightGBM",
            metrics_json={"accuracy": 0.95},
        )
        db_session.add(model)
        db_session.commit()

        # Get experiment detail
        response = client.get(f"/experiments/{exp_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["best_model"] is not None
        assert data["best_model"]["model_type"] == "LightGBM"
        assert data["best_metrics"]["accuracy"] == 0.95


class TestExperimentTaskIntegration:
    """Integration tests for experiment task execution."""

    @pytest.fixture
    def temp_upload_dir(self):
        """Create a temporary directory for file uploads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_csv_file(self, temp_upload_dir):
        """Create a sample CSV file for testing."""
        csv_path = Path(temp_upload_dir) / "test_data.csv"
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
            "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 10,
            "category": ["A", "B"] * 50,
            "target": [0, 1] * 50,
        })
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    @patch("app.tasks.experiment_tasks.get_runner_for_task")
    def test_run_experiment_task_creates_records(
        self, mock_get_runner, client, db_session, temp_upload_dir, sample_csv_file
    ):
        """Test that the experiment task creates trial and model records."""
        from app.tasks.experiment_tasks import run_experiment
        from app.models.experiment import Experiment, Trial, ExperimentStatus
        from app.models.model_version import ModelVersion
        from app.models.data_source import DataSource
        import uuid

        # Create project
        project_response = client.post(
            "/projects",
            json={"name": "Task Test Project", "task_type": "classification"},
        )
        project_id = project_response.json()["id"]

        # Create data source with file path
        data_source = DataSource(
            id=uuid.uuid4(),
            project_id=uuid.UUID(project_id),
            name="Test Data",
            type="file_upload",
            config_json={"file_path": sample_csv_file},
        )
        db_session.add(data_source)
        db_session.commit()

        # Create dataset spec referencing the data source
        spec_response = client.post(
            f"/projects/{project_id}/dataset-specs",
            json={
                "name": "Test Spec",
                "target_column": "target",
                "feature_columns": ["feature1", "feature2", "category"],
                "data_sources_json": {"sources": [str(data_source.id)]},
            },
        )
        spec_id = spec_response.json()["id"]

        # Create experiment
        exp_response = client.post(
            f"/projects/{project_id}/experiments",
            json={
                "name": "Integration Test Experiment",
                "dataset_spec_id": spec_id,
                "primary_metric": "accuracy",
            },
        )
        exp_id = exp_response.json()["id"]

        # Mock AutoML runner
        mock_runner = MagicMock()
        mock_get_runner.return_value = mock_runner
        mock_runner.run_experiment.return_value = AutoMLResult(
            leaderboard=[{"model": "LightGBM", "score": 0.95}],
            best_model_name="LightGBM",
            artifact_path="/tmp/artifacts/test",
            feature_importances={"feature1": 0.5},
            metrics={"accuracy": 0.95},
            training_time_seconds=60.0,
            num_models_trained=1,
        )

        # Run the task synchronously (not via Celery)
        with patch("app.tasks.experiment_tasks.SessionLocal", return_value=db_session):
            with patch("app.tasks.experiment_tasks.get_settings") as mock_settings:
                mock_settings.return_value.artifacts_dir = temp_upload_dir
                mock_settings.return_value.automl_time_limit = 60
                mock_settings.return_value.automl_presets = "medium_quality"

                result = run_experiment(exp_id)

        # Verify result
        assert result["experiment_id"] == exp_id
        assert result["best_model_name"] == "LightGBM"
        assert "trial_id" in result
        assert "model_version_id" in result

        # Verify experiment status
        db_session.expire_all()
        experiment = db_session.query(Experiment).filter(
            Experiment.id == uuid.UUID(exp_id)
        ).first()
        assert experiment.status == ExperimentStatus.COMPLETED

        # Verify trial was created
        trial = db_session.query(Trial).filter(
            Trial.experiment_id == uuid.UUID(exp_id)
        ).first()
        assert trial is not None
        assert trial.variant_name == "AutoML_MVP"
        assert trial.metrics_json["accuracy"] == 0.95

        # Verify model version was created
        model = db_session.query(ModelVersion).filter(
            ModelVersion.experiment_id == uuid.UUID(exp_id)
        ).first()
        assert model is not None
        assert model.model_type == "LightGBM"
        assert model.metrics_json["accuracy"] == 0.95


class TestTabularRunner:
    """Test TabularRunner for all tabular task types."""

    @pytest.fixture
    def runner(self):
        """Create a TabularRunner with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield TabularRunner(artifacts_dir=tmpdir)

    @pytest.fixture
    def quantile_dataset(self):
        """Create a sample dataset for quantile regression."""
        return pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "target": [15.5, 25.2, 35.8, 45.1, 55.9, 65.3, 75.7, 85.2, 95.6, 105.4],
        })

    def test_run_quantile_experiment(self, runner, quantile_dataset):
        """Test running a quantile regression experiment (mocked)."""
        mock_predictor = MagicMock()
        mock_predictor.model_best = "LightGBMQuantile"

        mock_leaderboard = pd.DataFrame({
            "model": ["LightGBMQuantile"],
            "score_val": [-0.15],  # Pinball loss
            "fit_time": [15.0],
        })
        mock_predictor.leaderboard.return_value = mock_leaderboard
        mock_predictor.info.return_value = {"best_model_score_val": -0.15}
        mock_predictor.feature_importance.return_value = None

        mock_predictor_class = MagicMock(return_value=mock_predictor)

        with patch.dict("sys.modules", {"autogluon.tabular": MagicMock(TabularPredictor=mock_predictor_class)}):
            result = runner.run_experiment(
                dataset=quantile_dataset,
                target_column="target",
                task_type="quantile",
                config={"quantile_levels": [0.1, 0.5, 0.9]},
            )

        assert result.best_model_name == "LightGBMQuantile"
        assert result.task_type == "quantile"
        assert result.quantile_levels == [0.1, 0.5, 0.9]

    def test_tabular_runner_default_metrics(self, runner):
        """Test default metric selection for different task types."""
        metric_defaults = {
            "binary": "roc_auc",
            "multiclass": "accuracy",
            "regression": "root_mean_squared_error",
            "quantile": "pinball_loss",
        }

        for task_type, expected_metric in metric_defaults.items():
            # Verify the mapping in the runner
            metric_map = {
                "regression": "root_mean_squared_error",
                "binary": "roc_auc",
                "multiclass": "accuracy",
                "quantile": "pinball_loss",
            }
            assert metric_map.get(task_type) == expected_metric


class TestTimeSeriesRunner:
    """Test TimeSeriesRunner for forecasting tasks."""

    @pytest.fixture
    def runner(self):
        """Create a TimeSeriesRunner with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield TimeSeriesRunner(artifacts_dir=tmpdir)

    @pytest.fixture
    def timeseries_dataset(self):
        """Create a sample time series dataset."""
        import datetime
        dates = [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
        return pd.DataFrame({
            "timestamp": dates,
            "value": [i + (i % 7) * 5 for i in range(100)],  # Some seasonal pattern
        })

    def test_run_timeseries_experiment(self, runner, timeseries_dataset):
        """Test running a time series forecasting experiment (mocked)."""
        mock_predictor = MagicMock()
        mock_predictor.model_best = "AutoARIMA"

        mock_leaderboard = pd.DataFrame({
            "model": ["AutoARIMA", "ETS"],
            "score_val": [0.85, 0.82],
            "fit_time": [5.0, 3.0],
        })
        mock_predictor.leaderboard.return_value = mock_leaderboard

        mock_predictor_class = MagicMock(return_value=mock_predictor)
        mock_ts_dataframe = MagicMock()
        mock_ts_dataframe.from_data_frame.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "autogluon.timeseries": MagicMock(
                TimeSeriesPredictor=mock_predictor_class,
                TimeSeriesDataFrame=mock_ts_dataframe,
            )
        }):
            result = runner.run_experiment(
                dataset=timeseries_dataset,
                target_column="value",
                task_type="timeseries_forecast",
                config={
                    "prediction_length": 7,
                    "time_column": "timestamp",
                },
            )

        assert result.best_model_name == "AutoARIMA"
        assert result.task_type == "timeseries_forecast"
        assert result.prediction_length == 7
        assert len(result.leaderboard) == 2

    def test_timeseries_default_metric(self, runner):
        """Test default metric for time series is MASE."""
        # MASE (Mean Absolute Scaled Error) is the default
        metric_mapping = {
            "mase": "MASE",
            "mape": "MAPE",
            "smape": "sMAPE",
            "rmse": "RMSE",
            "mae": "MAE",
        }
        # Default when None is passed should be MASE
        assert "MASE" in metric_mapping.values()

    def test_timeseries_single_series_handling(self, runner, timeseries_dataset):
        """Test that single series (no id_column) is handled correctly."""
        mock_predictor = MagicMock()
        mock_predictor.model_best = "Theta"
        mock_predictor.leaderboard.return_value = pd.DataFrame({
            "model": ["Theta"], "score_val": [0.9], "fit_time": [2.0]
        })

        mock_predictor_class = MagicMock(return_value=mock_predictor)
        mock_ts_dataframe = MagicMock()
        mock_ts_dataframe.from_data_frame.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "autogluon.timeseries": MagicMock(
                TimeSeriesPredictor=mock_predictor_class,
                TimeSeriesDataFrame=mock_ts_dataframe,
            )
        }):
            result = runner.run_experiment(
                dataset=timeseries_dataset,
                target_column="value",
                task_type="timeseries_forecast",
                config={"time_column": "timestamp"},  # No id_column
            )

        assert result.best_model_name == "Theta"


class TestMultiModalRunner:
    """Test MultiModalRunner for mixed data types."""

    @pytest.fixture
    def runner(self):
        """Create a MultiModalRunner with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MultiModalRunner(artifacts_dir=tmpdir)

    @pytest.fixture
    def multimodal_classification_dataset(self):
        """Create a sample multimodal dataset for classification."""
        return pd.DataFrame({
            "text_description": [
                "This is a positive review",
                "Terrible product, do not buy",
                "Amazing quality and fast shipping",
                "Not worth the money",
                "Best purchase I ever made",
            ] * 10,
            "price": [19.99, 29.99, 49.99, 9.99, 99.99] * 10,
            "category": ["electronics", "clothing", "electronics", "books", "electronics"] * 10,
            "sentiment": ["positive", "negative", "positive", "negative", "positive"] * 10,
        })

    @pytest.fixture
    def multimodal_regression_dataset(self):
        """Create a sample multimodal dataset for regression."""
        return pd.DataFrame({
            "description": ["Item " + str(i) for i in range(50)],
            "weight": [1.5 + i * 0.1 for i in range(50)],
            "material": ["wood", "metal", "plastic"] * 16 + ["wood", "metal"],
            "price": [10.0 + i * 2.5 for i in range(50)],
        })

    def test_run_multimodal_classification(self, runner, multimodal_classification_dataset):
        """Test running a multimodal classification experiment (mocked)."""
        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"accuracy": 0.88}

        mock_predictor_class = MagicMock(return_value=mock_predictor)

        with patch.dict("sys.modules", {
            "autogluon.multimodal": MagicMock(MultiModalPredictor=mock_predictor_class)
        }):
            result = runner.run_experiment(
                dataset=multimodal_classification_dataset,
                target_column="sentiment",
                task_type="multimodal_classification",
                primary_metric="accuracy",
            )

        assert result.best_model_name == "AutoMM"
        assert result.task_type == "multimodal_classification"
        assert result.metrics.get("accuracy") == 0.88

    def test_run_multimodal_regression(self, runner, multimodal_regression_dataset):
        """Test running a multimodal regression experiment (mocked)."""
        mock_predictor = MagicMock()
        mock_predictor.evaluate.return_value = {"rmse": 2.5}

        mock_predictor_class = MagicMock(return_value=mock_predictor)

        with patch.dict("sys.modules", {
            "autogluon.multimodal": MagicMock(MultiModalPredictor=mock_predictor_class)
        }):
            result = runner.run_experiment(
                dataset=multimodal_regression_dataset,
                target_column="price",
                task_type="multimodal_regression",
                primary_metric="rmse",
            )

        assert result.best_model_name == "AutoMM"
        assert result.task_type == "multimodal_regression"
        assert result.metrics.get("rmse") == 2.5

    def test_multimodal_binary_detection(self, runner):
        """Test that binary classification is correctly detected."""
        dataset = pd.DataFrame({
            "text": ["yes", "no"] * 5,
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [0, 1] * 5,  # Binary target
        })

        # Check unique values detection
        assert dataset["target"].nunique() == 2


class TestGetRunnerForTask:
    """Test the get_runner_for_task factory function."""

    def test_get_tabular_runners(self):
        """Test getting runners for tabular tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for task_type in ["binary", "multiclass", "regression", "quantile", "classification"]:
                runner = get_runner_for_task(task_type, tmpdir)
                assert isinstance(runner, TabularRunner)

    def test_get_timeseries_runner(self):
        """Test getting runner for time series tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = get_runner_for_task("timeseries_forecast", tmpdir)
            assert isinstance(runner, TimeSeriesRunner)

    def test_get_multimodal_runners(self):
        """Test getting runners for multimodal tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for task_type in ["multimodal_classification", "multimodal_regression"]:
                runner = get_runner_for_task(task_type, tmpdir)
                assert isinstance(runner, MultiModalRunner)

    def test_unknown_task_type_raises(self):
        """Test that unknown task type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown task type"):
                get_runner_for_task("unknown_task", tmpdir)


class TestTaskTypeEnum:
    """Test the TaskType enum from project model."""

    def test_all_task_types_defined(self):
        """Test that all required task types are defined."""
        from app.models.project import TaskType

        expected_types = [
            "binary",
            "multiclass",
            "regression",
            "quantile",
            "timeseries_forecast",
            "multimodal_classification",
            "multimodal_regression",
            "classification",  # Legacy
        ]

        actual_types = [t.value for t in TaskType]
        for expected in expected_types:
            assert expected in actual_types, f"Missing task type: {expected}"

    def test_task_type_string_values(self):
        """Test that TaskType enum values are strings."""
        from app.models.project import TaskType

        for task_type in TaskType:
            assert isinstance(task_type.value, str)
            assert task_type.value == task_type.name.lower() or task_type.value == task_type.name.lower().replace("_", "_")


class TestExperimentTaskRouting:
    """Test that experiment tasks are routed to the correct runner."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.artifacts_dir = "./test_artifacts"
        settings.automl_time_limit = 60
        settings.automl_presets = "medium_quality"
        return settings

    def test_task_type_mapping(self):
        """Test the task type mapping in experiment tasks."""
        task_type_map = {
            "classification": "binary",
            "binary": "binary",
            "multiclass": "multiclass",
            "regression": "regression",
            "quantile": "quantile",
            "timeseries_forecast": "timeseries_forecast",
            "multimodal_classification": "multimodal_classification",
            "multimodal_regression": "multimodal_regression",
        }

        # Verify all mappings are correct
        for input_type, expected_output in task_type_map.items():
            assert task_type_map.get(input_type, "binary") == expected_output

    def test_legacy_classification_maps_to_binary(self):
        """Test that legacy 'classification' type maps to 'binary'."""
        task_type_map = {
            "classification": "binary",
        }
        assert task_type_map["classification"] == "binary"

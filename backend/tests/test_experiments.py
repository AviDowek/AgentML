"""Tests for experiment and trial API endpoints."""
import pytest


class TestExperimentsCRUD:
    """Test experiment CRUD operations."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Test Project"})
        return response.json()["id"]

    @pytest.fixture
    def dataset_spec_id(self, client, project_id):
        """Create a dataset spec and return its ID."""
        response = client.post(
            f"/projects/{project_id}/dataset-specs",
            json={"name": "Test Spec", "target_column": "price"},
        )
        return response.json()["id"]

    def test_create_experiment(self, client, project_id, dataset_spec_id):
        """Test creating a new experiment."""
        response = client.post(
            f"/projects/{project_id}/experiments",
            json={
                "name": "Price Prediction Experiment",
                "description": "Testing different models for price prediction",
                "dataset_spec_id": dataset_spec_id,
                "primary_metric": "rmse",
                "metric_direction": "minimize",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Price Prediction Experiment"
        assert data["dataset_spec_id"] == dataset_spec_id
        assert data["primary_metric"] == "rmse"
        assert data["metric_direction"] == "minimize"
        assert data["status"] == "pending"

    def test_create_experiment_minimal(self, client, project_id):
        """Test creating an experiment with minimal fields."""
        response = client.post(
            f"/projects/{project_id}/experiments",
            json={"name": "Minimal Experiment"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal Experiment"
        assert data["dataset_spec_id"] is None

    def test_list_experiments(self, client, project_id):
        """Test listing experiments for a project."""
        client.post(
            f"/projects/{project_id}/experiments",
            json={"name": "Experiment 1"},
        )
        client.post(
            f"/projects/{project_id}/experiments",
            json={"name": "Experiment 2"},
        )

        response = client.get(f"/projects/{project_id}/experiments")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_experiment(self, client, project_id):
        """Test getting an experiment by ID."""
        create_response = client.post(
            f"/projects/{project_id}/experiments",
            json={"name": "Test Experiment"},
        )
        experiment_id = create_response.json()["id"]

        response = client.get(f"/experiments/{experiment_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == experiment_id
        assert data["name"] == "Test Experiment"

    def test_update_experiment(self, client, project_id):
        """Test updating an experiment."""
        create_response = client.post(
            f"/projects/{project_id}/experiments",
            json={"name": "Original"},
        )
        experiment_id = create_response.json()["id"]

        response = client.put(
            f"/experiments/{experiment_id}",
            json={
                "name": "Updated Experiment",
                "status": "running",
                "primary_metric": "accuracy",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Experiment"
        assert data["status"] == "running"
        assert data["primary_metric"] == "accuracy"

    def test_delete_experiment(self, client, project_id):
        """Test deleting an experiment."""
        create_response = client.post(
            f"/projects/{project_id}/experiments",
            json={"name": "To Delete"},
        )
        experiment_id = create_response.json()["id"]

        response = client.delete(f"/experiments/{experiment_id}")
        assert response.status_code == 204

        get_response = client.get(f"/experiments/{experiment_id}")
        assert get_response.status_code == 404


class TestTrialsCRUD:
    """Test trial CRUD operations."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Test Project"})
        return response.json()["id"]

    @pytest.fixture
    def experiment_id(self, client, project_id):
        """Create an experiment and return its ID."""
        response = client.post(
            f"/projects/{project_id}/experiments",
            json={"name": "Test Experiment"},
        )
        return response.json()["id"]

    def test_create_trial(self, client, experiment_id):
        """Test creating a new trial."""
        response = client.post(
            f"/experiments/{experiment_id}/trials",
            json={
                "variant_name": "LightGBM Default",
                "data_split_strategy": "random_80_20",
                "automl_config": {"model": "lightgbm", "time_limit": 300},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["variant_name"] == "LightGBM Default"
        assert data["data_split_strategy"] == "random_80_20"
        assert data["automl_config"] == {"model": "lightgbm", "time_limit": 300}
        assert data["status"] == "pending"

    def test_list_trials(self, client, experiment_id):
        """Test listing trials for an experiment."""
        client.post(
            f"/experiments/{experiment_id}/trials",
            json={"variant_name": "Trial 1"},
        )
        client.post(
            f"/experiments/{experiment_id}/trials",
            json={"variant_name": "Trial 2"},
        )

        response = client.get(f"/experiments/{experiment_id}/trials")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_trial(self, client, experiment_id):
        """Test getting a trial by ID."""
        create_response = client.post(
            f"/experiments/{experiment_id}/trials",
            json={"variant_name": "Test Trial"},
        )
        trial_id = create_response.json()["id"]

        response = client.get(f"/trials/{trial_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == trial_id
        assert data["variant_name"] == "Test Trial"

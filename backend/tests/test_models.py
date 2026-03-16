"""Tests for model version API endpoints."""
from unittest.mock import MagicMock, patch
import pytest


class TestModelVersionsCRUD:
    """Test model version CRUD operations."""

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

    @pytest.fixture
    def trial_id(self, client, experiment_id):
        """Create a trial and return its ID."""
        response = client.post(
            f"/experiments/{experiment_id}/trials",
            json={"variant_name": "Test Trial"},
        )
        return response.json()["id"]

    def test_create_model_version(self, client, project_id, experiment_id, trial_id):
        """Test creating a new model version."""
        response = client.post(
            f"/projects/{project_id}/models",
            json={
                "name": "LightGBM v1",
                "model_type": "LightGBM",
                "experiment_id": experiment_id,
                "trial_id": trial_id,
                "artifact_location": "/models/lightgbm_v1.pkl",
                "metrics_json": {"rmse": 0.15, "mae": 0.12},
                "feature_importances_json": {"price": 0.5, "mileage": 0.3},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "LightGBM v1"
        assert data["model_type"] == "LightGBM"
        assert data["status"] == "trained"
        assert data["metrics_json"]["rmse"] == 0.15

    def test_create_model_version_minimal(self, client, project_id):
        """Test creating a model version with minimal fields."""
        response = client.post(
            f"/projects/{project_id}/models",
            json={"name": "Minimal Model"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal Model"
        assert data["status"] == "trained"

    def test_list_model_versions(self, client, project_id):
        """Test listing model versions for a project."""
        client.post(
            f"/projects/{project_id}/models",
            json={"name": "Model 1"},
        )
        client.post(
            f"/projects/{project_id}/models",
            json={"name": "Model 2"},
        )

        response = client.get(f"/projects/{project_id}/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_model_version(self, client, project_id):
        """Test getting a model version by ID."""
        create_response = client.post(
            f"/projects/{project_id}/models",
            json={"name": "Test Model"},
        )
        model_id = create_response.json()["id"]

        response = client.get(f"/models/{model_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == model_id
        assert data["name"] == "Test Model"

    def test_promote_model_to_candidate(self, client, project_id):
        """Test promoting a model to candidate status."""
        create_response = client.post(
            f"/projects/{project_id}/models",
            json={"name": "Candidate Model"},
        )
        model_id = create_response.json()["id"]

        response = client.post(
            f"/models/{model_id}/promote",
            json={"status": "candidate"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "candidate"

    def test_promote_model_to_production(self, client, project_id):
        """Test promoting a model to production status."""
        create_response = client.post(
            f"/projects/{project_id}/models",
            json={"name": "Production Model"},
        )
        model_id = create_response.json()["id"]

        response = client.post(
            f"/models/{model_id}/promote",
            json={"status": "production"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "production"

    def test_promote_new_model_retires_old_production(self, client, project_id):
        """Test that promoting a new model to production retires the old one."""
        # Create first model and promote to production
        create_response1 = client.post(
            f"/projects/{project_id}/models",
            json={"name": "Model 1"},
        )
        model1_id = create_response1.json()["id"]
        client.post(f"/models/{model1_id}/promote", json={"status": "production"})

        # Create second model and promote to production
        create_response2 = client.post(
            f"/projects/{project_id}/models",
            json={"name": "Model 2"},
        )
        model2_id = create_response2.json()["id"]
        client.post(f"/models/{model2_id}/promote", json={"status": "production"})

        # Check that model 1 is now retired
        response = client.get(f"/models/{model1_id}")
        assert response.json()["status"] == "retired"

        # Check that model 2 is production
        response = client.get(f"/models/{model2_id}")
        assert response.json()["status"] == "production"

    def test_filter_models_by_status(self, client, project_id):
        """Test filtering models by status."""
        # Create models with different statuses
        client.post(f"/projects/{project_id}/models", json={"name": "Model 1"})
        create_response = client.post(
            f"/projects/{project_id}/models",
            json={"name": "Model 2"},
        )
        model2_id = create_response.json()["id"]
        client.post(f"/models/{model2_id}/promote", json={"status": "candidate"})

        # Filter by trained status
        response = client.get(f"/projects/{project_id}/models?status_filter=trained")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Model 1"

        # Filter by candidate status
        response = client.get(f"/projects/{project_id}/models?status_filter=candidate")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Model 2"

    def test_delete_model_version(self, client, project_id):
        """Test deleting a model version."""
        create_response = client.post(
            f"/projects/{project_id}/models",
            json={"name": "To Delete"},
        )
        model_id = create_response.json()["id"]

        response = client.delete(f"/models/{model_id}")
        assert response.status_code == 204

        get_response = client.get(f"/models/{model_id}")
        assert get_response.status_code == 404


class TestModelPrediction:
    """Test model prediction endpoint."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Test Project"})
        return response.json()["id"]

    @pytest.fixture
    def model_with_serving_config(self, client, project_id):
        """Create a model with serving config and return its ID."""
        response = client.post(
            f"/projects/{project_id}/models",
            json={
                "name": "Predictable Model",
                "model_type": "LightGBM",
                "artifact_location": "/artifacts/test_model",
                "serving_config_json": {
                    "features": [
                        {"name": "feature_a", "type": "numeric"},
                        {"name": "feature_b", "type": "numeric"},
                        {"name": "feature_c", "type": "categorical"},
                    ],
                    "target_column": "target",
                    "task_type": "binary",
                },
                "metrics_json": {"roc_auc": 0.85},
            },
        )
        return response.json()["id"]

    def test_predict_missing_model(self, client):
        """Test prediction with non-existent model returns 404."""
        response = client.post(
            "/models/00000000-0000-0000-0000-000000000000/predict",
            json={"features": {"feature_a": 1.0}},
        )
        assert response.status_code == 404

    def test_predict_missing_serving_config(self, client, project_id):
        """Test prediction without serving config returns 400."""
        # Create model without serving config
        create_response = client.post(
            f"/projects/{project_id}/models",
            json={"name": "No Config Model"},
        )
        model_id = create_response.json()["id"]

        response = client.post(
            f"/models/{model_id}/predict",
            json={"features": {"feature_a": 1.0}},
        )
        assert response.status_code == 400
        assert "artifact" in response.json()["detail"].lower() or "config" in response.json()["detail"].lower()

    def test_predict_missing_features(self, client, model_with_serving_config):
        """Test prediction with missing features returns 400."""
        response = client.post(
            f"/models/{model_with_serving_config}/predict",
            json={"features": {"feature_a": 1.0}},  # Missing feature_b and feature_c
        )
        assert response.status_code == 400
        assert "missing" in response.json()["detail"].lower()


class TestModelExplain:
    """Test model explanation endpoint."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Test Project"})
        return response.json()["id"]

    @pytest.fixture
    def model_with_metrics(self, client, project_id):
        """Create a model with metrics and return its ID."""
        response = client.post(
            f"/projects/{project_id}/models",
            json={
                "name": "Explainable Model",
                "model_type": "LightGBM",
                "metrics_json": {
                    "roc_auc": 0.85,
                    "accuracy": 0.78,
                    "f1": 0.76,
                },
                "feature_importances_json": {
                    "price": 0.45,
                    "quantity": 0.30,
                    "category": 0.15,
                    "region": 0.10,
                },
                "serving_config_json": {
                    "features": [
                        {"name": "price", "type": "numeric"},
                        {"name": "quantity", "type": "numeric"},
                        {"name": "category", "type": "categorical"},
                        {"name": "region", "type": "categorical"},
                    ],
                    "target_column": "purchased",
                    "task_type": "binary",
                },
            },
        )
        return response.json()["id"]

    def test_explain_missing_model(self, client):
        """Test explanation with non-existent model returns 404."""
        response = client.post(
            "/models/00000000-0000-0000-0000-000000000000/explain",
            json={"question": "Which features are most important?"},
        )
        assert response.status_code == 404

    def test_explain_empty_question(self, client, model_with_metrics):
        """Test explanation with empty question returns 422."""
        response = client.post(
            f"/models/{model_with_metrics}/explain",
            json={"question": ""},
        )
        assert response.status_code == 422

    def test_explain_endpoint_structure(self, client, model_with_metrics):
        """Test explanation endpoint with mocked LLM."""
        # Create a mock LLM client
        mock_client = MagicMock()
        # Make the async chat method return a coroutine
        async def mock_chat(messages, images=None):
            return "The most important features are price and quantity."
        mock_client.chat = mock_chat

        # Mock both get_llm_provider_and_key and get_llm_client
        # Use MagicMock for provider since we don't need the actual enum
        with patch("app.api.models.get_llm_provider_and_key", return_value=(MagicMock(), "test-key")):
            with patch("app.services.llm_client.get_llm_client", return_value=mock_client):
                response = client.post(
                    f"/models/{model_with_metrics}/explain",
                    json={"question": "Which features are most important?"},
                )

                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
                assert data["model_id"] == model_with_metrics
                assert "price" in data["answer"].lower() or "important" in data["answer"].lower()


class TestValidationSamples:
    """Test validation samples API endpoints."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Test Project"})
        return response.json()["id"]

    @pytest.fixture
    def model_with_serving_config(self, client, project_id):
        """Create a model with serving config and return its ID."""
        response = client.post(
            f"/projects/{project_id}/models",
            json={
                "name": "Test Model",
                "model_type": "LightGBM",
                "artifact_location": "/artifacts/test_model",
                "serving_config_json": {
                    "features": [
                        {"name": "feature_a", "type": "numeric"},
                        {"name": "feature_b", "type": "numeric"},
                        {"name": "feature_c", "type": "categorical"},
                    ],
                    "target_column": "target",
                    "task_type": "regression",
                },
                "metrics_json": {"rmse": 0.15},
            },
        )
        return response.json()["id"]

    @pytest.fixture
    def model_with_validation_samples(self, client, project_id, db_session):
        """Create a model with validation samples."""
        from uuid import UUID
        from app.models.validation_sample import ValidationSample

        # Create model
        response = client.post(
            f"/projects/{project_id}/models",
            json={
                "name": "Model with Samples",
                "model_type": "LightGBM",
                "artifact_location": "/artifacts/test_model",
                "serving_config_json": {
                    "features": [
                        {"name": "feature_a", "type": "numeric"},
                        {"name": "feature_b", "type": "numeric"},
                    ],
                    "target_column": "target",
                    "task_type": "regression",
                },
            },
        )
        model_id = response.json()["id"]

        # Add validation samples directly to database
        for i in range(10):
            sample = ValidationSample(
                model_version_id=UUID(model_id),
                row_index=i,
                features_json={"feature_a": float(i), "feature_b": float(i * 2)},
                target_value=str(float(i * 10)),
                predicted_value=str(float(i * 10 + 0.5)),
                error_value=0.5,
                absolute_error=0.5 * (i + 1),  # Varying absolute errors
            )
            db_session.add(sample)
        db_session.commit()

        return model_id

    def test_list_validation_samples_empty(self, client, model_with_serving_config):
        """Test listing validation samples for a model with no samples."""
        response = client.get(f"/models/{model_with_serving_config}/validation-samples")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_with_serving_config
        assert data["total"] == 0
        assert data["samples"] == []

    def test_list_validation_samples(self, client, model_with_validation_samples):
        """Test listing validation samples for a model."""
        response = client.get(f"/models/{model_with_validation_samples}/validation-samples")
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_with_validation_samples
        assert data["total"] == 10
        assert len(data["samples"]) == 10

    def test_list_validation_samples_pagination(self, client, model_with_validation_samples):
        """Test pagination of validation samples."""
        # Get first page
        response = client.get(
            f"/models/{model_with_validation_samples}/validation-samples?limit=3&offset=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 10
        assert data["limit"] == 3
        assert data["offset"] == 0
        assert len(data["samples"]) == 3

        # Get second page
        response = client.get(
            f"/models/{model_with_validation_samples}/validation-samples?limit=3&offset=3"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["samples"]) == 3
        assert data["offset"] == 3

    def test_list_validation_samples_sort_error_desc(self, client, model_with_validation_samples):
        """Test sorting by error descending (highest errors first)."""
        response = client.get(
            f"/models/{model_with_validation_samples}/validation-samples?sort=error_desc"
        )
        assert response.status_code == 200
        data = response.json()
        samples = data["samples"]
        # Highest absolute error should be first
        errors = [s["absolute_error"] for s in samples]
        assert errors == sorted(errors, reverse=True)

    def test_list_validation_samples_sort_error_asc(self, client, model_with_validation_samples):
        """Test sorting by error ascending (lowest errors first)."""
        response = client.get(
            f"/models/{model_with_validation_samples}/validation-samples?sort=error_asc"
        )
        assert response.status_code == 200
        data = response.json()
        samples = data["samples"]
        errors = [s["absolute_error"] for s in samples]
        assert errors == sorted(errors)

    def test_list_validation_samples_sort_row_index(self, client, model_with_validation_samples):
        """Test sorting by row index."""
        response = client.get(
            f"/models/{model_with_validation_samples}/validation-samples?sort=row_index"
        )
        assert response.status_code == 200
        data = response.json()
        samples = data["samples"]
        row_indices = [s["row_index"] for s in samples]
        assert row_indices == sorted(row_indices)

    def test_list_validation_samples_nonexistent_model(self, client):
        """Test listing validation samples for non-existent model returns 404."""
        response = client.get(
            "/models/00000000-0000-0000-0000-000000000000/validation-samples"
        )
        assert response.status_code == 404

    def test_get_validation_sample(self, client, model_with_validation_samples, db_session):
        """Test getting a single validation sample by ID."""
        from app.models.validation_sample import ValidationSample
        from uuid import UUID

        # Get a sample ID from the database
        sample = db_session.query(ValidationSample).filter(
            ValidationSample.model_version_id == UUID(model_with_validation_samples)
        ).first()
        assert sample is not None

        response = client.get(f"/validation-samples/{sample.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample.id)
        assert data["model_version_id"] == model_with_validation_samples
        assert "features" in data
        assert "target_value" in data
        assert "predicted_value" in data

    def test_get_validation_sample_nonexistent(self, client):
        """Test getting a non-existent validation sample returns 404."""
        response = client.get(
            "/validation-samples/00000000-0000-0000-0000-000000000000"
        )
        assert response.status_code == 404


class TestWhatIfPrediction:
    """Test what-if prediction endpoint."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Test Project"})
        return response.json()["id"]

    @pytest.fixture
    def model_with_sample(self, client, project_id, db_session):
        """Create a model with a validation sample for what-if testing."""
        from uuid import UUID
        from app.models.validation_sample import ValidationSample

        # Create model
        response = client.post(
            f"/projects/{project_id}/models",
            json={
                "name": "What-If Model",
                "model_type": "LightGBM",
                "artifact_location": "/artifacts/whatif_model",
                "serving_config_json": {
                    "features": [
                        {"name": "age", "type": "numeric"},
                        {"name": "income", "type": "numeric"},
                        {"name": "category", "type": "categorical"},
                    ],
                    "target_column": "target",
                    "task_type": "regression",
                },
            },
        )
        model_id = response.json()["id"]

        # Add a validation sample
        sample = ValidationSample(
            model_version_id=UUID(model_id),
            row_index=0,
            features_json={"age": 30.0, "income": 50000.0, "category": "A"},
            target_value="100.0",
            predicted_value="98.5",
            error_value=-1.5,
            absolute_error=1.5,
        )
        db_session.add(sample)
        db_session.commit()
        db_session.refresh(sample)

        return {"model_id": model_id, "sample_id": str(sample.id)}

    def test_what_if_missing_model(self, client):
        """Test what-if with non-existent model returns 404."""
        response = client.post(
            "/models/00000000-0000-0000-0000-000000000000/what-if",
            json={
                "sample_id": "00000000-0000-0000-0000-000000000001",
                "modified_features": {"age": 40},
            },
        )
        assert response.status_code == 404

    def test_what_if_missing_sample(self, client, project_id):
        """Test what-if with non-existent sample returns 404."""
        # Create a model without samples
        response = client.post(
            f"/projects/{project_id}/models",
            json={
                "name": "Empty Model",
                "artifact_location": "/artifacts/empty",
                "serving_config_json": {
                    "features": [{"name": "x", "type": "numeric"}],
                    "target_column": "y",
                    "task_type": "regression",
                },
            },
        )
        model_id = response.json()["id"]

        response = client.post(
            f"/models/{model_id}/what-if",
            json={
                "sample_id": "00000000-0000-0000-0000-000000000001",
                "modified_features": {"x": 10},
            },
        )
        assert response.status_code == 404

    def test_what_if_sample_belongs_to_different_model(
        self, client, model_with_sample, project_id
    ):
        """Test what-if with sample from different model returns 400."""
        # Create another model
        response = client.post(
            f"/projects/{project_id}/models",
            json={
                "name": "Other Model",
                "artifact_location": "/artifacts/other",
                "serving_config_json": {
                    "features": [{"name": "x", "type": "numeric"}],
                    "target_column": "y",
                    "task_type": "regression",
                },
            },
        )
        other_model_id = response.json()["id"]

        # Try to use sample from first model with second model
        response = client.post(
            f"/models/{other_model_id}/what-if",
            json={
                "sample_id": model_with_sample["sample_id"],
                "modified_features": {"x": 10},
            },
        )
        assert response.status_code == 400
        assert "does not belong to model" in response.json()["detail"]

    def test_what_if_invalid_feature_name(self, client, model_with_sample):
        """Test what-if with invalid feature name returns 400."""
        response = client.post(
            f"/models/{model_with_sample['model_id']}/what-if",
            json={
                "sample_id": model_with_sample["sample_id"],
                "modified_features": {"invalid_feature": 100},
            },
        )
        assert response.status_code == 400
        assert "Invalid feature names" in response.json()["detail"]

    def test_what_if_no_serving_config(self, client, project_id, db_session):
        """Test what-if with model lacking serving config returns 400."""
        from uuid import UUID
        from app.models.validation_sample import ValidationSample

        # Create model without serving config
        response = client.post(
            f"/projects/{project_id}/models",
            json={
                "name": "No Config Model",
                "artifact_location": "/artifacts/noconfig",
            },
        )
        model_id = response.json()["id"]

        # Add a sample anyway
        sample = ValidationSample(
            model_version_id=UUID(model_id),
            row_index=0,
            features_json={"x": 1.0},
            target_value="5.0",
            predicted_value="5.5",
        )
        db_session.add(sample)
        db_session.commit()
        db_session.refresh(sample)

        response = client.post(
            f"/models/{model_id}/what-if",
            json={
                "sample_id": str(sample.id),
                "modified_features": {"x": 10},
            },
        )
        assert response.status_code == 400
        assert "feature configuration" in response.json()["detail"].lower()

"""Tests for dataset specification API endpoints."""
import pytest


class TestDatasetSpecsCRUD:
    """Test dataset spec CRUD operations."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Test Project"})
        return response.json()["id"]

    @pytest.fixture
    def data_source_id(self, client, project_id):
        """Create a data source and return its ID."""
        response = client.post(
            f"/projects/{project_id}/data-sources",
            json={"name": "Test Source", "type": "file_upload"},
        )
        return response.json()["id"]

    def test_create_dataset_spec(self, client, project_id, data_source_id):
        """Test creating a new dataset specification."""
        response = client.post(
            f"/projects/{project_id}/dataset-specs",
            json={
                "name": "Car Price Dataset",
                "description": "Dataset for predicting car prices",
                "target_column": "price",
                "feature_columns": ["make", "model", "year", "mileage"],
                "data_sources_json": {"sources": [data_source_id]},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Car Price Dataset"
        assert data["target_column"] == "price"
        assert data["feature_columns"] == ["make", "model", "year", "mileage"]
        assert data["project_id"] == project_id

    def test_create_dataset_spec_minimal(self, client, project_id):
        """Test creating a dataset spec with minimal fields."""
        response = client.post(
            f"/projects/{project_id}/dataset-specs",
            json={"name": "Minimal Spec"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal Spec"
        assert data["target_column"] is None
        assert data["feature_columns"] is None  # Null when not provided

    def test_list_dataset_specs(self, client, project_id):
        """Test listing dataset specs for a project."""
        # Create two dataset specs
        client.post(
            f"/projects/{project_id}/dataset-specs",
            json={"name": "Spec 1"},
        )
        client.post(
            f"/projects/{project_id}/dataset-specs",
            json={"name": "Spec 2"},
        )

        response = client.get(f"/projects/{project_id}/dataset-specs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_dataset_spec(self, client, project_id):
        """Test getting a dataset spec by ID."""
        create_response = client.post(
            f"/projects/{project_id}/dataset-specs",
            json={"name": "Test Spec", "target_column": "target"},
        )
        spec_id = create_response.json()["id"]

        response = client.get(f"/dataset-specs/{spec_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == spec_id
        assert data["name"] == "Test Spec"
        assert data["target_column"] == "target"

    def test_update_dataset_spec(self, client, project_id):
        """Test updating a dataset specification."""
        create_response = client.post(
            f"/projects/{project_id}/dataset-specs",
            json={"name": "Original", "target_column": "old_target"},
        )
        spec_id = create_response.json()["id"]

        response = client.put(
            f"/dataset-specs/{spec_id}",
            json={
                "name": "Updated Spec",
                "target_column": "new_target",
                "feature_columns": ["col1", "col2"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Spec"
        assert data["target_column"] == "new_target"
        assert data["feature_columns"] == ["col1", "col2"]

    def test_delete_dataset_spec(self, client, project_id):
        """Test deleting a dataset specification."""
        create_response = client.post(
            f"/projects/{project_id}/dataset-specs",
            json={"name": "To Delete"},
        )
        spec_id = create_response.json()["id"]

        response = client.delete(f"/dataset-specs/{spec_id}")
        assert response.status_code == 204

        get_response = client.get(f"/dataset-specs/{spec_id}")
        assert get_response.status_code == 404

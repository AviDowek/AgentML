"""Tests for data source API endpoints."""
import tempfile
from pathlib import Path
import pandas as pd
import pytest


class TestDataSourcesCRUD:
    """Test data source CRUD operations."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Test Project"})
        return response.json()["id"]

    def test_create_data_source(self, client, project_id):
        """Test creating a new data source."""
        response = client.post(
            f"/projects/{project_id}/data-sources",
            json={
                "name": "Sales Data",
                "type": "file_upload",
                "config_json": {"file_path": "/data/sales.csv"},
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Sales Data"
        assert data["type"] == "file_upload"
        assert data["project_id"] == project_id
        assert data["config_json"] == {"file_path": "/data/sales.csv"}

    def test_create_data_source_invalid_project(self, client):
        """Test creating a data source for non-existent project."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.post(
            f"/projects/{fake_id}/data-sources",
            json={"name": "Test", "type": "file_upload"},
        )
        assert response.status_code == 404

    def test_list_data_sources(self, client, project_id):
        """Test listing data sources for a project."""
        # Create two data sources
        client.post(
            f"/projects/{project_id}/data-sources",
            json={"name": "Source 1", "type": "file_upload"},
        )
        client.post(
            f"/projects/{project_id}/data-sources",
            json={"name": "Source 2", "type": "database"},
        )

        response = client.get(f"/projects/{project_id}/data-sources")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_data_source(self, client, project_id):
        """Test getting a data source by ID."""
        # Create a data source
        create_response = client.post(
            f"/projects/{project_id}/data-sources",
            json={"name": "Test Source", "type": "file_upload"},
        )
        source_id = create_response.json()["id"]

        # Get the data source
        response = client.get(f"/data-sources/{source_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == source_id
        assert data["name"] == "Test Source"

    def test_update_data_source(self, client, project_id):
        """Test updating a data source."""
        # Create a data source
        create_response = client.post(
            f"/projects/{project_id}/data-sources",
            json={"name": "Original", "type": "file_upload"},
        )
        source_id = create_response.json()["id"]

        # Update it
        response = client.put(
            f"/data-sources/{source_id}",
            json={"name": "Updated Name", "config_json": {"path": "/new/path"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["config_json"] == {"path": "/new/path"}

    def test_delete_data_source(self, client, project_id):
        """Test deleting a data source."""
        # Create a data source
        create_response = client.post(
            f"/projects/{project_id}/data-sources",
            json={"name": "To Delete", "type": "file_upload"},
        )
        source_id = create_response.json()["id"]

        # Delete it
        response = client.delete(f"/data-sources/{source_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_response = client.get(f"/data-sources/{source_id}")
        assert get_response.status_code == 404


class TestDataSourceProfiling:
    """Test data source profiling endpoints."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Profiling Test Project"})
        return response.json()["id"]

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        df = pd.DataFrame({
            "id": list(range(100)),
            "name": [f"Item_{i}" for i in range(100)],
            "value": [i * 1.5 for i in range(100)],
            "category": ["A", "B", "C", "D"] * 25,
            "is_active": [True, False] * 50,
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            df.to_csv(f, index=False)
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def uploaded_data_source(self, client, project_id, sample_csv_file):
        """Upload a file and return the data source."""
        with open(sample_csv_file, "rb") as f:
            response = client.post(
                f"/projects/{project_id}/data-sources/upload",
                files={"file": ("test_data.csv", f, "text/csv")},
                data={"name": "Test Data"},
            )
        assert response.status_code == 201
        return response.json()

    def test_profile_single_data_source(self, client, uploaded_data_source):
        """Test profiling a single data source."""
        source_id = uploaded_data_source["id"]

        response = client.post(f"/data-sources/{source_id}/profile")
        assert response.status_code == 200

        data = response.json()
        assert data["source_id"] == source_id
        assert data["source_name"] == "Test Data"
        assert data["source_type"] == "file_upload"
        assert data["estimated_row_count"] == 100
        assert data["column_count"] == 5
        assert len(data["columns"]) == 5
        assert "profiled_at" in data
        assert isinstance(data["warnings"], list)

        # Check column structure
        col_names = [col["name"] for col in data["columns"]]
        assert "id" in col_names
        assert "name" in col_names
        assert "value" in col_names
        assert "category" in col_names
        assert "is_active" in col_names

        # Check column details
        value_col = next(col for col in data["columns"] if col["name"] == "value")
        assert value_col["inferred_type"] == "numeric"
        assert "statistics" in value_col
        assert "min" in value_col["statistics"]
        assert "max" in value_col["statistics"]

        category_col = next(col for col in data["columns"] if col["name"] == "category")
        assert category_col["inferred_type"] == "categorical"
        assert "top_values" in category_col["statistics"]

    def test_profile_single_data_source_custom_sample(self, client, uploaded_data_source):
        """Test profiling with custom sample size."""
        source_id = uploaded_data_source["id"]

        # API requires minimum 1000 sample_rows but file only has 100 rows
        response = client.post(f"/data-sources/{source_id}/profile?sample_rows=1000")
        assert response.status_code == 200

        data = response.json()
        assert data["sample_size"] <= 100  # Can't be more than total rows in file

    def test_profile_single_data_source_not_found(self, client):
        """Test profiling non-existent data source."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.post(f"/data-sources/{fake_id}/profile")
        assert response.status_code == 404

    def test_profile_saves_to_database(self, client, uploaded_data_source):
        """Test that profile is saved to data source."""
        source_id = uploaded_data_source["id"]

        # Profile the data source
        profile_response = client.post(f"/data-sources/{source_id}/profile")
        assert profile_response.status_code == 200

        # Get the data source and check profile_json
        get_response = client.get(f"/data-sources/{source_id}")
        assert get_response.status_code == 200

        data = get_response.json()
        assert data["profile_json"] is not None
        assert data["profile_json"]["source_id"] == source_id
        assert data["profile_json"]["column_count"] == 5

    def test_profile_all_data_sources(self, client, project_id, sample_csv_file):
        """Test profiling all data sources in a project."""
        # Upload multiple files
        for i in range(3):
            df = pd.DataFrame({
                "col1": list(range(50)),
                "col2": [f"val_{j}" for j in range(50)],
            })
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
                df.to_csv(f, index=False)
                temp_path = Path(f.name)

            with open(temp_path, "rb") as f:
                response = client.post(
                    f"/projects/{project_id}/data-sources/upload",
                    files={"file": (f"data_{i}.csv", f, "text/csv")},
                )
                assert response.status_code == 201

            temp_path.unlink()

        # Profile all data sources
        response = client.post(f"/projects/{project_id}/data-sources/profile-all")
        assert response.status_code == 200

        data = response.json()
        assert data["project_id"] == project_id
        assert data["total_sources"] == 3
        assert data["profiled_count"] == 3
        assert data["error_count"] == 0
        assert len(data["profiles"]) == 3

        # Each profile should have the expected structure
        for profile in data["profiles"]:
            assert "source_id" in profile
            assert "source_name" in profile
            assert "columns" in profile
            assert "profiled_at" in profile

    def test_profile_all_invalid_project(self, client):
        """Test profiling data sources for non-existent project."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.post(f"/projects/{fake_id}/data-sources/profile-all")
        assert response.status_code == 404

    def test_profile_all_empty_project(self, client, project_id):
        """Test profiling when project has no data sources."""
        response = client.post(f"/projects/{project_id}/data-sources/profile-all")
        assert response.status_code == 200

        data = response.json()
        assert data["project_id"] == project_id
        assert data["total_sources"] == 0
        assert data["profiled_count"] == 0
        assert len(data["profiles"]) == 0

    def test_profile_all_custom_sample_rows(self, client, project_id, sample_csv_file):
        """Test profile-all with custom sample size."""
        # Upload a file
        with open(sample_csv_file, "rb") as f:
            response = client.post(
                f"/projects/{project_id}/data-sources/upload",
                files={"file": ("test.csv", f, "text/csv")},
            )
        assert response.status_code == 201

        # Profile with custom sample size
        response = client.post(
            f"/projects/{project_id}/data-sources/profile-all?sample_rows=25000"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["profiled_count"] == 1

    def test_profile_all_saves_to_database(self, client, project_id, sample_csv_file):
        """Test that profile-all saves profiles to each data source."""
        # Upload files
        source_ids = []
        for i in range(2):
            with open(sample_csv_file, "rb") as f:
                response = client.post(
                    f"/projects/{project_id}/data-sources/upload",
                    files={"file": (f"data_{i}.csv", f, "text/csv")},
                )
            assert response.status_code == 201
            source_ids.append(response.json()["id"])

        # Profile all
        response = client.post(f"/projects/{project_id}/data-sources/profile-all")
        assert response.status_code == 200

        # Check each data source has profile_json saved
        for source_id in source_ids:
            get_response = client.get(f"/data-sources/{source_id}")
            assert get_response.status_code == 200
            data = get_response.json()
            assert data["profile_json"] is not None
            assert data["profile_json"]["source_id"] == source_id


class TestDataSourceProfilingQuality:
    """Test data quality detection in profiling."""

    @pytest.fixture
    def project_id(self, client):
        """Create a project and return its ID."""
        response = client.post("/projects", json={"name": "Quality Test Project"})
        return response.json()["id"]

    def test_profile_detects_high_nulls(self, client, project_id):
        """Test that profiling detects high null ratios."""
        df = pd.DataFrame({
            "good_col": list(range(100)),
            "bad_col": [None] * 80 + list(range(20)),
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            df.to_csv(f, index=False)
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                upload_response = client.post(
                    f"/projects/{project_id}/data-sources/upload",
                    files={"file": ("nulls.csv", f, "text/csv")},
                )
            assert upload_response.status_code == 201
            source_id = upload_response.json()["id"]

            profile_response = client.post(f"/data-sources/{source_id}/profile")
            assert profile_response.status_code == 200

            data = profile_response.json()
            # Should have warning about missing values
            assert any("missing values" in w.lower() for w in data["warnings"])

            # bad_col should have high null ratio
            bad_col = next(col for col in data["columns"] if col["name"] == "bad_col")
            assert bad_col["null_ratio"] == 0.8
        finally:
            temp_path.unlink()

    def test_profile_detects_constant_column(self, client, project_id):
        """Test that profiling detects constant columns."""
        df = pd.DataFrame({
            "constant": [1] * 100,
            "variable": list(range(100)),
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            df.to_csv(f, index=False)
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                upload_response = client.post(
                    f"/projects/{project_id}/data-sources/upload",
                    files={"file": ("constant.csv", f, "text/csv")},
                )
            assert upload_response.status_code == 201
            source_id = upload_response.json()["id"]

            profile_response = client.post(f"/data-sources/{source_id}/profile")
            assert profile_response.status_code == 200

            data = profile_response.json()
            # Should have warning about constant columns
            assert any("constant" in w.lower() for w in data["warnings"])

            # constant column should have distinct_count of 1
            const_col = next(col for col in data["columns"] if col["name"] == "constant")
            assert const_col["distinct_count"] == 1
        finally:
            temp_path.unlink()

    def test_profile_detects_id_column(self, client, project_id):
        """Test that profiling detects potential ID columns."""
        df = pd.DataFrame({
            "user_id": list(range(200)),
            "value": [1, 2, 3, 4] * 50,
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            df.to_csv(f, index=False)
            temp_path = Path(f.name)

        try:
            with open(temp_path, "rb") as f:
                upload_response = client.post(
                    f"/projects/{project_id}/data-sources/upload",
                    files={"file": ("ids.csv", f, "text/csv")},
                )
            assert upload_response.status_code == 201
            source_id = upload_response.json()["id"]

            profile_response = client.post(f"/data-sources/{source_id}/profile")
            assert profile_response.status_code == 200

            data = profile_response.json()
            # Should have warning about ID columns
            assert any("id columns" in w.lower() for w in data["warnings"])

            # user_id should be detected as ID type
            id_col = next(col for col in data["columns"] if col["name"] == "user_id")
            assert id_col["inferred_type"] == "id"
        finally:
            temp_path.unlink()

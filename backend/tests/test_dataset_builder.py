"""Tests for dataset builder service."""
import tempfile
import pandas as pd
import pytest
from pathlib import Path
from uuid import UUID

from app.models.project import Project
from app.models.data_source import DataSource
from app.models.dataset_spec import DatasetSpec
from app.services.dataset_builder import DatasetBuilder


class TestDatasetBuilder:
    """Test DatasetBuilder service."""

    @pytest.fixture
    def project(self, db_session):
        """Create a test project."""
        project = Project(name="Test Project")
        db_session.add(project)
        db_session.commit()
        db_session.refresh(project)
        return project

    @pytest.fixture
    def sample_csv_path(self):
        """Create a temporary CSV file with sample data."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "salary": [50000.0, 60000.0, 75000.0, 55000.0, 65000.0],
            "department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            return Path(f.name)

    @pytest.fixture
    def data_source(self, db_session, project, sample_csv_path):
        """Create a test data source."""
        source = DataSource(
            project_id=project.id,
            name="Test CSV",
            type="file_upload",
            config_json={
                "file_path": str(sample_csv_path),
                "delimiter": ",",
            },
        )
        db_session.add(source)
        db_session.commit()
        db_session.refresh(source)
        return source

    @pytest.fixture
    def dataset_spec(self, db_session, project, data_source):
        """Create a test dataset spec."""
        spec = DatasetSpec(
            project_id=project.id,
            name="Test Spec",
            target_column="salary",
            feature_columns=["age", "department"],
            data_sources_json={"sources": [str(data_source.id)]},
        )
        db_session.add(spec)
        db_session.commit()
        db_session.refresh(spec)
        return spec

    def test_build_dataset_from_spec(self, db_session, dataset_spec):
        """Test building a DataFrame from a DatasetSpec."""
        builder = DatasetBuilder(db_session)
        df = builder.build_dataset_from_spec(dataset_spec.id)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        # Should have feature columns + target column
        assert "age" in df.columns
        assert "department" in df.columns
        assert "salary" in df.columns
        # Should NOT have excluded columns
        assert "id" not in df.columns
        assert "name" not in df.columns

    def test_build_dataset_all_columns(self, db_session, project, data_source):
        """Test building dataset when no columns are specified (all columns)."""
        spec = DatasetSpec(
            project_id=project.id,
            name="All Columns Spec",
            data_sources_json={"sources": [str(data_source.id)]},
        )
        db_session.add(spec)
        db_session.commit()

        builder = DatasetBuilder(db_session)
        df = builder.build_dataset_from_spec(spec.id)

        # All columns should be present
        assert len(df.columns) == 5
        assert "id" in df.columns
        assert "name" in df.columns
        assert "age" in df.columns
        assert "salary" in df.columns
        assert "department" in df.columns

    def test_build_dataset_with_filter_range(self, db_session, project, data_source):
        """Test building dataset with range filters."""
        spec = DatasetSpec(
            project_id=project.id,
            name="Filtered Spec",
            data_sources_json={"sources": [str(data_source.id)]},
            filters_json={"age": {"min": 28, "max": 33}},
        )
        db_session.add(spec)
        db_session.commit()

        builder = DatasetBuilder(db_session)
        df = builder.build_dataset_from_spec(spec.id)

        # Should only include rows where 28 <= age <= 33
        assert len(df) == 3  # Bob(30), Diana(28), Eve(32)
        assert all(df["age"] >= 28)
        assert all(df["age"] <= 33)

    def test_build_dataset_with_filter_in_list(self, db_session, project, data_source):
        """Test building dataset with 'in' filter."""
        spec = DatasetSpec(
            project_id=project.id,
            name="In Filter Spec",
            data_sources_json={"sources": [str(data_source.id)]},
            filters_json={"department": {"in": ["Engineering", "Sales"]}},
        )
        db_session.add(spec)
        db_session.commit()

        builder = DatasetBuilder(db_session)
        df = builder.build_dataset_from_spec(spec.id)

        assert len(df) == 3  # Alice, Charlie, Diana
        assert all(df["department"].isin(["Engineering", "Sales"]))

    def test_build_dataset_with_exact_value_filter(self, db_session, project, data_source):
        """Test building dataset with exact value filter."""
        spec = DatasetSpec(
            project_id=project.id,
            name="Exact Filter Spec",
            data_sources_json={"sources": [str(data_source.id)]},
            filters_json={"department": "Engineering"},
        )
        db_session.add(spec)
        db_session.commit()

        builder = DatasetBuilder(db_session)
        df = builder.build_dataset_from_spec(spec.id)

        assert len(df) == 2  # Alice, Charlie
        assert all(df["department"] == "Engineering")

    def test_build_dataset_spec_not_found(self, db_session):
        """Test that ValueError is raised for non-existent spec."""
        builder = DatasetBuilder(db_session)
        fake_id = UUID("00000000-0000-0000-0000-000000000000")

        with pytest.raises(ValueError, match="not found"):
            builder.build_dataset_from_spec(fake_id)

    def test_build_dataset_no_sources(self, db_session, project):
        """Test that ValueError is raised when no data sources configured."""
        spec = DatasetSpec(
            project_id=project.id,
            name="No Sources Spec",
            data_sources_json={},
        )
        db_session.add(spec)
        db_session.commit()

        builder = DatasetBuilder(db_session)

        with pytest.raises(ValueError, match="no data sources"):
            builder.build_dataset_from_spec(spec.id)

    def test_build_dataset_source_not_found(self, db_session, project):
        """Test that ValueError is raised when data source doesn't exist."""
        fake_source_id = "00000000-0000-0000-0000-000000000000"
        spec = DatasetSpec(
            project_id=project.id,
            name="Bad Source Spec",
            data_sources_json={"sources": [fake_source_id]},
        )
        db_session.add(spec)
        db_session.commit()

        builder = DatasetBuilder(db_session)

        with pytest.raises(ValueError, match="DataSource.*not found"):
            builder.build_dataset_from_spec(spec.id)

    def test_get_dataset_info(self, db_session, dataset_spec, data_source):
        """Test getting dataset info without loading data."""
        builder = DatasetBuilder(db_session)
        info = builder.get_dataset_info(dataset_spec.id)

        assert info["dataset_spec_id"] == str(dataset_spec.id)
        assert info["name"] == "Test Spec"
        assert info["target_column"] == "salary"
        assert info["feature_columns"] == ["age", "department"]
        assert len(info["data_sources"]) == 1
        assert info["data_sources"][0]["id"] == str(data_source.id)
        assert info["data_sources"][0]["name"] == "Test CSV"

    def test_build_dataset_multiple_sources(self, db_session, project, sample_csv_path):
        """Test building dataset from multiple data sources."""
        # Create second CSV with different data
        df2 = pd.DataFrame({
            "id": [6, 7],
            "name": ["Frank", "Grace"],
            "age": [40, 45],
            "salary": [80000.0, 95000.0],
            "department": ["Engineering", "Marketing"],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df2.to_csv(f, index=False)
            csv_path2 = Path(f.name)

        # Create two data sources
        source1 = DataSource(
            project_id=project.id,
            name="Source 1",
            type="file_upload",
            config_json={"file_path": str(sample_csv_path), "delimiter": ","},
        )
        source2 = DataSource(
            project_id=project.id,
            name="Source 2",
            type="file_upload",
            config_json={"file_path": str(csv_path2), "delimiter": ","},
        )
        db_session.add_all([source1, source2])
        db_session.commit()

        # Create spec with both sources
        spec = DatasetSpec(
            project_id=project.id,
            name="Multi Source Spec",
            data_sources_json={"sources": [str(source1.id), str(source2.id)]},
        )
        db_session.add(spec)
        db_session.commit()

        builder = DatasetBuilder(db_session)
        df = builder.build_dataset_from_spec(spec.id)

        # Should have combined data from both sources
        assert len(df) == 7  # 5 + 2

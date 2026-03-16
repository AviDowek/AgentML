"""Tests for data profiler service."""
import tempfile
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.services.data_profiler import (
    DataProfiler,
    profile_data_source,
    profile_all_data_sources,
    DEFAULT_SAMPLE_ROWS,
    MAX_SAMPLE_ROWS,
)
from app.models.data_source import DataSource, DataSourceType


class TestDataProfiler:
    """Test DataProfiler class."""

    @pytest.fixture
    def profiler(self):
        """Create a DataProfiler instance."""
        return DataProfiler(sample_rows=1000)

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "salary": [50000.0, 60000.0, 75000.0, 55000.0, 65000.0],
            "department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"],
            "is_active": [True, True, False, True, True],
        })

    @pytest.fixture
    def temp_csv_file(self, sample_df):
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            sample_df.to_csv(f, index=False)
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    def test_init_default_sample_rows(self):
        """Test default sample rows initialization."""
        profiler = DataProfiler()
        assert profiler.sample_rows == DEFAULT_SAMPLE_ROWS

    def test_init_custom_sample_rows(self):
        """Test custom sample rows initialization."""
        profiler = DataProfiler(sample_rows=10000)
        assert profiler.sample_rows == 10000

    def test_init_max_sample_rows_limit(self):
        """Test that sample_rows is capped at MAX_SAMPLE_ROWS."""
        profiler = DataProfiler(sample_rows=500000)
        assert profiler.sample_rows == MAX_SAMPLE_ROWS


class TestDataProfilerColumnProfiling:
    """Test column profiling functionality."""

    @pytest.fixture
    def profiler(self):
        """Create a DataProfiler instance."""
        return DataProfiler(sample_rows=1000)

    def test_profile_numeric_column(self, profiler):
        """Test profiling a numeric column."""
        # Use values with duplicates so it's not detected as ID
        series = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        profile = profiler._profile_column(series, "numeric_col", len(series))

        assert profile["name"] == "numeric_col"
        assert profile["inferred_type"] == "numeric"
        assert profile["null_count"] == 0
        assert profile["null_ratio"] == 0.0
        assert profile["distinct_count"] == 5
        assert "min" in profile["statistics"]
        assert "max" in profile["statistics"]
        assert "mean" in profile["statistics"]
        assert profile["statistics"]["min"] == 1.5
        assert profile["statistics"]["max"] == 5.5

    def test_profile_categorical_column(self, profiler):
        """Test profiling a categorical column."""
        series = pd.Series(["A", "B", "A", "C", "B", "A", "A", "B", "C", "A"])
        profile = profiler._profile_column(series, "cat_col", len(series))

        assert profile["name"] == "cat_col"
        assert profile["inferred_type"] == "categorical"
        assert profile["distinct_count"] == 3
        assert "top_values" in profile["statistics"]
        assert "mode" in profile["statistics"]
        assert profile["statistics"]["mode"] == "A"

    def test_profile_boolean_column(self, profiler):
        """Test profiling a boolean column."""
        series = pd.Series([True, False, True, True, False])
        profile = profiler._profile_column(series, "bool_col", len(series))

        assert profile["name"] == "bool_col"
        assert profile["inferred_type"] == "boolean"
        assert profile["distinct_count"] == 2
        assert "true_count" in profile["statistics"]
        assert "false_count" in profile["statistics"]
        assert profile["statistics"]["true_count"] == 3
        assert profile["statistics"]["false_count"] == 2

    def test_profile_column_with_nulls(self, profiler):
        """Test profiling a column with null values."""
        series = pd.Series([1, 2, None, 4, None, 6, 7, None, 9, 10])
        profile = profiler._profile_column(series, "nulls_col", len(series))

        assert profile["null_count"] == 3
        assert profile["null_ratio"] == 0.3
        assert profile["distinct_count"] == 7  # Non-null unique values

    def test_profile_datetime_column(self, profiler):
        """Test profiling a datetime column."""
        series = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        profile = profiler._profile_column(series, "date_col", len(series))

        assert profile["name"] == "date_col"
        assert profile["inferred_type"] == "datetime"
        assert "min" in profile["statistics"]
        assert "max" in profile["statistics"]
        assert "range_days" in profile["statistics"]

    def test_profile_text_column(self, profiler):
        """Test profiling a text column (long strings with high cardinality)."""
        # Create strings that are long enough to be classified as text (>50 chars average)
        series = pd.Series([
            f"This is a very long description that contains a lot of text about item number {i}. " * 2
            for i in range(100)
        ])
        profile = profiler._profile_column(series, "text_col", len(series))

        assert profile["name"] == "text_col"
        assert profile["inferred_type"] == "text"
        assert "avg_length" in profile["statistics"]
        assert "min_length" in profile["statistics"]
        assert "max_length" in profile["statistics"]

    def test_profile_id_column(self, profiler):
        """Test profiling an ID column (unique integers)."""
        series = pd.Series(list(range(1, 1001)))
        profile = profiler._profile_column(series, "id_col", len(series))

        assert profile["name"] == "id_col"
        assert profile["inferred_type"] == "id"
        assert profile["distinct_ratio"] == 1.0


class TestDataProfilerInferType:
    """Test type inference functionality."""

    @pytest.fixture
    def profiler(self):
        """Create a DataProfiler instance."""
        return DataProfiler(sample_rows=1000)

    def test_infer_numeric(self, profiler):
        """Test inferring numeric type."""
        series = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        assert profiler._infer_semantic_type(series) == "numeric"

    def test_infer_boolean_from_bool(self, profiler):
        """Test inferring boolean from bool dtype."""
        series = pd.Series([True, False, True])
        assert profiler._infer_semantic_type(series) == "boolean"

    def test_infer_boolean_from_strings(self, profiler):
        """Test inferring boolean from string values."""
        series = pd.Series(["true", "false", "true", "false"])
        assert profiler._infer_semantic_type(series) == "boolean"

    def test_infer_categorical(self, profiler):
        """Test inferring categorical type."""
        series = pd.Series(["A", "B", "C"] * 100)
        assert profiler._infer_semantic_type(series) == "categorical"

    def test_infer_datetime_from_strings(self, profiler):
        """Test inferring datetime from string values."""
        series = pd.Series(["2024-01-01", "2024-02-01", "2024-03-01"])
        assert profiler._infer_semantic_type(series) == "datetime"


class TestDataProfilerQualityIssues:
    """Test quality issue detection."""

    @pytest.fixture
    def profiler(self):
        """Create a DataProfiler instance."""
        return DataProfiler(sample_rows=1000)

    def test_detect_high_null_columns(self, profiler):
        """Test detecting columns with high null ratios."""
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "bad": [1, None, None, None, None],
        })
        columns_profile = profiler._profile_dataframe_columns(df)
        warnings = profiler._detect_quality_issues(df, columns_profile)

        assert any("missing values" in w for w in warnings)
        assert any("bad" in w for w in warnings)

    def test_detect_id_columns(self, profiler):
        """Test detecting potential ID columns."""
        df = pd.DataFrame({
            "user_id": list(range(100)),
            "value": [1] * 100,
        })
        columns_profile = profiler._profile_dataframe_columns(df)
        warnings = profiler._detect_quality_issues(df, columns_profile)

        assert any("ID columns" in w for w in warnings)

    def test_detect_constant_columns(self, profiler):
        """Test detecting constant columns."""
        df = pd.DataFrame({
            "constant": [1, 1, 1, 1, 1],
            "variable": [1, 2, 3, 4, 5],
        })
        columns_profile = profiler._profile_dataframe_columns(df)
        warnings = profiler._detect_quality_issues(df, columns_profile)

        assert any("Constant columns" in w for w in warnings)
        assert any("constant" in w for w in warnings)

    def test_detect_small_dataset(self, profiler):
        """Test detecting small datasets."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        })
        columns_profile = profiler._profile_dataframe_columns(df)
        warnings = profiler._detect_quality_issues(df, columns_profile)

        assert any("small dataset" in w.lower() for w in warnings)

    def test_detect_unnamed_columns(self, profiler):
        """Test detecting unnamed columns."""
        df = pd.DataFrame({
            "Unnamed: 0": [1, 2, 3, 4, 5],
            "Unnamed: 1": [6, 7, 8, 9, 10],
            "Unnamed: 2": [11, 12, 13, 14, 15],
            "real_col": [1, 2, 3, 4, 5],
        })
        columns_profile = profiler._profile_dataframe_columns(df)
        warnings = profiler._detect_quality_issues(df, columns_profile)

        assert any("unnamed columns" in w.lower() for w in warnings)


class TestDataProfilerFileSampling:
    """Test file loading and sampling."""

    @pytest.fixture
    def profiler(self):
        """Create a DataProfiler instance."""
        return DataProfiler(sample_rows=100)

    @pytest.fixture
    def large_csv_file(self):
        """Create a larger CSV file for testing sampling."""
        df = pd.DataFrame({
            "id": list(range(500)),
            "value": [i * 1.5 for i in range(500)],
            "category": ["A", "B", "C", "D", "E"] * 100,
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            df.to_csv(f, index=False)
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    def test_csv_sampling(self, profiler, large_csv_file):
        """Test that CSV files are sampled correctly."""
        df, total_rows = profiler._load_file_with_sampling(
            str(large_csv_file), "csv", {}
        )

        assert len(df) == 100  # sample_rows
        assert total_rows == 500

    def test_csv_with_delimiter(self, profiler):
        """Test loading CSV with custom delimiter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name;value;count\n")
            for i in range(200):
                f.write(f"item{i};{i*10};{i}\n")
            temp_path = Path(f.name)

        try:
            df, total_rows = profiler._load_file_with_sampling(
                str(temp_path), "csv", {"delimiter": ";"}
            )
            assert len(df) == 100
            assert total_rows == 200
            assert "name" in df.columns
        finally:
            temp_path.unlink()


class TestDataProfilerFileSource:
    """Test profiling file-based data sources."""

    @pytest.fixture
    def profiler(self):
        """Create a DataProfiler instance."""
        return DataProfiler(sample_rows=1000)

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file."""
        df = pd.DataFrame({
            "id": list(range(100)),
            "name": [f"Name_{i}" for i in range(100)],
            "value": [i * 1.5 for i in range(100)],
            "category": ["A", "B", "C", "D"] * 25,
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            df.to_csv(f, index=False)
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    def test_profile_file_source_csv(self, profiler, mock_db, sample_csv_file):
        """Test profiling a CSV file source."""
        data_source = DataSource(
            id=uuid4(),
            project_id=uuid4(),
            name="test_file.csv",
            type=DataSourceType.FILE_UPLOAD,
            config_json={
                "file_path": str(sample_csv_file),
                "original_filename": "test_file.csv",
                "file_type": "csv",
            },
        )
        mock_db.query.return_value.filter.return_value.first.return_value = data_source

        profile = profiler.profile_data_source(mock_db, data_source.id)

        assert profile["source_id"] == str(data_source.id)
        assert profile["source_name"] == "test_file.csv"
        assert profile["source_type"] == "file_upload"
        assert profile["file_type"] == "csv"
        assert profile["estimated_row_count"] == 100
        assert profile["column_count"] == 4
        assert len(profile["columns"]) == 4
        assert "profiled_at" in profile
        assert isinstance(profile["warnings"], list)

    def test_profile_file_source_not_found(self, profiler, mock_db):
        """Test error when data source not found."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError, match="not found"):
            profiler.profile_data_source(mock_db, uuid4())

    def test_profile_file_source_missing_path(self, profiler, mock_db):
        """Test error when file path is missing."""
        data_source = DataSource(
            id=uuid4(),
            project_id=uuid4(),
            name="test_file.csv",
            type=DataSourceType.FILE_UPLOAD,
            config_json={},  # No file_path
        )
        mock_db.query.return_value.filter.return_value.first.return_value = data_source

        with pytest.raises(ValueError, match="missing file_path"):
            profiler.profile_data_source(mock_db, data_source.id)


class TestDataProfilerExternalDataset:
    """Test profiling external datasets."""

    @pytest.fixture
    def profiler(self):
        """Create a DataProfiler instance."""
        return DataProfiler(sample_rows=1000)

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_profile_external_dataset_from_schema(self, profiler, mock_db):
        """Test profiling an external dataset using schema_summary."""
        data_source = DataSource(
            id=uuid4(),
            project_id=uuid4(),
            name="External Dataset",
            type=DataSourceType.EXTERNAL_DATASET,
            config_json={
                "source_url": "https://example.com/dataset",
            },
            schema_summary={
                "row_count": 10000,
                "columns": [
                    {"name": "col1", "dtype": "int64", "inferred_type": "numeric", "null_count": 0},
                    {"name": "col2", "dtype": "object", "inferred_type": "categorical", "null_count": 10},
                ],
            },
        )
        mock_db.query.return_value.filter.return_value.first.return_value = data_source

        profile = profiler.profile_data_source(mock_db, data_source.id)

        assert profile["source_id"] == str(data_source.id)
        assert profile["source_type"] == "external_dataset"
        assert profile["estimated_row_count"] == 10000
        assert profile["is_estimate"] is True
        assert len(profile["columns"]) == 2


class TestProfileDataSourceFunction:
    """Test the convenience function profile_data_source."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            df.to_csv(f, index=False)
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    def test_profile_data_source_function(self, mock_db, sample_csv_file):
        """Test the convenience function."""
        data_source = DataSource(
            id=uuid4(),
            project_id=uuid4(),
            name="test.csv",
            type=DataSourceType.FILE_UPLOAD,
            config_json={
                "file_path": str(sample_csv_file),
                "file_type": "csv",
            },
        )
        mock_db.query.return_value.filter.return_value.first.return_value = data_source

        profile = profile_data_source(mock_db, data_source.id)

        assert profile["source_id"] == str(data_source.id)
        assert profile["column_count"] == 2


class TestProfileAllDataSources:
    """Test profiling all data sources in a project."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def sample_csv_files(self):
        """Create multiple sample CSV files."""
        files = []
        for i in range(3):
            df = pd.DataFrame({
                "id": list(range(10)),
                "value": [j * (i + 1) for j in range(10)],
            })
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
                df.to_csv(f, index=False)
                files.append(Path(f.name))
        yield files
        for f in files:
            if f.exists():
                f.unlink()

    def test_profile_all_data_sources(self, mock_db, sample_csv_files):
        """Test profiling all data sources in a project."""
        from app.models.project import Project

        project_id = uuid4()
        data_sources = [
            DataSource(
                id=uuid4(),
                project_id=project_id,
                name=f"file_{i}.csv",
                type=DataSourceType.FILE_UPLOAD,
                config_json={
                    "file_path": str(sample_csv_files[i]),
                    "file_type": "csv",
                },
            )
            for i in range(3)
        ]

        project = MagicMock()
        project.id = project_id
        project.data_sources = data_sources

        mock_db.query.return_value.filter.return_value.first.return_value = project

        # Also mock the individual data source queries
        def mock_query(model):
            mock_result = MagicMock()
            if model == Project:
                mock_result.filter.return_value.first.return_value = project
            else:  # DataSource
                def filter_side_effect(*args, **kwargs):
                    filter_mock = MagicMock()
                    # Return matching data source based on id
                    for ds in data_sources:
                        filter_mock.first.return_value = ds
                    return filter_mock
                mock_result.filter.side_effect = filter_side_effect
            return mock_result

        mock_db.query.side_effect = mock_query

        result = profile_all_data_sources(mock_db, project_id)

        assert result["project_id"] == str(project_id)
        assert result["total_sources"] == 3
        assert result["profiled_count"] == 3
        assert result["error_count"] == 0
        assert len(result["profiles"]) == 3

    def test_profile_all_handles_errors(self, mock_db):
        """Test that profiling handles individual errors gracefully."""
        from app.models.project import Project

        project_id = uuid4()
        good_source_id = uuid4()
        bad_source_id = uuid4()

        # Create mock data sources
        good_source = MagicMock()
        good_source.id = good_source_id
        good_source.name = "good.csv"

        bad_source = MagicMock()
        bad_source.id = bad_source_id
        bad_source.name = "missing.csv"

        data_sources = [good_source, bad_source]

        project = MagicMock()
        project.id = project_id
        project.data_sources = data_sources

        # Mock the project query
        mock_db.query.return_value.filter.return_value.first.return_value = project

        # Use patch to mock the profiler methods
        with patch.object(DataProfiler, 'profile_data_source') as mock_profile:
            # First call succeeds, second fails
            mock_profile.side_effect = [
                {"source_id": str(good_source_id), "source_name": "good.csv", "columns": []},
                ValueError("File not found: /nonexistent/path/file.csv"),
            ]

            result = profile_all_data_sources(mock_db, project_id)

            assert result["total_sources"] == 2
            assert result["profiled_count"] == 1
            assert result["error_count"] == 1
            assert len(result["errors"]) == 1
            assert "missing.csv" in result["errors"][0]["source_name"]


class TestDataProfilerExampleValues:
    """Test example value extraction."""

    @pytest.fixture
    def profiler(self):
        """Create a DataProfiler instance."""
        return DataProfiler(sample_rows=1000)

    def test_example_values_numeric(self, profiler):
        """Test example values for numeric columns."""
        series = pd.Series([1, 5, 10, 15, 20])
        examples = profiler._get_example_values(series, "numeric")

        assert len(examples) >= 2
        assert "1" in examples  # min
        assert "20" in examples  # max

    def test_example_values_categorical(self, profiler):
        """Test example values for categorical columns."""
        series = pd.Series(["A", "A", "A", "B", "B", "C"])
        examples = profiler._get_example_values(series, "categorical")

        assert "A" in examples  # Most common
        assert len(examples) <= 5

    def test_example_values_text(self, profiler):
        """Test example values for text columns are truncated."""
        long_text = "x" * 200
        series = pd.Series([long_text])
        examples = profiler._get_example_values(series, "text")

        assert len(examples[0]) == 100  # Truncated

    def test_example_values_empty(self, profiler):
        """Test example values for empty series."""
        series = pd.Series([], dtype=object)
        examples = profiler._get_example_values(series, "categorical")

        assert examples == []

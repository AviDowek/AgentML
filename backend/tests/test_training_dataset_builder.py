"""Tests for Training Dataset Builder and Build Step.

Phase 12.4: Materialize Training Dataset & Register as DataSource
"""
import pytest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

from app.models import (
    AgentRun,
    AgentStep,
    AgentStepType,
    AgentStepStatus,
    AgentStepLog,
    LogMessageType,
    DataSource,
    DataSourceType,
    DatasetSpec,
    Project,
)
from app.services.training_dataset_builder import (
    materialize_training_dataset,
    _build_table_source_mapping,
    _load_data_source,
    _apply_filters,
    _execute_join,
    _aggregate_and_join,
)
from app.services.agent_executor import (
    StepLogger,
    handle_training_dataset_build_step,
    run_agent_step,
)


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def sample_customers_df():
    """Create sample customers dataframe."""
    return pd.DataFrame({
        "customer_id": list(range(1, 101)),
        "name": [f"Customer {i}" for i in range(1, 101)],
        "segment": ["A", "B", "C", "D"] * 25,
        "churned": [0, 1] * 50,
    })


@pytest.fixture
def sample_transactions_df():
    """Create sample transactions dataframe."""
    transactions = []
    for i in range(1, 501):
        transactions.append({
            "transaction_id": i,
            "customer_id": (i % 100) + 1,  # Distribute across customers
            "amount": 50.0 + (i % 100),
            "category": ["Electronics", "Groceries", "Clothing"][i % 3],
        })
    return pd.DataFrame(transactions)


@pytest.fixture
def customers_csv_file(sample_customers_df):
    """Create a temporary CSV file with customers data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        sample_customers_df.to_csv(f, index=False)
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def transactions_csv_file(sample_transactions_df):
    """Create a temporary CSV file with transactions data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        sample_transactions_df.to_csv(f, index=False)
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def project_with_data_sources(db_session, customers_csv_file, transactions_csv_file):
    """Create a project with customers and transactions data sources."""
    project = Project(name="Test Project")
    db_session.add(project)
    db_session.commit()

    # Create customers data source
    customers_ds = DataSource(
        project_id=project.id,
        name="customers.csv",
        type=DataSourceType.FILE_UPLOAD,
        config_json={
            "file_path": str(customers_csv_file),
            "file_type": "csv",
        },
        profile_json={
            "estimated_row_count": 100,
            "column_count": 4,
            "columns": [
                {"name": "customer_id", "inferred_type": "id", "distinct_count": 100},
                {"name": "name", "inferred_type": "text", "distinct_count": 100},
                {"name": "segment", "inferred_type": "categorical", "distinct_count": 4},
                {"name": "churned", "inferred_type": "boolean", "distinct_count": 2},
            ],
        },
    )
    db_session.add(customers_ds)

    # Create transactions data source
    transactions_ds = DataSource(
        project_id=project.id,
        name="transactions.csv",
        type=DataSourceType.FILE_UPLOAD,
        config_json={
            "file_path": str(transactions_csv_file),
            "file_type": "csv",
        },
        profile_json={
            "estimated_row_count": 500,
            "column_count": 4,
            "columns": [
                {"name": "transaction_id", "inferred_type": "id", "distinct_count": 500},
                {"name": "customer_id", "inferred_type": "id", "distinct_count": 100},
                {"name": "amount", "inferred_type": "numeric", "distinct_count": 100},
                {"name": "category", "inferred_type": "categorical", "distinct_count": 3},
            ],
        },
    )
    db_session.add(transactions_ds)

    db_session.commit()

    return {
        "project": project,
        "customers_ds": customers_ds,
        "transactions_ds": transactions_ds,
    }


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.chat_json = AsyncMock()
    return client


# ============================================
# Helper Function Tests
# ============================================

class TestBuildTableSourceMapping:
    """Tests for _build_table_source_mapping."""

    def test_extracts_table_name_from_csv(self, db_session, project_with_data_sources):
        """Test table name extraction from CSV file names."""
        data_sources = db_session.query(DataSource).filter(
            DataSource.project_id == project_with_data_sources["project"].id
        ).all()

        mapping = _build_table_source_mapping(data_sources)

        assert "customers" in mapping
        assert "transactions" in mapping
        assert mapping["customers"].name == "customers.csv"
        assert mapping["transactions"].name == "transactions.csv"

    def test_mapping_preserves_original_names(self, db_session, project_with_data_sources):
        """Test that original names are also preserved in mapping."""
        data_sources = db_session.query(DataSource).filter(
            DataSource.project_id == project_with_data_sources["project"].id
        ).all()

        mapping = _build_table_source_mapping(data_sources)

        # Should also work with original names
        assert "customers.csv" in mapping
        assert "transactions.csv" in mapping


class TestLoadDataSource:
    """Tests for _load_data_source."""

    def test_loads_csv_file(self, db_session, project_with_data_sources):
        """Test loading a CSV data source."""
        customers_ds = project_with_data_sources["customers_ds"]

        df = _load_data_source(customers_ds)

        assert len(df) == 100
        assert "customer_id" in df.columns
        assert "name" in df.columns
        assert "churned" in df.columns

    def test_loads_with_max_rows(self, db_session, project_with_data_sources):
        """Test loading with row limit."""
        customers_ds = project_with_data_sources["customers_ds"]

        df = _load_data_source(customers_ds, max_rows=50)

        assert len(df) == 50

    def test_raises_on_missing_file(self, db_session):
        """Test error when file is missing."""
        project = Project(name="Missing File Test")
        db_session.add(project)
        db_session.commit()

        ds = DataSource(
            project_id=project.id,
            name="nonexistent.csv",
            type=DataSourceType.FILE_UPLOAD,
            config_json={
                "file_path": "/nonexistent/path/file.csv",
                "file_type": "csv",
            },
        )
        db_session.add(ds)
        db_session.commit()

        with pytest.raises(ValueError, match="File not found"):
            _load_data_source(ds)

    def test_raises_on_unsupported_type(self, db_session):
        """Test error for unsupported data source type."""
        project = Project(name="Unsupported Type Test")
        db_session.add(project)
        db_session.commit()

        ds = DataSource(
            project_id=project.id,
            name="database_source",
            type=DataSourceType.DATABASE,
            config_json={"connection_string": "postgresql://..."},
        )
        db_session.add(ds)
        db_session.commit()

        with pytest.raises(ValueError, match="Only file_upload data sources"):
            _load_data_source(ds)


class TestApplyFilters:
    """Tests for _apply_filters."""

    def test_applies_gte_filter(self, sample_customers_df):
        """Test >= filter operator."""
        filters = [{"column": "customer_id", "operator": ">=", "value": 50}]

        result = _apply_filters(sample_customers_df.copy(), filters)

        assert len(result) == 51  # IDs 50-100
        assert result["customer_id"].min() == 50

    def test_applies_eq_filter(self, sample_customers_df):
        """Test == filter operator."""
        filters = [{"column": "segment", "operator": "==", "value": "A"}]

        result = _apply_filters(sample_customers_df.copy(), filters)

        assert len(result) == 25
        assert all(result["segment"] == "A")

    def test_applies_in_filter(self, sample_customers_df):
        """Test 'in' filter operator."""
        filters = [{"column": "segment", "operator": "in", "value": ["A", "B"]}]

        result = _apply_filters(sample_customers_df.copy(), filters)

        assert len(result) == 50
        assert set(result["segment"].unique()) == {"A", "B"}

    def test_skips_missing_column(self, sample_customers_df):
        """Test that missing columns are skipped."""
        filters = [{"column": "nonexistent", "operator": "==", "value": "foo"}]

        # Should not raise, just skip
        result = _apply_filters(sample_customers_df.copy(), filters)

        assert len(result) == 100  # No rows filtered


class TestExecuteJoin:
    """Tests for _execute_join."""

    def test_one_to_one_join(self, sample_customers_df):
        """Test one-to-one join."""
        # Create a simple one-to-one mapping
        extra_df = pd.DataFrame({
            "customer_id": list(range(1, 101)),
            "loyalty_score": list(range(100)),
        })

        join_item = {
            "left_key": "customer_id",
            "right_key": "customer_id",
            "relationship": "one_to_one",
            "aggregation": None,
        }

        result = _execute_join(sample_customers_df.copy(), extra_df, join_item)

        assert len(result) == 100
        assert "loyalty_score" in result.columns

    def test_one_to_many_with_aggregation(self, sample_customers_df, sample_transactions_df):
        """Test one-to-many join with aggregation."""
        join_item = {
            "left_key": "customer_id",
            "right_key": "customer_id",
            "relationship": "one_to_many",
            "aggregation": {
                "window_days": None,
                "features": [
                    {"name": "total_amount", "agg": "sum", "column": "amount"},
                    {"name": "tx_count", "agg": "count", "column": "*"},
                ],
            },
        }

        result = _execute_join(sample_customers_df.copy(), sample_transactions_df, join_item)

        assert len(result) == 100
        assert "total_amount" in result.columns
        assert "tx_count" in result.columns
        # Each customer has 5 transactions (500 / 100)
        assert result["tx_count"].mean() == 5.0


class TestAggregateAndJoin:
    """Tests for _aggregate_and_join."""

    def test_sum_aggregation(self, sample_customers_df, sample_transactions_df):
        """Test sum aggregation."""
        aggregation = {
            "window_days": None,
            "features": [
                {"name": "total_spend", "agg": "sum", "column": "amount"},
            ],
        }

        result = _aggregate_and_join(
            sample_customers_df.copy(),
            sample_transactions_df,
            "customer_id",
            "customer_id",
            aggregation,
        )

        assert "total_spend" in result.columns
        assert result["total_spend"].sum() > 0

    def test_count_aggregation(self, sample_customers_df, sample_transactions_df):
        """Test count aggregation."""
        aggregation = {
            "window_days": None,
            "features": [
                {"name": "num_transactions", "agg": "count", "column": "*"},
            ],
        }

        result = _aggregate_and_join(
            sample_customers_df.copy(),
            sample_transactions_df,
            "customer_id",
            "customer_id",
            aggregation,
        )

        assert "num_transactions" in result.columns
        assert result["num_transactions"].sum() == 500

    def test_multiple_aggregations(self, sample_customers_df, sample_transactions_df):
        """Test multiple aggregations."""
        aggregation = {
            "window_days": None,
            "features": [
                {"name": "total", "agg": "sum", "column": "amount"},
                {"name": "average", "agg": "avg", "column": "amount"},
                {"name": "count", "agg": "count", "column": "*"},
            ],
        }

        result = _aggregate_and_join(
            sample_customers_df.copy(),
            sample_transactions_df,
            "customer_id",
            "customer_id",
            aggregation,
        )

        assert "total" in result.columns
        assert "average" in result.columns
        assert "count" in result.columns


# ============================================
# Materialize Training Dataset Tests
# ============================================

class TestMaterializeTrainingDataset:
    """Tests for materialize_training_dataset function."""

    def test_basic_materialization(self, db_session, project_with_data_sources):
        """Test basic dataset materialization without joins."""
        project = project_with_data_sources["project"]

        training_spec = {
            "base_table": "customers",
            "base_filters": [],
            "target_definition": {
                "table": "customers",
                "column": "churned",
            },
            "join_plan": [],
            "excluded_columns": ["customer_id"],
        }

        result = materialize_training_dataset(
            db=db_session,
            project_id=project.id,
            training_dataset_spec=training_spec,
            max_rows=1000,
            output_format="csv",
        )

        # Verify data source was created
        ds = db_session.query(DataSource).filter(DataSource.id == result.data_source_id).first()
        assert ds is not None
        assert "training_dataset" in ds.name
        assert ds.schema_summary is not None
        assert ds.schema_summary["row_count"] == 100
        assert ds.schema_summary["target_column"] == "churned"

        # Verify customer_id was excluded
        columns = [c["name"] for c in ds.schema_summary["columns"]]
        assert "customer_id" not in columns
        assert "churned" in columns

    def test_materialization_with_join(self, db_session, project_with_data_sources):
        """Test materialization with a join."""
        project = project_with_data_sources["project"]

        training_spec = {
            "base_table": "customers",
            "base_filters": [],
            "target_definition": {
                "table": "customers",
                "column": "churned",
            },
            "join_plan": [
                {
                    "from_table": "customers",
                    "to_table": "transactions",
                    "left_key": "customer_id",
                    "right_key": "customer_id",
                    "relationship": "one_to_many",
                    "aggregation": {
                        "window_days": None,
                        "features": [
                            {"name": "total_spend", "agg": "sum", "column": "amount"},
                            {"name": "tx_count", "agg": "count", "column": "*"},
                        ],
                    },
                }
            ],
            "excluded_columns": [],
        }

        result = materialize_training_dataset(
            db=db_session,
            project_id=project.id,
            training_dataset_spec=training_spec,
            max_rows=1000,
            output_format="csv",
        )

        # Verify data source was created with aggregated features
        ds = db_session.query(DataSource).filter(DataSource.id == result.data_source_id).first()
        columns = [c["name"] for c in ds.schema_summary["columns"]]
        assert "total_spend" in columns
        assert "tx_count" in columns

    def test_materialization_with_filter(self, db_session, project_with_data_sources):
        """Test materialization with base filter."""
        project = project_with_data_sources["project"]

        training_spec = {
            "base_table": "customers",
            "base_filters": [
                {"column": "segment", "operator": "==", "value": "A"}
            ],
            "target_definition": {
                "table": "customers",
                "column": "churned",
            },
            "join_plan": [],
            "excluded_columns": [],
        }

        result = materialize_training_dataset(
            db=db_session,
            project_id=project.id,
            training_dataset_spec=training_spec,
            max_rows=1000,
            output_format="csv",
        )

        # Verify row count matches filter
        ds = db_session.query(DataSource).filter(DataSource.id == result.data_source_id).first()
        assert ds.schema_summary["row_count"] == 25  # Only segment A

    def test_creates_dataset_spec(self, db_session, project_with_data_sources):
        """Test that DatasetSpec is created."""
        project = project_with_data_sources["project"]

        training_spec = {
            "base_table": "customers",
            "base_filters": [],
            "target_definition": {
                "table": "customers",
                "column": "churned",
            },
            "join_plan": [],
            "excluded_columns": [],
        }

        materialize_training_dataset(
            db=db_session,
            project_id=project.id,
            training_dataset_spec=training_spec,
            max_rows=1000,
            output_format="csv",
        )

        # Verify DatasetSpec was created
        dataset_spec = db_session.query(DatasetSpec).filter(
            DatasetSpec.project_id == project.id
        ).first()
        assert dataset_spec is not None
        assert dataset_spec.target_column == "churned"
        assert "churned" not in dataset_spec.feature_columns

    def test_raises_on_missing_project(self, db_session):
        """Test error when project doesn't exist."""
        fake_project_id = uuid.uuid4()

        training_spec = {
            "base_table": "customers",
            "target_definition": {"table": "customers", "column": "churned"},
        }

        with pytest.raises(ValueError, match="Project .* not found"):
            materialize_training_dataset(
                db=db_session,
                project_id=fake_project_id,
                training_dataset_spec=training_spec,
            )

    def test_raises_on_missing_base_table(self, db_session, project_with_data_sources):
        """Test error when base table doesn't exist."""
        project = project_with_data_sources["project"]

        training_spec = {
            "base_table": "nonexistent_table",
            "target_definition": {"table": "customers", "column": "churned"},
        }

        with pytest.raises(ValueError, match="Base table .* not found"):
            materialize_training_dataset(
                db=db_session,
                project_id=project.id,
                training_dataset_spec=training_spec,
            )

    def test_raises_on_missing_target_column(self, db_session, project_with_data_sources):
        """Test error when target column doesn't exist."""
        project = project_with_data_sources["project"]

        training_spec = {
            "base_table": "customers",
            "target_definition": {
                "table": "customers",
                "column": "nonexistent_target",
            },
            "join_plan": [],
        }

        with pytest.raises(ValueError, match="Target column .* not found"):
            materialize_training_dataset(
                db=db_session,
                project_id=project.id,
                training_dataset_spec=training_spec,
            )


# ============================================
# Large Dataset Sampling Tests
# ============================================

class TestLargeDatasetSampling:
    """Tests for large dataset safeguards and sampling."""

    def test_sampling_when_exceeds_max_rows(self, db_session, test_user, tmp_path):
        """Test that sampling is applied when dataset exceeds max_rows."""
        # Create a project
        project = Project(name="Large Dataset Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        # Create a large CSV file (1000 rows)
        large_csv = tmp_path / "large_customers.csv"
        rows = []
        rows.append("customer_id,name,churned")
        for i in range(1000):
            rows.append(f"{i},Customer{i},{i % 2}")
        large_csv.write_text("\n".join(rows))

        # Create data source
        ds = DataSource(
            project_id=project.id,
            name="large_customers",
            type=DataSourceType.FILE_UPLOAD,
            config_json={"file_path": str(large_csv), "file_type": "csv"},
        )
        db_session.add(ds)
        db_session.commit()

        training_spec = {
            "base_table": "large_customers",
            "base_filters": [],
            "target_definition": {"table": "large_customers", "column": "churned"},
            "join_plan": [],
            "excluded_columns": [],
        }

        # Materialize with max_rows=100 (less than 1000)
        result = materialize_training_dataset(
            db=db_session,
            project_id=project.id,
            training_dataset_spec=training_spec,
            max_rows=100,
            output_format="csv",
        )

        # Verify sampling occurred
        assert result.was_sampled is True
        assert result.row_count == 100
        assert result.original_row_count == 1000
        assert result.sampling_message is not None
        assert "1K rows" in result.sampling_message  # Formatted row count

    def test_no_sampling_when_under_max_rows(self, db_session, test_user, tmp_path):
        """Test that no sampling occurs when dataset is under max_rows."""
        project = Project(name="Small Dataset Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        # Create a small CSV file (50 rows)
        small_csv = tmp_path / "small_customers.csv"
        rows = ["customer_id,name,churned"]
        for i in range(50):
            rows.append(f"{i},Customer{i},{i % 2}")
        small_csv.write_text("\n".join(rows))

        ds = DataSource(
            project_id=project.id,
            name="small_customers",
            type=DataSourceType.FILE_UPLOAD,
            config_json={"file_path": str(small_csv), "file_type": "csv"},
        )
        db_session.add(ds)
        db_session.commit()

        training_spec = {
            "base_table": "small_customers",
            "base_filters": [],
            "target_definition": {"table": "small_customers", "column": "churned"},
            "join_plan": [],
            "excluded_columns": [],
        }

        result = materialize_training_dataset(
            db=db_session,
            project_id=project.id,
            training_dataset_spec=training_spec,
            max_rows=100,  # More than our 50 rows
            output_format="csv",
        )

        # Verify no sampling occurred
        assert result.was_sampled is False
        assert result.row_count == 50
        assert result.original_row_count is None
        assert result.sampling_message is None

    def test_uses_project_max_training_rows_setting(self, db_session, test_user, tmp_path):
        """Test that project's max_training_rows setting is used by default."""
        # Create a project with custom max_training_rows
        project = Project(
            name="Custom Limit Test",
            owner_id=test_user.id,
            max_training_rows=50,  # Low limit
        )
        db_session.add(project)
        db_session.commit()

        # Create CSV with 200 rows
        csv_file = tmp_path / "customers.csv"
        rows = ["customer_id,name,churned"]
        for i in range(200):
            rows.append(f"{i},Customer{i},{i % 2}")
        csv_file.write_text("\n".join(rows))

        ds = DataSource(
            project_id=project.id,
            name="customers",
            type=DataSourceType.FILE_UPLOAD,
            config_json={"file_path": str(csv_file), "file_type": "csv"},
        )
        db_session.add(ds)
        db_session.commit()

        training_spec = {
            "base_table": "customers",
            "base_filters": [],
            "target_definition": {"table": "customers", "column": "churned"},
            "join_plan": [],
            "excluded_columns": [],
        }

        # Don't pass max_rows - should use project's setting
        result = materialize_training_dataset(
            db=db_session,
            project_id=project.id,
            training_dataset_spec=training_spec,
            output_format="csv",
        )

        # Should have sampled to project's limit of 50
        assert result.was_sampled is True
        assert result.row_count == 50
        assert result.original_row_count == 200

    def test_materialized_rows_never_exceed_max(self, db_session, test_user, tmp_path):
        """Test that materialized dataset rows never exceed max_training_rows."""
        project = Project(name="Limit Enforcement Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        # Create CSV with various sizes
        for size in [100, 500, 2000]:
            csv_file = tmp_path / f"data_{size}.csv"
            rows = ["id,value,target"]
            for i in range(size):
                rows.append(f"{i},{i*10},{i % 3}")
            csv_file.write_text("\n".join(rows))

            ds = DataSource(
                project_id=project.id,
                name=f"data_{size}",
                type=DataSourceType.FILE_UPLOAD,
                config_json={"file_path": str(csv_file), "file_type": "csv"},
            )
            db_session.add(ds)
        db_session.commit()

        max_rows = 250

        for size in [100, 500, 2000]:
            training_spec = {
                "base_table": f"data_{size}",
                "base_filters": [],
                "target_definition": {"table": f"data_{size}", "column": "target"},
                "join_plan": [],
                "excluded_columns": [],
            }

            result = materialize_training_dataset(
                db=db_session,
                project_id=project.id,
                training_dataset_spec=training_spec,
                max_rows=max_rows,
                output_format="csv",
            )

            # Row count should never exceed max_rows
            assert result.row_count <= max_rows
            if size > max_rows:
                assert result.was_sampled is True
            else:
                assert result.was_sampled is False


class TestFormatRowCount:
    """Tests for the _format_row_count helper function."""

    def test_format_millions(self):
        """Test formatting for millions."""
        from app.services.training_dataset_builder import _format_row_count

        assert _format_row_count(1_000_000) == "1M"
        assert _format_row_count(25_000_000) == "25M"
        assert _format_row_count(1_500_000) == "1.5M"
        assert _format_row_count(10_250_000) == "10.2M"

    def test_format_thousands(self):
        """Test formatting for thousands."""
        from app.services.training_dataset_builder import _format_row_count

        assert _format_row_count(1_000) == "1K"
        assert _format_row_count(50_000) == "50K"
        assert _format_row_count(1_500) == "1.5K"
        assert _format_row_count(500_000) == "500K"

    def test_format_small_numbers(self):
        """Test formatting for small numbers."""
        from app.services.training_dataset_builder import _format_row_count

        assert _format_row_count(100) == "100"
        assert _format_row_count(999) == "999"
        assert _format_row_count(1) == "1"


# ============================================
# Handle Training Dataset Build Step Tests
# ============================================

class TestHandleTrainingDatasetBuildStep:
    """Tests for handle_training_dataset_build_step."""

    @pytest.mark.asyncio
    async def test_build_step_success(self, db_session, mock_llm_client, project_with_data_sources):
        """Test successful training dataset build step."""
        project = project_with_data_sources["project"]

        agent_run = AgentRun(name="Build Step Test", project_id=project.id)
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_BUILD,
            input_json={
                "project_id": str(project.id),
                "training_dataset_spec": {
                    "base_table": "customers",
                    "base_filters": [],
                    "target_definition": {
                        "table": "customers",
                        "column": "churned",
                    },
                    "join_plan": [],
                    "excluded_columns": [],
                },
                "max_rows": 1000,
                "output_format": "csv",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_training_dataset_build_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify output
        assert "data_source_id" in output
        assert "row_count" in output
        assert "column_count" in output
        assert "target_column" in output
        assert output["target_column"] == "churned"
        assert output["row_count"] == 100

        # Verify logs were created
        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).all()
        assert len(logs) > 0

        # Verify summary log exists
        summary_logs = [l for l in logs if l.message_type == LogMessageType.SUMMARY]
        assert len(summary_logs) == 1

    @pytest.mark.asyncio
    async def test_build_step_missing_project_id(self, db_session, mock_llm_client):
        """Test that missing project_id raises error."""
        agent_run = AgentRun(name="Missing Project Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_BUILD,
            input_json={
                # Missing project_id
                "training_dataset_spec": {
                    "base_table": "customers",
                    "target_definition": {"column": "churned"},
                },
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Missing 'project_id'"):
            await handle_training_dataset_build_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

    @pytest.mark.asyncio
    async def test_build_step_missing_spec(self, db_session, mock_llm_client, project_with_data_sources):
        """Test that missing training_dataset_spec raises error."""
        project = project_with_data_sources["project"]

        agent_run = AgentRun(name="Missing Spec Test", project_id=project.id)
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_BUILD,
            input_json={
                "project_id": str(project.id),
                # Missing training_dataset_spec
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Missing 'training_dataset_spec'"):
            await handle_training_dataset_build_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

    @pytest.mark.asyncio
    async def test_build_step_with_joins(self, db_session, mock_llm_client, project_with_data_sources):
        """Test build step with aggregation joins."""
        project = project_with_data_sources["project"]

        agent_run = AgentRun(name="Build with Joins Test", project_id=project.id)
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_BUILD,
            input_json={
                "project_id": str(project.id),
                "training_dataset_spec": {
                    "base_table": "customers",
                    "base_filters": [],
                    "target_definition": {
                        "table": "customers",
                        "column": "churned",
                    },
                    "join_plan": [
                        {
                            "from_table": "customers",
                            "to_table": "transactions",
                            "left_key": "customer_id",
                            "right_key": "customer_id",
                            "relationship": "one_to_many",
                            "aggregation": {
                                "window_days": None,
                                "features": [
                                    {"name": "total_spend", "agg": "sum", "column": "amount"},
                                ],
                            },
                        }
                    ],
                    "excluded_columns": [],
                },
                "max_rows": 1000,
                "output_format": "csv",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_training_dataset_build_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        assert "total_spend" in output["feature_columns"]

    @pytest.mark.asyncio
    async def test_build_step_runs_through_run_agent_step(
        self, db_session, mock_llm_client, project_with_data_sources
    ):
        """Test that step status becomes completed when run through run_agent_step."""
        project = project_with_data_sources["project"]

        agent_run = AgentRun(name="Full Run Test", project_id=project.id)
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_BUILD,
            status=AgentStepStatus.PENDING,
            input_json={
                "project_id": str(project.id),
                "training_dataset_spec": {
                    "base_table": "customers",
                    "base_filters": [],
                    "target_definition": {
                        "table": "customers",
                        "column": "churned",
                    },
                    "join_plan": [],
                    "excluded_columns": [],
                },
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        result = await run_agent_step(db_session, agent_step.id, mock_llm_client)

        assert result.status == AgentStepStatus.COMPLETED
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.output_json is not None
        assert "data_source_id" in result.output_json

    @pytest.mark.asyncio
    async def test_build_step_logs_info_and_thought(
        self, db_session, mock_llm_client, project_with_data_sources
    ):
        """Test that appropriate logs are created during build."""
        project = project_with_data_sources["project"]

        agent_run = AgentRun(name="Logging Test", project_id=project.id)
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_BUILD,
            input_json={
                "project_id": str(project.id),
                "training_dataset_spec": {
                    "base_table": "customers",
                    "base_filters": [],
                    "target_definition": {
                        "table": "customers",
                        "column": "churned",
                    },
                    "join_plan": [],
                    "excluded_columns": [],
                },
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        await handle_training_dataset_build_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify info logs
        info_logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.INFO,
        ).all()
        assert len(info_logs) >= 2

        # Verify thought logs
        thought_logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.THOUGHT,
        ).all()
        assert len(thought_logs) >= 1
        assert any("Base table" in log.message for log in thought_logs)

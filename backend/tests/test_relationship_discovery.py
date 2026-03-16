"""Tests for relationship discovery service."""
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.services.relationship_discovery import (
    discover_relationships_for_project,
    _extract_table_info,
    _extract_table_name,
    _is_potential_key,
    _is_potential_target,
    _find_matching_columns,
    _infer_cardinality,
    _identify_base_table_candidates,
)
from app.models.data_source import DataSource, DataSourceType


class TestExtractTableName:
    """Test table name extraction from source names."""

    def test_removes_csv_extension(self):
        assert _extract_table_name("customers.csv") == "customers"

    def test_removes_xlsx_extension(self):
        assert _extract_table_name("orders.xlsx") == "orders"

    def test_removes_json_extension(self):
        assert _extract_table_name("events.json") == "events"

    def test_removes_parquet_extension(self):
        assert _extract_table_name("transactions.parquet") == "transactions"

    def test_removes_data_prefix(self):
        assert _extract_table_name("data_customers.csv") == "customers"

    def test_removes_raw_prefix(self):
        assert _extract_table_name("raw_orders.csv") == "orders"

    def test_handles_spaces(self):
        assert _extract_table_name("customer data.csv") == "customer_data"

    def test_handles_mixed_case(self):
        assert _extract_table_name("CustomerOrders.CSV") == "customerorders"

    def test_handles_multiple_underscores(self):
        assert _extract_table_name("raw__data__final.csv") == "data"


class TestIsPotentialKey:
    """Test potential key column identification."""

    def test_id_column(self):
        assert _is_potential_key("id", {"inferred_type": "numeric"})

    def test_suffixed_id_column(self):
        assert _is_potential_key("customer_id", {"inferred_type": "numeric"})

    def test_camel_case_id(self):
        assert _is_potential_key("customerId", {"inferred_type": "numeric"})

    def test_uppercase_id(self):
        assert _is_potential_key("CustomerID", {"inferred_type": "numeric"})

    def test_uuid_column(self):
        assert _is_potential_key("uuid", {"inferred_type": "text"})

    def test_key_column(self):
        assert _is_potential_key("primary_key", {"inferred_type": "numeric"})

    def test_inferred_id_type(self):
        assert _is_potential_key("some_column", {"inferred_type": "id"})

    def test_high_distinct_ratio(self):
        assert _is_potential_key("code", {"inferred_type": "numeric", "distinct_ratio": 0.98})

    def test_non_key_column(self):
        assert not _is_potential_key("name", {"inferred_type": "text", "distinct_ratio": 0.5})

    def test_text_column_not_key(self):
        # Text columns with high distinctness are still excluded
        assert not _is_potential_key("description", {"inferred_type": "text", "distinct_ratio": 0.99})


class TestIsPotentialTarget:
    """Test potential target column identification."""

    def test_target_column(self):
        assert _is_potential_target("target", {"inferred_type": "categorical"})

    def test_label_column(self):
        assert _is_potential_target("label", {"inferred_type": "categorical"})

    def test_churn_column(self):
        assert _is_potential_target("churn", {"inferred_type": "boolean"})

    def test_is_churned_column(self):
        assert _is_potential_target("is_churned", {"inferred_type": "boolean"})

    def test_price_column(self):
        assert _is_potential_target("price", {"inferred_type": "numeric"})

    def test_amount_column(self):
        assert _is_potential_target("total_amount", {"inferred_type": "numeric"})

    def test_status_column(self):
        assert _is_potential_target("order_status", {"inferred_type": "categorical"})

    def test_binary_boolean_column(self):
        assert _is_potential_target("any_column", {"inferred_type": "boolean", "distinct_count": 2})

    def test_low_cardinality_categorical(self):
        assert _is_potential_target("category", {"inferred_type": "categorical", "distinct_count": 5})

    def test_non_target_column(self):
        assert not _is_potential_target("customer_name", {"inferred_type": "text", "distinct_count": 1000})


class TestInferCardinality:
    """Test cardinality inference."""

    def test_one_to_one(self):
        # Both sides have unique values
        result = _infer_cardinality(
            ratio1=0.99, ratio2=0.98,
            distinct1=1000, distinct2=1000,
            row_count1=1000, row_count2=1000
        )
        assert result == "one_to_one"

    def test_one_to_many(self):
        # Table 1 has unique values (high ratio), table 2 has duplicates
        result = _infer_cardinality(
            ratio1=0.99, ratio2=0.1,
            distinct1=1000, distinct2=1000,
            row_count1=1000, row_count2=10000
        )
        assert result == "one_to_many"

    def test_many_to_one(self):
        # Table 1 has duplicates, table 2 has unique values
        result = _infer_cardinality(
            ratio1=0.1, ratio2=0.99,
            distinct1=1000, distinct2=1000,
            row_count1=10000, row_count2=1000
        )
        assert result == "many_to_one"

    def test_many_to_many(self):
        # Neither has unique values
        result = _infer_cardinality(
            ratio1=0.3, ratio2=0.3,
            distinct1=300, distinct2=350,
            row_count1=1000, row_count2=1000
        )
        assert result == "many_to_many"


class TestFindMatchingColumns:
    """Test column matching between tables."""

    def test_exact_id_match(self):
        """Test exact column name match on ID columns."""
        table1 = {
            "table_name": "customers",
            "id_columns": [{"name": "customer_id", "distinct_ratio": 0.99}],
            "columns": [{"name": "customer_id"}, {"name": "name"}],
        }
        table2 = {
            "table_name": "orders",
            "id_columns": [{"name": "customer_id", "distinct_ratio": 0.1}],
            "columns": [{"name": "order_id"}, {"name": "customer_id"}],
        }

        matches = _find_matching_columns(table1, table2)

        assert len(matches) >= 1
        customer_id_match = next(
            (m for m in matches if m["col1_name"].lower() == "customer_id"),
            None
        )
        assert customer_id_match is not None
        assert customer_id_match["match_type"] == "exact_id"

    def test_prefixed_id_match(self):
        """Test table_id matches table.id pattern."""
        table1 = {
            "table_name": "customers",
            "id_columns": [{"name": "id", "distinct_ratio": 0.99}],
            "columns": [{"name": "id"}, {"name": "name"}],
        }
        table2 = {
            "table_name": "orders",
            "id_columns": [{"name": "order_id", "distinct_ratio": 0.99}],
            "columns": [{"name": "order_id"}, {"name": "customers_id"}],  # matches table1 name
        }

        matches = _find_matching_columns(table1, table2)

        # Should find customers.id -> orders.customers_id
        prefixed_match = next(
            (m for m in matches if m["match_type"] == "prefixed_id"),
            None
        )
        assert prefixed_match is not None

    def test_entity_id_match(self):
        """Test entity-based ID matching (e.g., user_id in both tables)."""
        table1 = {
            "table_name": "sessions",
            "id_columns": [{"name": "session_id", "distinct_ratio": 0.99}],
            "columns": [
                {"name": "session_id"},
                {"name": "user_id"},
            ],
        }
        table2 = {
            "table_name": "purchases",
            "id_columns": [{"name": "purchase_id", "distinct_ratio": 0.99}],
            "columns": [
                {"name": "purchase_id"},
                {"name": "user_id"},
            ],
        }

        matches = _find_matching_columns(table1, table2)

        user_match = next(
            (m for m in matches if "user_id" in m["col1_name"].lower()),
            None
        )
        assert user_match is not None


class TestIdentifyBaseTableCandidates:
    """Test base table candidate identification."""

    def test_prefers_table_with_target(self):
        """Tables with target columns should score higher."""
        tables = [
            {
                "source_id": "1",
                "source_name": "data1.csv",
                "table_name": "data1",
                "row_count": 1000,
                "has_obvious_id": True,
                "has_potential_target": False,
                "target_columns": [],
                "id_columns": [{"name": "id"}],
            },
            {
                "source_id": "2",
                "source_name": "data2.csv",
                "table_name": "data2",
                "row_count": 1000,
                "has_obvious_id": True,
                "has_potential_target": True,
                "target_columns": [{"name": "churn"}],
                "id_columns": [{"name": "id"}],
            },
        ]

        candidates = _identify_base_table_candidates(tables, [])

        # Table with target should be ranked first
        assert candidates[0]["table"] == "data2"
        assert candidates[0]["has_target"] is True

    def test_prefers_entity_table_names(self):
        """Tables with entity-like names should score higher."""
        tables = [
            {
                "source_id": "1",
                "source_name": "logs.csv",
                "table_name": "logs",
                "row_count": 10000,
                "has_obvious_id": True,
                "has_potential_target": False,
                "target_columns": [],
                "id_columns": [{"name": "id"}],
            },
            {
                "source_id": "2",
                "source_name": "customers.csv",
                "table_name": "customers",
                "row_count": 1000,
                "has_obvious_id": True,
                "has_potential_target": False,
                "target_columns": [],
                "id_columns": [{"name": "id"}],
            },
        ]

        candidates = _identify_base_table_candidates(tables, [])

        # Customer table should rank higher than logs
        assert candidates[0]["table"] == "customers"

    def test_penalizes_log_tables(self):
        """Log/event tables should score lower."""
        tables = [
            {
                "source_id": "1",
                "source_name": "event_log.csv",
                "table_name": "event_log",
                "row_count": 100000,
                "has_obvious_id": True,
                "has_potential_target": True,
                "target_columns": [{"name": "status"}],
                "id_columns": [{"name": "id"}],
            },
            {
                "source_id": "2",
                "source_name": "accounts.csv",
                "table_name": "accounts",
                "row_count": 1000,
                "has_obvious_id": True,
                "has_potential_target": True,
                "target_columns": [{"name": "churn"}],
                "id_columns": [{"name": "id"}],
            },
        ]

        candidates = _identify_base_table_candidates(tables, [])

        # Accounts should rank higher than event_log
        assert candidates[0]["table"] == "accounts"

    def test_prefers_one_side_of_relationship(self):
        """Tables on 'one' side of relationships should score higher."""
        tables = [
            {
                "source_id": "1",
                "source_name": "customers.csv",
                "table_name": "customers",
                "row_count": 1000,
                "has_obvious_id": True,
                "has_potential_target": False,
                "target_columns": [],
                "id_columns": [{"name": "customer_id"}],
            },
            {
                "source_id": "2",
                "source_name": "transactions.csv",
                "table_name": "transactions",
                "row_count": 10000,
                "has_obvious_id": True,
                "has_potential_target": False,
                "target_columns": [],
                "id_columns": [{"name": "transaction_id"}],
            },
        ]

        relationships = [
            {
                "from_table": "customers",
                "to_table": "transactions",
                "from_column": "customer_id",
                "to_column": "customer_id",
                "cardinality": "one_to_many",
            }
        ]

        candidates = _identify_base_table_candidates(tables, relationships)

        # Customers (one side) should rank higher than transactions
        assert candidates[0]["table"] == "customers"


class TestDiscoverRelationshipsForProject:
    """Test the main relationship discovery function."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def fake_profiles(self):
        """Create fake profiles for testing.

        Creates profiles for:
        - customers: Primary entity with customer_id as PK
        - transactions: Many transactions per customer
        - events: Many events per customer
        - unrelated_logs: No clear relationship
        """
        customers_profile = {
            "source_id": "customers-uuid",
            "source_name": "customers.csv",
            "estimated_row_count": 1000,
            "column_count": 5,
            "columns": [
                {
                    "name": "customer_id",
                    "inferred_type": "id",
                    "dtype": "int64",
                    "null_ratio": 0.0,
                    "distinct_count": 1000,
                    "distinct_ratio": 1.0,
                },
                {
                    "name": "name",
                    "inferred_type": "text",
                    "dtype": "object",
                    "null_ratio": 0.01,
                    "distinct_count": 950,
                    "distinct_ratio": 0.95,
                },
                {
                    "name": "email",
                    "inferred_type": "text",
                    "dtype": "object",
                    "null_ratio": 0.02,
                    "distinct_count": 990,
                    "distinct_ratio": 0.99,
                },
                {
                    "name": "signup_date",
                    "inferred_type": "datetime",
                    "dtype": "datetime64",
                    "null_ratio": 0.0,
                    "distinct_count": 365,
                    "distinct_ratio": 0.365,
                },
                {
                    "name": "churned",
                    "inferred_type": "boolean",
                    "dtype": "bool",
                    "null_ratio": 0.0,
                    "distinct_count": 2,
                    "distinct_ratio": 0.002,
                },
            ],
        }

        transactions_profile = {
            "source_id": "transactions-uuid",
            "source_name": "transactions.csv",
            "estimated_row_count": 50000,
            "column_count": 6,
            "columns": [
                {
                    "name": "transaction_id",
                    "inferred_type": "id",
                    "dtype": "int64",
                    "null_ratio": 0.0,
                    "distinct_count": 50000,
                    "distinct_ratio": 1.0,
                },
                {
                    "name": "customer_id",
                    "inferred_type": "numeric",
                    "dtype": "int64",
                    "null_ratio": 0.0,
                    "distinct_count": 1000,
                    "distinct_ratio": 0.02,
                },
                {
                    "name": "amount",
                    "inferred_type": "numeric",
                    "dtype": "float64",
                    "null_ratio": 0.0,
                    "distinct_count": 5000,
                    "distinct_ratio": 0.1,
                },
                {
                    "name": "transaction_date",
                    "inferred_type": "datetime",
                    "dtype": "datetime64",
                    "null_ratio": 0.0,
                    "distinct_count": 730,
                    "distinct_ratio": 0.0146,
                },
                {
                    "name": "product_id",
                    "inferred_type": "numeric",
                    "dtype": "int64",
                    "null_ratio": 0.0,
                    "distinct_count": 500,
                    "distinct_ratio": 0.01,
                },
                {
                    "name": "status",
                    "inferred_type": "categorical",
                    "dtype": "object",
                    "null_ratio": 0.0,
                    "distinct_count": 4,
                    "distinct_ratio": 0.00008,
                },
            ],
        }

        events_profile = {
            "source_id": "events-uuid",
            "source_name": "events.csv",
            "estimated_row_count": 100000,
            "column_count": 5,
            "columns": [
                {
                    "name": "event_id",
                    "inferred_type": "id",
                    "dtype": "int64",
                    "null_ratio": 0.0,
                    "distinct_count": 100000,
                    "distinct_ratio": 1.0,
                },
                {
                    "name": "customer_id",
                    "inferred_type": "numeric",
                    "dtype": "int64",
                    "null_ratio": 0.0,
                    "distinct_count": 1000,
                    "distinct_ratio": 0.01,
                },
                {
                    "name": "event_type",
                    "inferred_type": "categorical",
                    "dtype": "object",
                    "null_ratio": 0.0,
                    "distinct_count": 20,
                    "distinct_ratio": 0.0002,
                },
                {
                    "name": "timestamp",
                    "inferred_type": "datetime",
                    "dtype": "datetime64",
                    "null_ratio": 0.0,
                    "distinct_count": 50000,
                    "distinct_ratio": 0.5,
                },
                {
                    "name": "metadata",
                    "inferred_type": "text",
                    "dtype": "object",
                    "null_ratio": 0.3,
                    "distinct_count": 80000,
                    "distinct_ratio": 0.8,
                },
            ],
        }

        unrelated_logs_profile = {
            "source_id": "logs-uuid",
            "source_name": "unrelated_logs.csv",
            "estimated_row_count": 200000,
            "column_count": 4,
            "columns": [
                {
                    "name": "log_id",
                    "inferred_type": "id",
                    "dtype": "int64",
                    "null_ratio": 0.0,
                    "distinct_count": 200000,
                    "distinct_ratio": 1.0,
                },
                {
                    "name": "server_name",
                    "inferred_type": "categorical",
                    "dtype": "object",
                    "null_ratio": 0.0,
                    "distinct_count": 10,
                    "distinct_ratio": 0.00005,
                },
                {
                    "name": "log_level",
                    "inferred_type": "categorical",
                    "dtype": "object",
                    "null_ratio": 0.0,
                    "distinct_count": 5,
                    "distinct_ratio": 0.000025,
                },
                {
                    "name": "message",
                    "inferred_type": "text",
                    "dtype": "object",
                    "null_ratio": 0.0,
                    "distinct_count": 150000,
                    "distinct_ratio": 0.75,
                },
            ],
        }

        return {
            "customers": customers_profile,
            "transactions": transactions_profile,
            "events": events_profile,
            "unrelated_logs": unrelated_logs_profile,
        }

    def test_finds_customer_transaction_relationship(self, mock_db, fake_profiles):
        """Test that it finds the relationship between customers and transactions."""
        from app.models.project import Project

        project_id = uuid4()

        # Create mock data sources
        customers_ds = MagicMock(spec=DataSource)
        customers_ds.id = uuid4()
        customers_ds.name = "customers.csv"
        customers_ds.profile_json = fake_profiles["customers"]

        transactions_ds = MagicMock(spec=DataSource)
        transactions_ds.id = uuid4()
        transactions_ds.name = "transactions.csv"
        transactions_ds.profile_json = fake_profiles["transactions"]

        # Setup mock queries
        mock_project = MagicMock()
        mock_project.id = project_id

        mock_db.query.return_value.filter.return_value.first.return_value = mock_project
        mock_db.query.return_value.filter.return_value.all.return_value = [
            customers_ds, transactions_ds
        ]

        result = discover_relationships_for_project(mock_db, project_id)

        # Should find the customer_id relationship
        relationships = result["relationships"]
        customer_txn_rel = next(
            (r for r in relationships
             if "customer" in r["from_table"] or "customer" in r["to_table"]),
            None
        )
        assert customer_txn_rel is not None
        assert customer_txn_rel["from_column"] == "customer_id" or customer_txn_rel["to_column"] == "customer_id"

    def test_customers_ranked_higher_than_transactions(self, mock_db, fake_profiles):
        """Test that customers is ranked higher as base table than transactions."""
        from app.models.project import Project

        project_id = uuid4()

        customers_ds = MagicMock(spec=DataSource)
        customers_ds.id = uuid4()
        customers_ds.name = "customers.csv"
        customers_ds.profile_json = fake_profiles["customers"]

        transactions_ds = MagicMock(spec=DataSource)
        transactions_ds.id = uuid4()
        transactions_ds.name = "transactions.csv"
        transactions_ds.profile_json = fake_profiles["transactions"]

        mock_project = MagicMock()
        mock_project.id = project_id

        mock_db.query.return_value.filter.return_value.first.return_value = mock_project
        mock_db.query.return_value.filter.return_value.all.return_value = [
            customers_ds, transactions_ds
        ]

        result = discover_relationships_for_project(mock_db, project_id)

        candidates = result["base_table_candidates"]
        assert len(candidates) >= 2

        # Find positions
        customers_idx = next(
            (i for i, c in enumerate(candidates) if "customer" in c["table"]),
            None
        )
        transactions_idx = next(
            (i for i, c in enumerate(candidates) if "transaction" in c["table"]),
            None
        )

        # Customers should be ranked higher (lower index)
        assert customers_idx is not None
        assert transactions_idx is not None
        assert customers_idx < transactions_idx

    def test_ignores_unrelated_logs(self, mock_db, fake_profiles):
        """Test that unrelated_logs is ranked lowest as base table candidate."""
        from app.models.project import Project

        project_id = uuid4()

        customers_ds = MagicMock(spec=DataSource)
        customers_ds.id = uuid4()
        customers_ds.name = "customers.csv"
        customers_ds.profile_json = fake_profiles["customers"]

        logs_ds = MagicMock(spec=DataSource)
        logs_ds.id = uuid4()
        logs_ds.name = "unrelated_logs.csv"
        logs_ds.profile_json = fake_profiles["unrelated_logs"]

        mock_project = MagicMock()
        mock_project.id = project_id

        mock_db.query.return_value.filter.return_value.first.return_value = mock_project
        mock_db.query.return_value.filter.return_value.all.return_value = [
            customers_ds, logs_ds
        ]

        result = discover_relationships_for_project(mock_db, project_id)

        candidates = result["base_table_candidates"]

        # Customers should have higher score than logs
        customers_score = next(
            (c["score"] for c in candidates if "customer" in c["table"]),
            0
        )
        logs_score = next(
            (c["score"] for c in candidates if "log" in c["table"]),
            0
        )

        assert customers_score > logs_score

    def test_full_scenario_four_tables(self, mock_db, fake_profiles):
        """Test full scenario with all four tables."""
        from app.models.project import Project

        project_id = uuid4()

        # Create all data sources
        data_sources = []
        for name, profile in fake_profiles.items():
            ds = MagicMock(spec=DataSource)
            ds.id = uuid4()
            ds.name = f"{name}.csv"
            ds.profile_json = profile
            data_sources.append(ds)

        mock_project = MagicMock()
        mock_project.id = project_id

        mock_db.query.return_value.filter.return_value.first.return_value = mock_project
        mock_db.query.return_value.filter.return_value.all.return_value = data_sources

        result = discover_relationships_for_project(mock_db, project_id)

        # Verify structure
        assert "tables" in result
        assert "relationships" in result
        assert "base_table_candidates" in result

        # Should have 4 tables
        assert len(result["tables"]) == 4

        # Should find relationships involving customer_id
        customer_relationships = [
            r for r in result["relationships"]
            if "customer_id" in (r["from_column"], r["to_column"])
        ]
        # Should find at least customer-transaction and customer-events relationships
        assert len(customer_relationships) >= 2

        # Base table candidates should be ordered by score
        candidates = result["base_table_candidates"]
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

        # Customers should be top candidate (has target column 'churned')
        assert "customer" in candidates[0]["table"]

    def test_project_not_found(self, mock_db):
        """Test error when project not found."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError, match="not found"):
            discover_relationships_for_project(mock_db, uuid4())

    def test_no_profiled_sources(self, mock_db):
        """Test handling when no data sources have profiles."""
        from app.models.project import Project

        project_id = uuid4()
        mock_project = MagicMock()
        mock_project.id = project_id

        mock_db.query.return_value.filter.return_value.first.return_value = mock_project
        mock_db.query.return_value.filter.return_value.all.return_value = []

        result = discover_relationships_for_project(mock_db, project_id)

        assert result["tables"] == []
        assert result["relationships"] == []
        assert result["base_table_candidates"] == []
        assert "warnings" in result


class TestExtractTableInfo:
    """Test table info extraction from data sources."""

    def test_extracts_id_columns(self):
        """Test that ID columns are correctly identified."""
        ds = MagicMock(spec=DataSource)
        ds.id = uuid4()
        ds.name = "test.csv"
        ds.profile_json = {
            "estimated_row_count": 100,
            "column_count": 2,
            "columns": [
                {"name": "id", "inferred_type": "id", "distinct_ratio": 1.0},
                {"name": "value", "inferred_type": "numeric", "distinct_ratio": 0.5},
            ],
        }

        tables = _extract_table_info([ds])

        assert len(tables) == 1
        assert len(tables[0]["id_columns"]) == 1
        assert tables[0]["id_columns"][0]["name"] == "id"

    def test_extracts_target_columns(self):
        """Test that target columns are correctly identified."""
        ds = MagicMock(spec=DataSource)
        ds.id = uuid4()
        ds.name = "test.csv"
        ds.profile_json = {
            "estimated_row_count": 100,
            "column_count": 2,
            "columns": [
                {"name": "id", "inferred_type": "id", "distinct_ratio": 1.0},
                {"name": "churned", "inferred_type": "boolean", "distinct_count": 2},
            ],
        }

        tables = _extract_table_info([ds])

        assert len(tables) == 1
        assert len(tables[0]["target_columns"]) == 1
        assert tables[0]["target_columns"][0]["name"] == "churned"
        assert tables[0]["has_potential_target"] is True

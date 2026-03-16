"""Integration tests for Data Architect Pipeline.

Phase 12.5: Data Architect Orchestrator Pipeline
Tests the full pipeline: inventory → relationships → planning → build
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
    AgentRunStatus,
    AgentStepLog,
    DataSource,
    DataSourceType,
    DatasetSpec,
    Project,
)
from app.services.agent_executor import (
    create_data_architect_pipeline,
    run_data_architect_pipeline,
    DATA_ARCHITECT_PIPELINE_STEPS,
    handle_dataset_inventory_step,
    handle_relationship_discovery_step,
    StepLogger,
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
def sample_events_df():
    """Create sample events dataframe."""
    events = []
    for i in range(1, 301):
        events.append({
            "event_id": i,
            "customer_id": (i % 100) + 1,
            "event_type": ["login", "page_view", "purchase"][i % 3],
            "event_count": i % 10,
        })
    return pd.DataFrame(events)


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
def events_csv_file(sample_events_df):
    """Create a temporary CSV file with events data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        sample_events_df.to_csv(f, index=False)
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def project_with_multiple_sources(
    db_session,
    customers_csv_file,
    transactions_csv_file,
    events_csv_file,
):
    """Create a project with multiple data sources for testing the full pipeline."""
    project = Project(
        name="Churn Prediction Project",
        description="Predict customer churn based on transaction and event data",
    )
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
    )
    db_session.add(transactions_ds)

    # Create events data source
    events_ds = DataSource(
        project_id=project.id,
        name="events.csv",
        type=DataSourceType.FILE_UPLOAD,
        config_json={
            "file_path": str(events_csv_file),
            "file_type": "csv",
        },
    )
    db_session.add(events_ds)

    db_session.commit()

    return {
        "project": project,
        "customers_ds": customers_ds,
        "transactions_ds": transactions_ds,
        "events_ds": events_ds,
    }


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that returns valid training dataset spec."""
    client = MagicMock()

    # Mock the chat_json method to return a valid training dataset spec
    async def mock_chat_json(*args, **kwargs):
        return {
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
                                {"name": "tx_count", "agg": "count", "column": "*"},
                            ],
                        },
                    },
                ],
                "excluded_columns": ["customer_id"],
            },
            "natural_language_summary": (
                "The training dataset uses customers as the base table with churned as the target. "
                "Features are aggregated from transactions including total spend and transaction count."
            ),
        }

    client.chat_json = AsyncMock(side_effect=mock_chat_json)
    return client


# ============================================
# Pipeline Creation Tests
# ============================================

class TestCreateDataArchitectPipeline:
    """Tests for create_data_architect_pipeline function."""

    def test_creates_pipeline_with_4_steps(self, db_session, project_with_multiple_sources):
        """Test that pipeline is created with exactly 4 steps."""
        project = project_with_multiple_sources["project"]

        agent_run = create_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
            target_hint="churned",
        )

        assert agent_run is not None
        assert len(agent_run.steps) == 4

    def test_steps_have_correct_types(self, db_session, project_with_multiple_sources):
        """Test that steps are created with correct types in order."""
        project = project_with_multiple_sources["project"]

        agent_run = create_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
        )

        step_types = [step.step_type for step in sorted(agent_run.steps, key=lambda s: s.created_at)]
        expected_types = [
            AgentStepType.DATASET_INVENTORY,
            AgentStepType.RELATIONSHIP_DISCOVERY,
            AgentStepType.TRAINING_DATASET_PLANNING,
            AgentStepType.TRAINING_DATASET_BUILD,
        ]
        assert step_types == expected_types

    def test_all_steps_start_as_pending(self, db_session, project_with_multiple_sources):
        """Test that all steps start with PENDING status."""
        project = project_with_multiple_sources["project"]

        agent_run = create_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
        )

        for step in agent_run.steps:
            assert step.status == AgentStepStatus.PENDING

    def test_agent_run_has_correct_config(self, db_session, project_with_multiple_sources):
        """Test that agent run config includes target_hint."""
        project = project_with_multiple_sources["project"]

        agent_run = create_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
            target_hint="churned",
        )

        assert agent_run.config_json is not None
        assert agent_run.config_json.get("target_hint") == "churned"

    def test_raises_on_missing_project(self, db_session):
        """Test that error is raised for non-existent project."""
        fake_project_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Project not found"):
            create_data_architect_pipeline(
                db=db_session,
                project_id=fake_project_id,
            )

    def test_raises_on_project_without_data_sources(self, db_session):
        """Test that error is raised when project has no data sources."""
        project = Project(name="Empty Project")
        db_session.add(project)
        db_session.commit()

        with pytest.raises(ValueError, match="no data sources"):
            create_data_architect_pipeline(
                db=db_session,
                project_id=project.id,
            )


# ============================================
# Individual Step Handler Tests
# ============================================

class TestHandleDatasetInventoryStep:
    """Tests for handle_dataset_inventory_step handler."""

    @pytest.mark.asyncio
    async def test_profiles_all_data_sources(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that all data sources are profiled."""
        project = project_with_multiple_sources["project"]

        agent_run = AgentRun(name="Inventory Test", project_id=project.id)
        db_session.add(agent_run)
        db_session.commit()

        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_INVENTORY,
            input_json={"project_id": str(project.id)},
        )
        db_session.add(step)
        db_session.commit()

        step_logger = StepLogger(db_session, step.id)

        output = await handle_dataset_inventory_step(
            db_session, step, step_logger, mock_llm_client
        )

        assert output["total_sources"] == 3
        assert output["profiled_count"] == 3
        assert len(output["data_source_profiles"]) == 3

    @pytest.mark.asyncio
    async def test_profiles_contain_expected_columns(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that profiles contain expected column info."""
        project = project_with_multiple_sources["project"]

        agent_run = AgentRun(name="Inventory Test", project_id=project.id)
        db_session.add(agent_run)
        db_session.commit()

        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_INVENTORY,
            input_json={"project_id": str(project.id)},
        )
        db_session.add(step)
        db_session.commit()

        step_logger = StepLogger(db_session, step.id)

        output = await handle_dataset_inventory_step(
            db_session, step, step_logger, mock_llm_client
        )

        # Find customers profile
        customers_profile = next(
            p for p in output["data_source_profiles"]
            if "customers" in p["source_name"]
        )

        column_names = [c["name"] for c in customers_profile["columns"]]
        assert "customer_id" in column_names
        assert "churned" in column_names


class TestHandleRelationshipDiscoveryStep:
    """Tests for handle_relationship_discovery_step handler."""

    @pytest.mark.asyncio
    async def test_discovers_relationships(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that relationships are discovered between tables."""
        project = project_with_multiple_sources["project"]

        # First, run inventory to get profiles
        agent_run = AgentRun(name="Relationship Test", project_id=project.id)
        db_session.add(agent_run)
        db_session.commit()

        inv_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_INVENTORY,
            input_json={"project_id": str(project.id)},
        )
        db_session.add(inv_step)
        db_session.commit()

        step_logger = StepLogger(db_session, inv_step.id)
        inv_output = await handle_dataset_inventory_step(
            db_session, inv_step, step_logger, mock_llm_client
        )

        # Now run relationship discovery
        rel_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.RELATIONSHIP_DISCOVERY,
            input_json={
                "project_id": str(project.id),
                "data_source_profiles": inv_output["data_source_profiles"],
            },
        )
        db_session.add(rel_step)
        db_session.commit()

        rel_logger = StepLogger(db_session, rel_step.id)
        rel_output = await handle_relationship_discovery_step(
            db_session, rel_step, rel_logger, mock_llm_client
        )

        assert "tables" in rel_output
        assert "relationships" in rel_output
        assert "base_table_candidates" in rel_output
        assert len(rel_output["tables"]) == 3  # customers, transactions, events

    @pytest.mark.asyncio
    async def test_identifies_base_table_candidates(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that base table candidates are identified."""
        project = project_with_multiple_sources["project"]

        agent_run = AgentRun(name="Relationship Test", project_id=project.id)
        db_session.add(agent_run)
        db_session.commit()

        # First, run inventory
        inv_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_INVENTORY,
            input_json={"project_id": str(project.id)},
        )
        db_session.add(inv_step)
        db_session.commit()

        step_logger = StepLogger(db_session, inv_step.id)
        inv_output = await handle_dataset_inventory_step(
            db_session, inv_step, step_logger, mock_llm_client
        )

        # Now run relationship discovery
        rel_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.RELATIONSHIP_DISCOVERY,
            input_json={
                "project_id": str(project.id),
                "data_source_profiles": inv_output["data_source_profiles"],
            },
        )
        db_session.add(rel_step)
        db_session.commit()

        rel_logger = StepLogger(db_session, rel_step.id)
        rel_output = await handle_relationship_discovery_step(
            db_session, rel_step, rel_logger, mock_llm_client
        )

        # Should identify customers as a base table candidate
        candidates = rel_output["base_table_candidates"]
        assert len(candidates) > 0

        # Customers should be a top candidate (has FK references from other tables)
        candidate_names = [c["table"] for c in candidates]
        assert "customers" in candidate_names


# ============================================
# Full Pipeline Integration Tests
# ============================================

class TestRunDataArchitectPipeline:
    """Integration tests for run_data_architect_pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_completes(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that full pipeline completes successfully."""
        project = project_with_multiple_sources["project"]

        agent_run = await run_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
            target_hint="churned",
            llm_client=mock_llm_client,
        )

        assert agent_run.status == AgentRunStatus.COMPLETED
        assert agent_run.error_message is None

    @pytest.mark.asyncio
    async def test_pipeline_has_4_completed_steps(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that all 4 steps complete."""
        project = project_with_multiple_sources["project"]

        agent_run = await run_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
            target_hint="churned",
            llm_client=mock_llm_client,
        )

        # Reload to get updated steps
        db_session.refresh(agent_run)

        completed_steps = [
            s for s in agent_run.steps
            if s.status == AgentStepStatus.COMPLETED
        ]
        assert len(completed_steps) == 4

    @pytest.mark.asyncio
    async def test_pipeline_creates_training_data_source(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that pipeline creates a new training data source."""
        project = project_with_multiple_sources["project"]

        # Count data sources before
        before_count = db_session.query(DataSource).filter(
            DataSource.project_id == project.id
        ).count()

        agent_run = await run_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
            target_hint="churned",
            llm_client=mock_llm_client,
        )

        # Count data sources after
        after_count = db_session.query(DataSource).filter(
            DataSource.project_id == project.id
        ).count()

        assert after_count > before_count
        assert agent_run.result_json is not None
        assert "training_data_source_id" in agent_run.result_json

    @pytest.mark.asyncio
    async def test_pipeline_creates_dataset_spec(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that pipeline creates a DatasetSpec."""
        project = project_with_multiple_sources["project"]

        agent_run = await run_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
            target_hint="churned",
            llm_client=mock_llm_client,
        )

        # Check DatasetSpec was created
        dataset_spec = db_session.query(DatasetSpec).filter(
            DatasetSpec.project_id == project.id
        ).first()

        assert dataset_spec is not None
        assert dataset_spec.target_column == "churned"

    @pytest.mark.asyncio
    async def test_result_json_has_expected_fields(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that result_json contains expected fields."""
        project = project_with_multiple_sources["project"]

        agent_run = await run_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
            target_hint="churned",
            llm_client=mock_llm_client,
        )

        result = agent_run.result_json
        assert result is not None
        assert "training_data_source_id" in result
        assert "row_count" in result
        assert "column_count" in result
        assert "target_column" in result
        assert "feature_columns" in result

    @pytest.mark.asyncio
    async def test_training_data_source_is_valid(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that created training data source is valid."""
        project = project_with_multiple_sources["project"]

        agent_run = await run_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
            target_hint="churned",
            llm_client=mock_llm_client,
        )

        # Get the training data source
        training_ds_id = agent_run.result_json["training_data_source_id"]
        training_ds = db_session.query(DataSource).filter(
            DataSource.id == training_ds_id
        ).first()

        assert training_ds is not None
        assert training_ds.type == DataSourceType.FILE_UPLOAD
        assert training_ds.schema_summary is not None
        assert training_ds.schema_summary["row_count"] > 0

    @pytest.mark.asyncio
    async def test_step_outputs_flow_correctly(
        self, db_session, project_with_multiple_sources, mock_llm_client
    ):
        """Test that step outputs flow to next step inputs correctly."""
        project = project_with_multiple_sources["project"]

        agent_run = await run_data_architect_pipeline(
            db=db_session,
            project_id=project.id,
            target_hint="churned",
            llm_client=mock_llm_client,
        )

        # Get steps in order
        steps = sorted(agent_run.steps, key=lambda s: s.created_at)

        # Check inventory step output
        inv_step = steps[0]
        assert inv_step.step_type == AgentStepType.DATASET_INVENTORY
        assert inv_step.output_json is not None
        assert "data_source_profiles" in inv_step.output_json

        # Check relationship step used profiles
        rel_step = steps[1]
        assert rel_step.step_type == AgentStepType.RELATIONSHIP_DISCOVERY
        assert rel_step.input_json is not None
        assert "data_source_profiles" in rel_step.input_json

        # Check planning step used relationships
        plan_step = steps[2]
        assert plan_step.step_type == AgentStepType.TRAINING_DATASET_PLANNING
        assert plan_step.input_json is not None
        assert "relationships_summary" in plan_step.input_json

        # Check build step used spec
        build_step = steps[3]
        assert build_step.step_type == AgentStepType.TRAINING_DATASET_BUILD
        assert build_step.input_json is not None
        assert "training_dataset_spec" in build_step.input_json


# ============================================
# API Endpoint Tests
# ============================================

class TestDataArchitectEndpoint:
    """Tests for the /run-data-architect API endpoint.

    Note: These endpoint tests require actual LLM API calls because the pipeline
    runs synchronously in test mode. They are marked as integration tests.
    """

    @pytest.mark.skip(reason="Integration test - requires real LLM API key for background task")
    def test_endpoint_creates_pipeline(self, client, project_with_multiple_sources):
        """Test that endpoint creates the pipeline successfully."""
        from unittest.mock import patch

        project = project_with_multiple_sources["project"]

        # Mock the get_llm_provider_and_key to avoid requiring API keys in tests
        with patch("app.api.agent.get_llm_provider_and_key") as mock_get_llm:
            mock_get_llm.return_value = ("openai", "fake-api-key")

            response = client.post(
                f"/projects/{project.id}/agent/run-data-architect",
                json={"target_hint": "churned"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "agent_run_id" in data
        assert data["status"] == "pending"

    def test_endpoint_returns_404_for_missing_project(self, client):
        """Test that endpoint returns 404 for non-existent project."""
        fake_id = str(uuid.uuid4())

        response = client.post(
            f"/projects/{fake_id}/agent/run-data-architect",
            json={},
        )

        assert response.status_code == 404

    def test_endpoint_returns_400_for_empty_project(self, client, db_session):
        """Test that endpoint returns 400 when project has no data sources."""
        project = Project(name="Empty Project")
        db_session.add(project)
        db_session.commit()

        response = client.post(
            f"/projects/{project.id}/agent/run-data-architect",
            json={},
        )

        assert response.status_code == 400
        assert "no data sources" in response.json()["detail"]

    @pytest.mark.skip(reason="Integration test - requires real LLM API key for background task")
    def test_endpoint_works_without_target_hint(self, client, project_with_multiple_sources):
        """Test that endpoint works when target_hint is not provided."""
        from unittest.mock import patch

        project = project_with_multiple_sources["project"]

        # Mock the get_llm_provider_and_key to avoid requiring API keys in tests
        with patch("app.api.agent.get_llm_provider_and_key") as mock_get_llm:
            mock_get_llm.return_value = ("openai", "fake-api-key")

            response = client.post(
                f"/projects/{project.id}/agent/run-data-architect",
                json={},
            )

        assert response.status_code == 200

    @pytest.mark.skip(reason="Integration test - requires real LLM API key for background task")
    def test_created_run_has_4_steps(self, client, db_session, project_with_multiple_sources):
        """Test that created agent run has exactly 4 steps."""
        from unittest.mock import patch

        project = project_with_multiple_sources["project"]

        # Mock the get_llm_provider_and_key to avoid requiring API keys in tests
        with patch("app.api.agent.get_llm_provider_and_key") as mock_get_llm:
            mock_get_llm.return_value = ("openai", "fake-api-key")

            response = client.post(
                f"/projects/{project.id}/agent/run-data-architect",
                json={"target_hint": "churned"},
            )

        assert response.status_code == 200
        run_id = response.json()["agent_run_id"]

        # Check the run in database
        agent_run = db_session.query(AgentRun).filter(AgentRun.id == run_id).first()
        assert agent_run is not None
        assert len(agent_run.steps) == 4

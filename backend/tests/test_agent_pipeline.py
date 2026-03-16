"""Tests for Agent Setup Pipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from app.models import (
    AgentRun,
    AgentRunStatus,
    AgentStep,
    AgentStepType,
    AgentStepStatus,
    AgentStepLog,
    Project,
    DataSource,
)
from app.services.agent_executor import (
    create_setup_pipeline,
    run_setup_pipeline_for_project,
    get_agent_run_with_steps,
    list_agent_runs_for_project,
    SETUP_PIPELINE_STEPS,
)
from app.schemas.agent import (
    ProjectConfigSuggestion,
    DatasetSpecSuggestion,
    DatasetDesignSuggestion,
    DatasetVariant,
    ExperimentPlanSuggestion,
    ExperimentVariant,
)


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def mock_schema_analysis():
    """Create a mock schema analysis result."""
    return {
        "file_type": "csv",
        "row_count": 1000,
        "column_count": 5,
        "columns": [
            {"name": "id", "dtype": "int64", "inferred_type": "numeric", "null_percentage": 0.0, "unique_count": 1000},
            {"name": "age", "dtype": "int64", "inferred_type": "numeric", "null_percentage": 0.5, "unique_count": 60, "min": 18, "max": 80, "mean": 42.5},
            {"name": "income", "dtype": "float64", "inferred_type": "numeric", "null_percentage": 2.0, "unique_count": 900, "min": 20000, "max": 200000, "mean": 65000},
            {"name": "category", "dtype": "object", "inferred_type": "categorical", "null_percentage": 0.0, "unique_count": 5, "top_values": {"A": 300, "B": 250, "C": 200}},
            {"name": "target", "dtype": "object", "inferred_type": "categorical", "null_percentage": 0.0, "unique_count": 2, "top_values": {"yes": 600, "no": 400}},
        ],
    }


@pytest.fixture
def project_with_data_source(db_session, test_user, mock_schema_analysis):
    """Create a project with a data source that has schema analysis."""
    project = Project(
        name="Test Project",
        owner_id=test_user.id,
    )
    db_session.add(project)
    db_session.commit()

    data_source = DataSource(
        project_id=project.id,
        name="test_data.csv",
        type="file_upload",
        config_json={"file_path": "/tmp/test_data.csv"},
        schema_summary=mock_schema_analysis,
    )
    db_session.add(data_source)
    db_session.commit()

    return project, data_source


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that returns valid responses."""
    client = MagicMock()
    client.chat_json = AsyncMock()
    # Also mock chat_with_tools for the new tool-calling flow
    # By default, return content (no tool calls) to skip the tool loop
    client.chat_with_tools = AsyncMock(return_value={"content": {}})
    return client


# ============================================
# Create Setup Pipeline Tests
# ============================================

class TestCreateSetupPipeline:
    """Tests for create_setup_pipeline function."""

    def test_create_setup_pipeline_success(self, db_session, project_with_data_source):
        """Test successful pipeline creation."""
        project, data_source = project_with_data_source

        agent_run = create_setup_pipeline(
            db=db_session,
            project_id=project.id,
            data_source_id=data_source.id,
            description="I want to predict customer churn based on demographics",
        )

        # Verify run was created
        assert agent_run.id is not None
        assert agent_run.project_id == project.id
        assert agent_run.status == AgentRunStatus.PENDING
        assert agent_run.config_json["description"] == "I want to predict customer churn based on demographics"

        # Verify all 6 steps were created (including data_analysis)
        assert len(agent_run.steps) == 6

        # Verify step types and order
        step_types = [step.step_type for step in agent_run.steps]
        assert step_types == SETUP_PIPELINE_STEPS

        # Verify all steps are pending
        for step in agent_run.steps:
            assert step.status == AgentStepStatus.PENDING

        # Verify first step has proper input (now data_analysis is first)
        first_step = agent_run.steps[0]
        assert first_step.step_type == AgentStepType.DATA_ANALYSIS
        assert "description" in first_step.input_json
        assert "schema_summary" in first_step.input_json

    def test_create_setup_pipeline_with_time_budget(self, db_session, project_with_data_source):
        """Test pipeline creation with time budget."""
        project, data_source = project_with_data_source

        agent_run = create_setup_pipeline(
            db=db_session,
            project_id=project.id,
            data_source_id=data_source.id,
            description="Predict something",
            time_budget_minutes=30,
        )

        assert agent_run.config_json["time_budget_minutes"] == 30

        # Verify experiment design step has time budget
        exp_step = next(s for s in agent_run.steps if s.step_type == AgentStepType.EXPERIMENT_DESIGN)
        assert exp_step.input_json["time_budget_minutes"] == 30

    def test_create_setup_pipeline_project_not_found(self, db_session):
        """Test error when project not found."""
        fake_project_id = uuid.uuid4()
        fake_data_source_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Project not found"):
            create_setup_pipeline(
                db=db_session,
                project_id=fake_project_id,
                data_source_id=fake_data_source_id,
                description="Test description",
            )

    def test_create_setup_pipeline_data_source_not_found(self, db_session, project_with_data_source):
        """Test error when data source not found."""
        project, _ = project_with_data_source
        fake_data_source_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Data source not found"):
            create_setup_pipeline(
                db=db_session,
                project_id=project.id,
                data_source_id=fake_data_source_id,
                description="Test description",
            )

    def test_create_setup_pipeline_no_schema(self, db_session, test_user):
        """Test error when data source has no schema analysis."""
        project = Project(name="No Schema Project", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        data_source = DataSource(
            project_id=project.id,
            name="no_schema.csv",
            type="file_upload",
            schema_summary=None,  # No schema
        )
        db_session.add(data_source)
        db_session.commit()

        with pytest.raises(ValueError, match="no schema analysis"):
            create_setup_pipeline(
                db=db_session,
                project_id=project.id,
                data_source_id=data_source.id,
                description="Test description",
            )


# ============================================
# Run Setup Pipeline Tests
# ============================================

class TestRunSetupPipeline:
    """Tests for run_setup_pipeline_for_project function."""

    @pytest.mark.asyncio
    async def test_run_setup_pipeline_success(self, db_session, project_with_data_source, mock_llm_client):
        """Test successful pipeline execution with mocked LLM."""
        project, data_source = project_with_data_source

        # Set up mock for data_analysis step LLM call
        mock_llm_client.chat_json.return_value = {
            "suitability_score": 0.85,
            "can_proceed": True,
            "suggest_more_data": False,
            "target_column_suggestion": "target",
            "task_type_suggestion": "binary_classification",
            "key_observations": ["Data quality is good", "Target column is clearly defined"],
            "data_preparation_recommendations": ["Handle missing values"],
            "limitations": [],
            "natural_language_summary": "The dataset is suitable for ML experiments.",
        }

        # Mock all LLM responses
        # Also patch execute_with_tools to bypass tool-calling and return mocked content
        async def mock_execute_with_tools(client, messages, tool_executor, response_schema, step_logger=None):
            # Return the expected schema based on response_schema name
            schema_name = response_schema.__name__ if hasattr(response_schema, '__name__') else str(response_schema)
            if 'DatasetDesign' in schema_name:
                return {
                    "variants": [{"name": "baseline", "description": "Standard feature set",
                                  "feature_columns": ["age", "income", "category"],
                                  "excluded_columns": ["id", "target"],
                                  "exclusion_reasons": {"id": "ID column", "target": "Target column"},
                                  "train_test_split": "80_20", "preprocessing_strategy": "auto",
                                  "expected_tradeoff": "Balanced approach"}],
                    "recommended_variant": "baseline",
                    "reasoning": "Selected predictive features",
                    "warnings": [],
                }
            elif 'ExperimentPlan' in schema_name:
                return {
                    "variants": [
                        {"name": "quick", "description": "Fast iteration",
                         "automl_config": {"time_limit": 60, "presets": "medium_quality"},
                         "expected_tradeoff": "Faster training"},
                        {"name": "balanced", "description": "Good balance",
                         "automl_config": {"time_limit": 300, "presets": "good_quality"},
                         "expected_tradeoff": "Moderate time"},
                    ],
                    "recommended_variant": "balanced",
                    "reasoning": "Balanced approach",
                    "estimated_total_time_minutes": 10,
                }
            return {}

        with patch('app.services.agent_executor.generate_project_config') as mock_config, \
             patch('app.services.agent_executor.generate_dataset_design') as mock_design, \
             patch('app.services.agent_executor.generate_experiment_plan') as mock_plan, \
             patch('app.services.agent_executor.execute_with_tools', side_effect=mock_execute_with_tools):

            mock_config.return_value = ProjectConfigSuggestion(
                task_type="binary",
                target_column="target",
                primary_metric="roc_auc",
                reasoning="Binary classification for target prediction",
                confidence=0.95,
                suggested_name="Target Prediction",
            )

            mock_design.return_value = DatasetDesignSuggestion(
                variants=[
                    DatasetVariant(
                        name="baseline",
                        description="Standard feature set",
                        feature_columns=["age", "income", "category"],
                        excluded_columns=["id", "target"],
                        exclusion_reasons={"id": "ID column", "target": "Target column"},
                        train_test_split="80_20",
                        preprocessing_strategy="auto",
                        expected_tradeoff="Balanced approach",
                    ),
                ],
                recommended_variant="baseline",
                reasoning="Selected predictive features",
            )

            mock_plan.return_value = ExperimentPlanSuggestion(
                variants=[
                    ExperimentVariant(
                        name="quick",
                        description="Fast iteration",
                        automl_config={"time_limit": 60, "presets": "medium_quality"},
                        expected_tradeoff="Faster training",
                    ),
                    ExperimentVariant(
                        name="balanced",
                        description="Good balance",
                        automl_config={"time_limit": 300, "presets": "good_quality"},
                        expected_tradeoff="Moderate time",
                    ),
                ],
                recommended_variant="balanced",
                reasoning="Balanced approach",
                estimated_total_time_minutes=10,
            )

            agent_run = await run_setup_pipeline_for_project(
                db=db_session,
                project_id=project.id,
                data_source_id=data_source.id,
                description="I want to predict customer churn",
                llm_client=mock_llm_client,
            )

        # Verify run completed
        assert agent_run.status == AgentRunStatus.COMPLETED
        assert agent_run.result_json is not None

        # Verify all 5 steps completed
        assert len(agent_run.steps) == 6
        for step in agent_run.steps:
            assert step.status == AgentStepStatus.COMPLETED
            assert step.output_json is not None
            assert step.started_at is not None
            assert step.finished_at is not None

        # Verify step outputs were accumulated
        assert "task_type" in agent_run.result_json
        assert "feature_columns" in agent_run.result_json
        assert "variants" in agent_run.result_json

        # Verify logs were created for each step
        for step in agent_run.steps:
            logs = db_session.query(AgentStepLog).filter(
                AgentStepLog.agent_step_id == step.id
            ).all()
            assert len(logs) > 0


# ============================================
# Helper Function Tests
# ============================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_agent_run_with_steps(self, db_session, project_with_data_source):
        """Test getting an agent run with its steps."""
        project, data_source = project_with_data_source

        # Create a pipeline
        agent_run = create_setup_pipeline(
            db=db_session,
            project_id=project.id,
            data_source_id=data_source.id,
            description="Test description",
        )

        # Get with steps
        result = get_agent_run_with_steps(db_session, agent_run.id)

        assert result is not None
        assert result.id == agent_run.id
        assert len(result.steps) == 6

    def test_get_agent_run_not_found(self, db_session):
        """Test getting non-existent agent run."""
        fake_id = uuid.uuid4()
        result = get_agent_run_with_steps(db_session, fake_id)
        assert result is None

    def test_list_agent_runs_for_project(self, db_session, project_with_data_source):
        """Test listing agent runs for a project."""
        project, data_source = project_with_data_source

        # Create multiple pipelines
        for i in range(3):
            create_setup_pipeline(
                db=db_session,
                project_id=project.id,
                data_source_id=data_source.id,
                description=f"Test description {i}",
            )

        # List runs
        runs, total = list_agent_runs_for_project(db_session, project.id)

        assert total == 3
        assert len(runs) == 3

    def test_list_agent_runs_pagination(self, db_session, project_with_data_source):
        """Test pagination of agent runs list."""
        project, data_source = project_with_data_source

        # Create 5 pipelines
        for i in range(5):
            create_setup_pipeline(
                db=db_session,
                project_id=project.id,
                data_source_id=data_source.id,
                description=f"Test description {i}",
            )

        # Get first page
        runs, total = list_agent_runs_for_project(db_session, project.id, skip=0, limit=2)
        assert total == 5
        assert len(runs) == 2

        # Get second page
        runs, total = list_agent_runs_for_project(db_session, project.id, skip=2, limit=2)
        assert total == 5
        assert len(runs) == 2


# ============================================
# API Endpoint Tests
# ============================================

class TestAgentPipelineAPI:
    """Tests for agent pipeline API endpoints."""

    def test_run_setup_pipeline_endpoint_async(self, client, db_session, test_user, mock_schema_analysis):
        """Test POST /run-setup-pipeline with run_async=True."""
        # Create project and data source
        project = Project(name="API Test Project", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        data_source = DataSource(
            project_id=project.id,
            name="test.csv",
            type="file_upload",
            schema_summary=mock_schema_analysis,
        )
        db_session.add(data_source)
        db_session.commit()

        # Mock the LLM provider/key lookup to avoid API key requirement
        with patch('app.api.agent.get_llm_provider_and_key') as mock_llm:
            mock_llm.return_value = ("openai", "test-api-key")

            response = client.post(
                f"/projects/{project.id}/agent/run-setup-pipeline",
                json={
                    "data_source_id": str(data_source.id),
                    "description": "I want to predict customer churn",
                    "run_async": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "pending"

        # Verify run was created
        run_id = uuid.UUID(data["run_id"])
        agent_run = db_session.query(AgentRun).filter(AgentRun.id == run_id).first()
        assert agent_run is not None
        assert agent_run.status == AgentRunStatus.PENDING
        assert len(agent_run.steps) == 6

    def test_list_runs_endpoint(self, client, db_session, test_user, mock_schema_analysis):
        """Test GET /runs endpoint."""
        # Create project and data source
        project = Project(name="List Runs Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        data_source = DataSource(
            project_id=project.id,
            name="test.csv",
            type="file_upload",
            schema_summary=mock_schema_analysis,
        )
        db_session.add(data_source)
        db_session.commit()

        # Create a pipeline
        agent_run = create_setup_pipeline(
            db=db_session,
            project_id=project.id,
            data_source_id=data_source.id,
            description="Test description",
        )

        response = client.get(f"/projects/{project.id}/agent/runs")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == str(agent_run.id)

    def test_get_run_endpoint(self, client, db_session, test_user, mock_schema_analysis):
        """Test GET /runs/{run_id} endpoint."""
        # Create project and data source
        project = Project(name="Get Run Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        data_source = DataSource(
            project_id=project.id,
            name="test.csv",
            type="file_upload",
            schema_summary=mock_schema_analysis,
        )
        db_session.add(data_source)
        db_session.commit()

        # Create a pipeline
        agent_run = create_setup_pipeline(
            db=db_session,
            project_id=project.id,
            data_source_id=data_source.id,
            description="Test description",
        )

        response = client.get(f"/projects/{project.id}/agent/runs/{agent_run.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(agent_run.id)
        assert data["status"] == "pending"
        assert len(data["steps"]) == 6

        # Verify step types
        step_types = [step["step_type"] for step in data["steps"]]
        expected_types = [st.value for st in SETUP_PIPELINE_STEPS]
        assert step_types == expected_types

    def test_get_run_not_found(self, client, db_session, test_user):
        """Test GET /runs/{run_id} with non-existent run."""
        project = Project(name="404 Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        fake_run_id = uuid.uuid4()
        response = client.get(f"/projects/{project.id}/agent/runs/{fake_run_id}")

        assert response.status_code == 404

    def test_run_setup_pipeline_missing_description(self, client, db_session, test_user, mock_schema_analysis):
        """Test POST /run-setup-pipeline with missing description."""
        project = Project(name="Validation Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        data_source = DataSource(
            project_id=project.id,
            name="test.csv",
            type="file_upload",
            schema_summary=mock_schema_analysis,
        )
        db_session.add(data_source)
        db_session.commit()

        response = client.post(
            f"/projects/{project.id}/agent/run-setup-pipeline",
            json={
                "data_source_id": str(data_source.id),
                "description": "short",  # Too short (min 10 chars)
            },
        )

        assert response.status_code == 422  # Validation error


# ============================================
# Dataset Discovery Pipeline Tests
# ============================================

class TestDatasetDiscoveryPipeline:
    """Tests for the dataset discovery pipeline."""

    def test_create_dataset_discovery_pipeline(self, db_session, test_user):
        """Test creating a dataset discovery pipeline."""
        from app.services.agent_executor import create_dataset_discovery_pipeline

        # Create a project without data sources
        project = Project(name="Discovery Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        agent_run = create_dataset_discovery_pipeline(
            db=db_session,
            project_id=project.id,
            project_description="I want to predict used car prices in the US",
            constraints={"geography": "US", "allow_public_data": True},
        )

        assert agent_run is not None
        assert agent_run.project_id == project.id
        assert agent_run.status == AgentRunStatus.PENDING
        assert "Dataset Discovery" in agent_run.name

        # Verify single step was created
        steps = agent_run.steps
        assert len(steps) == 1
        assert steps[0].step_type == AgentStepType.DATASET_DISCOVERY
        assert steps[0].status == AgentStepStatus.PENDING

        # Verify step input
        input_json = steps[0].input_json
        assert input_json["project_description"] == "I want to predict used car prices in the US"
        assert input_json["constraints"]["geography"] == "US"

    def test_create_discovery_pipeline_project_not_found(self, db_session):
        """Test error when project doesn't exist."""
        from app.services.agent_executor import create_dataset_discovery_pipeline
        import uuid

        with pytest.raises(ValueError, match="Project not found"):
            create_dataset_discovery_pipeline(
                db=db_session,
                project_id=uuid.uuid4(),
                project_description="Test description",
            )

    @pytest.mark.asyncio
    async def test_run_dataset_discovery_pipeline(self, db_session, test_user, mock_llm_client):
        """Test running the full dataset discovery pipeline."""
        from app.services.agent_executor import run_dataset_discovery_pipeline

        # Create a project
        project = Project(name="Run Discovery Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        # Mock LLM response with discovered datasets
        mock_llm_client.chat_json.return_value = {
            "discovered_datasets": [
                {
                    "name": "US Used Car Prices Dataset",
                    "source_url": "https://kaggle.com/datasets/us-cars",
                    "schema_summary": {
                        "rows_estimate": 100000,
                        "columns": ["price", "mileage", "year", "make", "model"],
                        "target_candidate": "price",
                    },
                    "licensing": "CC BY 4.0",
                    "fit_for_purpose": "Perfect for US car price prediction.",
                },
            ],
            "natural_language_summary": "Found 1 excellent dataset for US car prices.",
        }

        agent_run = await run_dataset_discovery_pipeline(
            db=db_session,
            project_id=project.id,
            project_description="Predict used car prices in the US market",
            constraints={"geography": "US"},
            llm_client=mock_llm_client,
        )

        assert agent_run.status == AgentRunStatus.COMPLETED
        assert agent_run.result_json is not None
        assert "discovered_datasets" in agent_run.result_json
        assert len(agent_run.result_json["discovered_datasets"]) == 1


class TestApplyDiscoveredDatasetsEndpoint:
    """Tests for the apply-discovered-datasets endpoint.

    Note: The download tests require network access and are marked as integration tests.
    They are skipped by default and can be run with: pytest -m integration
    """

    @pytest.mark.skip(reason="Integration test - requires network access to download datasets")
    def test_apply_discovered_datasets_success(self, client, db_session, test_user):
        """Test successfully applying discovered datasets as data sources."""
        from app.models.data_source import DataSource, DataSourceType
        from unittest.mock import patch, MagicMock
        import tempfile
        import os

        # Create a project
        project = Project(name="Apply Datasets Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        # Create a completed discovery agent run with results
        agent_run = AgentRun(
            project_id=project.id,
            name="Discovery Run",
            status=AgentRunStatus.COMPLETED,
            result_json={
                "discovered_datasets": [
                    {
                        "name": "Dataset A",
                        "source_url": "https://example.com/dataset-a",
                        "schema_summary": {
                            "rows_estimate": 50000,
                            "columns": ["price", "mileage", "year"],
                            "target_candidate": "price",
                        },
                        "licensing": "MIT",
                        "fit_for_purpose": "Great for price prediction",
                    },
                    {
                        "name": "Dataset B",
                        "source_url": "https://example.com/dataset-b",
                        "schema_summary": {
                            "rows_estimate": 30000,
                            "columns": ["cost", "age", "type"],
                            "target_candidate": "cost",
                        },
                        "licensing": "Apache 2.0",
                        "fit_for_purpose": "Good supplementary data",
                    },
                ],
                "natural_language_summary": "Found 2 datasets.",
            },
        )
        db_session.add(agent_run)
        db_session.commit()

        # Create a temporary CSV file for the mock to return
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "dataset.csv")
        with open(temp_file, "w") as f:
            f.write("price,mileage,year\n10000,50000,2020\n15000,30000,2021\n")

        # Mock the DatasetDownloader at its source module
        with patch("app.services.dataset_downloader.DatasetDownloader") as mock_downloader_class:
            mock_instance = MagicMock()
            mock_downloader_class.return_value = mock_instance
            mock_instance.download.return_value = (
                temp_file,
                {"row_count": 50000, "columns": ["price", "mileage", "year"]},
            )

            # Apply first dataset only
            response = client.post(
                f"/projects/{project.id}/agent/apply-discovered-datasets/{agent_run.id}",
                json={"dataset_indices": [0]},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data_sources"]) == 1
        assert data["data_sources"][0]["name"] == "Dataset A"
        assert data["data_sources"][0]["source_url"] == "https://example.com/dataset-a"
        assert data["data_sources"][0]["licensing"] == "MIT"

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.skip(reason="Integration test - requires network access to download datasets")
    def test_apply_multiple_datasets(self, client, db_session, test_user):
        """Test applying multiple discovered datasets."""
        from unittest.mock import patch, MagicMock
        import tempfile
        import os

        # Create project and discovery run
        project = Project(name="Multi Apply Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        agent_run = AgentRun(
            project_id=project.id,
            name="Discovery Run",
            status=AgentRunStatus.COMPLETED,
            result_json={
                "discovered_datasets": [
                    {"name": "DS1", "source_url": "https://example.com/1", "licensing": "MIT", "fit_for_purpose": "Good"},
                    {"name": "DS2", "source_url": "https://example.com/2", "licensing": "CC0", "fit_for_purpose": "OK"},
                    {"name": "DS3", "source_url": "https://example.com/3", "licensing": "GPL", "fit_for_purpose": "Fair"},
                ],
            },
        )
        db_session.add(agent_run)
        db_session.commit()

        # Create a temporary CSV file for the mock to return
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "dataset.csv")
        with open(temp_file, "w") as f:
            f.write("col1,col2\n1,2\n3,4\n")

        # Mock the DatasetDownloader at its source module
        with patch("app.services.dataset_downloader.DatasetDownloader") as mock_downloader_class:
            mock_instance = MagicMock()
            mock_downloader_class.return_value = mock_instance
            mock_instance.download.return_value = (
                temp_file,
                {"row_count": 2, "columns": ["col1", "col2"]},
            )

            # Apply datasets 0 and 2
            response = client.post(
                f"/projects/{project.id}/agent/apply-discovered-datasets/{agent_run.id}",
                json={"dataset_indices": [0, 2]},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data_sources"]) == 2
        assert data["data_sources"][0]["name"] == "DS1"
        assert data["data_sources"][1]["name"] == "DS3"

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_apply_invalid_index(self, client, db_session, test_user):
        """Test error when applying invalid dataset index."""
        project = Project(name="Invalid Index Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        agent_run = AgentRun(
            project_id=project.id,
            name="Discovery Run",
            status=AgentRunStatus.COMPLETED,
            result_json={
                "discovered_datasets": [
                    {"name": "DS1", "source_url": "https://example.com/1", "licensing": "MIT"},
                ],
            },
        )
        db_session.add(agent_run)
        db_session.commit()

        response = client.post(
            f"/projects/{project.id}/agent/apply-discovered-datasets/{agent_run.id}",
            json={"dataset_indices": [0, 5]},  # Index 5 is invalid
        )

        assert response.status_code == 400
        assert "Invalid dataset indices" in response.json()["detail"]

    def test_apply_from_incomplete_run(self, client, db_session, test_user):
        """Test error when applying from an incomplete run."""
        project = Project(name="Incomplete Run Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        agent_run = AgentRun(
            project_id=project.id,
            name="Discovery Run",
            status=AgentRunStatus.RUNNING,  # Not completed
        )
        db_session.add(agent_run)
        db_session.commit()

        response = client.post(
            f"/projects/{project.id}/agent/apply-discovered-datasets/{agent_run.id}",
            json={"dataset_indices": [0]},
        )

        assert response.status_code == 400
        assert "must be completed" in response.json()["detail"]

    def test_apply_from_wrong_project(self, client, db_session, test_user):
        """Test error when run doesn't belong to project."""
        # Create two projects
        project1 = Project(name="Project 1", owner_id=test_user.id)
        project2 = Project(name="Project 2", owner_id=test_user.id)
        db_session.add_all([project1, project2])
        db_session.commit()

        # Create run for project1
        agent_run = AgentRun(
            project_id=project1.id,
            name="Discovery Run",
            status=AgentRunStatus.COMPLETED,
            result_json={"discovered_datasets": [{"name": "DS1", "source_url": "x", "licensing": "MIT"}]},
        )
        db_session.add(agent_run)
        db_session.commit()

        # Try to apply from project2
        response = client.post(
            f"/projects/{project2.id}/agent/apply-discovered-datasets/{agent_run.id}",
            json={"dataset_indices": [0]},
        )

        assert response.status_code == 400
        assert "does not belong to this project" in response.json()["detail"]

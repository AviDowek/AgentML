"""Tests for Agent API endpoints and services."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from app.schemas.agent import (
    SchemaSummary,
    ColumnSummary,
    ProjectConfigSuggestion,
    DatasetSpecSuggestion,
    ExperimentPlanSuggestion,
    ExperimentVariant,
)
from app.services.agent_service import (
    build_schema_summary,
    generate_project_config,
    generate_dataset_spec,
    generate_experiment_plan,
)


class TestBuildSchemaSummary:
    """Tests for build_schema_summary function."""

    def test_build_schema_summary_basic(self):
        """Test building schema summary from analysis results."""
        analysis_result = {
            "file_type": "csv",
            "row_count": 1000,
            "column_count": 3,
            "columns": [
                {
                    "name": "age",
                    "dtype": "int64",
                    "inferred_type": "numeric",
                    "null_percentage": 0.0,
                    "unique_count": 50,
                    "min": 18,
                    "max": 80,
                    "mean": 35.5,
                },
                {
                    "name": "name",
                    "dtype": "object",
                    "inferred_type": "text",
                    "null_percentage": 1.5,
                    "unique_count": 900,
                },
                {
                    "name": "status",
                    "dtype": "object",
                    "inferred_type": "categorical",
                    "null_percentage": 0.0,
                    "unique_count": 3,
                    "top_values": {"active": 500, "inactive": 300, "pending": 200},
                },
            ],
        }

        summary = build_schema_summary(
            data_source_id="123e4567-e89b-12d3-a456-426614174000",
            data_source_name="test_data.csv",
            analysis_result=analysis_result,
        )

        assert summary.data_source_name == "test_data.csv"
        assert summary.file_type == "csv"
        assert summary.row_count == 1000
        assert summary.column_count == 3
        assert len(summary.columns) == 3

        # Check numeric column
        age_col = next(c for c in summary.columns if c.name == "age")
        assert age_col.inferred_type == "numeric"
        assert age_col.min == 18
        assert age_col.max == 80

        # Check categorical column
        status_col = next(c for c in summary.columns if c.name == "status")
        assert status_col.inferred_type == "categorical"
        assert status_col.top_values == {"active": 500, "inactive": 300, "pending": 200}


class TestGenerateProjectConfig:
    """Tests for generate_project_config function."""

    @pytest.fixture
    def mock_schema(self):
        """Create a mock schema summary."""
        return SchemaSummary(
            data_source_id="123e4567-e89b-12d3-a456-426614174000",
            data_source_name="customers.csv",
            file_type="csv",
            row_count=5000,
            column_count=4,
            columns=[
                ColumnSummary(
                    name="customer_id",
                    dtype="int64",
                    inferred_type="numeric",
                    null_percentage=0.0,
                    unique_count=5000,
                ),
                ColumnSummary(
                    name="age",
                    dtype="int64",
                    inferred_type="numeric",
                    null_percentage=0.5,
                    unique_count=60,
                    min=18,
                    max=80,
                    mean=42.5,
                ),
                ColumnSummary(
                    name="income",
                    dtype="float64",
                    inferred_type="numeric",
                    null_percentage=2.0,
                    unique_count=4500,
                    min=20000,
                    max=200000,
                    mean=65000,
                ),
                ColumnSummary(
                    name="churned",
                    dtype="object",
                    inferred_type="categorical",
                    null_percentage=0.0,
                    unique_count=2,
                    top_values={"no": 4000, "yes": 1000},
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_generate_project_config_success(self, mock_schema):
        """Test successful project config generation."""
        mock_client = MagicMock()
        mock_client.chat_json = AsyncMock(return_value={
            "task_type": "binary",
            "target_column": "churned",
            "primary_metric": "roc_auc",
            "reasoning": "The 'churned' column has 2 unique values, making this a binary classification task.",
            "confidence": 0.95,
            "suggested_name": "Customer Churn Prediction",
        })

        result = await generate_project_config(
            client=mock_client,
            description="I want to predict which customers will churn",
            schema_summary=mock_schema,
        )

        assert result.task_type == "binary"
        assert result.target_column == "churned"
        assert result.primary_metric == "roc_auc"
        assert result.confidence >= 0.0 and result.confidence <= 1.0
        mock_client.chat_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_project_config_invalid_target(self, mock_schema):
        """Test that invalid target column triggers retry."""
        mock_client = MagicMock()
        # First call returns invalid target, second call returns valid
        mock_client.chat_json = AsyncMock(side_effect=[
            {
                "task_type": "binary",
                "target_column": "nonexistent_column",
                "primary_metric": "roc_auc",
                "reasoning": "Test",
                "confidence": 0.9,
            },
            {
                "task_type": "binary",
                "target_column": "churned",
                "primary_metric": "roc_auc",
                "reasoning": "Fixed response",
                "confidence": 0.9,
            },
        ])

        result = await generate_project_config(
            client=mock_client,
            description="Predict churn",
            schema_summary=mock_schema,
        )

        assert result.target_column == "churned"
        assert mock_client.chat_json.call_count == 2


class TestGenerateDatasetSpec:
    """Tests for generate_dataset_spec function."""

    @pytest.fixture
    def mock_schema(self):
        """Create a mock schema summary."""
        return SchemaSummary(
            data_source_id="123e4567-e89b-12d3-a456-426614174000",
            data_source_name="customers.csv",
            file_type="csv",
            row_count=5000,
            column_count=5,
            columns=[
                ColumnSummary(name="id", dtype="int64", inferred_type="numeric", null_percentage=0.0, unique_count=5000),
                ColumnSummary(name="age", dtype="int64", inferred_type="numeric", null_percentage=0.0, unique_count=60),
                ColumnSummary(name="income", dtype="float64", inferred_type="numeric", null_percentage=0.0, unique_count=4500),
                ColumnSummary(name="department", dtype="object", inferred_type="categorical", null_percentage=0.0, unique_count=10),
                ColumnSummary(name="churned", dtype="object", inferred_type="categorical", null_percentage=0.0, unique_count=2),
            ],
        )

    @pytest.mark.asyncio
    async def test_generate_dataset_spec_success(self, mock_schema):
        """Test successful dataset spec generation."""
        mock_client = MagicMock()
        mock_client.chat_json = AsyncMock(return_value={
            "feature_columns": ["age", "income", "department"],
            "excluded_columns": ["id", "churned"],
            "exclusion_reasons": {
                "id": "Unique identifier, not predictive",
                "churned": "Target column",
            },
            "suggested_filters": None,
            "reasoning": "Selected numeric and categorical features that could influence churn.",
            "warnings": [],
        })

        result = await generate_dataset_spec(
            client=mock_client,
            schema_summary=mock_schema,
            task_type="binary",
            target_column="churned",
        )

        assert "age" in result.feature_columns
        assert "income" in result.feature_columns
        assert "churned" not in result.feature_columns
        assert "id" in result.excluded_columns
        mock_client.chat_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_dataset_spec_removes_target(self, mock_schema):
        """Test that target column is removed from features if LLM includes it."""
        mock_client = MagicMock()
        mock_client.chat_json = AsyncMock(return_value={
            "feature_columns": ["age", "income", "churned"],  # Accidentally includes target
            "excluded_columns": ["id"],
            "exclusion_reasons": {"id": "Unique identifier"},
            "reasoning": "Test",
            "warnings": [],
        })

        result = await generate_dataset_spec(
            client=mock_client,
            schema_summary=mock_schema,
            task_type="binary",
            target_column="churned",
        )

        # Target should be automatically removed
        assert "churned" not in result.feature_columns
        assert "churned" in result.excluded_columns


class TestGenerateExperimentPlan:
    """Tests for generate_experiment_plan function."""

    @pytest.mark.asyncio
    async def test_generate_experiment_plan_success(self):
        """Test successful experiment plan generation."""
        mock_client = MagicMock()
        mock_client.chat_json = AsyncMock(return_value={
            "variants": [
                {
                    "name": "quick",
                    "description": "Fast iteration for initial testing",
                    "automl_config": {"time_limit": 60, "presets": "medium_quality"},
                    "expected_tradeoff": "Faster training, possibly lower accuracy",
                },
                {
                    "name": "balanced",
                    "description": "Good balance of speed and quality",
                    "automl_config": {"time_limit": 300, "presets": "good_quality"},
                    "expected_tradeoff": "Moderate training time, good accuracy",
                },
                {
                    "name": "high_quality",
                    "description": "Best possible model",
                    "automl_config": {"time_limit": 900, "presets": "best_quality"},
                    "expected_tradeoff": "Longer training, highest accuracy",
                },
            ],
            "recommended_variant": "balanced",
            "reasoning": "For a dataset of 5000 rows, the balanced variant offers good tradeoff.",
            "estimated_total_time_minutes": 25,
        })

        result = await generate_experiment_plan(
            client=mock_client,
            task_type="binary",
            target_column="churned",
            primary_metric="roc_auc",
            feature_columns=["age", "income", "department"],
            row_count=5000,
        )

        # 4 variants: quick_test (auto-added), quick, balanced, high_quality
        assert len(result.variants) == 4
        assert result.recommended_variant == "balanced"
        assert any(v.name == "quick_test" for v in result.variants)  # Auto-added
        assert any(v.name == "quick" for v in result.variants)
        assert any(v.name == "balanced" for v in result.variants)
        mock_client.chat_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_experiment_plan_adds_missing_config(self):
        """Test that missing config fields are added with defaults."""
        mock_client = MagicMock()
        mock_client.chat_json = AsyncMock(return_value={
            "variants": [
                {
                    "name": "test",
                    "description": "Test variant",
                    "automl_config": {},  # Missing time_limit and presets
                    "expected_tradeoff": "Test",
                },
            ],
            "recommended_variant": "test",
            "reasoning": "Test",
            "estimated_total_time_minutes": 5,
        })

        result = await generate_experiment_plan(
            client=mock_client,
            task_type="binary",
            target_column="target",
            primary_metric="accuracy",
            feature_columns=["f1", "f2"],
            row_count=1000,
        )

        # Find the "test" variant (quick_test is auto-added at index 0)
        test_variant = next(v for v in result.variants if v.name == "test")
        # Defaults should be added
        assert test_variant.automl_config["time_limit"] == 300
        assert test_variant.automl_config["presets"] == "medium_quality"


class TestAgentAPIEndpoints:
    """Tests for Agent API endpoints."""

    def test_schema_summary_endpoint(self, client, db_session, test_user):
        """Test GET /projects/{id}/agent/schema-summary/{data_source_id}."""
        from app.models.project import Project
        from app.models.data_source import DataSource

        # Create project
        project = Project(
            name="Test Project",
            owner_id=test_user.id,
        )
        db_session.add(project)
        db_session.commit()

        # Create data source with schema
        data_source = DataSource(
            project_id=project.id,
            name="test.csv",
            type="file_upload",
            schema_summary={
                "file_type": "csv",
                "row_count": 100,
                "column_count": 2,
                "columns": [
                    {"name": "col1", "dtype": "int64", "inferred_type": "numeric", "null_percentage": 0, "unique_count": 100},
                    {"name": "col2", "dtype": "object", "inferred_type": "text", "null_percentage": 0, "unique_count": 50},
                ],
            },
        )
        db_session.add(data_source)
        db_session.commit()

        response = client.get(f"/projects/{project.id}/agent/schema-summary/{data_source.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["data_source_name"] == "test.csv"
        assert data["row_count"] == 100
        assert len(data["columns"]) == 2

    def test_schema_summary_not_found(self, client, db_session, test_user):
        """Test schema summary with non-existent data source."""
        from app.models.project import Project

        project = Project(name="Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/projects/{project.id}/agent/schema-summary/{fake_id}")
        assert response.status_code == 404


class TestApplyAgentStepEndpoints:
    """Tests for applying agent step outputs to create resources."""

    def test_apply_dataset_spec_from_step_success(self, client, db_session, test_user):
        """Test POST /projects/{id}/agent/apply-dataset-spec-from-step/{step_id}."""
        from app.models import (
            Project, DataSource, AgentRun, AgentStep,
            AgentRunStatus, AgentStepType, AgentStepStatus,
        )
        from app.models.dataset_spec import DatasetSpec

        # Create project
        project = Project(name="Test Project", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        # Create data source
        data_source = DataSource(
            project_id=project.id,
            name="test.csv",
            type="file_upload",
            schema_summary={"file_type": "csv", "row_count": 100, "column_count": 3, "columns": []},
        )
        db_session.add(data_source)
        db_session.commit()

        # Create agent run
        agent_run = AgentRun(
            project_id=project.id,
            name="Test Run",
            status=AgentRunStatus.COMPLETED,
            config_json={"data_source_id": str(data_source.id)},
        )
        db_session.add(agent_run)
        db_session.commit()

        # Create dataset_design step with completed status and output
        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_DESIGN,
            status=AgentStepStatus.COMPLETED,
            input_json={"target_column": "churn"},
            output_json={
                "feature_columns": ["age", "income", "department"],
                "excluded_columns": ["id"],
                "exclusion_reasons": {"id": "Unique identifier"},
                "natural_language_summary": "Selected features for churn prediction.",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Call the endpoint
        response = client.post(
            f"/projects/{project.id}/agent/apply-dataset-spec-from-step/{agent_step.id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert "dataset_spec_id" in data
        assert "message" in data
        assert "3 features" in data["message"]

        # Verify DatasetSpec was created
        spec = db_session.query(DatasetSpec).filter_by(id=data["dataset_spec_id"]).first()
        assert spec is not None
        assert spec.target_column == "churn"
        assert spec.feature_columns == ["age", "income", "department"]
        assert spec.project_id == project.id

    def test_apply_dataset_spec_wrong_step_type(self, client, db_session, test_user):
        """Test that apply-dataset-spec fails for wrong step type."""
        from app.models import Project, AgentRun, AgentStep, AgentStepType, AgentStepStatus

        project = Project(name="Test Project", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        agent_run = AgentRun(project_id=project.id, name="Test Run")
        db_session.add(agent_run)
        db_session.commit()

        # Create wrong step type
        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PROBLEM_UNDERSTANDING,  # Wrong type
            status=AgentStepStatus.COMPLETED,
        )
        db_session.add(agent_step)
        db_session.commit()

        response = client.post(
            f"/projects/{project.id}/agent/apply-dataset-spec-from-step/{agent_step.id}"
        )

        assert response.status_code == 400
        assert "dataset_design" in response.json()["detail"]

    def test_apply_dataset_spec_step_not_completed(self, client, db_session, test_user):
        """Test that apply-dataset-spec fails for non-completed step."""
        from app.models import Project, AgentRun, AgentStep, AgentStepType, AgentStepStatus

        project = Project(name="Test Project", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        agent_run = AgentRun(project_id=project.id, name="Test Run")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_DESIGN,
            status=AgentStepStatus.RUNNING,  # Not completed
        )
        db_session.add(agent_step)
        db_session.commit()

        response = client.post(
            f"/projects/{project.id}/agent/apply-dataset-spec-from-step/{agent_step.id}"
        )

        assert response.status_code == 400
        assert "completed" in response.json()["detail"]

    def test_apply_experiment_plan_from_step_success(self, client, db_session, test_user):
        """Test POST /projects/{id}/agent/apply-experiment-plan-from-step/{step_id}."""
        from app.models import (
            Project, DataSource, AgentRun, AgentStep,
            AgentRunStatus, AgentStepType, AgentStepStatus,
        )
        from app.models.dataset_spec import DatasetSpec
        from app.models.experiment import Experiment

        # Create project
        project = Project(name="Test Project", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        # Create data source
        data_source = DataSource(
            project_id=project.id,
            name="test.csv",
            type="file_upload",
        )
        db_session.add(data_source)
        db_session.commit()

        # Create dataset spec (required for experiment)
        dataset_spec = DatasetSpec(
            project_id=project.id,
            name="Test Spec",
            target_column="churn",
            feature_columns=["age", "income"],
            data_sources_json=[str(data_source.id)],
        )
        db_session.add(dataset_spec)
        db_session.commit()

        # Create agent run
        agent_run = AgentRun(
            project_id=project.id,
            name="Test Run",
            status=AgentRunStatus.COMPLETED,
        )
        db_session.add(agent_run)
        db_session.commit()

        # Create experiment_design step with completed status and output
        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.EXPERIMENT_DESIGN,
            status=AgentStepStatus.COMPLETED,
            input_json={"primary_metric": "roc_auc"},
            output_json={
                "variants": [
                    {"name": "quick", "description": "Fast", "automl_config": {"time_limit": 60}},
                    {"name": "balanced", "description": "Good balance", "automl_config": {"time_limit": 300}},
                ],
                "recommended_variant": "balanced",
                "natural_language_summary": "Experiment plan with 2 variants.",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Call the endpoint
        response = client.post(
            f"/projects/{project.id}/agent/apply-experiment-plan-from-step/{agent_step.id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert "experiment_id" in data
        assert "message" in data
        assert "balanced" in data["message"]

        # Verify Experiment was created
        experiment = db_session.query(Experiment).filter_by(id=data["experiment_id"]).first()
        assert experiment is not None
        assert experiment.primary_metric == "roc_auc"
        assert experiment.dataset_spec_id == dataset_spec.id
        assert experiment.experiment_plan_json["variant_name"] == "balanced"

    def test_apply_experiment_plan_specific_variant(self, client, db_session, test_user):
        """Test applying experiment plan with specific variant selection."""
        from app.models import (
            Project, DataSource, AgentRun, AgentStep,
            AgentStepType, AgentStepStatus,
        )
        from app.models.dataset_spec import DatasetSpec
        from app.models.experiment import Experiment

        # Create project and dependencies
        project = Project(name="Test Project", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        data_source = DataSource(project_id=project.id, name="test.csv", type="file_upload")
        db_session.add(data_source)
        db_session.commit()

        dataset_spec = DatasetSpec(
            project_id=project.id,
            name="Test Spec",
            target_column="target",
            feature_columns=["f1"],
            data_sources_json=[str(data_source.id)],
        )
        db_session.add(dataset_spec)
        db_session.commit()

        agent_run = AgentRun(project_id=project.id, name="Test Run")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.EXPERIMENT_DESIGN,
            status=AgentStepStatus.COMPLETED,
            input_json={"primary_metric": "accuracy"},
            output_json={
                "variants": [
                    {"name": "quick", "description": "Fast", "automl_config": {}},
                    {"name": "balanced", "description": "Balanced", "automl_config": {}},
                    {"name": "high_quality", "description": "Best", "automl_config": {}},
                ],
                "recommended_variant": "balanced",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Call endpoint with specific variant
        response = client.post(
            f"/projects/{project.id}/agent/apply-experiment-plan-from-step/{agent_step.id}?variant=quick"
        )

        assert response.status_code == 200
        data = response.json()
        assert "quick" in data["message"]

        experiment = db_session.query(Experiment).filter_by(id=data["experiment_id"]).first()
        assert experiment.experiment_plan_json["variant_name"] == "quick"

    def test_apply_experiment_plan_no_dataset_spec(self, client, db_session, test_user):
        """Test that apply-experiment-plan fails when no dataset spec exists."""
        from app.models import Project, AgentRun, AgentStep, AgentStepType, AgentStepStatus

        project = Project(name="Test Project", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        agent_run = AgentRun(project_id=project.id, name="Test Run")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.EXPERIMENT_DESIGN,
            status=AgentStepStatus.COMPLETED,
            output_json={
                "variants": [{"name": "test", "description": "Test", "automl_config": {}}],
                "recommended_variant": "test",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        response = client.post(
            f"/projects/{project.id}/agent/apply-experiment-plan-from-step/{agent_step.id}"
        )

        assert response.status_code == 400
        assert "DatasetSpec" in response.json()["detail"]

    def test_apply_experiment_plan_invalid_variant(self, client, db_session, test_user):
        """Test that apply-experiment-plan fails for non-existent variant."""
        from app.models import (
            Project, DataSource, AgentRun, AgentStep,
            AgentStepType, AgentStepStatus,
        )
        from app.models.dataset_spec import DatasetSpec

        project = Project(name="Test Project", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        data_source = DataSource(project_id=project.id, name="test.csv", type="file_upload")
        db_session.add(data_source)
        db_session.commit()

        dataset_spec = DatasetSpec(
            project_id=project.id,
            name="Test Spec",
            target_column="target",
            feature_columns=["f1"],
            data_sources_json=[str(data_source.id)],
        )
        db_session.add(dataset_spec)
        db_session.commit()

        agent_run = AgentRun(project_id=project.id, name="Test Run")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.EXPERIMENT_DESIGN,
            status=AgentStepStatus.COMPLETED,
            output_json={
                "variants": [{"name": "quick", "description": "Fast", "automl_config": {}}],
                "recommended_variant": "quick",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        response = client.post(
            f"/projects/{project.id}/agent/apply-experiment-plan-from-step/{agent_step.id}?variant=nonexistent"
        )

        assert response.status_code == 400
        assert "nonexistent" in response.json()["detail"]
        assert "Available" in response.json()["detail"]

"""Tests for Agent Results Pipeline (experiment-level agents)."""
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
    DatasetSpec,
    Experiment,
    Trial,
    ModelVersion,
)
from app.services.agent_executor import (
    create_results_pipeline,
    run_results_pipeline_for_experiment,
    RESULTS_PIPELINE_STEPS,
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
            {"name": "age", "dtype": "int64", "inferred_type": "numeric", "null_percentage": 0.5, "unique_count": 60},
            {"name": "income", "dtype": "float64", "inferred_type": "numeric", "null_percentage": 2.0, "unique_count": 900},
            {"name": "category", "dtype": "object", "inferred_type": "categorical", "null_percentage": 0.0, "unique_count": 5},
            {"name": "target", "dtype": "object", "inferred_type": "categorical", "null_percentage": 0.0, "unique_count": 2},
        ],
    }


@pytest.fixture
def completed_experiment(db_session, test_user, mock_schema_analysis):
    """Create a completed experiment with trials and model versions."""
    # Create project
    project = Project(
        name="Results Test Project",
        owner_id=test_user.id,
    )
    db_session.add(project)
    db_session.commit()

    # Create data source
    data_source = DataSource(
        project_id=project.id,
        name="test_data.csv",
        type="file_upload",
        config_json={"file_path": "/tmp/test_data.csv"},
        schema_summary=mock_schema_analysis,
    )
    db_session.add(data_source)
    db_session.commit()

    # Create dataset spec
    dataset_spec = DatasetSpec(
        project_id=project.id,
        name="Test Dataset Spec",
        data_sources_json=[str(data_source.id)],
        target_column="target",
        feature_columns=["age", "income", "category"],
    )
    db_session.add(dataset_spec)
    db_session.commit()

    # Create experiment
    experiment = Experiment(
        project_id=project.id,
        dataset_spec_id=dataset_spec.id,
        name="Test Experiment",
        description="Test experiment for results pipeline",
        primary_metric="accuracy",
        metric_direction="maximize",
        status="completed",  # Must be completed to run results pipeline
        experiment_plan_json={
            "task_type": "binary",
            "target_column": "target",
        },
    )
    db_session.add(experiment)
    db_session.commit()

    # Create trial
    trial = Trial(
        experiment_id=experiment.id,
        variant_name="balanced",
        status="completed",
        metrics_json={"accuracy": 0.85, "f1": 0.82},
        best_model_ref="model_1",
    )
    db_session.add(trial)
    db_session.commit()

    # Create model versions
    model_version_1 = ModelVersion(
        project_id=project.id,
        experiment_id=experiment.id,
        trial_id=trial.id,
        name="LightGBM",
        model_type="lightgbm",
        status="trained",
        metrics_json={"accuracy": 0.85, "f1": 0.82, "roc_auc": 0.91},
        feature_importances_json={"age": 0.35, "income": 0.45, "category": 0.20},
    )
    model_version_2 = ModelVersion(
        project_id=project.id,
        experiment_id=experiment.id,
        trial_id=trial.id,
        name="XGBoost",
        model_type="xgboost",
        status="trained",
        metrics_json={"accuracy": 0.83, "f1": 0.80, "roc_auc": 0.89},
        feature_importances_json={"age": 0.30, "income": 0.50, "category": 0.20},
    )
    db_session.add(model_version_1)
    db_session.add(model_version_2)
    db_session.commit()

    return project, experiment, dataset_spec, [model_version_1, model_version_2]


@pytest.fixture
def pending_experiment(db_session, test_user, mock_schema_analysis):
    """Create an experiment that is not yet completed."""
    project = Project(
        name="Pending Experiment Project",
        owner_id=test_user.id,
    )
    db_session.add(project)
    db_session.commit()

    data_source = DataSource(
        project_id=project.id,
        name="test_data.csv",
        type="file_upload",
        schema_summary=mock_schema_analysis,
    )
    db_session.add(data_source)
    db_session.commit()

    dataset_spec = DatasetSpec(
        project_id=project.id,
        name="Test Dataset Spec",
        data_sources_json=[str(data_source.id)],
        target_column="target",
        feature_columns=["age", "income"],
    )
    db_session.add(dataset_spec)
    db_session.commit()

    experiment = Experiment(
        project_id=project.id,
        dataset_spec_id=dataset_spec.id,
        name="Pending Experiment",
        primary_metric="accuracy",
        status="running",  # Not completed
    )
    db_session.add(experiment)
    db_session.commit()

    return project, experiment


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that returns valid responses."""
    client = MagicMock()
    client.chat_json = AsyncMock()
    return client


# ============================================
# Create Results Pipeline Tests
# ============================================

class TestCreateResultsPipeline:
    """Tests for create_results_pipeline function."""

    def test_create_results_pipeline_success(self, db_session, completed_experiment):
        """Test successful results pipeline creation."""
        project, experiment, _, _ = completed_experiment

        agent_run = create_results_pipeline(
            db=db_session,
            experiment_id=experiment.id,
        )

        # Verify run was created
        assert agent_run.id is not None
        assert agent_run.project_id == project.id
        assert agent_run.experiment_id == experiment.id
        assert agent_run.status == AgentRunStatus.PENDING
        assert agent_run.config_json["experiment_id"] == str(experiment.id)

        # Verify all 2 steps were created
        assert len(agent_run.steps) == 2

        # Verify step types and order
        step_types = [step.step_type for step in agent_run.steps]
        assert step_types == RESULTS_PIPELINE_STEPS

        # Verify all steps are pending
        for step in agent_run.steps:
            assert step.status == AgentStepStatus.PENDING

        # Verify first step has proper input
        first_step = agent_run.steps[0]
        assert first_step.step_type == AgentStepType.RESULTS_INTERPRETATION
        assert "experiment_id" in first_step.input_json

        # Verify second step has proper input
        second_step = agent_run.steps[1]
        assert second_step.step_type == AgentStepType.RESULTS_CRITIC
        assert "experiment_id" in second_step.input_json

    def test_create_results_pipeline_experiment_not_found(self, db_session):
        """Test error when experiment not found."""
        fake_experiment_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Experiment not found"):
            create_results_pipeline(
                db=db_session,
                experiment_id=fake_experiment_id,
            )

    def test_create_results_pipeline_experiment_not_completed(self, db_session, pending_experiment):
        """Test error when experiment is not completed."""
        _, experiment = pending_experiment

        with pytest.raises(ValueError, match="Experiment must be completed"):
            create_results_pipeline(
                db=db_session,
                experiment_id=experiment.id,
            )


# ============================================
# Run Results Pipeline Tests
# ============================================

class TestRunResultsPipeline:
    """Tests for run_results_pipeline_for_experiment function."""

    @pytest.mark.asyncio
    async def test_run_results_pipeline_success(self, db_session, completed_experiment, mock_llm_client):
        """Test successful results pipeline execution with mocked LLM."""
        project, experiment, _, model_versions = completed_experiment

        # Mock LLM responses for both steps
        mock_llm_client.chat_json.side_effect = [
            # Results interpretation response
            {
                "results_summary": "The experiment trained 2 models. LightGBM achieved the best accuracy of 0.85.",
                "recommendation": {
                    "recommended_model_id": str(model_versions[0].id),
                    "reason": "LightGBM has the highest accuracy and ROC-AUC scores.",
                },
                "natural_language_summary": "Your experiment was successful! The LightGBM model performed best with 85% accuracy.",
            },
            # Results critic response
            {
                "critic_findings": {
                    "severity": "ok",
                    "issues": [],
                    "approved": True,
                },
                "natural_language_summary": "The experiment results look good. No significant issues detected.",
            },
        ]

        agent_run = await run_results_pipeline_for_experiment(
            db=db_session,
            experiment_id=experiment.id,
            llm_client=mock_llm_client,
        )

        # Verify run completed
        assert agent_run.status == AgentRunStatus.COMPLETED
        assert agent_run.result_json is not None

        # Verify all 2 steps completed
        assert len(agent_run.steps) == 2
        for step in agent_run.steps:
            assert step.status == AgentStepStatus.COMPLETED
            assert step.output_json is not None
            assert step.started_at is not None
            assert step.finished_at is not None

        # Verify step outputs
        interpretation_step = agent_run.steps[0]
        assert "results_summary" in interpretation_step.output_json
        assert "recommendation" in interpretation_step.output_json

        critic_step = agent_run.steps[1]
        assert "critic_findings" in critic_step.output_json

        # Verify LLM was called twice
        assert mock_llm_client.chat_json.call_count == 2

    @pytest.mark.asyncio
    async def test_run_results_pipeline_with_warnings(self, db_session, completed_experiment, mock_llm_client):
        """Test results pipeline with critic warnings."""
        project, experiment, _, model_versions = completed_experiment

        mock_llm_client.chat_json.side_effect = [
            # Results interpretation response
            {
                "results_summary": "The experiment trained 2 models.",
                "recommendation": {
                    "recommended_model_id": str(model_versions[0].id),
                    "reason": "Best performance.",
                },
                "natural_language_summary": "Model training complete.",
            },
            # Results critic response with warnings
            {
                "critic_findings": {
                    "severity": "warning",
                    "issues": [
                        {
                            "issue": "Small dataset size",
                            "severity": "warning",
                            "recommendation": "Consider collecting more data",
                        }
                    ],
                    "approved": True,
                },
                "natural_language_summary": "Results are acceptable but with minor concerns about dataset size.",
            },
        ]

        agent_run = await run_results_pipeline_for_experiment(
            db=db_session,
            experiment_id=experiment.id,
            llm_client=mock_llm_client,
        )

        # Verify run completed
        assert agent_run.status == AgentRunStatus.COMPLETED

        # Verify critic findings have warnings
        critic_step = agent_run.steps[1]
        findings = critic_step.output_json["critic_findings"]
        assert findings["severity"] == "warning"
        assert len(findings["issues"]) == 1
        assert findings["approved"] is True


# ============================================
# API Endpoint Tests
# ============================================

class TestResultsPipelineAPI:
    """Tests for results pipeline API endpoint."""

    def test_run_results_pipeline_endpoint_async(self, client, db_session, completed_experiment):
        """Test POST /experiments/{id}/agent/run-results-pipeline with run_async=True."""
        _, experiment, _, _ = completed_experiment

        # Mock the LLM provider/key lookup
        with patch('app.api.agent.get_llm_provider_and_key') as mock_llm:
            mock_llm.return_value = ("openai", "test-api-key")

            response = client.post(
                f"/experiments/{experiment.id}/agent/run-results-pipeline",
                json={"run_async": True},
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
        assert agent_run.experiment_id == experiment.id
        assert len(agent_run.steps) == 2

    def test_run_results_pipeline_experiment_not_found(self, client, db_session):
        """Test POST /experiments/{id}/agent/run-results-pipeline with non-existent experiment."""
        fake_experiment_id = uuid.uuid4()

        with patch('app.api.agent.get_llm_provider_and_key') as mock_llm:
            mock_llm.return_value = ("openai", "test-api-key")

            response = client.post(
                f"/experiments/{fake_experiment_id}/agent/run-results-pipeline",
                json={"run_async": True},
            )

        assert response.status_code == 404

    def test_run_results_pipeline_experiment_not_completed(self, client, db_session, pending_experiment):
        """Test POST /experiments/{id}/agent/run-results-pipeline with non-completed experiment."""
        _, experiment = pending_experiment

        with patch('app.api.agent.get_llm_provider_and_key') as mock_llm:
            mock_llm.return_value = ("openai", "test-api-key")

            response = client.post(
                f"/experiments/{experiment.id}/agent/run-results-pipeline",
                json={"run_async": True},
            )

        assert response.status_code == 400
        assert "must be completed" in response.json()["detail"]


# ============================================
# Step Handler Tests
# ============================================

class TestResultsInterpretationHandler:
    """Tests for the results interpretation step handler."""

    @pytest.mark.asyncio
    async def test_handler_builds_leaderboard(self, db_session, completed_experiment, mock_llm_client):
        """Test that the handler correctly builds the model leaderboard."""
        _, experiment, _, model_versions = completed_experiment

        mock_llm_client.chat_json.return_value = {
            "results_summary": "Test summary",
            "recommendation": {
                "recommended_model_id": str(model_versions[0].id),
                "reason": "Best model",
            },
            "natural_language_summary": "Test natural language summary",
        }

        # Create pipeline and run first step only
        agent_run = create_results_pipeline(db=db_session, experiment_id=experiment.id)

        from app.services.agent_executor import run_agent_step

        interpretation_step = agent_run.steps[0]
        await run_agent_step(db=db_session, step_id=interpretation_step.id, llm_client=mock_llm_client)

        # Refresh step
        db_session.refresh(interpretation_step)

        # Verify output includes leaderboard
        assert interpretation_step.status == AgentStepStatus.COMPLETED
        assert "leaderboard" in interpretation_step.output_json
        assert len(interpretation_step.output_json["leaderboard"]) == 2


class TestResultsCriticHandler:
    """Tests for the results critic step handler."""

    @pytest.mark.asyncio
    async def test_handler_detects_perfect_metrics(self, db_session, test_user, mock_schema_analysis, mock_llm_client):
        """Test that the handler detects suspiciously perfect metrics."""
        # Create project and experiment with perfect metrics
        project = Project(name="Perfect Metrics Test", owner_id=test_user.id)
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

        dataset_spec = DatasetSpec(
            project_id=project.id,
            name="Test Spec",
            data_sources_json=[str(data_source.id)],
            target_column="target",
            feature_columns=["age", "income"],
        )
        db_session.add(dataset_spec)
        db_session.commit()

        experiment = Experiment(
            project_id=project.id,
            dataset_spec_id=dataset_spec.id,
            name="Perfect Experiment",
            primary_metric="accuracy",
            status="completed",
        )
        db_session.add(experiment)
        db_session.commit()

        # Model with suspiciously perfect metrics
        model = ModelVersion(
            project_id=project.id,
            experiment_id=experiment.id,
            name="SuspiciousModel",
            model_type="unknown",
            status="trained",
            metrics_json={"accuracy": 1.0, "f1": 1.0},  # Perfect!
        )
        db_session.add(model)
        db_session.commit()

        # Mock responses for both steps
        mock_llm_client.chat_json.side_effect = [
            # Interpretation response
            {
                "results_summary": "Perfect accuracy",
                "recommendation": {"recommended_model_id": str(model.id), "reason": "Only model"},
                "natural_language_summary": "Perfect results",
            },
            # Critic response - should flag the perfect metrics
            {
                "critic_findings": {
                    "severity": "critical",
                    "issues": [
                        {
                            "issue": "Perfect accuracy score of 1.0 detected",
                            "severity": "critical",
                            "recommendation": "Check for data leakage",
                        }
                    ],
                    "approved": False,
                },
                "natural_language_summary": "Critical issues detected with perfect metrics.",
            },
        ]

        agent_run = await run_results_pipeline_for_experiment(
            db=db_session,
            experiment_id=experiment.id,
            llm_client=mock_llm_client,
        )

        # Verify critic found issues
        critic_step = agent_run.steps[1]
        findings = critic_step.output_json["critic_findings"]
        assert findings["severity"] == "critical"
        assert findings["approved"] is False

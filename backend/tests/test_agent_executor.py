"""Tests for Agent Step Executor and Handlers."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from app.models import (
    AgentRun,
    AgentRunStatus,
    AgentStep,
    AgentStepType,
    AgentStepStatus,
    AgentStepLog,
    LogMessageType,
    Project,
    DataSource,
)
from app.services.agent_executor import (
    StepLogger,
    append_step_log,
    log_step_thinking,
    log_step_hypothesis,
    log_step_action,
    log_step_summary,
    log_step_warning,
    log_step_error,
    log_step_info,
    handle_problem_understanding_step,
    handle_data_audit_step,
    handle_dataset_design_step,
    handle_experiment_design_step,
    handle_plan_critic_step,
    handle_dataset_discovery_step,
    handle_training_dataset_planning_step,
    handle_lab_notebook_summary_step,
    handle_robustness_audit_step,
    run_agent_step,
    run_agent_pipeline,
    build_agent_context_for_project,
    format_project_context_for_prompt,
)
from app.models.research_cycle import LabNotebookAuthorType, ResearchCycleStatus
from app.models.dataset_spec import DatasetSpec
from app.models.experiment import Trial, TrialStatus
from app.models.research_cycle import ResearchCycle, CycleExperiment, LabNotebookEntry
from app.models.experiment import Experiment
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
def mock_schema_summary():
    """Create a mock schema summary for testing."""
    return {
        "data_source_id": str(uuid.uuid4()),
        "data_source_name": "test_data.csv",
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
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.chat_json = AsyncMock()
    # Also mock chat_with_tools for the new tool-calling flow
    # By default, return content (no tool calls) to skip the tool loop
    client.chat_with_tools = AsyncMock(return_value={"content": {}})
    return client


# ============================================
# StepLogger Tests
# ============================================

class TestStepLogger:
    """Tests for StepLogger class."""

    def test_step_logger_creates_logs(self, db_session):
        """Test that StepLogger creates log entries."""
        agent_run = AgentRun(name="Logger Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        logger = StepLogger(db_session, agent_step.id)
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).all()

        assert len(logs) == 3
        assert logs[0].message_type == LogMessageType.INFO
        assert logs[1].message_type == LogMessageType.WARNING
        assert logs[2].message_type == LogMessageType.ERROR

    def test_step_logger_sequence_ordering(self, db_session):
        """Test that log sequences increment correctly."""
        agent_run = AgentRun(name="Sequence Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        logger = StepLogger(db_session, agent_step.id)
        logger.info("First")
        logger.info("Second")
        logger.info("Third")

        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).order_by(AgentStepLog.sequence).all()

        assert logs[0].sequence == 1
        assert logs[1].sequence == 2
        assert logs[2].sequence == 3

    def test_step_logger_with_metadata(self, db_session):
        """Test logging with metadata."""
        agent_run = AgentRun(name="Metadata Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        logger = StepLogger(db_session, agent_step.id)
        log = logger.info("With metadata", metadata={"key": "value", "count": 42})

        assert log.metadata_json is not None
        assert log.metadata_json["key"] == "value"
        assert log.metadata_json["count"] == 42


class TestAppendStepLog:
    """Tests for append_step_log function."""

    def test_append_step_log_creates_entry(self, db_session):
        """Test that append_step_log creates a log entry."""
        agent_run = AgentRun(name="Append Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = append_step_log(
            db=db_session,
            step_id=agent_step.id,
            message_type="info",
            message="Test message",
        )

        assert log.id is not None
        assert log.message == "Test message"
        assert log.message_type == LogMessageType.INFO


class TestStepLoggerRichTypes:
    """Tests for the new rich log types (thinking, hypothesis, action)."""

    def test_thinking_log_type(self, db_session):
        """Test that thinking() creates correct log type."""
        agent_run = AgentRun(name="Thinking Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        logger = StepLogger(db_session, agent_step.id)
        log = logger.thinking("Analyzing the data structure...")

        assert log.message_type == LogMessageType.THINKING
        assert log.message == "Analyzing the data structure..."

    def test_hypothesis_log_type(self, db_session):
        """Test that hypothesis() creates correct log type."""
        agent_run = AgentRun(name="Hypothesis Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        logger = StepLogger(db_session, agent_step.id)
        log = logger.hypothesis("The target column might be 'price'")

        assert log.message_type == LogMessageType.HYPOTHESIS
        assert log.message == "The target column might be 'price'"

    def test_action_log_type(self, db_session):
        """Test that action() creates correct log type."""
        agent_run = AgentRun(name="Action Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        logger = StepLogger(db_session, agent_step.id)
        log = logger.action("Now I will design features X, Y, Z")

        assert log.message_type == LogMessageType.ACTION
        assert log.message == "Now I will design features X, Y, Z"

    def test_all_rich_types_together(self, db_session):
        """Test using all rich log types in a realistic sequence."""
        agent_run = AgentRun(name="Rich Types Combined")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.EXPERIMENT_DESIGN,
        )
        db_session.add(agent_step)
        db_session.commit()

        logger = StepLogger(db_session, agent_step.id)

        # Simulate a realistic agent thinking process
        logger.action("Analyzing experiment design requirements")
        logger.thinking("Looking at the dataset structure...")
        logger.thinking("The data has 1000 rows and 10 features")
        logger.hypothesis("RandomForest might work well for this classification task")
        logger.hypothesis("XGBoost could also be effective given the data size")
        logger.action("Designing experiment variants")
        logger.summary("Created 3 experiment variants for comparison")

        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).order_by(AgentStepLog.sequence).all()

        assert len(logs) == 7
        assert logs[0].message_type == LogMessageType.ACTION
        assert logs[1].message_type == LogMessageType.THINKING
        assert logs[2].message_type == LogMessageType.THINKING
        assert logs[3].message_type == LogMessageType.HYPOTHESIS
        assert logs[4].message_type == LogMessageType.HYPOTHESIS
        assert logs[5].message_type == LogMessageType.ACTION
        assert logs[6].message_type == LogMessageType.SUMMARY


class TestStepLoggingHelperFunctions:
    """Tests for standalone step logging helper functions."""

    def test_log_step_thinking(self, db_session):
        """Test log_step_thinking helper function."""
        agent_run = AgentRun(name="Helper Thinking Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = log_step_thinking(db_session, agent_step.id, "Processing data...")
        assert log.message_type == LogMessageType.THINKING

    def test_log_step_hypothesis(self, db_session):
        """Test log_step_hypothesis helper function."""
        agent_run = AgentRun(name="Helper Hypothesis Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = log_step_hypothesis(db_session, agent_step.id, "This could be the cause")
        assert log.message_type == LogMessageType.HYPOTHESIS

    def test_log_step_action(self, db_session):
        """Test log_step_action helper function."""
        agent_run = AgentRun(name="Helper Action Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = log_step_action(db_session, agent_step.id, "Starting feature engineering")
        assert log.message_type == LogMessageType.ACTION

    def test_log_step_summary(self, db_session):
        """Test log_step_summary helper function."""
        agent_run = AgentRun(name="Helper Summary Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = log_step_summary(db_session, agent_step.id, "Analysis complete")
        assert log.message_type == LogMessageType.SUMMARY

    def test_log_step_warning(self, db_session):
        """Test log_step_warning helper function."""
        agent_run = AgentRun(name="Helper Warning Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = log_step_warning(db_session, agent_step.id, "Missing values detected")
        assert log.message_type == LogMessageType.WARNING

    def test_log_step_error(self, db_session):
        """Test log_step_error helper function."""
        agent_run = AgentRun(name="Helper Error Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = log_step_error(db_session, agent_step.id, "Failed to process")
        assert log.message_type == LogMessageType.ERROR

    def test_log_step_info(self, db_session):
        """Test log_step_info helper function."""
        agent_run = AgentRun(name="Helper Info Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = log_step_info(db_session, agent_step.id, "General info message")
        assert log.message_type == LogMessageType.INFO

    def test_helper_functions_with_metadata(self, db_session):
        """Test helper functions pass metadata correctly."""
        agent_run = AgentRun(name="Helper Metadata Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        metadata = {"model": "RandomForest", "confidence": 0.95}
        log = log_step_hypothesis(db_session, agent_step.id, "This model works", metadata)

        assert log.metadata_json == metadata
        assert log.metadata_json["model"] == "RandomForest"
        assert log.metadata_json["confidence"] == 0.95


# ============================================
# Handler Tests
# ============================================

class TestHandleProblemUnderstandingStep:
    """Tests for handle_problem_understanding_step."""

    @pytest.mark.asyncio
    async def test_problem_understanding_success(self, db_session, mock_llm_client, mock_schema_summary):
        """Test successful problem understanding."""
        agent_run = AgentRun(name="Problem Understanding Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PROBLEM_UNDERSTANDING,
            input_json={
                "description": "I want to predict customer churn",
                "schema_summary": mock_schema_summary,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "task_type": "binary",
            "target_column": "target",
            "primary_metric": "roc_auc",
            "reasoning": "Binary classification for churn prediction",
            "confidence": 0.95,
            "suggested_name": "Churn Prediction",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        with patch('app.services.agent_executor.generate_project_config') as mock_gen:
            mock_gen.return_value = ProjectConfigSuggestion(
                task_type="binary",
                target_column="target",
                primary_metric="roc_auc",
                reasoning="Binary classification for churn prediction",
                confidence=0.95,
                suggested_name="Churn Prediction",
            )

            output = await handle_problem_understanding_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

        assert output["task_type"] == "binary"
        assert output["target_column"] == "target"
        assert output["primary_metric"] == "roc_auc"
        assert output["confidence"] == 0.95

        # Verify logs were created
        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).all()
        assert len(logs) > 0

    @pytest.mark.asyncio
    async def test_problem_understanding_missing_description(self, db_session, mock_llm_client):
        """Test that missing description raises error."""
        agent_run = AgentRun(name="Missing Desc Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PROBLEM_UNDERSTANDING,
            input_json={},  # Missing description
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Missing 'description'"):
            await handle_problem_understanding_step(
                db_session, agent_step, step_logger, mock_llm_client
            )


class TestHandleDataAuditStep:
    """Tests for handle_data_audit_step."""

    @pytest.mark.asyncio
    async def test_data_audit_success(self, db_session, mock_llm_client, mock_schema_summary):
        """Test successful data audit."""
        agent_run = AgentRun(name="Data Audit Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
            input_json={
                "schema_summary": mock_schema_summary,
                "target_column": "target",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_data_audit_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        assert output["data_source_name"] == "test_data.csv"
        assert output["row_count"] == 1000
        assert output["column_count"] == 5
        assert "id" in output["potential_id_columns"]  # All unique values
        assert isinstance(output["issues"], list)
        assert isinstance(output["recommendations"], list)

    @pytest.mark.asyncio
    async def test_data_audit_detects_high_null(self, db_session, mock_llm_client):
        """Test that data audit detects high null columns."""
        schema = {
            "data_source_id": str(uuid.uuid4()),
            "data_source_name": "test.csv",
            "file_type": "csv",
            "row_count": 100,
            "column_count": 2,
            "columns": [
                {"name": "good_col", "dtype": "int64", "inferred_type": "numeric", "null_percentage": 5.0, "unique_count": 50},
                {"name": "bad_col", "dtype": "float64", "inferred_type": "numeric", "null_percentage": 45.0, "unique_count": 30},
            ],
        }

        agent_run = AgentRun(name="High Null Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
            input_json={"schema_summary": schema},
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_data_audit_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        assert "bad_col" in output["high_null_columns"]
        assert "good_col" not in output["high_null_columns"]


class TestHandleDatasetDesignStep:
    """Tests for handle_dataset_design_step."""

    @pytest.mark.asyncio
    async def test_dataset_design_success(self, db_session, mock_llm_client, mock_schema_summary):
        """Test successful dataset design."""
        agent_run = AgentRun(name="Dataset Design Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_DESIGN,
            input_json={
                "schema_summary": mock_schema_summary,
                "task_type": "binary",
                "target_column": "target",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with patch('app.services.agent_executor.generate_dataset_design') as mock_gen:
            mock_gen.return_value = DatasetDesignSuggestion(
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

            output = await handle_dataset_design_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

        # Check output has variants structure
        assert "variants" in output
        assert len(output["variants"]) == 1
        baseline = output["variants"][0]
        assert "age" in baseline["feature_columns"]
        assert "income" in baseline["feature_columns"]
        assert "target" in baseline["excluded_columns"]
        assert "id" in baseline["excluded_columns"]


class TestHandleExperimentDesignStep:
    """Tests for handle_experiment_design_step."""

    @pytest.mark.asyncio
    async def test_experiment_design_success(self, db_session, mock_llm_client):
        """Test successful experiment design."""
        agent_run = AgentRun(name="Experiment Design Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.EXPERIMENT_DESIGN,
            input_json={
                "task_type": "binary",
                "target_column": "target",
                "primary_metric": "roc_auc",
                "feature_columns": ["age", "income", "category"],
                "row_count": 1000,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with patch('app.services.agent_executor.generate_experiment_plan') as mock_gen:
            mock_gen.return_value = ExperimentPlanSuggestion(
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
                reasoning="Balanced approach for this dataset size",
                estimated_total_time_minutes=10,
            )

            output = await handle_experiment_design_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

        assert len(output["variants"]) == 2
        assert output["recommended_variant"] == "balanced"
        assert output["estimated_total_time_minutes"] == 10


class TestHandlePlanCriticStep:
    """Tests for handle_plan_critic_step."""

    @pytest.mark.asyncio
    async def test_plan_critic_approves_valid_plan(self, db_session, mock_llm_client):
        """Test that plan critic approves a valid plan."""
        agent_run = AgentRun(name="Plan Critic Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PLAN_CRITIC,
            input_json={
                "task_type": "binary",
                "target_column": "target",
                "feature_columns": ["age", "income", "category"],
                "variants": [
                    {"name": "quick", "automl_config": {"time_limit": 60}},
                    {"name": "balanced", "automl_config": {"time_limit": 300}},
                ],
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_plan_critic_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        assert output["approved"] is True
        assert output["status"] == "approved"
        assert output["feature_count"] == 3
        assert output["variant_count"] == 2

    @pytest.mark.asyncio
    async def test_plan_critic_rejects_empty_features(self, db_session, mock_llm_client):
        """Test that plan critic rejects plan with no features."""
        agent_run = AgentRun(name="Empty Features Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PLAN_CRITIC,
            input_json={
                "task_type": "binary",
                "target_column": "target",
                "feature_columns": [],  # No features!
                "variants": [{"name": "test", "automl_config": {"time_limit": 60}}],
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_plan_critic_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        assert output["approved"] is False
        assert any("No features" in issue for issue in output["issues"])


# ============================================
# Run Agent Step Tests
# ============================================

class TestRunAgentStep:
    """Tests for run_agent_step function."""

    @pytest.mark.asyncio
    async def test_run_agent_step_success(self, db_session, mock_llm_client, mock_schema_summary):
        """Test successful step execution."""
        agent_run = AgentRun(name="Run Step Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
            status=AgentStepStatus.PENDING,
            input_json={"schema_summary": mock_schema_summary},
        )
        db_session.add(agent_step)
        db_session.commit()

        # Run the step
        result = await run_agent_step(db_session, agent_step.id, mock_llm_client)

        # Verify step is completed
        assert result.status == AgentStepStatus.COMPLETED
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.output_json is not None

        # Verify logs exist
        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).all()
        assert len(logs) > 0

    @pytest.mark.asyncio
    async def test_run_agent_step_failure(self, db_session, mock_llm_client):
        """Test step execution failure handling."""
        agent_run = AgentRun(name="Failure Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
            status=AgentStepStatus.PENDING,
            input_json={},  # Missing schema_summary will cause failure
        )
        db_session.add(agent_step)
        db_session.commit()

        # Run the step (should fail)
        with pytest.raises(ValueError):
            await run_agent_step(db_session, agent_step.id, mock_llm_client)

        # Refresh and verify failure state
        db_session.refresh(agent_step)
        assert agent_step.status == AgentStepStatus.FAILED
        assert agent_step.error_message is not None
        assert agent_step.retry_count == 1

    @pytest.mark.asyncio
    async def test_run_agent_step_not_found(self, db_session, mock_llm_client):
        """Test error when step not found."""
        fake_id = uuid.uuid4()

        with pytest.raises(ValueError, match="Agent step not found"):
            await run_agent_step(db_session, fake_id, mock_llm_client)


# ============================================
# Run Agent Pipeline Tests
# ============================================

class TestRunAgentPipeline:
    """Tests for run_agent_pipeline function."""

    @pytest.mark.asyncio
    async def test_run_pipeline_success(self, db_session, mock_llm_client, mock_schema_summary):
        """Test successful pipeline execution."""
        agent_run = AgentRun(name="Pipeline Test")
        db_session.add(agent_run)
        db_session.commit()

        # Create multiple steps
        step1 = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
            status=AgentStepStatus.PENDING,
            input_json={"schema_summary": mock_schema_summary},
        )
        step2 = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PLAN_CRITIC,
            status=AgentStepStatus.PENDING,
            input_json={
                "feature_columns": ["age", "income"],
                "variants": [{"name": "test", "automl_config": {"time_limit": 60}}],
            },
        )
        db_session.add_all([step1, step2])
        db_session.commit()

        # Run the pipeline
        result = await run_agent_pipeline(db_session, agent_run.id, mock_llm_client)

        # Verify run is completed
        assert result.status == AgentRunStatus.COMPLETED
        assert result.result_json is not None

        # Verify all steps are completed
        db_session.refresh(step1)
        db_session.refresh(step2)
        assert step1.status == AgentStepStatus.COMPLETED
        assert step2.status == AgentStepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_pipeline_stops_on_failure(self, db_session, mock_llm_client):
        """Test that pipeline stops when a step fails."""
        agent_run = AgentRun(name="Failure Pipeline Test")
        db_session.add(agent_run)
        db_session.commit()

        # First step will fail (missing input)
        step1 = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
            status=AgentStepStatus.PENDING,
            input_json={},  # Will fail
        )
        step2 = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PLAN_CRITIC,
            status=AgentStepStatus.PENDING,
            input_json={"feature_columns": ["a"]},
        )
        db_session.add_all([step1, step2])
        db_session.commit()

        # Run the pipeline (should fail on first step)
        with pytest.raises(ValueError):
            await run_agent_pipeline(db_session, agent_run.id, mock_llm_client)

        # Verify run is failed
        db_session.refresh(agent_run)
        assert agent_run.status == AgentRunStatus.FAILED
        assert agent_run.error_message is not None

        # First step failed, second should still be pending
        db_session.refresh(step1)
        db_session.refresh(step2)
        assert step1.status == AgentStepStatus.FAILED
        assert step2.status == AgentStepStatus.PENDING


# ============================================
# Dataset Discovery Step Tests
# ============================================

class TestHandleDatasetDiscoveryStep:
    """Tests for handle_dataset_discovery_step."""

    @pytest.mark.asyncio
    async def test_dataset_discovery_success(self, db_session, mock_llm_client):
        """Test successful dataset discovery with mocked LLM response."""
        agent_run = AgentRun(name="Dataset Discovery Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_DISCOVERY,
            input_json={
                "project_description": "I want to predict used car prices in the US",
                "constraints": {
                    "geography": "US",
                    "allow_public_data": True,
                },
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response with discovered datasets
        mock_llm_client.chat_json.return_value = {
            "discovered_datasets": [
                {
                    "name": "US Used Car Listings 2019-2024",
                    "source_url": "https://www.kaggle.com/datasets/example/us-used-cars",
                    "schema_summary": {
                        "rows_estimate": 150000,
                        "columns": ["price", "mileage", "year", "make", "model", "state"],
                        "target_candidate": "price",
                    },
                    "licensing": "CC BY 4.0",
                    "fit_for_purpose": "Good match for US car price prediction.",
                },
                {
                    "name": "Kaggle UK Used Cars",
                    "source_url": "https://www.kaggle.com/datasets/example/uk-used-cars",
                    "schema_summary": {
                        "rows_estimate": 80000,
                        "columns": ["price", "mileage", "year", "manufacturer", "model"],
                        "target_candidate": "price",
                    },
                    "licensing": "CC BY-SA",
                    "fit_for_purpose": "Limited to UK; might be useful as supplementary data.",
                },
            ],
            "natural_language_summary": "I found 2 promising datasets. The first one is most relevant for your US pricing goal.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_dataset_discovery_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify output structure
        assert "discovered_datasets" in output
        assert "natural_language_summary" in output
        assert len(output["discovered_datasets"]) == 2

        # Verify each dataset has required fields
        for ds in output["discovered_datasets"]:
            assert "name" in ds
            assert "source_url" in ds
            assert "schema_summary" in ds
            assert "licensing" in ds
            assert "fit_for_purpose" in ds

            # Verify schema_summary structure
            schema = ds["schema_summary"]
            assert "rows_estimate" in schema
            assert "columns" in schema
            assert "target_candidate" in schema

        # Verify first dataset details
        first_ds = output["discovered_datasets"][0]
        assert first_ds["name"] == "US Used Car Listings 2019-2024"
        assert first_ds["licensing"] == "CC BY 4.0"
        assert first_ds["schema_summary"]["target_candidate"] == "price"

        # Verify logs were created
        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).all()
        assert len(logs) > 0

    @pytest.mark.asyncio
    async def test_dataset_discovery_missing_description(self, db_session, mock_llm_client):
        """Test that missing project_description raises error."""
        agent_run = AgentRun(name="Missing Description Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_DISCOVERY,
            input_json={},  # Missing project_description
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Missing 'project_description'"):
            await handle_dataset_discovery_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

    @pytest.mark.asyncio
    async def test_dataset_discovery_with_constraints(self, db_session, mock_llm_client):
        """Test dataset discovery with geographic constraints."""
        agent_run = AgentRun(name="Constrained Discovery Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_DISCOVERY,
            input_json={
                "project_description": "Predict house prices",
                "constraints": {
                    "geography": "Europe",
                    "allow_public_data": True,
                },
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "discovered_datasets": [
                {
                    "name": "European Housing Dataset",
                    "source_url": "https://example.com/eu-housing",
                    "schema_summary": {
                        "rows_estimate": 50000,
                        "columns": ["price", "size_sqm", "rooms", "location"],
                        "target_candidate": "price",
                    },
                    "licensing": "Public Domain",
                    "fit_for_purpose": "Good for European housing price prediction.",
                },
            ],
            "natural_language_summary": "Found 1 European housing dataset.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_dataset_discovery_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        assert len(output["discovered_datasets"]) == 1
        assert output["discovered_datasets"][0]["licensing"] == "Public Domain"

        # Verify thought log contains geographic constraint
        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.THOUGHT,
        ).all()
        thought_messages = [log.message for log in logs]
        assert any("Europe" in msg for msg in thought_messages)

    @pytest.mark.asyncio
    async def test_dataset_discovery_no_results(self, db_session, mock_llm_client):
        """Test dataset discovery when no datasets are found."""
        agent_run = AgentRun(name="No Results Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_DISCOVERY,
            input_json={
                "project_description": "Predict highly specialized niche data",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response with no datasets
        mock_llm_client.chat_json.return_value = {
            "discovered_datasets": [],
            "natural_language_summary": "No suitable public datasets found for this specialized use case.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_dataset_discovery_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        assert len(output["discovered_datasets"]) == 0
        assert "No suitable" in output["natural_language_summary"]

        # Verify warning log was created
        warning_logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.WARNING,
        ).all()
        assert any("No suitable datasets found" in log.message for log in warning_logs)

    @pytest.mark.asyncio
    async def test_dataset_discovery_licensing_always_present(self, db_session, mock_llm_client):
        """Test that licensing is always present in discovered datasets."""
        agent_run = AgentRun(name="Licensing Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_DISCOVERY,
            input_json={
                "project_description": "Customer churn prediction",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock response with various licensing types
        mock_llm_client.chat_json.return_value = {
            "discovered_datasets": [
                {
                    "name": "Dataset A",
                    "source_url": "https://example.com/a",
                    "schema_summary": {"rows_estimate": 1000, "columns": ["col1"], "target_candidate": "col1"},
                    "licensing": "MIT",
                    "fit_for_purpose": "Good",
                },
                {
                    "name": "Dataset B",
                    "source_url": "https://example.com/b",
                    "schema_summary": {"rows_estimate": 2000, "columns": ["col2"], "target_candidate": "col2"},
                    "licensing": "Unknown - verify before use",
                    "fit_for_purpose": "Okay",
                },
                {
                    "name": "Dataset C",
                    "source_url": "https://example.com/c",
                    "schema_summary": {"rows_estimate": 3000, "columns": ["col3"], "target_candidate": "col3"},
                    "licensing": "CC BY 4.0",
                    "fit_for_purpose": "Excellent",
                },
            ],
            "natural_language_summary": "Found 3 datasets with various licenses.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_dataset_discovery_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify all datasets have licensing field
        for ds in output["discovered_datasets"]:
            assert "licensing" in ds
            assert ds["licensing"] is not None
            assert len(ds["licensing"]) > 0


# ============================================
# Training Dataset Planning Step Tests
# ============================================

class TestHandleTrainingDatasetPlanningStep:
    """Tests for handle_training_dataset_planning_step."""

    @pytest.fixture
    def fake_profiles(self):
        """Create fake data source profiles for testing."""
        return [
            {
                "source_id": str(uuid.uuid4()),
                "source_name": "customers.csv",
                "estimated_row_count": 10000,
                "column_count": 5,
                "columns": [
                    {"name": "customer_id", "inferred_type": "id", "distinct_count": 10000},
                    {"name": "name", "inferred_type": "text", "distinct_count": 9500},
                    {"name": "signup_date", "inferred_type": "datetime", "distinct_count": 3000},
                    {"name": "churned", "inferred_type": "boolean", "distinct_count": 2},
                    {"name": "segment", "inferred_type": "categorical", "distinct_count": 5},
                ],
            },
            {
                "source_id": str(uuid.uuid4()),
                "source_name": "transactions.csv",
                "estimated_row_count": 100000,
                "column_count": 5,
                "columns": [
                    {"name": "transaction_id", "inferred_type": "id", "distinct_count": 100000},
                    {"name": "customer_id", "inferred_type": "id", "distinct_count": 10000},
                    {"name": "amount", "inferred_type": "numeric", "distinct_count": 5000},
                    {"name": "transaction_date", "inferred_type": "datetime", "distinct_count": 1000},
                    {"name": "category", "inferred_type": "categorical", "distinct_count": 20},
                ],
            },
        ]

    @pytest.fixture
    def fake_relationships_summary(self):
        """Create fake relationship discovery summary for testing."""
        return {
            "tables": [
                {
                    "table_name": "customers",
                    "source_name": "customers.csv",
                    "row_count": 10000,
                    "column_count": 5,
                    "id_columns": [{"name": "customer_id", "distinct_ratio": 1.0}],
                    "target_columns": [{"name": "churned"}],
                    "has_obvious_id": True,
                    "has_potential_target": True,
                },
                {
                    "table_name": "transactions",
                    "source_name": "transactions.csv",
                    "row_count": 100000,
                    "column_count": 5,
                    "id_columns": [
                        {"name": "transaction_id", "distinct_ratio": 1.0},
                        {"name": "customer_id", "distinct_ratio": 0.1},
                    ],
                    "target_columns": [],
                    "has_obvious_id": True,
                    "has_potential_target": False,
                },
            ],
            "relationships": [
                {
                    "from_table": "customers",
                    "to_table": "transactions",
                    "from_column": "customer_id",
                    "to_column": "customer_id",
                    "cardinality": "one_to_many",
                    "match_type": "exact_id",
                    "confidence": 0.95,
                },
            ],
            "base_table_candidates": [
                {
                    "table": "customers",
                    "score": 0.85,
                    "reasons": [
                        "Has obvious ID column",
                        "Has potential target column",
                        "Entity-like table name",
                    ],
                    "target_columns": ["churned"],
                },
                {
                    "table": "transactions",
                    "score": 0.45,
                    "reasons": ["Has obvious ID column", "High row count"],
                    "target_columns": [],
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_training_dataset_planning_success(
        self, db_session, mock_llm_client, fake_profiles, fake_relationships_summary
    ):
        """Test successful training dataset planning with mocked LLM."""
        agent_run = AgentRun(name="Training Dataset Planning Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_PLANNING,
            input_json={
                "project_description": "I want to predict which customers will churn",
                "target_hint": "churned",
                "data_source_profiles": fake_profiles,
                "relationships_summary": fake_relationships_summary,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "training_dataset_spec": {
                "base_table": "customers",
                "base_filters": [
                    {"column": "signup_date", "operator": ">=", "value": "2020-01-01"}
                ],
                "target_definition": {
                    "table": "customers",
                    "column": "churned",
                    "join_key": None,
                    "label_window_days": 365,
                },
                "join_plan": [
                    {
                        "from_table": "customers",
                        "to_table": "transactions",
                        "left_key": "customer_id",
                        "right_key": "customer_id",
                        "relationship": "one_to_many",
                        "aggregation": {
                            "window_days": 90,
                            "features": [
                                {"name": "total_spend_90d", "agg": "sum", "column": "amount"},
                                {"name": "tx_count_90d", "agg": "count", "column": "*"},
                            ],
                        },
                    }
                ],
                "excluded_tables": [],
                "excluded_columns": ["customer_id", "transaction_id"],
            },
            "natural_language_summary": "I will use customers as the base table since each row represents a customer and it contains the churned target column. I'll join with transactions to create aggregated features like total spend and transaction count over the last 90 days.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_training_dataset_planning_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify output structure
        assert "training_dataset_spec" in output
        assert "natural_language_summary" in output

        spec = output["training_dataset_spec"]
        assert spec["base_table"] == "customers"
        assert spec["target_definition"]["column"] == "churned"
        assert len(spec["join_plan"]) == 1
        assert spec["join_plan"][0]["to_table"] == "transactions"
        assert spec["join_plan"][0]["aggregation"]["window_days"] == 90

        # Verify logs were created
        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).all()
        assert len(logs) > 0

        # Verify specific log messages exist
        log_messages = [log.message for log in logs]
        assert any("Evaluating" in msg and "tables" in msg for msg in log_messages)
        assert any("customers" in msg.lower() for msg in log_messages)

    @pytest.mark.asyncio
    async def test_training_dataset_planning_logs_written(
        self, db_session, mock_llm_client, fake_profiles, fake_relationships_summary
    ):
        """Test that planning step writes appropriate logs."""
        agent_run = AgentRun(name="Logging Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_PLANNING,
            input_json={
                "project_description": "Predict customer churn",
                "data_source_profiles": fake_profiles,
                "relationships_summary": fake_relationships_summary,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "training_dataset_spec": {
                "base_table": "customers",
                "base_filters": [],
                "target_definition": {
                    "table": "customers",
                    "column": "churned",
                },
                "join_plan": [],
                "excluded_tables": [],
                "excluded_columns": [],
            },
            "natural_language_summary": "Using customers as base table.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        await handle_training_dataset_planning_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify info logs were created
        info_logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.INFO,
        ).all()
        assert len(info_logs) >= 2  # At least "Planning..." and "Evaluating..." logs

        # Verify thought logs were created
        thought_logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.THOUGHT,
        ).all()
        assert len(thought_logs) >= 1  # At least base table consideration

        # Verify summary log was created
        summary_logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.SUMMARY,
        ).all()
        assert len(summary_logs) == 1

    @pytest.mark.asyncio
    async def test_training_dataset_planning_missing_description(
        self, db_session, mock_llm_client, fake_profiles, fake_relationships_summary
    ):
        """Test that missing project_description raises error."""
        agent_run = AgentRun(name="Missing Description Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_PLANNING,
            input_json={
                # Missing project_description
                "data_source_profiles": fake_profiles,
                "relationships_summary": fake_relationships_summary,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Missing 'project_description'"):
            await handle_training_dataset_planning_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

    @pytest.mark.asyncio
    async def test_training_dataset_planning_missing_profiles(
        self, db_session, mock_llm_client, fake_relationships_summary
    ):
        """Test that missing data_source_profiles raises error."""
        agent_run = AgentRun(name="Missing Profiles Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_PLANNING,
            input_json={
                "project_description": "Predict churn",
                # Missing data_source_profiles
                "relationships_summary": fake_relationships_summary,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Missing 'data_source_profiles'"):
            await handle_training_dataset_planning_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

    @pytest.mark.asyncio
    async def test_training_dataset_planning_missing_relationships(
        self, db_session, mock_llm_client, fake_profiles
    ):
        """Test that missing relationships_summary raises error."""
        agent_run = AgentRun(name="Missing Relationships Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_PLANNING,
            input_json={
                "project_description": "Predict churn",
                "data_source_profiles": fake_profiles,
                # Missing relationships_summary
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Missing 'relationships_summary'"):
            await handle_training_dataset_planning_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

    @pytest.mark.asyncio
    async def test_training_dataset_planning_step_completes(
        self, db_session, mock_llm_client, fake_profiles, fake_relationships_summary
    ):
        """Test that step status becomes completed when run through run_agent_step."""
        agent_run = AgentRun(name="Full Step Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_PLANNING,
            status=AgentStepStatus.PENDING,
            input_json={
                "project_description": "Predict customer churn",
                "data_source_profiles": fake_profiles,
                "relationships_summary": fake_relationships_summary,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "training_dataset_spec": {
                "base_table": "customers",
                "base_filters": [],
                "target_definition": {"table": "customers", "column": "churned"},
                "join_plan": [],
                "excluded_tables": [],
                "excluded_columns": [],
            },
            "natural_language_summary": "Simple plan using customers table.",
        }

        # Run the step
        result = await run_agent_step(db_session, agent_step.id, mock_llm_client)

        # Verify step completed successfully
        assert result.status == AgentStepStatus.COMPLETED
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.output_json is not None
        assert "training_dataset_spec" in result.output_json

    @pytest.mark.asyncio
    async def test_training_dataset_planning_output_json_has_spec(
        self, db_session, mock_llm_client, fake_profiles, fake_relationships_summary
    ):
        """Test that output_json.training_dataset_spec exists after completion."""
        agent_run = AgentRun(name="Output Spec Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_PLANNING,
            status=AgentStepStatus.PENDING,
            input_json={
                "project_description": "Predict customer lifetime value",
                "target_hint": "ltv",
                "data_source_profiles": fake_profiles,
                "relationships_summary": fake_relationships_summary,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response with more complete spec
        mock_llm_client.chat_json.return_value = {
            "training_dataset_spec": {
                "base_table": "customers",
                "base_filters": [
                    {"column": "signup_date", "operator": ">=", "value": "2022-01-01"}
                ],
                "target_definition": {
                    "table": "customers",
                    "column": "churned",
                    "join_key": None,
                    "label_window_days": None,
                },
                "join_plan": [
                    {
                        "from_table": "customers",
                        "to_table": "transactions",
                        "left_key": "customer_id",
                        "right_key": "customer_id",
                        "relationship": "one_to_many",
                        "aggregation": {
                            "window_days": 30,
                            "features": [
                                {"name": "spend_30d", "agg": "sum", "column": "amount"},
                            ],
                        },
                    }
                ],
                "excluded_tables": [],
                "excluded_columns": ["transaction_id"],
            },
            "natural_language_summary": "Using customers with aggregated transactions.",
        }

        result = await run_agent_step(db_session, agent_step.id, mock_llm_client)

        # Verify output_json has the required structure
        assert result.output_json is not None
        assert "training_dataset_spec" in result.output_json

        spec = result.output_json["training_dataset_spec"]
        assert "base_table" in spec
        assert "target_definition" in spec
        assert "join_plan" in spec
        assert "base_filters" in spec
        assert "excluded_tables" in spec
        assert "excluded_columns" in spec

    @pytest.mark.asyncio
    async def test_training_dataset_planning_with_target_hint(
        self, db_session, mock_llm_client, fake_profiles, fake_relationships_summary
    ):
        """Test that target_hint is passed to LLM prompt."""
        agent_run = AgentRun(name="Target Hint Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.TRAINING_DATASET_PLANNING,
            input_json={
                "project_description": "Predict customer behavior",
                "target_hint": "churned",  # Explicit target hint
                "data_source_profiles": fake_profiles,
                "relationships_summary": fake_relationships_summary,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "training_dataset_spec": {
                "base_table": "customers",
                "base_filters": [],
                "target_definition": {"table": "customers", "column": "churned"},
                "join_plan": [],
                "excluded_tables": [],
                "excluded_columns": [],
            },
            "natural_language_summary": "Using churned as target based on hint.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        await handle_training_dataset_planning_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify LLM was called
        assert mock_llm_client.chat_json.called

        # Check that the prompt contained the target hint
        call_args = mock_llm_client.chat_json.call_args
        messages = call_args[0][0]  # First positional arg is messages
        user_message = next(m for m in messages if m["role"] == "user")
        assert "churned" in user_message["content"]


# ============================================
# Lab Notebook Summary Step Tests
# ============================================

class TestHandleLabNotebookSummaryStep:
    """Tests for handle_lab_notebook_summary_step."""

    @pytest.fixture
    def project_with_cycle(self, db_session):
        """Create a project with a research cycle and experiments."""
        # Create project
        project = Project(
            name="Test ML Project",
            task_type="binary",
            description="Predict customer churn",
        )
        db_session.add(project)
        db_session.commit()

        # Create research cycle
        cycle = ResearchCycle(
            project_id=project.id,
            sequence_number=1,
            status="running",
        )
        db_session.add(cycle)
        db_session.commit()

        # Create an experiment
        experiment = Experiment(
            project_id=project.id,
            name="Baseline Model",
            status="completed",
            primary_metric="roc_auc",
        )
        db_session.add(experiment)
        db_session.commit()

        # Link experiment to cycle
        cycle_experiment = CycleExperiment(
            research_cycle_id=cycle.id,
            experiment_id=experiment.id,
        )
        db_session.add(cycle_experiment)
        db_session.commit()

        return {
            "project": project,
            "cycle": cycle,
            "experiment": experiment,
        }

    @pytest.mark.asyncio
    async def test_lab_notebook_summary_success(self, db_session, mock_llm_client, project_with_cycle):
        """Test successful lab notebook summary generation."""
        project = project_with_cycle["project"]
        cycle = project_with_cycle["cycle"]

        agent_run = AgentRun(
            project_id=project.id,
            research_cycle_id=cycle.id,
            name="Lab Notebook Summary Test",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.LAB_NOTEBOOK_SUMMARY,
            input_json={
                "research_cycle_id": str(cycle.id),
                "project_id": str(project.id),
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "title": "Cycle #1: Initial Baseline Exploration",
            "body_markdown": "## Summary\n\nThis cycle established the baseline model...\n\n## Key Findings\n\n- Model achieved 0.85 ROC AUC\n\n## Next Steps\n\n- Try feature engineering",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_lab_notebook_summary_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify output structure
        assert "lab_note" in output
        assert "lab_notebook_entry_id" in output
        assert output["lab_note"]["title"] == "Cycle #1: Initial Baseline Exploration"
        assert "Key Findings" in output["lab_note"]["body_markdown"]

        # Verify lab notebook entry was created
        entry = db_session.query(LabNotebookEntry).filter(
            LabNotebookEntry.id == output["lab_notebook_entry_id"]
        ).first()
        assert entry is not None
        assert entry.author_type.value == "agent"
        assert entry.research_cycle_id == cycle.id
        assert entry.project_id == project.id
        assert "Key Findings" in entry.body_markdown

        # Verify logs were created
        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).all()
        assert len(logs) > 0

    @pytest.mark.asyncio
    async def test_lab_notebook_summary_updates_cycle_title(self, db_session, mock_llm_client, project_with_cycle):
        """Test that cycle summary_title is updated after generation."""
        project = project_with_cycle["project"]
        cycle = project_with_cycle["cycle"]

        agent_run = AgentRun(
            project_id=project.id,
            research_cycle_id=cycle.id,
            name="Update Title Test",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.LAB_NOTEBOOK_SUMMARY,
            input_json={
                "research_cycle_id": str(cycle.id),
                "project_id": str(project.id),
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "title": "Feature Engineering Breakthrough",
            "body_markdown": "## Summary\n\nMajor progress in this cycle...",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        await handle_lab_notebook_summary_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify cycle title was updated
        db_session.refresh(cycle)
        assert cycle.summary_title == "Feature Engineering Breakthrough"

    @pytest.mark.asyncio
    async def test_lab_notebook_summary_missing_cycle(self, db_session, mock_llm_client):
        """Test that missing research cycle raises error."""
        agent_run = AgentRun(name="Missing Cycle Test")
        db_session.add(agent_run)
        db_session.commit()

        fake_cycle_id = str(uuid.uuid4())
        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.LAB_NOTEBOOK_SUMMARY,
            input_json={
                "research_cycle_id": fake_cycle_id,
                "project_id": str(uuid.uuid4()),
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Research cycle not found"):
            await handle_lab_notebook_summary_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

    @pytest.mark.asyncio
    async def test_lab_notebook_summary_missing_input(self, db_session, mock_llm_client):
        """Test that missing research_cycle_id raises error."""
        agent_run = AgentRun(name="Missing Input Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.LAB_NOTEBOOK_SUMMARY,
            input_json={},  # Missing research_cycle_id
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Missing 'research_cycle_id'"):
            await handle_lab_notebook_summary_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

    @pytest.mark.asyncio
    async def test_lab_notebook_summary_creates_correct_logs(self, db_session, mock_llm_client, project_with_cycle):
        """Test that appropriate log types are created during summary generation."""
        project = project_with_cycle["project"]
        cycle = project_with_cycle["cycle"]

        agent_run = AgentRun(
            project_id=project.id,
            research_cycle_id=cycle.id,
            name="Logging Test",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.LAB_NOTEBOOK_SUMMARY,
            input_json={
                "research_cycle_id": str(cycle.id),
                "project_id": str(project.id),
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "title": "Test Summary",
            "body_markdown": "## Content\n\nTest content here.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        await handle_lab_notebook_summary_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify summary log was created
        summary_logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.SUMMARY,
        ).all()
        assert len(summary_logs) >= 1
        assert any("entry created" in log.message.lower() for log in summary_logs)

        # Verify action logs exist
        action_logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.ACTION,
        ).all()
        assert len(action_logs) >= 1

    @pytest.mark.asyncio
    async def test_lab_notebook_summary_with_no_experiments(self, db_session, mock_llm_client):
        """Test summary generation when cycle has no linked experiments."""
        # Create project and cycle without experiments
        project = Project(
            name="Empty Project",
            task_type="regression",
        )
        db_session.add(project)
        db_session.commit()

        cycle = ResearchCycle(
            project_id=project.id,
            sequence_number=1,
            status="pending",
        )
        db_session.add(cycle)
        db_session.commit()

        agent_run = AgentRun(
            project_id=project.id,
            research_cycle_id=cycle.id,
            name="No Experiments Test",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.LAB_NOTEBOOK_SUMMARY,
            input_json={
                "research_cycle_id": str(cycle.id),
                "project_id": str(project.id),
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response for empty cycle
        mock_llm_client.chat_json.return_value = {
            "title": "Cycle #1: Planning Phase",
            "body_markdown": "## Summary\n\nNo experiments have been run yet in this cycle.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_lab_notebook_summary_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Should succeed even with no experiments
        assert "lab_note" in output
        assert "lab_notebook_entry_id" in output

    @pytest.mark.asyncio
    async def test_lab_notebook_summary_step_completes_via_run_agent_step(
        self, db_session, mock_llm_client, project_with_cycle
    ):
        """Test that step completes successfully when run through run_agent_step."""
        project = project_with_cycle["project"]
        cycle = project_with_cycle["cycle"]

        agent_run = AgentRun(
            project_id=project.id,
            research_cycle_id=cycle.id,
            name="Full Step Test",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.LAB_NOTEBOOK_SUMMARY,
            status=AgentStepStatus.PENDING,
            input_json={
                "research_cycle_id": str(cycle.id),
                "project_id": str(project.id),
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response
        mock_llm_client.chat_json.return_value = {
            "title": "Test Title",
            "body_markdown": "## Summary\n\nTest content.",
        }

        # Run the step
        result = await run_agent_step(db_session, agent_step.id, mock_llm_client)

        # Verify step completed
        assert result.status == AgentStepStatus.COMPLETED
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.output_json is not None
        assert "lab_note" in result.output_json
        assert "lab_notebook_entry_id" in result.output_json


# ============================================
# Robustness Audit Step Tests
# ============================================

class TestHandleRobustnessAuditStep:
    """Tests for handle_robustness_audit_step."""

    @pytest.fixture
    def project_with_experiments(self, db_session):
        """Create a project with experiments and trials that have clear train/val gaps."""
        # Create project
        project = Project(
            name="Churn Prediction",
            task_type="binary",
            description="Predict customer churn",
        )
        db_session.add(project)
        db_session.commit()

        # Create experiment with overfitting (large train-val gap)
        experiment = Experiment(
            project_id=project.id,
            name="Overfitting Experiment",
            status="completed",
            primary_metric="roc_auc",
        )
        db_session.add(experiment)
        db_session.commit()

        # Create trial with clear train-val gap
        trial = Trial(
            experiment_id=experiment.id,
            variant_name="XGBoost",
            status=TrialStatus.COMPLETED,
            metrics_json={
                "train_roc_auc": 0.99,
                "val_roc_auc": 0.72,
                "roc_auc": 0.72,
            },
        )
        db_session.add(trial)
        db_session.commit()

        return {
            "project": project,
            "experiment": experiment,
            "trial": trial,
        }

    @pytest.fixture
    def project_with_good_experiment(self, db_session):
        """Create a project with a well-behaved experiment (no overfitting)."""
        project = Project(
            name="Good Model Project",
            task_type="binary",
            description="Classification task",
        )
        db_session.add(project)
        db_session.commit()

        experiment = Experiment(
            project_id=project.id,
            name="Healthy Experiment",
            status="completed",
            primary_metric="roc_auc",
        )
        db_session.add(experiment)
        db_session.commit()

        trial = Trial(
            experiment_id=experiment.id,
            variant_name="RandomForest",
            status=TrialStatus.COMPLETED,
            metrics_json={
                "train_roc_auc": 0.82,
                "val_roc_auc": 0.80,
                "roc_auc": 0.80,
            },
        )
        db_session.add(trial)
        db_session.commit()

        return {
            "project": project,
            "experiment": experiment,
            "trial": trial,
        }

    @pytest.mark.asyncio
    async def test_robustness_audit_detects_high_overfitting(
        self, db_session, mock_llm_client, project_with_experiments
    ):
        """Test that audit detects high overfitting risk with large train-val gap."""
        project = project_with_experiments["project"]
        experiment = project_with_experiments["experiment"]

        agent_run = AgentRun(
            project_id=project.id,
            name="Robustness Audit Test",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            input_json={
                "project_id": str(project.id),
                "experiment_ids": [str(experiment.id)],
                "primary_metric": "roc_auc",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response indicating high overfitting
        mock_llm_client.chat_json.return_value = {
            "overfitting_risk": "high",
            "train_val_analysis": {
                "worst_gap": 0.27,
                "avg_gap": 0.27,
                "interpretation": "Large train-validation gap indicates severe overfitting",
            },
            "suspicious_patterns": [
                {
                    "type": "train_val_gap",
                    "severity": "high",
                    "description": "Train AUC (0.99) vs Val AUC (0.72) shows 0.27 gap",
                }
            ],
            "baseline_comparison": {
                "baseline_type": "random_classifier",
                "baseline_metric": 0.5,
                "best_model_metric": 0.72,
                "relative_improvement": 0.44,
                "interpretation": "Model beats baseline but overfitting is concerning",
            },
            "cv_analysis": {},
            "recommendations": [
                "Use stronger regularization",
                "Reduce model complexity",
                "Increase training data",
            ],
            "natural_language_summary": "HIGH overfitting risk detected. Train AUC is 0.99 while validation is only 0.72.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_robustness_audit_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Verify output structure
        assert "robustness_audit" in output
        audit = output["robustness_audit"]
        assert audit["overfitting_risk"] == "high"
        assert len(audit["suspicious_patterns"]) > 0
        assert any(p["type"] == "train_val_gap" for p in audit["suspicious_patterns"])
        assert len(audit["recommendations"]) > 0

        # Verify logs were created with hypotheses
        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id
        ).all()
        assert len(logs) > 0

        # Check for hypothesis logs about overfitting
        hypothesis_logs = [l for l in logs if l.message_type == LogMessageType.HYPOTHESIS]
        assert len(hypothesis_logs) > 0

    @pytest.mark.asyncio
    async def test_robustness_audit_low_risk_for_good_model(
        self, db_session, mock_llm_client, project_with_good_experiment
    ):
        """Test that audit returns low risk for well-behaved model."""
        project = project_with_good_experiment["project"]
        experiment = project_with_good_experiment["experiment"]

        agent_run = AgentRun(
            project_id=project.id,
            name="Good Model Audit",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            input_json={
                "project_id": str(project.id),
                "experiment_ids": [str(experiment.id)],
                "primary_metric": "roc_auc",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        # Mock LLM response indicating low risk
        mock_llm_client.chat_json.return_value = {
            "overfitting_risk": "low",
            "train_val_analysis": {
                "worst_gap": 0.02,
                "avg_gap": 0.02,
                "interpretation": "Small gap between train and validation indicates good generalization",
            },
            "suspicious_patterns": [],
            "baseline_comparison": {
                "baseline_type": "random_classifier",
                "baseline_metric": 0.5,
                "best_model_metric": 0.80,
                "relative_improvement": 0.60,
                "interpretation": "Model significantly outperforms baseline",
            },
            "cv_analysis": {},
            "recommendations": [],
            "natural_language_summary": "Model appears robust with low overfitting risk.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_robustness_audit_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        assert output["robustness_audit"]["overfitting_risk"] == "low"
        assert len(output["robustness_audit"]["suspicious_patterns"]) == 0

    @pytest.mark.asyncio
    async def test_robustness_audit_no_experiments(self, db_session, mock_llm_client):
        """Test audit with no experiments returns appropriate response."""
        project = Project(
            name="Empty Project",
            task_type="regression",
        )
        db_session.add(project)
        db_session.commit()

        agent_run = AgentRun(
            project_id=project.id,
            name="Empty Audit",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            input_json={
                "project_id": str(project.id),
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_robustness_audit_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Should return "unknown" risk when no experiments
        assert output["robustness_audit"]["overfitting_risk"] == "unknown"
        assert "Run experiments first" in output["robustness_audit"]["recommendations"][0]

        # LLM should NOT be called when no experiments
        assert not mock_llm_client.chat_json.called

    @pytest.mark.asyncio
    async def test_robustness_audit_missing_project(self, db_session, mock_llm_client):
        """Test that missing project raises error."""
        agent_run = AgentRun(name="Missing Project Test")
        db_session.add(agent_run)
        db_session.commit()

        fake_project_id = str(uuid.uuid4())
        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            input_json={
                "project_id": fake_project_id,
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        step_logger = StepLogger(db_session, agent_step.id)

        with pytest.raises(ValueError, match="Project not found"):
            await handle_robustness_audit_step(
                db_session, agent_step, step_logger, mock_llm_client
            )

    @pytest.mark.asyncio
    async def test_robustness_audit_with_research_cycle(
        self, db_session, mock_llm_client, project_with_experiments
    ):
        """Test audit can pull experiments from a research cycle."""
        project = project_with_experiments["project"]
        experiment = project_with_experiments["experiment"]

        # Create research cycle and link experiment
        cycle = ResearchCycle(
            project_id=project.id,
            sequence_number=1,
            status="running",
        )
        db_session.add(cycle)
        db_session.commit()

        cycle_experiment = CycleExperiment(
            research_cycle_id=cycle.id,
            experiment_id=experiment.id,
        )
        db_session.add(cycle_experiment)
        db_session.commit()

        agent_run = AgentRun(
            project_id=project.id,
            research_cycle_id=cycle.id,
            name="Cycle Audit Test",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            input_json={
                "project_id": str(project.id),
                "research_cycle_id": str(cycle.id),
                "primary_metric": "roc_auc",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        mock_llm_client.chat_json.return_value = {
            "overfitting_risk": "high",
            "train_val_analysis": {"worst_gap": 0.27},
            "suspicious_patterns": [
                {"type": "train_val_gap", "severity": "high", "description": "test"}
            ],
            "baseline_comparison": {},
            "cv_analysis": {},
            "recommendations": ["test recommendation"],
            "natural_language_summary": "Test summary",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        output = await handle_robustness_audit_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Should have processed the experiment from the cycle
        assert "robustness_audit" in output
        assert mock_llm_client.chat_json.called

    @pytest.mark.asyncio
    async def test_robustness_audit_completes_via_run_agent_step(
        self, db_session, mock_llm_client, project_with_experiments
    ):
        """Test that step completes successfully when run through run_agent_step."""
        project = project_with_experiments["project"]
        experiment = project_with_experiments["experiment"]

        agent_run = AgentRun(
            project_id=project.id,
            name="Full Step Test",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            status=AgentStepStatus.PENDING,
            input_json={
                "project_id": str(project.id),
                "experiment_ids": [str(experiment.id)],
                "primary_metric": "roc_auc",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        mock_llm_client.chat_json.return_value = {
            "overfitting_risk": "medium",
            "train_val_analysis": {},
            "suspicious_patterns": [],
            "baseline_comparison": {},
            "cv_analysis": {},
            "recommendations": [],
            "natural_language_summary": "Medium risk detected.",
        }

        # Run the step
        result = await run_agent_step(db_session, agent_step.id, mock_llm_client)

        # Verify step completed
        assert result.status == AgentStepStatus.COMPLETED
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.output_json is not None
        assert "robustness_audit" in result.output_json
        assert result.output_json["robustness_audit"]["overfitting_risk"] == "medium"

    @pytest.mark.asyncio
    async def test_robustness_audit_logs_hypotheses_for_gaps(
        self, db_session, mock_llm_client, project_with_experiments
    ):
        """Test that audit logs hypotheses when detecting train-val gaps."""
        project = project_with_experiments["project"]
        experiment = project_with_experiments["experiment"]

        agent_run = AgentRun(
            project_id=project.id,
            name="Hypothesis Logging Test",
        )
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            input_json={
                "project_id": str(project.id),
                "experiment_ids": [str(experiment.id)],
                "primary_metric": "roc_auc",
            },
        )
        db_session.add(agent_step)
        db_session.commit()

        mock_llm_client.chat_json.return_value = {
            "overfitting_risk": "high",
            "train_val_analysis": {"worst_gap": 0.27},
            "suspicious_patterns": [],
            "baseline_comparison": {},
            "cv_analysis": {},
            "recommendations": [],
            "natural_language_summary": "High risk.",
        }

        step_logger = StepLogger(db_session, agent_step.id)

        await handle_robustness_audit_step(
            db_session, agent_step, step_logger, mock_llm_client
        )

        # Check for hypothesis logs
        hypothesis_logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == agent_step.id,
            AgentStepLog.message_type == LogMessageType.HYPOTHESIS,
        ).all()

        # Should have logged hypothesis about the large gap
        assert len(hypothesis_logs) > 0
        assert any("gap" in log.message.lower() for log in hypothesis_logs)


# ============================================
# Test Build Agent Context for Project
# ============================================

class TestBuildAgentContextForProject:
    """Tests for the build_agent_context_for_project helper function."""

    @pytest.fixture
    def project_with_history(self, db_session):
        """Create a project with research cycles, experiments, and notebook entries."""
        from app.models.project import TaskType
        project = Project(
            name="Test Project with History",
            description="Predict customer churn",
            task_type=TaskType.BINARY,
        )
        db_session.add(project)
        db_session.commit()

        # Create two research cycles
        cycle1 = ResearchCycle(
            project_id=project.id,
            sequence_number=1,
            status=ResearchCycleStatus.COMPLETED,
            summary_title="Initial exploration with simple features",
        )
        db_session.add(cycle1)
        db_session.commit()

        cycle2 = ResearchCycle(
            project_id=project.id,
            sequence_number=2,
            status=ResearchCycleStatus.COMPLETED,
            summary_title="Advanced feature engineering",
        )
        db_session.add(cycle2)
        db_session.commit()

        # Create experiments for cycle 1
        exp1 = Experiment(
            project_id=project.id,
            name="Baseline Model",
            status="completed",
            primary_metric="roc_auc",
            metric_direction="maximize",
        )
        db_session.add(exp1)
        db_session.commit()

        # Link experiment to cycle
        cycle_exp1 = CycleExperiment(
            research_cycle_id=cycle1.id,
            experiment_id=exp1.id,
        )
        db_session.add(cycle_exp1)
        db_session.commit()

        # Create trial with metrics
        trial1 = Trial(
            experiment_id=exp1.id,
            variant_name="RandomForest",
            status=TrialStatus.COMPLETED,
            metrics_json={
                "roc_auc": 0.75,
                "train_roc_auc": 0.82,
                "val_roc_auc": 0.75,
            },
        )
        db_session.add(trial1)
        db_session.commit()

        # Create lab notebook entry for cycle 1
        entry1 = LabNotebookEntry(
            project_id=project.id,
            research_cycle_id=cycle1.id,
            author_type=LabNotebookAuthorType.AGENT,
            title="Initial results analysis",
            body_markdown="Found that customer tenure is a strong predictor of churn.",
        )
        db_session.add(entry1)
        db_session.commit()

        # Create robustness audit step
        agent_run = AgentRun(
            project_id=project.id,
            name="Audit Run",
        )
        db_session.add(agent_run)
        db_session.commit()

        audit_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            status=AgentStepStatus.COMPLETED,
            output_json={
                "robustness_audit": {
                    "overfitting_risk": "medium",
                    "suspicious_patterns": [
                        {"description": "Moderate train-val gap of 0.07"},
                    ],
                    "recommendations": [
                        "Consider adding regularization",
                        "Try cross-validation with more folds",
                    ],
                    "natural_language_summary": "Model shows some overfitting risk.",
                }
            },
        )
        audit_step.finished_at = datetime.utcnow()
        db_session.add(audit_step)
        db_session.commit()

        # Create a dataset spec
        dataset_spec = DatasetSpec(
            project_id=project.id,
            name="Customer Features v1",
            target_column="churned",
            feature_columns=["tenure", "monthly_charges", "contract_type", "payment_method"],
            description="Basic customer features for churn prediction",
        )
        db_session.add(dataset_spec)
        db_session.commit()

        return {
            "project": project,
            "cycles": [cycle1, cycle2],
            "experiments": [exp1],
            "trials": [trial1],
            "entries": [entry1],
            "audit_step": audit_step,
            "dataset_spec": dataset_spec,
        }

    def test_context_includes_project_info(self, db_session, project_with_history):
        """Test that context includes basic project information."""
        project = project_with_history["project"]

        context = build_agent_context_for_project(db_session, project.id)

        assert context["problem_description"] == "Predict customer churn"
        assert context["task_type"].value == "binary"  # TaskType enum
        assert context["primary_metric"] == "roc_auc"  # inferred from binary task

    def test_context_includes_research_cycles(self, db_session, project_with_history):
        """Test that context includes research cycle history."""
        project = project_with_history["project"]

        context = build_agent_context_for_project(db_session, project.id)

        cycles = context["research_cycles"]
        assert len(cycles) >= 1
        assert cycles[0]["sequence_number"] == 2  # Most recent first
        assert cycles[0]["title"] == "Advanced feature engineering"

    def test_context_includes_best_models(self, db_session, project_with_history):
        """Test that context includes best performing models."""
        project = project_with_history["project"]

        context = build_agent_context_for_project(db_session, project.id)

        best_models = context["best_models"]
        assert len(best_models) >= 1
        assert best_models[0]["experiment"] == "Baseline Model"
        assert best_models[0]["trial"] == "RandomForest"
        assert best_models[0]["value"] == 0.75

    def test_context_includes_robustness_findings(self, db_session, project_with_history):
        """Test that context includes robustness audit findings."""
        project = project_with_history["project"]

        context = build_agent_context_for_project(db_session, project.id)

        robustness = context["robustness_findings"]
        assert robustness is not None
        assert robustness["overfitting_risk"] == "medium"
        assert len(robustness["suspicious_patterns"]) > 0
        assert len(robustness["recommendations"]) > 0

    def test_context_includes_notebook_highlights(self, db_session, project_with_history):
        """Test that context includes lab notebook highlights."""
        project = project_with_history["project"]

        context = build_agent_context_for_project(db_session, project.id)

        highlights = context["notebook_highlights"]
        assert len(highlights) >= 1
        assert highlights[0]["title"] == "Initial results analysis"
        assert highlights[0]["author"] == "agent"

    def test_context_includes_dataset_specs(self, db_session, project_with_history):
        """Test that context includes dataset specifications."""
        project = project_with_history["project"]

        context = build_agent_context_for_project(db_session, project.id)

        specs = context["dataset_specs"]
        assert len(specs) >= 1
        assert specs[0]["name"] == "Customer Features v1"
        assert specs[0]["target_column"] == "churned"
        assert specs[0]["feature_count"] == 4

    def test_context_excludes_current_cycle(self, db_session, project_with_history):
        """Test that current research cycle is excluded from history."""
        project = project_with_history["project"]
        current_cycle = project_with_history["cycles"][1]  # cycle2

        context = build_agent_context_for_project(
            db_session, project.id, research_cycle_id=current_cycle.id
        )

        cycles = context["research_cycles"]
        # Current cycle should be excluded
        assert all(c["sequence_number"] != 2 for c in cycles)

    def test_context_respects_max_limits(self, db_session, project_with_history):
        """Test that context respects max_cycles and other limits."""
        project = project_with_history["project"]

        context = build_agent_context_for_project(
            db_session, project.id, max_cycles=1, max_notebook_entries=1
        )

        assert len(context["research_cycles"]) <= 1
        assert len(context["notebook_highlights"]) <= 1

    def test_context_empty_for_nonexistent_project(self, db_session):
        """Test that context is empty for nonexistent project."""
        fake_id = uuid.uuid4()

        context = build_agent_context_for_project(db_session, fake_id)

        assert context["problem_description"] is None
        assert context["research_cycles"] == []
        assert context["best_models"] == []

    def test_format_project_context_includes_all_sections(self, db_session, project_with_history):
        """Test that formatted context includes all sections."""
        project = project_with_history["project"]

        context = build_agent_context_for_project(db_session, project.id)
        formatted = format_project_context_for_prompt(context)

        # Check for section headers
        assert "Project History & Context" in formatted
        assert "Previous Research Cycles" in formatted
        assert "Best Performing Models" in formatted
        assert "Latest Robustness Audit" in formatted
        assert "Previous Dataset Configurations" in formatted

    def test_format_empty_context(self):
        """Test that formatting handles empty context gracefully."""
        empty_context = {
            "problem_description": None,
            "task_type": None,
            "primary_metric": None,
            "research_cycles": [],
            "best_models": [],
            "robustness_findings": None,
            "notebook_highlights": [],
            "dataset_specs": [],
        }

        formatted = format_project_context_for_prompt(empty_context)

        assert "No previous history available" in formatted


class TestPlanningHandlersWithContext:
    """Integration tests for planning handlers with project history context."""

    @pytest.fixture
    def project_with_audit_history(self, db_session):
        """Create a project with robustness audit history."""
        from app.models.project import TaskType
        project = Project(
            name="Test Project",
            description="Test prediction task",
            task_type=TaskType.BINARY,
        )
        db_session.add(project)
        db_session.commit()

        # Create an agent run with robustness audit
        agent_run = AgentRun(
            project_id=project.id,
            name="Previous Run",
        )
        db_session.add(agent_run)
        db_session.commit()

        audit_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            status=AgentStepStatus.COMPLETED,
            output_json={
                "robustness_audit": {
                    "overfitting_risk": "high",
                    "suspicious_patterns": [
                        {"description": "Large train-val gap detected"},
                    ],
                    "recommendations": [
                        "Use temporal validation for time-series data",
                        "Add regularization",
                    ],
                    "natural_language_summary": "High overfitting risk.",
                }
            },
        )
        audit_step.finished_at = datetime.utcnow()
        db_session.add(audit_step)
        db_session.commit()

        return project

    @pytest.mark.asyncio
    async def test_dataset_design_uses_project_context(
        self, db_session, mock_llm_client, mock_schema_summary, project_with_audit_history
    ):
        """Test that dataset design handler includes project context in LLM call."""
        project = project_with_audit_history

        # Create agent run for this step
        agent_run = AgentRun(
            project_id=project.id,
            name="Current Run",
        )
        db_session.add(agent_run)
        db_session.commit()

        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATASET_DESIGN,
            input_json={
                "schema_summary": mock_schema_summary,
                "task_type": "binary",
                "target_column": "target",
                "description": "Test prediction",
            },
        )
        db_session.add(step)
        db_session.commit()

        # Mock LLM response - use chat_with_tools for new tool-calling flow
        mock_response = {
            "variants": [
                {
                    "name": "baseline",
                    "feature_columns": ["age", "income", "category"],
                    "excluded_columns": ["id"],
                    "exclusion_reasons": {"id": "ID column"},
                    "suggested_filters": None,
                    "train_test_split": "80/20",
                    "description": "Baseline features",
                    "engineered_features": [],
                    "expected_tradeoff": "Simple features, may need regularization given history",
                }
            ],
            "recommended_variant": "baseline",
            "reasoning": "Based on project history showing overfitting risk, using simple features.",
            "warnings": [],
        }
        # Set up chat_with_tools to return content directly (no tool calls)
        mock_llm_client.chat_with_tools.return_value = {"content": mock_response}

        step_logger = StepLogger(db_session, step.id)
        await handle_dataset_design_step(db_session, step, step_logger, mock_llm_client)

        # Verify LLM was called via chat_with_tools (new tool-enabled flow)
        assert mock_llm_client.chat_with_tools.called

        # Check that thinking logs include history review
        logs = db_session.query(AgentStepLog).filter(
            AgentStepLog.agent_step_id == step.id
        ).all()

        # Should have logs about reviewing history
        thinking_logs = [l for l in logs if l.message_type == LogMessageType.THINKING]
        assert len(thinking_logs) > 0
        # Check that some log mentions reviewing history or robustness
        history_mentioned = any(
            "history" in l.message.lower() or "robustness" in l.message.lower() or "audit" in l.message.lower()
            for l in thinking_logs
        )
        assert history_mentioned, "Should log about reviewing project history"

    @pytest.mark.asyncio
    async def test_different_history_produces_different_context(self, db_session, mock_llm_client):
        """Test that different project histories produce different contexts."""
        from app.models.project import TaskType

        # Create project 1 with high risk
        project1 = Project(
            name="Project 1",
            description="High risk project",
            task_type=TaskType.BINARY,
        )
        db_session.add(project1)
        db_session.commit()

        run1 = AgentRun(project_id=project1.id, name="Run 1")
        db_session.add(run1)
        db_session.commit()

        audit1 = AgentStep(
            agent_run_id=run1.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            status=AgentStepStatus.COMPLETED,
            output_json={
                "robustness_audit": {
                    "overfitting_risk": "high",
                    "suspicious_patterns": [],
                    "recommendations": ["Fix overfitting"],
                    "natural_language_summary": "High risk!",
                }
            },
        )
        audit1.finished_at = datetime.utcnow()
        db_session.add(audit1)
        db_session.commit()

        # Create project 2 with low risk
        project2 = Project(
            name="Project 2",
            description="Low risk project",
            task_type=TaskType.REGRESSION,
        )
        db_session.add(project2)
        db_session.commit()

        run2 = AgentRun(project_id=project2.id, name="Run 2")
        db_session.add(run2)
        db_session.commit()

        audit2 = AgentStep(
            agent_run_id=run2.id,
            step_type=AgentStepType.ROBUSTNESS_AUDIT,
            status=AgentStepStatus.COMPLETED,
            output_json={
                "robustness_audit": {
                    "overfitting_risk": "low",
                    "suspicious_patterns": [],
                    "recommendations": [],
                    "natural_language_summary": "Model is robust.",
                }
            },
        )
        audit2.finished_at = datetime.utcnow()
        db_session.add(audit2)
        db_session.commit()

        # Build contexts
        context1 = build_agent_context_for_project(db_session, project1.id)
        context2 = build_agent_context_for_project(db_session, project2.id)

        # Verify contexts are different
        assert context1["robustness_findings"]["overfitting_risk"] == "high"
        assert context2["robustness_findings"]["overfitting_risk"] == "low"

        # Format and verify different content
        formatted1 = format_project_context_for_prompt(context1)
        formatted2 = format_project_context_for_prompt(context2)

        assert "HIGH" in formatted1
        assert "LOW" in formatted2

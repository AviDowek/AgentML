"""Tests for Agent Run, Step, and Log models."""
import pytest
from datetime import datetime
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
    User,
)


class TestAgentRunModels:
    """Tests for agent run model operations."""

    def test_create_agent_run(self, db_session, test_user):
        """Test creating an agent run."""
        # Create a project first
        project = Project(
            name="Test Project",
            owner_id=test_user.id,
        )
        db_session.add(project)
        db_session.commit()

        # Create agent run
        agent_run = AgentRun(
            project_id=project.id,
            name="Test Agent Run",
            description="Testing agent run creation",
            status=AgentRunStatus.PENDING,
            config_json={"goal": "test"},
        )
        db_session.add(agent_run)
        db_session.commit()
        db_session.refresh(agent_run)

        assert agent_run.id is not None
        assert agent_run.project_id == project.id
        assert agent_run.name == "Test Agent Run"
        assert agent_run.status == AgentRunStatus.PENDING
        assert agent_run.config_json == {"goal": "test"}
        assert agent_run.created_at is not None

    def test_agent_run_without_project(self, db_session):
        """Test creating an agent run without a project (standalone)."""
        agent_run = AgentRun(
            name="Standalone Run",
            status=AgentRunStatus.PENDING,
        )
        db_session.add(agent_run)
        db_session.commit()

        assert agent_run.id is not None
        assert agent_run.project_id is None
        assert agent_run.experiment_id is None

    def test_agent_run_status_transitions(self, db_session):
        """Test agent run status transitions."""
        agent_run = AgentRun(
            name="Status Test Run",
            status=AgentRunStatus.PENDING,
        )
        db_session.add(agent_run)
        db_session.commit()

        # Transition to running
        agent_run.status = AgentRunStatus.RUNNING
        db_session.commit()
        assert agent_run.status == AgentRunStatus.RUNNING

        # Transition to completed
        agent_run.status = AgentRunStatus.COMPLETED
        agent_run.result_json = {"final_metric": 0.95}
        db_session.commit()
        assert agent_run.status == AgentRunStatus.COMPLETED
        assert agent_run.result_json["final_metric"] == 0.95


class TestAgentStepModels:
    """Tests for agent step model operations."""

    def test_create_agent_step(self, db_session):
        """Test creating an agent step."""
        # Create agent run first
        agent_run = AgentRun(
            name="Run with Steps",
            status=AgentRunStatus.RUNNING,
        )
        db_session.add(agent_run)
        db_session.commit()

        # Create agent step
        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
            status=AgentStepStatus.PENDING,
            input_json={"data_source_id": str(uuid.uuid4())},
        )
        db_session.add(agent_step)
        db_session.commit()
        db_session.refresh(agent_step)

        assert agent_step.id is not None
        assert agent_step.agent_run_id == agent_run.id
        assert agent_step.step_type == AgentStepType.DATA_AUDIT
        assert agent_step.status == AgentStepStatus.PENDING
        assert agent_step.retry_count == 0

    def test_agent_step_lifecycle(self, db_session):
        """Test agent step start, complete, and fail methods."""
        agent_run = AgentRun(name="Lifecycle Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PROBLEM_UNDERSTANDING,
            status=AgentStepStatus.PENDING,
        )
        db_session.add(agent_step)
        db_session.commit()

        # Test start
        agent_step.start()
        db_session.commit()
        assert agent_step.status == AgentStepStatus.RUNNING
        assert agent_step.started_at is not None

        # Test complete
        agent_step.complete(output={"task_type": "binary", "target": "churn"})
        db_session.commit()
        assert agent_step.status == AgentStepStatus.COMPLETED
        assert agent_step.finished_at is not None
        assert agent_step.output_json["task_type"] == "binary"

    def test_agent_step_failure(self, db_session):
        """Test agent step failure handling."""
        agent_run = AgentRun(name="Failure Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.EXPERIMENT_DESIGN,
            status=AgentStepStatus.PENDING,
        )
        db_session.add(agent_step)
        db_session.commit()

        agent_step.start()
        agent_step.fail("LLM API call failed: rate limit exceeded")
        db_session.commit()

        assert agent_step.status == AgentStepStatus.FAILED
        assert agent_step.finished_at is not None
        assert "rate limit" in agent_step.error_message

    def test_multiple_steps_in_run(self, db_session):
        """Test creating multiple steps in a run."""
        agent_run = AgentRun(name="Multi-step Run")
        db_session.add(agent_run)
        db_session.commit()

        step_types = [
            AgentStepType.PROBLEM_UNDERSTANDING,
            AgentStepType.DATA_AUDIT,
            AgentStepType.DATASET_DESIGN,
            AgentStepType.EXPERIMENT_DESIGN,
            AgentStepType.PLAN_CRITIC,
        ]

        for i, step_type in enumerate(step_types):
            step = AgentStep(
                agent_run_id=agent_run.id,
                step_type=step_type,
                status=AgentStepStatus.PENDING,
            )
            db_session.add(step)

        db_session.commit()
        db_session.refresh(agent_run)

        assert len(agent_run.steps) == 5
        assert agent_run.steps[0].step_type == AgentStepType.PROBLEM_UNDERSTANDING


class TestAgentStepLogModels:
    """Tests for agent step log model operations."""

    def test_create_step_log(self, db_session):
        """Test creating a step log."""
        agent_run = AgentRun(name="Log Test Run")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
            status=AgentStepStatus.RUNNING,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = AgentStepLog(
            agent_step_id=agent_step.id,
            sequence=1,
            message_type=LogMessageType.INFO,
            message="Analyzing data schema...",
        )
        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)

        assert log.id is not None
        assert log.agent_step_id == agent_step.id
        assert log.sequence == 1
        assert log.message_type == LogMessageType.INFO
        assert log.timestamp is not None

    def test_multiple_logs_with_ordering(self, db_session):
        """Test multiple logs maintain sequence ordering."""
        agent_run = AgentRun(name="Multi-log Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
            status=AgentStepStatus.RUNNING,
        )
        db_session.add(agent_step)
        db_session.commit()

        messages = [
            (LogMessageType.INFO, "Starting data audit..."),
            (LogMessageType.THOUGHT, "Examining column distributions"),
            (LogMessageType.WARNING, "High null percentage in 'income' column"),
            (LogMessageType.INFO, "Found 3 potential ID columns"),
            (LogMessageType.SUMMARY, "Data audit complete: 5 issues found"),
        ]

        for i, (msg_type, message) in enumerate(messages, start=1):
            log = AgentStepLog(
                agent_step_id=agent_step.id,
                sequence=i,
                message_type=msg_type,
                message=message,
            )
            db_session.add(log)

        db_session.commit()
        db_session.refresh(agent_step)

        assert len(agent_step.logs) == 5
        assert agent_step.logs[0].sequence == 1
        assert agent_step.logs[0].message_type == LogMessageType.INFO
        assert agent_step.logs[-1].message_type == LogMessageType.SUMMARY

    def test_log_with_metadata(self, db_session):
        """Test creating a log with structured metadata."""
        agent_run = AgentRun(name="Metadata Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.EXPERIMENT_DESIGN,
            status=AgentStepStatus.RUNNING,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = AgentStepLog(
            agent_step_id=agent_step.id,
            sequence=1,
            message_type=LogMessageType.INFO,
            message="Generated experiment variants",
            metadata_json={
                "variants": ["quick", "balanced", "high_quality"],
                "recommended": "balanced",
                "estimated_time_minutes": 25,
            },
        )
        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)

        assert log.metadata_json is not None
        assert len(log.metadata_json["variants"]) == 3
        assert log.metadata_json["recommended"] == "balanced"


class TestAgentRunRelationships:
    """Tests for agent run relationships and cascades."""

    def test_project_relationship(self, db_session, test_user):
        """Test relationship between project and agent runs."""
        project = Project(name="Relationship Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        run1 = AgentRun(project_id=project.id, name="Run 1")
        run2 = AgentRun(project_id=project.id, name="Run 2")
        db_session.add_all([run1, run2])
        db_session.commit()
        db_session.refresh(project)

        assert len(project.agent_runs) == 2

    def test_cascade_delete_steps_and_logs(self, db_session):
        """Test that deleting a run cascades to steps and logs."""
        agent_run = AgentRun(name="Cascade Test")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.DATA_AUDIT,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = AgentStepLog(
            agent_step_id=agent_step.id,
            sequence=1,
            message_type=LogMessageType.INFO,
            message="Test log",
        )
        db_session.add(log)
        db_session.commit()

        run_id = agent_run.id
        step_id = agent_step.id
        log_id = log.id

        # Delete the run
        db_session.delete(agent_run)
        db_session.commit()

        # Verify cascade deletion
        assert db_session.query(AgentRun).filter_by(id=run_id).first() is None
        assert db_session.query(AgentStep).filter_by(id=step_id).first() is None
        assert db_session.query(AgentStepLog).filter_by(id=log_id).first() is None


class TestMigration:
    """Tests to verify migration created tables correctly."""

    def test_tables_exist(self, db_session):
        """Test that all agent tables exist and are accessible."""
        from sqlalchemy import inspect

        inspector = inspect(db_session.get_bind())
        tables = inspector.get_table_names()

        assert "agent_runs" in tables
        assert "agent_steps" in tables
        assert "agent_step_logs" in tables

    def test_foreign_keys_valid(self, db_session, test_user):
        """Test that foreign keys work correctly."""
        project = Project(name="FK Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        agent_run = AgentRun(project_id=project.id, name="FK Run")
        db_session.add(agent_run)
        db_session.commit()

        agent_step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PROBLEM_UNDERSTANDING,
        )
        db_session.add(agent_step)
        db_session.commit()

        log = AgentStepLog(
            agent_step_id=agent_step.id,
            sequence=1,
            message_type=LogMessageType.INFO,
            message="FK test log",
        )
        db_session.add(log)
        db_session.commit()

        # Verify all relationships work
        assert agent_run.project.name == "FK Test"
        assert agent_step.agent_run.name == "FK Run"
        assert log.agent_step.step_type == AgentStepType.PROBLEM_UNDERSTANDING

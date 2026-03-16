"""Tests for Agent Streaming Logs & Run/Step APIs (Phase 5.4)."""
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
)


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def project_with_agent_run(db_session, test_user):
    """Create a project with a complete agent run including steps and logs."""
    # Create project
    project = Project(name="Streaming Test Project", owner_id=test_user.id)
    db_session.add(project)
    db_session.commit()

    # Create agent run
    agent_run = AgentRun(
        project_id=project.id,
        name="Test Pipeline Run",
        description="A test pipeline run",
        status=AgentRunStatus.COMPLETED,
        config_json={"description": "Test run"},
        result_json={"task_type": "binary", "target_column": "target"},
    )
    db_session.add(agent_run)
    db_session.commit()

    # Create steps
    steps = []
    step_types = [
        AgentStepType.PROBLEM_UNDERSTANDING,
        AgentStepType.DATA_AUDIT,
        AgentStepType.DATASET_DESIGN,
    ]
    for i, step_type in enumerate(step_types):
        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=step_type,
            status=AgentStepStatus.COMPLETED,
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            input_json={"input_key": f"input_value_{i}"},
            output_json={"output_key": f"output_value_{i}"},
        )
        db_session.add(step)
        steps.append(step)
    db_session.commit()

    # Create logs for the first step
    for seq in range(1, 6):
        log = AgentStepLog(
            agent_step_id=steps[0].id,
            sequence=seq,
            timestamp=datetime.utcnow(),
            message_type=LogMessageType.INFO,
            message=f"Log message {seq}",
            metadata_json={"seq": seq},
        )
        db_session.add(log)
    db_session.commit()

    return project, agent_run, steps


@pytest.fixture
def running_step_with_logs(db_session, test_user):
    """Create a project with a running step and some logs."""
    # Create project
    project = Project(name="Running Step Test", owner_id=test_user.id)
    db_session.add(project)
    db_session.commit()

    # Create agent run (running)
    agent_run = AgentRun(
        project_id=project.id,
        name="Running Pipeline",
        status=AgentRunStatus.RUNNING,
    )
    db_session.add(agent_run)
    db_session.commit()

    # Create a running step
    step = AgentStep(
        agent_run_id=agent_run.id,
        step_type=AgentStepType.PROBLEM_UNDERSTANDING,
        status=AgentStepStatus.RUNNING,
        started_at=datetime.utcnow(),
    )
    db_session.add(step)
    db_session.commit()

    # Create some logs
    for seq in range(1, 4):
        log = AgentStepLog(
            agent_step_id=step.id,
            sequence=seq,
            timestamp=datetime.utcnow(),
            message_type=LogMessageType.INFO,
            message=f"Running log {seq}",
        )
        db_session.add(log)
    db_session.commit()

    return project, agent_run, step


# ============================================
# GET /agent-runs/{run_id} Tests
# ============================================

class TestGetAgentRun:
    """Tests for GET /agent-runs/{run_id} endpoint."""

    def test_get_agent_run_success(self, client, db_session, project_with_agent_run):
        """Test getting an agent run with its steps."""
        project, agent_run, steps = project_with_agent_run

        response = client.get(f"/agent-runs/{agent_run.id}")

        assert response.status_code == 200
        data = response.json()

        # Verify run metadata
        assert data["id"] == str(agent_run.id)
        assert data["name"] == "Test Pipeline Run"
        assert data["status"] == "completed"
        assert data["project_id"] == str(project.id)

        # Verify steps are included
        assert "steps" in data
        assert len(data["steps"]) == 3

        # Verify step details
        step_types = [s["step_type"] for s in data["steps"]]
        assert "problem_understanding" in step_types
        assert "data_audit" in step_types
        assert "dataset_design" in step_types

        # Verify step has timestamps
        for step in data["steps"]:
            assert "id" in step
            assert "step_type" in step
            assert "status" in step
            assert "started_at" in step
            assert "finished_at" in step

    def test_get_agent_run_not_found(self, client, db_session, test_user):
        """Test 404 for non-existent agent run."""
        fake_id = uuid.uuid4()
        response = client.get(f"/agent-runs/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# ============================================
# GET /agent-steps/{step_id} Tests
# ============================================

class TestGetAgentStep:
    """Tests for GET /agent-steps/{step_id} endpoint."""

    def test_get_agent_step_success(self, client, db_session, project_with_agent_run):
        """Test getting an agent step with metadata and I/O."""
        project, agent_run, steps = project_with_agent_run
        step = steps[0]

        response = client.get(f"/agent-steps/{step.id}")

        assert response.status_code == 200
        data = response.json()

        # Verify step metadata
        assert data["id"] == str(step.id)
        assert data["agent_run_id"] == str(agent_run.id)
        assert data["step_type"] == "problem_understanding"
        assert data["status"] == "completed"

        # Verify input/output JSON
        assert data["input_json"] == {"input_key": "input_value_0"}
        assert data["output_json"] == {"output_key": "output_value_0"}

        # Verify timestamps
        assert data["started_at"] is not None
        assert data["finished_at"] is not None

    def test_get_agent_step_not_found(self, client, db_session, test_user):
        """Test 404 for non-existent agent step."""
        fake_id = uuid.uuid4()
        response = client.get(f"/agent-steps/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# ============================================
# GET /agent-steps/{step_id}/logs Tests
# ============================================

class TestGetStepLogs:
    """Tests for GET /agent-steps/{step_id}/logs endpoint."""

    def test_get_all_logs(self, client, db_session, project_with_agent_run):
        """Test getting all logs with since_sequence=0."""
        project, agent_run, steps = project_with_agent_run
        step = steps[0]

        response = client.get(f"/agent-steps/{step.id}/logs?since_sequence=0")

        assert response.status_code == 200
        data = response.json()

        # Verify all 5 logs returned
        assert len(data["logs"]) == 5

        # Verify logs are in order
        sequences = [log["sequence"] for log in data["logs"]]
        assert sequences == [1, 2, 3, 4, 5]

        # Verify last_sequence
        assert data["last_sequence"] == 5

        # Verify has_more is False (step is completed)
        assert data["has_more"] is False

        # Verify log content
        for log in data["logs"]:
            assert "id" in log
            assert "message_type" in log
            assert "message" in log
            assert "timestamp" in log
            assert "sequence" in log

    def test_get_logs_since_sequence(self, client, db_session, project_with_agent_run):
        """Test getting only new logs with since_sequence."""
        project, agent_run, steps = project_with_agent_run
        step = steps[0]

        # First call - get all logs
        response1 = client.get(f"/agent-steps/{step.id}/logs?since_sequence=0")
        assert response1.status_code == 200
        data1 = response1.json()
        assert len(data1["logs"]) == 5
        last_seq = data1["last_sequence"]
        assert last_seq == 5

        # Second call - get only new logs (should be empty)
        response2 = client.get(f"/agent-steps/{step.id}/logs?since_sequence={last_seq}")
        assert response2.status_code == 200
        data2 = response2.json()
        assert len(data2["logs"]) == 0
        assert data2["last_sequence"] == last_seq

    def test_get_logs_partial(self, client, db_session, project_with_agent_run):
        """Test getting partial logs."""
        project, agent_run, steps = project_with_agent_run
        step = steps[0]

        # Get logs after sequence 2
        response = client.get(f"/agent-steps/{step.id}/logs?since_sequence=2")

        assert response.status_code == 200
        data = response.json()

        # Should get logs 3, 4, 5
        assert len(data["logs"]) == 3
        sequences = [log["sequence"] for log in data["logs"]]
        assert sequences == [3, 4, 5]
        assert data["last_sequence"] == 5

    def test_get_logs_running_step_has_more(self, client, db_session, running_step_with_logs):
        """Test that running step returns has_more=True."""
        project, agent_run, step = running_step_with_logs

        response = client.get(f"/agent-steps/{step.id}/logs?since_sequence=0")

        assert response.status_code == 200
        data = response.json()

        assert len(data["logs"]) == 3
        assert data["has_more"] is True  # Step is still running

    def test_get_logs_with_limit(self, client, db_session, project_with_agent_run):
        """Test logs endpoint respects limit parameter."""
        project, agent_run, steps = project_with_agent_run
        step = steps[0]

        response = client.get(f"/agent-steps/{step.id}/logs?since_sequence=0&limit=2")

        assert response.status_code == 200
        data = response.json()

        # Should only get first 2 logs
        assert len(data["logs"]) == 2
        sequences = [log["sequence"] for log in data["logs"]]
        assert sequences == [1, 2]
        assert data["last_sequence"] == 2

    def test_get_logs_step_not_found(self, client, db_session, test_user):
        """Test 404 for non-existent step."""
        fake_id = uuid.uuid4()
        response = client.get(f"/agent-steps/{fake_id}/logs")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# ============================================
# Log Streaming Integration Test
# ============================================

class TestLogStreamingWorkflow:
    """Integration tests for log streaming workflow."""

    def test_polling_workflow(self, client, db_session, test_user):
        """Test the full polling workflow for streaming logs."""
        # Create project
        project = Project(name="Polling Test", owner_id=test_user.id)
        db_session.add(project)
        db_session.commit()

        # Create run and step
        agent_run = AgentRun(
            project_id=project.id,
            name="Polling Test Run",
            status=AgentRunStatus.RUNNING,
        )
        db_session.add(agent_run)
        db_session.commit()

        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=AgentStepType.PROBLEM_UNDERSTANDING,
            status=AgentStepStatus.RUNNING,
        )
        db_session.add(step)
        db_session.commit()

        # Poll 1: No logs yet
        response1 = client.get(f"/agent-steps/{step.id}/logs?since_sequence=0")
        assert response1.status_code == 200
        data1 = response1.json()
        assert len(data1["logs"]) == 0
        assert data1["last_sequence"] == 0
        assert data1["has_more"] is True

        # Add some logs (simulating step execution)
        for seq in range(1, 4):
            log = AgentStepLog(
                agent_step_id=step.id,
                sequence=seq,
                timestamp=datetime.utcnow(),
                message_type=LogMessageType.INFO,
                message=f"Message {seq}",
            )
            db_session.add(log)
        db_session.commit()

        # Poll 2: Get new logs
        response2 = client.get(f"/agent-steps/{step.id}/logs?since_sequence=0")
        assert response2.status_code == 200
        data2 = response2.json()
        assert len(data2["logs"]) == 3
        assert data2["last_sequence"] == 3
        assert data2["has_more"] is True

        # Add more logs
        for seq in range(4, 6):
            log = AgentStepLog(
                agent_step_id=step.id,
                sequence=seq,
                timestamp=datetime.utcnow(),
                message_type=LogMessageType.INFO,
                message=f"Message {seq}",
            )
            db_session.add(log)
        db_session.commit()

        # Poll 3: Get only new logs using last_sequence
        response3 = client.get(f"/agent-steps/{step.id}/logs?since_sequence={data2['last_sequence']}")
        assert response3.status_code == 200
        data3 = response3.json()
        assert len(data3["logs"]) == 2  # Only logs 4 and 5
        sequences = [log["sequence"] for log in data3["logs"]]
        assert sequences == [4, 5]
        assert data3["last_sequence"] == 5

        # Complete the step
        step.status = AgentStepStatus.COMPLETED
        db_session.commit()

        # Poll 4: has_more should now be False
        response4 = client.get(f"/agent-steps/{step.id}/logs?since_sequence=5")
        assert response4.status_code == 200
        data4 = response4.json()
        assert len(data4["logs"]) == 0
        assert data4["has_more"] is False


# ============================================
# Response Structure Tests
# ============================================

class TestResponseStructure:
    """Tests for verifying correct response structures."""

    def test_agent_run_response_structure(self, client, db_session, project_with_agent_run):
        """Verify agent run response has proper structure."""
        project, agent_run, steps = project_with_agent_run

        response = client.get(f"/agent-runs/{agent_run.id}")
        data = response.json()

        # Required fields for run
        assert "id" in data
        assert "status" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "steps" in data

        # Optional fields
        assert "project_id" in data
        assert "name" in data
        assert "description" in data
        assert "config_json" in data
        assert "result_json" in data

    def test_agent_step_response_structure(self, client, db_session, project_with_agent_run):
        """Verify agent step response has proper structure."""
        project, agent_run, steps = project_with_agent_run

        response = client.get(f"/agent-steps/{steps[0].id}")
        data = response.json()

        # Required fields
        assert "id" in data
        assert "agent_run_id" in data
        assert "step_type" in data
        assert "status" in data
        assert "created_at" in data
        assert "updated_at" in data

        # Optional fields
        assert "input_json" in data
        assert "output_json" in data
        assert "started_at" in data
        assert "finished_at" in data
        assert "error_message" in data
        assert "retry_count" in data

    def test_logs_response_structure(self, client, db_session, project_with_agent_run):
        """Verify logs response has proper structure."""
        project, agent_run, steps = project_with_agent_run

        response = client.get(f"/agent-steps/{steps[0].id}/logs")
        data = response.json()

        # Top-level fields
        assert "logs" in data
        assert "last_sequence" in data
        assert "has_more" in data

        # Log entry fields
        if data["logs"]:
            log = data["logs"][0]
            assert "id" in log
            assert "agent_step_id" in log
            assert "sequence" in log
            assert "timestamp" in log
            assert "message_type" in log
            assert "message" in log

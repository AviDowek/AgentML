"""Tests for agent history tools."""
import pytest
from uuid import uuid4
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from sqlalchemy.orm import Session

from app.services.agent_tools import (
    AGENT_HISTORY_TOOLS,
    AgentToolExecutor,
    get_tools_prompt_section,
)
from app.models.agent_run import AgentRun, AgentStep, AgentStepType, AgentStepStatus, AgentStepLog
from app.models.experiment import Experiment, Trial
from app.models.research_cycle import ResearchCycle, LabNotebookEntry
from app.models.project import Project, TaskType


class TestAgentHistoryToolsDefinitions:
    """Test tool definitions are properly structured."""

    def test_tools_list_not_empty(self):
        """Verify tools list is populated."""
        assert len(AGENT_HISTORY_TOOLS) > 0

    def test_all_tools_have_required_fields(self):
        """Verify each tool has name, description, and parameters."""
        for tool in AGENT_HISTORY_TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing 'description'"
            assert "parameters" in tool, f"Tool {tool.get('name')} missing 'parameters'"

    def test_tool_parameters_have_type(self):
        """Verify each tool's parameters has a type field."""
        for tool in AGENT_HISTORY_TOOLS:
            params = tool["parameters"]
            assert params.get("type") == "object", f"Tool {tool['name']} parameters not type 'object'"

    def test_expected_tools_exist(self):
        """Verify all expected tools are defined."""
        expected_tools = [
            "get_research_cycles",
            "get_agent_thinking",
            "get_experiment_results",
            "get_robustness_audit",
            "get_notebook_entries",
            "get_failed_experiments",
            "get_best_models",
            "get_dataset_designs",
            "search_project_history",
        ]
        tool_names = [t["name"] for t in AGENT_HISTORY_TOOLS]
        for expected in expected_tools:
            assert expected in tool_names, f"Expected tool '{expected}' not found"


class TestAgentToolExecutor:
    """Test the AgentToolExecutor class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock(spec=Session)

    @pytest.fixture
    def project_id(self):
        """Generate a test project ID."""
        return uuid4()

    @pytest.fixture
    def executor(self, mock_db, project_id):
        """Create an AgentToolExecutor instance."""
        return AgentToolExecutor(
            db=mock_db,
            project_id=project_id,
            current_cycle_id=None
        )

    def test_execute_unknown_tool(self, executor):
        """Test executing an unknown tool returns error."""
        result = executor.execute_tool("unknown_tool", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execute_tool_catches_exceptions(self, executor, mock_db):
        """Test that tool execution catches and returns exceptions."""
        # Make the query raise an exception
        mock_db.query.side_effect = Exception("Database error")

        result = executor.execute_tool("get_research_cycles", {})
        assert "error" in result
        assert "Database error" in result["error"]


class TestGetResearchCyclesToolDB:
    """Test get_research_cycles tool with database fixtures."""

    @pytest.fixture
    def project(self, db_session):
        """Create a test project."""
        project = Project(
            id=uuid4(),
            name="Test Project",
            description="Test description",
            task_type=TaskType.BINARY,
        )
        db_session.add(project)
        db_session.commit()
        return project

    @pytest.fixture
    def research_cycles(self, db_session, project):
        """Create test research cycles."""
        from app.models.research_cycle import ResearchCycleStatus
        cycles = []
        for i in range(3):
            cycle = ResearchCycle(
                id=uuid4(),
                project_id=project.id,
                sequence_number=i + 1,
                status=ResearchCycleStatus.COMPLETED if i < 2 else ResearchCycleStatus.RUNNING,
                summary_title=f"Cycle {i + 1} Title",
            )
            db_session.add(cycle)
            cycles.append(cycle)
        db_session.commit()
        return cycles

    def test_get_research_cycles_returns_cycles(self, db_session, project, research_cycles):
        """Test that get_research_cycles returns cycles from database."""
        executor = AgentToolExecutor(db=db_session, project_id=project.id)

        result = executor.execute_tool("get_research_cycles", {"limit": 10})

        assert "cycles" in result
        assert len(result["cycles"]) == 3
        # Should be ordered by sequence_number descending
        assert result["cycles"][0]["sequence_number"] == 3

    def test_get_research_cycles_respects_limit(self, db_session, project, research_cycles):
        """Test that limit parameter is respected."""
        executor = AgentToolExecutor(db=db_session, project_id=project.id)

        result = executor.execute_tool("get_research_cycles", {"limit": 2})

        assert len(result["cycles"]) == 2

    def test_get_research_cycles_marks_current(self, db_session, project, research_cycles):
        """Test that current cycle is marked."""
        current_cycle = research_cycles[2]  # The running one
        executor = AgentToolExecutor(
            db=db_session,
            project_id=project.id,
            current_cycle_id=current_cycle.id
        )

        result = executor.execute_tool("get_research_cycles", {})

        # Find the current cycle in results
        current_in_results = next(
            (c for c in result["cycles"] if c["id"] == str(current_cycle.id)),
            None
        )
        assert current_in_results is not None
        assert current_in_results["is_current"] is True


class TestGetNotebookEntriesToolDB:
    """Test get_notebook_entries tool with database fixtures."""

    @pytest.fixture
    def project(self, db_session):
        """Create a test project."""
        project = Project(
            id=uuid4(),
            name="Test Project",
            description="Test description",
            task_type=TaskType.BINARY,
        )
        db_session.add(project)
        db_session.commit()
        return project

    @pytest.fixture
    def notebook_entries(self, db_session, project):
        """Create test notebook entries."""
        entries = []
        for i, (author, title) in enumerate([
            ("human", "Feature X causes leakage"),
            ("agent", "Correlation analysis complete"),
            ("human", "Try regularization"),
        ]):
            entry = LabNotebookEntry(
                id=uuid4(),
                project_id=project.id,
                author_type=author,
                title=title,
                body_markdown=f"Content for {title}",
            )
            db_session.add(entry)
            entries.append(entry)
        db_session.commit()
        return entries

    def test_get_notebook_entries_returns_entries(self, db_session, project, notebook_entries):
        """Test that notebook entries are returned."""
        executor = AgentToolExecutor(db=db_session, project_id=project.id)

        result = executor.execute_tool("get_notebook_entries", {})

        assert "entries" in result
        assert len(result["entries"]) == 3

    def test_get_notebook_entries_filters_by_author(self, db_session, project, notebook_entries):
        """Test filtering by author type."""
        executor = AgentToolExecutor(db=db_session, project_id=project.id)

        result = executor.execute_tool("get_notebook_entries", {"author_type": "human"})

        assert len(result["entries"]) == 2
        for entry in result["entries"]:
            assert entry["author_type"] == "human"

    def test_get_notebook_entries_search(self, db_session, project, notebook_entries):
        """Test search functionality."""
        executor = AgentToolExecutor(db=db_session, project_id=project.id)

        result = executor.execute_tool("get_notebook_entries", {"search_query": "leakage"})

        assert len(result["entries"]) == 1
        assert "leakage" in result["entries"][0]["title"].lower()


class TestGetToolsPromptSection:
    """Test the tools prompt section generator."""

    def test_returns_string(self):
        """Test that get_tools_prompt_section returns a string."""
        result = get_tools_prompt_section()
        assert isinstance(result, str)

    def test_contains_tool_names(self):
        """Test that the prompt contains tool names."""
        result = get_tools_prompt_section()
        assert "get_research_cycles" in result
        assert "get_agent_thinking" in result
        assert "get_robustness_audit" in result

    def test_contains_usage_instructions(self):
        """Test that the prompt contains usage instructions."""
        result = get_tools_prompt_section()
        assert "IMPORTANT" in result or "MUST" in result



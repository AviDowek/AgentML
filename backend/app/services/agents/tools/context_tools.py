"""Context tools for agents to fetch information they need.

These tools allow agents to be self-sufficient - they can request
the context and resources they need rather than relying on the
orchestrator to pass everything correctly.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import (
    AgentRun,
    AgentStep,
    AgentStepStatus,
    AgentStepType,
    DataSource,
    Project,
)
from app.services.agent_service import build_schema_summary

logger = logging.getLogger(__name__)


# Tool definitions for LLM function calling
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_project_context",
            "description": "Get information about the current project including its name, description, and settings. Call this when you need to understand the project context.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_source_info",
            "description": "Get detailed information about a data source including its schema, column statistics, and sample data. Call this when you need to understand the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_source_id": {
                        "type": "string",
                        "description": "Optional specific data source ID. If not provided, returns info for all data sources in the project.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agent_output",
            "description": "Get the output from a previous agent step. Use this to access results from other agents in the pipeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_type": {
                        "type": "string",
                        "enum": [
                            "data_analysis",
                            "problem_understanding",
                            "data_audit",
                            "dataset_design",
                            "experiment_design",
                            "plan_critic",
                        ],
                        "description": "The type of agent step to get output from.",
                    },
                },
                "required": ["step_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_state",
            "description": "Get the current state of the pipeline including which steps have completed and their accumulated outputs.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_goal",
            "description": "Get the user's original goal description for this ML project. This contains what the user wants to achieve.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


class AgentToolExecutor:
    """Executes tool calls made by agents.

    This class handles the actual data fetching when an agent
    makes a tool call to get context or resources.
    """

    def __init__(
        self,
        db: Session,
        agent_run: AgentRun,
        project_id: Optional[UUID] = None,
    ):
        """Initialize the tool executor.

        Args:
            db: Database session
            agent_run: The current agent run
            project_id: Optional project ID (uses agent_run.project_id if not provided)
        """
        self.db = db
        self.agent_run = agent_run
        self.project_id = project_id or agent_run.project_id

        # Cache for frequently accessed data
        self._project_cache: Optional[Project] = None
        self._data_sources_cache: Optional[List[DataSource]] = None

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments passed to the tool

        Returns:
            Tool execution result as a dict
        """
        logger.info(f"Agent tool call: {tool_name} with args: {arguments}")

        if tool_name == "get_project_context":
            return self.get_project_context()
        elif tool_name == "get_data_source_info":
            return self.get_data_source_info(arguments.get("data_source_id"))
        elif tool_name == "get_agent_output":
            return self.get_agent_output(arguments.get("step_type"))
        elif tool_name == "get_pipeline_state":
            return self.get_pipeline_state()
        elif tool_name == "get_user_goal":
            return self.get_user_goal()
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

    def get_project_context(self) -> Dict[str, Any]:
        """Get project context information."""
        if self._project_cache is None:
            self._project_cache = self.db.query(Project).filter(
                Project.id == self.project_id
            ).first()

        project = self._project_cache
        if not project:
            return {"error": "Project not found"}

        return {
            "project_id": str(project.id),
            "name": project.name,
            "description": project.description or "",
            "created_at": project.created_at.isoformat() if project.created_at else None,
        }

    def get_data_source_info(self, data_source_id: Optional[str] = None) -> Dict[str, Any]:
        """Get data source information including schema."""
        if data_source_id:
            # Get specific data source
            data_source = self.db.query(DataSource).filter(
                DataSource.id == data_source_id
            ).first()

            if not data_source:
                return {"error": f"Data source not found: {data_source_id}"}

            return self._format_data_source(data_source)
        else:
            # Get all data sources for the project
            if self._data_sources_cache is None:
                self._data_sources_cache = self.db.query(DataSource).filter(
                    DataSource.project_id == self.project_id
                ).all()

            return {
                "data_sources": [
                    self._format_data_source(ds) for ds in self._data_sources_cache
                ]
            }

    def _format_data_source(self, data_source: DataSource) -> Dict[str, Any]:
        """Format a data source for tool response."""
        result = {
            "data_source_id": str(data_source.id),
            "name": data_source.name,
            "file_type": data_source.file_type,
            "row_count": data_source.row_count,
            "column_count": data_source.column_count,
        }

        # Include schema summary if available
        if data_source.schema_summary:
            try:
                schema_summary = build_schema_summary(
                    data_source_id=str(data_source.id),
                    data_source_name=data_source.name,
                    analysis_result=data_source.schema_summary,
                )
                result["schema_summary"] = schema_summary.model_dump(mode="json")
            except Exception as e:
                logger.warning(f"Could not build schema summary: {e}")
                result["schema_summary_raw"] = data_source.schema_summary

        return result

    def get_agent_output(self, step_type: str) -> Dict[str, Any]:
        """Get output from a previous agent step."""
        if not step_type:
            return {"error": "step_type is required"}

        try:
            step_type_enum = AgentStepType(step_type)
        except ValueError:
            return {"error": f"Invalid step type: {step_type}"}

        # Find completed step of this type in current run
        step = self.db.query(AgentStep).filter(
            AgentStep.agent_run_id == self.agent_run.id,
            AgentStep.step_type == step_type_enum,
            AgentStep.status == AgentStepStatus.COMPLETED,
        ).order_by(AgentStep.finished_at.desc()).first()

        if not step:
            return {
                "found": False,
                "message": f"No completed {step_type} step found in this pipeline run.",
            }

        return {
            "found": True,
            "step_id": str(step.id),
            "step_type": step.step_type.value,
            "status": step.status.value,
            "output": step.output_json or {},
            "finished_at": step.finished_at.isoformat() if step.finished_at else None,
        }

    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state."""
        # Get all steps in this run
        steps = self.db.query(AgentStep).filter(
            AgentStep.agent_run_id == self.agent_run.id
        ).all()

        # Group by status
        completed_steps = []
        pending_steps = []
        running_steps = []
        failed_steps = []

        accumulated_outputs = {}

        for step in steps:
            step_info = {
                "step_type": step.step_type.value,
                "status": step.status.value,
            }

            if step.status == AgentStepStatus.COMPLETED:
                completed_steps.append(step_info)
                # Accumulate outputs
                if step.output_json:
                    accumulated_outputs[step.step_type.value] = step.output_json
            elif step.status == AgentStepStatus.PENDING:
                pending_steps.append(step_info)
            elif step.status == AgentStepStatus.RUNNING:
                running_steps.append(step_info)
            elif step.status == AgentStepStatus.FAILED:
                failed_steps.append({
                    **step_info,
                    "error": step.error_message,
                })

        return {
            "run_id": str(self.agent_run.id),
            "run_status": self.agent_run.status.value,
            "completed_steps": completed_steps,
            "pending_steps": pending_steps,
            "running_steps": running_steps,
            "failed_steps": failed_steps,
            "accumulated_outputs": accumulated_outputs,
        }

    def get_user_goal(self) -> Dict[str, Any]:
        """Get the user's original goal description."""
        # First try to get from agent_run config
        config = self.agent_run.config_json or {}
        description = config.get("description", "")

        # Fallback to project description
        if not description:
            if self._project_cache is None:
                self._project_cache = self.db.query(Project).filter(
                    Project.id == self.project_id
                ).first()

            if self._project_cache:
                description = self._project_cache.description or ""

        # Fallback to first step's input
        if not description:
            first_step = self.db.query(AgentStep).filter(
                AgentStep.agent_run_id == self.agent_run.id,
                AgentStep.step_type == AgentStepType.DATA_ANALYSIS,
            ).first()

            if first_step and first_step.input_json:
                description = first_step.input_json.get("description", "")

        return {
            "goal_description": description,
            "source": "config" if config.get("description") else "fallback",
        }


# Convenience functions for direct use (without tool executor)
def get_project_context(db: Session, project_id: UUID) -> Dict[str, Any]:
    """Get project context directly."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return {"error": "Project not found"}

    return {
        "project_id": str(project.id),
        "name": project.name,
        "description": project.description or "",
    }


def get_data_source_info(
    db: Session,
    project_id: UUID,
    data_source_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get data source info directly."""
    if data_source_id:
        data_source = db.query(DataSource).filter(
            DataSource.id == data_source_id
        ).first()

        if not data_source:
            return {"error": f"Data source not found: {data_source_id}"}

        result = {
            "data_source_id": str(data_source.id),
            "name": data_source.name,
            "file_type": data_source.file_type,
        }

        if data_source.schema_summary:
            try:
                schema_summary = build_schema_summary(
                    data_source_id=str(data_source.id),
                    data_source_name=data_source.name,
                    analysis_result=data_source.schema_summary,
                )
                result["schema_summary"] = schema_summary.model_dump(mode="json")
            except Exception:
                result["schema_summary_raw"] = data_source.schema_summary

        return result
    else:
        data_sources = db.query(DataSource).filter(
            DataSource.project_id == project_id
        ).all()

        return {
            "data_sources": [
                {"data_source_id": str(ds.id), "name": ds.name}
                for ds in data_sources
            ]
        }


def get_agent_output(
    db: Session,
    agent_run_id: UUID,
    step_type: str
) -> Dict[str, Any]:
    """Get agent output directly."""
    try:
        step_type_enum = AgentStepType(step_type)
    except ValueError:
        return {"error": f"Invalid step type: {step_type}"}

    step = db.query(AgentStep).filter(
        AgentStep.agent_run_id == agent_run_id,
        AgentStep.step_type == step_type_enum,
        AgentStep.status == AgentStepStatus.COMPLETED,
    ).order_by(AgentStep.finished_at.desc()).first()

    if not step:
        return {"found": False}

    return {
        "found": True,
        "output": step.output_json or {},
    }


def get_pipeline_state(db: Session, agent_run_id: UUID) -> Dict[str, Any]:
    """Get pipeline state directly."""
    steps = db.query(AgentStep).filter(
        AgentStep.agent_run_id == agent_run_id
    ).all()

    return {
        "completed": [s.step_type.value for s in steps if s.status == AgentStepStatus.COMPLETED],
        "pending": [s.step_type.value for s in steps if s.status == AgentStepStatus.PENDING],
    }


def get_user_goal(db: Session, agent_run: AgentRun) -> Dict[str, Any]:
    """Get user goal directly."""
    config = agent_run.config_json or {}
    description = config.get("description", "")

    if not description and agent_run.project_id:
        project = db.query(Project).filter(Project.id == agent_run.project_id).first()
        if project:
            description = project.description or ""

    return {"goal_description": description}

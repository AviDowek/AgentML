"""Base agent class for all ML pipeline agents."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar

from sqlalchemy.orm import Session

from app.models import AgentRun, AgentStep, AgentStepType
from app.services.llm_client import BaseLLMClient

# Import will be updated after extraction
# For now, import from agent_executor to avoid circular imports during migration
from app.services.agents.utils.step_logger import StepLogger

T = TypeVar('T')
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all pipeline agents.

    Each agent handles a specific step in the ML experiment workflow.
    Agents receive inputs from step.input_json and return outputs
    that become step.output_json.

    Agents can use tools to fetch context they need:
        - get_project_context: Get project name, description
        - get_data_source_info: Get data source schema and info
        - get_agent_output: Get output from another agent step
        - get_pipeline_state: Get current pipeline state
        - get_user_goal: Get the user's ML goal description

    Subclasses must implement:
        - name: A short identifier for the agent
        - step_type: The AgentStepType this agent handles
        - execute(): The main execution logic

    Example:
        class MyAgent(BaseAgent):
            name = "my_agent"
            step_type = AgentStepType.MY_STEP
            uses_tools = True  # Enable tool calling

            async def execute(self) -> Dict[str, Any]:
                # Agent can call tools to get what it needs
                result = await self.chat_with_tools(messages, schema)
                return {"result": result}
    """

    # Class attributes to be set by subclasses
    name: str = "base_agent"
    step_type: Optional[AgentStepType] = None
    uses_tools: bool = False  # Set to True to enable tool calling

    def __init__(
        self,
        db: Session,
        step: AgentStep,
        step_logger: StepLogger,
        llm_client: BaseLLMClient,
    ):
        """Initialize the agent.

        Args:
            db: Database session for queries
            step: The AgentStep being executed
            step_logger: Logger for step progress/output
            llm_client: Client for LLM interactions
        """
        self.db = db
        self.step = step
        self.logger = step_logger
        self.llm = llm_client
        self.input_data = step.input_json or {}

        # Initialize tool executor if agent uses tools
        self._tool_executor = None
        if self.uses_tools:
            self._init_tool_executor()

    def _init_tool_executor(self) -> None:
        """Initialize the tool executor for this agent."""
        from app.services.agents.tools import AgentToolExecutor, AGENT_TOOLS

        # Get the agent run for context
        agent_run = self.db.query(AgentRun).filter(
            AgentRun.id == self.step.agent_run_id
        ).first()

        if agent_run:
            self._tool_executor = AgentToolExecutor(
                db=self.db,
                agent_run=agent_run,
            )
            self._available_tools = AGENT_TOOLS
        else:
            logger.warning(f"Could not initialize tool executor: agent_run not found")
            self._available_tools = []

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        response_schema: Optional[Any] = None,
        max_tool_calls: int = 10,
    ) -> Dict[str, Any]:
        """Make an LLM call that can use tools to fetch context.

        This method handles the tool call loop:
        1. Send messages to LLM with available tools
        2. If LLM returns tool calls, execute them
        3. Add tool results to messages and repeat
        4. When LLM returns final response, parse and return

        Args:
            messages: The conversation messages
            response_schema: Optional Pydantic model for structured output
            max_tool_calls: Maximum number of tool call rounds

        Returns:
            The final response from the LLM
        """
        if not self._tool_executor:
            # No tools available, just make a regular call
            if response_schema:
                return await self.llm.chat_json(messages, response_schema)
            else:
                response = await self.llm.chat(messages)
                return {"response": response}

        # Make a copy of messages to avoid mutating the original
        conversation = list(messages)
        tool_call_count = 0

        while tool_call_count < max_tool_calls:
            # Make LLM call with tools
            response = await self.llm.chat_with_tools(
                messages=conversation,
                tools=self._available_tools,
            )

            # Check if response contains tool calls
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                # No tool calls - this is the final response
                content = response.get("content", "")

                # Try to parse as JSON if we have a schema
                if response_schema and content:
                    try:
                        parsed = json.loads(content)
                        return parsed
                    except json.JSONDecodeError:
                        # If parsing fails, return as-is
                        return {"response": content, "parse_error": True}

                return {"response": content}

            # Execute tool calls
            tool_call_count += 1
            self.logger.thinking(f"Executing {len(tool_calls)} tool call(s)...")

            # Add assistant message with tool calls
            conversation.append({
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": tool_calls,
            })

            # Execute each tool and add results
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                tool_call_id = tool_call.get("id", "")

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                self.logger.thinking(f"Tool: {tool_name}({tool_args})")

                # Execute the tool
                result = self._tool_executor.execute_tool(tool_name, tool_args)

                # Add tool result to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result),
                })

        # Max tool calls reached
        self.logger.warning(f"Max tool calls ({max_tool_calls}) reached")
        return {"error": "Max tool calls reached", "partial": True}

    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute the agent's task.

        This method should:
        1. Read inputs from self.input_data
        2. Perform the agent's work (LLM calls, data processing, etc.)
        3. Log progress using self.logger
        4. Return a dict that will become step.output_json

        Returns:
            Dict containing the agent's output

        Raises:
            ValueError: For missing required inputs
            Other exceptions as appropriate for the agent
        """
        pass

    def get_input(self, key: str, default: T = None) -> T:
        """Get an input value with an optional default.

        Args:
            key: The input key to look up
            default: Value to return if key is not found

        Returns:
            The input value or the default
        """
        return self.input_data.get(key, default)

    def require_input(self, key: str) -> Any:
        """Get a required input value.

        Args:
            key: The input key to look up

        Returns:
            The input value

        Raises:
            ValueError: If the key is not found
        """
        if key not in self.input_data:
            raise ValueError(f"Missing required input: '{key}'")
        return self.input_data[key]

    def get_nested_input(self, *keys: str, default: T = None) -> T:
        """Get a nested input value using dot notation.

        Args:
            *keys: Keys to traverse (e.g., "audit_details", "leakage_candidates")
            default: Value to return if path is not found

        Returns:
            The nested value or the default
        """
        value = self.input_data
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key)
            if value is None:
                return default
        return value

    @property
    def project_id(self) -> Optional[str]:
        """Get the project ID from input if available."""
        return self.get_input("project_id")

    @property
    def experiment_id(self) -> Optional[str]:
        """Get the experiment ID from input if available."""
        return self.get_input("experiment_id")

    @property
    def data_source_id(self) -> Optional[str]:
        """Get the data source ID from input if available."""
        return self.get_input("data_source_id")

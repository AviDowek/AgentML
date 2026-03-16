"""Project Manager Agent - Orchestrates dynamic agent flow in pipelines.

The Project Manager (PM) agent controls the flow of the pipeline dynamically,
deciding which agent should run next based on the current state and outputs.
Unlike sequential pipelines, agents can run multiple times and in any order.

The PM declares when the pipeline is complete.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models import AgentStep, AgentStepType, AgentStepStatus
from app.services.agents.base import BaseAgent
from app.services.agents.utils.step_logger import StepLogger
from app.services.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class PMDecision(BaseModel):
    """Pydantic model for Project Manager decisions."""
    next_agent: Optional[Literal[
        "data_analysis", "problem_understanding", "data_audit",
        "dataset_design", "dataset_validation", "experiment_design", "plan_critic"
    ]] = Field(
        None,
        description="Name of the next agent to run, or null if pipeline is complete"
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation of why this decision was made"
    )
    is_complete: bool = Field(
        ...,
        description="Whether the pipeline has achieved its goals"
    )
    completion_summary: Optional[str] = Field(
        None,
        description="If complete, a summary of what was accomplished"
    )
    additional_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Extra context or instructions for the next agent"
    )
    needs_revision: Optional[bool] = Field(
        None,
        description="Whether a previous agent's output needs revision"
    )
    revision_target: Optional[str] = Field(
        None,
        description="Which agent's output needs revision"
    )
    revision_feedback: Optional[str] = Field(
        None,
        description="Feedback for the revision"
    )


# Available agents that PM can orchestrate
ORCHESTRABLE_AGENTS = {
    "data_analysis": AgentStepType.DATA_ANALYSIS,
    "problem_understanding": AgentStepType.PROBLEM_UNDERSTANDING,
    "data_audit": AgentStepType.DATA_AUDIT,
    "dataset_design": AgentStepType.DATASET_DESIGN,
    "dataset_validation": AgentStepType.DATASET_VALIDATION,
    "experiment_design": AgentStepType.EXPERIMENT_DESIGN,
    "plan_critic": AgentStepType.PLAN_CRITIC,
}

# Agent descriptions for the PM to understand capabilities
AGENT_DESCRIPTIONS = {
    "data_analysis": "Analyzes data sources to determine ML suitability, suggests targets, and assesses quality",
    "problem_understanding": "Determines task type (classification/regression), target column, and primary metric",
    "data_audit": "Examines data quality: missing values, distributions, outliers, leakage candidates",
    "dataset_design": "Decides which columns to include/exclude and what transformations to apply",
    "dataset_validation": "Validates that columns in the dataset design actually exist in the data source (catches AI hallucinations)",
    "experiment_design": "Creates experiment configurations with time budgets, validation strategies",
    "plan_critic": "Reviews and validates the entire plan, checking for issues before execution",
}


class ProjectManagerAgent(BaseAgent):
    """Project Manager Agent that orchestrates dynamic pipeline flow.

    The PM analyzes each agent's output and decides:
    1. Which agent should run next
    2. Whether to re-run an agent with new information
    3. When the pipeline is complete

    This enables non-sequential, iterative pipelines where agents can
    be called multiple times based on evolving understanding.
    """

    name = "project_manager"
    step_type = AgentStepType.PROJECT_MANAGER

    async def execute(self) -> Dict[str, Any]:
        """Execute the Project Manager's decision-making logic.

        Inputs:
            - pipeline_state: Current state of all agents and their outputs
            - last_agent_output: Output from the most recently completed agent
            - execution_history: List of agents already executed
            - goal: The user's original ML goal description

        Returns:
            Dict with:
            - next_agent: The agent to run next (or None if complete)
            - reasoning: Explanation of the decision
            - is_complete: Whether the pipeline is complete
            - additional_context: Any extra context for the next agent
        """
        self.logger.thinking("Analyzing pipeline state to determine next action...")

        # Get inputs
        pipeline_state = self.get_input("pipeline_state", {})
        last_agent = self.get_input("last_agent", None)
        last_agent_output = self.get_input("last_agent_output", {})
        execution_history = self.get_input("execution_history", [])
        goal = self.get_input("goal", "")
        accumulated_outputs = self.get_input("accumulated_outputs", {})

        # Build the prompt for the PM
        prompt = self._build_decision_prompt(
            pipeline_state=pipeline_state,
            last_agent=last_agent,
            last_agent_output=last_agent_output,
            execution_history=execution_history,
            goal=goal,
            accumulated_outputs=accumulated_outputs,
        )

        self.logger.thinking(f"Execution history: {execution_history}")
        self.logger.thinking(f"Last agent: {last_agent}")

        # Get the PM's decision
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        decision = await self.llm.chat_json(
            messages=messages,
            response_schema=PMDecision,
        )

        # Log the decision
        if decision.get("is_complete"):
            self.logger.summary(f"Pipeline complete: {decision.get('completion_summary', 'All goals achieved')}")
        else:
            next_agent = decision.get("next_agent")
            self.logger.action(f"Next agent: {next_agent}")
            self.logger.thinking(decision.get("reasoning", ""))

            if decision.get("needs_revision"):
                self.logger.warning(f"Revision needed for: {decision.get('revision_target')}")

        return decision

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Project Manager."""
        agent_list = "\n".join([
            f"- {name}: {desc}"
            for name, desc in AGENT_DESCRIPTIONS.items()
        ])

        return f"""You are the Project Manager (PM) agent responsible for orchestrating an ML pipeline.

Your role is to:
1. Analyze the current state of the pipeline
2. Review outputs from completed agents
3. Decide which agent should run next
4. Determine when the pipeline is complete

Available agents:
{agent_list}

Key principles:
- Agents can run multiple times if needed (e.g., re-audit data after design changes)
- The pipeline is complete when all necessary information is gathered for experiment execution

PIPELINE ORDER (follow this sequence, moving FORWARD after each step):
1. data_analysis → THEN move to problem_understanding
2. problem_understanding → THEN move to dataset_design
3. dataset_design → THEN move to data_audit (DO NOT repeat dataset_design!)
4. data_audit → THEN move to dataset_validation
5. dataset_validation → If valid, move to experiment_design. If invalid, handle errors.
6. experiment_design → THEN move to plan_critic
7. plan_critic → If approved, pipeline is COMPLETE

CRITICAL: After each step completes successfully, MOVE TO THE NEXT STEP.
Do NOT repeat the same step unless there's a validation failure that requires it.

CRITICAL ROUTING RULES (check in this EXACT order):

1. **MISSING TARGET** (highest priority): If dataset_validation output has `missing_target` set to ANY value:
   → FIRST check: Did problem_understanding already run AFTER dataset_validation and provide a fix?
     - If target_creation is now set (target_exists=false), the fix is ready → go to dataset_design
     - If target_column changed to a valid column, the fix is ready → go to dataset_design
   → ONLY re-run problem_understanding if it hasn't been run since the validation failure
   → Include the available_columns in additional_context so problem_understanding knows valid options

2. **MISSING FEATURES** (second priority): If dataset_validation output has `missing_features` but target is valid:
   → Re-run **dataset_design** with available_columns feedback

3. **INVALID COLUMNS IN EXPERIMENT**: If plan_critic finds invalid column references:
   → Re-run **experiment_design** with feedback

LOOK FOR THESE SPECIFIC FIELDS IN accumulated outputs:
- "missing_target": "column_name" from dataset_validation - means target needs fixing
- "target_creation": object from problem_understanding - means FIX IS READY, go to dataset_design!
- "missing_features": ["col1", "col2"] - means features are wrong → dataset_design
- "is_valid": false - check WHY it's false using the above fields

LOOP PREVENTION: If you see BOTH missing_target AND target_creation in accumulated outputs:
- target_creation means problem_understanding already provided the fix
- DO NOT re-run problem_understanding - go to dataset_design instead!

ENGINEERED TARGETS: Targets CAN be engineered if problem_understanding provides:
- target_exists: false
- target_creation: object with "formula", "source_columns", "description" fields
The source_columns must exist in the raw data.

Engineered features are ALLOWED - just verify their source columns exist.
The validation output contains 'available_columns' which lists all valid column names.

Make decisions based on what information is available and what is still needed."""

    def _build_decision_prompt(
        self,
        pipeline_state: Dict[str, Any],
        last_agent: Optional[str],
        last_agent_output: Dict[str, Any],
        execution_history: List[str],
        goal: str,
        accumulated_outputs: Dict[str, Any],
    ) -> str:
        """Build the decision prompt for the PM."""

        # Format execution history
        history_str = ", ".join(execution_history) if execution_history else "None (pipeline just started)"

        # Format last agent output (truncate if too long)
        if last_agent_output:
            output_str = str(last_agent_output)[:2000]
            if len(str(last_agent_output)) > 2000:
                output_str += "... (truncated)"
        else:
            output_str = "None"

        # Format accumulated outputs summary
        accumulated_summary = []
        for key, value in accumulated_outputs.items():
            if isinstance(value, dict):
                accumulated_summary.append(f"- {key}: {list(value.keys())}")
            elif isinstance(value, list):
                accumulated_summary.append(f"- {key}: [{len(value)} items]")
            else:
                accumulated_summary.append(f"- {key}: {str(value)[:100]}")

        accumulated_str = "\n".join(accumulated_summary) if accumulated_summary else "None"

        return f"""## Current Pipeline State

**User's Goal:**
{goal}

**Execution History:**
{history_str}

**Last Agent Executed:**
{last_agent or "None"}

**Last Agent Output:**
{output_str}

**Accumulated Information:**
{accumulated_str}

## Your Task

Based on the current state:
1. Determine if the pipeline has enough information to be complete
2. If not complete, decide which agent should run next
3. Provide clear reasoning for your decision
4. If a previous output needs revision, specify which agent and why

Remember:
- The pipeline is complete when we have: analyzed data, defined problem, audited data, designed dataset, VALIDATED dataset, designed experiment, and validated plan
- Some agents may need to run again if new information emerges
- CRITICAL: Always run dataset_validation after dataset_design to catch hallucinated columns
- If dataset_validation shows "missing_target" AND problem_understanding hasn't run since → run problem_understanding
- If problem_understanding already ran after validation failure AND provided target_creation → go to dataset_design (don't loop!)
- If dataset_validation shows only "missing_features" (target is valid), route to dataset_design
- AVOID LOOPS: Check execution_history - if problem_understanding ran after dataset_validation, MOVE FORWARD to dataset_design
- Always move toward having a validated, ready-to-execute experiment plan"""

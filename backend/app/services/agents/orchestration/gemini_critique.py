"""Gemini Critique Agent - Provides expert critique for each pipeline step.

Each step in the pipeline has a corresponding Gemini critique agent that:
1. Reviews the main agent's output
2. Provides expert critique from the same domain
3. Engages in debate with the main agent (up to 3 rounds)
4. Attempts to reach consensus

If consensus is not reached, the OpenAI Judge decides.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.agents.utils.step_logger import StepLogger
from app.services.llm_client import BaseLLMClient, GeminiClient

logger = logging.getLogger(__name__)


# Critique specializations for each agent type
CRITIQUE_SPECIALIZATIONS = {
    AgentStepType.DATA_ANALYSIS: {
        "name": "Data Analysis Critic",
        "expertise": "data quality assessment, statistical analysis, ML suitability evaluation",
        "focus_areas": [
            "Data quality issues that may have been missed",
            "Alternative interpretations of the data characteristics",
            "Potential biases in the data assessment",
            "ML suitability concerns",
        ],
    },
    AgentStepType.PROBLEM_UNDERSTANDING: {
        "name": "Problem Definition Critic",
        "expertise": "ML problem formulation, task type selection, metric optimization",
        "focus_areas": [
            "Task type appropriateness",
            "Target column selection validity",
            "Metric alignment with business goals",
            "Potential alternative problem formulations",
        ],
    },
    AgentStepType.DATA_AUDIT: {
        "name": "Data Quality Critic",
        "expertise": "data quality, statistical testing, leakage detection, anomaly identification",
        "focus_areas": [
            "Undetected data quality issues",
            "Potential data leakage sources",
            "Distribution concerns",
            "Missing value impact assessment",
        ],
    },
    AgentStepType.DATASET_DESIGN: {
        "name": "Feature Engineering Critic",
        "expertise": "feature selection, feature engineering, dimensionality reduction",
        "focus_areas": [
            "Feature relevance and redundancy",
            "Missing important features",
            "Feature engineering opportunities",
            "Potential leakage in feature design",
        ],
    },
    AgentStepType.EXPERIMENT_DESIGN: {
        "name": "Experiment Design Critic",
        "expertise": "experimental methodology, validation strategies, AutoML configuration",
        "focus_areas": [
            "Validation strategy appropriateness",
            "Time budget adequacy",
            "Model selection considerations",
            "Hyperparameter space coverage",
        ],
    },
    AgentStepType.PLAN_CRITIC: {
        "name": "Plan Validation Critic",
        "expertise": "end-to-end ML pipeline validation, risk assessment",
        "focus_areas": [
            "Overall plan coherence",
            "Risk factors not addressed",
            "Execution feasibility",
            "Expected outcome realism",
        ],
    },
}


class GeminiCritiqueAgent(BaseAgent):
    """Gemini-based critique agent that reviews and debates main agent outputs.

    This agent:
    1. Reviews the output of a main agent
    2. Provides expert critique from a specialized perspective
    3. Can engage in multi-round debate
    4. Signals agreement or disagreement

    The critique agent uses Gemini to provide an alternative perspective,
    helping ensure robustness of the pipeline decisions.
    """

    name = "gemini_critique"
    step_type = AgentStepType.GEMINI_CRITIQUE

    def __init__(
        self,
        db: Session,
        step: "AgentStep",
        step_logger: StepLogger,
        llm_client: BaseLLMClient,
        gemini_client: Optional[GeminiClient] = None,
    ):
        """Initialize the critique agent.

        Args:
            db: Database session
            step: The AgentStep being executed
            step_logger: Logger for step progress
            llm_client: Main LLM client (used if Gemini not available)
            gemini_client: Gemini client for critique (preferred)
        """
        super().__init__(db, step, step_logger, llm_client)
        self.gemini_client = gemini_client or llm_client  # Fallback to main client

    async def execute(self) -> Dict[str, Any]:
        """Execute the critique.

        Inputs:
            - target_agent_type: The type of agent being critiqued
            - main_agent_output: The output from the main agent
            - debate_round: Current round (1-3)
            - previous_messages: Messages from previous debate rounds

        Returns:
            Dict with:
            - agrees: Whether the critique agent agrees
            - critique: The critique or response
            - reasoning: Detailed reasoning
            - suggestions: Specific suggestions for improvement
            - confidence: Confidence level in the critique (0-100)
        """
        target_agent_type = self.get_input("target_agent_type")
        main_agent_output = self.get_input("main_agent_output", {})
        debate_round = self.get_input("debate_round", 1)
        previous_messages = self.get_input("previous_messages", [])

        # Get specialization for this agent type
        try:
            agent_type_enum = AgentStepType(target_agent_type)
            specialization = CRITIQUE_SPECIALIZATIONS.get(agent_type_enum, {
                "name": "General Critic",
                "expertise": "ML pipeline analysis",
                "focus_areas": ["General quality assessment"],
            })
        except ValueError:
            specialization = {
                "name": "General Critic",
                "expertise": "ML pipeline analysis",
                "focus_areas": ["General quality assessment"],
            }

        self.logger.thinking(f"Round {debate_round}: Critiquing {target_agent_type} as {specialization['name']}")

        # Build the critique prompt
        if debate_round == 1:
            # Initial critique
            result = await self._initial_critique(
                target_agent_type=target_agent_type,
                main_agent_output=main_agent_output,
                specialization=specialization,
            )
        else:
            # Response in ongoing debate
            result = await self._debate_response(
                target_agent_type=target_agent_type,
                main_agent_output=main_agent_output,
                specialization=specialization,
                previous_messages=previous_messages,
                debate_round=debate_round,
            )

        # Log the result
        if result.get("agrees"):
            self.logger.summary(f"Agreement reached: {result.get('reasoning', '')[:200]}")
        else:
            self.logger.hypothesis(f"Disagreement: {result.get('critique', '')[:200]}")

        return result

    async def _initial_critique(
        self,
        target_agent_type: str,
        main_agent_output: Dict[str, Any],
        specialization: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Provide initial critique of the main agent's output."""

        system_prompt = f"""You are {specialization['name']}, an expert in {specialization['expertise']}.

Your role is to critically review the output of another AI agent and provide expert feedback.
You are NOT trying to be difficult - you are trying to ensure the best possible outcome.

Focus areas for your critique:
{chr(10).join('- ' + area for area in specialization['focus_areas'])}

Guidelines:
- Be constructive and specific
- If the output is good, acknowledge it and agree
- If you have concerns, explain them clearly with reasoning
- Suggest specific improvements when disagreeing
- Consider edge cases and potential issues
- Be fair - don't disagree just to disagree

You can AGREE if:
- The output is well-reasoned and addresses key concerns
- Any issues are minor and don't affect the core conclusions
- The approach is sound even if you might have small suggestions

You should DISAGREE if:
- There are significant issues that could lead to problems
- Important considerations were missed
- The reasoning has logical flaws
- The conclusions don't follow from the analysis"""

        # Format the output for review
        output_str = self._format_output_for_review(main_agent_output)

        user_prompt = f"""Please review the following output from the {target_agent_type} agent:

{output_str}

Provide your critique. Remember to:
1. Assess the overall quality and completeness
2. Identify any issues or concerns
3. Note strengths and weaknesses
4. Decide if you agree with the output or have significant concerns"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return await self.gemini_client.chat_json(
            messages=messages,
            response_schema={
                "type": "object",
                "properties": {
                    "agrees": {
                        "type": "boolean",
                        "description": "Whether you agree with the output overall",
                    },
                    "critique": {
                        "type": "string",
                        "description": "Your detailed critique or endorsement",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning behind your assessment",
                    },
                    "key_concerns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific concerns (empty if agreeing)",
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific suggestions for improvement",
                    },
                    "confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Your confidence in this assessment (0-100)",
                    },
                    "strengths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Strengths of the output",
                    },
                },
                "required": ["agrees", "critique", "reasoning", "confidence"],
            },
        )

    async def _debate_response(
        self,
        target_agent_type: str,
        main_agent_output: Dict[str, Any],
        specialization: Dict[str, Any],
        previous_messages: List[Dict[str, str]],
        debate_round: int,
    ) -> Dict[str, Any]:
        """Respond in an ongoing debate."""

        system_prompt = f"""You are {specialization['name']}, an expert in {specialization['expertise']}.

You are in round {debate_round} of a debate about an agent's output.
Review the previous discussion and either:
1. Agree if the other side has addressed your concerns
2. Maintain your position if concerns remain unaddressed
3. Find middle ground if possible

This is round {debate_round} of 3 maximum rounds. If you still disagree after this,
a judge will make the final decision.

Be reasonable - if your concerns have been adequately addressed, acknowledge this.
But don't concede on important issues just to end the debate."""

        # Build conversation history
        conversation = [{"role": "system", "content": system_prompt}]

        for msg in previous_messages:
            conversation.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        # Add prompt for this round
        conversation.append({
            "role": "user",
            "content": f"Based on the discussion so far, provide your response for round {debate_round}.",
        })

        return await self.gemini_client.chat_json(
            messages=conversation,
            response_schema={
                "type": "object",
                "properties": {
                    "agrees": {
                        "type": "boolean",
                        "description": "Whether you now agree (concerns addressed)",
                    },
                    "critique": {
                        "type": "string",
                        "description": "Your response in this debate round",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why you agree or still disagree",
                    },
                    "remaining_concerns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Any remaining unaddressed concerns",
                    },
                    "conceded_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Points you now agree with",
                    },
                    "confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Your confidence in this position (0-100)",
                    },
                },
                "required": ["agrees", "critique", "reasoning", "confidence"],
            },
        )

    def _format_output_for_review(self, output: Dict[str, Any]) -> str:
        """Format agent output for critique review."""
        if not output:
            return "No output provided"

        # Try to format nicely
        import json
        try:
            return json.dumps(output, indent=2, default=str)[:4000]
        except Exception:
            return str(output)[:4000]

"""OpenAI Judge Agent - Makes final decisions when debate consensus isn't reached.

When the main agent and Gemini critique agent cannot reach consensus after
3 rounds of debate, the OpenAI Judge reviews all arguments and makes the
final decision.

The user can select which OpenAI model to use as the judge.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.agents.utils.step_logger import StepLogger
from app.services.llm_client import BaseLLMClient, OpenAIClient

logger = logging.getLogger(__name__)


# Available judge models (user can select)
AVAILABLE_JUDGE_MODELS = [
    # O-series reasoning models (deep thinking built-in)
    "o1",                  # O1 reasoning model
    "o1-mini",             # Smaller O1
    "o1-preview",          # O1 preview
    "o3",                  # O3 reasoning model
    "o3-mini",             # Smaller O3
    "o4-mini",             # O4 mini reasoning model
    # GPT-5.2 variants
    "gpt-5.2",             # Latest 5.2 model (no extended thinking)
    "gpt-5.2-thinking",    # 5.2 with high reasoning effort
    # GPT-5.1 variants
    "gpt-5.1",             # Base 5.1 model (no extended thinking)
    "gpt-5.1-thinking",    # 5.1 with high reasoning effort
    # GPT-4 series (legacy but still useful)
    "gpt-4.1",             # GPT-4.1 (previous generation)
    "gpt-4o",              # GPT-4o (optimized)
]

DEFAULT_JUDGE_MODEL = "o3"


class OpenAIJudgeAgent(BaseAgent):
    """OpenAI-based judge agent that makes final decisions in debates.

    This agent:
    1. Reviews the full debate transcript
    2. Evaluates arguments from both sides
    3. Makes an impartial final decision
    4. Provides detailed reasoning for the decision

    The judge is designed to be fair and focus on:
    - Technical correctness
    - Practical implications
    - Risk assessment
    - Overall benefit to the ML pipeline
    """

    name = "openai_judge"
    step_type = AgentStepType.OPENAI_JUDGE

    def __init__(
        self,
        db: Session,
        step: "AgentStep",
        step_logger: StepLogger,
        llm_client: BaseLLMClient,
        judge_model: Optional[str] = None,
        openai_client: Optional[OpenAIClient] = None,
    ):
        """Initialize the judge agent.

        Args:
            db: Database session
            step: The AgentStep being executed
            step_logger: Logger for step progress
            llm_client: Main LLM client (fallback)
            judge_model: Which OpenAI model to use as judge
            openai_client: OpenAI client for judging
        """
        super().__init__(db, step, step_logger, llm_client)
        self.judge_model = judge_model or DEFAULT_JUDGE_MODEL
        self.openai_client = openai_client or llm_client

    async def execute(self) -> Dict[str, Any]:
        """Execute the judging.

        Inputs:
            - target_agent_type: The type of agent whose output is being judged
            - main_agent_output: The final output from the main agent
            - critique_agent_output: The final position from the critique agent
            - debate_transcript: Full transcript of the debate rounds
            - main_agent_final_position: Main agent's final argument
            - critique_final_position: Critique agent's final argument

        Returns:
            Dict with:
            - decision: "main_agent" or "critique_agent" - whose position wins
            - final_output: The winning output (potentially modified)
            - reasoning: Detailed reasoning for the decision
            - key_factors: List of key factors in the decision
            - confidence: Confidence in the decision (0-100)
            - synthesis: Any synthesis of both positions (if applicable)
        """
        target_agent_type = self.get_input("target_agent_type")
        main_agent_output = self.get_input("main_agent_output", {})
        critique_agent_output = self.get_input("critique_agent_output", {})
        debate_transcript = self.get_input("debate_transcript", [])
        main_agent_final = self.get_input("main_agent_final_position", "")
        critique_final = self.get_input("critique_final_position", "")

        self.logger.thinking(f"Judging debate for {target_agent_type} using {self.judge_model}")

        # Build the judging prompt
        result = await self._render_judgment(
            target_agent_type=target_agent_type,
            main_agent_output=main_agent_output,
            critique_agent_output=critique_agent_output,
            debate_transcript=debate_transcript,
            main_agent_final=main_agent_final,
            critique_final=critique_final,
        )

        # Log the decision
        decision = result.get("decision", "unknown")
        confidence = result.get("confidence", 0)
        self.logger.summary(
            f"Judge decision: {decision} (confidence: {confidence}%)\n"
            f"Reasoning: {result.get('reasoning', '')[:300]}"
        )

        return result

    async def _render_judgment(
        self,
        target_agent_type: str,
        main_agent_output: Dict[str, Any],
        critique_agent_output: Dict[str, Any],
        debate_transcript: List[Dict[str, Any]],
        main_agent_final: str,
        critique_final: str,
    ) -> Dict[str, Any]:
        """Render final judgment on the debate."""

        system_prompt = """You are an impartial AI Judge evaluating a debate between two AI agents about a machine learning pipeline decision.

Your role is to:
1. Carefully review both positions and the full debate transcript
2. Evaluate the technical merits of each argument
3. Consider practical implications and risks
4. Make a fair, well-reasoned decision

Guidelines for judging:
- Focus on TECHNICAL CORRECTNESS first
- Consider PRACTICAL IMPLICATIONS for the ML pipeline
- Assess RISKS of each position
- Look for LOGICAL CONSISTENCY in arguments
- Value SPECIFIC EVIDENCE over general claims
- Consider EDGE CASES mentioned

You may also SYNTHESIZE the best elements of both positions if that produces a superior outcome.

Be fair and impartial. The goal is the best outcome for the ML pipeline, not declaring a "winner."
"""

        # Format the debate for review
        debate_summary = self._format_debate_transcript(debate_transcript)
        main_output_str = self._format_output(main_agent_output)
        critique_output_str = self._format_output(critique_agent_output)

        user_prompt = f"""Please judge the following debate about a {target_agent_type} agent's output.

## MAIN AGENT'S ORIGINAL OUTPUT:
{main_output_str}

## CRITIQUE AGENT'S CONCERNS:
{critique_output_str}

## DEBATE TRANSCRIPT:
{debate_summary}

## MAIN AGENT'S FINAL POSITION:
{main_agent_final}

## CRITIQUE AGENT'S FINAL POSITION:
{critique_final}

---

Based on the above, provide your judgment. You must decide:
1. Whose position should be adopted (main_agent or critique_agent)
2. OR provide a synthesized position that incorporates the best of both

Consider: Which position will lead to a better ML pipeline outcome?"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use the specified model for judging
        return await self.openai_client.chat_json(
            messages=messages,
            model=self.judge_model,
            response_schema={
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["main_agent", "critique_agent", "synthesis"],
                        "description": "Whose position to adopt, or 'synthesis' if combining both",
                    },
                    "final_output": {
                        "type": "object",
                        "description": "The final output to use (winner's output or synthesized)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed reasoning for the decision",
                    },
                    "key_factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key factors that influenced the decision",
                    },
                    "confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Confidence in this decision (0-100)",
                    },
                    "main_agent_strengths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Strengths of the main agent's position",
                    },
                    "main_agent_weaknesses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Weaknesses of the main agent's position",
                    },
                    "critique_strengths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Strengths of the critique agent's position",
                    },
                    "critique_weaknesses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Weaknesses of the critique agent's position",
                    },
                    "synthesis_notes": {
                        "type": "string",
                        "description": "If synthesizing, explain how elements were combined",
                    },
                },
                "required": ["decision", "final_output", "reasoning", "confidence"],
            },
        )

    def _format_debate_transcript(self, transcript: List[Dict[str, Any]]) -> str:
        """Format the debate transcript for review."""
        if not transcript:
            return "No debate transcript available."

        formatted = []
        for i, entry in enumerate(transcript, 1):
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            round_num = entry.get("round", i)

            # Truncate very long entries
            if len(content) > 1500:
                content = content[:1500] + "... [truncated]"

            formatted.append(f"### Round {round_num} - {role.upper()}:\n{content}")

        return "\n\n".join(formatted)

    def _format_output(self, output: Dict[str, Any]) -> str:
        """Format agent output for display."""
        if not output:
            return "No output provided"

        import json
        try:
            return json.dumps(output, indent=2, default=str)[:3000]
        except Exception:
            return str(output)[:3000]


def get_available_judge_models() -> List[str]:
    """Get list of available OpenAI models for judging."""
    return AVAILABLE_JUDGE_MODELS.copy()


def validate_judge_model(model: str) -> bool:
    """Check if a model is valid for judging."""
    return model in AVAILABLE_JUDGE_MODELS

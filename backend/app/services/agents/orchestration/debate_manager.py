"""Debate Manager - Orchestrates debates between main agents and Gemini critics.

The Debate Manager handles the full debate flow:
1. Main agent produces output
2. Gemini critique agent reviews and critiques
3. Main agent responds to critique
4. Back and forth for up to 3 rounds
5. If consensus reached, use agreed output
6. If no consensus, OpenAI Judge decides

This can be combined with Project Manager mode for dynamic orchestration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from sqlalchemy.orm import Session

from app.models import AgentStepType, AgentStep, AgentStepStatus
from app.services.agents.utils.step_logger import StepLogger
from app.services.llm_client import BaseLLMClient, GeminiClient, OpenAIClient

if TYPE_CHECKING:
    from app.models import AgentRun

logger = logging.getLogger(__name__)


DEFAULT_MAX_DEBATE_ROUNDS = 3


@dataclass
class DebateMessage:
    """A single message in a debate."""
    role: str  # "main_agent", "critique_agent", or "judge"
    content: str
    round: int
    agrees: Optional[bool] = None
    confidence: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateResult:
    """Result of a debate between agents."""
    consensus_reached: bool
    final_output: Dict[str, Any]
    winner: str  # "main_agent", "critique_agent", "synthesis", or "consensus"
    transcript: List[DebateMessage]
    total_rounds: int
    judge_decision: Optional[Dict[str, Any]] = None
    summary: str = ""


class DebateManager:
    """Manages debates between main agents and Gemini critique agents.

    The debate flow:
    1. Main agent produces initial output
    2. Gemini critique reviews and either agrees or critiques
    3. If critique disagrees:
       a. Main agent responds to concerns
       b. Critique agent responds
       c. Continue for up to 3 rounds
    4. If still no consensus:
       a. OpenAI Judge reviews full transcript
       b. Judge makes final decision
    5. Return final output with debate transcript
    """

    def __init__(
        self,
        db: Session,
        agent_run: "AgentRun",
        step_logger: StepLogger,
        main_llm_client: BaseLLMClient,
        gemini_client: Optional[GeminiClient] = None,
        openai_client: Optional[OpenAIClient] = None,
        judge_model: Optional[str] = None,
    ):
        """Initialize the debate manager.

        Args:
            db: Database session
            agent_run: The current agent run
            step_logger: Logger for step progress
            main_llm_client: The main LLM client (for main agent responses)
            gemini_client: Gemini client for critique agent
            openai_client: OpenAI client for judge
            judge_model: Which OpenAI model to use as judge
        """
        self.db = db
        self.agent_run = agent_run
        self.step_logger = step_logger
        self.main_llm_client = main_llm_client
        self.gemini_client = gemini_client or main_llm_client
        self.openai_client = openai_client or main_llm_client
        self.judge_model = judge_model or "gpt-5.1"

    @property
    def max_rounds(self) -> int:
        """Get max debate rounds from agent_run or use default."""
        if hasattr(self.agent_run, 'max_debate_rounds') and self.agent_run.max_debate_rounds:
            return self.agent_run.max_debate_rounds
        return DEFAULT_MAX_DEBATE_ROUNDS

    async def run_debate(
        self,
        agent_type: AgentStepType,
        main_agent_output: Dict[str, Any],
        context: Dict[str, Any],
    ) -> DebateResult:
        """Run a full debate on an agent's output.

        Args:
            agent_type: The type of main agent being debated
            main_agent_output: The output from the main agent
            context: Additional context for the debate

        Returns:
            DebateResult with final output and transcript
        """
        logger.info(f"=== DEBATE STARTING for {agent_type.value} ===")
        logger.info(f"Gemini client: {self.gemini_client is not None}, OpenAI client: {self.openai_client is not None}")
        logger.info(f"Max rounds: {self.max_rounds}, Judge model: {self.judge_model}")

        transcript: List[DebateMessage] = []
        current_round = 1

        self.step_logger.thinking(f"Starting debate for {agent_type.value}")

        # Round 1: Initial critique
        critique_result = await self._get_critique(
            agent_type=agent_type,
            main_agent_output=main_agent_output,
            debate_round=1,
            previous_messages=[],
        )

        transcript.append(DebateMessage(
            role="critique_agent",
            content=critique_result.get("critique", ""),
            round=1,
            agrees=critique_result.get("agrees", False),
            confidence=critique_result.get("confidence", 50),
            metadata=critique_result,
        ))

        # Check if critique agrees
        if critique_result.get("agrees", False):
            self.step_logger.summary("Critique agrees with main agent output")
            return DebateResult(
                consensus_reached=True,
                final_output=main_agent_output,
                winner="consensus",
                transcript=transcript,
                total_rounds=1,
                summary="Consensus reached in round 1 - critique agent agreed with main agent.",
            )

        # Continue debate for remaining rounds
        current_output = main_agent_output
        self.step_logger.info(f"Max debate rounds configured: {self.max_rounds}")
        for round_num in range(2, self.max_rounds + 1):
            current_round = round_num

            # Main agent responds to critique
            main_response = await self._get_main_agent_response(
                agent_type=agent_type,
                original_output=main_agent_output,
                current_output=current_output,
                critique=critique_result,
                round_num=round_num,
                context=context,
            )

            transcript.append(DebateMessage(
                role="main_agent",
                content=main_response.get("response", ""),
                round=round_num,
                agrees=main_response.get("accepts_critique", False),
                confidence=main_response.get("confidence", 50),
                metadata=main_response,
            ))

            # Check if main agent accepts critique
            if main_response.get("accepts_critique", False):
                # Main agent concedes - use modified output
                modified_output = main_response.get("modified_output", current_output)
                self.step_logger.summary(f"Main agent accepted critique in round {round_num}")
                return DebateResult(
                    consensus_reached=True,
                    final_output=modified_output,
                    winner="critique_agent",
                    transcript=transcript,
                    total_rounds=round_num,
                    summary=f"Consensus reached in round {round_num} - main agent accepted critique.",
                )

            # Update current output if main agent modified it
            if main_response.get("modified_output"):
                current_output = main_response["modified_output"]

            # Critique agent responds
            critique_result = await self._get_critique(
                agent_type=agent_type,
                main_agent_output=current_output,
                debate_round=round_num,
                previous_messages=self._format_transcript_for_llm(transcript),
            )

            transcript.append(DebateMessage(
                role="critique_agent",
                content=critique_result.get("critique", ""),
                round=round_num,
                agrees=critique_result.get("agrees", False),
                confidence=critique_result.get("confidence", 50),
                metadata=critique_result,
            ))

            # Check if critique now agrees
            if critique_result.get("agrees", False):
                self.step_logger.summary(f"Critique agreed in round {round_num}")
                return DebateResult(
                    consensus_reached=True,
                    final_output=current_output,
                    winner="consensus",
                    transcript=transcript,
                    total_rounds=round_num,
                    summary=f"Consensus reached in round {round_num} - critique accepted main agent's response.",
                )

        # No consensus after max rounds - call the judge
        self.step_logger.thinking("No consensus reached - calling OpenAI Judge")

        judge_decision = await self._call_judge(
            agent_type=agent_type,
            main_agent_output=current_output,
            critique_output=critique_result,
            transcript=transcript,
        )

        # Add judge decision to transcript
        transcript.append(DebateMessage(
            role="judge",
            content=judge_decision.get("reasoning", ""),
            round=current_round + 1,
            confidence=judge_decision.get("confidence", 50),
            metadata=judge_decision,
        ))

        # Determine final output based on judge decision
        decision = judge_decision.get("decision", "main_agent")
        if decision == "synthesis":
            final_output = judge_decision.get("final_output", current_output)
            winner = "synthesis"
        elif decision == "critique_agent":
            # Use critique's suggested modifications if available
            final_output = judge_decision.get("final_output", current_output)
            winner = "critique_agent"
        else:
            final_output = judge_decision.get("final_output", current_output)
            winner = "main_agent"

        self.step_logger.summary(f"Judge decision: {winner}")

        return DebateResult(
            consensus_reached=False,
            final_output=final_output,
            winner=winner,
            transcript=transcript,
            total_rounds=current_round,
            judge_decision=judge_decision,
            summary=f"No consensus after {current_round} rounds. Judge decided: {winner}",
        )

    async def _get_critique(
        self,
        agent_type: AgentStepType,
        main_agent_output: Dict[str, Any],
        debate_round: int,
        previous_messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Get critique from Gemini critique agent."""
        logger.info(f"Getting Gemini critique for {agent_type.value}, round {debate_round}")
        from app.services.agents.orchestration.gemini_critique import (
            GeminiCritiqueAgent,
            CRITIQUE_SPECIALIZATIONS,
        )

        # Create a lightweight step for the critique
        critique_step = AgentStep(
            agent_run_id=self.agent_run.id,
            step_type=AgentStepType.GEMINI_CRITIQUE,
            status=AgentStepStatus.RUNNING,
            input_json={
                "target_agent_type": agent_type.value,
                "main_agent_output": main_agent_output,
                "debate_round": debate_round,
                "previous_messages": previous_messages,
            },
        )
        self.db.add(critique_step)
        self.db.flush()

        # Create and execute critique agent
        critique_agent = GeminiCritiqueAgent(
            db=self.db,
            step=critique_step,
            step_logger=self.step_logger,
            llm_client=self.main_llm_client,
            gemini_client=self.gemini_client,
        )

        try:
            result = await critique_agent.execute()
            critique_step.status = AgentStepStatus.COMPLETED
            critique_step.output_json = result
            self.db.commit()
            logger.info(f"Critique completed: agrees={result.get('agrees')}, confidence={result.get('confidence')}")
            return result
        except Exception as e:
            logger.error(f"Critique failed: {e}", exc_info=True)
            critique_step.status = AgentStepStatus.FAILED
            critique_step.error_message = str(e)
            self.db.commit()
            # Return a default "agree" to avoid blocking pipeline
            return {
                "agrees": True,
                "critique": f"Critique unavailable due to error: {e}",
                "reasoning": "Defaulting to agreement due to critique error",
                "confidence": 0,
            }

    async def _get_main_agent_response(
        self,
        agent_type: AgentStepType,
        original_output: Dict[str, Any],
        current_output: Dict[str, Any],
        critique: Dict[str, Any],
        round_num: int,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get main agent's response to critique."""
        from app.services.agents.orchestration.gemini_critique import CRITIQUE_SPECIALIZATIONS

        # Get agent description for context
        specialization = CRITIQUE_SPECIALIZATIONS.get(agent_type, {})
        agent_name = specialization.get("name", agent_type.value)

        system_prompt = f"""You are a {agent_type.value} agent in a machine learning pipeline.
A critique agent has raised concerns about your output. You should:

1. Carefully consider the critique
2. If the critique has valid points, acknowledge them and modify your output
3. If you disagree, explain your reasoning clearly
4. Be open to improving your output, but don't concede on technically correct decisions

This is round {round_num} of the debate. Be constructive and focused on the best ML pipeline outcome."""

        critique_text = critique.get("critique", "")
        concerns = critique.get("key_concerns", []) or critique.get("remaining_concerns", [])
        suggestions = critique.get("suggestions", [])

        concerns_str = "\n".join(f"- {c}" for c in concerns) if concerns else "None specified"
        suggestions_str = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None specified"

        user_prompt = f"""The critique agent raised the following concerns about your output:

CRITIQUE:
{critique_text}

KEY CONCERNS:
{concerns_str}

SUGGESTIONS:
{suggestions_str}

YOUR CURRENT OUTPUT:
{self._format_output(current_output)}

Please respond to this critique. You may:
1. Accept the critique and provide modified output
2. Partially accept and address specific concerns
3. Defend your position with clear reasoning

What is your response?"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return await self.main_llm_client.chat_json(
            messages=messages,
            response_schema={
                "type": "object",
                "properties": {
                    "accepts_critique": {
                        "type": "boolean",
                        "description": "Whether you accept the critique fully",
                    },
                    "response": {
                        "type": "string",
                        "description": "Your detailed response to the critique",
                    },
                    "addressed_concerns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of concerns you're addressing",
                    },
                    "rejected_concerns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of concerns you disagree with, with reasons",
                    },
                    "modified_output": {
                        "type": "object",
                        "description": "Your modified output (if any changes made)",
                    },
                    "confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Your confidence in your position (0-100)",
                    },
                },
                "required": ["accepts_critique", "response", "confidence"],
            },
        )

    async def _call_judge(
        self,
        agent_type: AgentStepType,
        main_agent_output: Dict[str, Any],
        critique_output: Dict[str, Any],
        transcript: List[DebateMessage],
    ) -> Dict[str, Any]:
        """Call OpenAI Judge to make final decision."""
        logger.info(f"=== CALLING OPENAI JUDGE for {agent_type.value} ===")
        logger.info(f"Judge model: {self.judge_model}")
        from app.services.agents.orchestration.openai_judge import OpenAIJudgeAgent

        # Create a step for the judge
        judge_step = AgentStep(
            agent_run_id=self.agent_run.id,
            step_type=AgentStepType.OPENAI_JUDGE,
            status=AgentStepStatus.RUNNING,
            input_json={
                "target_agent_type": agent_type.value,
                "main_agent_output": main_agent_output,
                "critique_agent_output": critique_output,
                "debate_transcript": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "round": msg.round,
                        "agrees": msg.agrees,
                    }
                    for msg in transcript
                ],
            },
        )
        self.db.add(judge_step)
        self.db.flush()

        # Get final positions
        main_final = ""
        critique_final = ""
        for msg in reversed(transcript):
            if msg.role == "main_agent" and not main_final:
                main_final = msg.content
            elif msg.role == "critique_agent" and not critique_final:
                critique_final = msg.content
            if main_final and critique_final:
                break

        judge_step.input_json["main_agent_final_position"] = main_final
        judge_step.input_json["critique_final_position"] = critique_final

        # Create and execute judge agent
        judge_agent = OpenAIJudgeAgent(
            db=self.db,
            step=judge_step,
            step_logger=self.step_logger,
            llm_client=self.main_llm_client,
            judge_model=self.judge_model,
            openai_client=self.openai_client,
        )

        try:
            result = await judge_agent.execute()
            judge_step.status = AgentStepStatus.COMPLETED
            judge_step.output_json = result
            self.db.commit()
            logger.info(f"Judge decision: {result.get('decision')}, confidence={result.get('confidence')}")
            return result
        except Exception as e:
            logger.error(f"Judge failed: {e}", exc_info=True)
            judge_step.status = AgentStepStatus.FAILED
            judge_step.error_message = str(e)
            self.db.commit()
            # Default to main agent output on judge failure
            return {
                "decision": "main_agent",
                "final_output": main_agent_output,
                "reasoning": f"Judge unavailable due to error: {e}. Defaulting to main agent output.",
                "confidence": 0,
            }

    def _format_transcript_for_llm(
        self, transcript: List[DebateMessage]
    ) -> List[Dict[str, str]]:
        """Format transcript for LLM consumption."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "round": msg.round,
            }
            for msg in transcript
        ]

    def _format_output(self, output: Dict[str, Any]) -> str:
        """Format output for display."""
        import json
        try:
            return json.dumps(output, indent=2, default=str)[:3000]
        except Exception:
            return str(output)[:3000]


def create_debate_manager(
    db: Session,
    agent_run: "AgentRun",
    step_logger: StepLogger,
    main_llm_client: BaseLLMClient,
    gemini_client: Optional[GeminiClient] = None,
    openai_client: Optional[OpenAIClient] = None,
) -> DebateManager:
    """Factory function to create a DebateManager with proper clients.

    Args:
        db: Database session
        agent_run: Current agent run
        step_logger: Logger for progress
        main_llm_client: Main LLM client
        gemini_client: Optional Gemini client
        openai_client: Optional OpenAI client

    Returns:
        Configured DebateManager instance
    """
    return DebateManager(
        db=db,
        agent_run=agent_run,
        step_logger=step_logger,
        main_llm_client=main_llm_client,
        gemini_client=gemini_client,
        openai_client=openai_client,
        judge_model=agent_run.judge_model,
    )

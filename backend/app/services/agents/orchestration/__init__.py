"""Orchestration agents for dynamic pipeline control and debate system.

This module provides:
- Project Manager Agent: Dynamically orchestrates agent flow
- Debate System: Gemini critique + OpenAI judge for consensus
- Debate Manager: Coordinates debates between agents
"""

from app.services.agents.orchestration.project_manager import (
    ProjectManagerAgent,
    ORCHESTRABLE_AGENTS,
    AGENT_DESCRIPTIONS,
)
from app.services.agents.orchestration.debate_manager import (
    DebateManager,
    DebateResult,
    DebateMessage,
    create_debate_manager,
)
from app.services.agents.orchestration.gemini_critique import (
    GeminiCritiqueAgent,
    CRITIQUE_SPECIALIZATIONS,
)
from app.services.agents.orchestration.openai_judge import (
    OpenAIJudgeAgent,
    AVAILABLE_JUDGE_MODELS,
    DEFAULT_JUDGE_MODEL,
    get_available_judge_models,
    validate_judge_model,
)

__all__ = [
    # Project Manager
    "ProjectManagerAgent",
    "ORCHESTRABLE_AGENTS",
    "AGENT_DESCRIPTIONS",
    # Debate Manager
    "DebateManager",
    "DebateResult",
    "DebateMessage",
    "create_debate_manager",
    # Gemini Critique
    "GeminiCritiqueAgent",
    "CRITIQUE_SPECIALIZATIONS",
    # OpenAI Judge
    "OpenAIJudgeAgent",
    "AVAILABLE_JUDGE_MODELS",
    "DEFAULT_JUDGE_MODEL",
    "get_available_judge_models",
    "validate_judge_model",
]

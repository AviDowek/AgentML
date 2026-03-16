"""Agent system for ML experiment pipelines.

This module provides a class-based agent architecture where each agent
handles a specific step in the ML experiment workflow.

Agents are organized by pipeline:
- setup/: Setup pipeline agents (data analysis, problem understanding, etc.)
- results/: Results analysis agents
- data_architect/: Multi-source data architecture agents
- standalone/: Standalone utility agents

Usage:
    from app.services.agents import get_agent_class, is_agent_registered

    agent_class = get_agent_class(AgentStepType.DATA_ANALYSIS)
    agent = agent_class(db, step, step_logger, llm_client)
    output = await agent.execute()
"""

from app.services.agents.registry import (
    get_agent_class,
    get_all_agent_types,
    is_agent_registered,
)
from app.services.agents.base import BaseAgent

__all__ = [
    "get_agent_class",
    "get_all_agent_types",
    "is_agent_registered",
    "BaseAgent",
]

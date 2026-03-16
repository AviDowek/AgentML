"""Agent Registry - Maps AgentStepType to Agent classes.

This module provides a central registry for all agent classes, mapping
from AgentStepType enum values to the corresponding agent implementations.
"""

from typing import Dict, Type

from app.models import AgentStepType
from app.services.agents.base import BaseAgent


def _get_agent_registry() -> Dict[AgentStepType, Type[BaseAgent]]:
    """Build the agent registry lazily to avoid circular imports.

    Returns:
        Dictionary mapping AgentStepType to agent class.
    """
    # Import agent classes lazily
    from app.services.agents.setup import (
        DataAnalysisAgent,
        ProblemUnderstandingAgent,
        DataAuditAgent,
        DatasetDesignAgent,
        DatasetValidationAgent,
        ExperimentDesignAgent,
        PlanCriticAgent,
    )
    from app.services.agents.results import (
        ResultsInterpretationAgent,
        ResultsCriticAgent,
    )
    from app.services.agents.data_architect import (
        DatasetInventoryAgent,
        RelationshipDiscoveryAgent,
        TrainingDatasetPlanningAgent,
        TrainingDatasetBuildAgent,
    )
    from app.services.agents.standalone import (
        DatasetDiscoveryAgent,
        LabNotebookSummaryAgent,
        RobustnessAuditAgent,
    )
    from app.services.agents.improvement import (
        IterationContextAgent,
        ImprovementDataAnalysisAgent,
        ImprovementDatasetDesignAgent,
        ImprovementExperimentDesignAgent,
        ImprovementAnalysisAgent,
        ImprovementPlanAgent,
    )

    return {
        # Setup Pipeline
        AgentStepType.DATA_ANALYSIS: DataAnalysisAgent,
        AgentStepType.PROBLEM_UNDERSTANDING: ProblemUnderstandingAgent,
        AgentStepType.DATA_AUDIT: DataAuditAgent,
        AgentStepType.DATASET_DESIGN: DatasetDesignAgent,
        AgentStepType.DATASET_VALIDATION: DatasetValidationAgent,
        AgentStepType.EXPERIMENT_DESIGN: ExperimentDesignAgent,
        AgentStepType.PLAN_CRITIC: PlanCriticAgent,

        # Results Pipeline
        AgentStepType.RESULTS_INTERPRETATION: ResultsInterpretationAgent,
        AgentStepType.RESULTS_CRITIC: ResultsCriticAgent,

        # Data Architect Pipeline
        AgentStepType.DATASET_INVENTORY: DatasetInventoryAgent,
        AgentStepType.RELATIONSHIP_DISCOVERY: RelationshipDiscoveryAgent,
        AgentStepType.TRAINING_DATASET_PLANNING: TrainingDatasetPlanningAgent,
        AgentStepType.TRAINING_DATASET_BUILD: TrainingDatasetBuildAgent,

        # Standalone Agents
        AgentStepType.DATASET_DISCOVERY: DatasetDiscoveryAgent,
        AgentStepType.LAB_NOTEBOOK_SUMMARY: LabNotebookSummaryAgent,
        AgentStepType.ROBUSTNESS_AUDIT: RobustnessAuditAgent,

        # Improvement Pipeline (Enhanced)
        AgentStepType.ITERATION_CONTEXT: IterationContextAgent,
        AgentStepType.IMPROVEMENT_DATA_ANALYSIS: ImprovementDataAnalysisAgent,
        AgentStepType.IMPROVEMENT_DATASET_DESIGN: ImprovementDatasetDesignAgent,
        AgentStepType.IMPROVEMENT_EXPERIMENT_DESIGN: ImprovementExperimentDesignAgent,

        # Improvement Pipeline (Simple)
        AgentStepType.IMPROVEMENT_ANALYSIS: ImprovementAnalysisAgent,
        AgentStepType.IMPROVEMENT_PLAN: ImprovementPlanAgent,
    }


# Cached registry
_REGISTRY_CACHE: Dict[AgentStepType, Type[BaseAgent]] | None = None


def get_agent_class(step_type: AgentStepType) -> Type[BaseAgent]:
    """Get the agent class for a given step type.

    Args:
        step_type: The AgentStepType enum value.

    Returns:
        The agent class that handles this step type.

    Raises:
        ValueError: If no agent is registered for the step type.
    """
    global _REGISTRY_CACHE

    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _get_agent_registry()

    agent_class = _REGISTRY_CACHE.get(step_type)
    if agent_class is None:
        raise ValueError(f"No agent registered for step type: {step_type}")

    return agent_class


def get_all_agent_types() -> list[AgentStepType]:
    """Get all registered agent step types.

    Returns:
        List of all registered AgentStepType values.
    """
    global _REGISTRY_CACHE

    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _get_agent_registry()

    return list(_REGISTRY_CACHE.keys())


def is_agent_registered(step_type: AgentStepType) -> bool:
    """Check if an agent is registered for a step type.

    Args:
        step_type: The AgentStepType enum value.

    Returns:
        True if an agent is registered, False otherwise.
    """
    global _REGISTRY_CACHE

    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _get_agent_registry()

    return step_type in _REGISTRY_CACHE

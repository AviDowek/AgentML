"""Improvement pipeline agents."""


def __getattr__(name):
    """Lazy import of agent classes."""
    if name == "IterationContextAgent":
        from app.services.agents.improvement.iteration_context import IterationContextAgent
        return IterationContextAgent
    elif name == "ImprovementDataAnalysisAgent":
        from app.services.agents.improvement.improvement_data_analysis import ImprovementDataAnalysisAgent
        return ImprovementDataAnalysisAgent
    elif name == "ImprovementDatasetDesignAgent":
        from app.services.agents.improvement.improvement_dataset_design import ImprovementDatasetDesignAgent
        return ImprovementDatasetDesignAgent
    elif name == "ImprovementExperimentDesignAgent":
        from app.services.agents.improvement.improvement_experiment_design import ImprovementExperimentDesignAgent
        return ImprovementExperimentDesignAgent
    elif name == "ImprovementAnalysisAgent":
        from app.services.agents.improvement.improvement_analysis import ImprovementAnalysisAgent
        return ImprovementAnalysisAgent
    elif name == "ImprovementPlanAgent":
        from app.services.agents.improvement.improvement_plan import ImprovementPlanAgent
        return ImprovementPlanAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IterationContextAgent",
    "ImprovementDataAnalysisAgent",
    "ImprovementDatasetDesignAgent",
    "ImprovementExperimentDesignAgent",
    "ImprovementAnalysisAgent",
    "ImprovementPlanAgent",
]

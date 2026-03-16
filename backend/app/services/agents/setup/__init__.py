"""Setup pipeline agents.

These agents handle the initial experiment setup workflow:
1. DataAnalysisAgent - Analyze data quality and suitability
2. ProblemUnderstandingAgent - Determine task type, target, metrics
3. DataAuditAgent - Comprehensive data quality audit
4. DatasetDesignAgent - Generate dataset variants
5. DatasetValidationAgent - Validate dataset design against actual data
6. ExperimentDesignAgent - Design experiment configurations
7. PlanCriticAgent - Validate experiment plans
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "DataAnalysisAgent":
        from app.services.agents.setup.data_analysis import DataAnalysisAgent
        return DataAnalysisAgent
    elif name == "ProblemUnderstandingAgent":
        from app.services.agents.setup.problem_understanding import ProblemUnderstandingAgent
        return ProblemUnderstandingAgent
    elif name == "DataAuditAgent":
        from app.services.agents.setup.data_audit import DataAuditAgent
        return DataAuditAgent
    elif name == "DatasetDesignAgent":
        from app.services.agents.setup.dataset_design import DatasetDesignAgent
        return DatasetDesignAgent
    elif name == "DatasetValidationAgent":
        from app.services.agents.setup.dataset_validation import DatasetValidationAgent
        return DatasetValidationAgent
    elif name == "ExperimentDesignAgent":
        from app.services.agents.setup.experiment_design import ExperimentDesignAgent
        return ExperimentDesignAgent
    elif name == "PlanCriticAgent":
        from app.services.agents.setup.plan_critic import PlanCriticAgent
        return PlanCriticAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataAnalysisAgent",
    "ProblemUnderstandingAgent",
    "DataAuditAgent",
    "DatasetDesignAgent",
    "DatasetValidationAgent",
    "ExperimentDesignAgent",
    "PlanCriticAgent",
]

"""Results pipeline agents.

These agents analyze experiment results:
1. ResultsInterpretationAgent - Analyze results and recommend models
2. ResultsCriticAgent - Review for overfitting and issues
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "ResultsInterpretationAgent":
        from app.services.agents.results.results_interpretation import ResultsInterpretationAgent
        return ResultsInterpretationAgent
    elif name == "ResultsCriticAgent":
        from app.services.agents.results.results_critic import ResultsCriticAgent
        return ResultsCriticAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ResultsInterpretationAgent",
    "ResultsCriticAgent",
]

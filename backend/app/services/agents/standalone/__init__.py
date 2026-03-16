"""Standalone utility agents.

These agents can run independently:
1. DatasetDiscoveryAgent - Search for public datasets
2. LabNotebookSummaryAgent - Generate research summaries
3. RobustnessAuditAgent - Audit models for overfitting/leakage
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "DatasetDiscoveryAgent":
        from app.services.agents.standalone.dataset_discovery import DatasetDiscoveryAgent
        return DatasetDiscoveryAgent
    elif name == "LabNotebookSummaryAgent":
        from app.services.agents.standalone.lab_notebook_summary import LabNotebookSummaryAgent
        return LabNotebookSummaryAgent
    elif name == "RobustnessAuditAgent":
        from app.services.agents.standalone.robustness_audit import RobustnessAuditAgent
        return RobustnessAuditAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetDiscoveryAgent",
    "LabNotebookSummaryAgent",
    "RobustnessAuditAgent",
]

"""Data architect pipeline agents.

These agents handle multi-source dataset construction:
1. DatasetInventoryAgent - Profile all data sources
2. RelationshipDiscoveryAgent - Discover table relationships
3. TrainingDatasetPlanningAgent - Plan dataset construction
4. TrainingDatasetBuildAgent - Build the training dataset
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "DatasetInventoryAgent":
        from app.services.agents.data_architect.dataset_inventory import DatasetInventoryAgent
        return DatasetInventoryAgent
    elif name == "RelationshipDiscoveryAgent":
        from app.services.agents.data_architect.relationship_discovery import RelationshipDiscoveryAgent
        return RelationshipDiscoveryAgent
    elif name == "TrainingDatasetPlanningAgent":
        from app.services.agents.data_architect.training_dataset_planning import TrainingDatasetPlanningAgent
        return TrainingDatasetPlanningAgent
    elif name == "TrainingDatasetBuildAgent":
        from app.services.agents.data_architect.training_dataset_build import TrainingDatasetBuildAgent
        return TrainingDatasetBuildAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatasetInventoryAgent",
    "RelationshipDiscoveryAgent",
    "TrainingDatasetPlanningAgent",
    "TrainingDatasetBuildAgent",
]

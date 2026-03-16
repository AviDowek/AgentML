"""Dataset Inventory Agent - Profiles all data sources in a project.

This agent profiles all data sources in a project to create an inventory
of available datasets for the Data Architect pipeline.
"""

from typing import Any, Dict, List

from app.models import DataSource, Project
from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.data_profiler import profile_data_source


class DatasetInventoryAgent(BaseAgent):
    """Profiles all data sources in a project.

    Input JSON:
        - project_id: UUID of the project

    Output:
        - data_source_profiles: List of profile dictionaries
        - total_sources: Total number of sources
        - profiled_count: Number successfully profiled
        - error_count: Number of errors
        - errors: Error details if any
    """

    name = "dataset_inventory"
    step_type = AgentStepType.DATASET_INVENTORY

    async def execute(self) -> Dict[str, Any]:
        """Execute dataset inventory."""
        project_id = self.require_input("project_id")

        self.logger.info("Starting dataset inventory...")

        # Get the project
        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Get all data sources
        data_sources = self.db.query(DataSource).filter(
            DataSource.project_id == project_id
        ).all()

        if not data_sources:
            self.logger.warning("No data sources found in project")
            return {
                "data_source_profiles": [],
                "total_sources": 0,
                "profiled_count": 0,
            }

        self.logger.info(f"Found {len(data_sources)} data source(s) to profile")

        # Profile all data sources
        profiles = []
        errors = []

        for ds in data_sources:
            self.logger.thought(f"Profiling data source: {ds.name}")
            try:
                profile = profile_data_source(self.db, ds.id)
                profiles.append(profile)

                # Update the data source's profile_json
                ds.profile_json = profile
                self.db.commit()

                self.logger.info(
                    f"Profiled '{ds.name}': {profile.get('estimated_row_count', 0):,} rows, "
                    f"{profile.get('column_count', 0)} columns"
                )
            except Exception as e:
                error_msg = f"Failed to profile '{ds.name}': {str(e)}"
                self.logger.warning(error_msg)
                errors.append({
                    "source_id": str(ds.id),
                    "source_name": ds.name,
                    "error": str(e),
                })

        self.logger.summary(
            f"Dataset inventory complete. Profiled {len(profiles)}/{len(data_sources)} data sources."
        )

        return {
            "data_source_profiles": profiles,
            "total_sources": len(data_sources),
            "profiled_count": len(profiles),
            "error_count": len(errors),
            "errors": errors if errors else None,
        }

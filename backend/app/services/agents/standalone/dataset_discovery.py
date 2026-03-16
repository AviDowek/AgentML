"""Dataset Discovery Agent - Searches for relevant public datasets.

This agent searches for relevant public datasets based on a problem description
and presents them as options for the user.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.prompts import (
    SYSTEM_ROLE_DATASET_EXPERT,
    get_dataset_discovery_prompt,
)


class DatasetSchemaInfo(BaseModel):
    """Schema information for a discovered dataset."""
    rows_estimate: int = Field(description="Estimated number of rows")
    columns: List[str] = Field(description="List of column names")
    target_candidate: str = Field(description="Best guess for target column")


class DiscoveredDataset(BaseModel):
    """A discovered dataset candidate."""
    name: str = Field(description="Dataset name")
    source_url: str = Field(description="URL where dataset can be found")
    schema_summary: DatasetSchemaInfo = Field(description="Schema summary")
    licensing: str = Field(description="License type")
    fit_for_purpose: str = Field(description="How well it fits user's needs")


class DatasetDiscoveryResponse(BaseModel):
    """Response from dataset discovery."""
    discovered_datasets: List[DiscoveredDataset] = Field(description="List of datasets")
    natural_language_summary: str = Field(description="Summary for user")


class DatasetDiscoveryAgent(BaseAgent):
    """Searches for relevant public datasets.

    Input JSON:
        - project_description: Text description of the ML problem
        - constraints: Optional dict with:
            - geography: Geographic region constraint
            - allow_public_data: Whether to search public data sources

    Output:
        - discovered_datasets: List of dataset candidates with metadata
        - natural_language_summary: Summary for user
    """

    name = "dataset_discovery"
    step_type = AgentStepType.DATASET_DISCOVERY

    async def execute(self) -> Dict[str, Any]:
        """Execute dataset discovery."""
        project_description = self.require_input("project_description")
        constraints = self.get_input("constraints", {})
        geography = constraints.get("geography", "")
        allow_public_data = constraints.get("allow_public_data", True)

        self.logger.info(f"Searching for datasets for: {project_description[:100]}...")

        if geography:
            self.logger.thought(f"Geographic constraint: {geography}")

        if not allow_public_data:
            self.logger.warning("Public data sources are disabled - search may be limited")

        # Build prompt
        geography_constraint = f"\n- Geographic focus: {geography}" if geography else ""
        public_data_note = "" if allow_public_data else "\n- Note: User prefers private/proprietary data sources only"

        self.logger.info("Consulting LLM to search for relevant datasets...")

        prompt = get_dataset_discovery_prompt(
            project_description=project_description,
            geography_constraint=geography_constraint,
            public_data_note=public_data_note,
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_DATASET_EXPERT},
            {"role": "user", "content": prompt},
        ]

        self.logger.thought("Searching dataset repositories for verified, high-quality datasets...")

        response = await self.llm.chat_json(messages, DatasetDiscoveryResponse)

        discovered_datasets = response.get("discovered_datasets", [])
        natural_language_summary = response.get("natural_language_summary", "")

        if discovered_datasets:
            self.logger.info(f"Found {len(discovered_datasets)} potential dataset(s)")
            for i, ds in enumerate(discovered_datasets, 1):
                name = ds.get("name", "Unknown")
                licensing = ds.get("licensing", "Unknown")
                schema = ds.get("schema_summary", {})
                rows = schema.get("rows_estimate", 0)
                target = schema.get("target_candidate", "N/A")
                self.logger.thought(
                    f"Dataset {i}: {name} (~{rows:,} rows, target: {target}, license: {licensing})"
                )
        else:
            self.logger.warning("No suitable datasets found")

        self.logger.summary(
            f"Dataset discovery complete. Found {len(discovered_datasets)} dataset(s). "
            f"Summary: {natural_language_summary[:100]}..."
        )

        return {
            "discovered_datasets": discovered_datasets,
            "natural_language_summary": natural_language_summary,
        }

"""Relationship Discovery Agent - Discovers relationships between data sources.

This agent discovers relationships between data sources in a project,
identifying foreign keys and join patterns.
"""

from typing import Any, Dict, List
from uuid import UUID

from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.relationship_discovery import discover_relationships_for_project


class RelationshipDiscoveryAgent(BaseAgent):
    """Discovers relationships between data sources.

    Input JSON:
        - project_id: UUID of the project
        - data_source_profiles: List of profiles from dataset_inventory (for reference)

    Output:
        - tables: List of table metadata
        - relationships: List of discovered relationships
        - base_table_candidates: Candidates for base table
        - relationships_summary: Full summary of results
    """

    name = "relationship_discovery"
    step_type = AgentStepType.RELATIONSHIP_DISCOVERY

    async def execute(self) -> Dict[str, Any]:
        """Execute relationship discovery."""
        project_id = self.require_input("project_id")
        data_source_profiles = self.get_input("data_source_profiles", [])

        self.logger.info("Starting relationship discovery...")
        self.logger.thought(f"Analyzing {len(data_source_profiles)} data source profile(s)")

        try:
            project_uuid = UUID(project_id) if isinstance(project_id, str) else project_id
            relationships_result = discover_relationships_for_project(self.db, project_uuid)

            tables = relationships_result.get("tables", [])
            relationships = relationships_result.get("relationships", [])
            base_candidates = relationships_result.get("base_table_candidates", [])

            self.logger.info(f"Found {len(tables)} table(s)")
            self.logger.info(f"Discovered {len(relationships)} relationship(s)")

            # Log relationship details
            for rel in relationships[:5]:
                self.logger.thought(
                    f"  {rel.get('from_table')}.{rel.get('from_column')} -> "
                    f"{rel.get('to_table')}.{rel.get('to_column')} "
                    f"({rel.get('relationship_type', 'unknown')})"
                )
            if len(relationships) > 5:
                self.logger.thought(f"  ... and {len(relationships) - 5} more")

            # Log base table candidates
            self.logger.info(f"Identified {len(base_candidates)} base table candidate(s)")
            for candidate in base_candidates[:3]:
                self.logger.thought(
                    f"  {candidate.get('table')}: score={candidate.get('score', 0):.2f} "
                    f"- {', '.join(candidate.get('reasons', ['unknown']))}"
                )

            self.logger.summary(
                f"Relationship discovery complete. "
                f"{len(tables)} tables, {len(relationships)} relationships, "
                f"{len(base_candidates)} base table candidates."
            )

            return {
                "tables": tables,
                "relationships": relationships,
                "base_table_candidates": base_candidates,
                "relationships_summary": relationships_result,
            }

        except Exception as e:
            self.logger.error(f"Relationship discovery failed: {str(e)}")
            raise

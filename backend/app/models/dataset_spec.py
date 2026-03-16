"""Dataset specification model."""
from sqlalchemy import Column, String, Text, ForeignKey, Integer, Boolean
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class DatasetSpec(Base, TimestampMixin):
    """Dataset specification model - defines how to build a dataset from sources."""

    __tablename__ = "dataset_specs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)

    # Lineage tracking - links to parent dataset spec for version history
    parent_dataset_spec_id = Column(
        GUID(),
        ForeignKey("dataset_specs.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Data source references (list of data source IDs and join config)
    data_sources_json = Column(JSONType(), nullable=True)

    # Column specifications
    target_column = Column(String(255), nullable=True)
    # PostgreSQL ARRAY type for feature columns
    feature_columns = Column(ARRAY(String), nullable=True)

    # Filters and transformations
    filters_json = Column(JSONType(), nullable=True)

    # Full specification (for complex cases)
    spec_json = Column(JSONType(), nullable=True)

    # Agent experiment design configuration
    # Stores the experiment design outputs from agent pipeline so experiments can be created
    # even after new agent runs have occurred
    agent_experiment_design_json = Column(JSONType(), nullable=True)

    # Time-based task metadata
    # Indicates whether this is a time-series / temporal prediction task
    is_time_based = Column(Boolean, default=False, nullable=False)
    # The datetime column used for temporal ordering (e.g., "date", "timestamp")
    time_column = Column(String(255), nullable=True)
    # Column identifying unique entities for panel/longitudinal data (e.g., "ticker", "user_id")
    entity_id_column = Column(String(255), nullable=True)
    # Prediction horizon in human-readable format (e.g., "1d", "5d", "1w", "next_bar")
    prediction_horizon = Column(String(100), nullable=True)
    # For classification: the positive class value (e.g., "up", "1", "True")
    target_positive_class = Column(String(255), nullable=True)

    # Version tracking
    version = Column(Integer, default=1, nullable=False)

    # Lineage history - tracks what changes were made from parent
    lineage_json = Column(JSONType(), nullable=True)
    # Example: {
    #   "created_from": "parent_dataset_spec_uuid",
    #   "changes": [
    #     {"type": "add_features", "features": ["momentum_5d"], "source": "insight_uuid"},
    #     {"type": "change_split", "from": "random", "to": "temporal"}
    #   ],
    #   "reasoning": "Combining momentum features that showed +3% in other datasets"
    # }

    # Relationships
    project = relationship("Project", back_populates="dataset_specs")
    experiments = relationship("Experiment", back_populates="dataset_spec")

    # Self-referential relationship for lineage
    parent_dataset_spec = relationship(
        "DatasetSpec",
        remote_side="DatasetSpec.id",
        backref="child_dataset_specs",
        foreign_keys="DatasetSpec.parent_dataset_spec_id"
    )

    # Auto DS iteration experiments link
    auto_ds_iteration_experiments = relationship(
        "AutoDSIterationExperiment",
        back_populates="dataset_spec"
    )

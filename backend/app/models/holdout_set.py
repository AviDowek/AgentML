"""Holdout set model for storing user-held-out validation data."""
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class HoldoutSet(Base, TimestampMixin):
    """Holdout set model - stores 5% of data held out before pipeline processing.

    This data is kept separate from the training pipeline and is only accessible
    to the user for manual model validation/testing.
    """

    __tablename__ = "holdout_sets"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        GUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    data_source_id = Column(
        GUID(),
        ForeignKey("data_sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Holdout configuration
    holdout_percentage = Column(Float, default=5.0, nullable=False)  # Percentage held out (default 5%)
    total_rows_original = Column(Integer, nullable=False)  # Original dataset row count
    holdout_row_count = Column(Integer, nullable=False)  # Number of rows held out
    training_row_count = Column(Integer, nullable=False)  # Number of rows remaining for training

    # The actual holdout data stored as JSON array of row objects
    # Format: [{"col1": val1, "col2": val2, ...}, ...]
    holdout_data_json = Column(JSONType(), nullable=False)

    # Column info for reference
    target_column = Column(String(255), nullable=True)  # The target column name (for validation display)
    feature_columns_json = Column(JSONType(), nullable=True)  # List of feature column names

    # Random seed used for reproducible splitting
    random_seed = Column(Integer, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="holdout_sets")
    data_source = relationship("DataSource", back_populates="holdout_sets")

    def __repr__(self):
        return f"<HoldoutSet(project_id={self.project_id}, holdout_rows={self.holdout_row_count})>"

    def get_holdout_rows(self):
        """Get the holdout data as a list of dicts."""
        return self.holdout_data_json or []

    def get_row(self, index: int):
        """Get a specific row from the holdout set."""
        data = self.holdout_data_json or []
        if 0 <= index < len(data):
            return data[index]
        return None

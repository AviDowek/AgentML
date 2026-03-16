"""Validation sample model for storing predictions on validation data."""
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Index
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class ValidationSample(Base, TimestampMixin):
    """Validation sample model - stores predictions on validation set for a model.

    This allows inspection of model performance on individual validation samples,
    enabling error analysis, debugging, and interactive exploration in the UI.
    """

    __tablename__ = "validation_samples"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    model_version_id = Column(
        GUID(),
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Position in the validation set (for ordering/reference)
    row_index = Column(Integer, nullable=False)

    # Features used for prediction (stored as JSON dict)
    features_json = Column(JSONType(), nullable=False)

    # Target and prediction values
    # For regression: numeric values
    # For classification: can be string (class label) or numeric
    target_value = Column(String(500), nullable=False)  # Store as string for flexibility
    predicted_value = Column(String(500), nullable=False)

    # Error metrics
    # For regression: predicted - target
    # For classification: 0 if correct, 1 if incorrect (or distance metric)
    error_value = Column(Float, nullable=True)
    absolute_error = Column(Float, nullable=True)

    # Optional: prediction probabilities for classification
    # Format: {"class_a": 0.7, "class_b": 0.3}
    prediction_probabilities_json = Column(JSONType(), nullable=True)

    # Relationships
    model_version = relationship("ModelVersion", back_populates="validation_samples")

    # Indexes for efficient querying
    __table_args__ = (
        Index('ix_validation_samples_model_row', 'model_version_id', 'row_index'),
        Index('ix_validation_samples_error', 'model_version_id', 'absolute_error'),
    )

    def __repr__(self):
        return f"<ValidationSample(model_version_id={self.model_version_id}, row_index={self.row_index})>"

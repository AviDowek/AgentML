"""Model version model."""
from sqlalchemy import Column, String, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import uuid
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class ModelStatus(str, enum.Enum):
    """Model deployment status."""
    TRAINED = "trained"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PRODUCTION = "production"
    RETIRED = "retired"


class ModelVersion(Base, TimestampMixin):
    """Model version model - represents a trained model artifact."""

    __tablename__ = "model_versions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    experiment_id = Column(GUID(), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True)
    trial_id = Column(GUID(), ForeignKey("trials.id", ondelete="SET NULL"), nullable=True)

    name = Column(String(255), nullable=False)
    model_type = Column(String(100), nullable=True)  # e.g., "LightGBM", "XGBoost", "CatBoost"

    # Artifacts
    artifact_location = Column(String(500), nullable=True)  # Path to model file

    # Metrics and analysis
    metrics_json = Column(JSONType(), nullable=True)
    feature_importances_json = Column(JSONType(), nullable=True)

    # Deployment status
    status = Column(
        SQLEnum(ModelStatus, values_callable=lambda x: [e.value for e in x]),
        default=ModelStatus.TRAINED,
        nullable=False
    )
    serving_config_json = Column(JSONType(), nullable=True)  # Configuration for serving

    # Relationships
    project = relationship("Project", back_populates="model_versions")
    experiment = relationship("Experiment", back_populates="model_versions")
    trial = relationship("Trial", back_populates="model_versions")
    validation_samples = relationship(
        "ValidationSample",
        back_populates="model_version",
        cascade="all, delete-orphan",
        lazy="dynamic"  # Use dynamic loading for potentially large sets
    )

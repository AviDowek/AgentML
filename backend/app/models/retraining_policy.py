"""Retraining policy model."""
from sqlalchemy import Column, String, ForeignKey, Enum as SQLEnum, DateTime
from sqlalchemy.orm import relationship
import uuid
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class PolicyType(str, enum.Enum):
    """Retraining policy types."""
    SCHEDULED = "scheduled"  # Cron-based
    METRIC_THRESHOLD = "metric_threshold"  # Trigger when metric degrades
    DATA_DRIFT = "data_drift"  # Trigger when data distribution changes
    MANUAL = "manual"


class RetrainingPolicy(Base, TimestampMixin):
    """Retraining policy model - defines when to retrain models."""

    __tablename__ = "retraining_policies"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)

    name = Column(String(255), nullable=False)
    policy_type = Column(
        SQLEnum(PolicyType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )

    # Scheduling (for SCHEDULED type)
    schedule_cron = Column(String(100), nullable=True)  # e.g., "0 0 * * 0" (weekly)

    # Thresholds (for METRIC_THRESHOLD type)
    metric_thresholds_json = Column(JSONType(), nullable=True)

    # Execution tracking
    last_retrain_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="retraining_policies")

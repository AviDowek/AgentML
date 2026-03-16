"""Visualization model for persistent storage of generated visualizations."""
import uuid
from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class Visualization(Base, TimestampMixin):
    """A saved visualization for a project."""

    __tablename__ = "visualizations"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    data_source_id = Column(GUID, ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True)
    owner_id = Column(GUID, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Visualization metadata
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    chart_type = Column(String(50), nullable=True)

    # The request that generated this visualization
    request = Column(Text, nullable=True)

    # Generated code
    code = Column(Text, nullable=False)

    # Base64-encoded image
    image_base64 = Column(Text, nullable=True)

    # AI-generated explanation
    explanation = Column(Text, nullable=True)

    # Whether this was from an AI suggestion
    is_ai_suggested = Column(String(10), default="false")

    # Order for display (lower = earlier)
    display_order = Column(String(20), default="0")

    # Relationships
    project = relationship("Project", back_populates="visualizations")
    data_source = relationship("DataSource", backref="visualizations")
    owner = relationship("User", backref="visualizations")

    def __repr__(self):
        return f"<Visualization {self.id}: {self.title}>"

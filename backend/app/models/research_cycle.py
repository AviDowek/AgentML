"""Research Cycle and Lab Notebook models for project-level research memory."""
from datetime import datetime
from sqlalchemy import Column, String, Text, ForeignKey, Enum as SQLEnum, Integer, DateTime
from sqlalchemy.orm import relationship
import uuid
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, GUID


class ResearchCycleStatus(str, enum.Enum):
    """Research cycle status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LabNotebookAuthorType(str, enum.Enum):
    """Author type for lab notebook entries."""
    AGENT = "agent"
    HUMAN = "human"


class ResearchCycle(Base, TimestampMixin):
    """Research cycle model - represents a research iteration within a project.

    Each cycle groups related experiments, agent runs, and findings together,
    providing a structured way to track the evolution of ML research.
    """

    __tablename__ = "research_cycles"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)

    # Cycle ordering within project
    sequence_number = Column(Integer, nullable=False)

    # Status tracking
    status = Column(
        SQLEnum(ResearchCycleStatus, values_callable=lambda x: [e.value for e in x]),
        default=ResearchCycleStatus.PENDING,
        nullable=False
    )

    # Optional summary
    summary_title = Column(String(500), nullable=True)

    # Relationships
    project = relationship("Project", back_populates="research_cycles")
    cycle_experiments = relationship(
        "CycleExperiment",
        back_populates="research_cycle",
        cascade="all, delete-orphan"
    )
    lab_notebook_entries = relationship(
        "LabNotebookEntry",
        back_populates="research_cycle",
        cascade="all, delete-orphan"
    )
    agent_runs = relationship("AgentRun", back_populates="research_cycle")

    # Auto DS sessions linked to this cycle
    auto_ds_sessions = relationship("AutoDSSession", back_populates="research_cycle")

    @property
    def experiment_count(self) -> int:
        """Get the number of experiments linked to this cycle."""
        return len(self.cycle_experiments) if self.cycle_experiments else 0


class CycleExperiment(Base):
    """Link table between research cycles and experiments.

    This allows experiments to be associated with specific research cycles,
    enabling tracking of which experiments belong to which iteration of research.
    """

    __tablename__ = "cycle_experiments"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    research_cycle_id = Column(
        GUID(),
        ForeignKey("research_cycles.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    experiment_id = Column(
        GUID(),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # When this experiment was linked to the cycle
    linked_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    research_cycle = relationship("ResearchCycle", back_populates="cycle_experiments")
    experiment = relationship("Experiment", back_populates="cycle_experiments")


class LabNotebookEntry(Base, TimestampMixin):
    """Lab notebook entry model - stores research notes, findings, and insights.

    Entries can be created by agents (automatically during pipeline execution)
    or by humans (manually through the UI). They provide a rich history of
    research decisions, observations, and conclusions.
    """

    __tablename__ = "lab_notebook_entries"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)

    # Optional link to a specific research cycle (can be None for general notes)
    research_cycle_id = Column(
        GUID(),
        ForeignKey("research_cycles.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Optional link to the agent step that created this entry
    agent_step_id = Column(
        GUID(),
        ForeignKey("agent_steps.id", ondelete="SET NULL"),
        nullable=True
    )

    # Author information
    author_type = Column(
        SQLEnum(LabNotebookAuthorType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )

    # Content
    title = Column(String(500), nullable=False)
    body_markdown = Column(Text, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="lab_notebook_entries")
    research_cycle = relationship("ResearchCycle", back_populates="lab_notebook_entries")
    agent_step = relationship("AgentStep", back_populates="lab_notebook_entries")

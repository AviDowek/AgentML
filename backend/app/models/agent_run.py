"""Agent Run, Step, and Log models for multi-step agent pipelines."""
from datetime import datetime
from sqlalchemy import Column, String, Text, ForeignKey, Enum as SQLEnum, Integer, DateTime
from sqlalchemy.orm import relationship
import uuid
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class AgentRunStatus(str, enum.Enum):
    """Agent run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineOrchestrationMode(str, enum.Enum):
    """Pipeline orchestration modes."""
    SEQUENTIAL = "sequential"  # Default: agents run in fixed order
    PROJECT_MANAGER = "project_manager"  # Dynamic: PM decides agent flow


class DebateMode(str, enum.Enum):
    """Debate system modes."""
    DISABLED = "disabled"  # No debate (default)
    ENABLED = "enabled"  # Critique agent + judge enabled


class DebatePartner(str, enum.Enum):
    """Available debate partner LLMs for critique."""
    GEMINI_FLASH = "gemini-2.0-flash"  # Default: Fast Gemini model
    GEMINI_PRO = "gemini-2.0-pro"  # More capable Gemini model
    CLAUDE_SONNET = "claude-sonnet-4"  # Anthropic Claude Sonnet
    GPT_4O = "gpt-4o"  # OpenAI GPT-4o
    GPT_5_1 = "gpt-5.1"  # OpenAI GPT-5.1


class AgentStepType(str, enum.Enum):
    """Types of agent steps in the pipeline."""
    # Pre-setup discovery step
    DATASET_DISCOVERY = "dataset_discovery"
    # Data analysis step (first step in pipeline - interactive)
    DATA_ANALYSIS = "data_analysis"
    # Setup pipeline steps
    PROBLEM_UNDERSTANDING = "problem_understanding"
    DATA_AUDIT = "data_audit"
    DATASET_DESIGN = "dataset_design"
    DATASET_VALIDATION = "dataset_validation"  # Validates dataset design against actual data sources
    EXPERIMENT_DESIGN = "experiment_design"
    PLAN_CRITIC = "plan_critic"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    # Results pipeline steps (post-experiment)
    RESULTS_INTERPRETATION = "results_interpretation"
    RESULTS_CRITIC = "results_critic"
    # Training dataset planning step (uses relationship discovery)
    TRAINING_DATASET_PLANNING = "training_dataset_planning"
    # Training dataset build step (materializes the dataset)
    TRAINING_DATASET_BUILD = "training_dataset_build"
    # Data Architect pipeline steps
    DATASET_INVENTORY = "dataset_inventory"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"
    # Auto-improve pipeline steps (post-training improvement)
    IMPROVEMENT_ANALYSIS = "improvement_analysis"  # Analyzes what to improve based on results
    IMPROVEMENT_PLAN = "improvement_plan"  # Creates actionable improvement plan
    # Enhanced improvement pipeline - uses full agent pipeline with iteration context
    ITERATION_CONTEXT = "iteration_context"  # Gathers all iteration history, errors, and insights
    IMPROVEMENT_DATA_ANALYSIS = "improvement_data_analysis"  # Re-analyze data with iteration feedback
    IMPROVEMENT_DATASET_DESIGN = "improvement_dataset_design"  # Redesign features with iteration feedback
    FEATURE_VALIDATION = "feature_validation"  # Quick validation of engineered features before use
    IMPROVEMENT_EXPERIMENT_DESIGN = "improvement_experiment_design"  # Redesign experiment with feedback
    # Lab notebook summary step - summarizes a research cycle
    LAB_NOTEBOOK_SUMMARY = "lab_notebook_summary"  # Generates Markdown summary of a research cycle
    # Robustness audit step - checks for overfitting and suspicious results
    ROBUSTNESS_AUDIT = "robustness_audit"  # Audits experiments for overfitting, baselines, and suspicious patterns
    # Orchestration system steps (Project Manager + Debate System)
    PROJECT_MANAGER = "project_manager"  # Orchestrates dynamic agent flow
    GEMINI_CRITIQUE = "gemini_critique"  # Gemini critique agent for debate
    OPENAI_JUDGE = "openai_judge"  # OpenAI judge for resolving disagreements
    DEBATE_ROUND = "debate_round"  # A round of debate between main agent and critique


class AgentStepStatus(str, enum.Enum):
    """Agent step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LogMessageType(str, enum.Enum):
    """Types of log messages for agent step logging.

    Message types enable rich visualization of agent reasoning:
    - THINKING: Internal reasoning, step-by-step analysis
    - HYPOTHESIS: Candidate explanations or theories
    - ACTION: Specific actions/commands the agent is taking
    - SUMMARY: Final narrative summary for the step
    - INFO: General informational messages
    - WARNING: Potential issues
    - ERROR: Errors encountered
    - THOUGHT: Legacy alias for thinking (deprecated, use THINKING)
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    THOUGHT = "thought"  # Deprecated: use THINKING
    SUMMARY = "summary"
    # New rich thinking types
    THINKING = "thinking"
    HYPOTHESIS = "hypothesis"
    ACTION = "action"


class AgentRun(Base, TimestampMixin):
    """Agent run model - represents a multi-step agent pipeline execution."""

    __tablename__ = "agent_runs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True)
    experiment_id = Column(GUID(), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True)
    research_cycle_id = Column(GUID(), ForeignKey("research_cycles.id", ondelete="SET NULL"), nullable=True, index=True)

    # Run metadata
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)

    status = Column(
        SQLEnum(AgentRunStatus, values_callable=lambda x: [e.value for e in x]),
        default=AgentRunStatus.PENDING,
        nullable=False
    )

    # Overall run configuration and results
    config_json = Column(JSONType(), nullable=True)  # Initial configuration for the run
    result_json = Column(JSONType(), nullable=True)  # Final aggregated results
    error_message = Column(Text, nullable=True)

    # Pipeline orchestration settings
    orchestration_mode = Column(
        SQLEnum(PipelineOrchestrationMode, values_callable=lambda x: [e.value for e in x]),
        default=PipelineOrchestrationMode.SEQUENTIAL,
        nullable=False
    )
    debate_mode = Column(
        SQLEnum(DebateMode, values_callable=lambda x: [e.value for e in x]),
        default=DebateMode.DISABLED,
        nullable=False
    )
    judge_model = Column(String(100), nullable=True)  # OpenAI model for judge (e.g., "gpt-4o", "o1")
    debate_partner = Column(String(100), nullable=True, default="gemini-2.0-flash")  # LLM model for critique partner
    max_debate_rounds = Column(Integer, default=3, nullable=False)  # Max rounds before calling judge (default: 3)
    debate_transcript_json = Column(JSONType(), nullable=True)  # Stores all debate rounds

    # Relationships
    project = relationship("Project", back_populates="agent_runs")
    experiment = relationship("Experiment", back_populates="agent_runs")
    research_cycle = relationship("ResearchCycle", back_populates="agent_runs")
    steps = relationship(
        "AgentStep",
        back_populates="agent_run",
        cascade="all, delete-orphan",
        order_by="AgentStep.created_at"
    )


class AgentStep(Base, TimestampMixin):
    """Agent step model - represents a single step in the agent pipeline."""

    __tablename__ = "agent_steps"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    agent_run_id = Column(GUID(), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)

    step_type = Column(
        SQLEnum(AgentStepType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )

    status = Column(
        SQLEnum(AgentStepStatus, values_callable=lambda x: [e.value for e in x]),
        default=AgentStepStatus.PENDING,
        nullable=False
    )

    # Timing
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    # Input/Output
    input_json = Column(JSONType(), nullable=True)
    output_json = Column(JSONType(), nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)

    # Relationships
    agent_run = relationship("AgentRun", back_populates="steps")
    logs = relationship(
        "AgentStepLog",
        back_populates="agent_step",
        cascade="all, delete-orphan",
        order_by="AgentStepLog.sequence"
    )
    lab_notebook_entries = relationship("LabNotebookEntry", back_populates="agent_step")

    def start(self):
        """Mark step as started."""
        self.status = AgentStepStatus.RUNNING
        self.started_at = datetime.utcnow()

    def complete(self, output: dict = None):
        """Mark step as completed."""
        self.status = AgentStepStatus.COMPLETED
        self.finished_at = datetime.utcnow()
        if output:
            self.output_json = output

    def fail(self, error: str):
        """Mark step as failed."""
        self.status = AgentStepStatus.FAILED
        self.finished_at = datetime.utcnow()
        self.error_message = error


class AgentStepLog(Base):
    """Agent step log model - represents a log entry within a step."""

    __tablename__ = "agent_step_logs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    agent_step_id = Column(GUID(), ForeignKey("agent_steps.id", ondelete="CASCADE"), nullable=False)

    # Log ordering
    sequence = Column(Integer, nullable=False)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Message content
    message_type = Column(
        SQLEnum(LogMessageType, values_callable=lambda x: [e.value for e in x]),
        default=LogMessageType.INFO,
        nullable=False
    )
    message = Column(Text, nullable=False)

    # Optional structured data
    metadata_json = Column(JSONType(), nullable=True)

    # Relationships
    agent_step = relationship("AgentStep", back_populates="logs")

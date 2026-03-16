"""Auto DS Session models for autonomous ML research.

This module provides models for the Auto DS Team feature which enables
autonomous cross-dataset experimentation with learning and iteration.

The hierarchy is:
- AutoDSSession: Top-level session with configuration and stopping conditions
- AutoDSIteration: Each iteration within a session (experiments, analysis, strategy)
- ResearchInsight: Structured insights discovered during analysis
"""
from datetime import datetime
from sqlalchemy import Column, String, Text, ForeignKey, Integer, Float, Boolean, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import relationship
import uuid
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class AutoDSSessionStatus(str, enum.Enum):
    """Status of an Auto DS Session."""
    PENDING = "pending"           # Session created but not started
    RUNNING = "running"           # Currently executing
    PAUSED = "paused"            # User-paused
    COMPLETED = "completed"       # Finished successfully (hit stop condition)
    FAILED = "failed"            # Error occurred
    STOPPED = "stopped"          # User manually stopped


class AutoDSIterationStatus(str, enum.Enum):
    """Status of a single iteration within a session."""
    PENDING = "pending"
    RUNNING_EXPERIMENTS = "running_experiments"  # Executing experiments
    ANALYZING = "analyzing"                       # Cross-analysis phase
    STRATEGIZING = "strategizing"                 # Planning next iteration
    COMPLETED = "completed"
    FAILED = "failed"


class InsightType(str, enum.Enum):
    """Types of research insights."""
    FEATURE_IMPORTANCE = "feature_importance"      # Which features matter
    FEATURE_PATTERN = "feature_pattern"            # Feature engineering patterns
    SPLIT_STRATEGY = "split_strategy"              # Validation strategy findings
    MODEL_CONFIG = "model_config"                  # AutoML configuration insights
    DATA_QUALITY = "data_quality"                  # Data issues discovered
    TARGET_INSIGHT = "target_insight"              # Target variable insights
    PITFALL = "pitfall"                           # What to avoid
    HYPOTHESIS = "hypothesis"                      # Untested ideas
    GENERAL = "general"                           # Other findings


class InsightConfidence(str, enum.Enum):
    """Confidence level for insights."""
    HIGH = "high"         # Observed multiple times, statistically significant
    MEDIUM = "medium"     # Observed but not conclusive
    LOW = "low"          # Initial observation, needs more evidence
    HYPOTHESIS = "hypothesis"  # Untested theory


class ExecutionMode(str, enum.Enum):
    """Execution mode for Auto DS sessions.

    Controls how experiments are executed within an iteration:
    - LEGACY: Run all planned experiments sequentially, then analyze (current behavior)
    - ADAPTIVE: After each experiment, check if results are declining - skip remaining if so
    - PHASED: Run baseline variants first, analyze, then run targeted variants for promising datasets
    - DYNAMIC: AI designs each experiment based on previous results (maximum accuracy)
    """
    LEGACY = "legacy"       # Run all experiments, then analyze
    ADAPTIVE = "adaptive"   # Early stopping if results decline
    PHASED = "phased"       # Baseline → analysis → targeted variants
    DYNAMIC = "dynamic"     # AI plans each experiment dynamically based on results


class ValidationStrategy(str, enum.Enum):
    """Validation strategy for experiments.

    Controls how experiments are validated for robust, production-ready scores:
    - STANDARD: Default AutoML validation (single CV, single seed)
    - ROBUST: Multiple CV strategies and random seeds for stability
    - STRICT: Time-aware splits, leakage detection, multiple seeds
    """
    STANDARD = "standard"   # Default - single CV, single seed
    ROBUST = "robust"       # Multiple CV folds, multiple seeds
    STRICT = "strict"       # Time-aware, leakage detection, production-grade


class AutoDSSession(Base, TimestampMixin):
    """Auto DS Session - orchestrates autonomous ML research.

    This is the top-level entity for an autonomous research run.
    It tracks configuration, progress, and links to all iterations.
    """

    __tablename__ = "auto_ds_sessions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(GUID(), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)

    # Session identification
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Status
    status = Column(
        SQLEnum(AutoDSSessionStatus, values_callable=lambda x: [e.value for e in x]),
        default=AutoDSSessionStatus.PENDING,
        nullable=False
    )

    # Configuration - Stop conditions
    max_iterations = Column(Integer, default=10, nullable=False)
    accuracy_threshold = Column(Float, nullable=True)  # Stop if this score is achieved
    time_budget_minutes = Column(Integer, nullable=True)  # Optional time budget
    min_improvement_threshold = Column(Float, default=0.001, nullable=False)  # Min improvement to continue
    plateau_iterations = Column(Integer, default=3, nullable=False)  # Stop after N iterations without improvement

    # Configuration - Execution
    max_experiments_per_dataset = Column(Integer, default=3, nullable=False)  # How many experiment configs to try per dataset
    max_active_datasets = Column(Integer, default=5, nullable=False)  # Max datasets to maintain

    # Execution mode - controls how experiments are run within iterations
    execution_mode = Column(
        SQLEnum(ExecutionMode, values_callable=lambda x: [e.value for e in x]),
        default=ExecutionMode.LEGACY,
        nullable=False
    )
    # Adaptive mode settings: skip remaining if score drops below best * (1 - threshold)
    adaptive_decline_threshold = Column(Float, default=0.05, nullable=False)  # 5% decline triggers skip
    # Phased mode settings: minimum baseline score improvement to continue with variants
    phased_min_baseline_improvement = Column(Float, default=0.01, nullable=False)  # 1% improvement needed
    # Dynamic mode settings: how many experiments per AI planning cycle
    dynamic_experiments_per_cycle = Column(Integer, default=1, nullable=False)  # Default: 1 experiment at a time

    # Validation strategy - controls how robust the validation is
    validation_strategy = Column(
        SQLEnum(ValidationStrategy, values_callable=lambda x: [e.value for e in x]),
        default=ValidationStrategy.STANDARD,
        nullable=False
    )
    # Validation settings
    validation_num_seeds = Column(Integer, default=1, nullable=False)  # Number of random seeds to run
    validation_cv_folds = Column(Integer, default=5, nullable=False)  # Number of CV folds

    # Tier 1 Feature flags - accuracy improvements
    enable_feature_engineering = Column(Boolean, default=True, nullable=False)  # AI proposes engineered features
    enable_ensemble = Column(Boolean, default=True, nullable=False)  # Build ensembles from top models
    enable_ablation = Column(Boolean, default=True, nullable=False)  # Run ablation studies

    # Tier 2 Feature flags
    enable_diverse_configs = Column(Boolean, default=True, nullable=False)  # AI generates diverse experiment configs

    # Context document settings
    use_context_documents = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether to use context documents in AI analysis and experiment design"
    )
    context_ab_testing = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="A/B testing: create half experiments WITH context, half WITHOUT"
    )

    # Progress tracking
    current_iteration = Column(Integer, default=0, nullable=False)
    best_score = Column(Float, nullable=True)  # Primary score (holdout preferred)
    best_train_score = Column(Float, nullable=True)  # Best training score
    best_val_score = Column(Float, nullable=True)  # Best validation score
    best_holdout_score = Column(Float, nullable=True)  # Best holdout score (gold standard)
    best_experiment_id = Column(GUID(), nullable=True)  # Link to best experiment
    total_experiments_run = Column(Integer, default=0, nullable=False)
    iterations_without_improvement = Column(Integer, default=0, nullable=False)

    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Optional link to research cycle for notebook entries
    research_cycle_id = Column(
        GUID(),
        ForeignKey("research_cycles.id", ondelete="SET NULL"),
        nullable=True
    )

    # Celery task tracking
    celery_task_id = Column(String(255), nullable=True)

    # Additional config (for future extensibility)
    config_json = Column(JSONType(), nullable=True)

    # Relationships
    project = relationship(
        "Project",
        back_populates="auto_ds_sessions",
        foreign_keys=[project_id]
    )
    iterations = relationship(
        "AutoDSIteration",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AutoDSIteration.iteration_number"
    )
    insights = relationship(
        "ResearchInsight",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    research_cycle = relationship("ResearchCycle", back_populates="auto_ds_sessions")

    @property
    def elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds() / 60.0

    @property
    def should_stop(self) -> bool:
        """Check if any stopping condition is met."""
        # Max iterations
        if self.current_iteration >= self.max_iterations:
            return True

        # Accuracy threshold
        if self.accuracy_threshold and self.best_score:
            if self.best_score >= self.accuracy_threshold:
                return True

        # Time budget
        if self.time_budget_minutes and self.elapsed_minutes >= self.time_budget_minutes:
            return True

        # Plateau
        if self.iterations_without_improvement >= self.plateau_iterations:
            return True

        return False


class AutoDSIteration(Base, TimestampMixin):
    """A single iteration within an Auto DS Session.

    Each iteration consists of:
    1. Running experiments across active datasets
    2. Cross-analysis of results
    3. Strategy decisions for next iteration
    """

    __tablename__ = "auto_ds_iterations"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        GUID(),
        ForeignKey("auto_ds_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Iteration ordering
    iteration_number = Column(Integer, nullable=False)

    # Status
    status = Column(
        SQLEnum(AutoDSIterationStatus, values_callable=lambda x: [e.value for e in x]),
        default=AutoDSIterationStatus.PENDING,
        nullable=False
    )

    # Results summary
    experiments_planned = Column(Integer, default=0, nullable=False)
    experiments_completed = Column(Integer, default=0, nullable=False)
    experiments_failed = Column(Integer, default=0, nullable=False)
    best_score_this_iteration = Column(Float, nullable=True)  # Primary score (holdout preferred)
    best_train_score_this_iteration = Column(Float, nullable=True)
    best_val_score_this_iteration = Column(Float, nullable=True)
    best_holdout_score_this_iteration = Column(Float, nullable=True)
    best_experiment_id_this_iteration = Column(GUID(), nullable=True)

    # Phase tracking
    experiments_started_at = Column(DateTime, nullable=True)
    experiments_completed_at = Column(DateTime, nullable=True)
    analysis_started_at = Column(DateTime, nullable=True)
    analysis_completed_at = Column(DateTime, nullable=True)
    strategy_started_at = Column(DateTime, nullable=True)
    strategy_completed_at = Column(DateTime, nullable=True)

    # Analysis output
    analysis_summary_json = Column(JSONType(), nullable=True)  # Cross-analysis results

    # Strategy output
    strategy_decisions_json = Column(JSONType(), nullable=True)  # Decisions made for next iteration
    # Example: {
    #   "datasets_to_keep": ["uuid1", "uuid2"],
    #   "datasets_to_drop": [{"id": "uuid3", "reason": "persistent data leakage"}],
    #   "datasets_to_modify": [{"id": "uuid1", "changes": [...]}],
    #   "new_datasets_to_create": [{"parent_id": "uuid2", "changes": [...]}],
    #   "experiment_config_adjustments": {...}
    # }

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Relationships
    session = relationship("AutoDSSession", back_populates="iterations")
    iteration_experiments = relationship(
        "AutoDSIterationExperiment",
        back_populates="iteration",
        cascade="all, delete-orphan"
    )

    @property
    def total_duration_seconds(self) -> float:
        """Get total duration of this iteration."""
        if not self.experiments_started_at:
            return 0.0
        end = self.strategy_completed_at or datetime.utcnow()
        return (end - self.experiments_started_at).total_seconds()


class AutoDSIterationExperiment(Base, TimestampMixin):
    """Link table tracking experiments within an iteration.

    This tracks which experiments were run in which iteration,
    along with their role (which dataset, which config variant).
    """

    __tablename__ = "auto_ds_iteration_experiments"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    iteration_id = Column(
        GUID(),
        ForeignKey("auto_ds_iterations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    experiment_id = Column(
        GUID(),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    dataset_spec_id = Column(
        GUID(),
        ForeignKey("dataset_specs.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Experiment role/variant info
    experiment_variant = Column(Integer, default=1, nullable=False)  # 1, 2, 3 for same dataset
    experiment_hypothesis = Column(Text, nullable=True)  # What we're testing

    # Result tracking
    score = Column(Float, nullable=True)
    rank_in_iteration = Column(Integer, nullable=True)  # Ranking within this iteration

    # Relationships
    iteration = relationship("AutoDSIteration", back_populates="iteration_experiments")
    experiment = relationship("Experiment", back_populates="auto_ds_iteration_experiments")
    dataset_spec = relationship("DatasetSpec", back_populates="auto_ds_iteration_experiments")


class ResearchInsight(Base, TimestampMixin):
    """Structured research insight discovered during Auto DS analysis.

    Unlike free-form LabNotebookEntry, this is structured data that
    can be queried, compared, and potentially promoted to global insights.
    """

    __tablename__ = "research_insights"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        GUID(),
        ForeignKey("auto_ds_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    project_id = Column(
        GUID(),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Optional link to specific iteration where discovered
    iteration_id = Column(
        GUID(),
        ForeignKey("auto_ds_iterations.id", ondelete="SET NULL"),
        nullable=True
    )

    # Classification
    insight_type = Column(
        SQLEnum(InsightType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    confidence = Column(
        SQLEnum(InsightConfidence, values_callable=lambda x: [e.value for e in x]),
        default=InsightConfidence.LOW,
        nullable=False
    )

    # Content
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)

    # Structured data for the insight
    insight_data_json = Column(JSONType(), nullable=True)
    # Example for FEATURE_IMPORTANCE:
    # {
    #   "feature_name": "volume_ma_20",
    #   "avg_importance_rank": 3.5,
    #   "experiments_observed": ["uuid1", "uuid2", "uuid3"],
    #   "avg_improvement_when_present": 0.02
    # }

    # Evidence
    evidence_count = Column(Integer, default=1, nullable=False)  # How many observations support this
    supporting_experiments = Column(JSONType(), nullable=True)  # List of experiment IDs
    contradicting_experiments = Column(JSONType(), nullable=True)  # List of experiment IDs that contradict

    # For hypotheses
    is_tested = Column(Boolean, default=False, nullable=False)
    test_result = Column(Text, nullable=True)  # Outcome if tested

    # Global promotion
    promoted_to_global = Column(Boolean, default=False, nullable=False)
    global_insight_id = Column(
        GUID(),
        ForeignKey("global_insights.id", ondelete="SET NULL"),
        nullable=True
    )

    # Relationships
    session = relationship("AutoDSSession", back_populates="insights")
    project = relationship("Project", back_populates="research_insights")
    iteration = relationship("AutoDSIteration")
    global_insight = relationship("GlobalInsight", back_populates="source_insights")


class GlobalInsight(Base, TimestampMixin):
    """Cross-project insight that can be applied to new projects.

    These are abstracted, anonymized insights that transfer learning
    across projects. They don't contain project-specific details.
    """

    __tablename__ = "global_insights"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # Classification
    insight_type = Column(
        SQLEnum(InsightType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    category = Column(String(100), nullable=True)  # e.g., "moving_average", "temporal", "stacking"

    # Content (abstracted, no project-specific details)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)

    # Technical details (abstracted)
    technical_details_json = Column(JSONType(), nullable=True)
    # Example: {
    #   "pattern": "moving_average",
    #   "window_range": [15, 25],
    #   "applicable_data_types": ["time_series", "financial"]
    # }

    # Applicability criteria
    applicable_to = Column(JSONType(), nullable=True)  # ["time_series", "financial", "sequential"]
    task_types = Column(JSONType(), nullable=True)  # ["binary_classification", "regression"]
    data_characteristics = Column(JSONType(), nullable=True)  # {"has_time_column": true, ...}

    # Confidence and evidence
    evidence_count = Column(Integer, default=1, nullable=False)
    contradiction_count = Column(Integer, default=0, nullable=False)
    confidence_score = Column(Float, default=0.5, nullable=False)  # 0-1 scale

    # Tracking
    source_project_count = Column(Integer, default=1, nullable=False)  # How many projects contributed
    last_validated_at = Column(DateTime, nullable=True)

    # Usage stats
    times_applied = Column(Integer, default=0, nullable=False)
    times_successful = Column(Integer, default=0, nullable=False)

    # Active flag (can be deprecated if consistently wrong)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    source_insights = relationship("ResearchInsight", back_populates="global_insight")

    @property
    def success_rate(self) -> float:
        """Get success rate when this insight is applied."""
        if self.times_applied == 0:
            return 0.0
        return self.times_successful / self.times_applied

    @property
    def computed_confidence(self) -> float:
        """Compute confidence based on evidence."""
        total = self.evidence_count + self.contradiction_count
        if total == 0:
            return 0.5
        return self.evidence_count / total

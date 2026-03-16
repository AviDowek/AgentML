"""Auto DS session and related schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.auto_ds_session import (
    AutoDSSessionStatus,
    AutoDSIterationStatus,
    InsightType,
    InsightConfidence,
    ExecutionMode,
    ValidationStrategy,
)


# ============================================
# Auto DS Session Schemas
# ============================================

class AutoDSSessionCreate(BaseModel):
    """Schema for creating an Auto DS session."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    # Stop conditions
    max_iterations: int = Field(default=10, ge=1, le=100)
    accuracy_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    time_budget_minutes: Optional[int] = Field(default=None, ge=1)
    min_improvement_threshold: float = Field(default=0.001, ge=0)
    plateau_iterations: int = Field(default=3, ge=1)
    # Execution config
    max_experiments_per_dataset: int = Field(default=3, ge=1, le=10)
    max_active_datasets: int = Field(default=5, ge=1, le=20)
    # Execution mode settings
    execution_mode: ExecutionMode = Field(default=ExecutionMode.LEGACY)
    adaptive_decline_threshold: float = Field(default=0.05, ge=0, le=1)
    phased_min_baseline_improvement: float = Field(default=0.01, ge=0, le=1)
    dynamic_experiments_per_cycle: int = Field(default=1, ge=1, le=5)
    # Validation strategy settings (Tier 2)
    validation_strategy: ValidationStrategy = Field(default=ValidationStrategy.STANDARD)
    validation_num_seeds: int = Field(default=1, ge=1, le=10)
    validation_cv_folds: int = Field(default=5, ge=2, le=20)
    # Tier 1 feature flags
    enable_feature_engineering: bool = Field(default=True)
    enable_ensemble: bool = Field(default=True)
    enable_ablation: bool = Field(default=True)
    # Tier 2 feature flags
    enable_diverse_configs: bool = Field(default=True)
    # Optional link to research cycle
    research_cycle_id: Optional[UUID] = None
    # Additional config
    config_json: Optional[Dict[str, Any]] = None


class AutoDSSessionUpdate(BaseModel):
    """Schema for updating an Auto DS session."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    max_iterations: Optional[int] = Field(None, ge=1, le=100)
    accuracy_threshold: Optional[float] = Field(None, ge=0, le=1)
    time_budget_minutes: Optional[int] = Field(None, ge=1)
    # Execution mode settings
    execution_mode: Optional[ExecutionMode] = None
    adaptive_decline_threshold: Optional[float] = Field(None, ge=0, le=1)
    phased_min_baseline_improvement: Optional[float] = Field(None, ge=0, le=1)
    dynamic_experiments_per_cycle: Optional[int] = Field(None, ge=1, le=5)
    # Validation strategy settings
    validation_strategy: Optional[ValidationStrategy] = None
    validation_num_seeds: Optional[int] = Field(None, ge=1, le=10)
    validation_cv_folds: Optional[int] = Field(None, ge=2, le=20)
    # Feature flags
    enable_feature_engineering: Optional[bool] = None
    enable_ensemble: Optional[bool] = None
    enable_ablation: Optional[bool] = None
    enable_diverse_configs: Optional[bool] = None


class AutoDSSessionSummary(BaseModel):
    """Summary schema for listing Auto DS sessions."""
    id: UUID
    project_id: UUID
    name: str
    description: Optional[str] = None
    status: AutoDSSessionStatus
    execution_mode: ExecutionMode
    current_iteration: int
    max_iterations: int
    best_score: Optional[float] = None
    total_experiments_run: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AutoDSIterationSummary(BaseModel):
    """Summary of an iteration for session responses."""
    id: UUID
    iteration_number: int
    status: AutoDSIterationStatus
    experiments_planned: int
    experiments_completed: int
    experiments_failed: int
    best_score_this_iteration: Optional[float] = None
    # Individual scores for detailed display
    best_train_score_this_iteration: Optional[float] = None
    best_val_score_this_iteration: Optional[float] = None
    best_holdout_score_this_iteration: Optional[float] = None
    best_experiment_id_this_iteration: Optional[UUID] = None
    analysis_summary_json: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    # Experiments in this iteration (populated when loading iterations list)
    experiments: List["IterationExperimentInfo"] = Field(default_factory=list)

    class Config:
        from_attributes = True


class AutoDSSessionResponse(BaseModel):
    """Full Auto DS session response."""
    id: UUID
    project_id: UUID
    name: str
    description: Optional[str] = None
    status: AutoDSSessionStatus
    # Stop conditions
    max_iterations: int
    accuracy_threshold: Optional[float] = None
    time_budget_minutes: Optional[int] = None
    min_improvement_threshold: float
    plateau_iterations: int
    # Execution config
    max_experiments_per_dataset: int
    max_active_datasets: int
    # Execution mode
    execution_mode: ExecutionMode
    adaptive_decline_threshold: float
    phased_min_baseline_improvement: float
    dynamic_experiments_per_cycle: int
    # Validation strategy (Tier 2)
    validation_strategy: ValidationStrategy
    validation_num_seeds: int
    validation_cv_folds: int
    # Tier 1 feature flags
    enable_feature_engineering: bool
    enable_ensemble: bool
    enable_ablation: bool
    # Tier 2 feature flags
    enable_diverse_configs: bool
    # Progress
    current_iteration: int
    best_score: Optional[float] = None
    # Individual best scores for detailed display
    best_train_score: Optional[float] = None
    best_val_score: Optional[float] = None
    best_holdout_score: Optional[float] = None
    best_experiment_id: Optional[UUID] = None
    total_experiments_run: int
    iterations_without_improvement: int
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # Links
    research_cycle_id: Optional[UUID] = None
    celery_task_id: Optional[str] = None
    config_json: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    # Nested data
    iterations: List[AutoDSIterationSummary] = Field(default_factory=list)

    class Config:
        from_attributes = True


class AutoDSSessionListResponse(BaseModel):
    """Response for listing Auto DS sessions."""
    sessions: List[AutoDSSessionSummary]
    total: int


class AutoDSSessionStartRequest(BaseModel):
    """Request to start an Auto DS session."""
    initial_dataset_spec_ids: Optional[List[UUID]] = None


class AutoDSSessionStartResponse(BaseModel):
    """Response after starting an Auto DS session."""
    session_id: UUID
    status: AutoDSSessionStatus
    celery_task_id: str
    message: str


# ============================================
# Auto DS Iteration Schemas
# ============================================

class IterationExperimentInfo(BaseModel):
    """Summary info about an experiment in an iteration."""
    experiment_id: UUID
    experiment_name: str
    experiment_status: str
    dataset_spec_id: Optional[UUID] = None
    dataset_name: Optional[str] = None
    experiment_variant: int = 1
    hypothesis: Optional[str] = None
    score: Optional[float] = None
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    holdout_score: Optional[float] = None
    rank_in_iteration: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AutoDSIterationResponse(BaseModel):
    """Full iteration response with details."""
    id: UUID
    session_id: UUID
    iteration_number: int
    status: AutoDSIterationStatus
    experiments_planned: int
    experiments_completed: int
    experiments_failed: int
    best_score_this_iteration: Optional[float] = None
    # Individual scores for detailed display
    best_train_score_this_iteration: Optional[float] = None
    best_val_score_this_iteration: Optional[float] = None
    best_holdout_score_this_iteration: Optional[float] = None
    best_experiment_id_this_iteration: Optional[UUID] = None
    # Phase timing
    experiments_started_at: Optional[datetime] = None
    experiments_completed_at: Optional[datetime] = None
    analysis_started_at: Optional[datetime] = None
    analysis_completed_at: Optional[datetime] = None
    strategy_started_at: Optional[datetime] = None
    strategy_completed_at: Optional[datetime] = None
    # Output
    analysis_summary_json: Optional[Dict[str, Any]] = None
    strategy_decisions_json: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================
# Research Insight Schemas
# ============================================

class ResearchInsightCreate(BaseModel):
    """Schema for creating a research insight."""
    insight_type: InsightType
    confidence: InsightConfidence = InsightConfidence.LOW
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    insight_data_json: Optional[Dict[str, Any]] = None
    supporting_experiments: Optional[List[str]] = None
    contradicting_experiments: Optional[List[str]] = None


class ResearchInsightResponse(BaseModel):
    """Full research insight response."""
    id: UUID
    session_id: UUID
    project_id: UUID
    iteration_id: Optional[UUID] = None
    insight_type: InsightType
    confidence: InsightConfidence
    title: str
    description: Optional[str] = None
    insight_data_json: Optional[Dict[str, Any]] = None
    evidence_count: int
    supporting_experiments: Optional[List[str]] = None
    contradicting_experiments: Optional[List[str]] = None
    is_tested: bool
    test_result: Optional[str] = None
    promoted_to_global: bool
    global_insight_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ResearchInsightListResponse(BaseModel):
    """Response for listing research insights."""
    insights: List[ResearchInsightResponse]
    total: int


# ============================================
# Global Insight Schemas
# ============================================

class GlobalInsightResponse(BaseModel):
    """Global insight response."""
    id: UUID
    insight_type: str
    category: Optional[str] = None
    title: str
    description: Optional[str] = None
    technical_details_json: Optional[Dict[str, Any]] = None
    applicable_to: Optional[Dict[str, Any]] = None
    task_types: Optional[List[str]] = None
    data_characteristics: Optional[Dict[str, Any]] = None
    evidence_count: int
    contradiction_count: int
    confidence_score: float
    source_project_count: int
    last_validated_at: Optional[datetime] = None
    times_applied: int
    times_successful: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class GlobalInsightListResponse(BaseModel):
    """Response for listing global insights."""
    insights: List[GlobalInsightResponse]
    total: int


# ============================================
# Session Progress/Status Schemas
# ============================================

class AutoDSSessionProgress(BaseModel):
    """Detailed progress information for a running session."""
    session_id: UUID
    status: AutoDSSessionStatus
    current_iteration: int
    max_iterations: int
    best_score: Optional[float] = None
    total_experiments_run: int
    iterations_without_improvement: int
    # Current iteration details
    current_iteration_status: Optional[AutoDSIterationStatus] = None
    current_iteration_experiments_completed: int = 0
    current_iteration_experiments_planned: int = 0
    # Timing
    started_at: Optional[datetime] = None
    elapsed_minutes: Optional[float] = None
    time_budget_minutes: Optional[int] = None
    # Stopping condition status
    accuracy_threshold: Optional[float] = None
    threshold_reached: bool = False
    plateau_detected: bool = False

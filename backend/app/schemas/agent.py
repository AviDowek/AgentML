"""Agent schemas for LLM-powered configuration suggestions and multi-step pipelines."""
from datetime import datetime
from typing import Any, Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# Schema Summary (output of schema analysis for LLM consumption)
class ColumnSummary(BaseModel):
    """Summary of a single column for LLM context."""

    name: str
    dtype: str
    inferred_type: str  # numeric, categorical, datetime, text, boolean
    null_percentage: float
    unique_count: int
    # Type-specific stats (optional)
    # Note: min/max can be float (numeric), str (datetime), or None
    min: Optional[Any] = None
    max: Optional[Any] = None
    mean: Optional[float] = None
    top_values: Optional[dict[str, int]] = None
    # Datetime-specific (optional)
    min_date: Optional[str] = None
    max_date: Optional[str] = None


class SchemaSummary(BaseModel):
    """Summary of a data source schema for LLM context."""

    data_source_id: UUID
    data_source_name: str
    file_type: str
    row_count: int
    column_count: int
    columns: List[ColumnSummary]


# Feature Engineering Definition
class FeatureEngineeringStep(BaseModel):
    """A single feature engineering transformation."""

    output_column: str = Field(
        ...,
        description="Name of the new column to create"
    )
    formula: str = Field(
        ...,
        description="Python/pandas expression to compute the column (e.g., 'df[\"high\"] - df[\"low\"]')"
    )
    source_columns: List[str] = Field(
        ...,
        description="List of source columns used in the formula"
    )
    description: str = Field(
        ...,
        description="Human-readable description of what this feature represents"
    )

    @field_validator('formula')
    @classmethod
    def validate_formula_syntax(cls, v: str) -> str:
        """Validate formula is not empty and has basic syntax."""
        if not v or not v.strip():
            raise ValueError("Formula cannot be empty")

        v = v.strip()

        # Check for dangerous operations
        dangerous_patterns = [
            'import ', '__', 'eval(', 'exec(', 'compile(',
            'open(', 'os.', 'sys.', 'subprocess', 'shutil'
        ]
        for pattern in dangerous_patterns:
            if pattern in v.lower():
                raise ValueError(f"Formula contains disallowed pattern: {pattern}")

        # Basic syntax validation - check balanced brackets
        if v.count('[') != v.count(']'):
            raise ValueError("Formula has unbalanced square brackets")
        if v.count('(') != v.count(')'):
            raise ValueError("Formula has unbalanced parentheses")
        if v.count('"') % 2 != 0:
            raise ValueError("Formula has unbalanced double quotes")

        return v

    @field_validator('output_column')
    @classmethod
    def validate_output_column(cls, v: str) -> str:
        """Validate output column name is valid."""
        if not v or not v.strip():
            raise ValueError("Output column name cannot be empty")
        # Check for valid Python identifier (with some flexibility)
        v = v.strip()
        if v[0].isdigit():
            raise ValueError("Output column name cannot start with a digit")
        return v


class TargetColumnCreation(BaseModel):
    """Definition for creating a target column that doesn't exist."""

    column_name: str = Field(
        ...,
        description="Name of the target column to create"
    )
    formula: str = Field(
        ...,
        description="Python/pandas expression to compute the target (e.g., 'df[\"close\"].shift(-1) > df[\"close\"]')"
    )
    source_columns: List[str] = Field(
        ...,
        description="List of source columns used in the formula"
    )
    description: str = Field(
        ...,
        description="Human-readable description of what this target represents"
    )
    data_type: str = Field(
        default="infer",
        description="Expected data type: binary, categorical, numeric, or infer"
    )


# Project Config Suggestion (LLM output)
class ProjectConfigSuggestion(BaseModel):
    """LLM-suggested project configuration."""

    task_type: str = Field(
        ...,
        description="Inferred ML task type: binary, multiclass, regression, quantile, timeseries_forecast, multimodal_classification, multimodal_regression"
    )
    target_column: str = Field(
        ...,
        description="The column to predict (can be existing or to-be-created)"
    )
    target_exists: bool = Field(
        default=True,
        description="Whether the target column already exists in the data"
    )
    target_creation: Optional[TargetColumnCreation] = Field(
        None,
        description="If target_exists=False, how to create the target column"
    )
    primary_metric: str = Field(
        ...,
        description="The metric to optimize (e.g., accuracy, roc_auc, rmse, mse, f1)"
    )
    reasoning: str = Field(
        ...,
        description="Explanation for why these choices were made"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score 0-1"
    )
    # Optional suggestions
    suggested_name: Optional[str] = Field(
        None,
        description="Suggested project name based on the data and goal"
    )
    # Recommended feature engineering
    suggested_feature_engineering: List[FeatureEngineeringStep] = Field(
        default_factory=list,
        description="Suggested feature engineering steps to improve model performance"
    )

    # Time-based task metadata (set when task involves temporal/future prediction)
    is_time_based: bool = Field(
        default=False,
        description="True if this is a time-series or temporal prediction task (predicting future behavior based on past data)"
    )
    time_column: Optional[str] = Field(
        None,
        description="The datetime column used for temporal ordering (e.g., 'date', 'timestamp'). Required if is_time_based=True"
    )
    entity_id_column: Optional[str] = Field(
        None,
        description="Column identifying unique entities for panel/longitudinal data (e.g., 'ticker', 'user_id', 'customer_id')"
    )
    prediction_horizon: Optional[str] = Field(
        None,
        description="Human-readable prediction horizon (e.g., '1d' for next day, '5d' for 5 days, 'next_bar', '1w' for weekly)"
    )
    target_positive_class: Optional[str] = Field(
        None,
        description="For classification: the value representing the positive/target class (e.g., 'up', '1', 'True', 'churn')"
    )


class ProjectConfigRequest(BaseModel):
    """Request for project config suggestion."""

    description: str = Field(
        ...,
        min_length=10,
        description="User's description of what they want to predict/achieve"
    )
    data_source_id: UUID = Field(
        ...,
        description="ID of the data source to analyze"
    )


class ProjectConfigResponse(BaseModel):
    """Response for project config suggestion."""

    suggestion: ProjectConfigSuggestion
    schema_summary: SchemaSummary


# Dataset Spec Suggestion (LLM output)
class DatasetVariant(BaseModel):
    """A single dataset configuration variant."""

    name: str = Field(
        ...,
        description="Variant name (e.g., 'baseline', 'minimal_features', 'all_features')"
    )
    description: str = Field(
        ...,
        description="What this variant represents and why it might be useful"
    )
    feature_columns: List[str] = Field(
        ...,
        description="Existing columns to use as features for training"
    )
    engineered_features: List[FeatureEngineeringStep] = Field(
        default_factory=list,
        description="New features to create from existing columns"
    )
    excluded_columns: List[str] = Field(
        default_factory=list,
        description="Columns explicitly excluded (IDs, leaky features, etc.)"
    )
    exclusion_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="Reason for excluding each column"
    )
    train_test_split: str = Field(
        default="80_20",
        description="Train/test split ratio (e.g., '80_20', '70_30', '90_10')"
    )
    preprocessing_strategy: str = Field(
        default="auto",
        description="Preprocessing approach (e.g., 'auto', 'minimal', 'aggressive_imputation')"
    )
    suggested_filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional data filters as a dict (e.g., {'remove_nulls': True})"
    )
    expected_tradeoff: str = Field(
        ...,
        description="Expected tradeoff (e.g., 'more features may overfit', 'simpler but might miss patterns')"
    )


class DatasetSpecSuggestion(BaseModel):
    """LLM-suggested dataset specification - single variant (legacy)."""

    feature_columns: List[str] = Field(
        ...,
        description="Columns to use as features for training"
    )
    excluded_columns: List[str] = Field(
        default_factory=list,
        description="Columns explicitly excluded (IDs, leaky features, etc.)"
    )
    exclusion_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="Reason for excluding each column"
    )
    suggested_filters: Optional[dict[str, Any]] = Field(
        None,
        description="Optional data filters (e.g., remove nulls, date ranges)"
    )
    reasoning: str = Field(
        ...,
        description="Explanation for the feature selection"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Potential issues or considerations"
    )


class DatasetDesignSuggestion(BaseModel):
    """LLM-suggested dataset design with multiple variants."""

    variants: List[DatasetVariant] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of dataset configuration variants (up to 10)"
    )
    recommended_variant: str = Field(
        ...,
        description="Name of the recommended variant to start with"
    )
    reasoning: str = Field(
        ...,
        description="Overall explanation for the dataset design approach"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Potential issues or considerations across all variants"
    )


class DatasetSpecRequest(BaseModel):
    """Request for dataset spec suggestion."""

    data_source_id: UUID = Field(
        ...,
        description="ID of the data source"
    )
    task_type: str = Field(
        ...,
        description="The ML task type"
    )
    target_column: str = Field(
        ...,
        description="The target column to predict"
    )
    description: Optional[str] = Field(
        None,
        description="Optional additional context about the goal"
    )


class DatasetSpecResponse(BaseModel):
    """Response for dataset spec suggestion."""

    suggestion: DatasetSpecSuggestion
    schema_summary: SchemaSummary


# Validation Strategy (for train/test splitting)
class ValidationStrategy(BaseModel):
    """Validation strategy for data splitting.

    Split types:
    - 'time': Pure time-based split (sort by time_column, earliest N% train, latest N% test)
    - 'group_time': Time-based split respecting entity groups (for panel/longitudinal data)
    - 'random': Random shuffle split (ONLY for non-time-based tasks)
    - 'group_random': Random split keeping entity groups together
    - 'stratified': Random split preserving class proportions (classification only)
    - 'temporal': Legacy alias for 'time' (deprecated, use 'time')
    """

    split_strategy: str = Field(
        ...,
        description="Split type: 'time', 'group_time', 'random', 'group_random', 'stratified', or 'temporal' (legacy)"
    )
    validation_split: float = Field(
        default=0.2,
        description="Fraction of data for validation/test (0.1-0.3)"
    )
    time_column: Optional[str] = Field(
        default=None,
        description="DateTime column for time-based splits. REQUIRED if split_strategy is 'time' or 'group_time'"
    )
    entity_id_column: Optional[str] = Field(
        default=None,
        description="Column identifying unique entities (e.g., 'ticker', 'user_id'). Used for group-based splits."
    )
    group_column: Optional[str] = Field(
        default=None,
        description="Legacy alias for entity_id_column (deprecated)"
    )
    n_folds: Optional[int] = Field(
        default=None,
        description="Number of folds for cross-validation (if using CV instead of holdout)"
    )
    reasoning: str = Field(
        ...,
        description="Why this split strategy is appropriate for this data"
    )

    def get_entity_column(self) -> Optional[str]:
        """Get the entity/group column, preferring entity_id_column over legacy group_column."""
        return self.entity_id_column or self.group_column

    def is_time_based(self) -> bool:
        """Check if this is a time-based split strategy."""
        return self.split_strategy in ("time", "group_time", "temporal")


# Experiment Plan Suggestion (LLM output)
class ExperimentVariant(BaseModel):
    """A single experiment variant configuration."""

    name: str = Field(
        ...,
        description="Variant name (e.g., 'baseline', 'high_quality', 'fast_iteration')"
    )
    description: str = Field(
        ...,
        description="What this variant tests"
    )
    automl_config: dict[str, Any] = Field(
        ...,
        description="AutoML configuration for this variant"
    )
    validation_strategy: Optional[ValidationStrategy] = Field(
        default=None,
        description="Validation strategy for data splitting. REQUIRED for time-series/financial data."
    )
    expected_tradeoff: str = Field(
        ...,
        description="Expected tradeoff (e.g., 'faster training, possibly lower accuracy')"
    )


class ExperimentPlanSuggestion(BaseModel):
    """LLM-suggested experiment plan with multiple variants."""

    variants: List[ExperimentVariant] = Field(
        ...,
        min_length=1,
        description="List of experiment variants to try"
    )
    recommended_variant: str = Field(
        ...,
        description="Name of the recommended variant to start with"
    )
    reasoning: str = Field(
        ...,
        description="Explanation for the experiment plan"
    )
    estimated_total_time_minutes: int = Field(
        ...,
        description="Estimated total time to run all variants"
    )


class ExperimentPlanRequest(BaseModel):
    """Request for experiment plan suggestion."""

    task_type: str = Field(
        ...,
        description="The ML task type"
    )
    target_column: str = Field(
        ...,
        description="The target column"
    )
    primary_metric: str = Field(
        ...,
        description="The metric to optimize"
    )
    feature_columns: List[str] = Field(
        ...,
        description="Selected feature columns"
    )
    row_count: int = Field(
        ...,
        description="Number of rows in the dataset"
    )
    column_count: int = Field(
        ...,
        description="Number of feature columns"
    )
    time_budget_minutes: Optional[int] = Field(
        None,
        description="Optional time budget constraint"
    )
    description: Optional[str] = Field(
        None,
        description="Optional additional context"
    )


class ExperimentPlanResponse(BaseModel):
    """Response for experiment plan suggestion."""

    suggestion: ExperimentPlanSuggestion


# ============================================
# Agent Run/Step/Log Schemas (Multi-step Pipeline)
# ============================================

class AgentStepLogBase(BaseModel):
    """Base schema for agent step log."""

    message_type: str = Field(
        ...,
        description="Type of log message: thinking, hypothesis, action, summary, info, warning, error"
    )
    message: str = Field(
        ...,
        description="The log message content"
    )
    metadata_json: Optional[dict[str, Any]] = Field(
        None,
        description="Optional structured metadata"
    )


class AgentStepLogCreate(AgentStepLogBase):
    """Schema for creating a step log."""
    pass


class AgentStepLogRead(AgentStepLogBase):
    """Schema for reading a step log."""

    id: UUID
    agent_step_id: UUID
    sequence: int
    timestamp: datetime

    class Config:
        from_attributes = True


class AgentStepBase(BaseModel):
    """Base schema for agent step."""

    step_type: str = Field(
        ...,
        description="Type of step: problem_understanding, data_audit, dataset_design, experiment_design, plan_critic, execution, evaluation"
    )


class AgentStepCreate(AgentStepBase):
    """Schema for creating an agent step."""

    input_json: Optional[dict[str, Any]] = None


class AgentStepUpdate(BaseModel):
    """Schema for updating an agent step."""

    status: Optional[str] = None
    output_json: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None


class AgentStepRead(AgentStepBase):
    """Schema for reading an agent step."""

    id: UUID
    agent_run_id: UUID
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    input_json: Optional[dict[str, Any]] = None
    output_json: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentStepWithLogs(AgentStepRead):
    """Agent step with its logs."""

    logs: List[AgentStepLogRead] = []


class AgentRunBase(BaseModel):
    """Base schema for agent run."""

    name: Optional[str] = Field(
        None,
        description="Optional name for the run"
    )
    description: Optional[str] = Field(
        None,
        description="Optional description"
    )


class AgentRunCreate(AgentRunBase):
    """Schema for creating an agent run."""

    project_id: Optional[UUID] = None
    experiment_id: Optional[UUID] = None
    config_json: Optional[dict[str, Any]] = None


class AgentRunUpdate(BaseModel):
    """Schema for updating an agent run."""

    status: Optional[str] = None
    result_json: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None


class AgentRunRead(AgentRunBase):
    """Schema for reading an agent run."""

    id: UUID
    project_id: Optional[UUID] = None
    experiment_id: Optional[UUID] = None
    status: str
    config_json: Optional[dict[str, Any]] = None
    result_json: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentRunWithSteps(AgentRunRead):
    """Agent run with all its steps."""

    steps: List[AgentStepRead] = []


class AgentRunWithStepsAndLogs(AgentRunRead):
    """Agent run with all its steps and their logs."""

    steps: List[AgentStepWithLogs] = []


class AgentRunList(BaseModel):
    """Paginated list of agent runs."""

    items: List[AgentRunRead]
    total: int
    skip: int
    limit: int


class AgentStepLogList(BaseModel):
    """List of step logs for streaming."""

    logs: List[AgentStepLogRead]
    last_sequence: int = Field(
        ...,
        description="The highest sequence number in this batch. Use for since_sequence in next poll."
    )
    has_more: bool = Field(
        False,
        description="Whether there might be more logs (step still running)"
    )


# ============================================
# Training Dataset Spec Schemas (Phase 12.3)
# ============================================

class BaseFilter(BaseModel):
    """A filter condition for the base table."""

    column: str = Field(..., description="Column to filter on")
    operator: str = Field(
        ...,
        description="Filter operator: >=, <=, ==, !=, >, <, in, not_in, is_null, is_not_null"
    )
    value: Any = Field(None, description="Value to compare against (None for is_null/is_not_null)")


class TargetDefinition(BaseModel):
    """Definition of the prediction target."""

    table: str = Field(..., description="Table containing the target column")
    column: str = Field(..., description="Column to predict")
    join_key: Optional[str] = Field(
        None,
        description="Key to join target table to base table (if different from base)"
    )
    label_window_days: Optional[int] = Field(
        None,
        description="For time-based targets: how many days forward to look for the label"
    )


class AggregationFeature(BaseModel):
    """A single aggregated feature definition."""

    name: str = Field(..., description="Name of the resulting feature column")
    agg: str = Field(
        ...,
        description="Aggregation function: sum, count, avg, min, max, std, first, last"
    )
    column: str = Field(
        ...,
        description="Column to aggregate (* for count)"
    )


class JoinAggregation(BaseModel):
    """Aggregation configuration for a one-to-many join."""

    window_days: Optional[int] = Field(
        None,
        description="Time window in days for aggregation (None for all history)"
    )
    features: List[AggregationFeature] = Field(
        default_factory=list,
        description="List of features to create from this join"
    )


class JoinPlanItem(BaseModel):
    """A single join in the training dataset plan."""

    from_table: str = Field(..., description="Source table (typically base table)")
    to_table: str = Field(..., description="Table to join")
    left_key: str = Field(..., description="Key column in from_table")
    right_key: str = Field(..., description="Key column in to_table")
    relationship: str = Field(
        ...,
        description="Relationship type: one_to_one, one_to_many, many_to_one"
    )
    aggregation: Optional[JoinAggregation] = Field(
        None,
        description="Aggregation config for one_to_many joins"
    )


class TrainingDatasetSpec(BaseModel):
    """Complete specification for building a training dataset.

    This defines:
    - Base table (one row per prediction unit)
    - Target column definition
    - Join plan for bringing in features from related tables
    - Filters and exclusions
    """

    base_table: str = Field(
        ...,
        description="Name of the base table (one row per prediction unit)"
    )
    base_filters: List[BaseFilter] = Field(
        default_factory=list,
        description="Filters to apply to the base table"
    )
    target_definition: TargetDefinition = Field(
        ...,
        description="Definition of the prediction target"
    )
    join_plan: List[JoinPlanItem] = Field(
        default_factory=list,
        description="List of joins to bring in features from related tables"
    )
    excluded_tables: List[str] = Field(
        default_factory=list,
        description="Tables to exclude from the training dataset"
    )
    excluded_columns: List[str] = Field(
        default_factory=list,
        description="Columns to exclude from features"
    )


class TrainingDatasetPlanningInput(BaseModel):
    """Input for the training dataset planning step."""

    project_description: str = Field(
        ...,
        description="User's description of the ML goal"
    )
    target_hint: Optional[str] = Field(
        None,
        description="Optional hint about which column is the target"
    )
    data_source_profiles: List[dict[str, Any]] = Field(
        ...,
        description="List of data source profile summaries"
    )
    relationships_summary: dict[str, Any] = Field(
        ...,
        description="Output from relationship discovery service"
    )


class TrainingDatasetPlanningOutput(BaseModel):
    """Output from the training dataset planning step."""

    training_dataset_spec: TrainingDatasetSpec = Field(
        ...,
        description="The generated training dataset specification"
    )
    natural_language_summary: str = Field(
        ...,
        description="Human-readable explanation of the plan"
    )


# ============================================
# Training Critique Schemas
# ============================================

class TrainingIssue(BaseModel):
    """A single issue identified during training."""

    issue: str = Field(..., description="Description of the issue")
    severity: str = Field(
        ...,
        description="Issue severity: info, warning, or critical"
    )
    models_affected: List[str] = Field(
        default_factory=list,
        description="List of model names affected by this issue"
    )


class FeatureInsights(BaseModel):
    """Insights about feature quality and importance."""

    top_features: List[str] = Field(
        default_factory=list,
        description="Most important features for prediction"
    )
    suspected_leakage: List[str] = Field(
        default_factory=list,
        description="Features that might leak target information"
    )
    low_value_features: List[str] = Field(
        default_factory=list,
        description="Features that could be dropped"
    )


class ImprovementSuggestion(BaseModel):
    """A specific actionable suggestion for improvement."""

    priority: int = Field(..., description="Priority rank (1 = highest)")
    category: str = Field(
        ...,
        description="Category: feature_engineering, data_quality, target_modification, more_data, config_change"
    )
    suggestion: str = Field(..., description="Specific actionable suggestion")
    expected_impact: str = Field(
        ...,
        description="Expected impact: low, medium, or high"
    )
    implementation: str = Field(
        ...,
        description="How to implement this change"
    )


class NextExperimentConfig(BaseModel):
    """Suggested configuration for the next experiment."""

    time_limit: int = Field(300, description="Suggested time limit in seconds")
    presets: str = Field("best_quality", description="Suggested AutoGluon preset")
    suggested_features_to_add: List[str] = Field(
        default_factory=list,
        description="Features to add in next experiment"
    )
    suggested_features_to_remove: List[str] = Field(
        default_factory=list,
        description="Features to remove in next experiment"
    )


class TrainingCritiqueResult(BaseModel):
    """Complete training critique result from AI analysis."""

    performance_rating: str = Field(
        ...,
        description="Overall rating: poor, fair, good, or excellent"
    )
    performance_analysis: str = Field(
        ...,
        description="Brief assessment of the score and what it means"
    )
    training_issues: List[TrainingIssue] = Field(
        default_factory=list,
        description="Issues identified during training"
    )
    feature_insights: FeatureInsights = Field(
        ...,
        description="Insights about feature quality"
    )
    improvement_suggestions: List[ImprovementSuggestion] = Field(
        default_factory=list,
        description="Ranked list of improvement suggestions"
    )
    next_experiment_config: NextExperimentConfig = Field(
        ...,
        description="Suggested config for next experiment"
    )
    summary: str = Field(
        ...,
        description="2-3 sentence summary of key findings"
    )

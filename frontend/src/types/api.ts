/**
 * API Types for AgentML Frontend
 * These types mirror the backend Pydantic schemas
 */

// Enums
export type TaskType =
  | 'binary'
  | 'multiclass'
  | 'regression'
  | 'quantile'
  | 'timeseries_forecast'
  | 'multimodal_classification'
  | 'multimodal_regression'
  | 'classification'; // legacy

export type ProjectStatus = 'draft' | 'active' | 'archived';

export type DataSourceType = 'file_upload' | 'database' | 'api' | 's3' | 'external_dataset';

export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type TrialStatus = 'pending' | 'running' | 'completed' | 'failed';

export type ModelStatus = 'candidate' | 'shadow' | 'production' | 'retired';

export type MetricDirection = 'maximize' | 'minimize';

// Execution Mode for Auto DS Sessions
export type ExecutionMode = 'legacy' | 'adaptive' | 'phased' | 'dynamic';

// Validation Strategy for experiments
export type ValidationStrategy = 'standard' | 'robust' | 'strict';

// Auto DS Config Types
export interface AutoDSConfig {
  max_iterations?: number;
  accuracy_threshold?: number | null;
  time_budget_minutes?: number | null;
  parallel_experiments?: number;
  start_on_pipeline_complete?: boolean;
  max_experiments_per_dataset?: number;
  max_active_datasets?: number;
  // Execution mode settings
  execution_mode?: ExecutionMode;
  adaptive_decline_threshold?: number;
  phased_min_baseline_improvement?: number;
  dynamic_experiments_per_cycle?: number;
  // Validation strategy settings (Tier 2)
  validation_strategy?: ValidationStrategy;
  validation_num_seeds?: number;
  validation_cv_folds?: number;
  // Tier 1 feature flags
  enable_feature_engineering?: boolean;
  enable_ensemble?: boolean;
  enable_ablation?: boolean;
  // Tier 2 feature flags
  enable_diverse_configs?: boolean;
  // Context document settings
  use_context_documents?: boolean;
  context_ab_testing?: boolean;
}

// Project Types
export interface Project {
  id: string;
  name: string;
  description: string | null;
  task_type: TaskType | null;
  status: ProjectStatus;
  // Large dataset safeguard settings
  max_training_rows: number;
  profiling_sample_rows: number;
  max_aggregation_window_days: number;
  // Auto DS settings
  auto_ds_enabled: boolean;
  auto_ds_config_json: AutoDSConfig | null;
  active_auto_ds_session_id: string | null;
  // Context document settings
  context_ab_testing_enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface ProjectCreate {
  name: string;
  description?: string | null;
  task_type?: TaskType | null;
}

export interface ProjectUpdate {
  name?: string;
  description?: string | null;
  task_type?: TaskType | null;
  status?: ProjectStatus;
  // Large dataset safeguard settings
  max_training_rows?: number;
  profiling_sample_rows?: number;
  max_aggregation_window_days?: number;
  // Auto DS settings
  auto_ds_enabled?: boolean;
  auto_ds_config_json?: AutoDSConfig | null;
  // Context document settings
  context_ab_testing_enabled?: boolean;
}

export interface ProjectListResponse {
  items: Project[];
  total: number;
}

// Data Source Types
export interface DataSource {
  id: string;
  project_id: string;
  name: string;
  type: DataSourceType;
  config_json: Record<string, unknown> | null;
  schema_summary: SchemaSummary | null;
  created_at: string;
  updated_at: string;
}

export interface SchemaSummary {
  columns: ColumnInfo[];
  row_count: number;
  file_type?: string;
  analyzed_sheet?: string;
  sheet_names?: string[];
  warnings?: string[];
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  non_null_count: number;
  null_count: number;
  unique_count: number;
  sample_values: unknown[];
  inferred_type?: string;
}

export interface DataSourceCreate {
  name: string;
  type: DataSourceType;
  config_json?: Record<string, unknown> | null;
}

// Dataset Spec Types
export interface AgentExperimentDesignConfig {
  step_id?: string | null;
  agent_run_id?: string | null;
  variants?: Array<{
    name: string;
    description?: string;
    automl_config?: Record<string, unknown>;
    validation_strategy?: Record<string, unknown> | string;
  }>;
  recommended_variant?: string;
  primary_metric?: string;
  natural_language_summary?: string;
  stored_at?: string;
  source_type?: 'initial' | 'iteration';  // "initial" = first run, "iteration" = auto-improve
  parent_experiment_id?: string | null;  // Set for iteration configs
  iteration_number?: number;  // For iteration configs
}

export interface DatasetSpec {
  id: string;
  project_id: string;
  name: string;
  description: string | null;
  data_sources_json: Record<string, unknown> | null;
  target_column: string | null;
  feature_columns: string[] | null;
  filters_json: Record<string, unknown> | null;
  spec_json: Record<string, unknown> | null;
  agent_experiment_design_json: AgentExperimentDesignConfig | null;
  // Time-based task metadata
  is_time_based: boolean;
  time_column: string | null;
  entity_id_column: string | null;
  prediction_horizon: string | null;
  target_positive_class: string | null;
  created_at: string;
  updated_at: string;
}

export interface DatasetSpecCreate {
  name: string;
  description?: string | null;
  data_sources_json?: Record<string, unknown> | null;
  target_column?: string | null;
  feature_columns?: string[] | null;
  filters_json?: Record<string, unknown> | null;
  spec_json?: Record<string, unknown> | null;
}

export interface DatasetSpecUpdate {
  name?: string;
  description?: string | null;
  data_sources_json?: Record<string, unknown> | null;
  target_column?: string | null;
  feature_columns?: string[] | null;
  filters_json?: Record<string, unknown> | null;
  spec_json?: Record<string, unknown> | null;
}

// Experiment Types
export interface Experiment {
  id: string;
  project_id: string;
  name: string;
  description: string | null;
  dataset_spec_id: string | null;
  primary_metric: string | null;
  metric_direction: MetricDirection | null;
  experiment_plan_json: ExperimentPlan | null;
  status: ExperimentStatus;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface ExperimentDetail extends Experiment {
  trial_count: number;
  best_model: BestModelInfo | null;
  best_metrics: Record<string, number> | null;
  // Auto-improve iteration fields
  iteration_number: number;
  parent_experiment_id: string | null;
  improvement_context_json: ImprovementContext | null;
  // Holdout-based scoring (Make Holdout Score the Real Score)
  final_score: number | null;  // Canonical score from holdout evaluation
  val_score: number | null;  // Validation/CV score (NOT the final score)
  train_score: number | null;  // Training score (for overfitting analysis)
  has_holdout: boolean;  // Whether holdout evaluation was performed
  holdout_samples: number | null;  // Number of samples in holdout set
  overfitting_gap: number | null;  // Gap between val and holdout (positive = overfitting)
  score_source: 'holdout' | 'validation';  // Source of final_score
}

export interface ImprovementContext {
  improvement_analysis?: {
    improvement_summary?: string;
    key_issues?: string[];
    improvement_opportunities?: string[];
    recommended_changes?: string[];
  };
  improvement_plan?: {
    plan_summary?: string;
    iteration_name?: string;
    iteration_description?: string;
    feature_changes?: {
      features_to_keep?: string[];
      features_to_remove?: string[];
      engineered_features?: Array<{
        name?: string;
        output_column?: string;
        reasoning?: string;
      }>;
    };
    expected_improvement?: string;
  };
  summary?: string;
}

export interface BestModelInfo {
  id: string;
  name: string;
  model_type: string;
  status: ModelStatus;
}

export interface ValidationStrategy {
  split_strategy: string;
  validation_split?: number;
  time_column?: string | null;
  entity_id_column?: string | null;
  group_column?: string | null;  // Legacy alias for entity_id_column
  n_folds?: number | null;
  reasoning?: string;
}

export interface ExperimentPlan {
  automl_config?: AutoMLConfig;
  validation_strategy?: ValidationStrategy;
}

export interface AutoMLConfig {
  time_limit?: number;
  presets?: string;
  num_bag_folds?: number;
  num_stack_levels?: number;
  excluded_model_types?: string[];
  quantile_levels?: number[];
  prediction_length?: number;
  time_column?: string;
  id_column?: string;
  freq?: string;
}

export interface ExperimentCreate {
  name: string;
  description?: string | null;
  dataset_spec_id?: string | null;
  primary_metric?: string | null;
  metric_direction?: MetricDirection | null;
  experiment_plan_json?: ExperimentPlan | null;
}

export interface ExperimentUpdate {
  name?: string;
  description?: string | null;
  dataset_spec_id?: string | null;
  primary_metric?: string | null;
  metric_direction?: MetricDirection | null;
  experiment_plan_json?: ExperimentPlan | null;
  status?: ExperimentStatus;
}

// Training options
export type TrainingBackend = 'local' | 'modal';

export interface TrainingOptions {
  backend?: TrainingBackend;
  resource_limits_enabled?: boolean;
  num_cpus?: number | null;
  num_gpus?: number | null;
  memory_limit_gb?: number | null;
}

export interface ExperimentRunRequest {
  training_options?: TrainingOptions | null;
}

export interface ExperimentRunResponse {
  experiment_id: string;
  status: string;
  task_id: string | null;
  message: string | null;
  backend?: string | null;
}

export interface TrainingOptionsInfo {
  backends: {
    local: {
      available: boolean;
      description: string;
    };
    modal: {
      available: boolean;
      description: string;
      status: {
        installed: boolean;
        configured: boolean;
        enabled: boolean;
        token_set: boolean;
      };
    };
  };
  resource_limits: {
    enabled_by_default: boolean;
    defaults: {
      num_cpus: number;
      num_gpus: number;
      memory_limit_gb: number;
    };
    description: string;
  };
  automl_defaults: {
    time_limit: number;
    presets: string;
  };
}

export interface ExperimentProgress {
  experiment_id: string;
  status: string;  // pending, queued, running, completed, failed, cancelled
  progress: number;  // 0-100 percentage
  message: string | null;
  stage: string | null;  // Current stage (e.g., "loading_data", "training")
}

// Trial Types
export interface Trial {
  id: string;
  experiment_id: string;
  variant_name: string;
  data_split_strategy: string | null;
  automl_config: Record<string, unknown> | null;
  status: TrialStatus;
  metrics_json: Record<string, number> | null;
  baseline_metrics_json: BaselineMetrics | null;
  best_model_ref: string | null;
  logs_location: string | null;
  created_at: string;
  updated_at: string;
}

// Baseline metrics for sanity checking
export interface BaselineMetrics {
  // For classification
  majority_class?: {
    accuracy: number;
    roc_auc?: number;
  };
  // For regression
  mean_predictor?: {
    rmse: number;
    mae: number;
    r2: number;
  };
  // Simple model baseline
  simple_logistic?: {
    accuracy: number;
    roc_auc?: number;
    f1?: number;
  };
  simple_ridge?: {
    rmse: number;
    mae: number;
    r2: number;
  };
  // Label-shuffle sanity test
  label_shuffle?: {
    shuffled_accuracy?: number;
    shuffled_roc_auc?: number;
    shuffled_r2?: number;
    shuffled_rmse?: number;
    expected_random_accuracy?: number;
    leakage_detected?: boolean | null;
    warning?: string;
    error?: string;
  };
}

// Robustness audit results (Prompt 4 + Prompt 5 additions)
export interface RobustnessAudit {
  overfitting_risk: 'low' | 'medium' | 'high' | 'unknown';
  leakage_suspected: boolean;
  time_split_suspicious: boolean;
  metrics_summary: {
    best_val_metric?: number | null;
    primary_metric?: string | null;
    train_val_gap_worst?: number | null;
    train_val_gap_avg?: number | null;
    cv_variance?: number | null;
    baseline_value?: number | null;
    baseline_type?: string | null;
  };
  warnings: string[];
  recommendations: string[];
  natural_language_summary: string;
  // Prompt 5 additions
  too_good_to_be_true?: boolean;
  risk_adjusted_score?: number | null;
  risk_level?: 'low' | 'medium' | 'high' | 'critical' | 'unknown';
  requires_override?: boolean;
  risk_reason?: string;
  // Additional detailed analysis
  train_val_analysis?: Record<string, unknown>;
  suspicious_patterns?: Array<{
    type?: string;
    severity?: string;
    description?: string;
  }>;
  baseline_comparison?: Record<string, unknown>;
  baseline_metrics?: BaselineMetrics;
  cv_analysis?: Record<string, unknown>;
  is_time_based?: boolean;
  task_type?: string;
}

// Model Risk Status (Prompt 5)
export interface ModelRiskStatus {
  model_id: string;
  model_name: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical' | 'unknown';
  requires_override: boolean;
  risk_reason: string;
  overfitting_risk: string;
  leakage_suspected: boolean;
  time_split_suspicious: boolean;
  too_good_to_be_true: boolean;
  risk_adjusted_score: number | null;
}

export interface TrialCreate {
  variant_name: string;
  data_split_strategy?: string | null;
  automl_config?: Record<string, unknown> | null;
}

// Model Version Types
export interface ModelVersion {
  id: string;
  project_id: string;
  experiment_id: string | null;
  trial_id: string | null;
  name: string;
  model_type: string | null;
  artifact_location: string | null;
  metrics_json: Record<string, unknown> | null;
  feature_importances_json: Record<string, number> | null;
  serving_config_json: Record<string, unknown> | null;
  status: ModelStatus;
  created_at: string;
  updated_at: string;
}

export interface ModelVersionCreate {
  name: string;
  model_type?: string | null;
  artifact_location?: string | null;
  metrics_json?: Record<string, unknown> | null;
  feature_importances_json?: Record<string, number> | null;
  serving_config_json?: Record<string, unknown> | null;
  experiment_id?: string | null;
  trial_id?: string | null;
}

export interface ModelPromoteRequest {
  status: 'candidate' | 'shadow' | 'production';
}

export interface PredictionRequest {
  features: Record<string, unknown>;
}

export interface PredictionResponse {
  prediction: unknown;
  probabilities: Record<string, number> | null;
  model_id: string;
  model_name: string;
}

export interface ModelExplainRequest {
  question: string;
}

export interface ModelExplainResponse {
  answer: string;
  model_id: string;
}

// Raw Prediction Types
export interface RawPredictionRequest {
  data: Record<string, unknown>;
}

export interface RawPredictionResponse {
  prediction: unknown;
  probabilities: Record<string, number> | null;
  model_id: string;
  model_name: string;
  transformations_applied: boolean;
  transformed_features: Record<string, unknown> | null;
}

export interface BatchPredictionRequest {
  data: Record<string, unknown>[];
}

export interface BatchPredictionResponse {
  predictions: unknown[];
  probabilities: Record<string, number>[] | null;
  model_id: string;
  model_name: string;
  count: number;
}

// Feature Pipeline Types
export interface FeaturePipelineInfo {
  model_id: string;
  has_transformations: boolean;
  transformation_count: number;
  required_input_columns: string[];
  output_columns: string[];
  target_column: string | null;
  pipeline_config: Record<string, unknown> | null;
}

// Model Export Types
export interface ModelExportInfo {
  model_id: string;
  model_name: string;
  model_type: string;
  task_type: string;
  can_export: boolean;
  export_size_mb: number | null;
  has_pipeline: boolean;
  required_packages: string[];
}

export interface ServingFeature {
  name: string;
  type: 'numeric' | 'categorical' | 'boolean' | 'datetime';
}

// API Error - detail can be a string or an object with message/feedback
export interface ApiError {
  detail: string | { message?: string; feedback?: string; [key: string]: unknown };
}

// Health Check
export interface HealthStatus {
  status: string;
}

export interface ApiInfo {
  name: string;
  version: string;
  docs: string;
}

// API Key Types
export type LLMProvider = 'openai' | 'gemini';

export interface ApiKey {
  id: string;
  provider: LLMProvider;
  name: string | null;
  key_preview: string;
  created_at: string;
  updated_at: string;
}

export interface ApiKeyCreate {
  provider: LLMProvider;
  api_key: string;
  name?: string | null;
}

export interface ApiKeyStatus {
  openai: boolean;
  gemini: boolean;
}

// AI Model Selection Types
export type AIModel = 'gpt-5.1-thinking' | 'gpt-5.1' | 'gpt-4.1';

export interface AIModelOption {
  value: AIModel;
  display_name: string;
  description: string;
}

export interface AppSettings {
  ai_model: AIModel;
  ai_model_display_name: string;
  updated_at: string;
}

export interface AppSettingsUpdate {
  ai_model?: AIModel;
}

// User and Auth Types
export interface User {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_verified: boolean;
}

export interface UserCreate {
  email: string;
  password: string;
  full_name?: string | null;
}

export interface UserLogin {
  email: string;
  password: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
}

export interface GoogleAuthRequest {
  credential: string;
}

// Sharing Types
export type ShareRole = 'viewer' | 'editor' | 'admin';
export type InviteStatus = 'pending' | 'accepted' | 'declined' | 'expired';

export interface ShareRequest {
  email: string;
  role: ShareRole;
}

export interface Share {
  id: string;
  user_id: string | null;
  invited_email: string | null;
  role: ShareRole;
  status: InviteStatus;
  created_at: string;
  user_name: string | null;
}

export interface ShareListResponse {
  items: Share[];
  total: number;
}

export interface MyShares {
  shared_projects: SharedResource[];
  pending_invites: PendingInvite[];
}

export interface SharedResource {
  id: string;
  name: string;
  role: string;
  owner: string | null;
  shared_at: string;
}

export interface PendingInvite {
  type: 'project';
  id: string;
  name: string;
  role: string;
  token: string;
}

// Agent Pipeline Types
export type AgentStepType =
  | 'data_analysis'
  | 'problem_understanding'
  | 'data_audit'
  | 'dataset_design'
  | 'dataset_validation'
  | 'experiment_design'
  | 'plan_critic'
  | 'results_interpretation'
  | 'results_critic'
  | 'dataset_discovery'
  // Training dataset steps
  | 'training_dataset_planning'
  | 'training_dataset_build'
  // Data Architect pipeline steps
  | 'dataset_inventory'
  | 'relationship_discovery'
  // Auto-improve pipeline steps (simple)
  | 'improvement_analysis'
  | 'improvement_plan'
  // Enhanced improvement pipeline steps (full agent)
  | 'iteration_context'
  | 'improvement_data_analysis'
  | 'improvement_dataset_design'
  | 'improvement_experiment_design'
  // Lab notebook summary step
  | 'lab_notebook_summary'
  // Robustness audit step
  | 'robustness_audit'
  // Orchestration system steps
  | 'project_manager'
  | 'gemini_critique'
  | 'openai_judge'
  | 'debate_round';

export type AgentStepStatus = 'pending' | 'running' | 'completed' | 'failed';

export type AgentRunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type LogMessageType = 'info' | 'warning' | 'error' | 'thought' | 'summary' | 'thinking' | 'hypothesis' | 'action';

export interface AgentStepLog {
  id: string;
  agent_step_id: string;
  sequence: number;
  timestamp: string;
  message_type: LogMessageType;
  message: string;
  metadata_json?: Record<string, unknown>;
}

export interface AgentStep {
  id: string;
  agent_run_id: string;
  step_type: AgentStepType;
  status: AgentStepStatus;
  input_json?: Record<string, unknown>;
  output_json?: Record<string, unknown>;
  error_message?: string;
  retry_count: number;
  started_at?: string;
  finished_at?: string;
  created_at: string;
  updated_at: string;
  logs?: AgentStepLog[];
}

export interface AgentRun {
  id: string;
  project_id: string;
  name?: string;
  description?: string;
  status: AgentRunStatus;
  config_json?: Record<string, unknown>;
  result_json?: Record<string, unknown>;
  error_message?: string;
  created_at: string;
  updated_at: string;
  steps?: AgentStep[];
}

export interface AgentRunList {
  items: AgentRun[];
  total: number;
  skip: number;
  limit: number;
}

export interface AgentStepLogList {
  logs: AgentStepLog[];
  last_sequence: number;
  has_more: boolean;
}

export interface SetupPipelineRequest {
  data_source_id: string;
  description: string;
  time_budget_minutes?: number;
  run_async?: boolean;
  // Orchestration options
  orchestration_mode?: 'sequential' | 'project_manager';
  debate_mode?: 'disabled' | 'enabled';
  judge_model?: string;
  max_debate_rounds?: number;  // Max rounds before calling judge (default: 3, min: 1, max: 10)
  debate_partner?: string;  // Model name of the debate partner (default: gemini-2.0-flash)
  // Holdout validation options
  holdout_enabled?: boolean;  // If true, hold out data for user validation (default: false)
  holdout_percentage?: number;  // Percentage of data to hold out (default: 5%, range: 1-20%)
  // Context documents options
  use_context_documents?: boolean;  // If true (default), use uploaded context documents in AI prompts
  context_ab_testing?: boolean;  // If true, create experiments both with and without context for A/B comparison
}

// Holdout validation types
export interface HoldoutSet {
  id: string;
  project_id: string;
  data_source_id: string;
  holdout_percentage: number;
  total_rows_original: number;
  holdout_row_count: number;
  training_row_count: number;
  target_column: string | null;
  feature_columns: string[] | null;
  created_at: string;
}

export interface HoldoutRow {
  index: number;
  total_rows: number;
  data: Record<string, unknown>;
  target_column: string | null;
  target_value: unknown;
}

export interface HoldoutRowsResponse {
  holdout_set_id: string;
  total_rows: number;
  rows: Record<string, unknown>[];
  target_column: string | null;
}

// Orchestration types
export type OrchestrationMode = 'sequential' | 'project_manager';
export type DebateMode = 'disabled' | 'enabled';

export interface DebatePartnerOption {
  model: string;
  display_name: string;
  provider: string;
  description: string;
}

export interface OrchestrationOptions {
  orchestration_modes: OrchestrationMode[];
  debate_modes: DebateMode[];
  judge_models: string[];
  default_judge_model: string;
  default_max_debate_rounds: number;  // Default max debate rounds (default: 3)
  debate_partners: DebatePartnerOption[];  // Available debate partner LLMs
  default_debate_partner: string;  // Default debate partner model name
}

export interface SetupPipelineResponse {
  run_id: string;
  status: string;
  message: string;
}

// Dataset Discovery Types
export interface DatasetDiscoveryRequest {
  project_description: string;
  constraints?: {
    geography?: string;
    allow_public_data?: boolean;
    licensing_requirements?: string[];
  };
  run_async?: boolean;
}

export interface DatasetDiscoveryResponse {
  run_id: string;
  status: string;
  message: string;
}

export interface DiscoveredDataset {
  name: string;
  source_url: string;
  schema_summary?: {
    rows_estimate?: number;
    columns?: string[];
    target_candidate?: string;
  };
  licensing?: string;
  fit_for_purpose: string;
}

export interface ApplyDiscoveredDatasetsRequest {
  dataset_indices: number[];
}

export interface ApplyDiscoveredDatasetsResponse {
  data_sources: {
    data_source_id: string;
    name: string;
    source_url: string;
    licensing: string;
    has_schema_estimate: boolean;
  }[];
  message: string;
}

// Validation Sample Types
export interface ValidationSample {
  id: string;
  model_version_id: string;
  row_index: number;
  features: Record<string, unknown>;
  target_value: string;
  predicted_value: string;
  error_value: number | null;
  absolute_error: number | null;
  prediction_probabilities: Record<string, number> | null;
}

export interface ValidationSamplesListResponse {
  model_id: string;
  total: number;
  limit: number;
  offset: number;
  samples: ValidationSample[];
}

export type ValidationSampleSort = 'error_desc' | 'error_asc' | 'row_index' | 'random';

export interface WhatIfRequest {
  sample_id: string;
  modified_features: Record<string, unknown>;
}

export interface WhatIfResponse {
  original_sample: ValidationSample;
  modified_features: Record<string, unknown>;
  original_prediction: unknown;
  modified_prediction: unknown;
  prediction_delta: number | null;
  original_probabilities: Record<string, number> | null;
  modified_probabilities: Record<string, number> | null;
}

// Model Testing Data Types
export interface FeatureStatistics {
  name: string;
  type: string;  // "numeric", "categorical", "boolean"
  importance: number | null;
  // Numeric stats
  min_value: number | null;
  max_value: number | null;
  mean_value: number | null;
  median_value: number | null;
  // Categorical stats
  categories: string[] | null;
  most_common: string | null;
}

export interface ModelTestingDataResponse {
  model_id: string;
  model_name: string;
  task_type: string;
  target_column: string | null;
  features: FeatureStatistics[];
  top_features: string[];
  sample_data: Record<string, unknown> | null;
  has_validation_samples: boolean;
  validation_sample_count: number;
}

export interface RandomSampleResponse {
  features: Record<string, unknown>;
  sample_id: string;
  actual_value: string;
  predicted_value: string;
}

// Data Architect Pipeline Types
export interface DataArchitectRequest {
  target_hint?: string;
  run_async?: boolean;
  auto_run_setup?: boolean;
}

export interface DataArchitectResponse {
  agent_run_id: string;
  status: string;
  message: string;
  setup_run_id?: string;
}

export interface TrainingDatasetSummary {
  data_source_id: string;
  data_source_name: string;
  row_count: number;
  column_count: number;
  base_table: string;
  joined_tables: string[];
  target_column?: string;
  feature_columns: string[];
}

// Overfitting Analysis Types
export type OverfittingRiskLevel = 'low' | 'medium' | 'high';
export type OverfittingTrend = 'improving' | 'stable' | 'degrading' | 'unknown';
export type OverfittingRecommendation = 'continue' | 'warning' | 'stop';
export type OverfittingIterationStatus = 'healthy' | 'warning' | 'high_risk' | 'best' | 'unknown';

export interface OverfittingIterationEntry {
  experiment_id: string;
  iteration: number;
  holdout_score: number;
  metric: string;
  overfitting_risk: number;
  status: OverfittingIterationStatus;
  is_best: boolean;
}

export interface OverfittingReport {
  experiment_id: string;
  iteration_number: number;
  total_iterations: number;
  overall_risk: number;
  risk_level: OverfittingRiskLevel;
  trend: OverfittingTrend;
  best_iteration: number;
  best_score: number;
  current_score: number;
  recommendation: OverfittingRecommendation;
  message: string;
  iterations: OverfittingIterationEntry[];
}

// ============================================
// Research Cycle and Lab Notebook Types
// ============================================

export type ResearchCycleStatus = 'pending' | 'running' | 'completed' | 'failed';
export type LabNotebookAuthorType = 'agent' | 'human';

export interface ResearchCycleSummary {
  id: string;
  sequence_number: number;
  status: ResearchCycleStatus;
  summary_title: string | null;
  created_at: string;
  updated_at: string;
  experiment_count: number;
}

export interface ResearchCycleListResponse {
  cycles: ResearchCycleSummary[];
  total: number;
}

export interface ExperimentSummaryForCycle {
  id: string;
  name: string;
  status: string;
  best_metric: number | null;
  primary_metric: string | null;
  created_at: string;
}

export interface LabNotebookEntrySummary {
  id: string;
  title: string;
  author_type: LabNotebookAuthorType;
  created_at: string;
}

export interface ResearchCycleResponse {
  id: string;
  project_id: string;
  sequence_number: number;
  status: ResearchCycleStatus;
  summary_title: string | null;
  created_at: string;
  updated_at: string;
  experiments: ExperimentSummaryForCycle[];
  lab_notebook_entries: LabNotebookEntrySummary[];
}

export interface ResearchCycleCreate {
  summary_title?: string;
}

export interface ResearchCycleUpdate {
  summary_title?: string;
  status?: ResearchCycleStatus;
}

export interface CycleExperimentResponse {
  id: string;
  research_cycle_id: string;
  experiment_id: string;
  linked_at: string;
}

export interface LabNotebookEntry {
  id: string;
  project_id: string;
  research_cycle_id: string | null;
  agent_step_id: string | null;
  author_type: LabNotebookAuthorType;
  title: string;
  body_markdown: string | null;
  created_at: string;
  updated_at: string;
}

export interface LabNotebookEntryCreate {
  title: string;
  body_markdown?: string;
  research_cycle_id?: string;
  author_type?: LabNotebookAuthorType;
}

export interface LabNotebookEntryUpdate {
  title?: string;
  body_markdown?: string;
}

export interface LabNotebookEntryListResponse {
  entries: LabNotebookEntry[];
  total: number;
}


// ============================================
// Project History Types
// ============================================

export interface ToolCallDetail {
  name: string;
  arguments: Record<string, unknown>;
  result_preview: string | null;
  timestamp: string | null;
}

export interface AgentThinkingDetail {
  step_id: string;
  step_type: string;
  step_name: string | null;
  status: string;
  tool_calls: ToolCallDetail[];
  thinking_log: string[];
  observation_log: string[];
  action_log: string[];
  summary: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface AgentRunDetail {
  id: string;
  run_type: string | null;
  status: string;
  created_at: string;
  completed_at: string | null;
  total_steps: number;
  completed_steps: number;
  steps: AgentThinkingDetail[];
}

export interface ExperimentResultDetail {
  id: string;
  name: string;
  status: string;
  primary_metric: string | null;
  best_score: number | null;
  trial_count: number;
  best_trial_name: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface ResearchCycleDetail {
  id: string;
  sequence_number: number;
  status: string;
  title: string | null;
  created_at: string;
  updated_at: string | null;
  experiments: ExperimentResultDetail[];
  agent_runs: AgentRunDetail[];
}

export interface NotebookEntryDetail {
  id: string;
  title: string;
  body_markdown: string | null;
  author_type: string;
  research_cycle_id: string | null;
  agent_step_id: string | null;
  created_at: string;
}

export interface BestModelDetail {
  experiment_id: string;
  experiment_name: string;
  trial_id: string;
  trial_name: string;
  metric_name: string;
  metric_value: number;
  model_path: string | null;
  research_cycle_id: string | null;
}

export interface ProjectHistoryResponse {
  project_id: string;
  project_name: string;
  total_cycles: number;
  total_experiments: number;
  total_notebook_entries: number;
  research_cycles: ResearchCycleDetail[];
  notebook_entries: NotebookEntryDetail[];
  best_models: BestModelDetail[];
  last_experiment_at: string | null;
  last_agent_run_at: string | null;
}

// Auto DS Types
export type AutoDSSessionStatus = 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';
export type AutoDSIterationStatus = 'pending' | 'running_experiments' | 'analyzing' | 'planning' | 'completed' | 'failed';
export type InsightType = 'feature_importance' | 'model_performance' | 'preprocessing' | 'overfitting_pattern' | 'hyperparameter' | 'interaction' | 'other';
export type InsightConfidence = 'high' | 'medium' | 'low';

export interface AutoDSSession {
  id: string;
  project_id: string;
  name: string;
  description: string | null;
  status: AutoDSSessionStatus;
  max_iterations: number;
  accuracy_threshold: number | null;
  time_budget_minutes: number | null;
  min_improvement_threshold: number;
  plateau_iterations: number;
  max_experiments_per_dataset: number;
  max_active_datasets: number;
  // Execution mode settings
  execution_mode: ExecutionMode;
  adaptive_decline_threshold: number;
  phased_min_baseline_improvement: number;
  dynamic_experiments_per_cycle: number;
  // Validation strategy settings
  validation_strategy: ValidationStrategy;
  validation_num_seeds: number;
  validation_cv_folds: number;
  // Tier 1 feature flags
  enable_feature_engineering: boolean;
  enable_ensemble: boolean;
  enable_ablation: boolean;
  // Tier 2 feature flags
  enable_diverse_configs: boolean;
  current_iteration: number;
  best_score: number | null;
  best_train_score: number | null;
  best_val_score: number | null;
  best_holdout_score: number | null;
  best_experiment_id: string | null;
  total_experiments_run: number;
  iterations_without_improvement: number;
  started_at: string | null;
  completed_at: string | null;
  research_cycle_id: string | null;
  celery_task_id: string | null;
  config_json: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
  iterations?: AutoDSIteration[];
}

export interface AutoDSSessionCreate {
  name: string;
  description?: string | null;
  max_iterations?: number;
  accuracy_threshold?: number | null;
  time_budget_minutes?: number | null;
  min_improvement_threshold?: number;
  plateau_iterations?: number;
  max_experiments_per_dataset?: number;
  max_active_datasets?: number;
  // Execution mode settings
  execution_mode?: ExecutionMode;
  adaptive_decline_threshold?: number;
  phased_min_baseline_improvement?: number;
  dynamic_experiments_per_cycle?: number;
  // Validation strategy settings
  validation_strategy?: ValidationStrategy;
  validation_num_seeds?: number;
  validation_cv_folds?: number;
  // Tier 1 feature flags
  enable_feature_engineering?: boolean;
  enable_ensemble?: boolean;
  enable_ablation?: boolean;
  // Tier 2 feature flags
  enable_diverse_configs?: boolean;
  config_json?: Record<string, unknown> | null;
}

export interface AutoDSSessionSummary {
  id: string;
  project_id: string;
  name: string;
  description: string | null;
  status: AutoDSSessionStatus;
  execution_mode: ExecutionMode;
  current_iteration: number;
  max_iterations: number;
  best_score: number | null;
  total_experiments_run: number;
  created_at: string;
  updated_at: string;
}

export interface AutoDSSessionListResponse {
  sessions: AutoDSSessionSummary[];
  total: number;
}

export interface IterationExperimentInfo {
  experiment_id: string;
  experiment_name: string;
  experiment_status: string;
  dataset_spec_id: string | null;
  dataset_name: string | null;
  experiment_variant: number;
  hypothesis: string | null;
  score: number | null;
  train_score: number | null;
  val_score: number | null;
  holdout_score: number | null;
  rank_in_iteration: number | null;
  created_at: string | null;
}

export interface AutoDSIteration {
  id: string;
  session_id: string;
  iteration_number: number;
  status: AutoDSIterationStatus;
  experiments_planned: number;
  experiments_completed: number;
  experiments_failed: number;
  best_score_this_iteration: number | null;
  best_train_score_this_iteration: number | null;
  best_val_score_this_iteration: number | null;
  best_holdout_score_this_iteration: number | null;
  best_experiment_id_this_iteration: string | null;
  experiments_started_at: string | null;
  experiments_completed_at: string | null;
  analysis_started_at: string | null;
  analysis_completed_at: string | null;
  strategy_started_at: string | null;
  strategy_completed_at: string | null;
  analysis_summary_json: Record<string, unknown> | null;
  strategy_decisions_json: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  // Experiments in this iteration
  experiments?: IterationExperimentInfo[];
}

export interface AutoDSSessionProgress {
  session_id: string;
  status: AutoDSSessionStatus;
  current_iteration: number;
  max_iterations: number;
  best_score: number | null;
  total_experiments_run: number;
  iterations_without_improvement: number;
  // Current iteration details
  current_iteration_status: AutoDSIterationStatus | null;
  current_iteration_experiments_completed: number;
  current_iteration_experiments_planned: number;
  // Timing
  started_at: string | null;
  elapsed_minutes: number | null;
  time_budget_minutes: number | null;
  // Stopping condition status
  accuracy_threshold: number | null;
  threshold_reached: boolean;
  plateau_detected: boolean;
}

export interface ResearchInsight {
  id: string;
  session_id: string;
  project_id: string;
  iteration_id: string | null;
  insight_type: InsightType;
  confidence: InsightConfidence;
  title: string;
  description: string | null;
  insight_data_json: Record<string, unknown> | null;
  evidence_count: number;
  supporting_experiments: string[] | null;
  contradicting_experiments: string[] | null;
  is_tested: boolean;
  test_result: string | null;
  promoted_to_global: boolean;
  global_insight_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface ResearchInsightListResponse {
  insights: ResearchInsight[];
  total: number;
}

export interface GlobalInsight {
  id: string;
  insight_type: InsightType;
  category: string | null;
  title: string;
  description: string | null;
  technical_details_json: Record<string, unknown> | null;
  applicable_to: string[] | null;
  task_types: string[] | null;
  data_characteristics: Record<string, unknown> | null;
  evidence_count: number;
  contradiction_count: number;
  confidence_score: number;
  source_project_count: number;
  last_validated_at: string | null;
  times_applied: number;
  times_successful: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface GlobalInsightListResponse {
  insights: GlobalInsight[];
  total: number;
}

export interface AutoDSStartResponse {
  session_id: string;
  task_id: string;
  status: string;
  message: string;
}

// Context Document Types
export type ContextDocumentExtractionStatus = 'pending' | 'completed' | 'failed';

export interface ContextDocument {
  id: string;
  project_id: string;
  name: string;
  original_filename: string;
  file_type: string;
  file_size_bytes: number;
  explanation: string;
  extraction_status: ContextDocumentExtractionStatus;
  extraction_error: string | null;
  is_active: boolean;
  has_content: boolean;
  created_at: string;
  updated_at: string;
}

export interface ContextDocumentDetail extends ContextDocument {
  extracted_text: string | null;
  content_preview: string | null;
}

export interface ContextDocumentCreate {
  name: string;
  explanation: string;
}

export interface ContextDocumentUpdate {
  name?: string;
  explanation?: string;
  is_active?: boolean;
}

export interface ContextDocumentListResponse {
  documents: ContextDocument[];
  total: number;
  total_active: number;
}

export interface SupportedExtensionsResponse {
  extensions: Record<string, string>;
  max_file_size_mb: number | null;  // null means no limit
}

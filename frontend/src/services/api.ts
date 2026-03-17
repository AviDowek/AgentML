/**
 * API Service for AgentML Frontend
 * Centralized API client with error handling
 */

import type {
  Project,
  ProjectCreate,
  ProjectUpdate,
  ProjectListResponse,
  DataSource,
  DataSourceCreate,
  DatasetSpec,
  DatasetSpecCreate,
  DatasetSpecUpdate,
  Experiment,
  ExperimentDetail,
  ExperimentCreate,
  ExperimentUpdate,
  ExperimentRunResponse,
  ExperimentProgress,
  TrainingOptionsInfo,
  Trial,
  TrialCreate,
  ModelVersion,
  ModelVersionCreate,
  ModelPromoteRequest,
  PredictionResponse,
  ModelExplainResponse,
  HealthStatus,
  ApiInfo,
  ApiError,
  ApiKey,
  ApiKeyCreate,
  ApiKeyStatus,
  LLMProvider,
  User,
  UserCreate,
  UserLogin,
  AuthToken,
  GoogleAuthRequest,
  Share,
  ShareRequest,
  ShareListResponse,
  MyShares,
  ModelTestingDataResponse,
  RandomSampleResponse,
  ProjectHistoryResponse,
  AgentThinkingDetail,
  ContextDocument,
  ContextDocumentDetail,
  ContextDocumentUpdate,
  ContextDocumentListResponse,
  SupportedExtensionsResponse,
} from '../types/api';

// API Base URL - configurable via environment
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

// Auth token storage key
const AUTH_TOKEN_KEY = 'agentML_auth_token';

/**
 * Get stored auth token
 */
export function getAuthToken(): string | null {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

/**
 * Set auth token
 */
export function setAuthToken(token: string): void {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
}

/**
 * Clear auth token
 */
export function clearAuthToken(): void {
  localStorage.removeItem(AUTH_TOKEN_KEY);
}

/**
 * Custom error class for API errors
 */
export class ApiException extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiException';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Generic fetch wrapper with error handling and auth
 */
async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {},
  requireAuth = false
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
  };

  // Add auth token if available
  const token = getAuthToken();
  if (token) {
    defaultHeaders['Authorization'] = `Bearer ${token}`;
  } else if (requireAuth) {
    throw new ApiException(401, 'Authentication required');
  }

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  });

  // Handle no-content responses
  if (response.status === 204) {
    return undefined as T;
  }

  // Parse JSON response
  const data = await response.json();

  // Handle errors
  if (!response.ok) {
    const error = data as ApiError;
    // Handle both string and object error details
    let detailMessage = 'Unknown error';
    if (typeof error.detail === 'string') {
      detailMessage = error.detail;
    } else if (error.detail && typeof error.detail === 'object') {
      // Extract message from object detail (used by validation errors)
      const detailObj = error.detail as { message?: string; feedback?: string };
      detailMessage = detailObj.message || detailObj.feedback || JSON.stringify(error.detail);
    }
    throw new ApiException(response.status, detailMessage);
  }

  return data as T;
}

/**
 * Multipart form data fetch for file uploads
 */
async function apiUpload<T>(
  endpoint: string,
  formData: FormData
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const headers: HeadersInit = {};
  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(url, {
    method: 'POST',
    body: formData,
    headers,
    // Don't set Content-Type header - browser will set it with boundary
  });

  const data = await response.json();

  if (!response.ok) {
    const error = data as ApiError;
    // Handle both string and object error details
    let detailMessage = 'Upload failed';
    if (typeof error.detail === 'string') {
      detailMessage = error.detail;
    } else if (error.detail && typeof error.detail === 'object') {
      const detailObj = error.detail as { message?: string; feedback?: string };
      detailMessage = detailObj.message || detailObj.feedback || JSON.stringify(error.detail);
    }
    throw new ApiException(response.status, detailMessage);
  }

  return data as T;
}

// ============================================
// Health Check
// ============================================

export async function getHealth(): Promise<HealthStatus> {
  return apiFetch<HealthStatus>('/health');
}

export async function getApiInfo(): Promise<ApiInfo> {
  return apiFetch<ApiInfo>('/');
}

// ============================================
// Projects
// ============================================

export async function listProjects(
  skip = 0,
  limit = 100
): Promise<ProjectListResponse> {
  return apiFetch<ProjectListResponse>(`/projects?skip=${skip}&limit=${limit}`);
}

export async function getProject(projectId: string): Promise<Project> {
  return apiFetch<Project>(`/projects/${projectId}`);
}

export async function createProject(data: ProjectCreate): Promise<Project> {
  return apiFetch<Project>('/projects', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateProject(
  projectId: string,
  data: ProjectUpdate
): Promise<Project> {
  return apiFetch<Project>(`/projects/${projectId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteProject(projectId: string): Promise<void> {
  return apiFetch<void>(`/projects/${projectId}`, {
    method: 'DELETE',
  });
}

// ============================================
// Data Sources
// ============================================

export async function listDataSources(projectId: string): Promise<DataSource[]> {
  return apiFetch<DataSource[]>(`/projects/${projectId}/data-sources`);
}

export async function getDataSource(dataSourceId: string): Promise<DataSource> {
  return apiFetch<DataSource>(`/data-sources/${dataSourceId}`);
}

export async function createDataSource(
  projectId: string,
  data: DataSourceCreate
): Promise<DataSource> {
  return apiFetch<DataSource>(`/projects/${projectId}/data-sources`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function uploadDataSource(
  projectId: string,
  file: File,
  options?: {
    name?: string;
    delimiter?: string;
    sheetName?: string;
  }
): Promise<DataSource> {
  const formData = new FormData();
  formData.append('file', file);

  if (options?.name) {
    formData.append('name', options.name);
  }
  if (options?.delimiter) {
    formData.append('delimiter', options.delimiter);
  }
  if (options?.sheetName) {
    formData.append('sheet_name', options.sheetName);
  }

  return apiUpload<DataSource>(
    `/projects/${projectId}/data-sources/upload`,
    formData
  );
}

export async function deleteDataSource(dataSourceId: string): Promise<void> {
  return apiFetch<void>(`/data-sources/${dataSourceId}`, {
    method: 'DELETE',
  });
}

export interface DataPreviewResponse {
  columns: string[];
  rows: (string | number | boolean | null)[][];
  total_rows: number;
  total_columns: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export async function getDataSourceData(
  dataSourceId: string,
  page: number = 1,
  pageSize: number = 100
): Promise<DataPreviewResponse> {
  return apiFetch<DataPreviewResponse>(
    `/data-sources/${dataSourceId}/data?page=${page}&page_size=${pageSize}`
  );
}

// ============================================
// Dataset Specs
// ============================================

export async function listDatasetSpecs(
  projectId: string
): Promise<DatasetSpec[]> {
  return apiFetch<DatasetSpec[]>(`/projects/${projectId}/dataset-specs`);
}

export async function getDatasetSpec(
  datasetSpecId: string
): Promise<DatasetSpec> {
  return apiFetch<DatasetSpec>(`/dataset-specs/${datasetSpecId}`);
}

export async function createDatasetSpec(
  projectId: string,
  data: DatasetSpecCreate
): Promise<DatasetSpec> {
  return apiFetch<DatasetSpec>(`/projects/${projectId}/dataset-specs`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateDatasetSpec(
  datasetSpecId: string,
  data: DatasetSpecUpdate
): Promise<DatasetSpec> {
  return apiFetch<DatasetSpec>(`/dataset-specs/${datasetSpecId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteDatasetSpec(datasetSpecId: string): Promise<void> {
  return apiFetch<void>(`/dataset-specs/${datasetSpecId}`, {
    method: 'DELETE',
  });
}

export async function getDatasetSpecData(
  datasetSpecId: string,
  page: number = 1,
  pageSize: number = 100
): Promise<DataPreviewResponse> {
  return apiFetch<DataPreviewResponse>(
    `/dataset-specs/${datasetSpecId}/data?page=${page}&page_size=${pageSize}`
  );
}

/**
 * Download a dataset as a CSV file.
 * Triggers a browser download of the built dataset.
 */
export async function downloadDatasetSpec(datasetSpecId: string): Promise<void> {
  const token = getAuthToken();
  const headers: HeadersInit = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}/dataset-specs/${datasetSpecId}/download`, {
    method: 'GET',
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Download failed' }));
    throw new ApiException(response.status, errorData.detail || 'Download failed');
  }

  // Get filename from Content-Disposition header or use default
  const contentDisposition = response.headers.get('Content-Disposition');
  let filename = 'dataset.csv';
  if (contentDisposition) {
    const match = contentDisposition.match(/filename="?([^";\n]+)"?/);
    if (match) {
      filename = match[1];
    }
  }

  // Create blob and trigger download
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

// ============================================
// Dataset Results (Experiments by Dataset)
// ============================================

export interface DatasetExperiment {
  id: string;
  name: string;
  description: string | null;
  status: string;
  primary_metric: string | null;
  metric_direction: string | null;
  best_score: number | null;
  best_metrics: Record<string, number> | null;
  best_model_type: string | null;
  iteration_number: number;
  parent_experiment_id: string | null;
  improvement_summary: string | null;
  created_at: string;
  completed_at: string | null;
  training_time_seconds: number | null;
}

export interface DatasetResultsResponse {
  dataset_spec_id: string;
  dataset_name: string;
  dataset_description: string | null;
  target_column: string | null;
  total_experiments: number;
  completed_experiments: number;
  best_experiment_id: string | null;
  best_score: number | null;
  primary_metric: string | null;
  experiments: DatasetExperiment[];
}

/**
 * Get all experiments that use a specific dataset specification.
 * Useful for comparing multiple experiment runs on the same dataset.
 */
export async function getDatasetExperiments(
  datasetSpecId: string,
  includeIterations: boolean = true
): Promise<DatasetResultsResponse> {
  const params = new URLSearchParams();
  params.append('include_iterations', includeIterations.toString());
  return apiFetch<DatasetResultsResponse>(
    `/dataset-specs/${datasetSpecId}/experiments?${params.toString()}`
  );
}

// ============================================
// Experiments
// ============================================

export async function listExperiments(
  projectId: string
): Promise<Experiment[]> {
  return apiFetch<Experiment[]>(`/projects/${projectId}/experiments`);
}

export async function getExperiment(
  experimentId: string
): Promise<ExperimentDetail> {
  return apiFetch<ExperimentDetail>(`/experiments/${experimentId}`);
}

export async function createExperiment(
  projectId: string,
  data: ExperimentCreate
): Promise<Experiment> {
  return apiFetch<Experiment>(`/projects/${projectId}/experiments`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateExperiment(
  experimentId: string,
  data: ExperimentUpdate
): Promise<Experiment> {
  return apiFetch<Experiment>(`/experiments/${experimentId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteExperiment(experimentId: string): Promise<void> {
  return apiFetch<void>(`/experiments/${experimentId}`, {
    method: 'DELETE',
  });
}

export async function runExperiment(
  experimentId: string,
  options?: {
    backend?: 'modal';
  }
): Promise<ExperimentRunResponse> {
  // Build request body if options provided
  const body = options
    ? JSON.stringify({
        training_options: {
          backend: options.backend || 'modal',
        },
      })
    : undefined;

  return apiFetch<ExperimentRunResponse>(`/experiments/${experimentId}/run`, {
    method: 'POST',
    body,
  });
}

export async function getTrainingOptions(): Promise<TrainingOptionsInfo> {
  return apiFetch<TrainingOptionsInfo>('/training-options');
}

export async function cancelExperiment(
  experimentId: string
): Promise<ExperimentRunResponse> {
  return apiFetch<ExperimentRunResponse>(
    `/experiments/${experimentId}/cancel`,
    {
      method: 'POST',
    }
  );
}

export type IssueType = 'split_strategy' | 'overfitting' | 'class_imbalance' | 'data_leakage';

export interface ApplyFixRequest {
  issue_type: IssueType;
  issue_description: string;
  recommended_fix?: string;
}

export interface ApplyFixResponse {
  experiment_id: string;
  issue_type: string;
  changes_applied: Record<string, unknown>;
  message: string;
}

export async function applyExperimentFix(
  experimentId: string,
  request: ApplyFixRequest
): Promise<ApplyFixResponse> {
  return apiFetch<ApplyFixResponse>(
    `/experiments/${experimentId}/apply-fix`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

export async function getExperimentProgress(
  experimentId: string
): Promise<ExperimentProgress> {
  return apiFetch<ExperimentProgress>(`/experiments/${experimentId}/progress`);
}

// ============================================
// Notebook Generation
// ============================================

export interface NotebookCell {
  cell_type: 'markdown' | 'code';
  source: string[];
  outputs?: unknown[];
  execution_count?: number | null;
  metadata?: Record<string, unknown>;
}

export interface JupyterNotebook {
  cells: NotebookCell[];
  metadata: {
    kernelspec?: {
      display_name: string;
      language: string;
      name: string;
    };
    language_info?: {
      name: string;
      version: string;
    };
  };
  nbformat: number;
  nbformat_minor: number;
}

export async function getExperimentNotebook(
  experimentId: string
): Promise<JupyterNotebook> {
  return apiFetch<JupyterNotebook>(`/experiments/${experimentId}/notebook`);
}

export function downloadExperimentNotebook(experimentId: string): void {
  const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001/api';
  window.open(`${baseUrl}/experiments/${experimentId}/notebook?format=download`, '_blank');
}

// ============================================
// Auto-Improve Pipeline
// ============================================

export interface ImproveExperimentResponse {
  experiment_id: string;
  task_id: string;
  message: string;
  status: string;
}

export interface ExperimentIteration {
  id: string;
  name: string;
  iteration_number: number;
  status: string;
  best_score: number | null;  // DEPRECATED: Use final_score instead
  primary_metric: string | null;
  improvement_summary: string | null;
  metrics: Record<string, number> | null;  // Full metrics for comparison
  created_at: string;
  // Holdout-based scoring (Make Holdout Score the Real Score)
  final_score: number | null;  // Canonical score from holdout evaluation
  val_score: number | null;  // Validation/CV score (NOT the final score)
  has_holdout: boolean;  // Whether holdout evaluation was performed
  overfitting_gap: number | null;  // Gap between val and holdout (positive = overfitting)
  score_source: 'holdout' | 'validation';  // Source of final_score
}

export interface ExperimentIterationsResponse {
  root_experiment_id: string;
  total_iterations: number;
  iterations: ExperimentIteration[];
}

export interface ImprovementStatusStep {
  step_type: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
}

export interface ImprovementStatusResponse {
  experiment_id: string;
  has_improvement_run: boolean;
  agent_run_id?: string;
  status?: string;
  steps?: ImprovementStatusStep[];
  result?: {
    new_experiment_id: string;
    new_dataset_spec_id: string;
    iteration_number: number;
    improvement_summary: string;
  };
  error_message?: string;
  message?: string;
}

/**
 * Trigger the auto-improve pipeline for a completed experiment.
 * This will analyze results, create an improvement plan, and start a new training iteration.
 */
export async function triggerImprovement(
  experimentId: string
): Promise<ImproveExperimentResponse> {
  return apiFetch<ImproveExperimentResponse>(
    `/experiments/${experimentId}/improve`,
    {
      method: 'POST',
    }
  );
}

/**
 * Get all iterations of an experiment (the chain from root to current).
 */
export async function getExperimentIterations(
  experimentId: string
): Promise<ExperimentIterationsResponse> {
  return apiFetch<ExperimentIterationsResponse>(
    `/experiments/${experimentId}/iterations`
  );
}

/**
 * Get the status of an ongoing improvement pipeline.
 */
export async function getImprovementStatus(
  experimentId: string
): Promise<ImprovementStatusResponse> {
  return apiFetch<ImprovementStatusResponse>(
    `/experiments/${experimentId}/improvement-status`
  );
}

// ============================================
// Auto-Iterate Settings
// ============================================

export interface AutoIterateSettingsRequest {
  enabled: boolean;
  max_iterations: number;
}

export interface AutoIterateSettingsResponse {
  experiment_id: string;
  auto_iterate_enabled: boolean;
  auto_iterate_max: number;
  current_iteration: number;
  can_continue: boolean;
  message?: string;
}

/**
 * Update auto-iterate settings for an experiment.
 * When enabled, the experiment will automatically run AI feedback and create iterations.
 */
export async function updateAutoIterateSettings(
  experimentId: string,
  settings: AutoIterateSettingsRequest
): Promise<AutoIterateSettingsResponse> {
  return apiFetch<AutoIterateSettingsResponse>(
    `/experiments/${experimentId}/auto-iterate`,
    {
      method: 'PUT',
      body: JSON.stringify(settings),
    }
  );
}

/**
 * Get current auto-iterate settings for an experiment.
 */
export async function getAutoIterateSettings(
  experimentId: string
): Promise<AutoIterateSettingsResponse> {
  return apiFetch<AutoIterateSettingsResponse>(
    `/experiments/${experimentId}/auto-iterate`
  );
}

// ============================================
// Trials
// ============================================

export async function listTrials(experimentId: string): Promise<Trial[]> {
  return apiFetch<Trial[]>(`/experiments/${experimentId}/trials`);
}

export async function getTrial(trialId: string): Promise<Trial> {
  return apiFetch<Trial>(`/trials/${trialId}`);
}

export async function createTrial(
  experimentId: string,
  data: TrialCreate
): Promise<Trial> {
  return apiFetch<Trial>(`/experiments/${experimentId}/trials`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ============================================
// Model Versions
// ============================================

export async function listModelVersions(
  projectId: string,
  statusFilter?: string
): Promise<ModelVersion[]> {
  let url = `/projects/${projectId}/models`;
  if (statusFilter) {
    url += `?status_filter=${statusFilter}`;
  }
  return apiFetch<ModelVersion[]>(url);
}

export async function getModelVersion(modelId: string): Promise<ModelVersion> {
  return apiFetch<ModelVersion>(`/models/${modelId}`);
}

export async function createModelVersion(
  projectId: string,
  data: ModelVersionCreate
): Promise<ModelVersion> {
  return apiFetch<ModelVersion>(`/projects/${projectId}/models`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function promoteModel(
  modelId: string,
  data: ModelPromoteRequest
): Promise<ModelVersion> {
  return apiFetch<ModelVersion>(`/models/${modelId}/promote`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function deleteModelVersion(modelId: string): Promise<void> {
  return apiFetch<void>(`/models/${modelId}`, {
    method: 'DELETE',
  });
}

export async function predictWithModel(
  modelId: string,
  features: Record<string, unknown>
): Promise<PredictionResponse> {
  // Use predict-auto which handles both local and remote (Modal) predictions
  return apiFetch<PredictionResponse>(`/models/${modelId}/predict-auto`, {
    method: 'POST',
    body: JSON.stringify({ features }),
  });
}

export async function explainModel(
  modelId: string,
  question: string
): Promise<ModelExplainResponse> {
  return apiFetch<ModelExplainResponse>(`/models/${modelId}/explain`, {
    method: 'POST',
    body: JSON.stringify({ question }),
  });
}

// ============================================
// Model Testing
// ============================================

/**
 * Get feature statistics and sample data for the model testing UI.
 * Includes min/max/median for numeric features, categories for categorical,
 * feature importance ranking, and a random sample to start with.
 */
export async function getModelTestingData(
  modelId: string
): Promise<ModelTestingDataResponse> {
  return apiFetch<ModelTestingDataResponse>(`/models/${modelId}/testing-data`);
}

/**
 * Get a random sample from validation data for model testing.
 * Returns feature values that can be used directly in the prediction form.
 */
export async function getRandomSample(
  modelId: string
): Promise<RandomSampleResponse> {
  return apiFetch<RandomSampleResponse>(`/models/${modelId}/random-sample`);
}

// ============================================
// Raw Data Predictions
// ============================================

import type {
  RawPredictionResponse,
  BatchPredictionResponse,
  FeaturePipelineInfo,
  ModelExportInfo,
} from '../types/api';

/**
 * Make a prediction using raw, untransformed data.
 * The backend will automatically apply feature transformations.
 */
export async function predictWithRawData(
  modelId: string,
  data: Record<string, unknown>,
  includeTransformed: boolean = false
): Promise<RawPredictionResponse> {
  return apiFetch<RawPredictionResponse>(
    `/models/${modelId}/predict-raw?include_transformed=${includeTransformed}`,
    {
      method: 'POST',
      body: JSON.stringify({ data }),
    }
  );
}

/**
 * Make batch predictions using raw data.
 */
export async function predictBatchWithRawData(
  modelId: string,
  data: Record<string, unknown>[]
): Promise<BatchPredictionResponse> {
  return apiFetch<BatchPredictionResponse>(`/models/${modelId}/predict-batch`, {
    method: 'POST',
    body: JSON.stringify({ data }),
  });
}

/**
 * Get information about the feature pipeline for a model.
 */
export async function getModelPipeline(
  modelId: string
): Promise<FeaturePipelineInfo> {
  return apiFetch<FeaturePipelineInfo>(`/models/${modelId}/pipeline`);
}

/**
 * Get export information for a model.
 */
export async function getModelExportInfo(
  modelId: string
): Promise<ModelExportInfo> {
  return apiFetch<ModelExportInfo>(`/models/${modelId}/export-info`);
}

/**
 * Get the URL to download/export a model.
 * Returns the full URL that can be used in an anchor tag or window.open.
 */
export function getModelExportUrl(modelId: string): string {
  return `${API_BASE_URL}/models/${modelId}/export`;
}

// ============================================
// API Keys
// ============================================

export async function getApiKeyStatus(): Promise<ApiKeyStatus> {
  return apiFetch<ApiKeyStatus>('/api/v1/api-keys/status');
}

export async function listApiKeys(): Promise<ApiKey[]> {
  return apiFetch<ApiKey[]>('/api/v1/api-keys');
}

export async function createOrUpdateApiKey(data: ApiKeyCreate): Promise<ApiKey> {
  return apiFetch<ApiKey>('/api/v1/api-keys', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function deleteApiKey(provider: LLMProvider): Promise<void> {
  return apiFetch<void>(`/api/v1/api-keys/${provider}`, {
    method: 'DELETE',
  });
}

// ============================================
// App Settings
// ============================================

import type { AIModelOption, AppSettings, AppSettingsUpdate } from '../types/api';

export async function getAvailableAIModels(): Promise<AIModelOption[]> {
  return apiFetch<AIModelOption[]>('/api/v1/settings/ai-models');
}

export async function getAppSettings(): Promise<AppSettings> {
  return apiFetch<AppSettings>('/api/v1/settings');
}

export async function updateAppSettings(data: AppSettingsUpdate): Promise<AppSettings> {
  return apiFetch<AppSettings>('/api/v1/settings', {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

// ============================================
// Chat
// ============================================

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  message: string;
  context?: Record<string, unknown>;
  history?: ChatMessage[];
  provider?: 'openai' | 'gemini';
}

export interface ChatResponse {
  response: string;
  provider: string;
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  return apiFetch<ChatResponse>('/api/v1/chat', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

// ============================================
// Conversations (Persistent Chat)
// ============================================

export interface ConversationMessage {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  context_type: string | null;
  context_id: string | null;
  context_data: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  context_type: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message_preview: string | null;
}

export interface ConversationListResponse {
  items: ConversationSummary[];
  total: number;
}

export interface ConversationCreate {
  title?: string;
  context_type?: string;
  context_id?: string;
  context_data?: Record<string, unknown>;
}

export interface SendMessageResponse {
  user_message: ConversationMessage;
  assistant_message: ConversationMessage;
  provider: string;
}

export async function listConversations(
  contextType?: string,
  skip = 0,
  limit = 50
): Promise<ConversationListResponse> {
  let url = `/api/v1/chat/conversations?skip=${skip}&limit=${limit}`;
  if (contextType) {
    url += `&context_type=${contextType}`;
  }
  return apiFetch<ConversationListResponse>(url);
}

export async function getConversation(conversationId: string): Promise<Conversation> {
  return apiFetch<Conversation>(`/api/v1/chat/conversations/${conversationId}`);
}

export async function createConversation(data: ConversationCreate): Promise<Conversation> {
  return apiFetch<Conversation>('/api/v1/chat/conversations', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function updateConversation(
  conversationId: string,
  data: { title?: string }
): Promise<Conversation> {
  return apiFetch<Conversation>(`/api/v1/chat/conversations/${conversationId}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

export async function deleteConversation(conversationId: string): Promise<void> {
  return apiFetch<void>(`/api/v1/chat/conversations/${conversationId}`, {
    method: 'DELETE',
  });
}

export interface VisualizationForChat {
  title: string;
  description?: string;
  chart_type?: string;
  image_base64?: string;
}

export async function sendConversationMessage(
  conversationId: string,
  message: string,
  currentVisualizations?: VisualizationForChat[]
): Promise<SendMessageResponse> {
  const body: Record<string, unknown> = { message };
  if (currentVisualizations && currentVisualizations.length > 0) {
    body.current_visualizations = currentVisualizations;
  }
  return apiFetch<SendMessageResponse>(
    `/api/v1/chat/conversations/${conversationId}/messages`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    }
  );
}

// ============================================
// Authentication
// ============================================

export async function signup(data: UserCreate): Promise<User> {
  return apiFetch<User>('/api/v1/auth/signup', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function login(data: UserLogin): Promise<AuthToken> {
  return apiFetch<AuthToken>('/api/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function googleAuth(data: GoogleAuthRequest): Promise<AuthToken> {
  return apiFetch<AuthToken>('/api/v1/auth/google', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getCurrentUser(): Promise<User> {
  return apiFetch<User>('/api/v1/auth/me', {}, true);
}

export async function updateCurrentUser(data: { full_name?: string }): Promise<User> {
  return apiFetch<User>('/api/v1/auth/me', {
    method: 'PUT',
    body: JSON.stringify(data),
  }, true);
}

export async function changePassword(data: {
  current_password: string;
  new_password: string;
}): Promise<{ message: string }> {
  return apiFetch<{ message: string }>('/api/v1/auth/change-password', {
    method: 'POST',
    body: JSON.stringify(data),
  }, true);
}

export async function logout(): Promise<void> {
  clearAuthToken();
}

// ============================================
// Project Sharing
// ============================================

export async function listProjectShares(projectId: string): Promise<ShareListResponse> {
  return apiFetch<ShareListResponse>(`/api/v1/sharing/projects/${projectId}/shares`, {}, true);
}

export async function shareProject(projectId: string, data: ShareRequest): Promise<Share> {
  return apiFetch<Share>(`/api/v1/sharing/projects/${projectId}/shares`, {
    method: 'POST',
    body: JSON.stringify(data),
  }, true);
}

export async function updateProjectShare(
  projectId: string,
  shareId: string,
  data: { role: string }
): Promise<Share> {
  return apiFetch<Share>(`/api/v1/sharing/projects/${projectId}/shares/${shareId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }, true);
}

export async function removeProjectShare(projectId: string, shareId: string): Promise<void> {
  return apiFetch<void>(`/api/v1/sharing/projects/${projectId}/shares/${shareId}`, {
    method: 'DELETE',
  }, true);
}

// Sharing - Datasets
export async function listDatasetShares(datasetId: string): Promise<ShareListResponse> {
  return apiFetch<ShareListResponse>(`/api/v1/sharing/datasets/${datasetId}/shares`, {}, true);
}

export async function shareDataset(datasetId: string, data: ShareRequest): Promise<Share> {
  return apiFetch<Share>(`/api/v1/sharing/datasets/${datasetId}/shares`, {
    method: 'POST',
    body: JSON.stringify(data),
  }, true);
}

export async function removeDatasetShare(datasetId: string, shareId: string): Promise<void> {
  return apiFetch<void>(`/api/v1/sharing/datasets/${datasetId}/shares/${shareId}`, {
    method: 'DELETE',
  }, true);
}

// ============================================
// Sharing - Accept Invitations
// ============================================

export async function acceptInvite(token: string): Promise<{
  message: string;
  resource_type: string;
  resource_id: string;
}> {
  return apiFetch<{ message: string; resource_type: string; resource_id: string }>(
    '/api/v1/sharing/accept-invite',
    {
      method: 'POST',
      body: JSON.stringify({ token }),
    },
    true
  );
}

export async function getMyShares(): Promise<MyShares> {
  return apiFetch<MyShares>('/api/v1/sharing/my-shares', {}, true);
}

// ============================================
// Agent - LLM-powered Configuration
// ============================================

export interface ColumnSummary {
  name: string;
  dtype: string;
  inferred_type: string;
  null_percentage: number;
  unique_count: number;
  min?: number;
  max?: number;
  mean?: number;
  top_values?: Record<string, number>;
}

export interface SchemaSummary {
  data_source_id: string;
  data_source_name: string;
  file_type: string;
  row_count: number;
  column_count: number;
  columns: ColumnSummary[];
}

export interface ProjectConfigSuggestion {
  task_type: string;
  target_column: string;
  primary_metric: string;
  reasoning: string;
  confidence: number;
  suggested_name?: string;
}

export interface ProjectConfigRequest {
  description: string;
  data_source_id: string;
}

export interface ProjectConfigResponse {
  suggestion: ProjectConfigSuggestion;
  schema_summary: SchemaSummary;
}

export interface DatasetSpecSuggestion {
  feature_columns: string[];
  excluded_columns: string[];
  exclusion_reasons: Record<string, string>;
  suggested_filters?: Record<string, unknown>;
  reasoning: string;
  warnings: string[];
}

export interface DatasetSpecRequest {
  data_source_id: string;
  task_type: string;
  target_column: string;
  description?: string;
}

export interface DatasetSpecSuggestionResponse {
  suggestion: DatasetSpecSuggestion;
  schema_summary: SchemaSummary;
}

export interface ExperimentVariant {
  name: string;
  description: string;
  automl_config: Record<string, unknown>;
  expected_tradeoff: string;
}

export interface ExperimentPlanSuggestion {
  variants: ExperimentVariant[];
  recommended_variant: string;
  reasoning: string;
  estimated_total_time_minutes: number;
}

export interface ExperimentPlanRequest {
  task_type: string;
  target_column: string;
  primary_metric: string;
  feature_columns: string[];
  row_count: number;
  column_count: number;
  time_budget_minutes?: number;
  description?: string;
}

export interface ExperimentPlanResponse {
  suggestion: ExperimentPlanSuggestion;
}

export async function getSchemaSummary(
  projectId: string,
  dataSourceId: string
): Promise<SchemaSummary> {
  return apiFetch<SchemaSummary>(
    `/projects/${projectId}/agent/schema-summary/${dataSourceId}`
  );
}

export async function suggestProjectConfig(
  projectId: string,
  request: ProjectConfigRequest
): Promise<ProjectConfigResponse> {
  return apiFetch<ProjectConfigResponse>(
    `/projects/${projectId}/agent/suggest-config`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

export async function suggestDatasetSpec(
  projectId: string,
  request: DatasetSpecRequest
): Promise<DatasetSpecSuggestionResponse> {
  return apiFetch<DatasetSpecSuggestionResponse>(
    `/projects/${projectId}/agent/suggest-dataset-spec`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

export async function suggestExperimentPlan(
  projectId: string,
  request: ExperimentPlanRequest
): Promise<ExperimentPlanResponse> {
  return apiFetch<ExperimentPlanResponse>(
    `/projects/${projectId}/agent/suggest-experiment-plan`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

// ============================================
// Agent Pipeline
// ============================================

import type {
  AgentRun,
  AgentRunList,
  AgentStep,
  AgentStepLogList,
  SetupPipelineRequest,
  SetupPipelineResponse,
  DataArchitectRequest,
  DataArchitectResponse,
} from '../types/api';

export async function runSetupPipeline(
  projectId: string,
  request: SetupPipelineRequest
): Promise<SetupPipelineResponse> {
  return apiFetch<SetupPipelineResponse>(
    `/projects/${projectId}/agent/run-setup-pipeline`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

export async function listAgentRuns(
  projectId: string,
  skip = 0,
  limit = 20
): Promise<AgentRunList> {
  return apiFetch<AgentRunList>(
    `/projects/${projectId}/agent/runs?skip=${skip}&limit=${limit}`
  );
}

export async function getAgentRun(runId: string): Promise<AgentRun> {
  return apiFetch<AgentRun>(`/agent-runs/${runId}`);
}

export async function deleteAgentRun(runId: string): Promise<void> {
  await apiFetch<void>(`/agent-runs/${runId}`, { method: 'DELETE' });
}

export interface CancelAgentRunResponse {
  run_id: string;
  status: string;
  message: string;
}

export async function cancelAgentRun(runId: string): Promise<CancelAgentRunResponse> {
  return apiFetch<CancelAgentRunResponse>(`/agent-runs/${runId}/cancel`, {
    method: 'POST',
  });
}

export async function getAgentStep(stepId: string): Promise<AgentStep> {
  return apiFetch<AgentStep>(`/agent-steps/${stepId}`);
}

export async function getAgentStepLogs(
  stepId: string,
  sinceSequence = 0,
  limit = 100
): Promise<AgentStepLogList> {
  return apiFetch<AgentStepLogList>(
    `/agent-steps/${stepId}/logs?since_sequence=${sinceSequence}&limit=${limit}`
  );
}

/**
 * Run the Data Architect pipeline to automatically build a training dataset.
 * Steps: Dataset Inventory → Relationship Discovery → Training Dataset Planning → Training Dataset Build
 */
export async function runDataArchitectPipeline(
  projectId: string,
  request?: DataArchitectRequest
): Promise<DataArchitectResponse> {
  return apiFetch<DataArchitectResponse>(
    `/projects/${projectId}/agent/run-data-architect`,
    {
      method: 'POST',
      body: JSON.stringify(request || {}),
    }
  );
}

// ============================================
// Orchestration Options
// ============================================

import type { OrchestrationOptions } from '../types/api';

/**
 * Get available orchestration options for pipelines.
 * Returns available orchestration modes, debate modes, and judge models.
 */
export async function getOrchestrationOptions(): Promise<OrchestrationOptions> {
  return apiFetch<OrchestrationOptions>('/orchestration/options');
}

// ============================================
// Agent Step Actions (Apply Outputs)
// ============================================

export interface ApplyDatasetSpecRequest {
  target_column?: string;
  feature_columns?: string[];
  name?: string;
}

export interface ApplyDatasetSpecResponse {
  dataset_spec_id: string;
  message: string;
}

export interface ApplyExperimentPlanResponse {
  experiment_id: string;
  message: string;
}

/**
 * Create a DatasetSpec from a dataset_design agent step's output.
 * Users can optionally override the AI suggestions by passing modified values.
 */
export async function applyDatasetSpecFromStep(
  projectId: string,
  stepId: string,
  modifications?: ApplyDatasetSpecRequest
): Promise<ApplyDatasetSpecResponse> {
  return apiFetch<ApplyDatasetSpecResponse>(
    `/projects/${projectId}/agent/apply-dataset-spec-from-step/${stepId}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: modifications ? JSON.stringify(modifications) : undefined,
    }
  );
}

/**
 * Create an Experiment from an experiment_design agent step's output.
 *
 * @param projectId - The project UUID
 * @param stepId - The agent step UUID (must be a completed experiment_design step)
 * @param datasetSpecId - Optional DatasetSpec UUID to use
 * @param variant - Optional variant name (e.g., 'quick', 'balanced', 'high_quality')
 */
export async function applyExperimentPlanFromStep(
  projectId: string,
  stepId: string,
  datasetSpecId?: string,
  variant?: string
): Promise<ApplyExperimentPlanResponse> {
  let url = `/projects/${projectId}/agent/apply-experiment-plan-from-step/${stepId}`;
  const params = new URLSearchParams();

  if (datasetSpecId) {
    params.append('dataset_spec_id', datasetSpecId);
  }
  if (variant) {
    params.append('variant', variant);
  }

  const queryString = params.toString();
  if (queryString) {
    url += `?${queryString}`;
  }

  return apiFetch<ApplyExperimentPlanResponse>(url, {
    method: 'POST',
  });
}

// ============================================
// Batch Dataset and Experiment Operations
// ============================================

export interface BatchApplyDatasetSpecRequest {
  variant_names: string[];
}

export interface BatchDatasetSpecResult {
  dataset_spec_id: string;
  variant_name: string;
  feature_count: number;
  train_test_split: string;
}

export interface BatchApplyDatasetSpecResponse {
  dataset_specs: BatchDatasetSpecResult[];
  message: string;
}

/**
 * Create multiple DatasetSpecs from selected dataset variants.
 */
export async function applyDatasetSpecsBatch(
  projectId: string,
  stepId: string,
  variantNames: string[]
): Promise<BatchApplyDatasetSpecResponse> {
  return apiFetch<BatchApplyDatasetSpecResponse>(
    `/projects/${projectId}/agent/apply-dataset-specs-batch/${stepId}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ variant_names: variantNames }),
    }
  );
}

export interface BatchApplyExperimentPlanRequest {
  variant?: string;
  /** If true, create one experiment PER VARIANT for each dataset spec (follows agent's full recommendation) */
  create_all_variants?: boolean;
  run_immediately?: boolean;
}

export interface BatchExperimentResult {
  experiment_id: string;
  dataset_spec_id: string;
  dataset_spec_name: string;
  variant_name?: string;
  is_recommended?: boolean;
  name: string;
  status: string;
  task_id?: string;
  error?: string;
}

export interface BatchApplyExperimentPlanResponse {
  experiments: BatchExperimentResult[];
  message: string;
  created_count: number;
  queued_count: number;
}

/**
 * Create experiments for ALL dataset specs in the project from an experiment_design step.
 * Optionally queues them for immediate execution.
 */
export async function applyExperimentsBatch(
  projectId: string,
  stepId: string,
  options?: BatchApplyExperimentPlanRequest
): Promise<BatchApplyExperimentPlanResponse> {
  return apiFetch<BatchApplyExperimentPlanResponse>(
    `/projects/${projectId}/agent/apply-experiments-batch/${stepId}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options || { run_immediately: true }),
    }
  );
}

export interface CreateExperimentsForDatasetRequest {
  dataset_spec_id: string;
  /** If true, create one experiment per variant. If false, create only the recommended variant. */
  create_all_variants?: boolean;
  run_immediately?: boolean;
}

export interface CreateExperimentsForDatasetResponse {
  experiments: BatchExperimentResult[];
  message: string;
  created_count: number;
  queued_count: number;
}

/**
 * Create experiments for a SINGLE dataset spec following the agent's recommendation.
 * This is useful from the dataset detail modal when user wants to run experiments
 * for just one dataset.
 */
export async function createExperimentsForDataset(
  projectId: string,
  stepId: string,
  datasetSpecId: string,
  options?: { create_all_variants?: boolean; run_immediately?: boolean }
): Promise<CreateExperimentsForDatasetResponse> {
  return apiFetch<CreateExperimentsForDatasetResponse>(
    `/projects/${projectId}/agent/create-experiments-for-dataset/${stepId}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        dataset_spec_id: datasetSpecId,
        create_all_variants: options?.create_all_variants ?? true,
        run_immediately: options?.run_immediately ?? true,
      }),
    }
  );
}

/**
 * Create experiments for a dataset spec using its stored agent experiment design config.
 * This allows creating experiments without requiring the original agent step,
 * useful when the agent run has been superseded by newer runs.
 */
export async function createExperimentsFromStoredConfig(
  projectId: string,
  datasetSpecId: string,
  options?: { create_all_variants?: boolean; run_immediately?: boolean }
): Promise<CreateExperimentsForDatasetResponse> {
  return apiFetch<CreateExperimentsForDatasetResponse>(
    `/projects/${projectId}/agent/create-experiments-from-stored-config`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        dataset_spec_id: datasetSpecId,
        create_all_variants: options?.create_all_variants ?? true,
        run_immediately: options?.run_immediately ?? true,
      }),
    }
  );
}

export interface BatchRunExperimentsRequest {
  experiment_ids: string[];
}

export interface BatchExperimentStatus {
  experiment_id: string;
  status: string;
  task_id?: string;
  message: string;
}

export interface BatchRunExperimentsResponse {
  experiments: BatchExperimentStatus[];
  message: string;
  queued_count: number;
  failed_count: number;
}

/**
 * Run multiple experiments in parallel.
 */
export async function runExperimentsBatch(
  experimentIds: string[]
): Promise<BatchRunExperimentsResponse> {
  return apiFetch<BatchRunExperimentsResponse>(
    `/experiments/run-batch`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ experiment_ids: experimentIds }),
    }
  );
}

// ============================================
// Visualization
// ============================================

export interface VisualizationRequest {
  data_source_id: string;
  request: string;
  previous_visualizations?: { title: string; description: string }[];
}

export interface VisualizationResponse {
  code: string;
  title: string;
  description: string;
  chart_type: string;
  image_base64?: string;
  error?: string;
}

export interface ExecuteCodeRequest {
  code: string;
}

export interface ExecuteCodeResponse {
  image_base64?: string;
  error?: string;
}

export interface ExplainVisualizationRequest {
  data_source_id: string;
  visualization_info: {
    title: string;
    description: string;
    chart_type: string;
  };
}

export interface ExplainVisualizationResponse {
  explanation: string;
}

export interface VisualizationSuggestion {
  title: string;
  description: string;
  chart_type: string;
  request: string;
}

export interface SuggestVisualizationsRequest {
  data_source_id: string;
}

export interface SuggestVisualizationsResponse {
  suggestions: VisualizationSuggestion[];
  data_summary: DataSummary;
}

export interface DataSummary {
  row_count: number;
  column_count: number;
  columns: ColumnInfo[];
  column_names: string[];
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  is_numeric: boolean;
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
  unique_values?: string[];
  top_5_values?: string[];
}

export async function generateVisualization(
  projectId: string,
  request: VisualizationRequest
): Promise<VisualizationResponse> {
  return apiFetch<VisualizationResponse>(
    `/api/v1/projects/${projectId}/visualize/generate`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

export async function executeVisualizationCode(
  projectId: string,
  code: string
): Promise<ExecuteCodeResponse> {
  return apiFetch<ExecuteCodeResponse>(
    `/api/v1/projects/${projectId}/visualize/execute`,
    {
      method: 'POST',
      body: JSON.stringify({ code }),
    }
  );
}

export async function explainVisualization(
  projectId: string,
  request: ExplainVisualizationRequest
): Promise<ExplainVisualizationResponse> {
  return apiFetch<ExplainVisualizationResponse>(
    `/api/v1/projects/${projectId}/visualize/explain`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

export async function getVisualizationSuggestions(
  projectId: string,
  dataSourceId: string
): Promise<SuggestVisualizationsResponse> {
  return apiFetch<SuggestVisualizationsResponse>(
    `/api/v1/projects/${projectId}/visualize/suggestions`,
    {
      method: 'POST',
      body: JSON.stringify({ data_source_id: dataSourceId }),
    }
  );
}

export async function getDataSummary(
  projectId: string,
  dataSourceId: string
): Promise<DataSummary> {
  return apiFetch<DataSummary>(
    `/api/v1/projects/${projectId}/visualize/data-summary/${dataSourceId}`
  );
}

// ==================== Saved Visualizations ====================

export interface SaveVisualizationRequest {
  data_source_id?: string;
  title: string;
  description?: string;
  chart_type?: string;
  request?: string;
  code: string;
  image_base64?: string;
  explanation?: string;
  is_ai_suggested?: string;
  display_order?: string;
}

export interface SavedVisualization {
  id: string;
  project_id: string;
  data_source_id?: string;
  owner_id?: string;
  title: string;
  description?: string;
  chart_type?: string;
  request?: string;
  code: string;
  image_base64?: string;
  explanation?: string;
  is_ai_suggested?: string;
  display_order?: string;
  created_at: string;
  updated_at: string;
}

export async function saveVisualization(
  projectId: string,
  visualization: SaveVisualizationRequest
): Promise<SavedVisualization> {
  return apiFetch<SavedVisualization>(
    `/api/v1/projects/${projectId}/visualize/saved`,
    {
      method: 'POST',
      body: JSON.stringify(visualization),
    }
  );
}

export async function listSavedVisualizations(
  projectId: string
): Promise<SavedVisualization[]> {
  return apiFetch<SavedVisualization[]>(
    `/api/v1/projects/${projectId}/visualize/saved`
  );
}

export async function deleteSavedVisualization(
  projectId: string,
  visualizationId: string
): Promise<void> {
  await apiFetch<void>(
    `/api/v1/projects/${projectId}/visualize/saved/${visualizationId}`,
    {
      method: 'DELETE',
    }
  );
}

// ==================== Dataset Spec Visualization ====================

export interface DatasetSpecVisualizationRequest {
  dataset_spec_id: string;
  request: string;
  previous_visualizations?: { title: string; description: string }[];
}

export interface DatasetSpecSuggestionsRequest {
  dataset_spec_id: string;
}

/**
 * Generate a visualization for a dataset spec (with feature engineering applied).
 */
export async function generateDatasetSpecVisualization(
  projectId: string,
  request: DatasetSpecVisualizationRequest
): Promise<VisualizationResponse> {
  return apiFetch<VisualizationResponse>(
    `/api/v1/projects/${projectId}/visualize/dataset-spec/generate`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    }
  );
}

/**
 * Get visualization suggestions for a dataset spec.
 */
export async function getDatasetSpecVisualizationSuggestions(
  projectId: string,
  datasetSpecId: string
): Promise<SuggestVisualizationsResponse> {
  return apiFetch<SuggestVisualizationsResponse>(
    `/api/v1/projects/${projectId}/visualize/dataset-spec/suggestions`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_spec_id: datasetSpecId }),
    }
  );
}

// ============================================
// Experiment Agent (Results Pipeline)
// ============================================

export interface ResultsPipelineRequest {
  run_async?: boolean;
}

export interface ResultsPipelineResponse {
  run_id: string;
  status: string;
  message: string;
}

/**
 * Start the AI results analysis pipeline for a completed experiment.
 * This creates agent runs with RESULTS_INTERPRETATION and RESULTS_CRITIC steps.
 */
export async function runResultsPipeline(
  experimentId: string,
  options?: ResultsPipelineRequest
): Promise<ResultsPipelineResponse> {
  return apiFetch<ResultsPipelineResponse>(
    `/experiments/${experimentId}/agent/run-results-pipeline`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options || {}),
    }
  );
}

/**
 * List agent runs for an experiment (results pipelines).
 */
export async function listExperimentAgentRuns(
  experimentId: string,
  skip = 0,
  limit = 20
): Promise<AgentRunList> {
  return apiFetch<AgentRunList>(
    `/experiments/${experimentId}/agent/runs?skip=${skip}&limit=${limit}`
  );
}

/**
 * Get a specific agent run for an experiment with its steps and logs.
 */
export async function getExperimentAgentRun(
  experimentId: string,
  runId: string
): Promise<AgentRun> {
  return apiFetch<AgentRun>(
    `/experiments/${experimentId}/agent/runs/${runId}`
  );
}

// ============================================
// Dataset Discovery
// ============================================

import type {
  DatasetDiscoveryRequest,
  DatasetDiscoveryResponse,
  ApplyDiscoveredDatasetsRequest,
  ApplyDiscoveredDatasetsResponse,
  DiscoveredDataset,
  ValidationSamplesListResponse,
  ValidationSample,
  ValidationSampleSort,
  WhatIfRequest,
  WhatIfResponse,
} from '../types/api';

/**
 * Run the dataset discovery pipeline to find relevant public datasets.
 * Use this when a user wants to start a project but doesn't have their own data.
 */
export async function runDatasetDiscovery(
  projectId: string,
  request: DatasetDiscoveryRequest
): Promise<DatasetDiscoveryResponse> {
  return apiFetch<DatasetDiscoveryResponse>(
    `/projects/${projectId}/agent/run-dataset-discovery`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

/**
 * Get discovered datasets from a completed discovery run.
 * Returns the list of datasets found during the discovery process.
 */
export async function getDiscoveredDatasets(
  runId: string
): Promise<DiscoveredDataset[]> {
  const run = await getAgentRun(runId);
  if (!run.result_json) {
    return [];
  }
  return (run.result_json as { discovered_datasets?: DiscoveredDataset[] }).discovered_datasets || [];
}

/**
 * Apply selected discovered datasets as data sources for the project.
 */
export async function applyDiscoveredDatasets(
  projectId: string,
  runId: string,
  request: ApplyDiscoveredDatasetsRequest
): Promise<ApplyDiscoveredDatasetsResponse> {
  return apiFetch<ApplyDiscoveredDatasetsResponse>(
    `/projects/${projectId}/agent/apply-discovered-datasets/${runId}`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

// ============================================
// Validation Samples
// ============================================

/**
 * List validation samples for a model with pagination and sorting.
 */
export async function listValidationSamples(
  modelId: string,
  options?: {
    limit?: number;
    offset?: number;
    sort?: ValidationSampleSort;
  }
): Promise<ValidationSamplesListResponse> {
  const params = new URLSearchParams();
  if (options?.limit) params.append('limit', options.limit.toString());
  if (options?.offset) params.append('offset', options.offset.toString());
  if (options?.sort) params.append('sort', options.sort);

  const queryString = params.toString();
  const url = `/models/${modelId}/validation-samples${queryString ? `?${queryString}` : ''}`;

  return apiFetch<ValidationSamplesListResponse>(url);
}

/**
 * Get a single validation sample by ID.
 */
export async function getValidationSample(sampleId: string): Promise<ValidationSample> {
  return apiFetch<ValidationSample>(`/validation-samples/${sampleId}`);
}

/**
 * Run a what-if prediction by modifying features from an existing validation sample.
 */
export async function runWhatIfPrediction(
  modelId: string,
  request: WhatIfRequest
): Promise<WhatIfResponse> {
  return apiFetch<WhatIfResponse>(`/models/${modelId}/what-if`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

// ============================================
// Research Cycles and Lab Notebook
// ============================================

import type {
  ResearchCycleSummary,
  ResearchCycleListResponse,
  ResearchCycleResponse,
  ResearchCycleCreate,
  LabNotebookEntry,
  LabNotebookEntryCreate,
  LabNotebookEntryUpdate,
  LabNotebookEntryListResponse,
} from '../types/api';

/**
 * List all research cycles for a project.
 */
export async function listResearchCycles(
  projectId: string
): Promise<ResearchCycleListResponse> {
  return apiFetch<ResearchCycleListResponse>(
    `/projects/${projectId}/research-cycles`
  );
}

/**
 * Get a single research cycle with all linked experiments and notebook entries.
 */
export async function getResearchCycle(
  cycleId: string
): Promise<ResearchCycleResponse> {
  return apiFetch<ResearchCycleResponse>(`/research-cycles/${cycleId}`);
}

/**
 * Create a new research cycle for a project.
 */
export async function createResearchCycle(
  projectId: string,
  data?: ResearchCycleCreate
): Promise<ResearchCycleSummary> {
  return apiFetch<ResearchCycleSummary>(
    `/projects/${projectId}/research-cycles`,
    {
      method: 'POST',
      body: JSON.stringify(data || {}),
    }
  );
}

/**
 * Link an experiment to a research cycle.
 */
export async function linkExperimentToCycle(
  cycleId: string,
  experimentId: string
): Promise<{ id: string; research_cycle_id: string; experiment_id: string; linked_at: string }> {
  return apiFetch(
    `/research-cycles/${cycleId}/experiments`,
    {
      method: 'POST',
      body: JSON.stringify({ experiment_id: experimentId }),
    }
  );
}

/**
 * List lab notebook entries for a project.
 */
export async function listLabNotebookEntries(
  projectId: string,
  cycleId?: string
): Promise<LabNotebookEntryListResponse> {
  let url = `/projects/${projectId}/research-cycles/notebook`;
  if (cycleId) {
    url += `?cycle_id=${cycleId}`;
  }
  return apiFetch<LabNotebookEntryListResponse>(url);
}

/**
 * Get a single lab notebook entry by ID.
 */
export async function getLabNotebookEntry(
  entryId: string
): Promise<LabNotebookEntry> {
  return apiFetch<LabNotebookEntry>(`/notebook/${entryId}`);
}

/**
 * Create a lab notebook entry for a project.
 */
export async function createLabNotebookEntry(
  projectId: string,
  data: LabNotebookEntryCreate
): Promise<LabNotebookEntry> {
  return apiFetch<LabNotebookEntry>(
    `/projects/${projectId}/research-cycles/notebook`,
    {
      method: 'POST',
      body: JSON.stringify(data),
    }
  );
}

/**
 * Update a lab notebook entry.
 */
export async function updateLabNotebookEntry(
  entryId: string,
  data: LabNotebookEntryUpdate
): Promise<LabNotebookEntry> {
  return apiFetch<LabNotebookEntry>(`/notebook/${entryId}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

/**
 * Delete a lab notebook entry.
 */
export async function deleteLabNotebookEntry(entryId: string): Promise<void> {
  return apiFetch<void>(`/notebook/${entryId}`, {
    method: 'DELETE',
  });
}

// ============================================
// Project History
// ============================================

/**
 * Get complete project history for the History View UI.
 * Returns all research cycles, experiments, agent runs with their
 * reasoning/tool calls, notebook entries, and best models.
 */
export async function getProjectHistory(
  projectId: string,
  options?: {
    includeLogs?: boolean;
    limitCycles?: number;
    limitEntries?: number;
  }
): Promise<ProjectHistoryResponse> {
  const params = new URLSearchParams();
  if (options?.includeLogs !== undefined) {
    params.set('include_logs', String(options.includeLogs));
  }
  if (options?.limitCycles !== undefined) {
    params.set('limit_cycles', String(options.limitCycles));
  }
  if (options?.limitEntries !== undefined) {
    params.set('limit_entries', String(options.limitEntries));
  }
  const queryString = params.toString();
  const url = `/projects/${projectId}/history${queryString ? `?${queryString}` : ''}`;
  return apiFetch<ProjectHistoryResponse>(url);
}

/**
 * Get detailed thinking/reasoning for a specific agent step.
 */
export async function getAgentStepThinking(
  projectId: string,
  stepId: string
): Promise<AgentThinkingDetail> {
  return apiFetch<AgentThinkingDetail>(
    `/projects/${projectId}/history/agent-step/${stepId}/thinking`
  );
}

// =============================================================================
// User Holdout Set API
// =============================================================================

import type { HoldoutSet, HoldoutRow, HoldoutRowsResponse } from '../types/api';

/**
 * Get the holdout set for a project.
 * Returns null if no holdout set exists.
 */
export async function getHoldoutSet(projectId: string): Promise<HoldoutSet | null> {
  return apiFetch<HoldoutSet | null>(`/projects/${projectId}/agent/holdout-set`);
}

/**
 * Get a specific row from the holdout set.
 */
export async function getHoldoutRow(projectId: string, rowIndex: number): Promise<HoldoutRow> {
  return apiFetch<HoldoutRow>(`/projects/${projectId}/agent/holdout-set/row/${rowIndex}`);
}

/**
 * Get all rows from the holdout set.
 */
export async function getAllHoldoutRows(projectId: string): Promise<HoldoutRowsResponse> {
  return apiFetch<HoldoutRowsResponse>(`/projects/${projectId}/agent/holdout-set/rows`);
}

/**
 * Update the target column for the holdout set.
 */
export async function updateHoldoutTargetColumn(
  projectId: string,
  targetColumn: string
): Promise<{ message: string; target_column: string }> {
  return apiFetch<{ message: string; target_column: string }>(
    `/projects/${projectId}/agent/holdout-set/target-column`,
    {
      method: 'PUT',
      body: JSON.stringify({ target_column: targetColumn }),
    }
  );
}

// ============================================================================
// Auto DS (Autonomous Data Science) API
// ============================================================================

import type {
  AutoDSSessionListResponse,
  AutoDSSession,
  AutoDSSessionCreate,
  AutoDSStartResponse,
  AutoDSSessionProgress,
  AutoDSIteration,
  ResearchInsightListResponse,
  GlobalInsightListResponse,
} from '../types/api';

/**
 * List all Auto DS sessions for a project.
 */
export async function listAutoDSSessions(projectId: string): Promise<AutoDSSessionListResponse> {
  return apiFetch<AutoDSSessionListResponse>(`/projects/${projectId}/auto-ds-sessions`);
}

/**
 * Create a new Auto DS session.
 */
export async function createAutoDSSession(
  projectId: string,
  data: AutoDSSessionCreate
): Promise<AutoDSSession> {
  return apiFetch<AutoDSSession>(`/projects/${projectId}/auto-ds-sessions`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/**
 * Get a specific Auto DS session.
 */
export async function getAutoDSSession(
  projectId: string,
  sessionId: string
): Promise<AutoDSSession> {
  return apiFetch<AutoDSSession>(`/projects/${projectId}/auto-ds-sessions/${sessionId}`);
}

/**
 * Start an Auto DS session.
 */
export async function startAutoDSSession(
  projectId: string,
  sessionId: string,
  datasetSpecIds?: string[]
): Promise<AutoDSStartResponse> {
  return apiFetch<AutoDSStartResponse>(
    `/projects/${projectId}/auto-ds-sessions/${sessionId}/start`,
    {
      method: 'POST',
      body: JSON.stringify({ dataset_spec_ids: datasetSpecIds }),
    }
  );
}

/**
 * Pause an Auto DS session.
 */
export async function pauseAutoDSSession(
  projectId: string,
  sessionId: string
): Promise<AutoDSSession> {
  return apiFetch<AutoDSSession>(
    `/projects/${projectId}/auto-ds-sessions/${sessionId}/pause`,
    {
      method: 'POST',
    }
  );
}

/**
 * Stop an Auto DS session.
 */
export async function stopAutoDSSession(
  projectId: string,
  sessionId: string
): Promise<AutoDSSession> {
  return apiFetch<AutoDSSession>(
    `/projects/${projectId}/auto-ds-sessions/${sessionId}/stop`,
    {
      method: 'POST',
    }
  );
}

/**
 * Get real-time progress of an Auto DS session.
 */
export async function getAutoDSSessionProgress(
  projectId: string,
  sessionId: string
): Promise<AutoDSSessionProgress> {
  return apiFetch<AutoDSSessionProgress>(
    `/projects/${projectId}/auto-ds-sessions/${sessionId}/progress`
  );
}

/**
 * List all iterations for an Auto DS session.
 */
export async function listSessionIterations(
  projectId: string,
  sessionId: string
): Promise<AutoDSIteration[]> {
  return apiFetch<AutoDSIteration[]>(
    `/projects/${projectId}/auto-ds-sessions/${sessionId}/iterations`
  );
}

/**
 * List insights discovered during an Auto DS session.
 */
export async function listSessionInsights(
  projectId: string,
  sessionId: string
): Promise<ResearchInsightListResponse> {
  return apiFetch<ResearchInsightListResponse>(
    `/projects/${projectId}/auto-ds-sessions/${sessionId}/insights`
  );
}

/**
 * Delete an Auto DS session.
 */
export async function deleteAutoDSSession(
  projectId: string,
  sessionId: string
): Promise<void> {
  return apiFetch<void>(
    `/projects/${projectId}/auto-ds-sessions/${sessionId}`,
    { method: 'DELETE' }
  );
}

/**
 * List global insights across all projects.
 */
export async function listGlobalInsights(): Promise<GlobalInsightListResponse> {
  return apiFetch<GlobalInsightListResponse>('/global-insights');
}

/**
 * Response from bulk delete experiments.
 */
export interface BulkDeleteResponse {
  deleted_count: number;
  failed_ids: string[];
  errors: string[];
}

/**
 * Delete multiple experiments at once.
 */
export async function bulkDeleteExperiments(
  experimentIds: string[]
): Promise<BulkDeleteResponse> {
  return apiFetch<BulkDeleteResponse>('/experiments/bulk-delete', {
    method: 'POST',
    body: JSON.stringify({ experiment_ids: experimentIds }),
  });
}

// ============================================
// Project-level Auto DS Functions
// ============================================

export interface ProjectAutoDSStartResponse {
  session_id: string;
  task_id: string;
  status: string;
  message: string;
}

export interface ProjectAutoDSStopResponse {
  session_id: string;
  status: string;
  message: string;
}

export interface ProjectAutoDSStatus {
  active: boolean;
  session_id?: string;
  name?: string;
  status?: string;
  current_iteration?: number;
  max_iterations?: number;
  total_experiments_run?: number;
  best_score?: number | null;
  started_at?: string | null;
  completed_at?: string | null;
  message?: string;
}

export interface ProjectAutoDSSessionSummary {
  id: string;
  name: string;
  status: string;
  current_iteration: number;
  max_iterations: number;
  total_experiments_run: number;
  best_score: number | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface ProjectAutoDSSessionsResponse {
  items: ProjectAutoDSSessionSummary[];
  total: number;
}

/**
 * Start an Auto DS session for a project.
 */
export async function startProjectAutoDS(
  projectId: string,
  datasetSpecIds?: string[]
): Promise<ProjectAutoDSStartResponse> {
  const params = new URLSearchParams();
  if (datasetSpecIds && datasetSpecIds.length > 0) {
    datasetSpecIds.forEach(id => params.append('dataset_spec_ids', id));
  }
  const queryString = params.toString();
  return apiFetch<ProjectAutoDSStartResponse>(
    `/projects/${projectId}/auto-ds/start${queryString ? `?${queryString}` : ''}`,
    { method: 'POST' }
  );
}

/**
 * Stop the current Auto DS session for a project.
 */
export async function stopProjectAutoDS(
  projectId: string
): Promise<ProjectAutoDSStopResponse> {
  return apiFetch<ProjectAutoDSStopResponse>(
    `/projects/${projectId}/auto-ds/stop`,
    { method: 'POST' }
  );
}

/**
 * Get the Auto DS status for a project.
 */
export async function getProjectAutoDSStatus(
  projectId: string
): Promise<ProjectAutoDSStatus> {
  return apiFetch<ProjectAutoDSStatus>(`/projects/${projectId}/auto-ds/status`);
}

/**
 * List all Auto DS sessions for a project.
 */
export async function listProjectAutoDSSessions(
  projectId: string,
  skip = 0,
  limit = 20
): Promise<ProjectAutoDSSessionsResponse> {
  return apiFetch<ProjectAutoDSSessionsResponse>(
    `/projects/${projectId}/auto-ds/sessions?skip=${skip}&limit=${limit}`
  );
}

// =============================================================================
// Context Documents API
// =============================================================================

/**
 * Get supported file extensions for context documents.
 */
export async function getSupportedExtensions(): Promise<SupportedExtensionsResponse> {
  return apiFetch<SupportedExtensionsResponse>('/context-documents/supported-extensions');
}

/**
 * Upload a context document for a project.
 */
export async function uploadContextDocument(
  projectId: string,
  file: File,
  name: string,
  explanation: string
): Promise<ContextDocument> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', name);
  formData.append('explanation', explanation);

  const token = getAuthToken();
  const headers: HeadersInit = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(
    `${API_BASE_URL}/projects/${projectId}/context-documents/upload`,
    {
      method: 'POST',
      headers,
      body: formData,
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new ApiException(response.status, error.detail || 'Upload failed');
  }

  return response.json();
}

/**
 * List context documents for a project.
 */
export async function listContextDocuments(
  projectId: string,
  includeInactive = false
): Promise<ContextDocumentListResponse> {
  return apiFetch<ContextDocumentListResponse>(
    `/projects/${projectId}/context-documents?include_inactive=${includeInactive}`
  );
}

/**
 * Get a single context document.
 */
export async function getContextDocument(
  documentId: string,
  includeContent = false
): Promise<ContextDocumentDetail> {
  return apiFetch<ContextDocumentDetail>(
    `/context-documents/${documentId}?include_content=${includeContent}`
  );
}

/**
 * Update a context document.
 */
export async function updateContextDocument(
  documentId: string,
  data: ContextDocumentUpdate
): Promise<ContextDocument> {
  return apiFetch<ContextDocument>(`/context-documents/${documentId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

/**
 * Delete a context document.
 */
export async function deleteContextDocument(documentId: string): Promise<void> {
  await apiFetch<void>(`/context-documents/${documentId}`, {
    method: 'DELETE',
  });
}

/**
 * Re-extract text from a context document.
 */
export async function reextractContextDocument(
  documentId: string
): Promise<ContextDocument> {
  return apiFetch<ContextDocument>(`/context-documents/${documentId}/reextract`, {
    method: 'POST',
  });
}

/**
 * Toggle a context document's active status.
 */
export async function toggleContextDocumentActive(
  documentId: string
): Promise<ContextDocument> {
  return apiFetch<ContextDocument>(`/context-documents/${documentId}/toggle-active`, {
    method: 'POST',
  });
}

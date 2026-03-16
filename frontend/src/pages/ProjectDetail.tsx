import { useState, useEffect, useCallback } from 'react';
import { useParams, Link, useSearchParams } from 'react-router-dom';
import type {
  Project,
  DataSource,
  DatasetSpec,
  DatasetSpecCreate,
  Experiment,
  ExperimentCreate,
  ModelVersion,
  OrchestrationOptions,
  ContextDocument,
} from '../types/api';
import {
  getProject,
  listDataSources,
  listDatasetSpecs,
  listExperiments,
  listModelVersions,
  uploadDataSource,
  createDatasetSpec,
  createExperiment,
  runExperiment,
  runExperimentsBatch,
  deleteDataSource,
  deleteDatasetSpec,
  deleteExperiment,
  runDatasetDiscovery,
  getAgentRun,
  applyDiscoveredDatasets,
  getOrchestrationOptions,
  listContextDocuments,
  downloadDatasetSpec,
  ApiException,
} from '../services/api';
import ProjectSettingsModal from '../components/ProjectSettingsModal';
import type { DiscoveredDataset } from '../types/api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import Modal from '../components/Modal';
import ConfirmDialog from '../components/ConfirmDialog';
import StatusBadge from '../components/StatusBadge';
import FileUpload from '../components/FileUpload';
import DataSourceCard from '../components/DataSourceCard';
import DataSourceDetailModal from '../components/DataSourceDetailModal';
import DatasetSpecCard from '../components/DatasetSpecCard';
import DatasetSpecDetailModal from '../components/DatasetSpecDetailModal';
import CreateDatasetSpecForm from '../components/CreateDatasetSpecForm';
import CreateExperimentForm from '../components/CreateExperimentForm';
import ChatBot from '../components/ChatBot';
import AgentPipelineTimeline from '../components/AgentPipelineTimeline';
import DataArchitectPipeline from '../components/DataArchitectPipeline';
import DiscoveredDatasetsList from '../components/DiscoveredDatasetsList';
import type { AgentRun } from '../components/AgentPipelineTimeline';
import VisualizeData from '../components/VisualizeData';
import type { VisualizationItem } from '../components/VisualizeData';
import ResearchNotebook from '../components/ResearchNotebook';
import ProjectAutoDS from '../components/ProjectAutoDS';
import ContextDocuments from '../components/ContextDocuments';

type TabType = 'data' | 'datasets' | 'experiments' | 'models' | 'visualize' | 'notebook' | 'auto-ds';

export default function ProjectDetail() {
  const { projectId } = useParams<{ projectId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  // State
  const [project, setProject] = useState<Project | null>(null);
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [datasetSpecs, setDatasetSpecs] = useState<DatasetSpec[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [contextDocs, setContextDocs] = useState<ContextDocument[]>([]);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize active tab from URL query param or default to 'data'
  const tabFromUrl = searchParams.get('tab') as TabType | null;
  const validTabs: TabType[] = ['data', 'datasets', 'experiments', 'models', 'visualize', 'notebook', 'auto-ds'];
  const initialTab = tabFromUrl && validTabs.includes(tabFromUrl) ? tabFromUrl : 'data';
  const [activeTab, setActiveTab] = useState<TabType>(initialTab);

  // Handle tab changes - update URL when tab changes
  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab);
    if (tab === 'data') {
      // Remove tab param for default tab
      searchParams.delete('tab');
    } else {
      searchParams.set('tab', tab);
    }
    setSearchParams(searchParams, { replace: true });
  };

  // Modal states
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false);
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);
  const [isExperimentModalOpen, setIsExperimentModalOpen] = useState(false);
  const [isCreatingExperiment, setIsCreatingExperiment] = useState(false);

  // Data source detail modal state
  const [selectedDataSource, setSelectedDataSource] = useState<DataSource | null>(null);

  // Dataset spec detail modal state
  const [selectedDatasetSpec, setSelectedDatasetSpec] = useState<DatasetSpec | null>(null);

  // Delete confirmation states
  const [deleteDataSourceTarget, setDeleteDataSourceTarget] = useState<DataSource | null>(null);
  const [deleteDatasetTarget, setDeleteDatasetTarget] = useState<DatasetSpec | null>(null);
  const [deleteExperimentTarget, setDeleteExperimentTarget] = useState<Experiment | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Batch experiment selection states
  const [selectedExperiments, setSelectedExperiments] = useState<Set<string>>(new Set());
  const [isRunningBatch, setIsRunningBatch] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchSuccess, setBatchSuccess] = useState<string | null>(null);

  // Visualizations state for chat context
  const [visualizations, setVisualizations] = useState<VisualizationItem[]>([]);

  // Agent pipeline run state for chat context
  const [agentRun, setAgentRun] = useState<AgentRun | null>(null);

  // AI Dataset Search state
  const [isAISearchModalOpen, setIsAISearchModalOpen] = useState(false);
  const [aiSearchQuery, setAISearchQuery] = useState('');
  const [isSearchingDatasets, setIsSearchingDatasets] = useState(false);
  const [discoveredDatasets, setDiscoveredDatasets] = useState<DiscoveredDataset[]>([]);
  const [discoveryRunId, setDiscoveryRunId] = useState<string | null>(null);
  const [isApplyingDatasets, setIsApplyingDatasets] = useState(false);
  const [aiSearchError, setAISearchError] = useState<string | null>(null);

  // Project settings modal state
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);

  // Context documents modal state
  const [isContextDocsModalOpen, setIsContextDocsModalOpen] = useState(false);

  // Orchestration settings state (visible in header)
  const [orchestrationOptions, setOrchestrationOptions] = useState<OrchestrationOptions | null>(null);
  const [usePMMode, setUsePMMode] = useState<boolean>(() => {
    // Load from localStorage if available
    const saved = localStorage.getItem(`orchestration_pm_mode_${projectId}`);
    return saved === 'true';
  });
  const [useDebateMode, setUseDebateMode] = useState<boolean>(() => {
    const saved = localStorage.getItem(`orchestration_debate_mode_${projectId}`);
    return saved === 'true';
  });

  // Save orchestration settings to localStorage when they change
  useEffect(() => {
    if (projectId) {
      localStorage.setItem(`orchestration_pm_mode_${projectId}`, usePMMode.toString());
      localStorage.setItem(`orchestration_debate_mode_${projectId}`, useDebateMode.toString());
    }
  }, [projectId, usePMMode, useDebateMode]);

  // Fetch orchestration options on mount
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const options = await getOrchestrationOptions();
        setOrchestrationOptions(options);
      } catch (err) {
        console.warn('Failed to fetch orchestration options:', err);
      }
    };
    fetchOptions();
  }, []);

  const fetchProjectData = useCallback(async () => {
    if (!projectId) return;

    setIsLoading(true);
    setError(null);

    try {
      const [projectData, dataSourcesData, datasetSpecsData, experimentsData, modelsData, contextDocsData] =
        await Promise.all([
          getProject(projectId),
          listDataSources(projectId),
          listDatasetSpecs(projectId),
          listExperiments(projectId),
          listModelVersions(projectId),
          listContextDocuments(projectId, false), // Only active docs
        ]);

      setProject(projectData);
      setDataSources(dataSourcesData);
      setDatasetSpecs(datasetSpecsData);
      setExperiments(experimentsData);
      setModels(modelsData);
      setContextDocs(contextDocsData.documents);
    } catch (err) {
      if (err instanceof ApiException) {
        if (err.status === 404) {
          setError('Project not found');
        } else {
          setError(err.detail);
        }
      } else {
        setError('Failed to load project data');
      }
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchProjectData();
  }, [fetchProjectData]);

  // Sync tab state with URL when navigating to the page
  useEffect(() => {
    const tabParam = searchParams.get('tab') as TabType | null;
    if (tabParam && validTabs.includes(tabParam) && tabParam !== activeTab) {
      setActiveTab(tabParam);
    }
  }, [searchParams, activeTab]);

  // Refresh experiments periodically if any are running
  useEffect(() => {
    const hasRunning = experiments.some((e) => e.status === 'running' || e.status === 'pending');
    if (!hasRunning) return;

    const interval = setInterval(async () => {
      if (!projectId) return;
      try {
        const updatedExperiments = await listExperiments(projectId);
        setExperiments(updatedExperiments);
        // Also refresh models in case experiment completed
        const updatedModels = await listModelVersions(projectId);
        setModels(updatedModels);
      } catch {
        // Ignore errors during polling
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [projectId, experiments]);

  // Handlers
  const handleFileUpload = async (file: File, options?: { name?: string; delimiter?: string }) => {
    if (!projectId) return;
    setIsUploading(true);
    setError(null);
    try {
      const newDataSource = await uploadDataSource(projectId, file, options);
      setDataSources((prev) => [newDataSource, ...prev]);
      setIsUploadModalOpen(false);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to upload file');
      }
      // Keep modal open so user can see the error or try again
    } finally {
      setIsUploading(false);
    }
  };

  const handleCreateDataset = async (data: DatasetSpecCreate) => {
    if (!projectId) return;
    setIsCreatingDataset(true);
    try {
      const newDataset = await createDatasetSpec(projectId, data);
      setDatasetSpecs((prev) => [newDataset, ...prev]);
      setIsDatasetModalOpen(false);
    } finally {
      setIsCreatingDataset(false);
    }
  };

  const handleCreateExperiment = async (data: ExperimentCreate) => {
    if (!projectId) return;
    setIsCreatingExperiment(true);
    try {
      const newExperiment = await createExperiment(projectId, data);
      setExperiments((prev) => [newExperiment, ...prev]);
      setIsExperimentModalOpen(false);
    } finally {
      setIsCreatingExperiment(false);
    }
  };

  const handleRunExperiment = async (experimentId: string) => {
    try {
      await runExperiment(experimentId);
      // Refresh experiments to show new status
      if (projectId) {
        const updatedExperiments = await listExperiments(projectId);
        setExperiments(updatedExperiments);
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    }
  };

  // Toggle experiment selection
  const handleToggleExperiment = (experimentId: string) => {
    setSelectedExperiments(prev => {
      const newSet = new Set(prev);
      if (newSet.has(experimentId)) {
        newSet.delete(experimentId);
      } else {
        newSet.add(experimentId);
      }
      return newSet;
    });
  };

  // Select/deselect all runnable experiments
  const handleSelectAllExperiments = () => {
    const runnableExperiments = experiments.filter(
      exp => exp.status === 'pending' || exp.status === 'failed'
    );
    if (selectedExperiments.size === runnableExperiments.length && runnableExperiments.length > 0) {
      setSelectedExperiments(new Set());
    } else {
      setSelectedExperiments(new Set(runnableExperiments.map(e => e.id)));
    }
  };

  // Run selected experiments in parallel
  const handleRunSelectedExperiments = async () => {
    if (selectedExperiments.size === 0) return;

    setIsRunningBatch(true);
    setBatchError(null);
    setBatchSuccess(null);

    try {
      const experimentIds = Array.from(selectedExperiments);
      const response = await runExperimentsBatch(experimentIds);

      if (response.failed_count > 0) {
        const failedMsg = response.experiments
          .filter(e => e.status === 'error')
          .map(e => e.message)
          .join('; ');
        setBatchError(`Some experiments failed to queue: ${failedMsg}`);
      }

      if (response.queued_count > 0) {
        setBatchSuccess(`Queued ${response.queued_count} experiment(s) for parallel execution`);
        setSelectedExperiments(new Set());
      }

      // Refresh experiments to show new status
      if (projectId) {
        const updatedExperiments = await listExperiments(projectId);
        setExperiments(updatedExperiments);
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setBatchError(err.detail);
      } else {
        setBatchError('Failed to run experiments');
      }
    } finally {
      setIsRunningBatch(false);
    }
  };

  const handleDeleteDataSource = async () => {
    if (!deleteDataSourceTarget) return;
    setIsDeleting(true);
    try {
      await deleteDataSource(deleteDataSourceTarget.id);
      setDataSources((prev) => prev.filter((ds) => ds.id !== deleteDataSourceTarget.id));
      setDeleteDataSourceTarget(null);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDeleteDataset = async () => {
    if (!deleteDatasetTarget) return;
    setIsDeleting(true);
    try {
      await deleteDatasetSpec(deleteDatasetTarget.id);
      setDatasetSpecs((prev) => prev.filter((ds) => ds.id !== deleteDatasetTarget.id));
      setDeleteDatasetTarget(null);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setIsDeleting(false);
    }
  };

  const handleDownloadDataset = async (datasetSpec: DatasetSpec) => {
    try {
      await downloadDatasetSpec(datasetSpec.id);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(`Download failed: ${err.detail}`);
      } else {
        setError('Download failed. Please try again.');
      }
    }
  };

  const handleDeleteExperiment = async () => {
    if (!deleteExperimentTarget) return;
    setIsDeleting(true);
    try {
      await deleteExperiment(deleteExperimentTarget.id);
      setExperiments((prev) => prev.filter((e) => e.id !== deleteExperimentTarget.id));
      setDeleteExperimentTarget(null);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setIsDeleting(false);
    }
  };

  // AI Dataset Search handlers
  const handleAISearchDatasets = async () => {
    if (!projectId || !aiSearchQuery.trim()) return;

    setIsSearchingDatasets(true);
    setAISearchError(null);
    setDiscoveredDatasets([]);

    try {
      const response = await runDatasetDiscovery(projectId, {
        project_description: aiSearchQuery,
      });
      setDiscoveryRunId(response.run_id);

      // Fetch the discovered datasets from the completed run
      const runDetails = await getAgentRun(response.run_id);
      const datasets = (runDetails.result_json as { discovered_datasets?: DiscoveredDataset[] })?.discovered_datasets || [];
      setDiscoveredDatasets(datasets);
    } catch (err) {
      if (err instanceof ApiException) {
        setAISearchError(err.detail);
      } else {
        setAISearchError('Failed to search for datasets');
      }
    } finally {
      setIsSearchingDatasets(false);
    }
  };

  const handleApplyDiscoveredDatasets = async (selectedIndices: number[]) => {
    if (!projectId || !discoveryRunId || selectedIndices.length === 0) return;

    setIsApplyingDatasets(true);
    setAISearchError(null);

    try {
      await applyDiscoveredDatasets(projectId, discoveryRunId, {
        dataset_indices: selectedIndices,
      });
      setIsAISearchModalOpen(false);
      setDiscoveredDatasets([]);
      setDiscoveryRunId(null);
      setAISearchQuery('');
      fetchProjectData(); // Refresh to show new data sources
    } catch (err) {
      if (err instanceof ApiException) {
        setAISearchError(err.detail);
      } else {
        setAISearchError('Failed to download datasets');
      }
    } finally {
      setIsApplyingDatasets(false);
    }
  };

  const handleCloseAISearchModal = () => {
    setIsAISearchModalOpen(false);
    setDiscoveredDatasets([]);
    setDiscoveryRunId(null);
    setAISearchQuery('');
    setAISearchError(null);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  // Infer model type from model name or metrics if not explicitly set
  const getModelType = (model: ModelVersion): string => {
    if (model.model_type) return model.model_type;

    // Try to infer from model name
    const name = model.name?.toLowerCase() || '';
    if (name.includes('lightgbm') || name.includes('lgbm')) return 'LightGBM';
    if (name.includes('xgboost') || name.includes('xgb')) return 'XGBoost';
    if (name.includes('catboost')) return 'CatBoost';
    if (name.includes('random_forest') || name.includes('randomforest')) return 'Random Forest';
    if (name.includes('gradient_boost') || name.includes('gbm')) return 'Gradient Boosting';
    if (name.includes('logistic')) return 'Logistic Regression';
    if (name.includes('linear')) return 'Linear Model';
    if (name.includes('neural') || name.includes('nn') || name.includes('mlp')) return 'Neural Network';
    if (name.includes('ridge')) return 'Ridge Regression';
    if (name.includes('lasso')) return 'Lasso Regression';

    // Try to infer task type from metrics
    const metrics = model.metrics_json || {};
    const metricKeys = Object.keys(metrics).map(k => k.toLowerCase());
    if (metricKeys.some(k => k.includes('rmse') || k.includes('mae') || k === 'r2')) {
      return 'Regressor';
    }
    if (metricKeys.some(k => k.includes('roc_auc') || k === 'accuracy' || k === 'f1')) {
      return 'Classifier';
    }

    return 'AutoML Model';
  };

  if (isLoading) {
    return (
      <div className="project-detail-page">
        <LoadingSpinner message="Loading project..." />
      </div>
    );
  }

  if (error && !project) {
    return (
      <div className="project-detail-page">
        <ErrorMessage
          message={error}
          onRetry={error === 'Project not found' ? undefined : fetchProjectData}
        />
        <Link to="/projects" className="btn btn-secondary" style={{ marginTop: '1rem' }}>
          Back to Projects
        </Link>
      </div>
    );
  }

  if (!project) {
    return null;
  }

  const taskTypeLabel = project.task_type || 'Not set';

  return (
    <div className="project-detail-page">
      {error && (
        <div className="form-error" style={{ marginBottom: '1rem' }}>
          {error}
          <button onClick={() => setError(null)} style={{ marginLeft: '1rem' }}>
            Dismiss
          </button>
        </div>
      )}

      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="breadcrumb">
            <Link to="/projects">Projects</Link>
            <span>/</span>
            <span>{project.name}</span>
          </div>
          <h2>{project.name}</h2>
          <div className="project-meta">
            <StatusBadge status={project.status} />
            <span className="meta-separator">|</span>
            <span>Task: {taskTypeLabel}</span>
            <span className="meta-separator">|</span>
            <span>Created: {formatDate(project.created_at)}</span>
            {orchestrationOptions && (
              <>
                <span className="meta-separator">|</span>
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '0.9rem' }} title="Project Manager dynamically orchestrates pipeline flow">
                  <input
                    type="checkbox"
                    checked={usePMMode}
                    onChange={(e) => setUsePMMode(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  <span style={{ color: usePMMode ? 'var(--accent-color, #6366f1)' : 'inherit' }}>🎭 PM Mode</span>
                </label>
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '0.9rem', marginLeft: '8px' }} title="Enable Gemini critique with OpenAI judge">
                  <input
                    type="checkbox"
                    checked={useDebateMode}
                    onChange={(e) => setUseDebateMode(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  <span style={{ color: useDebateMode ? 'var(--accent-color, #6366f1)' : 'inherit' }}>💬 Debate</span>
                </label>
              </>
            )}
          </div>
          {project.description && (
            <p className="project-description">{project.description}</p>
          )}
        </div>
        <button
          className="btn btn-secondary"
          onClick={() => setIsSettingsModalOpen(true)}
          title="Project Settings"
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <span style={{ fontSize: '1rem' }}>&#9881;</span>
          Settings
        </button>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'data' ? 'active' : ''}`}
          onClick={() => handleTabChange('data')}
        >
          Data Sources ({dataSources.length})
        </button>
        <button
          className={`tab ${activeTab === 'datasets' ? 'active' : ''}`}
          onClick={() => handleTabChange('datasets')}
        >
          Datasets ({datasetSpecs.length})
        </button>
        <button
          className={`tab ${activeTab === 'experiments' ? 'active' : ''}`}
          onClick={() => handleTabChange('experiments')}
        >
          Experiments ({experiments.length})
        </button>
        <button
          className={`tab ${activeTab === 'models' ? 'active' : ''}`}
          onClick={() => handleTabChange('models')}
        >
          Models ({models.length})
        </button>
        <button
          className={`tab ${activeTab === 'visualize' ? 'active' : ''}`}
          onClick={() => handleTabChange('visualize')}
        >
          Visualize
        </button>
        <button
          className={`tab ${activeTab === 'notebook' ? 'active' : ''}`}
          onClick={() => handleTabChange('notebook')}
        >
          Research Notebook
        </button>
        <button
          className={`tab ${activeTab === 'auto-ds' ? 'active' : ''}`}
          onClick={() => handleTabChange('auto-ds')}
        >
          Auto DS {project.active_auto_ds_session_id && <span className="tab-badge running">Active</span>}
        </button>
      </div>

      <div className="tab-content">
        {/* Data Sources Tab */}
        {activeTab === 'data' && (
          <div className="tab-panel">
            {/* Data Architect Pipeline - Combine Multiple Tables (only show with 2+ data sources) */}
            {projectId && dataSources.length >= 2 && (
              <DataArchitectPipeline
                projectId={projectId}
                dataSources={dataSources}
                onPipelineComplete={fetchProjectData}
              />
            )}

            {/* AI Setup Pipeline - Analyze a Single Dataset (shown after Combine Tables so users understand the flow) */}
            {projectId && (
              <AgentPipelineTimeline
                projectId={projectId}
                dataSources={dataSources}
                onPipelineComplete={fetchProjectData}
                onAgentRunChange={setAgentRun}
                defaultPMMode={usePMMode}
                defaultDebateMode={useDebateMode}
              />
            )}

            {/* Context Documents Inline Display */}
            {contextDocs.length > 0 && (
              <div className="context-docs-summary card" style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: '#f0f9ff', border: '1px solid #bae6fd' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <h4 style={{ margin: 0, fontSize: '0.95rem', color: '#0369a1' }}>
                    📚 Context Documents ({contextDocs.length} active)
                  </h4>
                  <button
                    className="btn btn-sm btn-secondary"
                    onClick={() => setIsContextDocsModalOpen(true)}
                    style={{ fontSize: '0.75rem' }}
                  >
                    Manage
                  </button>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {contextDocs.map(doc => (
                    <div
                      key={doc.id}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.35rem',
                        padding: '0.35rem 0.65rem',
                        backgroundColor: 'white',
                        border: '1px solid #e0e7ef',
                        borderRadius: '4px',
                        fontSize: '0.8rem',
                      }}
                      title={doc.explanation}
                    >
                      <span>{doc.file_type === 'pdf' ? '📄' : doc.file_type === 'docx' ? '📝' : doc.file_type === 'excel' ? '📊' : doc.file_type === 'csv' ? '📈' : doc.file_type === 'image' ? '🖼️' : '📁'}</span>
                      <span style={{ fontWeight: 500 }}>{doc.name}</span>
                      <span style={{
                        fontSize: '0.65rem',
                        padding: '0.1rem 0.3rem',
                        backgroundColor: doc.extraction_status === 'completed' && doc.has_content ? '#dcfce7' : doc.extraction_status === 'pending' ? '#fef9c3' : '#fee2e2',
                        color: doc.extraction_status === 'completed' && doc.has_content ? '#166534' : doc.extraction_status === 'pending' ? '#854d0e' : '#991b1b',
                        borderRadius: '3px',
                      }}>
                        {doc.extraction_status === 'completed' && doc.has_content ? 'Ready' : doc.extraction_status === 'pending' ? 'Processing' : 'No Content'}
                      </span>
                    </div>
                  ))}
                </div>
                <p style={{ margin: '0.5rem 0 0', fontSize: '0.75rem', color: '#64748b' }}>
                  These documents will be used to help AI understand your problem when running pipelines.
                </p>
              </div>
            )}

            <div className="tab-header">
              <h3>Data Sources</h3>
              <div className="tab-header-actions">
                <button
                  className="btn btn-secondary"
                  onClick={() => setIsContextDocsModalOpen(true)}
                  title="Upload context documents to help AI understand your problem"
                >
                  Context Docs
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => setIsAISearchModalOpen(true)}
                >
                  Search with AI
                </button>
                <button
                  className="btn btn-primary"
                  onClick={() => setIsUploadModalOpen(true)}
                >
                  + Upload File
                </button>
              </div>
            </div>

            {dataSources.length === 0 ? (
              <EmptyState
                title="No data sources yet"
                description="Upload a CSV, Excel, or other data file to get started"
                actionLabel="Upload File"
                onAction={() => setIsUploadModalOpen(true)}
                icon="📊"
              />
            ) : (
              <div className="data-sources-grid">
                {dataSources.map((ds) => (
                  <DataSourceCard
                    key={ds.id}
                    dataSource={ds}
                    onDelete={setDeleteDataSourceTarget}
                    onClick={setSelectedDataSource}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Datasets Tab */}
        {activeTab === 'datasets' && (
          <div className="tab-panel">
            <div className="tab-header">
              <h3>Datasets</h3>
              <button
                className="btn btn-primary"
                onClick={() => setIsDatasetModalOpen(true)}
                disabled={dataSources.length === 0}
              >
                + Create Dataset
              </button>
            </div>

            {datasetSpecs.length === 0 ? (
              <EmptyState
                title="No datasets yet"
                description={
                  dataSources.length === 0
                    ? 'Upload data first, then create a dataset specification'
                    : 'Define which columns to use for training'
                }
                actionLabel={dataSources.length > 0 ? 'Create Dataset' : undefined}
                onAction={dataSources.length > 0 ? () => setIsDatasetModalOpen(true) : undefined}
                icon="📋"
              />
            ) : (
              <div className="data-sources-grid">
                {datasetSpecs.map((ds) => (
                  <DatasetSpecCard
                    key={ds.id}
                    datasetSpec={ds}
                    dataSources={dataSources}
                    onDelete={setDeleteDatasetTarget}
                    onClick={setSelectedDatasetSpec}
                    onDownload={handleDownloadDataset}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Experiments Tab */}
        {activeTab === 'experiments' && (
          <div className="tab-panel">
            <div className="tab-header">
              <h3>Experiments</h3>
              <div className="tab-header-actions">
                {experiments.filter(e => e.status === 'pending' || e.status === 'failed').length > 0 && (
                  <>
                    <button
                      className="btn btn-secondary btn-small"
                      onClick={handleSelectAllExperiments}
                    >
                      {selectedExperiments.size === experiments.filter(e => e.status === 'pending' || e.status === 'failed').length
                        ? 'Deselect All'
                        : 'Select Runnable'}
                    </button>
                    {selectedExperiments.size > 0 && (
                      <button
                        className="btn btn-primary"
                        onClick={handleRunSelectedExperiments}
                        disabled={isRunningBatch}
                      >
                        {isRunningBatch ? (
                          <>
                            <span className="spinner spinner-small"></span>
                            Running...
                          </>
                        ) : (
                          `Run ${selectedExperiments.size} Selected in Parallel`
                        )}
                      </button>
                    )}
                  </>
                )}
                <button
                  className="btn btn-primary"
                  onClick={() => setIsExperimentModalOpen(true)}
                  disabled={datasetSpecs.length === 0}
                >
                  + New Experiment
                </button>
              </div>
            </div>

            {/* Batch operation messages */}
            {batchError && (
              <div className="batch-error">
                {batchError}
                <button onClick={() => setBatchError(null)} className="btn-dismiss">
                  Dismiss
                </button>
              </div>
            )}
            {batchSuccess && (
              <div className="batch-success">
                {batchSuccess}
                <button onClick={() => setBatchSuccess(null)} className="btn-dismiss">
                  Dismiss
                </button>
              </div>
            )}

            {experiments.length === 0 ? (
              <EmptyState
                title="No experiments yet"
                description={
                  datasetSpecs.length === 0
                    ? 'Create a dataset first, then run experiments'
                    : 'Train ML models with AutoML'
                }
                actionLabel={datasetSpecs.length > 0 ? 'New Experiment' : undefined}
                onAction={
                  datasetSpecs.length > 0 ? () => setIsExperimentModalOpen(true) : undefined
                }
                icon="🧪"
              />
            ) : (
              <div className="experiments-list">
                {experiments.map((exp) => {
                  const isRunnable = exp.status === 'pending' || exp.status === 'failed';
                  const isSelected = selectedExperiments.has(exp.id);

                  return (
                    <div
                      key={exp.id}
                      className={`experiment-card ${isSelected ? 'selected' : ''}`}
                    >
                      <div className="experiment-header">
                        {isRunnable && (
                          <label className="experiment-checkbox">
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => handleToggleExperiment(exp.id)}
                            />
                          </label>
                        )}
                        <Link to={`/experiments/${exp.id}`} className="experiment-name">
                          {exp.name}
                        </Link>
                        <StatusBadge status={exp.status} />
                      </div>
                      {exp.description && (
                        <p className="experiment-description">{exp.description}</p>
                      )}
                      <div className="experiment-meta">
                        <span>Metric: {exp.primary_metric || 'Auto'}</span>
                        <span className="meta-separator">|</span>
                        <span>Created: {formatDate(exp.created_at)}</span>
                      </div>
                      <div className="experiment-actions">
                        {exp.status === 'pending' && (
                          <button
                            className="btn btn-primary btn-small"
                            onClick={() => handleRunExperiment(exp.id)}
                          >
                            Run Experiment
                          </button>
                        )}
                        {exp.status === 'failed' && (
                          <>
                            <button
                              className="btn btn-primary btn-small"
                              onClick={() => handleRunExperiment(exp.id)}
                            >
                              Retry Experiment
                            </button>
                            {exp.error_message && (
                              <span
                                className="error-snippet"
                                title={exp.error_message}
                                style={{
                                  color: '#dc2626',
                                  fontSize: '0.75rem',
                                  marginLeft: '0.5rem',
                                  maxWidth: '200px',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap',
                                  display: 'inline-block',
                                  verticalAlign: 'middle',
                                }}
                              >
                                {exp.error_message.substring(0, 50)}...
                              </span>
                            )}
                          </>
                        )}
                        {exp.status === 'running' && (
                          <span className="running-indicator">
                            <span className="spinner spinner-small"></span>
                            Training...
                          </span>
                        )}
                        <Link
                          to={`/experiments/${exp.id}`}
                          className="btn btn-secondary btn-small"
                        >
                          View Details
                        </Link>
                        <button
                          className="btn btn-danger btn-small"
                          onClick={() => setDeleteExperimentTarget(exp)}
                          disabled={exp.status === 'running'}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="tab-panel">
            <div className="tab-header">
              <h3>Models</h3>
            </div>

            {models.length === 0 ? (
              <EmptyState
                title="No models yet"
                description="Models are created when experiments complete successfully"
                icon="🤖"
              />
            ) : (
              <div className="models-list">
                {models.map((model) => (
                  <div key={model.id} className="model-card">
                    <div className="model-header">
                      <Link to={`/models/${model.id}`} className="model-name">
                        {model.name}
                      </Link>
                      <StatusBadge status={model.status} />
                    </div>
                    <div className="model-meta">
                      <div className="meta-row">
                        <span className="meta-label">Type:</span>
                        <span className="meta-value">{getModelType(model)}</span>
                      </div>
                      <div className="meta-row">
                        <span className="meta-label">Created:</span>
                        <span className="meta-value">{formatDate(model.created_at)}</span>
                      </div>
                    </div>
                    {model.metrics_json && (
                      <div className="model-metrics">
                        {Object.entries(model.metrics_json)
                          .filter(([key]) => !key.includes('time') && !key.includes('num_'))
                          .slice(0, 3)
                          .map(([key, value]) => {
                            // Error metrics should always be displayed as positive values
                            const isErrorMetric = key.toLowerCase().includes('error') ||
                              key.toLowerCase().includes('rmse') ||
                              key.toLowerCase().includes('mse') ||
                              key.toLowerCase().includes('mae') ||
                              key.toLowerCase().includes('loss');
                            const displayValue = typeof value === 'number'
                              ? (isErrorMetric ? Math.abs(value) : value)
                              : value;
                            const displayKey = key.replace(/^neg_/, '');
                            return (
                              <div key={key} className="metric-item">
                                <span className="metric-name">{displayKey}:</span>
                                <span className="metric-value">
                                  {typeof displayValue === 'number' ? displayValue.toFixed(4) : String(displayValue)}
                                </span>
                              </div>
                            );
                          })}
                      </div>
                    )}
                    <div className="model-actions">
                      <Link
                        to={`/models/${model.id}`}
                        className="btn btn-secondary btn-small"
                      >
                        View Details
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Visualize Tab */}
        {activeTab === 'visualize' && (
          <div className="tab-panel">
            {dataSources.length === 0 ? (
              <EmptyState
                title="No data to visualize"
                description="Upload a data file first to create visualizations"
                actionLabel="Upload File"
                onAction={() => {
                  handleTabChange('data');
                  setIsUploadModalOpen(true);
                }}
                icon="📊"
              />
            ) : (
              projectId && (
                <VisualizeData
                  projectId={projectId}
                  dataSources={dataSources}
                  onVisualizationsChange={setVisualizations}
                />
              )
            )}
          </div>
        )}

        {/* Research Notebook Tab */}
        {activeTab === 'notebook' && projectId && (
          <div className="tab-panel">
            <ResearchNotebook projectId={projectId} />
          </div>
        )}

        {/* Auto DS Tab */}
        {activeTab === 'auto-ds' && projectId && (
          <div className="tab-panel">
            <ProjectAutoDS
              projectId={projectId}
              project={project}
              datasetSpecs={datasetSpecs}
              onProjectUpdate={(updatedProject) => setProject(updatedProject)}
              onExperimentsRefresh={async () => {
                const updatedExperiments = await listExperiments(projectId);
                setExperiments(updatedExperiments);
              }}
            />
          </div>
        )}
      </div>

      {/* Modals */}
      <Modal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        title="Upload Data File"
        size="medium"
      >
        <FileUpload onUpload={handleFileUpload} isLoading={isUploading} />
      </Modal>

      <Modal
        isOpen={isDatasetModalOpen}
        onClose={() => setIsDatasetModalOpen(false)}
        title="Create Dataset"
        size="large"
      >
        <CreateDatasetSpecForm
          dataSources={dataSources}
          onSubmit={handleCreateDataset}
          onCancel={() => setIsDatasetModalOpen(false)}
          isLoading={isCreatingDataset}
        />
      </Modal>

      <Modal
        isOpen={isExperimentModalOpen}
        onClose={() => setIsExperimentModalOpen(false)}
        title="Create Experiment"
        size="medium"
      >
        <CreateExperimentForm
          datasetSpecs={datasetSpecs}
          projectTaskType={project.task_type}
          onSubmit={handleCreateExperiment}
          onCancel={() => setIsExperimentModalOpen(false)}
          isLoading={isCreatingExperiment}
        />
      </Modal>

      {/* AI Dataset Search Modal */}
      <Modal
        isOpen={isAISearchModalOpen}
        onClose={handleCloseAISearchModal}
        title="Search for Datasets with AI"
        size="large"
      >
        <div className="ai-search-modal">
          {/* Search Form */}
          {discoveredDatasets.length === 0 && !isSearchingDatasets && (
            <div className="ai-search-form">
              <p className="ai-search-description">
                Describe what kind of data you're looking for, and AI will search for relevant public datasets.
              </p>
              <div className="form-group">
                <label htmlFor="ai-search-query">What data are you looking for?</label>
                <textarea
                  id="ai-search-query"
                  className="form-control"
                  placeholder="e.g., Customer churn data with demographics and usage patterns for a classification model"
                  value={aiSearchQuery}
                  onChange={(e) => setAISearchQuery(e.target.value)}
                  rows={3}
                />
              </div>
              {aiSearchError && (
                <div className="form-error">{aiSearchError}</div>
              )}
              <div className="form-actions">
                <button
                  className="btn btn-secondary"
                  onClick={handleCloseAISearchModal}
                >
                  Cancel
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleAISearchDatasets}
                  disabled={!aiSearchQuery.trim()}
                >
                  Search Datasets
                </button>
              </div>
            </div>
          )}

          {/* Loading State */}
          {isSearchingDatasets && (
            <div className="ai-search-loading">
              <LoadingSpinner message="AI is searching for relevant datasets..." />
            </div>
          )}

          {/* Results */}
          {discoveredDatasets.length > 0 && !isSearchingDatasets && (
            <div className="ai-search-results">
              <DiscoveredDatasetsList
                datasets={discoveredDatasets}
                onApply={handleApplyDiscoveredDatasets}
                onBack={handleCloseAISearchModal}
                isApplying={isApplyingDatasets}
                error={aiSearchError}
              />
            </div>
          )}
        </div>
      </Modal>

      {/* Context Documents Modal */}
      <Modal
        isOpen={isContextDocsModalOpen}
        onClose={() => {
          setIsContextDocsModalOpen(false);
          // Refresh context docs list to show any changes
          listContextDocuments(projectId!, false).then(data => setContextDocs(data.documents)).catch(() => {});
        }}
        title="Context Documents"
        size="large"
      >
        <ContextDocuments
          projectId={projectId!}
          onClose={() => {
            setIsContextDocsModalOpen(false);
            // Refresh context docs list to show any changes
            listContextDocuments(projectId!, false).then(data => setContextDocs(data.documents)).catch(() => {});
          }}
        />
      </Modal>

      {/* Delete Confirmations */}
      <ConfirmDialog
        isOpen={!!deleteDataSourceTarget}
        onClose={() => setDeleteDataSourceTarget(null)}
        onConfirm={handleDeleteDataSource}
        title="Delete Data Source"
        message={`Are you sure you want to delete "${deleteDataSourceTarget?.name}"?`}
        confirmLabel="Delete"
        variant="danger"
        isLoading={isDeleting}
      />

      <ConfirmDialog
        isOpen={!!deleteDatasetTarget}
        onClose={() => setDeleteDatasetTarget(null)}
        onConfirm={handleDeleteDataset}
        title="Delete Dataset"
        message={`Are you sure you want to delete "${deleteDatasetTarget?.name}"? This may affect experiments using this dataset.`}
        confirmLabel="Delete"
        variant="danger"
        isLoading={isDeleting}
      />

      <ConfirmDialog
        isOpen={!!deleteExperimentTarget}
        onClose={() => setDeleteExperimentTarget(null)}
        onConfirm={handleDeleteExperiment}
        title="Delete Experiment"
        message={`Are you sure you want to delete "${deleteExperimentTarget?.name}"? This will also delete associated models.`}
        confirmLabel="Delete"
        variant="danger"
        isLoading={isDeleting}
      />

      {/* Data Source Detail Modal */}
      <DataSourceDetailModal
        dataSource={selectedDataSource}
        isOpen={!!selectedDataSource}
        onClose={() => setSelectedDataSource(null)}
      />

      {/* Dataset Spec Detail Modal */}
      <DatasetSpecDetailModal
        datasetSpec={selectedDatasetSpec}
        dataSources={dataSources}
        isOpen={!!selectedDatasetSpec}
        onClose={() => setSelectedDatasetSpec(null)}
        projectId={projectId}
        experimentDesignStepId={agentRun?.steps?.find(s => s.step_type === 'experiment_design' && s.status === 'completed')?.id}
        onExperimentsCreated={async () => {
          if (projectId) {
            const updatedExperiments = await listExperiments(projectId);
            setExperiments(updatedExperiments);
          }
        }}
      />

      {/* AI Assistant with project context */}
      <ChatBot
        title="Project Assistant"
        contextType="project"
        context={{
          project: {
            id: project.id,
            name: project.name,
            description: project.description,
            task_type: project.task_type,
            status: project.status,
            created_at: project.created_at,
          },
          data_sources: dataSources.map(ds => ({
            id: ds.id,
            name: ds.name,
            type: ds.type,
            schema: ds.schema_summary,
          })),
          datasets: datasetSpecs.map(ds => ({
            id: ds.id,
            name: ds.name,
            target_column: ds.target_column,
            feature_columns: ds.feature_columns,
          })),
          experiments: experiments.map(exp => ({
            id: exp.id,
            name: exp.name,
            status: exp.status,
            primary_metric: exp.primary_metric,
          })),
          models_count: models.length,
          visualizations: visualizations.map(viz => ({
            title: viz.title,
            description: viz.description,
            chart_type: viz.chart_type,
            explanation: viz.explanation,
          })),
          agent_pipeline: agentRun ? {
            status: agentRun.status,
            description: agentRun.description,
            steps: agentRun.steps?.map(step => ({
              step_type: step.step_type,
              status: step.status,
              output_json: step.output_json,
              error_message: step.error_message,
            })),
          } : null,
        }}
        visualizations={visualizations.map(viz => ({
          title: viz.title,
          description: viz.description,
          chart_type: viz.chart_type,
          image_base64: viz.image_base64,
        }))}
      />

      {/* Project Settings Modal */}
      <ProjectSettingsModal
        project={project}
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        onUpdate={(updatedProject) => setProject(updatedProject)}
      />
    </div>
  );
}

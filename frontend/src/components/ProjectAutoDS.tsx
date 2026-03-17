import { useState, useEffect, useCallback } from 'react';
import type { Project, DatasetSpec, AutoDSConfig, ExecutionMode } from '../types/api';
import {
  startProjectAutoDS,
  stopProjectAutoDS,
  getProjectAutoDSStatus,
  listProjectAutoDSSessions,
  updateProject,
  ApiException,
  type ProjectAutoDSStatus,
  type ProjectAutoDSSessionSummary,
} from '../services/api';
import StatusBadge from './StatusBadge';
import LoadingSpinner from './LoadingSpinner';
import Modal from './Modal';
import ContextDocuments from './ContextDocuments';

interface ProjectAutoDSProps {
  projectId: string;
  project: Project;
  datasetSpecs: DatasetSpec[];
  onProjectUpdate: (project: Project) => void;
  onExperimentsRefresh: () => Promise<void>;
}

export default function ProjectAutoDS({
  projectId,
  project,
  datasetSpecs,
  onProjectUpdate,
  onExperimentsRefresh,
}: ProjectAutoDSProps) {
  // Status state
  const [status, setStatus] = useState<ProjectAutoDSStatus | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(true);

  // Sessions history state
  const [sessions, setSessions] = useState<ProjectAutoDSSessionSummary[]>([]);
  const [sessionsTotal, setSessionsTotal] = useState(0);
  const [isLoadingSessions, setIsLoadingSessions] = useState(true);

  // Action states
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Dataset selection for new session
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string>>(new Set());

  // Context documents modal state
  const [isContextDocsModalOpen, setIsContextDocsModalOpen] = useState(false);

  // Config form state
  const [configForm, setConfigForm] = useState<AutoDSConfig>({
    max_iterations: 10,
    accuracy_threshold: null,
    time_budget_minutes: null,
    parallel_experiments: 1,
    start_on_pipeline_complete: true,
    max_experiments_per_dataset: 3,
    max_active_datasets: 5,
    // Execution mode settings
    execution_mode: 'legacy',
    adaptive_decline_threshold: 0.05,
    phased_min_baseline_improvement: 0.01,
    dynamic_experiments_per_cycle: 1,
    // Validation strategy settings
    validation_strategy: 'standard',
    validation_num_seeds: 1,
    validation_cv_folds: 5,
    // Tier 1 feature flags
    enable_feature_engineering: true,
    enable_ensemble: true,
    enable_ablation: true,
    // Tier 2 feature flags
    enable_diverse_configs: true,
    // Context document settings
    use_context_documents: true,
    context_ab_testing: false,
  });

  // Load status and sessions
  const fetchStatus = useCallback(async () => {
    try {
      const statusData = await getProjectAutoDSStatus(projectId);
      setStatus(statusData);
    } catch (err) {
      console.error('Failed to fetch Auto DS status:', err);
    } finally {
      setIsLoadingStatus(false);
    }
  }, [projectId]);

  const fetchSessions = useCallback(async () => {
    try {
      const response = await listProjectAutoDSSessions(projectId);
      setSessions(response.items);
      setSessionsTotal(response.total);
    } catch (err) {
      console.error('Failed to fetch Auto DS sessions:', err);
    } finally {
      setIsLoadingSessions(false);
    }
  }, [projectId]);

  // Initialize config form from project
  useEffect(() => {
    if (project.auto_ds_config_json) {
      setConfigForm({
        max_iterations: project.auto_ds_config_json.max_iterations ?? 10,
        accuracy_threshold: project.auto_ds_config_json.accuracy_threshold ?? null,
        time_budget_minutes: project.auto_ds_config_json.time_budget_minutes ?? null,
        parallel_experiments: project.auto_ds_config_json.parallel_experiments ?? 1,
        start_on_pipeline_complete: project.auto_ds_config_json.start_on_pipeline_complete ?? true,
        max_experiments_per_dataset: project.auto_ds_config_json.max_experiments_per_dataset ?? 3,
        max_active_datasets: project.auto_ds_config_json.max_active_datasets ?? 5,
        // Execution mode settings
        execution_mode: project.auto_ds_config_json.execution_mode ?? 'legacy',
        adaptive_decline_threshold: project.auto_ds_config_json.adaptive_decline_threshold ?? 0.05,
        phased_min_baseline_improvement: project.auto_ds_config_json.phased_min_baseline_improvement ?? 0.01,
        dynamic_experiments_per_cycle: project.auto_ds_config_json.dynamic_experiments_per_cycle ?? 1,
        // Validation strategy settings
        validation_strategy: project.auto_ds_config_json.validation_strategy ?? 'standard',
        validation_num_seeds: project.auto_ds_config_json.validation_num_seeds ?? 1,
        validation_cv_folds: project.auto_ds_config_json.validation_cv_folds ?? 5,
        // Tier 1 feature flags
        enable_feature_engineering: project.auto_ds_config_json.enable_feature_engineering ?? true,
        enable_ensemble: project.auto_ds_config_json.enable_ensemble ?? true,
        enable_ablation: project.auto_ds_config_json.enable_ablation ?? true,
        // Tier 2 feature flags
        enable_diverse_configs: project.auto_ds_config_json.enable_diverse_configs ?? true,
        // Context document settings
        use_context_documents: project.auto_ds_config_json.use_context_documents ?? true,
        context_ab_testing: project.auto_ds_config_json.context_ab_testing ?? false,
      });
    }
  }, [project.auto_ds_config_json]);

  // Fetch data on mount
  useEffect(() => {
    fetchStatus();
    fetchSessions();
  }, [fetchStatus, fetchSessions]);

  // Auto-refresh status when session is active
  useEffect(() => {
    if (!status?.active) return;

    const interval = setInterval(() => {
      fetchStatus();
      fetchSessions();
      onExperimentsRefresh();
    }, 10000);

    return () => clearInterval(interval);
  }, [status?.active, fetchStatus, fetchSessions, onExperimentsRefresh]);

  // Handle start Auto DS
  const handleStart = async () => {
    setIsStarting(true);
    setError(null);
    setSuccess(null);

    try {
      const datasetIds = selectedDatasets.size > 0
        ? Array.from(selectedDatasets)
        : undefined;

      await startProjectAutoDS(projectId, datasetIds);
      setSuccess('Auto DS session started successfully');
      setSelectedDatasets(new Set());

      // Refresh status and sessions
      await fetchStatus();
      await fetchSessions();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to start Auto DS session');
      }
    } finally {
      setIsStarting(false);
    }
  };

  // Handle stop Auto DS
  const handleStop = async () => {
    setIsStopping(true);
    setError(null);
    setSuccess(null);

    try {
      await stopProjectAutoDS(projectId);
      setSuccess('Auto DS session stopped');

      // Refresh status and sessions
      await fetchStatus();
      await fetchSessions();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to stop Auto DS session');
      }
    } finally {
      setIsStopping(false);
    }
  };

  // Handle toggle auto_ds_enabled
  const handleToggleEnabled = async () => {
    setError(null);

    try {
      const updatedProject = await updateProject(projectId, {
        auto_ds_enabled: !project.auto_ds_enabled,
      });
      onProjectUpdate(updatedProject);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to update Auto DS settings');
      }
    }
  };

  // Handle save config
  const handleSaveConfig = async () => {
    setIsSavingConfig(true);
    setError(null);
    setSuccess(null);

    try {
      const updatedProject = await updateProject(projectId, {
        auto_ds_config_json: configForm,
      });
      onProjectUpdate(updatedProject);
      setSuccess('Configuration saved');
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to save configuration');
      }
    } finally {
      setIsSavingConfig(false);
    }
  };

  // Toggle dataset selection
  const toggleDatasetSelection = (datasetId: string) => {
    setSelectedDatasets(prev => {
      const next = new Set(prev);
      if (next.has(datasetId)) {
        next.delete(datasetId);
      } else {
        next.add(datasetId);
      }
      return next;
    });
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleString();
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'running': return 'var(--info-color, #3b82f6)';
      case 'completed': return 'var(--success-color, #10b981)';
      case 'failed': return 'var(--error-color, #ef4444)';
      case 'stopped': return 'var(--warning-color, #f59e0b)';
      case 'pending': return 'var(--text-secondary, #6b7280)';
      default: return 'var(--text-secondary, #6b7280)';
    }
  };

  return (
    <div className="project-auto-ds">
      {/* Header with Context Docs button */}
      <div className="auto-ds-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h3 style={{ margin: 0 }}>Automated Data Science</h3>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#888' }}>
            Automatically iterate on experiments to find the best model
          </p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={() => setIsContextDocsModalOpen(true)}
          title="Upload context documents to help AI understand your problem"
        >
          Context Docs
        </button>
      </div>

      {/* Messages */}
      {error && (
        <div className="form-error" style={{ marginBottom: '1rem' }}>
          {error}
          <button onClick={() => setError(null)} style={{ marginLeft: '1rem' }}>
            Dismiss
          </button>
        </div>
      )}
      {success && (
        <div className="form-success" style={{ marginBottom: '1rem' }}>
          {success}
          <button onClick={() => setSuccess(null)} style={{ marginLeft: '1rem' }}>
            Dismiss
          </button>
        </div>
      )}

      {/* Current Status Section */}
      <div className="auto-ds-section">
        <h3>Current Status</h3>
        {isLoadingStatus ? (
          <LoadingSpinner message="Loading status..." />
        ) : status?.active ? (
          <div className="auto-ds-active-session">
            <div className="session-info-grid">
              <div className="info-item">
                <span className="label">Session:</span>
                <span className="value">{status.name}</span>
              </div>
              <div className="info-item">
                <span className="label">Status:</span>
                <StatusBadge status={status.status || 'unknown'} />
              </div>
              <div className="info-item">
                <span className="label">Progress:</span>
                <span className="value">
                  Iteration {status.current_iteration} / {status.max_iterations}
                </span>
              </div>
              <div className="info-item">
                <span className="label">Experiments Run:</span>
                <span className="value">{status.total_experiments_run}</span>
              </div>
              <div className="info-item">
                <span className="label">Best Score:</span>
                <span className="value">
                  {status.best_score != null ? status.best_score.toFixed(4) : '-'}
                </span>
              </div>
              <div className="info-item">
                <span className="label">Started:</span>
                <span className="value">{formatDate(status.started_at || null)}</span>
              </div>
            </div>

            <div className="session-actions" style={{ marginTop: '1rem' }}>
              <button
                className="btn btn-danger"
                onClick={handleStop}
                disabled={isStopping}
              >
                {isStopping ? (
                  <>
                    <span className="spinner spinner-small"></span>
                    Stopping...
                  </>
                ) : (
                  'Stop Session'
                )}
              </button>
            </div>
          </div>
        ) : (
          <div className="auto-ds-no-session">
            <p>No active Auto DS session</p>

            {/* Dataset Selection */}
            {datasetSpecs.length > 0 && (
              <div className="dataset-selection" style={{ marginTop: '1rem' }}>
                <h4>Select Datasets (optional)</h4>
                <p className="hint">Leave empty to use all datasets in the project</p>
                <div className="dataset-checkboxes">
                  {datasetSpecs.map(ds => (
                    <label key={ds.id} className="dataset-checkbox-label">
                      <input
                        type="checkbox"
                        checked={selectedDatasets.has(ds.id)}
                        onChange={() => toggleDatasetSelection(ds.id)}
                      />
                      <span>{ds.name}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            <div className="session-actions" style={{ marginTop: '1rem' }}>
              <button
                className="btn btn-primary"
                onClick={handleStart}
                disabled={isStarting || datasetSpecs.length === 0}
                title={datasetSpecs.length === 0 ? 'Create at least one dataset first' : undefined}
              >
                {isStarting ? (
                  <>
                    <span className="spinner spinner-small"></span>
                    Starting...
                  </>
                ) : (
                  'Start Auto DS Session'
                )}
              </button>
            </div>
            {datasetSpecs.length === 0 && (
              <p className="hint" style={{ marginTop: '0.5rem' }}>
                Create at least one dataset before starting Auto DS
              </p>
            )}
          </div>
        )}
      </div>

      {/* Configuration Section */}
      <div className="auto-ds-section" style={{ marginTop: '2rem' }}>
        <h3>Configuration</h3>

        <div className="config-toggle" style={{ marginBottom: '1rem' }}>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={project.auto_ds_enabled}
              onChange={handleToggleEnabled}
            />
            <span>Enable Auto DS (auto-start on pipeline completion)</span>
          </label>
        </div>

        <div className="config-form">
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="max_iterations">Max Iterations</label>
              <input
                type="number"
                id="max_iterations"
                className="form-control"
                min={1}
                max={100}
                value={configForm.max_iterations || 10}
                onChange={(e) => setConfigForm(prev => ({
                  ...prev,
                  max_iterations: parseInt(e.target.value) || 10,
                }))}
              />
              <span className="hint">Maximum research iterations (1-100)</span>
            </div>

            <div className="form-group">
              <label htmlFor="parallel_experiments">Parallel Experiments</label>
              <input
                type="number"
                id="parallel_experiments"
                className="form-control"
                min={1}
                max={5}
                value={configForm.parallel_experiments || 1}
                onChange={(e) => setConfigForm(prev => ({
                  ...prev,
                  parallel_experiments: parseInt(e.target.value) || 1,
                }))}
              />
              <span className="hint">Experiments to run in parallel (1-5)</span>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="accuracy_threshold">Accuracy Threshold (optional)</label>
              <input
                type="number"
                id="accuracy_threshold"
                className="form-control"
                min={0}
                max={1}
                step={0.01}
                value={configForm.accuracy_threshold ?? ''}
                placeholder="e.g., 0.95"
                onChange={(e) => setConfigForm(prev => ({
                  ...prev,
                  accuracy_threshold: e.target.value ? parseFloat(e.target.value) : null,
                }))}
              />
              <span className="hint">Stop when accuracy reaches this threshold (0-1)</span>
            </div>

            <div className="form-group">
              <label htmlFor="time_budget_minutes">Time Budget (minutes, optional)</label>
              <input
                type="number"
                id="time_budget_minutes"
                className="form-control"
                min={10}
                max={1440}
                value={configForm.time_budget_minutes ?? ''}
                placeholder="e.g., 60"
                onChange={(e) => setConfigForm(prev => ({
                  ...prev,
                  time_budget_minutes: e.target.value ? parseInt(e.target.value) : null,
                }))}
              />
              <span className="hint">Maximum time budget in minutes (10-1440)</span>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="max_experiments_per_dataset">Experiments per Dataset</label>
              <input
                type="number"
                id="max_experiments_per_dataset"
                className="form-control"
                min={1}
                max={10}
                value={configForm.max_experiments_per_dataset || 3}
                onChange={(e) => setConfigForm(prev => ({
                  ...prev,
                  max_experiments_per_dataset: parseInt(e.target.value) || 3,
                }))}
              />
              <span className="hint">Max experiment variants per dataset (1-10)</span>
            </div>

            <div className="form-group">
              <label htmlFor="max_active_datasets">Max Active Datasets</label>
              <input
                type="number"
                id="max_active_datasets"
                className="form-control"
                min={1}
                max={20}
                value={configForm.max_active_datasets || 5}
                onChange={(e) => setConfigForm(prev => ({
                  ...prev,
                  max_active_datasets: parseInt(e.target.value) || 5,
                }))}
              />
              <span className="hint">Max datasets to explore per iteration (1-20)</span>
            </div>
          </div>

          {/* Execution Mode Section */}
          <div className="execution-mode-section" style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--bg-secondary, #f9fafb)', borderRadius: '8px' }}>
            <h4 style={{ margin: '0 0 0.75rem 0' }}>Execution Mode</h4>
            <div className="execution-mode-options">
              <label className="execution-mode-option">
                <input
                  type="radio"
                  name="execution_mode"
                  value="legacy"
                  checked={configForm.execution_mode === 'legacy'}
                  onChange={() => setConfigForm(prev => ({ ...prev, execution_mode: 'legacy' }))}
                />
                <div className="option-content">
                  <span className="option-title">Legacy (Default)</span>
                  <span className="option-desc">Run all planned experiments, then analyze. Simple and predictable.</span>
                </div>
              </label>

              <label className="execution-mode-option">
                <input
                  type="radio"
                  name="execution_mode"
                  value="adaptive"
                  checked={configForm.execution_mode === 'adaptive'}
                  onChange={() => setConfigForm(prev => ({ ...prev, execution_mode: 'adaptive' }))}
                />
                <div className="option-content">
                  <span className="option-title">Adaptive (Option A)</span>
                  <span className="option-desc">Early stopping: skip remaining experiments if scores decline significantly.</span>
                </div>
              </label>

              <label className="execution-mode-option">
                <input
                  type="radio"
                  name="execution_mode"
                  value="phased"
                  checked={configForm.execution_mode === 'phased'}
                  onChange={() => setConfigForm(prev => ({ ...prev, execution_mode: 'phased' }))}
                />
                <div className="option-content">
                  <span className="option-title">Phased (Option B)</span>
                  <span className="option-desc">Run baselines first, analyze, then run targeted variants for promising datasets.</span>
                </div>
              </label>

              <label className="execution-mode-option">
                <input
                  type="radio"
                  name="execution_mode"
                  value="dynamic"
                  checked={configForm.execution_mode === 'dynamic'}
                  onChange={() => setConfigForm(prev => ({ ...prev, execution_mode: 'dynamic' }))}
                />
                <div className="option-content">
                  <span className="option-title">Dynamic AI Planning</span>
                  <span className="option-desc">AI designs each experiment based on previous results. Most adaptive, highest potential accuracy.</span>
                </div>
              </label>
            </div>

            {/* Conditional threshold settings */}
            {configForm.execution_mode === 'adaptive' && (
              <div className="form-group" style={{ marginTop: '1rem' }}>
                <label htmlFor="adaptive_decline_threshold">Decline Threshold (%)</label>
                <input
                  type="number"
                  id="adaptive_decline_threshold"
                  className="form-control"
                  min={1}
                  max={50}
                  step={1}
                  value={(configForm.adaptive_decline_threshold ?? 0.05) * 100}
                  onChange={(e) => setConfigForm(prev => ({
                    ...prev,
                    adaptive_decline_threshold: (parseFloat(e.target.value) || 5) / 100,
                  }))}
                />
                <span className="hint">Skip remaining experiments if score drops by more than this % from best (default: 5%)</span>
              </div>
            )}

            {configForm.execution_mode === 'phased' && (
              <div className="form-group" style={{ marginTop: '1rem' }}>
                <label htmlFor="phased_min_baseline_improvement">Min Baseline Improvement (%)</label>
                <input
                  type="number"
                  id="phased_min_baseline_improvement"
                  className="form-control"
                  min={0}
                  max={50}
                  step={0.5}
                  value={(configForm.phased_min_baseline_improvement ?? 0.01) * 100}
                  onChange={(e) => setConfigForm(prev => ({
                    ...prev,
                    phased_min_baseline_improvement: (parseFloat(e.target.value) || 1) / 100,
                  }))}
                />
                <span className="hint">Only run variants for datasets with baseline improvement above this % (default: 1%)</span>
              </div>
            )}

            {configForm.execution_mode === 'dynamic' && (
              <div className="form-group" style={{ marginTop: '1rem' }}>
                <label htmlFor="dynamic_experiments_per_cycle">Experiments per Cycle</label>
                <input
                  type="number"
                  id="dynamic_experiments_per_cycle"
                  className="form-control"
                  min={1}
                  max={5}
                  value={configForm.dynamic_experiments_per_cycle ?? 1}
                  onChange={(e) => setConfigForm(prev => ({
                    ...prev,
                    dynamic_experiments_per_cycle: parseInt(e.target.value) || 1,
                  }))}
                />
                <span className="hint">Number of experiments AI plans per cycle before re-evaluating (default: 1)</span>
              </div>
            )}
          </div>

          {/* Advanced Features Section */}
          <div className="advanced-features-section" style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--bg-secondary, #f9fafb)', borderRadius: '8px' }}>
            <h4 style={{ margin: '0 0 0.75rem 0' }}>Advanced AI Features (Tier 1)</h4>
            <p className="hint" style={{ marginBottom: '0.75rem' }}>Enable AI-driven experiment optimization strategies</p>

            <div className="feature-checkboxes" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={configForm.enable_feature_engineering ?? true}
                  onChange={(e) => setConfigForm(prev => ({ ...prev, enable_feature_engineering: e.target.checked }))}
                />
                <span><strong>Feature Engineering</strong> - AI creates derived features to improve predictions</span>
              </label>

              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={configForm.enable_ensemble ?? true}
                  onChange={(e) => setConfigForm(prev => ({ ...prev, enable_ensemble: e.target.checked }))}
                />
                <span><strong>Ensemble Building</strong> - AI combines successful models for better accuracy</span>
              </label>

              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={configForm.enable_ablation ?? true}
                  onChange={(e) => setConfigForm(prev => ({ ...prev, enable_ablation: e.target.checked }))}
                />
                <span><strong>Ablation Studies</strong> - AI tests which features really matter</span>
              </label>

              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={configForm.enable_diverse_configs ?? true}
                  onChange={(e) => setConfigForm(prev => ({ ...prev, enable_diverse_configs: e.target.checked }))}
                />
                <span><strong>Diverse Configs</strong> - AI explores varied model configurations</span>
              </label>
            </div>
          </div>

          {/* Validation Strategy Section */}
          <div className="validation-section" style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--bg-secondary, #f9fafb)', borderRadius: '8px' }}>
            <h4 style={{ margin: '0 0 0.75rem 0' }}>Validation Strategy</h4>
            <p className="hint" style={{ marginBottom: '0.75rem' }}>Control how thoroughly experiments are validated (affects accuracy of reported scores)</p>

            <div className="validation-options">
              <label className="execution-mode-option">
                <input
                  type="radio"
                  name="validation_strategy"
                  value="standard"
                  checked={configForm.validation_strategy === 'standard'}
                  onChange={() => setConfigForm(prev => ({ ...prev, validation_strategy: 'standard' }))}
                />
                <div className="option-content">
                  <span className="option-title">Standard</span>
                  <span className="option-desc">Single cross-validation (fast, good for exploration)</span>
                </div>
              </label>

              <label className="execution-mode-option">
                <input
                  type="radio"
                  name="validation_strategy"
                  value="robust"
                  checked={configForm.validation_strategy === 'robust'}
                  onChange={() => setConfigForm(prev => ({ ...prev, validation_strategy: 'robust' }))}
                />
                <div className="option-content">
                  <span className="option-title">Robust</span>
                  <span className="option-desc">Multiple seeds + more CV folds (higher confidence scores)</span>
                </div>
              </label>

              <label className="execution-mode-option">
                <input
                  type="radio"
                  name="validation_strategy"
                  value="strict"
                  checked={configForm.validation_strategy === 'strict'}
                  onChange={() => setConfigForm(prev => ({ ...prev, validation_strategy: 'strict' }))}
                />
                <div className="option-content">
                  <span className="option-title">Strict</span>
                  <span className="option-desc">Time-aware validation + leakage detection (production-ready)</span>
                </div>
              </label>
            </div>

            {configForm.validation_strategy === 'robust' && (
              <div className="form-row" style={{ marginTop: '1rem' }}>
                <div className="form-group">
                  <label htmlFor="validation_num_seeds">Number of Seeds</label>
                  <input
                    type="number"
                    id="validation_num_seeds"
                    className="form-control"
                    min={1}
                    max={10}
                    value={configForm.validation_num_seeds ?? 1}
                    onChange={(e) => setConfigForm(prev => ({
                      ...prev,
                      validation_num_seeds: parseInt(e.target.value) || 1,
                    }))}
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="validation_cv_folds">CV Folds</label>
                  <input
                    type="number"
                    id="validation_cv_folds"
                    className="form-control"
                    min={2}
                    max={20}
                    value={configForm.validation_cv_folds ?? 5}
                    onChange={(e) => setConfigForm(prev => ({
                      ...prev,
                      validation_cv_folds: parseInt(e.target.value) || 5,
                    }))}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Context Documents Section */}
          <div className="context-docs-section" style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--bg-secondary, #f9fafb)', borderRadius: '8px' }}>
            <h4 style={{ margin: '0 0 0.75rem 0' }}>Context Documents</h4>
            <p className="hint" style={{ marginBottom: '0.75rem' }}>
              Control how uploaded context documents (PDFs, Word docs, etc.) are used in AutoDS
            </p>

            <div className="feature-checkboxes" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={configForm.use_context_documents ?? true}
                  onChange={(e) => setConfigForm(prev => ({ ...prev, use_context_documents: e.target.checked }))}
                />
                <span>
                  <strong>Use Context Documents</strong>
                  <span style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary, #6b7280)' }}>
                    Include uploaded context documents in AI prompts for iteration planning and result analysis
                  </span>
                </span>
              </label>

              <label className="toggle-label" style={{ opacity: configForm.use_context_documents ? 1 : 0.5 }}>
                <input
                  type="checkbox"
                  checked={configForm.context_ab_testing ?? false}
                  onChange={(e) => setConfigForm(prev => ({ ...prev, context_ab_testing: e.target.checked }))}
                  disabled={!configForm.use_context_documents}
                />
                <span>
                  <strong>A/B Testing (Context vs No Context)</strong>
                  <span style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary, #6b7280)' }}>
                    Create experiments both with and without context documents to measure their impact.
                    Experiments will be named with [WITH CONTEXT] or [NO CONTEXT] suffixes.
                  </span>
                </span>
              </label>
            </div>
          </div>

          <div className="form-actions">
            <button
              className="btn btn-primary"
              onClick={handleSaveConfig}
              disabled={isSavingConfig}
            >
              {isSavingConfig ? (
                <>
                  <span className="spinner spinner-small"></span>
                  Saving...
                </>
              ) : (
                'Save Configuration'
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Session History Section */}
      <div className="auto-ds-section" style={{ marginTop: '2rem' }}>
        <h3>Session History ({sessionsTotal})</h3>

        {isLoadingSessions ? (
          <LoadingSpinner message="Loading sessions..." />
        ) : sessions.length === 0 ? (
          <p className="empty-state">No Auto DS sessions yet</p>
        ) : (
          <div className="sessions-table">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Experiments</th>
                  <th>Best Score</th>
                  <th>Started</th>
                  <th>Completed</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map(session => (
                  <tr key={session.id}>
                    <td>{session.name}</td>
                    <td>
                      <span
                        className="status-badge"
                        style={{
                          backgroundColor: getStatusColor(session.status),
                          color: 'white',
                          padding: '2px 8px',
                          borderRadius: '4px',
                          fontSize: '0.85em',
                        }}
                      >
                        {session.status}
                      </span>
                    </td>
                    <td>{session.current_iteration} / {session.max_iterations}</td>
                    <td>{session.total_experiments_run}</td>
                    <td>{session.best_score != null ? session.best_score.toFixed(4) : '-'}</td>
                    <td>{formatDate(session.started_at)}</td>
                    <td>{formatDate(session.completed_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <style>{`
        .project-auto-ds {
          padding: 1rem 0;
        }

        .auto-ds-section {
          background: var(--card-bg, #fff);
          border: 1px solid var(--border-color, #e5e7eb);
          border-radius: 8px;
          padding: 1.5rem;
        }

        .auto-ds-section h3 {
          margin: 0 0 1rem 0;
          font-size: 1.1rem;
          font-weight: 600;
        }

        .auto-ds-section h4 {
          margin: 0 0 0.5rem 0;
          font-size: 1rem;
          font-weight: 500;
        }

        .session-info-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 1rem;
        }

        .info-item {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .info-item .label {
          font-size: 0.85rem;
          color: var(--text-secondary, #6b7280);
        }

        .info-item .value {
          font-weight: 500;
        }

        .dataset-checkboxes {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          margin-top: 0.5rem;
        }

        .dataset-checkbox-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          cursor: pointer;
        }

        .dataset-checkbox-label input {
          cursor: pointer;
        }

        .toggle-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          cursor: pointer;
        }

        .toggle-label input {
          cursor: pointer;
        }

        .config-form {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .form-row {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
          gap: 1rem;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .form-group label {
          font-weight: 500;
          font-size: 0.9rem;
        }

        .form-group .hint {
          font-size: 0.8rem;
          color: var(--text-secondary, #6b7280);
        }

        .hint {
          font-size: 0.85rem;
          color: var(--text-secondary, #6b7280);
        }

        .empty-state {
          color: var(--text-secondary, #6b7280);
          font-style: italic;
        }

        .sessions-table {
          overflow-x: auto;
        }

        .sessions-table table {
          width: 100%;
          border-collapse: collapse;
        }

        .sessions-table th,
        .sessions-table td {
          text-align: left;
          padding: 0.75rem;
          border-bottom: 1px solid var(--border-color, #e5e7eb);
        }

        .sessions-table th {
          font-weight: 600;
          font-size: 0.85rem;
          color: var(--text-secondary, #6b7280);
          background: var(--table-header-bg, #f9fafb);
        }

        .sessions-table tbody tr:hover {
          background: var(--hover-bg, #f9fafb);
        }

        .form-success {
          background-color: var(--success-bg, #d1fae5);
          color: var(--success-color, #065f46);
          padding: 0.75rem 1rem;
          border-radius: 6px;
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .form-error {
          background-color: var(--error-bg, #fee2e2);
          color: var(--error-color, #991b1b);
          padding: 0.75rem 1rem;
          border-radius: 6px;
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .form-actions {
          margin-top: 1rem;
        }
      `}</style>

      {/* Context Documents Modal */}
      <Modal
        isOpen={isContextDocsModalOpen}
        onClose={() => setIsContextDocsModalOpen(false)}
        title="Context Documents"
        size="large"
      >
        <ContextDocuments
          projectId={projectId}
          onClose={() => setIsContextDocsModalOpen(false)}
        />
      </Modal>
    </div>
  );
}

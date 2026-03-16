import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import type { AutoDSSessionSummary, AutoDSSessionStatus, AutoDSSessionCreate } from '../types/api';
import { listProjects, listAutoDSSessions, createAutoDSSession, ApiException } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import Modal from '../components/Modal';

interface SessionWithProject extends AutoDSSessionSummary {
  projectName: string;
}

const statusColors: Record<AutoDSSessionStatus, string> = {
  pending: '#6c757d',
  running: '#0d6efd',
  paused: '#ffc107',
  completed: '#198754',
  failed: '#dc3545',
  stopped: '#6c757d',
};

const statusLabels: Record<AutoDSSessionStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  paused: 'Paused',
  completed: 'Completed',
  failed: 'Failed',
  stopped: 'Stopped',
};

export default function AutoDS() {
  const [sessions, setSessions] = useState<SessionWithProject[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [projects, setProjects] = useState<{ id: string; name: string }[]>([]);
  const [isCreating, setIsCreating] = useState(false);

  // Form state for new session
  const [formData, setFormData] = useState<AutoDSSessionCreate & { project_id: string }>({
    project_id: '',
    name: '',
    description: '',
    max_iterations: 10,
    accuracy_threshold: 0.95,
    time_budget_minutes: 120,
    min_improvement_threshold: 0.001,
    plateau_iterations: 3,
    max_experiments_per_dataset: 5,
    max_active_datasets: 3,
    // Tier 2 fields
    execution_mode: 'dynamic',
    adaptive_decline_threshold: 0.05,
    phased_min_baseline_improvement: 0.01,
    dynamic_experiments_per_cycle: 1,
    validation_strategy: 'standard',
    validation_num_seeds: 3,
    validation_cv_folds: 5,
    enable_feature_engineering: true,
    enable_ensemble: true,
    enable_ablation: true,
    enable_diverse_configs: true,
  });

  const fetchAll = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const projectsResponse = await listProjects();
      const projectList = projectsResponse.items;
      setProjects(projectList.map(p => ({ id: p.id, name: p.name })));

      const allSessions: SessionWithProject[] = [];
      for (const project of projectList) {
        try {
          const response = await listAutoDSSessions(project.id);
          allSessions.push(
            ...response.sessions.map((s) => ({
              ...s,
              projectName: project.name,
            }))
          );
        } catch {
          // Skip projects that fail
        }
      }

      // Sort by created_at desc
      allSessions.sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );

      setSessions(allSessions);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load Auto DS sessions');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const handleCreateSession = async () => {
    if (!formData.project_id || !formData.name) {
      setError('Please select a project and enter a session name');
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      const { project_id, ...sessionData } = formData;
      await createAutoDSSession(project_id, sessionData);
      setShowCreateModal(false);
      setFormData({
        project_id: '',
        name: '',
        description: '',
        max_iterations: 10,
        accuracy_threshold: 0.95,
        time_budget_minutes: 120,
        min_improvement_threshold: 0.001,
        plateau_iterations: 3,
        max_experiments_per_dataset: 5,
        max_active_datasets: 3,
        execution_mode: 'dynamic',
        adaptive_decline_threshold: 0.05,
        phased_min_baseline_improvement: 0.01,
        dynamic_experiments_per_cycle: 1,
        validation_strategy: 'standard',
        validation_num_seeds: 3,
        validation_cv_folds: 5,
        enable_feature_engineering: true,
        enable_ensemble: true,
        enable_ablation: true,
        enable_diverse_configs: true,
      });
      await fetchAll();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to create session');
      }
    } finally {
      setIsCreating(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (isLoading) {
    return (
      <div className="auto-ds-page">
        <LoadingSpinner message="Loading Auto DS sessions..." />
      </div>
    );
  }

  return (
    <div className="auto-ds-page">
      <div className="page-header">
        <div>
          <h2>Auto DS</h2>
          <p className="page-subtitle">
            Autonomous Data Science - Run intelligent, iterative ML research sessions
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => setShowCreateModal(true)}
        >
          New Session
        </button>
      </div>

      {error && <ErrorMessage message={error} />}

      {sessions.length === 0 ? (
        <EmptyState
          title="No Auto DS sessions yet"
          description="Create an autonomous research session to automatically explore datasets, run experiments, and discover insights"
          actionLabel="Create First Session"
          onAction={() => setShowCreateModal(true)}
          icon="robot"
        />
      ) : (
        <div className="auto-ds-sessions-grid">
          {sessions.map((session) => (
            <Link
              key={session.id}
              to={`/auto-ds/${session.id}?project=${session.project_id}`}
              className="auto-ds-session-card"
            >
              <div className="session-card-header">
                <h3>{session.name}</h3>
                <span
                  className="status-badge"
                  style={{ backgroundColor: statusColors[session.status] }}
                >
                  {statusLabels[session.status]}
                </span>
              </div>

              <p className="session-project">Project: {session.projectName}</p>
              {session.description && (
                <p className="session-description">{session.description}</p>
              )}

              <div className="session-stats">
                <div className="stat">
                  <span className="stat-label">Iterations</span>
                  <span className="stat-value">
                    {session.current_iteration} / {session.max_iterations}
                  </span>
                </div>
                <div className="stat">
                  <span className="stat-label">Experiments</span>
                  <span className="stat-value">{session.total_experiments_run}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Best Score</span>
                  <span className="stat-value">
                    {session.best_score ? session.best_score.toFixed(4) : '-'}
                  </span>
                </div>
                <div className="stat">
                  <span className="stat-label">Status</span>
                  <span className="stat-value">{statusLabels[session.status]}</span>
                </div>
              </div>

              <div className="session-footer">
                <span className="session-date">Created: {formatDate(session.created_at)}</span>
              </div>
            </Link>
          ))}
        </div>
      )}

      {/* Create Session Modal */}
      <Modal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create Auto DS Session"
      >
        <div className="modal-form">
          <div className="form-group">
            <label htmlFor="project">Project *</label>
            <select
              id="project"
              value={formData.project_id}
              onChange={(e) => setFormData({ ...formData, project_id: e.target.value })}
            >
              <option value="">Select a project...</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="name">Session Name *</label>
            <input
              type="text"
              id="name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g., Feature Engineering Exploration"
            />
          </div>

          <div className="form-group">
            <label htmlFor="description">Description</label>
            <textarea
              id="description"
              value={formData.description ?? ''}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="Optional description of the research goals..."
              rows={3}
            />
          </div>

          <div className="form-section-title">Stopping Conditions</div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="max_iterations">Max Iterations</label>
              <input
                type="number"
                id="max_iterations"
                value={formData.max_iterations}
                onChange={(e) => setFormData({ ...formData, max_iterations: parseInt(e.target.value) })}
                min={1}
                max={100}
              />
            </div>

            <div className="form-group">
              <label htmlFor="accuracy_threshold">Accuracy Threshold</label>
              <input
                type="number"
                id="accuracy_threshold"
                value={formData.accuracy_threshold ?? 0.95}
                onChange={(e) => setFormData({ ...formData, accuracy_threshold: parseFloat(e.target.value) })}
                min={0}
                max={1}
                step={0.01}
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="time_budget_minutes">Time Budget (minutes)</label>
              <input
                type="number"
                id="time_budget_minutes"
                value={formData.time_budget_minutes ?? 120}
                onChange={(e) => setFormData({ ...formData, time_budget_minutes: parseInt(e.target.value) })}
                min={1}
              />
            </div>

            <div className="form-group">
              <label htmlFor="plateau_iterations">Plateau Iterations</label>
              <input
                type="number"
                id="plateau_iterations"
                value={formData.plateau_iterations}
                onChange={(e) => setFormData({ ...formData, plateau_iterations: parseInt(e.target.value) })}
                min={1}
                max={10}
              />
            </div>
          </div>

          <div className="form-section-title">Research Settings</div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="max_experiments_per_dataset">Max Experiments/Dataset</label>
              <input
                type="number"
                id="max_experiments_per_dataset"
                value={formData.max_experiments_per_dataset}
                onChange={(e) => setFormData({ ...formData, max_experiments_per_dataset: parseInt(e.target.value) })}
                min={1}
                max={20}
              />
            </div>

            <div className="form-group">
              <label htmlFor="max_active_datasets">Max Active Datasets</label>
              <input
                type="number"
                id="max_active_datasets"
                value={formData.max_active_datasets}
                onChange={(e) => setFormData({ ...formData, max_active_datasets: parseInt(e.target.value) })}
                min={1}
                max={10}
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="min_improvement_threshold">Min Improvement Threshold</label>
            <input
              type="number"
              id="min_improvement_threshold"
              value={formData.min_improvement_threshold}
              onChange={(e) => setFormData({ ...formData, min_improvement_threshold: parseFloat(e.target.value) })}
              min={0}
              max={0.1}
              step={0.001}
            />
            <p className="form-help">Minimum score improvement required to continue</p>
          </div>

          <div className="form-section-title">Execution Mode</div>
          <div className="execution-mode-options">
            <label className="radio-card">
              <input
                type="radio"
                name="execution_mode"
                value="legacy"
                checked={formData.execution_mode === 'legacy'}
                onChange={(e) => setFormData({ ...formData, execution_mode: e.target.value as 'legacy' | 'adaptive' | 'phased' | 'dynamic' })}
              />
              <div className="radio-card-content">
                <strong>Legacy</strong>
                <span>Original behavior - plan all experiments upfront</span>
              </div>
            </label>
            <label className="radio-card">
              <input
                type="radio"
                name="execution_mode"
                value="adaptive"
                checked={formData.execution_mode === 'adaptive'}
                onChange={(e) => setFormData({ ...formData, execution_mode: e.target.value as 'legacy' | 'adaptive' | 'phased' | 'dynamic' })}
              />
              <div className="radio-card-content">
                <strong>Adaptive</strong>
                <span>Re-plan remaining experiments if accuracy declines</span>
              </div>
            </label>
            <label className="radio-card">
              <input
                type="radio"
                name="execution_mode"
                value="phased"
                checked={formData.execution_mode === 'phased'}
                onChange={(e) => setFormData({ ...formData, execution_mode: e.target.value as 'legacy' | 'adaptive' | 'phased' | 'dynamic' })}
              />
              <div className="radio-card-content">
                <strong>Phased</strong>
                <span>Baseline first, then targeted improvements</span>
              </div>
            </label>
            <label className="radio-card">
              <input
                type="radio"
                name="execution_mode"
                value="dynamic"
                checked={formData.execution_mode === 'dynamic'}
                onChange={(e) => setFormData({ ...formData, execution_mode: e.target.value as 'legacy' | 'adaptive' | 'phased' | 'dynamic' })}
              />
              <div className="radio-card-content">
                <strong>Dynamic (Recommended)</strong>
                <span>Plan experiments one at a time based on results</span>
              </div>
            </label>
          </div>

          <div className="form-section-title">Advanced AI Features</div>
          <div className="feature-toggles">
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={formData.enable_feature_engineering}
                onChange={(e) => setFormData({ ...formData, enable_feature_engineering: e.target.checked })}
              />
              <span>Feature Engineering - Auto-generate new features</span>
            </label>
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={formData.enable_ensemble}
                onChange={(e) => setFormData({ ...formData, enable_ensemble: e.target.checked })}
              />
              <span>Ensemble Methods - Combine top models</span>
            </label>
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={formData.enable_ablation}
                onChange={(e) => setFormData({ ...formData, enable_ablation: e.target.checked })}
              />
              <span>Ablation Studies - Identify key features</span>
            </label>
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={formData.enable_diverse_configs}
                onChange={(e) => setFormData({ ...formData, enable_diverse_configs: e.target.checked })}
              />
              <span>Diverse Configs - Explore varied hyperparameters</span>
            </label>
          </div>

          <div className="form-section-title">Validation Strategy</div>
          <div className="validation-options">
            <label className="radio-card">
              <input
                type="radio"
                name="validation_strategy"
                value="standard"
                checked={formData.validation_strategy === 'standard'}
                onChange={(e) => setFormData({ ...formData, validation_strategy: e.target.value as 'standard' | 'robust' | 'strict' })}
              />
              <div className="radio-card-content">
                <strong>Standard</strong>
                <span>Single train/test split (fastest)</span>
              </div>
            </label>
            <label className="radio-card">
              <input
                type="radio"
                name="validation_strategy"
                value="robust"
                checked={formData.validation_strategy === 'robust'}
                onChange={(e) => setFormData({ ...formData, validation_strategy: e.target.value as 'standard' | 'robust' | 'strict' })}
              />
              <div className="radio-card-content">
                <strong>Robust</strong>
                <span>Multi-seed + cross-validation for reliable scores</span>
              </div>
            </label>
            <label className="radio-card">
              <input
                type="radio"
                name="validation_strategy"
                value="strict"
                checked={formData.validation_strategy === 'strict'}
                onChange={(e) => setFormData({ ...formData, validation_strategy: e.target.value as 'standard' | 'robust' | 'strict' })}
              />
              <div className="radio-card-content">
                <strong>Strict</strong>
                <span>Maximum validation rigor (slowest)</span>
              </div>
            </label>
          </div>

          {(formData.validation_strategy === 'robust' || formData.validation_strategy === 'strict') && (
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="validation_num_seeds">Number of Seeds</label>
                <input
                  type="number"
                  id="validation_num_seeds"
                  value={formData.validation_num_seeds}
                  onChange={(e) => setFormData({ ...formData, validation_num_seeds: parseInt(e.target.value) })}
                  min={2}
                  max={10}
                />
              </div>
              <div className="form-group">
                <label htmlFor="validation_cv_folds">CV Folds</label>
                <input
                  type="number"
                  id="validation_cv_folds"
                  value={formData.validation_cv_folds}
                  onChange={(e) => setFormData({ ...formData, validation_cv_folds: parseInt(e.target.value) })}
                  min={3}
                  max={10}
                />
              </div>
            </div>
          )}

          <div className="modal-actions">
            <button
              className="btn btn-secondary"
              onClick={() => setShowCreateModal(false)}
              disabled={isCreating}
            >
              Cancel
            </button>
            <button
              className="btn btn-primary"
              onClick={handleCreateSession}
              disabled={isCreating || !formData.project_id || !formData.name}
            >
              {isCreating ? 'Creating...' : 'Create Session'}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

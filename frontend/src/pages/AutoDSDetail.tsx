import { useState, useEffect, useCallback } from 'react';
import { useParams, useSearchParams, Link, useNavigate } from 'react-router-dom';
import type {
  AutoDSSession,
  AutoDSSessionProgress,
  AutoDSIteration,
  ResearchInsight,
  AutoDSSessionStatus,
  AutoDSIterationStatus,
} from '../types/api';
import {
  getAutoDSSession,
  getAutoDSSessionProgress,
  listSessionIterations,
  listSessionInsights,
  startAutoDSSession,
  pauseAutoDSSession,
  stopAutoDSSession,
  deleteAutoDSSession,
  ApiException,
} from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import ConfirmDialog from '../components/ConfirmDialog';

const statusColors: Record<AutoDSSessionStatus, string> = {
  pending: '#6c757d',
  running: '#0d6efd',
  paused: '#ffc107',
  completed: '#198754',
  failed: '#dc3545',
  stopped: '#6c757d',
};

const iterationStatusColors: Record<AutoDSIterationStatus, string> = {
  pending: '#6c757d',
  planning: '#17a2b8',
  running_experiments: '#0d6efd',
  analyzing: '#6f42c1',
  completed: '#198754',
  failed: '#dc3545',
};

export default function AutoDSDetail() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const projectId = searchParams.get('project') || '';

  const [session, setSession] = useState<AutoDSSession | null>(null);
  const [progress, setProgress] = useState<AutoDSSessionProgress | null>(null);
  const [iterations, setIterations] = useState<AutoDSIteration[]>([]);
  const [insights, setInsights] = useState<ResearchInsight[]>([]);
  const [activeTab, setActiveTab] = useState<'progress' | 'iterations' | 'insights'>('progress');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const fetchData = useCallback(async () => {
    if (!sessionId || !projectId) return;

    try {
      const [sessionData, iterationsData, insightsData] = await Promise.all([
        getAutoDSSession(projectId, sessionId),
        listSessionIterations(projectId, sessionId),
        listSessionInsights(projectId, sessionId),
      ]);

      setSession(sessionData);
      setIterations(iterationsData);
      setInsights(insightsData.insights);

      // Get progress if session is running
      if (sessionData.status === 'running') {
        const progressData = await getAutoDSSessionProgress(projectId, sessionId);
        setProgress(progressData);
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load session details');
      }
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, projectId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Poll for updates when running
  useEffect(() => {
    if (session?.status !== 'running') return;

    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [session?.status, fetchData]);

  const handleStart = async () => {
    if (!sessionId || !projectId) return;
    setActionLoading('start');
    try {
      await startAutoDSSession(projectId, sessionId);
      await fetchData();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to start session');
      }
    } finally {
      setActionLoading(null);
    }
  };

  const handlePause = async () => {
    if (!sessionId || !projectId) return;
    setActionLoading('pause');
    try {
      await pauseAutoDSSession(projectId, sessionId);
      await fetchData();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to pause session');
      }
    } finally {
      setActionLoading(null);
    }
  };

  const handleStop = async () => {
    if (!sessionId || !projectId) return;
    if (!window.confirm('Are you sure you want to stop this session? This cannot be undone.')) {
      return;
    }
    setActionLoading('stop');
    try {
      await stopAutoDSSession(projectId, sessionId);
      await fetchData();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to stop session');
      }
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async () => {
    if (!sessionId || !projectId) return;
    setIsDeleting(true);
    try {
      await deleteAutoDSSession(projectId, sessionId);
      navigate('/auto-ds');
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to delete session');
      }
      setShowDeleteConfirm(false);
    } finally {
      setIsDeleting(false);
    }
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (isLoading) {
    return (
      <div className="auto-ds-detail-page">
        <LoadingSpinner message="Loading session..." />
      </div>
    );
  }

  if (!session) {
    return (
      <div className="auto-ds-detail-page">
        <ErrorMessage message="Session not found" />
      </div>
    );
  }

  return (
    <div className="auto-ds-detail-page">
      <div className="page-header">
        <div className="header-breadcrumb">
          <Link to="/auto-ds" className="back-link">Auto DS</Link>
          <span className="breadcrumb-separator">/</span>
          <h2>{session.name}</h2>
          <span
            className="status-badge large"
            style={{ backgroundColor: statusColors[session.status] }}
          >
            {session.status.charAt(0).toUpperCase() + session.status.slice(1)}
          </span>
        </div>

        <div className="header-actions">
          {session.status === 'pending' && (
            <button
              className="btn btn-primary"
              onClick={handleStart}
              disabled={actionLoading !== null}
            >
              {actionLoading === 'start' ? 'Starting...' : 'Start Session'}
            </button>
          )}
          {session.status === 'running' && (
            <>
              <button
                className="btn btn-secondary"
                onClick={handlePause}
                disabled={actionLoading !== null}
              >
                {actionLoading === 'pause' ? 'Pausing...' : 'Pause'}
              </button>
              <button
                className="btn btn-danger"
                onClick={handleStop}
                disabled={actionLoading !== null}
              >
                {actionLoading === 'stop' ? 'Stopping...' : 'Stop'}
              </button>
            </>
          )}
          {session.status === 'paused' && (
            <>
              <button
                className="btn btn-primary"
                onClick={handleStart}
                disabled={actionLoading !== null}
              >
                {actionLoading === 'start' ? 'Resuming...' : 'Resume'}
              </button>
              <button
                className="btn btn-danger"
                onClick={handleStop}
                disabled={actionLoading !== null}
              >
                {actionLoading === 'stop' ? 'Stopping...' : 'Stop'}
              </button>
            </>
          )}
          {session.status !== 'running' && (
            <button
              className="btn btn-danger"
              onClick={() => setShowDeleteConfirm(true)}
              disabled={actionLoading !== null}
              title="Delete this session"
            >
              Delete
            </button>
          )}
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {/* Stats Overview */}
      <div className="auto-ds-stats-grid">
        <div className="stat-card">
          <div className="stat-value">{session.current_iteration}</div>
          <div className="stat-label">Iterations</div>
          <div className="stat-sublabel">of {session.max_iterations} max</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{session.total_experiments_run}</div>
          <div className="stat-label">Experiments</div>
          <div className="stat-sublabel">total run</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">
            {session.best_score ? session.best_score.toFixed(4) : '-'}
          </div>
          <div className="stat-label">Best Score</div>
          <div className="stat-sublabel">{session.best_experiment_id ? 'achieved' : 'none yet'}</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{session.iterations_without_improvement}</div>
          <div className="stat-label">Iterations w/o Improvement</div>
          <div className="stat-sublabel">of {session.plateau_iterations} allowed</div>
        </div>
      </div>

      {/* Live Progress (when running) */}
      {session.status === 'running' && progress && (
        <div className="live-progress-card">
          <h3>Live Progress</h3>
          <div className="progress-details">
            <div className="progress-item">
              <span className="progress-label">Current Status:</span>
              <span className="progress-value">
                {progress.current_iteration_status?.replace(/_/g, ' ') || 'Starting'}
              </span>
            </div>
            <div className="progress-item">
              <span className="progress-label">Iteration:</span>
              <span className="progress-value">
                {progress.current_iteration} / {progress.max_iterations}
              </span>
            </div>
            {progress.current_iteration_experiments_planned > 0 && (
              <div className="progress-item">
                <span className="progress-label">Experiments:</span>
                <span className="progress-value">
                  {progress.current_iteration_experiments_completed} / {progress.current_iteration_experiments_planned}
                </span>
              </div>
            )}
            {progress.elapsed_minutes != null && (
              <div className="progress-item">
                <span className="progress-label">Elapsed Time:</span>
                <span className="progress-value">
                  {Math.round(progress.elapsed_minutes)} min
                  {progress.time_budget_minutes && ` / ${progress.time_budget_minutes} min`}
                </span>
              </div>
            )}
          </div>
          <div className="progress-bar-container">
            <div
              className="progress-bar"
              style={{
                width: `${(progress.current_iteration / progress.max_iterations) * 100}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'progress' ? 'active' : ''}`}
          onClick={() => setActiveTab('progress')}
        >
          Session Info
        </button>
        <button
          className={`tab ${activeTab === 'iterations' ? 'active' : ''}`}
          onClick={() => setActiveTab('iterations')}
        >
          Iterations ({iterations.length})
        </button>
        <button
          className={`tab ${activeTab === 'insights' ? 'active' : ''}`}
          onClick={() => setActiveTab('insights')}
        >
          Insights ({insights.length})
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'progress' && (
          <div className="session-info-grid">
            <div className="info-section">
              <h4>Configuration</h4>
              <div className="info-list">
                <div className="info-item">
                  <span className="info-label">Accuracy Threshold:</span>
                  <span className="info-value">{session.accuracy_threshold}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Min Improvement:</span>
                  <span className="info-value">{session.min_improvement_threshold}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Plateau Iterations:</span>
                  <span className="info-value">{session.plateau_iterations}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Max Experiments/Dataset:</span>
                  <span className="info-value">{session.max_experiments_per_dataset}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Max Active Datasets:</span>
                  <span className="info-value">{session.max_active_datasets}</span>
                </div>
              </div>
            </div>

            <div className="info-section">
              <h4>Timeline</h4>
              <div className="info-list">
                <div className="info-item">
                  <span className="info-label">Created:</span>
                  <span className="info-value">{formatDate(session.created_at)}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Started:</span>
                  <span className="info-value">{formatDate(session.started_at)}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Completed:</span>
                  <span className="info-value">{formatDate(session.completed_at)}</span>
                </div>
              </div>
            </div>

            {session.description && (
              <div className="info-section full-width">
                <h4>Description</h4>
                <p>{session.description}</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'iterations' && (
          <div className="iterations-list">
            {iterations.length === 0 ? (
              <div className="empty-tab">No iterations yet. Start the session to begin.</div>
            ) : (
              iterations.map((iteration) => (
                <div key={iteration.id} className="iteration-card">
                  <div className="iteration-header">
                    <h4>Iteration {iteration.iteration_number}</h4>
                    <span
                      className="status-badge small"
                      style={{ backgroundColor: iterationStatusColors[iteration.status] }}
                    >
                      {iteration.status.replace(/_/g, ' ')}
                    </span>
                  </div>

                  <div className="iteration-stats">
                    <div className="iter-stat">
                      <span className="iter-stat-label">Experiments:</span>
                      <span className="iter-stat-value">
                        {iteration.experiments_completed} / {iteration.experiments_planned}
                        {iteration.experiments_failed > 0 && (
                          <span className="failed-count"> ({iteration.experiments_failed} failed)</span>
                        )}
                      </span>
                    </div>
                    {/* Display all 3 scores when available */}
                    <div className="iter-scores">
                      <div className="iter-stat">
                        <span className="iter-stat-label">Train:</span>
                        <span className="iter-stat-value">{iteration.best_train_score_this_iteration?.toFixed(4) ?? '-'}</span>
                      </div>
                      <div className="iter-stat">
                        <span className="iter-stat-label">Val:</span>
                        <span className="iter-stat-value">{iteration.best_val_score_this_iteration?.toFixed(4) ?? '-'}</span>
                      </div>
                      <div className="iter-stat">
                        <span className="iter-stat-label">Holdout:</span>
                        <span className="iter-stat-value" style={{fontWeight: 'bold', color: '#16a34a'}}>{iteration.best_holdout_score_this_iteration?.toFixed(4) ?? '-'}</span>
                      </div>
                    </div>
                  </div>

                  {/* Experiments List */}
                  {iteration.experiments && iteration.experiments.length > 0 && (
                    <div className="iteration-experiments">
                      <h5>Experiments</h5>
                      <div className="experiments-table">
                        <div className="exp-table-header">
                          <span className="exp-col-name">Name</span>
                          <span className="exp-col-status">Status</span>
                          <span className="exp-col-score">Train</span>
                          <span className="exp-col-score">Val</span>
                          <span className="exp-col-score">Holdout</span>
                          <span className="exp-col-link">Actions</span>
                        </div>
                        {iteration.experiments.map((exp) => (
                          <div key={exp.experiment_id} className="exp-table-row">
                            <span className="exp-col-name" title={exp.hypothesis || undefined}>
                              {exp.experiment_name}
                            </span>
                            <span className={`exp-col-status exp-status-${exp.experiment_status}`}>
                              {exp.experiment_status}
                            </span>
                            <span className="exp-col-score">
                              {exp.train_score ? exp.train_score.toFixed(4) : '-'}
                            </span>
                            <span className="exp-col-score">
                              {exp.val_score ? exp.val_score.toFixed(4) : '-'}
                            </span>
                            <span className="exp-col-score" style={{fontWeight: 'bold', color: '#16a34a'}}>
                              {exp.holdout_score ? exp.holdout_score.toFixed(4) : '-'}
                            </span>
                            <span className="exp-col-link">
                              <Link
                                to={`/experiments/${exp.experiment_id}?project=${projectId}`}
                                className="exp-view-link"
                              >
                                View
                              </Link>
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {iteration.analysis_summary_json && (
                    <div className="iteration-analysis">
                      <h5>Analysis Summary</h5>
                      <pre>{JSON.stringify(iteration.analysis_summary_json, null, 2)}</pre>
                    </div>
                  )}

                  {iteration.error_message && (
                    <div className="iteration-error">
                      <strong>Error:</strong> {iteration.error_message}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'insights' && (
          <div className="insights-list">
            {insights.length === 0 ? (
              <div className="empty-tab">No insights discovered yet.</div>
            ) : (
              insights.map((insight) => (
                <div key={insight.id} className="insight-card">
                  <div className="insight-header">
                    <span className={`insight-type ${insight.insight_type}`}>
                      {insight.insight_type.replace(/_/g, ' ')}
                    </span>
                    <span className={`insight-confidence ${insight.confidence}`}>
                      {insight.confidence} confidence
                    </span>
                  </div>

                  <h4 className="insight-title">{insight.title}</h4>

                  {insight.description && (
                    <p className="insight-description">{insight.description}</p>
                  )}

                  <div className="insight-meta">
                    <span>From iteration {insight.iteration_id ? 'completed' : 'ongoing'}</span>
                    <span>{insight.evidence_count} supporting experiments</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        isOpen={showDeleteConfirm}
        title="Delete Auto DS Session"
        message={`Are you sure you want to delete "${session?.name}"? This will delete all iterations, insights, and associated data. This action cannot be undone.`}
        confirmLabel={isDeleting ? 'Deleting...' : 'Delete'}
        onConfirm={handleDelete}
        onClose={() => setShowDeleteConfirm(false)}
        variant="danger"
      />
    </div>
  );
}

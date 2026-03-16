import { useState, useEffect, useCallback } from 'react';
import type { AgentRun, AgentStep, AgentStepType, DataSource } from '../types/api';
import {
  getAgentRun,
  runDataArchitectPipeline,
  listAgentRuns,
  deleteAgentRun,
  ApiException,
} from '../services/api';
import StatusBadge from './StatusBadge';
import AgentStepDrawer from './AgentStepDrawer';
import LoadingSpinner from './LoadingSpinner';
import ConfirmDialog from './ConfirmDialog';

interface DataArchitectPipelineProps {
  projectId: string;
  dataSources: DataSource[];
  onPipelineComplete?: () => void;
}

// Step metadata for Data Architect pipeline steps
const DATA_ARCHITECT_STEP_INFO: Record<string, { number: number; name: string; role: string; icon: string }> = {
  dataset_inventory: {
    number: 1,
    name: 'Dataset Inventory',
    role: 'Data Profiler',
    icon: '📦',
  },
  relationship_discovery: {
    number: 2,
    name: 'Relationship Discovery',
    role: 'Data Analyst',
    icon: '🔗',
  },
  training_dataset_planning: {
    number: 3,
    name: 'Training Dataset Planning',
    role: 'Data Architect',
    icon: '📐',
  },
  training_dataset_build: {
    number: 4,
    name: 'Training Dataset Build',
    role: 'Data Engineer',
    icon: '🔧',
  },
};

// Training dataset summary from the pipeline result
interface TrainingDatasetResult {
  data_source_id?: string;
  data_source_name?: string;
  row_count?: number;
  column_count?: number;
  base_table?: string;
  joined_tables?: string[];
  target_column?: string;
  feature_columns?: string[];
}

// Check if a run is a Data Architect pipeline
function isDataArchitectRun(run: AgentRun): boolean {
  return run.name?.includes('Data Architect') ||
    (run.steps?.some(
      s => s.step_type === 'dataset_inventory' || s.step_type === 'relationship_discovery'
    ) ?? false);
}

export default function DataArchitectPipeline({
  projectId,
  dataSources,
  onPipelineComplete,
}: DataArchitectPipelineProps) {
  const [allRuns, setAllRuns] = useState<AgentRun[]>([]);
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [selectedStep, setSelectedStep] = useState<AgentStep | null>(null);

  // Dialog state
  const [showDialog, setShowDialog] = useState(false);
  const [targetHint, setTargetHint] = useState('');

  // Delete confirmation state
  const [runToDelete, setRunToDelete] = useState<AgentRun | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Fetch all Data Architect pipeline runs
  const fetchAllRuns = useCallback(async () => {
    try {
      // Get all agent runs
      const runList = await listAgentRuns(projectId, 0, 50);

      // Filter to only Data Architect runs and fetch full details
      const dataArchitectRuns = runList.items.filter(isDataArchitectRun);

      // Fetch full run details for each
      const fullRuns = await Promise.all(
        dataArchitectRuns.map(run => getAgentRun(run.id))
      );

      // Sort by created_at descending (newest first)
      fullRuns.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );

      setAllRuns(fullRuns);

      // Auto-expand the newest run if it's running, or if there's only one
      if (fullRuns.length > 0) {
        const runningRun = fullRuns.find(r => r.status === 'running');
        if (runningRun) {
          setExpandedRunId(runningRun.id);
        } else if (fullRuns.length === 1) {
          setExpandedRunId(fullRuns[0].id);
        }
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  // Initial fetch
  useEffect(() => {
    fetchAllRuns();
  }, [fetchAllRuns]);

  // Poll while any pipeline is running
  useEffect(() => {
    const runningRun = allRuns.find(r => r.status === 'running');
    if (!runningRun) return;

    const interval = setInterval(async () => {
      try {
        const updatedRun = await getAgentRun(runningRun.id);
        setAllRuns(prev => prev.map(r => r.id === updatedRun.id ? updatedRun : r));

        // Check if run just completed
        if (updatedRun.status === 'completed' && onPipelineComplete) {
          onPipelineComplete();
        }
      } catch {
        // Ignore polling errors
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [allRuns, onPipelineComplete]);

  // Start the Data Architect pipeline
  const handleStartPipeline = async () => {
    setIsStarting(true);
    setError(null);
    setShowDialog(false);

    try {
      const response = await runDataArchitectPipeline(projectId, {
        target_hint: targetHint.trim() || undefined,
        run_async: false,
      });

      // Fetch the created run
      const newRun = await getAgentRun(response.agent_run_id);
      setAllRuns(prev => [newRun, ...prev]);
      setExpandedRunId(newRun.id);
      setTargetHint('');

      // Notify parent
      if (newRun.status === 'completed' && onPipelineComplete) {
        onPipelineComplete();
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to start Data Architect pipeline');
      }
    } finally {
      setIsStarting(false);
    }
  };

  // Delete a run
  const handleDeleteRun = async () => {
    if (!runToDelete) return;

    setIsDeleting(true);
    try {
      await deleteAgentRun(runToDelete.id);
      setAllRuns(prev => prev.filter(r => r.id !== runToDelete.id));
      setRunToDelete(null);
      if (expandedRunId === runToDelete.id) {
        setExpandedRunId(null);
      }
      // Refresh data sources since we may have deleted one
      if (onPipelineComplete) {
        onPipelineComplete();
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to delete run');
      }
    } finally {
      setIsDeleting(false);
    }
  };

  // Get training dataset result from pipeline
  const getTrainingDatasetResult = (run: AgentRun): TrainingDatasetResult | null => {
    if (!run?.result_json) return null;

    // Try both structures
    const result = run.result_json as TrainingDatasetResult & { training_dataset?: TrainingDatasetResult };
    return result.training_dataset || result;
  };

  // Format date for display
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Get step info for display
  const getStepInfo = (stepType: AgentStepType) => {
    return DATA_ARCHITECT_STEP_INFO[stepType] || {
      number: 0,
      name: stepType,
      role: 'Agent',
      icon: '🤖',
    };
  };

  // Check if we can run the pipeline
  const canRunPipeline = dataSources.length >= 1;

  if (isLoading) {
    return (
      <div className="data-architect-section">
        <LoadingSpinner message="Checking for existing builds..." />
      </div>
    );
  }

  // Don't show if no data sources
  if (dataSources.length === 0) {
    return null;
  }

  return (
    <div className="data-architect-section">
      <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h4 style={{ margin: 0 }}>Step 1: Combine Multiple Tables</h4>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.875rem', color: '#6b7280' }}>
            AI discovers relationships between your {dataSources.length} tables and joins them into a single training dataset
          </p>
        </div>
        {canRunPipeline && (
          <button
            className="btn btn-primary"
            onClick={() => setShowDialog(true)}
            disabled={isStarting}
          >
            {isStarting ? (
              <>
                <span className="spinner spinner-small"></span>
                Building...
              </>
            ) : (
              <>
                <span className="btn-icon">🔧</span>
                {allRuns.length > 0 ? 'Build New Dataset' : 'Build Training Dataset'}
              </>
            )}
          </button>
        )}
      </div>

      {error && (
        <div className="form-error" style={{ marginBottom: '1rem' }}>
          {error}
          <button onClick={() => setError(null)} style={{ marginLeft: '1rem' }}>
            Dismiss
          </button>
        </div>
      )}

      {/* List of All Runs */}
      {allRuns.length > 0 && (
        <div className="runs-list" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {allRuns.map((run) => {
            const isExpanded = expandedRunId === run.id;
            const result = getTrainingDatasetResult(run);

            return (
              <div
                key={run.id}
                className="run-card"
                style={{
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  overflow: 'hidden',
                  backgroundColor: run.status === 'completed' ? '#fafff9' : run.status === 'failed' ? '#fff5f5' : '#fff',
                }}
              >
                {/* Run Header - Always Visible */}
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '0.75rem 1rem',
                    cursor: 'pointer',
                    backgroundColor: isExpanded ? '#f9fafb' : 'transparent',
                  }}
                  onClick={() => setExpandedRunId(isExpanded ? null : run.id)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{ fontSize: '1.25rem' }}>
                      {run.status === 'completed' ? '✅' : run.status === 'failed' ? '❌' : run.status === 'running' ? '⏳' : '📋'}
                    </span>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>
                        {result?.data_source_name || run.name || 'Training Dataset Build'}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                        {formatDate(run.created_at)}
                        {result?.row_count && ` • ${result.row_count.toLocaleString()} rows`}
                        {result?.column_count && ` • ${result.column_count} columns`}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <StatusBadge status={run.status} />
                    {run.status !== 'running' && (
                      <button
                        className="btn btn-small btn-danger"
                        onClick={(e) => {
                          e.stopPropagation();
                          setRunToDelete(run);
                        }}
                        title="Delete this run and its data"
                        style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                      >
                        Delete
                      </button>
                    )}
                    <span style={{ color: '#9ca3af', fontSize: '1rem' }}>
                      {isExpanded ? '▼' : '▶'}
                    </span>
                  </div>
                </div>

                {/* Expanded Content */}
                {isExpanded && (
                  <div style={{ padding: '1rem', borderTop: '1px solid #e5e7eb' }}>
                    {/* Pipeline Timeline */}
                    <div className="pipeline-timeline" style={{ marginBottom: '1rem' }}>
                      {run.status === 'running' && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#3b82f6', marginBottom: '0.75rem' }}>
                          <span className="spinner spinner-small"></span>
                          Building dataset...
                        </div>
                      )}

                      <div className="timeline-steps" style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        {run.steps
                          ?.filter(step => Object.keys(DATA_ARCHITECT_STEP_INFO).includes(step.step_type))
                          .sort((a, b) => getStepInfo(a.step_type).number - getStepInfo(b.step_type).number)
                          .map((step) => {
                            const info = getStepInfo(step.step_type);
                            return (
                              <div
                                key={step.id}
                                className={`timeline-step ${step.status}`}
                                onClick={(e) => { e.stopPropagation(); setSelectedStep(step); }}
                                style={{
                                  flex: '1 1 120px',
                                  padding: '0.5rem',
                                  border: '1px solid #e5e7eb',
                                  borderRadius: '6px',
                                  cursor: 'pointer',
                                  backgroundColor: step.status === 'completed' ? '#f0fdf4' : step.status === 'running' ? '#eff6ff' : step.status === 'failed' ? '#fef2f2' : '#fff',
                                }}
                              >
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', marginBottom: '0.25rem' }}>
                                  <span style={{ fontSize: '0.875rem' }}>{info.icon}</span>
                                  <span style={{ fontWeight: 600, fontSize: '0.75rem' }}>{info.name}</span>
                                </div>
                                <StatusBadge status={step.status} />
                              </div>
                            );
                          })}
                      </div>
                    </div>

                    {/* Training Dataset Summary */}
                    {run.status === 'completed' && result && (
                      <div style={{
                        background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
                        border: '1px solid #86efac',
                        padding: '1rem',
                        borderRadius: '6px',
                      }}>
                        <div className="dataset-summary-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '0.75rem' }}>
                          {result.base_table && (
                            <div className="summary-item">
                              <span style={{ fontSize: '0.7rem', color: '#6b7280', textTransform: 'uppercase' }}>Base Table</span>
                              <span style={{ fontWeight: 600, fontSize: '0.875rem', display: 'block' }}>{result.base_table}</span>
                            </div>
                          )}
                          {result.target_column && (
                            <div className="summary-item">
                              <span style={{ fontSize: '0.7rem', color: '#6b7280', textTransform: 'uppercase' }}>Target</span>
                              <span style={{ fontWeight: 600, fontSize: '0.875rem', display: 'block' }}>{result.target_column}</span>
                            </div>
                          )}
                          {result.feature_columns && result.feature_columns.length > 0 && (
                            <div className="summary-item">
                              <span style={{ fontSize: '0.7rem', color: '#6b7280', textTransform: 'uppercase' }}>Features</span>
                              <span style={{ fontWeight: 600, fontSize: '0.875rem', display: 'block' }}>{result.feature_columns.length} columns</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Failed run message */}
                    {run.status === 'failed' && run.error_message && (
                      <div style={{ background: '#fef2f2', border: '1px solid #fecaca', padding: '0.75rem', borderRadius: '6px' }}>
                        <h5 style={{ color: '#dc2626', marginBottom: '0.25rem', fontSize: '0.875rem' }}>Pipeline Failed</h5>
                        <p style={{ color: '#7f1d1d', margin: 0, fontSize: '0.75rem' }}>{run.error_message}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* No runs message */}
      {allRuns.length === 0 && !isLoading && (
        <div style={{
          padding: '2rem',
          textAlign: 'center',
          color: '#6b7280',
          border: '2px dashed #e5e7eb',
          borderRadius: '8px',
          marginTop: '1rem',
        }}>
          <p style={{ margin: 0 }}>No training datasets built yet. Click &quot;Build Training Dataset&quot; to start.</p>
        </div>
      )}

      {/* Start Dialog */}
      {showDialog && (
        <div className="modal-backdrop" onClick={() => setShowDialog(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '500px' }}>
            <div className="modal-header">
              <h3>Combine {dataSources.length} Tables into Training Dataset</h3>
              <button
                className="modal-close"
                onClick={() => setShowDialog(false)}
                aria-label="Close"
              >
                &times;
              </button>
            </div>

            <div className="modal-body">
              <p style={{ color: '#374151', marginBottom: '1rem' }}>
                The AI will automatically join your {dataSources.length} tables ({dataSources.map(ds => ds.name).join(', ')}) by:
              </p>
              <ol style={{ paddingLeft: '1.25rem', color: '#374151', marginBottom: '1.5rem' }}>
                <li>Profiling each table and identifying keys</li>
                <li>Discovering foreign key relationships</li>
                <li>Planning optimal joins with feature aggregations</li>
                <li>Building a single training dataset</li>
              </ol>

              <div className="form-group">
                <label htmlFor="target-hint" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>
                  Target hint (optional)
                </label>
                <input
                  type="text"
                  id="target-hint"
                  className="form-input"
                  value={targetHint}
                  onChange={(e) => setTargetHint(e.target.value)}
                  placeholder="e.g., predict customer churn, forecast sales"
                  style={{ width: '100%', padding: '0.5rem', border: '1px solid #d1d5db', borderRadius: '6px' }}
                />
                <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
                  Help the AI understand what you want to predict
                </p>
              </div>
            </div>

            <div className="modal-footer" style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', padding: '1rem', borderTop: '1px solid #e5e7eb' }}>
              <button
                className="btn btn-secondary"
                onClick={() => setShowDialog(false)}
                disabled={isStarting}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleStartPipeline}
                disabled={isStarting}
              >
                {isStarting ? (
                  <>
                    <span className="spinner spinner-small"></span>
                    Starting...
                  </>
                ) : (
                  'Start Building'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Step Drawer */}
      {selectedStep && (
        <AgentStepDrawer
          step={selectedStep}
          stepInfo={getStepInfo(selectedStep.step_type)}
          projectId={projectId}
          onClose={() => setSelectedStep(null)}
        />
      )}

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        isOpen={runToDelete !== null}
        title="Delete Pipeline Run"
        message={`Are you sure you want to delete this pipeline run and its training dataset? This action cannot be undone.`}
        confirmLabel={isDeleting ? 'Deleting...' : 'Delete'}
        onConfirm={handleDeleteRun}
        onClose={() => setRunToDelete(null)}
        variant="danger"
      />
    </div>
  );
}

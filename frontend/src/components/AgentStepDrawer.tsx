import { useState, useEffect, useRef, useCallback } from 'react';
import type { AgentStep, AgentStepLog, LogMessageType } from '../types/api';
import {
  getAgentStepLogs,
  applyDatasetSpecFromStep,
  applyExperimentPlanFromStep,
  ApiException,
} from '../services/api';
import StatusBadge from './StatusBadge';

interface AgentStepDrawerProps {
  step: AgentStep;
  stepInfo: { number: number; name: string; role: string; icon: string };
  projectId: string;
  onClose: () => void;
  onActionComplete?: (action: 'dataset_spec' | 'experiment', resourceId: string) => void;
}

// Log type styling - categorized for toggle visibility
const LOG_TYPE_STYLES: Record<LogMessageType, { label: string; color: string; icon: string; isThinkingType: boolean }> = {
  // Visible by default
  info: { label: 'Info', color: '#3498db', icon: 'ℹ️', isThinkingType: false },
  warning: { label: 'Warning', color: '#f39c12', icon: '⚠️', isThinkingType: false },
  error: { label: 'Error', color: '#e74c3c', icon: '❌', isThinkingType: false },
  summary: { label: 'Summary', color: '#2ecc71', icon: '✅', isThinkingType: false },
  // Thinking types - hidden by default, shown via toggle
  thinking: { label: 'Thinking', color: '#7f8c8d', icon: '🧠', isThinkingType: true },
  hypothesis: { label: 'Hypothesis', color: '#8e44ad', icon: '💡', isThinkingType: true },
  action: { label: 'Action', color: '#3498db', icon: '⚙️', isThinkingType: true },
  thought: { label: 'Thought', color: '#9b59b6', icon: '💭', isThinkingType: true },  // Legacy
};

export default function AgentStepDrawer({
  step,
  stepInfo,
  projectId,
  onClose,
  onActionComplete,
}: AgentStepDrawerProps) {
  const [logs, setLogs] = useState<AgentStepLog[]>([]);
  const [lastSequence, setLastSequence] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [activeTab, setActiveTab] = useState<'logs' | 'output' | 'input'>('logs');
  const [showFullThinking, setShowFullThinking] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const drawerRef = useRef<HTMLDivElement>(null);

  // Fetch logs with polling
  const fetchLogs = useCallback(async (sinceSequence: number) => {
    try {
      const response = await getAgentStepLogs(step.id, sinceSequence);
      if (response.logs.length > 0) {
        setLogs(prev => {
          // Merge new logs, avoiding duplicates
          const existingIds = new Set(prev.map(l => l.id));
          const newLogs = response.logs.filter(l => !existingIds.has(l.id));
          return [...prev, ...newLogs];
        });
        setLastSequence(response.last_sequence);
      }
      setHasMore(response.has_more);
    } catch {
      // Ignore fetch errors during polling
    }
  }, [step.id]);

  // Initial fetch
  useEffect(() => {
    setLogs([]);
    setLastSequence(0);
    setHasMore(true);
    fetchLogs(0);
  }, [step.id, fetchLogs]);

  // Poll for new logs while step is running
  useEffect(() => {
    if (!hasMore || step.status === 'completed' || step.status === 'failed') return;

    const interval = setInterval(() => {
      fetchLogs(lastSequence);
    }, 2000);

    return () => clearInterval(interval);
  }, [hasMore, step.status, lastSequence, fetchLogs]);

  // Scroll to bottom when new logs arrive
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  // Close on backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === drawerRef.current) {
      onClose();
    }
  };

  const formatTimestamp = (ts: string) => {
    return new Date(ts).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const formatDuration = (start?: string, end?: string) => {
    if (!start) return null;
    const startTime = new Date(start).getTime();
    const endTime = end ? new Date(end).getTime() : Date.now();
    const durationMs = endTime - startTime;
    const seconds = Math.floor(durationMs / 1000);
    const minutes = Math.floor(seconds / 60);
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  };

  const renderJsonValue = (value: unknown): React.ReactNode => {
    if (value === null || value === undefined) {
      return <span className="json-null">null</span>;
    }
    if (typeof value === 'string') {
      // Multi-line strings get special treatment
      if (value.includes('\n')) {
        return <pre className="json-string-multiline">{value}</pre>;
      }
      return <span className="json-string">"{value}"</span>;
    }
    if (typeof value === 'number') {
      return <span className="json-number">{value}</span>;
    }
    if (typeof value === 'boolean') {
      return <span className="json-boolean">{value.toString()}</span>;
    }
    if (Array.isArray(value)) {
      return (
        <span className="json-array">
          [{value.map((item, i) => (
            <span key={i}>
              {renderJsonValue(item)}
              {i < value.length - 1 && ', '}
            </span>
          ))}]
        </span>
      );
    }
    if (typeof value === 'object') {
      return (
        <div className="json-object">
          {Object.entries(value).map(([key, val]) => (
            <div key={key} className="json-property">
              <span className="json-key">"{key}"</span>: {renderJsonValue(val)}
            </div>
          ))}
        </div>
      );
    }
    return <>{String(value)}</>;
  };

  // Check if this is a debate step
  const isDebateStep = ['gemini_critique', 'openai_judge', 'debate_round'].includes(step.step_type);

  // Extract summary text - for debate steps use content, otherwise natural_language_summary
  let summaryText: string | null = null;
  if (step.output_json) {
    // For debate steps, use content instead of natural_language_summary
    if (isDebateStep && typeof step.output_json.content === 'string') {
      summaryText = step.output_json.content;
    } else if (typeof step.output_json.natural_language_summary === 'string') {
      summaryText = step.output_json.natural_language_summary;
    }
  }
  const outputWithoutSummary = step.output_json
    ? Object.fromEntries(
        Object.entries(step.output_json).filter(([k]) =>
          k !== 'natural_language_summary' && (!isDebateStep || k !== 'content')
        )
      )
    : null;

  // Determine if this step can have actions applied
  const canApplyDatasetSpec =
    step.step_type === 'dataset_design' && step.status === 'completed';
  const canApplyExperimentPlan =
    step.step_type === 'experiment_design' && step.status === 'completed';

  // Handle applying dataset spec from step
  const handleApplyDatasetSpec = async () => {
    setIsApplying(true);
    setActionError(null);
    setActionSuccess(null);

    try {
      const response = await applyDatasetSpecFromStep(projectId, step.id);
      setActionSuccess(response.message);
      onActionComplete?.('dataset_spec', response.dataset_spec_id);
    } catch (err) {
      if (err instanceof ApiException) {
        setActionError(err.detail);
      } else {
        setActionError('Failed to create DatasetSpec');
      }
    } finally {
      setIsApplying(false);
    }
  };

  // Handle applying experiment plan from step
  const handleApplyExperimentPlan = async (variant?: string) => {
    setIsApplying(true);
    setActionError(null);
    setActionSuccess(null);

    try {
      const response = await applyExperimentPlanFromStep(projectId, step.id, undefined, variant);
      setActionSuccess(response.message);
      onActionComplete?.('experiment', response.experiment_id);
    } catch (err) {
      if (err instanceof ApiException) {
        setActionError(err.detail);
      } else {
        setActionError('Failed to create Experiment');
      }
    } finally {
      setIsApplying(false);
    }
  };

  return (
    <div className="drawer-backdrop" ref={drawerRef} onClick={handleBackdropClick}>
      <div className="drawer">
        <div className="drawer-header">
          <div className="drawer-title">
            <span className="step-icon">{stepInfo.icon}</span>
            <span>{stepInfo.number}. {stepInfo.name}</span>
            <StatusBadge status={step.status} />
          </div>
          <button className="drawer-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        <div className="drawer-meta">
          <div className="meta-item">
            <span className="meta-label">Agent Role:</span>
            <span className="meta-value">{stepInfo.role}</span>
          </div>
          {step.started_at && (
            <div className="meta-item">
              <span className="meta-label">Duration:</span>
              <span className="meta-value">
                {formatDuration(step.started_at, step.finished_at)}
                {step.status === 'running' && ' (running)'}
              </span>
            </div>
          )}
          {step.retry_count > 0 && (
            <div className="meta-item">
              <span className="meta-label">Retries:</span>
              <span className="meta-value">{step.retry_count}</span>
            </div>
          )}
        </div>

        {/* Debate Step Metadata - shown for debate steps */}
        {isDebateStep && step.output_json && (
          <div className="drawer-debate-meta" style={{
            margin: '0 1rem 1rem 1rem',
            padding: '0.75rem',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            borderRadius: '8px',
            border: '1px solid rgba(99, 102, 241, 0.3)',
          }}>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
              {step.output_json.round !== undefined && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #a0a0b0)' }}>Round:</span>
                  <span style={{ fontWeight: 600, color: 'var(--text-primary, #e0e0e0)' }}>{String(step.output_json.round)}</span>
                </div>
              )}
              {(step.output_json.agrees as boolean) !== undefined && (
                <div style={{
                  padding: '0.25rem 0.5rem',
                  borderRadius: '4px',
                  backgroundColor: (step.output_json.agrees as boolean) ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                  color: (step.output_json.agrees as boolean) ? '#22c55e' : '#ef4444',
                  fontSize: '0.875rem',
                  fontWeight: 600,
                }}>
                  {(step.output_json.agrees as boolean) ? '✓ Agrees with proposal' : '✗ Disagrees with proposal'}
                </div>
              )}
              {(step.output_json.confidence as number) !== undefined && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #a0a0b0)' }}>Confidence:</span>
                  <span style={{ fontWeight: 600, color: 'var(--text-primary, #e0e0e0)' }}>{Math.round((step.output_json.confidence as number) * 100)}%</span>
                </div>
              )}
              {step.input_json?.debate_for !== undefined && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #a0a0b0)' }}>Debating:</span>
                  <span style={{ fontWeight: 600, color: 'var(--text-primary, #e0e0e0)' }}>{String(step.input_json.debate_for)}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Summary Section */}
        {summaryText && summaryText.length > 0 && (
          <div className="drawer-summary">
            <h4>{isDebateStep ? 'Full Message' : 'Summary'}</h4>
            <p style={{ whiteSpace: 'pre-wrap' }}>{summaryText}</p>
          </div>
        )}

        {/* Error Message */}
        {step.error_message && (
          <div className="drawer-error">
            <h4>Error</h4>
            <p>{step.error_message}</p>
          </div>
        )}

        {/* Leakage Candidates Section (Prompt 6) - for data_audit step */}
        {step.step_type === 'data_audit' && step.status === 'completed' && step.output_json?.leakage_candidates && (
          (() => {
            const candidates = step.output_json.leakage_candidates as Array<{
              column: string;
              reason: string;
              severity: string;
              detection_method: string;
            }>;
            if (candidates.length === 0) return null;
            const highSeverity = candidates.filter(c => c.severity === 'high');
            const mediumSeverity = candidates.filter(c => c.severity === 'medium');
            return (
              <div className="drawer-leakage-candidates" style={{
                margin: '1rem 0',
                padding: '1rem',
                backgroundColor: highSeverity.length > 0 ? 'rgba(239, 68, 68, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                borderLeft: `4px solid ${highSeverity.length > 0 ? '#ef4444' : '#f59e0b'}`,
                borderRadius: '0.5rem',
              }}>
                <h4 style={{ margin: '0 0 0.5rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>🔓</span>
                  Potential Leakage Features ({candidates.length})
                </h4>
                <p style={{ fontSize: '0.875rem', color: '#9ca3af', margin: '0 0 0.75rem 0' }}>
                  These features may cause data leakage. Review before using in models.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {candidates.map((candidate, idx) => (
                    <div key={idx} style={{
                      padding: '0.5rem',
                      backgroundColor: candidate.severity === 'high' ? 'rgba(239, 68, 68, 0.1)' :
                                       candidate.severity === 'medium' ? 'rgba(245, 158, 11, 0.1)' :
                                       'rgba(34, 197, 94, 0.1)',
                      borderRadius: '0.25rem',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '0.5rem',
                    }}>
                      <span style={{
                        fontSize: '0.75rem',
                        padding: '0.125rem 0.5rem',
                        borderRadius: '0.25rem',
                        fontWeight: 'bold',
                        backgroundColor: candidate.severity === 'high' ? '#ef4444' :
                                        candidate.severity === 'medium' ? '#f59e0b' : '#22c55e',
                        color: 'white',
                        flexShrink: 0,
                      }}>
                        {candidate.severity.toUpperCase()}
                      </span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontFamily: 'monospace', fontWeight: 'bold', color: '#e5e7eb' }}>
                          {candidate.column}
                        </div>
                        <div style={{ fontSize: '0.875rem', color: '#9ca3af' }}>
                          {candidate.reason}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
                          Detected by: {candidate.detection_method}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                {highSeverity.length > 0 && (
                  <div style={{
                    marginTop: '0.75rem',
                    padding: '0.5rem',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderRadius: '0.25rem',
                    fontSize: '0.875rem',
                    color: '#f87171',
                  }}>
                    ⚠️ {highSeverity.length} high-severity feature(s) detected. Consider excluding these from your models.
                  </div>
                )}
              </div>
            );
          })()
        )}

        {/* Action Buttons */}
        {(canApplyDatasetSpec || canApplyExperimentPlan) && (
          <div className="drawer-actions">
            <h4>Actions</h4>
            {actionError && (
              <div className="action-error">
                {actionError}
                <button onClick={() => setActionError(null)} className="btn-dismiss-small">
                  &times;
                </button>
              </div>
            )}
            {actionSuccess && (
              <div className="action-success">
                {actionSuccess}
                <button onClick={() => setActionSuccess(null)} className="btn-dismiss-small">
                  &times;
                </button>
              </div>
            )}
            {canApplyDatasetSpec && (
              <button
                className="btn btn-primary action-btn"
                onClick={handleApplyDatasetSpec}
                disabled={isApplying}
              >
                {isApplying ? (
                  <>
                    <span className="spinner spinner-small"></span>
                    Creating...
                  </>
                ) : (
                  'Create DatasetSpec from this'
                )}
              </button>
            )}
            {canApplyExperimentPlan && (
              <div className="experiment-actions">
                <button
                  className="btn btn-primary action-btn"
                  onClick={() => handleApplyExperimentPlan()}
                  disabled={isApplying}
                >
                  {isApplying ? (
                    <>
                      <span className="spinner spinner-small"></span>
                      Creating...
                    </>
                  ) : (
                    'Create Experiment with Recommended Plan'
                  )}
                </button>
                {(() => {
                  const variants = step.output_json?.variants;
                  if (!variants || !Array.isArray(variants)) return null;
                  const recommendedVariant = step.output_json?.recommended_variant as string | undefined;
                  return (
                    <div className="variant-buttons">
                      <span className="variant-label">Or select a variant:</span>
                      {(variants as Array<{ name: string }>).map((v) => {
                        const isRecommended = v.name === recommendedVariant;
                        return (
                          <button
                            key={v.name}
                            className={`btn btn-secondary btn-small ${isRecommended ? 'recommended' : ''}`}
                            onClick={() => handleApplyExperimentPlan(v.name)}
                            disabled={isApplying}
                          >
                            {v.name}
                            {isRecommended && ' (recommended)'}
                          </button>
                        );
                      })}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        )}

        {/* Tabs */}
        <div className="drawer-tabs">
          <button
            className={`drawer-tab ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
          >
            Logs ({logs.length})
          </button>
          <button
            className={`drawer-tab ${activeTab === 'output' ? 'active' : ''}`}
            onClick={() => setActiveTab('output')}
            disabled={!step.output_json}
          >
            Output
          </button>
          <button
            className={`drawer-tab ${activeTab === 'input' ? 'active' : ''}`}
            onClick={() => setActiveTab('input')}
            disabled={!step.input_json}
          >
            Input
          </button>
        </div>

        <div className="drawer-content">
          {/* Logs Tab */}
          {activeTab === 'logs' && (
            <div className="logs-container">
              {logs.length === 0 && step.status === 'pending' && (
                <div className="logs-empty">
                  Step hasn't started yet. Logs will appear here once it begins.
                </div>
              )}
              {logs.length === 0 && step.status === 'running' && (
                <div className="logs-loading">
                  <span className="spinner spinner-small"></span>
                  Waiting for logs...
                </div>
              )}
              {/* Show full thinking toggle */}
              {(() => {
                const thinkingCount = logs.filter(l => LOG_TYPE_STYLES[l.message_type]?.isThinkingType).length;
                if (thinkingCount === 0) return null;
                return (
                  <div className="thinking-toggle">
                    <button
                      className={`btn btn-small ${showFullThinking ? 'btn-secondary' : 'btn-outline'}`}
                      onClick={() => setShowFullThinking(!showFullThinking)}
                    >
                      {showFullThinking ? '🧠 Hide thinking' : `🧠 Show full thinking (${thinkingCount})`}
                    </button>
                  </div>
                );
              })()}
              {logs.map((log) => {
                const style = LOG_TYPE_STYLES[log.message_type] ?? { label: log.message_type, color: '#999', icon: '📝', isThinkingType: false };
                // Filter out thinking logs if toggle is off
                if (style.isThinkingType && !showFullThinking) {
                  return null;
                }
                return (
                  <div key={log.id} className={`log-entry log-${log.message_type} ${style.isThinkingType ? 'log-thinking-type' : ''}`}>
                    <div className="log-header">
                      <span className="log-icon">{style.icon}</span>
                      <span className="log-type" style={{ color: style.color }}>
                        {style.label}
                      </span>
                      <span className="log-time">{formatTimestamp(log.timestamp)}</span>
                    </div>
                    <div className="log-message">{log.message}</div>
                    {log.metadata_json && Object.keys(log.metadata_json).length > 0 && (
                      <details className="log-metadata">
                        <summary>Metadata</summary>
                        <pre>{JSON.stringify(log.metadata_json, null, 2)}</pre>
                      </details>
                    )}
                  </div>
                );
              })}
              {step.status === 'running' && hasMore && (
                <div className="logs-streaming">
                  <span className="spinner spinner-small"></span>
                  Streaming logs...
                </div>
              )}
              <div ref={logsEndRef} />
            </div>
          )}

          {/* Output Tab */}
          {activeTab === 'output' && step.output_json && (
            <div className="json-viewer">
              {outputWithoutSummary && Object.keys(outputWithoutSummary).length > 0 ? (
                <div className="json-tree">
                  {Object.entries(outputWithoutSummary).map(([key, value]) => (
                    <div key={key} className="json-root-property">
                      <span className="json-key">{key}</span>
                      <div className="json-value-wrapper">
                        {renderJsonValue(value)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="json-empty">No structured output available.</div>
              )}
            </div>
          )}

          {/* Input Tab */}
          {activeTab === 'input' && step.input_json && (
            <div className="json-viewer">
              <div className="json-tree">
                {Object.entries(step.input_json).map(([key, value]) => (
                  <div key={key} className="json-root-property">
                    <span className="json-key">{key}</span>
                    <div className="json-value-wrapper">
                      {renderJsonValue(value)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

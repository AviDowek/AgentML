import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import type { AgentRun, AgentStep, AgentStepType } from '../types/api';
import {
  listExperimentAgentRuns,
  getExperimentAgentRun,
  runResultsPipeline,
  promoteModel,
  triggerImprovement,
  getImprovementStatus,
  getAutoIterateSettings,
  updateAutoIterateSettings,
  ApiException,
  type ImprovementStatusResponse,
  type AutoIterateSettingsResponse,
} from '../services/api';
import StatusBadge from './StatusBadge';
import AgentStepDrawer from './AgentStepDrawer';
import LoadingSpinner from './LoadingSpinner';
import NotebookViewer from './NotebookViewer';

interface ExperimentResultsPipelineProps {
  experimentId: string;
  experimentStatus: string;
  onModelPromoted?: () => void;
}

// Step metadata for results pipeline steps
const RESULTS_STEP_INFO: Record<string, { number: number; name: string; role: string; icon: string }> = {
  results_interpretation: {
    number: 1,
    name: 'Results Interpretation',
    role: 'Analyst',
    icon: '📊',
  },
  results_critic: {
    number: 2,
    name: 'Results Critic',
    role: 'Reviewer',
    icon: '🔍',
  },
  // Simple improvement pipeline
  improvement_analysis: {
    number: 1,
    name: 'Improvement Analysis',
    role: 'ML Engineer',
    icon: '🔬',
  },
  improvement_plan: {
    number: 2,
    name: 'Improvement Plan',
    role: 'Architect',
    icon: '📋',
  },
  // Enhanced improvement pipeline (full agent)
  iteration_context: {
    number: 1,
    name: 'Iteration Context',
    role: 'Historian',
    icon: '📚',
  },
  improvement_data_analysis: {
    number: 2,
    name: 'Data Re-Analysis',
    role: 'Data Scientist',
    icon: '🔬',
  },
  improvement_dataset_design: {
    number: 3,
    name: 'Dataset Redesign',
    role: 'Feature Engineer',
    icon: '🛠️',
  },
  feature_validation: {
    number: 4,
    name: 'Feature Validation',
    role: 'Quality Assurance',
    icon: '✅',
  },
  improvement_experiment_design: {
    number: 5,
    name: 'Experiment Design',
    role: 'ML Architect',
    icon: '📐',
  },
};

// Type for the recommendation output
interface Recommendation {
  recommended_model_id?: string;
  reason?: string;
}

// Type for critic findings
interface CriticFindings {
  severity?: string;
  issues?: string[];
  approved?: boolean;
}

export default function ExperimentResultsPipeline({
  experimentId,
  experimentStatus,
  onModelPromoted,
}: ExperimentResultsPipelineProps) {
  const navigate = useNavigate();
  const [agentRun, setAgentRun] = useState<AgentRun | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [selectedStep, setSelectedStep] = useState<AgentStep | null>(null);
  const [isPromoting, setIsPromoting] = useState(false);
  const [promoteError, setPromoteError] = useState<string | null>(null);
  const [promoteSuccess, setPromoteSuccess] = useState<string | null>(null);

  // Auto-improve state
  const [isImproving, setIsImproving] = useState(false);
  const [improvementStatus, setImprovementStatus] = useState<ImprovementStatusResponse | null>(null);
  const [improveError, setImproveError] = useState<string | null>(null);

  // Auto-iterate state
  const [autoIterateSettings, setAutoIterateSettings] = useState<AutoIterateSettingsResponse | null>(null);
  const [autoIterateLoading, setAutoIterateLoading] = useState(false);
  const [autoIterateError, setAutoIterateError] = useState<string | null>(null);
  const [pendingMaxIterations, setPendingMaxIterations] = useState<number>(5);

  // Fetch latest results pipeline run
  const fetchLatestRun = useCallback(async () => {
    try {
      const runList = await listExperimentAgentRuns(experimentId, 0, 1);
      if (runList.items.length > 0) {
        const latestRun = await getExperimentAgentRun(experimentId, runList.items[0].id);
        setAgentRun(latestRun);
      } else {
        setAgentRun(null);
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setIsLoading(false);
    }
  }, [experimentId]);

  // Initial fetch
  useEffect(() => {
    fetchLatestRun();
  }, [fetchLatestRun]);

  // Poll while pipeline is running
  useEffect(() => {
    if (!agentRun || agentRun.status === 'completed' || agentRun.status === 'failed') {
      return;
    }

    const interval = setInterval(fetchLatestRun, 3000);
    return () => clearInterval(interval);
  }, [agentRun, fetchLatestRun]);

  // Start the results pipeline
  const handleStartPipeline = async () => {
    setIsStarting(true);
    setError(null);

    try {
      await runResultsPipeline(experimentId, { run_async: false });
      await fetchLatestRun();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to start results pipeline');
      }
    } finally {
      setIsStarting(false);
    }
  };

  // Start the auto-improve pipeline
  const handleImprove = async () => {
    console.log('handleImprove called, setting isImproving to true');
    setIsImproving(true);
    setImproveError(null);
    setImprovementStatus(null);

    try {
      console.log('Calling triggerImprovement API...');
      const response = await triggerImprovement(experimentId);
      console.log('triggerImprovement response:', response);
      // Start polling for improvement status
      pollImprovementStatus();
    } catch (err) {
      console.error('triggerImprovement error:', err);
      if (err instanceof ApiException) {
        setImproveError(err.detail);
      } else {
        setImproveError('Failed to start improvement pipeline');
      }
      setIsImproving(false);
    }
  };

  // Poll improvement status
  const pollImprovementStatus = async () => {
    try {
      const status = await getImprovementStatus(experimentId);
      setImprovementStatus(status);

      if (status.status === 'completed' && status.result?.new_experiment_id) {
        // Navigate to the new experiment
        setIsImproving(false);
        navigate(`/experiments/${status.result.new_experiment_id}`);
      } else if (status.status === 'failed') {
        setImproveError(status.error_message || 'Improvement pipeline failed');
        setIsImproving(false);
      } else if (status.status === 'running') {
        // Continue polling
        setTimeout(pollImprovementStatus, 2000);
      } else {
        setIsImproving(false);
      }
    } catch (err) {
      console.error('Error polling improvement status:', err);
      // Continue polling on error
      setTimeout(pollImprovementStatus, 3000);
    }
  };

  // Check for existing improvement run on mount
  useEffect(() => {
    const checkExistingImprovement = async () => {
      try {
        const status = await getImprovementStatus(experimentId);
        if (status.has_improvement_run && status.status === 'running') {
          setIsImproving(true);
          setImprovementStatus(status);
          pollImprovementStatus();
        }
      } catch {
        // Ignore errors
      }
    };
    if (experimentStatus === 'completed') {
      checkExistingImprovement();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experimentId, experimentStatus]);

  // Fetch auto-iterate settings on mount
  useEffect(() => {
    const fetchAutoIterateSettings = async () => {
      try {
        const settings = await getAutoIterateSettings(experimentId);
        setAutoIterateSettings(settings);
        setPendingMaxIterations(settings.auto_iterate_max);
      } catch {
        // Ignore errors - settings may not exist yet
      }
    };
    fetchAutoIterateSettings();
  }, [experimentId]);

  // Handle auto-iterate toggle
  const handleAutoIterateToggle = async () => {
    setAutoIterateLoading(true);
    setAutoIterateError(null);
    try {
      const newEnabled = !autoIterateSettings?.auto_iterate_enabled;
      const settings = await updateAutoIterateSettings(experimentId, {
        enabled: newEnabled,
        max_iterations: pendingMaxIterations,
      });
      setAutoIterateSettings(settings);
    } catch (err) {
      if (err instanceof ApiException) {
        setAutoIterateError(err.detail);
      } else {
        setAutoIterateError('Failed to update auto-iterate settings');
      }
    } finally {
      setAutoIterateLoading(false);
    }
  };

  // Handle max iterations change
  const handleMaxIterationsChange = async (value: number) => {
    setPendingMaxIterations(value);
    if (autoIterateSettings?.auto_iterate_enabled) {
      // If already enabled, update immediately
      setAutoIterateLoading(true);
      try {
        const settings = await updateAutoIterateSettings(experimentId, {
          enabled: true,
          max_iterations: value,
        });
        setAutoIterateSettings(settings);
      } catch (err) {
        if (err instanceof ApiException) {
          setAutoIterateError(err.detail);
        }
      } finally {
        setAutoIterateLoading(false);
      }
    }
  };

  // Get recommended model from results interpretation
  const getRecommendedModel = (): { id: string; reason: string } | null => {
    if (!agentRun?.steps) return null;

    const interpretationStep = agentRun.steps.find(
      (s) => s.step_type === 'results_interpretation' && s.status === 'completed'
    );

    if (!interpretationStep?.output_json) return null;

    const recommendation = interpretationStep.output_json.recommendation as Recommendation | undefined;
    if (!recommendation?.recommended_model_id) return null;

    return {
      id: recommendation.recommended_model_id,
      reason: recommendation.reason || 'AI recommended based on performance metrics',
    };
  };

  // Get critic findings
  const getCriticFindings = (): CriticFindings | null => {
    if (!agentRun?.steps) return null;

    const criticStep = agentRun.steps.find(
      (s) => s.step_type === 'results_critic' && s.status === 'completed'
    );

    if (!criticStep?.output_json) return null;

    const findings = criticStep.output_json.critic_findings as CriticFindings | undefined;
    if (!findings) return null;

    return {
      severity: findings.severity || 'low',
      issues: findings.issues || [],
      approved: findings.approved ?? true,
    };
  };

  // Handle promoting the recommended model
  const handlePromoteModel = async () => {
    const recommended = getRecommendedModel();
    if (!recommended) return;

    setIsPromoting(true);
    setPromoteError(null);
    setPromoteSuccess(null);

    try {
      await promoteModel(recommended.id, { status: 'candidate' });
      setPromoteSuccess('Model promoted to candidate status');
      onModelPromoted?.();
    } catch (err) {
      if (err instanceof ApiException) {
        setPromoteError(err.detail);
      } else {
        setPromoteError('Failed to promote model');
      }
    } finally {
      setIsPromoting(false);
    }
  };

  // Get step info for display
  const getStepInfo = (stepType: AgentStepType) => {
    return RESULTS_STEP_INFO[stepType] || {
      number: 0,
      name: stepType,
      role: 'Agent',
      icon: '🤖',
    };
  };

  // Check if experiment can run results pipeline
  const canRunPipeline = experimentStatus === 'completed' && !agentRun;

  if (isLoading) {
    return (
      <div className="results-pipeline-section">
        <h3>AI Results Analysis</h3>
        <LoadingSpinner message="Loading results analysis..." />
      </div>
    );
  }

  // If experiment is not completed, show a message
  if (experimentStatus !== 'completed') {
    return (
      <div className="results-pipeline-section">
        <h3>AI Results Analysis</h3>
        <div className="detail-card">
          <p className="empty-text">
            Complete the experiment to run AI-powered results analysis.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="results-pipeline-section">
      <div className="section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3>AI Results Analysis</h3>
        {canRunPipeline && (
          <button
            className="btn btn-primary"
            onClick={handleStartPipeline}
            disabled={isStarting}
          >
            {isStarting ? (
              <>
                <span className="spinner spinner-small"></span>
                Analyzing Results...
              </>
            ) : (
              <>
                <span className="btn-icon">🤖</span>
                Run AI Analysis
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

      {!agentRun ? (
        <div className="detail-card">
          <p className="empty-text">
            No results analysis yet. Click "Run AI Analysis" to have the AI interpret your experiment results and provide recommendations.
          </p>
        </div>
      ) : (
        <>
          {/* Pipeline Timeline */}
          <div className="pipeline-timeline" style={{ marginBottom: '1.5rem' }}>
            <div className="timeline-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span className="timeline-status">
                <StatusBadge status={agentRun.status} />
              </span>
              {agentRun.status === 'running' && (
                <span className="running-indicator" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#3b82f6' }}>
                  <span className="spinner spinner-small"></span>
                  Analyzing...
                </span>
              )}
            </div>

            <div className="timeline-steps" style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              {agentRun.steps?.map((step) => {
                const info = getStepInfo(step.step_type);
                return (
                  <div
                    key={step.id}
                    className={`timeline-step ${step.status}`}
                    onClick={() => setSelectedStep(step)}
                    style={{
                      flex: '1 1 200px',
                      padding: '1rem',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      backgroundColor: step.status === 'completed' ? '#f0fdf4' : step.status === 'running' ? '#eff6ff' : step.status === 'failed' ? '#fef2f2' : '#fff',
                      transition: 'transform 0.2s, box-shadow 0.2s',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'none';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                      <span style={{ fontSize: '1.25rem' }}>{info.icon}</span>
                      <span style={{ fontWeight: 600 }}>{info.name}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>{info.role}</span>
                      <StatusBadge status={step.status} />
                    </div>
                    {step.status === 'running' && (
                      <div style={{ marginTop: '0.5rem' }}>
                        <span className="spinner spinner-small"></span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Results Summary */}
          {agentRun.status === 'completed' && (
            <div className="results-summary">
              {/* Interpretation Summary */}
              {(() => {
                const interpretationStep = agentRun.steps?.find(s => s.step_type === 'results_interpretation' && s.status === 'completed');
                const summary = interpretationStep?.output_json?.natural_language_summary as string | undefined;
                if (!summary) return null;

                return (
                  <div className="detail-card" style={{ marginBottom: '1rem' }}>
                    <h4 style={{ marginBottom: '0.5rem' }}>Analysis Summary</h4>
                    <p style={{ color: '#374151', lineHeight: 1.6 }}>
                      {summary}
                    </p>
                  </div>
                );
              })()}

              {/* Critic Findings */}
              {(() => {
                const findings = getCriticFindings();
                if (!findings) return null;

                const severityColors: Record<string, string> = {
                  low: '#22c55e',
                  medium: '#f59e0b',
                  high: '#ef4444',
                };

                return (
                  <div className="detail-card" style={{ marginBottom: '1rem', borderLeft: `4px solid ${severityColors[findings.severity || 'low'] || '#6b7280'}` }}>
                    <h4 style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span>Review Findings</span>
                      <span style={{
                        fontSize: '0.75rem',
                        padding: '0.125rem 0.5rem',
                        borderRadius: '9999px',
                        backgroundColor: severityColors[findings.severity || 'low'] || '#6b7280',
                        color: 'white',
                        textTransform: 'uppercase',
                      }}>
                        {findings.severity || 'low'} severity
                      </span>
                    </h4>
                    {findings.issues && findings.issues.length > 0 ? (
                      <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#374151' }}>
                        {findings.issues.map((issue, i) => {
                          // Handle both string issues and object issues with {issue, severity, recommendation}
                          const issueText = typeof issue === 'string'
                            ? issue
                            : (issue as { issue?: string }).issue || JSON.stringify(issue);
                          return (
                            <li key={i} style={{ marginBottom: '0.25rem' }}>{issueText}</li>
                          );
                        })}
                      </ul>
                    ) : (
                      <p style={{ color: '#22c55e', margin: 0 }}>No issues found. Results look good!</p>
                    )}
                    <div style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontSize: '1.25rem' }}>{findings.approved ? '✅' : '⚠️'}</span>
                      <span style={{ fontWeight: 500 }}>
                        {findings.approved ? 'Approved for production' : 'Review recommended before production'}
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Recommended Model Action */}
              {(() => {
                const recommended = getRecommendedModel();
                if (!recommended) return null;

                return (
                  <div className="detail-card highlight" style={{
                    background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
                    border: '2px solid #22c55e',
                  }}>
                    <h4 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontSize: '1.25rem' }}>🏆</span>
                      Recommended Model
                    </h4>
                    <p style={{ color: '#374151', marginBottom: '1rem' }}>
                      {recommended.reason}
                    </p>

                    {promoteError && (
                      <div className="form-error" style={{ marginBottom: '0.75rem' }}>
                        {promoteError}
                      </div>
                    )}

                    {promoteSuccess ? (
                      <div style={{ color: '#22c55e', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span>✅</span>
                        {promoteSuccess}
                      </div>
                    ) : (
                      <button
                        className="btn btn-primary"
                        onClick={handlePromoteModel}
                        disabled={isPromoting}
                      >
                        {isPromoting ? (
                          <>
                            <span className="spinner spinner-small"></span>
                            Promoting...
                          </>
                        ) : (
                          <>
                            <span className="btn-icon">⬆️</span>
                            Promote to Candidate
                          </>
                        )}
                      </button>
                    )}
                  </div>
                );
              })()}

              {/* Auto-Iterate Toggle */}
              <div className="detail-card" style={{
                marginTop: '1rem',
                background: autoIterateSettings?.auto_iterate_enabled
                  ? 'linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)'
                  : 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
                border: autoIterateSettings?.auto_iterate_enabled
                  ? '2px solid #10b981'
                  : '2px solid #cbd5e1',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <h4 style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontSize: '1.25rem' }}>
                        {autoIterateSettings?.auto_iterate_enabled ? '🔄' : '⏸️'}
                      </span>
                      Auto-Iterate Mode
                      {autoIterateSettings?.auto_iterate_enabled && (
                        <span style={{
                          fontSize: '0.75rem',
                          padding: '0.125rem 0.5rem',
                          borderRadius: '9999px',
                          backgroundColor: '#10b981',
                          color: 'white',
                        }}>
                          ACTIVE
                        </span>
                      )}
                    </h4>
                    <p style={{ color: '#374151', fontSize: '0.875rem', marginBottom: '0.75rem' }}>
                      {autoIterateSettings?.auto_iterate_enabled
                        ? `Automatically runs AI analysis and creates improved iterations after each training. ${autoIterateSettings.current_iteration}/${autoIterateSettings.auto_iterate_max} iterations used.`
                        : 'Enable to automatically run AI feedback and iterate on experiments without manual intervention.'}
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleAutoIterateToggle}
                    disabled={autoIterateLoading}
                    style={{
                      position: 'relative',
                      width: '52px',
                      height: '28px',
                      borderRadius: '14px',
                      border: 'none',
                      cursor: autoIterateLoading ? 'wait' : 'pointer',
                      backgroundColor: autoIterateSettings?.auto_iterate_enabled ? '#10b981' : '#cbd5e1',
                      transition: 'background-color 0.2s',
                      flexShrink: 0,
                    }}
                    title={autoIterateSettings?.auto_iterate_enabled ? 'Disable auto-iterate' : 'Enable auto-iterate'}
                  >
                    <span
                      style={{
                        position: 'absolute',
                        top: '2px',
                        left: autoIterateSettings?.auto_iterate_enabled ? '26px' : '2px',
                        width: '24px',
                        height: '24px',
                        borderRadius: '12px',
                        backgroundColor: 'white',
                        boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                        transition: 'left 0.2s',
                      }}
                    />
                  </button>
                </div>

                {/* Max Iterations Selector */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.5rem' }}>
                  <label style={{ fontSize: '0.875rem', color: '#4b5563' }}>
                    Max iterations:
                  </label>
                  <select
                    value={pendingMaxIterations}
                    onChange={(e) => handleMaxIterationsChange(Number(e.target.value))}
                    disabled={autoIterateLoading}
                    style={{
                      padding: '0.375rem 0.75rem',
                      borderRadius: '6px',
                      border: '1px solid #d1d5db',
                      backgroundColor: 'white',
                      fontSize: '0.875rem',
                      cursor: 'pointer',
                    }}
                  >
                    {[1, 2, 3, 4, 5, 7, 10, 15, 20].map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                  {autoIterateSettings && (
                    <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                      ({autoIterateSettings.can_continue ? 'More iterations available' : 'Limit reached'})
                    </span>
                  )}
                </div>

                {autoIterateError && (
                  <div style={{
                    marginTop: '0.75rem',
                    padding: '0.5rem 0.75rem',
                    backgroundColor: '#fef2f2',
                    border: '1px solid #fecaca',
                    borderRadius: '6px',
                    color: '#991b1b',
                    fontSize: '0.875rem',
                  }}>
                    {autoIterateError}
                  </div>
                )}
              </div>

              {/* Auto-Improve Section */}
              <div className="detail-card" style={{
                marginTop: '1rem',
                background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
                border: '2px solid #3b82f6',
              }}>
                <h4 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontSize: '1.25rem' }}>🚀</span>
                  {autoIterateSettings?.auto_iterate_enabled ? 'Manual Improve' : 'Auto-Improve'}
                </h4>
                <p style={{ color: '#374151', marginBottom: '1rem' }}>
                  {autoIterateSettings?.auto_iterate_enabled
                    ? 'Auto-iterate is enabled. You can still manually trigger an improvement if needed.'
                    : 'Let AI analyze the results and automatically create an improved version with better features and configurations.'}
                </p>

                {improveError && !isImproving && (
                  <div style={{
                    marginBottom: '0.75rem',
                    padding: '1rem',
                    background: '#fef2f2',
                    border: '1px solid #fecaca',
                    borderRadius: '8px'
                  }}>
                    <div style={{ color: '#991b1b', fontWeight: 500, marginBottom: '0.5rem' }}>
                      ❌ Improvement Failed
                    </div>
                    <div style={{ color: '#7f1d1d', fontSize: '0.875rem', marginBottom: '0.75rem' }}>
                      {improveError}
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button
                        className="btn btn-primary"
                        onClick={() => {
                          setImproveError(null);
                          handleImprove();
                        }}
                        style={{ background: '#3b82f6' }}
                      >
                        🔄 Retry
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => setImproveError(null)}
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                )}

                {isImproving ? (
                  <div style={{ padding: '1rem', background: '#1e3a5f', borderRadius: '8px', border: '1px solid #3b82f6' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                      <span className="spinner"></span>
                      <span style={{ fontWeight: 600, fontSize: '1rem', color: '#ffffff' }}>
                        {improvementStatus?.steps?.length ? 'Improvement pipeline running...' : 'Starting improvement pipeline...'}
                      </span>
                    </div>

                    {/* Show improvement steps */}
                    {improvementStatus?.steps && improvementStatus.steps.length > 0 && (
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        {improvementStatus.steps.map((step, idx) => {
                          const info = RESULTS_STEP_INFO[step.step_type] || { name: step.step_type, icon: '⚙️' };
                          return (
                            <div
                              key={idx}
                              style={{
                                padding: '0.5rem 0.75rem',
                                borderRadius: '6px',
                                background: step.status === 'completed' ? '#166534' :
                                           step.status === 'running' ? '#1d4ed8' :
                                           step.status === 'failed' ? '#991b1b' : '#374151',
                                color: '#ffffff',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                                fontSize: '0.875rem',
                              }}
                            >
                              <span>{info.icon}</span>
                              <span>{info.name}</span>
                              {step.status === 'running' && <span className="spinner spinner-small"></span>}
                              {step.status === 'completed' && <span>✅</span>}
                              {step.status === 'failed' && <span>❌</span>}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ) : (
                  <button
                    className="btn btn-primary"
                    onClick={handleImprove}
                    style={{ background: '#3b82f6' }}
                  >
                    <span className="btn-icon">✨</span>
                    Improve & Retrain
                  </button>
                )}
              </div>
            </div>
          )}
        </>
      )}

      {/* Reproducible Notebook Section */}
      <div className="mt-6">
        {experimentStatus === 'completed' ? (
          <NotebookViewer experimentId={experimentId} />
        ) : experimentStatus === 'training' || experimentStatus === 'running' ? (
          <div className="border rounded-lg border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
            <div className="flex items-center gap-3">
              <svg
                className="w-6 h-6 text-orange-500 animate-pulse"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white">
                  Reproducible Notebook
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  A complete Python notebook will be available after training completes
                </p>
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {/* Step Drawer */}
      {selectedStep && (
        <AgentStepDrawer
          step={selectedStep}
          stepInfo={getStepInfo(selectedStep.step_type)}
          projectId={experimentId} // Using experimentId as context
          onClose={() => setSelectedStep(null)}
        />
      )}
    </div>
  );
}

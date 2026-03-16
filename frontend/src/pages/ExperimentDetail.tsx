import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import type { ExperimentDetail as ExperimentDetailType, Trial, ExperimentProgress, DatasetSpec, AgentRunList } from '../types/api';
import {
  getExperiment,
  listTrials,
  runExperiment,
  cancelExperiment,
  deleteExperiment,
  getExperimentProgress,
  getExperimentIterations,
  getDatasetSpec,
  listExperimentAgentRuns,
  applyExperimentFix,
  ApiException,
  type ExperimentIterationsResponse,
  type IssueType,
} from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import StatusBadge from '../components/StatusBadge';
import ChatBot from '../components/ChatBot';
import ExperimentResultsPipeline from '../components/ExperimentResultsPipeline';
import { TrainingOptionsModal } from '../components/TrainingOptionsModal';
import TrainingLogStream from '../components/TrainingLogStream';
import OverfittingMonitor from '../components/OverfittingMonitor';
import MetricExplainer from '../components/MetricExplainer';
import ResultsVisualization from '../components/ResultsVisualization';
import IterationComparison from '../components/IterationComparison';
import BaselineComparison from '../components/BaselineComparison';
import { RobustnessAuditPanel, type RobustnessAuditResult } from '../components/RobustnessAuditPanel';

export default function ExperimentDetail() {
  const { experimentId } = useParams<{ experimentId: string }>();
  const navigate = useNavigate();

  const [experiment, setExperiment] = useState<ExperimentDetailType | null>(null);
  const [trials, setTrials] = useState<Trial[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [progress, setProgress] = useState<ExperimentProgress | null>(null);
  const progressIntervalRef = useRef<number | null>(null);
  const [showTrainingOptions, setShowTrainingOptions] = useState(false);
  const [iterations, setIterations] = useState<ExperimentIterationsResponse | null>(null);
  const [datasetSpec, setDatasetSpec] = useState<DatasetSpec | null>(null);
  const [robustnessAudit, setRobustnessAudit] = useState<RobustnessAuditResult | null>(null);
  const [applyingFix, setApplyingFix] = useState<IssueType | null>(null);

  // Use ref to track if initial load is done
  const initialLoadDoneRef = useRef(false);

  const fetchData = useCallback(async (isRefresh = false) => {
    if (!experimentId) return;

    // Only show loading spinner on initial load, not refreshes
    if (!isRefresh && !initialLoadDoneRef.current) {
      setIsLoading(true);
    }
    setError(null);

    try {
      const [experimentData, trialsData] = await Promise.all([
        getExperiment(experimentId),
        listTrials(experimentId),
      ]);

      setExperiment(experimentData);
      setTrials(trialsData);
      initialLoadDoneRef.current = true;

      // Fetch iterations in background (don't block main load)
      getExperimentIterations(experimentId)
        .then(setIterations)
        .catch(() => setIterations(null));

      // Fetch dataset spec if available (for time-based task warnings)
      if (experimentData.dataset_spec_id) {
        getDatasetSpec(experimentData.dataset_spec_id)
          .then(setDatasetSpec)
          .catch(() => setDatasetSpec(null));
      }

      // Fetch robustness audit from agent runs (Prompt 4)
      if (experimentData.status === 'completed') {
        listExperimentAgentRuns(experimentId)
          .then((agentRuns: AgentRunList) => {
            // Find the most recent completed robustness audit
            for (const run of agentRuns.items) {
              if (run.name?.includes('Robustness Audit') && run.status === 'completed') {
                // Check result_json for robustness_audit data
                const result = run.result_json as Record<string, unknown> | undefined;
                if (result?.robustness_audit) {
                  setRobustnessAudit(result.robustness_audit as RobustnessAuditResult);
                  break;
                }
              }
              // Also check steps for robustness_audit output
              if (run.steps) {
                for (const step of run.steps) {
                  if (step.step_type === 'robustness_audit' && step.status === 'completed') {
                    const output = step.output_json as Record<string, unknown> | undefined;
                    if (output?.robustness_audit) {
                      setRobustnessAudit(output.robustness_audit as RobustnessAuditResult);
                      break;
                    }
                  }
                }
              }
            }
          })
          .catch(() => setRobustnessAudit(null));
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load experiment data');
      }
    } finally {
      setIsLoading(false);
    }
  }, [experimentId]);

  useEffect(() => {
    initialLoadDoneRef.current = false;
    fetchData();
  }, [fetchData]);

  // Auto-refresh while running (without showing loading spinner)
  useEffect(() => {
    if (!experiment || (experiment.status !== 'running' && experiment.status !== 'pending')) {
      return;
    }

    const interval = setInterval(() => fetchData(true), 5000);
    return () => clearInterval(interval);
  }, [experiment?.status, fetchData]);

  // Progress polling for running experiments
  useEffect(() => {
    if (!experimentId || !experiment) return;

    // Clear previous interval
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }

    // Only poll while running
    if (experiment.status !== 'running') {
      setProgress(null);
      return;
    }

    // Fetch progress immediately, then poll every 2 seconds
    const fetchProgress = async () => {
      try {
        const progressData = await getExperimentProgress(experimentId);
        setProgress(progressData);
      } catch {
        // Silently ignore progress fetch errors
      }
    };

    fetchProgress();
    progressIntervalRef.current = window.setInterval(fetchProgress, 2000);

    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
    };
  }, [experimentId, experiment?.status]);

  const handleRunClick = () => {
    setShowTrainingOptions(true);
  };

  const handleRunWithOptions = async (options: {
    backend: 'local' | 'modal';
    resourceLimitsEnabled: boolean;
    numCpus?: number;
    numGpus?: number;
    memoryLimitGb?: number;
  }) => {
    if (!experimentId) return;
    setShowTrainingOptions(false);
    setActionLoading(true);
    try {
      await runExperiment(experimentId, options);
      await fetchData();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setActionLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!experimentId) return;
    setActionLoading(true);
    try {
      await cancelExperiment(experimentId);
      await fetchData();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!experimentId || !experiment) return;

    const confirmed = window.confirm(
      `Are you sure you want to delete "${experiment.name}"? This action cannot be undone.`
    );
    if (!confirmed) return;

    setActionLoading(true);
    try {
      await deleteExperiment(experimentId);
      navigate(`/projects/${experiment.project_id}`);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setActionLoading(false);
    }
  };

  const handleApplyFix = async (issueType: IssueType, issueDescription: string, recommendedFix?: string) => {
    if (!experimentId) return;

    setApplyingFix(issueType);
    setError(null);
    try {
      const result = await applyExperimentFix(experimentId, {
        issue_type: issueType,
        issue_description: issueDescription,
        recommended_fix: recommendedFix,
      });
      // Refresh the experiment data to show updated configuration
      await fetchData(true);
      // Show success message briefly
      alert(result.message);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError(`Failed to apply fix for ${issueType}`);
      }
    } finally {
      setApplyingFix(null);
    }
  };

  const handleDownloadNotebook = () => {
    if (!experimentId) return;
    const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8001';
    window.open(`${apiBase}/experiments/${experimentId}/notebook?format=download`, '_blank');
  };


  const formatDate = (dateString: string) => {
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
      <div className="experiment-detail-page">
        <LoadingSpinner message="Loading experiment..." />
      </div>
    );
  }

  if (error && !experiment) {
    return (
      <div className="experiment-detail-page">
        <ErrorMessage message={error} onRetry={fetchData} />
        <Link to="/experiments" className="btn btn-secondary" style={{ marginTop: '1rem' }}>
          Back to Experiments
        </Link>
      </div>
    );
  }

  if (!experiment) return null;

  return (
    <div className="experiment-detail-page">
      {error && (
        <div className="form-error" style={{ marginBottom: '1rem' }}>
          {error}
          <button onClick={() => setError(null)} style={{ marginLeft: '1rem' }}>
            Dismiss
          </button>
        </div>
      )}

      {/* Time-based task warning for random splits */}
      {datasetSpec?.is_time_based && (
        (() => {
          const validationStrategy = experiment.experiment_plan_json?.validation_strategy;
          const splitType = validationStrategy?.split_strategy ||
            trials[0]?.data_split_strategy || 'unknown';
          const isRandomSplit = ['random', 'stratified', 'group_random'].includes(splitType.toLowerCase());

          if (isRandomSplit) {
            return (
              <div style={{
                padding: '1rem',
                marginBottom: '1rem',
                background: 'linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)',
                border: '1px solid #ef4444',
                borderLeft: '4px solid #dc2626',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.75rem',
              }}>
                <span style={{ fontSize: '1.5rem' }}>⚠️</span>
                <div>
                  <div style={{ fontWeight: 600, color: '#b91c1c', marginBottom: '0.25rem' }}>
                    Data Leakage Risk Detected
                  </div>
                  <div style={{ fontSize: '0.875rem', color: '#7f1d1d', lineHeight: 1.5 }}>
                    This dataset is marked as <strong>time-based</strong>
                    {datasetSpec.time_column && <> (time column: <code>{datasetSpec.time_column}</code>)</>}
                    {datasetSpec.entity_id_column && <>, entity: <code>{datasetSpec.entity_id_column}</code></>},
                    but the experiment uses a <strong>{splitType}</strong> split strategy.
                    Random splits on time-series data cause future information to leak into training,
                    producing overly optimistic results that won't generalize.
                  </div>
                  <div style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <button
                      onClick={() => handleApplyFix(
                        'split_strategy',
                        `Time-based dataset using ${splitType} split causes data leakage`,
                        `Use ${datasetSpec.entity_id_column ? 'group_time' : 'time'} split`
                      )}
                      disabled={applyingFix === 'split_strategy'}
                      style={{
                        padding: '0.5rem 1rem',
                        background: '#dc2626',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        fontWeight: 600,
                        cursor: applyingFix === 'split_strategy' ? 'not-allowed' : 'pointer',
                        opacity: applyingFix === 'split_strategy' ? 0.7 : 1,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                      }}
                    >
                      {applyingFix === 'split_strategy' ? 'Fixing...' : `Fix: Use ${datasetSpec.entity_id_column ? 'group_time' : 'time'} split`}
                    </button>
                    <span style={{ fontSize: '0.75rem', color: '#9a3412' }}>
                      Then re-run the experiment to apply the fix.
                    </span>
                  </div>
                </div>
              </div>
            );
          }
          return null;
        })()
      )}

      {/* Overfitting Warning Banner (Make Holdout Score the Real Score) */}
      {experiment.status === 'completed' && experiment.has_holdout && experiment.overfitting_gap !== null && experiment.overfitting_gap > 0.05 && (
        <div style={{
          padding: '1rem',
          marginBottom: '1rem',
          background: 'linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)',
          border: '1px solid #f97316',
          borderLeft: '4px solid #ea580c',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.75rem',
        }}>
          <span style={{ fontSize: '1.5rem' }}>⚠️</span>
          <div>
            <div style={{ fontWeight: 600, color: '#c2410c', marginBottom: '0.25rem' }}>
              Overfitting Detected
            </div>
            <div style={{ fontSize: '0.875rem', color: '#9a3412', lineHeight: 1.5 }}>
              The validation score ({experiment.val_score?.toFixed(4)}) is significantly higher than the
              holdout score ({experiment.final_score?.toFixed(4)}) — a gap of <strong>{experiment.overfitting_gap.toFixed(4)}</strong>.
              {experiment.overfitting_gap > 0.10 ? (
                <> This is a <strong>large discrepancy</strong> indicating the model may have overfit to the validation data.</>
              ) : (
                <> This suggests the model may have overfit to patterns in the validation set.</>
              )}
            </div>
            <div style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <button
                onClick={() => handleApplyFix(
                  'overfitting',
                  `Overfitting detected: validation score ${experiment.val_score?.toFixed(4)} vs holdout ${experiment.final_score?.toFixed(4)} (gap: ${experiment.overfitting_gap?.toFixed(4)})`,
                  'Increase regularization and simplify the model'
                )}
                disabled={applyingFix === 'overfitting'}
                style={{
                  padding: '0.5rem 1rem',
                  background: '#ea580c',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  fontWeight: 600,
                  cursor: applyingFix === 'overfitting' ? 'not-allowed' : 'pointer',
                  opacity: applyingFix === 'overfitting' ? 0.7 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                }}
              >
                {applyingFix === 'overfitting' ? 'Fixing...' : 'Fix: Add Regularization'}
              </button>
              <span style={{ fontSize: '0.75rem', color: '#9a3412' }}>
                Then re-run the experiment to apply the fix.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Feature Engineering Failures Warning Banner */}
      {experiment.experiment_plan_json?.feature_engineering_warnings &&
        (experiment.experiment_plan_json.feature_engineering_warnings as string[]).length > 0 && (
        <div style={{
          padding: '1rem',
          marginBottom: '1rem',
          background: 'linear-gradient(135deg, #fefce8 0%, #fef9c3 100%)',
          border: '1px solid #eab308',
          borderLeft: '4px solid #ca8a04',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.75rem',
        }}>
          <span style={{ fontSize: '1.5rem' }}>⚠️</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600, color: '#a16207', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              Feature Engineering Issues
              <span style={{
                padding: '0.125rem 0.5rem',
                background: '#fef08a',
                border: '1px solid #facc15',
                borderRadius: '9999px',
                fontSize: '0.75rem',
                fontWeight: 500,
              }}>
                {experiment.experiment_plan_json.feature_engineering_failure_count} failed
                {experiment.experiment_plan_json.feature_engineering_success_count !== undefined &&
                  ` / ${(experiment.experiment_plan_json.feature_engineering_success_count as number) + (experiment.experiment_plan_json.feature_engineering_failure_count as number)} total`}
              </span>
            </div>
            <div style={{ fontSize: '0.8125rem', color: '#854d0e', lineHeight: 1.6 }}>
              <strong>Some designed features could not be created.</strong> The experiment ran with the available features,
              but results may differ from what was planned.
            </div>
            <ul style={{
              margin: '0.75rem 0 0 0',
              paddingLeft: '1.25rem',
              fontSize: '0.8125rem',
              color: '#713f12',
              lineHeight: 1.6,
            }}>
              {(experiment.experiment_plan_json.feature_engineering_warnings as string[]).map((warning, idx) => (
                <li key={idx} style={{ marginBottom: '0.25rem' }}>{warning}</li>
              ))}
            </ul>
            <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: '#92400e', fontStyle: 'italic' }}>
              Check your formula syntax or source column availability. The AI may have generated invalid formulas.
            </div>
          </div>
        </div>
      )}

      <div className="page-header">
        <div>
          <div className="breadcrumb">
            <Link to="/projects">Projects</Link>
            <span>/</span>
            <Link to={`/projects/${experiment.project_id}`}>Project</Link>
            <span>/</span>
            <span>{experiment.name}</span>
          </div>
          <h2>{experiment.name}</h2>
          <div className="experiment-meta">
            <StatusBadge status={experiment.status} />
            {/* Iteration indicator */}
            {iterations && iterations.total_iterations > 1 && (
              <span style={{
                marginLeft: '0.5rem',
                padding: '0.25rem 0.75rem',
                background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
                border: '1px solid #3b82f6',
                borderRadius: '9999px',
                fontSize: '0.75rem',
                fontWeight: 600,
                color: '#1d4ed8',
              }}>
                Iteration {iterations.iterations.find(i => i.id === experimentId)?.iteration_number || 1} of {iterations.total_iterations}
              </span>
            )}
          </div>
          {experiment.description && (
            <p className="experiment-description">{experiment.description}</p>
          )}

          {/* Iterations Navigation */}
          {iterations && iterations.total_iterations > 1 && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem',
              background: '#f8fafc',
              borderRadius: '8px',
              border: '1px solid #e2e8f0',
            }}>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '0.5rem', fontWeight: 500 }}>
                Experiment Iterations
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                {iterations.iterations.map((iter, idx) => {
                  const isCurrent = iter.id === experimentId;
                  // Use final_score (holdout) for comparisons (Make Holdout Score the Real Score)
                  const currentScore = iter.final_score ?? iter.best_score;
                  const prevScore = iterations.iterations[idx - 1]?.final_score ?? iterations.iterations[idx - 1]?.best_score;
                  const improvement = iter.iteration_number > 1 && prevScore && currentScore
                    ? ((currentScore - prevScore) / Math.abs(prevScore) * 100)
                    : null;
                  const hasOverfitting = iter.has_holdout && iter.overfitting_gap !== null && iter.overfitting_gap > 0.05;

                  return (
                    <Link
                      key={iter.id}
                      to={`/experiments/${iter.id}`}
                      style={{
                        padding: '0.5rem 0.75rem',
                        background: isCurrent ? '#3b82f6' : (hasOverfitting ? '#fef3c7' : '#fff'),
                        color: isCurrent ? '#fff' : '#374151',
                        border: `1px solid ${isCurrent ? '#3b82f6' : (hasOverfitting ? '#f59e0b' : '#d1d5db')}`,
                        borderRadius: '6px',
                        textDecoration: 'none',
                        fontSize: '0.8125rem',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        minWidth: '80px',
                        transition: 'all 0.2s',
                      }}
                    >
                      <div style={{ fontWeight: 600 }}>#{iter.iteration_number}</div>
                      <div style={{ fontSize: '0.6875rem', opacity: 0.8 }}>
                        {iter.status === 'completed' && currentScore !== null
                          ? currentScore.toFixed(4)
                          : iter.status}
                      </div>
                      {iter.has_holdout && (
                        <div style={{ fontSize: '0.5625rem', opacity: 0.6 }}>
                          {iter.score_source === 'holdout' ? '(holdout)' : '(val)'}
                        </div>
                      )}
                      {improvement !== null && iter.iteration_number > 1 && (
                        <div style={{
                          fontSize: '0.625rem',
                          color: improvement >= 0 ? (isCurrent ? '#bbf7d0' : '#22c55e') : (isCurrent ? '#fecaca' : '#ef4444'),
                          fontWeight: 500,
                        }}>
                          {improvement >= 0 ? '+' : ''}{improvement.toFixed(1)}%
                        </div>
                      )}
                      {hasOverfitting && !isCurrent && (
                        <span style={{ fontSize: '0.625rem' }} title="Overfitting detected">⚠️</span>
                      )}
                    </Link>
                  );
                })}
              </div>

              {/* Overfitting Monitor - shows holdout validation scores */}
              {experiment.project_id && experimentId && iterations.total_iterations >= 1 && (
                <div style={{ marginTop: '1rem' }}>
                  <OverfittingMonitor
                    projectId={experiment.project_id}
                    experimentId={experimentId}
                    onIterationClick={(expId) => navigate(`/experiments/${expId}`)}
                  />
                </div>
              )}

              {/* Detailed Iteration Comparison Table */}
              {iterations.total_iterations > 1 && experimentId && (
                <div style={{ marginTop: '1rem' }}>
                  <IterationComparison
                    iterations={iterations.iterations}
                    currentExperimentId={experimentId}
                    primaryMetric={experiment.primary_metric}
                    metricDirection={experiment.metric_direction || 'maximize'}
                  />
                </div>
              )}
            </div>
          )}

          {/* Improvement Context (for iterations > 1) */}
          {experiment.improvement_context_json && experiment.iteration_number > 1 && (
            <div style={{
              marginTop: '1rem',
              padding: '1rem',
              background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
              borderRadius: '8px',
              border: '1px solid #86efac',
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginBottom: '0.75rem',
              }}>
                <span style={{ fontSize: '1.25rem' }}>🔄</span>
                <h4 style={{ margin: 0, fontSize: '0.9375rem', fontWeight: 600, color: '#166534' }}>
                  Improvement from Iteration {experiment.iteration_number - 1}
                </h4>
              </div>

              {/* Summary */}
              {experiment.improvement_context_json.summary && (
                <p style={{
                  margin: '0 0 1rem 0',
                  fontSize: '0.875rem',
                  color: '#15803d',
                  lineHeight: 1.5,
                }}>
                  {experiment.improvement_context_json.summary}
                </p>
              )}

              {/* Key Issues Identified */}
              {experiment.improvement_context_json.improvement_analysis?.key_issues &&
                experiment.improvement_context_json.improvement_analysis.key_issues.length > 0 && (
                <div style={{ marginBottom: '0.75rem' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#b91c1c', marginBottom: '0.25rem' }}>
                    Issues Identified:
                  </div>
                  <ul style={{
                    margin: 0,
                    paddingLeft: '1.25rem',
                    fontSize: '0.8125rem',
                    color: '#374151',
                    lineHeight: 1.6,
                  }}>
                    {experiment.improvement_context_json.improvement_analysis.key_issues.slice(0, 3).map((issue, idx) => (
                      <li key={idx}>{issue}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Changes Made */}
              {experiment.improvement_context_json.improvement_plan?.feature_changes && (
                <div style={{ marginBottom: '0.75rem' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#1d4ed8', marginBottom: '0.25rem' }}>
                    Changes Made:
                  </div>
                  <div style={{ fontSize: '0.8125rem', color: '#374151' }}>
                    {experiment.improvement_context_json.improvement_plan.feature_changes.features_to_remove &&
                      experiment.improvement_context_json.improvement_plan.feature_changes.features_to_remove.length > 0 && (
                      <div style={{ marginBottom: '0.25rem' }}>
                        <span style={{ color: '#dc2626' }}>Removed: </span>
                        {experiment.improvement_context_json.improvement_plan.feature_changes.features_to_remove.join(', ')}
                      </div>
                    )}
                    {experiment.improvement_context_json.improvement_plan.feature_changes.engineered_features &&
                      experiment.improvement_context_json.improvement_plan.feature_changes.engineered_features.length > 0 && (
                      <div>
                        <span style={{ color: '#16a34a' }}>New features: </span>
                        {experiment.improvement_context_json.improvement_plan.feature_changes.engineered_features
                          .map(f => f.output_column || f.name)
                          .join(', ')}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Expected Improvement */}
              {experiment.improvement_context_json.improvement_plan?.expected_improvement && (
                <div style={{
                  fontSize: '0.8125rem',
                  color: '#15803d',
                  fontStyle: 'italic',
                  padding: '0.5rem',
                  background: 'rgba(255,255,255,0.5)',
                  borderRadius: '4px',
                }}>
                  <strong>Expected: </strong>
                  {experiment.improvement_context_json.improvement_plan.expected_improvement}
                </div>
              )}
            </div>
          )}

          {/* Progress section for running experiments */}
          {experiment.status === 'running' && (
            <div className="experiment-progress-section" style={{ marginTop: '1rem' }}>
              <div className="progress-header" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <span className="spinner spinner-small"></span>
                <strong>{progress?.message || 'Training in progress...'}</strong>
              </div>
              <div className="progress-bar-container" style={{
                width: '100%',
                height: '24px',
                backgroundColor: '#e5e7eb',
                borderRadius: '12px',
                overflow: 'hidden',
                position: 'relative',
              }}>
                <div
                  className="progress-bar-fill"
                  style={{
                    width: `${progress?.progress || 0}%`,
                    height: '100%',
                    backgroundColor: '#3b82f6',
                    transition: 'width 0.3s ease',
                    borderRadius: '12px',
                  }}
                />
                <span style={{
                  position: 'absolute',
                  left: '50%',
                  top: '50%',
                  transform: 'translate(-50%, -50%)',
                  fontSize: '12px',
                  fontWeight: 600,
                  color: (progress?.progress || 0) > 50 ? '#fff' : '#374151',
                }}>
                  {progress?.progress || 0}%
                </span>
              </div>
              {progress?.stage && (
                <div className="progress-stage" style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#6b7280' }}>
                  Stage: {progress.stage}
                </div>
              )}

              {/* Live Training Logs */}
              <TrainingLogStream
                experimentId={experimentId!}
                isRunning={experiment.status === 'running'}
              />
            </div>
          )}
        </div>
        <div className="header-actions">
          {(experiment.status === 'pending' || experiment.status === 'failed') && (
            <button
              className="btn btn-primary"
              onClick={handleRunClick}
              disabled={actionLoading}
            >
              {actionLoading ? 'Starting...' : 'Run Experiment'}
            </button>
          )}
          {(experiment.status === 'running' || experiment.status === 'pending') && (
            <button
              className="btn btn-danger"
              onClick={handleCancel}
              disabled={actionLoading}
            >
              {actionLoading ? 'Cancelling...' : 'Cancel'}
            </button>
          )}
          {experiment.status !== 'running' && (
            <button
              className="btn btn-secondary"
              onClick={handleDelete}
              disabled={actionLoading}
              style={{ marginLeft: '0.5rem' }}
            >
              Delete
            </button>
          )}
          <button
            className="btn btn-secondary"
            onClick={handleDownloadNotebook}
            style={{ marginLeft: '0.5rem' }}
            title="Download Jupyter Notebook"
          >
            Download Notebook
          </button>
        </div>
      </div>

      {/* Error Message Banner */}
      {experiment.status === 'failed' && experiment.error_message && (
        <div style={{
          margin: '1rem 0',
          padding: '1rem',
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '0.5rem',
        }}>
          <h4 style={{ margin: '0 0 0.5rem 0', color: '#dc2626' }}>
            Experiment Failed
          </h4>
          <pre style={{
            margin: 0,
            padding: '0.75rem',
            backgroundColor: '#fff',
            border: '1px solid #fee2e2',
            borderRadius: '0.25rem',
            fontSize: '0.875rem',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            maxHeight: '300px',
            overflowY: 'auto',
            color: '#7f1d1d',
          }}>
            {experiment.error_message}
          </pre>
        </div>
      )}

      <div className="detail-grid">
        <div className="detail-section">
          <h3>Configuration</h3>
          <div className="detail-card">
            <div className="detail-row">
              <span className="detail-label">Primary Metric:</span>
              <span className="detail-value">{experiment.primary_metric || 'Auto'}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Direction:</span>
              <span className="detail-value">{experiment.metric_direction || 'Auto'}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Dataset:</span>
              <span className="detail-value">
                {experiment.dataset_spec_id ? (
                  <Link
                    to={`/dataset-results/${experiment.dataset_spec_id}`}
                    style={{ color: '#3b82f6', textDecoration: 'none' }}
                    title="View all experiments for this dataset"
                  >
                    {experiment.dataset_spec_id.slice(0, 8)}... 📊
                  </Link>
                ) : (
                  'Not set'
                )}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Created:</span>
              <span className="detail-value">{formatDate(experiment.created_at)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Updated:</span>
              <span className="detail-value">{formatDate(experiment.updated_at)}</span>
            </div>
          </div>
        </div>

        {experiment.experiment_plan_json?.automl_config && (
          <div className="detail-section">
            <h3>AutoML Configuration</h3>
            <div className="detail-card">
              {Object.entries(experiment.experiment_plan_json.automl_config).map(
                ([key, value]) => (
                  <div key={key} className="detail-row">
                    <span className="detail-label">{key}:</span>
                    <span className="detail-value">{String(value)}</span>
                  </div>
                )
              )}
            </div>
          </div>
        )}

        {/* Validation Strategy Section */}
        {(experiment.experiment_plan_json?.validation_strategy || trials[0]?.data_split_strategy) && (
          <div className="detail-section">
            <h3>Data Splitting</h3>
            <div className="detail-card">
              {(() => {
                const vs = experiment.experiment_plan_json?.validation_strategy;
                const splitType = vs?.split_strategy || trials[0]?.data_split_strategy || 'unknown';
                const isTimeBased = ['time', 'group_time', 'temporal'].includes(splitType.toLowerCase());
                const isRandomOnTimeBased = datasetSpec?.is_time_based &&
                  ['random', 'stratified', 'group_random'].includes(splitType.toLowerCase());

                return (
                  <>
                    <div className="detail-row">
                      <span className="detail-label">Split Strategy:</span>
                      <span className="detail-value" style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                      }}>
                        <span style={{
                          padding: '0.25rem 0.5rem',
                          borderRadius: '4px',
                          fontSize: '0.8125rem',
                          fontWeight: 600,
                          background: isTimeBased ? '#dcfce7' : (isRandomOnTimeBased ? '#fef2f2' : '#f3f4f6'),
                          color: isTimeBased ? '#166534' : (isRandomOnTimeBased ? '#b91c1c' : '#374151'),
                          border: `1px solid ${isTimeBased ? '#86efac' : (isRandomOnTimeBased ? '#fca5a5' : '#d1d5db')}`,
                        }}>
                          {splitType}
                        </span>
                        {isTimeBased && <span title="Time-based split prevents data leakage">✓</span>}
                        {isRandomOnTimeBased && <span title="Random split on time-based data may cause data leakage">⚠️</span>}
                      </span>
                    </div>
                    {vs?.validation_split && (
                      <div className="detail-row">
                        <span className="detail-label">Validation Size:</span>
                        <span className="detail-value">{(vs.validation_split * 100).toFixed(0)}%</span>
                      </div>
                    )}
                    {(vs?.time_column || datasetSpec?.time_column) && (
                      <div className="detail-row">
                        <span className="detail-label">Time Column:</span>
                        <span className="detail-value">{vs?.time_column || datasetSpec?.time_column}</span>
                      </div>
                    )}
                    {(vs?.entity_id_column || vs?.group_column || datasetSpec?.entity_id_column) && (
                      <div className="detail-row">
                        <span className="detail-label">Entity Column:</span>
                        <span className="detail-value">{vs?.entity_id_column || vs?.group_column || datasetSpec?.entity_id_column}</span>
                      </div>
                    )}
                    {vs?.n_folds && (
                      <div className="detail-row">
                        <span className="detail-label">Cross-Val Folds:</span>
                        <span className="detail-value">{vs.n_folds}</span>
                      </div>
                    )}
                    {vs?.reasoning && (
                      <div className="detail-row" style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #e5e7eb' }}>
                        <span className="detail-label">Reasoning:</span>
                        <span className="detail-value" style={{ fontStyle: 'italic', fontSize: '0.8125rem', color: '#6b7280' }}>
                          {vs.reasoning}
                        </span>
                      </div>
                    )}
                  </>
                );
              })()}
            </div>
          </div>
        )}

        {experiment.best_model && (
          <div className="detail-section">
            <h3>Best Model</h3>
            <div className="detail-card highlight">
              <div className="detail-row">
                <span className="detail-label">Name:</span>
                <span className="detail-value">{experiment.best_model.name}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Type:</span>
                <span className="detail-value">{experiment.best_model.model_type}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Status:</span>
                <StatusBadge status={experiment.best_model.status} />
              </div>
              <Link
                to={`/models/${experiment.best_model.id}`}
                className="btn btn-secondary btn-small"
                style={{ marginTop: '1rem' }}
              >
                View Model Details
              </Link>
            </div>
          </div>
        )}

        {/* Final Score Section (Make Holdout Score the Real Score) */}
        {experiment.status === 'completed' && experiment.final_score !== null && (
          <div className="detail-section">
            <h3>Final Score</h3>
            <div className="detail-card" style={{
              background: experiment.has_holdout ? 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)' : '#fff',
              borderColor: experiment.has_holdout ? '#86efac' : '#e5e7eb',
            }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {/* Primary Score - Holdout or Validation fallback */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{
                    fontSize: '2rem',
                    fontWeight: 700,
                    color: experiment.has_holdout ? '#166534' : '#374151',
                  }}>
                    {experiment.final_score.toFixed(4)}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <span style={{
                      fontSize: '0.875rem',
                      fontWeight: 600,
                      color: experiment.has_holdout ? '#166534' : '#6b7280',
                    }}>
                      {experiment.has_holdout ? 'Holdout Score' : 'Validation Score'} ({experiment.primary_metric || 'score'})
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                      {experiment.has_holdout
                        ? `Evaluated on ${experiment.holdout_samples?.toLocaleString() || 'held-out'} samples not seen during training`
                        : 'No holdout evaluation performed'}
                    </span>
                  </div>
                </div>

                {/* Score Breakdown - when holdout exists */}
                {experiment.has_holdout && (
                  <div style={{
                    display: 'flex',
                    gap: '1.5rem',
                    paddingTop: '0.75rem',
                    borderTop: '1px solid #d1fae5',
                  }}>
                    {experiment.train_score !== null && (
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '0.6875rem', color: '#6b7280', textTransform: 'uppercase' }}>Train</span>
                        <span style={{ fontSize: '1rem', fontWeight: 600, color: '#374151' }}>{experiment.train_score.toFixed(4)}</span>
                      </div>
                    )}
                    {experiment.val_score !== null && (
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '0.6875rem', color: '#6b7280', textTransform: 'uppercase' }}>Validation</span>
                        <span style={{ fontSize: '1rem', fontWeight: 600, color: '#374151' }}>{experiment.val_score.toFixed(4)}</span>
                      </div>
                    )}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <span style={{ fontSize: '0.6875rem', color: '#166534', textTransform: 'uppercase', fontWeight: 600 }}>Holdout ✓</span>
                      <span style={{ fontSize: '1rem', fontWeight: 700, color: '#166534' }}>{experiment.final_score.toFixed(4)}</span>
                    </div>

                    {/* Overfitting Gap Indicator */}
                    {experiment.overfitting_gap !== null && (
                      <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        marginLeft: 'auto',
                        padding: '0.5rem 0.75rem',
                        borderRadius: '6px',
                        background: experiment.overfitting_gap > 0.05 ? '#fef2f2' : '#f0fdf4',
                        border: `1px solid ${experiment.overfitting_gap > 0.05 ? '#fca5a5' : '#86efac'}`,
                      }}>
                        <span style={{
                          fontSize: '0.6875rem',
                          color: experiment.overfitting_gap > 0.05 ? '#b91c1c' : '#166534',
                          textTransform: 'uppercase',
                        }}>
                          {experiment.overfitting_gap > 0.05 ? 'Overfitting Gap ⚠️' : 'Generalization ✓'}
                        </span>
                        <span style={{
                          fontSize: '0.875rem',
                          fontWeight: 600,
                          color: experiment.overfitting_gap > 0.05 ? '#dc2626' : '#22c55e',
                        }}>
                          {experiment.overfitting_gap > 0 ? '+' : ''}{experiment.overfitting_gap.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {experiment.best_metrics && (
          <div className="detail-section">
            <h3>All Metrics</h3>
            <div className="detail-card">
              <div className="metrics-grid">
                {Object.entries(experiment.best_metrics)
                  .filter(([key]) => typeof experiment.best_metrics![key] === 'number')
                  .map(([key, value]) => {
                    // Error metrics should always be displayed as positive values
                    // sklearn uses negative values (neg_rmse, neg_mae) for optimization
                    const isErrorMetric = key.toLowerCase().includes('error') ||
                      key.toLowerCase().includes('rmse') ||
                      key.toLowerCase().includes('mse') ||
                      key.toLowerCase().includes('mae') ||
                      key.toLowerCase().includes('loss');
                    const displayValue = typeof value === 'number'
                      ? (isErrorMetric ? Math.abs(value) : value)
                      : value;
                    // Clean up metric name (remove neg_ prefix for display)
                    const displayKey = key.replace(/^neg_/, '');

                    return (
                      <div key={key} className="metric-box">
                        <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                          {displayKey}
                          <MetricExplainer metricKey={displayKey} value={displayValue as number} />
                        </span>
                        <span className="metric-value">
                          {typeof displayValue === 'number' ? displayValue.toFixed(4) : String(displayValue)}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Results Visualization Section - Shows predicted vs actual charts */}
      {experiment.status === 'completed' && experimentId && (
        <div className="detail-section full-width" style={{ marginTop: '1.5rem' }}>
          <ResultsVisualization experimentId={experimentId} experimentStatus={experiment.status} />
        </div>
      )}

      {/* Baselines & Sanity Checks Section */}
      {experiment.status === 'completed' && trials.length > 0 && trials[0].baseline_metrics_json && (
        <div className="detail-section full-width" style={{ marginTop: '1.5rem' }}>
          <BaselineComparison
            baselineMetrics={trials[0].baseline_metrics_json}
            modelMetrics={trials[0].metrics_json}
            taskType={(() => {
              // Try to get from automl_config first
              const automlConfig = (experiment.experiment_plan_json as Record<string, Record<string, string>>)?.automl_config;
              if (automlConfig?.problem_type) {
                return automlConfig.problem_type as 'binary' | 'multiclass' | 'regression';
              }
              // Infer from trial metrics
              const metrics = trials[0].metrics_json;
              if (metrics) {
                // If RMSE/MAE/R2 are present, it's regression
                if ('rmse' in metrics || 'mae' in metrics || 'r2' in metrics ||
                    'root_mean_squared_error' in metrics || 'mean_absolute_error' in metrics) {
                  return 'regression';
                }
                // If ROC AUC is present, it's binary classification
                if ('roc_auc' in metrics) {
                  return 'binary';
                }
                // If balanced_accuracy or f1_macro, could be multiclass
                if ('balanced_accuracy' in metrics || 'f1_macro' in metrics) {
                  return 'multiclass';
                }
              }
              // Default to regression if no classification metrics found
              return 'regression';
            })()}
          />
        </div>
      )}

      {/* Robustness Audit Section (Prompt 4) */}
      {experiment.status === 'completed' && robustnessAudit && (
        <div className="detail-section full-width" style={{ marginTop: '1.5rem' }}>
          <RobustnessAuditPanel
            audit={robustnessAudit}
          />
        </div>
      )}

      <div className="detail-section full-width">
        <h3>Trials ({trials.length})</h3>
        {trials.length === 0 ? (
          <div className="detail-card">
            <p className="empty-text">
              {experiment.status === 'pending'
                ? 'No trials yet. Run the experiment to start training.'
                : 'No trials recorded.'}
            </p>
          </div>
        ) : (
          <div className="trials-list">
            {trials.map((trial) => (
              <div key={trial.id} className="trial-card">
                <div className="trial-header">
                  <span className="trial-name">{trial.variant_name}</span>
                  <StatusBadge status={trial.status} />
                </div>
                <div className="trial-meta">
                  <span>Split: {trial.data_split_strategy || 'N/A'}</span>
                  {trial.best_model_ref && (
                    <>
                      <span className="meta-separator">|</span>
                      <span>Best Model: {trial.best_model_ref}</span>
                    </>
                  )}
                </div>
                {trial.metrics_json && (
                  <div className="trial-metrics">
                    {Object.entries(trial.metrics_json)
                      .filter(([key]) => !key.includes('time') && !key.includes('num_'))
                      .slice(0, 4)
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
                          <span key={key} className="trial-metric">
                            {displayKey}: {typeof displayValue === 'number' ? displayValue.toFixed(4) : String(displayValue)}
                          </span>
                        );
                      })}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* AI Results Analysis Section */}
      <div className="detail-section full-width" style={{ marginTop: '2rem' }}>
        <ExperimentResultsPipeline
          experimentId={experiment.id}
          experimentStatus={experiment.status}
          onModelPromoted={fetchData}
        />
      </div>

      {/* AI Assistant with experiment context */}
      <ChatBot
        title="Experiment Assistant"
        contextType="experiment"
        context={{
          experiment: {
            id: experiment.id,
            name: experiment.name,
            description: experiment.description,
            status: experiment.status,
            primary_metric: experiment.primary_metric,
            metric_direction: experiment.metric_direction,
            best_metrics: experiment.best_metrics,
            best_model: experiment.best_model,
            experiment_plan: experiment.experiment_plan_json,
            created_at: experiment.created_at,
            updated_at: experiment.updated_at,
          },
          trials: trials.map(t => ({
            id: t.id,
            variant_name: t.variant_name,
            status: t.status,
            data_split_strategy: t.data_split_strategy,
            metrics: t.metrics_json,
            best_model_ref: t.best_model_ref,
          })),
        }}
      />

      {/* Training Options Modal */}
      <TrainingOptionsModal
        isOpen={showTrainingOptions}
        onClose={() => setShowTrainingOptions(false)}
        onConfirm={handleRunWithOptions}
        experimentName={experiment.name}
      />
    </div>
  );
}

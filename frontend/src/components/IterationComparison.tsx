import { Link } from 'react-router-dom';
import MetricExplainer from './MetricExplainer';
import type { DatasetContext } from './MetricExplainer';

interface ExperimentIteration {
  id: string;
  name: string;
  iteration_number: number;
  status: string;
  best_score: number | null;  // DEPRECATED: Use final_score instead
  primary_metric: string | null;
  improvement_summary: string | null;
  created_at: string;
  metrics: Record<string, number> | null;  // Full metrics for comparison
  // Holdout-based scoring (Make Holdout Score the Real Score)
  final_score: number | null;  // Canonical score from holdout evaluation
  val_score: number | null;  // Validation/CV score (NOT the final score)
  has_holdout: boolean;  // Whether holdout evaluation was performed
  overfitting_gap: number | null;  // Gap between val and holdout (positive = overfitting)
  score_source: 'holdout' | 'validation';  // Source of final_score
}

interface Props {
  iterations: ExperimentIteration[];
  currentExperimentId: string;
  primaryMetric: string | null;
  metricDirection?: 'maximize' | 'minimize';
  datasetContext?: DatasetContext;
}

export default function IterationComparison({
  iterations,
  currentExperimentId,
  primaryMetric,
  metricDirection = 'maximize',
  datasetContext,
}: Props) {
  if (iterations.length <= 1) {
    return null;
  }

  // Sort iterations by iteration_number
  const sortedIterations = [...iterations].sort((a, b) => a.iteration_number - b.iteration_number);

  // Helper to get the canonical score (final_score preferred, best_score as fallback)
  const getScore = (iter: ExperimentIteration): number | null => {
    return iter.final_score ?? iter.best_score;
  };

  // Find the best iteration (using final_score for comparisons)
  const completedIterations = sortedIterations.filter(i => i.status === 'completed' && getScore(i) !== null);
  const bestIteration = completedIterations.reduce((best, current) => {
    const bestScore = best ? getScore(best) : null;
    const currentScore = getScore(current);
    if (!best || currentScore === null) return best;
    if (bestScore === null) return current;

    if (metricDirection === 'maximize') {
      return currentScore > bestScore ? current : best;
    } else {
      return currentScore < bestScore ? current : best;
    }
  }, completedIterations[0] || null);

  // Calculate improvements for each iteration (using final_score)
  const iterationsWithImprovement = sortedIterations.map((iter, idx) => {
    const prevIter = idx > 0 ? sortedIterations[idx - 1] : null;
    let improvement: number | null = null;
    let improvementFromBaseline: number | null = null;
    const currentScore = getScore(iter);
    const prevScore = prevIter ? getScore(prevIter) : null;
    const baselineScore = getScore(sortedIterations[0]);

    if (currentScore !== null && prevScore !== null) {
      improvement = ((currentScore - prevScore) / Math.abs(prevScore || 1)) * 100;
    }

    if (currentScore !== null && baselineScore !== null) {
      improvementFromBaseline = ((currentScore - baselineScore) / Math.abs(baselineScore || 1)) * 100;
    }

    return { ...iter, improvement, improvementFromBaseline, currentScore };
  });

  // Get all unique metrics across iterations
  const allMetrics = new Set<string>();
  sortedIterations.forEach(iter => {
    if (iter.metrics) {
      Object.keys(iter.metrics).forEach(key => allMetrics.add(key));
    }
  });

  // Determine if improvement is positive based on metric direction
  const isImproved = (improvement: number | null): boolean | null => {
    if (improvement === null) return null;
    return metricDirection === 'maximize' ? improvement > 0 : improvement < 0;
  };

  return (
    <div style={{
      background: '#fff',
      borderRadius: '12px',
      border: '1px solid #e2e8f0',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '1rem 1.25rem',
        background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
        borderBottom: '1px solid #e2e8f0',
      }}>
        <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#1e293b' }}>
          Iteration Comparison
        </h3>
        <p style={{ margin: '0.25rem 0 0', fontSize: '0.8125rem', color: '#64748b' }}>
          Compare performance across {iterations.length} iterations
        </p>
      </div>

      {/* Comparison Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: '0.875rem',
        }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{
                padding: '0.75rem 1rem',
                textAlign: 'left',
                fontWeight: 600,
                color: '#475569',
                borderBottom: '1px solid #e2e8f0',
                position: 'sticky',
                left: 0,
                background: '#f8fafc',
                zIndex: 1,
              }}>
                Iteration
              </th>
              <th style={{
                padding: '0.75rem 1rem',
                textAlign: 'center',
                fontWeight: 600,
                color: '#475569',
                borderBottom: '1px solid #e2e8f0',
                minWidth: '120px',
              }}>
                {primaryMetric || 'Primary Score'}
              </th>
              <th style={{
                padding: '0.75rem 1rem',
                textAlign: 'center',
                fontWeight: 600,
                color: '#475569',
                borderBottom: '1px solid #e2e8f0',
                minWidth: '100px',
              }}>
                vs Previous
              </th>
              <th style={{
                padding: '0.75rem 1rem',
                textAlign: 'center',
                fontWeight: 600,
                color: '#475569',
                borderBottom: '1px solid #e2e8f0',
                minWidth: '100px',
              }}>
                vs Baseline
              </th>
              <th style={{
                padding: '0.75rem 1rem',
                textAlign: 'left',
                fontWeight: 600,
                color: '#475569',
                borderBottom: '1px solid #e2e8f0',
                minWidth: '250px',
              }}>
                What Changed
              </th>
            </tr>
          </thead>
          <tbody>
            {iterationsWithImprovement.map((iter, idx) => {
              const isCurrent = iter.id === currentExperimentId;
              const isBest = bestIteration?.id === iter.id;
              const improved = isImproved(iter.improvement);
              const improvedFromBaseline = isImproved(iter.improvementFromBaseline);

              return (
                <tr
                  key={iter.id}
                  style={{
                    background: isCurrent ? '#eff6ff' : isBest ? '#f0fdf4' : 'transparent',
                    borderBottom: '1px solid #e2e8f0',
                  }}
                >
                  {/* Iteration Number */}
                  <td style={{
                    padding: '0.75rem 1rem',
                    position: 'sticky',
                    left: 0,
                    background: isCurrent ? '#eff6ff' : isBest ? '#f0fdf4' : '#fff',
                    zIndex: 1,
                  }}>
                    <Link
                      to={`/experiments/${iter.id}`}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        textDecoration: 'none',
                        color: '#1e293b',
                      }}
                    >
                      <span style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: '28px',
                        height: '28px',
                        background: isCurrent ? '#3b82f6' : isBest ? '#22c55e' : '#e2e8f0',
                        color: isCurrent || isBest ? '#fff' : '#475569',
                        borderRadius: '50%',
                        fontWeight: 600,
                        fontSize: '0.8125rem',
                      }}>
                        {iter.iteration_number}
                      </span>
                      <div>
                        <div style={{ fontWeight: isCurrent ? 600 : 500 }}>
                          {iter.iteration_number === 1 ? 'Baseline' : `Iteration ${iter.iteration_number}`}
                          {isCurrent && <span style={{ color: '#3b82f6', marginLeft: '0.25rem' }}>(current)</span>}
                        </div>
                        {isBest && (
                          <div style={{
                            fontSize: '0.6875rem',
                            color: '#16a34a',
                            fontWeight: 500,
                          }}>
                            Best Performance
                          </div>
                        )}
                      </div>
                    </Link>
                  </td>

                  {/* Primary Score (using final_score/holdout when available) */}
                  <td style={{
                    padding: '0.75rem 1rem',
                    textAlign: 'center',
                  }}>
                    {iter.status === 'completed' && iter.currentScore !== null ? (
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.125rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                          <span style={{
                            fontWeight: 600,
                            fontSize: '1rem',
                            color: isBest ? '#16a34a' : (iter.has_holdout && iter.overfitting_gap && iter.overfitting_gap > 0.05) ? '#b45309' : '#1e293b',
                          }}>
                            {iter.currentScore.toFixed(4)}
                          </span>
                          {primaryMetric && (
                            <MetricExplainer
                              metricKey={primaryMetric}
                              value={iter.currentScore}
                              datasetContext={datasetContext}
                            />
                          )}
                        </div>
                        {/* Score source indicator */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                          {iter.has_holdout && (
                            <span style={{
                              fontSize: '0.625rem',
                              color: '#166534',
                              background: '#dcfce7',
                              padding: '0.125rem 0.375rem',
                              borderRadius: '3px',
                            }}>
                              holdout ✓
                            </span>
                          )}
                          {iter.has_holdout && iter.overfitting_gap != null && iter.overfitting_gap > 0.05 && (
                            <span style={{
                              fontSize: '0.625rem',
                              color: '#b45309',
                              background: '#fef3c7',
                              padding: '0.125rem 0.375rem',
                              borderRadius: '3px',
                            }}>
                              ⚠️ gap {(iter.overfitting_gap as number).toFixed(3)}
                            </span>
                          )}
                        </div>
                      </div>
                    ) : (
                      <span style={{
                        padding: '0.25rem 0.5rem',
                        background: iter.status === 'running' ? '#fef3c7' : iter.status === 'failed' ? '#fee2e2' : '#f1f5f9',
                        color: iter.status === 'running' ? '#b45309' : iter.status === 'failed' ? '#dc2626' : '#64748b',
                        borderRadius: '4px',
                        fontSize: '0.75rem',
                        fontWeight: 500,
                      }}>
                        {iter.status}
                      </span>
                    )}
                  </td>

                  {/* vs Previous */}
                  <td style={{
                    padding: '0.75rem 1rem',
                    textAlign: 'center',
                  }}>
                    {idx === 0 ? (
                      <span style={{ color: '#94a3b8', fontSize: '0.8125rem' }}>-</span>
                    ) : iter.improvement !== null ? (
                      <div style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        padding: '0.25rem 0.5rem',
                        background: improved ? '#dcfce7' : improved === false ? '#fee2e2' : '#f1f5f9',
                        color: improved ? '#16a34a' : improved === false ? '#dc2626' : '#64748b',
                        borderRadius: '4px',
                        fontWeight: 600,
                        fontSize: '0.8125rem',
                      }}>
                        {improved ? '↑' : improved === false ? '↓' : ''}
                        {iter.improvement >= 0 ? '+' : ''}{iter.improvement.toFixed(2)}%
                      </div>
                    ) : (
                      <span style={{ color: '#94a3b8', fontSize: '0.8125rem' }}>-</span>
                    )}
                  </td>

                  {/* vs Baseline */}
                  <td style={{
                    padding: '0.75rem 1rem',
                    textAlign: 'center',
                  }}>
                    {idx === 0 ? (
                      <span style={{ color: '#94a3b8', fontSize: '0.8125rem' }}>baseline</span>
                    ) : iter.improvementFromBaseline !== null ? (
                      <div style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        padding: '0.25rem 0.5rem',
                        background: improvedFromBaseline ? '#dbeafe' : improvedFromBaseline === false ? '#fef3c7' : '#f1f5f9',
                        color: improvedFromBaseline ? '#1d4ed8' : improvedFromBaseline === false ? '#b45309' : '#64748b',
                        borderRadius: '4px',
                        fontWeight: 600,
                        fontSize: '0.8125rem',
                      }}>
                        {improvedFromBaseline ? '↑' : improvedFromBaseline === false ? '↓' : ''}
                        {iter.improvementFromBaseline >= 0 ? '+' : ''}{iter.improvementFromBaseline.toFixed(2)}%
                      </div>
                    ) : (
                      <span style={{ color: '#94a3b8', fontSize: '0.8125rem' }}>-</span>
                    )}
                  </td>

                  {/* What Changed */}
                  <td style={{
                    padding: '0.75rem 1rem',
                    color: '#475569',
                    fontSize: '0.8125rem',
                    lineHeight: 1.5,
                  }}>
                    {idx === 0 ? (
                      <span style={{ color: '#64748b', fontStyle: 'italic' }}>
                        Original experiment
                      </span>
                    ) : iter.improvement_summary ? (
                      <span>{iter.improvement_summary}</span>
                    ) : (
                      <span style={{ color: '#94a3b8', fontStyle: 'italic' }}>
                        No summary available
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Summary Footer */}
      {completedIterations.length > 1 && bestIteration && (
        (() => {
          const bestScore = getScore(bestIteration);
          const baselineScore = getScore(sortedIterations[0]);
          const bestHasOverfitting = bestIteration.has_holdout && bestIteration.overfitting_gap != null && bestIteration.overfitting_gap > 0.05;

          return (
            <div style={{
              padding: '1rem 1.25rem',
              background: bestHasOverfitting
                ? 'linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)'
                : 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
              borderTop: `1px solid ${bestHasOverfitting ? '#f97316' : '#86efac'}`,
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '1.5rem',
                flexWrap: 'wrap',
              }}>
                <div>
                  <div style={{ fontSize: '0.75rem', color: bestHasOverfitting ? '#c2410c' : '#166534', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                    Best Performance
                    {bestIteration.has_holdout && <span style={{ fontSize: '0.625rem' }}>(holdout)</span>}
                    {bestHasOverfitting && <span>⚠️</span>}
                  </div>
                  <div style={{ fontSize: '1.125rem', fontWeight: 700, color: bestHasOverfitting ? '#ea580c' : '#15803d' }}>
                    Iteration {bestIteration.iteration_number}: {bestScore?.toFixed(4)}
                  </div>
                  {bestHasOverfitting && (
                    <div style={{ fontSize: '0.6875rem', color: '#b45309' }}>
                      Overfitting gap: {bestIteration.overfitting_gap?.toFixed(3)}
                    </div>
                  )}
                </div>

                {bestIteration.iteration_number > 1 && baselineScore !== null && bestScore !== null && (
                  <div>
                    <div style={{ fontSize: '0.75rem', color: '#166534', fontWeight: 500 }}>
                      Total Improvement from Baseline
                    </div>
                    <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#15803d' }}>
                      {((bestScore - baselineScore) / Math.abs(baselineScore || 1) * 100) >= 0 ? '+' : ''}
                      {((bestScore - baselineScore) / Math.abs(baselineScore || 1) * 100).toFixed(2)}%
                    </div>
                  </div>
                )}

                {bestIteration.id !== currentExperimentId && (
                  <Link
                    to={`/experiments/${bestIteration.id}`}
                    style={{
                      marginLeft: 'auto',
                      padding: '0.5rem 1rem',
                      background: '#22c55e',
                      color: '#fff',
                      borderRadius: '6px',
                      textDecoration: 'none',
                      fontWeight: 600,
                      fontSize: '0.8125rem',
                    }}
                  >
                    View Best Iteration
                  </Link>
                )}
              </div>
            </div>
          );
        })()
      )}
    </div>
  );
}

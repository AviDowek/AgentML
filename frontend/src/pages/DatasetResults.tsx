import { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { getDatasetExperiments, type DatasetResultsResponse, type DatasetExperiment } from '../services/api';

type SortField = 'name' | 'best_score' | 'created_at' | 'iteration_number' | 'status';
type SortDirection = 'asc' | 'desc';

export default function DatasetResults() {
  const { datasetSpecId } = useParams<{ datasetSpecId: string }>();
  const navigate = useNavigate();
  const [results, setResults] = useState<DatasetResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [includeIterations, setIncludeIterations] = useState(true);
  const [sortField, setSortField] = useState<SortField>('created_at');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [showMetricDetails, setShowMetricDetails] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetSpecId) return;
    loadResults();
  }, [datasetSpecId, includeIterations]);

  const loadResults = async () => {
    if (!datasetSpecId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getDatasetExperiments(datasetSpecId, includeIterations);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection(field === 'best_score' ? 'desc' : 'asc');
    }
  };

  const sortedExperiments = results?.experiments ? [...results.experiments].sort((a, b) => {
    let comparison = 0;
    switch (sortField) {
      case 'name':
        comparison = a.name.localeCompare(b.name);
        break;
      case 'best_score':
        comparison = (a.best_score ?? -Infinity) - (b.best_score ?? -Infinity);
        break;
      case 'created_at':
        comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        break;
      case 'iteration_number':
        comparison = a.iteration_number - b.iteration_number;
        break;
      case 'status':
        comparison = a.status.localeCompare(b.status);
        break;
    }
    return sortDirection === 'asc' ? comparison : -comparison;
  }) : [];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return { bg: '#dcfce7', color: '#166534', border: '#86efac' };
      case 'running': return { bg: '#fef3c7', color: '#b45309', border: '#fcd34d' };
      case 'pending': return { bg: '#e0e7ff', color: '#4338ca', border: '#a5b4fc' };
      case 'failed': return { bg: '#fee2e2', color: '#dc2626', border: '#fca5a5' };
      case 'cancelled': return { bg: '#f3f4f6', color: '#6b7280', border: '#d1d5db' };
      default: return { bg: '#f1f5f9', color: '#64748b', border: '#cbd5e1' };
    }
  };

  const formatDuration = (seconds: number | null) => {
    if (seconds === null || seconds === undefined) return '-';
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getScoreChange = (current: DatasetExperiment, previous: DatasetExperiment | undefined) => {
    if (!previous || current.best_score === null || previous.best_score === null) return null;
    const change = ((current.best_score - previous.best_score) / Math.abs(previous.best_score || 1)) * 100;
    return change;
  };

  if (loading) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <div style={{ display: 'inline-block', width: '32px', height: '32px', border: '3px solid #e2e8f0', borderTopColor: '#3b82f6', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
        <p style={{ marginTop: '1rem', color: '#64748b' }}>Loading results...</p>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '2rem' }}>
        <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '8px', padding: '1rem', color: '#dc2626' }}>
          <strong>Error:</strong> {error}
        </div>
        <button
          onClick={() => navigate(-1)}
          style={{ marginTop: '1rem', padding: '0.5rem 1rem', background: '#f1f5f9', border: '1px solid #e2e8f0', borderRadius: '6px', cursor: 'pointer' }}
        >
          Go Back
        </button>
      </div>
    );
  }

  if (!results) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
        No results found
      </div>
    );
  }

  return (
    <div className="dataset-results-page" style={{ maxWidth: '1400px', margin: '0 auto', padding: '1.5rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
          <button
            onClick={() => navigate(-1)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#64748b', display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.875rem' }}
          >
            <span style={{ fontSize: '1.25rem' }}>&larr;</span> Back
          </button>
        </div>
        <h1 style={{ margin: '0 0 0.5rem', fontSize: '1.75rem', fontWeight: 700, color: '#1e293b' }}>
          Dataset Results
        </h1>
        <p style={{ margin: 0, color: '#64748b', fontSize: '1rem' }}>
          Compare all experiments for <strong>{results.dataset_name}</strong>
        </p>
        {results.target_column && (
          <p style={{ margin: '0.25rem 0 0', color: '#94a3b8', fontSize: '0.875rem' }}>
            Target: {results.target_column}
          </p>
        )}
      </div>

      {/* Summary Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0', padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            Total Experiments
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: '#1e293b' }}>
            {results.total_experiments}
          </div>
        </div>

        <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0', padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            Completed
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: '#16a34a' }}>
            {results.completed_experiments}
          </div>
        </div>

        <div style={{ background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)', borderRadius: '12px', border: '1px solid #86efac', padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#166534', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            Best Score ({results.primary_metric || 'score'})
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: '#15803d' }}>
            {results.best_score !== null ? results.best_score.toFixed(4) : '-'}
          </div>
          {results.best_experiment_id && (
            <Link
              to={`/experiments/${results.best_experiment_id}`}
              style={{ fontSize: '0.75rem', color: '#16a34a', textDecoration: 'none' }}
            >
              View best experiment &rarr;
            </Link>
          )}
        </div>

        <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0', padding: '1.25rem' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            Primary Metric
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 600, color: '#3b82f6' }}>
            {results.primary_metric || 'Not set'}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem', color: '#475569', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={includeIterations}
            onChange={(e) => setIncludeIterations(e.target.checked)}
            style={{ accentColor: '#3b82f6' }}
          />
          Include auto-improve iterations
        </label>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.875rem', color: '#64748b' }}>
          <span>Sort by:</span>
          <select
            value={sortField}
            onChange={(e) => handleSort(e.target.value as SortField)}
            style={{ padding: '0.375rem 0.75rem', borderRadius: '6px', border: '1px solid #e2e8f0', background: '#fff', cursor: 'pointer' }}
          >
            <option value="created_at">Date</option>
            <option value="best_score">Score</option>
            <option value="name">Name</option>
            <option value="iteration_number">Iteration</option>
            <option value="status">Status</option>
          </select>
          <button
            onClick={() => setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')}
            style={{ padding: '0.375rem 0.5rem', borderRadius: '6px', border: '1px solid #e2e8f0', background: '#fff', cursor: 'pointer' }}
            title={sortDirection === 'asc' ? 'Ascending' : 'Descending'}
          >
            {sortDirection === 'asc' ? '↑' : '↓'}
          </button>
        </div>
      </div>

      {/* Experiments Table */}
      {sortedExperiments.length === 0 ? (
        <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0', padding: '3rem', textAlign: 'center' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🔬</div>
          <h3 style={{ margin: '0 0 0.5rem', color: '#1e293b' }}>No Experiments Yet</h3>
          <p style={{ margin: 0, color: '#64748b' }}>
            No experiments have been run on this dataset yet.
          </p>
        </div>
      ) : (
        <div style={{ background: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0', overflow: 'hidden' }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'left', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0', position: 'sticky', left: 0, background: '#f8fafc', zIndex: 1 }}>
                    Experiment
                  </th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0', minWidth: '90px' }}>
                    Status
                  </th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0', minWidth: '120px' }}>
                    Score
                  </th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0', minWidth: '80px' }}>
                    Change
                  </th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0', minWidth: '100px' }}>
                    Model
                  </th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0', minWidth: '80px' }}>
                    Duration
                  </th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'left', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0', minWidth: '200px' }}>
                    Description
                  </th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0', minWidth: '100px' }}>
                    Date
                  </th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center', fontWeight: 600, color: '#475569', borderBottom: '1px solid #e2e8f0', minWidth: '80px' }}>
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedExperiments.map((exp, idx) => {
                  const statusColors = getStatusColor(exp.status);
                  const isBest = exp.id === results.best_experiment_id;
                  const previousExp = sortedExperiments.find(e =>
                    e.iteration_number === exp.iteration_number - 1 &&
                    e.parent_experiment_id === exp.parent_experiment_id
                  ) || (exp.iteration_number > 1 ? sortedExperiments.find(e => e.iteration_number === exp.iteration_number - 1) : undefined);
                  const scoreChange = getScoreChange(exp, previousExp);

                  return (
                    <tr
                      key={exp.id}
                      style={{
                        background: isBest ? '#f0fdf4' : 'transparent',
                        borderBottom: '1px solid #e2e8f0',
                      }}
                    >
                      {/* Experiment Name */}
                      <td style={{ padding: '0.75rem 1rem', position: 'sticky', left: 0, background: isBest ? '#f0fdf4' : '#fff', zIndex: 1 }}>
                        <Link
                          to={`/experiments/${exp.id}`}
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
                            width: '24px',
                            height: '24px',
                            background: isBest ? '#22c55e' : '#e2e8f0',
                            color: isBest ? '#fff' : '#64748b',
                            borderRadius: '50%',
                            fontWeight: 600,
                            fontSize: '0.75rem',
                          }}>
                            {exp.iteration_number}
                          </span>
                          <div>
                            <div style={{ fontWeight: 600 }}>
                              {exp.name}
                              {isBest && (
                                <span style={{ marginLeft: '0.5rem', fontSize: '0.75rem', background: '#dcfce7', color: '#16a34a', padding: '0.125rem 0.375rem', borderRadius: '4px' }}>
                                  Best
                                </span>
                              )}
                            </div>
                            {exp.parent_experiment_id && (
                              <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                                Auto-improve iteration
                              </div>
                            )}
                          </div>
                        </Link>
                      </td>

                      {/* Status */}
                      <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block',
                          padding: '0.25rem 0.5rem',
                          background: statusColors.bg,
                          color: statusColors.color,
                          border: `1px solid ${statusColors.border}`,
                          borderRadius: '4px',
                          fontWeight: 500,
                          fontSize: '0.75rem',
                          textTransform: 'capitalize',
                        }}>
                          {exp.status}
                        </span>
                      </td>

                      {/* Score */}
                      <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                        {exp.best_score !== null ? (
                          <div>
                            <div style={{ fontWeight: 600, fontSize: '1rem', color: isBest ? '#16a34a' : '#1e293b' }}>
                              {exp.best_score.toFixed(4)}
                            </div>
                            <button
                              onClick={() => setShowMetricDetails(showMetricDetails === exp.id ? null : exp.id)}
                              style={{ background: 'none', border: 'none', color: '#3b82f6', fontSize: '0.6875rem', cursor: 'pointer', padding: 0 }}
                            >
                              {showMetricDetails === exp.id ? 'Hide details' : 'More metrics'}
                            </button>
                            {showMetricDetails === exp.id && exp.best_metrics && (
                              <div style={{
                                position: 'absolute',
                                background: '#fff',
                                border: '1px solid #e2e8f0',
                                borderRadius: '8px',
                                padding: '0.75rem',
                                boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                                zIndex: 10,
                                textAlign: 'left',
                                minWidth: '200px',
                                marginTop: '0.25rem',
                              }}>
                                {Object.entries(exp.best_metrics).slice(0, 8).map(([key, val]) => (
                                  <div key={key} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.25rem 0', fontSize: '0.75rem' }}>
                                    <span style={{ color: '#64748b' }}>{key}:</span>
                                    <span style={{ fontWeight: 500 }}>{typeof val === 'number' ? val.toFixed(4) : val}</span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        ) : (
                          <span style={{ color: '#94a3b8' }}>-</span>
                        )}
                      </td>

                      {/* Score Change */}
                      <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                        {scoreChange !== null ? (
                          <span style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.25rem',
                            padding: '0.25rem 0.5rem',
                            background: scoreChange > 0 ? '#dcfce7' : scoreChange < 0 ? '#fee2e2' : '#f1f5f9',
                            color: scoreChange > 0 ? '#16a34a' : scoreChange < 0 ? '#dc2626' : '#64748b',
                            borderRadius: '4px',
                            fontWeight: 600,
                            fontSize: '0.75rem',
                          }}>
                            {scoreChange > 0 ? '↑' : scoreChange < 0 ? '↓' : ''}
                            {scoreChange >= 0 ? '+' : ''}{scoreChange.toFixed(2)}%
                          </span>
                        ) : (
                          <span style={{ color: '#94a3b8', fontSize: '0.75rem' }}>
                            {exp.iteration_number === 1 ? 'baseline' : '-'}
                          </span>
                        )}
                      </td>

                      {/* Model Type */}
                      <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                        <span style={{ fontSize: '0.75rem', color: '#475569' }}>
                          {exp.best_model_type || '-'}
                        </span>
                      </td>

                      {/* Duration */}
                      <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                        <span style={{ fontSize: '0.75rem', color: '#475569' }}>
                          {formatDuration(exp.training_time_seconds)}
                        </span>
                      </td>

                      {/* Description */}
                      <td style={{ padding: '0.75rem 1rem' }}>
                        <span style={{ fontSize: '0.8125rem', color: '#64748b', lineHeight: 1.4 }}>
                          {exp.improvement_summary || exp.description || (
                            <em style={{ color: '#94a3b8' }}>No description</em>
                          )}
                        </span>
                      </td>

                      {/* Date */}
                      <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                        <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                          {formatDate(exp.created_at)}
                        </span>
                      </td>

                      {/* Actions */}
                      <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                        <Link
                          to={`/experiments/${exp.id}`}
                          style={{
                            display: 'inline-block',
                            padding: '0.375rem 0.75rem',
                            background: '#3b82f6',
                            color: '#fff',
                            borderRadius: '6px',
                            textDecoration: 'none',
                            fontWeight: 500,
                            fontSize: '0.75rem',
                          }}
                        >
                          View
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{ marginTop: '1rem', display: 'flex', gap: '1.5rem', flexWrap: 'wrap', fontSize: '0.75rem', color: '#64748b' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#22c55e' }} />
          Best performing experiment
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ padding: '0.125rem 0.25rem', background: '#dcfce7', color: '#16a34a', borderRadius: '2px' }}>↑</span>
          Score improved from previous
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ padding: '0.125rem 0.25rem', background: '#fee2e2', color: '#dc2626', borderRadius: '2px' }}>↓</span>
          Score decreased from previous
        </div>
      </div>
    </div>
  );
}

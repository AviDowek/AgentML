import { useState, useEffect } from 'react';
import MetricExplainer from './MetricExplainer';
import type { DatasetContext } from './MetricExplainer';

interface VisualizationData {
  experiment_id: string;
  model_version_id: string;
  task_type: string;
  target_column: string;
  primary_metric: string;
  metrics: Record<string, number>;
  visualization_type: string;
  message?: string;
  data?: any;
  dataset_context?: DatasetContext;
  chart_config?: {
    recommended_charts: string[];
    title: string;
    description: string;
    x_axis?: string;
    y_axis?: string;
  };
}

interface Props {
  experimentId: string;
  experimentStatus: string;
}

export default function ResultsVisualization({ experimentId, experimentStatus }: Props) {
  const [vizData, setVizData] = useState<VisualizationData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (experimentStatus !== 'completed') {
      setIsLoading(false);
      return;
    }

    const fetchVisualization = async () => {
      try {
        const response = await fetch(`http://localhost:8001/experiments/${experimentId}/visualization`);
        if (!response.ok) {
          throw new Error('Failed to fetch visualization data');
        }
        const data = await response.json();
        setVizData(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load visualization');
      } finally {
        setIsLoading(false);
      }
    };

    fetchVisualization();
  }, [experimentId, experimentStatus]);

  if (experimentStatus !== 'completed') {
    return null;
  }

  if (isLoading) {
    return (
      <div className="detail-card" style={{ padding: '2rem', textAlign: 'center' }}>
        <div className="spinner" style={{ margin: '0 auto 1rem' }}></div>
        <p style={{ color: '#6b7280' }}>Loading visualization...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="detail-card" style={{
        padding: '1.5rem',
        background: '#fef2f2',
        border: '1px solid #fecaca',
      }}>
        <p style={{ color: '#991b1b', margin: 0 }}>
          Could not load visualization: {error}
        </p>
      </div>
    );
  }

  if (!vizData || !vizData.data || vizData.visualization_type === 'error') {
    return (
      <div className="detail-card" style={{ padding: '1.5rem' }}>
        <p style={{ color: vizData?.visualization_type === 'error' ? '#b91c1c' : '#6b7280', margin: 0 }}>
          {vizData?.message || 'No visualization data available'}
        </p>
      </div>
    );
  }

  return (
    <div className="results-visualization">
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '1rem',
      }}>
        <div>
          <h4 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 600, color: '#1e293b' }}>
            {vizData.chart_config?.title || 'Results Visualization'}
          </h4>
          <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: '#64748b' }}>
            {vizData.chart_config?.description || 'Visual representation of model performance'}
          </p>
        </div>
        <span style={{
          padding: '0.25rem 0.75rem',
          background: '#f1f5f9',
          borderRadius: '9999px',
          fontSize: '0.75rem',
          fontWeight: 500,
          color: '#475569',
          textTransform: 'capitalize',
        }}>
          {vizData.task_type.replace('_', ' ')}
        </span>
      </div>

      {/* Render based on visualization type */}
      {vizData.visualization_type === 'classification' && (
        <ClassificationVisualization data={vizData.data} config={vizData.chart_config} datasetContext={vizData.dataset_context} />
      )}

      {vizData.visualization_type === 'regression' && (
        <RegressionVisualization data={vizData.data} config={vizData.chart_config} datasetContext={vizData.dataset_context} />
      )}

      {vizData.visualization_type === 'timeseries' && (
        <TimeSeriesVisualization data={vizData.data} config={vizData.chart_config} datasetContext={vizData.dataset_context} />
      )}

      {vizData.visualization_type === 'metrics_summary' && (
        <MetricsSummaryVisualization data={vizData.data} config={vizData.chart_config} datasetContext={vizData.dataset_context} />
      )}
    </div>
  );
}

// Classification Visualization Component
function ClassificationVisualization({ data, config, datasetContext }: { data: any; config?: any; datasetContext?: DatasetContext }) {
  const { classes, confusion_matrix, class_distribution, class_accuracy, total_samples, correct_predictions, sample_predictions } = data;
  const accuracy = total_samples > 0 ? (correct_predictions / total_samples * 100).toFixed(1) : '0';

  return (
    <div>
      {/* Overall Accuracy Banner */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
        padding: '1rem',
        background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
        borderRadius: '12px',
        marginBottom: '1.5rem',
        border: '1px solid #bbf7d0',
      }}>
        <div style={{
          width: '64px',
          height: '64px',
          borderRadius: '50%',
          background: '#22c55e',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff',
          fontSize: '1.25rem',
          fontWeight: 700,
        }}>
          {accuracy}%
        </div>
        <div>
          <div style={{ fontWeight: 600, fontSize: '1.125rem', color: '#166534' }}>
            Overall Accuracy
          </div>
          <div style={{ fontSize: '0.875rem', color: '#15803d' }}>
            {correct_predictions} out of {total_samples} predictions correct
          </div>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h5 style={{ margin: '0 0 0.75rem', fontSize: '0.9375rem', fontWeight: 600, color: '#374151' }}>
          Confusion Matrix
          <span style={{ fontWeight: 400, fontSize: '0.8125rem', color: '#6b7280', marginLeft: '0.5rem' }}>
            (rows = actual, columns = predicted)
          </span>
        </h5>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            borderCollapse: 'collapse',
            width: '100%',
            maxWidth: '500px',
            fontSize: '0.875rem',
          }}>
            <thead>
              <tr>
                <th style={{ padding: '0.5rem', border: '1px solid #e5e7eb', background: '#f9fafb' }}></th>
                {classes.map((cls: string) => (
                  <th key={cls} style={{
                    padding: '0.5rem',
                    border: '1px solid #e5e7eb',
                    background: '#f9fafb',
                    fontWeight: 600,
                  }}>
                    Pred: {cls}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {classes.map((actualCls: string) => (
                <tr key={actualCls}>
                  <td style={{
                    padding: '0.5rem',
                    border: '1px solid #e5e7eb',
                    background: '#f9fafb',
                    fontWeight: 600,
                  }}>
                    Actual: {actualCls}
                  </td>
                  {classes.map((predCls: string) => {
                    const count = confusion_matrix[actualCls]?.[predCls] || 0;
                    const isCorrect = actualCls === predCls;
                    return (
                      <td key={predCls} style={{
                        padding: '0.5rem',
                        border: '1px solid #e5e7eb',
                        textAlign: 'center',
                        background: isCorrect ? '#dcfce7' : count > 0 ? '#fef2f2' : '#fff',
                        color: isCorrect ? '#166534' : count > 0 ? '#991b1b' : '#374151',
                        fontWeight: isCorrect ? 600 : 400,
                      }}>
                        {count}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ margin: '0.5rem 0 0', fontSize: '0.75rem', color: '#6b7280' }}>
          Green diagonal = correct predictions. Red = errors.
        </p>
      </div>

      {/* Class Accuracy Bars */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h5 style={{ margin: '0 0 0.75rem', fontSize: '0.9375rem', fontWeight: 600, color: '#374151' }}>
          Accuracy by Class
        </h5>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {classes.map((cls: string) => {
            const acc = (class_accuracy[cls] || 0) * 100;
            return (
              <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ width: '80px', fontSize: '0.875rem', fontWeight: 500, color: '#374151' }}>
                  {cls}
                </div>
                <div style={{
                  flex: 1,
                  height: '24px',
                  background: '#e5e7eb',
                  borderRadius: '4px',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    width: `${acc}%`,
                    height: '100%',
                    background: acc >= 80 ? '#22c55e' : acc >= 60 ? '#eab308' : '#ef4444',
                    borderRadius: '4px',
                    transition: 'width 0.3s',
                  }} />
                </div>
                <div style={{ width: '50px', fontSize: '0.875rem', fontWeight: 600, color: '#374151', textAlign: 'right' }}>
                  {acc.toFixed(0)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Sample Predictions */}
      {sample_predictions && sample_predictions.length > 0 && (
        <div>
          <h5 style={{ margin: '0 0 0.75rem', fontSize: '0.9375rem', fontWeight: 600, color: '#374151' }}>
            Sample Predictions
          </h5>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
            gap: '0.5rem',
            maxHeight: '200px',
            overflowY: 'auto',
          }}>
            {sample_predictions.slice(0, 20).map((pred: any, idx: number) => (
              <div key={idx} style={{
                padding: '0.5rem 0.75rem',
                borderRadius: '6px',
                background: pred.correct ? '#f0fdf4' : '#fef2f2',
                border: `1px solid ${pred.correct ? '#bbf7d0' : '#fecaca'}`,
                fontSize: '0.8125rem',
              }}>
                <span style={{ color: pred.correct ? '#166534' : '#991b1b' }}>
                  {pred.correct ? '✓' : '✗'}
                </span>
                <span style={{ marginLeft: '0.5rem', color: '#6b7280' }}>Actual:</span>
                <span style={{ marginLeft: '0.25rem', fontWeight: 500 }}>{pred.actual}</span>
                <span style={{ marginLeft: '0.5rem', color: '#6b7280' }}>Pred:</span>
                <span style={{ marginLeft: '0.25rem', fontWeight: 500 }}>{pred.predicted}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Regression Visualization Component
function RegressionVisualization({ data, config, datasetContext }: { data: any; config?: any; datasetContext?: DatasetContext }) {
  const { points, summary, worst_predictions, error_distribution, total_samples } = data;

  // Create a simple SVG scatter plot
  const svgWidth = 400;
  const svgHeight = 300;
  const padding = 40;

  const minVal = Math.min(summary.min_actual, summary.min_predicted);
  const maxVal = Math.max(summary.max_actual, summary.max_predicted);
  const range = maxVal - minVal || 1;

  const scaleX = (val: number) => padding + ((val - minVal) / range) * (svgWidth - 2 * padding);
  const scaleY = (val: number) => svgHeight - padding - ((val - minVal) / range) * (svgHeight - 2 * padding);

  return (
    <div>
      {/* Summary Stats */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '1rem',
        marginBottom: '1.5rem',
      }}>
        <div style={{
          padding: '1rem',
          background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
          borderRadius: '12px',
          border: '1px solid #93c5fd',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#1d4ed8' }}>
            {(summary.r_squared * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '0.8125rem', color: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            R² Score <MetricExplainer metricKey="r2" value={summary.r_squared} datasetContext={datasetContext} />
          </div>
        </div>
        <div style={{
          padding: '1rem',
          background: '#f8fafc',
          borderRadius: '12px',
          border: '1px solid #e2e8f0',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#475569' }}>
            {summary.mean_error.toFixed(2)}
          </div>
          <div style={{ fontSize: '0.8125rem', color: '#64748b' }}>
            Mean Error
          </div>
        </div>
        <div style={{
          padding: '1rem',
          background: '#f8fafc',
          borderRadius: '12px',
          border: '1px solid #e2e8f0',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#475569' }}>
            {total_samples}
          </div>
          <div style={{ fontSize: '0.8125rem', color: '#64748b' }}>
            Samples
          </div>
        </div>
      </div>

      {/* Scatter Plot */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h5 style={{ margin: '0 0 0.75rem', fontSize: '0.9375rem', fontWeight: 600, color: '#374151' }}>
          Predicted vs Actual Values
        </h5>
        <div style={{ background: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb', padding: '0.5rem' }}>
          <svg width="100%" viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ maxWidth: '500px' }}>
            {/* Perfect prediction line */}
            <line
              x1={scaleX(minVal)}
              y1={scaleY(minVal)}
              x2={scaleX(maxVal)}
              y2={scaleY(maxVal)}
              stroke="#94a3b8"
              strokeWidth="2"
              strokeDasharray="5,5"
            />

            {/* Data points */}
            {points.slice(0, 100).map((point: any, idx: number) => (
              <circle
                key={idx}
                cx={scaleX(point.actual)}
                cy={scaleY(point.predicted)}
                r="4"
                fill="#3b82f6"
                opacity="0.6"
              />
            ))}

            {/* Axes labels */}
            <text x={svgWidth / 2} y={svgHeight - 5} textAnchor="middle" fontSize="12" fill="#6b7280">
              {config?.x_axis || 'Actual Value'}
            </text>
            <text
              x={15}
              y={svgHeight / 2}
              textAnchor="middle"
              fontSize="12"
              fill="#6b7280"
              transform={`rotate(-90, 15, ${svgHeight / 2})`}
            >
              {config?.y_axis || 'Predicted'}
            </text>
          </svg>
        </div>
        <p style={{ margin: '0.5rem 0 0', fontSize: '0.75rem', color: '#6b7280' }}>
          Points on the diagonal line = perfect predictions. Closer to line = better.
        </p>
      </div>

      {/* Error Distribution */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h5 style={{ margin: '0 0 0.75rem', fontSize: '0.9375rem', fontWeight: 600, color: '#374151' }}>
          Error Distribution
        </h5>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          {Object.entries(error_distribution).map(([bucket, count]) => {
            const pct = total_samples > 0 ? ((count as number) / total_samples * 100) : 0;
            const isGood = bucket === '< 1%' || bucket === '1-5%';
            return (
              <div key={bucket} style={{
                flex: '1 1 100px',
                padding: '0.75rem',
                borderRadius: '8px',
                background: isGood ? '#f0fdf4' : '#fef2f2',
                border: `1px solid ${isGood ? '#bbf7d0' : '#fecaca'}`,
                textAlign: 'center',
              }}>
                <div style={{ fontWeight: 600, color: isGood ? '#166534' : '#991b1b' }}>
                  {pct.toFixed(0)}%
                </div>
                <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>{bucket} error</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Worst Predictions */}
      {worst_predictions && worst_predictions.length > 0 && (
        <div>
          <h5 style={{ margin: '0 0 0.75rem', fontSize: '0.9375rem', fontWeight: 600, color: '#374151' }}>
            Largest Errors (for investigation)
          </h5>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '0.8125rem', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f9fafb' }}>
                  <th style={{ padding: '0.5rem', border: '1px solid #e5e7eb', textAlign: 'left' }}>Actual</th>
                  <th style={{ padding: '0.5rem', border: '1px solid #e5e7eb', textAlign: 'left' }}>Predicted</th>
                  <th style={{ padding: '0.5rem', border: '1px solid #e5e7eb', textAlign: 'left' }}>Error</th>
                  <th style={{ padding: '0.5rem', border: '1px solid #e5e7eb', textAlign: 'left' }}>% Off</th>
                </tr>
              </thead>
              <tbody>
                {worst_predictions.slice(0, 5).map((pred: any, idx: number) => (
                  <tr key={idx}>
                    <td style={{ padding: '0.5rem', border: '1px solid #e5e7eb' }}>{pred.actual.toFixed(2)}</td>
                    <td style={{ padding: '0.5rem', border: '1px solid #e5e7eb' }}>{pred.predicted.toFixed(2)}</td>
                    <td style={{ padding: '0.5rem', border: '1px solid #e5e7eb', color: '#dc2626' }}>{pred.error.toFixed(2)}</td>
                    <td style={{ padding: '0.5rem', border: '1px solid #e5e7eb', color: '#dc2626' }}>{pred.pct_error.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// Time Series Visualization Component
function TimeSeriesVisualization({ data, config, datasetContext }: { data: any; config?: any; datasetContext?: DatasetContext }) {
  const { points, direction_accuracy, summary, total_samples } = data;

  // Create line chart SVG
  const svgWidth = 600;
  const svgHeight = 300;
  const padding = 50;

  if (!points || points.length === 0) {
    return <p style={{ color: '#6b7280' }}>No time series data available</p>;
  }

  const actuals = points.map((p: any) => p.actual);
  const predictions = points.map((p: any) => p.predicted);
  const allVals = [...actuals, ...predictions];
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const range = maxVal - minVal || 1;

  const scaleX = (idx: number) => padding + (idx / (points.length - 1 || 1)) * (svgWidth - 2 * padding);
  const scaleY = (val: number) => svgHeight - padding - ((val - minVal) / range) * (svgHeight - 2 * padding);

  const actualPath = points.map((p: any, i: number) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(p.actual)}`).join(' ');
  const predPath = points.map((p: any, i: number) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(p.predicted)}`).join(' ');

  return (
    <div>
      {/* Summary */}
      <div style={{
        display: 'flex',
        gap: '1rem',
        marginBottom: '1.5rem',
        flexWrap: 'wrap',
      }}>
        <div style={{
          padding: '1rem',
          background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
          borderRadius: '12px',
          border: '1px solid #93c5fd',
          flex: '1 1 150px',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#1d4ed8' }}>
            {(direction_accuracy * 100).toFixed(0)}%
          </div>
          <div style={{ fontSize: '0.8125rem', color: '#3b82f6' }}>
            Direction Accuracy
          </div>
          <div style={{ fontSize: '0.6875rem', color: '#64748b', marginTop: '0.25rem' }}>
            How often does it predict up/down correctly?
          </div>
        </div>
        <div style={{
          padding: '1rem',
          background: '#f8fafc',
          borderRadius: '12px',
          border: '1px solid #e2e8f0',
          flex: '1 1 150px',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#475569' }}>
            {total_samples}
          </div>
          <div style={{ fontSize: '0.8125rem', color: '#64748b' }}>
            Time Points
          </div>
        </div>
      </div>

      {/* Line Chart */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h5 style={{ margin: '0 0 0.75rem', fontSize: '0.9375rem', fontWeight: 600, color: '#374151' }}>
          Actual vs Predicted Over Time
        </h5>
        <div style={{ background: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb', padding: '0.5rem' }}>
          <svg width="100%" viewBox={`0 0 ${svgWidth} ${svgHeight}`}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct) => (
              <line
                key={pct}
                x1={padding}
                y1={padding + pct * (svgHeight - 2 * padding)}
                x2={svgWidth - padding}
                y2={padding + pct * (svgHeight - 2 * padding)}
                stroke="#e5e7eb"
                strokeWidth="1"
              />
            ))}

            {/* Actual line */}
            <path d={actualPath} fill="none" stroke="#22c55e" strokeWidth="2" />

            {/* Predicted line */}
            <path d={predPath} fill="none" stroke="#3b82f6" strokeWidth="2" strokeDasharray="5,3" />

            {/* Legend */}
            <rect x={svgWidth - 140} y={10} width="130" height="50" fill="#fff" stroke="#e5e7eb" rx="4" />
            <line x1={svgWidth - 130} y1={28} x2={svgWidth - 100} y2={28} stroke="#22c55e" strokeWidth="2" />
            <text x={svgWidth - 95} y={32} fontSize="11" fill="#374151">Actual</text>
            <line x1={svgWidth - 130} y1={45} x2={svgWidth - 100} y2={45} stroke="#3b82f6" strokeWidth="2" strokeDasharray="5,3" />
            <text x={svgWidth - 95} y={49} fontSize="11" fill="#374151">Predicted</text>

            {/* Axis labels */}
            <text x={svgWidth / 2} y={svgHeight - 10} textAnchor="middle" fontSize="12" fill="#6b7280">
              Time
            </text>
          </svg>
        </div>
        <p style={{ margin: '0.5rem 0 0', fontSize: '0.75rem', color: '#6b7280' }}>
          Green = actual values, Blue dashed = predictions. Closer lines = better predictions.
        </p>
      </div>
    </div>
  );
}

// Metrics Summary Visualization (fallback)
function MetricsSummaryVisualization({ data, config, datasetContext }: { data: any; config?: any; datasetContext?: DatasetContext }) {
  const { metrics, metric_quality, primary_metric } = data;

  const qualityColors: Record<string, string> = {
    excellent: '#22c55e',
    good: '#84cc16',
    fair: '#eab308',
    poor: '#f97316',
    very_poor: '#ef4444',
    unknown: '#94a3b8',
  };

  return (
    <div>
      {/* Primary Metric Highlight */}
      {primary_metric && (
        <div style={{
          padding: '1.5rem',
          background: `linear-gradient(135deg, ${qualityColors[primary_metric.quality]}15 0%, ${qualityColors[primary_metric.quality]}25 100%)`,
          borderRadius: '12px',
          border: `2px solid ${qualityColors[primary_metric.quality]}`,
          marginBottom: '1.5rem',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: qualityColors[primary_metric.quality] }}>
            {typeof primary_metric.value === 'number' ? primary_metric.value.toFixed(4) : primary_metric.value}
          </div>
          <div style={{ fontSize: '1rem', color: '#374151', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.25rem' }}>
            {primary_metric.name}
            <MetricExplainer metricKey={primary_metric.name} value={primary_metric.value} datasetContext={datasetContext} />
          </div>
          <div style={{
            marginTop: '0.5rem',
            padding: '0.25rem 0.75rem',
            background: qualityColors[primary_metric.quality],
            color: '#fff',
            borderRadius: '9999px',
            fontSize: '0.75rem',
            fontWeight: 600,
            display: 'inline-block',
            textTransform: 'capitalize',
          }}>
            {primary_metric.quality.replace('_', ' ')}
          </div>
        </div>
      )}

      {/* All Metrics Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
        gap: '0.75rem',
      }}>
        {Object.entries(metrics).map(([key, value]) => {
          const quality = metric_quality[key] || 'unknown';
          return (
            <div key={key} style={{
              padding: '0.75rem',
              background: '#f8fafc',
              borderRadius: '8px',
              border: '1px solid #e2e8f0',
              borderLeft: `4px solid ${qualityColors[quality]}`,
            }}>
              <div style={{ fontSize: '1.125rem', fontWeight: 600, color: '#1e293b' }}>
                {typeof value === 'number' ? value.toFixed(4) : String(value)}
              </div>
              <div style={{
                fontSize: '0.75rem',
                color: '#64748b',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
              }}>
                {key}
                <MetricExplainer metricKey={key} value={value as number} datasetContext={datasetContext} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

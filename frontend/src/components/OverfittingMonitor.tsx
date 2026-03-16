/**
 * Overfitting Monitor Component
 *
 * Displays overfitting analysis for experiment iterations, showing:
 * - Overall risk level with visual indicator
 * - Holdout score trend chart
 * - Per-iteration status with risk percentages
 * - Recommendations for the user
 */
import React, { useState, useEffect } from 'react';

// Define types inline to avoid import issues during hot reload
type OverfittingRiskLevel = 'low' | 'medium' | 'high';
type OverfittingTrend = 'improving' | 'stable' | 'degrading' | 'unknown';
type OverfittingRecommendation = 'continue' | 'warning' | 'stop';
type OverfittingIterationStatus = 'healthy' | 'warning' | 'high_risk' | 'best' | 'unknown';

interface OverfittingIterationEntry {
  experiment_id: string;
  iteration: number;
  holdout_score: number;
  metric: string;
  overfitting_risk: number;
  status: OverfittingIterationStatus;
  is_best: boolean;
}

interface OverfittingReport {
  experiment_id: string;
  iteration_number: number;
  total_iterations: number;
  overall_risk: number;
  risk_level: OverfittingRiskLevel;
  trend: OverfittingTrend;
  best_iteration: number;
  best_score: number;
  current_score: number;
  recommendation: OverfittingRecommendation;
  message: string;
  iterations: OverfittingIterationEntry[];
}

interface OverfittingMonitorProps {
  projectId: string;
  experimentId: string;
  onIterationClick?: (experimentId: string) => void;
}

// Risk level colors
const getRiskColor = (level: OverfittingRiskLevel | string): string => {
  switch (level) {
    case 'low':
      return '#22c55e'; // green
    case 'medium':
      return '#f59e0b'; // amber
    case 'high':
      return '#ef4444'; // red
    default:
      return '#6b7280'; // gray
  }
};

// Status badge colors
const getStatusColor = (status: string): string => {
  switch (status) {
    case 'healthy':
      return '#22c55e';
    case 'best':
      return '#3b82f6'; // blue
    case 'warning':
      return '#f59e0b';
    case 'high_risk':
      return '#ef4444';
    default:
      return '#6b7280';
  }
};

// Recommendation icon
const getRecommendationIcon = (rec: OverfittingRecommendation): string => {
  switch (rec) {
    case 'continue':
      return '✓';
    case 'warning':
      return '⚠';
    case 'stop':
      return '✕';
    default:
      return '?';
  }
};

export const OverfittingMonitor: React.FC<OverfittingMonitorProps> = ({
  projectId,
  experimentId,
  onIterationClick,
}) => {
  const [report, setReport] = useState<OverfittingReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `/api/v1/projects/${projectId}/experiments/${experimentId}/iterations/overfitting`
        );
        if (!response.ok) {
          throw new Error('Failed to fetch overfitting report');
        }
        const data = await response.json();
        setReport(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    if (projectId && experimentId) {
      fetchReport();
    }
  }, [projectId, experimentId]);

  if (loading) {
    return (
      <div className="p-4 bg-gray-800 rounded-lg animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-20 bg-gray-700 rounded"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-gray-800 rounded-lg text-red-400">
        Error loading overfitting data: {error}
      </div>
    );
  }

  if (!report || report.iterations.length === 0) {
    return (
      <div className="p-4 bg-gray-800 rounded-lg text-gray-400">
        <div className="text-sm">No holdout validation data available yet.</div>
        <div className="text-xs mt-1 opacity-75">
          Holdout scores will appear after experiments complete.
        </div>
      </div>
    );
  }

  const maxScore = Math.max(...report.iterations.map((i) => i.holdout_score), 0.001);
  const minScore = Math.min(...report.iterations.map((i) => i.holdout_score), 0);
  const scoreRange = maxScore - minScore || 0.1;

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      {/* Header with overall risk */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">Overfitting Monitor</h3>
        <div className="flex items-center gap-2">
          <div
            className="px-2 py-1 rounded text-xs font-medium"
            style={{
              backgroundColor: `${getRiskColor(report.risk_level)}20`,
              color: getRiskColor(report.risk_level),
            }}
          >
            {report.risk_level.toUpperCase()} RISK ({report.overall_risk}%)
          </div>
        </div>
      </div>

      {/* Recommendation banner */}
      <div
        className="p-3 rounded-lg text-sm flex items-start gap-2"
        style={{
          backgroundColor:
            report.recommendation === 'stop'
              ? '#ef444420'
              : report.recommendation === 'warning'
              ? '#f59e0b20'
              : '#22c55e20',
          borderLeft: `3px solid ${
            report.recommendation === 'stop'
              ? '#ef4444'
              : report.recommendation === 'warning'
              ? '#f59e0b'
              : '#22c55e'
          }`,
        }}
      >
        <span className="text-lg">{getRecommendationIcon(report.recommendation)}</span>
        <div>
          <div className="font-medium text-gray-200">{report.message}</div>
          {report.recommendation === 'stop' && (
            <div className="text-xs text-gray-400 mt-1">
              Consider using iteration {report.best_iteration} with score{' '}
              {report.best_score.toFixed(4)}
            </div>
          )}
        </div>
      </div>

      {/* Score summary */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="bg-gray-700/50 rounded p-2">
          <div className="text-xs text-gray-400">Best Score</div>
          <div className="text-lg font-mono text-blue-400">
            {report.best_score.toFixed(4)}
          </div>
          <div className="text-xs text-gray-500">Iteration {report.best_iteration}</div>
        </div>
        <div className="bg-gray-700/50 rounded p-2">
          <div className="text-xs text-gray-400">Current Score</div>
          <div className="text-lg font-mono text-gray-200">
            {report.current_score.toFixed(4)}
          </div>
          <div className="text-xs text-gray-500">Iteration {report.total_iterations}</div>
        </div>
        <div className="bg-gray-700/50 rounded p-2">
          <div className="text-xs text-gray-400">Trend</div>
          <div
            className="text-lg font-medium"
            style={{
              color:
                report.trend === 'improving'
                  ? '#22c55e'
                  : report.trend === 'degrading'
                  ? '#ef4444'
                  : '#f59e0b',
            }}
          >
            {report.trend === 'improving'
              ? '↑'
              : report.trend === 'degrading'
              ? '↓'
              : '→'}{' '}
            {report.trend}
          </div>
        </div>
      </div>

      {/* Iteration timeline */}
      <div className="space-y-2">
        <div className="text-xs text-gray-400 uppercase tracking-wide">
          Holdout Scores by Iteration
        </div>
        <div className="space-y-1">
          {report.iterations.map((iter, idx) => (
            <div
              key={iter.experiment_id}
              className={`flex items-center gap-2 p-2 rounded cursor-pointer hover:bg-gray-700/50 transition-colors ${
                iter.is_best ? 'bg-blue-900/30 border border-blue-700/50' : ''
              }`}
              onClick={() => onIterationClick?.(iter.experiment_id)}
            >
              {/* Iteration number */}
              <div className="w-8 text-xs text-gray-400 text-center">#{iter.iteration}</div>

              {/* Score bar */}
              <div className="flex-1 h-6 bg-gray-700 rounded overflow-hidden relative">
                <div
                  className="h-full transition-all duration-300"
                  style={{
                    width: `${((iter.holdout_score - minScore) / scoreRange) * 100}%`,
                    backgroundColor: getStatusColor(iter.status),
                    opacity: 0.7,
                  }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xs font-mono text-gray-200">
                    {iter.holdout_score.toFixed(4)}
                  </span>
                </div>
              </div>

              {/* Risk percentage */}
              <div
                className="w-16 text-right text-xs font-mono"
                style={{ color: getStatusColor(iter.status) }}
              >
                {iter.overfitting_risk}% risk
              </div>

              {/* Status badge */}
              <div
                className="w-16 text-center text-xs px-1 py-0.5 rounded"
                style={{
                  backgroundColor: `${getStatusColor(iter.status)}20`,
                  color: getStatusColor(iter.status),
                }}
              >
                {iter.is_best ? '★ BEST' : iter.status}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 text-xs text-gray-500 pt-2 border-t border-gray-700">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded" style={{ backgroundColor: '#22c55e' }}></div>
          <span>Healthy</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded" style={{ backgroundColor: '#3b82f6' }}></div>
          <span>Best</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded" style={{ backgroundColor: '#f59e0b' }}></div>
          <span>Warning</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded" style={{ backgroundColor: '#ef4444' }}></div>
          <span>High Risk</span>
        </div>
      </div>
    </div>
  );
};

export default OverfittingMonitor;

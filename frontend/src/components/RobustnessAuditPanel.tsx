/**
 * Robustness Audit Panel Component
 *
 * Displays the results of a ROBUSTNESS_AUDIT agent step, including:
 * - Overall overfitting risk level with visual indicator
 * - Suspicious patterns found
 * - Baseline comparison summary
 * - Recommendations
 * - Link to view full agent thinking/logs
 */
import React, { useState } from 'react';

// Types for robustness audit output
export interface SuspiciousPattern {
  type: 'train_val_gap' | 'metric_spike' | 'baseline_concern' | 'cv_variance' | 'data_leakage_suspicion' | string;
  severity: 'low' | 'medium' | 'high';
  description: string;
}

export interface TrainValAnalysis {
  worst_gap?: number;
  avg_gap?: number;
  interpretation?: string;
}

export interface BaselineComparison {
  baseline_type: 'majority_class' | 'mean_predictor' | 'random' | 'none_available' | string;
  baseline_metric?: number;
  best_model_metric?: number;
  relative_improvement?: number;
  interpretation?: string;
}

export interface CVAnalysis {
  fold_variance?: number;
  interpretation?: string;
}

// Prompt 6: Leakage feature with importance info
export interface ConcerningLeakageFeature {
  column: string;
  reason: string;
  severity: 'low' | 'medium' | 'high';
  detection_method: 'name' | 'correlation' | 'lineage';
  importance?: number;
  importance_rank?: number;
}

export interface RobustnessAuditResult {
  overfitting_risk: 'low' | 'medium' | 'high' | 'unknown';
  // Prompt 4 new fields
  leakage_suspected?: boolean;
  time_split_suspicious?: boolean;
  warnings?: string[];
  metrics_summary?: {
    best_val_metric?: number | null;
    primary_metric?: string | null;
    train_val_gap_worst?: number | null;
    train_val_gap_avg?: number | null;
    cv_variance?: number | null;
    baseline_value?: number | null;
    baseline_type?: string | null;
  };
  baseline_metrics?: Record<string, unknown>;
  is_time_based?: boolean;
  task_type?: string;
  // Prompt 5 new fields
  too_good_to_be_true?: boolean;
  risk_adjusted_score?: number | null;
  risk_level?: 'low' | 'medium' | 'high' | 'critical' | 'unknown';
  requires_override?: boolean;
  risk_reason?: string;
  // Prompt 6 new fields
  leakage_in_important_features?: boolean;
  concerning_leakage_features?: ConcerningLeakageFeature[];
  leakage_candidates_count?: number;
  // Existing fields
  train_val_analysis?: TrainValAnalysis;
  suspicious_patterns: SuspiciousPattern[];
  baseline_comparison?: BaselineComparison;
  cv_analysis?: CVAnalysis;
  recommendations: string[];
  natural_language_summary: string;
}

interface RobustnessAuditPanelProps {
  audit: RobustnessAuditResult;
  onShowFullThinking?: () => void;
}

// Risk level colors and styles
const getRiskColor = (level: string): string => {
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

const getRiskEmoji = (level: string): string => {
  switch (level) {
    case 'low':
      return '✅';
    case 'medium':
      return '⚠️';
    case 'high':
      return '🚨';
    default:
      return '❓';
  }
};

const getSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'low':
      return '#22c55e';
    case 'medium':
      return '#f59e0b';
    case 'high':
      return '#ef4444';
    default:
      return '#6b7280';
  }
};

const getPatternIcon = (type: string): string => {
  switch (type) {
    case 'train_val_gap':
      return '📊';
    case 'metric_spike':
      return '📈';
    case 'baseline_concern':
      return '📉';
    case 'cv_variance':
      return '📏';
    case 'data_leakage_suspicion':
      return '🔓';
    case 'leakage_in_important_feature':
      return '⚠️';
    case 'too_good_to_be_true':
      return '🎯';
    default:
      return '⚡';
  }
};

export const RobustnessAuditPanel: React.FC<RobustnessAuditPanelProps> = ({
  audit,
  onShowFullThinking,
}) => {
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      {/* Header with overall risk */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <span>🔍</span>
          Robustness Audit
        </h3>
        <div
          className="px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-2"
          style={{
            backgroundColor: `${getRiskColor(audit.overfitting_risk)}20`,
            color: getRiskColor(audit.overfitting_risk),
            border: `1px solid ${getRiskColor(audit.overfitting_risk)}40`,
          }}
        >
          <span>{getRiskEmoji(audit.overfitting_risk)}</span>
          <span>{audit.overfitting_risk.toUpperCase()} RISK</span>
        </div>
      </div>

      {/* Leakage Warning Banner (Prompt 4) */}
      {audit.leakage_suspected && (
        <div className="p-3 rounded-lg flex items-start gap-3" style={{
          backgroundColor: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
        }}>
          <span className="text-lg">🔓</span>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs px-2 py-0.5 rounded font-bold" style={{
                backgroundColor: '#ef4444',
                color: 'white',
              }}>
                LEAKAGE SUSPECTED
              </span>
            </div>
            <div className="text-sm text-red-300">
              Label-shuffle test indicates potential data leakage. Features may encode target information.
            </div>
          </div>
        </div>
      )}

      {/* Time-Split Warning Banner (Prompt 4) */}
      {audit.time_split_suspicious && (
        <div className="p-3 rounded-lg flex items-start gap-3" style={{
          backgroundColor: 'rgba(245, 158, 11, 0.15)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
        }}>
          <span className="text-lg">⏰</span>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs px-2 py-0.5 rounded font-bold" style={{
                backgroundColor: '#f59e0b',
                color: 'white',
              }}>
                TIME-SPLIT ISSUE
              </span>
            </div>
            <div className="text-sm text-amber-300">
              Time-based data is using random/stratified split, which may cause temporal leakage.
              Consider using time-based or group-time split instead.
            </div>
          </div>
        </div>
      )}

      {/* Too Good To Be True Warning Banner (Prompt 5) */}
      {audit.too_good_to_be_true && (
        <div className="p-3 rounded-lg flex items-start gap-3" style={{
          backgroundColor: 'rgba(168, 85, 247, 0.15)',
          border: '1px solid rgba(168, 85, 247, 0.3)',
        }}>
          <span className="text-lg">🎯</span>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs px-2 py-0.5 rounded font-bold" style={{
                backgroundColor: '#a855f7',
                color: 'white',
              }}>
                TOO GOOD TO BE TRUE
              </span>
            </div>
            <div className="text-sm text-purple-300">
              Model performance on time-based classification appears suspiciously high.
              This often indicates data leakage, look-ahead bias, or improper temporal validation.
              Review features for temporal information that wouldn't be available at prediction time.
            </div>
          </div>
        </div>
      )}

      {/* Leakage in Important Features Warning Banner (Prompt 6) */}
      {audit.leakage_in_important_features && audit.concerning_leakage_features && audit.concerning_leakage_features.length > 0 && (
        <div className="p-3 rounded-lg flex items-start gap-3" style={{
          backgroundColor: 'rgba(220, 38, 38, 0.15)',
          border: '1px solid rgba(220, 38, 38, 0.3)',
        }}>
          <span className="text-lg">🔓</span>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs px-2 py-0.5 rounded font-bold" style={{
                backgroundColor: '#dc2626',
                color: 'white',
              }}>
                LEAKAGE IN TOP FEATURES
              </span>
            </div>
            <div className="text-sm text-red-300 mb-2">
              Model relies heavily on {audit.concerning_leakage_features.length} suspicious feature(s) that were flagged
              as potential data leakage. Performance may not generalize to production.
            </div>
            <div className="space-y-1">
              {audit.concerning_leakage_features.map((feat, idx) => (
                <div key={idx} className="text-xs p-2 rounded flex items-center gap-2" style={{
                  backgroundColor: 'rgba(220, 38, 38, 0.1)',
                }}>
                  <span className="font-mono text-red-400 font-bold">#{feat.importance_rank || idx + 1}</span>
                  <span className="font-mono text-red-300">{feat.column}</span>
                  <span className="text-gray-400">-</span>
                  <span className="text-gray-300 flex-1">{feat.reason}</span>
                  {feat.importance !== undefined && (
                    <span className="text-xs text-gray-500 font-mono">
                      (imp: {feat.importance.toFixed(4)})
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Leakage Candidates Count Info (Prompt 6) */}
      {audit.leakage_candidates_count !== undefined && audit.leakage_candidates_count > 0 && !audit.leakage_in_important_features && (
        <div className="p-2 rounded-lg flex items-center gap-2 text-sm" style={{
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          border: '1px solid rgba(245, 158, 11, 0.2)',
        }}>
          <span>📋</span>
          <span className="text-amber-300">
            {audit.leakage_candidates_count} potential leakage feature(s) detected during data audit.
            None are among the top important features.
          </span>
        </div>
      )}

      {/* Risk-Adjusted Score (Prompt 5) */}
      {audit.risk_adjusted_score !== null && audit.risk_adjusted_score !== undefined && (
        <div className="p-3 rounded-lg flex items-center justify-between" style={{
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          border: '1px solid rgba(59, 130, 246, 0.2)',
        }}>
          <div className="flex items-center gap-2">
            <span className="text-lg">📊</span>
            <span className="text-sm text-gray-300">Risk-Adjusted Score:</span>
          </div>
          <span className="font-mono text-lg font-bold" style={{
            color: audit.risk_adjusted_score >= 0.7 ? '#22c55e' :
                   audit.risk_adjusted_score >= 0.5 ? '#f59e0b' : '#ef4444',
          }}>
            {audit.risk_adjusted_score.toFixed(4)}
          </span>
        </div>
      )}

      {/* Additional Warnings (Prompt 4) */}
      {audit.warnings && audit.warnings.length > 0 && (
        <div className="space-y-2">
          {audit.warnings.map((warning, idx) => (
            <div key={idx} className="p-2 rounded text-sm text-amber-300" style={{
              backgroundColor: 'rgba(245, 158, 11, 0.1)',
              borderLeft: '3px solid #f59e0b',
            }}>
              ⚠️ {warning}
            </div>
          ))}
        </div>
      )}

      {/* Summary */}
      {audit.natural_language_summary && (
        <div className="p-3 bg-gray-700/50 rounded-lg text-sm text-gray-300">
          {audit.natural_language_summary}
        </div>
      )}

      {/* Train-Val Analysis */}
      {audit.train_val_analysis && (
        <div className="space-y-2">
          <button
            onClick={() => toggleSection('train_val')}
            className="w-full flex items-center justify-between text-left text-sm font-medium text-gray-300 hover:text-gray-200"
          >
            <span>📊 Train-Validation Gap Analysis</span>
            <span className="text-gray-500">{expandedSection === 'train_val' ? '−' : '+'}</span>
          </button>
          {expandedSection === 'train_val' && (
            <div className="pl-4 space-y-2 text-sm">
              {audit.train_val_analysis.worst_gap !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Worst Gap:</span>
                  <span
                    className="font-mono"
                    style={{
                      color:
                        audit.train_val_analysis.worst_gap > 0.15
                          ? '#ef4444'
                          : audit.train_val_analysis.worst_gap > 0.08
                          ? '#f59e0b'
                          : '#22c55e',
                    }}
                  >
                    {audit.train_val_analysis.worst_gap.toFixed(4)}
                  </span>
                </div>
              )}
              {audit.train_val_analysis.avg_gap !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Average Gap:</span>
                  <span className="font-mono text-gray-300">
                    {audit.train_val_analysis.avg_gap.toFixed(4)}
                  </span>
                </div>
              )}
              {audit.train_val_analysis.interpretation && (
                <div className="text-gray-400 italic">
                  {audit.train_val_analysis.interpretation}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Suspicious Patterns */}
      {audit.suspicious_patterns.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-gray-400 uppercase tracking-wide">
            Suspicious Patterns ({audit.suspicious_patterns.length})
          </div>
          <div className="space-y-2">
            {audit.suspicious_patterns.map((pattern, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg text-sm flex items-start gap-3"
                style={{
                  backgroundColor: `${getSeverityColor(pattern.severity)}10`,
                  borderLeft: `3px solid ${getSeverityColor(pattern.severity)}`,
                }}
              >
                <span className="text-lg">{getPatternIcon(pattern.type)}</span>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className="text-xs px-2 py-0.5 rounded-full font-medium"
                      style={{
                        backgroundColor: `${getSeverityColor(pattern.severity)}20`,
                        color: getSeverityColor(pattern.severity),
                      }}
                    >
                      {pattern.severity.toUpperCase()}
                    </span>
                    <span className="text-xs text-gray-500 font-mono">
                      {pattern.type.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <div className="text-gray-300">{pattern.description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Baseline Comparison */}
      {audit.baseline_comparison && audit.baseline_comparison.baseline_type !== 'none_available' && (
        <div className="space-y-2">
          <button
            onClick={() => toggleSection('baseline')}
            className="w-full flex items-center justify-between text-left text-sm font-medium text-gray-300 hover:text-gray-200"
          >
            <span>📉 Baseline Comparison</span>
            <span className="text-gray-500">{expandedSection === 'baseline' ? '−' : '+'}</span>
          </button>
          {expandedSection === 'baseline' && (
            <div className="pl-4 space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Baseline Type:</span>
                <span className="text-gray-300">
                  {audit.baseline_comparison.baseline_type.replace(/_/g, ' ')}
                </span>
              </div>
              {audit.baseline_comparison.baseline_metric !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Baseline Metric:</span>
                  <span className="font-mono text-gray-400">
                    {audit.baseline_comparison.baseline_metric.toFixed(4)}
                  </span>
                </div>
              )}
              {audit.baseline_comparison.best_model_metric !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Best Model:</span>
                  <span className="font-mono text-blue-400">
                    {audit.baseline_comparison.best_model_metric.toFixed(4)}
                  </span>
                </div>
              )}
              {audit.baseline_comparison.relative_improvement !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-400">Improvement:</span>
                  <span
                    className="font-mono"
                    style={{
                      color:
                        audit.baseline_comparison.relative_improvement < 0.05
                          ? '#ef4444'
                          : audit.baseline_comparison.relative_improvement < 0.15
                          ? '#f59e0b'
                          : '#22c55e',
                    }}
                  >
                    {(audit.baseline_comparison.relative_improvement * 100).toFixed(1)}%
                  </span>
                </div>
              )}
              {audit.baseline_comparison.interpretation && (
                <div className="text-gray-400 italic">
                  {audit.baseline_comparison.interpretation}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* CV Analysis */}
      {audit.cv_analysis && audit.cv_analysis.fold_variance !== undefined && (
        <div className="space-y-2">
          <button
            onClick={() => toggleSection('cv')}
            className="w-full flex items-center justify-between text-left text-sm font-medium text-gray-300 hover:text-gray-200"
          >
            <span>📏 Cross-Validation Analysis</span>
            <span className="text-gray-500">{expandedSection === 'cv' ? '−' : '+'}</span>
          </button>
          {expandedSection === 'cv' && (
            <div className="pl-4 space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Fold Variance:</span>
                <span
                  className="font-mono"
                  style={{
                    color:
                      audit.cv_analysis.fold_variance! > 0.1
                        ? '#ef4444'
                        : audit.cv_analysis.fold_variance! > 0.05
                        ? '#f59e0b'
                        : '#22c55e',
                  }}
                >
                  {audit.cv_analysis.fold_variance!.toFixed(4)}
                </span>
              </div>
              {audit.cv_analysis.interpretation && (
                <div className="text-gray-400 italic">{audit.cv_analysis.interpretation}</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Recommendations */}
      {audit.recommendations.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-gray-400 uppercase tracking-wide">Recommendations</div>
          <ul className="list-disc list-inside space-y-1 text-sm text-gray-300">
            {audit.recommendations.map((rec, idx) => (
              <li key={idx}>{rec}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Show Full Thinking Link */}
      {onShowFullThinking && (
        <div className="pt-2 border-t border-gray-700">
          <button
            onClick={onShowFullThinking}
            className="text-sm text-blue-400 hover:text-blue-300 flex items-center gap-1"
          >
            <span>💭</span>
            <span>Show full agent thinking</span>
          </button>
        </div>
      )}
    </div>
  );
};

export default RobustnessAuditPanel;

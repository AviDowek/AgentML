import { useState, useRef, useEffect } from 'react';

/**
 * Dataset context for contextual explanations
 */
export interface DatasetContext {
  targetColumn?: string;
  targetMin?: number;
  targetMax?: number;
  targetMean?: number;
  targetStd?: number;
  rowCount?: number;
  featureCount?: number;
  taskType?: string;
}

/**
 * Metric metadata with explanations and score ranges
 */
interface MetricInfo {
  name: string;
  description: string;
  interpretation: string;
  goodRange: string;
  badRange: string;
  direction: 'higher' | 'lower';
  category: 'classification' | 'regression' | 'general';
}

const METRIC_DATABASE: Record<string, MetricInfo> = {
  // Classification Metrics
  roc_auc: {
    name: 'ROC AUC',
    description: 'Measures how well the model distinguishes between classes. It plots the true positive rate vs false positive rate.',
    interpretation: 'Think of it as: "How good is the model at ranking positive cases higher than negative ones?"',
    goodRange: '0.8 - 1.0 (Excellent), 0.7 - 0.8 (Good)',
    badRange: '0.5 - 0.6 (Poor, near random guessing)',
    direction: 'higher',
    category: 'classification',
  },
  accuracy: {
    name: 'Accuracy',
    description: 'The percentage of predictions that are correct. Simple but can be misleading for imbalanced data.',
    interpretation: 'If accuracy is 0.85, the model is right 85% of the time.',
    goodRange: '0.9+ (Excellent), 0.8 - 0.9 (Good)',
    badRange: 'Below 0.7 (May need improvement)',
    direction: 'higher',
    category: 'classification',
  },
  balanced_accuracy: {
    name: 'Balanced Accuracy',
    description: 'Average of accuracy for each class. Better than accuracy when classes are imbalanced.',
    interpretation: 'Gives equal importance to predicting each class correctly, even rare ones.',
    goodRange: '0.8+ (Good), 0.9+ (Excellent)',
    badRange: 'Below 0.6 (Poor)',
    direction: 'higher',
    category: 'classification',
  },
  f1: {
    name: 'F1 Score',
    description: 'The balance between precision (accuracy of positive predictions) and recall (catching all positives).',
    interpretation: 'High F1 means the model is both accurate and doesn\'t miss many positive cases.',
    goodRange: '0.8+ (Good), 0.9+ (Excellent)',
    badRange: 'Below 0.5 (Poor)',
    direction: 'higher',
    category: 'classification',
  },
  precision: {
    name: 'Precision',
    description: 'Of all the positive predictions, how many were actually positive?',
    interpretation: 'High precision means fewer false alarms. Important when false positives are costly.',
    goodRange: '0.8+ (Good)',
    badRange: 'Below 0.5 (Many false positives)',
    direction: 'higher',
    category: 'classification',
  },
  recall: {
    name: 'Recall (Sensitivity)',
    description: 'Of all the actual positives, how many did we catch?',
    interpretation: 'High recall means we catch most positive cases. Important when missing positives is costly.',
    goodRange: '0.8+ (Good)',
    badRange: 'Below 0.5 (Missing many positives)',
    direction: 'higher',
    category: 'classification',
  },
  mcc: {
    name: 'Matthews Correlation Coefficient',
    description: 'A balanced measure that works well even with imbalanced classes. Ranges from -1 to 1.',
    interpretation: '+1 is perfect, 0 is random, -1 is completely wrong. Good for imbalanced data.',
    goodRange: '0.5+ (Good), 0.7+ (Excellent)',
    badRange: 'Below 0.2 (Poor)',
    direction: 'higher',
    category: 'classification',
  },
  log_loss: {
    name: 'Log Loss',
    description: 'Measures how confident and correct the predictions are. Penalizes confident wrong predictions heavily.',
    interpretation: 'Lower is better. A model confident in wrong answers gets heavily penalized.',
    goodRange: 'Below 0.3 (Good)',
    badRange: 'Above 1.0 (Poor)',
    direction: 'lower',
    category: 'classification',
  },

  // Regression Metrics
  rmse: {
    name: 'RMSE (Root Mean Square Error)',
    description: 'Average prediction error, giving more weight to large errors. In the same units as your target.',
    interpretation: 'If RMSE is 10 and you\'re predicting prices, predictions are off by ~$10 on average.',
    goodRange: 'Depends on your data scale',
    badRange: 'Very high compared to target range',
    direction: 'lower',
    category: 'regression',
  },
  mse: {
    name: 'MSE (Mean Square Error)',
    description: 'Average of squared prediction errors. Penalizes large errors heavily.',
    interpretation: 'Lower is better. Take the square root to get RMSE for interpretable units.',
    goodRange: 'Depends on your data scale',
    badRange: 'Very high compared to target variance',
    direction: 'lower',
    category: 'regression',
  },
  mae: {
    name: 'MAE (Mean Absolute Error)',
    description: 'Average absolute prediction error. More robust to outliers than RMSE.',
    interpretation: 'If MAE is 5 and you\'re predicting prices, predictions are off by ~$5 on average.',
    goodRange: 'Depends on your data scale',
    badRange: 'Very high compared to target range',
    direction: 'lower',
    category: 'regression',
  },
  r2: {
    name: 'R² (R-Squared)',
    description: 'Proportion of variance in the target explained by the model. Ranges from -∞ to 1.',
    interpretation: 'R² of 0.8 means the model explains 80% of the variation in your target.',
    goodRange: '0.8+ (Good), 0.9+ (Excellent)',
    badRange: 'Below 0.5 (Poor), Negative (worse than average)',
    direction: 'higher',
    category: 'regression',
  },
  mape: {
    name: 'MAPE (Mean Absolute Percentage Error)',
    description: 'Average percentage error. Easy to interpret but undefined when actuals are zero.',
    interpretation: 'MAPE of 10% means predictions are off by about 10% on average.',
    goodRange: 'Below 10% (Good)',
    badRange: 'Above 30% (Poor)',
    direction: 'lower',
    category: 'regression',
  },

  // Additional Classification Metrics
  specificity: {
    name: 'Specificity',
    description: 'Of all actual negatives, how many were correctly identified?',
    interpretation: 'High specificity means fewer false positives. Important when negative predictions must be accurate.',
    goodRange: '0.8+ (Good)',
    badRange: 'Below 0.5 (Many false positives)',
    direction: 'higher',
    category: 'classification',
  },
  f1_macro: {
    name: 'F1 Macro',
    description: 'Average F1 score across all classes, treating each class equally.',
    interpretation: 'Good for imbalanced multiclass problems where all classes matter equally.',
    goodRange: '0.7+ (Good), 0.85+ (Excellent)',
    badRange: 'Below 0.5 (Poor)',
    direction: 'higher',
    category: 'classification',
  },
  f1_weighted: {
    name: 'F1 Weighted',
    description: 'Weighted average F1 score, weighted by class frequency.',
    interpretation: 'Good for imbalanced data where you care more about frequent classes.',
    goodRange: '0.7+ (Good), 0.85+ (Excellent)',
    badRange: 'Below 0.5 (Poor)',
    direction: 'higher',
    category: 'classification',
  },
  precision_macro: {
    name: 'Precision Macro',
    description: 'Average precision across all classes, treating each class equally.',
    interpretation: 'High value means fewer false positives across all classes.',
    goodRange: '0.7+ (Good)',
    badRange: 'Below 0.5 (Poor)',
    direction: 'higher',
    category: 'classification',
  },
  recall_macro: {
    name: 'Recall Macro',
    description: 'Average recall across all classes, treating each class equally.',
    interpretation: 'High value means we catch most cases in each class.',
    goodRange: '0.7+ (Good)',
    badRange: 'Below 0.5 (Poor)',
    direction: 'higher',
    category: 'classification',
  },

  // Additional Regression Metrics
  root_mean_squared_error: {
    name: 'RMSE',
    description: 'Average prediction error, giving more weight to large errors. In the same units as your target.',
    interpretation: 'If RMSE is 10 and you\'re predicting prices, predictions are off by ~$10 on average.',
    goodRange: 'Depends on your data scale',
    badRange: 'Very high compared to target range',
    direction: 'lower',
    category: 'regression',
  },
  mean_squared_error: {
    name: 'MSE',
    description: 'Average of squared prediction errors. Penalizes large errors heavily.',
    interpretation: 'Lower is better. Take the square root to get RMSE for interpretable units.',
    goodRange: 'Depends on your data scale',
    badRange: 'Very high compared to target variance',
    direction: 'lower',
    category: 'regression',
  },
  mean_absolute_error: {
    name: 'MAE',
    description: 'Average absolute prediction error. More robust to outliers than RMSE.',
    interpretation: 'If MAE is 5 and you\'re predicting prices, predictions are off by ~$5 on average.',
    goodRange: 'Depends on your data scale',
    badRange: 'Very high compared to target range',
    direction: 'lower',
    category: 'regression',
  },
  explained_variance: {
    name: 'Explained Variance',
    description: 'How much of the target\'s variance the model explains. Similar to R² but different calculation.',
    interpretation: 'A value of 0.8 means the model explains 80% of the variance.',
    goodRange: '0.8+ (Good), 0.9+ (Excellent)',
    badRange: 'Below 0.5 (Poor)',
    direction: 'higher',
    category: 'regression',
  },
  neg_mean_squared_error: {
    name: 'Negative MSE',
    description: 'Negative of MSE, used in sklearn for scoring where higher is better.',
    interpretation: 'Closer to zero is better. This is MSE multiplied by -1.',
    goodRange: 'Close to 0',
    badRange: 'Very negative (large error)',
    direction: 'higher',
    category: 'regression',
  },
  neg_root_mean_squared_error: {
    name: 'Negative RMSE',
    description: 'Negative of RMSE, used in sklearn for scoring where higher is better.',
    interpretation: 'Closer to zero is better. This is RMSE multiplied by -1.',
    goodRange: 'Close to 0',
    badRange: 'Very negative (large error)',
    direction: 'higher',
    category: 'regression',
  },
  neg_mean_absolute_error: {
    name: 'Negative MAE',
    description: 'Negative of MAE, used in sklearn for scoring where higher is better.',
    interpretation: 'Closer to zero is better. This is MAE multiplied by -1.',
    goodRange: 'Close to 0',
    badRange: 'Very negative (large error)',
    direction: 'higher',
    category: 'regression',
  },

  // General / Training metrics
  score_val: {
    name: 'Validation Score',
    description: 'The primary metric score on held-out validation data.',
    interpretation: 'This is the main score used to compare models during training.',
    goodRange: 'Depends on the metric type',
    badRange: 'Depends on the metric type',
    direction: 'higher',
    category: 'general',
  },
  training_time_seconds: {
    name: 'Training Time',
    description: 'How long the model took to train in seconds.',
    interpretation: 'Longer training often means more complex models or more data processed.',
    goodRange: 'Depends on your time budget',
    badRange: 'N/A',
    direction: 'lower',
    category: 'general',
  },
  num_models_trained: {
    name: 'Models Trained',
    description: 'The number of different model configurations that were tried.',
    interpretation: 'More models usually means more thorough search for the best solution.',
    goodRange: '10+ models tried',
    badRange: 'N/A',
    direction: 'higher',
    category: 'general',
  },
  best_score: {
    name: 'Best Score',
    description: 'The best score achieved during training on the primary metric.',
    interpretation: 'This represents the peak performance achieved by any model configuration.',
    goodRange: 'Depends on the metric type',
    badRange: 'Depends on the metric type',
    direction: 'higher',
    category: 'general',
  },
};

// Normalize metric key (handle variations)
function normalizeMetricKey(key: string): string {
  const normalized = key.toLowerCase().replace(/[-_\s]/g, '_');

  // Handle common variations
  const aliases: Record<string, string> = {
    'auc': 'roc_auc',
    'roc': 'roc_auc',
    'auroc': 'roc_auc',
    'acc': 'accuracy',
    'f1_score': 'f1',
    'r_squared': 'r2',
    'rsquared': 'r2',
    'root_mean_squared_error': 'rmse',
    'mean_squared_error': 'mse',
    'mean_absolute_error': 'mae',
    'mean_absolute_percentage_error': 'mape',
  };

  return aliases[normalized] || normalized;
}

interface MetricExplainerProps {
  metricKey: string;
  value?: number | null;
  size?: 'small' | 'medium';
  datasetContext?: DatasetContext;
}

/**
 * Generate contextual explanation based on dataset and metric value
 */
function getContextualExplanation(
  metricKey: string,
  value: number | null | undefined,
  datasetContext?: DatasetContext
): { interpretation: string; goodRange: string; badRange: string } | null {
  if (!datasetContext || value === null || value === undefined) return null;

  const { targetColumn, targetMin, targetMax, targetMean, targetStd, rowCount } = datasetContext;
  const targetRange = (targetMax !== undefined && targetMin !== undefined) ? targetMax - targetMin : null;

  // RMSE / MAE - explain in terms of target scale
  if (metricKey === 'rmse' || metricKey === 'mae') {
    const metricName = metricKey.toUpperCase();
    if (targetRange && targetMean !== undefined) {
      const errorPct = (value / targetRange * 100).toFixed(1);
      const relativeToMean = targetMean !== 0 ? (value / Math.abs(targetMean) * 100).toFixed(1) : null;

      let interpretation = `Your ${metricName} of ${value.toFixed(2)} means predictions are typically off by this amount.`;
      if (relativeToMean) {
        interpretation += ` This is ${relativeToMean}% of the average ${targetColumn || 'target'} value (${targetMean.toFixed(2)}).`;
      }
      interpretation += ` The ${targetColumn || 'target'} ranges from ${targetMin?.toFixed(2)} to ${targetMax?.toFixed(2)}.`;

      const goodThreshold = targetRange * 0.05; // 5% of range
      const badThreshold = targetRange * 0.20; // 20% of range

      return {
        interpretation,
        goodRange: `Below ${goodThreshold.toFixed(2)} (< 5% of ${targetColumn || 'target'} range)`,
        badRange: `Above ${badThreshold.toFixed(2)} (> 20% of ${targetColumn || 'target'} range)`,
      };
    }
  }

  // MSE - explain as squared error
  if (metricKey === 'mse' && targetRange) {
    const rmseEquiv = Math.sqrt(value);
    const errorPct = (rmseEquiv / targetRange * 100).toFixed(1);

    return {
      interpretation: `Your MSE of ${value.toFixed(4)} equals an RMSE of ${rmseEquiv.toFixed(2)}. This means predictions are typically off by ${rmseEquiv.toFixed(2)} units (${errorPct}% of the ${targetColumn || 'target'} range from ${targetMin?.toFixed(2)} to ${targetMax?.toFixed(2)}).`,
      goodRange: `Below ${(targetRange * 0.05) ** 2} (RMSE < 5% of range)`,
      badRange: `Above ${(targetRange * 0.20) ** 2} (RMSE > 20% of range)`,
    };
  }

  // R² - explain variance captured
  if (metricKey === 'r2' && targetStd !== undefined) {
    const varianceExplained = (value * 100).toFixed(1);
    const unexplainedStd = targetStd * Math.sqrt(1 - Math.max(0, value));

    return {
      interpretation: `Your R² of ${value.toFixed(4)} means the model explains ${varianceExplained}% of the variation in ${targetColumn || 'the target'}. The remaining prediction uncertainty is about ±${unexplainedStd.toFixed(2)} units.`,
      goodRange: '0.8+ (Explains 80%+ of variation)',
      badRange: 'Below 0.5 (Explains less than half the variation)',
    };
  }

  // MAPE - already percentage-based, but add context
  if (metricKey === 'mape' && targetMean !== undefined) {
    const avgError = targetMean * (value / 100);

    return {
      interpretation: `Your MAPE of ${value.toFixed(2)}% means predictions are off by about ${value.toFixed(1)}% on average. For a typical ${targetColumn || 'value'} of ${targetMean.toFixed(2)}, that's an error of about ±${avgError.toFixed(2)}.`,
      goodRange: 'Below 10% (Highly accurate)',
      badRange: 'Above 30% (Needs improvement)',
    };
  }

  return null;
}

export default function MetricExplainer({ metricKey, value, size = 'small', datasetContext }: MetricExplainerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const normalizedKey = normalizeMetricKey(metricKey);
  const info = METRIC_DATABASE[normalizedKey];

  // Get contextual explanation if dataset context is provided
  const contextualExplanation = getContextualExplanation(normalizedKey, value, datasetContext);

  // Close on click outside
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (
        tooltipRef.current &&
        !tooltipRef.current.contains(e.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // Calculate position when opening
  useEffect(() => {
    if (!isOpen || !buttonRef.current) return;

    const rect = buttonRef.current.getBoundingClientRect();
    const tooltipWidth = 320;
    const tooltipHeight = 280;

    let left = rect.left + rect.width / 2 - tooltipWidth / 2;
    let top = rect.bottom + 8;

    // Adjust if going off screen
    if (left < 10) left = 10;
    if (left + tooltipWidth > window.innerWidth - 10) {
      left = window.innerWidth - tooltipWidth - 10;
    }
    if (top + tooltipHeight > window.innerHeight - 10) {
      top = rect.top - tooltipHeight - 8;
    }

    setPosition({ top, left });
  }, [isOpen]);

  // Create a fallback info for unknown metrics
  const fallbackInfo: MetricInfo = {
    name: metricKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    description: `This metric measures ${metricKey.replace(/_/g, ' ')} performance.`,
    interpretation: 'Refer to your model training documentation for specific interpretation guidance.',
    goodRange: 'Depends on your specific use case',
    badRange: 'Depends on your specific use case',
    direction: metricKey.toLowerCase().includes('error') || metricKey.toLowerCase().includes('loss') ? 'lower' : 'higher',
    category: metricKey.toLowerCase().includes('accuracy') || metricKey.toLowerCase().includes('auc') ||
              metricKey.toLowerCase().includes('f1') || metricKey.toLowerCase().includes('precision') ||
              metricKey.toLowerCase().includes('recall') ? 'classification' :
              metricKey.toLowerCase().includes('rmse') || metricKey.toLowerCase().includes('mae') ||
              metricKey.toLowerCase().includes('mse') || metricKey.toLowerCase().includes('r2') ? 'regression' : 'general',
  };

  const displayInfo = info || fallbackInfo;

  // Determine score quality indicator
  const getScoreQuality = (): { label: string; color: string } | null => {
    if (value === null || value === undefined) return null;

    const v = typeof value === 'number' ? value : parseFloat(String(value));
    if (isNaN(v)) return null;

    // Classification metrics with standard ranges
    if (normalizedKey === 'roc_auc') {
      if (v >= 0.9) return { label: 'Excellent', color: '#22c55e' };
      if (v >= 0.8) return { label: 'Good', color: '#84cc16' };
      if (v >= 0.7) return { label: 'Fair', color: '#eab308' };
      if (v >= 0.6) return { label: 'Poor', color: '#f97316' };
      return { label: 'Very Poor', color: '#ef4444' };
    }

    if (normalizedKey === 'accuracy' || normalizedKey === 'balanced_accuracy') {
      if (v >= 0.95) return { label: 'Excellent', color: '#22c55e' };
      if (v >= 0.85) return { label: 'Good', color: '#84cc16' };
      if (v >= 0.75) return { label: 'Fair', color: '#eab308' };
      if (v >= 0.65) return { label: 'Poor', color: '#f97316' };
      return { label: 'Very Poor', color: '#ef4444' };
    }

    if (normalizedKey === 'f1' || normalizedKey === 'precision' || normalizedKey === 'recall') {
      if (v >= 0.9) return { label: 'Excellent', color: '#22c55e' };
      if (v >= 0.7) return { label: 'Good', color: '#84cc16' };
      if (v >= 0.5) return { label: 'Fair', color: '#eab308' };
      return { label: 'Poor', color: '#f97316' };
    }

    if (normalizedKey === 'mcc') {
      if (v >= 0.7) return { label: 'Excellent', color: '#22c55e' };
      if (v >= 0.5) return { label: 'Good', color: '#84cc16' };
      if (v >= 0.3) return { label: 'Fair', color: '#eab308' };
      if (v >= 0.1) return { label: 'Poor', color: '#f97316' };
      return { label: 'Very Poor', color: '#ef4444' };
    }

    if (normalizedKey === 'r2') {
      if (v >= 0.9) return { label: 'Excellent', color: '#22c55e' };
      if (v >= 0.7) return { label: 'Good', color: '#84cc16' };
      if (v >= 0.5) return { label: 'Fair', color: '#eab308' };
      if (v >= 0) return { label: 'Poor', color: '#f97316' };
      return { label: 'Very Poor', color: '#ef4444' };
    }

    return null;
  };

  const quality = getScoreQuality();
  const iconSize = size === 'small' ? '14px' : '16px';

  return (
    <>
      <button
        ref={buttonRef}
        onClick={() => setIsOpen(!isOpen)}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: iconSize,
          height: iconSize,
          borderRadius: '50%',
          border: '1px solid #94a3b8',
          background: isOpen ? '#3b82f6' : '#f1f5f9',
          color: isOpen ? '#fff' : '#64748b',
          fontSize: size === 'small' ? '10px' : '12px',
          fontWeight: 600,
          cursor: 'pointer',
          marginLeft: '4px',
          padding: 0,
          lineHeight: 1,
          transition: 'all 0.2s',
        }}
        title={`Learn about ${displayInfo.name}`}
        aria-label={`Information about ${displayInfo.name}`}
      >
        i
      </button>

      {isOpen && position && (
        <div
          ref={tooltipRef}
          style={{
            position: 'fixed',
            top: position.top,
            left: position.left,
            width: '320px',
            background: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: '12px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.15)',
            zIndex: 10000,
            padding: '16px',
            animation: 'fadeIn 0.15s ease-out',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
            <div>
              <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#1e293b' }}>
                {displayInfo.name}
              </h4>
              <span style={{
                fontSize: '0.6875rem',
                padding: '2px 8px',
                background: displayInfo.category === 'classification' ? '#dbeafe' : displayInfo.category === 'regression' ? '#fef3c7' : '#f1f5f9',
                color: displayInfo.category === 'classification' ? '#1d4ed8' : displayInfo.category === 'regression' ? '#b45309' : '#475569',
                borderRadius: '9999px',
                textTransform: 'capitalize',
              }}>
                {displayInfo.category}
              </span>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '18px',
                color: '#94a3b8',
                cursor: 'pointer',
                padding: '0 4px',
              }}
            >
              ×
            </button>
          </div>

          {/* Score quality badge */}
          {quality && value !== undefined && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '12px',
              padding: '8px 12px',
              background: `${quality.color}15`,
              borderRadius: '8px',
              border: `1px solid ${quality.color}40`,
            }}>
              <span style={{ fontSize: '1.25rem' }}>
                {quality.label === 'Excellent' ? '🎯' : quality.label === 'Good' ? '✅' : quality.label === 'Fair' ? '📊' : '⚠️'}
              </span>
              <div>
                <div style={{ fontWeight: 600, color: quality.color, fontSize: '0.875rem' }}>
                  {quality.label}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                  Your score: {typeof value === 'number' ? value.toFixed(4) : value}
                </div>
              </div>
            </div>
          )}

          <div style={{ marginBottom: '12px' }}>
            <p style={{ margin: 0, fontSize: '0.875rem', color: '#475569', lineHeight: 1.5 }}>
              {displayInfo.description}
            </p>
          </div>

          <div style={{
            padding: '10px',
            background: contextualExplanation ? '#eff6ff' : '#f8fafc',
            borderRadius: '8px',
            marginBottom: '12px',
            border: contextualExplanation ? '1px solid #bfdbfe' : 'none',
          }}>
            <div style={{ fontSize: '0.75rem', color: contextualExplanation ? '#1d4ed8' : '#64748b', marginBottom: '4px', fontWeight: 500 }}>
              {contextualExplanation ? '🎯 For Your Data' : '💡 In Plain English'}
            </div>
            <p style={{ margin: 0, fontSize: '0.8125rem', color: '#334155', fontStyle: contextualExplanation ? 'normal' : 'italic' }}>
              {contextualExplanation?.interpretation || displayInfo.interpretation}
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <div style={{
              padding: '8px',
              background: '#f0fdf4',
              borderRadius: '6px',
              border: '1px solid #bbf7d0',
            }}>
              <div style={{ fontSize: '0.6875rem', color: '#166534', fontWeight: 600, marginBottom: '2px' }}>
                ✅ Good Score
              </div>
              <div style={{ fontSize: '0.75rem', color: '#15803d' }}>
                {contextualExplanation?.goodRange || displayInfo.goodRange}
              </div>
            </div>
            <div style={{
              padding: '8px',
              background: '#fef2f2',
              borderRadius: '6px',
              border: '1px solid #fecaca',
            }}>
              <div style={{ fontSize: '0.6875rem', color: '#991b1b', fontWeight: 600, marginBottom: '2px' }}>
                ⚠️ Needs Work
              </div>
              <div style={{ fontSize: '0.75rem', color: '#b91c1c' }}>
                {contextualExplanation?.badRange || displayInfo.badRange}
              </div>
            </div>
          </div>

          <div style={{
            marginTop: '12px',
            paddingTop: '12px',
            borderTop: '1px solid #e2e8f0',
            fontSize: '0.6875rem',
            color: '#94a3b8',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
          }}>
            <span>{displayInfo.direction === 'higher' ? '📈' : '📉'}</span>
            {displayInfo.direction === 'higher' ? 'Higher is better' : 'Lower is better'}
          </div>
        </div>
      )}

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  );
}

/**
 * Wrapper component that displays a metric value with its explainer
 */
interface MetricWithExplainerProps {
  metricKey: string;
  value: number | null | undefined;
  format?: (v: number) => string;
}

export function MetricWithExplainer({ metricKey, value, format }: MetricWithExplainerProps) {
  const displayValue = value !== null && value !== undefined
    ? (format ? format(value) : (typeof value === 'number' ? value.toFixed(4) : String(value)))
    : 'N/A';

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
      {displayValue}
      <MetricExplainer metricKey={metricKey} value={value} />
    </span>
  );
}


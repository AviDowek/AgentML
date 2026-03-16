import type { BaselineMetrics } from '../types/api';

interface BaselineComparisonProps {
  baselineMetrics: BaselineMetrics | null;
  modelMetrics: Record<string, number> | null;
  taskType?: 'binary' | 'multiclass' | 'regression';
}

export default function BaselineComparison({
  baselineMetrics,
  modelMetrics,
  taskType = 'binary',
}: BaselineComparisonProps) {
  if (!baselineMetrics) {
    return null;
  }

  const isClassification = taskType === 'binary' || taskType === 'multiclass';

  // Get the appropriate baseline based on task type
  const majorityBaseline = baselineMetrics.majority_class;
  const meanBaseline = baselineMetrics.mean_predictor;
  const simpleBaseline = isClassification
    ? baselineMetrics.simple_logistic
    : baselineMetrics.simple_ridge;
  const labelShuffle = baselineMetrics.label_shuffle;

  // Get model metrics - try different keys
  // Error metrics may be negative (sklearn neg_ convention), so take absolute value
  const modelAccuracy = modelMetrics?.accuracy;
  const modelAuc = modelMetrics?.roc_auc;
  const rawRmse = modelMetrics?.rmse ?? modelMetrics?.root_mean_squared_error ?? modelMetrics?.neg_root_mean_squared_error;
  const modelRmse = rawRmse !== undefined ? Math.abs(rawRmse) : undefined;
  const modelR2 = modelMetrics?.r2;
  const rawMae = modelMetrics?.mae ?? modelMetrics?.mean_absolute_error ?? modelMetrics?.neg_mean_absolute_error;
  const modelMae = rawMae !== undefined ? Math.abs(rawMae) : undefined;

  // Calculate improvement over baselines
  const calcImprovement = (
    modelVal: number | undefined,
    baselineVal: number | undefined,
    higherIsBetter = true
  ): { value: number; isPositive: boolean } | null => {
    if (modelVal === undefined || baselineVal === undefined) return null;
    const diff = modelVal - baselineVal;
    const pct = baselineVal !== 0 ? (diff / Math.abs(baselineVal)) * 100 : 0;
    return {
      value: Math.abs(pct),
      isPositive: higherIsBetter ? diff > 0 : diff < 0,
    };
  };

  // Format value for display
  const formatValue = (val: number | undefined, decimals = 4): string => {
    if (val === undefined || val === null) return '-';
    return val.toFixed(decimals);
  };

  return (
    <div className="baseline-comparison">
      <h3>Baselines & Sanity Checks</h3>

      {/* Model vs Baselines Comparison */}
      <div className="baseline-section">
        <h4>Your Model vs Baselines</h4>
        <table className="baseline-table">
          <thead>
            <tr>
              <th>Model</th>
              {isClassification ? (
                <>
                  <th>Accuracy</th>
                  {taskType === 'binary' && <th>ROC AUC</th>}
                </>
              ) : (
                <>
                  <th>RMSE</th>
                  <th>MAE</th>
                  <th>R2</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            <tr className="model-row best-row">
              <td>Your Model (AutoML)</td>
              {isClassification ? (
                <>
                  <td>{formatValue(modelAccuracy)}</td>
                  {taskType === 'binary' && <td>{formatValue(modelAuc)}</td>}
                </>
              ) : (
                <>
                  <td>{formatValue(modelRmse)}</td>
                  <td>{formatValue(modelMae)}</td>
                  <td>{formatValue(modelR2)}</td>
                </>
              )}
            </tr>
            {isClassification && majorityBaseline && (
              <tr className="baseline-row">
                <td>
                  <span className="baseline-label">Majority Class</span>
                  <span className="baseline-hint">(always predicts most common)</span>
                </td>
                <td>{formatValue(majorityBaseline.accuracy)}</td>
                {taskType === 'binary' && <td>{formatValue(majorityBaseline.roc_auc)}</td>}
              </tr>
            )}
            {!isClassification && meanBaseline && (
              <tr className="baseline-row">
                <td>
                  <span className="baseline-label">Mean Predictor</span>
                  <span className="baseline-hint">(always predicts mean)</span>
                </td>
                <td>{formatValue(meanBaseline.rmse)}</td>
                <td>{formatValue(meanBaseline.mae)}</td>
                <td>{formatValue(meanBaseline.r2)}</td>
              </tr>
            )}
            {isClassification && simpleBaseline && 'accuracy' in simpleBaseline && (
              <tr className="baseline-row">
                <td>
                  <span className="baseline-label">Simple Logistic</span>
                  <span className="baseline-hint">(L2-regularized)</span>
                </td>
                <td>{formatValue(simpleBaseline.accuracy)}</td>
                {taskType === 'binary' && <td>{formatValue(simpleBaseline.roc_auc)}</td>}
              </tr>
            )}
            {!isClassification && simpleBaseline && 'rmse' in simpleBaseline && (
              <tr className="baseline-row">
                <td>
                  <span className="baseline-label">Simple Ridge</span>
                  <span className="baseline-hint">(L2-regularized)</span>
                </td>
                <td>{formatValue(simpleBaseline.rmse)}</td>
                <td>{formatValue(simpleBaseline.mae)}</td>
                <td>{formatValue(simpleBaseline.r2)}</td>
              </tr>
            )}
          </tbody>
        </table>

        {/* Improvement indicator */}
        {isClassification && majorityBaseline && (
          <div className="improvement-summary">
            {(() => {
              const improvement = calcImprovement(modelAccuracy, majorityBaseline.accuracy, true);
              if (!improvement) return null;
              return (
                <span className={improvement.isPositive ? 'positive' : 'negative'}>
                  {improvement.isPositive ? '+' : '-'}
                  {improvement.value.toFixed(1)}% vs majority class baseline
                </span>
              );
            })()}
          </div>
        )}
        {!isClassification && meanBaseline && (
          <div className="improvement-summary">
            {(() => {
              const improvement = calcImprovement(modelRmse, meanBaseline.rmse, false);
              if (!improvement) return null;
              return (
                <span className={improvement.isPositive ? 'positive' : 'negative'}>
                  {improvement.isPositive ? '' : '+'}
                  {improvement.value.toFixed(1)}% RMSE vs mean predictor
                  {improvement.isPositive ? ' (better)' : ' (worse)'}
                </span>
              );
            })()}
          </div>
        )}
      </div>

      {/* Label-Shuffle Sanity Check */}
      {labelShuffle && (
        <div className="baseline-section sanity-check">
          <h4>Sanity Check: Label-Shuffle Test</h4>
          <p className="sanity-description">
            Training on randomly shuffled labels should produce near-random performance.
            If not, your features may be leaking information about the target.
          </p>

          {labelShuffle.error ? (
            <div className="sanity-error">
              <span className="error-icon">Error</span>
              <span>{labelShuffle.error}</span>
            </div>
          ) : (
            <>
              <div className="sanity-results">
                {labelShuffle.shuffled_accuracy !== undefined && (
                  <div className="sanity-metric">
                    <span className="metric-label">Shuffled Accuracy:</span>
                    <span className="metric-value">{formatValue(labelShuffle.shuffled_accuracy)}</span>
                    {labelShuffle.expected_random_accuracy !== undefined && (
                      <span className="metric-expected">
                        (expected: ~{formatValue(labelShuffle.expected_random_accuracy, 2)})
                      </span>
                    )}
                  </div>
                )}
                {labelShuffle.shuffled_roc_auc !== undefined && (
                  <div className="sanity-metric">
                    <span className="metric-label">Shuffled ROC AUC:</span>
                    <span className="metric-value">{formatValue(labelShuffle.shuffled_roc_auc)}</span>
                    <span className="metric-expected">(expected: ~0.50)</span>
                  </div>
                )}
                {labelShuffle.shuffled_r2 !== undefined && (
                  <div className="sanity-metric">
                    <span className="metric-label">Shuffled R2:</span>
                    <span className="metric-value">{formatValue(labelShuffle.shuffled_r2)}</span>
                    <span className="metric-expected">(expected: ~0 or negative)</span>
                  </div>
                )}
              </div>

              {labelShuffle.leakage_detected === true && (
                <div className="leakage-warning">
                  <span className="warning-icon">Warning</span>
                  <span className="warning-text">
                    {labelShuffle.warning ||
                      'Potential data leakage detected! Model performs better than expected on shuffled labels.'}
                  </span>
                </div>
              )}

              {labelShuffle.leakage_detected === false && (
                <div className="sanity-pass">
                  <span className="pass-icon">Pass</span>
                  <span className="pass-text">No data leakage detected.</span>
                </div>
              )}
            </>
          )}
        </div>
      )}

      <style>{`
        .baseline-comparison {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 16px;
          margin-top: 16px;
        }

        .baseline-comparison h3 {
          margin: 0 0 16px 0;
          color: #e0e0e0;
          font-size: 16px;
          font-weight: 600;
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .baseline-comparison h3::before {
          content: "📊";
        }

        .baseline-section {
          margin-bottom: 20px;
        }

        .baseline-section:last-child {
          margin-bottom: 0;
        }

        .baseline-section h4 {
          margin: 0 0 12px 0;
          color: #b0b0b0;
          font-size: 14px;
          font-weight: 500;
        }

        .baseline-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
        }

        .baseline-table th,
        .baseline-table td {
          padding: 8px 12px;
          text-align: left;
          border-bottom: 1px solid #2a2a2a;
        }

        .baseline-table th {
          color: #888;
          font-weight: 500;
          font-size: 12px;
          text-transform: uppercase;
        }

        .baseline-table td {
          color: #d0d0d0;
        }

        .model-row {
          background: rgba(34, 197, 94, 0.1);
        }

        .model-row td:first-child {
          font-weight: 600;
          color: #22c55e;
        }

        .baseline-row td:first-child {
          color: #888;
        }

        .baseline-label {
          display: block;
          color: #a0a0a0;
        }

        .baseline-hint {
          display: block;
          font-size: 11px;
          color: #666;
          font-style: italic;
        }

        .improvement-summary {
          margin-top: 10px;
          font-size: 13px;
          padding: 8px 12px;
          background: #222;
          border-radius: 4px;
        }

        .improvement-summary .positive {
          color: #22c55e;
        }

        .improvement-summary .negative {
          color: #ef4444;
        }

        .sanity-check {
          background: #1e1e1e;
          border: 1px solid #333;
          border-radius: 6px;
          padding: 12px;
        }

        .sanity-description {
          color: #888;
          font-size: 12px;
          margin: 0 0 12px 0;
          line-height: 1.5;
        }

        .sanity-results {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .sanity-metric {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 13px;
        }

        .sanity-metric .metric-label {
          color: #888;
          min-width: 140px;
        }

        .sanity-metric .metric-value {
          color: #d0d0d0;
          font-family: monospace;
        }

        .sanity-metric .metric-expected {
          color: #666;
          font-size: 12px;
        }

        .leakage-warning {
          margin-top: 12px;
          padding: 10px 12px;
          background: rgba(239, 68, 68, 0.15);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 6px;
          display: flex;
          align-items: flex-start;
          gap: 10px;
        }

        .warning-icon {
          background: #ef4444;
          color: white;
          font-size: 11px;
          font-weight: 600;
          padding: 2px 8px;
          border-radius: 4px;
          flex-shrink: 0;
        }

        .warning-text {
          color: #fca5a5;
          font-size: 13px;
          line-height: 1.4;
        }

        .sanity-pass {
          margin-top: 12px;
          padding: 10px 12px;
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.2);
          border-radius: 6px;
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .pass-icon {
          background: #22c55e;
          color: white;
          font-size: 11px;
          font-weight: 600;
          padding: 2px 8px;
          border-radius: 4px;
        }

        .pass-text {
          color: #86efac;
          font-size: 13px;
        }

        .sanity-error {
          margin-top: 12px;
          padding: 10px 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.2);
          border-radius: 6px;
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .error-icon {
          background: #dc2626;
          color: white;
          font-size: 11px;
          font-weight: 600;
          padding: 2px 8px;
          border-radius: 4px;
        }
      `}</style>
    </div>
  );
}

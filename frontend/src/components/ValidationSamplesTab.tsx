import { useState, useEffect, useCallback } from 'react';
import type {
  ValidationSample,
  ValidationSampleSort,
  WhatIfResponse,
  ServingFeature,
} from '../types/api';
import {
  listValidationSamples,
  runWhatIfPrediction,
  ApiException,
} from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

interface ValidationSamplesTabProps {
  modelId: string;
  servingConfig?: {
    features?: ServingFeature[];
    target_column?: string;
    task_type?: string;
  } | null;
}

export default function ValidationSamplesTab({
  modelId,
  servingConfig,
}: ValidationSamplesTabProps) {
  const [samples, setSamples] = useState<ValidationSample[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Pagination
  const [offset, setOffset] = useState(0);
  const [limit] = useState(50);

  // Sorting
  const [sort, setSort] = useState<ValidationSampleSort>('error_desc');

  // Selected sample for detail view
  const [selectedSample, setSelectedSample] = useState<ValidationSample | null>(null);

  // What-if state
  const [modifiedFeatures, setModifiedFeatures] = useState<Record<string, unknown>>({});
  const [whatIfResult, setWhatIfResult] = useState<WhatIfResponse | null>(null);
  const [isRunningWhatIf, setIsRunningWhatIf] = useState(false);
  const [whatIfError, setWhatIfError] = useState<string | null>(null);

  // Get feature names from serving config
  const featureNames = servingConfig?.features?.map((f) => f.name) || [];
  const displayFeatures = featureNames.slice(0, 5); // Show first 5 features in table

  const fetchSamples = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await listValidationSamples(modelId, { limit, offset, sort });
      setSamples(data.samples);
      setTotal(data.total);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load validation samples');
      }
    } finally {
      setIsLoading(false);
    }
  }, [modelId, limit, offset, sort]);

  useEffect(() => {
    fetchSamples();
  }, [fetchSamples]);

  const handleSortChange = (newSort: ValidationSampleSort) => {
    setSort(newSort);
    setOffset(0); // Reset to first page when sorting changes
  };

  const handlePrevPage = () => {
    setOffset(Math.max(0, offset - limit));
  };

  const handleNextPage = () => {
    if (offset + limit < total) {
      setOffset(offset + limit);
    }
  };

  const handleSelectSample = (sample: ValidationSample) => {
    setSelectedSample(sample);
    setModifiedFeatures({});
    setWhatIfResult(null);
    setWhatIfError(null);
  };

  const handleCloseDrawer = () => {
    setSelectedSample(null);
    setModifiedFeatures({});
    setWhatIfResult(null);
    setWhatIfError(null);
  };

  const handleFeatureChange = (featureName: string, value: string) => {
    // Try to parse as number if it looks like one
    let parsedValue: unknown = value;
    if (value !== '' && !isNaN(Number(value))) {
      parsedValue = Number(value);
    }

    setModifiedFeatures((prev) => ({
      ...prev,
      [featureName]: parsedValue,
    }));
  };

  const handleRunWhatIf = async () => {
    if (!selectedSample || Object.keys(modifiedFeatures).length === 0) return;

    setIsRunningWhatIf(true);
    setWhatIfError(null);

    try {
      const result = await runWhatIfPrediction(modelId, {
        sample_id: selectedSample.id,
        modified_features: modifiedFeatures,
      });
      setWhatIfResult(result);
    } catch (err) {
      if (err instanceof ApiException) {
        setWhatIfError(err.detail);
      } else {
        setWhatIfError('Failed to run what-if prediction');
      }
    } finally {
      setIsRunningWhatIf(false);
    }
  };

  const formatValue = (value: unknown): string => {
    if (value === null || value === undefined) return '-';
    if (typeof value === 'number') {
      return Number.isInteger(value) ? value.toString() : value.toFixed(4);
    }
    return String(value);
  };

  const formatError = (error: number | null): string => {
    if (error === null) return '-';
    return error.toFixed(4);
  };

  if (isLoading && samples.length === 0) {
    return <LoadingSpinner message="Loading validation samples..." />;
  }

  if (error && samples.length === 0) {
    return <ErrorMessage message={error} onRetry={fetchSamples} />;
  }

  if (total === 0) {
    return (
      <div className="validation-samples-empty">
        <p className="empty-text">
          No validation samples available for this model.
        </p>
        <p style={{ color: '#6b7280', fontSize: '0.875rem' }}>
          Validation samples are captured during model training. Re-train the model
          to generate validation predictions.
        </p>
      </div>
    );
  }

  return (
    <div className="validation-samples-tab">
      {/* Controls */}
      <div className="validation-samples-controls">
        <div className="sort-controls">
          <label>Sort by:</label>
          <select
            value={sort}
            onChange={(e) => handleSortChange(e.target.value as ValidationSampleSort)}
            className="sort-select"
          >
            <option value="error_desc">Highest Error First</option>
            <option value="error_asc">Lowest Error First</option>
            <option value="row_index">Original Order</option>
            <option value="random">Random</option>
          </select>
        </div>
        <div className="pagination-info">
          Showing {offset + 1}-{Math.min(offset + limit, total)} of {total} samples
        </div>
      </div>

      {/* Table */}
      <div className="validation-samples-table-container">
        <table className="validation-samples-table">
          <thead>
            <tr>
              <th>Row</th>
              {displayFeatures.map((name) => (
                <th key={name}>{name}</th>
              ))}
              {displayFeatures.length < featureNames.length && (
                <th>...</th>
              )}
              <th>Target</th>
              <th>Predicted</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody>
            {samples.map((sample) => (
              <tr
                key={sample.id}
                onClick={() => handleSelectSample(sample)}
                className={selectedSample?.id === sample.id ? 'selected' : ''}
              >
                <td>{sample.row_index}</td>
                {displayFeatures.map((name) => (
                  <td key={name}>{formatValue(sample.features[name])}</td>
                ))}
                {displayFeatures.length < featureNames.length && (
                  <td className="more-features">+{featureNames.length - displayFeatures.length}</td>
                )}
                <td>{sample.target_value}</td>
                <td>{sample.predicted_value}</td>
                <td className={`error-cell ${sample.absolute_error && sample.absolute_error > 1 ? 'high-error' : ''}`}>
                  {formatError(sample.absolute_error)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="validation-samples-pagination">
        <button
          onClick={handlePrevPage}
          disabled={offset === 0}
          className="btn btn-secondary"
        >
          Previous
        </button>
        <span className="page-info">
          Page {Math.floor(offset / limit) + 1} of {Math.ceil(total / limit)}
        </span>
        <button
          onClick={handleNextPage}
          disabled={offset + limit >= total}
          className="btn btn-secondary"
        >
          Next
        </button>
      </div>

      {/* Sample Detail Drawer */}
      {selectedSample && (
        <div className="sample-drawer-backdrop" onClick={handleCloseDrawer}>
          <div className="sample-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="sample-drawer-header">
              <h3>Validation Sample Details</h3>
              <button className="close-btn" onClick={handleCloseDrawer}>
                &times;
              </button>
            </div>

            <div className="sample-drawer-content">
              {/* Summary */}
              <div className="sample-summary">
                <div className="summary-item">
                  <span className="label">Row Index:</span>
                  <span className="value">{selectedSample.row_index}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Target Value:</span>
                  <span className="value">{selectedSample.target_value}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Predicted Value:</span>
                  <span className="value">{selectedSample.predicted_value}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Error:</span>
                  <span className={`value ${selectedSample.absolute_error && selectedSample.absolute_error > 1 ? 'high-error' : ''}`}>
                    {formatError(selectedSample.error_value)} (abs: {formatError(selectedSample.absolute_error)})
                  </span>
                </div>
              </div>

              {/* Probabilities (for classification) */}
              {selectedSample.prediction_probabilities && (
                <div className="sample-section">
                  <h4>Prediction Probabilities</h4>
                  <div className="probabilities-grid">
                    {Object.entries(selectedSample.prediction_probabilities).map(([cls, prob]) => (
                      <div key={cls} className="probability-item">
                        <span className="class-name">{cls}</span>
                        <div className="probability-bar">
                          <div
                            className="probability-fill"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* All Features */}
              <div className="sample-section">
                <h4>All Features</h4>
                <div className="features-list">
                  {Object.entries(selectedSample.features).map(([name, value]) => (
                    <div key={name} className="feature-item">
                      <span className="feature-name">{name}</span>
                      <span className="feature-value">{formatValue(value)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* What-If Section */}
              <div className="sample-section what-if-section">
                <h4>What-If Testing</h4>
                <p className="what-if-description">
                  Modify feature values to see how the prediction changes.
                </p>

                <div className="what-if-inputs">
                  {featureNames.map((name) => {
                    const originalValue = selectedSample.features[name];
                    const isModified = name in modifiedFeatures;
                    const currentValue = isModified
                      ? String(modifiedFeatures[name])
                      : String(originalValue ?? '');

                    return (
                      <div key={name} className={`what-if-input ${isModified ? 'modified' : ''}`}>
                        <label>{name}</label>
                        <input
                          type="text"
                          value={currentValue}
                          onChange={(e) => handleFeatureChange(name, e.target.value)}
                          placeholder={String(originalValue ?? '')}
                        />
                        {isModified && (
                          <span className="original-value">
                            Original: {formatValue(originalValue)}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>

                <button
                  className="btn btn-primary what-if-btn"
                  onClick={handleRunWhatIf}
                  disabled={isRunningWhatIf || Object.keys(modifiedFeatures).length === 0}
                >
                  {isRunningWhatIf ? 'Running...' : 'Recompute Prediction'}
                </button>

                {whatIfError && (
                  <div className="what-if-error">{whatIfError}</div>
                )}

                {whatIfResult && (
                  <div className="what-if-result">
                    <h5>What-If Result</h5>
                    <div className="what-if-comparison">
                      <div className="comparison-item">
                        <span className="label">Original Prediction:</span>
                        <span className="value">{formatValue(whatIfResult.original_prediction)}</span>
                      </div>
                      <div className="comparison-item">
                        <span className="label">New Prediction:</span>
                        <span className="value new-prediction">{formatValue(whatIfResult.modified_prediction)}</span>
                      </div>
                      {whatIfResult.prediction_delta !== null && (
                        <div className="comparison-item">
                          <span className="label">Change (Delta):</span>
                          <span className={`value delta ${whatIfResult.prediction_delta > 0 ? 'positive' : 'negative'}`}>
                            {whatIfResult.prediction_delta > 0 ? '+' : ''}
                            {whatIfResult.prediction_delta.toFixed(4)}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Modified probabilities for classification */}
                    {whatIfResult.modified_probabilities && (
                      <div className="what-if-probabilities">
                        <h6>New Probabilities</h6>
                        <div className="probabilities-grid">
                          {Object.entries(whatIfResult.modified_probabilities).map(([cls, prob]) => (
                            <div key={cls} className="probability-item">
                              <span className="class-name">{cls}</span>
                              <div className="probability-bar">
                                <div
                                  className="probability-fill"
                                  style={{ width: `${prob * 100}%` }}
                                />
                              </div>
                              <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

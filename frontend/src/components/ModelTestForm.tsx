import { useState, useEffect } from 'react';
import type { FeatureStatistics, PredictionResponse, ModelTestingDataResponse, RawPredictionResponse, FeaturePipelineInfo, HoldoutSet, HoldoutRow } from '../types/api';
import { predictWithModel, getModelTestingData, getRandomSample, getModelPipeline, getModelExportUrl, predictWithRawData, ApiException, getHoldoutSet, getHoldoutRow } from '../services/api';

interface ModelTestFormProps {
  modelId: string;
  projectId?: string;
  servingConfig: {
    features?: { name: string; type: string }[];
    target_column?: string;
    task_type?: string;
  } | null;
}

export default function ModelTestForm({ modelId, projectId, servingConfig }: ModelTestFormProps) {
  const [features, setFeatures] = useState<Record<string, unknown>>({});
  const [testingData, setTestingData] = useState<ModelTestingDataResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingTestingData, setIsLoadingTestingData] = useState(true);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAllFeatures, setShowAllFeatures] = useState(false);

  // Raw data mode state
  const [inputMode, setInputMode] = useState<'features' | 'raw' | 'holdout'>('features');
  const [pipelineInfo, setPipelineInfo] = useState<FeaturePipelineInfo | null>(null);
  const [rawPrediction, setRawPrediction] = useState<RawPredictionResponse | null>(null);

  // Raw data input - file or JSON
  const [rawDataInput, setRawDataInput] = useState<string>('');
  const [parsedRecords, setParsedRecords] = useState<Record<string, unknown>[]>([]);
  const [uploadedFileName, setUploadedFileName] = useState<string>('');

  // Holdout validation state
  const [holdoutSet, setHoldoutSet] = useState<HoldoutSet | null>(null);
  const [holdoutRow, setHoldoutRow] = useState<HoldoutRow | null>(null);
  const [holdoutRowIndex, setHoldoutRowIndex] = useState(0);
  const [isLoadingHoldout, setIsLoadingHoldout] = useState(false);
  const [holdoutPrediction, setHoldoutPrediction] = useState<PredictionResponse | null>(null);
  const [holdoutValidationResult, setHoldoutValidationResult] = useState<'correct' | 'incorrect' | null>(null);
  const [holdoutStats, setHoldoutStats] = useState({ correct: 0, incorrect: 0, tested: 0 });

  // Load testing data and pipeline info on mount
  useEffect(() => {
    loadTestingData();
    loadPipelineInfo();
  }, [modelId]);

  // Load holdout set info if projectId is provided
  useEffect(() => {
    if (projectId) {
      loadHoldoutSet();
    }
  }, [projectId]);

  const loadHoldoutSet = async () => {
    if (!projectId) return;
    try {
      const holdout = await getHoldoutSet(projectId);
      setHoldoutSet(holdout);
      // If holdout exists, load the first row
      if (holdout && holdout.holdout_row_count > 0) {
        loadHoldoutRow(0);
      }
    } catch (err) {
      // No holdout set available - that's ok, just don't show the tab
      console.log('No holdout set available for this project');
    }
  };

  const loadHoldoutRow = async (index: number) => {
    if (!projectId) return;
    setIsLoadingHoldout(true);
    setHoldoutPrediction(null);
    setHoldoutValidationResult(null);
    try {
      const row = await getHoldoutRow(projectId, index);
      setHoldoutRow(row);
      setHoldoutRowIndex(index);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load holdout row');
      }
    } finally {
      setIsLoadingHoldout(false);
    }
  };

  const handleHoldoutPredict = async () => {
    if (!holdoutRow) return;
    setIsLoading(true);
    setError(null);
    setHoldoutValidationResult(null);

    try {
      const result = await predictWithModel(modelId, holdoutRow.data);
      setHoldoutPrediction(result);

      // Compare prediction with actual value
      const actualValue = holdoutRow.target_value;
      const predictedValue = result.prediction;

      // For classification, compare directly; for regression, check if close
      const taskType = testingData?.task_type || servingConfig?.task_type || 'unknown';
      let isCorrect = false;

      if (taskType === 'regression') {
        // For regression, consider "correct" if within 10% or very close
        const actual = Number(actualValue);
        const predicted = Number(predictedValue);
        const tolerance = Math.max(Math.abs(actual) * 0.1, 0.01);
        isCorrect = Math.abs(actual - predicted) <= tolerance;
      } else {
        // For classification, exact match
        isCorrect = String(actualValue) === String(predictedValue);
      }

      setHoldoutValidationResult(isCorrect ? 'correct' : 'incorrect');

      // Update stats
      setHoldoutStats(prev => ({
        correct: prev.correct + (isCorrect ? 1 : 0),
        incorrect: prev.incorrect + (isCorrect ? 0 : 1),
        tested: prev.tested + 1
      }));
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to make prediction');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleNextHoldoutRow = () => {
    if (holdoutSet && holdoutRowIndex < holdoutSet.holdout_row_count - 1) {
      loadHoldoutRow(holdoutRowIndex + 1);
    }
  };

  const handlePrevHoldoutRow = () => {
    if (holdoutRowIndex > 0) {
      loadHoldoutRow(holdoutRowIndex - 1);
    }
  };

  const loadPipelineInfo = async () => {
    try {
      const info = await getModelPipeline(modelId);
      setPipelineInfo(info);
    } catch (err) {
      console.error('Failed to load pipeline info:', err);
    }
  };

  const loadTestingData = async () => {
    setIsLoadingTestingData(true);
    try {
      const data = await getModelTestingData(modelId);
      setTestingData(data);
      // Auto-load sample data if available
      if (data.sample_data) {
        setFeatures(data.sample_data);
      } else {
        // Use defaults (median for numeric, most_common for categorical)
        const defaults: Record<string, unknown> = {};
        for (const feat of data.features) {
          if (feat.type === 'numeric' && feat.median_value !== null) {
            defaults[feat.name] = feat.median_value;
          } else if (feat.most_common !== null) {
            defaults[feat.name] = feat.most_common;
          } else if (feat.type === 'boolean') {
            defaults[feat.name] = false;
          } else {
            defaults[feat.name] = '';
          }
        }
        setFeatures(defaults);
      }
    } catch (err) {
      console.error('Failed to load testing data:', err);
      // Fall back to basic mode if testing data unavailable
    } finally {
      setIsLoadingTestingData(false);
    }
  };

  const loadRandomSample = async () => {
    setIsLoadingSample(true);
    setPrediction(null);
    try {
      const sample = await getRandomSample(modelId);
      setFeatures(sample.features);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load sample');
      }
    } finally {
      setIsLoadingSample(false);
    }
  };

  const handleInputChange = (featureName: string, value: unknown) => {
    setFeatures((prev) => ({
      ...prev,
      [featureName]: value,
    }));
    setPrediction(null);
  };

  const handleSliderChange = (featureName: string, value: number) => {
    handleInputChange(featureName, value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      // Convert string values to appropriate types
      const typedFeatures: Record<string, unknown> = {};
      const featureList = testingData?.features || servingConfig?.features || [];

      for (const feat of featureList) {
        const featName = typeof feat === 'object' && 'name' in feat ? feat.name : '';
        const featType = typeof feat === 'object' && 'type' in feat ? feat.type : 'string';
        const value = features[featName];

        if (featType === 'numeric') {
          typedFeatures[featName] = value === '' || value === null ? 0 : Number(value);
        } else if (featType === 'boolean') {
          typedFeatures[featName] = value === true || value === 'true' || value === '1';
        } else {
          typedFeatures[featName] = value ?? '';
        }
      }

      const result = await predictWithModel(modelId, typedFeatures);
      setPrediction(result);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to make prediction');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    if (testingData?.sample_data) {
      setFeatures(testingData.sample_data);
    } else {
      setFeatures({});
    }
    setPrediction(null);
    setRawPrediction(null);
    setError(null);
    setRawDataInput('');
    setParsedRecords([]);
    setUploadedFileName('');
  };

  // Parse CSV content to records
  const parseCSV = (content: string): Record<string, unknown>[] => {
    const lines = content.trim().split('\n');
    if (lines.length < 2) return [];

    // Parse header - handle quoted values
    const parseRow = (row: string): string[] => {
      const result: string[] = [];
      let current = '';
      let inQuotes = false;

      for (let i = 0; i < row.length; i++) {
        const char = row[i];
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          result.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      result.push(current.trim());
      return result;
    };

    const headers = parseRow(lines[0]);
    const records: Record<string, unknown>[] = [];

    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      const values = parseRow(lines[i]);
      const record: Record<string, unknown> = {};

      headers.forEach((header, idx) => {
        const value = values[idx] || '';
        // Try to parse as number
        const num = parseFloat(value);
        record[header] = isNaN(num) ? value : num;
      });

      records.push(record);
    }

    return records;
  };

  // Handle file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError(null);
    setUploadedFileName(file.name);

    try {
      const content = await file.text();
      const ext = file.name.toLowerCase().split('.').pop();

      let records: Record<string, unknown>[] = [];

      if (ext === 'json') {
        const parsed = JSON.parse(content);
        if (Array.isArray(parsed)) {
          records = parsed;
        } else if (typeof parsed === 'object') {
          records = [parsed];
        }
      } else if (ext === 'csv' || ext === 'txt') {
        records = parseCSV(content);
      } else {
        // Try to detect format
        const trimmed = content.trim();
        if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
          const parsed = JSON.parse(content);
          records = Array.isArray(parsed) ? parsed : [parsed];
        } else {
          records = parseCSV(content);
        }
      }

      if (records.length === 0) {
        throw new Error('No data found in file');
      }

      setParsedRecords(records);
      setRawDataInput(JSON.stringify(records[0], null, 2));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse file');
      setParsedRecords([]);
    }
  };

  // Handle raw data submit - parses JSON and calls predict-raw endpoint
  const handleRawSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setRawPrediction(null);

    try {
      // Parse the JSON input
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(rawDataInput);
      } catch {
        throw new Error('Invalid JSON format. Please enter valid JSON data.');
      }

      if (typeof data !== 'object' || data === null || Array.isArray(data)) {
        throw new Error('Please enter a JSON object with your data fields.');
      }

      if (Object.keys(data).length === 0) {
        throw new Error('Please enter at least one data field.');
      }

      // Call the raw prediction endpoint which applies transformations
      const result = await predictWithRawData(modelId, data, true);
      setRawPrediction(result);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to make prediction');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Get feature info
  const getFeatureStats = (name: string): FeatureStatistics | undefined => {
    return testingData?.features.find(f => f.name === name);
  };

  // Separate top features from others
  const topFeatureNames = testingData?.top_features || [];
  const allFeatureNames = testingData?.features.map(f => f.name) ||
    servingConfig?.features?.map(f => f.name) || [];
  const otherFeatureNames = allFeatureNames.filter(n => !topFeatureNames.includes(n));

  const taskType = testingData?.task_type || servingConfig?.task_type || 'unknown';
  const isClassification = taskType === 'binary' || taskType === 'multiclass' || taskType === 'classification';

  // Render a single feature input
  const renderFeatureInput = (featureName: string, isTopFeature: boolean) => {
    const stats = getFeatureStats(featureName);
    const value = features[featureName];
    const type = stats?.type || 'string';

    return (
      <div
        key={featureName}
        className={`feature-input-row ${isTopFeature ? 'top-feature' : ''}`}
      >
        <div className="feature-label">
          <span className="feature-name">{featureName}</span>
          {stats?.importance !== null && stats?.importance !== undefined && (
            <span className="feature-importance" title="Feature importance">
              {(stats.importance * 100).toFixed(1)}%
            </span>
          )}
        </div>

        <div className="feature-control">
          {type === 'numeric' && stats && stats.min_value !== null && stats.max_value !== null ? (
            <div className="slider-container">
              <input
                type="range"
                min={stats.min_value}
                max={stats.max_value}
                step={(stats.max_value - stats.min_value) / 100}
                value={Number(value) || stats.median_value || stats.min_value}
                onChange={(e) => handleSliderChange(featureName, parseFloat(e.target.value))}
                className="feature-slider"
              />
              <input
                type="number"
                value={value !== undefined && value !== null ? String(value) : ''}
                onChange={(e) => handleInputChange(featureName, e.target.value === '' ? null : parseFloat(e.target.value))}
                className="slider-value"
                step="any"
              />
            </div>
          ) : type === 'boolean' ? (
            <select
              value={String(value)}
              onChange={(e) => handleInputChange(featureName, e.target.value === 'true')}
              className="feature-select"
            >
              <option value="false">False</option>
              <option value="true">True</option>
            </select>
          ) : stats?.categories && stats.categories.length > 0 ? (
            <select
              value={String(value || '')}
              onChange={(e) => handleInputChange(featureName, e.target.value)}
              className="feature-select"
            >
              <option value="">Select...</option>
              {stats.categories.map(cat => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          ) : type === 'numeric' ? (
            <input
              type="number"
              value={value !== undefined && value !== null ? String(value) : ''}
              onChange={(e) => handleInputChange(featureName, e.target.value === '' ? null : parseFloat(e.target.value))}
              className="feature-input"
              step="any"
              placeholder={stats?.median_value !== null ? `e.g., ${stats?.median_value?.toFixed(2)}` : 'Enter number'}
            />
          ) : (
            <input
              type="text"
              value={String(value || '')}
              onChange={(e) => handleInputChange(featureName, e.target.value)}
              className="feature-input"
              placeholder={stats?.most_common ? `e.g., ${stats.most_common}` : 'Enter value'}
            />
          )}
        </div>
      </div>
    );
  };

  if (isLoadingTestingData) {
    return (
      <div className="model-test-form">
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Loading model testing data...</p>
        </div>
        <style>{styles}</style>
      </div>
    );
  }

  if (!testingData && (!servingConfig?.features || servingConfig.features.length === 0)) {
    return (
      <div className="model-test-form">
        <div className="empty-state">
          <div className="empty-icon">🔮</div>
          <h3>Model Testing Unavailable</h3>
          <p>This model has no feature configuration for predictions.</p>
        </div>
        <style>{styles}</style>
      </div>
    );
  }

  return (
    <div className="model-test-form">
      {/* Header with mode tabs */}
      <div className="test-header">
        <div className="header-info">
          <h3>Test Model</h3>
          {testingData && (
            <span className="sample-count">
              {testingData.validation_sample_count} validation samples available
            </span>
          )}
        </div>
        <div className="header-actions">
          {inputMode === 'features' && (
            <button
              type="button"
              className="btn-load-sample"
              onClick={loadRandomSample}
              disabled={isLoadingSample || !testingData?.has_validation_samples}
              title="Load a random sample from validation data"
            >
              {isLoadingSample ? (
                <>
                  <span className="btn-spinner"></span>
                  Loading...
                </>
              ) : (
                <>
                  <span className="btn-icon">🎲</span>
                  Load Sample
                </>
              )}
            </button>
          )}
          <a
            href={getModelExportUrl(modelId)}
            className="btn-export"
            title="Download model as ZIP file"
          >
            <span className="btn-icon">📦</span>
            Export
          </a>
          <button
            type="button"
            className="btn-reset"
            onClick={handleReset}
          >
            Reset
          </button>
        </div>
      </div>

      {/* Mode Toggle Tabs */}
      <div className="mode-tabs">
        <button
          type="button"
          className={`mode-tab ${inputMode === 'features' ? 'active' : ''}`}
          onClick={() => setInputMode('features')}
        >
          <span className="tab-icon">📋</span>
          Engineered Features
        </button>
        <button
          type="button"
          className={`mode-tab ${inputMode === 'raw' ? 'active' : ''}`}
          onClick={() => setInputMode('raw')}
        >
          <span className="tab-icon">📄</span>
          Raw Data
          {pipelineInfo?.has_transformations && (
            <span className="transform-badge" title="Transformations will be applied">
              +{pipelineInfo.transformation_count} transforms
            </span>
          )}
        </button>
        {holdoutSet && (
          <button
            type="button"
            className={`mode-tab ${inputMode === 'holdout' ? 'active' : ''}`}
            onClick={() => setInputMode('holdout')}
          >
            <span className="tab-icon">🔒</span>
            Holdout Validation
            <span className="holdout-badge" title={`${holdoutSet.holdout_row_count} rows held out`}>
              {holdoutSet.holdout_row_count} rows
            </span>
          </button>
        )}
      </div>

      {/* Features Mode Form */}
      {inputMode === 'features' && (
        <form onSubmit={handleSubmit}>
          {/* Top Features Section */}
          {topFeatureNames.length > 0 && (
            <div className="features-section top-features-section">
              <div className="section-header">
                <h4>Key Features</h4>
                <span className="section-hint">Most important for predictions</span>
              </div>
              <div className="features-grid">
                {topFeatureNames.map(name => renderFeatureInput(name, true))}
              </div>
            </div>
          )}

          {/* Other Features Section (Collapsible) */}
          {otherFeatureNames.length > 0 && (
            <div className="features-section other-features-section">
              <button
                type="button"
                className="section-toggle"
                onClick={() => setShowAllFeatures(!showAllFeatures)}
              >
                <span className="toggle-icon">{showAllFeatures ? '▼' : '▶'}</span>
                <span>Other Features ({otherFeatureNames.length})</span>
              </button>
              {showAllFeatures && (
                <div className="features-grid collapsed-features">
                  {otherFeatureNames.map(name => renderFeatureInput(name, false))}
                </div>
              )}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          {/* Submit Button */}
          <div className="form-actions">
            <button
              type="submit"
              className="btn-predict"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="btn-spinner"></span>
                  Predicting...
                </>
              ) : (
                <>
                  <span className="btn-icon">🔮</span>
                  Make Prediction
                </>
              )}
            </button>
          </div>
        </form>
      )}

      {/* Raw Data Mode Form */}
      {inputMode === 'raw' && (
        <form onSubmit={handleRawSubmit}>
          <div className="raw-data-section">
            <div className="raw-input-section">
              <h4>Upload or Enter Raw Data</h4>
              <p className="section-hint">
                Upload a CSV/JSON file or paste data. Transformations will be applied automatically.
              </p>

              {/* File Upload */}
              <div className="file-upload-area">
                <input
                  type="file"
                  id="raw-file-input"
                  accept=".csv,.json,.txt"
                  onChange={handleFileUpload}
                  className="file-input-hidden"
                />
                <label htmlFor="raw-file-input" className="file-upload-label">
                  <span className="upload-icon">📁</span>
                  <span className="upload-text">
                    {uploadedFileName || 'Click to upload CSV or JSON file'}
                  </span>
                  {uploadedFileName && (
                    <span className="file-info">
                      {parsedRecords.length} record{parsedRecords.length !== 1 ? 's' : ''} loaded
                    </span>
                  )}
                </label>
              </div>

              {/* Record selector if multiple records */}
              {parsedRecords.length > 1 && (
                <div className="record-selector">
                  <label>Select record to predict:</label>
                  <select
                    onChange={(e) => {
                      const idx = parseInt(e.target.value);
                      setRawDataInput(JSON.stringify(parsedRecords[idx], null, 2));
                    }}
                  >
                    {parsedRecords.map((_, idx) => (
                      <option key={idx} value={idx}>
                        Record {idx + 1}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Show pipeline info if available */}
              {pipelineInfo && pipelineInfo.has_transformations && (
                <div className="pipeline-info">
                  <div className="pipeline-badge">
                    <span className="badge-icon">⚙️</span>
                    {pipelineInfo.transformation_count} transformations will be applied
                  </div>
                </div>
              )}

              <div className="or-divider">
                <span>or paste JSON</span>
              </div>

              <textarea
                className="raw-data-input"
                value={rawDataInput}
                onChange={(e) => {
                  setRawDataInput(e.target.value);
                  setParsedRecords([]);
                  setUploadedFileName('');
                }}
                placeholder={`{\n  "column1": value1,\n  "column2": value2,\n  ...\n}`}
                rows={8}
              />
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          {/* Submit Button */}
          <div className="form-actions">
            <button
              type="submit"
              className="btn-predict"
              disabled={isLoading || !rawDataInput.trim()}
            >
              {isLoading ? (
                <>
                  <span className="btn-spinner"></span>
                  Processing...
                </>
              ) : (
                <>
                  <span className="btn-icon">🔮</span>
                  Predict from Raw Data
                </>
              )}
            </button>
          </div>
        </form>
      )}

      {/* Holdout Validation Mode */}
      {inputMode === 'holdout' && holdoutSet && (
        <div className="holdout-validation-section">
          {/* Stats Bar */}
          <div className="holdout-stats-bar">
            <div className="holdout-stat">
              <span className="stat-label">Total Rows:</span>
              <span className="stat-value">{holdoutSet.holdout_row_count}</span>
            </div>
            <div className="holdout-stat">
              <span className="stat-label">Tested:</span>
              <span className="stat-value">{holdoutStats.tested}</span>
            </div>
            <div className="holdout-stat correct">
              <span className="stat-label">Correct:</span>
              <span className="stat-value">{holdoutStats.correct}</span>
            </div>
            <div className="holdout-stat incorrect">
              <span className="stat-label">Incorrect:</span>
              <span className="stat-value">{holdoutStats.incorrect}</span>
            </div>
            {holdoutStats.tested > 0 && (
              <div className="holdout-stat accuracy">
                <span className="stat-label">Accuracy:</span>
                <span className="stat-value">
                  {((holdoutStats.correct / holdoutStats.tested) * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>

          {/* Row Navigator */}
          <div className="holdout-row-navigator">
            <button
              type="button"
              className="nav-btn"
              onClick={handlePrevHoldoutRow}
              disabled={holdoutRowIndex === 0 || isLoadingHoldout}
            >
              ← Previous
            </button>
            <span className="row-indicator">
              Row {holdoutRowIndex + 1} of {holdoutSet.holdout_row_count}
            </span>
            <button
              type="button"
              className="nav-btn"
              onClick={handleNextHoldoutRow}
              disabled={holdoutRowIndex >= holdoutSet.holdout_row_count - 1 || isLoadingHoldout}
            >
              Next →
            </button>
          </div>

          {/* Current Row Features */}
          {isLoadingHoldout ? (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Loading holdout row...</p>
            </div>
          ) : holdoutRow ? (
            <div className="holdout-row-data">
              <h4>Feature Values (Row {holdoutRowIndex + 1})</h4>
              <div className="holdout-features-grid">
                {Object.entries(holdoutRow.data).map(([key, value]) => (
                  <div key={key} className="holdout-feature-item">
                    <span className="feature-key">{key}</span>
                    <span className="feature-val">{String(value)}</span>
                  </div>
                ))}
              </div>

              {/* Target column info */}
              {holdoutSet.target_column && (
                <div className="holdout-target-info">
                  <span className="target-label">Target ({holdoutSet.target_column}):</span>
                  <span className="target-value hidden-until-predict">
                    {holdoutPrediction ? String(holdoutRow.target_value) : '🔒 Hidden until prediction'}
                  </span>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  {error}
                </div>
              )}

              {/* Predict Button */}
              <div className="form-actions">
                <button
                  type="button"
                  className="btn-predict"
                  onClick={handleHoldoutPredict}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <span className="btn-spinner"></span>
                      Predicting...
                    </>
                  ) : (
                    <>
                      <span className="btn-icon">🔮</span>
                      Predict This Row
                    </>
                  )}
                </button>
              </div>

              {/* Holdout Prediction Result */}
              {holdoutPrediction && (
                <div className={`holdout-prediction-result ${holdoutValidationResult}`}>
                  <div className="result-header">
                    <span className="result-icon">
                      {holdoutValidationResult === 'correct' ? '✓' : '✗'}
                    </span>
                    <h4>
                      {holdoutValidationResult === 'correct' ? 'Correct!' : 'Incorrect'}
                    </h4>
                  </div>

                  <div className="result-comparison">
                    <div className="result-row">
                      <span className="result-label">Predicted:</span>
                      <span className="result-val predicted">{String(holdoutPrediction.prediction)}</span>
                    </div>
                    <div className="result-row">
                      <span className="result-label">Actual:</span>
                      <span className="result-val actual">{String(holdoutRow.target_value)}</span>
                    </div>
                  </div>

                  {isClassification && holdoutPrediction.probabilities && (
                    <div className="probabilities-section">
                      <h5>Class Probabilities</h5>
                      <div className="probability-bars">
                        {Object.entries(holdoutPrediction.probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([className, prob]) => {
                            const isTop = prob === Math.max(...Object.values(holdoutPrediction.probabilities!));
                            const isActual = String(className) === String(holdoutRow.target_value);
                            return (
                              <div key={className} className={`probability-row ${isTop ? 'top-class' : ''} ${isActual ? 'actual-class' : ''}`}>
                                <span className="class-name">
                                  {className}
                                  {isActual && <span className="actual-marker"> (actual)</span>}
                                </span>
                                <div className="probability-bar-container">
                                  <div
                                    className="probability-bar"
                                    style={{ width: `${(prob * 100).toFixed(1)}%` }}
                                  />
                                </div>
                                <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state">
              <p>No holdout data available</p>
            </div>
          )}
        </div>
      )}

      {/* Prediction Result */}
      {/* Features Mode Prediction Result */}
      {prediction && inputMode === 'features' && (
        <div className="prediction-result">
          <div className="result-header">
            <span className="result-icon">🎯</span>
            <h4>Prediction Result</h4>
          </div>

          <div className="result-value">
            {String(prediction.prediction)}
          </div>

          {isClassification && prediction.probabilities && (
            <div className="probabilities-section">
              <h5>Class Probabilities</h5>
              <div className="probability-bars">
                {Object.entries(prediction.probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([className, prob]) => {
                    const isTop = prob === Math.max(...Object.values(prediction.probabilities!));
                    return (
                      <div key={className} className={`probability-row ${isTop ? 'top-class' : ''}`}>
                        <span className="class-name">{className}</span>
                        <div className="probability-bar-container">
                          <div
                            className="probability-bar"
                            style={{ width: `${(prob * 100).toFixed(1)}%` }}
                          />
                        </div>
                        <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Raw Data Mode Prediction Result */}
      {rawPrediction && inputMode === 'raw' && (
        <div className="prediction-result">
          <div className="result-header">
            <span className="result-icon">🎯</span>
            <h4>Prediction Result</h4>
            {rawPrediction.transformations_applied && (
              <span className="transform-applied-badge">✓ Transformations applied</span>
            )}
          </div>

          <div className="result-value">
            {String(rawPrediction.prediction)}
          </div>

          {isClassification && rawPrediction.probabilities && (
            <div className="probabilities-section">
              <h5>Class Probabilities</h5>
              <div className="probability-bars">
                {Object.entries(rawPrediction.probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([className, prob]) => {
                    const isTop = prob === Math.max(...Object.values(rawPrediction.probabilities!));
                    return (
                      <div key={className} className={`probability-row ${isTop ? 'top-class' : ''}`}>
                        <span className="class-name">{className}</span>
                        <div className="probability-bar-container">
                          <div
                            className="probability-bar"
                            style={{ width: `${(prob * 100).toFixed(1)}%` }}
                          />
                        </div>
                        <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Show transformed features (for debugging) */}
          {rawPrediction.transformed_features && (
            <details className="transformed-features-details">
              <summary>View Transformed Features</summary>
              <pre className="transformed-features-json">
                {JSON.stringify(rawPrediction.transformed_features, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}

      <style>{styles}</style>
    </div>
  );
}

const styles = `
  .model-test-form {
    padding: 0;
  }

  .loading-state,
  .empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #94a3b8;
  }

  .empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
  }

  .empty-state h3 {
    color: #e2e8f0;
    margin: 0 0 0.5rem;
  }

  .spinner,
  .btn-spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #334155;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    display: inline-block;
  }

  .btn-spinner {
    width: 14px;
    height: 14px;
    margin-right: 0.5rem;
    border-color: rgba(255,255,255,0.3);
    border-top-color: white;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Header */
  .test-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #334155;
  }

  .header-info h3 {
    margin: 0;
    font-size: 1.125rem;
    color: #e2e8f0;
  }

  .sample-count {
    font-size: 0.75rem;
    color: #64748b;
  }

  .header-actions {
    display: flex;
    gap: 0.5rem;
  }

  .btn-load-sample,
  .btn-reset {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-load-sample {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
  }

  .btn-load-sample:hover:not(:disabled) {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
  }

  .btn-load-sample:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .btn-reset {
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
  }

  .btn-reset:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .btn-icon {
    font-size: 1rem;
  }

  /* Features Sections */
  .features-section {
    margin-bottom: 1.5rem;
  }

  .section-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .section-header h4 {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 600;
    color: #e94560;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .section-hint {
    font-size: 0.75rem;
    color: #64748b;
  }

  .section-toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    width: 100%;
    text-align: left;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 500;
    color: #94a3b8;
    transition: all 0.2s;
  }

  .section-toggle:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .toggle-icon {
    font-size: 0.625rem;
    color: #64748b;
  }

  .features-grid {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .collapsed-features {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #334155;
  }

  /* Feature Input Row */
  .feature-input-row {
    display: grid;
    grid-template-columns: 180px 1fr;
    gap: 1rem;
    align-items: center;
    padding: 0.5rem 0;
  }

  .feature-input-row.top-feature {
    background: #1e293b;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0 -0.5rem;
    border: 1px solid #334155;
  }

  .feature-label {
    display: flex;
    flex-direction: column;
    gap: 0.125rem;
  }

  .feature-name {
    font-size: 0.875rem;
    font-weight: 500;
    color: #e2e8f0;
  }

  .feature-importance {
    font-size: 0.6875rem;
    color: #3b82f6;
    font-weight: 500;
  }

  .feature-control {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* Slider */
  .slider-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    width: 100%;
  }

  .feature-slider {
    flex: 1;
    height: 6px;
    -webkit-appearance: none;
    appearance: none;
    background: #334155;
    border-radius: 3px;
    cursor: pointer;
  }

  .feature-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    background: #3b82f6;
    border-radius: 50%;
    cursor: pointer;
    transition: transform 0.1s;
  }

  .feature-slider::-webkit-slider-thumb:hover {
    transform: scale(1.1);
  }

  .slider-value {
    width: 90px;
    padding: 0.375rem 0.5rem;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    font-size: 0.875rem;
    text-align: right;
    color: #e2e8f0;
  }

  /* Inputs */
  .feature-input,
  .feature-select {
    width: 100%;
    padding: 0.5rem 0.75rem;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    font-size: 0.875rem;
    color: #e2e8f0;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  .feature-input::placeholder {
    color: #64748b;
  }

  .feature-input:focus,
  .feature-select:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
  }

  .feature-select option {
    background: #1e293b;
    color: #e2e8f0;
  }

  /* Error */
  .error-message {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    color: #f87171;
    font-size: 0.875rem;
    margin-bottom: 1rem;
  }

  /* Form Actions */
  .form-actions {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #334155;
  }

  .btn-predict {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.75rem 2rem;
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-predict:hover:not(:disabled) {
    background: linear-gradient(135deg, #059669, #047857);
    transform: translateY(-1px);
  }

  .btn-predict:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }

  /* Prediction Result */
  .prediction-result {
    margin-top: 1.5rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
    border: 2px solid #10b981;
    border-radius: 12px;
  }

  .result-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .result-icon {
    font-size: 1.5rem;
  }

  .result-header h4 {
    margin: 0;
    font-size: 1rem;
    color: #10b981;
  }

  .result-value {
    font-size: 2rem;
    font-weight: 700;
    color: #34d399;
    margin-bottom: 1rem;
  }

  .probabilities-section h5 {
    margin: 0 0 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #10b981;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .probability-bars {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .probability-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .probability-row.top-class .class-name,
  .probability-row.top-class .probability-value {
    font-weight: 600;
    color: #34d399;
  }

  .class-name {
    min-width: 80px;
    font-size: 0.875rem;
    color: #94a3b8;
  }

  .probability-bar-container {
    flex: 1;
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }

  .probability-bar {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #059669);
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .probability-value {
    min-width: 50px;
    text-align: right;
    font-size: 0.875rem;
    color: #94a3b8;
  }

  /* Mode Tabs */
  .mode-tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    padding: 0.25rem;
    background: #1e293b;
    border-radius: 8px;
    border: 1px solid #334155;
  }

  .mode-tab {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: transparent;
    border: none;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s;
  }

  .mode-tab:hover {
    background: #334155;
    color: #e2e8f0;
  }

  .mode-tab.active {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
  }

  .tab-icon {
    font-size: 1rem;
  }

  .transform-badge {
    font-size: 0.625rem;
    padding: 0.125rem 0.375rem;
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border-radius: 4px;
    margin-left: 0.25rem;
  }

  .mode-tab.active .transform-badge {
    background: rgba(255, 255, 255, 0.2);
    color: white;
  }

  /* Export Button */
  .btn-export {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.5rem 1rem;
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-export:hover {
    background: #334155;
    color: #e2e8f0;
  }

  /* Raw Data Section */
  .raw-data-section {
    margin-bottom: 1rem;
  }

  .pipeline-info {
    margin-bottom: 1rem;
    padding: 0.75rem 1rem;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
  }

  .pipeline-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: #60a5fa;
    font-weight: 500;
  }

  .badge-icon {
    font-size: 1rem;
  }

  .required-columns {
    margin-top: 0.5rem;
    font-size: 0.75rem;
    color: #94a3b8;
  }

  .required-columns strong {
    color: #e2e8f0;
  }

  .raw-data-input {
    width: 100%;
    padding: 1rem;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    font-family: monospace;
    font-size: 0.875rem;
    color: #e2e8f0;
    resize: vertical;
    min-height: 150px;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  .raw-data-input::placeholder {
    color: #64748b;
  }

  .raw-data-input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
  }

  .input-format-hint {
    margin-top: 0.75rem;
  }

  .input-format-hint summary {
    font-size: 0.75rem;
    color: #64748b;
    cursor: pointer;
    padding: 0.5rem 0;
  }

  .input-format-hint summary:hover {
    color: #94a3b8;
  }

  .format-examples {
    margin-top: 0.5rem;
    padding: 0.75rem;
    background: #0f172a;
    border-radius: 6px;
    border: 1px solid #334155;
  }

  .format-example {
    margin-bottom: 0.75rem;
  }

  .format-example:last-child {
    margin-bottom: 0;
  }

  .format-example strong {
    display: block;
    font-size: 0.75rem;
    color: #94a3b8;
    margin-bottom: 0.25rem;
  }

  .format-example pre {
    margin: 0;
    padding: 0.5rem;
    background: #1e293b;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #e2e8f0;
    overflow-x: auto;
  }

  /* Simple Inputs */
  .simple-inputs {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .simple-input-row {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .simple-input-row label {
    min-width: 120px;
    font-size: 0.9375rem;
    font-weight: 500;
    color: #e2e8f0;
  }

  .simple-input-row input {
    flex: 1;
    padding: 0.75rem 1rem;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    font-size: 1rem;
    color: #e2e8f0;
  }

  .simple-input-row input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
  }

  .simple-input-row input::placeholder {
    color: #64748b;
  }

  .no-inputs-msg {
    color: #94a3b8;
    font-size: 0.875rem;
    text-align: center;
    padding: 1rem;
  }

  /* Raw Input Section */
  .raw-input-section {
    padding: 1.25rem;
    background: #1e293b;
    border-radius: 12px;
    border: 1px solid #334155;
  }

  .raw-input-section h4 {
    margin: 0 0 0.25rem;
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
  }

  .raw-input-section .section-hint {
    margin-bottom: 1rem;
    color: #94a3b8;
    font-size: 0.875rem;
  }

  /* File Upload */
  .file-upload-area {
    margin-bottom: 1rem;
  }

  .file-input-hidden {
    display: none;
  }

  .file-upload-label {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 1.5rem;
    background: #0f172a;
    border: 2px dashed #334155;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .file-upload-label:hover {
    border-color: #3b82f6;
    background: rgba(59, 130, 246, 0.05);
  }

  .upload-icon {
    font-size: 1.5rem;
  }

  .upload-text {
    font-size: 0.875rem;
    color: #94a3b8;
  }

  .file-info {
    font-size: 0.75rem;
    color: #10b981;
    font-weight: 500;
  }

  /* Record Selector */
  .record-selector {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding: 0.75rem;
    background: rgba(59, 130, 246, 0.1);
    border-radius: 6px;
  }

  .record-selector label {
    font-size: 0.875rem;
    color: #94a3b8;
  }

  .record-selector select {
    padding: 0.5rem 0.75rem;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #e2e8f0;
    font-size: 0.875rem;
  }

  /* Or Divider */
  .or-divider {
    display: flex;
    align-items: center;
    margin: 1rem 0;
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
  }

  .or-divider::before,
  .or-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #334155;
  }

  .or-divider span {
    padding: 0 0.75rem;
  }

  /* Transform Applied Badge */
  .transform-applied-badge {
    margin-left: auto;
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border-radius: 4px;
  }

  /* Transformed Features Details */
  .transformed-features-details {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(16, 185, 129, 0.3);
  }

  .transformed-features-details summary {
    font-size: 0.75rem;
    color: #34d399;
    cursor: pointer;
    padding: 0.5rem 0;
  }

  .transformed-features-json {
    margin-top: 0.5rem;
    padding: 0.75rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 6px;
    font-size: 0.75rem;
    color: #94a3b8;
    overflow-x: auto;
    max-height: 200px;
  }

  /* Responsive */
  @media (max-width: 640px) {
    .test-header {
      flex-direction: column;
      align-items: flex-start;
      gap: 1rem;
    }

    .mode-tabs {
      flex-direction: column;
    }

    .feature-input-row {
      grid-template-columns: 1fr;
      gap: 0.5rem;
    }

    .feature-input-row.top-feature {
      padding: 1rem;
    }

    .slider-container {
      flex-direction: column;
      align-items: stretch;
    }

    .slider-value {
      width: 100%;
      text-align: left;
    }
  }

  /* Holdout Validation Styles */
  .holdout-badge {
    font-size: 0.625rem;
    padding: 0.125rem 0.375rem;
    background: rgba(139, 92, 246, 0.2);
    color: #a78bfa;
    border-radius: 4px;
    margin-left: 0.25rem;
  }

  .mode-tab.active .holdout-badge {
    background: rgba(255, 255, 255, 0.2);
    color: white;
  }

  .holdout-validation-section {
    padding: 1rem;
    background: #1e293b;
    border-radius: 12px;
    border: 1px solid #334155;
  }

  .holdout-stats-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    padding: 1rem;
    background: #0f172a;
    border-radius: 8px;
    margin-bottom: 1.5rem;
  }

  .holdout-stat {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .holdout-stat .stat-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
  }

  .holdout-stat .stat-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e2e8f0;
  }

  .holdout-stat.correct .stat-value {
    color: #10b981;
  }

  .holdout-stat.incorrect .stat-value {
    color: #ef4444;
  }

  .holdout-stat.accuracy .stat-value {
    color: #3b82f6;
  }

  .holdout-row-navigator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding: 0.75rem;
    background: #0f172a;
    border-radius: 8px;
  }

  .holdout-row-navigator .nav-btn {
    padding: 0.5rem 1rem;
    background: #334155;
    border: none;
    border-radius: 6px;
    color: #e2e8f0;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .holdout-row-navigator .nav-btn:hover:not(:disabled) {
    background: #475569;
  }

  .holdout-row-navigator .nav-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .holdout-row-navigator .row-indicator {
    font-size: 0.875rem;
    color: #94a3b8;
    min-width: 120px;
    text-align: center;
  }

  .holdout-row-data h4 {
    margin: 0 0 1rem;
    font-size: 1rem;
    color: #e2e8f0;
  }

  .holdout-features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }

  .holdout-feature-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.75rem;
    background: #0f172a;
    border-radius: 6px;
    border: 1px solid #334155;
  }

  .holdout-feature-item .feature-key {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
  }

  .holdout-feature-item .feature-val {
    font-size: 0.875rem;
    color: #e2e8f0;
    font-weight: 500;
  }

  .holdout-target-info {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 8px;
    margin-bottom: 1rem;
  }

  .holdout-target-info .target-label {
    font-size: 0.875rem;
    color: #a78bfa;
    font-weight: 500;
  }

  .holdout-target-info .target-value {
    font-size: 0.875rem;
    color: #e2e8f0;
  }

  .holdout-target-info .target-value.hidden-until-predict {
    color: #64748b;
    font-style: italic;
  }

  .holdout-prediction-result {
    margin-top: 1.5rem;
    padding: 1.5rem;
    border-radius: 12px;
  }

  .holdout-prediction-result.correct {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
    border: 2px solid #10b981;
  }

  .holdout-prediction-result.incorrect {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.1));
    border: 2px solid #ef4444;
  }

  .holdout-prediction-result .result-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .holdout-prediction-result.correct .result-header .result-icon {
    font-size: 1.5rem;
    color: #10b981;
  }

  .holdout-prediction-result.incorrect .result-header .result-icon {
    font-size: 1.5rem;
    color: #ef4444;
  }

  .holdout-prediction-result.correct .result-header h4 {
    color: #10b981;
    margin: 0;
  }

  .holdout-prediction-result.incorrect .result-header h4 {
    color: #ef4444;
    margin: 0;
  }

  .result-comparison {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .result-comparison .result-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .result-comparison .result-label {
    font-size: 0.875rem;
    color: #94a3b8;
    min-width: 80px;
  }

  .result-comparison .result-val {
    font-size: 1.25rem;
    font-weight: 600;
  }

  .result-comparison .result-val.predicted {
    color: #3b82f6;
  }

  .result-comparison .result-val.actual {
    color: #a78bfa;
  }

  .probability-row.actual-class .class-name {
    color: #a78bfa;
    font-weight: 600;
  }

  .actual-marker {
    font-size: 0.75rem;
    color: #a78bfa;
    font-weight: normal;
  }
`;

import { useState } from 'react';
import type { DatasetSpec, ExperimentCreate, TaskType, MetricDirection } from '../types/api';

interface CreateExperimentFormProps {
  datasetSpecs: DatasetSpec[];
  projectTaskType: TaskType | null;
  onSubmit: (data: ExperimentCreate) => Promise<void>;
  onCancel: () => void;
  isLoading?: boolean;
}

const metricOptions: Record<string, { value: string; label: string; direction: MetricDirection }[]> = {
  binary: [
    { value: 'roc_auc', label: 'ROC AUC', direction: 'maximize' },
    { value: 'accuracy', label: 'Accuracy', direction: 'maximize' },
    { value: 'f1', label: 'F1 Score', direction: 'maximize' },
    { value: 'precision', label: 'Precision', direction: 'maximize' },
    { value: 'recall', label: 'Recall', direction: 'maximize' },
  ],
  multiclass: [
    { value: 'accuracy', label: 'Accuracy', direction: 'maximize' },
    { value: 'f1_macro', label: 'F1 Macro', direction: 'maximize' },
    { value: 'f1_weighted', label: 'F1 Weighted', direction: 'maximize' },
  ],
  regression: [
    { value: 'rmse', label: 'RMSE', direction: 'minimize' },
    { value: 'mae', label: 'MAE', direction: 'minimize' },
    { value: 'r2', label: 'R2 Score', direction: 'maximize' },
  ],
  quantile: [
    { value: 'pinball_loss', label: 'Pinball Loss', direction: 'minimize' },
  ],
  timeseries_forecast: [
    { value: 'MASE', label: 'MASE', direction: 'minimize' },
    { value: 'MAPE', label: 'MAPE', direction: 'minimize' },
    { value: 'RMSE', label: 'RMSE', direction: 'minimize' },
  ],
  multimodal_classification: [
    { value: 'accuracy', label: 'Accuracy', direction: 'maximize' },
    { value: 'f1', label: 'F1 Score', direction: 'maximize' },
  ],
  multimodal_regression: [
    { value: 'rmse', label: 'RMSE', direction: 'minimize' },
    { value: 'mae', label: 'MAE', direction: 'minimize' },
  ],
  classification: [
    { value: 'roc_auc', label: 'ROC AUC', direction: 'maximize' },
    { value: 'accuracy', label: 'Accuracy', direction: 'maximize' },
  ],
};

const presetOptions = [
  { value: 'medium_quality', label: 'Medium Quality (Recommended)', description: 'Good balance of speed and accuracy' },
  { value: 'best_quality', label: 'Best Quality', description: 'Maximum accuracy, slower training' },
  { value: 'high_quality', label: 'High Quality', description: 'Higher accuracy than medium' },
  { value: 'good_quality', label: 'Good Quality', description: 'Faster than high quality' },
];

export default function CreateExperimentForm({
  datasetSpecs,
  projectTaskType,
  onSubmit,
  onCancel,
  isLoading = false,
}: CreateExperimentFormProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [datasetSpecId, setDatasetSpecId] = useState('');
  const [primaryMetric, setPrimaryMetric] = useState('');
  const [metricDirection, setMetricDirection] = useState<MetricDirection>('maximize');
  const [timeLimit, setTimeLimit] = useState(300);
  const [presets, setPresets] = useState('medium_quality');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Time series specific options
  const [predictionLength, setPredictionLength] = useState(7);
  const [timeColumn, setTimeColumn] = useState('');

  const taskType = projectTaskType || 'binary';
  const metrics = metricOptions[taskType] || metricOptions.binary;
  const isTimeSeries = taskType === 'timeseries_forecast';

  const handleMetricChange = (metric: string) => {
    setPrimaryMetric(metric);
    const metricConfig = metrics.find((m) => m.value === metric);
    if (metricConfig) {
      setMetricDirection(metricConfig.direction);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('Experiment name is required');
      return;
    }

    if (!datasetSpecId) {
      setError('Please select a dataset');
      return;
    }

    if (isTimeSeries && !timeColumn) {
      setError('Time column is required for time series forecasting');
      return;
    }

    const experimentPlan: Record<string, unknown> = {
      automl_config: {
        time_limit: timeLimit,
        presets,
      },
    };

    if (isTimeSeries) {
      (experimentPlan.automl_config as Record<string, unknown>).prediction_length = predictionLength;
      (experimentPlan.automl_config as Record<string, unknown>).time_column = timeColumn;
    }

    try {
      await onSubmit({
        name: name.trim(),
        description: description.trim() || null,
        dataset_spec_id: datasetSpecId,
        primary_metric: primaryMetric || null,
        metric_direction: primaryMetric ? metricDirection : null,
        experiment_plan_json: experimentPlan,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create experiment');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="form">
      {error && <div className="form-error">{error}</div>}

      <div className="form-group">
        <label htmlFor="expName" className="form-label">
          Experiment Name <span className="required">*</span>
        </label>
        <input
          type="text"
          id="expName"
          className="form-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="My Experiment"
          maxLength={255}
          disabled={isLoading}
          autoFocus
        />
      </div>

      <div className="form-group">
        <label htmlFor="expDescription" className="form-label">
          Description
        </label>
        <textarea
          id="expDescription"
          className="form-textarea"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe what this experiment is testing..."
          rows={2}
          disabled={isLoading}
        />
      </div>

      <div className="form-group">
        <label htmlFor="datasetSpec" className="form-label">
          Dataset <span className="required">*</span>
        </label>
        <select
          id="datasetSpec"
          className="form-select"
          value={datasetSpecId}
          onChange={(e) => setDatasetSpecId(e.target.value)}
          disabled={isLoading || datasetSpecs.length === 0}
        >
          <option value="">Select a dataset</option>
          {datasetSpecs.map((ds) => (
            <option key={ds.id} value={ds.id}>
              {ds.name} (target: {ds.target_column || 'not set'})
            </option>
          ))}
        </select>
        {datasetSpecs.length === 0 && (
          <p className="form-hint">No datasets available. Create a dataset spec first.</p>
        )}
      </div>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="metric" className="form-label">
            Primary Metric
          </label>
          <select
            id="metric"
            className="form-select"
            value={primaryMetric}
            onChange={(e) => handleMetricChange(e.target.value)}
            disabled={isLoading}
          >
            <option value="">Auto (based on task type)</option>
            {metrics.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="timeLimit" className="form-label">
            Time Limit (seconds)
          </label>
          <input
            type="number"
            id="timeLimit"
            className="form-input"
            value={timeLimit}
            onChange={(e) => setTimeLimit(parseInt(e.target.value) || 60)}
            min={60}
            max={86400}
            disabled={isLoading}
          />
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="presets" className="form-label">
          Quality Preset
        </label>
        <select
          id="presets"
          className="form-select"
          value={presets}
          onChange={(e) => setPresets(e.target.value)}
          disabled={isLoading}
        >
          {presetOptions.map((p) => (
            <option key={p.value} value={p.value}>
              {p.label}
            </option>
          ))}
        </select>
        <p className="form-hint">
          {presetOptions.find((p) => p.value === presets)?.description}
        </p>
      </div>

      {isTimeSeries && (
        <>
          <div className="form-group">
            <label htmlFor="timeColumn" className="form-label">
              Time Column <span className="required">*</span>
            </label>
            <input
              type="text"
              id="timeColumn"
              className="form-input"
              value={timeColumn}
              onChange={(e) => setTimeColumn(e.target.value)}
              placeholder="timestamp"
              disabled={isLoading}
            />
            <p className="form-hint">Name of the datetime column in your data</p>
          </div>

          <div className="form-group">
            <label htmlFor="predictionLength" className="form-label">
              Prediction Length
            </label>
            <input
              type="number"
              id="predictionLength"
              className="form-input"
              value={predictionLength}
              onChange={(e) => setPredictionLength(parseInt(e.target.value) || 1)}
              min={1}
              max={365}
              disabled={isLoading}
            />
            <p className="form-hint">Number of time steps to forecast</p>
          </div>
        </>
      )}

      <button
        type="button"
        className="btn-link"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        {showAdvanced ? 'Hide' : 'Show'} Advanced Options
      </button>

      {showAdvanced && (
        <div className="advanced-options">
          <p className="form-hint">
            Advanced options can be configured via the API using experiment_plan_json.
            Options include: num_bag_folds, num_stack_levels, excluded_model_types, and more.
          </p>
        </div>
      )}

      <div className="form-actions">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onCancel}
          disabled={isLoading}
        >
          Cancel
        </button>
        <button type="submit" className="btn btn-primary" disabled={isLoading}>
          {isLoading ? 'Creating...' : 'Create Experiment'}
        </button>
      </div>
    </form>
  );
}

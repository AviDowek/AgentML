import { useState } from 'react';
import type { ProjectCreate, TaskType } from '../types/api';

interface CreateProjectFormProps {
  onSubmit: (data: ProjectCreate) => Promise<void>;
  onCancel: () => void;
  isLoading?: boolean;
}

const taskTypeOptions: { value: TaskType; label: string; description: string }[] = [
  {
    value: 'binary',
    label: 'Binary Classification',
    description: 'Predict one of two classes (e.g., spam/not spam)',
  },
  {
    value: 'multiclass',
    label: 'Multi-class Classification',
    description: 'Predict one of 3+ classes (e.g., product categories)',
  },
  {
    value: 'regression',
    label: 'Regression',
    description: 'Predict continuous values (e.g., house prices)',
  },
  {
    value: 'quantile',
    label: 'Quantile Regression',
    description: 'Predict value percentiles (e.g., 10th, 50th, 90th)',
  },
  {
    value: 'timeseries_forecast',
    label: 'Time Series Forecast',
    description: 'Forecast future values based on historical data',
  },
  {
    value: 'multimodal_classification',
    label: 'Multimodal Classification',
    description: 'Classify using text + tabular + images',
  },
  {
    value: 'multimodal_regression',
    label: 'Multimodal Regression',
    description: 'Predict values using text + tabular + images',
  },
];

export default function CreateProjectForm({
  onSubmit,
  onCancel,
  isLoading = false,
}: CreateProjectFormProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [taskType, setTaskType] = useState<TaskType | ''>('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('Project name is required');
      return;
    }

    try {
      await onSubmit({
        name: name.trim(),
        description: description.trim() || null,
        task_type: taskType || null,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="form">
      {error && <div className="form-error">{error}</div>}

      <div className="form-group">
        <label htmlFor="name" className="form-label">
          Project Name <span className="required">*</span>
        </label>
        <input
          type="text"
          id="name"
          className="form-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="My ML Project"
          maxLength={255}
          disabled={isLoading}
          autoFocus
        />
      </div>

      <div className="form-group">
        <label htmlFor="description" className="form-label">
          Description
        </label>
        <textarea
          id="description"
          className="form-textarea"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe what this project is for..."
          rows={3}
          disabled={isLoading}
        />
      </div>

      <div className="form-group">
        <label htmlFor="taskType" className="form-label">
          Task Type
        </label>
        <select
          id="taskType"
          className="form-select"
          value={taskType}
          onChange={(e) => setTaskType(e.target.value as TaskType)}
          disabled={isLoading}
        >
          <option value="">Select a task type (optional)</option>
          {taskTypeOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        {taskType && (
          <p className="form-hint">
            {taskTypeOptions.find((o) => o.value === taskType)?.description}
          </p>
        )}
      </div>

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
          {isLoading ? 'Creating...' : 'Create Project'}
        </button>
      </div>
    </form>
  );
}

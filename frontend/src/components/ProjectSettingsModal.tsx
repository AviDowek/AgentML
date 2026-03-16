import { useState } from 'react';
import type { Project, ProjectUpdate } from '../types/api';
import { updateProject, ApiException } from '../services/api';

interface ProjectSettingsModalProps {
  project: Project;
  isOpen: boolean;
  onClose: () => void;
  onUpdate: (project: Project) => void;
}

// Default values matching backend
const DEFAULT_MAX_TRAINING_ROWS = 1_000_000;
const DEFAULT_PROFILING_SAMPLE_ROWS = 50_000;
const DEFAULT_MAX_AGGREGATION_WINDOW_DAYS = 365;

export default function ProjectSettingsModal({
  project,
  isOpen,
  onClose,
  onUpdate,
}: ProjectSettingsModalProps) {
  const [maxTrainingRows, setMaxTrainingRows] = useState(
    project.max_training_rows || DEFAULT_MAX_TRAINING_ROWS
  );
  const [profilingSampleRows, setProfilingSampleRows] = useState(
    project.profiling_sample_rows || DEFAULT_PROFILING_SAMPLE_ROWS
  );
  const [maxAggregationWindowDays, setMaxAggregationWindowDays] = useState(
    project.max_aggregation_window_days || DEFAULT_MAX_AGGREGATION_WINDOW_DAYS
  );
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);

    try {
      const updateData: ProjectUpdate = {
        max_training_rows: maxTrainingRows,
        profiling_sample_rows: profilingSampleRows,
        max_aggregation_window_days: maxAggregationWindowDays,
      };

      const updatedProject = await updateProject(project.id, updateData);
      onUpdate(updatedProject);
      onClose();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to save settings');
      }
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    setMaxTrainingRows(DEFAULT_MAX_TRAINING_ROWS);
    setProfilingSampleRows(DEFAULT_PROFILING_SAMPLE_ROWS);
    setMaxAggregationWindowDays(DEFAULT_MAX_AGGREGATION_WINDOW_DAYS);
  };

  // Format number for display (e.g., 1000000 -> "1,000,000")
  const formatNumber = (num: number) => num.toLocaleString();

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '500px' }}>
        <div className="modal-header">
          <h3>Project Settings</h3>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        <div className="modal-body">
          {error && (
            <div className="form-error" style={{ marginBottom: '1rem' }}>
              {error}
            </div>
          )}

          <div style={{ marginBottom: '1.5rem' }}>
            <h4 style={{ marginBottom: '0.5rem', fontSize: '0.875rem', color: '#374151' }}>
              Large Dataset Safeguards
            </h4>
            <p style={{ fontSize: '0.75rem', color: '#6b7280', marginBottom: '1rem' }}>
              Control how large datasets are handled to prevent performance issues.
            </p>
          </div>

          <div className="form-group" style={{ marginBottom: '1.25rem' }}>
            <label
              htmlFor="max-training-rows"
              style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}
            >
              Max Training Rows
            </label>
            <input
              type="number"
              id="max-training-rows"
              className="form-input"
              value={maxTrainingRows}
              onChange={(e) => setMaxTrainingRows(parseInt(e.target.value) || 0)}
              min={1000}
              max={100000000}
              step={10000}
              style={{ width: '100%', padding: '0.5rem', border: '1px solid #d1d5db', borderRadius: '6px' }}
            />
            <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
              Maximum rows in materialized training datasets. Larger datasets will be sampled.
              <br />
              <span style={{ color: '#9ca3af' }}>
                Current: {formatNumber(maxTrainingRows)} rows
              </span>
            </p>
          </div>

          <div className="form-group" style={{ marginBottom: '1.25rem' }}>
            <label
              htmlFor="profiling-sample-rows"
              style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}
            >
              Profiling Sample Rows
            </label>
            <input
              type="number"
              id="profiling-sample-rows"
              className="form-input"
              value={profilingSampleRows}
              onChange={(e) => setProfilingSampleRows(parseInt(e.target.value) || 0)}
              min={100}
              max={1000000}
              step={1000}
              style={{ width: '100%', padding: '0.5rem', border: '1px solid #d1d5db', borderRadius: '6px' }}
            />
            <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
              Sample size used for data profiling and schema analysis.
              <br />
              <span style={{ color: '#9ca3af' }}>
                Current: {formatNumber(profilingSampleRows)} rows
              </span>
            </p>
          </div>

          <div className="form-group" style={{ marginBottom: '1rem' }}>
            <label
              htmlFor="max-aggregation-window"
              style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}
            >
              Max Aggregation Window (days)
            </label>
            <input
              type="number"
              id="max-aggregation-window"
              className="form-input"
              value={maxAggregationWindowDays}
              onChange={(e) => setMaxAggregationWindowDays(parseInt(e.target.value) || 0)}
              min={1}
              max={3650}
              step={30}
              style={{ width: '100%', padding: '0.5rem', border: '1px solid #d1d5db', borderRadius: '6px' }}
            />
            <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
              Maximum time window for aggregations when joining tables.
              <br />
              <span style={{ color: '#9ca3af' }}>
                Current: {maxAggregationWindowDays} days
              </span>
            </p>
          </div>
        </div>

        <div
          className="modal-footer"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: '1rem',
            borderTop: '1px solid #e5e7eb',
          }}
        >
          <button
            className="btn btn-secondary"
            onClick={handleReset}
            disabled={isSaving}
            style={{ fontSize: '0.875rem' }}
          >
            Reset to Defaults
          </button>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button className="btn btn-secondary" onClick={onClose} disabled={isSaving}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={handleSave} disabled={isSaving}>
              {isSaving ? (
                <>
                  <span className="spinner spinner-small"></span>
                  Saving...
                </>
              ) : (
                'Save Settings'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

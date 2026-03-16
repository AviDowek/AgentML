import { useState, useEffect } from 'react';
import type { DataSource, DatasetSpecCreate, ColumnInfo } from '../types/api';

interface CreateDatasetSpecFormProps {
  dataSources: DataSource[];
  onSubmit: (data: DatasetSpecCreate) => Promise<void>;
  onCancel: () => void;
  isLoading?: boolean;
}

export default function CreateDatasetSpecForm({
  dataSources,
  onSubmit,
  onCancel,
  isLoading = false,
}: CreateDatasetSpecFormProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedDataSourceId, setSelectedDataSourceId] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Get columns from selected data source
  const selectedDataSource = dataSources.find((ds) => ds.id === selectedDataSourceId);
  const availableColumns: ColumnInfo[] = selectedDataSource?.schema_summary?.columns || [];

  // Reset column selections when data source changes
  useEffect(() => {
    setTargetColumn('');
    setFeatureColumns([]);
  }, [selectedDataSourceId]);

  const handleFeatureToggle = (columnName: string) => {
    setFeatureColumns((prev) =>
      prev.includes(columnName)
        ? prev.filter((c) => c !== columnName)
        : [...prev, columnName]
    );
  };

  const handleSelectAllFeatures = () => {
    const allNonTarget = availableColumns
      .map((c) => c.name)
      .filter((name) => name !== targetColumn);
    setFeatureColumns(allNonTarget);
  };

  const handleClearFeatures = () => {
    setFeatureColumns([]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('Dataset name is required');
      return;
    }

    if (!selectedDataSourceId) {
      setError('Please select a data source');
      return;
    }

    if (!targetColumn) {
      setError('Please select a target column');
      return;
    }

    if (featureColumns.length === 0) {
      setError('Please select at least one feature column');
      return;
    }

    try {
      await onSubmit({
        name: name.trim(),
        description: description.trim() || null,
        data_sources_json: {
          primary: selectedDataSourceId,
        },
        target_column: targetColumn,
        feature_columns: featureColumns,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create dataset spec');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="form">
      {error && <div className="form-error">{error}</div>}

      <div className="form-group">
        <label htmlFor="dsName" className="form-label">
          Dataset Name <span className="required">*</span>
        </label>
        <input
          type="text"
          id="dsName"
          className="form-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="My Training Dataset"
          maxLength={255}
          disabled={isLoading}
          autoFocus
        />
      </div>

      <div className="form-group">
        <label htmlFor="dsDescription" className="form-label">
          Description
        </label>
        <textarea
          id="dsDescription"
          className="form-textarea"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe what this dataset is for..."
          rows={2}
          disabled={isLoading}
        />
      </div>

      <div className="form-group">
        <label htmlFor="dataSource" className="form-label">
          Data Source <span className="required">*</span>
        </label>
        <select
          id="dataSource"
          className="form-select"
          value={selectedDataSourceId}
          onChange={(e) => setSelectedDataSourceId(e.target.value)}
          disabled={isLoading || dataSources.length === 0}
        >
          <option value="">Select a data source</option>
          {dataSources.map((ds) => (
            <option key={ds.id} value={ds.id}>
              {ds.name} ({ds.schema_summary?.columns?.length || 0} columns, {ds.schema_summary?.row_count?.toLocaleString() || '?'} rows)
            </option>
          ))}
        </select>
        {dataSources.length === 0 && (
          <p className="form-hint">No data sources available. Upload a file first.</p>
        )}
      </div>

      {selectedDataSourceId && availableColumns.length > 0 && (
        <>
          <div className="form-group">
            <label htmlFor="targetColumn" className="form-label">
              Target Column <span className="required">*</span>
            </label>
            <select
              id="targetColumn"
              className="form-select"
              value={targetColumn}
              onChange={(e) => {
                setTargetColumn(e.target.value);
                // Remove from features if it was selected
                setFeatureColumns((prev) => prev.filter((c) => c !== e.target.value));
              }}
              disabled={isLoading}
            >
              <option value="">Select the column to predict</option>
              {availableColumns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.dtype})
                </option>
              ))}
            </select>
            <p className="form-hint">This is the column your model will learn to predict.</p>
          </div>

          <div className="form-group">
            <div className="form-label-row">
              <label className="form-label">
                Feature Columns <span className="required">*</span>
              </label>
              <div className="form-label-actions">
                <button
                  type="button"
                  className="btn-link"
                  onClick={handleSelectAllFeatures}
                  disabled={isLoading}
                >
                  Select All
                </button>
                <button
                  type="button"
                  className="btn-link"
                  onClick={handleClearFeatures}
                  disabled={isLoading}
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="column-selector">
              {availableColumns
                .filter((col) => col.name !== targetColumn)
                .map((col) => (
                  <label key={col.name} className="column-checkbox">
                    <input
                      type="checkbox"
                      checked={featureColumns.includes(col.name)}
                      onChange={() => handleFeatureToggle(col.name)}
                      disabled={isLoading}
                    />
                    <span className="column-checkbox-label">
                      {col.name}
                      <span className="column-type">({col.dtype})</span>
                    </span>
                  </label>
                ))}
            </div>
            <p className="form-hint">
              {featureColumns.length} of {availableColumns.length - (targetColumn ? 1 : 0)} columns selected
            </p>
          </div>
        </>
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
          {isLoading ? 'Creating...' : 'Create Dataset'}
        </button>
      </div>
    </form>
  );
}

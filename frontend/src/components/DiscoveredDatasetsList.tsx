/**
 * Discovered Datasets List Component
 * Displays datasets found by the AI discovery agent and allows selection.
 */

import { useState } from 'react';
import type { DiscoveredDataset } from '../types/api';
import LoadingSpinner from './LoadingSpinner';

interface DiscoveredDatasetsListProps {
  datasets: DiscoveredDataset[];
  onApply: (selectedIndices: number[]) => Promise<void>;
  onBack: () => void;
  onContinueWithoutDownload?: () => void;
  isApplying: boolean;
  error: string | null;
}

export default function DiscoveredDatasetsList({
  datasets,
  onApply,
  onBack,
  onContinueWithoutDownload,
  isApplying,
  error,
}: DiscoveredDatasetsListProps) {
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());

  const toggleSelection = (index: number) => {
    const newSelection = new Set(selectedIndices);
    if (newSelection.has(index)) {
      newSelection.delete(index);
    } else {
      newSelection.add(index);
    }
    setSelectedIndices(newSelection);
  };

  const selectAll = () => {
    setSelectedIndices(new Set(datasets.map((_, i) => i)));
  };

  const clearSelection = () => {
    setSelectedIndices(new Set());
  };

  const handleApply = async () => {
    if (selectedIndices.size === 0) return;
    await onApply(Array.from(selectedIndices).sort((a, b) => a - b));
  };

  return (
    <div className="discovered-datasets-list">
      <div className="list-header">
        <div className="header-info">
          <h3>Discovered Datasets</h3>
          <p>
            We found {datasets.length} relevant dataset{datasets.length !== 1 ? 's' : ''}.
            Select the ones you'd like to use for your project.
          </p>
        </div>

        <div className="selection-actions">
          <button
            type="button"
            className="btn btn-link"
            onClick={selectAll}
            disabled={isApplying}
          >
            Select All
          </button>
          <span className="action-divider">|</span>
          <button
            type="button"
            className="btn btn-link"
            onClick={clearSelection}
            disabled={isApplying}
          >
            Clear
          </button>
        </div>
      </div>

      <div className="datasets-grid">
        {datasets.map((dataset, index) => (
          <div
            key={index}
            className={`dataset-card ${selectedIndices.has(index) ? 'selected' : ''}`}
            onClick={() => !isApplying && toggleSelection(index)}
          >
            <div className="card-header">
              <input
                type="checkbox"
                checked={selectedIndices.has(index)}
                onChange={() => toggleSelection(index)}
                disabled={isApplying}
                onClick={(e) => e.stopPropagation()}
              />
              <h4 className="dataset-name">{dataset.name}</h4>
            </div>

            <p className="dataset-fit">{dataset.fit_for_purpose}</p>

            {dataset.schema_summary && (
              <div className="dataset-schema">
                {dataset.schema_summary.rows_estimate && (
                  <span className="schema-item">
                    ~{dataset.schema_summary.rows_estimate.toLocaleString()} rows
                  </span>
                )}
                {dataset.schema_summary.columns && (
                  <span className="schema-item">
                    {dataset.schema_summary.columns.length} columns
                  </span>
                )}
                {dataset.schema_summary.target_candidate && (
                  <span className="schema-item target">
                    Target: {dataset.schema_summary.target_candidate}
                  </span>
                )}
              </div>
            )}

            <div className="dataset-meta">
              {dataset.licensing && (
                <span className="meta-badge license">{dataset.licensing}</span>
              )}
              <a
                href={dataset.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="meta-link"
                onClick={(e) => e.stopPropagation()}
              >
                View Source ↗
              </a>
            </div>
          </div>
        ))}
      </div>

      {error && (
        <div className="download-error-container">
          <div className="form-error" style={{ margin: '0 0 12px 0' }}>
            {error}
          </div>
          {onContinueWithoutDownload && (
            <div className="download-error-actions">
              <p className="error-help-text">
                If automatic download failed, you can continue to your project and upload the dataset manually.
                The dataset URLs are saved and you can download them from the source links above.
              </p>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={onContinueWithoutDownload}
              >
                Continue Anyway →
              </button>
            </div>
          )}
        </div>
      )}

      {isApplying && (
        <div className="download-progress">
          <LoadingSpinner size="medium" />
          <div className="download-info">
            <h4>Downloading Datasets...</h4>
            <p>
              We're downloading {selectedIndices.size} dataset{selectedIndices.size !== 1 ? 's' : ''} and analyzing {selectedIndices.size !== 1 ? 'their' : 'its'} schema.
              This may take a few minutes depending on file sizes.
            </p>
            <ul className="download-steps">
              <li>Navigating to dataset pages</li>
              <li>Finding download links (using browser automation if needed)</li>
              <li>Downloading and verifying data files</li>
              <li>Analyzing schema and validating data</li>
            </ul>
          </div>
        </div>
      )}

      <div className="list-actions">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onBack}
          disabled={isApplying}
        >
          ← Back to Search
        </button>
        <button
          type="button"
          className="btn btn-primary"
          onClick={handleApply}
          disabled={isApplying || selectedIndices.size === 0}
        >
          {isApplying ? (
            <>
              <LoadingSpinner size="small" />
              <span style={{ marginLeft: '8px' }}>Downloading...</span>
            </>
          ) : (
            `Download & Use ${selectedIndices.size} Selected Dataset${selectedIndices.size !== 1 ? 's' : ''}`
          )}
        </button>
      </div>

      <style>{`
        .discovered-datasets-list {
          padding: 20px;
        }

        .list-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 20px;
          flex-wrap: wrap;
          gap: 12px;
        }

        .header-info h3 {
          margin: 0 0 4px 0;
          color: #333;
        }

        .header-info p {
          margin: 0;
          color: #666;
          font-size: 14px;
        }

        .selection-actions {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .action-divider {
          color: #ccc;
        }

        .btn-link {
          background: none;
          border: none;
          color: #1976d2;
          cursor: pointer;
          padding: 4px 8px;
          font-size: 14px;
        }

        .btn-link:hover:not(:disabled) {
          text-decoration: underline;
        }

        .btn-link:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .datasets-grid {
          display: grid;
          gap: 16px;
          max-height: 400px;
          overflow-y: auto;
          padding: 4px;
        }

        .dataset-card {
          border: 2px solid #e0e0e0;
          border-radius: 8px;
          padding: 16px;
          cursor: pointer;
          transition: all 0.2s ease;
          background-color: #fff;
        }

        .dataset-card:hover {
          border-color: #1976d2;
          box-shadow: 0 2px 8px rgba(25, 118, 210, 0.1);
        }

        .dataset-card.selected {
          border-color: #1976d2;
          background-color: #f0f7ff;
        }

        .card-header {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          margin-bottom: 8px;
        }

        .card-header input[type="checkbox"] {
          margin-top: 4px;
          flex-shrink: 0;
        }

        .dataset-name {
          margin: 0;
          font-size: 16px;
          color: #333;
          line-height: 1.4;
        }

        .dataset-fit {
          margin: 0 0 12px 0;
          color: #555;
          font-size: 14px;
          line-height: 1.5;
        }

        .dataset-schema {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 12px;
        }

        .schema-item {
          background-color: #f5f5f5;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          color: #666;
        }

        .schema-item.target {
          background-color: #e8f5e9;
          color: #2e7d32;
        }

        .dataset-meta {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 8px;
        }

        .meta-badge {
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 500;
          text-transform: uppercase;
        }

        .meta-badge.license {
          background-color: #e3f2fd;
          color: #1565c0;
        }

        .meta-link {
          font-size: 13px;
          color: #1976d2;
          text-decoration: none;
        }

        .meta-link:hover {
          text-decoration: underline;
        }

        .list-actions {
          display: flex;
          justify-content: space-between;
          gap: 16px;
          margin-top: 24px;
          padding-top: 16px;
          border-top: 1px solid #e0e0e0;
        }

        .list-actions .btn {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .download-progress {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 20px;
          background-color: #e3f2fd;
          border-radius: 8px;
          margin: 16px 0;
        }

        .download-info h4 {
          margin: 0 0 4px 0;
          color: #1565c0;
        }

        .download-info p {
          margin: 0;
          color: #1976d2;
          font-size: 14px;
        }

        .download-steps {
          margin: 12px 0 0 0;
          padding-left: 20px;
          color: #1565c0;
          font-size: 13px;
        }

        .download-steps li {
          margin: 4px 0;
          opacity: 0.8;
        }

        .download-error-container {
          background-color: #fff3f3;
          border: 1px solid #ffcdd2;
          border-radius: 8px;
          padding: 16px;
          margin: 16px 0;
        }

        .download-error-actions {
          margin-top: 8px;
        }

        .error-help-text {
          margin: 0 0 12px 0;
          font-size: 14px;
          color: #666;
          line-height: 1.5;
        }

        .download-error-actions .btn {
          margin-top: 4px;
        }
      `}</style>
    </div>
  );
}

/**
 * Dataset Spec Detail Modal
 * Shows full details of a dataset specification when clicked, including data viewer
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import type { DatasetSpec, DataSource } from '../types/api';
import { getDatasetSpecData, createExperimentsForDataset, createExperimentsFromStoredConfig, downloadDatasetSpec, type DataPreviewResponse } from '../services/api';
import Modal from './Modal';
import LoadingSpinner from './LoadingSpinner';

interface DatasetSpecDetailModalProps {
  datasetSpec: DatasetSpec | null;
  dataSources: DataSource[];
  isOpen: boolean;
  onClose: () => void;
  /** Project ID for creating experiments */
  projectId?: string;
  /** Experiment design step ID for creating experiments (from agent pipeline) */
  experimentDesignStepId?: string;
  /** Callback when experiments are created */
  onExperimentsCreated?: () => void;
}

type TabType = 'schema' | 'data';

export default function DatasetSpecDetailModal({
  datasetSpec,
  dataSources,
  isOpen,
  onClose,
  projectId,
  experimentDesignStepId,
  onExperimentsCreated,
}: DatasetSpecDetailModalProps) {
  const [activeTab, setActiveTab] = useState<TabType>('schema');
  const [dataPreview, setDataPreview] = useState<DataPreviewResponse | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [dataError, setDataError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [isCreatingExperiments, setIsCreatingExperiments] = useState(false);
  const [experimentCreationMessage, setExperimentCreationMessage] = useState<string | null>(null);
  const [autoRunExperiments, setAutoRunExperiments] = useState(true);  // Toggle for auto-running experiments
  const [isDownloading, setIsDownloading] = useState(false);
  const pageSize = 100;
  const tableContainerRef = useRef<HTMLDivElement>(null);

  // Reset state when modal opens/closes or dataset spec changes
  useEffect(() => {
    if (!isOpen) {
      setActiveTab('schema');
      setDataPreview(null);
      setCurrentPage(1);
      setDataError(null);
      setExperimentCreationMessage(null);
    }
  }, [isOpen, datasetSpec?.id]);

  // Check if we can create experiments - either from current agent step or stored config
  const hasStoredConfig = datasetSpec?.agent_experiment_design_json?.variants?.length;
  const canCreateExperiments = projectId && datasetSpec && (experimentDesignStepId || hasStoredConfig);

  // Handle creating experiments for this dataset
  const handleCreateExperiments = async () => {
    if (!projectId || !datasetSpec) return;

    setIsCreatingExperiments(true);
    setExperimentCreationMessage(null);

    try {
      let response;

      if (experimentDesignStepId) {
        // Use the current agent step
        response = await createExperimentsForDataset(
          projectId,
          experimentDesignStepId,
          datasetSpec.id,
          { create_all_variants: true, run_immediately: autoRunExperiments }
        );
      } else if (hasStoredConfig) {
        // Use the stored agent experiment design config
        response = await createExperimentsFromStoredConfig(
          projectId,
          datasetSpec.id,
          { create_all_variants: true, run_immediately: autoRunExperiments }
        );
      } else {
        throw new Error('No experiment design configuration available');
      }

      const queuedMsg = autoRunExperiments
        ? `${response.queued_count} experiment(s) queued for execution.`
        : 'Experiments created (not auto-running).';
      setExperimentCreationMessage(`✓ ${response.message}. ${queuedMsg}`);
      onExperimentsCreated?.();
    } catch (err) {
      setExperimentCreationMessage(
        `✗ Failed to create experiments: ${err instanceof Error ? err.message : 'Unknown error'}`
      );
    } finally {
      setIsCreatingExperiments(false);
    }
  };

  // Handle downloading the built dataset as CSV
  const handleDownload = async () => {
    if (!datasetSpec) return;

    setIsDownloading(true);
    try {
      await downloadDatasetSpec(datasetSpec.id);
    } catch (err) {
      setDataError(`Download failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsDownloading(false);
    }
  };

  // Load data when switching to data tab
  const loadData = useCallback(async (page: number) => {
    if (!datasetSpec) return;

    setIsLoadingData(true);
    setDataError(null);

    try {
      const data = await getDatasetSpecData(datasetSpec.id, page, pageSize);
      setDataPreview(data);
      setCurrentPage(page);
    } catch (err) {
      setDataError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoadingData(false);
    }
  }, [datasetSpec]);

  // Load data when switching to data tab
  useEffect(() => {
    if (activeTab === 'data' && !dataPreview && !isLoadingData && datasetSpec) {
      loadData(1);
    }
  }, [activeTab, dataPreview, isLoadingData, datasetSpec, loadData]);

  if (!datasetSpec) return null;

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Get linked data source details
  const getLinkedSources = (): DataSource[] => {
    const config = datasetSpec.data_sources_json;
    if (!config) return [];

    let sourceIds: string[] = [];

    if (Array.isArray(config)) {
      sourceIds = config.map((item) =>
        typeof item === 'string' ? item : (item as Record<string, unknown>).source_id || (item as Record<string, unknown>).id || ''
      ).filter(Boolean) as string[];
    } else {
      // Dict format
      const sources = (config as Record<string, unknown>).sources ||
        (config as Record<string, unknown>).source_ids ||
        [];
      if (Array.isArray(sources)) {
        sourceIds = sources.map((item) =>
          typeof item === 'string' ? item : (item as Record<string, unknown>).source_id || (item as Record<string, unknown>).id || ''
        ).filter(Boolean) as string[];
      }
      if (sourceIds.length === 0) {
        const primary = (config as Record<string, unknown>).primary;
        const sourceId = (config as Record<string, unknown>).source_id;
        if (primary) sourceIds = [primary as string];
        else if (sourceId) sourceIds = [sourceId as string];
      }
    }

    return sourceIds
      .map((id) => dataSources.find((ds) => ds.id === id))
      .filter((ds): ds is DataSource => ds !== undefined);
  };

  const linkedSources = getLinkedSources();
  const featureColumns = datasetSpec.feature_columns || [];
  const filters = datasetSpec.filters_json;

  const formatCellValue = (value: string | number | boolean | null): string => {
    if (value === null || value === undefined) return '';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    return String(value);
  };

  const handlePageChange = (newPage: number) => {
    loadData(newPage);
    // Scroll table back to top
    if (tableContainerRef.current) {
      tableContainerRef.current.scrollTop = 0;
    }
  };

  const totalPages = dataPreview ? Math.ceil(dataPreview.total_rows / pageSize) : 0;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={datasetSpec.name} size="xlarge">
      <div className="dataset-spec-detail">
        {/* Header */}
        <div className="detail-header">
          <div className="header-left">
            <span className="type-badge">Dataset Specification</span>
            <span className="detail-date">Created {formatDate(datasetSpec.created_at)}</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              className="download-btn"
              onClick={handleDownload}
              disabled={isDownloading}
              title="Download the built dataset as CSV"
            >
              {isDownloading ? '⏳ Downloading...' : '⬇ Download CSV'}
            </button>
            <Link
              to={`/dataset-results/${datasetSpec.id}`}
              className="view-results-btn"
              onClick={onClose}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.375rem',
                padding: '0.5rem 1rem',
                background: '#f1f5f9',
                color: '#475569',
                borderRadius: '6px',
                textDecoration: 'none',
                fontWeight: 500,
                fontSize: '0.875rem',
                border: '1px solid #e2e8f0',
              }}
            >
              <span style={{ fontSize: '1rem' }}>📊</span>
              View All Results
            </Link>
            <button
              className="create-experiments-btn"
              onClick={handleCreateExperiments}
              disabled={isCreatingExperiments || !canCreateExperiments}
              title={
                !canCreateExperiments
                  ? 'Run the AI Agent pipeline first to get experiment recommendations'
                  : hasStoredConfig && !experimentDesignStepId
                    ? 'Create experiments using saved AI recommendations'
                    : 'Create experiments based on AI recommendations'
              }
            >
              {isCreatingExperiments
                ? 'Creating...'
                : autoRunExperiments
                  ? '🧪 Run Experiments'
                  : '🧪 Create (No Auto-Run)'}
            </button>
          </div>
          {/* Auto-run toggle */}
          {canCreateExperiments && (
            <div className="auto-run-toggle-inline">
              <label className="toggle-label-inline">
                <input
                  type="checkbox"
                  checked={autoRunExperiments}
                  onChange={(e) => setAutoRunExperiments(e.target.checked)}
                  disabled={isCreatingExperiments}
                />
                <span className="toggle-switch-small"></span>
                <span className="toggle-text-small">
                  {autoRunExperiments ? 'Auto-run ON' : 'Auto-run OFF'}
                </span>
              </label>
            </div>
          )}
        </div>

        {/* Experiment creation feedback */}
        {experimentCreationMessage && (
          <div className={`experiment-message ${experimentCreationMessage.startsWith('✓') ? 'success' : 'error'}`}>
            {experimentCreationMessage}
          </div>
        )}

        {/* Tabs */}
        <div className="detail-tabs">
          <button
            className={`tab-button ${activeTab === 'schema' ? 'active' : ''}`}
            onClick={() => setActiveTab('schema')}
          >
            Schema
          </button>
          <button
            className={`tab-button ${activeTab === 'data' ? 'active' : ''}`}
            onClick={() => setActiveTab('data')}
          >
            View Data
          </button>
        </div>

        {/* Schema Tab */}
        {activeTab === 'schema' && (
          <div className="tab-content">
            {/* Description */}
            {datasetSpec.description && (
              <div className="detail-section">
                <h4>Description</h4>
                <p className="detail-description">{datasetSpec.description}</p>
              </div>
            )}

            {/* Target Column */}
            <div className="detail-section">
              <h4>Target Column</h4>
              <div className="target-display">
                {datasetSpec.target_column ? (
                  <span className="target-column">{datasetSpec.target_column}</span>
                ) : (
                  <span className="not-set">Not configured</span>
                )}
              </div>
            </div>

            {/* Time-Based Task Info */}
            {datasetSpec.is_time_based && (
              <div className="detail-section time-based-section">
                <h4>⏱️ Time-Based Task</h4>
                <div className="time-based-info">
                  <div className="time-info-row">
                    <span className="info-label">Time Column:</span>
                    <span className="info-value">{datasetSpec.time_column || 'Not specified'}</span>
                  </div>
                  {datasetSpec.prediction_horizon && (
                    <div className="time-info-row">
                      <span className="info-label">Prediction Horizon:</span>
                      <span className="info-value horizon-badge">{datasetSpec.prediction_horizon}</span>
                    </div>
                  )}
                  {datasetSpec.entity_id_column && (
                    <div className="time-info-row">
                      <span className="info-label">Entity ID Column:</span>
                      <span className="info-value">{datasetSpec.entity_id_column}</span>
                    </div>
                  )}
                  {datasetSpec.target_positive_class && (
                    <div className="time-info-row">
                      <span className="info-label">Positive Class:</span>
                      <span className="info-value positive-class-badge">{datasetSpec.target_positive_class}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Warning if time-based but no time column */}
            {datasetSpec.is_time_based && !datasetSpec.time_column && (
              <div className="time-warning">
                ⚠️ Time-based task detected but no time column specified. This may affect validation strategy.
              </div>
            )}

            {/* Feature Columns */}
            <div className="detail-section">
              <h4>Feature Columns ({featureColumns.length})</h4>
              {featureColumns.length > 0 ? (
                <div className="features-grid">
                  {featureColumns.map((col, idx) => (
                    <span key={idx} className="feature-item">
                      {col}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="not-set">No feature columns selected</span>
              )}
            </div>

            {/* Linked Data Sources */}
            <div className="detail-section">
              <h4>Data Sources ({linkedSources.length})</h4>
              {linkedSources.length > 0 ? (
                <div className="sources-list">
                  {linkedSources.map((source) => (
                    <div key={source.id} className="source-item">
                      <div className="source-header">
                        <span className="source-icon">
                          {source.type === 'external_dataset' ? '🌐' : '📊'}
                        </span>
                        <span className="source-name">{source.name}</span>
                        <span className="source-type">{source.type}</span>
                      </div>
                      {source.schema_summary && (
                        <div className="source-stats">
                          <span>{source.schema_summary.row_count?.toLocaleString() || 'N/A'} rows</span>
                          <span>{source.schema_summary.columns?.length || 0} columns</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <span className="not-set">No data sources linked</span>
              )}
            </div>

            {/* Filters */}
            {filters && Object.keys(filters).length > 0 && (
              <div className="detail-section">
                <h4>Filters</h4>
                <div className="filters-list">
                  {Object.entries(filters).map(([column, config]) => (
                    <div key={column} className="filter-item">
                      <span className="filter-column">{column}</span>
                      <span className="filter-config">
                        {typeof config === 'object' ? JSON.stringify(config) : String(config)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Additional Spec Config */}
            {datasetSpec.spec_json && Object.keys(datasetSpec.spec_json).length > 0 && (
              <div className="detail-section">
                <h4>Additional Configuration</h4>
                <pre className="config-json">
                  {JSON.stringify(datasetSpec.spec_json, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Data Tab */}
        {activeTab === 'data' && (
          <div className="tab-content data-tab">
            {isLoadingData && (
              <div className="data-loading">
                <LoadingSpinner message="Building and loading dataset..." />
              </div>
            )}

            {dataError && (
              <div className="data-error">
                <p>{dataError}</p>
                <button className="btn btn-secondary btn-sm" onClick={() => loadData(1)}>
                  Retry
                </button>
              </div>
            )}

            {dataPreview && !isLoadingData && (
              <>
                {/* Data stats bar */}
                <div className="data-stats-bar">
                  <span className="data-stat">
                    <strong>{dataPreview.total_rows.toLocaleString()}</strong> rows
                  </span>
                  <span className="data-stat">
                    <strong>{dataPreview.total_columns}</strong> columns
                  </span>
                  <span className="data-stat">
                    Showing rows {((currentPage - 1) * pageSize + 1).toLocaleString()} - {Math.min(currentPage * pageSize, dataPreview.total_rows).toLocaleString()}
                  </span>
                </div>

                {/* Scrollable data table */}
                <div className="data-table-container" ref={tableContainerRef}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th className="row-number-header">#</th>
                        {dataPreview.columns.map((col, idx) => (
                          <th key={idx} className={col === datasetSpec.target_column ? 'target-header' : ''}>
                            {col}
                            {col === datasetSpec.target_column && <span className="target-indicator"> (target)</span>}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {dataPreview.rows.map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          <td className="row-number">
                            {((currentPage - 1) * pageSize + rowIdx + 1).toLocaleString()}
                          </td>
                          {row.map((cell, cellIdx) => (
                            <td
                              key={cellIdx}
                              title={formatCellValue(cell)}
                              className={dataPreview.columns[cellIdx] === datasetSpec.target_column ? 'target-cell' : ''}
                            >
                              {formatCellValue(cell)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="data-pagination">
                    <button
                      className="btn btn-sm"
                      onClick={() => handlePageChange(1)}
                      disabled={currentPage === 1 || isLoadingData}
                    >
                      First
                    </button>
                    <button
                      className="btn btn-sm"
                      onClick={() => handlePageChange(currentPage - 1)}
                      disabled={currentPage === 1 || isLoadingData}
                    >
                      Previous
                    </button>
                    <span className="page-info">
                      Page {currentPage} of {totalPages.toLocaleString()}
                    </span>
                    <button
                      className="btn btn-sm"
                      onClick={() => handlePageChange(currentPage + 1)}
                      disabled={currentPage === totalPages || isLoadingData}
                    >
                      Next
                    </button>
                    <button
                      className="btn btn-sm"
                      onClick={() => handlePageChange(totalPages)}
                      disabled={currentPage === totalPages || isLoadingData}
                    >
                      Last
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        <style>{`
          .dataset-spec-detail {
            padding: 0 4px;
            display: flex;
            flex-direction: column;
            height: 100%;
            min-height: 500px;
          }

          .detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #404040;
          }

          .header-left {
            display: flex;
            align-items: center;
            gap: 12px;
          }

          .type-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            background-color: #6b21a8;
            color: #ffffff;
          }

          .detail-date {
            color: #b0b0b0;
            font-size: 13px;
          }

          .download-btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: #ffffff;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
          }

          .download-btn:hover:not(:disabled) {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            transform: translateY(-1px);
          }

          .download-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
          }

          .create-experiments-btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            color: #ffffff;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
          }

          .create-experiments-btn:hover:not(:disabled) {
            background: linear-gradient(135deg, #047857 0%, #065f46 100%);
            transform: translateY(-1px);
          }

          .create-experiments-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
          }

          /* Auto-run toggle inline */
          .auto-run-toggle-inline {
            margin-top: 8px;
          }

          .toggle-label-inline {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            user-select: none;
            font-size: 12px;
          }

          .toggle-label-inline input[type="checkbox"] {
            position: absolute;
            opacity: 0;
            width: 0;
            height: 0;
          }

          .toggle-switch-small {
            position: relative;
            width: 32px;
            height: 18px;
            background: #333;
            border-radius: 9px;
            transition: all 0.2s ease;
            flex-shrink: 0;
          }

          .toggle-switch-small::after {
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 14px;
            height: 14px;
            background: #666;
            border-radius: 50%;
            transition: all 0.2s ease;
          }

          .toggle-label-inline input:checked + .toggle-switch-small {
            background: #059669;
          }

          .toggle-label-inline input:checked + .toggle-switch-small::after {
            left: 16px;
            background: white;
          }

          .toggle-label-inline input:disabled + .toggle-switch-small {
            opacity: 0.5;
          }

          .toggle-text-small {
            color: #888;
          }

          .toggle-label-inline input:checked ~ .toggle-text-small {
            color: #10b981;
          }

          .experiment-message {
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 12px;
            font-size: 13px;
          }

          .experiment-message.success {
            background-color: #065f46;
            color: #a7f3d0;
            border: 1px solid #10b981;
          }

          .experiment-message.error {
            background-color: #7f1d1d;
            color: #fca5a5;
            border: 1px solid #ef4444;
          }

          /* Tabs */
          .detail-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            border-bottom: 2px solid #404040;
            padding-bottom: 0;
          }

          .tab-button {
            padding: 8px 16px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #b0b0b0;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s ease;
          }

          .tab-button:hover:not(:disabled) {
            color: #a855f7;
          }

          .tab-button.active {
            color: #a855f7;
            border-bottom-color: #a855f7;
          }

          .tab-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
          }

          /* Tab Content */
          .tab-content {
            flex: 1;
            overflow: auto;
          }

          .tab-content.data-tab {
            display: flex;
            flex-direction: column;
          }

          .detail-section {
            margin-bottom: 24px;
          }

          .detail-section h4 {
            margin: 0 0 12px 0;
            color: #e0e0e0;
            font-size: 14px;
            font-weight: 600;
          }

          .detail-description {
            margin: 0;
            color: #c0c0c0;
            line-height: 1.5;
          }

          .target-display {
            display: flex;
            align-items: center;
          }

          .target-column {
            display: inline-block;
            background-color: #166534;
            color: #ffffff;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
          }

          .not-set {
            color: #808080;
            font-style: italic;
          }

          .features-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
          }

          .feature-item {
            display: inline-block;
            background-color: #374151;
            color: #e0e0e0;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 13px;
            font-family: 'Consolas', 'Monaco', monospace;
          }

          .sources-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
          }

          .source-item {
            background-color: #1f2937;
            border: 1px solid #404040;
            border-radius: 6px;
            padding: 12px;
          }

          .source-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
          }

          .source-icon {
            font-size: 16px;
          }

          .source-name {
            flex: 1;
            font-weight: 500;
            color: #e0e0e0;
          }

          .source-type {
            background-color: #1e40af;
            color: #ffffff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            text-transform: uppercase;
          }

          .source-stats {
            display: flex;
            gap: 16px;
            color: #a0a0a0;
            font-size: 12px;
          }

          .filters-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
          }

          .filter-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 12px;
            background-color: #78350f;
            border-radius: 4px;
          }

          .filter-column {
            font-weight: 500;
            color: #fbbf24;
          }

          .filter-config {
            color: #e0e0e0;
            font-family: monospace;
            font-size: 12px;
          }

          .config-json {
            background-color: #1f2937;
            color: #e0e0e0;
            padding: 12px;
            border-radius: 6px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            overflow-x: auto;
            margin: 0;
          }

          /* Time-Based Task Styles */
          .time-based-section {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
            border: 1px solid rgba(147, 51, 234, 0.3);
            border-radius: 8px;
            padding: 16px;
          }

          .time-based-section h4 {
            color: #a78bfa;
            display: flex;
            align-items: center;
            gap: 6px;
          }

          .time-based-info {
            display: flex;
            flex-direction: column;
            gap: 10px;
          }

          .time-info-row {
            display: flex;
            align-items: center;
            gap: 12px;
          }

          .info-label {
            color: #a0a0a0;
            font-size: 13px;
            min-width: 140px;
          }

          .info-value {
            color: #e0e0e0;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
          }

          .horizon-badge {
            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 12px;
          }

          .positive-class-badge {
            background-color: #166534;
            color: #a7f3d0;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 500;
            font-size: 12px;
          }

          .time-warning {
            background-color: rgba(234, 179, 8, 0.15);
            border: 1px solid rgba(234, 179, 8, 0.4);
            color: #fbbf24;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 13px;
            margin-bottom: 16px;
          }

          /* Data Tab Styles */
          .data-loading {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 200px;
          }

          .data-error {
            background-color: #7f1d1d;
            color: #fca5a5;
            padding: 16px;
            border-radius: 4px;
            text-align: center;
          }

          .data-error p {
            margin: 0 0 12px 0;
          }

          .data-stats-bar {
            display: flex;
            gap: 24px;
            padding: 8px 12px;
            background-color: #1f2937;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #a0a0a0;
          }

          .data-stat strong {
            color: #e0e0e0;
          }

          .data-table-container {
            flex: 1;
            overflow: auto;
            border: 1px solid #404040;
            border-radius: 4px;
            max-height: 400px;
            min-height: 200px;
          }

          .data-table {
            width: max-content;
            min-width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
          }

          .data-table th {
            position: sticky;
            top: 0;
            background-color: #374151;
            color: #e0e0e0;
            padding: 8px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #4b5563;
            white-space: nowrap;
            z-index: 1;
          }

          .data-table th.target-header {
            background-color: #166534;
            color: #ffffff;
          }

          .target-indicator {
            font-size: 10px;
            opacity: 0.8;
          }

          .data-table td {
            padding: 6px 12px;
            border-bottom: 1px solid #374151;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #d0d0d0;
          }

          .data-table td.target-cell {
            background-color: rgba(22, 101, 52, 0.2);
          }

          .data-table tr:hover td {
            background-color: #1f2937;
          }

          .data-table tr:hover td.target-cell {
            background-color: rgba(22, 101, 52, 0.3);
          }

          .row-number-header,
          .row-number {
            background-color: #1f2937;
            color: #808080;
            font-weight: normal;
            text-align: right;
            padding-right: 16px;
            border-right: 1px solid #404040;
            position: sticky;
            left: 0;
            z-index: 1;
          }

          .row-number-header {
            background-color: #374151;
            z-index: 2;
          }

          .data-table tr:hover .row-number {
            background-color: #374151;
          }

          /* Pagination */
          .data-pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            padding: 12px 0;
            border-top: 1px solid #404040;
            margin-top: 12px;
          }

          .btn-sm {
            padding: 4px 12px;
            font-size: 12px;
            background-color: #374151;
            color: #e0e0e0;
            border: 1px solid #4b5563;
            border-radius: 4px;
            cursor: pointer;
          }

          .btn-sm:hover:not(:disabled) {
            background-color: #4b5563;
          }

          .btn-sm:disabled {
            opacity: 0.5;
            cursor: not-allowed;
          }

          .page-info {
            margin: 0 12px;
            font-size: 13px;
            color: #a0a0a0;
          }
        `}</style>
      </div>
    </Modal>
  );
}

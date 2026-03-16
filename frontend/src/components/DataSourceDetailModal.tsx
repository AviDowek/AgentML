/**
 * Data Source Detail Modal
 * Shows full details of a data source when clicked, including schema and full data viewer
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { DataSource } from '../types/api';
import { getDataSourceData, type DataPreviewResponse } from '../services/api';
import Modal from './Modal';
import LoadingSpinner from './LoadingSpinner';

interface DataSourceDetailModalProps {
  dataSource: DataSource | null;
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'schema' | 'data';

export default function DataSourceDetailModal({
  dataSource,
  isOpen,
  onClose,
}: DataSourceDetailModalProps) {
  const [activeTab, setActiveTab] = useState<TabType>('schema');
  const [dataPreview, setDataPreview] = useState<DataPreviewResponse | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [dataError, setDataError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 100;
  const tableContainerRef = useRef<HTMLDivElement>(null);

  // Reset state when modal opens/closes or data source changes
  useEffect(() => {
    if (!isOpen) {
      setActiveTab('schema');
      setDataPreview(null);
      setCurrentPage(1);
      setDataError(null);
    }
  }, [isOpen, dataSource?.id]);

  // Load data when switching to data tab
  const loadData = useCallback(async (page: number) => {
    if (!dataSource) return;

    // External datasets don't have file data
    if (dataSource.type === 'external_dataset') {
      setDataError('External datasets must be downloaded first to view data.');
      return;
    }

    setIsLoadingData(true);
    setDataError(null);

    try {
      const data = await getDataSourceData(dataSource.id, page, pageSize);
      setDataPreview(data);
      setCurrentPage(page);
    } catch (err) {
      setDataError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoadingData(false);
    }
  }, [dataSource]);

  // Load data when switching to data tab
  useEffect(() => {
    if (activeTab === 'data' && !dataPreview && !isLoadingData && dataSource) {
      loadData(1);
    }
  }, [activeTab, dataPreview, isLoadingData, dataSource, loadData]);

  if (!dataSource) return null;

  const schema = dataSource.schema_summary;
  const config = dataSource.config_json;
  const isExternalDataset = dataSource.type === 'external_dataset';

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getTypeLabel = () => {
    switch (dataSource.type) {
      case 'file_upload':
        return 'Uploaded File';
      case 'external_dataset':
        return 'External Dataset';
      case 'database':
        return 'Database Connection';
      case 'api':
        return 'API Connection';
      case 's3':
        return 'S3 Storage';
      default:
        return dataSource.type;
    }
  };

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
    <Modal isOpen={isOpen} onClose={onClose} title={dataSource.name} size="xlarge">
      <div className="data-source-detail">
        {/* Header with type badge */}
        <div className="detail-header">
          <span className={`type-badge ${isExternalDataset ? 'external' : 'uploaded'}`}>
            {getTypeLabel()}
          </span>
          <span className="detail-date">Added {formatDate(dataSource.created_at)}</span>
        </div>

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
            disabled={isExternalDataset}
            title={isExternalDataset ? 'External datasets must be downloaded to view data' : ''}
          >
            View Data
          </button>
        </div>

        {/* Schema Tab */}
        {activeTab === 'schema' && (
          <div className="tab-content">
            {/* Description for external datasets */}
            {isExternalDataset && config?.fit_for_purpose ? (
              <div className="detail-section">
                <h4>Description</h4>
                <p className="detail-description">{String(config.fit_for_purpose)}</p>
              </div>
            ) : null}

            {/* Source URL for external datasets */}
            {isExternalDataset && config?.source_url ? (
              <div className="detail-section">
                <h4>Source URL</h4>
                <a
                  href={String(config.source_url)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="source-url"
                >
                  {String(config.source_url)}
                </a>
                <p className="source-hint">
                  Click the link above to download or access the dataset from its original source.
                </p>
              </div>
            ) : null}

            {/* Licensing info */}
            {isExternalDataset && config?.licensing ? (
              <div className="detail-section">
                <h4>License</h4>
                <p>{String(config.licensing)}</p>
              </div>
            ) : null}

            {/* Schema summary */}
            {schema && (
              <div className="detail-section">
                <h4>Schema Summary</h4>
                <div className="schema-stats">
                  <div className="stat-item">
                    <span className="stat-value">{schema.row_count?.toLocaleString() || 'N/A'}</span>
                    <span className="stat-label">
                      Rows {isExternalDataset && schema.row_count && '(estimate)'}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{schema.columns?.length || 0}</span>
                    <span className="stat-label">Columns</span>
                  </div>
                  {schema.file_type && (
                    <div className="stat-item">
                      <span className="stat-value">{schema.file_type.toUpperCase()}</span>
                      <span className="stat-label">Format</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Warnings */}
            {schema?.warnings && schema.warnings.length > 0 && (
              <div className="detail-section">
                <div className="schema-warnings">
                  {schema.warnings.map((warning, idx) => {
                    // Handle both string warnings and object warnings with {issue, severity, recommendation}
                    const warningText = typeof warning === 'string'
                      ? warning
                      : (warning as { issue?: string; severity?: string; recommendation?: string }).issue || JSON.stringify(warning);
                    return (
                      <div key={idx} className="warning-item">
                        <span className="warning-icon">⚠️</span>
                        <span className="warning-text">{warningText}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Target candidate for external datasets */}
            {schema && (schema as unknown as Record<string, unknown>).target_candidate ? (
              <div className="detail-section">
                <h4>Suggested Target Column</h4>
                <span className="target-column">
                  {String((schema as unknown as Record<string, unknown>).target_candidate)}
                </span>
              </div>
            ) : null}

            {/* Columns list */}
            {schema?.columns && schema.columns.length > 0 && (
              <div className="detail-section">
                <h4>Columns ({schema.columns.length})</h4>
                <div className="columns-table-container">
                  <table className="columns-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Non-null</th>
                        <th>Unique</th>
                        <th>Sample Values</th>
                      </tr>
                    </thead>
                    <tbody>
                      {schema.columns.map((col) => (
                        <tr key={col.name}>
                          <td className="col-name">{col.name}</td>
                          <td className="col-type">{col.dtype || col.inferred_type || 'unknown'}</td>
                          <td className="col-count">{col.non_null_count?.toLocaleString() || '-'}</td>
                          <td className="col-count">{col.unique_count?.toLocaleString() || '-'}</td>
                          <td className="col-samples">
                            {col.sample_values?.slice(0, 3).map((v, i) => (
                              <span key={i} className="sample-value">
                                {String(v)}
                              </span>
                            ))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* File info for uploaded files */}
            {!isExternalDataset && config ? (
              <div className="detail-section">
                <h4>File Information</h4>
                <div className="info-grid">
                  {config.file_type ? (
                    <div className="info-item">
                      <span className="info-label">File Type:</span>
                      <span className="info-value">{String(config.file_type)}</span>
                    </div>
                  ) : null}
                  {config.original_filename ? (
                    <div className="info-item">
                      <span className="info-label">Original Filename:</span>
                      <span className="info-value">{String(config.original_filename)}</span>
                    </div>
                  ) : null}
                  {config.file_size_bytes ? (
                    <div className="info-item">
                      <span className="info-label">File Size:</span>
                      <span className="info-value">
                        {(Number(config.file_size_bytes) / 1024 / 1024).toFixed(2)} MB
                      </span>
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}
          </div>
        )}

        {/* Data Tab */}
        {activeTab === 'data' && (
          <div className="tab-content data-tab">
            {isLoadingData && (
              <div className="data-loading">
                <LoadingSpinner message="Loading data..." />
              </div>
            )}

            {dataError && (
              <div className="data-error">
                <p>{dataError}</p>
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
                          <th key={idx}>{col}</th>
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
                            <td key={cellIdx} title={formatCellValue(cell)}>
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
          .data-source-detail {
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
            border-bottom: 1px solid #e0e0e0;
          }

          .type-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
          }

          .type-badge.external {
            background-color: #e3f2fd;
            color: #1565c0;
          }

          .type-badge.uploaded {
            background-color: #e8f5e9;
            color: #2e7d32;
          }

          .detail-date {
            color: #666;
            font-size: 13px;
          }

          /* Tabs */
          .detail-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0;
          }

          .tab-button {
            padding: 8px 16px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #666;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s ease;
          }

          .tab-button:hover:not(:disabled) {
            color: #1976d2;
          }

          .tab-button.active {
            color: #1976d2;
            border-bottom-color: #1976d2;
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
            margin: 0 0 8px 0;
            color: #333;
            font-size: 14px;
            font-weight: 600;
          }

          .detail-description {
            margin: 0;
            color: #555;
            line-height: 1.5;
          }

          .source-url {
            display: block;
            color: #1976d2;
            word-break: break-all;
            margin-bottom: 8px;
          }

          .source-hint {
            margin: 0;
            color: #666;
            font-size: 13px;
            font-style: italic;
          }

          .schema-stats {
            display: flex;
            gap: 32px;
          }

          .stat-item {
            text-align: center;
          }

          .stat-value {
            display: block;
            font-size: 24px;
            font-weight: 600;
            color: #1976d2;
          }

          .stat-label {
            display: block;
            font-size: 12px;
            color: #666;
            margin-top: 4px;
          }

          /* Warnings */
          .schema-warnings {
            background-color: #fff8e1;
            border: 1px solid #ffcc02;
            border-radius: 6px;
            padding: 12px;
          }

          .warning-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            margin-bottom: 8px;
          }

          .warning-item:last-child {
            margin-bottom: 0;
          }

          .warning-icon {
            flex-shrink: 0;
          }

          .warning-text {
            font-size: 13px;
            color: #8a6914;
            line-height: 1.4;
          }

          .target-column {
            display: inline-block;
            background-color: #e8f5e9;
            color: #2e7d32;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: 500;
          }

          .columns-table-container {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
          }

          .columns-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
          }

          .columns-table th {
            position: sticky;
            top: 0;
            background-color: #f5f5f5;
            padding: 8px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 1px solid #e0e0e0;
          }

          .columns-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #f0f0f0;
          }

          .columns-table tr:last-child td {
            border-bottom: none;
          }

          .col-name {
            font-weight: 500;
            color: #333;
          }

          .col-type {
            color: #666;
            font-family: monospace;
            font-size: 12px;
          }

          .col-count {
            color: #888;
            text-align: right;
          }

          .col-samples {
            max-width: 200px;
          }

          .sample-value {
            display: inline-block;
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            margin-right: 4px;
            margin-bottom: 2px;
            font-size: 11px;
            max-width: 60px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }

          .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 12px;
          }

          .info-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
          }

          .info-label {
            font-size: 12px;
            color: #666;
          }

          .info-value {
            font-weight: 500;
            color: #333;
          }

          /* Data Tab Styles */
          .data-loading {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 200px;
          }

          .data-error {
            background-color: #ffebee;
            color: #c62828;
            padding: 16px;
            border-radius: 4px;
            text-align: center;
          }

          .data-stats-bar {
            display: flex;
            gap: 24px;
            padding: 8px 12px;
            background-color: #f5f5f5;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #666;
          }

          .data-stat strong {
            color: #333;
          }

          .data-table-container {
            flex: 1;
            overflow: auto;
            border: 1px solid #e0e0e0;
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
            background-color: #f5f5f5;
            padding: 8px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e0e0e0;
            white-space: nowrap;
            z-index: 1;
          }

          .data-table td {
            padding: 6px 12px;
            border-bottom: 1px solid #f0f0f0;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }

          .data-table tr:hover td {
            background-color: #f8f9fa;
          }

          .row-number-header,
          .row-number {
            background-color: #fafafa;
            color: #888;
            font-weight: normal;
            text-align: right;
            padding-right: 16px;
            border-right: 1px solid #e0e0e0;
            position: sticky;
            left: 0;
            z-index: 1;
          }

          .row-number-header {
            background-color: #f0f0f0;
            z-index: 2;
          }

          .data-table tr:hover .row-number {
            background-color: #f0f0f0;
          }

          /* Pagination */
          .data-pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            padding: 12px 0;
            border-top: 1px solid #e0e0e0;
            margin-top: 12px;
          }

          .btn-sm {
            padding: 4px 12px;
            font-size: 12px;
          }

          .page-info {
            margin: 0 12px;
            font-size: 13px;
            color: #666;
          }
        `}</style>
      </div>
    </Modal>
  );
}

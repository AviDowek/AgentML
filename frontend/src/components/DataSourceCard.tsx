import type { DataSource } from '../types/api';

interface DataSourceCardProps {
  dataSource: DataSource;
  onDelete?: (dataSource: DataSource) => void;
  onSelect?: (dataSource: DataSource) => void;
  onClick?: (dataSource: DataSource) => void;
  isSelected?: boolean;
}

export default function DataSourceCard({
  dataSource,
  onDelete,
  onSelect,
  onClick,
  isSelected = false,
}: DataSourceCardProps) {
  const schema = dataSource.schema_summary;
  const config = dataSource.config_json;
  const isExternalDataset = dataSource.type === 'external_dataset';

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const getFileTypeIcon = () => {
    // External datasets have their own icon
    if (isExternalDataset) {
      return '🌐';
    }

    const fileType = config?.file_type as string | undefined;
    switch (fileType) {
      case 'csv':
        return '📊';
      case 'excel':
        return '📗';
      case 'json':
        return '📋';
      case 'parquet':
        return '📦';
      default:
        return '📄';
    }
  };

  const handleCardClick = () => {
    if (onSelect) {
      onSelect(dataSource);
    } else if (onClick) {
      onClick(dataSource);
    }
  };

  const isClickable = onSelect || onClick;

  return (
    <div
      className={`data-source-card ${isSelected ? 'selected' : ''} ${isClickable ? 'clickable' : ''} ${isExternalDataset ? 'external' : ''}`}
      onClick={isClickable ? handleCardClick : undefined}
    >
      <div className="data-source-header">
        <span className="data-source-icon">{getFileTypeIcon()}</span>
        <span className="data-source-name">{dataSource.name}</span>
        {isExternalDataset && (
          <span className="external-badge">External</span>
        )}
        {onDelete && (
          <button
            className="btn-icon"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(dataSource);
            }}
            title="Delete"
          >
            &times;
          </button>
        )}
      </div>

      {/* Show fit_for_purpose for external datasets */}
      {isExternalDataset && config?.fit_for_purpose ? (
        <p className="data-source-description">
          {String(config.fit_for_purpose)}
        </p>
      ) : null}

      <div className="data-source-meta">
        {schema && (
          <>
            <div className="meta-row">
              <span className="meta-label">Columns:</span>
              <span className="meta-value">{schema.columns?.length || 0}</span>
            </div>
            <div className="meta-row">
              <span className="meta-label">Rows:</span>
              <span className="meta-value">
                {schema.row_count?.toLocaleString() || 'N/A'}
                {isExternalDataset && schema.row_count && ' (estimate)'}
              </span>
            </div>
          </>
        )}
        <div className="meta-row">
          <span className="meta-label">Type:</span>
          <span className="meta-value">
            {isExternalDataset ? 'External Dataset' : (config?.file_type ? String(config.file_type) : dataSource.type)}
          </span>
        </div>
        {isExternalDataset && config?.licensing ? (
          <div className="meta-row">
            <span className="meta-label">License:</span>
            <span className="meta-value">{String(config.licensing)}</span>
          </div>
        ) : null}
        <div className="meta-row">
          <span className="meta-label">Added:</span>
          <span className="meta-value">{formatDate(dataSource.created_at)}</span>
        </div>
      </div>

      {/* Source URL for external datasets */}
      {isExternalDataset && config?.source_url ? (
        <div className="data-source-external-link">
          <a
            href={String(config.source_url)}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="external-link"
          >
            View Source Dataset ↗
          </a>
        </div>
      ) : null}

      {schema?.columns && schema.columns.length > 0 && (
        <div className="data-source-columns">
          <span className="columns-label">Columns:</span>
          <div className="columns-list">
            {schema.columns.slice(0, 5).map((col) => (
              <span key={col.name} className="column-tag" title={`${col.dtype}`}>
                {col.name}
              </span>
            ))}
            {schema.columns.length > 5 && (
              <span className="column-more">+{schema.columns.length - 5} more</span>
            )}
          </div>
        </div>
      )}

      {/* Data quality warnings */}
      {schema?.warnings && schema.warnings.length > 0 && (
        <div className="data-source-warnings">
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
      )}

      {isClickable && (
        <div className="data-source-click-hint">
          Click to view details
        </div>
      )}

      <style>{`
        .data-source-card.clickable {
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .data-source-card.clickable:hover {
          border-color: #1976d2;
          box-shadow: 0 2px 8px rgba(25, 118, 210, 0.15);
        }

        .data-source-card.external {
          border-left: 4px solid #1976d2;
        }

        .external-badge {
          background-color: #e3f2fd;
          color: #1565c0;
          font-size: 11px;
          font-weight: 500;
          padding: 2px 8px;
          border-radius: 10px;
          text-transform: uppercase;
        }

        .data-source-description {
          margin: 8px 0;
          color: #555;
          font-size: 13px;
          line-height: 1.4;
        }

        .data-source-external-link {
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid #eee;
        }

        .external-link {
          color: #1976d2;
          text-decoration: none;
          font-size: 13px;
          display: inline-flex;
          align-items: center;
          gap: 4px;
        }

        .external-link:hover {
          text-decoration: underline;
        }

        .data-source-click-hint {
          margin-top: 8px;
          font-size: 11px;
          color: #999;
          text-align: center;
        }

        .data-source-warnings {
          margin-top: 12px;
          padding: 10px;
          background-color: #fff8e1;
          border: 1px solid #ffcc02;
          border-radius: 6px;
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
          font-size: 14px;
        }

        .warning-text {
          font-size: 12px;
          color: #8a6914;
          line-height: 1.4;
        }
      `}</style>
    </div>
  );
}

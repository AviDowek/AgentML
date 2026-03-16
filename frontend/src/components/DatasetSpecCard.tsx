import type { DatasetSpec, DataSource } from '../types/api';

interface DatasetSpecCardProps {
  datasetSpec: DatasetSpec;
  dataSources: DataSource[];
  onDelete?: (datasetSpec: DatasetSpec) => void;
  onClick?: (datasetSpec: DatasetSpec) => void;
  onDownload?: (datasetSpec: DatasetSpec) => void;
}

export default function DatasetSpecCard({
  datasetSpec,
  dataSources,
  onDelete,
  onClick,
  onDownload,
}: DatasetSpecCardProps) {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  // Get linked data source names
  const getLinkedSourceNames = (): string[] => {
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
      .map((id) => dataSources.find((ds) => ds.id === id)?.name || 'Unknown')
      .filter((name) => name !== 'Unknown');
  };

  const linkedSources = getLinkedSourceNames();
  const featureCount = datasetSpec.feature_columns?.length || 0;

  const handleCardClick = () => {
    if (onClick) {
      onClick(datasetSpec);
    }
  };

  return (
    <div
      className={`dataset-spec-card ${onClick ? 'clickable' : ''}`}
      onClick={onClick ? handleCardClick : undefined}
    >
      <div className="dataset-spec-header">
        <span className="dataset-spec-icon">📋</span>
        <span className="dataset-spec-name">{datasetSpec.name}</span>
        {datasetSpec.is_time_based && (
          <span className="time-based-badge" title={`Time-based task: ${datasetSpec.prediction_horizon || 'temporal prediction'}`}>
            ⏱️
          </span>
        )}
        {onDownload && (
          <button
            className="btn-icon btn-download"
            onClick={(e) => {
              e.stopPropagation();
              onDownload(datasetSpec);
            }}
            title="Download as CSV"
          >
            ⬇
          </button>
        )}
        {onDelete && (
          <button
            className="btn-icon"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(datasetSpec);
            }}
            title="Delete"
          >
            &times;
          </button>
        )}
      </div>

      {datasetSpec.description && (
        <p className="dataset-spec-description">{datasetSpec.description}</p>
      )}

      <div className="dataset-spec-meta">
        <div className="meta-row">
          <span className="meta-label">Target:</span>
          <span className="meta-value target-value">
            {datasetSpec.target_column || 'Not set'}
          </span>
        </div>
        <div className="meta-row">
          <span className="meta-label">Features:</span>
          <span className="meta-value">{featureCount} columns</span>
        </div>
        <div className="meta-row">
          <span className="meta-label">Created:</span>
          <span className="meta-value">{formatDate(datasetSpec.created_at)}</span>
        </div>
      </div>

      {linkedSources.length > 0 && (
        <div className="dataset-spec-sources">
          <span className="sources-label">Data Sources:</span>
          <div className="sources-list">
            {linkedSources.slice(0, 3).map((name, idx) => (
              <span key={idx} className="source-tag">
                {name}
              </span>
            ))}
            {linkedSources.length > 3 && (
              <span className="source-more">+{linkedSources.length - 3} more</span>
            )}
          </div>
        </div>
      )}

      {datasetSpec.feature_columns && datasetSpec.feature_columns.length > 0 && (
        <div className="dataset-spec-features">
          <span className="features-label">Features:</span>
          <div className="features-list">
            {datasetSpec.feature_columns.slice(0, 5).map((col, idx) => (
              <span key={idx} className="feature-tag">
                {col}
              </span>
            ))}
            {datasetSpec.feature_columns.length > 5 && (
              <span className="feature-more">
                +{datasetSpec.feature_columns.length - 5} more
              </span>
            )}
          </div>
        </div>
      )}

      {onClick && (
        <div className="dataset-spec-click-hint">Click to view details</div>
      )}

      <style>{`
        .dataset-spec-card {
          background: #1e1e1e;
          border: 1px solid #404040;
          border-radius: 8px;
          padding: 16px;
          transition: all 0.2s ease;
          border-left: 4px solid #a855f7;
        }

        .dataset-spec-card.clickable {
          cursor: pointer;
        }

        .dataset-spec-card.clickable:hover {
          border-color: #a855f7;
          box-shadow: 0 2px 8px rgba(168, 85, 247, 0.25);
        }

        .dataset-spec-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }

        .dataset-spec-icon {
          font-size: 18px;
        }

        .dataset-spec-name {
          flex: 1;
          font-weight: 600;
          color: #e0e0e0;
          font-size: 15px;
        }

        .time-based-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(147, 51, 234, 0.2) 100%);
          border: 1px solid rgba(147, 51, 234, 0.4);
          border-radius: 4px;
          padding: 2px 6px;
          font-size: 12px;
          cursor: help;
        }

        .dataset-spec-card .btn-icon {
          background: none;
          border: none;
          color: #808080;
          font-size: 20px;
          cursor: pointer;
          padding: 0 4px;
        }

        .dataset-spec-card .btn-icon:hover {
          color: #ef4444;
        }

        .dataset-spec-card .btn-icon.btn-download:hover {
          color: #22c55e;
        }

        .dataset-spec-description {
          margin: 8px 0;
          color: #b0b0b0;
          font-size: 13px;
          line-height: 1.4;
        }

        .dataset-spec-meta {
          margin: 12px 0;
        }

        .meta-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 4px 0;
        }

        .meta-label {
          color: #909090;
          font-size: 13px;
        }

        .meta-value {
          color: #d0d0d0;
          font-size: 13px;
          font-weight: 500;
        }

        .target-value {
          color: #ffffff;
          background-color: #6b21a8;
          padding: 2px 8px;
          border-radius: 4px;
        }

        .dataset-spec-sources {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid #333333;
        }

        .sources-label {
          display: block;
          color: #909090;
          font-size: 12px;
          margin-bottom: 6px;
        }

        .sources-list {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }

        .source-tag {
          display: inline-block;
          background-color: #1e3a5f;
          color: #60a5fa;
          padding: 3px 8px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 500;
        }

        .source-more {
          color: #707070;
          font-size: 11px;
        }

        .dataset-spec-features {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid #333333;
        }

        .features-label {
          display: block;
          color: #909090;
          font-size: 12px;
          margin-bottom: 6px;
        }

        .features-list {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }

        .feature-tag {
          display: inline-block;
          background-color: #374151;
          color: #d0d0d0;
          padding: 3px 8px;
          border-radius: 4px;
          font-size: 11px;
        }

        .feature-more {
          color: #707070;
          font-size: 11px;
        }

        .dataset-spec-click-hint {
          margin-top: 12px;
          font-size: 11px;
          color: #707070;
          text-align: center;
        }
      `}</style>
    </div>
  );
}

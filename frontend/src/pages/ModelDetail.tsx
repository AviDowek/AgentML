import { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import type { ModelVersion, ServingFeature } from '../types/api';
import { getModelVersion, promoteModel, ApiException } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import StatusBadge from '../components/StatusBadge';
import ChatBot from '../components/ChatBot';
import ModelTestForm from '../components/ModelTestForm';
import ModelExplainer from '../components/ModelExplainer';
import ValidationSamplesTab from '../components/ValidationSamplesTab';

type ModelTab = 'overview' | 'validation' | 'test' | 'explain';

export default function ModelDetail() {
  const { modelId } = useParams<{ modelId: string }>();

  const [model, setModel] = useState<ModelVersion | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [promoting, setPromoting] = useState(false);
  const [activeTab, setActiveTab] = useState<ModelTab>('overview');

  const fetchData = useCallback(async () => {
    if (!modelId) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await getModelVersion(modelId);
      setModel(data);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load model data');
      }
    } finally {
      setIsLoading(false);
    }
  }, [modelId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handlePromote = async (status: 'candidate' | 'shadow' | 'production') => {
    if (!modelId) return;
    setPromoting(true);
    try {
      const updated = await promoteModel(modelId, { status });
      setModel(updated);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setPromoting(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (isLoading) {
    return (
      <div className="model-detail-page">
        <LoadingSpinner message="Loading model..." />
      </div>
    );
  }

  if (error && !model) {
    return (
      <div className="model-detail-page">
        <ErrorMessage message={error} onRetry={fetchData} />
        <Link to="/models" className="btn btn-secondary" style={{ marginTop: '1rem' }}>
          Back to Models
        </Link>
      </div>
    );
  }

  if (!model) return null;

  // Organize metrics into categories
  const allMetrics = model.metrics_json || {};
  const performanceMetrics: Record<string, unknown> = {};
  const trainingMetrics: Record<string, unknown> = {};
  const modelDownloaded = allMetrics.model_downloaded !== false; // Default to true if not set

  Object.entries(allMetrics).forEach(([key, value]) => {
    if (key.includes('time') || key.includes('num_') || key === 'model_downloaded') {
      if (key !== 'model_downloaded') {
        trainingMetrics[key] = value;
      }
    } else {
      performanceMetrics[key] = value;
    }
  });

  // Infer model type from model name or metrics if not explicitly set
  const inferModelType = (): string => {
    if (model.model_type) return model.model_type;

    // Try to infer from model name
    const name = model.name?.toLowerCase() || '';
    if (name.includes('lightgbm') || name.includes('lgbm')) return 'LightGBM';
    if (name.includes('xgboost') || name.includes('xgb')) return 'XGBoost';
    if (name.includes('catboost')) return 'CatBoost';
    if (name.includes('random_forest') || name.includes('randomforest')) return 'Random Forest';
    if (name.includes('gradient_boost') || name.includes('gbm')) return 'Gradient Boosting';
    if (name.includes('logistic')) return 'Logistic Regression';
    if (name.includes('linear')) return 'Linear Model';
    if (name.includes('neural') || name.includes('nn') || name.includes('mlp')) return 'Neural Network';
    if (name.includes('ridge')) return 'Ridge Regression';
    if (name.includes('lasso')) return 'Lasso Regression';
    if (name.includes('svm') || name.includes('svc') || name.includes('svr')) return 'Support Vector Machine';
    if (name.includes('knn') || name.includes('neighbors')) return 'K-Nearest Neighbors';

    // Try to infer task type from metrics
    const metricKeys = Object.keys(performanceMetrics).map(k => k.toLowerCase());
    if (metricKeys.some(k => k.includes('rmse') || k.includes('mae') || k === 'r2')) {
      return 'Regressor';
    }
    if (metricKeys.some(k => k.includes('roc_auc') || k === 'accuracy' || k === 'f1')) {
      return 'Classifier';
    }

    return 'AutoML Model';
  };

  const displayModelType = inferModelType();

  const featureImportances = model.feature_importances_json || {};
  const sortedFeatures = Object.entries(featureImportances)
    .sort(([, a], [, b]) => (b as number) - (a as number))
    .slice(0, 15);

  return (
    <div className="model-detail-page">
      {error && (
        <div className="form-error" style={{ marginBottom: '1rem' }}>
          {error}
          <button onClick={() => setError(null)} style={{ marginLeft: '1rem' }}>
            Dismiss
          </button>
        </div>
      )}

      {!modelDownloaded && (
        <div
          className="info-banner"
          style={{
            marginBottom: '1rem',
            padding: '1rem',
            backgroundColor: '#1e3a5f',
            border: '1px solid #3b82f6',
            borderRadius: '0.5rem',
            color: '#bfdbfe',
          }}
        >
          <strong>🌐 Model Running on Cloud</strong>
          <p style={{ margin: '0.5rem 0 0 0' }}>
            Model files are stored on Modal cloud for faster training.
            Predictions run remotely on Modal infrastructure.
            The first prediction may take a few extra seconds while the container starts.
            Use the Export button to download the model locally if needed.
          </p>
        </div>
      )}

      <div className="page-header">
        <div>
          <div className="breadcrumb">
            <Link to="/projects">Projects</Link>
            <span>/</span>
            <Link to={`/projects/${model.project_id}`}>Project</Link>
            <span>/</span>
            <span>{model.name}</span>
          </div>
          <h2>{model.name}</h2>
          <div className="model-detail-meta">
            <StatusBadge status={model.status} />
            <span className="meta-separator">|</span>
            <span>Type: {displayModelType}</span>
          </div>
        </div>
        <div className="header-actions">
          {model.status === 'candidate' && (
            <>
              <button
                className="btn btn-secondary"
                onClick={() => handlePromote('shadow')}
                disabled={promoting}
              >
                Promote to Shadow
              </button>
              <button
                className="btn btn-primary"
                onClick={() => handlePromote('production')}
                disabled={promoting}
              >
                Promote to Production
              </button>
            </>
          )}
          {model.status === 'shadow' && (
            <button
              className="btn btn-primary"
              onClick={() => handlePromote('production')}
              disabled={promoting}
            >
              Promote to Production
            </button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="model-detail-tabs">
        <button
          className={`model-detail-tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={`model-detail-tab ${activeTab === 'validation' ? 'active' : ''}`}
          onClick={() => setActiveTab('validation')}
        >
          Validation Samples
        </button>
        <button
          className={`model-detail-tab ${activeTab === 'test' ? 'active' : ''}`}
          onClick={() => setActiveTab('test')}
          title={!modelDownloaded ? 'Predictions will run on Modal cloud' : undefined}
        >
          Test Model {!modelDownloaded && '(remote)'}
        </button>
        <button
          className={`model-detail-tab ${activeTab === 'explain' ? 'active' : ''}`}
          onClick={() => setActiveTab('explain')}
        >
          Q&A / Explainer
        </button>
      </div>

      {/* Tab Content */}
      <div className="model-tab-content">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <>
            <div className="detail-grid">
              <div className="detail-section">
                <h3>Model Information</h3>
                <div className="detail-card">
                  <div className="detail-row">
                    <span className="detail-label">Model Type:</span>
                    <span className="detail-value">{displayModelType}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Status:</span>
                    <StatusBadge status={model.status} />
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Created:</span>
                    <span className="detail-value">{formatDate(model.created_at)}</span>
                  </div>
                  {model.experiment_id && (
                    <div className="detail-row">
                      <span className="detail-label">Experiment:</span>
                      <Link to={`/experiments/${model.experiment_id}`} className="detail-link">
                        View Experiment
                      </Link>
                    </div>
                  )}
                  {model.artifact_location && (
                    <div className="detail-row">
                      <span className="detail-label">Artifacts:</span>
                      <span className="detail-value artifact-path">{model.artifact_location}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="detail-section">
                <h3>Performance Metrics</h3>
                <div className="detail-card">
                  {Object.keys(performanceMetrics).length === 0 ? (
                    <p className="empty-text">No performance metrics available</p>
                  ) : (
                    <div className="metrics-grid">
                      {Object.entries(performanceMetrics).map(([key, value]) => {
                        // Error metrics should always be displayed as positive values
                        const isErrorMetric = key.toLowerCase().includes('error') ||
                          key.toLowerCase().includes('rmse') ||
                          key.toLowerCase().includes('mse') ||
                          key.toLowerCase().includes('mae') ||
                          key.toLowerCase().includes('loss');
                        const displayValue = typeof value === 'number'
                          ? (isErrorMetric ? Math.abs(value) : value)
                          : value;
                        const displayKey = key.replace(/^neg_/, '');
                        return (
                          <div key={key} className="metric-box">
                            <span className="metric-label">{displayKey}</span>
                            <span className="metric-value">
                              {typeof displayValue === 'number' ? displayValue.toFixed(4) : String(displayValue)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              {Object.keys(trainingMetrics).length > 0 && (
                <div className="detail-section">
                  <h3>Training Statistics</h3>
                  <div className="detail-card">
                    {Object.entries(trainingMetrics).map(([key, value]) => (
                      <div key={key} className="detail-row">
                        <span className="detail-label">{key.replace(/_/g, ' ')}:</span>
                        <span className="detail-value">
                          {typeof value === 'number'
                            ? key.includes('time')
                              ? `${value.toFixed(1)}s`
                              : value.toLocaleString()
                            : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {sortedFeatures.length > 0 && (
                <div className="detail-section full-width">
                  <h3>Feature Importances (Top {sortedFeatures.length})</h3>
                  <div className="detail-card">
                    <div className="feature-importances">
                      {sortedFeatures.map(([feature, importance]) => {
                        const maxImportance = sortedFeatures[0][1] as number;
                        const width = ((importance as number) / maxImportance) * 100;
                        return (
                          <div key={feature} className="feature-row">
                            <span className="feature-name">{feature}</span>
                            <div className="feature-bar-container">
                              <div
                                className="feature-bar"
                                style={{ width: `${width}%` }}
                              />
                            </div>
                            <span className="feature-value">
                              {(importance as number).toFixed(4)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {model.serving_config_json && Object.keys(model.serving_config_json).length > 0 && (
              <div className="detail-section full-width" style={{ marginTop: '1.5rem' }}>
                <h3>Serving Configuration</h3>
                <div className="detail-card">
                  <pre className="json-display">
                    {JSON.stringify(model.serving_config_json, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </>
        )}

        {/* Validation Samples Tab */}
        {activeTab === 'validation' && (
          <ValidationSamplesTab
            modelId={model.id}
            servingConfig={model.serving_config_json as {
              features?: ServingFeature[];
              target_column?: string;
              task_type?: string;
            } | null}
          />
        )}

        {/* Test Model Tab */}
        {activeTab === 'test' && (
          <div className="detail-section full-width">
            <h3>Test Model</h3>
            <div className="detail-card">
              <p style={{ marginBottom: '1rem', color: '#6b7280' }}>
                Enter feature values to get a prediction from the trained model.
              </p>
              <ModelTestForm
                modelId={model.id}
                projectId={model.project_id}
                servingConfig={model.serving_config_json as {
                  features?: ServingFeature[];
                  target_column?: string;
                  task_type?: string;
                } | null}
              />
            </div>
          </div>
        )}

        {/* Q&A / Explainer Tab */}
        {activeTab === 'explain' && (
          <div className="detail-section full-width">
            <h3>Ask About Your Model</h3>
            <div className="detail-card">
              <p style={{ marginBottom: '1rem', color: '#6b7280' }}>
                Get AI-powered insights and explanations about your model's performance and features.
              </p>
              <ModelExplainer modelId={model.id} modelName={model.name} />
            </div>
          </div>
        )}
      </div>

      {/* AI Assistant with model context */}
      <ChatBot
        title="Model Assistant"
        contextType="model"
        context={{
          model: {
            id: model.id,
            name: model.name,
            model_type: model.model_type,
            status: model.status,
            metrics: model.metrics_json,
            feature_importances: model.feature_importances_json,
            created_at: model.created_at,
          },
        }}
      />
    </div>
  );
}

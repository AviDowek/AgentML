import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import type { ModelVersion } from '../types/api';
import { listProjects, listModelVersions, ApiException } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import StatusBadge from '../components/StatusBadge';

interface ModelWithProject extends ModelVersion {
  projectName: string;
}

export default function Models() {
  const [models, setModels] = useState<ModelWithProject[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');

  useEffect(() => {
    const fetchAll = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // First get all projects
        const projectsResponse = await listProjects();
        const projects = projectsResponse.items;

        // Then get models for each project
        const allModels: ModelWithProject[] = [];
        for (const project of projects) {
          try {
            const projectModels = await listModelVersions(project.id);
            allModels.push(
              ...projectModels.map((model) => ({
                ...model,
                projectName: project.name,
              }))
            );
          } catch {
            // Skip projects that fail
          }
        }

        // Sort by created_at desc
        allModels.sort(
          (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        );

        setModels(allModels);
      } catch (err) {
        if (err instanceof ApiException) {
          setError(err.detail);
        } else {
          setError('Failed to load models');
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchAll();
  }, []);

  const filteredModels =
    statusFilter === 'all'
      ? models
      : models.filter((m) => m.status === statusFilter);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  // Infer model type from model name or metrics if not explicitly set
  const getModelType = (model: ModelVersion): string => {
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

    // Try to infer task type from metrics
    const metrics = model.metrics_json || {};
    const metricKeys = Object.keys(metrics).map(k => k.toLowerCase());
    if (metricKeys.some(k => k.includes('rmse') || k.includes('mae') || k === 'r2')) {
      return 'Regressor';
    }
    if (metricKeys.some(k => k.includes('roc_auc') || k === 'accuracy' || k === 'f1')) {
      return 'Classifier';
    }

    return 'AutoML Model';
  };

  const getMainMetric = (model: ModelVersion): string => {
    if (!model.metrics_json) return 'N/A';
    const metrics = model.metrics_json;
    // Try common metric names
    const metricKeys = ['accuracy', 'roc_auc', 'rmse', 'mae', 'f1', 'r2'];
    for (const key of metricKeys) {
      if (key in metrics && typeof metrics[key] === 'number') {
        return `${key}: ${(metrics[key] as number).toFixed(4)}`;
      }
    }
    // Return first numeric metric
    for (const [key, value] of Object.entries(metrics)) {
      if (typeof value === 'number' && !key.includes('time') && !key.includes('num_')) {
        return `${key}: ${value.toFixed(4)}`;
      }
    }
    return 'N/A';
  };

  if (isLoading) {
    return (
      <div className="models-page">
        <LoadingSpinner message="Loading models..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="models-page">
        <ErrorMessage message={error} />
      </div>
    );
  }

  return (
    <div className="models-page">
      <div className="page-header">
        <div>
          <h2>All Models</h2>
          <p className="page-subtitle">
            {filteredModels.length} {filteredModels.length === 1 ? 'model' : 'models'}
            {statusFilter !== 'all' && ` (${statusFilter})`}
          </p>
        </div>
        <div className="filter-controls">
          <select
            className="form-select"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">All Statuses</option>
            <option value="candidate">Candidate</option>
            <option value="shadow">Shadow</option>
            <option value="production">Production</option>
            <option value="retired">Retired</option>
          </select>
        </div>
      </div>

      {models.length === 0 ? (
        <EmptyState
          title="No models yet"
          description="Run an experiment to train and generate models"
          actionLabel="Go to Projects"
          onAction={() => window.location.href = '/projects'}
          icon="🤖"
        />
      ) : filteredModels.length === 0 ? (
        <EmptyState
          title={`No ${statusFilter} models`}
          description="Try a different filter or train more models"
          icon="🔍"
        />
      ) : (
        <div className="models-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Project</th>
                <th>Type</th>
                <th>Status</th>
                <th>Metric</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((model) => (
                <tr key={model.id}>
                  <td>
                    <Link to={`/models/${model.id}`} className="table-link">
                      {model.name}
                    </Link>
                  </td>
                  <td>
                    <Link to={`/projects/${model.project_id}`} className="table-link-secondary">
                      {model.projectName}
                    </Link>
                  </td>
                  <td>{getModelType(model)}</td>
                  <td>
                    <StatusBadge status={model.status} />
                  </td>
                  <td className="metric-cell">{getMainMetric(model)}</td>
                  <td>{formatDate(model.created_at)}</td>
                  <td>
                    <Link to={`/models/${model.id}`} className="btn btn-small btn-secondary">
                      View
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

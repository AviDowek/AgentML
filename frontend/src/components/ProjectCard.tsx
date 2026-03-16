import { Link } from 'react-router-dom';
import type { Project } from '../types/api';
import StatusBadge from './StatusBadge';
import { useAuth } from '../contexts/AuthContext';

interface ProjectCardProps {
  project: Project;
  onDelete?: (project: Project) => void;
  onShare?: (project: Project) => void;
}

const taskTypeLabels: Record<string, string> = {
  binary: 'Binary Classification',
  multiclass: 'Multi-class Classification',
  regression: 'Regression',
  quantile: 'Quantile Regression',
  timeseries_forecast: 'Time Series Forecast',
  multimodal_classification: 'Multimodal Classification',
  multimodal_regression: 'Multimodal Regression',
  classification: 'Classification (Legacy)',
};

export default function ProjectCard({ project, onDelete, onShare }: ProjectCardProps) {
  const { isAuthenticated } = useAuth();
  const taskTypeLabel = project.task_type
    ? taskTypeLabels[project.task_type] || project.task_type
    : 'Not set';

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div className="project-card">
      <div className="project-card-header">
        <Link to={`/projects/${project.id}`} className="project-card-title">
          {project.name}
        </Link>
        <StatusBadge status={project.status} />
      </div>

      {project.description && (
        <p className="project-card-description">{project.description}</p>
      )}

      <div className="project-card-meta">
        <div className="meta-item">
          <span className="meta-label">Task Type:</span>
          <span className="meta-value">{taskTypeLabel}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Created:</span>
          <span className="meta-value">{formatDate(project.created_at)}</span>
        </div>
      </div>

      <div className="project-card-actions">
        <Link to={`/projects/${project.id}`} className="btn btn-primary btn-small">
          Open Project
        </Link>
        {isAuthenticated && onShare && (
          <button
            className="btn btn-share"
            onClick={() => onShare(project)}
            title="Share project"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="18" cy="5" r="3" />
              <circle cx="6" cy="12" r="3" />
              <circle cx="18" cy="19" r="3" />
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
              <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
            </svg>
            Share
          </button>
        )}
        {onDelete && (
          <button
            className="btn btn-danger btn-small"
            onClick={() => onDelete(project)}
          >
            Delete
          </button>
        )}
      </div>
    </div>
  );
}

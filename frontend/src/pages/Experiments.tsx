import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import type { Experiment } from '../types/api';
import { listProjects, listExperiments, deleteExperiment, bulkDeleteExperiments, ApiException } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import StatusBadge from '../components/StatusBadge';
import ConfirmDialog from '../components/ConfirmDialog';

interface ExperimentWithProject extends Experiment {
  projectName: string;
}

export default function Experiments() {
  const [experiments, setExperiments] = useState<ExperimentWithProject[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [experimentToDelete, setExperimentToDelete] = useState<ExperimentWithProject | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showBulkDeleteConfirm, setShowBulkDeleteConfirm] = useState(false);
  const [isBulkDeleting, setIsBulkDeleting] = useState(false);

  const fetchAll = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // First get all projects
      const projectsResponse = await listProjects();
      const projects = projectsResponse.items;

      // Then get experiments for each project
      const allExperiments: ExperimentWithProject[] = [];
      for (const project of projects) {
        try {
          const projectExperiments = await listExperiments(project.id);
          allExperiments.push(
            ...projectExperiments.map((exp) => ({
              ...exp,
              projectName: project.name,
            }))
          );
        } catch {
          // Skip projects that fail
        }
      }

      // Sort by created_at desc
      allExperiments.sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );

      setExperiments(allExperiments);
      setSelectedIds(new Set()); // Clear selection on refresh
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load experiments');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const handleDeleteExperiment = async () => {
    if (!experimentToDelete) return;

    setIsDeleting(true);
    try {
      await deleteExperiment(experimentToDelete.id);
      setExperimentToDelete(null);
      // Refresh the list
      await fetchAll();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to delete experiment');
      }
    } finally {
      setIsDeleting(false);
    }
  };

  const handleBulkDelete = async () => {
    if (selectedIds.size === 0) return;

    setIsBulkDeleting(true);
    try {
      const result = await bulkDeleteExperiments(Array.from(selectedIds));
      setShowBulkDeleteConfirm(false);

      if (result.failed_ids.length > 0) {
        setError(`Deleted ${result.deleted_count} experiments. ${result.failed_ids.length} failed: ${result.errors.join(', ')}`);
      }

      // Refresh the list
      await fetchAll();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to delete experiments');
      }
    } finally {
      setIsBulkDeleting(false);
    }
  };

  const toggleSelect = (id: string) => {
    const newSelected = new Set(selectedIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedIds(newSelected);
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === experiments.length) {
      // Deselect all
      setSelectedIds(new Set());
    } else {
      // Select all non-running experiments
      const selectable = experiments
        .filter(exp => exp.status !== 'running')
        .map(exp => exp.id);
      setSelectedIds(new Set(selectable));
    }
  };

  const selectableCount = experiments.filter(exp => exp.status !== 'running').length;
  const allSelectableSelected = selectableCount > 0 && selectedIds.size === selectableCount;

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  if (isLoading) {
    return (
      <div className="experiments-page">
        <LoadingSpinner message="Loading experiments..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="experiments-page">
        <ErrorMessage message={error} />
      </div>
    );
  }

  return (
    <div className="experiments-page">
      <div className="page-header">
        <div>
          <h2>All Experiments</h2>
          <p className="page-subtitle">
            {experiments.length} {experiments.length === 1 ? 'experiment' : 'experiments'} across all projects
            {selectedIds.size > 0 && ` (${selectedIds.size} selected)`}
          </p>
        </div>
        {selectedIds.size > 0 && (
          <div className="header-actions">
            <button
              className="btn btn-danger"
              onClick={() => setShowBulkDeleteConfirm(true)}
            >
              Delete Selected ({selectedIds.size})
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => setSelectedIds(new Set())}
            >
              Clear Selection
            </button>
          </div>
        )}
      </div>

      {experiments.length === 0 ? (
        <EmptyState
          title="No experiments yet"
          description="Create a project and run an experiment to see results here"
          actionLabel="Go to Projects"
          onAction={() => window.location.href = '/projects'}
          icon="🧪"
        />
      ) : (
        <div className="experiments-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '40px' }}>
                  <input
                    type="checkbox"
                    checked={allSelectableSelected}
                    onChange={toggleSelectAll}
                    title="Select all"
                  />
                </th>
                <th>Name</th>
                <th>Project</th>
                <th>Status</th>
                <th>Metric</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => (
                <tr key={exp.id} className={selectedIds.has(exp.id) ? 'selected-row' : ''}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedIds.has(exp.id)}
                      onChange={() => toggleSelect(exp.id)}
                      disabled={exp.status === 'running'}
                      title={exp.status === 'running' ? 'Cannot select running experiment' : 'Select'}
                    />
                  </td>
                  <td>
                    <Link to={`/experiments/${exp.id}`} className="table-link">
                      {exp.name}
                    </Link>
                  </td>
                  <td>
                    <Link to={`/projects/${exp.project_id}`} className="table-link-secondary">
                      {exp.projectName}
                    </Link>
                  </td>
                  <td>
                    <StatusBadge status={exp.status} />
                  </td>
                  <td>{exp.primary_metric || 'Auto'}</td>
                  <td>{formatDate(exp.created_at)}</td>
                  <td>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <Link to={`/experiments/${exp.id}`} className="btn btn-small btn-secondary">
                        View
                      </Link>
                      <button
                        className="btn btn-small btn-danger"
                        onClick={() => setExperimentToDelete(exp)}
                        disabled={exp.status === 'running'}
                        title={exp.status === 'running' ? 'Cannot delete running experiment' : 'Delete experiment'}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        isOpen={experimentToDelete !== null}
        title="Delete Experiment"
        message={`Are you sure you want to delete "${experimentToDelete?.name}"? This will also delete all associated models and results. This action cannot be undone.`}
        confirmLabel={isDeleting ? 'Deleting...' : 'Delete'}
        onConfirm={handleDeleteExperiment}
        onClose={() => setExperimentToDelete(null)}
        variant="danger"
      />

      {/* Bulk Delete Confirmation Dialog */}
      <ConfirmDialog
        isOpen={showBulkDeleteConfirm}
        title="Delete Selected Experiments"
        message={`Are you sure you want to delete ${selectedIds.size} experiment${selectedIds.size === 1 ? '' : 's'}? This will also delete all associated models and results. This action cannot be undone.`}
        confirmLabel={isBulkDeleting ? 'Deleting...' : `Delete ${selectedIds.size} Experiment${selectedIds.size === 1 ? '' : 's'}`}
        onConfirm={handleBulkDelete}
        onClose={() => setShowBulkDeleteConfirm(false)}
        variant="danger"
      />
    </div>
  );
}

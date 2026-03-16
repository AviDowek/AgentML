import { useState, useEffect, useCallback } from 'react';
import type { Project, ProjectCreate } from '../types/api';
import { listProjects, createProject, deleteProject, ApiException } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import Modal from '../components/Modal';
import ConfirmDialog from '../components/ConfirmDialog';
import ProjectCard from '../components/ProjectCard';
import CreateProjectForm from '../components/CreateProjectForm';
import ShareDialog from '../components/ShareDialog';

export default function Projects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [shareTarget, setShareTarget] = useState<Project | null>(null);

  const fetchProjects = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await listProjects();
      setProjects(response.items);
      setTotal(response.total);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load projects. Make sure the backend is running.');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleCreateProject = async (data: ProjectCreate) => {
    setIsCreating(true);
    try {
      const newProject = await createProject(data);
      setProjects((prev) => [newProject, ...prev]);
      setTotal((prev) => prev + 1);
      setIsCreateModalOpen(false);
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteProject = async () => {
    if (!deleteTarget) return;

    setIsDeleting(true);
    try {
      await deleteProject(deleteTarget.id);
      setProjects((prev) => prev.filter((p) => p.id !== deleteTarget.id));
      setTotal((prev) => prev - 1);
      setDeleteTarget(null);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to delete project');
      }
    } finally {
      setIsDeleting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="projects-page">
        <LoadingSpinner message="Loading projects..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="projects-page">
        <ErrorMessage message={error} onRetry={fetchProjects} />
      </div>
    );
  }

  return (
    <div className="projects-page">
      <div className="page-header">
        <div>
          <h2>Projects</h2>
          <p className="page-subtitle">
            {total} {total === 1 ? 'project' : 'projects'}
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => setIsCreateModalOpen(true)}
        >
          + Create Project
        </button>
      </div>

      {projects.length === 0 ? (
        <EmptyState
          title="No projects yet"
          description="Create your first ML project to get started"
          actionLabel="Create Project"
          onAction={() => setIsCreateModalOpen(true)}
          icon="📁"
        />
      ) : (
        <div className="projects-grid">
          {projects.map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onDelete={setDeleteTarget}
              onShare={setShareTarget}
            />
          ))}
        </div>
      )}

      <Modal
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        title="Create New Project"
        size="medium"
      >
        <CreateProjectForm
          onSubmit={handleCreateProject}
          onCancel={() => setIsCreateModalOpen(false)}
          isLoading={isCreating}
        />
      </Modal>

      <ConfirmDialog
        isOpen={!!deleteTarget}
        onClose={() => setDeleteTarget(null)}
        onConfirm={handleDeleteProject}
        title="Delete Project"
        message={`Are you sure you want to delete "${deleteTarget?.name}"? This will delete all associated data sources, datasets, experiments, and models.`}
        confirmLabel="Delete"
        variant="danger"
        isLoading={isDeleting}
      />

      {shareTarget && (
        <ShareDialog
          isOpen={!!shareTarget}
          onClose={() => setShareTarget(null)}
          resourceType="project"
          resourceId={shareTarget.id}
          resourceName={shareTarget.name}
        />
      )}
    </div>
  );
}

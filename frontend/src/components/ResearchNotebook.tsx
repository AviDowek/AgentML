import { useState, useEffect, useCallback } from 'react';
import type {
  ResearchCycleSummary,
  ResearchCycleResponse,
  LabNotebookEntry,
  LabNotebookEntryCreate,
} from '../types/api';
import {
  listResearchCycles,
  getResearchCycle,
  createResearchCycle,
  listLabNotebookEntries,
  createLabNotebookEntry,
  deleteLabNotebookEntry,
  ApiException,
} from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import StatusBadge from './StatusBadge';
import Modal from './Modal';
import ConfirmDialog from './ConfirmDialog';
import './ResearchNotebook.css';

interface ResearchNotebookProps {
  projectId: string;
}

export default function ResearchNotebook({ projectId }: ResearchNotebookProps) {
  const [cycles, setCycles] = useState<ResearchCycleSummary[]>([]);
  const [selectedCycle, setSelectedCycle] = useState<ResearchCycleResponse | null>(null);
  const [entries, setEntries] = useState<LabNotebookEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingCycle, setIsLoadingCycle] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Modal states
  const [isAddEntryModalOpen, setIsAddEntryModalOpen] = useState(false);
  const [isCreatingEntry, setIsCreatingEntry] = useState(false);
  const [newEntryTitle, setNewEntryTitle] = useState('');
  const [newEntryBody, setNewEntryBody] = useState('');
  const [deleteEntryTarget, setDeleteEntryTarget] = useState<LabNotebookEntry | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Expanded entry for viewing details
  const [expandedEntryId, setExpandedEntryId] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [cyclesResponse, entriesResponse] = await Promise.all([
        listResearchCycles(projectId),
        listLabNotebookEntries(projectId),
      ]);
      setCycles(cyclesResponse.cycles);
      setEntries(entriesResponse.entries);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load research notebook');
      }
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleSelectCycle = async (cycleId: string) => {
    if (selectedCycle?.id === cycleId) {
      setSelectedCycle(null);
      return;
    }

    setIsLoadingCycle(true);
    try {
      const cycleDetails = await getResearchCycle(cycleId);
      setSelectedCycle(cycleDetails);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setIsLoadingCycle(false);
    }
  };

  const handleCreateCycle = async () => {
    try {
      const newCycle = await createResearchCycle(projectId);
      setCycles(prev => [newCycle, ...prev]);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    }
  };

  const handleCreateEntry = async () => {
    if (!newEntryTitle.trim()) return;

    setIsCreatingEntry(true);
    try {
      const entryData: LabNotebookEntryCreate = {
        title: newEntryTitle,
        body_markdown: newEntryBody || undefined,
        research_cycle_id: selectedCycle?.id,
        author_type: 'human',
      };
      const newEntry = await createLabNotebookEntry(projectId, entryData);
      setEntries(prev => [newEntry, ...prev]);
      setIsAddEntryModalOpen(false);
      setNewEntryTitle('');
      setNewEntryBody('');
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setIsCreatingEntry(false);
    }
  };

  const handleDeleteEntry = async () => {
    if (!deleteEntryTarget) return;

    setIsDeleting(true);
    try {
      await deleteLabNotebookEntry(deleteEntryTarget.id);
      setEntries(prev => prev.filter(e => e.id !== deleteEntryTarget.id));
      setDeleteEntryTarget(null);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setIsDeleting(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return '#27ae60';
      case 'running':
        return '#3498db';
      case 'failed':
        return '#e74c3c';
      default:
        return '#95a5a6';
    }
  };

  if (isLoading) {
    return <LoadingSpinner message="Loading research notebook..." />;
  }

  // Filter entries based on selected cycle
  const filteredEntries = selectedCycle
    ? entries.filter(e => e.research_cycle_id === selectedCycle.id)
    : entries;

  return (
    <div className="research-notebook">
      {error && (
        <div className="notebook-error">
          {error}
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <div className="notebook-layout">
        {/* Sidebar: Research Cycles */}
        <div className="notebook-sidebar">
          <div className="sidebar-header">
            <h3>Research Cycles</h3>
            <button className="btn btn-small btn-primary" onClick={handleCreateCycle}>
              + New Cycle
            </button>
          </div>

          {cycles.length === 0 ? (
            <div className="sidebar-empty">
              <p>No research cycles yet.</p>
              <p className="sidebar-hint">
                Cycles group related experiments and notes together.
              </p>
            </div>
          ) : (
            <div className="cycles-list">
              {cycles.map(cycle => (
                <button
                  key={cycle.id}
                  className={`cycle-item ${selectedCycle?.id === cycle.id ? 'selected' : ''}`}
                  onClick={() => handleSelectCycle(cycle.id)}
                >
                  <div className="cycle-header">
                    <span className="cycle-number">Cycle #{cycle.sequence_number}</span>
                    <span
                      className="cycle-status-dot"
                      style={{ backgroundColor: getStatusColor(cycle.status) }}
                      title={cycle.status}
                    />
                  </div>
                  {cycle.summary_title && (
                    <div className="cycle-title">{cycle.summary_title}</div>
                  )}
                  <div className="cycle-meta">
                    <span>{cycle.experiment_count} experiments</span>
                    <span>{formatDate(cycle.created_at).split(',')[0]}</span>
                  </div>
                </button>
              ))}
            </div>
          )}

          <button
            className={`cycle-item all-entries ${!selectedCycle ? 'selected' : ''}`}
            onClick={() => setSelectedCycle(null)}
          >
            <div className="cycle-header">
              <span className="cycle-number">All Entries</span>
            </div>
            <div className="cycle-meta">
              <span>{entries.length} total notes</span>
            </div>
          </button>
        </div>

        {/* Main Content: Notebook Entries */}
        <div className="notebook-main">
          <div className="main-header">
            <div className="header-info">
              <h3>
                {selectedCycle
                  ? `Cycle #${selectedCycle.sequence_number} - ${selectedCycle.summary_title || 'Research Notes'}`
                  : 'All Lab Notebook Entries'}
              </h3>
              {selectedCycle && (
                <div className="cycle-details">
                  <StatusBadge status={selectedCycle.status} />
                  <span className="detail-separator">|</span>
                  <span>{selectedCycle.experiments.length} experiments</span>
                  <span className="detail-separator">|</span>
                  <span>Created {formatDate(selectedCycle.created_at)}</span>
                </div>
              )}
            </div>
            <button
              className="btn btn-primary"
              onClick={() => setIsAddEntryModalOpen(true)}
            >
              + Add Note
            </button>
          </div>

          {/* Selected Cycle Experiments */}
          {selectedCycle && selectedCycle.experiments.length > 0 && (
            <div className="cycle-experiments">
              <h4>Linked Experiments</h4>
              <div className="experiments-grid">
                {selectedCycle.experiments.map(exp => (
                  <div key={exp.id} className="experiment-summary">
                    <div className="exp-name">{exp.name}</div>
                    <div className="exp-meta">
                      <StatusBadge status={exp.status} />
                      {exp.best_metric !== null && exp.primary_metric && (
                        <span className="exp-metric">
                          {exp.primary_metric}: {exp.best_metric.toFixed(4)}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Notebook Entries List */}
          {isLoadingCycle ? (
            <LoadingSpinner message="Loading cycle details..." />
          ) : filteredEntries.length === 0 ? (
            <div className="entries-empty">
              <div className="empty-icon">📓</div>
              <h4>No notes yet</h4>
              <p>
                {selectedCycle
                  ? 'Add notes to document your findings and hypotheses for this research cycle.'
                  : 'Start documenting your ML research journey.'}
              </p>
              <button
                className="btn btn-primary"
                onClick={() => setIsAddEntryModalOpen(true)}
              >
                Add First Note
              </button>
            </div>
          ) : (
            <div className="entries-list">
              {filteredEntries.map(entry => (
                <div
                  key={entry.id}
                  className={`entry-card ${expandedEntryId === entry.id ? 'expanded' : ''}`}
                >
                  <div
                    className="entry-header"
                    onClick={() => setExpandedEntryId(
                      expandedEntryId === entry.id ? null : entry.id
                    )}
                  >
                    <div className="entry-title-row">
                      <span className={`entry-author ${entry.author_type}`}>
                        {entry.author_type === 'agent' ? '🤖' : '👤'}
                      </span>
                      <h4 className="entry-title">{entry.title}</h4>
                      <span className="entry-expand-icon">
                        {expandedEntryId === entry.id ? '▼' : '▶'}
                      </span>
                    </div>
                    <div className="entry-meta">
                      <span className="entry-date">{formatDate(entry.created_at)}</span>
                      {!selectedCycle && entry.research_cycle_id && (
                        <span className="entry-cycle-badge">
                          Cycle #{cycles.find(c => c.id === entry.research_cycle_id)?.sequence_number}
                        </span>
                      )}
                    </div>
                  </div>

                  {expandedEntryId === entry.id && (
                    <div className="entry-body">
                      {entry.body_markdown ? (
                        <div className="entry-content">{entry.body_markdown}</div>
                      ) : (
                        <div className="entry-content empty">No additional content</div>
                      )}
                      <div className="entry-actions">
                        <button
                          className="btn btn-small btn-danger"
                          onClick={(e) => {
                            e.stopPropagation();
                            setDeleteEntryTarget(entry);
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Add Entry Modal */}
      <Modal
        isOpen={isAddEntryModalOpen}
        onClose={() => {
          setIsAddEntryModalOpen(false);
          setNewEntryTitle('');
          setNewEntryBody('');
        }}
        title="Add Lab Notebook Entry"
        size="medium"
      >
        <div className="add-entry-form">
          <div className="form-group">
            <label htmlFor="entry-title">Title *</label>
            <input
              id="entry-title"
              type="text"
              className="form-control"
              placeholder="e.g., Initial hypothesis about feature importance"
              value={newEntryTitle}
              onChange={(e) => setNewEntryTitle(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label htmlFor="entry-body">Notes (Markdown supported)</label>
            <textarea
              id="entry-body"
              className="form-control"
              placeholder="Document your observations, hypotheses, or conclusions..."
              value={newEntryBody}
              onChange={(e) => setNewEntryBody(e.target.value)}
              rows={8}
            />
          </div>
          {selectedCycle && (
            <div className="form-info">
              This entry will be linked to Cycle #{selectedCycle.sequence_number}
            </div>
          )}
          <div className="form-actions">
            <button
              className="btn btn-secondary"
              onClick={() => {
                setIsAddEntryModalOpen(false);
                setNewEntryTitle('');
                setNewEntryBody('');
              }}
            >
              Cancel
            </button>
            <button
              className="btn btn-primary"
              onClick={handleCreateEntry}
              disabled={!newEntryTitle.trim() || isCreatingEntry}
            >
              {isCreatingEntry ? 'Adding...' : 'Add Entry'}
            </button>
          </div>
        </div>
      </Modal>

      {/* Delete Confirmation */}
      <ConfirmDialog
        isOpen={!!deleteEntryTarget}
        onClose={() => setDeleteEntryTarget(null)}
        onConfirm={handleDeleteEntry}
        title="Delete Entry"
        message={`Are you sure you want to delete "${deleteEntryTarget?.title}"?`}
        confirmLabel="Delete"
        variant="danger"
        isLoading={isDeleting}
      />
    </div>
  );
}

/**
 * Share Dialog Component
 * Modal for sharing projects
 */
import { useState, useEffect } from 'react';
import type { Share, ShareRole } from '../types/api';
import * as api from '../services/api';
import { ApiException } from '../services/api';

interface ShareDialogProps {
  isOpen: boolean;
  onClose: () => void;
  resourceType: 'project' | 'dataset';
  resourceId: string;
  resourceName: string;
}

export default function ShareDialog({
  isOpen,
  onClose,
  resourceType,
  resourceId,
  resourceName,
}: ShareDialogProps) {
  const [shares, setShares] = useState<Share[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // New share form
  const [newEmail, setNewEmail] = useState('');
  const [newRole, setNewRole] = useState<ShareRole>('viewer');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Load shares when dialog opens
  useEffect(() => {
    if (isOpen) {
      loadShares();
    }
  }, [isOpen, resourceId, resourceType]);

  const loadShares = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response =
        resourceType === 'project'
          ? await api.listProjectShares(resourceId)
          : await api.listDatasetShares(resourceId);
      setShares(response.items);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load shares');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddShare = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newEmail.trim()) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const newShare =
        resourceType === 'project'
          ? await api.shareProject(resourceId, { email: newEmail, role: newRole })
          : await api.shareDataset(resourceId, { email: newEmail, role: newRole });

      setShares([...shares, newShare]);
      setNewEmail('');
      setNewRole('viewer');
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to share');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRemoveShare = async (shareId: string) => {
    try {
      if (resourceType === 'project') {
        await api.removeProjectShare(resourceId, shareId);
      } else {
        await api.removeDatasetShare(resourceId, shareId);
      }
      setShares(shares.filter((s) => s.id !== shareId));
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to remove share');
      }
    }
  };

  const handleUpdateRole = async (shareId: string, role: ShareRole) => {
    if (resourceType !== 'project') return; // Only projects support role updates

    try {
      const updated = await api.updateProjectShare(resourceId, shareId, { role });
      setShares(shares.map((s) => (s.id === shareId ? updated : s)));
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to update role');
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog share-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <h2>Share "{resourceName}"</h2>
          <button className="dialog-close" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="dialog-content">
          {error && <div className="dialog-error">{error}</div>}

          {/* Add new share form */}
          <form onSubmit={handleAddShare} className="share-form">
            <div className="share-form-row">
              <input
                type="email"
                placeholder="Email address"
                value={newEmail}
                onChange={(e) => setNewEmail(e.target.value)}
                required
                className="share-email-input"
              />
              <select
                value={newRole}
                onChange={(e) => setNewRole(e.target.value as ShareRole)}
                className="share-role-select"
              >
                <option value="viewer">Viewer</option>
                <option value="editor">Editor</option>
                <option value="admin">Admin</option>
              </select>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={isSubmitting || !newEmail.trim()}
              >
                {isSubmitting ? 'Sharing...' : 'Share'}
              </button>
            </div>
          </form>

          {/* Existing shares list */}
          <div className="shares-list">
            <h3>People with access</h3>
            {isLoading ? (
              <div className="shares-loading">Loading...</div>
            ) : shares.length === 0 ? (
              <div className="shares-empty">No one else has access yet</div>
            ) : (
              <ul className="shares-items">
                {shares.map((share) => (
                  <li key={share.id} className="share-item">
                    <div className="share-user">
                      <span className="share-avatar">
                        {share.user_name?.[0] || share.invited_email?.[0] || '?'}
                      </span>
                      <div className="share-details">
                        <span className="share-name">
                          {share.user_name || share.invited_email || 'Unknown'}
                        </span>
                        {share.status === 'pending' && (
                          <span className="share-status-badge pending">Pending</span>
                        )}
                      </div>
                    </div>
                    <div className="share-actions">
                      {resourceType === 'project' ? (
                        <select
                          value={share.role}
                          onChange={(e) =>
                            handleUpdateRole(share.id, e.target.value as ShareRole)
                          }
                          className="share-role-select"
                        >
                          <option value="viewer">Viewer</option>
                          <option value="editor">Editor</option>
                          <option value="admin">Admin</option>
                        </select>
                      ) : (
                        <span className="share-role-label">{share.role}</span>
                      )}
                      <button
                        className="btn btn-icon btn-danger-icon"
                        onClick={() => handleRemoveShare(share.id)}
                        title="Remove access"
                      >
                        <svg width="16" height="16" viewBox="0 0 16 16">
                          <path
                            d="M4 4l8 8M12 4l-8 8"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            fill="none"
                          />
                        </svg>
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <div className="dialog-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

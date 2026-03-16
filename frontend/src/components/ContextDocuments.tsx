import { useState, useEffect, useRef } from 'react';
import type { ContextDocument, ContextDocumentListResponse, SupportedExtensionsResponse } from '../types/api';
import {
  listContextDocuments,
  uploadContextDocument,
  deleteContextDocument,
  toggleContextDocumentActive,
  getSupportedExtensions,
} from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ConfirmDialog from './ConfirmDialog';

interface ContextDocumentsProps {
  projectId: string;
  onClose?: () => void;
}

export default function ContextDocuments({ projectId, onClose }: ContextDocumentsProps) {
  const [documents, setDocuments] = useState<ContextDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [extensions, setExtensions] = useState<SupportedExtensionsResponse | null>(null);

  // Upload form state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [docName, setDocName] = useState('');
  const [explanation, setExplanation] = useState('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Delete confirmation
  const [deleteDoc, setDeleteDoc] = useState<ContextDocument | null>(null);

  useEffect(() => {
    loadDocuments();
    loadExtensions();
  }, [projectId]);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      const response = await listContextDocuments(projectId, true);
      setDocuments(response.documents);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const loadExtensions = async () => {
    try {
      const ext = await getSupportedExtensions();
      setExtensions(ext);
    } catch {
      // Ignore - not critical
    }
  };

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    if (file && !docName) {
      // Auto-fill name from filename (without extension)
      const baseName = file.name.replace(/\.[^/.]+$/, '');
      setDocName(baseName);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadError('Please select a file');
      return;
    }
    if (!docName.trim()) {
      setUploadError('Please enter a document name');
      return;
    }
    if (!explanation.trim() || explanation.length < 10) {
      setUploadError('Please provide an explanation (at least 10 characters)');
      return;
    }

    try {
      setUploading(true);
      setUploadError(null);
      await uploadContextDocument(projectId, selectedFile, docName, explanation);

      // Reset form
      setSelectedFile(null);
      setDocName('');
      setExplanation('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

      // Reload list
      await loadDocuments();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleToggleActive = async (doc: ContextDocument) => {
    try {
      const updated = await toggleContextDocumentActive(doc.id);
      setDocuments(docs => docs.map(d => d.id === updated.id ? updated : d));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update document');
    }
  };

  const handleDelete = async (doc: ContextDocument) => {
    try {
      await deleteContextDocument(doc.id);
      setDocuments(docs => docs.filter(d => d.id !== doc.id));
      setDeleteDoc(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete document');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getFileTypeIcon = (fileType: string) => {
    switch (fileType) {
      case 'pdf': return '📄';
      case 'docx': return '📝';
      case 'excel': return '📊';
      case 'csv': return '📈';
      case 'txt': return '📃';
      case 'md': return '📋';
      case 'html': return '🌐';
      case 'image': return '🖼️';
      default: return '📁';
    }
  };

  const getStatusBadge = (doc: ContextDocument) => {
    if (!doc.is_active) {
      return <span className="badge badge-secondary">Inactive</span>;
    }
    switch (doc.extraction_status) {
      case 'completed':
        return doc.has_content
          ? <span className="badge badge-success">Ready</span>
          : <span className="badge badge-warning">No Content</span>;
      case 'pending':
        return <span className="badge badge-info">Processing</span>;
      case 'failed':
        return <span className="badge badge-error">Failed</span>;
      default:
        return null;
    }
  };

  const acceptedExtensions = extensions
    ? Object.keys(extensions.extensions).join(',')
    : '.pdf,.docx,.doc,.xlsx,.xls,.csv,.txt,.md,.html,.png,.jpg,.jpeg';

  return (
    <div className="context-documents">
      <div className="context-documents-header">
        <h3>Context Documents</h3>
        <p className="text-muted">
          Upload supplementary documentation to help AI agents understand your problem better.
          Images will use your explanation only (no OCR).
        </p>
      </div>

      {error && <div className="form-error">{error}</div>}

      {/* Upload Form */}
      <div className="context-upload-form card">
        <h4>Upload Document</h4>

        {uploadError && <div className="form-error">{uploadError}</div>}

        <div className="form-group">
          <label>File</label>
          <input
            type="file"
            ref={fileInputRef}
            accept={acceptedExtensions}
            onChange={e => handleFileSelect(e.target.files?.[0] || null)}
            disabled={uploading}
          />
          {selectedFile && (
            <div className="selected-file">
              Selected: {selectedFile.name} ({formatFileSize(selectedFile.size)})
            </div>
          )}
          {extensions && extensions.max_file_size_mb && (
            <div className="help-text">
              Max size: {extensions.max_file_size_mb} MB
            </div>
          )}
        </div>

        <div className="form-group">
          <label>Document Name *</label>
          <input
            type="text"
            value={docName}
            onChange={e => setDocName(e.target.value)}
            placeholder="e.g., Business Requirements Document"
            disabled={uploading}
            maxLength={255}
          />
        </div>

        <div className="form-group">
          <label>Explanation * (How should the AI use this document?)</label>
          <textarea
            value={explanation}
            onChange={e => setExplanation(e.target.value)}
            placeholder="Explain what this document contains and how it should inform the AI's decisions about dataset design, feature selection, or model configuration..."
            disabled={uploading}
            rows={4}
            maxLength={5000}
          />
          <div className="help-text">
            {explanation.length}/5000 characters (min 10)
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleUpload}
          disabled={uploading || !selectedFile || !docName.trim() || explanation.length < 10}
        >
          {uploading ? <><LoadingSpinner size="small" /> Uploading...</> : 'Upload Document'}
        </button>
      </div>

      {/* Document List */}
      <div className="context-documents-list">
        <h4>Uploaded Documents ({documents.length})</h4>

        {loading ? (
          <LoadingSpinner />
        ) : documents.length === 0 ? (
          <div className="empty-state">
            <p>No context documents uploaded yet.</p>
            <p className="text-muted">
              Upload PDFs, Word docs, Excel spreadsheets, CSVs, text files, or images to provide additional context for AI agents.
            </p>
          </div>
        ) : (
          <div className="document-cards">
            {documents.map(doc => (
              <div key={doc.id} className={`document-card ${!doc.is_active ? 'inactive' : ''}`}>
                <div className="document-icon">{getFileTypeIcon(doc.file_type)}</div>
                <div className="document-info">
                  <div className="document-header">
                    <span className="document-name">{doc.name}</span>
                    {getStatusBadge(doc)}
                  </div>
                  <div className="document-meta">
                    {doc.original_filename} • {formatFileSize(doc.file_size_bytes)}
                  </div>
                  <div className="document-explanation">
                    {doc.explanation.length > 200
                      ? doc.explanation.substring(0, 200) + '...'
                      : doc.explanation}
                  </div>
                  {doc.extraction_error && (
                    <div className="document-error">
                      Error: {doc.extraction_error}
                    </div>
                  )}
                </div>
                <div className="document-actions">
                  <button
                    className={`btn btn-sm ${doc.is_active ? 'btn-outline' : 'btn-primary'}`}
                    onClick={() => handleToggleActive(doc)}
                    title={doc.is_active ? 'Deactivate' : 'Activate'}
                  >
                    {doc.is_active ? 'Deactivate' : 'Activate'}
                  </button>
                  <button
                    className="btn btn-sm btn-error"
                    onClick={() => setDeleteDoc(doc)}
                    title="Delete"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {onClose && (
        <div className="context-documents-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      )}

      {/* Delete Confirmation */}
      {deleteDoc && (
        <ConfirmDialog
          title="Delete Context Document"
          message={`Are you sure you want to delete "${deleteDoc.name}"? This cannot be undone.`}
          confirmLabel="Delete"
          confirmVariant="error"
          onConfirm={() => handleDelete(deleteDoc)}
          onCancel={() => setDeleteDoc(null)}
        />
      )}
    </div>
  );
}

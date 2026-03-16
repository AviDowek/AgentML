import { useState, useRef } from 'react';

// All supported file extensions
const ALL_SUPPORTED_EXTENSIONS = [
  // Tabular data
  '.csv', '.tsv',
  // Excel
  '.xlsx', '.xls', '.xlsm', '.xlsb',
  // JSON
  '.json', '.jsonl', '.ndjson',
  // Columnar formats
  '.parquet', '.pq', '.feather', '.arrow', '.ipc', '.orc',
  // Databases
  '.sqlite', '.sqlite3', '.db',
  // Statistical software
  '.sas7bdat', '.xpt', '.dta', '.sav', '.zsav', '.por',
  // Structured data
  '.xml', '.html', '.htm',
  // HDF5
  '.h5', '.hdf5', '.hdf',
  // Other
  '.pkl', '.pickle', '.fwf', '.dat', '.txt', '.log', '.docx', '.doc',
].join(',');

interface FileUploadProps {
  onUpload: (file: File, options?: { name?: string; delimiter?: string }) => Promise<void>;
  isLoading?: boolean;
  accept?: string;
}

export default function FileUpload({
  onUpload,
  isLoading = false,
  accept = ALL_SUPPORTED_EXTENSIONS,
}: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customName, setCustomName] = useState('');
  const [delimiter, setDelimiter] = useState(',');
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File | null) => {
    if (file) {
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file');
      return;
    }

    try {
      await onUpload(selectedFile, {
        name: customName || undefined,
        delimiter: selectedFile.name.endsWith('.csv') ? delimiter : undefined,
      });
      // Reset form on success
      setSelectedFile(null);
      setCustomName('');
      setDelimiter(',');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="file-upload">
      {error && <div className="form-error">{error}</div>}

      <div
        className={`file-dropzone ${isDragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          onChange={(e) => handleFileSelect(e.target.files?.[0] || null)}
          className="file-input-hidden"
          disabled={isLoading}
        />

        {selectedFile ? (
          <div className="file-selected">
            <span className="file-icon">📄</span>
            <div className="file-info">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">{formatFileSize(selectedFile.size)}</span>
            </div>
            <button
              type="button"
              className="file-remove"
              onClick={(e) => {
                e.stopPropagation();
                setSelectedFile(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
            >
              &times;
            </button>
          </div>
        ) : (
          <div className="file-placeholder">
            <span className="upload-icon">📁</span>
            <p>Drag and drop a file here, or click to browse</p>
            <p className="file-types">CSV, Excel, JSON, Parquet, SQLite, Feather, SAS, Stata, SPSS, XML, HTML, HDF5, and more</p>
          </div>
        )}
      </div>

      {selectedFile && (
        <div className="file-options">
          <div className="form-group">
            <label htmlFor="customName" className="form-label">
              Custom Name (optional)
            </label>
            <input
              type="text"
              id="customName"
              className="form-input"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              placeholder={selectedFile.name}
              disabled={isLoading}
            />
          </div>

          {selectedFile.name.endsWith('.csv') && (
            <div className="form-group">
              <label htmlFor="delimiter" className="form-label">
                CSV Delimiter
              </label>
              <select
                id="delimiter"
                className="form-select"
                value={delimiter}
                onChange={(e) => setDelimiter(e.target.value)}
                disabled={isLoading}
              >
                <option value=",">Comma (,)</option>
                <option value=";">Semicolon (;)</option>
                <option value="\t">Tab</option>
                <option value="|">Pipe (|)</option>
              </select>
            </div>
          )}
        </div>
      )}

      <div className="file-upload-actions">
        <button
          type="button"
          className="btn btn-primary"
          onClick={handleUpload}
          disabled={!selectedFile || isLoading}
        >
          {isLoading ? 'Uploading...' : 'Upload File'}
        </button>
      </div>
    </div>
  );
}

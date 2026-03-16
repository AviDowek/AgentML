import { useState, useEffect } from 'react';
import {
  getExperimentNotebook,
  downloadExperimentNotebook,
  type JupyterNotebook,
  type NotebookCell,
} from '../services/api';

interface NotebookViewerProps {
  experimentId: string;
  experimentName?: string;
}

function NotebookCellComponent({ cell, index }: { cell: NotebookCell; index: number }) {
  const isMarkdown = cell.cell_type === 'markdown';
  const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;

  return (
    <div
      className={`mb-4 rounded-lg border ${
        isMarkdown
          ? 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
          : 'border-blue-200 dark:border-blue-800 bg-gray-50 dark:bg-gray-900'
      }`}
    >
      <div className="flex items-center px-3 py-1 text-xs text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
        <span
          className={`px-2 py-0.5 rounded ${
            isMarkdown
              ? 'bg-gray-100 dark:bg-gray-700'
              : 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
          }`}
        >
          {isMarkdown ? 'Markdown' : 'Code'}
        </span>
        {!isMarkdown && cell.execution_count !== null && (
          <span className="ml-2">In [{cell.execution_count ?? ' '}]</span>
        )}
      </div>
      <div className="p-4">
        {isMarkdown ? (
          <div className="prose dark:prose-invert max-w-none text-sm">
            {source.split('\n').map((line, i) => {
              // Basic markdown rendering
              if (line.startsWith('# ')) {
                return (
                  <h1 key={i} className="text-2xl font-bold mb-2">
                    {line.slice(2)}
                  </h1>
                );
              }
              if (line.startsWith('## ')) {
                return (
                  <h2 key={i} className="text-xl font-semibold mb-2">
                    {line.slice(3)}
                  </h2>
                );
              }
              if (line.startsWith('### ')) {
                return (
                  <h3 key={i} className="text-lg font-medium mb-2">
                    {line.slice(4)}
                  </h3>
                );
              }
              if (line.startsWith('- ')) {
                return (
                  <li key={i} className="ml-4">
                    {line.slice(2)}
                  </li>
                );
              }
              if (line.startsWith('```')) {
                return null;
              }
              return (
                <p key={i} className="mb-1">
                  {line || '\u00A0'}
                </p>
              );
            })}
          </div>
        ) : (
          <pre className="text-sm font-mono text-gray-800 dark:text-gray-200 overflow-x-auto whitespace-pre-wrap">
            {source}
          </pre>
        )}
      </div>
    </div>
  );
}

export function NotebookViewer({ experimentId, experimentName }: NotebookViewerProps) {
  const [notebook, setNotebook] = useState<JupyterNotebook | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const fetchNotebook = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getExperimentNotebook(experimentId);
      setNotebook(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load notebook');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isExpanded && !notebook && !loading) {
      fetchNotebook();
    }
  }, [isExpanded, experimentId]);

  const handleDownload = () => {
    downloadExperimentNotebook(experimentId);
  };

  return (
    <div className="border rounded-lg border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
      {/* Header */}
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <svg
            className="w-6 h-6 text-orange-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <div>
            <h3 className="font-medium text-gray-900 dark:text-white">
              Reproducible Notebook
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Complete Python code to reproduce this experiment
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleDownload();
            }}
            className="px-3 py-1.5 text-sm bg-orange-500 hover:bg-orange-600 text-white rounded-md transition-colors flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
              />
            </svg>
            Download .ipynb
          </button>
          <svg
            className={`w-5 h-5 text-gray-500 transition-transform ${
              isExpanded ? 'rotate-180' : ''
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="border-t border-gray-200 dark:border-gray-700 p-4">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-500"></div>
              <span className="ml-3 text-gray-600 dark:text-gray-400">Loading notebook...</span>
            </div>
          )}

          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
              <p className="text-red-600 dark:text-red-400">{error}</p>
              <button
                onClick={fetchNotebook}
                className="mt-2 text-sm text-red-600 hover:text-red-800 dark:text-red-400 underline"
              >
                Retry
              </button>
            </div>
          )}

          {notebook && !loading && (
            <div className="max-h-[600px] overflow-y-auto">
              <div className="mb-4 p-3 bg-gray-100 dark:bg-gray-700 rounded-lg text-sm text-gray-600 dark:text-gray-300">
                <span className="font-medium">Kernel:</span>{' '}
                {notebook.metadata?.kernelspec?.display_name || 'Python 3'}
                {' | '}
                <span className="font-medium">Cells:</span> {notebook.cells?.length || 0}
              </div>
              {notebook.cells?.map((cell, index) => (
                <NotebookCellComponent key={index} cell={cell} index={index} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default NotebookViewer;

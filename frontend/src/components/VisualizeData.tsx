import { useState, useEffect, useCallback } from 'react';
import {
  generateVisualization,
  getVisualizationSuggestions,
  explainVisualization,
  saveVisualization,
  listSavedVisualizations,
  deleteSavedVisualization,
  generateDatasetSpecVisualization,
  getDatasetSpecVisualizationSuggestions,
  ApiException,
} from '../services/api';
import type { SavedVisualization } from '../services/api';
import type {
  VisualizationResponse,
  VisualizationSuggestion,
  DataSummary,
} from '../services/api';
import type { DataSource, DatasetSpec } from '../types/api';
import LoadingSpinner from './LoadingSpinner';

interface VisualizationItem {
  id: string;
  dbId?: string; // Database ID if saved
  title: string;
  description: string;
  chart_type: string;
  image_base64?: string;
  code: string;
  error?: string;
  isAiSuggested?: boolean;
  explanation?: string;
  isLoadingExplanation?: boolean;
  isSaved?: boolean;
  request?: string;
  dataSourceId?: string;
}

interface VisualizeDataProps {
  projectId: string;
  dataSources: DataSource[];
  datasetSpecs?: DatasetSpec[]; // Dataset specs (configured datasets with feature engineering)
  aiSuggestedDataSource?: DataSource; // AI-suggested version from pipeline
  onVisualizationsChange?: (visualizations: VisualizationItem[]) => void; // Callback to share visualizations
}

// Type for unified data source selection
type DataSourceSelection = {
  type: 'datasource' | 'datasetspec';
  id: string;
  name: string;
};

// Export the type for external use
export type { VisualizationItem };

export default function VisualizeData({
  projectId,
  dataSources,
  datasetSpecs = [],
  aiSuggestedDataSource,
  onVisualizationsChange,
}: VisualizeDataProps) {
  const [selectedSourceId, setSelectedSourceId] = useState<string>('');
  const [selectedSourceType, setSelectedSourceType] = useState<'datasource' | 'datasetspec'>('datasource');
  const [visualizations, setVisualizations] = useState<VisualizationItem[]>([]);
  const [suggestions, setSuggestions] = useState<VisualizationSuggestion[]>([]);
  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null);
  const [chatInput, setChatInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showComparison, setShowComparison] = useState(false);
  const [comparisonVisualizations, setComparisonVisualizations] = useState<VisualizationItem[]>([]);

  // Build combined list of data sources and dataset specs
  const allSources: DataSourceSelection[] = [
    ...dataSources.map(ds => ({ type: 'datasource' as const, id: ds.id, name: ds.name })),
    ...datasetSpecs.map(spec => ({ type: 'datasetspec' as const, id: spec.id, name: `📊 ${spec.name}` })),
  ];

  // Auto-select first source (prefer dataset specs if available)
  useEffect(() => {
    if (!selectedSourceId && allSources.length > 0) {
      // Prefer dataset specs over raw data sources
      const firstSpec = allSources.find(s => s.type === 'datasetspec');
      const firstSource = firstSpec || allSources[0];
      setSelectedSourceId(firstSource.id);
      setSelectedSourceType(firstSource.type);
    }
  }, [allSources, selectedSourceId]);

  // Load saved visualizations on mount
  const loadSavedVisualizations = useCallback(async () => {
    if (!projectId) return;

    try {
      const saved = await listSavedVisualizations(projectId);
      const loadedViz: VisualizationItem[] = saved.map((sv: SavedVisualization) => ({
        id: `saved-${sv.id}`,
        dbId: sv.id,
        title: sv.title,
        description: sv.description || '',
        chart_type: sv.chart_type || '',
        image_base64: sv.image_base64,
        code: sv.code,
        explanation: sv.explanation,
        isAiSuggested: sv.is_ai_suggested === 'true',
        isSaved: true,
        request: sv.request,
        dataSourceId: sv.data_source_id,
      }));
      setVisualizations(loadedViz);
    } catch (err) {
      console.error('Failed to load saved visualizations:', err);
    }
  }, [projectId]);

  useEffect(() => {
    loadSavedVisualizations();
  }, [loadSavedVisualizations]);

  // Notify parent when visualizations change
  useEffect(() => {
    if (onVisualizationsChange) {
      onVisualizationsChange(visualizations);
    }
  }, [visualizations, onVisualizationsChange]);

  // Load suggestions when source changes
  const loadSuggestions = useCallback(async () => {
    if (!selectedSourceId || !projectId) return;

    setIsLoadingSuggestions(true);
    setError(null);
    try {
      let response;
      if (selectedSourceType === 'datasetspec') {
        response = await getDatasetSpecVisualizationSuggestions(projectId, selectedSourceId);
      } else {
        response = await getVisualizationSuggestions(projectId, selectedSourceId);
      }
      setSuggestions(response.suggestions);
      setDataSummary(response.data_summary);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load visualization suggestions');
      }
    } finally {
      setIsLoadingSuggestions(false);
    }
  }, [projectId, selectedSourceId, selectedSourceType]);

  useEffect(() => {
    loadSuggestions();
  }, [loadSuggestions]);

  const handleGenerateVisualization = async (request: string, isFromSuggestion = false) => {
    if (!selectedSourceId || !projectId) return;

    setIsGenerating(true);
    setError(null);

    try {
      const previousViz = visualizations.slice(0, 3).map((v) => ({
        title: v.title,
        description: v.description,
      }));

      let response: VisualizationResponse;
      if (selectedSourceType === 'datasetspec') {
        response = await generateDatasetSpecVisualization(projectId, {
          dataset_spec_id: selectedSourceId,
          request,
          previous_visualizations: previousViz.length > 0 ? previousViz : undefined,
        });
      } else {
        response = await generateVisualization(projectId, {
          data_source_id: selectedSourceId,
          request,
          previous_visualizations: previousViz.length > 0 ? previousViz : undefined,
        });
      }

      // Save to database automatically (only for data sources, not dataset specs for now)
      let savedViz: SavedVisualization | null = null;
      if (selectedSourceType === 'datasource') {
        try {
          savedViz = await saveVisualization(projectId, {
            data_source_id: selectedSourceId,
            title: response.title,
            description: response.description,
            chart_type: response.chart_type,
            request: request,
            code: response.code,
            image_base64: response.image_base64,
            is_ai_suggested: isFromSuggestion ? 'true' : 'false',
          });
        } catch (saveErr) {
          console.error('Failed to save visualization to database:', saveErr);
        }
      }

      const newViz: VisualizationItem = {
        id: savedViz ? `saved-${savedViz.id}` : `viz-${Date.now()}`,
        dbId: savedViz?.id,
        title: response.title,
        description: response.description,
        chart_type: response.chart_type,
        image_base64: response.image_base64,
        code: response.code,
        error: response.error,
        isAiSuggested: isFromSuggestion,
        isSaved: !!savedViz,
        request: request,
        dataSourceId: selectedSourceId,
      };

      setVisualizations((prev) => [newViz, ...prev]);
      setChatInput('');
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to generate visualization');
      }
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRemoveVisualization = async (vizId: string, dbId?: string) => {
    // Delete from database if saved
    if (dbId && projectId) {
      try {
        await deleteSavedVisualization(projectId, dbId);
      } catch (err) {
        console.error('Failed to delete visualization from database:', err);
      }
    }
    setVisualizations((prev) => prev.filter((v) => v.id !== vizId));
  };

  const handleExplainVisualization = async (viz: VisualizationItem) => {
    if (!selectedSourceId || !projectId) return;

    // Update loading state for this visualization
    setVisualizations((prev) =>
      prev.map((v) => (v.id === viz.id ? { ...v, isLoadingExplanation: true } : v))
    );

    try {
      const response = await explainVisualization(projectId, {
        data_source_id: selectedSourceId,
        visualization_info: {
          title: viz.title,
          description: viz.description,
          chart_type: viz.chart_type,
        },
      });

      setVisualizations((prev) =>
        prev.map((v) =>
          v.id === viz.id
            ? { ...v, explanation: response.explanation, isLoadingExplanation: false }
            : v
        )
      );
    } catch (err) {
      setVisualizations((prev) =>
        prev.map((v) => (v.id === viz.id ? { ...v, isLoadingExplanation: false } : v))
      );
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    }
  };

  const handleComparisonToggle = async () => {
    if (!showComparison && aiSuggestedDataSource && comparisonVisualizations.length === 0) {
      // Generate visualizations for AI-suggested data source
      setIsGenerating(true);
      try {
        const response = await getVisualizationSuggestions(projectId, aiSuggestedDataSource.id);

        // Generate the first suggestion for comparison
        if (response.suggestions.length > 0) {
          const vizResponse = await generateVisualization(projectId, {
            data_source_id: aiSuggestedDataSource.id,
            request: response.suggestions[0].request,
          });

          setComparisonVisualizations([
            {
              id: `comparison-${Date.now()}`,
              title: vizResponse.title,
              description: vizResponse.description,
              chart_type: vizResponse.chart_type,
              image_base64: vizResponse.image_base64,
              code: vizResponse.code,
              error: vizResponse.error,
              isAiSuggested: true,
            },
          ]);
        }
      } catch (err) {
        if (err instanceof ApiException) {
          setError(err.detail);
        }
      } finally {
        setIsGenerating(false);
      }
    }
    setShowComparison(!showComparison);
  };

  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (chatInput.trim() && !isGenerating) {
      handleGenerateVisualization(chatInput.trim());
    }
  };

  // Handle source selection change
  const handleSourceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    const [type, id] = value.split(':') as ['datasource' | 'datasetspec', string];
    setSelectedSourceId(id);
    setSelectedSourceType(type);
    // Clear suggestions when source changes
    setSuggestions([]);
    setDataSummary(null);
  };

  const selectedSource = allSources.find((s) => s.id === selectedSourceId);

  return (
    <div className="visualize-data-container">
      {/* Header with data source selector */}
      <div className="visualize-header">
        <div className="visualize-header-left">
          <h3>Data Visualization</h3>
          <select
            value={`${selectedSourceType}:${selectedSourceId}`}
            onChange={handleSourceChange}
            className="data-source-select"
          >
            {datasetSpecs.length > 0 && (
              <optgroup label="Configured Datasets">
                {datasetSpecs.map((spec) => (
                  <option key={spec.id} value={`datasetspec:${spec.id}`}>
                    📊 {spec.name}
                  </option>
                ))}
              </optgroup>
            )}
            <optgroup label="Raw Data Sources">
              {dataSources.map((ds) => (
                <option key={ds.id} value={`datasource:${ds.id}`}>
                  {ds.name}
                </option>
              ))}
            </optgroup>
          </select>
        </div>

        {aiSuggestedDataSource && (
          <button
            type="button"
            className={`btn btn-secondary comparison-toggle ${showComparison ? 'active' : ''}`}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleComparisonToggle();
            }}
          >
            {showComparison ? 'Hide Comparison' : 'Compare with AI Data'}
          </button>
        )}
      </div>

      {error && (
        <div className="visualize-error">
          {error}
          <button type="button" onClick={() => setError(null)} className="error-dismiss">
            Dismiss
          </button>
        </div>
      )}

      {/* Data summary */}
      {dataSummary && (
        <div className="data-summary-panel">
          <div className="data-summary-stats">
            <span>{dataSummary.row_count.toLocaleString()} rows</span>
            <span>{dataSummary.column_count} columns</span>
            <span>
              {dataSummary.columns.filter((c) => c.is_numeric).length} numeric,{' '}
              {dataSummary.columns.filter((c) => !c.is_numeric).length} categorical
            </span>
          </div>
        </div>
      )}

      {/* AI Suggestions */}
      {isLoadingSuggestions ? (
        <div className="suggestions-loading">
          <LoadingSpinner message="Loading visualization suggestions..." />
        </div>
      ) : (
        suggestions.length > 0 && (
          <div className="suggestions-panel">
            <h4>Suggested Visualizations</h4>
            <div className="suggestions-grid">
              {suggestions.map((suggestion, index) => {
                // Use request if available, otherwise fall back to description or title
                const requestText = suggestion.request || suggestion.description || suggestion.title || '';
                return (
                  <button
                    key={index}
                    type="button"
                    className="suggestion-card"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      if (requestText.trim()) {
                        handleGenerateVisualization(requestText, true);
                      } else {
                        console.error('No valid request text for suggestion:', suggestion);
                        setError('Invalid suggestion: no request text available');
                      }
                    }}
                    disabled={isGenerating || !requestText.trim()}
                  >
                    <div className="suggestion-title">{suggestion.title || 'Untitled'}</div>
                    <div className="suggestion-description">{suggestion.description || 'No description'}</div>
                    <div className="suggestion-type">{suggestion.chart_type || 'chart'}</div>
                  </button>
                );
              })}
            </div>
          </div>
        )
      )}

      {/* Chat interface for custom requests */}
      <div className="visualize-chat">
        <form onSubmit={handleChatSubmit} className="visualize-chat-form">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="Describe what you want to visualize... (e.g., 'Show distribution of sales by region')"
            disabled={isGenerating || !selectedSourceId}
            className="visualize-chat-input"
          />
          <button
            type="submit"
            disabled={isGenerating || !chatInput.trim() || !selectedSourceId}
            className="visualize-chat-send"
          >
            {isGenerating ? (
              <span className="spinner spinner-small"></span>
            ) : (
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            )}
          </button>
        </form>
      </div>

      {/* Visualizations grid with comparison view */}
      <div className={`visualizations-container ${showComparison ? 'comparison-mode' : ''}`}>
        {showComparison && aiSuggestedDataSource && (
          <div className="comparison-panel">
            <h4>AI-Suggested Data: {aiSuggestedDataSource.name}</h4>
            <div className="visualization-grid">
              {comparisonVisualizations.map((viz) => (
                <VisualizationCard
                  key={viz.id}
                  visualization={viz}
                  onRemove={() =>
                    setComparisonVisualizations((prev) => prev.filter((v) => v.id !== viz.id))
                  }
                  onExplain={() => handleExplainVisualization(viz)}
                />
              ))}
            </div>
          </div>
        )}

        <div className={`main-visualizations ${showComparison ? 'with-comparison' : ''}`}>
          {showComparison && <h4>Original Data: {selectedSource?.name}</h4>}
          <div className="visualization-grid">
            {visualizations.length === 0 && !isGenerating && (
              <div className="no-visualizations">
                <p>No visualizations yet</p>
                <p>Click a suggestion above or type a request to create one</p>
              </div>
            )}
            {visualizations.map((viz) => (
              <VisualizationCard
                key={viz.id}
                visualization={viz}
                onRemove={() => handleRemoveVisualization(viz.id, viz.dbId)}
                onExplain={() => handleExplainVisualization(viz)}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// Sub-component for individual visualization cards
interface VisualizationCardProps {
  visualization: VisualizationItem;
  onRemove: () => void;
  onExplain: () => void;
}

function VisualizationCard({ visualization, onRemove, onExplain }: VisualizationCardProps) {
  const [showCode, setShowCode] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  return (
    <div className="visualization-card">
      <div className="visualization-card-header">
        <div className="visualization-title-row">
          <h5>{visualization.title}</h5>
          {visualization.isAiSuggested && (
            <span className="ai-badge" title="AI Suggested">
              AI
            </span>
          )}
        </div>
        <button type="button" className="remove-btn" onClick={onRemove} title="Remove visualization">
          &times;
        </button>
      </div>

      <p className="visualization-description">{visualization.description}</p>

      {visualization.error ? (
        <div className="visualization-error">
          <p>Error generating visualization:</p>
          <code>{visualization.error}</code>
        </div>
      ) : visualization.image_base64 ? (
        <div className="visualization-image">
          <img
            src={`data:image/png;base64,${visualization.image_base64}`}
            alt={visualization.title}
            onError={(e) => {
              console.error('Failed to load visualization image');
              e.currentTarget.style.display = 'none';
            }}
          />
        </div>
      ) : visualization.code ? (
        <div className="visualization-error">
          <p>Visualization code was generated but no image was produced.</p>
          <p>Check the code below for errors or try a different visualization request.</p>
        </div>
      ) : (
        <div className="visualization-loading">
          <span className="spinner"></span>
          <p>Generating...</p>
        </div>
      )}

      <div className="visualization-actions">
        <button
          type="button"
          className="btn btn-secondary btn-small"
          onClick={() => setShowCode(!showCode)}
        >
          {showCode ? 'Hide Code' : 'Show Code'}
        </button>
        <button
          type="button"
          className="btn btn-secondary btn-small"
          onClick={() => {
            if (!visualization.explanation && !visualization.isLoadingExplanation) {
              onExplain();
            }
            setShowExplanation(!showExplanation);
          }}
          disabled={visualization.isLoadingExplanation}
        >
          {visualization.isLoadingExplanation ? (
            <span className="spinner spinner-small"></span>
          ) : showExplanation ? (
            'Hide Explanation'
          ) : (
            'Explain'
          )}
        </button>
        <span className="chart-type-badge">{visualization.chart_type}</span>
      </div>

      {showCode && (
        <div className="visualization-code">
          <pre>
            <code>{visualization.code}</code>
          </pre>
        </div>
      )}

      {showExplanation && visualization.explanation && (
        <div className="visualization-explanation">
          <h6>What this shows:</h6>
          <p>{visualization.explanation}</p>
        </div>
      )}
    </div>
  );
}

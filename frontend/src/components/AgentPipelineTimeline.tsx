import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import type { AgentRun, AgentStep, AgentStepType, DataSource, OrchestrationOptions, OrchestrationMode, DebateMode } from '../types/api';
import type { DiscoveredDataset } from '../types/api';
import {
  listAgentRuns,
  getAgentRun,
  runSetupPipeline,
  applyDatasetSpecFromStep,
  applyDatasetSpecsBatch,
  applyExperimentsBatch,
  runDatasetDiscovery,
  applyDiscoveredDatasets,
  getOrchestrationOptions,
  cancelAgentRun,
  ApiException,
} from '../services/api';
import DiscoveredDatasetsList from './DiscoveredDatasetsList';
import StatusBadge from './StatusBadge';
import AgentStepDrawer from './AgentStepDrawer';
import LoadingSpinner from './LoadingSpinner';

interface AgentPipelineTimelineProps {
  projectId: string;
  dataSources: DataSource[];
  onPipelineComplete?: () => void;
  onAgentRunChange?: (agentRun: AgentRun | null) => void; // Callback to share agent run data
  // Optional default orchestration settings from parent
  defaultPMMode?: boolean;
  defaultDebateMode?: boolean;
}

// Export the AgentRun type for external use
export type { AgentRun, AgentStep };

// Dataset variant type from the pipeline
interface DatasetVariant {
  name: string;
  description: string;
  feature_columns: string[];
  excluded_columns: string[];
  exclusion_reasons: Record<string, string>;
  train_test_split: string;
  preprocessing_strategy: string;
  suggested_filters?: Record<string, unknown>;
  expected_tradeoff: string;
}

// Step metadata for display
const STEP_INFO: Record<AgentStepType, { number: number; name: string; role: string; icon: string }> = {
  data_analysis: {
    number: 1,
    name: 'Data Analysis',
    role: 'Data Analyst',
    icon: '📋',
  },
  problem_understanding: {
    number: 2,
    name: 'Problem Understanding',
    role: 'Planner',
    icon: '🎯',
  },
  data_audit: {
    number: 3,
    name: 'Data Audit',
    role: 'Data Auditor',
    icon: '🔍',
  },
  dataset_design: {
    number: 4,
    name: 'Dataset Design',
    role: 'Dataset Designer',
    icon: '📊',
  },
  dataset_validation: {
    number: 5,
    name: 'Dataset Validation',
    role: 'Data Validator',
    icon: '✓',
  },
  experiment_design: {
    number: 6,
    name: 'Experiment Design',
    role: 'ML Engineer',
    icon: '🧪',
  },
  plan_critic: {
    number: 7,
    name: 'Plan Review',
    role: 'Critic',
    icon: '✅',
  },
  // Dataset discovery (used before main pipeline)
  dataset_discovery: {
    number: 0,
    name: 'Dataset Discovery',
    role: 'Data Scout',
    icon: '🔍',
  },
  // Results pipeline steps (shown on experiment page, not here)
  results_interpretation: {
    number: 1,
    name: 'Results Interpretation',
    role: 'Analyst',
    icon: '📊',
  },
  results_critic: {
    number: 2,
    name: 'Results Critic',
    role: 'Reviewer',
    icon: '🔍',
  },
  // Data Architect pipeline steps (shown in DataArchitectPipeline component)
  dataset_inventory: {
    number: 1,
    name: 'Dataset Inventory',
    role: 'Data Profiler',
    icon: '📦',
  },
  relationship_discovery: {
    number: 2,
    name: 'Relationship Discovery',
    role: 'Data Analyst',
    icon: '🔗',
  },
  training_dataset_planning: {
    number: 3,
    name: 'Training Dataset Planning',
    role: 'Data Architect',
    icon: '📐',
  },
  training_dataset_build: {
    number: 4,
    name: 'Training Dataset Build',
    role: 'Data Engineer',
    icon: '🔧',
  },
  // Auto-improve pipeline steps (simple)
  improvement_analysis: {
    number: 1,
    name: 'Improvement Analysis',
    role: 'ML Analyst',
    icon: '🔬',
  },
  improvement_plan: {
    number: 2,
    name: 'Improvement Plan',
    role: 'ML Engineer',
    icon: '📋',
  },
  // Enhanced improvement pipeline (full agent)
  iteration_context: {
    number: 1,
    name: 'Iteration Context',
    role: 'Historian',
    icon: '📚',
  },
  improvement_data_analysis: {
    number: 2,
    name: 'Data Re-Analysis',
    role: 'Data Scientist',
    icon: '🔬',
  },
  improvement_dataset_design: {
    number: 3,
    name: 'Dataset Redesign',
    role: 'Feature Engineer',
    icon: '🛠️',
  },
  improvement_experiment_design: {
    number: 4,
    name: 'Experiment Design',
    role: 'ML Architect',
    icon: '📐',
  },
  // Lab notebook summary step
  lab_notebook_summary: {
    number: 1,
    name: 'Lab Notebook Summary',
    role: 'Lab Notebook Agent',
    icon: '📓',
  },
  // Robustness audit step
  robustness_audit: {
    number: 1,
    name: 'Robustness Audit',
    role: 'Robustness Auditor',
    icon: '🔍',
  },
  // Orchestration system steps
  project_manager: {
    number: 0,
    name: 'Project Manager',
    role: 'Orchestrator',
    icon: '🎭',
  },
  gemini_critique: {
    number: 0,
    name: 'Gemini Critique',
    role: 'Critic (Gemini)',
    icon: '💎',
  },
  openai_judge: {
    number: 0,
    name: 'OpenAI Judge',
    role: 'Judge (OpenAI)',
    icon: '⚖️',
  },
  debate_round: {
    number: 0,
    name: 'Debate Round',
    role: 'Debate',
    icon: '💬',
  },
};

// Helper to get next step type
const STEP_ORDER: AgentStepType[] = [
  'data_analysis',
  'problem_understanding',
  'data_audit',
  'dataset_design',
  'dataset_validation',
  'experiment_design',
  'plan_critic',
];

// Debate-related step types
const DEBATE_STEP_TYPES: AgentStepType[] = ['gemini_critique', 'openai_judge', 'debate_round'];

// Check if a step type is debate-related
const isDebateStep = (stepType: AgentStepType): boolean => DEBATE_STEP_TYPES.includes(stepType);

export default function AgentPipelineTimeline({
  projectId,
  dataSources,
  onPipelineComplete,
  onAgentRunChange,
  defaultPMMode = false,
  defaultDebateMode = false,
}: AgentPipelineTimelineProps) {
  const navigate = useNavigate();
  const [agentRun, setAgentRun] = useState<AgentRun | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [selectedStep, setSelectedStep] = useState<AgentStep | null>(null);

  // Track whether this is the initial load - we don't notify on initial load
  // to prevent infinite loops when opening a project with a completed run
  const isInitialLoadRef = useRef(true);
  // Track if we've already notified for current run to avoid duplicate calls
  const lastNotifiedRunIdRef = useRef<string | null>(null);

  // Form state for starting pipeline
  const [showStartForm, setShowStartForm] = useState(false);
  const [selectedDataSource, setSelectedDataSource] = useState<string>('');
  const [description, setDescription] = useState('');

  // Orchestration options state - use defaults from props
  const [orchestrationOptions, setOrchestrationOptions] = useState<OrchestrationOptions | null>(null);
  const [orchestrationMode, setOrchestrationMode] = useState<OrchestrationMode>(defaultPMMode ? 'project_manager' : 'sequential');
  const [debateMode, setDebateMode] = useState<DebateMode>(defaultDebateMode ? 'enabled' : 'disabled');
  const [judgeModel, setJudgeModel] = useState<string>('');
  const [maxDebateRounds, setMaxDebateRounds] = useState<number>(3);
  const [useContextDocuments, setUseContextDocuments] = useState<boolean>(true);
  const [contextABTesting, setContextABTesting] = useState<boolean>(false);

  // Sync orchestration mode with parent props when they change
  useEffect(() => {
    setOrchestrationMode(defaultPMMode ? 'project_manager' : 'sequential');
  }, [defaultPMMode]);

  useEffect(() => {
    setDebateMode(defaultDebateMode ? 'enabled' : 'disabled');
  }, [defaultDebateMode]);

  // Action state for bottom buttons
  const [isCreatingDataset, setIsCreatingDataset] = useState(false);
  const [isCreatingExperiment, setIsCreatingExperiment] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);
  const [autoRunExperiments, setAutoRunExperiments] = useState(true);  // Toggle for auto-running experiments

  // Confirmation modal state - now supports multiple variants
  const [showDatasetConfirmation, setShowDatasetConfirmation] = useState(false);
  const [datasetVariants, setDatasetVariants] = useState<DatasetVariant[]>([]);
  const [selectedVariants, setSelectedVariants] = useState<Set<string>>(new Set());
  const [recommendedVariant, setRecommendedVariant] = useState<string>('');
  const [variantsReasoning, setVariantsReasoning] = useState<string>('');
  const [variantsWarnings, setVariantsWarnings] = useState<string[]>([]);
  // Legacy single-variant state (for backward compatibility)
  const [datasetSpecPreview, setDatasetSpecPreview] = useState<{
    target_column: string;
    feature_columns: string[];
    filters_json?: Record<string, unknown>;
  } | null>(null);
  const [editableTargetColumn, setEditableTargetColumn] = useState('');
  const [editableFeatureColumns, setEditableFeatureColumns] = useState<string[]>([]);

  // Dataset discovery state (for "Find More Data" feature in data_analysis step)
  const [showDatasetDiscovery, setShowDatasetDiscovery] = useState(false);
  const [isDiscoveringDatasets, setIsDiscoveringDatasets] = useState(false);
  const [discoveredDatasets, setDiscoveredDatasets] = useState<DiscoveredDataset[]>([]);
  const [discoveryRunId, setDiscoveryRunId] = useState<string | null>(null);
  const [isApplyingDatasets, setIsApplyingDatasets] = useState(false);
  const [discoveryError, setDiscoveryError] = useState<string | null>(null);

  // Fetch latest agent run
  const fetchLatestRun = useCallback(async () => {
    try {
      const runList = await listAgentRuns(projectId, 0, 1);
      if (runList.items.length > 0) {
        const latestRun = await getAgentRun(runList.items[0].id);
        setAgentRun(latestRun);
        // Don't notify on initial load - only notify when a run transitions to completed
        // This is handled by the polling useEffect
      } else {
        setAgentRun(null);
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    } finally {
      setIsLoading(false);
      isInitialLoadRef.current = false;
    }
  }, [projectId]);

  useEffect(() => {
    fetchLatestRun();
  }, [fetchLatestRun]);

  // Notify parent when agentRun changes
  useEffect(() => {
    if (onAgentRunChange) {
      onAgentRunChange(agentRun);
    }
  }, [agentRun, onAgentRunChange]);

  // Fetch orchestration options on mount
  useEffect(() => {
    const fetchOrchestrationOptions = async () => {
      try {
        const options = await getOrchestrationOptions();
        setOrchestrationOptions(options);
        setJudgeModel(options.default_judge_model);
        setMaxDebateRounds(options.default_max_debate_rounds || 3);
      } catch (err) {
        // Non-critical - just log and continue without orchestration options
        console.warn('Failed to fetch orchestration options:', err);
      }
    };
    fetchOrchestrationOptions();
  }, []);

  // Poll for updates when pipeline is running
  useEffect(() => {
    if (!agentRun) return;
    if (agentRun.status !== 'running' && agentRun.status !== 'pending') return;

    const interval = setInterval(async () => {
      try {
        const updatedRun = await getAgentRun(agentRun.id);
        setAgentRun(updatedRun);

        // Update selected step if it's part of this run
        if (selectedStep && updatedRun.steps) {
          const updated = updatedRun.steps.find(s => s.id === selectedStep.id);
          if (updated) setSelectedStep(updated);
        }

        // Only notify completion once per run (when status transitions to completed)
        if (
          updatedRun.status === 'completed' &&
          onPipelineComplete &&
          lastNotifiedRunIdRef.current !== updatedRun.id
        ) {
          lastNotifiedRunIdRef.current = updatedRun.id;
          onPipelineComplete();
        }
      } catch {
        // Ignore polling errors
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [agentRun, selectedStep, onPipelineComplete]);

  const handleStartPipeline = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDataSource || !description.trim()) return;

    setIsStarting(true);
    setError(null);

    try {
      const response = await runSetupPipeline(projectId, {
        data_source_id: selectedDataSource,
        description: description.trim(),
        run_async: false, // Run synchronously for better UX
        orchestration_mode: orchestrationMode,
        debate_mode: debateMode,
        judge_model: debateMode === 'enabled' ? judgeModel : undefined,
        max_debate_rounds: debateMode === 'enabled' ? maxDebateRounds : undefined,
        use_context_documents: useContextDocuments,
        context_ab_testing: contextABTesting,
      });

      // Fetch the created run
      const newRun = await getAgentRun(response.run_id);
      setAgentRun(newRun);
      setShowStartForm(false);
      setDescription('');
      setSelectedDataSource('');
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to start pipeline');
      }
    } finally {
      setIsStarting(false);
    }
  };

  const handleCancelPipeline = async () => {
    if (!agentRun) return;

    setIsCancelling(true);
    setError(null);

    try {
      await cancelAgentRun(agentRun.id);
      // Immediately update local state to show cancelled status
      setAgentRun({ ...agentRun, status: 'cancelled' });
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to cancel pipeline');
      }
    } finally {
      setIsCancelling(false);
    }
  };

  const getNextStepName = (currentStep: AgentStep): string | null => {
    const currentIndex = STEP_ORDER.indexOf(currentStep.step_type);
    if (currentIndex < 0 || currentIndex >= STEP_ORDER.length - 1) return null;
    const nextType = STEP_ORDER[currentIndex + 1];
    return STEP_INFO[nextType].name;
  };

  // Get completed dataset_design step
  const getDatasetDesignStep = (): AgentStep | null => {
    if (!agentRun?.steps) return null;
    const step = agentRun.steps.find(
      s => s.step_type === 'dataset_design' && s.status === 'completed'
    );
    return step ?? null;
  };

  // Get completed experiment_design step
  const getExperimentDesignStep = (): AgentStep | null => {
    if (!agentRun?.steps) return null;
    const step = agentRun.steps.find(
      s => s.step_type === 'experiment_design' && s.status === 'completed'
    );
    return step ?? null;
  };

  // Get completed data_analysis step
  const getDataAnalysisStep = (): AgentStep | null => {
    if (!agentRun?.steps) return null;
    const step = agentRun.steps.find(
      s => s.step_type === 'data_analysis' && s.status === 'completed'
    );
    return step ?? null;
  };

  // Handle "Find More Data" - triggers dataset discovery
  const handleFindMoreData = async () => {
    const dataAnalysisStep = getDataAnalysisStep();
    if (!dataAnalysisStep) return;

    setShowDatasetDiscovery(true);
    setIsDiscoveringDatasets(true);
    setDiscoveryError(null);
    setDiscoveredDatasets([]);

    try {
      const projectDescription = description || (agentRun?.config_json as { description?: string })?.description || 'ML project';
      const response = await runDatasetDiscovery(projectId, {
        project_description: projectDescription,
      });
      setDiscoveryRunId(response.run_id);

      // Fetch the discovered datasets from the completed run
      const runDetails = await getAgentRun(response.run_id);
      const datasets = (runDetails.result_json as { discovered_datasets?: DiscoveredDataset[] })?.discovered_datasets || [];
      setDiscoveredDatasets(datasets);
    } catch (err) {
      if (err instanceof ApiException) {
        setDiscoveryError(err.detail);
      } else {
        setDiscoveryError('Failed to search for datasets');
      }
    } finally {
      setIsDiscoveringDatasets(false);
    }
  };

  // Handle applying discovered datasets
  const handleApplyDiscoveredDatasets = async (selectedIndices: number[]) => {
    if (!discoveryRunId || selectedIndices.length === 0) return;

    setIsApplyingDatasets(true);
    setDiscoveryError(null);

    try {
      await applyDiscoveredDatasets(projectId, discoveryRunId, {
        dataset_indices: selectedIndices,
      });
      setShowDatasetDiscovery(false);
      setDiscoveredDatasets([]);
      setDiscoveryRunId(null);
      onPipelineComplete?.(); // Refresh to show new data sources
    } catch (err) {
      if (err instanceof ApiException) {
        setDiscoveryError(err.detail);
      } else {
        setDiscoveryError('Failed to download datasets');
      }
    } finally {
      setIsApplyingDatasets(false);
    }
  };

  // Handle closing dataset discovery without applying
  const handleCloseDatasetDiscovery = () => {
    setShowDatasetDiscovery(false);
    setDiscoveredDatasets([]);
    setDiscoveryRunId(null);
    setDiscoveryError(null);
  };

  // Handle showing dataset confirmation with preview - supports multiple variants
  const handleShowDatasetConfirmation = () => {
    const step = getDatasetDesignStep();
    if (!step?.output_json) return;

    // Check if output has multiple variants (new format)
    const variants = step.output_json.variants as DatasetVariant[] | undefined;
    if (variants && variants.length > 0) {
      // New multi-variant format
      setDatasetVariants(variants);
      const recommended = step.output_json.recommended_variant as string || variants[0].name;
      setRecommendedVariant(recommended);
      // Pre-select the recommended variant
      setSelectedVariants(new Set([recommended]));
      setVariantsReasoning(step.output_json.reasoning as string || '');
      // Handle both string warnings and object warnings with {issue, severity, recommendation}
      const rawWarnings = step.output_json.warnings as (string | { issue?: string })[] || [];
      const normalizedWarnings = rawWarnings.map(w =>
        typeof w === 'string' ? w : (w.issue || JSON.stringify(w))
      );
      setVariantsWarnings(normalizedWarnings);
      // Clear legacy state
      setDatasetSpecPreview(null);
    } else {
      // Legacy single-variant format (backward compatibility)
      const preview = {
        target_column: step.output_json.target_column as string || '',
        feature_columns: (step.output_json.feature_columns as string[]) || [],
        filters_json: step.output_json.filters_json as Record<string, unknown> | undefined,
      };
      setDatasetSpecPreview(preview);
      setEditableTargetColumn(preview.target_column);
      setEditableFeatureColumns([...preview.feature_columns]);
      // Clear multi-variant state
      setDatasetVariants([]);
      setSelectedVariants(new Set());
    }

    setShowDatasetConfirmation(true);
    setActionError(null);
    setActionSuccess(null);
  };

  // Handle confirming and creating dataset spec with user modifications
  const handleConfirmDatasetSpec = async () => {
    const step = getDatasetDesignStep();
    if (!step) return;

    setIsCreatingDataset(true);
    setActionError(null);

    try {
      // Check if we're using multi-variant mode
      if (datasetVariants.length > 0 && selectedVariants.size > 0) {
        // Batch create from selected variants
        const variantNames = Array.from(selectedVariants);
        const response = await applyDatasetSpecsBatch(projectId, step.id, variantNames);

        const createdCount = response.dataset_specs.length;
        if (createdCount > 0) {
          setActionSuccess(`Created ${createdCount} DatasetSpec${createdCount !== 1 ? 's' : ''}`);
          setShowDatasetConfirmation(false);
          onPipelineComplete?.(); // Refresh project data
        } else {
          setActionError('No DatasetSpecs were created');
        }
      } else {
        // Legacy single-variant mode
        const modifications = {
          target_column: editableTargetColumn,
          feature_columns: editableFeatureColumns,
        };

        const response = await applyDatasetSpecFromStep(projectId, step.id, modifications);
        setActionSuccess(response.message);
        setShowDatasetConfirmation(false);
        onPipelineComplete?.(); // Refresh project data
      }
    } catch (err) {
      if (err instanceof ApiException) {
        setActionError(err.detail);
      } else {
        setActionError('Failed to create DatasetSpec');
      }
    } finally {
      setIsCreatingDataset(false);
    }
  };

  // Toggle variant selection
  const handleToggleVariant = (variantName: string) => {
    setSelectedVariants(prev => {
      const newSet = new Set(prev);
      if (newSet.has(variantName)) {
        newSet.delete(variantName);
      } else {
        newSet.add(variantName);
      }
      return newSet;
    });
  };

  // Select/deselect all variants
  const handleSelectAllVariants = () => {
    if (selectedVariants.size === datasetVariants.length) {
      setSelectedVariants(new Set());
    } else {
      setSelectedVariants(new Set(datasetVariants.map(v => v.name)));
    }
  };

  // Handle creating experiments for all datasets from step
  const handleCreateExperiment = async () => {
    const step = getExperimentDesignStep();
    if (!step) return;

    setIsCreatingExperiment(true);
    setActionError(null);

    try {
      // Create experiments for ALL dataset specs AND all variants (per agent's full recommendation)
      const response = await applyExperimentsBatch(projectId, step.id, {
        create_all_variants: true,  // Create one experiment per variant per dataset
        run_immediately: autoRunExperiments,  // Use toggle state
      });

      if (response.created_count === 0) {
        setActionError('No experiments were created. Make sure you have created dataset specs first.');
        setIsCreatingExperiment(false);
        return;
      }

      // Refresh project data
      onPipelineComplete?.();

      // Show success message and navigate to experiments tab
      const queuedMsg = autoRunExperiments
        ? ` (${response.queued_count} queued for execution)`
        : ' (not auto-running)';
      setActionSuccess(`Created ${response.created_count} experiment${response.created_count !== 1 ? 's' : ''}${queuedMsg}`);

      // Navigate to the project page with experiments tab active
      // Use a short delay to let the user see the success message
      setTimeout(() => {
        navigate(`/projects/${projectId}?tab=experiments`);
      }, 1000);
    } catch (err) {
      if (err instanceof ApiException) {
        setActionError(err.detail);
      } else {
        setActionError('Failed to create experiments');
      }
      setIsCreatingExperiment(false);
    }
  };

  // Remove a feature column from the editable list
  const handleRemoveFeatureColumn = (column: string) => {
    setEditableFeatureColumns(cols => cols.filter(c => c !== column));
  };

  if (isLoading) {
    return (
      <div className="pipeline-section">
        <LoadingSpinner message="Loading pipeline status..." />
      </div>
    );
  }

  const dataSourcesWithSchema = dataSources.filter(ds => ds.schema_summary);

  return (
    <div className="pipeline-section">
      <div className="pipeline-header">
        <div>
          <h3>{dataSources.length > 1 ? 'Step 2: Analyze Dataset' : 'Analyze Dataset'}</h3>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.875rem', color: '#6b7280' }}>
            {dataSources.length > 1
              ? 'Select the combined training dataset to analyze with AI and design ML experiments'
              : 'Select a dataset to analyze with AI. Audits data quality, identifies target column, and designs ML experiments.'
            }
          </p>
        </div>
        {!agentRun && dataSourcesWithSchema.length > 0 && !showStartForm && (
          <button
            className="btn btn-primary"
            onClick={() => setShowStartForm(true)}
          >
            Analyze Dataset
          </button>
        )}
        {agentRun && (agentRun.status === 'completed' || agentRun.status === 'failed' || agentRun.status === 'cancelled') && (
          <button
            className="btn btn-secondary"
            onClick={() => setShowStartForm(true)}
          >
            {agentRun.status === 'cancelled' ? 'Start New Pipeline' : 'Analyze Another Dataset'}
          </button>
        )}
      </div>

      {error && (
        <div className="pipeline-error">
          {error}
          <button onClick={() => setError(null)} className="btn-dismiss">
            Dismiss
          </button>
        </div>
      )}

      {/* Start Pipeline Form */}
      {showStartForm && (
        <div className="pipeline-start-form">
          <h4>Configure Dataset Analysis</h4>
          <form onSubmit={handleStartPipeline}>
            <div className="form-group">
              <label className="form-label">
                Data Source <span className="required">*</span>
              </label>
              <select
                className="form-select"
                value={selectedDataSource}
                onChange={(e) => setSelectedDataSource(e.target.value)}
                required
                disabled={isStarting}
              >
                <option value="">Select a data source...</option>
                {dataSourcesWithSchema.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.schema_summary?.row_count?.toLocaleString() || 0} rows)
                  </option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">
                What do you want to predict? <span className="required">*</span>
              </label>
              <textarea
                className="form-textarea"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="E.g., I want to predict whether customers will churn based on their behavior and demographics..."
                rows={3}
                required
                minLength={10}
                disabled={isStarting}
              />
              <p className="form-hint">
                Describe your prediction goal in natural language. The AI will analyze
                your data and suggest the best approach.
              </p>
            </div>

            {/* Orchestration Options */}
            {orchestrationOptions && (
              <div className="orchestration-options" style={{ marginBottom: '16px', padding: '12px', backgroundColor: 'var(--bg-secondary, #2a2a3e)', borderRadius: '8px', border: '1px solid var(--border-color, #3a3a50)' }}>
                <h5 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: 600, color: 'var(--text-primary, #e0e0e0)' }}>Advanced Options</h5>

                <label style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', cursor: 'pointer', marginBottom: '12px' }}>
                  <input
                    type="checkbox"
                    checked={orchestrationMode === 'project_manager'}
                    onChange={(e) => setOrchestrationMode(e.target.checked ? 'project_manager' : 'sequential')}
                    disabled={isStarting}
                    style={{ marginTop: '4px' }}
                  />
                  <span>
                    <strong style={{ display: 'block', fontSize: '13px', color: 'var(--text-primary, #e0e0e0)' }}>Project Manager Mode</strong>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary, #a0a0b0)' }}>
                      A meta-agent dynamically orchestrates pipeline flow, deciding which agent runs next.
                    </span>
                  </span>
                </label>

                <label style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', cursor: 'pointer', marginBottom: '12px' }}>
                  <input
                    type="checkbox"
                    checked={debateMode === 'enabled'}
                    onChange={(e) => setDebateMode(e.target.checked ? 'enabled' : 'disabled')}
                    disabled={isStarting}
                    style={{ marginTop: '4px' }}
                  />
                  <span>
                    <strong style={{ display: 'block', fontSize: '13px', color: 'var(--text-primary, #e0e0e0)' }}>Debate System</strong>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary, #a0a0b0)' }}>
                      Each step is reviewed by Gemini. They debate; if no consensus, OpenAI judge decides.
                    </span>
                  </span>
                </label>

                {debateMode === 'enabled' && (
                  <div style={{ marginLeft: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <label style={{ fontSize: '12px', color: 'var(--text-secondary, #a0a0b0)' }}>Judge Model:</label>
                      <select
                        value={judgeModel}
                        onChange={(e) => setJudgeModel(e.target.value)}
                        disabled={isStarting}
                        style={{ padding: '4px 8px', fontSize: '12px', borderRadius: '4px', border: '1px solid var(--border-color, #3a3a50)', backgroundColor: 'var(--bg-primary, #1e1e2e)', color: 'var(--text-primary, #e0e0e0)' }}
                      >
                        {orchestrationOptions.judge_models.map(model => (
                          <option key={model} value={model}>
                            {model}{model === orchestrationOptions.default_judge_model ? ' (default)' : ''}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <label style={{ fontSize: '12px', color: 'var(--text-secondary, #a0a0b0)' }}>Max Rounds:</label>
                      <input
                        type="number"
                        value={maxDebateRounds}
                        onChange={(e) => setMaxDebateRounds(Math.max(1, Math.min(10, parseInt(e.target.value) || 3)))}
                        min={1}
                        max={10}
                        disabled={isStarting}
                        style={{ width: '60px', padding: '4px 8px', fontSize: '12px', borderRadius: '4px', border: '1px solid var(--border-color, #3a3a50)', backgroundColor: 'var(--bg-primary, #1e1e2e)', color: 'var(--text-primary, #e0e0e0)' }}
                      />
                      <span style={{ fontSize: '11px', color: 'var(--text-muted, #888)' }}>(1-10)</span>
                    </div>
                  </div>
                )}

                <div style={{ borderTop: '1px solid var(--border-color, #3a3a50)', marginTop: '12px', paddingTop: '12px' }}>
                  <label style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', cursor: 'pointer', marginBottom: '12px' }}>
                    <input
                      type="checkbox"
                      checked={useContextDocuments}
                      onChange={(e) => {
                        setUseContextDocuments(e.target.checked);
                        if (!e.target.checked) setContextABTesting(false);
                      }}
                      disabled={isStarting}
                      style={{ marginTop: '4px' }}
                    />
                    <span>
                      <strong style={{ display: 'block', fontSize: '13px', color: 'var(--text-primary, #e0e0e0)' }}>Use Context Documents</strong>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary, #a0a0b0)' }}>
                        Include uploaded context documents (PDFs, Word docs, etc.) in AI prompts to help with dataset design and experiment planning.
                      </span>
                    </span>
                  </label>

                  <label style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', cursor: useContextDocuments ? 'pointer' : 'not-allowed', opacity: useContextDocuments ? 1 : 0.5 }}>
                    <input
                      type="checkbox"
                      checked={contextABTesting}
                      onChange={(e) => setContextABTesting(e.target.checked)}
                      disabled={isStarting || !useContextDocuments}
                      style={{ marginTop: '4px' }}
                    />
                    <span>
                      <strong style={{ display: 'block', fontSize: '13px', color: 'var(--text-primary, #e0e0e0)' }}>A/B Testing (Context vs No Context)</strong>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary, #a0a0b0)' }}>
                        Create dataset and experiment variants both with and without context documents. Names will include [WITH CONTEXT] or [NO CONTEXT] suffixes.
                      </span>
                    </span>
                  </label>
                </div>
              </div>
            )}

            <div className="form-actions">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowStartForm(false)}
                disabled={isStarting}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={isStarting || !selectedDataSource || description.trim().length < 10}
              >
                {isStarting ? (
                  <>
                    <span className="spinner spinner-small"></span>
                    Running Pipeline...
                  </>
                ) : (
                  'Start Pipeline'
                )}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* No data sources message */}
      {!agentRun && dataSourcesWithSchema.length === 0 && !showStartForm && (
        <div className="pipeline-empty">
          <p>
            Upload a data source to analyze it. The AI will understand your problem,
            audit data quality, and suggest the target column for prediction.
          </p>
        </div>
      )}

      {/* Pipeline Timeline */}
      {agentRun && agentRun.steps && (
        <div className="pipeline-timeline">
          <div className="pipeline-run-info">
            <div className="run-status">
              <StatusBadge status={agentRun.status} />
              <span className="run-name">{agentRun.name}</span>
            </div>
            {agentRun.status === 'running' && (
              <div className="run-progress-container" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <span className="run-progress">
                  <span className="spinner spinner-small"></span>
                  Pipeline in progress...
                </span>
                <button
                  className="btn btn-danger btn-small"
                  onClick={handleCancelPipeline}
                  disabled={isCancelling}
                  style={{
                    padding: '0.25rem 0.75rem',
                    fontSize: '0.75rem',
                    backgroundColor: '#dc2626',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: isCancelling ? 'not-allowed' : 'pointer',
                    opacity: isCancelling ? 0.7 : 1,
                  }}
                >
                  {isCancelling ? 'Cancelling...' : '✕ Cancel'}
                </button>
              </div>
            )}
            {agentRun.status === 'cancelled' && (
              <span className="run-cancelled" style={{ color: '#f59e0b', fontWeight: 500 }}>
                Pipeline cancelled
              </span>
            )}
          </div>

          <div className="steps-timeline">
            {/* Filter out debate steps from main timeline - they're shown separately */}
            {agentRun.steps
              .filter(step => !isDebateStep(step.step_type))
              .sort((a, b) => STEP_INFO[a.step_type].number - STEP_INFO[b.step_type].number)
              .map((step, index, filteredSteps) => {
                const info = STEP_INFO[step.step_type];
                const isLast = index === filteredSteps.length - 1;

                return (
                  <div
                    key={step.id}
                    className={`step-card ${step.status}`}
                    onClick={(e) => {
                      e.preventDefault();
                      setSelectedStep(step);
                    }}
                  >
                    <div className="step-connector">
                      <div className={`step-dot ${step.status}`}>
                        {step.status === 'completed' ? '✓' :
                         step.status === 'running' ? '◉' :
                         step.status === 'failed' ? '✕' : info.number}
                      </div>
                      {!isLast && <div className={`step-line ${step.status}`}></div>}
                    </div>
                    <div className="step-content">
                      <div className="step-header">
                        <span className="step-icon">{info.icon}</span>
                        <span className="step-name">{info.number}. {info.name}</span>
                        <StatusBadge status={step.status} />
                      </div>
                      <div className="step-role">
                        Agent Role: <strong>{info.role}</strong>
                      </div>
                      {step.status === 'running' && (
                        <div className="step-running">
                          <span className="spinner spinner-small"></span>
                          Processing...
                        </div>
                      )}
                      {step.status === 'completed' && step.output_json?.natural_language_summary != null && (
                        <div className="step-summary">
                          {String(step.output_json.natural_language_summary).substring(0, 150)}
                          {String(step.output_json.natural_language_summary).length > 150 ? '...' : null}
                        </div>
                      )}
                      {/* Special display for data_analysis step */}
                      {step.step_type === 'data_analysis' && step.status === 'completed' && step.output_json && (() => {
                        const score = Number(step.output_json.suitability_score) || 0;
                        const scoreClass = score >= 0.7 ? 'good' : score >= 0.4 ? 'fair' : 'poor';
                        const suggestMoreData = Boolean(step.output_json.suggest_more_data);
                        return (
                          <div className="data-analysis-extras">
                            <div className="suitability-score">
                              <span className="score-label">Suitability:</span>
                              <span className={`score-value ${scoreClass}`}>
                                {Math.round(score * 100)}%
                              </span>
                            </div>
                            {suggestMoreData && (
                              <button
                                type="button"
                                className="btn btn-secondary btn-small find-data-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleFindMoreData();
                                }}
                                disabled={isDiscoveringDatasets}
                              >
                                {isDiscoveringDatasets ? 'Searching...' : '🔍 Find More Data'}
                              </button>
                            )}
                          </div>
                        );
                      })()}
                      {step.status === 'failed' && step.error_message && (
                        <div className="step-error-preview">
                          {step.error_message.substring(0, 100)}
                          {step.error_message.length > 100 ? '...' : null}
                        </div>
                      )}
                      {step.status === 'completed' && !isLast && (
                        <div className="step-handoff">
                          Output passed to: <strong>{getNextStepName(step)}</strong>
                        </div>
                      )}
                      <button
                        type="button"
                        className="step-expand-btn"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          setSelectedStep(step);
                        }}
                      >
                        View Details →
                      </button>
                    </div>
                  </div>
                );
              })}
          </div>

          {/* Debate Transcript Section - shows when debate steps exist */}
          {(() => {
            const debateSteps = agentRun.steps.filter(step => isDebateStep(step.step_type));
            if (debateSteps.length === 0) return null;

            // Group debate steps by the step they're debating (from input_json.debate_for)
            const debatesByTarget: Record<string, typeof debateSteps> = {};
            debateSteps.forEach(step => {
              const target = (step.input_json?.debate_for as string) || 'unknown';
              if (!debatesByTarget[target]) debatesByTarget[target] = [];
              debatesByTarget[target].push(step);
            });

            // Sort each group by round number
            Object.values(debatesByTarget).forEach(group => {
              group.sort((a, b) => {
                const roundA = (a.input_json?.round as number) || 0;
                const roundB = (b.input_json?.round as number) || 0;
                return roundA - roundB;
              });
            });

            return (
              <div className="debate-transcript-section" style={{
                marginTop: '1.5rem',
                padding: '1rem',
                backgroundColor: 'var(--bg-secondary, #2a2a3e)',
                borderRadius: '8px',
                border: '1px solid var(--border-color, #3a3a50)',
              }}>
                <h4 style={{
                  margin: '0 0 1rem 0',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  color: 'var(--text-primary, #e0e0e0)',
                  fontSize: '1rem',
                }}>
                  <span>💬</span> Debate Transcript ({debateSteps.length} messages)
                </h4>

                {Object.entries(debatesByTarget).map(([target, steps]) => (
                  <div key={target} className="debate-group" style={{ marginBottom: '1rem' }}>
                    <div style={{
                      fontSize: '0.875rem',
                      color: 'var(--text-secondary, #a0a0b0)',
                      marginBottom: '0.5rem',
                      fontWeight: 600,
                    }}>
                      Debate for: {STEP_INFO[target as AgentStepType]?.name || target}
                    </div>

                    <div className="debate-messages" style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.75rem',
                    }}>
                      {steps.map((step) => {
                        const info = STEP_INFO[step.step_type];
                        const role = (step.output_json?.role as string) || step.step_type;
                        const content = (step.output_json?.content as string) || '';
                        const round = (step.output_json?.round as number) || (step.input_json?.round as number) || 0;
                        const agrees = step.output_json?.agrees as boolean | undefined;
                        const confidence = step.output_json?.confidence as number | undefined;

                        // Determine message alignment based on role
                        const isGemini = step.step_type === 'gemini_critique' || role === 'critique_agent';
                        const isJudge = step.step_type === 'openai_judge' || role === 'judge';

                        return (
                          <div
                            key={step.id}
                            className="debate-message"
                            style={{
                              padding: '0.75rem 1rem',
                              borderRadius: '8px',
                              backgroundColor: isJudge
                                ? 'rgba(245, 158, 11, 0.15)'
                                : isGemini
                                  ? 'rgba(99, 102, 241, 0.15)'
                                  : 'rgba(34, 197, 94, 0.15)',
                              borderLeft: `3px solid ${isJudge ? '#f59e0b' : isGemini ? '#6366f1' : '#22c55e'}`,
                              cursor: 'pointer',
                            }}
                            onClick={() => setSelectedStep(step)}
                          >
                            <div style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              marginBottom: '0.5rem',
                            }}>
                              <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                              }}>
                                <span>{info.icon}</span>
                                <strong style={{ color: 'var(--text-primary, #e0e0e0)' }}>
                                  {info.role}
                                </strong>
                                {round > 0 && (
                                  <span style={{
                                    fontSize: '0.75rem',
                                    padding: '0.125rem 0.375rem',
                                    borderRadius: '4px',
                                    backgroundColor: 'rgba(255,255,255,0.1)',
                                    color: 'var(--text-secondary, #a0a0b0)',
                                  }}>
                                    Round {round}
                                  </span>
                                )}
                              </div>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                {agrees !== undefined && (
                                  <span style={{
                                    fontSize: '0.75rem',
                                    padding: '0.125rem 0.375rem',
                                    borderRadius: '4px',
                                    backgroundColor: agrees ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)',
                                    color: agrees ? '#22c55e' : '#ef4444',
                                  }}>
                                    {agrees ? '✓ Agrees' : '✗ Disagrees'}
                                  </span>
                                )}
                                {confidence !== undefined && (
                                  <span style={{
                                    fontSize: '0.75rem',
                                    color: 'var(--text-secondary, #a0a0b0)',
                                  }}>
                                    {Math.round(confidence * 100)}% confident
                                  </span>
                                )}
                              </div>
                            </div>
                            <div style={{
                              fontSize: '0.875rem',
                              color: 'var(--text-primary, #e0e0e0)',
                              lineHeight: 1.5,
                              whiteSpace: 'pre-wrap',
                            }}>
                              {content.length > 500 ? `${content.substring(0, 500)}...` : content}
                            </div>
                            {content.length > 500 && (
                              <button
                                className="btn btn-secondary btn-small"
                                style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedStep(step);
                                }}
                              >
                                View Full Message →
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            );
          })()}

          {/* Pipeline Actions - shown when pipeline is completed */}
          {agentRun.status === 'completed' && (
            <div className="pipeline-actions">
              {actionError && (
                <div className="pipeline-error">
                  {actionError}
                  <button onClick={() => setActionError(null)} className="btn-dismiss">
                    Dismiss
                  </button>
                </div>
              )}
              {actionSuccess && (
                <div className="pipeline-success">
                  {actionSuccess}
                  <button onClick={() => setActionSuccess(null)} className="btn-dismiss">
                    Dismiss
                  </button>
                </div>
              )}

              <div className="pipeline-action-buttons">
                {getDatasetDesignStep() && (
                  <button
                    className="btn btn-primary"
                    onClick={handleShowDatasetConfirmation}
                    disabled={isCreatingDataset}
                  >
                    {isCreatingDataset ? (
                      <>
                        <span className="spinner spinner-small"></span>
                        Creating DatasetSpec...
                      </>
                    ) : (
                      <>
                        <span className="btn-icon">📊</span>
                        Review & Create DatasetSpec
                      </>
                    )}
                  </button>
                )}
                {getExperimentDesignStep() && (
                  <button
                    className="btn btn-primary"
                    onClick={handleCreateExperiment}
                    disabled={isCreatingExperiment}
                  >
                    {isCreatingExperiment ? (
                      <>
                        <span className="spinner spinner-small"></span>
                        {autoRunExperiments ? 'Creating & Running...' : 'Creating Experiments...'}
                      </>
                    ) : (
                      <>
                        <span className="btn-icon">🧪</span>
                        {autoRunExperiments ? 'Run Experiments for All Datasets' : 'Create Experiments (No Auto-Run)'}
                      </>
                    )}
                  </button>
                )}
              </div>

              {/* Auto-run toggle - show when experiment design step exists */}
              {getExperimentDesignStep() && (
                <div className="auto-run-toggle">
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={autoRunExperiments}
                      onChange={(e) => setAutoRunExperiments(e.target.checked)}
                      disabled={isCreatingExperiment}
                    />
                    <span className="toggle-switch"></span>
                    <span className="toggle-text">
                      Auto-run experiments after creation
                    </span>
                  </label>
                  <span className="toggle-hint">
                    {autoRunExperiments
                      ? 'Experiments will start training immediately'
                      : 'Experiments will be created but not started'}
                  </span>
                </div>
              )}

              <p className="pipeline-action-hint">
                Click "View Details" on any step above to see AI reasoning or create from a specific variant.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Dataset Spec Confirmation Modal - Multi-variant mode */}
      {showDatasetConfirmation && datasetVariants.length > 0 && (
        <div className="modal-backdrop" onClick={() => setShowDatasetConfirmation(false)}>
          <div className="modal-content dataset-variants-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Select Dataset Variants</h3>
              <button
                className="modal-close"
                onClick={() => setShowDatasetConfirmation(false)}
                aria-label="Close"
              >
                &times;
              </button>
            </div>

            <div className="modal-body">
              <p className="modal-description">
                The AI has generated {datasetVariants.length} dataset configuration variants.
                Select the ones you want to create as DatasetSpecs.
              </p>

              {variantsReasoning && (
                <div className="variants-reasoning">
                  <strong>AI Reasoning:</strong> {variantsReasoning}
                </div>
              )}

              {variantsWarnings.length > 0 && (
                <div className="variants-warnings">
                  {variantsWarnings.map((warning, idx) => (
                    <div key={idx} className="variant-warning">⚠️ {warning}</div>
                  ))}
                </div>
              )}

              <div className="variants-selection-header">
                <button
                  className="btn btn-secondary btn-small"
                  onClick={handleSelectAllVariants}
                >
                  {selectedVariants.size === datasetVariants.length ? 'Deselect All' : 'Select All'}
                </button>
                <span className="selection-count">
                  {selectedVariants.size} of {datasetVariants.length} selected
                </span>
              </div>

              <div className="variants-list">
                {datasetVariants.map(variant => (
                  <div
                    key={variant.name}
                    className={`variant-card ${selectedVariants.has(variant.name) ? 'selected' : ''} ${variant.name === recommendedVariant ? 'recommended' : ''}`}
                    onClick={() => handleToggleVariant(variant.name)}
                  >
                    <div className="variant-header">
                      <label className="variant-checkbox">
                        <input
                          type="checkbox"
                          checked={selectedVariants.has(variant.name)}
                          onChange={() => handleToggleVariant(variant.name)}
                        />
                        <span className="variant-name">{variant.name}</span>
                        {variant.name === recommendedVariant && (
                          <span className="recommended-badge">Recommended</span>
                        )}
                      </label>
                    </div>
                    <p className="variant-description">{variant.description}</p>
                    <div className="variant-details">
                      <div className="variant-detail">
                        <span className="detail-label">Features:</span>
                        <span className="detail-value">{variant.feature_columns.length} columns</span>
                      </div>
                      <div className="variant-detail">
                        <span className="detail-label">Split:</span>
                        <span className="detail-value">{variant.train_test_split}</span>
                      </div>
                      <div className="variant-detail">
                        <span className="detail-label">Preprocessing:</span>
                        <span className="detail-value">{variant.preprocessing_strategy}</span>
                      </div>
                    </div>
                    <div className="variant-tradeoff">
                      <span className="tradeoff-label">Expected Tradeoff:</span>
                      <span className="tradeoff-value">{variant.expected_tradeoff}</span>
                    </div>
                  </div>
                ))}
              </div>

              {actionError && (
                <div className="modal-error">
                  {actionError}
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button
                className="btn btn-secondary"
                onClick={() => setShowDatasetConfirmation(false)}
                disabled={isCreatingDataset}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleConfirmDatasetSpec}
                disabled={isCreatingDataset || selectedVariants.size === 0}
              >
                {isCreatingDataset ? (
                  <>
                    <span className="spinner spinner-small"></span>
                    Creating...
                  </>
                ) : (
                  `Create ${selectedVariants.size} DatasetSpec${selectedVariants.size !== 1 ? 's' : ''}`
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Dataset Spec Confirmation Modal - Legacy single-variant mode */}
      {showDatasetConfirmation && datasetSpecPreview && datasetVariants.length === 0 && (
        <div className="modal-backdrop" onClick={() => setShowDatasetConfirmation(false)}>
          <div className="modal-content dataset-confirmation-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Review DatasetSpec Suggestion</h3>
              <button
                className="modal-close"
                onClick={() => setShowDatasetConfirmation(false)}
                aria-label="Close"
              >
                &times;
              </button>
            </div>

            <div className="modal-body">
              <p className="modal-description">
                The AI has suggested the following configuration. You can remove feature columns
                before creating the DatasetSpec.
              </p>

              <div className="form-group">
                <label className="form-label">Target Column (to predict)</label>
                <input
                  type="text"
                  className="form-input"
                  value={editableTargetColumn}
                  onChange={e => setEditableTargetColumn(e.target.value)}
                  placeholder="Target column name"
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  Feature Columns ({editableFeatureColumns.length} selected)
                </label>
                <div className="feature-columns-list">
                  {editableFeatureColumns.map(col => (
                    <span key={col} className="feature-column-tag">
                      {col}
                      <button
                        className="feature-column-remove"
                        onClick={() => handleRemoveFeatureColumn(col)}
                        aria-label={`Remove ${col}`}
                      >
                        &times;
                      </button>
                    </span>
                  ))}
                  {editableFeatureColumns.length === 0 && (
                    <span className="no-features">No feature columns selected</span>
                  )}
                </div>
              </div>

              {datasetSpecPreview.filters_json && Object.keys(datasetSpecPreview.filters_json).length > 0 && (
                <div className="form-group">
                  <label className="form-label">Filters (read-only)</label>
                  <pre className="filters-preview">
                    {JSON.stringify(datasetSpecPreview.filters_json, null, 2)}
                  </pre>
                </div>
              )}

              {actionError && (
                <div className="modal-error">
                  {actionError}
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button
                className="btn btn-secondary"
                onClick={() => setShowDatasetConfirmation(false)}
                disabled={isCreatingDataset}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleConfirmDatasetSpec}
                disabled={isCreatingDataset || !editableTargetColumn}
              >
                {isCreatingDataset ? (
                  <>
                    <span className="spinner spinner-small"></span>
                    Creating...
                  </>
                ) : (
                  'Create DatasetSpec'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Dataset Discovery Modal */}
      {showDatasetDiscovery && (
        <div className="modal-backdrop" onClick={handleCloseDatasetDiscovery}>
          <div className="modal-content dataset-discovery-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Find Additional Datasets</h3>
              <button
                className="modal-close"
                onClick={handleCloseDatasetDiscovery}
                aria-label="Close"
              >
                &times;
              </button>
            </div>

            <div className="modal-body">
              {isDiscoveringDatasets ? (
                <div className="discovery-loading">
                  <LoadingSpinner message="Searching for relevant datasets..." />
                  <p>This may take a moment as we search various data sources.</p>
                </div>
              ) : discoveredDatasets.length > 0 ? (
                <DiscoveredDatasetsList
                  datasets={discoveredDatasets}
                  onApply={handleApplyDiscoveredDatasets}
                  onBack={handleCloseDatasetDiscovery}
                  isApplying={isApplyingDatasets}
                  error={discoveryError}
                  onContinueWithoutDownload={handleCloseDatasetDiscovery}
                />
              ) : (
                <div className="no-datasets-found">
                  <p>No additional datasets were found for your project.</p>
                  <p>You can try:</p>
                  <ul>
                    <li>Uploading your own data</li>
                    <li>Searching on Kaggle, UCI ML Repository, or other data sources</li>
                    <li>Refining your project description</li>
                  </ul>
                  {discoveryError && (
                    <div className="form-error">{discoveryError}</div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Step Drawer */}
      {selectedStep && (
        <AgentStepDrawer
          step={selectedStep}
          stepInfo={STEP_INFO[selectedStep.step_type]}
          projectId={projectId}
          onClose={() => setSelectedStep(null)}
          onActionComplete={(action, resourceId) => {
            // Close the drawer and optionally notify parent
            setSelectedStep(null);
            // Could add toast notification or callback here
            console.log(`Created ${action} with ID: ${resourceId}`);
          }}
        />
      )}
    </div>
  );
}

/**
 * Project Wizard Component
 * LLM-powered multi-step wizard for project setup
 * Integrates with the full AI agent pipeline for comprehensive ML setup
 */
import { useState, useCallback, useEffect } from 'react';
import type { DataSource, TaskType, DiscoveredDataset, AgentRun, AgentStep, AgentStepType } from '../types/api';
import type {
  ProjectConfigSuggestion,
  DatasetSpecSuggestion,
  ExperimentPlanSuggestion,
  ExperimentVariant,
  SchemaSummary,
} from '../services/api';
import {
  suggestProjectConfig,
  suggestDatasetSpec,
  suggestExperimentPlan,
  updateProject,
  uploadDataSource,
  createDatasetSpec,
  createExperiment,
  runExperiment,
  runDatasetDiscovery,
  getDiscoveredDatasets,
  applyDiscoveredDatasets,
  runSetupPipeline,
  getAgentRun,
  applyDatasetSpecsBatch,
  applyExperimentsBatch,
  getOrchestrationOptions,
  ApiException,
} from '../services/api';
import type { OrchestrationMode, DebateMode, OrchestrationOptions } from '../types/api';
import LoadingSpinner from './LoadingSpinner';
import FileUpload from './FileUpload';
import DatasetDiscoveryPanel from './DatasetDiscoveryPanel';
import DiscoveredDatasetsList from './DiscoveredDatasetsList';
import StatusBadge from './StatusBadge';

type WizardStep = 'choose' | 'upload' | 'discovery' | 'discovered-results' | 'pipeline' | 'describe' | 'config' | 'features' | 'experiment' | 'complete';

// Step metadata for pipeline display
const STEP_INFO: Record<AgentStepType, { number: number; name: string; role: string; icon: string }> = {
  data_analysis: { number: 1, name: 'Data Analysis', role: 'Data Analyst', icon: '📋' },
  problem_understanding: { number: 2, name: 'Problem Understanding', role: 'Planner', icon: '🎯' },
  data_audit: { number: 3, name: 'Data Audit', role: 'Data Auditor', icon: '🔍' },
  dataset_design: { number: 4, name: 'Dataset Design', role: 'Dataset Designer', icon: '📊' },
  dataset_validation: { number: 5, name: 'Dataset Validation', role: 'Data Validator', icon: '✓' },
  experiment_design: { number: 6, name: 'Experiment Design', role: 'ML Engineer', icon: '🧪' },
  plan_critic: { number: 7, name: 'Plan Review', role: 'Critic', icon: '✅' },
  dataset_discovery: { number: 0, name: 'Dataset Discovery', role: 'Data Scout', icon: '🔍' },
  results_interpretation: { number: 1, name: 'Results Interpretation', role: 'Analyst', icon: '📊' },
  results_critic: { number: 2, name: 'Results Critic', role: 'Reviewer', icon: '🔍' },
  // Data Architect pipeline steps
  dataset_inventory: { number: 1, name: 'Dataset Inventory', role: 'Data Profiler', icon: '📦' },
  relationship_discovery: { number: 2, name: 'Relationship Discovery', role: 'Data Analyst', icon: '🔗' },
  training_dataset_planning: { number: 3, name: 'Training Dataset Planning', role: 'Data Architect', icon: '📐' },
  training_dataset_build: { number: 4, name: 'Training Dataset Build', role: 'Data Engineer', icon: '🔧' },
  // Auto-improve pipeline steps
  improvement_analysis: { number: 1, name: 'Improvement Analysis', role: 'Improvement Analyst', icon: '📈' },
  improvement_plan: { number: 2, name: 'Improvement Plan', role: 'Improvement Planner', icon: '📝' },
  iteration_context: { number: 1, name: 'Iteration Context', role: 'Context Gatherer', icon: '🔄' },
  improvement_data_analysis: { number: 2, name: 'Improvement Data Analysis', role: 'Data Analyst', icon: '📋' },
  improvement_dataset_design: { number: 3, name: 'Improvement Dataset Design', role: 'Dataset Designer', icon: '📊' },
  improvement_experiment_design: { number: 4, name: 'Improvement Experiment Design', role: 'ML Engineer', icon: '🧪' },
  // Lab notebook and robustness
  lab_notebook_summary: { number: 0, name: 'Lab Notebook Summary', role: 'Summarizer', icon: '📓' },
  robustness_audit: { number: 0, name: 'Robustness Audit', role: 'Auditor', icon: '🛡️' },
  // Orchestration system steps - Project Manager + Debate System
  project_manager: { number: 0, name: 'Project Manager', role: 'Orchestrator', icon: '🎭' },
  gemini_critique: { number: 0, name: 'Gemini Critique', role: 'Critic (Gemini)', icon: '💎' },
  openai_judge: { number: 0, name: 'OpenAI Judge', role: 'Judge (OpenAI)', icon: '⚖️' },
  debate_round: { number: 0, name: 'Debate Round', role: 'Debate', icon: '💬' },
};

const PIPELINE_STEP_ORDER: AgentStepType[] = [
  'problem_understanding',
  'data_audit',
  'dataset_design',
  'dataset_validation',
  'experiment_design',
  'plan_critic',
];

interface ProjectWizardProps {
  projectId: string;  // Now required - wizard works within existing project
  existingDataSources?: DataSource[];  // Pass existing data sources if available
  onComplete: () => void;  // No longer passes projectId since it's already known
  onCancel: () => void;
}

export default function ProjectWizard({ projectId, existingDataSources = [], onComplete, onCancel }: ProjectWizardProps) {
  // Wizard state - start with 'choose' if no data sources, otherwise go to pipeline
  const [currentStep, setCurrentStep] = useState<WizardStep>(
    existingDataSources.length > 0 ? 'pipeline' : 'choose'
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data collected through wizard
  const [_uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [dataSource, setDataSource] = useState<DataSource | null>(
    existingDataSources.length > 0 ? existingDataSources[0] : null
  );
  const [description, setDescription] = useState('');

  // Dataset discovery state
  const [discoveryRunId, setDiscoveryRunId] = useState<string | null>(null);
  const [discoveredDatasets, setDiscoveredDatasets] = useState<DiscoveredDataset[]>([]);
  const [discoveryError, setDiscoveryError] = useState<string | null>(null);
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [isApplyingDiscovered, setIsApplyingDiscovered] = useState(false);

  // LLM suggestions
  const [configSuggestion, setConfigSuggestion] = useState<ProjectConfigSuggestion | null>(null);
  const [schemaSummary, setSchemaSummary] = useState<SchemaSummary | null>(null);
  const [datasetSpecSuggestion, setDatasetSpecSuggestion] = useState<DatasetSpecSuggestion | null>(null);
  const [experimentPlanSuggestion, setExperimentPlanSuggestion] = useState<ExperimentPlanSuggestion | null>(null);

  // User edits to suggestions
  const [editedTaskType, setEditedTaskType] = useState<TaskType | ''>('');
  const [editedTarget, setEditedTarget] = useState('');
  const [editedMetric, setEditedMetric] = useState('');
  const [editedFeatures, setEditedFeatures] = useState<string[]>([]);
  const [selectedVariant, setSelectedVariant] = useState<string>('');

  // Agent pipeline state
  const [agentRun, setAgentRun] = useState<AgentRun | null>(null);
  const [pipelineDescription, setPipelineDescription] = useState('');
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [isCreatingResources, setIsCreatingResources] = useState(false);
  const [resourceCreationStatus, setResourceCreationStatus] = useState<string | null>(null);

  // Dataset issue recovery state
  const [datasetIssue, setDatasetIssue] = useState<{
    stepType: 'data_audit' | 'dataset_design';
    feedback: string;
    suggestions?: string[];
  } | null>(null);
  const [showRecoveryOptions, setShowRecoveryOptions] = useState(false);
  const [isSearchingNewDataset, setIsSearchingNewDataset] = useState(false);

  // Orchestration state (Project Manager + Debate System)
  const [orchestrationOptions, setOrchestrationOptions] = useState<OrchestrationOptions | null>(null);
  const [orchestrationMode, setOrchestrationMode] = useState<OrchestrationMode>('sequential');
  const [debateMode, setDebateMode] = useState<DebateMode>('disabled');
  const [judgeModel, setJudgeModel] = useState<string>('');
  const [maxDebateRounds, setMaxDebateRounds] = useState<number>(3);
  const [debatePartner, setDebatePartner] = useState<string>('');

  // Holdout validation state
  const [holdoutEnabled, setHoldoutEnabled] = useState<boolean>(false);
  const [holdoutPercentage, setHoldoutPercentage] = useState<number>(5);

  // Fetch orchestration options on mount
  useEffect(() => {
    const fetchOrchestrationOptions = async () => {
      try {
        const options = await getOrchestrationOptions();
        setOrchestrationOptions(options);
        setJudgeModel(options.default_judge_model);
        setMaxDebateRounds(options.default_max_debate_rounds || 3);
        setDebatePartner(options.default_debate_partner || 'gemini-2.0-flash');
      } catch (err) {
        // Non-critical - just log and continue without orchestration options
        console.warn('Failed to fetch orchestration options:', err);
      }
    };
    fetchOrchestrationOptions();
  }, []);

  // Step 0: Choose data source
  const handleChooseUpload = () => {
    setCurrentStep('upload');
  };

  const handleChooseDiscovery = () => {
    setCurrentStep('discovery');
  };

  // Step 1a: Upload data - now transitions to pipeline step
  const handleFileUpload = async (file: File) => {
    setError(null);
    setIsLoading(true);
    setUploadedFile(file);

    try {
      // Upload the file to the existing project
      const ds = await uploadDataSource(projectId, file);
      setDataSource(ds);
      // Go to pipeline step (full AI agent pipeline)
      setCurrentStep('pipeline');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload file');
    } finally {
      setIsLoading(false);
    }
  };

  // Step 1b: Dataset discovery (new flow)
  const handleDiscoverySearch = async (
    searchDescription: string,
    constraints?: {
      geography?: string;
      allow_public_data?: boolean;
      licensing_requirements?: string[];
    }
  ) => {
    setDiscoveryError(null);
    setIsDiscovering(true);

    try {
      // Run discovery for the existing project
      const response = await runDatasetDiscovery(projectId, {
        project_description: searchDescription,
        constraints,
        run_async: false,
      });

      setDiscoveryRunId(response.run_id);

      // Get discovered datasets
      const datasets = await getDiscoveredDatasets(response.run_id);
      setDiscoveredDatasets(datasets);
      setDescription(searchDescription);

      if (datasets.length === 0) {
        setDiscoveryError('No datasets found. Try a different description or more general terms.');
      } else {
        setCurrentStep('discovered-results');
      }
    } catch (err) {
      setDiscoveryError(err instanceof Error ? err.message : 'Dataset discovery failed');
    } finally {
      setIsDiscovering(false);
    }
  };

  // Apply selected discovered datasets
  const handleApplyDiscoveredDatasets = async (selectedIndices: number[]) => {
    if (!discoveryRunId) return;

    setDiscoveryError(null);
    setIsApplyingDiscovered(true);

    try {
      await applyDiscoveredDatasets(projectId, discoveryRunId, {
        dataset_indices: selectedIndices,
      });

      // Discovered datasets create placeholder data sources with URLs
      // User will need to either download them externally or upload manually
      // Go to complete step - the project page will show the data sources
      setCurrentStep('complete');
    } catch (err) {
      setDiscoveryError(err instanceof Error ? err.message : 'Failed to apply datasets');
    } finally {
      setIsApplyingDiscovered(false);
    }
  };

  const handleBackToDiscovery = () => {
    setDiscoveredDatasets([]);
    setCurrentStep('discovery');
  };

  // Continue to project even if download failed (user can upload manually)
  const handleContinueWithoutDownload = () => {
    setCurrentStep('complete');
  };

  // Check if a step failure is related to dataset issues
  const checkForDatasetIssue = useCallback((run: AgentRun) => {
    if (run.status !== 'failed' || !run.steps) return null;

    // Check data_audit and dataset_design steps for failures
    const dataAuditStep = run.steps.find(s => s.step_type === 'data_audit');
    const datasetDesignStep = run.steps.find(s => s.step_type === 'dataset_design');

    // Check data_audit failure first (more fundamental issue)
    if (dataAuditStep?.status === 'failed') {
      const feedback = dataAuditStep.error_message ||
        (dataAuditStep.output_json?.error as string) ||
        'The data audit found issues with the dataset that prevent further analysis.';

      // Try to extract suggestions from the output
      const suggestions = dataAuditStep.output_json?.suggestions as string[] | undefined;

      return {
        stepType: 'data_audit' as const,
        feedback,
        suggestions,
      };
    }

    // Check dataset_design failure
    if (datasetDesignStep?.status === 'failed') {
      const feedback = datasetDesignStep.error_message ||
        (datasetDesignStep.output_json?.error as string) ||
        'The dataset design step found issues that prevent creating a valid ML configuration.';

      const suggestions = datasetDesignStep.output_json?.suggestions as string[] | undefined;

      return {
        stepType: 'dataset_design' as const,
        feedback,
        suggestions,
      };
    }

    return null;
  }, []);

  // Poll for pipeline updates
  const pollPipelineStatus = useCallback(async (runId: string) => {
    try {
      const updatedRun = await getAgentRun(runId);
      setAgentRun(updatedRun);

      // If still running, continue polling
      if (updatedRun.status === 'running' || updatedRun.status === 'pending') {
        setTimeout(() => pollPipelineStatus(runId), 3000);
      } else if (updatedRun.status === 'completed') {
        setIsPipelineRunning(false);
        setDatasetIssue(null);
        setShowRecoveryOptions(false);
      } else if (updatedRun.status === 'failed') {
        setIsPipelineRunning(false);

        // Check if the failure is related to dataset issues
        const issue = checkForDatasetIssue(updatedRun);
        if (issue) {
          setDatasetIssue(issue);
          setShowRecoveryOptions(true);
          setPipelineError(null); // Don't show generic error when we have specific dataset issue
        } else {
          setPipelineError('Pipeline failed. Please check the step details for more information.');
        }
      }
    } catch (err) {
      // Continue polling on transient errors
      setTimeout(() => pollPipelineStatus(runId), 3000);
    }
  }, [checkForDatasetIssue]);

  // Start the full AI agent pipeline
  const handleStartPipeline = async () => {
    if (!dataSource || !pipelineDescription.trim()) {
      setPipelineError('Please describe what you want to predict');
      return;
    }

    setPipelineError(null);
    setIsPipelineRunning(true);

    try {
      const response = await runSetupPipeline(projectId, {
        data_source_id: dataSource.id,
        description: pipelineDescription.trim(),
        run_async: false, // Run synchronously for better UX in wizard
        orchestration_mode: orchestrationMode,
        debate_mode: debateMode,
        judge_model: debateMode === 'enabled' ? judgeModel : undefined,
        max_debate_rounds: debateMode === 'enabled' ? maxDebateRounds : undefined,
        debate_partner: debateMode === 'enabled' ? debatePartner : undefined,
        holdout_enabled: holdoutEnabled,
        holdout_percentage: holdoutEnabled ? holdoutPercentage : undefined,
      });

      // Fetch the created run
      const newRun = await getAgentRun(response.run_id);
      setAgentRun(newRun);

      // If still running, start polling
      if (newRun.status === 'running' || newRun.status === 'pending') {
        pollPipelineStatus(response.run_id);
      } else {
        setIsPipelineRunning(false);
      }
    } catch (err) {
      setIsPipelineRunning(false);
      if (err instanceof ApiException) {
        setPipelineError(err.detail);
      } else {
        setPipelineError(err instanceof Error ? err.message : 'Failed to start pipeline');
      }
    }
  };

  // Get helper functions for pipeline step display
  const getNextStepName = (currentStep: AgentStep): string | null => {
    const currentIndex = PIPELINE_STEP_ORDER.indexOf(currentStep.step_type);
    if (currentIndex < 0 || currentIndex >= PIPELINE_STEP_ORDER.length - 1) return null;
    const nextType = PIPELINE_STEP_ORDER[currentIndex + 1];
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

  // Apply all recommendations from pipeline and create resources
  const handleApplyPipelineRecommendations = async () => {
    if (!agentRun) return;

    const datasetStep = getDatasetDesignStep();
    const experimentStep = getExperimentDesignStep();

    if (!datasetStep || !experimentStep) {
      setPipelineError('Pipeline incomplete. Missing dataset or experiment design steps.');
      return;
    }

    setIsCreatingResources(true);
    setPipelineError(null);
    setResourceCreationStatus('Creating dataset specifications...');

    try {
      // First, create dataset specs from the dataset_design step
      const variants = datasetStep.output_json?.variants as Array<{ name: string }> | undefined;
      const variantNames = variants?.map(v => v.name) || [];

      if (variantNames.length === 0) {
        // Legacy format - use recommended variant or first one
        const recommended = datasetStep.output_json?.recommended_variant as string;
        if (recommended) {
          variantNames.push(recommended);
        }
      }

      if (variantNames.length === 0) {
        throw new Error('No dataset variants found in pipeline output');
      }

      // Create dataset specs
      const datasetResponse = await applyDatasetSpecsBatch(projectId, datasetStep.id, variantNames);
      const createdSpecs = datasetResponse.dataset_specs.length;

      setResourceCreationStatus(`Created ${createdSpecs} dataset spec(s). Creating experiments...`);

      // Create experiments for ALL variants (per agent's full recommendation)
      await applyExperimentsBatch(projectId, experimentStep.id, {
        create_all_variants: true,  // Create one experiment per variant per dataset
        run_immediately: true,
      });

      setResourceCreationStatus(null);
      setIsCreatingResources(false);

      // Update project name if possible
      const summary = agentRun.steps?.find(s => s.step_type === 'problem_understanding')?.output_json?.natural_language_summary as string;
      if (summary) {
        const shortName = summary.split('.')[0].substring(0, 50);
        await updateProject(projectId, { name: shortName || 'ML Project' });
      }

      // Go to complete step
      setCurrentStep('complete');
    } catch (err) {
      setIsCreatingResources(false);
      setResourceCreationStatus(null);
      if (err instanceof ApiException) {
        setPipelineError(err.detail);
      } else {
        setPipelineError(err instanceof Error ? err.message : 'Failed to apply recommendations');
      }
    }
  };

  // Reset pipeline state for retry with new data
  const resetPipelineForNewData = () => {
    setAgentRun(null);
    setPipelineError(null);
    setDatasetIssue(null);
    setShowRecoveryOptions(false);
    setDataSource(null);
    setUploadedFile(null);
  };

  // Handle user choosing to upload a new dataset after failure
  const handleUploadNewDataset = () => {
    resetPipelineForNewData();
    setCurrentStep('upload');
  };

  // Handle user choosing to search for a new dataset using AI feedback
  const handleSearchNewDataset = async () => {
    if (!projectId || !datasetIssue) return;

    setIsSearchingNewDataset(true);
    setDiscoveryError(null);

    try {
      // Build enhanced search description using the feedback from the failed step
      const feedbackContext = datasetIssue.feedback;
      const suggestionsContext = datasetIssue.suggestions?.join('. ') || '';
      const enhancedDescription = `${pipelineDescription}

Based on previous dataset analysis, the following issues were found:
${feedbackContext}

${suggestionsContext ? `Suggestions for better data: ${suggestionsContext}` : ''}

Please find a dataset that addresses these issues.`;

      // Run discovery with enhanced description
      const response = await runDatasetDiscovery(projectId, {
        project_description: enhancedDescription,
        constraints: {
          allow_public_data: true,
        },
        run_async: false,
      });

      setDiscoveryRunId(response.run_id);

      // Get discovered datasets
      const datasets = await getDiscoveredDatasets(response.run_id);
      setDiscoveredDatasets(datasets);

      // Reset pipeline state
      resetPipelineForNewData();

      if (datasets.length === 0) {
        setDiscoveryError('No suitable datasets found. Try uploading your own data instead.');
        setCurrentStep('discovery');
      } else {
        setCurrentStep('discovered-results');
      }
    } catch (err) {
      setDiscoveryError(err instanceof Error ? err.message : 'Dataset search failed');
      // Go to discovery step so user can try manual search
      resetPipelineForNewData();
      setCurrentStep('discovery');
    } finally {
      setIsSearchingNewDataset(false);
    }
  };

  // Step 2: Get LLM config suggestion
  const handleDescriptionSubmit = async () => {
    if (!dataSource || !description.trim()) {
      setError('Please describe what you want to predict');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const response = await suggestProjectConfig(projectId, {
        description: description.trim(),
        data_source_id: dataSource.id,
      });

      setConfigSuggestion(response.suggestion);
      setSchemaSummary(response.schema_summary);
      setEditedTaskType(response.suggestion.task_type as TaskType);
      setEditedTarget(response.suggestion.target_column);
      setEditedMetric(response.suggestion.primary_metric);
      setCurrentStep('config');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get AI suggestions. Make sure you have an API key configured.');
    } finally {
      setIsLoading(false);
    }
  };

  // Step 3: Get dataset spec suggestion
  const handleConfigConfirm = async () => {
    if (!dataSource || !editedTaskType || !editedTarget) {
      setError('Please select task type and target column');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      // Update project with confirmed config
      await updateProject(projectId, {
        name: configSuggestion?.suggested_name || `${editedTarget} Prediction`,
        task_type: editedTaskType,
      });

      // Get dataset spec suggestion
      const response = await suggestDatasetSpec(projectId, {
        data_source_id: dataSource.id,
        task_type: editedTaskType,
        target_column: editedTarget,
        description,
      });

      setDatasetSpecSuggestion(response.suggestion);
      setEditedFeatures(response.suggestion.feature_columns);
      setCurrentStep('features');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get feature suggestions');
    } finally {
      setIsLoading(false);
    }
  };

  // Step 4: Get experiment plan
  const handleFeaturesConfirm = async () => {
    if (!editedFeatures.length) {
      setError('Please select at least one feature');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const response = await suggestExperimentPlan(projectId, {
        task_type: editedTaskType,
        target_column: editedTarget,
        primary_metric: editedMetric,
        feature_columns: editedFeatures,
        row_count: schemaSummary?.row_count || 0,
        column_count: editedFeatures.length,
        description,
      });

      setExperimentPlanSuggestion(response.suggestion);
      setSelectedVariant(response.suggestion.recommended_variant);
      setCurrentStep('experiment');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get experiment suggestions');
    } finally {
      setIsLoading(false);
    }
  };

  // Step 5: Create experiment and run
  const handleExperimentStart = async () => {
    if (!dataSource || !selectedVariant || !experimentPlanSuggestion) {
      setError('Please select an experiment variant');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      // Create dataset spec
      const datasetSpec = await createDatasetSpec(projectId, {
        name: `${editedTarget} Features`,
        description: `Auto-generated by AI wizard`,
        data_sources_json: { primary: dataSource.id },
        target_column: editedTarget,
        feature_columns: editedFeatures,
      });

      // Find selected variant config
      const variant = experimentPlanSuggestion.variants.find(v => v.name === selectedVariant);
      if (!variant) throw new Error('Selected variant not found');

      // Create experiment
      const experiment = await createExperiment(projectId, {
        name: `${selectedVariant} - ${editedTarget}`,
        description: variant.description,
        dataset_spec_id: datasetSpec.id,
        primary_metric: editedMetric,
        experiment_plan_json: { automl_config: variant.automl_config },
      });

      // Start the experiment
      await runExperiment(experiment.id);

      setCurrentStep('complete');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start experiment');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleFeature = (feature: string) => {
    if (editedFeatures.includes(feature)) {
      setEditedFeatures(editedFeatures.filter(f => f !== feature));
    } else {
      setEditedFeatures([...editedFeatures, feature]);
    }
  };

  const renderStepIndicator = () => {
    // Different step sequences based on path chosen
    const isDiscoveryPath = currentStep === 'discovery' || currentStep === 'discovered-results';
    const isPipelinePath = currentStep === 'pipeline';

    if (currentStep === 'choose') {
      // Don't show steps on choose screen
      return null;
    }

    // For pipeline path, show simplified 3-step indicator
    const pipelineSteps: { key: WizardStep; label: string }[] = [
      { key: 'upload', label: 'Upload Data' },
      { key: 'pipeline', label: 'AI Analysis' },
      { key: 'complete', label: 'Complete' },
    ];

    const uploadSteps: { key: WizardStep; label: string }[] = [
      { key: 'upload', label: 'Upload Data' },
      { key: 'describe', label: 'Describe Goal' },
      { key: 'config', label: 'Review Config' },
      { key: 'features', label: 'Select Features' },
      { key: 'experiment', label: 'Run Experiment' },
    ];

    const discoverySteps: { key: WizardStep; label: string }[] = [
      { key: 'discovery', label: 'Describe Goal' },
      { key: 'discovered-results', label: 'Select Datasets' },
    ];

    const steps = isPipelinePath ? pipelineSteps : isDiscoveryPath ? discoverySteps : uploadSteps;
    const currentIndex = steps.findIndex(s => s.key === currentStep);

    return (
      <div className="wizard-steps">
        {steps.map((step, index) => (
          <div
            key={step.key}
            className={`wizard-step ${index <= currentIndex ? 'active' : ''} ${
              index < currentIndex ? 'completed' : ''
            }`}
          >
            <div className="wizard-step-number">{index + 1}</div>
            <div className="wizard-step-label">{step.label}</div>
          </div>
        ))}
      </div>
    );
  };

  const renderChooseStep = () => (
    <div className="wizard-content">
      <h2>How would you like to start?</h2>
      <p className="wizard-description">
        Choose how you want to begin your ML project.
      </p>

      <div className="choose-options">
        <div className="choose-card" onClick={handleChooseUpload}>
          <div className="choose-icon">📁</div>
          <h3>Upload My Data</h3>
          <p>I have a CSV, Excel, or Parquet file ready to upload.</p>
          <button className="btn btn-primary">Upload Data →</button>
        </div>

        <div className="choose-card" onClick={handleChooseDiscovery}>
          <div className="choose-icon">🔍</div>
          <h3>Find Public Datasets</h3>
          <p>Help me find relevant public datasets based on what I want to predict.</p>
          <button className="btn btn-secondary">Search Datasets →</button>
        </div>
      </div>

      <div className="form-actions" style={{ marginTop: '24px' }}>
        <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
      </div>

      <style>{`
        .choose-options {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
          margin-top: 24px;
        }

        .choose-card {
          border: 2px solid #e0e0e0;
          border-radius: 12px;
          padding: 24px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .choose-card:hover {
          border-color: #1976d2;
          box-shadow: 0 4px 12px rgba(25, 118, 210, 0.15);
          transform: translateY(-2px);
        }

        .choose-icon {
          font-size: 48px;
          margin-bottom: 16px;
        }

        .choose-card h3 {
          margin: 0 0 8px 0;
          color: #333;
        }

        .choose-card p {
          margin: 0 0 16px 0;
          color: #666;
          font-size: 14px;
        }

        .choose-card .btn {
          width: 100%;
        }
      `}</style>
    </div>
  );

  const renderUploadStep = () => (
    <div className="wizard-content">
      <h2>Upload Your Data</h2>
      <p className="wizard-description">
        Upload a CSV, Excel, or Parquet file containing your training data.
      </p>
      <FileUpload
        onUpload={handleFileUpload}
        accept=".csv,.xlsx,.xls,.parquet,.json"
      />
      <div className="form-actions" style={{ marginTop: '16px' }}>
        <button className="btn btn-secondary" onClick={() => setCurrentStep('choose')}>← Back</button>
      </div>
    </div>
  );

  const renderDiscoveryStep = () => (
    <div className="wizard-content">
      <DatasetDiscoveryPanel
        onSearch={handleDiscoverySearch}
        isSearching={isDiscovering}
        error={discoveryError}
      />
      <div className="form-actions" style={{ marginTop: '16px' }}>
        <button className="btn btn-secondary" onClick={() => setCurrentStep('choose')}>← Back</button>
      </div>
    </div>
  );

  const renderDiscoveredResultsStep = () => (
    <div className="wizard-content">
      <DiscoveredDatasetsList
        datasets={discoveredDatasets}
        onApply={handleApplyDiscoveredDatasets}
        onBack={handleBackToDiscovery}
        onContinueWithoutDownload={discoveryError ? handleContinueWithoutDownload : undefined}
        isApplying={isApplyingDiscovered}
        error={discoveryError}
      />
    </div>
  );

  const renderPipelineStep = () => (
    <div className="wizard-content pipeline-wizard-content">
      <h2>AI-Powered Project Setup</h2>

      {/* Data info */}
      {dataSource && (
        <div className="wizard-data-info">
          <strong>Data uploaded:</strong> {dataSource.name}
          {dataSource.schema_summary && (
            <span className="text-muted">
              {' '}({dataSource.schema_summary.row_count?.toLocaleString()} rows, {dataSource.schema_summary.columns?.length ?? 0} columns)
            </span>
          )}
        </div>
      )}

      {/* Pipeline not started - show form */}
      {!agentRun && !isPipelineRunning && (
        <div className="pipeline-start-section">
          <p className="wizard-description">
            Describe what you want to predict. Our AI agents will analyze your data through 5 specialized steps:
            Problem Understanding, Data Audit, Dataset Design, Experiment Design, and Plan Review.
          </p>

          <div className="form-group">
            <label className="form-label">What do you want to predict?</label>
            <textarea
              className="form-textarea"
              value={pipelineDescription}
              onChange={(e) => setPipelineDescription(e.target.value)}
              placeholder="e.g., I want to predict which customers will churn based on their usage patterns and demographics"
              rows={4}
              autoFocus
            />
          </div>

          {/* Orchestration Options */}
          {orchestrationOptions && (
            <div className="orchestration-options">
              <h4 className="orchestration-title">Advanced Pipeline Options</h4>

              <div className="orchestration-toggle">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={orchestrationMode === 'project_manager'}
                    onChange={(e) => setOrchestrationMode(e.target.checked ? 'project_manager' : 'sequential')}
                  />
                  <span className="toggle-text">
                    <strong>Project Manager Mode</strong>
                    <span className="toggle-description">
                      A meta-agent dynamically orchestrates pipeline flow, deciding which agent runs next based on outputs.
                    </span>
                  </span>
                </label>
              </div>

              <div className="orchestration-toggle">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={debateMode === 'enabled'}
                    onChange={(e) => setDebateMode(e.target.checked ? 'enabled' : 'disabled')}
                  />
                  <span className="toggle-text">
                    <strong>Debate System</strong>
                    <span className="toggle-description">
                      Each step is reviewed by a Gemini critic. They debate up to 3 rounds; if no consensus, an OpenAI judge decides.
                    </span>
                  </span>
                </label>
              </div>

              {/* Judge Model Selection - only show when debate is enabled */}
              {debateMode === 'enabled' && orchestrationOptions.judge_models.length > 0 && (
                <div className="judge-model-selection">
                  <label className="form-label">OpenAI Judge Model</label>
                  <select
                    className="form-select"
                    value={judgeModel}
                    onChange={(e) => setJudgeModel(e.target.value)}
                  >
                    {orchestrationOptions.judge_models.map(model => (
                      <option key={model} value={model}>
                        {model}{model === orchestrationOptions.default_judge_model ? ' (default)' : ''}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Max Debate Rounds - only show when debate is enabled */}
              {debateMode === 'enabled' && (
                <div className="debate-rounds-selection">
                  <label className="form-label">Max Debate Rounds</label>
                  <div className="debate-rounds-input">
                    <input
                      type="number"
                      className="form-input"
                      value={maxDebateRounds}
                      onChange={(e) => setMaxDebateRounds(Math.max(1, Math.min(10, parseInt(e.target.value) || 3)))}
                      min={1}
                      max={10}
                    />
                    <span className="input-hint">
                      Number of debate rounds before calling judge (1-10, default: 3)
                    </span>
                  </div>
                </div>
              )}

              {/* Debate Partner Selection - only show when debate is enabled */}
              {debateMode === 'enabled' && orchestrationOptions.debate_partners && orchestrationOptions.debate_partners.length > 0 && (
                <div className="debate-partner-selection">
                  <label className="form-label">Debate Partner (Critic)</label>
                  <select
                    className="form-select"
                    value={debatePartner}
                    onChange={(e) => setDebatePartner(e.target.value)}
                  >
                    {orchestrationOptions.debate_partners.map(partner => (
                      <option key={partner.model} value={partner.model}>
                        {partner.display_name} ({partner.provider}){partner.model === orchestrationOptions.default_debate_partner ? ' - default' : ''}
                      </option>
                    ))}
                  </select>
                  <span className="select-hint">
                    {orchestrationOptions.debate_partners.find(p => p.model === debatePartner)?.description || 'Select a model to critique each step'}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Holdout Validation Section */}
          <div className="orchestration-section" style={{ marginTop: '24px' }}>
            <h4 style={{ marginBottom: '12px', color: '#4a5568' }}>Holdout Validation</h4>

            <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={holdoutEnabled}
                onChange={(e) => setHoldoutEnabled(e.target.checked)}
              />
              <span>Hold out data for manual validation</span>
            </label>
            <span className="option-description" style={{ marginLeft: '24px', display: 'block', marginTop: '4px' }}>
              Set aside a portion of data before training for manual model testing
            </span>

            {holdoutEnabled && (
              <div className="form-group" style={{ marginTop: '12px', marginLeft: '24px' }}>
                <label htmlFor="holdoutPercentage">Holdout Percentage: {holdoutPercentage}%</label>
                <input
                  type="range"
                  id="holdoutPercentage"
                  min="1"
                  max="20"
                  value={holdoutPercentage}
                  onChange={(e) => setHoldoutPercentage(Number(e.target.value))}
                  style={{ width: '200px' }}
                />
                <span className="option-description">
                  {holdoutPercentage}% of data will be held out for your validation testing
                </span>
              </div>
            )}
          </div>

          {pipelineError && (
            <div className="wizard-error">{pipelineError}</div>
          )}

          <div className="form-actions">
            <button className="btn btn-secondary" onClick={() => setCurrentStep('upload')}>
              ← Back
            </button>
            <button
              className="btn btn-primary"
              onClick={handleStartPipeline}
              disabled={!pipelineDescription.trim() || isPipelineRunning}
            >
              Start AI Analysis
            </button>
          </div>
        </div>
      )}

      {/* Pipeline running or completed - show timeline */}
      {(agentRun || isPipelineRunning) && (
        <div className="pipeline-timeline-section">
          {/* Pipeline status header */}
          <div className="pipeline-status-header">
            {agentRun && (
              <div className="run-status">
                <StatusBadge status={agentRun.status} />
                <span className="run-name">{agentRun.name || 'Setup Pipeline'}</span>
              </div>
            )}
            {isPipelineRunning && (
              <span className="run-progress">
                <LoadingSpinner size="small" />
                <span style={{ marginLeft: '8px' }}>Pipeline in progress...</span>
              </span>
            )}
          </div>

          {/* Steps timeline */}
          {agentRun?.steps && (
            <div className="wizard-pipeline-timeline">
              {agentRun.steps
                .filter(step => PIPELINE_STEP_ORDER.includes(step.step_type))
                .sort((a, b) => STEP_INFO[a.step_type].number - STEP_INFO[b.step_type].number)
                .map((step, index) => {
                  const info = STEP_INFO[step.step_type];
                  const isLast = index === agentRun.steps!.filter(s => PIPELINE_STEP_ORDER.includes(s.step_type)).length - 1;

                  return (
                    <div key={step.id} className={`pipeline-step-card ${step.status}`}>
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
                            <LoadingSpinner size="small" />
                            <span style={{ marginLeft: '8px' }}>Processing...</span>
                          </div>
                        )}
                        {step.status === 'completed' && step.output_json?.natural_language_summary != null && (
                          <div className="step-summary">
                            {(() => {
                              const summary = String(step.output_json.natural_language_summary);
                              return summary.length > 200 ? summary.substring(0, 200) + '...' : summary;
                            })()}
                          </div>
                        )}
                        {step.status === 'failed' && step.error_message && (
                          <div className="step-error-preview">
                            {step.error_message.substring(0, 100)}
                            {step.error_message.length > 100 ? '...' : ''}
                          </div>
                        )}
                        {step.status === 'completed' && !isLast && (
                          <div className="step-handoff">
                            Output passed to: <strong>{getNextStepName(step)}</strong>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
            </div>
          )}

          {/* Orchestration Steps - PM decisions, debates, judge rulings */}
          {agentRun?.steps && agentRun.steps.some(s =>
            ['project_manager', 'gemini_critique', 'openai_judge', 'debate_round'].includes(s.step_type)
          ) && (
            <div className="orchestration-timeline">
              <h4 className="orchestration-timeline-title">
                <span className="orchestration-icon">🎭</span>
                Orchestration Activity
              </h4>
              <div className="orchestration-steps">
                {agentRun.steps
                  .filter(step => ['project_manager', 'gemini_critique', 'openai_judge', 'debate_round'].includes(step.step_type))
                  .map((step) => {
                    const info = STEP_INFO[step.step_type];
                    return (
                      <div key={step.id} className={`orchestration-step-card ${step.status}`}>
                        <div className="orchestration-step-header">
                          <span className="orchestration-step-icon">{info?.icon || '🔄'}</span>
                          <span className="orchestration-step-name">{info?.name || step.step_type}</span>
                          <StatusBadge status={step.status} />
                        </div>

                        {step.status === 'running' && (
                          <div className="orchestration-step-running">
                            <LoadingSpinner size="small" />
                            <span style={{ marginLeft: '8px' }}>Processing...</span>
                          </div>
                        )}

                        {/* PM decision details */}
                        {step.step_type === 'project_manager' && step.status === 'completed' && step.output_json && (
                          <div className="pm-decision-details">
                            {step.output_json.next_agent != null && (
                              <div className="pm-decision-item">
                                <strong>Next Agent:</strong> {String(step.output_json.next_agent)}
                              </div>
                            )}
                            {step.output_json.reasoning != null && (
                              <div className="pm-decision-item pm-reasoning">
                                <strong>Reasoning:</strong>
                                <p>{String(step.output_json.reasoning).substring(0, 300)}{String(step.output_json.reasoning).length > 300 ? '...' : ''}</p>
                              </div>
                            )}
                            {step.output_json.is_complete === true && (
                              <div className="pm-decision-item pm-complete">
                                <strong>Pipeline Complete:</strong> {String(step.output_json.completion_summary ?? 'All goals achieved')}
                              </div>
                            )}
                            {step.output_json.needs_revision === true && (
                              <div className="pm-decision-item pm-revision">
                                <strong>Revision Needed:</strong> {String(step.output_json.revision_target ?? '')} - {String(step.output_json.revision_feedback ?? '')}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Critique details */}
                        {step.step_type === 'gemini_critique' && step.status === 'completed' && step.output_json && (
                          <div className="critique-details">
                            {step.output_json.agrees != null && (
                              <div className="critique-item">
                                <strong>Agrees:</strong> {step.output_json.agrees === true ? '✅ Yes' : '❌ No'}
                              </div>
                            )}
                            {step.output_json.critique != null && (
                              <div className="critique-item critique-feedback">
                                <strong>Feedback:</strong>
                                <p>{String(step.output_json.critique).substring(0, 200)}{String(step.output_json.critique).length > 200 ? '...' : ''}</p>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Judge details */}
                        {step.step_type === 'openai_judge' && step.status === 'completed' && step.output_json && (
                          <div className="judge-details">
                            {step.output_json.winner != null && (
                              <div className="judge-item">
                                <strong>Ruling:</strong> {String(step.output_json.winner) === 'main_agent' ? '🏆 Main Agent' : '🏆 Critique'}
                              </div>
                            )}
                            {step.output_json.reasoning != null && (
                              <div className="judge-item judge-reasoning">
                                <strong>Reasoning:</strong>
                                <p>{String(step.output_json.reasoning).substring(0, 200)}{String(step.output_json.reasoning).length > 200 ? '...' : ''}</p>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Show logs if available */}
                        {step.logs && step.logs.length > 0 && (
                          <div className="orchestration-step-logs">
                            {step.logs.slice(-5).map((log, idx) => (
                              <div key={idx} className={`log-entry log-${log.message_type}`}>
                                <span className="log-type">{log.message_type}</span>
                                <span className="log-message">{log.message.substring(0, 150)}{log.message.length > 150 ? '...' : ''}</span>
                              </div>
                            ))}
                          </div>
                        )}

                        {step.status === 'failed' && step.error_message && (
                          <div className="orchestration-step-error">
                            {step.error_message.substring(0, 100)}
                            {step.error_message.length > 100 ? '...' : ''}
                          </div>
                        )}
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Error display */}
          {pipelineError && (
            <div className="wizard-error">{pipelineError}</div>
          )}

          {/* Resource creation status */}
          {resourceCreationStatus && (
            <div className="resource-creation-status">
              <LoadingSpinner size="small" />
              <span style={{ marginLeft: '8px' }}>{resourceCreationStatus}</span>
            </div>
          )}

          {/* Actions when pipeline is complete */}
          {agentRun?.status === 'completed' && !isCreatingResources && (
            <div className="pipeline-complete-actions">
              <p className="pipeline-complete-message">
                The AI has analyzed your data and designed optimal dataset configurations and experiments.
                Click below to apply these recommendations and start training.
              </p>
              <div className="form-actions">
                <button className="btn btn-secondary" onClick={onCancel}>
                  Cancel
                </button>
                <button
                  className="btn btn-primary"
                  onClick={handleApplyPipelineRecommendations}
                  disabled={isCreatingResources}
                >
                  Apply Recommendations & Train
                </button>
              </div>
            </div>
          )}

          {/* Dataset Issue Recovery - shown when data audit or dataset design fails */}
          {showRecoveryOptions && datasetIssue && (
            <div className="dataset-issue-recovery">
              <div className="issue-header">
                <span className="issue-icon">⚠️</span>
                <h4>Dataset Issue Detected</h4>
              </div>

              <div className="issue-feedback">
                <p className="feedback-text">{datasetIssue.feedback}</p>
                {datasetIssue.suggestions && datasetIssue.suggestions.length > 0 && (
                  <div className="feedback-suggestions">
                    <strong>Suggestions:</strong>
                    <ul>
                      {datasetIssue.suggestions.map((suggestion, idx) => {
                        // Handle both string suggestions and object suggestions
                        const suggestionText = typeof suggestion === 'string'
                          ? suggestion
                          : (suggestion as { issue?: string; suggestion?: string }).suggestion ||
                            (suggestion as { issue?: string }).issue ||
                            JSON.stringify(suggestion);
                        return (
                          <li key={idx}>{suggestionText}</li>
                        );
                      })}
                    </ul>
                  </div>
                )}
              </div>

              <div className="recovery-options">
                <p className="recovery-prompt">
                  You can either upload a new dataset or let AI search for a better one based on this feedback:
                </p>

                <div className="recovery-buttons">
                  <button
                    className="btn btn-secondary recovery-btn"
                    onClick={handleUploadNewDataset}
                    disabled={isSearchingNewDataset}
                  >
                    <span className="btn-icon">📁</span>
                    Upload New Dataset
                  </button>

                  <button
                    className="btn btn-primary recovery-btn"
                    onClick={handleSearchNewDataset}
                    disabled={isSearchingNewDataset}
                  >
                    {isSearchingNewDataset ? (
                      <>
                        <LoadingSpinner size="small" />
                        <span style={{ marginLeft: '8px' }}>Searching...</span>
                      </>
                    ) : (
                      <>
                        <span className="btn-icon">🔍</span>
                        AI Search for Better Dataset
                      </>
                    )}
                  </button>
                </div>

                <button
                  className="btn btn-link cancel-recovery"
                  onClick={onCancel}
                  disabled={isSearchingNewDataset}
                >
                  Cancel and Exit
                </button>
              </div>
            </div>
          )}

          {/* Actions when pipeline failed (generic - non-dataset issues) */}
          {agentRun?.status === 'failed' && !showRecoveryOptions && (
            <div className="pipeline-failed-actions">
              <p className="pipeline-failed-message">
                The pipeline encountered an error. You can try running it again with a different description.
              </p>
              <div className="form-actions">
                <button className="btn btn-secondary" onClick={onCancel}>
                  Cancel
                </button>
                <button
                  className="btn btn-primary"
                  onClick={() => {
                    setAgentRun(null);
                    setPipelineError(null);
                  }}
                >
                  Try Again
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      <style>{`
        .pipeline-wizard-content {
          max-width: 800px;
        }

        .pipeline-start-section {
          margin-top: 16px;
        }

        .pipeline-timeline-section {
          margin-top: 16px;
        }

        .pipeline-status-header {
          display: flex;
          align-items: center;
          gap: 16px;
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 1px solid #e0e0e0;
        }

        .run-status {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .run-name {
          font-weight: 500;
          color: #333;
        }

        .run-progress {
          display: flex;
          align-items: center;
          color: #1976d2;
          font-size: 14px;
        }

        .wizard-pipeline-timeline {
          margin: 16px 0;
        }

        .pipeline-step-card {
          display: flex;
          gap: 16px;
          padding: 12px 0;
        }

        .step-connector {
          display: flex;
          flex-direction: column;
          align-items: center;
          width: 32px;
        }

        .step-dot {
          width: 28px;
          height: 28px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: 600;
          background-color: #e0e0e0;
          color: #666;
        }

        .step-dot.completed {
          background-color: #4caf50;
          color: white;
        }

        .step-dot.running {
          background-color: #1976d2;
          color: white;
          animation: pulse 1.5s infinite;
        }

        .step-dot.failed {
          background-color: #f44336;
          color: white;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.6; }
        }

        .step-line {
          flex: 1;
          width: 2px;
          background-color: #e0e0e0;
          margin: 4px 0;
          min-height: 20px;
        }

        .step-line.completed {
          background-color: #4caf50;
        }

        .step-content {
          flex: 1;
        }

        .step-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 4px;
        }

        .step-icon {
          font-size: 16px;
        }

        .step-name {
          font-weight: 500;
          color: #333;
        }

        .step-role {
          font-size: 13px;
          color: #666;
          margin-bottom: 6px;
        }

        .step-running {
          display: flex;
          align-items: center;
          color: #1976d2;
          font-size: 13px;
        }

        .step-summary {
          font-size: 13px;
          color: #555;
          background-color: #f5f5f5;
          padding: 8px 12px;
          border-radius: 4px;
          margin: 6px 0;
          line-height: 1.4;
        }

        .step-error-preview {
          font-size: 13px;
          color: #c62828;
          background-color: #ffebee;
          padding: 8px 12px;
          border-radius: 4px;
          margin: 6px 0;
        }

        .step-handoff {
          font-size: 12px;
          color: #888;
          margin-top: 6px;
        }

        .resource-creation-status {
          display: flex;
          align-items: center;
          padding: 12px 16px;
          background-color: #e3f2fd;
          border-radius: 8px;
          margin: 16px 0;
          color: #1565c0;
        }

        .pipeline-complete-actions,
        .pipeline-failed-actions {
          margin-top: 24px;
          padding-top: 16px;
          border-top: 1px solid #e0e0e0;
        }

        .pipeline-complete-message {
          margin-bottom: 16px;
          color: #333;
          line-height: 1.5;
        }

        .pipeline-failed-message {
          margin-bottom: 16px;
          color: #666;
        }

        /* Dataset Issue Recovery Styles */
        .dataset-issue-recovery {
          margin-top: 24px;
          padding: 20px;
          background-color: #fff8e1;
          border: 1px solid #ffcc02;
          border-radius: 8px;
        }

        .issue-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 16px;
        }

        .issue-icon {
          font-size: 24px;
        }

        .issue-header h4 {
          margin: 0;
          color: #f57c00;
          font-size: 18px;
        }

        .issue-feedback {
          background-color: #fff;
          padding: 16px;
          border-radius: 6px;
          margin-bottom: 16px;
        }

        .feedback-text {
          margin: 0 0 12px 0;
          color: #333;
          line-height: 1.5;
        }

        .feedback-suggestions {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid #e0e0e0;
        }

        .feedback-suggestions strong {
          color: #333;
          display: block;
          margin-bottom: 8px;
        }

        .feedback-suggestions ul {
          margin: 0;
          padding-left: 20px;
          color: #555;
        }

        .feedback-suggestions li {
          margin-bottom: 4px;
          line-height: 1.4;
        }

        .recovery-options {
          margin-top: 16px;
        }

        .recovery-prompt {
          margin: 0 0 16px 0;
          color: #555;
          font-size: 14px;
        }

        .recovery-buttons {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
        }

        .recovery-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 20px;
          font-size: 14px;
        }

        .recovery-btn .btn-icon {
          font-size: 16px;
        }

        .cancel-recovery {
          margin-top: 16px;
          color: #666;
          font-size: 13px;
        }

        .cancel-recovery:hover {
          color: #333;
        }

        /* Orchestration Options Styles */
        .orchestration-options {
          margin-top: 20px;
          padding: 16px;
          background-color: #f8f9fa;
          border-radius: 8px;
          border: 1px solid #e0e0e0;
        }

        .orchestration-title {
          margin: 0 0 16px 0;
          color: #333;
          font-size: 14px;
          font-weight: 600;
        }

        .orchestration-toggle {
          margin-bottom: 12px;
        }

        .orchestration-toggle:last-of-type {
          margin-bottom: 0;
        }

        .toggle-label {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          cursor: pointer;
        }

        .toggle-label input[type="checkbox"] {
          margin-top: 4px;
          width: 18px;
          height: 18px;
          cursor: pointer;
        }

        .toggle-text {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .toggle-text strong {
          color: #333;
          font-size: 14px;
        }

        .toggle-description {
          color: #666;
          font-size: 13px;
          line-height: 1.4;
        }

        .judge-model-selection {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid #e0e0e0;
        }

        .judge-model-selection .form-label {
          font-size: 13px;
          margin-bottom: 6px;
        }

        .judge-model-selection .form-select {
          max-width: 300px;
        }

        .debate-rounds-selection {
          margin-top: 12px;
        }

        .debate-rounds-selection .form-label {
          font-size: 13px;
          margin-bottom: 6px;
        }

        .debate-rounds-input {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .debate-rounds-input .form-input {
          width: 80px;
          padding: 6px 10px;
          font-size: 14px;
          border: 1px solid #ccc;
          border-radius: 4px;
        }

        .debate-rounds-input .input-hint {
          color: #666;
          font-size: 12px;
        }

        .debate-partner-selection {
          margin-top: 12px;
        }

        .debate-partner-selection .form-label {
          font-size: 13px;
          margin-bottom: 6px;
        }

        .debate-partner-selection .form-select {
          max-width: 400px;
          margin-bottom: 6px;
        }

        .debate-partner-selection .select-hint {
          display: block;
          color: #666;
          font-size: 12px;
          margin-top: 4px;
          line-height: 1.4;
        }

        /* Orchestration Timeline Styles */
        .orchestration-timeline {
          margin-top: 24px;
          padding: 16px;
          background-color: #f5f0ff;
          border: 1px solid #d4c4f0;
          border-radius: 8px;
        }

        .orchestration-timeline-title {
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 0 0 16px 0;
          color: #6b21a8;
          font-size: 16px;
        }

        .orchestration-icon {
          font-size: 20px;
        }

        .orchestration-steps {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .orchestration-step-card {
          background: white;
          border: 1px solid #e0e0e0;
          border-radius: 6px;
          padding: 12px;
        }

        .orchestration-step-card.completed {
          border-left: 3px solid #4caf50;
        }

        .orchestration-step-card.running {
          border-left: 3px solid #1976d2;
        }

        .orchestration-step-card.failed {
          border-left: 3px solid #f44336;
        }

        .orchestration-step-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }

        .orchestration-step-icon {
          font-size: 16px;
        }

        .orchestration-step-name {
          font-weight: 500;
          color: #333;
        }

        .orchestration-step-running {
          display: flex;
          align-items: center;
          color: #1976d2;
          font-size: 13px;
        }

        .pm-decision-details,
        .critique-details,
        .judge-details {
          margin-top: 8px;
          padding: 8px;
          background: #f8f9fa;
          border-radius: 4px;
          font-size: 13px;
        }

        .pm-decision-item,
        .critique-item,
        .judge-item {
          margin-bottom: 6px;
        }

        .pm-decision-item:last-child,
        .critique-item:last-child,
        .judge-item:last-child {
          margin-bottom: 0;
        }

        .pm-reasoning p,
        .critique-feedback p,
        .judge-reasoning p {
          margin: 4px 0 0 0;
          color: #555;
          line-height: 1.4;
        }

        .pm-complete {
          color: #2e7d32;
        }

        .pm-revision {
          color: #f57c00;
        }

        .orchestration-step-logs {
          margin-top: 8px;
          padding: 8px;
          background: #fafafa;
          border-radius: 4px;
          font-size: 12px;
        }

        .log-entry {
          display: flex;
          gap: 8px;
          margin-bottom: 4px;
          padding: 4px;
          border-radius: 2px;
        }

        .log-entry:last-child {
          margin-bottom: 0;
        }

        .log-type {
          font-weight: 500;
          text-transform: uppercase;
          font-size: 10px;
          padding: 2px 6px;
          border-radius: 3px;
        }

        .log-thinking .log-type {
          background: #e3f2fd;
          color: #1565c0;
        }

        .log-action .log-type {
          background: #fff3e0;
          color: #e65100;
        }

        .log-summary .log-type {
          background: #e8f5e9;
          color: #2e7d32;
        }

        .log-warning .log-type {
          background: #fff8e1;
          color: #f57c00;
        }

        .log-message {
          color: #555;
          flex: 1;
        }

        .orchestration-step-error {
          margin-top: 8px;
          padding: 8px;
          background: #ffebee;
          color: #c62828;
          border-radius: 4px;
          font-size: 13px;
        }
      `}</style>
    </div>
  );

  const renderDescribeStep = () => (
    <div className="wizard-content">
      <h2>What Do You Want to Predict?</h2>
      <p className="wizard-description">
        Describe your goal in plain language. Our AI will analyze your data and suggest the best configuration.
      </p>
      {dataSource && (
        <div className="wizard-data-info">
          <strong>Data uploaded:</strong> {dataSource.name}
          {dataSource.schema_summary && (
            <span className="text-muted">
              {' '}({dataSource.schema_summary.row_count?.toLocaleString()} rows, {dataSource.schema_summary.columns?.length ?? 0} columns)
            </span>
          )}
        </div>
      )}
      <div className="form-group">
        <label className="form-label">Describe what you want to predict:</label>
        <textarea
          className="form-textarea"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="e.g., I want to predict which customers will churn based on their usage patterns and demographics"
          rows={4}
          autoFocus
        />
      </div>
      <div className="form-actions">
        <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
        <button
          className="btn btn-primary"
          onClick={handleDescriptionSubmit}
          disabled={!description.trim()}
        >
          Analyze with AI
        </button>
      </div>
    </div>
  );

  const renderConfigStep = () => (
    <div className="wizard-content">
      <h2>Review Configuration</h2>
      {configSuggestion && (
        <>
          <div className="wizard-suggestion-box">
            <div className="suggestion-header">
              <span className="suggestion-icon">🤖</span>
              <span>AI Suggestion</span>
              <span className="confidence-badge">
                {Math.round(configSuggestion.confidence * 100)}% confident
              </span>
            </div>
            <p className="suggestion-reasoning">{configSuggestion.reasoning}</p>
          </div>

          <div className="form-group">
            <label className="form-label">Task Type</label>
            <select
              className="form-select"
              value={editedTaskType}
              onChange={(e) => setEditedTaskType(e.target.value as TaskType)}
            >
              <option value="binary">Binary Classification</option>
              <option value="multiclass">Multi-class Classification</option>
              <option value="regression">Regression</option>
              <option value="quantile">Quantile Regression</option>
              <option value="timeseries_forecast">Time Series Forecast</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Target Column (what to predict)</label>
            <select
              className="form-select"
              value={editedTarget}
              onChange={(e) => setEditedTarget(e.target.value)}
            >
              {schemaSummary?.columns.map(col => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.inferred_type})
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Primary Metric</label>
            <select
              className="form-select"
              value={editedMetric}
              onChange={(e) => setEditedMetric(e.target.value)}
            >
              {editedTaskType === 'binary' && (
                <>
                  <option value="roc_auc">ROC AUC</option>
                  <option value="accuracy">Accuracy</option>
                  <option value="f1">F1 Score</option>
                  <option value="precision">Precision</option>
                  <option value="recall">Recall</option>
                </>
              )}
              {editedTaskType === 'multiclass' && (
                <>
                  <option value="accuracy">Accuracy</option>
                  <option value="f1_macro">F1 Macro</option>
                  <option value="f1_weighted">F1 Weighted</option>
                </>
              )}
              {(editedTaskType === 'regression' || editedTaskType === 'quantile') && (
                <>
                  <option value="rmse">RMSE</option>
                  <option value="mse">MSE</option>
                  <option value="mae">MAE</option>
                  <option value="r2">R²</option>
                </>
              )}
              {editedTaskType === 'timeseries_forecast' && (
                <>
                  <option value="MASE">MASE</option>
                  <option value="MAPE">MAPE</option>
                  <option value="RMSE">RMSE</option>
                </>
              )}
            </select>
          </div>
        </>
      )}
      <div className="form-actions">
        <button className="btn btn-secondary" onClick={() => setCurrentStep('describe')}>Back</button>
        <button className="btn btn-primary" onClick={handleConfigConfirm}>Continue</button>
      </div>
    </div>
  );

  const renderFeaturesStep = () => (
    <div className="wizard-content">
      <h2>Select Features</h2>
      {datasetSpecSuggestion && (
        <>
          <div className="wizard-suggestion-box">
            <div className="suggestion-header">
              <span className="suggestion-icon">🤖</span>
              <span>AI Feature Selection</span>
            </div>
            <p className="suggestion-reasoning">{datasetSpecSuggestion.reasoning}</p>
            {datasetSpecSuggestion.warnings.length > 0 && (
              <div className="suggestion-warnings">
                {datasetSpecSuggestion.warnings.map((warning, i) => {
                  // Handle both string warnings and object warnings with {issue, severity, recommendation}
                  const warningText = typeof warning === 'string'
                    ? warning
                    : (warning as { issue?: string }).issue || JSON.stringify(warning);
                  return (
                    <p key={i} className="warning-text">⚠️ {warningText}</p>
                  );
                })}
              </div>
            )}
          </div>

          <div className="feature-selection">
            <h4>Select columns to use as features:</h4>
            <div className="feature-list">
              {schemaSummary?.columns
                .filter(col => col.name !== editedTarget)
                .map(col => {
                  const isExcluded = datasetSpecSuggestion.excluded_columns.includes(col.name);
                  const exclusionReason = datasetSpecSuggestion.exclusion_reasons[col.name];

                  return (
                    <label key={col.name} className={`feature-item ${isExcluded ? 'excluded' : ''}`}>
                      <input
                        type="checkbox"
                        checked={editedFeatures.includes(col.name)}
                        onChange={() => toggleFeature(col.name)}
                      />
                      <span className="feature-name">{col.name}</span>
                      <span className="feature-type">{col.inferred_type}</span>
                      {isExcluded && exclusionReason && (
                        <span className="feature-excluded-reason" title={exclusionReason}>
                          ⚠️
                        </span>
                      )}
                    </label>
                  );
                })}
            </div>
            <p className="feature-count">
              {editedFeatures.length} features selected
            </p>
          </div>
        </>
      )}
      <div className="form-actions">
        <button className="btn btn-secondary" onClick={() => setCurrentStep('config')}>Back</button>
        <button
          className="btn btn-primary"
          onClick={handleFeaturesConfirm}
          disabled={editedFeatures.length === 0}
        >
          Continue
        </button>
      </div>
    </div>
  );

  const renderExperimentStep = () => (
    <div className="wizard-content">
      <h2>Choose Experiment</h2>
      {experimentPlanSuggestion && (
        <>
          <div className="wizard-suggestion-box">
            <div className="suggestion-header">
              <span className="suggestion-icon">🤖</span>
              <span>AI Experiment Plan</span>
            </div>
            <p className="suggestion-reasoning">{experimentPlanSuggestion.reasoning}</p>
            <p className="text-muted">
              Estimated total time: ~{experimentPlanSuggestion.estimated_total_time_minutes} minutes
            </p>
          </div>

          <div className="experiment-variants">
            {experimentPlanSuggestion.variants.map((variant: ExperimentVariant) => (
              <label
                key={variant.name}
                className={`variant-card ${selectedVariant === variant.name ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="variant"
                  value={variant.name}
                  checked={selectedVariant === variant.name}
                  onChange={(e) => setSelectedVariant(e.target.value)}
                />
                <div className="variant-content">
                  <h4 className="variant-name">
                    {variant.name}
                    {experimentPlanSuggestion.recommended_variant === variant.name && (
                      <span className="recommended-badge">Recommended</span>
                    )}
                  </h4>
                  <p className="variant-description">{variant.description}</p>
                  <p className="variant-tradeoff">{variant.expected_tradeoff}</p>
                  <div className="variant-config">
                    Time: {Math.round((variant.automl_config.time_limit as number) / 60)} min |
                    Preset: {variant.automl_config.presets as string}
                  </div>
                </div>
              </label>
            ))}
          </div>
        </>
      )}
      <div className="form-actions">
        <button className="btn btn-secondary" onClick={() => setCurrentStep('features')}>Back</button>
        <button
          className="btn btn-primary"
          onClick={handleExperimentStart}
          disabled={!selectedVariant}
        >
          Start Training
        </button>
      </div>
    </div>
  );

  const renderCompleteStep = () => {
    // Check if we came from discovery flow (no data source uploaded)
    const isDiscoveryComplete = discoveredDatasets.length > 0 && !dataSource;
    // Check if there was a download error (user chose to continue anyway)
    const hadDownloadError = isDiscoveryComplete && discoveryError;

    return (
      <div className="wizard-content wizard-complete">
        <div className="complete-icon">{hadDownloadError ? '📁' : isDiscoveryComplete ? '✅' : '🎉'}</div>
        <h2>
          {hadDownloadError
            ? 'Project Created!'
            : isDiscoveryComplete
            ? 'Datasets Downloaded!'
            : 'Experiment Started!'}
        </h2>
        <p>
          {hadDownloadError
            ? 'Your project has been created. The dataset URLs have been saved - you can download them manually from the source links and upload them to your project.'
            : isDiscoveryComplete
            ? 'Your selected datasets have been downloaded and added to your project. The data is ready for analysis - you can now create dataset specifications and run experiments.'
            : 'Your ML experiment is now running. You can monitor its progress on the project page.'}
        </p>
        {hadDownloadError && (
          <div className="manual-upload-info">
            <h4>Next Steps:</h4>
            <ol>
              <li>Visit the dataset source links to download the files</li>
              <li>Go to your project and click "Upload Data Source"</li>
              <li>Upload the downloaded files to continue</li>
            </ol>
          </div>
        )}
        <div className="form-actions">
          <button
            className="btn btn-primary"
            onClick={onComplete}
          >
            View Project
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="project-wizard">
      {currentStep !== 'complete' && currentStep !== 'choose' && renderStepIndicator()}

      {error && <div className="wizard-error">{error}</div>}

      {isLoading ? (
        <div className="wizard-loading">
          <LoadingSpinner />
          <p>AI is analyzing your data...</p>
        </div>
      ) : (
        <>
          {currentStep === 'choose' && renderChooseStep()}
          {currentStep === 'upload' && renderUploadStep()}
          {currentStep === 'discovery' && renderDiscoveryStep()}
          {currentStep === 'discovered-results' && renderDiscoveredResultsStep()}
          {currentStep === 'pipeline' && renderPipelineStep()}
          {currentStep === 'describe' && renderDescribeStep()}
          {currentStep === 'config' && renderConfigStep()}
          {currentStep === 'features' && renderFeaturesStep()}
          {currentStep === 'experiment' && renderExperimentStep()}
          {currentStep === 'complete' && renderCompleteStep()}
        </>
      )}
    </div>
  );
}

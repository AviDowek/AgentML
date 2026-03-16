import { useState, useEffect } from 'react';
import type {
  ProjectHistoryResponse,
  ResearchCycleDetail,
  AgentRunDetail,
  AgentThinkingDetail,
  NotebookEntryDetail,
  BestModelDetail,
} from '../types/api';
import { getProjectHistory } from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';

interface ProjectHistoryViewProps {
  projectId: string;
  onRefresh?: () => void;
}

// Format date for display
function formatDate(dateStr: string | null): string {
  if (!dateStr) return 'N/A';
  const date = new Date(dateStr);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// Status badge component
function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    completed: 'bg-green-100 text-green-800',
    running: 'bg-blue-100 text-blue-800',
    pending: 'bg-gray-100 text-gray-800',
    failed: 'bg-red-100 text-red-800',
  };
  return (
    <span className={`px-2 py-0.5 text-xs rounded-full ${colors[status] || colors.pending}`}>
      {status}
    </span>
  );
}

// Collapsible section component
function CollapsibleSection({
  title,
  icon,
  children,
  defaultOpen = false,
  badge,
}: {
  title: string;
  icon: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border border-gray-200 rounded-lg mb-3">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between bg-gray-50 hover:bg-gray-100 rounded-t-lg transition-colors"
      >
        <div className="flex items-center gap-2">
          <span>{icon}</span>
          <span className="font-medium">{title}</span>
          {badge}
        </div>
        <span className="text-gray-500">{isOpen ? '▼' : '▶'}</span>
      </button>
      {isOpen && <div className="p-4 border-t border-gray-200">{children}</div>}
    </div>
  );
}

// Agent thinking display component
function AgentThinkingView({ step }: { step: AgentThinkingDetail }) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="border border-gray-100 rounded p-3 mb-2 bg-white">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{step.step_name || step.step_type}</span>
          <StatusBadge status={step.status} />
        </div>
        <span className="text-xs text-gray-500">{formatDate(step.created_at)}</span>
      </div>

      {/* Tool calls */}
      {step.tool_calls.length > 0 && (
        <div className="mb-2">
          <div className="text-xs font-medium text-gray-600 mb-1">Tool Calls:</div>
          <div className="space-y-1">
            {step.tool_calls.map((tool, idx) => (
              <div key={idx} className="text-xs bg-blue-50 rounded px-2 py-1">
                <span className="font-mono text-blue-700">{tool.name}</span>
                {tool.result_preview && (
                  <span className="text-gray-600 ml-2">
                    {tool.result_preview.substring(0, 100)}...
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary */}
      {step.summary && (
        <div className="text-sm text-gray-700 mb-2">
          <strong>Summary:</strong> {step.summary}
        </div>
      )}

      {/* Show/hide details toggle */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="text-xs text-blue-600 hover:text-blue-800"
      >
        {showDetails ? 'Hide details' : 'Show thinking logs'}
      </button>

      {/* Detailed logs */}
      {showDetails && (
        <div className="mt-2 space-y-2">
          {step.thinking_log.length > 0 && (
            <div>
              <div className="text-xs font-medium text-purple-700">Thinking:</div>
              <div className="text-xs bg-purple-50 rounded p-2 max-h-32 overflow-y-auto">
                {step.thinking_log.map((log, idx) => (
                  <div key={idx} className="mb-1">{log}</div>
                ))}
              </div>
            </div>
          )}
          {step.observation_log.length > 0 && (
            <div>
              <div className="text-xs font-medium text-green-700">Observations:</div>
              <div className="text-xs bg-green-50 rounded p-2 max-h-32 overflow-y-auto">
                {step.observation_log.map((log, idx) => (
                  <div key={idx} className="mb-1">{log}</div>
                ))}
              </div>
            </div>
          )}
          {step.action_log.length > 0 && (
            <div>
              <div className="text-xs font-medium text-orange-700">Actions:</div>
              <div className="text-xs bg-orange-50 rounded p-2 max-h-32 overflow-y-auto">
                {step.action_log.map((log, idx) => (
                  <div key={idx} className="mb-1">{log}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Agent run display component
function AgentRunView({ run }: { run: AgentRunDetail }) {
  return (
    <CollapsibleSection
      title={`Agent Run ${run.run_type ? `(${run.run_type})` : ''}`}
      icon="🤖"
      badge={
        <span className="text-xs text-gray-500">
          {run.completed_steps}/{run.total_steps} steps
        </span>
      }
    >
      <div className="flex items-center gap-4 mb-3 text-sm text-gray-600">
        <StatusBadge status={run.status} />
        <span>Started: {formatDate(run.created_at)}</span>
        {run.completed_at && <span>Completed: {formatDate(run.completed_at)}</span>}
      </div>
      {run.steps.length > 0 && (
        <div className="space-y-2">
          {run.steps.map((step) => (
            <AgentThinkingView key={step.step_id} step={step} />
          ))}
        </div>
      )}
    </CollapsibleSection>
  );
}

// Research cycle display component
function ResearchCycleView({ cycle }: { cycle: ResearchCycleDetail }) {
  return (
    <CollapsibleSection
      title={cycle.title || `Cycle #${cycle.sequence_number}`}
      icon="🔬"
      defaultOpen={cycle.sequence_number === 1}
      badge={<StatusBadge status={cycle.status} />}
    >
      <div className="text-sm text-gray-600 mb-3">
        Created: {formatDate(cycle.created_at)}
      </div>

      {/* Experiments */}
      {cycle.experiments.length > 0 && (
        <div className="mb-4">
          <h4 className="font-medium text-sm mb-2">Experiments ({cycle.experiments.length})</h4>
          <div className="space-y-2">
            {cycle.experiments.map((exp) => (
              <div
                key={exp.id}
                className="flex items-center justify-between p-2 bg-gray-50 rounded"
              >
                <div>
                  <span className="font-medium text-sm">{exp.name}</span>
                  <StatusBadge status={exp.status} />
                </div>
                <div className="text-sm text-gray-600">
                  {exp.best_score !== null && exp.primary_metric && (
                    <span>
                      {exp.primary_metric}: {exp.best_score.toFixed(4)}
                    </span>
                  )}
                  <span className="ml-2 text-xs">({exp.trial_count} trials)</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Agent runs */}
      {cycle.agent_runs.length > 0 && (
        <div>
          <h4 className="font-medium text-sm mb-2">Agent Runs ({cycle.agent_runs.length})</h4>
          {cycle.agent_runs.map((run) => (
            <AgentRunView key={run.id} run={run} />
          ))}
        </div>
      )}
    </CollapsibleSection>
  );
}

// Notebook entries display component
function NotebookEntriesView({ entries }: { entries: NotebookEntryDetail[] }) {
  if (entries.length === 0) return null;

  return (
    <CollapsibleSection title="Lab Notebook" icon="📓" badge={<span className="text-xs text-gray-500">{entries.length} entries</span>}>
      <div className="space-y-2">
        {entries.map((entry) => (
          <div key={entry.id} className="border border-gray-100 rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className={entry.author_type === 'human' ? '👤' : '🤖'}>
                  {entry.author_type === 'human' ? '👤' : '🤖'}
                </span>
                <span className="font-medium text-sm">{entry.title}</span>
              </div>
              <span className="text-xs text-gray-500">{formatDate(entry.created_at)}</span>
            </div>
            {entry.body_markdown && (
              <div className="text-sm text-gray-700 mt-2 whitespace-pre-wrap">
                {entry.body_markdown.substring(0, 500)}
                {entry.body_markdown.length > 500 && '...'}
              </div>
            )}
          </div>
        ))}
      </div>
    </CollapsibleSection>
  );
}

// Best models display component
function BestModelsView({ models }: { models: BestModelDetail[] }) {
  if (models.length === 0) return null;

  return (
    <CollapsibleSection title="Best Models" icon="🏆" defaultOpen={true}>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-2 px-3 font-medium">Experiment</th>
              <th className="text-left py-2 px-3 font-medium">Trial</th>
              <th className="text-left py-2 px-3 font-medium">Metric</th>
              <th className="text-right py-2 px-3 font-medium">Score</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model, idx) => (
              <tr key={`${model.experiment_id}-${model.trial_id}`} className={idx % 2 === 0 ? 'bg-gray-50' : ''}>
                <td className="py-2 px-3">{model.experiment_name}</td>
                <td className="py-2 px-3">{model.trial_name}</td>
                <td className="py-2 px-3">{model.metric_name}</td>
                <td className="py-2 px-3 text-right font-mono">
                  {model.metric_value.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </CollapsibleSection>
  );
}

// Main component
export default function ProjectHistoryView({ projectId, onRefresh }: ProjectHistoryViewProps) {
  const [history, setHistory] = useState<ProjectHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'cycles' | 'notebook' | 'models'>('cycles');

  useEffect(() => {
    loadHistory();
  }, [projectId]);

  const loadHistory = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getProjectHistory(projectId, {
        includeLogs: true,
        limitCycles: 20,
        limitEntries: 50,
      });
      setHistory(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load project history');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (error) {
    return <ErrorMessage message={error} onRetry={loadHistory} />;
  }

  if (!history) {
    return <div className="text-gray-500 text-center p-8">No history available</div>;
  }

  return (
    <div className="p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold">Project History</h2>
          <div className="text-sm text-gray-600 mt-1">
            {history.total_cycles} cycles, {history.total_experiments} experiments,{' '}
            {history.total_notebook_entries} notebook entries
          </div>
        </div>
        <button
          onClick={() => {
            loadHistory();
            onRefresh?.();
          }}
          className="px-3 py-1.5 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
        >
          Refresh
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 mb-4">
        <button
          onClick={() => setActiveTab('cycles')}
          className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
            activeTab === 'cycles'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Research Cycles ({history.research_cycles.length})
        </button>
        <button
          onClick={() => setActiveTab('notebook')}
          className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
            activeTab === 'notebook'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Lab Notebook ({history.notebook_entries.length})
        </button>
        <button
          onClick={() => setActiveTab('models')}
          className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
            activeTab === 'models'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Best Models ({history.best_models.length})
        </button>
      </div>

      {/* Content */}
      <div className="mt-4">
        {activeTab === 'cycles' && (
          <div>
            {history.research_cycles.length === 0 ? (
              <div className="text-gray-500 text-center py-8">
                No research cycles yet. Start the pipeline to begin.
              </div>
            ) : (
              history.research_cycles.map((cycle) => (
                <ResearchCycleView key={cycle.id} cycle={cycle} />
              ))
            )}
          </div>
        )}

        {activeTab === 'notebook' && (
          <NotebookEntriesView entries={history.notebook_entries} />
        )}

        {activeTab === 'models' && <BestModelsView models={history.best_models} />}
      </div>

      {/* Footer timestamps */}
      <div className="mt-6 pt-4 border-t border-gray-200 text-xs text-gray-500">
        {history.last_experiment_at && (
          <div>Last experiment: {formatDate(history.last_experiment_at)}</div>
        )}
        {history.last_agent_run_at && (
          <div>Last agent run: {formatDate(history.last_agent_run_at)}</div>
        )}
      </div>
    </div>
  );
}

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AgentStepDrawer from './AgentStepDrawer';
import * as api from '../services/api';
import type { AgentStep, AgentStepLog } from '../types/api';

// Mock the API module
vi.mock('../services/api', () => ({
  getAgentStepLogs: vi.fn(),
  applyDatasetSpecFromStep: vi.fn(),
  applyExperimentPlanFromStep: vi.fn(),
  ApiException: class ApiException extends Error {
    detail: string;
    status: number;
    constructor(status: number, detail: string) {
      super(detail);
      this.detail = detail;
      this.status = status;
    }
  },
}));

const mockStep: AgentStep = {
  id: 'step-1',
  agent_run_id: 'run-1',
  step_type: 'problem_understanding',
  status: 'completed',
  input_json: { description: 'Test prediction task' },
  output_json: {
    natural_language_summary: 'This is a binary classification problem.',
    task_type: 'binary',
    target_column: 'churn',
  },
  retry_count: 0,
  started_at: '2024-01-01T00:00:00Z',
  finished_at: '2024-01-01T00:01:30Z',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:01:30Z',
};

const mockStepInfo = {
  number: 1,
  name: 'Problem Understanding',
  role: 'Planner',
  icon: '🎯',
};

const mockLogs: AgentStepLog[] = [
  {
    id: 'log-1',
    agent_step_id: 'step-1',
    sequence: 1,
    timestamp: '2024-01-01T00:00:01Z',
    message_type: 'info',
    message: 'Starting problem understanding...',
  },
  {
    id: 'log-2',
    agent_step_id: 'step-1',
    sequence: 2,
    timestamp: '2024-01-01T00:00:05Z',
    message_type: 'thought',
    message: 'Analyzing the problem description...',
  },
  {
    id: 'log-3',
    agent_step_id: 'step-1',
    sequence: 3,
    timestamp: '2024-01-01T00:00:10Z',
    message_type: 'thought',
    message: 'Identifying task type and target variable',
  },
  {
    id: 'log-4',
    agent_step_id: 'step-1',
    sequence: 4,
    timestamp: '2024-01-01T00:01:30Z',
    message_type: 'summary',
    message: 'Analysis complete: Binary classification problem identified',
  },
];

describe('AgentStepDrawer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders drawer with step title and role', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogs,
      last_sequence: 4,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    expect(screen.getByText(/Problem Understanding/i)).toBeInTheDocument();
    expect(screen.getByText('Planner')).toBeInTheDocument();
  });

  it('displays step status badge', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogs,
      last_sequence: 4,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    expect(screen.getByText('completed')).toBeInTheDocument();
  });

  it('shows summary section for completed steps', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogs,
      last_sequence: 4,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/This is a binary classification problem/i)).toBeInTheDocument();
    });
  });

  it('fetches and displays logs', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogs,
      last_sequence: 4,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    await waitFor(() => {
      // Info log is always visible
      expect(screen.getByText('Starting problem understanding...')).toBeInTheDocument();
      // Summary log is always visible
      expect(screen.getByText('Analysis complete: Binary classification problem identified')).toBeInTheDocument();
    });

    // 'thought' logs are hidden by default (they are thinking types)
    expect(screen.queryByText('Analyzing the problem description...')).not.toBeInTheDocument();
  });

  it('shows correct log type labels and icons', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogs,
      last_sequence: 4,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    // Wait for logs to be fetched and rendered
    await waitFor(() => {
      expect(screen.getByText('Starting problem understanding...')).toBeInTheDocument();
    });

    // Now check for log type labels (info and summary are visible by default)
    expect(screen.getByText('Info')).toBeInTheDocument();
    // 'thought' logs are now hidden by default (they are thinking types)
    // There are multiple "Summary" elements (drawer section header + log type label)
    expect(screen.getAllByText('Summary').length).toBeGreaterThanOrEqual(1);
  });

  it('has three tabs: Logs, Output, Input', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogs,
      last_sequence: 4,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    expect(screen.getByRole('button', { name: /Logs/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Output/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Input/i })).toBeInTheDocument();
  });

  it('switches to Output tab and shows structured output', async () => {
    const user = userEvent.setup();
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogs,
      last_sequence: 4,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    await user.click(screen.getByRole('button', { name: /Output/i }));

    await waitFor(() => {
      // The component renders keys without quotes around them
      expect(screen.getByText('task_type')).toBeInTheDocument();
      expect(screen.getByText('target_column')).toBeInTheDocument();
    });
  });

  it('switches to Input tab and shows input data', async () => {
    const user = userEvent.setup();
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogs,
      last_sequence: 4,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    await user.click(screen.getByRole('button', { name: /Input/i }));

    await waitFor(() => {
      // The component renders keys without quotes around them
      expect(screen.getByText('description')).toBeInTheDocument();
    });
  });

  it('calls onClose when close button is clicked', async () => {
    const user = userEvent.setup();
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: [],
      last_sequence: 0,
      has_more: false,
    });

    const onClose = vi.fn();

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={onClose}
      />
    );

    await user.click(screen.getByText('×'));

    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose when Escape key is pressed', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: [],
      last_sequence: 0,
      has_more: false,
    });

    const onClose = vi.fn();

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={onClose}
      />
    );

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onClose).toHaveBeenCalled();
  });

  it('shows error message for failed steps', async () => {
    const failedStep: AgentStep = {
      ...mockStep,
      status: 'failed',
      error_message: 'LLM API error: Rate limit exceeded',
    };

    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: [],
      last_sequence: 0,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={failedStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    expect(screen.getByText(/LLM API error: Rate limit exceeded/i)).toBeInTheDocument();
  });

  it('displays duration for completed steps', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: [],
      last_sequence: 0,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    // Duration should show (1m 30s based on mock data)
    expect(screen.getByText(/1m 30s/i)).toBeInTheDocument();
  });

  it('fetches logs on mount', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogs,
      last_sequence: 4,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    await waitFor(() => {
      expect(api.getAgentStepLogs).toHaveBeenCalledWith(mockStep.id, 0);
    });
  });

  it('shows pending state when step has not started', async () => {
    const pendingStep: AgentStep = {
      ...mockStep,
      status: 'pending',
    };

    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: [],
      last_sequence: 0,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={pendingStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Step hasn't started yet/i)).toBeInTheDocument();
    });
  });

  it('shows retry count when step has retries', async () => {
    const stepWithRetries: AgentStep = {
      ...mockStep,
      retry_count: 2,
    };

    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: [],
      last_sequence: 0,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={stepWithRetries}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    expect(screen.getByText('2')).toBeInTheDocument(); // retry count value
    expect(screen.getByText(/Retries/i)).toBeInTheDocument();
  });
});

// ============================================
// Rich Log Type Tests
// ============================================

describe('AgentStepDrawer - Rich Log Types', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockLogsWithThinking: AgentStepLog[] = [
    {
      id: 'log-1',
      agent_step_id: 'step-1',
      sequence: 1,
      timestamp: '2024-01-01T00:00:01Z',
      message_type: 'action',
      message: 'Analyzing data for prediction task...',
    },
    {
      id: 'log-2',
      agent_step_id: 'step-1',
      sequence: 2,
      timestamp: '2024-01-01T00:00:05Z',
      message_type: 'thinking',
      message: 'Looking at the data structure...',
    },
    {
      id: 'log-3',
      agent_step_id: 'step-1',
      sequence: 3,
      timestamp: '2024-01-01T00:00:10Z',
      message_type: 'thinking',
      message: 'The dataset has 1000 rows and 10 features',
    },
    {
      id: 'log-4',
      agent_step_id: 'step-1',
      sequence: 4,
      timestamp: '2024-01-01T00:00:15Z',
      message_type: 'hypothesis',
      message: 'RandomForest might work well for this task',
    },
    {
      id: 'log-5',
      agent_step_id: 'step-1',
      sequence: 5,
      timestamp: '2024-01-01T00:00:20Z',
      message_type: 'hypothesis',
      message: 'XGBoost could also be effective',
    },
    {
      id: 'log-6',
      agent_step_id: 'step-1',
      sequence: 6,
      timestamp: '2024-01-01T00:01:30Z',
      message_type: 'summary',
      message: 'Analysis complete with recommendations',
    },
  ];

  it('hides thinking logs by default', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogsWithThinking,
      last_sequence: 6,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    await waitFor(() => {
      // Summary should be visible (not a thinking type)
      expect(screen.getByText('Analysis complete with recommendations')).toBeInTheDocument();
    });

    // Thinking and hypothesis logs should be hidden by default
    expect(screen.queryByText('Looking at the data structure...')).not.toBeInTheDocument();
    expect(screen.queryByText('RandomForest might work well for this task')).not.toBeInTheDocument();
  });

  it('shows thinking toggle button with count when thinking logs exist', async () => {
    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogsWithThinking,
      last_sequence: 6,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    await waitFor(() => {
      // Should show toggle with count of thinking-type logs (thinking, hypothesis, action = 5)
      expect(screen.getByRole('button', { name: /Show full thinking/i })).toBeInTheDocument();
    });
  });

  it('reveals all logs when Show full thinking is clicked', async () => {
    const user = userEvent.setup();

    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogsWithThinking,
      last_sequence: 6,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    // Wait for logs to load
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Show full thinking/i })).toBeInTheDocument();
    });

    // Click the toggle button
    await user.click(screen.getByRole('button', { name: /Show full thinking/i }));

    // Now thinking and hypothesis logs should be visible
    await waitFor(() => {
      expect(screen.getByText('Looking at the data structure...')).toBeInTheDocument();
      expect(screen.getByText('RandomForest might work well for this task')).toBeInTheDocument();
      expect(screen.getByText('XGBoost could also be effective')).toBeInTheDocument();
    });
  });

  it('hides thinking logs when Hide thinking is clicked', async () => {
    const user = userEvent.setup();

    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogsWithThinking,
      last_sequence: 6,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    // Wait for logs to load
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Show full thinking/i })).toBeInTheDocument();
    });

    // Show thinking
    await user.click(screen.getByRole('button', { name: /Show full thinking/i }));

    // Verify thinking logs are shown
    await waitFor(() => {
      expect(screen.getByText('Looking at the data structure...')).toBeInTheDocument();
    });

    // Hide thinking
    await user.click(screen.getByRole('button', { name: /Hide thinking/i }));

    // Thinking logs should be hidden again
    await waitFor(() => {
      expect(screen.queryByText('Looking at the data structure...')).not.toBeInTheDocument();
    });
  });

  it('shows correct styling for different log types', async () => {
    const user = userEvent.setup();

    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: mockLogsWithThinking,
      last_sequence: 6,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    // Wait for logs to load and show thinking
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Show full thinking/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /Show full thinking/i }));

    // Check that the correct type labels are shown
    await waitFor(() => {
      expect(screen.getByText('Action')).toBeInTheDocument();
      expect(screen.getAllByText('Thinking').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Hypothesis').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('does not show thinking toggle when no thinking logs exist', async () => {
    const logsWithoutThinking: AgentStepLog[] = [
      {
        id: 'log-1',
        agent_step_id: 'step-1',
        sequence: 1,
        timestamp: '2024-01-01T00:00:01Z',
        message_type: 'info',
        message: 'General info message',
      },
      {
        id: 'log-2',
        agent_step_id: 'step-1',
        sequence: 2,
        timestamp: '2024-01-01T00:00:10Z',
        message_type: 'summary',
        message: 'Final summary',
      },
    ];

    vi.mocked(api.getAgentStepLogs).mockResolvedValue({
      logs: logsWithoutThinking,
      last_sequence: 2,
      has_more: false,
    });

    render(
      <AgentStepDrawer
        step={mockStep}
        stepInfo={mockStepInfo}
        projectId="proj-1"
        onClose={() => {}}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('General info message')).toBeInTheDocument();
    });

    // The thinking toggle should not appear
    expect(screen.queryByRole('button', { name: /Show full thinking/i })).not.toBeInTheDocument();
  });
});

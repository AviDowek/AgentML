import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import AgentPipelineTimeline from './AgentPipelineTimeline';
import * as api from '../services/api';
import type { AgentRun, AgentStep, DataSource } from '../types/api';

// Helper to render with router
const renderWithRouter = (ui: React.ReactElement) => {
  return render(
    <MemoryRouter>{ui}</MemoryRouter>
  );
};

// Mock the API module
vi.mock('../services/api', () => ({
  listAgentRuns: vi.fn(),
  getAgentRun: vi.fn(),
  runSetupPipeline: vi.fn(),
  ApiException: class ApiException extends Error {
    detail: string;
    status: number;
    constructor(detail: string, status: number) {
      super(detail);
      this.detail = detail;
      this.status = status;
    }
  },
}));

const mockDataSources: DataSource[] = [
  {
    id: 'ds-1',
    project_id: 'proj-1',
    name: 'Test Dataset',
    type: 'file_upload',
    config_json: null,
    schema_summary: {
      columns: [
        { name: 'id', dtype: 'int64', non_null_count: 100, null_count: 0, unique_count: 100, sample_values: [1, 2, 3] },
        { name: 'target', dtype: 'object', non_null_count: 100, null_count: 0, unique_count: 2, sample_values: ['yes', 'no'] },
      ],
      row_count: 100,
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
];

const mockSteps: AgentStep[] = [
  {
    id: 'step-1',
    agent_run_id: 'run-1',
    step_type: 'problem_understanding',
    status: 'completed',
    input_json: { description: 'Test prediction' },
    output_json: { natural_language_summary: 'This is a classification problem.' },
    retry_count: 0,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'step-2',
    agent_run_id: 'run-1',
    step_type: 'data_audit',
    status: 'running',
    input_json: {},
    retry_count: 0,
    created_at: '2024-01-01T00:01:00Z',
    updated_at: '2024-01-01T00:01:00Z',
  },
];

const mockAgentRun: AgentRun = {
  id: 'run-1',
  project_id: 'proj-1',
  name: 'Setup Pipeline Run',
  status: 'running',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  steps: mockSteps,
};

describe('AgentPipelineTimeline', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading spinner initially', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    expect(screen.getByText(/Loading pipeline status/i)).toBeInTheDocument();
  });

  it('shows empty state when no data sources with schema', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={[]}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Upload a data source to analyze it/i)).toBeInTheDocument();
    });
  });

  it('shows "Analyze Dataset" button when data sources are available', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      // There's an h3 header and a button with the same text, so use role selector
      expect(screen.getByRole('button', { name: 'Analyze Dataset' })).toBeInTheDocument();
    });
  });

  it('displays pipeline timeline when agent run exists', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockAgentRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockAgentRun);

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('2. Problem Understanding')).toBeInTheDocument();
      expect(screen.getByText('3. Data Audit')).toBeInTheDocument();
    });
  });

  it('shows step status badges correctly', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockAgentRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockAgentRun);

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      // The pipeline has at least one completed step and one running step
      // Check that both types of status badges are present
      const allBadges = screen.getAllByText(/completed|running/i);
      expect(allBadges.length).toBeGreaterThanOrEqual(2);
    });
  });

  it('displays agent roles for each step', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockAgentRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockAgentRun);

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Planner')).toBeInTheDocument();
      expect(screen.getByText('Data Auditor')).toBeInTheDocument();
    });
  });

  it('shows summary preview for completed steps', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockAgentRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockAgentRun);

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/This is a classification problem/i)).toBeInTheDocument();
    });
  });

  it('shows start form when button is clicked', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Analyze Dataset' })).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Analyze Dataset' }));

    expect(screen.getByText('Configure Dataset Analysis')).toBeInTheDocument();
    expect(screen.getByText(/What do you want to predict/i)).toBeInTheDocument();
  });

  it('calls runSetupPipeline when form is submitted', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.runSetupPipeline).mockResolvedValue({
      run_id: 'new-run-1',
      status: 'running',
      message: 'Pipeline started',
    });
    vi.mocked(api.getAgentRun).mockResolvedValue({
      ...mockAgentRun,
      id: 'new-run-1',
    });

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Analyze Dataset' })).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Analyze Dataset' }));

    // Fill form
    const select = screen.getByRole('combobox');
    await user.selectOptions(select, 'ds-1');

    const textarea = screen.getByPlaceholderText(/predict/i);
    await user.type(textarea, 'I want to predict customer churn');

    // Submit
    await user.click(screen.getByText('Start Pipeline'));

    await waitFor(() => {
      expect(api.runSetupPipeline).toHaveBeenCalledWith('proj-1', {
        data_source_id: 'ds-1',
        description: 'I want to predict customer churn',
        run_async: false,
      });
    });
  });

  it('does NOT call onPipelineComplete on initial load with completed run', async () => {
    // This is intentional to prevent infinite reload loops
    const completedRun: AgentRun = {
      ...mockAgentRun,
      status: 'completed',
    };

    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [completedRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(completedRun);

    const onComplete = vi.fn();

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
        onPipelineComplete={onComplete}
      />
    );

    // Wait for loading to finish
    await waitFor(() => {
      expect(screen.getByText('2. Problem Understanding')).toBeInTheDocument();
    });

    // onComplete should NOT be called on initial load
    expect(onComplete).not.toHaveBeenCalled();
  });

  it('displays handoff notes between completed steps', async () => {
    const runWithTwoCompleted: AgentRun = {
      ...mockAgentRun,
      steps: [
        { ...mockSteps[0], status: 'completed' },
        { ...mockSteps[1], step_type: 'data_audit', status: 'completed' },
      ],
    };

    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [runWithTwoCompleted],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(runWithTwoCompleted);

    renderWithRouter(
      <AgentPipelineTimeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Output passed to:/i)).toBeInTheDocument();
      expect(screen.getByText('Data Audit')).toBeInTheDocument();
    });
  });
});

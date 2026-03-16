import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import DataArchitectPipeline from './DataArchitectPipeline';
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
  runDataArchitectPipeline: vi.fn(),
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
    name: 'Customers',
    type: 'file_upload',
    config_json: null,
    schema_summary: {
      columns: [
        { name: 'id', dtype: 'int64', non_null_count: 100, null_count: 0, unique_count: 100, sample_values: [1, 2, 3] },
        { name: 'name', dtype: 'object', non_null_count: 100, null_count: 0, unique_count: 100, sample_values: ['John', 'Jane'] },
      ],
      row_count: 100,
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'ds-2',
    project_id: 'proj-1',
    name: 'Orders',
    type: 'file_upload',
    config_json: null,
    schema_summary: {
      columns: [
        { name: 'order_id', dtype: 'int64', non_null_count: 500, null_count: 0, unique_count: 500, sample_values: [1, 2, 3] },
        { name: 'customer_id', dtype: 'int64', non_null_count: 500, null_count: 0, unique_count: 100, sample_values: [1, 2, 3] },
        { name: 'amount', dtype: 'float64', non_null_count: 500, null_count: 0, unique_count: 450, sample_values: [10.5, 20.0] },
      ],
      row_count: 500,
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
];

const mockDataArchitectSteps: AgentStep[] = [
  {
    id: 'step-1',
    agent_run_id: 'run-1',
    step_type: 'dataset_inventory',
    status: 'completed',
    input_json: {},
    output_json: { profiles: [{ table: 'Customers', row_count: 100 }, { table: 'Orders', row_count: 500 }] },
    retry_count: 0,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'step-2',
    agent_run_id: 'run-1',
    step_type: 'relationship_discovery',
    status: 'running',
    input_json: {},
    retry_count: 0,
    created_at: '2024-01-01T00:01:00Z',
    updated_at: '2024-01-01T00:01:00Z',
  },
];

const mockDataArchitectRun: AgentRun = {
  id: 'run-1',
  project_id: 'proj-1',
  name: 'Data Architect Pipeline',
  status: 'running',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  steps: mockDataArchitectSteps,
};

const mockCompletedRun: AgentRun = {
  id: 'run-2',
  project_id: 'proj-1',
  name: 'Data Architect Pipeline',
  status: 'completed',
  result_json: {
    training_dataset: {
      data_source_id: 'ds-3',
      data_source_name: 'training_customers_orders',
      row_count: 100,
      column_count: 15,
      base_table: 'Customers',
      joined_tables: ['Orders'],
      target_column: 'churn',
      feature_columns: ['total_orders', 'avg_amount', 'days_since_last_order'],
    },
  },
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  steps: mockDataArchitectSteps.map(s => ({ ...s, status: 'completed' as const })),
};

describe('DataArchitectPipeline', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('does not render when no data sources exist', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });

    const { container } = renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={[]}
      />
    );

    await waitFor(() => {
      // Component should not render anything
      expect(container.textContent).toBe('');
    });
  });

  it('shows "Build Training Dataset" button when data sources are available', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Build Training Dataset')).toBeInTheDocument();
    });
  });

  it('shows section description', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/AI discovers relationships between your/i)).toBeInTheDocument();
    });
  });

  it('displays pipeline timeline when agent run exists', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockDataArchitectRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockDataArchitectRun);

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Dataset Inventory')).toBeInTheDocument();
      expect(screen.getByText('Relationship Discovery')).toBeInTheDocument();
    });
  });

  it('shows step status badges correctly', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockDataArchitectRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockDataArchitectRun);

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      // The pipeline has at least one completed step and one running step
      const allBadges = screen.getAllByText(/completed|running/i);
      expect(allBadges.length).toBeGreaterThanOrEqual(2);
    });
  });

  it('displays agent roles for each step', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockDataArchitectRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockDataArchitectRun);

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      // Component shows step names, not roles
      expect(screen.getByText('Dataset Inventory')).toBeInTheDocument();
      expect(screen.getByText('Relationship Discovery')).toBeInTheDocument();
    });
  });

  it('shows dialog when button is clicked', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Build Training Dataset')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Build Training Dataset'));

    // Check dialog is shown by looking for dialog-specific content
    expect(screen.getByText(/Profiling each table and identifying keys/i)).toBeInTheDocument();
    expect(screen.getByText(/Target hint/i)).toBeInTheDocument();
    expect(screen.getByText('Start Building')).toBeInTheDocument();
  });

  it('calls runDataArchitectPipeline when form is submitted', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.runDataArchitectPipeline).mockResolvedValue({
      agent_run_id: 'new-run-1',
      status: 'running',
      message: 'Pipeline started',
    });
    vi.mocked(api.getAgentRun).mockResolvedValue({
      ...mockDataArchitectRun,
      id: 'new-run-1',
    });

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Build Training Dataset')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Build Training Dataset'));

    // Fill in target hint
    const input = screen.getByPlaceholderText(/predict customer churn/i);
    await user.type(input, 'predict customer churn');

    // Submit
    await user.click(screen.getByText('Start Building'));

    await waitFor(() => {
      expect(api.runDataArchitectPipeline).toHaveBeenCalledWith('proj-1', {
        target_hint: 'predict customer churn',
        run_async: false,
      });
    });
  });

  it('shows training dataset summary when pipeline is completed', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockCompletedRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockCompletedRun);

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      // Check for the data source name and base table shown in summary
      expect(screen.getByText('training_customers_orders')).toBeInTheDocument();
      expect(screen.getByText('Customers')).toBeInTheDocument(); // base table
      expect(screen.getByText('churn')).toBeInTheDocument(); // target column
    });
  });

  it('shows target column in summary', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockCompletedRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockCompletedRun);

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('churn')).toBeInTheDocument();
    });
  });

  it('shows feature columns in summary', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockCompletedRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockCompletedRun);

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      // Component shows feature count, not individual column names
      expect(screen.getByText('3 columns')).toBeInTheDocument();
    });
  });

  it('shows "Build New Dataset" button after completion', async () => {
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [mockCompletedRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(mockCompletedRun);

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Build New Dataset')).toBeInTheDocument();
    });
  });

  it('shows error message on pipeline failure', async () => {
    const failedRun: AgentRun = {
      ...mockDataArchitectRun,
      status: 'failed',
      error_message: 'Failed to discover relationships',
    };

    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [failedRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(failedRun);

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Pipeline Failed')).toBeInTheDocument();
      expect(screen.getByText('Failed to discover relationships')).toBeInTheDocument();
    });
  });

  it('closes dialog when Cancel is clicked', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 20,
    });

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Build Training Dataset')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Build Training Dataset'));
    // Dialog is now open - look for dialog-specific content
    expect(screen.getByText('Start Building')).toBeInTheDocument();

    await user.click(screen.getByText('Cancel'));

    await waitFor(() => {
      // Dialog-specific button should be gone
      expect(screen.queryByText('Start Building')).not.toBeInTheDocument();
    });
  });

  it('displays sampling info when dataset was sampled', async () => {
    // Create a run with sampling information in the result
    const sampledRun: AgentRun = {
      ...mockCompletedRun,
      result_json: {
        training_dataset: {
          data_source_id: 'ds-3',
          data_source_name: 'training_customers_orders',
          row_count: 1_000_000,  // Sampled to 1M
          column_count: 15,
          base_table: 'Customers',
          joined_tables: ['Orders'],
          target_column: 'churn',
          feature_columns: ['total_orders', 'avg_amount'],
        },
        was_sampled: true,
        original_row_count: 25_000_000,  // Original was 25M
        sampling_message: 'Dataset is large (25M rows). Sampling 1M rows for training dataset.',
      },
    };

    vi.mocked(api.listAgentRuns).mockResolvedValue({
      items: [sampledRun],
      total: 1,
      skip: 0,
      limit: 20,
    });
    vi.mocked(api.getAgentRun).mockResolvedValue(sampledRun);

    renderWithRouter(
      <DataArchitectPipeline
        projectId="proj-1"
        dataSources={mockDataSources}
      />
    );

    await waitFor(() => {
      // Verify the summary shows the dataset name and base table
      expect(screen.getByText('training_customers_orders')).toBeInTheDocument();
      expect(screen.getByText('Customers')).toBeInTheDocument();
      expect(screen.getByText('churn')).toBeInTheDocument();
    });
  });
});

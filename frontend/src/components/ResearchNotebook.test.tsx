import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ResearchNotebook from './ResearchNotebook';
import * as api from '../services/api';
import type {
  ResearchCycleSummary,
  ResearchCycleResponse,
  LabNotebookEntry,
} from '../types/api';

// Mock the API module
vi.mock('../services/api');

const mockCycles: ResearchCycleSummary[] = [
  {
    id: 'cycle-1',
    sequence_number: 1,
    status: 'completed',
    summary_title: 'Initial Exploration',
    created_at: '2024-01-15T10:00:00Z',
    updated_at: '2024-01-15T12:00:00Z',
    experiment_count: 2,
  },
  {
    id: 'cycle-2',
    sequence_number: 2,
    status: 'running',
    summary_title: 'Feature Engineering',
    created_at: '2024-01-16T10:00:00Z',
    updated_at: '2024-01-16T11:00:00Z',
    experiment_count: 1,
  },
];

const mockEntries: LabNotebookEntry[] = [
  {
    id: 'entry-1',
    project_id: 'project-1',
    research_cycle_id: 'cycle-1',
    agent_step_id: null,
    author_type: 'human',
    title: 'Initial Hypothesis',
    body_markdown: 'The target variable appears to correlate with feature X.',
    created_at: '2024-01-15T10:30:00Z',
    updated_at: '2024-01-15T10:30:00Z',
  },
  {
    id: 'entry-2',
    project_id: 'project-1',
    research_cycle_id: 'cycle-1',
    agent_step_id: 'step-1',
    author_type: 'agent',
    title: 'Data Analysis Results',
    body_markdown: 'Found 3 potential features with high correlation.',
    created_at: '2024-01-15T11:00:00Z',
    updated_at: '2024-01-15T11:00:00Z',
  },
  {
    id: 'entry-3',
    project_id: 'project-1',
    research_cycle_id: null,
    agent_step_id: null,
    author_type: 'human',
    title: 'General Note',
    body_markdown: 'Remember to check for data leakage.',
    created_at: '2024-01-16T09:00:00Z',
    updated_at: '2024-01-16T09:00:00Z',
  },
];

const mockCycleDetail: ResearchCycleResponse = {
  id: 'cycle-1',
  project_id: 'project-1',
  sequence_number: 1,
  status: 'completed',
  summary_title: 'Initial Exploration',
  created_at: '2024-01-15T10:00:00Z',
  updated_at: '2024-01-15T12:00:00Z',
  experiments: [
    {
      id: 'exp-1',
      name: 'Baseline Model',
      status: 'completed',
      best_metric: 0.85,
      primary_metric: 'accuracy',
      created_at: '2024-01-15T10:30:00Z',
    },
  ],
  lab_notebook_entries: [
    {
      id: 'entry-1',
      title: 'Initial Hypothesis',
      author_type: 'human',
      created_at: '2024-01-15T10:30:00Z',
    },
  ],
};

describe('ResearchNotebook', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    vi.mocked(api.listResearchCycles).mockResolvedValue({
      cycles: mockCycles,
      total: mockCycles.length,
    });
    vi.mocked(api.listLabNotebookEntries).mockResolvedValue({
      entries: mockEntries,
      total: mockEntries.length,
    });
    vi.mocked(api.getResearchCycle).mockResolvedValue(mockCycleDetail);
  });

  it('renders loading state initially', () => {
    render(<ResearchNotebook projectId="project-1" />);

    expect(screen.getByText(/loading research notebook/i)).toBeInTheDocument();
  });

  it('renders research cycles after loading', async () => {
    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('Research Cycles')).toBeInTheDocument();
    });

    // Use getAllByText since cycle badges also show cycle numbers
    const cycle1Elements = screen.getAllByText(/Cycle #1/);
    const cycle2Elements = screen.getAllByText(/Cycle #2/);
    expect(cycle1Elements.length).toBeGreaterThan(0);
    expect(cycle2Elements.length).toBeGreaterThan(0);
    expect(screen.getByText('Initial Exploration')).toBeInTheDocument();
    expect(screen.getByText('Feature Engineering')).toBeInTheDocument();
  });

  it('renders notebook entries', async () => {
    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('Initial Hypothesis')).toBeInTheDocument();
    });

    expect(screen.getByText('Data Analysis Results')).toBeInTheDocument();
    expect(screen.getByText('General Note')).toBeInTheDocument();
  });

  it('shows "All Lab Notebook Entries" by default', async () => {
    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('All Lab Notebook Entries')).toBeInTheDocument();
    });
  });

  it('displays cycle experiment counts', async () => {
    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('2 experiments')).toBeInTheDocument();
      expect(screen.getByText('1 experiments')).toBeInTheDocument();
    });
  });

  it('shows author type icons for entries', async () => {
    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('Initial Hypothesis')).toBeInTheDocument();
    });

    // Human entries show person icon, agent entries show robot icon
    // Check for the icons in the entry cards
    const humanIcons = screen.getAllByText('👤');
    const agentIcons = screen.getAllByText('🤖');

    expect(humanIcons.length).toBeGreaterThan(0);
    expect(agentIcons.length).toBeGreaterThan(0);
  });

  it('filters entries when selecting a cycle', async () => {
    const user = userEvent.setup();

    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('Research Cycles')).toBeInTheDocument();
    });

    // Click on the first cycle item button (in sidebar)
    const cycleButton = screen.getAllByRole('button').find(
      btn => btn.textContent?.includes('Initial Exploration')
    );
    expect(cycleButton).toBeDefined();
    if (cycleButton) {
      await user.click(cycleButton);
    }

    await waitFor(() => {
      expect(api.getResearchCycle).toHaveBeenCalledWith('cycle-1');
    });
  });

  it('opens add entry modal when clicking Add Note button', async () => {
    const user = userEvent.setup();

    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('+ Add Note')).toBeInTheDocument();
    });

    await user.click(screen.getByText('+ Add Note'));

    expect(screen.getByText('Add Lab Notebook Entry')).toBeInTheDocument();
    expect(screen.getByLabelText(/title/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/notes/i)).toBeInTheDocument();
  });

  it('creates new entry when form is submitted', async () => {
    const user = userEvent.setup();

    vi.mocked(api.createLabNotebookEntry).mockResolvedValue({
      id: 'new-entry',
      project_id: 'project-1',
      research_cycle_id: null,
      agent_step_id: null,
      author_type: 'human',
      title: 'New Note',
      body_markdown: 'Note content',
      created_at: '2024-01-17T10:00:00Z',
      updated_at: '2024-01-17T10:00:00Z',
    });

    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('+ Add Note')).toBeInTheDocument();
    });

    // Open modal
    await user.click(screen.getByText('+ Add Note'));

    // Fill form
    await user.type(screen.getByLabelText(/title/i), 'New Note');
    await user.type(screen.getByLabelText(/notes/i), 'Note content');

    // Submit
    await user.click(screen.getByText('Add Entry'));

    await waitFor(() => {
      expect(api.createLabNotebookEntry).toHaveBeenCalledWith('project-1', {
        title: 'New Note',
        body_markdown: 'Note content',
        research_cycle_id: undefined,
        author_type: 'human',
      });
    });
  });

  it('creates new research cycle when clicking New Cycle button', async () => {
    const user = userEvent.setup();

    vi.mocked(api.createResearchCycle).mockResolvedValue({
      id: 'new-cycle',
      sequence_number: 3,
      status: 'pending',
      summary_title: null,
      created_at: '2024-01-17T10:00:00Z',
      updated_at: '2024-01-17T10:00:00Z',
      experiment_count: 0,
    });

    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('+ New Cycle')).toBeInTheDocument();
    });

    await user.click(screen.getByText('+ New Cycle'));

    await waitFor(() => {
      expect(api.createResearchCycle).toHaveBeenCalledWith('project-1');
    });
  });

  it('shows empty state when no entries exist', async () => {
    vi.mocked(api.listLabNotebookEntries).mockResolvedValue({
      entries: [],
      total: 0,
    });
    vi.mocked(api.listResearchCycles).mockResolvedValue({
      cycles: [],
      total: 0,
    });

    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('No notes yet')).toBeInTheDocument();
    });

    expect(screen.getByText('Add First Note')).toBeInTheDocument();
  });

  it('shows empty sidebar message when no cycles exist', async () => {
    vi.mocked(api.listResearchCycles).mockResolvedValue({
      cycles: [],
      total: 0,
    });

    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('No research cycles yet.')).toBeInTheDocument();
    });

    expect(screen.getByText(/cycles group related experiments/i)).toBeInTheDocument();
  });

  it('expands entry to show content when clicked', async () => {
    const user = userEvent.setup();

    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('Initial Hypothesis')).toBeInTheDocument();
    });

    // Click on the entry title to expand
    await user.click(screen.getByText('Initial Hypothesis'));

    // Should show the body content
    await waitFor(() => {
      expect(screen.getByText(/The target variable appears to correlate/)).toBeInTheDocument();
    });
  });

  it('shows delete confirmation when delete is clicked', async () => {
    const user = userEvent.setup();

    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('Initial Hypothesis')).toBeInTheDocument();
    });

    // Expand entry first
    await user.click(screen.getByText('Initial Hypothesis'));

    await waitFor(() => {
      expect(screen.getByText('Delete')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Delete'));

    // Should show confirmation dialog
    await waitFor(() => {
      expect(screen.getByText('Delete Entry')).toBeInTheDocument();
    });
  });

  it('displays error messages', async () => {
    vi.mocked(api.listResearchCycles).mockRejectedValue(
      new Error('Network error')
    );

    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      // Component shows generic error message for non-ApiException errors
      expect(screen.getByText('Failed to load research notebook')).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it('shows All Entries option and total count', async () => {
    render(<ResearchNotebook projectId="project-1" />);

    await waitFor(() => {
      expect(screen.getByText('All Entries')).toBeInTheDocument();
    });

    expect(screen.getByText('3 total notes')).toBeInTheDocument();
  });
});

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DatasetDiscoveryPanel from './DatasetDiscoveryPanel';

describe('DatasetDiscoveryPanel', () => {
  const mockOnSearch = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the discovery panel with form elements', () => {
    render(
      <DatasetDiscoveryPanel
        onSearch={mockOnSearch}
        isSearching={false}
        error={null}
      />
    );

    expect(screen.getByText('Find Public Datasets')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/house prices/i)).toBeInTheDocument();
    expect(screen.getByText('Search for Datasets')).toBeInTheDocument();
  });

  it('disables search button when description is empty', () => {
    render(
      <DatasetDiscoveryPanel
        onSearch={mockOnSearch}
        isSearching={false}
        error={null}
      />
    );

    const button = screen.getByText('Search for Datasets');
    expect(button).toBeDisabled();
  });

  it('enables search button when description is provided', async () => {
    const user = userEvent.setup();

    render(
      <DatasetDiscoveryPanel
        onSearch={mockOnSearch}
        isSearching={false}
        error={null}
      />
    );

    const textarea = screen.getByPlaceholderText(/house prices/i);
    await user.type(textarea, 'I want to predict customer churn');

    const button = screen.getByText('Search for Datasets');
    expect(button).not.toBeDisabled();
  });

  it('calls onSearch with description when form is submitted', async () => {
    const user = userEvent.setup();

    render(
      <DatasetDiscoveryPanel
        onSearch={mockOnSearch}
        isSearching={false}
        error={null}
      />
    );

    const textarea = screen.getByPlaceholderText(/house prices/i);
    await user.type(textarea, 'I want to predict customer churn');

    const button = screen.getByText('Search for Datasets');
    await user.click(button);

    expect(mockOnSearch).toHaveBeenCalledWith(
      'I want to predict customer churn',
      undefined
    );
  });

  it('shows advanced options when toggled', async () => {
    const user = userEvent.setup();

    render(
      <DatasetDiscoveryPanel
        onSearch={mockOnSearch}
        isSearching={false}
        error={null}
      />
    );

    expect(screen.queryByText('Geographic Preference')).not.toBeInTheDocument();

    const advancedToggle = screen.getByText(/Show advanced options/i);
    await user.click(advancedToggle);

    expect(screen.getByText('Geographic Preference (optional)')).toBeInTheDocument();
    expect(screen.getByText('Licensing Preferences (optional)')).toBeInTheDocument();
  });

  it('includes constraints in search when advanced options are filled', async () => {
    const user = userEvent.setup();

    render(
      <DatasetDiscoveryPanel
        onSearch={mockOnSearch}
        isSearching={false}
        error={null}
      />
    );

    // Show advanced options
    await user.click(screen.getByText(/Show advanced options/i));

    // Fill in description
    const textarea = screen.getByPlaceholderText(/house prices/i);
    await user.type(textarea, 'I want to predict house prices');

    // Fill in geography
    const geoInput = screen.getByPlaceholderText(/United States/i);
    await user.type(geoInput, 'California');

    // Toggle a licensing checkbox
    const openSourceCheckbox = screen.getByLabelText(/Open Source/i);
    await user.click(openSourceCheckbox);

    // Submit
    await user.click(screen.getByText('Search for Datasets'));

    expect(mockOnSearch).toHaveBeenCalledWith(
      'I want to predict house prices',
      {
        geography: 'California',
        allow_public_data: true,
        licensing_requirements: ['open-source'],
      }
    );
  });

  it('shows loading state when searching', () => {
    render(
      <DatasetDiscoveryPanel
        onSearch={mockOnSearch}
        isSearching={true}
        error={null}
      />
    );

    expect(screen.getByText(/Searching for datasets/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /searching/i })).toBeDisabled();
  });

  it('displays error message when error prop is set', () => {
    render(
      <DatasetDiscoveryPanel
        onSearch={mockOnSearch}
        isSearching={false}
        error="Dataset discovery failed: API error"
      />
    );

    expect(screen.getByText('Dataset discovery failed: API error')).toBeInTheDocument();
  });

  it('disables inputs during search', async () => {
    render(
      <DatasetDiscoveryPanel
        onSearch={mockOnSearch}
        isSearching={true}
        error={null}
      />
    );

    const textarea = screen.getByPlaceholderText(/house prices/i);
    expect(textarea).toBeDisabled();
  });
});

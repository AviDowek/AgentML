import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DiscoveredDatasetsList from './DiscoveredDatasetsList';
import type { DiscoveredDataset } from '../types/api';

const mockDatasets: DiscoveredDataset[] = [
  {
    name: 'California Housing Dataset',
    source_url: 'https://example.com/housing',
    schema_summary: {
      rows_estimate: 20640,
      columns: ['longitude', 'latitude', 'median_income', 'median_house_value'],
      target_candidate: 'median_house_value',
    },
    licensing: 'CC-BY-4.0',
    fit_for_purpose: 'Perfect for house price prediction with geographic and demographic features.',
  },
  {
    name: 'Boston Housing (Historical)',
    source_url: 'https://example.com/boston',
    schema_summary: {
      rows_estimate: 506,
      columns: ['CRIM', 'ZN', 'INDUS', 'MEDV'],
      target_candidate: 'MEDV',
    },
    licensing: 'Public Domain',
    fit_for_purpose: 'Classic regression dataset with housing prices.',
  },
  {
    name: 'Real Estate Valuation Dataset',
    source_url: 'https://example.com/realestate',
    fit_for_purpose: 'Taiwan real estate data for price prediction.',
  },
];

describe('DiscoveredDatasetsList', () => {
  const mockOnApply = vi.fn();
  const mockOnBack = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the list of discovered datasets', () => {
    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    expect(screen.getByText('Discovered Datasets')).toBeInTheDocument();
    expect(screen.getByText(/We found 3 relevant datasets/i)).toBeInTheDocument();
    expect(screen.getByText('California Housing Dataset')).toBeInTheDocument();
    expect(screen.getByText('Boston Housing (Historical)')).toBeInTheDocument();
    expect(screen.getByText('Real Estate Valuation Dataset')).toBeInTheDocument();
  });

  it('shows schema information for datasets that have it', () => {
    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    expect(screen.getByText(/~20,640 rows/i)).toBeInTheDocument();
    // Multiple datasets have "4 columns" so use getAllByText
    const columnElements = screen.getAllByText('4 columns');
    expect(columnElements.length).toBeGreaterThan(0);
    expect(screen.getByText('Target: median_house_value')).toBeInTheDocument();
  });

  it('displays licensing information', () => {
    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    expect(screen.getByText('CC-BY-4.0')).toBeInTheDocument();
    expect(screen.getByText('Public Domain')).toBeInTheDocument();
  });

  it('allows selecting datasets by clicking cards', async () => {
    const user = userEvent.setup();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    // Initially no datasets selected
    expect(screen.getByText('Download & Use 0 Selected Datasets')).toBeInTheDocument();
    expect(screen.getByText('Download & Use 0 Selected Datasets')).toBeDisabled();

    // Click on first dataset card
    await user.click(screen.getByText('California Housing Dataset'));

    // Button should update
    expect(screen.getByText('Download & Use 1 Selected Dataset')).toBeInTheDocument();
    expect(screen.getByText('Download & Use 1 Selected Dataset')).not.toBeDisabled();
  });

  it('allows selecting multiple datasets', async () => {
    const user = userEvent.setup();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    // Select two datasets
    await user.click(screen.getByText('California Housing Dataset'));
    await user.click(screen.getByText('Boston Housing (Historical)'));

    expect(screen.getByText('Download & Use 2 Selected Datasets')).toBeInTheDocument();
  });

  it('allows deselecting datasets', async () => {
    const user = userEvent.setup();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    // Select and then deselect
    await user.click(screen.getByText('California Housing Dataset'));
    expect(screen.getByText('Download & Use 1 Selected Dataset')).toBeInTheDocument();

    await user.click(screen.getByText('California Housing Dataset'));
    expect(screen.getByText('Download & Use 0 Selected Datasets')).toBeInTheDocument();
  });

  it('has Select All button that selects all datasets', async () => {
    const user = userEvent.setup();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    await user.click(screen.getByText('Select All'));

    expect(screen.getByText('Download & Use 3 Selected Datasets')).toBeInTheDocument();
  });

  it('has Clear button that clears all selections', async () => {
    const user = userEvent.setup();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    // Select all, then clear
    await user.click(screen.getByText('Select All'));
    expect(screen.getByText('Download & Use 3 Selected Datasets')).toBeInTheDocument();

    await user.click(screen.getByText('Clear'));
    expect(screen.getByText('Download & Use 0 Selected Datasets')).toBeInTheDocument();
  });

  it('calls onApply with selected indices when apply button is clicked', async () => {
    const user = userEvent.setup();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    // Select first and third datasets (indices 0 and 2)
    await user.click(screen.getByText('California Housing Dataset'));
    await user.click(screen.getByText('Real Estate Valuation Dataset'));

    await user.click(screen.getByText('Download & Use 2 Selected Datasets'));

    expect(mockOnApply).toHaveBeenCalledWith([0, 2]);
  });

  it('calls onBack when back button is clicked', async () => {
    const user = userEvent.setup();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    await user.click(screen.getByText('← Back to Search'));

    expect(mockOnBack).toHaveBeenCalled();
  });

  it('shows loading state when applying', () => {
    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={true}
        error={null}
      />
    );

    expect(screen.getByText('Downloading Datasets...')).toBeInTheDocument();
    expect(screen.getByText('← Back to Search')).toBeDisabled();
  });

  it('displays error when error prop is set', () => {
    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error="Failed to apply datasets"
      />
    );

    expect(screen.getByText('Failed to apply datasets')).toBeInTheDocument();
  });

  it('shows "Continue Anyway" button when error and onContinueWithoutDownload provided', () => {
    const mockOnContinue = vi.fn();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        onContinueWithoutDownload={mockOnContinue}
        isApplying={false}
        error="Download failed: Connection timed out"
      />
    );

    expect(screen.getByText('Download failed: Connection timed out')).toBeInTheDocument();
    expect(screen.getByText('Continue Anyway →')).toBeInTheDocument();
    expect(screen.getByText(/you can continue to your project and upload the dataset manually/i)).toBeInTheDocument();
  });

  it('calls onContinueWithoutDownload when "Continue Anyway" is clicked', async () => {
    const user = userEvent.setup();
    const mockOnContinue = vi.fn();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        onContinueWithoutDownload={mockOnContinue}
        isApplying={false}
        error="Download failed"
      />
    );

    await user.click(screen.getByText('Continue Anyway →'));

    expect(mockOnContinue).toHaveBeenCalled();
  });

  it('does not show "Continue Anyway" button when no error', () => {
    const mockOnContinue = vi.fn();

    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        onContinueWithoutDownload={mockOnContinue}
        isApplying={false}
        error={null}
      />
    );

    expect(screen.queryByText('Continue Anyway →')).not.toBeInTheDocument();
  });

  it('does not show "Continue Anyway" when onContinueWithoutDownload not provided', () => {
    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error="Download failed"
      />
    );

    expect(screen.getByText('Download failed')).toBeInTheDocument();
    expect(screen.queryByText('Continue Anyway →')).not.toBeInTheDocument();
  });

  it('has external links to dataset sources', () => {
    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    const sourceLinks = screen.getAllByText('View Source ↗');
    expect(sourceLinks).toHaveLength(3);

    const firstLink = sourceLinks[0].closest('a');
    expect(firstLink).toHaveAttribute('href', 'https://example.com/housing');
    expect(firstLink).toHaveAttribute('target', '_blank');
  });

  it('displays fit for purpose description', () => {
    render(
      <DiscoveredDatasetsList
        datasets={mockDatasets}
        onApply={mockOnApply}
        onBack={mockOnBack}
        isApplying={false}
        error={null}
      />
    );

    expect(screen.getByText(/Perfect for house price prediction/i)).toBeInTheDocument();
    expect(screen.getByText(/Classic regression dataset/i)).toBeInTheDocument();
  });
});

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ValidationSamplesTab from './ValidationSamplesTab';
import * as api from '../services/api';
import type { ValidationSample, ValidationSamplesListResponse, WhatIfResponse, ServingFeature } from '../types/api';

// Mock the API module
vi.mock('../services/api', () => ({
  listValidationSamples: vi.fn(),
  getValidationSample: vi.fn(),
  runWhatIfPrediction: vi.fn(),
  ApiException: class ApiException extends Error {
    status: number;
    detail: string;
    constructor(status: number, detail: string) {
      super(detail);
      this.status = status;
      this.detail = detail;
    }
  },
}));

const mockServingConfig: { features: ServingFeature[]; target_column: string; task_type: string } = {
  features: [
    { name: 'age', type: 'numeric' },
    { name: 'income', type: 'numeric' },
    { name: 'gender', type: 'categorical' },
    { name: 'education', type: 'categorical' },
    { name: 'occupation', type: 'categorical' },
    { name: 'location', type: 'categorical' },
  ],
  target_column: 'churn',
  task_type: 'binary',
};

const mockSamples: ValidationSample[] = [
  {
    id: 'sample-1',
    model_version_id: 'model-1',
    row_index: 0,
    features: { age: 25, income: 50000, gender: 'M', education: 'Bachelor', occupation: 'Engineer', location: 'NYC' },
    target_value: '0',
    predicted_value: '1',
    error_value: 1,
    absolute_error: 1,
    prediction_probabilities: { '0': 0.35, '1': 0.65 },
  },
  {
    id: 'sample-2',
    model_version_id: 'model-1',
    row_index: 1,
    features: { age: 45, income: 80000, gender: 'F', education: 'Masters', occupation: 'Manager', location: 'LA' },
    target_value: '1',
    predicted_value: '1',
    error_value: 0,
    absolute_error: 0,
    prediction_probabilities: { '0': 0.15, '1': 0.85 },
  },
  {
    id: 'sample-3',
    model_version_id: 'model-1',
    row_index: 2,
    features: { age: 35, income: 60000, gender: 'M', education: 'PhD', occupation: 'Scientist', location: 'Boston' },
    target_value: '0',
    predicted_value: '0',
    error_value: 0,
    absolute_error: 0,
    prediction_probabilities: { '0': 0.92, '1': 0.08 },
  },
];

const mockApiResponse: ValidationSamplesListResponse = {
  model_id: 'model-1',
  total: 100,
  limit: 50,
  offset: 0,
  samples: mockSamples,
};

describe('ValidationSamplesTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading spinner initially', async () => {
    vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    expect(screen.getByText(/Loading validation samples/i)).toBeInTheDocument();
  });

  it('displays validation samples table with correct values', async () => {
    vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      // Check table headers - first 5 features shown
      expect(screen.getByText('age')).toBeInTheDocument();
      expect(screen.getByText('income')).toBeInTheDocument();
      expect(screen.getByText('gender')).toBeInTheDocument();
      expect(screen.getByText('education')).toBeInTheDocument();
      expect(screen.getByText('occupation')).toBeInTheDocument();
    });

    // Check sample data in table - use getAllByText for values that appear multiple times
    expect(screen.getAllByText('25').length).toBeGreaterThan(0); // age from sample 1
    expect(screen.getAllByText('50000').length).toBeGreaterThan(0); // income from sample 1
    expect(screen.getAllByText('M').length).toBeGreaterThan(0); // gender from samples

    // Check target and predicted columns
    const rows = screen.getAllByRole('row');
    // Row 0 is header, row 1 is first sample
    const firstDataRow = rows[1];
    expect(within(firstDataRow).getAllByText('0').length).toBeGreaterThan(0); // row_index or target
    expect(within(firstDataRow).getAllByText('1').length).toBeGreaterThan(0); // predicted value
  });

  it('shows empty state when no samples available', async () => {
    vi.mocked(api.listValidationSamples).mockResolvedValue({
      model_id: 'model-1',
      total: 0,
      limit: 50,
      offset: 0,
      samples: [],
    });

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/No validation samples available/i)).toBeInTheDocument();
    });
  });

  it('shows error message when API fails', async () => {
    const mockError = new (api as typeof import('../services/api')).ApiException(500, 'Failed to load');
    vi.mocked(api.listValidationSamples).mockRejectedValue(mockError);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Failed to load/i)).toBeInTheDocument();
    });
  });

  it('displays pagination info correctly', async () => {
    vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Showing 1-50 of 100 samples/i)).toBeInTheDocument();
      expect(screen.getByText(/Page 1 of 2/i)).toBeInTheDocument();
    });
  });

  it('handles pagination next/prev correctly', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Next')).toBeInTheDocument();
    });

    // Previous should be disabled on first page
    const prevButton = screen.getByText('Previous');
    expect(prevButton).toBeDisabled();

    // Click Next
    await user.click(screen.getByText('Next'));

    // API should be called with offset = 50
    await waitFor(() => {
      expect(api.listValidationSamples).toHaveBeenCalledWith('model-1', {
        limit: 50,
        offset: 50,
        sort: 'error_desc',
      });
    });
  });

  it('changes sort order when dropdown is changed', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Sort by:')).toBeInTheDocument();
    });

    const sortSelect = screen.getByRole('combobox');
    await user.selectOptions(sortSelect, 'error_asc');

    await waitFor(() => {
      expect(api.listValidationSamples).toHaveBeenCalledWith('model-1', {
        limit: 50,
        offset: 0,
        sort: 'error_asc',
      });
    });
  });

  it('opens drawer with sample details when row is clicked', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('25')).toBeInTheDocument();
    });

    // Click on the first row
    const rows = screen.getAllByRole('row');
    await user.click(rows[1]); // First data row

    // Drawer should open with details
    await waitFor(() => {
      expect(screen.getByText('Validation Sample Details')).toBeInTheDocument();
      expect(screen.getByText('All Features')).toBeInTheDocument();
    });

    // Check that all features are displayed in drawer - use getAllBy since 'location' may appear multiple times
    expect(screen.getAllByText('location').length).toBeGreaterThan(0); // 6th feature in drawer
    expect(screen.getAllByText('NYC').length).toBeGreaterThan(0); // Value for location
  });

  it('displays prediction probabilities in drawer for classification', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('25')).toBeInTheDocument();
    });

    // Click on the first row
    const rows = screen.getAllByRole('row');
    await user.click(rows[1]);

    await waitFor(() => {
      expect(screen.getByText('Prediction Probabilities')).toBeInTheDocument();
      expect(screen.getByText('65.0%')).toBeInTheDocument(); // Probability for class 1
      expect(screen.getByText('35.0%')).toBeInTheDocument(); // Probability for class 0
    });
  });

  it('closes drawer when close button is clicked', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('25')).toBeInTheDocument();
    });

    // Click to open drawer
    const rows = screen.getAllByRole('row');
    await user.click(rows[1]);

    await waitFor(() => {
      expect(screen.getByText('Validation Sample Details')).toBeInTheDocument();
    });

    // Click close button
    await user.click(screen.getByText('×'));

    await waitFor(() => {
      expect(screen.queryByText('Validation Sample Details')).not.toBeInTheDocument();
    });
  });

  it('closes drawer when backdrop is clicked', async () => {
    const user = userEvent.setup();
    vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

    render(
      <ValidationSamplesTab
        modelId="model-1"
        servingConfig={mockServingConfig}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('25')).toBeInTheDocument();
    });

    // Click to open drawer
    const rows = screen.getAllByRole('row');
    await user.click(rows[1]);

    await waitFor(() => {
      expect(screen.getByText('Validation Sample Details')).toBeInTheDocument();
    });

    // Click backdrop
    const backdrop = document.querySelector('.sample-drawer-backdrop');
    if (backdrop) {
      await user.click(backdrop);
    }

    await waitFor(() => {
      expect(screen.queryByText('Validation Sample Details')).not.toBeInTheDocument();
    });
  });

  describe('What-If Testing', () => {
    it('shows what-if section in drawer', async () => {
      const user = userEvent.setup();
      vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

      render(
        <ValidationSamplesTab
          modelId="model-1"
          servingConfig={mockServingConfig}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('25')).toBeInTheDocument();
      });

      const rows = screen.getAllByRole('row');
      await user.click(rows[1]);

      await waitFor(() => {
        expect(screen.getByText('What-If Testing')).toBeInTheDocument();
        expect(screen.getByText('Recompute Prediction')).toBeInTheDocument();
      });
    });

    it('disables recompute button when no features are modified', async () => {
      const user = userEvent.setup();
      vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

      render(
        <ValidationSamplesTab
          modelId="model-1"
          servingConfig={mockServingConfig}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('25')).toBeInTheDocument();
      });

      const rows = screen.getAllByRole('row');
      await user.click(rows[1]);

      await waitFor(() => {
        const recomputeBtn = screen.getByText('Recompute Prediction');
        expect(recomputeBtn).toBeDisabled();
      });
    });

    it('enables recompute button when a feature is modified', async () => {
      const user = userEvent.setup();
      vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

      render(
        <ValidationSamplesTab
          modelId="model-1"
          servingConfig={mockServingConfig}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('25')).toBeInTheDocument();
      });

      const rows = screen.getAllByRole('row');
      await user.click(rows[1]);

      await waitFor(() => {
        expect(screen.getByText('What-If Testing')).toBeInTheDocument();
      });

      // Find age input and change it
      const ageInputs = screen.getAllByDisplayValue('25');
      await user.clear(ageInputs[0]);
      await user.type(ageInputs[0], '30');

      const recomputeBtn = screen.getByText('Recompute Prediction');
      expect(recomputeBtn).not.toBeDisabled();
    });

    it('calls runWhatIfPrediction when recompute is clicked', async () => {
      const user = userEvent.setup();
      vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

      const mockWhatIfResponse: WhatIfResponse = {
        original_sample: mockSamples[0],
        modified_features: { age: 30 },
        original_prediction: '1',
        modified_prediction: '0',
        prediction_delta: -1,
        original_probabilities: { '0': 0.35, '1': 0.65 },
        modified_probabilities: { '0': 0.75, '1': 0.25 },
      };
      vi.mocked(api.runWhatIfPrediction).mockResolvedValue(mockWhatIfResponse);

      render(
        <ValidationSamplesTab
          modelId="model-1"
          servingConfig={mockServingConfig}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('25')).toBeInTheDocument();
      });

      const rows = screen.getAllByRole('row');
      await user.click(rows[1]);

      await waitFor(() => {
        expect(screen.getByText('What-If Testing')).toBeInTheDocument();
      });

      // Modify age
      const ageInputs = screen.getAllByDisplayValue('25');
      await user.clear(ageInputs[0]);
      await user.type(ageInputs[0], '30');

      // Click recompute
      await user.click(screen.getByText('Recompute Prediction'));

      await waitFor(() => {
        expect(api.runWhatIfPrediction).toHaveBeenCalledWith('model-1', {
          sample_id: 'sample-1',
          modified_features: { age: 30 },
        });
      });
    });

    it('displays what-if results with prediction delta', async () => {
      const user = userEvent.setup();
      vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

      const mockWhatIfResponse: WhatIfResponse = {
        original_sample: mockSamples[0],
        modified_features: { age: 30 },
        original_prediction: '1',
        modified_prediction: '0',
        prediction_delta: -1,
        original_probabilities: { '0': 0.35, '1': 0.65 },
        modified_probabilities: { '0': 0.75, '1': 0.25 },
      };
      vi.mocked(api.runWhatIfPrediction).mockResolvedValue(mockWhatIfResponse);

      render(
        <ValidationSamplesTab
          modelId="model-1"
          servingConfig={mockServingConfig}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('25')).toBeInTheDocument();
      });

      const rows = screen.getAllByRole('row');
      await user.click(rows[1]);

      await waitFor(() => {
        expect(screen.getByText('What-If Testing')).toBeInTheDocument();
      });

      // Modify age
      const ageInputs = screen.getAllByDisplayValue('25');
      await user.clear(ageInputs[0]);
      await user.type(ageInputs[0], '30');

      // Click recompute
      await user.click(screen.getByText('Recompute Prediction'));

      await waitFor(() => {
        expect(screen.getByText('What-If Result')).toBeInTheDocument();
        expect(screen.getByText('Original Prediction:')).toBeInTheDocument();
        expect(screen.getByText('New Prediction:')).toBeInTheDocument();
        expect(screen.getByText('Change (Delta):')).toBeInTheDocument();
        expect(screen.getByText('-1.0000')).toBeInTheDocument(); // Delta value
      });
    });

    it('displays what-if error when API fails', async () => {
      const user = userEvent.setup();
      vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

      const mockError = new (api as typeof import('../services/api')).ApiException(500, 'Model not loaded');
      vi.mocked(api.runWhatIfPrediction).mockRejectedValue(mockError);

      render(
        <ValidationSamplesTab
          modelId="model-1"
          servingConfig={mockServingConfig}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('25')).toBeInTheDocument();
      });

      const rows = screen.getAllByRole('row');
      await user.click(rows[1]);

      await waitFor(() => {
        expect(screen.getByText('What-If Testing')).toBeInTheDocument();
      });

      // Modify age
      const ageInputs = screen.getAllByDisplayValue('25');
      await user.clear(ageInputs[0]);
      await user.type(ageInputs[0], '30');

      // Click recompute
      await user.click(screen.getByText('Recompute Prediction'));

      await waitFor(() => {
        expect(screen.getByText('Model not loaded')).toBeInTheDocument();
      });
    });

    it('shows modified probabilities in what-if results', async () => {
      const user = userEvent.setup();
      vi.mocked(api.listValidationSamples).mockResolvedValue(mockApiResponse);

      const mockWhatIfResponse: WhatIfResponse = {
        original_sample: mockSamples[0],
        modified_features: { age: 30 },
        original_prediction: '1',
        modified_prediction: '0',
        prediction_delta: -1,
        original_probabilities: { '0': 0.35, '1': 0.65 },
        modified_probabilities: { '0': 0.75, '1': 0.25 },
      };
      vi.mocked(api.runWhatIfPrediction).mockResolvedValue(mockWhatIfResponse);

      render(
        <ValidationSamplesTab
          modelId="model-1"
          servingConfig={mockServingConfig}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('25')).toBeInTheDocument();
      });

      const rows = screen.getAllByRole('row');
      await user.click(rows[1]);

      await waitFor(() => {
        expect(screen.getByText('What-If Testing')).toBeInTheDocument();
      });

      // Modify age
      const ageInputs = screen.getAllByDisplayValue('25');
      await user.clear(ageInputs[0]);
      await user.type(ageInputs[0], '30');

      // Click recompute
      await user.click(screen.getByText('Recompute Prediction'));

      await waitFor(() => {
        expect(screen.getByText('New Probabilities')).toBeInTheDocument();
        expect(screen.getByText('75.0%')).toBeInTheDocument(); // New probability for class 0
        expect(screen.getByText('25.0%')).toBeInTheDocument(); // New probability for class 1
      });
    });
  });

  describe('Regression task type', () => {
    it('displays regression samples without probabilities', async () => {
      const regressionSamples: ValidationSample[] = [
        {
          id: 'sample-reg-1',
          model_version_id: 'model-1',
          row_index: 0,
          features: { sqft: 2000, bedrooms: 3, location: 'suburb' },
          target_value: '350000',
          predicted_value: '345000',
          error_value: -5000,
          absolute_error: 5000,
          prediction_probabilities: null,
        },
      ];

      vi.mocked(api.listValidationSamples).mockResolvedValue({
        model_id: 'model-1',
        total: 1,
        limit: 50,
        offset: 0,
        samples: regressionSamples,
      });

      const regressionConfig = {
        features: [
          { name: 'sqft', type: 'numeric' as const },
          { name: 'bedrooms', type: 'numeric' as const },
          { name: 'location', type: 'categorical' as const },
        ],
        target_column: 'price',
        task_type: 'regression',
      };

      render(
        <ValidationSamplesTab
          modelId="model-1"
          servingConfig={regressionConfig}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('2000')).toBeInTheDocument(); // sqft
        expect(screen.getByText('350000')).toBeInTheDocument(); // target
        expect(screen.getByText('345000')).toBeInTheDocument(); // predicted
      });

      // No probability section should exist
      expect(screen.queryByText('Prediction Probabilities')).not.toBeInTheDocument();
    });
  });
});

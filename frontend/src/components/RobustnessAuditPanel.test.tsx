import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RobustnessAuditPanel } from './RobustnessAuditPanel';
import type { RobustnessAuditResult, SuspiciousPattern } from './RobustnessAuditPanel';

// Helper to create a mock audit result
const createMockAudit = (overrides: Partial<RobustnessAuditResult> = {}): RobustnessAuditResult => ({
  overfitting_risk: 'low',
  suspicious_patterns: [],
  recommendations: [],
  natural_language_summary: 'Model appears to be well-trained with no significant issues.',
  ...overrides,
});

describe('RobustnessAuditPanel', () => {
  describe('Risk Level Display', () => {
    it('renders LOW risk with green styling and checkmark', () => {
      const audit = createMockAudit({ overfitting_risk: 'low' });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('LOW RISK')).toBeInTheDocument();
      expect(screen.getByText('✅')).toBeInTheDocument();
    });

    it('renders MEDIUM risk with amber styling and warning', () => {
      const audit = createMockAudit({ overfitting_risk: 'medium' });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('MEDIUM RISK')).toBeInTheDocument();
      expect(screen.getByText('⚠️')).toBeInTheDocument();
    });

    it('renders HIGH risk with red styling and alert', () => {
      const audit = createMockAudit({ overfitting_risk: 'high' });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('HIGH RISK')).toBeInTheDocument();
      expect(screen.getByText('🚨')).toBeInTheDocument();
    });

    it('renders UNKNOWN risk with gray styling and question mark', () => {
      const audit = createMockAudit({ overfitting_risk: 'unknown' });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('UNKNOWN RISK')).toBeInTheDocument();
      expect(screen.getByText('❓')).toBeInTheDocument();
    });
  });

  describe('Summary Display', () => {
    it('renders the natural language summary', () => {
      const audit = createMockAudit({
        natural_language_summary: 'The model shows signs of overfitting with a large train-val gap.',
      });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(
        screen.getByText('The model shows signs of overfitting with a large train-val gap.')
      ).toBeInTheDocument();
    });

    it('does not render summary section if empty', () => {
      const audit = createMockAudit({ natural_language_summary: '' });
      render(<RobustnessAuditPanel audit={audit} />);

      // Should not have a summary div with empty content
      const summaryText = 'The model shows signs';
      expect(screen.queryByText(summaryText)).not.toBeInTheDocument();
    });
  });

  describe('Suspicious Patterns', () => {
    const mockPatterns: SuspiciousPattern[] = [
      {
        type: 'train_val_gap',
        severity: 'high',
        description: 'Training accuracy 0.99 vs validation 0.72 - gap of 0.27',
      },
      {
        type: 'baseline_concern',
        severity: 'medium',
        description: 'Model only 5% better than random baseline',
      },
      {
        type: 'cv_variance',
        severity: 'low',
        description: 'Minor variance across CV folds',
      },
    ];

    it('renders count of suspicious patterns', () => {
      const audit = createMockAudit({ suspicious_patterns: mockPatterns });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('Suspicious Patterns (3)')).toBeInTheDocument();
    });

    it('renders each pattern with correct severity badge', () => {
      const audit = createMockAudit({ suspicious_patterns: mockPatterns });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('HIGH')).toBeInTheDocument();
      expect(screen.getByText('MEDIUM')).toBeInTheDocument();
      expect(screen.getByText('LOW')).toBeInTheDocument();
    });

    it('renders pattern descriptions', () => {
      const audit = createMockAudit({ suspicious_patterns: mockPatterns });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(
        screen.getByText('Training accuracy 0.99 vs validation 0.72 - gap of 0.27')
      ).toBeInTheDocument();
      expect(
        screen.getByText('Model only 5% better than random baseline')
      ).toBeInTheDocument();
    });

    it('renders pattern type icons', () => {
      const audit = createMockAudit({ suspicious_patterns: mockPatterns });
      render(<RobustnessAuditPanel audit={audit} />);

      // train_val_gap gets 📊, baseline_concern gets 📉, cv_variance gets 📏
      expect(screen.getByText('📊')).toBeInTheDocument();
      expect(screen.getByText('📉')).toBeInTheDocument();
      expect(screen.getByText('📏')).toBeInTheDocument();
    });

    it('does not render patterns section when empty', () => {
      const audit = createMockAudit({ suspicious_patterns: [] });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.queryByText(/Suspicious Patterns/)).not.toBeInTheDocument();
    });

    it('renders data leakage pattern with lock icon', () => {
      const audit = createMockAudit({
        suspicious_patterns: [
          {
            type: 'data_leakage_suspicion',
            severity: 'high',
            description: 'Perfect validation score suggests data leakage',
          },
        ],
      });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('🔓')).toBeInTheDocument();
      expect(screen.getByText('data leakage suspicion')).toBeInTheDocument();
    });
  });

  describe('Train-Validation Analysis', () => {
    it('renders train-val section when data is provided', async () => {
      const user = userEvent.setup();
      const audit = createMockAudit({
        train_val_analysis: {
          worst_gap: 0.27,
          avg_gap: 0.15,
          interpretation: 'Significant overfitting detected',
        },
      });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('📊 Train-Validation Gap Analysis')).toBeInTheDocument();

      // Expand section
      await user.click(screen.getByText('📊 Train-Validation Gap Analysis'));

      expect(screen.getByText('0.2700')).toBeInTheDocument(); // worst_gap
      expect(screen.getByText('0.1500')).toBeInTheDocument(); // avg_gap
      expect(screen.getByText('Significant overfitting detected')).toBeInTheDocument();
    });

    it('does not render train-val section when not provided', () => {
      const audit = createMockAudit({ train_val_analysis: undefined });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.queryByText('Train-Validation Gap Analysis')).not.toBeInTheDocument();
    });
  });

  describe('Baseline Comparison', () => {
    it('renders baseline comparison when data is provided', async () => {
      const user = userEvent.setup();
      const audit = createMockAudit({
        baseline_comparison: {
          baseline_type: 'majority_class',
          baseline_metric: 0.65,
          best_model_metric: 0.72,
          relative_improvement: 0.107,
          interpretation: 'Model shows meaningful improvement over baseline',
        },
      });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('📉 Baseline Comparison')).toBeInTheDocument();

      // Expand section
      await user.click(screen.getByText('📉 Baseline Comparison'));

      expect(screen.getByText('majority class')).toBeInTheDocument();
      expect(screen.getByText('0.6500')).toBeInTheDocument(); // baseline_metric
      expect(screen.getByText('0.7200')).toBeInTheDocument(); // best_model_metric
      expect(screen.getByText('10.7%')).toBeInTheDocument(); // relative_improvement * 100
      expect(screen.getByText('Model shows meaningful improvement over baseline')).toBeInTheDocument();
    });

    it('does not render baseline section when type is none_available', () => {
      const audit = createMockAudit({
        baseline_comparison: {
          baseline_type: 'none_available',
        },
      });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.queryByText('Baseline Comparison')).not.toBeInTheDocument();
    });
  });

  describe('CV Analysis', () => {
    it('renders CV analysis when data is provided', async () => {
      const user = userEvent.setup();
      const audit = createMockAudit({
        cv_analysis: {
          fold_variance: 0.03,
          interpretation: 'Low variance indicates stable model',
        },
      });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('📏 Cross-Validation Analysis')).toBeInTheDocument();

      // Expand section
      await user.click(screen.getByText('📏 Cross-Validation Analysis'));

      expect(screen.getByText('0.0300')).toBeInTheDocument();
      expect(screen.getByText('Low variance indicates stable model')).toBeInTheDocument();
    });

    it('does not render CV section when variance is undefined', () => {
      const audit = createMockAudit({
        cv_analysis: {
          interpretation: 'Some interpretation',
        },
      });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.queryByText('Cross-Validation Analysis')).not.toBeInTheDocument();
    });
  });

  describe('Recommendations', () => {
    it('renders recommendations list', () => {
      const audit = createMockAudit({
        recommendations: [
          'Try adding regularization (L1/L2) to reduce overfitting',
          'Consider collecting more training data',
          'Use early stopping during training',
        ],
      });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.getByText('Recommendations')).toBeInTheDocument();
      expect(
        screen.getByText('Try adding regularization (L1/L2) to reduce overfitting')
      ).toBeInTheDocument();
      expect(screen.getByText('Consider collecting more training data')).toBeInTheDocument();
      expect(screen.getByText('Use early stopping during training')).toBeInTheDocument();
    });

    it('does not render recommendations section when empty', () => {
      const audit = createMockAudit({ recommendations: [] });
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.queryByText('Recommendations')).not.toBeInTheDocument();
    });
  });

  describe('Show Full Thinking', () => {
    it('renders "Show full thinking" button when callback is provided', () => {
      const onShowFullThinking = vi.fn();
      const audit = createMockAudit();
      render(<RobustnessAuditPanel audit={audit} onShowFullThinking={onShowFullThinking} />);

      expect(screen.getByText('Show full agent thinking')).toBeInTheDocument();
    });

    it('calls onShowFullThinking when button is clicked', async () => {
      const user = userEvent.setup();
      const onShowFullThinking = vi.fn();
      const audit = createMockAudit();
      render(<RobustnessAuditPanel audit={audit} onShowFullThinking={onShowFullThinking} />);

      await user.click(screen.getByText('Show full agent thinking'));

      expect(onShowFullThinking).toHaveBeenCalledTimes(1);
    });

    it('does not render button when callback is not provided', () => {
      const audit = createMockAudit();
      render(<RobustnessAuditPanel audit={audit} />);

      expect(screen.queryByText('Show full agent thinking')).not.toBeInTheDocument();
    });
  });

  describe('Complete Audit Scenario', () => {
    it('renders a full audit with all sections', async () => {
      const user = userEvent.setup();
      const audit: RobustnessAuditResult = {
        overfitting_risk: 'high',
        natural_language_summary:
          'This model shows significant overfitting with a large gap between training and validation performance.',
        train_val_analysis: {
          worst_gap: 0.27,
          avg_gap: 0.18,
          interpretation: 'The model memorizes training data but fails to generalize.',
        },
        suspicious_patterns: [
          {
            type: 'train_val_gap',
            severity: 'high',
            description: 'Train accuracy 0.99, validation accuracy 0.72',
          },
          {
            type: 'metric_spike',
            severity: 'medium',
            description: 'Sudden improvement in epoch 15 suggests potential data leakage',
          },
        ],
        baseline_comparison: {
          baseline_type: 'majority_class',
          baseline_metric: 0.65,
          best_model_metric: 0.72,
          relative_improvement: 0.107,
          interpretation: 'Only marginally better than predicting the majority class.',
        },
        cv_analysis: {
          fold_variance: 0.08,
          interpretation: 'Moderate variance across folds suggests unstable model.',
        },
        recommendations: [
          'Add L2 regularization',
          'Increase dropout rate',
          'Use data augmentation',
        ],
      };

      const onShowFullThinking = vi.fn();
      render(<RobustnessAuditPanel audit={audit} onShowFullThinking={onShowFullThinking} />);

      // Check risk level
      expect(screen.getByText('HIGH RISK')).toBeInTheDocument();
      expect(screen.getByText('🚨')).toBeInTheDocument();

      // Check summary
      expect(
        screen.getByText(/This model shows significant overfitting/)
      ).toBeInTheDocument();

      // Check patterns count
      expect(screen.getByText('Suspicious Patterns (2)')).toBeInTheDocument();

      // Check recommendations
      expect(screen.getByText('Add L2 regularization')).toBeInTheDocument();

      // Expand train-val section
      await user.click(screen.getByText('📊 Train-Validation Gap Analysis'));
      expect(screen.getByText('0.2700')).toBeInTheDocument();

      // Expand baseline section
      await user.click(screen.getByText('📉 Baseline Comparison'));
      expect(screen.getByText('10.7%')).toBeInTheDocument();

      // Expand CV section
      await user.click(screen.getByText('📏 Cross-Validation Analysis'));
      expect(screen.getByText('0.0800')).toBeInTheDocument();

      // Check full thinking button
      expect(screen.getByText('Show full agent thinking')).toBeInTheDocument();
    });
  });

  describe('Expandable Sections', () => {
    it('toggles section expansion when clicked twice', async () => {
      const user = userEvent.setup();
      const audit = createMockAudit({
        train_val_analysis: {
          worst_gap: 0.27,
          interpretation: 'Test interpretation',
        },
      });
      render(<RobustnessAuditPanel audit={audit} />);

      const sectionButton = screen.getByText('📊 Train-Validation Gap Analysis');

      // Initially collapsed (shows +)
      expect(screen.getByText('+')).toBeInTheDocument();

      // Click to expand
      await user.click(sectionButton);
      expect(screen.getByText('−')).toBeInTheDocument();
      expect(screen.getByText('Test interpretation')).toBeInTheDocument();

      // Click to collapse
      await user.click(sectionButton);
      expect(screen.getByText('+')).toBeInTheDocument();
      expect(screen.queryByText('Test interpretation')).not.toBeInTheDocument();
    });

    it('only shows one expanded section at a time', async () => {
      const user = userEvent.setup();
      const audit = createMockAudit({
        train_val_analysis: {
          worst_gap: 0.27,
          interpretation: 'Train-val interpretation',
        },
        baseline_comparison: {
          baseline_type: 'majority_class',
          interpretation: 'Baseline interpretation',
        },
      });
      render(<RobustnessAuditPanel audit={audit} />);

      // Expand train-val section
      await user.click(screen.getByText('📊 Train-Validation Gap Analysis'));
      expect(screen.getByText('Train-val interpretation')).toBeInTheDocument();

      // Expand baseline section (should collapse train-val)
      await user.click(screen.getByText('📉 Baseline Comparison'));
      expect(screen.getByText('Baseline interpretation')).toBeInTheDocument();
      expect(screen.queryByText('Train-val interpretation')).not.toBeInTheDocument();
    });
  });
});

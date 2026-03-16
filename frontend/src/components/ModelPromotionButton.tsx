/**
 * Model Promotion Button Component (Prompt 5)
 *
 * Displays a promotion button with risk status indicator.
 * For high-risk models, shows a warning and requires an override reason.
 */
import React, { useState, useEffect } from 'react';
import type { ModelRiskStatus } from '../types/api';

interface ModelPromotionButtonProps {
  modelId: string;
  currentStatus: string;
  onPromote: (status: string, overrideReason?: string) => Promise<void>;
  className?: string;
}

const getRiskLevelColor = (level: string): string => {
  switch (level) {
    case 'low':
      return '#22c55e'; // green
    case 'medium':
      return '#f59e0b'; // amber
    case 'high':
      return '#ef4444'; // red
    case 'critical':
      return '#dc2626'; // dark red
    default:
      return '#6b7280'; // gray
  }
};

const getRiskLevelEmoji = (level: string): string => {
  switch (level) {
    case 'low':
      return '✅';
    case 'medium':
      return '⚠️';
    case 'high':
      return '🚨';
    case 'critical':
      return '⛔';
    default:
      return '❓';
  }
};

export const ModelPromotionButton: React.FC<ModelPromotionButtonProps> = ({
  modelId,
  currentStatus,
  onPromote,
  className = '',
}) => {
  const [riskStatus, setRiskStatus] = useState<ModelRiskStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [showOverrideModal, setShowOverrideModal] = useState(false);
  const [overrideReason, setOverrideReason] = useState('');
  const [targetStatus, setTargetStatus] = useState<string>('');
  const [promoting, setPromoting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch risk status when component mounts
  useEffect(() => {
    const fetchRiskStatus = async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/models/${modelId}/risk-status`);
        if (response.ok) {
          const data = await response.json();
          setRiskStatus(data);
        }
      } catch (err) {
        console.error('Failed to fetch risk status:', err);
      } finally {
        setLoading(false);
      }
    };

    if (modelId) {
      fetchRiskStatus();
    }
  }, [modelId]);

  const handlePromoteClick = (status: string) => {
    setTargetStatus(status);
    setError(null);

    if (riskStatus?.requires_override) {
      setShowOverrideModal(true);
    } else {
      handlePromote(status);
    }
  };

  const handlePromote = async (status: string, reason?: string) => {
    setPromoting(true);
    setError(null);
    try {
      await onPromote(status, reason);
      setShowOverrideModal(false);
      setOverrideReason('');
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to promote model';
      setError(errorMessage);
    } finally {
      setPromoting(false);
    }
  };

  const handleOverrideSubmit = () => {
    if (overrideReason.length < 10) {
      setError('Override reason must be at least 10 characters');
      return;
    }
    handlePromote(targetStatus, overrideReason);
  };

  const canPromoteTo = (status: string): boolean => {
    const statusOrder = ['registered', 'candidate', 'shadow', 'production'];
    const currentIndex = statusOrder.indexOf(currentStatus);
    const targetIndex = statusOrder.indexOf(status);
    return targetIndex > currentIndex;
  };

  return (
    <div className={`model-promotion ${className}`}>
      {/* Risk Status Badge */}
      {riskStatus && riskStatus.risk_level !== 'unknown' && (
        <div
          className="risk-status-badge mb-3 p-2 rounded-lg flex items-center gap-2"
          style={{
            backgroundColor: `${getRiskLevelColor(riskStatus.risk_level)}15`,
            border: `1px solid ${getRiskLevelColor(riskStatus.risk_level)}40`,
          }}
        >
          <span className="text-lg">{getRiskLevelEmoji(riskStatus.risk_level)}</span>
          <div className="flex-1">
            <span
              className="text-sm font-medium"
              style={{ color: getRiskLevelColor(riskStatus.risk_level) }}
            >
              {riskStatus.risk_level.toUpperCase()} RISK
            </span>
            {riskStatus.risk_adjusted_score !== null && (
              <span className="text-xs text-gray-400 ml-2">
                (Risk-adjusted: {riskStatus.risk_adjusted_score.toFixed(3)})
              </span>
            )}
          </div>
          {riskStatus.requires_override && (
            <span className="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-400">
              Override Required
            </span>
          )}
        </div>
      )}

      {/* Risk Details (collapsible) */}
      {riskStatus && (riskStatus.leakage_suspected || riskStatus.too_good_to_be_true || riskStatus.time_split_suspicious) && (
        <div className="risk-details mb-3 text-sm space-y-1">
          {riskStatus.leakage_suspected && (
            <div className="flex items-center gap-2 text-red-400">
              <span>🔓</span>
              <span>Data leakage suspected</span>
            </div>
          )}
          {riskStatus.too_good_to_be_true && (
            <div className="flex items-center gap-2 text-purple-400">
              <span>🎯</span>
              <span>Performance too good to be true</span>
            </div>
          )}
          {riskStatus.time_split_suspicious && (
            <div className="flex items-center gap-2 text-amber-400">
              <span>⏰</span>
              <span>Time-split issue detected</span>
            </div>
          )}
        </div>
      )}

      {/* Promotion Buttons */}
      <div className="promotion-buttons flex gap-2">
        {canPromoteTo('candidate') && (
          <button
            onClick={() => handlePromoteClick('candidate')}
            disabled={promoting || loading}
            className="px-3 py-1.5 text-sm rounded bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {promoting && targetStatus === 'candidate' ? 'Promoting...' : 'Promote to Candidate'}
          </button>
        )}
        {canPromoteTo('shadow') && (
          <button
            onClick={() => handlePromoteClick('shadow')}
            disabled={promoting || loading}
            className="px-3 py-1.5 text-sm rounded bg-purple-600 hover:bg-purple-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {promoting && targetStatus === 'shadow' ? 'Promoting...' : 'Promote to Shadow'}
          </button>
        )}
        {canPromoteTo('production') && (
          <button
            onClick={() => handlePromoteClick('production')}
            disabled={promoting || loading}
            className="px-3 py-1.5 text-sm rounded bg-green-600 hover:bg-green-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {promoting && targetStatus === 'production' ? 'Promoting...' : 'Promote to Production'}
          </button>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mt-2 p-2 rounded bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Override Modal */}
      {showOverrideModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-lg w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <span>⚠️</span>
              High-Risk Model Promotion
            </h3>

            <div
              className="p-3 rounded-lg mb-4"
              style={{
                backgroundColor: `${getRiskLevelColor(riskStatus?.risk_level || 'unknown')}15`,
                border: `1px solid ${getRiskLevelColor(riskStatus?.risk_level || 'unknown')}40`,
              }}
            >
              <div className="text-sm font-medium text-amber-400 mb-2">
                Identified Risks:
              </div>
              <div className="text-sm text-gray-300">{riskStatus?.risk_reason}</div>
            </div>

            <div className="mb-4">
              <label className="block text-sm text-gray-400 mb-2">
                Override Reason <span className="text-red-400">*</span>
              </label>
              <textarea
                value={overrideReason}
                onChange={(e) => setOverrideReason(e.target.value)}
                placeholder="Explain why you believe this model is safe to promote despite the identified risks..."
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm resize-none"
                rows={4}
              />
              <div className="text-xs text-gray-500 mt-1">
                Minimum 10 characters. This will be logged in the lab notebook.
              </div>
            </div>

            {error && (
              <div className="mb-4 p-2 rounded bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
                {error}
              </div>
            )}

            <div className="flex justify-end gap-3">
              <button
                onClick={() => {
                  setShowOverrideModal(false);
                  setOverrideReason('');
                  setError(null);
                }}
                className="px-4 py-2 text-sm rounded bg-gray-600 hover:bg-gray-500 text-white"
              >
                Cancel
              </button>
              <button
                onClick={handleOverrideSubmit}
                disabled={promoting || overrideReason.length < 10}
                className="px-4 py-2 text-sm rounded bg-amber-600 hover:bg-amber-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {promoting ? 'Promoting...' : 'Promote with Override'}
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .model-promotion {
          padding: 12px;
          background: rgba(31, 41, 55, 0.5);
          border-radius: 8px;
        }
      `}</style>
    </div>
  );
};

export default ModelPromotionButton;

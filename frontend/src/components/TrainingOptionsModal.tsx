/**
 * Training Options Modal Component
 * Allows users to configure training options before running an experiment
 */

import { useState, useEffect } from 'react';
import { getTrainingOptions } from '../services/api';
import type { TrainingOptionsInfo } from '../types/api';

interface TrainingOptionsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (options: {
    backend: 'local' | 'modal';
    resourceLimitsEnabled: boolean;
    numCpus?: number;
    numGpus?: number;
    memoryLimitGb?: number;
  }) => void;
  experimentName?: string;
}

export function TrainingOptionsModal({
  isOpen,
  onClose,
  onConfirm,
  experimentName,
}: TrainingOptionsModalProps) {
  const [trainingInfo, setTrainingInfo] = useState<TrainingOptionsInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [backend, setBackend] = useState<'local' | 'modal'>('local');
  const [resourceLimitsEnabled, setResourceLimitsEnabled] = useState(true);
  const [numCpus, setNumCpus] = useState<number | undefined>(undefined);
  const [numGpus, setNumGpus] = useState<number | undefined>(undefined);
  const [memoryLimitGb, setMemoryLimitGb] = useState<number | undefined>(undefined);

  useEffect(() => {
    if (isOpen) {
      loadTrainingOptions();
    }
  }, [isOpen]);

  const loadTrainingOptions = async () => {
    try {
      setLoading(true);
      const info = await getTrainingOptions();
      setTrainingInfo(info);
      setResourceLimitsEnabled(info.resource_limits.enabled_by_default);
      setNumCpus(info.resource_limits.defaults.num_cpus);
      setMemoryLimitGb(info.resource_limits.defaults.memory_limit_gb);
      setError(null);
    } catch (err) {
      setError('Failed to load training options');
      console.error('Failed to load training options:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleConfirm = () => {
    onConfirm({
      backend,
      resourceLimitsEnabled,
      numCpus: resourceLimitsEnabled ? numCpus : undefined,
      numGpus: resourceLimitsEnabled ? numGpus : undefined,
      memoryLimitGb: resourceLimitsEnabled ? memoryLimitGb : undefined,
    });
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          padding: '24px',
          maxWidth: '500px',
          width: '90%',
          maxHeight: '80vh',
          overflow: 'auto',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 style={{ marginTop: 0, marginBottom: '16px' }}>
          Training Options
        </h2>

        {experimentName && (
          <p style={{ color: '#666', marginBottom: '16px' }}>
            Running: <strong>{experimentName}</strong>
          </p>
        )}

        {loading ? (
          <p>Loading options...</p>
        ) : error ? (
          <p style={{ color: 'red' }}>{error}</p>
        ) : (
          <div>
            {/* Backend Selection */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>
                Training Backend
              </label>
              <div style={{ display: 'flex', gap: '12px' }}>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                  <input
                    type="radio"
                    name="backend"
                    value="local"
                    checked={backend === 'local'}
                    onChange={() => setBackend('local')}
                    style={{ marginRight: '8px' }}
                  />
                  <span>Local (Celery)</span>
                </label>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    cursor: trainingInfo?.backends.modal.available ? 'pointer' : 'not-allowed',
                    opacity: trainingInfo?.backends.modal.available ? 1 : 0.5,
                  }}
                >
                  <input
                    type="radio"
                    name="backend"
                    value="modal"
                    checked={backend === 'modal'}
                    onChange={() => setBackend('modal')}
                    disabled={!trainingInfo?.backends.modal.available}
                    style={{ marginRight: '8px' }}
                  />
                  <span>Cloud (Modal.com)</span>
                </label>
              </div>
              {backend === 'local' && (
                <p style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                  {trainingInfo?.backends.local.description}
                </p>
              )}
              {backend === 'modal' && trainingInfo?.backends.modal.available && (
                <p style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                  {trainingInfo?.backends.modal.description}
                </p>
              )}
              {!trainingInfo?.backends.modal.available && (
                <p style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
                  Modal not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env
                </p>
              )}
            </div>

            {/* Resource Limits - only for local backend */}
            {backend === 'local' && (
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'flex', alignItems: 'center', fontWeight: 'bold', marginBottom: '8px' }}>
                  <input
                    type="checkbox"
                    checked={resourceLimitsEnabled}
                    onChange={(e) => setResourceLimitsEnabled(e.target.checked)}
                    style={{ marginRight: '8px' }}
                  />
                  Enable Resource Limits
                </label>
                <p style={{ fontSize: '12px', color: '#666', marginTop: '4px', marginLeft: '24px' }}>
                  {trainingInfo?.resource_limits.description}
                </p>

                {resourceLimitsEnabled && (
                  <div style={{ marginTop: '12px', marginLeft: '24px' }}>
                    <div style={{ marginBottom: '12px' }}>
                      <label style={{ display: 'block', marginBottom: '4px' }}>
                        CPU Cores: {numCpus || 'auto'}
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="16"
                        value={numCpus || 2}
                        onChange={(e) => setNumCpus(parseInt(e.target.value))}
                        style={{ width: '100%' }}
                      />
                    </div>

                    <div style={{ marginBottom: '12px' }}>
                      <label style={{ display: 'block', marginBottom: '4px' }}>
                        Memory Limit: {memoryLimitGb || 'auto'} GB
                      </label>
                      <input
                        type="range"
                        min="2"
                        max="32"
                        value={memoryLimitGb || 8}
                        onChange={(e) => setMemoryLimitGb(parseInt(e.target.value))}
                        style={{ width: '100%' }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', marginBottom: '4px' }}>
                        GPUs: {numGpus || 0}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="4"
                        value={numGpus || 0}
                        onChange={(e) => setNumGpus(parseInt(e.target.value))}
                        style={{ width: '100%' }}
                      />
                    </div>
                  </div>
                )}

                {!resourceLimitsEnabled && (
                  <p style={{
                    fontSize: '12px',
                    color: '#e65100',
                    marginTop: '8px',
                    marginLeft: '24px',
                    backgroundColor: '#fff3e0',
                    padding: '8px',
                    borderRadius: '4px',
                  }}>
                    Warning: Disabling resource limits may cause system freezes on machines with limited RAM.
                  </p>
                )}
              </div>
            )}

            {backend === 'modal' && (
              <p style={{
                fontSize: '14px',
                color: '#1976d2',
                backgroundColor: '#e3f2fd',
                padding: '12px',
                borderRadius: '4px',
              }}>
                Cloud training runs with full resources - no limits needed!
              </p>
            )}

            {/* AutoML Defaults Info */}
            <div style={{
              marginTop: '20px',
              padding: '12px',
              backgroundColor: '#f5f5f5',
              borderRadius: '4px',
              fontSize: '12px',
            }}>
              <strong>AutoML Settings:</strong>
              <br />
              Time Limit: {trainingInfo?.automl_defaults.time_limit}s |
              Presets: {trainingInfo?.automl_defaults.presets}
            </div>
          </div>
        )}

        {/* Actions */}
        <div style={{
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '12px',
          marginTop: '24px',
          paddingTop: '16px',
          borderTop: '1px solid #eee',
        }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 16px',
              border: '1px solid #ccc',
              borderRadius: '4px',
              backgroundColor: 'white',
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={loading}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: '4px',
              backgroundColor: '#1976d2',
              color: 'white',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.6 : 1,
            }}
          >
            Run Experiment
          </button>
        </div>
      </div>
    </div>
  );
}

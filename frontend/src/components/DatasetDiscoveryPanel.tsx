/**
 * Dataset Discovery Panel Component
 * Allows users to search for public datasets based on their project description.
 * Used in the project wizard when the user doesn't have their own data yet.
 */

import { useState } from 'react';
import LoadingSpinner from './LoadingSpinner';

interface DatasetDiscoveryPanelProps {
  onSearch: (description: string, constraints?: {
    geography?: string;
    allow_public_data?: boolean;
    licensing_requirements?: string[];
  }) => Promise<void>;
  isSearching: boolean;
  error: string | null;
}

export default function DatasetDiscoveryPanel({
  onSearch,
  isSearching,
  error,
}: DatasetDiscoveryPanelProps) {
  const [description, setDescription] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [geography, setGeography] = useState('');
  const [licensingRequirements, setLicensingRequirements] = useState<string[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!description.trim()) return;

    const constraints = showAdvanced
      ? {
          geography: geography.trim() || undefined,
          allow_public_data: true,
          licensing_requirements: licensingRequirements.length > 0 ? licensingRequirements : undefined,
        }
      : undefined;

    await onSearch(description.trim(), constraints);
  };

  const toggleLicensing = (license: string) => {
    if (licensingRequirements.includes(license)) {
      setLicensingRequirements(licensingRequirements.filter(l => l !== license));
    } else {
      setLicensingRequirements([...licensingRequirements, license]);
    }
  };

  return (
    <div className="dataset-discovery-panel">
      <div className="discovery-header">
        <div className="discovery-icon">🔍</div>
        <div className="discovery-text">
          <h3>Find Public Datasets</h3>
          <p>
            Describe your ML problem and our AI will search for relevant public datasets.
            This is perfect when you want to explore ML but don't have your own data yet.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="discovery-form">
        <div className="form-group">
          <label htmlFor="discovery-description" className="form-label">
            What do you want to predict or analyze?
          </label>
          <textarea
            id="discovery-description"
            className="form-textarea"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="e.g., I want to predict house prices based on location, size, and amenities in the California market"
            rows={4}
            disabled={isSearching}
            required
            minLength={10}
          />
          <p className="form-hint">
            Be specific about your goal, domain, and any preferences.
          </p>
        </div>

        <button
          type="button"
          className="btn btn-link"
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{ marginBottom: '12px', padding: 0 }}
        >
          {showAdvanced ? '▼ Hide advanced options' : '▶ Show advanced options'}
        </button>

        {showAdvanced && (
          <div className="advanced-options">
            <div className="form-group">
              <label htmlFor="discovery-geography" className="form-label">
                Geographic Preference (optional)
              </label>
              <input
                type="text"
                id="discovery-geography"
                className="form-input"
                value={geography}
                onChange={(e) => setGeography(e.target.value)}
                placeholder="e.g., United States, Europe, Global"
                disabled={isSearching}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Licensing Preferences (optional)</label>
              <div className="checkbox-group">
                {['open-source', 'commercial-use-allowed', 'attribution-required'].map((license) => (
                  <label key={license} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={licensingRequirements.includes(license)}
                      onChange={() => toggleLicensing(license)}
                      disabled={isSearching}
                    />
                    {license.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="form-error" style={{ marginBottom: '12px' }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          className="btn btn-primary"
          disabled={isSearching || !description.trim()}
          style={{ width: '100%' }}
        >
          {isSearching ? (
            <>
              <LoadingSpinner size="small" />
              <span style={{ marginLeft: '8px' }}>Searching for datasets...</span>
            </>
          ) : (
            'Search for Datasets'
          )}
        </button>
      </form>

      <style>{`
        .dataset-discovery-panel {
          padding: 20px;
        }

        .discovery-header {
          display: flex;
          gap: 16px;
          align-items: flex-start;
          margin-bottom: 24px;
          padding: 16px;
          background-color: #f0f7ff;
          border-radius: 8px;
        }

        .discovery-icon {
          font-size: 32px;
          flex-shrink: 0;
        }

        .discovery-text h3 {
          margin: 0 0 8px 0;
          color: #1976d2;
        }

        .discovery-text p {
          margin: 0;
          color: #555;
          font-size: 14px;
        }

        .discovery-form {
          display: flex;
          flex-direction: column;
        }

        .advanced-options {
          padding: 16px;
          background-color: #fafafa;
          border-radius: 8px;
          margin-bottom: 16px;
        }

        .checkbox-group {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
        }

        .checkbox-label {
          display: flex;
          align-items: center;
          gap: 6px;
          cursor: pointer;
          font-size: 14px;
        }

        .checkbox-label input {
          margin: 0;
        }
      `}</style>
    </div>
  );
}

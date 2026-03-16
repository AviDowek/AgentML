import { useState, useEffect } from 'react';
import type { ApiKey, LLMProvider, ApiKeyStatus, AIModel, AIModelOption, AppSettings } from '../types/api';
import {
  listApiKeys,
  createOrUpdateApiKey,
  deleteApiKey,
  getApiKeyStatus,
  getAvailableAIModels,
  getAppSettings,
  updateAppSettings,
  ApiException,
} from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

export default function Settings() {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [keyStatus, setKeyStatus] = useState<ApiKeyStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // AI Model state
  const [aiModels, setAiModels] = useState<AIModelOption[]>([]);
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const [selectedAiModel, setSelectedAiModel] = useState<AIModel | null>(null);
  const [savingModel, setSavingModel] = useState(false);

  // Form state
  const [showForm, setShowForm] = useState(false);
  const [formProvider, setFormProvider] = useState<LLMProvider>('openai');
  const [formApiKey, setFormApiKey] = useState('');
  const [formName, setFormName] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [keys, status, models, settings] = await Promise.all([
        listApiKeys(),
        getApiKeyStatus(),
        getAvailableAIModels(),
        getAppSettings(),
      ]);
      setApiKeys(keys);
      setKeyStatus(status);
      setAiModels(models);
      setAppSettings(settings);
      setSelectedAiModel(settings.ai_model);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to load settings');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleAiModelChange = async (newModel: AIModel) => {
    setSavingModel(true);
    setError(null);
    try {
      const updated = await updateAppSettings({ ai_model: newModel });
      setAppSettings(updated);
      setSelectedAiModel(updated.ai_model);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to update AI model');
      }
    } finally {
      setSavingModel(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formApiKey.trim()) return;

    setSaving(true);
    setError(null);

    try {
      await createOrUpdateApiKey({
        provider: formProvider,
        api_key: formApiKey.trim(),
        name: formName.trim() || undefined,
      });
      setShowForm(false);
      setFormApiKey('');
      setFormName('');
      await fetchData();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to save API key');
      }
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (provider: LLMProvider) => {
    if (!confirm(`Are you sure you want to delete the ${provider} API key?`)) {
      return;
    }

    try {
      await deleteApiKey(provider);
      await fetchData();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to delete API key');
      }
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  if (isLoading) {
    return (
      <div className="settings-page">
        <LoadingSpinner message="Loading settings..." />
      </div>
    );
  }

  return (
    <div className="settings-page">
      <div className="page-header">
        <div>
          <h2>Settings</h2>
          <p className="page-subtitle">Configure AI models and API keys</p>
        </div>
      </div>

      {error && (
        <ErrorMessage
          message={error}
          onDismiss={() => setError(null)}
        />
      )}

      {/* AI Model Selection */}
      <div className="settings-section">
        <div className="section-header">
          <h3>AI Model</h3>
          <p className="section-description">
            Choose the AI model for agent analysis and recommendations. GPT-5.1 Thinking provides deeper reasoning for complex tasks.
          </p>
        </div>

        <div className="ai-model-selector">
          {aiModels.map((model) => (
            <label
              key={model.value}
              className={`ai-model-option ${selectedAiModel === model.value ? 'selected' : ''} ${savingModel ? 'disabled' : ''}`}
            >
              <input
                type="radio"
                name="ai-model"
                value={model.value}
                checked={selectedAiModel === model.value}
                onChange={() => handleAiModelChange(model.value)}
                disabled={savingModel}
              />
              <div className="model-info">
                <span className="model-name">{model.display_name}</span>
                <span className="model-description">{model.description}</span>
              </div>
              {selectedAiModel === model.value && (
                <span className="model-check">&#10003;</span>
              )}
            </label>
          ))}
        </div>
        {savingModel && <span className="saving-indicator">Saving...</span>}
      </div>

      <div className="settings-section">
        <div className="section-header">
          <h3>LLM API Keys</h3>
          <p className="section-description">
            Configure API keys for OpenAI and Google Gemini for AI-powered features.
          </p>
        </div>

        <div className="api-keys-status">
          <div className={`status-card ${keyStatus?.openai ? 'configured' : 'not-configured'}`}>
            <span className="provider-name">OpenAI</span>
            <span className="status-indicator">
              {keyStatus?.openai ? 'Configured' : 'Not configured'}
            </span>
          </div>
          <div className={`status-card ${keyStatus?.gemini ? 'configured' : 'not-configured'}`}>
            <span className="provider-name">Google Gemini</span>
            <span className="status-indicator">
              {keyStatus?.gemini ? 'Configured' : 'Not configured'}
            </span>
          </div>
        </div>

        {apiKeys.length > 0 && (
          <div className="api-keys-list">
            <h4>Saved Keys</h4>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>Name</th>
                  <th>Key Preview</th>
                  <th>Added</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {apiKeys.map((key) => (
                  <tr key={key.id}>
                    <td className="provider-cell">
                      {key.provider === 'openai' ? 'OpenAI' : 'Google Gemini'}
                    </td>
                    <td>{key.name || '-'}</td>
                    <td className="key-preview">{key.key_preview}</td>
                    <td>{formatDate(key.created_at)}</td>
                    <td>
                      <button
                        className="btn btn-small btn-secondary"
                        onClick={() => {
                          setFormProvider(key.provider);
                          setFormName(key.name || '');
                          setFormApiKey('');
                          setShowForm(true);
                        }}
                      >
                        Update
                      </button>
                      <button
                        className="btn btn-small btn-danger"
                        onClick={() => handleDelete(key.provider)}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!showForm && (
          <button
            className="btn btn-primary"
            onClick={() => setShowForm(true)}
          >
            Add API Key
          </button>
        )}

        {showForm && (
          <div className="api-key-form-container">
            <h4>{apiKeys.find(k => k.provider === formProvider) ? 'Update' : 'Add'} API Key</h4>
            <form onSubmit={handleSubmit} className="api-key-form">
              <div className="form-group">
                <label htmlFor="provider">Provider</label>
                <select
                  id="provider"
                  value={formProvider}
                  onChange={(e) => setFormProvider(e.target.value as LLMProvider)}
                  className="form-select"
                >
                  <option value="openai">OpenAI</option>
                  <option value="gemini">Google Gemini</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="apiKey">API Key</label>
                <input
                  type="password"
                  id="apiKey"
                  value={formApiKey}
                  onChange={(e) => setFormApiKey(e.target.value)}
                  placeholder="Enter your API key"
                  className="form-input"
                  required
                />
                <p className="form-hint">
                  {formProvider === 'openai'
                    ? 'Get your API key from https://platform.openai.com/api-keys'
                    : 'Get your API key from https://makersuite.google.com/app/apikey'}
                </p>
              </div>

              <div className="form-group">
                <label htmlFor="name">Name (optional)</label>
                <input
                  type="text"
                  id="name"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  placeholder="e.g., Production Key"
                  className="form-input"
                />
              </div>

              <div className="form-actions">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    setShowForm(false);
                    setFormApiKey('');
                    setFormName('');
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={saving || !formApiKey.trim()}
                >
                  {saving ? 'Saving...' : 'Save API Key'}
                </button>
              </div>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}

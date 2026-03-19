import { useState, useEffect } from 'react';
import { Outlet, Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import type { AIModel, AIModelOption, AppSettings } from '../types/api';
import { getAvailableAIModels, getAppSettings, updateAppSettings } from '../services/api';

export default function MainLayout() {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [aiModels, setAiModels] = useState<AIModelOption[]>([]);
  const [currentModel, setCurrentModel] = useState<AIModel | null>(null);
  const [savingModel, setSavingModel] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const [models, settings] = await Promise.all([
        getAvailableAIModels(),
        getAppSettings(),
      ]);
      setAiModels(models);
      setCurrentModel(settings.ai_model);
    } catch {
      // Silently fail - settings will use defaults
    }
  };

  const handleModelChange = async (newModel: AIModel) => {
    setSavingModel(true);
    try {
      const updated = await updateAppSettings({ ai_model: newModel });
      setCurrentModel(updated.ai_model);
      setShowModelMenu(false);
    } catch {
      // Silently fail
    } finally {
      setSavingModel(false);
    }
  };

  const getCurrentModelDisplay = () => {
    const model = aiModels.find(m => m.value === currentModel);
    return model?.display_name || 'GPT-5.1 Thinking';
  };

  const handleLogout = () => {
    logout();
    setShowUserMenu(false);
    navigate('/');
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <nav>
          <Link to="/" className="logo-link">
            <h1>Agentic ML Platform</h1>
          </Link>
          <ul className="nav-links">
            <li><Link to="/">Home</Link></li>
            {isAuthenticated && (
              <>
                <li><Link to="/projects">Projects</Link></li>
                <li><Link to="/experiments">Experiments</Link></li>
                <li><Link to="/models">Models</Link></li>
                <li><Link to="/auto-ds">Auto DS</Link></li>
                <li><Link to="/settings">Settings</Link></li>
                <li><Link to="/guide">Guide</Link></li>
                {user?.is_admin && <li><Link to="/admin" style={{ color: '#f59e0b' }}>Admin</Link></li>}
              </>
            )}
          </ul>
          <div className="model-selector-container">
            <button
              className={`model-selector-trigger ${savingModel ? 'saving' : ''}`}
              onClick={() => setShowModelMenu(!showModelMenu)}
              disabled={savingModel}
              aria-expanded={showModelMenu}
            >
              <span className="model-icon">&#x2728;</span>
              <span className="model-current">{getCurrentModelDisplay()}</span>
              <svg
                className={`dropdown-arrow ${showModelMenu ? 'open' : ''}`}
                width="10"
                height="10"
                viewBox="0 0 12 12"
              >
                <path d="M3 5l3 3 3-3" stroke="currentColor" strokeWidth="1.5" fill="none" />
              </svg>
            </button>
            {showModelMenu && (
              <div className="model-selector-dropdown">
                <div className="model-dropdown-header">AI Model</div>
                {aiModels.map((model) => (
                  <button
                    key={model.value}
                    className={`model-dropdown-item ${currentModel === model.value ? 'selected' : ''}`}
                    onClick={() => handleModelChange(model.value)}
                    disabled={savingModel}
                  >
                    <span className="model-item-name">{model.display_name}</span>
                    {currentModel === model.value && (
                      <span className="model-item-check">&#10003;</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="nav-auth">
            {isAuthenticated ? (
              <div className="user-menu-container">
                <button
                  className="user-menu-trigger"
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  aria-expanded={showUserMenu}
                >
                  <span className="user-avatar">
                    {user?.full_name?.[0] || user?.email?.[0] || '?'}
                  </span>
                  <span className="user-name">{user?.full_name || user?.email}</span>
                  <svg
                    className={`dropdown-arrow ${showUserMenu ? 'open' : ''}`}
                    width="12"
                    height="12"
                    viewBox="0 0 12 12"
                  >
                    <path d="M3 5l3 3 3-3" stroke="currentColor" strokeWidth="1.5" fill="none" />
                  </svg>
                </button>
                {showUserMenu && (
                  <div className="user-menu-dropdown">
                    <div className="user-menu-header">
                      <div className="user-email">{user?.email}</div>
                    </div>
                    <Link
                      to="/settings"
                      className="user-menu-item"
                      onClick={() => setShowUserMenu(false)}
                    >
                      Settings
                    </Link>
                    <button
                      className="user-menu-item user-menu-logout"
                      onClick={handleLogout}
                    >
                      Sign Out
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="auth-links">
                <Link to="/login" className="btn btn-text">Sign In</Link>
                <Link to="/signup" className="btn btn-primary btn-sm">Sign Up</Link>
              </div>
            )}
          </div>
        </nav>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
      <footer className="app-footer">
        <p>Agentic ML Platform v0.1.0</p>
      </footer>
    </div>
  );
}

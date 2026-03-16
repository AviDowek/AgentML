import { useState, useEffect } from 'react';

interface HealthStatus {
  status: string;
}

interface ApiInfo {
  name: string;
  version: string;
  docs: string;
}

export default function Home() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [apiInfo, setApiInfo] = useState<ApiInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const [healthRes, infoRes] = await Promise.all([
          fetch('http://localhost:8001/health'),
          fetch('http://localhost:8001/')
        ]);

        if (healthRes.ok) {
          setHealth(await healthRes.json());
        }
        if (infoRes.ok) {
          setApiInfo(await infoRes.json());
        }
      } catch {
        setError('Backend not available. Make sure the API is running on port 8001.');
      }
    };

    checkBackend();
  }, []);

  return (
    <div className="home-page">
      <section className="hero">
        <h2>Welcome to Agentic ML Platform</h2>
        <p>
          An intelligent ML engineering platform that helps you build, train,
          and deploy machine learning models for tabular data problems.
        </p>
      </section>

      <section className="status-section">
        <h3>System Status</h3>
        {error ? (
          <div className="status-card error">
            <span className="status-indicator">⚠️</span>
            <span>{error}</span>
          </div>
        ) : health ? (
          <div className="status-card success">
            <span className="status-indicator">✓</span>
            <span>Backend: {health.status}</span>
            {apiInfo && <span> | Version: {apiInfo.version}</span>}
          </div>
        ) : (
          <div className="status-card loading">
            <span>Checking backend status...</span>
          </div>
        )}
      </section>

      <section className="features">
        <h3>Key Features</h3>
        <div className="feature-grid">
          <div className="feature-card">
            <h4>📝 Natural Language Tasks</h4>
            <p>Describe your ML problem in plain English</p>
          </div>
          <div className="feature-card">
            <h4>🔗 Data Connections</h4>
            <p>Connect databases or upload files</p>
          </div>
          <div className="feature-card">
            <h4>🧪 Automated Experiments</h4>
            <p>Run multiple experiment variants automatically</p>
          </div>
          <div className="feature-card">
            <h4>🏆 Model Leaderboard</h4>
            <p>Compare and select the best models</p>
          </div>
        </div>
      </section>
    </div>
  );
}

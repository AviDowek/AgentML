import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function Home() {
  const { isAuthenticated, user } = useAuth();

  if (isAuthenticated) {
    return (
      <div className="home-page">
        <section className="hero">
          <h2>Welcome back, {user?.full_name || user?.email}!</h2>
          <p>
            Build, train, and deploy machine learning models for tabular data.
          </p>
        </section>

        <section className="features">
          <h3>Quick Start</h3>
          <div className="feature-grid">
            <Link to="/projects" className="feature-card" style={{ textDecoration: 'none', color: 'inherit' }}>
              <h4>Projects</h4>
              <p>Manage your ML projects and data</p>
            </Link>
            <Link to="/experiments" className="feature-card" style={{ textDecoration: 'none', color: 'inherit' }}>
              <h4>Experiments</h4>
              <p>View and run experiments</p>
            </Link>
            <Link to="/models" className="feature-card" style={{ textDecoration: 'none', color: 'inherit' }}>
              <h4>Models</h4>
              <p>Compare and deploy models</p>
            </Link>
            <Link to="/auto-ds" className="feature-card" style={{ textDecoration: 'none', color: 'inherit' }}>
              <h4>Auto DS</h4>
              <p>Automated data science sessions</p>
            </Link>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="home-page">
      <section className="hero">
        <h2>AgentML</h2>
        <p>
          An intelligent ML engineering platform that helps you build, train,
          and deploy machine learning models for tabular data. Describe your
          prediction task in natural language and let AI agents do the rest.
        </p>
        <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <Link to="/signup" className="btn btn-primary">Get Started</Link>
          <Link to="/login" className="btn btn-text">Sign In</Link>
        </div>
      </section>

      <section className="features">
        <h3>Key Features</h3>
        <div className="feature-grid">
          <div className="feature-card">
            <h4>Natural Language Tasks</h4>
            <p>Describe your ML problem in plain English</p>
          </div>
          <div className="feature-card">
            <h4>Multi-Table Data</h4>
            <p>Connect databases or upload files, auto-join multiple tables</p>
          </div>
          <div className="feature-card">
            <h4>Automated Experiments</h4>
            <p>AI agents design, run, and iterate on experiments</p>
          </div>
          <div className="feature-card">
            <h4>Model Lifecycle</h4>
            <p>Promote models from draft to production with guardrails</p>
          </div>
        </div>
      </section>
    </div>
  );
}

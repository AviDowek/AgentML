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

        <section style={{ marginTop: '2rem', textAlign: 'center' }}>
          <Link to="/guide" className="btn btn-primary">View Platform Guide</Link>
        </section>
      </div>
    );
  }

  return (
    <div className="landing-page">
      {/* Hero */}
      <section className="landing-hero">
        <div className="landing-hero-badge">AI-Powered ML Platform</div>
        <h1>Build ML Models with<br /><span className="landing-accent">Natural Language</span></h1>
        <p className="landing-hero-sub">
          Describe your prediction task in plain English. Our AI agents design experiments,
          engineer features, train models, and detect problems — so you can go from raw data
          to production-ready predictions without writing a single line of code.
        </p>
        <div className="landing-hero-cta">
          <Link to="/signup" className="btn btn-primary btn-lg">Get Started Free</Link>
          <Link to="/login" className="btn btn-text btn-lg">Sign In</Link>
        </div>
      </section>

      {/* How it works */}
      <section className="landing-section">
        <h2 className="landing-section-title">How It Works</h2>
        <p className="landing-section-sub">Three steps from raw data to deployed model</p>
        <div className="landing-steps">
          <div className="landing-step">
            <div className="landing-step-num">1</div>
            <h3>Upload Your Data</h3>
            <p>Upload CSV files or connect a database. The platform auto-discovers table relationships and profiles every column.</p>
          </div>
          <div className="landing-step-arrow">&#8594;</div>
          <div className="landing-step">
            <div className="landing-step-num">2</div>
            <h3>Describe Your Goal</h3>
            <p>Tell the AI what you want to predict in plain English. It designs the experiment, selects features, and chooses the best approach.</p>
          </div>
          <div className="landing-step-arrow">&#8594;</div>
          <div className="landing-step">
            <div className="landing-step-num">3</div>
            <h3>Train &amp; Deploy</h3>
            <p>Models train in the cloud. Review results, compare metrics, and promote the best model to production with one click.</p>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="landing-section landing-section-alt">
        <h2 className="landing-section-title">What Makes AgentML Different</h2>
        <div className="landing-features-grid">
          <div className="landing-feature">
            <div className="landing-feature-icon">&#129302;</div>
            <h3>21 Specialized AI Agents</h3>
            <p>A team of AI agents handles every step: data auditing, feature engineering, experiment design, training, evaluation, and improvement. Each agent is an expert at its job.</p>
          </div>
          <div className="landing-feature">
            <div className="landing-feature-icon">&#128274;</div>
            <h3>Data Leakage Detection</h3>
            <p>Automatically scans for data leakage — the #1 cause of models that look great in testing but fail in production. Catches problems before they cost you.</p>
          </div>
          <div className="landing-feature">
            <div className="landing-feature-icon">&#128202;</div>
            <h3>Three-Tier Validation</h3>
            <p>Every model is validated on train, validation, and a held-out test set that's never touched during development. No overfitting surprises.</p>
          </div>
          <div className="landing-feature">
            <div className="landing-feature-icon">&#9997;&#65039;</div>
            <h3>Natural Language Interface</h3>
            <p>No code required. Describe what you want to predict, provide context about your data, and the platform handles the rest.</p>
          </div>
          <div className="landing-feature">
            <div className="landing-feature-icon">&#128300;</div>
            <h3>Automated Improvement</h3>
            <p>Not happy with results? The improvement pipeline automatically tries new feature combinations, model types, and hyperparameter strategies.</p>
          </div>
          <div className="landing-feature">
            <div className="landing-feature-icon">&#9729;&#65039;</div>
            <h3>Cloud Training</h3>
            <p>Models train on Modal.com cloud infrastructure. No GPU setup, no environment hassles. Results appear in your dashboard when ready.</p>
          </div>
        </div>
      </section>

      {/* Use cases */}
      <section className="landing-section">
        <h2 className="landing-section-title">Built for Tabular Data</h2>
        <p className="landing-section-sub">The kind of data most businesses actually have</p>
        <div className="landing-usecases">
          <div className="landing-usecase">
            <h4>Customer Churn</h4>
            <p>"Predict which customers will cancel their subscription next month"</p>
          </div>
          <div className="landing-usecase">
            <h4>Sales Forecasting</h4>
            <p>"Forecast next quarter's revenue by product category"</p>
          </div>
          <div className="landing-usecase">
            <h4>Fraud Detection</h4>
            <p>"Flag transactions that are likely fraudulent"</p>
          </div>
          <div className="landing-usecase">
            <h4>Lead Scoring</h4>
            <p>"Score leads by likelihood to convert"</p>
          </div>
          <div className="landing-usecase">
            <h4>Price Optimization</h4>
            <p>"Predict optimal pricing based on market conditions"</p>
          </div>
          <div className="landing-usecase">
            <h4>Risk Assessment</h4>
            <p>"Classify loan applications by default risk"</p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="landing-section landing-cta-section">
        <h2>Ready to build your first model?</h2>
        <p>Sign up and have a trained model in minutes, not weeks.</p>
        <div className="landing-hero-cta">
          <Link to="/signup" className="btn btn-primary btn-lg">Get Started Free</Link>
        </div>
      </section>
    </div>
  );
}

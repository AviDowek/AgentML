import { Link } from 'react-router-dom';
import { useState } from 'react';

type Section = 'overview' | 'projects' | 'experiments' | 'models' | 'autods' | 'settings' | 'tips';

export default function Guide() {
  const [activeSection, setActiveSection] = useState<Section>('overview');

  const sections: { id: Section; label: string }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'projects', label: 'Projects & Data' },
    { id: 'experiments', label: 'Experiments' },
    { id: 'models', label: 'Models' },
    { id: 'autods', label: 'Auto DS' },
    { id: 'settings', label: 'Settings' },
    { id: 'tips', label: 'Tips & Best Practices' },
  ];

  return (
    <div className="guide-page">
      <div className="guide-layout">
        {/* Sidebar */}
        <nav className="guide-sidebar">
          <h3>Platform Guide</h3>
          {sections.map((s) => (
            <button
              key={s.id}
              className={`guide-nav-item ${activeSection === s.id ? 'active' : ''}`}
              onClick={() => setActiveSection(s.id)}
            >
              {s.label}
            </button>
          ))}
          <div className="guide-sidebar-divider" />
          <Link to="/" className="guide-nav-item guide-back">&#8592; Back to Dashboard</Link>
        </nav>

        {/* Content */}
        <main className="guide-content">
          {activeSection === 'overview' && (
            <div>
              <h2>Welcome to AgentML</h2>
              <p className="guide-intro">
                AgentML is an AI-powered platform for building machine learning models from tabular data.
                You describe what you want to predict, upload your data, and a team of 21 specialized AI agents
                handles everything from feature engineering to model training and evaluation.
              </p>

              <div className="guide-workflow">
                <h3>Typical Workflow</h3>
                <div className="guide-workflow-steps">
                  <div className="guide-wf-step">
                    <div className="guide-wf-num">1</div>
                    <div>
                      <strong>Create a Project</strong>
                      <p>A project is your workspace. It holds your data, experiments, and models for one prediction task.</p>
                    </div>
                  </div>
                  <div className="guide-wf-step">
                    <div className="guide-wf-num">2</div>
                    <div>
                      <strong>Upload Data</strong>
                      <p>Upload one or more CSV files. The platform profiles your data and discovers relationships between tables.</p>
                    </div>
                  </div>
                  <div className="guide-wf-step">
                    <div className="guide-wf-num">3</div>
                    <div>
                      <strong>Describe Your Target</strong>
                      <p>Tell the AI what you want to predict in plain English (e.g., "predict whether a customer will churn").</p>
                    </div>
                  </div>
                  <div className="guide-wf-step">
                    <div className="guide-wf-num">4</div>
                    <div>
                      <strong>Run the Agent Pipeline</strong>
                      <p>AI agents analyze your data, check for leakage, design features, and create an experiment plan.</p>
                    </div>
                  </div>
                  <div className="guide-wf-step">
                    <div className="guide-wf-num">5</div>
                    <div>
                      <strong>Train Models</strong>
                      <p>Run the experiment. Models train on cloud infrastructure (Modal.com). This typically takes 5-30 minutes.</p>
                    </div>
                  </div>
                  <div className="guide-wf-step">
                    <div className="guide-wf-num">6</div>
                    <div>
                      <strong>Review &amp; Improve</strong>
                      <p>Check metrics, review feature importance, and optionally run improvement cycles to boost performance.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'projects' && (
            <div>
              <h2>Projects &amp; Data</h2>

              <div className="guide-block">
                <h3>Creating a Project</h3>
                <p>Go to <Link to="/projects">Projects</Link> and click <strong>"New Project"</strong>. A project wizard will walk you through setup:</p>
                <ol className="guide-steps">
                  <li><strong>Name your project</strong> — give it a descriptive name like "Customer Churn Prediction"</li>
                  <li><strong>Upload data</strong> — drag and drop CSV files. You can upload multiple tables.</li>
                  <li><strong>Data discovery</strong> — the platform auto-profiles every column (types, distributions, missing values) and discovers relationships between tables</li>
                  <li><strong>Define your target</strong> — describe what you want to predict in natural language</li>
                  <li><strong>Agent pipeline</strong> — AI agents run to design your experiment, check for data quality issues, and plan feature engineering</li>
                </ol>
              </div>

              <div className="guide-block">
                <h3>Data Sources</h3>
                <p>Each project can have multiple data sources (CSV files). The platform will:</p>
                <ul className="guide-list">
                  <li>Auto-detect column types (numeric, categorical, datetime, text)</li>
                  <li>Identify potential join keys between tables</li>
                  <li>Flag columns with high missing rates</li>
                  <li>Detect potential data leakage risks</li>
                </ul>
              </div>

              <div className="guide-block">
                <h3>Context Documents</h3>
                <p>You can upload additional documents (PDF, DOCX, TXT) that give the AI agents more context about your data.
                  For example, a data dictionary, business rules, or domain knowledge. This helps agents make better decisions.</p>
              </div>

              <div className="guide-block">
                <h3>Sharing</h3>
                <p>Projects are private by default. You can share a project with other users by email from the project settings.
                  Shared users can view experiments and results but cannot delete the project.</p>
              </div>
            </div>
          )}

          {activeSection === 'experiments' && (
            <div>
              <h2>Experiments</h2>

              <div className="guide-block">
                <h3>What is an Experiment?</h3>
                <p>An experiment is a single training run. It takes your data, applies feature engineering,
                  trains multiple model types (XGBoost, LightGBM, CatBoost, neural networks, etc.), and evaluates
                  them using three-tier validation.</p>
              </div>

              <div className="guide-block">
                <h3>The Agent Pipeline</h3>
                <p>When you create an experiment, the AI agents work through a structured pipeline:</p>
                <div className="guide-pipeline">
                  <div className="guide-pipeline-stage">
                    <div className="guide-pipeline-label">Setup Pipeline</div>
                    <div className="guide-pipeline-agents">
                      <span>Data Analyst</span>
                      <span>Planner</span>
                      <span>Data Auditor</span>
                      <span>Feature Engineer</span>
                      <span>Plan Critic</span>
                      <span>Overfitting Guard</span>
                    </div>
                    <p>Analyzes your data, checks for leakage, designs features, and creates a training plan. The Plan Critic reviews the plan and may request revisions.</p>
                  </div>
                  <div className="guide-pipeline-stage">
                    <div className="guide-pipeline-label">Training</div>
                    <p>Models train on Modal.com cloud infrastructure using AutoGluon. Multiple model types are trained and ensembled automatically.</p>
                  </div>
                  <div className="guide-pipeline-stage">
                    <div className="guide-pipeline-label">Results Pipeline</div>
                    <div className="guide-pipeline-agents">
                      <span>Results Analyst</span>
                      <span>Quality Assessor</span>
                    </div>
                    <p>Evaluates model performance, checks for overfitting, generates feature importance analysis, and provides an overall quality assessment with a risk score.</p>
                  </div>
                </div>
              </div>

              <div className="guide-block">
                <h3>Running an Experiment</h3>
                <ol className="guide-steps">
                  <li>Go to your project and click <strong>"Run Experiment"</strong></li>
                  <li>Training runs on Modal.com cloud (you need an AI API key configured in Settings)</li>
                  <li>Watch live training logs with AI-interpreted explanations</li>
                  <li>When complete, review metrics, feature importance, and the risk score</li>
                </ol>
              </div>

              <div className="guide-block">
                <h3>Understanding Results</h3>
                <ul className="guide-list">
                  <li><strong>Metrics</strong> — accuracy, F1, AUC, RMSE, etc. depending on your task type</li>
                  <li><strong>Three-tier scores</strong> — train, validation, and holdout. Compare them to check for overfitting.</li>
                  <li><strong>Risk Score</strong> — 0-100, lower is better. Quantifies risks from overfitting, leakage, and data quality issues.</li>
                  <li><strong>Feature Importance</strong> — which features matter most for predictions</li>
                </ul>
              </div>

              <div className="guide-block">
                <h3>Improvement Pipeline</h3>
                <p>After reviewing results, you can run the <strong>Improvement Pipeline</strong>. It uses 6 specialized agents
                  that analyze what went wrong, try different feature combinations, and propose new experiment strategies.
                  You can run multiple improvement cycles.</p>
              </div>
            </div>
          )}

          {activeSection === 'models' && (
            <div>
              <h2>Models</h2>

              <div className="guide-block">
                <h3>Model Lifecycle</h3>
                <p>Models go through a lifecycle with these stages:</p>
                <div className="guide-lifecycle">
                  <div className="guide-lc-stage">
                    <span className="guide-lc-badge draft">Draft</span>
                    <p>Newly trained models start as drafts. Review their metrics and decide if they're ready.</p>
                  </div>
                  <div className="guide-lc-arrow">&#8594;</div>
                  <div className="guide-lc-stage">
                    <span className="guide-lc-badge staging">Staging</span>
                    <p>Promote to staging for further validation. Run on additional test data if needed.</p>
                  </div>
                  <div className="guide-lc-arrow">&#8594;</div>
                  <div className="guide-lc-stage">
                    <span className="guide-lc-badge production">Production</span>
                    <p>Ready for real-world use. Only one model per project can be in production at a time.</p>
                  </div>
                </div>
              </div>

              <div className="guide-block">
                <h3>Comparing Models</h3>
                <p>The <Link to="/models">Models</Link> page shows all your models across projects. You can:</p>
                <ul className="guide-list">
                  <li>Sort and filter by metric scores</li>
                  <li>Compare models from different experiments side by side</li>
                  <li>View detailed training configuration and hyperparameters</li>
                  <li>Check the holdout score (never seen during training) for a true performance estimate</li>
                </ul>
              </div>
            </div>
          )}

          {activeSection === 'autods' && (
            <div>
              <h2>Auto DS (Automated Data Science)</h2>

              <div className="guide-block">
                <h3>What is Auto DS?</h3>
                <p>Auto DS is a conversational interface for automated data science. Instead of manually configuring
                  experiments, you describe what you want in natural language, and the system handles everything
                  end-to-end — from data exploration to model training.</p>
              </div>

              <div className="guide-block">
                <h3>Starting a Session</h3>
                <ol className="guide-steps">
                  <li>Go to <Link to="/auto-ds">Auto DS</Link> and click <strong>"New Session"</strong></li>
                  <li>Select a project (must have data uploaded already)</li>
                  <li>Choose a validation strategy (standard, robust, or strict)</li>
                  <li>Describe your goal and any specific requirements</li>
                  <li>The system runs through the full pipeline automatically</li>
                </ol>
              </div>

              <div className="guide-block">
                <h3>When to Use Auto DS vs Manual Experiments</h3>
                <ul className="guide-list">
                  <li><strong>Use Auto DS</strong> when you want a quick, hands-off approach — great for initial exploration</li>
                  <li><strong>Use manual experiments</strong> when you want fine control over feature engineering, validation strategy, or model selection</li>
                </ul>
              </div>
            </div>
          )}

          {activeSection === 'settings' && (
            <div>
              <h2>Settings</h2>

              <div className="guide-block">
                <h3>AI Model Selection</h3>
                <p>The model selector in the top navigation bar lets you choose which AI model powers the agents.
                  Different models have different speed/quality tradeoffs. You can change this at any time — it
                  affects new experiments and agent runs.</p>
              </div>

              <div className="guide-block">
                <h3>API Keys</h3>
                <p>Go to <Link to="/settings">Settings</Link> to configure your AI provider API keys. You need at least
                  one configured to run experiments. The platform supports:</p>
                <ul className="guide-list">
                  <li><strong>OpenAI</strong> — GPT models for agent reasoning</li>
                  <li><strong>Google Gemini</strong> — alternative AI provider</li>
                  <li><strong>Anthropic (Claude)</strong> — alternative AI provider</li>
                </ul>
                <p>Keys are encrypted at rest and never shared. You can update or remove them at any time.</p>
              </div>

              <div className="guide-block">
                <h3>Account</h3>
                <p>Update your name and manage your account from the user menu in the top-right corner.</p>
              </div>
            </div>
          )}

          {activeSection === 'tips' && (
            <div>
              <h2>Tips &amp; Best Practices</h2>

              <div className="guide-block">
                <h3>Data Quality</h3>
                <ul className="guide-list">
                  <li><strong>Clean your data first</strong> — remove completely empty rows and clearly invalid records before uploading</li>
                  <li><strong>Include enough rows</strong> — the platform works best with at least a few hundred rows. More data generally means better models.</li>
                  <li><strong>Remove obvious leakage</strong> — if a column is derived from the target (like "cancellation_date" when predicting churn), remove it before uploading. The platform checks for this too, but prevention is best.</li>
                  <li><strong>Use descriptive column names</strong> — the AI agents read column names to understand your data. "monthly_revenue" is much better than "col_7".</li>
                </ul>
              </div>

              <div className="guide-block">
                <h3>Getting Better Results</h3>
                <ul className="guide-list">
                  <li><strong>Write detailed descriptions</strong> — the more context you give about what you're predicting and why, the better the agents will perform</li>
                  <li><strong>Upload context documents</strong> — data dictionaries, business rules, and domain knowledge help the agents make informed decisions</li>
                  <li><strong>Run improvement cycles</strong> — the first experiment is rarely the best. Use the improvement pipeline to iterate.</li>
                  <li><strong>Check the risk score</strong> — a high risk score means the model might not generalize well. Investigate before deploying.</li>
                  <li><strong>Compare holdout vs validation</strong> — if there's a big gap, the model may be overfitting</li>
                </ul>
              </div>

              <div className="guide-block">
                <h3>Common Issues</h3>
                <ul className="guide-list">
                  <li><strong>"No API key configured"</strong> — go to Settings and add at least one AI provider key</li>
                  <li><strong>"Modal not configured"</strong> — training requires Modal.com. Contact your admin if this isn't set up.</li>
                  <li><strong>Experiment stuck on "running"</strong> — training typically takes 5-30 minutes. If it's been over an hour, check the training logs for errors.</li>
                  <li><strong>Low model performance</strong> — this can be caused by too few rows, noisy data, or a prediction target that's genuinely hard to predict. Try running an improvement cycle.</li>
                </ul>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

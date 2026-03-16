# AgentML Capabilities Guide

Welcome to AgentML! This guide explains everything the platform can do, written so anyone can understand it - even if you're not a machine learning expert.

---

## Table of Contents

1. [What is AgentML?](#what-is-agentml)
2. [Getting Started](#getting-started)
3. [Core Features](#core-features)
   - [Projects](#projects)
   - [Data Management](#data-management)
     - [Build Training Dataset with AI](#build-training-dataset-with-ai)
   - [AI-Powered Setup](#ai-powered-setup)
   - [Experiments & Training](#experiments--training)
   - [Models](#models)
   - [Visualizations](#visualizations)
4. [User Flows](#user-flows)
5. [Advanced Features](#advanced-features)
   - [Project Settings](#project-settings)
6. [Sharing & Collaboration](#sharing--collaboration)
7. [Technical Details](#technical-details)

---

## What is AgentML?

AgentML is an **intelligent machine learning platform** that helps you build predictive models without needing to be a data scientist.

Think of it like having an AI assistant that:
- **Understands your data** - Upload your files and it automatically analyzes what's in them
- **Understands your goals** - Tell it what you want to predict in plain English
- **Does the technical work** - Automatically selects the right algorithms and settings
- **Explains the results** - Tells you what your model learned and why

### Who is it for?

- **Business analysts** who have data and want predictions
- **Data scientists** who want to speed up their workflow
- **Developers** who need to add ML to their applications
- **Anyone** curious about what machine learning can reveal in their data

---

## Getting Started

### Creating an Account

1. **Sign Up** - Create an account with your email and password
2. **Or use Google** - Click "Sign in with Google" for quick access
3. **Add API Keys** (Optional) - Go to Settings to add your OpenAI or Gemini API key for AI features

### Your First Project

1. Click **"New Project"**
2. Give it a name like "Customer Churn Prediction"
3. Add a description of what you want to predict
4. Choose to use the **AI Wizard** for guided setup, or configure manually

---

## Core Features

### Projects

A **Project** is your workspace for a specific prediction task. Each project contains:

- **Data Sources** - The files you upload
- **Dataset Specs** - Which columns to use for training
- **Experiments** - Different training configurations to try
- **Models** - The trained prediction machines
- **Visualizations** - Charts and graphs of your data

#### Project Types

AgentML supports many types of prediction problems:

| Type | What it Does | Example |
|------|--------------|---------|
| **Binary Classification** | Predicts yes/no, true/false | Will this customer cancel? |
| **Multiclass Classification** | Predicts one of several categories | What product category is this? |
| **Regression** | Predicts a number | What price should we set? |
| **Time Series Forecast** | Predicts future values over time | What will sales be next month? |
| **Quantile Regression** | Predicts ranges (10th, 50th, 90th percentile) | What's the likely range of demand? |

---

### Data Management

#### Uploading Data

AgentML accepts many file formats:

| Format | Extensions | Best For |
|--------|------------|----------|
| **CSV** | .csv | Most common data format |
| **Excel** | .xlsx, .xls | Spreadsheets with multiple sheets |
| **JSON** | .json | Web/API data |
| **Parquet** | .parquet | Large datasets |
| **Text** | .txt | Simple data files |
| **Word** | .docx | Documents with tables |

**To upload:**
1. Go to your project's **Data** tab
2. Click **"Upload File"**
3. Drag and drop your file, or click to browse
4. Optionally give it a custom name
5. The system automatically analyzes your data

#### What Happens After Upload?

When you upload a file, AgentML:

1. **Reads the structure** - Identifies all columns and their types
2. **Calculates statistics** - Counts rows, finds missing values, identifies unique values
3. **Detects data types** - Numbers, categories, dates, text
4. **Checks for issues** - Warns if data looks like metadata or has problems
5. **Shows you a preview** - View all your data in a scrollable table

#### Viewing Your Data

Click on any data source to see:
- **Schema tab** - Column names, types, statistics
- **Data tab** - Full table view with all rows (paginated for large files)
  - Scroll horizontally to see all columns
  - Navigate pages to see all rows
  - See row numbers for reference

#### AI Data Search

Don't have the right data? Use the **"Search with AI"** button to:
1. Describe what data you're looking for
2. AI searches for relevant public datasets
3. Preview what's available
4. Click to download and add to your project

#### Build Training Dataset with AI

When you have multiple related data sources (e.g., Customers and Orders), AgentML can automatically build an optimized training dataset:

1. Click **"Build Training Dataset"** in your project
2. Enter a **target hint** describing what you want to predict (e.g., "predict customer churn")
3. Click **"Start Building"**

The AI Data Architect pipeline runs through 5 intelligent steps:

| Step | Agent Role | What it Does |
|------|------------|--------------|
| **Dataset Inventory** | Data Profiler | Profiles all your data sources and catalogs their structure |
| **Relationship Discovery** | Data Analyst | Discovers join keys and relationships between tables |
| **Feature Blueprint** | Feature Engineer | Designs aggregations and derived features |
| **Join Plan** | Data Architect | Plans how to combine tables optimally |
| **Materialize** | Data Engineer | Builds the final training dataset |

**What you get:**
- A new data source with all tables joined intelligently
- Automatic feature engineering (aggregations, counts, time-based features)
- Clear summary showing base table, joined tables, target column, and feature columns
- Handling of large datasets through automatic sampling

**Example:**
> You have: Customers (100 rows), Orders (5,000 rows), Products (50 rows)
>
> Target hint: "predict customer lifetime value"
>
> Result: Training dataset with customer-level features like total_orders, avg_order_value, days_since_last_order, favorite_category, etc.

---

### AI-Powered Setup

This is where AgentML's intelligence shines. Instead of manually configuring everything, you can let AI guide you.

#### The AI Pipeline

When you start the AI setup wizard, it runs through 6 intelligent steps:

```
Step 1: Data Analysis
   ↓
Step 2: Problem Understanding
   ↓
Step 3: Data Audit
   ↓
Step 4: Dataset Design
   ↓
Step 5: Experiment Design
   ↓
Step 6: Plan Critic (Validation)
```

##### Step 1: Data Analysis

**What it does:** Analyzes your data sources to determine:
- **ML Suitability** - Is this data appropriate for machine learning?
- **Target suggestions** - What columns could be prediction targets?
- **Data quality overview** - Initial assessment of the data

##### Step 2: Problem Understanding

**What it does:** Analyzes your data and description to determine:
- **Task type** - Is this classification, regression, or forecasting?
- **Target column** - What are you trying to predict?
- **Primary metric** - How should we measure success?

**Example:**
> You describe: "Predict which customers will cancel their subscription"
>
> AI determines:
> - Task type: Binary Classification
> - Target: "churned" column
> - Metric: ROC AUC (good for imbalanced data)

##### Step 3: Data Audit

**What it does:** Examines your data quality:
- Missing values - How much data is incomplete?
- Distributions - Are values spread evenly or skewed?
- Outliers - Are there unusual values?
- Correlations - Which columns relate to each other?
- Leakage candidates - Columns that might leak future information
- Recommendations - What should be fixed or removed?

**Example output:**
> - Column "age" has 3% missing values - recommend imputation
> - Column "customer_id" is unique for each row - should be excluded (not predictive)
> - Column "cancellation_reason" flagged as potential leakage

##### Step 4: Dataset Design

**What it does:** Decides which columns to use:
- **Include** - Columns that will help prediction
- **Exclude** - Columns that won't help or could hurt (including leakage risks)
- **Transform** - Columns that need processing

**Example:**
> Include: age, tenure, monthly_charges, contract_type, payment_method
>
> Exclude: customer_id (just an identifier), phone_number (privacy), email (not predictive)

##### Step 5: Experiment Design

**What it does:** Creates multiple training configurations to try:
- **Quick test** - Sanity check only (3 minutes), verifies data pipeline works
- **Quick experiment** - Fast training (15 minutes), good for initial testing
- **Balanced experiment** - Medium time (1 hour), good accuracy
- **High-quality experiment** - Longer training (2+ hours), best results

Each experiment includes:
- Time budget and quality preset
- Validation strategy (stratified, time-based, or custom)
- Expected tradeoffs

##### Step 6: Plan Critic (Validation)

**What it does:** Reviews everything before executing:
- Validates split strategy is appropriate (e.g., time-based for temporal data)
- Checks for leakage in selected features
- Ensures metrics are realistic
- May request revisions up to 2 times before proceeding
- Gets your approval before training starts

#### Seeing the AI's Thinking

You can expand each step to see:
- **Thoughts** - The AI's reasoning process
- **Warnings** - Potential issues it noticed
- **Outputs** - The specific configurations generated
- **Logs** - Detailed step-by-step actions

---

### Experiments & Training

An **Experiment** is a training run that produces models.

#### Creating an Experiment

You can create experiments:
1. **Automatically** - Through the AI wizard
2. **Manually** - Click "Create Experiment" and configure:
   - Name and description
   - Dataset specification to use
   - Time budget (how long to train)
   - Quality preset (best_quality, high_quality, good_quality, medium_quality)

#### Running Experiments

Click **"Run"** on an experiment to start training. You can:
- **Run one at a time** - Focus resources on one experiment
- **Run in batch** - Train multiple experiments in parallel

#### Training Options

| Option | What it Controls |
|--------|------------------|
| **Time Budget** | How long to train (5 min to hours) |
| **Presets** | Quality vs speed tradeoff |
| **CPU Cores** | How many processors to use |
| **Memory** | How much RAM to allocate |
| **Backend** | Local machine or cloud (Modal.com) |

#### Watching Progress

While training, you see:
- **Progress bar** - Percentage complete
- **Current stage** - Data prep, training, validation, etc.
- **Status messages** - What's happening right now
- **Auto-updates** - Page refreshes every 2 seconds

#### Experiment Status

| Status | Meaning |
|--------|---------|
| **Pending** | Waiting to start |
| **Running** | Currently training |
| **Completed** | Finished successfully |
| **Failed** | Something went wrong |
| **Cancelled** | You stopped it |

---

### Models

A **Model** is your trained prediction machine. After an experiment completes, you'll have one or more models.

#### Model Information

Each model shows:
- **Name** - Usually the algorithm type (LightGBM, XGBoost, etc.)
- **Final Score** - The authoritative performance metric (from holdout evaluation)
- **Validation Score** - Score during training (used for tuning)
- **Feature Importance** - Which columns matter most
- **Status** - Where it is in its lifecycle

#### Understanding Your Scores

AgentML uses a **three-tier validation system** to ensure your model really works:

| Score | What It Is | Purpose |
|-------|------------|---------|
| **Training Score** | Performance on data used to train | Baseline check |
| **Validation Score** | Performance during cross-validation | Used for tuning |
| **Final Score** | Performance on reserved holdout set | **The real score** |

**Why does this matter?**
- The **Final Score** is always displayed prominently - this is your true performance
- A model might look great on validation but perform worse on holdout (overfitting)
- AgentML reserves 15% of your data that the model never sees during training
- This holdout set gives you an honest assessment of real-world performance

#### Overfitting Detection

When the validation score is much better than the holdout score, your model may be "overfitting" (memorizing patterns that don't generalize).

| Gap | What It Means | UI Indicator |
|-----|---------------|--------------|
| < 5% | Normal variation | No warning |
| 5-10% | Possible overfitting | Yellow warning banner |
| > 10% | Significant overfitting | Red warning, may block promotion |

**Example:**
> Validation Score: 0.92 (92%)
>
> Final Score: 0.85 (85%)
>
> Gap: 0.07 (7%)
>
> ⚠️ Warning: Model may be overfitting. The holdout score is 7% lower than validation.

#### Model Lifecycle

Models progress through stages:

```
Draft → Candidate → Shadow → Production → Retired
```

| Stage | Meaning |
|-------|---------|
| **Draft** | Just created, under evaluation |
| **Candidate** | Good performer, being considered |
| **Shadow** | Testing alongside production |
| **Production** | The active model being used |
| **Retired** | No longer in use |

Click **"Promote"** to move a model to the next stage.

#### Promotion Guardrails

AgentML protects you from promoting potentially problematic models. Models with certain risks require explicit justification:

| Risk | What Triggers It | Required Action |
|------|------------------|-----------------|
| **High Overfitting** | Gap > 10% between validation and holdout | Must provide override reason |
| **Suspected Leakage** | Failed label-shuffle test | Must acknowledge and explain |
| **Too Good to Be True** | AUC > 80% on time-based prediction | Must confirm understanding |

**Why guardrails?**
- Prevents accidentally deploying unreliable models
- Forces acknowledgment of known risks
- All overrides are logged in the Lab Notebook for audit

#### Making Predictions

Go to the **Test** tab to try your model:
1. Enter values for each feature
2. Click **"Predict"**
3. See the prediction (and probabilities for classification)

#### Understanding Your Model

The **Explain** tab lets you ask questions:
- "Why is this feature important?"
- "What patterns did you learn?"
- "Why did you predict X for this customer?"

The AI explains in plain language.

#### Validation Analysis

The **Validation** tab shows:
- All predictions on held-out test data
- Actual vs predicted values
- Errors sorted from worst to best
- Click any row for details

##### What-If Analysis

Found an interesting prediction? Click **"What-If"** to:
1. Modify any feature values
2. See how the prediction changes
3. Understand which features drive the outcome

**Example:**
> Original: Customer predicted to churn (85% confidence)
>
> What-if: Change monthly_charges from $89 to $59
>
> Result: Churn prediction drops to 42%
>
> Insight: Price is a major factor in churn

---

### Visualizations

The **Visualize** tab helps you explore your data visually.

#### Creating Visualizations

1. Select a data source
2. Type what you want to see:
   - "Show me a histogram of ages"
   - "Create a scatter plot of price vs sales"
   - "Show correlation between all numeric columns"
3. Click **"Generate"**
4. The AI creates Python code and renders the chart

#### Default Visualizations

When you first view a dataset, AI suggests:
- Distribution plots for numeric columns
- Bar charts for categorical columns
- Correlation heatmaps
- Missing value summaries

#### Understanding Visualizations

Click **"Explain"** on any chart to get:
- What the visualization shows
- Key patterns to notice
- Implications for your analysis

---

## User Flows

### Flow 1: Quick Start with AI

1. **Create Project** - Name it and describe your goal
2. **Upload Data** - Add your CSV or Excel file
3. **Start AI Wizard** - Click "Set up with AI"
4. **Review Steps** - Watch AI analyze and configure
5. **Approve Plan** - Review experiments before running
6. **Run Training** - Start all experiments
7. **View Results** - Check your trained models
8. **Make Predictions** - Use your best model

**Time: 15-30 minutes** (plus training time)

### Flow 2: Manual Configuration

1. **Create Project** - Set task type manually
2. **Upload Data** - Add your files
3. **Create Dataset Spec** - Select target and features yourself
4. **Create Experiment** - Configure time and quality
5. **Run Training** - Start the experiment
6. **Evaluate Model** - Review metrics and validation
7. **Iterate** - Adjust and try again if needed

**Time: 30-60 minutes** (plus training time)

### Flow 3: Finding Additional Data

1. **Create Project** - Describe your goal
2. **Click "Search with AI"** - In the Data tab
3. **Describe Needed Data** - "I need customer demographic data"
4. **Review Results** - See discovered datasets
5. **Select and Download** - Add to your project
6. **Combine Sources** - Use multiple data sources together

### Flow 4: Model Explanation

1. **Train Models** - Complete an experiment
2. **Select Best Model** - From the Models tab
3. **Go to Explain Tab** - Click on the model
4. **Ask Questions** - Type natural language questions
5. **Get Insights** - AI explains in plain English

---

## Advanced Features

### Feature Leakage Detection

**Data leakage** happens when your model accidentally uses information that wouldn't be available at prediction time. AgentML automatically detects potential leakage during experiment setup.

#### How It Works

1. **Name-based detection** - Scans column names for suspicious patterns:
   - Future indicators: `future_`, `next_`, `will_`, `predicted_`
   - Outcome data: `result_`, `outcome_`, `final_`, `actual_`
   - Post-event data: `_at_event`, `post_`, `after_`

2. **Correlation analysis** - Flags columns with suspiciously high correlation (>95%) to the target

3. **Time-based checks** - For time-series tasks, warns if using random splits instead of time-based splits

#### Severity Levels

| Severity | Example | Action |
|----------|---------|--------|
| **High** | `cancellation_reason` when predicting churn | Blocks training until removed |
| **Medium** | Very high correlation with target | Warning, continues with note |
| **Low** | Minor suspicious pattern | Informational only |

**Example:**
> Predicting: "Will customer churn?"
>
> ⛔ High severity: Column `cancellation_date` detected. This directly leaks the outcome.
>
> ⚠️ Medium severity: Column `days_until_cancel` has 0.98 correlation with target.

### Auto-Improve Iterations

AgentML can automatically iterate on your experiments to improve performance:

1. **Start Auto-Improve** - Click "Auto-Improve" on any completed experiment
2. **AI Analysis** - The system analyzes what worked and what didn't
3. **Automatic Iteration** - Creates improved experiments with:
   - Better feature engineering
   - Optimized hyperparameters
   - Refined data preprocessing
4. **Overfitting Protection** - Automatically stops if performance degrades

#### Feature Engineering Feedback

The auto-improve system tracks which feature engineering attempts succeed or fail:

| Feedback Type | What it Means |
|--------------|---------------|
| **Successful Features** | Transformations that improved the model |
| **Failed Features** | Attempted columns that didn't exist or couldn't be created |
| **Suggestions** | Similar column names when a feature failed |

This feedback helps the AI avoid repeating mistakes and build on successes.

#### Overfitting Monitor

View the **Overfitting Monitor** on any experiment to see:

- **Overall Risk Level** - Low (0-30%), Medium (30-60%), High (60%+)
- **Holdout Score Trend** - Chart showing performance across iterations
- **Per-Iteration Status** - Detailed breakdown for each training run
- **Recommendation** - Whether to continue, proceed with caution, or stop

**How it works:**
- A 15% holdout set is reserved and never used for training
- Each iteration is evaluated on this same holdout set
- Risk is calculated from:
  - Score degradation from the best iteration
  - Declining trend in recent iterations
  - Number of iterations since the best score

### Batch Experiments

Run multiple experiments simultaneously:
1. Create several experiments with different settings
2. Select all you want to run
3. Click **"Run Batch"**
4. Monitor all progress in parallel
5. Compare results when complete

### Real-Time Collaboration

Multiple team members can:
- View the same project
- See live training progress
- Review models together
- Share insights via chat

### Synthetic Data Generation

Need more data for testing? Generate realistic fake data:

1. Go to **Synthetic Data** section
2. Click **"Create Dataset"**
3. Choose type (classification, regression, etc.)
4. Set number of features and rows
5. AI generates realistic data
6. Download as CSV or use directly

**Use cases:**
- Testing without sensitive real data
- Augmenting small datasets
- Privacy-compliant demos

### Chat Assistant

Every project has an AI chat assistant that knows about:
- Your data characteristics
- Experiment configurations
- Model performance
- Best practices

Ask questions like:
- "Why is my model accuracy low?"
- "What should I try next?"
- "Explain this metric to me"

### Project Settings

Click the **⚙️ Settings** button in any project to configure advanced options.

#### Large Dataset Safeguards

Control how large datasets are handled to prevent performance issues:

| Setting | Default | What it Controls |
|---------|---------|------------------|
| **Max Training Rows** | 1,000,000 | Maximum rows in materialized training datasets. Larger datasets are automatically sampled. |
| **Profiling Sample Rows** | 50,000 | Sample size used for data profiling and schema analysis. Balances accuracy with speed. |
| **Max Aggregation Window (days)** | 365 | Maximum time window for aggregations when joining tables. Limits lookback for time-based features. |

**How sampling works:**
- When your dataset exceeds the max training rows limit, AgentML uses random sampling
- The sampling is reproducible (same seed each time)
- You see a clear message like "Dataset is large (25M rows). Sampling 1M rows for training dataset."
- The summary shows both the sampled size and indicates sampling occurred

**Why it matters:**
- Prevents out-of-memory errors on large datasets
- Keeps training times reasonable
- Still produces high-quality models (1M rows is typically sufficient)
- You can increase limits for specific projects if needed

---

## Sharing & Collaboration

### Sharing a Project

1. Open your project
2. Click the **Share** button
3. Enter collaborator's email
4. Choose their role:
   - **Viewer** - Can see everything, can't change
   - **Editor** - Can modify experiments and data
   - **Admin** - Can also manage other shares
5. They receive an email invitation

### Accepting an Invitation

1. Check your email for the invite
2. Click the invitation link
3. Sign in (or create account)
4. Project appears in your list

### Managing Shares

As owner or admin:
- View all shared users
- Change roles
- Remove access
- See pending invitations

---

## Technical Details

### Supported ML Algorithms

AgentML uses **AutoGluon** which automatically tries many algorithms:

**For Classification:**
- LightGBM (gradient boosting)
- XGBoost (gradient boosting)
- CatBoost (gradient boosting)
- Random Forest
- Neural Networks
- And many more...

**For Regression:**
- Same algorithms adapted for numeric prediction

**For Time Series:**
- AutoARIMA
- ETS (Exponential Smoothing)
- DeepAR
- Temporal Fusion Transformer
- And more...

### Metrics Explained

| Metric | Used For | What it Measures |
|--------|----------|------------------|
| **Accuracy** | Classification | % of correct predictions |
| **ROC AUC** | Binary Classification | Ability to rank positive cases higher |
| **F1 Score** | Classification | Balance of precision and recall |
| **RMSE** | Regression | Average prediction error |
| **MAE** | Regression | Average absolute error |
| **R²** | Regression | % of variance explained |
| **MAPE** | Time Series | % error in forecasts |

### File Size Limits

- Maximum upload: **100 MB** per file
- Recommended: Under 50 MB for best performance
- For larger files: Use Parquet format (more efficient)

### API Keys

To use AI features, you need API keys from:
- **OpenAI** - For GPT-based features (most common)
- **Gemini** - Alternative from Google
- **Anthropic** - Alternative (Claude)

Add keys in **Settings** → **API Keys**

Your keys are encrypted and stored securely.

---

## Troubleshooting

### Common Issues

**"My experiment failed"**
- Check the error message in experiment details
- Common causes: missing values, wrong data types
- Try the Data Audit step to identify issues

**"My model accuracy is low"**
- Add more training time
- Try the "high_quality" preset
- Check if you have enough data (aim for 1000+ rows)
- Review feature selection - maybe important columns were excluded

**"The AI wizard got something wrong"**
- You can manually override any AI suggestion
- Edit dataset specs and experiments after creation
- The AI learns from diverse data - some domains may need guidance

**"My data looks wrong in preview"**
- Check the delimiter setting for CSV files
- Make sure column headers are in the first row
- Verify the file isn't a data dictionary (metadata about data)

### Getting Help

1. **Use the Chat Assistant** - Ask questions about your specific situation
2. **Check Model Explanations** - Often reveals issues
3. **Review Agent Logs** - See AI reasoning in pipeline steps
4. **Contact Support** - For technical issues

---

## Glossary

| Term | Definition |
|------|------------|
| **AutoML** | Automatic Machine Learning - lets computers choose the best algorithms |
| **Binary Classification** | Predicting one of two outcomes (yes/no, true/false) |
| **Dataset Spec** | Configuration of which columns to use for training |
| **Experiment** | A training run with specific settings |
| **Feature** | An input column used for prediction |
| **Feature Importance** | How much each input affects the prediction |
| **Feature Leakage** | When a model accidentally uses future/unavailable information |
| **Holdout Set** | Reserved data (15%) never seen during training for honest evaluation |
| **Model** | A trained algorithm that can make predictions |
| **Overfitting** | When a model memorizes training data instead of learning patterns |
| **Regression** | Predicting a numeric value |
| **Target** | The column you're trying to predict |
| **Training** | The process of teaching a model from data |
| **Validation** | Testing model performance on unseen data |

---

## Quick Reference

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Close modal | `Escape` |
| Submit form | `Enter` |

### Status Colors

| Color | Meaning |
|-------|---------|
| 🟢 Green | Completed, Success |
| 🔵 Blue | Running, In Progress |
| 🟡 Yellow | Pending, Warning |
| 🔴 Red | Failed, Error |
| ⚪ Gray | Cancelled, Inactive |

---

*Last updated: March 2026*

# Architecture

## Overview

AgentML is a production-grade machine learning research platform designed to handle messy, real-world datasets (multi-table, time-based, >1M rows) and produce robust, auditable ML models.

## Design Principles

- **LLMs are planners, not executors**: They propose dataset specs, experiment plans, and interpretations. All data operations and checks are implemented in backend code.
- **Holdout-first scoring**: The holdout score is the canonical metric. Validation scores guide training, holdout scores judge models.
- **Idempotency**: All long-running processes can be resumed safely.
- **Auditability**: Every agent step includes `agent_name`, `parent_agent_name`, and structured streaming logs.
- **Security**: Environment variables for secrets/config (never hardcoded).
- **Data integrity**: Final test set isolation, leakage detection, and robustness audits in code.

## High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│                    React + TypeScript (Vite)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Backend API                              │
│                     Python + FastAPI                             │
├─────────────────────────────────────────────────────────────────┤
│  app/api/              │  REST endpoints                         │
│  app/services/         │  Business logic                         │
│  app/services/agents/  │  LLM agent class implementations        │
│  app/services/llm/     │  LLM client, prompts                    │
│  app/tasks/            │  Celery background tasks                │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  PostgreSQL  │    │    Redis     │    │   Modal.com      │
│  (Database)  │    │ (Task Queue) │    │ (Cloud Training) │
└──────────────┘    └──────────────┘    └──────────────────┘
```

## Backend Structure

```
backend/app/
├── api/                    # REST API routes
│   ├── projects.py         # Project CRUD
│   ├── data_sources.py     # File upload, data source management
│   ├── dataset_specs.py    # Dataset specification endpoints
│   ├── experiments.py      # Experiment management
│   └── models.py           # Model version endpoints
├── core/                   # Core infrastructure
│   ├── config.py           # Environment configuration
│   ├── database.py         # SQLAlchemy setup
│   └── exceptions.py       # Custom exception classes
├── models/                 # SQLAlchemy ORM models
│   ├── project.py
│   ├── data_source.py
│   ├── dataset_spec.py
│   ├── experiment.py
│   ├── model_version.py
│   └── agent_run.py        # AgentRun, AgentStep, AgentStepType
├── schemas/                # Pydantic validation schemas
├── services/               # Business logic
│   ├── agents/             # LLM agent implementations (class-based)
│   │   ├── base.py         # BaseAgent abstract class
│   │   ├── registry.py     # AgentStepType → Agent class mapping
│   │   ├── setup/          # Setup pipeline agents (6)
│   │   │   ├── data_analysis.py
│   │   │   ├── problem_understanding.py
│   │   │   ├── data_audit.py
│   │   │   ├── dataset_design.py
│   │   │   ├── experiment_design.py
│   │   │   └── plan_critic.py
│   │   ├── results/        # Results pipeline agents (2)
│   │   │   ├── results_interpretation.py
│   │   │   └── results_critic.py
│   │   ├── data_architect/ # Data architect pipeline agents (4)
│   │   │   ├── dataset_inventory.py
│   │   │   ├── relationship_discovery.py
│   │   │   ├── training_dataset_planning.py
│   │   │   └── training_dataset_build.py
│   │   ├── improvement/    # Improvement pipeline agents (6)
│   │   │   ├── iteration_context.py
│   │   │   ├── improvement_data_analysis.py
│   │   │   ├── improvement_dataset_design.py
│   │   │   ├── improvement_experiment_design.py
│   │   │   ├── improvement_analysis.py
│   │   │   └── improvement_plan.py
│   │   ├── standalone/     # Standalone agents (3)
│   │   │   ├── dataset_discovery.py
│   │   │   ├── lab_notebook_summary.py
│   │   │   └── robustness_audit.py
│   │   └── utils/
│   │       └── step_logger.py
│   ├── llm/                # LLM infrastructure
│   │   ├── client.py       # LLM API client
│   │   └── prompts.py      # Prompt templates
│   ├── agent_executor.py   # Step-based agent pipeline orchestration
│   ├── agent_service.py    # High-level agent service functions
│   ├── risk_scoring.py     # Overfitting & leakage risk assessment
│   ├── modal_runner.py     # Modal.com integration
│   └── modal_training_standalone.py  # Remote training function
└── tasks/                  # Celery background tasks
    ├── celery_app.py       # Celery configuration
    ├── experiment_tasks.py # Training orchestration
    └── agent_tasks.py      # Agent pipeline tasks
```

## Database Schema

### Core Tables

| Table | Purpose |
|-------|---------|
| `projects` | Top-level workspace container |
| `data_sources` | Uploaded files and their metadata |
| `dataset_specs` | Feature/target column configurations |
| `experiments` | Training run configurations |
| `trials` | Individual training trials within experiments |
| `model_versions` | Trained model artifacts and metrics |
| `agent_runs` | Agent pipeline execution records |
| `agent_steps` | Individual step records within agent runs |
| `agent_step_logs` | Detailed logs for each step (thinking, action, warning) |
| `lab_notebook_entries` | Human-readable experiment summaries |

### Key Relationships

```
Project
  └── DataSource[] (1:N)
  └── DatasetSpec[] (1:N)
       └── Experiment[] (1:N)
            └── Trial[] (1:N)
                 └── ModelVersion[] (1:N)
            └── AgentRun[] (1:N)
                 └── AgentStep[] (1:N)
                      └── AgentStepLog[] (1:N)
            └── LabNotebookEntry[] (1:N)
```

### Experiment Iteration Lineage

Experiments support parent-child relationships for auto-improvement:

```
Experiment (iteration=1, parent=null)
    └── Experiment (iteration=2, parent=iteration1)
        └── Experiment (iteration=3, parent=iteration2)
```

Each child experiment stores `improvement_context_json` with:
- Previous iteration scores
- Improvement hypothesis
- Changes made from parent

## Three-Tier Validation System

### Score Types

| Score | Source | Purpose | Visibility |
|-------|--------|---------|------------|
| `train_score` | Training set | Baseline fit check | Internal |
| `val_score` | Cross-validation (85% of data) | Hyperparameter tuning | Shown as "Validation Score" |
| `final_score` | Holdout set (15% reserved) | **Canonical comparison metric** | Shown as "Final Score" |

### Data Split Strategy

```
100% Data
├── 85% Training Pool
│   └── K-Fold Cross-Validation
│       ├── Fold 1: Train (80%) / Val (20%)
│       ├── Fold 2: Train (80%) / Val (20%)
│       └── ... (configurable folds)
└── 15% Holdout Set (NEVER touched during training)
```

### Holdout Evaluation Flow

1. **During Training**: AutoGluon trains on 85%, uses CV for model selection
2. **After Training**: Best model evaluated on reserved 15% holdout
3. **Metrics Storage**: Holdout metrics stored with `holdout_` prefix in `model_version.metrics_json`:
   ```json
   {
     "accuracy": 0.85,
     "roc_auc": 0.91,
     "holdout_accuracy": 0.82,
     "holdout_roc_auc": 0.88,
     "holdout_samples": 1500
   }
   ```

### Overfitting Detection

```python
overfitting_gap = val_score - holdout_score

if overfitting_gap > 0.05:
    # Warning: Significant overfitting detected
    # UI shows yellow warning banner

if overfitting_gap > 0.10:
    # Critical: Severe overfitting
    # May block model promotion
```

### API Response Fields

The `ExperimentDetailResponse` includes:

| Field | Type | Description |
|-------|------|-------------|
| `final_score` | float | Holdout score (canonical) |
| `val_score` | float | Validation/CV score |
| `train_score` | float | Training score |
| `has_holdout` | bool | Whether holdout evaluation ran |
| `holdout_samples` | int | Number of holdout samples |
| `overfitting_gap` | float | val_score - holdout_score |
| `score_source` | str | "holdout" or "validation" (fallback) |

## Training Pipeline

### Local Training

```
1. Load dataset from DatasetSpec
2. Create train/validation/holdout splits (85/15 with CV on train)
3. Run AutoGluon TabularPredictor with cross-validation
4. Select best model by validation score
5. Evaluate best model on holdout set → final_score
6. Compute overfitting gap and risk flags
7. Store all metrics and artifacts in ModelVersion
```

### Cloud Training (Modal.com)

```
1. Serialize dataset to JSON
2. Upload to Modal remote function
3. Execute training in cloud GPU environment
4. Return metrics and model artifacts (including holdout evaluation)
5. Download and store locally
```

## AI Agent System

See [AGENTS.md](AGENTS.md) for detailed agent documentation.

### Agent Architecture

All agents inherit from `BaseAgent` and are registered in `app/services/agents/registry.py`:

```python
from app.services.agents.registry import get_agent_class

agent_class = get_agent_class(step.step_type)
agent = agent_class(db, step, logger, llm_client)
output = await agent.execute()
```

### Pipeline Overview

| Pipeline | Agents | Purpose |
|----------|--------|---------|
| Setup | 6 | Configure experiment and generate training plan |
| Results | 2 | Analyze results and assess model quality |
| Data Architect | 4 | Build optimized training datasets from multiple sources |
| Improvement (Simple) | 2 | Quick improvement cycle for iterations |
| Improvement (Enhanced) | 4 | Full re-analysis with iteration context |
| Standalone | 3 | Independent tasks (discovery, summary, audit) |

**Total: 21 registered agents**

### Core Services

| Service | File | Purpose |
|---------|------|---------|
| Agent Registry | `agents/registry.py` | Maps AgentStepType to agent classes |
| Base Agent | `agents/base.py` | Abstract base class for all agents |
| Risk Scoring | `risk_scoring.py` | Overfitting, leakage, TGTBT detection |
| Agent Executor | `agent_executor.py` | Step-based pipeline orchestration |
| Step Logger | `agents/utils/step_logger.py` | Structured logging for agents |

## Risk Assessment System

### Risk Factors

| Factor | Detection | Penalty |
|--------|-----------|---------|
| Medium Overfitting | gap > 0.05 | -0.05 |
| High Overfitting | gap > 0.10 | -0.10 |
| Suspected Leakage | Label-shuffle test | -0.15 |
| Time-Split Issues | Random split on time data | -0.05 |
| Too Good To Be True | AUC > 0.80 on time-based | Blocks promotion |

### Promotion Guardrails

Models with critical risks require explicit override to promote:
- Suspected data leakage
- High overfitting risk
- Too-good-to-be-true metrics

Override requires a documented justification stored in the lab notebook.

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/projects` | Create project |
| GET | `/projects` | List projects |
| GET | `/projects/{id}` | Project detail |
| POST | `/projects/{id}/data-sources/upload` | Upload file |
| GET | `/projects/{id}/data-sources` | List data sources |
| POST | `/projects/{id}/dataset-specs` | Create dataset spec |
| POST | `/projects/{id}/experiments` | Create experiment |
| POST | `/experiments/{id}/run` | Start training |
| GET | `/experiments/{id}` | Experiment detail with holdout metrics |
| GET | `/experiments/{id}/iterations` | List experiment iterations |
| POST | `/experiments/{id}/improve` | Trigger auto-improvement |
| POST | `/models/{id}/promote` | Promote model stage |

## Frontend Components

### Key Pages

| Page | Path | Purpose |
|------|------|---------|
| Projects | `/projects` | Project list and creation |
| Project Detail | `/projects/:id` | Data sources, experiments |
| Experiment Detail | `/experiments/:id` | Results, metrics, iterations |
| Iteration Comparison | Component | Compare iterations side-by-side |

### Score Display Convention

- **Final Score**: Large, prominent display (holdout-based)
- **Validation Score**: Secondary, smaller display
- **Overfitting Warning**: Yellow banner when gap > 0.05
- **Iteration Navigator**: Shows final_score for each iteration

## Security & Provenance

- Every agent run, experiment, and model stores provenance metadata
- Secrets (LLM keys, DB URIs) read from environment variables
- All intermediate artifacts persisted and immutable
- Deterministic randomness with saved seeds
- Lab notebook entries link all decisions to experiments

## Standards

- **Traceability**: Every action stored with who/when/agent_name
- **Idempotency**: Jobs use deterministic keys and checkpoints
- **Auditability**: All artifacts persisted and immutable
- **Config via env vars**: No hardcoded secrets
- **Holdout-first**: All displayed scores use holdout unless unavailable

---

*Last updated: March 2026*

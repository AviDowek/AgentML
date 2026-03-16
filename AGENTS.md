# Agent System Architecture

This document describes the LLM-powered agent system that orchestrates ML experiment setup, analysis, and improvement.

## Core Principles

### Safety by Design
- **Agents propose, backend executes**: LLM agents produce structured JSON proposals that are validated and executed by deterministic backend code
- **No direct data manipulation**: Agents analyze and recommend; they cannot directly transform data or make irreversible decisions
- **Audit trail**: All agent outputs, prompts, responses, and decisions are persisted and linked to experiments via the Lab Notebook

### Class-Based Architecture

All agents inherit from `BaseAgent` and follow a consistent pattern:

```python
class MyAgent(BaseAgent):
    name = "my_agent"
    step_type = AgentStepType.MY_STEP

    async def execute(self) -> Dict[str, Any]:
        # Read inputs
        data = self.require_input("some_key")

        # Log progress
        self.logger.thinking("Analyzing data...")

        # Call LLM
        result = await self.llm.chat_json(messages, schema)

        # Return output
        return {"result": result}
```

### Agent Registry

Agents are registered in `app/services/agents/registry.py` and dispatched by `AgentStepType`:

```python
from app.services.agents.registry import get_agent_class

agent_class = get_agent_class(AgentStepType.DATA_ANALYSIS)
agent = agent_class(db, step, logger, llm_client)
output = await agent.execute()
```

## Agent Folder Structure

```
backend/app/services/agents/
├── __init__.py
├── base.py                 # BaseAgent abstract class
├── registry.py             # Agent type → class mapping
├── setup/                  # Setup pipeline agents
│   ├── data_analysis.py
│   ├── problem_understanding.py
│   ├── data_audit.py
│   ├── dataset_design.py
│   ├── experiment_design.py
│   └── plan_critic.py
├── results/                # Results pipeline agents
│   ├── results_interpretation.py
│   └── results_critic.py
├── data_architect/         # Data architect pipeline agents
│   ├── dataset_inventory.py
│   ├── relationship_discovery.py
│   ├── training_dataset_planning.py
│   └── training_dataset_build.py
├── improvement/            # Improvement pipeline agents
│   ├── iteration_context.py
│   ├── improvement_data_analysis.py
│   ├── improvement_dataset_design.py
│   ├── improvement_experiment_design.py
│   ├── improvement_analysis.py
│   └── improvement_plan.py
├── standalone/             # Standalone agents
│   ├── dataset_discovery.py
│   ├── lab_notebook_summary.py
│   └── robustness_audit.py
├── orchestration/          # Pipeline orchestration agents
│   ├── project_manager.py    # Dynamic pipeline orchestration
│   ├── gemini_critique.py    # Debate system critique agent
│   ├── openai_judge.py       # Debate system judge
│   └── debate_manager.py     # Debate orchestrator
└── utils/
    └── step_logger.py      # StepLogger for agent logging
```

## Agent Pipelines

### 1. Setup Pipeline (6 Agents)

Orchestrates experiment configuration and generates the training plan.

| Step | Agent Class | Purpose |
|------|-------------|---------|
| 1 | `DataAnalysisAgent` | Analyzes data sources, determines ML suitability, recommends target |
| 2 | `ProblemUnderstandingAgent` | Understands the ML task, suggests task type and metrics |
| 3 | `DataAuditAgent` | Audits data quality, detects nulls, outliers, leakage candidates |
| 4 | `DatasetDesignAgent` | Designs feature set, exclusions, and preprocessing strategy |
| 5 | `ExperimentDesignAgent` | Generates experiment variants with time budgets and presets |
| 6 | `PlanCriticAgent` | Validates plan, checks for leakage, validates split strategy |

**Plan Critic Feedback Loop**: If `PlanCriticAgent` rejects a plan, it loops back to `ExperimentDesignAgent` with revision instructions. Maximum 2 revision attempts before proceeding with warnings.

### 2. Results Pipeline (2 Agents)

Analyzes training results and assesses model quality.

| Step | Agent Class | Purpose |
|------|-------------|---------|
| 1 | `ResultsInterpretationAgent` | Interprets metrics, identifies best model, summarizes results |
| 2 | `ResultsCriticAgent` | Reviews results for issues, overfitting, and suspicious patterns |

### 3. Data Architect Pipeline (4 Agents)

Builds optimized training datasets from multiple data sources.

| Step | Agent Class | Purpose |
|------|-------------|---------|
| 1 | `DatasetInventoryAgent` | Profiles all data sources, catalogs structure and statistics |
| 2 | `RelationshipDiscoveryAgent` | Discovers join keys and relationships between tables |
| 3 | `TrainingDatasetPlanningAgent` | Plans joins, aggregations, and feature engineering |
| 4 | `TrainingDatasetBuildAgent` | Materializes the final training dataset |

### 4. Improvement Pipeline - Simple (2 Agents)

Quick improvement cycle for iterating on experiments.

| Step | Agent Class | Purpose |
|------|-------------|---------|
| 1 | `ImprovementAnalysisAgent` | Analyzes what worked/didn't, identifies improvement areas |
| 2 | `ImprovementPlanAgent` | Creates actionable plan with feature and config changes |

### 5. Improvement Pipeline - Enhanced (4 Agents)

Full re-analysis with iteration context for deeper improvements.

| Step | Agent Class | Purpose |
|------|-------------|---------|
| 1 | `IterationContextAgent` | Gathers full iteration history, errors, and insights |
| 2 | `ImprovementDataAnalysisAgent` | Re-analyzes data with feedback from previous iterations |
| 3 | `ImprovementDatasetDesignAgent` | Redesigns features based on what worked/failed |
| 4 | `ImprovementExperimentDesignAgent` | Redesigns experiment config with iteration insights |

### 6. Standalone Agents

These agents run independently, not as part of a pipeline.

| Agent Class | Purpose |
|-------------|---------|
| `DatasetDiscoveryAgent` | Searches for relevant external datasets |
| `LabNotebookSummaryAgent` | Generates human-readable research cycle summaries |
| `RobustnessAuditAgent` | Audits experiments for overfitting and suspicious patterns |

## Pipeline Orchestration

The orchestration system provides advanced pipeline control through two optional modes:

### Orchestration Modes

#### 1. Sequential Mode (Default)
Agents execute in a fixed order as defined in the pipeline. This is the traditional mode where each agent runs once in sequence.

#### 2. Project Manager Mode
A meta-agent (`ProjectManagerAgent`) dynamically orchestrates the pipeline:

- Decides which agent runs next based on previous outputs
- Can run agents multiple times if needed
- Declares when the pipeline is complete
- Provides reasoning for each orchestration decision

**How it works:**
```
┌──────────────────────────────────────────────────────────────┐
│   Project Manager Agent receives context and history          │
│   ↓                                                          │
│   Decides: Which agent next? Or is pipeline complete?        │
│   ↓                                                          │
│   If not complete → Run selected agent → Add to history      │
│   ↓                                                          │
│   Loop until PM declares completion                          │
└──────────────────────────────────────────────────────────────┘
```

**Configuration:**
```python
run_setup_pipeline(
    project_id=project.id,
    orchestration_mode="project_manager",  # Enable PM mode
)
```

### Debate System

The debate system adds quality assurance through agent debate:

#### Components

| Agent | LLM Provider | Purpose |
|-------|--------------|---------|
| Main Agent | Anthropic (Claude) | Produces initial output |
| `GeminiCritiqueAgent` | Google Gemini | Reviews and critiques output |
| `OpenAIJudgeAgent` | OpenAI | Makes final decision if no consensus |

#### Debate Flow

1. **Main agent** produces output
2. **Gemini critic** reviews and either agrees or critiques
3. If critic disagrees:
   - Main agent responds to concerns
   - Critic responds back
   - Continue for up to **3 rounds**
4. If still no consensus:
   - **OpenAI Judge** reviews full transcript
   - Judge makes final decision (main agent, critic, or synthesis)
5. Final output returned with debate transcript

```
┌──────────────────────────────────────────────────────────────┐
│   Main Agent Output                                          │
│   ↓                                                          │
│   Gemini Critique: Agree? ──Yes──> Consensus (use output)    │
│   ↓ No                                                       │
│   Main Agent Response                                        │
│   ↓                                                          │
│   Gemini Critique: Agree? ──Yes──> Consensus                 │
│   ↓ No                                                       │
│   ... (up to 3 rounds) ...                                   │
│   ↓                                                          │
│   OpenAI Judge Decision ──> Final output selected            │
└──────────────────────────────────────────────────────────────┘
```

#### Judge Model Selection

Users can choose which OpenAI model serves as the judge:
- `gpt-4o` (default) - Best quality decisions
- `gpt-4o-mini` - Faster, lower cost
- `gpt-4-turbo` - High quality
- `gpt-4` - Original GPT-4
- `o1-preview` - Advanced reasoning
- `o1-mini` - Efficient reasoning

**Configuration:**
```python
run_setup_pipeline(
    project_id=project.id,
    debate_mode="enabled",
    judge_model="gpt-4o",  # Optional: defaults to gpt-4o
)
```

### Combined Usage

Both modes can be enabled simultaneously:

```python
run_setup_pipeline(
    project_id=project.id,
    orchestration_mode="project_manager",
    debate_mode="enabled",
    judge_model="gpt-4o",
)
```

When combined:
1. Project Manager decides which agent runs
2. Each agent step goes through debate
3. Final (potentially debated) output informs PM's next decision

### Database Fields

The `agent_runs` table stores orchestration settings:

| Field | Type | Description |
|-------|------|-------------|
| `orchestration_mode` | enum | `sequential` or `project_manager` |
| `debate_mode` | enum | `disabled` or `enabled` |
| `judge_model` | string | OpenAI model for judge (e.g., `gpt-4o`) |
| `debate_transcript_json` | JSON | Full debate history for the run |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/orchestration/options` | GET | Available orchestration modes and judge models |
| `/orchestration/judge-models` | GET | List of available judge models |

## BaseAgent Interface

All agents inherit from `BaseAgent` which provides:

```python
class BaseAgent(ABC):
    # Class attributes (set by subclass)
    name: str = "base_agent"
    step_type: Optional[AgentStepType] = None

    # Instance attributes (set in __init__)
    db: Session           # Database session
    step: AgentStep       # Current step being executed
    logger: StepLogger    # For logging progress
    llm: BaseLLMClient    # LLM client for chat
    input_data: dict      # step.input_json

    # Required method
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute the agent's task and return output dict."""
        pass

    # Helper methods
    def get_input(self, key: str, default=None) -> Any
    def require_input(self, key: str) -> Any
    def get_nested_input(self, *keys: str, default=None) -> Any

    # Convenience properties
    @property
    def project_id(self) -> Optional[str]
    @property
    def experiment_id(self) -> Optional[str]
    @property
    def data_source_id(self) -> Optional[str]
```

## StepLogger

Records agent activity for audit trail:

```python
self.logger.thinking("Analyzing 50 columns...")
self.logger.action("Excluding high-cardinality columns")
self.logger.warning("Target imbalance detected: 95/5 split")
self.logger.summary("Found 3 potential leakage columns")
self.logger.info("Processing complete")
self.logger.error("Failed to load data")
```

All logs are persisted to `agent_step_logs` table with step_id linking to the agent step.

## Core Services

### Risk Scoring Service (`risk_scoring.py`)

Evaluates model trustworthiness:

1. **`compute_risk_adjusted_score()`**: Applies penalties to metrics based on risks
   - Medium overfitting: -0.05
   - High overfitting: -0.10
   - Leakage suspected: -0.15
   - Time-split suspicious: -0.05

2. **`check_too_good_to_be_true()`**: Flags suspiciously high performance
   - AUC > 0.80 on time-based classification
   - MCC > 0.50 on time-based prediction

3. **`get_model_risk_status()`**: Determines if model promotion requires override
   - Returns: `(risk_level, requires_override, reason)`
   - Risk levels: "low", "medium", "high", "critical"

## Holdout Scoring System

Three-tier validation to prevent overfitting:

| Score Type | Source | Purpose |
|------------|--------|---------|
| `train_score` | Training set | Baseline, detects underfitting |
| `val_score` | Cross-validation/validation set | Hyperparameter tuning |
| `final_score` | Holdout set (15% reserved) | **Canonical score for comparison** |

**Key principle**: The `final_score` (holdout) is the authoritative metric. Validation scores are used during training but never for final comparison.

**Overfitting Detection**:
```python
overfitting_gap = val_score - holdout_score
if overfitting_gap > 0.05:
    # Warning: Model may be overfitting
if overfitting_gap > 0.10:
    # Critical: Severe overfitting - may block promotion
```

## Feature Leakage Detection

### Detection Methods

1. **Name-based heuristics**: Flags columns with suspicious names
   - Future indicators: `future_`, `next_`, `will_`, `predicted_`
   - Outcome leakage: `result_`, `outcome_`, `final_`
   - Temporal: `_at_event`, `post_`, `after_`

2. **Correlation analysis**: High correlation with target (>0.95) flags review

3. **Metadata inspection**: Checks for columns derived from target

### Severity Levels

- **High**: Definite leakage (e.g., `cancellation_reason` when predicting churn) - blocks training
- **Medium**: Suspicious but uncertain - warns and logs
- **Low**: Minor concern - informational only

## Plan Critic Validation

The `PlanCriticAgent` performs these checks:

### Split Strategy Validation
- Rejects random splits on time-based data
- Accepts override with `time_split_override: true` and justification

### Leakage Validation
- Rejects plans including high-severity leakage features
- Warns on medium-severity features

### Metric Validation
- Warns when target metrics exceed expected ranges
- Ensures realistic performance expectations

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Project Created                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│   Dataset Uploaded → Data Source Created                            │
│   - Schema analyzed                                                 │
│   - Statistics computed                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│   Setup Pipeline (6 Agents)                                         │
│   1. DataAnalysisAgent                                              │
│   2. ProblemUnderstandingAgent                                      │
│   3. DataAuditAgent                                                 │
│   4. DatasetDesignAgent                                             │
│   5. ExperimentDesignAgent                                          │
│   6. PlanCriticAgent ←─────┐                                        │
│         │                  │ (max 2 revisions)                      │
│         └── if rejected ───┘                                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│   Training Execution (AutoGluon)                                    │
│   - Data split with 15% holdout                                     │
│   - Cross-validation on remaining 85%                               │
│   - Best model selected by val_score                                │
│   - Final evaluation on holdout → final_score                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│   Results Pipeline (2 Agents)                                       │
│   1. ResultsInterpretationAgent                                     │
│   2. ResultsCriticAgent                                             │
│      - Overfitting risk assessment                                  │
│      - Too-good-to-be-true check                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│   Optional: Auto-Improve                                            │
│   Simple: ImprovementAnalysisAgent → ImprovementPlanAgent           │
│   Enhanced: Full 4-agent improvement pipeline                       │
│   - Analyzes what worked/didn't                                     │
│   - Generates improvement hypotheses                                │
│   - Creates child experiment with modifications                     │
│   - Tracks iteration lineage                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Creating a New Agent

1. Create a new file in the appropriate subfolder (e.g., `agents/setup/my_agent.py`)

2. Implement the agent class:
```python
from typing import Any, Dict
from app.models import AgentStepType
from app.services.agents.base import BaseAgent

class MyNewAgent(BaseAgent):
    name = "my_new_agent"
    step_type = AgentStepType.MY_STEP_TYPE

    async def execute(self) -> Dict[str, Any]:
        # Get required inputs
        data = self.require_input("input_key")

        # Log what you're doing
        self.logger.thinking("Analyzing input data...")

        # Call LLM if needed
        messages = [
            {"role": "system", "content": "You are an expert..."},
            {"role": "user", "content": f"Analyze: {data}"},
        ]
        result = await self.llm.chat_json(messages, response_schema)

        # Log results
        self.logger.summary(f"Analysis complete: {result.get('summary')}")

        # Return output (becomes step.output_json)
        return result
```

3. Add to `__init__.py` in the subfolder:
```python
def __getattr__(name):
    if name == "MyNewAgent":
        from app.services.agents.setup.my_agent import MyNewAgent
        return MyNewAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [..., "MyNewAgent"]
```

4. Register in `registry.py`:
```python
from app.services.agents.setup import MyNewAgent

return {
    ...
    AgentStepType.MY_STEP_TYPE: MyNewAgent,
}
```

## Prohibitions

Agents must NOT:
- Invent column names, metrics, or data values not present in the dataset
- Execute data transformations directly
- Make irreversible decisions without backend validation
- Bypass leakage or overfitting checks
- Hide warnings or risks from the user

## Auditing

All agent activity is persisted:
- **`agent_runs` table**: Run metadata, status, timestamps
- **`agent_steps` table**: Individual step records with inputs/outputs
- **`agent_step_logs` table**: Detailed logs (thinking, action, warning, etc.)
- **Lab Notebook**: Human-readable summaries linked to experiments
- **`improvement_context_json`**: Iteration history and improvement reasoning

This enables full reproducibility and debugging of any agent decision.

---

*Last updated: March 2026 - Added Pipeline Orchestration (Project Manager Mode) and Debate System*

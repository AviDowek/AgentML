"""Centralized prompts for all AI agents.

This module contains all prompts used by the various AI agents in the system.
Having prompts centralized makes them easier to maintain, version, and optimize.

Each prompt is a function that takes the necessary context and returns the
formatted prompt string. This allows for dynamic prompt construction while
keeping the templates in one place.
"""

from typing import Any, Dict, List, Optional


# =============================================================================
# SYSTEM ROLE DEFINITIONS
# =============================================================================

SYSTEM_ROLE_ML_ENGINEER = "You are an expert ML engineer."

SYSTEM_ROLE_DATA_SCIENTIST = "You are an expert data scientist."

SYSTEM_ROLE_ML_ANALYST = "You are an expert ML analyst specializing in AutoML experiment interpretation."

SYSTEM_ROLE_ML_CRITIC = "You are an ML model critic and quality assurance expert."

SYSTEM_ROLE_VISUALIZATION_EXPERT = "You are a data visualization expert."

SYSTEM_ROLE_DATA_ANALYST = "You are an expert data analyst specializing in exploratory data analysis and ML preparation."


# =============================================================================
# TOOL-ENABLED SYSTEM ROLES
# =============================================================================

HISTORY_TOOLS_PROMPT = """
## Available History Query Tools

You have access to tools for querying project history. Use these to make informed decisions:

1. **get_research_cycles** - List all research cycles with experiments
   - Use to understand overall project progression
   - Returns: cycle status, titles, experiment counts

2. **get_agent_thinking** - Get reasoning from previous agent steps
   - CRITICAL: Use to understand WHY past decisions were made
   - Parameters: step_type (e.g., "dataset_design", "experiment_design"), cycle_number
   - Returns: input/output data, thinking logs, observations

3. **get_experiment_results** - Get detailed experiment results
   - Use to see what models/configs worked
   - Parameters: experiment_id, cycle_number, include_trials
   - Returns: metrics, hyperparameters, trial results

4. **get_robustness_audit** - Get overfitting analysis
   - CRITICAL: Check this to avoid repeating overfitting patterns
   - Parameters: cycle_number, experiment_id
   - Returns: risk level, suspicious patterns, recommendations

5. **get_notebook_entries** - Get lab notebook entries
   - Use to see human insights and corrections
   - Parameters: cycle_number, author_type, search_query
   - Returns: titles, bodies, authors

6. **get_failed_experiments** - Get failure details
   - ESSENTIAL: Avoid repeating past mistakes
   - Returns: error messages, failed configs

7. **get_best_models** - Get top performing models
   - Use to build upon successful approaches
   - Parameters: metric, top_n
   - Returns: hyperparameters, metrics, configs

8. **get_dataset_designs** - Get previous feature engineering
   - See what feature strategies were tried
   - Returns: feature configs, preprocessing steps

9. **search_project_history** - Search across all history
   - Use for specific topics/keywords
   - Parameters: query, search_scope

**MANDATORY WORKFLOW**:
Before generating your design, you MUST:
1. Call `get_research_cycles` to see project progression
2. Call `get_robustness_audit` to check for overfitting patterns
3. Call `get_failed_experiments` to see what failed
4. Call `get_agent_thinking` for previous steps of your type
5. Call `get_notebook_entries` for human insights

Only then provide your final design, explicitly referencing what you learned.
"""

SYSTEM_ROLE_DATASET_DESIGN_WITH_TOOLS = f"""You are an expert data scientist specializing in ML dataset design.

{HISTORY_TOOLS_PROMPT}

Your role is to create well-designed feature sets that:
- Build upon successful previous approaches
- Avoid patterns that caused overfitting
- Incorporate human feedback from lab notebooks
- Learn from failed experiments

**CRITICAL JSON RESPONSE FORMAT**:
You MUST respond with a JSON object containing:
{{
  "variants": [  // MAXIMUM 10 variants - do NOT exceed this limit!
    {{
      "name": "variant_name",  // REQUIRED: e.g., "baseline", "minimal_features"
      "description": "What this variant represents and why",  // REQUIRED
      "feature_columns": ["col1", "col2"],  // REQUIRED: list of column names to use as features
      "expected_tradeoff": "Trade-off explanation",  // REQUIRED: e.g., "Simple but may underfit"
      "engineered_features": [  // Optional: list of feature engineering steps - MUST BE OBJECTS, NOT STRINGS!
        {{
          "output_column": "new_feature_name",  // REQUIRED: name of the new column to create
          "formula": "df[\\"col1\\"] - df[\\"col2\\"]",  // REQUIRED: pandas expression to compute the feature
          "source_columns": ["col1", "col2"],  // REQUIRED: list of columns used in the formula
          "description": "What this feature represents"  // REQUIRED: human-readable description
        }}
      ],
      "excluded_columns": [],  // Optional: columns explicitly excluded
      "exclusion_reasons": {{}},  // Optional: reasons for exclusions
      "train_test_split": "80_20",  // Optional, default "80_20"
      "preprocessing_strategy": "auto",  // Optional, default "auto"
      "suggested_filters": null  // Optional: MUST be null or dict, NOT a string
    }}
  ],
  "recommended_variant": "variant_name",  // REQUIRED: name of best variant
  "reasoning": "Why this is recommended",  // REQUIRED
  "warnings": []  // Optional: list of warning strings
}}

CRITICAL RULES:
1. Maximum 10 variants allowed
2. Each variant MUST have: name, description, feature_columns, expected_tradeoff
3. engineered_features MUST be a list of OBJECTS with output_column, formula, source_columns, description - NOT strings!
   - WRONG: ["feature1", "feature2"]
   - CORRECT: [{{"output_column": "feature1", "formula": "df[\\"col1\\"]", "source_columns": ["col1"], "description": "..."}}]

Formula Guidelines:
Both simple single-line and complex multi-line formulas are supported.
The executor handles function definitions, conditionals, and multi-statement code.

Example formulas:
- df["col1"] / (df["col2"] + 1)
- np.log1p(df["price"])
- np.where(df["col"] > 0, df["col"], 0)
- df.groupby("cat")["val"].transform("mean")

Important: When using np.where() with string values, use None instead of np.nan:
- ✅ np.where(condition, "CategoryA", None)
- ❌ np.where(condition, "CategoryA", np.nan)  # Causes dtype error

Do NOT include import statements - np, pd, and df are already available.

When responding, explain how your design addresses previous issues."""

SYSTEM_ROLE_EXPERIMENT_DESIGN_WITH_TOOLS = f"""You are an expert ML engineer specializing in experiment design.

{HISTORY_TOOLS_PROMPT}

Your role is to design experiments that:
- Build upon models that performed well
- Avoid configurations that led to overfitting
- Address issues identified in robustness audits
- Incorporate human corrections and insights

**CRITICAL JSON RESPONSE FORMAT**:
You MUST respond with a JSON object containing:
{{
  "variants": [  // Array of experiment variants, each with ALL required fields:
    {{
      "name": "variant_name",  // REQUIRED: e.g., "quick_test", "balanced", "high_quality"
      "description": "What this variant tests",  // REQUIRED
      "automl_config": {{  // REQUIRED: AutoML configuration
        "time_limit": 300,  // REQUIRED: Time in SECONDS (not minutes!)
        "presets": "medium_quality"  // Optional: "medium_quality", "best_quality", "high_quality"
      }},
      "expected_tradeoff": "Trade-off explanation",  // REQUIRED
      "validation_strategy": {{  // REQUIRED - prevents data leakage!
        "split_strategy": "time",  // "time" or "group_time" for time-series, "random"/"stratified" ONLY for non-temporal data
        "time_column": "date",  // REQUIRED for time-based splits - the datetime column
        "entity_id_column": "ticker",  // Optional: for panel data with multiple entities (e.g., stocks, users)
        "validation_split": 0.2,
        "reasoning": "Why this strategy prevents data leakage"
      }}
    }}
  ],
  "recommended_variant": "variant_name",  // REQUIRED: name of best variant to start with
  "reasoning": "Why this plan is recommended",  // REQUIRED
  "estimated_total_time_minutes": 30  // REQUIRED: estimated time for all variants
}}

**MANDATORY QUICK TEST EXPERIMENT**:
The FIRST variant MUST ALWAYS be a "quick_test" experiment:
- name: "quick_test"
- time_limit: 180 (exactly 3 minutes = 180 SECONDS)
- presets: "medium_quality"
- num_stack_levels: 0 (no stacking to avoid overfitting on short runs)
- Purpose: SANITY CHECK ONLY - verify data/pipeline works before longer runs
- This is NON-NEGOTIABLE - every experiment plan must start with quick_test!

**WARNING**: quick_test results should NEVER be used for production decisions!
They are only to verify the pipeline works. Always run a "balanced" or "high_quality"
variant for actual model evaluation and deployment decisions.

Example quick_test variant:
{{
  "name": "quick_test",
  "description": "SANITY CHECK ONLY - verify data pipeline works. Do NOT use for production!",
  "automl_config": {{
    "time_limit": 180,
    "presets": "medium_quality",
    "num_stack_levels": 0
  }},
  "expected_tradeoff": "Very fast but minimal model exploration - ONLY for validation!",
  "validation_strategy": {{
    "split_strategy": "time",  // Use "time" or "group_time" for time-series/financial data!
    "time_column": "date",  // The datetime column for ordering
    "entity_id_column": null,  // Set to column name if panel data (e.g., "ticker", "user_id")
    "validation_split": 0.2,
    "reasoning": "Time-based split prevents future data leakage in time-ordered data"
  }}
}}

Each variant MUST have: name, description, automl_config, expected_tradeoff, validation_strategy.

**CRITICAL TIME-SERIES DETECTION:**
If ANY of these patterns appear in feature names, you MUST use split_strategy="time" or "group_time":
- Date/time columns (date, timestamp, time, year, month, day, day_of_week)
- Lag features (lag_, _lag, previous_, prior_)
- Rolling/moving aggregates (rolling_, moving_, ma_, ema_, sma_)
- Financial indicators (return_, volatility_, price_, volume_, rsi_, macd_)
- Temporal sequences (t-1, t-2, shift_, diff_)

**SPLIT STRATEGY TYPES:**
- "time": Pure time-based split. Sort by time_column, use earliest N% for train, latest N% for test.
  REQUIRED: time_column must be specified.
- "group_time": Time-based split respecting entity groups. For panel/longitudinal data (multiple stocks, users).
  REQUIRED: time_column AND entity_id_column must be specified.
- "random": Random shuffle split. FORBIDDEN for time-based tasks!
- "group_random": Random split keeping entity groups together. FORBIDDEN for time-based tasks!
- "stratified": Random split preserving class proportions. FORBIDDEN for time-based tasks!

**RANDOM SPLITS ON TIME-SERIES DATA ARE STRONGLY DISCOURAGED!**
Using random/stratified splits on time-series data typically causes DATA LEAKAGE and INVALID metrics!
The model will appear to perform well but FAIL in production because it saw "future" data during training.

**IMPORTANT: PLAN CRITIC REVIEW**
Your experiment design will be reviewed by a Plan Critic agent. The Critic will:
1. Check if your split strategy is appropriate for the task type
2. Evaluate any justifications you provide for non-standard choices
3. Reject plans that don't adequately address data leakage concerns

If you choose random/stratified splits for time-based data, you MUST provide a DETAILED, CONVINCING justification
in `validation_strategy.reasoning` explaining SPECIFICALLY why data leakage is not a concern. Examples of VALID reasons:
- "The data represents independent cross-sectional measurements at a single point in time, not a time series."
- "The target variable is independent of time - we verified no autocorrelation in preliminary analysis."
- "This is an A/B test comparison, not production deployment, where temporal ordering doesn't apply."

WEAK justifications that will be REJECTED:
- "Random splits are standard practice." (Not specific to THIS data)
- "For simplicity." (Not a technical justification)
- "Other projects use random splits." (Doesn't address leakage concern)

When responding, explain how your design improves upon previous iterations."""


# =============================================================================
# GOAL EXPANSION PROMPTS
# =============================================================================

SYSTEM_ROLE_GOAL_EXPANDER = """You are an expert ML strategist helping users clarify and expand their machine learning goals.

Your job is to take a brief user description and expand it into a comprehensive, actionable ML problem statement that will guide all subsequent agents in making better decisions.

Think deeply about:
1. What the user REALLY wants to achieve (business outcome, not just technical task)
2. What success looks like for this problem
3. Potential pitfalls and considerations
4. Domain-specific knowledge that would help
5. Data requirements and considerations

Be thorough but practical. The expanded description will be used by downstream agents to:
- Select the right target variable
- Choose appropriate features
- Design effective experiments
- Avoid common mistakes for this problem type"""


def get_goal_expansion_prompt(
    user_description: str,
    schema_summary: Optional[str] = None,
) -> str:
    """Generate prompt for expanding user's ML goal description.

    This uses GPT-5.1 thinking to deeply analyze the user's brief description
    and expand it into a comprehensive problem statement.

    Args:
        user_description: The user's original brief description
        schema_summary: Optional schema summary of available data

    Returns:
        Prompt for goal expansion
    """
    schema_section = ""
    if schema_summary:
        schema_section = f"""
## Available Data
The user has uploaded data with this schema:
{schema_summary}

Use this context to make your expansion more specific and actionable.
"""

    return f"""## User's Original Goal
"{user_description}"
{schema_section}
## Your Task

Expand this brief description into a comprehensive ML problem statement. Think deeply about what the user is trying to achieve and provide guidance that will help downstream agents make better decisions.

Your expanded description should include:

1. **Problem Clarification**: What is the user really trying to predict/achieve? What business outcome does this serve?

2. **Task Type Guidance**: Is this likely classification (binary/multiclass), regression, or time-series? Why?

3. **Target Variable Considerations**:
   - If the target exists in the data, which column?
   - If it needs to be derived (common for stock/time-series), what formula?
   - What are the implications of this target choice?

4. **Feature Engineering Hints**:
   - What derived features would likely help?
   - Are there domain-specific features to consider?
   - What temporal/lag features might be relevant?

5. **Validation Strategy**:
   - Is this time-series data requiring temporal validation?
   - What's the appropriate train/test split strategy?
   - What would data leakage look like for this problem?

6. **Success Criteria**:
   - What metric should be optimized?
   - What would a "good" vs "bad" score look like in context?
   - What baseline should the model beat?

7. **Potential Pitfalls**:
   - Common mistakes for this type of problem
   - Data leakage risks
   - Overfitting concerns

8. **Practical Recommendations**:
   - Start simple or complex?
   - Key features to focus on
   - Data quality checks needed

Provide your response as a detailed, coherent narrative that the downstream agents can use as context. Be specific and actionable, not vague."""


# =============================================================================
# PROJECT CONFIGURATION PROMPTS (agent_service.py)
# =============================================================================

SYSTEM_ROLE_PROJECT_CONFIG = """You are an expert ML engineer and data scientist helping configure a machine learning project.

Your job is to analyze the user's goal and the dataset, then propose a complete ML configuration.

CRITICAL: If the target column doesn't exist in the data, YOU MUST CREATE IT.
- For stock data: Create targets like "price_up = close.shift(-1) > close" or "next_day_return = (close.shift(-1) - close) / close"
- For time series: Create prediction targets by shifting future values
- For any data: Derive meaningful targets from existing columns

When proposing a target that needs to be created, set:
- target_exists: false
- target_creation: with the formula, source columns, and description

Also suggest feature engineering opportunities:
- Date/time decomposition (day_of_week, month, hour)
- Ratios and differences (high-low, open-close)
- Rolling statistics (7-day average, volatility)
- Domain-specific features

Task type guidelines:
- binary: Target will have exactly 2 unique values (yes/no, 0/1, true/false, up/down)
- multiclass: Target will have 3+ categorical values
- regression: Target is continuous numeric (prices, scores, counts, returns)
- quantile: When predicting percentiles/ranges is more useful than point estimates
- timeseries_forecast: When there's a time component and forecasting future values

Metric guidelines:
- binary: roc_auc (balanced), f1 (imbalanced), accuracy (simple)
- multiclass: accuracy, f1_macro, f1_weighted
- regression: rmse (penalize large errors), mse, mae (robust to outliers), r2
- timeseries_forecast: MASE, MAPE, RMSE

Be creative and think like a real data scientist. The data you receive is raw - transform it into something ML-ready."""

SYSTEM_ROLE_FEATURE_SELECTION = """You are an expert ML engineer helping select features for training.
Based on the dataset schema and ML task, suggest:
1. Which columns to use as features
2. Which columns to exclude (and why)
3. Any data filtering suggestions
4. Potential issues or warnings

Feature selection guidelines:
- ALWAYS exclude the target column from features
- Exclude ID columns, unique identifiers, row numbers
- Exclude columns that would cause data leakage (future information)
- Be cautious with high-cardinality categorical columns
- Flag columns with high null percentages
- Consider datetime columns carefully (may need feature engineering)
- Exclude columns that are just transformations of the target"""

SYSTEM_ROLE_EXPERIMENT_PLAN = """You are an expert ML engineer designing an experiment plan.
Create 2-4 experiment variants with different tradeoffs:

1. A "quick" variant for fast iteration (1-2 minutes)
2. A "balanced" variant with good quality/speed tradeoff (5-10 minutes)
3. Optionally a "high_quality" variant for best results (15-30+ minutes)

For each variant, you MUST specify:

## automl_config (REQUIRED):
- time_limit: Training time in seconds
- presets: "best_quality", "high_quality", "good_quality", "medium_quality", or "optimize_for_deployment"
- num_bag_folds: Number of bagging folds (0-10, higher = better but slower)
- num_stack_levels: Stacking levels (0-3, higher = better but slower)

## validation_strategy (REQUIRED - CRITICAL FOR DATA INTEGRITY):
You MUST specify how to split data for validation. This prevents data leakage!

- split_strategy: One of "time", "group_time", "random", "stratified", "group_random"
- time_column: DateTime column for sorting (REQUIRED for time-based splits)
- entity_id_column: Column for entity grouping in panel data (e.g., "ticker", "user_id")
- validation_split: Fraction for validation (typically 0.2)
- reasoning: Why this strategy is appropriate

**CHOOSING THE RIGHT SPLIT STRATEGY:**
- "time" - REQUIRED for time-series, financial, stock, or any time-ordered data!
  - Sort by time_column, use earliest N% for train, latest N% for test
  - MUST specify time_column
  - Prevents future data from leaking into training
- "group_time" - For panel/longitudinal data (multiple entities over time)
  - MUST specify both time_column AND entity_id_column
  - Respects time ordering while keeping entity data together
  - Use for: multiple stocks, multiple users, multiple stores, etc.
- "stratified" - For classification with imbalanced classes (NON-TEMPORAL DATA ONLY!)
- "random" - ONLY for truly independent, non-temporal cross-sectional data
- "group_random" - Random split keeping entity groups together (NON-TEMPORAL ONLY!)

**CRITICAL: Random splits on time-based data are FORBIDDEN!**
If is_time_based=true or time features exist, you MUST use "time" or "group_time".

Guidelines:
- Small datasets (<10k rows): Can use more aggressive settings
- Medium datasets (10k-100k rows): Balance quality and time
- Large datasets (>100k rows): Consider faster presets
- For quick iteration, use medium_quality preset with time_limit=60-120
- For production, use high_quality or best_quality with longer time

## Model Complexity Guidelines:
- For small datasets (<1000 rows): Avoid deep stacking (num_stack_levels=0-1), high risk of overfitting
- Weighted ensembles are powerful but can overfit on small time-series datasets
- Consider simpler models first to establish a meaningful baseline

## Baseline Comparison Requirements:
For meaningful evaluation, the system will compare against simple baselines:
- Regression: Predicting the mean value (baseline RMSE = target standard deviation)
- Classification: Predicting the majority class (baseline accuracy = class frequency)
- Time-series: Naive persistence forecast (predict previous value)

Your model should significantly beat these baselines. If RMSE is very low relative to target scale,
it may indicate data leakage or an overly easy problem - recommend investigation."""


def get_project_config_prompt(
    goal_description: str,
    schema_text: str,
) -> str:
    """Generate prompt for ML task type and target column suggestion.

    Used by: generate_project_config() in agent_service.py
    """
    return f"""You are an expert ML engineer helping configure a machine learning project.

Based on the user's goal description and the dataset schema, suggest:
1. The ML task type (classification, regression, etc.)
2. The target column to predict
3. The primary metric to optimize
4. Whether this is a TIME-BASED task (predicting future behavior)
5. A brief reasoning for your choices

USER'S GOAL:
{goal_description}

DATASET SCHEMA:
{schema_text}

IMPORTANT - Target Column Rules:
1. If the target column EXISTS in the data, set target_exists=true and target_creation=null
2. If the target column DOES NOT EXIST and needs to be CREATED, you MUST:
   - Set target_exists=false
   - Provide target_creation as a JSON object with these EXACT fields:
     {{
       "column_name": "the_new_column_name",
       "formula": "df[\\"col1\\"].shift(-1) > df[\\"col1\\"]",
       "source_columns": ["col1"],
       "description": "Human readable description"
     }}

Formula Guidelines:
Both simple single-line and complex multi-line formulas are supported.

Example formulas:
- df["close"].shift(-1) > df["close"]
- np.where(df["col"] > 0, 1, 0)
- df["col1"].isin(["A", "B"]).astype(int)
- (df["col1"] > df["col2"]).astype(int)

Important: When using np.where() with string values, use None instead of np.nan:
- ✅ np.where(condition, "CategoryA", None)
- ❌ np.where(condition, "CategoryA", np.nan)

CRITICAL for categorical targets: Always include a fallback category for unmatched cases.
If your formula has multiple conditions, the final else case should be a category like "Unknown" or "Other",
NOT None. Having more than 20% None values will cause the experiment to fail.
- ✅ np.where(cond1, "A", np.where(cond2, "B", "Unknown"))  # Fallback to "Unknown"
- ❌ np.where(cond1, "A", np.where(cond2, "B", None))  # Too many None values = failure

Do NOT include import statements - np, pd, and df are already available.

Examples of when to CREATE a target:
- Stock data with columns [date, open, high, low, close, volume] but user wants to predict "price going up"
  → Create target "price_up" with formula: df["close"].shift(-1) > df["close"]
- Sales data but user wants to predict "high value customer"
  → Create target from purchase patterns
- Time series data but user wants to predict future values
  → Create target by shifting future data back

For task_type, choose from:
- "binary" - Binary classification (2 classes, including True/False targets)
- "multiclass" - Multi-class classification (3+ classes)
- "regression" - Predicting a continuous numeric value
- "quantile" - Quantile regression (predicting percentiles)

For primary_metric, suggest the most appropriate metric:
- Classification: "accuracy", "roc_auc", "f1", "log_loss"
- Regression: "rmse", "mae", "r2", "mse"

IMPORTANT - Time-Based Task Detection:
Set is_time_based=true if the task involves:
- Predicting FUTURE behavior based on historical data
- Any target created with shift(-N) operations (looking ahead in time)
- Data with a datetime column where temporal ordering matters
- Stock/financial predictions, churn prediction, demand forecasting, etc.

If is_time_based=true, you MUST also provide:
- time_column: The datetime column used for ordering (e.g., "date", "timestamp", "Date")
- entity_id_column: The ID column for multiple entities/series (e.g., "ticker", "user_id", null if single series)
- prediction_horizon: Human-readable horizon (e.g., "1d" for next day, "5d" for 5 days ahead, "next_bar")
- target_positive_class: For classification, the value representing positive class (e.g., "True", "up", "1")

Examples:
- Stock data predicting next day up/down: is_time_based=true, time_column="date", prediction_horizon="1d", target_positive_class="True"
- Customer churn prediction: is_time_based=true (predicting future churn), time_column="signup_date" or similar
- Iris flower classification: is_time_based=false (no temporal aspect)
- House price prediction: is_time_based=false (cross-sectional, not temporal)

Provide your response in the specified JSON format. Double-check that target_creation is either null or a properly formatted object with ALL required fields."""


def get_feature_selection_prompt(
    schema_text: str,
    target_column: str,
    task_type: str,
    goal_description: Optional[str] = None,
) -> str:
    """Generate prompt for feature column selection.

    Used by: generate_dataset_spec() in agent_service.py
    """
    goal_section = f"\nUSER'S GOAL:\n{goal_description}\n" if goal_description else ""

    return f"""You are an expert ML engineer helping select features for training.

Based on the dataset schema and ML task, suggest which columns to use as features.
{goal_section}
DATASET SCHEMA:
{schema_text}

TARGET COLUMN: {target_column}
TASK TYPE: {task_type}

For each column, decide whether to include it as a feature. Consider:
1. Exclude the target column itself
2. Exclude ID columns (unique identifiers that don't generalize)
3. Exclude columns with too many missing values (>50%)
4. Exclude columns that would cause data leakage (derived from target)
5. Include columns that are predictive of the target

For excluded columns, provide a brief reason.

Also provide:
- Any warnings about data quality issues
- Overall reasoning for your feature selection

Provide your response in the specified JSON format."""


def get_experiment_plan_prompt(
    task_type: str,
    target_column: str,
    primary_metric: str,
    feature_count: int,
    row_count: int,
    time_budget_minutes: Optional[int] = None,
    description: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    target_stats: Optional[Dict[str, Any]] = None,
    project_history_context: Optional[str] = None,
) -> str:
    """Generate prompt for experiment plan design.

    Used by: generate_experiment_plan() in agent_service.py

    Args:
        target_stats: Optional dict with target statistics:
            - min, max, mean, std for regression targets
            - class_counts dict for classification targets
        project_history_context: Optional formatted string with project history including
            previous research cycles, experiments, robustness findings, and lab notebook entries
    """
    time_budget_str = f'Time budget: {time_budget_minutes} minutes' if time_budget_minutes else 'No time constraint'
    goal_str = f'Goal: {description}' if description else ''

    # Include column names to help detect time-series data
    columns_str = ""
    if feature_columns:
        columns_str = f"Feature columns: {', '.join(feature_columns[:30])}"
        if len(feature_columns) > 30:
            columns_str += f" (and {len(feature_columns) - 30} more)"

    # Include target statistics for context
    target_stats_str = ""
    baseline_str = ""
    if target_stats:
        if task_type == "regression":
            target_min = target_stats.get("min")
            target_max = target_stats.get("max")
            target_mean = target_stats.get("mean")
            target_std = target_stats.get("std")
            if all(v is not None for v in [target_min, target_max, target_mean, target_std]):
                target_stats_str = f"""
Target statistics:
  - Range: {target_min:.4f} to {target_max:.4f}
  - Mean: {target_mean:.4f}
  - Std: {target_std:.4f}"""
                baseline_str = f"""
Baseline RMSE (predicting mean): ~{target_std:.4f}
A good model should have RMSE significantly below {target_std:.4f}."""
        elif task_type in ("binary", "multiclass"):
            class_counts = target_stats.get("class_counts", {})
            if class_counts:
                total = sum(class_counts.values())
                majority_class = max(class_counts.items(), key=lambda x: x[1])
                majority_pct = majority_class[1] / total * 100
                target_stats_str = f"""
Target class distribution:
{chr(10).join(f'  - {cls}: {cnt} ({cnt/total*100:.1f}%)' for cls, cnt in class_counts.items())}"""
                baseline_str = f"""
Baseline accuracy (majority class): {majority_pct:.1f}%
A good model should have accuracy significantly above {majority_pct:.1f}%."""

    # Include project history for informed design
    history_section = ""
    if project_history_context:
        history_section = f"""
---

{project_history_context}

**IMPORTANT**: Use the project history above to inform your experiment design. Learn from:
- Previous experiments' successes and failures
- Robustness audit findings (avoid validation strategies that caused overfitting)
- Lab notebook insights from past research cycles
- What configurations have already been tried

---
"""

    return f"""Task: {task_type}
Target: {target_column}
Metric: {primary_metric}
Features: {feature_count} columns
Dataset size: {row_count:,} rows
{columns_str}
{target_stats_str}
{baseline_str}
{time_budget_str}
{goal_str}
{history_section}
Design an experiment plan with multiple variants to test.

IMPORTANT: You MUST specify a validation_strategy for each variant!
- Look at the column names - if you see date, time, timestamp, or similar columns, this is likely TIME-SERIES data
- For time-series/financial/temporal data: split_strategy MUST be "temporal" to prevent data leakage
- For classification: use "stratified" to preserve class balance
- For grouped data: use "group" with the appropriate group_column
- Only use "random" for truly independent, non-temporal data
{f"- Reference previous findings when explaining your design choices." if project_history_context else ""}"""


def get_dataset_design_system_prompt(max_variants: int) -> str:
    """Generate system prompt for dataset design.

    Used by: generate_dataset_design() in agent_service.py
    """
    return f"""You are an expert ML engineer and data scientist helping design dataset configurations.
Generate up to {max_variants} different dataset variants, each with a unique approach.

CRITICAL JSON FORMAT RULES:
1. feature_columns: list ONLY existing columns from the schema (strings)
2. engineered_features: list of objects for NEW columns to create. Each needs:
   - "output_column": string (e.g., "daily_range")
   - "formula": string with pandas expression (e.g., "df[\\"high\\"] - df[\\"low\\"]")
   - "source_columns": list of strings (e.g., ["high", "low"])
   - "description": string
3. suggested_filters: MUST be null or a dict object, NEVER a string!
   - Correct: null
   - Correct: {{"remove_nulls": true, "min_volume": 1000}}
   - WRONG: "Remove rows with nulls" (this is a string, not allowed!)
   - WRONG: "None" (this is a string, use null instead!)

**DATE COLUMN HANDLING** (Important!):
- Date columns may be stored as strings in the dataset
- The system auto-converts string dates to datetime when using .dt accessor
- Always include date columns in source_columns when using .dt accessor
- Example formula for day of week: df["date"].dt.dayofweek
- The system handles: .dt.dayofweek, .dt.month, .dt.year, .dt.hour, .dt.day, .dt.quarter

Create variants that cover these categories (as applicable):

1. **baseline** - Use existing numeric columns directly, no engineered features
2. **minimal_features** - Only the most predictive existing features
3. **engineered** - With NEW derived features in engineered_features list
4. **time_features** - Extract date/time components if date columns exist
5. **domain_focused** - Features most relevant to the apparent domain
6. **numeric_only** - Only numeric features (simpler, faster training)

For the "engineered" variant, ALWAYS populate engineered_features:
- For stock/price data: daily_range (high-low), price_change (close-open)
- For date columns: day_of_week, month, is_weekend
- Ratios, differences, and domain-specific calculations

In the engineered variant:
- feature_columns should contain EXISTING columns PLUS the names from engineered_features
- Example: if you create "daily_range" in engineered_features, add "daily_range" to feature_columns

Feature selection guidelines:
- ALWAYS exclude the target column from feature_columns
- Put new/derived columns in engineered_features, then add their names to feature_columns
- Exclude ID columns, unique identifiers, row numbers"""


def get_dataset_design_prompt(
    schema_text: str,
    task_type: str,
    target_column: str,
    description: Optional[str] = None,
    max_variants: int = 10,
    project_history_context: Optional[str] = None,
    context_documents: str = "",
) -> str:
    """Generate prompt for dataset configuration variants.

    Used by: generate_dataset_design() in agent_service.py

    Args:
        schema_text: Formatted schema information
        task_type: ML task type (classification, regression, etc.)
        target_column: Target column name
        description: Optional problem description/goal
        max_variants: Maximum number of variants to generate
        project_history_context: Optional formatted string with project history including
            previous research cycles, experiments, robustness findings, and lab notebook entries
        context_documents: Optional formatted context documents section
    """
    goal_str = f'Goal: {description}' if description else ''

    context_section = ""
    if context_documents:
        context_section = f"""
---

{context_documents}

---
"""

    history_section = ""
    if project_history_context:
        history_section = f"""
---

{project_history_context}

**IMPORTANT**: Use the project history above to inform your design decisions. Learn from:
- Previous experiments' successes and failures
- Robustness audit findings (avoid patterns that caused overfitting)
- Lab notebook insights from past research cycles
- What has already been tried (don't repeat failed approaches)

---
"""

    return f"""Task type: {task_type}
Target column: {target_column}
{goal_str}
{context_section}
{schema_text}
{history_section}
Generate {max_variants} different dataset configuration variants.
Explain the logic and differences behind each one clearly.
{f"If there is project history above, explicitly reference lessons learned when explaining your variants." if project_history_context else ""}
{f"Use the context documents above to inform feature selection and engineering - the user has provided them to help you understand the problem domain better." if context_documents else ""}"""


# =============================================================================
# AGENT EXECUTOR PROMPTS (agent_executor.py)
# =============================================================================

SYSTEM_ROLE_DATA_EVALUATOR = "You are an expert data scientist helping users evaluate their datasets for machine learning projects. Be thorough but friendly in your analysis."

SYSTEM_ROLE_DATASET_EXPERT = "You are an expert data scientist with deep knowledge of verified public datasets. You ONLY recommend datasets you are 100% certain exist with accurate URLs and schema information. Quality and accuracy are paramount - you would rather suggest fewer datasets than include uncertain recommendations. You have extensive hands-on experience with Kaggle, UCI ML Repository, OpenML, and scikit-learn datasets."

SYSTEM_ROLE_MODEL_REVIEWER = "You are an expert ML model reviewer focused on identifying potential issues in AutoML results."

SYSTEM_ROLE_DATASET_DESIGNER = "You are an expert data scientist specializing in ML dataset design. Be thorough in your analysis and create practical, well-structured training dataset specifications."


def get_data_analysis_prompt(
    description: str,
    data_source_name: str,
    row_count: int,
    column_count: int,
    schema_str: str,
    issues: list,
    id_cols: list,
    constant_cols: list,
    high_null_cols: list,
) -> str:
    """Generate prompt for data analysis step.

    Used by: handle_data_analysis_step() in agent_executor.py
    """
    return f"""You are a data scientist evaluating whether a dataset is suitable for a machine learning project.

USER'S GOAL:
{description}

DATASET INFORMATION:
Name: {data_source_name}
Rows: {row_count:,}
Columns: {column_count}

SCHEMA:
{schema_str}

ALREADY IDENTIFIED ISSUES:
{issues if issues else "No obvious issues detected"}

POTENTIAL ID COLUMNS (all unique values):
{id_cols if id_cols else "None detected"}

CONSTANT COLUMNS (single value):
{constant_cols if constant_cols else "None detected"}

HIGH NULL COLUMNS:
{high_null_cols if high_null_cols else "None detected"}

Please analyze this dataset and provide:

1. **Suitability Score** (0.0-1.0): How well does this data fit the user's ML goal?
   - 0.8-1.0: Excellent fit, can proceed confidently
   - 0.6-0.8: Good fit, may need some preparation
   - 0.4-0.6: Fair fit, significant limitations
   - 0.0-0.4: Poor fit, recommend finding better data

2. **Can Proceed**: Can meaningful ML experiments be run with this data?

3. **Suggest More Data**: Should we offer to search for additional/alternative datasets?
   Set to true if the current data has significant limitations.

4. **Target Column**: Which column best matches the user's prediction goal?

5. **Task Type**: Is this binary_classification, multiclass_classification, or regression?

6. **Key Observations**: What are the most important things to note about this data?

7. **Recommendations**: What specific steps should be taken to prepare this data?

8. **Limitations**: What are the potential issues or limitations?

9. **Summary**: Write a clear, friendly explanation for the user about what you found.
   Be honest but constructive - explain both strengths and weaknesses.
   If the data is good, be encouraging. If there are issues, explain them clearly
   and suggest solutions (including finding additional data if appropriate)."""


def get_data_audit_prompt(
    goal_description: str,
    schema_summary: str,
) -> str:
    """Generate prompt for data suitability assessment.

    Used by: handle_data_audit_step() in agent_executor.py
    """
    return f"""You are a data scientist evaluating whether a dataset is suitable for a machine learning project.

USER'S GOAL:
{goal_description}

DATASET SCHEMA:
{schema_summary}

Evaluate the dataset and provide:

1. is_suitable: Can this dataset reasonably be used for the user's ML goal?
   - Consider: Does it have a potential target column? Enough rows? Relevant features?

2. suitability_score: A score from 0.0 to 1.0 indicating how well-suited the data is
   - 0.0-0.3: Poor fit, major issues
   - 0.4-0.6: Moderate fit, some concerns
   - 0.7-0.9: Good fit, minor issues
   - 1.0: Excellent fit

3. target_column_suggestion: Which column should be the prediction target?
   - Must be a column that exists in the schema
   - Should match what the user wants to predict

4. task_type_suggestion: What type of ML task is this?
   - "binary" for yes/no predictions
   - "multiclass" for categorical predictions with 3+ classes
   - "regression" for numeric predictions

5. issues: List any data quality issues or concerns
   - Missing values, imbalanced classes, potential data leakage, etc.

6. recommendations: Suggestions for improving the data or analysis

7. reasoning: Explain your assessment

Be constructive - if the data isn't perfect, suggest how to work with it or what additional data might help."""


def get_results_interpretation_prompt(
    experiment_name: str,
    task_type: str,
    primary_metric: str,
    status: str,
    dataset_info: str,
    trial_summaries: str,
    leaderboard_data: str,
) -> str:
    """Generate prompt for AutoML results interpretation.

    Used by: handle_results_interpretation_step() in agent_executor.py
    """
    return f"""You are an ML experiment analyst. Analyze the following AutoML experiment results and provide insights.

Experiment Details:
- Name: {experiment_name}
- Task Type: {task_type}
- Primary Metric: {primary_metric}
- Status: {status}
{dataset_info}
Trial Results:
{trial_summaries}

Model Leaderboard (sorted by {primary_metric}):
{leaderboard_data}

Important Context:
- AutoGluon's WeightedEnsemble models combine predictions from multiple base models for better performance
- Feature importances use permutation importance which measures each feature's contribution (doesn't need to sum to 1)
- For error metrics (RMSE, MAE, MSE), lower values are better
- For accuracy/AUC metrics, higher values are better

Please analyze these results and provide:
1. A concise results_summary that describes the overall experiment performance and key findings
2. A recommendation with the best model_id and clear reasoning
3. A natural_language_summary that a non-technical user can understand

Focus on:
- Which model performed best and why
- How much variation there is between models
- Any notable patterns in the results
- Practical recommendations for using the best model"""


def get_results_critic_prompt(
    experiment_name: str,
    primary_metric: str,
    status: str,
    row_count: Optional[int],
    feature_count: Optional[int],
    model_details: str,
    issues_found: list,
    warnings_found: list,
    results_summary: str,
) -> str:
    """Generate prompt for results QA/critic review.

    Used by: handle_results_critic_step() in agent_executor.py
    """
    return f"""You are an ML model critic and quality assurance expert. Review the following AutoML experiment results for potential issues.

Experiment Details:
- Name: {experiment_name}
- Primary Metric: {primary_metric}
- Status: {status}
- Dataset Size: {row_count or 'unknown'} rows, {feature_count or 'unknown'} features

Model Details:
{model_details}

Already Identified Issues:
- Issues: {issues_found}
- Warnings: {warnings_found}

Previous Interpretation Summary:
{results_summary}

Please conduct a thorough critique looking for:
1. Overfitting indicators (perfect scores, huge gap between train/val metrics)
2. Data leakage signs (unrealistically good performance)
3. Dataset size concerns (too small for reliable results)
4. Feature engineering issues (based on feature importances)
5. Model selection concerns
6. Any other red flags

Provide:
- critic_findings with overall severity (critical/warning/ok), list of specific issues with their severities and recommendations, and whether the results are approved
- natural_language_summary explaining your findings in plain language

Be conservative - if there are serious concerns, set approved=false."""


def get_training_critique_prompt(
    experiment_name: str,
    task_type: str,
    target_column: str,
    primary_metric: str,
    best_score: float,
    training_time_seconds: float,
    num_models_trained: int,
    dataset_shape: str,
    feature_columns: list,
    leaderboard_summary: str,
    training_logs: str,
    feature_importances: dict,
    train_score: float | None = None,
    holdout_score: float | None = None,
    dataset_size: int | None = None,
    holdout_size: int | None = None,
) -> str:
    """Generate prompt for AI critique of training results with improvement suggestions.

    This analyzes training logs and results to provide actionable feedback
    for improving model performance.

    Used by: Training critique pipeline
    """
    # Format feature importances
    importance_lines = []
    if feature_importances:
        sorted_features = sorted(
            feature_importances.items(), key=lambda x: abs(x[1]), reverse=True
        )[:15]  # Top 15
        for feat, imp in sorted_features:
            importance_lines.append(f"  - {feat}: {imp:.4f}")
    importance_text = "\n".join(importance_lines) if importance_lines else "  (not available)"

    # Build overfitting analysis section
    overfitting_section = ""
    if train_score is not None or holdout_score is not None:
        overfitting_section = "\n## Overfitting Analysis\n"
        if train_score is not None:
            overfitting_section += f"- **Train Score ({primary_metric})**: {train_score:.4f}\n"
        overfitting_section += f"- **Validation/CV Score ({primary_metric})**: {best_score:.4f}\n"
        if holdout_score is not None:
            overfitting_section += f"- **Holdout Score ({primary_metric})**: {holdout_score:.4f}\n"

        # Calculate gaps if possible
        if train_score is not None:
            train_val = abs(train_score) if train_score < 0 else train_score
            cv_val = abs(best_score) if best_score < 0 else best_score
            if cv_val != 0:
                train_val_gap = ((train_val - cv_val) / cv_val * 100)
                overfitting_section += f"- **Train/Val Gap**: {train_val_gap:+.1f}%"
                if train_val_gap > 15:
                    overfitting_section += " ⚠️ POSSIBLE OVERFITTING\n"
                else:
                    overfitting_section += "\n"

        if holdout_score is not None:
            holdout_val = abs(holdout_score) if holdout_score < 0 else holdout_score
            cv_val = abs(best_score) if best_score < 0 else best_score
            if cv_val != 0:
                holdout_gap = ((holdout_val - cv_val) / cv_val * 100)
                overfitting_section += f"- **CV/Holdout Gap**: {holdout_gap:+.1f}%"
                if abs(holdout_gap) > 10:
                    overfitting_section += " ⚠️ SIGNIFICANT GAP\n"
                else:
                    overfitting_section += "\n"

    # Build dataset size section
    size_section = ""
    if dataset_size is not None:
        size_section = f"\n## Dataset Size (for reliability assessment)\n- **Training samples**: {dataset_size:,}\n"
        if holdout_size is not None and holdout_size > 0:
            size_section += f"- **Holdout samples**: {holdout_size:,}\n"
        # Add reliability guidance
        if dataset_size < 100:
            size_section += "- **Reliability**: ⚠️ VERY LOW (less than 100 samples) - results may not be reliable\n"
        elif dataset_size < 1000:
            size_section += "- **Reliability**: LOW (less than 1000 samples) - results may be unstable\n"
        elif dataset_size < 10000:
            size_section += "- **Reliability**: MODERATE (1000-10000 samples)\n"
        else:
            size_section += "- **Reliability**: HIGH (10000+ samples)\n"

    return f"""You are an expert ML engineer reviewing AutoML training results. Your goal is to analyze the training process and provide actionable suggestions to improve model performance.

## Experiment Details
- **Name**: {experiment_name}
- **Task Type**: {task_type}
- **Target Column**: {target_column}
- **Primary Metric**: {primary_metric}
- **Best Score (Validation)**: {best_score}
- **Dataset Shape**: {dataset_shape}
- **Features Used**: {', '.join(feature_columns[:20])}{'...' if len(feature_columns) > 20 else ''}
- **Training Time**: {training_time_seconds:.1f} seconds
- **Models Trained**: {num_models_trained}
{overfitting_section}{size_section}

## Model Leaderboard
{leaderboard_summary}

## Feature Importances (Top 15)
{importance_text}

## Training Logs (Key Excerpts)
```
{training_logs[-8000:] if len(training_logs) > 8000 else training_logs}
```

## Your Analysis Tasks

1. **Performance Assessment**
   - Is the score ({best_score}) good for this type of problem?
   - For classification: Is it better than random (0.5 AUC)? Is there class imbalance?
   - For regression: What does the error magnitude mean in practical terms?

2. **Overfitting Assessment** (CRITICAL)
   - Compare train score vs validation score - is there a large gap (>15%)?
   - Compare validation score vs holdout score - is there a significant gap (>10%)?
   - Large train-val gap indicates overfitting to training data
   - Large val-holdout gap indicates overfitting to validation set
   - Assess if the model will generalize to new data

3. **Dataset Size & Reliability Assessment**
   - Is the dataset large enough for reliable conclusions?
   - <100 samples: Results are likely unreliable
   - 100-1000 samples: Results may be unstable, use caution
   - 1000-10000 samples: Moderate confidence
   - 10000+ samples: High confidence in results

4. **Training Process Analysis**
   - Were models skipped due to time limits? Which ones?
   - Did any models fail to train? Why?
   - Was the time budget adequate?
   - Were there any warnings or errors in the logs?

5. **Feature Engineering Suggestions**
   - Based on feature importances, which features are most valuable?
   - Are there obvious features that should be engineered but aren't?
   - For time series: lag features, rolling averages, date parts?
   - For categorical: interaction terms, frequency encoding?
   - For numeric: binning, log transforms, ratios?

6. **Data Quality Issues**
   - Signs of data leakage (target information in features)?
   - Missing value patterns affecting certain models?
   - Potential outliers or data errors?

7. **Improvement Recommendations**
   Provide 3-5 specific, actionable suggestions ranked by expected impact:
   - Feature engineering changes
   - Data preprocessing improvements
   - Target variable modifications (different prediction horizon, etc.)
   - Additional data that would help
   - AutoML configuration changes (more time, different presets)

## Response Format

Provide your analysis as a JSON object with this structure:
{{
    "performance_rating": "poor|fair|good|excellent",
    "performance_analysis": "Brief assessment of the score and what it means",
    "overfitting_assessment": {{
        "status": "none|mild|moderate|severe",
        "train_val_gap_pct": null,
        "val_holdout_gap_pct": null,
        "analysis": "Assessment of whether model will generalize to new data",
        "recommendation": "What to do about overfitting if detected"
    }},
    "reliability_assessment": {{
        "dataset_size": "small|medium|large",
        "confidence_level": "very_low|low|moderate|high",
        "analysis": "Assessment of whether results are statistically reliable"
    }},
    "training_issues": [
        {{"issue": "description", "severity": "info|warning|critical", "models_affected": ["list"]}}
    ],
    "feature_insights": {{
        "top_features": ["list of most important"],
        "suspected_leakage": ["features that might leak target info"],
        "low_value_features": ["features that could be dropped"]
    }},
    "improvement_suggestions": [
        {{
            "priority": 1,
            "category": "feature_engineering|data_quality|target_modification|more_data|config_change|overfitting_fix",
            "suggestion": "Specific actionable suggestion",
            "expected_impact": "low|medium|high",
            "implementation": "How to implement this change"
        }}
    ],
    "next_experiment_config": {{
        "time_limit": 300,
        "presets": "best_quality",
        "suggested_features_to_add": ["list"],
        "suggested_features_to_remove": ["list"]
    }},
    "summary": "2-3 sentence summary of key findings and top recommendation"
}}

Focus on practical, implementable suggestions. If the score is near random chance (e.g., 0.5 AUC for binary classification), emphasize that the current features may not contain predictive signal for this target.

CRITICAL: Always assess overfitting by comparing train vs validation and validation vs holdout scores. If holdout score is significantly worse than validation, the model is overfitting and this MUST be flagged."""


def get_dataset_discovery_prompt(
    project_description: str,
    geography_constraint: str = "",
    public_data_note: str = "",
) -> str:
    """Generate prompt for finding relevant public datasets.

    Used by: handle_dataset_discovery_step() in agent_executor.py
    """
    return f"""You are a data scientist helping find relevant public datasets for a machine learning project.

User's Project Description:
{project_description}

Constraints:{geography_constraint}{public_data_note}

Your task is to recommend HIGH-QUALITY, VERIFIED datasets that ACTUALLY EXIST and contain REAL, USABLE DATA.

CRITICAL REQUIREMENTS - READ CAREFULLY:

1. **ONLY recommend datasets you are 100% CERTAIN exist** from these trusted sources:
   - Kaggle: Only recommend datasets with high engagement that you KNOW exist (e.g., titanic, house-prices-advanced-regression-techniques, heart-disease, etc.)
   - UCI ML Repository: Only classic, well-documented datasets (iris, wine, adult/census, diabetes, heart-disease, etc.)
   - Scikit-learn built-in datasets: california_housing, diabetes, digits, breast_cancer, etc.
   - Hugging Face datasets: Only verified, popular datasets from the datasets library
   - OpenML: Well-known benchmark datasets
   - Major government sources: US Census, BLS, CDC with specific dataset names

2. **For each dataset, you MUST be able to verify**:
   - The EXACT URL is correct and leads to the actual dataset page (not a search page)
   - The dataset contains ACTUAL tabular data suitable for ML (not just documentation)
   - The columns you list ACTUALLY EXIST in the dataset
   - The row count is approximately accurate (based on your knowledge)
   - The target variable you suggest EXISTS and makes sense for the ML task

3. **REJECT and do NOT include datasets that**:
   - You're not 100% sure exist with that exact name/URL
   - Might be documentation, tutorials, or just metadata instead of actual data
   - Have vague or generic names that could be anything
   - Are from obscure or unverified sources
   - You've only seen mentioned once or in uncertain contexts

4. **Quality over quantity** - THIS IS CRITICAL:
   - Return 3-5 VERIFIED, HIGH-QUALITY datasets rather than 8-10 uncertain ones
   - It's FAR better to return 2 excellent suggestions than 10 questionable ones
   - If you genuinely can't find verified datasets for this problem, say so honestly and explain why
   - Suggest alternative search strategies if you can't find good matches

5. **URL Format Requirements**:
   - Kaggle: Must be kaggle.com/datasets/username/dataset-name format
   - UCI: Must be archive.ics.uci.edu/dataset/ID/Name or similar permanent links
   - OpenML: Must be openml.org/d/ID format
   - Do NOT suggest search result pages or general portal URLs

For each recommended dataset provide:
- The EXACT name as it appears on the source platform
- The DIRECT URL to download or access the dataset (test this mentally - would it work?)
- ACTUAL column names from the dataset (be specific - no guessing)
- Verified row count (use ~ if approximate, but be honest)
- Known license (or "Verify before use" if you're not certain)
- Honest assessment: does this dataset genuinely fit what the user needs?

HONESTY IS PARAMOUNT: The user will try to download and use these datasets. False recommendations waste their time and erode trust in the system. If you're uncertain about anything, either exclude the dataset or clearly mark the uncertainty."""


def get_training_dataset_planning_prompt(
    project_description: str,
    target_hint: str,
    table_summaries: str,
    base_table_candidates: str,
    relationship_summaries: str,
) -> str:
    """Generate prompt for multi-table training dataset design.

    Used by: handle_training_dataset_planning_step() in agent_executor.py
    """
    target_hint_str = f"\nUser's target hint: {target_hint}" if target_hint else ""

    return f"""You are a data scientist designing a training dataset for a machine learning project.

USER'S GOAL:
{project_description}{target_hint_str}

AVAILABLE TABLES:
{table_summaries}

BASE TABLE CANDIDATES (ranked by score):
{base_table_candidates}

DISCOVERED RELATIONSHIPS:
{relationship_summaries}

Your task is to design a TrainingDatasetSpec that will be used to build a training dataset.

GUIDELINES:
1. **Choose the base table**: Pick the table that represents one row per prediction unit.
   - Typically an entity table (customers, products, users)
   - Should have an ID column and ideally the target column
   - Consider the base_table_candidates ranking

2. **Define the target**: Specify which column to predict.
   - If target is in the base table, just specify table and column
   - For time-based predictions (churn, conversion), specify label_window_days
   - Use join_key if target comes from a related table

3. **Plan joins**: For each related table with useful features:
   - For one_to_one joins: features come directly
   - For one_to_many joins: define aggregations (sum, count, avg, min, max)
   - Use window_days for time-windowed features (e.g., last 90 days)
   - Create meaningful feature names like "total_spend_90d" or "order_count_30d"

4. **Exclude appropriately**:
   - Exclude log/audit tables that don't add predictive value
   - Exclude columns that would cause leakage (future information)
   - Exclude obvious ID columns from features (but keep them for joins)

5. **Add filters if needed**:
   - Filter out incomplete records
   - Apply date ranges if appropriate

Provide a complete training_dataset_spec and a natural_language_summary explaining your choices."""


# =============================================================================
# VISUALIZATION PROMPTS (visualization_service.py)
# =============================================================================

SYSTEM_ROLE_VIZ_DEVELOPER = "You are an expert data visualization developer. Generate clean, working Python code."

SYSTEM_ROLE_VIZ_SCIENTIST = "You are an expert data scientist who creates insightful, actionable visualizations. You focus on visualizations that directly support ML model development - understanding target variables, feature distributions, correlations, and data quality. Your suggestions are specific and use exact column names."

SYSTEM_ROLE_VIZ_ANALYST = "You are a helpful data analyst explaining visualizations clearly."


def get_visualization_code_prompt(
    user_request: str,
    data_summary: Dict[str, Any],
    file_path: str,
    columns_details: str,
    error_context: str = "",
    prev_context: str = "",
) -> str:
    """Generate prompt for visualization code generation.

    Used by: generate_visualization() in visualization_service.py
    """
    return f"""You are a data visualization expert. Generate Python code to create a visualization based on the user's request.
{error_context}

DATA SUMMARY:
- Rows: {data_summary['row_count']}
- Columns: {data_summary['column_count']}
- Column names: {data_summary['column_names']}

COLUMN DETAILS:
{columns_details}
{prev_context}

USER REQUEST: {user_request}

REQUIREMENTS:
1. Use matplotlib and/or seaborn for visualization
2. The data file path is: {file_path}
3. Read the data using pandas (pd.read_csv, pd.read_excel, etc. based on extension)
4. Save the figure to a BytesIO buffer and return it as base64
5. Use plt.figure(figsize=(10, 6)) for good sizing
6. Include proper labels, title, and legend where appropriate
7. Use a clean, professional style
8. Handle any potential errors gracefully
9. DO NOT use plt.show() - we need to capture the image

YOUR CODE MUST:
- Start with necessary imports
- Read the data from the file
- Create the visualization
- Save to buffer and encode as base64
- Store the result in a variable called 'result' which should be a dict with key 'image_base64'

Example structure:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Read data
df = pd.read_csv("{file_path}")

# Create figure
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# ... visualization code ...

plt.tight_layout()

# Save to buffer
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

result = {{'image_base64': image_base64}}
```

Generate the complete code for the requested visualization."""


def get_default_visualizations_prompt(
    data_summary: Dict[str, Any],
    columns_details: str,
    project_section: str = "",
) -> str:
    """Generate prompt for suggesting default visualizations.

    Used by: generate_default_visualizations() in visualization_service.py
    """
    return f"""You are an expert data analyst specializing in exploratory data analysis and ML preparation.
Suggest 4-5 visualizations that would be MOST VALUABLE for understanding this dataset and preparing for the ML task.
{project_section}

DATA SUMMARY:
- Rows: {data_summary['row_count']:,}
- Columns: {data_summary['column_count']}
- Column names: {data_summary['column_names']}

COLUMN DETAILS:
{columns_details}

REQUIREMENTS FOR GOOD VISUALIZATIONS:

1. **Be Specific and Actionable**
   - Use EXACT column names from the data (case-sensitive)
   - Specify exactly what to plot on x and y axes
   - Be precise about groupings, colors, and aggregations

2. **Prioritize by Value**
   - If a target variable is known, visualizations showing its relationship to features are HIGHEST priority
   - Show distributions of the most important numeric columns
   - For classification: show class balance of the target
   - For regression: show target distribution and correlations

3. **Address Data Quality**
   - If missing values were flagged, suggest a visualization showing missing data patterns
   - If outliers were mentioned, include a box plot to visualize them

4. **Make It Practical**
   - Each "request" field should be a complete, specific instruction
   - Example good request: "Create a histogram of the 'price' column with 30 bins, showing the distribution of house prices"
   - Example bad request: "Show data distribution" (too vague)

5. **Chart Type Selection**
   - Numeric vs Numeric: scatter plot or line plot
   - Numeric distribution: histogram or box plot
   - Categorical counts: bar chart (not pie for many categories)
   - Target vs features: grouped bar charts, box plots by category
   - Correlations: heatmap (only if many numeric columns)

Generate visualizations that will genuinely help understand this data for the stated ML goal."""


def get_visualization_explanation_prompt(
    title: str,
    chart_type: str,
    description: str,
    row_count: int,
    column_count: int,
    column_names: List[str],
) -> str:
    """Generate prompt for explaining a visualization.

    Used by: explain_visualization() in visualization_service.py
    """
    return f"""You are a data analyst explaining visualizations to stakeholders.

VISUALIZATION:
- Title: {title}
- Type: {chart_type}
- Description: {description}

DATA CONTEXT:
- Dataset has {row_count} rows and {column_count} columns
- Columns: {column_names}

Provide a clear, non-technical explanation of:
1. What this visualization shows
2. How to interpret it
3. What insights or patterns to look for
4. Any potential implications for analysis

Keep the explanation concise but informative (2-3 paragraphs)."""

# =============================================================================
# AUTO-IMPROVE PIPELINE PROMPTS
# =============================================================================

SYSTEM_ROLE_IMPROVEMENT_ANALYST = """You are an expert ML engineer analyzing experiment results to identify specific, actionable improvements.

Your goal is to synthesize training logs, performance metrics, feature importances, and AI critique to create a comprehensive improvement plan that will boost model performance in the next iteration.

Focus on:
1. Training issues that caused model failures or skips
2. Feature engineering opportunities based on importance analysis
3. Data quality issues affecting performance
4. Configuration changes that could help (time limits, presets, etc.)
5. Potential data leakage or target variable issues"""


def get_improvement_analysis_prompt(
    experiment_name: str,
    iteration_number: int,
    task_type: str,
    target_column: str,
    primary_metric: str,
    best_score: float,
    training_time_seconds: float,
    num_models_trained: int,
    dataset_shape: str,
    feature_columns: List[str],
    leaderboard_summary: str,
    training_logs: str,
    feature_importances: Dict[str, float],
    critique_json: Optional[Dict[str, Any]] = None,
    previous_improvements: Optional[List[Dict[str, Any]]] = None,
    # New parameters for richer context
    data_statistics: Optional[Dict[str, Any]] = None,
    iteration_history: Optional[List[Dict[str, Any]]] = None,
    error_history: Optional[List[str]] = None,
) -> str:
    """Generate prompt for improvement analysis step.

    This analyzes all available information to identify what needs to be improved.

    Used by: auto_improve pipeline
    """
    # Format feature importances
    importance_lines = []
    if feature_importances:
        sorted_features = sorted(
            feature_importances.items(), key=lambda x: abs(x[1]), reverse=True
        )[:15]
        for feat, imp in sorted_features:
            importance_lines.append(f"  - {feat}: {imp:.4f}")
    importance_text = "\n".join(importance_lines) if importance_lines else "  (not available)"

    # Format previous improvements
    prev_improvements_text = ""
    if previous_improvements:
        prev_lines = []
        for i, imp in enumerate(previous_improvements, 1):
            prev_lines.append(f"  Iteration {i}: {imp.get('summary', 'No summary')}")
        prev_improvements_text = f"\n\n## Previous Improvement Attempts\n" + "\n".join(prev_lines)

    # Format critique if available
    critique_text = ""
    if critique_json:
        critique_text = f"""

## AI Critique of Current Results
- Performance Rating: {critique_json.get('performance_rating', 'unknown')}
- Summary: {critique_json.get('summary', 'No summary')}
- Key Issues: {', '.join(str(i.get('issue', '')) for i in critique_json.get('training_issues', [])[:5])}
- Top Suggestions: {', '.join(s.get('suggestion', '')[:100] for s in critique_json.get('improvement_suggestions', [])[:3])}"""

    # Format data statistics (actual data analysis)
    data_stats_text = ""
    if data_statistics:
        stats_lines = ["## Actual Data Statistics (from loaded dataset)"]
        if "columns" in data_statistics:
            stats_lines.append(f"- Available Columns: {', '.join(data_statistics['columns'][:30])}")
            if len(data_statistics['columns']) > 30:
                stats_lines.append(f"  ... and {len(data_statistics['columns']) - 30} more columns")
        if "row_count" in data_statistics:
            stats_lines.append(f"- Total Rows: {data_statistics['row_count']:,}")
        if "column_stats" in data_statistics:
            stats_lines.append("### Column Statistics:")
            for col, stats in list(data_statistics["column_stats"].items())[:20]:
                dtype = stats.get("dtype", "unknown")
                null_pct = stats.get("null_pct", 0)
                unique = stats.get("unique", "?")
                stats_lines.append(f"  - {col}: {dtype}, {unique} unique, {null_pct:.1f}% missing")
        if "sample_values" in data_statistics:
            stats_lines.append("### Sample Values (first row):")
            for col, val in list(data_statistics["sample_values"].items())[:15]:
                stats_lines.append(f"  - {col}: {str(val)[:50]}")
        data_stats_text = "\n".join(stats_lines)

    # Format complete iteration history
    iteration_history_text = ""
    if iteration_history:
        hist_lines = ["## Complete Iteration History"]
        for hist in iteration_history:
            iter_num = hist.get("iteration", "?")
            score = hist.get("score", 0)
            status = hist.get("status", "unknown")
            error = hist.get("error", "")
            changes = hist.get("changes_made", "")
            hist_lines.append(f"\n### Iteration {iter_num} - {status}")
            hist_lines.append(f"- Score: {score}")
            if changes:
                hist_lines.append(f"- Changes Made: {changes}")
            if error:
                hist_lines.append(f"- Error: {error}")
        iteration_history_text = "\n".join(hist_lines)

    # Format error history
    error_history_text = ""
    if error_history:
        error_lines = ["## Errors from Previous Iterations (LEARN FROM THESE!)"]
        for i, error in enumerate(error_history[-5:], 1):  # Last 5 errors
            error_lines.append(f"  {i}. {error[:500]}")
        error_history_text = "\n".join(error_lines)

    return f"""You are analyzing an ML experiment to identify specific improvements for the next iteration.

## Current Experiment: {experiment_name} (Iteration {iteration_number})

### Configuration
- Task Type: {task_type}
- Target Column: {target_column}
- Primary Metric: {primary_metric}
- Dataset Shape: {dataset_shape}
- Features Used: {len(feature_columns)} features

### Performance
- Best Score: {best_score}
- Training Time: {training_time_seconds:.1f} seconds
- Models Trained: {num_models_trained}

### Model Leaderboard
{leaderboard_summary}

### Feature Importances (Top 15)
{importance_text}

### Training Logs (Key Excerpts)
```
{training_logs[-6000:] if len(training_logs) > 6000 else training_logs}
```
{critique_text}
{prev_improvements_text}
{data_stats_text}
{iteration_history_text}
{error_history_text}

## Your Analysis Tasks

1. **Identify Training Issues**
   - Were models skipped due to time limits?
   - Did models fail to train? Why?
   - Were there memory or resource issues?
   - Any warnings about data quality?

2. **Analyze Feature Performance**
   - Which features are contributing most?
   - Are there low-value features that should be removed?
   - What features are missing that should be engineered?

3. **Assess Data Quality**
   - Signs of data leakage?
   - Missing value patterns?
   - Target variable issues?

4. **Determine Configuration Changes Needed**
   - Does time limit need adjusting?
   - Should presets be changed?
   - Are resource limits too strict?

5. **Review Feature Engineering Failures**
   - Check if any engineered features failed to create
   - Common issues: date columns stored as strings (use .dt accessor, system auto-converts)
   - Missing source columns (check column name spelling/case)
   - Invalid formulas (ensure proper pandas/numpy syntax)
   - If features failed, suggest corrected formulas

6. **Evaluate Validation Strategy**
   - Is the current split appropriate for this data type?
   - For time-series/financial data: temporal split is REQUIRED (data leakage otherwise!)
   - For classification with imbalanced classes: stratified split recommended
   - For data with natural groups (e.g., by user/asset): group-based split may be needed

## Response Format

Provide a JSON response with:
{{
    "current_performance_assessment": {{
        "score": {best_score},
        "rating": "poor|fair|good|excellent",
        "is_baseline_beaten": true|false,
        "main_bottleneck": "feature_quality|data_quality|time_limit|model_selection|data_leakage"
    }},
    "training_issues_identified": [
        {{"issue": "description", "severity": "critical|warning|info", "fix_action": "what to do"}}
    ],
    "feature_analysis": {{
        "high_value_features": ["list of top performing features"],
        "low_value_features": ["features to consider removing"],
        "suspected_leakage_features": ["features that might leak target"],
        "failed_features_to_retry": [
            {{"name": "feature_name", "original_formula": "what failed", "corrected_formula": "fixed version", "fix_reason": "why it failed"}}
        ],
        "missing_features": [
            {{"name": "suggested_feature_name", "formula": "how to compute", "expected_impact": "high|medium|low"}}
        ]
    }},
    "data_quality_issues": [
        {{"issue": "description", "affected_rows_pct": 0.0, "fix_action": "what to do"}}
    ],
    "configuration_changes": {{
        "time_limit": 300,
        "presets": "best_quality",
        "other_changes": {{}}
    }},
    "validation_strategy_assessment": {{
        "current_strategy": "random|temporal|stratified|group|unknown",
        "recommended_strategy": "temporal",  // What strategy SHOULD be used
        "is_appropriate": false,  // Is current strategy correct for this data?
        "data_leakage_risk": "high|medium|low|none",
        "reasoning": "Why this strategy is recommended",
        "group_column": null  // For group-based splits, which column to group by
    }},
    "prioritized_improvements": [
        {{
            "rank": 1,
            "category": "feature_engineering|data_quality|config|target_modification",
            "action": "specific action to take",
            "expected_impact": "high|medium|low",
            "implementation_notes": "details on how to implement"
        }}
    ],
    "improvement_summary": "2-3 sentence summary of what needs to change and why"
}}

Focus on actionable, specific improvements that can be automatically applied in the next iteration."""


def get_improvement_plan_prompt(
    experiment_name: str,
    iteration_number: int,
    task_type: str,
    target_column: str,
    current_features: List[str],
    improvement_analysis: Dict[str, Any],
    current_experiment_plan: Dict[str, Any],
    raw_columns: Optional[List[str]] = None,
    existing_engineered_features: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate prompt for creating an actionable improvement plan.

    Takes the analysis and creates specific changes to apply.

    Used by: auto_improve pipeline
    """
    # Format improvement analysis summary
    prioritized = improvement_analysis.get("prioritized_improvements", [])
    priority_text = "\n".join([
        f"  {i+1}. [{p.get('category', 'unknown')}] {p.get('action', 'No action')} (Impact: {p.get('expected_impact', 'unknown')})"
        for i, p in enumerate(prioritized[:5])
    ])

    feature_analysis = improvement_analysis.get("feature_analysis", {})

    # Format raw columns info
    raw_cols_text = ""
    if raw_columns:
        raw_cols_text = f"\n- Raw Dataset Columns: {', '.join(raw_columns)}"

    # Format existing engineered features
    existing_eng_text = ""
    if existing_engineered_features:
        eng_names = [f.get("output_column", "unknown") for f in existing_engineered_features]
        existing_eng_text = f"\n- Already Engineered Features: {', '.join(eng_names)}"

    return f"""Based on the improvement analysis, create a specific plan for the next iteration.

## Context
- Experiment: {experiment_name}
- Current Iteration: {iteration_number}
- Next Iteration: {iteration_number + 1}
- Task Type: {task_type}
- Target Column: {target_column}

## Current Configuration
- Current Features ({len(current_features)}): {', '.join(current_features[:20])}{'...' if len(current_features) > 20 else ''}{raw_cols_text}{existing_eng_text}
- Current AutoML Config: {current_experiment_plan.get('automl_config', {})}

**IMPORTANT**: When creating new engineered features, you can ONLY use columns from:
1. Raw Dataset Columns listed above
2. Already Engineered Features listed above
Do NOT reference columns that don't exist!

**DATE COLUMN HANDLING**:
- Date columns may be stored as strings - the system auto-converts when using .dt accessor
- Always include date columns in source_columns when using .dt accessor
- Valid formulas: df["date"].dt.dayofweek, df["date"].dt.month, df["date"].dt.year
- For datetime components: .dt.hour, .dt.day, .dt.quarter, .dt.is_weekend (dayofweek >= 5)

## Improvement Analysis Summary
{improvement_analysis.get('improvement_summary', 'No summary available')}

### Prioritized Actions
{priority_text}

### Feature Changes Suggested
- High Value Features to Keep: {feature_analysis.get('high_value_features', [])}
- Low Value Features to Remove: {feature_analysis.get('low_value_features', [])}
- New Features to Engineer: {feature_analysis.get('missing_features', [])}

### Configuration Changes
{improvement_analysis.get('configuration_changes', {})}

## Your Task

Create a specific improvement plan that will be automatically executed. The plan should include:

1. **New Feature List** - Exactly which features to use in the next iteration
2. **Engineered Features** - New features to create with exact formulas
3. **Features to Remove** - Which features to drop
4. **AutoML Configuration** - Updated settings for training
5. **Validation Strategy** - How to split data for validation (critical for time-series!)
6. **Data Filters** - Any filtering to apply

**CRITICAL: Validation Strategy**
For time-series, financial, or temporal data, you MUST use temporal validation (split_strategy: "temporal").
Random splits cause data leakage where future data predicts the past, giving invalid/optimistic metrics.
- "temporal" - Use chronological split (early data trains, later data validates). Required for time-series!
- "random" - Standard random train/test split. Use for non-temporal data only.
- "stratified" - Stratified random split. Use for classification with imbalanced classes.
- "group" - Group-based split by a column. Use when you need to keep related rows together.

## Response Format

Provide a JSON response with:
{{
    "iteration_name": "Iteration {iteration_number + 1} - [brief description of main change]",
    "iteration_description": "Description of what this iteration is testing",
    "feature_changes": {{
        "features_to_keep": ["list of existing features to retain"],
        "features_to_remove": ["list of features to drop"],
        "engineered_features": [
            {{
                "output_column": "new_feature_name",
                "formula": "pandas formula like df['col1'] - df['col2']",
                "source_columns": ["col1", "col2"],
                "description": "what this feature represents"
            }}
        ]
    }},
    "automl_config": {{
        "time_limit": 300,
        "presets": "best_quality",
        "num_bag_folds": 5,
        "num_stack_levels": 1
    }},
    "validation_strategy": {{
        "split_strategy": "temporal",  // "temporal", "random", "stratified", or "group"
        "validation_split": 0.2,       // Fraction of data for validation (0.1-0.3)
        "group_column": null,          // Column name for group-based splits (optional)
        "reasoning": "Why this split strategy is appropriate for this data"
    }},
    "data_filters": {{
        "remove_nulls_in_columns": [],
        "custom_filters": []
    }},
    "expected_improvements": [
        "What improvement 1 should achieve",
        "What improvement 2 should achieve"
    ],
    "success_criteria": {{
        "target_metric_improvement": 0.05,
        "acceptable_training_time": 1800
    }},
    "plan_summary": "1-2 sentence summary of what this iteration will test"
}}

Be specific and realistic. Only include changes that can be automatically applied."""


# =============================================================================
# LAB NOTEBOOK SUMMARY PROMPTS
# =============================================================================

SYSTEM_ROLE_LAB_NOTEBOOK_AGENT = """You are a research lab notebook agent responsible for documenting ML research cycles.

Your job is to create clear, comprehensive summaries of what was accomplished during a research cycle.
Your summaries should be:
- Readable by both technical and non-technical stakeholders
- Structured with clear sections
- Focused on insights and learnings, not just facts
- Forward-looking with proposed next steps

Write in a professional but accessible tone, similar to a well-kept research lab notebook."""


def get_lab_notebook_summary_prompt(
    cycle_number: int,
    cycle_title: Optional[str],
    project_name: str,
    problem_description: str,
    experiments_summary: str,
    best_model_info: str,
    step_outputs_summary: str,
    previous_cycles_context: Optional[str] = None,
) -> str:
    """Generate prompt for creating a lab notebook summary of a research cycle.

    Used by: handle_lab_notebook_summary_step() in agent_executor.py

    Args:
        cycle_number: The sequence number of this research cycle
        cycle_title: Optional title for the cycle
        project_name: Name of the project
        problem_description: What the user is trying to solve
        experiments_summary: Summary of experiments run in this cycle
        best_model_info: Information about the best performing model
        step_outputs_summary: Key outputs from other agent steps (dataset design, critique, etc.)
        previous_cycles_context: Optional summary of previous cycles

    Returns:
        Prompt for generating the lab notebook entry
    """
    previous_context = ""
    if previous_cycles_context:
        previous_context = f"""
## Previous Cycles Context
{previous_cycles_context}
"""

    return f"""## Research Cycle {cycle_number}{' - ' + cycle_title if cycle_title else ''}

**Project**: {project_name}

## Problem Statement
{problem_description}
{previous_context}
## Experiments in This Cycle
{experiments_summary}

## Best Model Performance
{best_model_info}

## Agent Insights from This Cycle
{step_outputs_summary}

---

## Your Task

Create a comprehensive lab notebook entry summarizing this research cycle. Your summary should be in Markdown format with the following sections:

### 1. Title
Create a descriptive title for this cycle that captures the key focus/outcome (e.g., "Cycle 2 - Feature Engineering with Time-Based Splits")

### 2. Problem Restatement
Briefly restate what we're trying to solve, including any refinements from this cycle.

### 3. What Was Attempted
Summarize what changes or approaches were tested in this cycle:
- New features or feature engineering strategies
- Different model configurations
- Validation strategy changes
- Data preprocessing modifications

### 4. Experiments Summary
Provide a concise summary of the experiments run:
- Number of experiments
- Key configurations tested
- Performance comparison

### 5. Current Best Model & Metrics
Document the best performing model:
- Model name/type
- Key performance metrics
- How it compares to previous best (if applicable)

### 6. Key Lessons & Hypotheses
What did we learn from this cycle?
- What worked well?
- What didn't work as expected?
- New hypotheses generated

### 7. Proposed Next Directions
Based on the findings, what should be explored next?
- Specific improvements to try
- New approaches to consider
- Data or feature recommendations

Return your response as JSON:
{{
    "title": "Cycle N - Brief descriptive title",
    "body_markdown": "Full Markdown content with all sections..."
}}

Make the summary actionable and insightful. Focus on learnings that will help guide future iterations."""


# =============================================================================
# ROBUSTNESS & OVERFITTING AUDIT PROMPTS
# =============================================================================

SYSTEM_ROLE_ROBUSTNESS_AUDITOR = """You are an ML robustness and overfitting auditor.

Your job is to critically analyze machine learning experiments to detect:
- Overfitting (training performance much better than validation)
- Data leakage (suspiciously high performance)
- Lack of improvement over trivial baselines
- High variance across cross-validation folds
- Other suspicious patterns that indicate unreliable models

You are skeptical by default. A model that seems "too good" should raise red flags.
Your analysis should be based on the actual metrics provided - never fabricate numbers.

Be thorough and honest in your assessment. False confidence in a flawed model
can lead to poor business decisions. It's better to flag potential issues for
human review than to miss them."""


def get_robustness_audit_prompt(
    project_name: str,
    problem_description: str,
    task_type: str,
    primary_metric: str,
    trials_data: str,
    baseline_info: str,
    cv_data: Optional[str] = None,
) -> str:
    """Generate prompt for robustness and overfitting audit.

    Used by: handle_robustness_audit_step() in agent_executor.py

    Args:
        project_name: Name of the project
        problem_description: What the user is trying to solve
        task_type: Type of ML task (classification, regression, etc.)
        primary_metric: The metric being optimized (e.g., roc_auc, rmse)
        trials_data: Formatted string with trial metrics (train vs val)
        baseline_info: Information about baseline comparisons
        cv_data: Optional cross-validation fold data

    Returns:
        Formatted prompt string
    """
    cv_section = ""
    if cv_data:
        cv_section = f"""
## Cross-Validation Fold Data
{cv_data}
"""

    return f"""## Project Context
**Project**: {project_name}
**Problem**: {problem_description}
**Task Type**: {task_type}
**Primary Metric**: {primary_metric}

## Trial Results (Training vs Validation)
{trials_data}
{cv_section}
## Baseline Comparisons
{baseline_info}

---

## Your Task

Analyze the experiment results for overfitting and robustness issues. Be thorough and critical.

### Analysis Steps:

1. **Train-Validation Gap Analysis**
   - Compare training and validation metrics for each trial
   - A large gap (e.g., >0.1 for AUC, >20% for error metrics) is a red flag
   - Consider the task difficulty when judging gaps

2. **Baseline Comparison**
   - How much does the best model improve over trivial baselines?
   - For classification: majority class baseline, stratified random
   - For regression: mean predictor, median predictor
   - Tiny improvements (<5% relative) are suspicious

3. **Cross-Validation Variance** (if fold data available)
   - High variance across folds suggests instability
   - Look for outlier folds that might indicate data issues

4. **Suspicious Patterns**
   - Near-perfect performance (AUC > 0.98, R² > 0.99) often indicates data leakage
   - All models performing similarly might indicate a ceiling effect
   - Sudden jumps in performance between trials

5. **Overall Risk Assessment**
   - LOW: Model appears robust, reasonable train-val gap, beats baselines comfortably
   - MEDIUM: Some concerns (moderate gap, limited baseline improvement, some variance)
   - HIGH: Serious issues (large overfitting gap, barely beats baseline, very high variance)

### Response Format

Return your analysis as JSON:
{{
    "overfitting_risk": "low" | "medium" | "high",
    "train_val_analysis": {{
        "worst_gap": 0.15,
        "avg_gap": 0.08,
        "interpretation": "Description of train-validation gaps observed"
    }},
    "suspicious_patterns": [
        {{
            "type": "train_val_gap" | "metric_spike" | "baseline_concern" | "cv_variance" | "data_leakage_suspicion",
            "severity": "low" | "medium" | "high",
            "description": "Detailed explanation of the issue"
        }}
    ],
    "baseline_comparison": {{
        "baseline_type": "majority_class" | "mean_predictor" | "random" | "none_available",
        "baseline_metric": 0.65,
        "best_model_metric": 0.72,
        "relative_improvement": 0.108,
        "interpretation": "Assessment of whether improvement is meaningful"
    }},
    "cv_analysis": {{
        "fold_variance": 0.02,
        "interpretation": "Assessment of cross-validation stability"
    }},
    "recommendations": [
        "Specific, actionable recommendations to address issues found"
    ],
    "natural_language_summary": "2-3 sentence summary of the audit findings"
}}

Be honest and specific. Reference actual numbers from the provided data.
If data is missing for certain analyses, note that in your response."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_schema_for_prompt(columns: List[Dict[str, Any]], max_columns: int = 50) -> str:
    """Format schema columns for inclusion in prompts.

    Args:
        columns: List of column dictionaries with 'name', 'dtype', etc.
        max_columns: Maximum number of columns to include

    Returns:
        Formatted string representation of the schema
    """
    lines = []
    for i, col in enumerate(columns[:max_columns]):
        name = col.get("name", "unknown")
        dtype = col.get("dtype", col.get("inferred_type", "unknown"))
        unique = col.get("unique_count", "?")
        null_pct = col.get("null_percentage", 0)

        line = f"- {name} ({dtype}): {unique} unique values"
        if null_pct > 0:
            line += f", {null_pct:.1f}% missing"
        lines.append(line)

    if len(columns) > max_columns:
        lines.append(f"... and {len(columns) - max_columns} more columns")

    return "\n".join(lines)


def format_leaderboard_for_prompt(leaderboard: List[Dict[str, Any]], max_models: int = 10) -> str:
    """Format model leaderboard for inclusion in prompts.

    Args:
        leaderboard: List of model dictionaries
        max_models: Maximum number of models to include

    Returns:
        Formatted string representation of the leaderboard
    """
    lines = []
    for i, model in enumerate(leaderboard[:max_models]):
        name = model.get("model_name", model.get("name", "Unknown"))
        model_type = model.get("model_type", "")
        metrics = model.get("metrics", {})

        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items()
                                if k not in ["training_time_seconds", "num_models_trained"])

        line = f"{i+1}. {name}"
        if model_type:
            line += f" ({model_type})"
        if metrics_str:
            line += f": {metrics_str}"
        lines.append(line)

    if len(leaderboard) > max_models:
        lines.append(f"... and {len(leaderboard) - max_models} more models")

    return "\n".join(lines)

"""Prompts for Auto DS Team agents.

These prompts power the autonomous data science research system,
which iteratively analyzes experiment results and plans new experiments.
"""

from typing import Any, Dict, List, Optional


# =============================================================================
# SYSTEM ROLE DEFINITIONS FOR AUTO DS TEAM
# =============================================================================

SYSTEM_ROLE_CROSS_ANALYST = """You are an expert ML research analyst specializing in meta-analysis of experiments.

Your role is to analyze patterns across multiple ML experiments to extract actionable insights.
You excel at:
- Identifying which features consistently improve model performance
- Finding optimal preprocessing strategies for different data types
- Recognizing model architectures that work well for specific problem types
- Detecting overfitting patterns and suggesting mitigation strategies
- Connecting disparate experimental results into coherent findings

You think rigorously and scientifically, always considering:
- Statistical significance of observed patterns
- Potential confounding factors
- The difference between correlation and causation
- Generalizability of findings
"""

SYSTEM_ROLE_STRATEGY_AGENT = """You are an expert ML research strategist who plans the next steps in autonomous research.

Your role is to synthesize insights from past experiments and plan the next round of research.
You excel at:
- Prioritizing which hypotheses to test next based on expected information gain
- Designing experiments that efficiently explore the hypothesis space
- Balancing exploration (trying new approaches) with exploitation (refining what works)
- Identifying when diminishing returns suggest stopping
- Planning experiments that build on each other logically

You think strategically about:
- Resource allocation (compute budget, time constraints)
- Risk vs reward of different experimental directions
- How to maximize learning from each experiment
- When to declare victory or cut losses
"""

SYSTEM_ROLE_EXPERIMENT_ORCHESTRATOR = """You are an expert ML experiment orchestrator who designs and coordinates experiments.

Your role is to take high-level strategic directions and turn them into concrete experiment designs.
You excel at:
- Translating research hypotheses into specific experiment configurations
- Designing feature engineering that tests specific hypotheses
- Configuring AutoML settings appropriately for the experiment goals
- Ensuring experiments are fair comparisons (controlled variables)
- Creating experiments that provide clear, interpretable results

You are practical and detail-oriented:
- You specify exact configurations, not vague directions
- You consider implementation constraints
- You ensure experiments can actually be executed
- You design for reproducibility
"""


SYSTEM_ROLE_DYNAMIC_PLANNER = """You are an expert ML research scientist who designs experiments dynamically based on live results.

Your role is to analyze real-time experiment results and design the next experiment to maximize learning and performance improvement. Unlike batch planners, you operate mid-iteration with access to fresh results.

You excel at:
- Rapid hypothesis generation based on recent experimental evidence
- Identifying the most informative next experiment to run
- Recognizing when to exploit (refine what works) vs explore (try new approaches)
- Detecting convergence (when further experiments won't help)
- Applying advanced techniques when appropriate:
  * Feature engineering to create more predictive variables
  * Ensemble strategies to combine successful models
  * Ablation studies to identify what really matters

You think adaptively:
- Each experiment should build on what you just learned
- Avoid redundant experiments that won't provide new information
- Focus resources on the most promising directions
- Know when to stop - more experiments aren't always better

You are decisive and efficient:
- Make clear recommendations based on available evidence
- Don't hedge - pick the most promising direction
- When improvements plateau, recommend stopping to avoid wasting compute
"""




# =============================================================================
# FEATURE ENGINEERING GUIDANCE
# =============================================================================

FEATURE_ENGINEERING_GUIDANCE = """
## Feature Engineering Best Practices

### Formula Guidelines
You can write both simple single-line formulas AND complex multi-line transformations.
The executor supports function definitions, conditionals, and multi-statement code.

**Simple formula examples:**
- `df['col1'] / (df['col2'] + 1)`
- `np.log1p(df['price'])`
- `df.groupby('category')['value'].transform('mean')`
- `np.where(df['col'] > 0, df['col'], 0)`

**Complex formula support:**
- Multi-line code with variable assignments
- Function definitions using `def`
- Complex conditional logic

**Important dtype note:**
When using `np.where()` or `np.select()` with string values, use `None` instead of `np.nan` for missing values:
- ✅ `np.where(condition, "CategoryA", None)`
- ❌ `np.where(condition, "CategoryA", np.nan)`  # Causes dtype error

**CRITICAL for categorical features/targets:**
When creating categorical columns with multiple conditions, always include a fallback category for unmatched cases.
Having more than 20% None/NaN values in a target column will cause the experiment to FAIL.
- ✅ `np.where(cond1, "A", np.where(cond2, "B", "Unknown"))`  # Fallback to "Unknown"
- ❌ `np.where(cond1, "A", np.where(cond2, "B", None))`  # Too many None values

Do NOT include import statements - `np`, `pd`, and `df` are already available.

When creating engineered features, consider these high-impact transformations:

### 1. Interaction Features
- **Ratios**: ratio_A_B = feature_A / (feature_B + epsilon)
  Use when relationship between features matters more than absolute values
- **Products**: interaction_AB = feature_A * feature_B
  Captures multiplicative relationships
- **Differences**: diff_AB = feature_A - feature_B
  Highlights relative changes

### 2. Mathematical Transforms
- **Log transform**: log_A = log1p(feature_A)
  For skewed distributions, monetary values, counts
- **Square root**: sqrt_A = sqrt(abs(feature_A))
  Mild de-skewing, variance stabilization
- **Polynomial**: feature_A_squared = feature_A ** 2
  Captures non-linear relationships

### 3. Binning & Discretization
- **Quantile bins**: bin_A = pd.qcut(feature_A, q=5, labels=False)
  Equal-frequency binning
- **Custom thresholds**: high_A = (feature_A > threshold).astype(int)
  Domain-specific thresholds

### 4. Aggregations (if entity/group exists)
- **Group statistics**: mean_by_group = df.groupby('group')['feature'].transform('mean')
- **Deviation from mean**: dev_from_mean = feature - group_mean
- **Rank within group**: rank_in_group = df.groupby('group')['feature'].rank()

### 5. Time-Based (if temporal data)
- **Rolling statistics**: rolling_mean_7d = feature.rolling(7).mean()
- **Lag features**: lag_1 = feature.shift(1)
- **Rate of change**: pct_change = feature.pct_change()

### 6. Text Features (if text columns exist)
- **Length features**: text_length = text.str.len()
- **Word count**: word_count = text.str.split().str.len()

### Feature Engineering Response Format
When suggesting feature engineering, use this format in the config:
{
    "feature_engineering": [
        {
            "name": "ratio_income_age",
            "formula": "income / (age + 1)",
            "description": "Income per year of age - captures earning capacity"
        },
        {
            "name": "log_balance",
            "formula": "np.log1p(balance)",
            "description": "Log-transformed balance for de-skewing"
        }
    ]
}

### Key Principles
1. **Domain Knowledge First**: Create features that make business/domain sense
2. **High Importance Features**: Focus on transforming features the model finds important
3. **Avoid Leakage**: Don't use target information to create features
4. **Keep It Simple**: Start with simple transforms before complex ones
5. **Test Impact**: Each feature should have a hypothesis about why it helps
"""

# =============================================================================
# CROSS-ANALYSIS PROMPTS
# =============================================================================

def get_cross_analysis_prompt(
    experiments_summary: List[Dict[str, Any]],
    existing_insights: List[Dict[str, Any]],
    project_context: Dict[str, Any],
    global_insights: Optional[List[Dict[str, Any]]] = None,
    context_documents: str = "",
    failed_experiments: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate prompt for cross-experiment analysis.

    Args:
        experiments_summary: List of experiment summaries with metrics, features, configs
        existing_insights: Previously discovered insights for this project
        project_context: Project goal, target, problem type
        global_insights: Relevant cross-project insights (if any)
        context_documents: Optional formatted context documents section
        failed_experiments: Failed experiments to learn from
    """
    experiments_str = "\n".join([
        f"""
Experiment: {exp.get('name', 'Unknown')}
- Dataset: {exp.get('dataset_name', 'Unknown')}
- Features Used: {exp.get('features', [])}
- Feature Engineering: {exp.get('feature_engineering', 'None')}
- Best Model: {exp.get('best_model', 'Unknown')}
- Primary Metric ({exp.get('primary_metric', 'unknown')}): {exp.get('score', 'N/A')}
- All Metrics: {exp.get('all_metrics', {})}
- Model Leaderboard: {exp.get('leaderboard', [])}
- Overfitting Indicators: {exp.get('overfitting_risk', 'Unknown')}
- Training Time: {exp.get('training_time_seconds', 'Unknown')} seconds
- Quality Preset: {exp.get('preset', 'Unknown')}
""" for exp in experiments_summary
    ])

    existing_insights_str = "\n".join([
        f"- [{ins.get('confidence', 'low')}] {ins.get('title', 'Unknown')}: {ins.get('description', '')}"
        for ins in existing_insights
    ]) if existing_insights else "No prior insights"

    global_insights_str = ""
    if global_insights:
        global_insights_str = "\n\nRelevant insights from other projects:\n" + "\n".join([
            f"- {ins.get('title', 'Unknown')}: {ins.get('description', '')} (confidence: {ins.get('confidence_score', 0):.2f})"
            for ins in global_insights
        ])

    # Build context section if provided
    context_section = ""
    if context_documents:
        context_section = f"""
{context_documents}
"""

    # Build failed experiments section if provided
    failed_experiments_section = ""
    if failed_experiments:
        failed_lines = ["## ⚠️ Failed Experiments (LEARN FROM THESE)", ""]
        failed_lines.append("The following experiments FAILED. Analyze why they failed and factor this into your insights:")
        failed_lines.append("")
        for exp in failed_experiments:
            failed_lines.append(f"**{exp.get('name', 'Unknown Experiment')}**")
            if exp.get('error_message'):
                failed_lines.append(f"- Error: {exp.get('error_message')}")
            if exp.get('features'):
                failed_lines.append(f"- Features: {exp.get('features')}")
            if exp.get('feature_engineering'):
                failed_lines.append(f"- Feature Engineering: {exp.get('feature_engineering')}")
            if exp.get('dataset_name'):
                failed_lines.append(f"- Dataset: {exp.get('dataset_name')}")
            failed_lines.append("")
        failed_lines.append("**Common failure causes to consider:**")
        failed_lines.append("- Feature leakage (features that won't be available at prediction time)")
        failed_lines.append("- Invalid feature engineering formulas")
        failed_lines.append("- Data type mismatches")
        failed_lines.append("- Missing required columns")
        failed_lines.append("- Memory/resource issues from too many features")
        failed_lines.append("")
        failed_experiments_section = "\n".join(failed_lines)

    return f"""Analyze the following experiments from this research session to extract patterns and insights.

## Project Context
- Goal: {project_context.get('goal', 'Unknown')}
- Target Variable: {project_context.get('target', 'Unknown')}
- Problem Type: {project_context.get('problem_type', 'Unknown')}
- Optimization Direction: {project_context.get('metric_direction', 'Unknown')}
{context_section}
{failed_experiments_section}## Experiments to Analyze
{experiments_str}

## Existing Project Insights
{existing_insights_str}
{global_insights_str}

## Your Task

Analyze these experiments and identify:

1. **Feature Patterns**: Which features or feature engineering approaches consistently help or hurt performance?
   - Are there features that should always be included?
   - Are there features that cause overfitting?
   - What feature engineering transformations were most effective?

2. **Model Patterns**: What model types or configurations work best?
   - Which algorithms consistently rank high?
   - Are there hyperparameter patterns that matter?

3. **Data Processing Patterns**: What preprocessing choices impact results?
   - Encoding strategies for categorical variables
   - Scaling/normalization approaches
   - Missing value handling

4. **Overfitting Patterns**: What configurations lead to overfitting?
   - Feature sets that are too large
   - Models that memorize training data
   - Insufficient regularization

5. **Contradictions**: Are there conflicting results that need investigation?
   - Features that help in one experiment but hurt in another
   - Inconsistent model rankings

6. **Failure Analysis**: If there are failed experiments listed above:
   - What caused each failure? (Parse the error messages)
   - Are there patterns in what fails? (e.g., certain features, engineering formulas)
   - What should be AVOIDED in future experiments to prevent similar failures?
   - Create specific "avoid_patterns" recommendations to prevent repeating mistakes

For each insight, provide:
- A clear, specific title
- Detailed description with evidence
- Confidence level (high/medium/low) based on evidence strength
- Supporting experiment references
- Any contradicting evidence

Focus on insights that are ACTIONABLE for future experiments.
"""


def get_cross_analysis_schema() -> Dict[str, Any]:
    """Schema for cross-analysis response."""
    return {
        "type": "object",
        "properties": {
            "insights": {
                "type": "array",
                "description": "List of discovered insights",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["feature_importance", "model_performance", "preprocessing",
                                     "overfitting_pattern", "hyperparameter", "interaction", "failure_analysis", "other"],
                            "description": "Category of insight"
                        },
                        "title": {
                            "type": "string",
                            "description": "Clear, specific title for the insight"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description with evidence"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Confidence based on evidence strength"
                        },
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Experiment names/IDs that support this insight"
                        },
                        "contradictions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Any contradicting evidence"
                        },
                        "recommendation": {
                            "type": "string",
                            "description": "Specific actionable recommendation"
                        }
                    },
                    "required": ["type", "title", "description", "confidence", "evidence", "recommendation"]
                }
            },
            "summary": {
                "type": "string",
                "description": "Overall summary of findings"
            },
            "best_configuration": {
                "type": "object",
                "description": "Synthesized best configuration based on all evidence",
                "properties": {
                    "recommended_features": {"type": "array", "items": {"type": "string"}},
                    "recommended_feature_engineering": {"type": "array", "items": {"type": "string"}},
                    "recommended_models": {"type": "array", "items": {"type": "string"}},
                    "avoid_features": {"type": "array", "items": {"type": "string"}},
                    "avoid_patterns": {"type": "array", "items": {"type": "string"}}
                }
            },
            "unresolved_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Questions that need more experimentation to answer"
            },
            "failure_analysis": {
                "type": "object",
                "description": "Analysis of failed experiments to prevent repeating mistakes",
                "properties": {
                    "failure_count": {"type": "integer", "description": "Number of failed experiments analyzed"},
                    "common_causes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Common reasons experiments failed"
                    },
                    "problematic_features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Features that caused failures - DO NOT use in future experiments"
                    },
                    "problematic_formulas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Feature engineering formulas that caused failures"
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific recommendations to avoid future failures"
                    }
                }
            }
        },
        "required": ["insights", "summary", "best_configuration", "unresolved_questions"]
    }


# =============================================================================
# STRATEGY AGENT PROMPTS
# =============================================================================

def get_strategy_prompt(
    analysis_results: Dict[str, Any],
    session_status: Dict[str, Any],
    stopping_conditions: Dict[str, Any],
    available_actions: List[str],
    context_documents: str = "",
    leakage_candidates: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate prompt for strategy planning.

    Args:
        analysis_results: Output from cross-analysis
        session_status: Current session state (iteration, best score, etc.)
        stopping_conditions: When to stop (max iterations, threshold, etc.)
        available_actions: What kinds of experiments can be run
        context_documents: Optional formatted context documents section
        leakage_candidates: Optional list of detected leakage candidate features
    """
    insights_str = "\n".join([
        f"- [{ins.get('confidence', 'low')}] {ins.get('title', '')}: {ins.get('recommendation', '')}"
        for ins in analysis_results.get('insights', [])
    ])

    best_config = analysis_results.get('best_configuration', {})
    questions = analysis_results.get('unresolved_questions', [])
    failure_analysis = analysis_results.get('failure_analysis', {})

    # Build context section if provided
    context_section = ""
    if context_documents:
        context_section = f"""
{context_documents}
"""

    # Build leakage warning section if candidates provided
    leakage_section = ""
    if leakage_candidates:
        high_severity = [lc for lc in leakage_candidates if lc.get("severity") == "high"]
        if high_severity:
            leakage_lines = ["## ⚠️ DATA LEAKAGE WARNING", ""]
            leakage_lines.append("**HIGH SEVERITY LEAKAGE FEATURES (DO NOT USE):**")
            for lc in high_severity[:10]:  # Limit to 10 to avoid prompt bloat
                leakage_lines.append(f"- `{lc.get('column')}`: {lc.get('reason', 'Potential leakage')}")
            leakage_lines.append("")
            leakage_lines.append("**IMPORTANT**: Do NOT propose experiments using these features.")
            leakage_lines.append("")
            leakage_section = "\n".join(leakage_lines)

    return f"""Based on the analysis of experiments, plan the next research iteration.
{context_section}{leakage_section}
## Current Session Status
- Iteration: {session_status.get('current_iteration', 0)} / {stopping_conditions.get('max_iterations', 10)}
- Best Score So Far: {session_status.get('best_score', 'None')}
- Iterations Without Improvement: {session_status.get('iterations_without_improvement', 0)}
- Target Threshold: {stopping_conditions.get('accuracy_threshold', 'None')}
- Plateau Limit: {stopping_conditions.get('plateau_iterations', 3)}
- Time Budget: {stopping_conditions.get('time_budget_minutes', 'Unlimited')} minutes

## Analysis Insights
{insights_str}

## Current Best Configuration
- Recommended Features: {best_config.get('recommended_features', [])}
- Recommended Engineering: {best_config.get('recommended_feature_engineering', [])}
- Recommended Models: {best_config.get('recommended_models', [])}
- Features to Avoid: {best_config.get('avoid_features', [])}
- Patterns to Avoid: {best_config.get('avoid_patterns', [])}

## ⚠️ Failure Analysis (CRITICAL - Avoid These Mistakes)
- Failed Experiment Count: {failure_analysis.get('failure_count', 0)}
- Common Failure Causes: {failure_analysis.get('common_causes', ['None identified'])}
- Problematic Features (DO NOT USE): {failure_analysis.get('problematic_features', [])}
- Problematic Formulas (DO NOT USE): {failure_analysis.get('problematic_formulas', [])}
- Recommendations to Prevent Failures: {failure_analysis.get('recommendations', [])}

## Unresolved Questions
""" + "\n".join(['- ' + q for q in questions]) + f"""

## Available Actions
""" + "\n".join(['- ' + a for a in available_actions]) + """

## Your Task

Decide the next steps for this research session:

1. **Should we continue?** Consider:
   - Have we reached the accuracy threshold?
   - Are we hitting diminishing returns (plateau)?
   - Is there still promising unexplored territory?

2. **If continuing, what should we focus on?** Consider:
   - Exploitation: Refine the best configuration further
   - Exploration: Test unresolved questions
   - Hybrid: Do a bit of both

3. **Specific experiment proposals**: For each proposed experiment:
   - What hypothesis does it test?
   - Expected information gain
   - Risk level (could waste resources if it fails)

4. **Priority ranking**: Order experiments by expected value

Be strategic - we have limited iterations. Make each one count.
"""


def get_strategy_schema() -> Dict[str, Any]:
    """Schema for strategy response."""
    return {
        "type": "object",
        "properties": {
            "should_continue": {
                "type": "boolean",
                "description": "Whether to continue with more iterations"
            },
            "stop_reason": {
                "type": "string",
                "description": "If stopping, why"
            },
            "strategy_mode": {
                "type": "string",
                "enum": ["exploit", "explore", "hybrid"],
                "description": "Overall strategy for this iteration"
            },
            "strategy_rationale": {
                "type": "string",
                "description": "Explanation of the chosen strategy"
            },
            "proposed_experiments": {
                "type": "array",
                "description": "Ordered list of proposed experiments",
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis": {
                            "type": "string",
                            "description": "What hypothesis does this test?"
                        },
                        "experiment_type": {
                            "type": "string",
                            "enum": ["refine_features", "new_features", "model_tuning",
                                     "data_augmentation", "ensemble", "ablation", "other"],
                            "description": "Type of experiment"
                        },
                        "description": {
                            "type": "string",
                            "description": "What specifically to do"
                        },
                        "expected_gain": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Expected improvement potential"
                        },
                        "risk": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Risk of wasting resources"
                        },
                        "priority": {
                            "type": "integer",
                            "description": "Priority rank (1 = highest)"
                        },
                        "builds_on": {
                            "type": "string",
                            "description": "Which previous experiment/insight this builds on"
                        }
                    },
                    "required": ["hypothesis", "experiment_type", "description", "expected_gain", "risk", "priority"]
                }
            },
            "datasets_to_create": {
                "type": "array",
                "description": "New dataset specs needed",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "base_on": {"type": "string", "description": "Parent dataset to derive from"},
                        "changes": {"type": "string", "description": "What to change"}
                    }
                }
            },
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key areas to focus on this iteration"
            }
        },
        "required": ["should_continue", "strategy_mode", "strategy_rationale", "proposed_experiments", "focus_areas"]
    }


# =============================================================================
# EXPERIMENT ORCHESTRATOR PROMPTS
# =============================================================================

def get_experiment_design_from_strategy_prompt(
    strategy: Dict[str, Any],
    project_context: Dict[str, Any],
    available_features: List[str],
    current_datasets: List[Dict[str, Any]],
    schema_summary: Dict[str, Any],
    context_documents: str = "",
    leakage_candidates: Optional[List[Dict[str, Any]]] = None,
    failure_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate prompt to design concrete experiments from strategy.

    Args:
        strategy: Output from strategy agent
        project_context: Project goal, target, etc.
        available_features: All available columns
        current_datasets: Existing dataset specs
        schema_summary: Data schema info
        context_documents: Optional formatted context documents section
        leakage_candidates: Optional list of detected leakage candidate features
        failure_analysis: Optional analysis of failed experiments from cross-analysis
    """
    proposed = strategy.get('proposed_experiments', [])
    proposals_str = "\n".join([
        f"""
{i+1}. {p.get('hypothesis', 'Unknown')}
   Type: {p.get('experiment_type', 'other')}
   Description: {p.get('description', '')}
   Expected Gain: {p.get('expected_gain', 'unknown')}
   Builds On: {p.get('builds_on', 'N/A')}
""" for i, p in enumerate(proposed)
    ])

    datasets_str = "\n".join([
        f"- ID: {ds.get('id', 'unknown')} | {ds.get('name', 'Unknown')}: {ds.get('description', '')} (features: {len(ds.get('features', []))})"
        for ds in current_datasets
    ])

    # Build context section if provided
    context_section = ""
    if context_documents:
        context_section = f"""
{context_documents}
"""

    # Build leakage warning section if candidates provided
    leakage_section = ""
    if leakage_candidates:
        high_severity = [lc for lc in leakage_candidates if lc.get("severity") == "high"]
        medium_severity = [lc for lc in leakage_candidates if lc.get("severity") == "medium"]

        if high_severity or medium_severity:
            leakage_lines = ["## ⚠️ DATA LEAKAGE WARNING", ""]
            leakage_lines.append("The following features have been flagged as potential data leakage:")
            leakage_lines.append("")

            if high_severity:
                leakage_lines.append("**HIGH SEVERITY (DO NOT USE):**")
                for lc in high_severity:
                    leakage_lines.append(f"- `{lc.get('column')}`: {lc.get('reason', 'Potential leakage')}")
                leakage_lines.append("")

            if medium_severity:
                leakage_lines.append("**MEDIUM SEVERITY (USE WITH CAUTION):**")
                for lc in medium_severity:
                    leakage_lines.append(f"- `{lc.get('column')}`: {lc.get('reason', 'Potential leakage')}")
                leakage_lines.append("")

            leakage_lines.append("**IMPORTANT**: Do NOT include high-severity features in your experiments.")
            leakage_lines.append("Models using these features may show artificially inflated performance that won't generalize.")
            leakage_lines.append("")

            leakage_section = "\n".join(leakage_lines)

    # Build failure analysis section if provided
    failure_section = ""
    if failure_analysis and failure_analysis.get('failure_count', 0) > 0:
        failure_lines = ["## ⚠️ PREVIOUS FAILURES (LEARN FROM THESE)", ""]
        failure_lines.append(f"**{failure_analysis.get('failure_count', 0)} experiments previously failed.**")
        failure_lines.append("")

        if failure_analysis.get('common_causes'):
            failure_lines.append("**Common Failure Causes:**")
            for cause in failure_analysis.get('common_causes', []):
                failure_lines.append(f"- {cause}")
            failure_lines.append("")

        if failure_analysis.get('problematic_features'):
            failure_lines.append("**Features That Caused Failures (DO NOT USE):**")
            for feat in failure_analysis.get('problematic_features', []):
                failure_lines.append(f"- `{feat}`")
            failure_lines.append("")

        if failure_analysis.get('problematic_formulas'):
            failure_lines.append("**Feature Engineering Formulas That Failed (DO NOT USE):**")
            for formula in failure_analysis.get('problematic_formulas', []):
                failure_lines.append(f"- `{formula}`")
            failure_lines.append("")

        if failure_analysis.get('recommendations'):
            failure_lines.append("**Recommendations:**")
            for rec in failure_analysis.get('recommendations', []):
                failure_lines.append(f"- {rec}")
            failure_lines.append("")

        failure_lines.append("**CRITICAL**: Design experiments that AVOID these failure patterns.")
        failure_lines.append("")
        failure_section = "\n".join(failure_lines)

    return f"""Design concrete experiments based on the research strategy.

## Project Context
- Goal: {project_context.get('goal', 'Unknown')}
- Target Variable: {project_context.get('target', 'Unknown')}
- Problem Type: {project_context.get('problem_type', 'Unknown')}
{context_section}{leakage_section}{failure_section}
{FEATURE_ENGINEERING_GUIDANCE}

## Available Features
{', '.join(available_features[:50])}{'...' if len(available_features) > 50 else ''}

## Schema Summary
{schema_summary}

## Current Dataset Specs
{datasets_str}

## Strategy to Implement
Mode: {strategy.get('strategy_mode', 'unknown')}
Rationale: {strategy.get('strategy_rationale', '')}

Focus Areas:
""" + "\n".join(['- ' + f for f in strategy.get('focus_areas', [])]) + f"""

## Proposed Experiments to Implement
{proposals_str}

## New Datasets to Create (from strategy)
{strategy.get('datasets_to_create', [])}

## Your Task

For each proposed experiment, create a concrete implementation:

1. **Dataset Configuration**: CRITICAL - always specify base_dataset_id:
   - **base_dataset_id**: REQUIRED - use an ID from "Current Dataset Specs" above to inherit data source
   - Which features to include (be specific - list actual column names)
   - Feature engineering to apply (if any)
   - Preprocessing configuration
   - Set create_new=true to derive a new dataset spec from base, or false to use base as-is

2. **Experiment Configuration**:
   - Name and description
   - **Training Time Strategy**: Decide how long to train based on:
     * Exploration experiments: Use shorter times (60-300 seconds) to quickly test hypotheses
     * Exploitation experiments: Use longer times (600-3600 seconds) to fully optimize promising configs
     * High-potential experiments: If previous results show promise, invest more training time
     * Diminishing returns: If similar configs already trained long, no need to repeat
   - AutoML settings (time limit, models to try, presets, etc.)
   - Evaluation strategy
   - How to measure if the hypothesis is confirmed

3. **Dependencies**: If experiments build on each other, specify order

Be SPECIFIC - no vague directions. Provide exact feature lists and configurations.
"""


def get_experiment_design_from_strategy_schema() -> Dict[str, Any]:
    """Schema for experiment design response."""
    return {
        "type": "object",
        "properties": {
            "experiments": {
                "type": "array",
                "description": "Concrete experiment specifications",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "hypothesis": {"type": "string"},
                        "dataset_spec": {
                            "type": "object",
                            "properties": {
                                "create_new": {"type": "boolean", "description": "True to derive new dataset, false to use base as-is"},
                                "base_dataset_id": {"type": "string", "description": "REQUIRED - ID from Current Dataset Specs to inherit data source"},
                                "name": {"type": "string"},
                                "features": {"type": "array", "items": {"type": "string"}, "description": "List of feature column names to use"},
                                "target": {"type": "string", "description": "Target column name"},
                                "feature_engineering": {
                                    "type": "array",
                                    "description": "Feature engineering formulas. CRITICAL: Each formula MUST be a single-line pandas/numpy expression. NO imports, NO function definitions, NO multi-line code.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string", "description": "Name for the new feature column"},
                                            "formula": {"type": "string", "description": "SINGLE-LINE pandas/numpy expression like: df['a'] / df['b'], np.log1p(df['x']), np.where(df['x'] > 0, 1, 0). NO def, NO import, NO multi-line."},
                                            "description": {"type": "string"}
                                        }
                                    }
                                },
                                "preprocessing": {
                                    "type": "object",
                                    "properties": {
                                        "handle_missing": {"type": "string"},
                                        "encoding": {"type": "string"},
                                        "scaling": {"type": "string"}
                                    }
                                }
                            },
                            "required": ["base_dataset_id"]
                        },
                        "automl_config": {
                            "type": "object",
                            "properties": {
                                "time_limit": {
                                    "type": "integer",
                                    "description": "Training time in seconds. Use 60-300 for exploration, 600-1800 for standard, 1800-3600 for high-potential experiments"
                                },
                                "time_rationale": {
                                    "type": "string",
                                    "description": "Explain why this training time was chosen (e.g., 'exploration - quick hypothesis test', 'high potential - previous similar config scored 0.85')"
                                },
                                "presets": {
                                    "type": "string",
                                    "enum": ["medium_quality", "good_quality", "high_quality", "best_quality"],
                                    "description": "Quality preset - use medium for exploration, good/high for promising configs, best for final optimization"
                                },
                                "models_to_try": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific models to try, or empty for auto-selection"
                                },
                                "eval_metric": {"type": "string"},
                                "hyperparameter_tune": {
                                    "type": "boolean",
                                    "description": "Whether to tune hyperparameters - use true for high-potential experiments"
                                },
                                "num_bag_folds": {
                                    "type": "integer",
                                    "description": "Number of bagging folds (0-10). Higher = more robust but slower. Use 0-5 for exploration, 5-10 for exploitation"
                                }
                            },
                            "required": ["time_limit", "time_rationale", "presets"]
                        },
                        "success_criteria": {"type": "string", "description": "How to know if hypothesis is confirmed"},
                        "priority": {"type": "integer"},
                        "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Experiment names this depends on"}
                    },
                    "required": ["name", "description", "hypothesis", "dataset_spec", "automl_config", "success_criteria", "priority"]
                }
            },
            "execution_order": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Order to run experiments (respecting dependencies)"
            },
            "notes": {
                "type": "string",
                "description": "Any additional implementation notes"
            }
        },
        "required": ["experiments", "execution_order"]
    }


# =============================================================================
# DYNAMIC PLANNING PROMPTS
# =============================================================================

def get_dynamic_planning_prompt(
    experiment_history: List[Dict[str, Any]],
    available_datasets: List[Dict[str, Any]],
    available_features: List[str],
    project_context: Dict[str, Any],
    session_progress: Dict[str, Any],
    experiments_to_design: int,
    feature_flags: Dict[str, bool],
    context_documents: str = "",
) -> str:
    """Generate prompt for dynamic mid-iteration experiment planning.

    Args:
        experiment_history: List of completed experiment summaries with scores
        available_datasets: Available datasets for new experiments
        available_features: All available feature columns
        project_context: Project goal, target, problem type
        session_progress: Current iteration progress info
        experiments_to_design: Number of experiments to design
        feature_flags: Which advanced features are enabled
        context_documents: Optional formatted context documents section
    """
    # Format experiment history
    history_str = ""
    if experiment_history:
        for i, exp in enumerate(experiment_history):
            score = exp.get("score", "N/A")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            history_str += f"""
Experiment {i+1}: {exp.get('name', 'Unknown')}
- Hypothesis: {exp.get('hypothesis', 'Not specified')}
- Score: {score_str}
- Config: {exp.get('config', {})}
- Key Results: {exp.get('results', {})}
"""
    else:
        history_str = "No experiments run yet in this iteration."

    # Format available datasets
    datasets_str = "\n".join([
        f"- ID: {ds.get('id')} | {ds.get('name')} (features: {len(ds.get('features', []))})"
        for ds in available_datasets
    ]) if available_datasets else "No datasets available"

    # Format feature flags
    enabled_features = []
    if feature_flags.get("enable_feature_engineering"):
        enabled_features.append("Feature Engineering (create new derived features)")
    if feature_flags.get("enable_ensemble"):
        enabled_features.append("Ensemble Strategies (combine successful models)")
    if feature_flags.get("enable_ablation"):
        enabled_features.append("Ablation Studies (identify what features really matter)")
    if feature_flags.get("enable_diverse_configs"):
        enabled_features.append("Diverse Configs (try varied model configurations)")



    # Add feature engineering guidance if enabled
    feature_eng_section = ""
    if feature_flags.get("enable_feature_engineering"):
        feature_eng_section = FEATURE_ENGINEERING_GUIDANCE

    # Add ablation study guidance if enabled
    ablation_section = ""
    if feature_flags.get("enable_ablation"):
        ablation_section = ABLATION_STUDY_GUIDANCE

    # Format enabled features list
    enabled_str = "\n".join([f"- {f}" for f in enabled_features]) if enabled_features else "None enabled"

    # Build context section if provided
    context_section = ""
    if context_documents:
        context_section = f"""
{context_documents}
"""

    return f"""Based on the experiment results so far, design the next {experiments_to_design} experiment(s) to maximize improvement.
{feature_eng_section}
{ablation_section}
## Project Context
- Goal: {project_context.get('goal', 'Unknown')}
- Target Variable: {project_context.get('target', 'Unknown')}
- Problem Type: {project_context.get('problem_type', 'Unknown')}
{context_section}
## Session Progress
- Current Iteration: {session_progress.get('current_iteration', 1)}
- Best Score So Far: {session_progress.get('best_score', 'N/A')}
- Experiments This Iteration: {session_progress.get('experiments_run_this_iteration', 0)}
- Total Experiments: {session_progress.get('total_experiments_run', 0)}

## Experiment History (This Iteration)
{history_str}

## Available Datasets
{datasets_str}

## Available Features
{', '.join(available_features[:40])}{'...' if len(available_features) > 40 else ''}

## Enabled Advanced Features
{enabled_str}

## Your Task

Analyze the results and design the next experiment(s). Consider:

1. **What patterns emerge from the results?**
   - Which approaches are working? Which aren't?
   - What hypotheses should you test next?

2. **Should you continue or stop?**
   - Is there room for meaningful improvement?
   - Are scores plateauing?
   - Have you tried the most promising directions?

3. **What's the most informative next experiment?**
   - Build on what's working
   - Avoid redundant experiments
   - Use enabled advanced features when appropriate:
     * Feature engineering to create predictive combinations
     * Ensembles to combine top performers
     * Ablation to identify key features

4. **Be specific in your design:**
   - Specify exact features to use
   - Specify exact model configurations
   - Set appropriate training time (longer for promising directions)

5. **CRITICAL - Always specify dataset_spec_id:**
   - You MUST use an ID from "Available Datasets" above
   - The dataset_spec_id field is REQUIRED for every experiment
   - If you don't see any datasets listed, stop and recommend stopping

If improvements have plateaued or you've exhausted promising directions, recommend stopping.
"""


# =============================================================================
# ABLATION STUDY GUIDANCE
# =============================================================================

ABLATION_STUDY_GUIDANCE = """
## Ablation Study Best Practices

Ablation studies help identify which features contribute most to model performance.
By systematically removing features and measuring impact, you can:
- Identify redundant features to remove
- Find harmful features that hurt performance
- Understand feature importance for model interpretation

### When to Use Ablation Studies
1. **After initial good results**: Once you have a working model, understand what drives it
2. **Before production deployment**: Verify that all features are necessary
3. **When suspicious of data leakage**: Test if removing a feature improves generalization
4. **To simplify the model**: Find minimal feature set for similar performance

### Ablation Experiment Design

Use these fields in your config:
- `ablation_target`: Name describing what you are removing (e.g., "age_feature", "all_categorical")
- `drop_columns`: List of column names to exclude from training

### Types of Ablation

1. **Single Feature Ablation**
   - Remove one feature at a time
   - Good for identifying top contributors
   - Example: {"ablation_target": "income", "drop_columns": ["income"]}

2. **Feature Group Ablation**
   - Remove related features together
   - Test if a category of features matters
   - Example: {"ablation_target": "demographic_features", "drop_columns": ["age", "gender", "location"]}

3. **Engineered Feature Ablation**
   - Test if your feature engineering helps
   - Example: {"ablation_target": "all_engineered", "drop_columns": ["ratio_income_age", "log_balance"]}

### Interpreting Ablation Results

| Score Change | Interpretation |
|-------------|----------------|
| Large drop  | Feature is critical - keep it |
| Small drop  | Feature is useful but not critical |
| No change   | Feature may be redundant |
| Improvement | Feature may be harmful - consider removing |
"""



def get_dynamic_planning_schema() -> Dict[str, Any]:
    """Schema for dynamic planning response."""
    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Your analysis of the current results and rationale for next steps"
            },
            "should_stop": {
                "type": "boolean",
                "description": "True if further experiments unlikely to improve results"
            },
            "stop_reason": {
                "type": "string",
                "description": "Why stopping is recommended (if should_stop is true)"
            },
            "experiments": {
                "type": "array",
                "description": "Experiment specifications to run next",
                "items": {
                    "type": "object",
                    "properties": {
                        "dataset_spec_id": {
                            "type": "string",
                            "description": "ID of dataset to use from available datasets"
                        },
                        "dataset_name": {
                            "type": "string",
                            "description": "Name for this experiment's dataset"
                        },
                        "hypothesis": {
                            "type": "string",
                            "description": "What you're testing with this experiment"
                        },
                        "variant": {
                            "type": "integer",
                            "description": "Variant number for tracking"
                        },
                        "config": {
                            "type": "object",
                            "description": "Experiment configuration",
                            "properties": {
                                "features": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Feature columns to use"
                                },
                                "feature_engineering": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string", "description": "Name for the new feature column"},
                                            "formula": {"type": "string", "description": "SINGLE-LINE pandas/numpy expression. NO imports, NO def, NO multi-line. Examples: df['a']/df['b'], np.log1p(df['x'])"},
                                            "description": {"type": "string"}
                                        }
                                    },
                                    "description": "New features to create. CRITICAL: formulas must be single-line pandas/numpy expressions only."
                                },
                                "time_limit": {
                                    "type": "integer",
                                    "description": "Training time in seconds"
                                },
                                "presets": {
                                    "type": "string",
                                    "enum": ["medium_quality", "good_quality", "high_quality", "best_quality"]
                                },
                                "ensemble_strategy": {
                                    "type": "string",
                                    "description": "If using ensemble, describe the strategy"
                                },
                                "ablation_target": {
                                    "type": "string",
                                    "description": "If ablation study, what feature/component to test"
                                },
                                "drop_columns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Column names to exclude from training for ablation"
                                }
                            }
                        },
                        "expected_outcome": {
                            "type": "string",
                            "description": "What result would confirm the hypothesis"
                        },
                        "builds_on": {
                            "type": "string",
                            "description": "Which previous experiment this builds on"
                        }
                    },
                    "required": ["dataset_spec_id", "hypothesis", "config"]
                }
            }
        },
        "required": ["reasoning", "should_stop", "experiments"]
    }

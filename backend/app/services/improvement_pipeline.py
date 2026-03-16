"""Enhanced improvement pipeline that uses full agent pipeline with iteration feedback.

This module provides a comprehensive improvement pipeline that:
1. Gathers complete iteration history (metrics, errors, logs, critiques)
2. Loads and analyzes actual data
3. Runs specialized agents with iteration context
4. Creates improved dataset designs and experiments

Unlike the simple 2-step auto-improve, this uses the full agent pipeline
to thoroughly analyze and improve based on all previous learnings.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session

from app.models import (
    AgentStep,
    AgentStepStatus,
    AgentStepType,
    AgentRun,
    AgentRunStatus,
    DataSource,
    Project,
    Experiment,
    Trial,
    ModelVersion,
)
from app.models.experiment import ExperimentStatus
from app.models.dataset_spec import DatasetSpec
from app.models.research_cycle import LabNotebookEntry, LabNotebookAuthorType
from app.services.agent_executor import (
    StepLogger,
    safe_get,
    _format_schema_for_prompt,
    build_schema_summary,
)
from app.services.llm_client import BaseLLMClient, get_llm_client
from app.services.prompts import (
    SYSTEM_ROLE_DATA_SCIENTIST,
    SYSTEM_ROLE_ML_ANALYST,
    SYSTEM_ROLE_IMPROVEMENT_ANALYST,
    get_data_analysis_prompt,
)
from app.tasks.automl import load_dataset_from_spec

logger = logging.getLogger(__name__)


# ============================================
# Helper Functions
# ============================================

def _collect_feature_engineering_feedback(experiments: List[Experiment]) -> Dict[str, Any]:
    """Collect feature engineering feedback from all experiments in a chain.

    This gathers information about which features succeeded vs failed across
    all iterations, helping the LLM avoid repeating failed approaches.

    Returns:
        Dict with:
        - successful_features: List of features that were successfully created
        - failed_features: List of features that failed with error messages
        - available_columns: Columns available in the most recent dataset
    """
    successful = []
    failed = []
    available_columns = []

    for exp in experiments:
        ctx = exp.improvement_context_json or {}

        # Get feature engineering results if recorded
        fe_results = ctx.get("feature_engineering_result", {})
        if fe_results:
            for feat in fe_results.get("successful_features", []):
                if feat not in successful:
                    successful.append({
                        **feat,
                        "iteration": exp.iteration_number,
                    })

            for feat in fe_results.get("failed_features", []):
                failed.append({
                    **feat,
                    "iteration": exp.iteration_number,
                })

            # Keep the most recent available columns
            cols = fe_results.get("available_columns", [])
            if cols:
                available_columns = cols

        # Also check spec_json for engineered features
        if exp.dataset_spec:
            spec_json = exp.dataset_spec.spec_json or {}
            eng_feats = spec_json.get("engineered_features", [])
            for feat in eng_feats:
                name = feat.get("output_column")
                if name and not any(s.get("feature") == name for s in successful):
                    successful.append({
                        "feature": name,
                        "formula": feat.get("formula", ""),
                        "description": feat.get("description", ""),
                        "iteration": exp.iteration_number,
                    })

    return {
        "successful_features": successful,
        "failed_features": failed,
        "available_columns": available_columns,
        "summary": f"{len(successful)} successful, {len(failed)} failed features across iterations"
    }


# ============================================
# Iteration Context Builder
# ============================================

def gather_iteration_context(
    db: Session,
    experiment: Experiment,
    primary_metric: str = "roc_auc",
) -> Dict[str, Any]:
    """Gather complete context from all iterations in an experiment chain.

    This traverses the entire experiment chain from root to current,
    collecting metrics, errors, logs, critiques, and improvement attempts.

    Args:
        db: Database session
        experiment: The current experiment to improve
        primary_metric: The metric to track across iterations

    Returns:
        Dictionary with:
        - iteration_history: List of all iterations with metrics and status
        - error_history: List of errors encountered
        - improvement_attempts: List of changes tried and their outcomes
        - best_score: Best score achieved across all iterations
        - worst_score: Worst score achieved
        - score_trend: Whether scores are improving, declining, or flat
        - total_training_time: Total time spent training
        - logs_summary: Key log excerpts
    """
    # Find root experiment
    root_experiment = experiment
    while root_experiment.parent_experiment_id:
        parent = db.query(Experiment).filter(
            Experiment.id == root_experiment.parent_experiment_id
        ).first()
        if parent:
            root_experiment = parent
        else:
            break

    # Traverse from root to current
    experiments_in_chain = [root_experiment]
    current_exp = root_experiment
    while True:
        child = db.query(Experiment).filter(
            Experiment.parent_experiment_id == current_exp.id
        ).first()
        if child:
            experiments_in_chain.append(child)
            current_exp = child
        else:
            break

    # Build context
    iteration_history = []
    error_history = []
    improvement_attempts = []
    all_scores = []
    total_training_time = 0.0
    all_logs = []
    all_critiques = []

    for exp in experiments_in_chain:
        trial = db.query(Trial).filter(
            Trial.experiment_id == exp.id
        ).order_by(Trial.created_at.desc()).first()

        metrics = (trial.metrics_json if trial and trial.metrics_json else {}) or {}
        score = metrics.get(primary_metric, metrics.get("score_val", 0))
        training_time = metrics.get("training_time_seconds", 0)

        if score:
            all_scores.append(float(score))
        total_training_time += float(training_time) if training_time else 0

        # Build iteration entry
        entry = {
            "iteration": exp.iteration_number,
            "experiment_id": str(exp.id),
            "name": exp.name,
            "score": float(score) if score else 0.0,
            "status": exp.status.value if exp.status else "unknown",
            "training_time_seconds": training_time,
            "num_models_trained": metrics.get("num_models_trained", 0),
        }

        # Capture changes made
        if exp.improvement_context_json:
            changes = exp.improvement_context_json.get("summary", "")
            if changes:
                entry["changes_made"] = changes
                improvement_attempts.append({
                    "iteration": exp.iteration_number,
                    "changes": changes,
                    "result_score": float(score) if score else 0.0,
                    "success": exp.status == ExperimentStatus.COMPLETED,
                })

        # Capture errors
        if exp.status == ExperimentStatus.FAILED and exp.error_message:
            entry["error"] = exp.error_message
            error_history.append({
                "iteration": exp.iteration_number,
                "error": exp.error_message,
            })

        # Capture logs
        if trial and trial.training_logs:
            logs = trial.training_logs
            # Extract key excerpts (warnings, errors, important info)
            log_lines = logs.split('\n')
            important_lines = [
                l for l in log_lines
                if any(kw in l.lower() for kw in ['error', 'warning', 'failed', 'skipped', 'best model'])
            ][:10]
            if important_lines:
                all_logs.append({
                    "iteration": exp.iteration_number,
                    "key_logs": important_lines,
                })

        # Capture critique
        if trial and trial.critique_json:
            all_critiques.append({
                "iteration": exp.iteration_number,
                "critique": trial.critique_json,
            })

        # Get leaderboard
        if trial and trial.leaderboard_json:
            leaderboard = trial.leaderboard_json[:5]
            entry["top_models"] = [
                {"model": m.get("model", "?"), "score": m.get("score_val", 0)}
                for m in leaderboard
            ]

        # Get feature importances
        model_version = db.query(ModelVersion).filter(
            ModelVersion.experiment_id == exp.id
        ).order_by(ModelVersion.created_at.desc()).first()

        if model_version and model_version.feature_importances_json:
            # Top 10 features
            importances = model_version.feature_importances_json
            sorted_feats = sorted(
                importances.items(), key=lambda x: abs(x[1]), reverse=True
            )[:10]
            entry["top_features"] = [
                {"feature": f, "importance": imp} for f, imp in sorted_feats
            ]

        iteration_history.append(entry)

    # Calculate score trend
    score_trend = "flat"
    if len(all_scores) >= 2:
        first_half_avg = sum(all_scores[:len(all_scores)//2]) / (len(all_scores)//2) if len(all_scores) >= 2 else 0
        second_half_avg = sum(all_scores[len(all_scores)//2:]) / (len(all_scores) - len(all_scores)//2) if len(all_scores) >= 2 else 0
        if second_half_avg > first_half_avg * 1.01:
            score_trend = "improving"
        elif second_half_avg < first_half_avg * 0.99:
            score_trend = "declining"

    # Collect feature engineering failures from all iterations
    feature_engineering_feedback = _collect_feature_engineering_feedback(experiments_in_chain)

    # Check for overfitting via holdout scores
    overfitting_report = None
    try:
        from app.services.holdout_validator import get_overfitting_report
        overfitting_report = get_overfitting_report(db, experiment)
    except Exception as e:
        logger.warning(f"Could not get overfitting report: {e}")

    return {
        "iteration_history": iteration_history,
        "error_history": error_history,
        "improvement_attempts": improvement_attempts,
        "best_score": max(all_scores) if all_scores else 0.0,
        "worst_score": min(all_scores) if all_scores else 0.0,
        "current_score": all_scores[-1] if all_scores else 0.0,
        "score_trend": score_trend,
        "total_iterations": len(experiments_in_chain),
        "total_training_time_seconds": total_training_time,
        "logs_summary": all_logs,
        "critiques": all_critiques,
        "feature_engineering_feedback": feature_engineering_feedback,
        "overfitting_report": overfitting_report,
    }


def load_data_statistics(
    db: Session,
    dataset_spec: DatasetSpec,
) -> Optional[Dict[str, Any]]:
    """Load actual data and compute statistics.

    Args:
        db: Database session
        dataset_spec: The dataset specification

    Returns:
        Dictionary with data statistics or None if loading fails
    """
    try:
        logger.info("Loading actual dataset for analysis...")
        df = load_dataset_from_spec(db, dataset_spec)

        # Build statistics
        stats = {
            "columns": list(df.columns),
            "row_count": len(df),
            "column_count": len(df.columns),
            "column_stats": {},
            "sample_values": {},
            "dtypes": {},
            "correlations": {},
        }

        # Column statistics
        for col in df.columns[:30]:
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_pct": float(df[col].isna().sum() / len(df) * 100),
                "unique": int(df[col].nunique()),
            }

            # Numeric stats
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                })

            stats["column_stats"][col] = col_stats
            stats["dtypes"][col] = str(df[col].dtype)

        # Sample values
        if len(df) > 0:
            first_row = df.iloc[0]
            for col in df.columns[:15]:
                stats["sample_values"][col] = str(first_row[col])[:50]

        logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return stats

    except Exception as e:
        logger.warning(f"Could not load dataset for analysis: {e}")
        return None


# ============================================
# Enhanced Improvement Step Handlers
# ============================================

async def handle_iteration_context_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle the iteration context gathering step.

    This step gathers all context from previous iterations including:
    - Metrics from all iterations
    - Errors and failures
    - Improvement attempts and their outcomes
    - Critiques and suggestions
    - Actual data statistics

    Input JSON should contain:
    - experiment_id: UUID of the experiment to improve
    - primary_metric: The metric to track

    Returns:
        Dict with complete iteration context
    """
    input_data = step.input_json or {}

    experiment_id = input_data.get("experiment_id")
    if not experiment_id:
        raise ValueError("Missing 'experiment_id' in input_json")

    primary_metric = input_data.get("primary_metric", "roc_auc")

    step_logger.info(f"Gathering iteration context for experiment {experiment_id}")

    # Load experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")

    # Get dataset spec
    dataset_spec = experiment.dataset_spec
    if not dataset_spec:
        raise ValueError("Experiment has no dataset spec")

    step_logger.thought("Traversing experiment chain to gather history...")

    # Gather iteration context
    context = gather_iteration_context(db, experiment, primary_metric)

    step_logger.info(f"Found {context['total_iterations']} iterations in chain")
    step_logger.info(f"Score trend: {context['score_trend']}")
    step_logger.info(f"Best score: {context['best_score']:.4f}")
    step_logger.info(f"Current score: {context['current_score']:.4f}")

    if context['error_history']:
        step_logger.warning(f"Found {len(context['error_history'])} errors in history")
        for err in context['error_history'][:3]:
            step_logger.thought(f"  Iteration {err['iteration']}: {err['error'][:100]}...")

    # Load actual data statistics
    step_logger.info("Loading actual dataset for analysis...")
    data_statistics = load_data_statistics(db, dataset_spec)

    if data_statistics:
        step_logger.info(f"Dataset: {data_statistics['row_count']} rows, {data_statistics['column_count']} columns")
        context["data_statistics"] = data_statistics
    else:
        step_logger.warning("Could not load dataset - will proceed with metadata only")

    # Add dataset spec info
    context["dataset_spec"] = {
        "id": str(dataset_spec.id),
        "name": dataset_spec.name,
        "target_column": dataset_spec.target_column,
        "feature_columns": dataset_spec.feature_columns or [],
        "spec_json": dataset_spec.spec_json or {},
    }

    # Add project info
    project = experiment.project
    if project:
        context["project"] = {
            "id": str(project.id),
            "name": project.name,
            "task_type": project.task_type.value if project.task_type else "unknown",
            "description": project.description or "",
        }

    step_logger.summary(
        f"Gathered context from {context['total_iterations']} iterations. "
        f"Score: {context['current_score']:.4f} (trend: {context['score_trend']}). "
        f"{len(context['error_history'])} errors found."
    )

    return context


async def handle_improvement_data_analysis_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle data re-analysis with iteration feedback.

    This step analyzes the data with full knowledge of what has worked
    and what hasn't in previous iterations.

    Input JSON should contain iteration_context from previous step.

    Returns:
        Dict with enhanced data analysis and recommendations
    """
    from pydantic import BaseModel, Field
    from typing import List as ListType

    input_data = step.input_json or {}
    iteration_context = input_data.get("iteration_context", {})

    if not iteration_context:
        raise ValueError("Missing iteration_context - run ITERATION_CONTEXT step first")

    step_logger.info("Analyzing data with iteration feedback...")

    # Get data statistics
    data_stats = iteration_context.get("data_statistics", {})
    dataset_spec = iteration_context.get("dataset_spec", {})
    project = iteration_context.get("project", {})

    # Format iteration history for prompt
    iteration_history = iteration_context.get("iteration_history", [])
    history_text = ""
    if iteration_history:
        history_lines = []
        for hist in iteration_history:
            line = f"  Iteration {hist['iteration']}: score={hist.get('score', 0):.4f}, status={hist.get('status', '?')}"
            if hist.get('changes_made'):
                line += f"\n    Changes: {hist['changes_made'][:100]}"
            if hist.get('error'):
                line += f"\n    Error: {hist['error'][:100]}"
            history_lines.append(line)
        history_text = "\n".join(history_lines)

    # Format error history
    errors = iteration_context.get("error_history", [])
    error_text = ""
    if errors:
        error_lines = [f"  Iteration {e['iteration']}: {e['error'][:200]}" for e in errors[-5:]]
        error_text = "\n".join(error_lines)

    # Format data columns
    columns = data_stats.get("columns", [])
    column_stats = data_stats.get("column_stats", {})
    columns_text = ""
    if column_stats:
        col_lines = []
        for col, stats in list(column_stats.items())[:20]:
            line = f"  {col}: {stats.get('dtype', '?')}, {stats.get('unique', '?')} unique, {stats.get('null_pct', 0):.1f}% missing"
            col_lines.append(line)
        columns_text = "\n".join(col_lines)

    # Get improvement attempts
    improvements = iteration_context.get("improvement_attempts", [])
    improvements_text = ""
    if improvements:
        imp_lines = []
        for imp in improvements:
            result = "succeeded" if imp.get('success') else "failed"
            imp_lines.append(f"  Iteration {imp['iteration']}: {imp['changes'][:100]} - {result}")
        improvements_text = "\n".join(imp_lines)

    # Format feature engineering feedback - CRITICAL for avoiding repeated mistakes
    fe_feedback = iteration_context.get("feature_engineering_feedback", {})
    fe_failed_text = ""
    fe_succeeded_text = ""
    if fe_feedback:
        failed_feats = fe_feedback.get("failed_features", [])
        if failed_feats:
            fe_failed_lines = []
            for f in failed_feats[-10:]:  # Last 10 failures
                fe_failed_lines.append(f"  ✗ {f.get('feature', '?')}: {f.get('error', 'Unknown error')[:100]}")
            fe_failed_text = "\n".join(fe_failed_lines)

        succeeded_feats = fe_feedback.get("successful_features", [])
        if succeeded_feats:
            fe_succeeded_lines = []
            for f in succeeded_feats[-10:]:
                fe_succeeded_lines.append(f"  ✓ {f.get('feature', '?')}: {f.get('formula', 'N/A')[:80]}")
            fe_succeeded_text = "\n".join(fe_succeeded_lines)

    # Format overfitting report
    overfitting = iteration_context.get("overfitting_report", {})
    overfitting_text = ""
    if overfitting:
        overfitting_text = f"""
## Overfitting Analysis (IMPORTANT!)
- Holdout Score Trend: {overfitting.get('trend', 'unknown')}
- Best Holdout Score: {overfitting.get('best_score', 0):.4f} (iteration {overfitting.get('best_iteration', '?')})
- Current Holdout Score: {overfitting.get('current_score', 0):.4f}
- Recommendation: {overfitting.get('recommendation', 'continue')}
- Analysis: {overfitting.get('message', 'N/A')}
"""

    class ImprovedDataAnalysis(BaseModel):
        key_insights: ListType[str] = Field(
            description="Key insights from analyzing data with iteration feedback"
        )
        failed_approaches: ListType[str] = Field(
            description="Approaches that were tried and failed - DO NOT REPEAT THESE"
        )
        successful_approaches: ListType[str] = Field(
            description="Approaches that showed improvement - BUILD ON THESE"
        )
        untried_opportunities: ListType[str] = Field(
            description="Feature engineering or data improvements NOT yet tried"
        )
        data_quality_issues: ListType[str] = Field(
            description="Data quality issues that may be limiting performance"
        )
        recommended_features: ListType[Dict[str, str]] = Field(
            description="Specific new features to engineer with formulas"
        )
        features_to_remove: ListType[str] = Field(
            description="Features that should be removed (low value or problematic)"
        )
        target_analysis: str = Field(
            description="Analysis of the target variable and any issues"
        )
        bottleneck_diagnosis: str = Field(
            description="Main bottleneck limiting model performance"
        )
        recommended_focus: str = Field(
            description="What the next iteration should focus on"
        )

    prompt = f"""You are analyzing data for an ML improvement iteration. You have access to the COMPLETE history
of what has been tried before, what worked, and what failed. USE THIS INFORMATION to avoid repeating mistakes
and to build on successful approaches.

## Project Goal
{project.get('description', 'Not specified')}
Task Type: {project.get('task_type', 'unknown')}
Target Column: {dataset_spec.get('target_column', 'unknown')}

## Current Performance
- Current Score: {iteration_context.get('current_score', 0):.4f}
- Best Score Ever: {iteration_context.get('best_score', 0):.4f}
- Score Trend: {iteration_context.get('score_trend', 'unknown')}
- Total Iterations: {iteration_context.get('total_iterations', 0)}
{overfitting_text}

## Iteration History
{history_text if history_text else "(No history available)"}

## Errors Encountered (AVOID THESE!)
{error_text if error_text else "(No errors)"}

## Previous Improvement Attempts
{improvements_text if improvements_text else "(No recorded improvements)"}

## Feature Engineering History (CRITICAL - DON'T REPEAT FAILURES!)
FAILED Features (these formulas didn't work - DO NOT SUGGEST SIMILAR):
{fe_failed_text if fe_failed_text else "(None recorded)"}

SUCCESSFUL Features (these worked - you can build on these):
{fe_succeeded_text if fe_succeeded_text else "(None recorded)"}

## Dataset Information
- Rows: {data_stats.get('row_count', 0):,}
- Columns: {data_stats.get('column_count', 0)}

## Column Details
{columns_text if columns_text else "(No column details)"}

## Available Raw Columns (USE ONLY THESE IN FORMULAS!)
{', '.join(columns[:30]) if columns else "(Unknown)"}

## Instructions
1. Analyze what has been tried and what the results were
2. Identify patterns in what works vs what fails
3. Find UNTRIED approaches that might help
4. Suggest specific, actionable improvements
5. DO NOT suggest any feature that already failed - check the failed features list!
6. Only use columns that exist in the Available Raw Columns list
7. Focus on the bottleneck limiting performance
8. If overfitting is detected, suggest simpler models or regularization

Provide your analysis with specific feature engineering formulas that use ONLY the columns listed above."""

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_DATA_SCIENTIST},
        {"role": "user", "content": prompt},
    ]

    step_logger.info("Consulting LLM for iteration-aware data analysis...")
    response = await llm_client.chat_json(messages, ImprovedDataAnalysis)

    # Log key findings
    step_logger.thought(f"Bottleneck: {response.get('bottleneck_diagnosis', 'unknown')}")
    step_logger.thought(f"Recommended focus: {response.get('recommended_focus', 'unknown')}")

    untried = response.get('untried_opportunities', [])
    if untried:
        step_logger.info(f"Found {len(untried)} untried opportunities")
        for opp in untried[:3]:
            step_logger.thought(f"  - {opp[:80]}")

    failed = response.get('failed_approaches', [])
    if failed:
        step_logger.warning(f"Identified {len(failed)} failed approaches to AVOID")

    step_logger.summary(
        f"Data analysis complete. Focus: {response.get('recommended_focus', 'optimization')[:100]}. "
        f"Found {len(untried)} untried opportunities."
    )

    return {
        **response,
        "iteration_context": iteration_context,  # Pass through for next steps
    }


async def handle_improvement_dataset_design_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle dataset redesign with iteration feedback.

    This step creates a new dataset design based on:
    - What features worked in previous iterations
    - What feature engineering failed
    - Untried opportunities identified by analysis

    Returns:
        Dict with new feature list and engineering steps
    """
    from pydantic import BaseModel, Field
    from typing import List as ListType

    input_data = step.input_json or {}
    data_analysis = input_data.get("data_analysis", {})
    iteration_context = input_data.get("iteration_context", {})

    if not iteration_context:
        raise ValueError("Missing iteration_context")

    step_logger.info("Designing improved dataset based on iteration feedback...")

    dataset_spec = iteration_context.get("dataset_spec", {})
    data_stats = iteration_context.get("data_statistics", {})

    # Get current features and engineered features
    current_features = dataset_spec.get("feature_columns", [])
    spec_json = dataset_spec.get("spec_json", {})
    existing_engineered = spec_json.get("engineered_features", [])

    # Get raw columns
    raw_columns = data_stats.get("columns", [])

    # Format recommendations from analysis
    recommended_features = data_analysis.get("recommended_features", [])
    features_to_remove = data_analysis.get("features_to_remove", [])
    untried = data_analysis.get("untried_opportunities", [])

    class DatasetDesign(BaseModel):
        features_to_keep: ListType[str] = Field(
            description="Existing features to retain"
        )
        features_to_drop: ListType[str] = Field(
            description="Features to remove from the dataset"
        )
        new_engineered_features: ListType[Dict[str, Any]] = Field(
            description="New features to create with output_column, formula, source_columns, description"
        )
        data_filters: ListType[Dict[str, Any]] = Field(
            description="Data filters to apply (column, operator, value)"
        )
        rationale: str = Field(
            description="Explanation of why these changes should improve results"
        )
        expected_impact: str = Field(
            description="Expected impact on model performance"
        )

    prompt = f"""You are designing an improved dataset for the next ML training iteration.
You have complete knowledge of what has been tried before and what worked.

## Current Dataset
- Target Column: {dataset_spec.get('target_column', 'unknown')}
- Current Features: {', '.join(current_features[:20])}{'...' if len(current_features) > 20 else ''}
- Existing Engineered Features: {len(existing_engineered)} features

## Available Raw Columns (you can ONLY use these in formulas!)
{', '.join(raw_columns[:40])}

## Analysis Recommendations
Recommended New Features:
{recommended_features}

Features to Remove:
{features_to_remove}

Untried Opportunities:
{untried}

## Guidelines
1. Only create features using columns that ACTUALLY EXIST (listed above)
2. Use pandas/numpy syntax for formulas: df["column"].operation()
3. Common patterns:
   - Rolling averages: df["col"].rolling(window=7).mean()
   - Differences: df["col1"] - df["col2"]
   - Ratios: df["col1"] / df["col2"]
   - Lag features: df["col"].shift(1)
   - Time features: df["date_col"].dt.dayofweek
   - Log transform: np.log1p(df["col"])

4. DO NOT create features that:
   - Reference columns that don't exist
   - Were tried before and failed
   - Could cause data leakage

Create specific, implementable features that address the identified bottlenecks."""

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_ML_ANALYST},
        {"role": "user", "content": prompt},
    ]

    step_logger.info("Consulting LLM for dataset design...")
    response = await llm_client.chat_json(messages, DatasetDesign)

    new_features = response.get("new_engineered_features", [])
    to_drop = response.get("features_to_drop", [])

    step_logger.info(f"Designed {len(new_features)} new engineered features")
    for feat in new_features[:5]:
        if isinstance(feat, dict):
            step_logger.thought(f"  {feat.get('output_column', '?')}: {feat.get('formula', '?')[:60]}")

    if to_drop:
        step_logger.info(f"Dropping {len(to_drop)} features")

    step_logger.thought(f"Rationale: {response.get('rationale', 'Not specified')[:200]}")

    step_logger.summary(
        f"Dataset design complete. +{len(new_features)} new features, -{len(to_drop)} removed. "
        f"Expected: {response.get('expected_impact', 'improved performance')[:50]}"
    )

    return {
        **response,
        "existing_engineered_features": existing_engineered,
        "iteration_context": iteration_context,
    }


async def handle_improvement_experiment_design_step(
    db: Session,
    step: AgentStep,
    step_logger: StepLogger,
    llm_client: BaseLLMClient,
) -> Dict[str, Any]:
    """Handle experiment redesign with iteration feedback.

    This step designs the training configuration based on:
    - What AutoML configs worked before
    - Time budget and resource constraints
    - Model types that performed well/poorly

    Returns:
        Dict with new experiment configuration
    """
    from pydantic import BaseModel, Field
    from typing import List as ListType

    input_data = step.input_json or {}
    dataset_design = input_data.get("dataset_design", {})
    iteration_context = input_data.get("iteration_context", {})

    if not iteration_context:
        raise ValueError("Missing iteration_context")

    step_logger.info("Designing improved experiment configuration...")

    project = iteration_context.get("project", {})
    iteration_history = iteration_context.get("iteration_history", [])

    # Analyze what AutoML configs have been used
    configs_used = []
    for hist in iteration_history:
        if hist.get("top_models"):
            configs_used.append({
                "iteration": hist["iteration"],
                "score": hist.get("score", 0),
                "top_models": hist["top_models"][:3],
            })

    class ValidationStrategy(BaseModel):
        split_strategy: str = Field(
            description="Split strategy: 'temporal', 'random', 'stratified', or 'group'"
        )
        validation_split: float = Field(
            default=0.2, description="Fraction of data for validation (0.1-0.3)"
        )
        group_column: Optional[str] = Field(
            default=None, description="Column name for group-based splits"
        )
        reasoning: str = Field(
            description="Why this split strategy is appropriate for this data"
        )

    class ExperimentDesign(BaseModel):
        iteration_name: str = Field(
            description="Name for this iteration"
        )
        iteration_description: str = Field(
            description="Description of what this iteration is testing"
        )
        automl_config: Dict[str, Any] = Field(
            description="AutoML configuration with time_limit, presets, etc."
        )
        validation_strategy: Optional[ValidationStrategy] = Field(
            default=None, description="Validation split strategy for train/test split"
        )
        expected_improvements: ListType[str] = Field(
            description="What improvements are expected"
        )
        success_criteria: Dict[str, Any] = Field(
            description="Criteria for considering this iteration successful"
        )
        training_strategy: str = Field(
            description="Strategy for training (aggressive, conservative, balanced)"
        )

    prompt = f"""You are designing an improved ML training configuration for the next iteration.

## Project
Task Type: {project.get('task_type', 'unknown')}
Goal: {project.get('description', 'Not specified')[:200]}

## Performance History
Current Score: {iteration_context.get('current_score', 0):.4f}
Best Score: {iteration_context.get('best_score', 0):.4f}
Score Trend: {iteration_context.get('score_trend', 'unknown')}
Iterations So Far: {iteration_context.get('total_iterations', 0)}
Total Training Time: {iteration_context.get('total_training_time_seconds', 0):.0f} seconds

## Previous Configs and Results
{configs_used[:5] if configs_used else "No history available"}

## Dataset Changes for This Iteration
New Features: {len(dataset_design.get('new_engineered_features', []))}
Removed Features: {len(dataset_design.get('features_to_drop', []))}
Rationale: {dataset_design.get('rationale', 'Not specified')[:200]}

## Guidelines
1. If score is improving, continue with similar config
2. If score is flat/declining, try different approach
3. Balance quality vs training time
4. Consider:
   - time_limit: 120-600 seconds typically
   - presets: "best_quality", "high_quality", "medium_quality"
   - num_bag_folds: 0-10 (higher = better but slower)
   - num_stack_levels: 0-3 (higher = better but slower)

## Validation Strategy (CRITICAL - REQUIRED)
You MUST specify a validation_strategy to prevent data leakage:
- split_strategy: Choose one of:
  - "temporal": For time-series or time-dependent data. Sort by time, use most recent data for validation.
  - "stratified": For classification with imbalanced classes. Preserves class distribution.
  - "random": For independent observations with no time dependence.
  - "group": When observations are grouped (e.g., multiple samples per user). Keeps groups together.
- validation_split: Fraction for validation (typically 0.2)
- group_column: Column name if using group split (null otherwise)
- reasoning: Explain why this strategy is appropriate

IMPORTANT: Re-evaluate the validation strategy each iteration. If you notice patterns suggesting
time dependence (date columns, sequential IDs, etc.), switch to temporal split!

Design a configuration that addresses the current bottleneck while being realistic."""

    messages = [
        {"role": "system", "content": SYSTEM_ROLE_ML_ANALYST},
        {"role": "user", "content": prompt},
    ]

    step_logger.info("Consulting LLM for experiment design...")
    response = await llm_client.chat_json(messages, ExperimentDesign)

    automl_config = response.get("automl_config", {})

    step_logger.info(f"Iteration: {response.get('iteration_name', 'Unknown')}")
    step_logger.thought(f"Strategy: {response.get('training_strategy', 'balanced')}")
    step_logger.thought(f"Config: time_limit={automl_config.get('time_limit', 300)}, presets={automl_config.get('presets', 'best_quality')}")

    # Log validation strategy (handle both dict and string responses)
    validation_strategy = response.get("validation_strategy")
    if validation_strategy:
        if isinstance(validation_strategy, dict):
            step_logger.thought(f"Validation: {validation_strategy.get('split_strategy', 'default')} split - {validation_strategy.get('reasoning', '')[:60]}")
        elif isinstance(validation_strategy, str):
            step_logger.thought(f"Validation strategy: {validation_strategy[:80]}")

    expected = response.get("expected_improvements", [])
    if expected:
        step_logger.info("Expected improvements:")
        for imp in expected[:3]:
            step_logger.thought(f"  - {imp[:80]}")

    step_logger.summary(
        f"Experiment design complete: {response.get('iteration_name', 'Next Iteration')}. "
        f"Strategy: {response.get('training_strategy', 'balanced')}."
    )

    return {
        **response,
        "dataset_design": dataset_design,
        "iteration_context": iteration_context,
    }


# ============================================
# Improvement Pipeline Definition
# ============================================

IMPROVE_PIPELINE_STEPS: List[AgentStepType] = [
    AgentStepType.ITERATION_CONTEXT,
    AgentStepType.IMPROVEMENT_DATA_ANALYSIS,
    AgentStepType.IMPROVEMENT_DATASET_DESIGN,
    AgentStepType.IMPROVEMENT_EXPERIMENT_DESIGN,
]

IMPROVE_STEP_HANDLERS = {
    AgentStepType.ITERATION_CONTEXT: handle_iteration_context_step,
    AgentStepType.IMPROVEMENT_DATA_ANALYSIS: handle_improvement_data_analysis_step,
    AgentStepType.IMPROVEMENT_DATASET_DESIGN: handle_improvement_dataset_design_step,
    AgentStepType.IMPROVEMENT_EXPERIMENT_DESIGN: handle_improvement_experiment_design_step,
}


def create_improve_pipeline(
    db: Session,
    experiment_id: UUID,
    primary_metric: str = "roc_auc",
) -> AgentRun:
    """Create an enhanced improvement pipeline for an experiment.

    Args:
        db: Database session
        experiment_id: UUID of the experiment to improve
        primary_metric: The metric to optimize

    Returns:
        The created AgentRun with steps
    """
    # Verify experiment exists and is completed
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")

    if experiment.status != ExperimentStatus.COMPLETED:
        raise ValueError(f"Experiment must be completed. Status: {experiment.status.value}")

    # Create the agent run
    agent_run = AgentRun(
        project_id=experiment.project_id,
        experiment_id=experiment_id,
        name=f"Enhanced Improvement Pipeline - Iteration {experiment.iteration_number + 1}",
        description=f"Full agent pipeline improvement for {experiment.name}",
        status=AgentRunStatus.PENDING,
        config_json={
            "experiment_id": str(experiment_id),
            "primary_metric": primary_metric,
            "iteration_number": experiment.iteration_number,
        },
    )
    db.add(agent_run)
    db.commit()
    db.refresh(agent_run)

    # Create steps
    for step_type in IMPROVE_PIPELINE_STEPS:
        if step_type == AgentStepType.ITERATION_CONTEXT:
            input_json = {
                "experiment_id": str(experiment_id),
                "primary_metric": primary_metric,
            }
        else:
            # Other steps receive context from accumulated outputs
            input_json = {}

        step = AgentStep(
            agent_run_id=agent_run.id,
            step_type=step_type,
            status=AgentStepStatus.PENDING,
            input_json=input_json,
        )
        db.add(step)

    db.commit()
    db.refresh(agent_run)

    return agent_run


async def run_improve_pipeline(
    db: Session,
    agent_run_id: UUID,
    llm_client: Optional[BaseLLMClient] = None,
) -> AgentRun:
    """Run the enhanced improvement pipeline.

    Args:
        db: Database session
        agent_run_id: UUID of the agent run
        llm_client: Optional LLM client

    Returns:
        The completed AgentRun
    """
    if not llm_client:
        llm_client = get_llm_client()

    agent_run = db.query(AgentRun).filter(AgentRun.id == agent_run_id).first()
    if not agent_run:
        raise ValueError(f"Agent run not found: {agent_run_id}")

    # Mark as running
    agent_run.status = AgentRunStatus.RUNNING
    db.commit()

    try:
        # Get steps in the correct pipeline order (by step_type, not created_at)
        # created_at can have identical timestamps causing random order
        step_order = {step_type: idx for idx, step_type in enumerate(IMPROVE_PIPELINE_STEPS)}
        steps = sorted(
            agent_run.steps,
            key=lambda s: step_order.get(s.step_type, 999)
        )

        accumulated_output = {}

        for step in steps:
            # Update step input with accumulated output
            if step.step_type != AgentStepType.ITERATION_CONTEXT:
                # Merge accumulated output into input
                step.input_json = {
                    **(step.input_json or {}),
                    "iteration_context": accumulated_output.get("iteration_context", {}),
                    "data_analysis": accumulated_output.get("data_analysis", {}),
                    "dataset_design": accumulated_output.get("dataset_design", {}),
                }

            # Mark step as running
            step.status = AgentStepStatus.RUNNING
            step.started_at = datetime.utcnow()
            db.commit()

            # Get handler
            handler = IMPROVE_STEP_HANDLERS.get(step.step_type)
            if not handler:
                raise ValueError(f"No handler for step type: {step.step_type}")

            # Create step logger
            step_logger = StepLogger(db, step.id)

            # Run the step
            try:
                output = await handler(db, step, step_logger, llm_client)

                step.status = AgentStepStatus.COMPLETED
                step.output_json = output
                step.finished_at = datetime.utcnow()
                db.commit()

                # Accumulate output based on step type
                if step.step_type == AgentStepType.ITERATION_CONTEXT:
                    accumulated_output["iteration_context"] = output
                elif step.step_type == AgentStepType.IMPROVEMENT_DATA_ANALYSIS:
                    accumulated_output["data_analysis"] = output
                elif step.step_type == AgentStepType.IMPROVEMENT_DATASET_DESIGN:
                    accumulated_output["dataset_design"] = output
                elif step.step_type == AgentStepType.IMPROVEMENT_EXPERIMENT_DESIGN:
                    accumulated_output["experiment_design"] = output

            except Exception as e:
                step.status = AgentStepStatus.FAILED
                step.error_message = str(e)
                step.finished_at = datetime.utcnow()
                db.commit()

                # PARTIAL RESULT RECOVERY: Save accumulated output even though this step failed
                # This allows users to see what was accomplished before the failure
                agent_run.result_json = {
                    **accumulated_output,
                    "_partial": True,
                    "_failed_step": step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type),
                    "_error": str(e),
                }
                db.commit()

                logger.warning(
                    f"Improvement pipeline step {step.step_type} failed. "
                    f"Partial results saved ({len(accumulated_output)} successful steps). Error: {e}"
                )
                raise

        # Mark run as completed
        agent_run.status = AgentRunStatus.COMPLETED
        agent_run.result_json = accumulated_output
        db.commit()

        return agent_run

    except Exception as e:
        # Mark run as failed but preserve any partial results already saved
        agent_run.status = AgentRunStatus.FAILED
        agent_run.error_message = str(e)
        # Don't overwrite result_json if it was already set with partial results
        if not agent_run.result_json:
            agent_run.result_json = {"_error": str(e), "_partial": True}
        db.commit()
        raise


async def run_full_improve_pipeline(
    db: Session,
    experiment_id: UUID,
    primary_metric: str = "roc_auc",
    llm_client: Optional[BaseLLMClient] = None,
) -> AgentRun:
    """Create and run the full enhanced improvement pipeline.

    This is the main entry point for the enhanced improvement pipeline.

    Args:
        db: Database session
        experiment_id: UUID of the experiment to improve
        primary_metric: The metric to optimize
        llm_client: Optional LLM client

    Returns:
        The completed AgentRun with all results
    """
    # Create the pipeline
    agent_run = create_improve_pipeline(db, experiment_id, primary_metric)

    # Run it
    completed_run = await run_improve_pipeline(db, agent_run.id, llm_client)

    # Create a lab notebook entry with the agent's reasoning
    if completed_run.status == AgentRunStatus.COMPLETED:
        try:
            create_improvement_notebook_entry(db, completed_run, experiment_id)
        except Exception as e:
            logger.warning(f"Failed to create lab notebook entry: {e}")

    return completed_run


def create_improvement_notebook_entry(
    db: Session,
    agent_run: AgentRun,
    experiment_id: UUID,
) -> LabNotebookEntry:
    """Create a lab notebook entry summarizing the improvement pipeline results.

    This captures the agent's reasoning and analysis in a human-readable format
    that appears in the Project History view.

    Args:
        db: Database session
        agent_run: The completed agent run
        experiment_id: UUID of the experiment being improved

    Returns:
        The created LabNotebookEntry
    """
    # Get the experiment to find project_id
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")

    result = agent_run.result_json or {}
    iteration_context = result.get("iteration_context", {})
    data_analysis = result.get("data_analysis", {})
    dataset_design = result.get("dataset_design", {})
    experiment_design = result.get("experiment_design", {})

    # Build title
    iteration_num = iteration_context.get("total_iterations", 0) + 1
    current_score = iteration_context.get("current_score", 0)
    title = f"Iteration {iteration_num} Analysis - Score: {current_score:.4f}"

    # Build markdown body with agent reasoning
    body_parts = []
    body_parts.append(f"# Improvement Pipeline Analysis - Iteration {iteration_num}\n")

    # Score context
    if iteration_context:
        body_parts.append("## Performance Context\n")
        body_parts.append(f"- **Current Score**: {current_score:.4f}")
        body_parts.append(f"- **Best Score**: {iteration_context.get('best_score', 0):.4f}")
        body_parts.append(f"- **Score Trend**: {iteration_context.get('score_trend', 'unknown')}")
        body_parts.append(f"- **Total Iterations**: {iteration_context.get('total_iterations', 0)}\n")

    # Data analysis insights
    if data_analysis:
        body_parts.append("## Data Analysis Insights\n")
        if data_analysis.get("key_observations"):
            body_parts.append("### Key Observations")
            for obs in data_analysis.get("key_observations", [])[:5]:
                body_parts.append(f"- {obs}")
            body_parts.append("")
        if data_analysis.get("data_quality_issues"):
            body_parts.append("### Data Quality Issues")
            for issue in data_analysis.get("data_quality_issues", [])[:5]:
                body_parts.append(f"- {issue}")
            body_parts.append("")
        if data_analysis.get("improvement_priorities"):
            body_parts.append("### Improvement Priorities")
            for priority in data_analysis.get("improvement_priorities", [])[:5]:
                body_parts.append(f"- {priority}")
            body_parts.append("")

    # Dataset changes
    if dataset_design:
        body_parts.append("## Dataset Redesign\n")
        if dataset_design.get("features_to_drop"):
            body_parts.append("### Features Removed")
            for f in dataset_design.get("features_to_drop", [])[:10]:
                body_parts.append(f"- `{f}`")
            body_parts.append("")
        if dataset_design.get("new_engineered_features"):
            body_parts.append("### New Engineered Features")
            for feat in dataset_design.get("new_engineered_features", [])[:5]:
                if isinstance(feat, dict):
                    body_parts.append(f"- **{feat.get('output_column', 'Unknown')}**: {feat.get('description', '')}")
                else:
                    body_parts.append(f"- {feat}")
            body_parts.append("")
        if dataset_design.get("reasoning"):
            body_parts.append(f"**Reasoning**: {dataset_design.get('reasoning')[:500]}\n")

    # Experiment design
    if experiment_design:
        body_parts.append("## Experiment Design\n")
        if experiment_design.get("iteration_name"):
            body_parts.append(f"**Iteration Name**: {experiment_design.get('iteration_name')}")
        if experiment_design.get("iteration_description"):
            body_parts.append(f"\n**Description**: {experiment_design.get('iteration_description')[:300]}")
        if experiment_design.get("expected_improvements"):
            body_parts.append("\n### Expected Improvements")
            for imp in experiment_design.get("expected_improvements", [])[:5]:
                body_parts.append(f"- {imp}")
            body_parts.append("")
        if experiment_design.get("training_strategy"):
            body_parts.append(f"**Training Strategy**: {experiment_design.get('training_strategy')}\n")

    body_markdown = "\n".join(body_parts)

    # Create the notebook entry
    entry = LabNotebookEntry(
        project_id=experiment.project_id,
        research_cycle_id=None,  # Not linked to a specific research cycle
        agent_step_id=None,
        author_type=LabNotebookAuthorType.AGENT,
        title=title,
        body_markdown=body_markdown,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    logger.info(f"Created lab notebook entry {entry.id} for improvement pipeline")
    return entry

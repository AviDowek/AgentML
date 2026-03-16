"""Task Context Builder Service (Prompt 7).

This module provides a unified task context that all core agents share and consume.
It assembles information from projects, dataset specs, experiments, baselines,
robustness audits, and leakage detection into a standard context object.

The TaskContext enables consistent reasoning across:
- Time-based data (stocks, churn, price prediction, sequences)
- Non-time-based tabular data (cross-sectional classification/regression)
- Any other problem types

Usage:
    from app.services.task_context import build_task_context

    context = build_task_context(
        db=db,
        project_id=project_id,
        dataset_spec_id=dataset_spec_id,
        experiment_id=experiment_id,
    )
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import (
    Project,
    DataSource,
    Experiment,
    Trial,
    ModelVersion,
)
from app.models.dataset_spec import DatasetSpec
from app.models.research_cycle import (
    ResearchCycle,
    CycleExperiment,
    LabNotebookEntry,
)

logger = logging.getLogger(__name__)


def build_task_context(
    db: Session,
    project_id: str,
    dataset_spec_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    research_cycle_id: Optional[str] = None,
    include_leakage_candidates: bool = True,
    include_past_cycles: bool = True,
    max_experiments: int = 5,
    max_past_cycles: int = 3,
) -> Dict[str, Any]:
    """Build a unified task context for agent consumption.

    This function assembles context from various database entities into a
    standardized dictionary that all agents can use for consistent reasoning.

    Args:
        db: Database session
        project_id: UUID of the project
        dataset_spec_id: Optional UUID of a specific dataset spec
        experiment_id: Optional UUID of a specific experiment
        research_cycle_id: Optional UUID of a research cycle
        include_leakage_candidates: Whether to include leakage candidate info
        include_past_cycles: Whether to include past research cycles summary
        max_experiments: Maximum number of experiments to include
        max_past_cycles: Maximum number of past cycles to summarize

    Returns:
        Dict containing the unified task context with sections:
        - project: Basic project information
        - dataset_spec: Dataset specification details (if available)
        - data_profile_summary: Data characteristics summary
        - latest_experiments: Recent experiment results
        - baselines: Baseline model metrics
        - label_shuffle: Label shuffle test results
        - robustness: Robustness audit summary
        - leakage_candidates: Potential leakage features
        - past_cycles_summary: Previous research cycle summaries
    """
    context: Dict[str, Any] = {
        "project": None,
        "dataset_spec": None,
        "data_profile_summary": None,
        "latest_experiments": [],
        "baselines": {"available": False},
        "label_shuffle": {"available": False},
        "robustness": None,
        "leakage_candidates": [],
        "past_cycles_summary": [],
    }

    # 1. Load project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        logger.warning(f"Project not found: {project_id}")
        return context

    context["project"] = _build_project_context(project)

    # 2. Load dataset spec (specific or latest for project)
    dataset_spec = None
    if dataset_spec_id:
        dataset_spec = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
    elif project.dataset_specs:
        # Get the most recent dataset spec
        dataset_spec = max(project.dataset_specs, key=lambda ds: ds.created_at or datetime.min)

    if dataset_spec:
        context["dataset_spec"] = _build_dataset_spec_context(dataset_spec)
        context["data_profile_summary"] = _build_data_profile_summary(db, dataset_spec, project)

    # 3. Load experiments
    experiments = _get_relevant_experiments(
        db, project_id, experiment_id, research_cycle_id, max_experiments
    )
    context["latest_experiments"] = [_build_experiment_context(exp) for exp in experiments]

    # 4. Extract baselines and label shuffle from experiments
    baselines, label_shuffle = _extract_baselines_and_shuffle(experiments)
    context["baselines"] = baselines
    context["label_shuffle"] = label_shuffle

    # 5. Build robustness summary from experiments
    context["robustness"] = _build_robustness_summary(experiments)

    # 6. Get leakage candidates (from dataset spec or experiments)
    if include_leakage_candidates:
        context["leakage_candidates"] = _get_leakage_candidates(db, dataset_spec, experiments)

    # 7. Get past research cycles summary
    if include_past_cycles and research_cycle_id:
        context["past_cycles_summary"] = _get_past_cycles_summary(
            db, project_id, research_cycle_id, max_past_cycles
        )

    return context


def _build_project_context(project: Project) -> Dict[str, Any]:
    """Build project section of context."""
    return {
        "id": str(project.id),
        "name": project.name,
        "description": project.description,
        "task_type": project.task_type,
        "created_at": project.created_at.isoformat() if project.created_at else None,
    }


def _build_dataset_spec_context(dataset_spec: DatasetSpec) -> Dict[str, Any]:
    """Build dataset spec section of context."""
    # Extract split strategy from spec_json (not a separate column)
    split_strategy = None
    spec_json = dataset_spec.spec_json or {}
    if isinstance(spec_json, dict):
        split_strategy = spec_json.get("split_strategy")

    return {
        "id": str(dataset_spec.id),
        "name": dataset_spec.name,
        "target_column": dataset_spec.target_column,
        "feature_columns": dataset_spec.feature_columns or [],
        "is_time_based": dataset_spec.is_time_based or False,
        "time_column": dataset_spec.time_column,
        "entity_id_column": dataset_spec.entity_id_column,
        "prediction_horizon": getattr(dataset_spec, 'prediction_horizon', None),
        "split_strategy": split_strategy,
        "preprocessing_strategy": spec_json.get("preprocessing_strategy"),
        "filters": dataset_spec.filters_json,
    }


def _build_data_profile_summary(
    db: Session,
    dataset_spec: DatasetSpec,
    project: Project,
) -> Dict[str, Any]:
    """Build data profile summary from dataset spec and related data sources."""
    summary = {
        "row_count": None,
        "column_count": None,
        "class_balance": None,
        "missingness_summary": None,
        "feature_count": len(dataset_spec.feature_columns or []),
    }

    # Try to get row/column counts from data source
    # DatasetSpec uses data_sources_json (list of source configs) not data_source_id
    data_source_id = None
    if dataset_spec.data_sources_json:
        # data_sources_json is a list of source configs, get the first one
        sources = dataset_spec.data_sources_json
        if isinstance(sources, list) and len(sources) > 0:
            first_source = sources[0]
            if isinstance(first_source, dict):
                data_source_id = first_source.get("data_source_id") or first_source.get("id")
            elif isinstance(first_source, str):
                data_source_id = first_source
        elif isinstance(sources, dict):
            # Single source object
            data_source_id = sources.get("data_source_id") or sources.get("id")

    if data_source_id:
        data_source = db.query(DataSource).filter(
            DataSource.id == data_source_id
        ).first()
        if data_source and data_source.schema_summary:
            schema = data_source.schema_summary
            summary["row_count"] = schema.get("row_count")
            summary["column_count"] = schema.get("column_count")

            # Extract class balance if available
            if "class_balance" in schema:
                summary["class_balance"] = schema["class_balance"]
            elif "target_distribution" in schema:
                summary["class_balance"] = schema["target_distribution"]

            # Extract missingness info
            if "missing_summary" in schema:
                summary["missingness_summary"] = schema["missing_summary"]
            elif "columns" in schema:
                # Calculate from column info
                cols_with_nulls = [
                    c for c in schema.get("columns", [])
                    if c.get("null_count", 0) > 0
                ]
                summary["missingness_summary"] = {
                    "columns_with_nulls": len(cols_with_nulls),
                    "total_columns": len(schema.get("columns", [])),
                }

    return summary


def _get_relevant_experiments(
    db: Session,
    project_id: str,
    experiment_id: Optional[str],
    research_cycle_id: Optional[str],
    max_experiments: int,
) -> List[Experiment]:
    """Get relevant experiments for context."""
    if experiment_id:
        # Single specific experiment
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        return [exp] if exp else []

    if research_cycle_id:
        # Experiments from research cycle
        cycle_experiments = (
            db.query(CycleExperiment)
            .filter(CycleExperiment.research_cycle_id == research_cycle_id)
            .all()
        )
        exp_ids = [ce.experiment_id for ce in cycle_experiments]
        if exp_ids:
            experiments = (
                db.query(Experiment)
                .filter(Experiment.id.in_(exp_ids))
                .order_by(Experiment.created_at.desc())
                .limit(max_experiments)
                .all()
            )
            return experiments

    # Get latest experiments for project
    experiments = (
        db.query(Experiment)
        .filter(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc())
        .limit(max_experiments)
        .all()
    )
    return experiments


def _build_experiment_context(experiment: Experiment) -> Dict[str, Any]:
    """Build experiment section of context."""
    # Get best trial metrics
    best_metric_value = None
    best_trial = None
    split_strategy = None

    if experiment.trials:
        for trial in experiment.trials:
            if trial.metrics_json:
                # Try to get primary metric
                metric_val = trial.metrics_json.get(experiment.primary_metric)
                if metric_val is not None:
                    if best_metric_value is None or metric_val > best_metric_value:
                        best_metric_value = metric_val
                        best_trial = trial

            # Get split strategy from trial
            if trial.data_split_strategy and not split_strategy:
                split_strategy = trial.data_split_strategy

    return {
        "id": str(experiment.id),
        "name": experiment.name,
        "status": experiment.status.value if hasattr(experiment.status, 'value') else str(experiment.status),
        "primary_metric_name": experiment.primary_metric,
        "primary_metric_value": best_metric_value,
        "split_strategy": split_strategy,
        "trial_count": len(experiment.trials) if experiment.trials else 0,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
    }


def _extract_baselines_and_shuffle(
    experiments: List[Experiment],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract baseline metrics and label shuffle results from experiments."""
    baselines: Dict[str, Any] = {"available": False}
    label_shuffle: Dict[str, Any] = {"available": False}

    for exp in experiments:
        if not exp.trials:
            continue

        for trial in exp.trials:
            if not trial.baseline_metrics_json:
                continue

            bm = trial.baseline_metrics_json

            # Extract majority class baseline
            if "majority_class" in bm and bm["majority_class"]:
                baselines["majority_class"] = bm["majority_class"]
                baselines["available"] = True

            # Extract simple model baseline (logistic or ridge)
            if "simple_logistic" in bm and bm["simple_logistic"]:
                baselines["simple_model"] = bm["simple_logistic"]
                baselines["available"] = True
            elif "simple_ridge" in bm and bm["simple_ridge"]:
                baselines["regression_baseline"] = bm["simple_ridge"]
                baselines["available"] = True

            # Extract mean predictor for regression
            if "mean_predictor" in bm and bm["mean_predictor"]:
                baselines["mean_predictor"] = bm["mean_predictor"]
                baselines["available"] = True

            # Extract label shuffle results
            if "label_shuffle" in bm and bm["label_shuffle"]:
                shuffle = bm["label_shuffle"]
                label_shuffle = {
                    "available": True,
                    "roc_auc": shuffle.get("shuffled_roc_auc"),
                    "accuracy": shuffle.get("shuffled_accuracy"),
                    "r2": shuffle.get("shuffled_r2"),
                    "leakage_detected": shuffle.get("leakage_detected", False),
                    "warning": shuffle.get("warning"),
                }

            # Found baselines, can stop searching
            if baselines["available"]:
                break

        if baselines["available"]:
            break

    return baselines, label_shuffle


def _build_robustness_summary(experiments: List[Experiment]) -> Optional[Dict[str, Any]]:
    """Build robustness summary from experiment trials."""
    robustness: Dict[str, Any] = {
        "overfitting_risk": None,
        "leakage_suspected": None,
        "time_split_suspicious": None,
        "too_good_to_be_true": None,
        "warnings": [],
        "risk_adjusted_score_best_model": None,
    }

    has_data = False

    for exp in experiments:
        if not exp.trials:
            continue

        for trial in exp.trials:
            metrics = trial.metrics_json or {}

            # Get risk-adjusted score
            if "risk_adjusted_score" in metrics:
                score = metrics["risk_adjusted_score"]
                if robustness["risk_adjusted_score_best_model"] is None or score > robustness["risk_adjusted_score_best_model"]:
                    robustness["risk_adjusted_score_best_model"] = score
                    has_data = True

            # Check for overfitting indicators
            # Look for train-val gap
            train_metric = None
            val_metric = None
            for key, val in metrics.items():
                key_lower = key.lower()
                if "train" in key_lower and isinstance(val, (int, float)):
                    train_metric = val
                elif ("val" in key_lower or "test" in key_lower) and isinstance(val, (int, float)):
                    val_metric = val

            if train_metric is not None and val_metric is not None:
                gap = abs(train_metric - val_metric)
                if gap > 0.15:
                    robustness["overfitting_risk"] = "high"
                    robustness["warnings"].append(f"Large train-val gap: {gap:.3f}")
                    has_data = True
                elif gap > 0.08:
                    if robustness["overfitting_risk"] != "high":
                        robustness["overfitting_risk"] = "medium"
                    has_data = True
                else:
                    if robustness["overfitting_risk"] is None:
                        robustness["overfitting_risk"] = "low"
                    has_data = True

            # Check baseline metrics for leakage indicators
            if trial.baseline_metrics_json:
                bm = trial.baseline_metrics_json
                shuffle = bm.get("label_shuffle", {})
                if shuffle.get("leakage_detected"):
                    robustness["leakage_suspected"] = True
                    if shuffle.get("warning"):
                        robustness["warnings"].append(shuffle["warning"])
                    has_data = True

            # Check split strategy for time-based issues
            if trial.data_split_strategy:
                # Check if time-based data using non-time split
                # This would be detected from dataset_spec.is_time_based
                pass

    return robustness if has_data else None


def _get_leakage_candidates(
    db: Session,
    dataset_spec: Optional[DatasetSpec],
    experiments: List[Experiment],
) -> List[Dict[str, Any]]:
    """Get leakage candidates from dataset spec or experiment results."""
    candidates = []

    # Check dataset spec for stored leakage candidates
    if dataset_spec and hasattr(dataset_spec, 'metadata_json') and dataset_spec.metadata_json:
        metadata = dataset_spec.metadata_json
        if "leakage_candidates" in metadata:
            candidates.extend(metadata["leakage_candidates"])

    # Check experiment trial results for leakage info
    for exp in experiments:
        if not exp.trials:
            continue

        for trial in exp.trials:
            if trial.metrics_json and "leakage_candidates" in trial.metrics_json:
                # Merge without duplicates
                existing_cols = {c["column"] for c in candidates}
                for lc in trial.metrics_json["leakage_candidates"]:
                    if lc.get("column") not in existing_cols:
                        candidates.append(lc)
                        existing_cols.add(lc["column"])

    return candidates


def _get_past_cycles_summary(
    db: Session,
    project_id: str,
    current_cycle_id: str,
    max_cycles: int,
) -> List[Dict[str, Any]]:
    """Get summary of past research cycles."""
    summaries = []

    # Get past cycles (excluding current)
    past_cycles = (
        db.query(ResearchCycle)
        .filter(
            ResearchCycle.project_id == project_id,
            ResearchCycle.id != current_cycle_id,
        )
        .order_by(ResearchCycle.created_at.desc())
        .limit(max_cycles)
        .all()
    )

    for cycle in past_cycles:
        summary = {
            "cycle_id": str(cycle.id),
            "goal": cycle.goal,
            "status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
            "best_model_metric": None,
            "overfitting_risk": None,
            "key_changes": None,
        }

        # Get best experiment metric from cycle
        cycle_experiments = (
            db.query(CycleExperiment)
            .filter(CycleExperiment.research_cycle_id == cycle.id)
            .all()
        )

        for ce in cycle_experiments:
            exp = db.query(Experiment).filter(Experiment.id == ce.experiment_id).first()
            if exp and exp.trials:
                for trial in exp.trials:
                    if trial.metrics_json and exp.primary_metric:
                        metric_val = trial.metrics_json.get(exp.primary_metric)
                        if metric_val is not None:
                            if summary["best_model_metric"] is None or metric_val > summary["best_model_metric"]:
                                summary["best_model_metric"] = metric_val

        # Get key changes from lab notebook
        notebook_entries = (
            db.query(LabNotebookEntry)
            .filter(LabNotebookEntry.research_cycle_id == cycle.id)
            .order_by(LabNotebookEntry.created_at.desc())
            .limit(1)
            .all()
        )

        if notebook_entries:
            entry = notebook_entries[0]
            # Extract key changes from entry content
            if entry.content:
                # Take first 200 chars as summary
                summary["key_changes"] = entry.content[:200] + ("..." if len(entry.content) > 200 else "")

        summaries.append(summary)

    return summaries


# ============================================
# Convenience Functions
# ============================================

def get_task_type_hints(context: Dict[str, Any]) -> Dict[str, Any]:
    """Get task type hints from context for agent reasoning.

    Returns helpful hints about the task based on context analysis.
    """
    hints = {
        "is_time_based": False,
        "is_classification": False,
        "is_regression": False,
        "is_binary": False,
        "is_multiclass": False,
        "recommended_split": "stratified",
        "recommended_metrics": [],
        "leakage_warnings": [],
        "data_quality_warnings": [],
    }

    # Determine task type from project
    project = context.get("project", {})
    task_type = project.get("task_type", "").lower()

    if task_type in ("binary", "classification", "multiclass"):
        hints["is_classification"] = True
        if task_type == "binary":
            hints["is_binary"] = True
            hints["recommended_metrics"] = ["roc_auc", "f1", "precision", "recall", "accuracy"]
        else:
            hints["is_multiclass"] = True
            hints["recommended_metrics"] = ["accuracy", "f1_macro", "f1_weighted"]
    elif task_type == "regression":
        hints["is_regression"] = True
        hints["recommended_metrics"] = ["rmse", "mae", "r2", "mape"]

    # Check if time-based
    dataset_spec = context.get("dataset_spec", {})
    if dataset_spec:
        hints["is_time_based"] = dataset_spec.get("is_time_based", False)

        if hints["is_time_based"]:
            hints["recommended_split"] = "time"
            hints["leakage_warnings"].append(
                "Time-based task: ensure features don't use future information"
            )

    # Add leakage warnings from candidates
    leakage_candidates = context.get("leakage_candidates", [])
    high_severity = [c for c in leakage_candidates if c.get("severity") == "high"]
    if high_severity:
        hints["leakage_warnings"].append(
            f"{len(high_severity)} high-severity leakage candidate(s) detected"
        )

    # Add data quality warnings
    data_profile = context.get("data_profile_summary", {})
    if data_profile:
        missingness = data_profile.get("missingness_summary", {})
        if missingness and missingness.get("columns_with_nulls", 0) > 0:
            hints["data_quality_warnings"].append(
                f"{missingness['columns_with_nulls']} column(s) have missing values"
            )

    return hints


def format_context_for_prompt(
    context: Dict[str, Any],
    include_sections: Optional[List[str]] = None,
    max_length: int = 4000,
) -> str:
    """Format task context as a string for LLM prompts.

    Args:
        context: The task context dictionary
        include_sections: Which sections to include (default: all)
        max_length: Maximum string length

    Returns:
        Formatted string suitable for LLM prompts
    """
    lines = []

    # Default sections
    if include_sections is None:
        include_sections = [
            "project", "dataset_spec", "data_profile_summary",
            "latest_experiments", "baselines", "robustness", "leakage_candidates"
        ]

    # Project info
    if "project" in include_sections and context.get("project"):
        proj = context["project"]
        lines.append("## Project Information")
        lines.append(f"- Name: {proj.get('name', 'Unknown')}")
        lines.append(f"- Description: {proj.get('description', 'N/A')}")
        lines.append(f"- Task Type: {proj.get('task_type', 'Unknown')}")
        lines.append("")

    # Dataset spec
    if "dataset_spec" in include_sections and context.get("dataset_spec"):
        ds = context["dataset_spec"]
        lines.append("## Dataset Configuration")
        lines.append(f"- Target Column: {ds.get('target_column', 'Unknown')}")
        lines.append(f"- Feature Count: {len(ds.get('feature_columns', []))}")
        lines.append(f"- Is Time-Based: {ds.get('is_time_based', False)}")
        if ds.get("time_column"):
            lines.append(f"- Time Column: {ds['time_column']}")
        if ds.get("entity_id_column"):
            lines.append(f"- Entity ID Column: {ds['entity_id_column']}")
        if ds.get("split_strategy"):
            lines.append(f"- Split Strategy: {ds['split_strategy']}")
        lines.append("")

    # Data profile
    if "data_profile_summary" in include_sections and context.get("data_profile_summary"):
        dp = context["data_profile_summary"]
        lines.append("## Data Profile")
        if dp.get("row_count"):
            lines.append(f"- Row Count: {dp['row_count']:,}")
        if dp.get("column_count"):
            lines.append(f"- Column Count: {dp['column_count']}")
        if dp.get("class_balance"):
            lines.append(f"- Class Balance: {dp['class_balance']}")
        lines.append("")

    # Experiments
    if "latest_experiments" in include_sections and context.get("latest_experiments"):
        lines.append("## Recent Experiments")
        for exp in context["latest_experiments"][:3]:  # Limit to 3
            lines.append(f"- {exp.get('name', 'Unknown')}: {exp.get('status', 'N/A')}")
            if exp.get("primary_metric_value") is not None:
                lines.append(f"  - {exp.get('primary_metric_name', 'Metric')}: {exp['primary_metric_value']:.4f}")
        lines.append("")

    # Baselines
    if "baselines" in include_sections and context.get("baselines", {}).get("available"):
        bl = context["baselines"]
        lines.append("## Baselines")
        if bl.get("majority_class"):
            mc = bl["majority_class"]
            lines.append(f"- Majority Class: acc={mc.get('accuracy', 'N/A')}, auc={mc.get('roc_auc', 'N/A')}")
        if bl.get("simple_model"):
            sm = bl["simple_model"]
            lines.append(f"- Simple Model: acc={sm.get('accuracy', 'N/A')}, auc={sm.get('roc_auc', 'N/A')}")
        if bl.get("mean_predictor"):
            mp = bl["mean_predictor"]
            lines.append(f"- Mean Predictor: rmse={mp.get('rmse', 'N/A')}, r2={mp.get('r2', 'N/A')}")
        lines.append("")

    # Robustness
    if "robustness" in include_sections and context.get("robustness"):
        rob = context["robustness"]
        lines.append("## Robustness Status")
        lines.append(f"- Overfitting Risk: {rob.get('overfitting_risk', 'Unknown')}")
        if rob.get("leakage_suspected"):
            lines.append("- **LEAKAGE SUSPECTED**")
        if rob.get("time_split_suspicious"):
            lines.append("- **TIME SPLIT SUSPICIOUS**")
        if rob.get("risk_adjusted_score_best_model") is not None:
            lines.append(f"- Risk-Adjusted Score: {rob['risk_adjusted_score_best_model']:.4f}")
        if rob.get("warnings"):
            for w in rob["warnings"][:3]:
                lines.append(f"  - Warning: {w}")
        lines.append("")

    # Leakage candidates
    if "leakage_candidates" in include_sections and context.get("leakage_candidates"):
        lc = context["leakage_candidates"]
        lines.append("## Leakage Candidates")
        lines.append(f"Total: {len(lc)} potential leakage feature(s)")
        for candidate in lc[:5]:  # Limit to 5
            lines.append(f"- {candidate.get('column', 'Unknown')} [{candidate.get('severity', 'N/A')}]: {candidate.get('reason', '')}")
        lines.append("")

    result = "\n".join(lines)

    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length - 20] + "\n\n... (truncated)"

    return result

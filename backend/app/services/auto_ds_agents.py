"""Auto DS Team agents for autonomous ML research.

This module contains the LLM-powered agents that drive the autonomous
data science research process:
- CrossAnalysisAgent: Analyzes patterns across experiments
- StrategyAgent: Plans next research steps
- ExperimentOrchestratorAgent: Designs concrete experiments
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.auto_ds_session import (
    AutoDSSession,
    AutoDSIteration,
    ResearchInsight,
    InsightType,
    InsightConfidence,
    GlobalInsight,
)
from app.models.experiment import Experiment, Trial
from app.models.dataset_spec import DatasetSpec
from app.models.data_source import DataSource
from app.models.project import Project
from app.services.llm_client import BaseLLMClient, get_llm_client
from app.services.context_builder import ContextBuilder
from app.services.leakage_detector import detect_potential_leakage_features
from app.services.auto_ds_prompts import (
    SYSTEM_ROLE_CROSS_ANALYST,
    SYSTEM_ROLE_STRATEGY_AGENT,
    SYSTEM_ROLE_EXPERIMENT_ORCHESTRATOR,
    SYSTEM_ROLE_DYNAMIC_PLANNER,
    get_cross_analysis_prompt,
    get_cross_analysis_schema,
    get_strategy_prompt,
    get_strategy_schema,
    get_experiment_design_from_strategy_prompt,
    get_experiment_design_from_strategy_schema,
    get_dynamic_planning_prompt,
    get_dynamic_planning_schema,
)

logger = logging.getLogger(__name__)

# Timeout for agent LLM calls (30 minutes for complex analysis)
AGENT_CALL_TIMEOUT = 1800


class CrossAnalysisAgent:
    """Agent that analyzes patterns across multiple experiments.

    This agent examines the results of multiple experiments to identify:
    - Which features consistently help or hurt performance
    - Optimal preprocessing and model configurations
    - Overfitting patterns to avoid
    - Contradictions that need further investigation
    """

    def __init__(self, db: Session, llm_client: Optional[BaseLLMClient] = None):
        """Initialize the cross-analysis agent.

        Args:
            db: Database session
            llm_client: Optional LLM client (will create one if not provided)
        """
        self.db = db
        self.llm_client = llm_client or get_llm_client()

    async def analyze_iteration(
        self,
        session: AutoDSSession,
        iteration: AutoDSIteration,
        experiments: List[Experiment],
        override_use_context: Optional[bool] = None,
        failed_experiments: Optional[List[Experiment]] = None,
    ) -> Dict[str, Any]:
        """Analyze experiments from an iteration to extract insights.

        Args:
            session: The Auto DS session
            iteration: The current iteration
            experiments: List of completed experiments to analyze
            override_use_context: If provided, overrides session.use_context_documents
            failed_experiments: Failed experiments to include for feedback/learning

        Returns:
            Dict containing insights, summary, best configuration, and questions
        """
        # Build failed experiments summary for feedback
        failed_summary = []
        if failed_experiments:
            for exp in failed_experiments:
                # Get dataset info
                dataset_info = {}
                if exp.dataset_spec:
                    spec_json = exp.dataset_spec.spec_json or {}
                    dataset_info = {
                        "name": exp.dataset_spec.name,
                        "features": spec_json.get("features", []),
                        "feature_engineering": spec_json.get("engineered_features", []),
                    }

                failed_summary.append({
                    "id": str(exp.id),
                    "name": exp.name,
                    "description": exp.description,
                    "dataset_name": dataset_info.get("name", "Unknown"),
                    "features": dataset_info.get("features", []),
                    "feature_engineering": dataset_info.get("feature_engineering", []),
                    "error_message": exp.error_message or "Unknown error",
                    "status": exp.status.value if exp.status else "failed",
                })

            logger.info(f"Including {len(failed_summary)} failed experiments in analysis for feedback")

        # Build experiment summaries
        experiments_summary = []
        for exp in experiments:
            # Get the best trial for this experiment
            best_trial = None
            if exp.trials:
                completed_trials = [t for t in exp.trials if t.status.value == "completed"]
                if completed_trials:
                    best_trial = max(
                        completed_trials,
                        key=lambda t: self._get_trial_score(t, exp.metric_direction.value if exp.metric_direction else "maximize")
                    )

            # Get dataset info
            dataset_info = {}
            if exp.dataset_spec:
                spec_json = exp.dataset_spec.spec_json or {}
                dataset_info = {
                    "name": exp.dataset_spec.name,
                    "features": spec_json.get("features", []),
                    "feature_engineering": spec_json.get("engineered_features", []),
                }

            # Extract training config from experiment plan or trial
            automl_config = {}
            if exp.experiment_plan_json:
                automl_config = exp.experiment_plan_json.get("automl_config", {})
            if best_trial and best_trial.automl_config:
                automl_config = best_trial.automl_config

            exp_summary = {
                "id": str(exp.id),
                "name": exp.name,
                "description": exp.description,
                "dataset_name": dataset_info.get("name", "Unknown"),
                "features": dataset_info.get("features", []),
                "feature_engineering": dataset_info.get("feature_engineering", []),
                "primary_metric": exp.primary_metric,
                "score": best_trial.metrics_json.get(exp.primary_metric) if best_trial and best_trial.metrics_json else None,
                "all_metrics": best_trial.metrics_json if best_trial else {},
                "best_model": best_trial.best_model_ref if best_trial else None,
                "leaderboard": best_trial.leaderboard_json if best_trial else [],
                "overfitting_risk": self._assess_overfitting(best_trial) if best_trial else "Unknown",
                "training_time_seconds": automl_config.get("time_limit", "Unknown"),
                "preset": automl_config.get("presets", automl_config.get("preset", "Unknown")),
            }
            experiments_summary.append(exp_summary)

        # Get existing insights for this project
        existing_insights = self.db.query(ResearchInsight).filter(
            ResearchInsight.project_id == session.project_id
        ).all()

        existing_insights_data = [
            {
                "title": ins.title,
                "description": ins.description,
                "confidence": ins.confidence.value if ins.confidence else "low",
                "type": ins.insight_type.value if ins.insight_type else "other",
            }
            for ins in existing_insights
        ]

        # Get project context
        project = session.project

        # Get target column from the first dataset spec's spec_json if available
        target_column = "Not specified"
        for exp in experiments:
            if exp.dataset_spec and exp.dataset_spec.spec_json:
                target_column = exp.dataset_spec.spec_json.get("target", target_column)
                break

        project_context = {
            "goal": project.description or "Not specified",
            "target": target_column,
            "problem_type": project.task_type.value if project.task_type else "Unknown",
            "metric_direction": "maximize",  # Default
        }

        # Get relevant global insights
        global_insights = self._get_relevant_global_insights(project_context)

        # Determine whether to use context (override takes precedence)
        use_context = override_use_context if override_use_context is not None else session.use_context_documents

        # Get context documents if configured to use them
        context_documents = ""
        if use_context:
            context_builder = ContextBuilder(self.db)
            context_documents = context_builder.build_context_section(session.project_id)
            if context_documents:
                logger.info(f"Using context documents for cross-analysis (session {session.id})")

        # Generate the analysis prompt (include failed experiments for feedback)
        prompt = get_cross_analysis_prompt(
            experiments_summary=experiments_summary,
            existing_insights=existing_insights_data,
            project_context=project_context,
            global_insights=global_insights,
            context_documents=context_documents,
            failed_experiments=failed_summary,
        )

        # Call the LLM
        messages = [
            {"role": "system", "content": SYSTEM_ROLE_CROSS_ANALYST},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await asyncio.wait_for(
                self.llm_client.chat_json(
                    messages=messages,
                    response_schema=get_cross_analysis_schema(),
                ),
                timeout=AGENT_CALL_TIMEOUT,
            )
            logger.info(f"Cross-analysis completed with {len(result.get('insights', []))} insights")
            return result

        except asyncio.TimeoutError:
            logger.error("Cross-analysis timed out")
            raise ValueError("Cross-analysis timed out after 30 minutes")
        except Exception as e:
            logger.error(f"Cross-analysis failed: {e}")
            raise

    def _get_trial_score(self, trial: Trial, direction: str) -> float:
        """Get comparable score from a trial."""
        if not trial.metrics_json:
            return float("-inf") if direction == "maximize" else float("inf")

        # Try common metric names
        for metric in ["accuracy", "roc_auc", "f1", "rmse", "mse", "mae"]:
            if metric in trial.metrics_json:
                score = trial.metrics_json[metric]
                # Invert for minimize metrics
                if metric in ["rmse", "mse", "mae"] and direction == "maximize":
                    return -score
                return score

        # Return first available metric
        if trial.metrics_json:
            return list(trial.metrics_json.values())[0]
        return 0.0

    def _assess_overfitting(self, trial: Trial) -> str:
        """Assess overfitting risk based on trial metrics.

        Checks two types of overfitting:
        1. Training overfitting: train_score >> val_score (model memorized training data)
        2. Validation overfitting: val_score >> holdout_score (model/tuning overfit to validation)
        """
        if not trial.metrics_json:
            return "Unknown"

        metrics = trial.metrics_json
        warnings = []

        # Extract all 3 scores
        train_score = (
            metrics.get("train_mcc") or
            metrics.get("train_score") or
            metrics.get("train_accuracy")
        )
        val_score = (
            metrics.get("validation_mcc") or
            metrics.get("validation_score") or
            metrics.get("val_score") or
            metrics.get("val_mcc") or
            metrics.get("val_accuracy") or
            metrics.get("score_val")  # AutoGluon's default key
        )
        holdout_score = (
            metrics.get("holdout_mcc") or
            metrics.get("holdout_score") or
            metrics.get("holdout_accuracy")
        )

        # Check 1: Training overfitting (train >> val)
        if train_score and val_score:
            train_val_gap = train_score - val_score
            if train_val_gap > 0.1:
                warnings.append(f"High train/val gap ({train_val_gap:.3f})")
            elif train_val_gap > 0.05:
                warnings.append(f"Moderate train/val gap ({train_val_gap:.3f})")

        # Check 2: Validation overfitting (val >> holdout) - this is the gold standard check
        if val_score and holdout_score:
            val_holdout_gap = val_score - holdout_score
            if val_holdout_gap > 0.05:
                warnings.append(f"CRITICAL: val/holdout gap ({val_holdout_gap:.3f}) - model may not generalize")
            elif val_holdout_gap > 0.02:
                warnings.append(f"val/holdout gap ({val_holdout_gap:.3f}) - possible validation overfit")

        # Return combined assessment
        if not warnings:
            if holdout_score:
                return "Low (holdout validates generalization)"
            elif val_score:
                return "Low (no holdout to confirm)"
            else:
                return "Unknown (insufficient metrics)"

        if any("CRITICAL" in w for w in warnings):
            return "HIGH: " + "; ".join(warnings)
        elif len(warnings) > 1:
            return "Medium: " + "; ".join(warnings)
        else:
            return "Medium: " + warnings[0]

    def _get_relevant_global_insights(self, project_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant global insights from other projects."""
        # Query global insights that might apply
        problem_type = project_context.get("problem_type", "").lower()

        insights = self.db.query(GlobalInsight).filter(
            GlobalInsight.is_active == True,
            GlobalInsight.confidence_score >= 0.6,
        ).order_by(GlobalInsight.confidence_score.desc()).limit(10).all()

        relevant = []
        for ins in insights:
            # Check if applicable to this problem type
            task_types = ins.task_types or []
            if task_types and problem_type not in [t.lower() for t in task_types]:
                continue

            relevant.append({
                "title": ins.title,
                "description": ins.description,
                "confidence_score": ins.confidence_score,
                "evidence_count": ins.evidence_count,
            })

        return relevant[:5]  # Top 5 relevant insights


class StrategyAgent:
    """Agent that plans the next research iteration.

    This agent takes the analysis results and decides:
    - Whether to continue or stop
    - What strategy to use (exploit vs explore)
    - What specific experiments to run
    """

    def __init__(self, db: Session, llm_client: Optional[BaseLLMClient] = None):
        """Initialize the strategy agent.

        Args:
            db: Database session
            llm_client: Optional LLM client
        """
        self.db = db
        self.llm_client = llm_client or get_llm_client()

    async def plan_next_iteration(
        self,
        session: AutoDSSession,
        analysis_results: Dict[str, Any],
        override_use_context: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Plan the next research iteration based on analysis.

        Args:
            session: The Auto DS session
            analysis_results: Output from CrossAnalysisAgent
            override_use_context: If provided, overrides session.use_context_documents

        Returns:
            Dict containing strategy decisions and proposed experiments
        """
        # Build session status
        session_status = {
            "current_iteration": session.current_iteration,
            "best_score": session.best_score,
            "iterations_without_improvement": session.iterations_without_improvement,
            "total_experiments_run": session.total_experiments_run,
        }

        # Build stopping conditions
        stopping_conditions = {
            "max_iterations": session.max_iterations,
            "accuracy_threshold": session.accuracy_threshold,
            "time_budget_minutes": session.time_budget_minutes,
            "min_improvement_threshold": session.min_improvement_threshold,
            "plateau_iterations": session.plateau_iterations,
        }

        # Available actions based on project capabilities
        available_actions = [
            "refine_features - Add/remove features based on insights",
            "new_features - Create new engineered features",
            "model_tuning - Adjust AutoML configuration",
            "ensemble - Try ensemble approaches",
            "ablation - Remove suspected bad features",
            "data_augmentation - Try different preprocessing",
        ]

        # Determine whether to use context (override takes precedence)
        use_context = override_use_context if override_use_context is not None else session.use_context_documents

        # Get context documents if configured to use them
        context_documents = ""
        if use_context:
            context_builder = ContextBuilder(self.db)
            context_documents = context_builder.build_context_section(session.project_id)
            if context_documents:
                logger.info(f"Using context documents for strategy planning (session {session.id})")

        # Run leakage detection for strategy awareness
        leakage_candidates = self._detect_leakage_for_session(session)
        if leakage_candidates:
            high_count = len([lc for lc in leakage_candidates if lc.get("severity") == "high"])
            logger.info(f"Strategy agent aware of {len(leakage_candidates)} leakage candidates ({high_count} high severity)")

        # Generate strategy prompt
        prompt = get_strategy_prompt(
            analysis_results=analysis_results,
            session_status=session_status,
            stopping_conditions=stopping_conditions,
            available_actions=available_actions,
            context_documents=context_documents,
            leakage_candidates=leakage_candidates,
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_STRATEGY_AGENT},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await asyncio.wait_for(
                self.llm_client.chat_json(
                    messages=messages,
                    response_schema=get_strategy_schema(),
                ),
                timeout=AGENT_CALL_TIMEOUT,
            )
            logger.info(f"Strategy planning completed: continue={result.get('should_continue')}, mode={result.get('strategy_mode')}")
            return result

        except asyncio.TimeoutError:
            logger.error("Strategy planning timed out")
            raise ValueError("Strategy planning timed out")
        except Exception as e:
            logger.error(f"Strategy planning failed: {e}")
            raise

    def _detect_leakage_for_session(
        self,
        session: AutoDSSession,
    ) -> List[Dict[str, Any]]:
        """Detect potential leakage features for the session's project.

        Args:
            session: The AutoDS session

        Returns:
            List of leakage candidate dicts
        """
        import os
        import pandas as pd

        project = session.project
        if not project:
            return []

        # Get target column from dataset specs
        target_column = None
        current_datasets = self.db.query(DatasetSpec).filter(
            DatasetSpec.project_id == session.project_id
        ).all()

        for ds in current_datasets:
            if ds.spec_json:
                target_column = ds.spec_json.get("target")
                if target_column:
                    break

        if not target_column:
            return []

        # Try to get data source
        data_source = self.db.query(DataSource).filter(
            DataSource.project_id == project.id
        ).first()

        if not data_source or not data_source.file_path:
            for ds in current_datasets:
                if ds.data_source_id:
                    data_source = self.db.query(DataSource).filter(
                        DataSource.id == ds.data_source_id
                    ).first()
                    if data_source and data_source.file_path:
                        break

        if not data_source or not data_source.file_path:
            return []

        if not os.path.exists(data_source.file_path):
            return []

        try:
            df = pd.read_csv(data_source.file_path, nrows=10000)

            if target_column not in df.columns:
                return []

            return detect_potential_leakage_features(
                df=df,
                target_column=target_column,
                correlation_threshold=0.9,
            )

        except Exception as e:
            logger.warning(f"Leakage detection failed in strategy agent: {e}")
            return []


class ExperimentOrchestratorAgent:
    """Agent that designs concrete experiments from strategy.

    This agent takes high-level strategic directions and creates
    specific, executable experiment configurations.
    """

    def __init__(self, db: Session, llm_client: Optional[BaseLLMClient] = None):
        """Initialize the experiment orchestrator agent.

        Args:
            db: Database session
            llm_client: Optional LLM client
        """
        self.db = db
        self.llm_client = llm_client or get_llm_client()

    async def design_experiments(
        self,
        session: AutoDSSession,
        strategy: Dict[str, Any],
        override_use_context: Optional[bool] = None,
        failure_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Design concrete experiments based on strategy.

        Args:
            session: The Auto DS session
            strategy: Output from StrategyAgent
            override_use_context: If provided, overrides session.use_context_documents
            failure_analysis: Optional analysis of failed experiments to avoid repeating mistakes

        Returns:
            Dict containing experiment specifications
        """
        project = session.project

        # Get current dataset specs
        current_datasets = self.db.query(DatasetSpec).filter(
            DatasetSpec.project_id == session.project_id
        ).all()

        # Get target column and available features from dataset specs
        target_column = "Not specified"
        available_features = []
        for ds in current_datasets:
            if ds.spec_json:
                if target_column == "Not specified":
                    target_column = ds.spec_json.get("target", target_column)
                # Collect features from all dataset specs
                features = ds.spec_json.get("features", [])
                for f in features:
                    if isinstance(f, str) and f not in available_features:
                        available_features.append(f)
                    elif isinstance(f, dict) and f.get("name") not in available_features:
                        available_features.append(f.get("name"))

        # Get project context
        project_context = {
            "goal": project.description or "Not specified",
            "target": target_column,
            "problem_type": project.task_type.value if project.task_type else "Unknown",
        }

        datasets_data = [
            {
                "id": str(ds.id),
                "name": ds.name,
                "description": ds.description,
                "features": (ds.spec_json or {}).get("features", []),
            }
            for ds in current_datasets
        ]

        # Build schema summary from dataset specs (no project.schema_json)
        schema_summary = self._build_schema_summary_from_datasets(current_datasets)

        # Determine whether to use context (override takes precedence)
        use_context = override_use_context if override_use_context is not None else session.use_context_documents

        # Get context documents if configured to use them
        context_documents = ""
        if use_context:
            context_builder = ContextBuilder(self.db)
            context_documents = context_builder.build_context_section(session.project_id)
            if context_documents:
                logger.info(f"Using context documents for experiment design (session {session.id})")

        # Run leakage detection on the data
        leakage_candidates = self._detect_leakage_for_project(
            project=project,
            target_column=target_column if target_column != "Not specified" else None,
            current_datasets=current_datasets,
        )
        if leakage_candidates:
            high_count = len([lc for lc in leakage_candidates if lc.get("severity") == "high"])
            logger.info(f"Detected {len(leakage_candidates)} potential leakage features ({high_count} high severity)")

        # Generate experiment design prompt
        prompt = get_experiment_design_from_strategy_prompt(
            strategy=strategy,
            project_context=project_context,
            available_features=available_features,
            current_datasets=datasets_data,
            schema_summary=schema_summary,
            context_documents=context_documents,
            leakage_candidates=leakage_candidates,
            failure_analysis=failure_analysis,
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_EXPERIMENT_ORCHESTRATOR},
            {"role": "user", "content": prompt},
        ]

        # Get high-severity leakage features for validation
        high_severity_features = set(
            lc.get("column") for lc in (leakage_candidates or [])
            if lc.get("severity") == "high" and lc.get("column")
        )

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self.llm_client.chat_json(
                        messages=messages,
                        response_schema=get_experiment_design_from_strategy_schema(),
                    ),
                    timeout=AGENT_CALL_TIMEOUT,
                )

                # Validate: Check if any experiments use high-severity leakage features
                if high_severity_features:
                    leakage_violations = self._validate_experiments_for_leakage(
                        result.get("experiments", []),
                        high_severity_features,
                    )

                    if leakage_violations:
                        if attempt < max_retries:
                            # Retry with feedback
                            logger.warning(
                                f"Experiment design attempt {attempt + 1} used leakage features: "
                                f"{leakage_violations}. Retrying with feedback..."
                            )
                            feedback_msg = self._build_leakage_feedback(leakage_violations)
                            messages.append({"role": "assistant", "content": json.dumps(result)})
                            messages.append({"role": "user", "content": feedback_msg})
                            continue
                        else:
                            # Max retries reached - auto-fix by removing leakage features
                            logger.warning(
                                f"Max retries reached. Auto-removing leakage features from experiments."
                            )
                            result = self._remove_leakage_features_from_experiments(
                                result, high_severity_features
                            )

                logger.info(f"Experiment design completed: {len(result.get('experiments', []))} experiments")
                return result

            except asyncio.TimeoutError:
                logger.error("Experiment design timed out")
                raise ValueError("Experiment design timed out")
            except Exception as e:
                logger.error(f"Experiment design failed: {e}")
                raise

        # Should not reach here, but just in case
        raise ValueError("Experiment design failed after all retries")

    def _build_schema_summary_from_datasets(self, datasets: List[DatasetSpec]) -> str:
        """Build a human-readable schema summary from dataset specs."""
        if not datasets:
            return "No datasets available"

        lines = []
        all_features = set()

        for ds in datasets:
            spec_json = ds.spec_json or {}
            features = spec_json.get("features", [])
            for f in features:
                if isinstance(f, str):
                    all_features.add(f)
                elif isinstance(f, dict):
                    all_features.add(f.get("name", "unknown"))

        if not all_features:
            return "No features defined in datasets"

        for feature in sorted(list(all_features))[:30]:
            lines.append(f"  - {feature}")

        if len(all_features) > 30:
            lines.append(f"  ... and {len(all_features) - 30} more features")

        return "\n".join(lines) if lines else "No features available"

    def _detect_leakage_for_project(
        self,
        project: Project,
        target_column: Optional[str],
        current_datasets: List[DatasetSpec],
    ) -> List[Dict[str, Any]]:
        """Detect potential leakage features for the project.

        Args:
            project: The project
            target_column: Target column name
            current_datasets: Current dataset specs

        Returns:
            List of leakage candidate dicts
        """
        import os
        import pandas as pd

        if not target_column:
            return []

        # Try to get data source from project
        data_source = self.db.query(DataSource).filter(
            DataSource.project_id == project.id
        ).first()

        if not data_source or not data_source.file_path:
            # Try to get from dataset specs
            for ds in current_datasets:
                if ds.data_source_id:
                    data_source = self.db.query(DataSource).filter(
                        DataSource.id == ds.data_source_id
                    ).first()
                    if data_source and data_source.file_path:
                        break

        if not data_source or not data_source.file_path:
            logger.debug("No data source found for leakage detection")
            return []

        if not os.path.exists(data_source.file_path):
            logger.debug(f"Data file not found: {data_source.file_path}")
            return []

        try:
            # Load a sample for analysis (limit to 10k rows)
            df = pd.read_csv(data_source.file_path, nrows=10000)

            if target_column not in df.columns:
                logger.debug(f"Target column '{target_column}' not in data")
                return []

            # Run leakage detection
            leakage_candidates = detect_potential_leakage_features(
                df=df,
                target_column=target_column,
                correlation_threshold=0.9,
            )

            return leakage_candidates

        except Exception as e:
            logger.warning(f"Leakage detection failed: {e}")
            return []

    def _validate_experiments_for_leakage(
        self,
        experiments: List[Dict[str, Any]],
        high_severity_features: set,
    ) -> Dict[str, List[str]]:
        """Check if experiments use high-severity leakage features.

        Args:
            experiments: List of experiment designs from LLM
            high_severity_features: Set of feature names flagged as high-severity leakage

        Returns:
            Dict mapping experiment names to list of leakage features used
        """
        violations = {}

        for exp in experiments:
            exp_name = exp.get("name", "Unknown")
            dataset_spec = exp.get("dataset_spec", {})

            # Check features list
            features = dataset_spec.get("features", [])
            leaky_features = []

            for feature in features:
                feature_name = feature if isinstance(feature, str) else feature.get("name", "")
                if feature_name in high_severity_features:
                    leaky_features.append(feature_name)

            # Also check feature engineering outputs
            feature_eng = dataset_spec.get("feature_engineering", [])
            for eng in feature_eng:
                output_col = eng.get("output_column", "")
                source_cols = eng.get("source_columns", [])

                # Check if output is a leakage feature (shouldn't happen but check anyway)
                if output_col in high_severity_features:
                    leaky_features.append(output_col)

                # Check if sources include leakage features
                for src in source_cols:
                    if src in high_severity_features:
                        leaky_features.append(f"{src} (used in {output_col})")

            if leaky_features:
                violations[exp_name] = leaky_features

        return violations

    def _build_leakage_feedback(
        self,
        leakage_violations: Dict[str, List[str]],
    ) -> str:
        """Build feedback message for retry after leakage violation.

        Args:
            leakage_violations: Dict mapping experiment names to leakage features

        Returns:
            Feedback message string
        """
        lines = [
            "❌ **EXPERIMENT DESIGN REJECTED - DATA LEAKAGE DETECTED**",
            "",
            "Your experiment design includes HIGH-SEVERITY leakage features that MUST be removed:",
            "",
        ]

        for exp_name, features in leakage_violations.items():
            lines.append(f"**{exp_name}**:")
            for feat in features:
                lines.append(f"  - `{feat}` ← REMOVE THIS")
            lines.append("")

        lines.extend([
            "**REQUIRED ACTION**: Redesign these experiments WITHOUT the leakage features listed above.",
            "",
            "These features have suspiciously high correlation with the target or contain target information.",
            "Using them will result in artificially inflated performance that won't generalize to production.",
            "",
            "Please provide a corrected experiment design that excludes ALL of the flagged features.",
        ])

        return "\n".join(lines)

    def _remove_leakage_features_from_experiments(
        self,
        result: Dict[str, Any],
        high_severity_features: set,
    ) -> Dict[str, Any]:
        """Auto-remove leakage features from experiment designs.

        This is the fallback when the LLM fails to remove them after retries.

        Args:
            result: The experiment design result from LLM
            high_severity_features: Set of feature names to remove

        Returns:
            Modified result with leakage features removed
        """
        experiments = result.get("experiments", [])

        for exp in experiments:
            exp_name = exp.get("name", "Unknown")
            dataset_spec = exp.get("dataset_spec", {})

            # Filter features list
            features = dataset_spec.get("features", [])
            if features:
                original_count = len(features)
                filtered_features = []
                for feature in features:
                    feature_name = feature if isinstance(feature, str) else feature.get("name", "")
                    if feature_name not in high_severity_features:
                        filtered_features.append(feature)
                    else:
                        logger.info(f"Auto-removed leakage feature '{feature_name}' from '{exp_name}'")

                dataset_spec["features"] = filtered_features
                if len(filtered_features) < original_count:
                    logger.warning(
                        f"Removed {original_count - len(filtered_features)} leakage features from '{exp_name}'"
                    )

            # Filter feature engineering sources
            feature_eng = dataset_spec.get("feature_engineering", [])
            if feature_eng:
                filtered_eng = []
                for eng in feature_eng:
                    source_cols = eng.get("source_columns", [])
                    # Remove if any source is a leakage feature
                    has_leakage_source = any(src in high_severity_features for src in source_cols)
                    if not has_leakage_source:
                        filtered_eng.append(eng)
                    else:
                        logger.info(
                            f"Auto-removed feature engineering '{eng.get('output_column')}' from '{exp_name}' "
                            f"(uses leakage source)"
                        )
                dataset_spec["feature_engineering"] = filtered_eng

        return result


class DynamicPlanningAgent:
    """Agent that designs experiments dynamically during iteration execution.

    Unlike the strategy/orchestrator agents that plan between iterations,
    this agent plans mid-iteration based on real-time experiment results.
    It enables true adaptive exploration where each experiment is informed
    by the results of previous experiments in the same iteration.

    Key capabilities:
    - Analyzes patterns from completed experiments in current cycle
    - Designs next experiment(s) to maximize information gain
    - Applies feature engineering, ensemble, and ablation strategies
    - Decides when to stop exploring (convergence detection)
    """

    def __init__(self, db: Session, llm_client: Optional[BaseLLMClient] = None):
        """Initialize the dynamic planning agent.

        Args:
            db: Database session
            llm_client: Optional LLM client (will create one if not provided)
        """
        self.db = db
        self.llm_client = llm_client or get_llm_client()

    async def design_next_experiments(
        self,
        session: AutoDSSession,
        iteration: AutoDSIteration,
        experiment_history: List[Dict[str, Any]],
        available_datasets: List[DatasetSpec],
        experiments_to_design: int = 1,
        feature_flags: Optional[Dict[str, bool]] = None,
        override_use_context: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Design the next experiment(s) based on current results.

        Args:
            session: The Auto DS session
            iteration: The current iteration
            experiment_history: List of completed experiment summaries with results
            available_datasets: Available datasets for new experiments
            experiments_to_design: Number of experiments to design
            feature_flags: Dict of enabled features (feature_engineering, ensemble, etc.)
            override_use_context: If provided, overrides session.use_context_documents

        Returns:
            Dict containing:
                - experiments: List of experiment plan dicts
                - should_stop: Whether to stop the iteration
                - stop_reason: Reason for stopping (if applicable)
                - reasoning: Agent's reasoning for the design
        """
        feature_flags = feature_flags or {
            "enable_feature_engineering": True,
            "enable_ensemble": True,
            "enable_ablation": True,
            "enable_diverse_configs": True,
        }

        # Get project context
        project = session.project

        # Get target column from available datasets
        target_column = "Not specified"
        available_features = []
        for ds in available_datasets:
            if ds.spec_json:
                if target_column == "Not specified":
                    target_column = ds.spec_json.get("target", target_column)
                features = ds.spec_json.get("features", [])
                for f in features:
                    if isinstance(f, str) and f not in available_features:
                        available_features.append(f)
                    elif isinstance(f, dict) and f.get("name") not in available_features:
                        available_features.append(f.get("name"))

        project_context = {
            "goal": project.description or "Not specified",
            "target": target_column,
            "problem_type": project.task_type.value if project.task_type else "Unknown",
        }

        datasets_data = [
            {
                "id": str(ds.id),
                "name": ds.name,
                "features": (ds.spec_json or {}).get("features", []),
            }
            for ds in available_datasets
        ]

        # Build session progress info
        session_progress = {
            "current_iteration": session.current_iteration,
            "best_score": session.best_score,
            "experiments_run_this_iteration": len(experiment_history),
            "total_experiments_run": session.total_experiments_run,
        }

        # Determine whether to use context (override takes precedence)
        use_context = override_use_context if override_use_context is not None else session.use_context_documents

        # Get context documents if configured to use them
        context_documents = ""
        if use_context:
            context_builder = ContextBuilder(self.db)
            context_documents = context_builder.build_context_section(session.project_id)
            if context_documents:
                logger.info(f"Using context documents for dynamic planning (session {session.id})")

        # Generate the prompt
        prompt = get_dynamic_planning_prompt(
            experiment_history=experiment_history,
            available_datasets=datasets_data,
            available_features=available_features,
            project_context=project_context,
            session_progress=session_progress,
            experiments_to_design=experiments_to_design,
            feature_flags=feature_flags,
            context_documents=context_documents,
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_DYNAMIC_PLANNER},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await asyncio.wait_for(
                self.llm_client.chat_json(
                    messages=messages,
                    response_schema=get_dynamic_planning_schema(),
                ),
                timeout=AGENT_CALL_TIMEOUT,
            )
            logger.info(
                f"Dynamic planning completed: {len(result.get('experiments', []))} experiments, "
                f"should_stop={result.get('should_stop', False)}"
            )
            return result

        except asyncio.TimeoutError:
            logger.error("Dynamic planning timed out")
            raise ValueError("Dynamic planning timed out")
        except Exception as e:
            logger.error(f"Dynamic planning failed: {e}")
            raise


async def _run_analysis_pipeline(
    db: Session,
    session: AutoDSSession,
    iteration: AutoDSIteration,
    experiments: List[Experiment],
    failed_experiments: Optional[List[Experiment]] = None,
    override_use_context: Optional[bool] = None,
    experiment_name_suffix: str = "",
) -> Dict[str, Any]:
    """Run a single analysis pipeline (with or without context).

    Args:
        db: Database session
        session: The Auto DS session
        iteration: The current iteration
        experiments: Completed experiments from this iteration
        failed_experiments: Failed experiments to include for feedback
        override_use_context: If provided, overrides session.use_context_documents
        experiment_name_suffix: Suffix to add to experiment names (for A/B testing)

    Returns:
        Dict with analysis_results, strategy, and experiment_designs
    """
    # Step 1: Cross-analysis (include failed experiments for learning)
    cross_agent = CrossAnalysisAgent(db)
    analysis_results = await cross_agent.analyze_iteration(
        session, iteration, experiments, override_use_context,
        failed_experiments=failed_experiments,
    )

    # Step 2: Strategy planning
    strategy_agent = StrategyAgent(db)
    strategy = await strategy_agent.plan_next_iteration(
        session, analysis_results, override_use_context
    )

    # Step 3: Experiment design (if continuing)
    experiment_designs = None
    logger.info(f"🔬 Step 3: should_continue={strategy.get('should_continue', False)}")
    print(f"🔬 Step 3: should_continue={strategy.get('should_continue', False)}")
    if strategy.get("should_continue", False):
        logger.info("🔬 Creating ExperimentOrchestratorAgent to design experiments...")
        print("🔬 Creating ExperimentOrchestratorAgent to design experiments...")
        orchestrator_agent = ExperimentOrchestratorAgent(db)
        try:
            # Pass failure_analysis from cross-analysis to help avoid repeating mistakes
            # Also include formula validation failures from previous iteration
            failure_analysis = analysis_results.get("failure_analysis") or {}

            # Check for formula validation failures from previous iteration
            if iteration.analysis_summary_json and iteration.analysis_summary_json.get("formula_validation_failures"):
                validation_failures = iteration.analysis_summary_json["formula_validation_failures"]
                logger.info(f"Including {len(validation_failures)} formula validation failures from previous iteration")

                # Merge validation failures into failure_analysis
                if not failure_analysis:
                    failure_analysis = {}

                # Add validation-specific data
                existing_formulas = failure_analysis.get("problematic_formulas", [])
                existing_recommendations = failure_analysis.get("recommendations", [])

                for vf in validation_failures:
                    # Add the formula to problematic list
                    formula_desc = f"{vf.get('output_column')}: {vf.get('error_message')} - FIX: {vf.get('suggested_fix')}"
                    if formula_desc not in existing_formulas:
                        existing_formulas.append(formula_desc)

                failure_analysis["problematic_formulas"] = existing_formulas
                failure_analysis["failure_count"] = failure_analysis.get("failure_count", 0) + len(validation_failures)

                # Add specific recommendation
                existing_recommendations.append(
                    f"CRITICAL: {len(validation_failures)} formula(s) failed pre-validation in the last iteration. "
                    f"Review the errors above and fix the formulas before using them."
                )
                failure_analysis["recommendations"] = existing_recommendations

            experiment_designs = await orchestrator_agent.design_experiments(
                session, strategy, override_use_context,
                failure_analysis=failure_analysis if failure_analysis else None,
            )
            logger.info(f"🔬 ExperimentOrchestratorAgent returned: {type(experiment_designs)}, experiments count: {len(experiment_designs.get('experiments', [])) if experiment_designs else 0}")
            print(f"🔬 ExperimentOrchestratorAgent returned: {type(experiment_designs)}, experiments count: {len(experiment_designs.get('experiments', [])) if experiment_designs else 0}")
        except Exception as e:
            logger.error(f"🔬 ExperimentOrchestratorAgent FAILED: {e}")
            print(f"🔬 ExperimentOrchestratorAgent FAILED: {e}")
            import traceback
            traceback.print_exc()
            experiment_designs = None

        # Track context usage and add {CONTEXT} to names when context is used
        if experiment_designs:
            use_context = override_use_context if override_use_context is not None else session.use_context_documents
            for exp in experiment_designs.get("experiments", []):
                # Track context usage
                exp["context_variant"] = "with_context" if use_context else "without_context"
                exp["used_context"] = use_context

                # Add suffix for A/B testing (WITH/NO CONTEXT) if provided
                if experiment_name_suffix:
                    exp["name"] = f"{exp.get('name', 'Experiment')} {experiment_name_suffix}"
                # Always add {CONTEXT} marker when context is used (regardless of A/B testing)
                elif use_context:
                    name = exp.get("name", "Experiment")
                    if "{CONTEXT}" not in name:
                        exp["name"] = f"{name} {{CONTEXT}}"

    return {
        "analysis_results": analysis_results,
        "strategy": strategy,
        "experiment_designs": experiment_designs,
    }


async def run_analysis_phase(
    db: Session,
    session: AutoDSSession,
    iteration: AutoDSIteration,
    experiments: List[Experiment],
    failed_experiments: Optional[List[Experiment]] = None,
) -> Dict[str, Any]:
    """Run the complete analysis phase of an iteration.

    This includes:
    1. Cross-analysis of experiment results (including failure analysis)
    2. Strategy planning for next iteration
    3. Experiment design (if continuing)

    If A/B testing is enabled (session.context_ab_testing=True), this runs
    the analysis twice - once WITH context and once WITHOUT - and merges
    the experiment designs with appropriate naming.

    Args:
        db: Database session
        session: The Auto DS session
        iteration: The current iteration
        experiments: Completed experiments from this iteration
        failed_experiments: Failed experiments to include for feedback/learning

    Returns:
        Dict with analysis_results, strategy, and experiment_designs
    """
    logger.info(f"Starting analysis phase for session {session.id}, iteration {iteration.iteration_number}")

    # Check if A/B testing is enabled and there are context documents available
    if session.context_ab_testing:
        from app.services.context_builder import ContextBuilder
        context_builder = ContextBuilder(db)

        if context_builder.has_context_documents(session.project_id):
            logger.info("A/B testing enabled: running analysis WITH and WITHOUT context")

            # Run WITH context
            with_context_result = await _run_analysis_pipeline(
                db, session, iteration, experiments,
                failed_experiments=failed_experiments,
                override_use_context=True,
                experiment_name_suffix="[WITH CONTEXT]"
            )

            # Run WITHOUT context
            without_context_result = await _run_analysis_pipeline(
                db, session, iteration, experiments,
                failed_experiments=failed_experiments,
                override_use_context=False,
                experiment_name_suffix="[NO CONTEXT]"
            )

            # Merge experiment designs
            merged_experiments = []
            if with_context_result.get("experiment_designs"):
                merged_experiments.extend(
                    with_context_result["experiment_designs"].get("experiments", [])
                )
            if without_context_result.get("experiment_designs"):
                merged_experiments.extend(
                    without_context_result["experiment_designs"].get("experiments", [])
                )

            # Use the WITH context analysis and strategy as primary (they had more info)
            merged_result = with_context_result.copy()
            if merged_experiments:
                merged_result["experiment_designs"] = {
                    "experiments": merged_experiments,
                    "execution_order": [exp.get("name") for exp in merged_experiments],
                    "notes": "A/B testing: half experiments WITH context, half WITHOUT",
                }

            # Combine strategies - continue if either says continue
            should_continue = (
                with_context_result.get("strategy", {}).get("should_continue", False) or
                without_context_result.get("strategy", {}).get("should_continue", False)
            )
            merged_result["strategy"]["should_continue"] = should_continue

            logger.info(
                f"A/B testing: {len(merged_experiments)} total experiments "
                f"({len(with_context_result.get('experiment_designs', {}).get('experiments', []))} with context, "
                f"{len(without_context_result.get('experiment_designs', {}).get('experiments', []))} without)"
            )

            return merged_result

    # Standard run (no A/B testing or no context documents)
    return await _run_analysis_pipeline(
        db, session, iteration, experiments,
        failed_experiments=failed_experiments,
    )

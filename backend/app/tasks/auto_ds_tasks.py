"""Celery tasks for Auto DS autonomous research sessions."""
import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.auto_ds_session import (
    AutoDSSession,
    AutoDSSessionStatus,
    AutoDSIteration,
    AutoDSIterationStatus,
    AutoDSIterationExperiment,
    ResearchInsight,
    InsightType,
    InsightConfidence,
    ExecutionMode,
)
from app.models.experiment import Experiment, ExperimentStatus
from app.models.dataset_spec import DatasetSpec
from app.services.auto_ds_orchestrator import AutoDSOrchestrator
from app.services.auto_ds_agents import (
    CrossAnalysisAgent,
    StrategyAgent,
    ExperimentOrchestratorAgent,
    run_analysis_phase,
)
from app.services.stop_flag import check_stop_flag, clear_stop_flag
from app.services.feature_engineering import (
    validate_feature_engineering_batch,
    get_validation_feedback_for_agent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS FOR ADVANCED FEATURES
# =============================================================================


def _create_derivative_dataset_spec(
    db,
    base_dataset_spec_id: UUID,
    session: AutoDSSession,
    feature_engineering: List[Dict[str, Any]],
    drop_columns: Optional[List[str]] = None,
    name_suffix: str = "",
    used_context: bool = False,
) -> DatasetSpec:
    """Create a derivative dataset spec with feature engineering or dropped columns.

    This allows AI-designed experiments to have custom feature engineering
    without modifying the original dataset spec.

    Args:
        db: Database session
        base_dataset_spec_id: UUID of the base dataset spec to derive from
        session: Auto DS session for context
        feature_engineering: List of feature engineering specs from AI
        drop_columns: Optional list of columns to exclude (for ablation)
        name_suffix: Optional suffix for the derived dataset name
        used_context: Whether context documents were used in creating this dataset

    Returns:
        New DatasetSpec with the modifications applied
    """
    # Get base dataset spec
    base_spec = db.query(DatasetSpec).filter(DatasetSpec.id == base_dataset_spec_id).first()
    if not base_spec:
        raise ValueError(f"Base dataset spec {base_dataset_spec_id} not found")

    # CRITICAL: Validate base dataset has data_sources_json - cannot create derivative without it
    if not base_spec.data_sources_json:
        raise ValueError(
            f"Base dataset spec '{base_spec.name}' ({base_dataset_spec_id}) has no data_sources_json. "
            f"Cannot create derived dataset without data source configuration."
        )

    # Build new spec_json with feature engineering
    base_spec_json = base_spec.spec_json or {}
    new_spec_json = dict(base_spec_json)

    # Merge feature engineering (add to existing)
    existing_features = new_spec_json.get("engineered_features", [])
    new_spec_json["engineered_features"] = existing_features + feature_engineering

    # Handle drop_columns by excluding them from feature_columns
    new_feature_columns = list(base_spec.feature_columns or [])
    if drop_columns:
        new_feature_columns = [c for c in new_feature_columns if c not in drop_columns]
        new_spec_json["dropped_columns"] = drop_columns

    # Track context usage
    new_spec_json["used_context"] = used_context

    # Build dataset name - add {CONTEXT} if context was used
    dataset_name = f"{base_spec.name} (Auto DS{name_suffix})"
    if used_context and "{CONTEXT}" not in dataset_name:
        dataset_name = f"{dataset_name} {{CONTEXT}}"

    # Create derived dataset spec
    derived_spec = DatasetSpec(
        project_id=session.project_id,
        name=dataset_name,
        description=f"Derived from {base_spec.name} by Auto DS session {session.name}",
        version=base_spec.version + 1,
        # CRITICAL: Copy data_sources_json for data loading
        data_sources_json=base_spec.data_sources_json,
        # CRITICAL: Set target_column
        target_column=base_spec.target_column,
        # Feature columns (possibly with some dropped for ablation)
        feature_columns=new_feature_columns if new_feature_columns else None,
        spec_json=new_spec_json,
        parent_dataset_spec_id=base_dataset_spec_id,
        lineage_json={
            "parent_id": str(base_dataset_spec_id),
            "derivation": "auto_ds_dynamic",
            "feature_engineering_count": len(feature_engineering),
            "dropped_columns": drop_columns or [],
        },
    )

    db.add(derived_spec)
    db.flush()  # Get the ID

    logger.info(
        f"Created derivative dataset spec {derived_spec.id} with "
        f"{len(feature_engineering)} engineered features, {len(drop_columns or [])} dropped columns"
    )

    return derived_spec


def _run_ensemble_analysis(
    db,
    session: AutoDSSession,
    iteration: AutoDSIteration,
    completed_experiments: List[Experiment],
) -> Optional[Dict[str, Any]]:
    """Build ensemble configuration from completed experiments.

    Since models are stored on Modal volumes (not locally), this function
    stores the ensemble configuration for later use by Modal-based prediction.
    The actual ensemble prediction is done by `predict_ensemble_remote` on Modal.

    Args:
        db: Database session
        session: Auto DS session
        iteration: Current iteration
        completed_experiments: List of completed experiments

    Returns:
        Ensemble config dict or None if not applicable
    """
    if not session.enable_ensemble:
        return None

    if len(completed_experiments) < 3:
        logger.info("Skipping ensemble - need at least 3 completed experiments")
        return None

    try:
        # Collect experiments with scores and Modal volume paths
        experiments_with_scores = []
        for exp in completed_experiments:
            score = _get_experiment_score(exp)
            if score is None:
                continue

            # Get volume model path from results_json (set by Modal training)
            results = exp.results_json or {}
            volume_path = results.get("volume_model_path")
            model_saved = results.get("model_saved_to_volume", False)

            if volume_path and model_saved:
                experiments_with_scores.append({
                    "experiment_id": str(exp.id),
                    "volume_model_path": volume_path,
                    "score": score,
                    "name": exp.name,
                    "best_model": results.get("best_model_name"),
                })

        if len(experiments_with_scores) < 3:
            logger.info(
                f"Not enough experiments with Modal models for ensemble "
                f"({len(experiments_with_scores)} found, need 3+)"
            )
            return None

        # Sort by score (higher is better) and take top 5
        experiments_with_scores.sort(key=lambda x: x["score"], reverse=True)
        top_experiments = experiments_with_scores[:5]

        # Calculate normalized weights based on scores
        total_score = sum(e["score"] for e in top_experiments)
        if total_score > 0:
            for exp in top_experiments:
                exp["weight"] = exp["score"] / total_score
        else:
            # Equal weights if all scores are 0
            for exp in top_experiments:
                exp["weight"] = 1.0 / len(top_experiments)

        logger.info(
            f"Created ensemble config with {len(top_experiments)} members, "
            f"scores: {[e['score'] for e in top_experiments]}"
        )

        # Return ensemble configuration for storage
        # This will be used by predict_ensemble_remote on Modal
        return {
            "member_count": len(top_experiments),
            "members": top_experiments,
            "member_ids": [e["experiment_id"] for e in top_experiments],
            "member_scores": [e["score"] for e in top_experiments],
            "member_weights": [e["weight"] for e in top_experiments],
            "method": "weighted_average",
            "ready_for_prediction": True,  # Flag that models are on Modal
        }

    except Exception as e:
        logger.error(f"Ensemble config creation failed: {e}")
        return None



def _get_robust_validation_config(session) -> dict:
    """Extract robust validation settings from Auto DS session for experiment plan."""
    validation_config = {}
    
    # Check if session has validation_strategy attribute
    if hasattr(session, 'validation_strategy') and session.validation_strategy:
        strategy = session.validation_strategy.value if hasattr(session.validation_strategy, 'value') else str(session.validation_strategy)
        validation_config['validation_strategy'] = strategy
        
        if strategy in ('ROBUST', 'STRICT'):
            validation_config['num_seeds'] = getattr(session, 'validation_num_seeds', 3)
            validation_config['cv_folds'] = getattr(session, 'validation_cv_folds', 5)
    
    return validation_config



class StopSessionException(Exception):
    """Raised when session stop is requested via stop flag."""
    pass


def _check_stop_requested(session_id: str) -> None:
    """Check if stop was requested and raise exception if so."""
    if check_stop_flag(session_id):
        raise StopSessionException(f'Stop requested for session {session_id}')


@celery_app.task(bind=True, name="app.tasks.auto_ds_tasks.run_auto_ds_session")
def run_auto_ds_session(
    self,
    session_id: str,
    initial_dataset_spec_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run an autonomous data science research session.

    This is the main entry point for Auto DS. It orchestrates the entire
    research loop:
    1. Run experiments on available datasets
    2. Analyze results to extract insights
    3. Plan next iteration based on insights
    4. Design new experiments
    5. Repeat until stopping conditions are met

    Args:
        session_id: UUID of the Auto DS session
        initial_dataset_spec_ids: Optional list of dataset spec IDs to start with

    Returns:
        Dict with session results summary
    """
    # Debug: Print immediately to verify task is being called
    print(f"🚀 AutoDS Task STARTED: session_id={session_id}")
    logger.info(f"🚀 AutoDS Task STARTED: session_id={session_id}")

    db = SessionLocal()
    try:
        session_uuid = UUID(session_id)

        # Get the session
        session = db.query(AutoDSSession).filter(AutoDSSession.id == session_uuid).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Update task ID (self.request.id for Celery, passed task_id for Modal)
        session.celery_task_id = getattr(getattr(self, 'request', None), 'id', None) or "modal-task"
        session.status = AutoDSSessionStatus.RUNNING
        session.started_at = datetime.utcnow()
        db.commit()

        logger.info(f"Starting Auto DS session {session_id}")

        orchestrator = AutoDSOrchestrator(db)

        # Main research loop
        while True:
            # Check stop flag in Redis (set when user clicks Stop button)
            if check_stop_flag(session_id):
                logger.info(f'Session {session_id} stopped via stop flag')
                session.status = AutoDSSessionStatus.STOPPED
                session.completed_at = datetime.utcnow()
                db.commit()
                clear_stop_flag(session_id)
                return {'status': 'stopped', 'session_id': session_id, 'reason': 'User requested stop'}

            # Check if session should stop
            db.refresh(session)
            should_stop, stop_reason = orchestrator.check_stopping_conditions(session)
            if should_stop:
                logger.info(f"Session {session_id} stopping: {stop_reason}")
                session.status = AutoDSSessionStatus.COMPLETED
                session.completed_at = datetime.utcnow()
                db.commit()
                break

            # Check if paused or stopped externally
            db.refresh(session)
            if session.status == AutoDSSessionStatus.PAUSED:
                logger.info(f"Session {session_id} paused")
                return {"status": "paused", "session_id": session_id}
            if session.status == AutoDSSessionStatus.STOPPED:
                logger.info(f"Session {session_id} stopped")
                return {"status": "stopped", "session_id": session_id}

            # Create new iteration
            iteration = orchestrator.create_iteration(session_uuid)
            logger.info(f"Starting iteration {iteration.iteration_number} for session {session_id}")

            try:
                # Phase 1: Check for pending AI-designed experiments from previous iterations
                # These are created by _create_next_iteration_experiments() but not yet run
                pending_ai_experiments = []
                if session.current_iteration > 1:
                    # Look for experiments created by previous iteration's AI analysis
                    all_pending = db.query(Experiment).filter(
                        Experiment.project_id == session.project_id,
                        Experiment.status == ExperimentStatus.PENDING,
                    ).all()

                    # Filter to only those belonging to this Auto DS session that don't have iteration assigned
                    for exp in all_pending:
                        if exp.experiment_plan_json:
                            plan = exp.experiment_plan_json
                            if (plan.get("auto_ds_session_id") == str(session.id) and
                                plan.get("iteration") is None):
                                pending_ai_experiments.append(exp)

                    logger.info(f"Found {len(pending_ai_experiments)} pending AI-designed experiments")

                # Phase 2: Run experiments
                completed_experiments = []

                if pending_ai_experiments:
                    # Run AI-designed experiments from previous analysis
                    logger.info(f"Running {len(pending_ai_experiments)} AI-designed experiments")
                    iteration.experiments_planned = len(pending_ai_experiments)
                    iteration.experiments_started_at = datetime.utcnow()
                    iteration.status = AutoDSIterationStatus.RUNNING_EXPERIMENTS
                    db.commit()

                    for exp in pending_ai_experiments:
                        try:
                            # Mark experiment with this iteration
                            exp.experiment_plan_json = {
                                **exp.experiment_plan_json,
                                "iteration": iteration.iteration_number,
                            }
                            db.commit()

                            # Run the experiment
                            _run_single_experiment(db, exp)

                            # Refresh and check status
                            db.refresh(exp)
                            if exp.status == ExperimentStatus.COMPLETED:
                                completed_experiments.append(exp)

                                # Record the result (this updates iteration and session counters)
                                orchestrator.record_experiment_result(
                                    iteration=iteration,
                                    experiment=exp,
                                    dataset_spec_id=exp.dataset_spec_id,
                                    variant=1,
                                    hypothesis=exp.experiment_plan_json.get("hypothesis"),
                                )
                            else:
                                iteration.experiments_failed += 1

                            db.commit()

                        except Exception as e:
                            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                            logger.error(f"AI-designed experiment failed: {error_msg}")
                            # Save error to experiment
                            exp.status = ExperimentStatus.FAILED
                            exp.error_message = error_msg
                            iteration.experiments_failed += 1
                            db.commit()
                else:
                    # Fall back to static baseline planning (iteration 1 or no AI experiments)
                    if session.current_iteration == 1 and initial_dataset_spec_ids:
                        # Use provided initial datasets
                        datasets = db.query(DatasetSpec).filter(
                            DatasetSpec.id.in_([UUID(id) for id in initial_dataset_spec_ids])
                        ).all()
                    else:
                        # Get active datasets from orchestrator
                        datasets = orchestrator.get_active_datasets(session)

                        # CRITICAL FIX: If no active datasets for iteration > 1, fall back to initial datasets
                        # This handles the case where AI-designed experiments from previous iteration all failed
                        if not datasets and initial_dataset_spec_ids:
                            logger.info(f"No active datasets, falling back to initial datasets for iteration {session.current_iteration}")
                            datasets = db.query(DatasetSpec).filter(
                                DatasetSpec.id.in_([UUID(id) for id in initial_dataset_spec_ids])
                            ).all()

                        # If still no datasets, try to find any dataset with completed experiments
                        if not datasets:
                            from sqlalchemy import and_
                            completed_exp = db.query(Experiment).filter(
                                and_(
                                    Experiment.project_id == session.project_id,
                                    Experiment.status == ExperimentStatus.COMPLETED,
                                    Experiment.dataset_spec_id.isnot(None),
                                )
                            ).first()
                            if completed_exp:
                                dataset = db.query(DatasetSpec).filter(
                                    DatasetSpec.id == completed_exp.dataset_spec_id
                                ).first()
                                if dataset:
                                    datasets = [dataset]
                                    logger.info(f"Using dataset from completed experiment as fallback")

                    if not datasets:
                        logger.warning(f"No datasets available for session {session_id}")
                        iteration.status = AutoDSIterationStatus.FAILED
                        iteration.error_message = "No datasets available"
                        db.commit()
                        break

                    # Plan baseline experiments for this iteration
                    experiment_plans = orchestrator.plan_experiments_for_iteration(
                        iteration=iteration,
                        datasets=datasets,
                    )

                    if not experiment_plans:
                        logger.warning(f"No experiments planned for iteration {iteration.iteration_number}")
                        iteration.status = AutoDSIterationStatus.COMPLETED
                        iteration.experiments_planned = 0
                        db.commit()
                        continue

                    iteration.experiments_planned = len(experiment_plans)
                    iteration.experiments_started_at = datetime.utcnow()
                    iteration.status = AutoDSIterationStatus.RUNNING_EXPERIMENTS
                    db.commit()

                    # Execute experiments based on execution mode
                    execution_mode = session.execution_mode
                    logger.info(f"Executing experiments using {execution_mode.value} mode")

                    if execution_mode == ExecutionMode.DYNAMIC:
                        completed_experiments = _execute_experiments_dynamic(
                            db=db,
                            session=session,
                            iteration=iteration,
                            experiment_plans=experiment_plans,
                            orchestrator=orchestrator,
                        )
                    elif execution_mode == ExecutionMode.ADAPTIVE:
                        completed_experiments = _execute_experiments_adaptive(
                            db=db,
                            session=session,
                            iteration=iteration,
                            experiment_plans=experiment_plans,
                            orchestrator=orchestrator,
                        )
                    elif execution_mode == ExecutionMode.PHASED:
                        completed_experiments = _execute_experiments_phased(
                            db=db,
                            session=session,
                            iteration=iteration,
                            experiment_plans=experiment_plans,
                            orchestrator=orchestrator,
                        )
                    else:
                        # Default to LEGACY mode
                        completed_experiments = _execute_experiments_legacy(
                            db=db,
                            session=session,
                            iteration=iteration,
                            experiment_plans=experiment_plans,
                            orchestrator=orchestrator,
                        )

                iteration.experiments_completed_at = datetime.utcnow()

                # Query failed experiments for this iteration to include in analysis
                failed_experiments = db.query(Experiment).filter(
                    Experiment.id.in_([ie.experiment_id for ie in iteration.iteration_experiments]),
                    Experiment.status == ExperimentStatus.FAILED
                ).all()

                if failed_experiments:
                    logger.info(f"Found {len(failed_experiments)} failed experiments to include in analysis feedback")

                # Phase 4: Analysis
                if completed_experiments or failed_experiments:
                    iteration.status = AutoDSIterationStatus.ANALYZING
                    iteration.analysis_started_at = datetime.utcnow()
                    db.commit()

                    # Run analysis using agents (async) - include both completed AND failed
                    analysis_results = asyncio.run(
                        run_analysis_phase(
                            db=db,
                            session=session,
                            iteration=iteration,
                            experiments=completed_experiments,
                            failed_experiments=failed_experiments,
                        )
                    )

                    iteration.analysis_summary_json = analysis_results.get("analysis_results")
                    iteration.analysis_completed_at = datetime.utcnow()

                    # Save insights from analysis
                    _save_insights_from_analysis(
                        db=db,
                        session=session,
                        iteration=iteration,
                        analysis_results=analysis_results.get("analysis_results", {}),
                    )

                    # Phase 4.5: Ensemble building (if enabled)
                    ensemble_info = _run_ensemble_analysis(
                        db=db,
                        session=session,
                        iteration=iteration,
                        completed_experiments=completed_experiments,
                    )
                    if ensemble_info:
                        iteration.analysis_summary_json = iteration.analysis_summary_json or {}
                        iteration.analysis_summary_json["ensemble"] = ensemble_info
                        logger.info(f"Built ensemble with {ensemble_info['member_count']} members")

                    # Phase 5: Strategy planning
                    iteration.status = AutoDSIterationStatus.STRATEGIZING
                    iteration.strategy_started_at = datetime.utcnow()
                    db.commit()

                    strategy = analysis_results.get("strategy")
                    iteration.strategy_decisions_json = strategy
                    iteration.strategy_completed_at = datetime.utcnow()

                    # If strategy says to stop, mark session as complete
                    if strategy and not strategy.get("should_continue", True):
                        logger.info(f"Strategy recommends stopping: {strategy.get('stop_reason')}")
                        session.status = AutoDSSessionStatus.COMPLETED
                        session.completed_at = datetime.utcnow()

                    # Phase 6: Create new datasets/experiments for next iteration
                    experiment_designs = analysis_results.get("experiment_designs")
                    print(f"🔬 Phase 6: experiment_designs={experiment_designs is not None}, should_continue={strategy.get('should_continue', True)}")
                    logger.info(f"🔬 Phase 6: experiment_designs={experiment_designs is not None}, should_continue={strategy.get('should_continue', True)}")
                    if experiment_designs:
                        exp_list = experiment_designs.get("experiments", [])
                        print(f"🔬 Phase 6: {len(exp_list)} experiments in design")
                        logger.info(f"🔬 Phase 6: {len(exp_list)} experiments in design")
                    if experiment_designs and strategy.get("should_continue", True):
                        print(f"🔬 Phase 6: Calling _create_next_iteration_experiments...")
                        logger.info(f"🔬 Phase 6: Calling _create_next_iteration_experiments...")
                        creation_result = _create_next_iteration_experiments(
                            db=db,
                            session=session,
                            experiment_designs=experiment_designs,
                        )
                        print(f"🔬 Phase 6: _create_next_iteration_experiments completed")
                        logger.info(f"🔬 Phase 6: _create_next_iteration_experiments completed")

                        # Store validation failures in iteration for next iteration's agents
                        if creation_result and creation_result.get("validation_failures"):
                            if iteration.analysis_summary_json is None:
                                iteration.analysis_summary_json = {}
                            iteration.analysis_summary_json["formula_validation_failures"] = creation_result["validation_failures"]
                            logger.info(
                                f"Stored {len(creation_result['validation_failures'])} formula validation failures "
                                f"for next iteration's agents"
                            )
                            db.commit()
                    else:
                        print(f"🔬 Phase 6: SKIPPED - no experiment_designs or should_continue=False")
                        logger.warning(f"🔬 Phase 6: SKIPPED - no experiment_designs or should_continue=False")

                # Complete the iteration
                orchestrator.complete_iteration(iteration=iteration)

                db.commit()

            except StopSessionException:
                # Stop was requested - exit the main loop
                logger.info(f"Session {session.id} stopped during iteration (user request)")
                session.status = AutoDSSessionStatus.STOPPED
                session.completed_at = datetime.utcnow()
                db.commit()
                clear_stop_flag(str(session.id))
                return {"status": "stopped", "session_id": session_id, "reason": "User requested stop"}
            except Exception as e:
                logger.error(f"Iteration {iteration.iteration_number} failed: {e}")
                iteration.status = AutoDSIterationStatus.FAILED
                iteration.error_message = str(e)
                db.commit()
                # Continue to next iteration, don't fail the whole session
                continue

        # Session complete - get summary
        db.refresh(session)
        return {
            "status": session.status.value,
            "session_id": session_id,
            "iterations_completed": session.current_iteration,
            "total_experiments": session.total_experiments_run,
            "best_score": session.best_score,
            "best_experiment_id": str(session.best_experiment_id) if session.best_experiment_id else None,
        }

    except Exception as e:
        logger.error(f"Auto DS session {session_id} failed: {e}")
        # Mark session as failed
        try:
            session = db.query(AutoDSSession).filter(AutoDSSession.id == UUID(session_id)).first()
            if session:
                session.status = AutoDSSessionStatus.FAILED
                session.completed_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
        raise

    finally:
        db.close()


def _run_single_experiment(db, experiment: Experiment) -> None:
    """Run a single experiment using existing infrastructure.

    This is a simplified version that runs synchronously for now.
    In production, this would use the Modal/AutoML pipeline.
    """
    from app.tasks.experiment_tasks import run_experiment_modal

    # Debug logging
    print(f"🧪 _run_single_experiment: Starting experiment {experiment.id} - {experiment.name}")
    logger.info(f"🧪 _run_single_experiment: Starting experiment {experiment.id} - {experiment.name}")

    # Mark as running
    experiment.status = ExperimentStatus.RUNNING
    db.commit()

    try:
        print(f"🧪 _run_single_experiment: Calling run_experiment_modal for {experiment.id}")
        # Call the task function directly (works in both Celery and Modal contexts)
        result = run_experiment_modal(None, str(experiment.id))  # self=None since task doesn't use it

    except Exception as e:
        # IMPORTANT: Refresh experiment from DB first - run_experiment_modal may have
        # already saved a detailed error message in its own db session
        db.refresh(experiment)

        # Only set error_message if not already set by the inner task
        if not experiment.error_message:
            experiment.error_message = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"

        experiment.status = ExperimentStatus.FAILED
        db.commit()
        raise


def _save_insights_from_analysis(
    db,
    session: AutoDSSession,
    iteration: AutoDSIteration,
    analysis_results: Dict[str, Any],
) -> None:
    """Save insights extracted from cross-analysis to the database."""
    insights = analysis_results.get("insights", [])

    for ins_data in insights:
        insight = ResearchInsight(
            session_id=session.id,
            project_id=session.project_id,
            iteration_id=iteration.id,
            insight_type=_map_insight_type(ins_data.get("type", "other")),
            confidence=_map_confidence(ins_data.get("confidence", "low")),
            title=ins_data.get("title", "Untitled Insight"),
            description=ins_data.get("description"),
            insight_data_json={
                "recommendation": ins_data.get("recommendation"),
                "evidence": ins_data.get("evidence", []),
                "contradictions": ins_data.get("contradictions", []),
            },
            evidence_count=len(ins_data.get("evidence", [])),
            supporting_experiments=ins_data.get("evidence"),
            contradicting_experiments=ins_data.get("contradictions"),
        )
        db.add(insight)

    db.commit()


def _map_insight_type(type_str: str) -> InsightType:
    """Map string insight type to enum."""
    mapping = {
        "feature_importance": InsightType.FEATURE_IMPORTANCE,
        "feature_pattern": InsightType.FEATURE_PATTERN,
        "split_strategy": InsightType.SPLIT_STRATEGY,
        "model_config": InsightType.MODEL_CONFIG,
        "data_quality": InsightType.DATA_QUALITY,
        "target_insight": InsightType.TARGET_INSIGHT,
        "pitfall": InsightType.PITFALL,
        "hypothesis": InsightType.HYPOTHESIS,
        # Legacy mappings for backwards compatibility with AI responses
        "model_performance": InsightType.MODEL_CONFIG,
        "preprocessing": InsightType.FEATURE_PATTERN,
        "overfitting_pattern": InsightType.PITFALL,
        "hyperparameter": InsightType.MODEL_CONFIG,
        "interaction": InsightType.FEATURE_PATTERN,
    }
    return mapping.get(type_str, InsightType.GENERAL)


def _map_confidence(conf_str: str) -> InsightConfidence:
    """Map string confidence to enum."""
    mapping = {
        "high": InsightConfidence.HIGH,
        "medium": InsightConfidence.MEDIUM,
        "low": InsightConfidence.LOW,
    }
    return mapping.get(conf_str, InsightConfidence.LOW)


def _execute_experiments_legacy(
    db,
    session: AutoDSSession,
    iteration: AutoDSIteration,
    experiment_plans: List[Dict[str, Any]],
    orchestrator: AutoDSOrchestrator,
) -> List[Experiment]:
    """Execute experiments using legacy mode - run all experiments sequentially.

    This is the original behavior: plan all experiments upfront, run them all,
    then analyze at the end.

    Args:
        db: Database session
        session: Auto DS session
        iteration: Current iteration
        experiment_plans: List of experiment plan dictionaries
        orchestrator: AutoDSOrchestrator instance

    Returns:
        List of completed experiments
    """
    completed_experiments = []

    for exp_plan in experiment_plans:
        try:
            # Validate dataset has data_sources_json before creating experiment
            dataset_spec_id = UUID(exp_plan["dataset_spec_id"])
            dataset = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
            if not dataset or not dataset.data_sources_json:
                logger.error(
                    f"Legacy mode: dataset {exp_plan.get('dataset_name', dataset_spec_id)} "
                    f"has no data_sources_json, skipping experiment"
                )
                iteration.experiments_failed += 1
                continue

            # Create actual Experiment object from plan
            exp = Experiment(
                project_id=session.project_id,
                dataset_spec_id=dataset_spec_id,
                name=f"[{session.name}] {exp_plan['dataset_name']} v{exp_plan['variant']}",
                description=exp_plan.get("hypothesis"),
                status=ExperimentStatus.PENDING,
                experiment_plan_json={
                    "hypothesis": exp_plan.get("hypothesis"),
                    "config": exp_plan.get("config", {}),
                    "auto_ds_session_id": str(session.id),
                    "iteration": iteration.iteration_number,
                    "robust_validation": _get_robust_validation_config(session),
                },
            )
            db.add(exp)
            db.flush()

            # Check stop flag before running experiment
            _check_stop_requested(str(session.id))

            # Run the experiment using existing infrastructure
            _run_single_experiment(db, exp)

            # Refresh and check status
            db.refresh(exp)
            if exp.status == ExperimentStatus.COMPLETED:
                completed_experiments.append(exp)

                # Record the result (this updates iteration and session counters)
                orchestrator.record_experiment_result(
                    iteration=iteration,
                    experiment=exp,
                    dataset_spec_id=UUID(exp_plan["dataset_spec_id"]),
                    variant=exp_plan["variant"],
                    hypothesis=exp_plan.get("hypothesis"),
                )
            else:
                iteration.experiments_failed += 1

            db.commit()

        except Exception as e:
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Experiment failed: {error_msg}")
            # Save error to experiment if it was created
            if 'exp' in locals():
                exp.status = ExperimentStatus.FAILED
                exp.error_message = error_msg
            iteration.experiments_failed += 1
            db.commit()

    return completed_experiments


def _execute_experiments_adaptive(
    db,
    session: AutoDSSession,
    iteration: AutoDSIteration,
    experiment_plans: List[Dict[str, Any]],
    orchestrator: AutoDSOrchestrator,
) -> List[Experiment]:
    """Execute experiments using adaptive mode - early stopping on score decline.

    After each experiment completes, check if the score is declining.
    If score drops below best_score * (1 - adaptive_decline_threshold),
    skip remaining experiments to save compute resources.

    Args:
        db: Database session
        session: Auto DS session
        iteration: Current iteration
        experiment_plans: List of experiment plan dictionaries
        orchestrator: AutoDSOrchestrator instance

    Returns:
        List of completed experiments
    """
    completed_experiments = []
    best_score_this_iteration = None
    decline_threshold = session.adaptive_decline_threshold
    experiments_skipped = 0

    logger.info(f"Adaptive mode: decline threshold = {decline_threshold * 100:.1f}%")

    for i, exp_plan in enumerate(experiment_plans):
        try:
            # Validate dataset has data_sources_json before creating experiment
            dataset_spec_id = UUID(exp_plan["dataset_spec_id"])
            dataset = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
            if not dataset or not dataset.data_sources_json:
                logger.error(
                    f"Adaptive mode: dataset {exp_plan.get('dataset_name', dataset_spec_id)} "
                    f"has no data_sources_json, skipping experiment"
                )
                iteration.experiments_failed += 1
                continue

            # Create actual Experiment object from plan
            exp = Experiment(
                project_id=session.project_id,
                dataset_spec_id=dataset_spec_id,
                name=f"[{session.name}] {exp_plan['dataset_name']} v{exp_plan['variant']}",
                description=exp_plan.get("hypothesis"),
                status=ExperimentStatus.PENDING,
                experiment_plan_json={
                    "hypothesis": exp_plan.get("hypothesis"),
                    "config": exp_plan.get("config", {}),
                    "auto_ds_session_id": str(session.id),
                    "iteration": iteration.iteration_number,
                    "execution_mode": "adaptive",
                    "robust_validation": _get_robust_validation_config(session),
                },
            )
            db.add(exp)
            db.flush()

            # Check stop flag before running experiment
            _check_stop_requested(str(session.id))

            # Run the experiment using existing infrastructure
            _run_single_experiment(db, exp)

            # Refresh and check status
            db.refresh(exp)
            if exp.status == ExperimentStatus.COMPLETED:
                completed_experiments.append(exp)

                # Record the result (this updates iteration and session counters)
                result_info = orchestrator.record_experiment_result(
                    iteration=iteration,
                    experiment=exp,
                    dataset_spec_id=UUID(exp_plan["dataset_spec_id"]),
                    variant=exp_plan["variant"],
                    hypothesis=exp_plan.get("hypothesis"),
                )

                # Get the score from the experiment
                current_score = _get_experiment_score(exp)
                metric_direction = _get_metric_direction(exp)

                if current_score is not None:
                    # Update best score if this is the first or if it's better
                    if best_score_this_iteration is None or _is_better_score(current_score, best_score_this_iteration, metric_direction):
                        best_score_this_iteration = current_score
                        logger.info(f"Adaptive mode: new best score = {best_score_this_iteration:.4f} (direction={metric_direction})")
                    else:
                        # Check for decline (direction-aware)
                        if metric_direction == "minimize":
                            # For minimize: threshold is best + decline% (higher error = worse)
                            threshold_score = best_score_this_iteration * (1 + decline_threshold)
                            is_declining = current_score > threshold_score
                        else:
                            # For maximize: threshold is best - decline% (lower score = worse)
                            threshold_score = best_score_this_iteration * (1 - decline_threshold)
                            is_declining = current_score < threshold_score

                        if is_declining:
                            remaining = len(experiment_plans) - i - 1
                            logger.warning(
                                f"Adaptive mode: score {current_score:.4f} beyond threshold "
                                f"{threshold_score:.4f} ({metric_direction}), skipping {remaining} remaining experiments"
                            )
                            experiments_skipped = remaining
                            break
            else:
                iteration.experiments_failed += 1

            db.commit()

        except Exception as e:
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Experiment failed: {error_msg}")
            # Save error to experiment if it was created
            if 'exp' in locals():
                exp.status = ExperimentStatus.FAILED
                exp.error_message = error_msg
            iteration.experiments_failed += 1
            db.commit()

    if experiments_skipped > 0:
        logger.info(f"Adaptive mode: skipped {experiments_skipped} experiments due to declining scores")
        # Update the planned count to reflect actual execution
        iteration.analysis_summary_json = iteration.analysis_summary_json or {}
        iteration.analysis_summary_json["adaptive_skipped"] = experiments_skipped

    return completed_experiments


def _execute_experiments_phased(
    db,
    session: AutoDSSession,
    iteration: AutoDSIteration,
    experiment_plans: List[Dict[str, Any]],
    orchestrator: AutoDSOrchestrator,
) -> List[Experiment]:
    """Execute experiments using phased mode - baseline first, then targeted variants.

    Phase 1: Run only baseline (variant 1) experiments for each dataset
    Phase 2: Analyze baseline results to identify promising datasets
    Phase 3: Run additional variants only for datasets that show improvement

    Args:
        db: Database session
        session: Auto DS session
        iteration: Current iteration
        experiment_plans: List of experiment plan dictionaries
        orchestrator: AutoDSOrchestrator instance

    Returns:
        List of completed experiments
    """
    completed_experiments = []
    min_improvement = session.phased_min_baseline_improvement

    # Separate baseline (variant 1) and variant experiments
    baseline_plans = [p for p in experiment_plans if p.get("variant", 1) == 1]
    variant_plans = [p for p in experiment_plans if p.get("variant", 1) > 1]

    logger.info(
        f"Phased mode: {len(baseline_plans)} baselines, {len(variant_plans)} variants, "
        f"min improvement threshold = {min_improvement * 100:.1f}%"
    )

    # Phase 1: Run baseline experiments
    logger.info("Phased mode Phase 1: Running baseline experiments")
    baseline_scores = {}  # dataset_spec_id -> (score, metric_direction)

    for exp_plan in baseline_plans:
        try:
            # Validate dataset has data_sources_json before creating experiment
            dataset_spec_id = UUID(exp_plan["dataset_spec_id"])
            dataset = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
            if not dataset or not dataset.data_sources_json:
                logger.error(
                    f"Phased mode: baseline dataset {exp_plan.get('dataset_name', dataset_spec_id)} "
                    f"has no data_sources_json, skipping experiment"
                )
                iteration.experiments_failed += 1
                continue

            exp = Experiment(
                project_id=session.project_id,
                dataset_spec_id=dataset_spec_id,
                name=f"[{session.name}] {exp_plan['dataset_name']} v{exp_plan['variant']}",
                description=exp_plan.get("hypothesis"),
                status=ExperimentStatus.PENDING,
                experiment_plan_json={
                    "hypothesis": exp_plan.get("hypothesis"),
                    "config": exp_plan.get("config", {}),
                    "auto_ds_session_id": str(session.id),
                    "iteration": iteration.iteration_number,
                    "execution_mode": "phased",
                    "robust_validation": _get_robust_validation_config(session),
                    "phase": "baseline",
                },
            )
            db.add(exp)
            db.flush()

            # Check stop flag before running experiment
            _check_stop_requested(str(session.id))

            _run_single_experiment(db, exp)

            db.refresh(exp)
            if exp.status == ExperimentStatus.COMPLETED:
                completed_experiments.append(exp)

                orchestrator.record_experiment_result(
                    iteration=iteration,
                    experiment=exp,
                    dataset_spec_id=UUID(exp_plan["dataset_spec_id"]),
                    variant=exp_plan["variant"],
                    hypothesis=exp_plan.get("hypothesis"),
                )

                # Store baseline score and metric direction for this dataset
                score = _get_experiment_score(exp)
                if score is not None:
                    direction = _get_metric_direction(exp)
                    baseline_scores[exp_plan["dataset_spec_id"]] = (score, direction)
                    logger.info(f"Phased mode: baseline for {exp_plan['dataset_name']} = {score:.4f} ({direction})")
            else:
                iteration.experiments_failed += 1

            db.commit()

        except Exception as e:
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Baseline experiment failed: {error_msg}")
            # Save error to experiment if it was created
            if 'exp' in locals():
                exp.status = ExperimentStatus.FAILED
                exp.error_message = error_msg
            iteration.experiments_failed += 1
            db.commit()

    # Phase 2: Determine which datasets are promising
    # A dataset is promising if its baseline score shows improvement over session's best
    # or if this is the first iteration
    promising_datasets = set()
    session_best = session.best_score or 0.0

    for dataset_id, (score, direction) in baseline_scores.items():
        # Calculate improvement respecting metric direction
        if direction == "minimize":
            # For minimize: improvement is how much lower (better) the score is
            # e.g., RMSE 0.2 vs session_best 0.25 -> improvement = (0.25 - 0.2) / 0.25 = 0.2 (20% better)
            improvement = (session_best - score) / max(session_best, 0.0001) if session_best > 0 else 0.0
        else:
            # For maximize: improvement is how much higher (better) the score is
            improvement = (score - session_best) / max(session_best, 0.0001)
        
        if session.current_iteration == 1 or improvement >= min_improvement:
            promising_datasets.add(dataset_id)
            logger.info(f"Phased mode: dataset {dataset_id} is promising (improvement: {improvement:.2%}, {direction})")
        else:
            logger.info(f"Phased mode: skipping variants for dataset {dataset_id} (improvement: {improvement:.2%} < {min_improvement:.2%}, {direction})")

    # Phase 3: Run variants only for promising datasets
    filtered_variant_plans = [
        p for p in variant_plans
        if p["dataset_spec_id"] in promising_datasets
    ]
    skipped_variants = len(variant_plans) - len(filtered_variant_plans)

    logger.info(
        f"Phased mode Phase 3: Running {len(filtered_variant_plans)} targeted variants "
        f"(skipped {skipped_variants} for non-promising datasets)"
    )

    for exp_plan in filtered_variant_plans:
        try:
            # Validate dataset has data_sources_json before creating experiment
            dataset_spec_id = UUID(exp_plan["dataset_spec_id"])
            dataset = db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
            if not dataset or not dataset.data_sources_json:
                logger.error(
                    f"Phased mode: variant dataset {exp_plan.get('dataset_name', dataset_spec_id)} "
                    f"has no data_sources_json, skipping experiment"
                )
                iteration.experiments_failed += 1
                continue

            exp = Experiment(
                project_id=session.project_id,
                dataset_spec_id=dataset_spec_id,
                name=f"[{session.name}] {exp_plan['dataset_name']} v{exp_plan['variant']}",
                description=exp_plan.get("hypothesis"),
                status=ExperimentStatus.PENDING,
                experiment_plan_json={
                    "hypothesis": exp_plan.get("hypothesis"),
                    "config": exp_plan.get("config", {}),
                    "auto_ds_session_id": str(session.id),
                    "iteration": iteration.iteration_number,
                    "execution_mode": "phased",
                    "robust_validation": _get_robust_validation_config(session),
                    "phase": "variant",
                },
            )
            db.add(exp)
            db.flush()

            # Check stop flag before running experiment
            _check_stop_requested(str(session.id))

            _run_single_experiment(db, exp)

            db.refresh(exp)
            if exp.status == ExperimentStatus.COMPLETED:
                completed_experiments.append(exp)

                orchestrator.record_experiment_result(
                    iteration=iteration,
                    experiment=exp,
                    dataset_spec_id=UUID(exp_plan["dataset_spec_id"]),
                    variant=exp_plan["variant"],
                    hypothesis=exp_plan.get("hypothesis"),
                )
            else:
                iteration.experiments_failed += 1

            db.commit()

        except Exception as e:
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Variant experiment failed: {error_msg}")
            # Save error to experiment if it was created
            if 'exp' in locals():
                exp.status = ExperimentStatus.FAILED
                exp.error_message = error_msg
            iteration.experiments_failed += 1
            db.commit()

    # Record phased execution summary
    iteration.analysis_summary_json = iteration.analysis_summary_json or {}
    iteration.analysis_summary_json["phased_summary"] = {
        "baselines_run": len(baseline_plans),
        "promising_datasets": len(promising_datasets),
        "variants_run": len(filtered_variant_plans),
        "variants_skipped": skipped_variants,
    }

    return completed_experiments


# Metrics where lower is better (error metrics)
MINIMIZE_METRICS = {
    "rmse", "root_mean_squared_error",
    "mse", "mean_squared_error",
    "mae", "mean_absolute_error",
    "mape", "mean_absolute_percentage_error",
    "log_loss", "logloss",
    "pinball_loss",
}


def _get_metric_direction(exp: Experiment) -> str:
    """Determine if metric should be minimized or maximized.

    Args:
        exp: The experiment

    Returns:
        'minimize' or 'maximize'
    """
    # First check if experiment has explicit metric_direction
    if exp.metric_direction:
        return exp.metric_direction.value if hasattr(exp.metric_direction, 'value') else str(exp.metric_direction)

    # Check if we can infer from primary_metric or results
    metric_name = None
    if exp.primary_metric:
        metric_name = exp.primary_metric.lower()
    elif exp.results_json:
        # Try to get metric name from results
        metric_name = exp.results_json.get("eval_metric", "").lower()
        if not metric_name:
            metric_name = exp.results_json.get("primary_metric", "").lower()

    # If we have a metric name, check if it's a "lower is better" metric
    if metric_name and any(m in metric_name for m in MINIMIZE_METRICS):
        return "minimize"

    # Default to maximize (accuracy, AUC, F1, etc.)
    return "maximize"


def _is_better_score(new_score: float, old_score: float, direction: str) -> bool:
    """Compare scores respecting metric direction.

    Args:
        new_score: The new score to compare
        old_score: The previous best score
        direction: 'minimize' or 'maximize'

    Returns:
        True if new_score is better than old_score
    """
    if direction == "minimize":
        return new_score < old_score
    return new_score > old_score


def _get_experiment_score(exp: Experiment) -> Optional[float]:
    """Extract the primary score from an experiment.

    Prefers holdout > validation > train score.

    Args:
        exp: The experiment to get score from

    Returns:
        The primary score, or None if no score available
    """
    # Try to get score from experiment results
    if exp.results_json:
        # Prefer holdout, then validation, then train
        for key in ["score_holdout", "holdout_score", "score_val", "val_score", "validation_score",
                    "score_train", "train_score", "score", "auc_roc", "auc"]:
            if key in exp.results_json and exp.results_json[key] is not None:
                return float(exp.results_json[key])

    return None


def _create_next_iteration_experiments(
    db,
    session: AutoDSSession,
    experiment_designs: Dict[str, Any],
) -> Dict[str, Any]:
    """Create dataset specs and experiments for the next iteration.

    Takes the output of the ExperimentOrchestratorAgent and creates
    the actual database objects.

    Returns:
        Dict containing:
        - created_count: Number of experiments successfully created
        - skipped_count: Number of experiments skipped due to validation
        - validation_failures: List of validation failure dicts for agent feedback
    """
    experiments_data = experiment_designs.get("experiments", [])
    created_count = 0
    skipped_count = 0
    all_validation_failures = []

    for exp_design in experiments_data:
        try:
            dataset_spec_data = exp_design.get("dataset_spec", {})

            # Create or reference dataset spec
            if dataset_spec_data.get("create_new", True):
                # Get base dataset for copying essential fields (data_sources_json, etc.)
                base_dataset_id = dataset_spec_data.get("base_dataset_id")

                # CRITICAL: base_dataset_id is REQUIRED - fail if not provided
                if not base_dataset_id:
                    exp_name = exp_design.get("name", "Unknown")
                    logger.error(
                        f"Experiment '{exp_name}' missing required base_dataset_id. "
                        f"AI must specify which dataset to inherit data sources from. "
                        f"Skipping this experiment."
                    )
                    continue

                # Look up the base dataset
                base_dataset = db.query(DatasetSpec).filter(
                    DatasetSpec.id == UUID(base_dataset_id)
                ).first()

                if not base_dataset:
                    exp_name = exp_design.get("name", "Unknown")
                    logger.error(
                        f"Experiment '{exp_name}' references non-existent base_dataset_id: {base_dataset_id}. "
                        f"Skipping this experiment."
                    )
                    continue

                # CRITICAL: base dataset must have data sources for training to work
                if not base_dataset.data_sources_json:
                    exp_name = exp_design.get("name", "Unknown")
                    logger.error(
                        f"Experiment '{exp_name}' base dataset '{base_dataset.name}' ({base_dataset_id}) "
                        f"has no data_sources_json configured. Cannot create derivative. "
                        f"Skipping this experiment."
                    )
                    continue

                # Extract feature column names (handle both string and dict formats)
                features_raw = dataset_spec_data.get("features", [])
                feature_columns = []
                for f in features_raw:
                    if isinstance(f, str):
                        feature_columns.append(f)
                    elif isinstance(f, dict) and f.get("name"):
                        feature_columns.append(f["name"])

                # Get target column - prefer explicit, fall back to base dataset
                target_column = dataset_spec_data.get("target")
                if not target_column and base_dataset:
                    target_column = base_dataset.target_column

                # If no features specified, inherit from base dataset
                if not feature_columns and base_dataset:
                    feature_columns = list(base_dataset.feature_columns or [])

                # FINAL VALIDATION: Ensure data_sources_json will be valid
                # This is a defensive check in case base_dataset.data_sources_json is null
                if not base_dataset or not base_dataset.data_sources_json:
                    exp_name = exp_design.get("name", "Unknown")
                    logger.error(
                        f"DEFENSIVE CHECK: Cannot create dataset for '{exp_name}' - "
                        f"base_dataset or data_sources_json is null. Skipping."
                    )
                    continue

                # PRE-VALIDATE feature engineering formulas before creating experiment
                feature_engineering = dataset_spec_data.get("feature_engineering", [])
                if feature_engineering:
                    try:
                        # Load a sample of the RAW data for validation
                        # We need to load data without applying base dataset's feature engineering
                        # so we can validate the new formulas against raw columns
                        from app.services.dataset_builder import DatasetBuilder
                        from app.models.data_source import DataSource
                        import pandas as pd

                        sample_df = None

                        # Get data source from base dataset
                        if base_dataset.data_sources_json:
                            sources = base_dataset.data_sources_json
                            source_id = None
                            if isinstance(sources, list) and sources:
                                first = sources[0]
                                source_id = first.get("source_id") or first.get("id") if isinstance(first, dict) else first
                            elif isinstance(sources, dict):
                                source_id = sources.get("source_id") or sources.get("id") or sources.get("primary")

                            if source_id:
                                data_source = db.query(DataSource).filter(DataSource.id == source_id).first()
                                if data_source and data_source.file_path:
                                    from pathlib import Path
                                    file_path = Path(data_source.file_path)
                                    if file_path.exists():
                                        # Load sample based on file type
                                        if str(file_path).endswith('.csv'):
                                            sample_df = pd.read_csv(file_path, nrows=100)
                                        elif str(file_path).endswith('.parquet'):
                                            sample_df = pd.read_parquet(file_path).head(100)
                                        elif str(file_path).endswith(('.xlsx', '.xls')):
                                            sample_df = pd.read_excel(file_path, nrows=100)

                        if sample_df is not None and len(sample_df) > 0:
                            # Validate all formulas
                            valid_features, failures = validate_feature_engineering_batch(
                                sample_df, feature_engineering
                            )

                            if failures:
                                exp_name = exp_design.get("name", "Unknown")
                                feedback = get_validation_feedback_for_agent(failures)
                                logger.error(
                                    f"FORMULA VALIDATION FAILED for experiment '{exp_name}': "
                                    f"{len(failures)} formula(s) failed validation.\n{feedback}"
                                )

                                # Collect validation failures for agent feedback
                                for failure in failures:
                                    all_validation_failures.append({
                                        "experiment_name": exp_name,
                                        "output_column": failure.output_column,
                                        "formula": failure.formula[:200],  # Truncate long formulas
                                        "error_type": failure.error_type,
                                        "error_message": failure.error_message,
                                        "suggested_fix": failure.suggested_fix,
                                    })

                                # Skip this experiment - formulas won't work
                                skipped_count += 1
                                logger.warning(
                                    f"Skipping experiment '{exp_name}' due to {len(failures)} invalid formulas. "
                                    f"Agent will receive feedback to fix formulas in next iteration."
                                )
                                continue

                            # Use only validated features
                            feature_engineering = valid_features
                            logger.info(
                                f"✓ Formula validation passed for experiment '{exp_design.get('name', 'Unknown')}': "
                                f"{len(valid_features)} features validated"
                            )
                    except Exception as e:
                        # If we can't load data for validation, log warning but continue
                        # The actual execution will catch errors
                        logger.warning(
                            f"Could not pre-validate formulas (will validate at execution): {e}"
                        )

                # Determine dataset name - add {CONTEXT} if context was used
                dataset_name = dataset_spec_data.get("name", f"Auto DS Dataset {session.current_iteration}")
                used_context = exp_design.get("used_context", False)
                if used_context and "{CONTEXT}" not in dataset_name:
                    dataset_name = f"{dataset_name} {{CONTEXT}}"

                # Create new dataset spec with all required fields
                spec = DatasetSpec(
                    project_id=session.project_id,
                    name=dataset_name,
                    description=f"Created by Auto DS iteration {session.current_iteration}",
                    version=1,
                    # CRITICAL: Copy data_sources_json from base dataset for data loading
                    data_sources_json=base_dataset.data_sources_json,
                    # CRITICAL: Set target_column for experiment execution
                    target_column=target_column,
                    # CRITICAL: Set feature_columns so AI's feature selections are used
                    feature_columns=feature_columns if feature_columns else None,
                    spec_json={
                        "features": dataset_spec_data.get("features", []),
                        "target": target_column,
                        "engineered_features": feature_engineering,  # Use validated features
                        "preprocessing": dataset_spec_data.get("preprocessing", {}),
                        "used_context": used_context,  # Track if context documents were used
                    },
                )

                # Set parent lineage if deriving from existing
                if base_dataset_id:
                    spec.parent_dataset_spec_id = UUID(base_dataset_id)
                    spec.lineage_json = {
                        "parent_id": base_dataset_id,
                        "derivation": "auto_ds_iteration",
                        "changes": dataset_spec_data.get("changes", ""),
                    }

                db.add(spec)
                db.flush()  # Get the ID

                dataset_spec_id = spec.id
            else:
                # Use existing dataset spec
                dataset_spec_id = UUID(dataset_spec_data.get("base_dataset_id"))

            # Create experiment
            automl_config = exp_design.get("automl_config", {})
            experiment = Experiment(
                project_id=session.project_id,
                dataset_spec_id=dataset_spec_id,
                name=exp_design.get("name", f"[{session.name}] Experiment {session.current_iteration}"),
                description=exp_design.get("description"),
                status=ExperimentStatus.PENDING,
                experiment_plan_json={
                    "hypothesis": exp_design.get("hypothesis"),
                    "success_criteria": exp_design.get("success_criteria"),
                    "automl_config": automl_config,
                    "auto_ds_session_id": str(session.id),
                },
            )
            db.add(experiment)
            created_count += 1

        except Exception as e:
            logger.error(f"Failed to create experiment from design: {e}")
            skipped_count += 1
            continue

    db.commit()

    # Log summary
    if all_validation_failures:
        logger.warning(
            f"Experiment creation summary: {created_count} created, {skipped_count} skipped "
            f"({len(all_validation_failures)} formula validation errors)"
        )
    else:
        logger.info(f"Experiment creation summary: {created_count} created, {skipped_count} skipped")

    return {
        "created_count": created_count,
        "skipped_count": skipped_count,
        "validation_failures": all_validation_failures,
    }


def _execute_experiments_dynamic(
    db,
    session: AutoDSSession,
    iteration: AutoDSIteration,
    experiment_plans: List[Dict[str, Any]],
    orchestrator: AutoDSOrchestrator,
) -> List[Experiment]:
    """Execute experiments using dynamic mode - AI designs each experiment based on results.

    This is the most sophisticated execution mode:
    1. Run initial experiments (from experiment_plans) in cycles
    2. After each cycle, call DynamicPlanningAgent to analyze results
    3. Agent designs next experiment(s) based on all accumulated knowledge
    4. Continue until agent recommends stopping or max experiments reached

    This enables true adaptive exploration where each experiment is informed by
    all previous results, leading to faster convergence on optimal solutions.

    Args:
        db: Database session
        session: Auto DS session
        iteration: Current iteration
        experiment_plans: Initial experiment plan dictionaries (used for first cycle)
        orchestrator: AutoDSOrchestrator instance

    Returns:
        List of completed experiments
    """
    from app.services.auto_ds_agents import DynamicPlanningAgent

    completed_experiments = []
    experiments_per_cycle = session.dynamic_experiments_per_cycle
    max_experiments_this_iteration = session.max_experiments_per_dataset * session.max_active_datasets
    cycle_number = 0

    logger.info(
        f"Dynamic mode: {experiments_per_cycle} experiments per cycle, "
        f"max {max_experiments_this_iteration} total"
    )

    # Initialize with provided experiment plans for first cycle
    current_plans = experiment_plans[:experiments_per_cycle] if experiment_plans else []
    remaining_initial_plans = experiment_plans[experiments_per_cycle:] if len(experiment_plans) > experiments_per_cycle else []

    while len(completed_experiments) < max_experiments_this_iteration:
        cycle_number += 1
        logger.info(f"Dynamic mode: starting cycle {cycle_number}")

        if not current_plans:
            logger.info("Dynamic mode: no more experiment plans, ending iteration")
            break

        # Run experiments in this cycle
        cycle_experiments = []
        for exp_plan in current_plans:
            try:
                # Extract config for checking feature engineering and ablation
                config = exp_plan.get("config", {})
                feature_engineering = config.get("feature_engineering", [])
                drop_columns = config.get("drop_columns", [])
                ablation_target = config.get("ablation_target")

                # Determine dataset spec to use
                base_dataset_spec_id = UUID(exp_plan["dataset_spec_id"])
                actual_dataset_spec_id = base_dataset_spec_id

                # CRITICAL: Validate that the dataset has data_sources_json
                base_dataset = db.query(DatasetSpec).filter(DatasetSpec.id == base_dataset_spec_id).first()
                if not base_dataset:
                    logger.error(f"Dynamic mode: dataset {base_dataset_spec_id} not found, skipping experiment")
                    iteration.experiments_failed += 1
                    continue
                if not base_dataset.data_sources_json:
                    logger.error(
                        f"Dynamic mode: dataset '{base_dataset.name}' ({base_dataset_spec_id}) "
                        f"has no data_sources_json, skipping experiment"
                    )
                    iteration.experiments_failed += 1
                    continue

                # If AI specified feature engineering or drop_columns, create derivative dataset
                if feature_engineering or drop_columns:
                    # Format feature engineering to expected format if needed
                    formatted_features = []
                    for fe in feature_engineering:
                        if isinstance(fe, dict):
                            # Ensure proper format: {name, formula, description}
                            # AI might use 'name' or 'output_column'
                            name = fe.get("name") or fe.get("output_column")
                            formula = fe.get("formula")
                            if name and formula:
                                formatted_features.append({
                                    "output_column": name,
                                    "formula": formula,
                                    "source_columns": fe.get("source_columns", []),
                                    "description": fe.get("description", ""),
                                })

                    name_suffix = ""
                    if ablation_target:
                        name_suffix = f" - Ablation: {ablation_target}"
                    elif formatted_features:
                        name_suffix = f" - {len(formatted_features)} Features"

                    try:
                        derived_spec = _create_derivative_dataset_spec(
                            db=db,
                            base_dataset_spec_id=base_dataset_spec_id,
                            session=session,
                            feature_engineering=formatted_features,
                            drop_columns=drop_columns if drop_columns else None,
                            name_suffix=name_suffix,
                            used_context=exp_plan.get("used_context", False),
                        )
                        actual_dataset_spec_id = derived_spec.id
                        logger.info(
                            f"Using derived dataset spec {actual_dataset_spec_id} "
                            f"with {len(formatted_features)} engineered features, "
                            f"{len(drop_columns)} dropped columns"
                        )
                    except Exception as e:
                        logger.error(f"Failed to create derivative dataset spec: {e}")
                        # Fall back to base dataset

                # Create actual Experiment object from plan
                exp = Experiment(
                    project_id=session.project_id,
                    dataset_spec_id=actual_dataset_spec_id,
                    name=f"[{session.name}] {exp_plan.get('dataset_name', 'Dynamic')} c{cycle_number}",
                    description=exp_plan.get("hypothesis"),
                    status=ExperimentStatus.PENDING,
                    experiment_plan_json={
                        "hypothesis": exp_plan.get("hypothesis"),
                        "config": config,
                        "auto_ds_session_id": str(session.id),
                        "iteration": iteration.iteration_number,
                        "execution_mode": "dynamic",
                        "robust_validation": _get_robust_validation_config(session),
                        "cycle": cycle_number,
                        "ablation_target": ablation_target,
                        "feature_engineering_applied": len(feature_engineering) if feature_engineering else 0,
                        "drop_columns_applied": drop_columns if drop_columns else [],
                    },
                )
                db.add(exp)
                db.flush()

                # Check stop flag before running experiment
                _check_stop_requested(str(session.id))

                # Run the experiment
                _run_single_experiment(db, exp)

                # Refresh and check status
                db.refresh(exp)
                if exp.status == ExperimentStatus.COMPLETED:
                    cycle_experiments.append(exp)
                    completed_experiments.append(exp)

                    # Record the result (use exp.dataset_spec_id which may be derived)
                    orchestrator.record_experiment_result(
                        iteration=iteration,
                        experiment=exp,
                        dataset_spec_id=exp.dataset_spec_id,
                        variant=exp_plan.get("variant", cycle_number),
                        hypothesis=exp_plan.get("hypothesis"),
                    )
                else:
                    iteration.experiments_failed += 1

                db.commit()

            except Exception as e:
                error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(f"Dynamic mode experiment failed: {error_msg}")
                # Save error to experiment if it was created
                if 'exp' in locals():
                    exp.status = ExperimentStatus.FAILED
                    exp.error_message = error_msg
                iteration.experiments_failed += 1
                db.commit()

        # Check if we should continue
        if len(completed_experiments) >= max_experiments_this_iteration:
            logger.info(f"Dynamic mode: reached max experiments ({max_experiments_this_iteration})")
            break

        # Use any remaining initial plans first
        if remaining_initial_plans:
            current_plans = remaining_initial_plans[:experiments_per_cycle]
            remaining_initial_plans = remaining_initial_plans[experiments_per_cycle:]
            logger.info(f"Dynamic mode: using {len(current_plans)} remaining initial plans")
            continue

        # Call DynamicPlanningAgent to design next experiments
        logger.info("Dynamic mode: calling DynamicPlanningAgent for next experiments")

        try:
            # Get available datasets for new experiments
            available_datasets = orchestrator.get_active_datasets(session)

            # Build context for the agent
            experiment_history = []
            for exp in completed_experiments:
                score = _get_experiment_score(exp)
                experiment_history.append({
                    "name": exp.name,
                    "hypothesis": exp.experiment_plan_json.get("hypothesis") if exp.experiment_plan_json else None,
                    "config": exp.experiment_plan_json.get("config", {}) if exp.experiment_plan_json else {},
                    "score": score,
                    "results": exp.results_json,
                })

            # Call the dynamic planning agent
            agent = DynamicPlanningAgent(db)
            next_plans = asyncio.run(
                agent.design_next_experiments(
                    session=session,
                    iteration=iteration,
                    experiment_history=experiment_history,
                    available_datasets=available_datasets,
                    experiments_to_design=experiments_per_cycle,
                    feature_flags={
                        "enable_feature_engineering": session.enable_feature_engineering,
                        "enable_ensemble": session.enable_ensemble,
                        "enable_ablation": session.enable_ablation,
                        "enable_diverse_configs": session.enable_diverse_configs,
                    },
                )
            )

            if next_plans and next_plans.get("experiments"):
                current_plans = next_plans["experiments"]
                logger.info(f"Dynamic mode: agent designed {len(current_plans)} new experiments")

                # Check if agent recommends stopping
                if next_plans.get("should_stop", False):
                    logger.info(f"Dynamic mode: agent recommends stopping - {next_plans.get('stop_reason', 'no reason given')}")
                    # Run the designed experiments first, then stop
            else:
                logger.info("Dynamic mode: agent designed no experiments, ending iteration")
                current_plans = []

        except Exception as e:
            logger.error(f"Dynamic mode: DynamicPlanningAgent failed: {e}")
            # Fall back to ending the iteration
            current_plans = []

    # Record dynamic execution summary
    iteration.analysis_summary_json = iteration.analysis_summary_json or {}
    iteration.analysis_summary_json["dynamic_summary"] = {
        "cycles_run": cycle_number,
        "experiments_completed": len(completed_experiments),
        "experiments_failed": iteration.experiments_failed,
    }

    return completed_experiments

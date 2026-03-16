"""Auto DS Team Orchestration Service.

This module provides the core orchestration logic for the Auto DS Team feature.
It manages the autonomous research loop:
1. Plan experiments for each active dataset
2. Execute experiments sequentially
3. Analyze results across all experiments
4. Make strategic decisions for the next iteration
5. Repeat until stopping conditions are met

The orchestrator coordinates between:
- AutoDSSession: Overall session configuration and progress
- AutoDSIteration: Individual iteration tracking
- Agents: ExperimentOrchestrator, CrossAnalysis, Strategy
- Existing infrastructure: Experiments, DatasetSpecs, AgentRuns
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import (
    Project,
    Experiment,
    ExperimentStatus,
    DatasetSpec,
)
from app.models.auto_ds_session import (
    AutoDSSession,
    AutoDSSessionStatus,
    AutoDSIteration,
    AutoDSIterationStatus,
    AutoDSIterationExperiment,
    ResearchInsight,
    InsightType,
    InsightConfidence,
    GlobalInsight,
)
from app.models.research_cycle import ResearchCycle, ResearchCycleStatus
from app.services.stop_flag import set_stop_flag, clear_stop_flag


logger = logging.getLogger(__name__)


# Metrics where lower is better (error metrics)
MINIMIZE_METRICS = {
    "rmse", "root_mean_squared_error",
    "mse", "mean_squared_error",
    "mae", "mean_absolute_error",
    "mape", "mean_absolute_percentage_error",
    "log_loss", "logloss",
    "pinball_loss",
}


def get_metric_direction(experiment: Experiment) -> str:
    """Determine if metric should be minimized or maximized.

    Returns:
        'minimize' for error metrics (RMSE, MAE, etc.)
        'maximize' for accuracy metrics (accuracy, AUC, F1, etc.)
    """
    # First check if experiment has explicit metric_direction
    if experiment.metric_direction:
        return experiment.metric_direction.value if hasattr(experiment.metric_direction, 'value') else str(experiment.metric_direction)

    # Check if we can infer from primary_metric or results
    metric_name = None
    if experiment.primary_metric:
        metric_name = experiment.primary_metric.lower()
    elif experiment.results_json:
        metric_name = experiment.results_json.get('eval_metric', '').lower()
        if not metric_name:
            metric_name = experiment.results_json.get('primary_metric', '').lower()

    # If we have a metric name, check if it's a 'lower is better' metric
    if metric_name and any(m in metric_name for m in MINIMIZE_METRICS):
        return 'minimize'

    # Default to maximize (accuracy, AUC, F1, etc.)
    return 'maximize'


def is_better_score(new_score: float, old_score: float, direction: str) -> bool:
    """Compare scores respecting metric direction.

    Args:
        new_score: The new score to compare
        old_score: The existing best score
        direction: 'minimize' or 'maximize'

    Returns:
        True if new_score is better than old_score
    """
    if direction == 'minimize':
        return new_score < old_score
    return new_score > old_score


def get_experiment_all_scores(experiment: Experiment) -> Dict[str, Optional[float]]:
    """Get all 3 scores (train, validation, holdout) from an experiment.

    Extracts training, validation, and holdout scores from the best trial.

    Args:
        experiment: The experiment to get scores from

    Returns:
        Dict with keys: train_score, val_score, holdout_score (all may be None)
    """
    result = {
        "train_score": None,
        "val_score": None,
        "holdout_score": None,
    }

    # Get metrics from best trial
    metrics = None
    if experiment.trials:
        for trial in experiment.trials:
            if trial.status.value == "completed" and trial.metrics_json:
                metrics = trial.metrics_json
                break

    # Fallback to model_versions
    if not metrics and experiment.model_versions:
        for mv in experiment.model_versions:
            if mv.metrics_json:
                metrics = mv.metrics_json
                break

    if not metrics:
        return result

    # Extract training score
    result["train_score"] = (
        metrics.get("train_mcc") or
        metrics.get("train_score") or
        metrics.get("train_accuracy")
    )

    # Extract validation score
    result["val_score"] = (
        metrics.get("validation_mcc") or
        metrics.get("validation_score") or
        metrics.get("val_score") or
        metrics.get("val_mcc") or
        metrics.get("val_accuracy") or
        metrics.get("score_val")  # AutoGluon's default key
    )

    # Extract holdout score (gold standard)
    result["holdout_score"] = (
        metrics.get("holdout_mcc") or
        metrics.get("holdout_score") or
        metrics.get("holdout_accuracy")
    )

    return result


def get_experiment_final_score(experiment: Experiment) -> Optional[float]:
    """Get the final score for an experiment, preferring holdout over validation.

    This is a convenience wrapper around get_experiment_all_scores.

    Args:
        experiment: The experiment to get the score from

    Returns:
        The best available score (holdout > validation > train), or None
    """
    scores = get_experiment_all_scores(experiment)

    # Priority: holdout > validation > train
    return scores["holdout_score"] or scores["val_score"] or scores["train_score"]


class AutoDSOrchestrator:
    """Orchestrates autonomous ML research sessions.

    This class manages the full lifecycle of an Auto DS Team session:
    - Creating and starting sessions
    - Running iterations
    - Coordinating agents
    - Checking stopping conditions
    - Recording insights
    """

    def __init__(self, db: Session):
        """Initialize the orchestrator.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def create_session(
        self,
        project_id: UUID,
        name: str,
        description: Optional[str] = None,
        max_iterations: int = 10,
        accuracy_threshold: Optional[float] = None,
        time_budget_minutes: Optional[int] = None,
        min_improvement_threshold: float = 0.001,
        plateau_iterations: int = 3,
        max_experiments_per_dataset: int = 3,
        max_active_datasets: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ) -> AutoDSSession:
        """Create a new Auto DS session.

        Args:
            project_id: The project to run the session for
            name: Session name
            description: Optional description
            max_iterations: Maximum number of iterations
            accuracy_threshold: Stop if this score is achieved
            time_budget_minutes: Optional time budget
            min_improvement_threshold: Minimum improvement to continue
            plateau_iterations: Stop after N iterations without improvement
            max_experiments_per_dataset: How many experiment configs to try per dataset
            max_active_datasets: Maximum datasets to maintain
            config: Additional configuration

        Returns:
            The created AutoDSSession
        """
        # Verify project exists
        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Create research cycle for this session
        max_seq = (
            self.db.query(ResearchCycle)
            .filter(ResearchCycle.project_id == project_id)
            .count()
        )
        research_cycle = ResearchCycle(
            project_id=project_id,
            sequence_number=max_seq + 1,
            status=ResearchCycleStatus.PENDING,
            summary_title=f"Auto DS Team: {name}",
        )
        self.db.add(research_cycle)
        self.db.flush()

        # Create the session
        session = AutoDSSession(
            project_id=project_id,
            name=name,
            description=description,
            status=AutoDSSessionStatus.PENDING,
            max_iterations=max_iterations,
            accuracy_threshold=accuracy_threshold,
            time_budget_minutes=time_budget_minutes,
            min_improvement_threshold=min_improvement_threshold,
            plateau_iterations=plateau_iterations,
            max_experiments_per_dataset=max_experiments_per_dataset,
            max_active_datasets=max_active_datasets,
            research_cycle_id=research_cycle.id,
            config_json=config,
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        logger.info(f"Created Auto DS session {session.id} for project {project_id}")
        return session

    def start_session(self, session_id: UUID) -> AutoDSSession:
        """Start a pending session.

        Args:
            session_id: The session to start

        Returns:
            The updated session
        """
        session = self.db.query(AutoDSSession).filter(AutoDSSession.id == session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.status != AutoDSSessionStatus.PENDING:
            raise ValueError(f"Session {session_id} is not pending (status: {session.status})")

        session.status = AutoDSSessionStatus.RUNNING
        session.started_at = datetime.utcnow()

        # Update research cycle status
        if session.research_cycle_id:
            research_cycle = self.db.query(ResearchCycle).filter(
                ResearchCycle.id == session.research_cycle_id
            ).first()
            if research_cycle:
                research_cycle.status = ResearchCycleStatus.RUNNING

        self.db.commit()
        self.db.refresh(session)

        logger.info(f"Started Auto DS session {session_id}")
        return session

    def pause_session(self, session_id: UUID) -> AutoDSSession:
        """Pause a running session.

        Args:
            session_id: The session to pause

        Returns:
            The updated session
        """
        session = self.db.query(AutoDSSession).filter(AutoDSSession.id == session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.status != AutoDSSessionStatus.RUNNING:
            raise ValueError(f"Session {session_id} is not running (status: {session.status})")

        session.status = AutoDSSessionStatus.PAUSED
        self.db.commit()
        self.db.refresh(session)

        logger.info(f"Paused Auto DS session {session_id}")
        return session

    def resume_session(self, session_id: UUID) -> AutoDSSession:
        """Resume a paused session.

        Args:
            session_id: The session to resume

        Returns:
            The updated session
        """
        session = self.db.query(AutoDSSession).filter(AutoDSSession.id == session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.status != AutoDSSessionStatus.PAUSED:
            raise ValueError(f"Session {session_id} is not paused (status: {session.status})")

        session.status = AutoDSSessionStatus.RUNNING
        self.db.commit()
        self.db.refresh(session)

        logger.info(f"Resumed Auto DS session {session_id}")
        return session

    def stop_session(self, session_id: UUID, reason: str = "User stopped") -> AutoDSSession:
        """Stop a session.

        Args:
            session_id: The session to stop
            reason: Reason for stopping

        Returns:
            The updated session
        """
        from app.core.celery_app import celery_app

        session = self.db.query(AutoDSSession).filter(AutoDSSession.id == session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # 1. Revoke the main Celery task if it exists
        # Set stop flag in Redis so running tasks can check and exit gracefully
        set_stop_flag(str(session_id))
        logger.info(f"Set stop flag for session {session_id}")
        
        if session.celery_task_id:
            try:
                celery_app.control.revoke(session.celery_task_id, terminate=True)
                logger.info(f"Revoked Celery task {session.celery_task_id}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task: {e}")

        # 2. Cancel all pending/running experiments for this session
        pending_experiments = self.db.query(Experiment).filter(
            Experiment.status.in_([ExperimentStatus.PENDING, ExperimentStatus.RUNNING]),
        ).all()

        cancelled_count = 0
        for exp in pending_experiments:
            # Check if this experiment belongs to this session
            exp_plan = exp.experiment_plan_json or {}
            if exp_plan.get("auto_ds_session_id") == str(session_id):
                exp.status = ExperimentStatus.CANCELLED
                exp.error = f"Cancelled: {reason}"
                cancelled_count += 1

                # Revoke the experiment's Celery task if it has one
                if exp.celery_task_id:
                    try:
                        celery_app.control.revoke(exp.celery_task_id, terminate=True)
                    except Exception as e:
                        logger.warning(f"Failed to revoke experiment task {exp.celery_task_id}: {e}")

        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} pending experiments for session {session_id}")

        # 3. Purge the Celery queue to clear any queued tasks
        try:
            celery_app.control.purge()
            logger.info("Purged Celery queue")
        except Exception as e:
            logger.warning(f"Failed to purge Celery queue: {e}")

        session.status = AutoDSSessionStatus.STOPPED
        session.completed_at = datetime.utcnow()

        # Update research cycle
        if session.research_cycle_id:
            research_cycle = self.db.query(ResearchCycle).filter(
                ResearchCycle.id == session.research_cycle_id
            ).first()
            if research_cycle:
                research_cycle.status = ResearchCycleStatus.COMPLETED

        self.db.commit()
        self.db.refresh(session)

        logger.info(f"Stopped Auto DS session {session_id}: {reason}")
        return session

    def check_stopping_conditions(self, session: AutoDSSession) -> Tuple[bool, str]:
        """Check if any stopping condition is met.

        Args:
            session: The session to check

        Returns:
            Tuple of (should_stop, reason)
        """
        # Max iterations
        if session.current_iteration >= session.max_iterations:
            return True, f"Reached max iterations ({session.max_iterations})"

        # Accuracy threshold
        if session.accuracy_threshold and session.best_score:
            if session.best_score >= session.accuracy_threshold:
                return True, f"Achieved accuracy threshold ({session.best_score:.4f} >= {session.accuracy_threshold:.4f})"

        # Time budget
        if session.time_budget_minutes:
            elapsed = session.elapsed_minutes
            if elapsed >= session.time_budget_minutes:
                return True, f"Exceeded time budget ({elapsed:.1f} >= {session.time_budget_minutes} minutes)"

        # Plateau
        if session.iterations_without_improvement >= session.plateau_iterations:
            return True, f"No improvement for {session.plateau_iterations} iterations"

        return False, ""

    def create_iteration(self, session_id: UUID) -> AutoDSIteration:
        """Create a new iteration for a session.

        Args:
            session_id: The session to create an iteration for

        Returns:
            The created iteration
        """
        session = self.db.query(AutoDSSession).filter(AutoDSSession.id == session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")

        iteration = AutoDSIteration(
            session_id=session_id,
            iteration_number=session.current_iteration + 1,
            status=AutoDSIterationStatus.PENDING,
        )
        self.db.add(iteration)

        # Update session current iteration
        session.current_iteration += 1

        self.db.commit()
        self.db.refresh(iteration)

        logger.info(f"Created iteration {iteration.iteration_number} for session {session_id}")
        return iteration

    def get_active_datasets(self, session: AutoDSSession) -> List[DatasetSpec]:
        """Get the active dataset specs for a session.

        For the first iteration, this gets all validated datasets from the project.
        For subsequent iterations, this gets datasets based on strategy decisions.

        Args:
            session: The session

        Returns:
            List of active DatasetSpec objects
        """
        # Get the latest iteration's strategy decisions
        latest_iteration = (
            self.db.query(AutoDSIteration)
            .filter(AutoDSIteration.session_id == session.id)
            .filter(AutoDSIteration.status == AutoDSIterationStatus.COMPLETED)
            .order_by(AutoDSIteration.iteration_number.desc())
            .first()
        )

        if latest_iteration and latest_iteration.strategy_decisions_json:
            # Use strategy decisions to determine active datasets
            decisions = latest_iteration.strategy_decisions_json
            keep_ids = decisions.get("datasets_to_keep", [])
            new_ids = [d.get("id") for d in decisions.get("new_datasets_created", []) if d.get("id")]

            all_ids = keep_ids + new_ids
            if all_ids:
                # CRITICAL: Only return datasets that have data_sources_json configured
                datasets = (
                    self.db.query(DatasetSpec)
                    .filter(DatasetSpec.id.in_(all_ids))
                    .filter(DatasetSpec.data_sources_json.isnot(None))
                    .limit(session.max_active_datasets)
                    .all()
                )
                if not datasets:
                    logger.warning("Strategy datasets have no data_sources_json - falling back to project datasets")
                else:
                    return datasets

        # First iteration or no strategy yet - get all validated datasets
        # CRITICAL: Only return datasets that have data_sources_json configured
        return (
            self.db.query(DatasetSpec)
            .filter(DatasetSpec.project_id == session.project_id)
            .filter(DatasetSpec.data_sources_json.isnot(None))
            .limit(session.max_active_datasets)
            .all()
        )

    def plan_experiments_for_iteration(
        self,
        iteration: AutoDSIteration,
        datasets: List[DatasetSpec],
    ) -> List[Dict[str, Any]]:
        """Plan experiments for an iteration.

        This creates experiment plans for each dataset. The actual
        experiment creation happens later when executing.

        Args:
            iteration: The iteration to plan for
            datasets: Active datasets to run experiments on

        Returns:
            List of experiment plans (dict with dataset_spec_id, configs, hypotheses)
        """
        session = iteration.session
        plans = []

        for dataset in datasets:
            # Create multiple experiment variants per dataset
            for variant in range(1, session.max_experiments_per_dataset + 1):
                plan = {
                    "dataset_spec_id": str(dataset.id),
                    "dataset_name": dataset.name,
                    "variant": variant,
                    "hypothesis": self._generate_experiment_hypothesis(dataset, variant, iteration),
                    "config": self._generate_experiment_config(dataset, variant, iteration),
                }
                plans.append(plan)

        # Update iteration with planned experiments count
        iteration.experiments_planned = len(plans)
        self.db.commit()

        logger.info(f"Planned {len(plans)} experiments for iteration {iteration.iteration_number}")
        return plans

    def _generate_experiment_hypothesis(
        self,
        dataset: DatasetSpec,
        variant: int,
        iteration: AutoDSIteration,
    ) -> str:
        """Generate a hypothesis for an experiment variant.

        Args:
            dataset: The dataset
            variant: Variant number (1, 2, 3, etc.)
            iteration: Current iteration

        Returns:
            Hypothesis string
        """
        # For now, simple variant-based hypotheses
        # In the future, the ExperimentOrchestrator agent will generate these
        hypotheses = {
            1: "Baseline configuration with high_quality preset",
            2: "Test with stacking enabled for potential ensemble gains",
            3: "Fast iteration with medium_quality to explore parameter space",
        }
        return hypotheses.get(variant, f"Variant {variant} experiment")

    def _generate_experiment_config(
        self,
        dataset: DatasetSpec,
        variant: int,
        iteration: AutoDSIteration,
    ) -> Dict[str, Any]:
        """Generate experiment configuration for a variant.

        Args:
            dataset: The dataset
            variant: Variant number
            iteration: Current iteration

        Returns:
            AutoML configuration dict
        """
        # Base configuration
        configs = {
            1: {
                "time_limit": 300,
                "presets": "high_quality",
                "num_bag_folds": 5,
                "num_stack_levels": 0,
            },
            2: {
                "time_limit": 300,
                "presets": "high_quality",
                "num_bag_folds": 5,
                "num_stack_levels": 2,
            },
            3: {
                "time_limit": 120,
                "presets": "medium_quality",
                "num_bag_folds": 0,
                "num_stack_levels": 0,
            },
        }
        return configs.get(variant, configs[1])

    def record_experiment_result(
        self,
        iteration: AutoDSIteration,
        experiment: Experiment,
        dataset_spec_id: UUID,
        variant: int,
        hypothesis: Optional[str] = None,
    ) -> AutoDSIterationExperiment:
        """Record an experiment result in the iteration.

        Args:
            iteration: The iteration
            experiment: The completed experiment
            dataset_spec_id: The dataset spec ID
            variant: Experiment variant number
            hypothesis: The hypothesis being tested

        Returns:
            The created AutoDSIterationExperiment link
        """
        # Get all scores from the experiment
        all_scores = get_experiment_all_scores(experiment)
        train_score = all_scores["train_score"]
        val_score = all_scores["val_score"]
        holdout_score = all_scores["holdout_score"]

        # Primary score for comparisons (prefer holdout > val > train)
        score = holdout_score or val_score or train_score

        # Create the link
        link = AutoDSIterationExperiment(
            iteration_id=iteration.id,
            experiment_id=experiment.id,
            dataset_spec_id=dataset_spec_id,
            experiment_variant=variant,
            experiment_hypothesis=hypothesis,
            score=score,
        )
        self.db.add(link)

        # Update iteration counters and scores
        if experiment.status == ExperimentStatus.COMPLETED:
            iteration.experiments_completed += 1
            # Use metric direction for proper comparison
            direction = get_metric_direction(experiment)
            if score and (iteration.best_score_this_iteration is None or is_better_score(score, iteration.best_score_this_iteration, direction)):
                iteration.best_score_this_iteration = score
                iteration.best_train_score_this_iteration = train_score
                iteration.best_val_score_this_iteration = val_score
                iteration.best_holdout_score_this_iteration = holdout_score
                iteration.best_experiment_id_this_iteration = experiment.id
        elif experiment.status == ExperimentStatus.FAILED:
            iteration.experiments_failed += 1

        # Update session counters and scores
        session = iteration.session
        session.total_experiments_run += 1
        if score and (session.best_score is None or is_better_score(score, session.best_score, direction)):
            session.best_score = score
            session.best_train_score = train_score
            session.best_val_score = val_score
            session.best_holdout_score = holdout_score
            session.best_experiment_id = experiment.id
            session.iterations_without_improvement = 0
        # NOTE: iterations_without_improvement is now updated in complete_iteration()
        # to ensure it's incremented once per iteration, not per experiment

        self.db.commit()
        self.db.refresh(link)

        return link

    def complete_iteration(
        self,
        iteration: AutoDSIteration,
        analysis_summary: Optional[Dict[str, Any]] = None,
        strategy_decisions: Optional[Dict[str, Any]] = None,
    ) -> AutoDSIteration:
        """Mark an iteration as completed.

        Args:
            iteration: The iteration to complete
            analysis_summary: Summary from cross-analysis
            strategy_decisions: Decisions from strategy agent

        Returns:
            The updated iteration
        """
        iteration.status = AutoDSIterationStatus.COMPLETED
        iteration.strategy_completed_at = datetime.utcnow()

        if analysis_summary:
            iteration.analysis_summary_json = analysis_summary
        if strategy_decisions:
            iteration.strategy_decisions_json = strategy_decisions

        # Update iterations_without_improvement counter (once per iteration)
        # Check if this iteration produced a new session best by comparing experiment IDs
        session = iteration.session
        iteration_improved_session_best = (
            iteration.best_experiment_id_this_iteration is not None and
            session.best_experiment_id == iteration.best_experiment_id_this_iteration
        )
        if not iteration_improved_session_best:
            # No experiment this iteration beat the session's best score
            session.iterations_without_improvement += 1
            logger.info(
                f"Iteration {iteration.iteration_number} did not improve session best. "
                f"iterations_without_improvement: {session.iterations_without_improvement}"
            )

        self.db.commit()
        self.db.refresh(iteration)

        logger.info(f"Completed iteration {iteration.iteration_number}")
        return iteration

    def complete_session(
        self,
        session: AutoDSSession,
        reason: str,
    ) -> AutoDSSession:
        """Mark a session as completed.

        Args:
            session: The session to complete
            reason: Reason for completion

        Returns:
            The updated session
        """
        session.status = AutoDSSessionStatus.COMPLETED
        session.completed_at = datetime.utcnow()

        # Update research cycle
        if session.research_cycle_id:
            research_cycle = self.db.query(ResearchCycle).filter(
                ResearchCycle.id == session.research_cycle_id
            ).first()
            if research_cycle:
                research_cycle.status = ResearchCycleStatus.COMPLETED

        self.db.commit()
        self.db.refresh(session)

        logger.info(f"Completed Auto DS session {session.id}: {reason}")
        return session

    def fail_session(
        self,
        session: AutoDSSession,
        error_message: str,
    ) -> AutoDSSession:
        """Mark a session as failed.

        Args:
            session: The session that failed
            error_message: Error message

        Returns:
            The updated session
        """
        session.status = AutoDSSessionStatus.FAILED
        session.completed_at = datetime.utcnow()

        # Update research cycle
        if session.research_cycle_id:
            research_cycle = self.db.query(ResearchCycle).filter(
                ResearchCycle.id == session.research_cycle_id
            ).first()
            if research_cycle:
                research_cycle.status = ResearchCycleStatus.FAILED

        self.db.commit()
        self.db.refresh(session)

        logger.error(f"Failed Auto DS session {session.id}: {error_message}")
        return session

    def create_insight(
        self,
        session: AutoDSSession,
        insight_type: InsightType,
        title: str,
        description: Optional[str] = None,
        confidence: InsightConfidence = InsightConfidence.LOW,
        iteration: Optional[AutoDSIteration] = None,
        insight_data: Optional[Dict[str, Any]] = None,
        supporting_experiments: Optional[List[str]] = None,
    ) -> ResearchInsight:
        """Create a research insight.

        Args:
            session: The session
            insight_type: Type of insight
            title: Insight title
            description: Optional description
            confidence: Confidence level
            iteration: Optional iteration where discovered
            insight_data: Structured insight data
            supporting_experiments: List of experiment IDs that support this

        Returns:
            The created insight
        """
        insight = ResearchInsight(
            session_id=session.id,
            project_id=session.project_id,
            iteration_id=iteration.id if iteration else None,
            insight_type=insight_type,
            confidence=confidence,
            title=title,
            description=description,
            insight_data_json=insight_data,
            supporting_experiments=supporting_experiments,
        )
        self.db.add(insight)
        self.db.commit()
        self.db.refresh(insight)

        logger.info(f"Created insight: {title}")
        return insight

    def get_session_summary(self, session_id: UUID) -> Dict[str, Any]:
        """Get a summary of a session's progress.

        Args:
            session_id: The session ID

        Returns:
            Summary dict with progress info
        """
        session = self.db.query(AutoDSSession).filter(AutoDSSession.id == session_id).first()
        if not session:
            return {"error": "Session not found"}

        iterations = (
            self.db.query(AutoDSIteration)
            .filter(AutoDSIteration.session_id == session_id)
            .order_by(AutoDSIteration.iteration_number)
            .all()
        )

        insights = (
            self.db.query(ResearchInsight)
            .filter(ResearchInsight.session_id == session_id)
            .all()
        )

        return {
            "session": {
                "id": str(session.id),
                "name": session.name,
                "status": session.status.value,
                "current_iteration": session.current_iteration,
                "max_iterations": session.max_iterations,
                "best_score": session.best_score,
                "best_experiment_id": str(session.best_experiment_id) if session.best_experiment_id else None,
                "total_experiments_run": session.total_experiments_run,
                "iterations_without_improvement": session.iterations_without_improvement,
                "elapsed_minutes": session.elapsed_minutes,
                "time_budget_minutes": session.time_budget_minutes,
                "accuracy_threshold": session.accuracy_threshold,
            },
            "iterations": [
                {
                    "iteration_number": it.iteration_number,
                    "status": it.status.value,
                    "experiments_planned": it.experiments_planned,
                    "experiments_completed": it.experiments_completed,
                    "experiments_failed": it.experiments_failed,
                    "best_score": it.best_score_this_iteration,
                }
                for it in iterations
            ],
            "insights": [
                {
                    "type": ins.insight_type.value,
                    "confidence": ins.confidence.value,
                    "title": ins.title,
                }
                for ins in insights
            ],
        }


# Utility functions for use in Celery tasks

def get_or_create_orchestrator(db: Session) -> AutoDSOrchestrator:
    """Get or create an orchestrator instance.

    Args:
        db: Database session

    Returns:
        AutoDSOrchestrator instance
    """
    return AutoDSOrchestrator(db)

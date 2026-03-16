"""Celery tasks for cleanup operations."""
import logging
from datetime import datetime, timedelta
from typing import Optional

from app.tasks.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.agent_run import AgentRun, AgentRunStatus
from app.models.experiment import Experiment, ExperimentStatus

logger = logging.getLogger(__name__)

# Cleanup thresholds
ORPHAN_TIMEOUT_HOURS = 4  # Runs stuck in RUNNING for this long are orphaned
STALE_PENDING_HOURS = 24  # Pending experiments older than this may be stuck


@celery_app.task(name="app.tasks.cleanup_orphaned_agent_runs")
def cleanup_orphaned_agent_runs(
    timeout_hours: int = ORPHAN_TIMEOUT_HOURS,
    dry_run: bool = False,
) -> dict:
    """Clean up agent runs that are stuck in RUNNING state.

    These runs likely failed without proper cleanup (e.g., worker crash).

    Args:
        timeout_hours: Hours after which a running task is considered orphaned
        dry_run: If True, only report what would be cleaned up

    Returns:
        Dict with cleanup results
    """
    db = SessionLocal()

    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=timeout_hours)

        # Find orphaned runs
        orphaned_runs = db.query(AgentRun).filter(
            AgentRun.status == AgentRunStatus.RUNNING,
            AgentRun.updated_at < cutoff_time,
        ).all()

        if not orphaned_runs:
            logger.info("No orphaned agent runs found")
            return {"orphaned_count": 0, "cleaned_count": 0}

        logger.info(f"Found {len(orphaned_runs)} orphaned agent runs")

        cleaned_count = 0
        for run in orphaned_runs:
            if dry_run:
                logger.info(f"[DRY RUN] Would clean up agent run {run.id}")
            else:
                run.status = AgentRunStatus.FAILED
                run.error_message = f"Orphaned - no update for {timeout_hours}+ hours (auto-cleanup)"
                cleaned_count += 1
                logger.info(f"Cleaned up orphaned agent run {run.id}")

        if not dry_run:
            db.commit()

        return {
            "orphaned_count": len(orphaned_runs),
            "cleaned_count": cleaned_count,
            "dry_run": dry_run,
        }

    except Exception as e:
        logger.error(f"Error cleaning up orphaned runs: {e}")
        return {"error": str(e)}

    finally:
        db.close()


@celery_app.task(name="app.tasks.cleanup_stuck_experiments")
def cleanup_stuck_experiments(
    timeout_hours: int = ORPHAN_TIMEOUT_HOURS,
    dry_run: bool = False,
) -> dict:
    """Clean up experiments that are stuck in RUNNING state.

    These experiments likely failed without proper cleanup.

    Args:
        timeout_hours: Hours after which a running experiment is considered stuck
        dry_run: If True, only report what would be cleaned up

    Returns:
        Dict with cleanup results
    """
    db = SessionLocal()

    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=timeout_hours)

        # Find stuck experiments
        stuck_experiments = db.query(Experiment).filter(
            Experiment.status == ExperimentStatus.RUNNING,
            Experiment.updated_at < cutoff_time,
        ).all()

        if not stuck_experiments:
            logger.info("No stuck experiments found")
            return {"stuck_count": 0, "cleaned_count": 0}

        logger.info(f"Found {len(stuck_experiments)} stuck experiments")

        cleaned_count = 0
        for exp in stuck_experiments:
            if dry_run:
                logger.info(f"[DRY RUN] Would clean up experiment {exp.id}")
            else:
                exp.status = ExperimentStatus.FAILED
                exp.error_message = f"Stuck - no update for {timeout_hours}+ hours (auto-cleanup)"
                cleaned_count += 1
                logger.info(f"Cleaned up stuck experiment {exp.id}")

        if not dry_run:
            db.commit()

        return {
            "stuck_count": len(stuck_experiments),
            "cleaned_count": cleaned_count,
            "dry_run": dry_run,
        }

    except Exception as e:
        logger.error(f"Error cleaning up stuck experiments: {e}")
        return {"error": str(e)}

    finally:
        db.close()


@celery_app.task(name="app.tasks.cleanup_all")
def cleanup_all(
    dry_run: bool = False,
    orphan_timeout_hours: int = ORPHAN_TIMEOUT_HOURS,
) -> dict:
    """Run all cleanup tasks.

    Args:
        dry_run: If True, only report what would be cleaned up
        orphan_timeout_hours: Hours threshold for orphaned/stuck resources

    Returns:
        Combined cleanup results
    """
    results = {}

    # Clean up agent runs
    agent_run_result = cleanup_orphaned_agent_runs(
        timeout_hours=orphan_timeout_hours,
        dry_run=dry_run,
    )
    results["agent_runs"] = agent_run_result

    # Clean up experiments
    experiment_result = cleanup_stuck_experiments(
        timeout_hours=orphan_timeout_hours,
        dry_run=dry_run,
    )
    results["experiments"] = experiment_result

    logger.info(f"Cleanup complete: {results}")
    return results


# Schedule periodic cleanup (every 2 hours)
@celery_app.on_after_configure.connect
def setup_periodic_cleanup(sender, **kwargs):
    """Set up periodic cleanup task."""
    try:
        sender.add_periodic_task(
            7200.0,  # Every 2 hours
            cleanup_all.s(dry_run=False),
            name="periodic-cleanup",
        )
        logger.info("Scheduled periodic cleanup task (every 2 hours)")
    except Exception as e:
        logger.warning(f"Could not schedule periodic cleanup: {e}")

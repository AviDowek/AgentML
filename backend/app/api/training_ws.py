"""Training log streaming endpoints with AI interpretation."""
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.core.database import SessionLocal
from app.models.experiment import Experiment, ExperimentStatus
from app.services.training_logs import TrainingLogStore, AILogInterpreter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["training-logs"])


class TrainingLogEntry(BaseModel):
    timestamp: str
    raw_log: str
    interpreted: Optional[str] = None
    log_type: str = "info"


class TrainingLogsResponse(BaseModel):
    experiment_id: str
    logs: list[TrainingLogEntry]
    next_index: int
    is_running: bool
    has_more: bool


@router.get("/training/{experiment_id}/logs", response_model=TrainingLogsResponse)
async def get_training_logs(
    experiment_id: str,
    start_index: int = Query(0, ge=0, description="Start index for pagination"),
):
    """Get training logs for an experiment with AI interpretation.

    Logs are interpreted by AI before being returned.
    Only interpreted logs are returned - raw technical logs are transformed
    into plain English explanations.
    """
    db = SessionLocal()
    try:
        experiment = db.query(Experiment).filter(
            Experiment.id == UUID(experiment_id)
        ).first()

        if not experiment:
            return TrainingLogsResponse(
                experiment_id=experiment_id,
                logs=[],
                next_index=0,
                is_running=False,
                has_more=False,
            )

        is_running = experiment.status == ExperimentStatus.RUNNING

        # Get logs from Redis
        log_store = TrainingLogStore(experiment_id)
        logs, next_index = log_store.get_logs(start_index)

        logger.info(f"Got {len(logs)} logs from Redis for {experiment_id}, start_index={start_index}")

        # ALWAYS interpret logs with AI (overwrite any regex interpretations)
        if logs:
            try:
                logger.info(f"Starting AI interpretation for {len(logs)} logs...")
                ai_interpreter = AILogInterpreter(experiment_id)
                interpreted_logs = await ai_interpreter.interpret_batch(logs, start_index)

                # Update Redis with interpreted logs
                interpreted_count = 0
                for i, log in enumerate(interpreted_logs):
                    if log.get("interpreted"):
                        log_store.update_log_interpretation(start_index + i, log)
                        interpreted_count += 1

                # Use the interpreted logs
                logs = interpreted_logs
                logger.info(f"AI interpreted {interpreted_count}/{len(logs)} logs for {experiment_id}")
            except Exception as e:
                logger.error(f"AI interpretation failed: {e}", exc_info=True)
                # Continue with uninterpreted logs as fallback

        log_store.close()

        return TrainingLogsResponse(
            experiment_id=experiment_id,
            logs=[TrainingLogEntry(**log) for log in logs],
            next_index=next_index,
            is_running=is_running,
            has_more=len(logs) > 0,
        )
    finally:
        db.close()


@router.get("/training/{experiment_id}/status")
async def get_training_status(experiment_id: str):
    """Get training status for an experiment."""
    db = SessionLocal()
    try:
        experiment = db.query(Experiment).filter(
            Experiment.id == UUID(experiment_id)
        ).first()

        if not experiment:
            return {"found": False}

        # Get log count from Redis
        log_store = TrainingLogStore(experiment_id)
        logs = log_store.get_all_logs()
        log_store.close()

        return {
            "found": True,
            "experiment_id": experiment_id,
            "status": experiment.status.value,
            "is_running": experiment.status == ExperimentStatus.RUNNING,
            "log_count": len(logs),
        }
    finally:
        db.close()

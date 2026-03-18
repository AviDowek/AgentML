"""Stop flag service for Auto DS sessions.

Uses the database (AutoDSSession.status) instead of Redis to check for stop requests.
This works across all backends (Celery, Modal, in-process).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# In-memory cache to avoid hammering the DB on every check
_stop_flags: dict[str, bool] = {}


def set_stop_flag(session_id: str) -> None:
    """Set a stop flag for a session."""
    _stop_flags[session_id] = True

    # Also update the DB so Modal workers can see it
    try:
        from app.core.database import SessionLocal
        from app.models.auto_ds_session import AutoDSSession, AutoDSSessionStatus
        db = SessionLocal()
        try:
            from uuid import UUID
            session = db.query(AutoDSSession).filter(AutoDSSession.id == UUID(session_id)).first()
            if session and session.status not in (AutoDSSessionStatus.COMPLETED, AutoDSSessionStatus.FAILED, AutoDSSessionStatus.STOPPED):
                session.status = AutoDSSessionStatus.STOPPING
                db.commit()
                logger.info(f"Set stop flag for session {session_id} (DB status → STOPPING)")
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to set stop flag in DB: {e}")


def check_stop_flag(session_id: str) -> bool:
    """Check if stop flag is set for a session."""
    # Check in-memory cache first
    if _stop_flags.get(session_id):
        return True

    # Check DB for cross-process stop signals (Modal workers)
    try:
        from app.core.database import SessionLocal
        from app.models.auto_ds_session import AutoDSSession, AutoDSSessionStatus
        db = SessionLocal()
        try:
            from uuid import UUID
            session = db.query(AutoDSSession).filter(AutoDSSession.id == UUID(session_id)).first()
            if session and session.status in (AutoDSSessionStatus.STOPPING, AutoDSSessionStatus.STOPPED):
                _stop_flags[session_id] = True
                return True
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to check stop flag in DB: {e}")

    return False


def clear_stop_flag(session_id: str) -> None:
    """Clear the stop flag for a session."""
    _stop_flags.pop(session_id, None)


import redis
from typing import Optional
from app.core.config import get_settings

# Redis key pattern for stop flags
STOP_FLAG_KEY = 'auto_ds:stop_flag:{session_id}'

def set_stop_flag(session_id: str) -> None:
    """Set a stop flag for a session."""
    settings = get_settings()
    r = redis.from_url(settings.redis_url)
    r.set(STOP_FLAG_KEY.format(session_id=session_id), '1', ex=3600)  # Expire in 1 hour

def check_stop_flag(session_id: str) -> bool:
    """Check if stop flag is set for a session."""
    settings = get_settings()
    r = redis.from_url(settings.redis_url)
    return r.get(STOP_FLAG_KEY.format(session_id=session_id)) is not None

def clear_stop_flag(session_id: str) -> None:
    """Clear the stop flag for a session."""
    settings = get_settings()
    r = redis.from_url(settings.redis_url)
    r.delete(STOP_FLAG_KEY.format(session_id=session_id))

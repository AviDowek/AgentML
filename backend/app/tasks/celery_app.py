"""Re-export celery_app from core for backward compatibility."""
from app.core.celery_app import celery_app

__all__ = ["celery_app"]

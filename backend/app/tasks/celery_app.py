from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "agentic_ml",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
)

# Auto-discover tasks in the tasks module
celery_app.autodiscover_tasks(["app.tasks"])

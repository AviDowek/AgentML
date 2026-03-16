"""Celery application configuration."""
from celery import Celery
from kombu import Queue
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "agentic_ml",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.automl", "app.tasks.experiment_tasks", "app.tasks.auto_ds_tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task tracking
    task_track_started=True,
    task_acks_late=True,

    # Result settings
    result_expires=3600,  # Results expire after 1 hour

    # Worker settings
    worker_prefetch_multiplier=1,  # Don't prefetch tasks (good for long-running tasks)
    worker_concurrency=4,  # Allow more concurrent workers for Modal tasks

    # Task time limits
    task_soft_time_limit=3600,  # 1 hour soft limit (Modal can take 30+ min)
    task_time_limit=7200,  # 2 hour hard limit

    # Retry settings
    task_default_retry_delay=60,  # 1 minute between retries
    task_max_retries=3,

    # Task queues - 'celery' is the default queue name in Celery
    task_queues=(
        Queue('celery', routing_key='celery'),
    ),
    task_default_queue='celery',

    # All tasks use the default 'celery' queue
    task_routes={},
)

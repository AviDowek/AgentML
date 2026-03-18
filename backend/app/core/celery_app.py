"""Celery application configuration.

When TASK_BACKEND=modal or Celery is not installed, provides a stub
that makes @celery_app.task decorators no-ops (functions remain plain callables).
"""
import os
import logging

logger = logging.getLogger(__name__)

TASK_BACKEND = os.environ.get("TASK_BACKEND", "celery")


class _CeleryStub:
    """Stub that mimics celery_app.task() decorator as a no-op."""

    class conf:
        beat_schedule = {}

        @classmethod
        def update(cls, **kwargs):
            pass

    def task(self, *args, **kwargs):
        """No-op decorator — returns the function unchanged."""
        def decorator(func):
            # Add .delay() and .apply_async() stubs for compatibility
            func.delay = lambda *a, **kw: func(*a, **kw)
            func.apply_async = lambda *a, **kw: func(*a, **kw)
            func.s = lambda *a, **kw: (func, a, kw)
            return func
        # Support @celery_app.task (no parens) and @celery_app.task(...) (with parens)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return decorator(args[0])
        return decorator

    def on_after_configure(self):
        pass

    class _Signal:
        def connect(self, func, **kwargs):
            pass  # Don't schedule periodic tasks without Celery

    on_after_configure = _Signal()


if TASK_BACKEND == "modal":
    logger.info("TASK_BACKEND=modal — using Celery stub (no broker needed)")
    celery_app = _CeleryStub()
else:
    try:
        from celery import Celery
        from app.core.config import get_settings

        settings = get_settings()

        celery_app = Celery(
            "agentic_ml",
            broker=settings.celery_broker_url,
            backend=settings.celery_result_backend,
            include=["app.tasks.automl", "app.tasks.experiment_tasks", "app.tasks.auto_ds_tasks"],
        )

        celery_app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="UTC",
            enable_utc=True,
            task_track_started=True,
            task_acks_late=True,
            result_expires=3600,
            worker_prefetch_multiplier=1,
            worker_concurrency=4,
            task_soft_time_limit=3600,
            task_time_limit=7200,
            task_default_retry_delay=60,
            task_max_retries=3,
        )
    except Exception as e:
        logger.warning(f"Celery not available ({e}), using stub")
        celery_app = _CeleryStub()

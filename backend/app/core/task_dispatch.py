"""Task dispatch abstraction layer.

Supports two backends:
- "celery": Tasks dispatched via Celery (requires Redis + Celery worker)
- "modal": Tasks dispatched via Modal.com (serverless, no Celery/Redis needed)

Set TASK_BACKEND=modal in environment to use Modal.
"""
import logging
import os

logger = logging.getLogger(__name__)

TASK_BACKEND = os.environ.get("TASK_BACKEND", "celery")

# Map task names to Modal function names
MODAL_APP_NAME = "agentic-ml"


class TaskHandle:
    """Unified handle for a dispatched task, regardless of backend."""

    def __init__(self, task_id: str, backend: str = "celery"):
        self.id = task_id
        self.backend = backend

    def __repr__(self):
        return f"TaskHandle(id={self.id}, backend={self.backend})"


def dispatch_task(task_name: str, *args, **kwargs) -> TaskHandle:
    """Dispatch a background task using the configured backend.

    Args:
        task_name: The task name (e.g., "run_experiment_modal", "generate_training_critique")
        *args: Positional arguments for the task
        **kwargs: Keyword arguments for the task

    Returns:
        TaskHandle with the task/function call ID
    """
    if TASK_BACKEND == "modal":
        return _dispatch_modal(task_name, *args, **kwargs)
    else:
        return _dispatch_celery(task_name, *args, **kwargs)


def _dispatch_modal(task_name: str, *args, **kwargs) -> TaskHandle:
    """Dispatch task via Modal."""
    import modal

    fn = modal.Function.lookup(MODAL_APP_NAME, task_name)
    call = fn.spawn(*args, **kwargs)
    logger.info(f"Dispatched Modal task {task_name}: {call.object_id}")
    return TaskHandle(task_id=call.object_id, backend="modal")


def _dispatch_celery(task_name: str, *args, **kwargs) -> TaskHandle:
    """Dispatch task via Celery."""
    task = _get_celery_task(task_name)
    result = task.delay(*args, **kwargs)
    logger.info(f"Dispatched Celery task {task_name}: {result.id}")
    return TaskHandle(task_id=result.id, backend="celery")


def revoke_task(task_id: str, backend: str = None):
    """Cancel/revoke a running task.

    Args:
        task_id: The task or function call ID
        backend: Override backend detection ("celery" or "modal")
    """
    if backend is None:
        backend = TASK_BACKEND

    if backend == "modal":
        try:
            import modal
            call = modal.FunctionCall.from_id(task_id)
            call.cancel()
            logger.info(f"Cancelled Modal function call {task_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel Modal function call {task_id}: {e}")
    else:
        try:
            from app.core.celery_app import celery_app
            celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"Revoked Celery task {task_id}")
        except Exception as e:
            logger.warning(f"Failed to revoke Celery task {task_id}: {e}")


# Celery task name mapping
_CELERY_TASK_MAP = {
    "run_experiment_modal": "app.tasks.run_experiment_modal",
    "generate_training_critique": "app.tasks.generate_training_critique",
    "run_robustness_audit": "app.tasks.run_robustness_audit",
    "run_auto_improve_pipeline": "app.tasks.run_auto_improve_pipeline",
    "run_auto_ds_session": "app.tasks.auto_ds_tasks.run_auto_ds_session",
    "cancel_experiment": "app.tasks.cancel_experiment",
}


def _get_celery_task(task_name: str):
    """Get the Celery task object by short name."""
    from app.core.celery_app import celery_app

    celery_name = _CELERY_TASK_MAP.get(task_name, task_name)
    return celery_app.tasks[celery_name]

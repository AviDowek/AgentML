"""Modal.com integration for cloud-based ML training.

Modal allows running heavy ML workloads in the cloud with:
- No resource limits (uses cloud compute)
- Pay-per-use pricing
- Fast cold starts
- Easy Python decorator-based API

To use Modal:
1. Install: pip install modal
2. Set up credentials: modal token new
3. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env
"""
import logging
import os
import sys
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Memory management constants
MAX_DATASET_ROWS_WARNING = 500_000  # Warn if dataset exceeds this
MAX_DATASET_MB_WARNING = 500  # Warn if serialized dataset exceeds 500MB
MAX_DATASET_ROWS_ERROR = 2_000_000  # Error if dataset exceeds this
MAX_DATASET_MB_ERROR = 2000  # Error if serialized dataset exceeds 2GB


def estimate_dataframe_memory_mb(df: pd.DataFrame) -> float:
    """Estimate memory usage of a DataFrame in MB.

    Args:
        df: DataFrame to estimate

    Returns:
        Estimated memory usage in MB
    """
    # Use pandas memory_usage with deep=True for accurate string memory
    memory_bytes = df.memory_usage(deep=True).sum()
    return memory_bytes / (1024 * 1024)


def check_dataset_size(df: pd.DataFrame, experiment_id: str) -> None:
    """Check if dataset is too large for Modal transfer.

    Args:
        df: DataFrame to check
        experiment_id: Experiment ID for logging

    Raises:
        ValueError: If dataset exceeds maximum size
    """
    num_rows = len(df)
    memory_mb = estimate_dataframe_memory_mb(df)

    # Check for errors first
    if num_rows > MAX_DATASET_ROWS_ERROR:
        raise ValueError(
            f"Dataset too large for Modal transfer: {num_rows:,} rows "
            f"(max {MAX_DATASET_ROWS_ERROR:,}). Consider sampling the data."
        )

    if memory_mb > MAX_DATASET_MB_ERROR:
        raise ValueError(
            f"Dataset too large for Modal transfer: {memory_mb:.1f}MB "
            f"(max {MAX_DATASET_MB_ERROR}MB). Consider reducing columns or sampling."
        )

    # Check for warnings
    if num_rows > MAX_DATASET_ROWS_WARNING:
        logger.warning(
            f"Large dataset for Modal ({num_rows:,} rows, {memory_mb:.1f}MB). "
            f"Transfer may be slow. Consider sampling for faster iteration."
        )

    if memory_mb > MAX_DATASET_MB_WARNING:
        logger.warning(
            f"Large dataset memory ({memory_mb:.1f}MB) for Modal transfer. "
            f"Consider reducing features or sampling data."
        )

# Check if Modal is available
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    logger.info("Modal not installed. Cloud training will not be available.")


def is_modal_configured() -> bool:
    """Check if Modal is properly configured."""
    if not MODAL_AVAILABLE:
        return False

    from app.core.config import get_settings
    settings = get_settings()

    return bool(settings.modal_token_id and settings.modal_token_secret)


def get_modal_status() -> dict:
    """Get Modal configuration status."""
    from app.core.config import get_settings
    settings = get_settings()

    return {
        "installed": MODAL_AVAILABLE,
        "configured": is_modal_configured(),
        "enabled": settings.modal_enabled,
        "token_set": bool(settings.modal_token_id),
    }


class ModalLogCapture:
    """Captures stdout and streams to Redis for real-time Modal log viewing."""

    def __init__(self, original_stream, experiment_id: str):
        self.original = original_stream
        self.experiment_id = experiment_id
        self._buffer = ""
        self._log_store = None

    def _get_log_store(self):
        if self._log_store is None:
            from app.services.training_logs import TrainingLogStore
            self._log_store = TrainingLogStore(self.experiment_id)
        return self._log_store

    def write(self, text):
        # Always write to original
        self.original.write(text)

        # Buffer and process complete lines
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                # Send to Redis for real-time streaming
                try:
                    self._get_log_store().add_raw_log(line)
                except Exception:
                    pass  # Don't break training if Redis fails

    def flush(self):
        self.original.flush()
        if self._buffer.strip():
            try:
                self._get_log_store().add_raw_log(self._buffer)
            except Exception:
                pass
            self._buffer = ""


def run_experiment_on_modal_sync(
    experiment_id: str,
    dataset: pd.DataFrame,
    target_column: str,
    task_type: str,
    primary_metric: Optional[str],
    config: dict,
    holdout_df: Optional[pd.DataFrame] = None,
    download_model: bool = False,
) -> dict:
    """Run an experiment on Modal cloud (synchronous version).

    Args:
        experiment_id: Unique experiment ID
        dataset: Training DataFrame
        target_column: Name of target column
        task_type: ML task type
        primary_metric: Metric to optimize
        config: AutoML configuration
        holdout_df: Optional holdout DataFrame for evaluation on Modal
        download_model: If True, compress and download model artifacts (slower).
                       If False (default), skip compression for faster results.
                       Model can be downloaded later via the Download button.

    Returns:
        Dict with training results including:
        - metrics: validation scores
        - train_metrics: training scores (for overfitting detection)
        - holdout_score: holdout evaluation score
        - dataset_size: training set size
        - holdout_size: holdout set size
        - model_downloaded: whether model artifacts were included

    Raises:
        RuntimeError: If Modal is not configured
    """
    if not MODAL_AVAILABLE:
        raise RuntimeError("Modal is not installed. Run: pip install modal")

    if not is_modal_configured():
        raise RuntimeError(
            "Modal is not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env"
        )

    # Set Modal environment variables from our config
    # Modal looks for these environment variables for authentication
    from app.core.config import get_settings
    settings = get_settings()

    if settings.modal_token_id:
        os.environ["MODAL_TOKEN_ID"] = settings.modal_token_id
    if settings.modal_token_secret:
        os.environ["MODAL_TOKEN_SECRET"] = settings.modal_token_secret

    # Import the standalone Modal training module from the backend root
    # This module is OUTSIDE the app package so Modal can upload it cleanly
    # without pulling in any local dependencies like sqlalchemy
    from pathlib import Path

    # Add backend root to path so we can import the standalone module
    backend_root = Path(__file__).parent.parent.parent
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from modal_worker import app, train_autogluon_remote

    # Check dataset size before transfer
    check_dataset_size(dataset, experiment_id)

    # Log dataset info
    memory_mb = estimate_dataframe_memory_mb(dataset)
    logger.info(
        f"Preparing dataset for Modal transfer: {len(dataset):,} rows, "
        f"{len(dataset.columns)} columns, ~{memory_mb:.1f}MB"
    )

    # Serialize dataset to JSON for transfer
    dataset_json = dataset.to_json(orient="records")

    # Serialize holdout data if provided
    holdout_json = None
    logger.info(f"Holdout data check: holdout_df is None = {holdout_df is None}, len = {len(holdout_df) if holdout_df is not None else 'N/A'}")
    if holdout_df is not None and len(holdout_df) > 0:
        holdout_memory_mb = estimate_dataframe_memory_mb(holdout_df)
        logger.info(
            f"Preparing holdout data for Modal: {len(holdout_df):,} rows, ~{holdout_memory_mb:.1f}MB"
        )
        holdout_json = holdout_df.to_json(orient="records")

    logger.info(f"Submitting experiment {experiment_id} to Modal cloud...")

    # Capture stdout to stream Modal logs to Redis in real-time
    original_stdout = sys.stdout
    log_capture = ModalLogCapture(sys.stdout, experiment_id)
    sys.stdout = log_capture

    # Call the Modal function within the app context with output enabled
    try:
        with modal.enable_output():
            with app.run():
                result = train_autogluon_remote.remote(
                    dataset_json=dataset_json,
                    target_column=target_column,
                    task_type=task_type,
                    primary_metric=primary_metric,
                    config=config,
                    experiment_id=experiment_id,
                    holdout_json=holdout_json,  # Pass holdout for evaluation on Modal
                    download_model=download_model,  # Skip model compression if False (faster)
                )

        logger.info(f"Modal training complete for experiment {experiment_id}")
        return result

    except Exception as e:
        logger.error(f"Modal training failed: {e}")
        raise RuntimeError(f"Modal training failed: {e}")

    finally:
        # Restore stdout
        sys.stdout = original_stdout
        log_capture.flush()


# Keep async version for compatibility
async def run_experiment_on_modal(
    experiment_id: str,
    dataset: pd.DataFrame,
    target_column: str,
    task_type: str,
    primary_metric: Optional[str],
    config: dict,
    holdout_df: Optional[pd.DataFrame] = None,
    download_model: bool = False,
) -> dict:
    """Run an experiment on Modal cloud (async wrapper).

    This is an async wrapper around the synchronous Modal call.
    """
    # Modal's app.run() is synchronous, so we just call the sync version
    return run_experiment_on_modal_sync(
        experiment_id=experiment_id,
        dataset=dataset,
        target_column=target_column,
        task_type=task_type,
        primary_metric=primary_metric,
        config=config,
        holdout_df=holdout_df,
        download_model=download_model,
    )

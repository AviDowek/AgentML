"""File storage utilities for handling ephemeral filesystems.

On platforms like Railway, the filesystem is ephemeral — uploaded files
are lost between deploys. This module provides utilities to restore files
from the database (file_data column on DataSource) when they're missing
from disk.
"""
import logging
from pathlib import Path

from app.models.data_source import DataSource

logger = logging.getLogger(__name__)


def ensure_file_on_disk(data_source: DataSource) -> Path:
    """Ensure the uploaded file exists on disk, restoring from DB if needed.

    Args:
        data_source: DataSource model with config_json containing file_path

    Returns:
        Path to the file on disk

    Raises:
        ValueError: If file cannot be found or restored
    """
    config = data_source.config_json or {}
    file_path = config.get("file_path")

    if not file_path:
        raise ValueError(f"DataSource {data_source.id} has no file_path configured")

    path = Path(file_path)
    if path.exists():
        return path

    # File missing from disk — try to restore from DB
    if data_source.file_data:
        logger.info(f"Restoring file from DB: {file_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data_source.file_data)
        return path

    raise ValueError(
        f"File not found: {file_path} (and no file_data stored in DB). "
        f"The file was likely lost due to an ephemeral filesystem. "
        f"Please re-upload the dataset."
    )

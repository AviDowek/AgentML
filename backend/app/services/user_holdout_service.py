"""User holdout service for creating and managing user-controlled holdout sets.

This is different from the automated holdout in holdout_validator.py which is used
for overfitting detection during training. This service manages user-controlled
holdout sets that are:
1. Created BEFORE the pipeline runs (at user request)
2. Stored separately, accessible only to the user
3. Used for manual model validation after training

IMPORTANT: When a user holdout is created:
- The holdout rows are stored in the HoldoutSet database record
- The REMAINING training data is saved to a new file
- The data source's file_path is updated to point to the new filtered file
- This ensures Modal/training NEVER sees the holdout data
"""
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session

from app.models.data_source import DataSource
from app.models.holdout_set import HoldoutSet
from app.services.file_handlers import read_file

logger = logging.getLogger(__name__)


def create_user_holdout_set(
    db: Session,
    project_id: UUID,
    data_source_id: UUID,
    holdout_percentage: float = 5.0,
    random_seed: Optional[int] = None,
) -> Tuple[HoldoutSet, pd.DataFrame]:
    """Create a user holdout set by splitting data from a data source.

    This function:
    1. Loads the full data from the data source
    2. Randomly samples holdout_percentage of rows
    3. Stores the holdout rows in HoldoutSet
    4. Returns both the HoldoutSet and the remaining training data

    Args:
        db: Database session
        project_id: UUID of the project
        data_source_id: UUID of the data source
        holdout_percentage: Percentage of data to hold out (default 5%)
        random_seed: Optional seed for reproducibility

    Returns:
        Tuple of (HoldoutSet, training_dataframe)

    Raises:
        ValueError: If data source not found or no data available
    """
    # Get data source
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not data_source:
        raise ValueError(f"Data source {data_source_id} not found")

    # Load the data from the file
    config = data_source.config_json or {}
    file_path = config.get("file_path") or config.get("path")

    if not file_path:
        raise ValueError("Data source has no file path configured")

    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"Data file not found: {file_path}")

    # Read the data using the appropriate handler
    df, metadata = read_file(file_path)

    if df.empty:
        raise ValueError("Data source contains no data")

    total_rows = len(df)
    holdout_count = max(1, int(total_rows * holdout_percentage / 100))
    training_count = total_rows - holdout_count

    logger.info(
        f"Creating user holdout set: {holdout_count} rows ({holdout_percentage}%) "
        f"from {total_rows} total rows"
    )

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    else:
        random_seed = random.randint(1, 999999)
        random.seed(random_seed)

    # Randomly select holdout indices
    all_indices = list(range(total_rows))
    holdout_indices = sorted(random.sample(all_indices, holdout_count))
    training_indices = [i for i in all_indices if i not in set(holdout_indices)]

    # Split the data
    holdout_df = df.iloc[holdout_indices].copy()
    training_df = df.iloc[training_indices].copy()

    # Convert holdout data to list of dicts for JSON storage
    holdout_data = holdout_df.to_dict(orient='records')

    # Get column info from schema summary if available
    schema = data_source.schema_summary or {}
    target_column = None
    feature_columns = []

    # Try to get target from schema
    if isinstance(schema, dict):
        # Check if we have columns info
        columns = schema.get("columns") or []
        feature_columns = [col.get("name") for col in columns if col.get("name")]
    else:
        feature_columns = list(df.columns)

    # Save filtered training data to a new file (so Modal never sees holdout rows)
    original_file_path = str(file_path)
    file_ext = file_path.suffix.lower()

    # Create training file path (add _training suffix before extension)
    training_file_name = file_path.stem + "_training" + file_ext
    training_file_path = file_path.parent / training_file_name

    logger.info(f"Saving filtered training data to: {training_file_path}")

    # Save training data in the same format as original
    if file_ext in ['.parquet']:
        training_df.to_parquet(training_file_path, index=False)
    elif file_ext in ['.xlsx', '.xls']:
        training_df.to_excel(training_file_path, index=False)
    elif file_ext == '.json':
        training_df.to_json(training_file_path, orient='records')
    else:
        # Default to CSV
        training_df.to_csv(training_file_path, index=False)

    # Update the data source to point to the filtered training file
    config = data_source.config_json or {}
    config["original_file_path"] = original_file_path  # Store original for reference
    config["file_path"] = str(training_file_path)  # Point to filtered data
    config["user_holdout_applied"] = True
    config["holdout_rows_removed"] = holdout_count
    data_source.config_json = config

    logger.info(
        f"Updated data source to use filtered training file. "
        f"Original: {original_file_path}, Training: {training_file_path}"
    )

    # Create HoldoutSet record
    holdout_set = HoldoutSet(
        project_id=project_id,
        data_source_id=data_source_id,
        holdout_percentage=holdout_percentage,
        total_rows_original=total_rows,
        holdout_row_count=holdout_count,
        training_row_count=training_count,
        holdout_data_json=holdout_data,
        target_column=target_column,
        feature_columns_json=feature_columns,
        random_seed=random_seed,
    )

    db.add(holdout_set)
    db.commit()
    db.refresh(holdout_set)

    logger.info(
        f"Created holdout set {holdout_set.id}: {holdout_count} holdout rows removed, "
        f"{training_count} training rows remain in {training_file_path}"
    )

    return holdout_set, training_df


def get_holdout_set_for_project(
    db: Session,
    project_id: UUID,
    data_source_id: Optional[UUID] = None,
) -> Optional[HoldoutSet]:
    """Get the holdout set for a project.

    Args:
        db: Database session
        project_id: UUID of the project
        data_source_id: Optional specific data source ID

    Returns:
        HoldoutSet or None if not found
    """
    query = db.query(HoldoutSet).filter(HoldoutSet.project_id == project_id)

    if data_source_id:
        query = query.filter(HoldoutSet.data_source_id == data_source_id)

    # Return the most recent holdout set
    return query.order_by(HoldoutSet.created_at.desc()).first()


def get_holdout_row(
    db: Session,
    holdout_set_id: UUID,
    row_index: int,
) -> Optional[Dict[str, Any]]:
    """Get a specific row from a holdout set.

    Args:
        db: Database session
        holdout_set_id: UUID of the holdout set
        row_index: Index of the row to get

    Returns:
        Dict with row data or None if not found
    """
    holdout_set = db.query(HoldoutSet).filter(HoldoutSet.id == holdout_set_id).first()
    if not holdout_set:
        return None

    return holdout_set.get_row(row_index)


def get_all_holdout_rows(
    db: Session,
    holdout_set_id: UUID,
) -> List[Dict[str, Any]]:
    """Get all rows from a holdout set.

    Args:
        db: Database session
        holdout_set_id: UUID of the holdout set

    Returns:
        List of row dicts
    """
    holdout_set = db.query(HoldoutSet).filter(HoldoutSet.id == holdout_set_id).first()
    if not holdout_set:
        return []

    return holdout_set.get_holdout_rows()


def delete_holdout_set(
    db: Session,
    holdout_set_id: UUID,
) -> bool:
    """Delete a holdout set.

    Args:
        db: Database session
        holdout_set_id: UUID of the holdout set

    Returns:
        True if deleted, False if not found
    """
    holdout_set = db.query(HoldoutSet).filter(HoldoutSet.id == holdout_set_id).first()
    if not holdout_set:
        return False

    db.delete(holdout_set)
    db.commit()
    return True


def update_holdout_target_column(
    db: Session,
    holdout_set_id: UUID,
    target_column: str,
) -> Optional[HoldoutSet]:
    """Update the target column for a holdout set.

    This is typically called after the pipeline determines the target column.

    Args:
        db: Database session
        holdout_set_id: UUID of the holdout set
        target_column: Name of the target column

    Returns:
        Updated HoldoutSet or None if not found
    """
    holdout_set = db.query(HoldoutSet).filter(HoldoutSet.id == holdout_set_id).first()
    if not holdout_set:
        return None

    holdout_set.target_column = target_column

    # Update feature columns to exclude target
    if holdout_set.feature_columns_json:
        holdout_set.feature_columns_json = [
            col for col in holdout_set.feature_columns_json
            if col != target_column
        ]

    db.commit()
    db.refresh(holdout_set)
    return holdout_set

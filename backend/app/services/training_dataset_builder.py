"""Training dataset builder service for materializing training datasets.

Implements Phase 12.4: Materialize Training Dataset & Register as DataSource.

Takes a TrainingDatasetSpec and builds the actual training dataset by:
- Joining tables according to the join plan
- Applying filters
- Computing aggregations for one-to-many relationships
- Saving the result as a new DataSource
"""
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session

from app.models.data_source import DataSource, DataSourceType
from app.models.dataset_spec import DatasetSpec
from app.models.project import Project

logger = logging.getLogger(__name__)


class MaterializationResult:
    """Result of materializing a training dataset, including sampling metadata."""

    def __init__(
        self,
        data_source_id: UUID,
        row_count: int,
        column_count: int,
        was_sampled: bool = False,
        original_row_count: int | None = None,
        sampling_message: str | None = None,
    ):
        self.data_source_id = data_source_id
        self.row_count = row_count
        self.column_count = column_count
        self.was_sampled = was_sampled
        self.original_row_count = original_row_count
        self.sampling_message = sampling_message


def materialize_training_dataset(
    db: Session,
    project_id: UUID,
    training_dataset_spec: dict[str, Any],
    max_rows: int | None = None,
    output_format: str = "parquet",
    step_logger: Any = None,
    # Time-based task metadata (from problem_understanding step)
    is_time_based: bool = False,
    time_column: str | None = None,
    entity_id_column: str | None = None,
    prediction_horizon: str | None = None,
    target_positive_class: str | None = None,
) -> MaterializationResult:
    """Materialize a training dataset from a TrainingDatasetSpec.

    This function:
    1. Loads the base table from its data source
    2. Applies base filters
    3. Executes the join plan (with aggregations for one-to-many)
    4. Extracts the target column
    5. Excludes specified columns
    6. Applies sampling if dataset exceeds max_rows
    7. Saves the result as a new file
    8. Creates a new DataSource for the training dataset
    9. Creates a DatasetSpec referencing the new DataSource

    Args:
        db: Database session
        project_id: UUID of the project
        training_dataset_spec: The TrainingDatasetSpec dictionary
        max_rows: Maximum rows to include (default: project's max_training_rows setting)
        output_format: Output format - "parquet" or "csv" (default "parquet")
        step_logger: Optional step logger for streaming logs to UI
        is_time_based: Whether this is a time-series/temporal prediction task
        time_column: Datetime column for temporal ordering
        entity_id_column: ID column for panel/longitudinal data
        prediction_horizon: Human-readable prediction horizon (e.g., "1d", "5d")
        target_positive_class: Positive class value for classification

    Returns:
        MaterializationResult with data source ID and sampling metadata

    Raises:
        ValueError: If project not found, data sources missing, or spec invalid
    """
    # Validate project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    # Use project's max_training_rows if not specified
    from app.models.project import DEFAULT_MAX_TRAINING_ROWS
    if max_rows is None:
        max_rows = getattr(project, 'max_training_rows', DEFAULT_MAX_TRAINING_ROWS)

    # Extract spec components with type safety for malformed LLM responses
    if not isinstance(training_dataset_spec, dict):
        raise ValueError(f"training_dataset_spec must be a dict, got {type(training_dataset_spec).__name__}")

    base_table = training_dataset_spec.get("base_table")
    if not base_table:
        raise ValueError("training_dataset_spec must include 'base_table'")

    target_definition = training_dataset_spec.get("target_definition", {})
    # Handle case where target_definition is a string instead of dict
    if isinstance(target_definition, str):
        logger.warning(f"target_definition is a string instead of dict: {target_definition}")
        raise ValueError("target_definition must be a dict with 'column' key")
    if not isinstance(target_definition, dict):
        raise ValueError(f"target_definition must be a dict, got {type(target_definition).__name__}")
    if not target_definition.get("column"):
        raise ValueError("training_dataset_spec must include 'target_definition.column'")

    base_filters = training_dataset_spec.get("base_filters", [])
    if not isinstance(base_filters, list):
        logger.warning(f"base_filters is not a list, treating as empty")
        base_filters = []

    join_plan = training_dataset_spec.get("join_plan", [])
    if not isinstance(join_plan, list):
        logger.warning(f"join_plan is not a list, treating as empty")
        join_plan = []

    excluded_columns = training_dataset_spec.get("excluded_columns", [])
    if not isinstance(excluded_columns, list):
        logger.warning(f"excluded_columns is not a list, treating as empty")
        excluded_columns = []

    # Get all data sources for the project
    data_sources = db.query(DataSource).filter(
        DataSource.project_id == project_id
    ).all()

    # Build a mapping from table names to data sources
    table_to_source = _build_table_source_mapping(data_sources)

    # Verify base table exists
    if base_table not in table_to_source:
        available = list(table_to_source.keys())
        raise ValueError(
            f"Base table '{base_table}' not found. Available tables: {available}"
        )

    logger.info(f"Building training dataset from base table '{base_table}'")

    # Load the base table (don't limit here - we'll sample after joins/filters)
    base_source = table_to_source[base_table]
    base_df = _load_data_source(base_source, max_rows=None)

    logger.info(f"Loaded base table: {len(base_df)} rows, {len(base_df.columns)} columns")

    # Apply base filters
    if base_filters:
        base_df = _apply_filters(base_df, base_filters)
        logger.info(f"After filtering: {len(base_df)} rows")

    # Execute join plan
    result_df = base_df
    for join_item in join_plan:
        # Safety check - join_item should be a dict
        if not isinstance(join_item, dict):
            logger.warning(f"join_item is not a dict (got {type(join_item).__name__}), skipping")
            continue

        to_table = join_item.get("to_table")
        if not to_table or to_table not in table_to_source:
            logger.warning(f"Join table '{to_table}' not found, skipping")
            continue

        to_source = table_to_source[to_table]
        to_df = _load_data_source(to_source, max_rows=None)  # Load full table for proper aggregation

        result_df = _execute_join(result_df, to_df, join_item)
        logger.info(f"After joining '{to_table}': {len(result_df)} rows, {len(result_df.columns)} columns")

    # Handle feature engineering first (before target creation, as target may use engineered features)
    feature_engineering = training_dataset_spec.get("feature_engineering", [])
    if feature_engineering:
        from app.services.feature_engineering import apply_feature_engineering
        logger.info(f"Applying {len(feature_engineering)} feature engineering steps")
        # Use strict=False to skip failed features instead of crashing
        result_df = apply_feature_engineering(result_df, feature_engineering, inplace=False, strict=False)
        logger.info(f"After feature engineering: {len(result_df.columns)} columns")

    # Handle target column
    target_table = target_definition.get("table", base_table)
    target_column = target_definition["column"]
    target_join_key = target_definition.get("join_key")

    # Check if target needs to be created (not exists in data)
    target_creation = target_definition.get("target_creation")
    if target_creation and target_column not in result_df.columns:
        from app.services.feature_engineering import apply_target_creation
        logger.info(f"Creating target column '{target_column}' using formula")
        # Ensure target_creation has the column_name set
        if "column_name" not in target_creation:
            target_creation["column_name"] = target_column
        result_df = apply_target_creation(result_df, target_creation, inplace=False)

    if target_table != base_table and target_join_key:
        # Target is in a different table - need to join
        if target_table in table_to_source:
            target_source = table_to_source[target_table]
            target_df = _load_data_source(target_source, max_rows=None)

            # Determine the join key on the result side
            result_join_key = target_join_key
            if target_join_key not in result_df.columns:
                # Try common alternatives
                for alt_key in ["id", f"{base_table}_id"]:
                    if alt_key in result_df.columns:
                        result_join_key = alt_key
                        break

            # Join to get target
            if target_column in target_df.columns and target_join_key in target_df.columns:
                target_subset = target_df[[target_join_key, target_column]].copy()
                result_df = result_df.merge(
                    target_subset,
                    left_on=result_join_key,
                    right_on=target_join_key,
                    how="left"
                )

    # Ensure target column exists
    if target_column not in result_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in result. "
            f"Available columns: {list(result_df.columns)[:20]}. "
            f"If the target needs to be created, include 'target_creation' in target_definition."
        )

    # Drop rows with missing target
    before_drop = len(result_df)
    result_df = result_df.dropna(subset=[target_column])
    if len(result_df) < before_drop:
        logger.info(f"Dropped {before_drop - len(result_df)} rows with missing target")

    # Exclude specified columns
    columns_to_drop = [c for c in excluded_columns if c in result_df.columns]
    if columns_to_drop:
        result_df = result_df.drop(columns=columns_to_drop)
        logger.info(f"Excluded {len(columns_to_drop)} columns")

    # Track sampling metadata
    was_sampled = False
    original_row_count = len(result_df)
    sampling_message = None

    # Limit rows if necessary (with clear logging for large datasets)
    if len(result_df) > max_rows:
        was_sampled = True
        # Format row counts for human-readable messages
        original_formatted = _format_row_count(original_row_count)
        max_formatted = _format_row_count(max_rows)

        sampling_message = (
            f"Dataset is large ({original_formatted} rows). "
            f"Sampling {max_formatted} rows for training dataset."
        )

        logger.info(sampling_message)
        if step_logger:
            step_logger.info(sampling_message)

        # Use random sampling for better representation
        result_df = result_df.sample(n=max_rows, random_state=42)
        logger.info(f"Sampled {max_rows:,} rows from {original_row_count:,} total")

    # Ensure we have data
    if len(result_df) == 0:
        raise ValueError("Resulting training dataset has no rows")

    final_row_count = len(result_df)
    final_col_count = len(result_df.columns)

    logger.info(
        f"Final training dataset: {final_row_count:,} rows, "
        f"{final_col_count} columns, target: {target_column}"
    )

    # Save the dataset
    output_path = _save_dataset(result_df, project_id, output_format)

    # Create a new DataSource for the training dataset
    training_source = DataSource(
        project_id=project_id,
        name=f"training_dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        type=DataSourceType.FILE_UPLOAD,
        config_json={
            "file_path": str(output_path),
            "file_type": output_format,
            "is_training_dataset": True,
            "source_spec": training_dataset_spec,
        },
        schema_summary={
            "row_count": len(result_df),
            "column_count": len(result_df.columns),
            "columns": [
                {"name": col, "dtype": str(result_df[col].dtype)}
                for col in result_df.columns
            ],
            "target_column": target_column,
        },
    )
    db.add(training_source)
    db.commit()
    db.refresh(training_source)

    # Create a DatasetSpec
    feature_columns = [c for c in result_df.columns if c != target_column]
    dataset_spec = DatasetSpec(
        project_id=project_id,
        name=f"Training Dataset - {base_table}",
        description=f"Materialized training dataset from {base_table} with {len(join_plan)} joins",
        data_sources_json=[{
            "source_id": str(training_source.id),
            "table_name": base_table,
        }],
        target_column=target_column,
        feature_columns=feature_columns,
        spec_json=training_dataset_spec,
        # Time-based task metadata
        is_time_based=is_time_based,
        time_column=time_column,
        entity_id_column=entity_id_column,
        prediction_horizon=prediction_horizon,
        target_positive_class=target_positive_class,
    )
    db.add(dataset_spec)
    db.commit()

    logger.info(f"Created training dataset: DataSource {training_source.id}")

    return MaterializationResult(
        data_source_id=training_source.id,
        row_count=final_row_count,
        column_count=final_col_count,
        was_sampled=was_sampled,
        original_row_count=original_row_count if was_sampled else None,
        sampling_message=sampling_message,
    )


def _format_row_count(count: int) -> str:
    """Format a row count for human-readable display.

    Args:
        count: Number of rows

    Returns:
        Formatted string (e.g., "25M", "1.5M", "500K", "1,234")
    """
    if count >= 1_000_000:
        if count % 1_000_000 == 0:
            return f"{count // 1_000_000}M"
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        if count % 1_000 == 0:
            return f"{count // 1_000}K"
        return f"{count / 1_000:.1f}K"
    else:
        return f"{count:,}"


def _build_table_source_mapping(data_sources: list[DataSource]) -> dict[str, DataSource]:
    """Build a mapping from table names to data sources.

    Uses the same table name extraction logic as relationship discovery.

    Args:
        data_sources: List of DataSource objects

    Returns:
        Dictionary mapping table names to DataSource objects
    """
    import re

    mapping = {}
    for ds in data_sources:
        # Extract table name from source name
        name = ds.name
        # Remove file extensions
        name = re.sub(r'\.(csv|xlsx|xls|json|parquet|txt)$', '', name, flags=re.IGNORECASE)
        # Remove common prefixes/suffixes
        name = re.sub(r'^(data_|raw_|processed_|final_)', '', name, flags=re.IGNORECASE)
        name = re.sub(r'(_data|_raw|_processed|_final)$', '', name, flags=re.IGNORECASE)
        # Convert to lowercase and clean up
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        name = re.sub(r'_+', '_', name).strip('_')

        table_name = name or ds.name
        mapping[table_name] = ds

        # Also add the original name for flexibility
        if ds.name != table_name:
            mapping[ds.name] = ds

    return mapping


def _load_data_source(data_source: DataSource, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load data from a data source.

    Args:
        data_source: The DataSource to load
        max_rows: Maximum rows to load (for sampling)

    Returns:
        DataFrame with the loaded data

    Raises:
        ValueError: If data source type is not supported or file not found
    """
    if data_source.type != DataSourceType.FILE_UPLOAD:
        raise ValueError(
            f"Only file_upload data sources are supported. Got: {data_source.type}"
        )

    from app.services.file_storage import ensure_file_on_disk

    config = data_source.config_json or {}
    file_path = str(ensure_file_on_disk(data_source))

    # Determine file type
    file_type = config.get("file_type", "")
    if not file_type:
        ext = Path(file_path).suffix.lower()
        file_type = ext.lstrip(".")

    # Load based on file type
    nrows = max_rows if max_rows else None

    if file_type in ("csv", ""):
        df = pd.read_csv(file_path, nrows=nrows)
    elif file_type == "parquet":
        df = pd.read_parquet(file_path)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
    elif file_type in ("xlsx", "xls"):
        df = pd.read_excel(file_path, nrows=nrows)
    elif file_type == "json":
        df = pd.read_json(file_path)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
    else:
        # Default to CSV
        df = pd.read_csv(file_path, nrows=nrows)

    return df


def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    """Apply filters to a DataFrame.

    Args:
        df: Input DataFrame
        filters: List of filter dictionaries with column, operator, value

    Returns:
        Filtered DataFrame
    """
    for f in filters:
        # Safety check - filter should be a dict
        if not isinstance(f, dict):
            logger.warning(f"Filter is not a dict (got {type(f).__name__}), skipping")
            continue

        column = f.get("column")
        operator = f.get("operator")
        value = f.get("value")

        if column not in df.columns:
            logger.warning(f"Filter column '{column}' not found, skipping")
            continue

        if operator == ">=":
            df = df[df[column] >= value]
        elif operator == "<=":
            df = df[df[column] <= value]
        elif operator == ">":
            df = df[df[column] > value]
        elif operator == "<":
            df = df[df[column] < value]
        elif operator == "==":
            df = df[df[column] == value]
        elif operator == "!=":
            df = df[df[column] != value]
        elif operator == "in":
            df = df[df[column].isin(value if isinstance(value, list) else [value])]
        elif operator == "not_in":
            df = df[~df[column].isin(value if isinstance(value, list) else [value])]
        elif operator == "is_null":
            df = df[df[column].isna()]
        elif operator == "is_not_null":
            df = df[df[column].notna()]
        else:
            logger.warning(f"Unknown operator '{operator}', skipping filter")

    return df


def _execute_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_item: dict[str, Any],
) -> pd.DataFrame:
    """Execute a join according to the join plan item.

    Args:
        left_df: Left DataFrame (typically the base/accumulated result)
        right_df: Right DataFrame (table being joined)
        join_item: Join specification dictionary

    Returns:
        Joined DataFrame
    """
    left_key = join_item.get("left_key")
    right_key = join_item.get("right_key")
    relationship = join_item.get("relationship", "one_to_one")
    aggregation = join_item.get("aggregation")

    # Handle case where aggregation is a string instead of dict (LLM malformed response)
    if aggregation and not isinstance(aggregation, dict):
        logger.warning(f"Aggregation is not a dict (got {type(aggregation).__name__}), treating as no aggregation")
        aggregation = None

    if left_key not in left_df.columns:
        logger.warning(f"Left key '{left_key}' not found in left DataFrame")
        return left_df

    if right_key not in right_df.columns:
        logger.warning(f"Right key '{right_key}' not found in right DataFrame")
        return left_df

    if relationship == "one_to_many" and aggregation:
        # Apply aggregation for one-to-many relationships
        result = _aggregate_and_join(left_df, right_df, left_key, right_key, aggregation)
    else:
        # Simple join for one-to-one or many-to-one
        # Avoid column name collisions
        right_cols_to_use = [right_key] + [
            c for c in right_df.columns
            if c != right_key and c not in left_df.columns
        ]
        right_subset = right_df[right_cols_to_use].drop_duplicates(subset=[right_key])

        result = left_df.merge(
            right_subset,
            left_on=left_key,
            right_on=right_key,
            how="left"
        )

        # Drop duplicate join key if different names
        if left_key != right_key and right_key in result.columns:
            result = result.drop(columns=[right_key])

    return result


def _aggregate_and_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    aggregation: dict[str, Any],
) -> pd.DataFrame:
    """Aggregate the right table and join to the left.

    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        left_key: Key column in left DataFrame
        right_key: Key column in right DataFrame
        aggregation: Aggregation specification

    Returns:
        DataFrame with aggregated features joined
    """
    # Safety check - handle malformed aggregation
    if not isinstance(aggregation, dict):
        logger.warning(f"Aggregation is not a dict in _aggregate_and_join, returning left_df unchanged")
        return left_df

    window_days = aggregation.get("window_days")
    features = aggregation.get("features", [])

    # Safety check - features should be a list
    if not isinstance(features, list):
        logger.warning(f"Features is not a list (got {type(features).__name__}), returning left_df unchanged")
        return left_df

    if not features:
        # No features defined - just return left as is
        return left_df

    # Build aggregation dict
    agg_dict = {}
    agg_rename = {}

    for feat in features:
        # Safety check - each feature should be a dict
        if not isinstance(feat, dict):
            logger.warning(f"Feature is not a dict (got {type(feat).__name__}), skipping")
            continue

        feat_name = feat.get("name")
        agg_func = feat.get("agg", "count")
        col = feat.get("column", "*")

        if col == "*" or agg_func == "count":
            # Count aggregation - use the key column
            actual_col = right_key
        else:
            if col not in right_df.columns:
                logger.warning(f"Aggregation column '{col}' not found, skipping")
                continue
            actual_col = col

        if actual_col not in agg_dict:
            agg_dict[actual_col] = []

        # Map aggregation function names
        agg_map = {
            "sum": "sum",
            "count": "count",
            "avg": "mean",
            "mean": "mean",
            "min": "min",
            "max": "max",
            "std": "std",
            "first": "first",
            "last": "last",
        }
        pandas_agg = agg_map.get(agg_func.lower(), "count")
        agg_dict[actual_col].append(pandas_agg)

        # Track rename mapping
        if actual_col not in agg_rename:
            agg_rename[actual_col] = {}
        agg_rename[actual_col][pandas_agg] = feat_name

    # Apply window filter if specified
    filtered_right = right_df.copy()
    # Note: Window filtering requires a date column - would need to be specified
    # For now, we skip window filtering if not easily available

    # Perform aggregation
    if agg_dict:
        grouped = filtered_right.groupby(right_key).agg(agg_dict)

        # Flatten multi-level columns and rename
        new_cols = []
        for col, agg_funcs in agg_dict.items():
            for agg_func in agg_funcs:
                if col in agg_rename and agg_func in agg_rename[col]:
                    new_cols.append(agg_rename[col][agg_func])
                else:
                    new_cols.append(f"{col}_{agg_func}")

        grouped.columns = new_cols
        grouped = grouped.reset_index()

        # Join to left
        result = left_df.merge(
            grouped,
            left_on=left_key,
            right_on=right_key,
            how="left"
        )

        # Drop duplicate key if different names
        if left_key != right_key and right_key in result.columns:
            result = result.drop(columns=[right_key])

        # Fill NaN with 0 for aggregated columns (missing = no records)
        for col in new_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)

        return result

    return left_df


def _save_dataset(
    df: pd.DataFrame,
    project_id: UUID,
    output_format: str,
) -> Path:
    """Save the dataset to a file.

    Args:
        df: DataFrame to save
        project_id: Project UUID
        output_format: "parquet" or "csv"

    Returns:
        Path to the saved file
    """
    # Create output directory
    output_dir = Path(tempfile.gettempdir()) / "agentML" / "training_datasets" / str(project_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"training_dataset_{timestamp}.{output_format}"
    output_path = output_dir / filename

    # Save based on format
    if output_format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    logger.info(f"Saved training dataset to {output_path}")

    return output_path

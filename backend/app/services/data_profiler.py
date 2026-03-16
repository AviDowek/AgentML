"""Data profiling service for training dataset analysis.

Provides lightweight profiling for datasets that the Data Architect Agent
can reason over, without loading everything into memory.
"""
import logging
import os
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.data_source import DataSource, DataSourceType
from app.services.schema_analyzer import get_file_type, SUPPORTED_EXTENSIONS
from app.services.file_handlers import get_handler, read_file

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SAMPLE_ROWS = 50000
MAX_SAMPLE_ROWS = 100000
PROFILE_TIMEOUT_SECONDS = 300  # 5 minutes max per data source
MAX_EXAMPLE_VALUES = 5
MAX_DISTINCT_FOR_EXACT = 10000  # Use exact count if distinct < this


class DataProfiler:
    """Profiles data sources for ML training analysis."""

    def __init__(self, sample_rows: int = DEFAULT_SAMPLE_ROWS):
        """Initialize the profiler.

        Args:
            sample_rows: Maximum rows to sample for profiling (default 50000)
        """
        self.sample_rows = min(sample_rows, MAX_SAMPLE_ROWS)

    def profile_data_source(
        self,
        db: Session,
        data_source_id: UUID,
    ) -> dict[str, Any]:
        """Profile a single data source.

        Args:
            db: Database session
            data_source_id: ID of the data source to profile

        Returns:
            JSON profile containing:
            - source_id: UUID of the data source
            - source_name: Name of the data source
            - source_type: Type (file_upload, database, etc.)
            - file_name: Original filename (for file sources)
            - estimated_row_count: Total or estimated row count
            - column_count: Number of columns
            - columns: List of column profiles with:
                - name: Column name
                - inferred_type: Data type (numeric, categorical, datetime, text, boolean)
                - dtype: Original pandas/SQL dtype
                - null_count: Number of null values in sample
                - null_ratio: Ratio of nulls (0.0 to 1.0)
                - distinct_count: Number of distinct values (approximate for large datasets)
                - distinct_ratio: Ratio of distinct values
                - example_values: Sample of actual values
                - statistics: Min/max/mean for numeric, top values for categorical
            - profiled_at: Timestamp of profiling
            - sample_size: Number of rows actually sampled
            - warnings: Any issues detected during profiling

        Raises:
            ValueError: If data source not found or type not supported
        """
        data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if not data_source:
            raise ValueError(f"Data source {data_source_id} not found")

        if data_source.type == DataSourceType.FILE_UPLOAD:
            return self._profile_file_source(data_source)
        elif data_source.type == DataSourceType.DATABASE:
            return self._profile_database_source(data_source)
        elif data_source.type == DataSourceType.EXTERNAL_DATASET:
            return self._profile_external_dataset(data_source)
        else:
            raise ValueError(f"Profiling not supported for source type: {data_source.type}")

    def _profile_file_source(self, data_source: DataSource) -> dict[str, Any]:
        """Profile a file-based data source.

        Args:
            data_source: DataSource model instance

        Returns:
            Profile dictionary
        """
        config = data_source.config_json or {}
        file_path = config.get("file_path")
        warnings = []

        if not file_path:
            raise ValueError("File source missing file_path in config")

        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        file_type = config.get("file_type") or get_file_type(file_path)

        # Load data with sampling
        try:
            df, total_rows = self._load_file_with_sampling(file_path, file_type, config)
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            raise ValueError(f"Failed to load file: {str(e)}")

        # Profile the dataframe
        columns_profile = self._profile_dataframe_columns(df)

        # Check for data quality issues
        warnings.extend(self._detect_quality_issues(df, columns_profile))

        return {
            "source_id": str(data_source.id),
            "source_name": data_source.name,
            "source_type": data_source.type.value,
            "file_name": config.get("original_filename", Path(file_path).name),
            "file_type": file_type,
            "estimated_row_count": total_rows,
            "column_count": len(df.columns),
            "columns": columns_profile,
            "profiled_at": pd.Timestamp.now().isoformat(),
            "sample_size": len(df),
            "warnings": warnings,
        }

    def _profile_database_source(self, data_source: DataSource) -> dict[str, Any]:
        """Profile a database-backed data source.

        Args:
            data_source: DataSource model instance

        Returns:
            Profile dictionary
        """
        config = data_source.config_json or {}
        warnings = []

        # Get connection details
        connection_string = config.get("connection_string")
        table_name = config.get("table_name")
        query = config.get("query")

        if not connection_string:
            raise ValueError("Database source missing connection_string in config")

        if not table_name and not query:
            raise ValueError("Database source missing table_name or query in config")

        try:
            from sqlalchemy import create_engine

            engine = create_engine(connection_string)

            # Get row count
            if table_name:
                count_query = f"SELECT COUNT(*) FROM {table_name}"
            else:
                count_query = f"SELECT COUNT(*) FROM ({query}) AS subq"

            with engine.connect() as conn:
                result = conn.execute(text(count_query))
                total_rows = result.scalar()

            # Sample data
            if table_name:
                sample_query = f"SELECT * FROM {table_name} LIMIT {self.sample_rows}"
            else:
                sample_query = f"SELECT * FROM ({query}) AS subq LIMIT {self.sample_rows}"

            df = pd.read_sql(sample_query, engine)

            # Get column info from information_schema if available
            columns_profile = self._profile_dataframe_columns(df)

            # Check for data quality issues
            warnings.extend(self._detect_quality_issues(df, columns_profile))

            return {
                "source_id": str(data_source.id),
                "source_name": data_source.name,
                "source_type": data_source.type.value,
                "table_name": table_name,
                "estimated_row_count": total_rows,
                "column_count": len(df.columns),
                "columns": columns_profile,
                "profiled_at": pd.Timestamp.now().isoformat(),
                "sample_size": len(df),
                "warnings": warnings,
            }

        except Exception as e:
            logger.error(f"Failed to profile database source: {e}")
            raise ValueError(f"Failed to profile database: {str(e)}")

    def _profile_external_dataset(self, data_source: DataSource) -> dict[str, Any]:
        """Profile an external dataset (discovered public dataset).

        These may not have actual data downloaded yet, so we use schema_summary.

        Args:
            data_source: DataSource model instance

        Returns:
            Profile dictionary
        """
        config = data_source.config_json or {}
        schema = data_source.schema_summary or {}
        warnings = []

        # Check if file was downloaded
        file_path = config.get("file_path")
        if file_path and os.path.exists(file_path):
            # Profile the actual file
            return self._profile_file_source(data_source)

        # Use schema summary if available
        if not schema:
            warnings.append("External dataset has no schema information - data may need to be downloaded first")

        columns_profile = []
        if schema.get("columns"):
            for col in schema["columns"]:
                columns_profile.append({
                    "name": col.get("name", "unknown"),
                    "inferred_type": col.get("inferred_type", "unknown"),
                    "dtype": col.get("dtype", "unknown"),
                    "null_count": col.get("null_count", 0),
                    "null_ratio": col.get("null_percentage", 0) / 100 if col.get("null_percentage") else 0,
                    "distinct_count": col.get("unique_count", 0),
                    "distinct_ratio": None,  # Can't calculate without data
                    "example_values": col.get("top_values", {}).keys() if col.get("top_values") else [],
                    "statistics": self._extract_column_statistics(col),
                })

        return {
            "source_id": str(data_source.id),
            "source_name": data_source.name,
            "source_type": data_source.type.value,
            "source_url": config.get("source_url"),
            "estimated_row_count": schema.get("row_count", 0),
            "column_count": len(columns_profile),
            "columns": columns_profile,
            "profiled_at": pd.Timestamp.now().isoformat(),
            "sample_size": schema.get("sample_rows", 0),
            "warnings": warnings,
            "is_estimate": True,  # Indicate this is from metadata, not actual data
        }

    def _load_file_with_sampling(
        self,
        file_path: str,
        file_type: str,
        config: dict,
    ) -> tuple[pd.DataFrame, int]:
        """Load file data with sampling for large files.

        Uses the extensible file handler system to support many formats.

        Args:
            file_path: Path to the file
            file_type: Type of file (csv, excel, sqlite, etc.)
            config: Configuration dict with file-specific options

        Returns:
            Tuple of (sampled DataFrame, total row count)
        """
        # Get the handler for this file
        handler = get_handler(file_path)

        if handler is None:
            raise ValueError(f"Unsupported file type for profiling: {file_type}")

        # Build kwargs from config
        read_kwargs = {}
        if config.get("delimiter"):
            read_kwargs["delimiter"] = config["delimiter"]
        if config.get("sheet_name"):
            read_kwargs["sheet_name"] = config["sheet_name"]
        if config.get("table_name"):
            read_kwargs["table_name"] = config["table_name"]
        if config.get("key"):
            read_kwargs["key"] = config["key"]
        if config.get("table_index"):
            read_kwargs["table_index"] = config["table_index"]

        # Read with sampling
        df, metadata = handler.read(
            Path(file_path),
            sample_rows=self.sample_rows,
            **read_kwargs
        )

        # Get total row count
        try:
            total_rows = handler.get_total_rows(Path(file_path), **read_kwargs)
        except Exception:
            total_rows = len(df)

        return df, total_rows

    def _profile_dataframe_columns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Profile all columns in a DataFrame.

        Args:
            df: DataFrame to profile

        Returns:
            List of column profile dictionaries
        """
        columns = []
        sample_size = len(df)

        for col_name in df.columns:
            col = df[col_name]
            profile = self._profile_column(col, col_name, sample_size)
            columns.append(profile)

        return columns

    def _profile_column(
        self,
        series: pd.Series,
        name: str,
        sample_size: int,
    ) -> dict[str, Any]:
        """Profile a single column.

        Args:
            series: Pandas Series to profile
            name: Column name
            sample_size: Total sample size for ratio calculations

        Returns:
            Column profile dictionary
        """
        null_count = int(series.isnull().sum())
        null_ratio = null_count / sample_size if sample_size > 0 else 0.0

        # Get distinct count (approximate for large series)
        non_null = series.dropna()
        if len(non_null) <= MAX_DISTINCT_FOR_EXACT:
            distinct_count = int(non_null.nunique())
        else:
            # Approximate using sample
            sample = non_null.sample(min(MAX_DISTINCT_FOR_EXACT, len(non_null)))
            distinct_count = int(sample.nunique() * (len(non_null) / len(sample)))

        distinct_ratio = distinct_count / len(non_null) if len(non_null) > 0 else 0.0

        # Infer semantic type
        inferred_type = self._infer_semantic_type(series)

        # Get example values (non-null, unique)
        example_values = self._get_example_values(non_null, inferred_type)

        # Get statistics based on type
        statistics = self._compute_statistics(series, inferred_type)

        return {
            "name": str(name),
            "inferred_type": inferred_type,
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_ratio": round(null_ratio, 4),
            "distinct_count": distinct_count,
            "distinct_ratio": round(distinct_ratio, 4),
            "example_values": example_values,
            "statistics": statistics,
        }

    def _infer_semantic_type(self, series: pd.Series) -> str:
        """Infer the semantic type of a column.

        Args:
            series: Pandas Series

        Returns:
            Semantic type: 'numeric', 'categorical', 'datetime', 'text', 'boolean', 'id'
        """
        dtype = str(series.dtype)

        # Check for boolean
        unique_values = set(series.dropna().unique())
        if dtype == "bool" or unique_values.issubset({True, False, 0, 1, "true", "false", "True", "False"}):
            return "boolean"

        # Check for numeric
        if dtype in ("int64", "int32", "float64", "float32", "Int64", "Float64"):
            # Check if it might be an ID column (high cardinality integers)
            if series.nunique() / len(series.dropna()) > 0.9 and dtype in ("int64", "int32", "Int64"):
                return "id"
            return "numeric"

        # Check for datetime
        if "datetime" in dtype:
            return "datetime"

        # Try to parse as datetime
        if dtype == "object":
            try:
                sample = series.dropna().head(100)
                pd.to_datetime(sample)
                return "datetime"
            except (ValueError, TypeError):
                pass

        # Check if categorical (low cardinality string)
        if dtype == "object":
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            if unique_ratio < 0.05 or series.nunique() < 50:
                return "categorical"
            # High cardinality string is likely text or ID
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 50:
                return "text"
            if unique_ratio > 0.9:
                return "id"
            return "categorical"

        return "unknown"

    def _get_example_values(self, series: pd.Series, inferred_type: str) -> list:
        """Get example values from a series.

        Args:
            series: Non-null Pandas Series
            inferred_type: Inferred semantic type

        Returns:
            List of example values (as strings)
        """
        if len(series) == 0:
            return []

        if inferred_type in ("numeric", "datetime"):
            # For numeric/datetime, show range
            examples = [
                str(series.min()),
                str(series.max()),
            ]
            if len(series) > 2:
                median = series.median() if inferred_type == "numeric" else series.sort_values().iloc[len(series) // 2]
                examples.insert(1, str(median))
            return examples[:MAX_EXAMPLE_VALUES]

        elif inferred_type == "categorical":
            # Show most common values
            top_values = series.value_counts().head(MAX_EXAMPLE_VALUES).index.tolist()
            return [str(v) for v in top_values]

        else:
            # Show unique samples
            unique = series.unique()[:MAX_EXAMPLE_VALUES]
            return [str(v)[:100] for v in unique]  # Truncate long strings

    def _compute_statistics(self, series: pd.Series, inferred_type: str) -> dict[str, Any]:
        """Compute statistics for a column based on its type.

        Args:
            series: Pandas Series
            inferred_type: Inferred semantic type

        Returns:
            Statistics dictionary
        """
        non_null = series.dropna()
        stats = {}

        if inferred_type == "numeric":
            if len(non_null) > 0:
                stats = {
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": round(float(non_null.mean()), 4),
                    "median": float(non_null.median()),
                    "std": round(float(non_null.std()), 4) if len(non_null) > 1 else 0,
                    "zeros": int((non_null == 0).sum()),
                    "negatives": int((non_null < 0).sum()),
                }

        elif inferred_type == "categorical":
            value_counts = non_null.value_counts()
            stats = {
                "top_values": {str(k): int(v) for k, v in value_counts.head(10).items()},
                "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            }

        elif inferred_type == "datetime":
            try:
                dt_series = pd.to_datetime(non_null)
                stats = {
                    "min": str(dt_series.min()),
                    "max": str(dt_series.max()),
                    "range_days": (dt_series.max() - dt_series.min()).days,
                }
            except (ValueError, TypeError):
                pass

        elif inferred_type == "text":
            str_series = non_null.astype(str)
            stats = {
                "avg_length": round(str_series.str.len().mean(), 2),
                "min_length": int(str_series.str.len().min()),
                "max_length": int(str_series.str.len().max()),
            }

        elif inferred_type == "boolean":
            true_count = int((non_null.isin([True, 1, "true", "True"])).sum())
            false_count = int((non_null.isin([False, 0, "false", "False"])).sum())
            stats = {
                "true_count": true_count,
                "false_count": false_count,
                "true_ratio": round(true_count / len(non_null), 4) if len(non_null) > 0 else 0,
            }

        return stats

    def _extract_column_statistics(self, col_info: dict) -> dict[str, Any]:
        """Extract statistics from schema summary column info.

        Args:
            col_info: Column info from schema_summary

        Returns:
            Statistics dictionary
        """
        stats = {}

        if col_info.get("min") is not None:
            stats["min"] = col_info["min"]
        if col_info.get("max") is not None:
            stats["max"] = col_info["max"]
        if col_info.get("mean") is not None:
            stats["mean"] = col_info["mean"]
        if col_info.get("median") is not None:
            stats["median"] = col_info["median"]
        if col_info.get("top_values"):
            stats["top_values"] = col_info["top_values"]

        return stats

    def _detect_quality_issues(
        self,
        df: pd.DataFrame,
        columns_profile: list[dict],
    ) -> list[str]:
        """Detect data quality issues.

        Args:
            df: DataFrame being profiled
            columns_profile: List of column profiles

        Returns:
            List of warning messages
        """
        warnings = []

        # Check for high null ratio columns
        high_null_cols = [
            col["name"] for col in columns_profile
            if col["null_ratio"] > 0.5
        ]
        if high_null_cols:
            warnings.append(
                f"Columns with >50% missing values: {', '.join(high_null_cols)}"
            )

        # Check for potential ID columns (not useful for training)
        id_cols = [
            col["name"] for col in columns_profile
            if col["inferred_type"] == "id"
        ]
        if id_cols:
            warnings.append(
                f"Potential ID columns (consider excluding): {', '.join(id_cols)}"
            )

        # Check for constant columns
        constant_cols = [
            col["name"] for col in columns_profile
            if col["distinct_count"] == 1
        ]
        if constant_cols:
            warnings.append(
                f"Constant columns (no variance): {', '.join(constant_cols)}"
            )

        # Check for unnamed columns
        unnamed_cols = [
            col["name"] for col in columns_profile
            if str(col["name"]).startswith("Unnamed:")
        ]
        if len(unnamed_cols) > len(df.columns) * 0.3:
            warnings.append(
                f"Many unnamed columns ({len(unnamed_cols)}) - file may not have proper headers"
            )

        # Check for very low row count
        if len(df) < 100:
            warnings.append(
                f"Very small dataset ({len(df)} rows) - may not be sufficient for training"
            )

        return warnings


def profile_data_source(
    db: Session,
    data_source_id: UUID,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> dict[str, Any]:
    """Profile a single data source.

    Convenience function that creates a profiler and profiles one data source.

    Args:
        db: Database session
        data_source_id: ID of the data source to profile
        sample_rows: Maximum rows to sample

    Returns:
        Profile dictionary
    """
    profiler = DataProfiler(sample_rows=sample_rows)
    return profiler.profile_data_source(db, data_source_id)


def profile_all_data_sources(
    db: Session,
    project_id: UUID,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> list[dict[str, Any]]:
    """Profile all data sources in a project.

    Args:
        db: Database session
        project_id: ID of the project
        sample_rows: Maximum rows to sample per data source

    Returns:
        List of profile dictionaries
    """
    from app.models.project import Project

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    profiler = DataProfiler(sample_rows=sample_rows)
    profiles = []
    errors = []

    for data_source in project.data_sources:
        try:
            profile = profiler.profile_data_source(db, data_source.id)
            profiles.append(profile)
        except Exception as e:
            logger.error(f"Failed to profile data source {data_source.id}: {e}")
            errors.append({
                "source_id": str(data_source.id),
                "source_name": data_source.name,
                "error": str(e),
            })

    return {
        "project_id": str(project_id),
        "profiles": profiles,
        "errors": errors,
        "total_sources": len(project.data_sources),
        "profiled_count": len(profiles),
        "error_count": len(errors),
    }

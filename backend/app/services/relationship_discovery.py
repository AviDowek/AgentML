"""Relationship discovery service for identifying joins and base tables.

Analyzes data source profiles to find:
- Likely join keys between tables
- A base table (one row per prediction unit)
- Relationship cardinality (one-to-one, one-to-many)
"""
import logging
import re
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.data_source import DataSource
from app.models.project import Project

logger = logging.getLogger(__name__)

# Patterns for identifying ID columns (case-insensitive matching)
ID_COLUMN_PATTERNS = [
    r'^id$',
    r'^.*_id$',
    r'^.*id$',  # Matches camelCase like customerId
    r'^id_.*$',
    r'^pk$',
    r'^.*_pk$',
    r'^key$',
    r'^.*_key$',
    r'^uuid$',
    r'^.*_uuid$',
    r'^guid$',
    r'^.*_guid$',
]

# Patterns for target-like columns (prediction targets)
TARGET_COLUMN_PATTERNS = [
    r'^target$',
    r'^label$',
    r'^class$',
    r'^y$',
    r'^outcome$',
    r'^churn.*$',
    r'^.*_churn$',
    r'^cancel.*$',
    r'^.*_cancel.*$',
    r'^price$',
    r'^.*_price$',
    r'^amount$',
    r'^.*_amount$',
    r'^revenue$',
    r'^.*_revenue$',
    r'^cost$',
    r'^.*_cost$',
    r'^value$',
    r'^.*_value$',
    r'^is_.*$',
    r'^has_.*$',
    r'^flag.*$',
    r'^.*_flag$',
    r'^status$',
    r'^.*_status$',
    r'^result$',
    r'^.*_result$',
    r'^prediction$',
    r'^.*_prediction$',
    r'^fraud.*$',
    r'^.*_fraud$',
    r'^default.*$',
    r'^.*_default$',
    r'^conversion.*$',
    r'^.*_conversion$',
    r'^score$',
    r'^.*_score$',
    r'^rating$',
    r'^.*_rating$',
]

# Common entity prefixes for matching related tables
ENTITY_PREFIXES = [
    'customer', 'user', 'account', 'client', 'member', 'person',
    'product', 'item', 'sku', 'article',
    'order', 'transaction', 'purchase', 'sale',
    'event', 'action', 'activity', 'log',
    'session', 'visit', 'click',
    'employee', 'staff', 'worker',
    'store', 'location', 'branch', 'shop',
    'category', 'type', 'class',
    'campaign', 'promotion', 'offer',
    'invoice', 'payment', 'billing',
]


def discover_relationships_for_project(
    db: Session,
    project_id: UUID,
) -> dict[str, Any]:
    """Discover relationships between data sources in a project.

    Analyzes profile_json of all data sources to find:
    - Likely join keys between tables
    - Base table candidates (one row per prediction unit)
    - Relationship cardinality

    Args:
        db: Database session
        project_id: UUID of the project

    Returns:
        Dictionary containing:
        - tables: List of table summaries
        - relationships: List of discovered relationships
        - base_table_candidates: Ranked list of base table candidates

    Raises:
        ValueError: If project not found or no profiled data sources
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project {project_id} not found")

    # Get all data sources with profiles
    data_sources = db.query(DataSource).filter(
        DataSource.project_id == project_id,
        DataSource.profile_json.isnot(None)
    ).all()

    if not data_sources:
        return {
            "tables": [],
            "relationships": [],
            "base_table_candidates": [],
            "warnings": ["No profiled data sources found. Run profile-all first."],
        }

    # Extract table information from profiles
    tables = _extract_table_info(data_sources)

    # Discover relationships between tables
    relationships = _discover_relationships(tables)

    # Identify base table candidates
    base_table_candidates = _identify_base_table_candidates(tables, relationships)

    return {
        "tables": tables,
        "relationships": relationships,
        "base_table_candidates": base_table_candidates,
    }


def _extract_table_info(data_sources: list[DataSource]) -> list[dict[str, Any]]:
    """Extract table information from data source profiles.

    Args:
        data_sources: List of DataSource objects with profile_json

    Returns:
        List of table info dictionaries
    """
    tables = []

    for ds in data_sources:
        profile = ds.profile_json
        if not profile:
            continue

        # Extract column information
        columns = []
        id_columns = []
        target_columns = []

        for col in profile.get("columns", []):
            col_name = col.get("name", "")
            col_info = {
                "name": col_name,
                "inferred_type": col.get("inferred_type", "unknown"),
                "dtype": col.get("dtype", "unknown"),
                "null_ratio": col.get("null_ratio", 0),
                "distinct_count": col.get("distinct_count", 0),
                "distinct_ratio": col.get("distinct_ratio", 0),
            }
            columns.append(col_info)

            # Check if this looks like an ID column
            if _is_potential_key(col_name, col):
                id_columns.append(col_info)

            # Check if this looks like a target column
            if _is_potential_target(col_name, col):
                target_columns.append(col_info)

        table_info = {
            "source_id": str(ds.id),
            "source_name": ds.name,
            "table_name": _extract_table_name(ds.name),
            "row_count": profile.get("estimated_row_count", 0),
            "column_count": profile.get("column_count", 0),
            "columns": columns,
            "id_columns": id_columns,
            "target_columns": target_columns,
            "has_obvious_id": len(id_columns) > 0,
            "has_potential_target": len(target_columns) > 0,
        }
        tables.append(table_info)

    return tables


def _extract_table_name(source_name: str) -> str:
    """Extract a clean table name from the source name.

    Args:
        source_name: Original data source name (e.g., "customers.csv")

    Returns:
        Clean table name (e.g., "customers")
    """
    # Remove file extensions
    name = re.sub(r'\.(csv|xlsx|xls|json|parquet|txt)$', '', source_name, flags=re.IGNORECASE)
    # Remove common prefixes/suffixes
    name = re.sub(r'^(data_|raw_|processed_|final_)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'(_data|_raw|_processed|_final)$', '', name, flags=re.IGNORECASE)
    # Convert to lowercase and replace spaces/special chars with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
    name = re.sub(r'_+', '_', name).strip('_')
    return name or source_name


def _is_potential_key(col_name: str, col_info: dict) -> bool:
    """Check if a column is a potential key/ID column.

    Args:
        col_name: Column name
        col_info: Column profile information

    Returns:
        True if column looks like a key
    """
    # Check name patterns
    col_lower = col_name.lower()
    for pattern in ID_COLUMN_PATTERNS:
        if re.match(pattern, col_lower):
            return True

    # Check if it's detected as an ID type
    if col_info.get("inferred_type") == "id":
        return True

    # Check for high distinctness (potential unique key)
    distinct_ratio = col_info.get("distinct_ratio", 0)
    if distinct_ratio and distinct_ratio > 0.95:
        # Also check it's not a text column (like descriptions)
        if col_info.get("inferred_type") not in ("text",):
            return True

    return False


def _is_potential_target(col_name: str, col_info: dict) -> bool:
    """Check if a column is a potential prediction target.

    Args:
        col_name: Column name
        col_info: Column profile information

    Returns:
        True if column looks like a target
    """
    col_lower = col_name.lower()
    for pattern in TARGET_COLUMN_PATTERNS:
        if re.match(pattern, col_lower):
            return True

    # Binary columns are often targets
    if col_info.get("inferred_type") == "boolean":
        distinct_count = col_info.get("distinct_count", 0)
        if distinct_count == 2:
            return True

    # Low cardinality categorical could be a classification target
    if col_info.get("inferred_type") == "categorical":
        distinct_count = col_info.get("distinct_count", 0)
        if 2 <= distinct_count <= 20:
            return True

    return False


def _discover_relationships(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Discover relationships between tables based on column matching.

    Args:
        tables: List of table info dictionaries

    Returns:
        List of relationship dictionaries
    """
    relationships = []

    # Compare each pair of tables
    for i, table1 in enumerate(tables):
        for table2 in tables[i + 1:]:
            # Find matching columns between the two tables
            matches = _find_matching_columns(table1, table2)

            for match in matches:
                relationship = _infer_relationship(table1, table2, match)
                if relationship:
                    relationships.append(relationship)

    return relationships


def _find_matching_columns(
    table1: dict[str, Any],
    table2: dict[str, Any],
) -> list[dict[str, Any]]:
    """Find columns that could be join keys between two tables.

    Args:
        table1: First table info
        table2: Second table info

    Returns:
        List of column match dictionaries
    """
    matches = []

    # Get ID columns from both tables
    id_cols1 = {col["name"].lower(): col for col in table1.get("id_columns", [])}
    id_cols2 = {col["name"].lower(): col for col in table2.get("id_columns", [])}

    # Also consider all columns for matching
    all_cols1 = {col["name"].lower(): col for col in table1.get("columns", [])}
    all_cols2 = {col["name"].lower(): col for col in table2.get("columns", [])}

    # Strategy 1: Exact name match on ID columns
    for name1, col1 in id_cols1.items():
        if name1 in id_cols2:
            matches.append({
                "col1_name": col1["name"],
                "col2_name": id_cols2[name1]["name"],
                "col1_info": col1,
                "col2_info": id_cols2[name1],
                "match_type": "exact_id",
                "confidence": 0.95,
            })

    # Strategy 2: Table-prefixed ID match (e.g., customers.id matches transactions.customer_id)
    table1_name = table1.get("table_name", "").lower()
    table2_name = table2.get("table_name", "").lower()

    # Check if table1.id matches table2.{table1}_id
    if "id" in id_cols1:
        prefixed_name = f"{table1_name}_id"
        if prefixed_name in all_cols2:
            matches.append({
                "col1_name": id_cols1["id"]["name"],
                "col2_name": all_cols2[prefixed_name]["name"],
                "col1_info": id_cols1["id"],
                "col2_info": all_cols2[prefixed_name],
                "match_type": "prefixed_id",
                "confidence": 0.90,
            })

    # Check if table2.id matches table1.{table2}_id
    if "id" in id_cols2:
        prefixed_name = f"{table2_name}_id"
        if prefixed_name in all_cols1:
            matches.append({
                "col1_name": all_cols1[prefixed_name]["name"],
                "col2_name": id_cols2["id"]["name"],
                "col1_info": all_cols1[prefixed_name],
                "col2_info": id_cols2["id"],
                "match_type": "prefixed_id",
                "confidence": 0.90,
            })

    # Strategy 3: Entity-based matching (e.g., customer_id in both tables)
    for entity in ENTITY_PREFIXES:
        entity_id = f"{entity}_id"
        if entity_id in all_cols1 and entity_id in all_cols2:
            # Avoid duplicate matches
            already_matched = any(
                m["col1_name"].lower() == entity_id and m["col2_name"].lower() == entity_id
                for m in matches
            )
            if not already_matched:
                matches.append({
                    "col1_name": all_cols1[entity_id]["name"],
                    "col2_name": all_cols2[entity_id]["name"],
                    "col1_info": all_cols1[entity_id],
                    "col2_info": all_cols2[entity_id],
                    "match_type": "entity_id",
                    "confidence": 0.85,
                })

    return matches


def _infer_relationship(
    table1: dict[str, Any],
    table2: dict[str, Any],
    match: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Infer the relationship between two tables based on a column match.

    Args:
        table1: First table info
        table2: Second table info
        match: Column match information

    Returns:
        Relationship dictionary or None if no clear relationship
    """
    col1_info = match["col1_info"]
    col2_info = match["col2_info"]

    row_count1 = table1.get("row_count", 0)
    row_count2 = table2.get("row_count", 0)

    distinct1 = col1_info.get("distinct_count", 0)
    distinct2 = col2_info.get("distinct_count", 0)

    # Calculate distinct ratios
    ratio1 = distinct1 / row_count1 if row_count1 > 0 else 0
    ratio2 = distinct2 / row_count2 if row_count2 > 0 else 0

    # Infer cardinality
    cardinality = _infer_cardinality(ratio1, ratio2, distinct1, distinct2, row_count1, row_count2)

    # Determine direction (from primary to foreign key)
    if cardinality == "one_to_many":
        # The table with higher distinct ratio is likely the "one" side
        if ratio1 > ratio2:
            from_table, to_table = table1, table2
            from_col, to_col = match["col1_name"], match["col2_name"]
        else:
            from_table, to_table = table2, table1
            from_col, to_col = match["col2_name"], match["col1_name"]
    elif cardinality == "many_to_one":
        # Normalize to one_to_many
        if ratio1 < ratio2:
            from_table, to_table = table2, table1
            from_col, to_col = match["col2_name"], match["col1_name"]
        else:
            from_table, to_table = table1, table2
            from_col, to_col = match["col1_name"], match["col2_name"]
        cardinality = "one_to_many"
    else:
        # one_to_one or many_to_many: use original order
        from_table, to_table = table1, table2
        from_col, to_col = match["col1_name"], match["col2_name"]

    return {
        "from_table": from_table.get("table_name"),
        "from_source_id": from_table.get("source_id"),
        "to_table": to_table.get("table_name"),
        "to_source_id": to_table.get("source_id"),
        "from_column": from_col,
        "to_column": to_col,
        "cardinality": cardinality,
        "confidence": match.get("confidence", 0.5),
        "match_type": match.get("match_type", "unknown"),
    }


def _infer_cardinality(
    ratio1: float,
    ratio2: float,
    distinct1: int,
    distinct2: int,
    row_count1: int,
    row_count2: int,
) -> str:
    """Infer the cardinality of a relationship.

    Args:
        ratio1: Distinct ratio for column in table 1
        ratio2: Distinct ratio for column in table 2
        distinct1: Distinct count for column in table 1
        distinct2: Distinct count for column in table 2
        row_count1: Row count for table 1
        row_count2: Row count for table 2

    Returns:
        Cardinality string: "one_to_one", "one_to_many", or "many_to_many"
    """
    # High distinct ratio (>0.95) suggests unique values (primary key)
    is_unique1 = ratio1 > 0.95
    is_unique2 = ratio2 > 0.95

    if is_unique1 and is_unique2:
        # Both have unique values - likely one_to_one
        return "one_to_one"
    elif is_unique1 and not is_unique2:
        # Table 1 has unique values, table 2 has duplicates - one_to_many
        return "one_to_many"
    elif not is_unique1 and is_unique2:
        # Table 2 has unique values, table 1 has duplicates - many_to_one
        return "many_to_one"
    else:
        # Neither has unique values
        # Check if distinct counts suggest a relationship direction
        if distinct1 < distinct2 * 0.5:
            return "many_to_one"
        elif distinct2 < distinct1 * 0.5:
            return "one_to_many"
        else:
            return "many_to_many"


def _identify_base_table_candidates(
    tables: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify and score base table candidates.

    Base table characteristics:
    - Large row count (but not excessively large like logs)
    - Has an obvious primary key / ID column
    - Has potential target columns for prediction
    - Is on the "one" side of one-to-many relationships
    - Represents a core entity (customers, products, etc.)

    Args:
        tables: List of table info dictionaries
        relationships: List of discovered relationships

    Returns:
        Sorted list of base table candidates with scores and reasons
    """
    if not tables:
        return []

    candidates = []

    # Calculate statistics for normalization
    row_counts = [t.get("row_count", 0) for t in tables]
    max_row_count = max(row_counts) if row_counts else 1
    min_row_count = min(row_counts) if row_counts else 0

    for table in tables:
        score = 0.0
        reasons = []

        table_name = table.get("table_name", "")
        row_count = table.get("row_count", 0)

        # Factor 1: Has obvious ID column (weight: 0.2)
        if table.get("has_obvious_id"):
            score += 0.2
            reasons.append("Has clear primary key column")

        # Factor 2: Has potential target columns (weight: 0.25)
        target_cols = table.get("target_columns", [])
        if target_cols:
            score += 0.25
            target_names = [c["name"] for c in target_cols[:3]]
            reasons.append(f"Has potential target columns: {', '.join(target_names)}")

        # Factor 3: Row count (weight: 0.15)
        # Prefer tables that aren't too small or too large (log tables)
        if row_count > 0 and max_row_count > 0:
            # Penalize very large tables (likely transaction/log tables)
            if row_count > max_row_count * 0.8 and len(tables) > 1:
                score += 0.05
                reasons.append(f"Very large table ({row_count:,} rows) - may be transaction/log data")
            elif row_count > min_row_count * 10:
                # Medium to large - good for base table
                score += 0.15
                reasons.append(f"Reasonable row count ({row_count:,} rows)")
            else:
                score += 0.10
                reasons.append(f"Smaller table ({row_count:,} rows)")

        # Factor 4: Is on "one" side of one-to-many relationships (weight: 0.2)
        one_side_count = sum(
            1 for r in relationships
            if r.get("from_table") == table_name and r.get("cardinality") == "one_to_many"
        )
        if one_side_count > 0:
            score += min(0.2, 0.1 * one_side_count)
            reasons.append(f"Primary table in {one_side_count} relationship(s)")

        # Factor 5: Table name suggests entity table (weight: 0.1)
        for entity in ['customer', 'user', 'account', 'client', 'member', 'product', 'item']:
            if entity in table_name.lower():
                score += 0.1
                reasons.append(f"Name suggests entity table ('{entity}')")
                break

        # Factor 6: Not a log/event table (weight: 0.1)
        log_indicators = ['log', 'event', 'audit', 'history', 'archive', 'backup']
        is_log_table = any(ind in table_name.lower() for ind in log_indicators)
        if not is_log_table:
            score += 0.1
        else:
            reasons.append("Appears to be a log/event table")

        candidates.append({
            "table": table_name,
            "source_id": table.get("source_id"),
            "source_name": table.get("source_name"),
            "score": round(score, 2),
            "reasons": reasons,
            "row_count": row_count,
            "has_target": len(target_cols) > 0,
            "target_columns": [c["name"] for c in target_cols],
        })

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    return candidates

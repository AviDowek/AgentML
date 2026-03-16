"""Dataset validation service to verify column existence and detect hallucinations."""
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.data_source import DataSource
from app.models.dataset_spec import DatasetSpec


class DatasetValidationResult:
    """Result of dataset validation."""

    def __init__(self):
        self.is_valid: bool = True
        self.missing_target: Optional[str] = None
        self.missing_features: List[str] = []
        self.missing_source_columns: List[str] = []  # For engineered features
        self.presumed_engineered: List[str] = []  # Columns assumed to be engineered (warning, not error)
        self.available_columns: Set[str] = set()
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def add_error(self, error: str):
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a validation warning."""
        self.warnings.append(warning)

    def to_feedback(self) -> str:
        """Generate feedback string for the dataset designer agent."""
        if self.is_valid:
            return "Dataset specification is valid. All columns exist in the data source."

        feedback_parts = ["## Dataset Validation Failed\n"]

        if self.missing_target:
            feedback_parts.append(f"### ❌ Missing Target Column: `{self.missing_target}`\n")
            feedback_parts.append(f"The target column `{self.missing_target}` does not exist in the data source.\n\n")
            feedback_parts.append("**You have TWO options:**\n\n")
            feedback_parts.append("**Option 1: Pick an existing column** (recommended)\n")
            feedback_parts.append("Choose a target from the available columns listed below.\n\n")
            feedback_parts.append("**Option 2: Engineer the target** (advanced)\n")
            feedback_parts.append("If you want to CREATE this target from other columns, you must provide:\n")
            feedback_parts.append("- `target_exists: false`\n")
            feedback_parts.append("- `target_creation` with:\n")
            feedback_parts.append("  - `formula`: How to compute the target (e.g., 'channel with highest revenue')\n")
            feedback_parts.append("  - `source_columns`: Which existing columns are needed\n")
            feedback_parts.append("  - `description`: What the engineered target represents\n\n")

        if self.missing_features:
            feedback_parts.append(f"\n### Missing Feature Columns ({len(self.missing_features)})\n")
            feedback_parts.append("The following feature columns do not exist in the data source:\n")
            for col in self.missing_features[:20]:  # Limit to 20 for readability
                feedback_parts.append(f"- `{col}`\n")
            if len(self.missing_features) > 20:
                feedback_parts.append(f"... and {len(self.missing_features) - 20} more\n")

        if self.presumed_engineered:
            feedback_parts.append(f"\n### Presumed Engineered Features ({len(self.presumed_engineered)})\n")
            feedback_parts.append("These columns are not in the raw data but are assumed to be engineered:\n")
            for col in self.presumed_engineered[:20]:
                feedback_parts.append(f"- `{col}` (will be created via feature engineering)\n")
            if len(self.presumed_engineered) > 20:
                feedback_parts.append(f"... and {len(self.presumed_engineered) - 20} more\n")
            feedback_parts.append("\n**Note**: The audit agent should verify these engineered features are properly defined.\n")

        if self.missing_source_columns:
            feedback_parts.append(f"\n### Missing Columns for Engineered Features ({len(self.missing_source_columns)})\n")
            feedback_parts.append("The following source columns needed for feature engineering do not exist:\n")
            for col in self.missing_source_columns[:10]:
                feedback_parts.append(f"- `{col}`\n")

        # Add available columns for reference
        feedback_parts.append(f"\n### Available Columns ({len(self.available_columns)})\n")
        feedback_parts.append("The following columns are actually available in the data source:\n")
        sorted_cols = sorted(self.available_columns)
        for col in sorted_cols[:50]:  # Limit to 50 for readability
            feedback_parts.append(f"- `{col}`\n")
        if len(sorted_cols) > 50:
            feedback_parts.append(f"... and {len(sorted_cols) - 50} more\n")

        feedback_parts.append("\n### Action Required\n")
        feedback_parts.append("Please revise the dataset design to only use columns that exist in the data source. ")
        feedback_parts.append("Check the available columns list above and select appropriate features from there.\n")

        return "".join(feedback_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "missing_target": self.missing_target,
            "missing_features": self.missing_features,
            "missing_source_columns": self.missing_source_columns,
            "presumed_engineered": self.presumed_engineered,
            "available_columns": list(self.available_columns),
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DatasetValidator:
    """Validates dataset specifications against actual data sources."""

    def __init__(self, db: Session):
        self.db = db

    def get_available_columns(self, data_source_ids: List[str | UUID]) -> Set[str]:
        """Get all available column names from the specified data sources.

        Args:
            data_source_ids: List of data source UUIDs

        Returns:
            Set of all column names available across all data sources
        """
        all_columns: Set[str] = set()

        for source_id in data_source_ids:
            if isinstance(source_id, str):
                source_id = UUID(source_id)

            source = self.db.query(DataSource).filter(DataSource.id == source_id).first()
            if not source:
                continue

            schema_summary = source.schema_summary or {}
            columns = schema_summary.get("columns", [])

            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name") or col.get("column_name")
                    if col_name:
                        all_columns.add(col_name)
                elif isinstance(col, str):
                    all_columns.add(col)

        return all_columns

    def validate_columns(
        self,
        target_column: Optional[str],
        feature_columns: List[str],
        available_columns: Set[str],
        engineered_features: Optional[List[Dict[str, Any]]] = None,
    ) -> DatasetValidationResult:
        """Validate that columns exist in the available columns set.

        Args:
            target_column: The target column name
            feature_columns: List of feature column names
            available_columns: Set of columns that actually exist
            engineered_features: Optional list of engineered feature definitions

        Returns:
            DatasetValidationResult with validation details
        """
        result = DatasetValidationResult()
        result.available_columns = available_columns

        # Validate target column
        if target_column:
            # Check if target exists directly or will be created by engineering
            target_is_engineered = False
            if engineered_features:
                for feat in engineered_features:
                    if feat.get("output_column") == target_column:
                        target_is_engineered = True
                        break

            if not target_is_engineered and target_column not in available_columns:
                result.missing_target = target_column
                result.add_error(f"Target column '{target_column}' not found in data source")

        # Validate feature columns
        for col in feature_columns:
            # Check if it's a base column that exists
            if col in available_columns:
                continue

            # Check if it will be created via explicit engineered_features definition
            is_explicitly_engineered = False
            if engineered_features:
                for feat in engineered_features:
                    if feat.get("output_column") == col:
                        is_explicitly_engineered = True
                        break

            if is_explicitly_engineered:
                # Has a definition - validated via source_columns check below
                continue

            # Column not in raw data and no explicit definition
            # Assume it's an intended engineered feature (data scientist creating new columns)
            # This is a WARNING, not an error - audit agent can verify it makes sense
            result.presumed_engineered.append(col)

        if result.presumed_engineered:
            result.add_warning(
                f"{len(result.presumed_engineered)} feature column(s) not in raw data - "
                f"assuming these will be created via feature engineering: {result.presumed_engineered[:5]}"
                + ("..." if len(result.presumed_engineered) > 5 else "")
            )

        # Validate source columns for engineered features
        if engineered_features:
            for feat in engineered_features:
                source_cols = feat.get("source_columns", [])
                for src_col in source_cols:
                    if src_col not in available_columns:
                        if src_col not in result.missing_source_columns:
                            result.missing_source_columns.append(src_col)

            if result.missing_source_columns:
                result.add_error(f"{len(result.missing_source_columns)} source column(s) for engineered features not found")

        return result

    def validate_dataset_spec(self, dataset_spec_id: UUID) -> DatasetValidationResult:
        """Validate a DatasetSpec against its data sources.

        Args:
            dataset_spec_id: UUID of the DatasetSpec to validate

        Returns:
            DatasetValidationResult with validation details
        """
        result = DatasetValidationResult()

        # Get the dataset spec
        spec = self.db.query(DatasetSpec).filter(DatasetSpec.id == dataset_spec_id).first()
        if not spec:
            result.add_error(f"DatasetSpec {dataset_spec_id} not found")
            return result

        # Get data source IDs
        source_ids = self._extract_source_ids(spec)
        if not source_ids:
            result.add_error("No data sources configured in dataset spec")
            return result

        # Get available columns from all sources
        available_columns = self.get_available_columns(source_ids)
        if not available_columns:
            result.add_error("Could not retrieve columns from data sources")
            return result

        # Get engineered features from spec_json
        spec_json = spec.spec_json or {}
        engineered_features = list(spec_json.get("engineered_features", []))

        # Check for target_creation (engineered target from problem_understanding)
        # If target_creation exists, assume target is engineered (target_exists defaults to True but may not have been saved)
        target_creation = spec_json.get("target_creation")
        target_exists = spec_json.get("target_exists", True)
        # If target_creation is defined, always treat it as engineered regardless of target_exists flag
        # This handles legacy specs that didn't save target_exists
        if target_creation and spec.target_column:
            # Add target as an engineered feature so validation knows it will be created
            engineered_features.append({
                "output_column": spec.target_column,
                "source_columns": target_creation.get("source_columns", []),
                "formula": target_creation.get("formula", ""),
                "description": target_creation.get("description", "Engineered target"),
            })

        # Validate columns
        return self.validate_columns(
            target_column=spec.target_column,
            feature_columns=spec.feature_columns or [],
            available_columns=available_columns,
            engineered_features=engineered_features,
        )

    def validate_dataset_design(
        self,
        target_column: str,
        feature_columns: List[str],
        data_source_ids: List[str | UUID],
        engineered_features: Optional[List[Dict[str, Any]]] = None,
    ) -> DatasetValidationResult:
        """Validate a dataset design (before creating DatasetSpec).

        Args:
            target_column: The target column name
            feature_columns: List of feature column names
            data_source_ids: List of data source UUIDs
            engineered_features: Optional list of engineered feature definitions

        Returns:
            DatasetValidationResult with validation details
        """
        available_columns = self.get_available_columns(data_source_ids)

        if not available_columns:
            result = DatasetValidationResult()
            result.add_error("Could not retrieve columns from data sources")
            return result

        return self.validate_columns(
            target_column=target_column,
            feature_columns=feature_columns,
            available_columns=available_columns,
            engineered_features=engineered_features,
        )

    def _extract_source_ids(self, spec: DatasetSpec) -> List[str]:
        """Extract data source IDs from a DatasetSpec."""
        source_ids = []
        data_sources_config = spec.data_sources_json or {}

        if isinstance(data_sources_config, list):
            source_ids = data_sources_config
        else:
            source_ids = data_sources_config.get("sources", [])
            if not source_ids:
                source_ids = data_sources_config.get("source_ids", [])
            if not source_ids:
                source_id = data_sources_config.get("source_id")
                if source_id:
                    source_ids = [source_id]
            if not source_ids:
                primary_id = data_sources_config.get("primary")
                if primary_id:
                    source_ids = [primary_id]

        # Handle both dict format and direct ID format
        result = []
        for entry in source_ids:
            if isinstance(entry, dict):
                sid = entry.get("source_id") or entry.get("id")
                if sid:
                    result.append(str(sid))
            else:
                result.append(str(entry))

        return result


def validate_dataset_columns(
    db: Session,
    target_column: str,
    feature_columns: List[str],
    data_source_ids: List[str | UUID],
    engineered_features: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Convenience function to validate dataset columns.

    Args:
        db: Database session
        target_column: The target column name
        feature_columns: List of feature column names
        data_source_ids: List of data source UUIDs
        engineered_features: Optional list of engineered feature definitions

    Returns:
        Tuple of (is_valid, feedback_string, validation_details_dict)
    """
    validator = DatasetValidator(db)
    result = validator.validate_dataset_design(
        target_column=target_column,
        feature_columns=feature_columns,
        data_source_ids=data_source_ids,
        engineered_features=engineered_features,
    )
    return result.is_valid, result.to_feedback(), result.to_dict()

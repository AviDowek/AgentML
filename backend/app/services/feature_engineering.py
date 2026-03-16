"""Feature engineering service for creating derived columns.

Implements the execution of FeatureEngineeringStep and TargetColumnCreation
definitions from the agent schemas.

CHANGELOG (for revert tracking):
- 2024-01: Added unified SAFE_BUILTINS constant for consistency between validation and execution
- 2024-01: Added 'out' variable support in validation (was only in execution)
- 2024-01: Improved MultiIndex handling with explicit NaN detection
- 2024-01: Improved regex patterns for edge cases
- 2024-01: Added explicit failure tracking in non-strict mode
- 2025-01: Fixed _is_multistatement_formula using AST instead of regex (was matching keyword args as assignments)
- 2025-01: Added auto-detection of result variables in multi-statement formulas (handles 'assignment' etc.)
- 2025-01: Added auto-capture of bare expression results (handles formulas where last line is np.where without assignment)
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED SAFE_BUILTINS - Single source of truth for all formula execution
# =============================================================================
# This constant ensures validation and execution use identical builtins.
# If you need to add a builtin, add it HERE and it will be available everywhere.
#
# FIX: Previously, validation (line ~185) and execution (line ~701) had
# different builtins, causing validation/execution mismatches.
# =============================================================================

def _get_safe_builtins() -> dict:
    """Get the unified safe builtins dictionary.

    Returns a fresh dict each time to avoid mutation issues.
    numpy and pandas are imported fresh to ensure they're available.

    This is the SINGLE SOURCE OF TRUTH for builtins in formula execution.
    Both validation and execution MUST use this function.
    """
    # Import here to ensure they're always available
    import numpy as _np
    import pandas as _pd

    builtins = {
        # Basic types
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "object": object,  # Needed for np.full(..., dtype=object)
        # Common functions
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "any": any,
        "all": all,
        "isinstance": isinstance,
        "type": type,
        # Important: numpy/pandas must be in builtins for lambda closures
        "np": _np,
        "pd": _pd,
    }

    logger.debug(f"[SAFE_BUILTINS] Returning unified builtins with {len(builtins)} entries")
    return builtins


# Log that unified builtins are being used (helps verify fix is active)
logger.info("[FEATURE_ENGINEERING] Module loaded with unified SAFE_BUILTINS")


class FeatureEngineeringError(Exception):
    """Error during feature engineering execution."""
    pass


@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering with feedback for LLM iteration."""
    df: pd.DataFrame
    successful_features: List[Dict[str, Any]] = field(default_factory=list)
    failed_features: List[Dict[str, Any]] = field(default_factory=list)
    available_columns: List[str] = field(default_factory=list)

    def get_feedback_for_llm(self) -> str:
        """Generate feedback string for the LLM about what worked and failed."""
        lines = []

        if self.successful_features:
            lines.append(f"✓ Successfully created {len(self.successful_features)} features:")
            for f in self.successful_features[:10]:  # Limit to first 10
                lines.append(f"  - {f['feature']}: {f.get('description', 'N/A')}")

        if self.failed_features:
            lines.append(f"\n✗ Failed to create {len(self.failed_features)} features:")
            for f in self.failed_features:
                lines.append(f"  - {f['feature']}: {f['error']}")
            lines.append(f"\nAvailable columns for feature engineering: {self.available_columns[:30]}")

        return "\n".join(lines) if lines else "No feature engineering was attempted."


@dataclass
class FormulaValidationResult:
    """Result of formula pre-validation."""
    is_valid: bool
    formula: str
    output_column: str
    error_message: Optional[str] = None
    error_type: Optional[str] = None  # 'syntax', 'column_missing', 'dtype', 'runtime'
    suggested_fix: Optional[str] = None
    available_columns: List[str] = field(default_factory=list)

    def get_agent_feedback(self) -> str:
        """Generate actionable feedback for the agent to fix the formula."""
        if self.is_valid:
            return f"✓ Formula for '{self.output_column}' is valid."

        lines = [
            f"❌ FORMULA VALIDATION FAILED for '{self.output_column}'",
            f"   Error Type: {self.error_type}",
            f"   Error: {self.error_message}",
        ]

        if self.suggested_fix:
            lines.append(f"   Suggested Fix: {self.suggested_fix}")

        if self.error_type == 'column_missing':
            lines.append(f"   Available columns: {', '.join(self.available_columns[:20])}")
            if len(self.available_columns) > 20:
                lines.append(f"   ... and {len(self.available_columns) - 20} more")

        return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS FOR REGEX EDGE CASES
# =============================================================================
# FIX: Previous regex patterns had edge case issues:
# - ["\'][^"\']*["\'] didn't handle escaped quotes
# - Commented line detection could match strings
# =============================================================================

def _has_string_literals(formula: str) -> bool:
    """Check if formula contains string literals (not in comments).

    FIX: Previous regex r'[\"\\'][^\"\\']*[\"\\']' didn't handle escaped quotes.
    This function removes comments first, then uses a safer heuristic.

    Args:
        formula: The formula string to check

    Returns:
        True if string literals are found
    """
    # Remove full-line comments first
    lines = formula.split('\n')
    code_only = []
    for line in lines:
        # Find # that's not inside a string (best effort heuristic)
        # Check if # appears and there's no quote after it on the same line
        # This isn't perfect but handles common cases
        if '#' in line:
            # Simple heuristic: if the # is followed by end of line or non-quote chars
            # and preceded by code, treat it as a comment
            hash_pos = line.find('#')
            before_hash = line[:hash_pos]
            # Count unescaped quotes before #
            single_quotes = before_hash.count("'") - before_hash.count("\\'")
            double_quotes = before_hash.count('"') - before_hash.count('\\"')
            # If both counts are even, # is outside strings
            if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                line = before_hash
        code_only.append(line)

    cleaned = '\n'.join(code_only)

    # Now check for strings using a safer pattern
    # Look for quote followed by content followed by matching quote
    has_single = "'" in cleaned and cleaned.count("'") >= 2
    has_double = '"' in cleaned and cleaned.count('"') >= 2

    result = has_single or has_double
    if result:
        logger.debug(f"[REGEX_FIX] Detected string literals in formula")
    return result


def _is_commented_assignment(line: str, output_column: str) -> bool:
    """Check if line is a commented-out assignment (not just a comment mentioning it).

    FIX: Previous pattern could match comments that just mentioned the column name.
    This function ensures the comment is at the start and the assignment is real code.

    Args:
        line: A single line from the formula
        output_column: The expected output column name

    Returns:
        True if this is a commented-out assignment that should be uncommented
    """
    stripped = line.strip()
    if not stripped.startswith('#'):
        return False

    # Must be at start of "real" code (after comment marker)
    after_hash = stripped[1:].lstrip()

    # Pattern: df['column'] = or df["column"] =
    pattern = rf'^df\s*\[\s*[\'\"]{re.escape(output_column)}[\'\"]\s*\]\s*='
    is_match = bool(re.match(pattern, after_hash))

    if is_match:
        logger.debug(f"[REGEX_FIX] Detected commented-out assignment to df['{output_column}']")

    return is_match


def validate_formula(
    df: pd.DataFrame,
    formula: str,
    output_column: str,
    source_columns: Optional[List[str]] = None,
) -> FormulaValidationResult:
    """Pre-validate a formula before experiment execution.

    Performs:
    1. Syntax check (can the formula be parsed?)
    2. Column existence check (do referenced columns exist?)
    3. Dry-run on sample data (does it execute without errors?)

    Returns actionable feedback if validation fails.

    Args:
        df: DataFrame to validate against (or sample of it)
        formula: The formula string to validate
        output_column: Name of the output column
        source_columns: Optional list of columns the formula uses

    Returns:
        FormulaValidationResult with validation status and feedback

    FIX APPLIED: Now uses _get_safe_builtins() for consistency with execution.
    FIX APPLIED: Now checks for 'out' variable (was missing, only execution had it).
    """
    logger.debug(f"[VALIDATION] Validating formula for '{output_column}'")

    available_columns = list(df.columns)

    # Step 0: Apply auto-fixes before validation (same as execution)
    original_formula = formula

    # Fix np.nan with strings in np.where - use improved regex for edge cases
    if 'np.where' in formula and 'np.nan' in formula:
        if _has_string_literals(formula):
            formula = re.sub(r'\bnp\.nan\b', 'None', formula)
            logger.info(f"[VALIDATION] Auto-fixed np.nan -> None for '{output_column}'")

    # Step 1: Check for obviously problematic patterns
    # Check for import statements (not allowed)
    if re.search(r'\bimport\s+\w+', formula):
        return FormulaValidationResult(
            is_valid=False,
            formula=original_formula,
            output_column=output_column,
            error_type='syntax',
            error_message="Formula contains import statement which is not allowed.",
            suggested_fix="Remove import statements. np, pd, and df are already available.",
            available_columns=available_columns,
        )

    # Step 2: Check if referenced columns exist
    if source_columns:
        missing_cols = [col for col in source_columns if col not in available_columns]
        if missing_cols:
            # Try to find similar column names
            suggestions = []
            for missing in missing_cols:
                similar = [c for c in available_columns if missing.lower() in c.lower() or c.lower() in missing.lower()]
                if similar:
                    suggestions.append(f"'{missing}' -> maybe '{similar[0]}'?")

            return FormulaValidationResult(
                is_valid=False,
                formula=original_formula,
                output_column=output_column,
                error_type='column_missing',
                error_message=f"Column(s) not found: {missing_cols}",
                suggested_fix=f"Check column names. {'; '.join(suggestions)}" if suggestions else "Use exact column names from the available columns list.",
                available_columns=available_columns,
            )

    # Step 3: Try to compile the formula (syntax check)
    try:
        compile(formula, '<formula>', 'eval')
    except SyntaxError as e:
        # Check if it's a multi-statement that needs exec instead
        try:
            compile(formula, '<formula>', 'exec')
            # It's valid as a multi-statement, that's OK
        except SyntaxError as e2:
            return FormulaValidationResult(
                is_valid=False,
                formula=original_formula,
                output_column=output_column,
                error_type='syntax',
                error_message=f"Syntax error: {str(e2)}",
                suggested_fix="Check for matching parentheses, quotes, and brackets. Ensure valid Python syntax.",
                available_columns=available_columns,
            )

    # Step 4: Dry-run on a small sample
    try:
        # Use a small sample for speed
        sample_size = min(10, len(df))
        if sample_size == 0:
            # Empty dataframe - can't validate
            return FormulaValidationResult(
                is_valid=True,
                formula=formula,
                output_column=output_column,
            )

        sample_df = df.head(sample_size).copy()

        # Build execution context using UNIFIED safe_builtins
        # FIX: Previously used inline dict that was missing 'object' and 'reversed'
        safe_builtins = _get_safe_builtins()
        logger.debug(f"[VALIDATION] Using unified safe_builtins for '{output_column}'")

        exec_context = {
            "df": sample_df,
            "pd": pd,
            "np": np,
            **{col: sample_df[col] for col in sample_df.columns},
        }

        # Try eval first
        try:
            result = eval(formula, {"__builtins__": safe_builtins, **exec_context}, exec_context)
        except SyntaxError:
            # Try exec for multi-statement
            exec_context_copy = exec_context.copy()
            exec_context_copy["df"] = sample_df
            exec(formula, {"__builtins__": safe_builtins, **exec_context_copy}, exec_context_copy)

            # Check if result was assigned - FIX: Added 'out' variable check
            # This matches the execution logic in _execute_multistatement_formula
            if output_column in exec_context_copy.get("df", sample_df).columns:
                result = exec_context_copy["df"][output_column]
                logger.debug(f"[VALIDATION] Found result in df['{output_column}']")
            elif "result" in exec_context_copy:
                result = exec_context_copy["result"]
                logger.debug(f"[VALIDATION] Found result in 'result' variable")
            elif "out" in exec_context_copy:
                # FIX: This was missing! Execution accepted 'out' but validation didn't
                result = exec_context_copy["out"]
                logger.debug(f"[VALIDATION] Found result in 'out' variable (newly supported)")
            elif output_column and output_column in exec_context_copy:
                result = exec_context_copy[output_column]
                logger.debug(f"[VALIDATION] Found result in '{output_column}' variable")
            else:
                return FormulaValidationResult(
                    is_valid=False,
                    formula=original_formula,
                    output_column=output_column,
                    error_type='runtime',
                    error_message=f"Multi-statement formula did not assign to df['{output_column}'], 'result', or 'out' variable.",
                    suggested_fix=f"Ensure your formula ends with: df['{output_column}'] = <expr>, result = <expr>, or out = <expr>",
                    available_columns=available_columns,
                )

        # Validation passed
        return FormulaValidationResult(
            is_valid=True,
            formula=formula,
            output_column=output_column,
        )

    except KeyError as e:
        col_name = str(e).strip("'\"")
        # Find similar columns
        similar = [c for c in available_columns if col_name.lower() in c.lower() or c.lower() in col_name.lower()]
        suggestion = f"Did you mean '{similar[0]}'?" if similar else "Check the exact column name."

        return FormulaValidationResult(
            is_valid=False,
            formula=original_formula,
            output_column=output_column,
            error_type='column_missing',
            error_message=f"Column '{col_name}' not found in DataFrame.",
            suggested_fix=suggestion,
            available_columns=available_columns,
        )

    except Exception as e:
        error_str = str(e)

        # Detect specific error types and provide targeted fixes
        if "DType" in error_str or "dtype" in error_str.lower():
            return FormulaValidationResult(
                is_valid=False,
                formula=original_formula,
                output_column=output_column,
                error_type='dtype',
                error_message=f"Data type error: {error_str}",
                suggested_fix="When using np.where() with strings, use None instead of np.nan: np.where(cond, 'value', None)",
                available_columns=available_columns,
            )

        if "unsupported operand" in error_str.lower():
            return FormulaValidationResult(
                is_valid=False,
                formula=original_formula,
                output_column=output_column,
                error_type='dtype',
                error_message=f"Type mismatch: {error_str}",
                suggested_fix="Check that you're not mixing incompatible types (e.g., string + number). Use .astype() to convert if needed.",
                available_columns=available_columns,
            )

        if "division" in error_str.lower() or "zero" in error_str.lower():
            return FormulaValidationResult(
                is_valid=False,
                formula=original_formula,
                output_column=output_column,
                error_type='runtime',
                error_message=f"Division error: {error_str}",
                suggested_fix="Add a small epsilon to avoid division by zero: df['col'] / (df['other'] + 1e-9)",
                available_columns=available_columns,
            )

        # Generic runtime error
        return FormulaValidationResult(
            is_valid=False,
            formula=original_formula,
            output_column=output_column,
            error_type='runtime',
            error_message=error_str[:200],
            suggested_fix="Check the formula syntax and ensure all referenced columns exist.",
            available_columns=available_columns,
        )


def validate_feature_engineering_batch(
    df: pd.DataFrame,
    engineered_features: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[FormulaValidationResult]]:
    """Validate a batch of feature engineering definitions before execution.

    Args:
        df: DataFrame to validate against
        engineered_features: List of feature engineering definitions

    Returns:
        Tuple of (valid_features, validation_failures)
        - valid_features: List of features that passed validation
        - validation_failures: List of FormulaValidationResult for failures
    """
    valid_features = []
    failures = []

    for feature_def in engineered_features:
        output_column = feature_def.get("output_column") or feature_def.get("name")
        formula = feature_def.get("formula", "")
        source_columns = feature_def.get("source_columns", [])

        if not formula:
            failures.append(FormulaValidationResult(
                is_valid=False,
                formula="",
                output_column=output_column or "unknown",
                error_type='syntax',
                error_message="Empty formula provided.",
                suggested_fix="Provide a valid pandas/numpy expression.",
                available_columns=list(df.columns),
            ))
            continue

        result = validate_formula(df, formula, output_column, source_columns)

        if result.is_valid:
            valid_features.append(feature_def)
        else:
            failures.append(result)

    return valid_features, failures


def get_validation_feedback_for_agent(failures: List[FormulaValidationResult]) -> str:
    """Generate consolidated feedback for the agent about validation failures.

    This feedback is designed to be included in prompts to help agents
    fix their formulas before re-attempting.

    Args:
        failures: List of validation failures

    Returns:
        Formatted string with actionable feedback
    """
    if not failures:
        return ""

    lines = [
        "## ⚠️ FORMULA VALIDATION FAILURES",
        "",
        f"**{len(failures)} feature engineering formula(s) failed validation.**",
        "These formulas will NOT execute successfully. You MUST fix them:",
        "",
    ]

    for i, failure in enumerate(failures, 1):
        lines.append(f"### {i}. `{failure.output_column}`")
        lines.append(f"- **Error Type**: {failure.error_type}")
        lines.append(f"- **Error**: {failure.error_message}")
        if failure.suggested_fix:
            lines.append(f"- **How to Fix**: {failure.suggested_fix}")
        lines.append("")

    lines.append("**CRITICAL**: Do NOT repeat these exact formulas. Apply the suggested fixes.")
    lines.append("")

    return "\n".join(lines)


def apply_feature_engineering_with_feedback(
    df: pd.DataFrame,
    engineered_features: List[Dict[str, Any]],
    inplace: bool = False,
) -> FeatureEngineeringResult:
    """Apply feature engineering and return detailed feedback.

    This version returns structured feedback about successes and failures,
    which can be fed back to the LLM for smarter iteration.

    Args:
        df: Input DataFrame
        engineered_features: List of feature engineering step definitions
        inplace: If True, modify df in place

    Returns:
        FeatureEngineeringResult with df and feedback
    """
    if not inplace:
        df = df.copy()

    available_columns = list(df.columns)
    successful = []
    failed = []

    if not engineered_features:
        return FeatureEngineeringResult(
            df=df,
            available_columns=available_columns,
        )

    for step in engineered_features:
        output_column = step.get("output_column")
        formula = step.get("formula")
        source_columns = step.get("source_columns", [])
        description = step.get("description", "")

        if not output_column or not formula:
            failed.append({
                "feature": output_column or "unknown",
                "error": "Missing output_column or formula in step definition",
                "step": step,
            })
            continue

        # Validate source columns exist
        missing_cols = [col for col in source_columns if col not in df.columns]
        if missing_cols:
            failed.append({
                "feature": output_column,
                "error": f"Missing source columns: {missing_cols}",
                "formula": formula,
                "available_similar": _find_similar_columns(missing_cols, df.columns),
            })
            continue

        try:
            new_column = _execute_formula(df, formula, source_columns, output_column)
            df[output_column] = new_column
            successful.append({
                "feature": output_column,
                "description": description,
                "formula": formula,
            })
            logger.info(f"Created feature '{output_column}': {description}")
        except Exception as e:
            failed.append({
                "feature": output_column,
                "error": str(e),
                "formula": formula,
            })
            logger.warning(f"Failed to create '{output_column}': {e}")

    return FeatureEngineeringResult(
        df=df,
        successful_features=successful,
        failed_features=failed,
        available_columns=list(df.columns),
    )


def _find_similar_columns(missing: List[str], available: pd.Index) -> List[str]:
    """Find columns with similar names to help LLM correct mistakes."""
    similar = []
    available_lower = {col.lower(): col for col in available}

    for col in missing:
        col_lower = col.lower()
        # Exact match different case
        if col_lower in available_lower:
            similar.append(f"{col} -> try '{available_lower[col_lower]}'")
        else:
            # Partial match
            for avail in available:
                if col_lower in avail.lower() or avail.lower() in col_lower:
                    similar.append(f"{col} -> maybe '{avail}'?")
                    break

    return similar[:5]  # Limit suggestions


def apply_feature_engineering(
    df: pd.DataFrame,
    engineered_features: List[Dict[str, Any]],
    inplace: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Apply feature engineering steps to create new columns.

    Args:
        df: Input DataFrame
        engineered_features: List of feature engineering step definitions.
            Each step should have:
            - output_column: Name of the new column to create
            - formula: Python/pandas expression to compute the column
            - source_columns: List of source columns used in the formula
            - description: Human-readable description
        inplace: If True, modify df in place. Otherwise, return a copy.
        strict: If True, raise on first error. If False, skip failed features and continue.

    Returns:
        DataFrame with new engineered columns added.
        Note: In non-strict mode, check df.attrs['_feature_engineering_failures'] for any failures.

    Raises:
        FeatureEngineeringError: If a formula fails to execute (only in strict mode)

    FIX APPLIED: Now stores failure details in df.attrs for explicit notification.
    """
    if not inplace:
        df = df.copy()

    if not engineered_features:
        return df

    failed_features = []
    successful_features = []
    total_requested = len(engineered_features)

    logger.info(f"[FEATURE_ENGINEERING] Starting to process {total_requested} feature(s)")

    for step in engineered_features:
        output_column = step.get("output_column")
        formula = step.get("formula")
        source_columns = step.get("source_columns", [])
        description = step.get("description", "")

        if not output_column or not formula:
            logger.warning(f"[FEATURE_ENGINEERING] Skipping invalid step (missing column/formula): {step}")
            failed_features.append({
                "feature": output_column or "unknown",
                "error": "Missing output_column or formula",
                "formula": formula or ""
            })
            continue

        # Validate source columns exist
        missing_cols = [col for col in source_columns if col not in df.columns]
        if missing_cols:
            error_msg = (
                f"Cannot create '{output_column}': missing source columns {missing_cols}. "
                f"Available columns: {list(df.columns)[:20]}..."
            )
            if strict:
                raise FeatureEngineeringError(error_msg)
            else:
                logger.warning(f"[FEATURE_ENGINEERING] SKIPPED '{output_column}': {error_msg}")
                failed_features.append({
                    "feature": output_column,
                    "error": error_msg,
                    "formula": formula
                })
                continue

        try:
            # Execute the formula
            # The formula can use 'df' to reference the DataFrame
            new_column = _execute_formula(df, formula, source_columns, output_column)
            df[output_column] = new_column
            successful_features.append(output_column)
            logger.info(f"[FEATURE_ENGINEERING] Created '{output_column}': {description}")
        except Exception as e:
            error_msg = f"Failed to create '{output_column}' with formula '{formula}': {e}"
            if strict:
                raise FeatureEngineeringError(error_msg) from e
            else:
                logger.warning(f"[FEATURE_ENGINEERING] FAILED '{output_column}': {e}")
                failed_features.append({
                    "feature": output_column,
                    "error": str(e),
                    "formula": formula
                })
                continue

    # FIX: Store failure details in DataFrame attrs for explicit notification
    # This allows callers to check if features were skipped without changing the return type
    if failed_features:
        df.attrs['_feature_engineering_failures'] = failed_features
        df.attrs['_feature_engineering_success_count'] = len(successful_features)
        df.attrs['_feature_engineering_failure_count'] = len(failed_features)
        df.attrs['_feature_engineering_total_requested'] = total_requested

        # Log prominently so it's visible in logs
        logger.error(
            f"[FEATURE_ENGINEERING] ⚠️ PARTIAL FAILURE: {len(failed_features)}/{total_requested} features FAILED! "
            f"Failed: {[f['feature'] for f in failed_features]}. "
            f"Successful: {successful_features}. "
            f"Check df.attrs['_feature_engineering_failures'] for details."
        )
    else:
        # Clear any previous failure info
        df.attrs.pop('_feature_engineering_failures', None)
        df.attrs.pop('_feature_engineering_failure_count', None)
        logger.info(
            f"[FEATURE_ENGINEERING] ✓ All {len(successful_features)}/{total_requested} features created successfully"
        )

    return df


def apply_target_creation(
    df: pd.DataFrame,
    target_creation: Dict[str, Any],
    inplace: bool = False,
) -> pd.DataFrame:
    """Create a target column using the specified formula.

    Args:
        df: Input DataFrame
        target_creation: Target creation definition with:
            - column_name: Name of the target column to create
            - formula: Python/pandas expression to compute the target
            - source_columns: List of source columns used in the formula
            - description: Human-readable description
            - data_type: Optional data type for the result (default: infer)
        inplace: If True, modify df in place. Otherwise, return a copy.

    Returns:
        DataFrame with the target column added

    Raises:
        FeatureEngineeringError: If target creation fails
    """
    if not inplace:
        df = df.copy()

    if not target_creation:
        return df

    column_name = target_creation.get("column_name")
    formula = target_creation.get("formula")
    source_columns = target_creation.get("source_columns", [])
    description = target_creation.get("description", "")
    data_type = target_creation.get("data_type", "infer")

    if not column_name or not formula:
        raise FeatureEngineeringError(
            f"Invalid target creation: must have column_name and formula. Got: {target_creation}"
        )

    # Validate source columns exist
    missing_cols = [col for col in source_columns if col not in df.columns]
    if missing_cols:
        raise FeatureEngineeringError(
            f"Cannot create target '{column_name}': missing source columns {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    try:
        # Execute the formula
        new_column = _execute_formula(df, formula, source_columns, column_name)

        # Apply data type conversion if specified
        if data_type and data_type != "infer":
            new_column = _convert_dtype(new_column, data_type)

        df[column_name] = new_column
        logger.info(f"Created target column '{column_name}': {description}")

        # For target creation, we often need to drop rows where target is NaN
        # (e.g., shifted values at the end of time series)
        nan_count = df[column_name].isna().sum()
        if nan_count > 0:
            logger.info(f"Target column has {nan_count} NaN values (expected for shifted targets)")

    except Exception as e:
        raise FeatureEngineeringError(
            f"Failed to create target '{column_name}' with formula '{formula}': {e}"
        ) from e

    return df


def _execute_formula(
    df: pd.DataFrame,
    formula: str,
    source_columns: List[str],
    output_column: Optional[str] = None,
) -> pd.Series:
    """Execute a formula to create a new column.

    The formula is evaluated as a Python expression with access to:
    - df: The input DataFrame
    - pd: pandas module
    - np: numpy module
    - Individual columns can be referenced as df["column_name"]

    Common formula patterns:
    - df["close"].shift(-1) > df["close"]  # Binary target for price prediction
    - df["high"] - df["low"]  # Price range
    - df["date"].dt.dayofweek  # Day of week extraction
    - (df["col1"] + df["col2"]) / 2  # Average

    Args:
        df: Input DataFrame
        formula: Python/pandas expression
        source_columns: List of columns used (for validation only)

    Returns:
        Series with the computed values
    """
    import numpy as np

    # Log complex formulas for debugging but don't block - allow multi-line complex formulas
    if '\ndef ' in formula or formula.strip().startswith('def '):
        logger.info(f"Executing complex formula with function definition for '{output_column}'")

    # Fix common AI mistake: using np.nan with string values in np.where causes dtype errors
    # numpy can't mix string dtype with float (np.nan). Replace np.nan with None for string contexts.
    # Pattern: np.where(..., "string", ..., np.nan) or np.where(..., np.nan, "string")
    import re
    if 'np.where' in formula and 'np.nan' in formula:
        # Check if formula contains string literals (quoted values)
        if re.search(r'["\'][^"\']*["\']', formula):
            # Replace np.nan with None - works with any dtype including strings
            formula = re.sub(r'\bnp\.nan\b', 'None', formula)
            logger.info(f"Auto-fixed np.nan -> None for string compatibility in formula")

    # Fix common AI mistake: mixing integers with strings in np.where causes DTypePromotionError
    # Example: np.where(condition, 0, 'Unknown') fails because numpy can't promote int to str
    # Solution: Convert bare integers to strings when mixed with string literals
    # This handles nested np.where calls like: np.where(cond1, 0, np.where(cond2, 1, 'Unknown'))
    if 'np.where' in formula:
        # Check if formula contains string literals (quoted values like 'Unknown', "Missing", etc.)
        has_string_literal = re.search(r'["\'][^"\']*["\']', formula)
        # Check for bare integers as np.where arguments: ", 0)" ", 1," ", -1,"
        has_bare_int_arg = re.search(r',\s*(-?\d+)\s*[,)]', formula)

        if has_string_literal and has_bare_int_arg:
            # Convert bare integers that appear as np.where value arguments to strings
            # Pattern matches: ", 0)" ", 1," ", -1," (integers after comma, before comma or closing paren)
            # This converts integers to strings to match the string dtype
            original_formula = formula
            formula = re.sub(r',(\s*)(-?\d+)(\s*[,)])', r",\1'\2'\3", formula)
            if formula != original_formula:
                logger.info(f"Auto-fixed integer/string mixing in np.where: converted bare ints to strings")

    # Auto-convert date/datetime columns if formula uses .dt accessor
    # This handles the common case where date columns are stored as strings
    working_df = df
    if ".dt." in formula:
        working_df = df.copy()
        # Find potential date columns referenced in formula
        for col in source_columns:
            if col in working_df.columns:
                col_data = working_df[col]
                # Check if column is string/object type and looks like dates
                if col_data.dtype == "object" or str(col_data.dtype) == "string":
                    try:
                        # Try to convert to datetime
                        working_df[col] = pd.to_datetime(col_data, errors="coerce")
                        logger.debug(f"Auto-converted '{col}' from string to datetime for .dt accessor")
                    except Exception:
                        pass  # Leave as-is if conversion fails

    # Build a safe execution context using UNIFIED safe_builtins
    # FIX: Previously this was an inline dict that could drift from validation's dict.
    # Now both validation and execution use _get_safe_builtins() for consistency.
    #
    # IMPORTANT: We need to include builtins so that lambda functions inside
    # pandas operations (like .apply()) have access to np, int, float, etc.
    # Lambda functions need these in their globals, not just locals.
    safe_builtins = _get_safe_builtins()
    logger.debug(f"[EXECUTION] Using unified safe_builtins for '{output_column}'")

    exec_context = {
        "df": working_df,
        "pd": pd,
        "np": np,
        # Common aggregation functions
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "len": len,
        # Allow access to specific columns for convenience
        **{col: working_df[col] for col in working_df.columns},
    }

    # Evaluate the formula
    # Use safe_builtins as globals so lambda functions have access to np, int, etc.

    # Check if this is a multi-statement script (not a single expression)
    # Multi-statement scripts need exec() instead of eval()
    is_multistatement = _is_multistatement_formula(formula)

    if is_multistatement:
        # Use exec() for multi-statement scripts
        result = _execute_multistatement_formula(
            working_df, formula, output_column, exec_context, safe_builtins
        )
    else:
        try:
            result = eval(formula, {"__builtins__": safe_builtins, **exec_context}, exec_context)
        except Exception as e:
            # Try alternative syntax patterns
            result = _try_alternative_formulas(working_df, formula, exec_context, safe_builtins, e)

    # Ensure result is a Series
    if isinstance(result, pd.DataFrame):
        raise FeatureEngineeringError(
            f"Formula returned a DataFrame, expected a Series. "
            f"Make sure the formula produces a single column."
        )

    if not isinstance(result, pd.Series):
        # If it's a scalar, broadcast to Series
        result = pd.Series(result, index=df.index)

    # Handle MultiIndex results from groupby().apply() operations
    # When using groupby().apply(), the result has a MultiIndex (group_key, original_index)
    # We need to flatten this to match the original DataFrame's index
    #
    # FIX: Previous implementation could silently inject NaNs during reindex.
    # Now we explicitly detect and log this to prevent silent data corruption.
    if isinstance(result.index, pd.MultiIndex):
        logger.debug(f"[MULTIINDEX_FIX] Handling MultiIndex result for '{output_column}' with {result.index.nlevels} levels")

        # Track original NaN count to detect silent corruption
        original_nan_count = result.isna().sum() if hasattr(result, 'isna') else 0

        try:
            # Only drop level if we have multiple levels
            if result.index.nlevels > 1:
                result = result.droplevel(0)
                logger.debug(f"[MULTIINDEX_FIX] Dropped level 0, now have {result.index.nlevels} level(s)")
            else:
                logger.warning(f"[MULTIINDEX_FIX] MultiIndex has only 1 level, cannot droplevel for '{output_column}'")

            # If there are still duplicate indices after droplevel, we have an issue
            if result.index.duplicated().any():
                dup_count = result.index.duplicated().sum()
                logger.warning(
                    f"[MULTIINDEX_FIX] Found {dup_count} duplicate indices after droplevel for '{output_column}'. "
                    f"Using groupby().first() to resolve (may lose data)."
                )
                # Take first occurrence instead of silently reindexing with NaN injection
                result = result.groupby(level=0).first()

            # Reindex to original df index, but CHECK for NaN injection
            if not result.index.equals(df.index):
                pre_reindex_nan = result.isna().sum() if hasattr(result, 'isna') else 0
                result = result.reindex(df.index)
                post_reindex_nan = result.isna().sum() if hasattr(result, 'isna') else 0

                nan_injected = post_reindex_nan - pre_reindex_nan
                if nan_injected > 0:
                    logger.warning(
                        f"[MULTIINDEX_FIX] Reindexing for '{output_column}' INJECTED {nan_injected} NaN values! "
                        f"This may indicate a groupby operation that changed row ordering or count. "
                        f"Original NaN: {original_nan_count}, Final NaN: {post_reindex_nan}"
                    )

        except Exception as e:
            logger.error(f"[MULTIINDEX_FIX] droplevel failed for '{output_column}': {e}")
            # If droplevel fails, try to reset index and create new series
            try:
                if len(result) == len(df):
                    result = pd.Series(result.values, index=df.index)
                    logger.info(f"[MULTIINDEX_FIX] Fallback: aligned by position for '{output_column}'")
                else:
                    # Length mismatch - the formula fundamentally changed the data shape
                    raise FeatureEngineeringError(
                        f"Formula result has {len(result)} rows but DataFrame has {len(df)} rows. "
                        f"Groupby operations that change row count are not supported for column creation."
                    )
            except ValueError as ve:
                raise FeatureEngineeringError(
                    f"Cannot align MultiIndex result for '{output_column}': {ve}"
                )

    # Ensure the result index matches the original DataFrame index
    if len(result) == len(df) and not result.index.equals(df.index):
        # Same length but different index - realign
        result = pd.Series(result.values, index=df.index)

    return result


def _is_multistatement_formula(formula: str) -> bool:
    """Detect if a formula is a multi-statement script that needs exec() instead of eval().

    Uses Python's AST to reliably detect if a formula is:
    - A single expression (can use eval()) -> returns False
    - Multiple statements or assignments (needs exec()) -> returns True

    FIX: Previous regex-based detection incorrectly flagged keyword arguments
    (like bins=..., labels=...) as variable assignments, causing single expressions
    spread across multiple lines (like pd.cut(...)) to be wrongly detected as
    multi-statement formulas.

    The AST approach is more reliable because Python's parser understands the
    difference between keyword arguments and variable assignments.
    """
    import ast

    formula_stripped = formula.strip()
    if not formula_stripped:
        return False

    # First, try to parse as a single expression
    try:
        ast.parse(formula_stripped, mode='eval')
        # Successfully parsed as expression - it's NOT multi-statement
        logger.debug(f"[MULTISTATEMENT_CHECK] Formula parsed as single expression")
        return False
    except SyntaxError:
        # Can't parse as expression, check if it's valid as statements
        pass

    # Try to parse as multiple statements
    try:
        tree = ast.parse(formula_stripped, mode='exec')
        # Count the number of statements
        num_statements = len(tree.body)

        # If more than one statement, or single statement is an assignment, it's multi-statement
        if num_statements > 1:
            logger.debug(f"[MULTISTATEMENT_CHECK] Formula has {num_statements} statements")
            return True

        if num_statements == 1:
            stmt = tree.body[0]
            # An Assign or AugAssign at top level means we need exec
            if isinstance(stmt, (ast.Assign, ast.AugAssign)):
                logger.debug(f"[MULTISTATEMENT_CHECK] Formula is a single assignment statement")
                return True
            # An Expr wrapping an expression means it can be evaluated
            # (This shouldn't happen if 'eval' mode failed above, but just in case)
            if isinstance(stmt, ast.Expr):
                logger.debug(f"[MULTISTATEMENT_CHECK] Formula is a single expression statement")
                return False

        return False
    except SyntaxError as e:
        # If it doesn't parse at all, fall back to regex heuristic
        logger.warning(f"[MULTISTATEMENT_CHECK] Could not parse formula with AST: {e}")
        # Fallback: simple heuristic - look for semicolons or multiple lines with '=' at start
        if ';' in formula_stripped:
            return True
        lines = [l.strip() for l in formula_stripped.split('\n') if l.strip() and not l.strip().startswith('#')]
        assignment_count = sum(1 for l in lines if re.match(r'^[a-zA-Z_]\w*\s*=(?!=)', l))
        return assignment_count >= 2


def _execute_multistatement_formula(
    df: pd.DataFrame,
    formula: str,
    output_column: Optional[str],
    exec_context: dict,
    safe_builtins: dict,
) -> pd.Series:
    """Execute a multi-statement formula using exec() and extract the result.

    The formula can assign to df[output_column] or to a 'result' variable.
    """
    import re

    logger.info(f"Executing multi-statement formula for column '{output_column}'")

    # Strip import statements - np and pd are already provided in the context
    # Common patterns: "import numpy as np", "import pandas as pd", "from X import Y"
    formula = re.sub(r'^\s*import\s+numpy\s+as\s+np\s*;?\s*$', '', formula, flags=re.MULTILINE)
    formula = re.sub(r'^\s*import\s+pandas\s+as\s+pd\s*;?\s*$', '', formula, flags=re.MULTILINE)
    formula = re.sub(r'^\s*import\s+numpy\s*;?\s*$', '', formula, flags=re.MULTILINE)
    formula = re.sub(r'^\s*import\s+pandas\s*;?\s*$', '', formula, flags=re.MULTILINE)
    formula = re.sub(r'^\s*from\s+numpy\s+import\s+[^\n]+\s*;?\s*$', '', formula, flags=re.MULTILINE)
    formula = re.sub(r'^\s*from\s+pandas\s+import\s+[^\n]+\s*;?\s*$', '', formula, flags=re.MULTILINE)
    # Also handle inline imports like "import numpy as np;" in the middle of code
    formula = re.sub(r'\bimport\s+numpy\s+as\s+np\s*;', '', formula)
    formula = re.sub(r'\bimport\s+pandas\s+as\s+pd\s*;', '', formula)

    # Fix common AI mistakes: np.nan as default in np.select with string choices
    # np.nan is float and can't mix with strings; use None instead
    formula = re.sub(r'default\s*=\s*np\.nan\b', 'default=None', formula)

    # Fix common AI mistake: np.nan in np.where with string values causes dtype errors
    # Check if formula has both np.where with string literals and np.nan
    # FIX: Use improved _has_string_literals helper for edge cases
    if 'np.where' in formula and 'np.nan' in formula:
        if _has_string_literals(formula):
            formula = re.sub(r'\bnp\.nan\b', 'None', formula)
            logger.info("[MULTISTATEMENT] Auto-fixed np.nan -> None for string compatibility")

    # Fix common AI mistake: commenting out the critical assignment line
    # Pattern: "# df['ColumnName'] = ..." should be "df['ColumnName'] = ..."
    # Only uncomment if the output_column matches
    # FIX: Use improved _is_commented_assignment helper to avoid false positives
    if output_column:
        lines = formula.split('\n')
        fixed_lines = []
        for line in lines:
            if _is_commented_assignment(line, output_column):
                # Uncomment by removing the leading # (preserve indentation)
                stripped = line.lstrip()
                indent = line[:len(line) - len(stripped)]
                uncommented = stripped[1:].lstrip()  # Remove # and leading space
                fixed_lines.append(indent + uncommented)
                logger.warning(
                    f"[MULTISTATEMENT] Auto-uncommenting assignment to df['{output_column}']"
                )
            else:
                fixed_lines.append(line)
        formula = '\n'.join(fixed_lines)

    # Create a mutable copy of df that exec() can modify
    working_df = df.copy()
    exec_context = exec_context.copy()
    exec_context["df"] = working_df

    # Add column name shortcuts
    for col in working_df.columns:
        exec_context[col] = working_df[col]

    globals_dict = {"__builtins__": safe_builtins, **exec_context}

    # FIX: Check if the last statement is a bare expression that returns a result
    # If so, wrap it to capture the result. This handles formulas like:
    #   major_assignors = ["A", "B"]
    #   np.where(df["Col"].isin(major_assignors), df["Col"], "Other")
    # where the np.where result is not assigned to anything.
    import ast
    try:
        tree = ast.parse(formula.strip(), mode='exec')
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            # Last statement is a bare expression - wrap it to capture result
            # Split formula and wrap the last expression
            lines = formula.strip().split('\n')
            # Find where the last statement starts (could be multi-line)
            last_expr_source = ast.get_source_segment(formula.strip(), tree.body[-1])
            if last_expr_source:
                # Replace the last expression with an assignment
                formula = formula.strip()
                # Find the position of the last expression
                last_pos = formula.rfind(last_expr_source)
                if last_pos >= 0:
                    formula = formula[:last_pos] + "_auto_result_ = " + formula[last_pos:]
                    logger.info(f"[MULTISTATEMENT] Auto-captured bare expression result for '{output_column}'")
    except Exception as e:
        logger.debug(f"[MULTISTATEMENT] Could not check for bare expression: {e}")

    # Execute the script
    try:
        exec(formula, globals_dict, exec_context)
    except Exception as e:
        raise FeatureEngineeringError(f"Failed to execute multi-statement formula: {e}")

    # Try to find the result in order of preference:
    # 1. Check if output_column was created/modified in df
    result = None

    # The df in exec_context might have been modified
    modified_df = exec_context.get("df", working_df)

    if output_column and output_column in modified_df.columns:
        result = modified_df[output_column]
        logger.info(f"Extracted result from df['{output_column}']")
    # 2. Check for auto-captured bare expression result
    elif "_auto_result_" in exec_context:
        result = exec_context["_auto_result_"]
        logger.info("Extracted result from auto-captured bare expression")
    # 3. Check for a 'result' variable
    elif "result" in exec_context:
        result = exec_context["result"]
        logger.info("Extracted result from 'result' variable")
    # 4. Check for an 'out' variable (common pattern)
    elif "out" in exec_context:
        result = exec_context["out"]
        logger.info("Extracted result from 'out' variable")
    # 4. Check for output_column as a variable
    elif output_column and output_column in exec_context:
        result = exec_context[output_column]
        logger.info(f"Extracted result from '{output_column}' variable")
    else:
        # Look for any new columns added to df
        original_cols = set(df.columns)
        new_cols = set(modified_df.columns) - original_cols
        if new_cols:
            # Use the first new column
            new_col = list(new_cols)[0]
            result = modified_df[new_col]
            logger.info(f"Extracted result from new column '{new_col}'")
        else:
            # FIX: Try to find any array-like variable that was created
            # This handles cases where the AI assigns to a custom variable name
            # (e.g., 'assignment' instead of 'result' or 'out')
            candidate_vars = []
            for var_name, var_value in exec_context.items():
                # Skip known context variables
                if var_name in ('df', 'pd', 'np', 'sum', 'min', 'max', 'abs', 'len', '__builtins__'):
                    continue
                # Skip if it's a column reference from original df
                if var_name in original_cols:
                    continue
                # Check if it's an array-like result
                if isinstance(var_value, (pd.Series, np.ndarray)):
                    if len(var_value) == len(df):
                        candidate_vars.append((var_name, var_value))
                        logger.debug(f"Found candidate result variable: '{var_name}'")

            if candidate_vars:
                # Use the last added variable (most likely the final result)
                var_name, result = candidate_vars[-1]
                logger.info(f"Extracted result from variable '{var_name}' (auto-detected)")
            else:
                raise FeatureEngineeringError(
                    f"Multi-statement formula did not produce a result. "
                    f"Ensure the formula assigns to df['{output_column}'], 'result', or 'out' variable."
                )

    # Convert to Series if needed
    if isinstance(result, pd.DataFrame):
        if len(result.columns) == 1:
            result = result.iloc[:, 0]
        else:
            raise FeatureEngineeringError(
                "Multi-statement formula produced a DataFrame with multiple columns"
            )

    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=df.index)

    return result


def _try_alternative_formulas(
    df: pd.DataFrame,
    formula: str,
    exec_context: dict,
    safe_builtins: dict,
    original_error: Exception,
) -> pd.Series:
    """Try alternative formula syntax patterns.

    Handles common variations in how formulas might be written.
    """
    import numpy as np

    globals_dict = {"__builtins__": safe_builtins, **exec_context}

    # Pattern 1: Formula uses column names directly without df[]
    # e.g., "close.shift(-1) > close" instead of "df['close'].shift(-1) > df['close']"
    try:
        modified = formula
        for col in df.columns:
            # Replace bare column names with df["column_name"]
            # Be careful not to replace inside existing df[] references
            if col in modified and f'df["{col}"]' not in modified and f"df['{col}']" not in modified:
                modified = modified.replace(col, f'df["{col}"]')
        if modified != formula:
            return eval(modified, globals_dict, exec_context)
    except Exception:
        pass

    # Pattern 2: Formula uses assignment syntax
    # e.g., "target = close.shift(-1) > close"
    if "=" in formula and not any(op in formula for op in ["==", ">=", "<=", "!="]):
        try:
            # Extract the right side of the assignment
            right_side = formula.split("=", 1)[1].strip()
            return eval(right_side, globals_dict, exec_context)
        except Exception:
            pass

    # Pattern 3: Formula has sort_values creating index misalignment issues
    # e.g., "df.sort_values(['Name', 'date']).groupby('Name')['close'].shift(-1) > df.sort_values(['Name', 'date']).groupby('Name')['close']"
    # This pattern fails because the two sort_values create misaligned indices
    if "sort_values" in formula and "groupby" in formula:
        try:
            import re
            # Extract the groupby column and the comparison
            # Simplified approach: sort the df once, then do the comparison

            # Find groupby column(s)
            groupby_match = re.search(r"groupby\(['\"]?(\w+)['\"]?\)", formula)
            group_col = groupby_match.group(1) if groupby_match else None

            # Find the target column in groupby
            target_col_match = re.search(r"\[[\'\"](\w+)[\'\"]\]\.shift", formula)
            target_col = target_col_match.group(1) if target_col_match else None

            # Find shift value
            shift_match = re.search(r"shift\((-?\d+)\)", formula)
            shift_val = int(shift_match.group(1)) if shift_match else -1

            # Detect comparison operator
            if " > " in formula:
                comp_op = ">"
            elif " < " in formula:
                comp_op = "<"
            elif " >= " in formula:
                comp_op = ">="
            elif " <= " in formula:
                comp_op = "<="
            elif " == " in formula:
                comp_op = "=="
            else:
                comp_op = ">"  # default

            if group_col and target_col:
                logger.info(f"Attempting simplified groupby formula: group={group_col}, col={target_col}, shift={shift_val}")

                # Sort df once by the groupby column and any date column
                date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
                sort_cols = [group_col] + date_cols[:1] if date_cols else [group_col]
                sort_cols = [c for c in sort_cols if c in df.columns]

                if sort_cols:
                    sorted_df = df.sort_values(sort_cols)
                else:
                    sorted_df = df

                # Now do the comparison with aligned indices
                shifted = sorted_df.groupby(group_col)[target_col].shift(shift_val)
                original = sorted_df[target_col]

                if comp_op == ">":
                    result = shifted > original
                elif comp_op == "<":
                    result = shifted < original
                elif comp_op == ">=":
                    result = shifted >= original
                elif comp_op == "<=":
                    result = shifted <= original
                else:
                    result = shifted == original

                # Reindex to original df index
                result = result.reindex(df.index)
                logger.info(f"Simplified groupby formula succeeded")
                return result
        except Exception as e:
            logger.warning(f"Pattern 3 (sort_values/groupby) failed: {e}")
            pass

    # Pattern 4: Formula compares groupby results that return SeriesGroupBy instead of Series
    # e.g., "df.groupby('col')['x'].shift(-1) > df.groupby('col')['x']" (wrong - right side is GroupBy object)
    if "groupby" in formula and (".shift(" in formula or ".transform(" in formula):
        try:
            import re
            # Check if the right side of comparison is a SeriesGroupBy (missing aggregation)
            # Pattern: groupby(...)['col'] without .shift/.transform/.apply etc.
            groupby_without_agg = re.search(
                r">\s*df\..*?groupby\([^)]+\)\[['\"][^'\"]+['\"]\]\s*$", formula
            )
            if groupby_without_agg:
                # The right side needs to just be the column value, not a groupby
                # Replace "df.groupby(...)['col']" with "df['col']"
                fixed = re.sub(
                    r">\s*df\..*?groupby\([^)]+\)\[(['\"][^'\"]+['\"])\]\s*$",
                    r"> df[\1]",
                    formula
                )
                logger.info(f"Pattern 4: Fixed groupby comparison to: {fixed[:100]}...")
                result = eval(fixed, globals_dict, exec_context)
                return result
        except Exception as e:
            logger.warning(f"Pattern 4 (groupby comparison) failed: {e}")
            pass

    # Pattern 5: DTypePromotionError - numpy can't mix string dtype with float (np.nan)
    # This happens when np.where uses string literals with np.nan as fallback
    # Fix: Replace np.nan with None which is compatible with any dtype
    error_str = str(original_error)
    if "DType" in error_str or "dtype" in error_str.lower() or "StrDType" in error_str:
        if "np.nan" in formula:
            try:
                import re
                fixed = re.sub(r'\bnp\.nan\b', 'None', formula)
                logger.info(f"Pattern 5: Fixing DType error by replacing np.nan with None")
                result = eval(fixed, globals_dict, exec_context)
                return result
            except Exception as e:
                logger.warning(f"Pattern 5 (DType fix) failed: {e}")
                pass

    # If nothing worked, raise the original error
    raise original_error


def _convert_dtype(series: pd.Series, dtype: str) -> pd.Series:
    """Convert a Series to the specified data type.

    Args:
        series: Input Series
        dtype: Target data type name

    Returns:
        Converted Series
    """
    dtype_map = {
        "bool": "bool",
        "boolean": "bool",
        "int": "Int64",  # Nullable int
        "integer": "Int64",
        "int64": "Int64",
        "float": "float64",
        "float64": "float64",
        "str": "string",
        "string": "string",
        "category": "category",
        "datetime": "datetime64[ns]",
    }

    target_dtype = dtype_map.get(dtype.lower(), dtype)

    try:
        return series.astype(target_dtype)
    except Exception as e:
        logger.warning(f"Could not convert to {dtype}: {e}")
        return series


def get_common_feature_formulas() -> Dict[str, Dict[str, Any]]:
    """Return a dictionary of common feature engineering patterns.

    This can be used by agents to suggest feature engineering steps.
    """
    return {
        # Time-based features
        "day_of_week": {
            "formula": 'df["{date_col}"].dt.dayofweek',
            "description": "Extract day of week (0=Monday, 6=Sunday)",
            "source_type": "datetime",
        },
        "month": {
            "formula": 'df["{date_col}"].dt.month',
            "description": "Extract month (1-12)",
            "source_type": "datetime",
        },
        "year": {
            "formula": 'df["{date_col}"].dt.year',
            "description": "Extract year",
            "source_type": "datetime",
        },
        "hour": {
            "formula": 'df["{date_col}"].dt.hour',
            "description": "Extract hour (0-23)",
            "source_type": "datetime",
        },
        "is_weekend": {
            "formula": 'df["{date_col}"].dt.dayofweek >= 5',
            "description": "True if Saturday or Sunday",
            "source_type": "datetime",
        },
        # Numeric features
        "range": {
            "formula": 'df["{high_col}"] - df["{low_col}"]',
            "description": "Difference between high and low values",
            "source_type": "numeric",
        },
        "ratio": {
            "formula": 'df["{numerator}"] / df["{denominator}"]',
            "description": "Ratio of two columns",
            "source_type": "numeric",
        },
        "log": {
            "formula": 'np.log1p(df["{col}"])',
            "description": "Log transform (log1p for stability)",
            "source_type": "numeric",
        },
        "rolling_mean": {
            "formula": 'df["{col}"].rolling(window={window}).mean()',
            "description": "Rolling average over window",
            "source_type": "numeric",
        },
        "pct_change": {
            "formula": 'df["{col}"].pct_change()',
            "description": "Percentage change from previous row",
            "source_type": "numeric",
        },
        # Target creation patterns
        "future_increase": {
            "formula": 'df["{col}"].shift(-1) > df["{col}"]',
            "description": "True if next value is higher (binary classification)",
            "source_type": "numeric",
            "for_target": True,
        },
        "future_value": {
            "formula": 'df["{col}"].shift(-{periods})',
            "description": "Future value shifted back (for prediction)",
            "source_type": "numeric",
            "for_target": True,
        },
        "future_change": {
            "formula": 'df["{col}"].shift(-1) - df["{col}"]',
            "description": "Change to next value (regression target)",
            "source_type": "numeric",
            "for_target": True,
        },
    }

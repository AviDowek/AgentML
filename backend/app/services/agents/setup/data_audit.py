"""Data Audit Agent - Comprehensive data quality audit.

This agent performs a COMPREHENSIVE analysis of data quality and characteristics,
checking for issues that could impact model training and validity.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from app.models import DataSource
from app.models import AgentStepType
from app.schemas.agent import SchemaSummary
from app.services.agents.base import BaseAgent


class DataAuditAgent(BaseAgent):
    """Performs comprehensive data quality audit.

    Input JSON:
        - schema_summary: Schema summary from previous step
        - target_column: The target column (optional)
        - data_source_id: UUID for loading actual data
        - project_id: Project ID
        - time_column: Time column if time-based
        - is_time_based: Whether task is time-based

    Output:
        - data_source_name: Name of data source
        - row_count: Number of rows
        - column_count: Number of columns
        - critical_issues: Must-fix issues
        - warnings: Should-investigate issues
        - info_notes: Good-to-know notes
        - recommendations: Specific recommendations
        - audit_details: Detailed findings
        - target_info: Target column analysis
        - target_stats: Target statistics
        - leakage_candidates: Enhanced leakage detection results
        - issues: Combined issues (legacy)
        - high_null_columns: Columns with >50% nulls
        - potential_id_columns: Likely ID columns
        - constant_columns: Single-value columns
    """

    name = "data_audit"
    step_type = AgentStepType.DATA_AUDIT

    async def execute(self) -> Dict[str, Any]:
        """Execute comprehensive data audit."""
        schema_data = self.require_input("schema_summary")
        schema_summary = SchemaSummary(**schema_data)
        target_column = self.get_input("target_column")

        self.logger.info(f"🔍 Starting comprehensive data audit for: {schema_summary.data_source_name}")
        self.logger.info(f"   Dataset: {schema_summary.row_count:,} rows × {schema_summary.column_count} columns")

        # Initialize tracking structures
        critical_issues: List[str] = []
        warnings: List[str] = []
        info_notes: List[str] = []
        recommendations: List[str] = []

        audit_details = {
            "high_null_columns": [],
            "moderate_null_columns": [],
            "potential_id_columns": [],
            "constant_columns": [],
            "low_variance_columns": [],
            "high_cardinality_columns": [],
            "potential_leakage_columns": [],
            "class_imbalance": None,
            "small_dataset_warning": False,
            "duplicate_risk": False,
        }

        # Run all audit checks
        self._check_dataset_size(schema_summary, critical_issues, warnings, info_notes, recommendations, audit_details)
        self._check_null_values(schema_summary, critical_issues, warnings, info_notes, recommendations, audit_details)
        self._check_id_columns(schema_summary, target_column, warnings, recommendations, audit_details)
        self._check_variance(schema_summary, warnings, info_notes, recommendations, audit_details)
        self._check_cardinality(schema_summary, warnings, recommendations, audit_details)
        self._check_leakage_patterns(schema_summary, target_column, critical_issues, info_notes, audit_details)

        # Enhanced leakage detection
        leakage_candidates = await self._run_enhanced_leakage_detection(
            schema_summary, target_column, critical_issues, warnings, recommendations, audit_details
        )

        # Target column analysis
        target_info, target_stats = self._analyze_target(
            schema_summary, target_column, critical_issues, warnings, recommendations, audit_details
        )

        # Duplicate risk assessment
        self._assess_duplicate_risk(schema_summary, info_notes, audit_details)

        # Generate summary
        self._log_summary(critical_issues, warnings, info_notes, recommendations)

        return {
            "data_source_name": schema_summary.data_source_name,
            "row_count": schema_summary.row_count,
            "column_count": schema_summary.column_count,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "info_notes": info_notes,
            "recommendations": recommendations,
            "audit_details": audit_details,
            "target_info": target_info,
            "target_stats": target_stats,
            "leakage_candidates": leakage_candidates,
            # Legacy fields
            "issues": critical_issues + warnings,
            "high_null_columns": audit_details["high_null_columns"],
            "potential_id_columns": audit_details["potential_id_columns"],
            "constant_columns": audit_details["constant_columns"],
        }

    def _check_dataset_size(
        self, schema: SchemaSummary, critical: List, warnings: List, info: List, recs: List, details: Dict
    ):
        """Check dataset size adequacy."""
        self.logger.thinking("Checking dataset size adequacy...")

        if schema.row_count < 100:
            critical.append(f"⚠️ CRITICAL: Very small dataset ({schema.row_count} rows) - insufficient for reliable ML training")
            recs.append("Collect more data or use simpler models with strong regularization")
            details["small_dataset_warning"] = True
        elif schema.row_count < 500:
            warnings.append(f"Small dataset ({schema.row_count} rows) - results may have high variance")
            recs.append("Consider using cross-validation and simpler models")
            details["small_dataset_warning"] = True
        elif schema.row_count < 1000:
            info.append(f"Moderate dataset size ({schema.row_count} rows) - watch for overfitting")

        icon = '⚠️' if details['small_dataset_warning'] else '✓'
        self.logger.info(f"   Dataset size: {schema.row_count:,} rows {icon}")

    def _check_null_values(
        self, schema: SchemaSummary, critical: List, warnings: List, info: List, recs: List, details: Dict
    ):
        """Analyze missing values across all columns."""
        self.logger.thinking("Analyzing missing values across all columns...")

        for col in schema.columns:
            if col.null_percentage > 50:
                details["high_null_columns"].append(col.name)
                critical.append(f"Column '{col.name}' has {col.null_percentage:.1f}% missing values - consider dropping")
            elif col.null_percentage > 20:
                details["moderate_null_columns"].append(col.name)
                warnings.append(f"Column '{col.name}' has {col.null_percentage:.1f}% missing values")
            elif col.null_percentage > 5:
                info.append(f"Column '{col.name}' has {col.null_percentage:.1f}% missing values")

        if details["high_null_columns"]:
            self.logger.warning(f"   Found {len(details['high_null_columns'])} columns with >50% nulls: {details['high_null_columns']}")
            recs.append("Drop columns with >50% missing values or use advanced imputation")
        if details["moderate_null_columns"]:
            self.logger.warning(f"   Found {len(details['moderate_null_columns'])} columns with 20-50% nulls")
            recs.append("Consider imputation strategies (mean/median/mode or ML-based)")

        total_null = len(details["high_null_columns"]) + len(details["moderate_null_columns"])
        icon = '⚠️' if total_null > 0 else '✓'
        self.logger.info(f"   Missing value analysis: {total_null} columns with significant nulls {icon}")

    def _check_id_columns(
        self, schema: SchemaSummary, target: Optional[str], warnings: List, recs: List, details: Dict
    ):
        """Detect potential ID columns."""
        self.logger.thinking("Detecting potential ID columns...")

        for col in schema.columns:
            if col.name == target:
                continue

            is_all_unique = col.unique_count == schema.row_count
            name_suggests_id = any(
                pattern in col.name.lower()
                for pattern in ['_id', 'id_', 'uuid', 'guid', 'key', 'index', 'row_num']
            )

            if is_all_unique or (name_suggests_id and col.unique_count > schema.row_count * 0.9):
                details["potential_id_columns"].append(col.name)
                warnings.append(f"Column '{col.name}' appears to be an ID column (should be excluded from features)")

        if details["potential_id_columns"]:
            self.logger.warning(f"   Potential ID columns (exclude from features): {details['potential_id_columns']}")
            recs.append("Remove ID columns from feature set - they cause overfitting")

        icon = '⚠️' if details['potential_id_columns'] else '✓'
        self.logger.info(f"   ID column detection: {len(details['potential_id_columns'])} found {icon}")

    def _check_variance(
        self, schema: SchemaSummary, warnings: List, info: List, recs: List, details: Dict
    ):
        """Check for constant and low-variance columns."""
        self.logger.thinking("Checking for constant and low-variance columns...")

        for col in schema.columns:
            if col.unique_count == 1:
                details["constant_columns"].append(col.name)
                warnings.append(f"Column '{col.name}' is constant (single value) - provides no information")
            elif col.unique_count == 2 and col.null_percentage > 90:
                details["low_variance_columns"].append(col.name)
                warnings.append(f"Column '{col.name}' has near-zero variance")
            elif schema.row_count > 100 and col.unique_count < 3:
                details["low_variance_columns"].append(col.name)
                info.append(f"Column '{col.name}' has very low variance ({col.unique_count} unique values)")

        if details["constant_columns"]:
            self.logger.warning(f"   Constant columns (remove): {details['constant_columns']}")
            recs.append("Remove constant columns - they provide no predictive value")

        icon = '⚠️' if details['constant_columns'] else '✓'
        self.logger.info(
            f"   Variance check: {len(details['constant_columns'])} constant, "
            f"{len(details['low_variance_columns'])} low-variance {icon}"
        )

    def _check_cardinality(
        self, schema: SchemaSummary, warnings: List, recs: List, details: Dict
    ):
        """Check categorical column cardinality."""
        self.logger.thinking("Checking categorical column cardinality...")

        for col in schema.columns:
            is_categorical = col.inferred_type == "categorical"
            is_text_like = col.inferred_type == "text" and col.unique_count < schema.row_count * 0.5

            if is_categorical or is_text_like:
                cardinality_ratio = col.unique_count / schema.row_count if schema.row_count > 0 else 0

                if col.unique_count > 100 and cardinality_ratio > 0.1:
                    details["high_cardinality_columns"].append({
                        "name": col.name,
                        "unique_count": col.unique_count,
                        "cardinality_ratio": cardinality_ratio
                    })
                    warnings.append(
                        f"Column '{col.name}' has high cardinality ({col.unique_count} categories) - may cause issues"
                    )

        if details["high_cardinality_columns"]:
            self.logger.warning(
                f"   High cardinality columns: {[c['name'] for c in details['high_cardinality_columns']]}"
            )
            recs.append("Consider encoding strategies for high-cardinality categoricals (target encoding, frequency encoding)")

        icon = '⚠️' if details['high_cardinality_columns'] else '✓'
        self.logger.info(f"   Cardinality check: {len(details['high_cardinality_columns'])} high-cardinality columns {icon}")

    def _check_leakage_patterns(
        self, schema: SchemaSummary, target: Optional[str], critical: List, info: List, details: Dict
    ):
        """Scan for potential data leakage indicators."""
        self.logger.thinking("Scanning for potential data leakage indicators...")

        leakage_patterns = [
            'target', 'label', 'outcome', 'result', 'prediction', 'pred_',
            'future_', 'next_', 'will_', 'actual_', 'true_', 'y_'
        ]

        for col in schema.columns:
            if col.name == target:
                continue

            col_lower = col.name.lower()

            for pattern in leakage_patterns:
                if pattern in col_lower:
                    details["potential_leakage_columns"].append({
                        "name": col.name,
                        "reason": f"Name contains '{pattern}' - may leak target information"
                    })
                    critical.append(f"⚠️ POTENTIAL LEAKAGE: Column '{col.name}' may contain target information")
                    break

            # Check for similar range to target (numeric columns)
            if col.inferred_type == "numeric" and target:
                target_col = next((c for c in schema.columns if c.name == target), None)
                if target_col and target_col.inferred_type == "numeric":
                    if (col.min is not None and col.max is not None and
                        target_col.min is not None and target_col.max is not None):
                        col_range = col.max - col.min if col.max != col.min else 1
                        target_range = target_col.max - target_col.min if target_col.max != target_col.min else 1
                        if 0.9 < col_range / target_range < 1.1 and col.name not in details["potential_id_columns"]:
                            info.append(
                                f"Column '{col.name}' has similar range to target - verify it's not derived from target"
                            )

        if details["potential_leakage_columns"]:
            self.logger.error(
                f"   ⚠️ POTENTIAL DATA LEAKAGE detected in: "
                f"{[c['name'] for c in details['potential_leakage_columns']]}"
            )

        icon = '🚨' if details['potential_leakage_columns'] else '✓'
        self.logger.info(f"   Leakage scan: {len(details['potential_leakage_columns'])} suspicious columns {icon}")

    async def _run_enhanced_leakage_detection(
        self, schema: SchemaSummary, target: Optional[str],
        critical: List, warnings: List, recs: List, details: Dict
    ) -> List[Dict[str, Any]]:
        """Run enhanced leakage detection with correlation analysis."""
        leakage_candidates: List[Dict[str, Any]] = []

        try:
            from app.services.leakage_detector import (
                detect_potential_leakage_features,
                get_leakage_summary,
            )

            time_column = self.get_input("time_column")
            is_time_based = self.get_input("is_time_based", False)
            data_source_id = self.get_input("data_source_id")

            df_for_leakage: Optional[pd.DataFrame] = None

            if data_source_id:
                self.logger.thinking("Loading data for enhanced leakage detection...")
                try:
                    data_source = self.db.query(DataSource).filter(DataSource.id == data_source_id).first()
                    if data_source and data_source.file_path:
                        import os
                        if os.path.exists(data_source.file_path):
                            df_for_leakage = pd.read_csv(data_source.file_path, nrows=10000)
                except Exception as e:
                    self.logger.info(f"   Could not load data for correlation analysis: {e}")

            if df_for_leakage is None:
                column_names = [col.name for col in schema.columns]
                df_for_leakage = pd.DataFrame(columns=column_names)

            if target and df_for_leakage is not None:
                self.logger.thinking("Running enhanced leakage detection heuristics...")

                leakage_candidates = detect_potential_leakage_features(
                    df=df_for_leakage,
                    target_column=target,
                    time_column=time_column,
                    correlation_threshold=0.9,
                )

                if leakage_candidates:
                    leakage_summary = get_leakage_summary(leakage_candidates)
                    self.logger.warning(
                        f"   🔍 Enhanced leakage detection found {leakage_summary['total_count']} suspicious features:"
                    )

                    for candidate in leakage_candidates:
                        severity_emoji = "🚨" if candidate["severity"] == "high" else "⚠️" if candidate["severity"] == "medium" else "ℹ️"
                        self.logger.warning(f"      {severity_emoji} {candidate['column']}: {candidate['reason']}")

                        if candidate["severity"] == "high":
                            critical.append(
                                f"⚠️ LEAKAGE CANDIDATE: '{candidate['column']}' - {candidate['reason']}"
                            )
                        elif candidate["severity"] == "medium":
                            warnings.append(
                                f"Potential leakage: '{candidate['column']}' - {candidate['reason']}"
                            )

                    if leakage_summary["high_severity_count"] > 0:
                        recs.append(
                            f"CRITICAL: Review {leakage_summary['high_severity_count']} high-severity leakage candidates before training"
                        )

                    if is_time_based and leakage_summary["total_count"] > 0:
                        recs.append(
                            "For time-based predictions, verify all features are computed from past data only (no look-ahead bias)"
                        )
                else:
                    self.logger.info("   ✓ Enhanced leakage detection: No suspicious features found")

            details["leakage_candidates"] = leakage_candidates

        except ImportError:
            self.logger.info("   Enhanced leakage detection not available")
        except Exception as e:
            self.logger.warning(f"   Enhanced leakage detection failed: {e}")

        return leakage_candidates

    def _analyze_target(
        self, schema: SchemaSummary, target: Optional[str],
        critical: List, warnings: List, recs: List, details: Dict
    ) -> tuple:
        """Analyze target column."""
        target_info = None
        target_stats = None

        if not target:
            return target_info, target_stats

        self.logger.thinking(f"Analyzing target column '{target}'...")
        target_col = next((c for c in schema.columns if c.name == target), None)

        if not target_col:
            critical.append(f"Target column '{target}' not found in dataset!")
            self.logger.error(f"   ❌ Target column '{target}' not found!")
            return target_info, target_stats

        target_info = {
            "name": target_col.name,
            "unique_count": target_col.unique_count,
            "null_percentage": target_col.null_percentage,
            "inferred_type": target_col.inferred_type,
        }

        if target_col.null_percentage > 0:
            critical.append(
                f"Target column has {target_col.null_percentage:.1f}% null values - must handle before training"
            )
            recs.append("Remove rows with null target or investigate data collection issues")

        self.logger.info(
            f"   Target '{target}': {target_col.unique_count} unique values, "
            f"{target_col.null_percentage:.1f}% nulls"
        )

        # Classification target
        if target_col.inferred_type == "categorical" or (target_col.unique_count and target_col.unique_count <= 20):
            target_stats = {"class_counts": target_col.top_values or {}}

            if target_col.top_values and len(target_col.top_values) >= 2:
                total = sum(target_col.top_values.values())
                class_counts = sorted(target_col.top_values.values(), reverse=True)
                majority_count = class_counts[0]
                minority_count = class_counts[-1]
                majority_pct = majority_count / total * 100 if total > 0 else 0
                imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')

                target_stats["majority_class_pct"] = majority_pct
                target_stats["imbalance_ratio"] = imbalance_ratio
                target_stats["num_classes"] = len(target_col.top_values)

                self.logger.info(
                    f"   Target distribution: {len(target_col.top_values)} classes, majority class: {majority_pct:.1f}%"
                )

                if imbalance_ratio > 10:
                    details["class_imbalance"] = {"ratio": imbalance_ratio, "severity": "severe"}
                    critical.append(
                        f"⚠️ SEVERE CLASS IMBALANCE: {imbalance_ratio:.1f}:1 ratio - standard metrics will be misleading"
                    )
                    recs.append("Use class weights, SMOTE, or evaluation metrics like F1/AUC instead of accuracy")
                    self.logger.error(f"   🚨 Severe class imbalance detected: {imbalance_ratio:.1f}:1 ratio")
                elif imbalance_ratio > 3:
                    details["class_imbalance"] = {"ratio": imbalance_ratio, "severity": "moderate"}
                    warnings.append(f"Moderate class imbalance ({imbalance_ratio:.1f}:1 ratio) - consider balanced metrics")
                    recs.append("Consider using balanced accuracy or F1 score as primary metric")
                    self.logger.warning(f"   ⚠️ Moderate class imbalance: {imbalance_ratio:.1f}:1 ratio")
                else:
                    self.logger.info(f"   Class balance: Good ({imbalance_ratio:.1f}:1 ratio) ✓")

        # Regression target
        elif target_col.inferred_type == "numeric":
            target_stats = {
                "min": target_col.min,
                "max": target_col.max,
                "mean": target_col.mean,
                "std": None,
            }

            if target_col.min is not None and target_col.max is not None:
                col_range = target_col.max - target_col.min
                target_stats["std"] = col_range / 4.0  # Rough estimate
                target_stats["baseline_rmse"] = target_stats["std"]

                self.logger.info(f"   Target range: {target_col.min:.4f} to {target_col.max:.4f}")
                self.logger.info(f"   Baseline RMSE (predicting mean): ~{target_stats['std']:.4f}")

                if target_col.mean is not None:
                    mean_to_max = abs(target_col.max - target_col.mean)
                    mean_to_min = abs(target_col.mean - target_col.min)
                    if mean_to_max > 5 * mean_to_min or mean_to_min > 5 * mean_to_max:
                        warnings.append(
                            "Target column may have extreme outliers - consider robust scaling or outlier removal"
                        )
                        self.logger.warning(f"   ⚠️ Target may have outliers (asymmetric distribution)")

        return target_info, target_stats

    def _assess_duplicate_risk(self, schema: SchemaSummary, info: List, details: Dict):
        """Assess duplicate row risk."""
        self.logger.thinking("Assessing duplicate row risk...")

        non_id_unique_cols = [
            col for col in schema.columns
            if col.name not in details["potential_id_columns"]
            and col.unique_count == schema.row_count
        ]

        if len(non_id_unique_cols) == 0 and schema.row_count > 100:
            info.append("No unique identifier column found - verify dataset has no duplicate rows")
            details["duplicate_risk"] = True
            self.logger.info(f"   Duplicate risk: Possible (no unique column) - verify manually")
        else:
            self.logger.info(f"   Duplicate risk: Low ✓")

    def _log_summary(self, critical: List, warnings: List, info: List, recs: List):
        """Log audit summary."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("📋 DATA AUDIT SUMMARY")
        self.logger.info("=" * 60)

        if critical:
            self.logger.error(f"🚨 CRITICAL ISSUES ({len(critical)}):")
            for issue in critical:
                self.logger.error(f"   • {issue}")

        if warnings:
            self.logger.warning(f"⚠️ WARNINGS ({len(warnings)}):")
            for warn in warnings[:10]:
                self.logger.warning(f"   • {warn}")
            if len(warnings) > 10:
                self.logger.warning(f"   ... and {len(warnings) - 10} more warnings")

        if info:
            self.logger.info(f"ℹ️ NOTES ({len(info)}):")
            for note in info[:5]:
                self.logger.info(f"   • {note}")

        if recs:
            self.logger.info(f"💡 RECOMMENDATIONS ({len(recs)}):")
            for rec in recs[:8]:
                self.logger.info(f"   • {rec}")

        if critical:
            self.logger.error(f"\n🔴 AUDIT RESULT: {len(critical)} critical issues require attention before training")
        elif warnings:
            self.logger.warning(f"\n🟡 AUDIT RESULT: {len(warnings)} warnings - proceed with caution")
        else:
            self.logger.info(f"\n🟢 AUDIT RESULT: Data looks good for training!")

        self.logger.info("=" * 60)

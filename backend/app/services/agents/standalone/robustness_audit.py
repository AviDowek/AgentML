"""Robustness Audit Agent - Audits models for overfitting and data leakage.

This agent analyzes experiments for overfitting, suspicious patterns,
data leakage, and compares against baselines. It produces a structured report.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models import DatasetSpec, Experiment, Project
from app.models import AgentRun, AgentStep, AgentStepType
from app.models.research_cycle import CycleExperiment
from app.services.agents.base import BaseAgent
from app.services.prompts import (
    SYSTEM_ROLE_ROBUSTNESS_AUDITOR,
    get_robustness_audit_prompt,
)
from app.services.task_context import (
    build_task_context,
    format_context_for_prompt,
    get_task_type_hints,
)


class RobustnessAuditResponse(BaseModel):
    """Response schema for robustness audit."""
    overfitting_risk: str = Field(..., description="Risk level: low, medium, or high")
    train_val_analysis: Dict[str, Any] = Field(default_factory=dict)
    suspicious_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    baseline_comparison: Dict[str, Any] = Field(default_factory=dict)
    cv_analysis: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    natural_language_summary: str = Field(default="")


class RobustnessAuditAgent(BaseAgent):
    """Audits experiments for overfitting and data leakage.

    Input JSON:
        - project_id: UUID of the project
        - experiment_id: Optional single experiment ID to audit
        - experiment_ids: List of experiment IDs to audit (or all if not provided)
        - research_cycle_id: Optional UUID of the research cycle
        - task_type: Type of ML task (binary, multiclass, regression)
        - is_time_based: Whether this is a time-based prediction task
        - primary_metric: The metric to focus on (e.g., "roc_auc", "rmse")

    Output:
        - robustness_audit: Full audit results including:
            - overfitting_risk: "low" | "medium" | "high"
            - leakage_suspected: bool
            - time_split_suspicious: bool
            - metrics_summary: {...}
            - warnings: [...]
            - recommendations: [...]
            - natural_language_summary: str
    """

    name = "robustness_audit"
    step_type = AgentStepType.ROBUSTNESS_AUDIT

    async def execute(self) -> Dict[str, Any]:
        """Execute robustness audit."""
        project_id = self.get_input("project_id")
        research_cycle_id = self.get_input("research_cycle_id")
        experiment_ids = self.get_input("experiment_ids", [])
        single_experiment_id = self.get_input("experiment_id")
        primary_metric = self.get_input("primary_metric")
        task_type = self.get_input("task_type")
        is_time_based_input = self.get_input("is_time_based", False)

        self.logger.info("Starting robustness and overfitting audit...")

        # Support single experiment_id as well
        if single_experiment_id and single_experiment_id not in experiment_ids:
            experiment_ids.append(single_experiment_id)

        # Load project
        if not project_id:
            if self.step.agent_run and self.step.agent_run.project_id:
                project_id = str(self.step.agent_run.project_id)
            else:
                raise ValueError("Missing 'project_id' in step input")

        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        if not task_type:
            task_type = project.task_type or "unknown"

        self.logger.info(f"Project: {project.name}")
        self.logger.info(f"Task type: {task_type}")

        # Build unified TaskContext
        task_context, task_hints, context_factors = self._build_task_context(
            project_id, research_cycle_id
        )

        # Determine if time-based from input or TaskContext
        is_time_based = is_time_based_input or task_hints.get("is_time_based", False)
        is_classification = task_type in ("binary", "multiclass", "classification")

        # Get experiments to audit
        experiments = self._get_experiments(
            experiment_ids, research_cycle_id, project_id
        )

        if not experiments:
            self.logger.warning("No experiments found to audit")
            return {
                "robustness_audit": {
                    "overfitting_risk": "unknown",
                    "train_val_analysis": {},
                    "suspicious_patterns": [],
                    "baseline_comparison": {},
                    "cv_analysis": {},
                    "recommendations": ["Run experiments first before auditing"],
                    "natural_language_summary": "No experiments available for audit.",
                }
            }

        self.logger.info(f"Auditing {len(experiments)} experiment(s)")

        # Determine primary metric if not provided
        if not primary_metric:
            for exp in experiments:
                if exp.primary_metric:
                    primary_metric = exp.primary_metric
                    break
            if not primary_metric:
                primary_metric = self._infer_metric_from_task(task_type)

        self.logger.info(f"Primary metric: {primary_metric}")

        # Analyze trials and collect metrics
        (
            trials_data,
            all_train_metrics,
            all_val_metrics,
            all_gaps,
            cv_fold_data,
            best_val_metric,
            all_baseline_metrics,
            leakage_suspected,
            leakage_warnings,
            time_split_suspicious,
            time_split_warnings,
        ) = self._analyze_trials(
            experiments, primary_metric, is_time_based, task_hints
        )

        # Build baseline information
        baseline_info, baseline_value, baseline_type = self._build_baseline_info(
            all_baseline_metrics, is_classification, primary_metric
        )

        # Build CV data summary
        cv_data = self._build_cv_summary(cv_fold_data)

        # Log summary statistics
        self._log_summary_statistics(
            all_gaps, best_val_metric, baseline_value
        )

        # Generate LLM prompt
        prompt = get_robustness_audit_prompt(
            project_name=project.name,
            problem_description=project.description or "No description provided",
            task_type=project.task_type or "unknown",
            primary_metric=primary_metric,
            trials_data=trials_data,
            baseline_info=baseline_info,
            cv_data=cv_data,
        )

        # Append unified TaskContext
        if task_context:
            task_context_str = format_context_for_prompt(
                task_context,
                include_sections=["baselines", "robustness", "leakage_candidates"],
                max_length=2000,
            )
            if task_context_str:
                prompt += "\n\n## ADDITIONAL PROJECT CONTEXT\n"
                prompt += task_context_str
                if task_hints.get("leakage_warnings"):
                    prompt += "\n\n**LEAKAGE WARNINGS TO CONSIDER**:\n"
                    for warning in task_hints["leakage_warnings"][:3]:
                        prompt += f"- {warning}\n"

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_ROBUSTNESS_AUDITOR},
            {"role": "user", "content": prompt},
        ]

        self.logger.info("Generating robustness audit with LLM...")
        response = await self.llm.chat_json(messages, RobustnessAuditResponse)

        # Extract results
        overfitting_risk = response.get("overfitting_risk", "unknown")
        suspicious_patterns = response.get("suspicious_patterns", [])
        recommendations = response.get("recommendations", [])
        summary = response.get("natural_language_summary", "")

        # Log the results
        risk_emoji = {"low": "OK", "medium": "WARNING", "high": "ALERT"}.get(
            overfitting_risk, "?"
        )
        self.logger.summary(
            f"Robustness Audit Complete - Risk: {risk_emoji} {overfitting_risk.upper()}"
        )

        if suspicious_patterns:
            self.logger.info(f"Found {len(suspicious_patterns)} suspicious pattern(s)")
            for pattern in suspicious_patterns:
                self.logger.warning(
                    f"[{pattern.get('type', 'unknown')}] {pattern.get('description', '')}"
                )

        if recommendations:
            self.logger.info(f"Generated {len(recommendations)} recommendation(s)")

        # Compile all warnings
        all_warnings = []
        if leakage_suspected:
            self.logger.warning("DATA LEAKAGE SUSPECTED - Label-shuffle test detected issues")
            all_warnings.extend(leakage_warnings)
        if time_split_suspicious:
            self.logger.warning("TIME-SPLIT SUSPICIOUS - Time-based data using inappropriate split")
            all_warnings.extend(time_split_warnings)

        # Build metrics summary
        metrics_summary = {
            "best_val_metric": best_val_metric,
            "primary_metric": primary_metric,
            "train_val_gap_worst": max(all_gaps) if all_gaps else None,
            "train_val_gap_avg": sum(all_gaps) / len(all_gaps) if all_gaps else None,
            "cv_variance": max(cv_fold_data) - min(cv_fold_data) if len(cv_fold_data) > 1 else None,
            "baseline_value": baseline_value,
            "baseline_type": baseline_type,
        }

        # Build baseline metrics summary for response
        baseline_metrics_summary = self._build_baseline_metrics_summary(all_baseline_metrics)

        # Check for "too good to be true" pattern
        (
            overfitting_risk,
            suspicious_patterns,
            all_warnings,
        ) = self._check_too_good_to_be_true(
            experiments,
            best_val_metric,
            primary_metric,
            is_time_based,
            task_type,
            overfitting_risk,
            suspicious_patterns,
            all_warnings,
            context_factors,
            task_context,
        )

        # Check if leakage candidates are among top important features
        (
            leakage_suspected,
            leakage_in_important_features,
            concerning_leakage_features,
            overfitting_risk,
            suspicious_patterns,
            all_warnings,
        ) = self._check_leakage_in_features(
            experiments,
            overfitting_risk,
            suspicious_patterns,
            all_warnings,
            leakage_suspected,
        )

        # Compute risk-adjusted score
        risk_adjusted_score = self._compute_risk_score(
            experiments,
            best_val_metric,
            overfitting_risk,
            leakage_suspected,
            time_split_suspicious,
        )

        # Get model risk status for promotion gating
        from app.services.risk_scoring import get_model_risk_status

        too_good_to_be_true = any(
            p.get("type") == "too_good_to_be_true" for p in suspicious_patterns
        )
        risk_level, requires_override, risk_reason = get_model_risk_status(
            overfitting_risk=overfitting_risk,
            leakage_suspected=leakage_suspected,
            time_split_suspicious=time_split_suspicious,
            too_good_to_be_true=too_good_to_be_true,
        )

        # Build context reasoning string
        context_reasoning = self._build_context_reasoning(
            baseline_value, baseline_type, best_val_metric, primary_metric,
            context_factors, task_context
        )

        return {
            "robustness_audit": {
                "overfitting_risk": overfitting_risk,
                "leakage_suspected": leakage_suspected,
                "time_split_suspicious": time_split_suspicious,
                "metrics_summary": metrics_summary,
                "warnings": all_warnings,
                "recommendations": recommendations,
                "natural_language_summary": summary,
                "too_good_to_be_true": too_good_to_be_true,
                "risk_adjusted_score": risk_adjusted_score,
                "risk_level": risk_level,
                "requires_override": requires_override,
                "risk_reason": risk_reason,
                "train_val_analysis": response.get("train_val_analysis", {}),
                "suspicious_patterns": suspicious_patterns,
                "baseline_comparison": response.get("baseline_comparison", {}),
                "baseline_metrics": baseline_metrics_summary,
                "cv_analysis": response.get("cv_analysis", {}),
                "is_time_based": is_time_based,
                "task_type": task_type,
                "leakage_in_important_features": leakage_in_important_features if 'leakage_in_important_features' in dir() else False,
                "concerning_leakage_features": concerning_leakage_features if 'concerning_leakage_features' in dir() else [],
                "leakage_candidates_count": len(self.get_input("leakage_candidates", [])),
                "context_reasoning": context_reasoning,
                "context_factors_used": context_factors,
            }
        }

    def _build_task_context(
        self, project_id: str, research_cycle_id: Optional[str]
    ) -> tuple:
        """Build unified TaskContext for the robustness auditor."""
        task_context = None
        task_hints = {}
        context_factors = {}

        try:
            task_context = build_task_context(
                db=self.db,
                project_id=str(project_id),
                research_cycle_id=str(research_cycle_id) if research_cycle_id else None,
                include_leakage_candidates=True,
                include_past_cycles=True,
                max_experiments=10,
            )
            task_hints = get_task_type_hints(task_context)
            self.logger.info("Built unified TaskContext for robustness audit")

            # Log context usage
            if task_context:
                if task_hints.get("is_time_based"):
                    self.logger.thought(f"Time-based task detected")
                if task_hints.get("leakage_warnings"):
                    self.logger.thought(f"Found {len(task_hints['leakage_warnings'])} leakage warnings")

        except Exception as e:
            self.logger.warning(f"Could not build TaskContext: {e}")

        return task_context, task_hints, context_factors

    def _get_experiments(
        self,
        experiment_ids: List[str],
        research_cycle_id: Optional[str],
        project_id: str,
    ) -> List[Experiment]:
        """Get experiments to audit."""
        if experiment_ids:
            return (
                self.db.query(Experiment)
                .filter(Experiment.id.in_(experiment_ids))
                .all()
            )
        elif research_cycle_id:
            cycle_experiments = (
                self.db.query(CycleExperiment)
                .filter(CycleExperiment.research_cycle_id == research_cycle_id)
                .all()
            )
            exp_ids = [ce.experiment_id for ce in cycle_experiments]
            return (
                self.db.query(Experiment)
                .filter(Experiment.id.in_(exp_ids))
                .all()
            ) if exp_ids else []
        else:
            return (
                self.db.query(Experiment)
                .filter(Experiment.project_id == project_id)
                .all()
            )

    def _infer_metric_from_task(self, task_type: str) -> str:
        """Infer the primary metric from task type."""
        if task_type in ("binary", "classification"):
            return "roc_auc"
        elif task_type == "multiclass":
            return "accuracy"
        elif task_type == "regression":
            return "rmse"
        return "accuracy"

    def _analyze_trials(
        self,
        experiments: List[Experiment],
        primary_metric: str,
        is_time_based: bool,
        task_hints: Dict,
    ) -> tuple:
        """Analyze trial data and collect metrics."""
        trials_data_lines = []
        all_train_metrics = []
        all_val_metrics = []
        all_gaps = []
        cv_fold_data = []
        best_val_metric = None
        all_baseline_metrics = []
        leakage_suspected = False
        leakage_warnings = []
        time_split_suspicious = False
        time_split_warnings = []

        for exp in experiments:
            self.logger.thought(f"Analyzing experiment: {exp.name}")

            if not exp.trials:
                trials_data_lines.append(f"**{exp.name}**: No trials found")
                continue

            for trial in exp.trials:
                trial_line = f"**{exp.name} - {trial.variant_name}** (Status: {trial.status.value if hasattr(trial.status, 'value') else trial.status})"

                metrics = trial.metrics_json or {}
                metric_lower = primary_metric.lower()

                # Look for training and validation metrics
                train_metric = None
                val_metric = None

                for key, value in metrics.items():
                    key_lower = key.lower()
                    if metric_lower in key_lower:
                        if 'train' in key_lower:
                            train_metric = value
                        elif 'val' in key_lower or 'test' in key_lower:
                            val_metric = value
                        elif train_metric is None and val_metric is None:
                            val_metric = value

                if val_metric is None and train_metric is not None:
                    val_metric = train_metric
                    train_metric = None
                elif train_metric is None and val_metric is None:
                    val_metric = metrics.get(primary_metric) or metrics.get(metric_lower)

                # Build trial data line
                if train_metric is not None and val_metric is not None:
                    gap = abs(train_metric - val_metric)
                    trial_line += f"\n  - Training {primary_metric}: {train_metric:.4f}"
                    trial_line += f"\n  - Validation {primary_metric}: {val_metric:.4f}"
                    trial_line += f"\n  - Gap: {gap:.4f}"

                    all_train_metrics.append(train_metric)
                    all_val_metrics.append(val_metric)
                    all_gaps.append(gap)

                    if gap > 0.15:
                        self.logger.thought(
                            f"Large train-val gap ({gap:.3f}) in {trial.variant_name} suggests overfitting"
                        )
                    elif gap > 0.08:
                        self.logger.thought(
                            f"Moderate train-val gap ({gap:.3f}) in {trial.variant_name}"
                        )
                elif val_metric is not None:
                    trial_line += f"\n  - Validation {primary_metric}: {val_metric:.4f}"
                    trial_line += "\n  - Training metrics not available"
                    all_val_metrics.append(val_metric)
                else:
                    trial_line += "\n  - Metrics not available"

                if val_metric is not None:
                    if best_val_metric is None or val_metric > best_val_metric:
                        best_val_metric = val_metric

                # Check for CV fold data
                fold_metrics = []
                for key, value in metrics.items():
                    if 'fold' in key.lower() and isinstance(value, (int, float)):
                        fold_metrics.append(value)

                if fold_metrics:
                    cv_fold_data.extend(fold_metrics)
                    fold_variance = max(fold_metrics) - min(fold_metrics) if len(fold_metrics) > 1 else 0
                    trial_line += f"\n  - CV Folds: {len(fold_metrics)}, Range: {fold_variance:.4f}"

                    if fold_variance > 0.1:
                        self.logger.thought(
                            f"High CV variance ({fold_variance:.3f}) suggests unstable model"
                        )

                # Extract baseline metrics
                if trial.baseline_metrics_json:
                    baseline_data = trial.baseline_metrics_json
                    all_baseline_metrics.append(baseline_data)

                    label_shuffle = baseline_data.get("label_shuffle", {})
                    if label_shuffle.get("leakage_detected") is True:
                        leakage_suspected = True
                        warning_msg = label_shuffle.get("warning") or "Label-shuffle test indicates potential data leakage"
                        if warning_msg not in leakage_warnings:
                            leakage_warnings.append(warning_msg)
                        self.logger.thought(f"LEAKAGE DETECTED in {trial.variant_name}")
                        trial_line += f"\n  - LEAKAGE WARNING: {warning_msg}"

                # Check for time-split issues
                split_strategy = trial.data_split_strategy
                if split_strategy:
                    if is_time_based and split_strategy in ("random", "stratified", "group_random"):
                        time_split_suspicious = True
                        recommended_split = task_hints.get("recommended_split", "time")
                        warning_msg = (
                            f"Time-based data using '{split_strategy}' split may cause temporal leakage. "
                            f"Recommended split: '{recommended_split}'."
                        )
                        if warning_msg not in time_split_warnings:
                            time_split_warnings.append(warning_msg)
                        self.logger.thought(f"TIME SPLIT ISSUE in {trial.variant_name}")
                        trial_line += f"\n  - SPLIT WARNING: {warning_msg}"

                # Also check dataset spec for is_time_based
                if not is_time_based and exp.dataset_spec_id:
                    dataset_spec = self.db.query(DatasetSpec).filter(
                        DatasetSpec.id == exp.dataset_spec_id
                    ).first()
                    if dataset_spec and dataset_spec.is_time_based:
                        is_time_based = True
                        if split_strategy and split_strategy in ("random", "stratified", "group_random"):
                            time_split_suspicious = True
                            warning_msg = (
                                f"Dataset is time-based but using '{split_strategy}' split. "
                                "This may cause temporal leakage."
                            )
                            if warning_msg not in time_split_warnings:
                                time_split_warnings.append(warning_msg)

                trials_data_lines.append(trial_line)

        trials_data = "\n\n".join(trials_data_lines) if trials_data_lines else "No trial data available"

        return (
            trials_data,
            all_train_metrics,
            all_val_metrics,
            all_gaps,
            cv_fold_data,
            best_val_metric,
            all_baseline_metrics,
            leakage_suspected,
            leakage_warnings,
            time_split_suspicious,
            time_split_warnings,
        )

    def _build_baseline_info(
        self,
        all_baseline_metrics: List[Dict],
        is_classification: bool,
        primary_metric: str,
    ) -> tuple:
        """Build baseline information from metrics."""
        baseline_lines = []
        baseline_value = None
        baseline_type = None

        if all_baseline_metrics:
            self.logger.info(f"Using real baseline metrics from {len(all_baseline_metrics)} trial(s)")

            real_baselines = all_baseline_metrics[0]

            if is_classification:
                majority = real_baselines.get("majority_class", {})
                simple = real_baselines.get("simple_logistic", {})
                shuffle = real_baselines.get("label_shuffle", {})

                if majority:
                    maj_acc = majority.get("accuracy")
                    maj_auc = majority.get("roc_auc")
                    if maj_acc:
                        baseline_lines.append(f"- Majority class baseline: accuracy={maj_acc:.4f}")
                    if maj_auc is not None:
                        baseline_lines.append(f"- Majority class ROC AUC: {maj_auc:.4f}")
                        if "auc" in primary_metric.lower() or "roc" in primary_metric.lower():
                            baseline_value = maj_auc
                            baseline_type = "majority_class"
                    elif maj_acc is not None and "accuracy" in primary_metric.lower():
                        baseline_value = maj_acc
                        baseline_type = "majority_class"

                if simple:
                    simple_acc = simple.get("accuracy")
                    simple_auc = simple.get("roc_auc")
                    simple_f1 = simple.get("f1")
                    if simple_acc:
                        baseline_lines.append(f"- Simple logistic baseline: accuracy={simple_acc:.4f}")
                    if simple_auc:
                        baseline_lines.append(f"- Simple logistic ROC AUC: {simple_auc:.4f}")
                    if simple_f1:
                        baseline_lines.append(f"- Simple logistic F1: {simple_f1:.4f}")

                if shuffle:
                    shuffle_acc = shuffle.get("shuffled_accuracy")
                    shuffle_auc = shuffle.get("shuffled_roc_auc")
                    if shuffle_acc:
                        baseline_lines.append(f"- Label-shuffle accuracy: {shuffle_acc:.4f}")
                    if shuffle_auc:
                        baseline_lines.append(f"- Label-shuffle ROC AUC: {shuffle_auc:.4f}")
                    if shuffle.get("leakage_detected"):
                        baseline_lines.append(f"- LEAKAGE DETECTED: {shuffle.get('warning', '')}")
            else:
                # Regression baselines
                mean_pred = real_baselines.get("mean_predictor", {})
                simple = real_baselines.get("simple_ridge", {})
                shuffle = real_baselines.get("label_shuffle", {})

                if mean_pred:
                    mean_rmse = mean_pred.get("rmse")
                    mean_mae = mean_pred.get("mae")
                    mean_r2 = mean_pred.get("r2")
                    if mean_rmse:
                        baseline_lines.append(f"- Mean predictor RMSE: {mean_rmse:.4f}")
                        if "rmse" in primary_metric.lower():
                            baseline_value = mean_rmse
                            baseline_type = "mean_predictor"
                    if mean_mae:
                        baseline_lines.append(f"- Mean predictor MAE: {mean_mae:.4f}")
                    if mean_r2 is not None:
                        baseline_lines.append(f"- Mean predictor R²: {mean_r2:.4f}")

                if simple:
                    ridge_rmse = simple.get("rmse")
                    ridge_mae = simple.get("mae")
                    ridge_r2 = simple.get("r2")
                    if ridge_rmse:
                        baseline_lines.append(f"- Simple ridge RMSE: {ridge_rmse:.4f}")
                    if ridge_mae:
                        baseline_lines.append(f"- Simple ridge MAE: {ridge_mae:.4f}")
                    if ridge_r2 is not None:
                        baseline_lines.append(f"- Simple ridge R²: {ridge_r2:.4f}")

                if shuffle:
                    shuffle_r2 = shuffle.get("shuffled_r2")
                    shuffle_rmse = shuffle.get("shuffled_rmse")
                    if shuffle_r2 is not None:
                        baseline_lines.append(f"- Label-shuffle R²: {shuffle_r2:.4f}")
                    if shuffle_rmse:
                        baseline_lines.append(f"- Label-shuffle RMSE: {shuffle_rmse:.4f}")
                    if shuffle.get("leakage_detected"):
                        baseline_lines.append(f"- LEAKAGE DETECTED: {shuffle.get('warning', '')}")

            baseline_lines = [line for line in baseline_lines if line]
        else:
            self.logger.info("No real baseline metrics found, using estimated baselines")

            if is_classification:
                if 'auc' in primary_metric.lower() or 'roc' in primary_metric.lower():
                    baseline_value = 0.5
                    baseline_type = "random_classifier"
                    baseline_lines.append(f"- Random classifier AUC baseline (estimated): {baseline_value:.4f}")
                elif 'accuracy' in primary_metric.lower():
                    baseline_value = 0.5
                    baseline_type = "majority_class"
                    baseline_lines.append(f"- Majority class accuracy baseline (estimated): {baseline_value:.4f}")
                else:
                    baseline_value = 0.5
                    baseline_type = "random"
                    baseline_lines.append(f"- Random baseline (estimated): {baseline_value:.4f}")
            else:
                baseline_type = "mean_predictor"
                baseline_value = None
                baseline_lines.append("- Mean predictor baseline: Not computed (no baseline data available)")

        baseline_info = "\n".join(baseline_lines) if baseline_lines else "Baseline comparison not available"
        return baseline_info, baseline_value, baseline_type

    def _build_cv_summary(self, cv_fold_data: List[float]) -> Optional[str]:
        """Build CV data summary."""
        if not cv_fold_data:
            return None

        cv_variance = max(cv_fold_data) - min(cv_fold_data) if len(cv_fold_data) > 1 else 0
        cv_mean = sum(cv_fold_data) / len(cv_fold_data)

        return f"""
- Number of folds observed: {len(cv_fold_data)}
- Mean across folds: {cv_mean:.4f}
- Range (max - min): {cv_variance:.4f}
- Min fold value: {min(cv_fold_data):.4f}
- Max fold value: {max(cv_fold_data):.4f}
"""

    def _log_summary_statistics(
        self,
        all_gaps: List[float],
        best_val_metric: Optional[float],
        baseline_value: Optional[float],
    ) -> None:
        """Log summary statistics."""
        if all_gaps:
            worst_gap = max(all_gaps)
            avg_gap = sum(all_gaps) / len(all_gaps)
            self.logger.thought(f"Train-val gap analysis: worst={worst_gap:.4f}, avg={avg_gap:.4f}")

            if worst_gap > 0.2:
                self.logger.thought("SEVERE overfitting detected - worst gap exceeds 0.20")
            elif worst_gap > 0.1:
                self.logger.thought("MODERATE overfitting risk - worst gap exceeds 0.10")

        if best_val_metric is not None and baseline_value is not None:
            relative_improvement = (best_val_metric - baseline_value) / baseline_value if baseline_value > 0 else 0
            self.logger.thought(
                f"Best model: {best_val_metric:.4f}, Baseline: {baseline_value:.4f}, Improvement: {relative_improvement:.1%}"
            )

            if relative_improvement < 0.05:
                self.logger.thought("Model barely improves over trivial baseline (<5% improvement)")
            elif best_val_metric > 0.98:
                self.logger.thought(f"Suspiciously high performance ({best_val_metric:.4f}) - check for data leakage")

    def _build_baseline_metrics_summary(
        self, all_baseline_metrics: List[Dict]
    ) -> Dict[str, Any]:
        """Build baseline metrics summary for response."""
        if not all_baseline_metrics:
            return {}

        bm = all_baseline_metrics[0]
        return {
            "majority_class": bm.get("majority_class", {}),
            "mean_predictor": bm.get("mean_predictor", {}),
            "simple_logistic": bm.get("simple_logistic", {}),
            "simple_ridge": bm.get("simple_ridge", {}),
            "label_shuffle": bm.get("label_shuffle", {}),
        }

    def _check_too_good_to_be_true(
        self,
        experiments: List[Experiment],
        best_val_metric: Optional[float],
        primary_metric: str,
        is_time_based: bool,
        task_type: str,
        overfitting_risk: str,
        suspicious_patterns: List[Dict],
        all_warnings: List[str],
        context_factors: Dict,
        task_context: Optional[Dict],
    ) -> tuple:
        """Check for 'too good to be true' pattern."""
        from app.services.risk_scoring import check_too_good_to_be_true

        additional_metrics: Dict[str, float] = {}
        for exp in experiments:
            for trial in exp.trials:
                if trial.metrics_json:
                    for key, value in trial.metrics_json.items():
                        if isinstance(value, (int, float)) and key not in additional_metrics:
                            additional_metrics[key] = value

        # Get expected metric range
        expected_metric_range = None
        input_data = self.step.input_json or {}
        context_analysis = input_data.get("context_analysis", {})

        if context_analysis and isinstance(context_analysis, dict):
            expected_metric_range = context_analysis.get("expected_metric_range")

        if not expected_metric_range and task_context:
            profile_summary = task_context.get("data_profile_summary", {})
            if profile_summary and isinstance(profile_summary, dict):
                expected_metric_range = profile_summary.get("expected_metric_range")
            if not expected_metric_range:
                ctx_analysis = task_context.get("context_analysis", {})
                if ctx_analysis:
                    expected_metric_range = ctx_analysis.get("expected_metric_range")

        if expected_metric_range:
            self.logger.thought(
                f"Expected metric range: [{expected_metric_range.get('lower_bound', '?')}-{expected_metric_range.get('upper_bound', '?')}]"
            )
            context_factors["expected_metric_range"] = expected_metric_range

        too_good_to_be_true, tgtbt_warning = check_too_good_to_be_true(
            is_time_based=is_time_based,
            task_type=task_type,
            best_val_metric=best_val_metric,
            primary_metric=primary_metric,
            additional_metrics=additional_metrics,
            expected_metric_range=expected_metric_range,
        )

        if too_good_to_be_true and tgtbt_warning:
            self.logger.warning(f"TOO GOOD TO BE TRUE: {tgtbt_warning}")
            all_warnings.append(tgtbt_warning)
            suspicious_patterns.append({
                "type": "too_good_to_be_true",
                "severity": "high",
                "description": tgtbt_warning,
            })
            if overfitting_risk == "low":
                overfitting_risk = "medium"

        # Check if actual exceeds expected upper bound
        if best_val_metric is not None and expected_metric_range:
            upper_bound = expected_metric_range.get("upper_bound")
            if upper_bound is not None and best_val_metric > upper_bound:
                self.logger.warning(
                    f"Risk assessment: actual exceeds expected upper bound by {best_val_metric - upper_bound:.3f}"
                )
                if overfitting_risk == "low":
                    overfitting_risk = "medium"
                elif overfitting_risk == "medium" and (best_val_metric - upper_bound) > 0.10:
                    overfitting_risk = "high"

        return overfitting_risk, suspicious_patterns, all_warnings

    def _check_leakage_in_features(
        self,
        experiments: List[Experiment],
        overfitting_risk: str,
        suspicious_patterns: List[Dict],
        all_warnings: List[str],
        leakage_suspected: bool,
    ) -> tuple:
        """Check if leakage candidates are among top important features."""
        leakage_in_important_features = False
        concerning_leakage_features = []
        input_data = self.step.input_json or {}
        leakage_candidates = input_data.get("leakage_candidates", [])

        if not leakage_candidates:
            return (
                leakage_suspected,
                leakage_in_important_features,
                concerning_leakage_features,
                overfitting_risk,
                suspicious_patterns,
                all_warnings,
            )

        try:
            from app.services.leakage_detector import check_leakage_in_important_features as check_leakage_imp
        except ImportError:
            return (
                leakage_suspected,
                leakage_in_important_features,
                concerning_leakage_features,
                overfitting_risk,
                suspicious_patterns,
                all_warnings,
            )

        all_feature_importances: Dict[str, float] = {}

        for exp in experiments:
            for trial in exp.trials:
                if trial.metrics_json and "feature_importances" in trial.metrics_json:
                    fi = trial.metrics_json["feature_importances"]
                    if isinstance(fi, dict):
                        all_feature_importances.update(fi)

                if hasattr(trial, "model_versions"):
                    for mv in trial.model_versions:
                        if mv.feature_importances_json:
                            fi = mv.feature_importances_json
                            if isinstance(fi, dict):
                                all_feature_importances.update(fi)

        if all_feature_importances:
            self.logger.thought(
                f"Checking {len(leakage_candidates)} leakage candidates against {len(all_feature_importances)} feature importances"
            )

            leakage_in_important_features, concerning_leakage_features, leakage_importance_warning = check_leakage_imp(
                leakage_candidates=leakage_candidates,
                feature_importances=all_feature_importances,
                top_n=10,
                importance_threshold=0.05,
            )

            if leakage_in_important_features:
                self.logger.warning(f"LEAKAGE IN IMPORTANT FEATURES: {leakage_importance_warning}")
                all_warnings.append(leakage_importance_warning)
                leakage_suspected = True

                for feat in concerning_leakage_features:
                    suspicious_patterns.append({
                        "type": "leakage_in_important_feature",
                        "severity": "high",
                        "description": f"Feature '{feat['column']}' (importance rank: {feat.get('importance_rank', 'N/A')}) "
                                       f"flagged for potential leakage: {feat.get('reason', 'Unknown reason')}",
                    })

                if overfitting_risk == "low":
                    overfitting_risk = "medium"
                elif overfitting_risk == "medium":
                    overfitting_risk = "high"

                self.logger.thought(
                    f"Model relies on {len(concerning_leakage_features)} suspicious feature(s)"
                )

        return (
            leakage_suspected,
            leakage_in_important_features,
            concerning_leakage_features,
            overfitting_risk,
            suspicious_patterns,
            all_warnings,
        )

    def _compute_risk_score(
        self,
        experiments: List[Experiment],
        best_val_metric: Optional[float],
        overfitting_risk: str,
        leakage_suspected: bool,
        time_split_suspicious: bool,
    ) -> Optional[float]:
        """Compute risk-adjusted score."""
        if best_val_metric is None:
            return None

        from app.services.risk_scoring import compute_risk_adjusted_score

        risk_adjusted_score = compute_risk_adjusted_score(
            primary_metric=best_val_metric,
            overfitting_risk=overfitting_risk,
            leakage_suspected=leakage_suspected,
            time_split_suspicious=time_split_suspicious,
        )

        self.logger.info(f"Risk-adjusted score: {risk_adjusted_score:.4f}")

        # Store risk_adjusted_score in all trials' metrics_json
        for exp in experiments:
            for trial in exp.trials:
                if trial.metrics_json is None:
                    trial.metrics_json = {}
                trial.metrics_json["risk_adjusted_score"] = risk_adjusted_score

        try:
            self.db.commit()
            self.logger.info("Saved risk_adjusted_score to trial metrics")
        except Exception as e:
            self.logger.warning(f"Failed to save risk_adjusted_score to trials: {e}")
            self.db.rollback()

        return risk_adjusted_score

    def _build_context_reasoning(
        self,
        baseline_value: Optional[float],
        baseline_type: Optional[str],
        best_val_metric: Optional[float],
        primary_metric: str,
        context_factors: Dict,
        task_context: Optional[Dict],
    ) -> Optional[str]:
        """Build context-aware reasoning string."""
        context_reasoning_parts = []

        if baseline_value is not None:
            context_reasoning_parts.append(f"Baseline {baseline_type or 'value'} ~{baseline_value:.3f}")

        expected_metric_range = context_factors.get("expected_metric_range")
        if expected_metric_range:
            lb = expected_metric_range.get("lower_bound", "?")
            ub = expected_metric_range.get("upper_bound", "?")
            context_reasoning_parts.append(f"expected realistic range [{lb}-{ub}]")

        if best_val_metric is not None:
            context_reasoning_parts.append(f"best model {primary_metric} {best_val_metric:.3f}")

        # Get split strategy from context
        if task_context:
            ds_spec = task_context.get("dataset_spec", {})
            if ds_spec:
                split_config = ds_spec.get("split_strategy", {})
                if isinstance(split_config, dict):
                    split_strategy_used = split_config.get("type")
                    if split_strategy_used:
                        context_reasoning_parts.append(f"using {split_strategy_used} split")

        return ", ".join(context_reasoning_parts) if context_reasoning_parts else None

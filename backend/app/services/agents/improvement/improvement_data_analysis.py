"""Improvement Data Analysis Agent - Re-analyzes data with iteration feedback.

This agent analyzes the data with full knowledge of what has worked
and what hasn't in previous iterations. It uses data scientist heuristics
to provide intelligent recommendations.
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.prompts import SYSTEM_ROLE_DATA_SCIENTIST
from app.services.data_scientist_heuristics import (
    DataScientistHeuristics,
    get_smart_recommendations,
)

logger = logging.getLogger(__name__)


class ImprovedDataAnalysis(BaseModel):
    """Response schema for improved data analysis."""
    key_insights: List[str] = Field(
        description="Key insights from analyzing data with iteration feedback"
    )
    failed_approaches: List[str] = Field(
        description="Approaches that were tried and failed - DO NOT REPEAT THESE"
    )
    successful_approaches: List[str] = Field(
        description="Approaches that showed improvement - BUILD ON THESE"
    )
    untried_opportunities: List[str] = Field(
        description="Feature engineering or data improvements NOT yet tried"
    )
    data_quality_issues: List[str] = Field(
        description="Data quality issues that may be limiting performance"
    )
    recommended_features: List[Dict[str, str]] = Field(
        description="Specific new features to engineer with formulas"
    )
    features_to_remove: List[str] = Field(
        description="Features that should be removed (low value or problematic)"
    )
    target_analysis: str = Field(
        description="Analysis of the target variable and any issues"
    )
    bottleneck_diagnosis: str = Field(
        description="Main bottleneck limiting model performance"
    )
    recommended_focus: str = Field(
        description="What the next iteration should focus on"
    )


class ImprovementDataAnalysisAgent(BaseAgent):
    """Analyzes data with iteration feedback.

    Input JSON:
        - iteration_context: Context from IterationContextAgent

    Output:
        - key_insights: Key insights from the analysis
        - failed_approaches: Approaches to avoid
        - successful_approaches: Approaches to build on
        - untried_opportunities: New opportunities to try
        - recommended_features: Specific features to create
        - bottleneck_diagnosis: Main bottleneck identified
        - recommended_focus: Focus for next iteration
    """

    name = "improvement_data_analysis"
    step_type = AgentStepType.IMPROVEMENT_DATA_ANALYSIS

    async def execute(self) -> Dict[str, Any]:
        """Execute data re-analysis with iteration feedback."""
        iteration_context = self.get_input("iteration_context", {})

        if not iteration_context:
            raise ValueError("Missing iteration_context - run ITERATION_CONTEXT step first")

        self.logger.info("Analyzing data with iteration feedback...")

        # Get smart recommendations from heuristics engine
        data_stats = iteration_context.get("data_statistics", {})
        diagnosis = get_smart_recommendations(iteration_context, data_stats)

        # Log heuristic findings
        if diagnosis.detected_patterns:
            self.logger.thinking(f"Detected patterns: {[p.value for p in diagnosis.detected_patterns]}")
        self.logger.thinking(f"Primary bottleneck: {diagnosis.primary_bottleneck}")

        if diagnosis.insights:
            for insight in diagnosis.insights[:3]:
                self.logger.thinking(f"Insight: {insight}")

        # Format heuristic recommendations for prompt
        heuristic_recommendations = self._format_heuristic_recommendations(diagnosis)

        # Get data statistics
        data_stats = iteration_context.get("data_statistics", {})
        dataset_spec = iteration_context.get("dataset_spec", {})
        project = iteration_context.get("project", {})

        # Format iteration history for prompt
        iteration_history = iteration_context.get("iteration_history", [])
        history_text = self._format_iteration_history(iteration_history)

        # Format error history
        errors = iteration_context.get("error_history", [])
        error_text = self._format_error_history(errors)

        # Format data columns
        columns = data_stats.get("columns", [])
        column_stats = data_stats.get("column_stats", {})
        columns_text = self._format_column_stats(column_stats)

        # Get improvement attempts
        improvements = iteration_context.get("improvement_attempts", [])
        improvements_text = self._format_improvements(improvements)

        # Format feature engineering feedback
        fe_feedback = iteration_context.get("feature_engineering_feedback", {})
        fe_failed_text, fe_succeeded_text = self._format_feature_feedback(fe_feedback)

        # Format overfitting report
        overfitting = iteration_context.get("overfitting_report", {})
        overfitting_text = self._format_overfitting_report(overfitting)

        prompt = f"""You are analyzing data for an ML improvement iteration. You have access to the COMPLETE history
of what has been tried before, what worked, and what failed. USE THIS INFORMATION to avoid repeating mistakes
and to build on successful approaches.

## Project Goal
{project.get('description', 'Not specified')}
Task Type: {project.get('task_type', 'unknown')}
Target Column: {dataset_spec.get('target_column', 'unknown')}

## Current Performance
- Current Score: {iteration_context.get('current_score', 0):.4f}
- Best Score Ever: {iteration_context.get('best_score', 0):.4f}
- Score Trend: {iteration_context.get('score_trend', 'unknown')}
- Total Iterations: {iteration_context.get('total_iterations', 0)}
{overfitting_text}

## AI Data Scientist Diagnosis (IMPORTANT - USE THESE INSIGHTS!)
Primary Bottleneck: {diagnosis.primary_bottleneck}
{heuristic_recommendations}

## Iteration History
{history_text if history_text else "(No history available)"}

## Errors Encountered (AVOID THESE!)
{error_text if error_text else "(No errors)"}

## Previous Improvement Attempts
{improvements_text if improvements_text else "(No recorded improvements)"}

## Feature Engineering History (CRITICAL - DON'T REPEAT FAILURES!)
FAILED Features (these formulas didn't work - DO NOT SUGGEST SIMILAR):
{fe_failed_text if fe_failed_text else "(None recorded)"}

SUCCESSFUL Features (these worked - you can build on these):
{fe_succeeded_text if fe_succeeded_text else "(None recorded)"}

## Dataset Information
- Rows: {data_stats.get('row_count', 0):,}
- Columns: {data_stats.get('column_count', 0)}

## Column Details
{columns_text if columns_text else "(No column details)"}

## Available Raw Columns (USE ONLY THESE IN FORMULAS!)
{', '.join(columns[:30]) if columns else "(Unknown)"}

## Instructions
1. Analyze what has been tried and what the results were
2. PRIORITIZE the AI Data Scientist recommendations above
3. Identify patterns in what works vs what fails
4. Find UNTRIED approaches that might help
5. Suggest specific, actionable improvements based on the diagnosis
6. DO NOT suggest any feature that already failed - check the failed features list!
7. Only use columns that exist in the Available Raw Columns list
8. Focus on the bottleneck limiting performance
9. If overfitting is detected, suggest simpler models or regularization

Provide your analysis with specific feature engineering formulas that use ONLY the columns listed above."""

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_DATA_SCIENTIST},
            {"role": "user", "content": prompt},
        ]

        self.logger.action("Consulting LLM for iteration-aware data analysis...")
        response = await self.llm.chat_json(messages, ImprovedDataAnalysis)

        # Log key findings
        self.logger.thinking(f"Bottleneck: {response.get('bottleneck_diagnosis', 'unknown')}")
        self.logger.thinking(f"Recommended focus: {response.get('recommended_focus', 'unknown')}")

        untried = response.get('untried_opportunities', [])
        if untried:
            self.logger.info(f"Found {len(untried)} untried opportunities")
            for opp in untried[:3]:
                self.logger.thinking(f"  - {opp[:80]}")

        failed = response.get('failed_approaches', [])
        if failed:
            self.logger.warning(f"Identified {len(failed)} failed approaches to AVOID")

        self.logger.summary(
            f"Data analysis complete. Focus: {response.get('recommended_focus', 'optimization')[:100]}. "
            f"Found {len(untried)} untried opportunities."
        )

        # Serialize diagnosis for return
        diagnosis_summary = {
            "detected_patterns": [p.value for p in diagnosis.detected_patterns],
            "primary_bottleneck": diagnosis.primary_bottleneck,
            "recommendations": [
                {
                    "strategy": r.strategy.value,
                    "priority": r.priority,
                    "description": r.description,
                    "rationale": r.rationale,
                    "expected_impact": r.expected_impact,
                }
                for r in diagnosis.recommendations
            ],
            "insights": diagnosis.insights,
            "confidence": diagnosis.confidence,
        }

        return {
            **response,
            "iteration_context": iteration_context,
            "ai_diagnosis": diagnosis_summary,
        }

    def _format_iteration_history(self, history: List[Dict]) -> str:
        """Format iteration history for prompt."""
        if not history:
            return ""
        lines = []
        for hist in history:
            line = f"  Iteration {hist['iteration']}: score={hist.get('score', 0):.4f}, status={hist.get('status', '?')}"
            if hist.get('changes_made'):
                line += f"\n    Changes: {hist['changes_made'][:100]}"
            if hist.get('error'):
                line += f"\n    Error: {hist['error'][:100]}"
            lines.append(line)
        return "\n".join(lines)

    def _format_error_history(self, errors: List[Dict]) -> str:
        """Format error history for prompt."""
        if not errors:
            return ""
        return "\n".join([
            f"  Iteration {e['iteration']}: {e['error'][:200]}"
            for e in errors[-5:]
        ])

    def _format_column_stats(self, column_stats: Dict) -> str:
        """Format column statistics for prompt."""
        if not column_stats:
            return ""
        lines = []
        for col, stats in list(column_stats.items())[:20]:
            line = f"  {col}: {stats.get('dtype', '?')}, {stats.get('unique', '?')} unique, {stats.get('null_pct', 0):.1f}% missing"
            lines.append(line)
        return "\n".join(lines)

    def _format_improvements(self, improvements: List[Dict]) -> str:
        """Format improvement attempts for prompt."""
        if not improvements:
            return ""
        lines = []
        for imp in improvements:
            result = "succeeded" if imp.get('success') else "failed"
            lines.append(f"  Iteration {imp['iteration']}: {imp['changes'][:100]} - {result}")
        return "\n".join(lines)

    def _format_feature_feedback(self, fe_feedback: Dict) -> tuple:
        """Format feature engineering feedback for prompt."""
        fe_failed_text = ""
        fe_succeeded_text = ""

        if not fe_feedback:
            return fe_failed_text, fe_succeeded_text

        failed_feats = fe_feedback.get("failed_features", [])
        if failed_feats:
            lines = [
                f"  x {f.get('feature', '?')}: {f.get('error', 'Unknown error')[:100]}"
                for f in failed_feats[-10:]
            ]
            fe_failed_text = "\n".join(lines)

        succeeded_feats = fe_feedback.get("successful_features", [])
        if succeeded_feats:
            lines = [
                f"  + {f.get('feature', '?')}: {f.get('formula', 'N/A')[:80]}"
                for f in succeeded_feats[-10:]
            ]
            fe_succeeded_text = "\n".join(lines)

        return fe_failed_text, fe_succeeded_text

    def _format_overfitting_report(self, overfitting: Dict) -> str:
        """Format overfitting report for prompt."""
        if not overfitting:
            return ""
        return f"""
## Overfitting Analysis (IMPORTANT!)
- Holdout Score Trend: {overfitting.get('trend', 'unknown')}
- Best Holdout Score: {overfitting.get('best_score', 0):.4f} (iteration {overfitting.get('best_iteration', '?')})
- Current Holdout Score: {overfitting.get('current_score', 0):.4f}
- Recommendation: {overfitting.get('recommendation', 'continue')}
- Analysis: {overfitting.get('message', 'N/A')}
"""

    def _format_heuristic_recommendations(self, diagnosis) -> str:
        """Format heuristic recommendations for the LLM prompt.

        Args:
            diagnosis: DiagnosisResult from DataScientistHeuristics

        Returns:
            Formatted string with recommendations and insights
        """
        lines = []

        # Add detected patterns
        if diagnosis.detected_patterns:
            patterns_str = ", ".join(p.value for p in diagnosis.detected_patterns)
            lines.append(f"Detected Patterns: {patterns_str}")

        # Add insights
        if diagnosis.insights:
            lines.append("\nKey Insights from Analysis:")
            for insight in diagnosis.insights[:5]:
                lines.append(f"  - {insight}")

        # Add top recommendations
        if diagnosis.recommendations:
            lines.append("\nRecommended Strategies (prioritized):")
            for i, rec in enumerate(diagnosis.recommendations[:3], 1):
                lines.append(f"  {i}. {rec.description} (priority: {rec.priority}/10)")
                lines.append(f"     Rationale: {rec.rationale}")
                lines.append(f"     Expected Impact: {rec.expected_impact}")
                if rec.specific_actions:
                    lines.append("     Specific Actions:")
                    for action in rec.specific_actions[:3]:
                        lines.append(f"       - {action}")

        return "\n".join(lines) if lines else "(No heuristic recommendations)"

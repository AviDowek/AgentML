"""Improvement Experiment Design Agent - Redesigns experiment with iteration feedback.

This agent designs the training configuration based on what AutoML configs
worked before and what the current bottleneck is.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models import AgentStepType
from app.services.agents.base import BaseAgent
from app.services.prompts import SYSTEM_ROLE_ML_ANALYST


class ValidationStrategy(BaseModel):
    """Validation split strategy."""
    split_strategy: str = Field(
        description="Split strategy: 'temporal', 'random', 'stratified', or 'group'"
    )
    validation_split: float = Field(
        default=0.2, description="Fraction of data for validation (0.1-0.3)"
    )
    group_column: Optional[str] = Field(
        default=None, description="Column name for group-based splits"
    )
    reasoning: str = Field(
        description="Why this split strategy is appropriate for this data"
    )


class ExperimentDesign(BaseModel):
    """Response schema for experiment design."""
    iteration_name: str = Field(
        description="Name for this iteration"
    )
    iteration_description: str = Field(
        description="Description of what this iteration is testing"
    )
    automl_config: Dict[str, Any] = Field(
        description="AutoML configuration with time_limit, presets, etc."
    )
    validation_strategy: Optional[ValidationStrategy] = Field(
        default=None, description="Validation split strategy for train/test split"
    )
    expected_improvements: List[str] = Field(
        description="What improvements are expected"
    )
    success_criteria: Dict[str, Any] = Field(
        description="Criteria for considering this iteration successful"
    )
    training_strategy: str = Field(
        description="Strategy for training (aggressive, conservative, balanced)"
    )


class ImprovementExperimentDesignAgent(BaseAgent):
    """Redesigns experiment based on iteration feedback.

    Input JSON:
        - dataset_design: Output from ImprovementDatasetDesignAgent
        - iteration_context: Context from IterationContextAgent

    Output:
        - iteration_name: Name for the iteration
        - iteration_description: Description
        - automl_config: AutoML configuration
        - validation_strategy: Validation split strategy
        - expected_improvements: Expected improvements
        - success_criteria: Success criteria
        - training_strategy: Training strategy
    """

    name = "improvement_experiment_design"
    step_type = AgentStepType.IMPROVEMENT_EXPERIMENT_DESIGN

    async def execute(self) -> Dict[str, Any]:
        """Execute experiment redesign with iteration feedback."""
        dataset_design = self.get_input("dataset_design", {})
        iteration_context = self.get_input("iteration_context", {})

        if not iteration_context:
            raise ValueError("Missing iteration_context")

        self.logger.info("Designing improved experiment configuration...")

        project = iteration_context.get("project", {})
        iteration_history = iteration_context.get("iteration_history", [])

        # Analyze what AutoML configs have been used
        configs_used = []
        for hist in iteration_history:
            if hist.get("top_models"):
                configs_used.append({
                    "iteration": hist["iteration"],
                    "score": hist.get("score", 0),
                    "top_models": hist["top_models"][:3],
                })

        prompt = f"""You are designing an improved ML training configuration for the next iteration.

## Project
Task Type: {project.get('task_type', 'unknown')}
Goal: {project.get('description', 'Not specified')[:200]}

## Performance History
Current Score: {iteration_context.get('current_score', 0):.4f}
Best Score: {iteration_context.get('best_score', 0):.4f}
Score Trend: {iteration_context.get('score_trend', 'unknown')}
Iterations So Far: {iteration_context.get('total_iterations', 0)}
Total Training Time: {iteration_context.get('total_training_time_seconds', 0):.0f} seconds

## Previous Configs and Results
{configs_used[:5] if configs_used else "No history available"}

## Dataset Changes for This Iteration
New Features: {len(dataset_design.get('new_engineered_features', []))}
Removed Features: {len(dataset_design.get('features_to_drop', []))}
Rationale: {dataset_design.get('rationale', 'Not specified')[:200]}

## Guidelines
1. If score is improving, continue with similar config
2. If score is flat/declining, try different approach
3. Balance quality vs training time
4. Consider:
   - time_limit: 120-600 seconds typically
   - presets: "best_quality", "high_quality", "medium_quality"
   - num_bag_folds: 0-10 (higher = better but slower)
   - num_stack_levels: 0-3 (higher = better but slower)

## Validation Strategy (CRITICAL - REQUIRED)
You MUST specify a validation_strategy to prevent data leakage:
- split_strategy: Choose one of:
  - "temporal": For time-series or time-dependent data. Sort by time, use most recent data for validation.
  - "stratified": For classification with imbalanced classes. Preserves class distribution.
  - "random": For independent observations with no time dependence.
  - "group": When observations are grouped (e.g., multiple samples per user). Keeps groups together.
- validation_split: Fraction for validation (typically 0.2)
- group_column: Column name if using group split (null otherwise)
- reasoning: Explain why this strategy is appropriate

IMPORTANT: Re-evaluate the validation strategy each iteration. If you notice patterns suggesting
time dependence (date columns, sequential IDs, etc.), switch to temporal split!

Design a configuration that addresses the current bottleneck while being realistic."""

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_ML_ANALYST},
            {"role": "user", "content": prompt},
        ]

        self.logger.action("Consulting LLM for experiment design...")
        response = await self.llm.chat_json(messages, ExperimentDesign)

        automl_config = response.get("automl_config", {})

        self.logger.info(f"Iteration: {response.get('iteration_name', 'Unknown')}")
        self.logger.thinking(f"Strategy: {response.get('training_strategy', 'balanced')}")
        self.logger.thinking(f"Config: time_limit={automl_config.get('time_limit', 300)}, presets={automl_config.get('presets', 'best_quality')}")

        # Log validation strategy
        validation_strategy = response.get("validation_strategy")
        if validation_strategy:
            if isinstance(validation_strategy, dict):
                self.logger.thinking(f"Validation: {validation_strategy.get('split_strategy', 'default')} split - {validation_strategy.get('reasoning', '')[:60]}")
            elif isinstance(validation_strategy, str):
                self.logger.thinking(f"Validation strategy: {validation_strategy[:80]}")

        expected = response.get("expected_improvements", [])
        if expected:
            self.logger.info("Expected improvements:")
            for imp in expected[:3]:
                self.logger.thinking(f"  - {imp[:80]}")

        self.logger.summary(
            f"Experiment design complete: {response.get('iteration_name', 'Next Iteration')}. "
            f"Strategy: {response.get('training_strategy', 'balanced')}."
        )

        return {
            **response,
            "dataset_design": dataset_design,
            "iteration_context": iteration_context,
        }

"""Lab Notebook Summary Agent - Generates research cycle summaries.

This agent creates a comprehensive summary of a research cycle, documenting
what was attempted, the results, and proposed next directions.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.models import (
    Experiment,
    LabNotebookEntry,
    Project,
    ResearchCycle,
    CycleExperiment,
)
from app.models import AgentRun, AgentStep, AgentStepType, AgentStepStatus, LabNotebookAuthorType
from app.services.agents.base import BaseAgent
from app.services.prompts import (
    SYSTEM_ROLE_LAB_NOTEBOOK_AGENT,
    get_lab_notebook_summary_prompt,
)


class LabNotebookSummaryResponse(BaseModel):
    """Response schema for lab notebook summary."""
    title: str = Field(description="Title for this cycle summary")
    body_markdown: str = Field(description="Full Markdown content of the lab notebook entry")


class LabNotebookSummaryAgent(BaseAgent):
    """Generates comprehensive research cycle summaries.

    Input JSON:
        - research_cycle_id: UUID of the research cycle to summarize
        - project_id: UUID of the project (fallback if cycle not found)

    Output:
        - lab_note: Dict containing title and body_markdown
        - lab_notebook_entry_id: ID of the created notebook entry
    """

    name = "lab_notebook_summary"
    step_type = AgentStepType.LAB_NOTEBOOK_SUMMARY

    async def execute(self) -> Dict[str, Any]:
        """Execute lab notebook summary generation."""
        research_cycle_id = self.get_input("research_cycle_id")
        project_id = self.get_input("project_id")

        # Validate required input
        if not research_cycle_id:
            # Try to get cycle from the agent run
            if self.step.agent_run and self.step.agent_run.research_cycle_id:
                research_cycle_id = str(self.step.agent_run.research_cycle_id)
            else:
                raise ValueError("Missing 'research_cycle_id' in step input")

        self.logger.info("Starting lab notebook summary generation...")

        # Load the research cycle
        cycle = self.db.query(ResearchCycle).filter(
            ResearchCycle.id == research_cycle_id
        ).first()

        if not cycle:
            raise ValueError(f"Research cycle not found: {research_cycle_id}")

        self.logger.thought(f"Summarizing research cycle #{cycle.sequence_number}")

        # Load the project
        project = self.db.query(Project).filter(Project.id == cycle.project_id).first()
        if not project:
            raise ValueError(f"Project not found for cycle {cycle.id}")

        project_id = project.id
        self.logger.info(f"Project: {project.name}")

        # Load experiments linked to this cycle
        cycle_experiments = (
            self.db.query(CycleExperiment)
            .filter(CycleExperiment.research_cycle_id == cycle.id)
            .all()
        )

        experiment_ids = [ce.experiment_id for ce in cycle_experiments]
        experiments = (
            self.db.query(Experiment)
            .filter(Experiment.id.in_(experiment_ids))
            .all()
        ) if experiment_ids else []

        self.logger.info(f"Found {len(experiments)} experiments in this cycle")

        # Build experiments summary
        experiments_summary, best_model_info, best_score, best_metric_name = self._build_experiments_summary(
            experiments
        )

        # Collect outputs from other agent steps in this cycle's runs
        self.logger.thought("Gathering insights from agent steps in this cycle...")
        step_outputs_summary = self._collect_step_outputs(cycle)

        # Get previous cycles context
        previous_cycles_context = self._get_previous_cycles_context(cycle, project_id)

        # Get problem description from project or agent run config
        problem_description = self._get_problem_description(project, cycle)

        # Build prompt
        prompt = get_lab_notebook_summary_prompt(
            cycle_number=cycle.sequence_number,
            cycle_title=cycle.summary_title,
            project_name=project.name,
            problem_description=problem_description,
            experiments_summary=experiments_summary,
            best_model_info=best_model_info,
            step_outputs_summary=step_outputs_summary,
            previous_cycles_context=previous_cycles_context,
        )

        messages = [
            {"role": "system", "content": SYSTEM_ROLE_LAB_NOTEBOOK_AGENT},
            {"role": "user", "content": prompt},
        ]

        self.logger.info("Generating lab notebook summary with LLM...")
        response = await self.llm.chat_json(messages, LabNotebookSummaryResponse)

        title = response.get("title", f"Cycle {cycle.sequence_number} Summary")
        body_markdown = response.get("body_markdown", "")

        self.logger.thought(f"Generated summary: {title}")

        # Create the lab notebook entry
        entry = LabNotebookEntry(
            project_id=project_id,
            research_cycle_id=cycle.id,
            agent_step_id=self.step.id,
            author_type=LabNotebookAuthorType.AGENT,
            title=title,
            body_markdown=body_markdown,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)

        self.logger.info(f"Created lab notebook entry: {entry.id}")

        # Update cycle summary title if not set
        if not cycle.summary_title:
            cycle.summary_title = title
            self.db.commit()

        self.logger.summary(f"Lab notebook entry created: '{title}'")

        return {
            "lab_note": {
                "title": title,
                "body_markdown": body_markdown,
            },
            "lab_notebook_entry_id": str(entry.id),
        }

    def _build_experiments_summary(
        self, experiments: List[Experiment]
    ) -> tuple:
        """Build experiments summary and track best model."""
        experiments_summary_lines = []
        best_model_info_lines = []
        best_score = None
        best_metric_name = None
        best_model_name = None

        for exp in experiments:
            # Calculate best metric from trials
            exp_best_metric = None
            exp_best_model = None
            if exp.trials:
                for trial in exp.trials:
                    if trial.metrics_json and exp.primary_metric:
                        metric_val = trial.metrics_json.get(exp.primary_metric)
                        if metric_val is not None:
                            if exp_best_metric is None:
                                exp_best_metric = metric_val
                                exp_best_model = trial.best_model_ref
                            else:
                                # Assume maximize for now (can be improved with metric_direction)
                                if metric_val > exp_best_metric:
                                    exp_best_metric = metric_val
                                    exp_best_model = trial.best_model_ref

            exp_line = f"- **{exp.name}** (Status: {exp.status.value if hasattr(exp.status, 'value') else exp.status})"
            if exp_best_metric is not None:
                exp_line += f" - Best {exp.primary_metric}: {exp_best_metric:.4f}"
            experiments_summary_lines.append(exp_line)

            # Track best model across experiments
            if exp_best_metric is not None:
                if best_score is None or (
                    # For error metrics (lower is better), compare appropriately
                    "error" in (exp.primary_metric or "").lower() or
                    "loss" in (exp.primary_metric or "").lower()
                ):
                    if best_score is None or exp_best_metric < best_score:
                        best_score = exp_best_metric
                        best_metric_name = exp.primary_metric
                        best_model_name = exp_best_model
                else:
                    if exp_best_metric > best_score:
                        best_score = exp_best_metric
                        best_metric_name = exp.primary_metric
                        best_model_name = exp_best_model

        if not experiments_summary_lines:
            experiments_summary_lines.append("No experiments completed in this cycle yet.")

        experiments_summary = "\n".join(experiments_summary_lines)

        # Build best model info
        if best_score is not None:
            best_model_info_lines.append(f"- **Best Model**: {best_model_name or 'Unknown'}")
            best_model_info_lines.append(f"- **{best_metric_name}**: {best_score:.4f}")
        else:
            best_model_info_lines.append("No model results available yet.")

        best_model_info = "\n".join(best_model_info_lines)

        return experiments_summary, best_model_info, best_score, best_metric_name

    def _collect_step_outputs(self, cycle: ResearchCycle) -> str:
        """Collect outputs from agent steps in this cycle."""
        agent_runs = (
            self.db.query(AgentRun)
            .filter(AgentRun.research_cycle_id == cycle.id)
            .all()
        )

        step_outputs_lines = []
        for run in agent_runs:
            if run.steps:
                for s in run.steps:
                    if s.output_json and s.status == AgentStepStatus.COMPLETED:
                        output = s.output_json
                        step_type = s.step_type

                        if step_type == AgentStepType.DATA_ANALYSIS:
                            summary = output.get("natural_language_summary", "")
                            if summary:
                                step_outputs_lines.append(f"**Data Analysis**: {summary[:500]}...")

                        elif step_type == AgentStepType.DATASET_DESIGN:
                            summary = output.get("natural_language_summary", "")
                            if summary:
                                step_outputs_lines.append(f"**Dataset Design**: {summary[:500]}...")

                        elif step_type == AgentStepType.EXPERIMENT_DESIGN:
                            summary = output.get("natural_language_summary", "")
                            if summary:
                                step_outputs_lines.append(f"**Experiment Design**: {summary[:500]}...")

                        elif step_type == AgentStepType.PLAN_CRITIC:
                            summary = output.get("natural_language_summary", "")
                            if summary:
                                step_outputs_lines.append(f"**Plan Critic**: {summary[:500]}...")

                        elif step_type == AgentStepType.RESULTS_INTERPRETATION:
                            summary = output.get("natural_language_summary", "")
                            if summary:
                                step_outputs_lines.append(f"**Results Interpretation**: {summary[:500]}...")

                        elif step_type == AgentStepType.RESULTS_CRITIC:
                            findings = output.get("critic_findings", {})
                            severity = findings.get("severity", "unknown")
                            approved = findings.get("approved", False)
                            step_outputs_lines.append(
                                f"**Results Critic**: Severity={severity}, Approved={approved}"
                            )

        if not step_outputs_lines:
            step_outputs_lines.append("No agent insights available for this cycle.")

        return "\n\n".join(step_outputs_lines)

    def _get_previous_cycles_context(
        self, cycle: ResearchCycle, project_id
    ) -> str | None:
        """Get context from previous cycles."""
        if cycle.sequence_number <= 1:
            return None

        previous_entries = (
            self.db.query(LabNotebookEntry)
            .filter(LabNotebookEntry.project_id == project_id)
            .filter(LabNotebookEntry.author_type == LabNotebookAuthorType.AGENT)
            .order_by(LabNotebookEntry.created_at.desc())
            .limit(3)
            .all()
        )

        if not previous_entries:
            return None

        context_lines = []
        for entry in reversed(previous_entries):
            context_lines.append(f"### {entry.title}\n{entry.body_markdown[:1000]}...")

        return "\n\n".join(context_lines)

    def _get_problem_description(
        self, project: Project, cycle: ResearchCycle
    ) -> str:
        """Get problem description from project or agent run config."""
        problem_description = project.description or "No problem description available."

        agent_runs = (
            self.db.query(AgentRun)
            .filter(AgentRun.research_cycle_id == cycle.id)
            .all()
        )

        for run in agent_runs:
            if run.config_json and run.config_json.get("description"):
                problem_description = run.config_json["description"]
                break

        return problem_description

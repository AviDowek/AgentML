"""
Agent History Tools Module

Provides tool definitions and handlers for agents to query project history,
previous experiments, agent reasoning, and lab notebook entries.

These tools allow agents to actively retrieve context they need rather than
receiving a pre-summarized context dump.
"""
from typing import Any, Dict, List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.agent_run import AgentStep, AgentStepType, AgentRun, AgentStepLog
from app.models.experiment import Experiment, Trial
from app.models.research_cycle import ResearchCycle, LabNotebookEntry, CycleExperiment
from app.models.dataset_spec import DatasetSpec


# =============================================================================
# Tool Definitions (for LLM function calling)
# =============================================================================

AGENT_HISTORY_TOOLS = [
    {
        "name": "get_research_cycles",
        "description": "Get a list of all research cycles for this project, including their status, experiment count, and summary. Use this to understand the overall progression of research.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of cycles to return (default: 10)",
                    "default": 10
                },
                "include_experiments": {
                    "type": "boolean",
                    "description": "Whether to include experiment summaries in each cycle",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "get_agent_thinking",
        "description": "Get the reasoning, thoughts, and logic from a previous agent step. Use this to understand WHY previous decisions were made.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_type": {
                    "type": "string",
                    "description": "Type of agent step to retrieve thinking from",
                    "enum": [
                        "data_analysis", "problem_understanding", "data_audit",
                        "dataset_design", "experiment_design", "experiment_run",
                        "robustness_audit", "improvement_analysis", "improvement_plan"
                    ]
                },
                "cycle_number": {
                    "type": "integer",
                    "description": "Research cycle number (1-indexed). If not specified, returns from most recent cycle."
                },
                "include_logs": {
                    "type": "boolean",
                    "description": "Whether to include detailed step logs (thinking, observations)",
                    "default": True
                }
            },
            "required": ["step_type"]
        }
    },
    {
        "name": "get_experiment_results",
        "description": "Get detailed results from a specific experiment or all experiments in a cycle. Includes trials, metrics, and hyperparameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "string",
                    "description": "Specific experiment ID to retrieve. If not provided, returns all experiments."
                },
                "cycle_number": {
                    "type": "integer",
                    "description": "Research cycle number to get experiments from"
                },
                "include_trials": {
                    "type": "boolean",
                    "description": "Whether to include individual trial results",
                    "default": True
                },
                "top_n_trials": {
                    "type": "integer",
                    "description": "Number of top trials to include per experiment",
                    "default": 5
                }
            },
            "required": []
        }
    },
    {
        "name": "get_robustness_audit",
        "description": "Get robustness audit results including overfitting risk, suspicious patterns, and recommendations. Critical for avoiding past mistakes.",
        "parameters": {
            "type": "object",
            "properties": {
                "cycle_number": {
                    "type": "integer",
                    "description": "Research cycle number. If not specified, returns most recent audit."
                },
                "experiment_id": {
                    "type": "string",
                    "description": "Specific experiment ID to get audit for"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_notebook_entries",
        "description": "Get lab notebook entries containing observations, hypotheses, and notes from both humans and agents.",
        "parameters": {
            "type": "object",
            "properties": {
                "cycle_number": {
                    "type": "integer",
                    "description": "Filter by research cycle number"
                },
                "author_type": {
                    "type": "string",
                    "description": "Filter by author: 'human' or 'agent'",
                    "enum": ["human", "agent"]
                },
                "search_query": {
                    "type": "string",
                    "description": "Search for entries containing this text"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum entries to return",
                    "default": 10
                }
            },
            "required": []
        }
    },
    {
        "name": "get_failed_experiments",
        "description": "Get details about failed experiments including error messages and stack traces. Essential for understanding what NOT to repeat.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of failures to return",
                    "default": 5
                },
                "include_error_details": {
                    "type": "boolean",
                    "description": "Include full error messages and logs",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "get_best_models",
        "description": "Get the top performing models across all experiments, ranked by the primary metric.",
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Metric to rank by (e.g., 'accuracy', 'f1_score', 'rmse')"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top models to return",
                    "default": 5
                },
                "include_config": {
                    "type": "boolean",
                    "description": "Include full hyperparameter configuration",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "get_dataset_designs",
        "description": "Get previous dataset designs including feature engineering strategies, preprocessing steps, and their outcomes.",
        "parameters": {
            "type": "object",
            "properties": {
                "cycle_number": {
                    "type": "integer",
                    "description": "Research cycle number"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum designs to return",
                    "default": 5
                }
            },
            "required": []
        }
    },
    {
        "name": "search_project_history",
        "description": "Search across all project history (experiments, notes, agent thinking, audits) for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query - will match against names, descriptions, notes, and thinking"
                },
                "search_scope": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["experiments", "notebook", "agent_thinking", "audits", "errors"]
                    },
                    "description": "Which areas to search in",
                    "default": ["experiments", "notebook", "agent_thinking", "audits", "errors"]
                }
            },
            "required": ["query"]
        }
    }
]


# =============================================================================
# Tool Handlers
# =============================================================================

class AgentToolExecutor:
    """Executes agent history tools against the database."""

    def __init__(self, db: Session, project_id: UUID, current_cycle_id: Optional[UUID] = None):
        self.db = db
        self.project_id = project_id
        self.current_cycle_id = current_cycle_id

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given arguments."""
        handlers = {
            "get_research_cycles": self._get_research_cycles,
            "get_agent_thinking": self._get_agent_thinking,
            "get_experiment_results": self._get_experiment_results,
            "get_robustness_audit": self._get_robustness_audit,
            "get_notebook_entries": self._get_notebook_entries,
            "get_failed_experiments": self._get_failed_experiments,
            "get_best_models": self._get_best_models,
            "get_dataset_designs": self._get_dataset_designs,
            "search_project_history": self._search_project_history,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(**arguments)
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def _get_research_cycles(
        self,
        limit: int = 10,
        include_experiments: bool = True
    ) -> Dict[str, Any]:
        """Get research cycles for the project."""
        cycles = (
            self.db.query(ResearchCycle)
            .filter(ResearchCycle.project_id == self.project_id)
            .order_by(desc(ResearchCycle.sequence_number))
            .limit(limit)
            .all()
        )

        result = []
        for cycle in cycles:
            # Handle status as enum or string
            status_value = cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
            cycle_data = {
                "id": str(cycle.id),
                "sequence_number": cycle.sequence_number,
                "status": status_value,
                "summary_title": cycle.summary_title,
                "created_at": cycle.created_at.isoformat() if cycle.created_at else None,
                "is_current": str(cycle.id) == str(self.current_cycle_id) if self.current_cycle_id else False,
            }

            if include_experiments:
                # Experiments are linked via CycleExperiment join table
                experiments = (
                    self.db.query(Experiment)
                    .join(CycleExperiment, Experiment.id == CycleExperiment.experiment_id)
                    .filter(CycleExperiment.research_cycle_id == cycle.id)
                    .all()
                )
                cycle_data["experiments"] = [
                    {
                        "id": str(exp.id),
                        "name": exp.name,
                        "status": exp.status,
                        "best_metric": exp.best_metric,
                        "primary_metric": exp.primary_metric,
                    }
                    for exp in experiments
                ]

            result.append(cycle_data)

        return {
            "cycles": result,
            "total_count": len(result),
            "note": "Cycles ordered from most recent to oldest"
        }

    def _get_agent_thinking(
        self,
        step_type: str,
        cycle_number: Optional[int] = None,
        include_logs: bool = True
    ) -> Dict[str, Any]:
        """Get agent thinking/reasoning from a specific step type."""
        # Find the research cycle
        cycle_query = self.db.query(ResearchCycle).filter(
            ResearchCycle.project_id == self.project_id
        )

        if cycle_number:
            cycle = cycle_query.filter(ResearchCycle.sequence_number == cycle_number).first()
        else:
            # Get most recent completed cycle (not current)
            cycle = (
                cycle_query
                .filter(ResearchCycle.id != self.current_cycle_id)
                .order_by(desc(ResearchCycle.sequence_number))
                .first()
            )

        if not cycle:
            return {"error": "No matching research cycle found", "step_type": step_type}

        # Find agent run for this cycle
        agent_run = (
            self.db.query(AgentRun)
            .filter(AgentRun.research_cycle_id == cycle.id)
            .first()
        )

        if not agent_run:
            return {"error": f"No agent run found for cycle {cycle.sequence_number}"}

        # Find the step
        step = (
            self.db.query(AgentStep)
            .filter(
                AgentStep.agent_run_id == agent_run.id,
                AgentStep.step_type == step_type
            )
            .first()
        )

        if not step:
            return {
                "error": f"No {step_type} step found in cycle {cycle.sequence_number}",
                "available_steps": [s.step_type for s in agent_run.steps] if agent_run.steps else []
            }

        result = {
            "step_type": step_type,
            "cycle_number": cycle.sequence_number,
            "cycle_title": cycle.summary_title,
            "status": step.status,
            "started_at": step.started_at.isoformat() if step.started_at else None,
            "completed_at": step.completed_at.isoformat() if step.completed_at else None,
            "input_data": step.input_data,
            "output_data": step.output_data,
        }

        if include_logs:
            logs = (
                self.db.query(AgentStepLog)
                .filter(AgentStepLog.agent_step_id == step.id)
                .order_by(AgentStepLog.created_at)
                .all()
            )
            result["thinking_logs"] = [
                {
                    "type": log.log_type,
                    "message": log.message,
                    "metadata": log.metadata_json,
                    "timestamp": log.created_at.isoformat() if log.created_at else None,
                }
                for log in logs
            ]

        return result

    def _get_experiment_results(
        self,
        experiment_id: Optional[str] = None,
        cycle_number: Optional[int] = None,
        include_trials: bool = True,
        top_n_trials: int = 5
    ) -> Dict[str, Any]:
        """Get experiment results with optional trial details."""
        query = self.db.query(Experiment).join(ResearchCycle).filter(
            ResearchCycle.project_id == self.project_id
        )

        if experiment_id:
            query = query.filter(Experiment.id == experiment_id)
        elif cycle_number:
            query = query.filter(ResearchCycle.sequence_number == cycle_number)

        experiments = query.order_by(desc(Experiment.created_at)).limit(20).all()

        results = []
        for exp in experiments:
            exp_data = {
                "id": str(exp.id),
                "name": exp.name,
                "status": exp.status,
                "best_metric": exp.best_metric,
                "primary_metric": exp.primary_metric,
                "config": exp.config,
                "cycle_number": exp.research_cycle.sequence_number if exp.research_cycle else None,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            }

            if include_trials:
                trials = (
                    self.db.query(Trial)
                    .filter(Trial.experiment_id == exp.id)
                    .order_by(desc(Trial.metric_value))
                    .limit(top_n_trials)
                    .all()
                )
                exp_data["top_trials"] = [
                    {
                        "id": str(trial.id),
                        "trial_number": trial.trial_number,
                        "hyperparameters": trial.hyperparameters,
                        "metric_name": trial.metric_name,
                        "metric_value": trial.metric_value,
                        "train_score": trial.train_score,
                        "val_score": trial.val_score,
                        "status": trial.status,
                    }
                    for trial in trials
                ]

            results.append(exp_data)

        return {
            "experiments": results,
            "count": len(results)
        }

    def _get_robustness_audit(
        self,
        cycle_number: Optional[int] = None,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get robustness audit results."""
        # Find robustness audit steps
        query = (
            self.db.query(AgentStep)
            .join(AgentRun)
            .filter(
                AgentRun.project_id == self.project_id,
                AgentStep.step_type == AgentStepType.ROBUSTNESS_AUDIT.value
            )
        )

        if cycle_number:
            query = query.join(ResearchCycle).filter(
                ResearchCycle.sequence_number == cycle_number
            )

        audit_step = query.order_by(desc(AgentStep.created_at)).first()

        if not audit_step:
            return {"error": "No robustness audit found", "suggestion": "Run the robustness audit first"}

        output = audit_step.output_data or {}

        return {
            "audit_id": str(audit_step.id),
            "cycle_number": cycle_number,
            "overfitting_risk": output.get("overfitting_risk"),
            "risk_level": output.get("risk_level"),
            "suspicious_patterns": output.get("suspicious_patterns", []),
            "train_val_analysis": output.get("train_val_analysis"),
            "baseline_comparison": output.get("baseline_comparison"),
            "recommendations": output.get("recommendations", []),
            "natural_language_summary": output.get("natural_language_summary"),
            "created_at": audit_step.created_at.isoformat() if audit_step.created_at else None,
        }

    def _get_notebook_entries(
        self,
        cycle_number: Optional[int] = None,
        author_type: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get lab notebook entries."""
        query = self.db.query(LabNotebookEntry).filter(
            LabNotebookEntry.project_id == self.project_id
        )

        if cycle_number:
            cycle = (
                self.db.query(ResearchCycle)
                .filter(
                    ResearchCycle.project_id == self.project_id,
                    ResearchCycle.sequence_number == cycle_number
                )
                .first()
            )
            if cycle:
                query = query.filter(LabNotebookEntry.research_cycle_id == cycle.id)

        if author_type:
            query = query.filter(LabNotebookEntry.author_type == author_type)

        if search_query:
            search_pattern = f"%{search_query}%"
            query = query.filter(
                (LabNotebookEntry.title.ilike(search_pattern)) |
                (LabNotebookEntry.body_markdown.ilike(search_pattern))
            )

        entries = query.order_by(desc(LabNotebookEntry.created_at)).limit(limit).all()

        return {
            "entries": [
                {
                    "id": str(entry.id),
                    "title": entry.title,
                    "body": entry.body_markdown,
                    "author_type": entry.author_type,
                    "cycle_id": str(entry.research_cycle_id) if entry.research_cycle_id else None,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                }
                for entry in entries
            ],
            "count": len(entries)
        }

    def _get_failed_experiments(
        self,
        limit: int = 5,
        include_error_details: bool = True
    ) -> Dict[str, Any]:
        """Get failed experiments and their errors."""
        # Get failed experiments
        failed_experiments = (
            self.db.query(Experiment)
            .join(ResearchCycle)
            .filter(
                ResearchCycle.project_id == self.project_id,
                Experiment.status == "failed"
            )
            .order_by(desc(Experiment.created_at))
            .limit(limit)
            .all()
        )

        # Also get failed agent steps
        failed_steps = (
            self.db.query(AgentStep)
            .join(AgentRun)
            .filter(
                AgentRun.project_id == self.project_id,
                AgentStep.status == "failed"
            )
            .order_by(desc(AgentStep.created_at))
            .limit(limit)
            .all()
        )

        result = {
            "failed_experiments": [],
            "failed_agent_steps": []
        }

        for exp in failed_experiments:
            exp_data = {
                "id": str(exp.id),
                "name": exp.name,
                "cycle_number": exp.research_cycle.sequence_number if exp.research_cycle else None,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            }
            if include_error_details:
                exp_data["error_message"] = exp.error_message if hasattr(exp, 'error_message') else None
                exp_data["config"] = exp.config
            result["failed_experiments"].append(exp_data)

        for step in failed_steps:
            step_data = {
                "id": str(step.id),
                "step_type": step.step_type,
                "created_at": step.created_at.isoformat() if step.created_at else None,
            }
            if include_error_details:
                # Get error logs
                error_logs = (
                    self.db.query(AgentStepLog)
                    .filter(
                        AgentStepLog.agent_step_id == step.id,
                        AgentStepLog.log_type == "error"
                    )
                    .all()
                )
                step_data["error_logs"] = [
                    {"message": log.message, "metadata": log.metadata_json}
                    for log in error_logs
                ]
                step_data["output_data"] = step.output_data
            result["failed_agent_steps"].append(step_data)

        return result

    def _get_best_models(
        self,
        metric: Optional[str] = None,
        top_n: int = 5,
        include_config: bool = True
    ) -> Dict[str, Any]:
        """Get top performing models."""
        query = (
            self.db.query(Trial)
            .join(Experiment)
            .join(ResearchCycle)
            .filter(
                ResearchCycle.project_id == self.project_id,
                Trial.status == "completed"
            )
        )

        if metric:
            query = query.filter(Trial.metric_name == metric)

        # Order by metric value (higher is better for most metrics)
        trials = query.order_by(desc(Trial.metric_value)).limit(top_n).all()

        return {
            "best_models": [
                {
                    "trial_id": str(trial.id),
                    "experiment_name": trial.experiment.name if trial.experiment else None,
                    "cycle_number": trial.experiment.research_cycle.sequence_number if trial.experiment and trial.experiment.research_cycle else None,
                    "metric_name": trial.metric_name,
                    "metric_value": trial.metric_value,
                    "train_score": trial.train_score,
                    "val_score": trial.val_score,
                    "hyperparameters": trial.hyperparameters if include_config else None,
                }
                for trial in trials
            ],
            "count": len(trials)
        }

    def _get_dataset_designs(
        self,
        cycle_number: Optional[int] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Get previous dataset designs."""
        # Get dataset design steps
        query = (
            self.db.query(AgentStep)
            .join(AgentRun)
            .filter(
                AgentRun.project_id == self.project_id,
                AgentStep.step_type == AgentStepType.DATASET_DESIGN.value
            )
        )

        if cycle_number:
            query = query.join(ResearchCycle).filter(
                ResearchCycle.sequence_number == cycle_number
            )

        steps = query.order_by(desc(AgentStep.created_at)).limit(limit).all()

        # Also get DatasetSpec records
        specs = (
            self.db.query(DatasetSpec)
            .filter(DatasetSpec.project_id == self.project_id)
            .order_by(desc(DatasetSpec.created_at))
            .limit(limit)
            .all()
        )

        return {
            "dataset_design_steps": [
                {
                    "id": str(step.id),
                    "output_data": step.output_data,
                    "created_at": step.created_at.isoformat() if step.created_at else None,
                }
                for step in steps
            ],
            "dataset_specs": [
                {
                    "id": str(spec.id),
                    "name": spec.name if hasattr(spec, 'name') else None,
                    "config": spec.config if hasattr(spec, 'config') else None,
                    "created_at": spec.created_at.isoformat() if spec.created_at else None,
                }
                for spec in specs
            ]
        }

    def _search_project_history(
        self,
        query: str,
        search_scope: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search across project history."""
        if search_scope is None:
            search_scope = ["experiments", "notebook", "agent_thinking", "audits", "errors"]

        results = {
            "query": query,
            "matches": []
        }
        search_pattern = f"%{query}%"

        # Search experiments
        if "experiments" in search_scope:
            experiments = (
                self.db.query(Experiment)
                .join(ResearchCycle)
                .filter(
                    ResearchCycle.project_id == self.project_id,
                    Experiment.name.ilike(search_pattern)
                )
                .limit(5)
                .all()
            )
            for exp in experiments:
                results["matches"].append({
                    "type": "experiment",
                    "id": str(exp.id),
                    "name": exp.name,
                    "preview": f"Experiment with best metric: {exp.best_metric}"
                })

        # Search notebook entries
        if "notebook" in search_scope:
            entries = (
                self.db.query(LabNotebookEntry)
                .filter(
                    LabNotebookEntry.project_id == self.project_id,
                    (LabNotebookEntry.title.ilike(search_pattern)) |
                    (LabNotebookEntry.body_markdown.ilike(search_pattern))
                )
                .limit(5)
                .all()
            )
            for entry in entries:
                results["matches"].append({
                    "type": "notebook_entry",
                    "id": str(entry.id),
                    "title": entry.title,
                    "preview": entry.body_markdown[:200] if entry.body_markdown else None
                })

        # Search agent thinking logs
        if "agent_thinking" in search_scope:
            logs = (
                self.db.query(AgentStepLog)
                .join(AgentStep)
                .join(AgentRun)
                .filter(
                    AgentRun.project_id == self.project_id,
                    AgentStepLog.message.ilike(search_pattern)
                )
                .limit(5)
                .all()
            )
            for log in logs:
                results["matches"].append({
                    "type": "agent_thinking",
                    "step_id": str(log.agent_step_id),
                    "log_type": log.log_type,
                    "preview": log.message[:200] if log.message else None
                })

        return results


def get_tools_prompt_section() -> str:
    """Generate the tools section for agent prompts."""
    tools_desc = """
## Available Tools for Querying Project History

You have access to the following tools to query project history and make informed decisions:

1. **get_research_cycles** - Get list of all research cycles with their experiments
2. **get_agent_thinking** - Get the reasoning/logic from previous agent steps (critical for understanding WHY decisions were made)
3. **get_experiment_results** - Get detailed results including trials and hyperparameters
4. **get_robustness_audit** - Get overfitting analysis, suspicious patterns, and recommendations
5. **get_notebook_entries** - Get lab notebook entries from humans and agents
6. **get_failed_experiments** - Get details about what failed and why (essential to avoid repeating mistakes)
7. **get_best_models** - Get top performing models and their configurations
8. **get_dataset_designs** - Get previous feature engineering strategies
9. **search_project_history** - Search across all history for specific topics

**IMPORTANT**: Before designing anything new, you MUST:
1. Call `get_research_cycles` to understand overall project progression
2. Call `get_agent_thinking` for the same step type in previous cycles to see what was tried
3. Call `get_robustness_audit` to check for overfitting patterns to avoid
4. Call `get_failed_experiments` to ensure you don't repeat past mistakes
5. Call `get_notebook_entries` for any human insights or corrections

Build upon what worked, avoid what didn't, and address any robustness concerns identified.
"""
    return tools_desc

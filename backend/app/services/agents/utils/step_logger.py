"""Step logger for agent execution progress tracking."""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import AgentStepLog, LogMessageType


class StepLogger:
    """Helper class to append logs to an agent step.

    Provides typed logging methods for different kinds of agent output:
    - info/warning/error: Standard log levels
    - thinking: Internal reasoning, step-by-step analysis
    - hypothesis: Candidate explanations or theories
    - action: Specific actions the agent is taking
    - summary: Final narrative summary

    Example:
        logger = StepLogger(db, step_id)
        logger.thinking("Analyzing 50 columns...")
        logger.action("Excluding high-cardinality columns")
        logger.warning("Target imbalance detected: 95/5 split")
        logger.summary("Analysis complete. Found 3 issues.")
    """

    def __init__(self, db: Session, step_id: Optional[UUID] = None, agent_run_id: Optional[UUID] = None):
        """Initialize the step logger.

        Args:
            db: Database session for persisting logs
            step_id: UUID of the agent step to log to (optional - if None, logs won't be persisted)
            agent_run_id: UUID of the agent run (optional - for context when step_id is None)
        """
        self.db = db
        self.step_id = step_id
        self.agent_run_id = agent_run_id
        self._sequence = 0

    def _get_next_sequence(self) -> int:
        """Get the next sequence number for this step."""
        # Get max sequence from existing logs
        max_seq = (
            self.db.query(AgentStepLog)
            .filter(AgentStepLog.agent_step_id == self.step_id)
            .count()
        )
        self._sequence = max_seq + 1
        return self._sequence

    def log(
        self,
        message: str,
        message_type: LogMessageType = LogMessageType.INFO,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentStepLog]:
        """Append a log entry to the step.

        Args:
            message: The log message content
            message_type: Type of log message
            metadata: Optional structured metadata

        Returns:
            The created AgentStepLog entry, or None if step_id is not set
        """
        # If no step_id, we can't persist logs - just return None
        if self.step_id is None:
            return None

        log_entry = AgentStepLog(
            agent_step_id=self.step_id,
            sequence=self._get_next_sequence(),
            timestamp=datetime.utcnow(),
            message_type=message_type,
            message=message,
            metadata_json=metadata,
        )
        self.db.add(log_entry)
        self.db.commit()
        return log_entry

    def info(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[AgentStepLog]:
        """Log an info message."""
        return self.log(message, LogMessageType.INFO, metadata)

    def warning(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[AgentStepLog]:
        """Log a warning message."""
        return self.log(message, LogMessageType.WARNING, metadata)

    def error(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[AgentStepLog]:
        """Log an error message."""
        return self.log(message, LogMessageType.ERROR, metadata)

    def thought(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[AgentStepLog]:
        """Log a thought/reasoning message (deprecated, use thinking())."""
        return self.log(message, LogMessageType.THOUGHT, metadata)

    def summary(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[AgentStepLog]:
        """Log a summary message - final narrative summary for the step."""
        return self.log(message, LogMessageType.SUMMARY, metadata)

    def thinking(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[AgentStepLog]:
        """Log a thinking message - internal reasoning, step-by-step analysis."""
        return self.log(message, LogMessageType.THINKING, metadata)

    def hypothesis(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[AgentStepLog]:
        """Log a hypothesis message - candidate explanations or theories."""
        return self.log(message, LogMessageType.HYPOTHESIS, metadata)

    def action(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[AgentStepLog]:
        """Log an action message - specific actions/commands the agent is taking."""
        return self.log(message, LogMessageType.ACTION, metadata)

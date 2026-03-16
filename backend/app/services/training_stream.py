"""Real-time training log streaming with AI interpretation.

This service provides:
1. Redis pub/sub for streaming logs from Celery workers to WebSocket clients
2. AI interpretation of training logs for non-ML engineers
3. Progress tracking and milestone detection
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Callable

import redis.asyncio as aioredis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Redis channel pattern for training logs
TRAINING_LOG_CHANNEL = "training:logs:{experiment_id}"


@dataclass
class TrainingLogEntry:
    """A single training log entry with optional AI interpretation."""
    timestamp: str
    raw_log: str
    interpreted: str | None = None
    log_type: str = "info"  # info, progress, warning, error, milestone
    progress_percent: int | None = None
    model_name: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "raw_log": self.raw_log,
            "interpreted": self.interpreted,
            "log_type": self.log_type,
            "progress_percent": self.progress_percent,
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
        }


class LogInterpreter:
    """Interprets AutoGluon training logs for non-ML engineers."""

    # Patterns for common AutoGluon log messages
    PATTERNS = [
        # Model training start
        (r"Fitting model: (\w+)", "model_start",
         lambda m: f"🚀 Now training the {m.group(1)} model..."),

        # Model training complete with score
        (r"(\w+) \s+(\-?[\d.]+)\s+(\d+\.\d+)s", "model_complete",
         lambda m: f"✅ {m.group(1)} finished! Score: {m.group(2)} (took {m.group(3)}s)"),

        # Validation score
        (r"score_val\s*[:=]\s*(\-?[\d.]+)", "score",
         lambda m: f"📊 Validation score: {m.group(1)}"),

        # Time remaining
        (r"Time remaining[:\s]+(\d+\.?\d*)s", "time_remaining",
         lambda m: f"⏱️ About {int(float(m.group(1)))} seconds remaining"),

        # Models to train
        (r"Fitting (\d+) models", "training_start",
         lambda m: f"🎯 Starting to train {m.group(1)} different models to find the best one"),

        # Bagging/ensemble
        (r"Fitting model: WeightedEnsemble", "ensemble",
         lambda m: "🔗 Creating an ensemble by combining the best models together"),

        # Feature importance
        (r"Feature Importance", "feature_importance",
         lambda m: "📈 Calculating which features are most important for predictions"),

        # Data preprocessing
        (r"Preprocessing|preprocessing", "preprocessing",
         lambda m: "🔧 Preparing and cleaning the data for training"),

        # Cross-validation
        (r"num_bag_folds=(\d+)", "cross_validation",
         lambda m: f"🔄 Using {m.group(1)}-fold cross-validation for reliable evaluation"),

        # Best model
        (r"Best model: (\w+)", "best_model",
         lambda m: f"🏆 Best model so far: {m.group(1)}"),

        # AutoGluon complete
        (r"AutoGluon training complete", "complete",
         lambda m: "🎉 Training complete! Finalizing results..."),

        # Time limit warning
        (r"Time limit reached|time budget", "time_limit",
         lambda m: "⚠️ Running low on time, wrapping up training"),

        # Skipping model
        (r"Skipping (\w+) due to", "skip",
         lambda m: f"⏭️ Skipping {m.group(1)} (not enough time or resources)"),

        # Memory warning
        (r"memory|Memory", "memory",
         lambda m: "💾 Managing memory usage..."),

        # GPU usage
        (r"GPU|cuda|CUDA", "gpu",
         lambda m: "🖥️ Using GPU for faster training"),
    ]

    # Progress milestones (rough estimates based on typical training flow)
    MILESTONES = {
        "preprocessing": 5,
        "training_start": 10,
        "model_start": None,  # Calculated dynamically
        "model_complete": None,
        "ensemble": 85,
        "feature_importance": 90,
        "complete": 100,
    }

    def __init__(self):
        self.models_expected = 0
        self.models_completed = 0
        self.best_score = None
        self.best_model = None

    def interpret(self, raw_log: str) -> TrainingLogEntry:
        """Interpret a raw log line and return a structured entry."""
        timestamp = datetime.now().isoformat()
        entry = TrainingLogEntry(
            timestamp=timestamp,
            raw_log=raw_log.strip(),
        )

        # Check for model count
        models_match = re.search(r"Fitting (\d+) L\d+ models", raw_log)
        if models_match:
            self.models_expected += int(models_match.group(1))

        # Try to match patterns
        for pattern, log_type, interpreter in self.PATTERNS:
            match = re.search(pattern, raw_log, re.IGNORECASE)
            if match:
                entry.log_type = log_type
                entry.interpreted = interpreter(match)

                # Track progress
                if log_type == "model_complete":
                    self.models_completed += 1
                    entry.model_name = match.group(1)
                    try:
                        entry.metric_value = float(match.group(2))
                        if self.best_score is None or entry.metric_value > self.best_score:
                            self.best_score = entry.metric_value
                            self.best_model = entry.model_name
                    except (ValueError, IndexError):
                        pass

                elif log_type == "model_start":
                    entry.model_name = match.group(1)

                elif log_type == "score":
                    try:
                        entry.metric_value = float(match.group(1))
                    except ValueError:
                        pass

                # Calculate progress
                entry.progress_percent = self._calculate_progress(log_type)
                break

        # Default interpretation for unmatched lines
        if entry.interpreted is None and raw_log.strip():
            # Check for common patterns without custom interpreters
            if "error" in raw_log.lower():
                entry.log_type = "error"
                entry.interpreted = f"⚠️ Error: {raw_log[:100]}..."
            elif "warning" in raw_log.lower():
                entry.log_type = "warning"
                entry.interpreted = f"⚠️ Warning detected in training"
            elif "epoch" in raw_log.lower():
                entry.log_type = "progress"
                entry.interpreted = "📈 Training neural network..."
            else:
                # Keep as raw technical log
                entry.log_type = "info"
                entry.interpreted = None  # Will show raw log on frontend

        return entry

    def _calculate_progress(self, log_type: str) -> int | None:
        """Calculate approximate training progress percentage."""
        if log_type in self.MILESTONES:
            milestone = self.MILESTONES[log_type]
            if milestone is not None:
                return milestone

        # Dynamic progress based on models
        if self.models_expected > 0 and self.models_completed > 0:
            # Models are trained between 10% and 85%
            model_progress = (self.models_completed / self.models_expected) * 75
            return min(85, int(10 + model_progress))

        return None

    def get_summary(self) -> dict:
        """Get current training summary."""
        return {
            "models_expected": self.models_expected,
            "models_completed": self.models_completed,
            "best_score": self.best_score,
            "best_model": self.best_model,
            "progress_percent": self._calculate_progress("model_complete") or 0,
        }


class TrainingLogPublisher:
    """Publishes training logs to Redis for real-time streaming."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.channel = TRAINING_LOG_CHANNEL.format(experiment_id=experiment_id)
        self.interpreter = LogInterpreter()
        self._redis = None

    def _get_sync_redis(self):
        """Get synchronous Redis client for Celery workers."""
        import redis
        settings = get_settings()
        return redis.from_url(settings.redis_url)

    def publish_log(self, raw_log: str):
        """Publish a log line to Redis (synchronous for Celery)."""
        try:
            entry = self.interpreter.interpret(raw_log)

            # Only publish meaningful entries
            if entry.raw_log.strip():
                redis_client = self._get_sync_redis()
                message = json.dumps(entry.to_dict())
                redis_client.publish(self.channel, message)
                redis_client.close()
        except Exception as e:
            logger.warning(f"Failed to publish training log: {e}")

    def publish_milestone(self, milestone: str, details: dict | None = None):
        """Publish a training milestone event."""
        entry = TrainingLogEntry(
            timestamp=datetime.now().isoformat(),
            raw_log=milestone,
            interpreted=milestone,
            log_type="milestone",
            progress_percent=self.interpreter.MILESTONES.get(milestone.lower().replace(" ", "_")),
        )
        if details:
            if "model_name" in details:
                entry.model_name = details["model_name"]
            if "metric_value" in details:
                entry.metric_value = details["metric_value"]

        try:
            redis_client = self._get_sync_redis()
            message = json.dumps(entry.to_dict())
            redis_client.publish(self.channel, message)
            redis_client.close()
        except Exception as e:
            logger.warning(f"Failed to publish milestone: {e}")

    def publish_summary(self):
        """Publish training summary."""
        summary = self.interpreter.get_summary()
        entry = TrainingLogEntry(
            timestamp=datetime.now().isoformat(),
            raw_log="Training Summary",
            interpreted=f"Trained {summary['models_completed']} models. Best: {summary['best_model']} ({summary['best_score']:.4f})" if summary['best_model'] else "Training in progress...",
            log_type="milestone",
            progress_percent=summary["progress_percent"],
            model_name=summary["best_model"],
            metric_value=summary["best_score"],
        )
        try:
            redis_client = self._get_sync_redis()
            message = json.dumps(entry.to_dict())
            redis_client.publish(self.channel, message)
            redis_client.close()
        except Exception as e:
            logger.warning(f"Failed to publish summary: {e}")


class TrainingLogSubscriber:
    """Subscribes to training logs from Redis for WebSocket streaming."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.channel = TRAINING_LOG_CHANNEL.format(experiment_id=experiment_id)
        self._redis: aioredis.Redis | None = None
        self._pubsub = None

    async def connect(self):
        """Connect to Redis."""
        settings = get_settings()
        self._redis = aioredis.from_url(settings.redis_url)
        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(self.channel)

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._pubsub:
            await self._pubsub.unsubscribe(self.channel)
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()

    async def stream_logs(self) -> AsyncGenerator[dict, None]:
        """Stream log entries as they arrive."""
        if not self._pubsub:
            await self.connect()

        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        yield data
                    except json.JSONDecodeError:
                        continue
        finally:
            await self.disconnect()


class StreamingLogCapture:
    """Captures stdout/stderr and streams to Redis during training.

    Use this in place of the LogCapture class in modal_training_standalone.py
    to get real-time streaming.
    """

    def __init__(self, original_stream, experiment_id: str):
        self.original = original_stream
        self.experiment_id = experiment_id
        self.publisher = TrainingLogPublisher(experiment_id)
        self.captured = []
        self._buffer = ""

    def write(self, text):
        self.original.write(text)
        self._buffer += text

        # Process complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.captured.append(line)
                self.publisher.publish_log(line)

    def flush(self):
        self.original.flush()
        # Flush any remaining buffer
        if self._buffer.strip():
            self.captured.append(self._buffer)
            self.publisher.publish_log(self._buffer)
            self._buffer = ""

    def get_logs(self, max_lines=500):
        """Get captured logs, truncated if needed."""
        lines = self.captured
        if len(lines) > max_lines:
            lines = lines[:100] + ["... (truncated) ..."] + lines[-400:]
        return "\n".join(lines)


class StreamingLogHandler(logging.Handler):
    """Logging handler that streams to Redis for real-time updates."""

    def __init__(self, experiment_id: str):
        super().__init__()
        self.experiment_id = experiment_id
        self.publisher = TrainingLogPublisher(experiment_id)
        self.captured = []

    def emit(self, record):
        try:
            msg = self.format(record)
            self.captured.append(msg)
            self.publisher.publish_log(msg)
        except Exception:
            self.handleError(record)

    def get_logs(self, max_lines=500):
        """Get captured logs, truncated if needed."""
        lines = self.captured
        if len(lines) > max_lines:
            lines = lines[:100] + ["... (truncated) ..."] + lines[-400:]
        return "\n".join(lines)


def setup_streaming_capture(experiment_id: str):
    """Set up comprehensive log capture for an experiment.

    Returns a context manager that captures stdout, stderr, and logging.

    Usage:
        with setup_streaming_capture(experiment_id) as capture:
            # Training code here
            pass
        logs = capture.get_logs()
    """
    import sys

    class StreamingContext:
        def __init__(self, experiment_id: str):
            self.experiment_id = experiment_id
            self.stdout_capture = None
            self.stderr_capture = None
            self.log_handler = None
            self.original_stdout = None
            self.original_stderr = None
            self._all_logs = []

        def __enter__(self):
            # Capture stdout
            self.original_stdout = sys.stdout
            self.stdout_capture = StreamingLogCapture(sys.stdout, self.experiment_id)
            sys.stdout = self.stdout_capture

            # Capture stderr
            self.original_stderr = sys.stderr
            self.stderr_capture = StreamingLogCapture(sys.stderr, self.experiment_id)
            sys.stderr = self.stderr_capture

            # Add logging handler to root logger and autogluon loggers
            self.log_handler = StreamingLogHandler(self.experiment_id)
            self.log_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            self.log_handler.setFormatter(formatter)

            # Add to root logger
            logging.getLogger().addHandler(self.log_handler)

            # Add to AutoGluon loggers specifically
            for logger_name in ['autogluon', 'autogluon.core', 'autogluon.tabular']:
                ag_logger = logging.getLogger(logger_name)
                ag_logger.addHandler(self.log_handler)

            # Send initial message
            publisher = TrainingLogPublisher(self.experiment_id)
            publisher.publish_milestone("Training started", {})

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

            # Remove logging handler
            if self.log_handler:
                logging.getLogger().removeHandler(self.log_handler)
                for logger_name in ['autogluon', 'autogluon.core', 'autogluon.tabular']:
                    logging.getLogger(logger_name).removeHandler(self.log_handler)

            # Send completion message
            publisher = TrainingLogPublisher(self.experiment_id)
            if exc_type is None:
                publisher.publish_milestone("Training completed", {})
            else:
                publisher.publish_milestone(f"Training failed: {exc_val}", {})

            return False

        def get_logs(self, max_lines=500):
            """Get all captured logs."""
            all_logs = []
            if self.stdout_capture:
                all_logs.extend(self.stdout_capture.captured)
            if self.stderr_capture:
                all_logs.extend(self.stderr_capture.captured)
            if self.log_handler:
                all_logs.extend(self.log_handler.captured)

            # Deduplicate while preserving order
            seen = set()
            unique_logs = []
            for log in all_logs:
                if log not in seen:
                    seen.add(log)
                    unique_logs.append(log)

            if len(unique_logs) > max_lines:
                unique_logs = unique_logs[:100] + ["... (truncated) ..."] + unique_logs[-400:]

            return "\n".join(unique_logs)

    return StreamingContext(experiment_id)

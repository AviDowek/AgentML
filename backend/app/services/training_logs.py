"""Training log service using Redis for storage and retrieval.

This module provides a simple, reliable way to stream training logs:
1. Worker pushes logs to a Redis list (when Redis is available)
2. Frontend polls an API endpoint to get new logs
3. Logs are interpreted by AI for non-ML engineers

When Redis is not available (e.g., Modal-only deployment), log storage
operations gracefully degrade to no-ops and retrieval returns empty lists.
"""
import asyncio
import json
import logging
import re
import threading
from datetime import datetime
from typing import List, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Redis key patterns
TRAINING_LOGS_KEY = "training:logs:{experiment_id}"
TRAINING_LOG_INDEX_KEY = "training:log_index:{experiment_id}"
AI_INTERPRETATION_KEY = "training:ai_interp:{experiment_id}"

# AI interpretation prompt for batched logs
AI_LOG_INTERPRETATION_PROMPT = """You are an AI assistant helping non-technical users understand machine learning training logs in real-time.

Your job is to translate technical AutoGluon logs into friendly, educational explanations. Imagine you're explaining to someone who has never done ML before.

## Model Explanations (use these when a model starts training):
- **LightGBM / LightGBMXT**: A fast gradient boosting model. Think of it like a team of simple decision-makers that learn from each other's mistakes. Very efficient and often gives great results.
- **XGBoost (XGB)**: Another powerful gradient boosting model, similar to LightGBM. Known for winning many data science competitions.
- **CatBoost (CAT)**: Gradient boosting that's especially good with categorical data (like colors, categories, labels).
- **RandomForest (RF)**: Builds many decision trees and averages their predictions. Like asking many experts and taking their average opinion. Very reliable.
- **ExtraTrees (XT)**: Similar to RandomForest but makes more random choices, which can sometimes find patterns others miss.
- **NeuralNet / NN_TORCH / FASTAI**: A neural network - inspired by how the brain works. Good at finding complex patterns.
- **WeightedEnsemble**: Combines the best models together, giving more weight to the most accurate ones. Usually the final, best model.

## Score Explanations:
- **ROC AUC (roc_auc)**: Measures how well the model distinguishes between classes. 0.5 = random guessing, 1.0 = perfect. Above 0.7 is decent, above 0.8 is good, above 0.9 is excellent.
- **Accuracy**: Percentage of correct predictions. 0.85 = 85% correct.
- **RMSE/MSE**: Error measures for predictions (lower is better).

## Guidelines:
1. For model starts: Explain what the model is in 1-2 simple sentences
2. For scores: Interpret if it's good/bad/average and what that means for predictions
3. For system info, preprocessing, feature generation: Skip these (return null) - they're technical details
4. For warnings about memory/disk: Simplify to "The system is managing resources efficiently"
5. For hyperparameters/config blocks: Skip (return null)
6. Be encouraging but honest about scores
7. Keep explanations to 1-2 sentences max

Training logs:
{logs}

Respond with a JSON array. For EACH log line, include:
- "index": the log line number (0-based)
- "interpretation": plain English explanation (or null to skip boring/technical lines)
- "log_type": one of "milestone", "model_start", "model_complete", "progress", "warning", "error", "info"

Example:
[
  {{"index": 0, "interpretation": "Starting training with 619,040 data points - that's a good amount of data to learn from!", "log_type": "milestone"}},
  {{"index": 1, "interpretation": null, "log_type": "info"}},
  {{"index": 2, "interpretation": "Now training LightGBM - a fast, efficient model that learns by correcting its own mistakes. Often one of the top performers!", "log_type": "model_start"}},
  {{"index": 3, "interpretation": "LightGBM scored 0.502 (ROC AUC). That's barely better than random guessing (0.5). The model is struggling to find patterns - this might be a hard problem or the features need work.", "log_type": "model_complete"}}
]"""


class AILogInterpreter:
    """Uses AI to interpret training logs for non-ML engineers."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self._redis = None
        self._last_interpreted_index = 0
        self._interpretation_lock = threading.Lock()

    def _get_redis(self):
        if self._redis is None:
            if not REDIS_AVAILABLE:
                return None
            try:
                settings = get_settings()
                self._redis = redis.from_url(settings.redis_url)
                self._redis.ping()
            except Exception:
                self._redis = None
                return None
        return self._redis

    async def interpret_batch(self, logs: List[dict], start_index: int = 0) -> List[dict]:
        """Interpret a batch of logs using AI.

        Args:
            logs: List of log entries with 'raw_log' field
            start_index: Starting index for these logs

        Returns:
            List of logs with AI interpretations added
        """
        if not logs:
            return logs

        try:
            from app.services.llm_client import get_llm_client

            # Format logs for AI - limit to avoid token overflow
            log_lines = []
            for i, log in enumerate(logs[:10]):  # Max 10 logs at a time for reliability
                raw = log.get("raw_log", "").strip()
                if raw:
                    # Truncate long log lines
                    if len(raw) > 150:
                        raw = raw[:150] + "..."
                    log_lines.append(f"{i}: {raw}")

            if not log_lines:
                return logs

            # Simple prompt
            prompt = f"""Interpret these ML training logs for a non-technical user. For each log, provide a 1-sentence plain English explanation. Skip boring technical lines by setting interpretation to null.

Logs:
{chr(10).join(log_lines)}

Return a JSON object with an "interpretations" array containing objects with: index (number), interpretation (string or null), log_type (one of: milestone, model_start, model_complete, progress, warning, error, info)"""

            # Use reasoning_effort="none" for log interpretation - fast, no thinking needed
            # use_app_settings=False ensures log parsing always uses fast model regardless of user preference
            llm_client = get_llm_client(model="gpt-4.1", reasoning_effort="none", use_app_settings=False)

            messages = [
                {"role": "system", "content": "You translate ML training logs into simple explanations for non-technical users."},
                {"role": "user", "content": prompt},
            ]

            # Use chat_json for reliable JSON parsing
            response = await llm_client.chat_json(messages)
            logger.info(f"AI JSON response keys: {list(response.keys()) if isinstance(response, dict) else type(response)}")

            # Extract interpretations from response
            interpretations = response.get("interpretations", [])
            if not interpretations and isinstance(response, list):
                interpretations = response

            if not interpretations:
                logger.warning("No interpretations in AI response")
                return logs

            # Apply interpretations to logs
            interp_map = {item["index"]: item for item in interpretations if isinstance(item, dict) and "index" in item}

            interpreted_count = 0
            for i, log in enumerate(logs):
                if i in interp_map:
                    interp = interp_map[i]
                    if interp.get("interpretation"):
                        log["interpreted"] = interp["interpretation"]
                        interpreted_count += 1
                    if interp.get("log_type"):
                        log["log_type"] = interp["log_type"]

            logger.info(f"Applied {interpreted_count} AI interpretations to logs")
            return logs

        except Exception as e:
            logger.warning(f"AI log interpretation failed: {e}", exc_info=True)
            return logs

    def interpret_sync(self, logs: List[dict], start_index: int = 0) -> List[dict]:
        """Synchronous wrapper for interpret_batch."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.interpret_batch(logs, start_index))


class LogInterpreter:
    """Interprets AutoGluon training logs using regex patterns (fallback)."""

    PATTERNS = [
        # Model training
        (r"Fitting model: (\w+)", "model_start",
         lambda m: f"Now training the {m.group(1)} model..."),
        (r"(\w+)\s+(-?[\d.]+)\s+(\d+\.?\d*)s", "model_complete",
         lambda m: f"{m.group(1)} finished! Score: {m.group(2)} (took {m.group(3)}s)"),

        # Training phases
        (r"Fitting (\d+) L\d+ models", "training_start",
         lambda m: f"Starting to train {m.group(1)} models to find the best one"),
        (r"Fitting model: WeightedEnsemble", "ensemble",
         lambda m: "Creating an ensemble by combining the best models"),
        (r"AutoGluon training complete", "complete",
         lambda m: "Training complete! Finalizing results..."),

        # Scores and metrics
        (r"Validation score[:\s]+(-?[\d.]+)", "score",
         lambda m: f"Validation score: {m.group(1)}"),
        (r"Best model[:\s]+(\w+)", "best_model",
         lambda m: f"Best model so far: {m.group(1)}"),

        # Time and progress
        (r"Time remaining[:\s]+(\d+\.?\d*)s", "time",
         lambda m: f"About {int(float(m.group(1)))} seconds remaining"),
        (r"(\d+)% complete", "progress",
         lambda m: f"{m.group(1)}% complete"),

        # Warnings
        (r"Skipping (\w+)", "skip",
         lambda m: f"Skipping {m.group(1)} (not enough time)"),
        (r"Time limit reached", "time_limit",
         lambda m: "Time limit reached, wrapping up"),
    ]

    @classmethod
    def interpret(cls, raw_log: str) -> dict:
        """Interpret a raw log line."""
        log_type = "info"
        interpreted = None

        for pattern, ltype, interpreter in cls.PATTERNS:
            match = re.search(pattern, raw_log, re.IGNORECASE)
            if match:
                log_type = ltype
                interpreted = interpreter(match)
                break

        # Check for errors/warnings
        if interpreted is None:
            lower = raw_log.lower()
            if "error" in lower:
                log_type = "error"
            elif "warning" in lower:
                log_type = "warning"

        return {
            "timestamp": datetime.now().isoformat(),
            "raw_log": raw_log.strip(),
            "interpreted": interpreted,
            "log_type": log_type,
        }


class TrainingLogStore:
    """Stores and retrieves training logs using Redis."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.logs_key = TRAINING_LOGS_KEY.format(experiment_id=experiment_id)
        self.index_key = TRAINING_LOG_INDEX_KEY.format(experiment_id=experiment_id)
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            if not REDIS_AVAILABLE:
                return None
            try:
                settings = get_settings()
                self._redis = redis.from_url(settings.redis_url)
                self._redis.ping()
            except Exception:
                self._redis = None
                return None
        return self._redis

    def add_log(self, message: str, log_type: str = "info", interpreted: str = None):
        """Add a log entry."""
        try:
            r = self._get_redis()
            if r is None:
                return
            entry = {
                "timestamp": datetime.now().isoformat(),
                "raw_log": message,
                "interpreted": interpreted,
                "log_type": log_type,
            }
            r.rpush(self.logs_key, json.dumps(entry))
            # Set expiry to 1 hour
            r.expire(self.logs_key, 3600)
        except Exception as e:
            logger.warning(f"Failed to add training log: {e}")

    def add_raw_log(self, raw_log: str):
        """Add a raw log and auto-interpret it."""
        if not raw_log.strip():
            return
        try:
            r = self._get_redis()
            if r is None:
                return
            entry = LogInterpreter.interpret(raw_log)
            r.rpush(self.logs_key, json.dumps(entry))
            r.expire(self.logs_key, 3600)
        except Exception as e:
            logger.warning(f"Failed to add training log: {e}")

    def add_milestone(self, message: str):
        """Add a milestone log entry."""
        self.add_log(message, log_type="milestone", interpreted=message)

    def get_logs(self, start_index: int = 0) -> tuple[List[dict], int]:
        """Get logs starting from an index.

        Returns:
            Tuple of (logs_list, next_index)
        """
        try:
            r = self._get_redis()
            if r is None:
                return [], start_index
            logs = r.lrange(self.logs_key, start_index, -1)
            parsed = [json.loads(log) for log in logs]
            next_index = start_index + len(parsed)
            return parsed, next_index
        except Exception as e:
            logger.warning(f"Failed to get training logs: {e}")
            return [], start_index

    def get_all_logs(self) -> List[dict]:
        """Get all logs."""
        logs, _ = self.get_logs(0)
        return logs

    def update_log_interpretation(self, index: int, log: dict):
        """Update a log entry with AI interpretation.

        Args:
            index: The index of the log in the list
            log: The log dict with updated interpretation
        """
        try:
            r = self._get_redis()
            if r is None:
                return
            r.lset(self.logs_key, index, json.dumps(log))
        except Exception as e:
            logger.warning(f"Failed to update log interpretation: {e}")

    def clear(self):
        """Clear logs for this experiment."""
        try:
            r = self._get_redis()
            if r is None:
                return
            r.delete(self.logs_key)
        except Exception as e:
            logger.warning(f"Failed to clear training logs: {e}")

    def close(self):
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            self._redis = None


class TrainingLogCapture:
    """Captures stdout/stderr and stores to Redis.

    Usage:
        log_store = TrainingLogStore(experiment_id)
        capture = TrainingLogCapture(sys.stdout, log_store)
        sys.stdout = capture
        # ... training ...
        sys.stdout = capture.original
    """

    def __init__(self, original_stream, log_store: TrainingLogStore):
        self.original = original_stream
        self.log_store = log_store
        self._buffer = ""
        # Cache the real file descriptor for Ray/faulthandler compatibility
        # Walk up the chain of wrappers to find a real file descriptor
        self._real_fileno = self._find_real_fileno(original_stream)

    def _find_real_fileno(self, stream):
        """Walk up wrapper chain to find a real file descriptor."""
        import sys
        visited = set()
        current = stream
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            # Try to get fileno directly
            if hasattr(current, 'fileno'):
                try:
                    fd = current.fileno()
                    if isinstance(fd, int) and fd >= 0:
                        return fd
                except (AttributeError, OSError, ValueError):
                    pass
            # Try to unwrap to find the real stream
            if hasattr(current, '_stream'):
                current = current._stream
            elif hasattr(current, 'stream'):
                current = current.stream
            elif hasattr(current, '_original'):
                current = current._original
            elif hasattr(current, 'original'):
                current = current.original
            else:
                break
        # Fallback: return stderr's fileno (2) as a safe default
        # This allows Ray to initialize even if we can't find the real fd
        try:
            return sys.__stderr__.fileno()
        except (AttributeError, OSError, ValueError):
            return 2  # stderr fd

    def write(self, text):
        # Always write to original
        self.original.write(text)

        # Buffer and process complete lines
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.log_store.add_raw_log(line)

    def flush(self):
        self.original.flush()
        if self._buffer.strip():
            self.log_store.add_raw_log(self._buffer)
            self._buffer = ""

    def fileno(self):
        """Return a valid file descriptor for Ray's faulthandler.enable().

        This is required because Ray needs a real file descriptor.
        We cache the fd on init by walking up any wrapper chains.
        """
        return self._real_fileno

    def isatty(self):
        """Return whether the original stream is a tty."""
        if hasattr(self.original, 'isatty'):
            try:
                return self.original.isatty()
            except (AttributeError, OSError):
                pass
        return False

    def readable(self):
        """Return whether the stream is readable."""
        return hasattr(self.original, 'readable') and self.original.readable()

    def writable(self):
        """Return whether the stream is writable."""
        return hasattr(self.original, 'writable') and self.original.writable()

    def seekable(self):
        """Return whether the stream is seekable."""
        return False


class TrainingLogContext:
    """Context manager for capturing training logs."""

    def __init__(self, experiment_id: str, clear_logs: bool = False):
        self.experiment_id = experiment_id
        self.log_store = TrainingLogStore(experiment_id)
        self.stdout_capture = None
        self.stderr_capture = None
        self.original_stdout = None
        self.original_stderr = None
        self.clear_logs = clear_logs

    def __enter__(self):
        import sys

        # Only clear logs if explicitly requested
        if self.clear_logs:
            self.log_store.clear()

        # Capture stdout
        self.original_stdout = sys.stdout
        self.stdout_capture = TrainingLogCapture(sys.stdout, self.log_store)
        sys.stdout = self.stdout_capture

        # Capture stderr
        self.original_stderr = sys.stderr
        self.stderr_capture = TrainingLogCapture(sys.stderr, self.log_store)
        sys.stderr = self.stderr_capture

        # Add start message
        self.log_store.add_milestone("Training started")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys

        # Restore streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Add completion message
        if exc_type is None:
            self.log_store.add_milestone("Training completed successfully")
        else:
            self.log_store.add_log(f"Training failed: {exc_val}", log_type="error")

        return False

    def add_log(self, message: str, log_type: str = "info"):
        """Manually add a log entry."""
        self.log_store.add_log(message, log_type=log_type, interpreted=message)

    def add_milestone(self, message: str):
        """Add a milestone."""
        self.log_store.add_milestone(message)

    def get_all_logs_text(self) -> str:
        """Get all logs as text for storage."""
        logs = self.log_store.get_all_logs()
        return "\n".join(log.get("raw_log", "") for log in logs)

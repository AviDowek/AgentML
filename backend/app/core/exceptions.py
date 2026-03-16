"""Custom exceptions for AgentML application.

This module defines specific exception types to replace bare `except Exception`
patterns throughout the codebase. Using specific exceptions improves:
- Debugging: Know exactly what failed
- Error handling: Handle different errors differently
- Logging: Better error categorization
- User experience: More informative error messages
"""

from typing import Optional, Any


class AgentMLError(Exception):
    """Base exception for all AgentML errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


# =============================================================================
# LLM / AI Related Errors
# =============================================================================

class LLMError(AgentMLError):
    """Base error for LLM-related issues."""
    pass


class LLMTimeoutError(LLMError):
    """LLM API call timed out."""

    def __init__(self, timeout_seconds: float, provider: str = "unknown"):
        super().__init__(
            f"LLM API call timed out after {timeout_seconds} seconds",
            {"timeout_seconds": timeout_seconds, "provider": provider}
        )


class LLMRateLimitError(LLMError):
    """LLM API rate limit exceeded."""

    def __init__(self, provider: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f". Retry after {retry_after}s"
        super().__init__(message, {"provider": provider, "retry_after": retry_after})


class LLMAuthenticationError(LLMError):
    """LLM API authentication failed (invalid API key)."""

    def __init__(self, provider: str):
        super().__init__(
            f"Authentication failed for {provider}. Check API key.",
            {"provider": provider}
        )


class LLMResponseError(LLMError):
    """LLM returned invalid or unexpected response."""

    def __init__(self, message: str, raw_response: Optional[str] = None):
        super().__init__(message, {"raw_response": raw_response[:500] if raw_response else None})


class LLMParsingError(LLMError):
    """Failed to parse LLM response as expected format (JSON, etc.)."""

    def __init__(self, expected_format: str, raw_response: Optional[str] = None):
        super().__init__(
            f"Failed to parse LLM response as {expected_format}",
            {"expected_format": expected_format, "raw_response": raw_response[:500] if raw_response else None}
        )


class ToolExecutionError(LLMError):
    """Error executing a tool called by the LLM agent."""

    def __init__(self, tool_name: str, error_message: str):
        super().__init__(
            f"Tool '{tool_name}' execution failed: {error_message}",
            {"tool_name": tool_name, "error": error_message}
        )


# =============================================================================
# Data / Dataset Errors
# =============================================================================

class DataError(AgentMLError):
    """Base error for data-related issues."""
    pass


class DataSourceError(DataError):
    """Error loading or accessing a data source."""

    def __init__(self, data_source_id: str, reason: str):
        super().__init__(
            f"Data source error: {reason}",
            {"data_source_id": data_source_id, "reason": reason}
        )


class DatasetBuildError(DataError):
    """Error building a dataset from spec."""

    def __init__(self, spec_id: str, reason: str):
        super().__init__(
            f"Failed to build dataset: {reason}",
            {"spec_id": spec_id, "reason": reason}
        )


class DataValidationError(DataError):
    """Data validation failed (missing columns, wrong types, etc.)."""

    def __init__(self, field: str, reason: str, value: Any = None):
        super().__init__(
            f"Validation failed for '{field}': {reason}",
            {"field": field, "reason": reason, "value": str(value)[:100] if value else None}
        )


class FeatureEngineeringError(DataError):
    """Error during feature engineering."""

    def __init__(self, feature_name: str, reason: str):
        super().__init__(
            f"Feature engineering failed for '{feature_name}': {reason}",
            {"feature_name": feature_name, "reason": reason}
        )


class FileUploadError(DataError):
    """Error uploading or processing a file."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            f"File upload failed for '{filename}': {reason}",
            {"filename": filename, "reason": reason}
        )


class FileParseError(DataError):
    """Error parsing a data file (CSV, JSON, etc.)."""

    def __init__(self, filename: str, format: str, reason: str):
        super().__init__(
            f"Failed to parse {format} file '{filename}': {reason}",
            {"filename": filename, "format": format, "reason": reason}
        )


# =============================================================================
# Training / ML Errors
# =============================================================================

class TrainingError(AgentMLError):
    """Base error for ML training issues."""
    pass


class AutoMLError(TrainingError):
    """Error during AutoML training."""

    def __init__(self, experiment_id: str, reason: str):
        super().__init__(
            f"AutoML training failed: {reason}",
            {"experiment_id": experiment_id, "reason": reason}
        )


class ModelNotFoundError(TrainingError):
    """Trained model artifact not found."""

    def __init__(self, model_path: str):
        super().__init__(
            f"Model artifact not found at: {model_path}",
            {"model_path": model_path}
        )


class PredictionError(TrainingError):
    """Error making predictions with a model."""

    def __init__(self, model_id: str, reason: str):
        super().__init__(
            f"Prediction failed: {reason}",
            {"model_id": model_id, "reason": reason}
        )


class ValidationSplitError(TrainingError):
    """Error creating train/validation split."""

    def __init__(self, reason: str, dataset_size: Optional[int] = None):
        super().__init__(
            f"Validation split failed: {reason}",
            {"reason": reason, "dataset_size": dataset_size}
        )


# =============================================================================
# Resource / Infrastructure Errors
# =============================================================================

class ResourceError(AgentMLError):
    """Base error for resource/infrastructure issues."""
    pass


class DatabaseError(ResourceError):
    """Database operation failed."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            f"Database {operation} failed: {reason}",
            {"operation": operation, "reason": reason}
        )


class CacheError(ResourceError):
    """Cache (Redis) operation failed."""

    def __init__(self, operation: str, key: Optional[str] = None):
        super().__init__(
            f"Cache {operation} failed",
            {"operation": operation, "key": key}
        )


class StorageError(ResourceError):
    """File storage operation failed."""

    def __init__(self, operation: str, path: str, reason: str):
        super().__init__(
            f"Storage {operation} failed for '{path}': {reason}",
            {"operation": operation, "path": path, "reason": reason}
        )


class ModalError(ResourceError):
    """Modal.com cloud execution failed."""

    def __init__(self, reason: str, function_name: Optional[str] = None):
        super().__init__(
            f"Modal execution failed: {reason}",
            {"reason": reason, "function_name": function_name}
        )


class CeleryTaskError(ResourceError):
    """Celery background task failed."""

    def __init__(self, task_name: str, task_id: str, reason: str):
        super().__init__(
            f"Celery task '{task_name}' ({task_id}) failed: {reason}",
            {"task_name": task_name, "task_id": task_id, "reason": reason}
        )


# =============================================================================
# Authentication / Authorization Errors
# =============================================================================

class AuthError(AgentMLError):
    """Base error for auth issues."""
    pass


class AuthenticationError(AuthError):
    """User authentication failed."""

    def __init__(self, reason: str = "Invalid credentials"):
        super().__init__(f"Authentication failed: {reason}", {"reason": reason})


class AuthorizationError(AuthError):
    """User not authorized for this action."""

    def __init__(self, resource: str, action: str, user_id: Optional[str] = None):
        super().__init__(
            f"Not authorized to {action} {resource}",
            {"resource": resource, "action": action, "user_id": user_id}
        )


class TokenError(AuthError):
    """JWT token error (expired, invalid, etc.)."""

    def __init__(self, reason: str):
        super().__init__(f"Token error: {reason}", {"reason": reason})


# =============================================================================
# API / Request Errors
# =============================================================================

class APIError(AgentMLError):
    """Base error for API-level issues."""
    pass


class ResourceNotFoundError(APIError):
    """Requested resource not found."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            f"{resource_type} not found: {resource_id}",
            {"resource_type": resource_type, "resource_id": resource_id}
        )


class DuplicateResourceError(APIError):
    """Resource already exists."""

    def __init__(self, resource_type: str, identifier: str):
        super().__init__(
            f"{resource_type} already exists: {identifier}",
            {"resource_type": resource_type, "identifier": identifier}
        )


class InvalidRequestError(APIError):
    """Request is invalid or malformed."""

    def __init__(self, reason: str, field: Optional[str] = None):
        super().__init__(
            f"Invalid request: {reason}",
            {"reason": reason, "field": field}
        )


class RateLimitError(APIError):
    """API rate limit exceeded."""

    def __init__(self, limit: int, window_seconds: int):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window_seconds}s",
            {"limit": limit, "window_seconds": window_seconds}
        )


# =============================================================================
# Visualization Errors
# =============================================================================

class VisualizationError(AgentMLError):
    """Error generating visualizations."""

    def __init__(self, chart_type: str, reason: str):
        super().__init__(
            f"Failed to generate {chart_type} visualization: {reason}",
            {"chart_type": chart_type, "reason": reason}
        )


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(AgentMLError):
    """Configuration is missing or invalid."""

    def __init__(self, config_key: str, reason: str):
        super().__init__(
            f"Configuration error for '{config_key}': {reason}",
            {"config_key": config_key, "reason": reason}
        )


# =============================================================================
# Agent Pipeline Errors
# =============================================================================

class AgentPipelineError(AgentMLError):
    """Error in the agent pipeline."""
    pass


class AgentStepError(AgentPipelineError):
    """Error in a specific agent pipeline step."""

    def __init__(self, step_type: str, step_id: Optional[str] = None, reason: str = ""):
        super().__init__(
            f"Agent step '{step_type}' failed: {reason}",
            {"step_type": step_type, "step_id": step_id, "reason": reason}
        )


class ImprovementPipelineError(AgentPipelineError):
    """Error in the auto-improvement pipeline."""

    def __init__(self, experiment_id: str, stage: str, reason: str):
        super().__init__(
            f"Improvement pipeline failed at '{stage}': {reason}",
            {"experiment_id": experiment_id, "stage": stage, "reason": reason}
        )


class OverfittingDetectedError(AgentPipelineError):
    """Overfitting detected - pipeline should stop."""

    def __init__(self, current_iteration: int, best_iteration: int, message: str):
        super().__init__(
            message,
            {
                "current_iteration": current_iteration,
                "best_iteration": best_iteration,
                "recommendation": f"Revert to iteration {best_iteration}"
            }
        )


class PipelineCancelledError(AgentPipelineError):
    """Pipeline was cancelled by user."""

    def __init__(self, run_id: str, step_type: str = None):
        msg = f"Pipeline {run_id} was cancelled"
        if step_type:
            msg += f" during step '{step_type}'"
        super().__init__(
            msg,
            {"run_id": run_id, "step_type": step_type, "reason": "User requested cancellation"}
        )

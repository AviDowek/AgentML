from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import secrets


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Application
    app_name: str = "Agentic ML Platform"
    debug: bool = False

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/agentic_ml"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # LLM API (placeholder)
    llm_api_key: str = ""
    llm_api_base_url: str = "https://api.openai.com/v1"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"]

    # File uploads
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 100

    # Model artifacts
    artifacts_dir: str = "./artifacts"

    # AutoML defaults
    automl_time_limit: int = 300  # 5 minutes default
    automl_presets: str = "medium_quality"

    # Resource limits for local training (helps prevent system freezes)
    resource_limits_enabled: bool = True  # Default ON for safety
    max_cpu_cores: int = 2  # Limit CPU cores for training
    max_memory_gb: int = 8  # Limit memory usage
    autogluon_num_cpus: int = 2  # AutoGluon-specific CPU limit
    autogluon_num_gpus: int = 0  # AutoGluon GPU limit (0 = CPU only)

    # Modal.com cloud training
    modal_token_id: str = ""
    modal_token_secret: str = ""
    modal_enabled: bool = False  # Whether Modal is configured

    # Authentication
    secret_key: str = secrets.token_urlsafe(32)  # JWT signing key
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""

    # Email settings (for invitations)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from_email: str = ""
    smtp_from_name: str = "AgentML"

    # Frontend URL (for email links)
    frontend_url: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

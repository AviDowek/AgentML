"""API routes."""
from app.api.health import router as health_router
from app.api.projects import router as projects_router
from app.api.data_sources import router as data_sources_router
from app.api.dataset_specs import router as dataset_specs_router
from app.api.experiments import router as experiments_router
from app.api.models import router as models_router
from app.api.api_keys import router as api_keys_router
from app.api.api_keys import settings_router
from app.api.chat import router as chat_router
from app.api.auth import router as auth_router
from app.api.sharing import router as sharing_router
from app.api.agent import router as agent_router
from app.api.agent import agent_runs_router
from app.api.agent import experiment_agent_router
from app.api.agent import orchestration_router
from app.api.visualization import router as visualization_router
from app.api.training_ws import router as training_ws_router
from app.api.research_cycles import router as research_cycles_router
from app.api.research_cycles import cycles_router as research_cycles_detail_router
from app.api.project_history import router as project_history_router
from app.api.auto_ds import router as auto_ds_router
from app.api.context_documents import router as context_documents_router

__all__ = [
    "health_router",
    "projects_router",
    "data_sources_router",
    "dataset_specs_router",
    "experiments_router",
    "models_router",
    "api_keys_router",
    "settings_router",
    "chat_router",
    "auth_router",
    "sharing_router",
    "agent_router",
    "agent_runs_router",
    "experiment_agent_router",
    "orchestration_router",
    "visualization_router",
    "training_ws_router",
    "research_cycles_router",
    "research_cycles_detail_router",
    "project_history_router",
    "auto_ds_router",
    "context_documents_router",
]

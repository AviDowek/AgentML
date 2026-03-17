import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.core.config import get_settings

# Configure logging to show all INFO+ messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Also set the app logger to DEBUG for more detail
logging.getLogger("app").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
from app.api import (
    health_router,
    projects_router,
    data_sources_router,
    dataset_specs_router,
    experiments_router,
    models_router,
    api_keys_router,
    settings_router,
    chat_router,
    auth_router,
    sharing_router,
    agent_router,
    agent_runs_router,
    experiment_agent_router,
    orchestration_router,
    visualization_router,
    training_ws_router,
    research_cycles_router,
    research_cycles_detail_router,
    project_history_router,
    auto_ds_router,
    context_documents_router,
)

settings = get_settings()

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])

app = FastAPI(
    title=settings.app_name,
    description="An agentic ML engineer platform for tabular ML problems",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,  # Disable Swagger UI in production
    redoc_url="/redoc" if settings.debug else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler that logs errors and returns CORS-friendly response."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())

    # Get origin from request
    origin = request.headers.get("origin", "")

    # Build response with CORS headers
    response = JSONResponse(
        status_code=500,
        content={"detail": str(exc) if settings.debug else "Internal server error"},
    )

    # Add CORS headers if origin is allowed
    if origin in settings.cors_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"

    return response


# Include routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(data_sources_router)
app.include_router(dataset_specs_router)
app.include_router(experiments_router)
app.include_router(models_router)
app.include_router(api_keys_router)
app.include_router(settings_router)
app.include_router(chat_router)
app.include_router(sharing_router)
app.include_router(agent_router)
app.include_router(agent_runs_router)
app.include_router(experiment_agent_router)
app.include_router(orchestration_router)
app.include_router(visualization_router)
app.include_router(training_ws_router)
app.include_router(research_cycles_router)
app.include_router(research_cycles_detail_router)
app.include_router(project_history_router)
app.include_router(auto_ds_router)
app.include_router(context_documents_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
    }

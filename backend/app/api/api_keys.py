"""API key management and app settings endpoints."""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user_required
from app.models.api_key import LLMProvider
from app.models.user import User
from app.models.app_settings import AppSettings, AIModel, AI_MODEL_CONFIG
from app.schemas.api_key import ApiKeyCreate, ApiKeyResponse, ApiKeyStatusResponse
from app.schemas.app_settings import AppSettingsResponse, AppSettingsUpdate, AIModelOption
from app.services import api_key_service
from app.services.encryption import decrypt

router = APIRouter(prefix="/api/v1/api-keys", tags=["API Keys"])


@router.get("/status", response_model=ApiKeyStatusResponse)
def get_key_status(db: Session = Depends(get_db)):
    """Check which LLM providers have API keys configured."""
    return api_key_service.get_api_key_status(db)


@router.get("", response_model=list[ApiKeyResponse])
def list_api_keys(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user_required),
):
    """List all stored API keys (masked). Requires authentication."""
    keys = api_key_service.get_all_api_keys(db)
    results = []
    for key in keys:
        try:
            plain = decrypt(key.api_key)
        except ValueError:
            plain = "****"
        results.append(
            ApiKeyResponse(
                id=key.id,
                provider=key.provider,
                name=key.name,
                key_preview=api_key_service.mask_api_key(plain),
                created_at=key.created_at,
                updated_at=key.updated_at,
            )
        )
    return results


@router.post("", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
def create_or_update_key(
    data: ApiKeyCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user_required),
):
    """Create or update an API key for a provider. Requires authentication."""
    key = api_key_service.create_or_update_api_key(db, data)
    return ApiKeyResponse(
        id=key.id,
        provider=key.provider,
        name=key.name,
        key_preview=api_key_service.mask_api_key(data.api_key),  # Mask the original plaintext
        created_at=key.created_at,
        updated_at=key.updated_at,
    )


@router.delete("/{provider}")
def delete_api_key(
    provider: LLMProvider,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user_required),
):
    """Delete an API key for a provider."""
    deleted = api_key_service.delete_api_key(db, provider)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No API key found for provider: {provider}",
        )
    return {"message": f"API key for {provider} deleted successfully"}


# ============== App Settings Endpoints ==============

settings_router = APIRouter(prefix="/api/v1/settings", tags=["Settings"])


@settings_router.get("/ai-models", response_model=list[AIModelOption])
def get_available_ai_models():
    """Get list of available AI models."""
    return [
        AIModelOption(
            value=model.value,
            display_name=config["display_name"],
            description=config["description"],
        )
        for model, config in AI_MODEL_CONFIG.items()
    ]


@settings_router.get("", response_model=AppSettingsResponse)
def get_app_settings(db: Session = Depends(get_db)):
    """Get current app settings."""
    settings = db.query(AppSettings).first()
    if not settings:
        # Create default settings if none exist
        settings = AppSettings(ai_model=AIModel.GPT_5_1_THINKING)
        db.add(settings)
        db.commit()
        db.refresh(settings)

    model_config = AI_MODEL_CONFIG.get(settings.ai_model, AI_MODEL_CONFIG[AIModel.GPT_5_1_THINKING])
    return AppSettingsResponse(
        ai_model=settings.ai_model,
        ai_model_display_name=model_config["display_name"],
        updated_at=settings.updated_at,
    )


@settings_router.patch("", response_model=AppSettingsResponse)
def update_app_settings(
    data: AppSettingsUpdate,
    db: Session = Depends(get_db),
):
    """Update app settings."""
    settings = db.query(AppSettings).first()
    if not settings:
        settings = AppSettings(ai_model=AIModel.GPT_5_1_THINKING)
        db.add(settings)
        db.commit()
        db.refresh(settings)

    if data.ai_model is not None:
        settings.ai_model = data.ai_model
        settings.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(settings)

    model_config = AI_MODEL_CONFIG.get(settings.ai_model, AI_MODEL_CONFIG[AIModel.GPT_5_1_THINKING])
    return AppSettingsResponse(
        ai_model=settings.ai_model,
        ai_model_display_name=model_config["display_name"],
        updated_at=settings.updated_at,
    )

"""API key management service."""
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.api_key import ApiKey, LLMProvider
from app.schemas.api_key import ApiKeyCreate


def mask_api_key(key: str) -> str:
    """Mask an API key for display, showing only first 4 and last 4 characters."""
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


def get_api_key(db: Session, provider: LLMProvider) -> Optional[ApiKey]:
    """Get API key for a specific provider."""
    return db.query(ApiKey).filter(ApiKey.provider == provider).first()


def get_api_key_by_id(db: Session, key_id: UUID) -> Optional[ApiKey]:
    """Get API key by ID."""
    return db.query(ApiKey).filter(ApiKey.id == key_id).first()


def get_all_api_keys(db: Session) -> list[ApiKey]:
    """Get all stored API keys."""
    return db.query(ApiKey).all()


def get_api_key_status(db: Session) -> dict[str, bool]:
    """Check which providers have keys configured."""
    keys = db.query(ApiKey.provider).all()
    providers = {k[0] for k in keys}
    return {
        "openai": LLMProvider.OPENAI in providers,
        "gemini": LLMProvider.GEMINI in providers,
    }


def create_or_update_api_key(db: Session, data: ApiKeyCreate) -> ApiKey:
    """Create or update an API key for a provider."""
    existing = get_api_key(db, data.provider)

    if existing:
        # Update existing key
        existing.api_key = data.api_key
        if data.name is not None:
            existing.name = data.name
        db.commit()
        db.refresh(existing)
        return existing
    else:
        # Create new key
        api_key = ApiKey(
            provider=data.provider,
            api_key=data.api_key,
            name=data.name,
        )
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        return api_key


def delete_api_key(db: Session, provider: LLMProvider) -> bool:
    """Delete API key for a provider."""
    api_key = get_api_key(db, provider)
    if api_key:
        db.delete(api_key)
        db.commit()
        return True
    return False


def get_decrypted_key(db: Session, provider: LLMProvider) -> Optional[str]:
    """Get the actual API key value for a provider (for internal use)."""
    api_key = get_api_key(db, provider)
    if api_key:
        return api_key.api_key
    return None

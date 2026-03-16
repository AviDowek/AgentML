"""API key storage model."""
from sqlalchemy import Column, String, Text, Enum as SQLEnum
import enum

from app.core.database import Base
from app.models.base import TimestampMixin, GUID
import uuid


class LLMProvider(str, enum.Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


class ApiKey(Base, TimestampMixin):
    """Stored API keys for LLM providers."""
    __tablename__ = "api_keys"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    provider = Column(
        SQLEnum(LLMProvider, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        unique=True
    )
    api_key = Column(Text, nullable=False)  # In production, encrypt this
    name = Column(String(255), nullable=True)  # Display name

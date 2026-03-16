"""App settings model for global application preferences."""
from sqlalchemy import Column, String
import enum
import uuid

from app.core.database import Base
from app.models.base import TimestampMixin, GUID


class AIModel(str, enum.Enum):
    """Available AI models for the platform.

    These map to specific OpenAI model configurations:
    - GPT_5_1_THINKING: gpt-5.1 with reasoning_effort="high" (deep thinking)
    - GPT_5_1: gpt-5.1 with reasoning_effort="none" (faster, no extended thinking)
    - GPT_4_1: gpt-4.1 (previous generation, faster and cheaper)
    - O3_DEEP_RESEARCH: o3-deep-research for comprehensive web research
    """
    GPT_5_1_THINKING = "gpt-5.1-thinking"  # gpt-5.1 with high reasoning
    GPT_5_1 = "gpt-5.1"  # gpt-5.1 without extended thinking
    GPT_4_1 = "gpt-4.1"  # Previous generation
    O3_DEEP_RESEARCH = "o3-deep-research"  # Deep research with web search


# Model configuration mapping
AI_MODEL_CONFIG = {
    AIModel.GPT_5_1_THINKING: {
        "model": "gpt-5.1",
        "reasoning_effort": "high",
        "display_name": "GPT-5.1 Thinking",
        "description": "Most capable. Uses extended thinking for complex tasks. Slower but higher quality.",
    },
    AIModel.GPT_5_1: {
        "model": "gpt-5.1",
        "reasoning_effort": "none",
        "display_name": "GPT-5.1",
        "description": "Fast and capable. Good balance of speed and quality.",
    },
    AIModel.GPT_4_1: {
        "model": "gpt-4.1",
        "reasoning_effort": "none",
        "display_name": "GPT-4.1",
        "description": "Previous generation. Fastest and most cost-effective.",
    },
    AIModel.O3_DEEP_RESEARCH: {
        "model": "o3-deep-research-2025-06-26",
        "reasoning_effort": "none",
        "display_name": "O3 Deep Research",
        "description": "Specialized for in-depth web research. Searches and synthesizes hundreds of sources.",
        "uses_responses_api": True,  # Uses responses.create instead of chat.completions
    },
}


class AppSettings(Base, TimestampMixin):
    """Global application settings stored in the database.

    Uses a key-value pattern for flexibility. Only one row should exist.
    """
    __tablename__ = "app_settings"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

    # AI Model preference
    ai_model = Column(
        String(50),
        nullable=False,
        default=AIModel.GPT_5_1_THINKING.value
    )

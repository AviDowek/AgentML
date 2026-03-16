"""Chat conversation models."""
import uuid
from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.models.base import TimestampMixin, GUID, JSONType


class Conversation(Base, TimestampMixin):
    """A chat conversation with the AI assistant."""

    __tablename__ = "conversations"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, default="New Conversation")
    context_type = Column(String(50), nullable=True)  # e.g., 'experiment', 'project', 'model'
    context_id = Column(String(36), nullable=True)  # ID of the related entity
    context_data = Column(JSONType(), nullable=True)  # Snapshot of context at conversation start

    # Owner relationship
    owner_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    # Relationships
    owner = relationship("User", back_populates="conversations")
    messages = relationship(
        "ConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at",
    )

    def __repr__(self):
        return f"<Conversation {self.id}: {self.title}>"


class ConversationMessage(Base, TimestampMixin):
    """A single message in a conversation."""

    __tablename__ = "conversation_messages"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        GUID(),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)

    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<ConversationMessage {self.id}: {self.role}>"

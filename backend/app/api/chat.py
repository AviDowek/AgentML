"""Chat endpoint for AI assistant with conversation persistence."""
import json
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.api_key import LLMProvider
from app.models.user import User
from app.models.conversation import Conversation, ConversationMessage
from app.services import api_key_service
from app.services.llm_client import get_llm_client
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationSummary,
    ConversationListResponse,
    ConversationMessageResponse,
    SendMessageRequest,
    SendMessageResponse,
)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str = Field(..., min_length=1, description="User's message")
    context: dict = Field(default_factory=dict, description="Page context data")
    history: list[ChatMessage] = Field(default_factory=list, description="Previous messages")
    provider: Optional[LLMProvider] = Field(None, description="LLM provider to use")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str
    provider: str


def build_system_prompt(context: dict) -> str:
    """Build a system prompt based on the context."""
    base_prompt = """You are an AI assistant for AgentML, an AutoML platform that helps users train machine learning models.
You help users understand their experiments, models, data, and guide them in making decisions.
Be concise, helpful, and technical when appropriate.

When discussing metrics:
- Higher accuracy, precision, recall, F1, ROC-AUC is better
- Lower RMSE, MAE, log_loss is better
- Explain what metrics mean in simple terms when asked

When recommending models:
- Consider the trade-off between accuracy and training time
- LightGBM and XGBoost are usually good starting points
- Neural networks may need more data
- Consider interpretability requirements
"""

    if not context:
        return base_prompt + "\n\nNo specific context provided. Help the user with general AutoML questions."

    context_type = context.get("type", "unknown")

    if context_type == "experiment":
        return base_prompt + f"""

CURRENT CONTEXT: Experiment Details
{json.dumps(context.get("data", {}), indent=2, default=str)}

Help the user understand this experiment's results, metrics, and model performance.
If they ask about model selection, explain the trade-offs.
If the experiment failed, help diagnose the issue.
"""

    elif context_type == "project":
        project_data = context.get("data", {})
        visualizations = project_data.get("visualizations", [])

        viz_context = ""
        if visualizations:
            viz_context = "\n\nCURRENT VISUALIZATIONS:\n"
            for viz in visualizations:
                viz_context += f"- {viz.get('title', 'Untitled')} ({viz.get('chart_type', 'chart')}): {viz.get('description', 'No description')}\n"
                if viz.get('explanation'):
                    viz_context += f"  Explanation: {viz.get('explanation')}\n"

        # Build agent pipeline context
        agent_pipeline = project_data.get("agent_pipeline")
        pipeline_context = ""
        if agent_pipeline:
            pipeline_context = f"\n\nAI PIPELINE ANALYSIS (Status: {agent_pipeline.get('status', 'unknown')}):\n"
            pipeline_context += f"Description: {agent_pipeline.get('description', 'N/A')}\n"
            steps = agent_pipeline.get("steps", [])
            for step in steps:
                step_type = step.get("step_type", "unknown")
                status = step.get("status", "unknown")
                pipeline_context += f"\n{step_type.upper()} ({status}):\n"
                output = step.get("output_json", {})
                if output:
                    # Extract key insights from each step
                    if step_type == "problem_understanding":
                        pipeline_context += f"  - Task Type: {output.get('task_type', 'N/A')}\n"
                        pipeline_context += f"  - Suggested Target: {output.get('suggested_target_column', 'N/A')}\n"
                        pipeline_context += f"  - Rationale: {output.get('rationale', 'N/A')}\n"
                    elif step_type == "data_audit":
                        pipeline_context += f"  - Quality Score: {output.get('quality_score', 'N/A')}\n"
                        pipeline_context += f"  - Issues: {output.get('issues', [])}\n"
                        pipeline_context += f"  - Recommendations: {output.get('recommendations', [])}\n"
                    elif step_type == "dataset_design":
                        pipeline_context += f"  - Target Column: {output.get('target_column', 'N/A')}\n"
                        pipeline_context += f"  - Feature Columns: {output.get('feature_columns', [])}\n"
                    elif step_type == "experiment_design":
                        pipeline_context += f"  - Primary Metric: {output.get('primary_metric', 'N/A')}\n"
                        pipeline_context += f"  - Time Budget: {output.get('automl_config', {}).get('time_limit_minutes', 'N/A')} minutes\n"
                    elif step_type == "plan_critic":
                        pipeline_context += f"  - Approval: {output.get('approved', 'N/A')}\n"
                        pipeline_context += f"  - Feedback: {output.get('feedback', 'N/A')}\n"
                if step.get("error_message"):
                    pipeline_context += f"  - Error: {step.get('error_message')}\n"

        return base_prompt + f"""

CURRENT CONTEXT: Project Overview
{json.dumps(project_data, indent=2, default=str)}
{viz_context}
{pipeline_context}
Help the user understand their project, data visualizations, AI pipeline analysis results, recommend next steps, or guide them in setting up experiments.
If the user asks about visualizations or charts, reference the current visualizations listed above.
If the user asks about the AI analysis or pipeline, explain the insights from each step.
"""

    elif context_type == "model":
        return base_prompt + f"""

CURRENT CONTEXT: Model Details
{json.dumps(context.get("data", {}), indent=2, default=str)}

Help the user understand this model's performance, feature importances, and when to use it.
"""

    elif context_type == "dataset":
        return base_prompt + f"""

CURRENT CONTEXT: Dataset Configuration
{json.dumps(context.get("data", {}), indent=2, default=str)}

Help the user understand their dataset, feature selection, and target column configuration.
"""

    else:
        return base_prompt + f"""

CURRENT CONTEXT:
{json.dumps(context, indent=2, default=str)}

Help the user with questions related to this context.
"""


def get_llm_provider_and_key(db: Session, requested_provider: Optional[LLMProvider] = None):
    """Get the LLM provider and API key to use."""
    key_status = api_key_service.get_api_key_status(db)

    provider = requested_provider
    if provider:
        provider_key = "openai" if provider == LLMProvider.OPENAI else "gemini"
        if not key_status.get(provider_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No API key configured for {provider.value}",
            )
    else:
        if key_status.get("openai"):
            provider = LLMProvider.OPENAI
        elif key_status.get("gemini"):
            provider = LLMProvider.GEMINI
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No API keys configured. Please add an OpenAI or Gemini API key in Settings.",
            )

    api_key = api_key_service.get_api_key(db, provider)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"API key for {provider.value} not found",
        )

    return provider, api_key.api_key


# ==================== Conversation Endpoints ====================


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
    context_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List all conversations, optionally filtered by context type."""
    query = db.query(Conversation)

    # Filter by owner if user is authenticated
    if current_user:
        query = query.filter(
            (Conversation.owner_id == current_user.id) | (Conversation.owner_id.is_(None))
        )

    if context_type:
        query = query.filter(Conversation.context_type == context_type)

    total = query.count()
    conversations = query.order_by(Conversation.updated_at.desc()).offset(skip).limit(limit).all()

    items = []
    for conv in conversations:
        message_count = len(conv.messages)
        last_message = conv.messages[-1] if conv.messages else None
        last_preview = None
        if last_message:
            last_preview = last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content

        items.append(ConversationSummary(
            id=conv.id,
            title=conv.title,
            context_type=conv.context_type,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=message_count,
            last_message_preview=last_preview,
        ))

    return ConversationListResponse(items=items, total=total)


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    data: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Create a new conversation."""
    conversation = Conversation(
        title=data.title,
        context_type=data.context_type,
        context_id=data.context_id,
        context_data=data.context_data,
        owner_id=current_user.id if current_user else None,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        context_type=conversation.context_type,
        context_id=conversation.context_id,
        context_data=conversation.context_data,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[],
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get a conversation with all its messages."""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Check ownership
    if current_user and conversation.owner_id and conversation.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this conversation",
        )

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        context_type=conversation.context_type,
        context_id=conversation.context_id,
        context_data=conversation.context_data,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            ConversationMessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
            )
            for msg in conversation.messages
        ],
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: UUID,
    data: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update a conversation (e.g., rename it)."""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Check ownership
    if current_user and conversation.owner_id and conversation.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this conversation",
        )

    if data.title is not None:
        conversation.title = data.title

    db.commit()
    db.refresh(conversation)

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        context_type=conversation.context_type,
        context_id=conversation.context_id,
        context_data=conversation.context_data,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            ConversationMessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at,
            )
            for msg in conversation.messages
        ],
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Delete a conversation and all its messages."""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Check ownership
    if current_user and conversation.owner_id and conversation.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this conversation",
        )

    db.delete(conversation)
    db.commit()

    return {"message": "Conversation deleted successfully"}


@router.post("/conversations/{conversation_id}/messages", response_model=SendMessageResponse)
async def send_message(
    conversation_id: UUID,
    data: SendMessageRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Send a message in a conversation and get AI response."""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Check ownership
    if current_user and conversation.owner_id and conversation.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this conversation",
        )

    # Get provider and key
    provider, api_key_str = get_llm_provider_and_key(db)

    # Build system prompt with context
    context = {"type": conversation.context_type, "data": conversation.context_data} if conversation.context_data else {}
    system_prompt = build_system_prompt(context)

    # Extract visualization images from request (preferred) or context
    images = []

    # First, try to get images from the request (most up-to-date)
    if data.current_visualizations:
        for viz in data.current_visualizations:
            if viz.image_base64:
                images.append({
                    "base64": viz.image_base64,
                    "description": f"{viz.title}: {viz.description or ''}"
                })
    # Fallback to context if no visualizations in request
    elif context.get("data"):
        visualizations = context["data"].get("visualizations", [])
        for viz in visualizations:
            if viz.get("image_base64"):
                images.append({
                    "base64": viz["image_base64"],
                    "description": f"{viz.get('title', 'Chart')}: {viz.get('description', '')}"
                })

    # Build messages for the LLM
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last 20 messages for context)
    for msg in conversation.messages[-20:]:
        messages.append({"role": msg.role, "content": msg.content})

    # Add current user message
    messages.append({"role": "user", "content": data.message})

    # Save user message
    user_msg = ConversationMessage(
        conversation_id=conversation.id,
        role="user",
        content=data.message,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # Get AI response
    try:
        client = get_llm_client(provider, api_key_str)
        # Pass images to the LLM if available (for vision models)
        response_text = await client.chat(messages, images=images if images else None)

        # Save assistant message
        assistant_msg = ConversationMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=response_text,
        )
        db.add(assistant_msg)

        # Auto-generate title from first message if still default
        if conversation.title == "New Conversation" and len(conversation.messages) <= 2:
            # Use first 50 chars of first user message as title
            conversation.title = data.message[:50] + ("..." if len(data.message) > 50 else "")

        db.commit()
        db.refresh(assistant_msg)

        return SendMessageResponse(
            user_message=ConversationMessageResponse(
                id=user_msg.id,
                conversation_id=user_msg.conversation_id,
                role=user_msg.role,
                content=user_msg.content,
                created_at=user_msg.created_at,
            ),
            assistant_message=ConversationMessageResponse(
                id=assistant_msg.id,
                conversation_id=assistant_msg.conversation_id,
                role=assistant_msg.role,
                content=assistant_msg.content,
                created_at=assistant_msg.created_at,
            ),
            provider=provider.value,
        )
    except Exception as e:
        # Rollback user message on error
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI response: {str(e)}",
        )


# ==================== Legacy Single-shot Chat Endpoint ====================


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Chat with the AI assistant (single-shot, no persistence).

    Accepts a message, optional context data, and conversation history.
    Returns the AI's response.
    """
    provider, api_key_str = get_llm_provider_and_key(db, request.provider)

    # Build system prompt with context
    system_prompt = build_system_prompt(request.context)

    # Build messages for the LLM
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for msg in request.history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})

    # Add current user message
    messages.append({"role": "user", "content": request.message})

    # Get LLM client and generate response
    try:
        client = get_llm_client(provider, api_key_str)
        response = await client.chat(messages)

        return ChatResponse(
            response=response,
            provider=provider.value,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI response: {str(e)}",
        )

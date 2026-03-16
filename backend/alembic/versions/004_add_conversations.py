"""Add conversations tables

Revision ID: 004_add_conversations
Revises: 003_add_celery_task_id
Create Date: 2024-01-16

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision = '004_add_conversations'
down_revision = '003_add_celery_task_id'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('title', sa.String(255), nullable=False, default='New Conversation'),
        sa.Column('context_type', sa.String(50), nullable=True),
        sa.Column('context_id', sa.String(36), nullable=True),
        sa.Column('context_data', JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    # Create conversation_messages table
    op.create_table(
        'conversation_messages',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('conversation_id', UUID(as_uuid=True), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    # Create index on conversation_id for faster lookups
    op.create_index('ix_conversation_messages_conversation_id', 'conversation_messages', ['conversation_id'])


def downgrade() -> None:
    op.drop_index('ix_conversation_messages_conversation_id', table_name='conversation_messages')
    op.drop_table('conversation_messages')
    op.drop_table('conversations')

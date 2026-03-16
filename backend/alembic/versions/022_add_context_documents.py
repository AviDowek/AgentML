"""add_context_documents

Revision ID: add_context_documents
Revises: tier1_tier2_features
Create Date: 2025-12-30 10:00:00.000000

Adds:
- context_documents table for storing supplementary project documentation
- context_ab_testing_enabled column to projects table
- use_context_documents and context_ab_testing columns to auto_ds_sessions table
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'add_context_documents'
down_revision: Union[str, None] = '88bb48784f88'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create context_documents table
    op.create_table(
        'context_documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, index=True),

        # Document metadata
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('original_filename', sa.String(500), nullable=False),
        sa.Column('file_path', sa.String(1000), nullable=False),
        sa.Column('file_type', sa.String(50), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),

        # User explanation (required)
        sa.Column('explanation', sa.Text(), nullable=False),

        # Extracted text content
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('extraction_status', sa.String(50), nullable=False, server_default='pending'),
        sa.Column('extraction_error', sa.Text(), nullable=True),

        # Usage control
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),

        # Timestamps (from TimestampMixin)
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Add context_ab_testing_enabled to projects table
    op.add_column(
        'projects',
        sa.Column(
            'context_ab_testing_enabled',
            sa.Boolean(),
            nullable=False,
            server_default='false'
        )
    )

    # Add context document settings to auto_ds_sessions table
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'use_context_documents',
            sa.Boolean(),
            nullable=False,
            server_default='true'
        )
    )
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'context_ab_testing',
            sa.Boolean(),
            nullable=False,
            server_default='false'
        )
    )


def downgrade() -> None:
    # Remove columns from auto_ds_sessions
    op.drop_column('auto_ds_sessions', 'context_ab_testing')
    op.drop_column('auto_ds_sessions', 'use_context_documents')

    # Remove column from projects
    op.drop_column('projects', 'context_ab_testing_enabled')

    # Drop context_documents table
    op.drop_table('context_documents')

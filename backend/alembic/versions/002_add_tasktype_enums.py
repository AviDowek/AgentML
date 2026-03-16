"""Add new TaskType enum values and create API key / synthetic dataset tables

Revision ID: 002_add_tasktype_enums
Revises: 001_initial
Create Date: 2024-01-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_add_tasktype_enums'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new values to tasktype enum
    # PostgreSQL requires ALTER TYPE to add enum values
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'binary'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'quantile'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'timeseries_forecast'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'multimodal_classification'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'multimodal_regression'")

    # Create LLM provider enum
    op.execute("CREATE TYPE llmprovider AS ENUM ('openai', 'gemini')")

    # Create dataset type enum for synthetic datasets
    op.execute("CREATE TYPE datasettype AS ENUM ('binary_classification', 'multiclass_classification', 'regression', 'timeseries')")

    # Create synthetic dataset status enum
    op.execute("CREATE TYPE syntheticdatasetstatus AS ENUM ('pending', 'generating', 'completed', 'failed')")

    # Create api_keys table
    op.create_table(
        'api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('provider', postgresql.ENUM('openai', 'gemini', name='llmprovider', create_type=False), nullable=False, unique=True),
        sa.Column('api_key', sa.Text(), nullable=False),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )

    # Create synthetic_datasets table
    op.create_table(
        'synthetic_datasets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('dataset_type', postgresql.ENUM('binary_classification', 'multiclass_classification', 'regression', 'timeseries', name='datasettype', create_type=False), nullable=False),
        sa.Column('provider', postgresql.ENUM('openai', 'gemini', name='llmprovider', create_type=False), nullable=False),
        sa.Column('target_training_minutes', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('num_rows', sa.Integer(), nullable=True),
        sa.Column('num_features', sa.Integer(), nullable=True),
        sa.Column('schema_info', postgresql.JSONB(), nullable=True),
        sa.Column('csv_content', sa.Text(), nullable=True),
        sa.Column('status', postgresql.ENUM('pending', 'generating', 'completed', 'failed', name='syntheticdatasetstatus', create_type=False), nullable=False, server_default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('generation_prompt', sa.Text(), nullable=True),
        sa.Column('data_source_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('data_sources.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_synthetic_datasets_status', 'synthetic_datasets', ['status'])


def downgrade() -> None:
    # Drop tables
    op.drop_index('ix_synthetic_datasets_status', 'synthetic_datasets')
    op.drop_table('synthetic_datasets')
    op.drop_table('api_keys')

    # Drop enums
    op.execute("DROP TYPE syntheticdatasetstatus")
    op.execute("DROP TYPE datasettype")
    op.execute("DROP TYPE llmprovider")

    # Note: PostgreSQL doesn't support removing enum values easily
    # The new tasktype values will remain in the database

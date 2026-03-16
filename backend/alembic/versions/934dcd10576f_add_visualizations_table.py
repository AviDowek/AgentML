"""add_visualizations_table

Revision ID: 934dcd10576f
Revises: 006_add_agent_runs
Create Date: 2025-12-04 01:04:22.991629

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '934dcd10576f'
down_revision: Union[str, None] = '006_add_agent_runs'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('visualizations',
        sa.Column('id', UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', UUID(as_uuid=True), nullable=False),
        sa.Column('data_source_id', UUID(as_uuid=True), nullable=True),
        sa.Column('owner_id', UUID(as_uuid=True), nullable=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('chart_type', sa.String(length=50), nullable=True),
        sa.Column('request', sa.Text(), nullable=True),
        sa.Column('code', sa.Text(), nullable=False),
        sa.Column('image_base64', sa.Text(), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('is_ai_suggested', sa.String(length=10), nullable=True),
        sa.Column('display_order', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_visualizations_project_id', 'visualizations', ['project_id'], unique=False)
    op.create_index('ix_visualizations_owner_id', 'visualizations', ['owner_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_visualizations_owner_id', table_name='visualizations')
    op.drop_index('ix_visualizations_project_id', table_name='visualizations')
    op.drop_table('visualizations')

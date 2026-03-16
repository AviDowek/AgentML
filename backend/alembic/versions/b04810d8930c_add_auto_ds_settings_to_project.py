"""add_auto_ds_settings_to_project

Revision ID: b04810d8930c
Revises: 021_add_auto_ds_team
Create Date: 2025-12-23 19:05:03.535025

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b04810d8930c'
down_revision: Union[str, None] = '021_add_auto_ds_team'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add Auto DS settings columns to projects table
    op.add_column('projects', sa.Column('auto_ds_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('projects', sa.Column('auto_ds_config_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('projects', sa.Column('active_auto_ds_session_id', postgresql.UUID(as_uuid=True), nullable=True))

    # Create index on active_auto_ds_session_id
    op.create_index(op.f('ix_projects_active_auto_ds_session_id'), 'projects', ['active_auto_ds_session_id'], unique=False)

    # Create foreign key constraint
    op.create_foreign_key(
        'fk_projects_active_auto_ds_session_id',
        'projects',
        'auto_ds_sessions',
        ['active_auto_ds_session_id'],
        ['id'],
        ondelete='SET NULL'
    )


def downgrade() -> None:
    # Drop foreign key constraint
    op.drop_constraint('fk_projects_active_auto_ds_session_id', 'projects', type_='foreignkey')

    # Drop index
    op.drop_index(op.f('ix_projects_active_auto_ds_session_id'), table_name='projects')

    # Drop columns
    op.drop_column('projects', 'active_auto_ds_session_id')
    op.drop_column('projects', 'auto_ds_config_json')
    op.drop_column('projects', 'auto_ds_enabled')

"""Add app_settings table for global preferences like AI model selection

Revision ID: 015_add_app_settings
Revises: 014_add_research_cycles
Create Date: 2024-12-11 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '015_add_app_settings'
down_revision: Union[str, None] = '014_add_research_cycles'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create app_settings table
    op.create_table(
        'app_settings',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('ai_model', sa.String(50), nullable=False, server_default='gpt-5.1-thinking'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    # Insert default settings row
    op.execute("""
        INSERT INTO app_settings (id, ai_model, created_at, updated_at)
        VALUES (gen_random_uuid(), 'gpt-5.1-thinking', NOW(), NOW())
    """)


def downgrade() -> None:
    op.drop_table('app_settings')

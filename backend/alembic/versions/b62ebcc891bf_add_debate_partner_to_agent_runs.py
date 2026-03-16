"""add_debate_partner_to_agent_runs

Revision ID: b62ebcc891bf
Revises: 020_debate_rounds
Create Date: 2025-12-20 18:32:45.086617

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b62ebcc891bf'
down_revision: Union[str, None] = '020_debate_rounds'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add debate_partner column to agent_runs table
    op.add_column('agent_runs', sa.Column('debate_partner', sa.String(length=100), nullable=True, server_default='gemini-2.0-flash'))


def downgrade() -> None:
    # Remove debate_partner column from agent_runs table
    op.drop_column('agent_runs', 'debate_partner')

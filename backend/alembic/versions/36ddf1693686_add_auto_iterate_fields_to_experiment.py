"""add_auto_iterate_fields_to_experiment

Revision ID: 36ddf1693686
Revises: a7928e83baa6
Create Date: 2025-12-21 09:05:33.637290

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '36ddf1693686'
down_revision: Union[str, None] = 'a7928e83baa6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add auto_iterate fields to experiments table
    op.add_column('experiments', sa.Column('auto_iterate_enabled', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('experiments', sa.Column('auto_iterate_max', sa.Integer(), nullable=False, server_default='5'))


def downgrade() -> None:
    op.drop_column('experiments', 'auto_iterate_max')
    op.drop_column('experiments', 'auto_iterate_enabled')

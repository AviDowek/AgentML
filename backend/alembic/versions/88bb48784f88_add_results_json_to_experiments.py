"""Add results_json to experiments

Revision ID: 88bb48784f88
Revises: tier1_tier2_features
Create Date: 2025-12-28 18:48:17.049997

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '88bb48784f88'
down_revision: Union[str, None] = 'tier1_tier2_features'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add results_json column to experiments table
    op.add_column('experiments', sa.Column('results_json', postgresql.JSON(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    # Remove results_json column from experiments table
    op.drop_column('experiments', 'results_json')

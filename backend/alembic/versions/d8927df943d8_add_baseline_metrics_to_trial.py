"""add_baseline_metrics_to_trial

Revision ID: d8927df943d8
Revises: 018_time_metadata
Create Date: 2025-12-14 22:03:18.137319

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd8927df943d8'
down_revision: Union[str, None] = '018_time_metadata'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add baseline_metrics_json column to trials table
    op.add_column('trials', sa.Column('baseline_metrics_json', postgresql.JSON(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    # Remove baseline_metrics_json column from trials table
    op.drop_column('trials', 'baseline_metrics_json')

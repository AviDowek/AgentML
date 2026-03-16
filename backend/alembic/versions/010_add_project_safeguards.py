"""Add large dataset safeguard columns to projects table

Revision ID: 010_add_project_safeguards
Revises: 009_add_profile_json
Create Date: 2024-12-04 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '010_add_project_safeguards'
down_revision: Union[str, None] = '009_add_profile_json'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add large dataset safeguard columns to projects table
    # These control sampling and limits for large datasets
    op.add_column(
        'projects',
        sa.Column('max_training_rows', sa.Integer(), nullable=False, server_default='1000000')
    )
    op.add_column(
        'projects',
        sa.Column('profiling_sample_rows', sa.Integer(), nullable=False, server_default='50000')
    )
    op.add_column(
        'projects',
        sa.Column('max_aggregation_window_days', sa.Integer(), nullable=False, server_default='365')
    )


def downgrade() -> None:
    # Remove safeguard columns
    op.drop_column('projects', 'max_aggregation_window_days')
    op.drop_column('projects', 'profiling_sample_rows')
    op.drop_column('projects', 'max_training_rows')

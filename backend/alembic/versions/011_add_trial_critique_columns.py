"""Add training logs and critique columns to trials table

Revision ID: 011_add_trial_critique_columns
Revises: 010_add_project_safeguards
Create Date: 2024-12-07 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '011_add_trial_critique_columns'
down_revision: Union[str, None] = '010_add_project_safeguards'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add training logs column for AI analysis
    op.add_column(
        'trials',
        sa.Column('training_logs', sa.Text(), nullable=True)
    )

    # Add leaderboard JSON for model comparison data
    op.add_column(
        'trials',
        sa.Column('leaderboard_json', sa.JSON(), nullable=True)
    )

    # Add critique JSON for AI-generated feedback
    op.add_column(
        'trials',
        sa.Column('critique_json', sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('trials', 'critique_json')
    op.drop_column('trials', 'leaderboard_json')
    op.drop_column('trials', 'training_logs')

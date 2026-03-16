"""add_execution_mode_to_auto_ds_sessions

Revision ID: d3cb8e708205
Revises: 556b03f40d4d
Create Date: 2025-12-24 20:00:57.389969

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3cb8e708205'
down_revision: Union[str, None] = '556b03f40d4d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the execution mode enum type
    execution_mode_enum = sa.Enum('legacy', 'adaptive', 'phased', name='executionmode')
    execution_mode_enum.create(op.get_bind(), checkfirst=True)

    # Add execution_mode column with default 'legacy' to maintain backward compatibility
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'execution_mode',
            sa.Enum('legacy', 'adaptive', 'phased', name='executionmode'),
            nullable=False,
            server_default='legacy'
        )
    )

    # Add adaptive mode threshold column
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'adaptive_decline_threshold',
            sa.Float(),
            nullable=False,
            server_default='0.05'
        )
    )

    # Add phased mode threshold column
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'phased_min_baseline_improvement',
            sa.Float(),
            nullable=False,
            server_default='0.01'
        )
    )


def downgrade() -> None:
    # Remove columns
    op.drop_column('auto_ds_sessions', 'phased_min_baseline_improvement')
    op.drop_column('auto_ds_sessions', 'adaptive_decline_threshold')
    op.drop_column('auto_ds_sessions', 'execution_mode')

    # Drop the enum type
    execution_mode_enum = sa.Enum('legacy', 'adaptive', 'phased', name='executionmode')
    execution_mode_enum.drop(op.get_bind(), checkfirst=True)

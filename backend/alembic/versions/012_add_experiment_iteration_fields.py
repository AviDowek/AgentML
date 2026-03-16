"""Add iteration tracking fields to experiments table for auto-improve pipeline

Revision ID: 012_add_experiment_iteration_fields
Revises: 011_add_trial_critique_columns
Create Date: 2024-12-07 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '012_add_exp_iter'
down_revision: Union[str, None] = '011_add_trial_critique_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add iteration number (which iteration this experiment is, 1=original, 2+=improved)
    op.add_column(
        'experiments',
        sa.Column('iteration_number', sa.Integer(), nullable=False, server_default='1')
    )

    # Add parent experiment ID for linking iterations together
    # Use UUID type to match the id column type
    op.add_column(
        'experiments',
        sa.Column('parent_experiment_id', postgresql.UUID(as_uuid=True), nullable=True)
    )

    # Add improvement context JSON to store what changes were made
    op.add_column(
        'experiments',
        sa.Column('improvement_context_json', sa.JSON(), nullable=True)
    )

    # Add foreign key constraint for parent_experiment_id
    op.create_foreign_key(
        'fk_experiments_parent_experiment',
        'experiments',
        'experiments',
        ['parent_experiment_id'],
        ['id'],
        ondelete='SET NULL'
    )

    # Create index for finding child experiments
    op.create_index(
        'ix_experiments_parent_experiment_id',
        'experiments',
        ['parent_experiment_id']
    )


def downgrade() -> None:
    op.drop_index('ix_experiments_parent_experiment_id', table_name='experiments')
    op.drop_constraint('fk_experiments_parent_experiment', 'experiments', type_='foreignkey')
    op.drop_column('experiments', 'improvement_context_json')
    op.drop_column('experiments', 'parent_experiment_id')
    op.drop_column('experiments', 'iteration_number')

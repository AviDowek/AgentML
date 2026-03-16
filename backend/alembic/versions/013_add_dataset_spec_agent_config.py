"""Add agent_experiment_design_json column to dataset_specs table

This column stores the experiment design configuration from the agent pipeline
so that experiments can be created from the dataset detail modal even after
the original agent run has been superseded by newer runs.

Revision ID: 013_add_dataset_spec_agent_config
Revises: 012_add_exp_iter
Create Date: 2024-12-08 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '013_add_ds_agent_config'
down_revision: Union[str, None] = '012_add_exp_iter'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add agent_experiment_design_json column to store experiment design config
    op.add_column(
        'dataset_specs',
        sa.Column('agent_experiment_design_json', sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('dataset_specs', 'agent_experiment_design_json')

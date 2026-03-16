"""add_3_score_columns_to_auto_ds

Revision ID: 556b03f40d4d
Revises: b04810d8930c
Create Date: 2025-12-24 09:17:23.690315

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '556b03f40d4d'
down_revision: Union[str, None] = 'b04810d8930c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add 3 score columns to auto_ds_iterations
    op.add_column('auto_ds_iterations', sa.Column('best_train_score_this_iteration', sa.Float(), nullable=True))
    op.add_column('auto_ds_iterations', sa.Column('best_val_score_this_iteration', sa.Float(), nullable=True))
    op.add_column('auto_ds_iterations', sa.Column('best_holdout_score_this_iteration', sa.Float(), nullable=True))

    # Add 3 score columns to auto_ds_sessions
    op.add_column('auto_ds_sessions', sa.Column('best_train_score', sa.Float(), nullable=True))
    op.add_column('auto_ds_sessions', sa.Column('best_val_score', sa.Float(), nullable=True))
    op.add_column('auto_ds_sessions', sa.Column('best_holdout_score', sa.Float(), nullable=True))


def downgrade() -> None:
    # Drop columns from auto_ds_sessions
    op.drop_column('auto_ds_sessions', 'best_holdout_score')
    op.drop_column('auto_ds_sessions', 'best_val_score')
    op.drop_column('auto_ds_sessions', 'best_train_score')

    # Drop columns from auto_ds_iterations
    op.drop_column('auto_ds_iterations', 'best_holdout_score_this_iteration')
    op.drop_column('auto_ds_iterations', 'best_val_score_this_iteration')
    op.drop_column('auto_ds_iterations', 'best_train_score_this_iteration')

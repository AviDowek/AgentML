"""Add holdout_sets table

Revision ID: a7928e83baa6
Revises: b62ebcc891bf
Create Date: 2025-12-20 18:52:10.245090

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a7928e83baa6'
down_revision: Union[str, None] = 'b62ebcc891bf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create holdout_sets table
    op.create_table('holdout_sets',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('data_source_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('holdout_percentage', sa.Float(), nullable=False),
        sa.Column('total_rows_original', sa.Integer(), nullable=False),
        sa.Column('holdout_row_count', sa.Integer(), nullable=False),
        sa.Column('training_row_count', sa.Integer(), nullable=False),
        sa.Column('holdout_data_json', sa.JSON(), nullable=False),
        sa.Column('target_column', sa.String(length=255), nullable=True),
        sa.Column('feature_columns_json', sa.JSON(), nullable=True),
        sa.Column('random_seed', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_holdout_sets_data_source_id'), 'holdout_sets', ['data_source_id'], unique=False)
    op.create_index(op.f('ix_holdout_sets_project_id'), 'holdout_sets', ['project_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_holdout_sets_project_id'), table_name='holdout_sets')
    op.drop_index(op.f('ix_holdout_sets_data_source_id'), table_name='holdout_sets')
    op.drop_table('holdout_sets')

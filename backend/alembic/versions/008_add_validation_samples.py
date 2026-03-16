"""Add validation_samples table for storing model predictions on validation data

Revision ID: 008_add_validation_samples
Revises: 007_add_external_dataset_type
Create Date: 2024-12-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '008_add_validation_samples'
down_revision: Union[str, None] = '007_add_external_dataset_type'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create validation_samples table
    op.create_table(
        'validation_samples',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_version_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('model_versions.id', ondelete='CASCADE'),
                  nullable=False),
        sa.Column('row_index', sa.Integer(), nullable=False),
        sa.Column('features_json', postgresql.JSONB(), nullable=False),
        sa.Column('target_value', sa.String(500), nullable=False),
        sa.Column('predicted_value', sa.String(500), nullable=False),
        sa.Column('error_value', sa.Float(), nullable=True),
        sa.Column('absolute_error', sa.Float(), nullable=True),
        sa.Column('prediction_probabilities_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create indexes for efficient querying
    op.create_index(
        'ix_validation_samples_model_version_id',
        'validation_samples',
        ['model_version_id']
    )
    op.create_index(
        'ix_validation_samples_model_row',
        'validation_samples',
        ['model_version_id', 'row_index']
    )
    op.create_index(
        'ix_validation_samples_error',
        'validation_samples',
        ['model_version_id', 'absolute_error']
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_validation_samples_error', table_name='validation_samples')
    op.drop_index('ix_validation_samples_model_row', table_name='validation_samples')
    op.drop_index('ix_validation_samples_model_version_id', table_name='validation_samples')

    # Drop table
    op.drop_table('validation_samples')

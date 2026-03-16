"""Add profile_json column to data_sources table for detailed data profiling

Revision ID: 009_add_profile_json
Revises: 008_add_validation_samples
Create Date: 2024-12-04 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '009_add_profile_json'
down_revision: Union[str, None] = '008_add_validation_samples'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add profile_json column to data_sources table
    # This stores detailed profiling information for AI agents to reason over
    op.add_column(
        'data_sources',
        sa.Column('profile_json', postgresql.JSONB(), nullable=True)
    )


def downgrade() -> None:
    # Remove profile_json column
    op.drop_column('data_sources', 'profile_json')

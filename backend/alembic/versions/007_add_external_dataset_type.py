"""Add external_dataset to datasourcetype enum

Revision ID: 007_add_external_dataset_type
Revises: 934dcd10576f
Create Date: 2024-12-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '007_add_external_dataset_type'
down_revision: Union[str, None] = '934dcd10576f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new value to datasourcetype enum
    # PostgreSQL requires ALTER TYPE to add enum values
    op.execute("ALTER TYPE datasourcetype ADD VALUE IF NOT EXISTS 'external_dataset'")


def downgrade() -> None:
    # Note: PostgreSQL doesn't support removing enum values easily
    # The external_dataset value will remain in the database
    pass

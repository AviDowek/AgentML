"""Add celery_task_id columns

Revision ID: 003_add_celery_task_id
Revises: 002_add_tasktype_enums
Create Date: 2024-01-15

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003_add_celery_task_id'
down_revision = '002_add_tasktype_enums'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add celery_task_id to synthetic_datasets
    op.add_column('synthetic_datasets', sa.Column('celery_task_id', sa.String(255), nullable=True))

    # Add celery_task_id and error_message to experiments
    op.add_column('experiments', sa.Column('celery_task_id', sa.String(255), nullable=True))
    op.add_column('experiments', sa.Column('error_message', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('synthetic_datasets', 'celery_task_id')
    op.drop_column('experiments', 'celery_task_id')
    op.drop_column('experiments', 'error_message')

"""Add file_data column to data_sources for persistent file storage.

Revision ID: add_file_data_to_data_sources
Revises: add_context_documents
Create Date: 2026-03-18
"""
from alembic import op
import sqlalchemy as sa

revision = 'add_file_data_to_data_sources'
down_revision = 'add_context_documents'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('data_sources', sa.Column('file_data', sa.LargeBinary(), nullable=True))


def downgrade():
    op.drop_column('data_sources', 'file_data')

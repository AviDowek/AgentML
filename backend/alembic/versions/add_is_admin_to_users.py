"""Add is_admin column to users table.

Revision ID: add_is_admin_to_users
Revises: add_file_data_to_data_sources
Create Date: 2026-03-19
"""
from alembic import op
import sqlalchemy as sa

revision = "add_is_admin_to_users"
down_revision = "add_file_data_to_data_sources"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("is_admin", sa.Boolean(), nullable=False, server_default="false"))
    # Set dowekavi@gmail.com as admin
    op.execute("UPDATE users SET is_admin = true WHERE email = 'dowekavi@gmail.com'")


def downgrade():
    op.drop_column("users", "is_admin")

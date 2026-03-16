"""Add users and sharing tables

Revision ID: 005_add_users_and_sharing
Revises: 004_add_conversations
Create Date: 2024-01-17

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision = '005_add_users_and_sharing'
down_revision = '004_add_conversations'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(255), nullable=True),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('google_id', sa.String(255), nullable=True, unique=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_google_id', 'users', ['google_id'])

    # Create project_shares table
    op.create_table(
        'project_shares',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('invited_email', sa.String(255), nullable=True),
        sa.Column('role', sa.String(20), nullable=False, default='viewer'),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('invite_token', sa.String(255), nullable=True, unique=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_project_shares_project_id', 'project_shares', ['project_id'])
    op.create_index('ix_project_shares_user_id', 'project_shares', ['user_id'])
    op.create_unique_constraint('uq_project_user', 'project_shares', ['project_id', 'user_id'])

    # Create synthetic_dataset_shares table
    op.create_table(
        'synthetic_dataset_shares',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('dataset_id', UUID(as_uuid=True), sa.ForeignKey('synthetic_datasets.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('invited_email', sa.String(255), nullable=True),
        sa.Column('role', sa.String(20), nullable=False, default='viewer'),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('invite_token', sa.String(255), nullable=True, unique=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_synthetic_dataset_shares_dataset_id', 'synthetic_dataset_shares', ['dataset_id'])
    op.create_index('ix_synthetic_dataset_shares_user_id', 'synthetic_dataset_shares', ['user_id'])
    op.create_unique_constraint('uq_dataset_user', 'synthetic_dataset_shares', ['dataset_id', 'user_id'])

    # Add owner_id to projects
    op.add_column('projects', sa.Column('owner_id', UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_projects_owner', 'projects', 'users', ['owner_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_projects_owner_id', 'projects', ['owner_id'])

    # Add owner_id to synthetic_datasets
    op.add_column('synthetic_datasets', sa.Column('owner_id', UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_synthetic_datasets_owner', 'synthetic_datasets', 'users', ['owner_id'], ['id'], ondelete='SET NULL')
    op.create_index('ix_synthetic_datasets_owner_id', 'synthetic_datasets', ['owner_id'])

    # Add owner_id to conversations
    op.add_column('conversations', sa.Column('owner_id', UUID(as_uuid=True), nullable=True))
    op.create_foreign_key('fk_conversations_owner', 'conversations', 'users', ['owner_id'], ['id'], ondelete='CASCADE')
    op.create_index('ix_conversations_owner_id', 'conversations', ['owner_id'])


def downgrade() -> None:
    # Remove owner_id from conversations
    op.drop_index('ix_conversations_owner_id', table_name='conversations')
    op.drop_constraint('fk_conversations_owner', 'conversations', type_='foreignkey')
    op.drop_column('conversations', 'owner_id')

    # Remove owner_id from synthetic_datasets
    op.drop_index('ix_synthetic_datasets_owner_id', table_name='synthetic_datasets')
    op.drop_constraint('fk_synthetic_datasets_owner', 'synthetic_datasets', type_='foreignkey')
    op.drop_column('synthetic_datasets', 'owner_id')

    # Remove owner_id from projects
    op.drop_index('ix_projects_owner_id', table_name='projects')
    op.drop_constraint('fk_projects_owner', 'projects', type_='foreignkey')
    op.drop_column('projects', 'owner_id')

    # Drop synthetic_dataset_shares table
    op.drop_constraint('uq_dataset_user', 'synthetic_dataset_shares', type_='unique')
    op.drop_index('ix_synthetic_dataset_shares_user_id', table_name='synthetic_dataset_shares')
    op.drop_index('ix_synthetic_dataset_shares_dataset_id', table_name='synthetic_dataset_shares')
    op.drop_table('synthetic_dataset_shares')

    # Drop project_shares table
    op.drop_constraint('uq_project_user', 'project_shares', type_='unique')
    op.drop_index('ix_project_shares_user_id', table_name='project_shares')
    op.drop_index('ix_project_shares_project_id', table_name='project_shares')
    op.drop_table('project_shares')

    # Drop users table
    op.drop_index('ix_users_google_id', table_name='users')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')

"""Add agent runs, steps, and logs tables

Revision ID: 006_add_agent_runs
Revises: 005_add_users_and_sharing
Create Date: 2024-01-18

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision = '006_add_agent_runs'
down_revision = '005_add_users_and_sharing'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create agent_runs table
    op.create_table(
        'agent_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=True),
        sa.Column('experiment_id', UUID(as_uuid=True), sa.ForeignKey('experiments.id', ondelete='SET NULL'), nullable=True),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('config_json', JSONB(), nullable=True),
        sa.Column('result_json', JSONB(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_agent_runs_project_id', 'agent_runs', ['project_id'])
    op.create_index('ix_agent_runs_experiment_id', 'agent_runs', ['experiment_id'])
    op.create_index('ix_agent_runs_status', 'agent_runs', ['status'])

    # Create agent_steps table
    op.create_table(
        'agent_steps',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('agent_run_id', UUID(as_uuid=True), sa.ForeignKey('agent_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('step_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('input_json', JSONB(), nullable=True),
        sa.Column('output_json', JSONB(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_agent_steps_agent_run_id', 'agent_steps', ['agent_run_id'])
    op.create_index('ix_agent_steps_step_type', 'agent_steps', ['step_type'])
    op.create_index('ix_agent_steps_status', 'agent_steps', ['status'])

    # Create agent_step_logs table
    op.create_table(
        'agent_step_logs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('agent_step_id', UUID(as_uuid=True), sa.ForeignKey('agent_steps.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sequence', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('message_type', sa.String(20), nullable=False, server_default='info'),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('metadata_json', JSONB(), nullable=True),
    )
    op.create_index('ix_agent_step_logs_agent_step_id', 'agent_step_logs', ['agent_step_id'])
    op.create_index('ix_agent_step_logs_sequence', 'agent_step_logs', ['agent_step_id', 'sequence'])


def downgrade() -> None:
    # Drop agent_step_logs table
    op.drop_index('ix_agent_step_logs_sequence', table_name='agent_step_logs')
    op.drop_index('ix_agent_step_logs_agent_step_id', table_name='agent_step_logs')
    op.drop_table('agent_step_logs')

    # Drop agent_steps table
    op.drop_index('ix_agent_steps_status', table_name='agent_steps')
    op.drop_index('ix_agent_steps_step_type', table_name='agent_steps')
    op.drop_index('ix_agent_steps_agent_run_id', table_name='agent_steps')
    op.drop_table('agent_steps')

    # Drop agent_runs table
    op.drop_index('ix_agent_runs_status', table_name='agent_runs')
    op.drop_index('ix_agent_runs_experiment_id', table_name='agent_runs')
    op.drop_index('ix_agent_runs_project_id', table_name='agent_runs')
    op.drop_table('agent_runs')

"""Add research cycles and lab notebook tables for project-level research memory

This migration adds:
- research_cycles: Tracks research iterations within a project
- cycle_experiments: Links research cycles to experiments
- lab_notebook_entries: Stores research notes, findings, and insights
- research_cycle_id column on agent_runs table

Revision ID: 014_add_research_cycles
Revises: 013_add_ds_agent_config
Create Date: 2024-12-11 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '014_add_research_cycles'
down_revision: Union[str, None] = '013_add_ds_agent_config'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create research_cycles table
    op.create_table(
        'research_cycles',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sequence_number', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('summary_title', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_research_cycles_project_id', 'research_cycles', ['project_id'])

    # Create cycle_experiments link table
    op.create_table(
        'cycle_experiments',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('research_cycle_id', UUID(as_uuid=True), sa.ForeignKey('research_cycles.id', ondelete='CASCADE'), nullable=False),
        sa.Column('experiment_id', UUID(as_uuid=True), sa.ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('linked_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_cycle_experiments_research_cycle_id', 'cycle_experiments', ['research_cycle_id'])
    op.create_index('ix_cycle_experiments_experiment_id', 'cycle_experiments', ['experiment_id'])

    # Create lab_notebook_entries table
    op.create_table(
        'lab_notebook_entries',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('research_cycle_id', UUID(as_uuid=True), sa.ForeignKey('research_cycles.id', ondelete='SET NULL'), nullable=True),
        sa.Column('agent_step_id', UUID(as_uuid=True), sa.ForeignKey('agent_steps.id', ondelete='SET NULL'), nullable=True),
        sa.Column('author_type', sa.String(20), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('body_markdown', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_lab_notebook_entries_project_id', 'lab_notebook_entries', ['project_id'])
    op.create_index('ix_lab_notebook_entries_research_cycle_id', 'lab_notebook_entries', ['research_cycle_id'])

    # Add research_cycle_id column to agent_runs
    op.add_column(
        'agent_runs',
        sa.Column('research_cycle_id', UUID(as_uuid=True), sa.ForeignKey('research_cycles.id', ondelete='SET NULL'), nullable=True)
    )
    op.create_index('ix_agent_runs_research_cycle_id', 'agent_runs', ['research_cycle_id'])


def downgrade() -> None:
    # Remove index and column from agent_runs
    op.drop_index('ix_agent_runs_research_cycle_id', table_name='agent_runs')
    op.drop_column('agent_runs', 'research_cycle_id')

    # Drop lab_notebook_entries table
    op.drop_index('ix_lab_notebook_entries_research_cycle_id', table_name='lab_notebook_entries')
    op.drop_index('ix_lab_notebook_entries_project_id', table_name='lab_notebook_entries')
    op.drop_table('lab_notebook_entries')

    # Drop cycle_experiments table
    op.drop_index('ix_cycle_experiments_experiment_id', table_name='cycle_experiments')
    op.drop_index('ix_cycle_experiments_research_cycle_id', table_name='cycle_experiments')
    op.drop_table('cycle_experiments')

    # Drop research_cycles table
    op.drop_index('ix_research_cycles_project_id', table_name='research_cycles')
    op.drop_table('research_cycles')

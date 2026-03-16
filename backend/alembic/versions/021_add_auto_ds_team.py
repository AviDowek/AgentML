"""Add Auto DS Team tables for autonomous ML research

This migration adds:
- auto_ds_sessions: Main session tracking for autonomous research
- auto_ds_iterations: Individual iterations within a session
- auto_ds_iteration_experiments: Links iterations to experiments
- research_insights: Structured insights discovered during analysis
- global_insights: Cross-project insights for transfer learning
- Adds parent_dataset_spec_id and lineage_json to dataset_specs for lineage tracking

Revision ID: 021_add_auto_ds_team
Revises: 36ddf1693686
Create Date: 2024-12-23 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = '021_add_auto_ds_team'
down_revision: Union[str, None] = '36ddf1693686'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add lineage columns to dataset_specs
    op.add_column(
        'dataset_specs',
        sa.Column('parent_dataset_spec_id', UUID(as_uuid=True), sa.ForeignKey('dataset_specs.id', ondelete='SET NULL'), nullable=True)
    )
    op.add_column(
        'dataset_specs',
        sa.Column('lineage_json', JSONB(), nullable=True)
    )
    op.create_index('ix_dataset_specs_parent_dataset_spec_id', 'dataset_specs', ['parent_dataset_spec_id'])

    # Create global_insights table first (referenced by research_insights)
    op.create_table(
        'global_insights',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('insight_type', sa.String(50), nullable=False),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('technical_details_json', JSONB(), nullable=True),
        sa.Column('applicable_to', JSONB(), nullable=True),
        sa.Column('task_types', JSONB(), nullable=True),
        sa.Column('data_characteristics', JSONB(), nullable=True),
        sa.Column('evidence_count', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('contradiction_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('confidence_score', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('source_project_count', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('last_validated_at', sa.DateTime(), nullable=True),
        sa.Column('times_applied', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('times_successful', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_global_insights_insight_type', 'global_insights', ['insight_type'])
    op.create_index('ix_global_insights_category', 'global_insights', ['category'])

    # Create auto_ds_sessions table
    op.create_table(
        'auto_ds_sessions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        # Stop conditions
        sa.Column('max_iterations', sa.Integer(), nullable=False, server_default='10'),
        sa.Column('accuracy_threshold', sa.Float(), nullable=True),
        sa.Column('time_budget_minutes', sa.Integer(), nullable=True),
        sa.Column('min_improvement_threshold', sa.Float(), nullable=False, server_default='0.001'),
        sa.Column('plateau_iterations', sa.Integer(), nullable=False, server_default='3'),
        # Execution config
        sa.Column('max_experiments_per_dataset', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('max_active_datasets', sa.Integer(), nullable=False, server_default='5'),
        # Progress tracking
        sa.Column('current_iteration', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('best_score', sa.Float(), nullable=True),
        sa.Column('best_experiment_id', UUID(as_uuid=True), nullable=True),
        sa.Column('total_experiments_run', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('iterations_without_improvement', sa.Integer(), nullable=False, server_default='0'),
        # Timing
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        # Links
        sa.Column('research_cycle_id', UUID(as_uuid=True), sa.ForeignKey('research_cycles.id', ondelete='SET NULL'), nullable=True),
        sa.Column('celery_task_id', sa.String(255), nullable=True),
        sa.Column('config_json', JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_auto_ds_sessions_project_id', 'auto_ds_sessions', ['project_id'])
    op.create_index('ix_auto_ds_sessions_status', 'auto_ds_sessions', ['status'])

    # Create auto_ds_iterations table
    op.create_table(
        'auto_ds_iterations',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', UUID(as_uuid=True), sa.ForeignKey('auto_ds_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('iteration_number', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(30), nullable=False, server_default='pending'),
        # Results summary
        sa.Column('experiments_planned', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('experiments_completed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('experiments_failed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('best_score_this_iteration', sa.Float(), nullable=True),
        sa.Column('best_experiment_id_this_iteration', UUID(as_uuid=True), nullable=True),
        # Phase tracking
        sa.Column('experiments_started_at', sa.DateTime(), nullable=True),
        sa.Column('experiments_completed_at', sa.DateTime(), nullable=True),
        sa.Column('analysis_started_at', sa.DateTime(), nullable=True),
        sa.Column('analysis_completed_at', sa.DateTime(), nullable=True),
        sa.Column('strategy_started_at', sa.DateTime(), nullable=True),
        sa.Column('strategy_completed_at', sa.DateTime(), nullable=True),
        # Output
        sa.Column('analysis_summary_json', JSONB(), nullable=True),
        sa.Column('strategy_decisions_json', JSONB(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_auto_ds_iterations_session_id', 'auto_ds_iterations', ['session_id'])

    # Create auto_ds_iteration_experiments link table
    op.create_table(
        'auto_ds_iteration_experiments',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('iteration_id', UUID(as_uuid=True), sa.ForeignKey('auto_ds_iterations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('experiment_id', UUID(as_uuid=True), sa.ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('dataset_spec_id', UUID(as_uuid=True), sa.ForeignKey('dataset_specs.id', ondelete='SET NULL'), nullable=True),
        sa.Column('experiment_variant', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('experiment_hypothesis', sa.Text(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('rank_in_iteration', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_auto_ds_iteration_experiments_iteration_id', 'auto_ds_iteration_experiments', ['iteration_id'])
    op.create_index('ix_auto_ds_iteration_experiments_experiment_id', 'auto_ds_iteration_experiments', ['experiment_id'])
    op.create_index('ix_auto_ds_iteration_experiments_dataset_spec_id', 'auto_ds_iteration_experiments', ['dataset_spec_id'])

    # Create research_insights table
    op.create_table(
        'research_insights',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', UUID(as_uuid=True), sa.ForeignKey('auto_ds_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('project_id', UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('iteration_id', UUID(as_uuid=True), sa.ForeignKey('auto_ds_iterations.id', ondelete='SET NULL'), nullable=True),
        sa.Column('insight_type', sa.String(50), nullable=False),
        sa.Column('confidence', sa.String(20), nullable=False, server_default='low'),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('insight_data_json', JSONB(), nullable=True),
        sa.Column('evidence_count', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('supporting_experiments', JSONB(), nullable=True),
        sa.Column('contradicting_experiments', JSONB(), nullable=True),
        sa.Column('is_tested', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('test_result', sa.Text(), nullable=True),
        sa.Column('promoted_to_global', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('global_insight_id', UUID(as_uuid=True), sa.ForeignKey('global_insights.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_research_insights_session_id', 'research_insights', ['session_id'])
    op.create_index('ix_research_insights_project_id', 'research_insights', ['project_id'])
    op.create_index('ix_research_insights_insight_type', 'research_insights', ['insight_type'])


def downgrade() -> None:
    # Drop research_insights table
    op.drop_index('ix_research_insights_insight_type', table_name='research_insights')
    op.drop_index('ix_research_insights_project_id', table_name='research_insights')
    op.drop_index('ix_research_insights_session_id', table_name='research_insights')
    op.drop_table('research_insights')

    # Drop auto_ds_iteration_experiments table
    op.drop_index('ix_auto_ds_iteration_experiments_dataset_spec_id', table_name='auto_ds_iteration_experiments')
    op.drop_index('ix_auto_ds_iteration_experiments_experiment_id', table_name='auto_ds_iteration_experiments')
    op.drop_index('ix_auto_ds_iteration_experiments_iteration_id', table_name='auto_ds_iteration_experiments')
    op.drop_table('auto_ds_iteration_experiments')

    # Drop auto_ds_iterations table
    op.drop_index('ix_auto_ds_iterations_session_id', table_name='auto_ds_iterations')
    op.drop_table('auto_ds_iterations')

    # Drop auto_ds_sessions table
    op.drop_index('ix_auto_ds_sessions_status', table_name='auto_ds_sessions')
    op.drop_index('ix_auto_ds_sessions_project_id', table_name='auto_ds_sessions')
    op.drop_table('auto_ds_sessions')

    # Drop global_insights table
    op.drop_index('ix_global_insights_category', table_name='global_insights')
    op.drop_index('ix_global_insights_insight_type', table_name='global_insights')
    op.drop_table('global_insights')

    # Remove lineage columns from dataset_specs
    op.drop_index('ix_dataset_specs_parent_dataset_spec_id', table_name='dataset_specs')
    op.drop_column('dataset_specs', 'lineage_json')
    op.drop_column('dataset_specs', 'parent_dataset_spec_id')

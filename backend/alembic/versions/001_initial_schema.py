"""Initial schema with all core tables

Revision ID: 001_initial
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE tasktype AS ENUM ('classification', 'regression', 'multiclass')")
    op.execute("CREATE TYPE projectstatus AS ENUM ('draft', 'active', 'archived')")
    op.execute("CREATE TYPE datasourcetype AS ENUM ('file_upload', 'database', 's3', 'api')")
    op.execute("CREATE TYPE experimentstatus AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled')")
    op.execute("CREATE TYPE metricdirection AS ENUM ('minimize', 'maximize')")
    op.execute("CREATE TYPE trialstatus AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled')")
    op.execute("CREATE TYPE modelstatus AS ENUM ('trained', 'candidate', 'shadow', 'production', 'retired')")
    op.execute("CREATE TYPE policytype AS ENUM ('scheduled', 'metric_threshold', 'data_drift', 'manual')")

    # Projects table
    op.create_table(
        'projects',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('task_type', postgresql.ENUM('classification', 'regression', 'multiclass', name='tasktype', create_type=False), nullable=True),
        sa.Column('status', postgresql.ENUM('draft', 'active', 'archived', name='projectstatus', create_type=False), nullable=False, server_default='draft'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )

    # Data sources table
    op.create_table(
        'data_sources',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('type', postgresql.ENUM('file_upload', 'database', 's3', 'api', name='datasourcetype', create_type=False), nullable=False),
        sa.Column('config_json', postgresql.JSONB(), nullable=True),
        sa.Column('schema_summary', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_data_sources_project_id', 'data_sources', ['project_id'])

    # Dataset specs table
    op.create_table(
        'dataset_specs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('data_sources_json', postgresql.JSONB(), nullable=True),
        sa.Column('target_column', sa.String(255), nullable=True),
        sa.Column('feature_columns', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('filters_json', postgresql.JSONB(), nullable=True),
        sa.Column('spec_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_dataset_specs_project_id', 'dataset_specs', ['project_id'])

    # Experiments table
    op.create_table(
        'experiments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('dataset_spec_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('dataset_specs.id', ondelete='SET NULL'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', postgresql.ENUM('pending', 'running', 'completed', 'failed', 'cancelled', name='experimentstatus', create_type=False), nullable=False, server_default='pending'),
        sa.Column('primary_metric', sa.String(100), nullable=True),
        sa.Column('metric_direction', postgresql.ENUM('minimize', 'maximize', name='metricdirection', create_type=False), nullable=True),
        sa.Column('experiment_plan_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_experiments_project_id', 'experiments', ['project_id'])
    op.create_index('ix_experiments_dataset_spec_id', 'experiments', ['dataset_spec_id'])

    # Trials table
    op.create_table(
        'trials',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('experiment_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('experiments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('variant_name', sa.String(255), nullable=False),
        sa.Column('data_split_strategy', sa.String(100), nullable=True),
        sa.Column('automl_config', postgresql.JSONB(), nullable=True),
        sa.Column('status', postgresql.ENUM('pending', 'running', 'completed', 'failed', 'cancelled', name='trialstatus', create_type=False), nullable=False, server_default='pending'),
        sa.Column('metrics_json', postgresql.JSONB(), nullable=True),
        sa.Column('best_model_ref', sa.String(500), nullable=True),
        sa.Column('logs_location', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_trials_experiment_id', 'trials', ['experiment_id'])

    # Model versions table
    op.create_table(
        'model_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('experiment_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('experiments.id', ondelete='SET NULL'), nullable=True),
        sa.Column('trial_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('trials.id', ondelete='SET NULL'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(100), nullable=True),
        sa.Column('artifact_location', sa.String(500), nullable=True),
        sa.Column('metrics_json', postgresql.JSONB(), nullable=True),
        sa.Column('feature_importances_json', postgresql.JSONB(), nullable=True),
        sa.Column('status', postgresql.ENUM('trained', 'candidate', 'shadow', 'production', 'retired', name='modelstatus', create_type=False), nullable=False, server_default='trained'),
        sa.Column('serving_config_json', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_model_versions_project_id', 'model_versions', ['project_id'])
    op.create_index('ix_model_versions_experiment_id', 'model_versions', ['experiment_id'])
    op.create_index('ix_model_versions_trial_id', 'model_versions', ['trial_id'])

    # Retraining policies table
    op.create_table(
        'retraining_policies',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('policy_type', postgresql.ENUM('scheduled', 'metric_threshold', 'data_drift', 'manual', name='policytype', create_type=False), nullable=False),
        sa.Column('schedule_cron', sa.String(100), nullable=True),
        sa.Column('metric_thresholds_json', postgresql.JSONB(), nullable=True),
        sa.Column('last_retrain_at', sa.DateTime(), nullable=True),
        sa.Column('next_run_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_retraining_policies_project_id', 'retraining_policies', ['project_id'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('retraining_policies')
    op.drop_table('model_versions')
    op.drop_table('trials')
    op.drop_table('experiments')
    op.drop_table('dataset_specs')
    op.drop_table('data_sources')
    op.drop_table('projects')

    # Drop enum types
    op.execute("DROP TYPE policytype")
    op.execute("DROP TYPE modelstatus")
    op.execute("DROP TYPE trialstatus")
    op.execute("DROP TYPE metricdirection")
    op.execute("DROP TYPE experimentstatus")
    op.execute("DROP TYPE datasourcetype")
    op.execute("DROP TYPE projectstatus")
    op.execute("DROP TYPE tasktype")

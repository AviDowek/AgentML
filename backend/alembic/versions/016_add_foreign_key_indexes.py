"""Add indexes on foreign keys for query performance.

Revision ID: 016
Revises: 015_add_app_settings
Create Date: 2025-01-01

Foreign keys without indexes cause slow joins and lookups.
This migration adds missing indexes to all foreign key columns.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "016_fk_indexes"
down_revision = "015_add_app_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Dataset specs - project_id
    op.create_index(
        "ix_dataset_specs_project_id",
        "dataset_specs",
        ["project_id"],
        unique=False,
        if_not_exists=True,
    )

    # Experiments - project_id, dataset_spec_id, parent_experiment_id
    op.create_index(
        "ix_experiments_project_id",
        "experiments",
        ["project_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_experiments_dataset_spec_id",
        "experiments",
        ["dataset_spec_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_experiments_parent_experiment_id",
        "experiments",
        ["parent_experiment_id"],
        unique=False,
        if_not_exists=True,
    )

    # Trials - experiment_id
    op.create_index(
        "ix_trials_experiment_id",
        "trials",
        ["experiment_id"],
        unique=False,
        if_not_exists=True,
    )

    # Model versions - project_id, experiment_id, trial_id
    op.create_index(
        "ix_model_versions_project_id",
        "model_versions",
        ["project_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_model_versions_experiment_id",
        "model_versions",
        ["experiment_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_model_versions_trial_id",
        "model_versions",
        ["trial_id"],
        unique=False,
        if_not_exists=True,
    )

    # Agent runs - project_id, experiment_id
    op.create_index(
        "ix_agent_runs_project_id",
        "agent_runs",
        ["project_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_runs_experiment_id",
        "agent_runs",
        ["experiment_id"],
        unique=False,
        if_not_exists=True,
    )

    # Agent steps - agent_run_id
    op.create_index(
        "ix_agent_steps_agent_run_id",
        "agent_steps",
        ["agent_run_id"],
        unique=False,
        if_not_exists=True,
    )

    # Agent step logs - agent_step_id
    op.create_index(
        "ix_agent_step_logs_agent_step_id",
        "agent_step_logs",
        ["agent_step_id"],
        unique=False,
        if_not_exists=True,
    )

    # Data sources - project_id
    op.create_index(
        "ix_data_sources_project_id",
        "data_sources",
        ["project_id"],
        unique=False,
        if_not_exists=True,
    )

    # Validation samples - model_version_id
    op.create_index(
        "ix_validation_samples_model_version_id",
        "validation_samples",
        ["model_version_id"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_validation_samples_model_version_id", table_name="validation_samples")
    op.drop_index("ix_data_sources_project_id", table_name="data_sources")
    op.drop_index("ix_agent_step_logs_agent_step_id", table_name="agent_step_logs")
    op.drop_index("ix_agent_steps_agent_run_id", table_name="agent_steps")
    op.drop_index("ix_agent_runs_experiment_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_project_id", table_name="agent_runs")
    op.drop_index("ix_model_versions_trial_id", table_name="model_versions")
    op.drop_index("ix_model_versions_experiment_id", table_name="model_versions")
    op.drop_index("ix_model_versions_project_id", table_name="model_versions")
    op.drop_index("ix_trials_experiment_id", table_name="trials")
    op.drop_index("ix_experiments_parent_experiment_id", table_name="experiments")
    op.drop_index("ix_experiments_dataset_spec_id", table_name="experiments")
    op.drop_index("ix_experiments_project_id", table_name="experiments")
    op.drop_index("ix_dataset_specs_project_id", table_name="dataset_specs")

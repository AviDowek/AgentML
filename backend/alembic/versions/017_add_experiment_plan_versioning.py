"""Add experiment plan versioning fields.

Revision ID: 017
Revises: 016_add_foreign_key_indexes
Create Date: 2025-01-01

Adds versioning support for experiment plans to track changes over time.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "017_plan_version"
down_revision = "016_fk_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add plan version to experiments
    op.add_column(
        "experiments",
        sa.Column(
            "plan_version",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )

    # Add plan history JSON to track changes
    op.add_column(
        "experiments",
        sa.Column(
            "plan_history_json",
            sa.JSON(),
            nullable=True,
        ),
    )

    # Add version to dataset specs for tracking spec changes
    op.add_column(
        "dataset_specs",
        sa.Column(
            "version",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )


def downgrade() -> None:
    op.drop_column("dataset_specs", "version")
    op.drop_column("experiments", "plan_history_json")
    op.drop_column("experiments", "plan_version")

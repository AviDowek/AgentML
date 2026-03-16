"""Add time-based task metadata fields to dataset_specs.

Revision ID: 018
Revises: 017_plan_version
Create Date: 2025-01-14

Adds fields to track time-series/temporal prediction tasks:
- is_time_based: Whether task predicts future behavior
- time_column: Datetime column for temporal ordering
- entity_id_column: ID column for panel/longitudinal data
- prediction_horizon: Human-readable horizon (e.g., "1d", "5d")
- target_positive_class: Positive class value for classification
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "018_time_metadata"
down_revision = "017_plan_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_time_based boolean flag
    op.add_column(
        "dataset_specs",
        sa.Column(
            "is_time_based",
            sa.Boolean(),
            nullable=False,
            server_default="0",  # Default to False
        ),
    )

    # Add time_column for datetime ordering
    op.add_column(
        "dataset_specs",
        sa.Column(
            "time_column",
            sa.String(255),
            nullable=True,
        ),
    )

    # Add entity_id_column for panel data
    op.add_column(
        "dataset_specs",
        sa.Column(
            "entity_id_column",
            sa.String(255),
            nullable=True,
        ),
    )

    # Add prediction_horizon for human-readable horizon
    op.add_column(
        "dataset_specs",
        sa.Column(
            "prediction_horizon",
            sa.String(100),
            nullable=True,
        ),
    )

    # Add target_positive_class for classification tasks
    op.add_column(
        "dataset_specs",
        sa.Column(
            "target_positive_class",
            sa.String(255),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("dataset_specs", "target_positive_class")
    op.drop_column("dataset_specs", "prediction_horizon")
    op.drop_column("dataset_specs", "entity_id_column")
    op.drop_column("dataset_specs", "time_column")
    op.drop_column("dataset_specs", "is_time_based")

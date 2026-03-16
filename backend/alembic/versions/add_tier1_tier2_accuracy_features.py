"""add_tier1_tier2_accuracy_features

Revision ID: tier1_tier2_features
Revises: d3cb8e708205
Create Date: 2025-12-24 21:00:00.000000

Adds:
- DYNAMIC execution mode
- ValidationStrategy enum and column
- Feature flags for Tier 1 (feature engineering, ensemble, ablation)
- Feature flags for Tier 2 (diverse configs)
- Validation settings (num_seeds, cv_folds)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'tier1_tier2_features'
down_revision: Union[str, None] = 'd3cb8e708205'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add 'dynamic' to the existing executionmode enum
    # PostgreSQL requires special handling for enum modifications
    op.execute("ALTER TYPE executionmode ADD VALUE IF NOT EXISTS 'dynamic'")

    # Create the validation strategy enum type
    validation_strategy_enum = sa.Enum('standard', 'robust', 'strict', name='validationstrategy')
    validation_strategy_enum.create(op.get_bind(), checkfirst=True)

    # Add validation_strategy column
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'validation_strategy',
            sa.Enum('standard', 'robust', 'strict', name='validationstrategy'),
            nullable=False,
            server_default='standard'
        )
    )

    # Add dynamic mode settings
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'dynamic_experiments_per_cycle',
            sa.Integer(),
            nullable=False,
            server_default='1'
        )
    )

    # Add validation settings
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'validation_num_seeds',
            sa.Integer(),
            nullable=False,
            server_default='1'
        )
    )
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'validation_cv_folds',
            sa.Integer(),
            nullable=False,
            server_default='5'
        )
    )

    # Add Tier 1 feature flags
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'enable_feature_engineering',
            sa.Boolean(),
            nullable=False,
            server_default='true'
        )
    )
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'enable_ensemble',
            sa.Boolean(),
            nullable=False,
            server_default='true'
        )
    )
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'enable_ablation',
            sa.Boolean(),
            nullable=False,
            server_default='true'
        )
    )

    # Add Tier 2 feature flags
    op.add_column(
        'auto_ds_sessions',
        sa.Column(
            'enable_diverse_configs',
            sa.Boolean(),
            nullable=False,
            server_default='true'
        )
    )


def downgrade() -> None:
    # Remove Tier 2 feature flags
    op.drop_column('auto_ds_sessions', 'enable_diverse_configs')

    # Remove Tier 1 feature flags
    op.drop_column('auto_ds_sessions', 'enable_ablation')
    op.drop_column('auto_ds_sessions', 'enable_ensemble')
    op.drop_column('auto_ds_sessions', 'enable_feature_engineering')

    # Remove validation settings
    op.drop_column('auto_ds_sessions', 'validation_cv_folds')
    op.drop_column('auto_ds_sessions', 'validation_num_seeds')

    # Remove dynamic mode settings
    op.drop_column('auto_ds_sessions', 'dynamic_experiments_per_cycle')

    # Remove validation_strategy column
    op.drop_column('auto_ds_sessions', 'validation_strategy')

    # Drop the validation strategy enum type
    validation_strategy_enum = sa.Enum('standard', 'robust', 'strict', name='validationstrategy')
    validation_strategy_enum.drop(op.get_bind(), checkfirst=True)

    # Note: Cannot easily remove 'dynamic' from executionmode enum in PostgreSQL
    # The enum value will remain but won't be used after downgrade

"""Add max_debate_rounds column to agent_runs.

Revision ID: 020_debate_rounds
Revises: 019_orchestration
Create Date: 2025-01-16

Adds configurable max debate rounds field.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "020_debate_rounds"
down_revision = "019_orchestration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add max_debate_rounds column with default of 3
    op.add_column(
        "agent_runs",
        sa.Column(
            "max_debate_rounds",
            sa.Integer(),
            nullable=False,
            server_default="3",
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "max_debate_rounds")

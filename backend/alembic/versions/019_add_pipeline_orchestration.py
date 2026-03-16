"""Add pipeline orchestration and debate system columns to agent_runs.

Revision ID: 019
Revises: d8927df943d8
Create Date: 2025-01-16

Adds fields to support:
- Project Manager orchestration mode
- Gemini critique debate system
- OpenAI judge selection
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = "019_orchestration"
down_revision = "d8927df943d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add orchestration_mode enum column
    op.add_column(
        "agent_runs",
        sa.Column(
            "orchestration_mode",
            sa.String(50),
            nullable=False,
            server_default="sequential",
        ),
    )

    # Add debate_mode enum column
    op.add_column(
        "agent_runs",
        sa.Column(
            "debate_mode",
            sa.String(50),
            nullable=False,
            server_default="disabled",
        ),
    )

    # Add judge_model for OpenAI model selection
    op.add_column(
        "agent_runs",
        sa.Column(
            "judge_model",
            sa.String(100),
            nullable=True,
        ),
    )

    # Add debate_transcript_json for storing debate history
    op.add_column(
        "agent_runs",
        sa.Column(
            "debate_transcript_json",
            sa.JSON(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "debate_transcript_json")
    op.drop_column("agent_runs", "judge_model")
    op.drop_column("agent_runs", "debate_mode")
    op.drop_column("agent_runs", "orchestration_mode")

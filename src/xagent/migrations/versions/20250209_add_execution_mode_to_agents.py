"""add execution_mode to agents table

Revision ID: 20250209_add_execution_mode
Revises: fix_agent_status_case
Create Date: 2025-02-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20250209_add_execution_mode_to_agents"
down_revision: Union[str, None] = "fix_agent_status_case"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add execution_mode column to agents table with default value "react"
    op.add_column(
        "agents",
        sa.Column(
            "execution_mode",
            sa.String(length=20),
            nullable=False,
            server_default="react",
        ),
    )


def downgrade() -> None:
    # Remove execution_mode column from agents table
    op.drop_column("agents", "execution_mode")

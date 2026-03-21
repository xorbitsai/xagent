"""add agent_id to tasks table

Revision ID: 20250209_add_agent_id_to_tasks
Revises: 20250209_add_suggested_prompts
Create Date: 2025-02-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20250209_add_agent_id_to_tasks"
down_revision: Union[str, None] = "20250209_add_suggested_prompts_to_agents"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add agent_id column to tasks table
    op.add_column("tasks", sa.Column("agent_id", sa.Integer(), nullable=True))


def downgrade() -> None:
    # Remove agent_id column
    op.drop_column("tasks", "agent_id")

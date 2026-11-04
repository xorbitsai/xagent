"""merge two alembic heads on main

20260703_add_user_mcpserver_env (per-user MCP env) and
20260703_backfill_strip_other_tool_category both branch from
20260625_add_user_skills_tables, leaving the migration graph with two heads.
This is a no-op merge revision that reunites them into a single head.

Revision ID: 20260704_merge_alembic_heads
Revises: 20260703_add_user_mcpserver_env, 20260703_backfill_strip_other_tool_category
Create Date: 2026-07-04 00:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "20260704_merge_alembic_heads"
down_revision: Union[str, Sequence[str], None] = (
    "20260703_add_user_mcpserver_env",
    "20260703_backfill_strip_other_tool_category",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

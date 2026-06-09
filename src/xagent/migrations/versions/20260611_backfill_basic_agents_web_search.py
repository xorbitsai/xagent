"""backfill web_search category for existing basic agents

Revision ID: 20260611_backfill_basic_agents_web_search
Revises: 20260602_add_agent_share_fields
Create Date: 2026-06-11 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260611_backfill_basic_agents_web_search"
down_revision: Union[str, tuple[str, str], None] = "20260602_add_agent_share_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "agents" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("agents")}
    if "tool_categories" not in existing_columns:
        return

    agents = sa.table(
        "agents",
        sa.column("id", sa.Integer),
        sa.column("tool_categories", sa.JSON),
    )

    rows = bind.execute(sa.select(agents.c.id, agents.c.tool_categories))
    for row in rows.mappings():
        categories = row["tool_categories"]
        if not isinstance(categories, list):
            continue
        if "basic" not in categories or "web_search" in categories:
            continue
        bind.execute(
            agents.update()
            .where(agents.c.id == row["id"])
            .values(tool_categories=[*categories, "web_search"])
        )


def downgrade() -> None:
    # Data migration is intentionally not reversed: removing web_search here
    # could strip categories that users explicitly selected after upgrade.
    pass

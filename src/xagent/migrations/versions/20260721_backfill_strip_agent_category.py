"""strip legacy 'agent' entries from persisted agent tool_categories

'agent' was demoted from an assignable tool category to a
Workforce-only concern (issue #802): selecting it gave an agent
delegation access to every published agent in the account, which is
now configured through Workforce instead. The runtime normalizer
(ToolSelectionSpec.from_raw) already strips 'agent' on every read, so
this migration does not change any agent's effective tool access -- it
just removes the now provably-inert 'agent' entries from the stored
column and logs the affected agent ids for traceability, instead of
that data being re-detected and re-warned-about on every read forever.
Mirrors 20260703_backfill_strip_other_tool_category.

Revision ID: 20260721_backfill_strip_agent_category
Revises: 20260721_merge_migration_heads
Create Date: 2026-07-21 00:00:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260721_backfill_strip_agent_category"
down_revision: Union[str, None] = "20260721_merge_migration_heads"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger(__name__)


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

    affected_ids = []
    rows = bind.execute(sa.select(agents.c.id, agents.c.tool_categories))
    for row in rows.mappings():
        categories = row["tool_categories"]
        if not isinstance(categories, list) or "agent" not in categories:
            continue
        affected_ids.append(row["id"])
        bind.execute(
            agents.update()
            .where(agents.c.id == row["id"])
            .values(tool_categories=[c for c in categories if c != "agent"])
        )

    if affected_ids:
        logger.warning(
            "Stripped legacy 'agent' tool_categories entry from %d agent(s): %r",
            len(affected_ids),
            affected_ids,
        )


def downgrade() -> None:
    # Data migration is intentionally not reversed: account-wide agent
    # delegation via the 'agent' category is withdrawn product behavior
    # (issue #802), so there is nothing meaningful to restore.
    pass

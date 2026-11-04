"""backfill sdk task visibility

Revision ID: 20260624_backfill_sdk_task_visibility
Revises: 20260616_add_agent_triggers
Create Date: 2026-06-24 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260624_backfill_sdk_task_visibility"
down_revision: str | tuple[str, str] | None = "20260616_add_agent_triggers"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "tasks" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("tasks")}
    if not {"source", "is_visible"}.issubset(existing_columns):
        return

    tasks = sa.table(
        "tasks",
        sa.column("source", sa.String(length=20)),
        sa.column("is_visible", sa.Boolean()),
    )
    bind.execute(
        tasks.update()
        .where(
            tasks.c.source == "sdk",
            tasks.c.is_visible.is_(True),
        )
        .values(is_visible=False)
    )


def downgrade() -> None:
    # Data migration is intentionally not reversed: after upgrade, hiding SDK
    # tasks is the product invariant and user-created visible tasks are
    # distinguished by source != "sdk".
    pass

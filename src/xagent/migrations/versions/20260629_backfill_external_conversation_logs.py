"""backfill external conversation logs

Historical REST API tasks are moved out of the main task list by setting
``is_visible=False`` for existing ``source='sdk'`` rows. Widget and shared-link
tasks with legacy channel names are also reclassified as hidden external
conversation logs.

Revision ID: 20260629_backfill_external_conversation_logs
Revises: 20260627_seed_meta_connectors
Create Date: 2026-06-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260629_backfill_external_conversation_logs"
down_revision: str | tuple[str, str] | None = "20260627_seed_meta_connectors"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_WIDGET_CHANNEL_NAME = "Web Widget"
LEGACY_SHARED_LINK_CHANNEL_NAME = "Shared Agent"


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
        sa.column("channel_name", sa.String(length=100)),
    )

    bind.execute(
        tasks.update()
        .where(
            tasks.c.source == "sdk",
            tasks.c.is_visible.is_(True),
        )
        .values(is_visible=False)
    )

    if "channel_name" not in existing_columns:
        return

    legacy_public_source = sa.or_(
        tasks.c.source.is_(None), tasks.c.source == "internal"
    )
    bind.execute(
        tasks.update()
        .where(
            legacy_public_source,
            tasks.c.is_visible.is_(True),
            tasks.c.channel_name == LEGACY_WIDGET_CHANNEL_NAME,
        )
        .values(source="widget", is_visible=False)
    )
    bind.execute(
        tasks.update()
        .where(
            legacy_public_source,
            tasks.c.is_visible.is_(True),
            tasks.c.channel_name == LEGACY_SHARED_LINK_CHANNEL_NAME,
        )
        .values(source="shared_link", is_visible=False)
    )


def downgrade() -> None:
    # Irreversible data migration: upgrade() rewrites matching tasks.source from
    # NULL/internal to widget/shared_link and flips matching visible external
    # tasks to hidden. The original NULL-vs-internal value is not retained, so
    # operators must restore those rows manually if a rollback needs old data.
    pass

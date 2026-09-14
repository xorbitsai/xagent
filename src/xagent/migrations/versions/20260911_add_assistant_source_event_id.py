"""add source_event_id column to task_chat_messages

Records which assistant trace event a transcript row was written from, so
historical replay can drop the trace twin of a question instead of printing
the same clarification form again on every reconnect (#2292).

Revision ID: 20260911_assistant_source_event
Revises: 20260909_seed_shopify_mcp_app
Create Date: 2026-09-11 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "20260911_assistant_source_event"
down_revision: Union[str, None] = "20260909_seed_shopify_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Deliberately not backfilled. A row's originating event cannot be
    # recovered from its text alone: consecutive rounds routinely ask the same
    # question verbatim, so a text backfill would hand some rows another
    # round's identity. Replay pairs those NULL rows by re-deriving the
    # transcript text from each trace event instead, which is exact.
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if "task_chat_messages" not in inspector.get_table_names():
        # Base table doesn't exist yet; nothing to do for this migration.
        return

    existing_columns = {
        col["name"] for col in inspector.get_columns("task_chat_messages")
    }
    if "source_event_id" not in existing_columns:
        op.add_column(
            "task_chat_messages",
            sa.Column("source_event_id", sa.String(255), nullable=True),
        )


def downgrade() -> None:
    from alembic import context

    bind = context.get_bind()
    inspector = Inspector.from_engine(bind)

    if "task_chat_messages" not in inspector.get_table_names():
        return

    existing_columns = {
        col["name"] for col in inspector.get_columns("task_chat_messages")
    }
    if "source_event_id" in existing_columns:
        # Plain drop_column, matching 20260522_add_turn_id_to_task_chat_messages.
        # batch_alter_table would rebuild the table on SQLite, putting the named
        # unique index uq_task_chat_messages_task_role_turn_id at risk.
        op.drop_column("task_chat_messages", "source_event_id")

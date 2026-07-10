"""add retry-safe chat message delivery state

Revision ID: 20260710_add_chat_delivery_state
Revises: 20260710_a2a_task_index
Create Date: 2026-07-10 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260710_add_chat_delivery_state"
down_revision: Union[str, None] = "20260710_a2a_task_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_chat_messages"
INDEX = "uq_task_chat_messages_task_role_turn_id"


def _columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE)}


def _indexes() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(TABLE)}


def upgrade() -> None:
    if not _columns():
        return
    if "delivery_status" not in _columns():
        op.add_column(TABLE, sa.Column("delivery_status", sa.String(32)))

    # Old code allowed duplicate turn ids. Preserve every transcript row but
    # detach duplicate identities before installing the cross-worker guard.
    op.execute(
        sa.text(
            "UPDATE task_chat_messages SET turn_id = NULL "
            "WHERE turn_id IS NOT NULL AND id NOT IN ("
            "SELECT MIN(id) FROM task_chat_messages "
            "WHERE turn_id IS NOT NULL GROUP BY task_id, role, turn_id)"
        )
    )
    if INDEX not in _indexes():
        op.create_index(
            INDEX,
            TABLE,
            ["task_id", "role", "turn_id"],
            unique=True,
        )


def downgrade() -> None:
    if not _columns():
        return
    if INDEX in _indexes():
        op.drop_index(INDEX, table_name=TABLE)
    if "delivery_status" in _columns():
        op.drop_column(TABLE, "delivery_status")

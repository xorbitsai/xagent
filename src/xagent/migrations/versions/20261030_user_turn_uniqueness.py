"""Limit chat turn uniqueness to user input rows.

Revision ID: 20261030_user_turn_uniqueness
Revises: 20261029_update_chartmogul_description
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261030_user_turn_uniqueness"
down_revision: Union[str, None] = "20261029_update_chartmogul_description"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_chat_messages"
INDEX = "uq_task_chat_messages_task_role_turn_id"


def _has_table() -> bool:
    return TABLE in sa.inspect(op.get_bind()).get_table_names()


def _has_index() -> bool:
    if not _has_table():
        return False
    return any(
        index["name"] == INDEX for index in sa.inspect(op.get_bind()).get_indexes(TABLE)
    )


def upgrade() -> None:
    if not _has_table():
        return
    if _has_index():
        op.drop_index(INDEX, table_name=TABLE)
    op.create_index(
        INDEX,
        TABLE,
        ["task_id", "role", "turn_id"],
        unique=True,
        sqlite_where=sa.text("role = 'user'"),
        postgresql_where=sa.text("role = 'user'"),
    )


def downgrade() -> None:
    if not _has_table():
        return
    if _has_index():
        op.drop_index(INDEX, table_name=TABLE)
    # Assistant rows can legitimately share a turn after this revision. The
    # old schema represented only one of them as turn-addressable.
    connection = op.get_bind()
    connection.execute(
        sa.text(
            "UPDATE task_chat_messages SET turn_id = NULL "
            "WHERE role = 'assistant' AND turn_id IS NOT NULL AND id NOT IN ("
            "SELECT MIN(id) FROM task_chat_messages "
            "WHERE role = 'assistant' AND turn_id IS NOT NULL "
            "GROUP BY task_id, role, turn_id)"
        )
    )
    op.create_index(
        INDEX,
        TABLE,
        ["task_id", "role", "turn_id"],
        unique=True,
    )

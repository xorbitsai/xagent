"""add versioned task execution control state

Revision ID: 20260711_task_control_state
Revises: 20260710_add_chat_delivery_state
Create Date: 2026-07-11

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260711_task_control_state"
down_revision: Union[str, None] = "20260710_add_chat_delivery_state"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
RUN_ID_INDEX = "ix_tasks_run_id"


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
    columns = _columns()
    if not columns:
        return
    if "run_id" not in columns:
        op.add_column(TABLE, sa.Column("run_id", sa.String(64), nullable=True))
    if "state_version" not in columns:
        op.add_column(
            TABLE,
            sa.Column(
                "state_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
    if "control_state" not in columns:
        op.add_column(
            TABLE,
            sa.Column(
                "control_state",
                sa.String(32),
                nullable=False,
                server_default="idle",
            ),
        )
        if "status" in columns:
            op.execute(
                sa.text(
                    "UPDATE tasks SET control_state = CASE "
                    "WHEN LOWER(CAST(status AS VARCHAR)) = 'running' THEN 'running' "
                    "WHEN LOWER(CAST(status AS VARCHAR)) = 'paused' THEN 'paused' "
                    "WHEN LOWER(CAST(status AS VARCHAR)) = 'waiting_for_user' "
                    "THEN 'waiting_for_user' "
                    "WHEN LOWER(CAST(status AS VARCHAR)) = 'completed' "
                    "THEN 'completed' "
                    "WHEN LOWER(CAST(status AS VARCHAR)) = 'failed' THEN 'failed' "
                    "ELSE 'idle' END"
                )
            )
    if RUN_ID_INDEX not in _indexes():
        op.create_index(RUN_ID_INDEX, TABLE, ["run_id"])


def downgrade() -> None:
    if not _columns():
        return
    if RUN_ID_INDEX in _indexes():
        op.drop_index(RUN_ID_INDEX, table_name=TABLE)
    if "control_state" in _columns():
        op.drop_column(TABLE, "control_state")
    if "state_version" in _columns():
        op.drop_column(TABLE, "state_version")
    if "run_id" in _columns():
        op.drop_column(TABLE, "run_id")

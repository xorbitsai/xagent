"""add composite index on trace_events (task_id, event_type)

Checkpoint pruning and latest-checkpoint loading both filter trace_events
by task_id + event_type on every checkpoint write/resume; without an index
each lookup scans the table, which keeps growing.

Revision ID: 20260711_add_trace_events_task_idx
Revises: 20260711_task_control_state
Create Date: 2026-07-11 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260711_add_trace_events_task_idx"
down_revision: Union[str, None] = "20260711_task_control_state"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "trace_events"
INDEX = "ix_trace_events_task_id_event_type"


def _indexes() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set()
    return {index["name"] for index in inspector.get_indexes(TABLE)}


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return
    if INDEX not in _indexes():
        op.create_index(INDEX, TABLE, ["task_id", "event_type"])


def downgrade() -> None:
    if INDEX in _indexes():
        op.drop_index(INDEX, table_name=TABLE)

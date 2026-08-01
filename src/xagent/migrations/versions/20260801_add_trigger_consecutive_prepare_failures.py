"""add consecutive_prepare_failures to agent_triggers

Persists the consecutive prepare_trigger_run failure counter on the trigger
row itself instead of the in-process dict scan_due_scheduled_triggers used
to keep (see triggers.py's _PREPARE_FAILURE_SURFACE_THRESHOLD comment).
scan_due_scheduled_triggers runs from at least two genuinely separate OS
processes concurrently (the backend's in-process asyncio dispatcher and a
separate Celery beat/worker scan), so a per-process dict could split one
trigger's failures across processes such that neither ever reached the
surface threshold, and a later successful prepare handled by a different
process than the one that set the failed badge would never clear it.

Revision ID: 20260801_add_trigger_consecutive_prepare_failures
Revises: 20260731_seed_granola_mcp_app
Create Date: 2026-08-01

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260801_add_trigger_consecutive_prepare_failures"
down_revision: Union[str, None] = "20260731_seed_granola_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "agent_triggers"
COLUMN = "consecutive_prepare_failures"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns(TABLE)}
    if COLUMN not in existing_columns:
        op.add_column(
            TABLE,
            sa.Column(COLUMN, sa.Integer(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns(TABLE)}
    if COLUMN in existing_columns:
        op.drop_column(TABLE, COLUMN)

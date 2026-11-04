"""add workforce_runs.last_activity_at, keyed by the preview-run reaper

created_at alone can't distinguish a genuinely-abandoned preview run from
one that's mid-conversation but has simply been open a long time:
sync_workforce_run_status (workforce_runtime.py) resets status/completed_at
on every turn but never touches created_at, so the preview-run reaper's
staleness check could permanently cancel an actively-used preview session
(PR review round 8, F-NEW-1). last_activity_at is bumped by that same sync
call whenever it actually changes the row -- i.e. once per turn -- and the
reaper now keys staleness off it instead.

Revision ID: 20260802_add_workforce_run_last_activity_at
Revises: 20260729_make_workforce_run_workforce_id_nullable
Create Date: 2026-08-02

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260802_add_workforce_run_last_activity_at"
down_revision: Union[str, None] = "20260729_make_workforce_run_workforce_id_nullable"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "workforce_runs"
COLUMN = "last_activity_at"

# WorkforceRun has FKs to workforces/tasks/users; without resolve_fks=False,
# batch mode's FK reflection can raise NoSuchTableError on a migration-only
# database where a referenced table isn't yet visible (see the same fix in
# 20260724_add_upload_source_to_uploaded_files.py).
BATCH_REFLECT_KWARGS = {"resolve_fks": False}


def _column_names(inspector: sa.engine.reflection.Inspector, table: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return
    if COLUMN in _column_names(inspector, TABLE):
        return

    with op.batch_alter_table(TABLE, reflect_kwargs=BATCH_REFLECT_KWARGS) as batch_op:
        batch_op.add_column(
            sa.Column(
                COLUMN,
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            )
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return
    if COLUMN not in _column_names(inspector, TABLE):
        return

    with op.batch_alter_table(TABLE, reflect_kwargs=BATCH_REFLECT_KWARGS) as batch_op:
        batch_op.drop_column(COLUMN)

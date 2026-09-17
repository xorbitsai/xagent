"""make workforce_runs.workforce_id nullable for ephemeral preview runs

Workforce "test before saving" now runs a manager + inline worker configs
that were never persisted as a Workforce row. The run itself is still
persisted as a hidden WorkforceRun + Task pair (mirroring how single-agent
preview persists a hidden Task), but there is no Workforce row to point
``workforce_id`` at, so the column must accept NULL.

Revision ID: 20260729_make_workforce_run_workforce_id_nullable
Revises: 20260731_seed_aws_mcp_app
Create Date: 2026-07-29

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "20260729_make_workforce_run_workforce_id_nullable"
down_revision: Union[str, None] = "20260731_seed_aws_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "workforce_runs"
COLUMN = "workforce_id"

# WorkforceRun has FKs to workforces/tasks/users; without resolve_fks=False,
# batch mode's FK reflection can raise NoSuchTableError on a migration-only
# database where a referenced table isn't yet visible (see the same fix in
# 20260724_add_upload_source_to_uploaded_files.py).
BATCH_REFLECT_KWARGS = {"resolve_fks": False}


def _column_nullable(inspector: Inspector, table: str, column: str) -> bool | None:
    for col in inspector.get_columns(table):
        if col["name"] == column:
            return bool(col["nullable"])
    return None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return

    if _column_nullable(inspector, TABLE, COLUMN) is False:
        with op.batch_alter_table(
            TABLE, reflect_kwargs=BATCH_REFLECT_KWARGS
        ) as batch_op:
            batch_op.alter_column(COLUMN, existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return

    # Ephemeral preview runs have no Workforce to backfill; they cannot
    # survive the NOT NULL restore, so drop them. Their paired hidden Task
    # rows are deliberately left as orphans rather than deleted here: four
    # child tables (DAGExecution, TraceEvent, TraceMessageBlob,
    # TraceCheckpointBlob -- see src/xagent/web/models/task.py) reference
    # tasks.id with NO ``ondelete`` clause, so a direct DELETE FROM tasks
    # raises a Postgres FK violation for any preview that executed even one
    # turn (the common case, not an edge case -- every turn writes a
    # TraceEvent). Cascade-deleting those four tables safely, in dependency
    # order, is out of scope for a nullable-column migration; a hidden
    # (``is_visible=False``) orphaned preview Task is harmless data debris,
    # not a functional issue.
    bind.execute(sa.text(f"DELETE FROM {TABLE} WHERE {COLUMN} IS NULL"))

    if _column_nullable(inspector, TABLE, COLUMN) is True:
        with op.batch_alter_table(
            TABLE, reflect_kwargs=BATCH_REFLECT_KWARGS
        ) as batch_op:
            batch_op.alter_column(COLUMN, existing_type=sa.Integer(), nullable=False)

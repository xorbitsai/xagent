"""add exact-row checkpoint anchor to tasks

Adds ``tasks.last_checkpoint_trace_event_id``, an integer pointer to the
``trace_events`` row that ``last_checkpoint_event_id`` (the legacy UUID
string column) names. On PostgreSQL the column is backed by a named foreign
key; on SQLite the migration adds the column only -- ``ALTER TABLE ADD
CONSTRAINT`` is not renderable through Alembic's SQLite batch mode without a
full table rebuild, so an upgraded SQLite database has no DB-level FK here.
A database built fresh through ``Base.metadata.create_all()`` (new installs,
tests) gets the FK from the model directly, regardless of dialect. That
divergence between fresh and upgraded SQLite databases is permanent under
the current migration set: this revision is the head and no later revision
reconciles the two shapes. On an upgraded SQLite database the pointer's
delete protection is therefore the application-level clearing order in
``task_deletion.py``, which nulls both pointer columns before deleting the
task's ``trace_events`` rows -- not the database.

Existing rows are backfilled by resolving the legacy string column against
the row it names, but only where that resolution is unambiguous: zero or
multiple matching trace_events rows leave the new column NULL rather than
guessing or aborting the migration.

On PostgreSQL ``create_foreign_key`` takes an ACCESS EXCLUSIVE lock on
``tasks`` and validates every existing row before returning, and the
backfill's correlated subquery matches ``trace_events`` on
``(task_id, event_id)`` where only ``task_id`` is indexed. On a large
deployment both steps should be scheduled with that lock window in mind; a
``NOT VALID`` constraint followed by a separate ``VALIDATE CONSTRAINT``
would move the validation out of the lock window if that becomes necessary.

Revision ID: 20260804_add_task_checkpoint_trace_event_anchor
Revises: 20260729_add_gmail_audience_grace
Create Date: 2026-08-04

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260804_add_task_checkpoint_trace_event_anchor"
down_revision: Union[str, None] = "20260729_add_gmail_audience_grace"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
TRACE_TABLE = "trace_events"
COLUMN = "last_checkpoint_trace_event_id"
LEGACY_COLUMN = "last_checkpoint_event_id"
FK_NAME = "fk_tasks_last_checkpoint_trace_event_id"

# One correlated-subquery UPDATE. The subquery is wrapped in COUNT/CASE so it
# always returns a single scalar row -- a bare correlated subquery would
# raise on multiple matches instead of resolving to NULL, and this backfill
# must never abort the migration on ambiguous or missing legacy data (a
# concurrently deleted target task resolves the same way: zero matches).
BACKFILL_SQL = sa.text(
    f"""
    UPDATE {TABLE}
    SET {COLUMN} = (
        SELECT CASE WHEN COUNT(*) = 1 THEN MIN(te.id) ELSE NULL END
        FROM trace_events te
        WHERE te.task_id = {TABLE}.id
          AND te.event_id = {TABLE}.{LEGACY_COLUMN}
          AND te.build_id IS NULL
          AND te.event_type = 'system_update_general'
    )
    WHERE {COLUMN} IS NULL
      AND {LEGACY_COLUMN} IS NOT NULL
    """
)


def _table_exists(name: str = TABLE) -> bool:
    inspector = sa.inspect(op.get_bind())
    return inspector.has_table(name)


def _columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {col["name"] for col in inspector.get_columns(TABLE)}


def _fk_names() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    return {fk["name"] for fk in inspector.get_foreign_keys(TABLE)}


def upgrade() -> None:
    # tasks is created by Base.metadata.create_all() in production, not by
    # a migration in this repo; a from-scratch (bare) database has no tasks
    # table yet when migrations run, so this is a no-op there.
    if not _table_exists():
        return

    bind = op.get_bind()
    is_postgresql = bind.dialect.name == "postgresql"

    if COLUMN not in _columns():
        op.add_column(TABLE, sa.Column(COLUMN, sa.Integer(), nullable=True))

    if not _table_exists(TRACE_TABLE):
        return

    if is_postgresql and FK_NAME not in _fk_names():
        op.create_foreign_key(
            FK_NAME,
            TABLE,
            TRACE_TABLE,
            [COLUMN],
            ["id"],
        )

    op.execute(BACKFILL_SQL)


def downgrade() -> None:
    if not _table_exists():
        return

    bind = op.get_bind()
    is_postgresql = bind.dialect.name == "postgresql"

    if COLUMN not in _columns():
        return

    if is_postgresql:
        if FK_NAME in _fk_names():
            op.drop_constraint(FK_NAME, TABLE, type_="foreignkey")
        op.drop_column(TABLE, COLUMN)
    else:
        # SQLite renders this column's foreign key inline in CREATE TABLE
        # even under use_alter=True (a fresh install builds tasks with
        # create_all, not a migration), and refuses to drop a column an
        # inline foreign key names. A full table rebuild removes both
        # together; nothing else can. The migration connection already runs
        # with SQLite foreign keys off for exactly this reason -- see
        # _migration_connection in src/xagent/db/migration.py -- and
        # re-checks for new violations afterwards.
        with op.batch_alter_table(TABLE, recreate="always") as batch_op:
            batch_op.drop_column(COLUMN)

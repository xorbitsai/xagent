"""Record what the retention purge expired, for readers to report (#2565).

* ``expired_task_tombstones``: one content-free row per conversation-expired
  task, holding only the inputs of each surface's access predicate.
* ``tasks.traces_expired_at``: when retention last removed a task's trace.
* ``trigger_runs.task_expired_at`` / ``workforce_runs.task_expired_at``: the
  run's task expired, kept apart from the run's execution status.

Every added column is nullable with no server default, so on PostgreSQL each
``ADD COLUMN`` is a catalog-only change: no table rewrite and no backfill,
which matters for ``tasks``. NULL is also the right value for every existing
row, since nothing has been expired before this revision.

The tombstone table references ``users``, ``agents`` and ``workforces``, so it
waits for all three the way the other task-adjacent revisions wait for their
parents; a database that has not got them yet gets the table from a later
upgrade or from ``create_all``.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260926_expired_task_tombstones"
down_revision = "20260916_durable_create_operations"
branch_labels = None
depends_on = None

TABLE = "expired_task_tombstones"
REFERENCED_TABLES = ("users", "agents", "workforces")
#: (table, column) pairs this revision adds.
ADDED_COLUMNS = (
    ("tasks", "traces_expired_at"),
    ("trigger_runs", "task_expired_at"),
    ("workforce_runs", "task_expired_at"),
)
INDEXED_COLUMNS = ("user_id", "agent_id", "workforce_id")


def _has_column(inspector: sa.Inspector, table: str, column: str) -> bool:
    return any(c["name"] == column for c in inspector.get_columns(table))


def upgrade() -> None:
    offline = op.get_context().as_sql
    inspector = None if offline else sa.inspect(op.get_bind())

    for table, column in ADDED_COLUMNS:
        if inspector is not None and (
            not inspector.has_table(table) or _has_column(inspector, table, column)
        ):
            continue
        op.add_column(
            table, sa.Column(column, sa.DateTime(timezone=True), nullable=True)
        )

    if inspector is not None and (
        inspector.has_table(TABLE)
        or not all(inspector.has_table(name) for name in REFERENCED_TABLES)
    ):
        return
    op.create_table(
        TABLE,
        sa.Column("task_id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "agent_id",
            sa.Integer(),
            sa.ForeignKey("agents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "workforce_id",
            sa.Integer(),
            sa.ForeignKey("workforces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source", sa.String(20), nullable=True),
        sa.Column("is_visible", sa.Boolean(), nullable=False),
        sa.Column("is_channel_plumbing", sa.Boolean(), nullable=False),
        sa.Column("task_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expired_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in INDEXED_COLUMNS:
        op.create_index(f"ix_{TABLE}_{column}", TABLE, [column])


def downgrade() -> None:
    offline = op.get_context().as_sql
    inspector = None if offline else sa.inspect(op.get_bind())

    if inspector is None or inspector.has_table(TABLE):
        op.drop_table(TABLE)
    for table, column in ADDED_COLUMNS:
        if inspector is not None and (
            not inspector.has_table(table) or not _has_column(inspector, table, column)
        ):
            continue
        op.drop_column(table, column)

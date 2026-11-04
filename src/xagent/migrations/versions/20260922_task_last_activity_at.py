"""Add the conversation retention anchor ``tasks.last_activity_at`` (#2562).

The column is added without a server default on purpose. A default would make
``ALTER TABLE`` stamp every pre-existing row with this migration's own clock --
the exact value the backfill below exists to avoid, and one that would push
every historical task's expiry out by the full retention period.

Three properties the backfill is shaped around:

* **It derives the anchor only from columns that exist.** ``tasks`` predates
  most of its current columns, and a partially-migrated or Alembic-only
  database can present a ``tasks`` table with no ``created_at`` at all (see
  the legacy shape built in tests/migration/test_migration.py). Naming a
  column that is not there fails the statement and takes the whole upgrade
  chain down with it, so each source is probed before it is used.

* **It leaves a row NULL rather than inventing a timestamp**, and NULL is
  much weaker protection than it looks. ``retention_anchor()`` is
  ``COALESCE(last_activity_at, created_at)``, so a NULL anchor beside a live
  ``created_at`` is not refused -- it expires the task from its creation date,
  which for a long conversation is far too early. NULL is only harmless where
  ``tasks.created_at`` does not exist either, and there it is harmless for a
  blunt reason rather than a designed one: the predicate cannot run against
  such a schema at all, because the mapped column it selects is not there.

  So the invariant this revision must hold is not "NULL is safe" but "NULL
  must never mean the backfill has not run". Every path that adds the column
  therefore also fills it before the revision can be stamped: the online
  upgrade runs the paged backfill below, and the offline script carries its
  own single-statement one ahead of the ``alembic_version`` update Alembic
  appends.

* **It pages forward by primary key.** Selecting the remaining NULL set on
  every pass looks equivalent and is not: a row written to NULL stays in that
  set, so the loop re-selects it forever and the revision dies at the batch
  ceiling. The cursor advances whatever value was written, which also keeps
  the walk linear instead of re-scanning the rows already done.

On PostgreSQL the backfill runs inside ``autocommit_block()``, on the online
path and the offline one alike. Offline rendering honours it -- on a
transactional-DDL dialect Alembic emits the COMMIT into the generated script
-- so an operator applying that script gets BEGIN / ALTER / COMMIT and then
the UPDATE outside the transaction, rather than holding the ALTER's lock for
a full pass over ``tasks``.

Without it
the ``ADD COLUMN`` and every batch share one transaction (``env.py`` sets
``transaction_per_migration`` for this dialect), and ``ALTER TABLE ... ADD
COLUMN`` holds ACCESS EXCLUSIVE on ``tasks`` until that transaction commits --
so every task read and write in the deployment, API calls and lease
heartbeats alike, would block for the length of the backfill rather than for
the length of the ALTER. The block commits the ALTER first and then commits
each batch, which is also what makes a re-run meaningful: an interrupted
backfill leaves committed progress and NULLs behind, and the next upgrade
finishes them. SQLite keeps the single-transaction behaviour, because there
the whole migration chain shares one transaction and breaking it mid-chain
would trade a lock problem this dialect does not have for a partial-upgrade
problem it does.

The backfill runs on every upgrade rather than only when the column is
created, because a database whose column arrived from ``create_all`` still
needs filling, and because an interrupted PostgreSQL run must be able to
resume.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260922_task_last_activity_at"
down_revision = "20260921_word_contract"
branch_labels = None
depends_on = None

#: Rows per UPDATE. Bounded so one statement cannot take an unbounded lock
#: footprint over a multi-million-row ``tasks`` table.
BACKFILL_BATCH_SIZE = 5000

#: The anchor expression the offline script carries. Shared with the test that
#: applies the rendered script, so the two cannot drift.
OFFLINE_ANCHOR_SQL = (
    "COALESCE((SELECT MAX(m.created_at) FROM task_chat_messages m "
    "WHERE m.task_id = tasks.id), tasks.created_at)"
)


def _has_column(inspector: sa.Inspector, table: str, column: str) -> bool:
    if not inspector.has_table(table):
        return False
    return any(c["name"] == column for c in inspector.get_columns(table))


def _anchor_expression(*, with_messages: bool, with_created_at: bool) -> str | None:
    """The SQL for one row's anchor, using only sources that exist.

    ``None`` means no source is available, which is not an error. Such a
    database has no instant to offer, and it is also one the retention
    predicate cannot query: ``retention_anchor()`` selects ``tasks.created_at``
    through the mapped column, so a schema without it raises rather than
    returning an anchor. Leaving these rows NULL is therefore moot, not a
    safety property to rely on elsewhere.
    """
    message_max = "(SELECT MAX(m.created_at) FROM task_chat_messages m WHERE m.task_id = tasks.id)"
    if with_messages and with_created_at:
        return f"COALESCE({message_max}, tasks.created_at)"
    if with_messages:
        return message_max
    if with_created_at:
        return "tasks.created_at"
    return None


def _backfill(connection: sa.Connection, *, anchor_sql: str) -> None:
    """Fill every NULL anchor, paging forward by primary key.

    Terminates because the cursor only moves forward: a batch that writes
    NULL still advances it, where a re-select of the NULL set would hand the
    same row back on every pass. Deliberately returns nothing -- a count
    would have to say whether it meant rows selected or rows written, and
    since the guard below can skip a row those differ.
    """
    select_batch = sa.text(
        "SELECT id FROM tasks "
        "WHERE id > :after AND last_activity_at IS NULL "
        "ORDER BY id LIMIT :batch"
    )
    # ``AND last_activity_at IS NULL`` repeats the SELECT's predicate on
    # purpose; it is not redundant. The ids were chosen by an earlier
    # statement, and on PostgreSQL each batch commits separately, so a
    # transcript writer can advance a selected task's anchor in between. A
    # blocked UPDATE re-checks its search condition against the *target* row
    # once the lock is released, but the ``MAX()`` subquery reads
    # ``task_chat_messages`` -- other rows -- and keeps the original command
    # snapshot, so without this guard the backfill would overwrite the newer
    # anchor with a value computed before that message existed. With it, a row
    # someone else has already anchored is simply skipped: touches only ever
    # move forward, so their value is the better one.
    #
    # noqa: S608 is on the UPDATE below: `anchor_sql` comes only from
    # _anchor_expression above, which returns one of three literals. No
    # caller-supplied text reaches it, and every value is bound.
    update_batch = sa.text(
        f"UPDATE tasks SET last_activity_at = {anchor_sql} "  # noqa: S608
        "WHERE id IN :ids AND last_activity_at IS NULL"
    ).bindparams(sa.bindparam("ids", expanding=True))

    after = -1
    while True:
        ids = [
            row[0]
            for row in connection.execute(
                select_batch, {"after": after, "batch": BACKFILL_BATCH_SIZE}
            )
        ]
        if not ids:
            return
        connection.execute(update_batch, {"ids": ids})
        after = ids[-1]


def upgrade() -> None:
    # ``op.get_context().as_sql`` rather than ``context.is_offline_mode()``:
    # the latter needs the EnvironmentContext proxy, which only an env.py run
    # establishes, so it cannot be reached when a test drives upgrade()
    # through Operations.context(). Both read the same flag.
    if op.get_context().as_sql:
        # Offline renders the backfill as well as the DDL, and the order
        # matters: Alembic appends the ``alembic_version`` update after this,
        # so filling here means the revision cannot be stamped complete while
        # historical anchors are still NULL. Emitting only the ADD COLUMN was
        # the round-1 defect -- a later online ``upgrade head`` would see the
        # revision applied, skip the backfill, and leave every historical task
        # reading as expired from its creation date.
        #
        # Unconditional and unpaged, because offline has a MockConnection:
        # nothing can be inspected and no rowcount can be read back. The
        # online path probes its sources for the legacy ``tasks`` shapes it
        # can meet; this one cannot, and does not need to for the range it
        # serves -- a database at the immediate predecessor carries the modern
        # schema. Generating offline SQL from an ancient revision is already
        # unavailable in this repository for an unrelated reason: 20260317
        # reflects through the MockConnection and raises first.
        op.add_column(
            "tasks",
            sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=True),
        )
        # Same reason as the online path: commit the ALTER before the scan so
        # its ACCESS EXCLUSIVE lock is held for the DDL rather than for a full
        # pass over ``tasks``. ``autocommit_block`` is honoured in offline
        # rendering too -- on a transactional-DDL dialect it emits the COMMIT
        # into the script (alembic/runtime/migration.py, emit_commit under
        # ``as_sql``), so PostgreSQL gets BEGIN / ALTER / COMMIT and then the
        # UPDATE outside that transaction.
        with op.get_context().autocommit_block():
            op.execute(
                # noqa: S608 -- OFFLINE_ANCHOR_SQL is a module literal.
                "UPDATE tasks SET last_activity_at = "  # noqa: S608
                f"{OFFLINE_ANCHOR_SQL} WHERE last_activity_at IS NULL"
            )
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    # ``tasks`` is metadata-owned and can be absent in Alembic-only runs.
    if not inspector.has_table("tasks"):
        return
    if not _has_column(inspector, "tasks", "last_activity_at"):
        op.add_column(
            "tasks",
            sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=True),
        )
        inspector = sa.inspect(bind)

    anchor_sql = _anchor_expression(
        with_messages=_has_column(inspector, "task_chat_messages", "created_at"),
        with_created_at=_has_column(inspector, "tasks", "created_at"),
    )
    if anchor_sql is None:
        return

    if bind.dialect.name == "postgresql":
        # Commits the ADD COLUMN, and with it the ACCESS EXCLUSIVE lock,
        # before the batches start. See the module docstring.
        with op.get_context().autocommit_block():
            _backfill(bind, anchor_sql=anchor_sql)
    else:
        _backfill(bind, anchor_sql=anchor_sql)


def downgrade() -> None:
    # Unlike upgrade(), this renders offline: dropping the column needs no
    # row work, so the emitted statement is the whole of the revision.
    if op.get_context().as_sql:
        op.drop_column("tasks", "last_activity_at")
        return

    inspector = sa.inspect(op.get_bind())
    if _has_column(inspector, "tasks", "last_activity_at"):
        op.drop_column("tasks", "last_activity_at")

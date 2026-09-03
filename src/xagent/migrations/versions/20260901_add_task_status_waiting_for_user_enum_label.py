"""add WAITING_FOR_USER to the native enum backing tasks.status

Revision ID: 20260901_taskstatus_waiting_for_user
Revises: 20260828_terminal_cmd_events
Create Date: 2026-09-01 00:00:00.000000

``TaskStatus.WAITING_FOR_USER`` entered the application enum on 2026-05-14.
On PostgreSQL ``tasks.status`` is a native enum type whose labels are fixed
when the type is created, and ``Base.metadata.create_all`` does not add
labels to a type that already exists -- so a database initialized before
that date still carries the five-label type, and every write of the new
label fails at the write. This revision adds the label to the type
``tasks.status`` is actually declared with.

It resolves that type the way ``check_task_status_enum_drift``
(``web/models/task.py``) resolves it -- through the column, not through
whichever type happens to be named ``taskstatus`` -- because the two must
agree: an asymmetric ``search_path`` can otherwise point the repair at one
type and the check at another.

Only this one label, deliberately. A revision runs once and is then recorded
in ``alembic_version``, so a "add whatever TaskStatus has that the database
lacks" body would still not add a member introduced after this revision had
already run. Each new ``TaskStatus`` member needs its own revision.

Non-PostgreSQL backends store the column as a plain string and have no
native type to repair, so this revision is a no-op there.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260901_taskstatus_waiting_for_user"
down_revision: Union[str, None] = "20260828_terminal_cmd_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "tasks"
COLUMN = "status"
LABEL = "WAITING_FOR_USER"
# Offline (``--sql``) rendering cannot reflect anything, so it emits the
# unqualified name -- the one an operator's own psql session resolves the
# same way the application does.
OFFLINE_TYPE_NAME = "taskstatus"

# ``regtype`` renders the OID as a name that resolves back to the same type
# under the current ``search_path``, schema-qualifying it exactly when an
# unqualified name would resolve somewhere else. That makes the rendering
# safe to put straight back into ``ALTER TYPE``, and it comes from the system
# catalog rather than from any caller. ``typtype = 'e'`` keeps a ``status``
# column that is not an enum out of the result entirely.
_COLUMN_ENUM_TYPE_SQL = sa.text(
    "SELECT a.atttypid::regtype::text "
    "FROM pg_catalog.pg_attribute a "
    "JOIN pg_catalog.pg_type t ON t.oid = a.atttypid "
    "WHERE a.attrelid = pg_catalog.to_regclass(:table_name) "
    "AND a.attname = :column_name "
    "AND t.typtype = 'e'"
)

_LABEL_PRESENT_SQL = sa.text(
    "SELECT 1 "
    "FROM pg_catalog.pg_attribute a "
    "JOIN pg_catalog.pg_enum e ON e.enumtypid = a.atttypid "
    "WHERE a.attrelid = pg_catalog.to_regclass(:table_name) "
    "AND a.attname = :column_name "
    "AND e.enumlabel = :label"
)


def _add_value_sql(type_name: str) -> str:
    return f"ALTER TYPE {type_name} ADD VALUE IF NOT EXISTS '{LABEL}'"


def upgrade() -> None:
    context = op.get_context()
    if context.dialect.name != "postgresql":
        return

    if context.as_sql:
        with context.autocommit_block():
            op.execute(_add_value_sql(OFFLINE_TYPE_NAME))
        return

    bind = op.get_bind()
    params = {"table_name": TABLE, "column_name": COLUMN}
    type_name = bind.execute(_COLUMN_ENUM_TYPE_SQL, params).scalar_one_or_none()
    if type_name is None:
        # Either there is no tasks table yet -- a brand-new database, where
        # create_all builds the type with every current label right after the
        # migrations run -- or status is not a native enum. ALTER TYPE
        # repairs neither, and check_task_status_enum_drift is the place that
        # reports the second one.
        return

    if bind.execute(_LABEL_PRESENT_SQL, {**params, "label": LABEL}).first() is not None:
        return

    # An autocommit block rather than the surrounding per-migration
    # transaction. database_startup_lock holds a *session*-level advisory lock
    # and commits before yielding precisely so Alembic can do this
    # (db/migration.py), so the intermediate commit does not release it. What
    # it buys: PostgreSQL rejects a write of a label added earlier in the same
    # transaction ("unsafe use of new value"), and that error aborts the
    # transaction, taking the ALTER TYPE down with it. Committing the label on
    # its own puts that failure mode out of reach instead of leaving it to
    # whatever a later edit adds below this line.
    with context.autocommit_block():
        op.execute(_add_value_sql(type_name))


def downgrade() -> None:
    # One-way by construction: PostgreSQL has no ALTER TYPE ... DROP VALUE.
    # Removing one label means building a replacement type, rewriting every
    # tasks.status value and the column default onto it under an exclusive
    # lock, and dropping the old type -- to reach a state whose only property
    # is that it cannot store WAITING_FOR_USER. A label no process writes
    # costs nothing to leave in place.
    pass

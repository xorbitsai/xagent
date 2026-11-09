"""record which access an MCP OAuth flow was started through

``mcp_oauth_flow_states.access_path`` is ``VARCHAR(16) NOT NULL`` with the server
default ``association``. ``association`` is a flow started by a user who holds a
``user_mcpservers`` row; ``team_member`` is a flow started by a team member who
holds none and reaches the connector through the team overlay. A named CHECK
admits ``team_member`` only when ``association_lifecycle_generation`` is NULL,
because the flow row itself is the lifecycle anchor on that path, and only when
``resource_owner_key`` is the starting user's own default,
``xagent:user:{user_id}``. Existing rows all read ``association``; those with a
NULL generation still fail closed in the callback, as before.

Upgrade guards, in this order:

1. A database without ``mcp_oauth_flow_states`` is left alone.
2. A database that already has the column (``create_all`` ran first) does not
   get it added again.
3. The CHECK is added only when it is not there yet, so a second run is a no-op.
4. On PostgreSQL every statement runs through ``run_ddl_with_lock_retry`` and the
   CHECK is added ``NOT VALID``: every existing row is ``association`` and
   satisfies it, so no table scan is needed under the lock, and every later
   write is checked. SQLite adds the column with one ``ADD COLUMN`` whose
   definition carries the named CHECK, online and in ``--sql`` output alike;
   only a table that already has the column but not the CHECK is rebuilt
   through a batch to add it.

Downgrade drops the CHECK before the column, except the SQLite ``--sql``
downgrade, which emits only ``DROP COLUMN``. On SQLite the online
downgrade rebuilds the table through a batch, which works wherever the CHECK
sits; the ``--sql`` downgrade is a plain ``DROP COLUMN``, which SQLite accepts
only while the CHECK is attached to the column, as the add-column path and
``create_all`` both leave it. A table whose CHECK the batch path added needs the
online downgrade.

Revision ID: 20261105_mcp_oauth_flow_access_path
Revises: 20261105_task_structured_output_record
Create Date: 2026-11-05

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.migrations.ddl_lock_retry import (
    offline_ddl_with_lock_timeout,
    run_ddl_with_lock_retry,
)

revision: str = "20261105_mcp_oauth_flow_access_path"
down_revision: Union[str, None] = "20261105_task_structured_output_record"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "mcp_oauth_flow_states"
COLUMN = "access_path"
CONSTRAINT = "ck_mcp_oauth_flow_states_access_path"
CHECK_CONDITION = (
    "(access_path = 'association') OR "
    "(access_path = 'team_member' AND association_lifecycle_generation IS NULL "
    "AND resource_owner_key = 'xagent:user:' || CAST(user_id AS VARCHAR))"
)
_LOCK_TIMEOUT = "5s"
# Seconds to wait after the first and the second lock timeout.
_LOCK_RETRY_DELAYS = (2.0, 4.0)

_ADD_COLUMN_SQL = (
    f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} "
    "VARCHAR(16) NOT NULL DEFAULT 'association'"
)
_ADD_CHECK_SQL = (
    f"ALTER TABLE {TABLE} ADD CONSTRAINT {CONSTRAINT} "
    f"CHECK ({CHECK_CONDITION}) NOT VALID"
)
_DROP_CHECK_SQL = f"ALTER TABLE {TABLE} DROP CONSTRAINT IF EXISTS {CONSTRAINT}"
_DROP_COLUMN_SQL = f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}"


def _column() -> sa.Column:
    return sa.Column(
        COLUMN,
        sa.String(16),
        sa.CheckConstraint(CHECK_CONDITION, name=CONSTRAINT),
        nullable=False,
        server_default="association",
    )


def _columns() -> set[str] | None:
    """The column names of the table now, or None without it; never cached."""
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return None
    return {column["name"] for column in inspector.get_columns(TABLE)}


def _has_check() -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(
        item.get("name") == CONSTRAINT
        for item in inspector.get_check_constraints(TABLE)
    )


def _apply(statement: str) -> None:
    run_ddl_with_lock_retry(
        statement,
        table=TABLE,
        lock_timeout=_LOCK_TIMEOUT,
        retry_delays=_LOCK_RETRY_DELAYS,
    )


def upgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"
    if context.as_sql:
        if is_postgresql:
            for statement in (_ADD_COLUMN_SQL, _ADD_CHECK_SQL):
                offline_ddl_with_lock_timeout(statement, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.add_column(TABLE, _column())
        return

    columns = _columns()
    if columns is None:
        return
    if is_postgresql:
        if COLUMN not in columns:
            _apply(_ADD_COLUMN_SQL)
        if not _has_check():
            _apply(_ADD_CHECK_SQL)
        return

    if COLUMN not in columns:
        # The CHECK travels with the column definition.
        op.add_column(TABLE, _column())
    elif not _has_check():
        with op.batch_alter_table(TABLE) as batch:
            batch.create_check_constraint(CONSTRAINT, CHECK_CONDITION)


def downgrade() -> None:
    context = op.get_context()
    is_postgresql = context.dialect.name == "postgresql"
    if context.as_sql:
        if is_postgresql:
            for statement in (_DROP_CHECK_SQL, _DROP_COLUMN_SQL):
                offline_ddl_with_lock_timeout(statement, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.drop_column(TABLE, COLUMN)
        return

    columns = _columns()
    if columns is None or COLUMN not in columns:
        return
    if is_postgresql:
        _apply(_DROP_CHECK_SQL)
        _apply(_DROP_COLUMN_SQL)
        return
    with op.batch_alter_table(TABLE) as batch:
        if _has_check():
            batch.drop_constraint(CONSTRAINT, type_="check")
        batch.drop_column(COLUMN)

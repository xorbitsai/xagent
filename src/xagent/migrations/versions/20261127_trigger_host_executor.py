"""Add agent_triggers.executor for host-executed trigger runs.

Revision ID: 20261127_trigger_host_executor
Revises: 20261127_mixpanel_description

MIGRATION_OVERLAP_REVIEW: additive nullable column on agent_triggers. The older
release inserts and updates triggers without it and its reads are unaffected;
the new release treats NULL as "no host executor", which is exactly the older
behavior (the engine starts its default agent task). Nothing writes the column
until a host application registers an executor and marks triggers with it, so
there is no backfill and nothing races the older release's writes. PostgreSQL
DDL runs under a five-second lock_timeout with bounded retries; SQLite uses
the normal add/drop column path. A trigger run started on one release and
finished on the other is unaffected because only triggers with a non-NULL
executor take the new path, and none exist before the host opts in.
"""

import sqlalchemy as sa
from alembic import op

from xagent.migrations.ddl_lock_retry import (
    offline_ddl_with_lock_timeout,
    run_ddl_with_lock_retry,
)

revision = "20261127_trigger_host_executor"
down_revision = "20261127_mixpanel_description"
branch_labels = None
depends_on = None

TABLE = "agent_triggers"
COLUMN = "executor"
_LOCK_TIMEOUT = "5s"
_LOCK_RETRY_DELAYS = (2.0, 4.0)
_ADD_SQL = f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR(64)"
_DROP_SQL = f"ALTER TABLE {TABLE} DROP COLUMN {COLUMN}"


def _has_column(inspector: sa.Inspector) -> bool:
    return COLUMN in {column["name"] for column in inspector.get_columns(TABLE)}


def upgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            offline_ddl_with_lock_timeout(_ADD_SQL, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.add_column(TABLE, sa.Column(COLUMN, sa.String(64), nullable=True))
        return
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names() or _has_column(inspector):
        return
    if context.dialect.name == "postgresql":
        run_ddl_with_lock_retry(
            _ADD_SQL,
            table=TABLE,
            lock_timeout=_LOCK_TIMEOUT,
            retry_delays=_LOCK_RETRY_DELAYS,
        )
    else:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(64), nullable=True))


def downgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            offline_ddl_with_lock_timeout(_DROP_SQL, lock_timeout=_LOCK_TIMEOUT)
        else:
            op.drop_column(TABLE, COLUMN)
        return
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names() or not _has_column(inspector):
        return
    if context.dialect.name == "postgresql":
        run_ddl_with_lock_retry(
            _DROP_SQL,
            table=TABLE,
            lock_timeout=_LOCK_TIMEOUT,
            retry_delays=_LOCK_RETRY_DELAYS,
        )
    elif context.dialect.name == "sqlite":
        with op.batch_alter_table(TABLE) as batch:
            batch.drop_column(COLUMN)
    else:
        op.drop_column(TABLE, COLUMN)

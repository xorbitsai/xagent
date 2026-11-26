"""Snapshot the authorization callback URI on each OAuth flow and grant.

Revision ID: 20261126_connector_oauth_redirect_uri
Revises: 20261119_shopify_store_locks

Why two nullable columns:

* ``mcp_oauth_flow_states.redirect_uri`` pins the URI sent to the
  authorization server for one remote-MCP flow. The shared MCP client row can
  be updated by a later connect; the code exchange must still repeat the URI
  the provider saw at authorization time.
* ``user_oauth.redirect_uri`` pins the URI used to mint a builtin grant.
  Deputy repeats ``redirect_uri`` on refresh, so a grant authorized before the
  callback-origin switch must keep refreshing with its old URI.

MIGRATION_OVERLAP_REVIEW: both changes are additive nullable columns. The older
release inserts and updates neither column and its reads are unaffected; the
new release treats a NULL as "authorized before this revision" and falls back
to the registered callback. There is no backfill, so no backfill can race the
older release's writes. PostgreSQL DDL runs under a five-second lock_timeout
with bounded retries; SQLite uses the normal add/drop column path. A flow
started on the older release and finished on the newer one exchanges with the
older configured URI (the state or flow row has no snapshot), while a flow
started after the new callback origin is enabled must finish on a worker that
runs this revision. Therefore the production sequence is: deploy this
compatible revision everywhere, then enable
``XAGENT_CONNECTOR_OAUTH_CALLBACK_BASE_URL``.
"""

import sqlalchemy as sa
from alembic import op

from xagent.migrations.ddl_lock_retry import (
    offline_ddl_with_lock_timeout,
    run_ddl_with_lock_retry,
)

revision = "20261126_connector_oauth_redirect_uri"
down_revision = "20261119_shopify_store_locks"
branch_labels = None
depends_on = None

COLUMN = "redirect_uri"


def _column() -> sa.Column:
    return sa.Column(COLUMN, sa.String(1000), nullable=True)


# (table, column) in upgrade order; downgrade reverses it.
TABLES = ("mcp_oauth_flow_states", "user_oauth")
_LOCK_TIMEOUT = "5s"
_LOCK_RETRY_DELAYS = (2.0, 4.0)


def _has_table(inspector: sa.Inspector, table: str) -> bool:
    return table in inspector.get_table_names()


def _has_column(inspector: sa.Inspector, table: str) -> bool:
    return COLUMN in {column["name"] for column in inspector.get_columns(table)}


def _add_column(table: str) -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            offline_ddl_with_lock_timeout(
                f"ALTER TABLE {table} ADD COLUMN {COLUMN} VARCHAR(1000)",
                lock_timeout=_LOCK_TIMEOUT,
            )
        else:
            op.add_column(table, _column())
        return
    inspector = sa.inspect(op.get_bind())
    if not _has_table(inspector, table):
        # Legacy partial schemas may omit either relation; skip just that one.
        return
    if _has_column(inspector, table):
        return
    if context.dialect.name == "postgresql":
        run_ddl_with_lock_retry(
            f"ALTER TABLE {table} ADD COLUMN {COLUMN} VARCHAR(1000)",
            table=table,
            lock_timeout=_LOCK_TIMEOUT,
            retry_delays=_LOCK_RETRY_DELAYS,
        )
    else:
        op.add_column(table, _column())


def _drop_column(table: str) -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name == "postgresql":
            offline_ddl_with_lock_timeout(
                f"ALTER TABLE {table} DROP COLUMN {COLUMN}",
                lock_timeout=_LOCK_TIMEOUT,
            )
        else:
            op.drop_column(table, COLUMN)
        return
    inspector = sa.inspect(op.get_bind())
    if not _has_table(inspector, table):
        # Legacy partial schemas may omit either relation; skip just that one.
        return
    if not _has_column(inspector, table):
        return
    if context.dialect.name == "postgresql":
        run_ddl_with_lock_retry(
            f"ALTER TABLE {table} DROP COLUMN {COLUMN}",
            table=table,
            lock_timeout=_LOCK_TIMEOUT,
            retry_delays=_LOCK_RETRY_DELAYS,
        )
    elif context.dialect.name == "sqlite":
        with op.batch_alter_table(table) as batch:
            batch.drop_column(COLUMN)
    else:
        op.drop_column(table, COLUMN)


def upgrade() -> None:
    for table in TABLES:
        _add_column(table)


def downgrade() -> None:
    for table in reversed(TABLES):
        _drop_column(table)

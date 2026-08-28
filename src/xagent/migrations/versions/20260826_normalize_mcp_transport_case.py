"""Normalize stored MCP transport values to their canonical lowercase form

Revision ID: 20260826_normalize_mcp_transport_case
Revises: 20260825_add_slack_channels_join_scope
Create Date: 2026-08-26

`transport` is a free-form string on the MCP API models, so rows written
before the write-time normalizing validators shipped may hold a mixed-case
or padded value (e.g. "Streamable_HTTP"). Such a row is classified as
connectable by the case-insensitive half of the MCP OAuth feature and
rejected by the exact-matching half, and it never matches the transport
dispatch in the core MCP session layer, so it can never actually connect.

The application code now normalizes on every write, but a shared catalog row
is only rewritten for its auth config, never its transport, so an
un-migrated row would stay mixed-case indefinitely. Backfill the two web-layer
tables once so the stored values agree with what every reader expects.

Idempotent: LOWER(TRIM(...)) is a no-op on already-canonical values, and the
WHERE clause skips rows that are already normalized.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260826_normalize_mcp_transport_case"
down_revision = "20260825_add_slack_channels_join_scope"
branch_labels = None
depends_on = None

_TABLES = ("mcp_servers", "public_mcp_apps")


def _tables_with_transport() -> list[str]:
    from alembic import context

    bind = context.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    present = []
    for table in _TABLES:
        if table not in existing:
            continue
        columns = {col["name"] for col in inspector.get_columns(table)}
        if "transport" in columns:
            present.append(table)
    return present


def upgrade() -> None:
    for table in _tables_with_transport():
        # LOWER/TRIM are ANSI SQL and behave identically on PostgreSQL and
        # SQLite, so no dialect branch is needed here. NULL transports are left
        # alone: the comparison below is NULL-safe (evaluates to NULL, not
        # true), so those rows are skipped rather than rewritten to ''.
        op.execute(
            f"""
            UPDATE {table}
            SET transport = LOWER(TRIM(transport))
            WHERE transport IS NOT NULL
              AND transport <> LOWER(TRIM(transport))
            """
        )


def downgrade() -> None:
    # Deliberately not reversible: the original mixed-case/padded spellings are
    # not recorded anywhere, and restoring them would reintroduce rows that the
    # application cannot connect. Normalized values remain valid for every
    # earlier revision, so leaving them in place is safe.
    pass

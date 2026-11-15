"""Record every deleted OAuth credential in a revocation outbox, by trigger.

Creates ``credential_revocations`` and the triggers that write it, so that a
native ``user_oauth`` row (not Composio-backed) and an ``mcp_oauth_grants``
row cannot leave the database without a record of their tokens as stored at
that moment -- whether an ORM delete, a bulk delete, a foreign-key cascade
or raw SQL removes them. An MCP grant also gets its record when its status
becomes ``revoked``. A worker (``services.credential_revocations``) revokes
each record with its provider. See ``models.credential_revocation`` for the
design, including why a ``BEFORE DELETE`` trigger on ``mcp_oauth_clients``
deletes the client's grants first: a cascade from the client would
otherwise run the grant triggers after the client row is gone, and the
record would lose the client fields its revocation needs.

PostgreSQL gets PL/pgSQL trigger functions, SQLite inline trigger bodies;
the statements they run are the same. The SQL is a frozen copy of what
``models.credential_revocation`` installs on a database created from the
models (migrations never import application models, whose shape may move on
after this revision); a test compares the two installations.

Revision ID: 20261116_credential_revocation_outbox
Revises: 20261115_user_oauth_issuer
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261116_credential_revocation_outbox"
down_revision: Union[str, None] = "20261115_user_oauth_issuer"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "credential_revocations"

_USER_OAUTH_INSERT = """INSERT INTO credential_revocations (
        kind, source_id, user_id, grant_key, issuer, provider_user_id,
        access_token, refresh_token
    ) VALUES (
        'user_oauth', {row}.id, {row}.user_id,
        CASE WHEN {row}.oauth_provider IS NULL OR {row}.provider_user_id IS NULL
            THEN 'user_oauth#' || CAST({row}.id AS VARCHAR)
            ELSE 'user_oauth:' || {row}.oauth_provider || ':' || {row}.provider_user_id
        END,
        {row}.oauth_provider, {row}.provider_user_id,
        {row}.access_token, {row}.refresh_token
    )"""
_MCP_GRANT_INSERT = """INSERT INTO credential_revocations (
        kind, source_id, user_id, grant_key, issuer, access_token, refresh_token,
        mcp_server_id, mcp_oauth_client_id, resource_owner_key, resource,
        client_id, client_secret, token_endpoint_auth_method, revocation_endpoint
    )
    SELECT
        'mcp_oauth_grant', {row}.id, {row}.user_id,
        'mcp_oauth_grant:' || {row}.lookup_hash, {row}.issuer,
        {row}.access_token, {row}.refresh_token,
        {row}.mcp_server_id, {row}.mcp_oauth_client_id,
        {row}.resource_owner_key, {row}.resource,
        client.client_id, client.client_secret,
        client.token_endpoint_auth_method, {revocation_endpoint}
    FROM (SELECT 1 AS one) AS anchor
    LEFT JOIN mcp_oauth_clients AS client
        ON client.id = {row}.mcp_oauth_client_id"""
_DELETE_CLIENT_GRANTS_FIRST = (
    "DELETE FROM mcp_oauth_grants WHERE mcp_oauth_client_id = OLD.id"
)
_NATIVE_ROW = "OLD.token_type IS NULL OR OLD.token_type <> 'composio'"
_PG_REVOCATION_ENDPOINT = "client.metadata ->> 'revocation_endpoint'"
_SQLITE_REVOCATION_ENDPOINT = (
    "CASE WHEN json_valid(client.metadata) "
    "THEN json_extract(client.metadata, '$.revocation_endpoint') END"
)

_POSTGRESQL_DDL = [
    f"""CREATE OR REPLACE FUNCTION capture_user_oauth_revocation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    {_USER_OAUTH_INSERT.format(row="OLD")};
    RETURN NULL;
END
$$""",
    "DROP TRIGGER IF EXISTS capture_user_oauth_revocation ON user_oauth",
    f"""CREATE TRIGGER capture_user_oauth_revocation
AFTER DELETE ON user_oauth FOR EACH ROW
WHEN ({_NATIVE_ROW})
EXECUTE FUNCTION capture_user_oauth_revocation()""",
    f"""CREATE OR REPLACE FUNCTION capture_mcp_oauth_grant_revocation()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    grant_row record;
BEGIN
    IF TG_OP = 'DELETE' THEN
        grant_row := OLD;
    ELSE
        grant_row := NEW;
    END IF;
    {_MCP_GRANT_INSERT.format(row="grant_row", revocation_endpoint=_PG_REVOCATION_ENDPOINT)};
    RETURN NULL;
END
$$""",
    "DROP TRIGGER IF EXISTS capture_mcp_oauth_grant_revocation_on_delete "
    "ON mcp_oauth_grants",
    """CREATE TRIGGER capture_mcp_oauth_grant_revocation_on_delete
AFTER DELETE ON mcp_oauth_grants FOR EACH ROW
WHEN (OLD.status <> 'revoked')
EXECUTE FUNCTION capture_mcp_oauth_grant_revocation()""",
    "DROP TRIGGER IF EXISTS capture_mcp_oauth_grant_revocation_on_revoke "
    "ON mcp_oauth_grants",
    """CREATE TRIGGER capture_mcp_oauth_grant_revocation_on_revoke
AFTER UPDATE OF status ON mcp_oauth_grants FOR EACH ROW
WHEN (NEW.status = 'revoked' AND OLD.status <> 'revoked')
EXECUTE FUNCTION capture_mcp_oauth_grant_revocation()""",
    f"""CREATE OR REPLACE FUNCTION delete_mcp_oauth_client_grants_first()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    {_DELETE_CLIENT_GRANTS_FIRST};
    RETURN OLD;
END
$$""",
    "DROP TRIGGER IF EXISTS delete_mcp_oauth_client_grants_first "
    "ON mcp_oauth_clients",
    """CREATE TRIGGER delete_mcp_oauth_client_grants_first
BEFORE DELETE ON mcp_oauth_clients FOR EACH ROW
EXECUTE FUNCTION delete_mcp_oauth_client_grants_first()""",
]

_SQLITE_DDL = [
    "DROP TRIGGER IF EXISTS capture_user_oauth_revocation",
    f"""CREATE TRIGGER capture_user_oauth_revocation
AFTER DELETE ON user_oauth FOR EACH ROW
WHEN {_NATIVE_ROW}
BEGIN
    {_USER_OAUTH_INSERT.format(row="OLD")};
END""",
    "DROP TRIGGER IF EXISTS capture_mcp_oauth_grant_revocation_on_delete",
    f"""CREATE TRIGGER capture_mcp_oauth_grant_revocation_on_delete
AFTER DELETE ON mcp_oauth_grants FOR EACH ROW
WHEN OLD.status <> 'revoked'
BEGIN
    {_MCP_GRANT_INSERT.format(row="OLD", revocation_endpoint=_SQLITE_REVOCATION_ENDPOINT)};
END""",
    "DROP TRIGGER IF EXISTS capture_mcp_oauth_grant_revocation_on_revoke",
    f"""CREATE TRIGGER capture_mcp_oauth_grant_revocation_on_revoke
AFTER UPDATE OF status ON mcp_oauth_grants FOR EACH ROW
WHEN NEW.status = 'revoked' AND OLD.status <> 'revoked'
BEGIN
    {_MCP_GRANT_INSERT.format(row="NEW", revocation_endpoint=_SQLITE_REVOCATION_ENDPOINT)};
END""",
    "DROP TRIGGER IF EXISTS delete_mcp_oauth_client_grants_first",
    f"""CREATE TRIGGER delete_mcp_oauth_client_grants_first
BEFORE DELETE ON mcp_oauth_clients FOR EACH ROW
BEGIN
    {_DELETE_CLIENT_GRANTS_FIRST};
END""",
]


def _ddl(dialect: str) -> list[str]:
    if dialect == "postgresql":
        return _POSTGRESQL_DDL
    if dialect == "sqlite":
        return _SQLITE_DDL
    raise RuntimeError("The credential revocation outbox requires PostgreSQL or SQLite")


def upgrade() -> None:
    bind = op.get_bind()
    statements = _ddl(bind.dialect.name)
    if not sa.inspect(bind).has_table(TABLE):
        op.create_table(
            TABLE,
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("kind", sa.String(32), nullable=False),
            sa.Column(
                "status", sa.String(16), nullable=False, server_default="pending"
            ),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "next_attempt_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("source_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("grant_key", sa.Text(), nullable=False),
            sa.Column("issuer", sa.String(1000), nullable=True),
            sa.Column("provider_user_id", sa.String(), nullable=True),
            sa.Column("access_token", sa.Text(), nullable=True),
            sa.Column("refresh_token", sa.Text(), nullable=True),
            sa.Column("mcp_server_id", sa.Integer(), nullable=True),
            sa.Column("mcp_oauth_client_id", sa.Integer(), nullable=True),
            sa.Column("resource_owner_key", sa.String(512), nullable=True),
            sa.Column("resource", sa.String(1000), nullable=True),
            sa.Column("client_id", sa.String(1000), nullable=True),
            sa.Column("client_secret", sa.Text(), nullable=True),
            sa.Column("token_endpoint_auth_method", sa.String(100), nullable=True),
            sa.Column("revocation_endpoint", sa.Text(), nullable=True),
            sqlite_autoincrement=True,
        )
        op.create_index(
            "ix_credential_revocations_status_due",
            TABLE,
            ["status", "next_attempt_at"],
        )
        op.create_index("ix_credential_revocations_grant_key", TABLE, ["grant_key"])
    for statement in statements:
        op.execute(statement)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for trigger, table in (
            ("capture_user_oauth_revocation", "user_oauth"),
            ("capture_mcp_oauth_grant_revocation_on_delete", "mcp_oauth_grants"),
            ("capture_mcp_oauth_grant_revocation_on_revoke", "mcp_oauth_grants"),
            ("delete_mcp_oauth_client_grants_first", "mcp_oauth_clients"),
        ):
            op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        for function in (
            "capture_user_oauth_revocation",
            "capture_mcp_oauth_grant_revocation",
            "delete_mcp_oauth_client_grants_first",
        ):
            op.execute(f"DROP FUNCTION IF EXISTS {function}()")
    else:
        for trigger in (
            "capture_user_oauth_revocation",
            "capture_mcp_oauth_grant_revocation_on_delete",
            "capture_mcp_oauth_grant_revocation_on_revoke",
            "delete_mcp_oauth_client_grants_first",
        ):
            op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    op.drop_table(TABLE)

"""Record every deleted OAuth credential in a revocation outbox, by trigger.

Creates ``credential_revocations`` and the triggers that write it, so that a
native ``user_oauth`` row (not Composio-backed) and an ``mcp_oauth_grants``
row cannot leave the database without a record of their tokens as stored at
that moment -- whether an ORM delete, a bulk delete, a foreign-key cascade
or raw SQL removes them. A native row also gets its record when an UPDATE
drops its credential in place (its refresh token, or without one its access
token, goes to NULL or empty), and an MCP grant when its status becomes
``revoked``. A worker (``services.credential_revocations``) revokes
each record with its provider. See ``models.credential_revocation`` for the
design, including why a ``BEFORE DELETE`` trigger on ``mcp_oauth_clients``
deletes the client's grants first: a cascade from the client would
otherwise run the grant triggers after the client row is gone, and the
record would lose the client fields its revocation needs.

It adds ``user_oauth.oauth_client_id``, the client id of the OAuth app a
callback exchanged its code with, which the native trigger copies into the
receipt so a GitHub revocation authenticates only as the issuing app. There
is no backfill: nothing recorded which app issued an existing row's tokens,
so existing rows stay NULL until they are reconnected. The worker revokes a
NULL row with the GitHub app configuration captured when it was deleted, as
before this column existed -- if the app was replaced between the token's
issue and its deletion, GitHub answers 404 and the record settles as gone
although the original grant may live on. That residual risk is limited to
rows connected before this revision.

It also seeds a receipt for every grant already marked ``revoked`` that still
holds a token: before this revision a revoked grant's provider revocation was
a best-effort call that may have failed, and nothing would retry it after
the cutover.

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
        access_token, refresh_token, client_id, client_secret, issuing_client_id
    )
    SELECT
        'user_oauth', {row}.id, {row}.user_id,
        CASE WHEN {row}.oauth_provider IS NULL OR {row}.provider_user_id IS NULL
            THEN 'user_oauth#' || CAST({row}.id AS VARCHAR)
            ELSE 'user_oauth:' || {row}.oauth_provider || ':' || {row}.provider_user_id
        END,
        {row}.oauth_provider, {row}.provider_user_id,
        {row}.access_token, {row}.refresh_token,
        provider.client_id, provider.client_secret, {row}.oauth_client_id
    FROM (SELECT 1 AS one) AS anchor
    LEFT JOIN oauth_providers AS provider
        ON provider.provider_name = {row}.oauth_provider
        AND {row}.oauth_provider = 'github'"""
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
_TOKEN_DROPPED = """(
    COALESCE(OLD.refresh_token, '') <> '' AND COALESCE(NEW.refresh_token, '') = ''
) OR (
    COALESCE(OLD.refresh_token, '') = ''
    AND COALESCE(OLD.access_token, '') <> ''
    AND COALESCE(NEW.access_token, '') = ''
)"""
_PG_REVOCATION_ENDPOINT = "client.metadata ->> 'revocation_endpoint'"
_SQLITE_REVOCATION_ENDPOINT = (
    "CASE WHEN json_valid(client.metadata) "
    "THEN json_extract(client.metadata, '$.revocation_endpoint') END"
)

_OUTBOX, _USER_OAUTH, _PROVIDERS = TABLE, "user_oauth", "oauth_providers"
_GRANTS, _CLIENTS = "mcp_oauth_grants", "mcp_oauth_clients"


def _pg_function(name: str, body: str, returns: str = "NULL") -> str:
    return f"""CREATE OR REPLACE FUNCTION {name}()
RETURNS trigger LANGUAGE plpgsql AS $$
{body}
    RETURN {returns};
END
$$"""


_PG_USER_OAUTH_FUNCTION = _pg_function(
    "capture_user_oauth_revocation",
    f"BEGIN\n    {_USER_OAUTH_INSERT.format(row='OLD')};",
)
_PG_GRANT_FUNCTION = _pg_function(
    "capture_mcp_oauth_grant_revocation",
    f"""DECLARE
    grant_row record;
BEGIN
    IF TG_OP = 'DELETE' THEN
        grant_row := OLD;
    ELSE
        grant_row := NEW;
    END IF;
    {_MCP_GRANT_INSERT.format(row="grant_row", revocation_endpoint=_PG_REVOCATION_ENDPOINT)};""",
)

# The columns each trigger body reads, by table. A trigger is installed only
# when every one of them exists: PostgreSQL rejects a trigger naming a missing
# column outright, and SQLite re-validates every trigger on each ALTER TABLE,
# so one naming a missing table or column would break unrelated schema
# changes. Schemas that lack them are partial (fixtures standing in for
# xagent's tables) and hold no credential to record. ``token_type`` is not
# required: a user_oauth that predates it holds no Composio row, so every row
# is native and the Composio exclusion is left out.
_USER_OAUTH_COLUMNS = frozenset(
    {"id", "user_id", "oauth_provider", "oauth_client_id", "provider_user_id"}
    | {"access_token", "refresh_token"}
)
_GRANT_COLUMNS = frozenset(
    {"id", "user_id", "lookup_hash", "issuer", "status", "resource"}
    | {"access_token", "refresh_token", "mcp_server_id", "mcp_oauth_client_id"}
    | {"resource_owner_key"}
)
_CLIENT_COLUMNS = frozenset(
    {"id", "client_id", "client_secret", "token_endpoint_auth_method", "metadata"}
)
_NATIVE_REQUIRES = {
    _USER_OAUTH: _USER_OAUTH_COLUMNS,
    _OUTBOX: frozenset(),
    _PROVIDERS: frozenset({"provider_name", "client_id", "client_secret"}),
}
_GRANT_REQUIRES = {
    _GRANTS: _GRANT_COLUMNS,
    _CLIENTS: _CLIENT_COLUMNS,
    _OUTBOX: frozenset(),
}


def _when(dialect: str, *conditions: str | None) -> str:
    present = [f"({condition})" for condition in conditions if condition]
    if not present:
        return ""
    joined = " AND ".join(present)
    return f"\nWHEN ({joined})" if dialect == "postgresql" else f"\nWHEN {joined}"


def _native_row(composio_aware: bool) -> str | None:
    return _NATIVE_ROW if composio_aware else None


def _user_oauth_delete_ddl(dialect: str, composio_aware: bool) -> tuple[str, ...]:
    head = "CREATE TRIGGER capture_user_oauth_revocation\nAFTER DELETE ON user_oauth"
    when = _when(dialect, _native_row(composio_aware))
    if dialect == "postgresql":
        return (
            _PG_USER_OAUTH_FUNCTION,
            f"{head} FOR EACH ROW{when}\n"
            "EXECUTE FUNCTION capture_user_oauth_revocation()",
        )
    insert = _USER_OAUTH_INSERT.format(row="OLD")
    return (f"{head} FOR EACH ROW{when}\nBEGIN\n    {insert};\nEND",)


def _user_oauth_drop_ddl(dialect: str, composio_aware: bool) -> tuple[str, ...]:
    head = (
        "CREATE TRIGGER capture_user_oauth_token_drop\n"
        "AFTER UPDATE OF access_token, refresh_token ON user_oauth"
    )
    when = _when(dialect, _native_row(composio_aware), _TOKEN_DROPPED)
    if dialect == "postgresql":
        return (
            _PG_USER_OAUTH_FUNCTION,
            f"{head} FOR EACH ROW{when}\n"
            "EXECUTE FUNCTION capture_user_oauth_revocation()",
        )
    insert = _USER_OAUTH_INSERT.format(row="OLD")
    return (f"{head} FOR EACH ROW{when}\nBEGIN\n    {insert};\nEND",)


def _grant_ddl(name: str, event_clause: str, condition: str, row: str):
    def ddl(dialect: str, _composio_aware: bool) -> tuple[str, ...]:
        head = f"CREATE TRIGGER {name}\n{event_clause} ON mcp_oauth_grants"
        when = _when(dialect, condition)
        if dialect == "postgresql":
            return (
                _PG_GRANT_FUNCTION,
                f"{head} FOR EACH ROW{when}\n"
                "EXECUTE FUNCTION capture_mcp_oauth_grant_revocation()",
            )
        insert = _MCP_GRANT_INSERT.format(
            row=row, revocation_endpoint=_SQLITE_REVOCATION_ENDPOINT
        )
        return (f"{head} FOR EACH ROW{when}\nBEGIN\n    {insert};\nEND",)

    return ddl


def _client_ddl(dialect: str, _composio_aware: bool) -> tuple[str, ...]:
    head = (
        "CREATE TRIGGER delete_mcp_oauth_client_grants_first\n"
        "BEFORE DELETE ON mcp_oauth_clients FOR EACH ROW"
    )
    if dialect == "postgresql":
        return (
            _pg_function(
                "delete_mcp_oauth_client_grants_first",
                f"BEGIN\n    {_DELETE_CLIENT_GRANTS_FIRST};",
                returns="OLD",
            ),
            f"{head}\nEXECUTE FUNCTION delete_mcp_oauth_client_grants_first()",
        )
    return (f"{head}\nBEGIN\n    {_DELETE_CLIENT_GRANTS_FIRST};\nEND",)


# (trigger, table, columns read by table, ddl(dialect, composio_aware)).
_TRIGGERS = (
    (
        "capture_user_oauth_revocation",
        _USER_OAUTH,
        _NATIVE_REQUIRES,
        _user_oauth_delete_ddl,
    ),
    (
        "capture_user_oauth_token_drop",
        _USER_OAUTH,
        _NATIVE_REQUIRES,
        _user_oauth_drop_ddl,
    ),
    (
        "capture_mcp_oauth_grant_revocation_on_delete",
        _GRANTS,
        _GRANT_REQUIRES,
        _grant_ddl(
            "capture_mcp_oauth_grant_revocation_on_delete",
            "AFTER DELETE",
            "OLD.status <> 'revoked'",
            "OLD",
        ),
    ),
    (
        "capture_mcp_oauth_grant_revocation_on_revoke",
        _GRANTS,
        _GRANT_REQUIRES,
        _grant_ddl(
            "capture_mcp_oauth_grant_revocation_on_revoke",
            "AFTER UPDATE OF status",
            "NEW.status = 'revoked' AND OLD.status <> 'revoked'",
            "NEW",
        ),
    ),
    (
        "delete_mcp_oauth_client_grants_first",
        _CLIENTS,
        {_CLIENTS: frozenset({"id"}), _GRANTS: frozenset({"mcp_oauth_client_id"})},
        _client_ddl,
    ),
)


# Grants marked revoked before this revision. Their provider revocation was
# a best-effort call at disconnect time that may have failed, leaving the
# tokens live, and the triggers never capture an already revoked grant (its
# status change predates them, and its later deletion is skipped as already
# recorded). So each one still holding a token gets its receipt here; a
# grant whose revocation did succeed is revoked again, which RFC 7009 makes
# a no-op.
_SEED_REVOKED_GRANTS = """INSERT INTO credential_revocations (
        kind, source_id, user_id, grant_key, issuer, access_token, refresh_token,
        mcp_server_id, mcp_oauth_client_id, resource_owner_key, resource,
        client_id, client_secret, token_endpoint_auth_method, revocation_endpoint
    )
    SELECT
        'mcp_oauth_grant', g.id, g.user_id,
        'mcp_oauth_grant:' || g.lookup_hash, g.issuer,
        g.access_token, g.refresh_token,
        g.mcp_server_id, g.mcp_oauth_client_id,
        g.resource_owner_key, g.resource,
        client.client_id, client.client_secret,
        client.token_endpoint_auth_method, {revocation_endpoint}
    FROM mcp_oauth_grants AS g
    LEFT JOIN mcp_oauth_clients AS client ON client.id = g.mcp_oauth_client_id
    WHERE g.status = 'revoked' AND COALESCE(g.access_token, '') <> ''
    ORDER BY g.id"""

_FUNCTIONS = (
    "capture_user_oauth_revocation",
    "capture_mcp_oauth_grant_revocation",
    "delete_mcp_oauth_client_grants_first",
)


def _drop_trigger(dialect: str, trigger: str, table: str) -> None:
    on = f" ON {table}" if dialect == "postgresql" else ""
    op.execute(f"DROP TRIGGER IF EXISTS {trigger}{on}")


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    if dialect not in ("postgresql", "sqlite"):
        raise RuntimeError(
            "The credential revocation outbox requires PostgreSQL or SQLite"
        )
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
            sa.Column("issuing_client_id", sa.String(500), nullable=True),
            sqlite_autoincrement=True,
        )
        op.create_index(
            "ix_credential_revocations_status_due",
            TABLE,
            ["status", "next_attempt_at"],
        )
        op.create_index("ix_credential_revocations_grant_key", TABLE, ["grant_key"])
    inspector = sa.inspect(bind)
    present = set(inspector.get_table_names())
    if _USER_OAUTH in present and "oauth_client_id" not in {
        column["name"] for column in inspector.get_columns(_USER_OAUTH)
    }:
        op.add_column(
            _USER_OAUTH, sa.Column("oauth_client_id", sa.String(500), nullable=True)
        )
        inspector = sa.inspect(bind)
    columns = {
        table: {column["name"] for column in inspector.get_columns(table)}
        for table in {_USER_OAUTH, _GRANTS, _CLIENTS, _PROVIDERS, _OUTBOX} & present
    }

    def has(requires: dict[str, frozenset[str]]) -> bool:
        return all(
            table in columns and needed <= columns[table]
            for table, needed in requires.items()
        )

    if has(_GRANT_REQUIRES):
        op.execute(
            _SEED_REVOKED_GRANTS.format(
                revocation_endpoint=_PG_REVOCATION_ENDPOINT
                if dialect == "postgresql"
                else _SQLITE_REVOCATION_ENDPOINT
            )
        )
    composio_aware = "token_type" in columns.get(_USER_OAUTH, set())
    for trigger, table, requires, ddl in _TRIGGERS:
        if not has(requires):
            continue
        _drop_trigger(dialect, trigger, table)
        for statement in ddl(dialect, composio_aware):
            op.execute(statement)


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    for trigger, table, _requires, _ddl in _TRIGGERS:
        if table in tables:
            _drop_trigger(dialect, trigger, table)
    if dialect == "postgresql":
        for function in _FUNCTIONS:
            op.execute(f"DROP FUNCTION IF EXISTS {function}()")
    op.drop_table(TABLE)
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table(_USER_OAUTH) and "oauth_client_id" in {
        column["name"] for column in inspector.get_columns(_USER_OAUTH)
    }:
        with op.batch_alter_table(_USER_OAUTH) as batch:
            batch.drop_column("oauth_client_id")

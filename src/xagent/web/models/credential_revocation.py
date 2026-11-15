"""The credential revocation outbox: one record per OAuth credential that left.

Invariant
---------
An OAuth credential this deployment holds never leaves the database without a
``credential_revocations`` record that holds its tokens exactly as they were
stored at that moment, written in the same transaction:

* a native ``user_oauth`` row (every row except a Composio-backed one, whose
  grant belongs to Composio's own OAuth app and is removed through Composio)
  when it is deleted;
* an ``mcp_oauth_grants`` row when it is deleted, or when its ``status``
  becomes ``revoked`` -- a revoked grant is never used again, so that is the
  moment it is given up. Deleting an already revoked grant records nothing
  more: its tokens cannot change while it is revoked, and its record exists.

Database triggers write the records, not application code, so the invariant
holds for every way a row can go: an ORM delete, a bulk ``Query.delete``, a
foreign-key cascade from its user, MCP server or OAuth client, or raw SQL from
another service sharing the database. A rolled-back delete takes its record
with it. ``services.credential_revocations`` is the worker that revokes each
record with its provider and the only place that does.

What a record holds
-------------------
Everything its revocation needs, because the source row is gone by the time
the worker reads it:

* the token columns copied as stored -- Fernet ciphertext, written by
  ``EncryptedOAuthToken`` for ``user_oauth`` and by ``encrypt_value`` for MCP
  grants; the worker opens them with ``decrypt_value_strict``;
* the issuer: ``user_oauth.oauth_provider`` for a native row (NULL when the
  issuer is unknown -- the worker then fails the record rather than guess),
  the grant's issuer URL for an MCP grant;
* for an MCP grant, its OAuth client's ``client_id``, encrypted
  ``client_secret``, client authentication method and RFC 7009 revocation
  endpoint, plus the grant identity the worker's reference check matches on;
* ``grant_key``, the upstream grant identity the worker serialises on:
  ``(oauth_provider, provider_user_id)`` for a native row -- the key of its
  revocation fence, since GitHub and Google revoke per account -- and the
  grant's own ``lookup_hash`` for an MCP grant.

Token material (both tokens and the client secret) is erased once a record
reaches a terminal status.

The cascade from an OAuth client
--------------------------------
A grant's record copies its client's fields through a join, and that join
must still find the client. A foreign-key cascade does not guarantee it: on
PostgreSQL the cascade from ``mcp_oauth_clients`` (directly, or through
``mcp_servers``) deletes the client row before the cascaded grant deletes run
their triggers, and a join from those triggers no longer sees it; SQLite
behaves the same for a client deleted directly. So a ``BEFORE DELETE``
trigger on ``mcp_oauth_clients`` deletes the client's grants first, while the
client row is still there; the cascade then finds nothing left to delete.

Where the triggers come from
----------------------------
Migration ``20261116_credential_revocation_outbox`` installs them on an
existing database. After every ``Base.metadata.create_all`` -- which builds a
fresh database (stamped at head, never migrated) and runs at each startup --
:func:`install_missing_credential_revocation_triggers` creates any trigger
that is missing, so a later SQLite table rebuild (Alembic's batch mode copies
and drops the table, and its triggers with it) is repaired at the next start.
Both carry the same SQL; a test compares what each installs. A trigger is
installed only once every table its body touches exists.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Connection, DateTime, Index, Integer, String, Text, event, inspect
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .database import Base
from .mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from .user_oauth import COMPOSIO_TOKEN_TYPE, UserOAuth


class CredentialRevocationKind(str, enum.Enum):
    """Which table a record's credential came from."""

    USER_OAUTH = "user_oauth"
    MCP_OAUTH_GRANT = "mcp_oauth_grant"


class CredentialRevocationStatus(str, enum.Enum):
    """Where a record stands. Every status but ``pending`` is terminal."""

    #: Waiting for (or leased to) the worker; ``next_attempt_at`` says when.
    PENDING = "pending"
    #: The provider revoked it, or answered that it was already invalid.
    REVOKED = "revoked"
    #: Deliberately not revoked: a live credential still references the same
    #: grant (whose own deletion enqueues later), the provider offers no
    #: revocation, or nothing revocable was stored.
    SKIPPED = "skipped"
    #: Cannot be revoked without guessing or without data that is gone: an
    #: unknown issuer, an unreadable token, a missing client registration.
    FAILED = "failed"
    #: The attempt budget ran out against provider failures.
    DEAD = "dead"


class CredentialRevocation(Base):  # type: ignore[no-any-unimported]
    """One credential that left the database and is owed a provider revocation.

    No foreign keys: a record must outlive its source row, its user and its
    MCP server or OAuth client, and every value its revocation needs is
    copied into it (see the module docstring).
    """

    __tablename__ = "credential_revocations"
    __table_args__ = (
        # The worker's scan: due pending rows.
        Index("ix_credential_revocations_status_due", "status", "next_attempt_at"),
        # The worker's per-grant serialisation: the oldest open record of a
        # grant_key goes first.
        Index("ix_credential_revocations_grant_key", "grant_key"),
        # Outcomes are fenced on (id, attempts). Without AUTOINCREMENT SQLite
        # may hand a deleted record's id to the next insert.
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=CredentialRevocationStatus.PENDING.value,
        server_default=CredentialRevocationStatus.PENDING.value,
    )
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    #: The deleted row's id in its own table (``kind`` names it).
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    grant_key: Mapped[str] = mapped_column(Text, nullable=False)
    #: ``user_oauth.oauth_provider`` (NULL: unknown) or the MCP grant's issuer.
    issuer: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    provider_user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    #: Ciphertext as stored in the source row; erased at a terminal status.
    access_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)

    # MCP grants only: the grant identity the reference check matches on.
    mcp_server_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mcp_oauth_client_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resource_owner_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    resource: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    # MCP grants only: the OAuth client as it was when the grant left.
    client_id: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    #: Ciphertext as stored in ``mcp_oauth_clients``; erased at a terminal status.
    client_secret: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_endpoint_auth_method: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    revocation_endpoint: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<CredentialRevocation(id={self.id}, kind={self.kind!r}, "
            f"source_id={self.source_id}, status={self.status!r})>"
        )


# --- trigger SQL --------------------------------------------------------------
#
# Kept dialect-neutral where the dialects agree (``||``, CAST, CASE, a LEFT
# JOIN from a one-row anchor) so the two installations differ only in their
# wrapping: PostgreSQL runs the statements from PL/pgSQL functions, SQLite
# inlines them in the trigger body. ``{row}`` is the trigger's row reference.


def _user_oauth_insert(row: str) -> str:
    return f"""INSERT INTO credential_revocations (
        kind, source_id, user_id, grant_key, issuer, provider_user_id,
        access_token, refresh_token
    ) VALUES (
        '{CredentialRevocationKind.USER_OAUTH.value}', {row}.id, {row}.user_id,
        CASE WHEN {row}.oauth_provider IS NULL OR {row}.provider_user_id IS NULL
            THEN 'user_oauth#' || CAST({row}.id AS VARCHAR)
            ELSE 'user_oauth:' || {row}.oauth_provider || ':' || {row}.provider_user_id
        END,
        {row}.oauth_provider, {row}.provider_user_id,
        {row}.access_token, {row}.refresh_token
    )"""


def _mcp_grant_insert(row: str, revocation_endpoint: str) -> str:
    return f"""INSERT INTO credential_revocations (
        kind, source_id, user_id, grant_key, issuer, access_token, refresh_token,
        mcp_server_id, mcp_oauth_client_id, resource_owner_key, resource,
        client_id, client_secret, token_endpoint_auth_method, revocation_endpoint
    )
    SELECT
        '{CredentialRevocationKind.MCP_OAUTH_GRANT.value}', {row}.id, {row}.user_id,
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
_NATIVE_ROW = f"OLD.token_type IS NULL OR OLD.token_type <> '{COMPOSIO_TOKEN_TYPE}'"
_PG_REVOCATION_ENDPOINT = "client.metadata ->> 'revocation_endpoint'"
# ``json_extract`` raises on malformed JSON, which would make the delete fail.
_SQLITE_REVOCATION_ENDPOINT = (
    "CASE WHEN json_valid(client.metadata) "
    "THEN json_extract(client.metadata, '$.revocation_endpoint') END"
)
_OUTBOX = "credential_revocations"
_USER_OAUTH = UserOAuth.__tablename__
_GRANTS = MCPOAuthGrant.__tablename__
_CLIENTS = MCPOAuthClient.__tablename__


@dataclass(frozen=True)
class _Trigger:
    """One trigger: the table it is on, every table its body touches (it is
    installed only when they all exist: SQLite re-validates every trigger on
    each ``ALTER TABLE``, so a trigger naming a missing table would break
    unrelated schema changes), and its creating statements per dialect."""

    name: str
    table: str
    requires: frozenset[str]
    ddl: dict[str, tuple[str, ...]]


def _pg_function(name: str, body: str, returns: str = "NULL") -> str:
    return f"""CREATE OR REPLACE FUNCTION {name}()
RETURNS trigger LANGUAGE plpgsql AS $$
{body}
    RETURN {returns};
END
$$"""


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
    {_mcp_grant_insert("grant_row", _PG_REVOCATION_ENDPOINT)};""",
)

CREDENTIAL_REVOCATION_TRIGGERS: tuple[_Trigger, ...] = (
    _Trigger(
        name="capture_user_oauth_revocation",
        table=_USER_OAUTH,
        requires=frozenset({_USER_OAUTH, _OUTBOX}),
        ddl={
            "postgresql": (
                _pg_function(
                    "capture_user_oauth_revocation",
                    f"BEGIN\n    {_user_oauth_insert('OLD')};",
                ),
                f"""CREATE TRIGGER capture_user_oauth_revocation
AFTER DELETE ON user_oauth FOR EACH ROW
WHEN ({_NATIVE_ROW})
EXECUTE FUNCTION capture_user_oauth_revocation()""",
            ),
            "sqlite": (
                f"""CREATE TRIGGER capture_user_oauth_revocation
AFTER DELETE ON user_oauth FOR EACH ROW
WHEN {_NATIVE_ROW}
BEGIN
    {_user_oauth_insert("OLD")};
END""",
            ),
        },
    ),
    _Trigger(
        name="capture_mcp_oauth_grant_revocation_on_delete",
        table=_GRANTS,
        requires=frozenset({_GRANTS, _CLIENTS, _OUTBOX}),
        ddl={
            "postgresql": (
                _PG_GRANT_FUNCTION,
                """CREATE TRIGGER capture_mcp_oauth_grant_revocation_on_delete
AFTER DELETE ON mcp_oauth_grants FOR EACH ROW
WHEN (OLD.status <> 'revoked')
EXECUTE FUNCTION capture_mcp_oauth_grant_revocation()""",
            ),
            "sqlite": (
                f"""CREATE TRIGGER capture_mcp_oauth_grant_revocation_on_delete
AFTER DELETE ON mcp_oauth_grants FOR EACH ROW
WHEN OLD.status <> 'revoked'
BEGIN
    {_mcp_grant_insert("OLD", _SQLITE_REVOCATION_ENDPOINT)};
END""",
            ),
        },
    ),
    _Trigger(
        name="capture_mcp_oauth_grant_revocation_on_revoke",
        table=_GRANTS,
        requires=frozenset({_GRANTS, _CLIENTS, _OUTBOX}),
        ddl={
            "postgresql": (
                _PG_GRANT_FUNCTION,
                """CREATE TRIGGER capture_mcp_oauth_grant_revocation_on_revoke
AFTER UPDATE OF status ON mcp_oauth_grants FOR EACH ROW
WHEN (NEW.status = 'revoked' AND OLD.status <> 'revoked')
EXECUTE FUNCTION capture_mcp_oauth_grant_revocation()""",
            ),
            "sqlite": (
                f"""CREATE TRIGGER capture_mcp_oauth_grant_revocation_on_revoke
AFTER UPDATE OF status ON mcp_oauth_grants FOR EACH ROW
WHEN NEW.status = 'revoked' AND OLD.status <> 'revoked'
BEGIN
    {_mcp_grant_insert("NEW", _SQLITE_REVOCATION_ENDPOINT)};
END""",
            ),
        },
    ),
    _Trigger(
        name="delete_mcp_oauth_client_grants_first",
        table=_CLIENTS,
        requires=frozenset({_CLIENTS, _GRANTS}),
        ddl={
            "postgresql": (
                _pg_function(
                    "delete_mcp_oauth_client_grants_first",
                    f"BEGIN\n    {_DELETE_CLIENT_GRANTS_FIRST};",
                    returns="OLD",
                ),
                """CREATE TRIGGER delete_mcp_oauth_client_grants_first
BEFORE DELETE ON mcp_oauth_clients FOR EACH ROW
EXECUTE FUNCTION delete_mcp_oauth_client_grants_first()""",
            ),
            "sqlite": (
                f"""CREATE TRIGGER delete_mcp_oauth_client_grants_first
BEFORE DELETE ON mcp_oauth_clients FOR EACH ROW
BEGIN
    {_DELETE_CLIENT_GRANTS_FIRST};
END""",
            ),
        },
    ),
)


def _existing_triggers(connection: Connection) -> set[str]:
    if connection.dialect.name == "postgresql":
        query = "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal"
    else:
        query = "SELECT name FROM sqlite_master WHERE type = 'trigger'"
    return set(connection.exec_driver_sql(query).scalars())


def install_missing_credential_revocation_triggers(connection: Connection) -> None:
    """Create every outbox trigger that is missing and whose tables exist.

    Runs after each ``create_all`` (see the listener below): that installs
    the triggers on a fresh database, which is created from the models and
    stamped at head rather than migrated, and puts back any trigger a later
    SQLite table rebuild dropped. An existing trigger is left alone, never
    dropped and re-created, so no delete can slip through a gap. PostgreSQL
    and SQLite only, like every other database feature here.
    """
    dialect = connection.dialect.name
    if dialect not in ("postgresql", "sqlite"):
        return
    tables = set(inspect(connection).get_table_names())
    existing = _existing_triggers(connection)
    for trigger in CREDENTIAL_REVOCATION_TRIGGERS:
        if trigger.name in existing or not trigger.requires <= tables:
            continue
        for statement in trigger.ddl[dialect]:
            connection.exec_driver_sql(statement)


@event.listens_for(Base.metadata, "after_create")
def _install_triggers_after_create_all(
    _metadata: object, connection: Connection, **_kw: object
) -> None:
    install_missing_credential_revocation_triggers(connection)

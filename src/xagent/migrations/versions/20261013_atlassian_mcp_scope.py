"""request a fixed scope subset for the remote Atlassian MCP connector

Revision ID: 20261013_atlassian_mcp_scope
Revises: 20261012_zendesk_oauth
Create Date: 2026-10-13

Atlassian's protected-resource metadata lists 38 scopes (1023 characters once
normalized) and its 401 challenge carries no scope, so the connector, which
requested every discovered scope, exceeded the 1000-character MCP OAuth scope
limit and could not start authorization. The catalog row now sets
``launch_config.auth.scope`` to a fixed subset, and
``launch_config.auth.required_scope`` to the basic identity scopes a grant
must include. Atlassian grants product scopes only for the products the
consenting account's site has, so a grant is usable without, for example, the
Bitbucket scopes.

The shared ``atlassian`` server row the connect path created still carries
the old auth, which the canonical-definition check rejects, so the connector
would stay unavailable there until someone connected again. Only the two
scope keys change, so this migration sets that row's ``auth`` to the scoped
value in place. The row keeps its id, its policy fields (headers, env,
timeout, managed and the like), its user links and its grants. A grant that
covers ``required_scope`` stays usable; a user whose grant does not must
connect again, as after any scope change.

The row's client registrations that no grant references are deleted, with
the OAuth flow states that point at them. The connect path reuses a stored
registration for the same server, issuer and redirect URI and has no way to
register again if the provider rejects that client. A registration no grant
references holds nothing a user depends on, so it is dropped rather than
kept at the risk of having gone stale at the provider; the next connect
registers a fresh client. A registration that a grant references is kept,
whatever the grant's status, and no grant is deleted.

- Only the row the catalog connect path writes is changed: named after the
  app id, with the catalog transport and URL, ``auth`` exactly the previous
  catalog value (an ``app_id`` key some rows carry is ignored and kept) and
  no owning user link. A user-created server under the same name and a
  server with other auth keys (a client id, another scope) are left alone,
  and so is every row when the columns that tell them apart are missing.
  Nothing is changed unless the catalog row carries the built-in provenance
  marker.
- A row an administrator edited keeps those edits. The canonical-definition
  check still rejects one whose ``managed`` value was changed, as before.
- Flow states are deleted before their registration. PostgreSQL would
  cascade them, but on SQLite the migration runner turns foreign keys off
  and rejects any new violation. Without the column that links grants to a
  registration, no registration is deleted.
- Downgrade sets ``auth`` back on rows that carry exactly the scoped auth,
  under the same conditions, and restores the previous catalog value. It
  deletes nothing, and a registration deleted on upgrade is not restored:
  the next connect registers a new one.
"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import builtin_provenance_identity

revision: str = "20261013_atlassian_mcp_scope"
down_revision: Union[str, None] = "20261012_zendesk_oauth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ID = "atlassian"
SERVER_TRANSPORT = "streamable_http"
SERVER_URL = "https://mcp.atlassian.com/v2/mcp"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}
# Frozen copies of builtin_mcp_registry.ATLASSIAN_MCP_OAUTH_SCOPE and
# ATLASSIAN_MCP_OAUTH_REQUIRED_SCOPE at the time this migration was written
# (tests/alembic pins them against the registry).
SCOPE = (
    "email offline_access read:account read:bitbucket:agent-interface "
    "read:confluence:agent-interface read:jira:agent-interface read:me "
    "search:confluence:agent-interface search:jira:agent-interface "
    "search:rovo:agent-interface write:bitbucket:agent-interface "
    "write:confluence:agent-interface write:jira:agent-interface"
)
REQUIRED_SCOPE = "email offline_access read:account read:me"
LEGACY_AUTH = {"type": "mcp_oauth"}
SCOPED_AUTH = {
    "type": "mcp_oauth",
    "scope": SCOPE,
    "required_scope": REQUIRED_SCOPE,
}
LEGACY_LAUNCH_CONFIG = {
    "url": SERVER_URL,
    "auth": LEGACY_AUTH,
    "builtin_provenance": BUILTIN_PROVENANCE,
}
SCOPED_LAUNCH_CONFIG = {
    "url": SERVER_URL,
    "auth": SCOPED_AUTH,
    "builtin_provenance": BUILTIN_PROVENANCE,
}

PUBLIC_MCP_APPS = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("launch_config", sa.JSON),
)
MCP_SERVERS = sa.table(
    "mcp_servers",
    sa.column("id", sa.Integer),
    sa.column("name", sa.String),
    sa.column("transport", sa.String),
    sa.column("url", sa.String),
    sa.column("auth", sa.JSON),
)
USER_MCPSERVERS = sa.table(
    "user_mcpservers",
    sa.column("mcpserver_id", sa.Integer),
    sa.column("is_owner", sa.Boolean),
)
MCP_OAUTH_CLIENTS = sa.table(
    "mcp_oauth_clients",
    sa.column("id", sa.Integer),
    sa.column("mcp_server_id", sa.Integer),
)
MCP_OAUTH_GRANTS = sa.table(
    "mcp_oauth_grants",
    sa.column("mcp_oauth_client_id", sa.Integer),
)
MCP_OAUTH_FLOW_STATES = sa.table(
    "mcp_oauth_flow_states",
    sa.column("mcp_oauth_client_id", sa.Integer),
)


def _columns(bind: sa.engine.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _owned_launch(value: object) -> bool:
    return isinstance(value, dict) and builtin_provenance_identity(
        value.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _set_owned_launch_config(
    bind: sa.engine.Connection, launch_config: dict[str, Any]
) -> bool:
    """Write the catalog launch config if this app's row is the builtin one.

    Returns whether the row exists and carries the builtin provenance marker.
    For such a row the execution fields are read from the code registry, so
    this write only keeps the persisted copy free of drift. A row without the
    marker is an operator's own app and is left alone, as is every server row
    (the caller stops).
    """
    if not {"app_id", "launch_config"}.issubset(_columns(bind, "public_mcp_apps")):
        return False
    row = (
        bind.execute(
            sa.select(PUBLIC_MCP_APPS.c.launch_config).where(
                PUBLIC_MCP_APPS.c.app_id == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if row is None or not _owned_launch(row["launch_config"]):
        return False
    if row["launch_config"] != launch_config:
        bind.execute(
            sa.update(PUBLIC_MCP_APPS)
            .where(PUBLIC_MCP_APPS.c.app_id == APP_ID)
            .values(launch_config=launch_config)
        )
    return True


def _has_identifying_columns(bind: sa.engine.Connection) -> bool:
    """Whether the columns that tell the catalog row from other servers exist."""
    return {"id", "name", "transport", "url", "auth"}.issubset(
        _columns(bind, "mcp_servers")
    ) and {"mcpserver_id", "is_owner"}.issubset(_columns(bind, "user_mcpservers"))


def _matches_catalog_definition(row: Any, auth: dict[str, Any]) -> bool:
    """Whether a server row has the catalog transport and URL and stores
    exactly ``auth``, apart from an ``app_id`` key."""
    stored_auth = row["auth"]
    if not isinstance(stored_auth, dict):
        return False
    comparable_auth = {
        key: value for key, value in stored_auth.items() if key != "app_id"
    }
    return (
        str(row["transport"] or "").lower() == SERVER_TRANSPORT
        and row["url"] == SERVER_URL
        and comparable_auth == auth
    )


def _has_owning_user(bind: sa.engine.Connection, server_id: int) -> bool:
    owned = bind.execute(
        sa.select(USER_MCPSERVERS.c.mcpserver_id).where(
            USER_MCPSERVERS.c.mcpserver_id == server_id,
            USER_MCPSERVERS.c.is_owner.is_(True),
        )
    ).first()
    return owned is not None


def _replace_catalog_server_auth(
    bind: sa.engine.Connection,
    old_auth: dict[str, Any],
    new_auth: dict[str, Any],
) -> list[int]:
    """Set ``auth`` to ``new_auth`` on the shared server row the catalog
    connect path created under ``old_auth``, and return the updated row ids.

    That row is named after the app id, uses the catalog transport and URL,
    stores exactly ``old_auth`` (an ``app_id`` key some rows carry is ignored,
    as the canonical-definition check ignores it, and is kept) and has no
    owning user link. A user-created server under the same name has an owner,
    and a server with other auth keys (a client id, another scope) is a custom
    OAuth configuration; neither is touched. Without the columns that prove
    either, nothing is updated. Only ``auth`` is written: the row id, policy
    columns and every row that references the server stay as they are.
    """
    if not _has_identifying_columns(bind):
        return []
    rows = (
        bind.execute(
            sa.select(
                MCP_SERVERS.c.id,
                MCP_SERVERS.c.transport,
                MCP_SERVERS.c.url,
                MCP_SERVERS.c.auth,
            ).where(MCP_SERVERS.c.name == APP_ID)
        )
        .mappings()
        .all()
    )
    updated_ids: list[int] = []
    for row in rows:
        if not _matches_catalog_definition(row, old_auth):
            continue
        server_id = int(row["id"])
        if _has_owning_user(bind, server_id):
            continue
        updated_auth = dict(new_auth)
        if "app_id" in row["auth"]:
            updated_auth["app_id"] = row["auth"]["app_id"]
        bind.execute(
            sa.update(MCP_SERVERS)
            .where(MCP_SERVERS.c.id == server_id)
            .values(auth=updated_auth)
        )
        updated_ids.append(server_id)
    return updated_ids


def _delete_unused_client_registrations(
    bind: sa.engine.Connection, server_ids: list[int]
) -> None:
    """Delete the client registrations of ``server_ids`` that no grant
    references, with the flow states that point at them.

    A grant of any status keeps its registration, so no grant is deleted or
    left pointing at a missing registration. Flow states are deleted first
    because the SQLite migration runner turns foreign keys off (PostgreSQL
    would cascade them). Without the column that links grants to a
    registration, a registration in use cannot be told apart and nothing is
    deleted.
    """
    if not {"id", "mcp_server_id"}.issubset(_columns(bind, "mcp_oauth_clients")):
        return
    if "mcp_oauth_client_id" not in _columns(bind, "mcp_oauth_grants"):
        return
    referenced = sa.exists().where(
        MCP_OAUTH_GRANTS.c.mcp_oauth_client_id == MCP_OAUTH_CLIENTS.c.id
    )
    unused_ids = list(
        bind.execute(
            sa.select(MCP_OAUTH_CLIENTS.c.id).where(
                MCP_OAUTH_CLIENTS.c.mcp_server_id.in_(server_ids), ~referenced
            )
        ).scalars()
    )
    if not unused_ids:
        return
    if "mcp_oauth_client_id" in _columns(bind, "mcp_oauth_flow_states"):
        bind.execute(
            sa.delete(MCP_OAUTH_FLOW_STATES).where(
                MCP_OAUTH_FLOW_STATES.c.mcp_oauth_client_id.in_(unused_ids)
            )
        )
    bind.execute(
        sa.delete(MCP_OAUTH_CLIENTS).where(MCP_OAUTH_CLIENTS.c.id.in_(unused_ids))
    )


def upgrade() -> None:
    bind = op.get_bind()
    if not _set_owned_launch_config(bind, SCOPED_LAUNCH_CONFIG):
        return
    updated_ids = _replace_catalog_server_auth(bind, LEGACY_AUTH, SCOPED_AUTH)
    _delete_unused_client_registrations(bind, updated_ids)


def downgrade() -> None:
    bind = op.get_bind()
    if not _set_owned_launch_config(bind, LEGACY_LAUNCH_CONFIG):
        return
    _replace_catalog_server_auth(bind, SCOPED_AUTH, LEGACY_AUTH)

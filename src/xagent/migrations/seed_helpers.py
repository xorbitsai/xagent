"""Shared helpers for Alembic migrations that seed rows into shared catalog
tables (e.g. ``public_mcp_apps``) and must avoid clobbering operator data on
downgrade.

These migrations insert a row only if no row with the same id already exists
(so upgrade() never overwrites an operator's pre-existing custom row). Without
a matching guard on the way down, a plain ``DELETE ... WHERE id = :id`` would
delete that operator row anyway, since it only matches on id and does not
check that the row is actually the one the migration inserted. This module is
plain, data-free infrastructure code (not seed data), so migrations may import
it directly.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import sqlalchemy as sa
from sqlalchemy.engine import Connection
from sqlalchemy.sql.expression import TableClause

logger = logging.getLogger(__name__)

# Every public_mcp_apps field a pre-existing row can plausibly differ on from
# the migration's seed snapshot, including the two JSON-typed columns
# (oauth_scopes, launch_config): a row that predates a migration's app_id
# being seeded (or predates the admin API's built-in protections) can carry
# arbitrary values there even if every other field happens to match. Matching
# is done in Python (see below), not in the SQL WHERE clause, so these
# JSON-typed columns compare by value regardless of key order or
# serialization differences across database backends. This relies on the
# caller's ``table`` declaring these two columns with ``sa.JSON`` (as every
# current caller does); see the function docstring below for what happens
# if a future caller's table does not.
#
# This default is specific to public_mcp_apps (every current caller deletes
# from that table). A future caller for a different table must pass its own
# ``match_columns`` explicitly rather than rely on this default - if that
# table happens to also have same-named columns, they would be compared
# against unrelated seed data.
PUBLIC_MCP_APPS_SEED_MATCH_COLUMNS: tuple[str, ...] = (
    "name",
    "description",
    "transport",
    "provider_name",
    "category",
    "icon",
    "is_visible_in_connector",
    "oauth_scopes",
    "launch_config",
)

# The static (non-env-dependent) oauth_providers fields every current built-in
# provider seed migration matches on when guarding its provider-row downgrade
# delete. client_id/client_secret/redirect_uri are env-dependent and
# deliberately excluded.
OAUTH_PROVIDER_SEED_MATCH_COLUMNS: tuple[str, ...] = (
    "name",
    "auth_url",
    "token_url",
    "userinfo_url",
    "user_id_path",
    "email_path",
    "default_scopes",
)


def delete_unmodified_seeded_rows(
    bind: Connection,
    table: TableClause,
    seed_rows: Iterable[dict[str, Any]],
    match_columns: Sequence[str] = PUBLIC_MCP_APPS_SEED_MATCH_COLUMNS,
    id_column: str = "app_id",
    accepted_seed_rows: Iterable[dict[str, Any]] = (),
) -> None:
    """Delete rows a migration seeded, but only the ones that still match its
    seed snapshot exactly (id plus ``match_columns``).

    Columns absent from the current schema (e.g. dropped by a later
    migration) are skipped rather than treated as a mismatch. A row whose id
    matches but whose other fields differ was created or edited by an
    operator after the fact and is left in place. A ``table`` that declares
    only a subset of columns (a common lightweight ``sa.table()`` pattern),
    or a seed row missing one of the match keys, is handled the same way:
    the column/key is skipped rather than raising ``KeyError``.

    If ``match_columns`` is non-empty but none of its columns exist in the
    current schema, or a given seed row supplies none of those columns,
    provenance can no longer be verified at all, so nothing is deleted for
    that row (matching on ``id_column`` alone would silently reproduce the
    exact bug this helper exists to prevent). Pass ``match_columns=()``
    explicitly to opt into matching on ``id_column`` only.

    Each candidate row is fetched and its ``match_columns`` compared in
    Python rather than matched via the SQL ``WHERE`` clause, so JSON-typed
    columns compare structurally (immune to key-order or whitespace
    differences between the stored value and the seed) instead of relying
    on backend-specific text/JSON equality. The final ``DELETE`` then
    re-asserts every non-JSON match column in its own ``WHERE`` clause too,
    so the delete does not rely solely on ``id_column`` being unique and the
    window where a concurrent writer could change the row between the
    ``SELECT`` and the ``DELETE`` is narrowed for those columns. JSON-typed
    columns are left out of that re-assertion (SQL equality on them is the
    same backend-fragile comparison being avoided above), so that specific
    narrow race remains: a concurrent write that changes only a JSON column
    between the two statements is not caught.

    ``accepted_seed_rows`` may contain alternate, sanctioned snapshots left by
    later migrations. They are considered only for an id also present in
    ``seed_rows``. This lets a historical seed downgrade accept both the exact
    row it inserted and a known descendant-migration representation without
    weakening comparison for arbitrary operator changes.

    A JSON-typed column only compares structurally when ``table`` declares
    it with ``sa.JSON``. A ``table`` that leaves a JSON column untyped (the
    "lightweight subset of columns" pattern above) gets that column back as
    a raw driver value instead of a deserialized Python object, so it will
    essentially never compare equal to the seed's Python value. That fails
    safe (the row is preserved, never wrongly deleted) rather than raising,
    but it does mean such a row is never cleaned up on downgrade either.
    """
    inspector = sa.inspect(bind)
    if table.name not in set(inspector.get_table_names()):
        return

    existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
    if id_column not in existing_columns:
        return
    columns_to_match = [
        column for column in match_columns if column in existing_columns
    ]
    if match_columns and not columns_to_match:
        logger.warning(
            "delete_unmodified_seeded_rows: none of match_columns %s exist on "
            "%s; preserving all seed rows for this downgrade since provenance "
            "cannot be verified",
            list(match_columns),
            table.name,
        )
        return

    def _column(name: str) -> Any:
        return table.c[name] if name in table.c else sa.column(name)

    id_col = _column(id_column)
    match_column_refs = [(column, _column(column)) for column in columns_to_match]
    scalar_column_refs = [
        (column, ref)
        for column, ref in match_column_refs
        if not isinstance(ref.type, sa.JSON)
    ]

    accepted_rows_by_id: dict[Any, list[dict[str, Any]]] = {}
    for accepted_row in accepted_seed_rows:
        if id_column not in accepted_row:
            logger.warning(
                "delete_unmodified_seeded_rows: ignoring an accepted seed row for "
                "%s missing id_column %r",
                table.name,
                id_column,
            )
            continue
        accepted_rows_by_id.setdefault(accepted_row[id_column], []).append(accepted_row)

    for row in seed_rows:
        if id_column not in row:
            logger.warning(
                "delete_unmodified_seeded_rows: skipping a seed row for %s "
                "missing id_column %r",
                table.name,
                id_column,
            )
            continue
        row_id = row[id_column]
        if match_column_refs and not any(
            column in row for column, _ in match_column_refs
        ):
            # The row supplies none of the columns we'd need to verify
            # provenance with, so (symmetrically with the schema-side check
            # above) refuse to delete rather than falling back to matching on
            # id_column alone.
            logger.warning(
                "delete_unmodified_seeded_rows: preserving %s row %s=%r; seed "
                "row supplies none of match_columns %s",
                table.name,
                id_column,
                row_id,
                columns_to_match,
            )
            continue

        delete_conditions = [id_col == row_id]

        if match_column_refs:
            candidate = (
                bind.execute(
                    sa.select(*(ref for _, ref in match_column_refs))
                    .select_from(table)
                    .where(id_col == row_id)
                )
                .mappings()
                .first()
            )
            if candidate is None:
                continue
            accepted_rows = [row, *accepted_rows_by_id.get(row_id, ())]
            matching_row = None
            mismatched_columns: set[str] = set()
            for accepted_row in accepted_rows:
                supplied_columns = [
                    column for column, _ in match_column_refs if column in accepted_row
                ]
                if not supplied_columns:
                    logger.warning(
                        "delete_unmodified_seeded_rows: ignoring an accepted seed "
                        "snapshot for %s row %s=%r; it supplies none of "
                        "match_columns %s",
                        table.name,
                        id_column,
                        row_id,
                        columns_to_match,
                    )
                    continue
                current_mismatches = {
                    column
                    for column in supplied_columns
                    if candidate[column] != accepted_row[column]
                }
                if not current_mismatches:
                    matching_row = accepted_row
                    break
                mismatched_columns.update(current_mismatches)

            if matching_row is None:
                logger.warning(
                    "delete_unmodified_seeded_rows: preserving %s row %s=%r; "
                    "current value differs from every accepted seed snapshot on %s "
                    "(likely operator-modified)",
                    table.name,
                    id_column,
                    row_id,
                    sorted(mismatched_columns),
                )
                continue
            delete_conditions.extend(
                ref == matching_row[column]
                for column, ref in scalar_column_refs
                if column in matching_row
            )

        bind.execute(sa.delete(table).where(sa.and_(*delete_conditions)))


# ---------------------------------------------------------------------------
# Remote (streamable_http / mcp_oauth) catalog identity vs. existing servers
# ---------------------------------------------------------------------------

# Every persisted mcp_servers field a fresh connect-created shared row never
# carries (NULL / false / empty on the row _ensure_catalog_mcp_oauth_server
# writes). A colliding row holding any of them is a custom server's own
# configuration, not ours, even when name/transport/URL/auth all match. Same
# intent as _server_has_policy_beyond_catalog_identity on the stdio connect
# path, which once missed fields by hand-picking a subset; tests/alembic pins
# this tuple against the ORM columns. Lifecycle fields (managed,
# restart_policy) are compared against their defaults separately. The
# docker_*/container_*/volumes/bind_ports/auto_start columns are not consulted:
# a streamable_http row cannot carry a meaningful value in them.
REMOTE_MCP_SERVER_POLICY_COLUMNS: tuple[str, ...] = (
    "command",
    "args",
    "cwd",
    "env",
    "headers",
    "timeout",
    "runtime_input_schema",
    "runtime_bindings",
    "concurrency_safe",
    "concurrent_tools",
    "allow_delegated_authorization",
)
_REMOTE_MCP_SERVER_LIFECYCLE_DEFAULTS: dict[str, tuple[Any, ...]] = {
    "managed": (None, "external"),
    "restart_policy": (None, "no"),
}
_REMOTE_MCP_SERVER_IDENTITY_COLUMNS: tuple[str, ...] = ("transport", "url", "auth")
_USER_MCPSERVER_OWNERSHIP_COLUMNS: tuple[str, ...] = ("mcpserver_id", "is_owner")

MCP_SERVERS_TABLE = sa.table(
    "mcp_servers",
    sa.column("id", sa.Integer),
    sa.column("name", sa.String),
    sa.column("transport", sa.String),
    sa.column("url", sa.String),
    sa.column("auth", sa.JSON),
    sa.column("command", sa.String),
    sa.column("args", sa.JSON),
    sa.column("cwd", sa.String),
    sa.column("env", sa.JSON),
    sa.column("headers", sa.JSON),
    sa.column("timeout", sa.Integer),
    sa.column("runtime_input_schema", sa.JSON),
    sa.column("runtime_bindings", sa.JSON),
    sa.column("concurrency_safe", sa.Boolean),
    sa.column("concurrent_tools", sa.JSON),
    sa.column("allow_delegated_authorization", sa.Boolean),
    sa.column("managed", sa.String),
    sa.column("restart_policy", sa.String),
)
USER_MCPSERVERS_TABLE = sa.table(
    "user_mcpservers",
    sa.column("mcpserver_id", sa.Integer),
    sa.column("is_owner", sa.Boolean),
)


def _policy_value_is_unset(value: Any) -> bool:
    """False/None/empty are the values a fresh catalog row stores; 0 is not.

    ``timeout=0`` is an accepted, forwarded value (an immediate HTTP timeout),
    so a plain truthiness test would wrongly treat it as absent.
    """
    if value is None or value is False:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) == 0
    return False


def remote_mcp_server_identity_is_claimable(
    bind: Connection,
    *,
    app_id: str,
    display_name: str,
    transport: str,
    url: str,
    auth: dict[str, Any],
    seed_label: str,
) -> bool:
    """Whether a remote-MCP catalog seed may claim ``app_id`` given mcp_servers.

    The connector listing resolves a catalog app's shared server row by
    normalized transport + app_id/display name, not by URL, auth or owner. A
    custom ``streamable_http`` server that predates the identity would
    therefore be presented as the official card while the runtime kept using
    its own URL/auth; the connect-time guards in
    ``_ensure_catalog_mcp_oauth_server`` only run when someone connects. So a
    seed must check ``mcp_servers`` before inserting its catalog row.

    The single accepted collision is the shared row our own mcp_oauth connect
    path creates: ``name == app_id``, the catalog transport and URL, ``auth``
    exactly equal to the catalog auth (extra keys such as a custom client_id or
    scope are a custom OAuth configuration the connect path would otherwise
    overwrite), none of :data:`REMOTE_MCP_SERVER_POLICY_COLUMNS` set, lifecycle
    fields at their defaults, and no ``user_mcpservers`` link with
    ``is_owner=true``. That row exists legitimately on a downgrade -> upgrade
    round trip. Anything else, including a schema that lacks the identity
    columns on ``mcp_servers`` or the ownership columns on
    ``user_mcpservers``, returns False and logs an ERROR naming the rows:
    the caller skips seeding and leaves every row untouched. Skipping rather
    than raising keeps ``alembic upgrade head`` (and therefore startup) going;
    with no catalog row seeded nothing claims the server. Alembic stamps the
    revision regardless, so the skip is permanent for that revision and the
    log spells out the manual remediation.

    Identity collision uses :func:`canonicalize_builtin_identity` on the
    server name against both ``app_id`` and ``display_name`` -- the same
    normalization the listing applies.
    """
    from xagent.builtin_identity import canonicalize_builtin_identity

    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "mcp_servers" not in tables:
        return True
    server_columns = {column["name"] for column in inspector.get_columns("mcp_servers")}
    identity_keys = {
        canonicalize_builtin_identity(app_id),
        canonicalize_builtin_identity(display_name),
    } - {None}
    colliding = [
        row
        for row in bind.execute(
            sa.select(MCP_SERVERS_TABLE.c.id, MCP_SERVERS_TABLE.c.name)
        ).mappings()
        if canonicalize_builtin_identity(row["name"]) in identity_keys
    ]
    if not colliding:
        return True
    described = sorted((int(row["id"]), str(row["name"])) for row in colliding)

    user_link_columns = (
        {column["name"] for column in inspector.get_columns("user_mcpservers")}
        if "user_mcpservers" in tables
        else set()
    )
    if (
        not set(_REMOTE_MCP_SERVER_IDENTITY_COLUMNS) <= server_columns
        or not set(_USER_MCPSERVER_OWNERSHIP_COLUMNS) <= user_link_columns
    ):
        logger.error(
            "Permanently skipping builtin %s seed: mcp_servers row(s) %s collide "
            "with '%s' and the schema cannot prove which one our connect path "
            "created. Re-running `alembic upgrade head` will NOT retry this; "
            "rename or delete the rows and seed the catalog row manually (see "
            "the migration's ROW/BUILTIN_PROVENANCE).",
            seed_label,
            described,
            app_id,
        )
        return False

    if len(colliding) == 1:
        server_id = described[0][0]
        policy_columns = [
            c for c in REMOTE_MCP_SERVER_POLICY_COLUMNS if c in server_columns
        ]
        lifecycle_columns = [
            c for c in _REMOTE_MCP_SERVER_LIFECYCLE_DEFAULTS if c in server_columns
        ]
        server = (
            bind.execute(
                sa.select(
                    MCP_SERVERS_TABLE.c.name,
                    MCP_SERVERS_TABLE.c.transport,
                    MCP_SERVERS_TABLE.c.url,
                    MCP_SERVERS_TABLE.c.auth,
                    *(MCP_SERVERS_TABLE.c[c] for c in policy_columns),
                    *(MCP_SERVERS_TABLE.c[c] for c in lifecycle_columns),
                ).where(MCP_SERVERS_TABLE.c.id == server_id)
            )
            .mappings()
            .one()
        )
        carries_policy = any(
            not _policy_value_is_unset(server[c]) for c in policy_columns
        ) or any(
            server[c] not in _REMOTE_MCP_SERVER_LIFECYCLE_DEFAULTS[c]
            for c in lifecycle_columns
        )
        owned = bind.execute(
            sa.select(USER_MCPSERVERS_TABLE.c.mcpserver_id).where(
                USER_MCPSERVERS_TABLE.c.mcpserver_id == server_id,
                USER_MCPSERVERS_TABLE.c.is_owner.is_(True),
            )
        ).first()
        if (
            server["name"] == app_id
            and str(server["transport"] or "").lower() == transport.lower()
            and server["url"] == url
            and server["auth"] == auth
            and not carries_policy
            and owned is None
        ):
            return True

    logger.error(
        "Permanently skipping builtin %s seed: mcp_servers row(s) %s collide with "
        "'%s' and could not all be proven to come from the catalog connect path "
        "(user-owned, foreign transport/URL, custom auth, or carrying their own "
        "policy). Re-running `alembic upgrade head` will NOT retry this; rename or "
        "delete the rows and seed the catalog row manually (see the migration's "
        "ROW/BUILTIN_PROVENANCE).",
        seed_label,
        described,
        app_id,
    )
    return False

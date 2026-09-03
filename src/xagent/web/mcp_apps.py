"""Centralized registry for MCP Applications and OAuth Providers.

This module provides a scalable structure for defining supported MCP applications,
their OAuth configurations, and server launch configurations.
"""

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .builtin_mcp_registry import get_builtin_execution_fields_and_optional_scopes
from .models.public_mcp import PublicMCPApp

# Apps that must not be satisfied by a bare provider-level OAuth grant (one
# created via the app_id-less connect flow, e.g. UserOAuth.provider == "meta").
# That flow requests only the provider's default scopes (see
# generic_oauth_login's app_scopes=None branch when app_id is absent), never
# an app's own oauth_scopes, so it can't carry a permission such as
# pages_read_user_content that was added after the bare flow already existed.
# Only an app-scoped grant (UserOAuth.provider == the app_id) counts for these
# apps; Instagram is deliberately excluded so its existing bare "meta" grants
# keep working, since its required scopes haven't changed.
#
# github: the github oauth_providers row's own default_scopes is
# identity-only ("read:user") -- the functional "repo"/"user:email" scopes
# live solely on the app row and are merged in only when generic_oauth_login
# is called with app_id="github". A bare GET /api/auth/github/login (no
# app_id) would otherwise request just "read:user", and the callback's bare
# batch-connect branch would still activate the github app's UserMCPServer
# against that under-scoped grant -- reporting "connected" while every
# repo-scoped tool then fails. This is a no-op for the normal connect flow
# (the catalog UI always passes app_id="github", and app_id == provider_name
# for this connector, so an already-connected grant already satisfies the
# app-scoped match trivially).
#
# myob: same structural situation as github, just more extreme -- the myob
# oauth_providers row's own default_scopes is empty (there's no shared
# identity scope to request; see the provider row's own comment), and every
# functional sme-* scope this connector needs lives solely on the app row.
# Without this entry, a bare GET /api/auth/myob/login would request NO
# scopes at all, yet the callback would still complete (MYOB's businessId
# guard has nothing to do with scopes) and activate the app's UserMCPServer
# against a grant with zero sme-* permissions.
APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT = frozenset({"facebook", "github", "myob"})


def _normalize_oauth_grant_key(value: object) -> str | None:
    """Case/whitespace-insensitive key, matching mcp.py's _normalize_app_key.

    Duplicated rather than imported: mcp.py imports this module, so importing
    back would cycle. An admin-created PublicMCPApp.app_id is free-form (see
    POST /admin/mcp/apps), so every APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT
    membership test must normalize the same way the connector-display layer
    does, or a differently-cased app_id (e.g. "Facebook") silently bypasses
    the policy at whichever call site compares raw strings instead.
    """
    if value is None:
        return None
    normalized = "-".join(str(value).strip().lower().split())
    return normalized or None


def requires_app_scoped_oauth_grant(app_id: object) -> bool:
    """Whether app_id must not be satisfied by a bare provider-level grant.

    Normalized the same way _app_lookup_keys resolves an app's own id, so a
    differently-cased or whitespace-padded admin-created app_id (e.g.
    "Facebook") is covered consistently everywhere this policy is checked.
    """
    return _normalize_oauth_grant_key(app_id) in APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT


def restrict_to_app_scoped_oauth_grant(
    app_id: object, candidates: Iterable[object]
) -> list[str]:
    """Narrow OAuth provider/grant candidates to app-scoped ones where required.

    ``candidates`` is typically ``(provider_name, app_id)`` or a list of
    ``UserOAuth.provider`` values to try. For an app in
    ``APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT``, only the candidate matching
    ``app_id`` (normalized) survives — a bare provider-level grant is dropped.
    For every other app, candidates pass through unchanged (deduped, order
    preserved). Candidates are returned in their original casing/whitespace:
    callers match them against exact-case stored values (e.g.
    ``UserOAuth.provider``), which is why normalization only decides
    membership and is never applied to the returned strings.
    """
    deduped = list(dict.fromkeys(c for c in candidates if isinstance(c, str) and c))

    if not requires_app_scoped_oauth_grant(app_id):
        return deduped
    normalized_app_id = _normalize_oauth_grant_key(app_id)
    return [
        candidate
        for candidate in deduped
        if _normalize_oauth_grant_key(candidate) == normalized_app_id
    ]


def classify_app_auth(transport: Any, launch_config: Any) -> str:
    """Single source of truth for how a catalog app is connected.

    Derived from the entry's own fields so the backend connect gate and both
    frontend dialogs can't drift apart. Values:
        - "builtin_oauth": provider redirect flow (transport == "oauth")
        - "api_key": static key, connected via /api/mcp/apps/{id}/connect
        - "keyless": local stdio module that needs no secrets (e.g. Chrome),
          connected via the same endpoint with no env
        - "mcp_oauth": remote MCP server, connected via per-user OAuth
          Authorization Code + PKCE (Dynamic Client Registration when no
          static client_id is configured) — /api/mcp/apps/{id}/oauth/connect
        - "unconnectable": none of the above
    """
    # Reuses the runtime's notion of which transports are remote ("oauth"
    # here instead means a static-provider redirect wrapping our own stdio
    # module). Lowercased like the builtin_oauth check above: an admin PATCH
    # can store a mixed-case transport, and the two halves of this feature
    # must not disagree about the same row.
    from .services.mcp_runtime import HTTP_MCP_TRANSPORTS

    if str(transport or "").lower() == "oauth":
        return "builtin_oauth"
    launch = launch_config if isinstance(launch_config, dict) else {}
    if launch.get("required_env") and launch.get("command"):
        return "api_key"
    # Keyless is deliberately stdio-only: a command on a remote transport is a
    # mis-authored entry, not a connectable app. Also excludes env_mapping —
    # that shape means the launcher expects an injected token (the builtin
    # OAuth apps' pattern, e.g. env_mapping={"SLACK_ACCESS_TOKEN":
    # "access_token"}), so a custom app authored with that shape is not
    # actually secret-free even though it has no required_env; classifying it
    # keyless would offer a no-secrets Connect button for a server that fails
    # at tool-call time for a missing token.
    if (
        str(transport or "").lower() == "stdio"
        and launch.get("command")
        and not launch.get("required_env")
        and not launch.get("env_mapping")
    ):
        return "keyless"
    auth = launch.get("auth")
    if (
        str(transport or "").lower() in HTTP_MCP_TRANSPORTS
        and launch.get("url")
        and isinstance(auth, dict)
        and auth.get("type") == "mcp_oauth"
    ):
        return "mcp_oauth"
    return "unconnectable"


def _app_to_dict(app: PublicMCPApp) -> Dict[str, Any]:
    # One registry scan (not two - see the helper's own docstring) since
    # this runs per app on the connector-listing path.
    execution_fields, optional_oauth_scopes = (
        get_builtin_execution_fields_and_optional_scopes(app.app_id)
    )
    if execution_fields is None:
        execution_fields = {
            "name": app.name,
            "transport": app.transport,
            "provider_name": app.provider_name,
            "oauth_scopes": deepcopy(app.oauth_scopes or []),
            "launch_config": deepcopy(app.launch_config or {}),
        }

    transport = execution_fields["transport"]
    launch_config = deepcopy(execution_fields["launch_config"])
    return {
        "id": app.app_id,
        "name": execution_fields["name"],
        "description": app.description,
        "icon": app.icon,
        "transport": transport,
        "provider": execution_fields["provider_name"],
        "category": app.category,
        "oauth_scopes": deepcopy(execution_fields["oauth_scopes"]),
        # Only builtin apps can declare these today (see
        # get_builtin_execution_fields_and_optional_scopes) - a custom
        # admin-created app has no column for it and always gets [].
        "optional_oauth_scopes": optional_oauth_scopes,
        "is_visible_in_connector": bool(app.is_visible_in_connector),
        "launch_config": launch_config,
        "auth_type": classify_app_auth(transport, launch_config),
    }


@dataclass(frozen=True)
class MCPAppSnapshot:
    """One catalog/server view for repeated canonical builtin validation."""

    catalog_apps: tuple[PublicMCPApp, ...]
    servers: tuple[Any, ...]


def load_mcp_app_snapshot(db: Session) -> MCPAppSnapshot:
    """Load the catalog and server rows once for one validation projection."""
    from .models.mcp import MCPServer

    return MCPAppSnapshot(
        catalog_apps=tuple(db.query(PublicMCPApp).all()),
        servers=tuple(db.query(MCPServer).all()),
    )


def get_all_mcp_apps(
    db: Session,
    *,
    snapshot: MCPAppSnapshot | None = None,
) -> List[Dict[str, Any]]:
    """Retrieve all MCP apps from the database or a fixed snapshot."""
    apps: Sequence[PublicMCPApp] = (
        snapshot.catalog_apps if snapshot is not None else db.query(PublicMCPApp).all()
    )
    return [_app_to_dict(app) for app in apps]


def get_app_by_id(db: Session, app_id: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its ID."""
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == app_id).first()
    return _app_to_dict(app) if app else None


def get_app_by_name(db: Session, name: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its exact name."""
    app = db.query(PublicMCPApp).filter(PublicMCPApp.name == name).first()
    return _app_to_dict(app) if app else None


class BuiltinOAuthServerDefinitionError(ValueError):
    """Raised when trusted builtin OAuth identity is absent or ambiguous."""


def _normalized_catalog_key(value: object) -> str | None:
    """Normalize only for collision detection, never for persisted identity."""
    if value is None:
        return None
    normalized = "-".join(str(value).strip().lower().split())
    return normalized or None


def _strict_catalog_app_by_id(
    db: Session,
    app_id: str,
    *,
    require_builtin_oauth: bool = False,
    require_visible: bool = False,
    snapshot: MCPAppSnapshot | None = None,
) -> Dict[str, Any]:
    """Resolve one exact app while rejecting normalized-ID collisions.

    Stable identity remains the exact ``PublicMCPApp.app_id``. Normalization is
    used only to detect an administrator-authored collision that a looser route
    or UI lookup could otherwise resolve inconsistently.
    """
    if not isinstance(app_id, str) or not app_id or app_id != app_id.strip():
        raise BuiltinOAuthServerDefinitionError(
            "builtin OAuth app_id must be an exact non-empty string"
        )

    catalog_apps: Sequence[PublicMCPApp] = (
        snapshot.catalog_apps if snapshot is not None else db.query(PublicMCPApp).all()
    )
    matches = [app for app in catalog_apps if app.app_id == app_id]
    if len(matches) != 1:
        raise BuiltinOAuthServerDefinitionError(
            f"builtin OAuth catalog app {app_id!r} is unavailable"
        )
    app = matches[0]

    normalized_id = _normalized_catalog_key(app_id)
    collisions = [
        candidate
        for candidate in catalog_apps
        if _normalized_catalog_key(candidate.app_id) == normalized_id
    ]
    if len(collisions) != 1:
        raise BuiltinOAuthServerDefinitionError(
            f"builtin OAuth catalog app {app_id!r} has ambiguous normalized identity"
        )

    execution_fields, _optional_scopes = (
        get_builtin_execution_fields_and_optional_scopes(app.app_id)
    )
    if require_builtin_oauth and execution_fields is None:
        raise BuiltinOAuthServerDefinitionError(
            f"OAuth catalog app {app_id!r} is absent from the builtin registry"
        )
    if execution_fields is not None:
        persisted_execution = {
            "name": app.name,
            "transport": app.transport,
            "provider_name": app.provider_name,
            "oauth_scopes": list(app.oauth_scopes or []),
            "launch_config": app.launch_config or {},
        }
        expected_execution = {
            "name": execution_fields["name"],
            "transport": execution_fields["transport"],
            "provider_name": execution_fields["provider_name"],
            "oauth_scopes": list(execution_fields["oauth_scopes"]),
            "launch_config": execution_fields["launch_config"],
        }
        drifted_fields = sorted(
            field
            for field, expected in expected_execution.items()
            if persisted_execution[field] != expected
        )
        if drifted_fields:
            raise BuiltinOAuthServerDefinitionError(
                f"builtin OAuth catalog app {app_id!r} has persisted catalog "
                f"drift ({', '.join(drifted_fields)})"
            )

    app_info = _app_to_dict(app)
    if require_builtin_oauth and app_info.get("auth_type") != "builtin_oauth":
        raise BuiltinOAuthServerDefinitionError(
            f"catalog app {app_id!r} is not builtin OAuth"
        )
    if require_visible and not app_info.get("is_visible_in_connector"):
        raise BuiltinOAuthServerDefinitionError(
            f"builtin OAuth catalog app {app_id!r} is unavailable"
        )
    return app_info


_CANONICAL_EMPTY_SERVER_FIELDS = (
    "command",
    "args",
    "url",
    "env",
    "cwd",
    "headers",
    "timeout",
    "runtime_input_schema",
    "runtime_bindings",
    "docker_url",
    "docker_image",
    "docker_environment",
    "docker_working_dir",
    "volumes",
    "bind_ports",
    "auto_start",
    "container_id",
    "container_name",
    "container_logs",
)


def _expected_builtin_auth(app_info: Mapping[str, Any]) -> dict[str, str]:
    expected = {"app_id": str(app_info["id"])}
    if app_info.get("provider"):
        expected["provider"] = str(app_info["provider"])
    return expected


def _adopt_builtin_auth(server: Any, app_info: Mapping[str, Any]) -> None:
    """Repair only missing canonical auth fields on a legacy server."""
    expected = _expected_builtin_auth(app_info)
    auth = getattr(server, "auth", None)
    if auth is None:
        server.auth = expected
        return
    if not isinstance(auth, Mapping):
        return

    unknown_keys = set(auth) - set(expected)
    mismatched = any(
        key in auth and str(auth[key]) != value for key, value in expected.items()
    )
    if not unknown_keys and not mismatched:
        server.auth = expected


def _validate_canonical_builtin_oauth_server(
    server: Any, app_info: Mapping[str, Any]
) -> None:
    """Reject every stored field that could supply non-catalog execution data."""
    app_id = str(app_info["id"])
    allowed_names = {app_id, str(app_info["name"])}
    failures: list[str] = []
    if str(getattr(server, "name", "")) not in allowed_names:
        failures.append("name")
    if getattr(server, "managed", None) != "external":
        failures.append("managed")
    if getattr(server, "transport", None) != "oauth":
        failures.append("transport")

    for field_name in _CANONICAL_EMPTY_SERVER_FIELDS:
        value = getattr(server, field_name, None)
        if value not in (None, "", [], {}):
            failures.append(field_name)
    if bool(getattr(server, "concurrency_safe", False)):
        failures.append("concurrency_safe")
    if getattr(server, "concurrent_tools", None) not in (None, []):
        failures.append("concurrent_tools")
    if bool(getattr(server, "allow_delegated_authorization", False)):
        failures.append("allow_delegated_authorization")
    if getattr(server, "restart_policy", None) not in (None, "no"):
        failures.append("restart_policy")

    auth = getattr(server, "auth", None)
    if not isinstance(auth, Mapping) or dict(auth) != _expected_builtin_auth(app_info):
        failures.append("auth")

    if failures:
        raise BuiltinOAuthServerDefinitionError(
            f"OAuth app {app_id!r} conflicts with an existing MCP server: "
            f"{getattr(server, 'name', '')!r} is not a canonical builtin OAuth "
            f"definition ({', '.join(sorted(set(failures)))})"
        )


def _ensure_sqlite_savepoint_root(db: Session) -> None:
    """Ensure SQLite SAVEPOINT writes remain owned by the caller transaction."""
    connection = db.connection()
    driver_connection = connection.connection.driver_connection
    if connection.dialect.name == "sqlite" and not bool(
        getattr(driver_connection, "in_transaction", False)
    ):
        connection.exec_driver_sql("BEGIN")


def _is_expected_unique_violation(
    exc: IntegrityError, *, constraint_names: set[str], sqlite_columns: str
) -> bool:
    """Classify only the unique races this helper can safely recover from."""
    original = exc.orig
    sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    constraint_name = getattr(getattr(original, "diag", None), "constraint_name", None)
    if sqlstate == "23505":
        return constraint_name in constraint_names
    message = str(original).lower()
    return "unique constraint failed" in message and sqlite_columns in message


def _builtin_server_candidates(
    db: Session,
    app_info: Mapping[str, Any],
    *,
    snapshot: MCPAppSnapshot | None = None,
) -> list[Any]:
    """Return candidates for one app and reject reserved-name conflicts."""
    from .models.mcp import MCPServer

    app_id = str(app_info["id"])
    app_name = str(app_info["name"])
    normalized_names = {
        key for key in map(_normalized_catalog_key, (app_id, app_name)) if key
    }
    catalog_apps: Sequence[PublicMCPApp] = (
        snapshot.catalog_apps if snapshot is not None else db.query(PublicMCPApp).all()
    )
    servers: Sequence[Any] = (
        snapshot.servers if snapshot is not None else db.query(MCPServer).all()
    )
    legacy_name_matches = [app for app in catalog_apps if app.name == app_name]
    legacy_name_is_unique = len({str(app.app_id) for app in legacy_name_matches}) == 1

    candidates: list[Any] = []
    for server in servers:
        server_auth: Mapping[str, Any] = (
            server.auth if isinstance(server.auth, Mapping) else {}
        )
        server_app_id = server_auth.get("app_id")
        exact_name_candidate = server.name == app_id or (
            server.name == app_name and legacy_name_is_unique
        )
        if server_app_id == app_id or (
            "app_id" not in server_auth and exact_name_candidate
        ):
            candidates.append(server)
            continue
        if (
            _normalized_catalog_key(server_app_id) in normalized_names
            or _normalized_catalog_key(server.name) in normalized_names
        ):
            raise BuiltinOAuthServerDefinitionError(
                f"builtin OAuth app {app_id!r} has an ambiguous reserved server identity"
            )

    if not legacy_name_is_unique and any(
        server.name == app_name
        and not (
            isinstance(server.auth, Mapping) and server.auth.get("app_id") == app_id
        )
        for server in servers
    ):
        raise BuiltinOAuthServerDefinitionError(
            f"builtin OAuth app {app_id!r} has ambiguous legacy catalog identity"
        )
    return candidates


def classify_actor_builtin_oauth_server(
    db: Session, server: Any
) -> Dict[str, Any] | None:
    """Classify one actor-visible row as canonical builtin OAuth or native.

    Stable ``auth.app_id`` wins when present. Legacy exact app-id/display-name
    rows remain supported, while normalized aliases stay reserved and fail
    closed instead of falling through to a native MCP transport.
    """
    catalog_apps = [
        app_info
        for app in db.query(PublicMCPApp).all()
        if (app_info := _app_to_dict(app)).get("auth_type") == "builtin_oauth"
    ]
    auth = getattr(server, "auth", None)
    has_app_id = isinstance(auth, Mapping) and "app_id" in auth
    server_app_id = auth.get("app_id") if isinstance(auth, Mapping) else None
    server_name = str(getattr(server, "name", ""))
    normalized_name = _normalized_catalog_key(server_name)
    normalized_app_id = _normalized_catalog_key(server_app_id)

    exact_app = next(
        (
            app_info
            for app_info in catalog_apps
            if has_app_id and server_app_id == app_info.get("id")
        ),
        None,
    )
    exact_legacy_apps = [
        app_info
        for app_info in catalog_apps
        if not has_app_id
        and server_name in {str(app_info.get("id")), str(app_info.get("name"))}
    ]
    reserved_apps = [
        app_info
        for app_info in catalog_apps
        if {
            normalized_name,
            normalized_app_id,
        }
        & {
            _normalized_catalog_key(app_info.get("id")),
            _normalized_catalog_key(app_info.get("name")),
        }
        - {None}
    ]

    if has_app_id:
        if exact_app is None:
            if reserved_apps:
                raise BuiltinOAuthServerDefinitionError(
                    "reserved builtin OAuth server identity conflicts with auth.app_id"
                )
            return None
        if any(app_info.get("id") != exact_app.get("id") for app_info in reserved_apps):
            raise BuiltinOAuthServerDefinitionError(
                "builtin OAuth server has ambiguous reserved catalog identity"
            )
        app_info = exact_app
    else:
        if len(exact_legacy_apps) != 1:
            if exact_legacy_apps or reserved_apps:
                raise BuiltinOAuthServerDefinitionError(
                    "builtin OAuth server has ambiguous legacy catalog identity"
                )
            return None
        app_info = exact_legacy_apps[0]
        if any(app.get("id") != app_info.get("id") for app in reserved_apps):
            raise BuiltinOAuthServerDefinitionError(
                "builtin OAuth server has ambiguous reserved catalog identity"
            )

    strict_app_info = _strict_catalog_app_by_id(
        db,
        str(app_info["id"]),
        require_builtin_oauth=True,
        require_visible=True,
    )
    candidates = _builtin_server_candidates(db, strict_app_info)
    if len(candidates) != 1 or getattr(candidates[0], "id", None) != getattr(
        server, "id", None
    ):
        raise BuiltinOAuthServerDefinitionError(
            f"builtin OAuth app {strict_app_info['id']!r} must have exactly one "
            "MCP server definition"
        )
    _validate_canonical_builtin_oauth_server(server, strict_app_info)
    return strict_app_info


def require_builtin_oauth_server_definition(
    db: Session,
    *,
    app_id: str,
    provider: str,
    snapshot: MCPAppSnapshot | None = None,
) -> Any:
    """Require one canonical server for an exact visible app/provider pair."""
    app_info = _strict_catalog_app_by_id(
        db,
        app_id,
        require_builtin_oauth=True,
        require_visible=True,
        snapshot=snapshot,
    )
    if app_info.get("provider") != provider:
        raise BuiltinOAuthServerDefinitionError(
            f"builtin OAuth app {app_id!r} does not use provider {provider!r}"
        )
    candidates = _builtin_server_candidates(db, app_info, snapshot=snapshot)
    if len(candidates) != 1:
        raise BuiltinOAuthServerDefinitionError(
            f"builtin OAuth app {app_id!r} must have exactly one MCP server definition"
        )
    _validate_canonical_builtin_oauth_server(candidates[0], app_info)
    return candidates[0]


def ensure_builtin_oauth_server_definition(db: Session, *, app_id: str) -> Any:
    """Resolve or create one canonical visible builtin OAuth server.

    The caller owns commit/rollback. Expected concurrent insert races are
    isolated to a SAVEPOINT so unrelated caller work is never rolled back.
    """
    from .models.mcp import MCPServer

    if not isinstance(app_id, str) or not app_id.strip():
        raise BuiltinOAuthServerDefinitionError(
            "builtin OAuth app_id must be a non-empty string"
        )
    app_info = _strict_catalog_app_by_id(
        db,
        app_id,
        require_builtin_oauth=True,
        require_visible=True,
    )
    candidates = _builtin_server_candidates(db, app_info)
    if len(candidates) > 1:
        raise BuiltinOAuthServerDefinitionError(
            f"builtin OAuth app {app_id!r} has multiple MCP server definitions"
        )

    server = candidates[0] if candidates else None
    if server is None:
        expected_auth = _expected_builtin_auth(app_info)
        proposed = MCPServer(
            name=str(app_info["name"]),
            description=app_info.get("description"),
            managed="external",
            transport="oauth",
            auth=expected_auth,
        )
        _ensure_sqlite_savepoint_root(db)
        try:
            with db.begin_nested():
                db.add(proposed)
                db.flush()
            server = proposed
        except IntegrityError as exc:
            if not _is_expected_unique_violation(
                exc,
                constraint_names={"mcp_servers_name_key", "ix_mcp_servers_name"},
                sqlite_columns="mcp_servers.name",
            ):
                raise
            candidates = _builtin_server_candidates(db, app_info)
            if len(candidates) != 1:
                raise BuiltinOAuthServerDefinitionError(
                    f"builtin OAuth app {app_id!r} did not converge on one server"
                ) from exc
            server = candidates[0]

    _adopt_builtin_auth(server, app_info)
    _validate_canonical_builtin_oauth_server(server, app_info)
    server.description = app_info.get("description") or server.description
    db.flush()
    return server


def ensure_builtin_oauth_server_visibility_for_user(
    db: Session, *, user_id: int, app_id: str
) -> Any:
    """Ensure canonical builtin visibility for one trusted internal account."""
    from .models.mcp import UserMCPServer

    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("user_id must be a persisted positive integer")
    server = ensure_builtin_oauth_server_definition(db, app_id=app_id)
    association = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user_id,
            UserMCPServer.mcpserver_id == int(server.id),
        )
        .one_or_none()
    )
    if association is None:
        proposed = UserMCPServer(
            user_id=user_id,
            mcpserver_id=int(server.id),
            is_owner=False,
            can_edit=False,
            can_delete=False,
            is_shared=False,
            is_active=True,
        )
        _ensure_sqlite_savepoint_root(db)
        try:
            with db.begin_nested():
                db.add(proposed)
                db.flush()
            association = proposed
        except IntegrityError as exc:
            if not _is_expected_unique_violation(
                exc,
                constraint_names={"uq_user_mcpservers"},
                sqlite_columns="user_mcpservers.user_id, user_mcpservers.mcpserver_id",
            ):
                raise
            association = (
                db.query(UserMCPServer)
                .filter(
                    UserMCPServer.user_id == user_id,
                    UserMCPServer.mcpserver_id == int(server.id),
                )
                .one_or_none()
            )
            if association is None:
                raise BuiltinOAuthServerDefinitionError(
                    "builtin OAuth visibility race did not converge"
                ) from exc
    cast_association: Any = association
    cast_association.is_active = True
    db.flush()
    return server


def get_app_for_mcp_server(db: Session, server: Any) -> Dict[str, Any] | None:
    """Resolve a server's catalog app by stable identity when it is available.

    Unstamped rows predate ``auth.app_id`` and are resolved by the exact name
    they were provisioned under -- the app id for catalog-connect rows, the
    display name for builtin OAuth rows. Because a bare name cannot say which
    convention wrote it, such a row resolves only when both namespaces agree
    on a single owner; anything ambiguous answers ``None`` so callers fail
    closed. Once a row carries ``app_id``, an invalid value must not fall back
    to a same-named app because that could select another connector's
    credentials or launch configuration.

    ``None`` therefore means "cannot prove whose this is", not merely "not
    found": destructive callers must leave credentials in place rather than
    treat it as nothing to do.
    """
    auth = getattr(server, "auth", None)
    if isinstance(auth, Mapping) and "app_id" in auth:
        app_id = auth.get("app_id")
        if not isinstance(app_id, str) or not app_id:
            return None
        return get_app_by_id(db, app_id)
    name = str(getattr(server, "name", ""))
    if not name:
        return None
    # Both provisioning conventions write ``MCPServer.name``: the catalog
    # connect helpers store the app **id** (``_ensure_catalog_app_server``,
    # ``_ensure_catalog_mcp_oauth_server``) while the builtin OAuth flow stores
    # the **display name** (``_ensure_user_mcp_server``). Resolving only by
    # display name leaves every id-named row unresolvable, which silently skips
    # whatever the caller does with the result -- for the disconnect path that
    # meant the user's OAuth credentials survived a successful teardown.
    #
    # Both namespaces are therefore enumerated together, hidden rows included,
    # and the row resolves only when they agree on a single owner. An unstamped
    # row carries no provenance of its own, so preferring one namespace over
    # the other would be a guess: ``app_id`` being unique proves that at most
    # one app holds that id, not that this row was provisioned from it rather
    # than from another app's (mutable, non-unique) display name. Both readings
    # are legal, so ambiguity is reported as "cannot resolve" and every caller
    # fails closed -- deletion keeps the credentials, listing and runtime
    # decline to name an app -- rather than acting on a coin flip. Only the
    # stamp settles it, which is what the branch above is for.
    candidates = (
        db.query(PublicMCPApp)
        .filter((PublicMCPApp.app_id == name) | (PublicMCPApp.name == name))
        .all()
    )
    owners = {str(candidate.app_id) for candidate in candidates}
    if len(owners) != 1:
        return None
    return _app_to_dict(candidates[0])

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Callable, Iterator, Protocol

from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)
from .mcp_oauth import MCPOAuthRuntimeError, resolve_mcp_oauth_runtime_auth
from .user_oauth import normalize_user_oauth_resource_owner_key

HTTP_MCP_TRANSPORTS = frozenset({"sse", "websocket", "streamable_http"})


class MCPBuiltinOAuthActorPolicyRequiredError(RuntimeError):
    """A marked task was used without its trusted actor policy."""


class MCPBuiltinOAuthActorPolicyMismatchError(RuntimeError):
    """A task's already-bound actor policy was replaced."""


@dataclass(frozen=True)
class MCPBuiltinOAuthActorPolicy:
    """Trusted actor owner namespace for builtin OAuth MCP execution.

    Server visibility and builtin classification remain xagent runtime
    decisions. The caller supplies only the immutable credential owner.
    """

    resource_owner_key: str = dataclass_field(repr=False)

    def __post_init__(self) -> None:
        owner_key = normalize_user_oauth_resource_owner_key(self.resource_owner_key)
        if owner_key is None:  # pragma: no cover - normalization preserves None only
            raise ValueError("resource_owner_key must not be null")
        object.__setattr__(self, "resource_owner_key", owner_key)


@dataclass(frozen=True)
class MCPRuntimeConnectionBuild:
    """Executable MCP connection plus any runtime authorization diagnostic."""

    connection: dict[str, Any] | None
    diagnostic: dict[str, Any] | None = None


def load_user_env_overrides(db: Any, user_id: int | None) -> dict[int, dict]:
    """Batch-load a user's decrypted per-user env overrides, keyed by server id.

    One query for all of a user's active overrides, so callers building many
    connections in a loop avoid an N+1 per-server lookup.
    """
    if not isinstance(user_id, int) or db is None:
        return {}
    from ...core.utils.encryption import decrypt_env_dict
    from ..models.mcp import UserMCPServer

    rows = (
        db.query(UserMCPServer.mcpserver_id, UserMCPServer.env)
        .filter(
            UserMCPServer.user_id == user_id,
            UserMCPServer.is_active,
            UserMCPServer.env.isnot(None),
        )
        .all()
    )
    overrides: dict[int, dict] = {}
    for mcpserver_id, env in rows:
        decrypted = decrypt_env_dict(env)
        if decrypted:
            overrides[mcpserver_id] = decrypted
    return overrides


def load_user_env_sources(db: Any, user_id: int | None) -> dict[int, str]:
    """Batch-load a user's env-source pick per server, keyed by server id.

    The pick (own/shared/platform) selects which env layer the runtime uses;
    a missing entry (NULL) means the legacy fallback (global < shared < user).
    """
    if not isinstance(user_id, int) or db is None:
        return {}
    from ..models.mcp import UserMCPServer

    rows = (
        db.query(UserMCPServer.mcpserver_id, UserMCPServer.env_source)
        .filter(
            UserMCPServer.user_id == user_id,
            UserMCPServer.is_active,
            UserMCPServer.env_source.isnot(None),
        )
        .all()
    )
    return {mcpserver_id: source for mcpserver_id, source in rows}


# Hook signature: (db, user_id) -> {mcpserver_id: {env}}. An application layer
# (e.g. a multi-tenant deployment) can inject a shared env layer that sits between
# the global and per-user env via set_mcp_shared_env_hook(). The core is agnostic
# to what "shared" means (team, org, ...); default: no shared layer. For team
# semantics specifically -- keying the shared layer on the team that owns the
# governing agent, rather than on the running user -- install the separate,
# team-keyed socket below instead: see set_mcp_team_env_hook.
_get_mcp_shared_env_hook: Callable[[Any, int | None], dict[int, dict]] | None = None


def set_mcp_shared_env_hook(
    hook: Callable[[Any, int | None], dict[int, dict]] | None,
) -> None:
    global _get_mcp_shared_env_hook
    _get_mcp_shared_env_hook = hook


def load_shared_env_overrides(db: Any, user_id: int | None) -> dict[int, dict]:
    """Batch-load a user's shared (app-injected) env overrides, keyed by server id.

    Empty unless an application layer registered a hook (see set_mcp_shared_env_hook).
    """
    if _get_mcp_shared_env_hook is None or not isinstance(user_id, int) or db is None:
        return {}
    # The hook is external application code; a failure there must not take down
    # MCP tool loading or the catalog endpoint. Degrade to no shared layer.
    try:
        return _get_mcp_shared_env_hook(db, user_id) or {}
    except Exception:
        import logging

        logging.getLogger(__name__).exception("shared env hook failed")
        return {}


class TeamMCPEnvHook(Protocol):
    """Resolver for a team's own shared MCP env overrides, keyed by server id.

    Typed as a ``Protocol`` with a keyword-only ``team_id`` for the same
    reason ``TeamConnectorVisibilityHook`` (the visibility twin in
    connector_team_scope.py) is: this hook and the user-keyed
    ``_get_mcp_shared_env_hook`` above are otherwise structurally
    identical -- ``(db, id) -> {mcpserver_id: {env}}`` -- and a swapped
    install would type-check while resolving an unrelated team's or
    user's env.
    """

    def __call__(self, db: Any, *, team_id: int) -> dict[int, dict]: ...


# Hook signature: (db, *, team_id) -> {mcpserver_id: {env}}. Deliberately a
# separate module-global slot from _get_mcp_shared_env_hook, not a keyword on
# set_mcp_shared_env_hook: the two hooks answer different questions -- "did
# the application register anything shared at all" (user-keyed, default:
# none) versus "does THIS SPECIFIC governing team have a row" (team-keyed,
# authorization-relevant) -- and load_shared_env_overrides's failure policy
# (degrade to no shared layer) would hide a signature TypeError or a
# malformed answer from this hook instead of surfacing it.
_get_mcp_team_env_hook: TeamMCPEnvHook | None = None


def set_mcp_team_env_hook(hook: TeamMCPEnvHook | None) -> None:
    global _get_mcp_team_env_hook
    _get_mcp_team_env_hook = hook


def team_env_hook_installed() -> bool:
    """Whether an application installed a team-keyed env hook.

    Callers select on this, never on an empty return value: an installed
    hook legitimately answers empty for a team that stored nothing for any
    server (see load_team_env_overrides).
    """
    return _get_mcp_team_env_hook is not None


_team_env_hook_missing_warned = False


def warn_team_env_hook_missing_once(*, team_id: int, connector_count: int) -> None:
    """Warn, once per process, that team-owned connectors are in scope while
    no team-keyed env hook is installed.

    An application that installed the connector visibility hook but not
    set_mcp_team_env_hook keeps the pre-existing, user-keyed shared env
    layer for team-owned servers, which resolves on whatever that hook was
    wired to -- typically the running user's own team, not the team that
    owns the governing agent. That is a half-installed state, and it is the
    only state this warns about: an application that installs neither hook
    never reaches this call, because without the visibility hook no server
    is ever team-owned and the caller's own scope check is already empty.

    Once per process rather than once per run: the condition is an
    installation fact that cannot change while the process is up, so
    repeating it per run would add volume without adding signal.
    """
    global _team_env_hook_missing_warned
    if _team_env_hook_missing_warned:
        return
    _team_env_hook_missing_warned = True
    logging.getLogger(__name__).warning(
        "Team-owned MCP connectors are in scope (team_id=%s, %s connector(s)) "
        "but no team-keyed env hook is installed: their shared env layer "
        "stays keyed on the running user, not on the team that owns the "
        "governing agent (see set_mcp_team_env_hook)",
        team_id,
        connector_count,
    )


def _validate_team_mcp_env_answer(answer: Any) -> dict[int, dict]:
    """Validate the team-env hook's answer shape.

    This is an authorization input, not user-facing data: a malformed
    answer must fail loudly, never be normalized, coerced, or defaulted to
    empty -- the same stance connector_team_scope's
    ``_validate_team_connector_answer`` takes for the visibility twin. The
    hook must return a ``dict`` whose keys are all ``int`` server ids
    (``bool`` is a subclass of ``int`` in Python but is rejected here -- a
    truthy/falsy value is never a legitimate server id) and whose values
    are all ``dict`` env mappings whose own keys and values are all
    ``str``: this env eventually becomes a subprocess environment, where a
    non-string surfaces as an unclassified failure at spawn time rather
    than as the typed error this boundary promises. There is no ``bool``
    special case on the inner side, unlike the server-id keys above,
    because ``bool`` is not a ``str`` subclass and is already rejected.
    A malformed env value's own content is never formatted into the error
    message -- these are credentials.
    """
    if not isinstance(answer, dict):
        raise ValueError(
            "team env hook returned a malformed answer: expected a dict, "
            f"got {type(answer).__name__}"
        )
    for key, value in answer.items():
        if isinstance(key, bool):
            raise ValueError(
                "team env hook returned a malformed answer: key "
                f"{key!r} is not a server id (must be int, not bool)"
            )
        if not isinstance(key, int):
            raise ValueError(
                "team env hook returned a malformed answer: key "
                f"{key!r} is not a server id (must be int, got "
                f"{type(key).__name__})"
            )
        if not isinstance(value, dict):
            raise ValueError(
                "team env hook returned a malformed answer: value for key "
                f"{key!r} must be a dict, got {type(value).__name__}"
            )
        for env_key, env_value in value.items():
            if not isinstance(env_key, str):
                raise ValueError(
                    "team env hook returned a malformed answer: env key "
                    f"{env_key!r} under server id {key!r} is not a string "
                    f"(got {type(env_key).__name__})"
                )
            if not isinstance(env_value, str):
                raise ValueError(
                    "team env hook returned a malformed answer: env value "
                    f"for env key {env_key!r} under server id {key!r} is "
                    f"not a string (got {type(env_value).__name__})"
                )
    # Defensive copy at the authorization boundary: the caller must not be
    # able to mutate the installing application's own answer object (or
    # vice versa). Shallow is sufficient -- every downstream consumer only
    # ever reads the inner per-server env dicts, never mutates them in
    # place (see the shallow-copy note in config.py's wiring block).
    return dict(answer)


def load_team_env_overrides(db: Any, team_id: int | None) -> dict[int, dict]:
    """Batch-load a governing team's own shared MCP env overrides, keyed by
    server id.

    Unlike load_shared_env_overrides, this loader's failure policy is
    raise, not degrade: any hook exception and any answer shape violation
    becomes a typed ConnectorRuntimeError instead of silently falling back
    to no shared layer. A misbehaving hook here must not be
    indistinguishable from "the governing team legitimately has no row" --
    the caller (config.py's wiring) already treats the latter as a
    deliberate, signal-losing degrade of its own; collapsing hook failure
    into the same outcome would double that loss silently.

    A row whose env mapping is empty is dropped here rather than passed on:
    an empty mapping carries nothing, and letting it through as "the team
    has a row" would make storing an empty row strictly worse for the
    runner than storing nothing at all -- the caller's row-present branch
    keeps a "shared" pick that then merges nothing, pinning the run to the
    global env, whereas a missing row degrades that pick and lets the
    runner's own key through. load_user_env_overrides above applies the
    same convention to a per-user override that decrypts to an empty
    mapping.
    """
    if db is None or team_id is None or _get_mcp_team_env_hook is None:
        return {}
    try:
        answer = _get_mcp_team_env_hook(db, team_id=int(team_id))
        validated = _validate_team_mcp_env_answer(answer)
        # Validate first, then drop rows whose env mapping is empty. The
        # order matters: dropping first would let a malformed server-id key
        # with an empty value escape the shape check entirely.
        return {server_id: env for server_id, env in validated.items() if env}
    except ConnectorRuntimeError:
        raise
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "team env hook failed for team %s", team_id, exc_info=True
        )
        raise ConnectorRuntimeError(
            ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
            "Team MCP env is unavailable.",
            details={"reason": "team_env_resolution_failed"},
            status_code=503,
        ) from exc


@contextmanager
def mcp_env_hooks_snapshot() -> Iterator[None]:
    """Save every env hook slot in this module on entry, restore all of
    them on exit.

    Covers both the pre-existing shared (user-keyed) slot and the
    team-keyed slot above, so a test that installs either hook is
    restored to the pre-entry state deterministically without reaching
    past the public setters. A hook slot added to this module later,
    following the same ``_get_mcp_*_hook`` naming pattern as the two
    above, that is not added here as well fails the coverage test in
    tests/web/services/test_mcp_team_env_hook.py, by design -- see that
    module's docstring rule. A slot that breaks the naming pattern is
    outside what that coverage test can discover and must be added here
    by hand.
    """
    global _get_mcp_shared_env_hook, _get_mcp_team_env_hook
    saved_shared = _get_mcp_shared_env_hook
    saved_team = _get_mcp_team_env_hook
    try:
        yield
    finally:
        _get_mcp_shared_env_hook = saved_shared
        _get_mcp_team_env_hook = saved_team


CALLER_ID_ENV_VAR = "XAGENT_MCP_CALLER_ID"


def caller_id_env(user_id: int | None) -> dict[str, str]:
    """Env layer identifying which xagent user triggered this stdio MCP call.

    A stdio connector runs as a short-lived per-call subprocess with no other
    way to learn who invoked it. Exposing the caller's user id lets a
    connector attribute its own external API calls (e.g. an AWS STS
    AssumeRole session name) to a specific user in the target system's own
    audit trail, instead of every user's calls being indistinguishable.
    """
    if not isinstance(user_id, int):
        return {}
    return {CALLER_ID_ENV_VAR: str(user_id)}


def merge_stdio_env(
    global_env: dict[str, Any] | None,
    *layers: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge env layers over the global env; later layers win (global is fallback).

    Two-arg form keeps the original meaning (global, user). A middle shared layer
    is passed as (global, shared, user) so precedence is global < shared < user.
    """
    result = dict(global_env or {})
    changed = False
    for layer in layers:
        if layer:
            result.update(layer)
            changed = True
    return result if changed else global_env


def resolve_stdio_env(
    env_source: str | None,
    global_env: dict[str, Any] | None,
    shared_env: dict[str, Any] | None,
    user_env: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Pick which env layers to merge over the global env, per the user's choice.

    - "platform": global only (ignore shared and user)
    - "shared":   global < shared (ignore user)
    - "own":      global < user   (ignore shared)
    - None:       global < shared < user  (legacy fallback; most specific wins)

    The stored own key is never deleted when another source is picked — it is
    simply not merged — so switching back to "own" needs no re-entry.

    This governs only the shared/own/global layers above. The caller-id layer
    (see caller_id_env, applied in build_mcp_runtime_connection after this
    function returns) is appended unconditionally for every stdio server
    regardless of env_source — it identifies the calling user, not a
    connector-owned credential, so it is not subject to this precedence.
    """
    if env_source == "platform":
        return global_env
    if env_source == "shared":
        return merge_stdio_env(global_env, shared_env)
    if env_source == "own":
        return merge_stdio_env(global_env, user_env)
    return merge_stdio_env(global_env, shared_env, user_env)


async def build_mcp_runtime_connection(
    db: Any,
    server: Any,
    *,
    user_id: int | None,
    mcp_auth_context: dict[str, Any] | None = None,
    user_env_overrides: dict[int, dict] | None = None,
    shared_env_overrides: dict[int, dict] | None = None,
    env_source_overrides: dict[int, str] | None = None,
) -> MCPRuntimeConnectionBuild:
    """Build an executable MCP connection for a specific user runtime.

    Static MCP connection serialization belongs to the MCP server model. Per-user
    MCP OAuth grant selection belongs here, at the web runtime boundary.
    """
    connection = server.to_connection_dict()

    # Resolve which env layer to apply for this user, per their env_source pick
    # (stdio only; global env is the fallback). Overrides are prefetched by the
    # caller (see load_user_env_overrides / load_user_env_sources) to avoid N+1.
    if connection.get("transport") == "stdio":
        if user_env_overrides or shared_env_overrides or env_source_overrides:
            server_id = getattr(server, "id", None)
            if isinstance(server_id, int):
                merged = resolve_stdio_env(
                    (env_source_overrides or {}).get(server_id),
                    connection.get("env"),
                    (shared_env_overrides or {}).get(server_id),
                    (user_env_overrides or {}).get(server_id),
                )
                if merged:
                    connection["env"] = merged
        # Applied to every stdio server unconditionally — including
        # arbitrary third-party stdio commands a user registers directly via
        # create_mcp_server, not just catalog connectors — since the value
        # is only an opaque internal xagent user id, not a credential or PII.
        caller_env = caller_id_env(user_id)
        if caller_env:
            connection["env"] = {**(connection.get("env") or {}), **caller_env}

    auth_config = server._decrypt_auth_config(getattr(server, "auth", None))
    if not _is_mcp_oauth_http_server(server, auth_config):
        return MCPRuntimeConnectionBuild(connection=connection)

    server_id = getattr(server, "id", None)
    if not isinstance(server_id, int) or not isinstance(user_id, int) or db is None:
        return MCPRuntimeConnectionBuild(
            connection=None,
            diagnostic=mcp_oauth_runtime_diagnostic(
                server,
                code="authorization_required",
                message=(
                    "MCP OAuth runtime requires a persisted server, user, and "
                    "database session"
                ),
            ),
        )

    selection = mcp_oauth_selection(mcp_auth_context, server_id)
    resource_owner_key = selection.get("resource_owner_key") or f"xagent:user:{user_id}"
    try:
        runtime_auth = await resolve_mcp_oauth_runtime_auth(
            db,
            server_id=server_id,
            user_id=user_id,
            auth_config=auth_config,
            resource_owner_key=str(resource_owner_key),
            resource=selection.get("resource"),
            scope=selection.get("scope"),
            issuer=selection.get("issuer"),
        )
    except MCPOAuthRuntimeError as exc:
        return MCPRuntimeConnectionBuild(
            connection=None,
            diagnostic=mcp_oauth_runtime_diagnostic(
                server,
                code=exc.code,
                message=exc.message,
                resource_owner_key=str(resource_owner_key),
                resource=selection.get("resource") or auth_config.get("resource"),
                scope=selection.get("scope") or auth_config.get("scope"),
                issuer=selection.get("issuer") or auth_config.get("issuer"),
            ),
        )

    connection = connection_with_bearer_authorization(
        connection, runtime_auth.access_token
    )
    connection.pop("auth", None)
    return MCPRuntimeConnectionBuild(connection=connection)


def connection_to_transport_config(connection: dict[str, Any]) -> dict[str, Any]:
    """Convert a direct MCP connection dict into WebToolConfig transport config."""
    return {
        key: value
        for key, value in connection.items()
        if key not in {"name", "transport"}
    }


def mcp_oauth_selection(
    mcp_auth_context: dict[str, Any] | None, server_id: int
) -> dict[str, Any]:
    """Return the runtime grant selection for one server."""
    selection = (mcp_auth_context or {}).get(str(server_id))
    return selection if isinstance(selection, dict) else {}


def effective_mcp_oauth_resource(
    server: Any, *, mcp_auth_context: dict[str, Any] | None
) -> str | None:
    """Select the first configured OAuth resource without normalizing it."""
    server_id = getattr(server, "id", None)
    selection = (
        mcp_oauth_selection(mcp_auth_context, server_id)
        if isinstance(server_id, int)
        else {}
    )
    auth_config = server._decrypt_auth_config(getattr(server, "auth", None))
    configured_resource = (
        auth_config.get("resource") if isinstance(auth_config, dict) else None
    )
    for resource in (
        selection.get("resource"),
        configured_resource,
        getattr(server, "url", None),
    ):
        if isinstance(resource, str) and resource:
            return resource
    return None


def connection_without_authorization(
    connection: dict[str, Any],
) -> dict[str, Any]:
    """Copy a connection and its headers while removing Authorization values."""
    copied_connection = dict(connection)
    raw_headers = connection.get("headers")
    copied_connection["headers"] = headers_without_authorization(
        raw_headers if isinstance(raw_headers, dict) else None
    )
    return copied_connection


def connection_with_bearer_authorization(
    connection: dict[str, Any], access_token: str
) -> dict[str, Any]:
    """Copy a connection and replace static authorization with a Bearer token."""
    copied_connection = connection_without_authorization(connection)
    copied_connection["headers"]["Authorization"] = f"Bearer {access_token}"
    return copied_connection


def headers_without_authorization(headers: dict[str, Any] | None) -> dict[str, Any]:
    """Copy headers while removing any static Authorization credential."""
    return {
        str(key): value
        for key, value in dict(headers or {}).items()
        if str(key).lower() != "authorization"
    }


def mcp_oauth_runtime_diagnostic(
    server: Any,
    *,
    code: str,
    message: str,
    resource_owner_key: str | None = None,
    resource: Any | None = None,
    scope: Any | None = None,
    issuer: Any | None = None,
) -> dict[str, Any]:
    """Build the common runtime diagnostic payload for MCP OAuth failures."""
    return {
        "code": code,
        "message": message,
        "server_id": getattr(server, "id", None),
        "server_name": getattr(server, "name", None),
        "resource_owner_key": resource_owner_key,
        "resource": str(resource) if resource else None,
        "scope": str(scope) if scope else "",
        "issuer": str(issuer) if issuer else None,
    }


def _is_mcp_oauth_http_server(server: Any, auth_config: Any) -> bool:
    # Runtime classification of a *connected* server from its decrypted auth,
    # a different layer than the catalog auth_type (mcp_apps.classify_app_auth):
    # this also covers user-added custom HTTP servers that were never catalog
    # entries, so it stays independent by design.
    return (
        getattr(server, "transport", None) in HTTP_MCP_TRANSPORTS
        and isinstance(auth_config, dict)
        and auth_config.get("type") == "mcp_oauth"
    )

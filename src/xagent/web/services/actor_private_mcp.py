"""Actor-private remote MCP servers.

An embedding application can run several actors as one shared executing user.
It can then link a custom remote MCP server to that user on behalf of one actor
and declare the server private to that actor through this module's hook.

The hook answers ``{server_id: resource_owner_key}`` for every actor-private
server linked to the executing user. The engine reads that answer at these
points:

- ``WebToolConfig._visible_mcp_server_query``: a mapped server is visible only
  to a run whose actor policy names exactly its owner key; with no policy, no
  mapped server is visible. The task loader and the connector session both use
  this query.
- ``connector_runtime._load_visible_runtime_connectors``: mapped servers carry
  no runtime connector views, so they are always left out.
- ``WebToolConfig._build_mcp_server_config``: a mapped server uses only its
  owner's credential.
- ``connect_mcp_oauth_server_for_owner`` (``web/api/mcp.py``): OAuth starts only
  for the server's own owner.

Rows stay linked to the executing user, so other consumers of that user's
links still see them. An embedding application that links private rows must
not expose the executing user's own listings. #3898 tracks the fail-closed
alternative.

Like the team scope hooks, the hook is called through
``_call_connector_hook_gate`` on the caller's live session. A failed hook or a
malformed answer becomes the seam's typed 503: the engine never continues
without the filter.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)
from .connector_team_scope import (
    ConnectorHookSessionBoundaryError,
    _call_connector_hook_gate,
)
from .user_oauth import normalize_user_oauth_resource_owner_key

logger = logging.getLogger(__name__)

# hook(db, *, user_id: int) -> {server_id: resource_owner_key}
ActorPrivateMCPServersHook = Callable[..., Mapping[int, str]]

# The only transports an actor-private server may use.
ACTOR_PRIVATE_MCP_TRANSPORTS = frozenset({"sse", "streamable_http"})

# ``xagent:user:`` is the engine's default-owner namespace, i.e. the workspace
# credential of any executing user. No actor lives there, so every key under it
# is refused, not only the executing user's own.
RESERVED_OWNER_KEY_PREFIX = "xagent:user:"

_hook: ActorPrivateMCPServersHook | None = None


def set_actor_private_mcp_servers_hook(
    hook: ActorPrivateMCPServersHook | None,
) -> None:
    """Install (or clear with ``None``) the actor-private server hook."""
    global _hook
    _hook = hook


def actor_private_mcp_servers_hook_installed() -> bool:
    return _hook is not None


def actor_private_row_refused(server: Any, catalog_entries: Iterable[Any]) -> bool:
    """Whether a mapped row must not run as an actor-private server.

    A private server is a custom remote HTTP server, so any catalog claim
    (resolvable or not) refuses it: a transport other than remote HTTP, an
    ``auth.app_id`` or ``auth.builtin_provenance`` key whatever its value, or a
    name a catalog app owns. The task builder and the OAuth start share this
    rule, so OAuth never starts for a row a task would refuse.
    """
    from ..mcp_apps import match_catalog_entries

    if str(getattr(server, "transport", "") or "") not in ACTOR_PRIVATE_MCP_TRANSPORTS:
        return True
    auth = getattr(server, "auth", None)
    if isinstance(auth, Mapping) and ("app_id" in auth or "builtin_provenance" in auth):
        return True
    name = str(getattr(server, "name", "") or "")
    return bool(name) and bool(match_catalog_entries(catalog_entries, name=name))


def _validate_answer(answer: Any) -> dict[int, str]:
    if not isinstance(answer, Mapping):
        raise ValueError("actor-private MCP servers must be a mapping")
    validated: dict[int, str] = {}
    for server_id, owner_key in answer.items():
        if type(server_id) is not int:
            raise ValueError("actor-private MCP server ids must be integers")
        # Compared exactly with the policy's owner key, so the answer must
        # already be in the policy's normalized form.
        if (
            not isinstance(owner_key, str)
            or normalize_user_oauth_resource_owner_key(owner_key) != owner_key
        ):
            raise ValueError("actor-private MCP owner keys must be normalized")
        # ``xagent:user:`` is the default-owner namespace (any executing user's
        # workspace credential). No actor lives there, so refuse every key
        # under it.
        if owner_key.startswith(RESERVED_OWNER_KEY_PREFIX):
            raise ValueError(
                "actor-private MCP owner keys must not use the reserved "
                f"{RESERVED_OWNER_KEY_PREFIX!r} namespace"
            )
        validated[server_id] = owner_key
    return validated


def resolve_actor_private_mcp_servers_or_raise(
    db: Any, *, user_id: int | None, session_boundary_checked: bool = False
) -> Mapping[int, str]:
    """The executing user's actor-private servers, or the seam's typed 503.

    Returns ``{}`` when no hook is installed or there is no executing user.
    As with the team scope hooks, a failed hook rolls the session back before
    the typed error is raised. A caller holding uncommitted work passes
    ``session_boundary_checked`` so a hook that ends its transaction is refused.
    """
    hook = _hook
    if hook is None or user_id is None:
        return {}
    try:
        return _call_connector_hook_gate(
            db,
            hook,
            db,
            user_id=int(user_id),
            slot="actor_private",
            validate=_validate_answer,
            session_boundary_checked=session_boundary_checked,
        )
    except (ConnectorRuntimeError, ConnectorHookSessionBoundaryError):
        # The boundary error keeps its own 500 handler, as at the team
        # scope's boundary-checked call sites.
        raise
    except Exception as exc:
        logger.warning(
            "Failed to resolve actor-private MCP servers for user %s",
            user_id,
            exc_info=True,
        )
        raise ConnectorRuntimeError(
            ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
            "Actor-private MCP servers are unavailable.",
            details={"reason": "actor_private_resolution_failed"},
            status_code=503,
        ) from exc


def excluded_actor_private_server_ids(
    private_servers: Mapping[int, str], *, resource_owner_key: str | None
) -> frozenset[int]:
    """Mapped servers a run with this owner key must not see.

    Every mapped server when there is no owner key; otherwise the ones another
    owner holds.
    """
    return frozenset(
        server_id
        for server_id, owner_key in private_servers.items()
        if resource_owner_key is None or owner_key != resource_owner_key
    )

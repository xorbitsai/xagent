"""Centralized registry for MCP Applications and OAuth Providers.

This module provides a scalable structure for defining supported MCP applications,
their OAuth configurations, and server launch configurations.
"""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, Dict, List

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
# google-calendar is included for the same reason, not a new one: a bare
# "google" row (app_id-less connect) never requested calendar.events either,
# so it can't be trusted to carry Calendar access on its own merit. Without
# this, Google's incremental consent (include_granted_scopes=true) could
# still let a bare "google" row satisfy Calendar tool calls if it happened to
# accumulate the old, broad calendar scope from a separate authorization --
# exactly the gap the 20260817_narrow_google_calendar_scope.py migration's
# data cleanup deliberately does not attempt to close by deleting that row
# (see that migration's docstring), because a scope-content-only match risks
# collateral damage on Gmail/Drive/Docs credentials sharing the same bare
# row. This closes it architecturally instead: the bare row is simply never
# accepted as a Calendar credential, so its scope content stops mattering.
APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT = frozenset({"facebook", "google-calendar"})


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


def get_all_mcp_apps(db: Session) -> List[Dict[str, Any]]:
    """Retrieve all MCP apps from the database dynamically."""
    apps = db.query(PublicMCPApp).all()
    return [_app_to_dict(app) for app in apps]


def get_app_by_id(db: Session, app_id: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its ID."""
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == app_id).first()
    return _app_to_dict(app) if app else None


def get_app_by_name(db: Session, name: str) -> Dict[str, Any] | None:
    """Retrieve an MCP app configuration by its exact name."""
    app = db.query(PublicMCPApp).filter(PublicMCPApp.name == name).first()
    return _app_to_dict(app) if app else None


def get_app_for_mcp_server(db: Session, server: Any) -> Dict[str, Any] | None:
    """Resolve a server's catalog app by stable identity when it is available.

    Older server rows predate ``auth.app_id`` and are still resolved by their
    exact catalog name. Once a row carries ``app_id``, an invalid value must not
    fall back to a same-named app because that could select another connector's
    credentials or launch configuration.
    """
    auth = getattr(server, "auth", None)
    if isinstance(auth, Mapping) and "app_id" in auth:
        app_id = auth.get("app_id")
        if not isinstance(app_id, str) or not app_id:
            return None
        return get_app_by_id(db, app_id)
    return get_app_by_name(db, str(getattr(server, "name", "")))

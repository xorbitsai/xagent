"""MCP connector operations for key-authenticated SDK callers.

The runtime owns no Session. Every operation runs in a worker that creates and
closes its own Session and returns detached values.

Reads never call the web read routes. They query through
``visible_mcp_server_rows`` and ``resolve_holder_connector`` and project through
``_project_mcp_server``. The projection reuses the web module's ``_mask_env``,
``_check_mcp_permission`` and OAuth account summaries; auth masking is this
module's own ``_masked_auth``.

A create calls the web ``create_mcp_server`` with a request model built from
the SDK body, so the web code decides every stored value, name rule and
encryption. This module never assigns ``url`` or ``auth`` on a stored row.
An update calls the web ``update_mcp_server`` with a request model that holds
exactly the fields the SDK body carried, ``url``, ``headers`` and ``auth``
inside ``config``. A delete calls the web ``delete_mcp_server`` and a switch
calls the web ``update_mcp_server`` with an ``is_active``-only request, after
this module has checked that the holder has a link of their own to change. The
web function's ``HTTPException`` propagates; the route translates it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional, cast

from sqlalchemy import String
from sqlalchemy import cast as sa_cast
from sqlalchemy import literal, or_

from ...core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ...core.tools.adapters.vibe.selection_spec import normalize_mcp_server_name
from ...core.tools.adapters.vibe.signed_upload import signed_upload_settings_error
from ...core.tools.core.mcp.model import MASKED_SECRET_VALUE, SENSITIVE_AUTH_FIELDS
from ..api.mcp import (
    MCPServerCreate,
    MCPServerUpdate,
    _check_mcp_permission,
    _default_resource_owner_key,
    _enrich_oauth_server_info,
    _format_optional_datetime,
    _mask_env,
    _mcp_oauth_configuration_error,
    _member_grant_binding,
    _oauth_account_summaries,
    _require_parsable_mcp_oauth_identifiers,
    create_mcp_server,
    delete_mcp_server,
    update_mcp_server,
)
from ..models.custom_api import CustomApi, UserCustomApi
from ..models.database import get_session_local
from ..models.mcp import MCPServer, UserMCPServer
from ..models.mcp_oauth import MCPOAuthGrant
from ..models.public_mcp import PublicMCPApp
from ..models.user import User
from ..schemas.v1 import (
    V1McpServer,
    V1McpServerAuthorization,
    V1McpServerCreateRequest,
    V1McpServerUpdateRequest,
)
from .connector_team_scope import (
    ConnectorHookSessionBoundaryError,
    connector_access_hook_installed,
    resolve_one_connector_access_or_raise,
    visible_team_connector_ids,
)
from .db_runtime import run_db_io_cancellation_safe
from .mcp_oauth import (
    MCPOAuthRuntimeError,
    _select_runtime_grant,
    select_mcp_oauth_grants,
)
from .mcp_runtime import _is_mcp_oauth_http_server

logger = logging.getLogger(__name__)

# Only connectors stored with one of these transports can report
# ``can_edit=true``.
_REMOTE_TRANSPORTS = frozenset({"streamable_http", "sse"})

# An ``auth`` dict carrying any of these keys was written by a catalog app
# connection (or claims to have been), so the connector is not a user's own.
_CATALOG_AUTH_KEYS = ("app_id", "provider", "builtin_provenance")


_TEAM_READ_ONLY_MESSAGE = (
    "This connector is shared with your team and is read-only through the API."
)
_CATALOG_NOT_EDITABLE_MESSAGE = (
    "This connector is a catalog app connection, or cannot be shown not to be "
    "one, so it can only be changed in the web app."
)


def _log_write(action: str, key_prefix: str, user_id: int, server_id: int) -> None:
    # Nothing but identifiers: no name, no configuration.
    logger.info(
        "personal key mcp server write: action=%s key_prefix=%s user_id=%d "
        "server_id=%d",
        action,
        key_prefix,
        user_id,
        server_id,
    )


def _load_holder(db: Any, user_id: int) -> User:
    user: User | None = db.get(User, user_id)
    if user is None:
        # Imported here: the ``api.v1`` package imports the service layer, so
        # a module-level import would make this module unimportable on its own.
        from ..api.v1.errors import V1ApiError, V1ErrorCode

        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
    return user


def _visible_mcp_listing(
    db: Any, user_id: int
) -> tuple[list[tuple[MCPServer, UserMCPServer | None]], dict[str, set[int]]]:
    """The rows of ``visible_mcp_server_rows`` and the team hook's answer.

    The team list hook is called exactly once. Its answer is returned whole so
    that a caller needing the Custom API half does not call it again.
    """
    own = (
        db.query(MCPServer, UserMCPServer)
        .join(UserMCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(UserMCPServer.user_id == user_id)
        .all()
    )
    own_ids = {int(server.id) for server, _link in own}
    team_shared = visible_team_connector_ids(db, user_id)
    team_ids = team_shared["mcp"] - own_ids
    team = (
        db.query(MCPServer).filter(MCPServer.id.in_(team_ids)).all() if team_ids else []
    )
    rows = [(server, link) for server, link in own] + [
        (server, None) for server in team
    ]
    return rows, team_shared


def visible_mcp_server_rows(
    db: Any, user_id: int
) -> list[tuple[MCPServer, UserMCPServer | None]]:
    """Every MCP connector the holder can see, with the holder's own link.

    The holder's own links come first, whether or not they are enabled: a
    disabled connector must stay listed or it could never be enabled again. A
    connector the holder's team shares and the holder also links is returned
    once, with the link. A connector seen only through the team comes with
    ``None``.

    A failing team hook propagates. Folding it into "the team shares nothing"
    would return a list that silently lacks connectors.
    """
    return _visible_mcp_listing(db, user_id)[0]


@dataclass(frozen=True)
class HolderConnectorNames:
    """The folded names an agent's ``mcp:<name>`` entry can select for a holder.

    A name is folded with ``normalize_mcp_server_name``. An entry selects an
    MCP connector or a Custom API, so both are counted.

    ``visible`` is every name the holder can see. ``own_active`` and
    ``own_inactive`` are the names of the holder's own links, enabled and
    disabled. ``team_keys`` are the names of every connector the team shares,
    including one the holder also links.
    """

    visible: frozenset[str]
    own_active: frozenset[str]
    own_inactive: frozenset[str]
    team_keys: frozenset[str]


def holder_connector_names(db: Any, user_id: int) -> HolderConnectorNames:
    """The folded connector names the holder can see, split by where they come from.

    The team list hook is called once. A failing hook propagates: an empty set
    would make every name look invisible.
    """
    rows, team_shared = _visible_mcp_listing(db, user_id)
    own_active: set[str] = set()
    own_inactive: set[str] = set()
    team_keys: set[str] = set()
    for server, link in rows:
        key = normalize_mcp_server_name(str(server.name))
        if link is None or int(server.id) in team_shared["mcp"]:
            team_keys.add(key)
        if link is not None:
            (own_active if link.is_active else own_inactive).add(key)

    own_apis = (
        db.query(CustomApi.id, CustomApi.name, UserCustomApi.is_active)
        .join(UserCustomApi, UserCustomApi.custom_api_id == CustomApi.id)
        .filter(UserCustomApi.user_id == user_id)
        .all()
    )
    for _api_id, name, is_active in own_apis:
        (own_active if is_active else own_inactive).add(
            normalize_mcp_server_name(str(name))
        )
    team_api_ids = team_shared["custom_api"]
    if team_api_ids:
        for (name,) in db.query(CustomApi.name).filter(CustomApi.id.in_(team_api_ids)):
            team_keys.add(normalize_mcp_server_name(str(name)))

    return HolderConnectorNames(
        visible=frozenset(own_active | own_inactive | team_keys),
        own_active=frozenset(own_active),
        own_inactive=frozenset(own_inactive),
        team_keys=frozenset(team_keys),
    )


def resolve_holder_connector(
    db: Any, user_id: int, server_id: int
) -> tuple[MCPServer, UserMCPServer | None] | None:
    """The connector with ``server_id`` as the holder sees it, or ``None``.

    ``None`` means the holder cannot see it or it does not exist; callers
    answer 404 for both. The link is ``None`` for a connector seen only
    through the holder's team.

    When the application installed a direct-id access hook, that hook alone
    decides whether the team shares the connector, as it does for the web
    detail route. Without one the team half of the list hook decides, which
    is the same set the list route returns.
    """
    # Imported here for the reason given in ``_load_holder``.
    from ..api.v1.errors import V1ApiError, V1ErrorCode

    row = (
        db.query(MCPServer, UserMCPServer)
        .join(UserMCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
        .one_or_none()
    )
    if row is not None:
        return row[0], row[1]
    server = db.get(MCPServer, server_id)
    if server is None:
        return None
    if connector_access_hook_installed():
        try:
            verdict = resolve_one_connector_access_or_raise(
                db, user_id, ("mcp", server_id)
            )
        except ConnectorHookSessionBoundaryError as exc:
            # The application maps this exception to a 500 body of its own
            # that lacks the SDK error envelope, so it is converted here.
            logger.error(
                "personal key connector access hook ended the transaction: "
                "user_id=%d server_id=%d",
                user_id,
                server_id,
            )
            raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500) from exc
        except ConnectorRuntimeError as exc:
            raise V1ApiError(
                V1ErrorCode.CONNECTOR_RUNTIME_UNAVAILABLE, 503, exc.safe_message
            ) from exc
        return None if verdict is None else (server, None)
    if server_id in visible_team_connector_ids(db, user_id)["mcp"]:
        return server, None
    return None


def _resolve_changeable(
    db: Any, user_id: int, server_id: int
) -> tuple[MCPServer, UserMCPServer]:
    """The connector and the holder's own link, for a change the holder may try.

    A connector the holder cannot see is a 404. One seen only through the
    team has no link of the holder's own to change, so it is refused with 403
    before any web function runs.
    """
    # Imported here for the reason given in ``_load_holder``.
    from ..api.v1.errors import V1ApiError, V1ErrorCode

    row = resolve_holder_connector(db, user_id, server_id)
    if row is None:
        raise V1ApiError(V1ErrorCode.CONNECTOR_NOT_FOUND, 404)
    server, link = row
    if link is None:
        raise V1ApiError(
            V1ErrorCode.CONNECTOR_FORBIDDEN,
            403,
            _TEAM_READ_ONLY_MESSAGE,
            details={"reason": "team_shared"},
        )
    return server, link


def _catalog_classification_keys(
    db: Any, servers: Iterable[MCPServer]
) -> frozenset[str]:
    """Every catalog app id and app name, hidden apps included, read once.

    Empty without a query when no connector reaches the name test. A failing
    read propagates: treating it as an empty catalog would classify catalog
    connectors as custom, and so editable.
    """
    # Without catalog keys only a connector that reaches the name test is
    # custom, so the classification itself decides whether the read is needed.
    if not any(classify_connector_kind(s, frozenset()) == "custom" for s in servers):
        return frozenset()
    keys: set[str] = set()
    for app_id, name in db.query(PublicMCPApp.app_id, PublicMCPApp.name).all():
        keys.add(app_id)
        keys.add(name)
    return frozenset(keys)


def classify_connector_kind(server: MCPServer, catalog_app_keys: frozenset[str]) -> str:
    """``"catalog"`` unless the connector can be shown to be a user's own.

    A connector is ``catalog`` when its transport is ``oauth``, when its
    ``auth`` carries a catalog marker (even one pointing at an app that no
    longer exists), or when any catalog app, hidden ones included, has an app
    id or name equal to the connector's name. That last test fires on a single
    match: the web code treats two matches as "cannot tell" and gives up,
    which here would let a connector that may belong to the platform be
    changed.

    ``catalog_app_keys`` is the set of app ids and names from
    ``_catalog_classification_keys``; the list and the detail share this
    function, which does no I/O.
    """
    if server.transport == "oauth":
        return "catalog"
    auth = server.auth
    if isinstance(auth, dict) and any(key in auth for key in _CATALOG_AUTH_KEYS):
        return "catalog"
    return "catalog" if str(server.name) in catalog_app_keys else "custom"


def connector_kind(db: Any, server: MCPServer) -> str:
    # Single-connector entry; it runs the same pure function as the projection.
    return classify_connector_kind(server, _catalog_classification_keys(db, [server]))


def _web_create_request(request: V1McpServerCreateRequest) -> MCPServerCreate:
    # ``url``, ``headers`` and ``auth`` travel inside ``config``, the web
    # function's own shape.
    config: dict[str, Any] = {"url": request.url}
    if request.headers:
        config["headers"] = dict(request.headers)
    if request.auth:
        config["auth"] = dict(request.auth)
    return MCPServerCreate(
        name=request.name,
        transport=request.transport,
        description=request.description,
        config=config,
        is_active=request.enabled,
        runtime_input_schema=request.runtime_input_schema,
        runtime_bindings=request.runtime_bindings,
        signed_uploads=request.signed_uploads,
        allow_delegated_authorization=request.allow_delegated_authorization,
    )


def _web_update_request(request: V1McpServerUpdateRequest) -> MCPServerUpdate:
    # The web function decides what to write from which fields are present, so
    # only the fields the body carried are passed, null ones included.
    data = request.model_dump(exclude_unset=True)
    config = {key: data.pop(key) for key in ("url", "headers", "auth") if key in data}
    if config:
        data["config"] = config
    return MCPServerUpdate(**data)


_REBIND_REFUSED_MESSAGE = (
    "Other accounts have authorized this connector with their own sign-in; "
    "change its URL, issuer or resource metadata URL in the web app."
)


def _grants_surviving_rebind(db: Any, *, server_id: int, user_id: int) -> Any:
    """Ids of the active grants on ``server_id`` that update_mcp_server keeps
    on a binding change, other than the holder's own default-identity grant.

    The filter is the complement of the deletion in
    _delete_team_path_mcp_oauth_state within active grants; a test runs both
    on one connector and checks that every such grant is deleted or counted.
    """
    holds_association = (
        db.query(UserMCPServer.id)
        .filter(
            UserMCPServer.user_id == MCPOAuthGrant.user_id,
            UserMCPServer.mcpserver_id == server_id,
        )
        .exists()
    )
    own_default = _default_resource_owner_key(user_id)
    return db.query(MCPOAuthGrant.id).filter(
        MCPOAuthGrant.mcp_server_id == server_id,
        MCPOAuthGrant.status == "active",
        or_(
            MCPOAuthGrant.user_id != user_id,
            MCPOAuthGrant.resource_owner_key != own_default,
        ),
        or_(
            holds_association,
            MCPOAuthGrant.resource_owner_key
            != literal("xagent:user:") + sa_cast(MCPOAuthGrant.user_id, String),
        ),
    )


def _has_active_grant_surviving_rebind(
    db: Any, *, server_id: int, user_id: int
) -> bool:
    """Whether an active grant that update_mcp_server keeps on a binding change
    exists, other than the holder's own default-identity grant.

    update_mcp_server deletes the grants of users without an association row
    held under their own default owner key (_delete_team_path_mcp_oauth_state).
    This matches the rest: grants of users who hold an association row (active
    or not), and grants under any other owner key. It does not look at
    expiry or at the current auth configuration: the runtime does not filter
    grants by expiry, and a configuration switched away and back selects the
    old grants again.

    The web app makes the same change and keeps these grants. This route
    refuses it instead, because such a change does not always leave a kept
    grant unusable: grant selection compares a linked user's grant with
    ``auth.resource`` and ``auth.issuer`` only when they are set.
    """
    return (
        _grants_surviving_rebind(db, server_id=server_id, user_id=user_id).first()
        is not None
    )


def _require_rebind_editor(db: Any, *, server_id: int, user_id: int) -> None:
    """Refuse a binding change from a holder who may not edit the shared
    configuration.

    Runs with the definition row locked and before any grant is read. The
    holder's link and admin flag are read again, as update_mcp_server reads
    them after its own lock, so a holder who may not edit gets the same
    answer whoever else has authorized the connector.
    """
    # Imported here for the reason given in ``_load_holder``.
    from ..api.v1.errors import V1ApiError, V1ErrorCode

    link = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user_id,
            UserMCPServer.mcpserver_id == server_id,
        )
        .populate_existing()
        .first()
    )
    if link is None:
        raise V1ApiError(V1ErrorCode.CONNECTOR_NOT_FOUND, 404)
    holder = db.query(User).filter(User.id == user_id).populate_existing().first()
    if holder is None:
        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
    if not _check_mcp_permission(link, bool(holder.is_admin), require="edit"):
        raise V1ApiError(
            V1ErrorCode.CONNECTOR_FORBIDDEN, 403, details={"reason": "not_permitted"}
        )


def _refuse_unsafe_update_under_lock(
    db: Any, request: V1McpServerUpdateRequest, *, server_id: int, user_id: int
) -> None:
    """Refuse an update the web function would apply wrongly or unsafely.

    Takes the definition row lock, as the web function does later in the same
    transaction, and judges against the row read under it. Reads only; nothing
    is written before the web function runs.
    """
    # Imported here for the reason given in ``_load_holder``.
    from ..api.v1.errors import V1ApiError, V1ErrorCode

    locked = (
        db.query(MCPServer)
        .filter(MCPServer.id == server_id)
        .populate_existing()
        .with_for_update(key_share=True)
        .first()
    )
    if locked is None:
        raise V1ApiError(V1ErrorCode.CONNECTOR_NOT_FOUND, 404)
    sent = request.model_fields_set
    before = _member_grant_binding(locked)
    # What the web function will store, read by the web function's own rule
    # for what a grant is bound to. The stored row is never assigned to.
    after = _member_grant_binding(
        cast(
            MCPServer,
            SimpleNamespace(
                url=request.url if "url" in sent else locked.url,
                auth=request.auth if "auth" in sent else locked.auth,
            ),
        )
    )
    if before != after:
        _require_rebind_editor(db, server_id=server_id, user_id=user_id)
        if _has_active_grant_surviving_rebind(db, server_id=server_id, user_id=user_id):
            raise V1ApiError(
                V1ErrorCode.CONNECTOR_NOT_EDITABLE,
                409,
                _REBIND_REFUSED_MESSAGE,
                details={"reason": "authorized_by_others"},
            )
    # The web function stores a masked secret that has nothing to keep as
    # empty, without an error.
    stored_auth = locked.auth if isinstance(locked.auth, dict) else {}
    for key, value in (request.auth or {}).items():
        if (
            key in SENSITIVE_AUTH_FIELDS
            and value == MASKED_SECRET_VALUE
            and not stored_auth.get(key)
        ):
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                400,
                f"Masked secret '{key}' has no stored value; provide a new value",
            )


def _masked_dict(value: Any) -> Optional[dict[str, Any]]:
    return _mask_env(value) if isinstance(value, dict) else None


def _masked_auth(auth: Any) -> Optional[dict[str, Any]]:
    if not isinstance(auth, dict):
        return None
    masked = auth.copy()
    for key in SENSITIVE_AUTH_FIELDS:
        if key in masked and masked[key]:
            masked[key] = MASKED_SECRET_VALUE
    return masked


_AUTHORIZATION_MESSAGE = (
    "Authorization can only be completed in the xagent web app, not with an "
    "API key. Open Connectors and select this connector; if it offers Connect, "
    "choose it, otherwise ask the connector's owner."
)
_CONFIGURATION_ERROR_MESSAGE = (
    "This connector's OAuth settings are unusable. Its owner must correct them "
    "in the xagent web app."
)
# A status absent here reads _AUTHORIZATION_MESSAGE.
_STATUS_MESSAGES: dict[str, Optional[str]] = {
    "authorized": None,
    "configuration_error": _CONFIGURATION_ERROR_MESSAGE,
}

_BUILTIN_OAUTH_STATUS = {
    "connected": "authorized",
    "needs_reconnect": "needs_reconnect",
}


def _authorization_value(kind: str, status: str) -> V1McpServerAuthorization:
    return V1McpServerAuthorization(
        kind=kind,
        status=status,
        message=_STATUS_MESSAGES.get(status, _AUTHORIZATION_MESSAGE),
    )


def _authorization(
    db: Any,
    server: MCPServer,
    *,
    holder_id: int,
    allow_team_member_grants: bool,
    app_id: Optional[str],
    connection_status: Optional[str],
    active_grant_server_ids: frozenset[int],
) -> Optional[V1McpServerAuthorization]:
    """The holder's own OAuth authorization of the connector, if it uses OAuth.

    A failing grant read propagates: it must not read as "not authorized". An OAuth
    resource or issuer that does not parse is reported as "configuration_error", so
    one misconfigured connector does not fail the whole response. A token supplied
    by an OAuth token resolver hook is not reflected.
    """
    if server.transport == "oauth":
        if app_id is None:
            # The runtime reports such a connector unavailable
            # (catalog_app_not_found); connecting cannot fix it.
            return _authorization_value("builtin_oauth", "configuration_error")
        return _authorization_value(
            "builtin_oauth",
            _BUILTIN_OAUTH_STATUS.get(connection_status or "", "not_authorized"),
        )
    if not _is_mcp_oauth_http_server(server, server.auth):
        return None
    # The check above guarantees a dict; the ORM column type does not say so.
    auth = cast(dict[str, Any], server.auth)
    if _mcp_oauth_configuration_error(auth) is not None:
        return _authorization_value("mcp_oauth", "configuration_error")
    try:
        # The selector raises ValueError for exactly these definitions.
        _require_parsable_mcp_oauth_identifiers(auth)
    except ValueError as exc:
        # Such settings are unusable, so report them for this connector
        # instead of failing the whole response.
        logger.warning(
            "Reporting MCP OAuth connector authorization as "
            "configuration_error, its OAuth resource or issuer does not "
            "parse: server_id=%s error=%s",
            server.id,
            type(exc).__name__,
        )
        return _authorization_value("mcp_oauth", "configuration_error")
    # The selector only returns grants whose server id is in this set.
    if int(server.id) not in active_grant_server_ids:
        return _authorization_value("mcp_oauth", "not_authorized")
    grants = select_mcp_oauth_grants(
        db,
        server_id=int(server.id),
        user_id=holder_id,
        auth_config=auth,
        resource_owner_key=_default_resource_owner_key(holder_id),
        allow_team_member_grants=allow_team_member_grants,
        endpoint_url=cast(Optional[str], server.url),
    )
    if not grants:
        return _authorization_value("mcp_oauth", "not_authorized")
    try:
        # The runtime's own pick; the selector has already applied the scope.
        _select_runtime_grant(grants, set())
    except MCPOAuthRuntimeError:
        return _authorization_value("mcp_oauth", "needs_reconnect")
    return _authorization_value("mcp_oauth", "authorized")


def _project_mcp_server(
    db: Any,
    server: MCPServer,
    link: UserMCPServer | None,
    *,
    holder_id: int,
    is_admin: bool,
    oauth_accounts: dict[str, tuple[Optional[str], str, int]],
    catalog_app_keys: frozenset[str],
    active_grant_server_ids: frozenset[int],
) -> V1McpServer:
    """The SDK view of one connector.

    Reads the stored values only. Calling ``server.to_config_dict`` would
    decrypt secrets and bring them into this process, and an undecryptable
    value could then fail a whole listing. Launch details (command, arguments,
    working directory) are left out: the SDK cannot change them and they can
    carry tokens.
    """
    kind = classify_connector_kind(server, catalog_app_keys)
    can_edit = (
        link is not None
        and kind == "custom"
        and server.transport in _REMOTE_TRANSPORTS
        and _check_mcp_permission(link, is_admin, require="edit")
    )
    can_delete = (
        link is not None
        and kind == "custom"
        and _check_mcp_permission(link, is_admin, require="delete")
    )
    app_id, provider, connected_account, connection_status = _enrich_oauth_server_info(
        db, server, oauth_accounts
    )
    return V1McpServer(
        id=int(server.id),
        name=server.name,
        description=server.description,
        transport=server.transport,
        url=server.url,
        headers=_masked_dict(server.headers),
        auth=_masked_auth(server.auth),
        env=_masked_dict(server.env),
        user_env=_masked_dict(link.env) if link is not None and link.env else None,
        runtime_input_schema=server.runtime_input_schema,
        runtime_bindings=server.runtime_bindings,
        signed_uploads=server.signed_uploads,
        signed_uploads_error=signed_upload_settings_error(
            server.signed_uploads, transport=server.transport, url=server.url
        ),
        allow_delegated_authorization=bool(server.allow_delegated_authorization),
        access="team" if link is None else "own",
        kind=kind,
        enabled=bool(link.is_active) if link is not None else None,
        can_edit=can_edit,
        can_delete=can_delete,
        app_id=app_id,
        provider=provider,
        connected_account=connected_account,
        authorization=_authorization(
            db,
            server,
            holder_id=holder_id,
            # A connector seen only through the team has no link: the holder's
            # own authorization of it counts when it targets the current URL.
            allow_team_member_grants=link is None,
            app_id=app_id,
            connection_status=connection_status,
            active_grant_server_ids=active_grant_server_ids,
        ),
        created_at=_format_optional_datetime(server.created_at),
        updated_at=_format_optional_datetime(server.updated_at),
    )


def _project_rows(
    db: Any, user: User, rows: list[tuple[MCPServer, UserMCPServer | None]]
) -> list[V1McpServer]:
    # Only built-in OAuth connectors read the holder's account summaries.
    oauth_accounts = (
        _oauth_account_summaries(db, int(user.id))
        if any(server.transport == "oauth" for server, _link in rows)
        else {}
    )
    # At most one catalog read serves every row; a failure here precedes any projection.
    catalog_app_keys = _catalog_classification_keys(
        db, [server for server, _link in rows]
    )
    # The grant selector only returns grants in this set, so one read lets rows
    # without an active grant skip it, as the web connector listing does.
    active_grant_server_ids = (
        frozenset(
            int(server_id)
            for (server_id,) in db.query(MCPOAuthGrant.mcp_server_id)
            .filter(
                MCPOAuthGrant.user_id == user.id,
                MCPOAuthGrant.status == "active",
            )
            .distinct()
            .all()
        )
        if any(_is_mcp_oauth_http_server(server, server.auth) for server, _link in rows)
        else frozenset()
    )
    return [
        _project_mcp_server(
            db,
            server,
            link,
            holder_id=int(user.id),
            is_admin=bool(user.is_admin),
            oauth_accounts=oauth_accounts,
            catalog_app_keys=catalog_app_keys,
            active_grant_server_ids=active_grant_server_ids,
        )
        for server, link in rows
    ]


class McpServerManagementRuntime:
    """Async owner for SDK MCP connector operations."""

    async def list_servers(self, *, user_id: int) -> list[V1McpServer]:
        """The connectors the holder can see: own ones first, then the team's."""

        def run() -> list[V1McpServer]:
            with get_session_local()() as db:
                user = _load_holder(db, user_id)
                projected = _project_rows(
                    db, user, visible_mcp_server_rows(db, user_id)
                )
            return sorted(
                projected,
                key=lambda s: (s.access != "own", s.name.casefold(), s.id),
            )

        return await run_db_io_cancellation_safe(run)

    async def get_server(self, *, user_id: int, server_id: int) -> V1McpServer | None:
        """One connector the holder can see, or ``None``."""

        def run() -> V1McpServer | None:
            with get_session_local()() as db:
                user = _load_holder(db, user_id)
                row = resolve_holder_connector(db, user_id, server_id)
                if row is None:
                    return None
                return _project_rows(db, user, [row])[0]

        return await run_db_io_cancellation_safe(run)

    async def create_server(
        self, *, user_id: int, key_prefix: str, request: V1McpServerCreateRequest
    ) -> V1McpServer:
        """Create a connector the holder owns and return it as the holder sees it.

        The web function commits before the projection runs, so a projection
        failure answers 500 with the connector already created.
        """

        def run() -> V1McpServer:
            with get_session_local()() as db:
                user = _load_holder(db, user_id)
                created = create_mcp_server(
                    _web_create_request(request), current_user=user, db=db
                )
                row = resolve_holder_connector(db, user_id, int(created.id))
                if row is None:
                    # The web function just made this link in its own commit.
                    from ..api.v1.errors import V1ApiError, V1ErrorCode

                    logger.error(
                        "personal key mcp server write failed: action=create "
                        "user_id=%d server_id=%d",
                        user_id,
                        int(created.id),
                    )
                    raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500)
                projected = _project_rows(db, user, [row])[0]
            _log_write("create", key_prefix, user_id, projected.id)
            return projected

        return await run_db_io_cancellation_safe(run)

    async def delete_server(
        self, *, user_id: int, key_prefix: str, server_id: int
    ) -> None:
        """Delete the holder's link to a connector through the web delete.

        The web function is a coroutine because it revokes OAuth grants over
        the network after committing. It runs on a new event loop in this
        worker thread, so none of the web function's database work blocks the
        loop that serves requests. Catalog connectors are refused before it
        runs.
        """
        # Imported here for the reason given in ``_load_holder``.
        from ..api.v1.errors import V1ApiError, V1ErrorCode

        def run() -> None:
            with get_session_local()() as db:
                user = _load_holder(db, user_id)
                server, _link = _resolve_changeable(db, user_id, server_id)
                if connector_kind(db, server) == "catalog":
                    raise V1ApiError(
                        V1ErrorCode.CONNECTOR_NOT_EDITABLE,
                        409,
                        _CATALOG_NOT_EDITABLE_MESSAGE,
                        details={"reason": "catalog"},
                    )
                asyncio.run(delete_mcp_server(server_id, current_user=user, db=db))
            _log_write("delete", key_prefix, user_id, server_id)

        await run_db_io_cancellation_safe(run)

    async def set_enabled(
        self, *, user_id: int, key_prefix: str, server_id: int, enabled: bool
    ) -> V1McpServer:
        """Switch the holder's own link on or off and return the connector.

        Only ``is_active`` is sent to the web function, so the shared
        definition is neither locked nor written.
        """
        # Imported here for the reason given in ``_load_holder``.
        from ..api.v1.errors import V1ApiError, V1ErrorCode

        def run() -> V1McpServer:
            with get_session_local()() as db:
                user = _load_holder(db, user_id)
                _resolve_changeable(db, user_id, server_id)
                update_mcp_server(
                    server_id,
                    MCPServerUpdate(is_active=enabled),
                    current_user=user,
                    db=db,
                )
                row = resolve_holder_connector(db, user_id, server_id)
                if row is None or row[1] is None:
                    # The web function has committed the update, so the
                    # holder's link is gone only if it was deleted in between,
                    # even when the team still shares the connector.
                    raise V1ApiError(V1ErrorCode.CONNECTOR_NOT_FOUND, 404)
                projected = _project_rows(db, user, [row])[0]
            _log_write(
                "enable" if enabled else "disable", key_prefix, user_id, server_id
            )
            return projected

        return await run_db_io_cancellation_safe(run)

    async def update_server(
        self,
        *,
        user_id: int,
        key_prefix: str,
        server_id: int,
        request: V1McpServerUpdateRequest,
        validate: Callable[[V1McpServerUpdateRequest], None],
    ) -> V1McpServer:
        """Change the fields the body carries and return the connector.

        ``validate`` holds the body rules that need no database; it runs after
        the connector is found and known to be one the holder may try to
        change, so a connector the holder may not change answers before a bad
        body does. A holder who may not edit the shared configuration answers
        403 for any non-empty body, before the web function runs and without a
        write line. A body with no field changes nothing. The web function
        commits before the projection runs, so a projection failure answers 500
        with the change stored.
        """
        # Imported here for the reason given in ``_load_holder``.
        from ..api.v1.errors import V1ApiError, V1ErrorCode

        def run() -> V1McpServer:
            with get_session_local()() as db:
                user = _load_holder(db, user_id)
                server, link = _resolve_changeable(db, user_id, server_id)
                if connector_kind(db, server) == "catalog":
                    raise V1ApiError(
                        V1ErrorCode.CONNECTOR_NOT_EDITABLE,
                        409,
                        _CATALOG_NOT_EDITABLE_MESSAGE,
                        details={"reason": "catalog"},
                    )
                if server.transport not in _REMOTE_TRANSPORTS:
                    raise V1ApiError(
                        V1ErrorCode.TRANSPORT_NOT_SUPPORTED,
                        400,
                        f"This connector uses the '{server.transport}' transport; "
                        "change it in the web app.",
                    )
                sent = request.model_fields_set
                # Every field this body can carry is shared configuration; the
                # web function refuses or silently drops it for a holder who may
                # not edit, so such a holder can change nothing here.
                if sent and not _check_mcp_permission(
                    link, bool(user.is_admin), require="edit"
                ):
                    raise V1ApiError(
                        V1ErrorCode.CONNECTOR_FORBIDDEN,
                        403,
                        details={"reason": "not_permitted"},
                    )
                validate(request)
                shown: tuple[MCPServer, UserMCPServer | None] = (server, link)
                if sent:
                    if sent & {"url", "auth"}:
                        _refuse_unsafe_update_under_lock(
                            db, request, server_id=server_id, user_id=user_id
                        )
                    update_mcp_server(
                        server_id,
                        _web_update_request(request),
                        current_user=user,
                        db=db,
                    )
                    resolved = resolve_holder_connector(db, user_id, server_id)
                    if resolved is None or resolved[1] is None:
                        # The web function has committed the update, so the
                        # holder's link is gone only if it was deleted in between,
                        # even when the team still shares the connector.
                        raise V1ApiError(V1ErrorCode.CONNECTOR_NOT_FOUND, 404)
                    shown = resolved
                projected = _project_rows(db, user, [shown])[0]
            if sent:
                _log_write("update", key_prefix, user_id, server_id)
            return projected

        return await run_db_io_cancellation_safe(run)

    async def require_listable(self, *, user_id: int, server_id: int) -> None:
        """Raise unless the holder may have this connector's tools listed.

        Read-only. The connector session starts only a connector the holder
        links to and has switched on, and reports any other one as not found;
        it does not look at the transport. Checking here keeps a local
        connector from being started and tells a team-shared or disabled
        connector apart from a missing one.
        """
        # Imported here for the reason given in ``_load_holder``.
        from ..api.v1.errors import V1ApiError, V1ErrorCode

        def run() -> None:
            with get_session_local()() as db:
                _load_holder(db, user_id)
                row = resolve_holder_connector(db, user_id, server_id)
                if row is None:
                    raise V1ApiError(V1ErrorCode.CONNECTOR_NOT_FOUND, 404)
                server, link = row
                if server.transport not in _REMOTE_TRANSPORTS:
                    raise V1ApiError(
                        V1ErrorCode.TRANSPORT_NOT_SUPPORTED,
                        400,
                        "Listing tools through the API supports only "
                        "streamable_http and sse connectors.",
                    )
                if link is None:
                    raise V1ApiError(
                        V1ErrorCode.CONNECTOR_UNAVAILABLE,
                        409,
                        details={"reason": "team_shared"},
                    )
                if not link.is_active:
                    raise V1ApiError(
                        V1ErrorCode.CONNECTOR_UNAVAILABLE,
                        409,
                        details={"reason": "disabled"},
                    )

        await run_db_io_cancellation_safe(run)

    async def owned_connector_id(self, *, user_id: int, name: str) -> Optional[int]:
        """The id of the connector named exactly ``name`` that the holder owns."""

        def run() -> Optional[int]:
            with get_session_local()() as db:
                found = (
                    db.query(MCPServer.id)
                    .join(UserMCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
                    .filter(
                        MCPServer.name == name,
                        UserMCPServer.user_id == user_id,
                        UserMCPServer.is_owner.is_(True),
                    )
                    .first()
                )
                return None if found is None else int(found[0])

        return await run_db_io_cancellation_safe(run)

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
The web function's ``HTTPException`` propagates; the route translates it.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, cast

from ...core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from ...core.tools.adapters.vibe.signed_upload import signed_upload_settings_error
from ...core.tools.core.mcp.model import MASKED_SECRET_VALUE, SENSITIVE_AUTH_FIELDS
from ..api.mcp import (
    MCPServerCreate,
    _check_mcp_permission,
    _default_resource_owner_key,
    _enrich_oauth_server_info,
    _format_optional_datetime,
    _mask_env,
    _mcp_oauth_configuration_error,
    _oauth_account_summaries,
    _require_parsable_mcp_oauth_identifiers,
    create_mcp_server,
)
from ..models.database import get_session_local
from ..models.mcp import MCPServer, UserMCPServer
from ..models.mcp_oauth import MCPOAuthGrant
from ..models.public_mcp import PublicMCPApp
from ..models.user import User
from ..schemas.v1 import (
    V1McpServer,
    V1McpServerAuthorization,
    V1McpServerCreateRequest,
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
    own = (
        db.query(MCPServer, UserMCPServer)
        .join(UserMCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(UserMCPServer.user_id == user_id)
        .all()
    )
    own_ids = {int(server.id) for server, _link in own}
    team_ids = visible_team_connector_ids(db, user_id)["mcp"] - own_ids
    team = (
        db.query(MCPServer).filter(MCPServer.id.in_(team_ids)).all() if team_ids else []
    )
    return [(server, link) for server, link in own] + [
        (server, None) for server in team
    ]


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

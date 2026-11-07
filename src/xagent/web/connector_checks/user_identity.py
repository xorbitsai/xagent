"""The ``user`` identity: one local user's connected catalog connectors.

It exists so the engine and its CLI (``cli.py``) can be verified locally and
on stage. The V2 workflow does not schedule it: the web identity's scheduled
coverage stays V1's HTTP L2/L3.

The user is named by username, or by email address compared
case-insensitively, and must be named in ``XAGENT_CONNECTOR_PROBE_USERS``
(``connector_probe.is_connector_probe_user``, the probe route's own rule).
``enumerate_targets`` refuses any other user, and a name no account has,
before a connector starts.

The targets are the catalog apps the Connectors page shows the user as
connected (``GET /api/mcp/apps``, the listing V1's L2/L3 smoke reads), each
with the server row that page selects for it. As in V1, an app hidden from
the catalog (``is_visible_in_connector`` false) is not a target even while
connected. A selected row that is not the catalog's own
(``connector_probe.owns_catalog_identity``: a user-owned row under a catalog
id, or a catalog row whose launch has drifted) is offered with the skip
reason ``not_catalog_identity`` and never started, as the probe route
answers ``not_declared`` for it. A custom server, stamped with a catalog app
id or not, is not a catalog app the page shows as connected, so it is not a
target.

Each connector starts as the user's own task would start it: no actor policy
and no team (``mcp_connector_session.list_connector_tools``). No fixture has
a value yet: the server-side source comes with #3284, so a check that needs
one is ``skip``/``fixture_missing``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func

from ..services.connector_probe import is_connector_probe_user, owns_catalog_identity
from ..services.mcp_connector_session import (
    ConnectorToolsResult,
    find_visible_server,
    list_connector_tools,
)
from .fixtures import FixtureValue
from .identity import ConnectorTarget, IdentitySetupError, SessionPurpose

USER_IDENTITY_LABEL = "user"
NOT_CATALOG_IDENTITY = "not_catalog_identity"

# The agent-facing tool name prefix; the engine finds a tool by its wire name.
_NAME_PREFIX = "smoke_"


class UserIdentityAdapter:
    """Runs checks as the local user named ``user_name``, through ``db``."""

    def __init__(self, db: Any, user_name: str) -> None:
        self._db = db
        self._user_name = user_name
        self._user: Any | None = None

    @property
    def label(self) -> str:
        return USER_IDENTITY_LABEL

    def enumerate_targets(self) -> list[ConnectorTarget]:
        user = _find_user(self._db, self._user_name)
        if user is None:
            raise IdentitySetupError("identity", "user_not_found")
        if not is_connector_probe_user(user):
            raise IdentitySetupError("identity", "user_not_allowed")
        connected = _connected_catalog_apps(self._db, user)
        targets = [
            self._target(user, app_id, server_id)
            for app_id, server_id in sorted(connected.items())
        ]
        self._user = user
        return targets

    async def open_session(
        self, target: ConnectorTarget, *, purpose: SessionPurpose
    ) -> ConnectorToolsResult:
        if self._user is None:
            raise RuntimeError("enumerate_targets must run before a session opens")
        return await list_connector_tools(
            self._db,
            user=self._user,
            server_id=target.ref,
            name_prefix=_NAME_PREFIX,
            refuse_sandboxed=purpose is SessionPurpose.CALL_TOOLS,
        )

    def resolve_fixture(self, key: str) -> FixtureValue | None:
        return None

    def _target(self, user: Any, app_id: str, server_id: int) -> ConnectorTarget:
        server = find_visible_server(
            self._db, user_id=int(user.id), server_id=server_id, connector_team_id=None
        )
        # A row a task would not see is offered as it is: its listing is
        # not_found, and nothing starts.
        if server is not None and not _is_catalog_row(self._db, server, app_id):
            return ConnectorTarget(app_id, skip_reason=NOT_CATALOG_IDENTITY)
        return ConnectorTarget(app_id, ref=server_id)


def _find_user(db: Any, name: str) -> Any | None:
    from ..models.user import User

    name = name.strip()
    if not name:
        return None
    user = db.query(User).filter(User.username == name).first()
    if user is None:
        user = db.query(User).filter(func.lower(User.email) == name.lower()).first()
    return user


def _connected_catalog_apps(db: Any, user: Any) -> dict[str, int]:
    """Each catalog app the Connectors page shows ``user`` as connected, with
    the server row it selects (V1's L2 ``select_targets`` reads the same)."""
    from ..api.mcp import list_mcp_apps

    apps = list_mcp_apps(location="remote", current_user=user, db=db)
    return {
        app["id"]: app["server_id"]
        for app in apps
        if isinstance(app.get("id"), str)
        and app["id"]
        and app.get("is_connected") is True
        and type(app.get("server_id")) is int
    }


def _is_catalog_row(db: Any, server: Any, app_id: str) -> bool:
    """Whether ``server`` is catalog app ``app_id``'s own row, by the probe
    route's rule."""
    from ..mcp_apps import get_app_for_mcp_server

    app = get_app_for_mcp_server(db, server)
    return (
        app is not None
        and app.get("id") == app_id
        and owns_catalog_identity(db, server, app)
    )

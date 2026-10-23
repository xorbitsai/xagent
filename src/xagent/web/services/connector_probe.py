"""Run one connected connector's declared read-only probe (L3).

The probe route (api/mcp.py) and the connector smoke check call this for a
server the caller has connected. The connector is started exactly as a task
would start it (``mcp_connector_session.list_connector_tools``), the one
tool declared for its catalog app in ``web/connector_probes.py`` is called
with the declared arguments, and the result is judged by the tool's own
envelope. The caller never chooses the tool or its arguments, so this cannot
serve as a general tool-execution path.

Two rules decide whether a probe may run at all. The row must be the
catalog's own: an OAuth-transport row is written by the platform and launches
from the catalog, so it resolves through ``get_app_for_mcp_server``; any other
row owns a catalog identity only as catalog connect adopts it (exact
``name == app_id``, no user owning it, and the catalog's own transport and
command or URL), because a custom row's ``auth`` is caller-authored. And the
row must not run in a sandbox: the listing's lease is released before the
call, so the call would open a new session on the web host. That is judged on
the effective connection the listing builds, not on the catalog's launch
config: an ``npx``/``uvx`` connection is refused whether or not a sandbox
manager is configured, after the task's config build (database lookups, env
decryption and, for an OAuth row, a possible token refresh) but before any
sandbox lease, process spawn or loader call.

Nothing here returns the tool's text: the in-repo servers put the vendor's
raw error in it, which may carry a URL or a credential fragment. A result
carries codes only, like the tools route.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ...config import get_connector_probe_users, get_mcp_tool_init_timeout_seconds
from ..connector_probes import (
    ConnectorProbe,
    get_connector_probe,
    get_unprobeable_reason,
)
from .mcp_connector_session import ConnectorToolsOutcome, list_connector_tools

logger = logging.getLogger(__name__)


class ConnectorProbeOutcome(str, Enum):
    OK = "ok"
    NOT_FOUND = "not_found"
    NOT_DECLARED = "not_declared"
    UNPROBEABLE = "unprobeable"
    # The listing outcomes, passed through from the connector session.
    NOT_CONNECTED = "not_connected"
    UNSUPPORTED = "unsupported"
    FAILED_TO_START = "failed_to_start"
    NO_TOOLS = "no_tools"
    TIMED_OUT = "timed_out"
    # The call itself.
    PROBE_TOOL_MISSING = "probe_tool_missing"
    PROBE_FAILED = "probe_failed"
    PROBE_ASSERTION_FAILED = "probe_assertion_failed"


_LISTING_OUTCOMES: Mapping[ConnectorToolsOutcome, ConnectorProbeOutcome] = {
    ConnectorToolsOutcome.NOT_FOUND: ConnectorProbeOutcome.NOT_FOUND,
    ConnectorToolsOutcome.NOT_CONNECTED: ConnectorProbeOutcome.NOT_CONNECTED,
    ConnectorToolsOutcome.UNSUPPORTED: ConnectorProbeOutcome.UNSUPPORTED,
    ConnectorToolsOutcome.FAILED_TO_START: ConnectorProbeOutcome.FAILED_TO_START,
    ConnectorToolsOutcome.NO_TOOLS: ConnectorProbeOutcome.NO_TOOLS,
    ConnectorToolsOutcome.TIMED_OUT: ConnectorProbeOutcome.TIMED_OUT,
}

# HTTP status and ``code`` the route answers with, per outcome. The listing
# outcomes follow GET /servers/{id}/tools: a credential problem is 400 with
# the builder's reason as the code, a start failure is 502.
_HTTP_STATUS: Mapping[ConnectorProbeOutcome, tuple[int, str]] = {
    ConnectorProbeOutcome.OK: (200, "ok"),
    ConnectorProbeOutcome.NOT_FOUND: (404, "mcp_server_not_found"),
    ConnectorProbeOutcome.NOT_DECLARED: (404, "probe_not_declared"),
    ConnectorProbeOutcome.UNPROBEABLE: (400, "unprobeable"),
    ConnectorProbeOutcome.NOT_CONNECTED: (400, "authorization_required"),
    ConnectorProbeOutcome.UNSUPPORTED: (400, "unsupported"),
    ConnectorProbeOutcome.FAILED_TO_START: (502, "mcp_tools_unavailable"),
    ConnectorProbeOutcome.NO_TOOLS: (502, "mcp_tools_unavailable"),
    ConnectorProbeOutcome.TIMED_OUT: (502, "mcp_tools_unavailable"),
    ConnectorProbeOutcome.PROBE_TOOL_MISSING: (502, "probe_tool_missing"),
    ConnectorProbeOutcome.PROBE_FAILED: (502, "probe_failed"),
    ConnectorProbeOutcome.PROBE_ASSERTION_FAILED: (502, "probe_assertion_failed"),
}


@dataclass(frozen=True)
class ConnectorProbeResult:
    """Outcome of one probe. Every field is a public-safe code or identifier."""

    outcome: ConnectorProbeOutcome
    server_id: int
    app_id: str | None = None
    tool: str | None = None
    reason: str | None = None

    @property
    def http_status(self) -> int:
        return _HTTP_STATUS[self.outcome][0]

    def payload(self) -> dict[str, Any]:
        """The route's response body, or its error ``detail``."""
        _, code = _HTTP_STATUS[self.outcome]
        if self.outcome is ConnectorProbeOutcome.NOT_CONNECTED and self.reason:
            code = self.reason
        body: dict[str, Any] = {
            "code": code,
            "outcome": self.outcome.value,
            "server_id": self.server_id,
            "app_id": self.app_id,
            "tool": self.tool,
        }
        if self.reason is not None and code != self.reason:
            body["reason"] = self.reason
        return body


class ConnectorProbeBusy(Exception):
    """Another probe is running in this process.

    A probe starts a connector and calls the vendor, so probes run one at a
    time per process instead of queueing (GET /servers/{id}/tools has no cap
    either, #3265).
    """


# Bound on the probe call when the init timeout is 0: 0 disables only the
# agent-setup/listing bound, never the probe call.
_PROBE_CALL_FALLBACK_SECONDS = 60

# A plain flag rather than an asyncio primitive: it is read and set without
# yielding, and binds to no event loop.
_probe_running = False


def is_connector_probe_user(user: Any) -> bool:
    """Whether ``user`` is named in ``XAGENT_CONNECTOR_PROBE_USERS``.

    An entry matches the exact username or the email address compared
    case-insensitively (the comparison the login path uses). Both columns are
    unique; ``is_admin`` grants nothing here, so the monitor account stays a
    plain user.
    """
    allowed = get_connector_probe_users()
    if not allowed:
        return False
    username = getattr(user, "username", None)
    if isinstance(username, str) and username in allowed:
        return True
    email = getattr(user, "email", None)
    if not isinstance(email, str):
        return False
    normalized = email.strip().lower()
    return any(entry.lower() == normalized for entry in allowed)


async def run_connector_probe(
    db: Any, *, user: Any, server_id: int
) -> ConnectorProbeResult:
    """Run the probe declared for ``server_id``'s catalog app, as ``user``.

    Raises ``ConnectorProbeBusy`` when a probe is already running.
    """
    global _probe_running
    if _probe_running:
        raise ConnectorProbeBusy()
    _probe_running = True
    try:
        return await _run(db, user=user, server_id=server_id)
    finally:
        _probe_running = False


async def _run(db: Any, *, user: Any, server_id: int) -> ConnectorProbeResult:
    from ..mcp_apps import get_app_for_mcp_server
    from ..models.mcp import MCPServer, UserMCPServer

    server = (
        db.query(MCPServer)
        .join(UserMCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(
            UserMCPServer.user_id == int(user.id),
            UserMCPServer.is_active,
            MCPServer.id == server_id,
        )
        .first()
    )
    if server is None:
        return ConnectorProbeResult(ConnectorProbeOutcome.NOT_FOUND, server_id)

    app = get_app_for_mcp_server(db, server)
    app_id = app.get("id") if app else None
    if (
        app is None
        or not isinstance(app_id, str)
        or not _owns_catalog_identity(db, server, app)
    ):
        return ConnectorProbeResult(ConnectorProbeOutcome.NOT_DECLARED, server_id)

    def result(
        outcome: ConnectorProbeOutcome, *, tool: str | None, reason: str | None
    ) -> ConnectorProbeResult:
        return ConnectorProbeResult(outcome, server_id, app_id, tool, reason)

    unprobeable = get_unprobeable_reason(app_id)
    if unprobeable is not None:
        return result(ConnectorProbeOutcome.UNPROBEABLE, tool=None, reason=unprobeable)
    probe = get_connector_probe(app_id)
    if probe is None:
        return result(ConnectorProbeOutcome.NOT_DECLARED, tool=None, reason=None)
    tool_name = probe.tool

    listing = await list_connector_tools(
        db,
        user=user,
        server_id=server_id,
        name_prefix="probe_",
        refuse_sandboxed=True,
    )
    if listing.outcome is not ConnectorToolsOutcome.OK or listing.load_result is None:
        return result(
            _LISTING_OUTCOMES[listing.outcome], tool=tool_name, reason=listing.reason
        )

    tool = _find_tool(listing.load_result.tools, tool_name)
    if tool is None:
        return result(
            ConnectorProbeOutcome.PROBE_TOOL_MISSING, tool=tool_name, reason=None
        )

    timeout = get_mcp_tool_init_timeout_seconds()
    try:
        call = tool.run_json_async(dict(probe.arguments))
        raw = await asyncio.wait_for(call, timeout or _PROBE_CALL_FALLBACK_SECONDS)
    except asyncio.TimeoutError:
        return result(
            ConnectorProbeOutcome.TIMED_OUT, tool=tool_name, reason="probe_call"
        )
    except Exception as error:
        logger.warning(
            "Connector probe %s for %s raised %s",
            tool_name,
            app_id,
            type(error).__name__,
        )
        return result(
            ConnectorProbeOutcome.PROBE_FAILED, tool=tool_name, reason="exception"
        )

    outcome, reason = evaluate_probe_result(raw, probe)
    return result(outcome, tool=tool_name, reason=reason)


def evaluate_probe_result(
    raw: Any, probe: ConnectorProbe
) -> tuple[ConnectorProbeOutcome, str | None]:
    """Judge a tool call's normalized result against the probe.

    The adapter's result is ``{"content": [...], "is_error": bool, ...}``.
    A flagged error fails the probe. Otherwise the single text item must be a
    JSON object; one carrying ``status`` must say ``success``, and one without
    it must have declared ``expect_keys``. Whenever ``expect_keys`` is declared,
    the object must carry every entry (presence only, values are not checked),
    with or without the status envelope.
    """
    if not isinstance(raw, Mapping):
        return ConnectorProbeOutcome.PROBE_FAILED, "invalid_result"
    if raw.get("is_error"):
        return ConnectorProbeOutcome.PROBE_FAILED, "tool_error"
    payload = _single_json_object(raw.get("content"))
    if payload is None:
        return ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "not_a_json_object"
    if "status" in payload:
        if payload["status"] != "success":
            return ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "status_not_success"
    elif not probe.expect_keys:
        return ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "no_status"
    if any(key not in payload for key in probe.expect_keys):
        return ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "missing_keys"
    return ConnectorProbeOutcome.OK, None


def _single_json_object(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, list) or len(content) != 1:
        return None
    item = content[0]
    text = item.get("text") if isinstance(item, Mapping) else None
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _find_tool(tools: Any, name: str) -> Any | None:
    """The loaded adapter whose MCP tool is ``name``.

    The loader prefixes and sanitizes the agent-facing name, so the match is
    on the wire name the adapter keeps in ``mcp_tool``.
    """
    for tool in tools:
        mcp_tool = getattr(tool, "mcp_tool", None)
        if getattr(mcp_tool, "name", None) == name and hasattr(tool, "run_json_async"):
            return tool
    return None


def _owns_catalog_identity(db: Any, server: Any, app: Mapping[str, Any]) -> bool:
    """Whether ``server`` is the catalog's row for ``app``, not a lookalike.

    OAuth-transport rows are written by the platform, so their resolution is
    trusted. For every other row a custom row's auth is caller-authored and
    must not be able to claim a catalog identity (api/mcp.py), so the row
    must be what catalog connect adopts: named exactly ``app["id"]``, not
    user-owned (``_reject_user_owned_catalog_squat``), and launching what the
    catalog launches (``_ensure_catalog_app_server`` for stdio,
    ``_ensure_catalog_mcp_oauth_server`` for remote rows).
    """
    from ..mcp_apps import is_user_owned_server

    transport = str(server.transport or "").lower()
    if transport == "oauth":
        return True
    if str(server.name or "") != app.get("id") or is_user_owned_server(db, server.id):
        return False
    launch = app.get("launch_config")
    if not isinstance(launch, Mapping):
        return False
    if transport != str(app.get("transport") or "").lower():
        return False
    if transport == "stdio":
        return bool(server.command == launch.get("command"))
    url = launch.get("url")
    return isinstance(url, str) and server.url == url

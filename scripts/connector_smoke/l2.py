"""L2 connection smoke: pure assertions over recorded HTTP responses.

Every catalog connector the monitor account has connected is started the way
a task would start it (``GET /api/mcp/servers/{id}/tools``, which goes through
``src/xagent/web/services/mcp_connector_session.py``) and must list at least
one tool. Nothing here performs I/O. Failure reasons carry only reason codes
and failure phases the server already reports as public-safe; a code is
printed only when it looks like one, and never a message or diagnostic. App
ids are printed as the catalog returns them, as in L1.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .l1 import CheckResult

_PLAIN_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")

# The codes the server treats as a missing or unusable credential
# (_CREDENTIAL_REASONS in src/xagent/web/services/mcp_connector_session.py).
# The route answers 400 for every builder diagnostic, so any other 400 code,
# such as a remote OAuth discovery failure, means the server could not start.
_CREDENTIAL_CODES = frozenset(
    {
        "authorization_required",
        "insufficient_scope",
        "oauth_token_refresh_failed",
        "oauth_token_required",
        "oauth_token_resolver_failed",
        "token_refresh_failed",
    }
)


@dataclass(frozen=True)
class ConnectorTarget:
    app_id: str
    server_id: int


def select_targets(
    apps: Iterable[Mapping[str, Any]], *, expect: Iterable[str] = ()
) -> tuple[list[ConnectorTarget], list[str]]:
    """Connected catalog apps to start, and expected apps that are not connected.

    Every connected app is started, expected or not, so a connector the account
    gains is checked without editing the expected list. The list exists so a
    connector the account loses fails instead of silently leaving the run.
    """
    targets: dict[str, ConnectorTarget] = {}
    for app in apps:
        app_id = app.get("id")
        server_id = app.get("server_id")
        if (
            isinstance(app_id, str)
            and app.get("is_connected") is True
            and type(server_id) is int
        ):
            targets[app_id] = ConnectorTarget(app_id, server_id)
    missing = sorted(set(expect) - targets.keys())
    return [targets[app_id] for app_id in sorted(targets)], missing


def select_unconnected(
    servers: Iterable[Mapping[str, Any]],
    *,
    connected: Iterable[str],
    missing: Iterable[str],
) -> list[str]:
    """Apps to report as not connected.

    That is every expected app that is not connected, plus every builtin OAuth
    app whose stored token was cleared (``needs_reconnect``), expected or not:
    the catalog stops listing such an app as connected, so it would otherwise
    leave the run without a trace. A builtin OAuth token that expired with no
    refresh token is not caught: the server list still reports it connected, so
    such an app is reported only when it is expected.
    """
    stale = {
        server["app_id"]
        for server in servers
        if isinstance(server.get("app_id"), str)
        and server.get("connection_status") == "needs_reconnect"
    }
    return sorted((set(missing) | stale) - set(connected))


def check_unconnected(
    app_id: str,
    apps: Iterable[Mapping[str, Any]],
    servers: Iterable[Mapping[str, Any]],
) -> CheckResult:
    """Why an expected app is not connected.

    The catalog drops a builtin OAuth app whose stored token was cleared, but
    its server row stays, marked ``needs_reconnect``. That status exists only
    for builtin OAuth rows; every other type reads ``null``.
    """
    name = _name(app_id)
    if not any(app.get("id") == app_id for app in apps):
        return CheckResult(name, False, "not in the catalog")
    if any(
        server.get("app_id") == app_id
        and server.get("connection_status") == "needs_reconnect"
        for server in servers
    ):
        return CheckResult(name, False, "needs reconnect")
    return CheckResult(name, False, "not connected")


def check_tools_response(app_id: str, status_code: int, payload: Any) -> CheckResult:
    """Classify one ``GET /api/mcp/servers/{id}/tools`` response."""
    name = _name(app_id)
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if status_code == 200:
        if not isinstance(payload, dict):
            return CheckResult(name, False, "unexpected response (HTTP 200)")
        count = payload.get("tool_count")
        if type(count) is int and count > 0:
            return CheckResult(name, True, f"{count} tools")
        return CheckResult(name, False, "no tools")
    if status_code == 400 and isinstance(detail, dict):
        code = _code(detail.get("code"))
        if code == "unsupported":
            return CheckResult(name, False, "unsupported outside a task")
        if code in _CREDENTIAL_CODES:
            return CheckResult(name, False, f"not connected ({code})")
        return CheckResult(name, False, f"failed to start ({code})")
    if (
        status_code == 502
        and isinstance(detail, dict)
        and detail.get("code") == "mcp_tools_unavailable"
    ):
        return CheckResult(name, False, _unavailable_reason(detail))
    if status_code == 404:
        return CheckResult(name, False, "server not found")
    return CheckResult(name, False, f"HTTP {status_code}")


def check_timeout(app_id: str, timeout: float) -> CheckResult:
    return CheckResult(_name(app_id), False, f"timed out after {timeout:g}s")


def check_request_error(app_id: str, reason: str) -> CheckResult:
    """A tools request that got no HTTP response; ``reason`` is a SmokeError's."""
    return CheckResult(_name(app_id), False, reason)


def _unavailable_reason(detail: Mapping[str, Any]) -> str:
    # A failure before any load (sandbox lease, config scan) carries a reason;
    # a load failure carries the phase of each server that failed.
    if detail.get("reason") is not None:
        return f"failed to start ({_code(detail.get('reason'))})"
    failures = detail.get("failures")
    phases = [
        _code(failure.get("phase"))
        for failure in (failures if isinstance(failures, list) else [])
        if isinstance(failure, dict)
    ]
    # No recorded failure at all is the route's own "no tools" answer.
    if all(phase == "no_tools_returned" for phase in phases):
        return "no tools"
    if "auth_rejected" in phases:
        # The vendor refused the credential (HTTP 401/403).
        return "not connected (auth_rejected)"
    return f"failed to start ({phases[0]})"


def _code(value: object) -> str:
    if isinstance(value, str) and _PLAIN_CODE.fullmatch(value):
        return value
    return "unknown"


def _name(app_id: str) -> str:
    return f"l2:{app_id}"

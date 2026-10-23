"""L3 tool-call smoke: pure assertions over recorded HTTP responses.

Every catalog connector the monitor account has connected gets its declared
read-only probe run once (``POST /api/mcp/servers/{id}/probe``, which goes
through ``src/xagent/web/services/connector_probe.py``). The route picks the
tool and its arguments from ``src/xagent/web/connector_probes.py``; this
script only names the server. Nothing here performs I/O.

The route reports an ``outcome`` on every answer, and classification keys on
it. Reasons are printed only when they look like plain codes, as in L2, so
the output never carries a message, a diagnostic or a vendor's error text.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .l1 import CheckResult, format_result
from .l2 import _code

# The route's outcomes (ConnectorProbeOutcome in
# src/xagent/web/services/connector_probe.py); a test pins this set.
OUTCOMES = frozenset(
    {
        "ok",
        "not_found",
        "not_declared",
        "unprobeable",
        "not_connected",
        "unsupported",
        "failed_to_start",
        "no_tools",
        "timed_out",
        "probe_tool_missing",
        "probe_failed",
        "probe_assertion_failed",
    }
)

# A connector that is connected but has nothing to probe, with the status the
# route answers it with: a remote MCP server (L2 already proves its
# credential), an npx row, an execution-scoped one or a document connector
# whose every read needs an object id (400 unprobeable), or a custom catalog
# app with no declaration (404 not_declared, naming that app).
_SKIPPED = {"unprobeable": 400, "not_declared": 404}

FORBIDDEN = CheckResult(
    "probe",
    False,
    "this account may not run connector probes (XAGENT_CONNECTOR_PROBE_USERS)",
)


@dataclass(frozen=True)
class Skipped:
    """A connected connector the run does not probe; not a check."""

    name: str
    reason: str


def is_forbidden(status_code: int, payload: Any) -> bool:
    """Whether the route refused the account itself, which holds for every
    connector, so the run stops at the first answer."""
    return status_code == 403 and _detail(payload).get("code") == "probe_forbidden"


def check_probe_response(
    app_id: str, status_code: int, payload: Any
) -> CheckResult | Skipped:
    """Classify one ``POST /api/mcp/servers/{id}/probe`` response."""
    name = _name(app_id)
    if status_code == 200:
        if not isinstance(payload, dict) or payload.get("outcome") != "ok":
            return CheckResult(name, False, "unexpected response (HTTP 200)")
        return CheckResult(name, True, _code(payload.get("tool")))
    detail = _detail(payload)
    if status_code == 429 and detail.get("code") == "probe_busy":
        # The server keeps running a probe this client gave up on, so the next
        # connector after a client timeout can land here.
        return CheckResult(
            name, False, "probe busy (another probe, or one that timed out, still runs)"
        )
    outcome = detail.get("outcome")
    if outcome not in OUTCOMES or outcome == "ok":
        return CheckResult(name, False, f"HTTP {status_code}")
    reason = _code(detail.get("reason")) if detail.get("reason") is not None else ""
    if outcome in _SKIPPED:
        if status_code != _SKIPPED[outcome]:
            return CheckResult(name, False, f"HTTP {status_code}")
        if outcome == "unprobeable":
            return Skipped(name, f"unprobeable ({reason})" if reason else "unprobeable")
        if detail.get("app_id") != app_id:
            # The server names no app, or another one, when the row is not the
            # catalog's own: a custom row claiming the app, a squat, a launch
            # that drifted from the catalog. The catalog still lists the app as
            # connected, so a skip would hide it.
            return CheckResult(
                name, False, "not the catalog's row for this app (no probe)"
            )
        return Skipped(name, "no probe declared")
    return CheckResult(name, False, _failure(outcome, detail, reason))


def check_timeout(app_id: str, timeout: float) -> CheckResult:
    return CheckResult(_name(app_id), False, f"timed out after {timeout:g}s")


def check_request_error(app_id: str, reason: str) -> CheckResult:
    """A probe request that got no HTTP response; ``reason`` is a SmokeError's."""
    return CheckResult(_name(app_id), False, reason)


def relabel(result: CheckResult) -> CheckResult:
    """An L2 per-connector result under this layer's name."""
    _, sep, app_id = result.name.partition(":")
    return CheckResult(_name(app_id), result.ok, result.reason) if sep else result


def format_outcome(result: CheckResult | Skipped) -> str:
    if isinstance(result, Skipped):
        return f"SKIP {result.name}: {result.reason}"
    return format_result(result)


def _failure(outcome: str, detail: Mapping[str, Any], reason: str) -> str:
    def with_reason(label: str) -> str:
        return f"{label} ({reason})" if reason else label

    if outcome == "not_found":
        return "server not found"
    if outcome == "not_connected":
        # The route puts the builder's credential reason in ``code``.
        return f"not connected ({_code(detail.get('code'))})"
    if outcome == "unsupported":
        if reason == "sandboxed":
            # The call would run on the web host after the listing's lease.
            return "cannot be probed: runs in a sandbox"
        return with_reason("unsupported outside a task")
    if outcome == "failed_to_start":
        return with_reason("failed to start")
    if outcome == "no_tools":
        return "no tools"
    if outcome == "timed_out":
        return with_reason("timed out on the server")
    if outcome == "probe_tool_missing":
        return f"probe tool missing ({_code(detail.get('tool'))})"
    if outcome == "probe_failed":
        return with_reason("probe failed")
    if outcome == "probe_assertion_failed":
        return with_reason("unexpected probe result")
    # OUTCOMES is pinned to the route's; a new one fails the contract test
    # before it can reach a run.
    return with_reason(_code(outcome))


def _detail(payload: Any) -> Mapping[str, Any]:
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, dict) else {}


def _name(app_id: str) -> str:
    return f"l3:{app_id}"

"""L3 tool-call smoke (#3247): recorded response shapes and an in-process fake.

The fake answers ``/api/auth/login``, ``/api/mcp/apps``, ``/api/mcp/servers``
and ``POST /api/mcp/servers/{id}/probe``. The contract tests build every
response from the route's own result type
(``src/xagent/web/services/connector_probe.py``), so a change on either side
fails here; no test opens a socket.
"""

import importlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

# Importable under both pytest and `unittest discover -s tests/scripts`.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
cli = importlib.import_module("connector_smoke.cli")
l1 = importlib.import_module("connector_smoke.l1")
l3 = importlib.import_module("connector_smoke.l3")

BASE = "https://cloud.example.com"
ACCESS_TOKEN = "access-token-SECRET"


def _app(app_id: str, server_id: int | None) -> dict[str, Any]:
    app: dict[str, Any] = {"id": app_id, "is_connected": server_id is not None}
    if server_id is not None:
        app["server_id"] = server_id
    return app


def _ok(tool: str, server_id: int = 7, app_id: str = "github") -> tuple[int, Any]:
    return 200, {
        "code": "ok",
        "outcome": "ok",
        "server_id": server_id,
        "app_id": app_id,
        "tool": tool,
    }


def _refused(
    status_code: int,
    outcome: str,
    *,
    code: str,
    reason: str | None = None,
    tool: str | None = "github_get_current_user",
) -> tuple[int, Any]:
    detail: dict[str, Any] = {
        "code": code,
        "outcome": outcome,
        "server_id": 7,
        "app_id": "github",
        "tool": tool,
    }
    if reason is not None:
        detail["reason"] = reason
    return status_code, {"detail": detail}


# --- pure classification ---------------------------------------------------


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (_ok("github_get_current_user"), (True, "github_get_current_user")),
        (
            _refused(400, "not_connected", code="oauth_token_required"),
            (False, "not connected (oauth_token_required)"),
        ),
        (
            _refused(400, "not_connected", code="auth_rejected"),
            (False, "not connected (auth_rejected)"),
        ),
        (
            _refused(400, "unsupported", code="unsupported", reason="execution_scoped"),
            (False, "unsupported outside a task (execution_scoped)"),
        ),
        (
            _refused(
                502,
                "failed_to_start",
                code="mcp_tools_unavailable",
                reason="initialize",
            ),
            (False, "failed to start (initialize)"),
        ),
        (
            _refused(502, "no_tools", code="mcp_tools_unavailable"),
            (False, "no tools"),
        ),
        (
            _refused(
                502, "timed_out", code="mcp_tools_unavailable", reason="probe_call"
            ),
            (False, "timed out on the server (probe_call)"),
        ),
        (
            _refused(502, "probe_tool_missing", code="probe_tool_missing"),
            (False, "probe tool missing (github_get_current_user)"),
        ),
        (
            _refused(502, "probe_failed", code="probe_failed", reason="tool_error"),
            (False, "probe failed (tool_error)"),
        ),
        (
            _refused(
                502,
                "probe_assertion_failed",
                code="probe_assertion_failed",
                reason="status_not_success",
            ),
            (False, "unexpected probe result (status_not_success)"),
        ),
        (
            _refused(404, "not_found", code="mcp_server_not_found", tool=None),
            (False, "server not found"),
        ),
        (
            (429, {"detail": {"code": "probe_busy", "message": "busy"}}),
            (False, "probe busy (another probe, or one that timed out, still runs)"),
        ),
        # A deployment without the probe route.
        ((404, {"detail": "Not Found"}), (False, "HTTP 404")),
        ((504, "gateway timeout"), (False, "HTTP 504")),
        ((200, ["not", "an", "object"]), (False, "unexpected response (HTTP 200)")),
        ((200, {"outcome": "probe_failed"}), (False, "unexpected response (HTTP 200)")),
        # An ``ok`` outcome on an error status is not a pass.
        ((502, {"detail": {"outcome": "ok"}}), (False, "HTTP 502")),
    ],
)
def test_probe_responses_are_classified(response, expected) -> None:
    status_code, payload = response
    result = l3.check_probe_response("github", status_code, payload)
    assert result == l1.CheckResult("l3:github", *expected)


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (
            _refused(400, "unprobeable", code="unprobeable", reason="covered_by_l2"),
            "unprobeable (covered_by_l2)",
        ),
        (
            _refused(404, "not_declared", code="probe_not_declared", tool=None),
            "no probe declared",
        ),
    ],
)
def test_connectors_without_a_probe_are_skipped(response, reason) -> None:
    """An unprobeable builtin, or a custom catalog app the route names as
    itself (``acme``) with nothing declared."""
    status_code, payload = response
    payload["detail"]["app_id"] = "acme"
    assert l3.check_probe_response("acme", status_code, payload) == l3.Skipped(
        "l3:acme", reason
    )


@pytest.mark.parametrize("route_app_id", [None, "stripe"])
def test_no_probe_for_a_row_that_is_not_the_catalogs_fails(route_app_id) -> None:
    """The server answers not_declared without an app id when the row is not the
    catalog's own (a custom row claiming the app, a squat, a drifted launch).
    The catalog listed it as connected, so skipping it would let a connector
    drop out of the run unnoticed."""
    status_code, payload = _refused(
        404, "not_declared", code="probe_not_declared", tool=None
    )
    payload["detail"]["app_id"] = route_app_id

    result = l3.check_probe_response("github", status_code, payload)

    assert result == l1.CheckResult(
        "l3:github", False, "not the catalog's row for this app (no probe)"
    )


@pytest.mark.parametrize(
    "response",
    [
        _refused(502, "unprobeable", code="unprobeable", reason="covered_by_l2"),
        _refused(400, "not_declared", code="probe_not_declared", tool=None),
    ],
)
def test_a_skip_outcome_on_another_status_is_not_a_skip(response) -> None:
    status_code, payload = response
    result = l3.check_probe_response("github", status_code, payload)
    assert result == l1.CheckResult("l3:github", False, f"HTTP {status_code}")


def test_a_sandboxed_connector_says_so() -> None:
    status_code, payload = _refused(
        400, "unsupported", code="unsupported", reason="sandboxed"
    )
    result = l3.check_probe_response("github", status_code, payload)
    assert result == l1.CheckResult(
        "l3:github", False, "cannot be probed: runs in a sandbox"
    )


def test_a_code_from_the_server_is_never_echoed_unless_it_is_a_plain_code() -> None:
    status_code, payload = _refused(
        502, "probe_failed", code="probe_failed", reason="Bad token ghp_SECRET"
    )
    result = l3.check_probe_response("github", status_code, payload)
    assert result == l1.CheckResult("l3:github", False, "probe failed (unknown)")


def test_only_the_probe_permission_refusal_is_forbidden() -> None:
    assert l3.is_forbidden(403, {"detail": {"code": "probe_forbidden"}})
    assert not l3.is_forbidden(403, {"detail": "Forbidden"})
    assert not l3.is_forbidden(401, {"detail": {"code": "probe_forbidden"}})


# --- the contract with the route ---------------------------------------------


def test_outcomes_match_the_routes() -> None:
    from xagent.web.services.connector_probe import ConnectorProbeOutcome

    assert l3.OUTCOMES == {outcome.value for outcome in ConnectorProbeOutcome}


def _route_response(result) -> tuple[int, Any]:
    """What POST /servers/{id}/probe answers for ``result`` (api/mcp.py)."""
    if result.http_status == 200:
        return 200, result.payload()
    return result.http_status, {"detail": result.payload()}


def test_every_route_outcome_gets_a_specific_line() -> None:
    from xagent.web.services.connector_probe import (
        ConnectorProbeOutcome,
        ConnectorProbeResult,
    )

    for outcome in ConnectorProbeOutcome:
        result = ConnectorProbeResult(
            outcome, 7, "github", "github_get_current_user", "initialize"
        )
        status_code, payload = _route_response(result)

        line = l3.check_probe_response("github", status_code, payload)

        assert not line.reason.startswith("HTTP "), (outcome, line)
        assert "unexpected response" not in line.reason, (outcome, line)
        if outcome is ConnectorProbeOutcome.OK:
            assert line == l1.CheckResult("l3:github", True, "github_get_current_user")
        elif outcome.value in {"unprobeable", "not_declared"}:
            assert isinstance(line, l3.Skipped), outcome
        else:
            assert isinstance(line, l1.CheckResult) and not line.ok, outcome


def test_a_route_credential_code_is_reported() -> None:
    from xagent.web.services.connector_probe import (
        ConnectorProbeOutcome,
        ConnectorProbeResult,
    )

    result = ConnectorProbeResult(
        ConnectorProbeOutcome.NOT_CONNECTED,
        7,
        "github",
        "github_get_current_user",
        "oauth_token_required",
    )

    line = l3.check_probe_response("github", *_route_response(result))

    assert line == l1.CheckResult(
        "l3:github", False, "not connected (oauth_token_required)"
    )


def test_a_route_identity_refusal_fails() -> None:
    """What the route answers for a row that is not the catalog's own."""
    from xagent.web.services.connector_probe import (
        ConnectorProbeOutcome,
        ConnectorProbeResult,
    )

    result = ConnectorProbeResult(ConnectorProbeOutcome.NOT_DECLARED, 7)

    line = l3.check_probe_response("stripe", *_route_response(result))

    assert line == l1.CheckResult(
        "l3:stripe", False, "not the catalog's row for this app (no probe)"
    )


@pytest.mark.asyncio
async def test_the_routes_refusals_are_recognised(monkeypatch) -> None:
    from fastapi import HTTPException

    from xagent.web.api.mcp import probe_mcp_server
    from xagent.web.services.connector_probe import ConnectorProbeBusy

    user = SimpleNamespace(id=1, username="monitor", email=None, is_admin=False)

    monkeypatch.delenv("XAGENT_CONNECTOR_PROBE_USERS", raising=False)
    with pytest.raises(HTTPException) as refused:
        await probe_mcp_server(7, user, db=object())
    assert l3.is_forbidden(refused.value.status_code, {"detail": refused.value.detail})

    async def busy(*args, **kwargs):
        raise ConnectorProbeBusy()

    monkeypatch.setenv("XAGENT_CONNECTOR_PROBE_USERS", "monitor")
    monkeypatch.setattr("xagent.web.services.connector_probe.run_connector_probe", busy)
    with pytest.raises(HTTPException) as busy_answer:
        await probe_mcp_server(7, user, db=object())
    line = l3.check_probe_response(
        "github", busy_answer.value.status_code, {"detail": busy_answer.value.detail}
    )
    assert (
        line.reason == "probe busy (another probe, or one that timed out, still runs)"
    )


# --- the whole run against a fake deployment -------------------------------


class FakeDeployment:
    def __init__(self) -> None:
        self.catalog = [
            _app("github", 7),
            _app("notion", 3),
            _app("stripe", 5),
            _app("google-calendar", None),
        ]
        self.servers = [
            {"id": 7, "app_id": "github", "connection_status": "connected"},
            {"id": 3, "app_id": None, "connection_status": None},
            {"id": 5, "app_id": None, "connection_status": None},
        ]
        self.probes: dict[int, tuple[int, Any]] = {
            7: _ok("github_get_current_user", 7, "github"),
            3: _refused(400, "unprobeable", code="unprobeable", reason="covered_by_l2"),
            5: _ok("stripe_list_customers", 5, "stripe"),
        }
        self.slow: set[int] = set()
        self.errors: set[int] = set()
        self.servers_status = 200
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/auth/login":
            body = json.loads(request.content)
            if body.get("password") != "pw":
                return httpx.Response(401, json={"detail": "Invalid credentials"})
            return httpx.Response(200, json={"access_token": ACCESS_TOKEN})
        if request.headers.get("authorization") != f"Bearer {ACCESS_TOKEN}":
            return httpx.Response(401, json={"detail": "Not authenticated"})
        if path == "/api/mcp/apps":
            return httpx.Response(200, json=self.catalog)
        if path == "/api/mcp/servers":
            return httpx.Response(self.servers_status, json=self.servers)
        parts = path.split("/")
        if (
            request.method == "POST"
            and len(parts) == 6
            and parts[1:4] == ["api", "mcp", "servers"]
            and parts[5] == "probe"
        ):
            server_id = int(parts[4])
            if server_id in self.slow:
                raise httpx.ReadTimeout("timed out", request=request)
            if server_id in self.errors:
                raise httpx.ConnectError("refused", request=request)
            status_code, payload = self.probes.get(
                server_id, (404, {"detail": "Not Found"})
            )
            return httpx.Response(status_code, json=payload)
        return httpx.Response(404, json={"detail": "Not Found"})


ENV = {
    "XAGENT_CONNECTOR_SMOKE_BASE_URL": BASE,
    "XAGENT_CONNECTOR_SMOKE_LOGIN": "monitor",
    "XAGENT_CONNECTOR_SMOKE_PASSWORD": "pw",
}


def _run(
    fake: FakeDeployment, *args: str, env: dict[str, str] | None = None
) -> tuple[int, list[str]]:
    out = io.StringIO()
    code = cli.main(
        ["l3", *args],
        env={**ENV, **(env or {})},
        transport=httpx.MockTransport(fake.handler),
        out=out,
    )
    return code, out.getvalue().splitlines()


def _probe_requests(fake: FakeDeployment) -> list[tuple[str, str]]:
    return [
        (request.method, request.url.path)
        for request in fake.requests
        if request.url.path.endswith("/probe")
    ]


def test_every_connected_connector_is_probed_once_in_order() -> None:
    fake = FakeDeployment()

    code, lines = _run(fake)

    assert code == 0, lines
    assert lines[:3] == [
        "PASS l3:github: github_get_current_user",
        "SKIP l3:notion: unprobeable (covered_by_l2)",
        "PASS l3:stripe: stripe_list_customers",
    ]
    assert lines[3].startswith("2 checks, 0 failed, 1 skipped in ")
    # Sorted by app id, POST only, no tool name or arguments in the request.
    assert _probe_requests(fake) == [
        ("POST", "/api/mcp/servers/7/probe"),
        ("POST", "/api/mcp/servers/3/probe"),
        ("POST", "/api/mcp/servers/5/probe"),
    ]
    assert all(
        request.content == b""
        for request in fake.requests
        if request.url.path.endswith("/probe")
    )


def test_a_failed_probe_fails_the_run_and_the_others_still_run() -> None:
    fake = FakeDeployment()
    fake.probes[7] = _refused(
        502,
        "probe_assertion_failed",
        code="probe_assertion_failed",
        reason="status_not_success",
    )

    code, lines = _run(fake)

    assert code == 1
    assert "FAIL l3:github: unexpected probe result (status_not_success)" in lines
    assert "PASS l3:stripe: stripe_list_customers" in lines


def test_a_timed_out_probe_fails_and_the_run_continues() -> None:
    fake = FakeDeployment()
    fake.slow.add(7)

    code, lines = _run(fake, "--probe-timeout", "5")

    assert code == 1
    assert "FAIL l3:github: timed out after 5s" in lines
    assert "PASS l3:stripe: stripe_list_customers" in lines


def test_probe_requests_use_the_probe_timeout() -> None:
    fake = FakeDeployment()
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/probe"):
            seen.append(request.extensions["timeout"]["read"])
        return fake.handler(request)

    out = io.StringIO()
    cli.main(
        ["l3", "--probe-timeout", "42", "--timeout", "3"],
        env=ENV,
        transport=httpx.MockTransport(handler),
        out=out,
    )

    assert seen == [42.0, 42.0, 42.0]


def test_the_default_probe_timeout_covers_a_listing_and_a_call() -> None:
    # The server bounds the listing and then the call by
    # XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS (60 s by default) each.
    assert cli.DEFAULT_PROBE_TIMEOUT > 2 * 60


def test_a_request_that_gets_no_answer_fails_and_the_run_continues() -> None:
    fake = FakeDeployment()
    fake.errors.add(7)

    code, lines = _run(fake)

    assert code == 1
    assert "FAIL l3:github: POST request failed: ConnectError" in lines
    assert "PASS l3:stripe: stripe_list_customers" in lines


def test_an_account_outside_the_allowlist_stops_the_run_at_once() -> None:
    fake = FakeDeployment()
    for server_id in fake.probes:
        fake.probes[server_id] = (
            403,
            {"detail": {"code": "probe_forbidden", "message": "no"}},
        )

    code, lines = _run(fake)

    assert code == 1
    assert lines[0] == (
        "FAIL probe: this account may not run connector probes "
        "(XAGENT_CONNECTOR_PROBE_USERS)"
    )
    assert lines[1].startswith("1 checks, 1 failed, 0 skipped in ")
    assert len(_probe_requests(fake)) == 1


def test_a_deployment_without_the_probe_route_fails_every_connector() -> None:
    fake = FakeDeployment()
    fake.probes = {}

    code, lines = _run(fake)

    assert code == 1
    assert "FAIL l3:github: HTTP 404" in lines
    assert "FAIL l3:notion: HTTP 404" in lines


def test_expected_connectors_that_are_not_connected_fail() -> None:
    fake = FakeDeployment()

    code, lines = _run(fake, "--expect", "github,google-calendar,typo")

    assert code == 1
    assert "FAIL l3:google-calendar: not connected" in lines
    assert "FAIL l3:typo: not in the catalog" in lines


def test_expect_list_comes_from_the_environment() -> None:
    fake = FakeDeployment()

    code, lines = _run(fake, env={"XAGENT_CONNECTOR_SMOKE_EXPECT": "google-calendar"})

    assert code == 1
    assert "FAIL l3:google-calendar: not connected" in lines


def test_a_builtin_app_that_needs_reconnect_fails_even_when_not_expected() -> None:
    fake = FakeDeployment()
    fake.catalog.append(_app("gmail", None))
    fake.servers.append(
        {"id": 9, "app_id": "gmail", "connection_status": "needs_reconnect"}
    )

    code, lines = _run(fake)

    assert code == 1
    assert "FAIL l3:gmail: needs reconnect" in lines


def test_a_failed_server_list_is_reported_and_probes_still_pass() -> None:
    fake = FakeDeployment()
    fake.servers_status = 500

    code, lines = _run(fake)

    assert code == 1
    assert "FAIL servers: server list request failed: HTTP 500" in lines
    assert "PASS l3:github: github_get_current_user" in lines


def test_a_run_that_only_skips_fails() -> None:
    fake = FakeDeployment()
    fake.catalog = [_app("notion", 3)]

    code, lines = _run(fake)

    assert code == 1
    assert lines[0] == "SKIP l3:notion: unprobeable (covered_by_l2)"
    assert lines[1] == "FAIL connectors: no connected connector to probe"


def test_a_run_that_checks_nothing_fails() -> None:
    fake = FakeDeployment()
    fake.catalog = [_app("github", None)]
    fake.servers = []

    code, lines = _run(fake)

    assert code == 1
    assert lines[0] == "FAIL connectors: no connected connector to probe"


@pytest.mark.parametrize("value", ["0", "-5", "nan"])
def test_a_probe_timeout_that_is_not_a_positive_number_is_a_usage_error(
    value,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run(FakeDeployment(), "--probe-timeout", value)
    assert exc_info.value.code == 2


def test_a_cookie_is_sent_on_every_request() -> None:
    fake = FakeDeployment()

    _run(fake, "--cookie", "selected_region=sg")

    assert fake.requests
    assert all(
        request.headers.get("cookie") == "selected_region=sg"
        for request in fake.requests
    )


def test_a_login_failure_is_one_failed_check() -> None:
    code, lines = _run(FakeDeployment(), env={"XAGENT_CONNECTOR_SMOKE_PASSWORD": "no"})

    assert code == 1
    assert lines[0] == "FAIL login: login failed: HTTP 401"


def test_output_never_contains_the_token() -> None:
    fake = FakeDeployment()
    fake.probes[7] = _refused(
        502, "probe_failed", code="probe_failed", reason=ACCESS_TOKEN
    )
    fake.probes[3] = _refused(
        400, "unprobeable", code="unprobeable", reason=ACCESS_TOKEN
    )
    fake.probes[5] = (200, {"outcome": "ok", "tool": ACCESS_TOKEN})

    _code, lines = _run(fake)

    assert ACCESS_TOKEN not in "\n".join(lines)
    assert "FAIL l3:github: probe failed (unknown)" in lines
    assert "SKIP l3:notion: unprobeable (unknown)" in lines
    assert "PASS l3:stripe: unknown" in lines

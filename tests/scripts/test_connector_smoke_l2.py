"""L2 connection smoke (#3246): recorded response shapes and an in-process fake.

The fake answers ``/api/auth/login``, ``/api/mcp/apps``, ``/api/mcp/servers``
and ``/api/mcp/servers/{id}/tools`` with the shapes the real routes return
(``src/xagent/web/api/mcp.py``); no test opens a socket.
"""

import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

# Importable under both pytest and `unittest discover -s tests/scripts`.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
cli = importlib.import_module("connector_smoke.cli")
l2 = importlib.import_module("connector_smoke.l2")

BASE = "https://cloud.example.com"
ACCESS_TOKEN = "access-token-SECRET"


def _app(app_id: str, server_id: int | None) -> dict[str, Any]:
    app: dict[str, Any] = {"id": app_id, "is_connected": server_id is not None}
    if server_id is not None:
        app["server_id"] = server_id
    return app


# --- pure classification ---------------------------------------------------


def test_targets_are_every_connected_app_plus_unconnected_expected_ones() -> None:
    apps = [
        _app("slack", 7),
        _app("notion", 3),
        _app("stripe", None),
        _app("gmail", None),
        {"id": "broken", "is_connected": True, "server_id": True},
    ]

    targets, missing = l2.select_targets(apps, expect={"slack", "stripe", "typo"})

    assert targets == [
        l2.ConnectorTarget("notion", 3),
        l2.ConnectorTarget("slack", 7),
    ]
    assert missing == ["stripe", "typo"]


def test_unconnected_apps_are_expected_ones_plus_any_needing_reconnect() -> None:
    servers = [
        {"id": 9, "app_id": "gmail", "connection_status": "needs_reconnect"},
        {"id": 7, "app_id": "slack", "connection_status": "connected"},
        {"id": 3, "app_id": None, "connection_status": None},
    ]

    unconnected = l2.select_unconnected(
        servers, connected=["slack", "notion"], missing=["stripe"]
    )

    assert unconnected == ["gmail", "stripe"]


def test_unconnected_expected_app_reports_why() -> None:
    apps = [_app("slack", None), _app("stripe", None)]
    servers = [
        {"id": 9, "app_id": "slack", "connection_status": "needs_reconnect"},
        {"id": 10, "app_id": None, "connection_status": None},
    ]

    assert l2.check_unconnected("slack", apps, servers) == l2.CheckResult(
        "l2:slack", False, "needs reconnect"
    )
    assert l2.check_unconnected("stripe", apps, servers) == l2.CheckResult(
        "l2:stripe", False, "not connected"
    )
    assert l2.check_unconnected("typo", apps, servers) == l2.CheckResult(
        "l2:typo", False, "not in the catalog"
    )


def _unavailable(*phases: str, reason: str | None = None) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "code": "mcp_tools_unavailable",
        "message": "MCP server tools could not be loaded.",
        "failures": [
            {"server_name": "srv", "phase": phase, "attempts": 1} for phase in phases
        ],
    }
    if reason is not None:
        detail["reason"] = reason
    return {"detail": detail}


@pytest.mark.parametrize(
    ("status_code", "payload", "ok", "reason"),
    [
        (
            200,
            {"server_name": "Slack", "tool_count": 12, "tools": []},
            True,
            "12 tools",
        ),
        (
            200,
            {"server_name": "Slack", "tool_count": 0, "tools": []},
            False,
            "no tools",
        ),
        (
            400,
            {"detail": {"code": "oauth_token_required", "message": "..."}},
            False,
            "not connected (oauth_token_required)",
        ),
        (
            400,
            {"detail": {"code": "authorization_required", "issuer": "https://a"}},
            False,
            "not connected (authorization_required)",
        ),
        (
            400,
            {"detail": {"code": "metadata_not_found", "issuer": "https://a"}},
            False,
            "failed to start (metadata_not_found)",
        ),
        (
            400,
            {"detail": {"code": "unsupported", "message": "..."}},
            False,
            "unsupported outside a task",
        ),
        (502, _unavailable("no_tools_returned"), False, "no tools"),
        (502, _unavailable(), False, "no tools"),
        (502, _unavailable("auth_rejected"), False, "not connected (auth_rejected)"),
        (502, _unavailable("session_start"), False, "failed to start (session_start)"),
        (502, _unavailable("initialize"), False, "failed to start (initialize)"),
        (
            502,
            _unavailable(reason="sandbox_unavailable"),
            False,
            "failed to start (sandbox_unavailable)",
        ),
        (404, {"detail": "MCP server not found"}, False, "server not found"),
        (500, {"detail": "Failed to get MCP server tools"}, False, "HTTP 500"),
        (502, "<html>Bad Gateway</html>", False, "HTTP 502"),
        (200, "<html>ok</html>", False, "unexpected response (HTTP 200)"),
    ],
)
def test_tools_responses_are_classified(
    status_code: int, payload: Any, ok: bool, reason: str
) -> None:
    result = l2.check_tools_response("slack", status_code, payload)

    assert result == l2.CheckResult("l2:slack", ok, reason)


def test_a_code_from_the_server_is_never_echoed_unless_it_is_a_plain_code() -> None:
    payload = {"detail": {"code": "Bearer abc.def SECRET", "message": "x"}}

    result = l2.check_tools_response("slack", 400, payload)

    assert result.reason == "failed to start (unknown)"


# --- end-to-end against a fake deployment ------------------------------------


class FakeDeployment:
    def __init__(self) -> None:
        self.catalog = [
            _app("slack", 7),
            _app("notion", 3),
            _app("stripe", 5),
            _app("google-calendar", None),
        ]
        self.servers = [
            {"id": 7, "app_id": "slack", "connection_status": "connected"},
            {"id": 3, "app_id": None, "connection_status": None},
            {"id": 5, "app_id": None, "connection_status": None},
        ]
        self.tools: dict[int, tuple[int, Any]] = {
            7: (200, {"server_name": "Slack", "tool_count": 12, "tools": []}),
            3: (200, {"server_name": "notion", "tool_count": 4, "tools": []}),
            5: (200, {"server_name": "stripe", "tool_count": 9, "tools": []}),
        }
        self.slow: set[int] = set()
        self.errors: set[int] = set()
        self.servers_status = 200
        self.apps_status = 200
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
            return httpx.Response(self.apps_status, json=self.catalog)
        if path == "/api/mcp/servers":
            return httpx.Response(self.servers_status, json=self.servers)
        parts = path.split("/")
        if len(parts) == 6 and parts[1:4] == ["api", "mcp", "servers"]:
            server_id = int(parts[4])
            if server_id in self.slow:
                raise httpx.ReadTimeout("timed out", request=request)
            if server_id in self.errors:
                raise httpx.ConnectError("refused", request=request)
            status_code, payload = self.tools.get(server_id, (404, {"detail": "x"}))
            return httpx.Response(status_code, json=payload)
        return httpx.Response(404)


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
        ["l2", *args],
        env={**ENV, **(env or {})},
        transport=httpx.MockTransport(fake.handler),
        out=out,
    )
    return code, out.getvalue().splitlines()


def _tools_requests(fake: FakeDeployment) -> list[str]:
    return [
        request.url.path
        for request in fake.requests
        if request.url.path.endswith("/tools")
    ]


def test_every_connected_connector_starts_and_lists_tools() -> None:
    fake = FakeDeployment()

    code, lines = _run(fake)

    assert code == 0, lines
    assert lines[:3] == [
        "PASS l2:notion: 4 tools",
        "PASS l2:slack: 12 tools",
        "PASS l2:stripe: 9 tools",
    ]
    assert lines[-1].startswith("3 checks, 0 failed in ")
    # Sequential, one call per connector, in a stable order.
    assert _tools_requests(fake) == [
        "/api/mcp/servers/3/tools",
        "/api/mcp/servers/7/tools",
        "/api/mcp/servers/5/tools",
    ]


def test_a_connector_that_fails_to_start_does_not_stop_the_run() -> None:
    fake = FakeDeployment()
    fake.tools[3] = (502, _unavailable("session_start"))

    code, lines = _run(fake)

    assert code == 1
    assert "FAIL l2:notion: failed to start (session_start)" in lines
    assert "PASS l2:slack: 12 tools" in lines
    assert "PASS l2:stripe: 9 tools" in lines


def test_a_revoked_remote_grant_fails_and_the_others_pass() -> None:
    fake = FakeDeployment()
    fake.tools[3] = (502, _unavailable("auth_rejected"))

    code, lines = _run(fake)

    assert code == 1
    assert [line for line in lines if line.startswith("FAIL")] == [
        "FAIL l2:notion: not connected (auth_rejected)"
    ]


def test_a_timed_out_connector_fails_and_the_run_continues() -> None:
    fake = FakeDeployment()
    fake.slow.add(7)

    code, lines = _run(fake, "--tools-timeout", "90")

    assert code == 1
    assert "FAIL l2:slack: timed out after 90s" in lines
    assert "PASS l2:stripe: 9 tools" in lines


def test_tools_requests_use_the_tools_timeout() -> None:
    seen: list[dict[str, Any]] = []
    fake = FakeDeployment()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, **request.extensions["timeout"]})
        return fake.handler(request)

    out = io.StringIO()
    cli.main(
        ["l2", "--tools-timeout", "130", "--timeout", "15"],
        env=ENV,
        transport=httpx.MockTransport(handler),
        out=out,
    )

    by_path = {entry["path"]: entry["read"] for entry in seen}
    assert by_path["/api/mcp/apps"] == 15
    assert by_path["/api/mcp/servers"] == 15
    assert by_path["/api/mcp/servers/7/tools"] == 130


def test_expected_connectors_that_are_not_connected_fail() -> None:
    fake = FakeDeployment()
    fake.catalog.append(_app("gmail", None))
    fake.servers.append(
        {"id": 11, "app_id": "gmail", "connection_status": "needs_reconnect"}
    )

    code, lines = _run(fake, "--expect", "slack,google-calendar", "--expect", "gmail")

    assert code == 1
    assert "FAIL l2:google-calendar: not connected" in lines
    assert "FAIL l2:gmail: needs reconnect" in lines
    assert "PASS l2:slack: 12 tools" in lines


def test_expect_list_comes_from_the_environment() -> None:
    fake = FakeDeployment()

    code, lines = _run(fake, env={"XAGENT_CONNECTOR_SMOKE_EXPECT": "google-calendar"})

    assert code == 1
    assert "FAIL l2:google-calendar: not connected" in lines


def test_a_builtin_app_that_needs_reconnect_fails_even_when_not_expected() -> None:
    fake = FakeDeployment()
    fake.catalog.append(_app("gmail", None))
    fake.servers.append(
        {"id": 11, "app_id": "gmail", "connection_status": "needs_reconnect"}
    )

    code, lines = _run(fake)

    assert code == 1
    assert "FAIL l2:gmail: needs reconnect" in lines


def test_a_connected_app_is_not_reported_when_its_server_row_needs_reconnect() -> None:
    # /api/mcp/apps picks the first connectable account and /api/mcp/servers
    # lists the newest, so the two can disagree about one app.
    fake = FakeDeployment()
    fake.servers[0]["connection_status"] = "needs_reconnect"

    _code, lines = _run(fake)

    assert "PASS l2:slack: 12 tools" in lines
    assert not any(line.startswith("FAIL l2:slack") for line in lines)
    assert _tools_requests(fake).count("/api/mcp/servers/7/tools") == 1


def test_a_request_that_gets_no_answer_fails_and_the_run_continues() -> None:
    fake = FakeDeployment()
    fake.errors.add(7)

    code, lines = _run(fake)

    assert code == 1
    assert "FAIL l2:slack: GET request failed: ConnectError" in lines
    assert "PASS l2:stripe: 9 tools" in lines


def test_a_failed_server_list_is_reported_and_connected_apps_still_pass() -> None:
    fake = FakeDeployment()
    fake.servers_status = 500

    code, lines = _run(fake, "--expect", "google-calendar")

    assert code == 1
    assert "FAIL servers: server list request failed: HTTP 500" in lines
    assert "FAIL l2:google-calendar: not connected" in lines
    assert "PASS l2:slack: 12 tools" in lines
    assert "PASS l2:notion: 4 tools" in lines
    assert "PASS l2:stripe: 9 tools" in lines


def test_a_failed_catalog_request_stops_the_run() -> None:
    fake = FakeDeployment()
    fake.apps_status = 500

    code, lines = _run(fake)

    assert code == 1
    assert lines[0] == "FAIL catalog: catalog request failed: HTTP 500"
    assert _tools_requests(fake) == []


@pytest.mark.parametrize("value", ["0", "-5", "nan"])
def test_a_tools_timeout_that_is_not_a_positive_number_is_a_usage_error(
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["l2", "--tools-timeout", value],
            env=ENV,
            transport=httpx.MockTransport(FakeDeployment().handler),
            out=io.StringIO(),
        )

    assert exc.value.code == 2


def test_a_malformed_cookie_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        _run(FakeDeployment(), "--cookie", "novalue")

    assert exc.value.code == 2


def test_a_cookie_is_sent_on_every_request() -> None:
    fake = FakeDeployment()

    code, lines = _run(fake, "--cookie", "selected_region=sg")

    assert code == 0, lines
    assert fake.requests
    for request in fake.requests:
        assert "selected_region=sg" in request.headers.get("cookie", "")


def test_a_run_that_checks_nothing_fails() -> None:
    fake = FakeDeployment()
    fake.catalog = [_app("slack", None)]
    fake.servers = []

    code, lines = _run(fake)

    assert code == 1
    assert lines[0] == "FAIL connectors: no connected connector to check"


def test_credential_codes_match_the_servers() -> None:
    from xagent.web.services.mcp_connector_session import _CREDENTIAL_REASONS

    assert l2._CREDENTIAL_CODES == _CREDENTIAL_REASONS


def test_a_login_failure_is_one_failed_check() -> None:
    code, lines = _run(FakeDeployment(), env={"XAGENT_CONNECTOR_SMOKE_PASSWORD": "no"})

    assert code == 1
    assert lines[0] == "FAIL login: login failed: HTTP 401"


def test_output_never_contains_the_token() -> None:
    fake = FakeDeployment()
    fake.tools[7] = (
        400,
        {"detail": {"code": ACCESS_TOKEN, "message": ACCESS_TOKEN}},
    )
    fake.tools[3] = (502, _unavailable(reason=ACCESS_TOKEN))
    fake.tools[5] = (
        502,
        {
            "detail": {
                "code": "mcp_tools_unavailable",
                "failures": [
                    {"server_name": "x", "phase": ACCESS_TOKEN, "attempts": 1}
                ],
            }
        },
    )

    _code, lines = _run(fake)

    assert ACCESS_TOKEN not in "\n".join(lines)
    assert "FAIL l2:slack: failed to start (unknown)" in lines
    assert "FAIL l2:notion: failed to start (unknown)" in lines
    assert "FAIL l2:stripe: failed to start (unknown)" in lines

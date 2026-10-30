"""Tests for the in-container connector startability check (#3264).

Every stdio check here spawns ``sys.executable`` with a fake server or an
inline script, so the venv's ``mcp`` package is on the child's path without
any network access. Remote checks replace the HTTP client factory.
"""

import io
import os
import shlex
import stat
import sys
import textwrap
import time
from types import SimpleNamespace

import anyio
import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web import connector_readiness
from xagent.web.connector_readiness import (
    PLACEHOLDER,
    ConnectorReadiness,
    check_rows,
    format_result,
)
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp

FAKE_SERVER = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("fake")

    {tool}

    if __name__ == "__main__":
        mcp.run(transport="stdio")
    """
)
FAKE_TOOL = textwrap.dedent(
    '''
    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    '''
)
# Prints the child's environment keys that carry the placeholder, plus the
# npm offline switch, then exits before any handshake.
ENV_DUMP = textwrap.dedent(
    """
    import os, sys
    keys = sorted(k for k, v in os.environ.items() if v == {placeholder!r})
    print("placeholders=" + ",".join(keys), file=sys.stderr)
    print("NPM_CONFIG_OFFLINE=" + os.environ.get("NPM_CONFIG_OFFLINE", "unset"), file=sys.stderr)
    print("UV_OFFLINE=" + os.environ.get("UV_OFFLINE", "unset"), file=sys.stderr)
    print("SLACK_ACCESS_TOKEN=" + os.environ.get("SLACK_ACCESS_TOKEN", "unset"), file=sys.stderr)
    sys.exit(1)
    """
).format(placeholder=PLACEHOLDER)


@pytest.fixture(autouse=True)
def sandbox_off(monkeypatch):
    monkeypatch.delenv("SANDBOX_ENABLED", raising=False)
    monkeypatch.delenv("XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS", raising=False)


def _write_server(tmp_path, *, tools=1, name="server.py"):
    path = tmp_path / name
    path.write_text(FAKE_SERVER.format(tool=FAKE_TOOL if tools else ""))
    return str(path)


def _row(app_id, launch_config, *, visible=True):
    return SimpleNamespace(
        app_id=app_id, launch_config=launch_config, is_visible_in_connector=visible
    )


async def _check(row, **kwargs):
    kwargs.setdefault("builtin_app_ids", set())
    kwargs.setdefault("sandbox_enabled", False)
    kwargs.setdefault("timeout", 20)
    (result,) = await check_rows([row], **kwargs)
    return result


# --- stdio shape ------------------------------------------------------------


async def test_python_module_with_tools_passes(tmp_path):
    row = _row(
        "fake",
        {
            "command": sys.executable,
            "args": [_write_server(tmp_path)],
            "env_mapping": {"FAKE_ACCESS_TOKEN": "access_token"},
        },
    )

    result = await _check(row)

    assert (result.status, result.reason) == ("PASS", None)
    assert result.detail == "1 tool"


async def test_server_without_tools_fails_no_tools(tmp_path):
    row = _row(
        "fake", {"command": sys.executable, "args": [_write_server(tmp_path, tools=0)]}
    )

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "no-tools")


async def test_missing_module_fails_missing_with_the_last_stderr_line():
    """A renamed builtin module: ``python -m`` exits before the handshake."""
    row = _row(
        "xero",
        {"command": sys.executable, "args": ["-m", "xagent.web.tools.mcp.renamed"]},
    )

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "missing")
    assert "No module named xagent.web.tools.mcp.renamed" in result.detail
    assert "Traceback" not in result.detail


async def test_import_error_reports_the_exception_line_not_the_header():
    row = _row(
        "fake",
        {"command": sys.executable, "args": ["-c", "import xagent_no_such_module"]},
    )

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "missing")
    assert result.detail.startswith("ModuleNotFoundError: ")


async def test_npm_log_pointer_is_skipped_for_the_error_line():
    script = (
        "import sys; print('npm error code ENOTCACHED', file=sys.stderr); "
        "print('npm error A complete log of this run can be found in: /x.log', "
        "file=sys.stderr); sys.exit(1)"
    )
    row = _row("fake", {"command": sys.executable, "args": ["-c", script]})

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "missing")
    assert result.detail == "npm error code ENOTCACHED"


async def test_unknown_command_fails_spawn():
    row = _row("acme", {"command": "xagent-no-such-command", "args": []})

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "spawn")
    assert result.detail.startswith("FileNotFoundError: ")


async def test_unexecutable_file_fails_spawn(tmp_path):
    """ENOEXEC and the other spawn-time OSErrors class as spawn, not handshake."""
    path = tmp_path / "garbage"
    path.write_bytes(b"\x00\x01not a program")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    row = _row("acme", {"command": str(path), "args": []})

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "spawn")
    assert result.detail.startswith("OSError: ")


async def test_timeout_kills_the_whole_process_group():
    """The SDK's teardown runs uncancelled, so a grandchild of a hung
    connector (npx -> node) does not outlive the check."""
    script = textwrap.dedent(
        """
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print("grandchild=" + str(child.pid), file=sys.stderr, flush=True)
        time.sleep(60)
        """
    )
    row = _row("fake", {"command": sys.executable, "args": ["-c", script]})

    result = await _check(row, timeout=1)

    assert (result.status, result.reason) == ("FAIL", "handshake")
    (line,) = [
        ln for ln in result.stderr_tail.splitlines() if ln.startswith("grandchild=")
    ]
    pid = int(line.split("=", 1)[1])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await anyio.sleep(0.1)
    else:
        os.kill(pid, 9)
        pytest.fail(f"grandchild {pid} survived the timed-out check")


async def test_server_that_never_answers_fails_handshake():
    row = _row(
        "fake",
        {"command": sys.executable, "args": ["-c", "import sys; sys.stdin.read()"]},
    )

    result = await _check(row, timeout=1)

    assert (result.status, result.reason) == ("FAIL", "handshake")
    assert "1s" in result.detail


async def test_placeholders_cover_every_env_mapping_and_required_env_key():
    row = _row(
        "fake",
        {
            "command": sys.executable,
            "args": ["-c", ENV_DUMP],
            "env_mapping": {"A_TOKEN": "access_token", "A_HOST": "instance_url"},
            "required_env": ["B_KEY", "C_KEY"],
        },
    )

    result = await _check(row)

    assert result.reason == "missing"
    # The dump's last line is the uv switch; the check keeps the whole tail
    # so the placeholder line is visible to the assertion below.
    assert "placeholders=A_HOST,A_TOKEN,B_KEY,C_KEY" in result.stderr_tail
    assert "NPM_CONFIG_OFFLINE=unset" in result.stderr_tail


async def test_row_without_command_or_url_fails_spawn():
    row = _row("broken", {"transport": "stdio"})

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "spawn")
    assert "neither" in result.detail


async def test_launch_config_that_is_not_a_mapping_fails_spawn():
    result = await _check(_row("broken", None))

    assert (result.status, result.reason) == ("FAIL", "spawn")


async def test_string_args_are_split_like_the_runtime_does(tmp_path):
    server = _write_server(tmp_path)
    row = _row("fake", {"command": sys.executable, "args": shlex.join([server])})

    result = await _check(row)

    assert (result.status, result.reason) == ("PASS", None)
    assert result.detail == "1 tool"


async def test_jsonrpc_error_on_initialize_fails_handshake():
    script = textwrap.dedent(
        """
        import json, sys
        msg = json.loads(sys.stdin.readline())
        reply = {"jsonrpc": "2.0", "id": msg["id"],
                 "error": {"code": -32600, "message": "nope"}}
        sys.stdout.write(json.dumps(reply) + "\\n")
        sys.stdout.flush()
        sys.stdin.read()
        sys.exit(0)
        """
    )
    row = _row("fake", {"command": sys.executable, "args": ["-c", script]})

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "handshake")


async def test_zero_timeout_leaves_the_handshake_unbounded(tmp_path):
    row = _row("fake", {"command": sys.executable, "args": [_write_server(tmp_path)]})

    result = await _check(row, timeout=0)

    assert (result.status, result.reason) == ("PASS", None)


async def test_parent_env_secret_never_reaches_the_child(monkeypatch):
    monkeypatch.setenv("SLACK_ACCESS_TOKEN", "distinctive-parent-secret")
    row = _row("fake", {"command": sys.executable, "args": ["-c", ENV_DUMP]})

    result = await _check(row)

    assert "SLACK_ACCESS_TOKEN=unset" in result.stderr_tail
    assert "distinctive-parent-secret" not in (result.detail + result.stderr_tail)


# --- npx / uvx shape --------------------------------------------------------


@pytest.mark.parametrize("command", ["npx", "/usr/local/bin/uvx"])
async def test_npx_and_uvx_rows_skip_under_a_sandbox(command):
    row = _row("google-maps", {"command": command, "args": ["-y", "pkg"]})

    result = await _check(row, sandbox_enabled=True)

    assert (result.status, result.reason) == ("SKIP", "sandboxed")


def _fake_npx(tmp_path, name):
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{ENV_DUMP}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


async def test_npx_runs_offline_when_the_sandbox_is_off(tmp_path):
    row = _row("google-maps", {"command": _fake_npx(tmp_path, "npx"), "args": []})

    result = await _check(row, sandbox_enabled=False)

    assert (result.status, result.reason) == ("FAIL", "missing")
    assert "NPM_CONFIG_OFFLINE=true" in result.stderr_tail


async def test_uvx_runs_offline_when_the_sandbox_is_off(tmp_path):
    row = _row("custom-uv", {"command": _fake_npx(tmp_path, "uvx"), "args": []})

    result = await _check(row, sandbox_enabled=False)

    assert result.reason == "missing"
    assert "UV_OFFLINE=1" in result.stderr_tail


def test_sandbox_switch_reads_sandbox_enabled(monkeypatch):
    monkeypatch.delenv("SANDBOX_ENABLED", raising=False)
    assert connector_readiness.sandbox_enabled_from_env() is False
    monkeypatch.setenv("SANDBOX_ENABLED", "TRUE")
    assert connector_readiness.sandbox_enabled_from_env() is True
    monkeypatch.setenv("SANDBOX_ENABLED", "1")
    assert connector_readiness.sandbox_enabled_from_env() is False


# --- remote shape -----------------------------------------------------------


def _mock_client_factory(handler, calls=None):
    def factory(headers=None, timeout=None, auth=None, *, endpoint_url):
        if calls is not None:
            calls.append(
                {"headers": headers, "auth": auth, "endpoint_url": endpoint_url}
            )
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory


async def test_remote_passes_on_any_http_response(monkeypatch):
    seen = []
    calls = []

    def handler(request):
        seen.append((request.method, str(request.url), dict(request.headers)))
        return httpx.Response(405)

    monkeypatch.setattr(
        connector_readiness,
        "create_mcp_http_client",
        _mock_client_factory(handler, calls),
    )
    row = _row(
        "notion", {"url": "https://mcp.example/mcp", "auth": {"type": "mcp_oauth"}}
    )

    result = await _check(row)

    assert (result.status, result.reason) == ("PASS", None)
    assert result.detail == "HTTP 405"
    assert seen == [("GET", "https://mcp.example/mcp", seen[0][2])]
    assert len(calls) == 1
    assert calls[0]["headers"] is None
    assert calls[0]["auth"] is None
    assert calls[0]["endpoint_url"] == "https://mcp.example/mcp"


async def test_remote_connect_failure_is_unreachable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("[Errno -2] Name or service not known")

    monkeypatch.setattr(
        connector_readiness, "create_mcp_http_client", _mock_client_factory(handler)
    )
    row = _row("notion", {"url": "https://mcp.example/mcp"})

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "unreachable")
    assert result.detail.startswith("ConnectError: ")


async def test_remote_timeout_is_unreachable(monkeypatch):
    async def handler(request):
        await anyio.sleep(3)
        return httpx.Response(200)

    monkeypatch.setattr(
        connector_readiness, "create_mcp_http_client", _mock_client_factory(handler)
    )
    row = _row("notion", {"url": "https://mcp.example/mcp"})

    result = await _check(row, timeout=0.5)

    assert (result.status, result.reason) == ("FAIL", "unreachable")
    assert result.detail.startswith("TimeoutError")


async def test_remote_sse_stream_is_not_read(monkeypatch):
    """An SSE endpoint answers GET with an endless body; only the status
    line decides, so the probe returns at once."""

    async def endless():
        while True:
            yield b"data: ping\n\n"
            await anyio.sleep(0.2)

    def handler(request):
        return httpx.Response(200, stream=_AsyncByteStream(endless()))

    monkeypatch.setattr(
        connector_readiness, "create_mcp_http_client", _mock_client_factory(handler)
    )
    row = _row("sse", {"url": "https://mcp.example/sse"})

    started = time.monotonic()
    result = await _check(row, timeout=5)

    assert (result.status, result.reason) == ("PASS", None)
    assert result.detail == "HTTP 200"
    assert time.monotonic() - started < 2


class _AsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        async for chunk in self._chunks:
            yield chunk


async def test_websocket_rows_are_skipped():
    row = _row("odd", {"url": "wss://mcp.example/ws"})

    result = await _check(row)

    assert (result.status, result.reason) == ("SKIP", "websocket")


async def test_remote_rejected_by_the_runtime_transport_is_unreachable():
    """The real factory refuses a non-HTTP URL before any connection."""
    row = _row("odd", {"url": "ftp://mcp.example/mcp"})

    result = await _check(row)

    assert (result.status, result.reason) == ("FAIL", "unreachable")
    assert result.detail.startswith("UnsafePublicHttpTarget: ")


# --- labels, ordering, bounds -------------------------------------------------


async def test_origin_and_hidden_labels(tmp_path):
    server = _write_server(tmp_path)
    rows = [
        _row("slack", {"command": sys.executable, "args": [server]}, visible=False),
        _row("acme", {"command": sys.executable, "args": [server]}),
    ]

    results = await check_rows(
        rows, builtin_app_ids={"slack"}, sandbox_enabled=False, timeout=20
    )

    assert [(r.app_id, r.origin, r.hidden) for r in results] == [
        ("acme", "custom", False),
        ("slack", "builtin", True),
    ]


async def test_exclude_unknown_app_id_raises(tmp_path):
    rows = [_row("acme", {"command": sys.executable, "args": []})]

    with pytest.raises(ValueError, match="acme-typo"):
        await check_rows(
            rows,
            builtin_app_ids=set(),
            sandbox_enabled=False,
            timeout=20,
            exclude={"acme-typo"},
        )


async def test_total_bound_leaves_later_rows_unchecked():
    hang = {"command": sys.executable, "args": ["-c", "import sys; sys.stdin.read()"]}
    rows = [_row("first", hang), _row("second", hang)]

    results = await check_rows(
        rows,
        builtin_app_ids=set(),
        sandbox_enabled=False,
        timeout=5,
        total_timeout=0.5,
    )

    assert [(r.app_id, r.reason) for r in results] == [
        ("first", "handshake"),
        ("second", "handshake"),
    ]
    assert "not checked" in results[1].detail


async def test_sandbox_skip_wins_over_the_total_bound():
    hang = {"command": sys.executable, "args": ["-c", "import sys; sys.stdin.read()"]}
    rows = [_row("a", hang), _row("b", {"command": "npx", "args": ["-y", "pkg"]})]

    results = await check_rows(
        rows,
        builtin_app_ids=set(),
        sandbox_enabled=True,
        timeout=5,
        total_timeout=0.5,
    )

    assert [(r.app_id, r.reason) for r in results] == [
        ("a", "handshake"),
        ("b", "sandboxed"),
    ]


def test_format_result():
    passed = ConnectorReadiness(
        app_id="slack",
        origin="builtin",
        hidden=False,
        status="PASS",
        reason=None,
        detail="14 tools",
    )
    skipped = ConnectorReadiness(
        app_id="google-maps",
        origin="builtin",
        hidden=True,
        status="SKIP",
        reason="sandboxed",
        detail="npx runs in",
    )
    failed = ConnectorReadiness(
        app_id="acme",
        origin="custom",
        hidden=False,
        status="FAIL",
        reason="spawn",
        detail="FileNotFoundError: x",
    )

    assert format_result(passed) == "PASS connector:slack builtin: 14 tools"
    assert (
        format_result(skipped)
        == "SKIP sandboxed connector:google-maps builtin hidden: npx runs in"
    )
    assert (
        format_result(failed)
        == "FAIL spawn connector:acme custom: FileNotFoundError: x"
    )


# --- main -------------------------------------------------------------------


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "xagent.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine, tables=[PublicMCPApp.__table__])
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _add_app(db, app_id, launch_config, *, visible=True):
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=app_id,
            transport="stdio",
            is_visible_in_connector=visible,
            launch_config=launch_config,
        )
    )


def _run(argv=()):
    out = io.StringIO()
    code = connector_readiness.main(list(argv), out=out)
    return code, out.getvalue()


def test_main_fails_only_the_renamed_module(database, tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_ACCESS_TOKEN", "distinctive-parent-secret")
    server = _write_server(tmp_path)
    _add_app(database, "acme", {"command": sys.executable, "args": [server]})
    _add_app(
        database,
        "xero",
        {"command": sys.executable, "args": ["-m", "xagent.web.tools.mcp.renamed"]},
    )
    database.commit()

    code, output = _run()

    lines = output.splitlines()
    assert code == 1
    assert lines[0] == "PASS connector:acme custom: 1 tool"
    assert lines[1].startswith("FAIL missing connector:xero builtin: ")
    assert lines[-1] == "2 checks, 1 failed, 0 skipped"
    assert os.environ["PATH"] not in output
    assert "distinctive-parent-secret" not in output


def test_main_skips_npx_rows_under_a_sandbox(database, monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    _add_app(database, "google-maps", {"command": "npx", "args": ["-y", "pkg"]})
    _add_app(
        database,
        "chrome-devtools",
        {"command": "npx", "args": ["-y", "chrome-devtools-mcp@1.6.0"]},
        visible=False,
    )
    database.commit()

    code, output = _run()

    assert code == 0
    assert output.splitlines() == [
        "SKIP sandboxed connector:chrome-devtools builtin hidden: "
        "tasks run npx/uvx rows in the sandbox image, not this one",
        "SKIP sandboxed connector:google-maps builtin: "
        "tasks run npx/uvx rows in the sandbox image, not this one",
        "2 checks, 0 failed, 2 skipped",
    ]


def test_main_custom_row_with_a_bogus_command_continues(database, tmp_path):
    _add_app(database, "acme", {"command": "xagent-no-such-command", "args": []})
    _add_app(
        database, "zeta", {"command": sys.executable, "args": [_write_server(tmp_path)]}
    )
    database.commit()

    code, output = _run()

    lines = output.splitlines()
    assert code == 1
    assert lines[0].startswith("FAIL spawn connector:acme custom: FileNotFoundError")
    assert lines[1] == "PASS connector:zeta custom: 1 tool"


def test_main_excludes_rows(database, tmp_path):
    _add_app(database, "acme", {"command": "xagent-no-such-command", "args": []})
    _add_app(
        database, "zeta", {"command": sys.executable, "args": [_write_server(tmp_path)]}
    )
    database.commit()

    code, output = _run(["--exclude", "acme"])

    assert code == 0
    assert "acme" not in output
    assert output.splitlines()[-1] == "1 checks, 0 failed, 0 skipped"


def test_main_rejects_an_unknown_exclusion(database):
    with pytest.raises(SystemExit) as exc:
        _run(["--exclude", "acme"])
    assert exc.value.code == 2


def test_main_reports_a_database_failure_by_class_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'missing.db'}")

    code, output = _run()

    assert code == 1
    assert output.splitlines() == ["FAIL database: OperationalError"]


def test_main_rejects_invalid_mcp_policy_before_checking_rows(monkeypatch):
    monkeypatch.setenv("XAGENT_MCP_PRIVATE_ENDPOINTS", "http://mcp-tools:8000/mcp")
    monkeypatch.setattr(
        connector_readiness,
        "_load_app_rows",
        lambda: pytest.fail("rows loaded before policy validation"),
    )

    code, output = _run()

    assert code == 1
    assert output.splitlines() == [
        "FAIL configuration: Invalid deployment MCP network policy "
        "configuration: XAGENT_MCP_PRIVATE_ENDPOINTS"
    ]


def test_main_total_timeout_flag(database, monkeypatch):
    monkeypatch.setenv("XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS", "5")
    hang = {"command": sys.executable, "args": ["-c", "import sys; sys.stdin.read()"]}
    _add_app(database, "first", hang)
    _add_app(database, "second", hang)
    database.commit()

    code, output = _run(["--total-timeout", "0.5"])

    lines = output.splitlines()
    assert code == 1
    assert lines[0].startswith("FAIL handshake connector:first custom: ")
    assert lines[1].startswith("FAIL handshake connector:second custom: not checked")


@pytest.mark.parametrize("value", ["-1", "nan", "x"])
def test_main_rejects_a_bad_total_timeout(database, value):
    with pytest.raises(SystemExit) as exc:
        _run(["--total-timeout", value])
    assert exc.value.code == 2


def test_main_default_total_timeout_covers_the_checked_rows(database, monkeypatch):
    """Default total bound is the per-connector timeout times the rows left
    after --exclude; XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS=0 leaves it unset."""
    for app_id in ("a", "b", "c"):
        _add_app(database, app_id, {"command": sys.executable, "args": []})
    database.commit()
    captured = []

    async def fake_check_rows(rows, **kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(connector_readiness, "check_rows", fake_check_rows)

    monkeypatch.setenv("XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS", "7")
    _run()
    _run(["--exclude", "a"])
    _run(["--total-timeout", "0"])
    monkeypatch.setenv("XAGENT_MCP_TOOL_INIT_TIMEOUT_SECONDS", "0")
    _run()

    assert [(c["timeout"], c["total_timeout"]) for c in captured] == [
        (7.0, 21.0),
        (7.0, 14.0),
        (7.0, 0.0),
        (0.0, None),
    ]


def test_main_fails_an_empty_catalog(database):
    code, output = _run()

    assert code == 1
    assert output.splitlines() == ["FAIL catalog: no public_mcp_apps rows to check"]

"""An in-repo connector's ``{"status": "error"}`` result is a failed call.

The in-repo FastMCP servers under ``xagent.web.tools.mcp`` catch vendor
errors and return ``{"status": "error", ...}`` as text with the MCP ``isError``
flag unset (xorbitsai/xagent#3574). The adapter recognizes that envelope only
for a connection that launches one of those modules, and marks the result
``success: False`` so ``tool_result_succeeded`` reports the call as failed,
but only for a tool whose wire annotations declare that a repeat adds no
effect (read-only, or idempotent). A write may return the envelope after its
request went out, so an undeclared or additive write stays a success.
``is_error`` keeps mirroring the wire flag.
"""

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations

import xagent.web.tools.mcp as in_repo_servers
from xagent.core.agent.pattern.react.duplicate_write_guard import (
    tool_requires_duplicate_write_guard,
)
from xagent.core.agent.result import tool_result_succeeded
from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe import mcp_error_envelope as envelope_module
from xagent.core.tools.adapters.vibe.mcp_adapter import MCPToolAdapter
from xagent.core.tools.adapters.vibe.mcp_error_envelope import (
    ERROR_ENVELOPE_MAX_CHARS,
    ERROR_ENVELOPE_SERVER_MODULES,
    IN_REPO_SERVER_PACKAGE,
    is_error_envelope,
    speaks_error_envelope,
)

_ERROR = json.dumps({"status": "error", "message": "Vendor rejected the token"})


def _in_repo(module: str = "gmail") -> dict:
    return {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", f"{IN_REPO_SERVER_PACKAGE}.{module}"],
    }


def _text(text: str) -> list[dict]:
    return [TextContent(type="text", text=text).model_dump()]


# --- Which connections speak the envelope ------------------------------------


@pytest.mark.parametrize(
    "connection",
    [
        _in_repo("gmail"),
        _in_repo("google_drive"),
        _in_repo("slack"),
        # Rows seeded before the launch shape was shortened run through uv.
        {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "python", "-m", "xagent.web.tools.mcp.gmail"],
        },
        {
            "transport": "stdio",
            "command": "/usr/local/bin/python3",
            "args": ("-m", "xagent.web.tools.mcp.facebook"),
        },
    ],
)
def test_in_repo_fastmcp_launch_speaks_the_envelope(connection):
    assert speaks_error_envelope(connection) is True


@pytest.mark.parametrize(
    "connection",
    [
        # LinkedIn's server passes the vendor's response text through as is.
        _in_repo("linkedin"),
        # A helper module, not a server.
        _in_repo("utils"),
        # The package itself and look-alike module paths.
        {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp"],
        },
        _in_repo("gmail.extra"),
        _in_repo("gmail_evil"),
        {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "evil.xagent.web.tools.mcp.gmail"],
        },
        # Python runs the first -m; anything after it is that module's argv.
        {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "other_server", "-m", "xagent.web.tools.mcp.gmail"],
        },
        # A -m with no module after it.
        {"transport": "stdio", "command": "python", "args": ["-m"]},
        # The module as a plain argument, not run with -m.
        {
            "transport": "stdio",
            "command": "python",
            "args": ["xagent.web.tools.mcp.gmail"],
        },
        # HubSpot's catalog entry is the vendor's remote server, whatever the
        # repo still carries under the same name.
        {"transport": "streamable_http", "url": "https://mcp.hubspot.com"},
        {"transport": "stdio", "command": "npx", "args": ["-y", "some-mcp-server"]},
        # A remote transport never runs a local module, whatever its args say.
        {
            "transport": "sse",
            "url": "https://example.com/sse",
            "args": ["-m", "xagent.web.tools.mcp.gmail"],
        },
        {
            "transport": "stdio",
            "command": "python",
            "args": "-m xagent.web.tools.mcp.gmail",
        },
        {"transport": "stdio", "command": "python"},
        None,
        "stdio",
    ],
)
def test_other_connections_do_not_speak_the_envelope(connection):
    assert speaks_error_envelope(connection) is False


def _fastmcp_server_modules() -> set[str]:
    package_dir = Path(in_repo_servers.__file__).parent
    return {
        path.stem
        for path in package_dir.glob("*.py")
        if re.search(r"\bFastMCP\(", path.read_text(encoding="utf-8"))
    }


def test_envelope_modules_are_exactly_the_in_repo_fastmcp_servers():
    """Every FastMCP server in the package reports a vendor error as the
    envelope (directly, or through ``meta_graph.error_response``), or raises
    and so sets ``isError``. A new server must be classified here."""
    assert ERROR_ENVELOPE_SERVER_MODULES == _fastmcp_server_modules()


def test_every_catalog_launched_module_is_classified():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    passes_vendor_text_through = {"linkedin"}
    launched = set()
    for row in get_builtin_public_mcp_app_rows():
        args = (row.get("launch_config") or {}).get("args") or []
        for arg in args:
            if isinstance(arg, str) and arg.startswith(f"{IN_REPO_SERVER_PACKAGE}."):
                launched.add(arg.removeprefix(f"{IN_REPO_SERVER_PACKAGE}."))

    assert launched
    assert launched <= ERROR_ENVELOPE_SERVER_MODULES | passes_vendor_text_through
    assert not ERROR_ENVELOPE_SERVER_MODULES & passes_vendor_text_through


# --- What counts as the envelope ---------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        _text(_ERROR),
        _text(json.dumps({"status": "error"})),
        _text("  \n" + _ERROR + "\n"),
        _text(json.dumps({"status": "error", "message": "x", "failure_code": "c"})),
        _text(json.dumps({"message": "first", "status": "error"}, ensure_ascii=False)),
    ],
)
def test_error_envelope_is_recognized(content):
    assert is_error_envelope(content) is True


@pytest.mark.parametrize(
    "content",
    [
        _text(json.dumps({"status": "success", "messages": []})),
        # A write that may have happened is not a failure.
        _text(json.dumps({"status": "indeterminate", "message": "unknown"})),
        _text(json.dumps({"status": "conflict", "conflicts": []})),
        _text(json.dumps({"status": "partial", "results": []})),
        _text(json.dumps({"status": "ERROR"})),
        _text(json.dumps({"status": "error "})),
        _text(json.dumps({"status": 500, "message": "error"})),
        _text(json.dumps({"result": {"status": "error"}})),
        _text(json.dumps([{"status": "error"}])),
        _text(json.dumps("error")),
        _text("Error listing invoices: boom"),
        _text('{"status": "error"'),
        _text(""),
        # Two items are a listing, never a lone envelope.
        _text(_ERROR) + _text(_ERROR),
        [ImageContent(type="image", data="AAAA", mimeType="image/png").model_dump()],
        [{"type": "text", "text": None}],
        [{"text": _ERROR}],
        ["not a dict"],
        [],
        None,
        _ERROR,
    ],
)
def test_other_content_is_not_the_envelope(content):
    assert is_error_envelope(content) is False


def test_envelope_over_the_size_bound_is_not_parsed():
    padded = json.dumps({"status": "error", "message": "x" * ERROR_ENVELOPE_MAX_CHARS})
    assert len(padded) > ERROR_ENVELOPE_MAX_CHARS

    assert is_error_envelope(_text(padded)) is False


def test_parser_recursion_error_is_not_the_envelope(monkeypatch):
    """JSON nested past the parser's depth limit raises RecursionError. Within
    the size bound CPython 3.12 does not reach that depth, so the error is
    injected: it must read as "not the envelope", never escape the call."""

    def _too_deep(_text):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(envelope_module, "json", SimpleNamespace(loads=_too_deep))

    assert is_error_envelope(_text(_ERROR)) is False


# --- Through the adapter, against a real FastMCP server ----------------------


_READ_ONLY = {"readOnlyHint": True}


def _server_returning(payload: str, annotations: dict | None = None) -> FastMCP:
    server = FastMCP("envelope-probe")

    @server.tool(
        annotations=ToolAnnotations(**annotations) if annotations is not None else None
    )
    def probe_tool() -> str:
        return payload

    return server


async def _adapter(monkeypatch, server: FastMCP, connection, **kwargs):
    @asynccontextmanager
    async def _session(_connection):
        async with create_connected_server_and_client_session(
            server._mcp_server
        ) as session:
            yield session

    monkeypatch.setattr(mcp_adapter_module, "create_session", _session)
    (mcp_tool,) = await server.list_tools()
    return MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection=connection,
        name_prefix="mcp_probe_",
        source_server=kwargs.pop("source_server", "probe"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_in_repo_error_envelope_is_a_failed_tool_call(monkeypatch):
    """The issue's regression: a server returns the envelope with ``isError``
    unset, and the call the model made counts as failed."""
    adapter = await _adapter(
        monkeypatch,
        _server_returning(_ERROR, _READ_ONLY),
        _in_repo(),
        raw_annotations=_READ_ONLY,
    )

    result = await adapter.run_json_async({})

    assert tool_result_succeeded(result) is False
    assert result["success"] is False
    # The wire flag is reported as the server sent it, and the model still
    # reads the server's own message.
    assert result["is_error"] is False
    assert result["content"][0]["text"] == _ERROR


@pytest.mark.asyncio
async def test_error_envelope_from_a_server_not_in_repo_stays_successful(monkeypatch):
    remote = {"transport": "streamable_http", "url": "https://mcp.example.com/mcp"}
    adapter = await _adapter(
        monkeypatch,
        _server_returning(_ERROR, _READ_ONLY),
        remote,
        raw_annotations=_READ_ONLY,
    )

    result = await adapter.run_json_async({})

    assert "success" not in result
    assert tool_result_succeeded(result) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        json.dumps({"status": "success", "messages": []}),
        json.dumps({"status": "indeterminate", "message": "may have been sent"}),
        "plain text result",
    ],
)
async def test_other_in_repo_results_are_unchanged(monkeypatch, payload):
    adapter = await _adapter(
        monkeypatch,
        _server_returning(payload, _READ_ONLY),
        _in_repo(),
        raw_annotations=_READ_ONLY,
    )

    result = await adapter.run_json_async({})

    assert "success" not in result
    assert tool_result_succeeded(result) is True


@pytest.mark.asyncio
async def test_server_flagged_error_payload_is_unchanged(monkeypatch):
    server = FastMCP("envelope-probe")

    @server.tool(annotations=ToolAnnotations(**_READ_ONLY))
    def probe_tool() -> str:
        raise RuntimeError("boom")

    adapter = await _adapter(
        monkeypatch, server, _in_repo(), raw_annotations=_READ_ONLY
    )

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert "success" not in result
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
async def test_excel_failure_code_keeps_its_classification(monkeypatch):
    payload = json.dumps(
        {
            "status": "error",
            "message": "locked",
            "failure_code": "excel_workbook_locked",
        }
    )
    server = FastMCP("envelope-probe")

    @server.tool()
    def excel_probe_tool() -> str:
        return payload

    adapter = await _adapter(
        monkeypatch, server, _in_repo("excel"), source_server="excel"
    )

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert result["success"] is False
    assert result["failure_code"] == "excel_workbook_locked"


@pytest.mark.asyncio
async def test_non_idempotent_write_envelope_keeps_the_duplicate_write_guard(
    monkeypatch,
):
    """An in-repo write can return the envelope after its request went out
    (``deputy_create_resource`` does on a timeout after the POST). The
    same-turn duplicate-write guard suppresses an identical repeat only of a
    ``completed`` call, and a ``failed`` one reads as "provably did not
    happen", so a non-idempotent write's envelope must stay a success."""
    annotations = {"destructiveHint": False, "idempotentHint": False}
    adapter = await _adapter(
        monkeypatch,
        _server_returning(_ERROR, annotations),
        _in_repo("deputy"),
        raw_annotations=annotations,
    )
    assert tool_requires_duplicate_write_guard(adapter) is True

    result = await adapter.run_json_async({})

    assert "success" not in result
    assert tool_result_succeeded(result) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "annotations",
    [
        {"readOnlyHint": True},
        {"destructiveHint": False, "idempotentHint": True},
        {"destructiveHint": True, "idempotentHint": True},
    ],
)
async def test_envelope_of_a_tool_whose_repeat_adds_no_effect_is_a_failure(
    monkeypatch, annotations
):
    adapter = await _adapter(
        monkeypatch,
        _server_returning(_ERROR, annotations),
        _in_repo("deputy"),
        raw_annotations=annotations,
    )
    assert tool_requires_duplicate_write_guard(adapter) is False

    result = await adapter.run_json_async({})

    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "annotations",
    [
        None,
        {"destructiveHint": False},
        # Destructive outranks read-only.
        {"readOnlyHint": True, "destructiveHint": True},
        # A non-boolean is not a declaration.
        {"idempotentHint": 1},
    ],
)
async def test_envelope_of_an_undeclared_or_additive_write_stays_successful(
    monkeypatch, annotations
):
    """The Freshdesk create/reply case: ``freshdesk_create_ticket`` is a plain
    ``@server.tool()`` that returns the envelope after a POST that may have
    committed. A failed result would reopen normal tools in flash mode, and the
    duplicate-write guard does not cover an undeclared tool, so the model could
    write twice. A tool that does not declare a no-effect repeat is a write."""
    adapter = await _adapter(
        monkeypatch,
        _server_returning(_ERROR, annotations),
        _in_repo("freshdesk"),
        raw_annotations=annotations,
    )

    result = await adapter.run_json_async({})

    assert "success" not in result
    assert tool_result_succeeded(result) is True


@pytest.mark.asyncio
async def test_envelope_among_several_content_items_stays_successful(monkeypatch):
    server = FastMCP("envelope-probe")

    @server.tool(annotations=ToolAnnotations(**_READ_ONLY))
    def probe_tool() -> CallToolResult:
        return CallToolResult(
            content=[
                TextContent(type="text", text=_ERROR),
                TextContent(type="text", text=_ERROR),
            ]
        )

    adapter = await _adapter(
        monkeypatch, server, _in_repo(), raw_annotations=_READ_ONLY
    )

    result = await adapter.run_json_async({})

    assert len(result["content"]) == 2
    assert "success" not in result
    assert tool_result_succeeded(result) is True


def test_envelope_detection_is_decided_from_the_bound_connection():
    def adapter(connection, raw_annotations=None) -> MCPToolAdapter:
        return MCPToolAdapter(
            mcp_tool=SimpleNamespace(
                name="probe_tool",
                description="probe",
                inputSchema={"type": "object", "properties": {}},
            ),
            connection=connection,
            name_prefix="mcp_probe_",
            source_server="probe",
            raw_annotations=raw_annotations,
        )

    assert adapter(_in_repo("gmail"), _READ_ONLY)._detect_error_envelope is True
    assert adapter(_in_repo("gmail"))._detect_error_envelope is False
    assert adapter(_in_repo("linkedin"), _READ_ONLY)._detect_error_envelope is False
    assert (
        adapter(
            {"transport": "streamable_http", "url": "https://x.example"},
            _READ_ONLY,
        )._detect_error_envelope
        is False
    )

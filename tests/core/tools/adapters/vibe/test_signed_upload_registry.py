from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe import signed_upload
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    _build_mcp_tool_adapter,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.signed_upload import (
    SignedUploadBinding,
    find_signed_upload_binding,
    is_signed_upload_issuing_tool,
    register_signed_upload_binding,
)

CONNECTION_HOST = "mcp.example.test"


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signed_upload, "_BINDINGS", {})


def _binding(**overrides: Any) -> SignedUploadBinding:
    binding = SignedUploadBinding(
        connection_host=CONNECTION_HOST,
        tool_name="attach_document",
        file_argument="document",
        issuing_tool_name="request_upload",
        allowed_upload_hosts=frozenset({"uploads.example.test"}),
        upload_path_prefix="/staged/",
        content_types={".pdf": "application/pdf", ".png": "image/png"},
    )
    return replace(binding, **overrides) if overrides else binding


def _connection(transport: str = "streamable_http", **extra: Any) -> dict[str, Any]:
    return {
        "transport": transport,
        "url": f"https://{CONNECTION_HOST}/mcp",
        **extra,
    }


def _stdio_connection() -> dict[str, Any]:
    # Carries the registered host in a ``url`` key so only the transport check
    # can keep it from matching a binding.
    return {
        "transport": "stdio",
        "command": "python",
        "args": [],
        "url": f"https://{CONNECTION_HOST}/mcp",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"connection_host": "*.example.test"}, id="host-star"),
        pytest.param({"connection_host": "mcp.example.test:8443"}, id="host-colon"),
        pytest.param({"connection_host": "user@mcp.example.test"}, id="host-at"),
        pytest.param({"connection_host": "mcp.example.test/x"}, id="host-slash"),
        pytest.param({"connection_host": "mcp example.test"}, id="host-space"),
        pytest.param({"connection_host": "mcp.example.test\t"}, id="host-tab"),
        pytest.param({"connection_host": "MCP.example.test"}, id="host-uppercase"),
        pytest.param({"connection_host": ""}, id="host-empty"),
        pytest.param({"connection_host": "mcp.example.test."}, id="host-trailing-dot"),
        pytest.param({"connection_host": "10.0.0.1"}, id="host-ip"),
        pytest.param({"allowed_upload_hosts": frozenset()}, id="upload-hosts-empty"),
        pytest.param(
            {"allowed_upload_hosts": frozenset({"*.example.test"})},
            id="upload-host-star",
        ),
        pytest.param(
            {"allowed_upload_hosts": frozenset({"192.168.1.5"})}, id="upload-host-ip"
        ),
        pytest.param(
            {"allowed_upload_hosts": frozenset({"Uploads.example.test"})},
            id="upload-host-uppercase",
        ),
        pytest.param(
            {"allowed_upload_hosts": frozenset({"uploads.example.test."})},
            id="upload-host-trailing-dot",
        ),
        pytest.param({"upload_path_prefix": "staged/"}, id="prefix-no-leading-slash"),
        pytest.param({"upload_path_prefix": "/staged"}, id="prefix-no-trailing-slash"),
        pytest.param({"upload_path_prefix": "/staged/../x/"}, id="prefix-dotdot"),
        pytest.param({"content_types": {"pdf": "application/pdf"}}, id="suffix-no-dot"),
        pytest.param(
            {"content_types": {".PDF": "application/pdf"}}, id="suffix-uppercase"
        ),
        pytest.param({"content_types": {".pdf": ""}}, id="mime-empty"),
        pytest.param(
            {"allowed_upload_hosts": "uploads.example.test"},
            id="upload-hosts-is-a-string",
        ),
        pytest.param(
            {"allowed_upload_hosts": ["uploads.example.test"]},
            id="upload-hosts-is-a-list",
        ),
        pytest.param({"content_types": None}, id="content-types-none"),
        pytest.param(
            {"content_types": [(".pdf", "application/pdf")]},
            id="content-types-not-a-mapping",
        ),
        pytest.param({"content_types": {}}, id="content-types-empty"),
        pytest.param(
            {"issuing_tool_name": "attach_document"}, id="issuing-tool-is-the-tool"
        ),
        pytest.param({"tool_name": ""}, id="tool-name-empty"),
        pytest.param({"file_argument": ""}, id="file-argument-empty"),
        pytest.param({"issuing_tool_name": ""}, id="issuing-tool-empty"),
    ],
)
def test_register_signed_upload_binding_validation(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        register_signed_upload_binding(_binding(**overrides))

    assert signed_upload._BINDINGS == {}


def test_register_signed_upload_binding_rejects_duplicate_host_and_tool() -> None:
    register_signed_upload_binding(_binding())

    with pytest.raises(ValueError):
        register_signed_upload_binding(_binding(issuing_tool_name="other_issuer"))

    assert len(signed_upload._BINDINGS) == 1
    assert (
        signed_upload._BINDINGS[(CONNECTION_HOST, "attach_document")].issuing_tool_name
        == "request_upload"
    )


def test_register_signed_upload_binding_rejects_tool_used_as_both_roles() -> None:
    register_signed_upload_binding(_binding())

    # Issuing tool of the new binding is the replaced tool of an existing one.
    with pytest.raises(ValueError):
        register_signed_upload_binding(
            _binding(tool_name="attach_image", issuing_tool_name="attach_document")
        )
    # Replaced tool of the new binding is the issuing tool of an existing one.
    with pytest.raises(ValueError):
        register_signed_upload_binding(
            _binding(tool_name="request_upload", issuing_tool_name="issue_other")
        )

    assert len(signed_upload._BINDINGS) == 1


def test_register_signed_upload_binding_allows_shared_issuing_tool() -> None:
    register_signed_upload_binding(_binding())
    other = _binding(tool_name="attach_image")

    register_signed_upload_binding(other)

    assert find_signed_upload_binding(_connection(), "attach_image") is other


def test_register_signed_upload_binding_role_clash_is_per_connection_host() -> None:
    register_signed_upload_binding(_binding())

    other = _binding(
        connection_host="other.example.test",
        tool_name="request_upload",
        issuing_tool_name="issue_other",
    )
    register_signed_upload_binding(other)

    assert (
        find_signed_upload_binding(
            {"transport": "streamable_http", "url": "https://other.example.test/mcp"},
            "request_upload",
        )
        is other
    )


def test_register_signed_upload_binding_accepts_valid_bindings() -> None:
    first = _binding()
    second = _binding(tool_name="attach_image")
    register_signed_upload_binding(first)
    register_signed_upload_binding(second)

    assert find_signed_upload_binding(_connection(), "attach_document") is first
    assert find_signed_upload_binding(_connection(), "attach_image") is second


@pytest.mark.parametrize(
    ("connection", "tool_name", "expect_match"),
    [
        pytest.param(
            _connection("streamable_http"), "attach_document", True, id="http"
        ),
        pytest.param(_connection("sse"), "attach_document", True, id="sse"),
        pytest.param(
            _connection("websocket"), "attach_document", False, id="websocket"
        ),
        pytest.param(
            _stdio_connection(),
            "attach_document",
            False,
            id="stdio",
        ),
        pytest.param(
            {
                "transport": "streamable_http",
                "url": "https://other.example.test/mcp",
            },
            "attach_document",
            False,
            id="other-host",
        ),
        pytest.param(
            {"transport": "streamable_http"}, "attach_document", False, id="no-url"
        ),
        pytest.param(
            {"transport": "streamable_http", "url": 7},
            "attach_document",
            False,
            id="url-not-a-string",
        ),
        pytest.param(
            {"transport": "streamable_http", "url": "https://[bad/mcp"},
            "attach_document",
            False,
            id="unparseable-url",
        ),
        pytest.param(
            [("transport", "streamable_http")],
            "attach_document",
            False,
            id="not-a-mapping",
        ),
        pytest.param(
            {
                "transport": "streamable_http",
                "url": "https://MCP.Example.TEST/mcp",
            },
            "attach_document",
            True,
            id="uppercase-url-host",
        ),
        pytest.param(_connection(), "other_tool", False, id="other-tool"),
    ],
)
def test_find_signed_upload_binding_matches_only_remote_host(
    connection: Any, tool_name: str, expect_match: bool
) -> None:
    binding = _binding()
    register_signed_upload_binding(binding)

    found = find_signed_upload_binding(connection, tool_name)

    assert found is (binding if expect_match else None)


@pytest.mark.parametrize(
    ("connection", "tool_name", "expected"),
    [
        pytest.param(_connection(), "request_upload", True, id="issuing-tool"),
        pytest.param(_connection(), "attach_document", False, id="business-tool"),
        pytest.param(
            {
                "transport": "streamable_http",
                "url": "https://other.example.test/mcp",
            },
            "request_upload",
            False,
            id="other-host",
        ),
        pytest.param(
            _stdio_connection(),
            "request_upload",
            False,
            id="stdio",
        ),
    ],
)
def test_is_signed_upload_issuing_tool(
    connection: Any, tool_name: str, expected: bool
) -> None:
    register_signed_upload_binding(_binding())

    assert is_signed_upload_issuing_tool(connection, tool_name) is expected


def test_empty_registry_matches_nothing() -> None:
    assert find_signed_upload_binding(_connection(), "attach_document") is None
    assert is_signed_upload_issuing_tool(_connection(), "request_upload") is False


def _server_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} tool",
        inputSchema={
            "type": "object",
            "properties": {"document": {"type": "string"}},
        },
    )


def _serve_tools(monkeypatch: pytest.MonkeyPatch, *tool_names: str) -> None:
    class FakeSession:
        async def initialize(self) -> None:
            return None

    @asynccontextmanager
    async def fake_create_session(_connection: Any):
        yield FakeSession()

    async def fake_load_mcp_tools(_session: Any) -> list[SimpleNamespace]:
        return [_server_tool(name) for name in tool_names]

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools", fake_load_mcp_tools)


def _describe(tool: Any) -> tuple[str, str, dict[str, Any]]:
    return (tool.name, tool.description, tool.args_type().model_json_schema())


_ALL_TOOLS = ("attach_document", "request_upload", "other")


@pytest.mark.asyncio
async def test_signed_upload_issuing_tool_is_withheld_from_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_tools(monkeypatch, *_ALL_TOOLS)
    remote = _connection()
    expected_all = [
        _describe(_build_mcp_tool_adapter("docs", remote, _server_tool(name)))
        for name in _ALL_TOOLS
    ]

    # No registration: every tool is loaded, identical to a direct build.
    unregistered = await load_mcp_tools_as_agent_tools({"docs": remote})
    assert [_describe(tool) for tool in unregistered.tools] == expected_all
    assert unregistered.failures == ()

    register_signed_upload_binding(_binding())

    # Registered on the same host: the issuing tool is withheld.
    registered = await load_mcp_tools_as_agent_tools({"docs": remote})
    assert [_describe(tool) for tool in registered.tools] == [
        expected_all[0],
        expected_all[2],
    ]
    assert registered.failures == ()
    assert registered.loaded_servers == ("docs",)

    # Registered for another host: nothing is withheld.
    other_host = {
        "transport": "streamable_http",
        "url": "https://other.example.test/mcp",
    }
    elsewhere = await load_mcp_tools_as_agent_tools({"docs": other_host})
    assert [tool.name for tool in elsewhere.tools] == [
        "mcp_docs_attach_document",
        "mcp_docs_request_upload",
        "mcp_docs_other",
    ]

    # A stdio connection never matches a binding.
    stdio = _stdio_connection()
    local = await load_mcp_tools_as_agent_tools({"docs": stdio}, sandbox=None)
    assert [tool.name for tool in local.tools] == [
        "mcp_docs_attach_document",
        "mcp_docs_request_upload",
        "mcp_docs_other",
    ]


@pytest.mark.asyncio
async def test_server_with_only_the_issuing_tool_loads_no_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_tools(monkeypatch, "request_upload")
    register_signed_upload_binding(_binding())

    result = await load_mcp_tools_as_agent_tools({"docs": _connection()})

    assert result.tools == ()
    assert result.failures == ()
    assert result.loaded_servers == ()

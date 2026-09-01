from __future__ import annotations

import logging
import re
from typing import Any

import pytest
from mcp.types import Tool as MCPTool

from xagent.core.agent.runtime import PatternRuntime
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    MCPServerLoadFailure,
    MCPToolAdapter,
    UnavailableMCPTool,
)
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec


def _unavailable_tool(
    *,
    server_name: str = "Google Drive",
    server_id: int | None = 42,
    allow_users: list[str] | None = None,
    failure_code: str | None = None,
    reason: str | None = None,
    message: str | None = None,
    app_name: str | None = None,
) -> UnavailableMCPTool:
    kwargs: dict[str, Any] = {
        "server_name": server_name,
        "server_id": server_id,
        "allow_users": allow_users,
        "failure_code": failure_code,
    }
    if reason is not None:
        kwargs["reason"] = reason
    if message is not None:
        kwargs["message"] = message
    if app_name is not None:
        kwargs["app_name"] = app_name
    return UnavailableMCPTool(
        **kwargs,
    )


def _unavailable_config(
    *,
    name: str | None = "Google Drive",
    server_id: int | None = 42,
    allow_users: list[str] | None = None,
    failure_code: object | None = None,
    app_name: str | None = None,
) -> dict:
    config = {
        "name": name,
        "transport": "unavailable",
        "description": f"{name} server",
        "config": {
            "unavailable": True,
            "reason": "oauth_token_resolver_failed",
            "message": "MCP server credentials are unavailable.",
            "server_id": server_id,
        },
        "allow_users": allow_users,
    }
    if failure_code is not None:
        config["config"]["failure_code"] = failure_code
    if app_name is not None:
        config["config"]["app_name"] = app_name
    return config


def test_unavailable_tool_name_is_llm_safe_and_uses_server_id():
    tool = _unavailable_tool(server_name="Google Drive", server_id=42)

    assert tool.name == "mcp_google_drive_42_unavailable"
    assert re.fullmatch(r"[A-Za-z0-9_]+", tool.name)


def test_unavailable_tool_names_avoid_flat_collisions_with_server_id():
    first = _unavailable_tool(server_name="Google Drive", server_id=1)
    second = _unavailable_tool(server_name="google-drive", server_id=2)

    assert first.name == "mcp_google_drive_1_unavailable"
    assert second.name == "mcp_google_drive_2_unavailable"
    assert first.name != second.name


def test_unavailable_tool_empty_server_name_has_stable_fallback():
    tool = _unavailable_tool(server_name="!!!", server_id=None)

    assert tool.name == "mcp_server_unavailable"
    assert re.fullmatch(r"[A-Za-z0-9_]+", tool.name)


def test_unavailable_tool_metadata_is_mcp_and_server_scoped():
    tool = _unavailable_tool(allow_users=["7"])

    assert tool.metadata.category == ToolCategory.MCP
    assert tool.metadata.source_server == "google_drive"
    assert tool.metadata.allow_users == ["7"]
    assert tool.metadata.read_only is True
    assert tool.metadata.concurrency_safe is True


@pytest.mark.asyncio
async def test_unavailable_tool_async_authorized_returns_clean_error(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    tool = _unavailable_tool(allow_users=["7"])

    result = await tool.run_json_async({})

    assert result["is_error"] is True
    text = result["content"][0]["text"]
    assert "MCP server credentials are unavailable" in text
    assert "resolver" not in text
    assert "provider" not in text
    assert "RuntimeError" not in text


@pytest.mark.asyncio
async def test_unavailable_tool_authorized_returns_classified_failure(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    tool = _unavailable_tool(allow_users=["7"], failure_code="oauth_token_required")

    result = await tool.run_json_async({})

    assert result == {
        "success": False,
        "status": "error",
        "is_error": True,
        "error": "MCP server credentials are unavailable.",
        "failure_code": "oauth_token_required",
        "content": [
            {
                "text": (
                    "MCP server credentials are unavailable. Please reconnect "
                    "the MCP server credentials and retry."
                )
            }
        ],
    }


@pytest.mark.asyncio
async def test_unavailable_tool_oauth_required_with_app_name_pauses_for_connect_apps(
    monkeypatch,
):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    tool = _unavailable_tool(
        allow_users=["7"],
        failure_code="oauth_token_required",
        app_name="Gmail",
    )

    result = await tool.run_json_async({})

    assert result == {
        "success": False,
        "status": "waiting_for_user",
        "message": (
            "I need access to Gmail to continue. "
            "Please connect it below, then let me know once you have."
        ),
        "interactions": [
            {
                "type": "connect_apps",
                "field": "connect_apps",
                "label": "Connect your apps",
                "apps": ["Gmail"],
            }
        ],
    }


@pytest.mark.asyncio
async def test_unavailable_tool_oauth_required_without_app_name_keeps_error(
    monkeypatch,
):
    """No catalog app was resolvable - a connect_apps card would have
    nothing to point the user at, so this keeps the original error contract
    rather than pausing with an interaction naming no app at all."""
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    tool = _unavailable_tool(allow_users=["7"], failure_code="oauth_token_required")

    result = await tool.run_json_async({})

    assert result["status"] == "error"
    assert result["is_error"] is True


@pytest.mark.asyncio
async def test_unavailable_tool_non_oauth_failure_keeps_error_even_with_app_name(
    monkeypatch,
):
    """An app_name being resolvable doesn't by itself mean the failure is
    something the user can fix by connecting - only an actual
    oauth_token_required failure_code pauses; every other failure (a broken
    launch config, a runtime connection error, etc.) keeps erroring."""
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    tool = _unavailable_tool(
        allow_users=["7"],
        reason="runtime_connection_failed",
        app_name="Gmail",
    )

    result = await tool.run_json_async({})

    assert result["status"] == "error"
    assert result["is_error"] is True


@pytest.mark.asyncio
async def test_unavailable_tool_accepts_public_runtime_reason_and_message(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    tool = _unavailable_tool(
        allow_users=["7"],
        reason="initialize",
        message="MCP server initialization failed.",
    )

    result = await tool.run_json_async({})

    assert tool.description == "MCP server initialization failed."
    assert result == {
        "success": False,
        "status": "error",
        "is_error": True,
        "error": "MCP server initialization failed.",
        "reason": "initialize",
        "content": [{"text": "MCP server initialization failed."}],
    }


@pytest.mark.parametrize(
    ("reason", "message"),
    [
        ("session_start", "MCP server could not be started."),
        ("initialize", "MCP server initialization failed."),
        ("list_tools", "MCP server tools could not be loaded."),
    ],
)
def test_unavailable_tool_supports_public_load_phase_messages(reason, message):
    tool = _unavailable_tool(reason=reason, message=message)

    result = tool.run_json_sync({})

    assert result["reason"] == reason
    assert result["error"] == message
    assert result["content"] == [{"text": message}]


def test_unavailable_tool_sync_authorized_returns_clean_error(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    tool = _unavailable_tool(allow_users=["7"])

    result = tool.run_json_sync({})

    assert result["is_error"] is True
    assert "MCP server credentials are unavailable" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_unavailable_tool_async_unauthorized_uses_mcp_access_denied(
    monkeypatch,
):
    monkeypatch.setenv("XAGENT_USER_ID", "8")
    tool = _unavailable_tool(allow_users=["7"])

    result = await tool.run_json_async({})

    assert result == {
        "content": [
            {
                "text": (
                    "Access denied: User 8 is not authorized to use tool "
                    "mcp_google_drive_42_unavailable"
                )
            }
        ],
        "is_error": True,
    }


def test_unavailable_tool_return_schema_accepts_access_denied_result(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "8")
    tool = _unavailable_tool(allow_users=["7"])

    result = tool.run_json_sync({})
    parsed = tool.return_type().model_validate(result)

    assert parsed.error is None
    assert parsed.is_error is True


def test_unavailable_tool_sync_unauthorized_uses_mcp_access_denied(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "8")
    tool = _unavailable_tool(allow_users=["7"])

    result = tool.run_json_sync({})

    assert result["is_error"] is True
    assert "Access denied" in result["content"][0]["text"]


def test_unavailable_tool_access_denied_wins_over_connect_apps_pause(monkeypatch):
    # An unauthorized user must never see (or learn the existence of) a
    # connect_apps card naming an app they may not be entitled to know
    # about, even when the underlying failure would otherwise qualify for
    # the OAuth-pause branch below it.
    monkeypatch.setenv("XAGENT_USER_ID", "8")
    tool = _unavailable_tool(
        allow_users=["7"],
        failure_code="oauth_token_required",
        app_name="Gmail",
    )

    result = tool.run_json_sync({})

    assert result.get("status") != "waiting_for_user"
    assert "interactions" not in result
    assert result["is_error"] is True
    assert "Access denied" in result["content"][0]["text"]


def test_unavailable_tool_missing_user_permission_regression(monkeypatch):
    monkeypatch.delenv("XAGENT_USER_ID", raising=False)

    assert _unavailable_tool(allow_users=None).run_json_sync({})["is_error"] is True
    assert (
        "MCP server credentials"
        in _unavailable_tool(allow_users=None).run_json_sync({})["content"][0]["text"]
    )
    assert (
        "Access denied"
        in _unavailable_tool(allow_users=["7"]).run_json_sync({})["content"][0]["text"]
    )
    assert (
        "MCP server credentials"
        in _unavailable_tool(allow_users=["system"]).run_json_sync({})["content"][0][
            "text"
        ]
    )


def test_mcp_tool_adapter_permission_helper_regression(monkeypatch):
    adapter = MCPToolAdapter(
        MCPTool(name="remote_tool", description="Remote tool", inputSchema={}),
        {"transport": "stdio", "command": "echo"},
        allow_users=["7"],
    )

    monkeypatch.delenv("XAGENT_USER_ID", raising=False)
    assert adapter._get_current_user_id() is None
    assert adapter._is_user_allowed(None) is False

    system_adapter = MCPToolAdapter(
        MCPTool(name="remote_tool", description="Remote tool", inputSchema={}),
        {"transport": "stdio", "command": "echo"},
        allow_users=["system"],
    )
    assert system_adapter._is_user_allowed(None) is True

    unrestricted_adapter = MCPToolAdapter(
        MCPTool(name="remote_tool", description="Remote tool", inputSchema={}),
        {"transport": "stdio", "command": "echo"},
        allow_users=None,
    )
    assert unrestricted_adapter._is_user_allowed(None) is True

    monkeypatch.setenv("XAGENT_USER_ID", "8")
    assert adapter._get_current_user_id() == "8"
    assert adapter._is_user_allowed("8") is False
    assert adapter._is_user_allowed("7") is True


def test_unavailable_tool_return_value_as_string_extracts_text():
    tool = _unavailable_tool()

    assert (
        tool.return_value_as_string(
            {"content": [{"text": "first"}, {"text": "second"}], "is_error": True}
        )
        == "first\nsecond"
    )


@pytest.mark.asyncio
async def test_factory_builds_unavailable_tools_without_normal_loader(monkeypatch):
    async def fail_loader(*args, **kwargs):
        raise AssertionError("normal loader should not be called")

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fail_loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs([_unavailable_config()])

    assert [tool.name for tool in tools] == ["mcp_google_drive_42_unavailable"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_failure_code", "expected_failure_code"),
    [
        ("oauth_token_required", "oauth_token_required"),
        ("other_valid_code", None),
        (" oauth_token_required", None),
        (123, None),
    ],
)
async def test_factory_revalidates_unavailable_failure_code(
    monkeypatch,
    raw_failure_code,
    expected_failure_code,
):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    config = _unavailable_config(allow_users=["7"], failure_code=raw_failure_code)
    config["config"]["diagnostic"] = {
        "actor": "internal-actor",
        "resource": "internal-resource",
        "provider": "internal-provider",
        "access_token": "internal-token",
    }

    tools = await ToolFactory._create_mcp_tools_from_configs([config])
    result = await tools[0].run_json_async({})

    assert result.get("failure_code") == expected_failure_code
    assert "internal-" not in repr(result)


@pytest.mark.asyncio
async def test_factory_threads_app_name_into_unavailable_tool(
    monkeypatch,
):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    config = _unavailable_config(
        allow_users=["7"],
        failure_code="oauth_token_required",
        app_name="Gmail",
    )

    tools = await ToolFactory._create_mcp_tools_from_configs([config])
    result = await tools[0].run_json_async({})

    assert result["status"] == "waiting_for_user"
    assert result["interactions"][0]["apps"] == ["Gmail"]


@pytest.mark.asyncio
async def test_factory_without_app_name_keeps_unavailable_tool_erroring(monkeypatch):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    config = _unavailable_config(allow_users=["7"], failure_code="oauth_token_required")

    tools = await ToolFactory._create_mcp_tools_from_configs([config])
    result = await tools[0].run_json_async({})

    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_unavailable_config_failure_code_reaches_tool_failure_trace(monkeypatch):
    class CapturingTracer:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        async def trace_event(self, event_type: Any, **kwargs: Any) -> None:
            self.events.append(
                {
                    "type": getattr(event_type, "value", str(event_type)),
                    "data": kwargs.get("data") or {},
                }
            )

    monkeypatch.setenv("XAGENT_USER_ID", "7")
    config = _unavailable_config(allow_users=["7"], failure_code="oauth_token_required")
    config["config"]["diagnostic"] = {
        "actor": "internal-actor",
        "resource": "internal-resource",
        "provider": "internal-provider",
        "access_token": "internal-token",
        "refresh_payload": "internal-refresh-payload",
        "webhook_url": "https://internal.example/webhook",
    }
    tools = await ToolFactory._create_mcp_tools_from_configs([config])
    result = await tools[0].run_json_async({})
    tracer = CapturingTracer()
    runtime = PatternRuntime(tracer=tracer, execution_id="task-oauth-required")

    await runtime.on_tool_end(
        tool_call={"name": tools[0].name, "id": "call-1"},
        result=result,
    )

    assert tracer.events[0]["type"] == "action_error_tool"
    assert tracer.events[0]["data"]["failure_code"] == "oauth_token_required"
    public_payload = repr(result) + repr(tracer.events)
    assert "internal-" not in public_payload
    assert "webhook" not in public_payload


@pytest.mark.asyncio
async def test_factory_normal_loader_failure_keeps_unavailable_tool(monkeypatch):
    async def fail_loader(*args, **kwargs):
        raise RuntimeError("loader failed")

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fail_loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            _unavailable_config(),
            {"name": "normal", "transport": "stdio", "config": {"command": "echo"}},
        ]
    )

    assert [tool.name for tool in tools] == [
        "mcp_google_drive_42_unavailable",
        "mcp_normal_unavailable",
    ]


@pytest.mark.asyncio
async def test_factory_keeps_successful_tools_and_exposes_each_failed_server(
    monkeypatch,
):
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    healthy_tool = _unavailable_tool(server_name="Healthy result", server_id=99)

    async def loader(connections, **kwargs):
        return MCPLoadResult(
            tools=(healthy_tool,),
            loaded_servers=("healthy",),
            failures=(
                MCPServerLoadFailure(
                    server_name="broken",
                    phase=MCPFailurePhase.INITIALIZE,
                    error_type="RuntimeError",
                    attempts=3,
                ),
                MCPServerLoadFailure(
                    server_name="partial",
                    phase=MCPFailurePhase.ADAPTER_CONSTRUCTION,
                    error_type="ValidationError",
                ),
            ),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "id": 11,
                "name": "healthy",
                "transport": "stdio",
                "config": {"command": "echo"},
                "allow_users": ["7"],
            },
            {
                "id": 12,
                "name": "broken",
                "transport": "stdio",
                "config": {"command": "echo"},
                "allow_users": ["7"],
            },
            {
                "id": 13,
                "name": "partial",
                "transport": "stdio",
                "config": {"command": "echo"},
                "allow_users": ["7"],
            },
        ]
    )

    assert [tool.name for tool in tools] == [
        "mcp_broken_12_unavailable",
        "mcp_partial_13_unavailable",
        "mcp_healthy_result_99_unavailable",
    ]
    broken = tools[0].run_json_sync({})
    partial = tools[1].run_json_sync({})
    assert broken["reason"] == "initialize"
    assert broken["error"] == "MCP server initialization failed."
    assert partial["reason"] == "adapter_construction"
    assert partial["error"] == "Some MCP server tools could not be prepared."
    public_result = repr(broken) + repr(partial)
    assert "RuntimeError" not in public_result
    assert "ValidationError" not in public_result


@pytest.mark.asyncio
async def test_factory_partial_failures_create_only_one_unavailable_tool_per_server(
    monkeypatch,
):
    async def loader(connections, **kwargs):
        return MCPLoadResult(
            tools=(),
            loaded_servers=(),
            failures=(
                MCPServerLoadFailure(
                    server_name="partial",
                    phase=MCPFailurePhase.ADAPTER_CONSTRUCTION,
                    error_type="FirstError",
                ),
                MCPServerLoadFailure(
                    server_name="partial",
                    phase=MCPFailurePhase.SANDBOX_TOOL_WRAP,
                    error_type="SecondError",
                ),
            ),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "id": 13,
                "name": "partial",
                "transport": "stdio",
                "config": {"command": "echo"},
            }
        ]
    )

    assert [tool.name for tool in tools] == ["mcp_partial_13_unavailable"]


@pytest.mark.asyncio
async def test_factory_exposes_server_returning_no_tools(monkeypatch):
    async def loader(connections, **kwargs):
        return MCPLoadResult(
            tools=(),
            loaded_servers=(),
            failures=(
                MCPServerLoadFailure(
                    server_name="empty",
                    phase=MCPFailurePhase.NO_TOOLS_RETURNED,
                    error_type=None,
                ),
            ),
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "id": 14,
                "name": "empty",
                "transport": "stdio",
                "config": {"command": "echo"},
            }
        ]
    )

    assert [tool.name for tool in tools] == ["mcp_empty_14_unavailable"]
    result = tools[0].run_json_sync({})
    assert result["reason"] == "no_tools_returned"
    assert result["error"] == "MCP server returned no available tools."


@pytest.mark.asyncio
async def test_factory_malformed_normal_config_keeps_unavailable_tool(caplog):
    caplog.set_level(logging.WARNING)

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            _unavailable_config(),
            {"name": "normal", "transport": "stdio", "config": None},
        ]
    )

    assert [tool.name for tool in tools] == [
        "mcp_google_drive_42_unavailable",
        "mcp_normal_unavailable",
    ]
    assert "MCP server config 'config' field for server 'normal'" in caplog.text
    assert "dictionary, got NoneType" in caplog.text


@pytest.mark.asyncio
async def test_factory_does_not_pass_unavailable_config_to_normal_loader(monkeypatch):
    seen_connections = None

    async def loader(connections, **kwargs):
        nonlocal seen_connections
        seen_connections = connections
        return MCPLoadResult(tools=(), loaded_servers=(), failures=())

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        loader,
    )

    await ToolFactory._create_mcp_tools_from_configs(
        [
            _unavailable_config(),
            {
                "name": "normal",
                "transport": "stdio",
                "config": {"command": "echo", "args": "--flag value"},
            },
        ]
    )

    assert seen_connections == {
        "normal": {"transport": "stdio", "command": "echo", "args": ["--flag", "value"]}
    }


@pytest.mark.asyncio
async def test_factory_preserves_runtime_keys_for_normal_configs(monkeypatch):
    seen_connections = None

    async def loader(connections, **kwargs):
        nonlocal seen_connections
        seen_connections = connections
        return MCPLoadResult(tools=(), loaded_servers=(), failures=())

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        loader,
    )

    await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "name": "normal",
                "transport": "streamable_http",
                "config": {"url": "https://mcp.example.test"},
                "runtime_bindings": [{"binding": "value"}],
                "runtime_input_schema": {"context": {"account": {"type": "string"}}},
                "connector_runtime": {"context": {"account": "a"}},
                "allow_delegated_authorization": True,
            },
        ]
    )

    assert seen_connections == {
        "normal": {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "runtime_bindings": [{"binding": "value"}],
            "runtime_input_schema": {"context": {"account": {"type": "string"}}},
            "connector_runtime": {"context": {"account": "a"}},
            "allow_delegated_authorization": True,
        }
    }


@pytest.mark.asyncio
async def test_factory_propagates_connector_runtime_error(monkeypatch):
    async def fail_loader(*args, **kwargs):
        raise ConnectorRuntimeError(
            "connector_runtime_unavailable",
            "Connector runtime context is unavailable.",
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        fail_loader,
    )

    with pytest.raises(ConnectorRuntimeError):
        await ToolFactory._create_mcp_tools_from_configs(
            [
                _unavailable_config(),
                {"name": "normal", "transport": "stdio", "config": {"command": "echo"}},
            ]
        )


@pytest.mark.asyncio
async def test_factory_returns_unavailable_tools_before_normal_tools(monkeypatch):
    normal_tool = _unavailable_tool(server_name="Normal", server_id=99)

    async def loader(connections, **kwargs):
        return MCPLoadResult(
            tools=(normal_tool,), loaded_servers=("normal",), failures=()
        )

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        loader,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            {"name": "normal", "transport": "stdio", "config": {"command": "echo"}},
            _unavailable_config(),
        ]
    )

    assert [tool.name for tool in tools] == [
        "mcp_google_drive_42_unavailable",
        "mcp_normal_99_unavailable",
    ]


@pytest.mark.asyncio
async def test_factory_malformed_unavailable_config_does_not_drop_valid_one():
    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            _unavailable_config(name=None, server_id=None),
            _unavailable_config(name="Google Drive", server_id=42),
        ]
    )

    assert [tool.name for tool in tools] == [
        "mcp_server_unavailable",
        "mcp_google_drive_42_unavailable",
    ]


def test_selection_plain_mcp_admits_unavailable_tool():
    tool = _unavailable_tool()
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp"])

    assert spec.compute_allowed_names([tool]) == frozenset({tool.name})


def test_selection_scoped_mcp_admits_unavailable_tool_by_source_server():
    tool = _unavailable_tool()
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp:Google Drive"])

    assert spec.compute_allowed_names([tool]) == frozenset({tool.name})


def test_selection_unrelated_scoped_mcp_rejects_unavailable_tool():
    tool = _unavailable_tool()
    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp:Slack"])

    assert spec.compute_allowed_names([tool]) == frozenset()

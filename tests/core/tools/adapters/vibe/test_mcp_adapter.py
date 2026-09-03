import json
import logging
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, create_model
from pydantic.alias_generators import to_camel

from xagent.core.model.chat.basic.claude import _fix_pydantic_schema_for_claude
from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    _FIELD_TEXT_MAX_CHARS,
    EmptyArgsModel,
    MCPFailurePhase,
    MCPServerLoadFailure,
    MCPToolAdapter,
    _build_mcp_tool_adapter,
    _compact_json,
    _exception_indicates_http_401,
    _mcp_return_value_as_string,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.tool_naming_limits import (
    MAX_AGENT_TOOL_NAME_LENGTH,
)


def _http_status_error(
    *, status_code: int = 401, authenticate: list[str] | None = None
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://mcp.example.test/tools")
    headers = [(b"www-authenticate", value.encode()) for value in (authenticate or [])]
    response = httpx.Response(status_code, headers=headers, request=request)
    return httpx.HTTPStatusError(
        "planted-secret-must-not-be-evidence",
        request=request,
        response=response,
    )


def _mcp_tool(name: str = "echo") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} tool",
        inputSchema={"type": "object", "properties": {}},
    )


class _SdkV2CallToolResult(BaseModel):
    """Mirrors mcp SDK 2.0.0's ``CallToolResult`` field naming (snake_case
    Python attributes with camelCase wire aliases), unlike the currently
    locked mcp 1.19.0's plain camelCase attributes. Guards against the
    adapter silently regressing to dropping structured_content/is_error
    if the SDK constraint ever allows resolving mcp>=2.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    content: list[Any] = []
    structured_content: Any = None
    is_error: bool = False


def test_mcp_server_load_failure_is_immutable():
    failure = MCPServerLoadFailure(
        server_name="mail",
        phase=MCPFailurePhase.INITIALIZE,
        error_type="RuntimeError",
    )

    with pytest.raises(FrozenInstanceError):
        failure.attempts = 2


@pytest.mark.asyncio
async def test_mcp_loader_returns_structured_success(monkeypatch):
    class FakeSession:
        async def initialize(self):
            return None

    @asynccontextmanager
    async def fake_create_session(_connection):
        yield FakeSession()

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(
        mcp_adapter_module,
        "load_mcp_tools",
        AsyncMock(return_value=[_mcp_tool()]),
    )

    result = await load_mcp_tools_as_agent_tools(
        {"mail": {"transport": "stdio", "command": "python", "args": []}}
    )

    assert len(result.tools) == 1
    assert result.loaded_servers == ("mail",)
    assert result.failures == ()


@pytest.mark.asyncio
async def test_mcp_loader_preserves_partial_server_progress(monkeypatch):
    class FakeSession:
        def __init__(self, should_fail: bool):
            self.should_fail = should_fail

        async def initialize(self):
            if self.should_fail:
                raise ConnectionError("Bearer planted-initialize-secret")

    @asynccontextmanager
    async def fake_create_session(connection):
        yield FakeSession(connection["command"] == "fail")

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(
        mcp_adapter_module,
        "load_mcp_tools",
        AsyncMock(return_value=[_mcp_tool()]),
    )
    monkeypatch.setattr(mcp_adapter_module.asyncio, "sleep", AsyncMock())

    result = await load_mcp_tools_as_agent_tools(
        {
            "healthy": {
                "transport": "stdio",
                "command": "python",
                "args": [],
            },
            "broken": {"transport": "stdio", "command": "fail", "args": []},
        }
    )

    assert len(result.tools) == 1
    assert result.loaded_servers == ("healthy",)
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.server_name == "broken"
    assert failure.phase is MCPFailurePhase.INITIALIZE
    assert failure.error_type == "ConnectionError"
    assert failure.attempts == 3
    assert "planted-initialize-secret" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing_stage", "expected_phase"),
    [
        ("session", MCPFailurePhase.SESSION_START),
        ("initialize", MCPFailurePhase.INITIALIZE),
        ("list_tools", MCPFailurePhase.LIST_TOOLS),
    ],
)
async def test_mcp_loader_classifies_direct_failure_phase(
    monkeypatch, caplog, failing_stage, expected_phase
):
    class FakeSession:
        async def initialize(self):
            if failing_stage == "initialize":
                raise ValueError("Bearer planted-phase-secret")

    @asynccontextmanager
    async def fake_create_session(_connection):
        if failing_stage == "session":
            raise ValueError("Bearer planted-phase-secret")
        yield FakeSession()

    async def fake_load_tools(_session):
        if failing_stage == "list_tools":
            raise ValueError("Bearer planted-phase-secret")
        return [_mcp_tool()]

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools", fake_load_tools)
    monkeypatch.setattr(mcp_adapter_module.asyncio, "sleep", AsyncMock())
    caplog.set_level("WARNING")

    result = await load_mcp_tools_as_agent_tools(
        {"broken": {"transport": "stdio", "command": "python", "args": []}}
    )

    assert result.tools == ()
    assert result.loaded_servers == ()
    assert len(result.failures) == 1
    assert result.failures[0].phase is expected_phase
    assert result.failures[0].error_type == "ValueError"
    assert result.failures[0].attempts == 3
    assert "planted-phase-secret" not in repr(result)
    assert "planted-phase-secret" not in caplog.text


@pytest.mark.asyncio
async def test_mcp_loader_direct_retry_exhaustion_logs_debug_traceback(
    monkeypatch, caplog
):
    # Companion to test_mcp_loader_classifies_direct_failure_phase above:
    # that test pins WARNING and checks no secret leaks into the always-on
    # log; this one pins DEBUG and checks the opt-in traceback is actually
    # there once retries are exhausted -- the retry loop's final attempt
    # falls through to a silent `else` branch otherwise, so this is the only
    # place that failure detail is ever logged for a direct-transport server.
    @asynccontextmanager
    async def fake_create_session(_connection):
        raise ValueError("boom")
        yield  # pragma: no cover - unreachable, satisfies asynccontextmanager

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(mcp_adapter_module.asyncio, "sleep", AsyncMock())
    caplog.set_level("DEBUG")

    result = await load_mcp_tools_as_agent_tools(
        {"broken": {"transport": "stdio", "command": "python", "args": []}}
    )

    assert len(result.failures) == 1
    assert result.failures[0].attempts == 3
    assert "ValueError: boom" in caplog.text


@pytest.mark.asyncio
async def test_mcp_loader_reports_no_tools(monkeypatch):
    class FakeSession:
        async def initialize(self):
            return None

    @asynccontextmanager
    async def fake_create_session(_connection):
        yield FakeSession()

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(
        mcp_adapter_module, "load_mcp_tools", AsyncMock(return_value=[])
    )

    result = await load_mcp_tools_as_agent_tools(
        {"empty": {"transport": "stdio", "command": "python", "args": []}}
    )

    assert result.tools == ()
    assert result.loaded_servers == ()
    assert len(result.failures) == 1
    assert result.failures[0].phase is MCPFailurePhase.NO_TOOLS_RETURNED
    assert result.failures[0].error_type is None


@pytest.mark.asyncio
async def test_mcp_loader_preserves_tools_when_one_adapter_fails(monkeypatch):
    class FakeSession:
        async def initialize(self):
            return None

    @asynccontextmanager
    async def fake_create_session(_connection):
        yield FakeSession()

    original_builder = mcp_adapter_module._build_mcp_tool_adapter

    def fake_builder(server_name, connection, mcp_tool, **kwargs):
        if mcp_tool.name == "broken":
            raise TypeError("adapter planted-secret")
        return original_builder(server_name, connection, mcp_tool, **kwargs)

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(
        mcp_adapter_module,
        "load_mcp_tools",
        AsyncMock(return_value=[_mcp_tool("healthy"), _mcp_tool("broken")]),
    )
    monkeypatch.setattr(mcp_adapter_module, "_build_mcp_tool_adapter", fake_builder)

    result = await load_mcp_tools_as_agent_tools(
        {"partial": {"transport": "stdio", "command": "python", "args": []}}
    )

    assert len(result.tools) == 1
    assert result.loaded_servers == ("partial",)
    assert len(result.failures) == 1
    assert result.failures[0].phase is MCPFailurePhase.ADAPTER_CONSTRUCTION
    assert result.failures[0].error_type == "TypeError"
    assert "planted-secret" not in repr(result)


def test_build_mcp_tool_adapter_stamps_normalized_source_server():
    """``_build_mcp_tool_adapter`` carries the originating server identity
    onto ``metadata.source_server``, normalized once via the shared SSOT,
    while the LLM-visible name keeps its original casing. Server-scoped
    selection matches on the structured field, not the tool name."""
    mcp_tool = SimpleNamespace(
        name="send_message",
        description="Send a message",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = _build_mcp_tool_adapter(
        "Google Drive",
        {"transport": "stdio", "command": "python", "args": []},
        mcp_tool,
    )

    assert adapter.source_server == "google_drive"
    assert adapter.metadata.source_server == "google_drive"
    # LLM-visible name keeps original casing / spacing folded to underscores.
    assert adapter.name == "mcp_Google Drive_send_message".replace(" ", "_")


def test_mcp_tool_adapter_source_server_defaults_none():
    """A directly constructed adapter with no server origin reports
    ``source_server`` as ``None`` (no scoped-selection match)."""
    mcp_tool = SimpleNamespace(
        name="ping",
        description="ping",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )
    assert adapter.source_server is None
    assert adapter.metadata.source_server is None


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        # `.` namespacing, the case the fix was written for.
        ("coding.start", "mcp_Coding_MCP_coding_start"),
        # Path-like separator.
        ("list/items", "mcp_Coding_MCP_list_items"),
        # Parentheses and a space alongside the version marker.
        ("run (v2)", "mcp_Coding_MCP_run__v2_"),
        # Non-ASCII names collapse to underscores rather than raising or
        # being romanized -- `_semantic_slug` (agent_tool_names.py) makes
        # the opposite, deliberate choice for agent-delegation tool names
        # (pinyin-transliterate instead of drop) because it can carry a
        # unique per-agent id in the suffix; MCP tool names have no such
        # id to fall back on for uniqueness, so this test pins today's
        # trade-off (two same-length CJK names collide) rather than
        # leaving it to be discovered later.
        ("查询", "mcp_Coding_MCP___"),
    ],
)
def test_mcp_tool_adapter_name_strips_disallowed_characters_for_openai_compatible_apis(
    tool_name, expected
):
    """OpenAI-compatible chat-completions APIs (and at least DeepSeek's)
    validate `tools[].function.name` against `^[a-zA-Z0-9_-]+$` and 400
    the whole call if any tool name fails it. MCP servers namespace tool
    names with all sorts of characters (e.g. the `.` in `coding.start`)
    to avoid collisions between generically-named tools -- none of that
    must survive into the LLM-visible name."""
    mcp_tool = SimpleNamespace(
        name=tool_name,
        description="Start a coding run",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = _build_mcp_tool_adapter(
        "Coding MCP",
        {"transport": "streamable_http", "url": "http://127.0.0.1:8642/mcp"},
        mcp_tool,
    )

    assert re.fullmatch(r"[A-Za-z0-9_-]+", adapter.name)
    assert adapter.name == expected


def test_mcp_tool_adapter_name_is_truncated_to_the_provider_length_limit():
    """Same failure mode as an illegal character -- an over-long name is
    rejected by the same providers, just on length. Nothing upstream
    (MCP server name, MCP tool name) is length-bounded, so this adapter
    must enforce the limit itself instead of assuming it never comes up.
    A tool name alone at or past the limit squeezes the prefix down to
    nothing rather than wrapping around from the end (a negative slice
    on the *tool* name would otherwise do exactly that).
    """
    mcp_tool = SimpleNamespace(
        name="x" * 200,
        description="A tool with an absurdly long name",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = _build_mcp_tool_adapter(
        "Coding MCP",
        {"transport": "streamable_http", "url": "http://127.0.0.1:8642/mcp"},
        mcp_tool,
    )

    assert len(adapter.name) == MAX_AGENT_TOOL_NAME_LENGTH
    assert adapter.name == "x" * MAX_AGENT_TOOL_NAME_LENGTH


def test_mcp_tool_adapter_truncation_keeps_same_server_tools_distinct():
    """Regression: truncating the *tool name* end of `prefix + tool_name`
    can make two distinct tools on one long-named server collapse into
    one identical LLM-visible name once the combined length passes
    `MAX_AGENT_TOOL_NAME_LENGTH` -- `_find_tool` (react.py) has no
    duplicate-name detection, so the model asking for one tool would
    silently get whichever tool happens to register first instead. MCP
    server names are bounded at 100 (web/api/mcp.py), so 59 characters
    -- long enough that `mcp_<server>_` alone already reaches the limit
    -- is a length a real deployment can produce, not a contrived one.
    The fix keeps the tool name whole and squeezes the prefix instead.
    """
    server = "s" * 59

    def _adapter(tool_name: str) -> MCPToolAdapter:
        return _build_mcp_tool_adapter(
            server,
            {"transport": "streamable_http", "url": "http://127.0.0.1:8642/mcp"},
            SimpleNamespace(
                name=tool_name,
                description=tool_name,
                inputSchema={"type": "object", "properties": {}},
            ),
        )

    read_name = _adapter("read_file").name
    delete_name = _adapter("delete_all_files").name

    assert read_name != delete_name
    assert read_name.endswith("read_file")
    assert delete_name.endswith("delete_all_files")


def test_mcp_tool_adapter_defaults_to_not_concurrency_safe():
    mcp_tool = SimpleNamespace(
        name="list_messages",
        description="List messages",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    assert adapter.metadata.concurrency_safe is False


def test_build_mcp_tool_adapter_marks_all_tools_safe_when_server_opts_in():
    mcp_tool = SimpleNamespace(
        name="list_messages",
        description="List messages",
        inputSchema={"type": "object", "properties": {}},
    )

    adapter = _build_mcp_tool_adapter(
        "mail",
        {"transport": "stdio", "command": "python", "args": []},
        mcp_tool,
        concurrency_safe=True,
    )

    assert adapter.metadata.concurrency_safe is True


def test_exception_indicates_http_401_uses_bounded_status_signals():
    class StatusError(RuntimeError):
        status_code = 401

    assert _exception_indicates_http_401(StatusError("request failed"))
    assert _exception_indicates_http_401(RuntimeError("HTTP status 401"))
    assert _exception_indicates_http_401(RuntimeError("401 Unauthorized"))
    assert not _exception_indicates_http_401(RuntimeError("Unauthorized"))
    assert not _exception_indicates_http_401(
        RuntimeError("connection reset on port 401")
    )
    assert not _exception_indicates_http_401(RuntimeError("tool returned id 40123"))


def test_resolver_challenge_extracts_from_nested_group_cause_and_context():
    cause = RuntimeError("outer")
    cause.__cause__ = _http_status_error(
        authenticate=['Bearer error="invalid_token", scope="records.read"']
    )
    context = RuntimeError("context")
    context.__context__ = cause
    nested = ExceptionGroup("nested", [RuntimeError("noise"), context])

    challenge = mcp_adapter_module._resolver_invalid_token_challenge(
        ExceptionGroup("root", [nested])
    )

    assert challenge is not None
    assert challenge.params["error"] == "invalid_token"
    assert challenge.scope == "records.read"


def test_resolver_challenge_traversal_handles_cycles():
    outer = RuntimeError("outer")
    inner = RuntimeError("inner")
    outer.__cause__ = inner
    inner.__context__ = outer

    assert mcp_adapter_module._resolver_invalid_token_challenge(outer) is None


def test_resolver_challenge_traversal_stops_at_named_node_budget():
    current: BaseException = _http_status_error(
        authenticate=['Bearer error="invalid_token"']
    )
    for index in range(mcp_adapter_module._RESOLVER_HTTP_401_NODE_LIMIT):
        wrapper = RuntimeError(f"wrapper-{index}")
        wrapper.__cause__ = current
        current = wrapper

    assert mcp_adapter_module._resolver_invalid_token_challenge(current) is None


def test_resolver_challenge_uses_all_www_authenticate_header_values():
    exc = _http_status_error(
        authenticate=[
            'Basic realm="legacy"',
            (
                'Bearer error="invalid_token", '
                'resource_metadata="https://mcp.example.test/.well-known/resource"'
            ),
        ]
    )

    challenge = mcp_adapter_module._resolver_invalid_token_challenge(exc)

    assert challenge is not None
    assert challenge.resource_metadata_url == (
        "https://mcp.example.test/.well-known/resource"
    )


@pytest.mark.parametrize(
    "exc",
    [
        _http_status_error(authenticate=[]),
        _http_status_error(authenticate=['Bearer error="invalid_token']),
        _http_status_error(authenticate=['Basic error="invalid_token"']),
        _http_status_error(authenticate=['Bearer error="insufficient_scope"']),
        _http_status_error(authenticate=['Bearer scope="records.read"']),
        _http_status_error(
            status_code=403, authenticate=['Bearer error="invalid_token"']
        ),
        RuntimeError("HTTP 401 Unauthorized; Bearer error=invalid_token"),
    ],
)
def test_resolver_challenge_rejects_non_refreshable_evidence(exc):
    assert mcp_adapter_module._resolver_invalid_token_challenge(exc) is None


def test_resolver_401_evidence_degrades_when_mcp_oauth_unimportable():
    """_resolver_401_evidence lazily imports xagent.web.services.mcp_oauth,
    which needs sqlalchemy (directly, and transitively through
    xagent.web.services.__init__'s other eager imports). That's fine on
    the backend host, but this function runs inside MCPToolAdapter.
    run_json_async -- the exact class/method tool_runner.py reconstructs
    and calls for every sandboxed npx/uvx MCP tool call (see PR #1710) --
    and the sandbox never installs sqlalchemy. A 401 from a sandboxed
    OAuth-protected connector (e.g. Xero, once its access token expires
    mid-session) must fail the same clean "authorization failed" way an
    unparsable challenge already does, not crash with
    ModuleNotFoundError."""
    exc = _http_status_error(authenticate=['Bearer error="invalid_token"'])

    with patch.dict(sys.modules, {"xagent.web.services.mcp_oauth": None}):
        challenge, response_ids = mcp_adapter_module._resolver_401_evidence(exc)

    assert challenge is None
    assert len(response_ids) == 1


def test_mcp_return_value_as_string_keeps_malformed_scalar_content_together():
    assert _mcp_return_value_as_string({"content": "error"}) == "error"


def test_mcp_return_value_as_string_surfaces_structured_content():
    value = {
        "content": [{"text": "I'll wait for the background agents to complete."}],
        "structured_content": {"status": "completed", "run_id": "abc"},
        "is_error": False,
    }

    rendered = _mcp_return_value_as_string(value)

    assert "I'll wait for the background agents to complete." in rendered
    assert "completed" in rendered
    assert "abc" in rendered


def test_mcp_return_value_as_string_omits_structured_content_when_absent():
    value = {"content": [{"text": "ok"}], "is_error": False}

    assert _mcp_return_value_as_string(value) == "ok"


def test_mcp_return_value_as_string_omits_no_content_placeholder_when_structured_content_present():
    value = {
        "content": [],
        "structured_content": {"status": "completed"},
        "is_error": False,
    }

    rendered = _mcp_return_value_as_string(value)

    assert "No content returned" not in rendered
    assert "completed" in rendered


@pytest.mark.asyncio
async def test_execute_mcp_call_forwards_structured_content(monkeypatch):
    mcp_tool = _mcp_tool("status")
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "streamable_http", "url": "https://mcp.example.test"},
    )

    class _FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            return CallToolResult(
                content=[],
                isError=False,
                structuredContent={"status": "completed", "run_id": "abc"},
            )

    @asynccontextmanager
    async def _fake_create_session(_connection):
        yield _FakeSession()

    monkeypatch.setattr(mcp_adapter_module, "create_session", _fake_create_session)

    result = await adapter._execute_mcp_call(adapter.connection, {}, {})

    assert result["structured_content"] == {"status": "completed", "run_id": "abc"}


@pytest.mark.asyncio
async def test_execute_mcp_call_structured_content_defaults_to_none(monkeypatch):
    mcp_tool = _mcp_tool("echo")
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "streamable_http", "url": "https://mcp.example.test"},
    )

    class _FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            return CallToolResult(content=[], isError=False)

    @asynccontextmanager
    async def _fake_create_session(_connection):
        yield _FakeSession()

    monkeypatch.setattr(mcp_adapter_module, "create_session", _fake_create_session)

    result = await adapter._execute_mcp_call(adapter.connection, {}, {})

    assert result["structured_content"] is None


@pytest.mark.asyncio
async def test_execute_mcp_call_reads_snake_case_sdk_structured_content(monkeypatch):
    """Guards against the adapter silently regressing on an mcp SDK version
    (e.g. 2.0.0) whose CallToolResult exposes structured_content instead of
    structuredContent as the Python attribute name."""
    mcp_tool = _mcp_tool("status")
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "streamable_http", "url": "https://mcp.example.test"},
    )

    class _FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            return _SdkV2CallToolResult(
                content=[],
                structured_content={"status": "completed", "run_id": "abc"},
                is_error=False,
            )

    @asynccontextmanager
    async def _fake_create_session(_connection):
        yield _FakeSession()

    monkeypatch.setattr(mcp_adapter_module, "create_session", _fake_create_session)

    result = await adapter._execute_mcp_call(adapter.connection, {}, {})

    assert result["structured_content"] == {"status": "completed", "run_id": "abc"}


@pytest.mark.asyncio
async def test_execute_mcp_call_reads_snake_case_sdk_error_flag(monkeypatch):
    """Same regression guard as above, for the error flag: a hasattr-based
    read on ``isError`` would silently report success on an SDK version
    whose attribute is ``is_error`` instead."""
    mcp_tool = _mcp_tool("status")
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "streamable_http", "url": "https://mcp.example.test"},
    )

    class _FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            return _SdkV2CallToolResult(content=[], is_error=True)

    @asynccontextmanager
    async def _fake_create_session(_connection):
        yield _FakeSession()

    monkeypatch.setattr(mcp_adapter_module, "create_session", _fake_create_session)

    result = await adapter._execute_mcp_call(adapter.connection, {}, {})

    assert result["is_error"] is True


@pytest.mark.asyncio
async def test_execute_mcp_call_structured_content_reaches_the_model_observation(
    monkeypatch,
):
    """The seam test: proves structured_content actually reaches the string
    the LLM reads, by running it through the real _execute_mcp_call and then
    the real ExecutionContext.add_tool_result -- not just that a hand-built
    dict happens to render correctly."""
    from xagent.core.agent.context import ExecutionContext

    mcp_tool = _mcp_tool("coding_agent_status")
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "streamable_http", "url": "https://mcp.example.test"},
    )

    class _FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            return CallToolResult(
                content=[],
                structuredContent={"status": "completed", "run_id": "abc"},
                isError=False,
            )

    @asynccontextmanager
    async def _fake_create_session(_connection):
        yield _FakeSession()

    monkeypatch.setattr(mcp_adapter_module, "create_session", _fake_create_session)

    result = await adapter._execute_mcp_call(adapter.connection, {}, {})

    context = ExecutionContext()
    message = context.add_tool_result(
        "coding_agent_status", result, tool_call_id="tool-1"
    )

    assert message.content.startswith("Tool coding_agent_status returned: ")
    assert "completed" in message.content
    assert "abc" in message.content


def test_return_type_declares_structured_content_field():
    mcp_tool = SimpleNamespace(
        name="status",
        description="status tool",
        inputSchema={"type": "object", "properties": {}},
        outputSchema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "streamable_http", "url": "https://mcp.example.test"},
    )

    return_model = adapter.return_type()

    assert "structured_content" in return_model.model_fields


def test_build_mcp_tool_adapter_honors_concurrent_tool_allowlist():
    safe_tool = SimpleNamespace(
        name="list_messages",
        description="List messages",
        inputSchema={"type": "object", "properties": {}},
    )
    unsafe_tool = SimpleNamespace(
        name="delete_message",
        description="Delete a message",
        inputSchema={"type": "object", "properties": {}},
    )

    safe_adapter = _build_mcp_tool_adapter(
        "mail",
        {"transport": "stdio", "command": "python", "args": []},
        safe_tool,
        concurrency_safe=True,
        concurrent_tools=["list_messages"],
    )
    unsafe_adapter = _build_mcp_tool_adapter(
        "mail",
        {"transport": "stdio", "command": "python", "args": []},
        unsafe_tool,
        concurrency_safe=True,
        concurrent_tools=["list_messages"],
    )

    assert safe_adapter.metadata.concurrency_safe is True
    assert unsafe_adapter.metadata.concurrency_safe is False


def test_build_args_model_handles_optional_array_schema():
    mcp_tool = SimpleNamespace(
        name="gmail_manage_labels",
        description="Manage Gmail labels",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "add_label_ids": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["action"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()
    parsed = args_model(action="modify_message", add_label_ids=["TRASH"])

    assert parsed.add_label_ids == ["TRASH"]


def test_normalize_args_by_schema_wraps_scalar_for_array_only_field():
    mcp_tool = SimpleNamespace(
        name="gmail_manage_labels",
        description="Manage Gmail labels",
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "add_label_ids": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["action"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema(
        {"action": "modify_message", "add_label_ids": "TRASH"}
    )

    assert normalized["add_label_ids"] == ["TRASH"]


def _google_analytics_run_report_mcp_tool() -> SimpleNamespace:
    return SimpleNamespace(
        name="google_analytics_run_report",
        description="Run a GA4 report",
        inputSchema={
            "type": "object",
            "properties": {
                "property_id": {"type": "string"},
                "dimensions": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": ["property_id"],
        },
    )


def test_normalize_args_by_schema_parses_json_encoded_array_string():
    """Regression test: an LLM sometimes double-encodes an array-only
    argument as a JSON string (e.g. '["date"]' instead of ["date"]). Without
    parsing it first, the naive scalar-wrap path used to produce
    ['["date"]'] — a single list item containing the literal brackets and
    quotes — which then reached the downstream API as a garbled field name.
    """
    adapter = MCPToolAdapter(
        mcp_tool=_google_analytics_run_report_mcp_tool(),
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema(
        {"property_id": "550713710", "dimensions": '["date"]'}
    )

    assert normalized["dimensions"] == ["date"]


def test_normalize_args_by_schema_parses_json_encoded_scalar_string():
    """Same bug, one step removed: a single item double-encoded as a JSON
    string (e.g. '"date"' instead of "date") must not fall through to the
    raw wrap, which would keep the item's own quotes as ['"date"'].
    """
    adapter = MCPToolAdapter(
        mcp_tool=_google_analytics_run_report_mcp_tool(),
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema(
        {"property_id": "550713710", "dimensions": '"date"'}
    )

    assert normalized["dimensions"] == ["date"]


@pytest.mark.parametrize(
    "value",
    [
        "TRASH",  # not valid JSON at all
        '{"a": 1}',  # valid JSON, but neither a list nor a string
        "null",  # valid JSON, decodes to None
        "123",  # valid JSON, decodes to an int
    ],
)
def test_normalize_args_by_schema_falls_back_to_raw_wrap_for_non_list_json(value):
    """When the string isn't JSON, or is JSON that decodes to something
    other than a list or a string, the field is wrapped as-is rather than
    silently dropped or coerced into an unrelated shape."""
    adapter = MCPToolAdapter(
        mcp_tool=_google_analytics_run_report_mcp_tool(),
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema(
        {"property_id": "550713710", "dimensions": value}
    )

    assert normalized["dimensions"] == [value]


def test_normalize_args_by_schema_keeps_scalar_for_union_scalar_or_array_field():
    mcp_tool = SimpleNamespace(
        name="multi_shape_tool",
        description="Accept string or string array input",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                }
            },
            "required": ["value"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    normalized = adapter._normalize_args_by_schema({"value": "abc"})

    assert normalized["value"] == "abc"


def test_build_args_model_handles_anyof_multi_type_schema():
    mcp_tool = SimpleNamespace(
        name="multi_type_tool",
        description="Accept string or integer input",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                }
            },
            "required": ["value"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()

    assert args_model(value="abc").value == "abc"
    assert args_model(value=123).value == 123


def test_build_args_model_handles_multi_value_type_list():
    mcp_tool = SimpleNamespace(
        name="multi_value_type_tool",
        description="Accept string or integer input",
        inputSchema={
            "type": "object",
            "properties": {"value": {"type": ["string", "integer", "null"]}},
            "required": ["value"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    args_model = adapter.args_type()

    assert args_model(value="abc").value == "abc"
    assert args_model(value=123).value == 123


@pytest.mark.asyncio
async def test_runtime_bindings_hide_and_inject_mcp_meta_and_tool_arguments(
    monkeypatch,
):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "account_id": {"type": "string"},
            },
            "required": ["query", "account_id"],
        },
    )
    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "runtime_bindings": [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "tool_arguments", "key": "account_id"},
            },
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "mcp_meta", "key": "account_id"},
            },
        ],
        "connector_runtime": {
            "context": {"account_id": "6185"},
            "secrets": {},
            "auth_selector": {},
        },
    }
    adapter = MCPToolAdapter(mcp_tool=mcp_tool, connection=connection)
    captured = {}

    class _FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            captured["name"] = name
            captured["arguments"] = arguments
            captured["kwargs"] = kwargs
            return CallToolResult(content=[], isError=False)

    @asynccontextmanager
    async def _fake_create_session(_connection):
        yield _FakeSession()

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    args_model = adapter.args_type()
    assert "account_id" not in args_model.model_fields

    result = await adapter.run_json_async(
        {"query": "active", "account_id": "llm-supplied"}
    )

    assert result["is_error"] is False
    assert captured["name"] == "list_clients"
    assert captured["arguments"] == {"query": "active", "account_id": "6185"}
    assert captured["kwargs"]["meta"] == {"account_id": "6185"}


def test_mcp_runtime_tool_argument_missing_source_warns(caplog):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={
            "transport": "stdio",
            "command": "python",
            "args": [],
            "runtime_bindings": [
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {"target_type": "tool_arguments", "key": "account_id"},
                },
            ],
            "connector_runtime": {"context": {}, "secrets": {}, "auth_selector": {}},
        },
    )

    caplog.set_level("WARNING")
    assert adapter._runtime_tool_arguments() == {}
    assert (
        "Skipping runtime MCP tool argument binding for missing context source"
        in caplog.text
    )
    assert "account_id" in caplog.text
    assert "list_clients" in caplog.text


def _resolver_retry_adapter(connection):
    return MCPToolAdapter(
        mcp_tool=SimpleNamespace(
            name="list_clients",
            description="List clients",
            inputSchema={"type": "object", "properties": {}},
        ),
        connection=connection,
    )


@pytest.mark.asyncio
async def test_resolver_retry_analyzes_initial_401_chain_once(monkeypatch):
    from xagent.core.agent.result import ClassifiedToolFailure

    strict_401_calls = 0
    strict_401_responses = mcp_adapter_module._strict_http_401_responses

    def counting_strict_401_responses(exc, **kwargs):
        nonlocal strict_401_calls
        strict_401_calls += 1
        yield from strict_401_responses(exc, **kwargs)

    async def _resolver_refresh(challenge):
        return ClassifiedToolFailure(failure_code="oauth_token_required")

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    monkeypatch.setattr(
        mcp_adapter_module,
        "_strict_http_401_responses",
        counting_strict_401_responses,
    )

    result = await adapter._retry_resolver_401(
        _http_status_error(authenticate=['Bearer error="invalid_token"']),
        {},
        {},
    )

    assert strict_401_calls == 1
    assert result is not None
    assert result["failure_code"] == "oauth_token_required"


@pytest.mark.asyncio
async def test_resolver_401_has_priority_and_retries_rebuilt_connection_once(
    monkeypatch,
):
    resolver_calls = []
    connector_calls = 0
    fresh_connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": "Bearer fresh-token"},
    }

    async def _resolver_refresh(challenge):
        resolver_calls.append(challenge)
        return fresh_connection

    def _connector_refresh():
        nonlocal connector_calls
        connector_calls += 1
        return fresh_connection

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "_oauth_token_resolver_refresh": _resolver_refresh,
        "_connector_runtime_refresh": _connector_refresh,
    }
    adapter = _resolver_retry_adapter(connection)
    attempted_connections = []

    async def _execute(attempted, tool_args, tool_meta):
        attempted_connections.append(attempted)
        if attempted is connection:
            raise _http_status_error(
                authenticate=['Bearer error="invalid_token", scope="records.read"']
            )
        return {"content": [{"text": "ok"}], "is_error": False}

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert result == {"content": [{"text": "ok"}], "is_error": False}
    assert len(resolver_calls) == 1
    assert resolver_calls[0].params["error"] == "invalid_token"
    assert connector_calls == 0
    assert attempted_connections == [connection, fresh_connection]


@pytest.mark.asyncio
async def test_real_mcp_session_retries_nested_resolver_401_once(monkeypatch, caplog):
    initial_token = "real-initial-access-token-secret"
    refreshed_token = "real-refreshed-access-token-secret"
    generation = "real-resolver-generation-secret"
    raw_exception_secret = "real-http-status-error-secret"
    tool_args = {"query": "active", "account_id": "6185"}
    tool_meta = {"account_id": "6185"}
    initial_requests = []
    refreshed_requests = []
    initial_client_builds = 0
    refreshed_client_builds = 0
    refresh_calls = []

    async def _initial_handler(request):
        payload = json.loads(request.content) if request.content else {}
        initial_requests.append((request, payload))
        assert payload["method"] == "initialize"
        return httpx.Response(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer error="invalid_token", scope="records.read"'
                )
            },
            extensions={"reason_phrase": raw_exception_secret.encode()},
            request=request,
        )

    async def _refreshed_handler(request):
        payload = json.loads(request.content) if request.content else {}
        refreshed_requests.append((request, payload))
        method = payload.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-mcp", "version": "1"},
            }
        elif method == "tools/call":
            result = {
                "content": [{"type": "text", "text": "clients-ok"}],
                "isError": False,
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "list_clients",
                        "description": "List clients",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }
        else:
            return httpx.Response(405 if request.method == "GET" else 202)
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "refreshed-session",
            },
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            request=request,
        )

    def _initial_client_factory(headers=None, timeout=None, auth=None):
        nonlocal initial_client_builds
        initial_client_builds += 1
        return httpx.AsyncClient(
            transport=httpx.MockTransport(_initial_handler),
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    def _refreshed_client_factory(headers=None, timeout=None, auth=None):
        nonlocal refreshed_client_builds
        refreshed_client_builds += 1
        return httpx.AsyncClient(
            transport=httpx.MockTransport(_refreshed_handler),
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    async def _resolver_refresh(challenge):
        refresh_calls.append((challenge, generation))
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": f"Bearer {refreshed_token}"},
            "httpx_client_factory": _refreshed_client_factory,
            "terminate_on_close": False,
        }

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": f"Bearer {initial_token}"},
        "httpx_client_factory": _initial_client_factory,
        "terminate_on_close": False,
        "runtime_bindings": [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "tool_arguments", "key": "account_id"},
            },
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "mcp_meta", "key": "account_id"},
            },
        ],
        "connector_runtime": {
            "context": {"account_id": "6185"},
            "secrets": {},
            "auth_selector": {},
        },
        "_oauth_token_resolver_refresh": _resolver_refresh,
    }
    adapter = MCPToolAdapter(
        mcp_tool=SimpleNamespace(
            name="list_clients",
            description="List clients",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "account_id": {"type": "string"},
                },
                "required": ["query", "account_id"],
            },
        ),
        connection=connection,
    )
    nested_failures = []
    retry = adapter._retry_after_authorization_failure

    async def _observe_nested_failure(exc, attempted_args, attempted_meta):
        nested_failures.append(exc)
        responses = list(mcp_adapter_module._strict_http_401_responses(exc))
        assert len(responses) == 1
        assert responses[0].headers.get_list("WWW-Authenticate") == [
            'Bearer error="invalid_token", scope="records.read"'
        ]
        return await retry(exc, attempted_args, attempted_meta)

    monkeypatch.setattr(
        adapter, "_retry_after_authorization_failure", _observe_nested_failure
    )
    caplog.set_level("DEBUG")

    result = await adapter.run_json_async(
        {"query": "active", "account_id": "llm-supplied"}
    )

    assert result == {
        "content": [
            {
                "type": "text",
                "text": "clients-ok",
                "annotations": None,
                "meta": None,
            }
        ],
        "structured_content": None,
        "is_error": False,
    }
    assert initial_client_builds == 1
    assert refreshed_client_builds == 1
    assert len(refresh_calls) == 1
    assert refresh_calls[0][0].params["error"] == "invalid_token"
    assert refresh_calls[0][0].scope == "records.read"
    assert len(nested_failures) == 1
    assert raw_exception_secret in repr(nested_failures[0])
    assert len(initial_requests) == 1
    assert initial_requests[0][0].headers["Authorization"] == (
        f"Bearer {initial_token}"
    )
    tool_calls = [
        (request, payload)
        for request, payload in refreshed_requests
        if payload.get("method") == "tools/call"
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0][0].headers["Authorization"] == f"Bearer {refreshed_token}"
    assert tool_calls[0][1]["params"] == {
        "_meta": tool_meta,
        "name": "list_clients",
        "arguments": tool_args,
    }
    public_output = repr(result) + caplog.text
    assert initial_token not in public_output
    assert refreshed_token not in public_output
    assert generation not in public_output
    assert raw_exception_secret not in public_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authenticate",
    [
        [],
        ['Bearer error="invalid_token'],
        ['Basic error="invalid_token"'],
        ['Bearer error="insufficient_scope"'],
        ['Bearer scope="records.read"'],
    ],
)
async def test_resolver_owned_invalid_401_challenge_fails_without_connector_fallback(
    monkeypatch, authenticate
):
    resolver_calls = 0
    connector_calls = 0

    def _resolver_refresh(challenge):
        nonlocal resolver_calls
        resolver_calls += 1
        return {}

    def _connector_refresh():
        nonlocal connector_calls
        connector_calls += 1
        return {}

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
            "_connector_runtime_refresh": _connector_refresh,
        }
    )

    async def _execute(connection, tool_args, tool_meta):
        raise _http_status_error(authenticate=authenticate)

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]
    assert resolver_calls == 0
    assert connector_calls == 0


@pytest.mark.asyncio
async def test_resolver_owned_text_only_401_does_not_use_either_refresh_callback(
    monkeypatch,
):
    resolver_calls = 0
    connector_calls = 0

    def _resolver_refresh(challenge):
        nonlocal resolver_calls
        resolver_calls += 1
        return {}

    def _connector_refresh():
        nonlocal connector_calls
        connector_calls += 1
        return {}

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
            "_connector_runtime_refresh": _connector_refresh,
        }
    )

    async def _execute(connection, tool_args, tool_meta):
        raise RuntimeError("HTTP 401 Unauthorized; Bearer error=invalid_token")

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert result == {
        "content": [{"text": "Error executing MCP tool."}],
        "is_error": True,
    }
    assert resolver_calls == 0
    assert connector_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_behavior", ["none", "raise", "malformed"])
async def test_resolver_refresh_failure_is_fixed_and_sanitized(
    monkeypatch, caplog, refresh_behavior
):
    secret = f"resolver-{refresh_behavior}-secret"

    async def _resolver_refresh(challenge):
        if refresh_behavior == "raise":
            raise RuntimeError(secret)
        if refresh_behavior == "malformed":
            return SimpleNamespace(secret=secret)
        return None

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )

    async def _execute(connection, tool_args, tool_meta):
        raise _http_status_error(authenticate=['Bearer error="invalid_token"'])

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)
    caplog.set_level("ERROR")

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]
    assert secret not in repr(result) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refreshed_connection",
    [
        {},
        {"transport": "unsupported", "url": "https://mcp.example.test"},
        {"transport": "stdio"},
        {"transport": "stdio", "command": ""},
        {"transport": "stdio", "command": ["python"]},
        {"transport": "stdio", "command": "python"},
        {"transport": "sse"},
        {"transport": "streamable_http"},
        {"transport": "streamable_http", "url": ""},
        {"transport": "streamable_http", "url": 42},
        {"transport": "websocket"},
    ],
)
async def test_resolver_refresh_rejects_non_executable_connection_before_retry(
    monkeypatch, refreshed_connection
):
    refresh_calls = 0

    async def _resolver_refresh(challenge):
        nonlocal refresh_calls
        refresh_calls += 1
        return refreshed_connection

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    execution_calls = 0

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        raise _http_status_error(authenticate=['Bearer error="invalid_token"'])

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert refresh_calls == 1
    assert execution_calls == 1
    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_resolver_retry_second_401_does_not_refresh_twice(monkeypatch):
    refresh_calls = 0

    async def _resolver_refresh(challenge):
        nonlocal refresh_calls
        refresh_calls += 1
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    execution_calls = 0

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        raise _http_status_error(authenticate=['Bearer error="invalid_token"'])

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert execution_calls == 2
    assert refresh_calls == 1
    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_resolver_retry_classifies_same_401_instance_without_second_refresh(
    monkeypatch,
):
    refresh_calls = 0

    async def _resolver_refresh(challenge):
        nonlocal refresh_calls
        refresh_calls += 1
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    execution_calls = 0
    repeated_error = _http_status_error(authenticate=['Bearer error="invalid_token"'])

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        raise repeated_error

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert execution_calls == 2
    assert refresh_calls == 1
    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_resolver_retry_ignores_rewrapped_initial_401_response(monkeypatch):
    async def _resolver_refresh(challenge):
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    initial_error = _http_status_error(authenticate=['Bearer error="invalid_token"'])
    rewrapped_initial_response = httpx.HTTPStatusError(
        "rewrapped initial response",
        request=initial_error.request,
        response=initial_error.response,
    )
    execution_calls = 0

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        if execution_calls == 1:
            raise initial_error
        raise rewrapped_initial_response

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert execution_calls == 2
    assert result == {
        "content": [
            {"text": "Error executing MCP tool after delegated authorization retry."}
        ],
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_resolver_retry_prunes_over_budget_initial_exception_subtree(
    monkeypatch,
):
    async def _resolver_refresh(challenge):
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }

    adapter = _resolver_retry_adapter(
        {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "_oauth_token_resolver_refresh": _resolver_refresh,
        }
    )
    deep_original: BaseException = _http_status_error(
        authenticate=['Bearer error="invalid_token"']
    )
    for index in range(mcp_adapter_module._RESOLVER_HTTP_401_NODE_LIMIT + 1):
        wrapper = RuntimeError(f"initial-wrapper-{index}")
        wrapper.__cause__ = deep_original
        deep_original = wrapper
    initial_error = ExceptionGroup(
        "initial",
        [
            _http_status_error(authenticate=['Bearer error="invalid_token"']),
            deep_original,
        ],
    )
    execution_calls = 0

    async def _execute(connection, tool_args, tool_meta):
        nonlocal execution_calls
        execution_calls += 1
        if execution_calls == 1:
            raise initial_error
        raise RuntimeError("retry transport failed")

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert execution_calls == 2
    assert result == {
        "content": [
            {"text": "Error executing MCP tool after delegated authorization retry."}
        ],
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_resolver_retry_non_401_failure_does_not_leak_secrets(
    monkeypatch, caplog
):
    initial_secret = "initial-resolver-secret"
    refreshed_secret = "refreshed-resolver-secret"

    async def _resolver_refresh(challenge):
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": f"Bearer {refreshed_secret}"},
        }

    initial_connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": f"Bearer {initial_secret}"},
        "_oauth_token_resolver_refresh": _resolver_refresh,
    }
    adapter = _resolver_retry_adapter(initial_connection)

    async def _execute(connection, tool_args, tool_meta):
        if connection is initial_connection:
            raise _http_status_error(authenticate=['Bearer error="invalid_token"'])
        raise RuntimeError(f"transport failed with {refreshed_secret}")

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)
    caplog.set_level("ERROR")

    result = await adapter.run_json_async({})

    assert result == {
        "content": [
            {"text": "Error executing MCP tool after delegated authorization retry."}
        ],
        "is_error": True,
    }
    public_output = repr(result) + caplog.text
    assert initial_secret not in public_output
    assert refreshed_secret not in public_output


@pytest.mark.asyncio
async def test_connector_refresh_empty_dict_preserves_legacy_retry_failure(
    monkeypatch,
):
    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "_connector_runtime_refresh": lambda: {},
    }
    adapter = _resolver_retry_adapter(connection)
    attempted_connections = []

    async def _execute(attempted, tool_args, tool_meta):
        attempted_connections.append(attempted)
        if attempted is connection:
            raise RuntimeError("HTTP 401 Unauthorized")
        raise RuntimeError("malformed connector retry connection")

    monkeypatch.setattr(adapter, "_execute_mcp_call", _execute)

    result = await adapter.run_json_async({})

    assert attempted_connections == [connection, {}]
    assert result == {
        "content": [
            {"text": "Error executing MCP tool after delegated authorization retry."}
        ],
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_delegated_authorization_401_refreshes_connection_once(monkeypatch):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )
    connections = []
    refresh_calls = 0

    def _refresh_connection():
        nonlocal refresh_calls
        refresh_calls += 1
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer fresh-token"},
        }

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": "Bearer expired-token"},
        "_connector_runtime_refresh": _refresh_connection,
    }
    adapter = MCPToolAdapter(mcp_tool=mcp_tool, connection=connection)

    class _FakeSession:
        def __init__(self, connection):
            self._connection = connection

        async def initialize(self):
            if self._connection["headers"]["Authorization"] == "Bearer expired-token":
                raise RuntimeError("HTTP 401 Unauthorized")

        async def call_tool(self, name, arguments, **kwargs):
            return CallToolResult(
                content=[TextContent(type="text", text="ok")],
                isError=False,
            )

    @asynccontextmanager
    async def _fake_create_session(connection):
        connections.append(connection)
        yield _FakeSession(connection)

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    result = await adapter.run_json_async({})

    assert result == {
        "content": [
            {
                "type": "text",
                "text": "ok",
                "annotations": None,
                "meta": None,
            }
        ],
        "structured_content": None,
        "is_error": False,
    }
    assert refresh_calls == 1
    assert [item["headers"]["Authorization"] for item in connections] == [
        "Bearer expired-token",
        "Bearer fresh-token",
    ]


@pytest.mark.asyncio
async def test_delegated_authorization_401_refresh_classifies_team_hook_failure():
    """The delegated (non-resolver) refresh branch calls the real per-tool-call
    refresh callback -- ``_refresh_delegated_mcp_connection_from_snapshot``,
    the function ``WebToolConfig`` binds into a connection's
    ``_connector_runtime_refresh`` slot -- against a real, DB-backed team
    hook that raises. Before this branch had its own try/except (mirroring
    the sibling resolver-refresh branch at ``_retry_resolver_401``), the
    raised exception propagated out of ``_retry_after_authorization_failure``
    uncaught instead of surfacing as the same classified failure result the
    resolver branch already returns.
    """
    from functools import partial

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from xagent.web.models import Base, MCPServer, Task, User
    from xagent.web.services import connector_team_scope
    from xagent.web.tools.config import _refresh_delegated_mcp_connection_from_snapshot

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as seed_db:
        owner = User(username="mcp-refresh-owner", password_hash="hash")
        seed_db.add(owner)
        seed_db.flush()
        server = MCPServer(
            name="team-refresh-probe",
            managed="external",
            transport="streamable_http",
            url="https://mcp.example.test",
        )
        seed_db.add(server)
        seed_db.flush()
        task = Task(
            user_id=owner.id,
            title="mcp refresh probe task",
            # Non-empty so ``load_connector_runtime_view`` proceeds past its
            # early-return and reaches the team hook.
            connector_runtime_selected_refs=[
                {"connector_type": "mcp", "connector_id": int(server.id)}
            ],
        )
        seed_db.add(task)
        seed_db.commit()
        owner_id = int(owner.id)
        server_id = int(server.id)
        task_id = int(task.id)

    def _raising_team_hook(db, *, team_id):
        raise RuntimeError("Bearer planted-refresh-secret-must-not-leak")

    connector_team_scope.set_connector_team_hooks(team_visibility=_raising_team_hook)
    try:
        refresh = partial(
            _refresh_delegated_mcp_connection_from_snapshot,
            session_factory=session_factory,
            task_id=task_id,
            turn_id=None,
            user_id=owner_id,
            server_id=server_id,
            connection_snapshot={},
            runtime_bindings=[],
            allow_delegated_authorization=True,
            agent_team_id=101,
        )
        connection = {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer expired-token"},
            mcp_adapter_module._RUNTIME_CONNECTION_REFRESH_KEY: refresh,
        }
        adapter = MCPToolAdapter(
            mcp_tool=_mcp_tool("list_clients"), connection=connection
        )

        result = await adapter._retry_after_authorization_failure(
            _http_status_error(), {}, {}
        )
    finally:
        connector_team_scope.set_connector_team_hooks()

    assert result is not None
    assert result["is_error"] is True
    assert "delegated authorization failed" in result["content"][0]["text"]
    # The raw hook message never reaches the caller -- only the classified,
    # public-safe failure text does.
    assert "planted-refresh-secret-must-not-leak" not in json.dumps(result)


@pytest.mark.asyncio
async def test_delegated_authorization_401_with_non_mapping_connection_does_not_crash(
    monkeypatch,
):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection=SimpleNamespace(transport="streamable_http"),
    )

    class _FakeSession:
        async def initialize(self):
            raise RuntimeError("HTTP 401 Unauthorized")

    @asynccontextmanager
    async def _fake_create_session(connection):
        yield _FakeSession()

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert result["content"][0]["text"] == "Error executing MCP tool."
    assert "AttributeError" not in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_tool_execution_error_does_not_echo_raw_exception(monkeypatch):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "streamable_http", "url": "https://mcp.example.test"},
    )

    class _FakeSession:
        async def initialize(self):
            raise RuntimeError("transport failed with Bearer runtime-token")

    @asynccontextmanager
    async def _fake_create_session(connection):
        yield _FakeSession()

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    result = await adapter.run_json_async({})

    assert result["is_error"] is True
    assert result["content"][0]["text"] == "Error executing MCP tool."
    assert "runtime-token" not in repr(result)


@pytest.mark.asyncio
async def test_delegated_authorization_401_after_refresh_returns_safe_error(
    monkeypatch,
):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )
    refresh_calls = 0

    def _refresh_connection():
        nonlocal refresh_calls
        refresh_calls += 1
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer still-expired-token"},
        }

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": "Bearer expired-token"},
        "_connector_runtime_refresh": _refresh_connection,
    }
    adapter = MCPToolAdapter(mcp_tool=mcp_tool, connection=connection)

    class _FakeSession:
        async def initialize(self):
            raise RuntimeError("HTTP 401 Unauthorized")

    @asynccontextmanager
    async def _fake_create_session(connection):
        yield _FakeSession()

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )

    result = await adapter.run_json_async({})

    assert refresh_calls == 1
    assert result["is_error"] is True
    assert "delegated_authorization_failed" in result["content"][0]["text"]
    assert "expired-token" not in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_delegated_authorization_retry_failure_does_not_leak_token(
    monkeypatch,
    caplog,
):
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={"type": "object", "properties": {}},
    )

    def _refresh_connection():
        return {
            "transport": "streamable_http",
            "url": "https://mcp.example.test",
            "headers": {"Authorization": "Bearer fresh-runtime-token"},
        }

    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "headers": {"Authorization": "Bearer expired-token"},
        "_connector_runtime_refresh": _refresh_connection,
    }
    adapter = MCPToolAdapter(mcp_tool=mcp_tool, connection=connection)
    calls = 0

    class _FakeSession:
        def __init__(self, connection):
            self._connection = connection

        async def initialize(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("HTTP 401 Unauthorized")
            raise RuntimeError(
                f"transport failed with {self._connection['headers']['Authorization']}"
            )

    @asynccontextmanager
    async def _fake_create_session(connection):
        yield _FakeSession(connection)

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.create_session",
        _fake_create_session,
    )
    caplog.set_level("ERROR")

    result = await adapter.run_json_async({})

    assert calls == 2
    assert result["is_error"] is True
    assert result["content"][0]["text"] == (
        "Error executing MCP tool after delegated authorization retry."
    )
    public_output = repr(result) + caplog.text
    assert "fresh-runtime-token" not in public_output
    assert "expired-token" not in public_output


def _schema_adapter(properties, required=None, *, tool_name="probe_tool"):
    """Build an adapter over a hand-written MCP input schema."""
    mcp_tool = SimpleNamespace(
        name=tool_name,
        description="probe tool",
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": list(required or []),
        },
    )
    return MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )


def _emitted_schema(properties, required=None, *, tool_name="probe_tool"):
    adapter = _schema_adapter(properties, required, tool_name=tool_name)
    return adapter.args_type().model_json_schema()


def _emitted_field(properties, required=None, *, field="f"):
    return _emitted_schema(properties, required)["properties"][field]


def _metadata_carrier(field):
    """The subschema of one emitted field that carries its metadata.

    An optional field is emitted as an ``anyOf`` of its declared type and
    ``null``, and its metadata sits on the declared-type branch so that a
    consumer resolving the wrapper down to that branch keeps it. A required
    field has no wrapper and carries its metadata directly.
    """
    options = field.get("anyOf")
    if options is None:
        return field
    non_null = [option for option in options if option.get("type") != "null"]
    assert len(non_null) == 1, options
    return non_null[0]


def _emitted_metadata(properties, required=None, *, field="f"):
    return _metadata_carrier(_emitted_field(properties, required, field=field))


@pytest.mark.parametrize("is_required", [True, False])
def test_field_description_reaches_emitted_schema(is_required):
    properties = {"f": {"type": "string", "description": "The city to look up."}}
    carrier = _emitted_metadata(properties, ["f"] if is_required else [])

    assert carrier["description"] == "The city to look up."


@pytest.mark.parametrize(
    "field_schema,key,expected",
    [
        (
            {"type": "string", "enum": ["sydney", "melbourne"]},
            "enum",
            ["sydney", "melbourne"],
        ),
        ({"type": "string", "pattern": "^[a-z]+$"}, "pattern", "^[a-z]+$"),
        ({"type": "string", "format": "date-time"}, "format", "date-time"),
        ({"type": "string", "minLength": 2}, "minLength", 2),
        ({"type": "string", "maxLength": 40}, "maxLength", 40),
        ({"type": "integer", "minimum": 1}, "minimum", 1),
        ({"type": "integer", "maximum": 14}, "maximum", 14),
    ],
)
def test_field_constraint_keys_reach_emitted_schema(field_schema, key, expected):
    field = _emitted_field({"f": field_schema}, ["f"])

    assert field[key] == expected


@pytest.mark.parametrize(
    "field_schema,violating_value",
    [
        ({"type": "string", "pattern": "^[a-z]+$"}, "123-NOT-MATCHING"),
        ({"type": "string", "maxLength": 2}, "far beyond the stated maximum length"),
        ({"type": "string", "minLength": 20}, "short"),
        ({"type": "integer", "minimum": 10}, 1),
        ({"type": "string", "enum": ["metric", "imperial"]}, "furlongs"),
        ({"type": "string", "format": "email"}, "not-an-email-address"),
    ],
)
def test_preserved_constraints_do_not_enforce_validation(field_schema, violating_value):
    args_model = _schema_adapter({"f": field_schema}, ["f"]).args_type()

    assert args_model(f=violating_value).f == violating_value


def test_overlong_field_description_is_bounded():
    raw = "  Sydney   weather lookup.\n\n" + "Extra guidance for the model. " * 40
    field = _emitted_field({"f": {"type": "string", "description": raw}}, ["f"])
    description = field["description"]
    collapsed = " ".join(raw.split())

    assert len(description) <= _FIELD_TEXT_MAX_CHARS
    assert description.endswith("…")
    assert collapsed.startswith(description[:-1])


def test_enum_is_all_or_nothing():
    """An emitted enum is the author's whole list; an over-long one is dropped.

    A shortened list would still read as the complete set of legal values, so
    it would tell the model a value the server accepts is illegal. The kept
    case carries enough members that dropping any of them is visible.
    """
    members = [f"v{index:02d}" for index in range(12)]
    assert len(_compact_json(members)) <= _FIELD_TEXT_MAX_CHARS
    field = _emitted_field({"f": {"type": "string", "enum": members}}, ["f"])
    assert field["enum"] == members

    oversized = [f"value-{index:04d}" for index in range(200)]
    field = _emitted_field({"f": {"type": "string", "enum": oversized}}, ["f"])
    assert "enum" not in field


def test_runtime_bound_field_still_excluded_with_metadata():
    mcp_tool = SimpleNamespace(
        name="list_clients",
        description="List clients",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text."},
                "account_id": {
                    "type": "string",
                    "description": "Tenant account identifier.",
                    "pattern": "^[0-9]+$",
                    "format": "uuid",
                },
            },
            "required": ["query", "account_id"],
        },
    )
    connection = {
        "transport": "streamable_http",
        "url": "https://mcp.example.test",
        "runtime_bindings": [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "tool_arguments", "key": "account_id"},
            }
        ],
        "connector_runtime": {
            "context": {"account_id": "6185"},
            "secrets": {},
            "auth_selector": {},
        },
    }
    adapter = MCPToolAdapter(mcp_tool=mcp_tool, connection=connection)
    args_model = adapter.args_type()
    schema = args_model.model_json_schema()

    assert "account_id" not in args_model.model_fields
    assert "account_id" not in schema["properties"]
    assert schema["properties"]["query"]["description"] == "Search text."


@pytest.mark.parametrize(
    "input_schema",
    [None, {}, {"properties": {}}, "not-a-dict"],
)
def test_empty_schema_paths_unchanged(input_schema):
    mcp_tool = SimpleNamespace(
        name="probe_tool", description="probe tool", inputSchema=input_schema
    )
    adapter = MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection={"transport": "stdio", "command": "python", "args": []},
    )

    assert adapter.args_type() is EmptyArgsModel


@pytest.mark.parametrize(
    "field_schema,key",
    [
        ({"type": "string", "description": 123}, "description"),
        ({"type": "string", "enum": {}}, "enum"),
        ({"type": "string", "enum": []}, "enum"),
        ({"type": "integer", "minimum": True}, "minimum"),
        ({"type": "string", "pattern": []}, "pattern"),
        ({"type": "string", "enum": ["ok", object()]}, "enum"),
    ],
)
def test_malformed_field_schema_values_are_not_emitted(field_schema, key):
    schema = _emitted_schema({"f": field_schema}, ["f"])

    assert "f" in schema["properties"]
    assert key not in schema["properties"]["f"]


@pytest.mark.parametrize("is_required", [True, False])
def test_a_field_schema_that_is_not_a_mapping_costs_only_itself(is_required):
    """One unreadable field schema does not cost the tool its other arguments.

    A property whose schema is not a mapping declares nothing this adapter can
    read, a default included. It still becomes a field, and the field beside
    it keeps the metadata its own schema declared.
    """
    schema = _emitted_schema(
        {"broken": 5, "kept": {"type": "string", "description": "Kept."}},
        ["broken", "kept"] if is_required else [],
    )

    assert set(schema["properties"]) == {"broken", "kept"}
    assert _metadata_carrier(schema["properties"]["kept"])["description"] == "Kept."


@pytest.mark.parametrize("is_required", [True, False])
def test_non_ascii_metadata_reaches_the_emitted_schema(is_required):
    """Metadata is carried as text, and the cap counts characters, not bytes.

    The emitted schema is serialized with ``ensure_ascii=False``, so text
    outside ASCII reaches the model as itself. A capped description of CJK
    characters is three times the cap in UTF-8 bytes, which is the point:
    the cap counts what the author wrote, not how it encodes.
    """
    long_description = "西" * (_FIELD_TEXT_MAX_CHARS + 50)
    carrier = _emitted_metadata(
        {
            "f": {
                "type": "string",
                "description": long_description,
                "enum": ["公制", "英制"],
                "pattern": "^[一-鿿]+$",
            }
        },
        ["f"] if is_required else [],
    )

    assert len(carrier["description"]) == _FIELD_TEXT_MAX_CHARS
    assert carrier["description"] == "西" * (_FIELD_TEXT_MAX_CHARS - 1) + "…"
    assert len(carrier["description"].encode("utf-8")) > _FIELD_TEXT_MAX_CHARS
    assert carrier["enum"] == ["公制", "英制"]
    assert carrier["pattern"] == "^[一-鿿]+$"


@pytest.mark.parametrize("is_required", [True, False])
def test_field_without_metadata_matches_baseline_shape(is_required):
    emitted = _emitted_schema({"f": {"type": "string"}}, ["f"] if is_required else [])
    if is_required:
        baseline = create_model("ProbeToolArgs", f=(str, ...))
    else:
        baseline = create_model("ProbeToolArgs", f=(Optional[str], None))

    assert emitted == baseline.model_json_schema()


def test_mcp_adapter_pulls_in_no_agent_modules():
    probe = (
        "import sys; "
        "import xagent.core.tools.adapters.vibe.mcp_adapter; "
        "print(len([n for n in sys.modules "
        "if n.startswith('xagent.core.agent')]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "0"


@pytest.mark.parametrize("key", ["minLength", "maxLength", "minimum", "maximum"])
@pytest.mark.parametrize("non_finite", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_numeric_constraints_are_not_emitted(key, non_finite):
    schema = _emitted_schema({"f": {"type": "number", key: non_finite}}, ["f"])

    assert key not in schema["properties"]["f"]
    _compact_json(schema)


@pytest.mark.parametrize("non_finite", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_default_is_not_emitted(non_finite):
    schema = _emitted_schema({"f": {"type": "number", "default": non_finite}})

    assert schema["properties"]["f"]["default"] is None
    _compact_json(schema)


@pytest.mark.parametrize("declared", ["metric", 7, False, ["a"], {"k": 1}])
def test_an_authored_default_is_emitted_as_written(declared):
    """A usable default the author wrote reaches the schema unchanged."""
    schema = _emitted_schema({"f": {"type": "string", "default": declared}})

    assert schema["properties"]["f"]["default"] == declared


def test_enum_containing_non_finite_number_is_not_emitted():
    schema = _emitted_schema(
        {"f": {"type": "number", "enum": [1, float("inf")]}}, ["f"]
    )

    assert "enum" not in schema["properties"]["f"]
    _compact_json(schema)


@pytest.mark.parametrize(
    "field_schema,enum_expected",
    [
        # No declared default: the emitted null is this adapter's, so the
        # author's enum stands.
        ({"type": "string", "enum": ["metric", "imperial"]}, True),
        ({"type": "string", "enum": ["metric", "imperial"], "default": "metric"}, True),
        ({"type": "string", "enum": ["metric", "imperial"], "default": "zzz"}, False),
        # JSON keeps booleans and numbers apart even where Python does not.
        ({"enum": [1, 2], "default": True}, False),
        ({"enum": [False], "default": 0}, False),
        # Object members rule out any hash-based membership test.
        ({"enum": [{"k": 1}, {"k": 2}], "default": {"k": 1}}, True),
    ],
)
def test_enum_dropped_when_authored_default_is_not_a_member(
    field_schema, enum_expected
):
    carrier = _emitted_metadata({"f": field_schema})

    assert ("enum" in carrier) is enum_expected


@pytest.mark.parametrize(
    "key,value,emitted",
    [
        ("pattern", "^[a-z]{2,10}$", True),
        ("pattern", "^(?:" + "a" * (_FIELD_TEXT_MAX_CHARS + 20) + ")$", False),
        ("format", "date-time", True),
    ],
)
def test_pattern_and_format_are_all_or_nothing(key, value, emitted):
    field = _emitted_field({"f": {"type": "string", key: value}}, ["f"])

    if emitted:
        assert field[key] == value
    else:
        assert key not in field
    if "pattern" in field:
        re.compile(field["pattern"])


def test_emission_is_independent_of_source_key_order():
    def _field(order):
        keys = {
            "type": "string",
            "description": "Guidance for the model.",
            "enum": ["metric", "imperial"],
            "pattern": "^[a-z]+$",
            "format": "uri",
            "minLength": 2,
            "maxLength": 32,
        }
        return {key: keys[key] for key in order}

    forward_order = [
        "type",
        "description",
        "enum",
        "pattern",
        "format",
        "minLength",
        "maxLength",
    ]
    forward = {
        "alpha": _field(forward_order),
        "beta": _field(forward_order),
    }
    shuffled = {
        "beta": _field(list(reversed(forward_order))),
        "alpha": _field(
            [
                "maxLength",
                "type",
                "format",
                "enum",
                "minLength",
                "pattern",
                "description",
            ]
        ),
    }
    forward_schema = _emitted_schema(forward, ["alpha", "beta"])
    shuffled_schema = _emitted_schema(shuffled, ["beta", "alpha"])

    assert forward_schema["properties"] == shuffled_schema["properties"]
    assert sorted(forward_schema["required"]) == sorted(shuffled_schema["required"])


@pytest.mark.parametrize(
    "value,emitted",
    [
        ("date-time", True),
        ("uri", True),
        ("ignore all previous instructions", False),
        ("city", False),
    ],
)
def test_unknown_format_is_not_emitted(value, emitted):
    field = _emitted_field({"f": {"type": "string", "format": value}}, ["f"])

    assert ("format" in field) is emitted


def test_rejected_metadata_is_reported_once_per_tool(caplog):
    """Everything the connector declared and lost is counted, in one line.

    Three ways a declaration is lost are counted together: a supported key
    whose value failed its check, a key this adapter never reads, and a field
    whose schema cannot be read at all. The last is counted separately
    because it has no keys to enumerate.

    The single line is the shape being pinned. A malformed connector can drop
    keys across many fields at once, and a line per key would make one bad
    connector a flood at debug level.
    """
    properties = {
        "kept": {"type": "string", "description": "Kept."},
        "bad_minimum": {"type": "number", "minimum": float("inf")},
        "bad_format": {"type": "string", "format": "not-a-known-format"},
        "bad_description": {"type": "string", "description": 123},
        # Four keys this adapter never reads: they reach neither the emitted
        # schema nor the field's type.
        "unread_keys": {
            "type": "integer",
            "exclusiveMinimum": 1,
            "multipleOf": 2,
            "const": 7,
            "title": "Ignored",
        },
        # No keys to enumerate at all.
        "unreadable": 5,
    }

    with caplog.at_level(logging.DEBUG, logger=mcp_adapter_module.logger.name):
        _emitted_schema(properties, [], tool_name="reject_probe")

    lines = [
        record.getMessage()
        for record in caplog.records
        if "field schema metadata keys" in record.getMessage()
    ]
    assert len(lines) == 1
    assert lines[0] == (
        "MCP tool reject_probe dropped 7 field schema metadata keys "
        "and 1 unreadable field schemas"
    )


def test_an_unreadable_field_schema_is_reported_on_its_own_count(caplog):
    """A field with no readable schema is reported without inventing a key count."""
    with caplog.at_level(logging.DEBUG, logger=mcp_adapter_module.logger.name):
        _emitted_schema({"unreadable": 5}, [], tool_name="unreadable_probe")

    lines = [
        record.getMessage()
        for record in caplog.records
        if "field schema metadata keys" in record.getMessage()
    ]
    assert lines == [
        "MCP tool unreadable_probe dropped 0 field schema metadata keys "
        "and 1 unreadable field schemas"
    ]


def test_no_report_when_every_metadata_key_is_kept(caplog):
    """A tool whose metadata all survives says nothing."""
    with caplog.at_level(logging.DEBUG, logger=mcp_adapter_module.logger.name):
        _emitted_schema(
            {"f": {"type": "string", "description": "Kept.", "format": "uri"}},
            ["f"],
            tool_name="quiet_probe",
        )

    assert not [
        record
        for record in caplog.records
        if "field schema metadata keys" in record.getMessage()
    ]


@pytest.mark.parametrize("is_required", [True, False])
def test_enum_survives_when_the_server_declared_no_default(is_required):
    """An adapter-invented null default never costs the author their enum.

    An optional field with no declared default still emits ``default: null``,
    but the author never wrote that null, so it carries no claim that can
    contradict their enum. This is the shape the fix exists for: an optional
    field the connector documented with a closed value set.
    """
    members = ["metric", "imperial"]
    carrier = _emitted_metadata(
        {"f": {"type": "string", "enum": members}},
        ["f"] if is_required else [],
    )

    assert carrier["enum"] == members


def test_enum_survives_when_a_non_finite_default_was_replaced():
    """Replacing an unusable default makes it this adapter's, not the author's."""
    field = _emitted_field(
        {"f": {"type": "number", "enum": [1, 2], "default": float("inf")}}, []
    )

    assert _metadata_carrier(field)["enum"] == [1, 2]
    assert field["default"] is None


@pytest.mark.parametrize(
    "declared_default,enum_expected",
    [
        ("metric", True),
        (None, False),
        ("celsius", False),
    ],
)
def test_authored_default_still_governs_the_enum(declared_default, enum_expected):
    """A default the author wrote must sit inside the enum they wrote."""
    members = ["metric", "imperial"]
    carrier = _emitted_metadata(
        {"f": {"type": "string", "enum": members, "default": declared_default}}, []
    )

    assert ("enum" in carrier) is enum_expected
    if enum_expected:
        assert carrier["enum"] == members


def test_optional_field_metadata_is_nested_in_the_non_null_branch():
    """An optional field carries its metadata inside its declared-type branch.

    Beside the ``anyOf`` wrapper the metadata would be lost to any consumer
    that resolves the wrapper down to the non-null branch. The default stays
    a sibling of the wrapper, because it belongs to the field, not to one
    branch of it.
    """
    field = _emitted_field(
        {
            "f": {
                "type": "string",
                "description": "Guidance.",
                "enum": ["a", "b"],
                "pattern": "^[ab]$",
            }
        },
        [],
    )

    assert field["anyOf"] == [
        {
            "description": "Guidance.",
            "enum": ["a", "b"],
            "pattern": "^[ab]$",
            "type": "string",
        },
        {"type": "null"},
    ]
    assert field["default"] is None
    assert "description" not in field
    assert "enum" not in field


@pytest.mark.parametrize("is_required", [True, False])
def test_field_metadata_survives_the_claude_schema_pass(is_required):
    """Metadata reaches Anthropic providers, whether or not the field is optional.

    Claude does not accept ``anyOf``, and the provider client resolves an
    optional field to its non-null branch, keeping only what that branch
    holds. The one documented loss is numeric bounds, which that same pass
    strips from every number and integer schema regardless of this adapter.

    This is the only provider pass there is to test: ``claude.py`` holds the
    repo's sole rewrite of an emitted tool schema, so every other provider is
    sent the schema this adapter emits, ``anyOf`` and nested metadata intact.
    """
    emitted = _emitted_schema(
        {
            "tax_reference_number": {
                "type": "string",
                "description": "Tax file number, 9 digits.",
                "enum": ["TFN", "ABN"],
                "pattern": "^[0-9]{9}$",
                "format": "regex",
                "minLength": 9,
            },
            "attempts": {"type": "integer", "minimum": 1, "maximum": 9},
        },
        ["tax_reference_number", "attempts"] if is_required else [],
    )
    fixed = _fix_pydantic_schema_for_claude(emitted)["properties"]

    assert fixed["tax_reference_number"]["description"] == "Tax file number, 9 digits."
    assert fixed["tax_reference_number"]["enum"] == ["TFN", "ABN"]
    assert fixed["tax_reference_number"]["pattern"] == "^[0-9]{9}$"
    assert fixed["tax_reference_number"]["format"] == "regex"
    assert fixed["tax_reference_number"]["minLength"] == 9
    assert "minimum" not in fixed["attempts"]
    assert "maximum" not in fixed["attempts"]


@pytest.mark.parametrize(
    "pattern",
    [
        "^a  b$",
        "^a\tb$",
        " ^abc$ ",
        "^[ ]{3}x$",
    ],
)
def test_pattern_is_emitted_character_for_character(pattern):
    """A regex is machine-read, so no whitespace normalization touches it.

    ``^a  b$`` and ``^a b$`` accept disjoint sets of strings, so collapsing
    the run would hand the model a rule the connector author never wrote.
    """
    carrier = _emitted_metadata({"f": {"type": "string", "pattern": pattern}}, ["f"])

    assert carrier["pattern"] == pattern


@pytest.mark.parametrize("pattern", ["", "   ", "\t\n"])
def test_blank_pattern_is_not_emitted(pattern):
    """A pattern with nothing but whitespace states nothing and is dropped."""
    carrier = _emitted_metadata({"f": {"type": "string", "pattern": pattern}}, ["f"])

    assert "pattern" not in carrier


@pytest.mark.parametrize("key", ["minLength", "maxLength"])
@pytest.mark.parametrize("value", [-5, 2.7, -0.5, True])
def test_length_bounds_must_be_non_negative_integers(key, value):
    """JSON Schema defines the length bounds as non-negative integers.

    A negative or fractional character count is a malformed rule, not a
    stricter one, so it is dropped instead of being handed to the model.
    """
    carrier = _emitted_metadata({"f": {"type": "string", key: value}}, ["f"])

    assert key not in carrier


@pytest.mark.parametrize("value", [0, 9])
def test_length_bounds_accept_zero_and_above(value):
    carrier = _emitted_metadata({"f": {"type": "string", "minLength": value}}, ["f"])

    assert carrier["minLength"] == value


@pytest.mark.parametrize(
    "enum_members,declared_default,enum_expected",
    [
        # JSON keeps `true` and `1` apart at every level, so a default of
        # `[1]` is not the listed member `[true]` and the two contradict.
        ([[True]], [1], False),
        ([{"k": True}], [{"k": 1}], False),
        ([[True]], [True], True),
        ([{"k": True}], {"k": True}, True),
    ],
)
def test_nested_bool_and_number_enum_members_stay_apart(
    enum_members, declared_default, enum_expected
):
    carrier = _emitted_metadata(
        {"f": {"enum": enum_members, "default": declared_default}}, []
    )

    assert ("enum" in carrier) is enum_expected


def test_nested_field_metadata_is_not_extracted():
    """Only the tool's own top-level fields are read.

    A nested object or an array item schema is flattened to a bare Python
    type here, so metadata written inside one does not reach the model. This
    pins the documented boundary rather than asserting it is desirable.
    """
    emitted = _emitted_schema(
        {
            "address": {
                "type": "object",
                "description": "Postal address.",
                "properties": {
                    "street": {"type": "string", "description": "STREET_DESC"}
                },
            },
            "tags": {
                "type": "array",
                "description": "Labels.",
                "items": {"type": "string", "description": "ITEM_DESC"},
            },
        },
        ["address", "tags"],
    )
    serialized = _compact_json(emitted)

    assert emitted["properties"]["address"]["description"] == "Postal address."
    assert emitted["properties"]["tags"]["description"] == "Labels."
    assert "STREET_DESC" not in serialized
    assert "ITEM_DESC" not in serialized


def test_an_unusable_numeric_bound_costs_only_its_own_key():
    """A value no provider can carry drops that key and nothing else.

    Python integers are unbounded, so an integer past the float range makes
    ``math.isfinite`` raise rather than answer. Left uncaught that reaches
    the args-model fallback and the whole tool loses its arguments, so the
    check has to refuse the value instead.
    """
    too_large = int("9" * 401)
    schema = _emitted_schema(
        {
            "n": {"type": "integer", "minimum": too_large, "description": "Count."},
            "other": {"type": "string", "description": "Untouched."},
        },
        ["n", "other"],
    )
    args_model = _schema_adapter(
        {
            "n": {"type": "integer", "minimum": too_large, "description": "Count."},
            "other": {"type": "string", "description": "Untouched."},
        },
        ["n", "other"],
    ).args_type()

    assert set(args_model.model_fields) == {"n", "other"}
    assert "minimum" not in schema["properties"]["n"]
    assert schema["properties"]["n"]["description"] == "Count."
    assert schema["properties"]["other"]["description"] == "Untouched."


def test_an_unserializable_deep_enum_costs_only_its_own_key():
    """A value nested past the recursion limit drops its key, not the tool.

    ``json.dumps`` gives up at roughly 1200 levels with a ``RecursionError``,
    which is refused where the value is judged.
    """
    deep: list = []
    cursor = deep
    for _ in range(1500):
        nested: list = []
        cursor.append(nested)
        cursor = nested
    properties = {
        "f": {"type": "string", "enum": [deep], "description": "Count."},
        "other": {"type": "string", "description": "Untouched."},
    }
    schema = _emitted_schema(properties, ["f", "other"])
    args_model = _schema_adapter(properties, ["f", "other"]).args_type()

    assert set(args_model.model_fields) == {"f", "other"}
    assert "enum" not in schema["properties"]["f"]
    assert schema["properties"]["f"]["description"] == "Count."
    assert schema["properties"]["other"]["description"] == "Untouched."


@pytest.mark.parametrize(
    "length,truncated",
    [
        (_FIELD_TEXT_MAX_CHARS - 1, False),
        (_FIELD_TEXT_MAX_CHARS, False),
        (_FIELD_TEXT_MAX_CHARS + 1, True),
    ],
)
def test_description_length_boundary(length, truncated):
    """The cap is inclusive: exactly the cap is kept, one over is shortened."""
    raw = "a" * length
    carrier = _emitted_metadata({"f": {"type": "string", "description": raw}}, ["f"])
    description = carrier["description"]

    assert len(description) <= _FIELD_TEXT_MAX_CHARS
    if truncated:
        assert description == "a" * (_FIELD_TEXT_MAX_CHARS - 1) + "…"
    else:
        assert description == raw

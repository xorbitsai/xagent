# ruff: noqa: F401, F811
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool

from xagent.core.agent.result import ClassifiedToolFailure
from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    _build_mcp_tool_adapter,
    _delegated_authorization_failed_result,
    _delegated_retry_failed_result,
    classify_write_hint,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.signed_upload import SignedUploadIssuer
from xagent.core.tools.core.mcp.tools import attach_raw_annotations
from xagent.core.workspace import ExternalUploadStagingError

# The fixtures and the fake connector are those of the replacement tests.  They
# are imported by module name because the test directory is not a package.
from test_signed_upload_substitution import (  # isort: skip
    _RESOLVER_KEY,
    _RUNTIME_KEY,
    _SCENARIOS,
    _SLACK_REFRESH_KEY,
    CONNECTION_AUTH,
    CONNECTION_SECRET,
    CONNECTION_URL,
    EXPIRES_AT,
    FILE_ARGUMENT,
    FILE_BYTES,
    FRESH_AUTH,
    HEADER_SECRET,
    ISSUING_TOOL,
    LEAKY_MESSAGE,
    OUR_FILE_ID,
    OUTER_TOOL,
    QUERY_SECRET,
    REFUSED_PREFIX,
    REMOTE_FILE_ID,
    UNSUPPORTED_TYPE,
    UNTRUSTED_CONNECTOR,
    UNTRUSTED_URL,
    UPLOAD_FAILED,
    UPLOAD_URL,
    _connection,
    _Env,
    _http_401,
    _issued,
    _rejecting_the_initial_credential,
    _Scenario,
    _scoped_env,
    _text,
    _ticket,
    env,
)

ARGUMENT_ERROR = "Pass one workspace file as `file`, written as file:<id>."
GENERIC_ERROR = "Error executing MCP tool."
FILE_UNAVAILABLE = "The workspace file is no longer available."
INSTRUCTIONS_SECRET = "instructions-secret-text"
FILE_PROPERTY_DESCRIPTION = (
    "A workspace file (attached or generated), written as file:<id>."
)
DESCRIPTION = (
    "Upload a workspace file (attached or generated) to this connector. "
    "Pass the file as `file`, written as file:<id>. The host works out the "
    "file's name, type, size and SHA-256 digest, asks the connector for an "
    "upload and sends the bytes itself; it never returns an upload URL. The "
    "result holds the connector's `file_id`, and `expires_at` when the "
    "connector reports one. Put that `file_id` in the argument of the "
    "connector tool that takes the file, also when that tool is called "
    "through another tool's arguments. A `file_id` may work only once and "
    "only until it expires; call this tool again for a new one."
)

ISSUER = SignedUploadIssuer(
    connection_host="mcp.example.test",
    issuing_tool_name=ISSUING_TOOL,
    allowed_upload_hosts=frozenset({"uploads.example.test"}),
    upload_path_prefix="/staged/",
    content_types={".pdf": "application/pdf"},
)

_MISSING = object()


def _text_only_reply(**overrides: Any) -> CallToolResult:
    """A reply shaped like a connector's real one.

    The payload is only in the text content, with no structured content, and
    it carries the file id's own expiry and the instructions text next to the
    signed upload fields.
    """
    payload: dict[str, Any] = {
        "file_id": REMOTE_FILE_ID,
        "upload_url": UPLOAD_URL,
        "method": "PUT",
        "headers": {
            "Content-Type": "application/pdf",
            "Content-Length": str(len(FILE_BYTES)),
        },
        "url_expires_at": EXPIRES_AT,
        "expires_at": "2030-01-01T01:00:00Z",
        "instructions": "Send the bytes with PUT, then pass file_id on.",
    }
    payload.update(overrides)
    payload = {key: value for key, value in payload.items() if value is not _MISSING}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=None,
        isError=False,
    )


def _wrapper(
    env: _Env,
    *,
    connection: Any = None,
    with_workspace: bool = True,
    allow_users: list[str] | None = None,
) -> Any:
    return mcp_adapter_module._build_signed_upload_issuer_adapter(
        "docs",
        _connection() if connection is None else connection,
        ISSUER,
        workspace=env.workspace if with_workspace else None,
        allow_users=allow_users,
    )


def _args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"file": f"file:{OUR_FILE_ID}"}
    args.update(overrides)
    return args


async def _run_scenario(env: _Env, scenario: _Scenario) -> dict[str, Any] | None:
    """Run one outcome through the wrapper; ``None`` means it was cancelled."""
    scenario.setup(env)
    wrapper = _wrapper(env)
    if not scenario.cancel:
        return await wrapper.run_json_async(_args())
    task = asyncio.ensure_future(wrapper.run_json_async(_args()))
    await asyncio.wait_for(env.put_started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return None


def _structured(result: dict[str, Any]) -> dict[str, Any]:
    structured = result["structured_content"]
    assert isinstance(structured, dict)
    return structured


# --------------------------------------------------------------------------
# What the model sees
# --------------------------------------------------------------------------


def _runtime_bound_connection(key: str | None) -> dict[str, Any]:
    if key is None:
        return _connection()
    return _connection(
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "document"},
                "target": {"target_type": "tool_arguments", "key": key},
            }
        ],
        connector_runtime={
            "context": {"document": "rf_fromcontext"},
            "secrets": {},
            "auth_selector": {},
        },
    )


def _remote_issuing_tool(annotations: dict[str, Any] | None) -> MCPTool:
    tool = MCPTool(
        name=ISSUING_TOOL,
        description="Ask the connector for a signed upload.",
        inputSchema={
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "mime_type": {"type": "string"},
                "size_bytes": {"type": "integer"},
                "sha256": {"type": "string"},
            },
            "required": ["filename", "mime_type", "size_bytes", "sha256"],
        },
    )
    return attach_raw_annotations(tool, annotations)


@pytest.mark.parametrize(
    ("annotations", "ordinary_non_idempotent"),
    [
        pytest.param(None, False, id="no-annotations"),
        pytest.param({"destructiveHint": True}, True, id="destructive"),
    ],
)
@pytest.mark.parametrize("bound_key", [None, "file", "filename"])
async def test_issuer_wrapper_schema_description_and_hints(
    monkeypatch: pytest.MonkeyPatch,
    annotations: dict[str, Any] | None,
    ordinary_non_idempotent: bool,
    bound_key: str | None,
) -> None:
    remote = _remote_issuing_tool(annotations)
    connection = _runtime_bound_connection(bound_key)
    connection["signed_uploads"] = [
        {
            "issuing_tool_name": ISSUING_TOOL,
            "allowed_upload_hosts": ["uploads.example.test"],
            "upload_path_prefix": "/staged/",
            "content_types": {".pdf": "application/pdf"},
        }
    ]

    class FakeSession:
        async def initialize(self) -> None:
            return None

    @asynccontextmanager
    async def fake_create_session(_connection: Any):
        yield FakeSession()

    async def fake_load_mcp_tools(_session: Any) -> list[MCPTool]:
        return [remote]

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools", fake_load_mcp_tools)

    result = await load_mcp_tools_as_agent_tools({"docs": connection})

    assert result.failures == ()
    (tool,) = result.tools
    wrapper = tool._target
    assert type(wrapper) is mcp_adapter_module.SignedUploadIssuerAdapter
    # The remote tool, built the ordinary way, would carry the remote hints.
    ordinary = _build_mcp_tool_adapter("docs", connection, remote)
    assert ordinary.non_idempotent_write is ordinary_non_idempotent
    schema = wrapper.args_type().model_json_schema()
    assert set(schema["properties"]) == {"file"}
    assert schema["required"] == ["file"]
    assert schema["properties"]["file"]["description"] == FILE_PROPERTY_DESCRIPTION
    assert wrapper.description == DESCRIPTION
    assert wrapper.non_idempotent_write is False
    assert wrapper.write_hint == classify_write_hint(None)


# --------------------------------------------------------------------------
# A successful call returns only the connector's file id
# --------------------------------------------------------------------------


def _structured_reply(expires_at: Any) -> CallToolResult:
    """A reply with structured content, which the parser reads first."""
    overrides = {} if expires_at is _MISSING else {"expires_at": expires_at}
    return _issued(_ticket(**overrides))


def _text_reply(expires_at: Any) -> CallToolResult:
    return _text_only_reply(expires_at=expires_at)


_REPLIES: list[Any] = [
    pytest.param(
        _text_reply, "2030-01-01T01:00:00Z", "2030-01-01T01:00:00Z", id="text-z"
    ),
    pytest.param(
        _text_reply,
        "2030-01-01T09:00:00+08:00",
        "2030-01-01T01:00:00Z",
        id="text-offset",
    ),
    pytest.param(_text_reply, _MISSING, None, id="text-missing"),
    pytest.param(_text_reply, "x", None, id="text-unreadable"),
    pytest.param(
        _text_reply, "9999-12-31T23:59:59-23:59", None, id="text-out-of-range"
    ),
    pytest.param(
        _structured_reply,
        "2030-01-01T01:00:00Z",
        "2030-01-01T01:00:00Z",
        id="structured-z",
    ),
    pytest.param(_structured_reply, _MISSING, None, id="structured-missing"),
]


@pytest.mark.parametrize(("reply", "reported", "expected"), _REPLIES)
async def test_issuer_wrapper_returns_only_the_remote_file_id(
    env: _Env,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reply: Callable[[Any], CallToolResult],
    reported: Any,
    expected: str | None,
) -> None:
    # With a user id set, the only warnings left are about the upload itself.
    monkeypatch.setenv("XAGENT_USER_ID", "user-1")
    caplog.set_level(logging.DEBUG, logger="xagent")
    env.issue = lambda args: reply(reported)

    result = await _wrapper(env).run_json_async(_args())

    assert result["is_error"] is False
    assert env.calls_to(ISSUING_TOOL) == [
        {
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "size_bytes": len(FILE_BYTES),
            "sha256": hashlib.sha256(FILE_BYTES).hexdigest(),
        }
    ]
    assert len(env.put_requests) == 1
    assert env.put_requests[0]["body"] == FILE_BYTES
    structured = _structured(result)
    if expected is None:
        assert structured == {"file_id": REMOTE_FILE_ID}
    else:
        assert structured == {"file_id": REMOTE_FILE_ID, "expires_at": expected}
    # The text content is the same object, serialized.
    (item,) = result["content"]
    assert item["type"] == "text"
    assert json.loads(item["text"]) == structured
    assert env.calls_to(OUTER_TOOL) == []
    assert env.staging_leftovers() == []
    # An expiry that cannot be read is simply left out, without a log line.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --------------------------------------------------------------------------
# Every outcome removes the staged copy and keeps the signed URL private
# --------------------------------------------------------------------------


def _text_only_success(env: _Env) -> None:
    env.issue = lambda args: _text_only_reply()


def _secret_instructions(env: _Env) -> None:
    env.issue = lambda args: _text_only_reply(
        instructions=f"Use this secret: {INSTRUCTIONS_SECRET}"
    )


_OUTCOMES = [
    *_SCENARIOS,
    _Scenario("text-only-success", _text_only_success, None, puts=1),
    _Scenario("text-only-secret-instructions", _secret_instructions, None, puts=1),
]


@pytest.mark.parametrize("scenario", _OUTCOMES, ids=[s.id for s in _OUTCOMES])
async def test_issuer_wrapper_outcomes(
    env: _Env, scenario: _Scenario, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="xagent")

    result = await _run_scenario(env, scenario)

    assert env.staging_leftovers() == []
    assert len(env.put_requests) == scenario.puts
    # The wrapper never calls the connector's receiving tool.
    assert env.calls_to(OUTER_TOOL) == []
    if result is not None:
        if scenario.message is None:
            assert result["is_error"] is False
            assert _structured(result)["file_id"] == REMOTE_FILE_ID
        else:
            assert result["is_error"] is True
            if scenario.expected_prefix:
                assert _text(result).startswith(scenario.message)
            else:
                assert _text(result) == scenario.message
    seen = {"xagent log": caplog.text, "result": repr(result)}
    for where, text in seen.items():
        for secret in (
            UPLOAD_URL,
            QUERY_SECRET,
            HEADER_SECRET,
            CONNECTION_SECRET,
            INSTRUCTIONS_SECRET,
        ):
            assert secret not in text, f"{secret!r} leaked into {where}"


# --------------------------------------------------------------------------
# What the model may pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({"file": OUR_FILE_ID}, id="bare-uuid"),
        pytest.param(
            {
                "file": f"file:{OUR_FILE_ID}",
                "filename": "other.png",
                "mime_type": "image/png",
                "size_bytes": 1,
                "sha256": "00",
            },
            id="extra-arguments-are-ignored",
        ),
    ],
)
async def test_issuer_wrapper_accepts_a_file(env: _Env, args: dict[str, Any]) -> None:
    result = await _wrapper(env).run_json_async(args)

    assert result["is_error"] is False
    assert _structured(result)["file_id"] == REMOTE_FILE_ID
    (issuing_args,) = env.calls_to(ISSUING_TOOL)
    assert set(issuing_args) == {"filename", "mime_type", "size_bytes", "sha256"}
    assert issuing_args["filename"] == "report.pdf"
    assert issuing_args["mime_type"] == "application/pdf"
    assert issuing_args["size_bytes"] == len(FILE_BYTES)
    assert len(env.put_requests) == 1


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({}, id="empty"),
        pytest.param({"file": None}, id="none"),
        pytest.param({"file": 7}, id="number"),
        pytest.param({"file": ""}, id="empty-string"),
        pytest.param({"file": [f"file:{OUR_FILE_ID}"]}, id="list"),
        pytest.param({"file": "remote_abcdefgh"}, id="connector-file-id"),
        pytest.param({"arguments": {"file": f"file:{OUR_FILE_ID}"}}, id="wrapped"),
        pytest.param(
            {"filename": "report.pdf", "mime_type": "application/pdf", "size_bytes": 3},
            id="issuing-tool-schema",
        ),
    ],
)
async def test_issuer_wrapper_argument_shapes(env: _Env, args: dict[str, Any]) -> None:
    result = await _wrapper(env).run_json_async(args)

    assert result["is_error"] is True
    assert _text(result) == ARGUMENT_ERROR
    assert env.server_calls == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


# --------------------------------------------------------------------------
# Refusals before the connector is asked
# --------------------------------------------------------------------------


class _StagingFails:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def __call__(self, *_args: Any) -> Any:
        raise self.error


@pytest.mark.parametrize(
    "cell",
    [
        "no-workspace",
        "unknown-file-reference",
        "unknown-bare-uuid",
        "suffix-not-in-the-table",
        "connection-to-another-host",
        "staging-os-error",
        "staging-error",
    ],
)
async def test_issuer_wrapper_refuses_before_issuing(
    env: _Env, monkeypatch: pytest.MonkeyPatch, cell: str
) -> None:
    unknown = "0b8f5d2e-7c41-4a9e-9d3b-2f6e8a1c4b70"
    wrapper = _wrapper(env)
    args = _args()
    expected = FILE_UNAVAILABLE
    if cell == "no-workspace":
        wrapper = _wrapper(env, with_workspace=False)
    elif cell == "unknown-file-reference":
        args = _args(file=f"file:{unknown}")
    elif cell == "unknown-bare-uuid":
        args = _args(file=unknown)
    elif cell == "suffix-not-in-the-table":
        text_file = env.source.parent / "notes.txt"
        text_file.write_text("plain text")
        env.source = text_file
        expected = UNSUPPORTED_TYPE
    elif cell == "connection-to-another-host":
        wrapper = _wrapper(
            env, connection=_connection(url="https://other.example.test/mcp")
        )
        expected = UNTRUSTED_CONNECTOR
    elif cell == "staging-os-error":
        monkeypatch.setattr(
            env.workspace,
            "stage_file_for_external_upload",
            _StagingFails(OSError("disk unavailable")),
        )
        expected = GENERIC_ERROR
    else:
        monkeypatch.setattr(
            env.workspace,
            "stage_file_for_external_upload",
            _StagingFails(ExternalUploadStagingError("cannot stage")),
        )
        expected = GENERIC_ERROR

    result = await wrapper.run_json_async(args)

    assert result["is_error"] is True
    assert _text(result) == expected
    assert env.server_calls == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


@pytest.mark.parametrize(
    "record",
    [
        pytest.param("other_user", id="another-end-user"),
        pytest.param("other_task", id="same-user-another-task"),
        pytest.param("deleted", id="record-deleted"),
        pytest.param("ok", id="current-task"),
    ],
)
async def test_issuer_wrapper_respects_workspace_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: str
) -> None:
    env, ids = _scoped_env(tmp_path, monkeypatch)

    result = await _wrapper(env).run_json_async({"file": f"file:{ids[record]}"})

    if record == "ok":
        assert result["is_error"] is False
        assert len(env.calls_to(ISSUING_TOOL)) == 1
        assert env.put_requests[0]["body"] == FILE_BYTES
    else:
        assert result["is_error"] is True
        assert _text(result) == FILE_UNAVAILABLE
        assert env.server_calls == []
        assert env.put_requests == []
    assert env.staging_leftovers() == []


async def test_issuer_wrapper_checks_the_user_first(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_USER_ID", "user-7")
    read: list[str] = []
    monkeypatch.setattr(
        env.workspace,
        "stage_file_for_external_upload",
        lambda file_id: read.append(file_id),
    )

    result = await _wrapper(env, allow_users=["user-42"]).run_json_async(_args())

    assert result == mcp_adapter_module._mcp_access_denied_result(
        "user-7", ISSUING_TOOL
    )
    assert env.server_calls == []
    assert env.put_requests == []
    assert read == []


# --------------------------------------------------------------------------
# Authorization of the issuing call
# --------------------------------------------------------------------------


def _unauthorized(args: dict[str, Any]) -> CallToolResult:
    raise _http_401()


def _no_refresh(env: _Env, calls: list[str]) -> tuple[Any, Any]:
    env.issue = _unauthorized
    return _connection(), _delegated_authorization_failed_result()


def _refresh_raises(env: _Env, calls: list[str]) -> tuple[Any, Any]:
    def refresh() -> Any:
        calls.append("refresh")
        raise RuntimeError(LEAKY_MESSAGE)

    env.issue = _unauthorized
    return (
        _connection(**{_RUNTIME_KEY: refresh}),
        _delegated_authorization_failed_result(),
    )


def _resolver_classifies_the_failure(env: _Env, calls: list[str]) -> tuple[Any, Any]:
    async def refresh(challenge: Any) -> Any:
        return ClassifiedToolFailure(failure_code="oauth_token_required")

    env.issue = lambda args: (_ for _ in ()).throw(
        _http_401(['Bearer error="invalid_token"'])
    )
    return (
        _connection(**{_RESOLVER_KEY: refresh}),
        _delegated_authorization_failed_result(failure_code="oauth_token_required"),
    )


def _retry_fails(env: _Env, calls: list[str]) -> tuple[Any, Any]:
    def refresh() -> dict[str, Any]:
        calls.append("refresh")
        return _connection(headers={"Authorization": FRESH_AUTH})

    def issue(args: dict[str, Any]) -> CallToolResult:
        raise RuntimeError(LEAKY_MESSAGE)

    env.issue = _rejecting_the_initial_credential(env, issue)
    return _connection(**{_RUNTIME_KEY: refresh}), _delegated_retry_failed_result()


def _retry_succeeds(env: _Env, calls: list[str]) -> tuple[Any, Any]:
    def refresh() -> dict[str, Any]:
        calls.append("refresh")
        return _connection(headers={"Authorization": FRESH_AUTH})

    env.issue = _rejecting_the_initial_credential(env, env.issue)
    return _connection(**{_RUNTIME_KEY: refresh}), None


@pytest.mark.parametrize(
    ("setup", "refreshes"),
    [
        pytest.param(_no_refresh, 0, id="401-without-a-refresh"),
        pytest.param(_refresh_raises, 1, id="refresh-fails"),
        pytest.param(_resolver_classifies_the_failure, 0, id="resolver-failure-code"),
        pytest.param(_retry_fails, 1, id="retry-fails"),
        pytest.param(_retry_succeeds, 1, id="retry-succeeds"),
    ],
)
async def test_issuer_wrapper_authorization_paths(
    env: _Env, setup: Callable[[_Env, list[str]], tuple[Any, Any]], refreshes: int
) -> None:
    calls: list[str] = []
    connection, expected = setup(env, calls)

    result = await _wrapper(env, connection=connection).run_json_async(_args())

    assert calls == ["refresh"] * refreshes
    assert env.staging_leftovers() == []
    if expected is None:
        assert result["is_error"] is False
        assert env.call_auth == [
            (ISSUING_TOOL, CONNECTION_AUTH),
            (ISSUING_TOOL, FRESH_AUTH),
        ]
        assert len(env.put_requests) == 1
    else:
        assert result == expected
        assert env.put_requests == []


@pytest.mark.parametrize("issuing_is_error", [False, True])
async def test_issuer_wrapper_refuses_a_refreshed_connection_to_another_host(
    env: _Env, issuing_is_error: bool
) -> None:
    def refresh() -> dict[str, Any]:
        return _connection(
            url="https://other.example.test/mcp", headers={"Authorization": FRESH_AUTH}
        )

    if issuing_is_error:
        env.issue = lambda args: CallToolResult(
            content=[TextContent(type="text", text="other host says hi")],
            isError=True,
        )
    env.issue = _rejecting_the_initial_credential(env, env.issue)

    result = await _wrapper(
        env, connection=_connection(**{_RUNTIME_KEY: refresh})
    ).run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result) == UNTRUSTED_CONNECTOR
    assert env.put_requests == []
    assert env.staging_leftovers() == []


def _refresh_is_not_callable() -> Any:
    return "not callable"


def _refresh_raises_an_error() -> Any:
    def refresh() -> Any:
        raise RuntimeError(LEAKY_MESSAGE)

    return refresh


def _refresh_returns_no_connection() -> Any:
    return lambda: ["not a connection"]


@pytest.mark.parametrize(
    "make_refresh",
    [
        pytest.param(_refresh_is_not_callable, id="not-callable"),
        pytest.param(_refresh_raises_an_error, id="raises"),
        pytest.param(_refresh_returns_no_connection, id="returns-no-connection"),
    ],
)
async def test_issuer_wrapper_refuses_when_the_actor_refresh_fails(
    env: _Env, monkeypatch: pytest.MonkeyPatch, make_refresh: Callable[[], Any]
) -> None:
    staged: list[str] = []
    original_stage = mcp_adapter_module.MCPToolAdapter._stage_external_upload_file

    async def recording_stage(self: Any, file_id: str) -> Any:
        staged.append(file_id)
        return await original_stage(self, file_id)

    monkeypatch.setattr(
        mcp_adapter_module.MCPToolAdapter,
        "_stage_external_upload_file",
        recording_stage,
    )
    wrapper = _wrapper(
        env, connection=_connection(**{_SLACK_REFRESH_KEY: make_refresh()})
    )

    result = await wrapper.run_json_async(_args())

    assert result == _delegated_authorization_failed_result()
    assert staged == []
    assert env.calls_to(ISSUING_TOOL) == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


# --------------------------------------------------------------------------
# One handler for a failed call on both paths
# --------------------------------------------------------------------------


def _fail_staging(env: _Env, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(
        env.workspace, "stage_file_for_external_upload", _StagingFails(error)
    )


async def _run_failing_call(env: _Env, path: str) -> dict[str, Any]:
    if path == "replacement":
        return await env.adapter().run_json_async(
            {FILE_ARGUMENT: f"file:{OUR_FILE_ID}"}
        )
    return await _wrapper(env).run_json_async(_args())


@pytest.mark.parametrize("path", ["replacement", "issuer"])
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError("disk unavailable"), id="single-exception"),
        pytest.param(ExceptionGroup("group", [OSError("leaf")]), id="exception-group"),
    ],
)
async def test_both_paths_end_in_the_shared_failure_handler(
    env: _Env, monkeypatch: pytest.MonkeyPatch, path: str, error: Exception
) -> None:
    sentinel = {"content": [{"text": "sentinel"}], "is_error": True}
    seen: list[BaseException] = []

    def failure_result(_self: Any, exc: BaseException) -> dict[str, Any]:
        seen.append(exc)
        return sentinel

    monkeypatch.setattr(
        mcp_adapter_module.MCPToolAdapter, "_execution_failure_result", failure_result
    )
    _fail_staging(env, monkeypatch, error)

    result = await _run_failing_call(env, path)

    assert result is sentinel
    assert seen == [error]


async def test_issuer_wrapper_logs_an_exception_group_by_its_leaves(
    env: _Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="xagent")
    _fail_staging(
        env,
        monkeypatch,
        ExceptionGroup("group", [OSError("first leaf"), ValueError("second leaf")]),
    )

    result = await _run_failing_call(env, "issuer")

    assert result["is_error"] is True
    assert _text(result) == GENERIC_ERROR
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "execution failed with exception group ExceptionGroup" in m for m in messages
    )
    assert any("related exception OSError" in m for m in messages)
    assert any("related exception ValueError" in m for m in messages)
    assert env.put_requests == []
    assert env.staging_leftovers() == []


# --------------------------------------------------------------------------
# Cancellation and concurrency
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", ["staging", "hashing", "issuing"])
async def test_issuer_wrapper_cancellation_discards_the_copy(
    env: _Env, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def block_worker() -> None:
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=10)

    if step == "staging":
        original_stage = env.workspace.stage_file_for_external_upload

        def blocking_stage(file_id: str) -> Any:
            staged = original_stage(file_id)
            block_worker()
            return staged

        monkeypatch.setattr(
            env.workspace, "stage_file_for_external_upload", blocking_stage
        )
    elif step == "hashing":
        original_hash = mcp_adapter_module._file_size_and_sha256

        def blocking_hash(path: Path) -> tuple[int, str]:
            block_worker()
            return original_hash(path)

        monkeypatch.setattr(mcp_adapter_module, "_file_size_and_sha256", blocking_hash)
    else:

        async def never_answers(args: dict[str, Any]) -> CallToolResult:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        env.issue = never_answers

    task = asyncio.ensure_future(_wrapper(env).run_json_async(_args()))
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert env.staging_leftovers() == []
    assert env.put_requests == []


@pytest.mark.parametrize("same_file", [False, True], ids=["two-files", "same-file"])
async def test_issuer_wrapper_concurrent_calls_are_independent(
    env: _Env, monkeypatch: pytest.MonkeyPatch, same_file: bool
) -> None:
    second_id = "7a2d3c4b-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
    other_bytes = FILE_BYTES[::-1]
    second = env.source.parent / "second.pdf"
    second.write_bytes(other_bytes)
    sources = {OUR_FILE_ID: env.source, second_id: second}
    monkeypatch.setattr(
        env.workspace, "resolve_file_id_detached", lambda file_id: sources.get(file_id)
    )
    issued = iter(range(1, 10))

    def issue(args: dict[str, Any]) -> CallToolResult:
        return _issued(
            _ticket(file_id=f"rf_issue_{next(issued)}_{args['filename'][:3]}")
        )

    env.issue = issue
    wrapper = _wrapper(env)
    ids = [OUR_FILE_ID, OUR_FILE_ID if same_file else second_id]

    results = await asyncio.gather(
        *[wrapper.run_json_async({"file": f"file:{file_id}"}) for file_id in ids]
    )

    assert [result["is_error"] for result in results] == [False, False]
    file_ids = [_structured(result)["file_id"] for result in results]
    assert len(set(file_ids)) == 2
    assert len(env.calls_to(ISSUING_TOOL)) == 2
    assert len(env.put_requests) == 2
    bodies = {request["body"] for request in env.put_requests}
    assert bodies == ({FILE_BYTES} if same_file else {FILE_BYTES, other_bytes})
    assert env.staging_leftovers() == []


# --------------------------------------------------------------------------
# Failures that end the upload in different places
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", ["hashing", "opening"])
async def test_issuer_wrapper_os_errors_match_the_shared_upload(
    env: _Env, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    def fail(*_args: Any) -> Any:
        raise OSError("disk unavailable")

    if step == "hashing":
        monkeypatch.setattr(mcp_adapter_module, "_file_size_and_sha256", fail)
        expected, issuing_calls = GENERIC_ERROR, 0
    else:
        # Hashing and the upload both read through ``open``, so hashing is
        # stubbed with the file's real size and digest first.
        monkeypatch.setattr(
            mcp_adapter_module,
            "_file_size_and_sha256",
            lambda path: (len(FILE_BYTES), hashlib.sha256(FILE_BYTES).hexdigest()),
        )
        monkeypatch.setattr(mcp_adapter_module, "open", fail, raising=False)
        expected, issuing_calls = UPLOAD_FAILED, 1

    result = await _wrapper(env).run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result) == expected
    assert len(env.calls_to(ISSUING_TOOL)) == issuing_calls
    assert env.put_requests == []
    assert env.staging_leftovers() == []


# --------------------------------------------------------------------------
# The upload checks have one owner
# --------------------------------------------------------------------------

_UPLOAD_CHECKS = (
    "check_signed_upload_url",
    "parse_signed_upload_response",
    "_signed_upload_transport",
)


@contextmanager
def _recorded_upload_checks() -> Iterator[list[tuple[str, bool]]]:
    """Record each run of an upload check and whether the shared method ran it.

    The checks are found by their code objects, so a call through another name
    for the same function is recorded too. The module's own names are read
    here, after ``env`` has put its fake transport in place, so the recorded
    transport function is the one the upload would have used.
    """
    codes = {
        getattr(mcp_adapter_module, name).__code__: name for name in _UPLOAD_CHECKS
    }
    shared = mcp_adapter_module.MCPToolAdapter._issue_and_upload_staged_file.__code__
    calls: list[tuple[str, bool]] = []

    def profiler(frame: Any, event: str, _arg: Any) -> None:
        if event == "call" and frame.f_code in codes:
            calls.append((codes[frame.f_code], frame.f_back.f_code is shared))

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("path", ["replacement", "issuer"])
async def test_signed_upload_checks_run_once_and_only_in_the_shared_upload(
    env: _Env, path: str
) -> None:
    """Each check runs once per upload, called by the shared method alone."""
    with _recorded_upload_checks() as calls:
        if path == "replacement":
            result = await env.adapter().run_json_async(
                {FILE_ARGUMENT: f"file:{OUR_FILE_ID}"}
            )
        else:
            result = await _wrapper(env).run_json_async(_args())

    assert result["is_error"] is False
    assert len(env.put_requests) == 1
    assert sorted(calls) == sorted((name, True) for name in _UPLOAD_CHECKS)

from __future__ import annotations

import asyncio
import builtins
import hashlib
import inspect
import json
import logging
import threading
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.result import ClassifiedToolFailure
from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPToolAdapter,
    _build_mcp_tool_adapter,
    _delegated_authorization_failed_result,
    _delegated_retry_failed_result,
)
from xagent.core.tools.adapters.vibe.mcp_approval_gate import (
    GatedCall,
    GateDecision,
    ToolCallExecutionContext,
    bind_tool_call_execution_context,
    gate_mcp_tools,
    register_mcp_approval_gate,
    unregister_mcp_approval_gate,
)
from xagent.core.tools.adapters.vibe.signed_upload import SignedUploadBinding
from xagent.core.tools.user_interaction import ToolInteractionSettlement
from xagent.core.utils.public_http_transport import (
    PublicHttpTransport,
    UnsafePublicHttpTarget,
)
from xagent.core.utils.security import PrivateNetworkHostError
from xagent.core.workspace import TaskWorkspace
from xagent.web import logging_config
from xagent.web.models import Base
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User

CONNECTION_URL = "https://mcp.example.test/mcp"
CONNECTION_SECRET = "conn-secret-sample"
CONNECTION_AUTH = f"Bearer {CONNECTION_SECRET}"
FRESH_AUTH = "Bearer fresh-secret-sample"
OUTER_TOOL = "attach_document"
ISSUING_TOOL = "request_upload"
FILE_ARGUMENT = "document"
OUR_FILE_ID = "6f1c2d3e-4b5a-4c6d-8e7f-9a0b1c2d3e4f"
REMOTE_FILE_ID = "rf_abcdefgh"
QUERY_SECRET = "secret-signature"
HEADER_SECRET = "header-secret-value"
UPLOAD_URL = f"https://uploads.example.test/staged/abc?sig={QUERY_SECRET}"
EXPIRES_AT = "2030-01-01T00:00:00+00:00"
NOW = datetime(2029, 12, 31, 23, 0, tzinfo=timezone.utc)
# Several chunks long, so the body really is read in pieces.
FILE_BYTES = b"0123456789ABCDEF" * 10000
LEAKY_MESSAGE = "secret https://x.test/?sig=abc"

FILE_UNAVAILABLE = "The attached file is no longer available."
UNTRUSTED_CONNECTOR = "Upload refused: untrusted connector."
UNSUPPORTED_TYPE = "This file type cannot be sent to the connector."
UNTRUSTED_URL = "Upload refused: untrusted upload URL."
UPLOAD_FAILED = "File upload failed."
REFUSED_PREFIX = "The connector refused the file: "

BINDING = SignedUploadBinding(
    connection_host="mcp.example.test",
    tool_name=OUTER_TOOL,
    file_argument=FILE_ARGUMENT,
    issuing_tool_name=ISSUING_TOOL,
    allowed_upload_hosts=frozenset({"uploads.example.test"}),
    upload_path_prefix="/staged/",
    content_types={".pdf": "application/pdf"},
)

SETTINGS = [
    {
        "tool_name": OUTER_TOOL,
        "file_argument": FILE_ARGUMENT,
        "issuing_tool_name": ISSUING_TOOL,
        "allowed_upload_hosts": ["uploads.example.test"],
        "upload_path_prefix": "/staged/",
        "content_types": {".pdf": "application/pdf"},
    }
]

_RUNTIME_KEY = mcp_adapter_module._RUNTIME_CONNECTION_REFRESH_KEY
_RESOLVER_KEY = mcp_adapter_module._OAUTH_TOKEN_RESOLVER_REFRESH_KEY
_SLACK_REFRESH_KEY = mcp_adapter_module._SLACK_ACTOR_RUNTIME_REFRESH_KEY
_REAL_SIGNED_UPLOAD_TRANSPORT = mcp_adapter_module._signed_upload_transport

_SCHEMA = {
    "type": "object",
    "properties": {
        FILE_ARGUMENT: {
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]
        },
        "record_id": {"type": "string"},
    },
}


def _connection(**extra: Any) -> dict[str, Any]:
    return {
        "transport": "streamable_http",
        "url": CONNECTION_URL,
        "headers": {"Authorization": CONNECTION_AUTH},
        **extra,
    }


def _ticket(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "file_id": REMOTE_FILE_ID,
        "upload_url": UPLOAD_URL,
        "method": "PUT",
        "headers": {"Content-Type": "application/pdf", "x-signature": HEADER_SECRET},
        "url_expires_at": EXPIRES_AT,
    }
    payload.update(overrides)
    return payload


def _issued(payload: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


def _http_401(authenticate: list[str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", CONNECTION_URL)
    headers = [(b"www-authenticate", v.encode()) for v in (authenticate or [])]
    response = httpx.Response(401, headers=headers, request=request)
    return httpx.HTTPStatusError("unauthorized", request=request, response=response)


class _Env:
    """A fake connector, a fake PUT transport and a real task workspace."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        workspace: TaskWorkspace | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        if workspace is None:
            workspace = TaskWorkspace(
                id="web_task_1", base_dir=str(tmp_path / "workspaces")
            )
            monkeypatch.setattr(
                workspace,
                "resolve_file_id_detached",
                lambda file_id: self.source if file_id == OUR_FILE_ID else None,
            )
        # The file lives where the workspace allows reading it from.
        self.source = workspace.input_dir / "report.pdf"
        self.source.write_bytes(FILE_BYTES)
        self.workspace = workspace
        self.now = NOW
        self.server_calls: list[tuple[str, dict[str, Any]]] = []
        # The Authorization header of the connection each call was made with.
        self.call_auth: list[tuple[str, str | None]] = []
        self.put_requests: list[dict[str, Any]] = []
        self.put_started = asyncio.Event()
        self.issue: Callable[[dict[str, Any]], Any] = lambda args: _issued(_ticket())
        self.business: Callable[[dict[str, Any]], Any] = self._default_result
        self.put_behavior: Callable[[httpx.Request], Awaitable[httpx.Response]] = (
            self._put_ok
        )
        self.transport_factory: Callable[[], httpx.AsyncBaseTransport] = (
            self._mock_transport
        )

        env = self

        class _Session:
            def __init__(self, connection: Any) -> None:
                headers = (
                    connection.get("headers") if isinstance(connection, dict) else None
                )
                self.auth = (headers or {}).get("Authorization")

            async def initialize(self) -> None:
                return None

            async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
                env.server_calls.append((name, json.loads(json.dumps(arguments))))
                env.call_auth.append((name, self.auth))
                if name == ISSUING_TOOL:
                    outcome = env.issue(arguments)
                else:
                    outcome = env.business(arguments)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                return outcome

        @asynccontextmanager
        async def _create_session(connection: Any):
            yield _Session(connection)

        monkeypatch.setattr(mcp_adapter_module, "create_session", _create_session)
        monkeypatch.setattr(mcp_adapter_module, "_utcnow", lambda: self.now)
        monkeypatch.setattr(
            mcp_adapter_module,
            "_signed_upload_transport",
            lambda: self.transport_factory(),
        )

    @staticmethod
    def _default_result(args: dict[str, Any]) -> CallToolResult:
        payload = {"attached": True, "document_id": args.get(FILE_ARGUMENT)}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structuredContent=payload,
            isError=False,
        )

    async def _put_ok(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request)

    def _mock_transport(self) -> httpx.AsyncBaseTransport:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = await request.aread()
            self.put_started.set()
            self.put_requests.append(
                {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": request.headers.multi_items(),
                    "body": body,
                }
            )
            return await self.put_behavior(request)

        return httpx.MockTransport(handler)

    def adapter(
        self,
        *,
        binding: SignedUploadBinding | None = BINDING,
        connection: Any = None,
        with_workspace: bool = True,
        schema: dict[str, Any] = _SCHEMA,
    ) -> MCPToolAdapter:
        tool = MCPTool(
            name=OUTER_TOOL,
            description="Attach a document to a record.",
            inputSchema=schema,
        )
        return MCPToolAdapter(
            mcp_tool=tool,
            connection=_connection() if connection is None else connection,
            workspace=self.workspace if with_workspace else None,
            signed_upload=binding,
        )

    def staging_leftovers(self) -> list[Path]:
        root = self.workspace.internal_temp_dir / "mcp-upload"
        return sorted(root.rglob("*")) if root.exists() else []

    def calls_to(self, tool_name: str) -> list[dict[str, Any]]:
        return [args for name, args in self.server_calls if name == tool_name]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Env:
    return _Env(tmp_path, monkeypatch)


def _args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {FILE_ARGUMENT: f"file:{OUR_FILE_ID}", "record_id": "r-1"}
    args.update(overrides)
    return args


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


def _rejecting_the_initial_credential(
    env: _Env, respond: Callable[[dict[str, Any]], Any]
) -> Callable[[dict[str, Any]], Any]:
    """A fake tool that answers 401 to a call made with ``CONNECTION_AUTH``."""

    def call(args: dict[str, Any]) -> Any:
        if env.call_auth[-1][1] == CONNECTION_AUTH:
            raise _http_401()
        return respond(args)

    return call


# --------------------------------------------------------------------------
# Without a binding the adapter is unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({"record_id": "r-1"}, id="plain-tool-call"),
        pytest.param(_args(), id="file-reference-value"),
    ],
)
async def test_mcp_adapter_without_signed_upload_binding_is_unchanged(
    env: _Env, args: dict[str, Any]
) -> None:
    tool = MCPTool(
        name=OUTER_TOOL,
        description="Attach a document to a record.",
        inputSchema=_SCHEMA,
    )
    adapter = _build_mcp_tool_adapter(
        "docs", _connection(), tool, workspace=env.workspace
    )

    result = await adapter.run_json_async(args)

    assert adapter._signed_upload is None
    assert adapter.description == "Attach a document to a record."
    assert env.server_calls == [(OUTER_TOOL, args)]
    assert env.put_requests == []
    assert result["is_error"] is False
    assert result["structured_content"] == {
        "attached": True,
        "document_id": args.get(FILE_ARGUMENT),
    }


def test_build_mcp_tool_adapter_passes_the_given_binding(env: _Env) -> None:
    tool = MCPTool(name=OUTER_TOOL, description="x", inputSchema=_SCHEMA)

    given = _build_mcp_tool_adapter("docs", _connection(), tool, signed_upload=BINDING)
    omitted = _build_mcp_tool_adapter("docs", _connection(), tool)
    # The builder never reads the connection's settings itself.
    not_read = _build_mcp_tool_adapter(
        "docs", _connection(signed_uploads=SETTINGS), tool
    )

    assert given._signed_upload == BINDING
    assert omitted._signed_upload is None
    assert not_read._signed_upload is None


# --------------------------------------------------------------------------
# The replacement itself
# --------------------------------------------------------------------------


async def test_signed_upload_replaces_the_file_reference_before_the_tool_call(
    env: _Env,
) -> None:
    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is False
    assert [name for name, _ in env.server_calls] == [ISSUING_TOOL, OUTER_TOOL]
    assert env.calls_to(OUTER_TOOL) == [
        {FILE_ARGUMENT: REMOTE_FILE_ID, "record_id": "r-1"}
    ]
    assert len(env.put_requests) == 1
    assert env.staging_leftovers() == []


async def test_signed_upload_hides_remote_file_id(env: _Env) -> None:
    payload = {
        "document_id": REMOTE_FILE_ID,
        "note": f"see {REMOTE_FILE_ID} for details",
        REMOTE_FILE_ID: "key is left alone",
        "nested": [{"id": REMOTE_FILE_ID}],
    }
    env.business = lambda args: CallToolResult(
        content=[TextContent(type="text", text="done")],
        structuredContent=payload,
        isError=False,
    )

    result = await env.adapter().run_json_async(_args())

    structured = result["structured_content"]
    assert structured["document_id"] == f"file:{OUR_FILE_ID}"
    assert structured["nested"] == [{"id": f"file:{OUR_FILE_ID}"}]
    assert structured["note"] == f"see {REMOTE_FILE_ID} for details"
    assert structured[REMOTE_FILE_ID] == "key is left alone"


async def test_signed_upload_scalar_argument_only(env: _Env) -> None:
    adapter = env.adapter()

    await adapter.run_json_async(_args(**{FILE_ARGUMENT: ["file:a", "file:b"]}))

    assert env.server_calls == [
        (OUTER_TOOL, {FILE_ARGUMENT: ["file:a", "file:b"], "record_id": "r-1"})
    ]
    assert env.put_requests == []

    env.server_calls.clear()
    await adapter.run_json_async(_args(**{FILE_ARGUMENT: OUR_FILE_ID}))

    assert [name for name, _ in env.server_calls] == [ISSUING_TOOL, OUTER_TOOL]
    assert env.calls_to(OUTER_TOOL)[0][FILE_ARGUMENT] == REMOTE_FILE_ID


async def test_signed_upload_description_sentence(env: _Env) -> None:
    sentence = (
        " A registered workspace file (file:<id>) may be supplied for `document`; "
        "the host uploads it to the connector before this tool runs."
    )

    assert env.adapter().description == "Attach a document to a record." + sentence
    assert env.adapter(with_workspace=False).description == (
        "Attach a document to a record."
    )
    assert env.adapter(binding=None).description == "Attach a document to a record."


# --------------------------------------------------------------------------
# Every outcome removes the staged copy and keeps the signed URL private
# --------------------------------------------------------------------------


@dataclass
class _Scenario:
    id: str
    setup: Callable[[_Env], None]
    message: str | None  # None: the upload and the tool call succeed
    puts: int
    outer_calls: int = 0
    cancel: bool = False
    expected_prefix: bool = False


def _issuing_raises(env: _Env) -> None:
    def issue(args: dict[str, Any]) -> CallToolResult:
        raise RuntimeError(f"{LEAKY_MESSAGE} {UPLOAD_URL} {HEADER_SECRET}")

    env.issue = issue


def _issuing_is_error(env: _Env) -> None:
    env.issue = lambda args: CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"denied for {UPLOAD_URL} " + "x" * 400,
            )
        ],
        isError=True,
    )


def _malformed_response(env: _Env) -> None:
    payload = _ticket()
    del payload["file_id"]
    env.issue = lambda args: _issued(payload)


def _forbidden_header(env: _Env) -> None:
    env.issue = lambda args: _issued(_ticket(headers={"Authorization": HEADER_SECRET}))


def _rejected_url(env: _Env) -> None:
    env.issue = lambda args: _issued(
        _ticket(upload_url=f"https://evil.example.test/staged/abc?sig={QUERY_SECRET}")
    )


def _expired(env: _Env) -> None:
    env.now = datetime(2030, 1, 1, 0, 0, 1, tzinfo=timezone.utc)


def _put_403(env: _Env) -> None:
    async def behavior(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=f"denied {UPLOAD_URL}", request=request)

    env.put_behavior = behavior


def _put_connect_error(env: _Env) -> None:
    async def behavior(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {UPLOAD_URL}", request=request)

    env.put_behavior = behavior


class _RaisingTransport(httpx.AsyncBaseTransport):
    """Refuses before anything is sent, as the public-only transport does."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise self._error


def _unsafe_target(env: _Env) -> None:
    env.transport_factory = lambda: _RaisingTransport(
        UnsafePublicHttpTarget(
            f"Host must not resolve to a private network. {UPLOAD_URL}"
        )
    )


def _private_network_proxy(env: _Env) -> None:
    def refuse() -> str | None:
        raise PrivateNetworkHostError(f"proxy not trusted {UPLOAD_URL}")

    env.monkeypatch.setattr(mcp_adapter_module, "get_trusted_proxy_url", refuse)
    env.transport_factory = _REAL_SIGNED_UPLOAD_TRANSPORT


def _cancelled_during_put(env: _Env) -> None:
    async def behavior(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    env.put_behavior = behavior


_SCENARIOS = [
    _Scenario("success", lambda env: None, None, puts=1, outer_calls=1),
    _Scenario("issuing-raises", _issuing_raises, UPLOAD_FAILED, puts=0),
    _Scenario(
        "issuing-is-error",
        _issuing_is_error,
        REFUSED_PREFIX,
        puts=0,
        expected_prefix=True,
    ),
    _Scenario("malformed-response", _malformed_response, UPLOAD_FAILED, puts=0),
    _Scenario("forbidden-header", _forbidden_header, UPLOAD_FAILED, puts=0),
    _Scenario("rejected-url", _rejected_url, UNTRUSTED_URL, puts=0),
    _Scenario("expired", _expired, UPLOAD_FAILED, puts=0),
    _Scenario("put-403", _put_403, UPLOAD_FAILED, puts=1),
    _Scenario("put-connect-error", _put_connect_error, UPLOAD_FAILED, puts=1),
    _Scenario("unsafe-public-target", _unsafe_target, UPLOAD_FAILED, puts=0),
    _Scenario("private-network-proxy", _private_network_proxy, UPLOAD_FAILED, puts=0),
    _Scenario("cancelled", _cancelled_during_put, None, puts=1, cancel=True),
]


async def _run_scenario(env: _Env, scenario: _Scenario) -> dict[str, Any] | None:
    """Run one outcome; ``None`` means the call was cancelled."""
    scenario.setup(env)
    adapter = env.adapter()
    if not scenario.cancel:
        return await adapter.run_json_async(_args())
    task = asyncio.ensure_future(adapter.run_json_async(_args()))
    await asyncio.wait_for(env.put_started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return None


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.id for s in _SCENARIOS])
async def test_signed_upload_always_discards_staged_copy(
    env: _Env, scenario: _Scenario
) -> None:
    result = await _run_scenario(env, scenario)

    assert env.staging_leftovers() == []
    assert len(env.put_requests) == scenario.puts
    assert len(env.calls_to(OUTER_TOOL)) == scenario.outer_calls
    if result is None:
        return
    if scenario.message is None:
        assert result["is_error"] is False
    else:
        assert result["is_error"] is True
        if scenario.expected_prefix:
            assert _text(result).startswith(scenario.message)
        else:
            assert _text(result) == scenario.message


@pytest.mark.parametrize(
    "scenario",
    [
        s
        for s in _SCENARIOS
        if s.id in {"unsafe-public-target", "private-network-proxy"}
    ],
    ids=lambda s: s.id,
)
async def test_signed_upload_public_egress_refusal_fails_closed(
    env: _Env, scenario: _Scenario
) -> None:
    result = await _run_scenario(env, scenario)

    assert result is not None
    assert result["is_error"] is True
    assert _text(result) == UPLOAD_FAILED
    assert env.calls_to(OUTER_TOOL) == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=[s.id for s in _SCENARIOS])
async def test_signed_upload_url_never_leaks(
    env: _Env, scenario: _Scenario, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="xagent")

    result = await _run_scenario(env, scenario)

    seen = {
        "xagent log": caplog.text,
        "result": repr(result),
        "outer tool arguments": repr(env.calls_to(OUTER_TOOL)),
    }
    for where, text in seen.items():
        for secret in (UPLOAD_URL, QUERY_SECRET, HEADER_SECRET, CONNECTION_SECRET):
            assert secret not in text, f"{secret!r} leaked into {where}"


@pytest.fixture
def restore_logging():
    names = [
        "",
        "aiohttp",
        "sqlalchemy",
        "urllib3",
        "uvicorn.access",
        "uvicorn.error",
        "httpx",
        "httpcore",
        "xagent",
    ]
    saved = {}
    for name in names:
        logger = logging.getLogger(name)
        saved[name] = (
            logger.level,
            list(logger.handlers),
            logger.propagate,
            logger.disabled,
        )
    applied = logging_config._is_applied
    yield
    for name, (level, handlers, propagate, disabled) in saved.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers[:] = handlers
        logger.propagate = propagate
        logger.disabled = disabled
    logging_config._is_applied = applied


def test_deployment_logging_keeps_httpx_request_lines_off(
    monkeypatch: pytest.MonkeyPatch, restore_logging: None
) -> None:
    """httpx logs every request URL, query string included, at INFO."""
    monkeypatch.delenv("XAGENT_LOG_LEVEL", raising=False)

    logging_config.setup_logging(force=True)

    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING


# --------------------------------------------------------------------------
# The upload request
# --------------------------------------------------------------------------


async def test_signed_upload_put_shape(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[httpx.URL] = []
    original_check = mcp_adapter_module.check_signed_upload_url

    def recording_check(raw: str, binding: SignedUploadBinding) -> httpx.URL:
        url = original_check(raw, binding)
        checked.append(url)
        return url

    monkeypatch.setattr(mcp_adapter_module, "check_signed_upload_url", recording_check)

    stream_calls: list[tuple[str, Any]] = []
    original_stream = httpx.AsyncClient.stream

    def recording_stream(
        self: httpx.AsyncClient, method: str, url: Any, **kwargs: Any
    ) -> Any:
        stream_calls.append((method, url))
        return original_stream(self, method, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "stream", recording_stream)
    env.issue = lambda args: _issued(
        _ticket(
            headers={
                "Content-Type": "application/pdf",
                "x-signature": HEADER_SECRET,
                "x-checksum-sha256": "abc123",
            }
        )
    )

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is False
    (request,) = env.put_requests
    assert request["method"] == "PUT"
    assert len(checked) == 1 and len(stream_calls) == 1
    sent_method, sent_url = stream_calls[0]
    assert sent_method == "PUT"
    assert sent_url is checked[0]
    headers = {name.lower(): value for name, value in request["headers"]}
    names = [name.lower() for name, _ in request["headers"]]
    assert headers["content-type"] == "application/pdf"
    assert headers["x-signature"] == HEADER_SECRET
    assert headers["x-checksum-sha256"] == "abc123"
    assert headers["content-length"] == str(len(FILE_BYTES))
    assert names.count("content-length") == 1
    assert "transfer-encoding" not in names
    assert request["body"] == FILE_BYTES
    assert env.calls_to(ISSUING_TOOL) == [
        {
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "size_bytes": len(FILE_BYTES),
            "sha256": hashlib.sha256(FILE_BYTES).hexdigest(),
        }
    ]
    for name, value in request["headers"]:
        assert CONNECTION_SECRET not in value
        assert name.lower() not in {"authorization", "cookie"}


async def test_signed_upload_put_keeps_a_lower_case_content_length_single(
    env: _Env,
) -> None:
    env.issue = lambda args: _issued(
        _ticket(
            headers={
                "Content-Type": "application/pdf",
                "content-length": str(len(FILE_BYTES)),
            }
        )
    )

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is False
    (request,) = env.put_requests
    names = [name.lower() for name, _ in request["headers"]]
    assert names.count("content-length") == 1
    assert "transfer-encoding" not in names
    assert request["body"] == FILE_BYTES


async def test_signed_upload_does_not_follow_a_redirect(env: _Env) -> None:
    async def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "https://uploads.example.test/staged/other"},
            request=request,
        )

    env.put_behavior = redirect

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result) == UPLOAD_FAILED
    assert len(env.put_requests) == 1
    assert env.calls_to(OUTER_TOOL) == []
    assert env.staging_leftovers() == []


@pytest.mark.parametrize("status", [201, 204])
async def test_signed_upload_any_2xx_put_status_is_success(
    env: _Env, status: int
) -> None:
    async def behavior(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request)

    env.put_behavior = behavior

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is False
    assert env.calls_to(OUTER_TOOL) == [
        {FILE_ARGUMENT: REMOTE_FILE_ID, "record_id": "r-1"}
    ]


class _BodyThatRecordsReads(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.read = False

    async def __aiter__(self) -> Any:
        self.read = True
        yield b"x" * 1024

    async def aclose(self) -> None:
        return None


async def test_signed_upload_never_reads_the_put_response_body(env: _Env) -> None:
    body = _BodyThatRecordsReads()

    async def behavior(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=body, request=request)

    env.put_behavior = behavior

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is False
    assert body.read is False


async def test_signed_upload_sends_a_zero_byte_file(env: _Env) -> None:
    env.source.write_bytes(b"")

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is False
    (issuing_args,) = env.calls_to(ISSUING_TOOL)
    assert issuing_args["size_bytes"] == 0
    assert issuing_args["sha256"] == hashlib.sha256(b"").hexdigest()
    (request,) = env.put_requests
    headers = {name.lower(): value for name, value in request["headers"]}
    assert headers["content-length"] == "0"
    assert "transfer-encoding" not in headers
    assert request["body"] == b""


async def test_signed_upload_reads_the_file_off_the_event_loop(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_threads: list[int] = []

    class _RecordingFile:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def read(self, size: int = -1) -> bytes:
            read_threads.append(threading.get_ident())
            return self._handle.read(size)

        def close(self) -> None:
            self._handle.close()

        def __enter__(self) -> _RecordingFile:
            return self

        def __exit__(self, *exc_info: Any) -> None:
            self._handle.close()

    def recording_open(path: Any, mode: str = "r") -> _RecordingFile:
        return _RecordingFile(builtins.open(path, mode))

    monkeypatch.setattr(mcp_adapter_module, "open", recording_open, raising=False)

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is False
    assert env.put_requests[0]["body"] == FILE_BYTES
    # Both the hash and the upload read the file in several chunks.
    assert len(read_threads) > 4
    assert threading.get_ident() not in read_threads


async def test_signed_upload_transport_is_public_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)

    transport = _REAL_SIGNED_UPLOAD_TRANSPORT()
    try:
        assert isinstance(transport, PublicHttpTransport)
    finally:
        await transport.aclose()

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:3128")
    with pytest.raises(PrivateNetworkHostError):
        _REAL_SIGNED_UPLOAD_TRANSPORT()


@pytest.mark.parametrize(
    ("host", "message"),
    [
        # The public-network rule refuses these spellings of an IP address.
        pytest.param("2130706433", UPLOAD_FAILED, id="decimal-integer"),
        pytest.param("127.1", UPLOAD_FAILED, id="short-dotted"),
        pytest.param("127.0.0.1", UPLOAD_FAILED, id="loopback"),
        # The URL parser already refuses a leading-zero octet.
        pytest.param("010.0.0.1", UNTRUSTED_URL, id="leading-zero-octet"),
    ],
)
async def test_signed_upload_never_reaches_an_ip_address_host(
    env: _Env, monkeypatch: pytest.MonkeyPatch, host: str, message: str
) -> None:
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
    # A binding that lists the address spelling as an allowed upload host,
    # built directly so that the check does not depend on settings validation.
    binding = SignedUploadBinding(
        connection_host=BINDING.connection_host,
        tool_name=BINDING.tool_name,
        file_argument=BINDING.file_argument,
        issuing_tool_name=BINDING.issuing_tool_name,
        allowed_upload_hosts=frozenset({host}),
        upload_path_prefix=BINDING.upload_path_prefix,
        content_types=BINDING.content_types,
    )
    env.transport_factory = _REAL_SIGNED_UPLOAD_TRANSPORT
    env.issue = lambda args: _issued(
        _ticket(upload_url=f"https://{host}/staged/abc?sig={QUERY_SECRET}")
    )

    result = await env.adapter(binding=binding).run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result) == message
    assert env.calls_to(OUTER_TOOL) == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


# --------------------------------------------------------------------------
# The issuing call and the refusals before it
# --------------------------------------------------------------------------


async def test_signed_upload_issuing_error_text_is_redacted_and_capped(
    env: _Env, caplog: pytest.LogCaptureFixture
) -> None:
    _issuing_is_error(env)

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is True
    text = _text(result)
    assert text.startswith(REFUSED_PREFIX)
    assert len(text) - len(REFUSED_PREFIX) <= 200
    assert QUERY_SECRET not in text
    assert "https://uploads.example.test/staged/abc" in text
    assert env.calls_to(OUTER_TOOL) == []
    assert env.put_requests == []


async def test_signed_upload_issuing_error_text_has_credential_shapes_redacted(
    env: _Env,
) -> None:
    env.issue = lambda args: CallToolResult(
        content=[
            TextContent(
                type="text",
                text="denied api_key=sk-live-1234567890ABCDEF "
                "Authorization: Bearer abcdef123456secret",
            )
        ],
        isError=True,
    )

    result = await env.adapter().run_json_async(_args())

    assert _text(result).startswith(REFUSED_PREFIX)
    assert "sk-live-1234567890ABCDEF" not in _text(result)
    assert "abcdef123456secret" not in _text(result)
    assert "denied" in _text(result)
    assert env.calls_to(OUTER_TOOL) == []
    assert env.put_requests == []


async def test_signed_upload_non_ascii_header_value_is_a_malformed_response(
    env: _Env, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="xagent")
    env.issue = lambda args: _issued(_ticket(headers={"X-Meta": "caf\u00e9"}))

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result) == UPLOAD_FAILED
    assert env.put_requests == []
    assert env.calls_to(OUTER_TOOL) == []
    assert env.staging_leftovers() == []
    assert "caf" not in caplog.text
    assert "\u00e9" not in caplog.text
    assert "malformed" in caplog.text


async def test_signed_upload_issuing_401_with_failed_refresh_is_final(
    env: _Env,
) -> None:
    refresh_calls = 0

    def refresh() -> Any:
        nonlocal refresh_calls
        refresh_calls += 1
        raise RuntimeError(LEAKY_MESSAGE)

    def issue(args: dict[str, Any]) -> CallToolResult:
        raise _http_401()

    env.issue = issue

    result = await env.adapter(
        connection=_connection(**{_RUNTIME_KEY: refresh})
    ).run_json_async(_args())

    assert result == _delegated_authorization_failed_result()
    assert refresh_calls == 1
    assert env.calls_to(OUTER_TOOL) == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


async def test_signed_upload_issuing_401_without_a_refresh_is_an_authorization_failure(
    env: _Env,
) -> None:
    """A model-issued call that ends the same way gets the generic error."""

    def unauthorized(args: dict[str, Any]) -> CallToolResult:
        raise _http_401()

    env.issue = unauthorized
    env.business = unauthorized

    upload = await env.adapter().run_json_async(_args())
    model_call = await env.adapter(binding=None).run_json_async(_args())

    assert upload == _delegated_authorization_failed_result()
    assert _text(model_call) == "Error executing MCP tool."
    assert env.put_requests == []
    assert env.staging_leftovers() == []


async def test_signed_upload_issuing_retry_failure_is_the_retry_result(
    env: _Env,
) -> None:
    def refresh() -> dict[str, Any]:
        return _connection(headers={"Authorization": FRESH_AUTH})

    def issue(args: dict[str, Any]) -> CallToolResult:
        raise RuntimeError(LEAKY_MESSAGE)

    env.issue = _rejecting_the_initial_credential(env, issue)

    result = await env.adapter(
        connection=_connection(**{_RUNTIME_KEY: refresh})
    ).run_json_async(_args())

    assert result == _delegated_retry_failed_result()
    assert env.put_requests == []
    assert env.calls_to(OUTER_TOOL) == []
    assert env.staging_leftovers() == []


async def test_signed_upload_issuing_resolver_failure_keeps_its_failure_code(
    env: _Env,
) -> None:
    async def refresh(challenge: Any) -> Any:
        return ClassifiedToolFailure(failure_code="oauth_token_required")

    def issue(args: dict[str, Any]) -> CallToolResult:
        raise _http_401(['Bearer error="invalid_token"'])

    env.issue = issue

    result = await env.adapter(
        connection=_connection(**{_RESOLVER_KEY: refresh})
    ).run_json_async(_args())

    assert result == _delegated_authorization_failed_result(
        failure_code="oauth_token_required"
    )
    assert result["failure_code"] == "oauth_token_required"
    assert env.calls_to(OUTER_TOOL) == []
    assert env.put_requests == []


async def test_signed_upload_issuing_401_then_refresh_continues_the_upload(
    env: _Env,
) -> None:
    """The tool call uses the connection the issuing call refreshed to."""
    refresh_calls = 0

    def refresh() -> dict[str, Any]:
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls > 1:
            raise RuntimeError("refresh refused")
        return _connection(headers={"Authorization": FRESH_AUTH})

    env.issue = _rejecting_the_initial_credential(env, env.issue)
    env.business = _rejecting_the_initial_credential(env, env.business)

    result = await env.adapter(
        connection=_connection(**{_RUNTIME_KEY: refresh})
    ).run_json_async(_args())

    assert result["is_error"] is False
    assert refresh_calls == 1
    assert env.call_auth == [
        (ISSUING_TOOL, CONNECTION_AUTH),
        (ISSUING_TOOL, FRESH_AUTH),
        (OUTER_TOOL, FRESH_AUTH),
    ]
    assert len(env.put_requests) == 1
    assert env.staging_leftovers() == []


# The other host's own error text must not reach the model: the host check
# runs before the issuing reply is read.
OTHER_HOST_SAYS = "other host says hi"


def _issuing_is_error_from_the_other_host(env: _Env) -> None:
    env.issue = lambda args: CallToolResult(
        content=[TextContent(type="text", text=OTHER_HOST_SAYS)],
        isError=True,
    )


def _issuing_succeeds(env: _Env) -> None:
    pass


@pytest.mark.parametrize(
    "set_up_issuing",
    [
        pytest.param(_issuing_succeeds, id="issuing-succeeds"),
        pytest.param(_issuing_is_error_from_the_other_host, id="issuing-is-error"),
    ],
)
async def test_signed_upload_refuses_a_refreshed_issuing_connection_to_another_host(
    env: _Env, set_up_issuing: Callable[[_Env], None]
) -> None:
    def refresh() -> dict[str, Any]:
        return _connection(
            url="https://other.example.test/mcp",
            headers={"Authorization": FRESH_AUTH},
        )

    set_up_issuing(env)
    env.issue = _rejecting_the_initial_credential(env, env.issue)

    result = await env.adapter(
        connection=_connection(**{_RUNTIME_KEY: refresh})
    ).run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result) == UNTRUSTED_CONNECTOR
    assert OTHER_HOST_SAYS not in _text(result)
    assert env.put_requests == []
    assert env.calls_to(OUTER_TOOL) == []
    assert env.staging_leftovers() == []


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(f"file:{OUR_FILE_ID}", id="file-reference"),
        # A UUID the workspace does not know is refused too: the host check
        # runs before the workspace is asked.
        pytest.param("0b8f5d2e-7c41-4a9e-9d3b-2f6e8a1c4b70", id="unknown-bare-uuid"),
    ],
)
async def test_signed_upload_refuses_a_per_call_connection_to_another_host(
    env: _Env, value: str
) -> None:
    connection = _connection(
        **{
            _SLACK_REFRESH_KEY: lambda: _connection(
                url="https://other.example.test/mcp"
            )
        }
    )

    result = await env.adapter(connection=connection).run_json_async(
        _args(**{FILE_ARGUMENT: value})
    )

    assert result["is_error"] is True
    assert _text(result) == UNTRUSTED_CONNECTOR
    assert env.server_calls == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


async def test_signed_upload_tool_call_401_retries_without_a_second_upload(
    env: _Env,
) -> None:
    refresh_calls = 0

    def refresh() -> dict[str, Any]:
        nonlocal refresh_calls
        refresh_calls += 1
        return _connection(headers={"Authorization": FRESH_AUTH})

    env.business = _rejecting_the_initial_credential(env, env.business)

    result = await env.adapter(
        connection=_connection(**{_RUNTIME_KEY: refresh})
    ).run_json_async(_args())

    assert result["is_error"] is False
    assert refresh_calls == 1
    assert env.call_auth == [
        (ISSUING_TOOL, CONNECTION_AUTH),
        (OUTER_TOOL, CONNECTION_AUTH),
        (OUTER_TOOL, FRESH_AUTH),
    ]
    assert (
        env.calls_to(OUTER_TOOL)
        == [{FILE_ARGUMENT: REMOTE_FILE_ID, "record_id": "r-1"}] * 2
    )
    assert len(env.put_requests) == 1
    assert result["structured_content"]["document_id"] == f"file:{OUR_FILE_ID}"


async def test_signed_upload_issuing_exception_text_stays_out_of_logs_and_result(
    env: _Env, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="xagent")

    def issue(args: dict[str, Any]) -> CallToolResult:
        raise RuntimeError(LEAKY_MESSAGE)

    env.issue = issue

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result) == UPLOAD_FAILED
    assert "sig=abc" not in caplog.text
    assert "sig=abc" not in repr(result)
    assert "RuntimeError" in caplog.text


async def test_signed_upload_issuing_result_is_checked_before_sending(
    env: _Env,
) -> None:
    """A reply whose size header disagrees with the file is not used."""
    env.issue = lambda args: _issued(
        _ticket(headers={"Content-Length": str(len(FILE_BYTES) + 1)})
    )

    result = await env.adapter().run_json_async(_args())

    assert _text(result) == UPLOAD_FAILED
    assert env.put_requests == []
    assert env.calls_to(OUTER_TOOL) == []


async def test_signed_upload_refusals(
    env: _Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text_file = env.source.parent / "notes.txt"
    text_file.write_text("plain text")
    hashed: list[str] = []
    original_hash = mcp_adapter_module._file_size_and_sha256

    def recording_hash(path: Path) -> tuple[int, str]:
        hashed.append(path.name)
        return original_hash(path)

    monkeypatch.setattr(mcp_adapter_module, "_file_size_and_sha256", recording_hash)

    cases: list[tuple[str, MCPToolAdapter, str]] = [
        ("no workspace", env.adapter(with_workspace=False), FILE_UNAVAILABLE),
        (
            "connection host differs",
            env.adapter(connection=_connection(url="https://other.example.test/mcp")),
            UNTRUSTED_CONNECTOR,
        ),
        (
            "connection has no url",
            env.adapter(
                connection={"transport": "streamable_http", "headers": {}},
            ),
            UNTRUSTED_CONNECTOR,
        ),
    ]
    for label, adapter, message in cases:
        env.server_calls.clear()
        result = await adapter.run_json_async(_args())
        assert result["is_error"] is True, label
        assert _text(result) == message, label
        assert env.server_calls == [], label
        assert env.put_requests == [], label

    # An extension the binding has no content type for is refused before the
    # file is hashed.
    env.source = text_file
    env.server_calls.clear()
    result = await env.adapter().run_json_async(_args())
    assert _text(result) == UNSUPPORTED_TYPE
    assert hashed == []
    assert env.server_calls == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []
    env.source = text_file.parent / "report.pdf"

    # Four seconds before the URL expires counts as expired.
    env.now = datetime(2030, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=4)
    result = await env.adapter().run_json_async(_args())
    assert _text(result) == UPLOAD_FAILED
    assert env.put_requests == []
    assert env.calls_to(OUTER_TOOL) == []
    assert env.staging_leftovers() == []

    # Six seconds before is still usable.
    env.now = datetime(2030, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=6)
    result = await env.adapter().run_json_async(_args())
    assert result["is_error"] is False
    assert len(env.put_requests) == 1


@pytest.mark.parametrize(
    "expires_at",
    ["0001-01-01T00:00:00+00:00", "0001-01-01T00:00:00+14:00"],
)
async def test_signed_upload_expiry_at_the_earliest_date_is_an_upload_failure(
    env: _Env, expires_at: str
) -> None:
    env.issue = lambda args: _issued(_ticket(url_expires_at=expires_at))

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result) == UPLOAD_FAILED
    assert env.put_requests == []
    assert env.calls_to(OUTER_TOOL) == []


async def test_signed_upload_refusal_text_is_cut_before_redaction(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    redacted_lengths: list[int] = []
    original_redact = mcp_adapter_module.redact_urls_in_text

    def recording_redact(text: str) -> str:
        redacted_lengths.append(len(text))
        return original_redact(text)

    monkeypatch.setattr(mcp_adapter_module, "redact_urls_in_text", recording_redact)
    env.issue = lambda args: CallToolResult(
        content=[TextContent(type="text", text="denied " + "x" * 1_000_000)],
        isError=True,
    )

    result = await env.adapter().run_json_async(_args())

    assert _text(result).startswith(REFUSED_PREFIX + "denied ")
    assert redacted_lengths
    assert max(redacted_lengths) <= mcp_adapter_module._MCP_TOOL_ERROR_RAW_MAX_CHARS + 1


# A URL whose query is dropped by redaction, sized so that the cut at
# _MCP_TOOL_ERROR_RAW_MAX_CHARS lands 20 characters after it. Redaction then
# shortens the text enough for the cut to fall inside the 200-character result.
_LONG_QUERY_URL = "https://q.example.test/p?" + "q" * (
    mcp_adapter_module._MCP_TOOL_ERROR_RAW_MAX_CHARS - 20 - 25
)


@pytest.mark.parametrize(
    ("tail", "forbidden"),
    [
        # The cut falls between a password and its "@", leaving a legal
        # host:port; the whole token must be redacted.
        pytest.param(" https://alice:12345@up.example.test/x", "12345", id="cut-url"),
        # The cut falls in plain text; the cut mark must not reach the model.
        pytest.param(
            " " + "y" * 200,
            mcp_adapter_module._TEXT_TRUNCATION_MARK,
            id="cut-text",
        ),
    ],
)
async def test_signed_upload_refusal_text_cut_keeps_redaction_whole(
    env: _Env, tail: str, forbidden: str
) -> None:
    env.issue = lambda args: CallToolResult(
        content=[TextContent(type="text", text=_LONG_QUERY_URL + tail)],
        isError=True,
    )

    result = await env.adapter().run_json_async(_args())

    text = _text(result)
    assert text.startswith(REFUSED_PREFIX + "https://q.example.test/p ")
    assert forbidden not in text


_RUNTIME_BOUND_FILE_ARGUMENT = _connection(
    runtime_bindings=[
        {
            "source": {"input_type": "context", "key": "document"},
            "target": {"target_type": "tool_arguments", "key": FILE_ARGUMENT},
        }
    ],
    connector_runtime={
        "context": {"document": "rf_fromcontext"},
        "secrets": {},
        "auth_selector": {},
    },
)


@pytest.mark.parametrize(
    ("schema", "connection"),
    [
        pytest.param(
            {"type": "object", "properties": {"record_id": {"type": "string"}}},
            None,
            id="not-in-the-input-schema",
        ),
        pytest.param(
            _SCHEMA, _RUNTIME_BOUND_FILE_ARGUMENT, id="filled-by-a-runtime-binding"
        ),
    ],
)
async def test_signed_upload_binding_the_model_cannot_fill_is_ignored(
    env: _Env,
    caplog: pytest.LogCaptureFixture,
    schema: dict[str, Any],
    connection: Any,
) -> None:
    caplog.set_level(logging.WARNING, logger="xagent")

    adapter = env.adapter(schema=schema, connection=connection)
    await adapter.run_json_async(_args())

    assert adapter._signed_upload is None
    assert "A registered workspace file" not in adapter.description
    assert [
        record.getMessage()
        for record in caplog.records
        if "signed-upload settings" in record.getMessage()
    ] == [
        "MCP tool attach_document ignores its signed-upload settings: "
        "the model cannot fill the file argument"
    ]
    assert env.calls_to(ISSUING_TOOL) == []
    assert env.put_requests == []


@pytest.mark.parametrize(
    "constraint",
    [
        pytest.param({"maxLength": 20}, id="max-length"),
        pytest.param({"enum": ["a", "b"]}, id="enum"),
    ],
)
async def test_signed_upload_local_validation_runs_before_the_upload(
    env: _Env, constraint: dict[str, Any]
) -> None:
    schema = {
        "type": "object",
        "properties": {
            FILE_ARGUMENT: {"type": "string", **constraint},
            "record_id": {"type": "string"},
        },
    }

    result = await env.adapter(schema=schema).run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result).startswith("Invalid MCP tool arguments at document: ")
    assert env.server_calls == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


async def test_signed_upload_passes_a_bare_id_through_without_a_workspace(
    env: _Env,
) -> None:
    result = await env.adapter(with_workspace=False).run_json_async(
        _args(**{FILE_ARGUMENT: OUR_FILE_ID})
    )

    assert result["is_error"] is False
    assert env.server_calls == [
        (OUTER_TOOL, {FILE_ARGUMENT: OUR_FILE_ID, "record_id": "r-1"})
    ]
    assert env.put_requests == []


@pytest.mark.parametrize("step", ["staging", "hashing"])
async def test_signed_upload_os_error_before_the_issuing_call_is_generic(
    env: _Env, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    def fail(*_args: Any) -> Any:
        raise OSError("disk unavailable")

    if step == "staging":
        monkeypatch.setattr(env.workspace, "stage_file_for_external_upload", fail)
    else:
        monkeypatch.setattr(mcp_adapter_module, "_file_size_and_sha256", fail)

    result = await env.adapter().run_json_async(_args())

    assert result["is_error"] is True
    assert _text(result) == "Error executing MCP tool."
    assert env.server_calls == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


@pytest.mark.parametrize("step", ["staging", "hashing", "issuing"])
async def test_signed_upload_cancellation_before_the_put_discards_the_copy(
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

    task = asyncio.ensure_future(env.adapter().run_json_async(_args()))
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert env.staging_leftovers() == []
    assert env.calls_to(OUTER_TOOL) == []
    assert env.put_requests == []


@pytest.fixture
def approval_gate() -> Any:
    async def gate(_call: GatedCall) -> GateDecision:
        return GateDecision.require_approval("interaction-1")

    async def resume(*, executor: Any, **_: Any) -> ToolInteractionSettlement:
        return ToolInteractionSettlement.succeeded(await executor(_args()))

    handle = register_mcp_approval_gate(task_source="slack", gate=gate, resume=resume)
    yield
    unregister_mcp_approval_gate(handle)


async def test_signed_upload_runs_only_when_the_approved_call_is_replayed(
    env: _Env, approval_gate: None
) -> None:
    (tool,) = gate_mcp_tools([env.adapter()], connection={"id": 41})
    context = ToolCallExecutionContext(
        task_source="slack",
        task_id="1",
        run_id="run-1",
        turn_id="turn-1",
        tool_call_id="call-1",
        pattern="react",
        react_step_id="react-1",
        dag_step_id=None,
    )

    with bind_tool_call_execution_context(context):
        paused = await tool.run_json_async(_args())
        assert env.server_calls == []
        assert env.put_requests == []
        settled = await tool.resume_user_interaction(
            interaction_id=paused["interaction_id"], response="approve"
        )

    assert settled.status == "succeeded"
    assert len(env.calls_to(ISSUING_TOOL)) == 1
    assert len(env.put_requests) == 1
    assert env.calls_to(OUTER_TOOL) == [
        {FILE_ARGUMENT: REMOTE_FILE_ID, "record_id": "r-1"}
    ]
    assert env.staging_leftovers() == []


# --------------------------------------------------------------------------
# The workspace decides which files can be uploaded
# --------------------------------------------------------------------------


def _scoped_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[_Env, dict]:
    """A real workspace over real file records of two users and two tasks."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'records.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(
        "xagent.core.storage.manager.create_db_session", lambda: session_factory()
    )

    base_dir = tmp_path / "workspaces"
    uploads = base_dir / "uploads"
    uploads.mkdir(parents=True)
    ids = {
        "ok": "00000000-0000-4000-8000-000000000001",
        "other_user": "00000000-0000-4000-8000-000000000002",
        "other_task": "00000000-0000-4000-8000-000000000003",
        "deleted": "00000000-0000-4000-8000-000000000004",
    }
    db = session_factory()
    try:
        owner = User(username="owner", password_hash="hash")
        stranger = User(username="stranger", password_hash="hash")
        db.add_all([owner, stranger])
        db.flush()
        db.add_all(
            [
                Task(id=1, user_id=owner.id, title="Current task"),
                Task(id=2, user_id=owner.id, title="Another task"),
            ]
        )
        db.commit()
        records = {
            "ok": (owner.id, 1),
            "other_user": (stranger.id, None),
            "other_task": (owner.id, 2),
            "deleted": (owner.id, 1),
        }
        for name, (user_id, task_id) in records.items():
            path = uploads / f"{name}.pdf"
            path.write_bytes(FILE_BYTES)
            db.add(
                UploadedFile(
                    file_id=ids[name],
                    user_id=user_id,
                    task_id=task_id,
                    filename=path.name,
                    storage_path=str(path),
                    file_size=len(FILE_BYTES),
                )
            )
        db.commit()
        db.delete(db.query(UploadedFile).filter_by(file_id=ids["deleted"]).one())
        db.commit()
        owner_id = int(owner.id)
    finally:
        db.close()

    workspace = TaskWorkspace(id="web_task_1", base_dir=str(base_dir))
    workspace.owner_user_id = owner_id
    env = _Env(tmp_path, monkeypatch, workspace=workspace)
    return env, ids


@pytest.mark.parametrize(
    "record",
    [
        pytest.param("other_user", id="another-end-user"),
        pytest.param("other_task", id="same-user-another-task"),
        pytest.param("deleted", id="record-deleted"),
    ],
)
async def test_signed_upload_respects_workspace_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: str
) -> None:
    env, ids = _scoped_env(tmp_path, monkeypatch)

    result = await env.adapter().run_json_async(
        _args(**{FILE_ARGUMENT: f"file:{ids[record]}"})
    )

    assert result["is_error"] is True
    assert _text(result) == FILE_UNAVAILABLE
    assert env.calls_to(ISSUING_TOOL) == []
    assert env.calls_to(OUTER_TOOL) == []
    assert env.put_requests == []
    assert env.staging_leftovers() == []


@pytest.mark.parametrize(
    "record",
    [
        pytest.param("other_user", id="another-end-user"),
        pytest.param("other_task", id="same-user-another-task"),
        pytest.param("deleted", id="record-deleted"),
    ],
)
async def test_signed_upload_passes_a_bare_id_the_task_cannot_use_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: str
) -> None:
    """Such a bare UUID is taken as the connector's own file id."""
    env, ids = _scoped_env(tmp_path, monkeypatch)

    result = await env.adapter().run_json_async(_args(**{FILE_ARGUMENT: ids[record]}))

    assert result["is_error"] is False
    assert env.calls_to(ISSUING_TOOL) == []
    assert env.calls_to(OUTER_TOOL) == [
        {FILE_ARGUMENT: ids[record], "record_id": "r-1"}
    ]
    assert env.put_requests == []
    assert env.staging_leftovers() == []


async def test_signed_upload_uploads_a_file_the_workspace_may_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same setup accepts the owner's file of the current task."""
    env, ids = _scoped_env(tmp_path, monkeypatch)

    result = await env.adapter().run_json_async(
        _args(**{FILE_ARGUMENT: f"file:{ids['ok']}"})
    )

    assert result["is_error"] is False
    assert len(env.calls_to(ISSUING_TOOL)) == 1
    assert env.put_requests[0]["body"] == FILE_BYTES
    assert env.calls_to(OUTER_TOOL) == [
        {FILE_ARGUMENT: REMOTE_FILE_ID, "record_id": "r-1"}
    ]
    assert env.staging_leftovers() == []

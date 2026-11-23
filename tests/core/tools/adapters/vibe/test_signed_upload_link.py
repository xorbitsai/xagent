"""A link entry of a connector's signed-upload settings, at the tool adapter.

A tool with a link entry takes a workspace image reference in one string
argument and calls the connector with a short-lived download link instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool

from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPToolAdapter,
    _build_mcp_tool_adapter,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.signed_upload import (
    SignedUploadBinding,
    SignedUploadLink,
)
from xagent.core.workspace import ExternalMediaNotIssuableError, TaskWorkspace

HOST = "mcp.example.test"
CONNECTION_URL = f"https://{HOST}/mcp"
OTHER_CONNECTION_URL = "https://other.example.test/mcp"
TOOL = "set_logo"
ARGUMENT = "image_url"
FILE_ID = "6f1c2d3e-4b5a-4c6d-8e7f-9a0b1c2d3e4f"
TICKET = "ticket-secret-value"
LINK_URL = (
    f"https://api.example.test/api/files/external-media/{FILE_ID}?ticket={TICKET}"
)
TASK_ID = 41

FILE_UNAVAILABLE = (
    "The file cannot be sent to this tool here; pass an https URL instead."
)
FILE_NOT_FOUND = "file_id not found or not accessible."
PREPARATION_FAILED = "The image could not be prepared for this tool."
TOO_LONG = (
    "The download link the host created is longer than this tool's argument "
    "allows, so the tool was not called."
)
UNTRUSTED_CONNECTOR = "Upload refused: untrusted connector."
DESCRIPTION_NOTE = (
    " A registered workspace image (file:<id>) may be supplied for "
    f"`{ARGUMENT}`; the host replaces it with a short-lived HTTPS download link "
    "that expires within minutes, so the tool must download the image during "
    "the call and must not store the link. A plain https URL is passed through "
    "unchanged."
)

LINK = SignedUploadLink(connection_host=HOST, tool_name=TOOL, link_argument=ARGUMENT)

_RUNTIME_KEY = mcp_adapter_module._RUNTIME_CONNECTION_REFRESH_KEY
_RESOLVER_KEY = mcp_adapter_module._OAUTH_TOKEN_RESOLVER_REFRESH_KEY

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        ARGUMENT: {"type": "string"},
        "display_logo_in_dashboard": {"type": "boolean"},
    },
}


def _schema(**argument: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            ARGUMENT: {"type": "string", **argument},
            "display_logo_in_dashboard": {"type": "boolean"},
        },
    }


def _connection(url: str = CONNECTION_URL, **extra: Any) -> dict[str, Any]:
    return {"transport": "streamable_http", "url": url, **extra}


def _http_401(authenticate: list[str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", CONNECTION_URL)
    headers = [(b"www-authenticate", v.encode()) for v in (authenticate or [])]
    response = httpx.Response(401, headers=headers, request=request)
    return httpx.HTTPStatusError("unauthorized", request=request, response=response)


def _ok(payload: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


class _Workspace:
    """A workspace that issues links; ``outcome`` is the link or an error."""

    owner_user_id = 7
    db_task_id = TASK_ID

    def __init__(self, outcome: Any = LINK_URL) -> None:
        self.outcome = outcome
        self.issued: list[tuple[str, str]] = []

    def issue_external_media_url_detached(self, file_id: str, purpose: str) -> str:
        self.issued.append((file_id, purpose))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return str(self.outcome)


class _Server:
    """A fake connector: records every tool call it receives."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.connections: list[Any] = []
        self.respond: Callable[[dict[str, Any]], Any] = lambda args: _ok(
            {"logo_updated": True, "image_url": args.get(ARGUMENT)}
        )
        server = self

        class _Session:
            async def initialize(self) -> None:
                return None

            async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
                server.calls.append((name, json.loads(json.dumps(arguments))))
                return server.respond(arguments)

        @asynccontextmanager
        async def _create_session(connection: Any):
            server.connections.append(connection)
            yield _Session()

        monkeypatch.setattr(mcp_adapter_module, "create_session", _create_session)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [args for _, args in self.calls]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _Server:
    return _Server(monkeypatch)


@pytest.fixture
def workspace() -> _Workspace:
    return _Workspace()


def _adapter(
    workspace: Any,
    *,
    link: SignedUploadLink | None = LINK,
    schema: dict[str, Any] = _SCHEMA,
    connection: Any = None,
    binding: SignedUploadBinding | None = None,
) -> MCPToolAdapter:
    return MCPToolAdapter(
        mcp_tool=MCPTool(name=TOOL, description="Set the logo.", inputSchema=schema),
        connection=_connection() if connection is None else connection,
        workspace=workspace,
        signed_upload=binding,
        signed_upload_link=link,
    )


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


def _link_logs(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == level and "download link" in record.getMessage()
    ]


# --------------------------------------------------------------------------
# Replacement
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(f"file:{FILE_ID}", id="file-reference"),
        pytest.param(FILE_ID, id="bare-uuid"),
        pytest.param(f"  file:{FILE_ID}  ", id="padded-with-whitespace"),
        pytest.param(f"file://{FILE_ID}", id="legacy-double-slash-form"),
    ],
)
async def test_a_file_reference_is_replaced_by_a_link_before_the_call(
    server: _Server, workspace: _Workspace, value: str
) -> None:
    result = await _adapter(workspace).run_json_async({ARGUMENT: value})

    assert result["is_error"] is False
    assert workspace.issued == [(FILE_ID, "connector-image")]
    assert server.tool_calls == [{ARGUMENT: LINK_URL}]


async def test_other_arguments_are_passed_unchanged(
    server: _Server, workspace: _Workspace
) -> None:
    await _adapter(workspace).run_json_async(
        {ARGUMENT: f"file:{FILE_ID}", "display_logo_in_dashboard": True}
    )

    assert server.tool_calls == [
        {ARGUMENT: LINK_URL, "display_logo_in_dashboard": True}
    ]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("https://example.com/a.png", id="https-url"),
        pytest.param("http://example.com/a.png", id="http-url"),
        pytest.param("", id="empty"),
        pytest.param("not a ref", id="free-text"),
        pytest.param("file:abc?x=1", id="query-on-a-non-id"),
        pytest.param(f"file:{FILE_ID}?x=1", id="query-on-an-id"),
        pytest.param(f"file:{FILE_ID}/sub", id="path-on-an-id"),
        pytest.param("file:///etc/passwd", id="real-file-uri"),
    ],
)
async def test_a_value_that_is_not_a_file_reference_is_passed_through(
    server: _Server, workspace: _Workspace, value: str
) -> None:
    result = await _adapter(workspace).run_json_async({ARGUMENT: value})

    assert result["is_error"] is False
    assert workspace.issued == []
    assert server.tool_calls == [{ARGUMENT: value}]


async def test_a_call_without_the_argument_is_passed_through(
    server: _Server, workspace: _Workspace
) -> None:
    await _adapter(workspace).run_json_async({"display_logo_in_dashboard": True})

    assert workspace.issued == []
    assert server.tool_calls == [{"display_logo_in_dashboard": True}]


async def test_a_tool_without_a_link_entry_passes_the_reference_on_as_it_is(
    server: _Server, workspace: _Workspace
) -> None:
    adapter = _adapter(workspace, link=None)

    result = await adapter.run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert result["is_error"] is False
    assert workspace.issued == []
    assert server.tool_calls == [{ARGUMENT: f"file:{FILE_ID}"}]
    assert adapter._signed_upload_link is None
    assert DESCRIPTION_NOTE not in adapter.description


async def test_a_link_in_the_result_becomes_the_file_reference_again(
    server: _Server, workspace: _Workspace
) -> None:
    def respond(args: dict[str, Any]) -> CallToolResult:
        url = args[ARGUMENT]
        escaped_text = json.dumps({"image_url": url}).replace("/", r"\/")
        return CallToolResult(
            content=[
                TextContent(type="text", text=url),
                TextContent(type="text", text=f"Downloaded {url} just now, thanks"),
                TextContent(type="text", text=escaped_text),
            ],
            structuredContent={"whole": url, "inside": f"saw {url} twice {url}"},
            isError=False,
        )

    server.respond = respond

    result = await _adapter(workspace).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    ref = f"file:{FILE_ID}"
    texts = [item["text"] for item in result["content"]]
    assert texts[0] == ref
    assert texts[1] == f"Downloaded {ref} just now, thanks"
    assert texts[2] == json.dumps({"image_url": ref})
    assert result["structured_content"] == {
        "whole": ref,
        "inside": f"saw {ref} twice {ref}",
    }
    assert TICKET not in json.dumps(result)
    assert "external-media" not in json.dumps(result)


# --------------------------------------------------------------------------
# Failure: the tool is not called
# --------------------------------------------------------------------------


def _real_workspace(tmp_path: Path, **overrides: Any) -> TaskWorkspace:
    workspace = TaskWorkspace(
        "web_task_41", str(tmp_path), db_task_id=overrides.get("db_task_id", TASK_ID)
    )
    workspace.owner_user_id = overrides.get("owner_user_id", 7)
    return workspace


def _service_error(name: str, message: str) -> Exception:
    from xagent.web.services import external_media

    return getattr(external_media, name)(message)


@pytest.mark.parametrize(
    ("service_error", "expected"),
    [
        pytest.param(
            ("ExternalMediaNotFound", "hidden detail"),
            FILE_NOT_FOUND,
            id="not-found-or-another-task",
        ),
        pytest.param(
            (
                "ExternalMediaUnsupportedImage",
                "Only PNG, JPEG or WebP images can be delivered",
            ),
            "Only PNG, JPEG or WebP images can be delivered",
            id="unsupported-image",
        ),
        pytest.param(
            (
                "ExternalMediaUnsupportedImage",
                "Registered image exceeds the 10 MB limit",
            ),
            "Registered image exceeds the 10 MB limit",
            id="too-large",
        ),
        pytest.param(
            (
                "ExternalMediaNotConfigured",
                "A public HTTPS XAGENT_S2S_API_BASE_URL or "
                "XAGENT_PUBLIC_API_BASE_URL is required for external media delivery",
            ),
            "A public HTTPS XAGENT_S2S_API_BASE_URL or "
            "XAGENT_PUBLIC_API_BASE_URL is required for external media delivery",
            id="no-public-address",
        ),
        pytest.param(
            ("ExternalMediaOutOfScope", "Registered image bytes are unavailable"),
            "Registered image bytes are unavailable",
            id="out-of-scope",
        ),
        pytest.param(
            (
                "ExternalMediaUnavailable",
                "Registered image could not be read for external delivery",
            ),
            PREPARATION_FAILED,
            id="storage-unavailable",
        ),
    ],
)
async def test_a_link_that_cannot_be_issued_ends_the_call_with_a_fixed_error(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service_error: tuple[str, str],
    expected: str,
) -> None:
    from xagent.core.storage import manager
    from xagent.web.services import external_media

    def issue(*args: Any, **kwargs: Any) -> Any:
        raise _service_error(*service_error)

    monkeypatch.setattr(
        manager, "create_db_session", lambda: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(external_media, "issue_external_media_url", issue)

    result = await _adapter(_real_workspace(tmp_path)).run_json_async(
        {ARGUMENT: f"file:{FILE_ID}"}
    )

    assert result["is_error"] is True
    assert _text(result) == expected
    assert server.calls == []


async def test_a_workspace_without_a_task_ends_the_call_without_calling_the_tool(
    server: _Server, tmp_path: Path
) -> None:
    result = await _adapter(_real_workspace(tmp_path, db_task_id=None)).run_json_async(
        {ARGUMENT: f"file:{FILE_ID}"}
    )

    assert result["is_error"] is True
    assert _text(result) == FILE_NOT_FOUND
    assert server.calls == []


async def test_a_workspace_without_an_owner_ends_the_call_without_calling_the_tool(
    server: _Server, tmp_path: Path
) -> None:
    result = await _adapter(
        _real_workspace(tmp_path, owner_user_id=None)
    ).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert result["is_error"] is True
    assert _text(result) == FILE_NOT_FOUND
    assert server.calls == []


async def test_a_database_failure_while_issuing_does_not_call_the_tool(
    server: _Server,
) -> None:
    workspace = _Workspace(RuntimeError(f"db down {LINK_URL}"))

    result = await _adapter(workspace).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert result["is_error"] is True
    assert _text(result) == "Error executing MCP tool."
    assert server.calls == []


async def test_an_image_the_service_refuses_names_the_webp_wording(
    server: _Server,
) -> None:
    workspace = _Workspace(
        ExternalMediaNotIssuableError("Only PNG, JPEG or WebP images can be delivered")
    )

    result = await _adapter(workspace).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert _text(result) == "Only PNG, JPEG or WebP images can be delivered"
    assert server.calls == []


async def test_a_not_issuable_error_is_not_read_as_an_unexpected_value_error(
    server: _Server,
) -> None:
    # ExternalMediaNotIssuableError is a ValueError; it keeps its own text
    # instead of the fixed preparation text.
    workspace = _Workspace(ExternalMediaNotIssuableError("its own text"))

    result = await _adapter(workspace).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert _text(result) == "its own text"


@pytest.mark.parametrize(
    "secret_error",
    [
        pytest.param(f"ticket={TICKET}", id="ticket-in-the-message"),
        pytest.param(
            "XAGENT_EXTERNAL_MEDIA_URL_TTL_SECONDS must be an integer",
            id="variable-name",
        ),
        pytest.param(f"bad value {LINK_URL}", id="whole-link"),
    ],
)
async def test_an_unexpected_value_error_never_reaches_the_model_or_the_log(
    server: _Server,
    caplog: pytest.LogCaptureFixture,
    secret_error: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    workspace = _Workspace(ValueError(secret_error))

    result = await _adapter(workspace).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert _text(result) == PREPARATION_FAILED
    assert server.calls == []
    assert secret_error not in caplog.text
    assert TICKET not in caplog.text
    assert "XAGENT_" not in caplog.text
    assert _link_logs(caplog, logging.WARNING) == [
        "MCP tool set_logo refused a download link reason=preparation_failed "
        "error_type=ValueError"
    ]


async def test_a_call_that_raises_with_the_link_in_its_message_leaks_nothing(
    server: _Server,
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    def boom(args: dict[str, Any]) -> Any:
        raise RuntimeError(f"fetching {args[ARGUMENT]} failed")

    server.respond = boom

    result = await _adapter(workspace).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert _text(result) == "Error executing MCP tool."
    assert "ticket=" not in json.dumps(result)
    assert "ticket=" not in caplog.text


async def test_without_a_workspace_an_explicit_file_reference_is_refused(
    server: _Server, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)

    result = await _adapter(None).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert result["is_error"] is True
    assert _text(result) == FILE_UNAVAILABLE
    assert server.calls == []
    assert _link_logs(caplog, logging.WARNING) == [
        "MCP tool set_logo refused a download link reason=no_workspace"
    ]


@pytest.mark.parametrize(
    "value", [FILE_ID, "https://example.com/a.png", "", "file:abc?x=1"]
)
async def test_without_a_workspace_any_other_value_is_passed_through(
    server: _Server, value: str
) -> None:
    result = await _adapter(None).run_json_async({ARGUMENT: value})

    assert result["is_error"] is False
    assert server.tool_calls == [{ARGUMENT: value}]


# --------------------------------------------------------------------------
# Length of the link
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("constraint", "refused"),
    [
        pytest.param({}, False, id="no-maxlength"),
        pytest.param({"maxLength": len(LINK_URL)}, False, id="exactly-long-enough"),
        pytest.param({"maxLength": len(LINK_URL) + 100}, False, id="longer-than-link"),
        pytest.param({"maxLength": len(LINK_URL) - 1}, True, id="one-too-short"),
        pytest.param(
            {"maxLength": len(f"file:{FILE_ID}")}, True, id="as-short-as-the-reference"
        ),
        pytest.param({"maxLength": "5"}, False, id="not-an-integer"),
        pytest.param({"maxLength": True}, False, id="a-boolean"),
        pytest.param({"maxLength": -1}, False, id="negative"),
    ],
)
async def test_the_link_must_fit_the_maxlength_of_the_argument(
    server: _Server,
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
    constraint: dict[str, Any],
    refused: bool,
) -> None:
    caplog.set_level(logging.WARNING)
    adapter = _adapter(workspace, schema=_schema(**constraint))

    result = await adapter.run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    if refused:
        assert result["is_error"] is True
        assert _text(result) == TOO_LONG
        assert server.calls == []
        assert _link_logs(caplog, logging.WARNING) == [
            "MCP tool set_logo refused a download link reason=too_long"
        ]
    else:
        assert result["is_error"] is False
        assert server.tool_calls == [{ARGUMENT: LINK_URL}]


# --------------------------------------------------------------------------
# Which connection
# --------------------------------------------------------------------------


async def test_a_connection_to_another_host_than_the_settings_is_refused(
    server: _Server, workspace: _Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    adapter = _adapter(workspace, connection=_connection(OTHER_CONNECTION_URL))

    result = await adapter.run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert result["is_error"] is True
    assert _text(result) == UNTRUSTED_CONNECTOR
    assert server.calls == []
    assert workspace.issued == []
    assert (
        "MCP tool set_logo refused a download link reason=untrusted_connector"
        in _link_logs(caplog, logging.WARNING)
    )


@pytest.mark.parametrize(
    "connection",
    [
        pytest.param({"transport": "streamable_http"}, id="no-url"),
        pytest.param({"transport": "streamable_http", "url": 7}, id="url-not-a-string"),
        pytest.param(
            {"transport": "streamable_http", "url": "https://[bad/"}, id="unparsable"
        ),
    ],
)
async def test_a_connection_without_a_usable_host_is_refused(
    server: _Server, workspace: _Workspace, connection: dict[str, Any]
) -> None:
    result = await _adapter(workspace, connection=connection).run_json_async(
        {ARGUMENT: f"file:{FILE_ID}"}
    )

    assert _text(result) == UNTRUSTED_CONNECTOR
    assert server.calls == []
    assert workspace.issued == []


async def test_a_plain_url_is_passed_to_a_connection_on_any_host(
    server: _Server, workspace: _Workspace
) -> None:
    adapter = _adapter(workspace, connection=_connection(OTHER_CONNECTION_URL))

    result = await adapter.run_json_async({ARGUMENT: "https://example.com/a.png"})

    assert result["is_error"] is False
    assert server.tool_calls == [{ARGUMENT: "https://example.com/a.png"}]


def _refreshed_connection(url: str) -> Callable[..., Any]:
    def refresh(*args: Any) -> dict[str, Any]:
        return _connection(url)

    return refresh


def _fail_once_with_401(server: _Server, authenticate: list[str] | None = None) -> None:
    first = server.respond

    def respond(args: dict[str, Any]) -> Any:
        if len(server.calls) == 1:
            raise _http_401(authenticate)
        return first(args)

    server.respond = respond


_INVALID_TOKEN = ['Bearer error="invalid_token"']


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("runtime", id="runtime-connection-refresh"),
        pytest.param("resolver", id="oauth-resolver-refresh"),
    ],
)
async def test_a_retry_over_a_refreshed_connection_to_another_host_is_refused(
    server: _Server, workspace: _Workspace, path: str
) -> None:
    refresh = _refreshed_connection(OTHER_CONNECTION_URL)
    key = _RUNTIME_KEY if path == "runtime" else _RESOLVER_KEY
    _fail_once_with_401(server, _INVALID_TOKEN if path == "resolver" else None)
    adapter = _adapter(workspace, connection=_connection(**{key: refresh}))

    result = await adapter.run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert result == {
        "content": [{"text": UNTRUSTED_CONNECTOR}],
        "is_error": True,
    }
    # Only the first call went out: the retry was not made.
    assert len(server.calls) == 1
    assert [c.get("url") for c in server.connections] == [CONNECTION_URL]


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("runtime", id="runtime-connection-refresh"),
        pytest.param("resolver", id="oauth-resolver-refresh"),
    ],
)
async def test_a_retry_over_a_refreshed_connection_to_the_same_host_reuses_the_link(
    server: _Server, workspace: _Workspace, path: str
) -> None:
    refresh = _refreshed_connection(f"https://{HOST}/other-path")
    key = _RUNTIME_KEY if path == "runtime" else _RESOLVER_KEY
    _fail_once_with_401(server, _INVALID_TOKEN if path == "resolver" else None)
    adapter = _adapter(workspace, connection=_connection(**{key: refresh}))

    result = await adapter.run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert result["is_error"] is False
    assert server.tool_calls == [{ARGUMENT: LINK_URL}, {ARGUMENT: LINK_URL}]
    assert workspace.issued == [(FILE_ID, "connector-image")]
    assert result["structured_content"]["image_url"] == f"file:{FILE_ID}"


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("runtime", id="runtime-connection-refresh"),
        pytest.param("resolver", id="oauth-resolver-refresh"),
    ],
)
async def test_a_retry_without_a_replaced_link_is_not_checked_for_the_host(
    server: _Server, workspace: _Workspace, path: str
) -> None:
    refresh = _refreshed_connection(OTHER_CONNECTION_URL)
    key = _RUNTIME_KEY if path == "runtime" else _RESOLVER_KEY
    _fail_once_with_401(server, _INVALID_TOKEN if path == "resolver" else None)
    adapter = _adapter(workspace, connection=_connection(**{key: refresh}))

    result = await adapter.run_json_async({ARGUMENT: "https://example.com/a.png"})

    assert result["is_error"] is False
    assert len(server.calls) == 2
    assert workspace.issued == []


# --------------------------------------------------------------------------
# Which arguments take part
# --------------------------------------------------------------------------

_RUNTIME_BOUND = _connection(
    runtime_bindings=[
        {
            "source": {"input_type": "context", "key": "logo"},
            "target": {"target_type": "tool_arguments", "key": ARGUMENT},
        }
    ],
    connector_runtime={
        "context": {"logo": f"file:{FILE_ID}"},
        "secrets": {},
        "auth_selector": {},
    },
)


@pytest.mark.parametrize(
    ("schema", "connection", "why"),
    [
        pytest.param(
            {"type": "object", "properties": {"other": {"type": "string"}}},
            None,
            "not in the input schema",
            id="parameter-missing",
        ),
        pytest.param(
            {"type": "object", "properties": {ARGUMENT: {"type": "integer"}}},
            None,
            "not a string",
            id="integer",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {
                    ARGUMENT: {"type": "array", "items": {"type": "string"}}
                },
            },
            None,
            "not a string",
            id="array",
        ),
        pytest.param(
            {"type": "object", "properties": {ARGUMENT: {"type": ["string", "null"]}}},
            None,
            "not a string",
            id="string-or-null",
        ),
        pytest.param(
            {"type": "object", "properties": {ARGUMENT: {"description": "no type"}}},
            None,
            "not a string",
            id="no-type",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {ARGUMENT: {"anyOf": [{"type": "string"}]}},
            },
            None,
            "not a string",
            id="any-of",
        ),
        pytest.param(
            {"type": "object", "properties": "nonsense"},
            None,
            "not in the input schema",
            id="properties-not-a-mapping",
        ),
        pytest.param(
            _SCHEMA,
            _RUNTIME_BOUND,
            "filled by the runtime",
            id="runtime-bound",
        ),
    ],
)
def test_a_link_entry_the_model_cannot_use_is_ignored_and_the_tool_still_loads(
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
    schema: dict[str, Any],
    connection: Any,
    why: str,
) -> None:
    caplog.set_level(logging.WARNING)

    adapter = _adapter(workspace, schema=schema, connection=connection)

    assert adapter._signed_upload_link is None
    assert DESCRIPTION_NOTE not in adapter.description
    assert [
        r.getMessage() for r in caplog.records if "link settings" in r.getMessage()
    ] == [
        "MCP tool set_logo ignores its link settings: the model cannot fill "
        f"the `{ARGUMENT}` argument"
    ]


async def test_a_runtime_bound_value_is_not_issued_a_link(
    server: _Server, workspace: _Workspace
) -> None:
    adapter = _adapter(workspace, connection=_RUNTIME_BOUND)

    # The model offers a reference of its own too; it is dropped and the
    # runtime value goes through as it is.
    await adapter.run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert workspace.issued == []
    assert server.tool_calls == [{ARGUMENT: f"file:{FILE_ID}"}]


async def test_a_binding_on_another_argument_runs_beside_a_link_entry(
    server: _Server, workspace: _Workspace
) -> None:
    binding = SignedUploadBinding(
        connection_host=HOST,
        tool_name=TOOL,
        file_argument="document",
        issuing_tool_name="request_upload",
        allowed_upload_hosts=frozenset({"uploads.example.test"}),
        upload_path_prefix="/staged/",
        content_types={".pdf": "application/pdf"},
    )
    schema = {
        "type": "object",
        "properties": {ARGUMENT: {"type": "string"}, "document": {"type": "string"}},
    }
    adapter = _adapter(workspace, binding=binding, schema=schema)

    # The document is not a workspace reference, so only the link applies.
    await adapter.run_json_async(
        {ARGUMENT: f"file:{FILE_ID}", "document": "rf_already_remote"}
    )

    assert server.tool_calls == [{ARGUMENT: LINK_URL, "document": "rf_already_remote"}]
    assert adapter._signed_upload is binding
    assert adapter._signed_upload_link is LINK


# --------------------------------------------------------------------------
# What the model is told
# --------------------------------------------------------------------------


def test_the_description_tells_the_model_how_to_pass_an_image(
    workspace: _Workspace,
) -> None:
    adapter = _adapter(workspace)

    assert adapter.description.endswith(DESCRIPTION_NOTE)
    assert adapter.description.startswith("Set the logo.")


def test_the_description_says_nothing_without_a_workspace() -> None:
    assert DESCRIPTION_NOTE not in _adapter(None).description


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------


async def test_a_replaced_link_is_logged_without_the_link_or_ticket(
    server: _Server, workspace: _Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    await _adapter(workspace).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    assert _link_logs(caplog, logging.INFO) == [
        f"MCP tool set_logo received a download link task_id={TASK_ID} "
        f"file_id={FILE_ID}"
    ]
    assert TICKET not in caplog.text
    assert "external-media" not in caplog.text


async def test_nothing_is_logged_for_a_value_that_is_passed_through(
    server: _Server, workspace: _Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    await _adapter(workspace).run_json_async({ARGUMENT: "https://example.com/a.png"})

    assert _link_logs(caplog, logging.INFO) == []
    assert _link_logs(caplog, logging.WARNING) == []


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        pytest.param(FileNotFoundError("x"), "not_found", id="not-found"),
        pytest.param(
            ExternalMediaNotIssuableError("x"), "not_issuable", id="not-issuable"
        ),
        pytest.param(ValueError("x"), "preparation_failed", id="preparation-failed"),
    ],
)
async def test_every_refusal_is_logged_once_with_its_category(
    server: _Server,
    caplog: pytest.LogCaptureFixture,
    outcome: Exception,
    reason: str,
) -> None:
    caplog.set_level(logging.WARNING)

    await _adapter(_Workspace(outcome)).run_json_async({ARGUMENT: f"file:{FILE_ID}"})

    [line] = _link_logs(caplog, logging.WARNING)
    assert line.startswith(f"MCP tool set_logo refused a download link reason={reason}")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _served_tool(name: str, properties: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} tool",
        inputSchema={"type": "object", "properties": properties},
    )


def _serve(monkeypatch: pytest.MonkeyPatch, tools: list[SimpleNamespace]) -> None:
    class _Session:
        async def initialize(self) -> None:
            return None

    @asynccontextmanager
    async def create_session(connection: Any):
        yield _Session()

    async def load_mcp_tools(_session: Any) -> list[SimpleNamespace]:
        return tools

    monkeypatch.setattr(mcp_adapter_module, "create_session", create_session)
    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools", load_mcp_tools)


async def test_the_loader_gives_each_tool_its_own_link_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve(
        monkeypatch,
        [
            _served_tool(TOOL, {ARGUMENT: {"type": "string"}}),
            _served_tool("other", {ARGUMENT: {"type": "string"}}),
        ],
    )
    settings = [{"tool_name": TOOL, "link_argument": ARGUMENT}]

    result = await load_mcp_tools_as_agent_tools(
        {
            "A": _connection(signed_uploads=settings),
            "B": _connection(),
        },
        sandbox=None,
    )

    adapters = {tool.name: tool._target for tool in result.tools}
    assert adapters["mcp_A_set_logo"]._signed_upload_link == LINK
    assert adapters["mcp_A_other"]._signed_upload_link is None
    assert adapters["mcp_B_set_logo"]._signed_upload_link is None
    assert result.failures == ()


async def test_the_loader_loads_a_tool_whose_link_argument_is_not_a_string(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    _serve(monkeypatch, [_served_tool(TOOL, {ARGUMENT: {"type": "integer"}})])

    result = await load_mcp_tools_as_agent_tools(
        {
            "A": _connection(
                signed_uploads=[{"tool_name": TOOL, "link_argument": ARGUMENT}]
            )
        },
        sandbox=None,
    )

    assert [tool.name for tool in result.tools] == ["mcp_A_set_logo"]
    assert result.tools[0]._target._signed_upload_link is None
    assert result.failures == ()
    assert "ignores its link settings" in caplog.text


def test_the_builder_passes_the_link_entry_to_the_adapter() -> None:
    adapter = _build_mcp_tool_adapter(
        "A",
        _connection(),
        MCPTool(name=TOOL, description="d", inputSchema=_SCHEMA),
        workspace=_Workspace(),
        signed_upload_link=LINK,
    )

    assert adapter._signed_upload_link == LINK

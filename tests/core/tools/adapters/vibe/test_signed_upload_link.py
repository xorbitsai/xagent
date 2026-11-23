"""A link entry of a connector's signed-upload settings, at the tool adapter.

A tool with a link entry takes a workspace image reference in one string
argument and calls the connector with a short-lived download link instead.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.signed_upload import (
    SignedUploadBinding,
    SignedUploadLink,
)
from xagent.core.workspace import TaskWorkspace

HOST = "mcp.example.test"
CONNECTION_URL = f"https://{HOST}/mcp"
SAME_HOST_URL = f"https://{HOST}/other-path"
OTHER_CONNECTION_URL = "https://other.example.test/mcp"
TOOL = "set_logo"
ARGUMENT = "image_url"
FILE_ID = "6f1c2d3e-4b5a-4c6d-8e7f-9a0b1c2d3e4f"
REF = f"file:{FILE_ID}"
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

_REFRESH_KEYS = {
    "runtime": mcp_adapter_module._RUNTIME_CONNECTION_REFRESH_KEY,
    "resolver": mcp_adapter_module._OAUTH_TOKEN_RESOLVER_REFRESH_KEY,
}


def _schema(**argument: Any) -> dict[str, Any]:
    """The tool's input schema; ``absent=True`` leaves the link argument out."""
    properties: dict[str, Any] = {"display_logo_in_dashboard": {"type": "boolean"}}
    if not argument.pop("absent", False):
        properties[ARGUMENT] = {"type": "string", **argument}
    return {"type": "object", "properties": properties}


def _schema_of(argument: dict[str, Any]) -> dict[str, Any]:
    """The tool's input schema with ``argument`` as the link argument's schema."""
    schema = _schema(absent=True)
    schema["properties"][ARGUMENT] = argument
    return schema


def _connection(url: str = CONNECTION_URL, **extra: Any) -> dict[str, Any]:
    return {"transport": "streamable_http", "url": url, **extra}


_RUNTIME_BOUND = _connection(
    runtime_bindings=[
        {
            "source": {"input_type": "context", "key": "logo"},
            "target": {"target_type": "tool_arguments", "key": ARGUMENT},
        }
    ],
    connector_runtime={"context": {"logo": REF}, "secrets": {}, "auth_selector": {}},
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
    """A fake connector: records the arguments of every tool call it receives."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[dict[str, Any]] = []
        self.connections: list[Any] = []
        self.respond: Any = lambda args: CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structuredContent={"image_url": args.get(ARGUMENT)},
            isError=False,
        )
        server = self

        class _Session:
            async def initialize(self) -> None:
                return None

            async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
                server.calls.append(json.loads(json.dumps(arguments)))
                return server.respond(arguments)

        @asynccontextmanager
        async def _create_session(connection: Any):
            server.connections.append(connection)
            yield _Session()

        monkeypatch.setattr(mcp_adapter_module, "create_session", _create_session)

    def fail_first_call_with_401(self, path: str) -> None:
        """Answer the first call with a 401 that the given retry path acts on."""
        then = self.respond
        challenge = ['Bearer error="invalid_token"'] if path == "resolver" else []

        def respond(args: dict[str, Any]) -> Any:
            if len(self.calls) == 1:
                request = httpx.Request("POST", CONNECTION_URL)
                headers = [(b"www-authenticate", v.encode()) for v in challenge]
                response = httpx.Response(401, headers=headers, request=request)
                raise httpx.HTTPStatusError(
                    "unauthorized", request=request, response=response
                )
            return then(args)

        self.respond = respond


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
    schema: dict[str, Any] | None = None,
    connection: Any = None,
) -> MCPToolAdapter:
    return MCPToolAdapter(
        mcp_tool=MCPTool(
            name=TOOL,
            description="Set the logo.",
            inputSchema=_schema() if schema is None else schema,
        ),
        connection=_connection() if connection is None else connection,
        workspace=workspace,
        signed_upload_link=link,
    )


def _retrying_adapter(workspace: Any, path: str, refreshed_url: str) -> MCPToolAdapter:
    """An adapter whose connection refreshes to ``refreshed_url`` after a 401."""

    def refresh(*args: Any) -> dict[str, Any]:
        return _connection(refreshed_url)

    return _adapter(workspace, connection=_connection(**{_REFRESH_KEYS[path]: refresh}))


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


def _link_logs(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == level and "download link" in record.getMessage()
    ]


# --------------------------------------------------------------------------
# Replacement and pass-through
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(REF, id="file-reference"),
        pytest.param(FILE_ID, id="bare-uuid"),
        pytest.param(f"  {REF}  ", id="padded-with-whitespace"),
        pytest.param(f"file://{FILE_ID}", id="legacy-double-slash-form"),
    ],
)
async def test_a_file_reference_is_replaced_by_a_link_logged_without_it(
    server: _Server,
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
    value: str,
) -> None:
    caplog.set_level(logging.DEBUG)

    result = await _adapter(workspace).run_json_async(
        {ARGUMENT: value, "display_logo_in_dashboard": True}
    )

    assert result["is_error"] is False
    assert workspace.issued == [(FILE_ID, "connector-image")]
    assert server.calls == [{ARGUMENT: LINK_URL, "display_logo_in_dashboard": True}]
    assert _link_logs(caplog, logging.INFO) == [
        f"MCP tool set_logo received a download link task_id={TASK_ID} "
        f"file_id={FILE_ID}"
    ]
    assert TICKET not in caplog.text
    assert "external-media" not in caplog.text


@pytest.mark.parametrize(
    ("args", "setup"),
    [
        pytest.param({ARGUMENT: "https://example.com/a.png"}, {}, id="https-url"),
        pytest.param({ARGUMENT: "http://example.com/a.png"}, {}, id="http-url"),
        pytest.param({ARGUMENT: ""}, {}, id="empty"),
        pytest.param({ARGUMENT: "not a ref"}, {}, id="free-text"),
        pytest.param({ARGUMENT: "file:abc?x=1"}, {}, id="query-on-a-non-id"),
        pytest.param({ARGUMENT: f"{REF}?x=1"}, {}, id="query-on-an-id"),
        pytest.param({ARGUMENT: f"{REF}/sub"}, {}, id="path-on-an-id"),
        pytest.param({ARGUMENT: "file:///etc/passwd"}, {}, id="real-file-uri"),
        pytest.param({"display_logo_in_dashboard": True}, {}, id="argument-absent"),
        pytest.param({ARGUMENT: REF}, {"link": None}, id="tool-without-link-entry"),
        pytest.param({ARGUMENT: FILE_ID}, {"workspace": None}, id="no-workspace-uuid"),
        pytest.param(
            {ARGUMENT: "https://example.com/a.png"},
            {"connection": _connection(OTHER_CONNECTION_URL)},
            id="plain-url-to-any-host",
        ),
        # A runtime value is the host page's, never turned into a link.
        pytest.param(
            {ARGUMENT: REF}, {"connection": _RUNTIME_BOUND}, id="runtime-bound"
        ),
    ],
)
async def test_a_value_that_is_not_a_workspace_image_is_passed_through(
    server: _Server,
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
    args: dict[str, Any],
    setup: dict[str, Any],
) -> None:
    caplog.set_level(logging.DEBUG)
    options = {"workspace": workspace, **setup}
    adapter_workspace = options.pop("workspace")

    result = await _adapter(adapter_workspace, **options).run_json_async(args)

    assert result["is_error"] is False
    if adapter_workspace is not None:
        # Without a workspace there is nothing that could issue a link.
        assert workspace.issued == []
    assert server.calls == [args]
    assert _link_logs(caplog, logging.INFO) == []
    assert _link_logs(caplog, logging.WARNING) == []


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

    result = await _adapter(workspace).run_json_async({ARGUMENT: REF})

    assert [item["text"] for item in result["content"]] == [
        REF,
        f"Downloaded {REF} just now, thanks",
        json.dumps({"image_url": REF}),
    ]
    assert result["structured_content"] == {
        "whole": REF,
        "inside": f"saw {REF} twice {REF}",
    }
    assert TICKET not in json.dumps(result)


# --------------------------------------------------------------------------
# Refusal: the tool is not called
# --------------------------------------------------------------------------


# A storage failure's text names an environment variable and the link; it
# reaches neither the model nor the log.
_SECRET = f"XAGENT_EXTERNAL_MEDIA_URL_TTL_SECONDS is invalid for {LINK_URL}"


@pytest.mark.parametrize(
    ("owner", "error", "message", "expected", "reason"),
    [
        ({}, "ExternalMediaNotFound", "hidden", FILE_NOT_FOUND, "not_found"),
        ({}, "ExternalMediaUnsupportedImage", "Only PNG", "Only PNG", "not_issuable"),
        ({}, "ExternalMediaNotConfigured", "No HTTPS", "No HTTPS", "not_issuable"),
        ({}, "ExternalMediaOutOfScope", "Out", "Out", "not_issuable"),
        (
            {},
            "ExternalMediaUnavailable",
            _SECRET,
            PREPARATION_FAILED,
            "preparation_failed",
        ),
        ({"db_task_id": None}, None, None, FILE_NOT_FOUND, "not_found"),
        ({"owner_user_id": None}, None, None, FILE_NOT_FOUND, "not_found"),
    ],
    ids=[
        "not-found-or-another-task",
        "unsupported-image",
        "no-public-address",
        "out-of-scope",
        "storage-unavailable",
        "no-task",
        "no-owner",
    ],
)
async def test_a_link_that_cannot_be_issued_ends_the_call_with_a_fixed_error(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    owner: dict[str, Any],
    error: str | None,
    message: str | None,
    expected: str,
    reason: str,
) -> None:
    from xagent.core.storage import manager
    from xagent.web.services import external_media

    def issue(*args: Any, **kwargs: Any) -> Any:
        raise getattr(external_media, str(error))(message)

    monkeypatch.setattr(
        manager, "create_db_session", lambda: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(external_media, "issue_external_media_url", issue)
    workspace = TaskWorkspace(
        "web_task_41", str(tmp_path), db_task_id=owner.get("db_task_id", TASK_ID)
    )
    workspace.owner_user_id = owner.get("owner_user_id", 7)
    caplog.set_level(logging.DEBUG)

    result = await _adapter(workspace).run_json_async({ARGUMENT: REF})

    assert result["is_error"] is True
    assert _text(result) == expected
    assert server.calls == []
    [line] = _link_logs(caplog, logging.WARNING)
    assert line.startswith(f"MCP tool set_logo refused a download link reason={reason}")
    assert TICKET not in caplog.text
    assert "XAGENT_" not in caplog.text


@pytest.mark.parametrize(
    ("workspace", "expected", "logs"),
    [
        pytest.param(
            None,
            FILE_UNAVAILABLE,
            ["MCP tool set_logo refused a download link reason=no_workspace"],
            id="no-workspace",
        ),
        pytest.param(
            _Workspace(RuntimeError(f"db down {LINK_URL}")),
            "Error executing MCP tool.",
            [],
            id="database-failure",
        ),
    ],
)
async def test_a_file_reference_that_cannot_be_looked_up_is_refused(
    server: _Server,
    caplog: pytest.LogCaptureFixture,
    workspace: Any,
    expected: str,
    logs: list[str],
) -> None:
    caplog.set_level(logging.WARNING)

    result = await _adapter(workspace).run_json_async({ARGUMENT: REF})

    assert result["is_error"] is True
    assert _text(result) == expected
    assert server.calls == []
    assert _link_logs(caplog, logging.WARNING) == logs


@pytest.mark.parametrize(
    ("constraint", "refused"),
    [
        pytest.param({}, False, id="no-maxlength"),
        pytest.param({"maxLength": len(LINK_URL)}, False, id="exactly-long-enough"),
        pytest.param({"maxLength": len(LINK_URL) - 1}, True, id="one-too-short"),
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

    result = await _adapter(workspace, schema=_schema(**constraint)).run_json_async(
        {ARGUMENT: REF}
    )

    if refused:
        assert _text(result) == TOO_LONG
        assert server.calls == []
        assert _link_logs(caplog, logging.WARNING) == [
            "MCP tool set_logo refused a download link reason=too_long"
        ]
    else:
        assert result["is_error"] is False
        assert server.calls == [{ARGUMENT: LINK_URL}]


# --------------------------------------------------------------------------
# Nothing that names the link leaks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", [None, *sorted(_REFRESH_KEYS)])
async def test_a_call_that_raises_with_the_link_in_its_message_leaks_nothing(
    server: _Server,
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
    path: str | None,
) -> None:
    caplog.set_level(logging.DEBUG)

    def boom(args: dict[str, Any]) -> Any:
        raise RuntimeError(f"fetching {args[ARGUMENT]} failed")

    server.respond = boom
    if path is None:
        adapter = _adapter(workspace)
    else:
        # The retry over the same host reuses the link and raises with it.
        server.fail_first_call_with_401(path)
        adapter = _retrying_adapter(workspace, path, SAME_HOST_URL)

    result = await adapter.run_json_async({ARGUMENT: REF})

    assert result["is_error"] is True
    assert server.calls == [{ARGUMENT: LINK_URL}] * (1 if path is None else 2)
    assert "ticket=" not in json.dumps(result)
    assert "ticket=" not in caplog.text


# --------------------------------------------------------------------------
# Which connection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "connection",
    [
        pytest.param(_connection(OTHER_CONNECTION_URL), id="another-host"),
        pytest.param({"transport": "streamable_http"}, id="no-url"),
        pytest.param({"transport": "streamable_http", "url": 7}, id="url-not-a-str"),
        pytest.param(_connection("https://[bad/"), id="unparsable"),
    ],
)
async def test_a_connection_to_another_host_than_the_settings_is_refused(
    server: _Server,
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    connection: dict[str, Any],
) -> None:
    # The caller's user id is read on every call and warned about when absent;
    # set it so the full warning list below holds only the refusal.
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    caplog.set_level(logging.WARNING)
    adapter = _adapter(workspace, connection=connection)
    caplog.clear()

    result = await adapter.run_json_async({ARGUMENT: REF})

    assert result["is_error"] is True
    assert _text(result) == UNTRUSTED_CONNECTOR
    assert server.calls == []
    assert workspace.issued == []
    # Every warning, not only the ones that name the link: the refusal is
    # logged once, and never as a signed upload.
    assert [r.getMessage() for r in caplog.records] == [
        "MCP tool set_logo refused a download link reason=untrusted_connector"
    ]


@pytest.mark.parametrize("path", sorted(_REFRESH_KEYS))
@pytest.mark.parametrize(
    ("value", "refreshed_url", "refused"),
    [
        pytest.param(REF, OTHER_CONNECTION_URL, True, id="link-to-another-host"),
        pytest.param(REF, SAME_HOST_URL, False, id="link-to-the-same-host"),
        pytest.param(
            "https://example.com/a.png", OTHER_CONNECTION_URL, False, id="no-link"
        ),
    ],
)
async def test_a_retry_carries_a_link_only_to_the_settings_host(
    server: _Server,
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: str,
    refreshed_url: str,
    refused: bool,
) -> None:
    server.fail_first_call_with_401(path)
    # The caller's user id is read on every call and warned about when absent;
    # set it so the full warning list below holds only the refusal.
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    caplog.set_level(logging.WARNING)
    adapter = _retrying_adapter(workspace, path, refreshed_url)
    caplog.clear()

    result = await adapter.run_json_async({ARGUMENT: value})

    if refused:
        assert result == {"content": [{"text": UNTRUSTED_CONNECTOR}], "is_error": True}
        assert [r.getMessage() for r in caplog.records] == [
            "MCP tool set_logo refused a download link reason=untrusted_connector"
        ]
        assert len(server.calls) == 1
        assert [c.get("url") for c in server.connections] == [CONNECTION_URL]
        return
    # The retry reuses the one link issued for the call.
    assert result["is_error"] is False
    sent = LINK_URL if value == REF else value
    assert server.calls == [{ARGUMENT: sent}, {ARGUMENT: sent}]
    assert len(workspace.issued) == (1 if value == REF else 0)
    assert result["structured_content"]["image_url"] == value


def test_a_signed_upload_to_another_host_keeps_its_own_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Why: the link path shares the host check with signed uploads but logs
    # its own refusal; the signed-upload warning must stay as it was.
    binding = SignedUploadBinding(
        connection_host=HOST,
        tool_name=TOOL,
        file_argument="document",
        issuing_tool_name="request_upload",
        allowed_upload_hosts=frozenset({"uploads.example.test"}),
        upload_path_prefix="/staged/",
        content_types={".pdf": "application/pdf"},
    )
    caplog.set_level(logging.WARNING)

    with pytest.raises(mcp_adapter_module._SignedUploadRefused) as refused:
        mcp_adapter_module._check_signed_upload_connection(
            _connection(OTHER_CONNECTION_URL), binding, TOOL
        )

    assert refused.value.result == {
        "content": [{"text": UNTRUSTED_CONNECTOR}],
        "is_error": True,
    }
    assert [r.getMessage() for r in caplog.records] == [
        "MCP tool set_logo refused a signed upload for an untrusted connector"
    ]


async def test_a_refused_link_ends_the_call_before_any_signed_upload(
    server: _Server,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Why: the upload sends bytes to an outside host, so a link that will be
    # refused must be found first.
    other_argument = "document"
    binding = SignedUploadBinding(
        connection_host=HOST,
        tool_name=TOOL,
        file_argument=other_argument,
        issuing_tool_name="request_upload",
        allowed_upload_hosts=frozenset({"uploads.example.test"}),
        upload_path_prefix="/staged/",
        content_types={".pdf": "application/pdf"},
    )
    staged = tmp_path / "report.pdf"
    staged.write_bytes(b"%PDF-1.4")
    puts: list[httpx.Request] = []

    def put(request: httpx.Request) -> httpx.Response:
        puts.append(request)
        return httpx.Response(200, request=request)

    async def stage(file_id: str) -> Path:
        return staged

    ticket = {
        "file_id": "rf_abcdefgh",
        "upload_url": "https://uploads.example.test/staged/abc?sig=s",
        "method": "PUT",
        "headers": {"Content-Type": "application/pdf"},
        "url_expires_at": "2030-01-01T00:00:00+00:00",
    }
    server.respond = lambda args: CallToolResult(
        content=[TextContent(type="text", text=json.dumps(ticket))],
        structuredContent=ticket,
        isError=False,
    )
    monkeypatch.setattr(
        mcp_adapter_module,
        "_signed_upload_transport",
        lambda: httpx.MockTransport(put),
    )
    monkeypatch.setattr(
        mcp_adapter_module,
        "_utcnow",
        lambda: datetime(2029, 12, 31, tzinfo=timezone.utc),
    )
    schema = _schema()
    schema["properties"][other_argument] = {"type": "string"}
    adapter = _adapter(_Workspace(FileNotFoundError("gone")), schema=schema)
    adapter._signed_upload = binding
    monkeypatch.setattr(adapter, "_stage_external_upload_file", stage)

    result = await adapter.run_json_async(
        {ARGUMENT: REF, other_argument: f"file:{FILE_ID}"}
    )

    assert result["is_error"] is True
    assert _text(result) == FILE_NOT_FOUND
    assert server.calls == []
    assert puts == []


# --------------------------------------------------------------------------
# Which arguments take part
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema", "connection"),
    [
        pytest.param(_schema(absent=True), None, id="parameter-missing"),
        pytest.param(_schema(type="integer"), None, id="integer"),
        pytest.param(_schema(type="array"), None, id="array"),
        pytest.param(
            _schema_of({"type": ["string", "integer"]}), None, id="string-or-integer"
        ),
        pytest.param(
            _schema_of({"type": ["string", "null", "integer"]}),
            None,
            id="three-types",
        ),
        pytest.param(
            _schema_of({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
            None,
            id="optional-integer",
        ),
        pytest.param(
            _schema_of({"anyOf": [{"type": "string"}, {"type": "integer"}]}),
            None,
            id="string-or-integer-anyof",
        ),
        pytest.param(
            _schema_of(
                {"anyOf": [{"type": "string"}, {"type": "null"}, {"type": "integer"}]}
            ),
            None,
            id="anyof-with-three-branches",
        ),
        pytest.param(
            _schema_of({"oneOf": [{"type": "string"}, {"type": "null"}]}),
            None,
            id="oneof",
        ),
        pytest.param(
            _schema_of(
                {"type": "integer", "anyOf": [{"type": "string"}, {"type": "null"}]}
            ),
            None,
            id="type-beside-anyof",
        ),
        pytest.param(_schema(), _RUNTIME_BOUND, id="runtime-bound"),
    ],
)
def test_a_link_entry_the_model_cannot_fill_is_ignored(
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
    schema: dict[str, Any],
    connection: Any,
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


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param({"type": "string"}, id="string"),
        pytest.param({"type": ["string", "null"]}, id="string-or-null"),
        pytest.param({"type": ["null", "string"]}, id="null-or-string"),
        pytest.param(
            {"anyOf": [{"type": "string"}, {"type": "null"}]}, id="optional-string"
        ),
        pytest.param(
            {"anyOf": [{"type": "null"}, {"type": "string"}]}, id="null-branch-first"
        ),
    ],
)
async def test_a_link_argument_that_takes_a_string_or_null_is_linked(
    server: _Server, workspace: _Workspace, argument: dict[str, Any]
) -> None:
    # Why: Pydantic and FastMCP describe an optional string as an anyOf, so a
    # link entry on such a tool must still take effect.
    adapter = _adapter(workspace, schema=_schema_of(argument))

    assert adapter._signed_upload_link == LINK
    assert adapter.description.endswith(DESCRIPTION_NOTE)
    result = await adapter.run_json_async({ARGUMENT: REF})

    assert result["is_error"] is False
    assert workspace.issued == [(FILE_ID, "connector-image")]
    assert server.calls == [{ARGUMENT: LINK_URL}]


async def test_a_null_for_an_optional_link_argument_is_left_out(
    server: _Server, workspace: _Workspace
) -> None:
    schema = _schema_of({"anyOf": [{"type": "string"}, {"type": "null"}]})

    result = await _adapter(workspace, schema=schema).run_json_async({ARGUMENT: None})

    assert result["is_error"] is False
    assert workspace.issued == []
    assert server.calls == [{}]


@pytest.mark.parametrize(
    ("max_length", "refused"),
    [
        pytest.param(len(LINK_URL), False, id="exactly-long-enough"),
        pytest.param(len(LINK_URL) - 1, True, id="one-too-short"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            lambda limit: {"type": ["string", "null"], "maxLength": limit},
            id="string-or-null",
        ),
        pytest.param(
            lambda limit: {
                "anyOf": [{"type": "string", "maxLength": limit}, {"type": "null"}]
            },
            id="optional-string",
        ),
    ],
)
async def test_the_link_must_fit_the_maxlength_of_an_optional_string(
    server: _Server,
    workspace: _Workspace,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    shape: Any,
    max_length: int,
    refused: bool,
) -> None:
    # The caller's user id is read on every call and warned about when absent;
    # set it so the full warning list below holds only the refusal.
    monkeypatch.setenv("XAGENT_USER_ID", "7")
    caplog.set_level(logging.WARNING)
    adapter = _adapter(workspace, schema=_schema_of(shape(max_length)))
    caplog.clear()

    result = await adapter.run_json_async({ARGUMENT: REF})

    if refused:
        assert _text(result) == TOO_LONG
        assert server.calls == []
        assert [r.getMessage() for r in caplog.records] == [
            "MCP tool set_logo refused a download link reason=too_long"
        ]
    else:
        assert result["is_error"] is False
        assert server.calls == [{ARGUMENT: LINK_URL}]


@pytest.mark.parametrize(
    ("link", "has_workspace", "told"),
    [(LINK, True, True), (LINK, False, False), (None, True, False)],
)
def test_the_description_tells_the_model_how_to_pass_an_image(
    link: SignedUploadLink | None, has_workspace: bool, told: bool
) -> None:
    adapter = _adapter(_Workspace() if has_workspace else None, link=link)

    assert adapter.description.startswith("Set the logo.")
    assert adapter.description.endswith(DESCRIPTION_NOTE) is told


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


async def test_the_loader_gives_each_tool_its_own_link_entry(
    server: _Server, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    tools = [
        SimpleNamespace(
            name=name,
            description=name,
            inputSchema={"type": "object", "properties": {ARGUMENT: {"type": kind}}},
        )
        for name, kind in [(TOOL, "string"), ("other", "string"), ("n", "integer")]
    ]

    async def load_mcp_tools(_session: Any) -> list[SimpleNamespace]:
        return tools

    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools", load_mcp_tools)
    settings = [
        {"tool_name": TOOL, "link_argument": ARGUMENT},
        {"tool_name": "n", "link_argument": ARGUMENT},
    ]

    result = await load_mcp_tools_as_agent_tools(
        {"A": _connection(signed_uploads=settings), "B": _connection()},
        sandbox=None,
    )

    # A link entry the model cannot fill is ignored and its tool still loads.
    links = {tool.name: tool._target._signed_upload_link for tool in result.tools}
    assert links == {
        "mcp_A_set_logo": LINK,
        "mcp_A_other": None,
        "mcp_A_n": None,
        "mcp_B_set_logo": None,
        "mcp_B_other": None,
        "mcp_B_n": None,
    }
    assert result.failures == ()
    assert "MCP tool n ignores its link settings" in caplog.text

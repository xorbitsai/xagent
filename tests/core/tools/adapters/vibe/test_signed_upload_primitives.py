from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

from xagent.core.agent.result import ClassifiedToolFailure
from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPToolAdapter,
    _replace_exact_string_values,
)
from xagent.core.tools.adapters.vibe.signed_upload import (
    SignedUploadBinding,
    SignedUploadResponseError,
    SignedUploadURLRejected,
    check_signed_upload_url,
    parse_signed_upload_response,
)

# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

SIZE_BYTES = 1234
REMOTE_FILE_ID = "rf_abcdefgh"
UPLOAD_URL = "https://uploads.example.test/staged/abc?sig=secret-signature"
EXPIRES_AT = "2030-01-01T00:00:00+00:00"


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "file_id": REMOTE_FILE_ID,
        "upload_url": UPLOAD_URL,
        "method": "PUT",
        "headers": {"Content-Type": "application/pdf"},
        "url_expires_at": EXPIRES_AT,
    }
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not _MISSING}


_MISSING = object()


def _structured(payload: Any) -> dict[str, Any]:
    return {"content": [], "structured_content": payload, "is_error": False}


def _text(payload: Any, **extra: Any) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "content": [{"type": "text", "text": text}],
        "structured_content": None,
        "is_error": False,
        **extra,
    }


def test_parse_signed_upload_response_accepts_structured_content() -> None:
    ticket = parse_signed_upload_response(
        _structured(_payload()), size_bytes=SIZE_BYTES
    )

    assert ticket.file_id == REMOTE_FILE_ID
    assert ticket.upload_url == UPLOAD_URL
    assert dict(ticket.headers) == {"Content-Type": "application/pdf"}
    assert ticket.expires_at.utcoffset() is not None


def test_parse_signed_upload_response_falls_back_to_first_text_item() -> None:
    ticket = parse_signed_upload_response(_text(_payload()), size_bytes=SIZE_BYTES)

    assert ticket.file_id == REMOTE_FILE_ID


def test_parse_signed_upload_response_ignores_non_dict_structured_content() -> None:
    result = _text(_payload())
    result["structured_content"] = "not a dict"

    ticket = parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    assert ticket.file_id == REMOTE_FILE_ID


def test_parse_signed_upload_response_skips_non_text_items_before_the_text() -> None:
    result = _text(_payload())
    result["content"].insert(0, {"type": "image", "data": "aGk=", "mimeType": "x/y"})

    ticket = parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    assert ticket.file_id == REMOTE_FILE_ID


def test_parse_signed_upload_response_prefers_structured_content_over_text() -> None:
    result = _text(_payload(file_id="rf_from_text"))
    result["structured_content"] = _payload(file_id="rf_from_struct")

    ticket = parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    assert ticket.file_id == "rf_from_struct"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"method": "put"}, id="method-lower-case"),
        pytest.param({"url_expires_at": "2030-01-01T00:00:00+00:00"}, id="offset"),
        pytest.param({"url_expires_at": "2030-01-01T00:00:00Z"}, id="zulu"),
        pytest.param(
            {
                "headers": {
                    "Content-Type": "application/pdf",
                    "Content-Length": str(SIZE_BYTES),
                    "x-checksum-sha256": "abc",
                }
            },
            id="allowed-headers",
        ),
        pytest.param(
            {"headers": {"X-Meta.v_1": "a b\tc;q=0.8, d", "x-empty": ""}},
            id="token-name-and-spaced-value",
        ),
    ],
)
def test_parse_signed_upload_response_accepts_variants(
    overrides: dict[str, Any],
) -> None:
    ticket = parse_signed_upload_response(
        _structured(_payload(**overrides)), size_bytes=SIZE_BYTES
    )

    assert ticket.file_id == REMOTE_FILE_ID
    assert ticket.expires_at.year == 2030
    assert ticket.expires_at.utcoffset() is not None
    if "headers" in overrides:
        assert dict(ticket.headers) == overrides["headers"]


def test_parse_signed_upload_response_headers_are_read_only() -> None:
    ticket = parse_signed_upload_response(
        _structured(_payload()), size_bytes=SIZE_BYTES
    )

    with pytest.raises(TypeError):
        ticket.headers["X-Extra"] = "1"  # type: ignore[index]


def _assert_error_hides_payload(exc: SignedUploadResponseError, *, reason: str) -> None:
    assert exc.reason == reason
    rendered = str(exc) + repr(exc)
    for needle in (
        REMOTE_FILE_ID,
        "secret-signature",
        "uploads.example.test",
        "application/pdf",
        "abcdefgh",
    ):
        assert needle not in rendered


_MALFORMED_CASES = [
    pytest.param(_structured(_payload(file_id=_MISSING)), id="file-id-missing"),
    pytest.param(_structured(_payload(file_id="abc")), id="file-id-too-short"),
    pytest.param(_structured(_payload(file_id="rf abcdefgh")), id="file-id-with-space"),
    pytest.param(_structured(_payload(file_id="a" * 129)), id="file-id-too-long"),
    pytest.param(
        _structured(_payload(file_id="rf_abcdefgh\n")), id="file-id-trailing-newline"
    ),
    pytest.param(_structured(_payload(file_id=12345678)), id="file-id-not-str"),
    pytest.param(_structured(_payload(upload_url=42)), id="upload-url-not-str"),
    pytest.param(_structured(_payload(upload_url=_MISSING)), id="upload-url-missing"),
    pytest.param(_structured(_payload(method="POST")), id="method-post"),
    pytest.param(_structured(_payload(method=_MISSING)), id="method-missing"),
    pytest.param(_structured(_payload(headers=["a: b"])), id="headers-not-dict"),
    pytest.param(_structured(_payload(headers=_MISSING)), id="headers-missing"),
    pytest.param(
        _structured(_payload(headers={"Content-Type": 5})), id="header-value-not-str"
    ),
    pytest.param(
        _structured(_payload(url_expires_at="2030-01-01T00:00:00")),
        id="expires-without-timezone",
    ),
    pytest.param(
        _structured(_payload(url_expires_at="tomorrow")), id="expires-not-iso"
    ),
    pytest.param(_structured(_payload(url_expires_at=_MISSING)), id="expires-missing"),
    pytest.param(
        _structured(_payload(url_expires_at=1893456000)), id="expires-not-str"
    ),
    pytest.param(_text("this is not json {"), id="text-not-json"),
    pytest.param(_text(json.dumps(["file_id", REMOTE_FILE_ID])), id="json-not-dict"),
    pytest.param(
        {"content": [], "structured_content": None, "is_error": False},
        id="no-payload-at-all",
    ),
    pytest.param(
        {
            "content": [{"type": "text", "text": 5}],
            "structured_content": None,
            "is_error": False,
        },
        id="text-item-without-text",
    ),
    pytest.param(
        _structured(
            _payload(headers={"Content-Length": str(SIZE_BYTES), "content-length": "1"})
        ),
        id="duplicate-content-length-different-case",
    ),
    pytest.param(
        _structured(_payload(headers={"X-Thing": "a", "x-thing": "a"})),
        id="duplicate-header-same-value",
    ),
    pytest.param(
        _structured(_payload(headers={"X-Thing": "caf\u00e9"})),
        id="header-value-non-ascii",
    ),
    pytest.param(
        _structured(_payload(headers={"X-Thing": "a\r\nX-Injected: 1"})),
        id="header-value-crlf",
    ),
    pytest.param(
        _structured(_payload(headers={"X-Thing": "a\nb"})), id="header-value-newline"
    ),
    pytest.param(
        _structured(_payload(headers={"X-Thing": "a\x00b"})), id="header-value-nul"
    ),
    pytest.param(
        _structured(_payload(headers={"X-Thing": "a\x7fb"})), id="header-value-delete"
    ),
    pytest.param(
        _structured(_payload(headers={"X Thing": "a"})), id="header-name-space"
    ),
    pytest.param(
        _structured(_payload(headers={"X:Thing": "a"})), id="header-name-colon"
    ),
    pytest.param(_structured(_payload(headers={"": "a"})), id="header-name-empty"),
    pytest.param(
        _structured(_payload(headers={"X-Th\u00efng": "a"})), id="header-name-non-ascii"
    ),
    pytest.param(
        _structured(_payload(headers={"X-Thing\n": "a"})),
        id="header-name-trailing-newline",
    ),
]


@pytest.mark.parametrize("result", _MALFORMED_CASES)
def test_parse_signed_upload_response_rejects_malformed(
    result: dict[str, Any],
) -> None:
    with pytest.raises(SignedUploadResponseError) as exc_info:
        parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    _assert_error_hides_payload(exc_info.value, reason="malformed")


def test_parse_signed_upload_response_uses_only_the_first_text_item() -> None:
    result = _text("not json")
    result["content"].append({"type": "text", "text": json.dumps(_payload())})

    with pytest.raises(SignedUploadResponseError) as exc_info:
        parse_signed_upload_response(result, size_bytes=SIZE_BYTES)

    assert exc_info.value.reason == "malformed"


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"Authorization": "Bearer planted"}, id="authorization"),
        pytest.param({"authorization": "Bearer planted"}, id="authorization-lower"),
        pytest.param({"Cookie": "a=b"}, id="cookie"),
        pytest.param({"Host": "evil.example.test"}, id="host"),
        pytest.param({"Proxy-Authorization": "Basic x"}, id="proxy-authorization"),
        pytest.param({"Proxy-Foo": "x"}, id="proxy-prefix"),
        pytest.param({"Transfer-Encoding": "chunked"}, id="transfer-encoding"),
        pytest.param({"Connection": "close"}, id="connection"),
        pytest.param({"Content-Length": "999"}, id="content-length-mismatch"),
        pytest.param({"content-length": "1234 "}, id="content-length-not-exact"),
    ],
)
def test_parse_signed_upload_response_rejects_forbidden_headers(
    headers: dict[str, str],
) -> None:
    with pytest.raises(SignedUploadResponseError) as exc_info:
        parse_signed_upload_response(
            _structured(_payload(headers=headers)), size_bytes=SIZE_BYTES
        )

    _assert_error_hides_payload(exc_info.value, reason="forbidden_header")
    for value in headers.values():
        assert value not in str(exc_info.value) + repr(exc_info.value)


# --------------------------------------------------------------------------
# URL policy
# --------------------------------------------------------------------------

BINDING = SignedUploadBinding(
    connection_host="mcp.example.test",
    tool_name="attach_document",
    file_argument="document",
    issuing_tool_name="request_upload",
    allowed_upload_hosts=frozenset({"uploads.example.test"}),
    upload_path_prefix="/staged/",
    content_types={".pdf": "application/pdf"},
)

_UPLOADS = "uploads.example.test"


@pytest.mark.parametrize(
    ("raw", "expected_host"),
    [
        pytest.param(f"http://{_UPLOADS}/staged/x", _UPLOADS, id="plain-http"),
        pytest.param(
            "https://other.example.test/staged/x",
            "other.example.test",
            id="host-not-allowed",
        ),
        pytest.param(
            f"https://evil.{_UPLOADS}/staged/x",
            f"evil.{_UPLOADS}",
            id="subdomain-of-allowed-host",
        ),
        pytest.param(
            f"https://{_UPLOADS}.evil/staged/x",
            f"{_UPLOADS}.evil",
            id="allowed-host-as-prefix",
        ),
        pytest.param(
            f"https://{_UPLOADS}./staged/x", f"{_UPLOADS}.", id="trailing-dot"
        ),
        pytest.param(f"https://user:pw@{_UPLOADS}/staged/x", _UPLOADS, id="userinfo"),
        pytest.param(f"https://{_UPLOADS}:8443/staged/x", _UPLOADS, id="port-8443"),
        pytest.param(f"https://{_UPLOADS}/other/x", _UPLOADS, id="prefix-mismatch"),
        pytest.param(f"https://{_UPLOADS}/staged", _UPLOADS, id="prefix-without-slash"),
        pytest.param(f"https://{_UPLOADS}/staged/../x", _UPLOADS, id="dot-dot"),
        pytest.param(
            f"https://{_UPLOADS}/staged/%2e%2e/x", _UPLOADS, id="encoded-dot-dot"
        ),
        pytest.param(
            f"https://{_UPLOADS}/staged/%2E%2e/x", _UPLOADS, id="mixed-case-dot-dot"
        ),
        pytest.param(
            f"https://{_UPLOADS}/staged/a%5cb", _UPLOADS, id="encoded-backslash"
        ),
        pytest.param(f"https://{_UPLOADS}/staged/a%2fb", _UPLOADS, id="encoded-slash"),
        pytest.param(f"https://{_UPLOADS}/staged/a\\b", None, id="backslash"),
        pytest.param(f"https://{_UPLOADS}/staged/a\r\nb", None, id="crlf"),
        pytest.param(f"https://{_UPLOADS}/staged/a\x00b", None, id="nul"),
        pytest.param(f"https://{_UPLOADS}/staged/a\x7fb", None, id="del"),
        pytest.param(
            "https://xn--ls8h.example.test/staged/x", None, id="invalid-idna-host"
        ),
    ],
)
def test_signed_upload_url_policy_rejects(raw: str, expected_host: str | None) -> None:
    with pytest.raises(SignedUploadURLRejected) as exc_info:
        check_signed_upload_url(raw, BINDING)

    assert exc_info.value.host == expected_host
    assert "example.test" not in str(exc_info.value) + repr(exc_info.value)


@pytest.mark.parametrize(
    ("raw", "raw_path"),
    [
        pytest.param(
            f"https://{_UPLOADS.upper()}/staged/x", b"/staged/x", id="upper-case-host"
        ),
        pytest.param(
            f"https://{_UPLOADS}:443/staged/x?sig=1",
            b"/staged/x?sig=1",
            id="default-port-with-query",
        ),
    ],
)
def test_signed_upload_url_policy_accepts(raw: str, raw_path: bytes) -> None:
    url = check_signed_upload_url(raw, BINDING)

    assert isinstance(url, httpx.URL)
    assert url.host == _UPLOADS
    assert url.raw_path == raw_path
    assert url.scheme == "https"


def test_signed_upload_url_policy_ignores_query_when_checking_the_path() -> None:
    with pytest.raises(SignedUploadURLRejected):
        check_signed_upload_url(f"https://{_UPLOADS}/other/x?next=/staged/", BINDING)


# --------------------------------------------------------------------------
# Whole-value replacement
# --------------------------------------------------------------------------


def test_replace_exact_string_values() -> None:
    replacements = {REMOTE_FILE_ID: "file:own-id"}

    assert _replace_exact_string_values(REMOTE_FILE_ID, replacements) == "file:own-id"
    assert (
        _replace_exact_string_values(f"see {REMOTE_FILE_ID} for details", replacements)
        == f"see {REMOTE_FILE_ID} for details"
    )
    nested = {
        "a": [REMOTE_FILE_ID, {"b": (REMOTE_FILE_ID, "keep")}],
        "c": {"d": REMOTE_FILE_ID},
        "n": 7,
        "none": None,
    }
    assert _replace_exact_string_values(nested, replacements) == {
        "a": ["file:own-id", {"b": ("file:own-id", "keep")}],
        "c": {"d": "file:own-id"},
        "n": 7,
        "none": None,
    }
    assert nested["a"][0] == REMOTE_FILE_ID


def test_replace_exact_string_values_does_not_touch_mapping_keys() -> None:
    value = {REMOTE_FILE_ID: REMOTE_FILE_ID}

    assert _replace_exact_string_values(value, {REMOTE_FILE_ID: "file:own-id"}) == {
        REMOTE_FILE_ID: "file:own-id"
    }


def test_replace_exact_string_values_without_replacements_returns_same_object() -> None:
    value = {"a": [REMOTE_FILE_ID]}

    assert _replace_exact_string_values(value, {}) is value


# --------------------------------------------------------------------------
# Issuing calls: same authorization policy as the call the model asked for
# --------------------------------------------------------------------------

CONNECTOR_URL = "https://mcp.example.test/mcp"
OUTER_TOOL = "attach_document"
ISSUING_TOOL = "request_upload"
INITIAL_AUTH = "Bearer initial-token-secret"
FRESH_AUTH = "Bearer fresh-token-secret"
LEAKY_MESSAGE = "secret https://x.test/?sig=abc"
INVALID_TOKEN = 'Bearer error="invalid_token"'

_RUNTIME_KEY = mcp_adapter_module._RUNTIME_CONNECTION_REFRESH_KEY
_RESOLVER_KEY = mcp_adapter_module._OAUTH_TOKEN_RESOLVER_REFRESH_KEY
_SLACK_KEY = mcp_adapter_module._SLACK_ACTOR_RUNTIME_REFRESH_KEY

_FORBIDDEN_IN_PUBLIC_OUTPUT = ("sig=abc", "initial-token-secret", "fresh-token-secret")


def _http_401(authenticate: list[str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", CONNECTOR_URL)
    headers = [(b"www-authenticate", v.encode()) for v in (authenticate or [])]
    response = httpx.Response(401, headers=headers, request=request)
    return httpx.HTTPStatusError(
        "planted-status-error-text", request=request, response=response
    )


def _wrap(exc: Exception, variant: str) -> BaseException:
    if variant == "group":
        return BaseExceptionGroup("fan-out", [exc])
    return exc


def _first_401(variant: str, authenticate: list[str] | None = None) -> BaseException:
    if variant == "text":
        return RuntimeError("HTTP 401 Unauthorized")
    return _wrap(_http_401(authenticate), variant)


def _connection(**extra: Any) -> dict[str, Any]:
    return {
        "transport": "streamable_http",
        "url": CONNECTOR_URL,
        "headers": {"Authorization": INITIAL_AUTH},
        **extra,
    }


def _fresh_connection() -> dict[str, Any]:
    return {
        "transport": "streamable_http",
        "url": CONNECTOR_URL,
        "headers": {"Authorization": FRESH_AUTH},
    }


def _label(connection: Any) -> str:
    if not isinstance(connection, dict):
        return "non-mapping"
    auth = (connection.get("headers") or {}).get("Authorization")
    return {INITIAL_AUTH: "initial", FRESH_AUTH: "fresh", None: "bare"}.get(
        auth, "other"
    )


class _World:
    """Counters shared by the fake connector and the refresh callbacks of one run."""

    def __init__(self) -> None:
        self.server_calls: list[tuple[str, str]] = []
        self.refresh_calls = {"resolver": 0, "connector": 0, "slack": 0}
        self.strict_401_calls = 0


Script = Callable[[str, int], BaseException | None]


@dataclass
class _Case:
    id: str
    connection: Callable[[_World], Any]
    make_script: Callable[[], Script]
    expected: str
    labels: list[str]
    failure_code: str | None = None
    refresh_calls: dict[str, int] = field(default_factory=dict)
    strict_401_calls: int | None = None


@dataclass
class _Outcome:
    category: str
    failure_code: str | None
    labels: list[str]
    tools: list[str]
    refresh_calls: dict[str, int]
    strict_401_calls: int
    public_output: str
    # Label of the connection the issuing helper handed back with its result.
    last_connection_label: str | None = None


def _result_category(result: dict[str, Any]) -> str:
    text = result["content"][0]["text"]
    if result["is_error"] is False:
        return "success"
    if text.startswith("Error executing MCP tool: delegated authorization failed"):
        return "auth_failed"
    if text == "Error executing MCP tool after delegated authorization retry.":
        return "retry_failed"
    if text == "Error executing MCP tool.":
        return "raised"
    return f"unexpected: {text}"


def _install_fake_server(
    monkeypatch: pytest.MonkeyPatch, world: _World, script: Script
):
    class _Session:
        def __init__(self, connection: Any) -> None:
            self._label = _label(connection)

        async def initialize(self) -> None:
            return None

        async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
            world.server_calls.append((name, self._label))
            exc = script(self._label, len(world.server_calls) - 1)
            if exc is not None:
                raise exc
            return CallToolResult(
                content=[TextContent(type="text", text="ok")], isError=False
            )

    @asynccontextmanager
    async def _create_session(connection: Any):
        yield _Session(connection)

    monkeypatch.setattr(mcp_adapter_module, "create_session", _create_session)

    original = mcp_adapter_module._strict_http_401_responses

    def _counting(exc: BaseException, **kwargs: Any):
        world.strict_401_calls += 1
        yield from original(exc, **kwargs)

    monkeypatch.setattr(mcp_adapter_module, "_strict_http_401_responses", _counting)


def _adapter(connection: Any) -> MCPToolAdapter:
    return MCPToolAdapter(
        mcp_tool=SimpleNamespace(
            name=OUTER_TOOL,
            description="Attach a document",
            inputSchema={"type": "object", "properties": {}},
        ),
        connection=connection,
    )


async def _run_side(
    case: _Case,
    side: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> _Outcome:
    world = _World()
    _install_fake_server(monkeypatch, world, case.make_script())
    adapter = _adapter(case.connection(world))
    caplog.clear()

    failure_code: str | None = None
    last_connection_label: str | None = None
    if side == "outer":
        result = await adapter.run_json_async({})
        category = _result_category(result)
        failure_code = result.get("failure_code")
        rendered = repr(result)
    else:
        invocation_connection, was_refreshed = await adapter._invocation_connection()
        if isinstance(invocation_connection, dict) and not was_refreshed:
            # What the call is made with differs from the adapter's own
            # connection, which carries the refresh callbacks.
            invocation_connection = {
                key: value
                for key, value in invocation_connection.items()
                if not key.startswith("_")
            }
        try:
            (
                result,
                last_connection,
            ) = await adapter._call_issuing_tool_with_authorization_retry(
                invocation_connection,
                ISSUING_TOOL,
                {"filename": "a.pdf"},
                {},
                was_refreshed=was_refreshed,
            )
        except (BaseExceptionGroup, Exception) as exc:
            category = "raised"
            rendered = type(exc).__name__
        else:
            category = _result_category(result)
            failure_code = result.get("failure_code")
            rendered = repr(result)
            last_connection_label = _label(last_connection)

    return _Outcome(
        category=category,
        failure_code=failure_code,
        labels=[label for _, label in world.server_calls],
        tools=[tool for tool, _ in world.server_calls],
        refresh_calls=dict(world.refresh_calls),
        strict_401_calls=world.strict_401_calls,
        public_output=rendered + caplog.text,
        last_connection_label=last_connection_label,
    )


def _async_value(world: _World, key: str, value: Any) -> Callable[..., Any]:
    async def _refresh(*_args: Any) -> Any:
        world.refresh_calls[key] += 1
        return value() if callable(value) else value

    return _refresh


def _raising(world: _World, key: str) -> Callable[..., Any]:
    async def _refresh(*_args: Any) -> Any:
        world.refresh_calls[key] += 1
        raise RuntimeError(LEAKY_MESSAGE)

    return _refresh


def _retry_failure() -> RuntimeError:
    return RuntimeError(f"transport failed with {FRESH_AUTH} {LEAKY_MESSAGE}")


def _runtime_cases(variant: str) -> list[_Case]:
    def refresh_to(world: _World, value: Any) -> Callable[..., Any]:
        def _refresh() -> Any:
            world.refresh_calls["connector"] += 1
            return value() if callable(value) else value

        return _refresh

    def raising_refresh(world: _World) -> Callable[..., Any]:
        def _refresh() -> Any:
            world.refresh_calls["connector"] += 1
            raise RuntimeError(LEAKY_MESSAGE)

        return _refresh

    def connection_with(refresh: Callable[[_World], Any]) -> Callable[[_World], Any]:
        return lambda world: _connection(**{_RUNTIME_KEY: refresh(world)})

    def fail_initial(
        then: Callable[[str], BaseException | None],
    ) -> Callable[[], Script]:
        def make() -> Script:
            def script(label: str, _n: int) -> BaseException | None:
                if label == "initial":
                    return _first_401(variant)
                return then(label)

            return script

        return make

    def retry_401(label: str) -> BaseException | None:
        return _first_401(variant) if label == "fresh" else None

    def retry_other(label: str) -> BaseException | None:
        return _retry_failure() if label == "fresh" else None

    def succeed(_label: str) -> BaseException | None:
        return None

    tag = f"[{variant}]"
    return [
        _Case(
            "runtime-first-call-succeeds" + tag,
            lambda world: _connection(),
            lambda: lambda label, n: None,
            "success",
            ["initial"],
        ),
        _Case(
            "runtime-refresh-then-retry-succeeds" + tag,
            connection_with(lambda w: refresh_to(w, _fresh_connection)),
            fail_initial(succeed),
            "success",
            ["initial", "fresh"],
            refresh_calls={"connector": 1, "resolver": 0},
        ),
        _Case(
            "runtime-refresh-raises" + tag,
            connection_with(raising_refresh),
            fail_initial(succeed),
            "auth_failed",
            ["initial"],
            refresh_calls={"connector": 1},
        ),
        _Case(
            "runtime-refresh-returns-non-dict" + tag,
            connection_with(lambda w: refresh_to(w, ["not", "a", "dict"])),
            fail_initial(succeed),
            "auth_failed",
            ["initial"],
            refresh_calls={"connector": 1},
        ),
        _Case(
            "runtime-refresh-returns-empty-dict-and-retry-fails" + tag,
            connection_with(lambda w: refresh_to(w, dict)),
            fail_initial(lambda label: RuntimeError("malformed connector retry")),
            "retry_failed",
            ["initial", "bare"],
            refresh_calls={"connector": 1},
        ),
        _Case(
            "runtime-retry-is-401-again" + tag,
            connection_with(lambda w: refresh_to(w, _fresh_connection)),
            fail_initial(retry_401),
            "auth_failed",
            ["initial", "fresh"],
            refresh_calls={"connector": 1},
        ),
        _Case(
            "runtime-retry-fails-with-other-error" + tag,
            connection_with(lambda w: refresh_to(w, _fresh_connection)),
            fail_initial(retry_other),
            "retry_failed",
            ["initial", "fresh"],
            refresh_calls={"connector": 1},
        ),
        _Case(
            "runtime-refresh-is-not-callable" + tag,
            lambda world: _connection(**{_RUNTIME_KEY: "not callable"}),
            fail_initial(succeed),
            "raised",
            ["initial"],
        ),
        _Case(
            "runtime-connection-is-not-a-mapping" + tag,
            lambda world: SimpleNamespace(transport="streamable_http"),
            lambda: lambda label, n: _first_401(variant),
            "raised",
            ["non-mapping"],
        ),
        _Case(
            "runtime-trusted-grant-401-is-not-retried" + tag,
            lambda world: _connection(
                **{
                    _SLACK_KEY: _async_value(world, "slack", _fresh_connection),
                    _RUNTIME_KEY: refresh_to(world, _fresh_connection),
                }
            ),
            lambda: lambda label, n: _first_401(variant),
            "raised",
            ["fresh"],
            refresh_calls={"slack": 1, "connector": 0},
        ),
    ]


def _resolver_cases(variant: str) -> list[_Case]:
    tag = f"[{variant}]"

    def connection(refresh: Callable[[_World], Any], with_connector: bool = True):
        def build(world: _World) -> dict[str, Any]:
            extra: dict[str, Any] = {_RESOLVER_KEY: refresh(world)}
            if with_connector:
                extra[_RUNTIME_KEY] = _async_value(world, "connector", {})
            return _connection(**extra)

        return build

    def fresh_refresh(world: _World) -> Callable[..., Any]:
        return _async_value(world, "resolver", _fresh_connection)

    def first_then(label_to_exc: Callable[[], Callable[[str], BaseException | None]]):
        def make() -> Script:
            then = label_to_exc()

            def script(label: str, _n: int) -> BaseException | None:
                if label == "initial":
                    return _first_401(variant, [INVALID_TOKEN])
                return then(label)

            return script

        return make

    def ok() -> Callable[[str], BaseException | None]:
        return lambda label: None

    def again_401() -> Callable[[str], BaseException | None]:
        return lambda label: _first_401(variant, [INVALID_TOKEN])

    def other_error() -> Callable[[str], BaseException | None]:
        return lambda label: _retry_failure()

    def same_instance() -> Script:
        shared = _first_401(variant, [INVALID_TOKEN])
        return lambda label, n: shared

    def chained_from_initial() -> Script:
        initial = _first_401(variant, [INVALID_TOKEN])
        failure = _retry_failure()
        failure.__cause__ = initial

        def script(label: str, n: int) -> BaseException | None:
            return initial if n == 0 else failure

        return script

    def rewrapped() -> Script:
        initial = _http_401([INVALID_TOKEN])
        rewrapped_error = httpx.HTTPStatusError(
            "rewrapped initial response",
            request=initial.request,
            response=initial.response,
        )

        def script(label: str, n: int) -> BaseException | None:
            return (
                _wrap(initial, variant) if n == 0 else _wrap(rewrapped_error, variant)
            )

        return script

    def over_budget() -> Script:
        deep: BaseException = _http_401([INVALID_TOKEN])
        for index in range(mcp_adapter_module._EXCEPTION_WALK_NODE_LIMIT + 1):
            wrapper = RuntimeError(f"initial-wrapper-{index}")
            wrapper.__cause__ = deep
            deep = wrapper
        initial: BaseException = ExceptionGroup(
            "initial", [_http_401([INVALID_TOKEN]), deep]
        )
        if variant == "group":
            initial = BaseExceptionGroup("outer", [initial])  # type: ignore[list-item]

        def script(label: str, n: int) -> BaseException | None:
            return initial if n == 0 else _retry_failure()

        return script

    cases = [
        _Case(
            "resolver-refresh-then-retry-succeeds" + tag,
            connection(fresh_refresh),
            first_then(ok),
            "success",
            ["initial", "fresh"],
            refresh_calls={"resolver": 1, "connector": 0},
        ),
        _Case(
            "resolver-classified-failure-code" + tag,
            connection(
                lambda w: _async_value(
                    w,
                    "resolver",
                    ClassifiedToolFailure(failure_code="oauth_token_required"),
                )
            ),
            first_then(ok),
            "auth_failed",
            ["initial"],
            failure_code="oauth_token_required",
            refresh_calls={"resolver": 1, "connector": 0},
            strict_401_calls=1,
        ),
        _Case(
            "resolver-refresh-raises" + tag,
            connection(lambda w: _raising(w, "resolver")),
            first_then(ok),
            "auth_failed",
            ["initial"],
            refresh_calls={"resolver": 1, "connector": 0},
        ),
        _Case(
            "resolver-retry-is-401-again" + tag,
            connection(fresh_refresh),
            first_then(again_401),
            "auth_failed",
            ["initial", "fresh"],
            refresh_calls={"resolver": 1, "connector": 0},
        ),
        _Case(
            "resolver-retry-fails-with-other-error" + tag,
            connection(fresh_refresh),
            first_then(other_error),
            "retry_failed",
            ["initial", "fresh"],
            refresh_calls={"resolver": 1, "connector": 0},
        ),
        _Case(
            "resolver-retry-raises-the-initial-exception-instance" + tag,
            connection(fresh_refresh),
            same_instance,
            # The retried exception is the initial one, whose subtree is not
            # walked again: a bare 401 is the root itself and counts, the same
            # 401 inside a group does not.
            "auth_failed" if variant == "bare" else "retry_failed",
            ["initial", "fresh"],
            refresh_calls={"resolver": 1, "connector": 0},
        ),
        _Case(
            "resolver-retry-failure-is-caused-by-the-initial-401" + tag,
            connection(fresh_refresh),
            chained_from_initial,
            "retry_failed",
            ["initial", "fresh"],
            refresh_calls={"resolver": 1, "connector": 0},
        ),
        _Case(
            "resolver-retry-rewraps-the-initial-401-response" + tag,
            connection(fresh_refresh),
            rewrapped,
            "retry_failed",
            ["initial", "fresh"],
            refresh_calls={"resolver": 1, "connector": 0},
        ),
        _Case(
            "resolver-initial-chain-over-the-node-limit" + tag,
            connection(fresh_refresh),
            over_budget,
            "retry_failed",
            ["initial", "fresh"],
            refresh_calls={"resolver": 1, "connector": 0},
        ),
        _Case(
            "resolver-initial-error-has-no-401-response" + tag,
            connection(fresh_refresh),
            lambda: lambda label, n: _wrap(RuntimeError(LEAKY_MESSAGE), variant),
            "raised",
            ["initial"],
            refresh_calls={"resolver": 0, "connector": 0},
        ),
    ]
    for index, refreshed in enumerate(_NON_EXECUTABLE_CONNECTIONS):
        cases.append(
            _Case(
                f"resolver-refresh-returns-unusable-connection-{index}" + tag,
                connection(
                    lambda w, refreshed=refreshed: _async_value(
                        w, "resolver", lambda: refreshed
                    )
                ),
                first_then(ok),
                "auth_failed",
                ["initial"],
                refresh_calls={"resolver": 1, "connector": 0},
            )
        )
    for index, authenticate in enumerate(_UNUSABLE_CHALLENGES):
        cases.append(
            _Case(
                f"resolver-401-without-invalid-token-challenge-{index}" + tag,
                connection(fresh_refresh),
                lambda authenticate=authenticate: (
                    lambda label, n: _first_401(variant, authenticate)
                ),
                "auth_failed",
                ["initial"],
                refresh_calls={"resolver": 0, "connector": 0},
            )
        )
    return cases


_NON_EXECUTABLE_CONNECTIONS: list[Any] = [
    None,
    SimpleNamespace(secret="planted-malformed-connection"),
    {},
    {"transport": "unsupported", "url": CONNECTOR_URL},
    {"transport": "stdio"},
    {"transport": "stdio", "command": ""},
    {"transport": "stdio", "command": ["python"]},
    {"transport": "stdio", "command": "python"},
    {"transport": "sse"},
    {"transport": "streamable_http"},
    {"transport": "streamable_http", "url": ""},
    {"transport": "streamable_http", "url": 42},
    {"transport": "websocket"},
]

_UNUSABLE_CHALLENGES: list[list[str]] = [
    [],
    ['Bearer error="invalid_token'],
    ['Basic error="invalid_token"'],
    ['Bearer error="insufficient_scope"'],
    ['Bearer scope="records.read"'],
]


def _text_only_resolver_case() -> _Case:
    return _Case(
        "resolver-text-only-401-uses-no-refresh",
        lambda world: _connection(
            **{
                _RESOLVER_KEY: _async_value(world, "resolver", {}),
                _RUNTIME_KEY: _async_value(world, "connector", {}),
            }
        ),
        lambda: (
            lambda label, n: RuntimeError(
                "HTTP 401 Unauthorized; Bearer error=invalid_token"
            )
        ),
        "raised",
        ["initial"],
        refresh_calls={"resolver": 0, "connector": 0},
    )


def _no_refresh_cases() -> list[_Case]:
    return [
        _Case(
            "other-error-without-any-refresh",
            lambda world: _connection(),
            lambda: lambda label, n: RuntimeError(LEAKY_MESSAGE),
            "raised",
            ["initial"],
        ),
    ]


def _all_cases() -> list[_Case]:
    cases = _no_refresh_cases() + [_text_only_resolver_case()]
    for variant in ("bare", "group", "text"):
        cases += _runtime_cases(variant)
    for variant in ("bare", "group"):
        cases += _resolver_cases(variant)
    return cases


_CASES = _all_cases()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=[case.id for case in _CASES])
async def test_issuing_call_authorization_policy_matches_outer_call(
    case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="xagent")

    outer = await _run_side(case, "outer", monkeypatch, caplog)
    issuing = await _run_side(case, "issuing", monkeypatch, caplog)

    assert outer.category == case.expected
    assert issuing.category == case.expected
    assert outer.failure_code == issuing.failure_code == case.failure_code
    assert outer.labels == issuing.labels == case.labels
    assert outer.refresh_calls == issuing.refresh_calls
    for name, count in case.refresh_calls.items():
        assert outer.refresh_calls[name] == count
    assert outer.strict_401_calls == issuing.strict_401_calls
    if case.strict_401_calls is not None:
        assert outer.strict_401_calls == case.strict_401_calls
    assert set(outer.tools) == {OUTER_TOOL}
    assert set(issuing.tools) == {ISSUING_TOOL}
    if issuing.category != "raised":
        # The helper hands back the connection its last call was made with.
        assert issuing.last_connection_label == issuing.labels[-1]
    for outcome in (outer, issuing):
        for secret in _FORBIDDEN_IN_PUBLIC_OUTPUT:
            assert secret not in outcome.public_output


def test_issuing_call_cases_cover_both_exception_shapes() -> None:
    ids = {case.id for case in _CASES}
    for variant in ("bare", "group"):
        assert f"runtime-refresh-then-retry-succeeds[{variant}]" in ids
        assert f"resolver-refresh-then-retry-succeeds[{variant}]" in ids


@pytest.mark.asyncio
async def test_issuing_call_succeeds_through_a_real_session_after_401_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both paths survive the nested exception group a real MCP session raises."""

    async def run(side: str) -> tuple[Any, list[str], list[str], int]:
        initial_methods: list[str] = []
        refreshed_tools: list[str] = []
        refreshed_methods: list[str] = []
        refresh_calls = 0

        async def initial_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content) if request.content else {}
            initial_methods.append(payload.get("method", ""))
            return httpx.Response(
                401,
                headers={"WWW-Authenticate": INVALID_TOKEN},
                request=request,
            )

        async def refreshed_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content) if request.content else {}
            method = payload.get("method")
            refreshed_methods.append(method or "")
            if method == "initialize":
                result: dict[str, Any] = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock-mcp", "version": "1"},
                }
            elif method == "tools/call":
                refreshed_tools.append(payload["params"]["name"])
                result = {
                    "content": [{"type": "text", "text": "ok"}],
                    "isError": False,
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": name,
                            "description": name,
                            "inputSchema": {"type": "object"},
                        }
                        for name in (OUTER_TOOL, ISSUING_TOOL)
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

        def factory_for(handler: Any) -> Callable[..., httpx.AsyncClient]:
            def _factory(headers=None, timeout=None, auth=None):
                return httpx.AsyncClient(
                    transport=httpx.MockTransport(handler),
                    headers=headers,
                    timeout=timeout,
                    auth=auth,
                )

            return _factory

        async def resolver_refresh(challenge: Any) -> dict[str, Any]:
            nonlocal refresh_calls
            refresh_calls += 1
            return {
                "transport": "streamable_http",
                "url": CONNECTOR_URL,
                "headers": {"Authorization": FRESH_AUTH},
                "httpx_client_factory": factory_for(refreshed_handler),
                "terminate_on_close": False,
            }

        connection = {
            "transport": "streamable_http",
            "url": CONNECTOR_URL,
            "headers": {"Authorization": INITIAL_AUTH},
            "httpx_client_factory": factory_for(initial_handler),
            "terminate_on_close": False,
            _RESOLVER_KEY: resolver_refresh,
        }
        adapter = _adapter(connection)
        if side == "outer":
            result_value = await adapter.run_json_async({})
        else:
            (
                result_value,
                last_connection,
            ) = await adapter._call_issuing_tool_with_authorization_retry(
                {k: v for k, v in connection.items() if not k.startswith("_")},
                ISSUING_TOOL,
                {"filename": "a.pdf"},
                {},
                was_refreshed=False,
            )
            assert last_connection["headers"] == {"Authorization": FRESH_AUTH}
        return result_value, refreshed_tools, initial_methods, refresh_calls

    outer_result, outer_tools, outer_initial, outer_refreshes = await run("outer")
    issuing_result, issuing_tools, issuing_initial, issuing_refreshes = await run(
        "issuing"
    )

    assert outer_result["is_error"] is False
    assert issuing_result["is_error"] is False
    assert outer_tools == [OUTER_TOOL]
    assert issuing_tools == [ISSUING_TOOL]
    assert outer_initial == issuing_initial == ["initialize"]
    assert outer_refreshes == issuing_refreshes == 1


# --------------------------------------------------------------------------
# _execute_named_mcp_call
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_named_mcp_call_sends_name_arguments_and_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, dict[str, Any], Any]] = []

    class _Session:
        async def initialize(self) -> None:
            return None

        async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
            received.append((name, arguments, kwargs.get("meta")))
            return CallToolResult(
                content=[TextContent(type="text", text="done")],
                structuredContent={"k": "v"},
                isError=False,
            )

    @asynccontextmanager
    async def _create_session(connection: Any):
        yield _Session()

    monkeypatch.setattr(mcp_adapter_module, "create_session", _create_session)
    adapter = _adapter(_connection())
    arguments = {"filename": "a.pdf"}

    result = await adapter._execute_named_mcp_call(
        adapter.connection, ISSUING_TOOL, arguments, {"trace": "t1"}
    )
    empty_meta = await adapter._execute_named_mcp_call(
        adapter.connection, ISSUING_TOOL, arguments, {}
    )

    assert received[0] == (ISSUING_TOOL, {"filename": "a.pdf"}, {"trace": "t1"})
    assert received[0][1] is not arguments
    assert received[1][2] is None
    assert result["structured_content"] == {"k": "v"}
    assert result["is_error"] is False
    assert empty_meta["content"][0]["text"] == "done"


@pytest.mark.asyncio
async def test_execute_named_mcp_call_has_no_tool_specific_post_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Session:
        async def initialize(self) -> None:
            return None

        async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "status": "error",
                                "failure_code": "excel_workbook_access_denied",
                            }
                        ),
                    )
                ],
                isError=False,
            )

    @asynccontextmanager
    async def _create_session(connection: Any):
        yield _Session()

    monkeypatch.setattr(mcp_adapter_module, "create_session", _create_session)
    adapter = MCPToolAdapter(
        mcp_tool=SimpleNamespace(
            name="excel_list_worksheets",
            description="List worksheets",
            inputSchema={"type": "object", "properties": {}},
        ),
        connection=_connection(),
        source_server="excel",
    )

    post_processed = await adapter._execute_mcp_call(adapter.connection, {}, {})
    named = await adapter._execute_named_mcp_call(
        adapter.connection, ISSUING_TOOL, {}, {}
    )

    assert post_processed["failure_code"] == "excel_workbook_access_denied"
    assert "failure_code" not in named
    assert "success" not in named
    assert named["is_error"] is False


@pytest.mark.asyncio
async def test_issuing_call_does_not_use_the_model_call_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _World()
    _install_fake_server(monkeypatch, world, lambda label, n: None)
    adapter = _adapter(_connection())

    async def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("_execute_mcp_call must not run for a host-issued call")

    monkeypatch.setattr(adapter, "_execute_mcp_call", _forbidden)

    result, last_connection = await adapter._call_issuing_tool_with_authorization_retry(
        adapter.connection, ISSUING_TOOL, {}, {}, was_refreshed=False
    )

    assert result["is_error"] is False
    assert last_connection is adapter.connection
    assert world.server_calls == [(ISSUING_TOOL, "initial")]

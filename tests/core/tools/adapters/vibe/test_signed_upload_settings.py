from __future__ import annotations

import copy
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe import signed_upload
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPServerLoadFailure,
    _build_mcp_tool_adapter,
    load_mcp_tools_as_agent_tools,
    mcp_load_failure_message,
)
from xagent.core.tools.adapters.vibe.selection_spec import (
    ToolSelectionSpec,
    normalize_mcp_server_name,
)
from xagent.core.tools.adapters.vibe.signed_upload import (
    SignedUploadBinding,
    SignedUploadIssuer,
    connection_signed_uploads,
    parse_signed_upload_response,
    parse_signed_upload_settings,
    signed_upload_settings_error,
)

CONNECTION_HOST = "mcp.example.test"
CONNECTION_URL = f"https://{CONNECTION_HOST}/mcp"
UNREADABLE = "Upload settings could not be read."


def _entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "tool_name": "attach_document",
        "file_argument": "document",
        "issuing_tool_name": "request_upload",
        "allowed_upload_hosts": ["uploads.example.test"],
        "upload_path_prefix": "/staged/",
        "content_types": {".pdf": "application/pdf", ".png": "image/png"},
    }
    entry.update(overrides)
    return entry


def _without(key: str, **overrides: Any) -> dict[str, Any]:
    entry = _entry(**overrides)
    del entry[key]
    return entry


def _binding(**overrides: Any) -> SignedUploadBinding:
    fields: dict[str, Any] = {
        "connection_host": CONNECTION_HOST,
        "tool_name": "attach_document",
        "file_argument": "document",
        "issuing_tool_name": "request_upload",
        "allowed_upload_hosts": frozenset({"uploads.example.test"}),
        "upload_path_prefix": "/staged/",
        "content_types": {".pdf": "application/pdf", ".png": "image/png"},
    }
    fields.update(overrides)
    return SignedUploadBinding(**fields)


def _issuer_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "issuing_tool_name": "request_upload",
        "allowed_upload_hosts": ["uploads.example.test"],
        "upload_path_prefix": "/staged/",
        "content_types": {".pdf": "application/pdf", ".png": "image/png"},
    }
    entry.update(overrides)
    return entry


def _issuer(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "connection_host": CONNECTION_HOST,
        "issuing_tool_name": "request_upload",
        "allowed_upload_hosts": frozenset({"uploads.example.test"}),
        "upload_path_prefix": "/staged/",
        "content_types": {".pdf": "application/pdf", ".png": "image/png"},
    }
    fields.update(overrides)
    return SignedUploadIssuer(**fields)


def _connection(transport: str = "streamable_http", **extra: Any) -> dict[str, Any]:
    return {"transport": transport, "url": CONNECTION_URL, **extra}


def _stdio_connection(**extra: Any) -> dict[str, Any]:
    # Carries a ``url`` key so only the transport check can keep it from
    # reading the settings.
    return {
        "transport": "stdio",
        "command": "python",
        "args": [],
        "url": CONNECTION_URL,
        **extra,
    }


def _parse(raw: Any, connection_host: str | None = CONNECTION_HOST) -> Any:
    return parse_signed_upload_settings(raw, connection_host=connection_host)


def _many_entries(count: int) -> list[dict[str, Any]]:
    return [
        _entry(tool_name=f"tool_{index}", issuing_tool_name="request_upload")
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def _exactly(message: str) -> str:
    return f"^{re.escape(message)}$"


NOT_A_LIST = "signed_uploads must be a list"
NOT_AN_OBJECT = "signed_uploads[0] must be an object"
KEYS = "signed_uploads[0] must have exactly the documented keys"
CLASH = "a tool cannot be both an issuing tool and a replaced tool"
HOSTS = "signed_uploads[0].allowed_upload_hosts"
HOST = f"{HOSTS} entry"
NOT_EXACT_HOST = f"{HOST} must be an exact hostname"
PREFIX = "signed_uploads[0].upload_path_prefix"
TYPES = "signed_uploads[0].content_types"


def _name_field_cells(field: str) -> list[Any]:
    label = f"signed_uploads[0].{field}"
    empty = f"{label} must be a non-empty string"
    return [
        pytest.param([_entry(**{field: ""})], empty, id=f"{field}-empty"),
        pytest.param([_entry(**{field: 7})], empty, id=f"{field}-not-a-string"),
        pytest.param([_entry(**{field: None})], empty, id=f"{field}-none"),
        pytest.param(
            [_entry(**{field: "x" * 256})],
            f"{label} must not be longer than 255",
            id=f"{field}-too-long",
        ),
    ]


# Each case names the exact error it must raise, so a case that a different
# rule happens to reject fails.
REJECTED_SETTINGS = [
    # not a list
    pytest.param({"tool_name": "attach_document"}, NOT_A_LIST, id="value-is-a-dict"),
    pytest.param("attach_document", NOT_A_LIST, id="value-is-a-string"),
    pytest.param(7, NOT_A_LIST, id="value-is-a-number"),
    pytest.param(False, NOT_A_LIST, id="value-is-false"),
    pytest.param("", NOT_A_LIST, id="value-is-an-empty-string"),
    pytest.param({}, NOT_A_LIST, id="value-is-an-empty-dict"),
    pytest.param((_entry(),), NOT_A_LIST, id="value-is-a-tuple"),
    # too many entries
    pytest.param(
        _many_entries(21),
        "signed_uploads must not have more than 20 entries",
        id="too-many-entries",
    ),
    # entry shape
    pytest.param(["attach_document"], NOT_AN_OBJECT, id="entry-is-a-string"),
    pytest.param([None], NOT_AN_OBJECT, id="entry-is-none"),
    pytest.param([[_entry()]], NOT_AN_OBJECT, id="entry-is-a-list"),
    pytest.param(
        [_entry(), 7],
        "signed_uploads[1] must be an object",
        id="second-entry-is-a-number",
    ),
    pytest.param([{}], KEYS, id="entry-is-empty"),
    *[pytest.param([_without(key)], KEYS, id=f"missing-{key}") for key in _entry()],
    pytest.param([_entry(extra="x")], KEYS, id="extra-key"),
    # names
    *_name_field_cells("tool_name"),
    *_name_field_cells("file_argument"),
    *_name_field_cells("issuing_tool_name"),
    # clashes between entries
    pytest.param(
        [_entry(), _entry(issuing_tool_name="other_issuer")],
        "signed_uploads[1].tool_name repeats an earlier entry",
        id="duplicate-tool-name",
    ),
    pytest.param(
        [
            _entry(),
            _entry(tool_name="attach_image", issuing_tool_name="attach_document"),
        ],
        CLASH,
        id="issuing-tool-of-a-later-entry-is-an-earlier-tool",
    ),
    pytest.param(
        [_entry(), _entry(tool_name="request_upload", issuing_tool_name="issue_other")],
        CLASH,
        id="tool-of-a-later-entry-is-an-earlier-issuing-tool",
    ),
    pytest.param(
        [_entry(issuing_tool_name="attach_document")],
        CLASH,
        id="issuing-tool-is-the-tool",
    ),
    # allowed_upload_hosts
    pytest.param(
        [_entry(allowed_upload_hosts="uploads.example.test")],
        f"{HOSTS} must be a list of hostnames",
        id="hosts-is-a-string",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=frozenset({"uploads.example.test"}))],
        f"{HOSTS} must be a list of hostnames",
        id="hosts-is-a-frozenset",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=None)],
        f"{HOSTS} must be a list of hostnames",
        id="hosts-is-none",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=[])],
        f"{HOSTS} must not be empty",
        id="hosts-empty",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=[f"h{i}.example.test" for i in range(21)])],
        f"{HOSTS} must not have more than 20 entries",
        id="hosts-too-many",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["*.example.test"])],
        NOT_EXACT_HOST,
        id="host-star",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["192.168.1.5"])],
        f"{HOST} must be a hostname, not an IP address",
        id="host-ip",
    ),
    # An IPv6 literal always contains ":", so the character rule rejects it
    # before the IP-address rule is reached.
    pytest.param(
        [_entry(allowed_upload_hosts=["::1"])], NOT_EXACT_HOST, id="host-ipv6-colon"
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["Uploads.example.test"])],
        f"{HOST} must be lower-case",
        id="host-uppercase",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["uploads.example.test."])],
        f"{HOST} must not end with a dot",
        id="host-trailing-dot",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["uploads.example.test:8443"])],
        NOT_EXACT_HOST,
        id="host-colon",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["user@uploads.example.test"])],
        NOT_EXACT_HOST,
        id="host-at",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["uploads.example.test/x"])],
        NOT_EXACT_HOST,
        id="host-slash",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["uploads example.test"])],
        NOT_EXACT_HOST,
        id="host-space",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["uploads.example.test\t"])],
        NOT_EXACT_HOST,
        id="host-tab",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=[""])],
        f"{HOST} must be a non-empty string",
        id="host-empty",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=[7])],
        f"{HOST} must be a non-empty string",
        id="host-not-a-string",
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["uploads.example.test", "a" * 254])],
        f"{HOST} must not be longer than 253",
        id="host-too-long",
    ),
    # upload_path_prefix
    pytest.param(
        [_entry(upload_path_prefix="staged/")],
        f"{PREFIX} must start and end with '/'",
        id="prefix-no-leading-slash",
    ),
    pytest.param(
        [_entry(upload_path_prefix="/staged")],
        f"{PREFIX} must start and end with '/'",
        id="prefix-no-trailing-slash",
    ),
    pytest.param(
        [_entry(upload_path_prefix="/staged/../x/")],
        f"{PREFIX} must not contain '..'",
        id="prefix-dotdot",
    ),
    pytest.param(
        [_entry(upload_path_prefix="/" + "p" * 254 + "/")],
        f"{PREFIX} must not be longer than 255",
        id="prefix-too-long",
    ),
    pytest.param(
        [_entry(upload_path_prefix=None)],
        f"{PREFIX} must start and end with '/'",
        id="prefix-not-a-string",
    ),
    # content_types
    pytest.param(
        [_entry(content_types=None)],
        f"{TYPES} must be an object",
        id="content-types-none",
    ),
    pytest.param(
        [_entry(content_types=[(".pdf", "application/pdf")])],
        f"{TYPES} must be an object",
        id="content-types-not-a-dict",
    ),
    pytest.param(
        [_entry(content_types={})],
        f"{TYPES} must not be empty",
        id="content-types-empty",
    ),
    pytest.param(
        [_entry(content_types={f".t{i}": "text/plain" for i in range(51)})],
        f"{TYPES} must not have more than 50 entries",
        id="content-types-too-many",
    ),
    pytest.param(
        [_entry(content_types={"pdf": "application/pdf"})],
        f"{TYPES} keys must start with '.'",
        id="suffix-no-dot",
    ),
    pytest.param(
        [_entry(content_types={".PDF": "application/pdf"})],
        f"{TYPES} keys must be lower-case",
        id="suffix-uppercase",
    ),
    pytest.param(
        [_entry(content_types={"." + "s" * 16: "application/pdf"})],
        f"{TYPES} keys must not be longer than 16",
        id="suffix-too-long",
    ),
    pytest.param(
        [_entry(content_types={".tar.gz": "application/gzip"})],
        f"{TYPES} keys must contain only one '.'",
        id="suffix-two-dots",
    ),
    pytest.param(
        [_entry(content_types={".": "application/pdf"})],
        f"{TYPES} keys must have a suffix after '.'",
        id="content-type-key-dot-only",
    ),
    pytest.param(
        [_entry(content_types={".pdf": ""})],
        f"{TYPES} values must be non-empty",
        id="mime-empty",
    ),
    pytest.param(
        [_entry(content_types={".pdf": 7})],
        f"{TYPES} values must be non-empty",
        id="mime-not-a-string",
    ),
    pytest.param(
        [_entry(content_types={".pdf": "m" * 256})],
        f"{TYPES} values must not be longer than 255",
        id="mime-too-long",
    ),
]


@pytest.mark.parametrize(("raw", "message"), REJECTED_SETTINGS)
def test_parse_signed_upload_settings_rejects(raw: Any, message: str) -> None:
    with pytest.raises(ValueError, match=_exactly(message)):
        _parse(raw)


PLANTED_VALUES = [
    pytest.param([_entry(PLANTED_extra="x")], id="extra-key-name"),
    pytest.param(
        [_entry(content_types={".PLANTED": "application/pdf"})], id="suffix-uppercase"
    ),
    pytest.param(
        [_entry(allowed_upload_hosts=["PLANTED.example.test"])], id="host-uppercase"
    ),
    pytest.param([_entry(tool_name="PLANTED" + "x" * 300)], id="tool-name-too-long"),
    pytest.param([_entry(upload_path_prefix="PLANTED")], id="prefix-no-slashes"),
    pytest.param([_entry(upload_path_prefix="/PLANTED/../")], id="prefix-dotdot"),
    pytest.param(
        [_entry(content_types={".pdf": "PLANTED" + "m" * 300})], id="mime-too-long"
    ),
    pytest.param(
        [_entry(tool_name="PLANTED"), _entry(tool_name="PLANTED")],
        id="duplicate-tool-name",
    ),
    pytest.param(
        [_entry(tool_name="PLANTED", issuing_tool_name="PLANTED")],
        id="issuing-tool-is-the-tool",
    ),
]


@pytest.mark.parametrize("raw", PLANTED_VALUES)
def test_parse_signed_upload_settings_errors_carry_no_stored_value(raw: Any) -> None:
    with pytest.raises(ValueError) as exc_info:
        _parse(raw)

    assert "PLANTED" not in str(exc_info.value)


def _boundary_entry() -> dict[str, Any]:
    return _entry(
        tool_name="t" * 255,
        file_argument="f" * 255,
        issuing_tool_name="i" * 255,
        allowed_upload_hosts=[f"h{i}.example.test" for i in range(19)] + ["a" * 253],
        upload_path_prefix="/" + "p" * 253 + "/",
        content_types={f".t{i:02d}": "m" * 255 for i in range(49)}
        | {"." + "s" * 15: "x/y"},
    )


def test_parse_signed_upload_settings_accepts() -> None:
    raw = [_entry(), _entry(tool_name="attach_image")]

    result = _parse(raw)

    assert isinstance(result, tuple)
    assert result == (_binding(), _binding(tool_name="attach_image"))
    for binding in result:
        assert isinstance(binding.allowed_upload_hosts, frozenset)
        assert isinstance(binding.content_types, MappingProxyType)
        assert binding.connection_host == CONNECTION_HOST
    with pytest.raises(TypeError):
        result[0].content_types[".txt"] = "text/plain"  # type: ignore[index]


def test_parse_signed_upload_settings_copies_what_it_keeps() -> None:
    raw = [_entry()]

    result = _parse(raw)
    raw[0]["content_types"][".txt"] = "text/plain"
    raw[0]["allowed_upload_hosts"].append("later.example.test")

    assert result == (_binding(),)


@pytest.mark.parametrize(
    "connection_host", ["10.0.0.1", "mcp.example.test.", "::1", "MCP.Example.Test"]
)
def test_parse_signed_upload_settings_takes_connection_host_as_given(
    connection_host: str,
) -> None:
    result = _parse([_entry()], connection_host=connection_host)

    assert result == (_binding(connection_host=connection_host),)
    assert result[0].connection_host == connection_host


def test_parse_signed_upload_settings_accepts_limits_exactly() -> None:
    result = _parse([_boundary_entry()])

    assert len(result) == 1
    assert len(result[0].allowed_upload_hosts) == 20
    assert len(result[0].content_types) == 50
    assert len(_parse(_many_entries(20))) == 20


def test_parse_signed_upload_settings_accepts_repeated_upload_host() -> None:
    result = _parse(
        [_entry(allowed_upload_hosts=["uploads.example.test", "uploads.example.test"])]
    )

    assert result[0].allowed_upload_hosts == frozenset({"uploads.example.test"})


@pytest.mark.parametrize("raw", [None, []], ids=["none", "empty-list"])
@pytest.mark.parametrize("connection_host", [CONNECTION_HOST, None])
def test_parse_signed_upload_settings_empty_value(
    raw: Any, connection_host: str | None
) -> None:
    assert _parse(raw, connection_host=connection_host) == ()


NO_HOSTNAME = "the connection URL has no hostname"


@pytest.mark.parametrize("connection_host", [None, ""], ids=["none", "empty"])
def test_parse_signed_upload_settings_non_empty_needs_a_connection_host(
    connection_host: str | None,
) -> None:
    with pytest.raises(ValueError, match=_exactly(NO_HOSTNAME)):
        _parse([_entry()], connection_host=connection_host)


# ---------------------------------------------------------------------------
# signed_upload_settings_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transport", ["streamable_http", "sse"])
def test_signed_upload_settings_error(transport: str) -> None:
    def error(raw: Any, url: Any = CONNECTION_URL) -> str | None:
        return signed_upload_settings_error(raw, transport=transport, url=url)

    assert error([_entry()]) is None
    assert error(None) is None
    assert error([]) is None
    assert error(None, url=None) is None
    # Present but unusable.
    for bad in ({"tool_name": "x"}, [{}], [_entry(allowed_upload_hosts=["10.0.0.1"])]):
        message = error(bad)
        assert isinstance(message, str) and message
    for bad_url in (None, 7, "https://[bad/mcp", "https:///mcp"):
        message = error([_entry()], url=bad_url)
        assert isinstance(message, str) and message
    # The reason never carries a stored value.
    message = error([_entry(PLANTED_extra="x")])
    assert message and "PLANTED" not in message


@pytest.mark.parametrize(
    "transport",
    ["stdio", "websocket", "oauth", "", None, 7, ["sse"], {"sse"}, ("sse",)],
    ids=repr,
)
def test_signed_upload_settings_error_ignores_other_transports(
    transport: Any,
) -> None:
    for raw in ([_entry()], {"tool_name": "x"}, [{}]):
        assert (
            signed_upload_settings_error(raw, transport=transport, url=CONNECTION_URL)
            is None
        )


def test_signed_upload_settings_error_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(raw: Any, *, connection_host: str | None) -> Any:
        raise TypeError("PLANTED")

    monkeypatch.setattr(signed_upload, "parse_signed_upload_settings", broken)

    assert (
        signed_upload_settings_error([_entry()], transport="sse", url=CONNECTION_URL)
        == UNREADABLE
    )


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def _issuing_tools(connection: Any) -> set[str]:
    return {
        binding.issuing_tool_name for binding in connection_signed_uploads(connection)
    }


def _binding_for(connection: Any, tool_name: str) -> SignedUploadBinding | None:
    bindings = {
        binding.tool_name: binding for binding in connection_signed_uploads(connection)
    }
    return bindings.get(tool_name)


def _with_settings(connection: dict[str, Any]) -> dict[str, Any]:
    return {
        **connection,
        "signed_uploads": [_entry(), _entry(tool_name="attach_image")],
    }


@pytest.mark.parametrize(
    ("connection", "tool_name", "expected"),
    [
        pytest.param(
            _with_settings(_connection("streamable_http")),
            "attach_document",
            _binding(),
            id="http",
        ),
        pytest.param(
            _with_settings(_connection("sse")),
            "attach_document",
            _binding(),
            id="sse",
        ),
        pytest.param(
            _with_settings(_connection("streamable_http")),
            "attach_image",
            _binding(tool_name="attach_image"),
            id="second-entry",
        ),
        pytest.param(
            _with_settings(_connection("websocket")), "attach_document", None, id="ws"
        ),
        pytest.param(
            _with_settings(_stdio_connection()), "attach_document", None, id="stdio"
        ),
        pytest.param(
            _with_settings({"transport": "oauth", "url": CONNECTION_URL}),
            "attach_document",
            None,
            id="other-transport",
        ),
        pytest.param(
            _with_settings({"transport": ["sse"], "url": CONNECTION_URL}),
            "attach_document",
            None,
            id="unhashable-transport",
        ),
        pytest.param(
            _with_settings({"url": CONNECTION_URL}),
            "attach_document",
            None,
            id="no-transport",
        ),
        pytest.param(
            _with_settings(_connection()), "other_tool", None, id="other-tool"
        ),
        pytest.param(
            _with_settings(_connection()), "request_upload", None, id="issuing-tool"
        ),
        pytest.param(_connection(), "attach_document", None, id="no-settings"),
        pytest.param(
            _connection(signed_uploads=None), "attach_document", None, id="none"
        ),
        pytest.param(
            _connection(signed_uploads=[]), "attach_document", None, id="empty"
        ),
        pytest.param(
            {"transport": "streamable_http", "signed_uploads": []},
            "attach_document",
            None,
            id="empty-and-no-url",
        ),
        pytest.param(
            _with_settings(
                {"transport": "streamable_http", "url": "https://MCP.Example.TEST/mcp"}
            ),
            "attach_document",
            _binding(),
            id="uppercase-url-host",
        ),
        pytest.param(
            _with_settings(
                {"transport": "streamable_http", "url": "https://other.example.test/x"}
            ),
            "attach_document",
            _binding(connection_host="other.example.test"),
            id="host-is-the-connection-own",
        ),
    ],
)
def test_connection_signed_uploads_binding_for_tool(
    connection: Any, tool_name: str, expected: SignedUploadBinding | None
) -> None:
    assert _binding_for(connection, tool_name) == expected


@pytest.mark.parametrize(
    ("connection", "tool_name", "expected"),
    [
        pytest.param(_with_settings(_connection()), "request_upload", True, id="http"),
        pytest.param(
            _with_settings(_connection("sse")), "request_upload", True, id="sse"
        ),
        pytest.param(
            _with_settings(_connection()), "attach_document", False, id="business-tool"
        ),
        pytest.param(
            _with_settings(_connection("websocket")),
            "request_upload",
            False,
            id="websocket",
        ),
        pytest.param(
            _with_settings(_stdio_connection()), "request_upload", False, id="stdio"
        ),
        pytest.param(_connection(), "request_upload", False, id="no-settings"),
        pytest.param(
            _connection(signed_uploads=[]), "request_upload", False, id="empty"
        ),
    ],
)
def test_connection_signed_uploads_issuing_tools(
    connection: Any, tool_name: str, expected: bool
) -> None:
    assert (tool_name in _issuing_tools(connection)) is expected


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(None, id="no-url"),
        pytest.param(7, id="url-not-a-string"),
        pytest.param("https://[bad/mcp", id="unparseable-url"),
        pytest.param("https:///mcp", id="url-without-hostname"),
    ],
)
def test_lookups_raise_for_settings_on_a_connection_without_hostname(
    url: Any,
) -> None:
    connection = {
        "transport": "streamable_http",
        "url": url,
        "signed_uploads": [_entry()],
    }

    with pytest.raises(ValueError, match=_exactly(NO_HOSTNAME)):
        connection_signed_uploads(connection)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"tool_name": "x"}, NOT_A_LIST),
        ([{}], KEYS),
        (
            [_entry(allowed_upload_hosts=["10.0.0.1"])],
            f"{HOST} must be a hostname, not an IP address",
        ),
    ],
    ids=["not-a-list", "missing-keys", "ip-upload-host"],
)
def test_lookups_raise_for_invalid_settings(raw: Any, message: str) -> None:
    connection = _connection(signed_uploads=raw)

    with pytest.raises(ValueError, match=_exactly(message)):
        connection_signed_uploads(connection)


@pytest.mark.parametrize("transport", ["stdio", "websocket"])
def test_lookups_ignore_invalid_settings_on_other_transports(transport: str) -> None:
    connection = _connection(transport, signed_uploads={"tool_name": "x"})

    assert connection_signed_uploads(connection) == ()


def test_settings_of_two_connections_do_not_interact() -> None:
    first = _connection(signed_uploads=[_entry()])
    second = {
        "transport": "streamable_http",
        "url": "https://other.example.test/mcp",
        "signed_uploads": [
            _entry(tool_name="request_upload", issuing_tool_name="issue_other")
        ],
    }

    assert _binding_for(first, "attach_document") == _binding()
    assert _binding_for(first, "request_upload") is None
    assert _issuing_tools(first) == {"request_upload"}
    assert _binding_for(second, "request_upload") == _binding(
        connection_host="other.example.test",
        tool_name="request_upload",
        issuing_tool_name="issue_other",
    )
    assert _binding_for(second, "attach_document") is None
    assert _issuing_tools(second) == {"issue_other"}


def test_two_connections_on_one_host_keep_their_own_settings() -> None:
    first = _connection(signed_uploads=[_entry()])
    second = _connection(signed_uploads=[_entry(issuing_tool_name="issue_other")])

    assert _issuing_tools(first) == {"request_upload"}
    assert _issuing_tools(second) == {"issue_other"}


# ---------------------------------------------------------------------------
# Loading through the MCP adapter
# ---------------------------------------------------------------------------


def _server_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} tool",
        inputSchema={
            "type": "object",
            "properties": {"document": {"type": "string"}},
        },
    )


def _serve_tools(monkeypatch: pytest.MonkeyPatch, *tool_names: str) -> list[Any]:
    """Serve ``tool_names`` from every server; return the sessions' connections."""
    opened: list[Any] = []

    class FakeSession:
        async def initialize(self) -> None:
            return None

    @asynccontextmanager
    async def fake_create_session(connection: Any):
        opened.append(connection)
        yield FakeSession()

    async def fake_load_mcp_tools(_session: Any) -> list[SimpleNamespace]:
        return [_server_tool(name) for name in tool_names]

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    monkeypatch.setattr(mcp_adapter_module, "load_mcp_tools", fake_load_mcp_tools)
    return opened


def _describe(tool: Any) -> tuple[str, str, dict[str, Any]]:
    return (tool.name, tool.description, tool.args_type().model_json_schema())


_ALL_TOOLS = ("attach_document", "request_upload", "other")


async def _load(connections: dict[str, Any]) -> Any:
    return await load_mcp_tools_as_agent_tools(connections, sandbox=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connection",
    [
        pytest.param(_connection(), id="no-key"),
        pytest.param(_connection(signed_uploads=None), id="none"),
        pytest.param(_connection(signed_uploads=[]), id="empty-list"),
        pytest.param(
            _stdio_connection(signed_uploads=[_entry()]), id="stdio-with-settings"
        ),
        pytest.param(
            {
                "transport": "websocket",
                "url": f"wss://{CONNECTION_HOST}/mcp",
                "signed_uploads": [_entry()],
            },
            id="websocket-with-settings",
        ),
    ],
)
async def test_no_signed_uploads_keeps_tool_list(
    monkeypatch: pytest.MonkeyPatch, connection: dict[str, Any]
) -> None:
    _serve_tools(monkeypatch, *_ALL_TOOLS)
    plain = {k: v for k, v in connection.items() if k != "signed_uploads"}

    expected = await _load({"docs": plain})
    actual = await _load({"docs": connection})

    assert [tool.name for tool in expected.tools] == [
        f"mcp_docs_{name}" for name in _ALL_TOOLS
    ]
    assert [_describe(tool) for tool in actual.tools] == [
        _describe(tool) for tool in expected.tools
    ]
    assert actual.failures == ()
    assert expected.failures == ()
    assert actual.loaded_servers == expected.loaded_servers == ("docs",)


@pytest.mark.parametrize(
    "connection",
    [
        pytest.param(
            [("transport", "streamable_http"), ("signed_uploads", [_entry()])],
            id="list",
        ),
        pytest.param(None, id="none"),
    ],
)
def test_no_signed_uploads_keeps_tool_list_for_non_mapping_connection(
    connection: Any,
) -> None:
    assert connection_signed_uploads(connection) == ()


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

    # No settings: every tool is loaded, identical to a direct build.
    plain = await load_mcp_tools_as_agent_tools({"docs": remote})
    assert [_describe(tool) for tool in plain.tools] == expected_all
    assert plain.failures == ()

    # Settings on the connection: the issuing tool is withheld.
    configured_connection = _connection(signed_uploads=[_entry()])
    configured = await load_mcp_tools_as_agent_tools({"docs": configured_connection})
    assert [_describe(tool) for tool in configured.tools] == [
        expected_all[0],
        expected_all[2],
    ]
    assert configured.failures == ()
    assert configured.loaded_servers == ("docs",)

    # Another connector on another host without settings: nothing is withheld.
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

    # A stdio connection never reads settings.
    stdio = _stdio_connection(signed_uploads=[_entry()])
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

    result = await load_mcp_tools_as_agent_tools(
        {"docs": _connection(signed_uploads=[_entry()])}
    )

    assert result.tools == ()
    assert result.failures == ()
    assert result.loaded_servers == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other_url",
    [
        pytest.param("https://other.example.test/mcp", id="other-host"),
        pytest.param(CONNECTION_URL, id="same-url"),
    ],
)
async def test_issuing_tool_hidden_only_on_its_connector(
    monkeypatch: pytest.MonkeyPatch, other_url: str
) -> None:
    _serve_tools(monkeypatch, *_ALL_TOOLS)
    connector_a = _connection(signed_uploads=[_entry()])
    connector_b = {"transport": "streamable_http", "url": other_url}

    result = await load_mcp_tools_as_agent_tools({"A": connector_a, "B": connector_b})

    names = [tool.name for tool in result.tools]
    assert names == [
        "mcp_A_attach_document",
        "mcp_A_other",
        "mcp_B_attach_document",
        "mcp_B_request_upload",
        "mcp_B_other",
    ]
    assert result.failures == ()
    assert result.loaded_servers == ("A", "B")


_MARKER = "planted_marker_tool"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connector_a",
    [
        pytest.param(
            _connection(signed_uploads={"tool_name": _MARKER}), id="not-a-list"
        ),
        pytest.param(
            _connection(signed_uploads=[_without("file_argument", tool_name=_MARKER)]),
            id="missing-key",
        ),
        pytest.param(
            _connection(
                signed_uploads=[
                    _entry(tool_name=_MARKER, allowed_upload_hosts=["10.0.0.1"])
                ]
            ),
            id="ip-upload-host",
        ),
        pytest.param(
            _connection(
                url="https:///mcp", signed_uploads=[_entry(issuing_tool_name=_MARKER)]
            ),
            id="url-without-hostname",
        ),
        pytest.param(
            _connection(url=7, signed_uploads=[_entry(issuing_tool_name=_MARKER)]),
            id="url-not-a-string",
        ),
        pytest.param(
            _connection(
                signed_uploads=[
                    _issuer_entry(
                        issuing_tool_name=_MARKER, allowed_upload_hosts=["10.0.0.1"]
                    )
                ]
            ),
            id="issuer-ip-upload-host",
        ),
        pytest.param(
            _connection(
                signed_uploads=[
                    _issuer_entry(issuing_tool_name=_MARKER),
                    _issuer_entry(issuing_tool_name=_MARKER),
                ]
            ),
            id="issuers-repeat-an-issuing-tool",
        ),
        pytest.param(
            _connection(
                url="https://[bad/mcp", signed_uploads=[_entry(tool_name=_MARKER)]
            ),
            id="unparseable-url",
        ),
    ],
)
async def test_invalid_settings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    connector_a: dict[str, Any],
) -> None:
    caplog.set_level("DEBUG")
    _serve_tools(monkeypatch, *_ALL_TOOLS)
    connector_b = _connection()
    assert "url" in connector_a

    result = await load_mcp_tools_as_agent_tools(
        {"A": copy.deepcopy(connector_a), "B": connector_b}
    )

    failures_of_a = [f for f in result.failures if f.server_name == "A"]
    assert len(failures_of_a) == 1
    assert failures_of_a[0].phase is MCPFailurePhase.INVALID_CONFIGURATION
    assert failures_of_a[0].error_type == "ValueError"
    assert not [tool for tool in result.tools if tool.name.startswith("mcp_A_")]
    assert "A" not in result.loaded_servers
    assert [tool.name for tool in result.tools] == [
        "mcp_B_attach_document",
        "mcp_B_request_upload",
        "mcp_B_other",
    ]
    assert "B" in result.loaded_servers
    assert _MARKER not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "served", [pytest.param(_ALL_TOOLS, id="tools"), pytest.param((), id="no-tools")]
)
async def test_invalid_settings_are_found_before_any_session(
    monkeypatch: pytest.MonkeyPatch, served: tuple[str, ...]
) -> None:
    opened = _serve_tools(monkeypatch, *served)

    result = await _load({"A": _connection(signed_uploads={"tool_name": "x"})})

    assert opened == []
    assert result.tools == ()
    assert result.loaded_servers == ()
    assert result.failures == (
        MCPServerLoadFailure(
            server_name="A",
            phase=MCPFailurePhase.INVALID_CONFIGURATION,
            error_type="ValueError",
            attempts=0,
        ),
    )
    assert mcp_load_failure_message(result.failures[0].phase) == (
        "MCP server upload settings are invalid."
    )


@pytest.mark.asyncio
async def test_a_bug_while_reading_settings_is_not_reported_as_invalid_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_connection: Any) -> Any:
        raise TypeError("planted")

    monkeypatch.setattr(mcp_adapter_module, "connection_signed_uploads", broken)

    result = await _load({"A": _connection()})

    assert [(failure.phase, failure.error_type) for failure in result.failures] == [
        (MCPFailurePhase.SESSION_START, "TypeError")
    ]


@pytest.mark.asyncio
async def test_settings_are_read_once_per_load(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_tools(monkeypatch, *_ALL_TOOLS)
    parse = signed_upload.parse_signed_upload_settings
    reads: list[Any] = []

    def counting_parse(raw: Any, *, connection_host: str | None) -> Any:
        reads.append(raw)
        return parse(raw, connection_host=connection_host)

    monkeypatch.setattr(signed_upload, "parse_signed_upload_settings", counting_parse)

    result = await _load({"docs": _connection(signed_uploads=[_entry()])})

    assert [tool.name for tool in result.tools] == [
        "mcp_docs_attach_document",
        "mcp_docs_other",
    ]
    assert len(reads) == 1


@pytest.mark.asyncio
async def test_loader_passes_each_tool_its_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_tools(monkeypatch, *_ALL_TOOLS)
    connection_a = _connection(signed_uploads=[_entry()])

    result = await _load({"A": connection_a, "B": _connection()})

    # The loader wraps each adapter in the approval gate; ``_target`` is the adapter.
    adapters = {tool.name: tool._target for tool in result.tools}
    assert "mcp_A_request_upload" not in adapters
    expected = _binding_for(connection_a, "attach_document")
    assert expected is not None
    assert adapters["mcp_A_attach_document"]._signed_upload == expected
    assert adapters["mcp_A_other"]._signed_upload is None
    assert adapters["mcp_B_attach_document"]._signed_upload is None


# ---------------------------------------------------------------------------
# No process-wide registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["_BINDINGS", "register_signed_upload_binding", "_REMOTE_TRANSPORTS"]
)
def test_no_process_registry(name: str) -> None:
    assert not hasattr(signed_upload, name)


# ---------------------------------------------------------------------------
# Issuer entries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param([_issuer_entry()], lambda: (_issuer(),), id="issuer-only"),
        pytest.param(
            [_entry(issuing_tool_name="issue_other"), _issuer_entry()],
            lambda: (_binding(issuing_tool_name="issue_other"), _issuer()),
            id="replacement-then-issuer",
        ),
        pytest.param(
            [_issuer_entry(), _entry(issuing_tool_name="issue_other")],
            lambda: (_issuer(), _binding(issuing_tool_name="issue_other")),
            id="issuer-then-replacement",
        ),
    ],
)
def test_parse_signed_upload_settings_accepts_issuer_entries(
    raw: Any, expected: Any
) -> None:
    given = copy.deepcopy(raw)

    result = _parse(raw)

    assert result == expected()
    assert [type(item) for item in result] == [type(item) for item in expected()]
    for item in result:
        assert isinstance(item.allowed_upload_hosts, frozenset)
        assert isinstance(item.content_types, MappingProxyType)
        assert item.connection_host == CONNECTION_HOST
    # What was kept is a copy: changing the stored value later changes nothing.
    for entry in raw:
        entry["content_types"][".txt"] = "text/plain"
        entry["allowed_upload_hosts"].append("later.example.test")
    assert result == _parse(given)
    assert result == expected()


_ISSUER_FIELD_PREFIXES = (
    "signed_uploads[0].issuing_tool_name",
    "signed_uploads[0].allowed_upload_hosts",
    "signed_uploads[0].upload_path_prefix",
    "signed_uploads[0].content_types",
)


def _issuer_rejected_cells() -> list[Any]:
    cells: list[Any] = []
    for cell in REJECTED_SETTINGS:
        raw, message = cell.values
        if not message.startswith(_ISSUER_FIELD_PREFIXES):
            continue
        assert isinstance(raw, list) and len(raw) == 1
        entry = {
            key: value
            for key, value in raw[0].items()
            if key not in ("tool_name", "file_argument")
        }
        cells.append(pytest.param([entry], message, id=cell.id))
    return cells


@pytest.mark.parametrize(("raw", "message"), _issuer_rejected_cells())
def test_parse_signed_upload_settings_rejects_issuer_entries(
    raw: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=_exactly(message)):
        _parse(raw)


def test_issuer_entry_rejection_cells_cover_every_issuer_field() -> None:
    covered: set[str] = set()
    for cell in _issuer_rejected_cells():
        _raw, message = cell.values
        covered.add(message.split(" ")[0].split(".")[1])

    assert covered == set(signed_upload._ISSUER_ENTRY_KEYS)


def _with_key(entry: dict[str, Any], key: str, value: Any = "x") -> dict[str, Any]:
    return {**entry, key: value}


def _without_issuer_key(key: str) -> dict[str, Any]:
    entry = _issuer_entry()
    del entry[key]
    return entry


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(_with_key(_issuer_entry(), "tool_name"), id="plus-tool-name"),
        pytest.param(
            _with_key(_issuer_entry(), "file_argument"), id="plus-file-argument"
        ),
        pytest.param(_with_key(_issuer_entry(), "extra"), id="plus-extra-key"),
        *[
            pytest.param(_without_issuer_key(key), id=f"missing-{key}")
            for key in _issuer_entry()
        ],
        pytest.param(
            {
                key: value
                for key, value in _entry().items()
                if key not in ("tool_name", "issuing_tool_name")
            },
            id="replacement-keys-minus-two",
        ),
    ],
)
def test_signed_upload_entry_key_sets(entry: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=_exactly(KEYS)):
        _parse([entry])


def test_signed_upload_entry_key_sets_are_tied_together() -> None:
    assert signed_upload._ISSUER_ENTRY_KEYS == (
        signed_upload._SETTINGS_ENTRY_KEYS - {"tool_name", "file_argument"}
    )
    assert set(_issuer_entry()) == set(signed_upload._ISSUER_ENTRY_KEYS)


def _mixed_entries(replacements: int, issuers: int) -> list[dict[str, Any]]:
    return [
        *[_entry(tool_name=f"tool_{index}") for index in range(replacements)],
        *[
            _issuer_entry(issuing_tool_name=f"issue_{index}")
            for index in range(issuers)
        ],
    ]


def _issuer_also_replacement(position: int) -> str:
    return (
        f"signed_uploads[{position}].issuing_tool_name "
        "is also the issuing tool of a replacement entry"
    )


CLASHING_ENTRIES = [
    pytest.param(
        [_entry(), _entry(issuing_tool_name="other_issuer")],
        "signed_uploads[1].tool_name repeats an earlier entry",
        id="rule1-two-replacements-same-tool",
    ),
    pytest.param(
        [_issuer_entry(), _issuer_entry()],
        "signed_uploads[1].issuing_tool_name repeats an earlier entry",
        id="rule2-two-issuers-same-issuing-tool",
    ),
    pytest.param(
        [_issuer_entry(), _entry()],
        _issuer_also_replacement(0),
        id="rule3-issuer-first",
    ),
    pytest.param(
        [_entry(), _issuer_entry()],
        _issuer_also_replacement(1),
        id="rule3-issuer-last",
    ),
    pytest.param(
        [_issuer_entry(issuing_tool_name="attach_document"), _entry()],
        CLASH,
        id="rule4-issuer-tool-is-a-replaced-tool",
    ),
    pytest.param(
        [_issuer_entry(), _entry(), _entry()],
        "signed_uploads[2].tool_name repeats an earlier entry",
        id="mixed-issuer-then-duplicate-replacements",
    ),
    pytest.param(
        [
            _entry(),
            _issuer_entry(issuing_tool_name="issue_a"),
            _issuer_entry(issuing_tool_name="issue_a"),
        ],
        "signed_uploads[2].issuing_tool_name repeats an earlier entry",
        id="mixed-replacement-then-duplicate-issuers",
    ),
    pytest.param(
        [
            _issuer_entry(issuing_tool_name="issue_a"),
            _issuer_entry(issuing_tool_name="issue_a"),
            _entry(),
            _entry(),
        ],
        "signed_uploads[3].tool_name repeats an earlier entry",
        id="rule1-is-checked-before-rule2",
    ),
    pytest.param(
        _mixed_entries(10, 11),
        "signed_uploads must not have more than 20 entries",
        id="twenty-one-entries",
    ),
]


@pytest.mark.parametrize(("raw", "message"), CLASHING_ENTRIES)
def test_signed_upload_entries_clash(raw: Any, message: str) -> None:
    with pytest.raises(ValueError, match=_exactly(message)):
        _parse(raw)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            [_issuer_entry(), _entry(issuing_tool_name="issue_other")],
            id="different-issuing-tools",
        ),
        pytest.param(_mixed_entries(10, 10), id="twenty-entries"),
    ],
)
def test_signed_upload_entries_accepts_compatible_entries(raw: Any) -> None:
    assert len(_parse(raw)) == len(raw)


# ---------------------------------------------------------------------------
# The issuing tool's expiry of the file id
# ---------------------------------------------------------------------------


def _issuing_reply(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "file_id": "rf_abcdefgh",
        "upload_url": "https://uploads.example.test/staged/abc?sig=1",
        "method": "PUT",
        "headers": {"Content-Type": "application/pdf"},
        "url_expires_at": "2030-01-01T00:10:00Z",
    }
    payload.update(overrides)
    payload = {k: v for k, v in payload.items() if v is not _MISSING}
    return {"structured_content": payload, "content": []}


_MISSING = object()
_UTC = timezone.utc


@pytest.mark.parametrize(
    ("expires_at", "expected"),
    [
        pytest.param(
            "2030-01-01T01:00:00Z",
            datetime(2030, 1, 1, 1, 0, tzinfo=_UTC),
            id="z-suffix",
        ),
        pytest.param(
            "2030-01-01T09:00:00+08:00",
            datetime(2030, 1, 1, 1, 0, tzinfo=_UTC),
            id="offset-converted-to-utc",
        ),
        pytest.param(_MISSING, None, id="missing"),
        pytest.param("x", None, id="not-iso"),
        pytest.param(7, None, id="not-a-string"),
        pytest.param(None, None, id="none"),
        pytest.param("2030-01-01T01:00:00", None, id="no-timezone"),
        pytest.param("9999-12-31T23:59:59-23:59", None, id="converts-out-of-range"),
        pytest.param("0001-01-01T00:00:00+01:00", None, id="converts-below-range"),
    ],
)
def test_parse_signed_upload_response_reads_file_id_expiry(
    expires_at: Any, expected: Any
) -> None:
    ticket = parse_signed_upload_response(
        _issuing_reply(expires_at=expires_at), size_bytes=3
    )

    assert ticket.file_id == "rf_abcdefgh"
    assert ticket.file_id_expires_at == expected
    if expected is not None:
        assert ticket.file_id_expires_at.tzinfo is _UTC
    # The expiry of the upload URL is read as before.
    assert ticket.expires_at == datetime(2030, 1, 1, 0, 10, tzinfo=_UTC)


# ---------------------------------------------------------------------------
# Loading issuer entries through the MCP adapter
# ---------------------------------------------------------------------------


def _gated_targets(result: Any) -> dict[str, Any]:
    # The loader wraps each adapter in the approval gate; ``_target`` is the adapter.
    return {tool.name: tool._target for tool in result.tools}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signed_uploads", "replaces_attach_document"),
    [
        pytest.param([_issuer_entry()], False, id="issuer-only"),
        pytest.param(
            [_issuer_entry(), _entry(issuing_tool_name="issue_other")],
            True,
            id="issuer-and-replacement-of-another-issuing-tool",
        ),
        pytest.param(None, False, id="no-settings"),
    ],
)
async def test_issuer_entry_replaces_issuing_tool_with_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    signed_uploads: Any,
    replaces_attach_document: bool,
) -> None:
    _serve_tools(monkeypatch, *_ALL_TOOLS)
    connection = _connection()
    if signed_uploads is not None:
        connection["signed_uploads"] = signed_uploads
    visibility = mcp_adapter_module.ToolVisibility.PUBLIC
    allow_users = ["user-42"]
    workspace = SimpleNamespace(marker="workspace")

    result = await load_mcp_tools_as_agent_tools(
        {"docs": connection},
        visibility=visibility,
        allow_users=allow_users,
        workspace=workspace,
    )

    assert result.failures == ()
    assert [tool.name for tool in result.tools] == [
        f"mcp_docs_{name}" for name in _ALL_TOOLS
    ]
    adapters = _gated_targets(result)
    issuing_tool = adapters["mcp_docs_request_upload"]
    if signed_uploads is None:
        assert type(issuing_tool) is mcp_adapter_module.MCPToolAdapter
    else:
        assert type(issuing_tool) is mcp_adapter_module.SignedUploadIssuerAdapter
    # The wrapper stands in the tool's place as that connector's own tool.
    ordinary = _build_mcp_tool_adapter(
        "docs",
        connection,
        _server_tool("request_upload"),
        visibility=visibility,
        allow_users=allow_users,
        workspace=workspace,
    )
    assert issuing_tool.name == ordinary.name == "mcp_docs_request_upload"
    assert issuing_tool.metadata.source_server == ordinary.metadata.source_server
    assert issuing_tool.metadata.source_server == normalize_mcp_server_name("docs")
    assert issuing_tool._visibility == ordinary._visibility == visibility
    assert issuing_tool._allow_users == ordinary._allow_users == allow_users
    assert issuing_tool._workspace is ordinary._workspace is workspace
    assert issuing_tool.concurrency_safe == ordinary.concurrency_safe
    # The other tools are unaffected.
    assert (
        adapters["mcp_docs_attach_document"]._signed_upload is not None
    ) is replaces_attach_document
    assert adapters["mcp_docs_other"]._signed_upload is None


@pytest.mark.asyncio
async def test_issuer_wrapper_is_admitted_by_a_server_scoped_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_tools(monkeypatch, *_ALL_TOOLS)

    result = await load_mcp_tools_as_agent_tools(
        {"docs": _connection(signed_uploads=[_issuer_entry()])}
    )

    spec = ToolSelectionSpec.from_raw(tool_categories=["mcp:docs"])
    allowed = spec.compute_allowed_names(list(_gated_targets(result).values()))
    assert allowed is not None
    assert "mcp_docs_request_upload" in allowed
    other_spec = ToolSelectionSpec.from_raw(tool_categories=["mcp:elsewhere"])
    other_allowed = other_spec.compute_allowed_names(
        list(_gated_targets(result).values())
    )
    assert other_allowed is not None
    assert "mcp_docs_request_upload" not in other_allowed


@pytest.mark.asyncio
async def test_issuer_wrapper_needs_the_listed_tool(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    _serve_tools(monkeypatch, "attach_document", "other")

    result = await load_mcp_tools_as_agent_tools(
        {"docs": _connection(signed_uploads=[_issuer_entry()])},
        allow_users=["user-42"],
    )

    assert [tool.name for tool in result.tools] == [
        "mcp_docs_attach_document",
        "mcp_docs_other",
    ]
    assert result.failures == ()
    assert result.loaded_servers == ("docs",)
    warnings = [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert len(warnings) == 1
    assert warnings[0].args == ("docs", "request_upload")
    assert "user-42" not in caplog.text


@pytest.mark.asyncio
async def test_server_with_only_a_taken_over_issuing_tool_loads_the_wrapper(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    _serve_tools(monkeypatch, "request_upload")

    result = await load_mcp_tools_as_agent_tools(
        {"docs": _connection(signed_uploads=[_issuer_entry()])}
    )

    assert [tool.name for tool in result.tools] == ["mcp_docs_request_upload"]
    assert (
        type(_gated_targets(result)["mcp_docs_request_upload"])
        is mcp_adapter_module.SignedUploadIssuerAdapter
    )
    assert result.loaded_servers == ("docs",)
    assert result.failures == ()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_hide_only_entry_hides_the_issuing_tool_without_a_wrapper(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    _serve_tools(monkeypatch, "request_upload", "other")
    hide_only = _entry(tool_name="unlisted_tool", file_argument="document_file_id")

    result = await load_mcp_tools_as_agent_tools(
        {"docs": _connection(signed_uploads=[hide_only])}
    )

    assert [tool.name for tool in result.tools] == ["mcp_docs_other"]
    assert _gated_targets(result)["mcp_docs_other"]._signed_upload is None
    assert result.failures == ()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

"""Request URLs are logged without credential-bearing query values."""

import io
import json
import logging
import re
from collections.abc import Iterator
from logging.config import dictConfig
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xagent.core.utils import security
from xagent.web import logging_config
from xagent.web.app import global_exception_handler, validation_exception_handler
from xagent.web.logging_config import setup_logging
from xagent.web.request_log_redaction import (
    REDACTED,
    RequestTargetRedactionFilter,
    install_request_log_redaction,
    redact_request_target,
)

SECRET = "guest.jwt.SECRET-VALUE"
_JWT_SAMPLE = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJl"


@pytest.fixture
def restore_logging_state() -> Iterator[None]:
    """Snapshot every logger's level/handlers/filters and restore them after.

    ``setup_logging(force=True)`` and ``dictConfig`` rewrite global logging
    state (including resetting child loggers of configured names), and the
    process-level record factory is global too, so all of it is restored.
    """
    manager = logging.Logger.manager
    loggers = [logging.getLogger()] + [
        logger
        for logger in list(manager.loggerDict.values())
        if isinstance(logger, logging.Logger)
    ]
    snapshot = [
        (
            logger,
            logger.level,
            list(logger.handlers),
            list(logger.filters),
            logger.propagate,
            logger.disabled,
        )
        for logger in loggers
    ]
    record_factory = logging.getLogRecordFactory()
    was_applied = logging_config._is_applied
    try:
        yield
    finally:
        for logger, level, handlers, filters, propagate, disabled in snapshot:
            logger.setLevel(level)
            logger.handlers[:] = handlers
            logger.filters[:] = filters
            logger.propagate = propagate
            logger.disabled = disabled
        logging.setLogRecordFactory(record_factory)
        logging_config._is_applied = was_applied


@pytest.mark.parametrize(
    "target, expected",
    [
        ("/api/files/public/preview/abc", "/api/files/public/preview/abc"),
        (
            f"/api/widget/chat/ws/1?token={SECRET}",
            f"/api/widget/chat/ws/1?token={REDACTED}",
        ),
        (
            f"/api/files/public/preview/abc?relative_path=a.png&ticket={SECRET}",
            f"/api/files/public/preview/abc?relative_path=a.png&ticket={REDACTED}",
        ),
        (
            f"/widget/chat/x?guest_id=g1&widget_key={SECRET}&embed_ticket={SECRET}",
            f"/widget/chat/x?guest_id=g1&widget_key={REDACTED}&embed_ticket={REDACTED}",
        ),
        (
            f"/cb?code={SECRET}&state={SECRET}&access_token={SECRET}",
            f"/cb?code={REDACTED}&state={REDACTED}&access_token={REDACTED}",
        ),
        # Composio's verification trip and its denial redirect (#3532).
        (
            "/api/auth/meta/callback?session_uri=https%3A%2F%2Fbackend.composio.dev"
            "%2Fverify%2Fsess_1",
            f"/api/auth/meta/callback?session_uri={REDACTED}",
        ),
        (
            "/api/auth/meta/callback?status=failed&error=access_denied"
            "&connected_account_id=ca_1",
            "/api/auth/meta/callback?status=failed&error=access_denied"
            f"&connected_account_id={REDACTED}",
        ),
        (
            f"https://host.example/a?api_key={SECRET}&page=2",
            f"https://host.example/a?api_key={REDACTED}&page=2",
        ),
        # Signed-URL and pre-authenticated download spellings share the
        # unified sensitive-name list.
        (
            f"/download.aspx?tempauth={SECRET}&sig={SECRET}&X-Amz-Signature={SECRET}",
            f"/download.aspx?tempauth={REDACTED}&sig={REDACTED}"
            f"&X-Amz-Signature={REDACTED}",
        ),
        # Every name the previous request-target filter masked by suffix is
        # still masked; a paging cursor may be over-masked in a log line, an
        # access token may not be under-masked.
        (
            f"/a?page_token={SECRET}&next_token={SECRET}&primary_key={SECRET}"
            f"&sort_key={SECRET}&cache_key={SECRET}&public_key={SECRET}"
            f"&s3_key={SECRET}&idempotency_key={SECRET}",
            f"/a?page_token={REDACTED}&next_token={REDACTED}"
            f"&primary_key={REDACTED}&sort_key={REDACTED}"
            f"&cache_key={REDACTED}&public_key={REDACTED}&s3_key={REDACTED}"
            f"&idempotency_key={REDACTED}",
        ),
        # Names that are clearly not credentials stay readable.
        (
            "/a?next_page=2&guest_id=g1&page_size=10",
            "/a?next_page=2&guest_id=g1&page_size=10",
        ),
        ("/a?token=", "/a?token="),
        ("/a?validation_only=true", "/a?validation_only=true"),
        # Nothing sensitive: returned byte-for-byte, never re-serialized.
        ("/search?q=a+b&flag&x=%5B1%5D", "/search?q=a+b&flag&x=%5B1%5D"),
    ],
)
def test_redact_request_target(target, expected):
    assert redact_request_target(target) == expected


def test_access_log_record_args_are_redacted():
    # uvicorn.access: '%s - "%s %s HTTP/%s" %d' with the raw path+query.
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1", "GET", f"/api/share/chat/ws/1?token={SECRET}", "1.1", 101),
        None,
    )
    assert RequestTargetRedactionFilter().filter(record) is True
    assert SECRET not in record.getMessage()
    assert f"?token={REDACTED}" in record.getMessage()


def test_access_log_record_redacts_an_encoded_composio_session_uri():
    # The verifier URI arrives percent-encoded in the callback's query.
    session_uri = "https%3A%2F%2Fbackend.composio.dev%2Fverify%2Fsess_SECRET"
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        (
            "127.0.0.1:1",
            "GET",
            f"/api/auth/meta/callback?session_uri={session_uri}",
            "1.1",
            200,
        ),
        None,
    )
    assert RequestTargetRedactionFilter().filter(record) is True
    message = record.getMessage()
    assert "sess_SECRET" not in message
    assert "backend.composio.dev" not in message
    assert f"?session_uri={REDACTED}" in message


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_keeps_uvicorn_access_formatter_working() -> None:
    # uvicorn's AccessFormatter unpacks record.args into five fields and
    # ignores record.msg. A factory that rewrites the message and clears the
    # arguments makes every access line fail with "not enough values to
    # unpack", so a credential-bearing request disappears from the log.
    # ``token`` is in the query on purpose: the previous text redactor saw it
    # in the interpolated message and flattened the record, so this case is a
    # real regression test for the access formatter, not just for the filter.
    import uvicorn.logging

    setup_logging(force=True)
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(logging.INFO)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        uvicorn.logging.AccessFormatter(
            '%(client_addr)s - "%(request_line)s" %(status_code)s',
            use_colors=False,
        )
    )
    access_logger.addHandler(handler)
    try:
        access_logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:1",
            "GET",
            f"/api/files/public/preview/abc"
            f"?ticket={SECRET}&code={SECRET}&token={SECRET}",
            "1.1",
            200,
        )
    finally:
        access_logger.removeHandler(handler)

    output = stream.getvalue()
    assert SECRET not in output
    assert (
        f'127.0.0.1:1 - "GET /api/files/public/preview/abc'
        f'?ticket={REDACTED}&code={REDACTED}&token={REDACTED} HTTP/1.1" 200 OK'
    ) in output


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_does_not_trust_the_access_logger_by_name_alone() -> None:
    # Only the five-tuple uvicorn's AccessFormatter unpacks owns the argument
    # shape. An ordinary record on the same logger must be formatted like any
    # other, or a plain ``%(message)s`` handler prints the credential.
    setup_logging(force=True)
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    access_logger.addHandler(handler)
    records, capture_handler = _capture(access_logger)
    try:
        access_logger.info("Authorization: Bearer %s", SECRET)
        access_logger.info(f"Authorization: Bearer {SECRET}")
    finally:
        access_logger.removeHandler(handler)
        access_logger.removeHandler(capture_handler)

    output = stream.getvalue()
    assert SECRET not in output
    assert all(record.args == () for record in records)
    assert all(SECRET not in record.getMessage() for record in records)


@pytest.mark.usefixtures("restore_logging_state")
def test_server_loggers_keep_the_filter_across_logging_reconfiguration():
    setup_logging(force=True)
    # The ASGI server applies its own dictConfig after ours.
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "loggers": {
                "uvicorn.access": {"level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
            },
        }
    )
    install_request_log_redaction()  # idempotent
    for name in ("uvicorn.access", "uvicorn.error"):
        filters = [
            f
            for f in logging.getLogger(name).filters
            if isinstance(f, RequestTargetRedactionFilter)
        ]
        assert len(filters) == 1

    # The server attaches its own non-propagating handlers; capture the same way.
    records: list[logging.LogRecord] = []
    capture = logging.Handler()
    capture.emit = records.append  # type: ignore[method-assign]
    server_logger = logging.getLogger("uvicorn.error")
    server_logger.addHandler(capture)
    server_logger.setLevel(logging.INFO)
    server_logger.info(
        '%s - "WebSocket %s" [accepted]',
        "127.0.0.1:1",
        f"/api/widget/chat/ws/1?token={SECRET}",
    )
    message = records[-1].getMessage()
    assert SECRET not in message
    assert f"/api/widget/chat/ws/1?token={REDACTED}" in message


def _app_with_handlers() -> FastAPI:
    app = FastAPI()
    from fastapi.exceptions import RequestValidationError

    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    @app.get("/v1/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    @app.get("/api/typed")
    def typed(count: int) -> dict[str, int]:
        return {"count": count}

    return app


def test_exception_handlers_log_the_path_not_the_query(caplog):
    client = TestClient(_app_with_handlers(), raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="xagent.web.app"):
        assert client.get(f"/v1/boom?token={SECRET}").status_code == 500
        # The rejected input value itself is a potential secret too.
        assert (
            client.get(f"/api/typed?count={SECRET}&ticket={SECRET}").status_code == 422
        )
    assert SECRET not in caplog.text
    assert "GET /v1/boom" in caplog.text
    assert "GET /api/typed" in caplog.text


@pytest.mark.parametrize(
    "direction, name, value, expected",
    [
        (
            "<",
            "Sec-WebSocket-Protocol",
            f"xagent-auth-v1, xagent-auth-token.{SECRET}",
            f"xagent-auth-v1, xagent-auth-token.{REDACTED}",
        ),
        ("<", "Authorization", f"Bearer {SECRET}", f"Bearer {REDACTED}"),
        ("<", "authorization", SECRET, REDACTED),
        ("<", "Proxy-Authorization", f"Basic {SECRET}", f"Basic {REDACTED}"),
        ("<", "Cookie", f"session={SECRET}", REDACTED),
        ("<", "X-Api-Key", SECRET, REDACTED),
        ("<", "X-Auth-Token", SECRET, REDACTED),
        ("<", "Sec-WebSocket-Protocol", "xagent-auth-v1", "xagent-auth-v1"),
        ("<", "Sec-WebSocket-Key", "dGhlIHNhbXBsZQ==", "dGhlIHNhbXBsZQ=="),
        ("<", "Host", "localhost:8000", "localhost:8000"),
        ("<", "Origin", "https://example.com", "https://example.com"),
        (">", "Sec-WebSocket-Protocol", "xagent-auth-v1", "xagent-auth-v1"),
        (">", "Set-Cookie", f"session={SECRET}", REDACTED),
    ],
)
def test_handshake_header_records_are_redacted(direction, name, value, expected):
    # websockets' legacy and Sans-I/O servers both log one debug record per
    # handshake header to the logger uvicorn passes them (uvicorn.error):
    # logger.debug("< %s: %s", name, value) for the request, "> %s: %s" for
    # the response.
    record = logging.LogRecord(
        "uvicorn.error",
        logging.DEBUG,
        __file__,
        1,
        f"{direction} %s: %s",
        (name, value),
        None,
    )
    assert RequestTargetRedactionFilter().filter(record) is True
    assert record.getMessage() == f"{direction} {name}: {expected}"


def _capture_debug_handshake(ws_impl: str, **connect_kwargs: object) -> list[str]:
    """Run one real WebSocket handshake through uvicorn at debug level."""
    import asyncio
    import threading
    import time

    import uvicorn
    from websockets.sync.client import connect

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] != "websocket":
            return
        await receive()
        offered = scope.get("subprotocols") or []
        await send(
            {
                "type": "websocket.accept",
                "subprotocol": "xagent-auth-v1" if offered else None,
            }
        )
        await send({"type": "websocket.close", "code": 1000})

    install_request_log_redaction()
    records: list[logging.LogRecord] = []
    capture = logging.Handler()
    capture.emit = records.append  # type: ignore[method-assign]
    server_logger = logging.getLogger("uvicorn.error")
    server_logger.addHandler(capture)

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        ws=ws_impl,
        log_level="debug",
        log_config=None,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.02)
        port = server.servers[0].sockets[0].getsockname()[1]
        with connect(f"ws://127.0.0.1:{port}/api/widget/chat/ws/1", **connect_kwargs):
            pass
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        server_logger.removeHandler(capture)
    return [record.getMessage() for record in records]


@pytest.mark.usefixtures("restore_logging_state")
@pytest.mark.parametrize("ws_impl", ["websockets", "websockets-sansio"])
@pytest.mark.parametrize(
    "connect_kwargs, carrier_line",
    [
        (
            {"subprotocols": ["xagent-auth-v1", f"xagent-auth-token.{SECRET}"]},
            f"< Sec-WebSocket-Protocol: xagent-auth-v1, xagent-auth-token.{REDACTED}",
        ),
        (
            {"additional_headers": {"Authorization": f"Bearer {SECRET}"}},
            f"< Authorization: Bearer {REDACTED}",
        ),
    ],
)
def test_debug_websocket_handshake_logs_no_header_credentials(
    ws_impl, connect_kwargs, carrier_line
):
    messages = _capture_debug_handshake(ws_impl, **connect_kwargs)
    assert messages, "the debug handshake logged nothing"
    assert not [m for m in messages if SECRET in m]
    # Header names are logged as the client spelled them.
    lowered = [m.lower() for m in messages]
    assert carrier_line.lower() in lowered
    # Safe diagnostics remain.
    assert any(m.startswith("< sec-websocket-key: ") for m in lowered)
    assert "< get /api/widget/chat/ws/1 http/1.1" in lowered


# ---------------------------------------------------------------------------
# One sensitive-name list for every log sink
# ---------------------------------------------------------------------------

_MASKED_QUERY_PARAMS = (
    "token",
    "refresh_token",
    "access_token",
    "tempauth",
    "sig",
    "X-Amz-Signature",
    "code",
    "client_secret",
    "ticket",
    # Suffix rule the previous request-target filter applied: the new name
    # rule is a union, so none of these may become readable again.
    "page_token",
    "next_token",
    "primary_key",
    "sort_key",
    "cache_key",
    "public_key",
    "s3_key",
    "idempotency_key",
)
_PRESERVED_QUERY_PARAMS = ("guest_id", "page_size", "next_page")


@pytest.mark.parametrize("name", _MASKED_QUERY_PARAMS)
def test_sensitive_query_values_are_masked(name: str) -> None:
    text = f"GET https://api.example.com/cb?{name}={SECRET}&page=2"

    for redact in (security.redact_sensitive_text, security.redact_log_text):
        redacted = redact(text)
        assert SECRET not in redacted
        assert f"{name}=" in redacted
        assert "page=2" in redacted


@pytest.mark.parametrize("name", _PRESERVED_QUERY_PARAMS)
def test_paging_and_non_secret_query_values_are_preserved(name: str) -> None:
    value = f"{name}-value-123"
    text = f"GET https://api.example.com/cb?{name}={value}&page=2"

    redacted = security.redact_sensitive_text(text)
    assert value in redacted


def test_code_is_only_sensitive_inside_a_url_query() -> None:
    assert not security.is_sensitive_param_name("code")
    assert security.is_sensitive_param_name("code", url_query=True)
    assert security.redact_sensitive_text("code=400") == "code=400"


# Suffix names the previous request-target filter accepted. The new name rule
# is a union of that list and the new one, so none of them may become readable
# again in a URL query.
_PREVIOUS_REQUEST_TARGET_SUFFIX_NAMES = (
    "widget_key",
    "share_token",
    "embed_ticket",
    "client_secret",
    "x_custom_signature",
    "x-api-key",
    "page_token",
    "next_token",
    "primary_key",
    "sort_key",
    "cache_key",
    "public_key",
    "s3_key",
    "idempotency_key",
)


def test_url_query_rule_keeps_the_previous_request_target_suffixes() -> None:
    for name in _PREVIOUS_REQUEST_TARGET_SUFFIX_NAMES:
        assert security.is_sensitive_param_name(name, url_query=True), name
    assert not security.is_sensitive_param_name("page_size", url_query=True)
    assert not security.is_sensitive_param_name("guest_id", url_query=True)


_AUTHORIZATION_HEADERS = (
    "Authorization: Bearer sk-secret-value",
    '{"Authorization": "Bearer sk-secret-value"}',
    "{'Authorization': 'Bearer sk-secret-value'}",
    # The quote can sit after the scheme rather than before it, and the value
    # character class must not stop at a bracket the value itself contains.
    'Authorization: Bearer "sk-secret-value"',
    "Authorization: Bearer 'sk-secret-value'",
    'Authorization: Basic "sk-secret-value"',
    "Authorization: Bearer sk-secret-value]tail",
    "Proxy-Authorization: Basic c2VjcmV0OnZhbHVl",
)


@pytest.mark.parametrize("header", _AUTHORIZATION_HEADERS)
def test_authorization_header_values_are_masked(header: str) -> None:
    # Both the text redactor (which carries AUTH_HEADER_PATTERN) and the
    # log-record redactor must accept the quoted dict/JSON spellings.
    for redact in (security.redact_sensitive_text, security.redact_log_text):
        redacted = redact(header)
        assert "sk-secret-value" not in redacted
        assert "c2VjcmV0OnZhbHVl" not in redacted


def test_quoted_authorization_value_keeps_its_quotes() -> None:
    assert (
        security.redact_log_text('{"Authorization": "Bearer sk-secret-value"}')
        == '{"Authorization": "Bearer [REDACTED]"}'
    )
    assert (
        security.redact_log_text('Authorization: Bearer "sk-secret-value"')
        == 'Authorization: Bearer "[REDACTED]"'
    )


def test_an_already_masked_authorization_value_is_not_masked_again() -> None:
    # An outer redaction layer (or an earlier pass) may already have produced
    # the marker; appending another ``]`` breaks exact log assertions.
    for text in (
        "Authorization: Bearer [REDACTED]",
        'Authorization: Bearer "[REDACTED]"',
        "Authorization: Basic ***c123",
    ):
        assert security.redact_log_text(text) == text


def test_jwt_shaped_values_are_masked_outside_a_url() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJl"
    text = f"tool output: {jwt} (end)"

    redacted = security.redact_log_text(text)

    assert jwt not in redacted
    assert "tool output:" in redacted
    assert "(end)" in redacted


def test_jwt_in_a_url_path_is_masked() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJl"

    redacted = security.redact_log_text(
        f"GET https://api.example.com/files/{jwt}/content"
    )

    assert jwt not in redacted
    assert "https://api.example.com/files/" in redacted
    assert "/content" in redacted


# The prefix characters the R2 review compared against the previous
# implementation, with the JWT placed directly behind each one.
_JWT_PREFIX_CHARS = (
    "",
    " ",
    "=",
    "-",
    "_",
    ".",
    "/",
    ":",
    ",",
    ";",
    "(",
    "[",
    "{",
    "<",
    '"',
    "'",
    "\\",
    "|",
    "+",
    "*",
    "~",
    "!",
    "?",
    "&",
    "%",
    "#",
    "@",
    "\n",
    "\t",
    "0",
    "a",
    "A",
    "$",
    "^",
)


@pytest.mark.parametrize(
    "prefix", _JWT_PREFIX_CHARS, ids=[repr(char) for char in _JWT_PREFIX_CHARS]
)
def test_a_jwt_is_recognized_after_each_prefix_character(prefix: str) -> None:
    text = f"x{prefix}{_JWT_SAMPLE}"

    redacted = security.redact_log_text(text)

    # A header opens a JWT when it starts a token run or follows a hyphen (the
    # boundary ``\b`` gave the previous implementation). A digit, letter or
    # underscore keeps the header inside the surrounding run, exactly as it did
    # before.
    inside_the_run = not prefix or (
        prefix.isascii() and (prefix.isalnum() or prefix == "_")
    )
    if inside_the_run:
        assert redacted == text
    else:
        assert _JWT_SAMPLE not in redacted
        assert redacted == f"x{prefix}{REDACTED}"


def test_an_already_masked_websocket_credential_is_not_masked_twice() -> None:
    # An outer layer can mask the secret but leave the version in front of it
    # (``v1.[REDACTED]``). Replacing the whole tail must not append a second
    # placeholder, which would break exact log assertions.
    already_masked = "xagent-auth-token.v1.[REDACTED]"
    for text in (
        already_masked,
        "xagent-auth-token.v1.[REDACTED][REDACTED]",
        "xagent-auth-token.[REDACTED]",
    ):
        once = security.redact_log_text(text)

        assert once == "xagent-auth-token.[REDACTED]"
        assert security.redact_log_text(once) == once


def test_xagent_auth_token_values_are_masked() -> None:
    text = "Sec-WebSocket-Protocol: xagent-auth-v1, xagent-auth-token.s3cr3t.va1ue"

    redacted = security.redact_log_text(text)

    assert "xagent-auth-token.s3cr3t.va1ue" not in redacted
    assert "xagent-auth-v1" in redacted


def test_log_text_redacts_a_relative_request_target_query() -> None:
    text = f"GET /api/files/public/preview/abc?code={SECRET}&ticket={SECRET} 404"

    redacted = security.redact_log_text(text)

    assert SECRET not in redacted
    assert (
        f"/api/files/public/preview/abc?code={REDACTED}&ticket={REDACTED}" in redacted
    )
    assert "404" in redacted


def test_log_text_keeps_url_placeholders_readable() -> None:
    text = f"GET https://api.example.com/cb?code={SECRET}&token={SECRET}"

    redacted = security.redact_log_text(text)

    assert "code=[REDACTED]" in redacted
    assert "token=[REDACTED]" in redacted
    assert "%5BREDACTED%5D" not in redacted


def test_a_rebuilt_url_keeps_an_existing_marker_readable() -> None:
    # One placeholder plus one still-sensitive value forces a rebuild; the
    # existing marker must not become percent-encoded on the way through.
    text = f"GET https://api.example.com/cb?token={REDACTED}&key={SECRET}"

    redacted = security.redact_log_text(text)

    assert "token=[REDACTED]" in redacted
    assert "key=[REDACTED]" in redacted
    assert "%5BREDACTED%5D" not in redacted


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_tolerates_a_query_value_that_cannot_be_encoded() -> None:
    # An unpaired surrogate in an ordinary query value used to raise from
    # urlencode at the logging call site. A URL with nothing to mask is
    # returned untouched before any percent-encoding happens.
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.unencodable_url")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    records, handler = _capture(logger)
    try:
        logger.warning("GET https://api.example.com/y?q=\ud800")
    finally:
        logger.removeHandler(handler)

    assert len(records) == 1
    assert "\ud800" in records[0].getMessage()


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_redacts_a_line_whose_query_holds_an_unpaired_surrogate() -> (
    None
):
    # A sensitive name without a value forces the URL rebuild even when the
    # query also holds an unpaired surrogate. The rebuild must encode the
    # surrogate instead of raising, otherwise the whole record -- including
    # the Authorization value on the same line -- would be emitted unredacted.
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.unencodable_sensitive_url")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    records, handler = _capture(logger)
    try:
        logger.warning(
            "Authorization: Bearer %s https://e.com/x?token&bad=\ud800",
            "sk-LEAK-0123456789",
        )
    finally:
        logger.removeHandler(handler)

    message = records[0].getMessage()
    assert "sk-LEAK-0123456789" not in message
    assert "Bearer [REDACTED]" in message
    assert "\ud800" not in message


def test_log_text_does_not_rewrite_a_url_without_sensitive_parameters() -> None:
    for text in (
        "GET https://api.example.com/cb?q=a+b&flag&x=%5B1%5D",
        "GET https://api.example.com/cb?q=a%20b",
        "see https://api.example.com/x?q=1).",
    ):
        assert security.redact_log_text(text) == text


_IDEMPOTENT_LOG_TEXTS = (
    "Authorization: Bearer [REDACTED]",
    'Authorization: Bearer "[REDACTED]"',
    "MCP_API_KEY=***c123 rejected",
    "token=<redacted>",
    f"GET https://api.example.com/cb?code={SECRET}&token={SECRET}",
    f"GET /api/files/public/preview/abc?code={SECRET}&ticket={SECRET}",
    f'{{"headers": {{"Authorization": "Bearer {SECRET}"}}}}',
)


@pytest.mark.parametrize("text", _IDEMPOTENT_LOG_TEXTS)
def test_log_text_redaction_is_idempotent(text: str) -> None:
    once = security.redact_log_text(text)

    assert security.redact_log_text(once) == once


# ---------------------------------------------------------------------------
# Recorded outputs of the pre-change implementation
# ---------------------------------------------------------------------------

# ``fixtures/legacy_redaction_golden.json`` records the inputs and the output
# the old redactors produced for every sample shape this branch must keep
# masking. The expected values come from ``ad253647`` (the last main commit
# before this branch):
# - ``security.redact_sensitive_text``
# - ``security.redact_url_credentials_for_logging``
# - ``request_log_redaction.redact_request_target`` (the request-log filter's
#   target masker)
# To regenerate, extract both files with ``git show ad253647:<path>`` into a
# temporary directory, load them with ``importlib`` and run the three functions
# over the recorded inputs; the one-off generator is intentionally not checked
# in, so never edit ``legacy_output`` by hand.
_GOLDEN_PATH = Path(__file__).with_name("fixtures") / "legacy_redaction_golden.json"
_LEGACY_GOLDEN: dict[str, list[dict[str, str]]] = json.loads(
    _GOLDEN_PATH.read_text(encoding="utf-8")
)

# A fragment is a maximal run of token characters. Comparing whole fragments
# keeps the assertion readable where the old helper retained a short suffix of
# the value (``***6789``): the full value is still gone even though four of its
# characters remain, and the new output must drop at least the same fragments.
_FRAGMENT_PATTERN = re.compile(r"[A-Za-z0-9._~%+-]+")


def _masked_fragments(original: str, redacted: str) -> set[str]:
    """Return token-sized pieces of ``original`` that ``redacted`` dropped."""
    return {
        fragment
        for fragment in _FRAGMENT_PATTERN.findall(original)
        if len(fragment) > 1 and fragment not in redacted
    }


def _golden_cases(section: str) -> list[tuple[str, str]]:
    return [
        (record["input"], record["legacy_output"]) for record in _LEGACY_GOLDEN[section]
    ]


def _golden_ids(section: str) -> list[str]:
    ids = []
    for index, record in enumerate(_LEGACY_GOLDEN[section]):
        slug = re.sub(r"[^a-z0-9]+", "-", record["input"].lower()).strip("-")
        ids.append(f"{index:02d}-{slug[:60]}")
    return ids


_TEXT_CASES = _golden_cases("redact_sensitive_text")
_URL_CASES = _golden_cases("redact_url_credentials_for_logging")
_REQUEST_TARGET_CASES = _golden_cases("redact_request_target")


def _assert_never_weaker(text: str, legacy_output: str, new_output: str) -> None:
    # Only compare what the old implementation actually masked: masking more
    # than it did is fine (a log may over-mask), masking less is the bug.
    legacy_masked = _masked_fragments(text, legacy_output)
    assert legacy_masked, f"sample does not exercise the legacy redactor: {text!r}"
    # Assert the same thing directly for every original secret substring the
    # old implementation masked: the failure then names the retained value
    # instead of only the fragment-set difference.
    retained = sorted(fragment for fragment in legacy_masked if fragment in new_output)
    assert not retained, (
        text,
        legacy_output,
        new_output,
        retained,
    )
    new_masked = _masked_fragments(text, new_output)
    assert legacy_masked <= new_masked, (
        text,
        legacy_output,
        new_output,
        sorted(legacy_masked - new_masked),
    )


@pytest.mark.parametrize(
    ("text", "legacy_output"),
    _TEXT_CASES,
    ids=_golden_ids("redact_sensitive_text"),
)
def test_text_redaction_is_never_weaker_than_ad253647(
    text: str, legacy_output: str
) -> None:
    _assert_never_weaker(text, legacy_output, security.redact_sensitive_text(text))


@pytest.mark.parametrize(
    ("text", "legacy_output"),
    _TEXT_CASES,
    ids=_golden_ids("redact_sensitive_text"),
)
def test_log_text_redaction_is_never_weaker_than_ad253647(
    text: str, legacy_output: str
) -> None:
    # ``redact_log_text`` replaces values outright, but must still mask every
    # value the older text redactor masked (with a short suffix or not).
    _assert_never_weaker(text, legacy_output, security.redact_log_text(text))


@pytest.mark.parametrize(
    ("text", "legacy_output"),
    _URL_CASES,
    ids=_golden_ids("redact_url_credentials_for_logging"),
)
def test_url_redaction_is_never_weaker_than_ad253647(
    text: str, legacy_output: str
) -> None:
    _assert_never_weaker(
        text,
        legacy_output,
        security.redact_url_credentials_for_logging(text),
    )


@pytest.mark.parametrize(
    ("text", "legacy_output"),
    _REQUEST_TARGET_CASES,
    ids=_golden_ids("redact_request_target"),
)
def test_request_target_redaction_is_never_weaker_than_ad253647(
    text: str, legacy_output: str
) -> None:
    _assert_never_weaker(text, legacy_output, redact_request_target(text))


# The pre-change log redactor treated every hyphen as a token boundary, so a
# JWT after ``prefix-``, inside a hyphenated path segment, or before a trailing
# hyphen was masked. These contexts pin that each of those shapes stays masked.
_JWT_CONTEXTS = (
    pytest.param(f"tool output: {_JWT_SAMPLE} (end)", id="bare"),
    pytest.param(
        f"GET https://api.example.com/files/{_JWT_SAMPLE}/content",
        id="url-path",
    ),
    pytest.param(f"prefix-{_JWT_SAMPLE}", id="after-hyphen"),
    pytest.param(
        f"GET /files/upload-{_JWT_SAMPLE} failed",
        id="hyphenated-upload-segment",
    ),
    pytest.param(f"/files/slug-{_JWT_SAMPLE}", id="hyphenated-path"),
    pytest.param(f"{_JWT_SAMPLE}-trailing", id="before-hyphen"),
    pytest.param(f"first {_JWT_SAMPLE} then {_JWT_SAMPLE}", id="two-values"),
)


@pytest.mark.parametrize("text", _JWT_CONTEXTS)
def test_a_jwt_is_masked_in_each_context(text: str) -> None:
    assert _JWT_SAMPLE not in security.redact_log_text(text)


# ---------------------------------------------------------------------------
# Process-level record redaction
# ---------------------------------------------------------------------------


def _capture(logger: logging.Logger) -> tuple[list[logging.LogRecord], logging.Handler]:
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    return records, handler


class _ExplodingMessage:
    def __str__(self) -> str:
        raise ValueError("message cannot be formatted")


_BAD_LOG_CALLS = [
    pytest.param("%s %s", (1,), id="too-few-arguments"),
    pytest.param("%(a)s %(b)s", ({"a": 1},), id="missing-mapping-key"),
    pytest.param(_ExplodingMessage(), (), id="exploding-str"),
]


@pytest.mark.usefixtures("restore_logging_state")
@pytest.mark.parametrize("message, args", _BAD_LOG_CALLS)
def test_record_factory_leaves_unformattable_records_to_handle_error(
    message: object, args: tuple[object, ...], capsys
) -> None:
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.unformattable")
    logger.setLevel(logging.INFO)

    # Exactly what logging does without this factory: the call returns and
    # the handler reports the bad record on stderr through handleError.
    logger.info(message, *args)

    assert "--- Logging error ---" in capsys.readouterr().err


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_falls_back_to_pattern_only_when_the_redactor_raises(
    monkeypatch, capsys
) -> None:
    setup_logging(force=True)

    def explode(text: str) -> str:
        raise RuntimeError("redactor exploded")

    monkeypatch.setattr(logging_config, "redact_log_text", explode)
    logger = logging.getLogger("xagent.tests.redactor_failure")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.warning("token=%s", SECRET)
    finally:
        logger.removeHandler(handler)

    # The full pass failed, so the fallback masks the assignment and writes
    # the message flat; the logging call still returns and emits one line.
    record = records[0]
    assert record.getMessage() == f"token={REDACTED}"
    assert record.msg == f"token={REDACTED}"
    assert record.args == ()
    assert getattr(record, "_xagent_log_redacted", False)
    assert "redactor exploded" not in capsys.readouterr().err


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_returns_the_record_when_the_fallback_also_raises(
    monkeypatch, capsys
) -> None:
    setup_logging(force=True)

    def explode(text: str) -> str:
        raise RuntimeError("redactor exploded")

    monkeypatch.setattr(logging_config, "redact_log_text", explode)
    monkeypatch.setattr(logging_config, "redact_log_text_pattern_only", explode)
    logger = logging.getLogger("xagent.tests.fallback_failure")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.warning("token=%s", SECRET)
    finally:
        logger.removeHandler(handler)

    # Both passes failed: the record is returned unchanged (and unmarked), so
    # logging behaves as it would without a factory.
    assert [record.getMessage() for record in records] == [f"token={SECRET}"]
    assert not getattr(records[0], "_xagent_log_redacted", False)
    assert "redactor exploded" not in capsys.readouterr().err


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_keeps_argument_structure_when_each_secret_is_in_an_argument():
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.structured_args")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.warning("headers: %s", f"Authorization: Bearer {SECRET}")
        logger.warning("count=%d event=%s", 3, f"Authorization: Bearer {SECRET}")
        logger.warning("event=%(event)s", {"event": f"Authorization: Bearer {SECRET}"})
    finally:
        logger.removeHandler(handler)

    first, second, third = records
    assert first.msg == "headers: %s"
    assert first.args == (f"Authorization: Bearer {REDACTED}",)
    assert second.msg == "count=%d event=%s"
    assert second.args == (3, f"Authorization: Bearer {REDACTED}")
    assert third.msg == "event=%(event)s"
    assert third.args == {"event": f"Authorization: Bearer {REDACTED}"}
    assert all(SECRET not in record.getMessage() for record in records)


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_flattens_only_when_the_secret_is_formed_by_interpolation():
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.flattened_args")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.warning("token=%s", SECRET)
    finally:
        logger.removeHandler(handler)

    record = records[0]
    assert record.msg == f"token={REDACTED}"
    assert record.args == ()


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_keeps_exc_text_unset_when_nothing_needs_redaction() -> None:
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.custom_exception")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    class CustomFormatter(logging.Formatter):
        def formatException(self, ei: object) -> str:
            return "CUSTOM-TRACEBACK"

    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(CustomFormatter("%(message)s"))
    logger.addHandler(stream_handler)
    records, capture_handler = _capture(logger)
    try:
        try:
            raise RuntimeError("plain failure")
        except RuntimeError:
            logger.exception("request failed")
    finally:
        logger.removeHandler(stream_handler)
        logger.removeHandler(capture_handler)

    # The factory left exc_text unset, so the handler's own formatException
    # filled it in: the default traceback text (which a pre-formatted
    # exc_text would have frozen) never appears.
    assert records[0].exc_text == "CUSTOM-TRACEBACK"
    assert "Traceback" not in records[0].exc_text
    assert "CUSTOM-TRACEBACK" in stream.getvalue()


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_leaves_an_existing_short_mask_alone() -> None:
    # An earlier layer may already have applied the text redactor's
    # ``***suffix`` mask; a second pass must not eat part of the suffix.
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.prefix_mask")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.warning("MCP_API_KEY=***c123 rejected")
    finally:
        logger.removeHandler(handler)

    assert records[0].getMessage() == "MCP_API_KEY=***c123 rejected"


@pytest.mark.usefixtures("restore_logging_state")
@pytest.mark.parametrize("name", ["httpx", "httpcore", "httpcore.connection"])
def test_transport_info_lines_stay_out_of_the_log_after_a_level_change(
    name: str,
) -> None:
    # Create the child before setup so the setup scan attaches the filter;
    # the point of the last case is that the filter alone drops the INFO
    # record, not the inherited WARNING level.
    logging.getLogger(name)
    setup_logging(force=True)
    if "." in name:
        logging.getLogger(name.split(".", 1)[0]).setLevel(logging.NOTSET)
    logger = logging.getLogger(name)
    # A third-party SDK constructing its own httpx client does exactly this.
    logger.setLevel(logging.NOTSET)
    # Level alone would let the INFO record through; only the filter drops it.
    assert logger.isEnabledFor(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.info(
            "HTTP Request: POST https://oauth.example.com/token?refresh_token=%s",
            SECRET,
        )
        logger.warning(
            "HTTP Request: POST https://oauth.example.com/token?refresh_token=%s",
            SECRET,
        )
    finally:
        logger.removeHandler(handler)

    assert [record.levelno for record in records] == [logging.WARNING]
    message = records[0].getMessage()
    assert SECRET not in message
    assert "https://oauth.example.com/token" in message


@pytest.mark.usefixtures("restore_logging_state")
def test_a_transport_child_created_after_setup_is_still_text_redacted() -> None:
    setup_logging(force=True)
    logging.getLogger("httpcore").setLevel(logging.NOTSET)
    late_child = logging.getLogger("httpcore.created_after_setup")
    late_child.setLevel(logging.NOTSET)
    records, handler = _capture(late_child)
    try:
        late_child.info(
            "HTTP Request: POST https://oauth.example.com/token?refresh_token=%s",
            SECRET,
        )
    finally:
        late_child.removeHandler(handler)

    # No logger-level filter was installed on this child (it did not exist at
    # setup), so the INFO line is emitted; the record factory is the fallback
    # that keeps the credential out of it.
    assert len(records) == 1
    assert SECRET not in records[0].getMessage()


@pytest.mark.usefixtures("restore_logging_state")
def test_repeated_setup_logging_attaches_the_filter_to_a_late_child() -> None:
    setup_logging(force=True)
    late_child = logging.getLogger("httpcore.late_filter_child")
    assert logging_config._TRANSPORT_FILTER not in late_child.filters

    # A repeated non-forced setup (the app import path) re-scans the logger
    # registry, so children created since the first call get the filter.
    setup_logging()

    assert logging_config._TRANSPORT_FILTER in late_child.filters


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_redacts_fstring_and_argument_messages() -> None:
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.redaction")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.warning(f"Authorization: Bearer {SECRET}")
        logger.warning("token=%s", SECRET)
    finally:
        logger.removeHandler(handler)

    messages = [record.getMessage() for record in records]
    assert len(messages) == 2
    assert all(SECRET not in message for message in messages)
    assert all("[REDACTED]" in message for message in messages)
    # The formatted message is final; stale arguments must not be
    # interpolated a second time by a handler's formatter.
    assert all(record.args == () for record in records)


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_redacts_exception_text() -> None:
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.redaction_exception")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        try:
            raise RuntimeError(f"Authorization: Bearer {SECRET}")
        except RuntimeError:
            logger.exception("request failed")
    finally:
        logger.removeHandler(handler)

    record = records[-1]
    assert record.getMessage() == "request failed"
    assert record.exc_text is not None
    assert SECRET not in record.exc_text
    assert "Traceback" in record.exc_text


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_redacts_stack_info() -> None:
    setup_logging(force=True)
    factory = logging.getLogRecordFactory()
    record = factory(
        "xagent.tests.stack",
        logging.WARNING,
        __file__,
        1,
        "task failed",
        (),
        None,
        None,
        f"Stack (most recent call last):\n  Authorization: Bearer {SECRET}",
    )

    assert record.stack_info is not None
    assert SECRET not in record.stack_info


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_reaches_a_non_propagating_logger_with_its_own_handler() -> None:
    setup_logging(force=True)
    logger = logging.getLogger("xagent.tests.own_handler")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.warning("token=%s", SECRET)
    finally:
        logger.removeHandler(handler)

    assert len(records) == 1
    assert SECRET not in records[0].getMessage()


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_survives_basic_config_force() -> None:
    setup_logging(force=True)
    # Replaces the handlers setup_logging installed; the record factory is
    # untouched, so the next record is still redacted.
    logging.basicConfig(force=True, level=logging.INFO)
    records, handler = _capture(logging.getLogger())
    try:
        logging.getLogger("xagent.tests.basic_config").warning("token=%s", SECRET)
    finally:
        logging.getLogger().removeHandler(handler)

    assert len(records) == 1
    assert SECRET not in records[0].getMessage()


@pytest.mark.usefixtures("restore_logging_state")
def test_repeated_setup_logging_does_not_stack_redaction_layers() -> None:
    setup_logging(force=True)
    first_factory = logging.getLogRecordFactory()
    first_filters = list(logging.getLogger("httpx").filters)

    setup_logging(force=True)

    assert logging.getLogRecordFactory() is first_factory
    assert logging.getLogger("httpx").filters == first_filters

    logger = logging.getLogger("xagent.tests.repeat")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.warning("Authorization: Bearer %s", SECRET)
    finally:
        logger.removeHandler(handler)
    assert SECRET not in records[0].getMessage()


@pytest.mark.usefixtures("restore_logging_state")
def test_record_factory_chains_with_a_preinstalled_factory() -> None:
    base = logging.getLogRecordFactory()

    def outer(*args, **kwargs):
        record = base(*args, **kwargs)
        record.msg = f"outer: {record.msg}"
        return record

    logging.setLogRecordFactory(outer)
    setup_logging(force=True)

    logger = logging.getLogger("xagent.tests.chained")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)
    try:
        logger.warning("token=%s", SECRET)
    finally:
        logger.removeHandler(handler)

    message = records[0].getMessage()
    assert message.startswith("outer: ")
    assert SECRET not in message


@pytest.mark.asyncio
@pytest.mark.usefixtures("restore_logging_state")
async def test_console_trace_payload_is_redacted_in_info_output() -> None:
    from xagent.core.agent.trace import (
        ConsoleTraceHandler,
        TraceAction,
        TraceCategory,
        TraceEvent,
        TraceEventType,
        TraceScope,
    )

    setup_logging(force=True)
    logger = logging.getLogger("xagent.core.agent.trace")
    logger.setLevel(logging.INFO)
    records, handler = _capture(logger)

    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJl"
    url = (
        "https://helpdesk.example.com/inline/attachment"
        f"?token={jwt}&tempauth=one-time-token"
    )
    event = TraceEvent(
        TraceEventType(TraceScope.ACTION, TraceAction.INFO, TraceCategory.GENERAL),
        data=f"Fetched attachment {url} for ticket 42",
        step_id="s1",
    )
    try:
        await ConsoleTraceHandler().handle_event(event)
    finally:
        logger.removeHandler(handler)

    assert len(records) == 1
    message = records[0].getMessage()
    assert jwt not in message
    assert "one-time-token" not in message
    # The rest of the payload is still there for debugging.
    assert "https://helpdesk.example.com/inline/attachment" in message
    assert "Fetched attachment" in message
    assert "for ticket 42" in message

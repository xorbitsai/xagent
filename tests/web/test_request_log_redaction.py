"""Request URLs are logged without credential-bearing query values."""

import logging
from collections.abc import Iterator
from logging.config import dictConfig

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
        # Paging cursors and ordinary parameters stay readable.
        (
            "/a?next_token=cursor-1&guest_id=g1&page_size=10",
            "/a?next_token=cursor-1&guest_id=g1&page_size=10",
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
)
_PRESERVED_QUERY_PARAMS = ("next_token", "guest_id", "page_size")


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


_AUTHORIZATION_HEADERS = (
    "Authorization: Bearer sk-secret-value",
    '{"Authorization": "Bearer sk-secret-value"}',
    "{'Authorization': 'Bearer sk-secret-value'}",
    "Authorization: Basic c2VjcmV0OnZhbHVl",
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


def test_xagent_auth_token_values_are_masked() -> None:
    text = "Sec-WebSocket-Protocol: xagent-auth-v1, xagent-auth-token.s3cr3t.va1ue"

    redacted = security.redact_log_text(text)

    assert "xagent-auth-token.s3cr3t.va1ue" not in redacted
    assert "xagent-auth-v1" in redacted


# ---------------------------------------------------------------------------
# Process-level record redaction
# ---------------------------------------------------------------------------


def _capture(logger: logging.Logger) -> tuple[list[logging.LogRecord], logging.Handler]:
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    return records, handler


@pytest.mark.usefixtures("restore_logging_state")
@pytest.mark.parametrize("name", ["httpx", "httpcore", "httpcore.connection"])
def test_transport_info_lines_stay_out_of_the_log_after_a_level_change(
    name: str,
) -> None:
    setup_logging(force=True)
    logger = logging.getLogger(name)
    # A third-party SDK constructing its own httpx client does exactly this.
    logger.setLevel(logging.NOTSET)
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

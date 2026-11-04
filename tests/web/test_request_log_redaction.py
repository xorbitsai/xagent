"""Request URLs are logged without credential-bearing query values."""

import logging
from collections.abc import Iterator
from logging.config import dictConfig

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    state (including resetting child loggers of configured names), which would
    otherwise leak into later tests.
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
        (
            f"https://host.example/a?api_key={SECRET}&page=2",
            f"https://host.example/a?api_key={REDACTED}&page=2",
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

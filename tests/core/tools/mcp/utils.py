import contextlib
import socket
import threading
import time
from collections.abc import Generator

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.websocket import websocket_server
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute

from tests.core.tools.mcp.servers.time_server import mcp as time_mcp


def start_server_thread(
    target_func, **kwargs
) -> tuple[threading.Thread, threading.Event]:
    """Start a server in a thread with graceful shutdown support.

    Returns:
        tuple: (thread, stop_event) for managing the server lifecycle
    """
    stop_event = threading.Event()
    thread = threading.Thread(
        target=target_func,
        kwargs=kwargs,
        daemon=True,
    )
    thread.start()
    return thread, stop_event


def wait_for_server_ready(port: int, max_attempts: int = 20) -> None:
    """Wait for a server to be ready by checking socket connection."""
    attempt = 0
    while attempt < max_attempts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", port))
                break
        except ConnectionRefusedError:
            time.sleep(0.1)
            attempt += 1
    else:
        raise RuntimeError(f"Server failed to start after {max_attempts} attempts")


def stop_server_thread(
    thread: threading.Thread, stop_event: threading.Event, timeout: float = 2.0
) -> None:
    """Gracefully stop a server thread."""
    stop_event.set()
    if thread.is_alive():
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError(
                "Server thread is still alive after attempting to terminate it"
            )


def make_server_app() -> Starlette:
    server = time_mcp._mcp_server

    async def handle_ws(websocket):
        async with websocket_server(
            websocket.scope, websocket.receive, websocket.send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )

    app = Starlette(routes=[WebSocketRoute("/ws", endpoint=handle_ws)])

    return app


def run_server(server_port: int, stop_event: threading.Event = None) -> None:
    app = make_server_app()
    config = uvicorn.Config(
        app=app, host="127.0.0.1", port=server_port, log_level="error"
    )

    # Run server in a way that can be stopped gracefully
    if stop_event:
        # Use threading with graceful shutdown
        server = uvicorn.Server(config=config)

        # Start server in a way that can be stopped
        import asyncio

        async def run_server_async():
            await server.serve()

        def run_server_sync():
            asyncio.run(run_server_async())

        # Start server thread
        server_thread = threading.Thread(target=run_server_sync, daemon=True)
        server_thread.start()

        # Wait for server to start
        max_attempts = 20
        attempt = 0
        while attempt < max_attempts and not server.started:
            time.sleep(0.1)
            attempt += 1

        # Wait for stop event
        stop_event.wait()

        # Graceful shutdown
        if server.started:
            server.should_exit = True
            server_thread.join(timeout=2.0)
    else:
        # Original behavior for backward compatibility
        server = uvicorn.Server(config=config)
        server.run()

        # Give server time to start
        while not server.started:
            time.sleep(0.5)


def run_streamable_http_server(
    server: FastMCP, server_port: int, stop_event: threading.Event = None
) -> None:
    """Run a FastMCP server exposing a streamable HTTP endpoint."""
    app = server.streamable_http_app()
    config = uvicorn.Config(
        app=app, host="127.0.0.1", port=server_port, log_level="error"
    )

    if stop_event:
        # Use threading with graceful shutdown
        uvicorn_server = uvicorn.Server(config=config)

        # Start server in a way that can be stopped
        import asyncio

        async def run_server_async():
            await uvicorn_server.serve()

        def run_server_sync():
            asyncio.run(run_server_async())

        # Start server thread
        server_thread = threading.Thread(target=run_server_sync, daemon=True)
        server_thread.start()

        # Wait for server to start
        max_attempts = 20
        attempt = 0
        while attempt < max_attempts and not uvicorn_server.started:
            time.sleep(0.1)
            attempt += 1

        # Wait for stop event
        stop_event.wait()

        # Graceful shutdown
        if uvicorn_server.started:
            uvicorn_server.should_exit = True
            server_thread.join(timeout=2.0)
    else:
        # Original behavior
        uvicorn_server = uvicorn.Server(config=config)
        uvicorn_server.run()


@contextlib.contextmanager
def run_streamable_http(server: FastMCP) -> Generator[None, None, None]:
    """Run the server in a separate thread exposing a streamable HTTP endpoint.

    The endpoint will be available at `http://localhost:{server.settings.port}/mcp/`.
    """
    # Create stop event and start server thread
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_streamable_http_server,
        kwargs={
            "server": server,
            "server_port": server.settings.port,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    thread.start()

    # Wait for server to be ready
    wait_for_server_ready(server.settings.port)

    try:
        yield
    finally:
        # Stop server thread
        stop_server_thread(thread, stop_event)

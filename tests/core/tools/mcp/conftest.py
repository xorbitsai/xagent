import socket
import threading
from collections.abc import Generator

import pytest

from tests.core.tools.mcp.utils import (
    run_server,
    stop_server_thread,
    wait_for_server_ready,
)


@pytest.fixture
def websocket_server_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    raise ValueError("Free port not found!")


@pytest.fixture
def websocket_server(websocket_server_port: int) -> Generator[None, None, None]:
    # Create stop event and start server thread
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_server,
        kwargs={"server_port": websocket_server_port, "stop_event": stop_event},
        daemon=True,
    )
    thread.start()

    # Wait for server to be ready
    wait_for_server_ready(websocket_server_port)

    yield

    # Stop server thread
    stop_server_thread(thread, stop_event)


@pytest.fixture
def socket_enabled():
    """Temporarily enable socket connections for websocket tests."""
    try:
        import pytest_socket

        pytest_socket.enable_socket()
        previous_state = pytest_socket.socket_allow_hosts()
        # Only allow connections to localhost
        pytest_socket.socket_allow_hosts(
            ["127.0.0.1", "localhost"], allow_unix_socket=True
        )
        yield
    finally:
        # Restore previous state
        pytest_socket.socket_allow_hosts(previous_state)

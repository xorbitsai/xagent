"""Shared stand-ins for the tests in this directory.

The MCP server rows are namespaces carrying exactly the attributes the
``WebToolConfig`` builders read, so both builders can be driven from more than
one module. The Composio fakes replay the Proxy Execute answer shape for the
tests that drive the Gmail tools and the googleapiclient transport, so the
response envelope and the launch variables live here once.
"""

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.web.tools.mcp import composio_proxy

API_KEY = "test-composio-api-key-value"
ACCOUNT_ID = "ca_gmail_account_1"
COMPOSIO_USER = "xagent-0123456789abcdef"
GMAIL_HOSTS = ("gmail.googleapis.com",)


class _Response:
    """The ``requests`` response shape the proxy client reads."""

    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeProxy:
    """Answers by endpoint substring, checked in insertion order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.by_endpoint: list[tuple[str, _Response]] = []
        self.default = _Response(200, {"status": 200, "data": {}})
        self.lock = threading.Lock()

    def answer(self, needle: str, status: int, data: Any = None) -> None:
        self.by_endpoint.append(
            (needle, _Response(200, {"status": status, "data": data}))
        )

    def answer_with(self, status_code: int, payload: Any) -> None:
        self.default = _Response(status_code, payload)

    def request(self, **kwargs: Any) -> _Response:
        assert kwargs["url"] == composio_proxy.COMPOSIO_PROXY_URL
        payload = kwargs["json"]
        with self.lock:
            self.calls.append(payload)
        for needle, response in self.by_endpoint:
            if needle in payload["endpoint"]:
                return response
        return self.default


@pytest.fixture
def composio_env(monkeypatch) -> None:
    """The launch variables of a Composio-backed grant, and no Google token."""
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    # A Composio grant launches without these.
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)


@pytest.fixture
def proxy(monkeypatch, composio_env) -> FakeProxy:
    fake = FakeProxy()
    monkeypatch.setattr(composio_proxy.requests, "request", fake.request)
    return fake


@pytest.fixture
def unavailable_mcp_server() -> SimpleNamespace:
    """A server whose config cannot be built, so no runtime connection is tried."""
    return SimpleNamespace(
        id=1,
        name="Example",
        description=None,
        transport="unsupported-test-transport",
        managed="external",
        concurrency_safe=False,
        concurrent_tools=[],
    )


@pytest.fixture
def stdio_mcp_server() -> SimpleNamespace:
    """A server whose config builds fine, so any failure happens at load time."""
    return SimpleNamespace(
        id=5,
        name="Test Stdio Server",
        transport="stdio",
        description="",
        command="python",
        args=["-m", "xagent.web.tools.mcp.aws"],
        env={"FOO": "bar"},
        cwd=None,
        managed="external",
        docker_url=None,
        docker_image=None,
        docker_environment=None,
        docker_working_dir=None,
        volumes=None,
        bind_ports=None,
        restart_policy=None,
        auto_start=None,
    )

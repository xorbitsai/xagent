"""Stand-ins for the MCP server rows the ``WebToolConfig`` builders read.

The builders only ever read plain attributes off a server row, so a namespace
carrying exactly the attributes they touch is enough to drive them. Both
builders are exercised from more than one module in this directory, so the two
shapes live here rather than being rebuilt per module.
"""

from types import SimpleNamespace

import pytest


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

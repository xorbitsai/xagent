"""Coverage for the caller-id env injection site inside
WebToolConfig._build_mcp_server_config's plain-stdio branch.

Every other test of this mechanism exercises
mcp_runtime.build_mcp_runtime_connection only; this file targets the
independent injection at config.py's stdio branch, whose only other caller
is _build_oauth_mcp_stdio_transport_config (a different code path). Deleting
that line previously failed nothing in the suite."""

import asyncio
from types import SimpleNamespace

from xagent.web.tools.config import WebToolConfig


def _stdio_server(**overrides):
    defaults = dict(
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
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_mcp_server_config_injects_caller_id_for_plain_stdio():
    cfg = WebToolConfig(db=None, request=None, user_id=42)
    server = _stdio_server()

    config = asyncio.run(
        cfg._build_mcp_server_config(
            server=server,
            user_env_by_id={},
            shared_env_by_id={},
            env_source_by_id={},
        )
    )

    assert config["config"]["env"] == {"FOO": "bar", "XAGENT_MCP_CALLER_ID": "42"}


def test_build_mcp_server_config_merges_env_overrides_before_caller_id():
    """The user-env override layer still applies; caller-id is merged in
    after, matching build_mcp_runtime_connection's precedence."""
    cfg = WebToolConfig(db=None, request=None, user_id=7)
    server = _stdio_server(id=9)

    config = asyncio.run(
        cfg._build_mcp_server_config(
            server=server,
            user_env_by_id={9: {"OVERRIDE": "own"}},
            shared_env_by_id={},
            env_source_by_id={9: "own"},
        )
    )

    assert config["config"]["env"] == {
        "FOO": "bar",
        "OVERRIDE": "own",
        "XAGENT_MCP_CALLER_ID": "7",
    }

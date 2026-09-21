"""A stdio MCP connector subprocess and the parent process's own
``OutputFilteredToolWrapper`` each independently bound tool output length
(also field count / recursion depth): the child reads
``XAGENT_TOOL_MAX_OUTPUT_LENGTH`` etc. from its OWN environment to build a
bounded, valid-JSON result; the parent reads the same-named env var (or a
per-config override) from ITS OWN environment to decide where to blindly
character-slice that result. ``_create_stdio_session`` launches the child
with a minimal, explicitly-built env (credentials + caller id only), never
inheriting the parent's environment, so the two budgets can silently
disagree -- and when the parent's budget is smaller, its blind slice can cut
the child's already-valid JSON mid-structure.

``create_mcp_tools`` closes this by mirroring the parent's own effective
budget into every stdio config's env before tools are built from it, so
both sides always agree regardless of which env var / config override
produced the parent's number -- except for a server that bypasses the
generic MCP loader for its own actor/execution-scoped session consumer
(e.g. chrome-devtools), which fail-closes on any env key it didn't itself
put there: injecting into that config would take the connector down
instead of fixing its output budget, so those servers are exempted.
"""

import logging

import pytest

from xagent.config import (
    TOOL_MAX_FIELD_COUNT,
    TOOL_MAX_OUTPUT_LENGTH,
    TOOL_MAX_RECURSION_DEPTH,
)
from xagent.core.tools.adapters.vibe.config import MCPFailurePolicy
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.mcp_tools import create_mcp_tools
from xagent.core.tools.adapters.vibe.output_filter import OutputValueFilter


class _FakeConfig:
    def __init__(
        self,
        mcp_configs,
        *,
        session_identities=None,
        session_consumer=None,
    ):
        self._mcp_configs = mcp_configs
        self._session_identities = session_identities or {}
        self._session_consumer = session_consumer

    def get_tool_selection_spec(self):
        return None

    async def get_mcp_server_configs(self):
        return self._mcp_configs

    def get_mcp_failure_policy(self):
        return MCPFailurePolicy.BEST_EFFORT

    def get_sandbox(self):
        return None

    def get_max_output_length(self):
        return 12345

    def get_max_field_count(self):
        return 99

    def get_max_recursion_depth(self):
        return 7

    def get_actor_mcp_stdio_session_identities(self):
        return dict(self._session_identities)

    def get_actor_mcp_stdio_session_consumer(self):
        return self._session_consumer


def _capture_create(monkeypatch, captured):
    async def fake_create(mcp_configs, sandbox=None, **kwargs):
        captured["mcp_configs"] = mcp_configs
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(
        ToolFactory, "_create_mcp_tools_from_configs", staticmethod(fake_create)
    )


@pytest.mark.asyncio
async def test_stdio_configs_receive_the_parents_effective_output_limits(monkeypatch):
    captured: dict = {}
    _capture_create(monkeypatch, captured)

    config = _FakeConfig(
        [
            {
                "name": "jira",
                "transport": "stdio",
                "config": {"command": "python", "args": ["-m", "probe"]},
            }
        ]
    )
    await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    env = cfg["config"]["env"]
    assert env[TOOL_MAX_OUTPUT_LENGTH] == "12345"
    assert env[TOOL_MAX_FIELD_COUNT] == "99"
    assert env[TOOL_MAX_RECURSION_DEPTH] == "7"


@pytest.mark.asyncio
async def test_existing_numeric_stdio_env_entries_are_preserved(monkeypatch):
    """Credentials and caller-id env already placed on the config (e.g. by
    the actor-owned/team-owned loaders) must survive untouched, and an
    already-present NUMERIC output-limit value on the config -- from some
    future per-server override -- is never clobbered. (A non-numeric value
    is a different case, covered separately: the real child-side getter
    would silently ignore it, so it is replaced rather than preserved.)"""
    captured: dict = {}
    _capture_create(monkeypatch, captured)

    config = _FakeConfig(
        [
            {
                "name": "jira",
                "transport": "stdio",
                "config": {
                    "command": "python",
                    "args": [],
                    "env": {
                        "XAGENT_MCP_CALLER_ID": "user-1",
                        TOOL_MAX_OUTPUT_LENGTH: "4096",
                    },
                },
            }
        ]
    )
    await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    env = cfg["config"]["env"]
    assert env["XAGENT_MCP_CALLER_ID"] == "user-1"
    assert env[TOOL_MAX_OUTPUT_LENGTH] == "4096"
    assert env[TOOL_MAX_FIELD_COUNT] == "99"


@pytest.mark.asyncio
async def test_non_numeric_stdio_env_override_is_replaced_and_warned(
    monkeypatch, caplog
):
    """A non-numeric preset value would make the real
    ``get_tool_max_output_length()`` silently fall back to its own default
    inside the child, defeating the point of mirroring a value at all -- so
    it is replaced with the effective value instead of preserved, and the
    replacement is logged."""
    captured: dict = {}
    _capture_create(monkeypatch, captured)

    config = _FakeConfig(
        [
            {
                "name": "jira",
                "transport": "stdio",
                "config": {
                    "command": "python",
                    "args": [],
                    "env": {TOOL_MAX_OUTPUT_LENGTH: "not-a-number"},
                },
            }
        ]
    )
    with caplog.at_level(logging.WARNING):
        await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    env = cfg["config"]["env"]
    assert env[TOOL_MAX_OUTPUT_LENGTH] == "12345"
    assert any("non-numeric" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_existing_int_env_value_is_coerced_to_str(monkeypatch):
    """A pre-existing value that is numerically valid but not itself a
    string (e.g. a real int, as opposed to its string form) must still come
    out as a string: a subprocess env must be all-str, and returning early
    without coercing would leave a non-str value in it."""
    captured: dict = {}
    _capture_create(monkeypatch, captured)

    config = _FakeConfig(
        [
            {
                "name": "jira",
                "transport": "stdio",
                "config": {
                    "command": "python",
                    "args": [],
                    "env": {TOOL_MAX_OUTPUT_LENGTH: 4096},
                },
            }
        ]
    )
    await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    env = cfg["config"]["env"]
    assert env[TOOL_MAX_OUTPUT_LENGTH] == "4096"
    assert isinstance(env[TOOL_MAX_OUTPUT_LENGTH], str)


@pytest.mark.asyncio
async def test_identity_resolution_failure_falls_back_to_no_exemptions(
    monkeypatch, caplog
):
    """If resolving actor session identities itself raises, create_mcp_tools
    must not crash: it falls back to treating no server as
    actor/execution-scoped, the same degrade a config that never defined
    the getter at all already gets, and logs the failure."""
    captured: dict = {}
    _capture_create(monkeypatch, captured)

    class _RaisingIdentityConfig(_FakeConfig):
        def get_actor_mcp_stdio_session_identities(self):
            raise RuntimeError("boom")

    config = _RaisingIdentityConfig(
        [
            {
                "name": "jira",
                "transport": "stdio",
                "config": {"command": "python", "args": []},
            }
        ]
    )
    with caplog.at_level(logging.WARNING):
        await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    assert cfg["config"]["env"][TOOL_MAX_OUTPUT_LENGTH] == "12345"
    assert any(
        "Failed to resolve actor MCP stdio session identities" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_non_stdio_configs_are_left_untouched(monkeypatch):
    captured: dict = {}
    _capture_create(monkeypatch, captured)

    config = _FakeConfig(
        [
            {
                "name": "http-server",
                "transport": "streamable_http",
                "config": {"url": "https://example.com/mcp"},
            }
        ]
    )
    await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    assert "env" not in cfg["config"]


@pytest.mark.asyncio
async def test_unavailable_placeholder_is_left_untouched(monkeypatch):
    """Production never produces transport="stdio" + config.unavailable=True
    -- ``_build_unavailable_mcp_config`` always uses transport="unavailable"
    -- so this is the real shape the transport check must pass through
    untouched."""
    captured: dict = {}
    _capture_create(monkeypatch, captured)

    config = _FakeConfig(
        [
            {
                "name": "broken",
                "transport": "unavailable",
                "config": {"unavailable": True, "reason": "oauth_token_required"},
            }
        ]
    )
    await create_mcp_tools(config)

    (cfg,) = captured["mcp_configs"]
    assert "env" not in cfg["config"]


@pytest.mark.asyncio
async def test_actor_execution_scoped_server_is_exempt_from_injection(monkeypatch):
    """Regression for the chrome-devtools break: a server with a
    session_identity bypasses the generic MCP loader for its own consumer
    (``consume_chrome_actor_stdio_session`` -> ``bind_chrome_execution_scope``
    in production), which fail-closes if the connection carries any env key
    besides the caller id. This runs the REAL
    ``ToolFactory._create_mcp_tools_from_configs`` dispatch (not stubbed) so
    the connection object it hands to the consumer is the actual one that a
    strict consumer would inspect, and asserts it carries none of the
    injected keys."""
    session_identity = object()
    seen: dict = {}

    async def fake_consumer(*, server_name, connection, session_identity, sandbox):
        seen["connection"] = connection
        return []

    config = _FakeConfig(
        [
            {
                "name": "chrome-devtools",
                "transport": "stdio",
                "config": {
                    "command": "npx",
                    "args": ["chrome-devtools-mcp"],
                    "env": {"XAGENT_MCP_CALLER_ID": "user-1"},
                },
            }
        ],
        session_identities={"chrome-devtools": session_identity},
        session_consumer=fake_consumer,
    )

    await create_mcp_tools(config)

    connection = seen["connection"]
    assert connection["env"] == {"XAGENT_MCP_CALLER_ID": "user-1"}
    assert TOOL_MAX_OUTPUT_LENGTH not in connection["env"]
    assert TOOL_MAX_FIELD_COUNT not in connection["env"]
    assert TOOL_MAX_RECURSION_DEPTH not in connection["env"]


def test_non_exempt_stdio_server_alongside_an_exempt_one():
    """The exemption is per-server, not all-or-nothing: an ordinary stdio
    connector still gets its budget mirrored even when a different,
    actor-scoped server in the same batch is exempt. Exercises
    ``_apply_stdio_output_limits_env`` directly (rather than through
    ``create_mcp_tools``) so the non-exempt "jira" entry never reaches the
    real MCP loader -- there is no fake consumer for it here, and letting
    it fall into ``connections`` would attempt a real stdio handshake."""
    from xagent.core.tools.adapters.vibe.mcp_tools import (
        _apply_stdio_output_limits_env,
    )

    config = _FakeConfig([])  # configs passed directly below, not through it
    configs = [
        {
            "name": "chrome-devtools",
            "transport": "stdio",
            "config": {"command": "npx", "args": [], "env": {}},
        },
        {
            "name": "jira",
            "transport": "stdio",
            "config": {"command": "python", "args": []},
        },
    ]

    result = _apply_stdio_output_limits_env(
        configs, config, exempt_server_names=frozenset({"chrome-devtools"})
    )

    by_name = {cfg["name"]: cfg for cfg in result}
    assert by_name["chrome-devtools"]["config"]["env"] == {}
    assert by_name["jira"]["config"]["env"][TOOL_MAX_OUTPUT_LENGTH] == "12345"


def test_same_budget_child_payload_survives_the_parents_filter():
    """The actual regression this whole mechanism guards against: when the
    parent's OutputFilteredToolWrapper budget matches what the child used
    to build its own bounded, valid-JSON output (which this PR's env
    mirroring guarantees), the wrapper's blind character-slice must never
    fire -- a same-budget payload comes back byte-for-byte unchanged, not
    truncated mid-structure."""
    budget = 200
    payload = '{"issues": [' + ", ".join(f'"issue-{i}"' for i in range(20)) + "]}"
    payload = payload[:budget]  # the child bounds its own output to `budget`

    filtered = OutputValueFilter(
        max_chars=budget, max_fields=1000, max_recursion=10
    ).filter(payload, tool_name="jira")

    assert filtered == payload

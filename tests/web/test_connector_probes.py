"""The probe table in web/connector_probes.py against the connector registry.

Every builtin catalog app must either declare one read-only probe or be
recorded as unprobeable with a known reason; a declared probe must name a tool
the in-repo server actually registers, pass only arguments that tool takes,
and sit on a python stdio row, because the probe runs the call on the web
host after the listing released any sandbox lease.
"""

from __future__ import annotations

import importlib
import inspect
import re
from typing import Any

import pytest

from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
from xagent.web.connector_probes import (
    CONNECTOR_PROBES,
    UNPROBEABLE_CONNECTORS,
    UNPROBEABLE_REASONS,
    ConnectorProbe,
    get_connector_probe,
    get_unprobeable_reason,
)

# A tool whose name carries one of these is not read-only. The list is a
# lint on the declaration, not a proof; the proof is reading the tool.
_WRITE_VERBS = re.compile(
    r"(^|_)(create|add|update|set|delete|remove|send|post|publish|upload|"
    r"reply|join|mark|transition|assign|unassign|append|replace|insert|"
    r"import|clear|move|rename|share|manage|refund|comment|export)(_|$)"
)


def _rows() -> dict[str, dict[str, Any]]:
    return {row["app_id"]: row for row in get_builtin_public_mcp_app_rows()}


def _module_name(row: dict[str, Any]) -> str | None:
    launch = row["launch_config"]
    if launch.get("command") != "python":
        return None
    args = launch.get("args") or []
    if len(args) == 2 and args[0] == "-m":
        return args[1]
    return None


async def _registered_tools(module_name: str) -> dict[str, dict[str, set[str]]]:
    """Tool name -> accepted and required argument names, per the server's schema."""
    module = importlib.import_module(module_name)
    fastmcp = getattr(module, "mcp", None)
    if fastmcp is not None and hasattr(fastmcp, "list_tools"):
        tools = await fastmcp.list_tools()
    else:
        # The low-level Server modules (linkedin) expose the handler directly.
        tools = await module.list_tools()
    return {
        tool.name: {
            "properties": set((tool.inputSchema or {}).get("properties", {})),
            "required": set((tool.inputSchema or {}).get("required", [])),
        }
        for tool in tools
    }


def test_every_builtin_app_declares_a_probe_or_a_reason():
    rows = _rows()
    declared = set(CONNECTOR_PROBES) | set(UNPROBEABLE_CONNECTORS)
    assert set(rows) - declared == set(), "builtin apps without a declaration"
    assert declared - set(rows) == set(), "declarations for unknown apps"
    assert set(CONNECTOR_PROBES) & set(UNPROBEABLE_CONNECTORS) == set()


def test_unprobeable_reasons_are_known():
    assert set(UNPROBEABLE_CONNECTORS.values()) <= UNPROBEABLE_REASONS


def test_remote_and_sandboxed_rows_are_unprobeable_for_that_reason():
    for app_id, row in _rows().items():
        launch = row["launch_config"]
        if "url" in launch:
            assert get_unprobeable_reason(app_id) == "covered_by_l2", app_id
        elif launch.get("command") in {"npx", "uvx"}:
            assert get_unprobeable_reason(app_id) in {
                "sandboxed",
                "execution_scoped",
            }, app_id


def test_chrome_devtools_is_execution_scoped():
    assert get_unprobeable_reason("chrome-devtools") == "execution_scoped"


def test_probes_sit_on_python_stdio_rows_only():
    rows = _rows()
    for app_id in CONNECTOR_PROBES:
        assert _module_name(rows[app_id]) is not None, app_id


def test_probe_tools_read_only_by_name():
    for app_id, probe in CONNECTOR_PROBES.items():
        assert not _WRITE_VERBS.search(probe.tool), (app_id, probe.tool)


def test_probe_shape():
    probe = ConnectorProbe("x")
    assert probe.arguments == {}
    assert probe.expect_keys == ()
    assert get_connector_probe("github") == ConnectorProbe("github_get_current_user")
    assert get_connector_probe("not-an-app") is None
    assert get_unprobeable_reason("not-an-app") is None


def test_key_scoped_probes_stay_inside_the_catalog_description():
    """The catalog descriptions list feature flags and customers; /users/@me/
    and /v1/account need scopes they never ask for."""
    assert get_connector_probe("posthog") == ConnectorProbe(
        "posthog_list_feature_flags", {"limit": 1}
    )
    assert get_connector_probe("stripe") == ConnectorProbe(
        "stripe_list_customers", {"limit": 1}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("app_id", sorted(CONNECTOR_PROBES))
async def test_declared_tool_is_registered_with_those_arguments(app_id):
    """Catches a probe naming a tool the server does not have (the issue's
    original "Slack auth check") or an argument the tool does not take, which
    the adapter would reject before the call."""
    probe = CONNECTOR_PROBES[app_id]
    module_name = _module_name(_rows()[app_id])
    assert module_name is not None
    tools = await _registered_tools(module_name)
    assert probe.tool in tools, (app_id, probe.tool, sorted(tools))
    unknown = set(probe.arguments) - tools[probe.tool]["properties"]
    assert unknown == set(), (app_id, probe.tool, unknown)


@pytest.mark.asyncio
@pytest.mark.parametrize("app_id", sorted(CONNECTOR_PROBES))
async def test_declared_tool_needs_no_argument_beyond_the_declared_ones(app_id):
    """Every parameter without a default must be declared, or the server
    rejects the call."""
    probe = CONNECTOR_PROBES[app_id]
    module_name = _module_name(_rows()[app_id])
    assert module_name is not None
    module = importlib.import_module(module_name)
    fn = getattr(module, probe.tool, None)
    if fn is None:
        # The low-level Server modules dispatch by name, so there is no
        # function signature to read; check the registered input schema.
        tools = await _registered_tools(module_name)
        schema_required = tools[probe.tool]["required"]
        assert schema_required <= set(probe.arguments), (
            app_id,
            probe.tool,
            schema_required,
        )
        return
    required = {
        name
        for name, parameter in inspect.signature(fn).parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    assert required <= set(probe.arguments), (app_id, probe.tool, required)

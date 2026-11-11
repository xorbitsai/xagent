"""Every tool of a connector that adopted V2 has a check or an exemption (#3565).

A builtin connector either adopted V2 (``CONNECTOR_CHECKS``) or is on the
legacy list (``LEGACY_CONNECTORS``), which only shrinks. For an in-repo
connector, started as ``python -m``, the tools its server registers must each
have a check or a written exemption, and every tool a declaration names must
exist. A remote (``url``) connector's tools are the vendor's and change
without a release, and an ``npx`` connector's are not in this repository, so
neither is held to per-tool coverage.
"""

from __future__ import annotations

import asyncio
import importlib
import re
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
from xagent.web.connector_checks.registry import (
    CONNECTOR_CHECKS,
    LEGACY_CONNECTORS,
    WRITE_WITHOUT_CLEANUP,
)

# LEGACY_CONNECTORS, pinned. A connector that adopts V2 leaves both lists in
# the same PR; never add a name here, so going back to legacy shows up in the
# diff of this file.
_PINNED_LEGACY = frozenset(
    {
        "aws",
        "chartmogul",
        "chrome-devtools",
        "deputy",
        "employment-hero",
        "facebook",
        "fireflies",
        "freshdesk",
        "github",
        "gmail",
        "google-analytics",
        "google-calendar",
        "google-drive",
        "google-maps",
        "google-search-console",
        "granola",
        "hubspot",
        "instagram",
        "intercom",
        "linear",
        "linkedin",
        "magento",
        "meta-ads",
        "mixpanel",
        "myob",
        "onedrive",
        "outlook",
        "planner",
        "posthog",
        "rocketlane",
        "salesforce",
        "sharepoint",
        "shopify",
        "slack",
        "stripe",
        "teams",
        "whatsapp",
        "xero",
    }
)


def _rows() -> dict[str, dict[str, Any]]:
    return {str(row["app_id"]): row for row in get_builtin_public_mcp_app_rows()}


def _module_name(row: dict[str, Any]) -> str | None:
    """The in-repo server module of a ``python -m`` connector, else None."""
    launch = row["launch_config"]
    args = launch.get("args") or []
    if launch.get("command") == "python" and len(args) == 2 and args[0] == "-m":
        return str(args[1])
    return None


def _registered_tools(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    server = getattr(module, "mcp", None)
    if server is not None and hasattr(server, "list_tools"):
        tools = asyncio.run(server.list_tools())
    else:
        # The low-level Server modules (linkedin) expose the handler directly.
        tools = asyncio.run(module.list_tools())
    registered = {tool.name for tool in tools}
    # zoom offers its meeting-write tool only when a deployment switch is on;
    # an exemption has to cover it whether or not this process sees it.
    register_gated = getattr(module, "_register_meeting_write_tools", None)
    if register_gated is not None:
        gated = FastMCP(f"{module_name}-coverage")
        register_gated(gated)
        registered |= {tool.name for tool in asyncio.run(gated.list_tools())}
    return registered


def _adopted_in_repo() -> list[str]:
    rows = _rows()
    return sorted(
        app_id
        for app_id in CONNECTOR_CHECKS
        if app_id in rows and _module_name(rows[app_id]) is not None
    )


def test_the_legacy_list_is_pinned():
    added = LEGACY_CONNECTORS - _PINNED_LEGACY
    assert not added, (
        f"{sorted(added)} joined the legacy list; a connector ships its V2 "
        "declaration instead"
    )
    left = _PINNED_LEGACY - LEGACY_CONNECTORS
    assert not left, (
        f"{sorted(left)} left the legacy list; remove them from _PINNED_LEGACY too"
    )


def test_an_adopted_connector_is_not_legacy():
    assert not LEGACY_CONNECTORS & set(CONNECTOR_CHECKS)


def test_every_builtin_connector_is_adopted_or_legacy():
    """Remote and ``npx`` connectors included: per-tool coverage skips them,
    so this is what stops one shipping with no V2 declaration."""
    rows = _rows()
    assert any(row["launch_config"].get("url") for row in rows.values())
    assert any(row["launch_config"].get("command") == "npx" for row in rows.values())
    undeclared = set(rows) - set(CONNECTOR_CHECKS) - LEGACY_CONNECTORS
    assert not undeclared, (
        f"{sorted(undeclared)} need a V2 declaration in "
        "web/connector_checks/registry.py"
    )


def test_a_legacy_connector_is_a_builtin_connector():
    removed = LEGACY_CONNECTORS - set(_rows())
    assert not removed, f"{sorted(removed)} are no longer builtin connectors"


@pytest.mark.parametrize("app_id", sorted(CONNECTOR_CHECKS))
def test_a_pending_exemption_names_the_issue_that_decides_it(app_id):
    pending = [
        exemption
        for exemption in CONNECTOR_CHECKS[app_id].exemptions
        if exemption.reason != WRITE_WITHOUT_CLEANUP
    ]
    unnamed = [e.tool for e in pending if not re.search(r"decided in #\d+$", e.reason)]
    assert not unnamed, f"{app_id}: {unnamed} need the issue that decides their check"


@pytest.mark.parametrize("app_id", _adopted_in_repo())
def test_every_tool_has_a_check_or_an_exemption(app_id):
    checks = CONNECTOR_CHECKS[app_id]
    module_name = _module_name(_rows()[app_id])
    assert module_name is not None
    registered = _registered_tools(module_name)
    exempted = {exemption.tool for exemption in checks.exemptions}
    covered = set(checks.checked_tools()) | exempted

    uncovered = registered - covered
    assert not uncovered, (
        f"{app_id}: {sorted(uncovered)} need a check or an exemption with a reason"
    )
    unknown = covered - registered
    assert not unknown, f"{app_id}: {sorted(unknown)} are not tools of the server"


def test_a_remote_connector_declares_no_exemptions():
    """Its tools change with the vendor, so an exemption list would rot."""
    rows = _rows()
    remote = [
        app_id
        for app_id in CONNECTOR_CHECKS
        if app_id in rows and rows[app_id]["launch_config"].get("url")
    ]
    assert remote, "expected remote connectors in CONNECTOR_CHECKS"
    assert all(not CONNECTOR_CHECKS[app_id].exemptions for app_id in remote)

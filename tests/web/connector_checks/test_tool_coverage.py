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

from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
from xagent.web.connector_checks.registry import (
    CONNECTOR_CHECKS,
    LEGACY_CONNECTORS,
    WRITE_WITHOUT_CLEANUP,
)

# The legacy list when coverage became required. LEGACY_CONNECTORS may only
# lose names; never add one here.
_INITIAL_LEGACY = frozenset(
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
        "google-ads",
        "google-analytics",
        "google-calendar",
        "google-drive",
        "google-maps",
        "google-search-console",
        "granola",
        "hubspot",
        "instagram",
        "intercom",
        "jira",
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
        "zendesk",
        "zoom",
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
    return {tool.name for tool in tools}


def _adopted_in_repo() -> list[str]:
    rows = _rows()
    return sorted(
        app_id
        for app_id in CONNECTOR_CHECKS
        if app_id in rows and _module_name(rows[app_id]) is not None
    )


def test_the_legacy_list_only_shrinks():
    added = LEGACY_CONNECTORS - _INITIAL_LEGACY
    assert not added, (
        f"{sorted(added)} joined the legacy list; a connector ships its V2 "
        "declaration instead"
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


@pytest.mark.parametrize("app_id", _adopted_in_repo())
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
        for app_id, checks in CONNECTOR_CHECKS.items()
        if app_id in rows and rows[app_id]["launch_config"].get("url")
    ]
    assert remote, "expected remote connectors in CONNECTOR_CHECKS"
    assert all(not CONNECTOR_CHECKS[app_id].exemptions for app_id in remote)

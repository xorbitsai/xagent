"""Every migrated entrypoint must keep the facade, not merely five connectors."""

import importlib

import pytest

from xagent.core.utils import public_requests

MIGRATED_MODULES = [
    "xagent.web.api.auth",
    "xagent.skills.registries.base",
    "xagent.skills.registries.clawhub",
    *[
        f"xagent.web.tools.mcp.{name}"
        for name in (
            "chartmogul",
            "deputy",
            "employment_hero",
            "excel",
            "freshdesk",
            "github",
            "google_ads",
            "google_analytics",
            "google_search_console",
            "hubspot",
            "intercom",
            "jira",
            "jira_attachment",
            "linear",
            "linkedin",
            "magento",
            "meta_graph",
            "mixpanel",
            "myob",
            "onedrive",
            "outlook",
            "planner",
            "posthog",
            "powerpoint",
            "salesforce",
            "sharepoint",
            "shopify",
            "slack",
            "stripe",
            "teams",
            "word",
            "xero",
            "zendesk",
            "zoom",
        )
    ],
]


@pytest.mark.parametrize("module_name", MIGRATED_MODULES)
def test_migrated_module_uses_public_requests(module_name):
    assert importlib.import_module(module_name).requests is public_requests

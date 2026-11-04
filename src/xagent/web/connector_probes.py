"""The read-only tool-call probe declared for each builtin connector (L3).

A connector's ``tools/list`` never reaches the vendor for the in-repo stdio
servers: their tool lists are static, so an invalid credential, a missing
scope or a vendor API change passes a listing. The probe is the one tool call
that does reach the vendor. Each builtin catalog app either declares one
probe here or is recorded as unprobeable with a reason; the registry test
fails for an app that has neither.

The table is code-owned, like ``_EXECUTION_SCOPED_STDIO_APP_IDS`` in the
registry, and deliberately not a ``launch_config`` field: ``launch_config`` is
persisted to ``public_mcp_apps`` and ``validate_builtin_public_mcp_apps`` pins
its shape. A probe is keyed by the catalog app id, so an admin-created custom
server row never has one.

Every probe must be read-only and idempotent, take no argument that names a
vendor object the monitor account would have to own, and ask for the
smallest page the tool accepts. Its result is accepted only when the tool
reports success in its own envelope (see
``services/connector_probe.py``): the in-repo servers catch vendor errors and
return ``{"status": "error", ...}`` with the MCP ``isError`` flag unset, so
the flag alone would pass an invalid token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, get_args

UnprobeableReason = Literal[
    "covered_by_l2",
    "sandboxed",
    "execution_scoped",
    "needs_fixture",
]

# Why a connector declares no probe:
# - covered_by_l2: a remote MCP server; its tools/list reaches the vendor and
#   a rejected credential is classified AUTH_REJECTED, and its tool names are
#   vendor-owned and may change.
# - sandboxed: an npx/uvx row. The connector session releases its sandbox
#   lease once the listing is done and a tool call opens a new session per
#   call, so the probe would run the package on the web host.
# - execution_scoped: needs a task's execution identity (chrome-devtools).
# - needs_fixture: every read-only tool takes a vendor object id (a document,
#   a spreadsheet, a file path) the monitor account would have to own.
UNPROBEABLE_REASONS: frozenset[str] = frozenset(get_args(UnprobeableReason))


@dataclass(frozen=True)
class ConnectorProbe:
    """One read-only tool call.

    ``arguments`` are the fixed call arguments. ``expect_keys`` names keys the
    result object must carry, checked whenever declared; it is required for a
    tool that does not use the ``{"status": ...}`` envelope.
    """

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    expect_keys: tuple[str, ...] = ()


CONNECTOR_PROBES: Mapping[str, ConnectorProbe] = {
    # Google
    "gmail": ConnectorProbe("gmail_search_messages", {"max_results": 1}),
    # Under the drive.file scope this sees only picker-selected or
    # Xagent-created files, so an empty page is a pass.
    "google-drive": ConnectorProbe("google_drive_search", {"max_results": 1}),
    "google-calendar": ConnectorProbe(
        "google_calendar_search_events", {"max_results": 1}
    ),
    "google-ads": ConnectorProbe("google_ads_list_accessible_customers"),
    "google-analytics": ConnectorProbe("google_analytics_list_properties"),
    "google-search-console": ConnectorProbe("google_search_console_list_sites"),
    # Microsoft
    "teams": ConnectorProbe("get_current_user"),
    "outlook": ConnectorProbe("outlook_get_profile"),
    "onedrive": ConnectorProbe("onedrive_get_profile"),
    "planner": ConnectorProbe("planner_list_my_plans"),
    "sharepoint": ConnectorProbe("sharepoint_get_root_site"),
    # Meta: each auth_status tool reads /me.
    "facebook": ConnectorProbe("facebook_auth_status"),
    "instagram": ConnectorProbe("instagram_auth_status"),
    "meta-ads": ConnectorProbe("meta_ads_auth_status"),
    "whatsapp": ConnectorProbe("whatsapp_auth_status"),
    # Other builtin OAuth
    # The LinkedIn server returns the raw userinfo document on success and
    # an ``Error: ...`` line on failure, both without the status envelope.
    "linkedin": ConnectorProbe("get_profile", expect_keys=("sub",)),
    "zoom": ConnectorProbe("zoom_get_current_user"),
    "slack": ConnectorProbe("slack_list_channels", {"limit": 1}),
    "github": ConnectorProbe("github_get_current_user"),
    # A read through Intercom's POST /conversations/search, the only listing
    # the API offers.
    "intercom": ConnectorProbe("intercom_list_conversations", {"limit": 1}),
    "zendesk": ConnectorProbe("zendesk_list_users", {"limit": 1}),
    "salesforce": ConnectorProbe("salesforce_get_current_user"),
    "deputy": ConnectorProbe("deputy_get_current_user"),
    "linear": ConnectorProbe("linear_get_current_user"),
    "jira": ConnectorProbe("jira_list_accessible_sites"),
    "employment-hero": ConnectorProbe(
        "employment_hero_list_organisations", {"item_per_page": 1}
    ),
    "xero": ConnectorProbe("list_organisation_details"),
    "myob": ConnectorProbe("myob_get_business_info"),
    # API key
    "aws": ConnectorProbe("aws_get_caller_identity"),
    # /users/@me/ needs user:read, which the catalog description never asks for.
    "posthog": ConnectorProbe("posthog_list_feature_flags", {"limit": 1}),
    "mixpanel": ConnectorProbe("mixpanel_list_event_names", {"limit": 1}),
    # /v1/account needs Accounts Read, which the catalog description never asks for.
    "stripe": ConnectorProbe("stripe_list_customers", {"limit": 1}),
    "freshdesk": ConnectorProbe("freshdesk_list_agents", {"per_page": 1}),
    "chartmogul": ConnectorProbe("chartmogul_list_customers", {"per_page": 1}),
    "shopify": ConnectorProbe("shopify_validate_connection"),
    "magento": ConnectorProbe("magento_list_products", {"limit": 1}),
}

UNPROBEABLE_CONNECTORS: Mapping[str, UnprobeableReason] = {
    "hubspot": "covered_by_l2",
    "granola": "covered_by_l2",
    "notion": "covered_by_l2",
    "atlassian": "covered_by_l2",
    "miro": "covered_by_l2",
    "fireflies": "covered_by_l2",
    "rocketlane": "covered_by_l2",
    "google-maps": "sandboxed",
    "chrome-devtools": "execution_scoped",
    "google-docs": "needs_fixture",
    "google-sheets": "needs_fixture",
    "google-slides": "needs_fixture",
    "powerpoint": "needs_fixture",
    "word": "needs_fixture",
    "excel": "needs_fixture",
}


def get_connector_probe(app_id: str) -> ConnectorProbe | None:
    return CONNECTOR_PROBES.get(app_id)


def get_unprobeable_reason(app_id: str) -> UnprobeableReason | None:
    return UNPROBEABLE_CONNECTORS.get(app_id)

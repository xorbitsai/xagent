"""Every connector's V2 declaration: declared, or derived from its L3 probe."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from xagent.core.tools.adapters.vibe import mcp_adapter
from xagent.core.tools.adapters.vibe.mcp_adapter import MCPToolAdapter
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
from xagent.web.connector_checks.assertions import judge_result, resolve_expected
from xagent.web.connector_checks.declarations import (
    Assertion,
    ConnectorChecks,
    Contains,
    CreatedRef,
    DeclarationError,
    EachHasKeys,
    Equals,
    FixtureRef,
    HasKeys,
    IsType,
    ItemRef,
    L3Envelope,
    Length,
    ReadCheck,
    RunScopedName,
    StatusSuccess,
    Sweep,
    TenantPin,
    ToolStep,
    UnprobeableCheck,
    WriteCheck,
    validate_connector_checks,
)
from xagent.web.connector_checks.fixtures import FixtureValue, resolve_fixtures
from xagent.web.connector_checks.registry import (
    _GITHUB_READS_BEYOND_THE_FIVE,
    _GMAIL_LABEL_WITHOUT_READ,
    _SLIDE_THUMBNAIL_WITHOUT_SLIDE_ID,
    CONNECTOR_CHECKS,
    WRITE_WITHOUT_CLEANUP,
    _declared_l3,
    connector_checks_for,
    l3_read_check,
)
from xagent.web.connector_checks.results import FailureClass
from xagent.web.tools.mcp import calendar as google_calendar
from xagent.web.tools.mcp import (
    excel,
    github,
    gmail,
    google_ads,
    google_analytics,
    google_docs,
    google_drive,
    google_sheets,
    google_slides,
    outlook,
    planner,
    powerpoint,
    sharepoint,
    word,
    zendesk,
    zoom,
)

# The three Microsoft document connectors and the module of each. What their
# checks assert is covered end to end in test_microsoft_365_checks.py.
_APP_MODULES = {
    "excel": excel,
    "powerpoint": powerpoint,
    "word": word,
}


def _fixture_refs(value: Any) -> Iterator[FixtureRef]:
    if isinstance(value, FixtureRef):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _fixture_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _fixture_refs(item)


def test_a_probed_connector_gets_its_l3_probe_as_one_read_check() -> None:
    assert connector_checks_for("stripe") == ConnectorChecks(
        "stripe",
        reads=(
            ReadCheck(
                "stripe.l3_probe",
                "stripe_list_customers",
                {"limit": 1},
                assertions=(L3Envelope(),),
            ),
        ),
    )


def test_the_l3_expect_keys_carry_over() -> None:
    checks = connector_checks_for("linkedin")

    assert checks is not None
    assert checks.reads == (
        ReadCheck(
            "linkedin.l3_probe", "get_profile", assertions=(L3Envelope(("sub",)),)
        ),
    )


@pytest.mark.parametrize(
    ("app_id", "reason"),
    [
        ("granola", "covered_by_l2"),
        ("google-maps", "sandboxed"),
        ("chrome-devtools", "execution_scoped"),
    ],
)
def test_an_unprobeable_connector_gets_an_unprobeable_slot(
    app_id: str, reason: str
) -> None:
    assert connector_checks_for(app_id) == ConnectorChecks(
        app_id,
        reads=(UnprobeableCheck(f"{app_id}.l3_probe", reason),),  # type: ignore[arg-type]
    )


def test_an_unknown_app_has_no_declaration() -> None:
    assert connector_checks_for("not-a-catalog-app") is None


@pytest.mark.parametrize(
    ("app_id", "check"),
    [
        (
            "notion",
            ReadCheck(
                "notion.current_user",
                "notion-get-self",
                payload="json_text",
                assertions=(
                    HasKeys((), ("workspace", "user")),
                    HasKeys(("user",), ("id",)),
                ),
                require_read_only_hint=True,
            ),
        ),
        (
            "miro",
            ReadCheck(
                "miro.current_user",
                "user_who_am_i",
                payload="structured_content",
                assertions=(HasKeys((), ("user_id",)),),
                require_read_only_hint=True,
            ),
        ),
        (
            "atlassian",
            ReadCheck(
                "atlassian.current_user",
                "atlassianUserInfo",
                payload="json_text",
                assertions=(HasKeys(("data",), ("accountId",)),),
                require_read_only_hint=True,
            ),
        ),
    ],
)
def test_a_vendor_hosted_read_check_is_declared_exactly(
    app_id: str, check: ReadCheck
) -> None:
    assert connector_checks_for(app_id) == ConnectorChecks(app_id, reads=(check,))


def test_every_remote_mcp_read_check_requires_the_read_only_hint() -> None:
    remote_app_ids = {
        row["app_id"]
        for row in get_builtin_public_mcp_app_rows()
        if "url" in row["launch_config"]
    }
    assert {"notion", "miro", "atlassian"} <= remote_app_ids & set(CONNECTOR_CHECKS)
    for app_id in remote_app_ids & set(CONNECTOR_CHECKS):
        for check in CONNECTOR_CHECKS[app_id].reads:
            if isinstance(check, ReadCheck):
                assert check.require_read_only_hint, (app_id, check.id)


def test_every_builtin_catalog_app_has_a_valid_declaration() -> None:
    app_ids = sorted(row["app_id"] for row in get_builtin_public_mcp_app_rows())

    assert app_ids
    for app_id in app_ids:
        checks = connector_checks_for(app_id)
        assert checks is not None, app_id
        assert checks.app_id == app_id
        validate_connector_checks(checks)


def test_declared_v2_checks_are_valid_and_keyed_by_their_app() -> None:
    app_ids = {row["app_id"] for row in get_builtin_public_mcp_app_rows()}

    assert set(CONNECTOR_CHECKS) <= app_ids
    for app_id, checks in CONNECTOR_CHECKS.items():
        assert checks.app_id == app_id
        validate_connector_checks(checks)


def test_every_document_fixture_key_is_namespaced_by_its_app() -> None:
    refs = [
        (app_id, ref)
        for app_id in _APP_MODULES
        for check in CONNECTOR_CHECKS[app_id].reads
        if isinstance(check, ReadCheck)
        for ref in _fixture_refs(check.arguments)
    ]

    # Every document connector contributes at least one fixture.
    assert set(_APP_MODULES) <= {app_id for app_id, _ in refs}
    for app_id, ref in refs:
        assert ref.key.startswith(f"{app_id}."), (app_id, ref.key)


@pytest.mark.parametrize("app_id", sorted(_APP_MODULES))
def test_the_probe_route_still_needs_a_fixture(app_id: str) -> None:
    assert l3_read_check(app_id) == UnprobeableCheck(
        f"{app_id}.l3_probe", "needs_fixture"
    )


@pytest.mark.parametrize("app_id", sorted(_APP_MODULES))
def test_the_fixture_is_bound_to_a_required_parameter(app_id: str) -> None:
    """The tool is the module attribute of its own name: the in-repo servers
    register each ``@mcp.tool`` function under that name, and the decorator
    returns the function itself, so its signature is the tool's schema."""
    module = _APP_MODULES[app_id]
    for check in CONNECTOR_CHECKS[app_id].reads:
        assert isinstance(check, ReadCheck)
        tool = getattr(module, check.tool)
        parameters = inspect.signature(tool).parameters
        bound = [
            name
            for name, value in check.arguments.items()
            if isinstance(value, FixtureRef)
        ]
        assert bound
        for name in bound:
            assert name in parameters
            assert parameters[name].default is inspect.Parameter.empty
        required = {
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        }
        assert required <= set(check.arguments), (check.tool, required)


def _linear_pin(
    tool: str, field: str, item_keys: tuple[str, ...]
) -> tuple[str, dict[str, Any], tuple[Assertion, ...]]:
    return (
        tool,
        {"limit": 1},
        (
            StatusSuccess(),
            HasKeys((), (field, "truncated")),
            EachHasKeys((field,), item_keys),
            Length((field,), max=1),
            IsType(("truncated",), "boolean"),
        ),
    )


def _xero_unpaged_pin(
    field: str,
) -> tuple[str, dict[str, Any], tuple[Assertion, ...]]:
    return (
        f"list_{field}",
        {},
        (
            StatusSuccess(),
            HasKeys((), (field, "total_count", "returned")),
            IsType(("total_count",), "number"),
            IsType((field,), "array"),
            IsType(("returned",), "number"),
        ),
    )


def _xero_paged_pin(
    field: str, *, capped: bool = True
) -> tuple[str, dict[str, Any], tuple[Assertion, ...]]:
    # `GET /Quotes` has no `pageSize`, so the quotes check caps nothing.
    cap: tuple[Assertion, ...] = (Length((field,), max=1),) if capped else ()
    return (
        f"list_{field}",
        {"page_size": 1},
        (
            StatusSuccess(),
            HasKeys((), (field, "page", "page_size", "returned", "has_more")),
            IsType(("has_more",), "boolean"),
            IsType((field,), "array"),
            *cap,
            IsType(("returned",), "number"),
        ),
    )


# The Tier 2 read checks (issue #3573): the tool, its arguments, and every
# assertion, so dropping an item key or a whole assertion fails here.
_TIER_2_READS: Mapping[str, tuple[str, dict[str, Any], tuple[Assertion, ...]]] = {
    "google-ads.list_accessible_customers": (
        "google_ads_list_accessible_customers",
        {},
        (
            StatusSuccess(),
            HasKeys((), ("customer_ids",)),
            IsType(("customer_ids",), "array"),
        ),
    ),
    "zendesk.list_tickets": (
        "zendesk_list_tickets",
        {"limit": 1},
        (
            StatusSuccess(),
            HasKeys((), ("tickets", "has_more", "after_cursor", "truncated")),
            EachHasKeys(
                ("tickets",),
                (
                    "id",
                    "subject",
                    "status",
                    "priority",
                    "requester_id",
                    "assignee_id",
                    "group_id",
                    "tags",
                    "created_at",
                    "updated_at",
                ),
            ),
            Length(("tickets",), max=1),
        ),
    ),
    "zendesk.list_organizations": (
        "zendesk_list_organizations",
        {"limit": 1},
        (
            StatusSuccess(),
            HasKeys((), ("organizations", "has_more", "after_cursor", "truncated")),
            EachHasKeys(
                ("organizations",), ("id", "name", "domain_names", "created_at")
            ),
            Length(("organizations",), max=1),
        ),
    ),
    "zendesk.list_macros": (
        "zendesk_list_macros",
        {"limit": 1},
        (
            StatusSuccess(),
            HasKeys((), ("macros", "has_more", "after_cursor", "truncated")),
            EachHasKeys(("macros",), ("id", "title", "description", "updated_at")),
            Length(("macros",), max=1),
        ),
    ),
    "zoom.list_meetings": (
        "zoom_list_meetings",
        {},
        (
            StatusSuccess(),
            HasKeys((), ("meetings", "next_page_token")),
            IsType(("meetings",), "array"),
        ),
    ),
    # Graph returns only the $select fields an item has (a draft has no
    # ``from``), so an outlook list item asserts its id alone.
    "outlook.get_profile": (
        "outlook_get_profile",
        {},
        (
            StatusSuccess(),
            HasKeys(("user",), ("id", "userPrincipalName")),
            IsType(("user", "id"), "string"),
        ),
    ),
    "outlook.list_messages": (
        "outlook_list_messages",
        {"top": 1},
        (
            StatusSuccess(),
            HasKeys((), ("messages", "next_link")),
            EachHasKeys(("messages",), ("id",)),
            Length(("messages",), max=1),
        ),
    ),
    "outlook.list_events": (
        "outlook_list_events",
        {"top": 1},
        (
            StatusSuccess(),
            HasKeys((), ("events", "next_link")),
            EachHasKeys(("events",), ("id",)),
            Length(("events",), max=1),
        ),
    ),
    "outlook.list_contacts": (
        "outlook_list_contacts",
        {"top": 1},
        (
            StatusSuccess(),
            HasKeys((), ("contacts", "next_link")),
            EachHasKeys(("contacts",), ("id",)),
            Length(("contacts",), max=1),
        ),
    ),
    # A SharePoint read takes a site id; Graph's "root" alias names the
    # tenant's root site without a fixture.
    "sharepoint.get_root_site": (
        "sharepoint_get_root_site",
        {},
        (
            StatusSuccess(),
            HasKeys(("site",), ("id", "webUrl")),
            IsType(("site", "id"), "string"),
        ),
    ),
    "sharepoint.get_site": (
        "sharepoint_get_site",
        {"site_id": "root"},
        (
            StatusSuccess(),
            HasKeys(("site",), ("id", "webUrl")),
            IsType(("site", "id"), "string"),
        ),
    ),
    "sharepoint.list_drives": (
        "sharepoint_list_drives",
        {"site_id": "root"},
        (
            StatusSuccess(),
            HasKeys((), ("drives", "truncated")),
            EachHasKeys(("drives",), ("id", "name")),
            IsType(("truncated",), "boolean"),
        ),
    ),
    "sharepoint.list_items": (
        "sharepoint_list_items",
        {"site_id": "root", "top": 1},
        (
            StatusSuccess(),
            HasKeys((), ("items", "truncated")),
            EachHasKeys(("items",), ("id", "name")),
            IsType(("truncated",), "boolean"),
            Length(("items",), max=1),
        ),
    ),
    "sharepoint.list_lists": (
        "sharepoint_list_lists",
        {"site_id": "root", "top": 1},
        (
            StatusSuccess(),
            HasKeys((), ("lists", "truncated")),
            EachHasKeys(("lists",), ("id", "name")),
            IsType(("truncated",), "boolean"),
            Length(("lists",), max=1),
        ),
    ),
    # A Planner list takes no page size and projects a page too large for the
    # output limit down to each item's id and label (and a plan's web_url).
    "planner.list_my_plans": (
        "planner_list_my_plans",
        {},
        (
            StatusSuccess(),
            HasKeys((), ("plans", "next_link", "truncated")),
            EachHasKeys(("plans",), ("id", "web_url")),
            IsType(("truncated",), "boolean"),
        ),
    ),
    "planner.list_my_tasks": (
        "planner_list_my_tasks",
        {},
        (
            StatusSuccess(),
            HasKeys((), ("tasks", "next_link", "truncated")),
            EachHasKeys(("tasks",), ("id",)),
            IsType(("truncated",), "boolean"),
        ),
    ),
    "planner.list_my_groups": (
        "planner_list_my_groups",
        {},
        (
            StatusSuccess(),
            HasKeys((), ("groups", "next_link", "truncated")),
            EachHasKeys(("groups",), ("id",)),
            IsType(("truncated",), "boolean"),
        ),
    ),
    "linear.list_teams": _linear_pin(
        "linear_list_teams", "teams", ("id", "key", "name")
    ),
    "linear.list_projects": _linear_pin(
        "linear_list_projects",
        "projects",
        ("id", "name", "status", "progress", "url", "teams"),
    ),
    "linear.search_users": _linear_pin(
        "linear_search_users", "users", ("id", "name", "email", "active")
    ),
    "linear.search_issues": _linear_pin(
        "linear_search_issues",
        "issues",
        (
            "id",
            "identifier",
            "title",
            "priority",
            "url",
            "createdAt",
            "updatedAt",
            "state",
            "assignee",
            "team",
            "project",
            "labels",
        ),
    ),
    "intercom.list_conversations": (
        "intercom_list_conversations",
        {"state": "all", "limit": 1},
        (
            StatusSuccess(),
            HasKeys((), ("conversations", "total_count", "has_more", "next_cursor")),
            EachHasKeys(
                ("conversations",),
                (
                    "id",
                    "state",
                    "title",
                    "subject",
                    "preview",
                    "created_at",
                    "updated_at",
                ),
            ),
            Length(("conversations",), max=1),
            IsType(("has_more",), "boolean"),
        ),
    ),
    "deputy.list_companies": (
        "deputy_list_resource",
        {"resource": "Company"},
        (
            StatusSuccess(),
            HasKeys((), ("records", "truncated")),
            IsType(("records",), "array"),
            IsType(("truncated",), "boolean"),
        ),
    ),
    "deputy.employee_info": (
        "deputy_resource_info",
        {"resource": "Employee"},
        (StatusSuccess(), HasKeys((), ("info",)), IsType(("info",), "object")),
    ),
    "employment-hero.list_organisations": (
        "employment_hero_list_organisations",
        {"item_per_page": 1},
        (
            StatusSuccess(),
            HasKeys((), ("organisations",)),
            IsType(("organisations",), "object"),
            HasKeys(("organisations",), ("items",)),
            IsType(("organisations", "items"), "array"),
            Length(("organisations", "items"), max=1),
        ),
    ),
    "xero.list_accounts": _xero_unpaged_pin("accounts"),
    "xero.list_contact_groups": _xero_unpaged_pin("contact_groups"),
    "xero.list_items": _xero_unpaged_pin("items"),
    "xero.list_tax_rates": _xero_unpaged_pin("tax_rates"),
    "xero.list_tracking_categories": _xero_unpaged_pin("tracking_categories"),
    "xero.list_contacts": _xero_paged_pin("contacts"),
    "xero.list_invoices": _xero_paged_pin("invoices"),
    "xero.list_credit_notes": _xero_paged_pin("credit_notes"),
    "xero.list_bank_transactions": _xero_paged_pin("bank_transactions"),
    "xero.list_payments": _xero_paged_pin("payments"),
    "xero.list_quotes": _xero_paged_pin("quotes", capped=False),
    "xero.list_manual_journals": _xero_paged_pin("manual_journals"),
}


# HubSpot's Tier 1 read checks (#3572): the tool, its arguments, and every
# assertion, so dropping an item key or a whole assertion fails here.
_HUBSPOT_READS: Mapping[str, tuple[str, dict[str, Any], tuple[Assertion, ...]]] = {
    "hubspot.search_contacts": (
        "search_crm_objects",
        {"objectType": "CONTACT", "limit": 1, "properties": ["email"]},
        (
            HasKeys((), ("results", "total", "offset")),
            EachHasKeys(("results",), ("id", "properties")),
            Length(("results",), max=1),
        ),
    ),
    # The empty result's shape was not observed, so only ``results`` is
    # asserted: an absent ``instructions`` key would fail a passing check.
    "hubspot.query_crm_data": (
        "query_crm_data",
        {"sql": "SELECT hs_object_id FROM contacts LIMIT 1"},
        (
            HasKeys((), ("results",)),
            IsType(("results",), "array"),
        ),
    ),
    "hubspot.object_types": (
        "discover_hubspot_schema",
        {
            "query": {
                "_queryType": "GET_OBJECT_TYPES",
                "typeNameFilter": ["CONTACT"],
                "limit": 1,
            }
        },
        (
            HasKeys((), ("objectTypeResults",)),
            HasKeys(("objectTypeResults",), ("results", "hasMore")),
            EachHasKeys(
                ("objectTypeResults", "results"),
                ("typeName", "objectTypeId", "readAccess", "writeAccess"),
            ),
            IsType(("objectTypeResults", "hasMore"), "boolean"),
            Length(("objectTypeResults", "results"), min=1),
            Equals(("objectTypeResults", "results", 0, "typeName"), "CONTACT"),
        ),
    ),
    "hubspot.search_properties": (
        "search_properties",
        {"objectType": "CONTACT", "keywords": ["email"]},
        (
            HasKeys((), ("results",)),
            EachHasKeys(("results",), ("name", "label")),
            Length(("results",), min=1),
        ),
    ),
    "hubspot.search_segments": (
        "query_segments",
        {"operation": {"type": "SearchSegmentsQuery", "limit": 1}},
        (HasKeys(("result",), ("segments",)),),
    ),
}


def _read_check(check_id: str) -> ReadCheck:
    app_id = check_id.split(".", 1)[0]
    check = next(c for c in CONNECTOR_CHECKS[app_id].reads if c.id == check_id)
    assert isinstance(check, ReadCheck)
    return check


def _assert_check_output(check_id: str, result: str) -> None:
    check = _read_check(check_id)
    raw = {
        "content": [{"type": "text", "text": result}],
        "structured_content": None,
        "is_error": False,
    }
    assertions = resolve_expected(check.assertions, _FIXTURES)
    assert judge_result(raw, check.payload, assertions) == (None, None)


def _assert_check_fails(check_id: str, result: str, reason: str) -> None:
    check = _read_check(check_id)
    raw = {
        "content": [{"type": "text", "text": result}],
        "structured_content": None,
        "is_error": False,
    }
    assertions = resolve_expected(check.assertions, _FIXTURES)
    assert judge_result(raw, check.payload, assertions) == (
        FailureClass.ASSERTION_FAILED,
        reason,
    )


@pytest.mark.parametrize(("check_id", "expected"), sorted(_TIER_2_READS.items()))
def test_a_tier_2_read_check_is_declared_exactly(
    check_id: str, expected: tuple[str, dict[str, Any], tuple[Assertion, ...]]
) -> None:
    tool, arguments, assertions = expected
    check = _read_check(check_id)
    assert (check.tool, dict(check.arguments)) == (tool, arguments)
    assert check.assertions == assertions


@pytest.mark.parametrize(("check_id", "expected"), sorted(_HUBSPOT_READS.items()))
def test_a_hubspot_read_check_is_declared_exactly(
    check_id: str, expected: tuple[str, dict[str, Any], tuple[Assertion, ...]]
) -> None:
    tool, arguments, assertions = expected
    check = _read_check(check_id)
    assert (check.tool, dict(check.arguments)) == (tool, arguments)
    assert check.assertions == assertions


# Result shapes each check accepts, with synthetic values: an empty list too,
# except where every portal has a match.
_HUBSPOT_OUTPUTS: Mapping[str, tuple[dict[str, Any], ...]] = {
    "hubspot.search_contacts": (
        {
            "results": [
                {
                    "id": 1,
                    "properties": {"hs_object_id": "1", "email": "a@example.com"},
                    "displayName": "A",
                }
            ],
            "total": 3,
            "urlTemplate": "https://app.hubspot.com/contacts/0/record/0-1/{id}",
            "offset": 1,
        },
        # An empty result omits urlTemplate, which no check asserts.
        {"results": [], "total": 0, "offset": 0},
    ),
    "hubspot.query_crm_data": (
        {
            "results": [
                {
                    "content": (
                        '{"objectTypeId":"0-1","properties":{"hs_object_id":"1"}}'
                    )
                }
            ],
            "instructions": "Showing 1 of 3 total objects.",
        },
        {"results": []},
    ),
    "hubspot.object_types": (
        {
            "objectTypeResults": {
                "results": [
                    {
                        "typeName": "CONTACT",
                        "objectTypeId": "0-1",
                        "typeCategory": "HUBSPOT_DEFINED_OBJECTS",
                        "readAccess": "AVAILABLE",
                        "writeAccess": "AVAILABLE",
                        "activationStatus": "PERMANENTLY_ACTIVATED",
                    }
                ],
                "hasMore": False,
            }
        },
    ),
    "hubspot.search_properties": (
        {
            "results": [
                {
                    "name": "email",
                    "label": "Email",
                    "description": "A contact's email address",
                    "matchScore": 0.87,
                }
            ]
        },
    ),
    "hubspot.search_segments": (
        {
            "result": {
                "segments": {
                    "segments": [
                        {
                            "listId": 9,
                            "objectTypeId": "0-1",
                            "listType": "STATIC",
                            "name": "Example",
                        }
                    ],
                    "totalCount": 1,
                    "hasMore": False,
                },
                "__typename": "x",
                "type": "SegmentSearchResult",
            }
        },
        {"result": {"segments": {}}},
    ),
}


@pytest.mark.parametrize(
    ("check_id", "payload"),
    [
        (check_id, payload)
        for check_id, payloads in sorted(_HUBSPOT_OUTPUTS.items())
        for payload in payloads
    ],
)
def test_a_hubspot_read_output_satisfies_its_declaration(
    check_id: str, payload: dict[str, Any]
) -> None:
    _assert_check_output(check_id, json.dumps(payload))


@pytest.mark.parametrize(
    ("check_id", "payload", "reason"),
    [
        (
            "hubspot.object_types",
            {"objectTypeResults": {"results": [], "hasMore": False}},
            "length",
        ),
        (
            "hubspot.object_types",
            {
                "objectTypeResults": {
                    "results": [
                        {
                            "typeName": "COMPANY",
                            "objectTypeId": "0-2",
                            "readAccess": "AVAILABLE",
                            "writeAccess": "AVAILABLE",
                        }
                    ],
                    "hasMore": False,
                }
            },
            "equals",
        ),
        ("hubspot.search_properties", {"results": []}, "length"),
    ],
)
def test_a_hubspot_check_fails_without_what_every_portal_has(
    check_id: str, payload: dict[str, Any], reason: str
) -> None:
    _assert_check_fails(check_id, json.dumps(payload), reason)


@pytest.mark.parametrize(
    "app_id",
    [
        "deputy",
        "employment-hero",
        "google-ads",
        "intercom",
        "linear",
        "outlook",
        "planner",
        "sharepoint",
        "xero",
        "zendesk",
        "zoom",
    ],
)
def test_a_tier_2_l3_check_is_the_probe_derived_one(app_id: str) -> None:
    expected = l3_read_check(app_id)
    assert expected is not None

    l3_checks = [
        check
        for check in CONNECTOR_CHECKS[app_id].reads
        if check.id == f"{app_id}.l3_probe"
    ]
    assert l3_checks == [expected]


@pytest.mark.parametrize("app_id", ["acme", "granola"])
def test_the_declared_l3_helper_rejects_anything_but_a_read_check(app_id: str) -> None:
    with pytest.raises(DeclarationError, match=app_id):
        _declared_l3(app_id)


def test_zoom_create_meeting_has_no_cleanup_tool() -> None:
    assert [
        exemption.tool
        for exemption in CONNECTOR_CHECKS["zoom"].exemptions
        if exemption.reason == WRITE_WITHOUT_CLEANUP
    ] == ["zoom_create_meeting"]


def _list_payload(list_key: str, item: dict[str, Any], count: int) -> dict[str, Any]:
    return {list_key: [item] * count, "meta": {"has_more": False}}


@pytest.mark.parametrize(
    ("check_id", "list_key", "item", "tool"),
    [
        (
            "zendesk.list_tickets",
            "tickets",
            {
                "id": 7,
                "subject": "Printer down",
                "status": "open",
                "priority": "normal",
                "requester_id": 3,
                "tags": ["hardware"],
            },
            zendesk.zendesk_list_tickets,
        ),
        (
            "zendesk.list_organizations",
            "organizations",
            {"id": 4, "name": "Acme", "domain_names": ["acme.example"]},
            zendesk.zendesk_list_organizations,
        ),
        (
            "zendesk.list_macros",
            "macros",
            {"id": 5, "title": "Close ticket", "description": "Closes it"},
            zendesk.zendesk_list_macros,
        ),
    ],
)
@pytest.mark.parametrize("count", [0, 1])
def test_a_zendesk_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
    check_id: str,
    list_key: str,
    item: dict[str, Any],
    tool: Callable[..., str],
    count: int,
) -> None:
    monkeypatch.setattr(
        zendesk,
        "_request",
        lambda *args, **kwargs: _list_payload(list_key, item, count),
    )

    _assert_check_output(check_id, tool(limit=1))


@pytest.mark.parametrize("count", [0, 1])
def test_zoom_list_meetings_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    monkeypatch.setattr(
        zoom,
        "_request",
        lambda *args, **kwargs: {
            "meetings": [{"id": 111, "topic": "Sync"}] * count,
            "next_page_token": "",
        },
    )

    _assert_check_output("zoom.list_meetings", zoom.zoom_list_meetings())


@pytest.mark.parametrize("count", [0, 1])
def test_google_ads_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    monkeypatch.setattr(
        google_ads,
        "_request",
        lambda *args, **kwargs: {"resourceNames": ["customers/1112223333"] * count},
    )

    _assert_check_output(
        "google-ads.list_accessible_customers",
        google_ads.google_ads_list_accessible_customers(),
    )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        # A rejected token or a malformed request: the tool catches it and
        # reports it in its own envelope, with the MCP error flag unset.
        ({"status": "error", "message": "PERMISSION_DENIED"}, "status_success"),
        ({"status": "success"}, "has_keys"),
        ({"status": "success", "customer_ids": "1112223333"}, "is_type"),
    ],
)
def test_google_ads_list_check_fails_on_another_shape(
    payload: dict[str, Any], reason: str
) -> None:
    check = _read_check("google-ads.list_accessible_customers")
    raw = {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structured_content": None,
        "is_error": False,
    }

    assert judge_result(raw, check.payload, check.assertions) == (
        FailureClass.ASSERTION_FAILED,
        reason,
    )


def _call_declared(check_id: str, tool: Callable[..., str]) -> str:
    """``tool`` called with the arguments its check declares."""
    return tool(**_read_check(check_id).arguments)


_OUTLOOK_LISTS: Mapping[str, tuple[str, Callable[..., str]]] = {
    "outlook.list_messages": ("messages", outlook.outlook_list_messages),
    "outlook.list_events": ("events", outlook.outlook_list_events),
    "outlook.list_contacts": ("contacts", outlook.outlook_list_contacts),
}


@pytest.mark.parametrize("check_id", sorted(_OUTLOOK_LISTS))
@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("next_link", [None, "https://graph.microsoft.com/v1.0/me/x"])
def test_an_outlook_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
    check_id: str,
    count: int,
    next_link: str | None,
) -> None:
    _, tool = _OUTLOOK_LISTS[check_id]
    page: dict[str, Any] = {"value": [{"id": "AAMkAD1"}] * count}
    if next_link is not None:
        page["@odata.nextLink"] = next_link
    monkeypatch.setattr(outlook, "_graph_request", lambda *args, **kwargs: page)

    _assert_check_output(check_id, _call_declared(check_id, tool))


def test_outlook_get_profile_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        outlook,
        "_graph_request",
        lambda *args, **kwargs: {
            "id": "00000000-0000-0000-0000-000000000001",
            "userPrincipalName": "qa@example.onmicrosoft.com",
            "displayName": None,
        },
    )

    _assert_check_output(
        "outlook.get_profile",
        _call_declared("outlook.get_profile", outlook.outlook_get_profile),
    )


@pytest.mark.parametrize("check_id", sorted(_OUTLOOK_LISTS))
@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        # A tool's error envelope, whether or not the tool returns one:
        # outlook_list_messages raises instead (see the test below).
        (lambda key: {"status": "error", "message": "403"}, "status_success"),
        (lambda key: {"status": "success", key: []}, "has_keys"),
        (
            lambda key: {"status": "success", key: [{}], "next_link": None},
            "each_has_keys",
        ),
        (
            lambda key: {
                "status": "success",
                key: [{"id": "a"}, {"id": "b"}],
                "next_link": None,
            },
            "length",
        ),
    ],
)
def test_an_outlook_list_check_fails_on_another_shape(
    check_id: str, payload: Callable[[str], dict[str, Any]], reason: str
) -> None:
    list_key, _ = _OUTLOOK_LISTS[check_id]

    _assert_check_fails(check_id, json.dumps(payload(list_key)), reason)


def _graph_fails(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("Graph returned 403")


def test_outlook_list_messages_raises_on_a_graph_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP reports the ToolError with its error flag set, so the check fails
    as a call failure rather than on its assertions."""
    monkeypatch.setattr(outlook, "_graph_request", _graph_fails)

    with pytest.raises(ToolError):
        _call_declared("outlook.list_messages", outlook.outlook_list_messages)


@pytest.mark.parametrize("check_id", ["outlook.list_events", "outlook.list_contacts"])
def test_an_outlook_list_error_envelope_fails_its_check(
    monkeypatch: pytest.MonkeyPatch, check_id: str
) -> None:
    _, tool = _OUTLOOK_LISTS[check_id]
    monkeypatch.setattr(outlook, "_graph_request", _graph_fails)

    _assert_check_fails(check_id, _call_declared(check_id, tool), "status_success")


@pytest.mark.parametrize(
    ("user", "reason"),
    [
        ({"userPrincipalName": "qa@x"}, "has_keys"),
        ({"id": 1, "userPrincipalName": "qa@x"}, "is_type"),
    ],
)
def test_outlook_get_profile_check_fails_on_another_user(
    user: dict[str, Any], reason: str
) -> None:
    _assert_check_fails(
        "outlook.get_profile",
        json.dumps({"status": "success", "user": user}),
        reason,
    )


def test_outlook_send_message_has_no_cleanup_tool() -> None:
    assert [
        exemption.tool
        for exemption in CONNECTOR_CHECKS["outlook"].exemptions
        if exemption.reason == WRITE_WITHOUT_CLEANUP
    ] == ["outlook_send_message"]


_SHAREPOINT_LISTS: Mapping[str, tuple[str, Callable[..., str]]] = {
    "sharepoint.list_drives": ("drives", sharepoint.sharepoint_list_drives),
    "sharepoint.list_items": ("items", sharepoint.sharepoint_list_items),
    "sharepoint.list_lists": ("lists", sharepoint.sharepoint_list_lists),
}


@pytest.mark.parametrize("check_id", sorted(_SHAREPOINT_LISTS))
@pytest.mark.parametrize(
    ("count", "next_link"),
    [
        (0, None),
        (1, None),
        # A non-empty library read with top 1: Graph links a next page and
        # the tool reports truncated=True.
        (1, "https://graph.microsoft.com/v1.0/sites/root/next"),
    ],
)
def test_a_sharepoint_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, check_id: str, count: int, next_link: str | None
) -> None:
    _, tool = _SHAREPOINT_LISTS[check_id]
    paths: list[str] = []
    item = {"id": "01ABC", "name": "Documents"}

    def _graph_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        paths.append(path)
        page: dict[str, Any] = {"value": [item] * count}
        if next_link is not None:
            page["@odata.nextLink"] = next_link
        return page

    monkeypatch.setattr(sharepoint, "_graph_request", _graph_request)
    # list_drives pages up to 200 items, so it follows the link: the next
    # page holds one more item and no further link.
    monkeypatch.setattr(
        sharepoint, "_graph_get_absolute", lambda *args, **kwargs: {"value": [item]}
    )

    result = _call_declared(check_id, tool)

    _assert_check_output(check_id, result)
    assert paths and all(path.startswith("/sites/root/") for path in paths)
    if next_link is not None and check_id != "sharepoint.list_drives":
        assert json.loads(result)["truncated"] is True


def test_sharepoint_get_root_site_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sharepoint,
        "_graph_request",
        lambda *args, **kwargs: {
            "id": "contoso.sharepoint.com,1111,2222",
            "webUrl": "https://contoso.sharepoint.com",
            "name": "",
        },
    )

    _assert_check_output(
        "sharepoint.get_root_site",
        _call_declared("sharepoint.get_root_site", sharepoint.sharepoint_get_root_site),
    )


def test_sharepoint_get_site_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[str] = []

    def _graph_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        paths.append(path)
        return {
            "id": "contoso.sharepoint.com,1111,2222",
            "webUrl": "https://contoso.sharepoint.com",
        }

    monkeypatch.setattr(sharepoint, "_graph_request", _graph_request)

    _assert_check_output(
        "sharepoint.get_site",
        _call_declared("sharepoint.get_site", sharepoint.sharepoint_get_site),
    )
    assert paths == ["/sites/root"]


def test_a_halved_sharepoint_list_still_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page too large for the output limit is halved down to no items, with
    a message; the check bounds no lower count and allows extra keys."""
    monkeypatch.setattr(
        sharepoint,
        "_graph_request",
        lambda *args, **kwargs: {
            "value": [{"id": "b!1", "name": "Documents", "description": "x" * 400}] * 2
        },
    )
    monkeypatch.setattr(sharepoint, "get_tool_max_output_length", lambda: 300)

    result = _call_declared("sharepoint.list_drives", sharepoint.sharepoint_list_drives)

    payload = json.loads(result)
    assert (payload["drives"], payload["truncated"]) == ([], True)
    assert "message" in payload
    _assert_check_output("sharepoint.list_drives", result)


@pytest.mark.parametrize(
    ("check_id", "payload", "reason"),
    [
        (
            "sharepoint.list_drives",
            {"status": "error", "message": "accessDenied"},
            "status_success",
        ),
        ("sharepoint.list_drives", {"status": "success", "drives": []}, "has_keys"),
        (
            "sharepoint.list_drives",
            {"status": "success", "drives": [{"id": "b!1"}], "truncated": False},
            "each_has_keys",
        ),
        (
            "sharepoint.list_drives",
            {"status": "success", "drives": [], "truncated": "false"},
            "is_type",
        ),
        (
            "sharepoint.list_items",
            {
                "status": "success",
                "items": [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}],
                "truncated": True,
            },
            "length",
        ),
        (
            "sharepoint.list_lists",
            {
                "status": "success",
                "lists": [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}],
                "truncated": True,
            },
            "length",
        ),
        (
            "sharepoint.get_root_site",
            {"status": "success", "site": {"id": "x"}},
            "has_keys",
        ),
    ],
)
def test_a_sharepoint_check_fails_on_another_shape(
    check_id: str, payload: dict[str, Any], reason: str
) -> None:
    _assert_check_fails(check_id, json.dumps(payload), reason)


def test_the_sharepoint_writes_stay_pending() -> None:
    """An upload's file is a document-library list item, which
    sharepoint_delete_list_item may be able to delete by a list item id the
    upload does not return; whether a check can clean up that way is not
    settled, so no sharepoint write takes the permanent no-cleanup reason."""
    assert not [
        exemption.tool
        for exemption in CONNECTOR_CHECKS["sharepoint"].exemptions
        if exemption.reason == WRITE_WITHOUT_CLEANUP
    ]


# Each Planner list check: its list key, its tool, and an item whose last field
# is long enough to push the page past the output limit.
_PLANNER_LISTS: Mapping[str, tuple[str, Callable[..., str], dict[str, Any]]] = {
    "planner.list_my_plans": (
        "plans",
        planner.planner_list_my_plans,
        {"id": "plan-1", "title": "Roadmap", "owner": "group-1"},
    ),
    "planner.list_my_tasks": (
        "tasks",
        planner.planner_list_my_tasks,
        {"id": "task-1", "title": "Ship", "planId": "plan-1"},
    ),
    "planner.list_my_groups": (
        "groups",
        planner.planner_list_my_groups,
        {"id": "group-1", "displayName": "Team", "description": "Our team"},
    ),
}


@pytest.mark.parametrize("check_id", sorted(_PLANNER_LISTS))
@pytest.mark.parametrize("count", [0, 1])
def test_a_planner_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, check_id: str, count: int
) -> None:
    _, tool, item = _PLANNER_LISTS[check_id]
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        planner, "_graph_request", lambda *args, **kwargs: {"value": [item] * count}
    )

    _assert_check_output(check_id, _call_declared(check_id, tool))


@pytest.mark.parametrize("check_id", sorted(_PLANNER_LISTS))
@pytest.mark.parametrize(
    ("limit", "projection"), [(3000, "summary_fields"), (1000, "identity_and_label")]
)
def test_a_projected_planner_list_still_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, check_id: str, limit: int, projection: str
) -> None:
    """A page over the output limit keeps the summary fields, then only each
    item's id and label: the per-item keys hold at both levels."""
    list_key, tool, item = _PLANNER_LISTS[check_id]
    *_, long_field = item
    oversized = {**item, long_field: "x" * 2000, "notes": "y" * 4000}
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        planner, "_graph_request", lambda *args, **kwargs: {"value": [oversized]}
    )
    monkeypatch.setattr(planner, "get_tool_max_output_length", lambda: limit)

    result = _call_declared(check_id, tool)

    payload = json.loads(result)
    assert payload["projection"] == projection
    assert len(payload[list_key]) == 1
    _assert_check_output(check_id, result)


@pytest.mark.parametrize(
    ("check_id", "payload", "reason"),
    [
        (
            "planner.list_my_groups",
            {"status": "error", "message": "Planner cannot list the groups"},
            "status_success",
        ),
        (
            "planner.list_my_tasks",
            {"status": "success", "tasks": [], "truncated": False},
            "has_keys",
        ),
        (
            "planner.list_my_plans",
            {
                "status": "success",
                "plans": [{"id": "plan-1"}],
                "next_link": None,
                "truncated": False,
            },
            "each_has_keys",
        ),
        (
            "planner.list_my_tasks",
            {"status": "success", "tasks": [], "next_link": None, "truncated": 0},
            "is_type",
        ),
    ],
)
def test_a_planner_check_fails_on_another_shape(
    check_id: str, payload: dict[str, Any], reason: str
) -> None:
    _assert_check_fails(check_id, json.dumps(payload), reason)


def test_no_planner_write_is_exempt_as_without_cleanup() -> None:
    assert not [
        exemption.tool
        for exemption in CONNECTOR_CHECKS["planner"].exemptions
        if exemption.reason == WRITE_WITHOUT_CLEANUP
    ]


# The Google family (issue #3567): the read checks of the tenant the seed
# script (xorbitsai/xagent-saas#2533) fills, with the tool, its arguments and
# every assertion, so dropping an argument, an assertion or an expected seed
# value fails here. ``google-calendar`` and ``google-analytics`` keep their
# probe-derived L3 call instead (``_declared_l3``).
_GOOGLE_READS: Mapping[str, tuple[str, dict[str, Any], tuple[Assertion, ...]]] = {
    "gmail.search_seed_message": (
        "gmail_search_messages",
        {"label_ids": [FixtureRef("gmail.seed_label_id")], "max_results": 10},
        (
            StatusSuccess(),
            HasKeys((), ("messages",)),
            Length(("messages",), 1, 1),
            EachHasKeys(
                ("messages",), ("id", "threadId", "snippet", "subject", "from")
            ),
            Equals(("messages", 0, "subject"), "xagent smoke seed message"),
        ),
    ),
    "google-analytics.run_report": (
        "google_analytics_run_report",
        {
            "property_id": FixtureRef("google-analytics.property_id"),
            "metrics": ["sessions"],
            "date_ranges": [{"start_date": "7daysAgo", "end_date": "yesterday"}],
        },
        (
            StatusSuccess(),
            HasKeys(
                (),
                (
                    "query",
                    "metric_headers",
                    "totals",
                    "rows",
                    "row_count",
                    "returned_count",
                    "has_more",
                    "truncated",
                ),
            ),
            HasKeys(
                ("query",),
                ("property_id", "metrics", "date_ranges", "limit", "offset"),
            ),
            IsType(("rows",), "array"),
        ),
    ),
    "google-docs.read_fixture": (
        "google_docs_get_document",
        {"document_id": FixtureRef("google-docs.document_id")},
        (
            StatusSuccess(),
            HasKeys((), ("document_id", "title", "content")),
            Equals(("document_id",), FixtureRef("google-docs.document_id")),
            Equals(("title",), "xagent-smoke-seed-doc"),
            Contains(("content",), "xagent smoke seed document"),
        ),
    ),
    "google-drive.read_seed_file": (
        "google_drive_get_file_content",
        {"file_id": FixtureRef("google-docs.document_id")},
        (
            StatusSuccess(),
            HasKeys((), ("file", "content", "encoding", "truncated")),
            HasKeys(("file",), ("id", "name", "mimeType")),
            Equals(("file", "id"), FixtureRef("google-docs.document_id")),
            Equals(("file", "name"), "xagent-smoke-seed-doc"),
            Equals(("encoding",), "utf-8"),
            Contains(("content",), "xagent smoke seed document"),
        ),
    ),
    "google-drive.search_seed_folder": (
        "google_drive_search",
        {
            "query": (
                "name = 'xagent-smoke-seed' and "
                "mimeType = 'application/vnd.google-apps.folder' and "
                "trashed = false"
            ),
            "max_results": 10,
        },
        (
            StatusSuccess(),
            HasKeys((), ("files", "truncated")),
            Length(("files",), 1, 1),
            EachHasKeys(("files",), ("id", "name", "mimeType", "modifiedTime")),
            Equals(("files", 0, "id"), FixtureRef("google-drive.seed_folder_id")),
        ),
    ),
    "google-sheets.read_fixture": (
        "google_sheets_get_spreadsheet",
        {"spreadsheet_id": FixtureRef("google-sheets.spreadsheet_id")},
        (
            StatusSuccess(),
            HasKeys((), ("spreadsheet_id", "title", "url", "sheets")),
            Equals(("spreadsheet_id",), FixtureRef("google-sheets.spreadsheet_id")),
            Equals(("title",), "xagent-smoke-seed-sheet"),
            Length(("sheets",), 1, 1),
            EachHasKeys(
                ("sheets",), ("sheet_id", "title", "row_count", "column_count")
            ),
        ),
    ),
    "google-sheets.read_seed_range": (
        "google_sheets_read_range",
        {
            "spreadsheet_id": FixtureRef("google-sheets.spreadsheet_id"),
            "range_name": "A1:B3",
        },
        (
            StatusSuccess(),
            HasKeys((), ("range", "values", "truncated")),
            Length(("values",), 3, 3),
            Equals(("values", 0, 0), "item"),
            Equals(("values", 0, 1), "quantity"),
            Equals(("values", 1, 0), "alpha"),
            Equals(("values", 1, 1), "1"),
            Equals(("values", 2, 0), "beta"),
            Equals(("values", 2, 1), "2"),
        ),
    ),
    "google-slides.read_fixture": (
        "google_slides_get_presentation",
        {"presentation_id": FixtureRef("google-slides.presentation_id")},
        (
            StatusSuccess(),
            HasKeys((), ("presentation_id", "title", "slide_count", "slides")),
            Equals(("presentation_id",), FixtureRef("google-slides.presentation_id")),
            Equals(("title",), "xagent-smoke-seed-slides"),
            Equals(("slide_count",), 1),
        ),
    ),
}

_GOOGLE_APP_MODULES = {
    "gmail": gmail,
    "google-analytics": google_analytics,
    "google-calendar": google_calendar,
    "google-docs": google_docs,
    "google-drive": google_drive,
    "google-sheets": google_sheets,
    "google-slides": google_slides,
}

# The fixture keys the Google checks use, pinned here so that adding or
# dropping one shows in the diff. The tenant registry
# (xorbitsai/xagent-saas#2627) records each of them for the Google seed layout.
_GOOGLE_FIXTURE_KEYS_USED = frozenset(
    {
        "gmail.seed_label_id",
        "google-analytics.property_id",
        "google-docs.document_id",
        "google-drive.seed_folder_id",
        "google-drive.write_folder_id",
        "google-drive.write_sweep_query",
        "google-sheets.spreadsheet_id",
        "google-slides.presentation_id",
    }
)


class _Fixtures:
    """The seeds' values, as the identity adapter would resolve them."""

    _VALUES = {
        "gmail.seed_label_id": "Label_1",
        "github.issue_number": 1,
        "github.login": "xagent-app",
        "github.pr_branch": "xagent-smoke-seed",
        "github.pull_number": 2,
        "github.read_repo": "xagent-smoke/smoke-read",
        "github.seed_file_path": "seed.md",
        "google-analytics.property_id": "563334905",
        "google-docs.document_id": "doc-1",
        "google-drive.seed_folder_id": "folder-1",
        "google-drive.write_folder_id": "folder-write",
        "google-drive.write_sweep_query": (
            "'folder-write' in parents and trashed = false"
        ),
        "google-sheets.spreadsheet_id": "sheet-1",
        "google-slides.presentation_id": "deck-1",
    }

    def resolve_fixture(self, key: str) -> FixtureValue | None:
        return self._VALUES.get(key)


_FIXTURES = _Fixtures()


def _fixture_keys(step: ReadCheck | ToolStep) -> Iterator[FixtureRef]:
    yield from _fixture_refs(step.arguments)
    for assertion in step.assertions:
        if isinstance(assertion, Equals):
            yield from _fixture_refs(assertion.value)


def _google_fixture_keys() -> Iterator[FixtureRef]:
    """Every fixture key a Google declaration names: a read check's, a write
    check's (its pin's expected included), and its sweep's."""
    for app_id in _GOOGLE_APP_MODULES:
        checks = CONNECTOR_CHECKS[app_id]
        for read in checks.reads:
            if isinstance(read, ReadCheck):
                yield from _fixture_keys(read)
        for write in checks.writes:
            yield from _fixture_refs(write.tenant_pin.expected)
            for step in (
                write.create,
                write.verify,
                write.cleanup,
                write.cleanup_guard,
            ):
                if step is not None:
                    yield from _fixture_keys(step)
        if checks.sweep is not None:
            yield from _fixture_refs(checks.sweep.tenant_pin.expected)
            yield from _fixture_keys(checks.sweep.find)
            yield from _fixture_keys(checks.sweep.delete)


@pytest.mark.parametrize(("check_id", "expected"), sorted(_GOOGLE_READS.items()))
def test_a_google_read_check_is_declared_exactly(
    check_id: str, expected: tuple[str, dict[str, Any], tuple[Assertion, ...]]
) -> None:
    tool, arguments, assertions = expected
    check = _read_check(check_id)
    assert (check.tool, dict(check.arguments), check.assertions) == (
        tool,
        arguments,
        assertions,
    )


@pytest.mark.parametrize(
    ("app_id", "check_ids"),
    [
        ("gmail", ["gmail.search_seed_message"]),
        (
            "google-analytics",
            ["google-analytics.l3_probe", "google-analytics.run_report"],
        ),
        ("google-calendar", ["google-calendar.l3_probe"]),
        ("google-docs", ["google-docs.read_fixture"]),
        (
            "google-drive",
            ["google-drive.search_seed_folder", "google-drive.read_seed_file"],
        ),
        (
            "google-sheets",
            ["google-sheets.read_fixture", "google-sheets.read_seed_range"],
        ),
        ("google-slides", ["google-slides.read_fixture"]),
    ],
)
def test_a_google_connector_declares_exactly_its_reads(
    app_id: str, check_ids: list[str]
) -> None:
    assert [check.id for check in CONNECTOR_CHECKS[app_id].reads] == check_ids


@pytest.mark.parametrize("app_id", ["google-analytics", "google-calendar"])
def test_a_google_l3_check_is_the_probe_derived_one(app_id: str) -> None:
    expected = l3_read_check(app_id)
    assert expected is not None

    l3_checks = [
        check
        for check in CONNECTOR_CHECKS[app_id].reads
        if check.id == f"{app_id}.l3_probe"
    ]
    assert l3_checks == [expected]


@pytest.mark.parametrize("app_id", ["google-docs", "google-sheets", "google-slides"])
def test_a_google_document_probe_still_needs_a_fixture(app_id: str) -> None:
    assert l3_read_check(app_id) == UnprobeableCheck(
        f"{app_id}.l3_probe", "needs_fixture"
    )


def test_the_google_checks_use_exactly_these_fixture_keys() -> None:
    keys = {ref.key for ref in _google_fixture_keys()}

    assert keys == _GOOGLE_FIXTURE_KEYS_USED


@pytest.mark.parametrize("app_id", sorted(_GOOGLE_APP_MODULES))
def test_a_google_fixture_is_bound_to_a_parameter_of_its_tool(app_id: str) -> None:
    """A fixture in the arguments names a parameter of the tool, at any depth
    (Gmail's is the one item of ``label_ids``), and every required argument is
    pinned: an omitted one would make the tool prompt for it."""
    module = _GOOGLE_APP_MODULES[app_id]
    for check in CONNECTOR_CHECKS[app_id].reads:
        if not isinstance(check, ReadCheck):
            continue
        tool = getattr(module, check.tool)
        parameters = inspect.signature(tool).parameters
        bound = [
            name for name, value in check.arguments.items() if any(_fixture_refs(value))
        ]
        for name in bound:
            assert name in parameters, (check.tool, name)
        required = {
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        }
        assert required <= set(check.arguments), (check.tool, required)


def test_google_drive_declares_its_write_check() -> None:
    """The pin, the create, the read-back that also guards the cleanup, and
    the verdicts, pinned so that changing a step shows in the diff."""
    pin = TenantPin(
        ToolStep(
            "google_drive_search",
            {
                "query": (
                    "name = 'xagent-smoke-write' and "
                    "mimeType = 'application/vnd.google-apps.folder' and "
                    "'root' in parents and "
                    "trashed = false"
                ),
                "max_results": 10,
            },
            assertions=(Length(("files",), 1, 1),),
        ),
        path=("files", 0, "id"),
        # Drive has no tool that returns the tenant's customer id, so the pin
        # reads the write-area container (#3567): the folder the seed keeps
        # unshared, found by the seed's own rule - its name, its type and the
        # root as its parent - and required to be the registry's id.
        expected=FixtureRef("google-drive.write_folder_id"),
    )
    name = RunScopedName("drive-file")
    created = CreatedRef(("file", "id"))
    probe = "xagent smoke write probe"
    # The guard asserts the name alone, so content that did not land as sent
    # still lets the cleanup delete the file.
    guard = ToolStep(
        "google_drive_get_file_content",
        {"file_id": created},
        assertions=(StatusSuccess(), Equals(("file", "name"), name)),
    )

    assert CONNECTOR_CHECKS["google-drive"].writes == (
        WriteCheck(
            "google-drive.file_round_trip",
            create=ToolStep(
                "google_drive_create_file",
                {
                    "name": name,
                    "content": probe,
                    "mime_type": "text/plain",
                    "parent_id": FixtureRef("google-drive.write_folder_id"),
                },
                assertions=(StatusSuccess(), HasKeys(("file",), ("id",))),
            ),
            verify=ToolStep(
                "google_drive_get_file_content",
                {"file_id": created},
                assertions=(
                    StatusSuccess(),
                    Equals(("file", "name"), name),
                    Equals(("content",), probe),
                ),
            ),
            cleanup=ToolStep(
                "google_drive_delete_file",
                {"file_id": created},
                assertions=(StatusSuccess(),),
            ),
            cleanup_guard=guard,
            tenant_pin=pin,
            # Toby classifies only Jira's writes, so it withholds neither
            # Drive tool from a member.
            expected_verdicts={
                "google_drive_create_file": "allow",
                "google_drive_delete_file": "allow",
            },
        ),
    )


def test_google_drive_declares_its_sweep() -> None:
    checks = CONNECTOR_CHECKS["google-drive"]

    assert checks.sweep == Sweep(
        tenant_pin=checks.writes[0].tenant_pin,
        find=ToolStep(
            "google_drive_search",
            {
                "query": FixtureRef("google-drive.write_sweep_query"),
                "max_results": 100,
            },
            assertions=(StatusSuccess(), HasKeys((), ("files",))),
        ),
        items=("files",),
        name=("name",),
        delete=ToolStep(
            "google_drive_delete_file",
            {"file_id": ItemRef(("id",))},
            assertions=(StatusSuccess(),),
        ),
    )


def test_a_cleanable_write_keeps_a_pending_exemption() -> None:
    """The writes whose own cleanup tool can delete what they create still
    wait for a later check (#3567); none carries the permanent no-cleanup
    reason."""
    cleanable = (
        "google_calendar_create_events",
        "google_calendar_delete_events",
        "google_drive_create_folder",
        "google_sheets_add_sheet",
        "google_sheets_delete_sheet",
        "google_slides_add_slide",
        "google_slides_delete_slide",
    )
    exemptions = {
        exemption.tool: exemption.reason
        for app_id in (
            "google-calendar",
            "google-drive",
            "google-sheets",
            "google-slides",
        )
        for exemption in CONNECTOR_CHECKS[app_id].exemptions
    }

    for tool in cleanable:
        assert exemptions[tool] != WRITE_WITHOUT_CLEANUP, tool
        assert exemptions[tool].endswith("decided in #3567"), tool


# Each Google app's write exemptions that carry a permanent reason instead of a
# pending one. A connector whose writes can all be cleaned up has none.
_GOOGLE_PERMANENT_WRITE_REASONS: Mapping[str, Mapping[str, str]] = {
    "gmail": {"gmail_send_messages": WRITE_WITHOUT_CLEANUP},
    "google-analytics": {},
    "google-calendar": {},
    "google-docs": {"google_docs_create_document": WRITE_WITHOUT_CLEANUP},
    "google-drive": {},
    "google-sheets": {"google_sheets_create_spreadsheet": WRITE_WITHOUT_CLEANUP},
    "google-slides": {
        "google_slides_create_presentation": WRITE_WITHOUT_CLEANUP,
        "google_slides_import_pptx": WRITE_WITHOUT_CLEANUP,
    },
}


@pytest.mark.parametrize("app_id", sorted(_GOOGLE_APP_MODULES))
def test_each_permanent_write_reason_names_its_own_tools(app_id: str) -> None:
    actual = {
        exemption.tool: exemption.reason
        for exemption in CONNECTOR_CHECKS[app_id].exemptions
        if exemption.reason == WRITE_WITHOUT_CLEANUP
    }

    assert actual == _GOOGLE_PERMANENT_WRITE_REASONS[app_id], app_id


def test_the_docs_in_place_edits_wait_on_an_edit_step() -> None:
    """A write check creates, verifies and deletes its own object and has no
    step for an edit, so the three Docs edits stay pending (#3567)."""
    exemptions = {
        exemption.tool: exemption.reason
        for exemption in CONNECTOR_CHECKS["google-docs"].exemptions
    }

    for tool in (
        "google_docs_append_text",
        "google_docs_batch_update",
        "google_docs_replace_text",
    ):
        assert exemptions[tool] == (
            "write that edits content in place: a write check creates, "
            "verifies and deletes its own object and has no step for an "
            "edit, so whether a check is added is decided in #3567"
        ), tool


def test_gmail_manage_labels_is_exempt_for_the_read_step_rule() -> None:
    exemption = next(
        exemption
        for exemption in CONNECTOR_CHECKS["gmail"].exemptions
        if exemption.tool == "gmail_manage_labels"
    )

    assert exemption.reason == _GMAIL_LABEL_WITHOUT_READ


def test_the_slide_thumbnail_waits_on_a_slide_id_fixture() -> None:
    """The deck the seed creates holds Google's default slide, so what keeps
    this read exempt is a fixture for that slide's id, not a slide."""
    exemption = next(
        exemption
        for exemption in CONNECTOR_CHECKS["google-slides"].exemptions
        if exemption.tool == "google_slides_get_slide_thumbnail"
    )

    assert exemption.reason == _SLIDE_THUMBNAIL_WITHOUT_SLIDE_ID


# Output samples for the Google read checks


class _GmailBatch:
    """The batch object ``gmail_search_messages`` drives: it calls back once
    per request with the response its test gave it."""

    def __init__(
        self, callback: Callable[..., None], responses: list[dict[str, Any]]
    ) -> None:
        self._callback = callback
        self._responses = responses

    def add(self, request: Any) -> None:
        return None

    def execute(self) -> None:
        for index, response in enumerate(self._responses):
            self._callback(str(index), response, None)


def _gmail_service(responses: list[dict[str, Any]]) -> Mock:
    service = Mock()
    service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": f"m{index}"} for index in range(len(responses))]
    }
    service.new_batch_http_request.side_effect = lambda callback: _GmailBatch(
        callback, responses
    )
    return service


def _gmail_message(subject: str = "xagent smoke seed message") -> dict[str, Any]:
    return {
        "id": "msg-1",
        "threadId": "thread-1",
        "snippet": "Seeded message for the Gmail read checks.",
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": "seed@xagent-smoke.invalid"},
            ]
        },
    }


def test_gmail_search_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gmail, "get_gmail_service", lambda: _gmail_service([_gmail_message()])
    )

    _assert_check_output(
        "gmail.search_seed_message",
        gmail.gmail_search_messages(label_ids=["Label_1"], max_results=10),
    )


@pytest.mark.parametrize(
    ("messages", "reason"),
    [
        ([_gmail_message(), _gmail_message()], "length"),
        ([_gmail_message("another subject")], "equals"),
    ],
)
def test_gmail_search_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, Any]], reason: str
) -> None:
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _gmail_service(messages))

    _assert_check_fails(
        "gmail.search_seed_message",
        gmail.gmail_search_messages(label_ids=["Label_1"], max_results=10),
        reason,
    )


_DRIVE_FOLDER = {
    "id": "folder-1",
    "name": "xagent-smoke-seed",
    "mimeType": "application/vnd.google-apps.folder",
    "modifiedTime": "2026-11-01T00:00:00.000Z",
}


def _drive_list_service(files: list[dict[str, Any]]) -> Mock:
    service = Mock()
    service.files.return_value.list.return_value.execute.return_value = {"files": files}
    return service


def _drive_search() -> str:
    return google_drive.google_drive_search(
        query=(
            "name = 'xagent-smoke-seed' and "
            "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        ),
        max_results=10,
    )


def test_google_drive_search_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_drive, "get_drive_service", lambda: _drive_list_service([_DRIVE_FOLDER])
    )

    _assert_check_output("google-drive.search_seed_folder", _drive_search())


@pytest.mark.parametrize(
    ("files", "reason"),
    [
        ([], "length"),
        ([{**_DRIVE_FOLDER, "id": "other"}], "equals"),
        (
            [
                {
                    key: value
                    for key, value in _DRIVE_FOLDER.items()
                    if key != "modifiedTime"
                }
            ],
            "each_has_keys",
        ),
    ],
)
def test_google_drive_search_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch, files: list[dict[str, Any]], reason: str
) -> None:
    monkeypatch.setattr(
        google_drive, "get_drive_service", lambda: _drive_list_service(files)
    )

    _assert_check_fails("google-drive.search_seed_folder", _drive_search(), reason)


def _drive_file_service() -> Mock:
    service = Mock()
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "doc-1",
        "name": "xagent-smoke-seed-doc",
        "mimeType": "application/vnd.google-apps.document",
    }
    return service


def test_google_drive_file_content_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_drive,
        "get_drive_service",
        lambda: _drive_file_service(),
    )
    monkeypatch.setattr(
        google_drive,
        "_download_media",
        lambda request: b"xagent smoke seed document\nSecond line: seeded content\n",
    )

    _assert_check_output(
        "google-drive.read_seed_file",
        google_drive.google_drive_get_file_content("doc-1"),
    )


def test_google_drive_file_content_output_fails_without_the_seed_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_drive,
        "get_drive_service",
        lambda: _drive_file_service(),
    )
    monkeypatch.setattr(
        google_drive, "_download_media", lambda request: b"another document"
    )

    _assert_check_fails(
        "google-drive.read_seed_file",
        google_drive.google_drive_get_file_content("doc-1"),
        "contains",
    )


def test_google_drive_file_content_output_fails_on_base64_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text that is not valid UTF-8 comes back base64-encoded, not as text."""
    monkeypatch.setattr(
        google_drive,
        "get_drive_service",
        lambda: _drive_file_service(),
    )
    monkeypatch.setattr(
        google_drive, "_download_media", lambda request: b"caf\xe9 legacy text"
    )

    _assert_check_fails(
        "google-drive.read_seed_file",
        google_drive.google_drive_get_file_content("doc-1"),
        "equals",
    )


_DOCUMENT_TEXT = "xagent smoke seed document"


def _document_body(text: str = _DOCUMENT_TEXT) -> dict[str, Any]:
    return {
        "content": [
            {"paragraph": {"elements": [{"textRun": {"content": text + "\n"}}]}},
            {
                "paragraph": {
                    "elements": [
                        {"textRun": {"content": "Second line: seeded content\n"}}
                    ]
                }
            },
        ]
    }


def _docs_service(title: str, text: str = _DOCUMENT_TEXT) -> Mock:
    service = Mock()
    service.documents.return_value.get.return_value.execute.return_value = {
        "documentId": "doc-1",
        "title": title,
        "body": _document_body(text),
    }
    return service


def test_google_docs_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_docs, "get_docs_service", lambda: _docs_service("xagent-smoke-seed-doc")
    )

    _assert_check_output(
        "google-docs.read_fixture", google_docs.google_docs_get_document("doc-1")
    )


@pytest.mark.parametrize(
    ("title", "text", "reason"),
    [
        ("another doc", _DOCUMENT_TEXT, "equals"),
        ("xagent-smoke-seed-doc", "another document", "contains"),
    ],
)
def test_google_docs_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch, title: str, text: str, reason: str
) -> None:
    monkeypatch.setattr(
        google_docs, "get_docs_service", lambda: _docs_service(title, text)
    )

    _assert_check_fails(
        "google-docs.read_fixture",
        google_docs.google_docs_get_document("doc-1"),
        reason,
    )


def _seed_sheet() -> dict[str, Any]:
    return {
        "spreadsheetId": "sheet-1",
        "properties": {"title": "xagent-smoke-seed-sheet"},
        "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/sheet-1",
        "sheets": [
            {
                "properties": {
                    "sheetId": 0,
                    "title": "Sheet1",
                    "gridProperties": {"rowCount": 1000, "columnCount": 26},
                }
            }
        ],
    }


def _spreadsheet_service(payload: dict[str, Any]) -> Mock:
    service = Mock()
    service.spreadsheets.return_value.get.return_value.execute.return_value = payload
    return service


def test_google_sheets_spreadsheet_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_sheets,
        "get_sheets_service",
        lambda: _spreadsheet_service(_seed_sheet()),
    )

    _assert_check_output(
        "google-sheets.read_fixture",
        google_sheets.google_sheets_get_spreadsheet("sheet-1"),
    )


@pytest.mark.parametrize(
    ("title", "sheets", "reason"),
    [
        ("another book", _seed_sheet()["sheets"], "equals"),
        ("xagent-smoke-seed-sheet", [], "length"),
        ("xagent-smoke-seed-sheet", _seed_sheet()["sheets"] * 2, "length"),
    ],
)
def test_google_sheets_spreadsheet_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch,
    title: str,
    sheets: list[dict[str, Any]],
    reason: str,
) -> None:
    payload = {**_seed_sheet(), "properties": {"title": title}, "sheets": sheets}
    monkeypatch.setattr(
        google_sheets, "get_sheets_service", lambda: _spreadsheet_service(payload)
    )

    _assert_check_fails(
        "google-sheets.read_fixture",
        google_sheets.google_sheets_get_spreadsheet("sheet-1"),
        reason,
    )


def _values_service(values: list[list[str]]) -> Mock:
    service = Mock()
    service.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = {
        "range": "Sheet1!A1:B3",
        "values": values,
    }
    return service


def _read_seed_range() -> str:
    return google_sheets.google_sheets_read_range(
        spreadsheet_id="sheet-1", range_name="A1:B3"
    )


def test_google_sheets_range_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = [["item", "quantity"], ["alpha", "1"], ["beta", "2"]]
    monkeypatch.setattr(
        google_sheets, "get_sheets_service", lambda: _values_service(values)
    )

    _assert_check_output("google-sheets.read_seed_range", _read_seed_range())


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ([["item", "quantity"], ["alpha", "1"]], "length"),
        ([["item", "quantity"], ["alpha", "2"], ["beta", "2"]], "equals"),
    ],
)
def test_google_sheets_range_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch, values: list[list[str]], reason: str
) -> None:
    monkeypatch.setattr(
        google_sheets, "get_sheets_service", lambda: _values_service(values)
    )

    _assert_check_fails("google-sheets.read_seed_range", _read_seed_range(), reason)


def _presentation_service(title: str, slide_count: int = 1) -> Mock:
    service = Mock()
    service.presentations.return_value.get.return_value.execute.return_value = {
        "presentationId": "deck-1",
        "title": title,
        "slides": [
            {"objectId": f"slide-{index}", "pageElements": []}
            for index in range(slide_count)
        ],
    }
    return service


def test_google_slides_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_slides,
        "get_slides_service",
        lambda: _presentation_service("xagent-smoke-seed-slides"),
    )

    _assert_check_output(
        "google-slides.read_fixture",
        google_slides.google_slides_get_presentation("deck-1"),
    )


def test_google_slides_output_fails_on_another_deck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_slides,
        "get_slides_service",
        lambda: _presentation_service("another deck"),
    )

    _assert_check_fails(
        "google-slides.read_fixture",
        google_slides.google_slides_get_presentation("deck-1"),
        "equals",
    )


@pytest.mark.parametrize("slide_count", [0, 2])
def test_google_slides_output_fails_on_another_slide_count(
    monkeypatch: pytest.MonkeyPatch, slide_count: int
) -> None:
    monkeypatch.setattr(
        google_slides,
        "get_slides_service",
        lambda: _presentation_service("xagent-smoke-seed-slides", slide_count),
    )

    _assert_check_fails(
        "google-slides.read_fixture",
        google_slides.google_slides_get_presentation("deck-1"),
        "equals",
    )


def _analytics_report() -> dict[str, Any]:
    return {
        "dimensionHeaders": [],
        "metricHeaders": [{"name": "sessions", "type": "TYPE_INTEGER"}],
        "rows": [],
        "totals": [{"metricValues": [{"value": "0"}]}],
        "rowCount": 0,
    }


def _run_report(property_id: str) -> str:
    return google_analytics.google_analytics_run_report(
        property_id=property_id,
        metrics=["sessions"],
        date_ranges=[{"start_date": "7daysAgo", "end_date": "yesterday"}],
    )


def test_google_analytics_report_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_analytics, "_request", lambda *args, **kwargs: _analytics_report()
    )

    _assert_check_output("google-analytics.run_report", _run_report("563334905"))


def test_google_analytics_list_properties_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_analytics,
        "_request",
        lambda *args, **kwargs: {
            "accountSummaries": [
                {
                    "displayName": "Seed account",
                    "propertySummaries": [
                        {
                            "property": "properties/563334905",
                            "displayName": "xagent-smoke-seed",
                        }
                    ],
                }
            ]
        },
    )

    _assert_check_output(
        "google-analytics.l3_probe", google_analytics.google_analytics_list_properties()
    )


def test_google_analytics_list_properties_output_fails_on_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("unauthorized")

    monkeypatch.setattr(google_analytics, "_request", fail)

    _assert_check_fails(
        "google-analytics.l3_probe",
        google_analytics.google_analytics_list_properties(),
        "status_not_success",
    )


def _calendar_service(items: list[dict[str, Any]]) -> Mock:
    service = Mock()
    service.events.return_value.list.return_value.execute.return_value = {
        "timeZone": "UTC",
        "items": items,
    }
    return service


def test_google_calendar_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_calendar, "get_calendar_service", lambda: _calendar_service([])
    )

    _assert_check_output(
        "google-calendar.l3_probe",
        google_calendar.google_calendar_search_events(max_results=1),
    )


def test_google_calendar_output_fails_on_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    service.events.return_value.list.return_value.execute.side_effect = RuntimeError(
        "unauthorized"
    )
    monkeypatch.setattr(google_calendar, "get_calendar_service", lambda: service)

    _assert_check_fails(
        "google-calendar.l3_probe",
        google_calendar.google_calendar_search_events(max_results=1),
        "status_not_success",
    )


def _unreachable(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("unauthorized")


@pytest.mark.parametrize(
    ("check_id", "module", "dependency", "call"),
    [
        (
            "gmail.search_seed_message",
            gmail,
            "get_gmail_service",
            lambda: gmail.gmail_search_messages(label_ids=["Label_1"], max_results=10),
        ),
        (
            "google-analytics.run_report",
            google_analytics,
            "_request",
            lambda: _run_report("563334905"),
        ),
        (
            "google-docs.read_fixture",
            google_docs,
            "get_docs_service",
            lambda: google_docs.google_docs_get_document("doc-1"),
        ),
        (
            "google-drive.search_seed_folder",
            google_drive,
            "get_drive_service",
            _drive_search,
        ),
        (
            "google-drive.read_seed_file",
            google_drive,
            "get_drive_service",
            lambda: google_drive.google_drive_get_file_content("doc-1"),
        ),
        (
            "google-sheets.read_fixture",
            google_sheets,
            "get_sheets_service",
            lambda: google_sheets.google_sheets_get_spreadsheet("sheet-1"),
        ),
        (
            "google-sheets.read_seed_range",
            google_sheets,
            "get_sheets_service",
            _read_seed_range,
        ),
        (
            "google-slides.read_fixture",
            google_slides,
            "get_slides_service",
            lambda: google_slides.google_slides_get_presentation("deck-1"),
        ),
    ],
)
def test_a_google_fixture_check_fails_on_an_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
    check_id: str,
    module: Any,
    dependency: str,
    call: Callable[[], str],
) -> None:
    monkeypatch.setattr(module, dependency, _unreachable)

    _assert_check_fails(check_id, call(), "status_success")


# --- GitHub (#3568) ------------------------------------------------------------------

_READ_REPO = "xagent-smoke/smoke-read"

# The read checks, pinned: the tool, its arguments and its assertions. Five of
# them are the five most called tools of the 89 GitHub calls in the 7-day
# production-au tool-call volume reported on xorbitsai/xagent-saas#2532, the
# five read checks a Tier 1 connector declares at most; `github.l3_probe` is
# the probe-derived one the connector keeps alongside them, so a check that
# reads no fixture still calls GitHub.
_GITHUB_READS: Mapping[str, tuple[str, dict[str, Any], tuple[Assertion, ...]]] = {
    "github.l3_probe": ("github_get_current_user", {}, (L3Envelope(),)),
    "github.current_user": (
        "github_get_current_user",
        {},
        (
            StatusSuccess(),
            HasKeys(("user",), ("login", "id", "html_url")),
            Equals(("user", "login"), FixtureRef("github.login")),
        ),
    ),
    "github.read_repository": (
        "github_get_repository",
        {"repo": FixtureRef("github.read_repo")},
        (
            StatusSuccess(),
            HasKeys(
                ("repository",),
                ("full_name", "private", "default_branch", "html_url"),
            ),
            Equals(("repository", "full_name"), FixtureRef("github.read_repo")),
            Equals(("repository", "private"), True),
        ),
    ),
    "github.read_issue": (
        "github_get_issue",
        {
            "repo": FixtureRef("github.read_repo"),
            "issue_number": FixtureRef("github.issue_number"),
        },
        (
            StatusSuccess(),
            HasKeys(
                ("issue",),
                ("number", "title", "state", "labels", "user", "html_url"),
            ),
            Equals(("issue", "number"), FixtureRef("github.issue_number")),
            Equals(("issue", "title"), "xagent smoke seed issue"),
            Equals(("issue", "state"), "open"),
            Equals(("issue", "is_pull_request"), False),
        ),
    ),
    "github.read_commits": (
        "github_list_commits",
        {
            "repo": FixtureRef("github.read_repo"),
            "path": FixtureRef("github.seed_file_path"),
            "limit": 1,
        },
        (
            StatusSuccess(),
            HasKeys((), ("commits", "truncated")),
            Length(("commits",), 1, 1),
            EachHasKeys(("commits",), ("sha", "message", "author", "date")),
            Equals(("commits", 0, "message"), "xagent smoke seed"),
        ),
    ),
    "github.read_pull_request": (
        "github_get_pull_request",
        {
            "repo": FixtureRef("github.read_repo"),
            "pull_number": FixtureRef("github.pull_number"),
        },
        (
            StatusSuccess(),
            HasKeys(
                ("pull_request",),
                ("number", "title", "state", "head", "base", "draft"),
            ),
            Equals(("pull_request", "number"), FixtureRef("github.pull_number")),
            Equals(("pull_request", "title"), "xagent smoke seed pull request"),
            Equals(("pull_request", "state"), "open"),
            Equals(("pull_request", "draft"), False),
            Equals(("pull_request", "head"), FixtureRef("github.pr_branch")),
        ),
    ),
}

# The fixture keys the GitHub read checks use, pinned here so that adding or
# dropping one shows in the diff. The xagent-saas tenant fixtures record the
# login, the three ids the seed's layout knows, and the two numbers GitHub
# assigns (xorbitsai/xagent-saas#2533).
_GITHUB_FIXTURE_KEYS_USED = frozenset(
    {
        "github.issue_number",
        "github.login",
        "github.pr_branch",
        "github.pull_number",
        "github.read_repo",
        "github.seed_file_path",
    }
)

# The reads left to an exemption: GitHub's 80% cut needs eight tools, and the
# five most called have the checks.
_GITHUB_READS_BEYOND_THE_FIVE_TOOLS = frozenset(
    {
        "github_get_file_contents",
        "github_get_latest_release",
        "github_get_ref_status",
        "github_list_issues",
        "github_list_pull_requests",
        "github_list_workflow_runs",
        "github_search_code",
        "github_search_repositories",
    }
)

# The writes that create an object GitHub's own tools cannot remove: the server
# has no delete tool at all, so a write check would have nothing to clean up
# with. A release edit is the one write that creates no object.
_GITHUB_WRITES_WITHOUT_CLEANUP = frozenset(
    {
        "github_comment_on_issue",
        "github_create_branch",
        "github_create_issue",
        "github_create_or_update_file",
        "github_create_pull_request",
        "github_create_release",
        "github_update_submodule",
    }
)


@pytest.mark.parametrize(("check_id", "expected"), sorted(_GITHUB_READS.items()))
def test_a_github_read_check_is_declared_exactly(
    check_id: str, expected: tuple[str, dict[str, Any], tuple[Assertion, ...]]
) -> None:
    tool, arguments, assertions = expected
    check = _read_check(check_id)
    assert (check.tool, dict(check.arguments), check.assertions) == (
        tool,
        arguments,
        assertions,
    )


def test_a_github_connector_declares_exactly_its_reads() -> None:
    assert [check.id for check in CONNECTOR_CHECKS["github"].reads] == list(
        _GITHUB_READS
    )


def test_the_github_read_checks_use_exactly_these_fixture_keys() -> None:
    keys = {
        ref.key
        for check in CONNECTOR_CHECKS["github"].reads
        if isinstance(check, ReadCheck)
        for ref in _fixture_keys(check)
    }

    assert keys == _GITHUB_FIXTURE_KEYS_USED


def test_a_github_fixture_is_bound_to_a_parameter_of_its_tool() -> None:
    """A fixture in the arguments names a parameter of the tool, and every
    required argument is pinned: an omitted one would make the tool prompt.
    That the value is the type the parameter declares is the adapter test's
    judgement, below."""
    for check in CONNECTOR_CHECKS["github"].reads:
        if not isinstance(check, ReadCheck):
            continue
        parameters = inspect.signature(getattr(github, check.tool)).parameters
        bound = [
            name for name, value in check.arguments.items() if any(_fixture_refs(value))
        ]
        for name in bound:
            assert name in parameters, (check.tool, name)
        required = {
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        }
        assert required <= set(check.arguments), (check.tool, required)


async def test_a_github_fixture_bound_argument_passes_the_tools_own_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The value a fixture resolves to survives the adapter's own checks.

    The tools validate their arguments locally before a call leaves, and a
    string where a tool declares an integer is refused there, so a fixture the
    registry holds as a string would make its check fail on every run. The
    session is replaced with one that refuses: reaching it means the adapter
    accepted the arguments.
    """
    reached: list[str] = []

    @asynccontextmanager
    async def _refuse_session(connection: Any) -> Any:
        reached.append(str(connection))
        raise RuntimeError("no server in this test")
        yield  # pragma: no cover

    monkeypatch.setattr(mcp_adapter, "create_session", _refuse_session)
    tools = {tool.name: tool for tool in await github.mcp.list_tools()}
    for check in CONNECTOR_CHECKS["github"].reads:
        if not isinstance(check, ReadCheck) or not any(_fixture_refs(check.arguments)):
            continue
        adapter = MCPToolAdapter(
            mcp_tool=tools[check.tool],
            connection={
                "transport": "streamable_http",
                "url": "https://mcp.example.test",
            },
        )
        reached.clear()

        raw = await adapter.run_json_async(
            resolve_fixtures(dict(check.arguments), _FIXTURES)
        )

        assert reached, (check.id, raw)
        assert "Invalid MCP tool arguments" not in raw["content"][0]["text"], check.id


def test_a_github_read_outside_the_five_is_exempt_for_the_cut() -> None:
    exemptions = {
        exemption.tool: exemption.reason
        for exemption in CONNECTOR_CHECKS["github"].exemptions
    }

    for tool in _GITHUB_READS_BEYOND_THE_FIVE_TOOLS:
        assert exemptions[tool] == _GITHUB_READS_BEYOND_THE_FIVE, tool
    assert set(exemptions) == _GITHUB_READS_BEYOND_THE_FIVE_TOOLS | (
        _GITHUB_WRITES_WITHOUT_CLEANUP | {"github_update_release"}
    )


def test_a_github_write_has_no_cleanup_tool() -> None:
    """Every write but a release edit creates an object, and the server has no
    delete tool for any of them, so no write check can clean up yet."""
    exemptions = {
        exemption.tool: exemption.reason
        for exemption in CONNECTOR_CHECKS["github"].exemptions
    }

    for tool in _GITHUB_WRITES_WITHOUT_CLEANUP:
        assert exemptions[tool] == WRITE_WITHOUT_CLEANUP, tool


def test_the_github_release_edit_waits_on_an_edit_step() -> None:
    exemption = next(
        exemption
        for exemption in CONNECTOR_CHECKS["github"].exemptions
        if exemption.tool == "github_update_release"
    )

    assert exemption.reason == (
        "write that edits content in place: a write check creates, verifies "
        "and deletes its own object and has no step for an edit, so whether a "
        "check is added is decided in #3568"
    )


# Output samples for the GitHub read checks


class _GitHubResponse:
    """The part of a ``requests`` response ``github.py`` reads."""

    def __init__(self, body: Any, status: int = 200) -> None:
        self.status_code = status
        self.content = json.dumps(body).encode()
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return json.loads(self.content)


def _fake_github(
    monkeypatch: pytest.MonkeyPatch,
    *routes: tuple[str, str, Any],
    status: int = 200,
) -> dict[tuple[str, str], Any]:
    """``github.py``'s HTTP layer: one body per route, answered with ``status``,
    and the query params every request carried. A callable body is answered
    from the params, for a route whose answer depends on them."""
    # The token the server reads its grant from; the fake transport never sends
    # it anywhere.
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "access-token")
    bodies = {(method, path): body for method, path, body in routes}
    params_of: dict[tuple[str, str], Any] = {}

    def request(**kwargs: Any) -> _GitHubResponse:
        route = (
            kwargs["method"],
            str(kwargs["url"]).removeprefix(github.GITHUB_BASE_URL),
        )
        params_of[route] = kwargs.get("params")
        try:
            body = bodies[route]
        except KeyError:
            raise AssertionError(f"unrouted request: {route[0]} {route[1]}") from None
        if callable(body):
            body = body(params_of[route] or {})
        return _GitHubResponse(body, status)

    monkeypatch.setattr(github, "requests", SimpleNamespace(request=request))
    return params_of


def _github_repository(**overrides: Any) -> dict[str, Any]:
    return {
        "full_name": _READ_REPO,
        "private": True,
        "default_branch": "main",
        "html_url": "https://github.com/xagent-smoke/smoke-read",
        "description": "Seed repository of the GitHub smoke tenant",
        "stargazers_count": 0,
        "open_issues_count": 2,
        "language": None,
        "updated_at": "2026-11-12T00:00:00Z",
        **overrides,
    }


def _github_issue(**overrides: Any) -> dict[str, Any]:
    return {
        "number": 1,
        "title": "xagent smoke seed issue",
        "state": "open",
        "body": "Seeded issue the read checks assert on.",
        "labels": [],
        "user": {"login": "xagent-app"},
        "comments": 0,
        "html_url": "https://github.com/xagent-smoke/smoke-read/issues/1",
        "created_at": "2026-11-12T00:00:00Z",
        "updated_at": "2026-11-12T00:00:00Z",
        **overrides,
    }


def _github_commit(**overrides: Any) -> dict[str, Any]:
    return {
        "sha": "0" * 40,
        "commit": {
            "message": "xagent smoke seed",
            "author": {"name": "xagent-app", "date": "2026-11-12T00:00:00Z"},
        },
        "html_url": "https://github.com/xagent-smoke/smoke-read/commit/" + "0" * 40,
        **overrides,
    }


def _github_pull_request(**overrides: Any) -> dict[str, Any]:
    return {
        "number": 2,
        "title": "xagent smoke seed pull request",
        "state": "open",
        "body": "Seeded pull request the read checks assert on.",
        "user": {"login": "xagent-app"},
        "head": {"ref": "xagent-smoke-seed"},
        "base": {"ref": "main"},
        "draft": False,
        "merged": False,
        "mergeable": True,
        "html_url": "https://github.com/xagent-smoke/smoke-read/pull/2",
        "created_at": "2026-11-12T00:00:00Z",
        "updated_at": "2026-11-12T00:00:00Z",
        **overrides,
    }


def _github_user(**overrides: Any) -> dict[str, Any]:
    return {
        "login": "xagent-app",
        "id": 424242,
        "name": "Xagent Smoke",
        "email": None,
        "company": None,
        "bio": None,
        "public_repos": 0,
        "followers": 0,
        "following": 0,
        "html_url": "https://github.com/xagent-app",
        **overrides,
    }


def _github_declared_output(check_id: str) -> str:
    """The check's own tool called with the arguments its declaration resolves
    to: the sample is the call the check makes, not one built from literals."""
    check = _read_check(check_id)
    return getattr(github, check.tool)(
        **resolve_fixtures(dict(check.arguments), _FIXTURES)
    )


def test_github_current_user_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_github(monkeypatch, ("GET", "/user", _github_user()))

    _assert_check_output(
        "github.current_user", _github_declared_output("github.current_user")
    )


def test_github_current_user_output_fails_on_another_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tenant registry pins the account the QA member's connection uses. An
    # account that can see the private repository passes the other four checks,
    # so this is the one that notices the wrong account.
    _fake_github(monkeypatch, ("GET", "/user", _github_user(login="qhhh-personal")))

    _assert_check_fails(
        "github.current_user", github.github_get_current_user(), "equals"
    )


def test_github_repository_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_github(
        monkeypatch,
        ("GET", f"/repos/{_READ_REPO}", _github_repository()),
    )

    _assert_check_output(
        "github.read_repository", _github_declared_output("github.read_repository")
    )


def test_github_repository_output_fails_on_another_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_github(
        monkeypatch,
        (
            "GET",
            f"/repos/{_READ_REPO}",
            _github_repository(full_name="xagent-smoke/smoke-write"),
        ),
    )

    _assert_check_fails(
        "github.read_repository",
        github.github_get_repository(_READ_REPO),
        "equals",
    )


def test_github_repository_output_fails_on_a_public_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A public repository is readable without the organization's grant, so the
    # check would no longer see a revoked one.
    _fake_github(
        monkeypatch,
        ("GET", f"/repos/{_READ_REPO}", _github_repository(private=False)),
    )

    _assert_check_fails(
        "github.read_repository",
        github.github_get_repository(_READ_REPO),
        "equals",
    )


def test_github_issue_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_github(
        monkeypatch,
        ("GET", f"/repos/{_READ_REPO}/issues/1", _github_issue()),
    )

    _assert_check_output(
        "github.read_issue", _github_declared_output("github.read_issue")
    )


def test_github_issue_output_fails_on_a_pull_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GitHub answers both from the issues endpoints, and a number that reaches
    # a pull request proves the fixture names the wrong one.
    _fake_github(
        monkeypatch,
        (
            "GET",
            f"/repos/{_READ_REPO}/issues/1",
            _github_issue(pull_request={"url": "https://api.github.com/x"}),
        ),
    )

    _assert_check_fails(
        "github.read_issue", github.github_get_issue(_READ_REPO, 1), "equals"
    )


def test_github_issue_output_fails_on_another_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fixture names the issue the seed creates, so a payload carrying
    # another number is not the issue the check reads.
    _fake_github(
        monkeypatch,
        ("GET", f"/repos/{_READ_REPO}/issues/1", _github_issue(number=7)),
    )

    _assert_check_fails(
        "github.read_issue", github.github_get_issue(_READ_REPO, 1), "equals"
    )


def test_github_commits_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params_of = _fake_github(
        monkeypatch,
        ("GET", f"/repos/{_READ_REPO}/commits", [_github_commit()]),
    )

    _assert_check_output(
        "github.read_commits", _github_declared_output("github.read_commits")
    )
    # The declaration's `path` and `limit` are what GitHub is asked for: a
    # dropped path would judge the branch head instead of the seed file's
    # commit.
    assert params_of[("GET", f"/repos/{_READ_REPO}/commits")] == {
        "per_page": 1,
        "page": 1,
        "path": "seed.md",
    }


def test_github_commits_output_judges_the_seed_files_commit_not_the_branch_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default branch's newest commit is the seed's second file, under the
    message the seed writes there, so it is not the message the check asserts;
    the check reads the seed file's history, so it still passes."""
    head = _github_commit(
        commit={
            "message": "xagent smoke seed after",
            "author": {"name": "xagent-app", "date": "2026-11-12T00:00:01Z"},
        }
    )
    seeded = _github_commit()
    _fake_github(
        monkeypatch,
        (
            "GET",
            f"/repos/{_READ_REPO}/commits",
            lambda params: [seeded] if params.get("path") == "seed.md" else [head],
        ),
    )

    _assert_check_output(
        "github.read_commits", _github_declared_output("github.read_commits")
    )


def test_github_commits_output_fails_on_another_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    another = _github_commit(
        commit={"message": "edit by hand", "author": {"name": "someone"}}
    )
    _fake_github(monkeypatch, ("GET", f"/repos/{_READ_REPO}/commits", [another]))

    _assert_check_fails(
        "github.read_commits",
        github.github_list_commits(_READ_REPO, path="seed.md", limit=1),
        "equals",
    )


def test_github_pull_request_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_github(
        monkeypatch,
        ("GET", f"/repos/{_READ_REPO}/pulls/2", _github_pull_request()),
    )

    _assert_check_output(
        "github.read_pull_request", _github_declared_output("github.read_pull_request")
    )


def test_github_pull_request_output_fails_on_another_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The head is the seed's branch, so a pull request opened from anywhere
    # else is not the seeded one.
    _fake_github(
        monkeypatch,
        (
            "GET",
            f"/repos/{_READ_REPO}/pulls/2",
            _github_pull_request(head={"ref": "someone-elses-branch"}),
        ),
    )

    _assert_check_fails(
        "github.read_pull_request",
        github.github_get_pull_request(_READ_REPO, 2),
        "equals",
    )


def test_github_pull_request_output_fails_on_another_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fixture names the pull request the seed opens, so a payload carrying
    # another number is not the one the check reads.
    _fake_github(
        monkeypatch,
        ("GET", f"/repos/{_READ_REPO}/pulls/2", _github_pull_request(number=9)),
    )

    _assert_check_fails(
        "github.read_pull_request",
        github.github_get_pull_request(_READ_REPO, 2),
        "equals",
    )


@pytest.mark.parametrize(
    ("check_id", "path"),
    [
        ("github.current_user", "/user"),
        ("github.read_repository", f"/repos/{_READ_REPO}"),
        ("github.read_issue", f"/repos/{_READ_REPO}/issues/1"),
        ("github.read_commits", f"/repos/{_READ_REPO}/commits"),
        ("github.read_pull_request", f"/repos/{_READ_REPO}/pulls/2"),
    ],
)
def test_a_github_check_fails_on_a_revoked_grant(
    monkeypatch: pytest.MonkeyPatch, check_id: str, path: str
) -> None:
    # A revoked token is a 401 with this body: the tool catches the error and
    # returns its error envelope, which the check sees as not a success.
    _fake_github(monkeypatch, ("GET", path, {"message": "Bad credentials"}), status=401)

    _assert_check_fails(check_id, _github_declared_output(check_id), "status_success")

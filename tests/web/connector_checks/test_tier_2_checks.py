"""Smoke V2 Tier 2 checks of linear, intercom, deputy, employment-hero and xero (#3573).

Each connector keeps its L3 call and adds structural read checks on the list
tools that take no object id; every other tool is exempted, because Tier 2
seeds no objects and runs no write checks. A check runs the real tool with the
vendor call replaced by a realistic answer and judges what the tool returns, so
a tool whose output shape drifts from its declaration fails here, not on the
QA member.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from xagent.web.connector_checks.assertions import judge_result
from xagent.web.connector_checks.declarations import (
    Assertion,
    EachHasKeys,
    HasKeys,
    IsType,
    Length,
    ReadCheck,
    StatusSuccess,
)
from xagent.web.connector_checks.registry import (
    _TIER_2,
    CONNECTOR_CHECKS,
    WRITE_WITHOUT_CLEANUP,
    l3_read_check,
)
from xagent.web.connector_checks.results import FailureClass
from xagent.web.tools.mcp import deputy, employment_hero, intercom, linear, xero

APPS = ("linear", "intercom", "deputy", "employment-hero", "xero")

_XERO_UNPAGED = {
    "xero.list_accounts": ("accounts", "list_accounts", "Accounts"),
    "xero.list_contact_groups": (
        "contact_groups",
        "list_contact_groups",
        "ContactGroups",
    ),
    "xero.list_items": ("items", "list_items", "Items"),
    "xero.list_tax_rates": ("tax_rates", "list_tax_rates", "TaxRates"),
    "xero.list_tracking_categories": (
        "tracking_categories",
        "list_tracking_categories",
        "TrackingCategories",
    ),
}
_XERO_PAGED = {
    "xero.list_contacts": ("contacts", "list_contacts", "Contacts"),
    "xero.list_invoices": ("invoices", "list_invoices", "Invoices"),
    "xero.list_credit_notes": ("credit_notes", "list_credit_notes", "CreditNotes"),
    "xero.list_bank_transactions": (
        "bank_transactions",
        "list_bank_transactions",
        "BankTransactions",
    ),
    "xero.list_payments": ("payments", "list_payments", "Payments"),
    "xero.list_quotes": ("quotes", "list_quotes", "Quotes"),
    "xero.list_manual_journals": (
        "manual_journals",
        "list_manual_journals",
        "ManualJournals",
    ),
}


def _check(check_id: str) -> ReadCheck:
    app_id = check_id.split(".", 1)[0]
    check = next(c for c in CONNECTOR_CHECKS[app_id].reads if c.id == check_id)
    assert isinstance(check, ReadCheck)
    return check


def _result(payload: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": payload}],
        "structured_content": None,
        "is_error": False,
    }


def _judge(check_id: str, payload: str) -> tuple[FailureClass | None, str | None]:
    check = _check(check_id)
    return judge_result(_result(payload), check.payload, check.assertions)


def _structural_ids(app_id: str) -> list[str]:
    return [
        c.id
        for c in CONNECTOR_CHECKS[app_id].reads
        if isinstance(c, ReadCheck) and c.id != f"{app_id}.l3_probe"
    ]


# --- what is declared ------------------------------------------------------------


@pytest.mark.parametrize("app_id", APPS)
def test_each_connector_keeps_its_probe_derived_l3_check(app_id: str) -> None:
    expected = l3_read_check(app_id)
    assert expected is not None

    l3_checks = [
        c for c in CONNECTOR_CHECKS[app_id].reads if c.id == f"{app_id}.l3_probe"
    ]
    assert l3_checks == [expected]


@pytest.mark.parametrize("app_id", APPS)
def test_tier_2_declares_reads_only(app_id: str) -> None:
    checks = CONNECTOR_CHECKS[app_id]

    assert checks.writes == () and checks.sweep is None
    assert len(_structural_ids(app_id)) >= 1


@pytest.mark.parametrize("app_id", APPS)
def test_every_exemption_names_tier_2_or_says_why_it_has_no_cleanup(
    app_id: str,
) -> None:
    for exemption in CONNECTOR_CHECKS[app_id].exemptions:
        assert exemption.reason == WRITE_WITHOUT_CLEANUP or exemption.reason.endswith(
            f"decided in {_TIER_2}"
        ), (app_id, exemption.tool)


# What each check calls: the tool and its arguments. A resource of Deputy is an
# object type, not an object id.
_CALLS: dict[str, tuple[str, dict[str, Any]]] = {
    "linear.list_teams": ("linear_list_teams", {"limit": 1}),
    "linear.list_projects": ("linear_list_projects", {"limit": 1}),
    "linear.search_users": ("linear_search_users", {"limit": 1}),
    "linear.search_issues": ("linear_search_issues", {"limit": 1}),
    "intercom.list_conversations": ("intercom_list_conversations", {"limit": 1}),
    "deputy.list_companies": ("deputy_list_resource", {"resource": "Company"}),
    "deputy.employee_info": ("deputy_resource_info", {"resource": "Employee"}),
    "employment-hero.list_organisations": (
        "employment_hero_list_organisations",
        {"item_per_page": 1},
    ),
    **{cid: (tool, {}) for cid, (_, tool, _) in _XERO_UNPAGED.items()},
    **{cid: (tool, {"page_size": 1}) for cid, (_, tool, _) in _XERO_PAGED.items()},
}

_LINEAR_ITEM_KEYS: dict[str, tuple[str, ...]] = {
    "teams": ("id", "key", "name"),
    "projects": ("id", "name", "status", "progress", "url", "teams"),
    "users": ("id", "name", "email", "active"),
    "issues": (
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
}
_INTERCOM_KEYS = (
    "id",
    "state",
    "title",
    "subject",
    "preview",
    "created_at",
    "updated_at",
)


def _expected_assertions(check_id: str) -> tuple[Assertion, ...]:
    """What each check must assert, built here and not read from the registry."""
    if check_id.startswith("linear."):
        field = {"list_teams": "teams", "list_projects": "projects"}.get(
            check_id.split(".")[1], check_id.split("_")[-1]
        )
        return (
            StatusSuccess(),
            HasKeys((), (field, "truncated")),
            EachHasKeys((field,), _LINEAR_ITEM_KEYS[field]),
            Length((field,), max=1),
            IsType(("truncated",), "boolean"),
        )
    if check_id == "intercom.list_conversations":
        return (
            StatusSuccess(),
            HasKeys((), ("conversations", "total_count", "has_more", "next_cursor")),
            EachHasKeys(("conversations",), _INTERCOM_KEYS),
            Length(("conversations",), max=1),
            IsType(("has_more",), "boolean"),
        )
    if check_id == "deputy.list_companies":
        return (
            StatusSuccess(),
            HasKeys((), ("records", "truncated")),
            IsType(("records",), "array"),
            IsType(("truncated",), "boolean"),
        )
    if check_id == "deputy.employee_info":
        return (StatusSuccess(), HasKeys((), ("info",)), IsType(("info",), "object"))
    if check_id == "employment-hero.list_organisations":
        return (
            StatusSuccess(),
            HasKeys((), ("organisations",)),
            IsType(("organisations",), "object"),
            HasKeys(("organisations",), ("items",)),
            IsType(("organisations", "items"), "array"),
            Length(("organisations", "items"), max=1),
        )
    if check_id in _XERO_UNPAGED:
        field = _XERO_UNPAGED[check_id][0]
        return (
            StatusSuccess(),
            HasKeys((), (field, "total_count", "returned")),
            IsType(("total_count",), "number"),
            IsType((field,), "array"),
            IsType(("returned",), "number"),
        )
    field = _XERO_PAGED[check_id][0]
    return (
        StatusSuccess(),
        HasKeys((), (field, "page", "page_size", "returned", "has_more")),
        IsType(("has_more",), "boolean"),
        Length((field,), max=1),
        IsType((field,), "array"),
        IsType(("returned",), "number"),
    )


@pytest.mark.parametrize("check_id", sorted(_CALLS))
def test_a_tier_2_read_check_is_declared_exactly(check_id: str) -> None:
    tool, arguments = _CALLS[check_id]
    check = _check(check_id)

    assert (check.tool, dict(check.arguments)) == (tool, arguments)
    assert check.assertions == _expected_assertions(check_id)
    assert not check.require_read_only_hint  # an in-repo server, not vendor-owned


def test_every_structural_check_is_one_of_those_declared_here() -> None:
    declared = {cid for app in APPS for cid in _structural_ids(app)}

    assert declared == set(_CALLS)


# --- what the tools really return -------------------------------------------------


def _linear(field: str, item: dict[str, Any], count: int):
    return lambda *a, **k: (
        {field: {"nodes": [item] * count, "pageInfo": {"hasNextPage": False}}},
        None,
    )


_LINEAR_ITEMS = {
    "teams": (
        "linear_list_teams",
        "teams",
        {"id": "t1", "key": "ENG", "name": "Engineering"},
    ),
    "projects": (
        "linear_list_projects",
        "projects",
        {
            "id": "p1",
            "name": "Roadmap",
            "status": {"id": "s1", "name": "Planned", "type": "planned"},
            "progress": 0.5,
            "url": "https://linear.app/x/project/roadmap",
            "teams": {"nodes": [{"id": "t1", "key": "ENG", "name": "Engineering"}]},
        },
    ),
    "users": (
        "linear_search_users",
        "users",
        {"id": "u1", "name": "Ada", "email": "ada@example.com", "active": True},
    ),
    "issues": (
        "linear_search_issues",
        "issues",
        {
            "id": "i1",
            "identifier": "ENG-1",
            "title": "A bug",
            "priority": 2,
            "url": "https://linear.app/x/issue/ENG-1",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-02T00:00:00Z",
            "state": {"id": "s2", "name": "Todo", "type": "unstarted"},
            # An unassigned, project-less issue still carries the keys, as null.
            "assignee": None,
            "team": {"id": "t1", "key": "ENG", "name": "Engineering"},
            "project": None,
            "labels": {"nodes": []},
        },
    ),
}


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("field", sorted(_LINEAR_ITEMS))
def test_a_linear_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, field: str, count: int
) -> None:
    tool, graphql_field, item = _LINEAR_ITEMS[field]
    monkeypatch.setattr(linear, "_graphql", _linear(graphql_field, item, count))
    check_id = next(c for c in _CALLS if _CALLS[c][0] == tool)

    assert _judge(check_id, getattr(linear, tool)(**_CALLS[check_id][1])) == (
        None,
        None,
    )


@pytest.mark.parametrize("count", [0, 1])
def test_intercom_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    conversation = {
        "id": "1",
        "state": "open",
        "title": None,
        "source": {"subject": "Hello", "body": "<p>Hi</p>"},
        "created_at": 1700000000,
        "updated_at": 1700000100,
    }
    monkeypatch.setattr(
        intercom,
        "_request",
        lambda *a, **k: {
            "conversations": [conversation] * count,
            "total_count": count,
            "pages": {},
        },
    )

    output = intercom.intercom_list_conversations(limit=1)

    assert _judge("intercom.list_conversations", output) == (None, None)


@pytest.mark.parametrize("count", [0, 1])
def test_deputy_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    monkeypatch.setattr(
        deputy, "_request", lambda *a, **k: [{"Id": 1, "CompanyName": "Demo"}] * count
    )

    output = deputy.deputy_list_resource(resource="Company")

    assert _judge("deputy.list_companies", output) == (None, None)


def test_deputy_a_null_list_is_an_empty_one_and_still_satisfies_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deputy serializes "no records" as null on some installs.
    monkeypatch.setattr(deputy, "_request", lambda *a, **k: None)

    output = deputy.deputy_list_resource(resource="Company")

    assert _judge("deputy.list_companies", output) == (None, None)


def test_deputy_resource_info_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deputy,
        "_request",
        lambda *a, **k: {"fields": {"Id": "Integer"}, "assocs": {}, "joins": {}},
    )

    output = deputy.deputy_resource_info(resource="Employee")

    assert _judge("deputy.employee_info", output) == (None, None)


@pytest.mark.parametrize("count", [0, 1])
def test_employment_hero_organisations_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    monkeypatch.setattr(
        employment_hero,
        "_request",
        lambda *a, **k: {
            "data": {
                "items": [{"id": "o1", "name": "Demo", "country": "AU"}] * count,
                "page_index": 1,
                "item_per_page": 1,
                "total_items": count,
                "total_pages": 1,
            }
        },
    )

    output = employment_hero.employment_hero_list_organisations(item_per_page=1)

    assert _judge("employment-hero.list_organisations", output) == (None, None)


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("check_id", sorted(_XERO_UNPAGED))
def test_a_xero_unpaged_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, check_id: str, count: int
) -> None:
    _, tool, vendor_key = _XERO_UNPAGED[check_id]
    monkeypatch.setattr(xero, "_request", lambda *a, **k: {vendor_key: [{}] * count})

    assert _judge(check_id, getattr(xero, tool)()) == (None, None)


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("check_id", sorted(_XERO_PAGED))
def test_a_xero_paged_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, check_id: str, count: int
) -> None:
    _, tool, vendor_key = _XERO_PAGED[check_id]
    monkeypatch.setattr(
        xero,
        "_request",
        lambda *a, **k: {
            vendor_key: [{}] * count,
            "pagination": {
                "page": 1,
                "pageSize": 1,
                "pageCount": 1,
                "itemCount": count,
            },
        },
    )

    assert _judge(check_id, getattr(xero, tool)(page_size=1)) == (None, None)


def test_a_xero_list_without_a_pagination_block_still_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Some endpoints carry none: the tool falls back to "a full page may have more".
    monkeypatch.setattr(xero, "_request", lambda *a, **k: {"Quotes": [{}]})

    assert _judge("xero.list_quotes", xero.list_quotes(page_size=1)) == (None, None)


# --- what a check rejects ----------------------------------------------------------


@pytest.mark.parametrize("check_id", sorted(_CALLS))
def test_a_check_fails_on_a_tool_error_envelope(check_id: str) -> None:
    # A rejected token or a malformed request: the tool reports it in its own
    # envelope with the MCP error flag unset.
    payload = json.dumps({"status": "error", "message": "401 Unauthorized"})

    assert _judge(check_id, payload) == (
        FailureClass.ASSERTION_FAILED,
        "status_success",
    )


@pytest.mark.parametrize("check_id", sorted(_CALLS))
def test_a_check_fails_when_the_list_is_missing(check_id: str) -> None:
    assert _judge(check_id, json.dumps({"status": "success"})) == (
        FailureClass.ASSERTION_FAILED,
        "has_keys",
    )


@pytest.mark.parametrize(
    ("check_id", "payload", "reason"),
    [
        (
            "linear.list_teams",
            {
                "status": "success",
                "teams": [{"id": "t1", "name": "x"}],
                "truncated": False,
            },
            "each_has_keys",
        ),
        (
            "linear.list_teams",
            {"status": "success", "teams": [], "truncated": "no"},
            "is_type",
        ),
        (
            "intercom.list_conversations",
            {
                "status": "success",
                "conversations": [{"id": "1"}],
                "total_count": 1,
                "has_more": False,
                "next_cursor": None,
            },
            "each_has_keys",
        ),
        (
            "deputy.employee_info",
            {"status": "success", "info": "not an object"},
            "is_type",
        ),
        (
            "employment-hero.list_organisations",
            {"status": "success", "organisations": {"total_items": 0}},
            "has_keys",
        ),
        (
            "employment-hero.list_organisations",
            {"status": "success", "organisations": {"items": [{}, {}]}},
            "length",
        ),
        (
            "xero.list_accounts",
            {"status": "success", "accounts": {}, "total_count": 0, "returned": 0},
            "is_type",
        ),
        (
            "xero.list_invoices",
            {
                "status": "success",
                "invoices": [{}, {}],
                "page": 1,
                "page_size": 1,
                "returned": 2,
                "has_more": False,
            },
            "length",
        ),
    ],
)
def test_a_check_fails_on_another_shape(
    check_id: str, payload: dict[str, Any], reason: str
) -> None:
    assert _judge(check_id, json.dumps(payload)) == (
        FailureClass.ASSERTION_FAILED,
        reason,
    )


def test_the_list_checks_use_a_named_tool_of_the_server() -> None:
    """Each tool a check calls exists, under that name, in its server module."""
    modules: dict[str, Callable[..., str]] = {}
    for app_module in (deputy, employment_hero, intercom, linear, xero):
        for name in dir(app_module):
            modules[name] = getattr(app_module, name)

    for check_id, (tool, _) in _CALLS.items():
        assert callable(modules.get(tool)), (check_id, tool)

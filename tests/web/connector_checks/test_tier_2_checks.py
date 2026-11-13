"""Smoke V2 Tier 2 checks of linear, intercom, deputy, employment-hero and xero (#3573).

The declarations are pinned in ``test_registry.py`` (``_TIER_2_READS``). This
runs each declared call against the real tool with only the vendor request
replaced by a realistic answer, judges what the tool returns with the check's
own assertions, and records the request, so a tool whose output shape drifts
from its declaration, or that stops sending the page size a check relies on,
fails here and not on the QA member.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

from xagent.web.connector_checks.assertions import judge_result
from xagent.web.connector_checks.declarations import ReadCheck
from xagent.web.connector_checks.registry import (
    _TIER_2,
    CONNECTOR_CHECKS,
    WRITE_WITHOUT_CLEANUP,
)
from xagent.web.connector_checks.results import FailureClass
from xagent.web.tools.mcp import deputy, employment_hero, intercom, linear, xero

MODULES: dict[str, ModuleType] = {
    "linear": linear,
    "intercom": intercom,
    "deputy": deputy,
    "employment-hero": employment_hero,
    "xero": xero,
}


def _check(check_id: str) -> ReadCheck:
    app_id = check_id.split(".", 1)[0]
    check = next(c for c in CONNECTOR_CHECKS[app_id].reads if c.id == check_id)
    assert isinstance(check, ReadCheck)
    return check


def _structural_ids(app_id: str) -> list[str]:
    return [
        c.id
        for c in CONNECTOR_CHECKS[app_id].reads
        if isinstance(c, ReadCheck) and c.id != f"{app_id}.l3_probe"
    ]


ALL_IDS = [cid for app in MODULES for cid in _structural_ids(app)]


def _judge(check_id: str, output: str) -> tuple[FailureClass | None, str | None]:
    check = _check(check_id)
    raw = {
        "content": [{"type": "text", "text": output}],
        "structured_content": None,
        "is_error": False,
    }
    return judge_result(raw, check.payload, check.assertions)


def _run(check_id: str) -> str:
    """The check's own call, made on the real tool: its arguments must fit it."""
    check = _check(check_id)
    tool: Callable[..., str] = getattr(MODULES[check_id.split(".", 1)[0]], check.tool)
    return tool(**check.arguments)


# --- what is declared ------------------------------------------------------------


@pytest.mark.parametrize("app_id", MODULES)
def test_tier_2_declares_reads_only(app_id: str) -> None:
    checks = CONNECTOR_CHECKS[app_id]

    assert checks.writes == () and checks.sweep is None
    assert _structural_ids(app_id)


@pytest.mark.parametrize("app_id", MODULES)
def test_every_exemption_names_tier_2_or_says_why_it_has_no_cleanup(
    app_id: str,
) -> None:
    for exemption in CONNECTOR_CHECKS[app_id].exemptions:
        assert exemption.reason == WRITE_WITHOUT_CLEANUP or exemption.reason.endswith(
            f"decided in {_TIER_2}"
        ), (app_id, exemption.tool)


# --- what the tools really return, and what they send ------------------------------


class _Capture:
    """The vendor requests of one tool call."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []


def _linear_node(field: str) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {
        "teams": {"id": "t1", "key": "ENG", "name": "Engineering"},
        "projects": {
            "id": "p1",
            "name": "Roadmap",
            "status": {"id": "s1", "name": "Planned", "type": "planned"},
            "progress": 0.5,
            "url": "https://linear.app/x/project/roadmap",
            "teams": {"nodes": [{"id": "t1", "key": "ENG", "name": "Engineering"}]},
        },
        "users": {
            "id": "u1",
            "name": "Ada",
            "email": "ada@example.com",
            "active": True,
        },
        "issues": {
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
    }
    return nodes[field]


_LINEAR = {
    "linear.list_teams": "teams",
    "linear.list_projects": "projects",
    "linear.search_users": "users",
    "linear.search_issues": "issues",
}


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("check_id", sorted(_LINEAR))
def test_a_linear_list_output_satisfies_its_declaration_and_asks_for_one_item(
    monkeypatch: pytest.MonkeyPatch, check_id: str, count: int
) -> None:
    field = _LINEAR[check_id]
    seen = _Capture()

    def graphql(query: str, variables: dict[str, Any] | None = None):
        seen.requests.append(variables or {})
        return (
            {field: {"nodes": [_linear_node(field)] * count, "pageInfo": {}}},
            None,
        )

    monkeypatch.setattr(linear, "_graphql", graphql)

    assert _judge(check_id, _run(check_id)) == (None, None)
    assert any(v.get("first") == 1 for v in seen.requests), seen.requests


@pytest.mark.parametrize("count", [0, 1])
def test_intercom_list_output_satisfies_its_declaration_and_asks_for_one_item(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    seen = _Capture()
    conversation = {
        "id": "1",
        "state": "open",
        "title": None,
        "source": {"subject": "Hello", "body": "<p>Hi</p>"},
        "created_at": 1700000000,
        "updated_at": 1700000100,
    }

    def request(method: str, path: str, *, body: dict[str, Any] | None = None):
        seen.requests.append(body or {})
        return {
            "conversations": [conversation] * count,
            "total_count": count,
            "pages": {},
        }

    monkeypatch.setattr(intercom, "_request", request)

    assert _judge(
        "intercom.list_conversations", _run("intercom.list_conversations")
    ) == (
        None,
        None,
    )
    (body,) = seen.requests
    assert body["pagination"]["per_page"] == 1
    # "all" is not a state filter: an open-only call would miss a closed
    # conversation, the only kind a QA workspace may hold.
    assert body["query"]["field"] != "state"


@pytest.mark.parametrize("count", [0, 1])
def test_deputy_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    monkeypatch.setattr(
        deputy, "_request", lambda *a, **k: [{"Id": 1, "CompanyName": "Demo"}] * count
    )

    assert _judge("deputy.list_companies", _run("deputy.list_companies")) == (
        None,
        None,
    )


def test_deputy_a_null_list_is_an_empty_one_and_still_satisfies_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deputy serializes "no records" as null on some installs.
    monkeypatch.setattr(deputy, "_request", lambda *a, **k: None)

    assert _judge("deputy.list_companies", _run("deputy.list_companies")) == (
        None,
        None,
    )


def test_deputy_resource_info_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deputy,
        "_request",
        lambda *a, **k: {"fields": {"Id": "Integer"}, "assocs": {}, "joins": {}},
    )

    assert _judge("deputy.employee_info", _run("deputy.employee_info")) == (None, None)


@pytest.mark.parametrize("count", [0, 1])
def test_employment_hero_organisations_output_satisfies_its_declaration_and_asks_for_one(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    seen = _Capture()

    def request(method: str, path: str, *, params: dict[str, Any] | None = None, **k):
        seen.requests.append(params or {})
        return {
            "data": {
                "items": [{"id": "o1", "name": "Demo", "country": "AU"}] * count,
                "page_index": 1,
                "item_per_page": 1,
                "total_items": count,
                "total_pages": 1,
            }
        }

    monkeypatch.setattr(employment_hero, "_request", request)

    check_id = "employment-hero.list_organisations"
    assert _judge(check_id, _run(check_id)) == (None, None)
    assert [r.get("item_per_page") for r in seen.requests] == [1]


# What a Xero list is called in the vendor's answer.
_XERO_UNPAGED = {
    "xero.list_accounts": "Accounts",
    "xero.list_contact_groups": "ContactGroups",
    "xero.list_items": "Items",
    "xero.list_tax_rates": "TaxRates",
    "xero.list_tracking_categories": "TrackingCategories",
}
# Every paged list but quotes sends `pageSize`; `GET /Quotes` has none.
_XERO_PAGED_CAPPED = {
    "xero.list_contacts": "Contacts",
    "xero.list_invoices": "Invoices",
    "xero.list_credit_notes": "CreditNotes",
    "xero.list_bank_transactions": "BankTransactions",
    "xero.list_payments": "Payments",
    "xero.list_manual_journals": "ManualJournals",
}


def _xero_request(
    seen: _Capture, vendor_key: str, count: int, *, pagination: bool
) -> Callable[..., Any]:
    def request(method: str, path: str, *, params: dict[str, Any] | None = None, **k):
        seen.requests.append(params or {})
        answer: dict[str, Any] = {vendor_key: [{}] * count}
        if pagination:
            answer["pagination"] = {
                "page": 1,
                "pageSize": 1,
                "pageCount": 1,
                "itemCount": count,
            }
        return answer

    return request


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("check_id", sorted(_XERO_UNPAGED))
def test_a_xero_unpaged_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, check_id: str, count: int
) -> None:
    monkeypatch.setattr(
        xero,
        "_request",
        _xero_request(_Capture(), _XERO_UNPAGED[check_id], count, pagination=False),
    )

    assert _judge(check_id, _run(check_id)) == (None, None)


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("check_id", sorted(_XERO_PAGED_CAPPED))
def test_a_xero_paged_list_output_satisfies_its_declaration_and_asks_for_one_item(
    monkeypatch: pytest.MonkeyPatch, check_id: str, count: int
) -> None:
    seen = _Capture()
    monkeypatch.setattr(
        xero,
        "_request",
        _xero_request(seen, _XERO_PAGED_CAPPED[check_id], count, pagination=True),
    )

    assert _judge(check_id, _run(check_id)) == (None, None)
    assert [r.get("pageSize") for r in seen.requests] == [1]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_xero_quotes_answer_a_whole_page_because_the_vendor_has_no_page_size(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    # `GET /Quotes` takes only `page` and answers no pagination block, so an
    # organisation with several quotes answers all of them to a page_size of 1.
    monkeypatch.setattr(
        xero,
        "_request",
        _xero_request(_Capture(), "Quotes", count, pagination=False),
    )

    assert _judge("xero.list_quotes", _run("xero.list_quotes")) == (None, None)


# --- what a check rejects ----------------------------------------------------------


@pytest.mark.parametrize("check_id", ALL_IDS)
def test_a_check_fails_on_a_tool_error_envelope(check_id: str) -> None:
    # A rejected token or a malformed request: the tool reports it in its own
    # envelope with the MCP error flag unset.
    output = json.dumps({"status": "error", "message": "401 Unauthorized"})

    assert _judge(check_id, output) == (
        FailureClass.ASSERTION_FAILED,
        "status_success",
    )


@pytest.mark.parametrize("check_id", ALL_IDS)
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
        # Not a list at all: the type is judged before the length, so this is
        # reported as what is wrong with it.
        (
            "xero.list_invoices",
            {
                "status": "success",
                "invoices": "none",
                "page": 1,
                "page_size": 1,
                "returned": 0,
                "has_more": False,
            },
            "is_type",
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

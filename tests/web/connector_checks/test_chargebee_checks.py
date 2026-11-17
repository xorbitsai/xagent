"""Smoke V2 checks of the Chargebee connector, run on the real tools.

Each declared call runs against the real tool with only the Chargebee request
replaced, and the check's own assertions judge what the tool returns. A tool
whose output drifts from its declaration, or that stops asking for one item,
fails here and not on the QA site.
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.web.connector_checks.assertions import judge_result
from xagent.web.connector_checks.declarations import (
    EachHasKeys,
    HasKeys,
    IsType,
    Length,
    ReadCheck,
    StatusSuccess,
)
from xagent.web.connector_checks.registry import _TIER_2, CONNECTOR_CHECKS
from xagent.web.tools.mcp import chargebee

# Only the attributes Chargebee's API reference marks as always present.
_ENTRIES = {
    "customers": {"customer": {"id": "cust_1", "created_at": 1700000000}},
    "subscriptions": {
        "subscription": {"id": "sub_1", "customer_id": "cust_1", "status": "active"}
    },
    "invoices": {"invoice": {"id": "inv_1", "customer_id": "cust_1", "status": "paid"}},
    "transactions": {
        "transaction": {"id": "txn_1", "type": "payment", "currency_code": "USD"}
    },
}


def _read_checks() -> list[ReadCheck]:
    return [c for c in CONNECTOR_CHECKS["chargebee"].reads if isinstance(c, ReadCheck)]


def _judge(check: ReadCheck, output: str) -> Any:
    raw = {
        "content": [{"type": "text", "text": output}],
        "structured_content": None,
        "is_error": False,
    }
    return judge_result(raw, check.payload, check.assertions)


def _run(
    monkeypatch: pytest.MonkeyPatch, check: ReadCheck, body: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def fake_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        seen.append(params or {})
        return body

    monkeypatch.setattr(chargebee, "_get", fake_get)
    return getattr(chargebee, check.tool)(**check.arguments), seen


def test_every_list_tool_is_checked_and_every_get_tool_is_pending():
    checks = CONNECTOR_CHECKS["chargebee"]
    assert {c.tool for c in _read_checks()} == {
        f"chargebee_list_{field}" for field in _ENTRIES
    }
    assert checks.writes == () and checks.sweep is None
    for exemption in checks.exemptions:
        assert exemption.tool.startswith("chargebee_get_")
        assert exemption.reason.endswith(f"decided in {_TIER_2}")


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("check", _read_checks(), ids=lambda c: c.id)
def test_a_list_output_satisfies_its_declaration_and_asks_for_one_item(
    monkeypatch: pytest.MonkeyPatch, check: ReadCheck, count: int
):
    field = check.tool.removeprefix("chargebee_list_")
    body = {"list": [_ENTRIES[field]] * count}
    output, seen = _run(monkeypatch, check, body)
    assert _judge(check, output) == (None, None)
    assert [params["limit"] for params in seen] == [1]


@pytest.mark.parametrize("check", _read_checks(), ids=lambda c: c.id)
def test_a_list_check_fails_on_an_error_envelope(
    monkeypatch: pytest.MonkeyPatch, check: ReadCheck
):
    output, _ = _run(monkeypatch, check, {"unexpected": []})
    failure, _ = _judge(check, output)
    assert failure is not None


# Each list check, pinned exactly: dropping an assertion fails here.
_PINS = {
    "chargebee.list_subscriptions": ("subscriptions", ("id", "customer_id", "status")),
    "chargebee.list_invoices": ("invoices", ("id", "customer_id", "status")),
    "chargebee.list_transactions": (
        "transactions",
        ("id", "type", "currency_code"),
    ),
}


@pytest.mark.parametrize("check_id", sorted(_PINS))
def test_a_list_check_is_declared_exactly(check_id: str):
    field, item_keys = _PINS[check_id]
    check = next(c for c in _read_checks() if c.id == check_id)
    assert check.tool == f"chargebee_list_{field}"
    assert check.arguments == {"limit": 1}
    assert check.assertions == (
        StatusSuccess(),
        HasKeys((), (field, "next_offset", "has_more", "truncated")),
        IsType(("has_more",), "boolean"),
        IsType((field,), "array"),
        EachHasKeys((field,), item_keys),
        Length((field,), max=1),
    )


@pytest.mark.parametrize("check_id", sorted(_PINS))
def test_a_list_check_fails_on_an_entry_without_a_required_key(
    monkeypatch: pytest.MonkeyPatch, check_id: str
):
    check = next(c for c in _read_checks() if c.id == check_id)
    field = check.tool.removeprefix("chargebee_list_")
    resource, item = next(iter(_ENTRIES[field].items()))
    output, _ = _run(monkeypatch, check, {"list": [{resource: {"id": item["id"]}}]})
    failure, _ = _judge(check, output)
    assert failure is not None


@pytest.mark.parametrize("check_id", sorted(_PINS))
def test_a_list_check_fails_on_more_than_one_entry(
    monkeypatch: pytest.MonkeyPatch, check_id: str
):
    check = next(c for c in _read_checks() if c.id == check_id)
    field = check.tool.removeprefix("chargebee_list_")
    output, _ = _run(monkeypatch, check, {"list": [_ENTRIES[field]] * 2})
    failure, _ = _judge(check, output)
    assert failure is not None

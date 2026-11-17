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
from xagent.web.connector_checks.declarations import ReadCheck
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

"""The post-run metering seam forwards the metered run's task to the hook."""

from __future__ import annotations

from typing import Any

import pytest

from xagent.web.services import quota_hooks


@pytest.fixture
def captured():
    calls: list[dict[str, Any]] = []

    def hook(db, user_id, delta_details, delta_actions, *, task_id=None):
        calls.append(
            {
                "db": db,
                "user_id": user_id,
                "details": delta_details,
                "actions": delta_actions,
                "task_id": task_id,
            }
        )

    quota_hooks.set_usage_record_hook(hook)
    try:
        yield calls
    finally:
        quota_hooks.set_usage_record_hook(None)


def test_record_usage_forwards_the_task_id_as_a_keyword(captured):
    details = [{"type": "input", "tokens": 3}]

    quota_hooks.record_usage("db", 7, details, 2, task_id=123)

    assert captured == [
        {"db": "db", "user_id": 7, "details": details, "actions": 2, "task_id": 123}
    ]


def test_record_usage_without_a_task_forwards_none(captured):
    quota_hooks.record_usage("db", 7, [], 0)

    assert [call["task_id"] for call in captured] == [None]


def test_record_usage_is_a_noop_without_a_hook_or_a_user(captured):
    quota_hooks.record_usage("db", None, [], 1, task_id=123)
    quota_hooks.set_usage_record_hook(None)
    quota_hooks.record_usage("db", 7, [], 1, task_id=123)

    assert captured == []

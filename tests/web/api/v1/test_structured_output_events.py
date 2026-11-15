"""The task.completed frame of GET /v1/chat/tasks/{id}/events carries the same
structured output fields as GET /v1/chat/tasks/{id}."""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent.result import COMPLETION_OUTCOMES
from xagent.web.api.v1 import _events_stream as es
from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.models.task import TaskStatus
from xagent.web.services import structured_output_projection
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)

from ..conftest import client
from .structured_output_shared import (
    COLUMN,
    EFFECTIVE_STATES,
    FORMAT_CONFIG,
    NEW_KEYS,
    RECORD,
    SWITCH_OFF_CONFIGS,
    SWITCH_ON_CONFIG,
    TODAY_COMPLETED_FRAME_KEYS,
    TODAY_TASK_INFO_KEYS,
    bearer,
    create_agent_with_key,
    insert_task,
    recorded_statements,
    task_selects,
    with_snapshot,
)
from .test_events_stream import _principal_for

pytestmark = pytest.mark.usefixtures("_test_db")


def completed_data(key: str, task_id: int) -> tuple[dict[str, Any], str]:
    """The parsed payload and the raw text of the stream's task.completed frame."""
    resp = client.get(f"/v1/chat/tasks/{task_id}/events", headers=bearer(key))
    assert resp.status_code == 200, resp.text
    blocks = [
        b for b in resp.text.split("\n\n") if b.startswith("event: task.completed")
    ]
    assert len(blocks) == 1, resp.text
    raw = blocks[0].split("data: ", 1)[1]
    return json.loads(raw), raw


def test_the_completed_frame_carries_the_ready_record_of_the_run():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, record=RECORD, agent_config=FORMAT_CONFIG)

    data, raw = completed_data(key, task_id)

    assert data["structured_output"] == RECORD["output"]
    assert data["structured_output_status"] == "ready"
    assert data["structured_output_reason"] is None
    assert raw.index('"zeta"') < raw.index('"alpha"')


def test_the_completed_frame_has_null_fields_for_a_task_without_a_format():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id)

    data, _raw = completed_data(key, task_id)

    assert [data[name] for name in NEW_KEYS] == [None, None, None]


def test_the_completed_frame_reports_not_produced_for_a_format_without_a_record():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, agent_config=FORMAT_CONFIG)

    data, _raw = completed_data(key, task_id)

    assert data["structured_output_status"] == "incomplete"
    assert data["structured_output_reason"]["code"] == "not_produced"


def test_the_frame_writes_non_ascii_characters_escaped():
    agent_id, key = create_agent_with_key()
    wide = chr(0x4E2D) + chr(0x6587) + chr(0x1F600)
    record = {**RECORD, "output": {"name": wide}}
    task_id = insert_task(agent_id, record=record, agent_config=FORMAT_CONFIG)

    data, raw = completed_data(key, task_id)

    assert data["structured_output"] == record["output"]
    assert raw.isascii()


@pytest.mark.parametrize("code", ["agent_partial", "model_error", "not_produced"])
def test_a_reason_without_field_lists_has_no_such_keys_in_the_frame_or_the_response(
    code,
):
    agent_id, key = create_agent_with_key()
    if code == "not_produced":
        task_id = insert_task(agent_id, agent_config=FORMAT_CONFIG)
    else:
        status = "failed" if code == "model_error" else "incomplete"
        record = {
            "run_id": "run-1",
            "status": status,
            "output": None,
            "reason": {"code": code},
        }
        task_id = insert_task(agent_id, record=record)

    _data, raw = completed_data(key, task_id)
    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key))

    for text in (raw, resp.text):
        assert "missing_fields" not in text
        assert "unexpressible_fields" not in text
        assert json.loads(text)["structured_output_reason"]["code"] == code


@pytest.mark.asyncio
async def test_the_watchdog_close_frame_carries_the_same_fields_as_the_response():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, record=RECORD, agent_config=FORMAT_CONFIG)
    principal = _principal_for(key)
    sink = es.V1EventStreamSink(
        task_id=task_id,
        principal_key_prefix=principal.key.key_prefix,
        initial_status="running",
    )

    closed = await es.watchdog_check_once(
        sink,
        task_id,
        principal,
        read_task_snapshot=lambda tid, p: v1_tasks.effective_task_snapshot(
            v1_tasks._load_task_info_snapshot(tid, p)
        ),
    )

    assert closed is True
    frame, _is_close = sink.queue.get_nowait()
    data = json.loads(frame.split("data: ", 1)[1])
    response = client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key)).json()
    assert {name: data[name] for name in NEW_KEYS} == {
        name: response[name] for name in NEW_KEYS
    }
    assert data["structured_output_status"] == "ready"


def test_the_response_and_the_stream_call_the_same_projection(monkeypatch):
    calls: list[str] = []
    real = structured_output_projection.project_structured_output

    def spy(name: str):
        def wrapper(**kwargs):
            calls.append(name)
            return real(**kwargs)

        return wrapper

    monkeypatch.setattr(v1_tasks, "project_structured_output", spy("get"))
    monkeypatch.setattr(es, "project_structured_output", spy("events"))
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, record=RECORD, agent_config=FORMAT_CONFIG)

    client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key))
    completed_data(key, task_id)

    assert calls == ["get", "events"]


def test_both_read_paths_import_the_one_projection_function():
    projection = structured_output_projection.project_structured_output
    assert v1_tasks.project_structured_output is projection
    assert es.project_structured_output is projection


def test_the_stream_reads_the_record_with_the_row_in_one_select():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, record=RECORD, agent_config=FORMAT_CONFIG)

    with recorded_statements() as statements:
        completed_data(key, task_id)

    selects = task_selects(statements)
    assert selects, statements
    assert all(f"tasks.{COLUMN}" in s for s in selects)


def test_a_frame_built_without_fields_carries_null_for_each():
    frame = es.completed_frame(
        status=TaskStatus.FAILED.value, output=None, error="x", error_code=None
    )

    data = json.loads(frame.split("data: ", 1)[1])
    assert [data[name] for name in NEW_KEYS] == [None, None, None]


def _incomplete_record(code: str, **extra: Any) -> dict[str, Any]:
    reason: dict[str, Any] = {"code": code}
    if code == "information_missing":
        reason["missing_fields"] = ["/g"]
    if code == "value_not_expressible":
        reason["unexpressible_fields"] = ["/g"]
    return {
        "run_id": "run-1",
        "status": "incomplete",
        "output": None,
        "reason": reason,
        **extra,
    }


PARTIAL = {"name": "x", "g": None}
RETURNED_RECORDS = {
    "gap-code-with-values": _incomplete_record(
        "information_missing", partial_output=PARTIAL
    ),
    "agent-partial-with-values": _incomplete_record(
        "agent_partial", partial_output=PARTIAL
    ),
    "agent-partial-with-values-and-lists": _incomplete_record(
        "agent_partial",
        partial_output=PARTIAL,
        partial_gaps={"missing_fields": ["/g"], "unexpressible_fields": ["/n"]},
    ),
    "agent-blocked-with-values": _incomplete_record(
        "agent_blocked", partial_output=PARTIAL
    ),
    "agent-blocked-with-values-and-lists": _incomplete_record(
        "agent_blocked",
        partial_output=PARTIAL,
        partial_gaps={"missing_fields": ["/g"]},
    ),
    "incomplete-without-values": _incomplete_record("agent_partial"),
    "ready": RECORD,
}


@pytest.mark.parametrize("name", list(RETURNED_RECORDS))
def test_the_response_and_the_frame_return_the_same_result_and_reason(name):
    record = RETURNED_RECORDS[name]
    assert len(RETURNED_RECORDS) == 7
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, record=record, agent_config=FORMAT_CONFIG)

    data, _raw = completed_data(key, task_id)
    response = client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key)).json()

    assert {k: data[k] for k in NEW_KEYS} == {k: response[k] for k in NEW_KEYS}
    expected = record.get("partial_output", record["output"])
    assert response["structured_output"] == expected
    assert response["structured_output_status"] == record["status"]
    if record["reason"] is not None:
        assert response["structured_output_reason"]["code"] == record["reason"]["code"]


# --- the completion_outcome key -------------------------------------------------

SURFACES = ["get-miss", "get-hit", "sse-attach", "sse-watchdog"]
# A queued start has no conclusion frame, so only GET shows that state.
CASES = [
    pytest.param(surface, rating, changes, id=f"{surface}-{name}")
    for surface in SURFACES
    for name, rating, changes in EFFECTIVE_STATES
    if surface.startswith("get") or "start_queued" not in changes
]


async def _watchdog_frame(key: str, task_id: int) -> dict[str, Any]:
    principal = _principal_for(key)
    sink = es.V1EventStreamSink(
        task_id=task_id,
        principal_key_prefix=principal.key.key_prefix,
        initial_status="running",
    )
    closed = await es.watchdog_check_once(
        sink,
        task_id,
        principal,
        read_task_snapshot=lambda tid, p: v1_tasks.effective_task_snapshot(
            v1_tasks._load_task_info_snapshot(tid, p)
        ),
    )
    assert closed is True
    frame, _is_close = sink.queue.get_nowait()
    return json.loads(frame.split("data: ", 1)[1])


async def _read(surface: str, key: str, task_id: int, request) -> dict[str, Any]:
    """The parsed body of a GET response or of a conclusion frame."""
    if surface == "sse-attach":
        return completed_data(key, task_id)[0]
    if surface == "sse-watchdog":
        return await _watchdog_frame(key, task_id)
    if surface == "get-hit":
        set_cache_backend_for_testing(InMemoryTTLCache())
        request.addfinalizer(lambda: set_cache_backend_for_testing(None))
        client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key))
    return client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key)).json()


def _today_keys(surface: str) -> tuple[str, ...]:
    return (
        TODAY_TASK_INFO_KEYS
        if surface.startswith("get")
        else TODAY_COMPLETED_FRAME_KEYS
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("surface", "rating", "changes"), CASES)
@pytest.mark.parametrize("config", SWITCH_OFF_CONFIGS.values(), ids=SWITCH_OFF_CONFIGS)
async def test_switch_off_responses_have_no_completion_outcome(
    config, surface, rating, changes, monkeypatch, request
):
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, agent_config=config, completion_outcome=rating)
    with_snapshot(monkeypatch, **changes)

    body = await _read(surface, key, task_id, request)

    assert tuple(body) == _today_keys(surface)


@pytest.mark.asyncio
@pytest.mark.parametrize(("surface", "rating", "changes"), CASES)
async def test_switch_on_responses_end_with_the_reported_outcome(
    surface, rating, changes, monkeypatch, request
):
    agent_id, key = create_agent_with_key()
    task_id = insert_task(
        agent_id, agent_config=SWITCH_ON_CONFIG, completion_outcome=rating
    )
    with_snapshot(monkeypatch, **changes)

    body = await _read(surface, key, task_id, request)

    assert tuple(body) == (*_today_keys(surface), "completion_outcome")
    assert body["completion_outcome"] == (rating if not changes else None)


@pytest.mark.parametrize("rating", [*sorted(COMPLETION_OUTCOMES), None])
def test_the_response_and_the_frame_return_the_same_completion_outcome(rating):
    agent_id, key = create_agent_with_key()
    task_id = insert_task(
        agent_id,
        record=RECORD,
        agent_config=SWITCH_ON_CONFIG,
        completion_outcome=rating,
    )

    data, _raw = completed_data(key, task_id)
    response = client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key)).json()

    assert data["completion_outcome"] == response["completion_outcome"] == rating


@pytest.mark.parametrize(
    ("reported", "snapshot_total_steps", "expected_tail"),
    [
        (None, None, ()),
        (
            {"completion_outcome": "partial"},
            3,
            ("completion_outcome", "snapshot_truncated", "snapshot_total_steps"),
        ),
    ],
    ids=["not-given", "before-the-truncation-marker"],
)
def test_the_completion_outcome_key_of_a_built_frame(
    reported, snapshot_total_steps, expected_tail
):
    frame = es.completed_frame(
        status=TaskStatus.COMPLETED.value,
        output="x",
        error=None,
        error_code=None,
        reported_outcome=reported,
        snapshot_total_steps=snapshot_total_steps,
    )

    data = json.loads(frame.split("data: ", 1)[1])
    assert tuple(data) == (*TODAY_COMPLETED_FRAME_KEYS, *expected_tail)

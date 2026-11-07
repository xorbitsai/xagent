"""GET /v1/chat/tasks/{id}: the structured output record column and the fields
projected from it."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tests.web.services.test_structured_output_projection import MESSAGES
from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.models.task import Task
from xagent.web.schemas.v1 import TaskInfoResponse
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    cache_get,
    cache_set,
    set_cache_backend_for_testing,
    task_snapshot_key,
)
from xagent.web.services.task_auto_resume import TerminalView

from ..conftest import client
from .structured_output_get_baseline import BASELINE_BODIES
from .structured_output_shared import (
    BASELINE_TASKS,
    COLUMN,
    FORMAT_CONFIG,
    MOMENT,
    NEW_KEYS,
    RECORD,
    bearer,
    create_agent_with_key,
    insert_task,
    recorded_statements,
    task_selects,
)

pytestmark = pytest.mark.usefixtures("_test_db")


def test_the_column_is_a_plain_mapped_column_and_the_response_allows_extra_keys():
    """The record is read with the rest of the row, never lazily afterwards."""
    column = Task.__table__.c[COLUMN]
    prop = Task.__mapper__.get_property(COLUMN)

    assert not prop.deferred
    assert column.nullable
    assert TaskInfoResponse.model_config.get("extra") != "forbid"


def test_get_reads_the_record_column_in_the_same_select_as_the_row():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(
        agent_id,
        record={"run_id": "run-1", "status": "ready", "output": {}, "reason": None},
    )

    with recorded_statements() as statements:
        resp = client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key))

    assert resp.status_code == 200, resp.text
    selects = task_selects(statements)
    assert selects, statements
    assert all(f"tasks.{COLUMN}" in s for s in selects)


# --- the three response fields ------------------------------------------------

NOT_PRODUCED = {
    "code": "not_produced",
    "message": "No structured result was produced for the run shown in this response.",
}


@pytest.fixture
def task_cache():
    """A working cache backend: the default one stores nothing."""
    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        yield
    finally:
        set_cache_backend_for_testing(None)


def get(key: str, task_id: int):
    resp = client.get(f"/v1/chat/tasks/{task_id}", headers=bearer(key))
    assert resp.status_code == 200, resp.text
    return resp


def test_the_response_models_new_fields_all_default_to_none():
    """A downstream embedder that builds the response without them still can."""
    response = TaskInfoResponse(
        task_id=1, agent_id=2, status="completed", created_at=MOMENT
    )

    assert response.structured_output is None
    assert response.structured_output_status is None
    assert response.structured_output_reason is None


def test_a_ready_record_of_the_current_run_is_returned_in_stored_key_order():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, record=RECORD, agent_config=FORMAT_CONFIG)

    resp = get(key, task_id)

    assert resp.json()["structured_output"] == RECORD["output"]
    assert resp.json()["structured_output_status"] == "ready"
    assert resp.json()["structured_output_reason"] is None
    assert resp.text.index('"zeta"') < resp.text.index('"alpha"')


def test_a_task_without_a_format_or_a_record_reports_null_fields():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id)

    body = get(key, task_id).json()

    assert [body[name] for name in NEW_KEYS] == [None, None, None]


def test_a_completed_task_with_a_format_and_no_record_was_not_produced():
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, agent_config=FORMAT_CONFIG)

    body = get(key, task_id).json()

    assert body["structured_output"] is None
    assert body["structured_output_status"] == "incomplete"
    assert body["structured_output_reason"] == NOT_PRODUCED


def test_the_existing_fields_are_byte_identical_to_the_recorded_baseline():
    """The response, less the three new keys, is what it was before them."""
    agent_id, key = create_agent_with_key()

    for spec, expected in zip(BASELINE_TASKS, BASELINE_BODIES, strict=True):
        task_id = insert_task(agent_id, moment=MOMENT, **spec)
        body = json.loads(get(key, task_id).text)
        assert all(name in body for name in NEW_KEYS)
        for name in NEW_KEYS:
            del body[name]
        assert json.dumps(body, ensure_ascii=False, separators=(",", ":")) == expected


# A record of the run the task shows, with a format configured: what a reader
# would report if nothing hid it.
def _with_snapshot(monkeypatch, **changes):
    real = v1_tasks._load_task_info_snapshot
    monkeypatch.setattr(
        v1_tasks,
        "_load_task_info_snapshot",
        lambda task_id, principal: replace(real(task_id, principal), **changes),
    )


def test_a_record_of_an_earlier_run_is_not_shown_for_the_current_run():
    agent_id, key = create_agent_with_key()
    stale = insert_task(
        agent_id, run_id="run-2", record=RECORD, agent_config=FORMAT_CONFIG
    )
    unformatted = insert_task(agent_id, run_id="run-2", record=RECORD)

    shown = get(key, stale).json()
    hidden = get(key, unformatted).json()

    assert shown["structured_output"] is None
    assert shown["structured_output_status"] == "incomplete"
    assert shown["structured_output_reason"] == NOT_PRODUCED
    assert [hidden[name] for name in NEW_KEYS] == [None, None, None]


def test_a_queued_start_hides_the_record_of_the_previous_run(monkeypatch):
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, record=RECORD, agent_config=FORMAT_CONFIG)
    _with_snapshot(monkeypatch, start_queued=True)

    body = get(key, task_id).json()

    assert body["status"] == "pending"
    assert [body[name] for name in NEW_KEYS] == [None, None, None]


def test_a_terminal_view_hides_the_record(monkeypatch):
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, record=RECORD, agent_config=FORMAT_CONFIG)
    view = TerminalView(state="stopped", error="stopped", stopped_at=MOMENT, reason="x")
    _with_snapshot(monkeypatch, view=view)

    body = get(key, task_id).json()

    assert body["status"] == "failed"
    assert [body[name] for name in NEW_KEYS] == [None, None, None]


def test_the_three_fields_are_never_taken_from_the_cached_body(task_cache):
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id)
    get(key, task_id)
    entry = cache_get(task_snapshot_key(task_id))
    assert entry is not None
    entry["response"].update(
        structured_output={"planted": True},
        structured_output_status="ready",
        structured_output_reason=None,
    )
    cache_set(task_snapshot_key(task_id), entry, ttl_seconds=60)

    body = get(key, task_id).json()

    assert [body[name] for name in NEW_KEYS] == [None, None, None]


def test_the_cache_body_holds_no_values_for_the_three_fields(task_cache):
    agent_id, key = create_agent_with_key()
    task_id = insert_task(agent_id, record=RECORD, agent_config=FORMAT_CONFIG)

    assert get(key, task_id).json()["structured_output_status"] == "ready"

    cached = cache_get(task_snapshot_key(task_id))["response"]
    assert [cached[name] for name in NEW_KEYS] == [None, None, None]


CODES = {code: text for code, text in MESSAGES.items() if code != "not_produced"}


INCOMPLETE_CODES = tuple(CODES)[:4]


@pytest.mark.parametrize("code", CODES)
def test_the_message_of_a_reason_is_the_fixed_text_of_its_code(code):
    agent_id, key = create_agent_with_key()
    reason: dict = {"code": code, "message": f"OWNER-DIAG-{code}"}
    if code == "information_missing":
        reason["missing_fields"] = ["/a"]
    if code == "value_not_expressible":
        reason["unexpressible_fields"] = ["/a"]
    status = "incomplete" if code in INCOMPLETE_CODES else "failed"
    record = {"run_id": "run-1", "status": status, "output": None, "reason": reason}
    task_id = insert_task(agent_id, record=record)

    resp = get(key, task_id)

    assert "OWNER-DIAG" not in resp.text
    assert resp.json()["structured_output_reason"]["code"] == code
    assert resp.json()["structured_output_reason"]["message"] == CODES[code]
    if code != "information_missing":
        assert "missing_fields" not in resp.text
    if code not in ("information_missing", "value_not_expressible"):
        assert "unexpressible_fields" not in resp.text

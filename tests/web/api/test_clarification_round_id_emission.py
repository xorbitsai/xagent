"""The clarification round id (``request_id``) on task-state frames.

Issue #1500: a client gating clarification retries must bind replies to the
exact ask. The runtime mints one ``event_id`` per ask; these tests pin that
every surface a waiting round reaches the client through carries it as
``request_id`` — the live/resume waiting ``task_info``, the history-replay
``task_info``, and the history-replay ``task_waiting_for_user`` reassertion.
(The question-less lease-restore corrective broadcast deliberately stays
id-less; the frontend preserves a known id across it.)
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tests.web.api.conftest import _direct_db_session, _test_db

__all__ = ["_test_db"]
from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    _clarification_request_id,
    _latest_ask_event_id,
)
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User


def test_request_id_reads_a_dict_draft() -> None:
    result = {"clarification_draft": {"event_id": "evt-1"}}
    assert _clarification_request_id(result) == "evt-1"


def test_request_id_reads_a_dataclass_draft() -> None:
    result = {"clarification_draft": SimpleNamespace(event_id="evt-2")}
    assert _clarification_request_id(result) == "evt-2"


@pytest.mark.parametrize(
    "result",
    [
        None,
        "waiting",
        {},
        {"clarification_draft": None},
        {"clarification_draft": {}},
        {"clarification_draft": {"event_id": ""}},
        {"clarification_draft": SimpleNamespace(event_id=None)},
    ],
    ids=[
        "non-dict-none",
        "non-dict-str",
        "no-draft",
        "none-draft",
        "empty-dict-draft",
        "empty-id",
        "none-id-attr",
    ],
)
def test_request_id_degrades_to_none(result: object) -> None:
    assert _clarification_request_id(result) is None


def _trace_row(event_id: str | None, *, expect_response: bool = True) -> object:
    data: dict[str, object] = {"expect_response": expect_response}
    if event_id is not None:
        data["event_id"] = event_id
    return SimpleNamespace(data=data)


def test_latest_ask_wins_over_an_older_round() -> None:
    rows = [_trace_row("evt-old"), SimpleNamespace(data=None), _trace_row("evt-new")]
    assert _latest_ask_event_id(rows) == "evt-new"


def test_an_id_less_newest_ask_yields_no_identity() -> None:
    # The scan must not reach past the newest ask into an older round's id:
    # a stale identity on a newer question is worse than none.
    rows = [_trace_row("evt-old"), _trace_row(None)]
    assert _latest_ask_event_id(rows) is None


def test_non_ask_rows_are_ignored() -> None:
    rows = [
        _trace_row("evt-ask"),
        SimpleNamespace(data={"event_id": "evt-progress"}),
        SimpleNamespace(data={"expect_response": False, "event_id": "evt-no"}),
    ]
    assert _latest_ask_event_id(rows) == "evt-ask"
    assert _latest_ask_event_id([]) is None


def _waiting_task_with_asks(ask_event_ids: list[str | None]) -> tuple[int, int]:
    db = _direct_db_session()
    try:
        user = User(username="round-id-replay-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title="Round id replay",
            description="Round id replay",
            status=TaskStatus.WAITING_FOR_USER,
        )
        db.add(task)
        db.flush()
        base = datetime.now(timezone.utc) - timedelta(minutes=10)
        for index, ask_event_id in enumerate(ask_event_ids):
            data: dict[str, object] = {
                "expect_response": True,
                "message": f"Question {index}?",
                "metadata": {
                    "interactions": [
                        {"type": "text_input", "field": "answer", "label": "Answer"}
                    ]
                },
            }
            if ask_event_id is not None:
                data["event_id"] = ask_event_id
            db.add(
                TraceEvent(
                    task_id=int(task.id),
                    event_id=ask_event_id or f"row-{index}",
                    event_type="agent_message",
                    timestamp=base + timedelta(minutes=index),
                    data=data,
                )
            )
        db.commit()
        return int(task.id), int(user.id)
    finally:
        db.close()


async def _replay(
    monkeypatch: pytest.MonkeyPatch, task_id: int, user_id: int
) -> list[dict]:
    sent: list[dict] = []

    async def send_personal_message(event: dict, _websocket: object) -> None:
        sent.append(event)

    monkeypatch.setattr(websocket_api, "cache_get", lambda _key: None)
    monkeypatch.setattr(websocket_api, "cache_set", lambda *_a, **_k: None)
    monkeypatch.setattr(
        websocket_api.manager, "send_personal_message", send_personal_message
    )
    await websocket_api.send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )
    return sent


@pytest.mark.asyncio
async def test_replay_carries_the_newest_ask_id_on_both_waiting_frames(
    monkeypatch: pytest.MonkeyPatch,
    _test_db: None,
) -> None:
    task_id, user_id = _waiting_task_with_asks(["evt-round-1", "evt-round-2"])

    sent = await _replay(monkeypatch, task_id, user_id)

    task_infos = [
        event
        for event in sent
        if event.get("event_type") == "task_info"
        and isinstance(event.get("data"), dict)
    ]
    assert task_infos, "replay produced no task_info"
    assert task_infos[0]["data"]["request_id"] == "evt-round-2"

    reasserts = [
        event for event in sent if event.get("type") == "task_waiting_for_user"
    ]
    assert reasserts, "replay produced no waiting reassertion"
    assert reasserts[0]["request_id"] == "evt-round-2"


@pytest.mark.asyncio
async def test_replay_without_a_persisted_ask_stays_id_less(
    monkeypatch: pytest.MonkeyPatch,
    _test_db: None,
) -> None:
    task_id, user_id = _waiting_task_with_asks([])

    sent = await _replay(monkeypatch, task_id, user_id)

    task_infos = [
        event
        for event in sent
        if event.get("event_type") == "task_info"
        and isinstance(event.get("data"), dict)
    ]
    assert task_infos
    assert "request_id" not in task_infos[0]["data"]
    reasserts = [
        event for event in sent if event.get("type") == "task_waiting_for_user"
    ]
    assert reasserts
    assert "request_id" not in reasserts[0]

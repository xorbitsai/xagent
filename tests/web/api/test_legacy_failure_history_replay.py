from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tests.web.api.conftest import _direct_db_session
from xagent.web.api import websocket as websocket_api
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.assistant_history_safety import (
    CLIENT_SAFE_FAILURE_MESSAGE_TYPE,
)
from xagent.web.services.chat_history_service import persist_assistant_message_no_commit
from xagent.web.services.hot_path_cache import (
    cache_version_token,
    web_task_history_key,
)
from xagent.web.services.managed_task_lease import finalize_managed_task_lease_result
from xagent.web.services.task_lease_service import TaskLease
from xagent.web.services.task_orchestrator import settle_task_lease_isolated

CLIENT_SAFE_TASK_FAILURE = "Task execution failed."


def _running_task(
    *,
    username: str,
    title: str,
    runner_id: str | None = None,
    run_id: str | None = None,
) -> tuple[int, int]:
    db = _direct_db_session()
    try:
        user = User(username=username, password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title=title,
            description=title,
            status=TaskStatus.RUNNING,
            runner_id=runner_id,
            run_id=run_id,
            lease_expires_at=(
                datetime.now(timezone.utc) + timedelta(minutes=5)
                if runner_id is not None
                else None
            ),
        )
        db.add(task)
        db.commit()
        return int(task.id), int(user.id)
    finally:
        db.close()


def _assert_safe_failure_persisted(task_id: int, raw_error: str) -> None:
    db = _direct_db_session()
    try:
        row = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .one()
        )
        task = db.get(Task, task_id)
        assert task is not None
        assert row.message_type == "task_failure"
        assert row.content == CLIENT_SAFE_TASK_FAILURE
        assert row.interactions is None
        assert task.error_message == raw_error
    finally:
        db.close()


def _failed_task_with_message(
    *,
    content: str,
    message_type: str,
    interactions: list[dict] | None = None,
    attachments: list[dict] | None = None,
) -> tuple[int, int]:
    db = _direct_db_session()
    try:
        user = User(username=f"legacy-history-{message_type}", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title="Legacy failure history",
            description="Legacy failure history",
            status=TaskStatus.FAILED,
            error_message=content,
        )
        db.add(task)
        db.flush()
        db.add(
            TaskChatMessage(
                task_id=int(task.id),
                user_id=int(user.id),
                role="assistant",
                content=content,
                message_type=message_type,
                interactions=interactions,
                turn_id=None,
                attachments=attachments,
            )
        )
        db.commit()
        return int(task.id), int(user.id)
    finally:
        db.close()


async def _replay_history(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_id: int,
    user_id: int,
    cache_values: dict[str, object] | None = None,
) -> list[dict]:
    sent_events: list[dict] = []

    async def send_personal_message(event: dict, _websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr(
        websocket_api,
        "cache_get",
        lambda key: (cache_values or {}).get(key),
    )
    monkeypatch.setattr(websocket_api, "cache_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        websocket_api.manager,
        "send_personal_message",
        send_personal_message,
    )

    await websocket_api.send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )
    return sent_events


def _history_cache_entry(task_id: int, events: list[dict]) -> dict[str, object]:
    db = _direct_db_session()
    try:
        task = db.get(Task, task_id)
        assert task is not None
        max_chat_message_id = (
            db.query(TaskChatMessage.id)
            .filter(TaskChatMessage.task_id == task_id)
            .order_by(TaskChatMessage.id.desc())
            .limit(1)
            .scalar()
            or 0
        )
        return {
            "trace_scope": "public-v1",
            "updated_at": cache_version_token(task.updated_at),
            "max_trace_event_id": 0,
            "max_chat_message_id": int(max_chat_message_id),
            "events": events,
        }
    finally:
        db.close()


@pytest.mark.asyncio
async def test_legacy_failure_row_is_redacted_on_cache_miss(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_error = "provider failed at /srv/private/config with token=secret"
    task_id, user_id = _failed_task_with_message(
        content=raw_error,
        message_type="chat_response",
    )

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert raw_error not in repr(events)
    assert any(
        event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == CLIENT_SAFE_TASK_FAILURE
        for event in events
    )


@pytest.mark.asyncio
async def test_no_commit_persistence_without_provenance_fails_closed_on_replay(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_content = "unproven no-commit token=secret"
    task_id, user_id = _running_task(
        username="unproven-no-commit-history",
        title="Unproven no-commit history",
    )
    db = _direct_db_session()
    try:
        row = persist_assistant_message_no_commit(
            db,
            task_id,
            user_id,
            raw_content,
        )
        assert row is not None
        assert row.message_type == "assistant_message"
        db.commit()
    finally:
        db.close()

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert raw_content not in repr(events)
    assert any(
        event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == CLIENT_SAFE_TASK_FAILURE
        for event in events
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "This response was interrupted.",
        "Required MCP servers are unavailable.",
    ],
)
async def test_explicitly_safe_failure_replays_unchanged(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    task_id, user_id = _failed_task_with_message(
        content=content,
        message_type=CLIENT_SAFE_FAILURE_MESSAGE_TYPE,
    )

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert any(
        event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == content
        for event in events
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_type", "content"),
    [
        ("chat_response", "A legitimate pre-cutover assistant response."),
        ("assistant_message", "Legacy managed failure token=secret"),
        ("final_answer", "Legacy websocket failure token=secret"),
        ("unknown_assistant_type", "Unknown provenance token=secret"),
    ],
)
async def test_unproven_assistant_history_is_intentionally_redacted(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
    message_type: str,
    content: str,
) -> None:
    task_id, user_id = _failed_task_with_message(
        content=content,
        message_type=message_type,
    )

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert content not in repr(events)
    assert any(
        event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == CLIENT_SAFE_TASK_FAILURE
        for event in events
    )


@pytest.mark.asyncio
async def test_unsafe_legacy_assistant_ancillary_payload_is_not_replayed(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interaction_secret = "legacy interaction token=secret"
    attachment_secret = "legacy attachment token=secret"
    task_id, user_id = _failed_task_with_message(
        content="legacy failure",
        message_type="task_failure",
        interactions=[{"label": interaction_secret}],
        attachments=[{"file_id": attachment_secret, "type": "image/png"}],
    )

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert interaction_secret not in repr(events)
    assert attachment_secret not in repr(events)


@pytest.mark.asyncio
async def test_failure_trace_and_chat_are_both_redacted_on_history_replay(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_error = "trace context exposed provider token=secret"
    task_id, user_id = _failed_task_with_message(
        content=raw_error,
        message_type="task_failure",
    )
    db = _direct_db_session()
    try:
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id="failure-trace-1730",
                event_type="trace_error",
                timestamp=datetime.now(timezone.utc),
                data={
                    "status": "failed",
                    "execution_id": "execution-1730",
                    "error_message": raw_error,
                    "context": {"messages": [{"content": raw_error}]},
                },
            )
        )
        db.add(
            TraceEvent(
                task_id=task_id,
                event_id="failed-pattern-end-1730",
                event_type="dag_execute_end",
                timestamp=datetime.now(timezone.utc),
                data={
                    "status": "failed",
                    "result": {"success": False, "error": raw_error},
                },
            )
        )
        db.commit()
    finally:
        db.close()

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert raw_error not in repr(events)
    assert any(
        event.get("event_type") == "dag_execute_end"
        and event.get("data", {}).get("result") == {"success": False}
        for event in events
    )
    assert any(
        event.get("event_type") == "trace_error"
        and event.get("data")
        == {
            "status": "failed",
            "execution_id": "execution-1730",
            "error_message": CLIENT_SAFE_TASK_FAILURE,
        }
        for event in events
    )


@pytest.mark.asyncio
async def test_pre_cutover_cached_failure_event_is_not_replayed(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_error = "cached provider payload contains credential=secret"
    task_id, user_id = _failed_task_with_message(
        content=raw_error,
        message_type="chat_response",
    )
    unsafe_cached_event = {
        "type": "trace_event",
        "event_id": "chat_message_legacy",
        "event_type": "agent_message",
        "task_id": task_id,
        "timestamp": 0,
        "data": {"message": raw_error, "content": raw_error},
    }
    cache_values = {
        f"task:web:history:{task_id}": _history_cache_entry(
            task_id,
            [unsafe_cached_event],
        )
    }

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
        cache_values=cache_values,
    )

    assert raw_error not in repr(events)
    assert any(
        event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == CLIENT_SAFE_TASK_FAILURE
        for event in events
    )


@pytest.mark.asyncio
async def test_v2_history_cache_hit_replays_cached_events_without_rebuilding(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id, user_id = _failed_task_with_message(
        content="database history that must not be rebuilt",
        message_type="chat_response",
    )
    cached_event = {
        "type": "trace_event",
        "event_id": "distinct-v2-cache-hit",
        "event_type": "agent_message",
        "task_id": task_id,
        "timestamp": 123.0,
        "data": {
            "message": "safe cached response",
            "content": "safe cached response",
        },
    }
    expected_key = web_task_history_key(task_id)
    requested_keys: list[str] = []
    cache_set_calls: list[tuple[object, ...]] = []
    sent_events: list[dict] = []

    def cache_get(key: str) -> object | None:
        requested_keys.append(key)
        return _history_cache_entry(task_id, [cached_event])

    async def send_personal_message(event: dict, _websocket: object) -> None:
        sent_events.append(event)

    monkeypatch.setattr(websocket_api, "cache_get", cache_get)
    monkeypatch.setattr(
        websocket_api,
        "cache_set",
        lambda *args, **kwargs: cache_set_calls.append((*args, kwargs)),
    )
    monkeypatch.setattr(
        websocket_api.manager,
        "send_personal_message",
        send_personal_message,
    )

    await websocket_api.send_historical_data_as_stream(
        websocket=object(),
        task_id=task_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )

    assert requested_keys == [expected_key]
    assert expected_key == f"task:web:history:v2:{task_id}"
    assert sent_events == [cached_event]
    assert cache_set_calls == []


@pytest.mark.asyncio
async def test_marked_failure_row_stays_redacted_after_a_later_run(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_error = "earlier run failed with database password=secret"
    task_id, user_id = _failed_task_with_message(
        content=raw_error,
        message_type="task_failure",
    )
    db = _direct_db_session()
    try:
        task = db.get(Task, task_id)
        assert task is not None
        task.status = TaskStatus.COMPLETED
        task.error_message = None
        db.commit()
    finally:
        db.close()

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert raw_error not in repr(events)
    assert any(
        event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == CLIENT_SAFE_TASK_FAILURE
        for event in events
    )


@pytest.mark.asyncio
async def test_new_plain_assistant_response_replays_unchanged(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal_answer = "Here is the requested deployment summary."
    task_id, user_id = _running_task(
        username="marked-normal-history",
        title="Normal response history",
    )

    finalized = websocket_api._finalize_task_execution_result_isolated(
        task_id=task_id,
        task_user_id=user_id,
        pre_run_status=TaskStatus.RUNNING,
        result={
            "success": True,
            "output": normal_answer,
            "chat_response": {"message": normal_answer},
        },
        expected_run_id=None,
        task_lease=None,
        resolved_scope_segments=(),
        prepared_outputs=websocket_api._PreparedTaskFileOutputs((), (), ()),
    )
    assert not finalized.late_result

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert any(
        event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == normal_answer
        for event in events
    )
    assert CLIENT_SAFE_TASK_FAILURE not in repr(events)


def test_failed_websocket_result_with_null_diagnostics_uses_safe_fallback(
    _test_db,
) -> None:
    task_id, user_id = _running_task(
        username="failed-websocket-null-diagnostics",
        title="Failed WebSocket null diagnostics",
    )

    finalized = websocket_api._finalize_task_execution_result_isolated(
        task_id=task_id,
        task_user_id=user_id,
        pre_run_status=TaskStatus.RUNNING,
        result={
            "success": False,
            "status": "error",
            "output": None,
            "error": None,
        },
        expected_run_id=None,
        task_lease=None,
        resolved_scope_segments=(),
        prepared_outputs=websocket_api._PreparedTaskFileOutputs((), (), ()),
    )

    assert not finalized.late_result
    db = _direct_db_session()
    try:
        task = db.get(Task, task_id)
        assert task is not None
        assert task.error_message == CLIENT_SAFE_TASK_FAILURE
    finally:
        db.close()


def test_successful_websocket_result_with_null_output_skips_empty_history(
    _test_db,
) -> None:
    task_id, user_id = _running_task(
        username="successful-websocket-null-output",
        title="Successful WebSocket null output",
    )

    finalized = websocket_api._finalize_task_execution_result_isolated(
        task_id=task_id,
        task_user_id=user_id,
        pre_run_status=TaskStatus.RUNNING,
        result={"success": True, "output": None},
        expected_run_id=None,
        task_lease=None,
        resolved_scope_segments=(),
        prepared_outputs=websocket_api._PreparedTaskFileOutputs((), (), ()),
    )

    assert not finalized.late_result
    db = _direct_db_session()
    try:
        rows = (
            db.query(TaskChatMessage)
            .filter(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.role == "assistant",
            )
            .all()
        )
        assert rows == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_failed_websocket_result_replays_only_safe_history(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_error = "Agent 'private-agent-name' has no LLM configured."
    interaction_secret = "interaction token=secret"
    task_id, user_id = _running_task(
        username="failed-websocket-result",
        title="Failed WebSocket result",
    )

    finalized = websocket_api._finalize_task_execution_result_isolated(
        task_id=task_id,
        task_user_id=user_id,
        pre_run_status=TaskStatus.RUNNING,
        result={
            "success": False,
            "status": "error",
            "output": raw_error,
            "chat_response": {
                "message": raw_error,
                "interactions": [{"label": interaction_secret}],
            },
        },
        expected_run_id=None,
        task_lease=None,
        resolved_scope_segments=(),
        prepared_outputs=websocket_api._PreparedTaskFileOutputs((), (), ()),
    )
    assert not finalized.late_result

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert raw_error not in repr(events)
    assert interaction_secret not in repr(events)
    assert any(
        event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == CLIENT_SAFE_TASK_FAILURE
        for event in events
    )
    _assert_safe_failure_persisted(task_id, raw_error)


@pytest.mark.asyncio
async def test_failed_websocket_result_prefers_diagnostic_error_over_display_text(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_error = "provider token=secret"
    display_text = "I could not complete that request."
    task_id, user_id = _running_task(
        username="failed-websocket-diagnostic",
        title="Failed WebSocket diagnostic",
    )

    finalized = websocket_api._finalize_task_execution_result_isolated(
        task_id=task_id,
        task_user_id=user_id,
        pre_run_status=TaskStatus.RUNNING,
        result={
            "success": False,
            "status": "error",
            "output": display_text,
            "error": raw_error,
        },
        expected_run_id=None,
        task_lease=None,
        resolved_scope_segments=(),
        prepared_outputs=websocket_api._PreparedTaskFileOutputs((), (), ()),
    )

    assert not finalized.late_result
    _assert_safe_failure_persisted(task_id, raw_error)


def test_failed_resumed_websocket_result_prefers_diagnostic_error_over_display_text(
    _test_db,
) -> None:
    raw_error = "resume provider token=secret"
    display_text = "I could not complete that resumed request."
    task_id, user_id = _running_task(
        username="failed-resumed-websocket-diagnostic",
        title="Failed resumed WebSocket diagnostic",
        runner_id="resume-runner-1730",
        run_id="resume-run-1730",
    )

    finalized = websocket_api._finalize_resumed_task(
        task_id,
        status="error",
        success=False,
        output=display_text,
        task_owner_user_id=user_id,
        result={
            "success": False,
            "status": "error",
            "output": display_text,
            "error": raw_error,
        },
        task_lease=TaskLease(
            task_id=task_id,
            runner_id="resume-runner-1730",
            run_id="resume-run-1730",
        ),
        prepared_outputs=websocket_api._PreparedTaskFileOutputs((), (), ()),
    )

    assert not finalized["late_result"]
    _assert_safe_failure_persisted(task_id, raw_error)


def test_terminal_failure_writer_persists_safe_provenance(_test_db) -> None:
    raw_error = "sandbox rejection exposed host=/srv/private and token=secret"
    task_id, _ = _running_task(
        username="terminal-failure-provenance",
        title="Terminal failure provenance",
    )

    websocket_api._terminal_task_error_payload(task_id, raw_error)

    _assert_safe_failure_persisted(task_id, raw_error)


def test_lease_failure_writer_persists_safe_provenance(_test_db) -> None:
    raw_error = "provider failure included database password=secret"
    task_id, _ = _running_task(
        username="lease-failure-provenance",
        title="Lease failure provenance",
        runner_id="runner-1730",
        run_id="run-1730",
    )

    committed = settle_task_lease_isolated(
        TaskLease(
            task_id=task_id,
            runner_id="runner-1730",
            run_id="run-1730",
        ),
        error_message=raw_error,
    )
    assert committed

    _assert_safe_failure_persisted(task_id, raw_error)


@pytest.mark.asyncio
async def test_failed_managed_result_replays_only_safe_history(
    _test_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_error = "channel execution failed with provider token=secret"
    interaction_secret = "managed interaction token=secret"
    task_id, user_id = _running_task(
        username="failed-managed-result",
        title="Failed managed result",
        runner_id="managed-runner-1730",
        run_id="managed-run-1730",
    )
    db = _direct_db_session()
    try:
        committed = finalize_managed_task_lease_result(
            db,
            TaskLease(
                task_id=task_id,
                runner_id="managed-runner-1730",
                run_id="managed-run-1730",
            ),
            status=TaskStatus.FAILED,
            assistant_content="I could not complete that request.",
            interactions=[{"label": interaction_secret}],
            error_message=raw_error,
            message_type="assistant_message",
        )
        assert committed
    finally:
        db.close()

    events = await _replay_history(
        monkeypatch,
        task_id=task_id,
        user_id=user_id,
    )

    assert raw_error not in repr(events)
    assert interaction_secret not in repr(events)
    assert any(
        event.get("event_type") == "agent_message"
        and event.get("data", {}).get("message") == CLIENT_SAFE_TASK_FAILURE
        for event in events
    )
    _assert_safe_failure_persisted(task_id, raw_error)

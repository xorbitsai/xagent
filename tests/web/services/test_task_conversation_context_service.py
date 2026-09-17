import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.agent.result import CONTROL_TOOL_NAMES
from xagent.core.context_ref import CONTEXT_REFS_KEY
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.chat_history_service import _MAX_HISTORICAL_IMAGE_CONTEXT_REFS
from xagent.web.services.task_conversation_context_service import (
    load_task_conversation_context_sync,
)

# Serialized size matters here: the old buggy path blind-sliced tool results
# at 240 chars, and this fixture must land comfortably past that boundary so
# the test still catches a truncation regression after future edits. The
# original 7-field version of this fixture serialized (as ``{"handle": ...}``)
# to 241 chars -- a single character of margin over the old 240-char cutoff,
# which the old buggy code would have slipped past undetected on nearly any
# edit. ``commit_sha``/``container_image`` are added purely to push this to
# ~368 chars, well clear of that boundary; don't shrink this fixture without
# re-checking it stays well above 240.
STRUCTURED_HANDLE = {
    "workspace": "4b33784773d5",
    "branch": "review-pr-1392",
    "provider": "omp",
    "profile": "omp",
    "provider_session_id": "01a00aff-558e-7000-8ebf-ec547e8018b0",
    "last_run_id": "76ff56bf27554b899e74bce25f9dc769",
    "node_id": "node-0",
    "commit_sha": "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
    "container_image": "ghcr.io/xagent/sandbox-runner:2026.08.17-py311",
}


def _create_db_session():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal()


def _create_task(db_session):
    user = User(username="tester", password_hash="hashed_password", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    task = Task(
        user_id=int(user.id),
        title="Conversation context task",
        description="Task conversation context",
        status=TaskStatus.COMPLETED,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


def _add_chat_message(
    db_session, task, *, role, content, created_at, turn_id=None, attachments=None
):
    message = TaskChatMessage(
        task_id=int(task.id),
        user_id=int(task.user_id),
        role=role,
        content=content,
        message_type=role,
        turn_id=turn_id,
        created_at=created_at,
        attachments=attachments,
    )
    db_session.add(message)
    db_session.commit()
    db_session.refresh(message)
    return message


def _add_trace_event(
    db_session,
    task,
    *,
    event_type,
    data,
    timestamp,
    event_id=None,
    step_id=None,
    turn_id=None,
):
    # ``turn_id`` mirrors how PatternRuntime.on_tool_start/on_tool_end/
    # on_tool_error stamp it into the event's JSON ``data`` (never a
    # dedicated column, unlike ``step_id``) -- see runtime.py.
    payload = dict(data)
    if turn_id is not None and "turn_id" not in payload:
        payload["turn_id"] = turn_id
    event = TraceEvent(
        task_id=int(task.id),
        build_id=None,
        event_id=event_id or f"{event_type}-{timestamp.isoformat()}",
        event_type=event_type,
        timestamp=timestamp,
        step_id=step_id,
        parent_event_id=None,
        data=payload,
    )
    db_session.add(event)
    db_session.commit()
    db_session.refresh(event)
    return event


def _ts(seconds_offset, base=None, tz=timezone.utc):
    base = base or datetime(2026, 1, 1, tzinfo=tz)
    return base + timedelta(seconds=seconds_offset)


def test_structured_handle_survives_reconstruction_byte_for_byte():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="resume",
            created_at=_ts(-1),
            turn_id="turn-1",
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            turn_id="turn-1",
            data={
                "tool_name": "resume_task",
                "tool_params": {"workspace": "4b33784773d5"},
                "tool_call_id": "call-handle-1",
                "assistant_content": "Resuming previous workspace.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1),
            turn_id="turn-1",
            data={
                "tool_name": "resume_task",
                "tool_call_id": "call-handle-1",
                "success": True,
                "result": {"handle": dict(STRUCTURED_HANDLE)},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["raw_result"]["handle"] == STRUCTURED_HANDLE

        serialized = json.dumps(messages, ensure_ascii=False, default=str)
        # Discriminating check: the old bug did a blind fixed-length
        # character slice, which would corrupt whichever field that offset
        # happened to land inside. A field near the FRONT of the object
        # (like "branch", checked by an earlier version of this test) would
        # still survive most such slices and give a false pass -- the
        # 237-char legacy boundary sat well past "branch"'s offset. A field
        # placed at the very END of the object, like "container_image"
        # here, is what a fixed-length slice would actually clip first;
        # its exact, untruncated presence is the meaningful signal.
        assert (
            '"container_image": "ghcr.io/xagent/sandbox-runner:2026.08.17-py311"'
            in serialized
        )
        assert json.loads(serialized) == messages  # round-trips byte-for-byte
    finally:
        db_session.close()


def test_tool_call_pairing_never_orphaned():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="run some tools",
            created_at=_ts(-1),
            turn_id="turn-1",
        )

        # Normal start + end.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            turn_id="turn-1",
            data={
                "tool_name": "list_files",
                "tool_params": {"path": "."},
                "tool_call_id": "call-1",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1),
            turn_id="turn-1",
            data={
                "tool_name": "list_files",
                "tool_call_id": "call-1",
                "success": True,
                "result": {"files": ["a.txt"]},
            },
        )

        # End with no start.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(2),
            turn_id="turn-1",
            data={
                "tool_name": "read_file",
                "tool_call_id": "call-orphan-end",
                "success": True,
                "result": {"content": "hello"},
            },
        )

        # Start with no end -> should yield nothing.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(3),
            turn_id="turn-1",
            data={
                "tool_name": "write_file",
                "tool_params": {"path": "b.txt"},
                "tool_call_id": "call-orphan-start",
            },
        )

        # Missing tool_call_id entirely.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(4),
            turn_id="turn-1",
            data={
                "tool_name": "web_search",
                "tool_params": {"query": "xagent"},
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(5),
            turn_id="turn-1",
            data={
                "tool_name": "web_search",
                "success": True,
                "result": {"top_result": "https://example.com"},
            },
        )

        # tool_execution_failed.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(6),
            turn_id="turn-1",
            data={
                "tool_name": "run_command",
                "tool_params": {"cmd": "false"},
                "tool_call_id": "call-fail",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_failed",
            timestamp=_ts(7),
            turn_id="turn-1",
            data={
                "tool_name": "run_command",
                "tool_call_id": "call-fail",
                "error_type": "agent_tool_error",
                "error": "boom",
                "error_message": "boom",
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        tool_calls_by_id = {}
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or not message.get("tool_calls"):
                continue
            for call in message["tool_calls"]:
                tool_calls_by_id[call["id"]] = index

        for index, message in enumerate(messages):
            if message["role"] != "tool":
                continue
            call_id = message["tool_call_id"]
            assert call_id in tool_calls_by_id
            assert tool_calls_by_id[call_id] == index - 1

        tool_names_emitted = [m["tool_name"] for m in messages if m["role"] == "tool"]
        assert "write_file" not in tool_names_emitted  # start-with-no-end -> nothing
        assert "list_files" in tool_names_emitted
        assert "read_file" in tool_names_emitted
        assert "run_command" in tool_names_emitted

        run_command_tool = next(
            m
            for m in messages
            if m["role"] == "tool" and m["tool_name"] == "run_command"
        )
        assert run_command_tool["raw_result"]["error"] == "boom"
    finally:
        db_session.close()


def test_multi_turn_history_is_chronological():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for turn in range(3):
            offset_minutes = turn * 10
            turn_id = f"turn-{turn}"
            _add_chat_message(
                db_session,
                task,
                role="user",
                content=f"user turn {turn}",
                created_at=base + timedelta(minutes=offset_minutes),
                turn_id=turn_id,
            )
            for tool_index in range(2):
                call_id = f"turn{turn}-call{tool_index}"
                ts_start = base + timedelta(
                    minutes=offset_minutes, seconds=1 + tool_index * 2
                )
                ts_end = base + timedelta(
                    minutes=offset_minutes, seconds=2 + tool_index * 2
                )
                _add_trace_event(
                    db_session,
                    task,
                    event_type="tool_execution_start",
                    timestamp=ts_start,
                    turn_id=turn_id,
                    data={
                        "tool_name": "list_files",
                        "tool_params": {"turn": turn, "tool_index": tool_index},
                        "tool_call_id": call_id,
                    },
                )
                _add_trace_event(
                    db_session,
                    task,
                    event_type="tool_execution_end",
                    timestamp=ts_end,
                    turn_id=turn_id,
                    data={
                        "tool_name": "list_files",
                        "tool_call_id": call_id,
                        "success": True,
                        "result": {"turn": turn, "tool_index": tool_index},
                    },
                )
            _add_chat_message(
                db_session,
                task,
                role="assistant",
                content=f"assistant answer {turn}",
                created_at=base + timedelta(minutes=offset_minutes, seconds=9),
            )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]

        assert roles.count("user") == 3
        assert (
            roles.count("assistant") == 3 + 3 * 2
        )  # turn answers + tool-call assistants
        assert roles.count("tool") == 6

        for turn in range(3):
            user_index = messages.index(
                {"role": "user", "content": f"user turn {turn}"}
            )
            answer_index = next(
                index
                for index, m in enumerate(messages)
                if m["role"] == "assistant"
                and m.get("content") == f"assistant answer {turn}"
            )
            assert user_index < answer_index
            # every tool exchange for this turn sits strictly between the two
            for index in range(user_index + 1, answer_index):
                message = messages[index]
                if message["role"] == "tool":
                    assert message["raw_result"]["turn"] == turn
    finally:
        db_session.close()


def test_as_aware_utc_normalizes_naive_and_preserves_aware():
    """Direct unit test of ``_as_aware_utc`` itself, independent of any DB
    round trip. A naive datetime gets stamped UTC; an already-aware datetime
    (in a non-UTC zone) passes through unchanged rather than being re-based.
    """
    from xagent.web.services.task_conversation_context_service import _as_aware_utc

    naive = datetime(2026, 1, 1, 12, 0, 0)
    normalized = _as_aware_utc(naive)
    assert normalized.tzinfo is timezone.utc
    assert normalized == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    aware_non_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    normalized_aware = _as_aware_utc(aware_non_utc)
    assert normalized_aware.tzinfo == timezone(timedelta(hours=5))
    assert normalized_aware == aware_non_utc


def test_merge_chronologically_places_group_immediately_after_its_turn_user_row():
    """Direct unit test of the merge path (R4) with hand-built rows/exchanges,
    bypassing the DB round trip entirely.

    Placement is a ``turn_id`` join, not a timestamp comparison (see
    ``_merge_chronologically``'s docstring), so this no longer needs to
    prove anything about naive-vs-aware datetimes -- ``_merge_chronologically``
    doesn't compare timestamps at all any more, and ``_TranscriptRow`` no
    longer even carries a clock value. What it must still prove: a group is
    spliced in immediately after the transcript row carrying its turn_id,
    and before whatever comes next, regardless of the exchange's own
    ``sort_key``.
    """
    from xagent.web.services.task_conversation_context_service import (
        _merge_chronologically,
        _ToolExchange,
        _TranscriptRow,
    )

    user_row = _TranscriptRow(
        row_id=1,
        role="user",
        content="hi",
        turn_id="turn-1",
    )
    reply_row = _TranscriptRow(
        row_id=2,
        role="assistant",
        content="done",
    )
    exchange = _ToolExchange(
        call_id="call-1",
        tool_name="list_files",
        tool_params={},
        result={"ok": True},
        assistant_content="",
        sort_key=(datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc), 10),
        turn_id="turn-1",
    )

    entries = _merge_chronologically([user_row, reply_row], [exchange])
    kinds_in_order = [entry.kind for entry in entries]
    assert kinds_in_order == ["transcript", "group", "transcript"]
    assert entries[1].group == [exchange]
    assert entries[0].transcript is user_row
    assert entries[2].transcript is reply_row


def test_control_tools_are_excluded():
    """Every name in ``CONTROL_TOOL_NAMES`` -- not just a couple of them --
    must be excluded from reconstructed tool exchanges. This is checked
    against the real ``CONTROL_TOOL_NAMES`` set rather than a hardcoded
    tuple so a future addition to that set is automatically covered here."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="go",
            created_at=_ts(-1),
            turn_id="turn-1",
        )

        for index, tool_name in enumerate(sorted(CONTROL_TOOL_NAMES)):
            call_id = f"call-control-{index}"
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_start",
                timestamp=_ts(index * 10),
                turn_id="turn-1",
                data={
                    "tool_name": tool_name,
                    "tool_params": {},
                    "tool_call_id": call_id,
                },
            )
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_end",
                timestamp=_ts(index * 10 + 1),
                turn_id="turn-1",
                data={
                    "tool_name": tool_name,
                    "tool_call_id": call_id,
                    "success": True,
                    "result": {"ok": True},
                },
            )

        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(1000),
            turn_id="turn-1",
            data={
                "tool_name": "list_files",
                "tool_params": {},
                "tool_call_id": "call-real",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1001),
            turn_id="turn-1",
            data={
                "tool_name": "list_files",
                "tool_call_id": "call-real",
                "success": True,
                "result": {"files": []},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        tool_names = [m["tool_name"] for m in messages if m["role"] == "tool"]
        for control_tool_name in CONTROL_TOOL_NAMES:
            assert control_tool_name not in tool_names
        assert tool_names == ["list_files"]
    finally:
        db_session.close()


def test_concurrent_dag_steps_with_colliding_tool_call_ids_are_not_mispaired():
    """Two concurrent DAG branches whose tool-call ids collide (both fall
    back to ``tool_call_0`` because the provider omitted a real id) must NOT
    have their starts/ends cross-paired.

    This reproduces the real corruption: DAG runs steps concurrently
    (``asyncio.create_task`` in ``dag.py``), and ``_normalize_tool_calls``
    (react.py) synthesizes ``tool_call_{index}`` where ``index`` is only
    unique within a single LLM response -- so two concurrent steps can each
    emit id "tool_call_0". Interleaving start A, start B, end A, end B with a
    bare-``tool_call_id`` pending map would let B's start overwrite A's, and
    A's end would then pop B's start, attaching branch B's tool_name /
    assistant_content to branch A's result (and vice versa).

    Each step carries its own ``step_id`` (DAG's ``_with_step`` stamps a
    unique step id per concurrent branch, persisted on
    ``TraceEvent.step_id``), which is exactly the discriminator this test
    checks is actually used to disambiguate.
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="go",
            created_at=_ts(-1),
            turn_id="turn-1",
        )

        # Branch A start (step "step-a"), then branch B start (step "step-b"),
        # both using the colliding id "tool_call_0" -- interleaved as
        # start A, start B, end A, end B, matching real concurrent arrival.
        # Both branches belong to the same DAG turn, so they share one
        # turn_id even though their step_id (the pairing discriminator)
        # differs.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            step_id="step-a",
            turn_id="turn-1",
            data={
                "tool_name": "read_file",
                "tool_params": {"path": "a.txt"},
                "tool_call_id": "tool_call_0",
                "assistant_content": "Branch A: reading a.txt.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(1),
            step_id="step-b",
            turn_id="turn-1",
            data={
                "tool_name": "list_files",
                "tool_params": {"path": "b/"},
                "tool_call_id": "tool_call_0",
                "assistant_content": "Branch B: listing b/.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(2),
            step_id="step-a",
            turn_id="turn-1",
            data={
                "tool_name": "read_file",
                "tool_call_id": "tool_call_0",
                "success": True,
                "result": {"content": "contents of a.txt"},
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(3),
            step_id="step-b",
            turn_id="turn-1",
            data={
                "tool_name": "list_files",
                "tool_call_id": "tool_call_0",
                "success": True,
                "result": {"files": ["b1.txt", "b2.txt"]},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        assistant_messages = [
            m for m in messages if m["role"] == "assistant" and m.get("tool_calls")
        ]
        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert len(assistant_messages) == 2
        assert len(tool_messages) == 2

        by_result = {}
        for assistant_message, tool_message in zip(assistant_messages, tool_messages):
            call = assistant_message["tool_calls"][0]
            by_result[tool_message["raw_result"].get("content") or "listing"] = (
                assistant_message["content"],
                call["function"]["name"],
                tool_message["tool_name"],
            )

        content_a, tool_name_a, result_tool_name_a = by_result["contents of a.txt"]
        assert content_a == "Branch A: reading a.txt."
        assert tool_name_a == "read_file"
        assert result_tool_name_a == "read_file"

        content_b, tool_name_b, result_tool_name_b = by_result["listing"]
        assert content_b == "Branch B: listing b/."
        assert tool_name_b == "list_files"
        assert result_tool_name_b == "list_files"
    finally:
        db_session.close()


def test_react_shaped_events_without_step_discriminator_still_pair_correctly():
    """ReAct never sets a colliding id within one turn, and older rows
    persisted before ``TraceEvent.step_id`` was populated carry
    ``step_id=None`` for every event. Both must still pair by bare
    ``tool_call_id`` exactly as before this fix -- the step-id
    discriminator must be a no-op here, not a behavior change.
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="go",
            created_at=_ts(-1),
            turn_id="turn-1",
        )

        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            step_id=None,
            turn_id="turn-1",
            data={
                "tool_name": "search_web",
                "tool_params": {"query": "xagent"},
                "tool_call_id": "call-1",
                "assistant_content": "Searching the web.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1),
            step_id=None,
            turn_id="turn-1",
            data={
                "tool_name": "search_web",
                "tool_call_id": "call-1",
                "success": True,
                "result": {"top_result": "https://example.com"},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        assistant_messages = [
            m for m in messages if m["role"] == "assistant" and m.get("tool_calls")
        ]
        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert len(assistant_messages) == 1
        assert len(tool_messages) == 1
        assert assistant_messages[0]["content"] == "Searching the web."
        assert assistant_messages[0]["tool_calls"][0]["function"]["name"] == (
            "search_web"
        )
        assert tool_messages[0]["raw_result"] == {"top_result": "https://example.com"}
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Part C -- coverage for previously-untested paths
# ---------------------------------------------------------------------------


def test_before_message_id_excludes_current_turn_user_message():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        prior = _add_chat_message(
            db_session,
            task,
            role="user",
            content="prior turn message",
            created_at=base,
        )
        current = _add_chat_message(
            db_session,
            task,
            role="user",
            content="current turn message",
            created_at=base + timedelta(seconds=10),
        )
        assert prior.id < current.id

        messages = load_task_conversation_context_sync(
            db_session, int(task.id), before_message_id=int(current.id)
        )
        contents = [m.get("content") for m in messages]
        assert "prior turn message" in contents
        assert "current turn message" not in contents
    finally:
        db_session.close()


def test_resolve_tool_result_interrupted_branch_without_result():
    """A ``tool_execution_end`` marked ``interrupted`` with no ``result`` key
    must produce the degraded interrupted dict, not fall through to the
    "missing result" branch or a bare ``None``."""
    from xagent.web.services.task_conversation_context_service import (
        _resolve_tool_result,
    )

    resolved = _resolve_tool_result(
        "tool_execution_end",
        {
            "tool_name": "run_command",
            "interrupted": True,
            "interrupt_reason": "user cancelled the run",
        },
    )
    assert resolved == {
        "success": False,
        "interrupted": True,
        "error": "user cancelled the run",
    }


def test_final_pairing_sweep_drops_orphaned_tool_message():
    """Direct unit test of ``_final_pairing_sweep``: no fixture in this suite
    naturally produces an orphaned ``tool`` message (construction is
    designed to never emit one), so this exercises the safety-net sweep
    itself with a hand-built orphan."""
    from xagent.web.services.task_conversation_context_service import (
        _final_pairing_sweep,
    )

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "tool",
            "tool_call_id": "orphan-1",
            "content": "",
            "tool_name": "orphan_tool",
            "raw_result": {},
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "real_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "",
            "tool_name": "real_tool",
            "raw_result": {},
        },
    ]

    sanitized = _final_pairing_sweep(messages)
    assert sanitized == [messages[0], messages[2], messages[3]]
    tool_call_ids = [m["tool_call_id"] for m in sanitized if m["role"] == "tool"]
    assert tool_call_ids == ["call-1"]


def test_final_pairing_sweep_drops_orphan_when_both_ids_are_none():
    """Review regression (PR #1601): the old comparison was
    ``str(call.get("id")) == str(message.get("tool_call_id"))``, so a
    ``tool`` message with ``tool_call_id: None`` preceded by an assistant
    whose declared call also has ``id: None`` rendered as
    ``"None" == "None"`` and was kept -- an orphan slipping through the
    exact defense meant to catch it. This must fail against that old
    ``str(...) == str(...)`` form and pass with ``_ids_match``."""
    from xagent.web.services.task_conversation_context_service import (
        _final_pairing_sweep,
    )

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "real_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": None,
            "content": "",
            "tool_name": "real_tool",
            "raw_result": {},
        },
    ]

    sanitized = _final_pairing_sweep(messages)
    assert sanitized == [messages[0], messages[1]]
    assert all(m.get("role") != "tool" for m in sanitized)


def test_final_pairing_sweep_keeps_tool_message_with_matching_real_id():
    """Normal path, unchanged: a tool message with a real id preceded by
    the assistant that declared that same id is kept."""
    from xagent.web.services.task_conversation_context_service import (
        _final_pairing_sweep,
    )

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-42",
                    "type": "function",
                    "function": {"name": "real_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-42",
            "content": "",
            "tool_name": "real_tool",
            "raw_result": {},
        },
    ]

    sanitized = _final_pairing_sweep(messages)
    assert sanitized == messages


def test_final_pairing_sweep_drops_tool_message_with_mismatched_real_id():
    """A tool message with a real id whose preceding assistant declares a
    different id is still dropped as an orphan."""
    from xagent.web.services.task_conversation_context_service import (
        _final_pairing_sweep,
    )

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "real_tool", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "",
            "tool_name": "real_tool",
            "raw_result": {},
        },
    ]

    sanitized = _final_pairing_sweep(messages)
    assert sanitized == [messages[0]]


def test_ids_match_ignores_malformed_tool_call_entries():
    """Malformed ``tool_calls`` entries -- a non-dict item, or a dict with
    no ``id`` key -- must not raise and must not produce a false match."""
    from xagent.web.services.task_conversation_context_service import (
        _ids_match,
    )

    assert _ids_match("call-1", ["not-a-dict", {"type": "function"}]) is False
    assert _ids_match(None, ["not-a-dict", {"type": "function"}]) is False
    assert (
        _ids_match(
            "call-1",
            ["not-a-dict", {"type": "function"}, {"id": "call-1"}],
        )
        is True
    )


# ---------------------------------------------------------------------------
# Part D -- regression tests for the fixes just made
# ---------------------------------------------------------------------------


def test_tool_execution_end_missing_result_yields_degraded_dict_not_none():
    """Fix 6 regression: a ``tool_execution_end`` with no ``result`` key and
    not marked ``interrupted`` must produce a degraded placeholder dict,
    never a bare ``None`` flowing into ``raw_result``."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="go",
            created_at=_ts(-1),
            turn_id="turn-1",
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(0),
            turn_id="turn-1",
            data={
                "tool_name": "flaky_tool",
                "tool_params": {},
                "tool_call_id": "call-degraded",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(1),
            turn_id="turn-1",
            data={
                "tool_name": "flaky_tool",
                "tool_call_id": "call-degraded",
                "success": True,
                # No "result" key, and not interrupted -- the degraded case.
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        tool_message = next(m for m in messages if m["role"] == "tool")
        assert tool_message["raw_result"] is not None
        assert tool_message["raw_result"] == {
            "success": False,
            "status": "unknown",
            "error": "tool result missing from persisted trace event",
        }
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Uploaded-image context refs (regression: our resume path replaced
# ``load_task_transcript`` with ``load_task_conversation_context_sync``, but
# this module never read ``attachments`` -- resumed conversations silently
# dropped uploaded images even though ``load_task_transcript`` still carries
# them. These tests pin the fix.)
# ---------------------------------------------------------------------------


def test_uploaded_images_on_transcript_rows_survive_reconstruction():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = _ts(0)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="What is shown?",
            created_at=base,
            attachments=[
                {
                    "file_id": "image-id",
                    "name": "diagram.png",
                    "size": 321,
                    "type": "image/png",
                },
                {
                    "file_id": "pdf-id",
                    "name": "notes.pdf",
                    "size": 654,
                    "type": "application/pdf",
                },
            ],
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="It's a diagram.",
            created_at=base + timedelta(seconds=1),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is shown?"
        references = messages[0][CONTEXT_REFS_KEY]
        # Only the image attachment (not the pdf) becomes a context ref,
        # matching ``build_image_context_references``'s image-only filter.
        assert len(references) == 1
        assert references[0]["file_ref"]["file_id"] == "image-id"
        assert references[0]["metadata"] == {"source": "user_upload"}

        # No image attachment on the assistant row -- no key at all, not an
        # empty list, matching ``load_task_transcript``'s shape exactly.
        assert CONTEXT_REFS_KEY not in messages[1]
    finally:
        db_session.close()


def test_historical_image_budget_keeps_only_the_newest_n_across_messages():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = _ts(0)
        total_images = _MAX_HISTORICAL_IMAGE_CONTEXT_REFS + 2
        for index in range(total_images):
            _add_chat_message(
                db_session,
                task,
                role="user",
                content=f"Image {index}",
                created_at=base + timedelta(seconds=index),
                attachments=[
                    {
                        "file_id": f"image-{index}",
                        "name": f"image-{index}.png",
                        "type": "image/png",
                    }
                ],
            )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        assert len(messages) == total_images
        # The oldest two messages, over budget, carry no context refs at all.
        assert CONTEXT_REFS_KEY not in messages[0]
        assert CONTEXT_REFS_KEY not in messages[1]
        retained_ids = [
            message[CONTEXT_REFS_KEY][0]["file_ref"]["file_id"]
            for message in messages
            if CONTEXT_REFS_KEY in message
        ]
        assert retained_ids == [f"image-{index}" for index in range(2, total_images)]
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Part D -- turn_id-join placement and image-only-turn retention (#1601)
#
# fa650272 tried to solve turn placement by grouping tool exchanges on
# ``TraceEvent.step_id`` and relocating groups by timestamp adjacency. A
# reviewer found two structural counterexamples (reproduced below as
# ``test_dag_replan_reusing_step_id_keeps_turns_separate`` and
# ``test_failed_turn_without_assistant_row_stays_before_next_user_message``),
# so that approach was replaced with the ``turn_id`` join implemented in
# ``_merge_chronologically``/``_group_exchanges_by_turn``: every tool trace
# event now carries the same durable ``turn_id`` as its triggering user
# message (``PatternRuntime.on_tool_start``/``on_tool_end``/``on_tool_error``
# in runtime.py), so placement is a lookup, not an inference from clocks or
# a reused step id.
# ---------------------------------------------------------------------------


def test_second_turn_clock_skew_placed_correctly_by_turn_id_join():
    """Reproduces the original clock-skew scenario -- turn 2's
    ``tool_execution_start``/``tool_execution_end`` are timestamped 1s
    before turn 2's own ``TaskChatMessage`` (DB clock vs app clock skew) --
    but placement is now a ``turn_id`` join, so the skew is irrelevant: the
    exchange still lands right after turn 2's question because that's what
    its ``turn_id`` says, not because of any timestamp comparison.
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # Turn 1: well-timestamped, no skew.
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="turn 1 question",
            created_at=base,
            turn_id="turn1",
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=1),
            turn_id="turn1",
            data={
                "tool_name": "list_files",
                "tool_params": {"turn": 1},
                "tool_call_id": "turn1-call",
                "assistant_content": "Checking turn 1.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=2),
            turn_id="turn1",
            data={
                "tool_name": "list_files",
                "tool_call_id": "turn1-call",
                "success": True,
                "result": {"turn": 1},
            },
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="turn 1 answer",
            created_at=base + timedelta(seconds=3),
        )

        # Turn 2: the TaskChatMessage row (DB clock).
        turn2_user_created_at = base + timedelta(minutes=5)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="turn 2 question",
            created_at=turn2_user_created_at,
            turn_id="turn2",
        )

        # Turn 2's tool exchange (app clock), timestamped BEFORE its own
        # user message due to clock skew between the two clocks.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=turn2_user_created_at - timedelta(seconds=1),
            turn_id="turn2",
            data={
                "tool_name": "list_files",
                "tool_params": {"turn": 2},
                "tool_call_id": "turn2-call",
                "assistant_content": "Checking turn 2.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=turn2_user_created_at - timedelta(milliseconds=500),
            turn_id="turn2",
            data={
                "tool_name": "list_files",
                "tool_call_id": "turn2-call",
                "success": True,
                "result": {"turn": 2},
            },
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="turn 2 answer",
            created_at=turn2_user_created_at + timedelta(seconds=3),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]

        assert roles == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert messages[0]["content"] == "turn 1 question"
        assert messages[1]["tool_calls"][0]["id"] == "turn1-call"
        assert messages[2]["tool_call_id"] == "turn1-call"
        assert messages[3]["content"] == "turn 1 answer"
        assert messages[4]["content"] == "turn 2 question"
        # Turn 2's exchange sits AFTER turn 2's own question, not before it.
        assert messages[5]["tool_calls"][0]["id"] == "turn2-call"
        assert messages[6]["tool_call_id"] == "turn2-call"
        assert messages[7]["content"] == "turn 2 answer"
    finally:
        db_session.close()


def test_dag_replan_reusing_step_id_keeps_turns_separate():
    """Reviewer counterexample #1 (fa650272): ``ExecutionPlan.validate()``
    only enforces ``step_id`` uniqueness within one plan, so a later re-plan
    can emit ``step_1`` again -- two DAG turns can legitimately share one
    ``step_id``. Grouping by ``step_id`` (the fa650272 approach) merged such
    turns into one group; joining on ``turn_id`` instead keeps them separate
    regardless of ``step_id`` collisions, because placement never looks at
    ``step_id`` at all.
    """
    from xagent.web.services.task_conversation_context_service import (
        _group_exchanges_by_turn,
        _ToolExchange,
    )

    # Direct unit check: two turns whose exchanges carry the identical
    # step_id="step_1" (a DAG re-plan reusing the id) must still land in two
    # distinct groups, keyed by turn_id.
    turn1_call = _ToolExchange(
        call_id="t1-a",
        tool_name="list_files",
        tool_params={},
        result={"ok": True},
        assistant_content="",
        sort_key=(_ts(0), 1),
        turn_id="turn1",
    )
    turn2_call = _ToolExchange(
        call_id="t2-a",
        tool_name="list_files",
        tool_params={},
        result={"ok": True},
        assistant_content="",
        sort_key=(_ts(2), 3),
        turn_id="turn2",
    )
    groups = _group_exchanges_by_turn([turn1_call, turn2_call])
    assert groups == {"turn1": [turn1_call], "turn2": [turn2_call]}

    # End-to-end: both turns' trace events are stamped step_id="step_1" (the
    # DAG re-plan collision), but distinct turn_id -- the reconstruction
    # must still keep the turns separate and correctly ordered: user1,
    # tools1, answer1, user2, tools2, answer2 -- never user1, tools1,
    # tools2, answer1, user2, answer2 (the fa650272 regression).
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="q1",
            created_at=base,
            turn_id="turn1",
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=1),
            step_id="step_1",
            turn_id="turn1",
            data={
                "tool_name": "list_files",
                "tool_params": {"call": "t1-a"},
                "tool_call_id": "t1-a",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=1, milliseconds=500),
            step_id="step_1",
            turn_id="turn1",
            data={
                "tool_name": "list_files",
                "tool_call_id": "t1-a",
                "success": True,
                "result": {"call": "t1-a"},
            },
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="a1",
            created_at=base + timedelta(seconds=5),
        )
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="q2",
            created_at=base + timedelta(minutes=1),
            turn_id="turn2",
        )
        # The re-plan: same step_id="step_1" as turn 1, different turn_id.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(minutes=1, seconds=1),
            step_id="step_1",
            turn_id="turn2",
            data={
                "tool_name": "list_files",
                "tool_params": {"call": "t2-a"},
                "tool_call_id": "t2-a",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(minutes=1, seconds=2),
            step_id="step_1",
            turn_id="turn2",
            data={
                "tool_name": "list_files",
                "tool_call_id": "t2-a",
                "success": True,
                "result": {"call": "t2-a"},
            },
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="a2",
            created_at=base + timedelta(minutes=1, seconds=3),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        shapes = [
            (m["role"], m.get("content"), m.get("tool_call_id")) for m in messages
        ]
        assert shapes == [
            ("user", "q1", None),
            ("assistant", "", None),
            ("tool", "", "t1-a"),
            ("assistant", "a1", None),
            ("user", "q2", None),
            ("assistant", "", None),
            ("tool", "", "t2-a"),
            ("assistant", "a2", None),
        ]
    finally:
        db_session.close()


def test_failed_turn_without_assistant_row_stays_before_next_user_message():
    """Reviewer counterexample #2 (fa650272): a failed/interrupted turn
    produces no assistant transcript row, so
    ``_relocate_groups_before_their_user_message`` (the fa650272 relocation
    pass) flushed its buffered exchange group *after* the next user message
    regardless of timestamps -- turning ``user1, failed_tools1, user2`` into
    ``user1, user2, failed_tools1``. Joining on ``turn_id`` places the
    exchange immediately after its own turn's user row, before the next
    transcript row, with no relocation pass to misfire.
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        _add_chat_message(
            db_session,
            task,
            role="user",
            content="q1",
            created_at=base,
            turn_id="turn1",
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=1),
            turn_id="turn1",
            data={
                "tool_name": "run_command",
                "tool_params": {"cmd": "false"},
                "tool_call_id": "call-failed",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_failed",
            timestamp=base + timedelta(seconds=2),
            turn_id="turn1",
            data={
                "tool_name": "run_command",
                "tool_call_id": "call-failed",
                "error_type": "agent_tool_error",
                "error": "boom",
                "error_message": "boom",
            },
        )
        # Turn 1 never gets an assistant answer row -- the run was
        # interrupted/failed before producing a final_answer.
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="q2",
            created_at=base + timedelta(minutes=1),
            turn_id="turn2",
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="a2",
            created_at=base + timedelta(minutes=1, seconds=1),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]

        # user1, failed_tools1, user2, answer2 -- never user1, user2, ...tools1.
        assert roles == ["user", "assistant", "tool", "user", "assistant"]
        assert messages[0]["content"] == "q1"
        assert messages[2]["tool_call_id"] == "call-failed"
        assert messages[2]["raw_result"]["error"] == "boom"
        assert messages[3]["content"] == "q2"
        assert messages[4]["content"] == "a2"
    finally:
        db_session.close()


def test_legacy_exchange_without_turn_id_is_omitted_transcript_intact():
    """A tool exchange whose trace events predate ``turn_id`` support (both
    events carry no ``turn_id`` in their JSON ``data``) is omitted from the
    reconstruction entirely -- not placed by guesswork -- while the
    surrounding transcript rows still reconstruct intact. This is the
    deliberate legacy behavior: conversations recorded before this shipped
    resume with transcript text only, exactly what they got before this
    module existed.
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        _add_chat_message(db_session, task, role="user", content="q1", created_at=base)
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=1),
            data={
                "tool_name": "list_files",
                "tool_params": {},
                "tool_call_id": "legacy-call",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=2),
            data={
                "tool_name": "list_files",
                "tool_call_id": "legacy-call",
                "success": True,
                "result": {"files": []},
            },
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="a1",
            created_at=base + timedelta(seconds=3),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]

        # No trace of the legacy tool exchange -- just the transcript text.
        assert roles == ["user", "assistant"]
        assert messages[0]["content"] == "q1"
        assert messages[1]["content"] == "a1"
    finally:
        db_session.close()


def test_mixed_turn_id_and_legacy_exchanges_in_same_task():
    """A task spanning the turn_id deploy: turn 1's tool exchange predates
    turn_id support (omitted, transcript-only), turn 2's postdates it
    (placed exactly, right after turn 2's question)."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # Turn 1: legacy -- no turn_id anywhere.
        _add_chat_message(db_session, task, role="user", content="q1", created_at=base)
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=1),
            data={
                "tool_name": "list_files",
                "tool_params": {},
                "tool_call_id": "legacy-call",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=2),
            data={
                "tool_name": "list_files",
                "tool_call_id": "legacy-call",
                "success": True,
                "result": {"files": []},
            },
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="a1",
            created_at=base + timedelta(seconds=3),
        )

        # Turn 2: post-deploy -- turn_id everywhere.
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="q2",
            created_at=base + timedelta(minutes=1),
            turn_id="turn2",
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(minutes=1, seconds=1),
            turn_id="turn2",
            data={
                "tool_name": "list_files",
                "tool_params": {},
                "tool_call_id": "post-deploy-call",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(minutes=1, seconds=2),
            turn_id="turn2",
            data={
                "tool_name": "list_files",
                "tool_call_id": "post-deploy-call",
                "success": True,
                "result": {"files": ["a.txt"]},
            },
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="a2",
            created_at=base + timedelta(minutes=1, seconds=3),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]

        # Turn 1: transcript text only, no tool exchange. Turn 2: full
        # reconstruction, placed right after its own question.
        assert roles == ["user", "assistant", "user", "assistant", "tool", "assistant"]
        assert messages[0]["content"] == "q1"
        assert messages[1]["content"] == "a1"
        assert messages[2]["content"] == "q2"
        assert messages[3]["tool_calls"][0]["id"] == "post-deploy-call"
        assert messages[4]["tool_call_id"] == "post-deploy-call"
        assert messages[5]["content"] == "a2"
    finally:
        db_session.close()


def test_image_only_user_turn_survives_with_blank_content():
    """A ``TaskChatMessage`` with ``content=""`` and an image attachment is
    a supported product path (``persist_user_message_no_commit``,
    chat_history_service.py, documents this explicitly) and must survive
    reconstruction with its context refs -- not be dropped as if it were
    truly empty."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = _ts(0)

        _add_chat_message(
            db_session,
            task,
            role="user",
            content="",
            created_at=base,
            attachments=[
                {
                    "file_id": "image-only",
                    "name": "screenshot.png",
                    "type": "image/png",
                }
            ],
        )
        # A truly empty row -- no content, no attachments -- must still be
        # dropped.
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="",
            created_at=base + timedelta(seconds=1),
            attachments=None,
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="I see the screenshot.",
            created_at=base + timedelta(seconds=2),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == ""
        assert len(messages[0][CONTEXT_REFS_KEY]) == 1
        assert messages[0][CONTEXT_REFS_KEY][0]["file_ref"]["file_id"] == ("image-only")
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "I see the screenshot."
    finally:
        db_session.close()


def test_image_only_turn_participates_correctly_in_chronological_ordering():
    """The image-only row is not just retained -- it must land in its
    correct chronological slot relative to a tool exchange interleaved
    around it."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = _ts(0)

        _add_chat_message(
            db_session,
            task,
            role="user",
            content="",
            created_at=base,
            turn_id="turn1",
            attachments=[
                {
                    "file_id": "image-mid",
                    "name": "diagram.png",
                    "type": "image/png",
                }
            ],
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=1),
            step_id="turn1",
            turn_id="turn1",
            data={
                "tool_name": "analyze_image",
                "tool_params": {},
                "tool_call_id": "call-analyze",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=2),
            step_id="turn1",
            turn_id="turn1",
            data={
                "tool_name": "analyze_image",
                "tool_call_id": "call-analyze",
                "success": True,
                "result": {"label": "diagram"},
            },
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="It's a diagram.",
            created_at=base + timedelta(seconds=3),
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]

        assert roles == ["user", "assistant", "tool", "assistant"]
        assert messages[0]["content"] == ""
        assert messages[0][CONTEXT_REFS_KEY][0]["file_ref"]["file_id"] == ("image-mid")
        assert messages[1]["tool_calls"][0]["id"] == "call-analyze"
        assert messages[2]["tool_call_id"] == "call-analyze"
        assert messages[3]["content"] == "It's a diagram."
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Part E -- turn_id in the pairing key (item 2), the F1 silent-drop decision
# (item 4), and the summary log line (item 5).
# ---------------------------------------------------------------------------


def test_pairing_key_with_turn_id_does_not_change_well_formed_single_turn_pairing():
    """Item 2: folding ``turn_id`` into the pending-start pairing key must
    not change pairing for the common, well-formed case -- a single turn
    whose start and end events both carry the same ``turn_id``."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="go",
            created_at=_ts(0),
            turn_id="turn-1",
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(1),
            turn_id="turn-1",
            data={
                "tool_name": "search",
                "tool_params": {"query": "docs"},
                "tool_call_id": "call-1",
                "assistant_content": "Searching the docs.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(2),
            turn_id="turn-1",
            data={
                "tool_name": "search",
                "tool_call_id": "call-1",
                "success": True,
                "result": {"hits": 3},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool"]
        assert messages[1]["content"] == "Searching the docs."
        assert messages[1]["tool_calls"][0]["id"] == "call-1"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {
            "query": "docs"
        }
        assert messages[2]["tool_call_id"] == "call-1"
        assert messages[2]["raw_result"] == {"hits": 3}
    finally:
        db_session.close()


def test_pairing_key_turn_id_separates_colliding_step_and_call_ids():
    """Item 2, exercising the actual mechanism the ``turn_id`` key addition
    defends against, not just confirming it's a no-op for the common case.

    The task description for item 2 is explicit that the exploit path --
    two different turns' tool calls interleaving with the same
    ``(step_id, tool_call_id)`` pair -- is unreachable today through the
    real runtime (serialized turns, atomic DB claims, etc.). But the
    pending-map keying logic itself can still be unit-tested directly by
    constructing colliding trace rows (as a DAG replan restarting step
    numbering at ``step_1`` would, paired with a reused synthesized
    ``tool_call_{index}``-style id) and checking it does not mis-pair.

    Without ``turn_id`` in the key: turn B's start (same step_id, same
    call_id, different turn_id) overwrites turn A's pending entry, so
    turn A's end pops turn B's start (wrong content) and turn B's end
    finds nothing (already consumed). With ``turn_id`` folded in, the two
    turns get distinct keys and both pair correctly.
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="turn a",
            created_at=_ts(0),
            turn_id="turn-A",
        )
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="turn b",
            created_at=_ts(10),
            turn_id="turn-B",
        )

        # Turn A's start -- step_id/call_id deliberately colliding with
        # turn B's below (simulating a DAG replan that restarts step
        # numbering at the same id).
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(1),
            step_id="step_1",
            turn_id="turn-A",
            data={
                "tool_name": "tool_a",
                "tool_params": {},
                "tool_call_id": "call_0",
                "assistant_content": "doing A",
            },
        )
        # Turn B's start -- same step_id and call_id as turn A's, but a
        # different turn_id.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=_ts(2),
            step_id="step_1",
            turn_id="turn-B",
            data={
                "tool_name": "tool_b",
                "tool_params": {},
                "tool_call_id": "call_0",
                "assistant_content": "doing B",
            },
        )
        # Turn A's end arrives next -- must pop turn A's own start, not
        # turn B's.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(3),
            step_id="step_1",
            turn_id="turn-A",
            data={
                "tool_name": "tool_a",
                "tool_call_id": "call_0",
                "success": True,
                "result": {"who": "a"},
            },
        )
        # Turn B's end arrives last -- must still find turn B's own start.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=_ts(4),
            step_id="step_1",
            turn_id="turn-B",
            data={
                "tool_name": "tool_b",
                "tool_call_id": "call_0",
                "success": True,
                "result": {"who": "b"},
            },
        )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        assistant_by_tool = {}
        result_by_tool = {}
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or not message.get("tool_calls"):
                continue
            tool_name = message["tool_calls"][0]["function"]["name"]
            assistant_by_tool[tool_name] = message["content"]
            tool_message = messages[index + 1]
            assert tool_message["role"] == "tool"
            result_by_tool[tool_name] = tool_message["raw_result"]

        assert assistant_by_tool["tool_a"] == "doing A"
        assert assistant_by_tool["tool_b"] == "doing B"
        assert result_by_tool["tool_a"] == {"who": "a"}
        assert result_by_tool["tool_b"] == {"who": "b"}
    finally:
        db_session.close()


def test_image_budget_eviction_drops_tool_exchanges_deliberately_not_silently():
    """Item 4 (F1): an image-only user turn evicted by the historical-image
    ref budget must not silently take its tool exchanges down with it in a
    way that's indistinguishable from a bug -- the row drop itself matches
    upstream and is correct, but the turn's tool exchanges must be dropped
    as a deliberate, counted choice (see ``_ReconstructionStats
    .exchanges_with_unmatched_turn``), never fabricated a placement
    elsewhere in the output."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = _ts(0)

        # The earliest image-only turn: its ref will be evicted once enough
        # newer images exist to exhaust the budget.
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="",
            created_at=base,
            turn_id="turn-evicted",
            attachments=[
                {
                    "file_id": "image-evicted",
                    "name": "old-screenshot.png",
                    "type": "image/png",
                }
            ],
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=1),
            turn_id="turn-evicted",
            data={
                "tool_name": "analyze_image",
                "tool_params": {},
                "tool_call_id": "call-evicted",
                "assistant_content": "Let me look at that image.",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=2),
            turn_id="turn-evicted",
            data={
                "tool_name": "analyze_image",
                "tool_call_id": "call-evicted",
                "success": True,
                "result": {"label": "old diagram"},
            },
        )
        # Its answer -- a plain transcript row, unaffected by the image
        # budget -- must still survive.
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="It's an old diagram.",
            created_at=base + timedelta(seconds=3),
        )

        # Enough newer image uploads to push the evicted turn's single ref
        # entirely outside the budget (reverse scan keeps only the newest
        # _MAX_HISTORICAL_IMAGE_CONTEXT_REFS refs).
        for index in range(_MAX_HISTORICAL_IMAGE_CONTEXT_REFS):
            _add_chat_message(
                db_session,
                task,
                role="user",
                content=f"newer image {index}",
                created_at=base + timedelta(seconds=10 + index),
                attachments=[
                    {
                        "file_id": f"image-newer-{index}",
                        "name": f"newer-{index}.png",
                        "type": "image/png",
                    }
                ],
            )

        messages = load_task_conversation_context_sync(db_session, int(task.id))

        # The row drop matches upstream: the image-only row with no
        # surviving refs is gone.
        assert all(m.get("content") != "" or CONTEXT_REFS_KEY in m for m in messages)
        for message in messages:
            if CONTEXT_REFS_KEY in message:
                file_ids = {
                    ref["file_ref"]["file_id"] for ref in message[CONTEXT_REFS_KEY]
                }
                assert "image-evicted" not in file_ids

        # The answer to the evicted turn survives as ordinary transcript
        # text.
        assert any(
            m["role"] == "assistant" and m.get("content") == "It's an old diagram."
            for m in messages
        )

        # The deliberate part: no fabricated placement anywhere for the
        # evicted turn's tool exchange -- it must not appear at all.
        tool_call_ids = {
            call.get("id")
            for m in messages
            if m["role"] == "assistant"
            for call in (m.get("tool_calls") or [])
        }
        assert "call-evicted" not in tool_call_ids
        assert not any(
            m["role"] == "tool" and m.get("tool_call_id") == "call-evicted"
            for m in messages
        )
    finally:
        db_session.close()


def test_summary_log_line_reports_every_drop_reason_with_correct_counts(caplog):
    """Item 5: the single summary log line at the end of
    ``load_task_conversation_context_sync`` must report accurate counts for
    transcript rows, placed exchanges, and every drop reason, exercised
    together in one fixture:

    - a legacy exchange with no usable turn_id (dropped by grouping)
    - a turn whose user row is excluded by ``before_message_id`` (turn_id
      present, but no surviving row to anchor on)
    - a dangling tool_execution_start with no matching end
    - a tool_execution_end with no matching start (still rendered, just
      degraded)
    - one well-formed turn whose exchange is actually placed
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        base = _ts(0)

        # turn-a: a normal, fully paired turn that survives.
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="turn a",
            created_at=base,
            turn_id="turn-a",
        )
        _add_chat_message(
            db_session,
            task,
            role="assistant",
            content="turn a answer",
            created_at=base + timedelta(seconds=1),
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=2),
            turn_id="turn-a",
            data={
                "tool_name": "tool_a",
                "tool_params": {},
                "tool_call_id": "call-a",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=3),
            turn_id="turn-a",
            data={
                "tool_name": "tool_a",
                "tool_call_id": "call-a",
                "success": True,
                "result": {"ok": True},
            },
        )

        # An orphan end, attached to turn-a's own turn_id so it lands in a
        # surviving group (isolating the "end without start" count from
        # the "unmatched turn" count).
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=4),
            turn_id="turn-a",
            data={
                "tool_name": "orphan_end_tool",
                "tool_call_id": "call-orphan-end",
                "success": True,
                "result": {"ok": True},
            },
        )

        # turn-b: its user row will be excluded via before_message_id, but
        # its tool exchange (turn_id="turn-b") is fully paired.
        turn_b_user_row = _add_chat_message(
            db_session,
            task,
            role="user",
            content="turn b",
            created_at=base + timedelta(seconds=5),
            turn_id="turn-b",
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=6),
            turn_id="turn-b",
            data={
                "tool_name": "tool_b",
                "tool_params": {},
                "tool_call_id": "call-b",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=7),
            turn_id="turn-b",
            data={
                "tool_name": "tool_b",
                "tool_call_id": "call-b",
                "success": True,
                "result": {"ok": True},
            },
        )

        # A legacy exchange with no turn_id at all.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=8),
            data={
                "tool_name": "legacy_tool",
                "tool_params": {},
                "tool_call_id": "call-legacy",
            },
        )
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_end",
            timestamp=base + timedelta(seconds=9),
            data={
                "tool_name": "legacy_tool",
                "tool_call_id": "call-legacy",
                "success": True,
                "result": {"ok": True},
            },
        )

        # A dangling start with no matching end at all.
        _add_trace_event(
            db_session,
            task,
            event_type="tool_execution_start",
            timestamp=base + timedelta(seconds=10),
            turn_id="turn-a",
            data={
                "tool_name": "never_finishes",
                "tool_params": {},
                "tool_call_id": "call-dangling",
            },
        )

        logger_name = "xagent.web.services.task_conversation_context_service"
        with caplog.at_level(logging.INFO, logger=logger_name):
            load_task_conversation_context_sync(
                db_session, int(task.id), before_message_id=int(turn_b_user_row.id)
            )

        records = [
            record
            for record in caplog.records
            if record.name == logger_name
            and "task_conversation_context_reconstructed" in record.getMessage()
        ]
        assert len(records) == 1
        message = records[0].getMessage()

        assert "transcript_rows=2" in message
        assert "exchanges_placed=2" in message
        assert "exchanges_without_turn_id=1" in message
        assert "exchanges_with_unmatched_turn=1" in message
        assert "dangling_tool_starts=1" in message
        assert "tool_ends_without_start=1" in message
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Part F -- assistant prose is carried through verbatim (no dedup).
# ---------------------------------------------------------------------------


def test_serialized_iterations_with_identical_prose_both_keep_it():
    """Two *serialized* ReAct iterations inside one turn -- A start, A end,
    B start, B end -- whose LLM responses happen to carry byte-identical
    prose must both keep it.

    Serialized iterations share a step_id (minted once per pattern run) and
    a turn_id, so they are indistinguishable from a parallel batch by key
    alone; the removed prose dedup blanked B here, dropping content the live
    message history had. A parallel batch cannot produce this shape in the
    first place -- ``_remember_tool_call_content`` stamps prose on only the
    first non-control call of a batch -- so there is nothing to trade off.
    """
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="go",
            created_at=_ts(-1),
            turn_id="turn-1",
        )
        repeated_prose = "Let me check the file."

        # Fully serialized: A's end precedes B's start.
        for index, call_id in enumerate(("call-A", "call-B")):
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_start",
                timestamp=_ts(index * 10),
                turn_id="turn-1",
                data={
                    "tool_name": "read_file",
                    "tool_params": {"path": f"f{index}.py"},
                    "tool_call_id": call_id,
                    "assistant_content": repeated_prose,
                },
            )
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_end",
                timestamp=_ts(index * 10 + 5),
                turn_id="turn-1",
                data={
                    "tool_name": "read_file",
                    "tool_call_id": call_id,
                    "success": True,
                    "result": {"index": index},
                },
            )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        assistant_contents = [
            m["content"]
            for m in messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        assert assistant_contents == [repeated_prose, repeated_prose]
    finally:
        db_session.close()


def test_same_prose_separated_by_a_prose_less_exchange_is_kept():
    """The removed dedup skipped prose-less exchanges without resetting its
    "previous" value, so an intervening exchange with no prose did not break
    the run of duplicates. All three exchanges here must come through with
    exactly the prose their own start event carried."""
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        _add_chat_message(
            db_session,
            task,
            role="user",
            content="go",
            created_at=_ts(-1),
            turn_id="turn-1",
        )
        repeated_prose = "Let me check the file."
        proses = [repeated_prose, None, repeated_prose]

        for index, prose in enumerate(proses):
            call_id = f"call-{index}"
            data = {
                "tool_name": "read_file",
                "tool_params": {"path": f"f{index}.py"},
                "tool_call_id": call_id,
            }
            if prose is not None:
                data["assistant_content"] = prose
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_start",
                timestamp=_ts(index * 10),
                turn_id="turn-1",
                data=data,
            )
            _add_trace_event(
                db_session,
                task,
                event_type="tool_execution_end",
                timestamp=_ts(index * 10 + 5),
                turn_id="turn-1",
                data={
                    "tool_name": "read_file",
                    "tool_call_id": call_id,
                    "success": True,
                    "result": {"index": index},
                },
            )

        messages = load_task_conversation_context_sync(db_session, int(task.id))
        assistant_contents = [
            m["content"]
            for m in messages
            if m["role"] == "assistant" and m.get("tool_calls")
        ]
        assert assistant_contents == [repeated_prose, "", repeated_prose]
    finally:
        db_session.close()

"""Model setup consumes facts even after compatibility content is removed."""

from unittest.mock import patch

import pytest
import sqlalchemy as sa

from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.context.execution import MODEL_CONTEXT_WATERMARK_METADATA_KEY
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services.chat_history_service import persist_user_message_no_commit
from xagent.web.services.task_event_context_service import load_task_event_context
from xagent.web.services.task_execution_event_writer import (
    append_fact_no_commit,
    stage_applied_inputs_no_commit,
)

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


def fact(db, task_id, kind, key, payload, **kwargs):
    return append_fact_no_commit(
        db, task_id=task_id, kind=kind, key=key, payload=payload, **kwargs
    )


def accept(db, task_id, turn, text="same", **kwargs):
    return persist_user_message_no_commit(
        db, task_id, db.get(Task, task_id).user_id, text, turn_id=turn, **kwargs
    )


def apply(db, task_id, turn):
    state = fact(
        db,
        task_id,
        "recovery_state",
        f"state:{turn}",
        {
            "protocol_event_id": f"state:{turn}",
            "data": {
                "checkpoint_type": "agent_execution_checkpoint",
                "snapshot_schema_version": 1,
                "snapshot": {
                    "execution_id": "root-execution",
                    "context": {
                        "messages": [
                            {
                                "role": "user",
                                "content": "same",
                                "metadata": {"turn_id": turn},
                            }
                        ]
                    },
                },
            },
        },
    )
    stage_applied_inputs_no_commit(db, state)


def tool(
    db,
    task_id,
    batch,
    attempt,
    *,
    kind="tool_execution_start",
    result=None,
    scope_id="root",
):
    data = {
        "tool_name": "search",
        "tool_call_id": attempt,
        "tool_params": {"query": attempt},
    }
    if kind != "tool_execution_start":
        data["result"] = result
    return fact(
        db,
        task_id,
        kind,
        f"{attempt}:{kind}",
        {"data": data},
        tool_attempt_id=attempt,
        assistant_message_id=batch,
        scope_id=scope_id,
    )


def purge_legacy(db, task_id):
    db.execute(sa.delete(TaskChatMessage).where(TaskChatMessage.task_id == task_id))
    db.execute(sa.delete(TraceEvent).where(TraceEvent.task_id == task_id))
    db.commit()


def test_applied_inputs_and_cutoff_are_occurrences_not_text(canonical):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "t1")
        apply(db, task_id, "t1")
        accept(db, task_id, "t2")
        apply(db, task_id, "t2")
        current = accept(db, task_id, "t3")
        accept(db, task_id, "pending", "not applied")
        db.commit()
        old_id = current.id
        mapped = load_task_event_context(db, task_id, before_message_id=old_id)
        purge_legacy(db, task_id)
        native = load_task_event_context(db, task_id, before_turn_id="t3")
        assert native == mapped
        assert native.messages == [{"role": "user", "content": "same"}] * 2
        assert load_task_event_context(db, task_id).messages == native.messages


@pytest.mark.parametrize(
    ("task_status", "result_status"),
    [
        ("waiting_for_user", "waiting_for_user"),
        ("paused", "interrupted"),
        ("paused", None),
    ],
)
def test_normal_pause_then_success_does_not_add_failure(
    canonical, task_status, result_status
):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "question", text="Find the weather")
        apply(db, task_id, "question")
        fact(
            db,
            task_id,
            "agent_message",
            "clarification",
            {"message": "Which city?", "expect_response": True},
        )
        result = {"success": False}
        if result_status is not None:
            result["status"] = result_status
        fact(
            db,
            task_id,
            "execution_settled",
            "pause",
            {"status": task_status, "result": result},
        )
        db.commit()
        expected = [
            {"role": "user", "content": "Find the weather"},
            {"role": "assistant", "content": "Which city?"},
        ]
        assert load_task_event_context(db, task_id).messages == expected
        accept(db, task_id, "answer", text="Taipei")
        apply(db, task_id, "answer")
        fact(
            db,
            task_id,
            "execution_settled",
            "complete",
            {"status": "completed", "result": {"success": True, "status": "completed"}},
        )
        db.commit()
        assert load_task_event_context(db, task_id).messages == expected + [
            {"role": "user", "content": "Taipei"}
        ]


@pytest.mark.parametrize(
    ("status", "summary"),
    [
        ("failed", "- Previous execution failed."),
        ("cancelled", "- Previous execution was cancelled."),
    ],
)
def test_terminal_settlement_still_adds_outcome(canonical, status, summary):
    factory, task_id = canonical
    with factory() as db:
        fact(
            db,
            task_id,
            "execution_settled",
            "terminal",
            {"status": status, "result": {"success": False, "status": status}},
        )
        db.commit()
        assert load_task_event_context(db, task_id).messages == [
            {"role": "system", "content": summary}
        ]


def test_batch_pairing_failure_skill_and_root_isolation(canonical):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "t1")
        apply(db, task_id, "t1")
        tool(db, task_id, "batch", "a")
        tool(db, task_id, "batch", "b")
        tool(
            db,
            task_id,
            "batch",
            "b",
            kind="tool_execution_end",
            result={"cursor": "keep-whole", "value": 2},
        )
        tool(
            db,
            task_id,
            "batch",
            "a",
            kind="tool_execution_failed",
            result={"success": False, "error": "bad input"},
        )
        tool(db, task_id, "child", "child", scope_id="child-1")
        fact(
            db,
            task_id,
            "assistant_message",
            "failure",
            {"content": "safe placeholder", "message_type": "task_failure"},
        )
        fact(
            db,
            task_id,
            "execution_settled",
            "settled",
            {
                "status": "failed",
                "result": {"success": False, "failure_reason": "quota"},
            },
        )
        fact(
            db,
            task_id,
            "skill_select_end",
            "skill",
            {"data": {"selected": True, "skill_name": "csv"}},
        )
        fact(
            db,
            task_id,
            "skill_select_end",
            "child-skill",
            {"data": {"selected": True, "skill_name": "private"}},
            scope_id="child-1",
        )
        purge_legacy(db, task_id)
        context = load_task_event_context(db, task_id)
    assert [m["role"] for m in context.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "system",
    ]
    assert [c["id"] for c in context.messages[1]["tool_calls"]] == ["a", "b"]
    assert context.messages[2]["raw_result"]["error"] == "bad input"
    assert context.messages[3]["raw_result"]["cursor"] == "keep-whole"
    assert context.messages[4]["content"] == "- Previous execution failed: reason=quota"
    assert context.selected_skill_name == "csv"


def test_native_summary_suffix_retains_late_application_and_tool_result(canonical):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "old", "covered")
        apply(db, task_id, "old")
        accept(db, task_id, "late", "accepted early applied late")
        tool(db, task_id, "batch", "tool")
        db.commit()
        prior = load_task_event_context(db, task_id)
        fact(
            db,
            task_id,
            "action_end_compact",
            "summary",
            {
                "data": {
                    "summary": "saved summary",
                    MODEL_CONTEXT_WATERMARK_METADATA_KEY: prior.watermark,
                }
            },
        )
        apply(db, task_id, "late")
        tool(
            db,
            task_id,
            "batch",
            "tool",
            kind="tool_execution_end",
            result={"value": "late outcome"},
        )
        accept(db, task_id, "current", "current")
        purge_legacy(db, task_id)
        context = load_task_event_context(db, task_id, before_turn_id="current")
    assert context.messages[0] == {"role": "system", "content": "saved summary"}
    assert [m["content"] for m in context.messages if m["role"] == "user"] == [
        "accepted early applied late"
    ]
    assert next(m for m in context.messages if m["role"] == "tool")["raw_result"] == {
        "value": "late outcome"
    }


def test_legacy_summary_covers_transcript_only_and_historical_cutoff(canonical):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "old")
        apply(db, task_id, "old")
        tool(db, task_id, "batch", "a")
        tool(db, task_id, "batch", "a", kind="tool_execution_end", result={"keep": 1})
        accepted = db.scalar(
            sa.select(TaskExecutionEvent).where(
                TaskExecutionEvent.kind == "input_accepted"
            )
        )
        accept(db, task_id, "cutoff", "later")
        coordinate = {
            "scope_id": "root",
            "sequence": accepted.sequence,
            "event_id": accepted.event_id,
        }
        fact(
            db,
            task_id,
            "action_end_compact",
            "summary",
            {
                "data": {"summary": "old format summary"},
                "transcript_watermark": coordinate,
            },
        )
        db.commit()
        before = load_task_event_context(db, task_id, before_turn_id="cutoff")
        after = load_task_event_context(db, task_id)
    assert before.messages[0]["content"] == "same"
    assert after.messages[0]["content"] == "old format summary"
    assert len([m for m in after.messages if m["role"] == "tool"]) == 1


def test_window_keeps_entire_boundary_batch(canonical):
    factory, task_id = canonical
    with factory() as db:
        for index in range(12):
            tool(db, task_id, f"batch-{index // 3}", str(index))
            tool(
                db,
                task_id,
                f"batch-{index // 3}",
                str(index),
                kind="tool_execution_end",
                result={"index": index},
            )
        db.commit()
        messages = load_task_event_context(db, task_id).messages
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == [
        str(i) for i in range(3, 12)
    ]
    assert [len(m["tool_calls"]) for m in messages if m["role"] == "assistant"] == [
        3,
        3,
        3,
    ]


def test_question_and_unknown_tool_outcome(canonical):
    factory, task_id = canonical
    with factory() as db:
        fact(
            db,
            task_id,
            "agent_message",
            "question",
            {"data": {"message": "Which file?", "expect_response": True}},
        )
        tool(db, task_id, "batch", "unknown")
        db.commit()
        messages = load_task_event_context(db, task_id).messages
    assert messages[0]["content"] == "Which file?"
    assert messages[-1]["raw_result"]["status"] == "unknown"


@pytest.mark.parametrize("broken", ["version", "missing-start", "summary-scope"])
def test_corrupt_required_facts_fail_explicitly(canonical, broken):
    factory, task_id = canonical
    with factory() as db:
        if broken == "version":
            accept(db, task_id, "t1")
            db.execute(sa.update(TaskExecutionEvent).values(payload_version=2))
        elif broken == "missing-start":
            tool(db, task_id, "batch", "missing", kind="tool_execution_end", result={})
        else:
            fact(
                db,
                task_id,
                "action_end_compact",
                "bad",
                {
                    "data": {
                        "summary": "bad",
                        MODEL_CONTEXT_WATERMARK_METADATA_KEY: {
                            "scope_id": "child",
                            "sequence": 1,
                        },
                    }
                },
            )
        db.commit()
        with pytest.raises(ValueError):
            load_task_event_context(db, task_id)


def test_fixed_horizon_paging_does_not_absorb_later_commit(canonical):
    factory, task_id = canonical
    with factory() as db:
        for i in range(105):
            fact(
                db,
                task_id,
                "assistant_message",
                f"a{i}",
                {"content": str(i), "message_type": "assistant_response"},
            )
        db.commit()
    original = factory.class_.scalars
    inserted = False

    def scalars(session, statement, *args, **kwargs):
        nonlocal inserted
        result = original(session, statement, *args, **kwargs)
        if not inserted and "ORDER BY task_execution_events.sequence" in str(statement):
            inserted = True
            # Same session models a later visible append without a SQLite reader lock.
            fact(
                session,
                task_id,
                "assistant_message",
                "later",
                {"content": "later", "message_type": "assistant_response"},
            )
        return result

    with factory() as db, patch.object(factory.class_, "scalars", scalars):
        context = load_task_event_context(db, task_id)
        assert len(context.messages) == 105
        assert all(message["content"] != "later" for message in context.messages)


def test_actual_setup_never_reads_legacy_content(canonical, monkeypatch):
    from xagent.web.services import task_setup_snapshot as setup

    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "old", "previous")
        apply(db, task_id, "old")
        accept(db, task_id, "current", "new")
        purge_legacy(db, task_id)
    monkeypatch.setattr(setup, "get_session_local", lambda: factory)

    def block(*args, **kwargs):
        raise AssertionError("legacy content reader called")

    monkeypatch.setattr(
        "xagent.web.services.chat_history_service.load_task_transcript_window", block
    )
    monkeypatch.setattr(setup, "load_task_execution_recovery_snapshot_sync", block)
    # Runtime config uses existing model resolution, without network calls.
    snapshot = setup.load_task_setup_snapshot_sync(
        task_id, None, before_turn_id="current"
    )
    assert snapshot.task.conversation_storage_version == 2
    assert snapshot.conversation_history == ({"role": "user", "content": "previous"},)
    assert snapshot.conversation_watermark is None
    assert snapshot.conversation_event_watermark["scope_id"] == "root"


@pytest.mark.asyncio
async def test_native_compaction_roundtrip_without_projection(canonical):
    from tests.web.services.test_execution_event_read_contract import compact_event
    from xagent.web.tracing import ExecutionEventTraceAdapter

    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "t1", "prior history")
        apply(db, task_id, "t1")
        purge_legacy(db, task_id)
        loaded = load_task_event_context(db, task_id)
    context = ExecutionContext(execution_id="native")
    context.metadata[MODEL_CONTEXT_WATERMARK_METADATA_KEY] = loaded.watermark
    context.add_user_message("prior history")
    context.add_assistant_message("worked")
    context.add_user_message("continue")
    compact = context.compact_with_llm_response({"summary": "Work is saved"})
    assert compact.metadata[MODEL_CONTEXT_WATERMARK_METADATA_KEY] == loaded.watermark
    assert "watermark_message_id" not in compact.metadata
    event = compact_event(task_id, compact.metadata)
    await ExecutionEventTraceAdapter(task_id).commit_event(event)
    await ExecutionEventTraceAdapter(task_id).commit_event(event)
    with factory() as db:
        after = load_task_event_context(db, task_id)
    assert len(after.messages) == 1
    assert "Work is saved" in after.messages[0]["content"]
    assert (
        MODEL_CONTEXT_WATERMARK_METADATA_KEY
        not in context.create_child_context().metadata
    )


def test_attachment_budget_prefers_recent_images_over_summary(canonical):
    from xagent.core.agent.attachments import build_image_context_references
    from xagent.core.context_ref import CONTEXT_REFS_KEY

    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "old")
        apply(db, task_id, "old")
        db.commit()
        before = load_task_event_context(db, task_id)

        def image(i):
            return {"file_id": f"image-{i}", "name": f"{i}.png", "type": "image/png"}

        summary_refs = [
            r.durable_dict()
            for r in build_image_context_references([image(-1), image(-2)])
        ]
        fact(
            db,
            task_id,
            "action_end_compact",
            "summary",
            {
                "data": {
                    "summary": "with images",
                    "summary_context_refs": summary_refs,
                    MODEL_CONTEXT_WATERMARK_METADATA_KEY: before.watermark,
                }
            },
        )
        for i in range(17):
            accept(db, task_id, str(i), "", attachments=[image(i)])
            apply(db, task_id, str(i))
        purge_legacy(db, task_id)
        after = load_task_event_context(db, task_id)
    assert CONTEXT_REFS_KEY not in after.messages[0]
    references = [
        ref for msg in after.messages for ref in msg.get(CONTEXT_REFS_KEY, [])
    ]
    assert len(references) == 16
    assert [ref["file_ref"]["file_id"] for ref in references] == [
        f"image-{i}" for i in range(1, 17)
    ]


@pytest.mark.asyncio
async def test_event_context_reaches_actual_model_with_batch_and_current_input_once(
    canonical,
):
    from tests.core.agent.test_execution_adapter import FakeLLM, FakeTool
    from xagent.core.agent.service import AgentService

    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "old", "earlier")
        apply(db, task_id, "old")
        tool(db, task_id, "batch", "a")
        tool(db, task_id, "batch", "b")
        tool(db, task_id, "batch", "b", kind="tool_execution_end", result={"n": 2})
        tool(db, task_id, "batch", "a", kind="tool_execution_end", result={"n": 1})
        accept(db, task_id, "current", "continue")
        purge_legacy(db, task_id)
        loaded = load_task_event_context(db, task_id, before_turn_id="current")
    llm = FakeLLM(["done"])
    service = AgentService(
        name="context",
        id="context",
        pattern="react",
        llm=llm,
        tools=[FakeTool()],
        tool_config=None,
    )
    service.allowed_skills = []
    service.set_conversation_history(loaded.messages, event_watermark=loaded.watermark)
    result = await service.execute_task("continue", task_id="model-context")
    assert result["success"] is True
    messages = llm.calls[0]["messages"]
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "earlier",
        "continue",
    ]
    index = next(i for i, m in enumerate(messages) if m.get("tool_calls"))
    assert [c["id"] for c in messages[index]["tool_calls"]] == ["a", "b"]
    assert [messages[index + i]["role"] for i in (1, 2)] == ["tool", "tool"]
    assert (
        service.execution_metadata[MODEL_CONTEXT_WATERMARK_METADATA_KEY]
        == loaded.watermark
    )


def test_reference_queries_are_batched_and_include_internal_summary_anchors(canonical):
    factory, task_id = canonical
    with factory() as db:
        for index in range(105):
            accept(db, task_id, str(index))
            apply(db, task_id, str(index))
            anchor = fact(
                db,
                task_id,
                "llm_call_end",
                f"internal-{index}",
                {"data": {"response": "not model history"}},
            )
            fact(
                db,
                task_id,
                "action_end_compact",
                f"compact-{index}",
                {
                    "data": {
                        "summary": f"summary {index}",
                        MODEL_CONTEXT_WATERMARK_METADATA_KEY: {
                            "scope_id": "root",
                            "event_id": anchor.event_id,
                            "sequence": anchor.sequence,
                        },
                    }
                },
            )
        db.commit()
        statements = []

        def record(_conn, _cursor, statement, parameters, _context, _many):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append((statement, parameters))

        sa.event.listen(db.bind, "before_cursor_execute", record)
        try:
            loaded = load_task_event_context(db, task_id)
        finally:
            sa.event.remove(db.bind, "before_cursor_execute", record)
    assert loaded.messages == [{"role": "system", "content": "summary 104"}]
    # 105 input applications and 105 summaries must not cause 210 round trips.
    assert len(statements) <= 10
    lookups = [
        (sql, parameters) for sql, parameters in statements if "event_id IN" in sql
    ]
    assert len(lookups) == 3
    assert all(len(parameters) <= 103 for _, parameters in lookups)
    assert all("task_execution_events.payload," not in sql for sql, _ in lookups)


@pytest.mark.parametrize("invalid", [None, "", [], 7])
def test_coordinate_identity_is_rejected_before_writer_sql(canonical, invalid):
    from xagent.web.services.task_execution_event_writer import (
        compact_transcript_event_watermark,
    )

    factory, task_id = canonical
    coordinate = {"scope_id": "root", "sequence": 1}
    if invalid is not None:
        coordinate["event_id"] = invalid
    data = {
        "summary": "invalid identity",
        MODEL_CONTEXT_WATERMARK_METADATA_KEY: coordinate,
    }
    with factory() as db:
        with patch.object(
            db, "scalar", side_effect=AssertionError("invalid identity reached SQL")
        ):
            with pytest.raises(
                ValueError, match="Invalid native model context watermark"
            ):
                compact_transcript_event_watermark(db, task_id, "invalid", data)
        fact(db, task_id, "action_end_compact", "invalid", {"data": data})
        db.commit()
        with pytest.raises(ValueError, match="Invalid model context coverage identity"):
            load_task_event_context(db, task_id)


@pytest.mark.parametrize("invalid", ["scope", "kind", "version", "future"])
def test_batched_application_anchors_keep_existing_rejection_rules(canonical, invalid):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "turn")
        apply(db, task_id, "turn")
        state = db.scalar(
            sa.select(TaskExecutionEvent).where(
                TaskExecutionEvent.kind == "recovery_state"
            )
        )
        if invalid == "scope":
            state.scope_id = "child"
        elif invalid == "kind":
            state.kind = "llm_call_end"
        elif invalid == "version":
            state.payload_version = 2
        else:
            later = fact(db, task_id, "recovery_state", "later-state", {})
            applied = db.scalar(
                sa.select(TaskExecutionEvent).where(
                    TaskExecutionEvent.kind == "input_applied"
                )
            )
            applied.payload = {"recovery_event_id": later.event_id}
        db.commit()
        with pytest.raises(
            ValueError, match="Applied input has no prior root recovery state"
        ):
            load_task_event_context(db, task_id)


def test_empty_task_keeps_zero_horizon(canonical):
    factory, task_id = canonical
    with factory() as db:
        assert db.get(Task, task_id).conversation_event_sequence == 0
        loaded = load_task_event_context(db, task_id)
        assert loaded.messages == []
        assert loaded.watermark is None


def test_boolean_sequence_is_not_a_native_coordinate(canonical):
    from xagent.web.services.task_execution_event_writer import (
        compact_transcript_event_watermark,
    )

    factory, task_id = canonical
    data = {
        "summary": "invalid sequence",
        MODEL_CONTEXT_WATERMARK_METADATA_KEY: {
            "scope_id": "root",
            "sequence": True,
            "event_id": "anchor",
        },
    }
    with factory() as db:
        with patch.object(
            db, "scalar", side_effect=AssertionError("boolean sequence reached SQL")
        ):
            with pytest.raises(
                ValueError, match="Invalid native model context watermark"
            ):
                compact_transcript_event_watermark(db, task_id, "invalid", data)
        fact(db, task_id, "action_end_compact", "invalid", {"data": data})
        db.commit()
        with pytest.raises(ValueError, match="Invalid model context coverage sequence"):
            load_task_event_context(db, task_id)


def test_skill_only_recovery_reads_latest_root_selection_without_model_payloads(
    canonical,
):
    from xagent.web.services.task_execution_context_service import (
        load_task_execution_recovery_snapshot_sync,
    )

    factory, task_id = canonical
    with factory() as db:
        assert (
            load_task_execution_recovery_snapshot_sync(db, task_id).selected_skill_name
            is None
        )
        fact(
            db,
            task_id,
            "skill_select_end",
            "first-skill",
            {"data": {"selected": True, "skill_name": " csv "}},
        )
        child = fact(
            db,
            task_id,
            "skill_select_end",
            "child-skill",
            {"data": {"selected": True, "skill_name": "child"}},
            scope_id="child",
        )
        horizon = child.sequence
        for index in range(105):
            # Skill recovery does not load or interpret unrelated model facts.
            fact(
                db,
                task_id,
                "tool_execution_end",
                f"unrelated:{index}",
                {"large_result": "unrelated"},
            )
        future = fact(
            db,
            task_id,
            "skill_select_end",
            "future-skill",
            {"data": {"selected": True, "skill_name": "future"}},
        )
        db.get(Task, task_id).conversation_event_sequence = horizon
        db.commit()
        statements = []

        def record(_conn, _cursor, statement, parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append((statement, parameters))

        sa.event.listen(db.get_bind(), "before_cursor_execute", record)
        try:
            with patch(
                "xagent.web.services.task_event_context_service.load_task_event_context",
                side_effect=AssertionError("full history must not be reconstructed"),
            ):
                recovered = load_task_execution_recovery_snapshot_sync(db, task_id)
        finally:
            sa.event.remove(db.get_bind(), "before_cursor_execute", record)
        assert recovered.selected_skill_name == "csv"
        assert recovered.messages == ()
        assert len(statements) == 2  # storage version + one bounded skill selection
        parameters = statements[-1][1]
        assert "skill_select_end" in (
            parameters.values() if isinstance(parameters, dict) else parameters
        )
        assert "LIMIT" in statements[-1][0]
        db.get(Task, task_id).conversation_event_sequence = future.sequence
        fact(
            db,
            task_id,
            "skill_select_end",
            "deselected",
            {"data": {"selected": False, "skill_name": "csv"}},
        )
        db.commit()
        assert (
            load_task_execution_recovery_snapshot_sync(db, task_id).selected_skill_name
            is None
        )


def test_skill_only_recovery_rejects_invalid_selected_event(canonical):
    from xagent.web.services.task_execution_context_service import (
        load_task_execution_recovery_snapshot_sync,
    )

    factory, task_id = canonical
    with factory() as db:
        event = fact(db, task_id, "skill_select_end", "bad-skill", {"data": []})
        event_id = event.event_id
        db.commit()
        with pytest.raises(ValueError) as error:
            load_task_execution_recovery_snapshot_sync(db, task_id)
        assert f"task_id={task_id}" in str(error.value)
        assert event_id in str(error.value)


@pytest.mark.parametrize(
    ("kind", "payload", "identity"),
    [
        ("input_accepted", {"role": "assistant", "content": "private content"}, {}),
        (
            "action_end_compact",
            {
                "data": {
                    "summary": "private content",
                    MODEL_CONTEXT_WATERMARK_METADATA_KEY: {
                        "scope_id": "child",
                        "event_id": "invalid",
                        "sequence": 1,
                    },
                }
            },
            {},
        ),
        ("execution_settled", {"result": []}, {}),
        (
            "tool_execution_end",
            {"data": {"tool_name": "search", "result": "private content"}},
            {"tool_attempt_id": "orphan", "assistant_message_id": "batch"},
        ),
        (
            "tool_execution_start",
            {"data": {"tool_name": "search"}},
            {"tool_attempt_id": "start", "assistant_message_id": "batch"},
        ),
    ],
)
def test_model_history_errors_identify_task_and_offending_event(
    canonical, kind, payload, identity
):
    factory, task_id = canonical
    with factory() as db:
        event = fact(db, task_id, kind, "malformed", payload, **identity)
        event_id = event.event_id
        db.commit()
        with pytest.raises(ValueError) as error:
            load_task_event_context(db, task_id)
        assert f"task_id={task_id}" in str(error.value)
        assert f"event_id={event_id}" in str(error.value)
        assert "private content" not in str(error.value)


@pytest.mark.parametrize(
    "cutoff", [{"before_turn_id": "missing-turn"}, {"before_message_id": 123456}]
)
def test_missing_cutoff_errors_identify_task_and_requested_cutoff(canonical, cutoff):
    factory, task_id = canonical
    with factory() as db:
        with pytest.raises(ValueError) as error:
            load_task_event_context(db, task_id, **cutoff)
        assert f"task_id={task_id}" in str(error.value)
        assert str(next(iter(cutoff.values()))) in str(error.value)


@pytest.mark.parametrize("metadata", [[], ["private content"]])
def test_invalid_question_metadata_names_the_event(canonical, metadata):
    factory, task_id = canonical
    with factory() as db:
        event = fact(
            db,
            task_id,
            "agent_message",
            "question",
            {
                "data": {
                    "expect_response": True,
                    "message": "private content",
                    "metadata": metadata,
                }
            },
        )
        event_id = event.event_id
        db.commit()
        with pytest.raises(ValueError, match="Invalid question metadata") as error:
            load_task_event_context(db, task_id)
        assert f"task_id={task_id}" in str(error.value)
        assert f"event_id={event_id}" in str(error.value)
        assert "private content" not in str(error.value)


@pytest.mark.parametrize("fields", [{}, {"metadata": None}, {"metadata": {}}])
def test_question_metadata_may_be_absent_null_or_an_object(canonical, fields):
    factory, task_id = canonical
    with factory() as db:
        fact(
            db,
            task_id,
            "agent_message",
            "question",
            {"data": {"expect_response": True, "message": "Which city?", **fields}},
        )
        db.commit()
        assert load_task_event_context(db, task_id).messages == [
            {"role": "assistant", "content": "Which city?"}
        ]

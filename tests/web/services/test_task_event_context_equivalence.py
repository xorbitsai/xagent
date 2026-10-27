"""The model-context reader must project exactly what the full-history oracle does.

Every scenario is swept over each committed horizon H, and at each H over every
accepted root turn and legacy message cutoff. Each read uses its own Session.
"""

import pytest
import sqlalchemy as sa

from tests.shared.execution_event_read_guard import forbid_legacy_content
from tests.web.services.task_event_context_reference import (
    reference_load_task_event_context,
)
from tests.web.services.test_task_event_context_service import (
    accept,
    apply,
    fact,
    purge_legacy,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.core.agent.context.execution import MODEL_CONTEXT_WATERMARK_METADATA_KEY
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.services import task_event_context_service

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture


def _read(factory, loader, task_id, **cutoff):
    with factory() as db:
        try:
            return ("ok", loader(db, task_id, **cutoff))
        except ValueError as error:
            return ("error", str(error))


def compare(factory, task_id, **cutoff):
    expected = _read(factory, reference_load_task_event_context, task_id, **cutoff)
    actual = _read(
        factory, task_event_context_service.load_task_event_context, task_id, **cutoff
    )
    assert actual == expected, cutoff
    return expected


def _live_projection(db, task_id):
    horizon = db.get(Task, task_id).conversation_event_sequence
    return task_event_context_service._project(
        db,
        task_id,
        horizon,
        task_event_context_service._root_events(
            db,
            task_id,
            after=0,
            through=horizon,
            kinds=task_event_context_service._MODEL_KINDS,
        ),
    )


def compare_live(factory, task_id):
    """The floor-based reader must equal the projection of the whole history.

    Unlike the frozen oracle, this stays after the oracle is removed. It guards
    ``_covered_facts_still_used`` against drifting from ``_project``. Only the
    plain horizon is compared: before_turn_id / before_message_id cutoffs just
    lower the horizon, and reproducing that here would duplicate production
    logic; the frozen oracle already covers them.
    """
    expected = _read(factory, _live_projection, task_id)
    actual = _read(factory, task_event_context_service.load_task_event_context, task_id)
    assert actual == expected


def set_horizon(factory, task_id, horizon):
    with factory() as db:
        db.execute(
            sa.update(Task)
            .where(Task.id == task_id)
            .values(conversation_event_sequence=horizon)
        )
        db.commit()


def sweep(factory, task_id, *, live=True):
    """Compare at every horizon and cutoff; return the outcome at the top.

    ``live`` adds the permanent full-history comparison; scenarios with
    deliberately invalid data opt out, since the reader no longer validates
    facts it does not use.
    """
    with factory() as db:
        top = db.get(Task, task_id).conversation_event_sequence
        turns = sorted(
            db.scalars(
                sa.select(TaskExecutionEvent.turn_id).where(
                    TaskExecutionEvent.task_id == task_id,
                    TaskExecutionEvent.scope_id == "root",
                    TaskExecutionEvent.kind == "input_accepted",
                    TaskExecutionEvent.turn_id.is_not(None),
                )
            )
        )
        message_ids = sorted(
            db.scalars(
                sa.select(TaskChatMessage.id).where(
                    TaskChatMessage.task_id == task_id,
                    TaskChatMessage.execution_event_id.is_not(None),
                )
            )
        )
    try:
        with forbid_legacy_content(factory.kw["bind"]):
            for horizon in range(top + 1):
                set_horizon(factory, task_id, horizon)
                latest = compare(factory, task_id)
                if live:
                    compare_live(factory, task_id)
                for turn in turns:
                    compare(factory, task_id, before_turn_id=turn)
                for message_id in message_ids:
                    compare(factory, task_id, before_message_id=message_id)
    finally:
        set_horizon(factory, task_id, top)
    return latest


def last_root(db, task_id):
    db.flush()
    return db.scalar(
        sa.select(TaskExecutionEvent)
        .where(
            TaskExecutionEvent.task_id == task_id,
            TaskExecutionEvent.scope_id == "root",
        )
        .order_by(TaskExecutionEvent.sequence.desc())
        .limit(1)
    )


def coordinate(event, **override):
    return {
        "scope_id": "root",
        "sequence": int(event.sequence),
        "event_id": event.event_id,
        **override,
    }


def summarize(db, task_id, key, anchor, *, text=None, legacy=False, **extra):
    """Append a summary covering everything through ``anchor``."""
    data = {"summary": f"summary {key}" if text is None else text, **extra}
    payload = {"data": data}
    if anchor is not None and legacy:
        payload["transcript_watermark"] = coordinate(anchor)
    elif anchor is not None:
        data[MODEL_CONTEXT_WATERMARK_METADATA_KEY] = coordinate(anchor)
    return fact(db, task_id, "action_end_compact", key, payload)


def recover(db, event, coverage):
    """Rewrite a stored summary's coverage claim, as corruption would."""
    data = {**event.payload["data"], MODEL_CONTEXT_WATERMARK_METADATA_KEY: coverage}
    event.payload = {**event.payload, "data": data}
    db.flush()


def say(db, task_id, key, text=None):
    return fact(
        db,
        task_id,
        "assistant_message",
        key,
        {"content": text or key, "message_type": "assistant_response"},
    )


def settle(db, task_id, key, status):
    return fact(
        db,
        task_id,
        "execution_settled",
        key,
        {"status": status, "result": {"success": False, "status": status}},
    )


def call(db, task_id, batch, attempt, kind="tool_execution_start", **kwargs):
    scope_id = kwargs.pop("scope_id", "root")
    data = {
        "tool_name": kwargs.pop("tool_name", "search"),
        "tool_call_id": attempt,
        "tool_params": {"q": attempt},
    }
    if kind != "tool_execution_start":
        data["result"] = kwargs.pop("result", {"value": attempt})
    return fact(
        db,
        task_id,
        kind,
        f"{attempt}:{kind}",
        {"data": data},
        tool_attempt_id=attempt,
        assistant_message_id=batch,
        scope_id=scope_id,
        **kwargs,
    )


def done(db, task_id, batch, attempt, **kwargs):
    return call(db, task_id, batch, attempt, "tool_execution_end", **kwargs)


def failed(db, task_id, batch, attempt, **kwargs):
    return call(db, task_id, batch, attempt, "tool_execution_failed", **kwargs)


def messages(outcome):
    assert outcome[0] == "ok", outcome
    return outcome[1].messages


def tool_batches(result):
    return [[c["id"] for c in m["tool_calls"]] for m in result if "tool_calls" in m]


def test_native_summary_with_late_application_and_pending_input(canonical):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "old", "covered")
        apply(db, task_id, "old")
        accept(db, task_id, "late", "accepted before summary")
        accept(db, task_id, "pending", "never applied")
        say(db, task_id, "covered-answer")
        summarize(db, task_id, "s", last_root(db, task_id))
        apply(db, task_id, "late")
        say(db, task_id, "tail")
        accept(db, task_id, "current", "current")
        apply(db, task_id, "current")
        purge_legacy(db, task_id)
    assert messages(sweep(factory, task_id)) == [
        {"role": "system", "content": "summary s"},
        {"role": "user", "content": "accepted before summary"},
        {"role": "assistant", "content": "tail"},
        {"role": "user", "content": "current"},
    ]


def test_tool_batches_across_the_summary_floor(canonical):
    factory, task_id = canonical
    with factory() as db:
        # Entirely covered, and covered without any outcome.
        call(db, task_id, "early", "e1")
        done(db, task_id, "early", "e1")
        call(db, task_id, "dangling", "d1")
        # One batch crosses the floor: a finishes before it, b fails after
        # it, and c is entirely after it.
        call(db, task_id, "cross", "a")
        call(db, task_id, "cross", "b")
        done(db, task_id, "cross", "a")
        # A start before the floor whose only outcome lands after it.
        call(db, task_id, "single", "s1")
        summarize(db, task_id, "s", last_root(db, task_id))
        failed(db, task_id, "cross", "b", result={"success": False, "error": "x"})
        call(db, task_id, "cross", "c")
        done(db, task_id, "cross", "c")
        done(db, task_id, "single", "s1")
        db.commit()
    result = messages(sweep(factory, task_id))
    assert result[0]["content"] == "summary s"
    assert tool_batches(result) == [["a", "b", "c"], ["s1"]]


def test_tool_window_out_of_order_completion_and_more_than_nine_calls(canonical):
    factory, task_id = canonical
    with factory() as db:
        call(db, task_id, "first", "f1")
        call(db, task_id, "first", "f2")
        summarize(db, task_id, "s", last_root(db, task_id))
        done(db, task_id, "first", "f2")
        done(db, task_id, "first", "f1")
        for batch in range(4):
            for index in range(3):
                call(db, task_id, f"b{batch}", f"{batch}-{index}")
            for index in reversed(range(3)):
                done(db, task_id, f"b{batch}", f"{batch}-{index}")
        db.commit()
    result = messages(sweep(factory, task_id))
    assert [len(batch) for batch in tool_batches(result)] == [3, 3, 3]


def test_multiple_summaries_pick_the_last_usable_coverage(canonical):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "t1", "first")
        apply(db, task_id, "t1")
        say(db, task_id, "one")
        anchor = last_root(db, task_id)
        summarize(db, task_id, "s1", anchor)
        say(db, task_id, "two")
        summarize(db, task_id, "s2", anchor, text="same coverage again")
        say(db, task_id, "three")
        summarize(db, task_id, "s3", last_root(db, task_id), text="  ")
        summarize(db, task_id, "s4", None, text="no coverage claim")
        say(db, task_id, "four")
        purge_legacy(db, task_id)
    assert messages(sweep(factory, task_id)) == [
        {"role": "system", "content": "same coverage again"},
        {"role": "assistant", "content": "two"},
        {"role": "assistant", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def test_legacy_summary_keeps_earlier_tool_and_settlement_facts(canonical):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "old", "covered")
        apply(db, task_id, "old")
        call(db, task_id, "batch", "a")
        done(db, task_id, "batch", "a")
        settle(db, task_id, "failed", "failed")
        settle(db, task_id, "cancelled", "cancelled")
        accept(db, task_id, "late", "accepted before summary")
        say(db, task_id, "covered-answer")
        summarize(db, task_id, "s", last_root(db, task_id), legacy=True)
        apply(db, task_id, "late")
        accept(db, task_id, "after", "after summary")
        apply(db, task_id, "after")
        say(db, task_id, "tail")
        purge_legacy(db, task_id)
    result = messages(sweep(factory, task_id))
    assert result[0] == {"role": "system", "content": "summary s"}
    assert [m["content"] for m in result if m["role"] == "user"] == ["after summary"]
    assert len([m for m in result if m["role"] == "tool"]) == 1
    assert [m["content"] for m in result[1:] if m["role"] == "system"] == [
        "- Previous execution failed.",
        "- Previous execution was cancelled.",
    ]


@pytest.mark.parametrize("legacy", [False, True])
def test_settlement_statuses_on_both_sides_of_the_floor(canonical, legacy):
    factory, task_id = canonical
    statuses = ["failed", "cancelled", "paused", "waiting_for_user", "interrupted"]
    with factory() as db:
        for status in statuses:
            settle(db, task_id, f"before-{status}", status)
        say(db, task_id, "covered")
        summarize(db, task_id, "s", last_root(db, task_id), legacy=legacy)
        for status in statuses:
            settle(db, task_id, f"after-{status}", status)
        db.commit()
    result = messages(sweep(factory, task_id))
    assert len([m for m in result if m["role"] == "system"]) == (5 if legacy else 3)


def _settle_payload(db, task_id, key, payload):
    return fact(db, task_id, "execution_settled", key, payload)


@pytest.mark.parametrize("legacy", [False, True])
def test_settlements_with_agent_result_project_without_it(canonical, legacy):
    factory, task_id = canonical
    context = {"messages": [{"role": "user", "content": "q" * 4096}]}
    with factory() as db:
        for key, status in [("before", "failed"), ("before-ok", "completed")]:
            _settle_payload(
                db,
                task_id,
                key,
                {
                    "status": status,
                    "result": {
                        "success": status != "failed",
                        "status": status,
                        "failure_reason": f"{key} reason",
                        "failed_step_id": "s1",
                        "agent_result": {"context": context},
                    },
                },
            )
        say(db, task_id, "covered")
        summarize(db, task_id, "s", last_root(db, task_id), legacy=legacy)
        _settle_payload(
            db,
            task_id,
            "after",
            {
                "status": "failed",
                "result": {
                    "status": "failed",
                    "failure_reason": "after reason",
                    "agent_result": {"context": context},
                },
            },
        )
        db.commit()
    systems = [
        m["content"] for m in messages(sweep(factory, task_id)) if m["role"] == "system"
    ]
    assert any("after reason" in text for text in systems)
    assert any("before reason" in text for text in systems) == legacy


def _result_missing(db, task_id, key):
    _settle_payload(db, task_id, key, {"status": "failed"})


def _result_string(db, task_id, key):
    _settle_payload(db, task_id, key, {"status": "failed", "result": "boom"})


def _result_list(db, task_id, key):
    _settle_payload(db, task_id, key, {"status": "failed", "result": [1, 2]})


def _result_null(db, task_id, key):
    _settle_payload(db, task_id, key, {"status": "failed", "result": None})


def _agent_result_missing(db, task_id, key):
    _settle_payload(
        db,
        task_id,
        key,
        {"status": "failed", "result": {"status": "failed", "failure_reason": "r"}},
    )


def _agent_result_scalar(db, task_id, key):
    _settle_payload(
        db,
        task_id,
        key,
        {"result": {"success": False, "agent_result": "text", "failure_reason": "r"}},
    )


def _agent_result_null_with_top_status(db, task_id, key):
    _settle_payload(
        db,
        task_id,
        key,
        {"status": "cancelled", "result": {"agent_result": None}},
    )


def _payload_is_list(db, task_id, key):
    _settle_payload(db, task_id, key, [{"result": {"agent_result": 1}}])


def _payload_data_wrapper(db, task_id, key):
    _settle_payload(
        db,
        task_id,
        key,
        {
            "result": {"status": "failed", "agent_result": {"big": "x" * 64}},
            "data": {"result": {"status": "failed", "agent_result": {"k": 1}}},
        },
    )


@pytest.mark.parametrize(
    "shape",
    [
        _result_missing,
        _result_string,
        _result_list,
        _result_null,
        _agent_result_missing,
        _agent_result_scalar,
        _agent_result_null_with_top_status,
        _payload_is_list,
        _payload_data_wrapper,
    ],
)
@pytest.mark.parametrize("legacy", [False, True])
def test_settlement_shapes_without_agent_result_match_the_oracle(
    canonical, shape, legacy
):
    factory, task_id = canonical
    with factory() as db:
        shape(db, task_id, "before")
        say(db, task_id, "covered")
        summarize(db, task_id, "s", last_root(db, task_id), legacy=legacy)
        shape(db, task_id, "after")
        purge_legacy(db, task_id)
    sweep(factory, task_id)


def test_child_scopes_reuse_root_identities(canonical):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "turn", "root input")
        call(db, task_id, "batch", "a")
        call(db, task_id, "batch", "a", scope_id="child")
        fact(
            db,
            task_id,
            "skill_select_end",
            "child-skill",
            {"data": {"selected": True, "skill_name": "child"}},
            scope_id="child",
        )
        anchor = last_root(db, task_id)
        summarize(db, task_id, "s", anchor)
        fact(
            db,
            task_id,
            "action_end_compact",
            "child-summary",
            {
                "data": {
                    "summary": "child summary",
                    MODEL_CONTEXT_WATERMARK_METADATA_KEY: coordinate(anchor),
                }
            },
            scope_id="child",
        )
        done(db, task_id, "batch", "a", scope_id="child")
        fact(
            db,
            task_id,
            "input_applied",
            "child-applied",
            {"recovery_event_id": "child"},
            turn_id="turn",
            scope_id="child",
        )
        apply(db, task_id, "turn")
        done(db, task_id, "batch", "a")
        purge_legacy(db, task_id)
    outcome = sweep(factory, task_id)
    result = messages(outcome)
    assert result[0]["content"] == "summary s"
    assert [m["content"] for m in result if m["role"] == "user"] == ["root input"]
    assert len([m for m in result if m["role"] == "tool"]) == 1
    assert outcome[1].selected_skill_name is None


def test_skill_selected_only_before_the_floor_then_deselected(canonical):
    factory, task_id = canonical
    with factory() as db:
        fact(
            db,
            task_id,
            "skill_select_end",
            "first",
            {"data": {"selected": True, "skill_name": "csv"}},
        )
        say(db, task_id, "covered")
        summarize(db, task_id, "s", last_root(db, task_id))
        say(db, task_id, "kept")
        before_deselect = int(last_root(db, task_id).sequence)
        fact(
            db,
            task_id,
            "skill_select_end",
            "deselect",
            {"data": {"selected": False, "skill_name": "csv"}},
        )
        db.commit()
    assert sweep(factory, task_id)[1].selected_skill_name is None
    set_horizon(factory, task_id, before_deselect)
    assert compare(factory, task_id)[1].selected_skill_name == "csv"


def test_control_tool_batch_across_the_floor(canonical):
    factory, task_id = canonical
    with factory() as db:
        call(db, task_id, "batch", "answer", tool_name="final_answer")
        call(db, task_id, "batch", "work")
        summarize(db, task_id, "s", last_root(db, task_id))
        done(db, task_id, "batch", "answer", tool_name="final_answer")
        done(db, task_id, "batch", "work")
        db.commit()
    assert tool_batches(messages(sweep(factory, task_id))) == [["work"]]


def sweep_live(factory, task_id):
    """``sweep`` against the full-history projection only.

    The frozen oracle predates settlement facts that carry attempt identity.
    """
    with factory() as db:
        top = db.get(Task, task_id).conversation_event_sequence
    try:
        with forbid_legacy_content(factory.kw["bind"]):
            for horizon in range(top + 1):
                set_horizon(factory, task_id, horizon)
                compare_live(factory, task_id)
    finally:
        set_horizon(factory, task_id, top)
    return _read(factory, task_event_context_service.load_task_event_context, task_id)


def settlement(db, task_id, batch, attempt, kind="tool_execution_start", **kwargs):
    """A resumed interaction's settlement fact for ``attempt``, in a later run."""
    data = {
        "tool_name": "search",
        "tool_call_id": attempt,
        "tool_params": {"q": attempt},
        "settlement_delivery": True,
    }
    if kind != "tool_execution_start":
        data["result"] = kwargs.pop("result")
    return fact(
        db,
        task_id,
        kind,
        f"{attempt}:settlement:{kind}",
        {"data": data},
        tool_attempt_id=attempt,
        assistant_message_id=batch,
        run_id="resumed-run",
        **kwargs,
    )


@pytest.mark.parametrize("floor", [None, "after_wait", "inside_settlement"])
def test_settlement_replaces_the_waiting_result_across_the_floor(canonical, floor):
    factory, task_id = canonical
    with factory() as db:
        call(db, task_id, "batch", "gate", run_id="first-run")
        done(
            db,
            task_id,
            "batch",
            "gate",
            run_id="first-run",
            result={"success": False, "status": "waiting_for_user"},
        )
        if floor == "after_wait":
            summarize(db, task_id, "s", last_root(db, task_id))
        settlement(db, task_id, "batch", "gate")
        if floor == "inside_settlement":
            summarize(db, task_id, "s", last_root(db, task_id))
        settlement(
            db,
            task_id,
            "batch",
            "gate",
            "tool_execution_end",
            result={"success": True, "value": "settled"},
        )
        db.commit()
    result = messages(sweep_live(factory, task_id))
    assert tool_batches(result) == [["gate"]]
    [observation] = [m for m in result if m["role"] == "tool"]
    assert observation["raw_result"] == {"success": True, "value": "settled"}


def test_message_cutoffs_match_turn_cutoffs(canonical):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "t1", "first")
        apply(db, task_id, "t1")
        accept(db, task_id, "t2", "second")
        summarize(db, task_id, "s", last_root(db, task_id))
        apply(db, task_id, "t2")
        accept(db, task_id, "t3", "third")
        apply(db, task_id, "t3")
        accept(db, task_id, "t4", "fourth")
        db.commit()
        rows = db.execute(
            sa.select(TaskChatMessage.id, TaskChatMessage.turn_id).where(
                TaskChatMessage.task_id == task_id
            )
        ).all()
    # Legacy rows are kept: the sweep also covers every message cutoff.
    sweep(factory, task_id)
    for message_id, turn in rows:
        assert compare(factory, task_id, before_message_id=message_id) == compare(
            factory, task_id, before_turn_id=turn
        )


def test_summary_refs_and_image_budget(canonical):
    from xagent.core.agent.attachments import build_image_context_references
    from xagent.core.context_ref import CONTEXT_REFS_KEY

    def image(i):
        return {"file_id": f"image-{i}", "name": f"{i}.png", "type": "image/png"}

    factory, task_id = canonical
    with factory() as db:
        for i in range(3):
            accept(db, task_id, f"old-{i}", "", attachments=[image(100 + i)])
            apply(db, task_id, f"old-{i}")
        accept(db, task_id, "late", "", attachments=[image(-3)])
        refs = [
            r.durable_dict()
            for r in build_image_context_references([image(-1), image(-2)])
        ]
        summarize(db, task_id, "s", last_root(db, task_id), summary_context_refs=refs)
        apply(db, task_id, "late")
        for i in range(15):
            accept(db, task_id, str(i), "", attachments=[image(i)])
            apply(db, task_id, str(i))
        purge_legacy(db, task_id)
    result = messages(sweep(factory, task_id))
    references = [r for m in result for r in m.get(CONTEXT_REFS_KEY, [])]
    # The late input applied after the floor and the 15 newer ones fill the
    # budget; the summary's own references are the oldest and are dropped.
    assert CONTEXT_REFS_KEY not in result[0]
    assert [r["file_ref"]["file_id"] for r in references] == [
        "image--3",
        *(f"image-{i}" for i in range(15)),
    ]


def _break_summary_identity(db, task_id):
    say(db, task_id, "covered")
    anchor = last_root(db, task_id)
    bad = summarize(db, task_id, "bad", anchor)
    recover(db, bad, coordinate(anchor, event_id="ghost"))
    say(db, task_id, "tail")


def _break_summary_future(db, task_id):
    say(db, task_id, "covered")
    bad = summarize(db, task_id, "bad", last_root(db, task_id))
    future = say(db, task_id, "future")
    db.flush()
    recover(db, bad, coordinate(future))
    say(db, task_id, "tail")


def _break_suffix_accepted(db, task_id):
    fact(
        db,
        task_id,
        "input_accepted",
        "bad-accepted",
        {"role": "assistant", "content": "x"},
        turn_id="bad",
    )


def _break_suffix_applied(db, task_id):
    fact(
        db,
        task_id,
        "input_applied",
        "bad-applied",
        {"recovery_event_id": "ghost"},
        turn_id="t1",
    )


def _break_suffix_orphan(db, task_id):
    done(db, task_id, "orphan", "orphan")


def _break_retained_run(db, task_id):
    # Its start precedes the valid floor; the outcome keeps the batch.
    done(db, task_id, "pre", "pre", run_id="other-run")


def _break_retained_batch(db, task_id):
    # The batch is found from the start, not from the outcome's own batch id.
    done(db, task_id, "other-batch", "pre", run_id="run")


@pytest.mark.parametrize(
    "corrupt",
    [
        _break_summary_identity,
        _break_summary_future,
        _break_suffix_accepted,
        _break_suffix_applied,
        _break_suffix_orphan,
        _break_retained_run,
        _break_retained_batch,
    ],
)
def test_errors_after_the_floor_match_the_oracle(canonical, corrupt):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "t1", "first")
        apply(db, task_id, "t1")
        call(db, task_id, "pre", "pre", run_id="run")
        summarize(db, task_id, "valid", last_root(db, task_id))
        corrupt(db, task_id)
        purge_legacy(db, task_id)
    assert sweep(factory, task_id, live=False)[0] == "error"


def _covered_invalid_acceptance(db, task_id):
    fact(
        db,
        task_id,
        "input_accepted",
        "bad-accepted",
        {"role": "assistant", "content": "x"},
        turn_id="bad",
    )


def _covered_orphan_outcome(db, task_id):
    done(db, task_id, "orphan", "orphan")


def _covered_invalid_summary(db, task_id):
    anchor = say(db, task_id, "first")
    bad = summarize(db, task_id, "bad", anchor)
    recover(db, bad, coordinate(anchor, scope_id="child"))


@pytest.mark.parametrize(
    ("corrupt", "legacy", "validated"),
    [
        (_covered_invalid_acceptance, False, False),
        (_covered_orphan_outcome, False, False),
        (_covered_invalid_summary, False, False),
        (_covered_invalid_acceptance, True, False),
        # Legacy coverage still reads, and so validates, earlier tool facts.
        (_covered_orphan_outcome, True, True),
    ],
)
def test_covered_facts_that_are_not_projected_are_not_revalidated(
    canonical, corrupt, legacy, validated
):
    """B1: a usable summary's floor ends validation of facts it does not use.

    The floor is a coordinate a previous successful read returned. Only facts
    the suffix still projects are read and validated again.
    """
    factory, task_id = canonical
    with factory() as db:
        corrupt(db, task_id)
        say(db, task_id, "covered")
        summarize(db, task_id, "s", last_root(db, task_id), legacy=legacy)
        say(db, task_id, "tail")
        db.commit()
    expected = _read(factory, reference_load_task_event_context, task_id)
    actual = _read(factory, task_event_context_service.load_task_event_context, task_id)
    assert expected[0] == "error"
    if validated:
        assert actual == expected
    else:
        assert messages(actual) == [
            {"role": "system", "content": "summary s"},
            {"role": "assistant", "content": "tail"},
        ]


def test_child_summary_with_a_later_root_anchor_does_not_set_the_floor(canonical):
    factory, task_id = canonical
    with factory() as db:
        say(db, task_id, "covered")
        summarize(db, task_id, "root", last_root(db, task_id))
        say(db, task_id, "kept-1")
        say(db, task_id, "kept-2")
        later = last_root(db, task_id)
        fact(
            db,
            task_id,
            "action_end_compact",
            "child-summary",
            {
                "data": {
                    "summary": "child summary",
                    MODEL_CONTEXT_WATERMARK_METADATA_KEY: coordinate(later),
                }
            },
            scope_id="child",
        )
        say(db, task_id, "tail")
        purge_legacy(db, task_id)
    assert [m["content"] for m in messages(sweep(factory, task_id))] == [
        "summary root",
        "kept-1",
        "kept-2",
        "tail",
    ]


def test_summary_lookup_pages_past_many_unusable_summaries(canonical, monkeypatch):
    factory, task_id = canonical
    with factory() as db:
        say(db, task_id, "covered")
        summarize(db, task_id, "usable", last_root(db, task_id))
        say(db, task_id, "kept")
        anchor = last_root(db, task_id)
        # More than the limit-1 first page and one full page of newer ones.
        for i in range(110):
            if i % 2:
                summarize(db, task_id, f"empty-{i}", anchor, text="  ")
            else:
                summarize(db, task_id, f"bare-{i}", None)
        say(db, task_id, "tail")
        purge_legacy(db, task_id)
    assert [m["content"] for m in messages(sweep(factory, task_id))] == [
        "summary usable",
        "kept",
        "tail",
    ]
    # A lookup that gave up would fall back to full history with the same
    # output, so also require that the suffix read starts at the floor.
    reads = []
    original = task_event_context_service._root_events

    def spy(db, task_id, *, after, **kwargs):
        reads.append(after)
        return original(db, task_id, after=after, **kwargs)

    monkeypatch.setattr(task_event_context_service, "_root_events", spy)
    _read(factory, task_event_context_service.load_task_event_context, task_id)
    assert reads and reads[0] > 0


def test_summary_floors_need_not_increase_with_sequence(canonical):
    factory, task_id = canonical
    with factory() as db:
        say(db, task_id, "a")
        first = last_root(db, task_id)
        summarize(db, task_id, "s1", first)
        say(db, task_id, "b")
        say(db, task_id, "c")
        summarize(db, task_id, "s2", last_root(db, task_id))
        say(db, task_id, "d")
        # The latest usable summary may cover less than an earlier one.
        summarize(db, task_id, "s3", first)
        say(db, task_id, "e")
        purge_legacy(db, task_id)
    assert [m["content"] for m in messages(sweep(factory, task_id))] == [
        "summary s3",
        "b",
        "c",
        "d",
        "e",
    ]


@pytest.mark.parametrize("newest_legacy", [True, False])
def test_native_and_legacy_summaries_alternate(canonical, newest_legacy):
    factory, task_id = canonical
    with factory() as db:
        accept(db, task_id, "old", "covered")
        apply(db, task_id, "old")
        call(db, task_id, "early", "e1")
        done(db, task_id, "early", "e1")
        settle(db, task_id, "failed", "failed")
        summarize(
            db, task_id, "first", last_root(db, task_id), legacy=not newest_legacy
        )
        say(db, task_id, "middle")
        call(db, task_id, "mid", "m1")
        done(db, task_id, "mid", "m1")
        summarize(db, task_id, "second", last_root(db, task_id), legacy=newest_legacy)
        say(db, task_id, "tail")
        purge_legacy(db, task_id)
    result = messages(sweep(factory, task_id))
    assert result[0]["content"] == "summary second"
    # A legacy newest summary keeps every earlier tool and settlement fact.
    assert len([m for m in result if m["role"] == "tool"]) == (
        2 if newest_legacy else 0
    )


def test_skill_selected_exactly_at_the_floor(canonical):
    factory, task_id = canonical
    with factory() as db:
        say(db, task_id, "covered")
        fact(
            db,
            task_id,
            "skill_select_end",
            "at-floor",
            {"data": {"selected": True, "skill_name": "csv"}},
        )
        summarize(db, task_id, "s", last_root(db, task_id))
        say(db, task_id, "tail")
        purge_legacy(db, task_id)
    assert sweep(factory, task_id)[1].selected_skill_name == "csv"

"""A host settle only answers the waiting request it names."""

from types import SimpleNamespace

from xagent.core.agent.attachments import SERVER_METADATA_KEY
from xagent.core.agent.context.message import Message
from xagent.core.agent.pattern.react.react import ReActPattern


def _pattern(*interaction_ids: str) -> ReActPattern:
    pattern = ReActPattern.__new__(ReActPattern)
    pattern.waiting_for_user_request = {
        "kind": "tool_waiting_for_user",
        "message": "connect an app",
        "message_type": "question",
        "requests": [{"interaction_id": i} for i in interaction_ids],
    }
    return pattern


def _settle(interaction_id: str) -> Message:
    return Message(
        role="user",
        content="[interaction settled]",
        metadata={
            SERVER_METADATA_KEY: {
                "hidden": True,
                "kind": "interaction_settled",
                "interaction_id": interaction_id,
            }
        },
    )


def test_a_settle_for_a_pending_request_is_an_answer() -> None:
    pattern = _pattern("a", "b")
    context = SimpleNamespace(
        messages=[Message(role="assistant", content="q"), _settle("b")]
    )
    assert not pattern._is_stale_settle_message(context.messages[1])
    assert pattern._has_answer_after(context, 1)


def test_a_settle_for_another_interaction_is_not_an_answer() -> None:
    pattern = _pattern("a")
    context = SimpleNamespace(
        messages=[Message(role="assistant", content="q"), _settle("old")]
    )
    assert pattern._is_stale_settle_message(context.messages[1])
    assert not pattern._has_answer_after(context, 1)


def test_a_settle_never_answers_a_plain_question() -> None:
    pattern = ReActPattern.__new__(ReActPattern)
    pattern.waiting_for_user_request = {"message": "which one?"}
    assert pattern._is_stale_settle_message(_settle("a"))


def test_ordinary_user_messages_are_unaffected() -> None:
    pattern = _pattern("a")
    plain = Message(role="user", content="blue")
    assert not pattern._is_stale_settle_message(plain)
    context = SimpleNamespace(messages=[Message(role="assistant", content="q"), plain])
    assert pattern._has_answer_after(context, 1)


def test_the_real_answer_is_picked_over_an_earlier_stale_settle() -> None:
    pattern = _pattern("a")
    context = SimpleNamespace(
        messages=[
            Message(role="assistant", content="q"),
            _settle("old"),
            Message(role="user", content="blue"),
        ]
    )
    answer = pattern._mark_latest_user_message_as_waiting_response(
        context=context, after_message_count=1
    )
    assert answer == "blue"
    assert "response_to_waiting_for_user" not in context.messages[1].metadata
    assert "response_to_waiting_for_user" in context.messages[2].metadata


async def test_a_stale_settle_leaves_the_run_waiting() -> None:
    from unittest.mock import AsyncMock

    pattern = _pattern("a")
    pattern.status = "waiting_for_user"
    pattern.waiting_for_user_request["message_count"] = 1
    runtime = SimpleNamespace(
        approve_waiting_result=AsyncMock(return_value=True),
        checkpoint=AsyncMock(),
    )
    context = SimpleNamespace(
        messages=[Message(role="assistant", content="q"), _settle("old")],
        execution_id="e1",
    )
    result = await pattern._resume_waiting_for_user_if_needed(
        context=context, runtime=runtime, tools=[]
    )
    assert result is not None and result["status"] == "waiting_for_user"
    assert "response_to_waiting_for_user" not in context.messages[1].metadata


def test_a_settle_after_the_match_is_stale_once_the_run_resumed() -> None:
    """After the matched settle resumed the run no request is open any more."""
    pattern = _pattern("a")
    assert not pattern._is_stale_settle_message(_settle("a"))
    pattern.waiting_for_user_request = None
    assert pattern._is_stale_settle_message(_settle("a"))


class _ResumableTool:
    def __init__(self, name: str) -> None:
        self.name = name

    def resume_user_interaction(self, *, interaction_id: str, response: str):
        return None


def _batch_pattern() -> ReActPattern:
    pattern = ReActPattern.__new__(ReActPattern)
    pattern.pending_tool_interaction_responses = []
    pattern.waiting_for_user_request = {
        "kind": "tool_waiting_for_user",
        "requests": [
            {"tool_name": "connect_a", "interaction_id": "a", "tool_call_id": "ca"},
            {"tool_name": "connect_b", "interaction_id": "b", "tool_call_id": "cb"},
        ],
    }
    return pattern


def _queue(pattern: ReActPattern, **kwargs) -> list[str]:
    tools = [_ResumableTool("connect_a"), _ResumableTool("connect_b")]
    pattern._tool_name = lambda tool: tool.name  # type: ignore[method-assign]
    pattern._normalized_invocation_id = lambda value: None  # type: ignore[method-assign]
    pattern._queue_tool_interaction_responses(
        waiting_request=pattern.waiting_for_user_request,
        response="[interaction settled]",
        tools=tools,
        **kwargs,
    )
    return [
        item["interaction_id"] for item in pattern.pending_tool_interaction_responses
    ]


def test_a_settle_is_handed_to_the_matched_tool_only() -> None:
    assert _queue(_batch_pattern(), only_interaction_id="b") == ["b"]


def test_an_ordinary_reply_still_reaches_every_resumable_tool() -> None:
    assert _queue(_batch_pattern()) == ["a", "b"]


def test_the_answering_settle_id_is_read_from_the_first_answer() -> None:
    pattern = _batch_pattern()
    context = SimpleNamespace(
        messages=[
            Message(role="assistant", content="q"),
            _settle("old"),
            _settle("b"),
        ]
    )
    assert pattern._answering_settle_id(context, 1) == "b"
    context = SimpleNamespace(
        messages=[
            Message(role="assistant", content="q"),
            Message(role="user", content="x"),
        ]
    )
    assert pattern._answering_settle_id(context, 1) is None


def test_a_settle_is_never_the_top_level_request() -> None:
    from xagent.core.agent.context.enrichment import top_level_user_request

    context = SimpleNamespace(
        messages=[
            Message(role="user", content="Connect my CRM"),
            Message(role="assistant", content="Which one?"),
            _settle("old"),
        ],
        metadata={},
    )
    assert top_level_user_request(context).execution_text == "Connect my CRM"


def _waiting_dag(*interaction_ids: str):
    from xagent.core.agent import DAGPattern, ExecutionContext, ExecutionPlan, PlanStep

    step = PlanStep(id="confirm", task="Confirm")
    step.status = "running"
    pattern = DAGPattern(lambda **_: ExecutionPlan(steps=[step]))
    pattern.status = "waiting_for_user"
    pattern.plan = ExecutionPlan(steps=[step])
    pattern.active_step_id = "confirm"
    pattern.active_step_ids = ["confirm"]
    pattern.active_step_pattern_states = {
        "confirm": {
            "status": "waiting_for_user",
            "waiting_for_user_request": {
                "message": "Connect",
                "requests": [{"interaction_id": i} for i in interaction_ids],
            },
        }
    }
    child = ExecutionContext(execution_id="dag:confirm")
    child.add_user_message("Connect my CRM")
    pattern.active_step_contexts = {"confirm": child.to_dict()}
    pattern.planned_user_message_count = 1
    root = ExecutionContext(execution_id="dag")
    root.add_user_message("Connect my CRM")
    return pattern, root


def _add_settle(context, interaction_id: str) -> None:
    context.add_user_message(
        "[interaction settled]",
        metadata=dict(_settle(interaction_id).metadata),
    )


def test_dag_forwards_a_settle_for_the_waiting_step() -> None:
    pattern, root = _waiting_dag("a", "b")
    _add_settle(root, "b")
    assert pattern._needs_replan(root)
    assert pattern._forward_user_response_to_waiting_step(root) is True
    from xagent.core.agent import ExecutionContext

    child = ExecutionContext.from_dict(pattern.active_step_contexts["confirm"])
    forwarded = child.messages[-1]
    assert forwarded.metadata[SERVER_METADATA_KEY]["interaction_id"] == "b"
    assert forwarded.metadata["response_to_waiting_for_user"]["question"] == "Connect"


def test_dag_ignores_a_settle_for_another_interaction() -> None:
    pattern, root = _waiting_dag("a")
    _add_settle(root, "old")
    assert not pattern._needs_replan(root)
    assert pattern._forward_user_response_to_waiting_step(root) is False
    assert pattern.status == "waiting_for_user"
    assert pattern.planned_user_message_count == 1


def test_dag_ignores_a_settle_while_it_is_not_waiting() -> None:
    pattern, root = _waiting_dag("a")
    pattern.status = "interrupted"
    _add_settle(root, "a")
    assert not pattern._needs_replan(root)


def test_dag_forwards_the_real_answer_and_drops_an_earlier_stale_settle() -> None:
    from xagent.core.agent import ExecutionContext

    pattern, root = _waiting_dag("a")
    _add_settle(root, "old")
    root.add_user_message("blue")
    assert pattern._forward_user_response_to_waiting_step(root) is True
    child = ExecutionContext.from_dict(pattern.active_step_contexts["confirm"])
    contents = [m.content for m in child.messages if m.role == "user"]
    assert contents == ["Connect my CRM", "blue"]
    assert pattern.planned_user_message_count == 3

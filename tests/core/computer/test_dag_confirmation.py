"""A computer confirmation raised inside a DAG step must reach the user.

DAG plan-execute runs every step through a nested ReAct pattern, so the pause
travels: step result -> DAG status -> execution result, and the user's answer
travels back into the step's own context where the grant is issued. This test
pins that path, since a regression would silently turn a confirmation into a
failed step.
"""

from __future__ import annotations

from typing import Any

from xagent.core.agent import ExecutionContext, ReActPattern
from xagent.core.agent.pattern.dag.dag import DAGPattern
from xagent.core.agent.pattern.dag.plan_generator import ExecutionPlan, PlanStep


def _confirmation_request() -> dict[str, Any]:
    return {
        "kind": "tool_waiting_for_user",
        "tool_call_id": "call-1",
        "tool_name": "computer",
        "session_id": "task-1:checkout",
        "message": "Xagent wants to click “Place order” on shop.example.",
        "message_type": "confirmation",
        "interactions": [],
        "confirmation": {
            "confirmation_id": "confirmation-1",
            "kind": "computer_action_confirmation",
            "risk": "elevated",
            "reason": "The control may create an external side effect.",
            "action_indexes": [0],
            "action_summary": "click “Place order”",
            "frame_signature": {
                "active_url": "https://shop.example/checkout",
                "viewport": {"width": 1280, "height": 720, "device_pixel_ratio": 1.0},
                "structure": "digest",
                "targets": [{"element_id": "dom-1"}],
            },
        },
        "message_count": 1,
    }


def test_dag_step_confirmation_round_trips_through_the_user() -> None:
    step = PlanStep(id="checkout", task="Complete the checkout")
    step.status = "running"
    plan = ExecutionPlan(steps=[step])

    waiting_react = ReActPattern()
    waiting_react.status = "waiting_for_user"
    waiting_react.waiting_for_user_request = _confirmation_request()

    pattern = DAGPattern(lambda **_: plan)
    pattern.plan = plan
    pattern.status = "waiting_for_user"
    pattern.active_step_id = "checkout"
    pattern.active_step_ids = ["checkout"]
    pattern.active_step_pattern_states = {"checkout": waiting_react.get_state()}

    child_context = ExecutionContext(execution_id="dag-1:checkout")
    child_context.add_user_message("Buy the item in my cart")
    pattern.active_step_contexts = {"checkout": child_context.to_dict()}
    pattern.planned_user_message_count = 1

    root_context = ExecutionContext(execution_id="dag-1")
    root_context.add_user_message("Buy the item in my cart")
    root_context.add_user_message("computer_action_decision: approve")

    assert pattern._waiting_step_id() == "checkout"
    assert pattern._forward_user_response_to_waiting_step(root_context) is True

    # The step resumes from its own restored ReAct state, which is where the
    # approval is turned into a one-use grant for the next computer call.
    resumed_react = ReActPattern()
    resumed_react.load_state(pattern.active_step_pattern_states["checkout"])
    resumed_context = ExecutionContext.from_dict(
        pattern.active_step_contexts["checkout"]
    )

    assert resumed_react.status == "waiting_for_user"
    resumed_react._apply_tool_confirmation_response(
        waiting_request=dict(resumed_react.waiting_for_user_request or {}),
        response=resumed_context.messages[-1].content,
    )

    grant = resumed_react.approved_tool_confirmation
    assert grant is not None
    assert grant["confirmation_id"] == "confirmation-1"
    assert grant["decision"] == "approve"
    assert grant["session_id"] == "task-1:checkout"
    assert grant["frame_signature"]["active_url"] == "https://shop.example/checkout"


def test_a_grant_survives_a_checkpoint_round_trip() -> None:
    """The frame signature is what makes a resumed approval verifiable.

    Flattening the grant on restore would silently drop it, and the tool would
    then refuse the approved action and ask the user all over again.
    """
    pattern = ReActPattern()
    pattern._apply_tool_confirmation_response(
        waiting_request=_confirmation_request(),
        response="approve",
    )
    granted = pattern.approved_tool_confirmation
    assert granted is not None

    restored = ReActPattern()
    restored.load_state(pattern.get_state())

    assert restored.approved_tool_confirmation == granted
    assert restored.approved_tool_confirmation["frame_signature"]["targets"] == [
        {"element_id": "dom-1"}
    ]


def test_a_denied_grant_is_not_restored() -> None:
    restored = ReActPattern()
    restored.load_state(
        {
            "approved_tool_confirmation": {
                "confirmation_id": "confirmation-1",
                "decision": "deny",
            }
        }
    )

    assert restored.approved_tool_confirmation is None


def test_dag_step_takeover_never_produces_a_grant() -> None:
    """A hand-over asks the user to act themselves; nothing is authorized."""
    waiting_request = _confirmation_request()
    waiting_request["confirmation"]["kind"] = "computer_user_takeover"

    pattern = ReActPattern()
    pattern._apply_tool_confirmation_response(
        waiting_request=waiting_request,
        response="I completed the sensitive step",
    )

    assert pattern.approved_tool_confirmation is None

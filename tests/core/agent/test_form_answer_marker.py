"""``pending_user_response_marker`` gains a ``form`` key only when the
waiting request it was built from is itself flagged as a form (see the
``waiting_for_user_request["form"]`` truth table pinned in
``test_react_form_answer_flag.py``). Every other waiting request produces
the same two-key dict.
"""

from __future__ import annotations

import pytest

from xagent.core.agent.context.enrichment import pending_user_response_marker


@pytest.mark.parametrize(
    "form_value",
    [None, False, "yes", 1, [True]],
    ids=["absent", "false", "truthy-string", "truthy-int", "truthy-list"],
)
def test_form_key_is_carried_over_only_for_the_literal_true_value(
    form_value: object,
) -> None:
    waiting_request = {"message": "Well?", "message_type": "question"}
    if form_value is not None:
        waiting_request["form"] = form_value
    marker = pending_user_response_marker(waiting_request)
    assert marker == {"question": "Well?", "message_type": "question"}


def test_carries_form_true_over() -> None:
    marker = pending_user_response_marker(
        {"message": "Create a shift?", "message_type": "question", "form": True}
    )
    assert marker == {
        "question": "Create a shift?",
        "message_type": "question",
        "form": True,
    }

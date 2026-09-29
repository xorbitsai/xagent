"""Tests for the form-answer-continuation global switch.

``_form_answer_decision`` applies the form-answer-continuation text (see
``ExecutionContext.get_messages_for_llm``) only while one global switch,
``get_form_answer_continuation_enabled``, is on. The switch covers every
model and is on by default: unset means on, and any set value other than
1/true/yes/on (case-insensitive, whitespace ignored) means off, including an
empty value. The getter's own parsing is pinned in tests/core/test_config.py;
these rows pin that the decision follows it.
"""

from __future__ import annotations

import pytest

from xagent.config import FORM_ANSWER_CONTINUATION_ENABLED
from xagent.core.agent import ExecutionContext
from xagent.core.agent.pattern.react.react import (
    FormAnswerDecision,
    _form_answer_decision,
)


def _context_with_form_answer() -> ExecutionContext:
    context = ExecutionContext()
    context.add_user_message("Do the thing")
    context.add_user_message(
        "Morning shift",
        metadata={
            "response_to_waiting_for_user": {
                "question": "What is the shift name?",
                "message_type": "question",
                "form": True,
            }
        },
    )
    return context


@pytest.mark.parametrize(
    ("env_value", "applied"),
    [
        pytest.param(None, True, id="unset-is-on"),
        pytest.param("true", True, id="true"),
        pytest.param(" TRUE ", True, id="true-padded-uppercase"),
        pytest.param("false", False, id="false"),
        pytest.param("0", False, id="zero"),
        pytest.param("", False, id="empty-is-off"),
    ],
)
def test_switch_decides_whether_the_text_applies(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, applied: bool
) -> None:
    if env_value is None:
        monkeypatch.delenv(FORM_ANSWER_CONTINUATION_ENABLED, raising=False)
    else:
        monkeypatch.setenv(FORM_ANSWER_CONTINUATION_ENABLED, env_value)
    decision = _form_answer_decision(
        _context_with_form_answer(), force_final_answer=False
    )
    # The turn key is recorded whether or not the switch is on.
    assert decision == FormAnswerDecision(turn=True, applied=applied)

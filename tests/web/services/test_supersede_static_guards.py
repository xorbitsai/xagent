"""Static guard for the mid-turn WebSocket persistence path.

One check: neither function on the WebSocket mid-turn persistence path
mentions the superseded message-type value at all -- not to write it, and
not to branch on it. That value has exactly one sanctioned writer, the
legacy-question supersede helper in the chat history service, and a
mid-turn write of it would relabel a question that is still live.

The value is recognized in every spelling the shared constant resolver
handles (bare literal, module-scope constant, imported or aliased name,
attribute access), not only as an inline literal, and the check is
async-aware so it keeps working if either function becomes ``async def``.

The tree-wide guards that used to live here -- the single-row delete ban,
the message-type monotonicity sweep and the rollout import ban -- are
repository-level architecture rules rather than facts about this feature,
and now live in ``tests/architecture/test_architecture_guards.py``. This
file imports the constant-resolution helpers from there.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.architecture.test_architecture_guards import (
    _string_constant_bindings,
    _string_values,
)
from xagent.web.api import websocket
from xagent.web.services import chat_history_service

# ---------------------------------------------------------------------------
# The mid-turn WebSocket path never mentions the superseded value at
# all -- not to write it, and not to branch on it. That value belongs
# to this helper's one sanctioned write site.
# ---------------------------------------------------------------------------

MID_TURN_FUNCTIONS = ("_persist_agent_outbound_event", "make_agent_outbound_handler")


def mid_turn_functions_mentioning_superseded_value(
    tree: ast.Module,
) -> tuple[list[str], set[str]]:
    """Return (hit function names, found function names).

    ``found`` is every name out of ``MID_TURN_FUNCTIONS`` that this source
    actually defines -- tracked separately from ``hits`` so a caller can
    tell "neither function mentions the value" apart from "neither
    function exists here anymore" (e.g. after a rename); a check would be
    vacuously green in that second case without this.

    The superseded value is recognized in any of the spellings
    ``_string_constant_bindings`` resolves -- the bare literal, a
    module-scope constant, an imported or aliased
    ``SUPERSEDED_MESSAGE_TYPE``, or ``<module>.SUPERSEDED_MESSAGE_TYPE``
    -- not only as an inline ``ast.Constant``. That function lists the
    spellings it cannot resolve; each of them is a blind spot of this
    guard too.
    """
    bindings = _string_constant_bindings(tree)
    superseded = chat_history_service.SUPERSEDED_MESSAGE_TYPE
    hits: list[str] = []
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in MID_TURN_FUNCTIONS
        ):
            found.add(node.name)
            for inner in ast.walk(node):
                if superseded in _string_values(inner, bindings):
                    hits.append(node.name)
                    break
    return hits, found


def test_mid_turn_websocket_path_never_mentions_the_superseded_value() -> None:
    source = Path(websocket.__file__).read_text(encoding="utf-8")
    hits, found = mid_turn_functions_mentioning_superseded_value(ast.parse(source))

    # Sanity: both tracked functions actually exist in this source -- an
    # empty or partial `found` set would make the hits == [] assertion
    # below vacuous (a renamed or removed function trivially "mentions
    # nothing" because this guard never sees it at all).
    assert found == set(MID_TURN_FUNCTIONS)
    assert hits == [], (
        f"mid-turn websocket path mentions the superseded value "
        f"outside its one sanctioned site: {hits}"
    )


def test_mid_turn_guard_flags_a_literal_mention_in_an_async_def_shape() -> None:
    """The two functions this guard names are plain ``def`` today, but
    the guard must not silently stop working the day either becomes
    ``async def`` -- the same async-aware match the single-row-delete
    guard above already applies to its own function walk."""
    fixture = """
async def _persist_agent_outbound_event(task_id, event):
    message_type = "question_superseded"
    return message_type
"""
    hits, _found = mid_turn_functions_mentioning_superseded_value(ast.parse(fixture))
    assert hits == ["_persist_agent_outbound_event"]


def test_mid_turn_guard_flags_a_literal_mention_in_either_function() -> None:
    fixture = """
def _persist_agent_outbound_event(task_id, event):
    message_type = "question_superseded"
    return message_type


def make_agent_outbound_handler(task_id):
    async def handle_outbound_message(payload):
        return "question_superseded"
    return handle_outbound_message
"""
    hits, _found = mid_turn_functions_mentioning_superseded_value(ast.parse(fixture))
    assert set(hits) == {"_persist_agent_outbound_event", "make_agent_outbound_handler"}


@pytest.mark.parametrize(
    "fixture,expected",
    [
        (
            """
from xagent.web.services.chat_history_service import SUPERSEDED_MESSAGE_TYPE

def _persist_agent_outbound_event(task_id, event):
    message_type = SUPERSEDED_MESSAGE_TYPE
    return message_type
""",
            ["_persist_agent_outbound_event"],
        ),
        (
            """
from xagent.web.services.chat_history_service import SUPERSEDED_MESSAGE_TYPE as SUP

def make_agent_outbound_handler(task_id):
    async def handle_outbound_message(payload):
        return SUP
    return handle_outbound_message
""",
            ["make_agent_outbound_handler"],
        ),
    ],
    ids=["imported", "aliased-import"],
)
def test_mid_turn_guard_flags_a_constant_spelled_mention(
    fixture: str, expected: list[str]
) -> None:
    hits, _found = mid_turn_functions_mentioning_superseded_value(ast.parse(fixture))
    assert hits == expected

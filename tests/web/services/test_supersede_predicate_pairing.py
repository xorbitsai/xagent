"""Static guard: the supersede WHERE predicate matches the reader's first
pass, byte-for-byte, as a set.

``supersede_legacy_question_rows`` and ``get_latest_waiting_question`` must
agree on which rows count as "a still-pending assistant question" -- the
reader picks the newest one to show the user, the writer collapses the
whole set. If either side gains, loses, or changes a predicate leg without
the other following, they silently start disagreeing about which rows are
"pending", and that drift has no other guard.

Scope: only the reader's *first* ``.filter(...)`` call -- the one that runs
before its ``.order_by(...)`` -- is compared. ``get_latest_waiting_question``
has exactly one ``.filter(...)`` call today; a future second pass (e.g. a
widened read that also considers already-superseded rows) is explicitly
out of scope for this pairing, by design -- it would encode a different
question ("what should the reader show") than this predicate answers
("what does the writer collapse").

Extraction is intentionally narrow: only top-level ``Attribute == <value>``
comparisons passed as positional args to a ``.filter(...)`` call are read,
matching the only shape either function uses today. A predicate spelled
any other way (``.filter_by(...)``, a compound ``and_()``, a comparison
method call such as ``.isnot(None)``) is invisible to this walk on
whichever side carries it -- that is a real blind spot, not a caught case:
if one side alone grows a non-``==``-shaped leg, this guard stays green.
It only catches drift in the shape both functions are written in.
"""

from __future__ import annotations

import ast
from pathlib import Path

from xagent.web.services import chat_history_service

READER_FUNCTION = "get_latest_waiting_question"
WRITER_FUNCTION = "supersede_legacy_question_rows"


def _predicate_value(node: ast.expr) -> str:
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return f"name:{node.id}"
    return ast.dump(node)


def _first_filter_call(func: ast.FunctionDef) -> ast.Call | None:
    calls = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "filter"
    ]
    if not calls:
        return None
    return min(calls, key=lambda call: (call.lineno, call.col_offset))


def _filter_predicate_set(func: ast.FunctionDef) -> frozenset[tuple[str, str]]:
    call = _first_filter_call(func)
    if call is None:
        return frozenset()
    predicates: set[tuple[str, str]] = set()
    for arg in call.args:
        if (
            isinstance(arg, ast.Compare)
            and len(arg.ops) == 1
            and isinstance(arg.ops[0], ast.Eq)
            and isinstance(arg.left, ast.Attribute)
            and len(arg.comparators) == 1
        ):
            predicates.add((arg.left.attr, _predicate_value(arg.comparators[0])))
    return frozenset(predicates)


def predicate_sets(source: str) -> tuple[frozenset, frozenset]:
    """Return (reader's first-pass predicate set, writer's predicate set)
    for a module source string containing both functions by name."""
    tree = ast.parse(source)
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    reader = functions.get(READER_FUNCTION)
    writer = functions.get(WRITER_FUNCTION)
    reader_predicates = (
        _filter_predicate_set(reader) if reader is not None else frozenset()
    )
    writer_predicates = (
        _filter_predicate_set(writer) if writer is not None else frozenset()
    )
    return reader_predicates, writer_predicates


def test_supersede_predicate_matches_the_readers_first_pass_filter() -> None:
    source = Path(chat_history_service.__file__).read_text()
    reader_predicates, writer_predicates = predicate_sets(source)

    # Sanity: both sides actually extracted something -- an empty set on
    # either side would make the equality assertion below vacuous.
    assert reader_predicates
    assert writer_predicates
    assert reader_predicates == writer_predicates


def test_pairing_flags_a_reader_that_drifts_ahead_of_the_writer() -> None:
    """Positive control: the reader gains a fourth predicate leg the
    writer does not follow -- this is exactly the drift the guard exists
    to catch."""
    source = """
def get_latest_waiting_question(db, task_id):
    return (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "assistant",
            TaskChatMessage.message_type == "question",
            TaskChatMessage.turn_id == None,
        )
        .order_by(TaskChatMessage.id.desc())
        .first()
    )


def supersede_legacy_question_rows(db, *, task_id):
    return (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "assistant",
            TaskChatMessage.message_type == "question",
        )
        .update({TaskChatMessage.message_type: "question_superseded"},
                synchronize_session=False)
    )
"""
    reader_predicates, writer_predicates = predicate_sets(source)
    assert reader_predicates != writer_predicates


def test_pairing_flags_a_writer_that_drifts_ahead_of_the_reader() -> None:
    """Positive control, the other direction: the writer narrows its own
    predicate without the reader following -- unilateral drift on either
    side must flag."""
    source = """
def get_latest_waiting_question(db, task_id):
    return (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.role == "assistant",
            TaskChatMessage.message_type == "question",
        )
        .order_by(TaskChatMessage.id.desc())
        .first()
    )


def supersede_legacy_question_rows(db, *, task_id):
    return (
        db.query(TaskChatMessage)
        .filter(
            TaskChatMessage.task_id == task_id,
            TaskChatMessage.message_type == "question",
        )
        .update({TaskChatMessage.message_type: "question_superseded"},
                synchronize_session=False)
    )
"""
    reader_predicates, writer_predicates = predicate_sets(source)
    assert reader_predicates != writer_predicates

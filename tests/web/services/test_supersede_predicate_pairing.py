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
The same applies to statement-split accumulation (``q = q.filter(...)``
across several statements): each re-assignment starts a new chain at a
new position, so those legs fall outside the first chain this guard
compares. What that costs is the opposite of what it sounds like. A side
that *gains* a leg this way -- reader or writer, either one alone -- keeps
an unchanged extracted set, so the two sets still match and this guard
stays green while the real predicates have diverged: single-side drift of
that shape goes silent. A side that merely *moves* an existing leg out of
the first chain, changing nothing about which rows match, drops it from
the extracted set and turns this guard red -- a false positive on a
behavior-preserving refactor. Of the two statement-split shapes, this
guard is blind to the one that changes behavior and noisy about the one
that does not.
It only catches drift in the shape both functions are written in.

Known false-positive mode: a pure identifier rename on either side (e.g.
renaming the ``task_id`` parameter to ``target_task_id`` with no change
in behavior) flips this guard red, because comparison values are read as
literal names, not resolved through binding -- a cosmetic rename reads
the same as a real predicate change.
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


def _first_filter_chain_calls(func: ast.FunctionDef) -> list[ast.Call]:
    """Return every ``.filter(...)`` call belonging to the *first* filter
    chain in ``func``.

    An ``ast.Call`` node's ``(lineno, col_offset)`` is the position of
    the whole call expression, which for a chained call is the position
    of the receiver the chain starts from -- so every ``.filter(...)``
    call in one ``db.query(...).filter(a).filter(b)`` chain shares the
    *same* ``(lineno, col_offset)``, no matter how many legs it has.
    Picking a single node with ``min()`` over all filter calls therefore
    does not select "the earliest filter call": on a tie it falls back to
    ``ast.walk``'s BFS order, which visits the outermost node first --
    and the outermost node in a chain is the *last*-written ``.filter()``
    call, not the first. That silently drops every earlier leg.

    Grouping by the shared ``(lineno, col_offset)`` key and returning
    every call in the minimal-key group recovers the whole chain, while
    a second, later filter chain in the same function -- which starts
    from a different receiver and so carries a different, larger key --
    is correctly excluded from the group.
    """
    calls = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "filter"
    ]
    if not calls:
        return []
    min_key = min((call.lineno, call.col_offset) for call in calls)
    return [call for call in calls if (call.lineno, call.col_offset) == min_key]


def _filter_predicate_set(func: ast.FunctionDef) -> frozenset[tuple[str, str]]:
    predicates: set[tuple[str, str]] = set()
    for call in _first_filter_chain_calls(func):
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


def test_chained_filter_calls_union_into_every_leg_of_the_first_chain() -> None:
    """A reader written as ``.filter(a).filter(b).filter(c)`` -- three
    separate ``.filter()`` calls chained instead of one call with three
    positional args -- must still yield the union of all three legs.
    This is the exact shape a bare ``min()`` over filter-call nodes gets
    wrong: every call in the chain shares the same
    ``(lineno, col_offset)``, so tie-breaking by AST walk order picks
    only the outermost (last-written) call and silently drops the
    earlier two."""
    source = """
def get_latest_waiting_question(db, task_id):
    return (
        db.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == task_id)
        .filter(TaskChatMessage.role == "assistant")
        .filter(TaskChatMessage.message_type == "question")
        .order_by(TaskChatMessage.id.desc())
        .first()
    )
"""
    reader_predicates, _writer_predicates = predicate_sets(source)
    assert reader_predicates == frozenset(
        {
            ("task_id", "name:task_id"),
            ("role", "'assistant'"),
            ("message_type", "'question'"),
        }
    )


def test_chained_filter_calls_ignore_a_second_later_chain() -> None:
    """A second, positionally distinct ``.filter(...)`` chain later in the
    same function -- e.g. a second, unrelated query -- must not
    contribute any predicates. Only the first chain's legs count; this
    is what keeps the guard's scope to "the reader's first pass" (see
    the module docstring) even once chain grouping recovers multi-leg
    chains."""
    source = """
def get_latest_waiting_question(db, task_id):
    first = (
        db.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == task_id)
        .filter(TaskChatMessage.role == "assistant")
        .first()
    )
    second = (
        db.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == task_id)
        .filter(TaskChatMessage.role == "user")
        .first()
    )
    return first
"""
    reader_predicates, _writer_predicates = predicate_sets(source)
    assert reader_predicates == frozenset(
        {
            ("task_id", "name:task_id"),
            ("role", "'assistant'"),
        }
    )


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

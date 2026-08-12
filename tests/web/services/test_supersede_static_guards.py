"""Static guards for the legacy-question supersede helper.

Four independent, source-level checks, kept in one file because each is a
small, focused test rather than a full behavior suite:

* single-row delete ban on ``TaskChatMessage`` (the static half of the
  transcript-invariance test) -- the only sanctioned write shapes are the bulk
  ``.update()`` in ``mark_user_message_delivery`` /
  ``supersede_legacy_question_rows`` and the bulk account-purge
  ``.delete()`` in ``admin_users.py``; a single-row delete would silently
  destroy transcript history instead of relabeling it.
* monotonicity: nothing in the tree writes ``message_type`` back from
  ``"question_superseded"`` to ``"question"``.
* import ban: ``chat_history_service.py`` does not import anything from
  the native-rollout module.
* mid-turn marker-write ban: the WebSocket mid-turn persistence path never
  writes the ``"question_superseded"`` literal itself -- that is this
  helper's only sanctioned write site.

Each check operates on parsed source text (``ast.parse``), not on a live
import of every module in the tree -- module-level side effects on import
are out of scope for what these guards need to prove.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from xagent.web.api import admin_users, websocket
from xagent.web.services import chat_history_service

SRC_ROOT = Path(chat_history_service.__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def parsed_src_files() -> tuple[tuple[str, str, ast.Module], ...]:
    """Every ``.py`` file under ``src/xagent``, read and parsed once for
    this module: ``(path relative to SRC_ROOT, source text, parsed tree)``.

    Two guards below sweep the whole tree. Each used to run its own
    ``rglob`` + ``read_text``. Module-scoped rather than a module-level
    cache so pytest owns the lifetime and no state survives the module;
    the guards only read these trees and never mutate them.

    One consequence of reading and parsing everything up front: a file
    under ``src/xagent`` that is not valid UTF-8, or that this Python
    cannot parse, now fails this fixture and takes both tree-wide guards
    down with it rather than being skipped by whichever prefilter used to
    miss it. That is deliberate. A guard whose answer is "no violations"
    because it silently could not read part of the tree is worse than one
    that fails loudly, and every file in the tree parses today under the
    Python this suite runs on. It does mean an unrelated syntax error
    shows up here as well as wherever else it breaks.
    """
    parsed: list[tuple[str, str, ast.Module]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        parsed.append((str(path.relative_to(SRC_ROOT)), source, ast.parse(source)))
    return tuple(parsed)


def test_src_root_is_resolvable() -> None:
    """``SRC_ROOT`` is computed at import time and every other guard in
    this file depends on it pointing at the right directory. Pinning the
    check in a test function -- rather than a module-scope ``assert`` --
    means a wrong path fails as one ordinary test failure instead of an
    uncollectable module that takes the rest of this file's guards down
    with it."""
    assert SRC_ROOT.name == "xagent", SRC_ROOT


def _mentions_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


# ---------------------------------------------------------------------------
# No single-row delete of a TaskChatMessage row anywhere in src/.
# ---------------------------------------------------------------------------


def _scope_partition(scope: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    """Split everything lexically inside ``scope`` into the nodes that
    belong to its own scope and the nested definitions that start a new
    one (``def`` / ``async def`` / ``class``).

    ``ast.walk`` makes no such split, which is why the caller below does
    not use it: a name bound inside a nested ``def`` would otherwise be
    read as if the enclosing function had bound it, and the same
    ``db.delete(...)`` call would be visited once per enclosing scope.
    A ``lambda`` is deliberately *not* a boundary here -- nothing else
    visits a lambda body, so treating it as one would drop
    ``lambda: db.delete(msg)`` from the scan entirely.

    The boundary tuple is doing two jobs at once: it says where one scope
    ends, and the ``nested`` list it fills is the only thing the caller
    recurses into. Dropping a node type from it therefore does not widen
    the scan, it removes those definitions from the walk altogether.
    """
    own: list[ast.AST] = []
    nested: list[ast.AST] = []
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nested.append(node)
            continue
        own.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return own, nested


def _names_assigned_from_task_chat_message(own_nodes: list[ast.AST]) -> set[str]:
    """Names bound *in one scope* to an expression that mentions
    ``TaskChatMessage`` -- a query result, a ``.get()`` lookup, a direct
    construction, or a ``for`` loop iterating such a query. Used to trace
    a ``db.delete(<name>)`` call's argument back to a ``TaskChatMessage``
    origin, the same by-name (not full dataflow) resolution style as the
    checkpoint-pointer pairing guard uses for its ``**`` carriers.

    Takes the caller's already-partitioned own-scope nodes rather than a
    function node, so a binding made inside a nested definition stays in
    that definition's scope.
    """
    names: set[str] = set()
    for node in own_nodes:
        if isinstance(node, ast.Assign) and _mentions_name(
            node.value, "TaskChatMessage"
        ):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.For) and _mentions_name(node.iter, "TaskChatMessage"):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def single_row_task_chat_message_deletes(tree: ast.Module) -> list[tuple[int, str]]:
    """Instance-level ``<session>.delete(<name>)`` calls whose argument was
    bound to a ``TaskChatMessage``-mentioning expression in the same
    function, or in a function that one encloses.

    A bulk ``db.query(TaskChatMessage).filter(...).delete(...)`` chain is
    never a candidate here at all: its ``.delete(...)`` receiver is
    itself a ``Call`` (the ``.filter(...)`` in the chain), not a bare
    session name, so it is excluded before the name-tracing step even
    runs -- this is what keeps the multi-model purge loop in
    ``admin_users.py`` (``db.query(model).filter(...).delete(...)`` for
    several unrelated models sharing a function with the real
    ``TaskChatMessage`` bulk delete) from being misread as a per-row
    delete.

    Scopes are walked explicitly rather than with ``ast.walk`` so each
    ``delete`` is attributed to the function that lexically contains it.
    Names flow inward only: a nested function sees what its enclosing
    functions bound (a real closure read), and a binding made inside a
    nested function does not leak back out to the enclosing one.

    Known blind spot: an instance delete whose argument arrives as an
    already-typed function parameter (never locally assigned) is
    invisible to this by-name trace.
    """
    findings: list[tuple[int, str]] = []

    def visit(scope: ast.AST, inherited: frozenset[str]) -> None:
        own, nested = _scope_partition(scope)
        tied = inherited
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tied = inherited | _names_assigned_from_task_chat_message(own)
            for call in own:
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "delete"
                ):
                    continue
                if isinstance(call.func.value, ast.Call):
                    continue  # bulk query-chain delete, not instance-level
                if (
                    call.args
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id in tied
                ):
                    findings.append((call.lineno, scope.name))
        for child in nested:
            visit(child, frozenset(tied))

    visit(tree, frozenset())
    return findings


def test_no_single_row_delete_of_task_chat_message_rows_anywhere_in_src(
    parsed_src_files,
) -> None:
    findings: list[tuple[str, str]] = []
    for rel_path, source, tree in parsed_src_files:
        # Exactly implied by what the walk can match: every finding needs
        # an ``ast.Name`` whose id is ``TaskChatMessage``, which cannot
        # exist unless that name appears verbatim in the text. A skip on
        # ``".delete("`` would not be exactly implied -- ``ast`` does not
        # care about the whitespace between the attribute and the paren --
        # so it is not used.
        if "TaskChatMessage" not in source:
            continue
        for _lineno, func_name in single_row_task_chat_message_deletes(tree):
            findings.append((rel_path, func_name))
    assert findings == [], (
        "single-row delete(s) of a TaskChatMessage instance found outside "
        f"the sanctioned bulk-update paths: {findings}"
    )


def test_admin_users_bulk_purge_is_not_flagged_as_a_single_row_delete() -> None:
    """Positive baseline: the multi-model purge loop and the real
    ``TaskChatMessage`` bulk delete share a function, and neither must
    trip the guard -- proving the bulk-chain exclusion actually applies
    to this repo's real shape, not just to a fixture."""
    source = Path(admin_users.__file__).read_text(encoding="utf-8")
    findings = single_row_task_chat_message_deletes(ast.parse(source))
    assert findings == [], (
        "admin_users.py's bulk purge was misread as a single-row "
        f"TaskChatMessage delete: {findings}"
    )


def test_single_row_delete_guard_flags_a_row_level_delete() -> None:
    fixture = """
def purge_one(db, message_id):
    message = db.query(TaskChatMessage).filter(TaskChatMessage.id == message_id).first()
    db.delete(message)
"""
    findings = single_row_task_chat_message_deletes(ast.parse(fixture))
    assert [func_name for _lineno, func_name in findings] == ["purge_one"]


def test_delete_guard_does_not_attribute_a_nested_binding_to_the_outer_function() -> (
    None
):
    """A name bound inside a nested ``def`` is not the outer function's.
    Before the scope split, ``ast.walk`` made it look like one and the
    outer ``db.delete(msg)`` -- which refers to nothing of the sort --
    was reported."""
    fixture = """
def outer(db):
    def inner(db):
        msg = db.query(TaskChatMessage).first()
        return msg
    db.delete(msg)
"""
    assert single_row_task_chat_message_deletes(ast.parse(fixture)) == []


def test_delete_guard_follows_a_binding_a_nested_function_closes_over() -> None:
    """Names still flow inward: the inner function really can read the
    outer function's ``msg``, so the delete is a real finding and must be
    reported under the inner function's name."""
    fixture = """
def outer(db):
    msg = db.query(TaskChatMessage).first()
    def inner():
        db.delete(msg)
    return inner
"""
    assert [
        name
        for _lineno, name in single_row_task_chat_message_deletes(ast.parse(fixture))
    ] == ["inner"]


def test_delete_guard_reports_a_nested_definition_under_its_own_name() -> None:
    fixture = """
def outer(db):
    def inner(db):
        msg = db.query(TaskChatMessage).first()
        db.delete(msg)
    return inner
"""
    assert [
        name
        for _lineno, name in single_row_task_chat_message_deletes(ast.parse(fixture))
    ] == ["inner"]


# ---------------------------------------------------------------------------
# message_type never writes 'question_superseded' -> 'question' (monotonicity).
# ---------------------------------------------------------------------------


def _writes_message_type_to(node: ast.Call, value: str) -> bool:
    for arg in node.args:
        if isinstance(arg, ast.Dict):
            for key, val in zip(arg.keys, arg.values):
                key_name = (
                    key.attr
                    if isinstance(key, ast.Attribute)
                    else (key.value if isinstance(key, ast.Constant) else None)
                )
                if (
                    key_name == "message_type"
                    and isinstance(val, ast.Constant)
                    and val.value == value
                ):
                    return True
    for kw in node.keywords:
        if (
            kw.arg == "message_type"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == value
        ):
            return True
    return False


def _filters_message_type_equal(node: ast.Call, value: str) -> bool:
    for arg in node.args:
        if (
            isinstance(arg, ast.Compare)
            and len(arg.ops) == 1
            and isinstance(arg.ops[0], ast.Eq)
            and isinstance(arg.left, ast.Attribute)
            and arg.left.attr == "message_type"
            and len(arg.comparators) == 1
            and isinstance(arg.comparators[0], ast.Constant)
            and arg.comparators[0].value == value
        ):
            return True
    return False


def reverse_supersede_writes(tree: ast.Module) -> list[int]:
    """Statement sites that write ``message_type='question'`` on rows
    selected via ``message_type == 'question_superseded'`` -- the mirror
    image of what ``supersede_legacy_question_rows`` does, and the shape
    a hypothetical revert helper built by copy-and-flip would take.

    Two mutation carrier forms are checked, matching the two forms this
    repo's ORM bulk writes actually use: a dict-literal passed to
    ``.update(...)``/``.values(...)``, and a ``message_type=`` keyword on
    the same calls.

    Five shapes are known blind spots, disclosed rather than caught,
    matching the disclosure style the single-row-delete guard above in
    this file already uses for its own known gaps:

    * per-row attribute assignment -- ``msg.message_type = "question"``
      on an ORM instance fetched via ``message_type == "question_superseded"``
      -- invisible because this walk only inspects ``.update()``/
      ``.values()`` calls, not plain attribute assignment.
    * a filter and its mutating call split across two statements --
      ``q = db.query(TaskChatMessage).filter(...)`` bound to a name in
      one statement, ``q.update(...)`` called on that name in a later
      one -- invisible because the walk only recognizes a ``.filter(...)``
      call inline in the same receiver chain as the mutating call, the
      same by-name (not full dataflow) limit the delete guard above notes
      for its own carrier tracing.
    * a constant-composed literal -- a ``Name`` resolved back to a
      constant, e.g. ``.update({TaskChatMessage.message_type:
      QUESTION_MESSAGE_TYPE})`` -- invisible because the walk only
      recognizes an inline ``ast.Constant`` value, not a ``Name``. This
      is not a hypothetical shape: ``chat_history_service.py`` exports
      both ``QUESTION_MESSAGE_TYPE`` and ``SUPERSEDED_MESSAGE_TYPE`` as
      public constants, so a revert helper written with the same style
      the forward write already uses -- ``.filter(message_type ==
      SUPERSEDED_MESSAGE_TYPE).update({message_type:
      QUESTION_MESSAGE_TYPE})`` -- returns zero findings from this walk
      (probe-verified). Disclosed rather than closed only because no
      such revert helper exists yet to trigger it, not because closing
      it would be expensive.
    * a negated comparison -- ``.filter(message_type != "question")``
      selecting everything that is not already ``"question"`` -- invisible
      because ``_filters_message_type_equal`` only matches ``ast.Eq``, not
      ``ast.NotEq``.
    * a membership test -- ``.filter(message_type.in_([...]))`` -- invisible
      because ``_filters_message_type_equal`` only matches an ``ast.Compare``
      node, and ``.in_(...)`` parses as an ``ast.Call``.

    Closing the remaining shapes would need the same dataflow-sensitive
    analysis the checkpoint-pointer pairing guard explicitly declines for
    its own ``**`` carriers, for the same cost/benefit reason: a
    materially more expensive walk to close gaps with no live instance
    in this repo today.

    Separately: the caller of this function (below) only scans a source
    file at all if the literal substring ``"question_superseded"``
    appears in its text. A file that imports and uses only
    ``SUPERSEDED_MESSAGE_TYPE`` / ``QUESTION_MESSAGE_TYPE`` by name, with
    that literal string appearing nowhere in its own source, is skipped
    by that prefilter entirely and never reaches this walk
    (probe-verified).
    """
    findings: list[int] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"update", "values"}
        ):
            continue
        if not _writes_message_type_to(node, "question"):
            continue
        receiver = node.func.value
        reads_superseded = False
        while isinstance(receiver, ast.Call):
            if (
                isinstance(receiver.func, ast.Attribute)
                and receiver.func.attr == "filter"
                and _filters_message_type_equal(receiver, "question_superseded")
            ):
                reads_superseded = True
                break
            receiver = (
                receiver.func.value
                if isinstance(receiver.func, ast.Attribute)
                else None
            )
        if reads_superseded:
            findings.append(node.lineno)
    return findings


def test_nothing_writes_message_type_back_to_question(parsed_src_files) -> None:
    """The ``"question_superseded" not in source`` line below is a
    prefilter, not a full scan: a file using only the imported
    ``SUPERSEDED_MESSAGE_TYPE`` / ``QUESTION_MESSAGE_TYPE`` names, with
    that literal string appearing nowhere in its own text, is skipped
    and never reaches ``reverse_supersede_writes`` (see that function's
    docstring for the probe that confirms this)."""
    findings: list[tuple[str, list[int]]] = []
    for rel_path, source, tree in parsed_src_files:
        if "question_superseded" not in source:
            continue
        hits = reverse_supersede_writes(tree)
        if hits:
            findings.append((rel_path, hits))
    assert findings == [], (
        "message_type written back from question_superseded to question "
        f"(monotonicity violation): {findings}"
    )


def test_monotonicity_guard_flags_a_reverse_assignment() -> None:
    fixture = """
def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.task_id == task_id,
        TaskChatMessage.message_type == "question_superseded",
    ).update({TaskChatMessage.message_type: "question"}, synchronize_session=False)
"""
    assert len(reverse_supersede_writes(ast.parse(fixture))) == 1


# ---------------------------------------------------------------------------
# chat_history_service.py imports nothing from the native-rollout module.
#
# This guard exists so chat_history_service.py never becomes a module
# that imports rollout controls: the supersede helper is unconditional by
# contract and must never branch on rollout mode. The rollout module
# (``interaction_rollout.py``) exists in this repository --
# this guard is what keeps this service module from ever importing it,
# and it now flags real code, not a hypothetical future import.
# ---------------------------------------------------------------------------

_BANNED_ROLLOUT_NAMES = frozenset(
    {
        "interaction_rollout",
        "evaluate_native_publication",
        "get_interaction_rollout_policy",
        "InteractionRolloutPolicy",
    }
)


def banned_rollout_imports(tree: ast.Module) -> list[str]:
    """Banned identifiers reachable through any import statement.

    Two carrier shapes for the module itself: a dotted module *path*
    containing ``interaction_rollout`` (``import ...interaction_rollout``,
    ``from ...interaction_rollout import X``) -- checked against
    ``alias.name``/``module`` split on ``.`` -- and the module imported as
    a bare *name* (``from . import interaction_rollout``,
    ``from ..services import interaction_rollout``), where
    ``interaction_rollout`` is the imported name itself rather than part
    of the dotted path. The set membership check on each ``ImportFrom``
    alias catches that second shape because ``"interaction_rollout"`` is
    itself in ``_BANNED_ROLLOUT_NAMES``.
    """
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "interaction_rollout" in alias.name.split("."):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "interaction_rollout" in module.split("."):
                hits.append(module)
            for alias in node.names:
                if alias.name in _BANNED_ROLLOUT_NAMES:
                    hits.append(f"{module}.{alias.name}" if module else alias.name)
    return hits


def test_chat_history_service_imports_nothing_from_native_rollout() -> None:
    source = Path(chat_history_service.__file__).read_text(encoding="utf-8")
    hits = banned_rollout_imports(ast.parse(source))
    assert hits == [], (
        f"chat_history_service.py imports from the banned native-rollout "
        f"module (supersede must stay unconditional): {hits}"
    )


def test_rollout_import_guard_flags_a_module_import() -> None:
    fixture = (
        "from ...services.interaction_rollout import evaluate_native_publication\n"
    )
    hits = banned_rollout_imports(ast.parse(fixture))
    assert hits


def test_rollout_import_guard_flags_the_policy_dataclass_by_name() -> None:
    fixture = "from xagent.web.services.policy import InteractionRolloutPolicy\n"
    hits = banned_rollout_imports(ast.parse(fixture))
    assert hits == ["xagent.web.services.policy.InteractionRolloutPolicy"]


def test_rollout_import_guard_flags_the_module_imported_as_a_bare_name() -> None:
    """The module itself, imported relatively and named directly rather
    than appearing inside a dotted path: ``from . import interaction_rollout``
    and ``from ..services import interaction_rollout``."""
    same_package = "from . import interaction_rollout\n"
    sibling_package = "from ..services import interaction_rollout\n"
    assert banned_rollout_imports(ast.parse(same_package))
    assert banned_rollout_imports(ast.parse(sibling_package))


def test_rollout_import_guard_ignores_an_unrelated_name_containing_the_substring() -> (
    None
):
    """Negative control: a name that merely *contains*
    ``interaction_rollout`` as a substring, rather than being exactly
    that module or one of the banned symbols, must not be flagged --
    otherwise a coincidentally similar identifier becomes unusable."""
    fixture = "from xagent.web.services import interaction_rollout_helper\n"
    hits = banned_rollout_imports(ast.parse(fixture))
    assert hits == [], (
        f"rollout import guard false-positived on an unrelated substring match: {hits}"
    )


# ---------------------------------------------------------------------------
# The mid-turn WebSocket path never writes the "question_superseded"
# literal itself -- that is this helper's only sanctioned write site.
# ---------------------------------------------------------------------------

MID_TURN_FUNCTIONS = ("_persist_agent_outbound_event", "make_agent_outbound_handler")


def mid_turn_functions_writing_superseded_literal(
    tree: ast.Module,
) -> tuple[list[str], set[str]]:
    """Return (hit function names, found function names).

    ``found`` is every name out of ``MID_TURN_FUNCTIONS`` that this source
    actually defines -- tracked separately from ``hits`` so a caller can
    tell "neither function writes the literal" apart from "neither
    function exists here anymore" (e.g. after a rename); a check would be
    vacuously green in that second case without this.

    Known blind spot: this only recognizes an inline ``ast.Constant``
    equal to ``"question_superseded"``, not a ``Name`` resolved back to
    one. ``chat_history_service.py`` exports ``SUPERSEDED_MESSAGE_TYPE``
    as a public constant, so a mid-turn write spelled as
    ``message_type = SUPERSEDED_MESSAGE_TYPE`` instead of the bare
    literal produces zero hits from this walk even though ``found``
    correctly reports the function as present -- the vacuousness check
    above does not cover this gap (probe-verified).
    """
    hits: list[str] = []
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in MID_TURN_FUNCTIONS
        ):
            found.add(node.name)
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Constant)
                    and inner.value == "question_superseded"
                ):
                    hits.append(node.name)
                    break
    return hits, found


def test_mid_turn_websocket_path_never_writes_the_superseded_literal() -> None:
    source = Path(websocket.__file__).read_text(encoding="utf-8")
    hits, found = mid_turn_functions_writing_superseded_literal(ast.parse(source))

    # Sanity: both tracked functions actually exist in this source -- an
    # empty or partial `found` set would make the hits == [] assertion
    # below vacuous (a renamed or removed function trivially "writes
    # nothing" because this guard never sees it at all).
    assert found == set(MID_TURN_FUNCTIONS)
    assert hits == [], (
        f"mid-turn websocket path writes the question_superseded literal "
        f"outside its one sanctioned site: {hits}"
    )


def test_mid_turn_guard_flags_a_literal_write_in_an_async_def_shape() -> None:
    """The two functions this guard names are plain ``def`` today, but
    the guard must not silently stop working the day either becomes
    ``async def`` -- the same async-aware match the single-row-delete
    guard above already applies to its own function walk."""
    fixture = """
async def _persist_agent_outbound_event(task_id, event):
    message_type = "question_superseded"
    return message_type
"""
    hits, _found = mid_turn_functions_writing_superseded_literal(ast.parse(fixture))
    assert hits == ["_persist_agent_outbound_event"]


def test_mid_turn_guard_flags_a_literal_write_in_either_function() -> None:
    fixture = """
def _persist_agent_outbound_event(task_id, event):
    message_type = "question_superseded"
    return message_type


def make_agent_outbound_handler(task_id):
    async def handle_outbound_message(payload):
        return "question_superseded"
    return handle_outbound_message
"""
    hits, _found = mid_turn_functions_writing_superseded_literal(ast.parse(fixture))
    assert set(hits) == {"_persist_agent_outbound_event", "make_agent_outbound_handler"}

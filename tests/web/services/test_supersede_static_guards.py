"""Static guards for the legacy-question supersede helper.

Four independent, source-level checks, kept in one file because each is a
single reduced-density cell rather than a full behavior suite:

* single-row delete ban on ``TaskChatMessage`` (the transcript-invariance
  cell's static half) -- the only sanctioned write shapes are the bulk
  ``.update()`` in ``mark_user_message_delivery`` /
  ``supersede_legacy_question_rows`` and the bulk account-purge
  ``.delete()`` in ``admin_users.py``; a single-row delete would silently
  destroy transcript history instead of relabeling it.
* monotonicity: nothing in the tree writes ``message_type`` back from
  ``"question_superseded"`` to ``"question"``.
* import ban: ``chat_history_service.py`` does not import anything from
  the not-yet-existing native-rollout module.
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

from xagent.web.api import admin_users, websocket
from xagent.web.services import chat_history_service

SRC_ROOT = Path(chat_history_service.__file__).resolve().parents[2]
assert SRC_ROOT.name == "xagent", SRC_ROOT


def _iter_source_files():
    for path in SRC_ROOT.rglob("*.py"):
        yield path


def _mentions_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


# ---------------------------------------------------------------------------
# T-S-10 static half: no single-row delete of a TaskChatMessage row.
# ---------------------------------------------------------------------------


def _delete_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "delete"
    ]


def _names_assigned_from_task_chat_message(func: ast.AST) -> set[str]:
    """Local names this function ever binds to an expression that mentions
    ``TaskChatMessage`` -- a query result, a ``.get()`` lookup, a direct
    construction, or a ``for`` loop iterating such a query. Used to trace
    a ``db.delete(<name>)`` call's argument back to a ``TaskChatMessage``
    origin within the same function, the same by-name (not full
    dataflow) resolution style as the checkpoint-pointer pairing guard
    uses for its ``**`` carriers."""
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and _mentions_name(
            node.value, "TaskChatMessage"
        ):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.For) and _mentions_name(node.iter, "TaskChatMessage"):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def single_row_task_chat_message_deletes(source: str) -> list[tuple[int, str]]:
    """Instance-level ``<session>.delete(<name>)`` calls, in this file,
    whose argument was bound to a ``TaskChatMessage``-mentioning
    expression earlier in the same function.

    A bulk ``db.query(TaskChatMessage).filter(...).delete(...)`` chain is
    never a candidate here at all: its ``.delete(...)`` receiver is
    itself a ``Call`` (the ``.filter(...)`` in the chain), not a bare
    session name, so it is excluded before the name-tracing step even
    runs -- this is what keeps the multi-model purge loop in
    ``admin_users.py`` (``db.query(model).filter(...).delete(...)`` for
    several unrelated models sharing a function with the real
    ``TaskChatMessage`` bulk delete) from being misread as a per-row
    delete. Known blind spot: an instance delete whose argument arrives
    as an already-typed function parameter (never locally assigned) is
    invisible to this by-name trace.
    """
    tree = ast.parse(source)
    findings: list[tuple[int, str]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tied_names = _names_assigned_from_task_chat_message(func)
        if not tied_names:
            continue
        for call in _delete_calls(func):
            if isinstance(call.func.value, ast.Call):
                continue  # bulk query-chain delete, not instance-level
            if (
                call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in tied_names
            ):
                findings.append((call.lineno, func.name))
    return findings


def test_no_single_row_delete_of_task_chat_message_rows_anywhere_in_src() -> None:
    findings: list[tuple[str, str]] = []
    for path in _iter_source_files():
        source = path.read_text()
        if "TaskChatMessage" not in source or ".delete(" not in source:
            continue
        for _lineno, func_name in single_row_task_chat_message_deletes(source):
            findings.append((str(path.relative_to(SRC_ROOT)), func_name))
    assert findings == []


def test_admin_users_bulk_purge_is_not_flagged_as_a_single_row_delete() -> None:
    """Positive baseline: the multi-model purge loop and the real
    ``TaskChatMessage`` bulk delete share a function, and neither must
    trip the guard -- proving the bulk-chain exclusion actually applies
    to this repo's real shape, not just to a fixture."""
    source = Path(admin_users.__file__).read_text()
    findings = single_row_task_chat_message_deletes(source)
    assert findings == []


def test_single_row_delete_guard_flags_a_row_level_delete() -> None:
    fixture = """
def purge_one(db, message_id):
    message = db.query(TaskChatMessage).filter(TaskChatMessage.id == message_id).first()
    db.delete(message)
"""
    findings = single_row_task_chat_message_deletes(fixture)
    assert [func_name for _lineno, func_name in findings] == ["purge_one"]


# ---------------------------------------------------------------------------
# T-S-12: message_type never writes 'question_superseded' -> 'question'.
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


def reverse_supersede_writes(source: str) -> list[int]:
    """Statement sites that write ``message_type='question'`` on rows
    selected via ``message_type == 'question_superseded'`` -- the mirror
    image of what ``supersede_legacy_question_rows`` does, and the shape
    a hypothetical revert helper built by copy-and-flip would take.

    Two mutation carrier forms are checked, matching the two forms this
    repo's ORM bulk writes actually use: a dict-literal passed to
    ``.update(...)``/``.values(...)``, and a ``message_type=`` keyword on
    the same calls.

    Three shapes are known blind spots, disclosed rather than caught,
    matching the disclosure style the single-row-delete guard in this
    file and the pairing guard in test_supersede_predicate_pairing.py
    already use for their own known gaps:

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
    * a constant-composed literal -- ``TARGET = "question"`` bound once,
      then ``.update({TaskChatMessage.message_type: TARGET})`` -- invisible
      because the walk only recognizes an inline ``ast.Constant`` value,
      not a ``Name`` resolved back to one.

    Closing these would need the same dataflow-sensitive analysis the
    checkpoint-pointer pairing guard explicitly declines for its own ``**``
    carriers, for the same cost/benefit reason: a materially more
    expensive walk to close gaps with no live instance in this repo today.
    """
    tree = ast.parse(source)
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


def test_nothing_writes_message_type_back_to_question() -> None:
    findings: list[tuple[str, list[int]]] = []
    for path in _iter_source_files():
        source = path.read_text()
        if "question_superseded" not in source:
            continue
        hits = reverse_supersede_writes(source)
        if hits:
            findings.append((str(path.relative_to(SRC_ROOT)), hits))
    assert findings == []


def test_monotonicity_guard_flags_a_reverse_assignment() -> None:
    fixture = """
def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.task_id == task_id,
        TaskChatMessage.message_type == "question_superseded",
    ).update({TaskChatMessage.message_type: "question"}, synchronize_session=False)
"""
    assert len(reverse_supersede_writes(fixture)) == 1


# ---------------------------------------------------------------------------
# T-S-13: chat_history_service.py imports nothing from the native-rollout
# module (A-group import ban).
#
# This guard exists so chat_history_service.py does not become the first
# module to import rollout controls: the supersede helper is unconditional
# by contract and must never branch on rollout mode. It is written as a
# source-import scan rather than a runtime check precisely so it keeps
# working once that module lands -- today it cannot flag any real code,
# because the rollout module does not exist yet; that is known and
# accepted, not an oversight.
# ---------------------------------------------------------------------------

_BANNED_ROLLOUT_NAMES = frozenset(
    {
        "interaction_rollout",
        "evaluate_native_publication",
        "get_interaction_rollout_policy",
        "InteractionRolloutPolicy",
    }
)


def banned_rollout_imports(source: str) -> list[str]:
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
    tree = ast.parse(source)
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
    source = Path(chat_history_service.__file__).read_text()
    assert banned_rollout_imports(source) == []


def test_rollout_import_guard_flags_a_module_import() -> None:
    fixture = (
        "from ...services.interaction_rollout import evaluate_native_publication\n"
    )
    hits = banned_rollout_imports(fixture)
    assert hits


def test_rollout_import_guard_flags_the_policy_dataclass_by_name() -> None:
    fixture = "from xagent.web.services.policy import InteractionRolloutPolicy\n"
    hits = banned_rollout_imports(fixture)
    assert hits == ["xagent.web.services.policy.InteractionRolloutPolicy"]


def test_rollout_import_guard_flags_the_module_imported_as_a_bare_name() -> None:
    """The module itself, imported relatively and named directly rather
    than appearing inside a dotted path: ``from . import interaction_rollout``
    and ``from ..services import interaction_rollout``."""
    same_package = "from . import interaction_rollout\n"
    sibling_package = "from ..services import interaction_rollout\n"
    assert banned_rollout_imports(same_package)
    assert banned_rollout_imports(sibling_package)


def test_rollout_import_guard_ignores_an_unrelated_name_containing_the_substring() -> (
    None
):
    """Negative control: a name that merely *contains*
    ``interaction_rollout`` as a substring, rather than being exactly
    that module or one of the banned symbols, must not be flagged --
    otherwise a coincidentally similar identifier becomes unusable."""
    fixture = "from xagent.web.services import interaction_rollout_helper\n"
    assert banned_rollout_imports(fixture) == []


# ---------------------------------------------------------------------------
# Obligation 10, second half: the mid-turn WebSocket path never writes the
# "question_superseded" literal -- that is this helper's only site.
# ---------------------------------------------------------------------------

MID_TURN_FUNCTIONS = ("_persist_agent_outbound_event", "make_agent_outbound_handler")


def mid_turn_functions_writing_superseded_literal(
    source: str,
) -> tuple[list[str], set[str]]:
    """Return (hit function names, found function names).

    ``found`` is every name out of ``MID_TURN_FUNCTIONS`` that this source
    actually defines -- tracked separately from ``hits`` so a caller can
    tell "neither function writes the literal" apart from "neither
    function exists here anymore" (e.g. after a rename), the same
    non-vacuousness check ``test_supersede_predicate_pairing.py`` runs on
    its own reader/writer predicate sets before trusting an equality on
    them."""
    tree = ast.parse(source)
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
    source = Path(websocket.__file__).read_text()
    hits, found = mid_turn_functions_writing_superseded_literal(source)

    # Sanity: both tracked functions actually exist in this source -- an
    # empty or partial `found` set would make the hits == [] assertion
    # below vacuous (a renamed or removed function trivially "writes
    # nothing" because this guard never sees it at all).
    assert found == set(MID_TURN_FUNCTIONS)
    assert hits == []


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
    hits, _found = mid_turn_functions_writing_superseded_literal(fixture)
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
    hits, _found = mid_turn_functions_writing_superseded_literal(fixture)
    assert set(hits) == {"_persist_agent_outbound_event", "make_agent_outbound_handler"}

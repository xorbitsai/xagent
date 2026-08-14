"""Repository-wide architecture guards: rules about the shape of the source
tree, not about what any one feature does.

A guard belongs here when breaking it is a structural mistake -- a write
shape that destroys data the system is supposed to relabel, a value that
must only ever move in one direction, a module that must not depend on
another -- and when the rule is stated over the source tree rather than
over a running behaviour. A test that pins what one function returns
belongs with that function's own suite; a test that pins what may appear
anywhere under ``src/xagent`` belongs here.

Three guards live here today.

* **Single-row deletes of chat message rows are banned.** Transcript rows
  are relabeled, never removed: the only sanctioned write shapes are the
  bulk ``.update()`` calls in the chat history service and the bulk
  account-purge ``.delete()`` in the admin users API. A single-row delete
  would destroy transcript history instead of marking it superseded.
* **The message-type value only moves one way.** Nothing in the tree
  writes a chat message's ``message_type`` back from the superseded value
  to the question value, in any spelling those two constants have. The
  guard resolves module-scope constants and alias chains rather than
  matching the literal, so a file that imports the constant and never
  spells it is exactly the file it must still see.
* **The chat history service does not depend on the interaction rollout
  controls.** This one is a layering rule with a single subject rather
  than a tree-wide sweep: it reads that one module's source and rejects
  any import of the rollout module, its policy dataclass or its accessor,
  by dotted path or as a bare name. It is here, with the tree-wide
  guards, because it constrains the dependency graph -- which is an
  architectural fact -- and not because of how many files it reads.

Every guard parses source text (``ast.parse``) rather than importing each
module, so module-level import side effects stay out of scope. The two
tree-wide guards share one parse of ``src/xagent`` through the
``parsed_src_files`` fixture; the layering guard parses its one module
directly.

Each guard ships with its own controls in this file: a positive control
proving it does not fire on the repository's real sanctioned shape, and
negative controls proving it fires on each escaping shape it claims to
catch. A guard added here without both is a guard nobody can tell apart
from a guard that always passes.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

import pytest

from xagent.web.api import admin_users
from xagent.web.services import chat_history_service

SRC_ROOT = Path(chat_history_service.__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def parsed_src_files() -> tuple[tuple[str, str, ast.Module], ...]:
    """Every ``.py`` file under ``src/xagent``, read and parsed once for
    this module: ``(path relative to SRC_ROOT, source text, parsed tree)``.

    Two guards below sweep the whole tree. Each used to run its own
    ``rglob`` + ``read_text``, and the monotonicity sweep additionally
    skipped any file whose text did not contain the literal
    ``"question_superseded"``. That prefilter is gone: the guards now
    resolve the message-type constants by name, so a file that imports
    ``SUPERSEDED_MESSAGE_TYPE`` and never spells the literal is exactly
    the file that most needs scanning. Measured on this tree, the one full
    pass costs roughly an order of magnitude more than the two prefiltered
    passes it replaces -- the cost of scanning everything, paid once per
    module.

    Module-scoped rather than a module-level cache so pytest owns the
    lifetime and no state survives the module; the guards only read these
    trees and never mutate them.

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


def test_the_shared_source_pass_covers_every_file_in_src(parsed_src_files) -> None:
    """The substring prefilter is gone, so this fixture must really be
    every file, not a filtered subset."""
    on_disk = {str(p.relative_to(SRC_ROOT)) for p in SRC_ROOT.rglob("*.py")}
    assert {rel for rel, _source, _tree in parsed_src_files} == on_disk
    assert len(on_disk) > 1


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

    What this recognizes, stated as the rule rather than as a list of
    misses: the deleted object must be named by a bare ``Name`` in the
    call, and that name must have been bound somewhere in this function
    or one enclosing it, by an assignment whose right-hand side mentions
    ``TaskChatMessage`` literally. Anything that does not fit that shape
    is invisible -- the argument resolution is by name within a scope
    chain, not by dataflow.

    Shapes that consequently do not resolve, as examples rather than an
    exhaustive list: an argument arriving as an already-typed function
    parameter and never locally assigned; an inline expression such as
    ``db.delete(db.query(Model).first())``, where the argument is a
    ``Call``; a module-scope binding, since only bindings in the
    enclosing function chain are collected; and an attribute form such as
    ``db.delete(self.message)``, where the argument is an ``Attribute``.

    Every session-scoped instance delete in this tree today passes a bare
    local name, so these are latent rather than live. Closing them needs
    the dataflow-sensitive analysis this guard deliberately declines, for
    the same cost reason the checkpoint-pointer pairing guard declines it.
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
        # exist unless that name appears verbatim in the text. Unlike the
        # value-matching the monotonicity guard does, no constant or alias
        # can route around an identifier the guard matches by id. A skip
        # on ``".delete("`` would not be exactly implied -- ``ast`` does
        # not care about the whitespace between the attribute and the
        # paren -- so it is not used.
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


@pytest.mark.parametrize(
    "fixture,expected_names",
    [
        (
            """
def purge_one(db, message_id):
    message = db.query(TaskChatMessage).filter(TaskChatMessage.id == message_id).first()
    db.delete(message)
""",
            ["purge_one"],
        ),
        (
            """
def outer(db):
    def inner(db):
        msg = db.query(TaskChatMessage).first()
        return msg
    db.delete(msg)
""",
            [],
        ),
        (
            """
def outer(db):
    msg = db.query(TaskChatMessage).first()
    def inner():
        db.delete(msg)
    return inner
""",
            ["inner"],
        ),
        (
            """
def outer(db):
    def inner(db):
        msg = db.query(TaskChatMessage).first()
        db.delete(msg)
    return inner
""",
            ["inner"],
        ),
    ],
    ids=[
        "flat-function",
        "nested-binding-stays-in-its-own-scope",
        "closure-reads-the-enclosing-binding",
        "nested-function-binds-and-deletes-its-own",
    ],
)
def test_single_row_delete_guard_attributes_each_delete_to_its_own_scope(
    fixture: str, expected_names: list[str]
) -> None:
    """One table over the four scope shapes that matter. ``ast.walk`` does
    not stop at nested definitions, so the guard walks scopes explicitly:
    a name bound inside a nested ``def`` is not the enclosing function's
    (case 2), a nested function does read what its enclosing function
    bound (case 3), and every finding is reported under the function that
    lexically contains the delete (cases 3 and 4 both name ``inner``).
    """
    findings = single_row_task_chat_message_deletes(ast.parse(fixture))
    assert [name for _lineno, name in findings] == expected_names


# ---------------------------------------------------------------------------
# message_type never writes 'question_superseded' -> 'question' (monotonicity).
# ---------------------------------------------------------------------------


# The two message-type values, taken from the module that owns them rather
# than respelled here, so a guard can never end up hunting for a value the
# service no longer uses. ``test_message_type_constants_have_their_documented_values``
# in the behavior suite is what pins the values themselves.
_MESSAGE_TYPE_CONSTANTS = {
    "QUESTION_MESSAGE_TYPE": chat_history_service.QUESTION_MESSAGE_TYPE,
    "SUPERSEDED_MESSAGE_TYPE": chat_history_service.SUPERSEDED_MESSAGE_TYPE,
}


def _module_level_statements(tree: ast.Module):
    """Statements in module scope, descending through module-level ``if``
    and ``try`` blocks -- a constant defined under ``if TYPE_CHECKING:``
    or in a ``try``/``except ImportError`` fallback is still a
    module-scope binding. Function and class bodies are deliberately not
    descended into: a local ``x = "question"`` in some unrelated function
    must not make every ``x`` in the file resolve to that value.
    """
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, ast.If):
            pending.extend(node.body)
            pending.extend(node.orelse)
        elif isinstance(node, (ast.Try, ast.TryStar)):
            pending.extend(node.body)
            pending.extend(node.orelse)
            pending.extend(node.finalbody)
            for handler in node.handlers:
                pending.extend(handler.body)


def _string_constant_bindings(tree: ast.Module) -> dict[str, set[str]]:
    """Module-scope names mapped to every string value they are ever bound
    to, so a guard can recognize ``.update({message_type: SUPERSEDED})``
    as the same write as ``.update({message_type: "question_superseded"})``.

    Four carrier shapes are resolved:

    * a module-scope assignment to a string literal, plain or annotated
      (``S = "question_superseded"``, ``S: str = "question_superseded"``);
    * ``from ... import SUPERSEDED_MESSAGE_TYPE``, including
      ``... as S`` -- the imported name takes the value the owning module
      actually holds (``_MESSAGE_TYPE_CONSTANTS``), not a respelling;
    * a module-scope assignment from an attribute of one of those
      constants (``S = chat_history_service.SUPERSEDED_MESSAGE_TYPE``);
    * a chain of module-scope name-to-name assignments over any of the
      above (``A = S``; ``B = A``), resolved to a fixed point. Value sets
      only grow and the name set is finite, so a cycle (``A = B``;
      ``B = A``) terminates instead of looping.

    A name rebound at module scope keeps *every* value it was ever bound
    to rather than only the last one. "The last one" is not a defined
    notion here: ``_module_level_statements`` drains its work list with
    ``pop()``, so statements arrive last-in-first-out and not in source
    order. Keeping the union is the only well-defined answer, and it is
    also the safe one for something that feeds a ban.

    Shapes this does not resolve. Values built by anything other than a
    plain literal or one of the forms above -- an f-string,
    ``.format()``, ``"".join(...)``, an augmented assignment
    (``B += "..."``), ``getattr(module, "NAME")`` -- plus tuple-unpacking
    targets (``A, B = "x", "y"``), names arriving through
    ``from ... import *``, imports made inside a module-level ``with``
    block, constants defined in a class body, and a value reaching the
    write as a dict key or ``**`` carrier rather than as the mapped
    value. None of these is resolved; the table therefore stays a ban on
    the spellings it can see, not a proof that no other spelling exists.
    """
    bindings: dict[str, set[str]] = {}
    aliases: list[tuple[str, str]] = []

    def bind(name: str, value: str) -> None:
        bindings.setdefault(name, set()).add(value)

    for node in _module_level_statements(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _MESSAGE_TYPE_CONSTANTS:
                    bind(
                        alias.asname or alias.name, _MESSAGE_TYPE_CONSTANTS[alias.name]
                    )
            continue
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.value is None:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not names:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for name in names:
                bind(name, value.value)
        elif isinstance(value, ast.Name):
            aliases.extend((name, value.id) for name in names)
        elif isinstance(value, ast.Attribute) and value.attr in _MESSAGE_TYPE_CONSTANTS:
            for name in names:
                bind(name, _MESSAGE_TYPE_CONSTANTS[value.attr])

    changed = True
    while changed:
        changed = False
        for target, source in aliases:
            values = bindings.get(source)
            if values and not values <= bindings.get(target, set()):
                bindings.setdefault(target, set()).update(values)
                changed = True
    return bindings


def _string_values(node: ast.AST, bindings: dict[str, set[str]]) -> set[str]:
    """Every string value this expression can denote: an inline literal, a
    module-scope name bound to one, or ``<module>.NAME`` for one of the
    message-type constants. Anything else resolves to nothing, so a guard
    built on this stays a ban on known values rather than a guess."""
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, ast.Name):
        return set(bindings.get(node.id, ()))
    if isinstance(node, ast.Attribute) and node.attr in _MESSAGE_TYPE_CONSTANTS:
        return {_MESSAGE_TYPE_CONSTANTS[node.attr]}
    return set()


def _writes_message_type_to(
    node: ast.Call, value: str, bindings: dict[str, set[str]]
) -> bool:
    for arg in node.args:
        if isinstance(arg, ast.Dict):
            for key, val in zip(arg.keys, arg.values):
                key_name = (
                    key.attr
                    if isinstance(key, ast.Attribute)
                    else (key.value if isinstance(key, ast.Constant) else None)
                )
                if key_name == "message_type" and value in _string_values(
                    val, bindings
                ):
                    return True
    for kw in node.keywords:
        if kw.arg == "message_type" and value in _string_values(kw.value, bindings):
            return True
    return False


def _filters_message_type_equal(
    node: ast.Call, value: str, bindings: dict[str, set[str]]
) -> bool:
    for arg in node.args:
        if (
            isinstance(arg, ast.Compare)
            and len(arg.ops) == 1
            and isinstance(arg.ops[0], ast.Eq)
            and isinstance(arg.left, ast.Attribute)
            and arg.left.attr == "message_type"
            and len(arg.comparators) == 1
            and value in _string_values(arg.comparators[0], bindings)
        ):
            return True
    return False


def reverse_supersede_writes(tree: ast.Module) -> list[int]:
    """Statement sites that write ``message_type`` back to the question
    value on rows selected by the superseded value -- the mirror image of
    what ``supersede_legacy_question_rows`` does, and the shape a
    hypothetical revert helper built by copy-and-flip would take.

    Two mutation carrier forms are checked, matching the two forms this
    repo's ORM bulk writes actually use: a dict-literal passed to
    ``.update(...)``/``.values(...)``, and a ``message_type=`` keyword on
    the same calls. Both the written value and the filtered value are
    resolved through ``_string_constant_bindings``, so the constant
    spelling (``SUPERSEDED_MESSAGE_TYPE``, an aliased import of it, or
    ``chat_history_service.SUPERSEDED_MESSAGE_TYPE``) is caught alongside
    the bare literal. That matters because ``chat_history_service.py``
    exports both constants publicly, making the constant spelling the
    likelier one for a real revert helper. Which spellings that
    resolution does and does not cover is listed in
    ``_string_constant_bindings``; a value it cannot resolve is a blind
    spot of this guard too.

    Three further shapes are known blind spots of the walk itself,
    disclosed rather than caught, matching the disclosure style the
    single-row-delete guard above in this file already uses for its own
    known gaps:

    * per-row attribute assignment -- ``msg.message_type = "question"``
      on an ORM instance fetched via the superseded value -- invisible
      because this walk only inspects ``.update()``/``.values()`` calls,
      not plain attribute assignment.
    * a filter and its mutating call split across two statements --
      ``q = db.query(TaskChatMessage).filter(...)`` bound to a name in
      one statement, ``q.update(...)`` called on that name in a later
      one -- invisible because the walk only recognizes a ``.filter(...)``
      call inline in the same receiver chain as the mutating call, the
      same by-name (not full dataflow) limit the delete guard above notes
      for its own carrier tracing.
    * a comparison that is not an equality against a known value --
      ``.filter(message_type != "question")`` or
      ``.filter(message_type.in_([...]))`` -- invisible because
      ``_filters_message_type_equal`` matches only an ``ast.Compare``
      with a single ``ast.Eq``.

    Closing the remaining shapes would need the same dataflow-sensitive
    analysis the checkpoint-pointer pairing guard explicitly declines for
    its own ``**`` carriers, and for the same reason: a materially more
    expensive walk than what these shapes are worth guarding against.
    """
    bindings = _string_constant_bindings(tree)
    findings: list[int] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"update", "values"}
        ):
            continue
        if not _writes_message_type_to(
            node, chat_history_service.QUESTION_MESSAGE_TYPE, bindings
        ):
            continue
        receiver = node.func.value
        reads_superseded = False
        while isinstance(receiver, ast.Call):
            if (
                isinstance(receiver.func, ast.Attribute)
                and receiver.func.attr == "filter"
                and _filters_message_type_equal(
                    receiver, chat_history_service.SUPERSEDED_MESSAGE_TYPE, bindings
                )
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


def files_with_reverse_supersede_writes(
    files: Iterable[tuple[str, str, ast.Module]],
) -> list[tuple[str, list[int]]]:
    """Run the monotonicity guard over a sequence of
    ``(path, source, tree)`` triples and return the offending files.

    Split out from the test below so the sweep can be exercised against
    synthetic triples as well as against the real tree -- a guard that is
    only ever run over a tree with zero violations in it has never been
    shown to report one. No prefilter: the guard resolves the constants
    by name, so a file that imports ``SUPERSEDED_MESSAGE_TYPE`` and never
    spells the literal is precisely the file that must be scanned.
    """
    findings: list[tuple[str, list[int]]] = []
    for rel_path, _source, tree in files:
        hits = reverse_supersede_writes(tree)
        if hits:
            findings.append((rel_path, hits))
    return findings


def test_nothing_writes_message_type_back_to_question(parsed_src_files) -> None:
    findings = files_with_reverse_supersede_writes(parsed_src_files)
    assert findings == [], (
        "message_type written back from question_superseded to question "
        f"(monotonicity violation): {findings}"
    )


def test_the_monotonicity_sweep_reports_a_file_that_only_imports_the_constant() -> None:
    """The sweep itself, against a file shaped like the one the deleted
    substring prefilter used to skip: it imports the constants and the
    literal ``"question_superseded"`` appears nowhere in its text. Run in
    memory over a synthetic triple -- nothing is written into ``src/``.
    """
    source = """
from xagent.web.services.chat_history_service import (
    QUESTION_MESSAGE_TYPE,
    SUPERSEDED_MESSAGE_TYPE,
)


def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.message_type == SUPERSEDED_MESSAGE_TYPE,
    ).update({TaskChatMessage.message_type: QUESTION_MESSAGE_TYPE})
"""
    assert "question_superseded" not in source
    findings = files_with_reverse_supersede_writes(
        [("web/services/revert_helper.py", source, ast.parse(source))]
    )
    assert [rel_path for rel_path, _hits in findings] == [
        "web/services/revert_helper.py"
    ]


def test_monotonicity_guard_flags_a_reverse_assignment() -> None:
    fixture = """
def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.task_id == task_id,
        TaskChatMessage.message_type == "question_superseded",
    ).update({TaskChatMessage.message_type: "question"}, synchronize_session=False)
"""
    assert len(reverse_supersede_writes(ast.parse(fixture))) == 1


@pytest.mark.parametrize(
    "fixture",
    [
        # module-scope constants, the shape chat_history_service.py itself uses
        """
QUESTION_MESSAGE_TYPE = "question"
SUPERSEDED_MESSAGE_TYPE = "question_superseded"

def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.message_type == SUPERSEDED_MESSAGE_TYPE,
    ).update({TaskChatMessage.message_type: QUESTION_MESSAGE_TYPE})
""",
        # imported by name
        """
from xagent.web.services.chat_history_service import (
    QUESTION_MESSAGE_TYPE,
    SUPERSEDED_MESSAGE_TYPE,
)

def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.message_type == SUPERSEDED_MESSAGE_TYPE,
    ).update({TaskChatMessage.message_type: QUESTION_MESSAGE_TYPE})
""",
        # imported under an alias
        """
from xagent.web.services.chat_history_service import QUESTION_MESSAGE_TYPE as Q
from xagent.web.services.chat_history_service import SUPERSEDED_MESSAGE_TYPE as S

def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.message_type == S,
    ).update({TaskChatMessage.message_type: Q})
""",
        # reached through the owning module
        """
from xagent.web.services import chat_history_service

def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.message_type
        == chat_history_service.SUPERSEDED_MESSAGE_TYPE,
    ).update({TaskChatMessage.message_type: chat_history_service.QUESTION_MESSAGE_TYPE})
""",
    ],
    ids=["module-constant", "imported", "aliased-import", "module-attribute"],
)
def test_monotonicity_guard_flags_a_constant_spelled_revert(fixture: str) -> None:
    """The reverse write is the same violation whether the values are
    spelled as literals or as the public constants; a revert helper
    written in the same style as the forward write is the likelier
    shape."""
    assert len(reverse_supersede_writes(ast.parse(fixture))) == 1


def test_monotonicity_guard_resolves_a_module_level_alias_chain() -> None:
    fixture = """
from xagent.web.services.chat_history_service import SUPERSEDED_MESSAGE_TYPE as S
FIRST = S
SECOND = FIRST
QUESTION = "question"

def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.message_type == SECOND,
    ).update({TaskChatMessage.message_type: QUESTION})
"""
    assert len(reverse_supersede_writes(ast.parse(fixture))) == 1


def test_monotonicity_guard_ignores_a_function_local_binding() -> None:
    """Negative control: only module-scope bindings resolve. A local
    ``x = "question"`` in an unrelated function must not make every ``x``
    in the file read as that value -- that would false-positive on
    ordinary code."""
    fixture = """
def unrelated():
    x = "question"
    return x

def revert(db, task_id):
    db.query(TaskChatMessage).filter(
        TaskChatMessage.message_type == "question_superseded",
    ).update({TaskChatMessage.message_type: x})
"""
    assert reverse_supersede_writes(ast.parse(fixture)) == []


def test_constant_bindings_record_every_module_scope_string_shape() -> None:
    """One exact-dict assertion over all the module-scope shapes at once:
    a plain literal, an annotated literal, a rebound name keeping both
    values, and the four shapes that resolve to nothing."""
    fixture = """
PLAIN = "question"
ANNOTATED: str = "question_superseded"
REBOUND = "question"
REBOUND = "something else"
NUMBER = 3
PAIR = ("a", "b")
BUILT = "que"
BUILT += "stion"
DECLARED_ONLY: str
"""
    assert _string_constant_bindings(ast.parse(fixture)) == {
        "PLAIN": {"question"},
        "ANNOTATED": {"question_superseded"},
        "REBOUND": {"question", "something else"},
        "BUILT": {"que"},
    }


def test_constant_bindings_terminate_on_a_cyclic_alias_chain() -> None:
    """The fixed-point loop must not hang on ``A = B``/``B = A``: value
    sets only grow over a finite name set, so it settles at empty."""
    assert _string_constant_bindings(ast.parse("A = B\nB = A\n")) == {}


def test_constant_bindings_reach_module_level_if_and_try_blocks() -> None:
    fixture = """
import typing

if typing.TYPE_CHECKING:
    UNDER_IF = "question_superseded"

try:
    UNDER_TRY = "question"
except ImportError:
    UNDER_TRY = "question"
"""
    bindings = _string_constant_bindings(ast.parse(fixture))
    assert bindings["UNDER_IF"] == {"question_superseded"}
    assert bindings["UNDER_TRY"] == {"question"}


# ---------------------------------------------------------------------------
# chat_history_service.py imports nothing from the native-rollout module.
#
# This guard exists so chat_history_service.py never becomes a module
# that imports rollout controls: this module must not become a place
# where rollout mode is read; whether a future call site gates the call
# is that call site's own contract, not something this guard can see.
# The rollout module (``interaction_rollout.py``) exists in this
# repository -- this guard is what keeps this service module from ever
# importing it, and it now flags real code, not a hypothetical future
# import.
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
        f"module (this module must not import rollout controls): {hits}"
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

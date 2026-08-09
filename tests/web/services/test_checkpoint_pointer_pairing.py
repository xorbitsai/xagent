"""Static guard: the two checkpoint pointer columns are always moved together.

``tasks.last_checkpoint_event_id`` (legacy string) and
``tasks.last_checkpoint_trace_event_id`` (exact-row anchor) are dual-written
and dual-cleared across six mutation sites in five modules, and the recovery
CAS fence conjoins both. A statement that sets one without the other makes
that fence unmatchable, so recovery would stop working rather than fail
loudly. This check is the primary guard keeping the pair together once the
call sites drift apart during the compatibility window.

Reads are deliberately out of scope: several call sites legitimately name
exactly one column when resolving or querying it. Only mutation carriers are
matched -- dict-literal keys, call keywords, and subscript assignments.

Known limit: pairing is judged per statement block, because the lease claim
writes the pair through two sibling subscript assignments. The block union
therefore accepts two sibling statements that each write one column against
*different* targets -- a shape no current site uses. All six real sites
write both columns through one carrier (or two sibling assignments into the
same dict), so the blind spot has no live instance; it is recorded here so a
future reviewer does not mistake the guard for a per-statement check.

``**`` unpacking resolution: ``.values(**name)`` / ``.update(**name)`` name
a dict whose keys this walk cannot see without resolving what ``name`` is
bound to. A name bound to a dict *literal* somewhere in the enclosing
function is resolved -- its keys were already picked up by the ``ast.Dict``
branch below when the literal itself was walked, so the carrier contributes
nothing further and is treated as already-inspected. Every other carrier
(the call result of a builder, a name that only ever arrives as a function
parameter, ...) cannot be resolved statically, so it is counted as an
opaque, and therefore unpaired, write.

Because a plain ``dict.update(**name)`` is spelled the same way as a
SQLAlchemy bulk ``.update(**name)``, an unrelated dict merge in one of
these five modules is flagged too, even with no pointer column anywhere
near it. That is deliberate: separating the two would mean trusting the
receiver expression, and a real pointer write through a query object
bound to a local name (``stmt.update(**vals)``) would then stop being
flagged. A loud false positive on an unrelated merge is the better
trade for a guard whose silent failure mode is "recovery stops working";
write ``d.update(other)`` rather than ``d.update(**other)`` here.

This resolution is by name, not dataflow: once a name has been bound to a
dict literal anywhere in its function, every ``**`` of that name in that
function is accepted, even if the name is later reassigned or mutated.
Two shapes escape as a known consequence, both covered by a dedicated
negative-control test below so the limit stays visible instead of being
rediscovered:

* a local dict literal later merged with ``.update(builder())`` before
  being unpacked -- the merge's keys are invisible to this walk;
* a local dict literal later rebound to the result of a call before being
  unpacked -- the same name now points at unrelated content, but the name
  was "seen" as a literal once, so it stays resolved.

Widening the walk to be dataflow-sensitive would close both gaps, but that
is a materially different (and materially more expensive) analysis; the
name-based rule is deliberately the cheaper, coarser one, with its blind
spots recorded rather than silently accepted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from xagent.web.api import admin_users, trace_handlers
from xagent.web.services import task_deletion, task_lease_service, task_orchestrator

POINTER_COLUMNS = frozenset(
    {"last_checkpoint_event_id", "last_checkpoint_trace_event_id"}
)

# Sentinel column name for a ** carrier this walk could not resolve to a
# local dict literal. It is never a real column name, so it can never
# accidentally complete a pair -- it can only ever make a block's column set
# diverge from POINTER_COLUMNS and therefore get flagged.
_OPAQUE_STAR_CARRIER = "<opaque **>"

SUBJECT_MODULES = [
    trace_handlers,
    admin_users,
    task_lease_service,
    task_deletion,
    task_orchestrator,
]


def _enclosing_scopes(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    """Map every node to the nearest enclosing function, or the module
    itself for top-level code.

    A node's own scope is the function that directly contains it -- a
    nested function gets its own entry, not its parent's, so a name local
    to the inner function can never be resolved against the outer one's
    dict-literal bindings.
    """
    scope_of: dict[ast.AST, ast.AST] = {}

    def visit(node: ast.AST, scope: ast.AST) -> None:
        scope_of[node] = scope
        child_scope = (
            node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
        )
        for child in ast.iter_child_nodes(node):
            visit(child, child_scope)

    visit(tree, tree)
    return scope_of


def _local_dict_literal_names(
    tree: ast.Module, scope_of: dict[ast.AST, ast.AST]
) -> dict[ast.AST, set[str]]:
    """Per scope, the names ever bound to a dict literal in that scope:
    ``name = {...}`` / ``name: T = {...}``. A name resolved this way is
    treated as already inspected wherever it is later unpacked with ``**``
    in the same scope, regardless of what happens to it in between."""
    names_by_scope: dict[ast.AST, set[str]] = {}
    for node in ast.walk(tree):
        targets: list[ast.Name] = []
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.value, ast.Dict)
            and isinstance(node.target, ast.Name)
        ):
            targets = [node.target]
        if targets:
            scope = scope_of[node]
            names_by_scope.setdefault(scope, set()).update(t.id for t in targets)
    return names_by_scope


def _mutated_pointer_columns(
    statement: ast.stmt,
    scope_of: dict[ast.AST, ast.AST],
    local_dict_names: dict[ast.AST, set[str]],
) -> set[str]:
    """Pointer columns this statement *writes*, in any of the carrier forms
    the call sites actually use."""
    found: set[str] = set()
    for node in ast.walk(statement):
        if isinstance(node, ast.Dict):
            # {Task.last_checkpoint_event_id: None, ...} / {"col": None, ...}
            for key in node.keys:
                if isinstance(key, ast.Attribute) and key.attr in POINTER_COLUMNS:
                    found.add(key.attr)
                elif isinstance(key, ast.Constant) and key.value in POINTER_COLUMNS:
                    found.add(str(key.value))
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in POINTER_COLUMNS:
                    # .values(last_checkpoint_event_id=..., ...). ``else_=
                    # Task.<col>`` inside a case() is a keyword named
                    # "else_", so it never matches.
                    found.add(str(keyword.arg))
                elif keyword.arg is None:
                    # .values(**name) / .update(**name). Resolve only for
                    # these two method names -- an unrelated **-call (e.g.
                    # log.info(**kwargs)) is not a pointer-column carrier
                    # at all and must not be flagged.
                    callee = node.func
                    method_name = (
                        callee.attr if isinstance(callee, ast.Attribute) else None
                    )
                    if method_name in {"values", "update"}:
                        scope = scope_of.get(node)
                        resolved_names = (
                            local_dict_names.get(scope, set())
                            if scope is not None
                            else set()
                        )
                        if not (
                            isinstance(keyword.value, ast.Name)
                            and keyword.value.id in resolved_names
                        ):
                            found.add(_OPAQUE_STAR_CARRIER)
        elif isinstance(node, ast.Assign):
            # values["last_checkpoint_event_id"] = ...
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value in POINTER_COLUMNS
                ):
                    found.add(str(target.slice.value))
    return found


def unpaired_pointer_writes(source: str) -> list[tuple[int, tuple[str, ...]]]:
    """Statement blocks that write one pointer column but not the other.

    The pairing unit is the enclosing statement block, not the enclosing
    function: acquire_task_lease_no_commit sets both columns in two separate
    branches, so a function-level check would stay green if one branch lost a
    column. ``ast.walk`` from a block's statements descends into nested
    blocks, so an outer block's union is a superset of its children's -- which
    is harmless, because every nested block is also checked on its own
    iteration and no imbalance can hide inside one.
    """
    tree = ast.parse(source)
    scope_of = _enclosing_scopes(tree)
    local_dict_names = _local_dict_literal_names(tree, scope_of)
    findings: list[tuple[int, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            columns: set[str] = set()
            first_line: int | None = None
            for statement in block:
                if not isinstance(statement, ast.stmt):
                    continue
                mutated = _mutated_pointer_columns(
                    statement, scope_of, local_dict_names
                )
                if mutated and first_line is None:
                    first_line = statement.lineno
                columns |= mutated
            if columns and columns != POINTER_COLUMNS:
                findings.append((first_line or 0, tuple(sorted(columns))))
    return sorted(set(findings))


@pytest.mark.parametrize(
    "module",
    SUBJECT_MODULES,
    ids=lambda module: module.__name__.rsplit(".", 1)[-1],
)
def test_pointer_columns_are_always_written_in_pairs(module) -> None:
    source = Path(module.__file__).read_text()
    unpaired = unpaired_pointer_writes(source)
    assert unpaired == [], (
        f"{module.__name__} writes one checkpoint pointer column without the "
        f"other: {unpaired}"
    )


def test_pointer_pairing_ignores_single_column_reads() -> None:
    """Negative control: reading one column is not a mutation."""
    fixture = """
def h(db):
    return db.query(Task.last_checkpoint_event_id).filter(Task.id == 1).one_or_none()
"""
    assert unpaired_pointer_writes(fixture) == []


def test_pointer_pairing_ignores_a_case_else_naming_one_column() -> None:
    """The subtlest true negative: ``else_=Task.last_checkpoint_event_id`` is a
    call keyword, but its ``arg`` is "else_", not a column name, so the
    single-column case() builders must stay clean."""
    fixture = """
from sqlalchemy import case

def lease_checkpoint_event_id_case():
    return case((_predicate(), None), else_=Task.last_checkpoint_event_id)
"""
    assert unpaired_pointer_writes(fixture) == []


@pytest.mark.parametrize(
    "carrier,fixture",
    [
        (
            "dict_literal",
            """
def purge(db, task_id):
    db.query(Task).filter(Task.id == task_id).update(
        {Task.last_checkpoint_event_id: None},
        synchronize_session=False,
    )
""",
        ),
        (
            "call_keyword",
            """
def write(db, task_id, event):
    db.execute(
        update(Task).where(Task.id == task_id).values(
            last_checkpoint_event_id=str(event.id),
        )
    )
""",
        ),
        (
            "subscript_assign",
            """
def claim(values):
    values["last_checkpoint_event_id"] = None
    return values
""",
        ),
    ],
    ids=["dict_literal", "call_keyword", "subscript_assign"],
)
def test_pointer_pairing_flags_each_carrier_form(carrier: str, fixture: str) -> None:
    """Positive controls, one per carrier form. Asserted on the flagged column
    names rather than fixture line numbers, so the controls stay meaningful if
    the fixture text is reformatted."""
    unpaired = unpaired_pointer_writes(fixture)
    assert unpaired, carrier
    assert {columns for _, columns in unpaired} == {("last_checkpoint_event_id",)}


def test_pointer_pairing_accepts_a_paired_local_dict_unpacked_with_star() -> None:
    """Positive control for the ** resolution itself: a dict literal that
    pairs both columns, later unpacked with ``**``, must not be flagged --
    its keys were already seen on the literal."""
    fixture = """
def write(db, task_id, event):
    values = {
        "last_checkpoint_event_id": str(event.id),
        "last_checkpoint_trace_event_id": event.trace_event_id,
    }
    db.execute(update(Task).where(Task.id == task_id).values(**values))
"""
    assert unpaired_pointer_writes(fixture) == []


def test_pointer_pairing_ignores_an_unrelated_star_call() -> None:
    """Negative control: ``**`` unpacking into a method that is not
    ``.values``/``.update`` is not a pointer-column carrier at all."""
    fixture = """
def log_event(**kwargs):
    log.info(**kwargs)
"""
    assert unpaired_pointer_writes(fixture) == []


def test_pointer_pairing_flags_a_builder_result_unpacked_directly() -> None:
    """The gap this hardening closes: unpacking a builder call's result
    gives the walk no dict literal to read, so it must be flagged rather
    than silently treated as paired."""
    fixture = """
def write(db, task_id, anchor):
    db.execute(
        update(Task).where(Task.id == task_id).values(
            **checkpoint_pointer_values(anchor)
        )
    )
"""
    unpaired = unpaired_pointer_writes(fixture)
    assert unpaired
    assert {columns for _, columns in unpaired} == {(_OPAQUE_STAR_CARRIER,)}


def test_pointer_pairing_flags_a_name_bound_only_from_a_call() -> None:
    """A name that is never bound to a dict literal -- only ever assigned
    from a call -- stays unresolved even though it has a name at all."""
    fixture = """
def write(db, task_id, anchor):
    vals = checkpoint_pointer_values(anchor)
    db.execute(update(Task).where(Task.id == task_id).values(**vals))
"""
    unpaired = unpaired_pointer_writes(fixture)
    assert unpaired
    assert {columns for _, columns in unpaired} == {(_OPAQUE_STAR_CARRIER,)}


def test_pointer_pairing_flags_a_cross_function_dict_parameter() -> None:
    """A dict arriving as a parameter is never "seen" as a literal in this
    function's own body, so it stays unresolved regardless of what the
    caller passed."""
    fixture = """
def write(db, task_id, vals):
    db.execute(update(Task).where(Task.id == task_id).values(**vals))
"""
    unpaired = unpaired_pointer_writes(fixture)
    assert unpaired
    assert {columns for _, columns in unpaired} == {(_OPAQUE_STAR_CARRIER,)}


def test_pointer_pairing_known_gap_a_literal_merged_via_update_before_star() -> None:
    """Known gap (not a bug): a paired local dict literal later merged with
    ``.update(builder())`` before being unpacked. The merge's keys are
    invisible to this walk -- only the original literal's keys were ever
    seen -- so this stays green even though the values actually unpacked
    could differ from the literal. Changing this to catch the merge is a
    deliberate rule change, not a bug fix; if this test ever needs to
    become a positive control, that change needs its own decision, not a
    silent tightening.
    """
    fixture = """
def write(db, task_id, anchor):
    values = {
        "last_checkpoint_event_id": None,
        "last_checkpoint_trace_event_id": None,
    }
    values.update(checkpoint_pointer_values(anchor))
    db.execute(update(Task).where(Task.id == task_id).values(**values))
"""
    assert unpaired_pointer_writes(fixture) == []


def test_pointer_pairing_known_gap_a_literal_rebound_to_a_call_before_star() -> None:
    """Known gap (not a bug): a name first bound to a paired dict literal,
    then rebound to the result of a call, then unpacked. The rule is by
    name, not dataflow -- once ``values`` has been "seen" as a literal
    anywhere in the function, every ``**values`` in that function is
    accepted, even though this particular unpack sees only the rebound
    call result. See the sibling ``..._via_update...`` test for the other
    known gap; both are recorded rather than silently accepted.
    """
    fixture = """
def write(db, task_id, anchor):
    values = {
        "last_checkpoint_event_id": None,
        "last_checkpoint_trace_event_id": None,
    }
    values = checkpoint_pointer_values(anchor)
    db.execute(update(Task).where(Task.id == task_id).values(**values))
"""
    assert unpaired_pointer_writes(fixture) == []


def test_pointer_pairing_still_flags_a_single_column_keyword() -> None:
    """Regression control: the pre-existing single-column-keyword carrier
    (no ``**`` involved at all) must still be flagged after this
    hardening -- the ** resolution must not loosen the old rule."""
    fixture = """
def write(db, task_id):
    db.execute(update(Task).where(Task.id == task_id).values(
        last_checkpoint_event_id=None,
    ))
"""
    unpaired = unpaired_pointer_writes(fixture)
    assert unpaired
    assert {columns for _, columns in unpaired} == {("last_checkpoint_event_id",)}

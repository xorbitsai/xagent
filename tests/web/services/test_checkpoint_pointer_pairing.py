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

SUBJECT_MODULES = [
    trace_handlers,
    admin_users,
    task_lease_service,
    task_deletion,
    task_orchestrator,
]


def _mutated_pointer_columns(statement: ast.stmt) -> set[str]:
    """Pointer columns this statement *writes*, in any of the three carrier
    forms the call sites actually use."""
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
            # .values(last_checkpoint_event_id=..., ...). ``else_=Task.<col>``
            # inside a case() is a keyword named "else_", so it never matches.
            for keyword in node.keywords:
                if keyword.arg in POINTER_COLUMNS:
                    found.add(str(keyword.arg))
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
                mutated = _mutated_pointer_columns(statement)
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

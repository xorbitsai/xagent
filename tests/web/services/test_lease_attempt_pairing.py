"""Static guard: the runner id and its attempt id are always cleared together.

``tasks.runner_id`` names who holds the lease and ``tasks.lease_attempt_id``
names which acquisition minted that hold. Every writer that clears the
runner must also clear the attempt id in the same statement, or a later
attempt check could compare a stale ``lease_attempt_id`` left over from a
finished hold against a live one and treat them as a match. This check is
the guard that keeps the pair together as call sites are added or edited.

Unlike a plain "does this line mention runner_id" scan, matching here is
restricted to statements that write through an update carrier -- a call
whose method name is ``values`` or ``update``, the two SQLAlchemy entry
points used to build ``UPDATE ... SET`` statements in this codebase. Reads,
filters, and unrelated constructor/function calls that merely take a
``runner_id`` keyword (building a ``TaskLease`` snapshot, calling
``acquire_task_lease_no_commit``) are common and legitimate, and must not be
flagged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from xagent.web.api import a2a
from xagent.web.services import task_lease_service, task_orchestrator

LEASE_COLUMNS = frozenset({"runner_id", "lease_attempt_id"})

CARRIER_METHOD_NAMES = frozenset({"values", "update"})

SUBJECT_MODULES = [
    task_lease_service,
    task_orchestrator,
    a2a,
]


def _mutated_lease_columns(statement: ast.stmt) -> set[str]:
    """Lease columns this statement writes through an update carrier.

    Only ``Call`` nodes whose method name is ``values`` or ``update`` are
    considered carriers. Within such a call, both a keyword argument
    (``.values(runner_id=None)``) and a dict-literal argument
    (``.update({Task.runner_id: None})`` or ``.update({"runner_id": None})``)
    count as a write of that column.
    """
    found: set[str] = set()
    for node in ast.walk(statement):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in CARRIER_METHOD_NAMES):
            continue
        for keyword in node.keywords:
            if keyword.arg in LEASE_COLUMNS:
                found.add(str(keyword.arg))
        for arg in node.args:
            if not isinstance(arg, ast.Dict):
                continue
            for key in arg.keys:
                if isinstance(key, ast.Attribute) and key.attr in LEASE_COLUMNS:
                    found.add(key.attr)
                elif isinstance(key, ast.Constant) and key.value in LEASE_COLUMNS:
                    found.add(str(key.value))
    return found


def unpaired_lease_writes(source: str) -> list[tuple[int, tuple[str, ...]]]:
    """Statement blocks that write one lease column but not the other.

    The pairing unit is the enclosing statement block, not the enclosing
    function, mirroring the checkpoint-pointer guard this test is modeled
    on: ``ast.walk`` from a block's statements descends into nested blocks,
    so an outer block's union is a superset of its children's, and every
    nested block is also checked on its own iteration.
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
                mutated = _mutated_lease_columns(statement)
                if mutated and first_line is None:
                    first_line = statement.lineno
                columns |= mutated
            if columns and columns != LEASE_COLUMNS:
                findings.append((first_line or 0, tuple(sorted(columns))))
    return sorted(set(findings))


@pytest.mark.parametrize(
    "module",
    SUBJECT_MODULES,
    ids=lambda module: module.__name__.rsplit(".", 1)[-1],
)
def test_lease_columns_are_always_written_in_pairs(module) -> None:
    source = Path(module.__file__).read_text()
    unpaired = unpaired_lease_writes(source)
    assert unpaired == [], (
        f"{module.__name__} writes runner_id or lease_attempt_id without "
        f"the other: {unpaired}"
    )


def test_lease_pairing_ignores_a_function_call_keyword_naming_runner_id() -> None:
    """Negative control: a constructor or plain function call that happens to
    take a ``runner_id`` keyword is not an update carrier and must not be
    flagged -- e.g. building a ``TaskLease`` snapshot or calling
    ``acquire_task_lease_no_commit``."""
    fixture = """
def snapshot(task):
    return TaskLease(task_id=task.id, runner_id=task.runner_id, run_id=task.run_id)

def acquire(db, task_id):
    return acquire_task_lease_no_commit(db, task_id, runner_id=get_runner_id())
"""
    assert unpaired_lease_writes(fixture) == []


def test_lease_pairing_ignores_reads_and_filters() -> None:
    """Negative control: reading or filtering on runner_id is not a mutation."""
    fixture = """
def h(db):
    return db.query(Task.runner_id).filter(Task.runner_id == "x").one_or_none()
"""
    assert unpaired_lease_writes(fixture) == []


@pytest.mark.parametrize(
    "carrier,fixture",
    [
        (
            "values_keyword",
            """
def write(db, task_id):
    db.execute(
        update(Task).where(Task.id == task_id).values(
            runner_id=None,
        )
    )
""",
        ),
        (
            "update_dict_task_col",
            """
def purge(db, task_id):
    db.query(Task).filter(Task.id == task_id).update(
        {Task.runner_id: None},
        synchronize_session=False,
    )
""",
        ),
        (
            "update_dict_string_key",
            """
def purge(db, task_id):
    db.query(Task).filter(Task.id == task_id).update(
        {"runner_id": None},
        synchronize_session=False,
    )
""",
        ),
    ],
    ids=["values_keyword", "update_dict_task_col", "update_dict_string_key"],
)
def test_lease_pairing_flags_each_carrier_form(carrier: str, fixture: str) -> None:
    """Positive controls, one per carrier form. Asserted on the flagged column
    names rather than fixture line numbers, so the controls stay meaningful if
    the fixture text is reformatted."""
    unpaired = unpaired_lease_writes(fixture)
    assert unpaired, carrier
    assert {columns for _, columns in unpaired} == {("runner_id",)}

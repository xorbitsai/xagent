"""Repo-wide sentinel for a raw string literal compared against Task.status.

xagent.web.models.task.task_status_predicate is the one typed entry point
for every SQL predicate against Task.status; the column stores enum member
*names*, not values (see that module). A hand-written string literal
placed beside Task.status reaches an ORM/Core comparison, which fails at
bind time with StatementError wrapping LookupError, symmetrically on
SQLite and PostgreSQL; the zero-row outcome this check exists to prevent
survives only for raw text() SQL that bypasses that bind layer.

Scope note: this is a narrow, mechanical check for that one mistake shape
(a string literal appearing directly in source next to a Task.status
attribute access). It does not evaluate every Task.status comparison in the
repository -- typed TaskStatus comparisons are legitimate and common, and
are backstopped separately by validate_strings=True on the column (a write
of a string that is not a valid member name fails loudly at bind time,
StatementError/LookupError). A comparison against a variable that happens
to hold a raw string at runtime, rather than a literal appearing in the
source, is outside what static AST scanning can see; so is a string
literal passed to a comparator method via `*args`/`**kwargs` unpacking
instead of a plain positional or keyword argument. Both gaps are accepted,
not closed, by this check.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_ROOT = REPO_ROOT / "src" / "xagent"

# Every SQLAlchemy ColumnOperators method that compares Task.status against
# a caller-supplied value. label/op/collate/cast/distinct/asc/desc are
# deliberately excluded -- they take a string legitimately (a label name, an
# operator symbol, a collation name) or take no compared value at all, so
# flagging them would produce false positives on ordinary code rather than
# catching the value-vs-name mistake this scan exists for.
_COMPARISON_METHODS = {
    # membership
    "in_",
    "not_in",
    "notin_",
    # identity
    "is_",
    "is_not",
    "isnot",
    "is_distinct_from",
    "is_not_distinct_from",
    # string matching
    "like",
    "not_like",
    "notlike",
    "ilike",
    "not_ilike",
    "notilike",
    "startswith",
    "endswith",
    "contains",
    "match",
    "regexp_match",
    # range
    "between",
}


def _is_task_status_attr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "status"
        and isinstance(node.value, ast.Name)
        and node.value.id == "Task"
    )


def _has_string_literal(node: ast.AST) -> bool:
    """A bare string constant, or a list/tuple/set literal containing one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(
            isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            for elt in node.elts
        )
    return False


def _violations(tree: ast.AST) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(_is_task_status_attr(o) for o in operands) and any(
                _has_string_literal(o) for o in operands
            ):
                hits.append((node.lineno, ast.unparse(node)))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _COMPARISON_METHODS
            and _is_task_status_attr(node.func.value)
            and any(
                _has_string_literal(value)
                for value in [*node.args, *(kw.value for kw in node.keywords)]
            )
        ):
            hits.append((node.lineno, ast.unparse(node)))
    return hits


def scan_source(source: str, label: str) -> list[tuple[str, int, str]]:
    tree = ast.parse(source, filename=label)
    return [(label, lineno, sig) for lineno, sig in _violations(tree)]


def scan_file(path: Path) -> list[tuple[str, int, str]]:
    relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    return scan_source(path.read_text(), relative)


def scan_tree(root: Path = SCAN_ROOT) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        hits.extend(scan_file(path))
    return hits


def test_repo_has_zero_literal_status_predicates() -> None:
    violations = scan_tree()
    assert violations == [], (
        "found a raw string literal compared against Task.status -- use a "
        "typed TaskStatus member (or task_status_predicate) instead: "
        f"{violations}"
    )


# --- Meta-test: prove the scan actually fires on every shape --------------

_FIXTURE_VIOLATIONS = """
from xagent.web.models.task import Task

def eq_literal():
    return Task.status == "waiting_for_user"

def ne_literal():
    return Task.status != "waiting_for_user"

def in_call_literal():
    return Task.status.in_(["waiting_for_user"])

def not_in_call_literal():
    return Task.status.not_in(["waiting_for_user"])

def in_compare_literal():
    return Task.status in ["waiting_for_user"]

def not_in_compare_literal():
    return Task.status not in ["waiting_for_user"]
"""

_FIXTURE_NEGATIVE_CONTROLS = """
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.agent import Agent

def typed_eq_comparison():
    return Task.status == TaskStatus.RUNNING

def typed_in_call():
    return Task.status.in_([TaskStatus.RUNNING, TaskStatus.PENDING])

def unrelated_status_attr_on_another_name():
    return Agent.status == "draft"

def string_literal_compared_to_a_different_column():
    return Task.runner_id == "waiting_for_user"

def label_takes_a_string_legitimately():
    return Task.status.label("status")

def op_takes_a_string_legitimately():
    return Task.status.op("=")("running")

def is_none_is_not_a_miscompile():
    return Task.status.is_(None)
"""


def test_scan_finds_every_violating_shape() -> None:
    violations = scan_source(_FIXTURE_VIOLATIONS, "fixture_violations.py")
    assert len(violations) == 6, violations


def test_scan_ignores_negative_controls() -> None:
    violations = scan_source(_FIXTURE_NEGATIVE_CONTROLS, "fixture_negative_controls.py")
    assert violations == [], violations


# --- One row per _COMPARISON_METHODS entry, plus the keyword-argument shape

_COMPARISON_METHOD_CASES = [
    ("keyword_in", 'Task.status.in_(other=["running"])'),
    ("in_", 'Task.status.in_(["running"])'),
    ("not_in", 'Task.status.not_in(["running"])'),
    ("notin_", 'Task.status.notin_(["running"])'),
    ("is_", 'Task.status.is_("running")'),
    ("is_not", 'Task.status.is_not("running")'),
    ("isnot", 'Task.status.isnot("running")'),
    ("is_distinct_from", 'Task.status.is_distinct_from("running")'),
    ("is_not_distinct_from", 'Task.status.is_not_distinct_from("running")'),
    ("like", 'Task.status.like("run%")'),
    ("not_like", 'Task.status.not_like("run%")'),
    ("notlike", 'Task.status.notlike("run%")'),
    ("ilike", 'Task.status.ilike("run%")'),
    ("not_ilike", 'Task.status.not_ilike("run%")'),
    ("notilike", 'Task.status.notilike("run%")'),
    ("startswith", 'Task.status.startswith("run")'),
    ("endswith", 'Task.status.endswith("ing")'),
    ("contains", 'Task.status.contains("run")'),
    ("match", 'Task.status.match("running")'),
    ("regexp_match", 'Task.status.regexp_match("running")'),
    ("between", 'Task.status.between("a", "z")'),
]


@pytest.mark.parametrize(
    "case_id,source",
    _COMPARISON_METHOD_CASES,
    ids=[case_id for case_id, _ in _COMPARISON_METHOD_CASES],
)
def test_scan_flags_every_comparison_method(case_id: str, source: str) -> None:
    violations = scan_source(source, f"fixture_{case_id}.py")
    assert len(violations) == 1, (case_id, violations)


def test_every_banned_method_has_a_capability_case() -> None:
    covered = {case_id for case_id, _ in _COMPARISON_METHOD_CASES}
    assert _COMPARISON_METHODS <= covered, sorted(_COMPARISON_METHODS - covered)


def test_fixtures_are_syntactically_valid() -> None:
    ast.parse(_FIXTURE_VIOLATIONS)
    ast.parse(_FIXTURE_NEGATIVE_CONTROLS)

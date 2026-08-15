"""Static equivalence between the shared lease-fence predicates
(``task_lease_service.py``) and the two WebSocket finalizers'
(``websocket.py``) own inline ownership fencing.

The two finalizers, ``_finalize_task_execution_result_isolated`` and
``_finalize_resumed_task``, do not call ``task_row_matches_lease_owner`` --
see that predicate's module comment for why: they compile the ownership
condition into the WHERE clause of a locking ``SELECT``, so "no row came
back" is what tells them ownership changed, and the row lock they take is
scoped by that same condition. Loading a row and then calling the shared
predicate would move the lock and change what a late result means, so the
finalizers keep their existing inline form rather than being rewritten to
call the shared predicate. This file pins the agreement between the two
forms structurally, rather than leaving it to be kept by review alone.

AST, not text: a substring scan would also match this module's own prose
mentioning ``Task.runner_id`` or ``lease_attempt_id``, and would not
distinguish "this identifier appears" from "this identifier appears in the
fence condition".

Known, deliberate differences between the two forms -- each entry in this
table is either normalized away or explicitly excluded by name below (one
shape outside this table is a disclosed blind spot: see the ownership
helper's own docstring on constant-valued filter conditions):

| Difference | Why it is not drift |
|---|---|
| Finalizers write ``Task.runner_id`` (an ORM column), predicate writes ``task.runner_id`` (an instance attribute) | Same field, SQL-side spelling versus Python-side spelling |
| Finalizers use ``.filter(a, b)``'s implicit AND, predicate uses explicit ``and`` | Same conjunction; SQLAlchemy's multi-arg ``filter`` is AND |
| The clarification resolver (not a finalizer) writes the negated form (``!=`` with ``or``), the predicate writes the positive form (``==`` with ``and``) | De Morgan's law; the resolver now calls the predicate and negates the result, so the two forms no longer coexist as separate spellings |
| Finalizers' filter judges by "the filtered SELECT returned no row", predicate judges by "the comparison is false" | Same condition, two landing points; the finalizers compile it into the WHERE clause so the row lock they take is scoped by it |
| Finalizers additionally filter on ``Task.id == task_id`` | That identifies which row, not whether it is still owned; excluded explicitly below |
| Finalizers additionally call ``.with_for_update()`` | Locking is the caller's concern; the predicate is a pure boolean, takes no lock and issues no SQL |
| Finalizers never compare ``lease_attempt_id`` / ``attempt_id`` | A deliberate capability gap -- the WebSocket finalizers do not carry the attempt contract; the predicate's third check does not apply to them (cell 3 below) |
| Finalizers read the lease off a local named ``task_lease``, predicate reads it off a parameter named ``lease`` | Same object, different local name at each call site; normalized by ``_ROOT_NAME_REWRITES`` below so cell 2 compares field *and* object, not field name alone |
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

from xagent.web.api import websocket as websocket_module
from xagent.web.services import task_lease_service as lease_service

_FINALIZER_NAMES = (
    "_finalize_task_execution_result_isolated",
    "_finalize_resumed_task",
)


def _parse_function(func: Any) -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    (node,) = tree.body
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return node


def _unfenced_lease_guard(func_node: ast.AST) -> ast.If:
    """The ``if task_lease.run_id is None: ...`` guard inside a finalizer
    (or, for the isolated-session finalizer, the nested variant of the
    same shape one level deeper under ``if task_lease is not None:``)."""

    candidates = [
        node
        for node in ast.walk(func_node)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Attribute)
        and node.test.left.attr == "run_id"
        and isinstance(node.test.left.value, ast.Name)
        and node.test.left.value.id == "task_lease"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Is)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None
    ]
    assert len(candidates) == 1, (
        f"expected exactly one `if task_lease.run_id is None` guard, "
        f"found {len(candidates)}"
    )
    return candidates[0]


def test_both_finalizers_reject_an_unfenced_lease_the_same_way() -> None:
    """Cell 1: both finalizers gate on the identical
    ``task_lease.run_id is None`` comparison, and ``lease_is_fenced``
    (``task_lease_service.py``) is that same comparison's negation.

    If this turns red, either a finalizer's unfenced-lease guard changed
    shape, or ``lease_is_fenced`` no longer agrees with what "unfenced"
    means to the code that never calls it.
    """

    for name in _FINALIZER_NAMES:
        func = getattr(websocket_module, name)
        func_node = _parse_function(func)
        guard = _unfenced_lease_guard(func_node)
        # The guard's own shape is asserted by _unfenced_lease_guard's
        # search predicate above; reaching here means it matched.
        assert guard is not None

    predicate_node = _parse_function(lease_service.lease_is_fenced)
    (return_stmt,) = [
        node for node in ast.walk(predicate_node) if isinstance(node, ast.Return)
    ]
    compare = return_stmt.value
    assert isinstance(compare, ast.Compare)
    assert isinstance(compare.left, ast.Attribute) and compare.left.attr == "run_id"
    assert isinstance(compare.left.value, ast.Name) and compare.left.value.id == "lease"
    assert len(compare.ops) == 1 and isinstance(compare.ops[0], ast.IsNot)
    assert len(compare.comparators) == 1
    assert (
        isinstance(compare.comparators[0], ast.Constant)
        and compare.comparators[0].value is None
    )


def _ownership_filter_call(func_node: ast.AST) -> ast.Call:
    """The ``.filter(...)`` call inside a finalizer whose arguments compare
    ``Task.runner_id`` -- the ownership-fencing SELECT, not any other
    ``.filter(...)`` call the function may contain."""

    candidates = []
    for node in ast.walk(func_node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "filter":
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Compare)
                and isinstance(arg.left, ast.Attribute)
                and isinstance(arg.left.value, ast.Name)
                and arg.left.value.id == "Task"
                and arg.left.attr == "runner_id"
            ):
                candidates.append(node)
                break
    assert len(candidates) == 1, (
        f"expected exactly one ownership .filter(...) call, found {len(candidates)}"
    )
    return candidates[0]


# Known, deliberate root-name difference between the two forms (see the
# module docstring's table): the finalizers' inline filter reads the lease
# off a local named `task_lease`, the shared predicate reads the same
# runtime object off a parameter named `lease`. Normalized here so the two
# sides' triples compare on "same object, same field" rather than making
# every ownership comparison fail to match on root name alone.
_ROOT_NAME_REWRITES = {"task_lease": "lease"}


def _compare_triple(node: ast.Compare) -> tuple[str, str, str, str]:
    """The comparison's shape as (left field, op, right root, right field).

    The right side's root -- the object its attribute access reads off of,
    normalized through ``_ROOT_NAME_REWRITES`` for the one known legal
    spelling difference above -- is recorded alongside the field name.
    Without it, a comparison against the wrong object (``task.run_id``
    written where ``lease.run_id`` was meant) would still report the field
    name "run_id" and match by coincidence; recording the root turns that
    into a real mismatch.
    """
    assert isinstance(node.left, ast.Attribute)
    assert len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)
    assert len(node.comparators) == 1
    right = node.comparators[0]
    assert isinstance(right, ast.Attribute)
    assert isinstance(right.value, ast.Name)
    right_root = _ROOT_NAME_REWRITES.get(right.value.id, right.value.id)
    return (node.left.attr, "Eq", right_root, right.attr)


def _finalizer_ownership_triples(func_node: ast.AST) -> set[tuple[str, str, str, str]]:
    """The ownership filter call may also carry the primary-key term
    (``Task.id == task_id``) alongside the runner/run comparisons, when
    the finalizer folds both into the same ``.filter()`` call -- see
    ``_finalizer_has_primary_key_filter``'s docstring. That term's right
    side is a bare local variable (``task_id``), not a ``task_lease.*``
    attribute access, so it is excluded here by shape rather than
    included and then subtracted -- its presence is asserted separately.

    Known blind spot: this only collects comparisons whose right side is a
    plain-name-rooted attribute access (``task_lease.runner_id``, not a
    longer chain or a constant). A comparison filtered against a constant
    (for example ``Task.runner_id == "some-literal"``) does not match that
    shape, so it is silently dropped from the returned set rather than
    surfaced as a mismatch -- the equivalence assertion below would not go
    red for it. If a finalizer's ownership filter ever takes that shape,
    it needs a human to notice and review it directly.
    """

    call = _ownership_filter_call(func_node)
    triples = set()
    for arg in call.args:
        if (
            isinstance(arg, ast.Compare)
            and len(arg.comparators) == 1
            and isinstance(arg.comparators[0], ast.Attribute)
            and isinstance(arg.comparators[0].value, ast.Name)
        ):
            triples.add(_compare_triple(arg))
    return triples


def _finalizer_has_primary_key_filter(func_node: ast.AST) -> bool:
    """Whether ``Task.id == task_id`` appears anywhere in the finalizer's
    body, not necessarily inside the same ``.filter(...)`` call as the
    ownership fields: ``_finalize_task_execution_result_isolated`` filters
    on it in an earlier, outer ``.filter()`` call
    (``task_query = finalize_db.query(Task).filter(Task.id == task_id)``),
    while ``_finalize_resumed_task`` folds it into the same call as the
    ownership fields. Both are legitimate; only the presence of the term
    somewhere in the function is asserted here, not its call-site
    grouping."""

    for node in ast.walk(func_node):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Attribute)
            and isinstance(node.left.value, ast.Name)
            and node.left.value.id == "Task"
            and node.left.attr == "id"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id == "task_id"
        ):
            return True
    return False


def _predicate_ownership_triples() -> set[tuple[str, str, str, str]]:
    """Built through the same ``_compare_triple`` the finalizer side uses,
    so a comparison against the wrong object on this side (``task.run_id``
    written where ``lease.run_id`` was meant) is caught by the identical
    mechanism, not a second, independently-written one that could drift
    from it.
    """

    predicate_node = _parse_function(lease_service.task_row_matches_lease_owner)
    (return_stmt,) = [
        node for node in ast.walk(predicate_node) if isinstance(node, ast.Return)
    ]
    body = return_stmt.value
    # The predicate returns `bool(<expr>)` to satisfy mypy against
    # SQLAlchemy's instrumented-attribute typing on Task's columns; unwrap
    # that call to reach the actual boolean expression being compared.
    if (
        isinstance(body, ast.Call)
        and isinstance(body.func, ast.Name)
        and body.func.id == "bool"
        and len(body.args) == 1
    ):
        body = body.args[0]
    assert isinstance(body, ast.BoolOp) and isinstance(body.op, ast.And)
    triples = set()
    for value in body.values:
        assert isinstance(value, ast.Compare)
        triples.add(_compare_triple(value))
    return triples


def test_ownership_filter_fields_match_the_shared_predicate() -> None:
    """Cell 2: the field set each finalizer's locking SELECT filters on for
    ownership -- excluding the primary-key lookup, which identifies which
    row to look at, not whether it is still owned -- equals the field set
    ``task_row_matches_lease_owner`` compares, and both equal
    ``{(runner_id, lease, runner_id), (run_id, lease, run_id)}`` once the
    finalizers' ``task_lease`` root is normalized to ``lease`` (see
    ``_ROOT_NAME_REWRITES``). The primary-key term itself is required to be
    present somewhere in the finalizer (see
    ``_finalizer_has_primary_key_filter``'s own docstring for why it is not
    required to share the same ``.filter()`` call as the ownership terms).
    """

    predicate_triples = _predicate_ownership_triples()
    assert predicate_triples == {
        ("runner_id", "Eq", "lease", "runner_id"),
        ("run_id", "Eq", "lease", "run_id"),
    }

    for name in _FINALIZER_NAMES:
        func = getattr(websocket_module, name)
        func_node = _parse_function(func)
        assert _finalizer_has_primary_key_filter(func_node), (
            f"{name}: expected a Task.id == task_id filter somewhere in the function"
        )
        finalizer_triples = _finalizer_ownership_triples(func_node)
        assert finalizer_triples == predicate_triples, (
            f"{name}: ownership filter fields diverged from the shared predicate"
        )


def test_neither_finalizer_compares_attempt_identity() -> None:
    """Cell 3: the WebSocket finalizers do not carry the attempt contract
    ``task_row_matches_lease_attempt`` adds as its third check -- that
    predicate does not apply to them. If this turns red, someone gave a
    finalizer an attempt comparison, and its relationship to
    ``task_row_matches_lease_attempt`` needs a deliberate decision, not a
    silently passing assertion.
    """

    forbidden = {"lease_attempt_id", "attempt_id"}
    for name in _FINALIZER_NAMES:
        func = getattr(websocket_module, name)
        func_node = _parse_function(func)
        offenders = set()
        for node in ast.walk(func_node):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                offenders.add(node.attr)
            if isinstance(node, ast.Name) and node.id in forbidden:
                offenders.add(node.id)
        assert offenders == set(), (
            f"{name}: unexpected attempt-identity reference {offenders}"
        )

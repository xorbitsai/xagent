"""Static guards for two boundaries the clarification resolver leans on
without re-checking them itself.

First: a resume anchor's mere existence is trusted by
``resolve_publishable_clarification`` to already mean "the task row was
``RUNNING``, owned by the runner and run that wrote this checkpoint, and not
scoped to a child build" -- the resolver never re-queries the task row for
any of that. That trust is only sound as long as the one place an anchor's
backing pointer gets written (``trace_handlers.py``'s checkpoint pointer
UPDATE) keeps exactly those conditions in its own ``where`` clause and
keeps computing the lease it updates against under the same
``build_id is None`` gate. If a future change loosens that UPDATE's
guard, the resolver would silently inherit the loosened guarantee with no
test anywhere pointing at the connection -- these two tests are that
pointer.

Second: the resolver and the ops-signal registry it uses are both web-layer
modules; nothing in the core agent/pattern layer is allowed to import the
registry directly, since a degradation signal is meaningful only once a
web-layer caller has resolved a Task row's ownership -- the pattern layer
never holds that context.
"""

from __future__ import annotations

import ast
from pathlib import Path

import xagent.web.api.trace_handlers as trace_handlers_module

TRACE_HANDLERS_PATH = Path(trace_handlers_module.__file__)


def _function_named(tree: ast.Module, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"no function named {name!r} found")


def _pointer_update_where_calls(func_node: ast.AST) -> list[ast.Call]:
    """``.where(...)`` calls chained directly onto ``update(Task)``."""

    calls = []
    for node in ast.walk(func_node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "where"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "update"
            and node.func.value.args
            and isinstance(node.func.value.args[0], ast.Name)
            and node.func.value.args[0].id == "Task"
        ):
            calls.append(node)
    return calls


def _task_equality_targets(where_call: ast.Call) -> set[str]:
    """``Task.<attr>`` names compared with ``==`` among this call's args."""

    targets: set[str] = set()
    for arg in where_call.args:
        if (
            isinstance(arg, ast.Compare)
            and len(arg.ops) == 1
            and isinstance(arg.ops[0], ast.Eq)
            and isinstance(arg.left, ast.Attribute)
            and isinstance(arg.left.value, ast.Name)
            and arg.left.value.id == "Task"
        ):
            targets.add(arg.left.attr)
    return targets


def _status_compares_to_running(where_call: ast.Call) -> bool:
    for arg in where_call.args:
        if not (
            isinstance(arg, ast.Compare)
            and len(arg.ops) == 1
            and isinstance(arg.ops[0], ast.Eq)
            and isinstance(arg.left, ast.Attribute)
            and arg.left.attr == "status"
            and len(arg.comparators) == 1
        ):
            continue
        comparator = arg.comparators[0]
        if (
            isinstance(comparator, ast.Attribute)
            and isinstance(comparator.value, ast.Name)
            and comparator.value.id == "TaskStatus"
            and comparator.attr == "RUNNING"
        ):
            return True
    return False


def _checkpoint_lease_gated_by_build_id_is_none(func_node: ast.AST) -> bool:
    """Whether some ``checkpoint_lease = ...`` assignment in ``func_node``
    is a ternary whose test is exactly ``self.build_id is None``."""

    for node in ast.walk(func_node):
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "checkpoint_lease"
                for target in node.targets
            )
            and isinstance(node.value, ast.IfExp)
        ):
            continue
        test = node.value.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is)
            and isinstance(test.left, ast.Attribute)
            and isinstance(test.left.value, ast.Name)
            and test.left.value.id == "self"
            and test.left.attr == "build_id"
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            return True
    return False


def checkpoint_pointer_guard_findings(source: str) -> dict[str, object]:
    """Everything the real assertion below needs, factored out so the
    mutation tests can exercise it against synthetic fixtures too."""

    tree = ast.parse(source)
    func_node = _function_named(tree, "_save_trace_event")
    where_calls = _pointer_update_where_calls(func_node)
    targets: set[str] = set()
    status_ok = False
    for call in where_calls:
        targets |= _task_equality_targets(call)
        status_ok = status_ok or _status_compares_to_running(call)
    return {
        "where_call_count": len(where_calls),
        "equality_targets": targets,
        "status_compares_to_running": status_ok,
        "checkpoint_lease_gated": _checkpoint_lease_gated_by_build_id_is_none(
            func_node
        ),
    }


REQUIRED_EQUALITY_TARGETS = {"runner_id", "run_id"}


def test_checkpoint_pointer_update_carries_its_representation_guards() -> None:
    source = TRACE_HANDLERS_PATH.read_text(encoding="utf-8")
    findings = checkpoint_pointer_guard_findings(source)
    assert findings["where_call_count"] >= 1
    assert REQUIRED_EQUALITY_TARGETS <= findings["equality_targets"]
    assert findings["status_compares_to_running"] is True
    assert findings["checkpoint_lease_gated"] is True


def test_dropping_the_build_id_gate_is_caught_by_the_guard() -> None:
    """Mutation control: a version of the function that always resolves a
    lease, dropping the ``self.build_id is None`` ternary, must be flagged
    -- this is the regression the guard above exists to catch."""

    mutated = """
def _save_trace_event(self, db, event):
    checkpoint_lease = current_task_lease()
    db.execute(
        update(Task)
        .where(
            Task.id == self.task_id,
            Task.status == TaskStatus.RUNNING,
            Task.runner_id == checkpoint_lease.runner_id,
            Task.run_id == checkpoint_lease.run_id,
        )
        .values(last_checkpoint_event_id=str(event.id))
    )
"""
    findings = checkpoint_pointer_guard_findings(mutated)
    assert findings["checkpoint_lease_gated"] is False


def test_dropping_a_where_condition_is_caught_by_the_guard() -> None:
    """Mutation control: dropping the ``run_id`` fence from the ``where``
    clause must be flagged."""

    mutated = """
def _save_trace_event(self, db, event):
    checkpoint_lease = current_task_lease() if self.build_id is None else None
    db.execute(
        update(Task)
        .where(
            Task.id == self.task_id,
            Task.status == TaskStatus.RUNNING,
            Task.runner_id == checkpoint_lease.runner_id,
        )
        .values(last_checkpoint_event_id=str(event.id))
    )
"""
    findings = checkpoint_pointer_guard_findings(mutated)
    assert not (REQUIRED_EQUALITY_TARGETS <= findings["equality_targets"])


def test_a_function_with_no_pointer_update_reports_zero_where_calls() -> None:
    """Negative control: a function that never touches the pointer at all
    must not be mistaken for one that carries the guard."""

    source = """
def _save_trace_event(self, db, event):
    pass
"""
    findings = checkpoint_pointer_guard_findings(source)
    assert findings["where_call_count"] == 0


# ---------------------------------------------------------------------------
# Core layer must not import the web-layer degradation registry
# ---------------------------------------------------------------------------


def _imports_ops_signals(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == (
            "ops_signals"
        ):
            return True
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[-1] == "ops_signals" for alias in node.names
        ):
            return True
    return False


def test_core_agent_layer_never_imports_the_degradation_registry() -> None:
    import xagent.core as core_package

    root = Path(next(iter(core_package.__path__)))
    offenders = []
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if _imports_ops_signals(source):
            offenders.append(str(path))
    assert offenders == [], offenders


def test_import_detector_catches_a_synthetic_offender() -> None:
    """Positive control for the detector itself."""

    source = "from xagent.web.services.ops_signals import register_degradation\n"
    assert _imports_ops_signals(source) is True


def test_import_detector_ignores_unrelated_imports() -> None:
    source = "from xagent.web.services import task_lease_service\n"
    assert _imports_ops_signals(source) is False

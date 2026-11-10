"""AST guard: every production ``Task(...)`` picks its storage version.

New creation sites must choose deliberately: either route through
``new_task_conversation_storage_version`` with a pinned entry name, or be an
allowlisted legacy path fixed at version 1.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import xagent.config as _config_module
from xagent.web.models.task import Task

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="AST guards require Python >= 3.11 to parse this codebase's syntax",
)

_SRC_ROOT = Path(_config_module.__file__).parent
_TASK_MODULE = "xagent.web.models.task"
_SELECTOR = "new_task_conversation_storage_version"

# (file relative to src/xagent, enclosing function) -> entry names used.
EXPECTED_SELECTOR_SITES: dict[tuple[str, str], frozenset[str]] = {
    ("web/api/chat.py", "create_task"): frozenset({"web_chat", "agent_preview"}),
    ("web/api/public_chat_access.py", "create_public_chat_task"): frozenset({"widget"}),
    ("web/api/public_chat_access.py", "create_share_chat_task"): frozenset(
        {"shared_link"}
    ),
    ("web/services/task_start.py", "_prepare_created_task_isolated"): frozenset(
        {"sdk"}
    ),
    ("web/services/task_start.py", "_prepare_a2a_turn_sync"): frozenset({"a2a"}),
    ("web/services/channel_runtime.py", "prepare_channel_task_no_commit"): frozenset(
        {"channel"}
    ),
    ("web/services/triggers.py", "_attach_task_to_trigger_run"): frozenset({"trigger"}),
    ("web/services/workforce_runs.py", "_create_run_and_task_no_commit"): frozenset(
        {"workforce"}
    ),
}

# Legacy paths fixed at version 1 (never rolled out).
EXPECTED_FIXED_V1_SITES: frozenset[tuple[str, str]] = frozenset(
    {
        ("web/services/task_command_execution.py", "_prepare_task_message_sync"),
        ("web/services/agent_service_manager.py", "_get_agent_for_task_unlocked"),
    }
)


class _Finder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.task_names: set[str] = set()
        self.stack: list[str] = []
        self.sites: list[tuple[str, ast.Call]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.task_names:
            self.sites.append((self.stack[-1] if self.stack else "<module>", node))
        self.generic_visit(node)


def _resolve_task_imports(tree: ast.Module, path: Path) -> set[str]:
    """Names bound to the Task model, resolving relative imports."""
    rel = path.relative_to(_SRC_ROOT.parent).with_suffix("")
    package = list(rel.parts[:-1])
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = package[: len(package) - (node.level - 1)]
            module = ".".join(base + ([node.module] if node.module else []))
        else:
            module = node.module or ""
        if module == _TASK_MODULE:
            names.update(a.asname or a.name for a in node.names if a.name == "Task")
        elif module == "xagent.web.models":
            names.update(a.asname or a.name for a in node.names if a.name == "Task")
    return names


def _scan() -> list[tuple[str, str, ast.Call]]:
    found: list[tuple[str, str, ast.Call]] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = _resolve_task_imports(tree, path)
        if not names:
            continue
        finder = _Finder()
        finder.task_names = names
        finder.visit(tree)
        rel = path.relative_to(_SRC_ROOT).as_posix()
        found.extend((rel, func, call) for func, call in finder.sites)
    return found


def _version_value(call: ast.Call) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == "conversation_storage_version":
            return keyword.value
    return None


def _selector_calls(expr: ast.expr) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(expr)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _SELECTOR
    ]


def _entry_names(selector_call: ast.Call) -> set[str]:
    entry = next((k.value for k in selector_call.keywords if k.arg == "entry"), None)
    assert entry is not None, "selector call must pass entry= by keyword"
    return _constants(entry)


def _constants(expr: ast.expr) -> set[str]:
    """String constants of a constant or a conditional between constants."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {expr.value}
    if isinstance(expr, ast.IfExp):
        return _constants(expr.body) | _constants(expr.orelse)
    raise AssertionError("entry= must be a string constant (or a conditional of them)")


def test_every_task_construction_sets_conversation_storage_version() -> None:
    sites = _scan()
    assert sites, "scan found no Task(...) constructions; the guard is broken"
    problems = []
    selector_sites: dict[tuple[str, str], set[str]] = {}
    fixed_v1: set[tuple[str, str]] = set()
    for rel, func, call in sites:
        value = _version_value(call)
        where = f"{rel}:{call.lineno} in {func}"
        if value is None:
            problems.append(f"{where}: missing conversation_storage_version=")
            continue
        selectors = _selector_calls(value)
        if selectors:
            assert len(selectors) == 1, where
            selector_sites.setdefault((rel, func), set()).update(
                _entry_names(selectors[0])
            )
        elif isinstance(value, ast.Constant) and value.value == 1:
            fixed_v1.add((rel, func))
        else:
            problems.append(f"{where}: must call {_SELECTOR} or be the literal 1")
    assert not problems, "\n".join(problems)

    assert {k: frozenset(v) for k, v in selector_sites.items()} == (
        EXPECTED_SELECTOR_SITES
    )
    assert fixed_v1 == EXPECTED_FIXED_V1_SITES
    # Every construction is accounted for: one site per expected (file, function).
    assert len(sites) == len(EXPECTED_SELECTOR_SITES) + len(EXPECTED_FIXED_V1_SITES)


def test_column_default_and_server_default_remain_legacy() -> None:
    column = Task.__table__.c.conversation_storage_version
    assert column.default.arg == 1
    assert str(column.server_default.arg) == "1"

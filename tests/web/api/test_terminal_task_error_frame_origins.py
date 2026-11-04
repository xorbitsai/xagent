"""Pins where a terminal-frame ``code`` argument is allowed to come from.

``create_terminal_task_error_event``'s own runtime gate only checks the
*value* of ``code`` (a member of the closed set, and nothing else) -- it has
no way to know whether that value came from a curated projector or from an
incidental string a future caller happened to have on hand. This test closes
that gap statically: every call site that passes ``code=`` must bind it, in
the same function, from a call to ``connector_runtime_client_code`` -- the
one function in this repository that projects an exception onto a
client-visible code. Anything else (a literal, ``str(exc)``, a field read off
the exception directly) is an unauthorized source, even if the value it
produces happens to be a real closed-set member today.

The second test pins the other way a code reaches a frame: a caller
asserting it through ``asserted_code=``, on the terminal frame or on the
external command error frame.
"""

from __future__ import annotations

import ast
from pathlib import Path

import xagent

# Anchored on a real package file rather than assumed relative to this test
# file: xagent is a namespace package, so it has no single __file__ of its
# own, but the first path entry is the actual source tree to scan.
WEB_ROOT = Path(xagent.__path__[0]) / "web"

PROJECTOR_NAME = "connector_runtime_client_code"


def _call_func_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _enclosing_function(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _has_projector_binding(
    func: ast.FunctionDef | ast.AsyncFunctionDef, bound_name: str
) -> bool:
    """True when ``bound_name`` is assigned from a bare projector call.

    Matches only ``<bound_name> = connector_runtime_client_code(...)`` --
    a single-target ``Assign`` whose value is a direct call to the
    projector. A tuple-unpacking or boolean-fallback shape does not count:
    the projector already returns ``None`` for anything it does not
    recognize, so a caller does not need (and should not add) a second
    layer of fallback between the call and the frame.
    """

    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Name) and target.id == bound_name):
            continue
        value = stmt.value
        if isinstance(value, ast.Call) and _call_func_name(value) == PROJECTOR_NAME:
            return True
    return False


def _scan() -> tuple[set[tuple[str, str]], int]:
    """Returns (call sites that pass code=, count of recognized bindings).

    A call site is included in the first element only when its ``code=``
    argument is a bare name AND that name is bound, in the same enclosing
    function, from a direct call to the projector -- anything else is a
    hard failure, not a silently-excluded call site, so an unauthorized
    source cannot pass this test by looking like an unrecognized shape.
    """

    call_sites: set[tuple[str, str]] = set()
    recognized_bindings = 0

    for path in sorted(WEB_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_func_name(node) != "create_terminal_task_error_event":
                continue
            code_kw = next((kw for kw in node.keywords if kw.arg == "code"), None)
            if code_kw is None:
                continue

            func = _enclosing_function(node, parents)
            func_name = func.name if func is not None else "<module>"
            rel_path = path.relative_to(WEB_ROOT.parent).as_posix()

            if not isinstance(code_kw.value, ast.Name):
                raise AssertionError(
                    f"unauthorized code= source at {rel_path}:{node.lineno} "
                    f"in {func_name}: {ast.dump(code_kw.value)}"
                )
            if func is None or not _has_projector_binding(func, code_kw.value.id):
                raise AssertionError(
                    f"code= argument {code_kw.value.id!r} at "
                    f"{rel_path}:{node.lineno} in {func_name} is not bound "
                    f"from {PROJECTOR_NAME}(...) in the same function"
                )

            call_sites.add((rel_path, func_name))
            recognized_bindings += 1

    return call_sites, recognized_bindings


def test_every_code_argument_traces_to_the_projector() -> None:
    call_sites, recognized_bindings = _scan()

    assert call_sites == {("web/services/task_orchestrator.py", "_runner")}
    # Guards against the scanner silently matching nothing: a rewritten
    # projector call, a renamed binding, or a moved raise site should fail
    # loudly here rather than let the assertion above pass on an empty set.
    assert recognized_bindings == 1


PACKAGE_ROOT = Path(xagent.__path__[0])
ASSERTED_KEYWORD = "asserted_code"


def _asserted_code_sites() -> list[tuple[str, str, str]]:
    """Every ``asserted_code=`` keyword in the package, one entry per call.

    Matched on the keyword, not on the callee's name, so an aliased import of
    the builder is still seen. The value must be spelled
    ``ClientErrorCode.<MEMBER>``; anything else is a hard failure.
    """

    sites: list[tuple[str, str, str]] = []

    def visit(node: ast.AST, rel_path: str, scope: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = node.name
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg != ASSERTED_KEYWORD:
                    continue
                value = keyword.value
                if not (
                    isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "ClientErrorCode"
                ):
                    raise AssertionError(
                        f"asserted_code= at {rel_path}:{node.lineno} in {scope} "
                        f"is not a literal ClientErrorCode member: {ast.dump(value)}"
                    )
                sites.append((rel_path, scope, value.attr))
        for child in ast.iter_child_nodes(node):
            visit(child, rel_path, scope)

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visit(tree, path.relative_to(PACKAGE_ROOT).as_posix(), "<module>")
    return sites


def test_every_asserted_code_has_a_pinned_caller() -> None:
    """Each caller-asserted code is asserted once, by the caller that proved it."""

    assert _asserted_code_sites() == [
        (
            "web/services/external_task_cancel.py",
            "_broadcast_external_cancel_terminal_event",
            "EXTERNAL_TURN_INTERRUPTED",
        ),
        (
            "web/services/task_command_execution.py",
            "_broadcast_terminal_command_error",
            "EXTERNAL_STOP_NOT_APPLIED",
        ),
    ]

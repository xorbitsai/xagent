"""No function that can reach an installed connector team hook may be a
coroutine.

An installed hook is slow synchronous work: the seam is designed on the
assumption that the installing application answers from its own tables.
FastAPI runs a coroutine route on the event loop thread itself, so a slow
hook call inside an ``async def`` stalls every other request the process is
serving; a plain ``def`` goes to the threadpool instead, where a slow call
occupies one worker.

Stated for the whole module by reachability rather than as a hand-written
list of routes, because an earlier fix for this same risk class swept
siblings along the "takes a row lock" axis and therefore missed a route that
calls a hook without taking one.
"""

from __future__ import annotations

import ast
import importlib
import inspect

_SEAM_MODULE = "xagent.web.api.mcp"

# Every top-level function in this module that can reach an installed
# connector team hook. Written out so the discovery below cannot pass by
# finding nothing.
_SEAM_REACHING_FUNCTIONS = {
    "_local_mcp_can_attach",
    # The coroutine that owns app-scoped teardown, ``teardown_mcp_app_server``,
    # is absent on purpose: it hands this helper to ``asyncio.to_thread``
    # instead of calling it, so the seam runs in a worker thread and the
    # coroutine reaches it on no thread of its own. The discovery below follows
    # plain-name calls, which is exactly the distinction that matters here --
    # turning that dispatch back into a direct call would put the seam back on
    # the event loop, and would also put the coroutine back in this set and in
    # the offender list.
    "_teardown_mcp_app_server_locally",
    "delete_mcp_server",
    "get_mcp_servers",
    "list_mcp_apps",
    "update_mcp_server",
}

# The one function that reaches the seam and is still a coroutine, with the
# fact that makes it impossible to convert. Its own await -- one that is not
# the seam call itself -- is asserted below, so this entry cannot be claimed
# by a route whose coroutine is only the seam's doing.
_COROUTINE_EXEMPTIONS = {"delete_mcp_server"}


def _functions_reaching_the_connector_seam() -> dict[str, ast.AST]:
    """Every top-level function in this module that can reach an installed
    connector team hook.

    Seeded on the functions that import ``connector_team_scope`` in their own
    body, then closed transitively over plain-name calls, so that a route
    reaching the seam only through a helper is enumerated as well. Every name
    found on this module today is a seed, and the closure currently adds
    nothing; it is here so that introducing such a helper cannot quietly drop
    a route out of this check.
    """
    module = importlib.import_module(_SEAM_MODULE)
    tree = ast.parse(inspect.getsource(module))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reaching = {
        name
        for name, node in functions.items()
        if any(
            isinstance(child, ast.ImportFrom)
            and child.module is not None
            and child.module.endswith("connector_team_scope")
            for child in ast.walk(node)
        )
    }
    changed = True
    while changed:
        changed = False
        for name, node in functions.items():
            if name in reaching:
                continue
            called = {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            if called & reaching:
                reaching.add(name)
                changed = True
    return {name: functions[name] for name in reaching}


def _seam_names_imported_by(node: ast.AST) -> set[str]:
    """The names this function imports from ``connector_team_scope``.

    Read off the function's own body because that is how every call site in
    this module reaches the seam -- the same fact
    ``_functions_reaching_the_connector_seam`` above is seeded on.
    """
    return {
        alias.asname or alias.name
        for child in ast.walk(node)
        if isinstance(child, ast.ImportFrom)
        and child.module is not None
        and child.module.endswith("connector_team_scope")
        for alias in child.names
    }


def test_the_discovery_of_seam_reaching_functions_is_not_vacuous():
    """Pins the enumeration itself, so the assertion below cannot pass by
    finding nothing."""
    assert set(_functions_reaching_the_connector_seam()) == _SEAM_REACHING_FUNCTIONS


def test_no_function_that_reaches_the_connector_seam_is_a_coroutine():
    """An installed connector team hook may be slow -- the seam is designed on
    the assumption that the installing application answers from its own
    tables. FastAPI runs a coroutine route on the event loop thread itself, so
    a slow hook call inside an ``async def`` stalls every other request the
    process is serving, not just this one; a plain ``def`` goes to the
    threadpool instead, where a slow call occupies one worker.

    Enumerated by reachability rather than by a hand-written list of routes:
    an earlier fix for this same risk class swept siblings along the "takes a
    row lock" axis and therefore missed a route that calls a hook without
    taking one.
    """
    offenders = []
    for name, node in _functions_reaching_the_connector_seam().items():
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if name in _COROUTINE_EXEMPTIONS:
            # An exemption is only legitimate for a function that genuinely
            # cannot be converted, so it must carry an await that is NOT the
            # seam call itself. A function whose only await IS the seam call
            # is a coroutine of the seam's own making -- convertible by making
            # that call synchronous -- and "contains some await" would still
            # wave it through.
            seam_names = _seam_names_imported_by(node)
            non_seam_awaits = [
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Await)
                and not (
                    isinstance(child.value, ast.Call)
                    and isinstance(child.value.func, ast.Name)
                    and child.value.func.id in seam_names
                )
            ]
            assert non_seam_awaits, (
                f"{name} is exempted from this invariant, but every await it "
                "has is a seam call -- the coroutine is the seam's own doing, "
                "so make that call synchronous instead of exempting the route"
            )
            continue
        offenders.append(name)
    assert offenders == [], (
        "these functions can reach an installed connector team hook while "
        f"running on the event loop thread: {sorted(offenders)}"
    )

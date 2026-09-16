"""No function in ``custom_api`` or ``mcp`` that can reach an installed
connector team hook may be a coroutine.

An installed hook is slow synchronous work: the seam is designed on the
assumption that the installing application answers from its own tables.
FastAPI runs a coroutine route on the event loop thread itself, so a slow
hook call inside an ``async def`` stalls every other request the process is
serving; a plain ``def`` goes to the threadpool instead, where a slow call
occupies one worker.

Stated for each seam module by reachability rather than as a hand-written
list of routes, because an earlier fix for this same risk class swept
siblings along the "takes a row lock" axis and therefore missed a route that
calls a hook without taking one.

The reachability this check follows is scoped to the two modules in
``_SEAM_MODULES``: a function that reaches the seam only by calling into
another module -- for example ``chat.get_agent_connector_runtime_requirements``,
which reaches ``connector_team_scope`` through ``connector_runtime.py`` --
is not enumerated and not checked here. That cross-module path is tracked
separately (xorbitsai/xagent#2192); widening ``_SEAM_MODULES`` to include
``connector_runtime`` or ``chat`` would not close it, since the closure
below only follows calls within one module.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import threading

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.mcp import teardown_mcp_app_server
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    ConnectorDeleteDecision,
    set_connector_team_hooks,
    snapshot_connector_team_hooks,
)

_SEAM_MODULES = ("xagent.web.api.custom_api", "xagent.web.api.mcp")

# Every function or method, across every seam module above, that can reach
# an installed connector team hook. Written out so the discovery below
# cannot pass by finding nothing. Keyed by the function's fully qualified
# ``module.function`` or ``module.Class.method`` name -- the full dotted
# module path, not its basename, so that two seam modules sharing a
# function's basename could never collide on one dict/set key.
_SEAM_REACHING_FUNCTIONS = {
    "xagent.web.api.custom_api._recheck_team_access_under_definition_lock",
    "xagent.web.api.custom_api._resolve_custom_api_for_request",
    "xagent.web.api.custom_api.get_custom_api",
    "xagent.web.api.custom_api.update_custom_api",
    "xagent.web.api.custom_api.delete_custom_api",
    "xagent.web.api.mcp._local_mcp_can_attach",
    "xagent.web.api.mcp._resolve_mcp_server_for_request",
    # The coroutine that owns app-scoped teardown, ``teardown_mcp_app_server``,
    # is absent on purpose: it hands this helper to ``asyncio.to_thread``
    # instead of calling it, so the seam runs in a worker thread and the
    # coroutine reaches it on no thread of its own. The discovery below follows
    # plain-name calls, which is exactly the distinction that matters here --
    # turning that dispatch back into a direct call would put the seam back on
    # the event loop, and would also put the coroutine back in this set and in
    # the offender list.
    "xagent.web.api.mcp._teardown_mcp_app_server_locally",
    "xagent.web.api.mcp.delete_mcp_server",
    "xagent.web.api.mcp.get_mcp_server",
    "xagent.web.api.mcp.get_mcp_servers",
    "xagent.web.api.mcp.list_mcp_apps",
    "xagent.web.api.mcp.update_mcp_server",
}


def _type_checking_block_node_ids(tree: ast.Module) -> set[int]:
    """``id()`` of every node inside an ``if TYPE_CHECKING:`` block's body.

    An import there exists only for the type checker; at runtime that branch
    never executes, so a hook name imported only under ``TYPE_CHECKING``
    reaches nothing. Skipping these nodes keeps ``_seam_names_in_scope``
    from treating a type-only import (for example, a ``ConnectorAccess``
    annotation import) as a live way to reach the seam.
    """
    skip: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for statement in node.body:
                for descendant in ast.walk(statement):
                    skip.add(id(descendant))
    return skip


def _seam_names_in_scope(tree: ast.Module) -> tuple[set[str], set[str]]:
    """The two kinds of local name a module can use to reach
    ``connector_team_scope``, from an import anywhere in that module --
    inside a function body or at module scope, but not inside an
    ``if TYPE_CHECKING:`` block, which never runs.

    The first set is a hook function's own name, from
    ``from ...connector_team_scope import delete_team_connector``; a call to
    one of these is only a seam call if it is a bare name or the attribute of
    some object (``x.delete_team_connector(...)``), matched by name alone,
    since the object it hangs off cannot be resolved statically. The second
    set is a module alias, from ``from ...services import
    connector_team_scope`` or ``import ...connector_team_scope``; any
    attribute call on that alias reaches the seam, whichever hook function it
    names, because the alias itself has no other reason to exist in this
    module.

    A name reachable only through a module-scope import has no import node
    inside the function that uses it, so seeding on a function's own body
    alone would miss it.

    A hook name assigned to a plain variable (``x = delete_team_connector``,
    or ``x = connector_team_scope.delete_team_connector``) reaches the seam
    under that new name too, so both sets are closed transitively over such
    assignments -- a chain like ``y = x`` after ``x = delete_team_connector``
    is followed until no new name is added.
    """
    skip_ids = _type_checking_block_node_ids(tree)
    function_names: set[str] = set()
    module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in skip_ids:
            continue
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.endswith("connector_team_scope"):
                # from ...services.connector_team_scope import delete_team_connector
                function_names.update(
                    alias.asname or alias.name for alias in node.names
                )
            for alias in node.names:
                if alias.name == "connector_team_scope":
                    # from ...services import connector_team_scope
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("connector_team_scope"):
                    module_aliases.add(alias.asname or alias.name.rsplit(".", 1)[-1])

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if id(node) in skip_ids or not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            if target in function_names:
                continue
            value = node.value
            if isinstance(value, ast.Name) and value.id in function_names:
                function_names.add(target)
                changed = True
            elif (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in module_aliases
            ):
                function_names.add(target)
                changed = True
    return function_names, module_aliases


def _calls_a_seam_name(
    node: ast.AST, function_names: set[str], module_aliases: set[str]
) -> bool:
    """Whether this function or method calls a hook function name -- bare
    (``delete_team_connector(...)``) or as an attribute
    (``x.delete_team_connector(...)``) -- or calls any attribute of a
    ``connector_team_scope`` module alias
    (``connector_team_scope.delete_team_connector(...)``).
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name) and child.func.id in function_names:
            return True
        if isinstance(child.func, ast.Attribute):
            if child.func.attr in function_names:
                return True
            if (
                isinstance(child.func.value, ast.Name)
                and child.func.value.id in module_aliases
            ):
                return True
    return False


def _functions_reaching_the_connector_seam() -> dict[str, ast.AST]:
    """Every function or method, across every seam module, that can reach an
    installed connector team hook.

    Seeded on every function or method whose body calls a name that reaches
    ``connector_team_scope`` -- imported either in its own body or at its
    module's top level -- then closed transitively over plain-name calls to
    another function in the same module already in the reaching set, so that
    a route reaching the seam only through a local helper is enumerated as
    well: on the MCP side, ``get_mcp_server`` reaches the seam only through
    ``_resolve_mcp_server_for_request``, and a seed-only check would miss it.
    A method defined in a class body is kept, keyed as
    ``module.Class.method``, rather than dropped, since these modules already
    have ``async def`` methods.
    """
    reaching: dict[str, ast.AST] = {}
    for module_path in _SEAM_MODULES:
        module = importlib.import_module(module_path)
        tree = ast.parse(inspect.getsource(module))
        function_names, module_aliases = _seam_names_in_scope(tree)

        functions: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions[f"{node.name}.{child.name}"] = child

        local_reaching = {
            name
            for name, node in functions.items()
            if _calls_a_seam_name(node, function_names, module_aliases)
        }
        changed = True
        while changed:
            changed = False
            for name, node in functions.items():
                if name in local_reaching:
                    continue
                # Bare-name calls only (``helper()``), not
                # ``self.helper()`` or ``mod.helper()``: this closure is
                # for a route reaching the seam through a same-module
                # helper, which is always called by its own name, not
                # dispatched. Widening it to attribute calls would also
                # start matching ``asyncio.to_thread(helper, ...)``-style
                # dispatch as a direct call, which is exactly the
                # distinction ``_teardown_mcp_app_server_locally``'s
                # exemption from this module's coroutine below relies on
                # staying an exemption.
                called = {
                    child.func.id
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                }
                if called & local_reaching:
                    local_reaching.add(name)
                    changed = True

        for name in local_reaching:
            reaching[f"{module_path}.{name}"] = functions[name]
    return reaching


def _seam_names_imported_by(node: ast.AST) -> set[str]:
    """The names this function imports from ``connector_team_scope``.

    Read off the function's own body because that is how the exempted call
    site reaches the seam -- the same fact
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
        if name == "xagent.web.api.mcp.delete_mcp_server":
            # The one function that reaches the seam and is still a
            # coroutine. It could be converted with the same split this PR
            # applies to ``teardown_mcp_app_server`` -- its own await,
            # asserted below, is a real external OAuth revocation at the end
            # of the function, in the same shape this PR already handles. It
            # is exempted instead because it is an existing production route
            # and this PR deliberately leaves that route unchanged;
            # converting it is tracked separately.
            #
            # An exemption is only legitimate for a function that carries an
            # await that is NOT the seam call itself. A function whose only
            # await IS the seam call is a coroutine of the seam's own
            # making -- convertible by making that call synchronous -- and
            # "contains some await" would still wave it through. The check
            # below only proves that some other await exists, not that the
            # function resists conversion -- it could be satisfied by an
            # await that does nothing (``await asyncio.sleep(0)``) and still
            # pass. Tightening it to verify the hook call itself is off the
            # event loop thread is not done here.
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


@pytest.fixture()
def _db_session(tmp_path):
    db_path = tmp_path / "seam-runtime.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = SessionLocal()
    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield db, user
    db.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_teardown_locally_runs_the_connector_team_hook_off_the_event_loop_thread(
    _db_session,
):
    """Every static check above only proves the local teardown half is a
    plain ``def`` dispatched with ``asyncio.to_thread`` -- none of them prove
    that an installed hook actually observes a thread other than the one
    running this coroutine. This pins that fact directly: an installed
    connector-deleted hook records its own thread id, and that id must
    differ from the event loop thread id this test itself is running on.
    """
    db, user = _db_session
    event_loop_thread_id = threading.get_ident()
    observed_thread_ids: list[int] = []

    def hook(db, user_id, connector_type, connector_id):
        observed_thread_ids.append(threading.get_ident())
        return ConnectorDeleteDecision()

    app = PublicMCPApp(app_id="calendar", name="Calendar", transport="stdio")
    db.add(app)
    db.commit()
    server = MCPServer.from_config(
        {
            "name": "calendar",
            "managed": "external",
            "transport": "stdio",
            "auth": {"app_id": "calendar"},
        }
    )
    db.add(server)
    db.flush()
    association = UserMCPServer(
        user_id=user.id,
        mcpserver_id=server.id,
        is_owner=True,
        can_delete=True,
        is_active=True,
    )
    db.add(association)
    db.commit()
    db.refresh(app)
    db.refresh(association)

    with snapshot_connector_team_hooks():
        set_connector_team_hooks(deleted=hook)
        await teardown_mcp_app_server(
            int(server.id),
            app_id="calendar",
            expected_provider_name=None,
            expected_catalog_generation=app.generation,
            expected_association_generation=association.lifecycle_generation,
            current_user=user,
            db=db,
        )

    assert observed_thread_ids, "the connector team hook was never called"
    assert event_loop_thread_id not in observed_thread_ids

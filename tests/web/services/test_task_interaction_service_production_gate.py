"""Zero-production-caller gate for ``task_interaction_service.create`` and
``task_interaction_service.respond``.

Both names are typed seams with no call body wired to them yet: ``create()``
validates and returns a typed outcome without ever staging a row, and
``respond()`` does not exist as a callable at all in this delivery. A static
test asserts that no production module calls either name through this
module, so the seam cannot grow a caller before its production behavior is
actually implemented and tested.

Retired by the change that routes the existing resume coordinator through
this module's compatibility seam. That change is what makes ``respond()``
reachable in production: the seam refuses a legacy resume on a task that
holds an active native interaction row, so from that change on the only way
such a task can be answered is a caller that goes through ``respond()``. The
first HTTP caller arrives separately, with the endpoint issue, and does not
retire anything here. ``create()`` is not covered by this gate's retirement
at all: its production call body arrives with the wiring batch that also
fills in ``stage_interaction_request``'s caller obligations -- the change
that deletes this gate file must add a replacement guard that locks
``create()`` alone, or the seam stops being provably a seam the moment this
file goes away.

AST-based, module-qualified matching -- not a bare-name or substring scan.
``create`` and ``respond`` are ordinary English verbs already used as method
names elsewhere in this codebase (``docker_client.containers.create(...)``,
unrelated ``.respond(...)`` calls); a scanner that flagged every call named
``create`` or ``respond`` would be red on day one. This gate only counts a
call when the callee name is resolved -- through an import binding recorded
in the same module -- to ``task_interaction_service.create`` or
``task_interaction_service.respond`` specifically. This is a different scan
shape from the staging primitive's own gate
(``test_interaction_staging_production_gate.py``), which gates on two names
that are unique enough in this codebase for a bare attribute match to be
safe. ``create`` and ``respond`` are not unique, so this gate additionally
tracks which local names are bound to this module before it will count a
call against them; merely importing this module, or importing some other
name from it (``InteractionPrincipal``, an outcome type, the shared
ownership predicate), never trips this gate on its own.

Known blind spots, not fixed here because closing them is out of scope for a
static AST scan of one package tree (the staging gate documents the same
class of gaps for the same reason):

(a) Dynamic access: ``importlib.import_module(...)`` plus ``getattr(...)``
    matches none of the node shapes below.
(b) A gated name reached through an alias chain the AST walk does not
    resolve -- e.g. binding ``create`` to a local variable and calling the
    variable, or re-exporting it from a third module and importing that
    name elsewhere.
(c) The primitive-module exclusion in ``_production_modules`` is
    stem-keyed (``path.stem != SERVICE_MODULE``), not path-keyed: any other
    file anywhere under this package that happened to also be named
    ``task_interaction_service.py`` would be excluded from the scan right
    alongside the real one, on filename alone.
(d) The scan root is ``xagent.__path__`` (``src/xagent``), not the
    repository root -- a caller under the top-level ``scripts/`` directory
    (outside that tree) is invisible to this gate entirely.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import xagent
from xagent.web.services import (  # noqa: F401 -- negative control import
    task_interaction_service as service,
)

GATED_NAMES = frozenset({"create", "respond"})
SERVICE_MODULE = "task_interaction_service"


def _dotted_prefix(node: ast.expr) -> str | None:
    """Flatten a ``Name``/``Attribute`` chain to a dotted string, e.g. the
    ``a.b.c`` in ``a.b.c.create(...)``. Returns ``None`` for anything that
    is not a plain attribute chain (a call, subscript, etc. anywhere in it),
    which this gate then treats as unresolved rather than guessed at.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_prefix(node.value)
        return f"{prefix}.{node.attr}" if prefix is not None else None
    return None


def _production_uses(source: str) -> set[str]:
    """Gated names this module actually calls through a binding to
    ``task_interaction_service``, resolved in two passes: first collect
    every import binding this module makes to the service module or to its
    gated names directly, then walk every call and count it only if its
    callee resolves through one of those bindings.
    """
    tree = ast.parse(source)

    # Local name -> gated name it refers to, from a direct from-import
    # (``from ...task_interaction_service import create``) or an aliased
    # one (``... import create as make_interaction``).
    direct_bindings: dict[str, str] = {}
    # Local names bound to the service module itself (``from ..services
    # import task_interaction_service`` or ``import ...task_interaction_service
    # as svc``), usable for an attribute call.
    module_bindings: set[str] = set()
    star_imported = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_path = (node.module or "").split(".")
            if module_path and module_path[-1] == SERVICE_MODULE:
                for alias in node.names:
                    if alias.name == "*":
                        star_imported = True
                    elif alias.name in GATED_NAMES:
                        direct_bindings[alias.asname or alias.name] = alias.name
            else:
                for alias in node.names:
                    if alias.name == SERVICE_MODULE:
                        module_bindings.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] == SERVICE_MODULE:
                    # ``import a.b.task_interaction_service`` (no ``as``)
                    # binds only the top-level package name in real Python
                    # scoping; an aliased import binds a directly usable
                    # name. Either way, record the name that is actually
                    # reachable in this module's namespace -- resolving the
                    # unaliased dotted-chain call site itself is blind spot
                    # (b) above.
                    module_bindings.add(alias.asname or alias.name.split(".")[0])

    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in direct_bindings:
                found.add(direct_bindings[func.id])
            elif star_imported and func.id in GATED_NAMES:
                found.add(func.id)
        elif isinstance(func, ast.Attribute) and func.attr in GATED_NAMES:
            if isinstance(func.value, ast.Name) and func.value.id in module_bindings:
                found.add(func.attr)
            else:
                dotted = _dotted_prefix(func.value)
                if dotted is not None and dotted.split(".")[-1] == SERVICE_MODULE:
                    found.add(func.attr)
    return found


def _production_modules() -> list[Path]:
    """Every source file under the installed xagent package, excluding this
    service module's own definition file. See
    ``test_interaction_staging_production_gate.py``'s twin for why the walk
    starts at ``xagent.__path__`` and asserts the scan set is non-empty."""

    root = Path(next(iter(xagent.__path__)))
    modules = [path for path in root.rglob("*.py") if path.stem != SERVICE_MODULE]
    assert modules, "production scan set is empty"
    return modules


# --------------------------------------------------------------------------
# T-GATE-1: the gate itself
# --------------------------------------------------------------------------


def test_no_production_module_calls_create_or_respond() -> None:
    offenders: dict[str, set[str]] = {}
    for path in _production_modules():
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"{path}: not valid UTF-8, cannot be AST-scanned by this gate"
            ) from exc
        uses = _production_uses(source)
        if uses:
            offenders[str(path)] = uses
    assert offenders == {}, offenders


# --------------------------------------------------------------------------
# T-GATE-2: positive controls -- each call shape must be individually
# detected once it is bound to this module
# --------------------------------------------------------------------------


def test_detects_direct_call_after_from_import() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services.task_interaction_service import create

        def handler(db, **kwargs):
            return create(db, **kwargs)
        """
    )
    assert _production_uses(source) == {"create"}


def test_detects_aliased_direct_call() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services.task_interaction_service import create as make

        def handler(db, **kwargs):
            return make(db, **kwargs)
        """
    )
    assert _production_uses(source) == {"create"}


def test_detects_module_attribute_call() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services import task_interaction_service

        def handler(db, **kwargs):
            return task_interaction_service.respond(db, **kwargs)
        """
    )
    assert _production_uses(source) == {"respond"}


def test_detects_aliased_module_attribute_call() -> None:
    source = textwrap.dedent(
        """
        import xagent.web.services.task_interaction_service as svc

        def handler(db, **kwargs):
            return svc.create(db, **kwargs)
        """
    )
    assert _production_uses(source) == {"create"}


def test_detects_star_import_call() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services.task_interaction_service import *

        def handler(db, **kwargs):
            return respond(db, **kwargs)
        """
    )
    assert _production_uses(source) == {"respond"}


# --------------------------------------------------------------------------
# T-GATE-3: negative controls -- the whole reason this gate cannot reuse the
# staging gate's bare-name shape
# --------------------------------------------------------------------------


def test_bare_module_import_alone_is_not_flagged() -> None:
    """Unlike the staging gate, merely referencing this module -- with no
    call to ``create``/``respond`` bound through it -- must not trip the
    gate: production code (the shared ownership predicate's consumer in
    ``public_chat_access.py``) legitimately imports other names from this
    module in this same delivery."""

    source = "from xagent.web.services import task_interaction_service\n"
    assert _production_uses(source) == set()


def test_importing_unrelated_name_is_not_flagged() -> None:
    source = textwrap.dedent(
        """
        from xagent.web.services.task_interaction_service import InteractionPrincipal

        def handler(**kwargs):
            return InteractionPrincipal(**kwargs)
        """
    )
    assert _production_uses(source) == set()


def test_same_named_call_on_an_unrelated_object_is_not_flagged() -> None:
    """The whole point of module-qualified matching: a call named
    ``create`` or ``respond`` on an object with no binding to this module
    must not match, or this gate would be red against half the codebase."""

    source = textwrap.dedent(
        """
        def handler(docker_client, ticket):
            docker_client.containers.create(image="x")
            return ticket.respond(status=200)
        """
    )
    assert _production_uses(source) == set()


def test_test_module_consumers_are_not_flagged() -> None:
    """Test consumers are allowed: this file, and any other test module
    that imports the gated names, live under ``tests/``, which
    ``_production_modules`` never walks -- it starts from
    ``xagent.__path__`` (``src/xagent``), not the repository root."""

    scanned = {str(path) for path in _production_modules()}
    this_file = str(Path(__file__).resolve())
    assert this_file not in scanned
    assert all("/tests/" not in path for path in scanned)


def test_service_module_itself_is_excluded_from_the_scan_set() -> None:
    modules = _production_modules()
    assert all(path.stem != SERVICE_MODULE for path in modules)

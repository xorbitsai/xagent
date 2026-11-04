"""Zero-production-caller gate for ``task_interaction_service.create`` and
``respond``.

Replaces the two-name gate this file's predecessor enforced, keeping both
names under watch. The predecessor named its retirement condition as "the
change that routes the existing resume coordinator through this module's
compatibility seam" -- the change meant to give ``respond()`` its first
production caller. That compatibility seam has since landed: it reads
through ``task_interaction_close.active_interaction_id_sync``, which
imports ``_active_native_row_criteria`` from this module (``websocket.py``
itself no longer imports it directly). It does not,
though, give ``respond()`` a caller -- the seam only reads that filter
predicate, it never calls ``respond()`` -- so this gate's retirement
condition is still open. Zero production code anywhere in this package
calls either name, and this gate is what keeps that true until the change
that wires a caller retires the name it wires.

``create()``'s production call body -- the write that actually calls
``stage_interaction_request`` -- arrives with the change that wires
interaction creation end-to-end and fills in that primitive's caller
obligations. Retiring this gate is that change's job, not this one's: the
change that deletes this file must add whatever replacement guard it needs,
the same way this file replaced its own predecessor.

AST-based, module-qualified matching -- not a bare-name or substring scan.
``create`` is an ordinary English verb already used as a method name
elsewhere in this codebase (``docker_client.containers.create(...)``); a
scanner that flagged every call named ``create`` would be red on day one.
This gate only counts a call when the callee name is resolved -- through an
import binding recorded in the same module -- to one of the gated names,
``task_interaction_service.create`` or ``respond``.

Known blind spots, not fixed here because closing them is out of scope for a
static AST scan of one package tree (identical to the predecessor gate's own
list, for the identical reasons):

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
(e) A gated name used in value position rather than called directly --
    passed as an argument (``Depends(create)``,
    ``functools.partial(create, ...)``), a decorator reference, or a
    dict value -- is invisible to this gate. The walk below only inspects
    ``ast.Call.func`` on each ``Call`` node it visits; it never inspects a
    call's ``args``/``keywords``, so a gated name reachable only as an
    argument is never counted.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

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
# The gate itself
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
# Positive controls -- each call shape must be individually detected once it
# is bound to this module
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
            return task_interaction_service.create(db, **kwargs)
        """
    )
    assert _production_uses(source) == {"create"}


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
            return create(db, **kwargs)
        """
    )
    assert _production_uses(source) == {"create"}


# --------------------------------------------------------------------------
# Negative controls -- the whole reason this gate cannot reuse the staging
# gate's bare-name shape
# --------------------------------------------------------------------------


def test_bare_module_import_alone_is_not_flagged() -> None:
    """Merely referencing this module -- with no call to ``create`` bound
    through it -- must not trip the gate: production code (the shared
    ownership predicate's consumer in ``public_chat_access.py``, which
    imports ``InteractionPrincipal``, ``public_chat_identity_matches``, and
    ``task_is_owned_by_public_principal``) legitimately imports other names
    from this module. The same holds for both gated names: importing the
    module, or even binding ``create`` or ``respond`` without calling
    either, is not a call -- only a call expression through a traced
    binding trips the gate."""

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


def test_detects_a_production_call_to_respond() -> None:
    """``respond()`` is under this gate for the same reason ``create()``
    is: it has no production caller, and the change that gives it one is
    the change that takes it out of here. This is the positive control for
    that half of the gate -- a hypothetical production call must be
    detected, or the gate would be watching a name it cannot see."""

    source = textwrap.dedent(
        """
        from xagent.web.services.task_interaction_service import respond

        def handler(**kwargs):
            return respond(**kwargs)
        """
    )
    assert _production_uses(source) == {"respond"}


def test_same_named_call_on_an_unrelated_object_is_not_flagged() -> None:
    """The whole point of module-qualified matching: a call named
    ``create`` on an object with no binding to this module must not match,
    or this gate would be red against half the codebase."""

    source = textwrap.dedent(
        """
        def handler(docker_client):
            docker_client.containers.create(image="x")
        """
    )
    assert _production_uses(source) == set()


def test_test_module_consumers_are_not_flagged() -> None:
    """Test consumers are allowed: this file, and any other test module
    that imports the gated name, live under ``tests/``, which
    ``_production_modules`` never walks -- it starts from
    ``xagent.__path__`` (``src/xagent``), not the repository root."""

    scanned = {str(path) for path in _production_modules()}
    this_file = str(Path(__file__).resolve())
    assert this_file not in scanned
    assert all("/tests/" not in path for path in scanned)


def test_service_module_itself_is_excluded_from_the_scan_set() -> None:
    modules = _production_modules()
    assert all(path.stem != SERVICE_MODULE for path in modules)


# --------------------------------------------------------------------------
# Forward-verification: adding a real production caller of ``create`` must
# turn this gate red, not just its unit-level ``_production_uses`` checks
# above. ``_production_modules`` walks ``xagent.__path__`` rather than a
# hardcoded root, so pointing that at a scratch directory for the duration
# of this one test exercises the exact same scan code the real gate runs,
# without writing a canary file into the actual ``src/xagent`` tree -- a
# crash between planting and cleanup there would leave a stray file behind
# and make every later run of the real gate red for an unrelated reason.
# --------------------------------------------------------------------------


def test_gate_turns_red_when_a_production_caller_is_added(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xagent, "__path__", [str(tmp_path)])
    planted = tmp_path / "_zzz_test_task_interaction_service_create_gate_canary.py"
    planted.write_text(
        textwrap.dedent(
            """
            from xagent.web.services.task_interaction_service import create


            def handler(db, **kwargs):
                return create(db, **kwargs)
            """
        ),
        encoding="utf-8",
    )

    offenders: dict[str, set[str]] = {}
    for path in _production_modules():
        uses = _production_uses(path.read_text(encoding="utf-8"))
        if uses:
            offenders[str(path)] = uses
    assert offenders == {str(planted): {"create"}}

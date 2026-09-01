"""Three static guards that replace the retired zero-production-caller gate
(``test_interaction_staging_production_gate.py``) now that ``create()``
supplies that gate's call body.

The retired gate watched two names (``stage_interaction_request`` and
``interaction_handoff``) for *any* production use at all, because the
scanner it used cannot be narrowed to "only this one caller" -- its own
docstring documents why (a bare module import trips it unconditionally,
outside the ``if ... == PRIMITIVE_MODULE:`` guard that scopes the
``ImportFrom`` check). Once ``create()`` becomes the first legitimate
caller, that all-or-nothing shape stops being useful, and it is replaced
here by guards that assert the *shape* of the one caller that exists
instead of the absence of every caller:

* ``test_handoff_is_entered_only_from_the_create_seam`` -- the production
  use points of ``interaction_handoff`` are exactly
  ``{task_interaction_service}``.
* ``test_create_validates_before_entering_the_handoff`` -- validation
  always runs, and always runs before the handoff, on every call.
* ``test_staging_module_import_surface_is_closed`` -- the set of modules
  importing anything from ``task_interaction_staging`` is exactly the
  three that need to (unchanged by this delivery -- see ``create()``'s own
  docstring on what it reuses).

Window-not-lost note: the static create()/respond() production-caller gate
(``test_task_interaction_service_create_gate.py``) still separately watches
``create()``/``respond()`` for a *production caller* -- neither name has one
yet in this delivery. These three guards watch a narrower thing: given that
``create()``'s own call body now exists, does it reach the staging
primitive only the one sanctioned way. The two gates are not redundant;
together they are what keeps this delivery's "no production caller, and
what caller entry it would use is well-formed" window closed from both
ends.
"""

from __future__ import annotations

import ast
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent
from tests.web.services.task_interaction_schema_shared import (
    anchor_event_id,
    make_task,
    make_trace_event,
    make_user,
)
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import task_interaction_service as svc
from xagent.web.services.task_interaction_staging import InteractionAnchor
from xagent.web.services.task_lease_service import TaskLease

# ---------------------------------------------------------------------------
# Shared AST scanning helpers -- same production-tree walk the retired gate
# used, adapted to a per-name "which modules use this" query instead of a
# yes/no "does anything use this" one.
# ---------------------------------------------------------------------------


def _production_modules(exclude_stem: str) -> list[Path]:
    root = Path(next(iter(xagent.__path__)))
    modules = [path for path in root.rglob("*.py") if path.stem != exclude_stem]
    assert modules, "production scan set is empty"
    return modules


def _module_stem_from_import(node: ast.ImportFrom, target_stem: str) -> bool:
    return (node.module or "").split(".")[-1] == target_stem


# ---------------------------------------------------------------------------
# Guard 1: interaction_handoff's production use points are exactly
# {task_interaction_service}, and Guard 1b: stage_interaction_request has no
# direct production caller at all -- create()'s own docstring says it
# reaches the row through interaction_handoff, "never
# stage_interaction_request directly".
#
# One shared scanner, watching each name separately rather than as one set:
# interaction_handoff has exactly one legitimate production use point
# (create()'s seam), while stage_interaction_request has none. The two
# names are never checked against a merged "either name" set for that
# reason -- a module calling the primitive directly must not be hidden by
# a passing handoff check on the same line.
# ---------------------------------------------------------------------------

_STAGING_MODULE = "task_interaction_staging"
_HANDOFF = "interaction_handoff"
_PRIMITIVE = "stage_interaction_request"


def _names_used_by(source: str, watched: frozenset[str]) -> set[str]:
    """Which watched names this source imports or calls.

    Four shapes, each independently detected -- the retired gate and this
    file's first version each saw only a subset:

      from ...task_interaction_staging import X      (ImportFrom, per alias)
      from ...task_interaction_staging import *      (ImportFrom, star)
      import xagent...task_interaction_staging [as m] (Import)
      X(...) / anything.X(...)                        (Call, Name/Attribute)

    The Import shape matters because it needs no ImportFrom at all: a
    module can reach the primitive through `import ... as s` plus
    `s.stage_interaction_request(...)`, which an ImportFrom-only scanner
    cannot see. The blind spots the retired gate documented and did not
    close (importlib + getattr, alias chains through a third module) stay
    open here for the same reason: they are out of scope for a static AST
    scan.
    """

    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if _module_stem_from_import(node, _STAGING_MODULE):
                found.update(a.name for a in node.names if a.name in watched)
                if any(a.name == "*" for a in node.names):
                    found.update(watched)
        elif isinstance(node, ast.Import):
            if any(a.name.split(".")[-1] == _STAGING_MODULE for a in node.names):
                # A bare module import binds every watched name behind an
                # attribute access; the Call arm below is what says which
                # one is actually reached, so this arm records the module,
                # not the names.
                found.add(_STAGING_MODULE)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in watched:
                found.add(func.id)
            elif isinstance(func, ast.Attribute) and func.attr in watched:
                found.add(func.attr)
    return found


def _modules_using(root: Path, watched: frozenset[str]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in root.rglob("*.py"):
        if path.stem == _STAGING_MODULE:
            continue  # the primitives' own definition module
        uses = _names_used_by(path.read_text(encoding="utf-8"), watched)
        if uses:
            found[path.stem] = uses
    return found


def test_handoff_is_entered_only_from_the_create_seam() -> None:
    root = Path(next(iter(xagent.__path__)))
    assert set(_modules_using(root, frozenset({_HANDOFF}))) == {
        "task_interaction_service"
    }


def test_the_staging_primitive_has_no_direct_production_caller() -> None:
    """The half of the retired gate this file's first version did not
    replace. create()'s own docstring says it reaches the row through
    interaction_handoff, "never stage_interaction_request directly"; this
    is the only thing that holds that sentence to account.

    Scans for the NAME, not for which modules import the staging module:
    task_interaction_service is already an allowed importer (Guard 3
    below), so a module-level "does it import from staging" check stays
    green while that very module starts calling the primitive directly
    (measured before this guard existed -- a synthetic module named
    task_interaction_service that called stage_interaction_request via a
    bare module import passed both of the guards that existed at the
    time)."""

    root = Path(next(iter(xagent.__path__)))
    assert _modules_using(root, frozenset({_PRIMITIVE})) == {}


@pytest.mark.parametrize(
    ("source", "watched", "expected"),
    [
        pytest.param(
            "from xagent.web.services.task_interaction_staging import"
            " stage_interaction_request\n"
            "def h(db, **kw):\n"
            "    return stage_interaction_request(db, **kw)\n",
            frozenset({_PRIMITIVE}),
            {_PRIMITIVE},
            id="from_import_and_direct_call",
        ),
        pytest.param(
            "from xagent.web.services.task_interaction_staging import"
            " interaction_handoff\n"
            "def h(db, lease, task, anchor, now):\n"
            "    with interaction_handoff(db, lease, task=task, anchor=anchor,"
            " now=now):\n"
            "        pass\n",
            frozenset({_HANDOFF}),
            {_HANDOFF},
            id="context_manager_call_by_name",
        ),
        pytest.param(
            "from xagent.web.services.task_interaction_staging import *\n",
            frozenset({_HANDOFF, _PRIMITIVE}),
            {_HANDOFF, _PRIMITIVE},
            id="star_import_binds_both",
        ),
        pytest.param(
            "import xagent.web.services.task_interaction_staging as s\n"
            "def h(db, **kw):\n"
            "    return s.stage_interaction_request(db, **kw)\n",
            frozenset({_PRIMITIVE}),
            {_STAGING_MODULE, _PRIMITIVE},
            id="bare_import_alias_and_attribute_call",
        ),
        pytest.param(
            "import xagent.web.services.task_interaction_staging\n"
            "def h(db, **kw):\n"
            "    return xagent.web.services.task_interaction_staging."
            "stage_interaction_request(db, **kw)\n",
            frozenset({_PRIMITIVE}),
            {_STAGING_MODULE, _PRIMITIVE},
            id="bare_import_fully_qualified_attribute_call",
        ),
    ],
)
def test_scanner_positive_controls(
    source: str, watched: frozenset[str], expected: set[str]
) -> None:
    """Each import/call shape must be individually detected. The last two
    are the ones an ImportFrom-only scanner cannot see at all -- measured
    against the module-level scanner this replaces (Guard 3's
    ``_modules_importing_from_staging``), a module written that way passed
    that scanner while directly calling the primitive."""

    assert _names_used_by(source, watched) == expected


def test_scanner_negative_control_unrelated_module_is_clean() -> None:
    """The scanner must not fire on a module that merely mentions the names
    in prose or defines something similarly named -- otherwise the real-tree
    assertions above would be passing for the wrong reason."""

    source = (
        '"""Prose mentioning stage_interaction_request and'
        ' interaction_handoff."""\n'
        "def interaction_handoff_helper_name_only():\n"
        "    return None\n"
    )
    assert _names_used_by(source, frozenset({_HANDOFF, _PRIMITIVE})) == set()


# ---------------------------------------------------------------------------
# Guard 2: (a) static -- the validation call sits at the function body's top
# level, never inside an If/For/While, and precedes the With statement that
# enters interaction_handoff; (b) behavioral -- a bad payload never reaches
# the handoff and stages nothing.
#
# Hard rule: (a) alone is not enough. Line-number order does not
# prove a statement always runs -- moving validation into an `if False:`
# branch, or a `try` that is itself unreachable, would keep any purely
# textual/line-number check green while skipping the check entirely. (a)
# below therefore walks the AST's actual nesting (not source line numbers)
# and requires the validation call be a statement directly in the function
# body's own top-level list (allowing at most one wrapping `Try`, itself
# also top-level) -- never inside an `If`/`For`/`While` at any depth. (b) is
# what actually has discriminating power: it proves the property with a
# live call, not a shape assertion about the source.
# ---------------------------------------------------------------------------


def test_create_validates_before_entering_the_handoff_static() -> None:
    import inspect

    source = inspect.getsource(svc.create)
    tree = ast.parse(textwrap.dedent(source))
    func_def = tree.body[0]
    assert isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef))

    def _call_name(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    with_index: int | None = None
    validate_index: int | None = None
    for index, stmt in enumerate(func_def.body):
        if isinstance(stmt, ast.With):
            with_index = index
        # The validation call, and every other guard/pre-check statement in
        # create()'s body, must not be reachable only through a
        # conditional -- walk each top-level statement (including into a
        # top-level Try's body) and refuse If/For/While nesting.
        forbidden = [
            node
            for node in ast.walk(stmt)
            if isinstance(node, (ast.If, ast.For, ast.While))
        ]
        names_here = {
            _call_name(node)
            for node in ast.walk(stmt)
            if _call_name(node) == "validate_v1_write_payload"
        }
        if names_here:
            assert not forbidden, (
                "validate_v1_write_payload is reachable only through a "
                f"conditional: {ast.dump(forbidden[0])[:120]}"
            )
            validate_index = index

    assert with_index is not None, "create() must call interaction_handoff via `with`"
    assert validate_index is not None, "validate_v1_write_payload call not found"
    assert validate_index < with_index, (
        "validate_v1_write_payload must run before the `with interaction_handoff` "
        "statement"
    )


@pytest.fixture
def _engine(tmp_path: Path):
    db_path = tmp_path / "handoff_surface.db"
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def _db(_engine):
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def _seed_system_call(db: Any) -> dict[str, Any]:
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-a"
    task.source = "widget"
    task.lease_attempt_id = "attempt-1"
    db.commit()
    db.refresh(task)

    trace_event_id = make_trace_event(db, task_id=task_id)
    anchor = InteractionAnchor(
        trace_event_id=trace_event_id,
        resume_event_id=anchor_event_id(db, trace_event_id),
        resume_execution_id="exec-1",
        resume_run_partition="run-a",
    )
    lease = TaskLease(
        task_id=int(task.id),
        runner_id="runner-1",
        run_id="run-a",
        attempt_id="attempt-1",
    )
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=24)
    principal = svc.InteractionPrincipal(
        kind=svc.InteractionPrincipalKind.SYSTEM,
        user_id=None,
        is_admin=False,
        auth_mode=None,
    )
    return {
        "task": task,
        "anchor": anchor,
        "lease": lease,
        "now": now,
        "expires_at": expires_at,
        "principal": principal,
    }


def test_create_validates_before_entering_the_handoff_behavioral(_db: Any) -> None:
    ctx = _seed_system_call(_db)
    bad_envelope = svc.CreateInteractionEnvelope(
        kind="clarification",
        protocol_version=1,
        request_idempotency_key="behavioral-guard-key",
        values={"message": "", "interactions": []},  # blank message -> refused
        ttl_seconds=None,
    )
    with mock.patch(
        "xagent.web.services.task_interaction_service.interaction_handoff"
    ) as mocked_handoff:
        outcome = svc.create(
            _db,
            task_id=int(ctx["task"].id),
            principal=ctx["principal"],
            envelope=bad_envelope,
            system_context=svc.SystemWriteContext(
                task=ctx["task"],
                lease=ctx["lease"],
                anchor=ctx["anchor"],
                now=ctx["now"],
                expires_at=ctx["expires_at"],
            ),
        )
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")
    mocked_handoff.assert_not_called()
    assert _db.query(TaskInteractionRequest).count() == 0


# ---------------------------------------------------------------------------
# Guard 3: the staging module's import surface is exactly the three modules
# that need it, unchanged by this delivery -- the only new user,
# task_interaction_service, was already in the set.
# ---------------------------------------------------------------------------


def _modules_importing_from_staging(root: Path) -> set[str]:
    found: set[str] = set()
    for path in root.rglob("*.py"):
        if path.stem == "task_interaction_staging":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _module_stem_from_import(
                node, "task_interaction_staging"
            ):
                found.add(path.stem)
    return found


def test_staging_module_import_surface_is_closed() -> None:
    """Which modules depend on the staging module at all -- a dependency-
    surface guard, not a name-use guard. It does not replace
    ``test_the_staging_primitive_has_no_direct_production_caller`` above:
    task_interaction_service is already in the set this asserts, so this
    guard alone would stay green the moment that very module started
    calling stage_interaction_request directly -- the failure Guard 1b
    exists to catch."""

    root = Path(next(iter(xagent.__path__)))
    assert _modules_importing_from_staging(root) == {
        "task_interaction_anchor",
        "task_clarification_draft",
        "task_interaction_service",
    }


# ---------------------------------------------------------------------------
# Carried over verbatim from the retired gate file (test_interaction_staging
# _production_gate.py): an unrelated invariant that happened to live in
# that file (the staging module must consume the model's origin vocabulary
# rather than redeclare it) and has no connection to the zero-caller gate
# being retired here. Preserved rather than deleted as collateral damage.
# ---------------------------------------------------------------------------


def test_staging_module_defines_no_second_vocabulary_copy() -> None:
    """The staging module must consume the model's column-domain constants,
    never redeclare them. The merge-order rule for the origin vocabulary is
    that the model owns the one literal list (INTERACTION_ORIGIN_VOCABULARY,
    INTERACTION_PROTOCOL_VERSION); every other surface derives from it. This
    guard fails if task_interaction_staging.py grows its own literal set,
    tuple, or list re-enumerating vocabulary members, its own integer
    protocol-version constant, or an inline origin-normalization function --
    each of which would be a drift-capable second copy.
    """
    import xagent.web.services.task_interaction_staging as staging_module

    source = Path(staging_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    from xagent.web.models.task_interaction import INTERACTION_ORIGIN_VOCABULARY

    for node in ast.walk(tree):
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            literals = {
                el.value
                for el in node.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            }
            overlap = literals & INTERACTION_ORIGIN_VOCABULARY
            assert not overlap, (
                f"literal collection re-enumerates origin vocabulary members "
                f"{sorted(overlap)}; derive from the model's "
                f"INTERACTION_ORIGIN_VOCABULARY instead"
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            assert not ("normalize" in name and "origin" in name), (
                f"{node.name} looks like inline origin normalization; the "
                "model module owns normalize_interaction_origin"
            )
    aliases = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"_PROTOCOL_VERSION", "_ORIGIN_VOCABULARY"}
    }
    assert set(aliases) == {"_PROTOCOL_VERSION", "_ORIGIN_VOCABULARY"}
    for name, value in aliases.items():
        assert isinstance(value, ast.Name), (
            f"{name} must alias the model's constant by name, got "
            f"{ast.dump(value)[:80]}"
        )
    assert aliases["_ORIGIN_VOCABULARY"].id == "INTERACTION_ORIGIN_VOCABULARY"
    assert aliases["_PROTOCOL_VERSION"].id == "INTERACTION_PROTOCOL_VERSION"

"""Contract tests for ``InteractionHandoff``'s public surface: the
``staged`` observability field, the ``__all__`` public boundary, and the
zero-direct-construction guard the public rename makes necessary.

This is a companion to ``test_interaction_staging.py``, not a replacement
for any of it -- the six-swallowed-exception degrade matrix, the savepoint
containment group, and every other primitive-level behavior stay pinned
there. This file isolates the surface that changed here: the renamed
public class, the field that replaced ``_staged_row``, and the explicit
``__all__`` list -- kept separate so a reviewer looking for "what did this
change touch" does not have to diff a 2000-line file to find it.
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.services.interaction_static_scan_shared import _scan_root
from tests.web.services.task_interaction_schema_shared import (
    make_task,
    make_trace_event,
    make_user,
)
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.services import ops_signals
from xagent.web.services import task_interaction_staging as staging
from xagent.web.services.task_interaction_staging import (
    InteractionAnchor,
    StagedInteractionRequest,
    interaction_handoff,
    stage_interaction_request,
)
from xagent.web.services.task_lease_service import TaskLease

_key_counter = count()


def _engine(tmp_path: Path):
    db_path = tmp_path / "interaction_handoff_surface.db"
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


def _session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _seed(session_factory) -> tuple[int, int]:
    db = session_factory()
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    anchor_id = make_trace_event(db, task_id=task_id)
    db.close()
    return task_id, anchor_id


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _anchor(trace_event_id: int, **overrides: Any) -> InteractionAnchor:
    values: dict[str, Any] = {
        "trace_event_id": trace_event_id,
        "resume_event_id": "resume-event-1",
        "resume_execution_id": "resume-exec-1",
        "resume_run_partition": "run-a",
    }
    values.update(overrides)
    return InteractionAnchor(**values)


def _lease(task_id: int, *, run_id: str = "run-a") -> TaskLease:
    return TaskLease(
        task_id=task_id, runner_id="runner-1", run_id=run_id, attempt_id=None
    )


def _next_key() -> str:
    return f"key-{next(_key_counter)}"


def _stage_kwargs(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "kind": "clarification",
        "protocol_version": 1,
        "request_payload": {"prompt": "example"},
        "request_idempotency_key": _next_key(),
        "expires_at": _now() + timedelta(minutes=15),
    }
    values.update(overrides)
    return values


@pytest.fixture(autouse=True)
def _reset_ops_signals():
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)
    yield
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)


# ---------------------------------------------------------------------------
# T-H-1: success -- handoff.staged is stage()'s own return value, the same
# object, not merely an equal one.
# ---------------------------------------------------------------------------


def test_th1_staged_holds_the_row_stage_returned(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        result = h.stage(**_stage_kwargs())
        # Asserted *inside* the with-block, before the savepoint commits, so
        # this pins the object identity stage() itself hands back -- not
        # some later re-derivation after commit.
        assert h.staged is result
        assert isinstance(h.staged, StagedInteractionRequest)

    db.commit()
    assert h.staged is result
    db.close()


# ---------------------------------------------------------------------------
# T-H-2: zero stage() calls is legal -- documented, not an error path.
# ---------------------------------------------------------------------------


def test_th2_zero_stage_calls_leaves_staged_none(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        pass

    db.commit()
    assert h.staged is None
    db.close()


# ---------------------------------------------------------------------------
# T-H-3: degraded -- staged stays None, and the caller's code after the
# with-block runs normally (the with-block itself never raises).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["slot-taken", "run-partition-mismatch"])
def test_th3_degraded_stage_leaves_staged_none(tmp_path: Path, case: str) -> None:
    """Two of the six swallowed types, not all six -- test_interaction_staging.py's
    T-CM-1 already parametrizes the full six-plus-replay matrix for
    degrade-vs-signal-vs-caller-write behavior; this only needs enough
    cells to prove ``staged`` behaves the same way regardless of which
    signal fired, per the module's own dispatch table
    (``_DEGRADATION_SIGNALS``). ``InteractionRunPartitionMismatch`` is
    included specifically because it registers the *other* signal
    (``INTERACTION_RUN_PARTITION_MISMATCH_DEGRADED``), not
    ``INTERACTION_HANDOFF_DEGRADED`` -- proving ``staged`` does not depend
    on which signal a given swallowed type maps to.
    """

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)

    if case == "slot-taken":
        stage_interaction_request(
            db,
            task_id=task_id,
            run_id=lease.run_id,
            anchor=anchor,
            origin="internal",
            now=_now(),
            **_stage_kwargs(),
        )
        db.commit()
    else:
        assert case == "run-partition-mismatch"
        anchor = _anchor(anchor_id, resume_run_partition="some-other-run")

    task = db.get(Task, task_id)
    with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
        h.stage(**_stage_kwargs())
    db.commit()

    assert h.staged is None
    # The with-block above completed without raising -- reaching this line
    # at all is part of the assertion; a caller's code after the block runs
    # exactly the way it does on a clean exit.
    db.close()


# ---------------------------------------------------------------------------
# T-H-5: equivalence -- the two misuse messages must still tell "a row was
# staged" apart from "no row was staged" after _staged_row's removal.
# ---------------------------------------------------------------------------


def test_th5a_misuse_message_after_a_successful_stage_names_a_row(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with pytest.raises(
        staging.InteractionHandoffMisuse,
        match="the staged interaction row no longer exists",
    ):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()) as h:
            h.stage(**_stage_kwargs())
            db.commit()

    db.close()


def test_th5b_misuse_message_after_zero_stage_calls_names_no_row(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    task_id, anchor_id = _seed(session_factory)
    db = session_factory()
    anchor = _anchor(anchor_id)
    lease = _lease(task_id)
    task = db.get(Task, task_id)

    with pytest.raises(
        staging.InteractionHandoffMisuse,
        match="no interaction row was staged",
    ):
        with interaction_handoff(db, lease, task=task, anchor=anchor, now=_now()):
            db.commit()

    db.close()


# ---------------------------------------------------------------------------
# T-H-6: __all__ -- every name resolves, none are private, InteractionHandoff
# is in it.
# ---------------------------------------------------------------------------


def test_th6_all_names_resolve_and_are_public() -> None:
    assert staging.__all__, "task_interaction_staging.__all__ must not be empty"
    for name in staging.__all__:
        assert not name.startswith("_"), (
            f"{name!r} in __all__ starts with an underscore"
        )
        assert hasattr(staging, name), (
            f"{name!r} is in __all__ but not defined on the module"
        )
    assert "InteractionHandoff" in staging.__all__
    assert len(staging.__all__) == len(set(staging.__all__)), (
        "duplicate name in __all__"
    )


# ---------------------------------------------------------------------------
# T-H-7: __all__ -- completeness, not just soundness. T-H-6 above proves
# every name in __all__ resolves and is public; this proves the converse,
# that every public name this module defines at module level is in
# __all__. Neither direction implies the other.
# ---------------------------------------------------------------------------

_NOT_PUBLIC_API = frozenset({"logger"})


def _module_level_public_definitions(module: Any) -> set[str]:
    """Names this module defines at module level (function, class, or a
    module-level assignment target) that are not underscore-prefixed --
    imported names are excluded, since ``__all__`` states what this module
    offers, not what it happened to import."""

    source = inspect.getsource(module)
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                names.add(node.target.id)
    return names


def test_th7_all_names_include_every_public_module_level_definition() -> None:
    """``logger`` is excluded deliberately, not overlooked:
    ``logging.getLogger(__name__)`` is boilerplate every module in this
    tree carries at module level, an implementation detail rather than
    something this module wants a caller to import. Admitting it into
    ``__all__`` would turn ``__all__`` from "this module's public surface"
    into "this module's module-level names", which is not the same
    promise.
    """

    defined = _module_level_public_definitions(staging)
    missing = (defined - _NOT_PUBLIC_API) - set(staging.__all__)
    assert missing == set(), (
        f"public module-level definitions missing from __all__: {missing}"
    )


# ---------------------------------------------------------------------------
# InteractionHandoff has zero direct-construction points in src/ outside
# its own defining module (task_interaction_staging.py). Publicizing the
# class (this module's own rename) makes direct construction syntactically
# possible for the first time -- this guard depends on that rename having
# already landed, which is why it lives here rather than in
# test_task_interaction_anchor.py, whose own dependencies do not carry it.
# ---------------------------------------------------------------------------

_HANDOFF_CLASS_NAME = "InteractionHandoff"
_STAGING_MODULE_STEM = "task_interaction_staging"


def _direct_construction_uses(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == _HANDOFF_CLASS_NAME:
                return True
            if isinstance(func, ast.Attribute) and func.attr == _HANDOFF_CLASS_NAME:
                return True
    return False


def test_ta12_interaction_handoff_has_zero_direct_construction_points() -> None:
    """A direct ``InteractionHandoff(...)`` call anywhere else in ``src/``
    would build an object whose ``stage()`` runs with no containment at
    all: ``interaction_handoff`` is the only place that owns the savepoint
    a constructed instance needs to mean anything.
    """

    offenders: dict[str, bool] = {}
    for path in _scan_root(_STAGING_MODULE_STEM):
        source = path.read_text(encoding="utf-8")
        if _direct_construction_uses(source):
            offenders[str(path)] = True
    assert offenders == {}
    # Positive control: the module's own defining source, scanned directly,
    # does contain exactly this construction (interaction_handoff's own
    # `handoff = InteractionHandoff(...)`) -- proving the stem exclusion is
    # doing real work, not vacuously passing because nothing in the tree
    # constructs this class at all.
    import xagent.web.services.task_interaction_staging as staging_module

    own_source = Path(staging_module.__file__).read_text(encoding="utf-8")
    assert _direct_construction_uses(own_source)

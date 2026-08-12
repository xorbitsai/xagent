"""Contract tests for ``stage_trace_event_row``: the invariants the existing
``_save_trace_event`` tests cannot see because they only observe the shell's
combined, already-assembled behaviour (see the door tests in
``tests/web/test_agent_checkpoint_stream.py`` and
``tests/web/test_trace_message_storage.py``, which these tests leave
untouched). The tests below pin, directly on the primitive: that it never
commits, rolls back, or prunes on its own; the flush/encode/partition shape
of its four paths plus the NULL-anchor guard; and, as a source-level fact,
that it sends no notification.

The shell-side tests are supplements that exercise ``trace_handlers.py``
directly: one guards the single most dangerous line there -- the
``data = staged.stored_data`` rebind after the ``stage_trace_event_row``
call (see that file's comment above the pointer UPDATE) -- and two more are
the forward and reverse controls for the new prune-entry guard.

Each test builds its own file-backed sqlite database under ``tmp_path``
rather than the process-wide ``get_engine()`` singleton, so none of these
tests need ``tests/shared/db_teardown.py::drop_all_tables`` -- pytest tears
down ``tmp_path`` itself. A file-backed engine (not an in-memory
``StaticPool`` one) is used deliberately: the commit- and rollback-isolation
checks need two independent sessions with genuine transaction isolation
between them, which an in-memory database shared over one pooled connection
does not give.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE, CHECKPOINT_TYPE
from xagent.core.agent.trace import TraceEvent as CoreTraceEvent
from xagent.web.api.trace_handlers import DatabaseTraceHandler
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task import TraceEvent as DatabaseTraceEvent
from xagent.web.models.user import User
from xagent.web.services import trace_event_staging
from xagent.web.services.task_lease_service import (
    TASK_RUN_ID_TRACE_FIELD,
    TaskLease,
    bind_task_lease_context,
)
from xagent.web.services.trace_event_staging import (
    checkpoint_run_partition_filter,
    stage_trace_event_row,
)


def _engine(tmp_path: Path):
    """A file-backed sqlite engine, private to one test, with the same FK
    enforcement production connections get (src/xagent/db/sqlite.py's
    connect hook applies only to the process-wide engine, so a private
    test engine has to opt in separately)."""
    db_path = tmp_path / "trace_event_staging.db"
    engine = create_engine(f"sqlite:///{db_path}")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    return engine


def _session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _create_task(session_factory, *, username: str) -> tuple[Session, int]:
    db = session_factory()
    user = User(username=username, password_hash="hashed_password", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    task = Task(
        user_id=int(user.id),
        title="Trace staging task",
        description="Trace staging task",
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return db, int(task.id)


def _checkpoint_data(execution_id: str, label: str) -> dict[str, Any]:
    return {
        "checkpoint_type": CHECKPOINT_TYPE,
        "execution_id": execution_id,
        "snapshot": {"label": label},
    }


def _count_flush_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original_flush = Session.flush

    def counting_flush(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        calls.append(1)
        return original_flush(self, *args, **kwargs)

    monkeypatch.setattr(Session, "flush", counting_flush)
    return calls


def _count_encode_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original_encode = trace_event_staging.encode_checkpoint_data_for_storage

    def counting_encode(db, *, task_id, data, use_v2=None):  # noqa: ANN001
        calls.append(1)
        return original_encode(db, task_id=task_id, data=data, use_v2=use_v2)

    monkeypatch.setattr(
        trace_event_staging, "encode_checkpoint_data_for_storage", counting_encode
    )
    return calls


# --------------------------------------------------------------------------
# Commit, rollback, and prune independence -- verbs the shell-level tests
# cannot pin directly on the primitive
# --------------------------------------------------------------------------


def test_stage_trace_event_row_does_not_commit(tmp_path: Path) -> None:
    """Joins the caller's transaction, does not commit it. Path (d) is
    used because it is the one path that flushes -- the row is genuinely
    written at the database level (not just held in the ORM's identity
    map), so "a second session can't see it" is a real transaction-
    isolation check, not an artifact of ORM-local state."""
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="s1-commit")
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a")

    staged = stage_trace_event_row(
        db,
        task_id=task_id,
        build_id=None,
        event_id="evt-s1",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        step_id=None,
        parent_event_id=None,
        data=_checkpoint_data("exec-s1", "s1"),
        checkpoint_lease=lease,
    )
    assert staged.row_id is not None

    other = session_factory()
    try:
        assert db.query(DatabaseTraceEvent).filter_by(task_id=task_id).count() == 1
        assert other.query(DatabaseTraceEvent).filter_by(task_id=task_id).count() == 0
        assert db.in_transaction()
    finally:
        other.close()
        db.rollback()
        db.close()


def test_stage_trace_event_row_does_not_rollback_after_integrity_error(
    tmp_path: Path,
) -> None:
    """An IntegrityError raised inside the primitive (here, a
    trace_events.task_id FK violation on a task that was never created)
    must propagate without being rolled back. The session is left in the
    "flush failed, roll back before you can use it again" state SQLAlchemy
    itself puts a session into after an uncaught flush error; the caller
    (the shell's own except IntegrityError block) is who rolls it back."""
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db = session_factory()
    missing_task_id = 999999
    lease = TaskLease(task_id=missing_task_id, runner_id="runner-a", run_id="run-a")

    with pytest.raises(IntegrityError):
        stage_trace_event_row(
            db,
            task_id=missing_task_id,
            build_id=None,
            event_id="evt-s2",
            event_type="system_update_general",
            timestamp=datetime.now(timezone.utc),
            step_id=None,
            parent_event_id=None,
            data=_checkpoint_data("exec-s2", "s2"),
            checkpoint_lease=lease,
        )

    assert db.in_transaction()
    # Proof the primitive did not roll back on its own: SQLAlchemy refuses
    # any further statement on a session left in this state until the
    # caller rolls it back explicitly. If the primitive had already rolled
    # back, this query would succeed instead of raising.
    with pytest.raises(Exception, match="rolled back|PendingRollback"):
        db.query(DatabaseTraceEvent).count()

    db.rollback()
    assert db.query(DatabaseTraceEvent).count() == 0
    db.close()


def test_stage_trace_event_row_does_not_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The primitive has no reference to _prune_checkpoint_history at
    all -- it does not import trace_handlers, let alone call the method.
    Patching the method and calling the primitive directly (not through
    DatabaseTraceHandler) proves the call count stays zero even on the one
    path (d) that does the most work."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        DatabaseTraceHandler,
        "_prune_checkpoint_history",
        lambda self, db, data: calls.append(data),
    )

    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="s3-prune")
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a")

    stage_trace_event_row(
        db,
        task_id=task_id,
        build_id=None,
        event_id="evt-s3",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        step_id=None,
        parent_event_id=None,
        data=_checkpoint_data("exec-s3", "s3"),
        checkpoint_lease=lease,
    )

    assert calls == []
    db.rollback()
    db.close()


# --------------------------------------------------------------------------
# The four paths' flush / encode / partition shape
# --------------------------------------------------------------------------


def test_stage_trace_event_row_path_a_non_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-checkpoint event neither encodes nor flushes."""
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="s4-non-checkpoint")

    # Patched only after setup's own commits run, so those aren't counted.
    flush_calls = _count_flush_calls(monkeypatch)
    encode_calls = _count_encode_calls(monkeypatch)

    staged = stage_trace_event_row(
        db,
        task_id=task_id,
        build_id=None,
        event_id="evt-s4",
        event_type="tool_execution_start",
        timestamp=datetime.now(timezone.utc),
        step_id=None,
        parent_event_id=None,
        data={"tool_name": "noop"},
        checkpoint_lease=None,
    )

    assert staged.row_id is None
    assert staged.anchor is None
    assert flush_calls == []
    assert encode_calls == []
    assert TASK_RUN_ID_TRACE_FIELD not in staged.stored_data
    db.rollback()
    db.close()


def test_stage_trace_event_row_path_b_sub_agent_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sub-agent checkpoint (build_id set) encodes but never
    flushes and never gets a run partition stamp. The realistic call
    mirrors the shell's own calling convention (checkpoint_lease=None,
    since the shell never passes a live lease when build_id is set); the
    second call additionally proves the primitive's own build_id is None
    guard holds even if a caller passed a live lease anyway -- the
    original function had this same redundant check (trace_handlers.py
    checked self.build_id is None a second time even though the lease
    variable it was paired with could only be non-None when build_id was
    already None), and this defends the same thing here."""
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="s5-sub-agent")
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a-s5")

    flush_calls = _count_flush_calls(monkeypatch)
    encode_calls = _count_encode_calls(monkeypatch)

    staged = stage_trace_event_row(
        db,
        task_id=task_id,
        build_id="agent_123_abcd1234",
        event_id="evt-s5",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        step_id=None,
        parent_event_id=None,
        data=_checkpoint_data("exec-s5", "s5"),
        checkpoint_lease=None,
    )

    assert staged.row_id is None
    assert staged.anchor is None
    assert flush_calls == []
    assert encode_calls == [1]
    assert TASK_RUN_ID_TRACE_FIELD not in staged.stored_data

    # Defense in depth: even a caller that (incorrectly) passes a live
    # lease alongside a non-None build_id must not get a flush.
    staged_off_label = stage_trace_event_row(
        db,
        task_id=task_id,
        build_id="agent_123_abcd1234",
        event_id="evt-s5-off-label",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        step_id=None,
        parent_event_id=None,
        data=_checkpoint_data("exec-s5-off-label", "s5-off-label"),
        checkpoint_lease=lease,
    )
    assert staged_off_label.row_id is None
    assert staged_off_label.anchor is None
    assert flush_calls == []

    db.rollback()
    db.close()


def test_stage_trace_event_row_path_c_root_checkpoint_without_a_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root checkpoint with no live lease (current_task_lease()
    returned None) encodes but never flushes and gets no run partition."""
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="s6-no-lease")

    flush_calls = _count_flush_calls(monkeypatch)
    encode_calls = _count_encode_calls(monkeypatch)

    staged = stage_trace_event_row(
        db,
        task_id=task_id,
        build_id=None,
        event_id="evt-s6",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        step_id=None,
        parent_event_id=None,
        data=_checkpoint_data("exec-s6", "s6"),
        checkpoint_lease=None,
    )

    assert staged.row_id is None
    assert staged.anchor is None
    assert flush_calls == []
    assert encode_calls == [1]
    assert TASK_RUN_ID_TRACE_FIELD not in staged.stored_data
    db.rollback()
    db.close()


def test_stage_trace_event_row_path_d_root_checkpoint_under_a_live_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root checkpoint under a live lease flushes exactly once,
    stamps the run partition, and returns an anchor naming the flushed
    row's own primary key."""
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="s7-live-lease")
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a-s7")

    flush_calls = _count_flush_calls(monkeypatch)
    encode_calls = _count_encode_calls(monkeypatch)

    staged = stage_trace_event_row(
        db,
        task_id=task_id,
        build_id=None,
        event_id="evt-s7",
        event_type="system_update_general",
        timestamp=datetime.now(timezone.utc),
        step_id=None,
        parent_event_id=None,
        data=_checkpoint_data("exec-s7", "s7"),
        checkpoint_lease=lease,
    )

    assert flush_calls == [1]
    assert encode_calls == [1]
    assert staged.row_id is not None
    assert staged.anchor is not None
    assert staged.anchor.trace_event_id == staged.row_id
    assert staged.anchor.checkpoint_event_id == "evt-s7"
    assert staged.stored_data[TASK_RUN_ID_TRACE_FIELD] == lease.run_id
    db.rollback()
    db.close()


# --------------------------------------------------------------------------
# The NULL-anchor guard, pinned directly on the primitive
# --------------------------------------------------------------------------


def test_stage_trace_event_row_refuses_a_null_anchor_after_a_no_op_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors the shell-level guard test
    ``test_database_trace_handler_flush_without_primary_key_refuses_to_write_the_anchor``
    in test_agent_checkpoint_stream.py, but exercised directly against the
    primitive, and additionally checking that the primitive itself does not
    roll back on this path -- the shell-level test's NULL/zero-rows
    assertions depend on the shell's own bare except doing that rollback,
    not on this function doing it."""
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="s8-null-anchor")
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a-s8")

    monkeypatch.setattr(Session, "flush", lambda self, *a, **k: None)  # noqa: ARG005

    with pytest.raises(
        RuntimeError, match="refusing to write a NULL checkpoint anchor"
    ):
        stage_trace_event_row(
            db,
            task_id=task_id,
            build_id=None,
            event_id="evt-s8",
            event_type="system_update_general",
            timestamp=datetime.now(timezone.utc),
            step_id=None,
            parent_event_id=None,
            data=_checkpoint_data("exec-s8", "s8"),
            checkpoint_lease=lease,
        )

    assert db.in_transaction()
    db.rollback()
    db.close()


# --------------------------------------------------------------------------
# No notification, pinned as a source-level fact
# --------------------------------------------------------------------------


def test_trace_event_staging_module_sends_no_notifications() -> None:
    """No notification is delivered, asserted as a zero-site fact about this
    module's imports and call targets rather than its prose: the substring
    form also matched comments, which constrained how this module could be
    documented."""
    tree = ast.parse(Path(trace_event_staging.__file__).read_text())
    roots = ("notify", "notification", "dispatch", "publish", "broadcast")
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.name.split(".")[-1] for a in node.names)
            names.update(a.asname for a in node.names if a.asname)
    offenders = sorted(n for n in names if any(r in n.lower() for r in roots))
    assert offenders == [], offenders


# --------------------------------------------------------------------------
# Shell-side supplements (trace_handlers.py, not the primitive)
# --------------------------------------------------------------------------


def test_shell_passes_the_partition_stamped_data_into_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the single most dangerous line in trace_handlers.py --
    the ``data = staged.stored_data`` rebind after the
    stage_trace_event_row call. Without that rebind, the shell's local
    ``data`` stays the pre-partition-stamp object and
    _prune_checkpoint_history silently loses its run-partition filter,
    which would not necessarily turn any existing assertion red."""
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="t6-prune-data")
    task = db.get(Task, task_id)
    task.status = TaskStatus.RUNNING
    task.runner_id = "runner-a"
    task.run_id = "run-a-t6"
    db.commit()
    lease = TaskLease(task_id=task_id, runner_id="runner-a", run_id="run-a-t6")

    captured: list[dict[str, Any]] = []
    original_prune = DatabaseTraceHandler._prune_checkpoint_history

    def capturing_prune(self, db, data):  # noqa: ANN001
        captured.append(data)
        return original_prune(self, db, data)

    monkeypatch.setattr(
        DatabaseTraceHandler, "_prune_checkpoint_history", capturing_prune
    )

    event = CoreTraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id=str(task_id),
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": str(task_id),
            "snapshot": {"label": "t6"},
        },
    )

    try:
        with bind_task_lease_context(lease):
            DatabaseTraceHandler(task_id)._save_trace_event(db, event)

        assert len(captured) == 1
        assert captured[0].get(TASK_RUN_ID_TRACE_FIELD) == lease.run_id
    finally:
        db.close()


def test_prune_checkpoint_history_guard_rejects_pending_writes(
    tmp_path: Path,
) -> None:
    """Reverse control: a session carrying an uncommitted write must
    not be handed to the prune path -- without the guard, prune's own
    db.commit() would silently commit that pending write as a side effect
    of retention cleanup."""
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="t7-reverse")

    db.add(User(username="t7-dirty", password_hash="x", is_admin=False))

    with pytest.raises(RuntimeError, match="pending writes"):
        DatabaseTraceHandler(task_id)._prune_checkpoint_history(
            db, {"checkpoint_type": CHECKPOINT_TYPE, "execution_id": "exec-t7"}
        )

    db.rollback()
    db.close()


def test_prune_checkpoint_history_guard_allows_a_clean_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forward control: the guard must not reject the one state the
    real call site ever hands it -- a clean session, since prune always
    runs immediately after the checkpoint commit."""
    monkeypatch.setattr(
        "xagent.web.api.trace_handlers.get_checkpoint_history_limit", lambda: 5
    )
    engine = _engine(tmp_path)
    session_factory = _session_factory(engine)
    db, task_id = _create_task(session_factory, username="t7-forward")

    DatabaseTraceHandler(task_id)._prune_checkpoint_history(
        db, {"checkpoint_type": CHECKPOINT_TYPE, "execution_id": "exec-t7-forward"}
    )

    db.close()


# ---------------------------------------------------------------------------
# checkpoint_run_partition_filter's move from a trace_handlers.py
# staticmethod into this module must leave the old staticmethod delegating,
# not reimplementing. An earlier draft of the cell below compiled both
# callables' SQL to a string and compared -- that stays green even if the
# old path grew its own copy of the predicate body instead of calling this
# module's function (a real mutation, tried directly), so the cell pins the
# structural fact instead: the staticmethod's body is exactly one `return`
# statement calling checkpoint_run_partition_filter.
# ---------------------------------------------------------------------------


def test_legacy_staticmethod_delegates_to_the_shared_predicate() -> None:
    source = textwrap.dedent(
        inspect.getsource(DatabaseTraceHandler._checkpoint_run_partition_filter)
    )
    func_def = ast.parse(source).body[0]
    assert isinstance(func_def, ast.FunctionDef)
    assert len(func_def.body) == 1, (
        "_checkpoint_run_partition_filter must be exactly one statement "
        "(a delegating return), not a reimplementation"
    )
    stmt = func_def.body[0]
    assert isinstance(stmt, ast.Return)
    call = stmt.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "checkpoint_run_partition_filter"


def test_positive_control_both_run_partition_predicates_compile_identical_sql() -> None:
    """Not the cell above's own guard (see its docstring for why), but a
    positive control that the two still agree on output today, for both a
    real run_id and None."""

    for run_id in ("run-a", None):
        old = str(
            DatabaseTraceHandler._checkpoint_run_partition_filter(run_id).compile(
                compile_kwargs={"literal_binds": True}
            )
        )
        new = str(
            checkpoint_run_partition_filter(run_id).compile(
                compile_kwargs={"literal_binds": True}
            )
        )
        assert old == new

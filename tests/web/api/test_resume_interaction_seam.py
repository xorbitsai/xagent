"""The compatibility seam routing ``_handle_resume_task_unserialized``
through the interaction lifecycle service: a task that still has an active
native interaction row must refuse a resume that does not carry the
receipt ``respond()`` stages, on both of this handler's two resume paths
(the ``supports_live_control`` branch and the bare ``resume_execution``
fallback) -- see ``active_interaction_id_sync``'s own docstring in
``task_interaction_close.py`` for the shared predicate this seam keys off
(including the ``tasks.interaction_protocol_version`` marker gate ahead of
it), and that function's call site in ``websocket.py`` for why the check
runs before ``agent_service`` is built.

Mocking follows ``test_pause_resume_offturn_build_workspace.py``'s
established shape: a real ``AgentServiceManager``/``TaskSetupSnapshot``
double for everything upstream of the seam, and only the genuinely
external boundaries patched. Unlike that file, these tests seed a real,
temporary SQLite database so the seam's own DB query -- which that file's
tests never reach, because none of their tasks have an interaction row --
has real data to answer against.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_interaction_schema_shared import (
    make_row,
    make_task,
    make_user,
)
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.api import chat as chat_api
from xagent.web.api import websocket as websocket_api
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import ops_signals
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)

OWNER_ID = 7
RUN_ID = "run-a"


@pytest.fixture(autouse=True)
def _reset_ops_signals():
    """Keep this file's degradation signals out of every other test in the
    same process.

    The only signal these tests themselves register is
    ``INTERACTION_LEGACY_RESUME_SHIM``, on each refusal. The registry is
    process-global with no clear site for that name, and
    ``tests/web/test_health_degradations.py`` asserts ``/health``'s payload
    by exact equality, so a leaked signal fails a test in a different file.
    Clearing every name rather than that one also makes this file's own
    "the signal was NOT registered" assertion sound: it cannot pass or fail
    on what some earlier module left behind. Same shape as
    ``tests/web/services/test_interaction_handoff_surface.py``'s fixture.
    """

    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)
    yield
    for name in list(ops_signals.active_degradations()):
        ops_signals.clear_degradation(name)


@pytest.fixture()
def _engine(tmp_path):
    db_path = tmp_path / "resume_seam.db"
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture()
def _session_factory(_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture()
def _seeded_task(_session_factory) -> int:
    db = _session_factory()
    try:
        user_id = make_user(db)
        task_id = make_task(db, user_id=user_id)
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.WAITING_FOR_USER
        task.run_id = RUN_ID
        task.interaction_protocol_version = 1
        db.commit()
        return task_id
    finally:
        db.close()


def _make_active_row(session_factory, *, task_id: int, run_id: str = RUN_ID) -> int:
    db = session_factory()
    try:
        now = datetime.now(timezone.utc)
        event = TraceEvent(
            task_id=task_id,
            event_id="resume-seam-trace-event",
            event_type="agent_execution_checkpoint",
            timestamp=now,
            data={
                TASK_RUN_ID_TRACE_FIELD: run_id,
                "checkpoint_type": "agent_execution_checkpoint",
                "execution_id": "exec-1",
            },
        )
        db.add(event)
        db.commit()
        db.refresh(event)

        row = TaskInteractionRequest(
            **make_row(
                task_id=task_id,
                resume_trace_event_id=event.id,
                db=db,
                run_id=run_id,
                request_payload={"message": "Which environment?", "interactions": []},
                resume_run_partition=run_id,
                created_at=now,
                expires_at=now + timedelta(minutes=15),
            )
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)
    finally:
        db.close()


def _snapshot(*, task_id: int, run_id: str = RUN_ID) -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=task_id,
            user_id=OWNER_ID,
            status=TaskStatus.WAITING_FOR_USER,
            agent_id=None,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
            run_id=run_id,
            state_version=1,
            control_state="waiting_for_user",
        ),
        runtime_user=RuntimeUserFields(id=OWNER_ID, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=None,
        agent_config=None,
        excluded_agent_id=None,
    )


def _connection_manager() -> MagicMock:
    manager = MagicMock()
    manager.send_personal_message = AsyncMock()
    manager.broadcast_to_task = AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_legacy_resume_without_a_receipt_is_refused_with_an_active_row(
    monkeypatch: pytest.MonkeyPatch, _session_factory, _seeded_task: int
) -> None:
    """Neither resume path may run when an active native interaction row
    exists and the incoming command carries no receipt: this asserts zero
    commands issued and zero messages beyond the refusal, on the
    ``supports_live_control`` path (agent_service reports
    ``supports_live_control() -> True``)."""

    import xagent.web.models.database as database_module

    monkeypatch.setattr(
        database_module, "get_optional_session_local", lambda: _session_factory
    )
    _make_active_row(_session_factory, task_id=_seeded_task)

    snapshot = _snapshot(task_id=_seeded_task)
    connection_manager = _connection_manager()
    background_manager = MagicMock()
    background_manager.running_tasks = {}
    transition = AsyncMock()
    agent_service = MagicMock()
    agent_service.supports_live_control = MagicMock(return_value=True)

    from xagent.web.services import task_setup_snapshot as snapshot_module

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch.object(
                websocket_api.task_execution_controller, "transition", new=transition
            )
        )
        stack.enter_context(
            patch.object(websocket_api, "background_task_manager", background_manager)
        )
        # The handler asks the DB whether another process still holds a live
        # lease before it schedules; these suites drive the handler without a
        # task row, so answer "no foreign owner" explicitly.
        stack.enter_context(
            patch.object(
                websocket_api, "task_has_live_foreign_runner", return_value=False
            )
        )
        agent_manager = MagicMock()
        agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: agent_manager)
        )

        await websocket_api._handle_resume_task_unserialized(
            MagicMock(),
            _seeded_task,
            {"user": SimpleNamespace(id=OWNER_ID, is_admin=False)},
        )

        agent_manager.get_agent_for_task.assert_not_called()

    transition.assert_not_called()
    background_manager.try_reserve_resume.assert_not_called()
    assert connection_manager.send_personal_message.await_count == 1
    sent = connection_manager.send_personal_message.await_args.args[0]
    assert sent["type"] == "error"
    assert "unanswered question" in sent["message"]
    assert (
        ops_signals.INTERACTION_LEGACY_RESUME_SHIM in ops_signals.active_degradations()
    )


@pytest.mark.asyncio
async def test_legacy_resume_without_a_receipt_is_refused_on_the_fallback_path(
    monkeypatch: pytest.MonkeyPatch, _session_factory, _seeded_task: int
) -> None:
    """The second resume path (``hasattr(agent_service, "resume_execution")``,
    no ``task_status`` gate of its own in the pre-existing code) must be
    refused identically -- a seam placed only ahead of the other branch
    would leave this one able to append to or replan around an unanswered
    question."""

    import xagent.web.models.database as database_module

    monkeypatch.setattr(
        database_module, "get_optional_session_local", lambda: _session_factory
    )
    _make_active_row(_session_factory, task_id=_seeded_task)

    snapshot = _snapshot(task_id=_seeded_task)
    connection_manager = _connection_manager()
    background_manager = MagicMock()
    background_manager.running_tasks = {}
    transition = AsyncMock()

    # supports_live_control absent entirely (not just False) -- exercises
    # the getattr(..., lambda: False)() default, landing this call on the
    # second path if the seam did not refuse first.
    agent_service = SimpleNamespace(resume_execution=AsyncMock())

    from xagent.web.services import task_setup_snapshot as snapshot_module

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch.object(
                websocket_api.task_execution_controller, "transition", new=transition
            )
        )
        stack.enter_context(
            patch.object(websocket_api, "background_task_manager", background_manager)
        )
        # The handler asks the DB whether another process still holds a live
        # lease before it schedules; these suites drive the handler without a
        # task row, so answer "no foreign owner" explicitly.
        stack.enter_context(
            patch.object(
                websocket_api, "task_has_live_foreign_runner", return_value=False
            )
        )
        agent_manager = MagicMock()
        agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: agent_manager)
        )

        await websocket_api._handle_resume_task_unserialized(
            MagicMock(),
            _seeded_task,
            {"user": SimpleNamespace(id=OWNER_ID, is_admin=False)},
        )

        agent_manager.get_agent_for_task.assert_not_called()

    transition.assert_not_called()
    agent_service.resume_execution.assert_not_called()
    assert connection_manager.send_personal_message.await_count == 1
    sent = connection_manager.send_personal_message.await_args.args[0]
    assert sent["type"] == "error"
    assert "unanswered question" in sent["message"]


@pytest.mark.asyncio
async def test_legacy_resume_is_not_refused_when_the_task_marker_is_null(
    monkeypatch: pytest.MonkeyPatch, _session_factory, _seeded_task: int
) -> None:
    """The refusal gate reads through active_interaction_id_sync, which
    stops at a NULL tasks.interaction_protocol_version before it queries
    the interaction table at all. A task carrying an active row under a
    NULL marker is therefore not refused -- deliberately, and matching what
    the read surface does with the same state: get_pending_interaction_
    question (task_interaction_read.py) also stops at the NULL marker and
    shows the legacy transcript question, so refusing a resume here would
    be refusing it on account of a row no reader will ever show. The state
    is unreachable today (nothing in src/ writes the marker to 1), and the
    change that wires the first writer has to write it -- see the marker's
    own ownership comment on Task.interaction_protocol_version."""

    import xagent.web.models.database as database_module

    monkeypatch.setattr(
        database_module, "get_optional_session_local", lambda: _session_factory
    )
    _make_active_row(_session_factory, task_id=_seeded_task)

    db = _session_factory()
    try:
        db.query(Task).filter(Task.id == _seeded_task).update(
            {Task.interaction_protocol_version: None}
        )
        db.commit()
    finally:
        db.close()

    snapshot = _snapshot(task_id=_seeded_task)
    connection_manager = _connection_manager()
    background_manager = MagicMock()
    background_manager.running_tasks = {}
    background_manager.resume_admission_state.return_value = None
    background_manager.try_reserve_resume.return_value = (
        websocket_api.ResumeReservationOutcome.RESERVED
    )
    transition = AsyncMock(
        return_value=SimpleNamespace(run_id=RUN_ID, status=TaskStatus.WAITING_FOR_USER)
    )
    agent_service = MagicMock()
    agent_service.supports_live_control = MagicMock(return_value=True)

    resume_scheduled = asyncio.Event()

    async def _stub_execute_resume_background(**kwargs: object) -> None:
        resume_scheduled.set()

    from xagent.web.services import task_setup_snapshot as snapshot_module

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch.object(
                websocket_api.task_execution_controller, "transition", new=transition
            )
        )
        stack.enter_context(
            patch.object(websocket_api, "background_task_manager", background_manager)
        )
        # The handler asks the DB whether another process still holds a live
        # lease before it schedules; these suites drive the handler without a
        # task row, so answer "no foreign owner" explicitly.
        stack.enter_context(
            patch.object(
                websocket_api, "task_has_live_foreign_runner", return_value=False
            )
        )
        stack.enter_context(
            patch.object(
                websocket_api,
                "execute_resume_background",
                side_effect=_stub_execute_resume_background,
            )
        )
        agent_manager = MagicMock()
        agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: agent_manager)
        )

        await websocket_api._handle_resume_task_unserialized(
            MagicMock(),
            _seeded_task,
            {"user": SimpleNamespace(id=OWNER_ID, is_admin=False)},
        )

        agent_manager.get_agent_for_task.assert_awaited()
        await asyncio.wait_for(resume_scheduled.wait(), timeout=1)

    assert resume_scheduled.is_set()
    refusals = [
        call.args[0]
        for call in connection_manager.send_personal_message.await_args_list
        if isinstance(call.args[0], dict)
        and call.args[0].get("type") == "error"
        and "unanswered question" in str(call.args[0].get("message", ""))
    ]
    assert refusals == []
    assert (
        ops_signals.INTERACTION_LEGACY_RESUME_SHIM
        not in ops_signals.active_degradations()
    )


# ---------------------------------------------------------------------------
# Receipt shapes the seam must refuse.
#
# The comparison at the seam is deliberately type-strict: a receipt it
# cannot recognize as the exact int respond() stages is no receipt at all.
# Each factory below returns an ``interaction_id`` a coercion-first rewrite
# would wrongly accept, so this table is the pin against that rewrite.
# ---------------------------------------------------------------------------


def _string_form_id(interaction_id: int) -> object:
    """The active row's own id, as a string.

    Refused by the plain ``!=`` on its own (``"1" != 1``). Pinned so a
    future rewrite cannot start coercing the payload before comparing.
    """

    return str(interaction_id)


def _float_form_id(interaction_id: int) -> object:
    """The active row's own id, as a JSON number with a fraction part.

    ``1.0 == 1`` in Python, so the plain ``!=`` accepts this for any row.
    Only the ``isinstance(..., int)`` term refuses it.
    """

    return float(interaction_id)


def _json_true_id(interaction_id: int) -> object:
    """JSON ``true``.

    ``True == 1`` in Python, so the plain ``!=`` accepts JSON ``true`` as
    the receipt for row 1 specifically. Only the ``isinstance(..., bool)``
    term refuses it. Every test in this file seeds exactly one row into its
    own fresh temporary database, so the active row's id is 1 here; that is
    asserted rather than assumed, because the case is meaningless against
    any other id.
    """

    assert interaction_id == 1
    return True


def _exact_id(interaction_id: int) -> object:
    """The active row's own id, as the int respond() stages.

    Paired with a payload that omits ``responder_identity`` so the
    ``or not receipt_responder_identity`` disjunct is what refuses this
    case -- every other case short-circuits on the id comparison, leaving
    that disjunct never exercised in the True direction.
    """

    return interaction_id


def _wrong_int_id(interaction_id: int) -> object:
    """A well-typed int naming a different row.

    The most realistic bad receipt: a client resuming with the receipt of
    an earlier, already-retired question. Both ``isinstance`` terms pass,
    so only the ``!=`` comparison itself refuses this case -- the
    wrong-typed cases above all short-circuit on the type checks and never
    reach it, which would otherwise leave the id equality with no coverage
    in the True direction.
    """

    return interaction_id + 1


_UNVERIFIABLE_RECEIPTS = [
    pytest.param(_string_form_id, True, id="string-form-id"),
    pytest.param(_float_form_id, True, id="float-form-id"),
    pytest.param(_json_true_id, True, id="json-true-id"),
    pytest.param(_wrong_int_id, True, id="wrong-int-id"),
    pytest.param(_exact_id, False, id="exact-id-without-responder-identity"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("receipt_id_factory", "include_responder_identity"), _UNVERIFIABLE_RECEIPTS
)
async def test_receipts_the_seam_cannot_verify_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    _session_factory,
    _seeded_task: int,
    receipt_id_factory,
    include_responder_identity: bool,
) -> None:
    """Every receipt shape the seam cannot verify as the exact continuation
    ``respond()`` staged must land on the refusal path, identically to a
    command that carries no receipt at all."""

    import xagent.web.models.database as database_module

    monkeypatch.setattr(
        database_module, "get_optional_session_local", lambda: _session_factory
    )
    interaction_id = _make_active_row(_session_factory, task_id=_seeded_task)

    snapshot = _snapshot(task_id=_seeded_task)
    connection_manager = _connection_manager()
    background_manager = MagicMock()
    background_manager.running_tasks = {}
    transition = AsyncMock()
    agent_service = MagicMock()
    agent_service.supports_live_control = MagicMock(return_value=True)

    payload: dict[str, object] = {
        "user": SimpleNamespace(id=OWNER_ID, is_admin=False),
        "interaction_id": receipt_id_factory(interaction_id),
    }
    if include_responder_identity:
        payload["responder_identity"] = f"user:{OWNER_ID}"

    from xagent.web.services import task_setup_snapshot as snapshot_module

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch.object(
                websocket_api.task_execution_controller, "transition", new=transition
            )
        )
        stack.enter_context(
            patch.object(websocket_api, "background_task_manager", background_manager)
        )
        # The handler asks the DB whether another process still holds a live
        # lease before it schedules; these suites drive the handler without a
        # task row, so answer "no foreign owner" explicitly.
        stack.enter_context(
            patch.object(
                websocket_api, "task_has_live_foreign_runner", return_value=False
            )
        )
        agent_manager = MagicMock()
        agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: agent_manager)
        )

        await websocket_api._handle_resume_task_unserialized(
            MagicMock(),
            _seeded_task,
            payload,
        )

        agent_manager.get_agent_for_task.assert_not_called()

    transition.assert_not_called()
    background_manager.try_reserve_resume.assert_not_called()
    assert connection_manager.send_personal_message.await_count == 1
    sent = connection_manager.send_personal_message.await_args.args[0]
    assert sent["type"] == "error"
    assert "unanswered question" in sent["message"]
    assert (
        ops_signals.INTERACTION_LEGACY_RESUME_SHIM in ops_signals.active_degradations()
    )


@pytest.mark.asyncio
async def test_resume_with_a_matching_receipt_is_not_refused(
    monkeypatch: pytest.MonkeyPatch, _session_factory, _seeded_task: int
) -> None:
    """A command payload naming the active row's own id and a
    ``responder_identity`` -- exactly the shape ``respond()`` stages -- must
    pass the seam and reach the pre-existing resume logic."""

    import xagent.web.models.database as database_module

    monkeypatch.setattr(
        database_module, "get_optional_session_local", lambda: _session_factory
    )
    interaction_id = _make_active_row(_session_factory, task_id=_seeded_task)

    snapshot = _snapshot(task_id=_seeded_task)
    connection_manager = _connection_manager()
    background_manager = MagicMock()
    background_manager.running_tasks = {}
    background_manager.resume_admission_state.return_value = None
    background_manager.try_reserve_resume.return_value = (
        websocket_api.ResumeReservationOutcome.RESERVED
    )
    transition = AsyncMock(
        return_value=SimpleNamespace(run_id=RUN_ID, status=TaskStatus.WAITING_FOR_USER)
    )
    agent_service = MagicMock()
    agent_service.supports_live_control = MagicMock(return_value=True)

    resume_scheduled = asyncio.Event()

    async def _stub_execute_resume_background(**kwargs: object) -> None:
        resume_scheduled.set()

    from xagent.web.services import task_setup_snapshot as snapshot_module

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch.object(
                websocket_api.task_execution_controller, "transition", new=transition
            )
        )
        stack.enter_context(
            patch.object(websocket_api, "background_task_manager", background_manager)
        )
        # The handler asks the DB whether another process still holds a live
        # lease before it schedules; these suites drive the handler without a
        # task row, so answer "no foreign owner" explicitly.
        stack.enter_context(
            patch.object(
                websocket_api, "task_has_live_foreign_runner", return_value=False
            )
        )
        stack.enter_context(
            patch.object(
                websocket_api,
                "execute_resume_background",
                side_effect=_stub_execute_resume_background,
            )
        )
        agent_manager = MagicMock()
        agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: agent_manager)
        )

        await websocket_api._handle_resume_task_unserialized(
            MagicMock(),
            _seeded_task,
            {
                "user": SimpleNamespace(id=OWNER_ID, is_admin=False),
                "interaction_id": interaction_id,
                "responder_identity": f"user:{OWNER_ID}",
            },
        )
        await asyncio.wait_for(resume_scheduled.wait(), timeout=1)

    transition.assert_awaited()
    assert resume_scheduled.is_set()
    # The signal marks a refusal, not "the seam ran". Without this, an
    # unconditional register_degradation would pass the whole suite.
    assert (
        ops_signals.INTERACTION_LEGACY_RESUME_SHIM
        not in ops_signals.active_degradations()
    )


@pytest.mark.asyncio
async def test_stale_run_active_row_does_not_trip_the_seam(
    monkeypatch: pytest.MonkeyPatch, _session_factory, _seeded_task: int
) -> None:
    """An active row anchored to a run the task has since moved past is
    invisible to the seam, identically to the fence and the read surface
    (the same four-field predicate, the same run-scoping join)."""

    import xagent.web.models.database as database_module

    monkeypatch.setattr(
        database_module, "get_optional_session_local", lambda: _session_factory
    )
    _make_active_row(_session_factory, task_id=_seeded_task, run_id="run-old")

    snapshot = _snapshot(task_id=_seeded_task, run_id=RUN_ID)
    connection_manager = _connection_manager()
    background_manager = MagicMock()
    background_manager.running_tasks = {}
    background_manager.resume_admission_state.return_value = None
    background_manager.try_reserve_resume.return_value = (
        websocket_api.ResumeReservationOutcome.RESERVED
    )
    transition = AsyncMock(
        return_value=SimpleNamespace(run_id=RUN_ID, status=TaskStatus.WAITING_FOR_USER)
    )
    agent_service = MagicMock()
    agent_service.supports_live_control = MagicMock(return_value=True)
    resume_scheduled = asyncio.Event()

    async def _stub_execute_resume_background(**kwargs: object) -> None:
        resume_scheduled.set()

    from xagent.web.services import task_setup_snapshot as snapshot_module

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                snapshot_module, "load_task_setup_snapshot_sync", return_value=snapshot
            )
        )
        stack.enter_context(patch.object(websocket_api, "manager", connection_manager))
        stack.enter_context(
            patch.object(
                websocket_api.task_execution_controller, "transition", new=transition
            )
        )
        stack.enter_context(
            patch.object(websocket_api, "background_task_manager", background_manager)
        )
        # The handler asks the DB whether another process still holds a live
        # lease before it schedules; these suites drive the handler without a
        # task row, so answer "no foreign owner" explicitly.
        stack.enter_context(
            patch.object(
                websocket_api, "task_has_live_foreign_runner", return_value=False
            )
        )
        stack.enter_context(
            patch.object(
                websocket_api,
                "execute_resume_background",
                side_effect=_stub_execute_resume_background,
            )
        )
        agent_manager = MagicMock()
        agent_manager.get_agent_for_task = AsyncMock(return_value=agent_service)
        stack.enter_context(
            patch.object(chat_api, "get_agent_manager", lambda: agent_manager)
        )

        await websocket_api._handle_resume_task_unserialized(
            MagicMock(),
            _seeded_task,
            {"user": SimpleNamespace(id=OWNER_ID, is_admin=False)},
        )
        await asyncio.wait_for(resume_scheduled.wait(), timeout=1)

    assert resume_scheduled.is_set()


# active_interaction_id_sync's own four fail-open branches (no database
# configured yet, a session that fails to open, the interaction table not
# existing yet, the row lookup itself raising) used to be pinned here,
# against websocket.py's own _active_native_interaction_id_sync. That
# function is gone -- the seam now reads through the same
# task_interaction_close.active_interaction_id_sync the four legacy-resume
# injection sites use -- and its four fail-open branches are pinned in
# tests/web/services/test_task_interaction_close.py instead, alongside the
# marker gate that reader adds ahead of them.

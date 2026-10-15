"""The marker and active-row pairing, checked on PostgreSQL after the commits
of the legacy resume close paths and the resume-abandonment compensation
paths.

When each test samples:

- The online chat close is sampled after ``handle_task_message`` returns.
  The background resume it hands off to is stubbed, so nothing writes to
  the task or the interaction row after the close commits.
- The deferred close is sampled when the agent's resume call is reached,
  which is after the close committed; the task is RUNNING at that moment.
- The A2A and v1 reply resume-input closes are sampled after
  ``resume_a2a_task`` / ``resume_task_reply`` return. The background
  resume they hand off to is stubbed, as for the online chat close.

Every test writes the marker and the interaction row itself and relies on
no production code to set the marker.

The compensation path, ``clear_interaction_marker_if_unpaired``, has three
callers: the deferred resume's restore to the prior status, the A2A
prelease restore and the v1 reply prelease restore. The online chat path
has no compensation path, only the close statement.

None of the three callers takes a lock read first: the first statement
their transaction directs at ``tasks`` is the UPDATE issued by
``release_task_lease_no_commit``, which writes only non-key columns.

Each parameter id of the compensation test is one route by which an
abandoned resume reaches a compensation caller, written out route by route
rather than merged by group. Each route also lists the injection
dispositions its caller classified, so the group its id names (not
injected, acquisition cancelled, not written, failed before write,
accepted) is checked rather than only named.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_session_factory
from tests.web.services.interaction_marker_shared import (
    assert_interaction_marker_invariant,
    assert_marker_residue,
)
from tests.web.services.task_interaction_schema_shared import (
    make_row,
    make_trace_event,
    row_state,
    seed_active_row,
    seed_task_with_run,
)
from tests.web.services.task_lease_shared import (
    live_task_lease as live_task_lease_fixture,
)
from xagent.core.agent.checkpoint import CheckpointUnavailableError
from xagent.core.agent.runner import (
    InjectionDisposition,
    UserMessageInjectionOutcome,
    UserMessageInjectionRejectedError,
    _record_injection_outcome,
    classify_injection,
)
from xagent.web.api.websocket import _make_command_reply
from xagent.web.models.agent import Agent
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.user import User
from xagent.web.services import task_interaction_close, task_resume
from xagent.web.services.task_command_execution import handle_task_message
from xagent.web.services.task_execution import (
    ResumeReservationOutcome,
    execute_resume_background,
)
from xagent.web.services.task_interaction_close import (
    clear_interaction_marker_if_unpaired,
    close_legacy_resume_interaction_sync,
)
from xagent.web.services.task_resume import (
    TaskReplyInput,
    TaskResumeBusyError,
    TaskResumeNotAcceptedError,
    TaskResumeNotResumableError,
    TaskResumeRetryableError,
    resume_a2a_task,
    resume_task_reply,
)

pytestmark = pytest.mark.postgresql

live_task_lease = live_task_lease_fixture

_RUN = "run-1"
_GET_AGENT_MANAGER = "xagent.web.services.agent_service_manager.get_agent_manager"


@pytest.fixture()
def pg_sessions(monkeypatch) -> Iterator[sessionmaker[Session]]:
    with disposable_session_factory("marker_residue", monkeypatch) as sessions:
        yield sessions


class _Seeded(NamedTuple):
    task_id: int
    owner_id: int
    agent_id: int | None
    row_id: int


def _seed(
    pg_sessions: sessionmaker[Session],
    *,
    status: TaskStatus,
    source: str,
    with_agent: bool = False,
    row_status: str = "active",
) -> _Seeded:
    """Seed a task on ``_RUN`` with marker 1 and one interaction row on that
    run, in ``row_status``."""
    with pg_sessions() as db:
        task_id = seed_task_with_run(db, run_id=_RUN, marker=1)
        owner_id = int(db.get(Task, task_id).user_id)
        agent_id: int | None = None
        if with_agent:
            agent = Agent(user_id=owner_id, name="marker residue agent")
            db.add(agent)
            db.flush()
            agent_id = int(agent.id)
        db.query(Task).filter(Task.id == task_id).update(
            {
                Task.status: status,
                Task.source: source,
                Task.description: "d",
                Task.execution_mode: "balanced",
                Task.agent_id: agent_id,
            }
        )
        db.commit()
        if row_status == "active":
            row_id = seed_active_row(db, task_id=task_id, run_id=_RUN)
        else:
            row = TaskInteractionRequest(
                **make_row(
                    task_id=task_id,
                    resume_trace_event_id=make_trace_event(db, task_id=task_id),
                    status=row_status,
                    run_id=_RUN,
                )
            )
            db.add(row)
            db.commit()
            row_id = int(row.id)
    return _Seeded(task_id, owner_id, agent_id, row_id)


def _owner(pg_sessions: sessionmaker[Session], owner_id: int) -> User:
    with pg_sessions() as db:
        owner = db.get(User, owner_id)
        assert owner is not None
        db.expunge(owner)
        return owner


def _task_state(
    observer: Session, task_id: int
) -> tuple[TaskStatus, int | None, str | None]:
    """``(status, interaction_protocol_version, input)`` of one task."""
    status, marker, text = observer.execute(
        sa.select(Task.status, Task.interaction_protocol_version, Task.input).where(
            Task.id == task_id
        )
    ).one()
    return status, marker, text


def _row(observer: Session, row_id: int) -> tuple[str, str | None]:
    """``(status, terminal_reason)`` of one interaction row."""
    row = row_state(observer, row_id)
    return row.status, row.terminal_reason


def _ws_manager() -> MagicMock:
    return MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )


def _completed_result() -> dict[str, Any]:
    context = SimpleNamespace(
        messages=[SimpleNamespace(role="user", metadata={"turn_id": "deferred-turn"})]
    )
    return {
        "status": "completed",
        "success": True,
        "output": "Applied",
        "agent_result": {"context": context},
    }


_Injection = UserMessageInjectionOutcome | BaseException | Callable[..., Awaitable[Any]]


def _agent(injection: _Injection) -> MagicMock:
    """An agent whose ``post_user_message`` returns ``injection`` when it is
    an outcome, and otherwise raises it or runs it."""
    agent = MagicMock()
    if isinstance(injection, UserMessageInjectionOutcome):
        agent.post_user_message = AsyncMock(return_value=injection)
    else:
        agent.post_user_message = AsyncMock(side_effect=injection)
    return agent


async def _resume_deferred(
    agent: MagicMock, seeded: _Seeded, *, with_message: bool
) -> object:
    pending: dict[str, Any] | None = None
    if with_message:
        pending = {
            "execution_message": "Here is the answer",
            "display_message": "Here is the answer",
            "files": [],
            "turn_id": "deferred-turn",
            "interaction_id": seeded.row_id,
        }
    with (
        patch("xagent.web.api.websocket.manager", _ws_manager()),
        patch(
            "xagent.web.services.task_execution.background_task_manager.promote_resume_task"
        ),
    ):
        return await execute_resume_background(
            task_id=seeded.task_id,
            agent_service=agent,
            task_owner_user_id=seeded.owner_id,
            pending_user_message=pending,
            expected_run_id=_RUN,
        )


@dataclass(frozen=True)
class _Site:
    """One compensation caller's resume path and the task it starts from."""

    # Seeded, and the status an abandoned resume restores.
    status: TaskStatus
    source: str
    # The module the caller resolves classify_injection and
    # clear_interaction_marker_if_unpaired from.
    module: str
    # task_resume's prelease acquisition function; the deferred resume has
    # none.
    acquire: str | None


_DEFERRED = _Site(
    TaskStatus.PAUSED,
    "sdk",
    "xagent.web.services.task_execution",
    None,
)
_A2A = _Site(
    TaskStatus.WAITING_FOR_USER,
    "a2a",
    "xagent.web.services.task_resume",
    "_acquire_a2a_resume_prelease_sync",
)
_REPLY = _Site(
    TaskStatus.WAITING_FOR_USER,
    "sdk",
    "xagent.web.services.task_resume",
    "_acquire_reply_prelease_sync",
)


def _seed_site(
    pg_sessions: sessionmaker[Session], site: _Site, *, row_status: str = "active"
) -> _Seeded:
    return _seed(
        pg_sessions,
        status=site.status,
        source=site.source,
        with_agent=site.acquire is not None,
        row_status=row_status,
    )


async def _resume_input(site: _Site, seeded: _Seeded, *, message_id: str) -> object:
    """Answer the seeded task through the A2A or v1 reply entry point."""
    assert seeded.agent_id is not None
    if site is _A2A:
        return await resume_a2a_task(
            agent_id=seeded.agent_id,
            task_owner_user_id=seeded.owner_id,
            task_id=seeded.task_id,
            previous_run_id=_RUN,
            resumable_status=TaskStatus.WAITING_FOR_USER,
            text="answer",
            message_id=message_id,
        )
    assert site is _REPLY
    return await resume_task_reply(
        TaskReplyInput(
            task_id=seeded.task_id,
            agent_id=seeded.agent_id,
            task_owner_user_id=seeded.owner_id,
            run_id=_RUN,
            status=TaskStatus.WAITING_FOR_USER,
            text="answer",
            actor_user_id=seeded.owner_id,
        )
    )


async def _cancel_during_acquisition(site: _Site, seeded: _Seeded) -> object:
    """Cancel the resume while its prelease acquisition is still in its
    worker thread, after that acquisition committed.

    ``acquire_task_lease_cancellation_safe`` then hands the committed lease
    to the restore it was given as its cleanup callback. The pre-injection
    read is never reached, which shows the restore came from that callback
    and not from the resume body's own cleanup.
    """
    assert site.acquire is not None
    acquire = getattr(task_resume, site.acquire)
    committed = threading.Event()
    release = threading.Event()

    def acquire_then_hold(**kwargs: Any) -> object:
        lease = acquire(**kwargs)
        committed.set()
        release.wait(30)
        return lease

    with (
        patch.object(task_resume, site.acquire, acquire_then_hold),
        patch.object(
            task_resume,
            "active_interaction_id_sync",
            wraps=task_resume.active_interaction_id_sync,
        ) as pre_injection_read,
    ):
        resume = asyncio.ensure_future(
            _resume_input(site, seeded, message_id="cancelled-message")
        )
        try:
            assert await asyncio.to_thread(committed.wait, 30)
            resume.cancel()
        finally:
            release.set()
        try:
            return await resume
        finally:
            assert pre_injection_read.call_count == 0


async def _accept_then_cancel(*_args: object, **_kwargs: object) -> None:
    _record_injection_outcome(UserMessageInjectionOutcome.POSTED_FRESH)
    raise asyncio.CancelledError()


@dataclass(frozen=True)
class _Route:
    """One way an abandoned resume reaches its site's compensation caller."""

    site: _Site
    # What post_user_message returns, raises or runs; None when the route
    # never reaches it.
    injection: _Injection | None = None
    agent_lookup_error: BaseException | None = None
    # Deferred only: the resume call fails and no message is pending.
    resume_error: BaseException | None = None
    cancel_during_acquisition: bool = False
    # None: the call returns None. A type: it raises one. An instance: it
    # raises that very object.
    raises: type[BaseException] | BaseException | None = None
    # What the caller's classify_injection returned, in order.
    dispositions: tuple[InjectionDisposition, ...] = ()


async def _abandon(route: _Route, seeded: _Seeded) -> object:
    if route.site is _DEFERRED:
        if route.resume_error is not None:
            agent = MagicMock(
                resume_execution_by_id=AsyncMock(side_effect=route.resume_error)
            )
            return await _resume_deferred(agent, seeded, with_message=False)
        assert route.injection is not None
        return await _resume_deferred(
            _agent(route.injection), seeded, with_message=True
        )
    if route.cancel_during_acquisition:
        return await _cancel_during_acquisition(route.site, seeded)
    if route.agent_lookup_error is not None:
        lookup = AsyncMock(side_effect=route.agent_lookup_error)
    else:
        assert route.injection is not None
        lookup = AsyncMock(return_value=_agent(route.injection))
    with patch(_GET_AGENT_MANAGER, return_value=MagicMock(get_agent_for_task=lookup)):
        return await _resume_input(route.site, seeded, message_id="abandoned-message")


async def _settle(route: _Route, abandoned: Awaitable[object]) -> None:
    if route.raises is None:
        assert await abandoned is None
        return
    expected = route.raises if isinstance(route.raises, type) else type(route.raises)
    with pytest.raises(expected) as excinfo:
        await abandoned
    if isinstance(route.raises, BaseException):
        assert excinfo.value is route.raises


@contextmanager
def _compensation_spies(
    site: _Site,
) -> Iterator[tuple[MagicMock, list[InjectionDisposition]]]:
    """Spy on the site's compensation call and record each disposition its
    caller classified."""
    dispositions: list[InjectionDisposition] = []

    def recording_classify(*args: Any, **kwargs: Any) -> InjectionDisposition:
        disposition = classify_injection(*args, **kwargs)
        dispositions.append(disposition)
        return disposition

    with (
        patch(
            f"{site.module}.clear_interaction_marker_if_unpaired",
            wraps=clear_interaction_marker_if_unpaired,
        ) as clear,
        patch(f"{site.module}.classify_injection", recording_classify),
    ):
        yield clear, dispositions


# --------------------------------------------------------------------------
# Close paths
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_online_chat_close_leaves_marker_and_row_paired(
    pg_sessions: sessionmaker[Session], live_task_lease
) -> None:
    seeded = _seed(pg_sessions, status=TaskStatus.RUNNING, source="sdk")
    with pg_sessions() as db:
        live_task_lease(db, db.get(Task, seeded.task_id))
    owner = _owner(pg_sessions, seeded.owner_id)
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        return_value=UserMessageInjectionOutcome.POSTED_FRESH
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = ResumeReservationOutcome.RESERVED
    bg_mgr.running_tasks.get.return_value = None

    with (
        patch(
            _GET_AGENT_MANAGER,
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.api.websocket.manager", _ws_manager()),
        patch("xagent.web.services.task_execution.background_task_manager", bg_mgr),
        patch(
            "xagent.web.services.task_execution.execute_resume_background", AsyncMock()
        ),
    ):
        await handle_task_message(
            _make_command_reply(MagicMock()),
            seeded.task_id,
            {
                "message": "Here is the answer",
                "client_message_id": "online-turn",
                "user": owner,
                "files": [],
            },
        )

    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=seeded.task_id, run_id=_RUN)
        assert _task_state(observer, seeded.task_id)[0] == TaskStatus.RUNNING
        assert _row(observer, seeded.row_id) == (
            "terminated",
            "answered_via_legacy_resume",
        )
    assert agent.post_user_message.await_count == 1


@pytest.mark.asyncio
async def test_deferred_injection_close_leaves_marker_and_row_paired_while_running(
    pg_sessions: sessionmaker[Session],
) -> None:
    seeded = _seed_site(pg_sessions, _DEFERRED)
    samples: list[tuple[TaskStatus, AssertionError | None]] = []

    async def sample_then_complete(*_args: object, **_kwargs: object) -> Any:
        # Raising here would be swallowed by execute_resume_background's own
        # error handling, so the failure is recorded instead.
        with pg_sessions() as observer:
            failure: AssertionError | None = None
            try:
                assert_marker_residue(observer, task_id=seeded.task_id, run_id=_RUN)
            except AssertionError as exc:
                failure = exc
            samples.append((_task_state(observer, seeded.task_id)[0], failure))
        return _completed_result()

    agent = _agent(UserMessageInjectionOutcome.POSTED_FRESH)
    agent.resume_execution_by_id = AsyncMock(side_effect=sample_then_complete)

    await _resume_deferred(agent, seeded, with_message=True)

    assert samples == [(TaskStatus.RUNNING, None)]
    with pg_sessions() as observer:
        assert _row(observer, seeded.row_id) == (
            "terminated",
            "answered_via_legacy_resume",
        )
    assert agent.post_user_message.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "site", [pytest.param(_A2A, id="a2a"), pytest.param(_REPLY, id="reply")]
)
async def test_resume_input_close_leaves_marker_and_row_paired(
    pg_sessions: sessionmaker[Session], site: _Site
) -> None:
    seeded = _seed_site(pg_sessions, site)
    agent = _agent(UserMessageInjectionOutcome.POSTED_FRESH)
    bg_mgr = MagicMock()
    bg_mgr.reserve_resume.return_value = True
    bg_mgr.running_tasks.get.return_value = None
    resume_background = AsyncMock()

    with (
        patch(
            _GET_AGENT_MANAGER,
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch("xagent.web.services.task_execution.background_task_manager", bg_mgr),
        patch(
            "xagent.web.services.task_execution.execute_resume_background",
            resume_background,
        ),
    ):
        await _resume_input(site, seeded, message_id="answer-message")
    # The stubbed background resume would have owned the heartbeat.
    handoff = resume_background.call_args.kwargs
    handoff["preacquired_heartbeat_stop"].set()
    await handoff["preacquired_heartbeat_task"]

    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=seeded.task_id, run_id=_RUN)
        status, _marker, text = _task_state(observer, seeded.task_id)
        assert (status, text) == (TaskStatus.RUNNING, "answer")
        assert _row(observer, seeded.row_id) == (
            "terminated",
            "answered_via_legacy_resume",
        )
    assert agent.post_user_message.await_count == 1


class _SimulatedCrash(Exception):
    pass


def test_close_and_marker_clear_commit_together(
    pg_sessions: sessionmaker[Session],
) -> None:
    """A failure after the close statement and before the marker clear
    leaves both halves as they were: they are one transaction. The close
    statement is shown to have matched the row before the failure, so the
    final state is a rollback, not a close that never ran."""
    with pg_sessions() as db:
        task_id = seed_task_with_run(db, run_id=_RUN, marker=1)
        row_id = seed_active_row(db, task_id=task_id, run_id=_RUN)

    def crash_before_the_marker_clear(**_kwargs: object) -> sa.Exists:
        raise _SimulatedCrash("process died before the marker clear")

    classify_rowcount = MagicMock(wraps=task_interaction_close._classify_close_rowcount)
    with pytest.MonkeyPatch.context() as crash:
        crash.setattr(
            task_interaction_close, "_classify_close_rowcount", classify_rowcount
        )
        # Both marker clears in the module build their condition from this
        # function; the close path's clear is the one reached here, after
        # its close statement has run.
        crash.setattr(
            task_interaction_close,
            "_active_row_exists",
            crash_before_the_marker_clear,
        )
        with pytest.raises(_SimulatedCrash):
            close_legacy_resume_interaction_sync(
                task_id=task_id, run_id=_RUN, interaction_id=row_id
            )

    assert classify_rowcount.call_args_list == [
        call(1, task_id=task_id, run_id=_RUN, unmatched_row=None)
    ]
    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=task_id, run_id=_RUN)
        assert _row(observer, row_id) == ("active", None)
        assert _task_state(observer, task_id)[1] == 1


# --------------------------------------------------------------------------
# Compensation paths
# --------------------------------------------------------------------------


def _lookup_failure(site: _Site) -> _Route:
    error = RuntimeError("agent lookup failed")
    return _Route(
        site,
        agent_lookup_error=error,
        raises=error,
        dispositions=(InjectionDisposition.FAILED_BEFORE_WRITE,),
    )


_DEFERRED_NOT_WRITTEN = _Route(
    _DEFERRED,
    injection=UserMessageInjectionRejectedError("not written"),
    dispositions=(InjectionDisposition.NOT_ACCEPTED_RETRYABLE,),
)
_A2A_NOT_POSTED = _Route(
    _A2A,
    injection=UserMessageInjectionOutcome.NOT_POSTED,
    raises=TaskResumeNotResumableError,
    dispositions=(InjectionDisposition.DEFER,),
)
_REPLY_NOT_POSTED = _Route(
    _REPLY,
    injection=UserMessageInjectionOutcome.NOT_POSTED,
    raises=TaskResumeNotResumableError,
    dispositions=(InjectionDisposition.DEFER,),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        pytest.param(
            _Route(
                _DEFERRED,
                resume_error=CheckpointUnavailableError("checkpoint read failed"),
            ),
            id="not_injected-deferred-resume_checkpoint_unavailable",
        ),
        pytest.param(
            _Route(_A2A, cancel_during_acquisition=True, raises=asyncio.CancelledError),
            id="acquisition_cancelled-a2a-settlement_callback",
        ),
        pytest.param(
            _Route(
                _REPLY, cancel_during_acquisition=True, raises=asyncio.CancelledError
            ),
            id="acquisition_cancelled-reply-settlement_callback",
        ),
        pytest.param(_DEFERRED_NOT_WRITTEN, id="not_written-deferred-rejected_error"),
        pytest.param(_A2A_NOT_POSTED, id="not_written-a2a-not_posted"),
        pytest.param(
            _Route(
                _A2A,
                injection=UserMessageInjectionOutcome.REJECTED_RETRYABLE,
                raises=TaskResumeNotAcceptedError,
                dispositions=(InjectionDisposition.NOT_ACCEPTED_RETRYABLE,),
            ),
            id="not_written-a2a-rejected_retryable",
        ),
        pytest.param(
            _Route(
                _A2A,
                injection=UserMessageInjectionRejectedError("not written"),
                raises=TaskResumeNotAcceptedError,
                dispositions=(InjectionDisposition.NOT_ACCEPTED_RETRYABLE,),
            ),
            id="not_written-a2a-rejected_error",
        ),
        pytest.param(_REPLY_NOT_POSTED, id="not_written-reply-not_posted"),
        pytest.param(
            _Route(
                _REPLY,
                injection=UserMessageInjectionOutcome.REJECTED_RETRYABLE,
                raises=TaskResumeNotAcceptedError,
                dispositions=(InjectionDisposition.NOT_ACCEPTED_RETRYABLE,),
            ),
            id="not_written-reply-rejected_retryable",
        ),
        pytest.param(
            _Route(
                _REPLY,
                injection=UserMessageInjectionRejectedError("not written"),
                raises=TaskResumeNotAcceptedError,
                dispositions=(InjectionDisposition.NOT_ACCEPTED_RETRYABLE,),
            ),
            id="not_written-reply-rejected_error",
        ),
        pytest.param(
            _Route(
                _DEFERRED,
                injection=CheckpointUnavailableError("checkpoint read failed"),
                dispositions=(InjectionDisposition.FAILED_BEFORE_WRITE,),
            ),
            id="failed_before_write-deferred-checkpoint_unavailable",
        ),
        pytest.param(
            _Route(
                _A2A,
                injection=CheckpointUnavailableError("checkpoint read failed"),
                raises=TaskResumeRetryableError,
                dispositions=(InjectionDisposition.FAILED_BEFORE_WRITE,),
            ),
            id="failed_before_write-a2a-checkpoint_unavailable",
        ),
        pytest.param(
            _lookup_failure(_A2A), id="failed_before_write-a2a-agent_lookup_error"
        ),
        pytest.param(
            _Route(
                _REPLY,
                injection=CheckpointUnavailableError("checkpoint read failed"),
                raises=TaskResumeRetryableError,
                dispositions=(InjectionDisposition.FAILED_BEFORE_WRITE,),
            ),
            id="failed_before_write-reply-checkpoint_unavailable",
        ),
        pytest.param(
            _lookup_failure(_REPLY), id="failed_before_write-reply-agent_lookup_error"
        ),
        pytest.param(
            _Route(
                _A2A,
                injection=_accept_then_cancel,
                raises=asyncio.CancelledError,
                dispositions=(InjectionDisposition.ACCEPTED,),
            ),
            id="accepted_then_cancelled-a2a",
        ),
        pytest.param(
            _Route(
                _REPLY,
                injection=_accept_then_cancel,
                raises=asyncio.CancelledError,
                dispositions=(InjectionDisposition.ACCEPTED,),
            ),
            id="accepted_then_cancelled-reply",
        ),
    ],
)
async def test_abandoned_resume_keeps_a_marker_that_names_an_active_row(
    pg_sessions: sessionmaker[Session], route: _Route
) -> None:
    seeded = _seed_site(pg_sessions, route.site)

    with _compensation_spies(route.site) as (clear, dispositions):
        await _settle(route, _abandon(route, seeded))

    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=seeded.task_id, run_id=_RUN)
        assert _task_state(observer, seeded.task_id)[:2] == (route.site.status, 1)
        assert _row(observer, seeded.row_id) == ("active", None)
    assert clear.call_args_list == [call(ANY, task_id=seeded.task_id, run_id=_RUN)]
    assert dispositions == list(route.dispositions)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        pytest.param(_DEFERRED_NOT_WRITTEN, id="deferred"),
        pytest.param(_A2A_NOT_POSTED, id="a2a"),
        pytest.param(_REPLY_NOT_POSTED, id="reply"),
    ],
)
async def test_abandoned_resume_clears_a_marker_whose_run_has_no_active_row(
    pg_sessions: sessionmaker[Session], route: _Route
) -> None:
    """The starting point, marker 1 while the run's only interaction row is
    already terminated, is one the pairing contract does not let a
    production writer leave behind; the test writes it directly so that
    the compensation path's clear has a marker to clear. Every other
    compensation test here starts from an active row, where the clear must
    leave the marker alone."""
    seeded = _seed_site(pg_sessions, route.site, row_status="terminated")

    with _compensation_spies(route.site) as (clear, dispositions):
        await _settle(route, _abandon(route, seeded))

    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=seeded.task_id, run_id=_RUN)
        assert _task_state(observer, seeded.task_id)[:2] == (route.site.status, None)
        assert _row(observer, seeded.row_id) == ("terminated", "deadline_elapsed")
    assert clear.call_args_list == [call(ANY, task_id=seeded.task_id, run_id=_RUN)]
    assert dispositions == list(route.dispositions)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("site", "busy", "message"),
    [
        pytest.param(
            _A2A,
            RuntimeError,
            "Task {task_id} already has a resume in progress",
            id="a2a",
        ),
        pytest.param(_REPLY, TaskResumeBusyError, "", id="reply"),
    ],
)
async def test_abandoned_resume_after_a_completed_answer_leaves_nothing_behind(
    pg_sessions: sessionmaker[Session],
    site: _Site,
    busy: type[Exception],
    message: str,
) -> None:
    """The resume-input close commits, then reserving the background resume
    fails, so the prelease restore and its compensation run after the row
    was retired and the marker cleared."""
    seeded = _seed_site(pg_sessions, site)
    agent = _agent(UserMessageInjectionOutcome.POSTED_FRESH)

    with (
        patch(
            _GET_AGENT_MANAGER,
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch(
            "xagent.web.services.task_execution.background_task_manager.reserve_resume",
            return_value=False,
        ),
        _compensation_spies(site) as (clear, dispositions),
        pytest.raises(busy) as excinfo,
    ):
        await _resume_input(site, seeded, message_id="answered-message")

    assert str(excinfo.value) == message.format(task_id=seeded.task_id)
    with pg_sessions() as observer:
        assert_marker_residue(observer, task_id=seeded.task_id, run_id=_RUN)
        assert _task_state(observer, seeded.task_id)[:2] == (
            TaskStatus.WAITING_FOR_USER,
            None,
        )
        assert_interaction_marker_invariant(observer, task_id=seeded.task_id)
        assert _row(observer, seeded.row_id) == (
            "terminated",
            "answered_via_legacy_resume",
        )
    assert clear.call_args_list == [call(ANY, task_id=seeded.task_id, run_id=_RUN)]
    assert dispositions == [InjectionDisposition.ACCEPTED]

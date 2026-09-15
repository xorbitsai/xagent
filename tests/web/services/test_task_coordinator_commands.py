"""Exercise the production command/owner boundary, including cross-process ingress."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import (
    task_command_execution,
    task_command_transport,
    task_completion,
    task_coordinator_runtime,
    task_event_bridge,
    task_execution,
    task_lease_service,
    task_orchestrator,
    task_resume,
    task_resume_command,
    task_start,
)


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def database_url(request, tmp_path):
    if request.param == "postgresql":
        with disposable_database_factory("coordinator_commands") as make:
            engine = make("host")
            with engine.connect() as connection:
                name = connection.execute(
                    text("SELECT current_database()")
                ).scalar_one()
            yield (
                make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
                .set(database=name)
                .render_as_string(hide_password=False)
            )
    else:
        yield f"sqlite:///{tmp_path / 'coordinator.db'}"


@pytest.fixture
async def host(tmp_path, monkeypatch, database_url):
    url = database_url
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    for module in (
        task_coordinator_runtime,
        task_command_transport,
        task_lease_service,
    ):
        monkeypatch.setattr(module, "get_runner_id", lambda: "worker-1")
    monkeypatch.setattr(
        task_coordinator_runtime, "get_task_lease_heartbeat_seconds", lambda: 0.05
    )
    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    monkeypatch.setattr(
        task_execution,
        "background_task_manager",
        task_execution.BackgroundTaskManager(),
    )
    monkeypatch.setattr(
        task_lease_service,
        "_get_task_lease_heartbeat_manager",
        Mock(
            side_effect=AssertionError(
                "Shared execution must only observe its coordinator"
            )
        ),
    )
    init_db(db_url=url)
    with get_session_local()() as db:
        owner = User(username="owner", password_hash="unused")
        db.add(owner)
        db.flush()
        agent = Agent(user_id=owner.id, name="test agent")
        db.add(agent)
        db.commit()
        owner_id, agent_id = owner.id, agent.id
    yield SimpleNamespace(owner=owner_id, agent=agent_id, url=url, root=tmp_path)
    await task_coordinator_runtime.close_task_coordinators()
    await task_execution.background_task_manager.shutdown()
    Base.metadata.drop_all(bind=get_engine())
    get_engine().dispose()


def claim(task_id):
    with get_session_local()() as db:
        row = (
            db.query(TaskExecutionCommand)
            .filter_by(task_id=task_id, status="pending")
            .order_by(TaskExecutionCommand.id)
            .first()
        )
        return (
            None
            if row is None
            else task_command_transport.claim_task_command(
                db, runner_id="worker-1", command_db_id=row.id
            )
        )


async def eventually(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


async def create(host):
    return await task_start.create_sdk_task(
        agent_id=host.agent,
        task_owner_user_id=host.owner,
        actor_user_id=host.owner,
        message="first",
        file_ids=(),
        connector_runtime_context=(),
        timezone=None,
    )


def waiting_task(host):
    with get_session_local()() as db:
        task = Task(
            user_id=host.owner,
            agent_id=host.agent,
            title="waiting",
            source="sdk",
            status=TaskStatus.WAITING_FOR_USER,
            run_id="reply-run",
            state_version=2,
            control_state="waiting_for_user",
        )
        db.add(task)
        db.commit()
        return task_resume.TaskReplyInput(
            task_id=task.id,
            agent_id=host.agent,
            task_owner_user_id=host.owner,
            actor_user_id=host.owner,
            run_id=task.run_id,
            status=task.status,
            text="answer",
            command_id="original-reply",
        )


@pytest.mark.asyncio
async def test_remote_append_preserves_owner_until_real_finalizer_and_cleanup_exit(
    host, monkeypatch
):
    terminal, release, second_finished = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    leases = []
    monkeypatch.setattr(
        task_orchestrator, "load_task_setup_snapshot_sync", lambda *a, **kw: object()
    )
    monkeypatch.setattr(
        task_orchestrator, "resolve_execution_scope", lambda *a: object()
    )
    monkeypatch.setattr(task_orchestrator, "_get_agent_manager", Mock())

    async def execute(**kwargs):
        lease = kwargs["task_lease"]
        leases.append(lease)
        result = await asyncio.to_thread(
            task_execution._finalize_task_execution_result_isolated,
            task_id=lease.task_id,
            task_user_id=host.owner,
            pre_run_status=TaskStatus.RUNNING,
            result={"success": True, "output": "done"},
            expected_run_id=lease.run_id,
            task_lease=lease,
            resolved_scope_segments=(),
            prepared_outputs=task_execution._PreparedTaskFileOutputs((), (), ()),
        )
        assert result.terminal_state_committed
        if len(leases) == 1:
            terminal.set()
            await release.wait()
        else:
            second_finished.set()

    monkeypatch.setattr(task_execution, "execute_task_background", execute)
    first = await create(host)
    assert first.status == TaskStatus.PENDING
    await task_command_execution.execute_durable_task_command(claim(first.task_id))
    await asyncio.wait_for(terminal.wait(), 10)
    script = """
import asyncio, json, sys
from unittest.mock import Mock
from xagent.web.models.database import init_db
from xagent.web.services import task_start, task_event_bridge
init_db(db_url=sys.argv[1])
task_event_bridge._bridge = Mock()
async def main():
    result = await task_start.append_sdk_turn(
        task_id=int(sys.argv[2]), scope=task_start.SdkTaskScope(agent_id=int(sys.argv[3]), workforce_id=None),
        actor_user_id=int(sys.argv[4]), request_agent_id=int(sys.argv[3]), request_workforce_id=None,
        message="next", file_ids=(), connector_runtime_context=())
    print(json.dumps({"run_id": result.run_id, "status": result.status.value}))
asyncio.run(main())
"""
    followup = None
    try:
        process = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-c",
                script,
                host.url,
                str(first.task_id),
                str(host.agent),
                str(host.owner),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy(),
        )
        assert process.returncode == 0, process.stderr
        accepted = json.loads(process.stdout.splitlines()[-1])
        assert accepted["status"] == "pending"
        assert accepted["run_id"] != first.run_id
        with get_session_local()() as db:
            task = db.get(Task, first.task_id)
            assert task.run_id == first.run_id
            assert task.lease_attempt_id == leases[0].attempt_id
            assert task.input == "first"
        other = task_coordinator_runtime.TaskCoordinatorRegistry()
        other.runner_id = "worker-2"
        try:
            assert await other.ensure(first.task_id) is None
        finally:
            await other.close()
        followup = asyncio.create_task(
            task_command_execution.execute_durable_task_command(claim(first.task_id))
        )
        await asyncio.sleep(0.1)
        assert not followup.done()
        assert len(leases) == 1
        release.set()
        await asyncio.wait_for(followup, 10)
        await asyncio.wait_for(second_finished.wait(), 10)
        await eventually(
            lambda: task_completion._is_run_finished(first.task_id, accepted["run_id"])
        )
        assert leases[0].attempt_id == leases[1].attempt_id
        assert leases[0].run_id != leases[1].run_id
        with get_session_local()() as db:
            assert db.get(Task, first.task_id).status == TaskStatus.COMPLETED
    finally:
        release.set()
        if followup is not None:
            await asyncio.gather(followup, return_exceptions=True)


def test_completion_poll_does_not_contend_with_unrelated_sqlite_writer(
    host, monkeypatch
):
    if not host.url.startswith("sqlite"):
        pytest.skip("SQLite writer contention regression")
    with get_session_local()() as db:
        finished = Task(
            user_id=host.owner, title="done", status=TaskStatus.COMPLETED, run_id="done"
        )
        other = Task(user_id=host.owner, title="other", status=TaskStatus.PENDING)
        db.add_all([finished, other])
        db.commit()
        finished_id, other_id = finished.id, other.id
    engine = create_engine(host.url, connect_args={"timeout": 0.03})
    monkeypatch.setattr(
        task_completion, "get_session_local", lambda: sessionmaker(bind=engine)
    )
    try:
        with get_session_local()() as writer:
            writer.query(Task).filter_by(id=other_id).update({"title": "uncommitted"})
            assert task_completion._is_run_finished(finished_id, "done")
            writer.rollback()
    finally:
        engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["after_post", "during_post"])
async def test_unknown_reply_returns_original_identity_without_reinjection(
    host, monkeypatch, failure
):
    ctx = waiting_task(host)
    error = OperationalError(
        "UPDATE tasks", None, RuntimeError("connection interrupted")
    )
    post = AsyncMock(return_value=True)
    if failure == "during_post":
        post.side_effect = error
    else:
        monkeypatch.setattr(
            task_resume, "_update_reply_input_sync", Mock(side_effect=error)
        )
    manager = SimpleNamespace(
        get_agent_for_task=AsyncMock(
            return_value=SimpleNamespace(post_user_message=post)
        )
    )
    monkeypatch.setattr(
        task_resume.agent_runtime_service, "get_agent_manager", lambda: manager
    )
    request = asyncio.create_task(task_resume.resume_task_reply(ctx))
    try:
        await eventually(lambda: _has_pending(ctx.task_id))
        command = claim(ctx.task_id)
        await task_command_execution.execute_durable_task_command(command)
        with pytest.raises(task_resume.TaskResumeOutcomeUnknownError) as unknown:
            await asyncio.wait_for(request, 10)
        assert unknown.value.command_id == ctx.command_id
        assert (
            task_resume_command._read_reply_outcome(command.id)["outcome"] == "unknown"
        )
        with pytest.raises(task_resume.TaskResumeOutcomeUnknownError):
            await task_resume.resume_task_reply(replace(ctx, status=TaskStatus.RUNNING))
        post.assert_awaited_once()
        with get_session_local()() as db:
            task = db.get(Task, ctx.task_id)
            assert task.status == TaskStatus.RUNNING
            assert task.lease_attempt_id is not None
            assert db.query(TaskExecutionCommand).count() == 1
    finally:
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)


def _has_pending(task_id):
    with get_session_local()() as db:
        return (
            db.query(TaskExecutionCommand.id)
            .filter_by(task_id=task_id, status="pending")
            .first()
            is not None
        )


@pytest.mark.asyncio
async def test_reply_carries_authenticated_actor_after_agent_owner_transfer(
    host, monkeypatch
):
    from xagent.web.api.v1.deps import (
        AgentPrincipalSnapshot,
        ApiKeyPrincipal,
        RuntimeApiKeySnapshot,
    )
    from xagent.web.api.v1.task_reply import _prepare_reply_context_sync
    from xagent.web.schemas.v1 import ReplyRequest

    ctx = waiting_task(host)
    with get_session_local()() as db:
        actor = User(username="new owner", password_hash="unused")
        db.add(actor)
        db.flush()
        db.get(Agent, host.agent).user_id = actor.id
        db.commit()
        actor_id, actor_subject = actor.id, actor.actor_subject
    principal = ApiKeyPrincipal(
        key=RuntimeApiKeySnapshot("test"),
        agent=AgentPrincipalSnapshot(
            id=host.agent,
            user_id=actor_id,
            execution_mode="balanced",
            status="published",
            origin="user",
        ),
    )
    ctx = _prepare_reply_context_sync(
        task_id=ctx.task_id,
        principal=principal,
        request=ReplyRequest.model_validate(
            {
                "agent_id": host.agent,
                "command_id": "reply",
                "message": {"role": "user", "content": "answer"},
            }
        ),
    )
    assert ctx.actor_user_id == actor_id
    assert ctx.task_owner_user_id == host.owner
    command_id = task_resume_command._admit_reply(ctx, "sdk", "", "reply")
    prepare = AsyncMock()
    monkeypatch.setattr(task_resume, "resume_task_reply", prepare)
    await task_command_execution.execute_durable_task_command(claim(ctx.task_id))
    delivered = prepare.call_args.args[0]
    assert delivered.actor_user_id == actor_id
    assert delivered.task_owner_user_id == host.owner
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, command_id)
        assert command.actor_user_id == actor_id
        assert command.actor_subject == actor_subject
        assert command.task_owner_user_id == host.owner


@pytest.mark.asyncio
async def test_controls_apply_while_execution_is_running(host, monkeypatch):
    started, finish = asyncio.Event(), asyncio.Event()
    first = await create(host)

    async def schedule(**kwargs):
        # This is the normal registration boundary; keep a long execution alive.
        async def execute():
            started.set()
            await finish.wait()
            with get_session_local()() as db:
                task = db.get(Task, first.task_id)
                task.status = TaskStatus.COMPLETED
                task.control_state = "completed"
                db.commit()

        handle = asyncio.create_task(execute())
        task_execution.background_task_manager.register_task(first.task_id, handle)
        return handle

    from xagent.web.services import task_start_consumer

    monkeypatch.setattr(task_start_consumer, "_schedule_committed_turn", schedule)
    await task_command_execution.execute_durable_task_command(claim(first.task_id))
    await asyncio.wait_for(started.wait(), 5)
    with get_session_local()() as db:
        task_command_transport.stage_task_command(
            db,
            task_id=first.task_id,
            actor_user_id=host.owner,
            command_id="pause",
            kind=task_command_transport.TaskCommandKind.PAUSE,
            payload={},
        )
        db.commit()
    control = AsyncMock(return_value={})
    monkeypatch.setattr(
        task_command_execution, "_execute_and_report_task_command", control
    )
    try:
        await asyncio.wait_for(
            task_command_execution.execute_durable_task_command(claim(first.task_id)), 2
        )
        control.assert_awaited_once()
        assert not finish.is_set()
    finally:
        finish.set()
        await task_execution.background_task_manager.wait_for_previous(first.task_id)


@pytest.mark.asyncio
async def test_shutdown_drains_command_waiting_for_execution_cleanup(host):
    first = await create(host)
    registry = task_coordinator_runtime.get_task_coordinator_registry()
    coordinator = await registry.ensure(first.task_id)
    assert coordinator is not None
    cleanup_entered, cleanup_done = asyncio.Event(), asyncio.Event()

    async def previous_execution():
        try:
            cleanup_entered.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleanup_done.set()

    previous = asyncio.create_task(previous_execution())
    coordinator.track_execution(previous)
    await cleanup_entered.wait()
    command = claim(first.task_id)
    apply = AsyncMock()
    pending = asyncio.create_task(coordinator.execute_command(command, apply))
    await eventually(coordinator._command_lock.locked)
    await asyncio.wait_for(coordinator.close(), 5)
    await asyncio.gather(pending, return_exceptions=True)
    assert cleanup_done.is_set()
    assert pending.cancelled()
    apply.assert_not_awaited()
    assert coordinator.state == task_coordinator_runtime.CoordinatorState.CLOSED
    with get_session_local()() as db:
        assert db.get(Task, first.task_id).runner_id is None
        assert db.get(TaskExecutionCommand, command.id).status == "processing"

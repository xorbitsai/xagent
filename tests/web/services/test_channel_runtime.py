from __future__ import annotations

import asyncio
import threading
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import FsspecFileStorage
from xagent.core.workspace import TaskWorkspace
from xagent.web.api.chat import AgentServiceManager
from xagent.web.models import database as database_module
from xagent.web.models.agent import Agent, AgentOrigin
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import channel_runtime
from xagent.web.services.channel_runtime import (
    ChannelAuthorizationError,
    DownloadedChannelFile,
    get_channel_owner_agent,
    list_channel_owner_agents,
    load_active_channel_configs,
    prepare_channel_task,
    register_channel_uploaded_files,
)
from xagent.web.services.task_lease_service import TaskLease, utc_now
from xagent.web.services.task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY,
)
from xagent.web.services.uploaded_file_store import UploadedFileStore


def _create_channel_session_local(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'channel-claim.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.2,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        bind=engine,
    )
    with SessionLocal() as db:
        user = User(username="channel-claim-user", password_hash="hash")
        db.add(user)
        db.flush()
        channel = UserChannel(
            user_id=int(user.id),
            channel_type="telegram",
            channel_name="Telegram claim",
            config={"allowed_users": ["telegram-user"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        return engine, SessionLocal, int(user.id), int(channel.id)


def _add_agent(
    SessionLocal,
    *,
    user_id: int,
    name: str,
    origin: str = AgentOrigin.USER.value,
) -> int:
    with SessionLocal() as db:
        agent = Agent(user_id=user_id, name=name, origin=origin)
        db.add(agent)
        db.commit()
        return int(agent.id)


async def _create_waiting_actor_task(
    SessionLocal,
    *,
    channel_id: int,
    agent_id: int,
) -> int:
    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="prepare approval",
        channel_name="Trusted direct channel",
        agent_id=agent_id,
        new_task_is_visible=False,
        mcp_runtime_authorization_policy_required=True,
    )
    assert prepared is not None
    with SessionLocal() as db:
        task = db.get(Task, prepared.task_id)
        assert task is not None
        task.source = "external"
        db.commit()
    assert await prepared.managed_lease.finalize_result(
        status=TaskStatus.WAITING_FOR_USER
    )
    return prepared.task_id


@pytest.mark.asyncio
async def test_prepare_channel_task_binds_owned_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Research Agent")
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
        agent_id=agent_id,
    )

    assert prepared is not None
    assert prepared.requested_agent_missing is False
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == prepared.task_id).one()
        assert task.agent_id == agent_id
        assert task.execution_mode == "balanced"
        assert task.connector_runtime_selected_refs == []

    assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_trusted_direct_channel_path_creates_fresh_hidden_marked_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    """The trusted caller persists the fence before either task executes."""
    del mock_workspace_db
    engine, SessionLocal, _user_id, channel_id = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    task_ids: list[int] = []
    for message_text in ("first", "second"):
        prepared = await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=None,
            text=message_text,
            channel_name="Trusted direct channel",
            new_task_is_visible=False,
            mcp_runtime_authorization_policy_required=True,
        )
        assert prepared is not None
        task_ids.append(prepared.task_id)
        with SessionLocal() as db:
            task = db.query(Task).filter(Task.id == prepared.task_id).one()
            assert task.is_visible is False
            assert task.agent_config == {
                MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True
            }
        assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)

    assert task_ids[0] != task_ids[1]
    with pytest.raises(RuntimeError, match="channel reuse is unsupported"):
        await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=task_ids[0],
            text="generic reuse",
            channel_name="Telegram",
        )

    # A new manager has no warm marker cache. It must re-read the persisted
    # marker and fail before attempting to construct a policyless runtime.
    monkeypatch.setattr(
        "xagent.web.services.task_setup_snapshot.get_session_local",
        lambda: SessionLocal,
    )
    cold_manager = AgentServiceManager()
    with pytest.raises(RuntimeError, match="requires an MCP runtime authorization"):
        await cold_manager.get_agent_for_task(
            task_ids[0],
            task_owner_user_id=_user_id,
            resolved_execution_scope=None,
        )
    engine.dispose()


@pytest.mark.asyncio
async def test_actor_marker_forces_new_channel_task_hidden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, _user_id, channel_id = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="trusted actor message",
        channel_name="Trusted direct channel",
        mcp_runtime_authorization_policy_required=True,
    )

    assert prepared is not None
    with SessionLocal() as db:
        task = db.get(Task, prepared.task_id)
        assert task is not None
        assert task.is_visible is False
        assert task.agent_config == {
            MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True
        }
    assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_actor_task_persists_policy_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, _user_id, channel_id = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="trusted actor message",
        channel_name="Trusted direct channel",
        mcp_runtime_authorization_policy_required=True,
        mcp_runtime_authorization_policy_identity="actor:alice",
    )

    assert prepared is not None
    with SessionLocal() as db:
        task = db.get(Task, prepared.task_id)
        assert task is not None
        assert task.agent_config == {
            MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True,
            "mcp_runtime_authorization_policy_identity": "actor:alice",
        }
    assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_actor_interaction_reuses_exact_waiting_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        previous_run_id = task.run_id
        task.last_checkpoint_event_id = "waiting-checkpoint"
        task.output = "waiting output"
        task.error_message = "waiting error"
        db.commit()

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=task_id,
        text="approve",
        channel_name="Trusted direct channel",
        expected_owner_user_id=user_id,
        agent_id=agent_id,
        new_task_is_visible=False,
        mcp_runtime_authorization_policy_required=True,
        task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
    )

    assert prepared is not None
    assert prepared.task_id == task_id
    assert prepared.is_new_task is False
    assert prepared.requested_agent_missing is False
    with SessionLocal() as db:
        assert db.query(Task).count() == 1
        task = db.get(Task, task_id)
        assert task is not None
        assert task.status == TaskStatus.RUNNING
        assert task.run_id != previous_run_id
        assert task.last_checkpoint_event_id is None
        assert task.output is None
        assert task.error_message is None
    assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_actor_interaction_resume_claim_keeps_run_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    """``resume_run_id`` claims the waiting run instead of replacing it.

    The default claim mints a new run id and nulls the checkpoint pointers
    (pinned by ``test_actor_interaction_reuses_exact_waiting_task``), which
    leaves the waiting checkpoint unreadable in the new run's partition. A
    resume has to land in the same run, or there is nothing to resume.
    """
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        waiting_run_id = task.run_id
        assert waiting_run_id is not None
        task.last_checkpoint_event_id = "waiting-checkpoint"
        task.last_checkpoint_trace_event_id = 4242
        db.commit()

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=task_id,
        text="approve",
        channel_name="Trusted direct channel",
        expected_owner_user_id=user_id,
        agent_id=agent_id,
        new_task_is_visible=False,
        mcp_runtime_authorization_policy_required=True,
        task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
        resume_run_id=waiting_run_id,
    )

    assert prepared is not None
    assert prepared.task_id == task_id
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        assert task.status == TaskStatus.RUNNING
        # The run the checkpoint was written under, still current.
        assert task.run_id == waiting_run_id
        assert task.last_checkpoint_event_id == "waiting-checkpoint"
        assert task.last_checkpoint_trace_event_id == 4242
    assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_abandoned_resume_claim_restores_the_waiting_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    """A resume abandoned before the answer lands must stay resumable.

    Compensation exists for a claim that committed and then had nobody to
    execute it. For a fresh claim, FAILED is the honest record. For a resume
    it is destructive: the loader and the next claim both require
    WAITING_FOR_USER, so failing the row makes the pending approval
    permanently unanswerable -- checkpoint intact and unreachable.
    """
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        waiting_run_id = task.run_id
        task.last_checkpoint_event_id = "waiting-checkpoint"
        task.last_checkpoint_trace_event_id = 77
        db.commit()

    def _explode(_lease):
        raise RuntimeError("heartbeat could not start")

    monkeypatch.setattr(channel_runtime, "start_managed_task_lease", _explode)
    with pytest.raises(RuntimeError):
        await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=task_id,
            text="approve",
            channel_name="Trusted direct channel",
            expected_owner_user_id=user_id,
            agent_id=agent_id,
            new_task_is_visible=False,
            mcp_runtime_authorization_policy_required=True,
            task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
            resume_run_id=waiting_run_id,
        )
    monkeypatch.undo()
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    # Nothing was injected; the claim is abandoned. Driven through the real
    # trigger -- heartbeat startup failing after the claim committed -- rather
    # than by calling compensation directly, so the test covers the path
    # production actually takes.

    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.run_id == waiting_run_id
        assert task.last_checkpoint_event_id == "waiting-checkpoint"
        assert task.last_checkpoint_trace_event_id == 77

    # And the same resume can be taken again.
    again = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=task_id,
        text="approve",
        channel_name="Trusted direct channel",
        expected_owner_user_id=user_id,
        agent_id=agent_id,
        new_task_is_visible=False,
        mcp_runtime_authorization_policy_required=True,
        task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
        resume_run_id=waiting_run_id,
    )
    assert again is not None
    assert await again.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_abandoned_resume_of_a_non_waiting_task_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    """Only a genuinely waiting task is restored to waiting.

    ``resume_run_id`` is not restricted to actor interactions, and outside
    that mode the claim carries no ``expected_status`` -- only "not
    RUNNING". So a resume can commit against an ordinary channel task that
    was never parked on a question. Assuming WAITING_FOR_USER for every
    resume would then write a status the task never held, presenting a row
    as answerable when no interaction is pending.
    """
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    # An ordinary channel task, not actor-marked: DEFAULT mode refuses to
    # reuse an actor-marked row, so that is the shape this path can reach.
    seed = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="ordinary turn",
        channel_name="Trusted direct channel",
        agent_id=agent_id,
    )
    assert seed is not None
    task_id = seed.task_id
    assert await seed.managed_lease.finalize_result(status=TaskStatus.PAUSED)
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        run_id = task.run_id
        assert task.status == TaskStatus.PAUSED

    def _explode(_lease):
        raise RuntimeError("heartbeat could not start")

    monkeypatch.setattr(channel_runtime, "start_managed_task_lease", _explode)
    with pytest.raises(RuntimeError):
        await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=task_id,
            text="resume me",
            channel_name="Trusted direct channel",
            expected_owner_user_id=user_id,
            agent_id=agent_id,
            resume_run_id=run_id,
        )

    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        # Not restored to WAITING_FOR_USER, which it never was.
        assert task.status == TaskStatus.FAILED
    engine.dispose()


@pytest.mark.asyncio
async def test_abandoned_fresh_claim_still_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    """The restore must not leak into fresh claims: those still fail."""
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    def _explode(_lease):
        raise RuntimeError("heartbeat could not start")

    monkeypatch.setattr(channel_runtime, "start_managed_task_lease", _explode)
    with pytest.raises(RuntimeError):
        await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=None,
            text="hello",
            channel_name="Trusted direct channel",
            expected_owner_user_id=user_id,
            agent_id=agent_id,
        )

    with SessionLocal() as db:
        task = db.query(Task).order_by(Task.id.desc()).first()
        assert task is not None
        assert task.status == TaskStatus.FAILED
    engine.dispose()


@pytest.mark.asyncio
async def test_resume_claim_refuses_a_run_id_that_is_not_the_waiting_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    """A stale or guessed run id must not claim the task.

    The run id names which execution is being resumed. If a mismatched id
    still claimed, a caller holding a stale id would resume a run that has
    since moved on -- and take the waiting task's lease while doing it.
    """
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        waiting_run_id = task.run_id
        task.last_checkpoint_event_id = "waiting-checkpoint"
        db.commit()

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=task_id,
        text="approve",
        channel_name="Trusted direct channel",
        expected_owner_user_id=user_id,
        agent_id=agent_id,
        new_task_is_visible=False,
        mcp_runtime_authorization_policy_required=True,
        task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
        resume_run_id="not-the-waiting-run",
    )

    assert prepared is None
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        # Untouched: still waiting, still on its own run, checkpoint intact.
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.run_id == waiting_run_id
        assert task.last_checkpoint_event_id == "waiting-checkpoint"
    engine.dispose()


@pytest.mark.asyncio
async def test_actor_interaction_rejects_live_waiting_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    """A waiting task is not reclaimable until its prior lease is released."""
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(
        "xagent.web.services.task_lease_service.get_runner_id",
        lambda: "shared-runner",
    )
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        task.runner_id = "shared-runner"
        task.run_id = "waiting-run"
        task.lease_attempt_id = "waiting-attempt"
        task.lease_expires_at = utc_now() + timedelta(minutes=5)
        db.commit()

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=task_id,
        text="approve",
        channel_name="Trusted direct channel",
        expected_owner_user_id=user_id,
        agent_id=agent_id,
        new_task_is_visible=False,
        mcp_runtime_authorization_policy_required=True,
        task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
    )

    if prepared is not None:
        await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
        pytest.fail("The live waiting lease was overwritten")
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_USER
        assert task.run_id == "waiting-run"
        assert task.lease_attempt_id == "waiting-attempt"
    engine.dispose()


@pytest.mark.asyncio
async def test_actor_interaction_claim_rechecks_waiting_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )
    load_task = channel_runtime._load_actor_interaction_task

    def terminate_after_validation(db, **kwargs):
        task, agent = load_task(db, **kwargs)
        task.status = TaskStatus.COMPLETED
        db.commit()
        return task, agent

    monkeypatch.setattr(
        channel_runtime,
        "_load_actor_interaction_task",
        terminate_after_validation,
    )

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=task_id,
        text="approve",
        channel_name="Trusted direct channel",
        expected_owner_user_id=user_id,
        agent_id=agent_id,
        new_task_is_visible=False,
        mcp_runtime_authorization_policy_required=True,
        task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
    )

    if prepared is not None:
        await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
        pytest.fail("The terminal actor task was claimed again")
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["agent_id", "source", "policy"])
async def test_actor_interaction_claim_rechecks_task_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
    changed_field: str,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )
    load_task = channel_runtime._load_actor_interaction_task

    def change_lineage_after_validation(db, **kwargs):
        task, agent = load_task(db, **kwargs)
        if changed_field == "agent_id":
            task.agent_id = None
        elif changed_field == "source":
            task.source = None
        else:
            task.agent_config = {}
        db.commit()
        return task, agent

    monkeypatch.setattr(
        channel_runtime,
        "_load_actor_interaction_task",
        change_lineage_after_validation,
    )

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=task_id,
        text="approve",
        channel_name="Trusted direct channel",
        expected_owner_user_id=user_id,
        agent_id=agent_id,
        new_task_is_visible=False,
        mcp_runtime_authorization_policy_required=True,
        task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
    )

    if prepared is not None:
        await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
        pytest.fail(f"Actor task with changed {changed_field} was claimed")
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_USER
    engine.dispose()


@pytest.mark.asyncio
async def test_actor_interaction_claim_rechecks_agent_visibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    with SessionLocal() as db:
        other_user = User(username="other-owner", password_hash="hash")
        db.add(other_user)
        db.commit()
        other_user_id = int(other_user.id)
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )
    load_task = channel_runtime._load_actor_interaction_task

    def hide_agent_after_validation(db, **kwargs):
        task, agent = load_task(db, **kwargs)
        agent.user_id = other_user_id
        db.commit()
        return task, agent

    monkeypatch.setattr(
        channel_runtime,
        "_load_actor_interaction_task",
        hide_agent_after_validation,
    )

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=task_id,
        text="approve",
        channel_name="Trusted direct channel",
        expected_owner_user_id=user_id,
        agent_id=agent_id,
        new_task_is_visible=False,
        mcp_runtime_authorization_policy_required=True,
        task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
    )

    if prepared is not None:
        await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
        pytest.fail("Actor task with a hidden agent was claimed")
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        assert task.status == TaskStatus.WAITING_FOR_USER
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_field", ["source", "status", "policy"])
async def test_actor_interaction_rejects_invalid_task_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
    invalid_field: str,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        assert task is not None
        if invalid_field == "source":
            task.source = None
        elif invalid_field == "status":
            task.status = TaskStatus.FAILED
        else:
            task.agent_config = None
        db.commit()

    with pytest.raises(ChannelAuthorizationError, match="actor interaction task"):
        await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=task_id,
            text="approve",
            channel_name="Trusted direct channel",
            expected_owner_user_id=user_id,
            agent_id=agent_id,
            new_task_is_visible=False,
            mcp_runtime_authorization_policy_required=True,
            task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
        )

    with SessionLocal() as db:
        assert db.query(Task).count() == 1
    engine.dispose()


@pytest.mark.asyncio
async def test_actor_interaction_requires_task_and_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    with pytest.raises(ValueError, match="active task"):
        await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=None,
            text="approve",
            channel_name="Trusted direct channel",
            expected_owner_user_id=user_id,
            agent_id=agent_id,
            new_task_is_visible=False,
            mcp_runtime_authorization_policy_required=True,
            task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
        )

    with pytest.raises(ChannelAuthorizationError, match="actor interaction task"):
        await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=1,
            text="approve",
            channel_name="Trusted direct channel",
            expected_owner_user_id=user_id,
            agent_id=agent_id,
            new_task_is_visible=False,
            mcp_runtime_authorization_policy_required=True,
            task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
        )

    with pytest.raises(ValueError, match="actor policy"):
        await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=1,
            text="approve",
            channel_name="Trusted direct channel",
            expected_owner_user_id=user_id,
            agent_id=agent_id,
            new_task_is_visible=False,
            mcp_runtime_authorization_policy_required=False,
            task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
        )

    with SessionLocal() as db:
        assert db.query(Task).count() == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_actor_interaction_rejects_unavailable_agent_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Actor Agent")
    monkeypatch.setattr(channel_runtime, "get_session_local", lambda: SessionLocal)
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)
    task_id = await _create_waiting_actor_task(
        SessionLocal,
        channel_id=channel_id,
        agent_id=agent_id,
    )

    with pytest.raises(ChannelAuthorizationError, match="actor interaction task"):
        await prepare_channel_task(
            channel_id=channel_id,
            external_user_id="telegram-user",
            active_task_id=task_id,
            text="approve",
            channel_name="Trusted direct channel",
            expected_owner_user_id=user_id,
            agent_id=agent_id + 1,
            new_task_is_visible=False,
            mcp_runtime_authorization_policy_required=True,
            task_mode=channel_runtime.ChannelTaskMode.ACTOR_INTERACTION,
        )

    with SessionLocal() as db:
        assert db.query(Task).count() == 1
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_owner", [False, True])
async def test_prepare_channel_task_falls_back_when_agent_not_selectable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
    foreign_owner: bool,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, _user_id, channel_id = _create_channel_session_local(tmp_path)
    if foreign_owner:
        with SessionLocal() as db:
            other = User(username="other-user", password_hash="hash")
            db.add(other)
            db.commit()
            other_id = int(other.id)
        agent_id = _add_agent(SessionLocal, user_id=other_id, name="Foreign Agent")
    else:
        agent_id = 424242
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
        agent_id=agent_id,
    )

    assert prepared is not None
    assert prepared.requested_agent_missing is True
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == prepared.task_id).one()
        assert task.agent_id is None

    assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_prepare_channel_task_rejects_workforce_manager_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(
        SessionLocal,
        user_id=user_id,
        name="Workforce Manager",
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
        agent_id=agent_id,
    )

    assert prepared is not None
    assert prepared.requested_agent_missing is True
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == prepared.task_id).one()
        assert task.agent_id is None

    assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_list_channel_owner_agents_scopes_and_orders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    with SessionLocal() as db:
        other = User(username="other-user", password_hash="hash")
        db.add(other)
        db.commit()
        other_id = int(other.id)
    first_id = _add_agent(SessionLocal, user_id=user_id, name="First Agent")
    second_id = _add_agent(SessionLocal, user_id=user_id, name="Second Agent")
    _add_agent(SessionLocal, user_id=other_id, name="Foreign Agent")
    _add_agent(
        SessionLocal,
        user_id=user_id,
        name="Workforce Manager",
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )

    agents = await list_channel_owner_agents(
        channel_id=channel_id,
        external_user_id="telegram-user",
    )
    assert [agent.name for agent in agents] == ["Second Agent", "First Agent"]
    assert {agent.agent_id for agent in agents} == {first_id, second_id}

    with pytest.raises(ChannelAuthorizationError):
        await list_channel_owner_agents(
            channel_id=channel_id,
            external_user_id="unauthorized-user",
        )
    engine.dispose()


@pytest.mark.asyncio
async def test_get_channel_owner_agent_hides_foreign_agents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    with SessionLocal() as db:
        other = User(username="other-user", password_hash="hash")
        db.add(other)
        db.commit()
        other_id = int(other.id)
    owned_id = _add_agent(SessionLocal, user_id=user_id, name="Owned Agent")
    foreign_id = _add_agent(SessionLocal, user_id=other_id, name="Foreign Agent")
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )

    owned = await get_channel_owner_agent(
        channel_id=channel_id,
        external_user_id="telegram-user",
        agent_id=owned_id,
    )
    assert owned is not None
    assert owned.name == "Owned Agent"

    assert (
        await get_channel_owner_agent(
            channel_id=channel_id,
            external_user_id="telegram-user",
            agent_id=foreign_id,
        )
        is None
    )
    engine.dispose()


@pytest.mark.asyncio
async def test_prepare_channel_task_commits_creation_and_exact_claim_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
    )

    assert prepared is not None
    assert prepared.user_id == user_id
    assert prepared.is_new_task is True
    assert prepared.managed_lease.lease.task_id == prepared.task_id
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == prepared.task_id).one()
        assert task.status == TaskStatus.RUNNING
        assert task.run_id == prepared.managed_lease.lease.run_id
        assert task.runner_id == prepared.managed_lease.lease.runner_id
        assert task.lease_expires_at is not None

    assert await prepared.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


def test_prepare_channel_task_rolls_back_new_task_when_claim_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, _user_id, channel_id = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.acquire_task_lease_no_commit",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    prepared = channel_runtime._prepare_channel_task_sync(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
        expected_owner_user_id=None,
    )

    assert prepared is None
    with SessionLocal() as db:
        assert db.query(Task).count() == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_prepare_channel_task_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    worker_started = threading.Event()
    allow_worker = threading.Event()

    claim = channel_runtime._ChannelTaskClaimSnapshot(
        user_id=7,
        task_id=11,
        is_new_task=False,
        lease=TaskLease(task_id=11, runner_id="runner-a", run_id="run-a"),
    )
    managed = object()

    def blocking_prepare(**_kwargs):  # type: ignore[no-untyped-def]
        worker_started.set()
        assert threading.get_ident() != event_loop_thread
        assert allow_worker.wait(timeout=30)
        return claim

    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._prepare_channel_task_sync",
        blocking_prepare,
    )
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.start_managed_task_lease",
        lambda _lease: managed,
    )

    preparation = asyncio.create_task(
        prepare_channel_task(
            channel_id=3,
            external_user_id="telegram-user",
            active_task_id=11,
            text="hello",
            channel_name="Telegram",
        )
    )
    try:
        assert await asyncio.to_thread(worker_started.wait, 30)
        await asyncio.sleep(0)

        assert preparation.done() is False
    finally:
        allow_worker.set()
        result = await asyncio.wait_for(preparation, timeout=30)
    prepared = result
    assert prepared is not None
    assert prepared.user_id == 7
    assert prepared.task_id == 11
    assert prepared.is_new_task is False
    assert prepared.managed_lease is managed


@pytest.mark.asyncio
async def test_register_channel_uploaded_files_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event_loop_thread = threading.get_ident()
    worker_started = threading.Event()
    allow_worker = threading.Event()
    downloaded = DownloadedChannelFile(
        name="report.txt",
        path=tmp_path / "report.txt",
        mime_type="text/plain",
        size=6,
        source_id="remote-file",
    )

    def blocking_register(**_kwargs) -> tuple:  # type: ignore[no-untyped-def]
        worker_started.set()
        assert threading.get_ident() != event_loop_thread
        assert allow_worker.wait(timeout=30)
        return ()

    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._register_channel_uploaded_files_sync",
        blocking_register,
    )

    registration = asyncio.create_task(
        register_channel_uploaded_files(
            workspace=object(),
            task_id=11,
            user_id=7,
            files=(downloaded,),
        )
    )
    try:
        assert await asyncio.to_thread(worker_started.wait, 30)
        await asyncio.sleep(0)

        assert registration.done() is False
    finally:
        allow_worker.set()
        result = await asyncio.wait_for(registration, timeout=30)
    assert result == ()


@pytest.mark.asyncio
async def test_channel_durable_upload_does_not_hold_pool_connection_or_upload_twice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine = create_engine(
        f"sqlite:///{tmp_path / 'channel.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.2,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        bind=engine,
    )
    with SessionLocal() as db:
        user = User(username="channel-upload-user", password_hash="hash")
        db.add(user)
        db.flush()
        task = Task(id=71, user_id=int(user.id), title="Channel upload")
        db.add(task)
        db.commit()
        user_id = int(user.id)

    workspace = TaskWorkspace(
        id="web_task_71",
        base_dir=str(tmp_path / "workspaces"),
        db_task_id=71,
    )
    source = workspace.input_dir / "report.txt"
    source.write_text("report", encoding="utf-8")
    downloaded = DownloadedChannelFile(
        name=source.name,
        path=source,
        mime_type="text/plain",
        size=source.stat().st_size,
        source_id="remote-file",
    )

    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )

    put_started = threading.Event()
    allow_put = threading.Event()
    put_calls = 0
    original_put_file = FsspecFileStorage.put_file

    def blocking_put_file(
        self: FsspecFileStorage,
        source_path: Path,
        key: str,
        content_type: str | None = None,
    ):
        nonlocal put_calls
        put_calls += 1
        put_started.set()
        assert allow_put.wait(timeout=3)
        return original_put_file(self, source_path, key, content_type)

    monkeypatch.setattr(FsspecFileStorage, "put_file", blocking_put_file)

    registration = asyncio.create_task(
        register_channel_uploaded_files(
            workspace=workspace,
            task_id=71,
            user_id=user_id,
            files=(downloaded,),
        )
    )
    assert await asyncio.to_thread(put_started.wait, 2)

    def probe_pool() -> int:
        with SessionLocal() as db:
            return int(db.execute(text("SELECT 1")).scalar_one())

    try:
        assert await asyncio.wait_for(asyncio.to_thread(probe_pool), timeout=1) == 1
    finally:
        allow_put.set()

    registered = await registration
    assert len(registered) == 1
    assert put_calls == 1
    assert workspace.resolve_file_id(registered[0].file_id) == source
    with SessionLocal() as db:
        record = (
            db.query(UploadedFile)
            .filter(UploadedFile.file_id == registered[0].file_id)
            .one()
        )
        assert record.task_id == 71
        assert record.workspace_relative_path == "input/report.txt"
        assert record.workspace_category == "input"
        assert record.storage_status == "available"
        assert record.storage_key == (
            f"users/{user_id}/tasks/71/outputs/{record.file_id}/input/report.txt"
        )

    durable_files_before_failure = {
        path.relative_to(object_root)
        for path in object_root.rglob("*")
        if path.is_file()
    }
    failed_source = workspace.input_dir / "metadata-failure.txt"
    failed_source.write_text("must be compensated", encoding="utf-8")

    def fail_metadata_persistence(
        self: UploadedFileStore,
        file_record: UploadedFile,
    ) -> UploadedFile:
        raise RuntimeError("simulated metadata transaction failure")

    monkeypatch.setattr(
        UploadedFileStore,
        "add_already_durable",
        fail_metadata_persistence,
    )
    failed_registration = await register_channel_uploaded_files(
        workspace=workspace,
        task_id=71,
        user_id=user_id,
        files=(
            DownloadedChannelFile(
                name=failed_source.name,
                path=failed_source,
                mime_type="text/plain",
                size=failed_source.stat().st_size,
            ),
        ),
    )
    assert failed_registration == ()
    assert {
        path.relative_to(object_root)
        for path in object_root.rglob("*")
        if path.is_file()
    } == durable_files_before_failure
    with SessionLocal() as db:
        assert db.query(UploadedFile).count() == 1

    engine.dispose()
    get_unscoped_file_storage.cache_clear()


@pytest.mark.asyncio
async def test_prepare_channel_task_compensates_late_claim_before_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    allow_worker = threading.Event()
    claim = channel_runtime._ChannelTaskClaimSnapshot(
        user_id=7,
        task_id=19,
        is_new_task=True,
        lease=TaskLease(task_id=19, runner_id="runner-a", run_id="run-a"),
    )
    compensated: list[channel_runtime._ChannelTaskClaimSnapshot] = []

    def blocking_prepare(**_kwargs):  # type: ignore[no-untyped-def]
        worker_started.set()
        assert allow_worker.wait(timeout=2)
        return claim

    def compensate(snapshot: channel_runtime._ChannelTaskClaimSnapshot) -> bool:
        compensated.append(snapshot)
        return True

    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._prepare_channel_task_sync",
        blocking_prepare,
    )
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._compensate_channel_task_claim_sync",
        compensate,
    )

    preparation = asyncio.create_task(
        prepare_channel_task(
            channel_id=3,
            external_user_id="telegram-user",
            active_task_id=None,
            text="hello",
            channel_name="Telegram",
        )
    )
    await asyncio.to_thread(worker_started.wait, 2)
    preparation.cancel()
    allow_worker.set()

    with pytest.raises(asyncio.CancelledError):
        await preparation
    assert compensated == [claim]


@pytest.mark.asyncio
async def test_load_active_channel_configs_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    worker_started = threading.Event()
    allow_worker = threading.Event()

    def blocking_load(*_args, **_kwargs) -> tuple:  # type: ignore[no-untyped-def]
        worker_started.set()
        assert threading.get_ident() != event_loop_thread
        assert allow_worker.wait(timeout=30)
        return ()

    monkeypatch.setattr(
        "xagent.web.services.channel_runtime._load_active_channel_configs_sync",
        blocking_load,
    )

    loading = asyncio.create_task(
        load_active_channel_configs(
            channel_type="telegram",
            required_config_keys=("bot_token",),
        )
    )
    try:
        assert await asyncio.to_thread(worker_started.wait, 30)
        await asyncio.sleep(0)

        assert loading.done() is False
    finally:
        allow_worker.set()
        result = await asyncio.wait_for(loading, timeout=30)
    assert result == ()


@pytest.mark.asyncio
async def test_prepare_channel_task_evicts_existing_task_when_selection_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Doomed Agent")
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    first = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
        agent_id=agent_id,
    )
    assert first is not None
    assert await first.managed_lease.finalize_result(status=TaskStatus.COMPLETED)
    await first.managed_lease.close()

    # Mirror AgentStore.stage_delete_agent: the agent row goes away and the
    # bound tasks are nulled, while the Telegram selection still points at it.
    with SessionLocal() as db:
        db.query(Task).filter(Task.agent_id == agent_id).update(
            {Task.agent_id: None}, synchronize_session=False
        )
        db.query(Agent).filter(Agent.id == agent_id).delete(synchronize_session=False)
        db.commit()

    second = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=first.task_id,
        text="hello again",
        channel_name="Telegram",
        agent_id=agent_id,
    )

    assert second is not None
    assert second.is_new_task is True
    assert second.task_id != first.task_id
    assert second.requested_agent_missing is True
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == second.task_id).one()
        assert task.agent_id is None
        assert task.connector_runtime_selected_refs == []

    assert await second.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_prepare_channel_task_continues_task_when_selection_still_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Stable Agent")
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    first = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
        agent_id=agent_id,
    )
    assert first is not None
    assert await first.managed_lease.finalize_result(status=TaskStatus.COMPLETED)
    await first.managed_lease.close()

    second = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=first.task_id,
        text="hello again",
        channel_name="Telegram",
        agent_id=agent_id,
    )

    assert second is not None
    assert second.is_new_task is False
    assert second.task_id == first.task_id
    assert second.requested_agent_missing is False
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == second.task_id).one()
        assert task.agent_id == agent_id

    assert await second.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()


@pytest.mark.asyncio
async def test_numeric_allowed_users_authorizes_the_matching_sender(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """allowed_users is arbitrary JSON, so its ids may be numeric.

    External sender ids are always strings, so an untyped comparison denied
    every sender for a numeric config. Membership is compared as strings --
    which widens authorization for numeric configs -- but a non-matching
    sender must still be denied.
    """

    engine, SessionLocal, user_id, _ = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )

    with SessionLocal() as db:
        channel = UserChannel(
            user_id=user_id,
            channel_type="telegram",
            channel_name="Numeric allowlist",
            config={"allowed_users": [123, 456]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        channel_id = int(channel.id)

    snapshot = await channel_runtime.authorize_channel_sender(
        channel_id=channel_id,
        external_user_id="123",
    )
    assert snapshot.user_id == user_id

    with pytest.raises(channel_runtime.ChannelAuthorizationError):
        await channel_runtime.authorize_channel_sender(
            channel_id=channel_id,
            external_user_id="789",
        )

    engine.dispose()


@pytest.mark.asyncio
async def test_dict_allowed_users_still_authorizes_by_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pre-PR, `in` on a dict config matched keys; that must keep working.

    Hard-failing would lock out a channel configured as {"id": "label"} via
    the API, a regression the type guard must not introduce.
    """

    engine, SessionLocal, user_id, _ = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )

    with SessionLocal() as db:
        channel = UserChannel(
            user_id=user_id,
            channel_type="telegram",
            channel_name="Dict allowlist",
            config={"allowed_users": {"123": "alice"}},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        channel_id = int(channel.id)

    snapshot = await channel_runtime.authorize_channel_sender(
        channel_id=channel_id,
        external_user_id="123",
    )
    assert snapshot.user_id == user_id

    with pytest.raises(channel_runtime.ChannelAuthorizationError):
        await channel_runtime.authorize_channel_sender(
            channel_id=channel_id,
            external_user_id="alice",
        )

    engine.dispose()


@pytest.mark.asyncio
async def test_bare_string_allowed_users_is_not_matched_per_character(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """config is unconstrained JSON, so allowed_users may be a bare string.

    Iterating one yields its characters, which would authorize "1" and deny the
    intended "101". It must be treated as a single-entry allowlist.
    """

    engine, SessionLocal, user_id, _ = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )

    with SessionLocal() as db:
        channel = UserChannel(
            user_id=user_id,
            channel_type="telegram",
            channel_name="Bare string allowlist",
            config={"allowed_users": "101"},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        channel_id = int(channel.id)

    snapshot = await channel_runtime.authorize_channel_sender(
        channel_id=channel_id,
        external_user_id="101",
    )
    assert snapshot.user_id == user_id

    # A single character of the configured id must not authorize anyone.
    with pytest.raises(channel_runtime.ChannelAuthorizationError):
        await channel_runtime.authorize_channel_sender(
            channel_id=channel_id,
            external_user_id="1",
        )

    engine.dispose()


@pytest.mark.asyncio
async def test_closing_a_running_claim_would_fail_the_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    """ManagedTaskLease.close() is not status-neutral on a RUNNING task.

    prepare_channel_task commits the claim as RUNNING, and close() maps RUNNING
    to FAILED. Any caller that means "release without changing the status" must
    finalize explicitly instead. Pins the behaviour the Telegram fence relies on.
    """

    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    prepared = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
    )
    assert prepared is not None
    with SessionLocal() as db:
        assert (
            db.query(Task).filter(Task.id == prepared.task_id).one().status
            == TaskStatus.RUNNING
        )

    await prepared.managed_lease.close()

    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == prepared.task_id).one()
        assert task.status == TaskStatus.FAILED

    engine.dispose()


@pytest.mark.asyncio
async def test_deactivated_feishu_channel_stops_authorizing_senders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The is_active gate is channel-agnostic and applies to Feishu too.

    Feishu's bot turns ChannelConfigurationError into a config-error reply, so
    this is a deliberate behavior change rather than a crash.
    """

    engine, SessionLocal, user_id, _ = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )

    with SessionLocal() as db:
        feishu = UserChannel(
            user_id=user_id,
            channel_type="feishu",
            channel_name="Feishu",
            config={"allowed_users": ["open-id-1"]},
            is_active=True,
        )
        db.add(feishu)
        db.commit()
        feishu_channel_id = int(feishu.id)

    snapshot = await channel_runtime.authorize_channel_sender(
        channel_id=feishu_channel_id,
        external_user_id="open-id-1",
    )
    assert snapshot.user_id == user_id

    with SessionLocal() as db:
        db.query(UserChannel).filter(UserChannel.id == feishu_channel_id).update(
            {"is_active": False}
        )
        db.commit()

    with pytest.raises(channel_runtime.ChannelConfigurationError):
        await channel_runtime.authorize_channel_sender(
            channel_id=feishu_channel_id,
            external_user_id="open-id-1",
        )

    engine.dispose()


@pytest.mark.asyncio
async def test_prepare_channel_task_resumes_claimed_legacy_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    """A pre-migration task must be resumed, not abandoned for a fresh one.

    The legacy claim stamps ``telegram_user_id`` on the ORM object only. Sessions
    run with ``autoflush=False``, so the resume query that filters on that column
    cannot see the pending UPDATE unless the claim is flushed first.
    """

    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    # A task that predates the migration: owned by this channel, no sender stamp.
    with SessionLocal() as db:
        legacy = Task(
            user_id=user_id,
            title="Legacy conversation",
            status=TaskStatus.COMPLETED,
            channel_id=channel_id,
            telegram_user_id=None,
        )
        db.add(legacy)
        db.commit()
        legacy_task_id = int(legacy.id)

    resumed = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=legacy_task_id,
        text="hello again",
        channel_name="Telegram",
    )

    assert resumed is not None
    assert resumed.task_id == legacy_task_id
    assert resumed.is_new_task is False

    with SessionLocal() as db:
        claimed = db.query(Task).filter(Task.id == legacy_task_id).one()
        assert claimed.telegram_user_id == "telegram-user"
        assert db.query(Task).count() == 1

    assert await resumed.managed_lease.finalize_result(status=TaskStatus.COMPLETED)
    engine.dispose()


@pytest.mark.asyncio
async def test_prepare_channel_task_starts_new_task_when_binding_drifts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine, SessionLocal, user_id, channel_id = _create_channel_session_local(tmp_path)
    agent_id = _add_agent(SessionLocal, user_id=user_id, name="Drift Agent")
    monkeypatch.setattr(
        "xagent.web.services.channel_runtime.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(database_module, "get_session_local", lambda: SessionLocal)

    default_task = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=None,
        text="hello",
        channel_name="Telegram",
    )
    assert default_task is not None
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == default_task.task_id).one()
        assert task.agent_id is None
    assert await default_task.managed_lease.finalize_result(status=TaskStatus.COMPLETED)
    await default_task.managed_lease.close()

    switched = await prepare_channel_task(
        channel_id=channel_id,
        external_user_id="telegram-user",
        active_task_id=default_task.task_id,
        text="hello again",
        channel_name="Telegram",
        agent_id=agent_id,
    )

    assert switched is not None
    assert switched.is_new_task is True
    assert switched.task_id != default_task.task_id
    assert switched.requested_agent_missing is False
    with SessionLocal() as db:
        task = db.query(Task).filter(Task.id == switched.task_id).one()
        assert task.agent_id == agent_id

    assert await switched.managed_lease.finalize_result(status=TaskStatus.FAILED)
    engine.dispose()

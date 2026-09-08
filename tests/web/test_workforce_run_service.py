import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from xagent.core.execution_scope import (
    EXECUTION_SCOPE_AGENT_CONFIG_KEY,
    ExecutionScope,
    ExecutionScopeContext,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.web.api.chat import (
    AgentServiceManager,
    _build_tool_selection_spec_for_task,
    create_default_tools,
)
from xagent.web.models import Agent, Base, Task, User, Workforce, WorkforceRun
from xagent.web.models import database as database_module
from xagent.web.models.agent import AgentStatus
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import task_orchestrator as task_orchestrator_module
from xagent.web.services import workforce_runs as workforce_runs_module
from xagent.web.services.task_lease_service import acquire_task_lease
from xagent.web.services.workforce_access import WorkforcePolicy, set_workforce_policy
from xagent.web.services.workforce_runs import (
    create_preview_workforce_run,
    create_workforce_run,
)
from xagent.web.services.workforce_runtime import (
    WorkforceTaskRuntime,
    _map_task_status,
    ensure_workforce_turn_allowed,
    release_current_runner_task_lease_with_workforce_sync,
    release_task_lease_with_workforce_sync,
    resolve_workforce_task_runtime,
    sync_workforce_run_status,
)
from xagent.web.services.workforce_workers import create_workforce_worker


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    # begin_turn now runs its atomic claim on an isolated session opened via
    # the global SessionLocal (``get_session_local``) inside ``asyncio.to_thread``.
    # Point that global at this test's StaticPool engine (single shared
    # connection, check_same_thread=False) so the off-loop claim hits the same
    # in-memory DB the test reads from.
    import xagent.web.models.database as _db_module

    _prev_session_local = _db_module._SessionLocal
    _db_module._SessionLocal = session_factory
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        _db_module._SessionLocal = _prev_session_local
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture()
def single_connection_workforce_db(tmp_path, monkeypatch):
    """Real one-slot pool shared by the caller and turn orchestrator."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'workforce-turn-boundary.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.15,
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    monkeypatch.setattr(database_module, "_SessionLocal", session_local)
    db = session_local()
    try:
        yield engine, db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def reset_workforce_policy() -> None:
    set_workforce_policy(WorkforcePolicy())
    yield
    set_workforce_policy(WorkforcePolicy())


def _create_user(db: Session, username: str, *, is_admin: bool = False) -> User:
    user = User(
        username=username,
        password_hash="hash",
        is_admin=is_admin,
    )
    db.add(user)
    db.flush()
    return user


def _create_agent(
    db: Session,
    user: User,
    name: str,
    *,
    execution_mode: str = "balanced",
    status: AgentStatus = AgentStatus.PUBLISHED,
) -> Agent:
    agent = Agent(
        user_id=user.id,
        name=name,
        description=f"{name} description",
        instructions=f"{name} instructions",
        execution_mode=execution_mode,
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=status,
    )
    db.add(agent)
    db.flush()
    return agent


def _create_workforce(
    db: Session,
    user: User,
    manager: Agent,
) -> Workforce:
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="Research Team",
        description="Coordinates research tasks",
        manager_agent_id=manager.id,
        status="active",
    )
    db.add(workforce)
    db.flush()
    return workforce


def _add_worker(
    db: Session,
    user: User,
    workforce: Workforce,
    worker_agent: Agent,
) -> None:
    create_workforce_worker(
        db,
        workforce,
        user,
        source_type="existing",
        agent_id=worker_agent.id,
        alias="Research Analyst",
        assignment_instructions="Collect evidence and cite sources.",
    )


def _patch_schedule_bg(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    scheduled: dict[str, Any] = {}

    def fake_schedule_bg(**kwargs: Any) -> asyncio.Task[None]:
        scheduled.update(kwargs)

        async def noop() -> None:
            return None

        return asyncio.create_task(noop())

    monkeypatch.setattr(task_orchestrator_module, "_schedule_bg", fake_schedule_bg)
    return scheduled


def _mock_tool(name: str, category: str) -> Any:
    tool = MagicMock()
    tool.name = name
    tool.metadata = MagicMock()
    tool.metadata.category = MagicMock()
    tool.metadata.category.value = category
    return tool


def _workforce_runtime_with_worker_tools(*tool_names: str) -> WorkforceTaskRuntime:
    return WorkforceTaskRuntime(
        workforce_run_id=1,
        workforce_id=1,
        snapshot={},
        allowed_agent_ids=[idx + 1 for idx, _ in enumerate(tool_names)],
        agent_tool_overrides={},
        worker_tool_names=set(tool_names),
        manager_system_prompt=None,
        manager_agent_id=100,
    )


@pytest.mark.asyncio
async def test_create_workforce_run_forwards_the_caller_timezone(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opening turn starts inside task creation, so the zone has to ride
    the create request; there is no chat frame to carry it."""
    scheduled = _patch_schedule_bg(monkeypatch)

    user = _create_user(db_session, "tz-owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, _create_agent(db_session, user, "Analyst"))
    db_session.commit()

    result = await create_workforce_run(
        db_session,
        user,
        workforce,
        message="how many shifts do we have on tomorrow?",
        timezone="Australia/Melbourne",
    )
    await result.background_task

    assert scheduled["context"] == {"timezone": "Australia/Melbourne"}


@pytest.mark.asyncio
async def test_create_workforce_run_sends_no_context_without_a_timezone(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled = _patch_schedule_bg(monkeypatch)

    user = _create_user(db_session, "no-tz-owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, _create_agent(db_session, user, "Analyst"))
    db_session.commit()

    result = await create_workforce_run(
        db_session,
        user,
        workforce,
        message="hello",
    )
    await result.background_task

    assert scheduled["context"] is None


@pytest.mark.asyncio
async def test_create_workforce_run_treats_a_blank_timezone_as_absent(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled = _patch_schedule_bg(monkeypatch)

    user = _create_user(db_session, "blank-tz-owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, _create_agent(db_session, user, "Analyst"))
    db_session.commit()

    result = await create_workforce_run(
        db_session,
        user,
        workforce,
        message="hello",
        timezone="   ",
    )
    await result.background_task

    assert scheduled["context"] is None


@pytest.mark.asyncio
async def test_create_workforce_run_creates_task_run_and_starts_turn(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled = _patch_schedule_bg(monkeypatch)

    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager", execution_mode="think")
    worker_agent = _create_agent(db_session, user, "Analyst")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, worker_agent)
    uploaded_file = UploadedFile(
        file_id="file-1",
        user_id=user.id,
        filename="input.txt",
        storage_path="/tmp/input.txt",
        file_size=5,
    )
    db_session.add(uploaded_file)
    db_session.commit()

    result = await create_workforce_run(
        db_session,
        user,
        workforce,
        message="Coordinate a launch brief",
        selected_file_ids=["file-1"],
    )
    await result.background_task
    assert not hasattr(result.task, "_sa_instance_state")
    assert not hasattr(result.workforce_run, "_sa_instance_state")

    task = db_session.query(Task).filter(Task.id == int(result.task.id)).one()
    workforce_run = (
        db_session.query(WorkforceRun)
        .filter(WorkforceRun.id == int(result.workforce_run.id))
        .one()
    )
    db_session.refresh(uploaded_file)

    assert task.status == TaskStatus.RUNNING
    assert task.agent_id == manager.id
    assert result.task.agent_id == manager.id
    assert result.task.run_id == task.run_id
    assert result.task.state_version == task.state_version
    assert result.task.control_state == task.control_state
    assert task.execution_mode == "think"
    assert task.input == "Coordinate a launch brief"
    assert task.agent_config["workforce_id"] == workforce.id
    assert task.agent_config["workforce_run_id"] == workforce_run.id
    assert task.agent_config["selected_file_ids"] == ["file-1"]
    assert task.agent_config["workforce_snapshot"]["manager"]["agent_id"] == manager.id
    assert task.connector_runtime_selected_refs == []
    assert workforce_run.task_id == task.id
    assert workforce_run.status == "running"
    assert workforce_run.is_preview is False
    assert uploaded_file.task_id == task.id
    assert scheduled["task_id"] == task.id
    assert scheduled["payload"].transcript_message == "Coordinate a launch brief"
    assert (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == task.id, TaskChatMessage.role == "user")
        .count()
        == 1
    )


@pytest.mark.asyncio
async def test_create_preview_workforce_run_forwards_the_caller_timezone(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled = _patch_schedule_bg(monkeypatch)

    user = _create_user(db_session, "preview-tz-owner")
    manager = _create_agent(db_session, user, "Draft Manager")
    worker_agent = _create_agent(db_session, user, "Draft Analyst")
    db_session.commit()

    result = await create_preview_workforce_run(
        db_session,
        user_id=user.id,
        name="Launch Team",
        description=None,
        manager_agent_id=manager.id,
        workers=[
            {
                "agent_id": worker_agent.id,
                "alias": "Analyst",
                "assignment_instructions": "Do the work.",
            }
        ],
        message="how many shifts do we have on tomorrow?",
        timezone="Australia/Melbourne",
    )
    await result.background_task

    assert scheduled["context"] == {"timezone": "Australia/Melbourne"}


@pytest.mark.asyncio
async def test_create_preview_workforce_run_sends_no_context_without_a_timezone(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled = _patch_schedule_bg(monkeypatch)

    user = _create_user(db_session, "preview-no-tz-owner")
    manager = _create_agent(db_session, user, "Draft Manager")
    worker_agent = _create_agent(db_session, user, "Draft Analyst")
    db_session.commit()

    result = await create_preview_workforce_run(
        db_session,
        user_id=user.id,
        name="Launch Team",
        description=None,
        manager_agent_id=manager.id,
        workers=[
            {
                "agent_id": worker_agent.id,
                "alias": "Analyst",
                "assignment_instructions": "Do the work.",
            }
        ],
        message="hello",
    )
    await result.background_task

    assert scheduled["context"] is None


@pytest.mark.asyncio
async def test_create_preview_workforce_run_never_persists_a_workforce(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builder "test before save": manager + inline workers, no Workforce row."""

    scheduled = _patch_schedule_bg(monkeypatch)

    user = _create_user(db_session, "draft-owner")
    manager = _create_agent(db_session, user, "Draft Manager")
    worker_agent = _create_agent(db_session, user, "Draft Analyst")
    db_session.commit()

    result = await create_preview_workforce_run(
        db_session,
        user_id=user.id,
        name="Launch Team",
        description="Coordinates the launch",
        manager_agent_id=manager.id,
        workers=[
            {
                "agent_id": worker_agent.id,
                "alias": "Analyst",
                "assignment_instructions": "Collect evidence and cite sources.",
            }
        ],
        message="Draft a launch brief",
    )
    await result.background_task

    assert db_session.query(Workforce).count() == 0

    task = db_session.query(Task).filter(Task.id == int(result.task.id)).one()
    workforce_run = (
        db_session.query(WorkforceRun)
        .filter(WorkforceRun.id == int(result.workforce_run.id))
        .one()
    )

    assert workforce_run.workforce_id is None
    assert workforce_run.is_preview is True
    assert task.is_visible is False
    assert task.status == TaskStatus.RUNNING
    assert task.agent_id == manager.id
    assert task.agent_config["workforce_id"] is None
    assert task.agent_config["workforce_run_id"] == workforce_run.id
    snapshot = task.agent_config["workforce_snapshot"]
    assert snapshot["workforce"]["id"] is None
    assert snapshot["workforce"]["name"] == "Launch Team"
    assert snapshot["manager"]["agent_id"] == manager.id
    assert snapshot["workers"][0]["agent_id"] == worker_agent.id
    assert snapshot["workers"][0]["alias"] == "Analyst"
    assert scheduled["task_id"] == task.id

    # The manager can actually delegate: tool-override resolution works from
    # the run's own snapshot even though workforce_id is None.
    runtime = resolve_workforce_task_runtime(db_session, task)
    assert runtime is not None
    assert runtime.allowed_agent_ids == [worker_agent.id]

    # Turn-gating on a later message must not try to load a nonexistent
    # Workforce by a None id.
    ensure_workforce_turn_allowed(
        db_session,
        task_id=int(task.id),
        task_owner_user_id=int(user.id),
    )


@pytest.mark.asyncio
async def test_create_preview_workforce_run_requires_at_least_one_worker(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "no-worker-owner")
    manager = _create_agent(db_session, user, "Solo Manager")
    db_session.commit()

    with pytest.raises(HTTPException) as excinfo:
        await create_preview_workforce_run(
            db_session,
            user_id=user.id,
            name="Solo Team",
            description=None,
            manager_agent_id=manager.id,
            workers=[],
            message="Hello",
        )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_create_preview_workforce_run_rejects_unpublished_manager(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "unpublished-owner")
    manager = _create_agent(db_session, user, "Draft Manager", status=AgentStatus.DRAFT)
    worker_agent = _create_agent(db_session, user, "Draft Analyst")
    db_session.commit()

    with pytest.raises(HTTPException) as excinfo:
        await create_preview_workforce_run(
            db_session,
            user_id=user.id,
            name="Team",
            description=None,
            manager_agent_id=manager.id,
            workers=[
                {
                    "agent_id": worker_agent.id,
                    "alias": None,
                    "assignment_instructions": "Do the work.",
                }
            ],
            message="Hello",
        )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_create_preview_workforce_run_rejects_duplicate_worker_agent(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "duplicate-worker-owner")
    manager = _create_agent(db_session, user, "Draft Manager")
    worker_agent = _create_agent(db_session, user, "Draft Analyst")
    db_session.commit()

    with pytest.raises(HTTPException) as excinfo:
        await create_preview_workforce_run(
            db_session,
            user_id=user.id,
            name="Team",
            description=None,
            manager_agent_id=manager.id,
            workers=[
                {
                    "agent_id": worker_agent.id,
                    "alias": "First",
                    "assignment_instructions": "Do the work.",
                },
                {
                    "agent_id": worker_agent.id,
                    "alias": "Second",
                    "assignment_instructions": "Do it again.",
                },
            ],
            message="Hello",
        )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_create_preview_workforce_run_rejects_manager_as_worker(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "manager-also-worker-owner")
    manager = _create_agent(db_session, user, "Draft Manager")
    db_session.commit()

    with pytest.raises(HTTPException) as excinfo:
        await create_preview_workforce_run(
            db_session,
            user_id=user.id,
            name="Team",
            description=None,
            manager_agent_id=manager.id,
            workers=[
                {
                    "agent_id": manager.id,
                    "alias": None,
                    "assignment_instructions": "Do the work.",
                }
            ],
            message="Hello",
        )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_create_preview_workforce_run_excludes_disabled_workers(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker unchecked in the builder must not actually run in the preview,
    matching ``validate_workforce_for_run``'s ``enabled_workers`` filtering."""
    _patch_schedule_bg(monkeypatch)
    user = _create_user(db_session, "disabled-worker-owner")
    manager = _create_agent(db_session, user, "Draft Manager")
    active_worker = _create_agent(db_session, user, "Active Analyst")
    disabled_worker = _create_agent(db_session, user, "Disabled Analyst")
    db_session.commit()

    result = await create_preview_workforce_run(
        db_session,
        user_id=user.id,
        name="Launch Team",
        description=None,
        manager_agent_id=manager.id,
        workers=[
            {
                "agent_id": disabled_worker.id,
                "enabled": False,
                "assignment_instructions": "Should never run.",
            },
            {
                "agent_id": active_worker.id,
                "assignment_instructions": "Collect evidence and cite sources.",
            },
        ],
        message="Draft a launch brief",
    )
    await result.background_task

    task = db_session.query(Task).filter(Task.id == int(result.task.id)).one()
    snapshot = task.agent_config["workforce_snapshot"]
    snapshot_agent_ids = [worker["agent_id"] for worker in snapshot["workers"]]
    assert snapshot_agent_ids == [active_worker.id]

    runtime = resolve_workforce_task_runtime(db_session, task)
    assert runtime is not None
    assert runtime.allowed_agent_ids == [active_worker.id]


@pytest.mark.asyncio
async def test_create_preview_workforce_run_sorts_workers_by_sort_order(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker order in the snapshot/manager prompt must match sort_order, not
    request-array order, matching the persisted path's ``_sorted_workers``."""
    _patch_schedule_bg(monkeypatch)
    user = _create_user(db_session, "sort-order-owner")
    manager = _create_agent(db_session, user, "Draft Manager")
    second_worker = _create_agent(db_session, user, "Second Analyst")
    first_worker = _create_agent(db_session, user, "First Analyst")
    db_session.commit()

    result = await create_preview_workforce_run(
        db_session,
        user_id=user.id,
        name="Launch Team",
        description=None,
        manager_agent_id=manager.id,
        # Requested in reverse of the intended sort_order.
        workers=[
            {
                "agent_id": second_worker.id,
                "sort_order": 2,
                "assignment_instructions": "Runs second.",
            },
            {
                "agent_id": first_worker.id,
                "sort_order": 1,
                "assignment_instructions": "Runs first.",
            },
        ],
        message="Draft a launch brief",
    )
    await result.background_task

    task = db_session.query(Task).filter(Task.id == int(result.task.id)).one()
    snapshot = task.agent_config["workforce_snapshot"]
    snapshot_agent_ids = [worker["agent_id"] for worker in snapshot["workers"]]
    assert snapshot_agent_ids == [first_worker.id, second_worker.id]


@pytest.mark.asyncio
async def test_create_preview_workforce_run_rejects_boolean_worker_agent_id(
    db_session: Session,
) -> None:
    """bool is an int subclass in Python; a worker's agent_id must reject it."""
    user = _create_user(db_session, "boolean-worker-owner")
    manager = _create_agent(db_session, user, "Draft Manager")
    db_session.commit()

    with pytest.raises(HTTPException) as excinfo:
        await create_preview_workforce_run(
            db_session,
            user_id=user.id,
            name="Team",
            description=None,
            manager_agent_id=manager.id,
            workers=[
                {
                    "agent_id": True,
                    "alias": None,
                    "assignment_instructions": "Do the work.",
                }
            ],
            message="Hello",
        )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_create_preview_workforce_run_rejects_admin_using_anothers_private_agent(
    db_session: Session,
) -> None:
    """An admin must not be able to preview-run (i.e. actually execute) an
    agent they don't own, even though ``ensure_agent_access`` (used for
    agent *selection*) grants admins a bypass. The preview run path must
    enforce the persisted run path's strict ownership instead."""
    owner = _create_user(db_session, "private-agent-owner")
    admin = _create_user(db_session, "unrelated-admin", is_admin=True)
    manager = _create_agent(db_session, admin, "Admin Manager")
    private_worker = _create_agent(db_session, owner, "Owner's Private Analyst")
    db_session.commit()

    with pytest.raises(HTTPException) as excinfo:
        await create_preview_workforce_run(
            db_session,
            user_id=admin.id,
            name="Team",
            description=None,
            manager_agent_id=manager.id,
            workers=[
                {
                    "agent_id": private_worker.id,
                    "alias": None,
                    "assignment_instructions": "Do the work.",
                }
            ],
            message="Hello",
        )
    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_create_preview_workforce_run_honors_custom_run_scope_policy(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom ``WorkforcePolicy`` (e.g. team-shared execution) must apply
    identically to preview and saved runs. The preview path checks agent
    access through the same ``is_agent_in_workforce_run_scope`` hook as the
    persisted path -- passing ``workforce=None`` -- instead of hardcoding the
    default ownership rule, so a policy that widens (or narrows) run scope
    isn't silently bypassed for one path but not the other."""

    class TeamScopePolicy(WorkforcePolicy):
        def is_agent_in_workforce_run_scope(
            self,
            db: Session,
            user: User,
            workforce: Workforce | None,
            agent: Agent,
        ) -> bool:
            del db, user, workforce, agent
            return True

    _patch_schedule_bg(monkeypatch)
    set_workforce_policy(TeamScopePolicy())

    owner = _create_user(db_session, "team-scope-owner")
    runner = _create_user(db_session, "team-scope-runner")
    manager = _create_agent(db_session, owner, "Owner's Manager")
    worker_agent = _create_agent(db_session, owner, "Owner's Analyst")
    db_session.commit()

    # `runner` doesn't own either agent, but the installed policy grants
    # cross-user run scope -- must not 403 like the default policy would.
    result = await create_preview_workforce_run(
        db_session,
        user_id=runner.id,
        name="Team",
        description=None,
        manager_agent_id=manager.id,
        workers=[
            {
                "agent_id": worker_agent.id,
                "alias": None,
                "assignment_instructions": "Do the work.",
            }
        ],
        message="Hello",
    )
    await result.background_task


@pytest.mark.asyncio
async def test_create_preview_workforce_run_invokes_policy_run_hooks(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """before_workforce_run / after_workforce_run_created must fire for
    preview runs too (with workforce=None) -- they're the extension point a
    custom WorkforcePolicy would use for quota gating, audit logging, or
    billing side-effects, and preview runs execute real agents and consume
    real spend just like a saved run."""

    _patch_schedule_bg(monkeypatch)
    calls: list[tuple[str, Workforce | None]] = []

    class AuditingPolicy(WorkforcePolicy):
        def before_workforce_run(
            self, db: Session, user: User, workforce: Workforce | None
        ) -> None:
            del db, user
            calls.append(("before", workforce))

        def after_workforce_run_created(
            self,
            db: Session,
            user: User,
            workforce: Workforce | None,
            run: Any,
            task: Any,
        ) -> None:
            del db, user, run, task
            calls.append(("after", workforce))

    set_workforce_policy(AuditingPolicy())

    user = _create_user(db_session, "audited-owner")
    manager = _create_agent(db_session, user, "Draft Manager")
    worker_agent = _create_agent(db_session, user, "Draft Analyst")
    db_session.commit()

    result = await create_preview_workforce_run(
        db_session,
        user_id=user.id,
        name="Team",
        description=None,
        manager_agent_id=manager.id,
        workers=[
            {
                "agent_id": worker_agent.id,
                "alias": None,
                "assignment_instructions": "Do the work.",
            }
        ],
        message="Hello",
    )
    await result.background_task

    assert calls == [("before", None), ("after", None)]


@pytest.mark.asyncio
async def test_create_workforce_run_releases_connection_before_worker_transaction(
    single_connection_workforce_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, db = single_connection_workforce_db
    _patch_schedule_bg(monkeypatch)
    user = _create_user(db, "single-pool-owner")
    manager = _create_agent(db, user, "Manager")
    worker_agent = _create_agent(db, user, "Analyst")
    workforce = _create_workforce(db, user, manager)
    _add_worker(db, user, workforce, worker_agent)
    db.commit()

    checked_out: list[int] = []
    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    original = workforce_runs_module._create_claimed_workforce_run_isolated

    def observed(*args: Any, **kwargs: Any):
        checked_out.append(engine.pool.checkedout())
        entered.set()
        assert threading.get_ident() != loop_thread
        assert release.wait(timeout=30), "workforce transaction was never released"
        return original(*args, **kwargs)

    monkeypatch.setattr(
        workforce_runs_module,
        "_create_claimed_workforce_run_isolated",
        observed,
    )

    startup = asyncio.create_task(
        create_workforce_run(
            db,
            user,
            workforce,
            message="Coordinate a launch brief",
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        assert not startup.done()
        assert checked_out == [0]
    finally:
        release.set()
        result = await asyncio.wait_for(startup, timeout=30)
    await result.background_task

    assert result.task.status == TaskStatus.RUNNING
    assert result.workforce_run.status == "running"
    assert checked_out == [0]


@pytest.mark.asyncio
async def test_create_preview_workforce_run_releases_connection_before_worker_transaction(
    single_connection_workforce_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same pool-exhaustion guard as the persisted path (issue #889), for preview."""
    engine, db = single_connection_workforce_db
    _patch_schedule_bg(monkeypatch)
    user = _create_user(db, "single-pool-preview-owner")
    manager = _create_agent(db, user, "Draft Manager")
    worker_agent = _create_agent(db, user, "Draft Analyst")
    db.commit()

    checked_out: list[int] = []
    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    original = workforce_runs_module._create_claimed_preview_run_isolated

    def observed(*args: Any, **kwargs: Any):
        checked_out.append(engine.pool.checkedout())
        entered.set()
        assert threading.get_ident() != loop_thread
        assert release.wait(timeout=30), "workforce transaction was never released"
        return original(*args, **kwargs)

    monkeypatch.setattr(
        workforce_runs_module,
        "_create_claimed_preview_run_isolated",
        observed,
    )

    startup = asyncio.create_task(
        create_preview_workforce_run(
            db,
            user_id=user.id,
            name="Launch Team",
            description=None,
            manager_agent_id=manager.id,
            workers=[
                {
                    "agent_id": worker_agent.id,
                    "alias": None,
                    "assignment_instructions": "Collect evidence and cite sources.",
                }
            ],
            message="Draft a launch brief",
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        assert not startup.done()
        assert checked_out == [0]
    finally:
        release.set()
        result = await asyncio.wait_for(startup, timeout=30)
    await result.background_task

    assert result.task.status == TaskStatus.RUNNING
    assert result.workforce_run.status == "running"
    assert checked_out == [0]


@pytest.mark.asyncio
async def test_create_workforce_run_propagates_execution_scope_to_worker(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_schedule_bg(monkeypatch)
    user = _create_user(db_session, "scoped-owner")
    manager = _create_agent(db_session, user, "Manager")
    worker_agent = _create_agent(db_session, user, "Analyst")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, worker_agent)
    db_session.commit()
    scope = ExecutionScope(
        sandbox_key_suffix="tenant-a",
        workspace_segments=("team", "tenant-a"),
    )

    with ExecutionScopeContext(scope):
        result = await create_workforce_run(
            db_session,
            user,
            workforce,
            message="Run in the caller scope",
        )
    await result.background_task

    task = db_session.query(Task).filter(Task.id == int(result.task.id)).one()
    assert task.agent_config[EXECUTION_SCOPE_AGENT_CONFIG_KEY] == scope.to_dict()


@pytest.mark.asyncio
async def test_create_workforce_run_drains_schedule_after_caller_cancellation(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule_entered = asyncio.Event()
    allow_schedule = asyncio.Event()
    schedule_calls = 0

    async def delayed_schedule(**_kwargs: Any) -> SimpleNamespace:
        nonlocal schedule_calls
        schedule_calls += 1
        schedule_entered.set()
        await allow_schedule.wait()

        async def noop() -> None:
            return None

        return SimpleNamespace(background_task=asyncio.create_task(noop()))

    monkeypatch.setattr(
        task_orchestrator_module.TaskTurnOrchestrator,
        "schedule_claimed_create_turn",
        delayed_schedule,
    )
    user = _create_user(db_session, "cancel-owner")
    manager = _create_agent(db_session, user, "Manager")
    worker_agent = _create_agent(db_session, user, "Analyst")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, worker_agent)
    db_session.commit()

    caller = asyncio.create_task(
        create_workforce_run(
            db_session,
            user,
            workforce,
            message="Keep the committed turn owned",
        )
    )
    await schedule_entered.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    allow_schedule.set()
    with pytest.raises(asyncio.CancelledError):
        await caller

    task = db_session.query(Task).filter(Task.agent_id == manager.id).one()
    workforce_run = db_session.query(WorkforceRun).one()
    assert schedule_calls == 1
    assert task.status == TaskStatus.RUNNING
    assert workforce_run.status == "running"


@pytest.mark.asyncio
async def test_create_workforce_run_claim_timeout_rolls_back_created_records(
    single_connection_workforce_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine, db = single_connection_workforce_db
    user = _create_user(db, "pool-timeout-owner")
    manager = _create_agent(db, user, "Manager")
    worker_agent = _create_agent(db, user, "Analyst")
    workforce = _create_workforce(db, user, manager)
    _add_worker(db, user, workforce, worker_agent)
    uploaded_file = UploadedFile(
        file_id="claim-timeout-file",
        user_id=user.id,
        filename="claim-timeout.txt",
        storage_path="/tmp/claim-timeout.txt",
        file_size=5,
    )
    db.add(uploaded_file)
    db.commit()

    synthetic_timeout = SQLAlchemyTimeoutError("synthetic turn-claim timeout")
    invalidate_task_cache = MagicMock()

    monkeypatch.setattr(
        task_orchestrator_module,
        "_persist_claimed_turn_no_commit",
        MagicMock(side_effect=synthetic_timeout),
    )
    monkeypatch.setattr(
        task_orchestrator_module,
        "invalidate_task_cache",
        invalidate_task_cache,
    )

    with pytest.raises(SQLAlchemyTimeoutError) as exc_info:
        await create_workforce_run(
            db,
            user,
            workforce,
            message="Coordinate a launch brief",
            selected_file_ids=["claim-timeout-file"],
        )

    assert exc_info.value is synthetic_timeout
    db.rollback()
    assert db.query(Task).count() == 0
    assert db.query(WorkforceRun).count() == 0
    assert db.query(TaskChatMessage).count() == 0
    db.refresh(uploaded_file)
    assert uploaded_file.task_id is None
    invalidate_task_cache.assert_not_called()


@pytest.mark.asyncio
async def test_create_workforce_run_allows_draft_only_for_preview(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_schedule_bg(monkeypatch)

    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    worker_agent = _create_agent(db_session, user, "Analyst")
    workforce = _create_workforce(db_session, user, manager)
    workforce.status = "draft"
    _add_worker(db_session, user, workforce, worker_agent)
    db_session.commit()

    with pytest.raises(HTTPException) as run_error:
        await create_workforce_run(
            db_session,
            user,
            workforce,
            message="Run before publish",
        )
    assert run_error.value.status_code == 400
    assert run_error.value.detail == "Workforce must be active to run"

    result = await create_workforce_run(
        db_session,
        user,
        workforce,
        message="Preview before publish",
        is_preview=True,
        is_visible=False,
    )
    await result.background_task

    task = db_session.query(Task).filter(Task.id == int(result.task.id)).one()
    workforce_run = (
        db_session.query(WorkforceRun)
        .filter(WorkforceRun.id == int(result.workforce_run.id))
        .one()
    )
    assert task.is_visible is False
    assert workforce_run.status == "running"
    assert workforce_run.is_preview is True


@pytest.mark.asyncio
async def test_create_workforce_run_revalidates_after_lifecycle_fence(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_schedule_bg(monkeypatch)

    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    worker_agent = _create_agent(db_session, user, "Analyst")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, worker_agent)
    db_session.commit()

    fence_calls: list[int] = []

    def fake_fence(db: Session, workforce_id: int) -> Workforce:
        assert db is not db_session
        fence_calls.append(workforce_id)
        fenced = db.get(Workforce, workforce_id)
        assert fenced is not None
        fenced.status = "archived"
        return fenced

    monkeypatch.setattr(
        workforce_runs_module,
        "acquire_workforce_lifecycle_fence",
        fake_fence,
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_workforce_run(
            db_session,
            user,
            workforce,
            message="Run after archive wins",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Archived workforce cannot run"
    assert fence_calls == [int(workforce.id)]
    assert db_session.query(Task).count() == 0
    assert db_session.query(WorkforceRun).count() == 0


@pytest.mark.asyncio
async def test_create_workforce_run_marks_task_failed_when_turn_start_fails_after_claim(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_schedule_bg(**kwargs: Any) -> asyncio.Task[None]:
        del kwargs
        raise RuntimeError("schedule failed")

    monkeypatch.setattr(task_orchestrator_module, "_schedule_bg", fail_schedule_bg)

    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    worker_agent = _create_agent(db_session, user, "Analyst")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, worker_agent)
    db_session.commit()

    with pytest.raises(RuntimeError, match="schedule failed"):
        await create_workforce_run(
            db_session,
            user,
            workforce,
            message="Coordinate a launch brief",
        )

    task = db_session.query(Task).filter(Task.agent_id == manager.id).one()
    workforce_run = db_session.query(WorkforceRun).one()

    assert task.status == TaskStatus.FAILED
    assert task.error_message == "turn scheduling failed after claim commit"
    assert task.output is None
    assert workforce_run.task_id == task.id
    assert workforce_run.status == "failed"
    assert workforce_run.completed_at is not None
    assert (
        db_session.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == task.id, TaskChatMessage.role == "user")
        .count()
        == 1
    )


def test_resolve_workforce_task_runtime_requires_verified_run(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    worker_agent = _create_agent(db_session, user, "Analyst")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, worker_agent)

    task = Task(
        user_id=user.id,
        title="Workforce task",
        description="Run workforce",
        status=TaskStatus.PENDING,
        agent_id=manager.id,
        execution_mode="balanced",
    )
    db_session.add(task)
    db_session.flush()

    from xagent.web.services.workforce_snapshot import (
        build_workforce_snapshot,
        build_workforce_task_config,
    )

    snapshot = build_workforce_snapshot(db_session, user, workforce)
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="pending",
        snapshot=snapshot,
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = build_workforce_task_config(
        snapshot,
        workforce_run_id=run.id,
    )
    db_session.commit()

    runtime = resolve_workforce_task_runtime(db_session, task)

    assert runtime is not None
    assert runtime.workforce_run_id == run.id
    assert runtime.allowed_agent_ids == [worker_agent.id]
    assert runtime.enable_global_agent_tools is False
    assert runtime.allow_cross_user_agent_ids is True
    assert runtime.agent_call_stack == [manager.id]
    assert runtime.manager_system_prompt
    assert runtime.agent_tool_overrides[worker_agent.id]["workforce_run_id"] == run.id

    forged_task = Task(
        user_id=user.id,
        title="Forged task",
        description="Forged",
        status=TaskStatus.PENDING,
        agent_id=manager.id,
        agent_config=task.agent_config,
    )
    db_session.add(forged_task)
    db_session.commit()

    assert resolve_workforce_task_runtime(db_session, forged_task) is None


def test_sync_workforce_run_status_tracks_task_lifecycle(db_session: Session) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager)
    task = Task(
        user_id=user.id,
        title="Workforce task",
        description="Run workforce",
        status=TaskStatus.PENDING,
        agent_id=manager.id,
        agent_config={},
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="pending",
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": run.id}
    db_session.commit()

    assert sync_workforce_run_status(db_session, task, TaskStatus.RUNNING) is True
    db_session.commit()
    db_session.refresh(run)
    assert run.status == "running"
    assert run.completed_at is None

    assert _map_task_status(TaskStatus.PAUSED) == "paused"
    assert _map_task_status(TaskStatus.WAITING_FOR_USER) == "paused"
    assert _map_task_status("waiting_for_user") == "paused"

    assert (
        sync_workforce_run_status(db_session, task, TaskStatus.WAITING_FOR_USER) is True
    )
    db_session.commit()
    db_session.refresh(run)
    assert run.status == "paused"
    assert run.completed_at is None

    assert sync_workforce_run_status(db_session, task, TaskStatus.COMPLETED) is True
    db_session.commit()
    db_session.refresh(run)
    assert run.status == "completed"
    assert run.completed_at is not None


def test_sync_workforce_run_status_bumps_last_activity_at_on_every_real_transition(
    db_session: Session,
) -> None:
    """PR #1060 review round 8, F-NEW-1: the preview-run reaper keys
    staleness off last_activity_at, not created_at, specifically because
    created_at stays fixed for a run's whole lifetime while a multi-turn
    conversation keeps transitioning status. This pins the mechanism the
    reaper trusts: each real transition (a status/completed_at change, not a
    no-op resync) must advance last_activity_at, simulating the
    completed -> running -> completed shape of successive turns."""
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager)
    task = Task(
        user_id=user.id,
        title="Workforce task",
        description="Run workforce",
        status=TaskStatus.PENDING,
        agent_id=manager.id,
        agent_config={},
    )
    db_session.add(task)
    db_session.flush()
    # Naive, matching what SQLite round-trips a DateTime(timezone=True)
    # column as -- comparisons below stay in this same naive space
    # throughout, rather than mixing it with datetime.now(timezone.utc).
    old_activity = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=3)
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="pending",
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.flush()
    run.last_activity_at = old_activity
    task.agent_config = {"workforce_run_id": run.id}
    db_session.commit()

    # Turn 1 starts.
    assert sync_workforce_run_status(db_session, task, TaskStatus.RUNNING) is True
    db_session.commit()
    db_session.refresh(run)
    turn_1_start_activity = run.last_activity_at
    assert turn_1_start_activity is not None
    assert turn_1_start_activity > old_activity

    # Turn 1 completes.
    assert sync_workforce_run_status(db_session, task, TaskStatus.COMPLETED) is True
    db_session.commit()
    db_session.refresh(run)
    assert run.last_activity_at >= turn_1_start_activity

    # A second sync call with the SAME status is a no-op (already correct):
    # must not manufacture activity that didn't happen.
    turn_1_complete_activity = run.last_activity_at
    assert sync_workforce_run_status(db_session, task, TaskStatus.COMPLETED) is False
    db_session.commit()
    db_session.refresh(run)
    assert run.last_activity_at == turn_1_complete_activity

    # Turn 2 starts -- the exact transition the reaper needs to see as fresh
    # activity even though the run's created_at (unset here, but fixed at
    # row-creation time in production) never changes.
    assert sync_workforce_run_status(db_session, task, TaskStatus.RUNNING) is True
    db_session.commit()
    db_session.refresh(run)
    assert run.last_activity_at >= turn_1_complete_activity


def test_release_task_lease_with_workforce_sync_marks_run_failed(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager)
    task = Task(
        user_id=user.id,
        title="Workforce task",
        description="Run workforce",
        status=TaskStatus.PENDING,
        agent_id=manager.id,
        execution_mode="balanced",
        agent_config={},
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="running",
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": run.id}
    db_session.commit()

    lease = acquire_task_lease(db_session, int(task.id))
    assert lease is not None

    assert (
        release_task_lease_with_workforce_sync(
            db_session,
            lease,
            status=TaskStatus.FAILED,
        )
        is True
    )
    db_session.refresh(task)
    db_session.refresh(run)

    assert task.status == TaskStatus.FAILED
    assert run.status == "failed"
    assert run.completed_at is not None


def test_release_current_runner_task_lease_with_workforce_sync_pauses_run(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager)
    task = Task(
        user_id=user.id,
        title="Workforce task",
        description="Run workforce",
        status=TaskStatus.PENDING,
        agent_id=manager.id,
        execution_mode="balanced",
        agent_config={},
    )
    db_session.add(task)
    db_session.flush()
    run = WorkforceRun(
        workforce_id=workforce.id,
        task_id=task.id,
        user_id=user.id,
        status="running",
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": run.id}
    db_session.commit()

    lease = acquire_task_lease(db_session, int(task.id))
    assert lease is not None

    assert (
        release_current_runner_task_lease_with_workforce_sync(
            db_session,
            int(task.id),
            status=TaskStatus.WAITING_FOR_USER,
        )
        is True
    )
    db_session.refresh(task)
    db_session.refresh(run)

    assert task.status == TaskStatus.WAITING_FOR_USER
    assert run.status == "paused"
    assert run.completed_at is None


@pytest.mark.asyncio
async def test_create_workforce_run_revalidates_policy_visible_agents(
    db_session: Session,
) -> None:
    class RunOnlyPolicy(WorkforcePolicy):
        def can_run_workforce(
            self, db: Session, user: User, workforce: Workforce
        ) -> bool:
            del db, user, workforce
            return True

        def get_visible_agent_ids(
            self, db: Session, user: User, purpose: str
        ) -> set[int]:
            del db, user, purpose
            return set()

    owner = _create_user(db_session, "owner")
    runner = _create_user(db_session, "runner")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, worker_agent)
    db_session.commit()

    set_workforce_policy(RunOnlyPolicy())

    with pytest.raises(HTTPException) as run_error:
        await create_workforce_run(
            db_session,
            runner,
            workforce,
            message="Run with no visible agents",
        )

    assert run_error.value.status_code == 403
    assert run_error.value.detail == "Access denied to agent"


@pytest.mark.asyncio
async def test_create_workforce_run_rejects_policy_visible_agents_outside_run_scope(
    db_session: Session,
) -> None:
    class VisibleRunPolicy(WorkforcePolicy):
        def __init__(self, visible_ids: set[int]):
            self.visible_ids = visible_ids

        def can_run_workforce(
            self, db: Session, user: User, workforce: Workforce
        ) -> bool:
            del db, user, workforce
            return True

        def get_visible_agent_ids(
            self, db: Session, user: User, purpose: str
        ) -> set[int]:
            del db, user, purpose
            return self.visible_ids

    owner = _create_user(db_session, "owner")
    runner = _create_user(db_session, "runner")
    manager = _create_agent(db_session, owner, "Manager")
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, worker_agent)
    db_session.commit()

    set_workforce_policy(VisibleRunPolicy({manager.id, worker_agent.id}))

    with pytest.raises(HTTPException) as run_error:
        await create_workforce_run(
            db_session,
            runner,
            workforce,
            message="Run with visible agents outside run scope",
        )

    assert run_error.value.status_code == 403
    assert run_error.value.detail == "Access denied to agent"


@pytest.mark.asyncio
async def test_verified_workforce_run_scope_loads_manager_config(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TeamScopePolicy(WorkforcePolicy):
        def can_run_workforce(
            self, db: Session, user: User, workforce: Workforce
        ) -> bool:
            del db, user, workforce
            return True

        def is_agent_in_workforce_run_scope(
            self,
            db: Session,
            user: User,
            workforce: Workforce | None,
            agent: Agent,
        ) -> bool:
            del db, user, workforce, agent
            return True

    _patch_schedule_bg(monkeypatch)
    owner = _create_user(db_session, "owner")
    runner = _create_user(db_session, "runner")
    manager = _create_agent(
        db_session,
        owner,
        "Manager",
        execution_mode="think",
    )
    manager.instructions = "Use the workforce manager instructions."
    manager.tool_categories = ["browser"]
    manager.knowledge_bases = ["kb-1"]
    manager.skills = ["skill-1"]
    manager.models = {}
    worker_agent = _create_agent(db_session, owner, "Analyst")
    workforce = _create_workforce(db_session, owner, manager)
    _add_worker(db_session, owner, workforce, worker_agent)
    db_session.commit()

    set_workforce_policy(TeamScopePolicy())

    result = await create_workforce_run(
        db_session,
        runner,
        workforce,
        message="Run with team scope",
        execution_mode="balanced",
    )
    await result.background_task
    task = db_session.query(Task).filter(Task.id == int(result.task.id)).one()

    default_llm = MagicMock()
    default_llm.model_name = "default-model"
    with patch("xagent.web.api.chat.create_default_llm", return_value=default_llm):
        runtime_config = AgentServiceManager()._resolve_task_runtime_config(
            task_id=int(result.task.id),
            task=task,
            db=db_session,
            user=runner,
        )

    assert runtime_config["agent_config"]["instructions"] == manager.instructions
    assert runtime_config["agent_config"]["tool_categories"] == ["browser"]
    assert runtime_config["agent_config"]["knowledge_bases"] == ["kb-1"]
    assert runtime_config["agent_config"]["skills"] == ["skill-1"]
    assert runtime_config["task_pattern"] == "react"


@pytest.mark.asyncio
async def test_create_default_tools_forwards_workforce_delegation_config(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_all_tools(config: Any) -> list[Any]:
        captured["allowed_agent_ids"] = config.get_allowed_agent_ids()
        captured["agent_tool_overrides"] = config.get_agent_tool_overrides()
        captured["enable_global_agent_tools"] = config.get_enable_global_agent_tools()
        captured["allow_cross_user_agent_ids"] = config.get_allow_cross_user_agent_ids()
        captured["parent_task_id"] = config.get_parent_task_id()
        captured["agent_call_stack"] = config.get_agent_call_stack()
        return []

    monkeypatch.setattr(
        ToolFactory,
        "create_all_tools",
        staticmethod(fake_create_all_tools),
    )

    user = _create_user(db_session, "owner")
    overrides = {42: {"tool_name": "agent_42"}}

    await create_default_tools(
        db_session,
        user=user,
        task_id="web_task_123",
        allowed_agent_ids=[42],
        agent_tool_overrides=overrides,
        enable_global_agent_tools=False,
        allow_cross_user_agent_ids=True,
        parent_task_id="123",
        agent_call_stack=[7],
    )

    assert captured == {
        "allowed_agent_ids": [42],
        "agent_tool_overrides": overrides,
        "enable_global_agent_tools": False,
        "allow_cross_user_agent_ids": True,
        "parent_task_id": "123",
        "agent_call_stack": [7],
    }


def test_workforce_manager_without_tool_categories_gets_only_worker_tools() -> None:
    spec = _build_tool_selection_spec_for_task(
        {"tool_categories": []},
        _workforce_runtime_with_worker_tools("agent_1", "agent_2"),
        task_id=123,
    )

    assert spec.is_by_categories()
    assert spec.categories == frozenset()
    assert spec.compute_allowed_names(
        [
            _mock_tool("exa_web_search", "basic"),
            _mock_tool("write_file", "file"),
            _mock_tool("agent_1", "agent"),
            _mock_tool("agent_2", "agent"),
            _mock_tool("agent_99", "agent"),
        ]
    ) == frozenset({"agent_1", "agent_2"})


def test_workforce_manager_with_tool_categories_keeps_categories_and_workers() -> None:
    spec = _build_tool_selection_spec_for_task(
        {"tool_categories": ["browser"]},
        _workforce_runtime_with_worker_tools("agent_1"),
        task_id=123,
    )

    assert spec.is_by_categories()
    assert spec.categories == frozenset({"browser"})
    assert spec.compute_allowed_names(
        [
            _mock_tool("exa_web_search", "basic"),
            _mock_tool("browser_use", "browser"),
            _mock_tool("agent_1", "agent"),
            _mock_tool("agent_99", "agent"),
        ]
    ) == frozenset({"browser_use", "agent_1"})


# ===== Generated-manager agent run-access asymmetry (#1060 round-4 review) =====


def _create_generated_manager_agent(db: Session, user: User, name: str) -> Agent:
    """A ``create_workforce_from_prompt``-style auto-generated manager agent."""
    from xagent.web.models.agent import AgentOrigin

    agent = Agent(
        user_id=user.id,
        name=name,
        description=f"{name} description",
        instructions=f"{name} instructions",
        execution_mode="think",
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=AgentStatus.PUBLISHED,
        origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )
    db.add(agent)
    db.flush()
    return agent


def test_run_allows_a_workforces_own_generated_manager(db_session: Session) -> None:
    """The AI-prompt creation flow's manager legitimately IS generated -- must still run."""
    from xagent.web.services.workforce_snapshot import validate_workforce_for_run

    user = _create_user(db_session, "prompt-owner")
    manager = _create_generated_manager_agent(db_session, user, "Auto Manager")
    worker = _create_agent(db_session, user, "Worker")
    workforce = _create_workforce(db_session, user, manager)
    _add_worker(db_session, user, workforce, worker)

    manager_agent, enabled_workers = validate_workforce_for_run(
        db_session, user, workforce
    )

    assert manager_agent.id == manager.id
    assert [w.agent_id for w in enabled_workers] == [worker.id]


def test_run_rejects_a_foreign_generated_manager_agent_used_as_worker(
    db_session: Session,
) -> None:
    """Defence-in-depth: a generated manager agent smuggled in as a worker
    (unreachable via create_workforce_worker's own check today, but the run
    path should not silently trust it either) must not be allowed to run.
    """
    from xagent.web.models.workforce import WorkforceAgent
    from xagent.web.services.workforce_snapshot import validate_workforce_for_run

    user = _create_user(db_session, "victim-owner")
    manager = _create_agent(db_session, user, "Real Manager")
    workforce = _create_workforce(db_session, user, manager)

    foreign_generated_manager = _create_generated_manager_agent(
        db_session, user, "Someone Else's Auto Manager"
    )
    # Bypasses create_workforce_worker's ensure_agent_access gate on purpose,
    # to simulate the row existing despite that (the scenario the run-time
    # check now also guards against).
    db_session.add(
        WorkforceAgent(
            workforce_id=workforce.id,
            agent_id=foreign_generated_manager.id,
            alias="Smuggled Worker",
            assignment_instructions="Should never run",
            enabled=True,
            sort_order=1,
        )
    )
    db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        validate_workforce_for_run(db_session, user, workforce)
    assert exc_info.value.status_code == 404

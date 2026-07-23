import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.web.api.chat import (
    AgentServiceManager,
    _build_tool_selection_spec_for_task,
    create_default_tools,
)
from xagent.web.models import Agent, Base, Task, User, Workforce, WorkforceRun
from xagent.web.models.agent import AgentStatus
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import task_orchestrator as task_orchestrator_module
from xagent.web.services import workforce_runs as workforce_runs_module
from xagent.web.services.task_lease_service import acquire_task_lease
from xagent.web.services.workforce_access import WorkforcePolicy, set_workforce_policy
from xagent.web.services.workforce_runs import create_workforce_run
from xagent.web.services.workforce_runtime import (
    WorkforceTaskRuntime,
    _map_task_status,
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

    task = result.task
    workforce_run = result.workforce_run
    db_session.refresh(task)
    db_session.refresh(workforce_run)
    db_session.refresh(uploaded_file)

    assert task.status == TaskStatus.RUNNING
    assert task.agent_id == manager.id
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

    assert result.task.is_visible is False
    assert result.workforce_run.status == "running"
    assert result.workforce_run.is_preview is True


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
        assert db is db_session
        fence_calls.append(workforce_id)
        workforce.status = "archived"
        return workforce

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
    assert task.error_message == "Workforce run failed to start"
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
            workforce: Workforce,
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
    db_session.refresh(result.task)

    default_llm = MagicMock()
    default_llm.model_name = "default-model"
    with patch("xagent.web.api.chat.create_default_llm", return_value=default_llm):
        runtime_config = AgentServiceManager()._resolve_task_runtime_config(
            task_id=int(result.task.id),
            task=result.task,
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

"""Integration test for the off-loop snapshot path: ``get_agent_for_task``
consumes ``TaskSetupSnapshot`` produced on a worker thread.

What this test pins:

    1. ``get_agent_for_task`` calls
       ``load_task_setup_snapshot_sync`` via ``asyncio.to_thread`` --
       i.e. the loader runs on a separate thread, not on the loop.
       The test asserts the loader's ``threading.get_ident()`` differs
       from the loop thread's.

    2. While the loader is sleeping, the event loop is still able to
       drive other coroutines forward. We verify by kicking off a
       concurrent ``asyncio.sleep`` task and confirming it advances
       during the snapshot load window. This is the core invariant
       the off-loop snapshot loader exists to provide -- main-loop
       release during the synchronous DB block (issue #427).

    3. The snapshot's fields are observed by ``get_agent_for_task``
       on the loop thread without lazy-loading from the loader's
       session (which has already closed). Equivalent to the no-leak
       contract enforced unit-side in ``test_task_setup_snapshot``,
       restated here at the integration boundary.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.tools.adapters.vibe.config import MCPFailurePolicy
from xagent.core.workspace import TaskWorkspace
from xagent.web.api import chat as chat_module
from xagent.web.api.chat import AgentServiceManager
from xagent.web.models import Base, Task
from xagent.web.models import database as database_module
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import uploaded_file_store as uploaded_file_store_module
from xagent.web.services.managed_file_ref import ensure_uploaded_file_local_path
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)


def _make_user() -> User:
    return User(id=1, username="snap-int-user", password_hash="hash", is_admin=False)


def _build_snapshot(*, source: str | None = "internal") -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.PENDING,
            source=source,
            agent_id=None,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
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


@pytest.mark.asyncio
async def test_live_request_session_releases_clean_read_before_snapshot_worker(
    tmp_path,
    monkeypatch,
) -> None:
    """A legacy request Session must not pin the worker's only pool slot."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'agent-setup-boundary.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.15,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(database_module, "_SessionLocal", session_factory)

    with session_factory() as setup_db:
        user = User(
            username="legacy-pool-user",
            password_hash="hash",
            is_admin=False,
        )
        setup_db.add(user)
        setup_db.flush()
        task = Task(
            user_id=int(user.id),
            title="legacy pool task",
            description="test",
            status=TaskStatus.PENDING,
            execution_mode="flash",
        )
        setup_db.add(task)
        setup_db.commit()
        user_id = int(user.id)
        task_id = int(task.id)

    request_db = session_factory()
    request_user = request_db.get(User, user_id)
    assert request_user is not None
    loaded_snapshots: list[TaskSetupSnapshot] = []
    real_loader = chat_module.load_task_setup_snapshot_sync

    def tracking_loader(
        loaded_task_id: int,
        owner_user_id: int | None,
    ) -> TaskSetupSnapshot | None:
        snapshot = real_loader(loaded_task_id, owner_user_id)
        if snapshot is not None:
            loaded_snapshots.append(snapshot)
        return snapshot

    runtime_users: list[RuntimeUserFields] = []

    async def observed_create_default_tools(*_args: Any, **kwargs: Any):
        runtime_user = kwargs["user"]
        assert isinstance(runtime_user, RuntimeUserFields)
        runtime_users.append(runtime_user)
        assert engine.pool.checkedout() == 0
        await asyncio.sleep(0.03)
        assert engine.pool.checkedout() == 0
        return [], MagicMock()

    manager = AgentServiceManager()
    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            side_effect=tracking_loader,
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch("xagent.web.api.chat.create_task_tracer", return_value=MagicMock()),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=observed_create_default_tools,
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch("xagent.web.api.chat.AgentService", return_value=MagicMock()),
    ):
        await manager.get_agent_for_task(task_id, request_db, user=request_user)

    assert len(loaded_snapshots) == 1
    assert loaded_snapshots[0].task.id == task_id
    assert runtime_users == [RuntimeUserFields(id=user_id, is_admin=False)]
    assert engine.pool.checkedout() == 0
    request_db.close()
    engine.dispose()


def test_selected_file_registration_materializes_after_read_session_closes(
    tmp_path,
    monkeypatch,
    mock_workspace_db,
) -> None:
    del mock_workspace_db
    engine = create_engine(
        f"sqlite:///{tmp_path / 'selected-file-boundary.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.15,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(database_module, "_SessionLocal", session_factory)
    monkeypatch.setattr(
        uploaded_file_store_module,
        "get_session_local",
        lambda: session_factory,
    )

    object_root = tmp_path / "objects"
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()
    source_dir = tmp_path / "uploads"
    source_dir.mkdir()
    source_path = source_dir / "selected.txt"
    source_path.write_text("selected input", encoding="utf-8")

    with session_factory() as setup_db:
        user = User(
            username="selected-file-boundary-user",
            password_hash="hash",
            is_admin=False,
        )
        setup_db.add(user)
        setup_db.flush()
        task = Task(
            user_id=int(user.id),
            title="selected file task",
            description="test",
            status=TaskStatus.PENDING,
            execution_mode="flash",
        )
        setup_db.add(task)
        setup_db.flush()
        selected = UploadedFile(
            file_id="selected-file-id",
            user_id=int(user.id),
            task_id=None,
            filename=source_path.name,
            storage_path=str(source_path),
            storage_status="available",
            mime_type="text/plain",
            file_size=source_path.stat().st_size,
        )
        setup_db.add(selected)
        setup_db.commit()
        user_id = int(user.id)
        task_id = int(task.id)

    observed_checked_out: list[int] = []

    def observed_materialization(record):
        observed_checked_out.append(engine.pool.checkedout())
        return ensure_uploaded_file_local_path(record)

    monkeypatch.setattr(
        chat_module,
        "ensure_uploaded_file_local_path",
        observed_materialization,
    )
    workspace = TaskWorkspace(
        id=f"web_task_{task_id}",
        base_dir=str(tmp_path / "workspaces"),
        allowed_external_dirs=[str(source_dir)],
    )

    chat_module._register_selected_task_files_isolated(
        workspace,
        task_id=task_id,
        task_owner_id=user_id,
        selected_file_ids=["selected-file-id"],
    )

    assert observed_checked_out == [0]
    assert engine.pool.checkedout() == 0
    with session_factory() as verify_db:
        record = (
            verify_db.query(UploadedFile)
            .filter(UploadedFile.file_id == "selected-file-id")
            .one()
        )
        assert record.task_id == task_id
        assert str(record.storage_key).startswith(
            f"users/{user_id}/tasks/{task_id}/outputs/selected-file-id/_versions/"
        )

    get_unscoped_file_storage.cache_clear()
    engine.dispose()


@pytest.mark.asyncio
async def test_snapshot_worker_rejects_caller_session_with_pending_writes() -> None:
    """The boundary must not roll back writes or start a nested checkout."""

    caller_db = MagicMock(spec=Session)
    worker = AsyncMock()
    with (
        patch(
            "xagent.web.api.chat.release_db_connection_if_clean",
            return_value=False,
        ),
        patch(
            "xagent.web.api.chat.run_db_io_cancellation_safe",
            new=worker,
        ),
        pytest.raises(RuntimeError, match="pending writes"),
    ):
        await chat_module._load_task_setup_snapshot_for_agent(42, 1, caller_db)

    worker.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("task_status", [TaskStatus.PENDING, TaskStatus.RUNNING])
async def test_snapshot_pool_timeout_is_not_replaced_by_default_runtime(
    tmp_path,
    monkeypatch,
    task_status: TaskStatus,
) -> None:
    """Pool exhaustion must propagate instead of building a fallback agent."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'agent-setup-timeout.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(database_module, "_SessionLocal", session_factory)
    held_connection = engine.connect()

    task_row = MagicMock(status=task_status)
    query = MagicMock()
    query.filter.return_value.first.side_effect = [(1,), task_row]
    caller_db = MagicMock()
    caller_db.query.return_value = query
    agent_constructor = MagicMock(return_value=MagicMock())

    manager = AgentServiceManager()
    with (
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch("xagent.web.api.chat.create_task_tracer", return_value=MagicMock()),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch("xagent.web.api.chat.AgentService", new=agent_constructor),
        pytest.raises(SQLAlchemyTimeoutError),
    ):
        await manager.get_agent_for_task(42, caller_db, user=_make_user())

    agent_constructor.assert_not_called()
    held_connection.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_snapshot_runs_off_loop_thread() -> None:
    """``asyncio.to_thread`` must hand the loader off to a worker
    thread -- otherwise the main loop hasn't been released and the
    off-loop optimization is a no-op. Compare the loader's thread
    id to the loop thread's."""
    loop_thread_id = threading.get_ident()
    loader_thread_id: dict[str, int] = {}

    loaded_snapshot = _build_snapshot(source="trigger")

    def fake_loader(task_id: int, user_id: int | None) -> TaskSetupSnapshot:
        loader_thread_id["id"] = threading.get_ident()
        return loaded_snapshot

    manager = AgentServiceManager()
    user = _make_user()

    db = MagicMock()
    # Pre-Step-3 existence check still uses the request db. Mock the
    # row presence so we go down the normal-creation path.
    task_row = MagicMock()
    task_row.status = TaskStatus.PENDING
    db.query.return_value.filter.return_value.first.return_value = task_row

    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            side_effect=fake_loader,
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            return_value=None,
        ),
        patch("xagent.web.api.chat.AgentService"),
    ):
        try:
            await manager.get_agent_for_task(task_id=42, db=db, user=user)
        except Exception:
            # Downstream AgentService mock may raise during workspace
            # setup; the off-loop assertion runs before that point.
            pass

    assert "id" in loader_thread_id, (
        "Loader was never invoked -- patch path or call site changed."
    )
    assert loader_thread_id["id"] != loop_thread_id, (
        f"Loader ran on the loop thread (id={loop_thread_id}). "
        "The snapshot loader exists to push the synchronous DB "
        "block off the loop via ``asyncio.to_thread``; this check "
        "fails when the ``to_thread`` wrapper is removed or the "
        "loader is being called inline."
    )
    assert loaded_snapshot.task.source == "trigger"


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_snapshot_load() -> None:
    """The other half of the off-loop contract: while the snapshot
    loader is sleeping (simulating a slow DB read), other coroutines
    on the same loop must still be able to make progress.

    We block the loader for 0.3s and concurrently schedule a tight
    polling task that records its tick count. If ``to_thread`` works,
    the polling task progresses across many ticks during the loader's
    sleep. If the loader regresses back to an inline synchronous
    call, the polling task records at most one tick (no progress
    until the blocking sleep returns).
    """
    snapshot = _build_snapshot()
    ticks: list[float] = []
    loader_done = asyncio.Event()

    def slow_loader(task_id: int, user_id: int | None) -> TaskSetupSnapshot:
        # Synchronous sleep on the worker thread. If the call is
        # actually executed inline on the loop thread, this freezes
        # the entire loop and the poll task can't tick.
        import time

        time.sleep(0.3)
        return snapshot

    async def poll() -> None:
        loop = asyncio.get_running_loop()
        start = loop.time()
        while not loader_done.is_set():
            ticks.append(loop.time() - start)
            # Short sleep to yield, but the *loop* must run to come
            # back to us. If the loader is hogging the loop this
            # await never resumes until the sleep returns.
            await asyncio.sleep(0.02)

    manager = AgentServiceManager()
    user = _make_user()
    db = MagicMock()
    task_row = MagicMock()
    task_row.status = TaskStatus.PENDING
    db.query.return_value.filter.return_value.first.return_value = task_row

    async def driver() -> None:
        with (
            patch(
                "xagent.web.api.chat.load_task_setup_snapshot_sync",
                side_effect=slow_loader,
            ),
            patch.object(manager, "_load_persisted_conversation_history"),
            patch(
                "xagent.web.api.chat.create_task_tracer",
                return_value=MagicMock(),
            ),
            patch(
                "xagent.web.api.chat.create_default_tools",
                new=AsyncMock(return_value=([], MagicMock())),
            ),
            patch(
                "xagent.web.sandbox_manager.get_sandbox_manager",
                return_value=None,
            ),
            patch("xagent.web.api.chat.AgentService"),
        ):
            try:
                await manager.get_agent_for_task(task_id=42, db=db, user=user)
            except Exception:
                pass
        loader_done.set()

    await asyncio.gather(driver(), poll())

    # With a 0.3s blocking sleep on the worker thread and 0.02s
    # polling intervals on the loop, we expect on the order of ~10
    # ticks. Use a permissive floor of >= 5 to keep the test stable
    # under busy CI while still failing loudly if the loop genuinely
    # freezes (which would yield 0-1 ticks).
    assert len(ticks) >= 5, (
        f"Loop ticked only {len(ticks)} times during the 0.3s snapshot "
        "load -- the loader appears to be running inline on the loop "
        "thread (the off-loop invariant regressed)."
    )


@pytest.mark.asyncio
async def test_loop_consumes_snapshot_after_session_close() -> None:
    """When the loader returns, the snapshot must be fully usable on
    the loop thread with the loader's session already closed. We
    simulate the post-close state by passing a snapshot whose dict
    contents were copied (not still backed by an ORM proxy), and
    confirm ``get_agent_for_task`` reaches the AgentService
    construction step using snapshot fields.

    A snapshot that secretly held an ORM ref would normally raise
    ``DetachedInstanceError`` on attribute access here -- but because
    the snapshot is a frozen dataclass holding primitives, the loop
    consumes it without further DB access. This test pins that
    expectation at the integration boundary.
    """
    snapshot = _build_snapshot()

    constructed: dict[str, Any] = {}

    class _FakeAgentService:
        def __init__(self, **kwargs: Any) -> None:
            constructed.update(kwargs)
            self.workspace = None

        def cleanup_workspace(self) -> None: ...

        def set_conversation_history(self, _messages: list[dict[str, str]]) -> None: ...

        def set_execution_context_messages(self, _messages: list[Any]) -> None: ...

        def set_recovered_skill_context(self, _context: Any) -> None: ...

    manager = AgentServiceManager()
    user = _make_user()
    db = MagicMock()
    task_row = MagicMock()
    task_row.status = TaskStatus.PENDING
    db.query.return_value.filter.return_value.first.return_value = task_row

    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            return_value=None,
        ),
        patch("xagent.web.api.chat.AgentService", new=_FakeAgentService),
    ):
        await manager.get_agent_for_task(task_id=42, db=db, user=user)

    # ``pattern`` and ``task_id`` should have flowed through from the
    # snapshot to the AgentService constructor. If they didn't, the
    # consumer was reading from a stale ORM ref and would have raised
    # before reaching this point.
    assert constructed.get("pattern") == "single_call"
    assert constructed.get("task_id") == "42"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_excluded"),
    [
        (AgentStatus.PUBLISHED, True),
        (AgentStatus.DRAFT, False),
    ],
)
async def test_get_agent_for_task_snapshot_reader_uses_persisted_task_owner(
    tmp_path,
    monkeypatch,
    status,
    expected_excluded,
) -> None:
    """The reachable get-agent path delegates preview exclusion to its snapshot."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'preview-owner.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr(database_module, "_SessionLocal", session_factory)

    with session_factory() as setup_db:
        owner = User(username="preview-owner", password_hash="hash", is_admin=False)
        actor = User(username="preview-actor", password_hash="hash", is_admin=False)
        setup_db.add_all([owner, actor])
        setup_db.flush()
        preview_agent = Agent(
            user_id=int(owner.id),
            name="preview-agent",
            status=status,
            instructions="preview instructions",
            execution_mode="flash",
            models={},
            knowledge_bases=[],
            skills=[],
            tool_categories=[],
        )
        setup_db.add(preview_agent)
        setup_db.flush()
        task = Task(
            user_id=int(owner.id),
            title="preview task",
            description="preview",
            status=TaskStatus.PENDING,
            execution_mode="flash",
            agent_config={
                "instructions": "preview instructions",
                "skills": [],
                "knowledge_bases": [],
                "tool_categories": [],
                "preview_agent_id": str(int(preview_agent.id)),
            },
        )
        setup_db.add(task)
        setup_db.commit()
        owner_id = int(owner.id)
        actor_id = int(actor.id)
        preview_agent_id = int(preview_agent.id)
        task_id = int(task.id)

    resolved_owner_ids: list[int] = []
    from xagent.web.services import agent_team_scope

    real_resolver = agent_team_scope.resolve_authorized_agent

    def observe_resolver(session, owner_user_id, candidate_id):
        resolved_owner_ids.append(owner_user_id)
        return real_resolver(session, owner_user_id, candidate_id)

    observed_excluded_ids: list[int | None] = []

    async def capture_tools(*_args: Any, **kwargs: Any):
        observed_excluded_ids.append(kwargs["excluded_agent_id"])
        return [], MagicMock()

    manager = AgentServiceManager()
    agent_service = MagicMock()
    agent_service.workspace = None
    request_db = session_factory()
    actor_user = request_db.get(User, actor_id)
    assert actor_user is not None
    monkeypatch.setattr(agent_team_scope, "resolve_authorized_agent", observe_resolver)
    try:
        with (
            patch.object(manager, "_load_persisted_conversation_history"),
            patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
            patch("xagent.web.api.chat.create_task_tracer", return_value=MagicMock()),
            patch("xagent.web.api.chat.create_default_tools", new=capture_tools),
            patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
            patch("xagent.web.api.chat.AgentService", return_value=agent_service),
        ):
            await manager.get_agent_for_task(task_id, request_db, user=actor_user)
    finally:
        request_db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()

    assert actor_id != owner_id
    assert resolved_owner_ids == [owner_id]
    assert observed_excluded_ids == [preview_agent_id if expected_excluded else None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "expected_policy"),
    [
        ("trigger", MCPFailurePolicy.STRICT),
        ("internal", MCPFailurePolicy.BEST_EFFORT),
        (None, MCPFailurePolicy.BEST_EFFORT),
    ],
)
async def test_snapshot_source_controls_mcp_failure_policy(
    source: str | None,
    expected_policy: MCPFailurePolicy,
) -> None:
    snapshot = _build_snapshot(source=source)
    manager = AgentServiceManager()
    user = _make_user()
    db = MagicMock()
    task_row = MagicMock(status=TaskStatus.PENDING)
    db.query.return_value.filter.return_value.first.return_value = task_row
    create_tools = AsyncMock(return_value=([], MagicMock()))

    tracer = MagicMock()
    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch("xagent.web.api.chat.create_task_tracer", return_value=tracer),
        patch("xagent.web.api.chat.create_default_tools", new=create_tools),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch("xagent.web.api.chat.AgentService"),
    ):
        await manager.get_agent_for_task(task_id=42, db=db, user=user)

    assert create_tools.await_args.kwargs["mcp_failure_policy"] is expected_policy
    assert create_tools.await_args.kwargs["mcp_load_summary_tracer"] is tracer
    assert create_tools.await_args.kwargs["mcp_load_summary_trace_task_id"] == "42"


@pytest.mark.asyncio
async def test_snapshot_fallback_raises_on_no_default_llm_with_agent_builder() -> None:
    """Snapshot path must share the same fail-fast failure policy as
    the reconstruct path.

    Without this guard, an agent-builder task whose models couldn't
    be resolved AND whose deployment had no global default LLM
    (``self._default_llm`` is None -- typical of CI / un-configured
    deployments) would silently get ``task_llm = None``, build the
    AgentService anyway, and crash later on the first LLM call. The
    reconstruct path raises ``HTTPException(500)`` via
    ``_pick_default_llm_with_warning``; the snapshot path must do
    the same.

    This test pins the invariant: snapshot path raises when
    snapshot.agent is set, snapshot.task_llm is None, and
    ``self._default_llm`` is None. ``saved_model_*`` diagnostic
    fields from the snapshot's ``agent_config`` flow into the log
    line via the same helper.
    """
    from fastapi import HTTPException

    from xagent.web.services.llm_utils import AgentRuntimeFields

    # Snapshot whose agent_builder ran but resolved no LLMs.
    agent_builder_snapshot = TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.PENDING,
            source="trigger",
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="balanced",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="react",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="builder-agent",
            status="published",
            instructions="be terse",
        ),
        agent_config={
            "llms": (None, None, None, None),
            "saved_model_ids": {"general": 123},
            "saved_model_descriptors": {
                "general": {"pk": 123, "model_id": "missing-model", "model_name": "X"}
            },
            "execution_mode": "balanced",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": ["basic"],
        },
        excluded_agent_id=7,
    )

    manager = AgentServiceManager()
    manager._default_llm = None  # type: ignore[assignment]

    user = _make_user()
    db = MagicMock()
    task_row = MagicMock()
    task_row.status = TaskStatus.PENDING
    db.query.return_value.filter.return_value.first.return_value = task_row

    with (
        patch(
            "xagent.web.api.chat.load_task_setup_snapshot_sync",
            return_value=agent_builder_snapshot,
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.api.chat.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.api.chat.create_default_tools",
            new=AsyncMock(return_value=([], MagicMock())),
        ),
        patch(
            "xagent.web.sandbox_manager.get_sandbox_manager",
            return_value=None,
        ),
        patch("xagent.web.api.chat.AgentService"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await manager.get_agent_for_task(task_id=42, db=db, user=user)

    assert exc_info.value.status_code == 500
    assert "Agent model configuration is unavailable" in str(exc_info.value.detail)

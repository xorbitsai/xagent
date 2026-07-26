"""Unit tests for AgentServiceManager task existence checking and reconstruction"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from xagent.core.tools.adapters.vibe.config import MCPFailurePolicy
from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.task import (
    DAGExecution,
    DAGExecutionPhase,
    Task,
    TaskStatus,
    TraceEvent,
)
from xagent.web.models.user import User
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskReconstructionSnapshot,
    TaskSetupSnapshot,
    _TaskFields,
)


def _build_reconstruction_snapshot(
    task: Task,
    user: User,
    *,
    trace_events: list[TraceEvent] | None = None,
    dag_execution: DAGExecution | None = None,
    task_llm=None,
    task_pattern: str = "dag_plan_execute",
    agent_config: dict | None = None,
) -> TaskSetupSnapshot:
    """Build the detached reconstruction contract consumed on the event loop."""
    trace_events = trace_events or []
    reconstruction = TaskReconstructionSnapshot(
        tracer_events=tuple(
            {
                "id": str(event.event_id),
                "event_type": str(event.event_type),
                "task_id": str(event.task_id),
                "step_id": str(event.step_id) if event.step_id is not None else None,
                "timestamp": event.timestamp.timestamp() if event.timestamp else None,
                "data": dict(event.data or {}),
                "parent_id": (
                    str(event.parent_event_id)
                    if event.parent_event_id is not None
                    else None
                ),
            }
            for event in trace_events
        ),
        plan_state=(
            dict(dag_execution.current_plan)
            if dag_execution is not None and dag_execution.current_plan
            else None
        ),
        has_history=bool(trace_events) or dag_execution is not None,
    )
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=int(task.id),
            user_id=int(task.user_id),
            status=task.status,
            source=task.source,
            agent_id=task.agent_id,
            agent_config=task.agent_config,
            model_name=task.model_name,
            compact_model_name=task.compact_model_name,
            execution_mode=task.execution_mode,
            agent_type=task.agent_type,
        ),
        runtime_user=RuntimeUserFields(
            id=int(user.id),
            is_admin=bool(user.is_admin),
        ),
        has_reconstructable_history=reconstruction.has_history,
        task_pattern=task_pattern,
        task_llm=task_llm,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=None,
        agent_config=agent_config,
        excluded_agent_id=None,
        reconstruction=reconstruction,
    )


class TestAgentServiceManagerReconstruction:
    """测试AgentServiceManager的任务重建功能"""

    @pytest.fixture
    def agent_manager(self):
        """创建AgentServiceManager实例"""
        return AgentServiceManager()

    @pytest.fixture
    def mock_db(self):
        """创建mock数据库会话"""
        db = MagicMock()
        db.query = MagicMock()
        return db

    @pytest.fixture
    def mock_user(self):
        """创建mock用户"""
        return User(
            id=1,
            username="test_user",
            password_hash="hashed_password",
            is_admin=False,
        )

    @pytest.fixture
    def sample_task(self):
        """创建示例任务"""
        return Task(
            id=1,
            user_id=1,
            title="Test Task",
            description="Test description",
            status=TaskStatus.PENDING,
            model_name="gpt-4",
            small_fast_model_name="gpt-3.5-turbo",
            agent_type="standard",
        )

    @pytest.fixture
    def sample_trace_events(self):
        """创建示例追踪事件"""
        return [
            TraceEvent(
                id=1,
                task_id=1,
                event_id="event1",
                event_type="task_start_general",
                timestamp=datetime.now(),
                step_id=None,
                parent_event_id=None,
                data={"goal": "Test goal"},
            ),
            TraceEvent(
                id=2,
                task_id=1,
                event_id="event2",
                event_type="step_end_dag",
                timestamp=datetime.now(),
                step_id="step1",
                parent_event_id="event1",
                data={"success": True, "result": "4"},
            ),
        ]

    @pytest.fixture
    def sample_dag_execution(self):
        """创建示例DAG执行记录"""
        return DAGExecution(
            id=1,
            task_id=1,
            phase=DAGExecutionPhase.PLANNING,
            progress_percentage=50.0,
            completed_steps=1,
            total_steps=2,
            current_plan={
                "id": "test_plan",
                "goal": "Test goal",
                "steps": [
                    {
                        "id": "step1",
                        "name": "Test Step",
                        "description": "Test description",
                        "tool_name": "calculator",
                        "tool_args": {"expression": "2+2"},
                        "dependencies": [],
                        "status": "completed",
                        "result": {"result": "4"},
                        "context": {},
                        "difficulty": "easy",
                    }
                ],
            },
        )

    @pytest.mark.asyncio
    async def test_get_agent_for_task_new_task(self, agent_manager, mock_db, mock_user):
        """A missing task is created before its detached snapshot is loaded."""
        persisted_task = Task(
            id=1,
            user_id=1,
            title="Task 1",
            description="Auto-created task",
            status=TaskStatus.PENDING,
            agent_type="standard",
        )
        post_create_snapshot = _build_reconstruction_snapshot(
            persisted_task,
            mock_user,
            task_llm=MagicMock(),
        )
        owner_query = MagicMock()
        owner_query.filter.return_value = owner_query
        owner_query.first.return_value = None
        task_query = MagicMock()
        task_query.filter.return_value = task_query
        task_query.first.return_value = None

        def query(entity):
            if entity is Task.user_id:
                return owner_query
            if entity is Task:
                return task_query
            return MagicMock()

        mock_db.query.side_effect = query
        mock_db.commit = MagicMock()
        mock_db.refresh = MagicMock()
        mock_db.add = MagicMock()

        with (
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
            patch(
                "xagent.web.api.chat.load_task_setup_snapshot_sync",
                side_effect=[None, post_create_snapshot],
            ) as snapshot_loader,
            patch(
                "xagent.web.api.chat.create_default_tools",
                new=AsyncMock(return_value=([], MagicMock())),
            ),
            patch("xagent.web.api.chat.get_memory_store", return_value=MagicMock()),
            patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        ):
            # 创建mock AgentService实例
            mock_agent_service = MagicMock()
            mock_agent_service_class.return_value = mock_agent_service

            agent = await agent_manager.get_agent_for_task(
                1,
                mock_db,
                user=mock_user,
                resolved_execution_scope=None,
            )

        assert agent is mock_agent_service
        assert 1 in agent_manager._agents
        assert snapshot_loader.call_args_list == [call(1, None), call(1, 1)]
        created_task = mock_db.add.call_args.args[0]
        assert isinstance(created_task, Task)
        assert created_task.user_id == 1
        assert created_task.status == TaskStatus.PENDING
        mock_db.commit.assert_called_once_with()
        mock_db.refresh.assert_called_once_with(created_task)

    @pytest.mark.asyncio
    async def test_get_agent_for_task_existing_task_no_reconstruction(
        self, agent_manager, mock_db, sample_task, mock_user
    ):
        """测试获取已存在任务的agent，但没有历史数据"""
        snapshot = _build_reconstruction_snapshot(
            sample_task,
            mock_user,
            task_llm=MagicMock(),
        )
        with (
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
            patch(
                "xagent.web.api.chat.create_default_tools",
                new=AsyncMock(return_value=([], MagicMock())),
            ),
            patch("xagent.web.api.chat.get_memory_store", return_value=MagicMock()),
            patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        ):
            mock_agent_service = MagicMock()
            mock_agent_service_class.return_value = mock_agent_service

            agent = await agent_manager.get_agent_for_task(
                1,
                mock_db,
                user=mock_user,
                task_setup_snapshot=snapshot,
                task_owner_user_id=1,
                resolved_execution_scope=None,
            )

        assert agent is not None
        assert 1 in agent_manager._agents
        assert agent_manager._agents[1] == mock_agent_service

    @pytest.mark.asyncio
    async def test_get_agent_for_task_cached_sandbox_without_agent_config(
        self, agent_manager, mock_db, sample_task, mock_user
    ):
        """A cached sandbox should not skip allowed_tools initialization."""
        sandbox_mgr = MagicMock()
        sandbox_mgr.get_or_create_lease_provider = AsyncMock(return_value=MagicMock())
        snapshot = _build_reconstruction_snapshot(
            sample_task,
            mock_user,
            task_llm=MagicMock(),
        )

        with (
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
            patch("xagent.web.api.chat.get_memory_store", return_value=MagicMock()),
            patch(
                "xagent.web.sandbox_manager.get_sandbox_manager",
                return_value=sandbox_mgr,
            ),
            patch(
                "xagent.core.tools.adapters.vibe.factory.ToolFactory"
            ) as mock_tool_factory,
        ):
            mock_tool_factory.create_all_tools = AsyncMock(return_value=[])
            mock_agent_service = MagicMock()
            mock_agent_service_class.return_value = mock_agent_service

            agent = await agent_manager.get_agent_for_task(
                1,
                mock_db,
                user=mock_user,
                task_setup_snapshot=snapshot,
                task_owner_user_id=1,
                resolved_execution_scope=None,
            )

        assert agent is mock_agent_service
        assert 1 in agent_manager._agents

    @pytest.mark.asyncio
    async def test_admin_task_uses_task_owner_workspace_dirs(
        self, agent_manager, mock_db, tmp_path
    ):
        """Admin executing another user's task should use the task owner's workspace."""
        owner_task = Task(
            id=1,
            user_id=7,
            title="Owner Task",
            description="Task owned by another user",
            status=TaskStatus.PENDING,
            agent_type="standard",
        )
        admin_user = User(
            id=1,
            username="admin",
            password_hash="hashed_password",
            is_admin=True,
        )
        owner_user = User(
            id=7,
            username="owner",
            password_hash="hashed_password",
            is_admin=False,
        )
        snapshot = _build_reconstruction_snapshot(
            owner_task,
            owner_user,
            task_llm=MagicMock(),
        )
        uploads_dir = tmp_path / "uploads"

        with (
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
            patch(
                "xagent.web.services.llm_utils.UserAwareModelStorage.resolve_llms_from_names"
            ) as mock_resolve_llms,
            patch("xagent.web.api.chat.get_memory_store") as mock_get_memory,
            patch("xagent.web.api.chat.get_uploads_dir", return_value=uploads_dir),
            patch(
                "xagent.core.tools.adapters.vibe.factory.ToolFactory"
            ) as mock_tool_factory,
        ):
            mock_resolve_llms.return_value = (MagicMock(), None, None, None)
            mock_get_memory.return_value = MagicMock()
            mock_tool_factory.create_all_tools = AsyncMock(return_value=[])
            mock_agent_service = MagicMock()
            mock_agent_service_class.return_value = mock_agent_service

            mock_task_query = MagicMock()
            mock_task_query.filter.return_value = mock_task_query
            mock_task_query.first.return_value = owner_task
            mock_db.query.return_value = mock_task_query

            await agent_manager.get_agent_for_task(
                1,
                mock_db,
                user=admin_user,
                task_setup_snapshot=snapshot,
                task_owner_user_id=7,
                resolved_execution_scope=None,
            )

        kwargs = mock_agent_service_class.call_args.kwargs
        assert kwargs["workspace_base_dir"] == str(uploads_dir / "user_7")
        assert str(uploads_dir / "user_7") in kwargs["allowed_external_dirs"]
        tool_config = kwargs["tool_config"]
        workspace_config = tool_config.get_workspace_config()
        assert tool_config.get_user_id() == 7
        assert workspace_config["base_dir"] == str(uploads_dir / "user_7")
        assert str(uploads_dir / "user_7") in workspace_config["allowed_external_dirs"]

    @pytest.mark.asyncio
    async def test_build_tools_maps_categories_from_full_catalog(
        self, agent_manager, mock_db, sample_task, mock_user, monkeypatch
    ):
        """``_build_tools_for_task`` constructs a ``ToolSelectionSpec``
        once via ``from_raw`` and hands it to
        ``WebToolConfig.tool_selection_spec``. The factory's
        ``spec.compute_allowed_names`` dispatch then drives the
        name-level filter -- a single registry build, not a two-pass
        ``create_all_tools`` with a manual select+merge stage.
        """
        from xagent.core.tools.adapters.vibe.selection_spec import (
            _SpecByCategories,
        )

        class _Tool:
            description = ""

            def __init__(self, name: str, category: str) -> None:
                self.name = name
                self.metadata = SimpleNamespace(
                    category=SimpleNamespace(value=category)
                )

        basic_tool = _Tool("calculator", "basic")
        browser_tool = _Tool("browser_navigate", "browser")

        async def create_all_tools(
            config,
            apply_user_override_filter: bool = True,
        ):
            return [basic_tool, browser_tool]

        monkeypatch.setattr(
            "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
            create_all_tools,
        )

        with patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None):
            _tools, tool_config = await agent_manager._build_tools_for_task(
                task_id=sample_task.id,
                task=sample_task,
                db=mock_db,
                user=mock_user,
                agent_config={
                    "tool_categories": ["browser"],
                    "knowledge_bases": [],
                    "skills": [],
                },
                task_llm=None,
                task_vision_llm=None,
            )

        spec = tool_config.get_tool_selection_spec()
        assert isinstance(spec, _SpecByCategories), (
            "BY_CATEGORIES mode expected for non-empty tool_categories"
        )
        assert "browser" in spec.categories

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("source", "expected_policy"),
        [
            ("trigger", MCPFailurePolicy.STRICT),
            ("sdk", MCPFailurePolicy.BEST_EFFORT),
            (None, MCPFailurePolicy.BEST_EFFORT),
        ],
    )
    async def test_build_tools_maps_task_source_to_mcp_failure_policy(
        self,
        source,
        expected_policy,
        agent_manager,
        mock_db,
        sample_task,
        mock_user,
        monkeypatch,
    ):
        sample_task.source = source

        async def create_all_tools(config, apply_user_override_filter=True):
            return []

        monkeypatch.setattr(
            "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
            create_all_tools,
        )

        task_tracer = object()
        with patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None):
            _tools, tool_config = await agent_manager._build_tools_for_task(
                task_id=sample_task.id,
                task=sample_task,
                db=mock_db,
                user=mock_user,
                agent_config={
                    "tool_categories": ["mcp:Gmail"],
                    "knowledge_bases": [],
                    "skills": [],
                },
                task_llm=None,
                task_vision_llm=None,
                parent_tracer=task_tracer,
            )

        assert tool_config.get_mcp_failure_policy() is expected_policy
        assert tool_config._mcp_load_summary_tracer is task_tracer
        assert tool_config._mcp_load_summary_trace_task_id == str(sample_task.id)

    @pytest.mark.asyncio
    async def test_get_agent_for_task_existing_task_with_reconstruction(
        self,
        agent_manager,
        sample_task,
        sample_trace_events,
        sample_dag_execution,
        mock_user,
    ):
        """测试获取已存在任务的agent，并进行重建"""
        sample_task.status = TaskStatus.RUNNING
        runtime_llm = MagicMock()
        snapshot = _build_reconstruction_snapshot(
            sample_task,
            mock_user,
            trace_events=sample_trace_events,
            dag_execution=sample_dag_execution,
            task_llm=runtime_llm,
        )
        # 使用更高级的方法直接patch AgentService创建
        with (
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
            patch(
                "xagent.web.api.chat.create_default_tools",
                new=AsyncMock(return_value=([], MagicMock())),
            ),
            patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        ):
            # 创建mock AgentService实例
            mock_agent_instance = MagicMock()
            mock_agent_instance.reconstruct_from_history = AsyncMock()
            mock_agent_service_class.return_value = mock_agent_instance

            # 调用方法
            agent = await agent_manager.get_agent_for_task(
                1,
                None,
                user=mock_user,
                task_setup_snapshot=snapshot,
            )

        # 验证结果
        assert agent is not None
        assert 1 in agent_manager._agents
        assert agent_manager._agents[1] == mock_agent_instance
        # 验证reconstruct_from_history被调用
        mock_agent_instance.reconstruct_from_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconstruct_agent_from_history_success(
        self,
        agent_manager,
        mock_db,
        mock_user,
        sample_task,
        sample_trace_events,
        sample_dag_execution,
    ):
        """测试从历史数据重建agent成功"""
        # Mock AgentService创建和reconstruct_from_history
        runtime_llm = MagicMock()
        runtime_llm.model_name = "task-qwen"
        snapshot = _build_reconstruction_snapshot(
            sample_task,
            mock_user,
            trace_events=sample_trace_events,
            dag_execution=sample_dag_execution,
            task_llm=runtime_llm,
        )
        with (
            patch(
                "xagent.web.api.chat.create_default_tools",
                new=AsyncMock(return_value=(["tool"], "tool_config")),
            ),
            patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
        ):
            # 设置mock AgentService实例
            mock_agent_instance = MagicMock()
            mock_agent_instance.reconstruct_from_history = AsyncMock()
            mock_agent_service_class.return_value = mock_agent_instance

            # 调用方法
            await agent_manager._reconstruct_agent_from_history(
                1,
                None,
                task_setup_snapshot=snapshot,
            )

        # 验证agent被创建
        assert 1 in agent_manager._agents
        # 验证reconstruct_from_history被调用
        mock_agent_instance.reconstruct_from_history.assert_called_once()
        _, agent_kwargs = mock_agent_service_class.call_args
        assert agent_kwargs["tools"] == ["tool"]
        assert agent_kwargs["tool_config"] == "tool_config"

    @pytest.mark.asyncio
    async def test_reconstruct_agent_from_history_uses_shared_runtime_config(
        self,
        agent_manager,
        mock_db,
        mock_user,
        sample_task,
        sample_trace_events,
        sample_dag_execution,
    ):
        """Active-task reconstruction consumes the shared runtime snapshot."""
        sample_task.agent_id = 9
        runtime_llm = MagicMock()
        runtime_llm.model_name = "task-qwen"
        agent_config = {
            "instructions": "",
            "skills": [],
            "knowledge_bases": [],
        }
        snapshot = _build_reconstruction_snapshot(
            sample_task,
            mock_user,
            trace_events=sample_trace_events,
            dag_execution=sample_dag_execution,
            task_llm=runtime_llm,
            task_pattern="react",
            agent_config=agent_config,
        )
        agent_manager._resolve_task_runtime_config = MagicMock(
            side_effect=AssertionError("reconstruction re-read runtime config")
        )

        with (
            patch(
                "xagent.web.api.chat.create_default_tools",
                new=AsyncMock(return_value=(["tool"], "tool_config")),
            ),
            patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
        ):
            mock_agent_instance = MagicMock()
            mock_agent_instance.reconstruct_from_history = AsyncMock()
            mock_agent_service_class.return_value = mock_agent_instance

            await agent_manager._reconstruct_agent_from_history(
                1,
                None,
                task_setup_snapshot=snapshot,
            )

        agent_manager._resolve_task_runtime_config.assert_not_called()
        _, agent_kwargs = mock_agent_service_class.call_args
        assert agent_kwargs["llm"] is runtime_llm
        assert agent_kwargs["pattern"] == "react"

    @pytest.mark.asyncio
    async def test_reconstruct_agent_from_history_no_data(
        self,
        agent_manager,
        sample_task,
        mock_user,
    ):
        """测试没有历史数据时的重建"""
        snapshot = _build_reconstruction_snapshot(
            sample_task,
            mock_user,
            task_llm=MagicMock(),
        )

        # 调用方法应该抛出异常
        with pytest.raises(ValueError) as exc_info:
            await agent_manager._reconstruct_agent_from_history(
                1,
                None,
                task_setup_snapshot=snapshot,
            )

        assert "No historical data found" in str(exc_info.value)
        # 验证没有创建agent
        assert 1 not in agent_manager._agents

    @pytest.mark.asyncio
    async def test_reconstruct_agent_from_history_error_handling(self, agent_manager):
        """测试重建过程中的错误处理"""
        # 调用方法应该抛出异常
        with (
            patch(
                "xagent.web.api.chat.load_task_setup_snapshot_sync",
                side_effect=Exception("Database error"),
            ),
            pytest.raises(Exception) as exc_info,
        ):
            await agent_manager._reconstruct_agent_from_history(1, None)

        assert "Database error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_agent_for_task_snapshot_db_error_propagates(
        self, agent_manager, mock_db, mock_user
    ):
        """Snapshot DB failures do not masquerade as a missing task."""
        error = RuntimeError("Database error")

        with (
            patch(
                "xagent.web.api.chat.load_task_setup_snapshot_sync",
                side_effect=error,
            ) as snapshot_loader,
            pytest.raises(RuntimeError) as exc_info,
        ):
            await agent_manager.get_agent_for_task(1, mock_db, user=mock_user)

        assert exc_info.value is error
        snapshot_loader.assert_called_once_with(1, None)
        mock_db.query.assert_not_called()
        assert 1 not in agent_manager._agents

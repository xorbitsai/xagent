"""Unit tests for AgentServiceManager task existence checking and reconstruction"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.task import (
    DAGExecution,
    DAGExecutionPhase,
    Task,
    TaskStatus,
    TraceEvent,
)
from xagent.web.models.user import User


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
        """测试获取新任务的agent"""
        # 使用更高级的方法直接patch AgentService创建
        with (
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
            patch("xagent.web.api.chat.resolve_llms_from_names") as mock_resolve_llms,
            patch("xagent.web.api.chat.get_memory_store") as mock_get_memory,
            patch("xagent.web.api.chat.Tracer"),
            patch("xagent.web.api.chat.create_default_tools") as mock_create_tools,
        ):
            # 设置所有必要的mock
            mock_resolve_llms.return_value = (MagicMock(), None, None, None)
            mock_get_memory.return_value = MagicMock()
            mock_create_tools.return_value = ([], MagicMock())  # (tools, tool_config)

            # 创建mock AgentService实例
            mock_agent_service = MagicMock()
            mock_agent_service_class.return_value = mock_agent_service

            # Mock database operations
            mock_db.commit = MagicMock()
            mock_db.refresh = MagicMock()
            mock_db.add = MagicMock()

            # Mock Task queries - 第一次返回None（任务不存在），后续返回新创建的任务
            created_task = MagicMock()
            created_task.agent_type = "standard"
            created_task.model_name = None
            created_task.small_fast_model_name = None
            created_task.visual_model_name = None
            created_task.compact_model_name = None

            mock_task_query = MagicMock()
            call_count = [0]

            def mock_first():
                call_count[0] += 1
                return None if call_count[0] == 1 else created_task

            mock_task_query.first.side_effect = mock_first
            mock_db.query.return_value = mock_task_query

            # 调用方法
            agent = await agent_manager.get_agent_for_task(1, mock_db, user=mock_user)

        # 验证结果
        assert agent is not None
        assert 1 in agent_manager._agents
        assert agent_manager._agents[1] == mock_agent_service

    @pytest.mark.asyncio
    async def test_get_agent_for_task_existing_task_no_reconstruction(
        self, agent_manager, mock_db, sample_task, mock_user
    ):
        """测试获取已存在任务的agent，但没有历史数据"""
        # 使用更高级的方法直接patch AgentService创建
        with (
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
            patch("xagent.web.api.chat.resolve_llms_from_names") as mock_resolve_llms,
            patch("xagent.web.api.chat.get_memory_store") as mock_get_memory,
            patch("xagent.web.api.chat.Tracer"),
            patch("xagent.web.api.chat.create_default_tools") as mock_create_tools,
        ):
            # 设置所有必要的mock
            mock_resolve_llms.return_value = (MagicMock(), None, None, None)
            mock_get_memory.return_value = MagicMock()
            mock_create_tools.return_value = ([], MagicMock())  # (tools, tool_config)

            # 创建mock AgentService实例
            mock_agent_service = MagicMock()
            mock_agent_service_class.return_value = mock_agent_service

            # Mock Task query返回任务
            mock_task_query = MagicMock()
            mock_task_query.first.return_value = sample_task
            mock_db.query.return_value = mock_task_query

            # 调用方法
            agent = await agent_manager.get_agent_for_task(1, mock_db, user=mock_user)

        # 验证结果
        assert agent is not None
        assert 1 in agent_manager._agents
        assert agent_manager._agents[1] == mock_agent_service

    @pytest.mark.asyncio
    async def test_get_agent_for_task_existing_task_with_reconstruction(
        self,
        agent_manager,
        mock_db,
        sample_task,
        sample_trace_events,
        sample_dag_execution,
        mock_user,
    ):
        """测试获取已存在任务的agent，并进行重建"""
        # 使用更高级的方法直接patch AgentService创建
        with (
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
            patch("xagent.web.api.chat.resolve_llms_from_names") as mock_resolve_llms,
            patch("xagent.web.api.chat.get_memory_store") as mock_get_memory,
            patch("xagent.web.api.chat.Tracer"),
            patch("xagent.web.api.chat.create_default_tools") as mock_create_tools,
        ):
            # 设置所有必要的mock
            mock_resolve_llms.return_value = (MagicMock(), None, None, None)
            mock_get_memory.return_value = MagicMock()
            mock_create_tools.return_value = ([], MagicMock())  # (tools, tool_config)

            # 创建mock AgentService实例
            mock_agent_instance = MagicMock()
            mock_agent_instance.reconstruct_from_history = AsyncMock()
            mock_agent_service_class.return_value = mock_agent_instance

            # Mock database查询
            mock_task_query = MagicMock()
            mock_task_query.first.return_value = sample_task
            mock_task_query.filter.return_value = mock_task_query

            mock_dag_query = MagicMock()
            mock_dag_query.first.return_value = sample_dag_execution
            mock_dag_query.filter.return_value = mock_dag_query

            mock_trace_query = MagicMock()
            mock_trace_query.all.return_value = sample_trace_events
            mock_trace_query.filter.return_value = mock_trace_query

            def mock_query_side_effect(model):
                from xagent.web.models.task import DAGExecution, Task, TraceEvent

                if model == Task:
                    return mock_task_query
                elif model == DAGExecution:
                    return mock_dag_query
                elif model == TraceEvent:
                    return mock_trace_query
                else:
                    return MagicMock()

            mock_db.query.side_effect = mock_query_side_effect

            # 调用方法
            agent = await agent_manager.get_agent_for_task(1, mock_db, user=mock_user)

        # 验证结果
        assert agent is not None
        assert 1 in agent_manager._agents
        assert agent_manager._agents[1] == mock_agent_instance
        # 验证reconstruct_from_history被调用
        mock_agent_instance.reconstruct_from_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconstruct_agent_from_history_success(
        self, agent_manager, mock_db, sample_trace_events, sample_dag_execution
    ):
        """测试从历史数据重建agent成功"""
        # Mock查询
        mock_trace_query = MagicMock()
        mock_trace_query.all.return_value = sample_trace_events
        mock_trace_query.filter.return_value = mock_trace_query
        mock_dag_query = MagicMock()
        mock_dag_query.first.return_value = sample_dag_execution
        mock_dag_query.filter.return_value = mock_dag_query

        def mock_query_side_effect(model):
            from xagent.web.models.task import DAGExecution, TraceEvent

            if model == TraceEvent:
                return mock_trace_query
            elif model == DAGExecution:
                return mock_dag_query
            else:
                return MagicMock()

        mock_db.query.side_effect = mock_query_side_effect

        # Mock AgentService创建和reconstruct_from_history
        with (
            patch("xagent.web.api.chat.create_default_tools") as mock_create_tools,
            patch("xagent.web.api.chat.AgentService") as mock_agent_service_class,
        ):
            # 设置mock AgentService实例
            mock_agent_instance = MagicMock()
            mock_agent_instance.reconstruct_from_history = AsyncMock()
            mock_agent_service_class.return_value = mock_agent_instance

            mock_create_tools.return_value = ([], MagicMock())  # (tools, tool_config)

            # 调用方法
            await agent_manager._reconstruct_agent_from_history(1, mock_db)

        # 验证agent被创建
        assert 1 in agent_manager._agents
        # 验证reconstruct_from_history被调用
        mock_agent_instance.reconstruct_from_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconstruct_agent_from_history_no_data(self, agent_manager, mock_db):
        """测试没有历史数据时的重建"""
        # Mock查询返回空结果
        mock_trace_query = MagicMock()
        mock_trace_query.all.return_value = []
        mock_dag_query = MagicMock()
        mock_dag_query.first.return_value = None

        query_results = [mock_trace_query, mock_dag_query]
        mock_db.query.side_effect = lambda model: (
            query_results.pop(0) if query_results else MagicMock()
        )

        # 调用方法应该抛出异常
        with pytest.raises(ValueError) as exc_info:
            await agent_manager._reconstruct_agent_from_history(1, mock_db)

        assert "No historical data found" in str(exc_info.value)
        # 验证没有创建agent
        assert 1 not in agent_manager._agents

    @pytest.mark.asyncio
    async def test_reconstruct_agent_from_history_error_handling(
        self, agent_manager, mock_db
    ):
        """测试重建过程中的错误处理"""
        # Mock查询抛出异常
        mock_db.query.side_effect = Exception("Database error")

        # 调用方法应该抛出异常
        with pytest.raises(Exception) as exc_info:
            await agent_manager._reconstruct_agent_from_history(1, mock_db)

        assert "Database error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_agent_for_task_db_error_handling(
        self, agent_manager, mock_db, mock_user
    ):
        """测试数据库错误处理"""
        # Mock task query抛出异常
        mock_db.query.side_effect = Exception("Database error")

        # 调用方法应该正常处理错误并创建新agent
        with patch("xagent.web.api.chat.create_default_tools") as mock_create_tools:
            mock_create_tools.return_value = ([], MagicMock())  # (tools, tool_config)
            agent = await agent_manager.get_agent_for_task(1, mock_db, user=mock_user)

        # 验证结果
        assert agent is not None
        assert 1 in agent_manager._agents

    @pytest.mark.asyncio
    async def test_get_agent_for_task_task_creation(
        self, agent_manager, mock_db, mock_user
    ):
        """测试自动创建任务记录"""
        # Mock task query返回None（任务不存在）
        mock_task_query = MagicMock()
        mock_task_query.first.return_value = None
        mock_task_query.filter.return_value = mock_task_query

        # Mock DAG query也返回None
        mock_dag_query = MagicMock()
        mock_dag_query.first.return_value = None
        mock_dag_query.filter.return_value = mock_dag_query

        # Mock TraceEvent查询返回空列表
        mock_trace_query = MagicMock()
        mock_trace_query.all.return_value = []
        mock_trace_query.filter.return_value = mock_trace_query

        # 设置query的side effect来模拟多个查询
        def mock_query_side_effect(model):
            from xagent.web.models.task import DAGExecution, Task, TraceEvent

            if model == Task:
                return mock_task_query
            elif model == DAGExecution:
                return mock_dag_query
            elif model == TraceEvent:
                return mock_trace_query
            else:
                return MagicMock()

        mock_db.query.side_effect = mock_query_side_effect

        # Mock commit
        mock_db.commit = MagicMock()
        mock_db.refresh = MagicMock()
        mock_db.add = MagicMock()

        # 创建 mock user
        mock_user = MagicMock()
        mock_user.id = 1

        # 调用方法
        with patch("xagent.web.api.chat.create_default_tools") as mock_create_tools:
            mock_create_tools.return_value = ([], MagicMock())  # (tools, tool_config)
            await agent_manager.get_agent_for_task(1, mock_db, user=mock_user)

        # 验证任务被创建
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()

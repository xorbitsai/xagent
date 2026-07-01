"""Tests for CreateAgentTool - dynamically creating agents during task execution."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.utils.mock_helpers import create_langfuse_mock
from xagent.core.tools.adapters.vibe.agent_tool import (
    AgentTool,
    CreateAgentTool,
    ListAgentsTool,
    ListToolCategoriesTool,
    UpdateAgentTool,
    _coerce_db_task_id,
    gen_agent_tool_name,
    get_published_agents_tools,
)
from xagent.core.tracing.langfuse.handler import LangfuseTraceHandler
from xagent.core.workspace import TaskWorkspace
from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


def _create_session() -> tuple[Session, str, Any]:
    """Create a temporary database session for testing.

    Returns:
        (session, db_path, SessionLocal) — the open session, path to the
        SQLite file (for cleanup), and the sessionmaker so callers can open
        fresh sessions after the per-call session is closed.
    """
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    db_url = f"sqlite:///{temp_db.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal(), temp_db.name, SessionLocal


@pytest.fixture
def mock_workspace_db():
    yield


class TestCreateAgentTool:
    """Test suite for CreateAgentTool."""

    def test_create_agent_tool_schema_anchors_persisted_text_language(self) -> None:
        tool = CreateAgentTool(session_factory=None, user_id=1)

        assert (
            "same natural language as the current output language policy"
            in tool.description
        )
        assert "Do not inherit another language from DAG step text" in tool.description

        schema = tool.args_type().model_json_schema()
        description_schema = schema["properties"]["description"]["description"]
        instructions_schema = schema["properties"]["instructions"]["description"]
        assert (
            "same natural language as the current output language policy"
            in description_schema
        )
        assert (
            "same natural language as the current output language policy"
            in instructions_schema
        )

    @pytest.mark.asyncio
    async def test_assignable_tool_categories_hide_other(self) -> None:
        categories = (await ListToolCategoriesTool().run_json_async({}))["categories"]
        create_description = CreateAgentTool(
            session_factory=None, user_id=1
        ).description
        update_description = UpdateAgentTool(
            session_factory=None, user_id=1
        ).description
        create_categories_line = next(
            line
            for line in create_description.splitlines()
            if "Available categories:" in line
        )
        update_categories_line = next(
            line
            for line in update_description.splitlines()
            if "Available categories:" in line
        )

        assert "other" not in categories
        assert "other" not in create_categories_line
        assert "other" not in update_categories_line

    def test_coerce_db_task_id_accepts_only_db_task_formats(self) -> None:
        assert _coerce_db_task_id(12) == 12
        assert _coerce_db_task_id("12") == 12
        assert _coerce_db_task_id("web_task_12") == 12
        assert _coerce_db_task_id("task_12") == 12

        assert _coerce_db_task_id("agent_1_abcd1234") is None
        assert _coerce_db_task_id("agent_1_12") is None
        assert _coerce_db_task_id("web_task_abc") is None
        assert _coerce_db_task_id(True) is None

    @pytest.mark.asyncio
    async def test_create_agent_success(self) -> None:
        """Test successful agent creation."""
        db, db_path, SessionLocal = _create_session()
        try:
            # Create test user
            user = User(username="testuser", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = user.id
            db.close()

            # Mock model storage to return default LLM
            mock_llm = Mock()
            mock_llm.model_id = "gpt-4"

            with (
                patch(
                    "xagent.web.services.llm_utils.UserAwareModelStorage"
                ) as mock_storage_class,
                patch(
                    "xagent.web.services.agent_store.invalidate_agent_cache"
                ) as mock_invalidate_agent_cache,
            ):
                mock_storage = Mock()
                mock_storage.get_configured_defaults.return_value = (
                    mock_llm,
                    None,
                    None,
                    None,
                )
                mock_storage_class.return_value = mock_storage

                # Create tool
                tool = CreateAgentTool(
                    session_factory=SessionLocal, user_id=user_id, task_id="test_task"
                )

                # Execute tool
                result = await tool.run_json_async(
                    {
                        "name": "test_agent",
                        "description": "A test agent for unit testing",
                        "instructions": "You are a test agent for unit testing.",
                    }
                )

                # Verify result
                assert result["status"] == "success"
                assert result["agent_name"] == "test_agent"
                assert result["agent_id"] > 0
                assert result["tool_name"] == f"agent_{result['agent_id']}"
                assert "test_agent" in result["markdown_link"]
                assert "agent://" in result["markdown_link"]
                mock_invalidate_agent_cache.assert_called_once_with(
                    user_id, result["agent_id"]
                )

                # Verify agent was created in database (use a fresh session)
                verify_db = SessionLocal()
                try:
                    agent = (
                        verify_db.query(Agent)
                        .filter(Agent.name == "test_agent", Agent.user_id == user_id)
                        .first()
                    )
                    assert agent is not None
                    assert agent.status == AgentStatus.DRAFT
                    assert (
                        agent.instructions == "You are a test agent for unit testing."
                    )
                finally:
                    verify_db.close()

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_agent_tool_rejects_generated_workforce_manager(self) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(
                username="testuser_generated_manager_run",
                password_hash="x",
                is_admin=False,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            generated_manager = Agent(
                user_id=user.id,
                name="Generated Manager",
                description="Private workforce manager",
                instructions="Coordinate the workforce.",
                status=AgentStatus.PUBLISHED,
                origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
            )
            db.add(generated_manager)
            db.commit()
            db.refresh(generated_manager)

            tool = AgentTool(
                agent_id=generated_manager.id,
                agent_name=generated_manager.name,
                agent_description=generated_manager.description or "",
                session_factory=SessionLocal,
                user_id=user.id,
            )

            with patch(
                "xagent.core.agent.service.AgentService"
            ) as mock_agent_service_class:
                result = await tool.run_json_async({"task": "run private manager"})

            assert (
                result["response"] == f"Error: Agent {generated_manager.id} not found"
            )
            mock_agent_service_class.assert_not_called()
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_agent_tool_applies_workforce_runtime_overrides(self) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser11", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            model = Model(
                model_id="test-model-id",
                category="llm",
                model_provider="openai",
                model_name="gpt-4",
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                temperature=0.7,
                abilities=["chat"],
            )
            db.add(model)
            db.commit()
            db.refresh(model)

            agent = Agent(
                user_id=user.id,
                name="Worker Agent",
                description="Nested workforce worker",
                instructions="Base worker instructions.",
                status=AgentStatus.PUBLISHED,
                models={"general": model.id},
            )
            db.add(agent)
            db.commit()
            db.refresh(agent)

            parent_handler = Mock()

            class ParentTracer:
                def __init__(self) -> None:
                    self.handlers = [parent_handler]
                    self.events = []

                async def trace_event(
                    self, event_type, task_id=None, step_id=None, data=None
                ):
                    self.events.append(
                        {
                            "event_type": event_type.value,
                            "task_id": task_id,
                            "step_id": step_id,
                            "data": data or {},
                        }
                    )

            parent_tracer = ParentTracer()

            tool = AgentTool(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_description=agent.description or "",
                session_factory=SessionLocal,
                user_id=user.id,
                task_id="tool-session",
                tool_name="call_workforce_worker_7_writer",
                tool_description="Write the final report.",
                extra_system_prompt="Workforce assignment: write only.",
                parent_task_id="parent-task-2",
                parent_tracer=parent_tracer,
                agent_call_stack=[99],
                delegation_allowed_agent_ids=[],
                enable_global_agent_tools=False,
                runtime_metadata={"workforce_id": 123, "worker_alias": "Writer"},
            )

            with (
                patch(
                    "xagent.web.services.llm_utils.UserAwareModelStorage"
                ) as mock_storage_class,
                patch(
                    "xagent.core.agent.service.AgentService"
                ) as mock_agent_service_class,
                patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
            ):
                mock_storage = Mock()
                mock_llm = Mock()
                mock_storage.get_llm_by_name_with_access.return_value = mock_llm
                mock_storage_class.return_value = mock_storage

                mock_agent_service = mock_agent_service_class.return_value
                mock_agent_service.execute_task = AsyncMock(
                    return_value={
                        "output": "worker response",
                        "file_outputs": [{"filename": "report.txt"}],
                    }
                )

                result = await tool.run_json_async({"task": "draft report"})

            assert tool.name == f"agent_{agent.id}"
            assert tool.description == "Write the final report."
            assert result["response"] == "worker response"
            assert result["file_outputs"] == []
            assert [event["data"]["event_type"] for event in parent_tracer.events] == [
                "workforce_delegation_start",
                "workforce_delegation_end",
            ]
            assert [event["event_type"] for event in parent_tracer.events] == [
                "task_update_general",
                "task_update_general",
            ]
            assert parent_tracer.events[0]["task_id"] == "parent-task-2"
            assert "__audit_only__" not in parent_tracer.events[0]["data"]
            assert parent_tracer.events[0]["data"]["status"] == "start"
            assert parent_tracer.events[0]["data"]["workforce_id"] == 123
            assert parent_tracer.events[0]["data"]["worker_alias"] == "Writer"
            assert "__audit_only__" not in parent_tracer.events[1]["data"]
            assert parent_tracer.events[1]["data"]["status"] == "end"
            assert parent_tracer.events[1]["data"]["output"] == "worker response"
            assert parent_tracer.events[1]["data"]["output_length"] == len(
                "worker response"
            )
            assert "file_outputs" not in parent_tracer.events[1]["data"]

            tool_config = mock_agent_service_class.call_args.kwargs["tool_config"]
            assert tool_config.get_allowed_agent_ids() == []
            assert tool_config.get_enable_global_agent_tools() is False
            assert tool_config.get_parent_task_id() == "parent-task-2"
            assert tool_config.get_parent_tracer() is parent_tracer
            assert tool_config.get_agent_call_stack() == [99, agent.id]

            tracer = mock_agent_service_class.call_args.kwargs["tracer"]
            assert tracer is not parent_tracer
            assert parent_handler not in tracer.handlers

            execute_context = mock_agent_service.execute_task.call_args.kwargs[
                "context"
            ]
            assert execute_context["system_prompt"] == (
                "Base worker instructions.\n\nWorkforce assignment: write only."
            )
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_agent_tool_returns_parent_owned_file_refs_for_worker_outputs(
        self,
    ) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(
                username="worker-output-user", password_hash="x", is_admin=False
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            task = Task(id=77, user_id=user.id, title="Parent task")
            db.add(task)
            db.commit()

            model = Model(
                model_id="test-model-id",
                category="llm",
                model_provider="openai",
                model_name="gpt-4",
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                temperature=0.7,
                abilities=["chat"],
            )
            db.add(model)
            db.commit()
            db.refresh(model)

            agent = Agent(
                user_id=user.id,
                name="File Worker",
                description="Writes files",
                instructions="Write a report.",
                status=AgentStatus.PUBLISHED,
                models={"general": model.id},
            )
            db.add(agent)
            db.commit()
            db.refresh(agent)

            class ParentTracer:
                def __init__(self) -> None:
                    self.handlers = []
                    self.events = []

                async def trace_event(
                    self, event_type, task_id=None, step_id=None, data=None
                ):
                    self.events.append(
                        {
                            "event_type": event_type.value,
                            "task_id": task_id,
                            "step_id": step_id,
                            "data": data or {},
                        }
                    )

            parent_tracer = ParentTracer()

            with tempfile.TemporaryDirectory() as workspace_root:
                worker_workspace = TaskWorkspace(
                    id=f"agent_{agent.id}_abcd1234",
                    base_dir=workspace_root,
                    db_task_id=77,
                )
                output_path = worker_workspace.output_dir / "report.txt"
                output_path.write_text("worker report", encoding="utf-8")

                tool = AgentTool(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    agent_description=agent.description or "",
                    session_factory=SessionLocal,
                    user_id=user.id,
                    task_id="77",
                    parent_task_id="77",
                    parent_tracer=parent_tracer,
                    workspace_base_dir=workspace_root,
                    runtime_metadata={"workforce_id": 1},
                )

                with (
                    patch(
                        "xagent.web.services.llm_utils.UserAwareModelStorage"
                    ) as mock_storage_class,
                    patch(
                        "xagent.core.agent.service.AgentService"
                    ) as mock_agent_service_class,
                    patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
                ):
                    mock_storage = Mock()
                    mock_llm = Mock()
                    mock_storage.get_llm_by_name_with_access.return_value = mock_llm
                    mock_storage_class.return_value = mock_storage

                    mock_agent_service = mock_agent_service_class.return_value
                    mock_agent_service.workspace = worker_workspace
                    mock_agent_service.execute_task = AsyncMock(
                        return_value={
                            "output": "worker response",
                            "file_outputs": [
                                {
                                    "file_path": str(output_path),
                                    "relative_path": "output/report.txt",
                                    "filename": "report.txt",
                                }
                            ],
                        }
                    )

                    result = await tool.run_json_async({"task": "draft report"})

                file_outputs = result["file_outputs"]
                assert len(file_outputs) == 1
                assert file_outputs[0]["filename"] == "report.txt"
                assert file_outputs[0]["download_url"].startswith(
                    "/api/files/download/"
                )
                assert "file_path" not in file_outputs[0]
                assert "relative_path" not in file_outputs[0]

                file_record = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.file_id == file_outputs[0]["file_id"])
                    .one()
                )
                canonical_path = (
                    Path(workspace_root)
                    / f"user_{user.id}"
                    / "web_task_77"
                    / "output"
                    / "report.txt"
                )
                assert file_record.user_id == user.id
                assert file_record.task_id == 77
                assert file_record.storage_path == str(canonical_path)
                assert canonical_path.read_text(encoding="utf-8") == "worker report"
                assert file_record.workspace_relative_path == "output/report.txt"
                assert file_record.workspace_category == "output"

                tool_config = mock_agent_service_class.call_args.kwargs["tool_config"]
                assert tool_config.get_workspace_config()["db_task_id"] == 77
                assert parent_tracer.events[-1]["data"]["file_outputs"] == file_outputs

                tracer = mock_agent_service_class.call_args.kwargs["tracer"]
                execution_task_id = mock_agent_service_class.call_args.kwargs["task_id"]
                db_handlers = [
                    handler
                    for handler in tracer.handlers
                    if handler.__class__.__name__
                    == "_DelegatedAgentDatabaseTraceHandler"
                ]
                assert len(db_handlers) == 1
                assert db_handlers[0].task_id == 77
                assert db_handlers[0].build_id == execution_task_id
                assert db_handlers[0].metadata["worker_task_id"] == execution_task_id
                assert db_handlers[0].metadata["parent_task_id"] == "77"
                assert db_handlers[0].metadata["parent_db_task_id"] == 77
                assert db_handlers[0].metadata["agent_id"] == agent.id
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_agent_tool_rebinds_worker_owned_file_ids_to_parent_task(self) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(
                username="worker-owned-output-user",
                password_hash="x",
                is_admin=False,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            parent_task = Task(id=77, user_id=user.id, title="Parent task")
            worker_task = Task(id=88, user_id=user.id, title="Worker task")
            db.add_all([parent_task, worker_task])
            db.commit()

            with tempfile.TemporaryDirectory() as workspace_root:
                worker_workspace = TaskWorkspace(
                    id="agent_1_abcd1234",
                    base_dir=workspace_root,
                    db_task_id=77,
                )
                output_path = worker_workspace.output_dir / "report.txt"
                output_path.write_text("worker report", encoding="utf-8")

                db.add(
                    UploadedFile(
                        file_id="worker-owned-output",
                        user_id=user.id,
                        task_id=88,
                        filename="report.txt",
                        storage_path=str(output_path),
                        mime_type="text/plain",
                        file_size=len("worker report"),
                        workspace_relative_path="output/report.txt",
                        workspace_category="output",
                    )
                )
                db.commit()

                tool = AgentTool(
                    agent_id=1,
                    agent_name="File Worker",
                    agent_description="Writes files",
                    session_factory=SessionLocal,
                    user_id=user.id,
                    task_id="77",
                    parent_task_id="77",
                    workspace_base_dir=workspace_root,
                )

                file_outputs = tool._parent_owned_file_outputs(
                    [
                        {
                            "file_id": "worker-owned-output",
                            "file_path": str(output_path),
                            "filename": "report.txt",
                        }
                    ],
                    worker_workspace,
                    db,
                )

                assert file_outputs is not None
                assert len(file_outputs) == 1
                assert file_outputs[0]["file_id"] == "worker-owned-output"
                file_record = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.file_id == "worker-owned-output")
                    .one()
                )
                canonical_path = (
                    Path(workspace_root)
                    / f"user_{user.id}"
                    / "web_task_77"
                    / "output"
                    / "report.txt"
                )
                assert file_record.task_id == 77
                assert file_record.storage_path == str(canonical_path)
                assert canonical_path.read_text(encoding="utf-8") == "worker report"
                assert file_record.workspace_relative_path == "output/report.txt"
                assert file_record.workspace_category == "output"
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_agent_tool_omits_workforce_file_outputs_without_parent_db_task_id(
        self,
    ) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            tool = AgentTool(
                agent_id=1,
                agent_name="File Worker",
                agent_description="Writes files",
                session_factory=SessionLocal,
                user_id=1,
                task_id="agent_1_abcd1234",
                parent_task_id="agent_1_abcd1234",
                runtime_metadata={"workforce_id": 1},
            )

            file_outputs = [{"file_path": "/tmp/worker-output.txt"}]

            assert (
                tool._parent_owned_file_outputs(file_outputs, workspace=None, db=db)
                == []
            )
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_agent_tool_preserves_non_workforce_file_outputs_without_parent_db_task_id(
        self,
    ) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            tool = AgentTool(
                agent_id=1,
                agent_name="File Worker",
                agent_description="Writes files",
                session_factory=SessionLocal,
                user_id=1,
                task_id="agent_1_abcd1234",
                parent_task_id="agent_1_abcd1234",
            )
            file_outputs = [{"file_path": "/tmp/legacy-output.txt"}]

            assert (
                tool._parent_owned_file_outputs(file_outputs, workspace=None, db=db)
                is file_outputs
            )
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_agent_tool_does_not_swallow_delegated_output_registration_errors(
        self,
        tmp_path,
    ) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            output_path = tmp_path / "report.txt"
            output_path.write_text("worker report", encoding="utf-8")
            workspace = Mock()
            workspace.resolve_path.return_value = output_path
            workspace.register_file.side_effect = RuntimeError("storage unavailable")

            tool = AgentTool(
                agent_id=1,
                agent_name="File Worker",
                agent_description="Writes files",
                session_factory=SessionLocal,
                user_id=1,
                task_id="77",
                parent_task_id="77",
            )

            with pytest.raises(RuntimeError, match="storage unavailable"):
                tool._parent_owned_file_outputs(
                    [{"file_path": str(output_path), "filename": "report.txt"}],
                    workspace,
                    db,
                )
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_with_tool_filters(self) -> None:
        """Test agent creation with tool categories and skills filters."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser2", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = user.id
            db.close()

            mock_llm = Mock()
            mock_llm.model_id = "gpt-4"

            with patch(
                "xagent.web.services.llm_utils.UserAwareModelStorage"
            ) as mock_storage_class:
                mock_storage = Mock()
                mock_storage.get_configured_defaults.return_value = (
                    mock_llm,
                    None,
                    None,
                    None,
                )
                mock_storage_class.return_value = mock_storage

                tool = CreateAgentTool(session_factory=SessionLocal, user_id=user_id)

                result = await tool.run_json_async(
                    {
                        "name": "filtered_agent",
                        "description": "Agent with filtered tools",
                        "instructions": "Agent with filtered tools",
                        "tool_categories": ["file", "knowledge"],
                        "skills": ["web_search"],
                    }
                )

                assert result["status"] == "success"

                # Verify filters were saved (use a fresh session)
                verify_db = SessionLocal()
                try:
                    agent = (
                        verify_db.query(Agent)
                        .filter(Agent.name == "filtered_agent")
                        .first()
                    )
                    assert agent is not None
                    assert agent.tool_categories == ["file", "knowledge"]
                    assert agent.skills == ["web_search"]
                finally:
                    verify_db.close()

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_name_auto_renames(self) -> None:
        """Test that duplicate agent names are auto-renamed and created."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser3", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            # Create existing agent
            existing_agent = Agent(
                user_id=user.id,
                name="duplicate_name",
                status=AgentStatus.DRAFT,
            )
            db.add(existing_agent)
            db.commit()
            user_id = user.id
            db.close()

            mock_llm = Mock()
            mock_llm.model_id = "gpt-4"

            with patch(
                "xagent.web.services.llm_utils.UserAwareModelStorage"
            ) as mock_storage_class:
                mock_storage = Mock()
                mock_storage.get_configured_defaults.return_value = (
                    mock_llm,
                    None,
                    None,
                    None,
                )
                mock_storage_class.return_value = mock_storage

                tool = CreateAgentTool(session_factory=SessionLocal, user_id=user_id)

                result = await tool.run_json_async(
                    {
                        "name": "duplicate_name",
                        "description": "Duplicate name test agent",
                        "instructions": "This should be auto-renamed",
                    }
                )

                assert result["status"] == "success"
                assert result["agent_name"] == "duplicate_name Assistant"
                assert "auto-renamed" in result["message"].lower()

                verify_db = SessionLocal()
                try:
                    created_agent = (
                        verify_db.query(Agent)
                        .filter(
                            Agent.user_id == user_id,
                            Agent.name == "duplicate_name Assistant",
                        )
                        .first()
                    )
                    assert created_agent is not None
                finally:
                    verify_db.close()

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_rejects_missing_knowledge_base(self) -> None:
        """Test that create_agent rejects knowledge bases that do not exist."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_missing_kb", password_hash="x")
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = user.id
            db.close()

            tool = CreateAgentTool(session_factory=SessionLocal, user_id=user_id)

            with patch(
                "xagent.core.tools.adapters.vibe.agent_tool.find_missing_knowledge_bases",
                new=AsyncMock(return_value=["missing_kb"]),
            ):
                result = await tool.run_json_async(
                    {
                        "name": "kb_agent",
                        "description": "Agent with KB",
                        "instructions": "Use the KB.",
                        "knowledge_bases": ["missing_kb"],
                    }
                )

            assert result["status"] == "error"
            assert "missing_kb" in result["message"]
            verify_db = SessionLocal()
            try:
                assert (
                    verify_db.query(Agent).filter(Agent.name == "kb_agent").first()
                    is None
                )
            finally:
                verify_db.close()

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_name_uses_next_available_variant(
        self,
    ) -> None:
        """Test that auto-rename skips occupied fallback names."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser3b", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            db.add_all(
                [
                    Agent(
                        user_id=user.id,
                        name="duplicate_name",
                        status=AgentStatus.DRAFT,
                    ),
                    Agent(
                        user_id=user.id,
                        name="duplicate_name Assistant",
                        status=AgentStatus.DRAFT,
                    ),
                ]
            )
            db.commit()
            user_id = user.id
            db.close()

            mock_llm = Mock()
            mock_llm.model_id = "gpt-4"

            with patch(
                "xagent.web.services.llm_utils.UserAwareModelStorage"
            ) as mock_storage_class:
                mock_storage = Mock()
                mock_storage.get_configured_defaults.return_value = (
                    mock_llm,
                    None,
                    None,
                    None,
                )
                mock_storage_class.return_value = mock_storage

                tool = CreateAgentTool(session_factory=SessionLocal, user_id=user_id)

                result = await tool.run_json_async(
                    {
                        "name": "duplicate_name",
                        "description": "Duplicate name test agent",
                        "instructions": "This should use the next fallback",
                    }
                )

                assert result["status"] == "success"
                assert result["agent_name"] == "duplicate_name V2"

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_missing_name(self) -> None:
        """Test that missing name returns error."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser4", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = user.id
            db.close()

            tool = CreateAgentTool(session_factory=SessionLocal, user_id=user_id)

            result = await tool.run_json_async(
                {
                    "name": "",
                    "description": "Test missing name",
                    "instructions": "Instructions without name",
                }
            )

            assert result["status"] == "error"
            assert "required" in result["message"].lower()

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_missing_instructions(self) -> None:
        """Test that missing instructions returns error."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser5", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = user.id
            db.close()

            tool = CreateAgentTool(session_factory=SessionLocal, user_id=user_id)

            result = await tool.run_json_async(
                {
                    "name": "test",
                    "description": "Test missing instructions",
                    "instructions": "",
                }
            )

            assert result["status"] == "error"
            assert "required" in result["message"].lower()

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass


class TestUpdateAgentTool:
    """Test suite for UpdateAgentTool."""

    @pytest.mark.asyncio
    async def test_update_agent_success(self) -> None:
        """Test successful agent update."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_update", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            # Create existing agent
            existing_agent = Agent(
                user_id=user.id,
                name="original_name",
                description="Original description",
                instructions="Original instructions",
                status=AgentStatus.DRAFT,
            )
            db.add(existing_agent)
            db.commit()
            db.refresh(existing_agent)

            with patch(
                "xagent.web.services.agent_store.invalidate_agent_cache"
            ) as mock_invalidate_agent_cache:
                tool = UpdateAgentTool(
                    session_factory=SessionLocal, user_id=user.id, task_id="test_task"
                )

                result = await tool.run_json_async(
                    {
                        "agent_id": existing_agent.id,
                        "name": "updated_name",
                        "description": "Updated description",
                        "instructions": "Updated instructions",
                    }
                )
                mock_invalidate_agent_cache.assert_called_once_with(
                    user.id, existing_agent.id
                )

            # Verify result
            assert result["status"] == "success"
            assert result["agent_name"] == "updated_name"
            assert result["agent_id"] == existing_agent.id

            # Verify agent was updated in database
            db.refresh(existing_agent)
            assert existing_agent.name == "updated_name"
            assert existing_agent.description == "Updated description"
            assert existing_agent.instructions == "Updated instructions"

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_update_agent_partial_update(self) -> None:
        """Test partial agent update (only some fields)."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_partial", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            existing_agent = Agent(
                user_id=user.id,
                name="partial_agent",
                description="Original description",
                instructions="Original instructions",
                status=AgentStatus.DRAFT,
            )
            db.add(existing_agent)
            db.commit()
            db.refresh(existing_agent)

            tool = UpdateAgentTool(session_factory=SessionLocal, user_id=user.id)

            # Update only description, keep name and instructions
            result = await tool.run_json_async(
                {
                    "agent_id": existing_agent.id,
                    "description": "New description only",
                }
            )

            assert result["status"] == "success"

            # Verify only description changed
            db.refresh(existing_agent)
            assert existing_agent.name == "partial_agent"  # Unchanged
            assert existing_agent.description == "New description only"  # Changed
            assert existing_agent.instructions == "Original instructions"  # Unchanged

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_update_agent_not_found(self) -> None:
        """Test updating non-existent agent."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_notfound", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            tool = UpdateAgentTool(session_factory=SessionLocal, user_id=user.id)

            result = await tool.run_json_async(
                {
                    "agent_id": 99999,  # Non-existent ID
                    "name": "new_name",
                }
            )

            assert result["status"] == "error"
            assert "not found" in result["message"].lower()

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_update_agent_rejects_generated_workforce_manager(self) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(
                username="testuser_update_generated_manager",
                password_hash="x",
                is_admin=False,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            generated_manager = Agent(
                user_id=user.id,
                name="Generated Manager",
                description="Private manager",
                instructions="Coordinate the workforce.",
                status=AgentStatus.PUBLISHED,
                origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
            )
            db.add(generated_manager)
            db.commit()
            db.refresh(generated_manager)

            tool = UpdateAgentTool(session_factory=SessionLocal, user_id=user.id)

            result = await tool.run_json_async(
                {
                    "agent_id": generated_manager.id,
                    "name": "renamed_manager",
                }
            )

            assert result["status"] == "error"
            assert "not found" in result["message"].lower()
            db.refresh(generated_manager)
            assert generated_manager.name == "Generated Manager"

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_update_agent_name_conflict_ignores_generated_workforce_manager(
        self,
    ) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(
                username="testuser_update_generated_manager_name",
                password_hash="x",
                is_admin=False,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            visible_agent = Agent(
                user_id=user.id,
                name="Visible Agent",
                status=AgentStatus.DRAFT,
            )
            generated_manager = Agent(
                user_id=user.id,
                name="Hidden Manager Name",
                status=AgentStatus.PUBLISHED,
                origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
            )
            db.add_all([visible_agent, generated_manager])
            db.commit()
            db.refresh(visible_agent)

            tool = UpdateAgentTool(session_factory=SessionLocal, user_id=user.id)

            result = await tool.run_json_async(
                {
                    "agent_id": visible_agent.id,
                    "name": "Hidden Manager Name",
                }
            )

            assert result["status"] == "success"
            db.refresh(visible_agent)
            assert visible_agent.name == "Hidden Manager Name"

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_update_agent_rejects_missing_knowledge_base(self) -> None:
        """Test that update_agent rejects knowledge bases that do not exist."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_update_missing_kb", password_hash="x")
            db.add(user)
            db.commit()
            db.refresh(user)

            existing_agent = Agent(
                user_id=user.id,
                name="kb_update_agent",
                description="Original description",
                instructions="Original instructions",
                status=AgentStatus.DRAFT,
                knowledge_bases=[],
            )
            db.add(existing_agent)
            db.commit()
            db.refresh(existing_agent)

            tool = UpdateAgentTool(session_factory=SessionLocal, user_id=user.id)

            with patch(
                "xagent.core.tools.adapters.vibe.agent_tool.find_missing_knowledge_bases",
                new=AsyncMock(return_value=["missing_kb"]),
            ):
                result = await tool.run_json_async(
                    {
                        "agent_id": existing_agent.id,
                        "knowledge_bases": ["missing_kb"],
                    }
                )

            assert result["status"] == "error"
            assert "missing_kb" in result["message"]
            db.refresh(existing_agent)
            assert existing_agent.knowledge_bases == []

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_update_published_agent_success_preserves_status(self) -> None:
        """Test that published agents can be updated and remain published."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(
                username="testuser_published", password_hash="x", is_admin=False
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            published_agent = Agent(
                user_id=user.id,
                name="published_agent",
                description="Published agent",
                instructions="Instructions",
                status=AgentStatus.PUBLISHED,
            )
            db.add(published_agent)
            db.commit()
            db.refresh(published_agent)

            tool = UpdateAgentTool(session_factory=SessionLocal, user_id=user.id)

            result = await tool.run_json_async(
                {
                    "agent_id": published_agent.id,
                    "name": "trying_to_rename",
                    "instructions": "Updated published instructions",
                }
            )

            assert result["status"] == "success"
            assert result["agent_id"] == published_agent.id
            assert result["agent_name"] == "trying_to_rename"
            assert "Status: PUBLISHED" in result["message"]

            db.refresh(published_agent)
            assert published_agent.name == "trying_to_rename"
            assert published_agent.instructions == "Updated published instructions"
            assert published_agent.status == AgentStatus.PUBLISHED

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_update_archived_agent_rejected(self) -> None:
        """Test that archived agents cannot be updated."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_archived", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            archived_agent = Agent(
                user_id=user.id,
                name="archived_agent",
                description="Archived agent",
                instructions="Original instructions",
                status=AgentStatus.ARCHIVED,
            )
            db.add(archived_agent)
            db.commit()
            db.refresh(archived_agent)

            tool = UpdateAgentTool(session_factory=SessionLocal, user_id=user.id)

            result = await tool.run_json_async(
                {
                    "agent_id": archived_agent.id,
                    "name": "trying_to_rename",
                    "instructions": "Attempted update",
                }
            )

            assert result["status"] == "error"
            assert "archived agents cannot be updated" in result["message"].lower()

            db.refresh(archived_agent)
            assert archived_agent.name == "archived_agent"
            assert archived_agent.instructions == "Original instructions"
            assert archived_agent.status == AgentStatus.ARCHIVED

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_update_agent_duplicate_name(self) -> None:
        """Test that duplicate names are rejected when updating."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_dup", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            # Create two agents
            agent1 = Agent(
                user_id=user.id,
                name="agent_one",
                status=AgentStatus.DRAFT,
            )
            agent2 = Agent(
                user_id=user.id,
                name="agent_two",
                status=AgentStatus.DRAFT,
            )
            db.add_all([agent1, agent2])
            db.commit()
            db.refresh(agent1)
            db.refresh(agent2)

            tool = UpdateAgentTool(session_factory=SessionLocal, user_id=user.id)

            # Try to rename agent2 to agent_one (duplicate)
            result = await tool.run_json_async(
                {
                    "agent_id": agent2.id,
                    "name": "agent_one",
                }
            )

            assert result["status"] == "error"
            assert "already exists" in result["message"].lower()

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_update_agent_tool_opens_and_closes_session_per_call(self) -> None:
        """UpdateAgentTool must open exactly one session per call and close it."""
        import asyncio

        db, db_path, SessionLocal = _create_session()
        try:
            # Seed one agent to update
            user = User(
                username="testuser_update_session_lifecycle",
                password_hash="x",
                is_admin=False,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            existing_agent = Agent(
                user_id=user.id,
                name="session_lifecycle_agent",
                description="Agent for session lifecycle test",
                instructions="Original instructions",
                status=AgentStatus.DRAFT,
            )
            db.add(existing_agent)
            db.commit()
            db.refresh(existing_agent)
            seeded_id = existing_agent.id
            seeded_user_id = user.id
            db.close()

            opened: list = []
            closed: list = []

            def tracking_factory():
                s = SessionLocal()
                orig_close = s.close

                # Session has no __slots__; instance attr shadows the method
                def tracked_close():
                    closed.append(s)
                    orig_close()

                s.close = tracked_close
                opened.append(s)
                return s

            with patch("xagent.web.services.agent_store.invalidate_agent_cache"):
                tool = UpdateAgentTool(
                    session_factory=tracking_factory, user_id=seeded_user_id
                )
                result = asyncio.run(
                    tool.run_json_async(
                        {"agent_id": seeded_id, "name": "Renamed Agent"}
                    )
                )

            assert result["status"] == "success"
            assert len(opened) == 1, f"expected 1 session opened, got {len(opened)}"
            assert closed == opened, "session was not closed after the call"

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass


class TestListAgentsTool:
    """Test suite for ListAgentsTool."""

    @pytest.mark.asyncio
    async def test_list_all_agents(self) -> None:
        """Test listing all agents."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_list", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            # Create agents with different statuses
            draft_agent = Agent(
                user_id=user.id,
                name="draft_agent",
                description="Draft agent description",
                instructions="Draft instructions",
                status=AgentStatus.DRAFT,
            )
            published_agent = Agent(
                user_id=user.id,
                name="published_agent",
                description="Published agent description",
                instructions="Published instructions",
                status=AgentStatus.PUBLISHED,
            )
            archived_agent = Agent(
                user_id=user.id,
                name="archived_agent",
                description="Archived agent description",
                status=AgentStatus.ARCHIVED,
            )
            db.add_all([draft_agent, published_agent, archived_agent])
            db.commit()
            user_id = user.id
            db.close()

            tool = ListAgentsTool(session_factory=SessionLocal, user_id=user_id)

            result = await tool.run_json_async({})

            assert result["status"] == "success"
            assert result["total_count"] == 3
            assert len(result["agents"]) == 3

            # Check agent info
            agent_names = {agent["name"] for agent in result["agents"]}
            assert "draft_agent" in agent_names
            assert "published_agent" in agent_names
            assert "archived_agent" in agent_names

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_list_agents_hides_generated_workforce_managers(self) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(
                username="testuser_list_generated_manager",
                password_hash="x",
                is_admin=False,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            regular_agent = Agent(
                user_id=user.id,
                name="reusable_agent",
                description="User reusable agent",
                status=AgentStatus.PUBLISHED,
            )
            generated_manager = Agent(
                user_id=user.id,
                name="generated_manager",
                description="Private workforce manager",
                status=AgentStatus.PUBLISHED,
                origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
            )
            db.add_all([regular_agent, generated_manager])
            db.commit()
            user_id = user.id
            db.close()

            tool = ListAgentsTool(session_factory=SessionLocal, user_id=user_id)

            result = await tool.run_json_async({})

            assert result["status"] == "success"
            assert result["total_count"] == 1
            agent_names = {agent["name"] for agent in result["agents"]}
            assert agent_names == {"reusable_agent"}

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_list_agents_with_status_filter(self) -> None:
        """Test listing agents with status filter."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_filter", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            draft_agent = Agent(
                user_id=user.id,
                name="draft_agent",
                status=AgentStatus.DRAFT,
            )
            published_agent = Agent(
                user_id=user.id,
                name="published_agent",
                status=AgentStatus.PUBLISHED,
            )
            db.add_all([draft_agent, published_agent])
            db.commit()
            user_id = user.id
            db.close()

            tool = ListAgentsTool(session_factory=SessionLocal, user_id=user_id)

            # List only draft agents
            result = await tool.run_json_async({"status_filter": "draft"})

            assert result["status"] == "success"
            assert result["total_count"] == 1
            assert result["agents"][0]["name"] == "draft_agent"
            assert result["agents"][0]["status"] == "draft"

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_list_agents_user_isolation(self) -> None:
        """Test that users can only see their own agents."""
        db, db_path, SessionLocal = _create_session()
        try:
            user1 = User(username="listuser1", password_hash="x", is_admin=False)
            user2 = User(username="listuser2", password_hash="x", is_admin=False)
            db.add_all([user1, user2])
            db.commit()
            db.refresh(user1)
            db.refresh(user2)

            # Create agents for user1
            user1_agent = Agent(
                user_id=user1.id,
                name="user1_agent",
                status=AgentStatus.DRAFT,
            )
            # Create agents for user2
            user2_agent = Agent(
                user_id=user2.id,
                name="user2_agent",
                status=AgentStatus.DRAFT,
            )
            db.add_all([user1_agent, user2_agent])
            db.commit()
            user1_id = user1.id
            db.close()

            # User1 should only see their own agents
            tool = ListAgentsTool(session_factory=SessionLocal, user_id=user1_id)
            result = await tool.run_json_async({})

            assert result["status"] == "success"
            assert result["total_count"] == 1
            assert result["agents"][0]["name"] == "user1_agent"

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_list_agents_invalid_status_filter(self) -> None:
        """Test that invalid status filter returns error."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser_invalid", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = user.id
            db.close()

            tool = ListAgentsTool(session_factory=SessionLocal, user_id=user_id)

            result = await tool.run_json_async({"status_filter": "invalid_status"})

            assert result["status"] == "error"
            assert "invalid" in result["message"].lower()

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_list_agents_tool_opens_and_closes_session_per_call(self) -> None:
        """ListAgentsTool must open exactly one session per call and close it."""
        import asyncio

        db, db_path, SessionLocal = _create_session()
        try:
            user = User(
                username="testuser_list_session_lifecycle",
                password_hash="x",
                is_admin=False,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            agent = Agent(
                user_id=user.id,
                name="session_lifecycle_list_agent",
                description="Agent for list session lifecycle test",
                instructions="Some instructions",
                status=AgentStatus.DRAFT,
            )
            db.add(agent)
            db.commit()
            seeded_user_id = user.id
            db.close()

            opened: list = []
            closed: list = []

            def tracking_factory():
                s = SessionLocal()
                orig_close = s.close

                # Session has no __slots__; instance attr shadows the method
                def tracked_close():
                    closed.append(s)
                    orig_close()

                s.close = tracked_close
                opened.append(s)
                return s

            tool = ListAgentsTool(
                session_factory=tracking_factory, user_id=seeded_user_id
            )
            result = asyncio.run(tool.run_json_async({}))

            assert result["status"] == "success"
            assert result["total_count"] == 1
            assert len(opened) == 1, f"expected 1 session opened, got {len(opened)}"
            assert closed == opened, "session was not closed after the call"

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass


class TestAgentToolNameGeneration:
    """Test suite for agent tool name generation."""

    def test_gen_agent_tool_name_simple(self) -> None:
        """Test tool name generation with an agent ID."""
        result = gen_agent_tool_name(42)
        assert result == "agent_42"

    def test_gen_agent_tool_name_with_string_id(self) -> None:
        """Test tool name generation with a string agent ID."""
        result = gen_agent_tool_name("42")
        assert result == "agent_42"

    def test_gen_agent_tool_name_rejects_names(self) -> None:
        """Test tool name generation rejects display names."""
        with pytest.raises(ValueError):
            gen_agent_tool_name("Research Assistant")


class TestDraftAgentsInTools:
    """Test suite for including draft agents in tool lists."""

    def test_get_tools_with_draft_disabled(self) -> None:
        """Test that draft agents are excluded when include_draft=False."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser6", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            published_agent = Agent(
                user_id=user.id,
                name="Published Agent",
                status=AgentStatus.PUBLISHED,
            )
            draft_agent = Agent(
                user_id=user.id,
                name="Draft Agent",
                status=AgentStatus.DRAFT,
            )
            db.add_all([published_agent, draft_agent])
            db.commit()

            tools = get_published_agents_tools(
                session_factory=SessionLocal, user_id=user.id, include_draft=False
            )
            tool_names = {tool.name for tool in tools}

            assert f"agent_{published_agent.id}" in tool_names
            assert f"agent_{draft_agent.id}" not in tool_names

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_get_tools_with_draft_enabled(self) -> None:
        """Test that draft agents are included when include_draft=True."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser7", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            published_agent = Agent(
                user_id=user.id,
                name="Published Agent",
                status=AgentStatus.PUBLISHED,
            )
            draft_agent = Agent(
                user_id=user.id,
                name="Draft Agent",
                status=AgentStatus.DRAFT,
            )
            db.add_all([published_agent, draft_agent])
            db.commit()

            tools = get_published_agents_tools(
                session_factory=SessionLocal, user_id=user.id, include_draft=True
            )
            tool_names = {tool.name for tool in tools}

            assert f"agent_{published_agent.id}" in tool_names
            assert f"agent_{draft_agent.id}" in tool_names

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_generated_workforce_managers_are_hidden_from_agent_tools(self) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(
                username="testuser_generated_manager_tools",
                password_hash="x",
                is_admin=False,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            reusable_agent = Agent(
                user_id=user.id,
                name="Reusable Agent",
                status=AgentStatus.PUBLISHED,
            )
            generated_manager = Agent(
                user_id=user.id,
                name="Generated Manager",
                status=AgentStatus.PUBLISHED,
                origin=AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
            )
            db.add_all([reusable_agent, generated_manager])
            db.commit()
            db.refresh(reusable_agent)
            db.refresh(generated_manager)

            tools = get_published_agents_tools(
                session_factory=SessionLocal, user_id=user.id
            )
            tool_names = {tool.name for tool in tools}

            assert f"agent_{reusable_agent.id}" in tool_names
            assert f"agent_{generated_manager.id}" not in tool_names

            explicitly_allowed_tools = get_published_agents_tools(
                session_factory=SessionLocal,
                user_id=user.id,
                allowed_agent_ids=[generated_manager.id],
            )

            assert explicitly_allowed_tools == []
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_user_isolation_for_draft_agents(self) -> None:
        """Test that users cannot see other users' draft agents."""
        db, db_path, SessionLocal = _create_session()
        try:
            user1 = User(username="user1", password_hash="x", is_admin=False)
            user2 = User(username="user2", password_hash="x", is_admin=False)
            db.add_all([user1, user2])
            db.commit()
            db.refresh(user1)
            db.refresh(user2)

            # User1's draft agent
            draft_agent = Agent(
                user_id=user1.id,
                name="User1 Draft",
                status=AgentStatus.DRAFT,
            )
            db.add(draft_agent)
            db.commit()

            # User2 should not see User1's draft agent
            tools_for_user2 = get_published_agents_tools(
                session_factory=SessionLocal, user_id=user2.id, include_draft=True
            )
            tool_names = {tool.name for tool in tools_for_user2}

            assert f"agent_{draft_agent.id}" not in tool_names

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass


class TestCreateAndCallAgent:
    """Integration test for creating and calling an agent."""

    @pytest.mark.asyncio
    async def test_create_then_call_draft_agent(self) -> None:
        """Test creating a draft agent and then calling it."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser8", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = user.id
            db.close()

            # Mock LLM
            mock_llm = Mock()
            mock_llm.model_id = "gpt-4"
            mock_llm.chat = Mock(return_value="Test response")

            with patch(
                "xagent.web.services.llm_utils.UserAwareModelStorage"
            ) as mock_storage_class:
                mock_storage = Mock()
                mock_storage.get_configured_defaults.return_value = (
                    mock_llm,
                    None,
                    None,
                    None,
                )
                mock_storage.get_llm_by_name_with_access.return_value = mock_llm
                mock_storage_class.return_value = mock_storage

                # Step 1: Create agent
                create_tool = CreateAgentTool(
                    session_factory=SessionLocal, user_id=user_id, task_id="test_task"
                )

                create_result = await create_tool.run_json_async(
                    {
                        "name": "simple_calculator",
                        "description": "A simple calculator for basic math operations",
                        "instructions": "You are a calculator. Return the result.",
                    }
                )

                assert create_result["status"] == "success"
                agent_id = create_result["agent_id"]

                # Step 2: Verify agent is in tools list (use a fresh session)
                verify_db = SessionLocal()
                try:
                    tools = get_published_agents_tools(
                        session_factory=SessionLocal,
                        user_id=user_id,
                        include_draft=True,
                    )
                    tool_names = {tool.name for tool in tools}
                    assert f"agent_{agent_id}" in tool_names

                    # Step 3: Verify agent can be loaded
                    agent = verify_db.query(Agent).filter(Agent.id == agent_id).first()
                    assert agent is not None
                    assert agent.status == AgentStatus.DRAFT
                finally:
                    verify_db.close()

        finally:
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_agent_tool_injects_langfuse_tracer(
        self, mocker, monkeypatch, langfuse_client_reset
    ) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
            monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
            create_langfuse_mock(mocker)

            user = User(username="testuser9", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            model = Model(
                model_id="test-model-id",
                category="llm",
                model_provider="openai",
                model_name="gpt-4",
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                temperature=0.7,
                abilities=["chat"],
            )
            db.add(model)
            db.commit()
            db.refresh(model)

            agent = Agent(
                user_id=user.id,
                name="Delegated Agent",
                description="Nested agent",
                instructions="You are delegated.",
                status=AgentStatus.PUBLISHED,
                models={"general": model.id},
            )
            db.add(agent)
            db.commit()
            db.refresh(agent)

            tool = AgentTool(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_description=agent.description or "",
                session_factory=SessionLocal,
                user_id=user.id,
                task_id="parent-task-1",
            )

            with (
                patch(
                    "xagent.web.services.llm_utils.UserAwareModelStorage"
                ) as mock_storage_class,
                patch(
                    "xagent.core.agent.service.AgentService"
                ) as mock_agent_service_class,
                patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
            ):
                mock_storage = Mock()
                mock_llm = Mock()
                mock_storage.get_llm_by_name_with_access.return_value = mock_llm
                mock_storage_class.return_value = mock_storage

                mock_agent_service = mock_agent_service_class.return_value
                mock_agent_service.execute_task = AsyncMock(
                    return_value={"output": "nested response"}
                )

                result = await tool.run_json_async({"task": "do nested work"})

            assert result["response"] == "nested response"
            tracer = mock_agent_service_class.call_args.kwargs["tracer"]
            assert any(
                isinstance(handler, LangfuseTraceHandler) for handler in tracer.handlers
            )
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_agent_tool_keeps_mcp_tools_when_filtering_by_category(self) -> None:
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="testuser10", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            model = Model(
                model_id="test-model-id",
                category="llm",
                model_provider="openai",
                model_name="gpt-4",
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                temperature=0.7,
                abilities=["chat"],
            )
            db.add(model)
            db.commit()
            db.refresh(model)

            agent = Agent(
                user_id=user.id,
                name="LinkedIn Assistant",
                description="Nested MCP agent",
                instructions="You are delegated.",
                status=AgentStatus.PUBLISHED,
                models={"general": model.id},
                tool_categories=["mcp:LinkedIn"],
            )
            db.add(agent)
            db.commit()
            db.refresh(agent)

            tool = AgentTool(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_description=agent.description or "",
                session_factory=SessionLocal,
                user_id=user.id,
                task_id="parent-task-mcp",
            )

            with (
                patch(
                    "xagent.web.services.llm_utils.UserAwareModelStorage"
                ) as mock_storage_class,
                patch(
                    "xagent.core.agent.service.AgentService"
                ) as mock_agent_service_class,
                patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
            ):
                mock_storage = Mock()
                mock_llm = Mock()
                mock_storage.get_llm_by_name_with_access.return_value = mock_llm
                mock_storage_class.return_value = mock_storage

                mock_agent_service = mock_agent_service_class.return_value
                mock_agent_service.execute_task = AsyncMock(
                    return_value={"output": "nested response"}
                )

                result = await tool.run_json_async({"task": "get linkedin profile"})

            # The delegation path now hands a typed ToolSelectionSpec to
            # WebToolConfig (rather than a pre-computed allowed_tools
            # list). The factory's spec.compute_allowed_names does the
            # name-level filter at build time. Pin the spec shape here
            # so a regression that drops mcp:<server> derivation surfaces.
            from xagent.core.tools.adapters.vibe.selection_spec import (
                _SpecByCategories,
            )

            assert result["response"] == "nested response"
            tool_config = mock_agent_service_class.call_args.kwargs["tool_config"]
            spec = tool_config.get_tool_selection_spec()
            assert isinstance(spec, _SpecByCategories), (
                "mcp:LinkedIn tool_categories must produce a BY_CATEGORIES "
                "spec, not ALL/NONE -- otherwise the factory's name filter "
                "won't restrict to LinkedIn MCP tools."
            )
            # Orthogonal model: the mcp:<server> scope lands in
            # mcp_servers only, not in categories (no raw string leak).
            assert "mcp:LinkedIn" not in spec.categories
            assert "mcp" not in spec.categories
            # mcp_servers holds the normalized server key (SSOT lower-cases /
            # folds), matched at build time against each tool's
            # metadata.source_server.
            assert spec.mcp_servers == frozenset({"linkedin"})
            # Delegated WebToolConfig uses the shared MCP config-loading
            # helper instead of falling back to the default True. An
            # mcp:<server> selection -> include_mcp_tools True.
            assert tool_config._include_mcp_tools is True
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.parametrize(
        "tool_categories",
        [["basic"], [], None],
        ids=["basic-only", "empty-list", "null"],
    )
    async def test_delegated_non_mcp_agent_disables_mcp_init(
        self, tool_categories: list[str] | None
    ) -> None:
        """A delegated agent that did not select MCP (``tool_categories``
        without ``mcp``) must build its WebToolConfig with
        ``include_mcp_tools=False`` -- so it does not pay MCP server init.
        NULL categories build an ALL-mode spec (legacy unconfigured agent).
        Empty list ``[]`` builds a NONE-mode spec (explicitly zero tools).
        Neither opts into MCP config loading."""
        db, db_path, SessionLocal = _create_session()
        try:
            user = User(username="basicdeleg", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            model = Model(
                model_id="test-model-id",
                category="llm",
                model_provider="openai",
                model_name="gpt-4",
                api_key="test-api-key",
                base_url="https://api.openai.com/v1",
                temperature=0.7,
                abilities=["chat"],
            )
            db.add(model)
            db.commit()
            db.refresh(model)

            agent = Agent(
                user_id=user.id,
                name="Basic Assistant",
                description="Nested non-MCP agent",
                instructions="You are delegated.",
                status=AgentStatus.PUBLISHED,
                models={"general": model.id},
                tool_categories=tool_categories,
            )
            db.add(agent)
            db.commit()
            db.refresh(agent)

            tool = AgentTool(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_description=agent.description or "",
                session_factory=SessionLocal,
                user_id=user.id,
                task_id="parent-task-basic",
            )

            with (
                patch(
                    "xagent.web.services.llm_utils.UserAwareModelStorage"
                ) as mock_storage_class,
                patch(
                    "xagent.core.agent.service.AgentService"
                ) as mock_agent_service_class,
                patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
            ):
                mock_storage = Mock()
                mock_llm = Mock()
                mock_storage.get_llm_by_name_with_access.return_value = mock_llm
                mock_storage_class.return_value = mock_storage

                mock_agent_service = mock_agent_service_class.return_value
                mock_agent_service.execute_task = AsyncMock(
                    return_value={"output": "nested response"}
                )

                await tool.run_json_async({"task": "do basic work"})

            tool_config = mock_agent_service_class.call_args.kwargs["tool_config"]
            spec = tool_config.get_tool_selection_spec()
            if tool_categories:
                # non-empty list: scoped, MCP not selected
                assert spec.includes_mcp() is False
            elif tool_categories is None:
                # None → _SpecAll (legacy unconfigured agent, all tools)
                assert spec.is_all()
                assert spec.includes_mcp() is True
            else:
                # [] → _SpecNone (explicitly zero tools)
                assert spec.is_none()
                assert spec.includes_mcp() is False
            assert tool_config._include_mcp_tools is False
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

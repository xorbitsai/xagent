"""Tests for CreateAgentTool - dynamically creating agents during task execution."""

import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.utils.mock_helpers import create_langfuse_mock
from xagent.core.tools.adapters.vibe.agent_tool import (
    AgentTool,
    CreateAgentTool,
    ListAgentsTool,
    UpdateAgentTool,
    gen_agent_tool_name,
    get_published_agents_tools,
)
from xagent.core.tracing.langfuse.handler import LangfuseTraceHandler
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User


def _create_session() -> tuple[Session, str]:
    """Create a temporary database session for testing."""
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    db_url = f"sqlite:///{temp_db.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal(), temp_db.name


class TestCreateAgentTool:
    """Test suite for CreateAgentTool."""

    @pytest.mark.asyncio
    async def test_create_agent_success(self) -> None:
        """Test successful agent creation."""
        db, db_path = _create_session()
        try:
            # Create test user
            user = User(username="testuser", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

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
                tool = CreateAgentTool(db=db, user_id=user.id, task_id="test_task")

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
                assert result["tool_name"] == "call_agent_test_agent"
                assert "test_agent" in result["markdown_link"]
                assert "agent://" in result["markdown_link"]
                mock_invalidate_agent_cache.assert_called_once_with(
                    user.id, result["agent_id"]
                )

                # Verify agent was created in database
                agent = (
                    db.query(Agent)
                    .filter(Agent.name == "test_agent", Agent.user_id == user.id)
                    .first()
                )
                assert agent is not None
                assert agent.status == AgentStatus.DRAFT
                assert agent.instructions == "You are a test agent for unit testing."

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
        db, db_path = _create_session()
        try:
            user = User(username="testuser2", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

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

                tool = CreateAgentTool(db=db, user_id=user.id)

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

                # Verify filters were saved
                agent = db.query(Agent).filter(Agent.name == "filtered_agent").first()
                assert agent is not None
                assert agent.tool_categories == ["file", "knowledge"]
                assert agent.skills == ["web_search"]

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_duplicate_name_auto_renames(self) -> None:
        """Test that duplicate agent names are auto-renamed and created."""
        db, db_path = _create_session()
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

                tool = CreateAgentTool(db=db, user_id=user.id)

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

                created_agent = (
                    db.query(Agent)
                    .filter(
                        Agent.user_id == user.id,
                        Agent.name == "duplicate_name Assistant",
                    )
                    .first()
                )
                assert created_agent is not None

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_rejects_missing_knowledge_base(self) -> None:
        """Test that create_agent rejects knowledge bases that do not exist."""
        db, db_path = _create_session()
        try:
            user = User(username="testuser_missing_kb", password_hash="x")
            db.add(user)
            db.commit()
            db.refresh(user)

            tool = CreateAgentTool(db=db, user_id=user.id)

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
            assert db.query(Agent).filter(Agent.name == "kb_agent").first() is None

        finally:
            db.close()
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
        db, db_path = _create_session()
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

                tool = CreateAgentTool(db=db, user_id=user.id)

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
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_missing_name(self) -> None:
        """Test that missing name returns error."""
        db, db_path = _create_session()
        try:
            user = User(username="testuser4", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            tool = CreateAgentTool(db=db, user_id=user.id)

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
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_create_agent_missing_instructions(self) -> None:
        """Test that missing instructions returns error."""
        db, db_path = _create_session()
        try:
            user = User(username="testuser5", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            tool = CreateAgentTool(db=db, user_id=user.id)

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
            db.close()
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
        db, db_path = _create_session()
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
                tool = UpdateAgentTool(db=db, user_id=user.id, task_id="test_task")

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
        db, db_path = _create_session()
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

            tool = UpdateAgentTool(db=db, user_id=user.id)

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
        db, db_path = _create_session()
        try:
            user = User(username="testuser_notfound", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            tool = UpdateAgentTool(db=db, user_id=user.id)

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
    async def test_update_agent_rejects_missing_knowledge_base(self) -> None:
        """Test that update_agent rejects knowledge bases that do not exist."""
        db, db_path = _create_session()
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

            tool = UpdateAgentTool(db=db, user_id=user.id)

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
        db, db_path = _create_session()
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

            tool = UpdateAgentTool(db=db, user_id=user.id)

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
        db, db_path = _create_session()
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

            tool = UpdateAgentTool(db=db, user_id=user.id)

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
        db, db_path = _create_session()
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

            tool = UpdateAgentTool(db=db, user_id=user.id)

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


class TestListAgentsTool:
    """Test suite for ListAgentsTool."""

    @pytest.mark.asyncio
    async def test_list_all_agents(self) -> None:
        """Test listing all agents."""
        db, db_path = _create_session()
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

            tool = ListAgentsTool(db=db, user_id=user.id)

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
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_list_agents_with_status_filter(self) -> None:
        """Test listing agents with status filter."""
        db, db_path = _create_session()
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

            tool = ListAgentsTool(db=db, user_id=user.id)

            # List only draft agents
            result = await tool.run_json_async({"status_filter": "draft"})

            assert result["status"] == "success"
            assert result["total_count"] == 1
            assert result["agents"][0]["name"] == "draft_agent"
            assert result["agents"][0]["status"] == "draft"

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_list_agents_user_isolation(self) -> None:
        """Test that users can only see their own agents."""
        db, db_path = _create_session()
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

            # User1 should only see their own agents
            tool = ListAgentsTool(db=db, user_id=user1.id)
            result = await tool.run_json_async({})

            assert result["status"] == "success"
            assert result["total_count"] == 1
            assert result["agents"][0]["name"] == "user1_agent"

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_list_agents_invalid_status_filter(self) -> None:
        """Test that invalid status filter returns error."""
        db, db_path = _create_session()
        try:
            user = User(username="testuser_invalid", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

            tool = ListAgentsTool(db=db, user_id=user.id)

            result = await tool.run_json_async({"status_filter": "invalid_status"})

            assert result["status"] == "error"
            assert "invalid" in result["message"].lower()

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass


class TestAgentToolNameGeneration:
    """Test suite for agent tool name generation."""

    def test_gen_agent_tool_name_simple(self) -> None:
        """Test tool name generation with simple name."""
        result = gen_agent_tool_name("TestAgent")
        assert result == "call_agent_testagent"  # No spaces, just lowercased

    def test_gen_agent_tool_name_with_spaces(self) -> None:
        """Test tool name generation with spaces."""
        result = gen_agent_tool_name("Research Assistant")
        assert result == "call_agent_research_assistant"

    def test_gen_agent_tool_name_with_special_chars(self) -> None:
        """Test tool name generation with special characters."""
        result = gen_agent_tool_name("AI-Research-Agent_2024")
        assert result == "call_agent_ai-research-agent_2024"


class TestDraftAgentsInTools:
    """Test suite for including draft agents in tool lists."""

    def test_get_tools_with_draft_disabled(self) -> None:
        """Test that draft agents are excluded when include_draft=False."""
        db, db_path = _create_session()
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
                db=db, user_id=user.id, include_draft=False
            )
            tool_names = {tool.name for tool in tools}

            assert "call_agent_published_agent" in tool_names
            assert "call_agent_draft_agent" not in tool_names

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_get_tools_with_draft_enabled(self) -> None:
        """Test that draft agents are included when include_draft=True."""
        db, db_path = _create_session()
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
                db=db, user_id=user.id, include_draft=True
            )
            tool_names = {tool.name for tool in tools}

            assert "call_agent_published_agent" in tool_names
            assert "call_agent_draft_agent" in tool_names

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    def test_user_isolation_for_draft_agents(self) -> None:
        """Test that users cannot see other users' draft agents."""
        db, db_path = _create_session()
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
                db=db, user_id=user2.id, include_draft=True
            )
            tool_names = {tool.name for tool in tools_for_user2}

            assert "call_agent_user1_draft" not in tool_names

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
        db, db_path = _create_session()
        try:
            user = User(username="testuser8", password_hash="x", is_admin=False)
            db.add(user)
            db.commit()
            db.refresh(user)

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
                    db=db, user_id=user.id, task_id="test_task"
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

                # Step 2: Verify agent is in tools list
                tools = get_published_agents_tools(
                    db=db, user_id=user.id, include_draft=True
                )
                tool_names = {tool.name for tool in tools}

                assert "call_agent_simple_calculator" in tool_names

                # Step 3: Verify agent can be loaded
                agent = db.query(Agent).filter(Agent.id == agent_id).first()
                assert agent is not None
                assert agent.status == AgentStatus.DRAFT

        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_agent_tool_injects_langfuse_tracer(
        self, mocker, monkeypatch, langfuse_client_reset
    ) -> None:
        db, db_path = _create_session()
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
                db=db,
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
        db, db_path = _create_session()
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
                db=db,
                user_id=user.id,
                task_id="parent-task-mcp",
            )

            fake_mcp_tool = Mock()
            fake_mcp_tool.name = "mcp_LinkedIn_get_profile"
            fake_mcp_tool.metadata = Mock()
            fake_mcp_tool.metadata.category = Mock()
            fake_mcp_tool.metadata.category.value = "mcp"

            with (
                patch(
                    "xagent.web.services.llm_utils.UserAwareModelStorage"
                ) as mock_storage_class,
                patch(
                    "xagent.core.agent.service.AgentService"
                ) as mock_agent_service_class,
                patch(
                    "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
                    new=AsyncMock(return_value=[fake_mcp_tool]),
                ),
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

            assert result["response"] == "nested response"
            tool_config = mock_agent_service_class.call_args.kwargs["tool_config"]
            assert tool_config.get_allowed_tools() == ["mcp_LinkedIn_get_profile"]
        finally:
            db.close()
            try:
                import os

                os.remove(db_path)
            except OSError:
                pass

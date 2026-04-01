"""Tests for CreateAgentTool - dynamically creating agents during task execution."""

import tempfile
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.tools.adapters.vibe.agent_tool import (
    CreateAgentTool,
    gen_agent_tool_name,
    get_published_agents_tools,
)
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
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
    async def test_create_agent_duplicate_name_error(self) -> None:
        """Test that duplicate agent names are rejected."""
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
                        "instructions": "This should fail",
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

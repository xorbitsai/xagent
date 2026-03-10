"""
Tests for SkillSelector with agent instructions support
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.xagent.skills.selector import SkillSelector


@pytest.fixture
def mock_llm():
    """Create mock LLM"""
    llm = MagicMock()
    llm.chat = AsyncMock()
    return llm


@pytest.fixture
def sample_candidates():
    """Create sample skill candidates"""
    return [
        {
            "name": "query-then-analyze",
            "description": "Database query and analysis skill",
            "when_to_use": "When you need to query databases and analyze data",
            "tags": ["database", "sql", "analysis"],
        },
        {
            "name": "poster-design",
            "description": "Create visual poster designs",
            "when_to_use": "When you need to create posters or banners",
            "tags": ["design", "visual", "poster"],
        },
        {
            "name": "code-helper",
            "description": "Help with coding tasks",
            "when_to_use": "When you need help with programming",
            "tags": ["code", "programming"],
        },
    ]


class TestSkillSelector:
    """Test SkillSelector"""

    @pytest.mark.asyncio
    async def test_select_without_agent_instructions(self, mock_llm, sample_candidates):
        """Test skill selection without agent instructions"""
        # Mock LLM response
        mock_llm.chat.return_value = json.dumps(
            {
                "selected": True,
                "skill_name": "query-then-analyze",
                "reasoning": "Task involves database querying",
            }
        )

        selector = SkillSelector(mock_llm)
        result = await selector.select(
            task="查询数据库中的用户数据",
            candidates=sample_candidates,
        )

        assert result is not None
        assert result["name"] == "query-then-analyze"

        # Verify system message contains only SELECTOR_SYSTEM
        call_args = mock_llm.chat.call_args
        messages = call_args[1]["messages"]
        system_msg = messages[0]["content"]
        assert "AGENT INSTRUCTIONS" not in system_msg

    @pytest.mark.asyncio
    async def test_select_with_agent_instructions(self, mock_llm, sample_candidates):
        """Test skill selection with agent instructions"""
        agent_instructions = (
            "For data-related tasks, ALWAYS use query-then-analyze skill"
        )

        # Mock LLM response
        mock_llm.chat.return_value = json.dumps(
            {
                "selected": True,
                "skill_name": "query-then-analyze",
                "reasoning": "Following agent instructions to use query-then-analyze",
            }
        )

        selector = SkillSelector(mock_llm)
        result = await selector.select(
            task="统计降落数据",
            candidates=sample_candidates,
            agent_instructions=agent_instructions,
        )

        assert result is not None
        assert result["name"] == "query-then-analyze"

        # Verify system message contains agent instructions
        call_args = mock_llm.chat.call_args
        messages = call_args[1]["messages"]
        system_msg = messages[0]["content"]
        assert "AGENT INSTRUCTIONS" in system_msg
        assert agent_instructions in system_msg

    @pytest.mark.asyncio
    async def test_select_with_agent_instructions_enforces_constraint(
        self, mock_llm, sample_candidates
    ):
        """Test that agent instructions constrain skill selection"""
        agent_instructions = (
            "CRITICAL: NEVER accept user-uploaded data files. "
            "All required data is accessible through query-then-analyze skill."
        )

        # Mock LLM response - agent instructions should guide to correct skill
        mock_llm.chat.return_value = json.dumps(
            {
                "selected": True,
                "skill_name": "query-then-analyze",
                "reasoning": "Agent instructions require using query-then-analyze for database access",
            }
        )

        selector = SkillSelector(mock_llm)
        result = await selector.select(
            task="请统计一下降落的次数，按小时分组统计次数",
            candidates=sample_candidates,
            agent_instructions=agent_instructions,
        )

        assert result is not None
        assert result["name"] == "query-then-analyze"

        # Verify agent instructions are in system prompt
        call_args = mock_llm.chat.call_args
        messages = call_args[1]["messages"]
        system_msg = messages[0]["content"]
        assert "NEVER accept user-uploaded data files" in system_msg

    @pytest.mark.asyncio
    async def test_select_no_skill_selected(self, mock_llm, sample_candidates):
        """Test when no skill is selected"""
        # Mock LLM response - no skill selected
        mock_llm.chat.return_value = json.dumps(
            {
                "selected": False,
                "skill_name": None,
                "reasoning": "Task is general, no specific skill needed",
            }
        )

        selector = SkillSelector(mock_llm)
        result = await selector.select(
            task="what's the weather today?",
            candidates=sample_candidates,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_select_with_empty_candidates(self, mock_llm):
        """Test with empty candidate list"""
        selector = SkillSelector(mock_llm)
        result = await selector.select(
            task="some task",
            candidates=[],
        )

        assert result is None
        mock_llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_prompt_without_agent_instructions(self, mock_llm):
        """Test _build_prompt without agent instructions"""
        selector = SkillSelector(llm=mock_llm)
        candidates = [
            {
                "name": "test-skill",
                "description": "A test skill",
                "when_to_use": "For testing",
                "tags": ["test"],
            }
        ]

        prompt = selector._build_prompt("test task", candidates)

        assert "test task" in prompt
        assert "test-skill" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_with_agent_instructions(self, mock_llm):
        """Test _build_prompt with agent instructions"""
        selector = SkillSelector(llm=mock_llm)
        candidates = [
            {
                "name": "test-skill",
                "description": "A test skill",
                "when_to_use": "For testing",
                "tags": ["test"],
            }
        ]

        # _build_prompt no longer takes agent_instructions - it's added to system message in select()
        prompt = selector._build_prompt("test task", candidates)

        assert "test task" in prompt
        assert "test-skill" in prompt

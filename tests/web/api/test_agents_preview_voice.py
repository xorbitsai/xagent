"""Agent-preview endpoint must apply the current user's onboarding voice
preference to the previewed system prompt, the same way a saved agent's
real conversation does (see apply_user_voice's call sites in chat.py).

Before this fix, ``preview_agent`` called ``enhance_system_prompt_with_kb``
but never ``apply_user_voice`` - a user testing an agent draft in the
builder's preview panel would see their chosen voice silently not applied,
even though the same agent, once saved, would honor it in a real chat.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api.agents import AgentPreviewRequest, preview_agent
from xagent.web.models.user import User


def _make_user(preferences: dict | None = None) -> User:
    user = User()
    user.id = 7
    user.is_admin = False
    user.preferences = preferences
    return user


async def _run_preview(current_user: User) -> str | None:
    """Run preview_agent with a stubbed LLM/AgentService and return the
    system_prompt it was constructed with."""
    db = MagicMock()
    model_record = MagicMock()
    model_record.model_id = "test-model"
    db.query.return_value.filter.return_value.first.return_value = model_record

    request = AgentPreviewRequest(
        instructions="Base instructions.",
        execution_mode="balanced",
        models={"general": 1},
        knowledge_bases=[],
        skills=[],
        tool_categories=["basic"],
        message="hello",
    )

    with (
        patch("xagent.web.api.agents.UserAwareModelStorage") as mock_storage_class,
        patch("xagent.web.api.agents.InMemoryMemoryStore"),
        patch("xagent.web.api.agents.AgentService") as mock_agent_service_class,
    ):
        mock_storage = MagicMock()
        mock_storage.get_llm_by_name_with_access.return_value = MagicMock()
        mock_storage_class.return_value = mock_storage

        mock_agent_service = mock_agent_service_class.return_value
        mock_agent_service.execute_task = AsyncMock(
            return_value={"output": "preview response", "status": "completed"}
        )

        await preview_agent(request=request, current_user=current_user, db=db)

    return mock_agent_service_class.call_args.kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_preview_appends_output_voice_section_for_a_known_voice():
    system_prompt = await _run_preview(_make_user({"voice": "concise"}))

    assert system_prompt.startswith("Base instructions.\n\n## OUTPUT VOICE\n")
    assert "As short as possible" in system_prompt


@pytest.mark.asyncio
async def test_preview_leaves_prompt_unchanged_without_a_voice_preference():
    system_prompt = await _run_preview(_make_user(None))

    assert system_prompt == "Base instructions."


@pytest.mark.asyncio
async def test_preview_leaves_prompt_unchanged_for_an_unrecognized_voice():
    system_prompt = await _run_preview(_make_user({"voice": "sarcastic"}))

    assert system_prompt == "Base instructions."


@pytest.mark.asyncio
async def test_preview_threads_voice_into_tool_config_for_delegated_agents():
    """A delegated AgentTool a previewed agent calls must also honor the
    previewing user's voice (see BaseToolConfig.get_voice), not just the
    preview's own top-level system prompt."""
    current_user = _make_user({"voice": "friendly"})
    db = MagicMock()
    model_record = MagicMock()
    model_record.model_id = "test-model"
    db.query.return_value.filter.return_value.first.return_value = model_record

    request = AgentPreviewRequest(
        instructions="Base instructions.",
        execution_mode="balanced",
        models={"general": 1},
        knowledge_bases=[],
        skills=[],
        tool_categories=["basic"],
        message="hello",
    )

    with (
        patch("xagent.web.api.agents.UserAwareModelStorage") as mock_storage_class,
        patch("xagent.web.api.agents.InMemoryMemoryStore"),
        patch("xagent.web.api.agents.AgentService") as mock_agent_service_class,
        patch("xagent.web.api.agents.WebToolConfig") as mock_web_tool_config_class,
    ):
        mock_storage = MagicMock()
        mock_storage.get_llm_by_name_with_access.return_value = MagicMock()
        mock_storage_class.return_value = mock_storage

        mock_agent_service = mock_agent_service_class.return_value
        mock_agent_service.execute_task = AsyncMock(
            return_value={"output": "preview response", "status": "completed"}
        )

        await preview_agent(request=request, current_user=current_user, db=db)

    assert mock_web_tool_config_class.call_args.kwargs["voice"] == "friendly"

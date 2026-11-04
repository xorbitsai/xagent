"""REST agent-preview endpoint must scope tools to the request's
``tool_categories`` (issues #798 / #117).

Before the fix, ``preview_agent`` built its ``WebToolConfig`` without a
``ToolSelectionSpec`` (and with the default ``include_mcp_tools=True``),
so a preview loaded every Custom API / MCP server the *user* had
configured — regardless of what the previewed agent actually selected.
These tests pin the spec handed to ``WebToolConfig`` for each
``tool_categories`` shape, mirroring the runtime chat-path semantics.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api.agents import AgentPreviewRequest, preview_agent
from xagent.web.models.user import User


def _make_user() -> User:
    user = User()
    user.id = 7
    user.is_admin = False
    return user


async def _run_preview(tool_categories):
    """Run preview_agent with a stubbed LLM/AgentService and return the
    WebToolConfig it was constructed with."""
    current_user = _make_user()

    db = MagicMock()
    model_record = MagicMock()
    model_record.model_id = "test-model"
    db.query.return_value.filter.return_value.first.return_value = model_record

    request = AgentPreviewRequest(
        instructions="preview instructions",
        execution_mode="balanced",
        models={"general": 1},
        knowledge_bases=[],
        skills=[],
        tool_categories=tool_categories,
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

    return mock_agent_service_class.call_args.kwargs["tool_config"]


@pytest.mark.asyncio
async def test_preview_scoped_categories_exclude_custom_api_and_mcp():
    """A preview for an agent that selected only ``basic`` must not load
    the user's Custom API registry nor initialize MCP servers."""
    tool_config = await _run_preview(["basic"])

    spec = tool_config.get_tool_selection_spec()
    assert spec.is_by_categories()
    assert spec.includes_category("basic") is True
    assert spec.includes_custom_api() is False
    assert spec.includes_mcp() is False
    assert tool_config._include_mcp_tools is False


@pytest.mark.asyncio
async def test_preview_mcp_server_scope_loads_only_that_connector():
    """``mcp:<server>`` selection opts into MCP config loading and admits
    the matching Custom API wrapper only."""
    tool_config = await _run_preview(["mcp:LinkedIn"])

    spec = tool_config.get_tool_selection_spec()
    assert spec.is_by_categories()
    assert spec.mcp_servers == frozenset({"linkedin"})
    assert spec.includes_custom_api() is True
    assert tool_config._include_mcp_tools is True


@pytest.mark.asyncio
async def test_preview_empty_categories_yield_zero_tools():
    """Explicit ``[]`` mirrors a saved agent with zero tools selected."""
    tool_config = await _run_preview([])

    spec = tool_config.get_tool_selection_spec()
    assert spec.is_none()
    assert spec.includes_custom_api() is False
    assert tool_config._include_mcp_tools is False


@pytest.mark.asyncio
async def test_preview_omitted_categories_keep_builtins_but_not_custom_apis():
    """Field omitted (None, legacy "unconfigured") keeps the full built-in
    tool set, but must NOT bulk-load the user-level Custom API registry —
    the same opt-out the delegated-agent path applies (#798 / #117)."""
    tool_config = await _run_preview(None)

    spec = tool_config.get_tool_selection_spec()
    assert spec.is_all()
    assert spec.includes_custom_api() is False
    # Legacy ALL mode does not pay MCP server init on the preview path.
    assert tool_config._include_mcp_tools is False

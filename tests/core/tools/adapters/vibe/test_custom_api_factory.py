from unittest.mock import MagicMock

import pytest

from xagent.core.tools.adapters.vibe.config import BaseToolConfig
from xagent.core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)
from xagent.core.tools.adapters.vibe.custom_api_factory import (
    create_db_custom_api_tools,
)
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec


@pytest.mark.asyncio
async def test_create_db_custom_api_tools_no_user():
    config = MagicMock(spec=BaseToolConfig)
    config.get_user_id.return_value = None

    tools = await create_db_custom_api_tools(config)
    assert tools == []


@pytest.mark.asyncio
async def test_create_db_custom_api_tools_no_configs():
    config = MagicMock(spec=BaseToolConfig)
    config.get_user_id.return_value = 1
    config.get_custom_api_configs.return_value = []

    tools = await create_db_custom_api_tools(config)
    assert tools == []


@pytest.mark.asyncio
async def test_create_db_custom_api_tools_other_category_does_not_load_configs():
    config = MagicMock()
    config.get_tool_selection_spec.return_value = ToolSelectionSpec.from_raw(
        tool_categories=["other"]
    )
    config.get_user_id.return_value = 1
    config.get_custom_api_configs.return_value = []

    tools = await create_db_custom_api_tools(config)

    assert tools == []
    config.get_custom_api_configs.assert_not_called()


@pytest.mark.asyncio
async def test_create_db_custom_api_tools_unconfigured_delegated_spec_skips_db():
    """Issue #798: a delegated workforce worker with NULL tool_categories
    builds an ALL-mode spec that opts out of the Custom API creator, so
    the user-level Custom API registry is never enumerated."""
    config = MagicMock()
    config.get_tool_selection_spec.return_value = ToolSelectionSpec.from_raw(
        tool_categories=None,
        exclude_custom_api_when_unconfigured=True,
    )
    config.get_user_id.return_value = 1

    tools = await create_db_custom_api_tools(config)

    assert tools == []
    config.get_custom_api_configs.assert_not_called()


@pytest.mark.asyncio
async def test_create_db_custom_api_tools_with_configs():
    config = MagicMock(spec=BaseToolConfig)
    config.get_user_id.return_value = 1
    config.get_custom_api_configs.return_value = [
        {
            "name": "api1",
            "description": "desc1",
            "url": "https://api.example.com/api1",
            "method": "POST",
            "headers": {"X-Key": "$k1"},
            "body": '{"message": "hello"}',
            "env": {"k1": "v1"},
        }
    ]

    tools = await create_db_custom_api_tools(config)
    assert len(tools) == 1
    assert tools[0].name == "api_api1_call"
    assert "Configured endpoint: https://api.example.com/api1" in tools[0].description
    # Body template must thread through to the tool so POST requests
    # actually carry the configured payload at runtime.
    assert tools[0]._default_body == '{"message": "hello"}'
    assert "Configured body template:" in tools[0].description


@pytest.mark.asyncio
async def test_create_db_custom_api_tools_exception():
    config = MagicMock(spec=BaseToolConfig)
    config.get_user_id.side_effect = Exception("Test Error")

    tools = await create_db_custom_api_tools(config)
    assert tools == []


@pytest.mark.asyncio
async def test_create_db_custom_api_tools_propagates_connector_runtime_error():
    config = MagicMock(spec=BaseToolConfig)
    config.get_user_id.return_value = 1
    config.get_custom_api_configs.side_effect = ConnectorRuntimeError(
        ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
        "Connector runtime context is unavailable.",
        details={"reason": "runtime_view_resolution_failed"},
        status_code=503,
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        await create_db_custom_api_tools(config)

    assert exc_info.value.code == ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
    assert exc_info.value.status_code == 503

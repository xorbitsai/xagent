from unittest.mock import MagicMock

import pytest

from xagent.core.tools.adapters.vibe.config import BaseToolConfig
from xagent.core.tools.adapters.vibe.custom_api_factory import (
    create_db_custom_api_tools,
)


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
async def test_create_db_custom_api_tools_with_configs():
    config = MagicMock(spec=BaseToolConfig)
    config.get_user_id.return_value = 1
    config.get_custom_api_configs.return_value = [
        {"name": "api1", "description": "desc1", "env": {"k1": "v1"}}
    ]

    tools = await create_db_custom_api_tools(config)
    assert len(tools) == 1
    assert tools[0].name == "api_api1_call"


@pytest.mark.asyncio
async def test_create_db_custom_api_tools_exception():
    config = MagicMock(spec=BaseToolConfig)
    config.get_user_id.side_effect = Exception("Test Error")

    tools = await create_db_custom_api_tools(config)
    assert tools == []

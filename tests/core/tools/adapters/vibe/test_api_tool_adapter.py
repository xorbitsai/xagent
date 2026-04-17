from unittest.mock import patch

import pytest

from xagent.core.tools.adapters.vibe.api_tool_adapter import (
    CustomApiTool,
    create_custom_api_tools,
)


def test_custom_api_tool_init():
    tool = CustomApiTool(
        name="my-test api",
        description="A test API",
        env={"API_KEY": "secret123", "API_KEY_BACKUP": "secret456"},
    )

    assert tool.name == "api_my_test_api_call"
    assert "A test API" in tool.description
    assert "- API_KEY" in tool.description
    assert "- API_KEY_BACKUP" in tool.description


def test_custom_api_tool_replace_secrets():
    # Use unencrypted secrets for simplicity since decrypt_value handles unencrypted fallback or we can mock it
    # We will mock decrypt_value to just return the value for testing replace
    with patch(
        "xagent.core.tools.adapters.vibe.api_tool_adapter.decrypt_value",
        side_effect=lambda x: x,
    ):
        tool = CustomApiTool(
            name="test",
            description="test",
            env={"API_KEY": "secret123", "API_KEY_BACKUP": "secret456"},
        )

        # Test word boundaries
        result = tool._replace_secrets("Bearer $API_KEY")
        assert result == "Bearer secret123"

        # Test word boundaries avoiding partial replacement
        result2 = tool._replace_secrets("Bearer $API_KEY_BACKUP")
        assert result2 == "Bearer secret456"

        # Test bracket notation
        result3 = tool._replace_secrets("Bearer ${API_KEY}")
        assert result3 == "Bearer secret123"

        # Test recursive
        dict_val = {
            "url": "http://example.com?key=$API_KEY",
            "headers": {"Authorization": "Bearer ${API_KEY_BACKUP}"},
            "list": ["$API_KEY", "normal"],
        }
        res_dict = tool._replace_secrets(dict_val)
        assert res_dict["url"] == "http://example.com?key=secret123"
        assert res_dict["headers"]["Authorization"] == "Bearer secret456"
        assert res_dict["list"] == ["secret123", "normal"]


@pytest.mark.asyncio
async def test_run_json_async():
    with (
        patch(
            "xagent.core.tools.adapters.vibe.api_tool_adapter.decrypt_value",
            side_effect=lambda x: x,
        ),
        patch(
            "xagent.core.tools.adapters.vibe.api_tool_adapter.call_api"
        ) as mock_call_api,
    ):
        mock_call_api.return_value = {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": {"data": "test"},
            "error": None,
        }

        tool = CustomApiTool(name="test", description="test", env={"KEY": "val"})

        args = {"url": "http://test.com/$KEY", "method": "GET"}

        res = await tool.run_json_async(args)
        assert res["success"] is True
        assert res["status_code"] == 200
        assert res["body"] == {"data": "test"}

        mock_call_api.assert_called_once_with(
            url="http://test.com/val", method="GET", headers={}, params={}, body=None
        )


def test_run_json_sync_raises_runtime_error():
    tool = CustomApiTool(name="test", description="test", env={})

    # Since pytest-asyncio runs tests in an event loop if marked with @pytest.mark.asyncio
    # We can test that calling the sync version raises an error when a loop is running
    async def inner():
        with pytest.raises(RuntimeError, match="Event loop is already running"):
            tool.run_json_sync({"url": "http://test", "method": "GET"})

    import asyncio

    asyncio.run(inner())


def test_create_custom_api_tools():
    configs = [
        {"name": "api1", "description": "desc1", "env": {"k1": "v1"}},
        {"name": "api2", "description": "desc2", "env": {"k2": "v2"}},
    ]
    tools = create_custom_api_tools(configs)
    assert len(tools) == 2
    assert tools[0].name == "api_api1_call"
    assert tools[1].name == "api_api2_call"

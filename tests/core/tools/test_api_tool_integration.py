"""
Integration test for API Tool with Xagent tool system
"""

import pytest

from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory


class TestAPIToolIntegration:
    """Test API tool integration with Xagent tool system"""

    @pytest.mark.asyncio
    async def test_api_tool_registration(self):
        """Test that API tool is properly registered"""
        config = ToolConfig({"basic_tools_enabled": True})

        # Create all tools
        tools = await ToolFactory.create_all_tools(config)

        # Find API tool
        api_tool = None
        for tool in tools:
            if hasattr(tool, "name") and tool.name == "api_call":
                api_tool = tool
                break

        assert api_tool is not None, "API tool not found in registered tools"
        assert api_tool.name == "api_call"
        assert "HTTP requests to arbitrary APIs" in api_tool.description
        assert "api" in api_tool.tags
        assert api_tool.category.value == "basic"

    @pytest.mark.asyncio
    async def test_api_tool_execution(self):
        """Test API tool execution through the tool system"""
        config = ToolConfig({"basic_tools_enabled": True})
        tools = await ToolFactory.create_all_tools(config)

        # Find API tool
        api_tool = None
        for tool in tools:
            if hasattr(tool, "name") and tool.name == "api_call":
                api_tool = tool
                break

        assert api_tool is not None

        # Execute a simple GET request
        result = await api_tool.run_json_async(
            {
                "url": "https://httpbin.org/get",
                "method": "GET",
                "params": {"test": "value"},
            }
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert "body" in result

    @pytest.mark.asyncio
    async def test_api_tool_with_post(self):
        """Test API tool with POST request"""
        config = ToolConfig({"basic_tools_enabled": True})
        tools = await ToolFactory.create_all_tools(config)

        # Find API tool
        api_tool = None
        for tool in tools:
            if hasattr(tool, "name") and tool.name == "api_call":
                api_tool = tool
                break

        assert api_tool is not None

        # Execute POST request
        result = await api_tool.run_json_async(
            {
                "url": "https://httpbin.org/post",
                "method": "POST",
                "body": {"name": "test", "value": 123},
            }
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["body"]["json"]["name"] == "test"
        assert result["body"]["json"]["value"] == 123

    @pytest.mark.asyncio
    async def test_api_tool_return_formatting(self):
        """Test API tool return value formatting"""
        config = ToolConfig({"basic_tools_enabled": True})
        tools = await ToolFactory.create_all_tools(config)

        # Find API tool
        api_tool = None
        for tool in tools:
            if hasattr(tool, "name") and tool.name == "api_call":
                api_tool = tool
                break

        assert api_tool is not None

        # Test success formatting
        success_result = {
            "success": True,
            "status_code": 200,
            "body": {"result": "success"},
        }
        formatted = api_tool.return_value_as_string(success_result)
        assert "✅" in formatted
        assert "200" in formatted
        assert "success" in formatted

        # Test error formatting
        error_result = {
            "success": False,
            "error": "Connection failed",
        }
        formatted = api_tool.return_value_as_string(error_result)
        assert "❌" in formatted
        assert "Connection failed" in formatted

    @pytest.mark.asyncio
    async def test_api_tool_metadata(self):
        """Test API tool metadata properties"""
        config = ToolConfig({"basic_tools_enabled": True})
        tools = await ToolFactory.create_all_tools(config)

        # Find API tool
        api_tool = None
        for tool in tools:
            if hasattr(tool, "name") and tool.name == "api_call":
                api_tool = tool
                break

        assert api_tool is not None

        # Check metadata
        metadata = api_tool.metadata
        assert metadata.name == "api_call"
        assert metadata.category.value == "basic"
        assert "api" in metadata.tags
        assert "http" in metadata.tags

        # Check args and return types
        from pydantic import BaseModel

        assert issubclass(api_tool.args_type(), BaseModel)
        assert issubclass(api_tool.return_type(), BaseModel)

        # Check async support
        assert api_tool.is_async() is True

        # Check state (should be None for API tool)
        assert api_tool.state_type() is None

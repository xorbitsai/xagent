"""
Test AgentService workspace integration to reproduce the DAG file access issue.
"""

from unittest.mock import Mock

import pytest

from xagent.core.agent.service import AgentService
from xagent.core.model.chat.basic.openai import OpenAILLM


class TestAgentServiceWorkspaceIntegration:
    """Test AgentService workspace integration to identify the file access issue."""

    async def test_agent_service_workspace_flow(self, tmp_path):
        """Test the complete AgentService flow with workspace to identify the issue."""
        # Create mock LLM
        mock_llm = Mock(spec=OpenAILLM)

        # Mock LLM responses to simulate the DAG execution
        # First response: write file
        mock_llm.generate = Mock()
        mock_llm.generate.side_effect = [
            # Response for write_file step
            {
                "type": "tool_call",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"file_path": "gradient_hello.html", "content": "<!DOCTYPE html>\\n<html>\\n<head>\\n</head>\\n<body>\\n<h1>Hello</h1>\\n</body>\\n</html>"}',
                        },
                    }
                ],
            },
            # Response for read_file step
            {
                "type": "tool_call",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"file_path": "gradient_hello.html"}',
                        },
                    }
                ],
            },
        ]

        # Create AgentService with workspace enabled
        service = AgentService(
            name="test_agent",
            id="test_agent",
            enable_workspace=True,
            workspace_base_dir=str(tmp_path),
            llm=mock_llm,
        )

        # Verify workspace is set up
        assert service.workspace is not None
        assert service.enable_workspace is True

        # Initialize tools (lazy initialization)
        await service._ensure_tools_initialized()

        # Verify tools include workspace tools
        tool_names = [tool.metadata.name for tool in service.tools]
        assert "write_file" in tool_names
        assert "read_file" in tool_names

        # Verify DAG pattern has workspace reference
        dag_pattern = service.patterns[0]
        assert hasattr(dag_pattern, "workspace")
        assert dag_pattern.workspace is not None

        # Test simple write and read without DAG execution first
        # Find the tools
        write_tool = None
        read_tool = None
        for tool in service.tools:
            if tool.metadata.name == "write_file":
                write_tool = tool
            elif tool.metadata.name == "read_file":
                read_tool = tool

        assert write_tool is not None
        assert read_tool is not None

        # Test direct write and read (bypassing DAG complexity)
        test_content = "<html><body>Test</body></html>"

        # Write file
        write_result = write_tool.func("test_direct.html", test_content)
        assert isinstance(write_result, dict)
        assert write_result.get("success") is True
        assert isinstance(write_result.get("file_id"), str)

        # Verify file exists
        output_file = service.workspace.output_dir / "test_direct.html"
        assert output_file.exists()

        # Read file
        read_content = read_tool.func("test_direct.html")
        assert read_content == test_content

        # Now test with a more complex scenario that might reveal the issue
        # The issue might be in step agent creation or tool passing

        # Check if all tools share the same workspace instance
        # Note: Skill tools are global resource access and have their own workspace binding
        # They should be excluded from this check
        workspace_refs = []
        for tool in service.tools:
            # Skip skill tools (they have their own workspace for global resource access)
            if tool.metadata.category.value == "skill":
                continue
            # Handle both function tools and other tool types
            if hasattr(tool, "func") and hasattr(tool.func, "__self__"):
                tool_instance = tool.func.__self__
                if hasattr(tool_instance, "workspace"):
                    workspace_refs.append(tool_instance.workspace)
            elif hasattr(tool, "workspace"):
                workspace_refs.append(tool.workspace)

        # All workspace references should be the same
        if workspace_refs:
            assert all(w == workspace_refs[0] for w in workspace_refs), (
                "All tools should share the same workspace instance"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

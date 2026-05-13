"""
Tests for AgentService security features, specifically workspace-bound file tools.
"""

import pytest

from xagent.core.agent.service import AgentService
from xagent.core.tools.adapters.vibe.file_tool import SAFE_FILE_TOOLS


class TestAgentServiceSecurity:
    """Test AgentService security features."""

    def test_default_tools_exclude_unsafe_file_tools(self):
        """Test that default tools don't include unsafe file tools."""
        # Create AgentService without workspace (should not have file tools)
        service = AgentService(
            name="test_no_workspace",
            id="test_no_workspace",
            enable_workspace=False,
            llm=None,  # No LLM to avoid DAG pattern requirements
        )

        # Should have no tools when no workspace and no explicit tools provided
        assert len(service.tools) == 0

    def test_workspace_tools_are_bound_to_workspace(self, test_workspace_dir):
        """Test that workspace-enabled AgentService creates workspace-bound tools."""
        service = AgentService(
            name="test_workspace",
            id="test_workspace",
            enable_workspace=True,
            workspace_base_dir=test_workspace_dir,
            llm=None,  # No LLM to avoid DAG pattern requirements
        )

        # Initialize tools (lazy initialization)
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(service._ensure_tools_initialized())
        loop.close()

        # Should have workspace tools
        assert len(service.tools) > 0

        # Verify workspace is properly set up
        assert service.workspace is not None
        # Workspace creates a subdirectory within the base directory
        assert str(service.workspace.workspace_dir).startswith(test_workspace_dir)

        # Check that we have workspace-specific tools
        workspace_tool_names = [
            "read_file",
            "write_file",
            "append_file",
            "delete_file",
            "list_files",
            "create_directory",
            "file_exists",
            "get_file_info",
            "read_json_file",
            "write_json_file",
            "read_csv_file",
            "write_csv_file",
            "get_workspace_output_files",
        ]

        actual_tool_names = [tool.metadata.name for tool in service.tools]
        workspace_tools = [
            name for name in actual_tool_names if name in workspace_tool_names
        ]

        assert len(workspace_tools) == len(workspace_tool_names)

    def test_unsafe_file_tools_not_in_default_tools(self):
        """Test that unsafe file tools are not included in default web tools."""
        # Create AgentService with auto tool config and workspace disabled
        from xagent.core.agent.service import AgentService
        from xagent.core.memory.in_memory import InMemoryMemoryStore

        agent_service = AgentService(
            name="test_agent",
            id="test_agent",
            memory=InMemoryMemoryStore(),
            enable_workspace=False,  # Disable workspace to test "default" behavior
        )

        # Trigger tool initialization
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(agent_service._ensure_tools_initialized())

        tools = agent_service.tools

        # Should not include unsafe file tools
        unsafe_file_tool_names = [tool.metadata.name for tool in SAFE_FILE_TOOLS]
        actual_tool_names = [tool.metadata.name for tool in tools]

        # Verify no unsafe file tools are present
        for unsafe_tool_name in unsafe_file_tool_names:
            assert unsafe_tool_name not in actual_tool_names, (
                f"Unsafe tool {unsafe_tool_name} found in default tools"
            )

    def test_workspace_tools_restrict_file_access(self, test_access_dir):
        """Test that workspace tools properly restrict file access to workspace directory."""
        service = AgentService(
            name="test_access_restriction",
            id="test_access_restriction",
            enable_workspace=True,
            workspace_base_dir=test_access_dir,
            llm=None,
        )

        # Initialize tools (lazy initialization)
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(service._ensure_tools_initialized())
        loop.close()

        # Find the write_file tool
        write_tool = None
        for tool in service.tools:
            if tool.metadata.name == "write_file":
                write_tool = tool
                break

        assert write_tool is not None, "write_file tool not found"

        # Test writing to a file within workspace
        test_content = "Hello, workspace!"
        result = write_tool.func("test.txt", test_content)
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert isinstance(result.get("file_id"), str)

        # Verify file was created in workspace
        test_file_path = service.workspace.output_dir / "test.txt"
        assert test_file_path.exists()
        assert test_file_path.read_text() == test_content

        # Clean up
        test_file_path.unlink()

    def test_workspace_tools_reject_outside_access(self, test_security_dir):
        """Test that workspace tools reject attempts to access files outside workspace."""
        service = AgentService(
            name="test_security",
            id="test_security",
            enable_workspace=True,
            workspace_base_dir=test_security_dir,
            llm=None,
        )

        # Initialize tools (lazy initialization)
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(service._ensure_tools_initialized())
        loop.close()

        # Find the read_file tool
        read_tool = None
        for tool in service.tools:
            if tool.metadata.name == "read_file":
                read_tool = tool
                break

        assert read_tool is not None, "read_file tool not found"

        # Try to read a file outside workspace - should raise ValueError
        with pytest.raises(
            ValueError, match="Path /etc/passwd is outside allowed directories"
        ):
            read_tool.func("/etc/passwd")  # Attempt to read system file


class TestAgentServicePauseResume:
    """Test AgentService pause and resume functionality."""

    @pytest.mark.asyncio
    async def test_pause_resume_functionality(self):
        """Test that AgentService can be paused and resumed."""
        service = AgentService(
            name="test_pause_resume",
            id="test_pause_resume",
            enable_workspace=False,
            llm=None,  # No LLM to avoid DAG pattern requirements
        )

        # Initially should not be paused
        assert not service.is_paused()

        # Test pausing
        await service.pause_execution()
        assert service.is_paused()

        # Test resuming
        await service.resume_execution()
        assert not service.is_paused()

    @pytest.mark.asyncio
    async def test_double_pause_handling(self):
        """Test that pausing an already paused service handles gracefully."""
        service = AgentService(
            name="test_double_pause",
            id="test_double_pause",
            enable_workspace=False,
            llm=None,
        )

        # First pause
        await service.pause_execution()
        assert service.is_paused()

        # Second pause should not cause issues
        await service.pause_execution()
        assert service.is_paused()  # Should still be paused

    @pytest.mark.asyncio
    async def test_resume_without_pause(self):
        """Test that resuming a non-paused service handles gracefully."""
        service = AgentService(
            name="test_resume_without_pause",
            id="test_resume_without_pause",
            enable_workspace=False,
            llm=None,
        )

        # Resume without pausing first
        await service.resume_execution()
        assert not service.is_paused()  # Should not be paused

    @pytest.mark.asyncio
    async def test_pause_resume_cycle(self):
        """Test multiple pause/resume cycles."""
        service = AgentService(
            name="test_cycle",
            id="test_cycle",
            enable_workspace=False,
            llm=None,
        )

        # Multiple cycles
        for _ in range(3):
            await service.pause_execution()
            assert service.is_paused()
            await service.resume_execution()
            assert not service.is_paused()

    @pytest.mark.asyncio
    async def test_v2_resume_clears_service_pause_state(self):
        """Test that v2 resume clears the service pause flag for later pauses."""

        class FakeAdapter:
            def __init__(self):
                self.config = type("Config", (), {"tools": []})()

            def pause(self, execution_id, reason=None):
                return True

            async def resume(self, execution_id, **kwargs):
                return {
                    "success": False,
                    "status": "interrupted",
                    "execution_id": execution_id,
                    "kwargs": kwargs,
                }

        service = AgentService(
            name="test_v2_resume",
            id="test_v2_resume",
            enable_workspace=False,
            llm=None,
        )
        service.agent_runtime = "v2"
        service._v2_adapter = FakeAdapter()

        await service.pause_execution()
        assert service.is_paused()

        result = await service.resume_v2_execution("test_v2_resume")

        assert result is not None
        assert result["status"] == "interrupted"
        assert not service.is_paused()

    @pytest.mark.asyncio
    async def test_v2_resume_initializes_configured_tools(self, monkeypatch):
        """Test that v2 resume restores dynamically configured tools."""

        class FakeTool:
            name = "restored_search"

        class FakeToolConfig:
            _workspace_config = {}

            def get_allowed_tools(self):
                return ["restored_search"]

        class FakeAdapter:
            def __init__(self):
                self.config = type("Config", (), {"tools": []})()

            async def resume(self, execution_id, **kwargs):
                return {
                    "success": True,
                    "execution_id": execution_id,
                    "tool_names": [tool.name for tool in self.config.tools],
                }

        async def fake_create_all_tools(_config):
            return [FakeTool()]

        from xagent.core.tools.adapters.vibe.factory import ToolFactory

        monkeypatch.setattr(ToolFactory, "create_all_tools", fake_create_all_tools)

        service = AgentService(
            name="test_v2_resume_tools",
            id="test_v2_resume_tools",
            enable_workspace=False,
            llm=None,
            tool_config=FakeToolConfig(),
        )
        service.agent_runtime = "v2"
        service._v2_adapter = FakeAdapter()

        result = await service.resume_v2_execution("test_v2_resume_tools")

        assert result is not None
        assert result["tool_names"] == ["restored_search"]
        assert service._tools_initialized is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

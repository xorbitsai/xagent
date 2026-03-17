"""
Test instance-bound workspace tools.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.workspace import create_workspace


async def test_instance_bound_tools():
    """
    Test that workspace tools are properly bound to instances.
    """
    print("Testing instance-bound workspace tools...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create two different workspaces
        workspace1 = create_workspace("agent1_task1", str(temp_path))
        workspace2 = create_workspace("agent2_task2", str(temp_path))

        # Create tool instances for each workspace
        tools1 = WorkspaceFileTools(workspace1)
        tools2 = WorkspaceFileTools(workspace2)

        # Test that tools are bound to different workspaces
        assert tools1.workspace != tools2.workspace
        print(f"✅ Tools1 bound to: {tools1.workspace.workspace_dir}")
        print(f"✅ Tools2 bound to: {tools2.workspace.workspace_dir}")

        # Test file operations in workspace1
        tools1.write_file("test1.txt", "Workspace1 content")
        assert tools1.file_exists("test1.txt")

        content1 = tools1.read_file("test1.txt")
        assert content1 == "Workspace1 content"
        print("✅ Workspace1 tools working correctly")

        # Test file operations in workspace2
        tools2.write_file("test2.txt", "Workspace2 content")
        assert tools2.file_exists("test2.txt")

        content2 = tools2.read_file("test2.txt")
        assert content2 == "Workspace2 content"
        print("✅ Workspace2 tools working correctly")

        # Verify isolation: tools1 cannot see workspace2 files
        assert not tools1.file_exists("test2.txt")
        assert not tools2.file_exists("test1.txt")
        print("✅ Tool isolation verified")

        # Verify files are in different workspaces
        files1 = workspace1.get_output_files()
        files2 = workspace2.get_output_files()

        print(f"✅ Workspace1 output files: {[f['filename'] for f in files1]}")
        print(f"✅ Workspace2 output files: {[f['filename'] for f in files2]}")

        # Workspace1 should have test1.txt but not test2.txt
        filenames1 = [f["filename"] for f in files1]
        filenames2 = [f["filename"] for f in files2]

        assert "test1.txt" in filenames1
        assert "test2.txt" not in filenames1
        assert "test1.txt" not in filenames2
        assert "test2.txt" in filenames2

        print("✅ Instance-bound tools test passed!")


async def test_concurrent_tool_instances():
    """
    Test concurrent execution with different tool instances.
    """
    print("\nTesting concurrent tool instances...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create multiple workspaces
        workspaces = []
        tool_instances = []
        for i in range(3):
            workspace = create_workspace(f"agent{i}_task{i}", str(temp_path))
            workspaces.append(workspace)

            # Create tool instance for each workspace
            tools = WorkspaceFileTools(workspace)
            tool_instances.append(tools)

        # Define concurrent tasks
        async def write_with_tools(tools, index):
            tools.write_file(
                f"concurrent_test_{index}.txt", f"Content from tools {index}"
            )
            return tools.workspace.get_output_files()

        # Execute tasks concurrently
        tasks = []
        for i, tools in enumerate(tool_instances):
            task = write_with_tools(tools, i)
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Verify each workspace has its own file
        for i, workspace in enumerate(workspaces):
            files = workspace.get_output_files()
            filenames = [f["filename"] for f in files]
            assert f"concurrent_test_{i}.txt" in filenames
            print(f"✅ Workspace {i} has file: concurrent_test_{i}.txt")

        print("✅ Concurrent tool instances test passed!")


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows short filename path case-sensitivity issue"
)
async def test_tool_creation_function():
    """
    Test the create_workspace_file_tools function.
    """
    print("\nTesting tool creation function...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create workspace
        workspace = create_workspace("test_agent_task", str(temp_path))

        # Import the creation functions
        from xagent.core.tools.adapters.vibe.skill_tools import create_skill_tools
        from xagent.core.tools.adapters.vibe.workspace_file_tool import (
            create_workspace_file_tools,
        )

        # Create file tools using the function
        file_tools = create_workspace_file_tools(workspace)
        assert len(file_tools) == 15  # Should have 15 file tools

        # Test file tool functionality
        write_tool = next(tool for tool in file_tools if tool.name == "write_file")
        read_tool = next(tool for tool in file_tools if tool.name == "read_file")

        # Use the tools
        write_tool.func("function_test.txt", "Test content")
        content = read_tool.func("function_test.txt")
        assert content == "Test content"
        print("✅ File tools creation and functionality test passed!")

        # Create skill tools using the function
        skill_tools = create_skill_tools(workspace)
        assert len(skill_tools) == 3  # Should have 3 skill tools

        # Verify skill tool names
        skill_tool_names = {tool.name for tool in skill_tools}
        assert "read_skill_doc" in skill_tool_names
        assert "list_skill_docs" in skill_tool_names
        assert "fetch_skill_file" in skill_tool_names
        print("✅ Skill tools creation test passed!")

        # Total tools: 15 file + 3 skill = 18
        all_tools = file_tools + skill_tools
        assert len(all_tools) == 18

        print("✅ Tool creation function test passed!")


async def main():
    """
    Run all instance-bound tools tests.
    """
    try:
        await test_instance_bound_tools()
        await test_concurrent_tool_instances()
        await test_tool_creation_function()
        print("\n🎉 All instance-bound tools tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

"""
Command Line Execution Tool for xagent
Framework wrapper around the pure command executor tool
"""

import asyncio
import logging
from typing import Any, Dict, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ....workspace import TaskWorkspace
from ...core.command_executor import CommandExecutorCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .function import FunctionTool

logger = logging.getLogger(__name__)


class CommandExecutorFunctionTool(FunctionTool):
    """Command executor tool with BASIC category."""

    category = ToolCategory.BASIC


class CommandExecutorArgs(BaseModel):
    command: str = Field(description="Shell command to execute")
    timeout: Optional[int] = Field(
        default=None, description="Execution timeout in seconds (default: 300)"
    )


class CommandExecutorResult(BaseModel):
    success: bool = Field(description="Whether the command executed successfully")
    output: str = Field(description="Standard output from the command")
    error: str = Field(default="", description="Standard error from the command")
    return_code: int = Field(description="Process exit code")


class CommandExecutorTool(AbstractBaseTool):
    """Framework wrapper for the pure command executor tool"""

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "command_executor"

    @property
    def description(self) -> str:
        return """Execute shell commands and scripts.
        Supports any shell command including system commands, scripts, pipes, and redirects.
        Examples: ls -la, grep -r 'pattern' /path, ./deploy.sh, cat file.txt | grep error"""

    @property
    def tags(self) -> list[str]:
        return ["shell", "command", "bash", "script", "terminal"]

    def args_type(self) -> Type[BaseModel]:
        return CommandExecutorArgs

    def return_type(self) -> Type[BaseModel]:
        return CommandExecutorResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        exec_args = CommandExecutorArgs.model_validate(args)

        # Determine working directory
        working_directory = self._get_working_directory()

        # Create core executor instance
        executor = CommandExecutorCore(working_directory)

        # Execute command
        result = executor.execute_command(exec_args.command, timeout=exec_args.timeout)

        return CommandExecutorResult(**result).model_dump()

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(self.run_json_sync, args)

    def _get_working_directory(self) -> Optional[str]:
        """Determine the working directory based on workspace settings"""
        if self._workspace:
            # Use workspace output directory as working directory
            return str(self._workspace.resolve_path(""))
        return None


def get_command_executor_tool(info: Optional[dict[str, Any]] = None) -> FunctionTool:
    """
    Create a workspace-bound command executor tool.

    Args:
        info: Dictionary containing workspace information

    Returns:
        A command executor tool bound to the specified workspace
    """
    # Extract workspace from info if provided
    workspace = None
    if info and "workspace" in info:
        workspace = info["workspace"]

    # Create workspace-bound command executor
    executor = CommandExecutorTool(workspace=workspace)

    # Wrap as LangChain tool
    def execute_command(command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute shell command."""
        result: Dict[str, Any] = executor.run_json_sync(
            {"command": command, "timeout": timeout}
        )
        return result

    return CommandExecutorFunctionTool(execute_command)


def create_command_executor_tool(
    workspace: TaskWorkspace,
) -> AbstractBaseTool:
    """Create command executor tool bound to workspace"""
    return CommandExecutorTool(workspace)

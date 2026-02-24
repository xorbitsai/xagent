"""
JavaScript Code Execution Tool for xagent
Framework wrapper around the JavaScript executor tool
"""

import asyncio
import logging
from typing import Any, Dict, Mapping, Optional, Type

from pydantic import BaseModel, Field

from xagent.core.tools.core.javascript_executor import JavaScriptExecutorCore
from xagent.core.workspace import TaskWorkspace

from .base import AbstractBaseTool, ToolVisibility
from .function import FunctionTool

logger = logging.getLogger(__name__)


class JavaScriptExecutorArgs(BaseModel):
    code: str = Field(description="JavaScript to execute")
    packages: Optional[str] = Field(
        default=None,
        description="Comma-separated list of npm packages (e.g., 'pptxgenjs,axios')",
    )


class JavaScriptExecutorResult(BaseModel):
    success: bool = Field(description="Whether the execution was successful")
    output: str = Field(description="Output from the execution")
    error: str = Field(default="", description="Error message if execution failed")


class JavaScriptExecutorTool(AbstractBaseTool):
    """Framework wrapper for the JavaScript executor tool"""

    def __init__(self, workspace: Optional[TaskWorkspace] = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "javascript_executor"

    @property
    def description(self) -> str:
        return """Execute JavaScript using Node.js runtime.
        Supports npm packages for extended functionality including PPTX generation, HTTP requests, and more.
        Captures stdout and stderr from the execution."""

    @property
    def tags(self) -> list[str]:
        return ["javascript", "code", "execution", "computation"]

    def args_type(self) -> Type[BaseModel]:
        return JavaScriptExecutorArgs

    def return_type(self) -> Type[BaseModel]:
        return JavaScriptExecutorResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        exec_args = JavaScriptExecutorArgs.model_validate(args)

        # Determine working directory
        working_directory = self._get_working_directory()

        # Parse packages if provided
        pkg_list = None
        if exec_args.packages:
            pkg_list = [p.strip() for p in exec_args.packages.split(",")]

        # Create core executor instance
        executor = JavaScriptExecutorCore(working_directory)

        # Execute code
        result = executor.execute_code(exec_args.code, packages=pkg_list)

        return JavaScriptExecutorResult(**result).model_dump()

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(self.run_json_sync, args)

    def _get_working_directory(self) -> Optional[str]:
        """Determine the working directory based on workspace settings"""
        if self._workspace:
            # Use workspace output directory as working directory
            return str(self._workspace.resolve_path(""))
        return None


def get_javascript_executor_tool(info: Optional[dict[str, Any]] = None) -> FunctionTool:
    """
    Create a workspace-bound JavaScript executor tool.

    Args:
        info: Dictionary containing workspace information

    Returns:
        A JavaScript executor tool bound to the specified workspace
    """
    # Extract workspace from info if provided
    workspace = None
    if info and "workspace" in info:
        workspace = (
            info["workspace"] if isinstance(info["workspace"], TaskWorkspace) else None
        )

    # Create workspace-bound JavaScript executor
    executor = JavaScriptExecutorTool(workspace=workspace)

    # Wrap as LangChain tool
    def execute_javascript_code(
        code: str, packages: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute JavaScript code using Node.js runtime."""
        result: Dict[str, Any] = executor.run_json_sync(
            {"code": code, "packages": packages}
        )
        return result

    return FunctionTool(execute_javascript_code)


def create_javascript_executor_tool(workspace: TaskWorkspace) -> AbstractBaseTool:
    """Create JavaScript executor tool bound to workspace"""
    return JavaScriptExecutorTool(workspace)

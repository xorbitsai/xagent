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

from ...artifacts import (
    build_generated_file_metadata,
    changed_generated_artifact_files,
    snapshot_generated_artifact_files,
)
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .function import FunctionTool
from .sandboxed_tool.sandbox_config import sandbox_config

logger = logging.getLogger(__name__)


class JavaScriptExecutorFunctionTool(FunctionTool):
    """JavaScript executor tool with CODE_EXECUTION category."""

    category = ToolCategory.CODE_EXECUTION


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
    generated_files: list[str] = Field(default_factory=list)
    file_refs: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


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
        working_directory = self._get_working_directory()
        workspace_line = (
            f"JavaScript runs with current working directory: {working_directory}."
            if working_directory
            else "JavaScript runs in a temporary directory."
        )
        return f"""Execute JavaScript using Node.js runtime.
        Supports npm packages for extended functionality including PPTX generation, HTTP requests, and more.
        Captures stdout and stderr from the execution.
        {workspace_line}"""

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

        # Execute code within auto_register context
        if self._workspace and working_directory:
            files_before = snapshot_generated_artifact_files(working_directory)
            with self._workspace.auto_register_files():
                result = executor.execute_code(exec_args.code, packages=pkg_list)
            if result.get("success"):
                files_after = snapshot_generated_artifact_files(working_directory)
                metadata = build_generated_file_metadata(
                    workspace=self._workspace,
                    file_paths=changed_generated_artifact_files(
                        files_before, files_after
                    ),
                )
                result.update(metadata)
                if metadata["generated_files"]:
                    output = (
                        result.get("output") or "Code executed successfully (no output)"
                    )
                    result["output"] = (
                        f"{output}\n\nGenerated files: "
                        f"{', '.join(metadata['generated_files'])}"
                    )
        else:
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


@sandbox_config()
class JavaScriptExecutorToolForBasic(JavaScriptExecutorTool):
    """JavaScript executor tool with CODE_EXECUTION category."""

    category = ToolCategory.CODE_EXECUTION

    @property
    def name(self) -> str:
        return "execute_javascript_code"


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

    return JavaScriptExecutorFunctionTool(
        execute_javascript_code, description=executor.description
    )


def create_javascript_executor_tool(workspace: TaskWorkspace) -> AbstractBaseTool:
    """Create JavaScript executor tool bound to workspace"""
    return JavaScriptExecutorTool(workspace)

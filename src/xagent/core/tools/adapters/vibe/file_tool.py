"""
Basic file tool module for xagent

This module provides basic file operation functions, excluding workspace-related features.
For workspace-related file operations, please use the workspace_file_tool.py module.
"""

from typing import Optional

from ...core.file_tool import (
    append_file,
    create_directory,
    delete_file,
    edit_file,
    file_exists,
    find_and_replace,
    get_file_info,
    list_files,
    read_csv_file,
    read_file,
    read_json_file,
    write_csv_file,
    write_file,
    write_json_file,
)
from .base import ToolCategory
from .function import FunctionTool


class FileTool(FunctionTool):
    """FileTool with ToolCategory.FILE category."""

    category = ToolCategory.FILE


# Create basic tool instances (these tools are unsafe, for internal use only)
read_file_tool = FileTool(
    read_file,
    name="read_file",
    description=(
        "Read file content. For large files, results may be truncated in model "
        "context; use start_line/end_line to inspect a specific 1-based "
        "inclusive line range instead of repeating the same full-file read."
    ),
    read_only=True,
)
write_file_tool = FileTool(
    write_file, name="write_file", description="Write content to file"
)
append_file_tool = FileTool(
    append_file, name="append_file", description="Append content to file"
)
delete_file_tool = FileTool(delete_file, name="delete_file", description="Delete file")
list_files_tool = FileTool(
    list_files,
    name="list_files",
    description="List files in directory",
    read_only=True,
)
create_directory_tool = FileTool(
    create_directory, name="create_directory", description="Create directory"
)
file_exists_tool = FileTool(
    file_exists,
    name="file_exists",
    description="Check if file exists",
    read_only=True,
)
get_file_info_tool = FileTool(
    get_file_info,
    name="get_file_info",
    description="Get detailed file information",
    read_only=True,
)
read_json_file_tool = FileTool(
    read_json_file,
    name="read_json_file",
    description="Read JSON file",
    read_only=True,
)
write_json_file_tool = FileTool(
    write_json_file, name="write_json_file", description="Write JSON file"
)
read_csv_file_tool = FileTool(
    read_csv_file,
    name="read_csv_file",
    description="Read CSV file",
    read_only=True,
)
write_csv_file_tool = FileTool(
    write_csv_file, name="write_csv_file", description="Write CSV file"
)
edit_file_tool = FileTool(
    edit_file,
    name="edit_file",
    description="Precisely edit file content, supporting various editing operations based on line numbers and pattern matching",
)
find_and_replace_tool = FileTool(
    find_and_replace,
    name="find_and_replace",
    description="Convenient function to find and replace text content",
)


# Basic file tool list (unsafe, for special scenarios only)
BASIC_FILE_TOOLS = [
    read_file_tool,
    write_file_tool,
    append_file_tool,
    delete_file_tool,
    list_files_tool,
    create_directory_tool,
    file_exists_tool,
    get_file_info_tool,
    read_json_file_tool,
    write_json_file_tool,
    read_csv_file_tool,
    write_csv_file_tool,
    edit_file_tool,
    find_and_replace_tool,
]


# Tool getter functions for auto-discovery mechanism
def get_edit_file_tool(info: Optional[dict[str, str]] = None) -> FunctionTool:
    """Get edit_file tool"""
    return edit_file_tool


def get_find_and_replace_tool(info: Optional[dict[str, str]] = None) -> FunctionTool:
    """Get find_and_replace tool"""
    return find_and_replace_tool


# Safe tool list (includes basic tools only, should use workspace_file_tool in actual usage)
SAFE_FILE_TOOLS = BASIC_FILE_TOOLS.copy()

# FILE_TOOLS now points to basic tools (not recommended)
FILE_TOOLS = BASIC_FILE_TOOLS

# Note: Tools in this module are unsafe, mainly for backward compatibility
# For safe file tools, please use the workspace_file_tool.py module

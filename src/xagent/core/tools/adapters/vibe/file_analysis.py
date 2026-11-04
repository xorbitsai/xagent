"""File analysis tools for agent"""

from ...core.file_analysis import (
    analyze_uploaded_file,
    get_file_context,
    list_uploaded_files,
)
from .base import ToolCategory
from .function import FunctionTool


class FileAnalysisTool(FunctionTool):
    """FileAnalysisTool with ToolCategory.FILE category."""

    category = ToolCategory.FILE


# Create tool instances
analyze_uploaded_file_tool = FileAnalysisTool(
    analyze_uploaded_file,
    name="analyze_uploaded_file",
    description="Analyze uploaded file content, supporting text, JSON, CSV, Python, Markdown and other formats",
)

list_uploaded_files_tool = FileAnalysisTool(
    list_uploaded_files,
    name="list_uploaded_files",
    description="List all files in upload directory",
)

get_file_context_tool = FileAnalysisTool(
    get_file_context,
    name="get_file_context",
    description="Get complete context information for file, including analysis results and available file list",
)

# File analysis tool list
FILE_ANALYSIS_TOOLS = [
    analyze_uploaded_file_tool,
    list_uploaded_files_tool,
    get_file_context_tool,
]

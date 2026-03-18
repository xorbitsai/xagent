"""
SQL Tool adapter for xagent framework.

Wraps the core SQL tool for framework integration.
"""

import logging
from textwrap import dedent, indent
from typing import TYPE_CHECKING, Any, Optional

from ....workspace import TaskWorkspace
from ...core.sql_tool import execute_sql_query, get_database_type
from .base import ToolCategory
from .factory import ToolFactory, register_tool
from .function import FunctionTool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


class SQLQueryFunctionTool(FunctionTool):
    """SQL query tool with DATABASE category."""

    category = ToolCategory.DATABASE


class SqlQueryTool:
    """
    SQL query tool that executes SQL queries on configured databases.
    """

    def __init__(
        self,
        workspace: Optional[TaskWorkspace] = None,
        connection_map: Optional[dict[str, str]] = None,
    ):
        """
        Initialize SQL query tool.

        Args:
            workspace: Optional workspace for file-based operations
        """
        self._workspace = workspace
        self._connection_map = {
            key.upper(): value for key, value in (connection_map or {}).items()
        }

    def _resolve_connection_url(self, connection_name: str) -> Optional[str]:
        return self._connection_map.get(connection_name.upper())

    def execute_sql_query(
        self, connection_name: str, query: str, output_file: Optional[str] = None
    ) -> dict[str, Any]:
        return execute_sql_query(
            connection_name,
            query,
            output_file,
            self._workspace,
            self._resolve_connection_url(connection_name),
        )

    def get_database_type(self, connection_name: str) -> str:
        return get_database_type(
            connection_name, self._resolve_connection_url(connection_name)
        )

    def get_tools(self) -> list:
        """Get all tool instances."""
        tools = [
            SQLQueryFunctionTool(
                self.get_database_type,
                name="get_database_type",
                description=indent(
                    dedent("""
                    Get the database type for a connection name.

                    This helps determine the SQL dialect to use when writing queries.
                    Different databases have different syntax and functions.

                    Args:
                        connection_name: Database connection name to check

                    Returns:
                        str: Database type (postgresql, mysql, sqlite, duckdb, etc.)
                """),
                    "" * 4,
                ),
                tags=["sql", "database", "metadata"],
            ),
            SQLQueryFunctionTool(
                self.execute_sql_query,
                name="execute_sql_query",
                description=indent(
                    dedent("""
                    Execute SQL queries on databases and return structured results.

                    TIP: Call get_database_type(connection_name) first to learn the SQL dialect
                    (postgresql, mysql, sqlite, duckdb have different syntax).

                        Args:
                            connection_name: (REQUIRED) The database connection name.
                            query: (REQUIRED) SQL statement to execute.
                                Use syntax matching the database type.
                            output_file: (OPTIONAL) Export results to file instead of returning them.
                                Supported: .csv, .parquet, .json, .jsonl, .ndjson (relative to workspace).
                                Use for large datasets to avoid response size limits.

                        Returns:
                            dict with keys:
                            - success: true if query worked
                            - rows: query results as list of dicts (SELECT only, empty when exported)
                            - row_count: number of rows returned or affected
                            - columns: column names in the result
                            - message: what happened (includes export info when applicable)
                """),
                    "" * 4,
                ),
                tags=[
                    "sql",
                    "database",
                    "query",
                    "postgresql",
                    "mysql",
                    "sqlite",
                    "duckdb",
                ],
            ),
        ]

        return tools


def get_sql_tool(info: Optional[dict[str, Any]] = None) -> list[FunctionTool]:
    workspace: TaskWorkspace | None = None
    connection_map: dict[str, str] | None = None
    if info and "workspace" in info:
        workspace = (
            info["workspace"] if isinstance(info["workspace"], TaskWorkspace) else None
        )
    if info and "connection_map" in info and isinstance(info["connection_map"], dict):
        connection_map = {
            str(key): str(value)
            for key, value in info["connection_map"].items()
            if isinstance(key, str) and isinstance(value, str)
        }

    tool_instance = SqlQueryTool(workspace=workspace, connection_map=connection_map)
    return tool_instance.get_tools()


@register_tool
async def create_sql_tools(config: "BaseToolConfig") -> list:
    """
    Create SQL query tools with workspace support.

    Registered via @register_tool decorator for auto-discovery.

    Args:
        config: Tool configuration with workspace settings

    Returns:
        List of tool instances
    """
    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    connection_map = config.get_sql_connections()
    tool_instance = SqlQueryTool(workspace, connection_map=connection_map)
    return tool_instance.get_tools()

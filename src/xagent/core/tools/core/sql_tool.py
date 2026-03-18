"""
SQL Tool for xagent - SQL execution using SQLAlchemy

Database connections are configured via environment variables, not raw URLs.
Connection format: XAGENT_EXTERNAL_DB_<NAME>=<connection_url>

Example:
    XAGENT_EXTERNAL_DB_ANALYTICS=postgresql://user:pass@localhost:5432/analytics
    XAGENT_EXTERNAL_DB_PROD=mysql+pymysql://user:pass@localhost:3306/production
    XAGENT_EXTERNAL_DB_LOCAL=sqlite:///path/to/database.db
    XAGENT_EXTERNAL_DB_DUCKDB=duckdb:///path/to/database.duckdb

Note: This tool uses SQLAlchemy's synchronous engine.
Async drivers are not supported currently.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import CursorResult, Row, make_url

if TYPE_CHECKING:
    from ...workspace import TaskWorkspace

logger = logging.getLogger(__name__)


class SQLQueryArgs(BaseModel):
    """Arguments for SQL query execution."""

    connection_name: str = Field(description="Database connection name to use")
    query: str = Field(description="SQL query to execute")


class SQLQueryResult(BaseModel):
    """Result from SQL query execution in LLM-friendly format"""

    success: bool = Field(description="Whether the query executed successfully")
    rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Query result rows as list of dictionaries"
    )
    row_count: int = Field(default=0, description="Number of rows affected/returned")
    columns: list[str] = Field(
        default_factory=list, description="Column names in result set"
    )
    message: str = Field(default="", description="Summary of what happened")


def _get_connection_url(
    connection_name: str, connection_url: Optional[str] = None
) -> URL:
    """Get database connection URL from environment variable.

    Environment variable format: XAGENT_EXTERNAL_DB_<NAME>=<connection_url>

    Args:
        connection_name: Name of the connection (case-insensitive)

    Returns:
        Connection URL if found
    """
    url = connection_url
    if not url:
        env_key = f"XAGENT_EXTERNAL_DB_{connection_name.upper()}"
        url = os.getenv(env_key)

    if not url:
        raise ValueError(f"Database connection '{connection_name}' not found.")

    # Validate URL format using SQLAlchemy
    return make_url(url)


def get_database_type(
    connection_name: str, connection_url: Optional[str] = None
) -> str:
    """Get database type for a connection name.

    Returns the database driver/type which helps LLM write appropriate SQL dialect.
    Examples: postgresql, mysql, sqlite, duckdb

    Args:
        connection_name: Name of the connection (case-insensitive)

    Returns:
        Database type (driver name)
    """
    url = _get_connection_url(connection_name, connection_url)
    # Extract driver name from URL (e.g., "postgresql+asyncpg" -> "postgresql")
    return url.drivername.split("+")[0]


def _row_to_dict(row: Row) -> dict[str, Any]:
    """Convert SQLAlchemy Row to dictionary"""
    return dict(row._mapping)


def execute_sql_query(
    connection_name: str,
    query: str,
    output_file: Optional[str] = None,
    workspace: Optional["TaskWorkspace"] = None,
    connection_url: Optional[str] = None,
) -> dict[str, Any]:
    """Execute SQL queries on databases and return structured results.

    Args:
        connection_name: Database connection name to use
        query: SQL statement to execute
        output_file: Optional file path to export query results.
            Supported formats: .csv, .parquet, .json, .jsonl, .ndjson (relative to workspace output directory).
            When provided, query results are exported to file instead of being returned.
        workspace: Optional TaskWorkspace instance for file exports.

    Returns:
        dict:
            with keys:
            - success: true if query worked
            - rows: query results as list of dicts (SELECT only, empty when exported)
            - row_count: number of rows returned or affected
            - columns: column names in the result
            - message: what happened
    """
    # Get connection URL from environment
    url = _get_connection_url(connection_name, connection_url)
    stmt = text(query)
    engine = create_engine(url)

    try:
        with engine.connect() as conn:
            # Check if export to file is requested first
            if output_file and workspace:
                file_ext = Path(output_file).suffix.lower()
                if file_ext == ".csv":
                    # Streaming export for large datasets
                    result = conn.execute(stmt)
                    _, exported_count, columns = _stream_export_to_csv(
                        workspace, output_file, result
                    )
                    return SQLQueryResult(
                        success=True,
                        rows=[],
                        row_count=exported_count,
                        columns=columns,
                        message=f"Query executed successfully on '{connection_name}', exported {exported_count} row(s) to {output_file}",
                    ).model_dump()
                elif file_ext == ".parquet":
                    # Streaming export with Parquet (better compression & type preservation)
                    result = conn.execute(stmt)
                    (
                        _,
                        exported_count,
                        columns,
                    ) = _stream_export_to_parquet(workspace, output_file, result)
                    return SQLQueryResult(
                        success=True,
                        rows=[],
                        row_count=exported_count,
                        columns=columns,
                        message=f"Query executed successfully on '{connection_name}', exported {exported_count} row(s) to {output_file}",
                    ).model_dump()
                elif file_ext in (".json", ".jsonl", ".ndjson"):
                    # Streaming JSON Lines (NDJSON) export
                    result = conn.execute(stmt)
                    (
                        _,
                        exported_count,
                        columns,
                    ) = _stream_export_to_jsonlines(workspace, output_file, result)
                    return SQLQueryResult(
                        success=True,
                        rows=[],
                        row_count=exported_count,
                        columns=columns,
                        message=f"Query executed successfully on '{connection_name}', exported {exported_count} row(s) to {output_file}",
                    ).model_dump()
                else:
                    raise ValueError(
                        f"Unsupported file format: {file_ext}. "
                        f"Supported: .csv (streaming), .parquet (streaming), .json/.jsonl/.ndjson (streaming JSON Lines)"
                    )

            # Original behavior: return data in response
            result = conn.execute(stmt)

            # Get column names from result
            if result.returns_rows:
                rows = result.all()
                row_list = [_row_to_dict(row) for row in rows]

                # Extract column names from first row
                columns = list(row_list[0].keys()) if row_list else []

                return SQLQueryResult(
                    success=True,
                    rows=row_list,
                    row_count=len(row_list),
                    columns=columns,
                    message=f"Query executed successfully on '{connection_name}', returned {len(row_list)} row(s)",
                ).model_dump()
            else:
                # For INSERT, UPDATE, DELETE operations
                rowcount = result.rowcount if hasattr(result, "rowcount") else 0

                # Commit the transaction for non-SELECT queries
                conn.commit()

                return SQLQueryResult(
                    success=True,
                    rows=[],
                    row_count=rowcount,
                    columns=[],
                    message=f"Query executed successfully on '{connection_name}', affected {rowcount} row(s)",
                ).model_dump()
    finally:
        engine.dispose()


def _stream_export_to_csv(
    workspace: "TaskWorkspace",
    file_path: str,
    result: CursorResult,
    batch_size: int = 1000,
) -> tuple[str, int, list[str]]:
    """Streaming export to CSV.

    Returns:
        Tuple of (exported_file_path, row_count, column_names)
    """
    resolved_path = workspace.resolve_path(file_path, default_dir="output")

    # Get column names BEFORE iteration
    columns = list(result.keys())

    row_count = 0
    writer: csv.DictWriter | None = None

    with open(resolved_path, "w", encoding="utf-8", newline="") as f:
        # Fetch in batches
        while True:
            batch = result.fetchmany(batch_size)
            if not batch:
                break

            # Convert batch to dict format
            batch_dicts = [_row_to_dict(row) for row in batch]

            # Initialize writer on first batch
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

            # Write batch to file
            if writer is not None:
                writer.writerows(batch_dicts)
            row_count += len(batch)

    return str(resolved_path), row_count, columns


def _stream_export_to_jsonlines(
    workspace: "TaskWorkspace",
    file_path: str,
    result: CursorResult,
    batch_size: int = 1000,
) -> tuple[str, int, list[str]]:
    """Streaming export to JSON Lines (NDJSON).

    Returns:
        Tuple of (exported_file_path, row_count, column_names)
    """
    resolved_path = workspace.resolve_path(file_path, default_dir="output")

    # Get column names BEFORE iteration
    columns = list(result.keys())

    row_count = 0

    with open(resolved_path, "w", encoding="utf-8") as f:
        # Fetch in batches
        while True:
            batch = result.fetchmany(batch_size)
            if not batch:
                break

            # Convert batch to JSON lines and write
            for row in batch:
                row_dict = _row_to_dict(row)
                print(json.dumps(row_dict, ensure_ascii=False), file=f)
                row_count += 1

    return str(resolved_path), row_count, columns


def _stream_export_to_parquet(
    workspace: "TaskWorkspace",
    file_path: str,
    result: CursorResult,
    batch_size: int = 5000,
) -> tuple[str, int, list[str]]:
    """Streaming export to Parquet.

    Parquet provides excellent compression and preserves data types.

    Returns:
        Tuple of (exported_file_path, row_count, column_names)
    """
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as err:
        raise ImportError(
            f"{err}\n"
            "pyarrow is required for Parquet export. "
            "Install it with: pip install pyarrow"
        )

    resolved_path = workspace.resolve_path(file_path, default_dir="output")

    # Get column names BEFORE iteration
    columns = list(result.keys())

    row_count = 0
    writer = None

    # Fetch in batches
    while True:
        batch = result.fetchmany(batch_size)
        if not batch:
            break

        # Convert batch to dict format
        batch_dicts = [_row_to_dict(row) for row in batch]

        # Create Arrow Table from batch
        table = pa.Table.from_pylist(batch_dicts)

        # Initialize writer with schema from first batch
        if writer is None:
            writer = pq.ParquetWriter(resolved_path, table.schema)

        # Write batch to file
        writer.write_table(table)
        row_count += len(batch)

    # Close writer to finalize file
    if writer:
        writer.close()

    return str(resolved_path), row_count, columns

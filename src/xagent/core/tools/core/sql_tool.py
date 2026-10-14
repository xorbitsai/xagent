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
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field
from sqlalchemy import URL, create_engine, event, text
from sqlalchemy.engine import CursorResult, Engine, Row, make_url

from ....config import get_database_url
from ...file_ref import build_workspace_file_ref

if TYPE_CHECKING:
    from ...workspace import TaskWorkspace

logger = logging.getLogger(__name__)
PLATFORM_DATABASE_ERROR = "The platform database cannot be used by the SQL tool"
_OPAQUE_TARGET_QUERY_KEYS = frozenset(
    {"service", "odbc_connect", "dsn", "read_default_file", "read_default_group"}
)


def _canonical_backend(url: URL) -> str:
    backend = url.get_backend_name().lower()
    return "mysql" if backend == "mariadb" else backend


def _query_values(url: URL, name: str) -> tuple[str, ...]:
    for key, values in url.normalized_query.items():
        if key.lower() == name:
            return tuple(str(value) for value in values)
    return ()


def _single_query_value(url: URL, name: str) -> str | None:
    values = _query_values(url, name)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"Database URL repeats the target parameter {name!r}")
    value = values[0]
    if name in {"host", "hostaddr", "port"} and "," in value:
        raise ValueError(f"Database URL contains multiple {name!r} targets")
    return value


def _opaque_target_query_keys(url: URL) -> list[str]:
    return sorted(key for key in _OPAQUE_TARGET_QUERY_KEYS if _query_values(url, key))


def _sqlite_database_path(url: URL) -> str:
    database = unquote(url.database or ":memory:")
    uri_mode = any(
        value.strip().lower() in {"1", "on", "true", "yes"}
        for value in _query_values(url, "uri")
    )
    if uri_mode and database.lower().startswith("file:"):
        parsed_file = urlsplit(database)
        if parsed_file.netloc.lower() in {"", "localhost"}:
            database = unquote(parsed_file.path)
        else:
            database = f"//{parsed_file.netloc}{unquote(parsed_file.path)}"
    if database != ":memory:":
        database = str(Path(database).expanduser().resolve())
    return database


def _database_identity(url: URL) -> tuple[str, str, int | None, str]:
    """Return the driver's effective target, ignoring credentials."""
    backend = _canonical_backend(url)
    if backend == "sqlite":
        return backend, "", None, _sqlite_database_path(url)
    opaque_keys = _opaque_target_query_keys(url)
    if opaque_keys:
        raise ValueError(
            "Opaque database target configuration is not allowed: "
            + ", ".join(opaque_keys)
        )

    default_ports = {"postgresql": 5432, "mysql": 3306, "mssql": 1433}
    host = _single_query_value(url, "host") or url.host or ""
    if backend == "postgresql":
        host = _single_query_value(url, "hostaddr") or host
    unix_socket = _single_query_value(url, "unix_socket")
    if unix_socket:
        host = f"unix:{Path(unix_socket).expanduser().resolve()}"
        port = None
    elif host.startswith("/"):
        host = f"unix:{Path(host).expanduser().resolve()}"
        port = None
    else:
        host = host.rstrip(".").lower()
        query_port = _single_query_value(url, "port")
        port = int(query_port) if query_port is not None else url.port
        port = port or default_ports.get(backend)
    database = (
        _single_query_value(url, "dbname")
        or _single_query_value(url, "database")
        or url.database
        or ""
    ).lstrip("/")
    return backend, host, port, database


def is_platform_database_url(candidate: str | URL) -> bool:
    """Whether a candidate URL points at the application's own database."""
    try:
        candidate_url = make_url(candidate)
        platform_raw = get_database_url()
        if not platform_raw:
            return False
        return _database_identity(candidate_url) == _database_identity(
            make_url(platform_raw)
        )
    except (TypeError, ValueError):
        # This helper protects a security boundary. A malformed platform URL
        # or ambiguous target parameter must disable external SQL rather than
        # silently turning the protection off.
        return True


def validate_external_database_url(candidate: str | URL) -> URL:
    """Parse and reject connection spellings that can target the platform DB."""
    parsed = make_url(candidate)
    configured_opaque_keys = _opaque_target_query_keys(parsed)
    if configured_opaque_keys:
        # Driver-side service/DSN/config files can replace the host/database
        # without exposing their values in the URL, so their effective target
        # cannot be proven distinct from the platform database here.
        raise ValueError(
            "Opaque database target configuration is not allowed: "
            + ", ".join(configured_opaque_keys)
        )
    _database_identity(parsed)
    if is_platform_database_url(parsed):
        raise ValueError(PLATFORM_DATABASE_ERROR)
    return parsed


def _install_sqlite_authorizer(dbapi_connection: Any, _connection_record: Any) -> None:
    """Deny SQLite ATTACH exactly, without rejecting identifiers or strings."""

    def authorize(
        action_code: int,
        _parameter_one: str | None,
        _parameter_two: str | None,
        _database_name: str | None,
        _trigger_or_view: str | None,
    ) -> int:
        return (
            sqlite3.SQLITE_DENY
            if action_code == sqlite3.SQLITE_ATTACH
            else sqlite3.SQLITE_OK
        )

    dbapi_connection.set_authorizer(authorize)


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

    # Never allow the SQL tool to reuse the platform's own database. This
    # catches equivalent URL/driver spellings and effective target overrides;
    # real isolation still requires distinct credentials or network policy,
    # because DNS/container aliases cannot be proven equivalent from strings.
    return validate_external_database_url(url)


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
            - file_ref/file_id/download_url when output_file exports results
    """
    # Get connection URL from environment
    url = _get_connection_url(connection_name, connection_url)
    stmt = text(query)
    engine = create_engine(url)
    if _canonical_backend(url) == "sqlite" and isinstance(engine, Engine):
        event.listen(engine, "connect", _install_sqlite_authorizer)

    try:
        with engine.connect() as conn:
            # Check if export to file is requested first
            if output_file and workspace:
                file_ext = Path(output_file).suffix.lower()
                if file_ext == ".csv":
                    # Streaming export for large datasets. The export target
                    # is resolved -- and the engine-owned subtree refused --
                    # before the statement runs, in every branch below, so a
                    # refused export executes nothing on the database.
                    resolved_path = workspace.resolve_write_path(
                        output_file, default_dir="output"
                    )
                    result = conn.execute(stmt)
                    with workspace.auto_register_files():
                        exported_path, exported_count, columns = _stream_export_to_csv(
                            resolved_path, result
                        )
                    return _build_export_result(
                        workspace=workspace,
                        exported_path=exported_path,
                        connection_name=connection_name,
                        output_file=output_file,
                        exported_count=exported_count,
                        columns=columns,
                    )
                elif file_ext == ".parquet":
                    # Streaming export with Parquet (better compression & type preservation)
                    resolved_path = workspace.resolve_write_path(
                        output_file, default_dir="output"
                    )
                    result = conn.execute(stmt)
                    with workspace.auto_register_files():
                        (
                            exported_path,
                            exported_count,
                            columns,
                        ) = _stream_export_to_parquet(resolved_path, result)
                    return _build_export_result(
                        workspace=workspace,
                        exported_path=exported_path,
                        connection_name=connection_name,
                        output_file=output_file,
                        exported_count=exported_count,
                        columns=columns,
                    )
                elif file_ext in (".json", ".jsonl", ".ndjson"):
                    # Streaming JSON Lines (NDJSON) export
                    resolved_path = workspace.resolve_write_path(
                        output_file, default_dir="output"
                    )
                    result = conn.execute(stmt)
                    with workspace.auto_register_files():
                        (
                            exported_path,
                            exported_count,
                            columns,
                        ) = _stream_export_to_jsonlines(resolved_path, result)
                    return _build_export_result(
                        workspace=workspace,
                        exported_path=exported_path,
                        connection_name=connection_name,
                        output_file=output_file,
                        exported_count=exported_count,
                        columns=columns,
                    )
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


def _build_export_result(
    *,
    workspace: "TaskWorkspace",
    exported_path: str,
    connection_name: str,
    output_file: str,
    exported_count: int,
    columns: list[str],
) -> dict[str, Any]:
    file_ref = build_workspace_file_ref(
        workspace=workspace,
        file_path=exported_path,
    )
    payload = SQLQueryResult(
        success=True,
        rows=[],
        row_count=exported_count,
        columns=columns,
        message=f"Query executed successfully on '{connection_name}', exported {exported_count} row(s) to {output_file}",
    ).model_dump()
    payload.update(
        {
            "output_file": output_file,
            **file_ref,
            "file_ref": file_ref,
        }
    )
    return payload


def _stream_export_to_csv(
    resolved_path: Path,
    result: CursorResult,
    batch_size: int = 1000,
) -> tuple[str, int, list[str]]:
    """Streaming export to CSV.

    Returns:
        Tuple of (exported_file_path, row_count, column_names)
    """
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

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
    resolved_path: Path,
    result: CursorResult,
    batch_size: int = 1000,
) -> tuple[str, int, list[str]]:
    """Streaming export to JSON Lines (NDJSON).

    Returns:
        Tuple of (exported_file_path, row_count, column_names)
    """
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

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
    resolved_path: Path,
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

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

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
    else:
        schema = pa.schema([pa.field(str(column), pa.string()) for column in columns])
        empty_table = pa.Table.from_batches([], schema=schema)
        pq.write_table(empty_table, resolved_path)

    return str(resolved_path), row_count, columns

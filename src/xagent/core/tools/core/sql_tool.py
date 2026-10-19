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

For safety, connection URLs whose effective target cannot be proven distinct
from the platform database are rejected. This includes opaque driver-side
configuration such as ``service``/DSN files and SQLite URI mode; existing
connections using those spellings must be rewritten with explicit targets.
The URL check is a defense-in-depth measure, not a substitute for separate
platform credentials or network isolation.
"""

import csv
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field
from sqlalchemy import URL, create_engine, event, text
from sqlalchemy.engine import CursorResult, Engine, Row, make_url

from ....config import get_database_url, get_sql_worker_url, requires_secure_secrets
from ...file_ref import build_workspace_file_ref
from .sql_result import bound_sql_result
from .sql_worker_client import execute_sql_query_remote

if TYPE_CHECKING:
    from ...workspace import TaskWorkspace

logger = logging.getLogger(__name__)
PLATFORM_DATABASE_ERROR = "The platform database cannot be used by the SQL tool"
SQLITE_URI_UNSUPPORTED_ERROR = "SQLite URI mode is not supported"
_OPAQUE_TARGET_QUERY_KEYS = frozenset(
    {"service", "odbc_connect", "dsn", "read_default_file", "read_default_group"}
)
_external_database_url_policy: Callable[[URL], None] | None = None


def set_external_database_url_policy(policy: Callable[[URL], None] | None) -> None:
    """Register an application deployment's SQL URL policy (standalone: none).

    The same policy is checked before saving a connection and before runtime
    execution, including already persisted and environment-provided URLs.
    """
    global _external_database_url_policy
    _external_database_url_policy = policy


def validate_external_database_url_policy(candidate: str | URL) -> None:
    """Apply only the deployment policy, also when reporting saved connections."""
    if _external_database_url_policy is not None:
        try:
            parsed = make_url(candidate)
        except Exception as exc:
            raise ValueError("Invalid SQLAlchemy connection URL") from exc
        _external_database_url_policy(parsed)


def _canonical_backend(url: URL) -> str:
    backend = url.get_backend_name().lower()
    return "mysql" if backend == "mariadb" else backend


def _query_values(url: URL, name: str) -> tuple[str, ...]:
    for key, values in url.normalized_query.items():
        if key.lower() == name:
            return tuple(str(value) for value in values)
    return ()


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


def _effective_connect_args(url: URL) -> dict[str, Any]:
    """Return SQLAlchemy's parsed driver arguments without opening a connection."""
    _args, connect_args = url.get_dialect()().create_connect_args(url)
    return dict(connect_args)


def _split_hosts(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = (value,)
    hosts: list[str] = []
    for item in values:
        if item is None:
            continue
        hosts.extend(str(item).split(","))
    return [host.rstrip(".").strip().lower() for host in hosts if host.strip()]


def _database_identity(url: URL) -> tuple[str, tuple[str, ...], int | None, str]:
    """Return the driver's effective target, ignoring credentials.

    ``URL.host`` is not sufficient here: SQLAlchemy drivers apply query
    overrides, default ports, multi-host syntax, and PostgreSQL's omitted
    database default before opening the socket.  Ask the dialect to perform
    that parsing, but never connect while checking the identity.
    """
    backend = _canonical_backend(url)
    if backend == "sqlite":
        return backend, ("",), None, _sqlite_database_path(url)
    opaque_keys = _opaque_target_query_keys(url)
    if opaque_keys:
        raise ValueError(
            "Opaque database target configuration is not allowed: "
            + ", ".join(opaque_keys)
        )

    default_ports = {"postgresql": 5432, "mysql": 3306, "mssql": 1433}
    for key in ("host", "hostaddr"):
        if any("," in value for value in _query_values(url, key)):
            raise ValueError(f"Database URL contains multiple '{key}' targets")
    connect_args = _effective_connect_args(url)
    # libpq treats hostaddr as the actual destination when both host and
    # hostaddr are present. Compare that effective destination, rather than
    # allowing a public host name to hide a platform hostaddr override.
    raw_hosts = _split_hosts(connect_args.get("hostaddr"))
    if not raw_hosts:
        raw_hosts = _split_hosts(connect_args.get("host"))
    if not raw_hosts and url.host:
        raw_hosts.extend(_split_hosts(url.host))
    unix_socket = connect_args.get("unix_socket")
    if unix_socket:
        raw_hosts = [f"unix:{Path(str(unix_socket)).expanduser().resolve()}"]
    elif raw_hosts and raw_hosts[0].startswith("/"):
        raw_hosts = [f"unix:{Path(host).expanduser().resolve()}" for host in raw_hosts]
    hosts = tuple(sorted(set(raw_hosts or [""])))

    raw_port = connect_args.get("port")
    if raw_port is None:
        port = url.port or default_ports.get(backend)
    else:
        try:
            port = int(raw_port)
        except (TypeError, ValueError) as exc:
            raise ValueError("Database URL contains an invalid port") from exc
    database = str(
        connect_args.get("dbname")
        or connect_args.get("database")
        or url.database
        or (connect_args.get("user") if backend == "postgresql" else "")
        or ""
    ).lstrip("/")
    return backend, hosts, port, database


def is_platform_database_url(candidate: str | URL) -> bool:
    """Whether a candidate URL points at the application's own database."""
    try:
        candidate_url = make_url(candidate)
        platform_raw = get_database_url()
        if not platform_raw:
            return False
        platform_url = make_url(platform_raw)
        candidate_identity = _database_identity(candidate_url)
        platform_identity = _database_identity(platform_url)
        if candidate_identity[0] != platform_identity[0]:
            return False
        if candidate_identity[0] == "sqlite":
            candidate_path = candidate_identity[3]
            platform_path = platform_identity[3]
            if candidate_path == platform_path:
                return True
            if candidate_path == ":memory:" or platform_path == ":memory:":
                return False
            try:
                return os.path.samefile(candidate_path, platform_path)
            except (FileNotFoundError, OSError):
                return False

        # A MySQL account can qualify table names across schemas, so sharing
        # the same server is sufficient to make the platform target unsafe.
        hosts_overlap = bool(set(candidate_identity[1]) & set(platform_identity[1]))
        # A connection to the platform server is unsafe even when it names a
        # different database: the deployment credentials may be allowed to
        # create/drop databases or use server-side links. Database names are
        # therefore not an isolation boundary for the SQL tool.
        return hosts_overlap and candidate_identity[2] == platform_identity[2]
    except Exception:  # noqa: BLE001 - ambiguous platform targets must fail closed
        # This helper protects a security boundary. A malformed platform URL
        # or ambiguous target parameter must disable external SQL rather than
        # silently turning the protection off.
        return True


def validate_external_database_url(candidate: str | URL) -> URL:
    """Parse and reject connection spellings that can target the platform DB."""
    try:
        parsed = make_url(candidate)
    except Exception as exc:
        raise ValueError("Invalid SQLAlchemy connection URL") from exc
    validate_external_database_url_policy(parsed)
    if _canonical_backend(parsed) == "sqlite" and (
        _query_values(parsed, "uri")
        or (parsed.database or "").strip().lower().startswith("file:")
    ):
        # URI mode lets pysqlite reinterpret and percent-decode the path after
        # SQLAlchemy has validated it.  Reject it rather than trying to mirror
        # every SQLite URI spelling (including repeated/true-ish `uri` keys).
        raise ValueError(SQLITE_URI_UNSUPPORTED_ERROR)
    configured_opaque_keys = _opaque_target_query_keys(parsed)
    if configured_opaque_keys:
        # Driver-side service/DSN/config files can replace the host/database
        # without exposing their values in the URL, so their effective target
        # cannot be proven distinct from the platform database here.
        raise ValueError(
            "Opaque database target configuration is not allowed: "
            + ", ".join(configured_opaque_keys)
        )
    try:
        _database_identity(parsed)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Unable to determine the database target") from exc
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
    connection_name: str, connection_url: str | URL | None = None
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


def get_database_type(connection_name: str, connection_url: str | None = None) -> str:
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


def _can_stream_query(query: str) -> bool:
    """Conservatively select reads eligible for a server-side cursor.

    This is only an optimization, not SQL validation or authorization. Unknown
    forms, comments, write CTEs, SELECT INTO, and multi-statements still execute
    unchanged through an ordinary cursor; none are retried after execution.
    """
    statement = query.strip().removesuffix(";").rstrip()
    return bool(
        re.match(r"(?:SELECT|WITH|VALUES)\b", statement, re.IGNORECASE)
    ) and not bool(
        re.search(
            r";|\b(?:INSERT|UPDATE|DELETE|MERGE|INTO|CREATE|ALTER|DROP|TRUNCATE|COPY|CALL|DO)\b",
            statement,
            re.IGNORECASE,
        )
    )


def execute_sql_query(
    connection_name: str,
    query: str,
    output_file: str | None = None,
    workspace: Optional["TaskWorkspace"] = None,
    connection_url: str | None = None,
) -> dict[str, Any]:
    """Execute SQL queries on databases and return structured results.

    Args:
        connection_name: Database connection name to use
        query: SQL statement to execute
        output_file: Optional file path to export query results.
            Supported formats: .csv, .parquet, .json, .jsonl, .ndjson (relative to workspace output directory).
            When provided, query results are exported to file instead of being returned.
            In isolated-worker mode the export is reconstructed from the bounded
            JSON response, so it remains subject to the worker row/result-size
            limits and Parquet values use JSON-compatible representations.
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
    # Resolve and validate on the credential-owning Agent process before the
    # value crosses the internal worker boundary. The worker repeats URL
    # validation, but deliberately has no platform DATABASE_URL to compare.
    url = _get_connection_url(connection_name, connection_url)
    worker_url = get_sql_worker_url()
    if worker_url is not None:
        return execute_sql_query_remote(
            connection_name=connection_name,
            connection_url=url.render_as_string(hide_password=False),
            query=query,
            output_file=output_file,
            workspace=workspace,
        )
    if requires_secure_secrets():
        logger.warning(
            "SQL tool execution is running in-process because "
            "XAGENT_SQL_WORKER_URL is not configured"
        )
    return _execute_sql_query_local(
        connection_name,
        query,
        output_file,
        workspace,
        url,
    )


def _execute_sql_query_local(
    connection_name: str,
    query: str,
    output_file: str | None = None,
    workspace: Optional["TaskWorkspace"] = None,
    connection_url: str | URL | None = None,
    *,
    max_rows: int | None = None,
    max_result_bytes: int | None = None,
) -> dict[str, Any]:
    """Execute SQL in the current process.

    Production Agent processes route through :func:`execute_sql_query` and an
    isolated worker. This entry point exists for the worker subprocess and for
    backwards-compatible local installations with no worker URL configured.
    """

    url = _get_connection_url(connection_name, connection_url)
    stmt = text(query)
    engine = create_engine(url, hide_parameters=True)
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
            execution_connection = (
                conn.execution_options(stream_results=True)
                if max_rows is not None and _can_stream_query(query)
                else conn
            )
            result = execution_connection.execute(stmt)

            # Get column names from result
            if result.returns_rows:
                # Cursor metadata exists even when the query returns no rows.
                # Keep it for the response and worker-side export handoff.
                columns = list(result.keys())
                if max_rows is None:
                    rows = result.all()
                else:
                    rows = result.fetchmany(max_rows + 1)
                    if len(rows) > max_rows:
                        raise ValueError(
                            f"SQL query returned more than the {max_rows} row limit"
                        )
                row_list = [_row_to_dict(row) for row in rows]

                payload = SQLQueryResult(
                    success=True,
                    rows=row_list,
                    row_count=len(row_list),
                    columns=columns,
                    message=f"Query executed successfully on '{connection_name}', returned {len(row_list)} row(s)",
                ).model_dump()
                payload = bound_sql_result(payload, max_result_bytes)
                # RETURNING and data-modifying CTEs also return rows. Settle
                # their transaction only after limits pass, closing any named
                # cursor before commit. No automatic retry is safe for writes.
                result.close()
                conn.commit()
                return payload
            else:
                # For INSERT, UPDATE, DELETE operations
                rowcount = result.rowcount if hasattr(result, "rowcount") else 0

                payload = bound_sql_result(
                    SQLQueryResult(
                        success=True,
                        rows=[],
                        row_count=rowcount,
                        columns=[],
                        message=f"Query executed successfully on '{connection_name}', affected {rowcount} row(s)",
                    ).model_dump(),
                    max_result_bytes,
                )
                result.close()
                conn.commit()
                return payload
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

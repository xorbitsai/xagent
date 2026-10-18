"""Authenticated client for the isolated SQL execution service."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ....config import (
    get_sql_worker_query_timeout_seconds,
    get_sql_worker_token,
    get_sql_worker_url,
)
from ...file_ref import build_workspace_file_ref

if TYPE_CHECKING:
    from ...workspace import TaskWorkspace


class SqlWorkerError(RuntimeError):
    """An isolated SQL execution request failed without exposing credentials."""


def _validate_output_file(
    output_file: str | None, workspace: "TaskWorkspace | None"
) -> tuple[Path, str] | None:
    if not output_file or workspace is None:
        return None
    suffix = Path(output_file).suffix.lower()
    if suffix not in {".csv", ".parquet", ".json", ".jsonl", ".ndjson"}:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported: .csv, .parquet, "
            ".json/.jsonl/.ndjson"
        )
    # Resolve before the remote statement runs. A refused destination must not
    # execute a query and then fail while publishing its result.
    return Path(workspace.resolve_write_path(output_file, default_dir="output")), suffix


def _write_rows(
    path: Path,
    suffix: str,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return
    if suffix in {".json", ".jsonl", ".ndjson"}:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                print(json.dumps(row, ensure_ascii=False, default=str), file=stream)
        return
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            f"{exc}\npyarrow is required for Parquet export. Install it with: pip install pyarrow"
        ) from exc
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        schema = pa.schema([pa.field(column, pa.string()) for column in columns])
        table = pa.Table.from_batches([], schema=schema)
    pq.write_table(table, path)


def _validated_result(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SqlWorkerError("The SQL worker returned an invalid response")
    rows = payload.get("rows")
    columns = payload.get("columns")
    row_count = payload.get("row_count")
    if (
        payload.get("success") is not True
        or not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
        or not isinstance(columns, list)
        or not all(isinstance(column, str) for column in columns)
        or type(row_count) is not int
    ):
        raise SqlWorkerError("The SQL worker returned an invalid response")
    return dict(payload)


def execute_sql_query_remote(
    *,
    connection_name: str,
    connection_url: str,
    query: str,
    output_file: str | None,
    workspace: "TaskWorkspace | None",
) -> dict[str, Any]:
    """Execute a query through the configured isolated worker.

    The connection URL is sent only in the authenticated request body. It is
    never placed in the endpoint URL or an exception message.
    """

    worker_url = get_sql_worker_url()
    token = get_sql_worker_token()
    if worker_url is None:
        raise SqlWorkerError("The SQL worker URL is not configured")
    if token is None:
        raise SqlWorkerError("The SQL worker token is not configured")
    resolved_output = _validate_output_file(output_file, workspace)
    timeout = get_sql_worker_query_timeout_seconds() + 5
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(
                f"{worker_url}/v1/query",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "connection_name": connection_name,
                    "connection_url": connection_url,
                    "query": query,
                },
            )
    except httpx.TimeoutException as exc:
        raise SqlWorkerError(
            "The isolated SQL query timed out; its completion state is unknown "
            "and it must not be retried automatically"
        ) from exc
    except httpx.HTTPError as exc:
        raise SqlWorkerError("The isolated SQL worker is unavailable") from exc
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        safe_detail = detail if isinstance(detail, str) else "SQL worker request failed"
        raise SqlWorkerError(safe_detail)
    try:
        result = _validated_result(response.json())
    except ValueError as exc:
        raise SqlWorkerError("The SQL worker returned an invalid response") from exc

    if resolved_output is None:
        return result
    path, suffix = resolved_output
    rows = result["rows"]
    columns = result["columns"]
    assert workspace is not None
    with workspace.auto_register_files():
        _write_rows(path, suffix, rows, columns)
    file_ref = build_workspace_file_ref(workspace=workspace, file_path=str(path))
    result.update(
        {
            "rows": [],
            "message": (
                f"Query executed successfully on '{connection_name}', exported "
                f"{result['row_count']} row(s) to {output_file}"
            ),
            "output_file": output_file,
            **file_ref,
            "file_ref": file_ref,
        }
    )
    return result

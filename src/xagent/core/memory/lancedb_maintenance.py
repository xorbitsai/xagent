"""Explicit, resumable maintenance for existing LanceDB memory tables."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import pyarrow as pa  # type: ignore
from filelock import FileLock, Timeout

from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from .scope_columns import (
    SCOPE_DIMS_COLUMN,
    USER_ID_COLUMN,
    derive_scope_columns,
)

MAINTENANCE_METADATA_KEY = b"xagent.memory.scope_maintenance"
MAINTENANCE_TABLE_VERSION_KEY = b"xagent.memory.scope_maintenance_table_version"
MAINTENANCE_VERSION = b"1"
DEFAULT_BATCH_SIZE = 512
DEFAULT_LOCK_TIMEOUT = 10.0
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class MaintenanceStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INVALID_LEGACY_DATA = "invalid_legacy_data"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"


@dataclass(frozen=True)
class MaintenanceOutcome:
    status: MaintenanceStatus
    scanned_rows: int = 0
    updated_rows: int = 0
    cas_skipped_rows: int = 0
    batches_committed: int = 0
    detail: str | None = None


class MaintenanceLockTimeout(RuntimeError):
    """Raised when another process holds a table's maintenance lock too long."""


def _checkpoint(_stage: str, _batch: int | None = None) -> None:
    """Test seam for deterministic interruption; intentionally does nothing."""


def _scope_schema_error(schema: Any) -> str | None:
    expected = {
        USER_ID_COLUMN: pa.int64(),
        SCOPE_DIMS_COLUMN: pa.list_(pa.string()),
    }
    incompatible = [
        f"{name} is {schema.field(name).type}, expected {field_type}"
        for name, field_type in expected.items()
        if name in schema.names and schema.field(name).type != field_type
    ]
    if not incompatible:
        return None
    return (
        "incompatible scope column types: "
        + "; ".join(incompatible)
        + "; convert the legacy schema explicitly before retrying maintenance"
    )


def _is_complete(table: Any) -> bool:
    schema = table.schema
    names = set(schema.names)
    if not {USER_ID_COLUMN, SCOPE_DIMS_COLUMN} <= names or _scope_schema_error(schema):
        return False
    metadata = schema.field(USER_ID_COLUMN).metadata or {}
    # LanceDB versions increase monotonically on every commit. Binding completion
    # to the exact marker commit makes any later write invalidate this fast path.
    return (
        metadata.get(MAINTENANCE_METADATA_KEY) == MAINTENANCE_VERSION
        and metadata.get(MAINTENANCE_TABLE_VERSION_KEY) == str(table.version).encode()
    )


def lancedb_lock_path(connection: Any, table_name: str, scope: str) -> str:
    """Validate a local URI and derive a stable, scope-specific lock path."""
    uri = str(getattr(connection, "uri", "") or "")
    if not uri or "://" in uri or not os.path.isdir(uri):
        raise ValueError("LanceDB locking requires a writable local database URI")
    digest = hashlib.sha256(table_name.encode()).hexdigest()[:16]
    return os.path.join(uri, f".memory-{scope}-{digest}.lock")


def _lock_path(connection: Any, table_name: str) -> str:
    return lancedb_lock_path(connection, table_name, "maintenance")


def _read_rows(table: Any) -> list[dict[str, Any]]:
    names = set(table.schema.names)
    required = {"id", "metadata"}
    if not required <= names:
        missing = ", ".join(sorted(required - names))
        raise ValueError(f"memory table is missing required columns: {missing}")
    projected = ["id", "metadata"]
    projected += [c for c in (USER_ID_COLUMN, SCOPE_DIMS_COLUMN) if c in names]
    return cast(
        list[dict[str, Any]],
        table.search().select(projected).limit(None).to_arrow().to_pylist(),
    )


def _invalid_ids(rows: list[dict[str, Any]]) -> str | None:
    ids = [row["id"] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        return "legacy IDs must be non-empty strings"
    if len(ids) != len(set(ids)):
        return "legacy IDs must be unique"
    return None


def _invalid_legacy_data(rows: list[dict[str, Any]]) -> str | None:
    invalid_ids = _invalid_ids(rows)
    if invalid_ids:
        return invalid_ids
    for row in rows:
        metadata = row["metadata"]
        if metadata is not None and not isinstance(metadata, str):
            return f"metadata for legacy ID {row['id']!r} must be a string or SQL NULL"
        user_id, _scope_dims = derive_scope_columns(metadata)
        if user_id is not None and not _INT64_MIN <= user_id <= _INT64_MAX:
            return f"user_id for legacy ID {row['id']!r} must fit signed int64"
    return None


def _needs_update(row: dict[str, Any]) -> bool:
    expected = derive_scope_columns(row["metadata"])
    return (row.get(USER_ID_COLUMN), row.get(SCOPE_DIMS_COLUMN)) != expected


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_list(values: list[str]) -> str:
    if not values:
        return "arrow_cast([], 'List(Utf8)')"
    return "[" + ", ".join(_sql_string(value) for value in values) + "]"


def _sql_nullable_string(value: str | None) -> str:
    return "arrow_cast(NULL, 'Utf8')" if value is None else _sql_string(value)


def _backfill_batch(table: Any, rows: list[dict[str, Any]]) -> int:
    ids = [_sql_string(row["id"]) for row in rows]
    derived = [derive_scope_columns(row["metadata"]) for row in rows]
    position = f"cast(array_position([{', '.join(ids)}], id) as bigint)"
    metadata = ", ".join(_sql_nullable_string(row["metadata"]) for row in rows)
    expected_metadata = f"array_element([{metadata}], {position})"
    condition = (
        f"id IN ({', '.join(ids)}) AND "
        f"(metadata = {expected_metadata} OR "
        f"(metadata IS NULL AND {expected_metadata} IS NULL))"
    )
    user_ids = [
        "cast(NULL as bigint)" if user_id is None else str(user_id)
        for user_id, _scope_dims in derived
    ]
    scope_dims = [_sql_list(values) for _user_id, values in derived]
    result = table.update(
        where=condition,
        values_sql={
            USER_ID_COLUMN: f"array_element([{', '.join(user_ids)}], {position})",
            SCOPE_DIMS_COLUMN: f"array_element([{', '.join(scope_dims)}], {position})",
        },
    )
    return int(result.rows_updated)


def _mark_complete(table: Any, expected_version: int) -> int:
    marker = {
        MAINTENANCE_METADATA_KEY.decode(): MAINTENANCE_VERSION.decode(),
        MAINTENANCE_TABLE_VERSION_KEY.decode(): str(expected_version),
    }
    if hasattr(table, "update_field_metadata"):
        result = table.update_field_metadata(
            {"path": USER_ID_COLUMN, "metadata": marker}
        )
        return int(result.version)
    metadata = {
        key.decode(): value.decode()
        for key, value in (table.schema.field(USER_ID_COLUMN).metadata or {}).items()
    }
    table.replace_field_metadata(USER_ID_COLUMN, metadata | marker)
    return int(table.version)


def maintain_lancedb_memory_table(
    connection: Any,
    table_name: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> MaintenanceOutcome:
    """Backfill scope projections under an explicit, serialized admin boundary.

    Completion requires a short write-quiet window. Every later table commit
    invalidates the version-bound O(1) fast path, so a subsequent explicit call
    scans and validates the table again.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(lock_timeout, bool)
        or not isinstance(lock_timeout, (int, float))
        or not math.isfinite(lock_timeout)
        or lock_timeout <= 0
    ):
        raise ValueError("lock_timeout must be a finite positive duration")

    lock = FileLock(_lock_path(connection, table_name), timeout=lock_timeout)
    try:
        lock.acquire()
    except Timeout as exc:
        raise MaintenanceLockTimeout(
            f"Timed out after {lock_timeout}s acquiring maintenance lock for "
            f"table {table_name!r}; stop the other maintenance process and retry"
        ) from exc

    table = None
    try:
        table = connection.open_table(table_name)
        schema_error = _scope_schema_error(table.schema)
        if schema_error:
            return MaintenanceOutcome(
                MaintenanceStatus.INCOMPATIBLE_SCHEMA,
                detail=schema_error,
            )
        if _is_complete(table):
            return MaintenanceOutcome(MaintenanceStatus.COMPLETE)

        rows = _read_rows(table)
        invalid = _invalid_legacy_data(rows)
        if invalid:
            return MaintenanceOutcome(
                MaintenanceStatus.INVALID_LEGACY_DATA,
                scanned_rows=len(rows),
                detail=invalid,
            )

        names = set(table.schema.names)
        missing = [
            field
            for field in (
                pa.field(USER_ID_COLUMN, pa.int64()),
                pa.field(SCOPE_DIMS_COLUMN, pa.list_(pa.string())),
            )
            if field.name not in names
        ]
        if missing:
            table.add_columns(missing)
            _checkpoint("columns_added")

        candidates = [row for row in rows if missing or _needs_update(row)]
        updated = commits = skipped = 0
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset : offset + batch_size]
            changed = _backfill_batch(table, batch)
            updated += changed
            skipped += len(batch) - changed
            commits += 1
            _checkpoint("batch_committed", commits)

        validated_version = int(table.version)
        final_rows = _read_rows(table)
        final_invalid = _invalid_legacy_data(final_rows)
        if final_invalid:
            return MaintenanceOutcome(
                MaintenanceStatus.INVALID_LEGACY_DATA,
                scanned_rows=len(rows),
                updated_rows=updated,
                cas_skipped_rows=skipped,
                batches_committed=commits,
                detail=final_invalid,
            )
        remaining = sum(_needs_update(row) for row in final_rows)
        # Default LanceDB connections cache read snapshots, while callers may opt
        # into strong reads with read_consistency_interval=timedelta(0).
        version_changed = int(table.version) != validated_version
        if skipped or remaining or version_changed:
            return MaintenanceOutcome(
                MaintenanceStatus.INCOMPLETE,
                scanned_rows=len(rows),
                updated_rows=updated,
                cas_skipped_rows=skipped,
                batches_committed=commits,
                detail="concurrent changes require another pass",
            )

        _checkpoint("before_completion")
        expected_marker_version = validated_version + 1
        actual_marker_version = _mark_complete(table, expected_marker_version)
        # Writes always commit against the latest table version even when this
        # handle's reads are cached. A concurrent commit therefore makes the
        # marker land after V+1 and leaves it intentionally invalid/resumable.
        if (
            actual_marker_version != expected_marker_version
            or int(table.version) != expected_marker_version
        ):
            return MaintenanceOutcome(
                MaintenanceStatus.INCOMPLETE,
                scanned_rows=len(rows),
                updated_rows=updated,
                batches_committed=commits,
                detail="concurrent changes invalidated the completion marker",
            )
        return MaintenanceOutcome(
            MaintenanceStatus.COMPLETE,
            scanned_rows=len(rows),
            updated_rows=updated,
            batches_committed=commits,
        )
    finally:
        _safe_close_table(table)
        lock.release()

"""Cross-process embedding-migration freeze state for a KB collection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pyarrow as pa  # type: ignore

from ..core.exceptions import CollectionMigratingError
from ..LanceDB.schema_manager import _safe_close_table
from ..storage.factory import get_metadata_store
from ..utils.lancedb_query_utils import list_table_names
from ..utils.string_utils import escape_lancedb_string

MIGRATION_TABLE = "collection_migration_state"
IDLE = "idle"
FROZEN = "frozen"

_SCHEMA = pa.schema(
    [
        ("name", pa.string()),
        ("state", pa.string()),
        ("source_model_id", pa.string()),
        ("target_model_id", pa.string()),
        ("updated_at", pa.timestamp("us")),
    ]
)


@dataclass(frozen=True)
class CollectionMigrationState:
    """Migration freeze state stored in its own LanceDB table."""

    state: str = IDLE
    source_model_id: Optional[str] = None
    target_model_id: Optional[str] = None
    updated_at: Optional[datetime] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "source_model_id": self.source_model_id,
            "target_model_id": self.target_model_id,
            "updated_at": self.updated_at,
        }


def _row_state(row: Dict[str, Any]) -> str:
    # Fail closed: a row we cannot interpret freezes the collection.
    state = row.get("state")
    if state in (IDLE, FROZEN):
        return str(state)
    return FROZEN


def _ensure_table(conn: Any) -> None:
    if MIGRATION_TABLE in list_table_names(conn):
        return
    try:
        conn.create_table(MIGRATION_TABLE, schema=_SCHEMA)
    except Exception:
        if MIGRATION_TABLE not in list_table_names(conn):
            raise


def read_migration_state(collection_name: str) -> CollectionMigrationState:
    """Read the collection's state; missing table or row means idle."""
    conn = get_metadata_store().get_raw_connection()
    if MIGRATION_TABLE not in list_table_names(conn):
        return CollectionMigrationState()
    table = conn.open_table(MIGRATION_TABLE)
    try:
        where = f"name = '{escape_lancedb_string(collection_name)}'"
        rows = table.search().where(where).to_arrow().to_pylist()
    finally:
        _safe_close_table(table)
    if not rows:
        return CollectionMigrationState()
    row = rows[0]
    return CollectionMigrationState(
        state=_row_state(row),
        source_model_id=row.get("source_model_id"),
        target_model_id=row.get("target_model_id"),
        updated_at=row.get("updated_at"),
    )


def write_migration_state(
    collection_name: str,
    state: CollectionMigrationState,
) -> CollectionMigrationState:
    """Store ``state`` for one collection; only this module writes the table."""
    if state.state not in (IDLE, FROZEN):
        raise ValueError(f"Unsupported migration state: {state.state!r}")
    stamped = replace(
        state,
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    conn = get_metadata_store().get_raw_connection()
    _ensure_table(conn)
    row = {"name": collection_name, **stamped.to_row()}
    table = conn.open_table(MIGRATION_TABLE)
    try:
        table.merge_insert(
            ["name"]
        ).when_matched_update_all().when_not_matched_insert_all().execute([row])
    finally:
        _safe_close_table(table)
    return stamped


def _raise_if_frozen(collection_name: str, state: CollectionMigrationState) -> None:
    if state.state == FROZEN:
        raise CollectionMigratingError(
            f"Knowledge base '{collection_name}' is being upgraded to a new "
            "embedding model; writes are disabled until it finishes.",
            details={"collection": collection_name, "migration_state": state.state},
        )


def ensure_collection_writable(collection_name: str) -> None:
    """Raise before a write when the collection is frozen for migration."""
    _raise_if_frozen(collection_name, read_migration_state(collection_name))


async def ensure_collection_writable_async(collection_name: str) -> None:
    """Async guard: runs the blocking state read off the event loop."""
    state = await asyncio.to_thread(read_migration_state, collection_name)
    _raise_if_frozen(collection_name, state)

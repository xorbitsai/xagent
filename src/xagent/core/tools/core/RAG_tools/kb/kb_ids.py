"""Stable kb_ids of KB collections, one per (collection name, owner).

The rows live in the LanceDB ledger table ``kb_ids``; Milvus rows carry the
kb_id, so renaming a collection changes only the ledger.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from ..core.exceptions import DatabaseOperationError
from ..LanceDB.schema_manager import (
    KB_IDS_TABLE,
    _safe_close_table,
    ensure_kb_ids_table,
)
from ..utils.lancedb_query_utils import query_to_list
from ..utils.string_utils import escape_lancedb_string

_LOCK_TIMEOUT_SECONDS = 120


def _read_rows(conn: Any, where: str | None) -> list[dict[str, Any]]:
    ensure_kb_ids_table(conn)
    table = conn.open_table(KB_IDS_TABLE)
    try:
        query = table.search()
        if where:
            query = query.where(where)
        rows = query_to_list(query.select(["collection", "user_id", "kb_id"]).limit(-1))
    finally:
        _safe_close_table(table)
    owners = Counter((row["collection"], row["user_id"]) for row in rows)
    duplicated = [
        row for row in rows if owners[(row["collection"], row["user_id"])] > 1
    ]
    if duplicated:
        raise DatabaseOperationError(
            f"{KB_IDS_TABLE} holds several kb_ids for one collection owner: "
            f"{duplicated}"
        )
    return rows


def _collection_filter(collection: str) -> str:
    return f"collection = '{escape_lancedb_string(collection)}'"


@contextmanager
def _kb_ids_lock(conn: Any) -> Iterator[None]:
    # No fallback: without the lock one kb_id per owner is no longer guaranteed.
    path = str(Path(conn.uri) / f"{KB_IDS_TABLE}.lock")
    lock = FileLock(path, timeout=_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except (Timeout, OSError, NotImplementedError) as error:
        raise DatabaseOperationError(
            f"Cannot lock {path} to change kb_ids: {error!r}"
        ) from error
    try:
        yield
    finally:
        lock.release()


def get_or_create_kb_id(conn: Any, collection: str, user_id: int | None) -> str:
    """Return the kb_id of ``collection`` owned by ``user_id``, created on first use.

    Runs under a file lock beside the ledger and reads a freshly opened table, so
    concurrent first writes of a new collection, in any process, get one kb_id.
    """
    owner = "user_id IS NULL" if user_id is None else f"user_id = {int(user_id)}"
    where = f"{_collection_filter(collection)} AND {owner}"
    with _kb_ids_lock(conn):
        rows = _read_rows(conn, where)
        if rows:
            return str(rows[0]["kb_id"])
        kb_id = uuid.uuid4().hex
        table = conn.open_table(KB_IDS_TABLE)
        try:
            table.add(
                [
                    {
                        "collection": collection,
                        "user_id": user_id,
                        "kb_id": kb_id,
                        "created_at": datetime.now(timezone.utc),
                    }
                ]
            )
        finally:
            _safe_close_table(table)
        return kb_id


def _owner(user_id: int | None) -> str:
    return "user_id IS NULL" if user_id is None else f"user_id = {int(user_id)}"


def _scoped(collection: str, user_id: int | None, is_admin: bool) -> str:
    """Rows of ``collection`` the caller may change: a caller without a user gets none."""
    where = _collection_filter(collection)
    if is_admin:
        return where
    return f"{where} AND " + ("false" if user_id is None else _owner(user_id))


def delete_kb_ids(
    conn: Any, collection: str, *, user_id: int | None, is_admin: bool
) -> int:
    """Delete the caller's kb_id rows of ``collection`` (every owner's for an admin)."""
    where = _scoped(collection, user_id, is_admin)
    with _kb_ids_lock(conn):
        ensure_kb_ids_table(conn)
        table = conn.open_table(KB_IDS_TABLE)
        try:
            deleted = int(table.count_rows(where))
            if deleted:
                table.delete(where)
        finally:
            _safe_close_table(table)
        return deleted


def _set_collection(conn: Any, kb_ids: list[str], collection: str) -> None:
    table = conn.open_table(KB_IDS_TABLE)
    try:
        # kb_ids are uuid hex strings read from this table, safe to inline.
        quoted = ", ".join(f"'{kb_id}'" for kb_id in kb_ids)
        table.update(where=f"kb_id IN ({quoted})", values={"collection": collection})
    finally:
        _safe_close_table(table)


def rename_kb_ids(
    conn: Any, old: str, new: str, *, user_id: int | None, is_admin: bool
) -> list[str]:
    """Rename the caller's kb_id rows of ``old`` (every owner's for an admin).

    An owner's leftover row under ``new`` goes first. Returns the kb_ids, which keep their value.
    """
    with _kb_ids_lock(conn):
        rows = _read_rows(conn, _scoped(old, user_id, is_admin))
        if rows:
            table = conn.open_table(KB_IDS_TABLE)
            try:
                for row in rows:
                    table.delete(
                        f"{_collection_filter(new)} AND {_owner(row['user_id'])}"
                    )
            finally:
                _safe_close_table(table)
            _set_collection(conn, [row["kb_id"] for row in rows], new)
        return [row["kb_id"] for row in rows]


def set_kb_ids_collection(conn: Any, kb_ids: list[str], collection: str) -> None:
    """Move the rows of ``kb_ids`` to ``collection``, to undo :func:`rename_kb_ids`."""
    if kb_ids:
        with _kb_ids_lock(conn):
            _set_collection(conn, kb_ids, collection)


def read_kb_ids(
    conn: Any,
    *,
    user_id: int | None,
    is_admin: bool,
    collection: str | None = None,
) -> dict[str, list[str]]:
    """Map each collection name to the kb_ids whose rows the caller may read.

    Admins get the kb_id of every owner of a name, others only their own, and a
    caller with no user gets none.
    """
    clauses = [] if collection is None else [_collection_filter(collection)]
    if not is_admin:
        if user_id is None:
            return {}
        clauses.append(f"user_id = {int(user_id)}")
    kb_ids: dict[str, list[str]] = {}
    for row in _read_rows(conn, " AND ".join(clauses) or None):
        kb_ids.setdefault(row["collection"], []).append(row["kb_id"])
    return kb_ids

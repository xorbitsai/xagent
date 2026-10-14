"""Purge Milvus rows that no live ledger record keeps; the grace cutoff is
re-applied in each delete filter, without re-reading the ledger before a delete."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any

from ......config import get_background_job_stale_seconds
from ..core.config import DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE
from ..core.exceptions import DatabaseOperationError
from ..core.schemas import DocumentProcessingStatus
from ..LanceDB.schema_manager import _safe_close_table
from ..storage.factory import StorageFactory
from ..utils.lancedb_query_utils import _safe_count_rows, list_table_names
from .collection_handle import (
    _milvus_collections,
    _unloaded_as_empty,
    _with_loaded,
    deployment_kb_backend,
)
from .kb_ids import _read_rows
from .models import KBStorageBackend

logger = logging.getLogger(__name__)

# Soft cap on rows deleted per run, checked before each delete call; one call
# covers up to a batch of documents and may overshoot, so it does not bound a run.
MAX_DELETES_PER_RUN = 50_000
_CLEANUP_ROW_BATCH = 1_000
_TERMINAL_RUN_STATUSES = frozenset(
    {
        DocumentProcessingStatus.SUCCESS.value,
        DocumentProcessingStatus.FAILED.value,
        DocumentProcessingStatus.CANCELLED.value,
    }
)
_SCAN_FIELDS = ["chunk_id", "kb_id", "doc_id", "visible", "created_at"]


def _epoch_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    return float(value)


def _read_table(
    store: Any, table_name: str, columns: list[str]
) -> list[dict[str, Any]] | None:
    """Read a whole ledger table; ``None`` when it does not exist, raise on failure.

    ``count_rows`` and ``iter_batches`` use separate handles, so a concurrent
    delete-then-add can make one read short; the whole read is retried once.
    """
    table = None
    try:
        conn = store.get_raw_connection()
        if table_name not in list_table_names(conn):
            return None
        table = conn.open_table(table_name)
        for _attempt in range(2):
            counted = _safe_count_rows(table, None, on_error="raise")
            rows = (
                [
                    row
                    for batch in store.iter_batches(
                        table_name=table_name,
                        columns=columns,
                        user_id=None,
                        is_admin=True,
                    )
                    for row in batch.to_pylist()
                ]
                if counted
                else []
            )
            if len(rows) >= counted:
                return rows
        raise DatabaseOperationError(
            f"Read {len(rows)} of {counted} ledger {table_name} rows; "
            "aborting the Milvus cleanup"
        )
    except DatabaseOperationError:
        raise
    except Exception as error:
        raise DatabaseOperationError(
            f"Cannot read the ledger {table_name} rows: {error}"
        ) from error
    finally:
        _safe_close_table(table)


def _scan_rows(
    client: Any, name: str, cutoff: int, not_loaded: set[str]
) -> dict[tuple[str, str], bool] | None:
    """Map old ``(kb_id, doc_id)`` keys to whether any row is invisible; ``None``
    when not loaded, and it raises on any other error."""
    scanned: dict[tuple[str, str], bool] = {}
    with _unloaded_as_empty(name, not_loaded):
        iterator = client.query_iterator(
            name,
            batch_size=_CLEANUP_ROW_BATCH,
            filter=f"created_at < {cutoff}",
            output_fields=_SCAN_FIELDS,
            consistency_level="Strong",
        )
        try:
            while batch := iterator.next():
                for row in batch:
                    key = (row["kb_id"], row["doc_id"])
                    scanned[key] = scanned.get(key, False) or row["visible"] is False
        finally:
            iterator.close()
        return scanned
    return None


def _purgeable_run_keys(
    rows: list[dict[str, Any]], cutoff: int
) -> set[tuple[str, str]]:
    """Keys whose every run row is terminal and older than the cutoff.

    A missing, newer, or unparsable row keeps its key out of the set.
    """
    terminal: dict[tuple[str, str], bool] = {}
    for row in rows:
        try:
            updated_at = _epoch_seconds(row["updated_at"])
        except (TypeError, ValueError):
            updated_at = None
        key = (row["collection"], row["doc_id"])
        old_terminal = (
            row["status"] in _TERMINAL_RUN_STATUSES
            and updated_at is not None
            and updated_at < cutoff
        )
        terminal[key] = terminal.get(key, True) and old_terminal
    return {key for key, old_terminal in terminal.items() if old_terminal}


def _delete_rows(
    client: Any,
    name: str,
    kb_id: str,
    doc_ids: list[str],
    cutoff: int,
    *,
    invisible_only: bool,
) -> int:
    """Delete the selected rows of one kb_id; the cutoff spares fresh rewrites."""
    expr = "kb_id == {kb_id} and doc_id in {doc_ids} and created_at < {cutoff}"
    params: dict[str, Any] = {"kb_id": kb_id, "doc_ids": doc_ids, "cutoff": cutoff}
    if invisible_only:
        expr += " and visible == false"
    deleted = _with_loaded(
        name,
        client,
        partial(client.delete, name, filter=expr, filter_params=params),
    )
    return int(deleted.get("delete_count", 0))


def _batches(doc_ids: list[str]) -> Iterable[list[str]]:
    size = DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE
    for start in range(0, len(doc_ids), size):
        yield doc_ids[start : start + size]


@dataclass
class _RunState:
    """Mutable per-run counters; a raise must not lose completed deletes."""

    budget_left: int
    result: dict[str, int]


def _purge_collection(
    client: Any,
    name: str,
    scanned: dict[tuple[str, str], bool],
    kb_owners: dict[str, tuple[str, int | None]],
    document_keys: set[tuple[str, str]] | None,
    document_pairs: set[tuple[str, int | None]],
    run_keys: set[tuple[str, str]] | None,
    cutoff: int,
    state: _RunState,
    warned_pairs: set[tuple[str, int | None]],
) -> None:
    """Delete the leftover rows of ``name``, recording every delete in ``state``."""
    orphans: dict[str, set[str]] = {}
    missing: dict[str, set[str]] = {}
    uncommitted: dict[str, set[str]] = {}
    for (kb_id, doc_id), has_invisible in scanned.items():
        owner = kb_owners.get(kb_id)
        if owner is None:
            orphans.setdefault(kb_id, set()).add(doc_id)
            continue
        collection, user_id = owner
        if document_keys is not None:
            pair = (collection, user_id)
            if pair not in document_pairs:
                if pair not in warned_pairs:
                    warned_pairs.add(pair)
                    logger.warning(
                        "Ledger documents has no row for collection %s owner %s; "
                        "skipping the missing-document check for its kb_ids",
                        collection,
                        user_id,
                    )
            elif (collection, doc_id) not in document_keys:
                missing.setdefault(kb_id, set()).add(doc_id)
                continue
        if not has_invisible or run_keys is None:
            continue
        if (collection, doc_id) in run_keys:
            uncommitted.setdefault(kb_id, set()).add(doc_id)

    for candidates, invisible_only in (
        (orphans, False),
        (missing, False),
        (uncommitted, True),
    ):
        for kb_id in sorted(candidates):
            for batch in _batches(sorted(candidates[kb_id])):
                if state.budget_left <= 0:
                    return
                deleted = _delete_rows(
                    client,
                    name,
                    kb_id,
                    batch,
                    cutoff,
                    invisible_only=invisible_only,
                )
                state.budget_left -= deleted
                if deleted:
                    state.result[name] = state.result.get(name, 0) + deleted


def purge_leftover_milvus_rows(
    *, grace_seconds: int | None = None, now: float | None = None
) -> dict[str, int]:
    """Delete Milvus rows whose ledger record no longer keeps them; return counts.

    A LanceDB deployment returns ``{}``; a failed ledger read aborts the run.
    """
    if deployment_kb_backend() is not KBStorageBackend.MILVUS:
        return {}
    if grace_seconds is None:
        grace_seconds = 2 * get_background_job_stale_seconds()
    if now is None:
        now = time.time()
    cutoff = int(now - grace_seconds)

    from ......providers.vector_store.milvus import MilvusConnectionManager

    client = MilvusConnectionManager().get_shared_client_from_env()
    first_error: Exception | None = None
    not_loaded: set[str] = set()
    # Scan every collection before reading the ledger: the fewer ledger reads
    # between a row's scan and its delete, the smaller the stale-delete window.
    scans: dict[str, dict[tuple[str, str], bool]] = {}
    for name in _milvus_collections(client):
        try:
            scanned = _scan_rows(client, name, cutoff, not_loaded)
        except Exception as error:
            logger.error(
                "Milvus cleanup failed to scan collection %s: %s",
                name,
                error,
                exc_info=True,
            )
            if first_error is None:
                first_error = error
            continue
        if scanned is not None:
            scans[name] = scanned
    if not any(scans.values()):
        # No old row anywhere: the ledger cannot change the outcome, so do not
        # read it (and do not log the empty-kb_ids error) on an idle interval.
        if first_error is not None:
            raise first_error
        return {}

    store = StorageFactory.get_factory().get_vector_index_store()
    conn = store.get_raw_connection()
    kb_owners = {
        row["kb_id"]: (row["collection"], row["user_id"])
        for row in _read_rows(conn, None)
    }
    if not kb_owners:
        logger.error(
            "Ledger holds no kb_ids while the Milvus cleanup needs them; "
            "deleting nothing this run"
        )
        if first_error is not None:
            raise first_error
        return {}
    documents = _read_table(store, "documents", ["collection", "doc_id", "user_id"])
    runs = _read_table(
        store,
        "ingestion_runs",
        ["collection", "doc_id", "status", "updated_at"],
    )
    if documents is None:
        logger.warning(
            "Ledger documents table is missing; skipping the missing-document check"
        )
    elif not documents:
        logger.warning(
            "Ledger documents table is empty; skipping the missing-document check"
        )
    if runs is None:
        logger.warning(
            "Ledger ingestion_runs table is missing; skipping the uncommitted check"
        )
    documents = documents or []
    document_keys = {(row["collection"], row["doc_id"]) for row in documents} or None
    document_pairs = {(row["collection"], row["user_id"]) for row in documents}
    run_keys = None if runs is None else _purgeable_run_keys(runs, cutoff)

    state = _RunState(budget_left=MAX_DELETES_PER_RUN, result={})
    warned_pairs: set[tuple[str, int | None]] = set()
    for name, scanned in scans.items():
        try:
            _purge_collection(
                client,
                name,
                scanned,
                kb_owners,
                document_keys,
                document_pairs,
                run_keys,
                cutoff,
                state,
                warned_pairs,
            )
        except Exception as error:
            logger.error(
                "Milvus cleanup failed for collection %s: %s",
                name,
                error,
                exc_info=True,
            )
            if first_error is None:
                first_error = error
            continue
        if state.budget_left <= 0:
            logger.warning(
                "Milvus cleanup reached the soft %s-row cap; the next run continues",
                MAX_DELETES_PER_RUN,
            )
            break
    if first_error is not None:
        raise first_error
    return state.result

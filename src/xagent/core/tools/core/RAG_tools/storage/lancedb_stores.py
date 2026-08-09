"""LanceDB-backed implementations of storage contracts."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, cast

import lancedb
import pyarrow as pa  # type: ignore
from filelock import FileLock, Timeout
from lancedb.db import DBConnection

from xagent.providers.vector_store.lancedb import get_connection_from_env

from ..core.config import (
    DEFAULT_INDEX_POLICY,
    DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE,
    DEFAULT_VECTOR_STORE_SCAN_LIMIT,
    IndexPolicy,
)
from ..core.schemas import CollectionInfo, IndexResult
from ..LanceDB.schema_manager import ensure_documents_table
from ..utils.lancedb_query_utils import list_table_names, query_to_list
from ..utils.string_utils import (
    build_lancedb_filter_expression,
    build_user_id_filter_for_table,
    escape_lancedb_string,
)
from ..utils.user_permissions import UserPermissions
from .contracts import (
    DocumentRecord,
    FilterCondition,
    FilterExpression,
    FilterOperator,
    IngestionStatusStore,
    MainPointerStore,
    MetadataStore,
    PromptTemplateStore,
    VectorIndexStore,
    build_filter_from_dict,
)
from .lancedb_filter_utils import (
    translate_filter_expression,
)
from .logging_utils import log_audit, log_performance

logger = logging.getLogger(__name__)


def _fragment_count(table: Any) -> int:
    """Number of on-disk data files backing ``table`` (0 when unavailable)."""
    try:
        stats = table.stats()
        fragment_stats = (
            stats["fragment_stats"] if isinstance(stats, dict) else stats.fragment_stats
        )
        count = (
            fragment_stats["num_fragments"]
            if isinstance(fragment_stats, dict)
            else fragment_stats.num_fragments
        )
        return int(count)
    except Exception as e:  # noqa: BLE001
        # WARNING, not DEBUG: a lancedb API shape change would otherwise disable
        # compaction permanently with no operator-visible signal.
        logger.warning("Could not count fragments, treating as 0: %s", e)
        return 0


@contextmanager
def _compaction_lock(conn: Any, table_name: str) -> Iterator[bool]:
    """Yield whether this process took the compaction lock for one table.

    Concurrent ``optimize()`` calls each rewrite the table in full and then all
    but one lose the commit, so the losers waste a complete rewrite. A
    non-blocking lock keeps one worker doing the work; the rest skip, which
    costs nothing because the next ingestion compacts instead.

    Per table, not per database: the work is per table, so a database-wide lock
    would let one worker's collection starve another's embeddings table.

    Yields True (unlocked) when locking is unavailable -- a non-local database
    URI, or an unwritable directory -- which just restores the old behaviour.
    """
    uri = str(getattr(conn, "uri", "") or "")
    if not uri or "://" in uri or not os.path.isdir(uri):
        yield True
        return

    try:
        lock = FileLock(os.path.join(uri, f".{table_name}.compaction.lock"), timeout=0)
        lock.acquire()
    except Timeout:
        yield False
        return
    except Exception as e:  # noqa: BLE001
        logger.debug("Compaction lock unavailable, proceeding unlocked: %s", e)
        yield True
        return

    # Nothing but the caller's body runs under this try, so an exception from
    # the body propagates untouched instead of being caught and re-yielded.
    try:
        yield True
    finally:
        # release() can raise, and this is maintenance: never let it become the
        # caller's exception, or mask one already propagating from the body.
        try:
            lock.release()
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not release compaction lock for %s: %s", table_name, e)


def _stale_version_count(table: Any, cutoff: datetime) -> int:
    """How many of ``table``'s versions predate ``cutoff`` (0 when unavailable).

    Update-heavy tables never accumulate fragments -- ``merge_insert`` rewrites
    rows rather than appending -- so version history is the only signal that
    they need compacting.

    Counts only versions old enough for the next ``optimize()`` to delete, never
    the total. Totals ratchet: once past a threshold they stay past it, because
    compaction cannot remove versions still inside the retention window and each
    run commits one more. This measures exactly what a compaction would reclaim,
    so a successful run drives it back to zero.

    Cost: O(retained versions), ~43 us each; measured 3.9 ms at 100 and
    87 ms at 2000. Pruning bounds the *age* of history, not its size: versions
    inside the retention window are write-rate x retention-days, so this grows
    linearly with throughput. At 10k ingestions/day the ingestion_runs table
    holds ~140k in-window versions, i.e. ~6 s of scanning on every ingestion --
    unsolved, and the first thing to fix if a busy deployment slows down.
    The upgrade path is scanning ``<table>.lance/_versions`` directly: the
    manifests carry usable mtimes, so os.scandir gives both the count and the
    stale/fresh split for ~0.7 ms, at the cost of depending on on-disk layout.
    """
    try:
        stale = 0
        for entry in table.list_versions():
            ts = entry["timestamp"] if isinstance(entry, dict) else entry.timestamp
            # LanceDB reports naive timestamps in *local* time, not UTC, so the
            # cutoff must be local too; mixing in utcnow() skews the window by
            # the whole UTC offset.
            if ts.tzinfo is not None:
                ts = ts.astimezone().replace(tzinfo=None)
            if ts < cutoff:
                stale += 1
        return stale
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not count stale versions, treating as 0: %s", e)
        return 0


class LanceDBMetadataStore(MetadataStore):
    """LanceDB implementation for control-plane metadata operations."""

    def __init__(self) -> None:
        self._conn: Optional[DBConnection] = None

    async def _get_connection(self) -> DBConnection:
        """Open (and memoise) the connection without stalling the event loop.

        Opening a LanceDB connection is blocking I/O, so it is dispatched to a
        worker thread. ``get_raw_connection`` is the synchronous equivalent for
        callers already running off the loop.
        """
        return await asyncio.to_thread(self.get_raw_connection)

    async def get_collection(self, collection_name: str) -> CollectionInfo:
        from ..LanceDB.schema_manager import _safe_close_table

        conn = await self._get_connection()
        table = conn.open_table("collection_metadata")
        try:
            safe_name = escape_lancedb_string(collection_name)
            result = table.search().where(f"name = '{safe_name}'").to_arrow()
            if len(result) == 0:
                raise ValueError(f"Collection '{collection_name}' not found")
            # Convert Arrow table to list of dicts and take first row
            data = result.to_pylist()[0]
            return CollectionInfo.from_storage(data)
        finally:
            _safe_close_table(table)

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection entry from metadata table."""
        conn = await self._get_connection()
        await self.ensure_collection_metadata_table()

        try:
            table = conn.open_table("collection_metadata")
            safe_name = escape_lancedb_string(collection_name)
            table.delete(f"name = '{safe_name}'")
        except Exception as exc:
            logger.debug("Failed to delete collection metadata: %s", exc)

    async def rename_collection(
        self,
        old_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> None:
        """Rename ``collection_config`` and ``collection_metadata`` keys.

        See :meth:`MetadataStore.rename_collection`.
        """
        from ..LanceDB.schema_manager import (
            _safe_close_table,
            ensure_collection_config_table,
        )

        conn = await self._get_connection()
        await self.ensure_collection_metadata_table()
        ensure_collection_config_table(conn)

        safe_old = escape_lancedb_string(old_name)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if is_admin:
            config_where = f"collection = '{safe_old}'"
        elif user_id is None:
            config_where = f"collection = '{safe_old}' AND user_id = 0"
        else:
            config_where = f"collection = '{safe_old}' AND user_id = {user_id}"

        config_table = None
        try:
            config_table = conn.open_table("collection_config")
            config_table.update(
                config_where,
                {"collection": new_name, "updated_at": now},
            )
        finally:
            _safe_close_table(config_table)

        meta_table = None
        try:
            meta_table = conn.open_table("collection_metadata")
            if is_admin:
                meta_table.update(
                    f"name = '{safe_old}'",
                    {"name": new_name, "updated_at": now},
                )
            else:
                # Tenant-scoped rename invalidates the global cache for the old
                # name; admin/global list can rebuild accurate aggregate stats.
                meta_table.delete(f"name = '{safe_old}'")
        finally:
            _safe_close_table(meta_table)

    async def save_collection(self, collection: CollectionInfo) -> None:
        from ..LanceDB.schema_manager import _safe_close_table

        conn = await self._get_connection()
        await self.ensure_collection_metadata_table()

        data = collection.to_storage()
        data["updated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

        table = conn.open_table("collection_metadata")
        try:
            table.merge_insert(
                ["name"]
            ).when_matched_update_all().when_not_matched_insert_all().execute([data])
        finally:
            _safe_close_table(table)

    async def save_collections(self, collections: Sequence[CollectionInfo]) -> None:
        if not collections:
            return
        await asyncio.to_thread(self._save_collections_sync, collections)

    def _save_collections_sync(self, collections: Sequence[CollectionInfo]) -> None:
        """Blocking body of :meth:`save_collections`; must not run on the loop."""
        from ..LanceDB.schema_manager import _safe_close_table

        conn = self.get_raw_connection()
        self._ensure_collection_metadata_table_sync(conn)

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rows_by_name = OrderedDict()
        for collection in collections:
            if not collection.name:
                continue
            data = collection.to_storage()
            data["updated_at"] = data.get("updated_at") or now
            rows_by_name[collection.name] = data

        if not rows_by_name:
            return

        table = conn.open_table("collection_metadata")
        try:
            table.merge_insert(
                ["name"]
            ).when_matched_update_all().when_not_matched_insert_all().execute(
                list(rows_by_name.values())
            )
        finally:
            _safe_close_table(table)

    async def list_collections(
        self,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[CollectionInfo]:
        return await asyncio.to_thread(self._list_collections_sync, user_id, is_admin)

    def _list_collections_sync(
        self,
        user_id: Optional[int],
        is_admin: bool,
    ) -> List[CollectionInfo]:
        """Blocking body of :meth:`list_collections`; must not run on the loop."""
        conn = self.get_raw_connection()
        self._ensure_collection_metadata_table_sync(conn)
        table = conn.open_table("collection_metadata")
        rows = table.search().to_arrow().to_pylist()

        if is_admin:
            return [CollectionInfo.from_storage(row) for row in rows]

        if user_id is None:
            return []

        from ..LanceDB.schema_manager import ensure_collection_config_table

        ensure_collection_config_table(conn)
        config_table = conn.open_table("collection_config")
        config_rows = (
            config_table.search().where(f"user_id = {user_id}").to_arrow().to_pylist()
        )
        visible_names = {
            str(row.get("collection", "")).strip()
            for row in config_rows
            if str(row.get("collection", "")).strip()
        }
        if not visible_names:
            return []

        rows = [
            row for row in rows if str(row.get("name", "")).strip() in visible_names
        ]
        return [CollectionInfo.from_storage(row) for row in rows]

    async def delete_collection_metadata(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
        delete_orphaned_metadata: bool = False,
    ) -> dict[str, int]:
        from ..LanceDB.schema_manager import ensure_collection_config_table

        conn = await self._get_connection()
        await self.ensure_collection_metadata_table()
        ensure_collection_config_table(conn)

        metadata_table = conn.open_table("collection_metadata")
        safe_collection = escape_lancedb_string(collection_name)
        metadata_deleted = 0

        config_table = conn.open_table("collection_config")
        if is_admin:
            config_where = f"collection = '{safe_collection}'"
        elif user_id is None:
            config_where = f"collection = '{safe_collection}' AND user_id = 0"
        else:
            config_where = f"collection = '{safe_collection}' AND user_id = {user_id}"

        config_rows = config_table.search().where(config_where).to_arrow()
        config_deleted = len(config_rows)
        if config_deleted:
            config_table.delete(config_where)

        should_delete_metadata = is_admin
        if delete_orphaned_metadata and not is_admin:
            remaining_configs = (
                config_table.search()
                .where(f"collection = '{safe_collection}'")
                .to_arrow()
            )
            should_delete_metadata = len(remaining_configs) == 0

        if should_delete_metadata:
            metadata_rows = (
                metadata_table.search().where(f"name = '{safe_collection}'").to_arrow()
            )
            metadata_deleted = len(metadata_rows)
            if metadata_deleted:
                metadata_table.delete(f"name = '{safe_collection}'")

        return {
            "metadata_rows": metadata_deleted,
            "config_rows": config_deleted,
        }

    async def ensure_collection_metadata_table(self) -> None:
        conn = await self._get_connection()
        await asyncio.to_thread(self._ensure_collection_metadata_table_sync, conn)

    def _ensure_collection_metadata_table_sync(self, conn: DBConnection) -> None:
        """Blocking body of :meth:`ensure_collection_metadata_table`.

        Exposed synchronously so the other blocking bodies can call it inside
        their own worker thread rather than paying a second thread hop.
        """
        schema = pa.schema(
            [
                ("name", pa.string()),
                ("schema_version", pa.string()),
                ("embedding_model_id", pa.string()),
                ("embedding_dimension", pa.int32()),
                ("documents", pa.int32()),
                ("processed_documents", pa.int32()),
                ("parses", pa.int32()),
                ("chunks", pa.int32()),
                ("embeddings", pa.int32()),
                ("document_names", pa.string()),
                # Schema-only compat column; owners are derived at list time.
                ("owners", pa.string()),
                ("collection_locked", pa.bool_()),
                ("allow_mixed_parse_methods", pa.bool_()),
                ("skip_config_validation", pa.bool_()),
                ("ingestion_config", pa.string()),
                ("created_at", pa.timestamp("us")),
                ("updated_at", pa.timestamp("us")),
                ("last_accessed_at", pa.timestamp("us")),
                ("extra_metadata", pa.string()),
            ]
        )
        table_exists = False
        try:
            table_exists = "collection_metadata" in list_table_names(conn)
        except Exception as exc:  # noqa: BLE001
            logger.debug("collection_metadata existence check failed: %s", exc)
        if not table_exists:
            try:
                conn.create_table("collection_metadata", schema=schema)
            except Exception as exc:  # noqa: BLE001
                logger.debug("collection_metadata create_table no-op/failure: %s", exc)
        else:
            # Backward compatibility: old tables may miss "owners".
            from ..LanceDB.schema_manager import _safe_close_table

            table = None
            try:
                table = conn.open_table("collection_metadata")
                table_schema = getattr(table, "schema", None)
                names = getattr(table_schema, "names", None) or []
                if "owners" not in names:
                    add_fn = getattr(table, "add_columns", None)
                    if add_fn is not None:
                        add_fn({"owners": "cast('[]' as string)"})
                        logger.info(
                            "collection_metadata: added missing 'owners' column"
                        )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "collection_metadata add owners column skipped/failed: %s", exc
                )
            finally:
                _safe_close_table(table)

    async def save_collection_config(
        self,
        collection: str,
        config_json: str,
        user_id: int,
    ) -> None:
        """Save collection ingestion configuration to LanceDB."""
        from ..LanceDB.schema_manager import (
            _safe_close_table,
            ensure_collection_config_table,
        )

        conn = await self._get_connection()
        ensure_collection_config_table(conn)

        table = conn.open_table("collection_config")
        try:
            safe_collection = escape_lancedb_string(collection)

            # Delete existing config for this collection and user
            try:
                table.delete(
                    f"collection = '{safe_collection}' AND user_id = {user_id}"
                )
            except Exception as exc:
                logger.debug("Error deleting old config: %s", exc)

            # Insert new config
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            data = [
                {
                    "collection": collection,
                    "config_json": config_json,
                    "updated_at": now,
                    "user_id": user_id,
                }
            ]
            table.add(data)
        finally:
            _safe_close_table(table)

    async def get_collection_config(
        self,
        collection: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> str | None:
        """Get collection ingestion configuration from LanceDB.

        When ``is_admin`` is True, returns the most recently updated config for
        the collection across all users (tenant-agnostic listing).

        Args:
            collection: Collection name.
            user_id: User ID for multi-tenancy. None is treated as 0 for non-admin,
                and as "load all configs" for admin mode (ignored when ``is_admin``).
            is_admin: If True, omit ``user_id`` filter and resolve duplicates by
                latest ``updated_at``.

        Returns:
            Config JSON string if found, None otherwise.
        """
        return await asyncio.to_thread(
            self._get_collection_config_sync, collection, user_id, is_admin
        )

    def _get_collection_config_sync(
        self,
        collection: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> str | None:
        """Blocking body of :meth:`get_collection_config`; not for the loop."""
        from ..LanceDB.schema_manager import (
            _safe_close_table,
            ensure_collection_config_table,
        )

        table = None
        try:
            conn = self.get_raw_connection()
            ensure_collection_config_table(conn)

            table = conn.open_table("collection_config")
            safe_collection = escape_lancedb_string(collection)
            if is_admin:
                where_clause = f"collection = '{safe_collection}'"
            elif user_id is None:
                # Non-admin with user_id=None: treat as user_id=0 for backward compatibility
                where_clause = f"collection = '{safe_collection}' AND user_id = 0"
            else:
                where_clause = (
                    f"collection = '{safe_collection}' AND user_id = {user_id}"
                )
            result = table.search().where(where_clause).to_arrow()

            if len(result) == 0:
                return None
            if not is_admin or len(result) == 1:
                return str(result["config_json"][0].as_py())

            best_idx = 0
            for i in range(1, len(result)):
                cur = result["updated_at"][i].as_py()
                best = result["updated_at"][best_idx].as_py()
                if cur is not None and (best is None or cur > best):
                    best_idx = i
            return str(result["config_json"][best_idx].as_py())
        except Exception as exc:
            logger.debug("Error reading collection config: %s", exc, exc_info=True)
            raise
        finally:
            _safe_close_table(table)

    def list_collection_config_owner_ids(self, collection_name: str) -> set[int]:
        """List owners with collection_config rows for a collection."""
        from ..LanceDB.schema_manager import (
            _safe_close_table,
            ensure_collection_config_table,
        )

        owner_ids: set[int] = set()
        table = None
        try:
            conn = self.get_raw_connection()
            ensure_collection_config_table(conn)
            table = conn.open_table("collection_config")
            safe_collection = escape_lancedb_string(collection_name)
            rows = query_to_list(
                table.search()
                .where(f"collection = '{safe_collection}'")
                .select(["user_id"])
                .limit(-1)
            )
            for row in rows:
                raw_user_id = row.get("user_id")
                if raw_user_id is None:
                    continue
                try:
                    owner_ids.add(int(raw_user_id))
                except (TypeError, ValueError):
                    logger.debug(
                        "Skipping invalid collection_config user_id: %r",
                        raw_user_id,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to list collection_config owners for %s: %s",
                collection_name,
                exc,
            )
        finally:
            _safe_close_table(table)
        return owner_ids

    def get_raw_connection(self) -> DBConnection:
        """Get the underlying LanceDB connection.

        This method provides access to the raw connection for operations that
        cannot be performed through the storage abstraction. It initializes
        and caches the connection for consistency with async methods.

        Returns:
            DBConnection: The LanceDB connection object
        """
        if self._conn is None:
            self._conn = get_connection_from_env()
        return self._conn


class LanceDBVectorIndexStore(VectorIndexStore):
    """LanceDB implementation for vector/data-plane operations.

    Phase 1A Option C: Provides both sync and async methods.
    Sync methods use legacy lancedb.connect(); async methods use lancedb.connect_async().
    Both sync and async methods return native Arrow format for efficient zero-copy operations.
    """

    _TABLE_CACHE_MAXSIZE = 64

    def __init__(self) -> None:
        self._conn: Optional[DBConnection] = None
        self._async_conn: Optional[Any] = None  # AsyncConnection
        self._async_lock = asyncio.Lock()  # Protect async connection initialization
        self._table_cache: OrderedDict[str, Any] = OrderedDict()

    def _get_connection(self) -> DBConnection:
        if self._conn is None:
            self._conn = get_connection_from_env()
        return self._conn

    def _get_table(self, table_name: str, *, use_cache: bool = True) -> Any:
        """Get a table handle, optionally reusing the per-process cache."""
        from ..LanceDB.schema_manager import _safe_close_table

        if use_cache:
            cached = self._table_cache.get(table_name)
            if cached is not None:
                self._table_cache.move_to_end(table_name)
                return cached
        table = self._get_connection().open_table(table_name)
        if use_cache:
            self._table_cache[table_name] = table
            if len(self._table_cache) > self._TABLE_CACHE_MAXSIZE:
                _evicted_name, _evicted_table = self._table_cache.popitem(last=False)
                _safe_close_table(_evicted_table)
        return table

    def invalidate_table_cache(self, table_name: str | None = None) -> None:
        """Clear table cache after drop/delete to avoid stale handles.

        Cached handles are closed before removal so underlying file
        descriptors are released promptly.
        """
        from ..LanceDB.schema_manager import _safe_close_table

        if table_name is None:
            for _name, _table in list(self._table_cache.items()):
                _safe_close_table(_table)
            self._table_cache.clear()
        else:
            _table = self._table_cache.pop(table_name, None)
            _safe_close_table(_table)

    async def _get_async_connection(self) -> Any:
        """Get or create async LanceDB connection with thread-safe initialization."""
        # Fast path: return existing connection without lock
        if self._async_conn is not None:
            return self._async_conn

        # Slow path: initialize with lock to prevent race condition
        async with self._async_lock:
            # Double-check after acquiring lock
            if self._async_conn is not None:
                return self._async_conn

            # Get URI from sync connection for reuse
            sync_conn = self._get_connection()
            uri = getattr(sync_conn, "uri", None)
            if uri is None:
                # Fallback: use LANCEDB_DIR env var

                uri = os.getenv("LANCEDB_DIR", "./data/lancedb")
            self._async_conn = await lancedb.connect_async(uri)  # type: ignore[attr-defined]
            return self._async_conn

    def list_document_records(
        self,
        collection_name: Optional[str],
        user_id: Optional[int],
        is_admin: bool,
        max_results: int = DEFAULT_VECTOR_STORE_SCAN_LIMIT,
    ) -> List[DocumentRecord]:
        # Audit log for data access
        log_audit(
            "data_access",
            action="list_documents",
            user_id=user_id or -1,
            is_admin=is_admin,
            collection=collection_name,
            max_results=max_results,
        )

        # Build filter expression using common function (includes validation)
        filters = {}
        if collection_name is not None:
            filters["collection"] = collection_name

        filter_expr_obj = build_filter_from_dict(filters)
        combined_filter = self.build_filter_expression(
            filters=filter_expr_obj,
            user_id=user_id,
            is_admin=is_admin,
        )

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_connection()
        ensure_documents_table(conn)
        table = conn.open_table("documents")
        try:
            raw_records = query_to_list(
                table.search().where(combined_filter).limit(max_results)
                if combined_filter
                else table.search().limit(max_results)
            )

            records: List[DocumentRecord] = []
            for item in raw_records:
                raw_doc_id = item.get("doc_id")
                if not raw_doc_id:
                    continue
                raw_user_id = item.get("user_id")
                user_id_value = None
                if raw_user_id is not None:
                    try:
                        user_id_value = int(raw_user_id)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Skipping invalid user_id on document record %s: %r",
                            raw_doc_id,
                            raw_user_id,
                        )
                records.append(
                    DocumentRecord(
                        doc_id=str(raw_doc_id),
                        file_id=str(item["file_id"]) if item.get("file_id") else None,
                        source_path=(
                            str(item["source_path"])
                            if item.get("source_path")
                            else None
                        ),
                        user_id=user_id_value,
                    )
                )
            return records
        finally:
            _safe_close_table(table)

    def count_documents_grouped_by_collection(
        self,
        collection_names: Sequence[str],
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        """Count documents grouped by collection names with tenant filtering."""
        names = [str(name).strip() for name in collection_names if str(name).strip()]
        if not names:
            return {}

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_connection()
        ensure_documents_table(conn)
        table = conn.open_table("documents")
        try:
            collection_expr: list[FilterExpression] = [
                FilterCondition("collection", FilterOperator.EQ, name) for name in names
            ]
            base_filter = self.build_filter_expression(
                filters=collection_expr,
                user_id=user_id,
                is_admin=is_admin,
            )
            if not base_filter:
                return {}

            rows = query_to_list(
                table.search().where(base_filter).select(["collection"]).limit(-1)
            )
            counts: Dict[str, int] = {}
            for row in rows:
                name = str(row.get("collection") or "")
                if not name:
                    continue
                counts[name] = counts.get(name, 0) + 1
            return counts
        finally:
            _safe_close_table(table)

    # Deprecated: use KBCollectionHandle.rename_collection_data instead. Remove in #514.
    def rename_collection_data(
        self,
        collection_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> List[str]:
        from ..LanceDB.schema_manager import _safe_close_table

        warnings: List[str] = []
        safe_old_name = escape_lancedb_string(collection_name)
        filters = FilterCondition("collection", FilterOperator.EQ, collection_name)
        where_expr = self.build_filter_expression(
            filters=filters,
            user_id=user_id,
            is_admin=is_admin,
        )
        if not where_expr:
            where_expr = f"collection = '{safe_old_name}'"
        conn = self._get_connection()
        for table_name in self.list_table_names():
            if table_name not in {
                "documents",
                "parses",
                "chunks",
            } and not table_name.startswith("embeddings_"):
                continue
            table = None
            try:
                table = conn.open_table(table_name)
                table.update(
                    where_expr,
                    {"collection": new_name},
                )
            except Exception as exc:  # noqa: BLE001
                message = f"Failed to update '{table_name}': {exc}"
                logger.warning(message)
                warnings.append(message)
            finally:
                _safe_close_table(table)
        return warnings

    def list_table_names(self) -> Sequence[str]:
        conn = self._get_connection()
        try:
            return list_table_names(conn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list LanceDB tables: %s", exc)
            return []

    def get_vector_dimension(self, table_name: str) -> Optional[int]:
        """Get the vector dimension from a table's schema."""
        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_connection()
        table = None
        try:
            table = conn.open_table(table_name)
            schema = table.schema
            vector_field = schema.field("vector")
            if hasattr(vector_field, "type"):
                vector_type = vector_field.type
                if hasattr(vector_type, "list_size"):
                    return cast(int, vector_type.list_size)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to get vector dimension for %s: %s", table_name, exc)
        finally:
            _safe_close_table(table)
        return None

    def open_embeddings_table(self, model_tag: str) -> Tuple[Any, str]:
        """Open embeddings table with legacy fallback support.

        Tries the primary Hub ID-based table name first, then falls back
        to legacy provider-based naming if the primary doesn't exist.

        Args:
            model_tag: Model tag for the embeddings table.

        Returns:
            Tuple of (table_object, actual_table_name_used).

        Raises:
            DatabaseOperationError: If neither primary nor legacy table exists.
        """
        from ..core.exceptions import DatabaseOperationError
        from ..LanceDB.model_tag_utils import to_model_tag
        from ..utils.model_resolver import resolve_embedding_adapter

        conn = self._get_connection()
        primary_table_name = f"embeddings_{to_model_tag(model_tag)}"

        # Try primary table first
        try:
            table = conn.open_table(primary_table_name)
            return table, primary_table_name
        except Exception as primary_exc:
            last_error = primary_exc

        # Try legacy fallback
        legacy_table_name: Optional[str] = None
        try:
            cfg, _ = resolve_embedding_adapter(model_tag)
            legacy_table_name = f"embeddings_{to_model_tag(cfg.model_name)}"
        except Exception:
            legacy_table_name = None

        if legacy_table_name and legacy_table_name != primary_table_name:
            try:
                table = conn.open_table(legacy_table_name)
                logger.info(
                    "Using legacy embeddings table '%s' for model_tag='%s'. "
                    "Consider migrating to '%s' for consistency.",
                    legacy_table_name,
                    model_tag,
                    primary_table_name,
                )
                return table, legacy_table_name
            except Exception as legacy_exc:
                last_error = legacy_exc

        # Neither table exists
        error_msg = f"Embeddings table not found for model_tag='{model_tag}'"
        if primary_table_name:
            error_msg += f" (tried: '{primary_table_name}'"
            if legacy_table_name:
                error_msg += f", '{legacy_table_name}'"
            error_msg += ")"
        raise DatabaseOperationError(error_msg) from last_error

    # Deprecated: use KBCollectionHandle.delete_collection_data instead. Remove in #514.
    def delete_collection_data(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool,
        warnings_out: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Delete all data for a collection from vector-side tables."""
        return self.cascade_delete(
            target="collection",
            collection=collection_name,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=False,
            confirm=True,
        )

    def delete_document_data(
        self,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        """Delete all vector-side data for a document."""
        return self.cascade_delete(
            target="document",
            collection=collection_name,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=False,
            confirm=True,
        )

    def delete_document_record(
        self,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> int:
        """Delete only the ``documents`` table row(s) for a single document.

        Row-only counterpart to :meth:`delete_document_data`; it does not
        cascade into parses/chunks/embeddings. Reuses the cascade document
        filter so tenant scoping stays identical. Idempotent: returns 0 when
        the row or the ``documents`` table is absent.
        """
        from ..LanceDB.schema_manager import _safe_close_table

        if "documents" not in self.list_table_names():
            return 0

        conn = self._get_connection()
        filter_expr = _vis_build_document_filter(
            conn=conn,
            table_name="documents",
            collection=collection_name,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )
        table = conn.open_table("documents")
        try:
            deleted = _vis_delete_rows_by_filters(table, [filter_expr])
        finally:
            _safe_close_table(table)
        self.invalidate_table_cache("documents")
        return deleted

    def _delete_rows_for_table(
        self,
        *,
        table_name: str,
        collection_name: str,
        doc_id: str,
        extra_predicates: List[tuple[str, Optional[str]]],
        user_id: Optional[int],
        is_admin: bool,
    ) -> int:
        """Row-only delete on a parse/chunk table with optional equality narrowing.

        Builds the same tenant-safe document filter used by
        :meth:`delete_document_record`, then ANDs equality predicates for the
        supplied (column, value) pairs whose value is not ``None``. Idempotent:
        returns 0 when the table is absent.
        """
        from ..LanceDB.schema_manager import _safe_close_table
        from ..utils.string_utils import escape_lancedb_string

        if table_name not in self.list_table_names():
            return 0

        conn = self._get_connection()
        filter_expr = _vis_build_document_filter(
            conn=conn,
            table_name=table_name,
            collection=collection_name,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )
        for column, value in extra_predicates:
            if value is not None:
                filter_expr = (
                    f"{filter_expr} AND {column} == '{escape_lancedb_string(value)}'"
                )
        table = conn.open_table(table_name)
        try:
            deleted = _vis_delete_rows_by_filters(table, [filter_expr])
        finally:
            _safe_close_table(table)
        self.invalidate_table_cache(table_name)
        return deleted

    def delete_parse_records(
        self,
        collection_name: str,
        doc_id: str,
        parse_hash: Optional[str],
        user_id: Optional[int],
        is_admin: bool,
    ) -> int:
        """Delete only ``parses`` rows for a document (optionally one parse_hash)."""
        return self._delete_rows_for_table(
            table_name="parses",
            collection_name=collection_name,
            doc_id=doc_id,
            extra_predicates=[("parse_hash", parse_hash)],
            user_id=user_id,
            is_admin=is_admin,
        )

    def delete_chunk_records(
        self,
        collection_name: str,
        doc_id: str,
        parse_hash: Optional[str],
        config_hash: Optional[str],
        user_id: Optional[int],
        is_admin: bool,
    ) -> int:
        """Delete only ``chunks`` rows for a document (optionally narrowed)."""
        return self._delete_rows_for_table(
            table_name="chunks",
            collection_name=collection_name,
            doc_id=doc_id,
            extra_predicates=[
                ("parse_hash", parse_hash),
                ("config_hash", config_hash),
            ],
            user_id=user_id,
            is_admin=is_admin,
        )

    def delete_embedding_records(
        self,
        collection_name: str,
        doc_id: str,
        *,
        parse_hash: Optional[str] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        """Delete only ``embeddings_{model_tag}`` rows for a document.

        Enumerates the matching per-model embedding tables (all of them when
        ``model_tag`` is ``None``) and deletes the rows for the resolved scope
        without cascading into documents/parses/chunks. Idempotent: returns 0
        when no embeddings table matches.
        """
        from ..kb.cleanup_filters import (
            build_embedding_cleanup_filters,
            resolve_cleanup_scope,
        )
        from ..LanceDB.schema_manager import _safe_close_table

        scope = resolve_cleanup_scope(
            collection=collection_name,
            doc_id=doc_id,
            parse_hash=parse_hash,
            chunk_ids=chunk_ids,
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            require_target=False,
        )

        conn = self._get_connection()
        table_filters = build_embedding_cleanup_filters(conn, scope)
        if not table_filters:
            return 0

        deleted = 0
        for table_name, filter_exprs in table_filters.items():
            table = None
            try:
                table = conn.open_table(table_name)
                deleted += _vis_delete_rows_by_filters(table, filter_exprs)
            finally:
                _safe_close_table(table)
            self.invalidate_table_cache(table_name)
        return deleted

    # Deprecated: use KBCollectionHandle.delete_documents_data instead. Remove in #514.
    def delete_documents_data(
        self,
        collection_name: str,
        doc_ids: Sequence[str],
        user_id: Optional[int],
        is_admin: bool,
        warnings_out: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Delete vector-side data for multiple documents in batches."""
        normalized_doc_ids = sorted({str(doc_id) for doc_id in doc_ids if doc_id})
        if not normalized_doc_ids:
            return {}

        conn = self._get_connection()

        batch_size = DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE
        merged: Dict[str, int] = {}
        deleted_doc_ids: List[str] = []
        try:
            for start in range(0, len(normalized_doc_ids), batch_size):
                batch = normalized_doc_ids[start : start + batch_size]
                try:
                    counts = _vis_cascade_delete_documents(
                        conn,
                        collection=collection_name,
                        doc_ids=batch,
                        user_id=user_id,
                        is_admin=is_admin,
                        preview_only=False,
                        confirm=True,
                    )
                except Exception as exc:
                    if warnings_out is not None:
                        warnings_out.append(
                            "Failed to delete document batch "
                            f"{start // batch_size + 1}: {exc}"
                        )
                    from ..core.exceptions import DatabaseOperationError

                    raise DatabaseOperationError(
                        "Failed to delete document batch",
                        details={
                            "deleted_counts": dict(merged),
                            "deleted_doc_ids": list(deleted_doc_ids),
                            "failed_batch_index": start // batch_size + 1,
                        },
                    ) from exc
                for key, value in counts.items():
                    deleted_count = int(value)
                    if deleted_count <= 0:
                        continue
                    merged[str(key)] = merged.get(str(key), 0) + deleted_count
                deleted_doc_ids.extend(batch)

            return merged
        finally:
            # Ensure subsequent reads don't observe stale cached table handles.
            self.invalidate_table_cache()

    def _count_collections_fast(
        self,
        table_name: str,
        stat_key: str,
        stats: Dict[str, Dict[str, int]],
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        """Count records per collection using PyArrow C++ value_counts.

        Avoids iter_batches overhead and Python-level row iteration.
        """
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            table = self._get_table(table_name, use_cache=False)
            combined_filter = UserPermissions.get_user_filter(user_id, is_admin)

            if combined_filter:
                arrow_table = (
                    table.search()
                    .where(combined_filter)
                    .select(["collection"])
                    .limit(None)
                    .to_arrow()
                )
            else:
                arrow_table = (
                    table.search().select(["collection"]).limit(None).to_arrow()
                )

            if arrow_table.num_rows == 0:
                return

            counts = arrow_table.column("collection").value_counts()
            for collection, count in zip(counts.field(0), counts.field(1)):
                c = str(collection.as_py())
                if c:
                    stats.setdefault(
                        c,
                        {"documents": 0, "parses": 0, "chunks": 0, "embeddings": 0},
                    )
                    stats[c][stat_key] = count.as_py()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to fast-count table '%s': %s", table_name, exc)
        finally:
            _safe_close_table(table)

    def aggregate_collection_stats(
        self,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, Dict[str, int]]:
        """Aggregate statistics for all collections using memory-efficient batched iteration."""
        from ..LanceDB.schema_manager import (
            ensure_chunks_table,
            ensure_documents_table,
            ensure_parses_table,
        )

        stats: Dict[str, Dict[str, int]] = {}
        conn = self._get_connection()

        # Ensure tables exist
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)

        # Count documents, parses, and chunks
        self._count_collections_fast("documents", "documents", stats, user_id, is_admin)
        self._count_collections_fast("parses", "parses", stats, user_id, is_admin)
        self._count_collections_fast("chunks", "chunks", stats, user_id, is_admin)

        # Count embeddings from all embeddings_* tables
        for table_name in self.list_table_names():
            if not table_name.startswith("embeddings_"):
                continue
            self._count_collections_fast(
                table_name, "embeddings", stats, user_id, is_admin
            )

        return stats

    def aggregate_document_stats(
        self,
        collection_name: str,
        doc_id: str,
        user_id: Optional[int],
        is_admin: bool,
    ) -> Dict[str, int]:
        """Aggregate statistics for a single document."""
        from ..LanceDB.schema_manager import (
            _safe_close_table,
            ensure_chunks_table,
            ensure_documents_table,
            ensure_parses_table,
        )

        stats = {"documents": 0, "parses": 0, "chunks": 0, "embeddings": 0}
        conn = self._get_connection()

        # Ensure tables exist
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)

        safe_collection = escape_lancedb_string(collection_name)
        safe_doc_id = escape_lancedb_string(doc_id)

        base_filter = f"collection = '{safe_collection}' AND doc_id = '{safe_doc_id}'"

        def _count_table(table_name: str) -> int:
            table = None
            try:
                table = conn.open_table(table_name)
                return int(table.count_rows(base_filter))
            except Exception:  # noqa: BLE001
                return 0
            finally:
                _safe_close_table(table)

        stats["documents"] = _count_table("documents")
        stats["parses"] = _count_table("parses")
        stats["chunks"] = _count_table("chunks")

        # Count embeddings across all embeddings tables
        for table_name in self.list_table_names():
            if not table_name.startswith("embeddings_"):
                continue
            stats["embeddings"] += _count_table(table_name)

        return stats

    def create_index(self, model_tag: str, readonly: bool = False) -> IndexResult:
        """Create or check vector index for embeddings table.

        This method implements full index management logic including automatic
        index type selection based on row count and FTS index management.

        Args:
            model_tag: Model tag for the embeddings table.
            readonly: If True, don't trigger index creation.

        Returns:
            IndexResult containing status, advice, and FTS enabled state.
        """
        from ..core.config import IndexPolicy
        from ..core.schemas import IndexResult
        from ..LanceDB.model_tag_utils import to_model_tag

        # Import LanceDB index types
        try:
            from lancedb.index import IVF_HNSW_SQ, IVF_PQ  # type: ignore
        except ImportError:
            IVF_HNSW_SQ = "IVF_HNSW_SQ"
            IVF_PQ = "IVF_PQ"

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_connection()
        table_name = f"embeddings_{to_model_tag(model_tag)}"

        if readonly:
            # In readonly mode, check if FTS index exists without creating any indexes
            fts_enabled = False
            table = None
            try:
                table = conn.open_table(table_name)
                indexes = table.list_indices()
                fts_enabled = any(
                    idx.index_type == "FTS" and "text" in idx.columns for idx in indexes
                )
            except Exception as e:
                logger.debug("Unable to check FTS index status in readonly mode: %s", e)
            finally:
                _safe_close_table(table)

            return IndexResult(
                status="readonly",
                advice=f"Readonly mode - no index operations for {table_name}",
                fts_enabled=fts_enabled,
            )

        try:
            table = conn.open_table(table_name)
        except Exception as exc:
            logger.debug("Unable to open table '%s': %s", table_name, exc)
            return IndexResult(status="failed", advice=None, fts_enabled=False)

        # Use default index policy
        policy = IndexPolicy()
        vector_index_status: str = "no_index"
        vector_index_advice: Optional[str] = None

        try:
            # Get row count efficiently
            row_count = table.count_rows()

            if row_count < policy.enable_threshold_rows:
                vector_index_status = "below_threshold"
                vector_index_advice = (
                    f"Table {table_name} has {row_count} rows - below threshold "
                    f"({policy.enable_threshold_rows}) for index creation"
                )
            else:
                # Auto-select index type based on scale
                from ..core.schemas import IndexType

                if row_count >= policy.ivfpq_threshold_rows:
                    recommended_type = IndexType.IVFPQ
                else:
                    recommended_type = IndexType.HNSW

                # Check existing indexes
                indexes = table.list_indices()
                has_vector_index = any(idx.name == "vector" for idx in indexes)

                if not has_vector_index:
                    # Create index with recommended type
                    if recommended_type == IndexType.IVFPQ:
                        index_type = IVF_PQ
                        create_params = policy.ivfpq_params or {}
                    else:  # HNSW
                        index_type = IVF_HNSW_SQ
                        create_params = policy.hnsw_params or {}

                    # Merge metric with create_params
                    all_params = {
                        "metric": policy.metric.value,
                        "index_type": index_type,
                        **create_params,
                    }

                    table.create_index(**all_params)
                    vector_index_status = "index_building"
                    logger.info(
                        "Successfully created vector index for %s (type=%s, metric=%s)",
                        table_name,
                        index_type,
                        policy.metric.value,
                    )
                    if recommended_type == IndexType.IVFPQ:
                        vector_index_advice = (
                            f"IVFPQ index created for {table_name} "
                            f"({row_count} rows, using IVFPQ strategy for large-scale data), metric: {policy.metric.value}"
                        )
                    else:  # HNSW
                        vector_index_advice = (
                            f"HNSW index created for {table_name} "
                            f"({row_count} rows, using HNSW strategy for medium-scale data), metric: {policy.metric.value}"
                        )
                else:
                    vector_index_status = "index_ready"
                    vector_index_advice = f"Index ready for {table_name} ({row_count} rows), metric: {policy.metric.value}"

        except Exception as e:
            logger.error("Vector index operation failed for %s: %s", table_name, e)
            vector_index_status = "index_corrupted"
            vector_index_advice = (
                f"Vector index check failed for {table_name}: {str(e)}"
            )

        try:
            # Check actual FTS index status (not just whether we tried to create it)
            fts_enabled = False
            try:
                indexes = table.list_indices()
                fts_enabled = any(
                    idx.index_type == "FTS" and "text" in idx.columns for idx in indexes
                )
            except Exception as e:
                logger.warning("Failed to check FTS index status: %s", e)

            # FTS Index Management (if enabled)
            if policy.fts_enabled and not fts_enabled:
                try:
                    fts_params = {"with_position": True, **(policy.fts_params or {})}
                    table.create_fts_index("text", replace=True, **fts_params)
                    logger.info("Created FTS index on 'text' column for %s", table_name)
                    # Re-check FTS status after creation
                    try:
                        indexes = table.list_indices()
                        fts_enabled = any(
                            idx.index_type == "FTS" and "text" in idx.columns
                            for idx in indexes
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(
                        "FTS index creation/check failed for %s: %s", table_name, str(e)
                    )
        finally:
            _safe_close_table(table)

        return IndexResult(
            status=vector_index_status,
            advice=vector_index_advice,
            fts_enabled=fts_enabled,
        )

    # --- Index Management (Phase 1A Part 2) ---

    def should_reindex(
        self, table_name: str, total_upserted: int, policy: IndexPolicy
    ) -> bool:
        """Determine if reindex should be triggered (sync)."""
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            conn = self._get_connection()
            table = conn.open_table(table_name)

            # Immediate reindex if enabled
            if policy.enable_immediate_reindex and total_upserted > 0:
                return True

            # Batch size threshold
            if total_upserted >= policy.reindex_batch_size:
                return True

            # Smart reindex: check unindexed ratio
            if policy.enable_smart_reindex:
                try:
                    stats = table.index_stats("vector_idx")
                    if stats.num_indexed_rows > 0:
                        unindexed_ratio = (
                            stats.num_unindexed_rows / stats.num_indexed_rows
                        )
                        if unindexed_ratio > policy.reindex_unindexed_ratio_threshold:
                            return True

                    # Absolute threshold for unindexed rows
                    if stats.num_unindexed_rows > 10000:
                        return True
                except Exception as e:  # noqa: BLE001
                    logger.debug("Could not get index stats for %s: %s", table_name, e)

            return False

        except Exception as e:  # noqa: BLE001
            # warning, not error, to match the rest of this best-effort
            # maintenance chain. Nothing in production calls should_reindex --
            # compaction runs through should_compact -- so this is consistency,
            # not a hot-path fix.
            logger.warning("Failed to check reindex status for %s: %s", table_name, e)
            return False
        finally:
            _safe_close_table(table)

    def trigger_reindex(
        self, table_name: str, cleanup_older_than: Optional[timedelta] = None
    ) -> bool:
        """Compact data files, prune versions older than the retention window."""
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            # Neutral wording: compact_tables routes tables here that have no
            # index at all (collection_metadata, ingestion_runs, parses).
            logger.info("Optimizing %s", table_name)
            conn = self._get_connection()
            table = conn.open_table(table_name)
            if cleanup_older_than is None:
                cleanup_older_than = timedelta(
                    days=DEFAULT_INDEX_POLICY.version_retention_days
                )
            table.optimize(cleanup_older_than=cleanup_older_than)
            # Pruning drops the versions cached handles point at, same reason
            # the delete paths invalidate after mutating a table.
            self.invalidate_table_cache(table_name)
            logger.info("Optimized %s", table_name)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Optimize failed for %s: %s", table_name, e)
            return False
        finally:
            _safe_close_table(table)

    def should_compact(
        self, table_name: str, policy: Optional[IndexPolicy] = None
    ) -> bool:
        """Whether a table has degraded enough to be worth optimizing.

        Two independent ways a table goes bad, and a table typically hits only
        one of them:

        * fragments -- append-heavy tables (documents, chunks, embeddings) gain
          a data file per write and reads slow down with the file count;
        * stale versions -- update-heavy tables (collection_metadata) upsert via
          ``merge_insert``, which rewrites rows instead of appending, so
          fragments plateau near the row count and never reach the threshold
          while version history grows without bound and holds the disk.

        Both measure something a compaction removes, so both fall back to zero
        after a successful run and cannot latch the trigger on permanently.

        Deliberately independent of :meth:`should_reindex`: index staleness is a
        third, unrelated problem, and ``optimize()`` bundles compaction with
        index rebuilds, so reusing that predicate would rebuild large unrelated
        indices on every ingest.
        """
        from ..LanceDB.schema_manager import _safe_close_table

        policy = policy or DEFAULT_INDEX_POLICY
        cutoff = datetime.now() - timedelta(days=policy.version_retention_days)
        table = None
        try:
            table = self._get_connection().open_table(table_name)
            # Fragment stats are the cheaper probe -- ~0.2 ms at 50 fragments,
            # ~2.2 ms at 1000, so it grows too, just far slower -- hence first.
            # ``or`` short-circuits only on True, so a healthy table always pays
            # the version scan, whose cost model lives on ``_stale_version_count``
            # -- roughly linear in retained versions, and paid on every healthy
            # table because ``or`` only short-circuits when fragments settle it.
            return (
                _fragment_count(table) >= policy.compact_fragment_threshold
                or _stale_version_count(table, cutoff)
                >= policy.compact_stale_version_threshold
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not check fragmentation for %s: %s", table_name, e)
            return False
        finally:
            _safe_close_table(table)

    def compact_tables(
        self, table_names: Sequence[str], policy: Optional[IndexPolicy] = None
    ) -> List[str]:
        """Compact the degraded tables among ``table_names``; returns those done.

        Callers pass the tables they just wrote; sweeping the whole database on
        every ingest costs more than it saves. Names that do not exist are
        skipped without opening anything -- callers probe more than one spelling
        of the embeddings table, so a miss is routine, not exceptional.

        Takes a per-table advisory lock: concurrent ``optimize()`` calls all
        rewrite the table and then all but one lose the commit, so the losers
        burn a full rewrite for nothing. Whoever cannot take a table's lock
        skips that table -- free, the next ingestion compacts instead -- and
        moves on to the rest, so one busy table cannot starve the others.

        Best-effort maintenance: individual failures are logged, never raised.
        """
        if not table_names:
            return []

        policy = policy or DEFAULT_INDEX_POLICY
        cleanup_older_than = timedelta(days=policy.version_retention_days)
        try:
            conn = self._get_connection()
            existing = set(list_table_names(conn))
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not reach the database, skipping compaction: %s", e)
            return []

        candidates = [name for name in table_names if name in existing]
        if not candidates:
            # list_table_names() returns [] rather than raising when it cannot
            # read the listing, so without this compaction would switch itself
            # off permanently and silently.
            logger.warning(
                "None of the tables to compact were found in the database "
                "listing (asked for %s); skipping compaction",
                ", ".join(table_names),
            )
            return []

        compacted: List[str] = []
        for name in candidates:
            with _compaction_lock(conn, name) as acquired:
                if not acquired:
                    logger.debug("%s is being compacted elsewhere; skipping", name)
                    continue
                if self.should_compact(name, policy) and self.trigger_reindex(
                    name, cleanup_older_than=cleanup_older_than
                ):
                    compacted.append(name)
        return compacted

    async def should_reindex_async(
        self, table_name: str, total_upserted: int, policy: IndexPolicy
    ) -> bool:
        """Async version of should_reindex.

        Note: Current implementation uses sync operations under the hood.
        True async I/O will be added in Phase 1B with RDB backend.
        """
        # Delegate to sync implementation for now
        return self.should_reindex(table_name, total_upserted, policy)

    async def trigger_reindex_async(self, table_name: str) -> bool:
        """Async version of trigger_reindex.

        Delegates to the sync path, so it also compacts data files and prunes
        versions past the retention window -- not just an index rebuild. Uses
        the backend default window, having no ``cleanup_older_than`` parameter.

        Note: Current implementation uses sync operations under the hood.
        True async I/O will be added in Phase 1B with RDB backend.
        """
        # Delegate to sync implementation for now
        return self.trigger_reindex(table_name)

    def migrate_embeddings_table(
        self,
        model_id: str,
        batch_size: int = 1000,
    ) -> dict[str, Any]:
        """Migrate legacy embeddings table to Hub ID-based naming.

        This method copies data from a legacy table (embeddings_{model_name})
        to a new Hub ID-based table (embeddings_{hub_id}), rewriting the
        per-row ``model`` field to the Hub model ID.

        Args:
            model_id: Hub model ID to migrate.
            batch_size: Number of rows to copy per batch.

        Returns:
            Dictionary with migration results.
        """
        from ..utils import migration_utils

        return migration_utils.migrate_embeddings_table(
            model_id=model_id,
            batch_size=batch_size,
            conn=self._get_connection(),
        )

    def get_raw_connection(self) -> DBConnection:
        return self._get_connection()

    def iter_batches(
        self,
        table_name: str,
        columns: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Iterator[Any]:
        """Iterate over table data in batches.

        Yields backend-specific batch objects (e.g., PyArrow RecordBatch).
        """
        from ..LanceDB.schema_manager import (
            ensure_chunks_table,
            ensure_documents_table,
            ensure_parses_table,
        )

        conn = self._get_connection()

        # Ensure table exists based on name
        if table_name == "documents":
            ensure_documents_table(conn)
        elif table_name == "parses":
            ensure_parses_table(conn)
        elif table_name == "chunks":
            ensure_chunks_table(conn)

        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            table = self._get_table(table_name, use_cache=False)
        except Exception as exc:
            logger.debug("Unable to open table '%s': %s", table_name, exc)
            return

        try:
            # Build filter expression using common function (includes validation)
            combined_filter = None
            if filters:
                filter_expr_obj = build_filter_from_dict(filters)
                combined_filter = self.build_filter_expression(
                    filters=filter_expr_obj,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            else:
                # Just apply user filter
                combined_filter = UserPermissions.get_user_filter(user_id, is_admin)

            # Helper method to select columns from a batch
            def _select_columns(batch: Any, cols: Optional[Sequence[str]]) -> Any:
                if cols is None:
                    return batch
                arrays = []
                names = []
                for col_name in cols:
                    idx = batch.schema.get_field_index(col_name)
                    if idx != -1:
                        arrays.append(batch.column(idx))
                        names.append(col_name)
                if not arrays:
                    return pa.RecordBatch.from_arrays([], [])
                return pa.RecordBatch.from_arrays(arrays, names)

            # Preferred path: streaming batches directly from LanceDB
            try:
                if combined_filter:
                    for raw_batch in table.to_batches(
                        filter=combined_filter, batch_size=batch_size
                    ):
                        batch = raw_batch
                        if columns is not None:
                            batch = _select_columns(batch, columns)
                        if batch.num_rows > 0:
                            yield batch
                else:
                    for raw_batch in table.to_batches(batch_size=batch_size):
                        batch = raw_batch
                        if columns is not None:
                            batch = _select_columns(batch, columns)
                        if batch.num_rows > 0:
                            yield batch
                return
            except Exception as exc:
                logger.debug(
                    "Batch streaming unavailable for table '%s': %s", table_name, exc
                )

            # Arrow fallback: materialize table as Arrow then iterate
            try:
                # Note: LanceDB's to_arrow() doesn't accept filter parameter
                # Use search().where().to_arrow() instead
                if combined_filter:
                    arrow_table = table.search().where(combined_filter).to_arrow()
                else:
                    arrow_table = table.to_arrow()
            except Exception as exc:
                logger.debug(
                    "Unable to read table '%s' via to_arrow(): %s", table_name, exc
                )
                return

            if columns is not None:
                try:
                    arrow_table = arrow_table.select(columns)
                except Exception as exc:
                    logger.debug(
                        "Table '%s' missing expected columns %s: %s",
                        table_name,
                        columns,
                        exc,
                    )
                    return

            for batch in arrow_table.to_batches(max_chunksize=batch_size):
                if batch.num_rows > 0:
                    yield batch
        finally:
            _safe_close_table(table)

    def count_rows(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        """Count rows in a table with optional filters.

        Raises:
            DatabaseOperationError: If table cannot be opened or count fails.
        """
        from ..core.exceptions import DatabaseOperationError

        try:
            table = self._get_table(table_name)
        except Exception as exc:
            raise DatabaseOperationError(
                f"Failed to open table '{table_name}': {exc}"
            ) from exc

        # Build filter expression using common function (includes validation)
        backend_filter = None
        if filters:
            filter_expr_obj = build_filter_from_dict(filters)
            backend_filter = self.build_filter_expression(
                filters=filter_expr_obj,
                user_id=user_id,
                is_admin=is_admin,
            )
        else:
            # Just apply user filter
            backend_filter = UserPermissions.get_user_filter(user_id, is_admin)

        try:
            if backend_filter:
                return int(table.count_rows(backend_filter))
            return int(table.count_rows())
        except Exception as exc:
            raise DatabaseOperationError(
                f"Failed to count rows in table '{table_name}': {exc}"
            ) from exc

    def aggregate_document_counts(
        self,
        table_name: str,
        doc_id_column: str,
        collection_name: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Dict[str, int]:
        """Aggregate records per document for a specific table."""
        counts: Dict[str, int] = defaultdict(int)

        for batch in self.iter_batches(
            table_name=table_name,
            columns=["collection", doc_id_column],
            user_id=user_id,
            is_admin=is_admin,
        ):
            collection_idx = batch.schema.get_field_index("collection")
            doc_idx = batch.schema.get_field_index(doc_id_column)

            if collection_idx == -1 or doc_idx == -1:
                continue

            collection_array = batch.column(collection_idx)
            doc_array = batch.column(doc_idx)

            for idx in range(batch.num_rows):
                collection_raw = collection_array[idx].as_py()
                if not collection_raw or str(collection_raw) != collection_name:
                    continue
                doc_raw = doc_array[idx].as_py()
                if not doc_raw:
                    continue
                counts[str(doc_raw)] += 1

        return dict(counts)

    def build_filter_expression(
        self,
        filters: Optional[FilterExpression],
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Optional[str]:
        """Convert abstract filter expression to LanceDB SQL syntax."""
        if not filters:
            # Still apply user filter for multi-tenancy
            return UserPermissions.get_user_filter(user_id, is_admin)

        backend_filter = translate_filter_expression(filters)

        # Combine with user filter
        user_filter = UserPermissions.get_user_filter(user_id, is_admin)
        if user_filter:
            return f"({backend_filter}) AND ({user_filter})"
        return backend_filter

    def upsert_documents(self, records: List[Dict[str, Any]]) -> None:
        """Upsert document records to LanceDB.

        Args:
            records: List of document record dictionaries to upsert.
        """
        from ..LanceDB.schema_manager import ensure_documents_table

        if not records:
            return

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_connection()
        ensure_documents_table(conn)
        table = conn.open_table("documents")
        try:
            # Use merge_insert for efficient upsert
            table.merge_insert(
                ["collection", "doc_id"]
            ).when_matched_update_all().when_not_matched_insert_all().execute(records)
        finally:
            _safe_close_table(table)
        self.invalidate_table_cache("documents")

    def upsert_parses(self, records: List[Dict[str, Any]]) -> None:
        """Upsert parse records to LanceDB.

        Args:
            records: List of parse record dictionaries to upsert.
        """
        from ..LanceDB.schema_manager import ensure_parses_table

        if not records:
            return

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_connection()
        ensure_parses_table(conn)
        table = conn.open_table("parses")
        try:
            # Use merge_insert for efficient upsert
            table.merge_insert(
                ["collection", "doc_id", "parse_hash"]
            ).when_matched_update_all().when_not_matched_insert_all().execute(records)
        finally:
            _safe_close_table(table)
        self.invalidate_table_cache("parses")

    def upsert_chunks(self, records: List[Dict[str, Any]]) -> None:
        """Upsert chunk records to LanceDB.

        Args:
            records: List of chunk record dictionaries to upsert.
        """
        from ..LanceDB.schema_manager import ensure_chunks_table

        if not records:
            return

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_connection()
        ensure_chunks_table(conn)
        table = conn.open_table("chunks")
        try:
            # Use merge_insert for efficient upsert
            table.merge_insert(
                ["collection", "doc_id", "parse_hash", "chunk_id"]
            ).when_matched_update_all().when_not_matched_insert_all().execute(records)
        finally:
            _safe_close_table(table)
        self.invalidate_table_cache("chunks")

    def upsert_embeddings(self, model_tag: str, records: List[Dict[str, Any]]) -> None:
        """Upsert embedding records to LanceDB with fallback pattern.

        Args:
            model_tag: Model tag for the embeddings table.
            records: List of embedding record dictionaries to upsert.

        Raises:
            Exception: If both merge_insert and add() methods fail.
        """
        from ..LanceDB.model_tag_utils import to_model_tag
        from ..LanceDB.schema_manager import ensure_embeddings_table
        from .merge_errors import is_non_recoverable_merge_error

        if not records:
            return

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_connection()
        table_name = f"embeddings_{to_model_tag(model_tag)}"

        # Infer vector dimension from first record
        vector_dim = None
        if records and "vector" in records[0]:
            vector = records[0]["vector"]
            if isinstance(vector, (list, tuple)):
                vector_dim = len(vector)

        ensure_embeddings_table(conn, to_model_tag(model_tag), vector_dim=vector_dim)
        table = conn.open_table(table_name)

        try:
            # Try merge_insert first (preferred method for upserts)
            table.merge_insert(
                ["collection", "doc_id", "chunk_id"]
            ).when_matched_update_all().when_not_matched_insert_all().execute(records)
        except Exception as merge_error:
            if is_non_recoverable_merge_error(merge_error):
                # Log critical error and re-raise without fallback
                logger.error(
                    "merge_insert failed with non-recoverable error (error_type=%s): %s. "
                    "This may indicate schema mismatch or data corruption. "
                    "Not attempting fallback to add() method.",
                    type(merge_error).__name__,
                    merge_error,
                )
                raise

            # For recoverable errors (e.g., temporary issues, network errors), attempt fallback
            logger.warning(
                "merge_insert failed (error_type=%s): %s; "
                "attempting fallback to add() method",
                type(merge_error).__name__,
                merge_error,
            )
            try:
                # Use dict list directly (LanceDB add() accepts list-of-dict)
                table.add(records)
                logger.info(
                    "Successfully used add() fallback for %d embeddings after merge_insert failure",
                    len(records),
                )
            except Exception as add_error:
                logger.error(
                    "Fallback add() also failed: %s. "
                    "Both merge_insert and add() methods failed.",
                    add_error,
                )
                raise
        finally:
            _safe_close_table(table)
        self.invalidate_table_cache(table_name)

    # --- Sync search methods (Phase 1A Option C) ---

    def search_vectors(
        self,
        table_name: str,
        query_vector: List[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute vector search using sync LanceDB API.

        Returns native Arrow format converted to list of dicts.
        """
        # Log search parameters for performance tracking
        log_performance(
            "search_vectors_start",
            top_k=top_k,
            vector_dim=len(query_vector),
            table_name=table_name,
            has_filters=filters is not None,
        )

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_connection()

        # Open table (no legacy fallback at abstraction layer - handled by caller)
        try:
            table = conn.open_table(table_name)
        except Exception as exc:
            logger.debug("Unable to open table '%s': %s", table_name, exc)
            return []

        try:
            # Build filter expression
            backend_filter = self.build_filter_expression(
                filters, user_id=user_id, is_admin=is_admin
            )

            # Build search query
            search_query = table.search(
                query_vector,
                vector_column_name=vector_column_name,
            )

            if backend_filter:
                search_query = search_query.where(backend_filter)

            search_query = search_query.limit(top_k)

            try:
                # Use query_to_list for three-tier fallback (to_arrow, to_list, to_pandas)
                raw_results = query_to_list(search_query)

                # Log performance metric
                log_performance(
                    "search_vectors_complete",
                    result_count=len(raw_results),
                    table_name=table_name,
                )
                return raw_results

            except Exception as exc:
                logger.error("Sync vector search failed: %s", exc)
                return []
        finally:
            _safe_close_table(table)

    # --- Async method implementations (Phase 1A Option C) ---

    async def search_vectors_async(
        self,
        table_name: str,
        query_vector: List[float],
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        vector_column_name: str = "vector",
    ) -> List[Dict[str, Any]]:
        """Execute vector search using async LanceDB API.

        Returns native Arrow format converted to list of dicts.
        """
        # Log search parameters for performance tracking
        log_performance(
            "search_vectors_start",
            top_k=top_k,
            vector_dim=len(query_vector),
            table_name=table_name,
            has_filters=filters is not None,
        )

        async_conn = await self._get_async_connection()
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            table = await async_conn.open_table(table_name)

            # Build filter expression
            backend_filter = self.build_filter_expression(
                filters, user_id=None, is_admin=False
            )

            # Build search query
            search_query = table.search(
                query_vector,
                vector_column_name=vector_column_name,
            )

            if backend_filter:
                search_query = search_query.where(backend_filter)

            search_query = search_query.limit(top_k)

            # Async search returns Arrow table
            results_table = await search_query.to_arrow()

            # Convert Arrow to list of dicts
            results = []
            for batch in results_table.to_batches():
                for i in range(batch.num_rows):
                    row = {}
                    for j in range(batch.num_columns):
                        col_name = batch.schema.names[j]
                        col_array = batch.column(j)
                        value = col_array[i].as_py()
                        row[col_name] = value
                    results.append(row)

            # Log performance metric
            log_performance(
                "search_vectors_complete",
                result_count=len(results),
                table_name=table_name,
            )
            return results
        except Exception as exc:
            logger.error("Async vector search failed: %s", exc)
            return []
        finally:
            _safe_close_table(table)

    async def search_fts_async(
        self,
        table_name: str,
        query_text: str,
        *,
        top_k: int,
        filters: Optional[FilterExpression] = None,
        text_column_name: str = "text",
    ) -> List[Dict[str, Any]]:
        """Execute full-text search using async LanceDB FTS API.

        Returns native Arrow format converted to list of dicts.
        """
        async_conn = await self._get_async_connection()
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            table = await async_conn.open_table(table_name)

            # Build filter expression
            backend_filter = self.build_filter_expression(
                filters, user_id=None, is_admin=False
            )

            # Build FTS search query
            # Note: LanceDB async API supports query_type="fts"
            search_query = table.search(
                query_text,
                query_type="fts",
            )

            if backend_filter:
                search_query = search_query.where(backend_filter)

            search_query = search_query.limit(top_k)

            # Async FTS search returns Arrow table
            results_table = await search_query.to_arrow()

            # Convert Arrow to list of dicts
            results = []
            for batch in results_table.to_batches():
                for i in range(batch.num_rows):
                    row = {}
                    for j in range(batch.num_columns):
                        col_name = batch.schema.names[j]
                        col_array = batch.column(j)
                        value = col_array[i].as_py()
                        row[col_name] = value
                    results.append(row)
            return results

        except Exception as exc:
            logger.error("Async FTS search failed: %s", exc)
            return []
        finally:
            _safe_close_table(table)

    async def iter_batches_async(
        self,
        table_name: str,
        columns: Optional[Sequence[str]] = None,
        batch_size: int = 1000,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> Any:  # # Returns AsyncIterator (async generator), see contract for details
        """Iterate over table data in batches using async LanceDB API.

        Yields PyArrow RecordBatch objects (native async format).
        """
        # Log batch iteration parameters for performance tracking
        log_performance(
            "iter_batches_start",
            table_name=table_name,
            batch_size=batch_size,
            columns_provided=columns is not None,
            has_filters=filters is not None,
        )

        async_conn = await self._get_async_connection()
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            table = await async_conn.open_table(table_name)

            # Build filter expression using common function (includes validation)
            combined_filter = None
            if filters:
                filter_expr_obj = build_filter_from_dict(filters)
                combined_filter = self.build_filter_expression(
                    filters=filter_expr_obj,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            else:
                # Just apply user filter
                combined_filter = UserPermissions.get_user_filter(user_id, is_admin)

            # Helper method to select columns from a batch
            def _select_columns(batch: Any, cols: Optional[Sequence[str]]) -> Any:
                if cols is None:
                    return batch
                arrays = []
                names = []
                for col_name in cols:
                    idx = batch.schema.get_field_index(col_name)
                    if idx != -1:
                        arrays.append(batch.column(idx))
                        names.append(col_name)
                if not arrays:
                    return pa.RecordBatch.from_arrays([], [])
                return pa.RecordBatch.from_arrays(arrays, names)

            # Use LanceDB async to_batches() with column projection for efficiency
            # Note: LanceDB to_batches supports columns parameter to avoid reading all data
            if combined_filter:
                async for batch in table.to_batches(
                    filter=combined_filter,
                    batch_size=batch_size,
                    columns=columns,  # Pass columns directly to avoid reading all data
                ):
                    if batch.num_rows > 0:
                        yield batch
            else:
                async for batch in table.to_batches(
                    batch_size=batch_size,
                    columns=columns,  # Pass columns directly to avoid reading all data
                ):
                    if batch.num_rows > 0:
                        yield batch
        except Exception as exc:
            logger.debug(
                "Async batch iteration failed for table '%s': %s", table_name, exc
            )
        finally:
            _safe_close_table(table)

    async def count_rows_async(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> int:
        """Count rows in a table with optional filters using async LanceDB API."""
        async_conn = await self._get_async_connection()
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            table = await async_conn.open_table(table_name)

            # Build filter expression using common function (includes validation)
            combined_filter = None
            if filters:
                filter_expr_obj = build_filter_from_dict(filters)
                combined_filter = self.build_filter_expression(
                    filters=filter_expr_obj,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            else:
                # Just apply user filter
                combined_filter = UserPermissions.get_user_filter(user_id, is_admin)

            if combined_filter:
                count = int(await table.count_rows(combined_filter))
            else:
                count = int(await table.count_rows())

            # Log performance metric
            log_performance(
                "count_rows_complete",
                table_name=table_name,
                row_count=count,
                has_filter=combined_filter is not None,
            )
            return count
        except Exception as exc:
            logger.debug("Failed to count rows in '%s': %s", table_name, exc)
            return 0
        finally:
            _safe_close_table(table)

    async def get_vector_dimension_async(self, table_name: str) -> Optional[int]:
        """Get the vector dimension from a table's schema (async).

        Note: LanceDB schema operations are sync-only, so this wraps the sync
        implementation. True async I/O will be added in Phase 1B with RDB backend.
        """
        # LanceDB schema operations don't have async variants, use sync
        return self.get_vector_dimension(table_name)

    async def upsert_documents_async(self, records: List[Dict[str, Any]]) -> None:
        """Upsert document records using async LanceDB API."""
        from ..LanceDB.schema_manager import ensure_documents_table

        if not records:
            return

        # Log upsert operation parameters for performance tracking
        log_performance(
            "upsert_documents_start", record_count=len(records), table="documents"
        )

        async_conn = await self._get_async_connection()

        # Note: ensure_documents_table uses sync connection - may need async variant
        # For now, reuse sync connection for table creation
        sync_conn = self._get_connection()
        ensure_documents_table(sync_conn)

        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            table = await async_conn.open_table("documents")

            # Use merge_insert for efficient upsert
            await (
                table.merge_insert(["collection", "doc_id"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records)
            )
        finally:
            _safe_close_table(table)

    async def upsert_chunks_async(self, records: List[Dict[str, Any]]) -> None:
        """Upsert chunk records using async LanceDB API."""
        from ..LanceDB.schema_manager import ensure_chunks_table

        if not records:
            return

        async_conn = await self._get_async_connection()

        # Reuse sync connection for table creation
        sync_conn = self._get_connection()
        ensure_chunks_table(sync_conn)

        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            table = await async_conn.open_table("chunks")

            # Use merge_insert for efficient upsert
            await (
                table.merge_insert(["collection", "doc_id", "parse_hash", "chunk_id"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records)
            )
        finally:
            _safe_close_table(table)

    async def upsert_embeddings_async(
        self, model_tag: str, records: List[Dict[str, Any]]
    ) -> None:
        """Upsert embedding records using async LanceDB API.

        Note: This method uses merge_insert without fallback for simplicity.
        For production use with error recovery, use the sync upsert_embeddings method.
        """
        from ..LanceDB.model_tag_utils import to_model_tag
        from ..LanceDB.schema_manager import ensure_embeddings_table

        if not records:
            return

        async_conn = await self._get_async_connection()
        sync_conn = self._get_connection()

        table_name = f"embeddings_{to_model_tag(model_tag)}"

        # Infer vector dimension from first record
        vector_dim = None
        if records and "vector" in records[0]:
            vector = records[0]["vector"]
            if isinstance(vector, (list, tuple)):
                vector_dim = len(vector)

        ensure_embeddings_table(
            sync_conn, to_model_tag(model_tag), vector_dim=vector_dim
        )
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            table = await async_conn.open_table(table_name)

            # Use merge_insert for efficient upsert
            await (
                table.merge_insert(["collection", "doc_id", "chunk_id"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records)
            )
        finally:
            _safe_close_table(table)

    # --- Version candidate listing (#513 Task 4) ---

    @staticmethod
    def _vis_query_table(
        connection: Any,
        table_name: str,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Query a table with given filters; return List[Dict]. Returns [] if table missing.

        Uses ``skip_user_filter=True`` because this is an internal store-level
        query with no user authentication context (admin-level candidate scan).
        """
        from ..LanceDB.schema_manager import _safe_close_table as _sct

        if table_name not in list_table_names(connection):
            return []
        tbl = None
        try:
            tbl = connection.open_table(table_name)
            filter_expr = build_lancedb_filter_expression(
                filters, skip_user_filter=True
            )
            return query_to_list(tbl.search().where(filter_expr))
        finally:
            _sct(tbl)

    @staticmethod
    def _vis_generate_semantic_id(
        step_type_str: str, technical_id: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a semantic ID from technical ID and parameters."""
        hash_prefix = technical_id[:8] if technical_id else "unknown"
        if step_type_str == "parse":
            method = params.get("parse_method", "unknown") if params else "unknown"
            return f"parse_{method}_{hash_prefix}"
        elif step_type_str == "chunk":
            strategy = params.get("chunk_strategy", "unknown") if params else "unknown"
            size = params.get("chunk_size", "unknown") if params else "unknown"
            return f"chunk_{strategy}_{size}_{hash_prefix}"
        elif step_type_str == "embed":
            model = params.get("model", "unknown") if params else "unknown"
            return f"embed_{model}_{hash_prefix}"
        else:
            return f"{step_type_str}_{hash_prefix}"

    @staticmethod
    def _vis_method_from_parser(parser: Any) -> str:
        """Recover the real parse method from the ``parser`` column value.

        The ``parses`` table has no top-level ``parse_method`` column; the
        method is only carried by the ``parser`` column as
        ``local:{method}@v1.0.0`` (written by ``parse_document`` and
        ``KBCollectionHandle.write_parse``). This reverse-parse therefore
        depends on that write-side format — if the ``parser`` string format
        changes, update this helper. Any non-conforming value yields
        ``"unknown"``.
        """
        if not isinstance(parser, str) or not parser.startswith("local:"):
            return "unknown"
        method = parser[len("local:") :].split("@", 1)[0]
        return method or "unknown"

    def _vis_get_parse_candidates(
        self, connection: Any, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get parse candidates from the database."""
        result = self._vis_query_table(connection, "parses", filters)
        if not result:
            return []
        parse_candidates: List[Dict[str, Any]] = []
        for row in result:
            parser = row.get("parser", "unknown")
            parse_method = self._vis_method_from_parser(parser)
            merged_params: Dict[str, Any] = {
                "parse_method": parse_method,
                "parser": parser,
            }
            semantic_id = self._vis_generate_semantic_id(
                "parse", row["parse_hash"], merged_params
            )
            stats: Dict[str, Any] = {
                "paragraphs_count": 0,
                "elapsed_ms": 0,
                "parse_method": parse_method,
                "parser": parser,
            }
            parse_candidates.append(
                {
                    "semantic_id": semantic_id,
                    "technical_id": row["parse_hash"],
                    "params_brief": merged_params,
                    "stats": stats,
                    "state": "candidate",
                    "created_at": row.get(
                        "created_at", datetime.now(timezone.utc).replace(tzinfo=None)
                    ),
                    "operator": "unknown",
                }
            )
        return parse_candidates

    def _vis_get_chunk_candidates(
        self, connection: Any, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get chunk candidates from the database."""
        result = self._vis_query_table(connection, "chunks", filters)
        if not result:
            return []
        chunk_configs: Dict[str, Dict[str, Any]] = {}
        for row in result:
            parse_hash = row["parse_hash"]
            if parse_hash not in chunk_configs:
                chunk_configs[parse_hash] = {
                    "chunk_count": 0,
                    "avg_length": 0,
                    "created_at": row.get(
                        "created_at", datetime.now(timezone.utc).replace(tzinfo=None)
                    ),
                }
            chunk_configs[parse_hash]["chunk_count"] += 1
            text_len = len(row.get("text") or "")
            cfg = chunk_configs[parse_hash]
            cfg["avg_length"] = (
                cfg["avg_length"] * (cfg["chunk_count"] - 1) + text_len
            ) / cfg["chunk_count"]

        chunk_candidates: List[Dict[str, Any]] = []
        for parse_hash, cfg in chunk_configs.items():
            semantic_id = self._vis_generate_semantic_id(
                "chunk", parse_hash, {"chunk_count": cfg["chunk_count"]}
            )
            stats = {
                "chunks_count": cfg["chunk_count"],
                "avg_length": int(cfg["avg_length"]),
                "parse_hash": parse_hash,
            }
            chunk_candidates.append(
                {
                    "semantic_id": semantic_id,
                    "technical_id": parse_hash,
                    "params_brief": {"chunk_count": cfg["chunk_count"]},
                    "stats": stats,
                    "state": "candidate",
                    "created_at": cfg["created_at"],
                    "operator": "unknown",
                }
            )
        return chunk_candidates

    def _vis_get_embed_candidates(
        self, connection: Any, filters: Dict[str, Any], model_tag: str
    ) -> List[Dict[str, Any]]:
        """Get embedding candidates from the database."""
        table_names = list_table_names(connection)
        embed_tables = [name for name in table_names if name.startswith("embeddings_")]
        if not embed_tables:
            return []
        embed_candidates: List[Dict[str, Any]] = []
        for table_name in embed_tables:
            table_model_tag = table_name.replace("embeddings_", "")
            if model_tag != table_model_tag:
                continue
            result = self._vis_query_table(connection, table_name, filters)
            if not result:
                continue
            embed_configs: Dict[tuple, Dict[str, Any]] = {}
            for row in result:
                model = row.get("model", "unknown")
                parse_hash = row.get("parse_hash", "unknown")
                key = (model, parse_hash)
                if key not in embed_configs:
                    embed_configs[key] = {
                        "vector_count": 0,
                        "vector_dim": 0,
                        "created_at": row.get(
                            "created_at",
                            datetime.now(timezone.utc).replace(tzinfo=None),
                        ),
                    }
                embed_configs[key]["vector_count"] += 1
                vector = row.get("vector", [])
                if vector is not None and embed_configs[key]["vector_dim"] == 0:
                    embed_configs[key]["vector_dim"] = len(vector)

            for (model, parse_hash), cfg in embed_configs.items():
                semantic_id = self._vis_generate_semantic_id(
                    "embed", parse_hash, {"model": model, "model_tag": model_tag}
                )
                stats = {
                    "upsert_count": cfg["vector_count"],
                    "vector_dim": cfg["vector_dim"],
                    "model": model,
                    "model_tag": model_tag,
                    "parse_hash": parse_hash,
                }
                embed_candidates.append(
                    {
                        "semantic_id": semantic_id,
                        "technical_id": parse_hash,
                        "params_brief": {"model": model, "model_tag": model_tag},
                        "stats": stats,
                        "state": "candidate",
                        "created_at": cfg["created_at"],
                        "operator": "unknown",
                    }
                )
        return embed_candidates

    def list_version_candidate_rows(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List raw candidate rows for a document/stage (unsorted, unlimited, unfiltered by state).

        Raises:
            DatabaseOperationError: If a database operation fails.
            VersionManagementError: If step_type is unknown or model_tag missing for embed.
        """
        from ..core.exceptions import DatabaseOperationError, VersionManagementError

        try:
            connection = self._get_connection()
            filters: Dict[str, Any] = {"collection": collection, "doc_id": doc_id}

            if step_type == "parse":
                return self._vis_get_parse_candidates(connection, filters)
            elif step_type == "chunk":
                return self._vis_get_chunk_candidates(connection, filters)
            elif step_type == "embed":
                if not model_tag:
                    raise VersionManagementError("model_tag is required for embed step")
                return self._vis_get_embed_candidates(connection, filters, model_tag)
            else:
                raise VersionManagementError(f"Unknown step_type: {step_type}")
        except (DatabaseOperationError, VersionManagementError):
            raise
        except Exception as e:
            raise DatabaseOperationError(f"Failed to get candidates: {e}") from e

    def cleanup_cascade_by_scope(
        self,
        collection: str,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash: Optional[str] = None,
        old_parse_hash: Optional[str] = None,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..core.exceptions import CascadeCleanupError
        from ..kb.cleanup_filters import (
            KBCleanupScope,
            build_embedding_cleanup_filters,
        )
        from ..LanceDB.schema_manager import (
            ensure_chunks_table,
            ensure_documents_table,
            ensure_main_pointers_table,
            ensure_parses_table,
        )
        from ..version_management.main_pointer_manager import (
            _get_main_pointer_impl as get_main_pointer,
        )

        conn = self._get_connection()
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)
        ensure_main_pointers_table(conn)

        if scope == "document":
            raw = self.cascade_delete(
                target="document",
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
                model_tag=model_tag,
                preview_only=preview_only,
                confirm=confirm,
            )
            embeddings_total = sum(
                int(v) for k, v in raw.items() if str(k).startswith("embeddings_")
            )
            return {
                "embeddings": embeddings_total,
                "chunks": int(raw.get("chunks", 0)),
                "parses": int(raw.get("parses", 0)),
                "main_pointers": int(raw.get("main_pointers", 0)),
                "documents": int(raw.get("documents", 0)),
                "ingestion_runs": int(raw.get("ingestion_runs", 0)),
            }

        predicates: Dict[str, list] = {}

        if scope == "parse":
            if old_parse_hash is None:
                pointer = get_main_pointer(collection, doc_id, "parse")
                old_parse_hash = pointer["technical_id"] if pointer else None

            if old_parse_hash:
                from ..utils.string_utils import build_lancedb_filter_expression

                base_filters = {
                    "collection": collection,
                    "doc_id": doc_id,
                    "parse_hash": old_parse_hash,
                }
                base = build_lancedb_filter_expression(
                    base_filters,
                    user_id=user_id,
                    is_admin=is_admin,
                    skip_user_filter=True,
                )
                _vis_replace_embedding_predicates(
                    predicates=predicates,
                    conn=conn,
                    base_expr=base,
                    user_id=user_id,
                    is_admin=is_admin,
                    model_tag=model_tag,
                )
                predicates["chunks"] = [
                    _vis_append_user_filter_if_needed(
                        conn=conn,
                        table_name="chunks",
                        base_expr=base,
                        user_id=user_id,
                        is_admin=is_admin,
                    )
                ]
            if new_parse_hash:
                from ..utils.string_utils import escape_lancedb_string

                escaped_collection = escape_lancedb_string(collection)
                escaped_doc_id = escape_lancedb_string(doc_id)
                escaped_new_parse_hash = escape_lancedb_string(new_parse_hash)
                other = (
                    f"collection == '{escaped_collection}' AND "
                    f"doc_id == '{escaped_doc_id}' AND "
                    f"parse_hash != '{escaped_new_parse_hash}'"
                )
                _vis_replace_embedding_predicates(
                    predicates=predicates,
                    conn=conn,
                    base_expr=other,
                    user_id=user_id,
                    is_admin=is_admin,
                    model_tag=model_tag,
                )
                predicates["chunks"] = [
                    _vis_append_user_filter_if_needed(
                        conn=conn,
                        table_name="chunks",
                        base_expr=other,
                        user_id=user_id,
                        is_admin=is_admin,
                    )
                ]
                predicates["parses"] = [
                    _vis_append_user_filter_if_needed(
                        conn=conn,
                        table_name="parses",
                        base_expr=other,
                        user_id=user_id,
                        is_admin=is_admin,
                    )
                ]
        elif scope == "chunk":
            if old_parse_hash is None:
                pointer = get_main_pointer(collection, doc_id, "chunk")
                old_parse_hash = pointer["technical_id"] if pointer else None
            if old_parse_hash:
                from ..utils.string_utils import build_lancedb_filter_expression

                base_filters = {
                    "collection": collection,
                    "doc_id": doc_id,
                    "parse_hash": old_parse_hash,
                }
                base = build_lancedb_filter_expression(
                    base_filters,
                    user_id=user_id,
                    is_admin=is_admin,
                    skip_user_filter=True,
                )
                _vis_replace_embedding_predicates(
                    predicates=predicates,
                    conn=conn,
                    base_expr=base,
                    user_id=user_id,
                    is_admin=is_admin,
                    model_tag=model_tag,
                )
            if new_parse_hash:
                from ..utils.string_utils import escape_lancedb_string

                escaped_collection = escape_lancedb_string(collection)
                escaped_doc_id = escape_lancedb_string(doc_id)
                escaped_parse_hash = escape_lancedb_string(new_parse_hash)
                other = (
                    f"collection == '{escaped_collection}' AND "
                    f"doc_id == '{escaped_doc_id}' AND "
                    f"parse_hash != '{escaped_parse_hash}'"
                )
                _vis_replace_embedding_predicates(
                    predicates=predicates,
                    conn=conn,
                    base_expr=other,
                    user_id=user_id,
                    is_admin=is_admin,
                    model_tag=model_tag,
                )
                predicates["chunks"] = [
                    _vis_append_user_filter_if_needed(
                        conn=conn,
                        table_name="chunks",
                        base_expr=other,
                        user_id=user_id,
                        is_admin=is_admin,
                    )
                ]
        elif scope == "embeddings":
            scope_obj = KBCleanupScope(
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
                model_tag=model_tag,
            )
            filters_by_table = build_embedding_cleanup_filters(conn, scope_obj)
            for table_name, filter_exprs in filters_by_table.items():
                if filter_exprs:
                    predicates.setdefault(table_name, []).extend(filter_exprs)
        elif scope == "pointers":
            filt = _vis_build_document_filter(
                conn=conn,
                table_name="main_pointers",
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )
            predicates["main_pointers"] = [filt]
        else:
            raise CascadeCleanupError(f"Unsupported scope: {scope}")

        result = _vis_execute_or_plan_by_predicates(
            conn,
            predicates,
            preview_only=preview_only,
            confirm=confirm,
            model_tag=model_tag,
        )
        if scope in {"parse", "chunk", "embeddings"}:
            return _vis_collapse_embedding_table_counts(result)
        return result

    def cascade_delete(
        self,
        *,
        target: Literal["collection", "document"],
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        model_tag: Optional[str] = None,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        from ..core.exceptions import CascadeCleanupError
        from ..kb.cleanup_filters import select_embedding_tables
        from ..LanceDB.schema_manager import (
            ensure_chunks_table,
            ensure_documents_table,
            ensure_ingestion_runs_table,
            ensure_main_pointers_table,
            ensure_parses_table,
        )
        from ..utils.user_scope import resolve_user_scope

        user_scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
        user_id = user_scope.user_id
        is_admin = user_scope.is_admin

        if target == "document" and not doc_id:
            raise CascadeCleanupError("doc_id is required for document cascade delete")

        conn = self._get_connection()
        ensure_documents_table(conn)
        ensure_parses_table(conn)
        ensure_chunks_table(conn)
        ensure_main_pointers_table(conn)
        ensure_ingestion_runs_table(conn)

        table_names = _vis_get_table_names(conn)
        predicates: Dict[str, list] = {}

        core_tables = [
            "documents",
            "parses",
            "chunks",
            "main_pointers",
            "ingestion_runs",
        ]
        for table_name in core_tables:
            if table_name not in table_names:
                continue
            if target == "collection":
                filter_expr = _vis_build_collection_filter(
                    conn=conn,
                    table_name=table_name,
                    collection=collection,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            else:
                filter_expr = _vis_build_document_filter(
                    conn=conn,
                    table_name=table_name,
                    collection=collection,
                    doc_id=str(doc_id),
                    user_id=user_id,
                    is_admin=is_admin,
                )
            predicates[table_name] = [filter_expr]

        for table_name in select_embedding_tables(conn, model_tag=model_tag):
            if target == "collection":
                filter_expr = _vis_build_collection_filter(
                    conn=conn,
                    table_name=table_name,
                    collection=collection,
                    user_id=user_id,
                    is_admin=is_admin,
                )
            else:
                filter_expr = _vis_build_document_filter(
                    conn=conn,
                    table_name=table_name,
                    collection=collection,
                    doc_id=str(doc_id),
                    user_id=user_id,
                    is_admin=is_admin,
                )
            predicates[table_name] = [filter_expr]

        result = _vis_execute_or_plan_by_predicates(
            conn,
            predicates,
            preview_only=preview_only,
            confirm=confirm,
            model_tag=None,
        )
        if confirm:
            self.invalidate_table_cache()
        return result


# ============================================================================
# Phase 1A Part 2: Additional LanceDB Store Implementations
# ============================================================================

# ---------------------------------------------------------------------------
# _vis_* helpers: predicate pipeline for collection/document cascade delete
# ---------------------------------------------------------------------------


def _vis_count_rows_by_filters(table: Any, filter_exprs: list) -> int:
    from ..utils.lancedb_query_utils import _safe_count_rows

    return sum(_safe_count_rows(table, f) for f in filter_exprs)


def _vis_delete_rows_by_filters(table: Any, filter_exprs: list) -> int:
    from ..utils.lancedb_query_utils import _safe_count_rows

    deleted_count = 0
    for filter_expr in filter_exprs:
        count = _safe_count_rows(table, filter_expr)
        if count > 0:
            table.delete(filter_expr)
        deleted_count += count
    return deleted_count


def _vis_append_user_filter_if_needed(
    *,
    conn: Any,
    table_name: str,
    base_expr: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    from ..kb.cleanup_filters import (
        append_user_filter_for_table,
        append_user_filter_without_schema,
    )
    from ..LanceDB.schema_manager import _safe_close_table

    table = None
    try:
        table = conn.open_table(table_name)
        return append_user_filter_for_table(
            table=table,
            filter_expr=base_expr,
            user_id=user_id,
            is_admin=is_admin,
        )
    except Exception:
        return append_user_filter_without_schema(
            filter_expr=base_expr,
            user_id=user_id,
            is_admin=is_admin,
        )
    finally:
        _safe_close_table(table)


def _vis_replace_embedding_predicates(
    *,
    predicates: Dict[str, list],
    conn: Any,
    base_expr: str,
    user_id: Optional[int],
    is_admin: bool,
    model_tag: Optional[str] = None,
) -> None:
    from ..kb.cleanup_filters import build_embedding_cleanup_filters_from_base

    table_filters = build_embedding_cleanup_filters_from_base(
        conn,
        base_filter=base_expr,
        user_id=user_id,
        is_admin=is_admin,
        model_tag=model_tag,
    )
    for table_name, filter_exprs in table_filters.items():
        if filter_exprs:
            predicates[table_name] = list(filter_exprs)


def _vis_get_table_names(conn: Any) -> list:
    from ..utils.lancedb_query_utils import list_table_names

    try:
        return list(list_table_names(conn))
    except Exception:
        return []


def _vis_plan_by_predicates(
    conn: Any, table_to_filter: Dict[str, list], model_tag: Optional[str] = None
) -> Dict[str, int]:
    from ..kb.cleanup_filters import select_embedding_tables
    from ..LanceDB.schema_manager import _safe_close_table

    counts: Dict[str, int] = {}
    table_names = _vis_get_table_names(conn)

    for t in table_names:
        if t.startswith("embeddings_") and t in table_to_filter:
            table = None
            try:
                table = conn.open_table(t)
                counts[t] = _vis_count_rows_by_filters(table, table_to_filter[t])
            finally:
                _safe_close_table(table)

    for table_name, filter_exprs in table_to_filter.items():
        if table_name == "__embeddings__":
            total = 0
            target_tables = select_embedding_tables(conn, model_tag=model_tag)
            for t in target_tables:
                table = None
                try:
                    table = conn.open_table(t)
                    total += _vis_count_rows_by_filters(table, filter_exprs)
                finally:
                    _safe_close_table(table)
            counts[table_name] = total
            continue
        if table_name.startswith("embeddings_"):
            continue
        if table_name not in table_names:
            counts[table_name] = 0
            continue
        table = None
        try:
            table = conn.open_table(table_name)
            counts[table_name] = _vis_count_rows_by_filters(table, filter_exprs)
        finally:
            _safe_close_table(table)
    return counts


def _vis_delete_by_predicates(
    conn: Any, table_to_filter: Dict[str, list], model_tag: Optional[str] = None
) -> Dict[str, int]:
    import logging as _logging

    from ..kb.cleanup_filters import select_embedding_tables
    from ..LanceDB.schema_manager import _safe_close_table

    _logger = _logging.getLogger(__name__)
    deleted: Dict[str, int] = {}
    table_names = _vis_get_table_names(conn)

    for t in table_names:
        if not t.startswith("embeddings_") or t not in table_to_filter:
            continue
        filter_exprs = table_to_filter[t]
        table = None
        try:
            table = conn.open_table(t)
            cnt = _vis_delete_rows_by_filters(table, filter_exprs)
            if cnt > 0:
                _logger.info("Cascade cleanup: deleted %s rows from %s", cnt, t)
            deleted[t] = cnt
        finally:
            _safe_close_table(table)

    order = [
        "__embeddings__",
        "chunks",
        "parses",
        "main_pointers",
        "ingestion_runs",
        "documents",
    ]

    if "__embeddings__" in table_to_filter:
        filter_exprs = table_to_filter["__embeddings__"]
        total = 0
        target_tables = select_embedding_tables(conn, model_tag=model_tag)
        for t in target_tables:
            table = None
            try:
                table = conn.open_table(t)
                total += _vis_delete_rows_by_filters(table, filter_exprs)
            finally:
                _safe_close_table(table)
        deleted["embeddings"] = total
        if total > 0:
            _logger.info(
                "Cascade cleanup: deleted %s rows from embeddings tables", total
            )

    for name in order[1:]:
        if name in table_to_filter and name in table_names:
            table = None
            try:
                table = conn.open_table(name)
                cnt = _vis_delete_rows_by_filters(table, table_to_filter[name])
                if cnt > 0:
                    _logger.info("Cascade cleanup: deleted %s rows from %s", cnt, name)
                deleted[name] = cnt
            finally:
                _safe_close_table(table)

    for name, filter_exprs in table_to_filter.items():
        if name in (
            "__embeddings__",
            "chunks",
            "parses",
            "main_pointers",
            "ingestion_runs",
            "documents",
        ) or name.startswith("embeddings_"):
            continue
        if name not in table_names:
            deleted[name] = 0
            continue
        table = None
        try:
            table = conn.open_table(name)
            cnt = _vis_delete_rows_by_filters(table, filter_exprs)
            if cnt > 0:
                _logger.info("Cascade cleanup: deleted %s rows from %s", cnt, name)
            deleted[name] = cnt
        finally:
            _safe_close_table(table)

    return deleted


def _vis_execute_or_plan_by_predicates(
    conn: Any,
    predicates: Dict[str, list],
    *,
    preview_only: bool,
    confirm: bool,
    model_tag: Optional[str] = None,
) -> Dict[str, int]:
    if not (confirm and not preview_only):
        return _vis_plan_by_predicates(conn, predicates, model_tag=model_tag)
    return _vis_delete_by_predicates(conn, predicates, model_tag=model_tag)


def _vis_collapse_embedding_table_counts(counts: Dict[str, int]) -> Dict[str, int]:
    collapsed = dict(counts)
    embeddings_total = int(collapsed.pop("__embeddings__", 0)) + int(
        collapsed.pop("embeddings", 0)
    )
    for table_name in list(collapsed):
        if str(table_name).startswith("embeddings_"):
            embeddings_total += int(collapsed.pop(table_name, 0))
    collapsed["embeddings"] = embeddings_total
    return collapsed


def _vis_build_collection_filter(
    *,
    conn: Any,
    table_name: str,
    collection: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Build a safe filter for collection-scoped deletion.

    Adds user_id filtering only when the target table contains a user_id column.
    """
    from ..kb.cleanup_filters import table_has_column as _table_has_column
    from ..LanceDB.schema_manager import _safe_close_table

    base: Dict[str, str] = {"collection": collection}
    table = None
    try:
        table = conn.open_table(table_name)
        if not is_admin and user_id is not None:
            if _table_has_column(table, "user_id"):
                base_expr = build_lancedb_filter_expression(
                    base, user_id=user_id, is_admin=is_admin, skip_user_filter=True
                )
                user_expr = build_user_id_filter_for_table(table, int(user_id))
                return f"{base_expr} AND {user_expr}"
            # Legacy schemas without user_id must remain compatible.
            return build_lancedb_filter_expression(base, skip_user_filter=True)
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    except Exception:
        # If table introspection fails, keep tenant-safe fallback.
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    finally:
        _safe_close_table(table)


def _vis_build_document_filter(
    *,
    conn: Any,
    table_name: str,
    collection: str,
    doc_id: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Build a safe filter for document-scoped deletion."""
    from ..kb.cleanup_filters import table_has_column as _table_has_column
    from ..LanceDB.schema_manager import _safe_close_table

    base: Dict[str, str] = {"collection": collection, "doc_id": doc_id}
    table = None
    try:
        table = conn.open_table(table_name)
        if not is_admin and user_id is not None:
            if _table_has_column(table, "user_id"):
                base_expr = build_lancedb_filter_expression(
                    base, user_id=user_id, is_admin=is_admin, skip_user_filter=True
                )
                user_expr = build_user_id_filter_for_table(table, int(user_id))
                return f"{base_expr} AND {user_expr}"
            # Legacy schemas without user_id must remain compatible.
            return build_lancedb_filter_expression(base, skip_user_filter=True)
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    except Exception:
        # If table introspection fails, keep tenant-safe fallback.
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    finally:
        _safe_close_table(table)


def _vis_doc_ids_filter(doc_ids: list) -> str:
    if len(doc_ids) == 1:
        return f"doc_id == '{escape_lancedb_string(doc_ids[0])}'"
    values = ", ".join(f"'{escape_lancedb_string(doc_id)}'" for doc_id in doc_ids)
    return f"doc_id IN ({values})"


def _vis_build_documents_filter(
    *,
    conn: Any,
    table_name: str,
    collection: str,
    doc_ids: list,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Build a safe filter for deleting multiple document-scoped rows."""
    from ..kb.cleanup_filters import table_has_column as _table_has_column
    from ..LanceDB.schema_manager import _safe_close_table

    base_expr = build_lancedb_filter_expression(
        {"collection": collection}, skip_user_filter=True
    )
    doc_expr = _vis_doc_ids_filter(doc_ids)
    scoped_expr = f"{base_expr} AND {doc_expr}"

    if is_admin:
        return scoped_expr
    if user_id is None:
        return f"{scoped_expr} AND ({UserPermissions.get_no_access_filter()})"

    table = None
    try:
        table = conn.open_table(table_name)
        if _table_has_column(table, "user_id"):
            return (
                f"{scoped_expr} AND "
                f"{build_user_id_filter_for_table(table, int(user_id))}"
            )
        # Legacy schemas without user_id stay document-scoped by doc_id.
        return scoped_expr
    except Exception:
        # If introspection fails, fail closed with an explicit user_id predicate.
        return f"{scoped_expr} AND user_id == {int(user_id)}"
    finally:
        _safe_close_table(table)


def _vis_cascade_delete_documents(
    conn: Any,
    *,
    collection: str,
    doc_ids: list,
    user_id: Optional[int],
    is_admin: bool,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Cascade delete multiple documents using one predicate set per table."""
    from ..kb.cleanup_filters import select_embedding_tables
    from ..LanceDB.schema_manager import (
        ensure_chunks_table,
        ensure_documents_table,
        ensure_ingestion_runs_table,
        ensure_main_pointers_table,
        ensure_parses_table,
    )
    from ..utils.user_scope import resolve_user_scope

    normalized_doc_ids = sorted({str(d) for d in doc_ids if d})
    if not normalized_doc_ids:
        return {}

    user_scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
    user_id = user_scope.user_id
    is_admin = user_scope.is_admin
    if not is_admin and user_id is None:
        return {}

    ensure_documents_table(conn)
    ensure_parses_table(conn)
    ensure_chunks_table(conn)
    ensure_main_pointers_table(conn)
    ensure_ingestion_runs_table(conn)

    table_names = _vis_get_table_names(conn)
    predicates: Dict[str, list] = {}

    core_tables = ["documents", "parses", "chunks", "main_pointers", "ingestion_runs"]
    for table_name in core_tables:
        if table_name not in table_names:
            continue
        predicates[table_name] = [
            _vis_build_documents_filter(
                conn=conn,
                table_name=table_name,
                collection=collection,
                doc_ids=normalized_doc_ids,
                user_id=user_id,
                is_admin=is_admin,
            )
        ]

    for table_name in select_embedding_tables(conn):
        predicates[table_name] = [
            _vis_build_documents_filter(
                conn=conn,
                table_name=table_name,
                collection=collection,
                doc_ids=normalized_doc_ids,
                user_id=user_id,
                is_admin=is_admin,
            )
        ]

    return _vis_execute_or_plan_by_predicates(
        conn,
        predicates,
        preview_only=preview_only,
        confirm=confirm,
        model_tag=None,
    )


class LanceDBIngestionStatusStore(IngestionStatusStore):
    """LanceDB implementation for ingestion status tracking.

    Manages ingestion_runs table for tracking document processing status.
    """

    def __init__(self) -> None:
        self._sync_conn: Optional[DBConnection] = None
        self._async_conn: Optional[Any] = None
        self._async_lock = asyncio.Lock()

    def _get_sync_connection(self) -> DBConnection:
        """Get sync LanceDB connection."""
        if self._sync_conn is None:
            self._sync_conn = get_connection_from_env()
        return self._sync_conn

    async def _get_async_connection(self) -> Any:
        """Get async LanceDB connection."""
        if self._async_conn is None:
            async with self._async_lock:
                if self._async_conn is None:
                    self._async_conn = await lancedb.connect_async(  # type: ignore[attr-defined]
                        get_connection_from_env().uri  # type: ignore[attr-defined]
                    )
        return self._async_conn

    def _ensure_ingestion_runs_table(self, conn: DBConnection) -> None:
        """Ensure ingestion_runs table exists."""
        from ..LanceDB.schema_manager import ensure_ingestion_runs_table

        ensure_ingestion_runs_table(conn)

    # --- Sync methods ---

    def write_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Write ingestion status record (sync)."""
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            conn = self._get_sync_connection()
            self._ensure_ingestion_runs_table(conn)
            table = conn.open_table("ingestion_runs")

            # Delete existing record for this collection/doc_id
            base_filter = self._build_base_filter(collection, doc_id)
            if base_filter:
                table.delete(base_filter)

            # Create new record
            timestamp = datetime.now(timezone.utc)
            record = {
                "collection": collection,
                "doc_id": doc_id,
                "status": status,
                "message": message or "",
                "parse_hash": parse_hash or "",
                "created_at": timestamp,
                "updated_at": timestamp,
                "user_id": user_id,
            }
            table.add([record])

        except Exception as e:
            logger.error("Failed to write ingestion status: %s", e)
            raise
        finally:
            _safe_close_table(table)

    def load_ingestion_status(
        self,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load ingestion status records (sync)."""
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            conn = self._get_sync_connection()
            self._ensure_ingestion_runs_table(conn)
            table = conn.open_table("ingestion_runs")

            # Build filter expression
            filter_expr = self._build_load_filter(collection, doc_id, user_id, is_admin)

            # Execute query
            search = table.search()
            if filter_expr:
                search = search.where(filter_expr)
            result = search.to_arrow()

            # Convert Arrow table to list of dicts (records format)
            if len(result) == 0:
                return []
            return cast(List[Dict[str, Any]], result.to_pylist())

        except Exception as e:
            logger.error("Failed to load ingestion status: %s", e)
            raise
        finally:
            _safe_close_table(table)

    def clear_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        """Remove ingestion status record (sync)."""
        from ..LanceDB.schema_manager import _safe_close_table

        table = None
        try:
            conn = self._get_sync_connection()
            self._ensure_ingestion_runs_table(conn)
            table = conn.open_table("ingestion_runs")

            # Build filter with user permissions
            base_filter = self._build_base_filter(collection, doc_id)
            user_filter = UserPermissions.get_user_filter(user_id, is_admin)

            filter_expr = self._combine_filters(base_filter, user_filter)
            if filter_expr:
                table.delete(filter_expr)

        except Exception as e:
            logger.error("Failed to clear ingestion status: %s", e)
            raise
        finally:
            _safe_close_table(table)

    # Deprecated: use KBCollectionHandle.rename_collection_status instead. Remove in #514.
    def rename_collection_status(
        self,
        old_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> List[str]:
        """Rename ingestion status rows for a collection."""
        from ..LanceDB.schema_manager import _safe_close_table

        warnings: List[str] = []
        table = None
        try:
            conn = self._get_sync_connection()
            self._ensure_ingestion_runs_table(conn)
            table = conn.open_table("ingestion_runs")
            where_expr = self._build_load_filter(old_name, None, user_id, is_admin)
            if where_expr:
                table.update(
                    where_expr,
                    {
                        "collection": new_name,
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to rename ingestion status: {exc}"
            logger.warning(message)
            warnings.append(message)
        finally:
            _safe_close_table(table)
        return warnings

    # --- Async methods ---

    async def write_ingestion_status_async(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Write ingestion status record (async).

        Note: Current implementation uses sync operations under the hood.
        True async I/O will be added in Phase 1B with RDB backend.
        """
        # Delegate to sync implementation for now
        return self.write_ingestion_status(
            collection=collection,
            doc_id=doc_id,
            status=status,
            message=message,
            parse_hash=parse_hash,
            user_id=user_id,
        )

    async def load_ingestion_status_async(
        self,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load ingestion status records (async).

        Note: Current implementation uses sync operations under the hood.
        True async I/O will be added in Phase 1B with RDB backend.
        """
        # Delegate to sync implementation for now
        return self.load_ingestion_status(
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def clear_ingestion_status_async(
        self,
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        """Remove ingestion status record (async).

        Note: Current implementation uses sync operations under the hood.
        True async I/O will be added in Phase 1B with RDB backend.
        """
        # Delegate to sync implementation for now
        return self.clear_ingestion_status(
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )

    async def rename_collection_status_async(
        self,
        old_name: str,
        new_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
    ) -> List[str]:
        """Async version of rename_collection_status."""
        return self.rename_collection_status(
            old_name=old_name,
            new_name=new_name,
            user_id=user_id,
            is_admin=is_admin,
        )

    # --- Helper methods ---

    def _build_base_filter(self, collection: str, doc_id: str) -> str:
        """Build base filter for collection/doc_id."""
        safe_collection = escape_lancedb_string(collection)
        safe_doc_id = escape_lancedb_string(doc_id)
        return f"collection == '{safe_collection}' AND doc_id == '{safe_doc_id}'"

    def _build_load_filter(
        self,
        collection: Optional[str],
        doc_id: Optional[str],
        user_id: Optional[int],
        is_admin: bool,
    ) -> Optional[str]:
        """Build filter for loading status records."""
        conditions = []

        if collection is not None:
            safe_collection = escape_lancedb_string(collection)
            conditions.append(f"collection == '{safe_collection}'")

        if doc_id is not None:
            safe_doc_id = escape_lancedb_string(doc_id)
            conditions.append(f"doc_id == '{safe_doc_id}'")

        # Combine with user filter
        base_filter = " AND ".join(conditions) if conditions else None
        user_filter = UserPermissions.get_user_filter(user_id, is_admin)

        return self._combine_filters(base_filter, user_filter)

    def _combine_filters(
        self, base_filter: Optional[str], user_filter: Optional[str]
    ) -> Optional[str]:
        """Combine base and user filters."""
        if user_filter and base_filter:
            return f"({base_filter}) AND ({user_filter})"
        elif user_filter:
            return user_filter
        return base_filter


class LanceDBPromptTemplateStore(PromptTemplateStore):
    """LanceDB implementation for prompt template management.

    Manages prompt_templates table for storing and retrieving prompt templates.
    """

    def __init__(self) -> None:
        self._sync_conn: Optional[DBConnection] = None

    def _get_sync_connection(self) -> DBConnection:
        """Get or create sync connection."""
        if self._sync_conn is None:
            self._sync_conn = get_connection_from_env()
        return self._sync_conn

    def _ensure_table(self) -> None:
        """Ensure prompt_templates table exists."""
        from ..LanceDB.schema_manager import ensure_prompt_templates_table

        conn = self._get_sync_connection()
        ensure_prompt_templates_table(conn)

    # --- Sync methods ---

    def save_prompt_template(
        self,
        name: str,
        template: str,
        user_id: Optional[int] = None,
        metadata: Optional[str] = None,
    ) -> str:
        """Save or update a prompt template (sync)."""
        import uuid

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("prompt_templates")

        try:
            # Generate new template ID
            template_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).replace(tzinfo=None)

            # Check for existing templates with same name to get next version
            base_filter = f"name == '{escape_lancedb_string(name)}'"
            if user_id is not None:
                base_filter += f" AND user_id == {user_id}"

            existing = table.search().where(base_filter).to_arrow()
            if len(existing) > 0:
                import pyarrow.compute as pc  # type: ignore[import-not-found]

                max_version = pc.max(existing["version"]).as_py()
                new_version = max_version + 1

                # Mark previous versions as not latest
                for row in existing.to_pylist():
                    if row["is_latest"]:
                        table.update(
                            where=f"id == '{row['id']}'",
                            values={"is_latest": False},
                        )
            else:
                new_version = 1

            # Create new template record
            record = {
                "id": template_id,
                "name": name,
                "template": template,
                "version": new_version,
                "is_latest": True,
                "metadata": metadata or "",
                "user_id": user_id or 0,
                "created_at": now,
                "updated_at": now,
            }

            table.add([record])
            logger.info("Saved prompt template: %s (version %d)", name, new_version)
            return template_id
        finally:
            _safe_close_table(table)

    def get_prompt_template(
        self,
        template_id: str,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a prompt template by ID (sync)."""
        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("prompt_templates")

        try:
            base_filter = f"id == '{escape_lancedb_string(template_id)}'"
            if user_id is not None:
                base_filter += f" AND user_id == {user_id}"

            result = table.search().where(base_filter).to_arrow()
            if len(result) == 0:
                return None

            # Convert Arrow table to list of dicts and take first row
            row = result.to_pylist()[0]
            return {
                "id": row["id"],
                "name": row["name"],
                "template": row["template"],
                "version": int(row["version"]),
                "is_latest": bool(row["is_latest"]),
                "metadata": row["metadata"],
                "user_id": int(row["user_id"]) if row["user_id"] else None,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        finally:
            _safe_close_table(table)

    def get_latest_prompt_template(
        self,
        name: str,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the latest version of a prompt template by name (sync)."""
        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("prompt_templates")

        try:
            base_filter = (
                f"name == '{escape_lancedb_string(name)}' AND is_latest == true"
            )
            if user_id is not None:
                base_filter += f" AND user_id == {user_id}"

            result = table.search().where(base_filter).to_arrow()
            if len(result) == 0:
                return None

            # Convert Arrow table to list of dicts and take first row
            row = result.to_pylist()[0]
            return {
                "id": row["id"],
                "name": row["name"],
                "template": row["template"],
                "version": int(row["version"]),
                "is_latest": bool(row["is_latest"]),
                "metadata": row["metadata"],
                "user_id": int(row["user_id"]) if row["user_id"] else None,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        finally:
            _safe_close_table(table)

    def list_prompt_templates(
        self,
        name_filter: Optional[str] = None,
        latest_only: bool = False,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List prompt templates with optional filtering (sync)."""
        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("prompt_templates")

        try:
            filters = []
            if name_filter:
                filters.append(f"name LIKE '%{escape_lancedb_string(name_filter)}%'")
            if latest_only:
                filters.append("is_latest == true")
            if user_id is not None:
                filters.append(f"user_id == {user_id}")

            filter_expr = " AND ".join(filters) if filters else None

            query = table.search()
            if filter_expr:
                query = query.where(filter_expr)

            result = query.limit(limit).to_arrow()
            templates = []
            for row_dict in result.to_pylist():
                templates.append(
                    {
                        "id": row_dict["id"],
                        "name": row_dict["name"],
                        "template": row_dict["template"],
                        "version": int(row_dict["version"]),
                        "is_latest": bool(row_dict["is_latest"]),
                        "metadata": row_dict["metadata"],
                        "user_id": int(row_dict["user_id"])
                        if row_dict["user_id"]
                        else None,
                        "created_at": row_dict["created_at"],
                        "updated_at": row_dict["updated_at"],
                    }
                )

            return templates
        finally:
            _safe_close_table(table)

    def delete_prompt_template(
        self,
        template_id: str,
        user_id: Optional[int] = None,
    ) -> bool:
        """Delete a prompt template by ID (sync).

        Updates is_latest flag for remaining versions if latest version is deleted.
        """
        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("prompt_templates")

        try:
            base_filter = f"id == '{escape_lancedb_string(template_id)}'"
            if user_id is not None:
                base_filter += f" AND user_id == {user_id}"

            # Check if exists and get info
            result = table.search().where(base_filter).to_arrow()
            if len(result) == 0:
                return False

            # Check if this was the latest version and get the name
            # Convert Arrow table to list of dicts and take first row
            row_dict = result.to_pylist()[0]
            was_latest = row_dict["is_latest"]
            template_name = row_dict["name"]

            table.delete(base_filter)

            # If we deleted the latest version, update the latest flag for the remaining versions
            if was_latest:
                name_filter = f"name == '{escape_lancedb_string(template_name)}'"
                if user_id is not None:
                    name_filter += f" AND user_id == {user_id}"

                remaining_versions = table.search().where(name_filter).to_arrow()
                if len(remaining_versions) > 0:
                    import pyarrow.compute as pc

                    max_version = pc.max(remaining_versions["version"]).as_py()
                    update_filter = f"{name_filter} AND version == {max_version}"
                    table.update(where=update_filter, values={"is_latest": True})

            logger.info("Deleted prompt template: %s", template_id)
            return True
        finally:
            _safe_close_table(table)

    def update_metadata(
        self,
        template_id: str,
        metadata: Optional[str],
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update metadata only, keeping same version and ID (sync)."""
        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("prompt_templates")

        try:
            base_filter = f"id == '{escape_lancedb_string(template_id)}'"
            if user_id is not None:
                base_filter += f" AND user_id == {user_id}"

            # Check if exists
            result = table.search().where(base_filter).to_arrow()
            if len(result) == 0:
                return None

            # Update metadata
            table.update(
                where=base_filter,
                values={
                    "metadata": metadata or "",
                    "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
                },
            )
            logger.info("Updated metadata for prompt template: %s", template_id)

            # Return updated template
            return self.get_prompt_template(template_id, user_id)
        finally:
            _safe_close_table(table)

    def delete_by_name(
        self,
        name: str,
        version: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> int:
        """Delete template(s) by name (sync).

        Handles is_latest flag updates for remaining versions.
        """
        from ..core.exceptions import DocumentNotFoundError
        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("prompt_templates")

        try:
            escaped_name = escape_lancedb_string(name)
            base_filter = f"name == '{escaped_name}'"
            if user_id is not None:
                base_filter += f" AND user_id == {user_id}"

            if version is not None:
                # Delete specific version
                version_filter = f"{base_filter} AND version == {version}"
                result = table.search().where(version_filter).to_arrow()
                if len(result) == 0:
                    raise DocumentNotFoundError(
                        f"Prompt template '{name}' version {version} not found."
                    )

                # Convert Arrow table to list of dicts and take first row
                row_dict = result.to_pylist()[0]
                was_latest = row_dict["is_latest"]
                table.delete(version_filter)

                # If we deleted the latest version, update the latest flag
                if was_latest:
                    remaining = table.search().where(base_filter).to_arrow()
                    if len(remaining) > 0:
                        import pyarrow.compute as pc

                        max_version = pc.max(remaining["version"]).as_py()
                        table.update(
                            where=f"{base_filter} AND version == {max_version}",
                            values={"is_latest": True},
                        )

                logger.info("Deleted prompt template '%s' version %d", name, version)
                return 1
            else:
                # Delete all versions
                result = table.search().where(base_filter).to_arrow()
                if len(result) == 0:
                    raise DocumentNotFoundError(f"Prompt template '{name}' not found.")

                count = len(result)
                table.delete(base_filter)
                logger.info(
                    "Deleted all %d versions of prompt template '%s'", count, name
                )
                return count
        finally:
            _safe_close_table(table)

    def get_versions_by_name(
        self,
        name: str,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get all versions of a template by name (sync)."""
        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("prompt_templates")

        try:
            base_filter = f"name == '{escape_lancedb_string(name)}'"
            if user_id is not None:
                base_filter += f" AND user_id == {user_id}"

            result = table.search().where(base_filter).limit(limit).to_arrow()
            templates = []
            for row_dict in result.to_pylist():
                templates.append(
                    {
                        "id": row_dict["id"],
                        "name": row_dict["name"],
                        "template": row_dict["template"],
                        "version": int(row_dict["version"]),
                        "is_latest": bool(row_dict["is_latest"]),
                        "metadata": row_dict["metadata"],
                        "user_id": int(row_dict["user_id"])
                        if row_dict["user_id"]
                        else None,
                        "created_at": row_dict["created_at"],
                        "updated_at": row_dict["updated_at"],
                    }
                )

            return templates
        finally:
            _safe_close_table(table)

    # --- Async methods (delegate to sync) ---

    async def save_prompt_template_async(
        self,
        name: str,
        template: str,
        user_id: Optional[int] = None,
        metadata: Optional[str] = None,
    ) -> str:
        """Async version of save_prompt_template."""
        return self.save_prompt_template(name, template, user_id, metadata)

    async def get_prompt_template_async(
        self,
        template_id: str,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of get_prompt_template."""
        return self.get_prompt_template(template_id, user_id)

    async def get_latest_prompt_template_async(
        self,
        name: str,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of get_latest_prompt_template."""
        return self.get_latest_prompt_template(name, user_id)

    async def list_prompt_templates_async(
        self,
        name_filter: Optional[str] = None,
        latest_only: bool = False,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Async version of list_prompt_templates."""
        return self.list_prompt_templates(name_filter, latest_only, user_id, limit)

    async def delete_prompt_template_async(
        self,
        template_id: str,
        user_id: Optional[int] = None,
    ) -> bool:
        """Async version of delete_prompt_template."""
        return self.delete_prompt_template(template_id, user_id)

    async def update_metadata_async(
        self,
        template_id: str,
        metadata: Optional[str],
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of update_metadata."""
        return self.update_metadata(template_id, metadata, user_id)

    async def delete_by_name_async(
        self,
        name: str,
        version: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> int:
        """Async version of delete_by_name."""
        return self.delete_by_name(name, version, user_id)

    async def get_versions_by_name_async(
        self,
        name: str,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Async version of get_versions_by_name."""
        return self.get_versions_by_name(name, user_id, limit)


class LanceDBMainPointerStore(MainPointerStore):
    """LanceDB implementation for main pointer management.

    Manages main_pointers table for tracking current versions across
    processing stages (parse, chunk, embed).

    NOTE: user_id parameter is logged but not used, as main_pointers table
    schema does not include user_id field. Schema migration required for
    multi-tenancy support.
    """

    def __init__(self) -> None:
        self._sync_conn: Optional[DBConnection] = None

    def _get_sync_connection(self) -> DBConnection:
        """Get or create sync connection."""
        if self._sync_conn is None:
            self._sync_conn = get_connection_from_env()
        return self._sync_conn

    def _ensure_table(self) -> None:
        """Ensure main_pointers table exists."""
        from ..LanceDB.schema_manager import ensure_main_pointers_table

        conn = self._get_sync_connection()
        ensure_main_pointers_table(conn)

    def _normalize_model_tag(self, model_tag: Optional[str]) -> str:
        """Normalize model_tag to empty string if None."""
        return model_tag if model_tag is not None else ""

    # --- Sync methods ---

    def set_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Set or update a main pointer (sync)."""
        if user_id is not None:
            logger.warning(
                "user_id parameter provided to set_main_pointer but "
                "main_pointers table does not have user_id field. "
                "Schema migration required for multi-tenancy support."
            )

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("main_pointers")

        try:
            normalized_tag = self._normalize_model_tag(model_tag)
            now = datetime.now(timezone.utc).replace(tzinfo=None)

            # Check if pointer already exists to preserve created_at
            existing = self.get_main_pointer(collection, doc_id, step_type, model_tag)

            created_at = existing["created_at"] if existing else now

            # Prepare data for merge_insert
            update_data: Dict[str, List[Any]] = {
                "collection": [collection],
                "doc_id": [doc_id],
                "step_type": [step_type],
                "model_tag": [normalized_tag],
                "semantic_id": [semantic_id],
                "technical_id": [technical_id],
                "created_at": [created_at],
                "updated_at": [now],
                "operator": [operator or "unknown"],
            }
            # Convert dict of lists to list of dicts for merge_insert
            records = [
                {key: values[idx] for key, values in update_data.items()}
                for idx in range(len(update_data["collection"]))
            ]

            (
                table.merge_insert(
                    on=["collection", "doc_id", "step_type", "model_tag"]
                )
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records)
            )

            logger.info(
                "Set main pointer for %s/%s/%s to %s (semantic: %s)",
                collection,
                doc_id,
                step_type,
                technical_id,
                semantic_id,
            )
        finally:
            _safe_close_table(table)

    def get_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a main pointer (sync)."""
        if user_id is not None:
            logger.warning(
                "user_id parameter provided to get_main_pointer but "
                "main_pointers table does not have user_id field. "
                "Schema migration required for multi-tenancy support."
            )

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("main_pointers")

        try:
            # Build filter expression using FilterCondition
            base_conditions: List[FilterCondition] = [
                FilterCondition(
                    field="collection", operator=FilterOperator.EQ, value=collection
                ),
                FilterCondition(
                    field="doc_id", operator=FilterOperator.EQ, value=doc_id
                ),
                FilterCondition(
                    field="step_type", operator=FilterOperator.EQ, value=step_type
                ),
            ]

            normalized_tag = self._normalize_model_tag(model_tag)
            if normalized_tag == "":
                # Check for both empty string AND NULL (backward compatibility)
                model_tag_null_cond = FilterCondition(
                    field="model_tag", operator=FilterOperator.IS_NULL, value=None
                )
                model_tag_empty_cond = FilterCondition(
                    field="model_tag", operator=FilterOperator.EQ, value=""
                )
                # Combine as: (base) AND (model_tag IS NULL OR model_tag == '')
                model_tag_filter: FilterExpression = (
                    model_tag_null_cond,
                    model_tag_empty_cond,
                )  # OR tuple
                filter_expr: FilterExpression = (
                    *base_conditions,
                    model_tag_filter,
                )  # AND tuple
            else:
                base_conditions.append(
                    FilterCondition(
                        field="model_tag",
                        operator=FilterOperator.EQ,
                        value=normalized_tag,
                    )
                )
                filter_expr = tuple(base_conditions)  # AND tuple

            # Translate to LanceDB syntax using shared utility
            filter_str = translate_filter_expression(filter_expr)

            result = table.search().where(filter_str).to_arrow()

            if len(result) == 0:
                return None

            # Return the first result, preferring non-NULL model_tag if multiple found
            if len(result) > 1:
                import pyarrow.compute as pc

                # Sort by model_tag descending (NULLs last)
                sort_indices = pc.sort_indices(
                    result, sort_keys=[("model_tag", "descending")]
                )
                result = result.take(sort_indices)

            # Convert Arrow table to list of dicts and take first row
            row_dict = result.to_pylist()[0]
            return {
                "collection": row_dict["collection"],
                "doc_id": row_dict["doc_id"],
                "step_type": row_dict["step_type"],
                "model_tag": row_dict["model_tag"]
                if row_dict["model_tag"] is not None
                else None,
                "semantic_id": row_dict["semantic_id"],
                "technical_id": row_dict["technical_id"],
                "created_at": row_dict["created_at"],
                "updated_at": row_dict["updated_at"],
                "operator": row_dict["operator"],
            }
        finally:
            _safe_close_table(table)

    def list_main_pointers(
        self,
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List main pointers (sync)."""
        if user_id is not None:
            logger.warning(
                "user_id parameter provided to list_main_pointers but "
                "main_pointers table does not have user_id field. "
                "Schema migration required for multi-tenancy support."
            )

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("main_pointers")

        try:
            filters_dict = {"collection": collection}
            if doc_id is not None:
                filters_dict["doc_id"] = doc_id

            filter_expr = build_lancedb_filter_expression(filters_dict)

            # First check if any pointers exist using efficient count_rows
            if table.search().where(filter_expr).count_rows() == 0:
                return []

            result = table.search().where(filter_expr).limit(limit).to_arrow()

            pointers = []
            for row_dict in result.to_pylist():
                pointers.append(
                    {
                        "collection": row_dict["collection"],
                        "doc_id": row_dict["doc_id"],
                        "step_type": row_dict["step_type"],
                        "model_tag": row_dict["model_tag"]
                        if row_dict["model_tag"] is not None
                        else None,
                        "semantic_id": row_dict["semantic_id"],
                        "technical_id": row_dict["technical_id"],
                        "created_at": row_dict["created_at"],
                        "updated_at": row_dict["updated_at"],
                        "operator": row_dict["operator"],
                    }
                )

            return pointers
        finally:
            _safe_close_table(table)

    def delete_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        """Delete a main pointer (sync)."""
        if user_id is not None:
            logger.warning(
                "user_id parameter provided to delete_main_pointer but "
                "main_pointers table does not have user_id field. "
                "Schema migration required for multi-tenancy support."
            )

        from ..LanceDB.schema_manager import _safe_close_table

        conn = self._get_sync_connection()
        self._ensure_table()
        table = conn.open_table("main_pointers")

        try:
            # Build filter expression using FilterCondition
            base_conditions: List[FilterCondition] = [
                FilterCondition(
                    field="collection", operator=FilterOperator.EQ, value=collection
                ),
                FilterCondition(
                    field="doc_id", operator=FilterOperator.EQ, value=doc_id
                ),
                FilterCondition(
                    field="step_type", operator=FilterOperator.EQ, value=step_type
                ),
            ]

            normalized_tag = self._normalize_model_tag(model_tag)
            if normalized_tag == "":
                # Check for both empty string AND NULL (backward compatibility)
                model_tag_null_cond = FilterCondition(
                    field="model_tag", operator=FilterOperator.IS_NULL, value=None
                )
                model_tag_empty_cond = FilterCondition(
                    field="model_tag", operator=FilterOperator.EQ, value=""
                )
                # Combine as: (base) AND (model_tag IS NULL OR model_tag == '')
                model_tag_filter: FilterExpression = (
                    model_tag_null_cond,
                    model_tag_empty_cond,
                )  # OR tuple
                filter_expr: FilterExpression = (
                    *base_conditions,
                    model_tag_filter,
                )  # AND tuple
            else:
                base_conditions.append(
                    FilterCondition(
                        field="model_tag",
                        operator=FilterOperator.EQ,
                        value=normalized_tag,
                    )
                )
                filter_expr = tuple(base_conditions)  # AND tuple

            # Translate to LanceDB syntax using shared utility
            filter_str = translate_filter_expression(filter_expr)

            # Check if exists
            result = table.search().where(filter_str).to_arrow()
            if len(result) == 0:
                return False

            table.delete(filter_str)
            logger.info(
                "Deleted main pointer for %s/%s/%s", collection, doc_id, step_type
            )
            return True
        finally:
            _safe_close_table(table)

    # --- Async methods (delegate to sync) ---

    async def set_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Async version of set_main_pointer."""
        return self.set_main_pointer(
            collection,
            doc_id,
            step_type,
            semantic_id,
            technical_id,
            model_tag,
            operator,
            user_id,
        )

    async def get_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async version of get_main_pointer."""
        return self.get_main_pointer(collection, doc_id, step_type, model_tag, user_id)

    async def list_main_pointers_async(
        self,
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Async version of list_main_pointers."""
        return self.list_main_pointers(collection, doc_id, user_id, limit)

    async def delete_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        """Async version of delete_main_pointer."""
        return self.delete_main_pointer(
            collection, doc_id, step_type, model_tag, user_id
        )

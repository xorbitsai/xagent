"""Warm up a target embedding model's vectors for collections that stay online.

    python -m xagent.migrations.embedding.embed --model TARGET \
        [--collection NAME ...] [--dry-run] [--rate-limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.tools.core.RAG_tools.core.exceptions import CollectionMigratingError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    ChunkEmbeddingData,
)
from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_candidates,
    to_model_tag,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.core.tools.core.RAG_tools.management.migration_state import (
    ensure_collection_writable,
    leased_collections,
    verify_active_leases,
)
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from xagent.core.tools.core.RAG_tools.utils.embedding_utils import (
    normalize_raw_embedding_to_vectors,
)
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    write_vectors_to_db,
)
from xagent.migrations.embedding.scope import (
    CORE_TABLE,
    SKIP_NO_CORE_CHUNK,
    SKIP_NO_METADATA,
    MigrationCliError,
    MissingCollectionError,
    SearchableChunk,
    _owner,
    _usable_vector,
    chunk_key,
    collection_filter,
    collection_info,
    connection,
    core_chunk_pairs,
    holds_rows,
    iter_rows,
    precompute_eligibility,
    resolve_collection_names,
    skip_reason,
    source_table,
    table_names,
    target_chunk_ids,
)

logger = logging.getLogger(__name__)

# The core table holds no row of the source row's document, or there is no table.
SKIP_NO_CORE_DOCUMENT = "no-core-document"
# The write path stores rows under the stripped name, where the re-check would miss them.
SKIP_NAME_NOT_STRIPPED = "name-not-stripped"
STOP_FROZEN = "frozen"
STOP_REBOUND = "binding-changed"
LEASE_OPERATION = "embedding-warmup"

WRITE_COLUMNS = tuple(
    "doc_id chunk_id user_id text parse_hash chunk_hash metadata".split()
)

# A broken provider or credential fails every document alike.
MAX_CONSECUTIVE_DOCUMENT_FAILURES = 5

FAILED_INDEX_STATUSES = ("failed", "index_corrupted")

# Each compaction call scans every retained version, so calling it per document
# costs more than the compaction saves.
COMPACT_EVERY_DOCUMENTS = 100


class RateLimiter:
    """Paces encode calls at ``rate`` texts per second on average; 0 is unlimited."""

    def __init__(self, rate: float) -> None:
        self._rate = float(rate)
        self._next_start = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, count: int) -> None:
        if self._rate <= 0 or count <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + count / self._rate
        if start > now:
            time.sleep(start - now)


class CollectionStopped(Exception):
    """The collection froze or was rebound; warm-up must not write it again."""


class TargetTableMissing(MigrationCliError):
    """A write now would create the target's table, and with it a search route."""


@dataclass(frozen=True)
class Target:
    model_id: str
    dimension: int
    adapter: Optional[BaseEmbedding]
    limiter: RateLimiter


@dataclass
class CollectionReport:
    """One collection: the rows to write per the scan, then what the run did."""

    name: str
    source_model_id: Optional[str] = None
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    documents: int = 0
    pending: int = 0
    written: int = 0
    removed: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    index_error: Optional[str] = None
    aborted: bool = False
    stopped: Optional[str] = None

    @property
    def failed(self) -> bool:
        return bool(self.error or self.failures or self.index_error)


def warm_up(
    model_id: str,
    *,
    collections: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    rate_limit: float = 0.0,
) -> List[CollectionReport]:
    """Process every collection in scope; a failed one does not stop the next, an
    aborted one does."""
    target, conn, names = start_run(
        model_id, collections=collections, dry_run=dry_run, rate_limit=rate_limit
    )
    reports: List[CollectionReport] = []
    for name in names:
        report = CollectionReport(name=name)
        reports.append(report)
        try:
            warm_collection(report, target, conn)
        except MissingCollectionError:
            report.skip_reason = SKIP_NO_METADATA
        except Exception as exc:  # noqa: BLE001 - kept on the collection, exit 1
            report.error = f"{type(exc).__name__}: {exc}"
        log_report(report, dry_run=dry_run)
        if report.aborted:
            logger.error("Run aborted after %s", name)
            break
    return reports


def start_run(
    model_id: str,
    *,
    collections: Optional[Sequence[str]],
    dry_run: bool,
    rate_limit: float,
) -> Tuple[Target, Any, List[str]]:
    """The target, the connection and the collections in scope; raises when the
    run must not start."""
    target_id, dimension, adapter = resolve_hub_target(model_id, dry_run=dry_run)
    conn = connection()
    names = resolve_collection_names(collections=collections, conn=conn)
    require_target_table(target_id, conn)
    refuse_owner_migration(target_id, conn)
    if adapter is not None and rate_limit <= 0:
        logger.warning("No --rate-limit: the provider's whole quota may be used")
    return Target(target_id, dimension, adapter, RateLimiter(rate_limit)), conn, names


def resolve_hub_target(
    model_id: str, *, dry_run: bool
) -> Tuple[str, int, Optional[BaseEmbedding]]:
    """The hub's canonical id and dimension, never an environment fallback, and the
    adapter built from that one loaded config unless ``dry_run``."""
    from xagent.core.tools.core.RAG_tools.management.embedding_switch import (
        _resolve_configured_model,
    )
    from xagent.core.tools.core.RAG_tools.utils import model_resolver

    canonical, dimension, config = _resolve_configured_model(model_id)
    if dry_run:
        return canonical, dimension, None
    return canonical, dimension, model_resolver.create_embedding_adapter(config)


def require_target_table(model_id: str, conn: Any) -> None:
    table = source_table(model_id)
    if table not in table_names(conn):
        raise TargetTableMissing(
            f"{table} does not exist; this command writes only into an existing "
            "table, which ingesting into a collection bound to the target creates"
        )


def refuse_owner_migration(model_id: str, conn: Any) -> None:
    """The write path converts a ``user_id`` column whose type name lacks ``int`` by
    dropping the table and creating it again; a run must not set that off."""
    table = source_table(model_id)
    opened = conn.open_table(table)
    try:
        schema = opened.schema
    finally:
        _safe_close_table(opened)
    if "user_id" not in schema.names:
        return
    owner_type = str(schema.field("user_id").type)
    if "int" not in owner_type.lower():
        raise MigrationCliError(
            f"{table} stores user_id as {owner_type}, a type the write path "
            "converts by dropping and recreating the table"
        )


def warm_collection(report: CollectionReport, target: Target, conn: Any) -> None:
    """Write the target's vectors for one collection's backed, usable source rows."""
    name = report.name
    if name != name.strip():
        report.skip_reason = SKIP_NAME_NOT_STRIPPED
        return
    eligibility = precompute_eligibility(
        name, conn=conn, target_model_id=target.model_id
    )
    if eligibility.reason:
        report.skip_reason = eligibility.reason
        return
    source = eligibility.model_id
    assert source is not None
    report.source_model_id = source
    have = target_chunk_ids(
        name, target.model_id, conn=conn, dimension=target.dimension
    )
    pending = pending_rows(report, source, have, conn)
    report.documents = len(pending)
    report.pending = sum(len(rows) for rows in pending.values())
    if target.adapter is None:
        return
    consecutive = 0
    for index, doc_id in enumerate(sorted(pending), start=1):
        try:
            with leased(report, conn):
                warm_document(report, doc_id, pending[doc_id], target, conn)
                if index % COMPACT_EVERY_DOCUMENTS == 0:
                    compact_target(report.name, target.model_id)
        except CollectionStopped as exc:
            report.stopped = str(exc)
            return
        except TargetTableMissing as exc:
            report.error = str(exc)
            report.aborted = True
            return
        except Exception as exc:  # noqa: BLE001 - kept on the document
            report.failures[doc_id] = f"{type(exc).__name__}: {exc}"
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_DOCUMENT_FAILURES:
                report.aborted = True
                return
            continue
        consecutive = 0
        logger.info("%s: %d/%d documents", name, index, len(pending))
    try:
        build_indexes(report, target.model_id, conn)
    except CollectionStopped as exc:
        report.stopped = str(exc)


@contextmanager
def leased(report: CollectionReport, conn: Any) -> Iterator[None]:
    """A write lease taken while the collection is not frozen and still bound."""
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        _normalize_model_id,
    )

    with leased_collections(report.name, operation=LEASE_OPERATION):
        try:
            ensure_collection_writable(report.name)
        except CollectionMigratingError as exc:
            raise CollectionStopped(STOP_FROZEN) from exc
        try:
            bound = _normalize_model_id(
                collection_info(report.name, conn).embedding_model_id
            )
        except MissingCollectionError:
            bound = None
        if bound != report.source_model_id:
            raise CollectionStopped(STOP_REBOUND)
        yield


def pending_rows(
    report: CollectionReport,
    source: str,
    have: Set[Tuple[str, str]],
    conn: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """Usable source rows a core chunk row backs and the target does not hold, by
    document; every other row is counted in ``report.skipped``."""
    core = core_chunk_pairs(report.name, conn=conn) or set()
    core_docs = {doc_id for doc_id, _ in core}
    table = source_table(source)
    pending: Dict[str, List[Dict[str, Any]]] = {}
    if table not in table_names(conn):
        return pending
    for row in iter_rows(conn, table, collection_filter(report.name), WRITE_COLUMNS):
        key = chunk_key(row)
        reason = row_skip_reason(row, core, core_docs)
        if reason:
            report.skipped[reason] = report.skipped.get(reason, 0) + 1
        elif key not in have:
            pending.setdefault(key[0], []).append(row)
    return pending


def row_skip_reason(
    row: Dict[str, Any], core: Set[Tuple[str, str]], core_docs: Set[str]
) -> Optional[str]:
    """Why a source row cannot be written to a target; None for a writable one."""
    key = chunk_key(row)
    reason = skip_reason(
        SearchableChunk(
            doc_id=key[0],
            chunk_id=key[1],
            text=str(row.get("text") or ""),
            user_id=_owner(row.get("user_id")),
        )
    )
    if reason is None and key not in core:
        reason = SKIP_NO_CORE_CHUNK if key[0] in core_docs else SKIP_NO_CORE_DOCUMENT
    return reason


def warm_document(
    report: CollectionReport,
    doc_id: str,
    rows: List[Dict[str, Any]],
    target: Target,
    conn: Any,
) -> None:
    """Write ``rows``, then remove those whose core chunk row is gone meanwhile."""
    attempted = {chunk_key(row) for row in rows}
    try:
        write_rows(report, rows, target, conn)
    except Exception:
        try:
            report.removed += drop_unbacked(
                report.name, doc_id, attempted, target.model_id, conn
            )
        except Exception:  # noqa: BLE001 - the write's error is the document's
            logger.exception(
                "%s/%s: re-check after a failed write", report.name, doc_id
            )
        raise
    # A delete whose second sweep ran before this write committed cannot see it.
    report.removed += drop_unbacked(
        report.name, doc_id, attempted, target.model_id, conn
    )


def write_rows(
    report: CollectionReport,
    rows: List[Dict[str, Any]],
    target: Target,
    conn: Any,
    *,
    before_write: Optional[Callable[[], Any]] = None,
) -> None:
    assert target.adapter is not None
    by_owner: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        # Every row is checked before the first provider call, not batch by batch.
        record(row, target.model_id)
        by_owner.setdefault(int(row["user_id"]), []).append(row)
    for owner, items in by_owner.items():
        for start in range(0, len(items), DEFAULT_EMBEDDING_BATCH_SIZE):
            batch = items[start : start + DEFAULT_EMBEDDING_BATCH_SIZE]
            texts = [str(row["text"]) for row in batch]
            target.limiter.acquire(len(texts))
            vectors = normalize_raw_embedding_to_vectors(target.adapter.encode(texts))
            if len(vectors) != len(texts) or not all(
                _usable_vector(vector, target.dimension) for vector in vectors
            ):
                raise ValueError(f"unusable provider answer for {len(texts)} texts")
            verify_active_leases(report.name)
            if before_write is not None:
                before_write()
            require_target_table(target.model_id, conn)
            write_vectors_to_db(
                report.name,
                [record(row, target.model_id, v) for row, v in zip(batch, vectors)],
                create_index=False,
                user_id=owner,
                verify_bound=False,
            )
            report.written += len(batch)


def record(row: Dict[str, Any], model: str, vector: Any = None) -> ChunkEmbeddingData:
    """What ``write_rows`` writes for a source row; raises for a row it cannot write."""
    keys = dict(zip(("doc_id", "chunk_id"), chunk_key(row)), model=model)
    meta = source_metadata(row["metadata"])
    data = {**row, **keys, "text": str(row["text"]), "metadata": meta}
    return ChunkEmbeddingData(**data, vector=vector or [])


def source_metadata(raw: Any) -> Optional[Dict[str, Any]]:
    """The source row's metadata object; anything but none or a JSON object raises."""
    if raw is None or not str(raw).strip():
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("source metadata is not a JSON object")
    return value


def drop_unbacked(
    collection: str,
    doc_id: str,
    attempted: Set[Tuple[str, str]],
    model_id: str,
    conn: Any,
) -> int:
    """Delete the writer-table rows of ``attempted`` whose core chunk row is gone."""
    document = (
        f"{collection_filter(collection)} "
        f"AND doc_id = '{escape_lancedb_string(doc_id)}'"
    )
    live: Set[Tuple[str, str]] = set()
    if CORE_TABLE in table_names(conn):
        live = {
            chunk_key(row)
            for row in iter_rows(conn, CORE_TABLE, document, ["doc_id", "chunk_id"])
        }
    gone = sorted(chunk_id for _, chunk_id in attempted - live)
    table = source_table(model_id)
    if not gone or table not in table_names(conn):
        return 0
    ids = ", ".join(f"'{escape_lancedb_string(chunk_id)}'" for chunk_id in gone)
    where = f"{document} AND chunk_id IN ({ids})"
    opened = conn.open_table(table)
    try:
        count = int(opened.count_rows(where))
        if count:
            verify_active_leases(collection)
            opened.delete(where)
        return count
    finally:
        _safe_close_table(opened)


def build_indexes(report: CollectionReport, model_id: str, conn: Any) -> None:
    """The index call ingestion's final batch makes, if the target holds rows here."""
    table = source_table(model_id)
    if table not in table_names(conn) or not holds_rows(conn, table, report.name):
        return
    with leased(report, conn):
        status = index_target(report.name, model_id)
    if status in FAILED_INDEX_STATUSES:
        report.index_error = f"index build on {table} ended {status}"


def index_target(collection: str, model_id: str) -> str:
    store = get_vector_index_store()
    status = store.create_index(to_model_tag(model_id), readonly=False).status
    compact_target(collection, model_id)
    return status


def compact_target(collection: str, model_id: str) -> None:
    """Ingestion's post-write compaction, on the target's tables only."""
    verify_active_leases(collection)
    tables = list(dict.fromkeys(embeddings_table_candidates(model_id)))
    try:
        get_vector_index_store().compact_tables(tables)
    except Exception as exc:  # noqa: BLE001 - maintenance never fails a write
        logger.warning("Compaction of %s skipped: %s", ", ".join(tables), exc)


def describe(report: CollectionReport, *, dry_run: bool) -> str:
    if report.skip_reason:
        return f"{report.name}: skipped ({report.skip_reason})"
    if report.error:
        return f"{report.name}: failed: {report.error}"
    parts = [
        f"source model {report.source_model_id}",
        f"{report.pending} rows in {report.documents} documents to write",
    ]
    if not dry_run:
        parts += [f"wrote {report.written}", f"removed {report.removed}"]
    if report.skipped:
        parts.append(
            "skipped "
            + ", ".join(f"{k}={v}" for k, v in sorted(report.skipped.items()))
        )
    if report.failures:
        parts.append(f"{len(report.failures)} documents failed")
    if report.index_error:
        parts.append(report.index_error)
    if report.stopped:
        parts.append(f"stopped ({report.stopped})")
    return f"{report.name}: " + "; ".join(parts)


def log_report(report: CollectionReport, *, dry_run: bool) -> None:
    log = logger.error if report.failed else logger.info
    log("%s", describe(report, dry_run=dry_run))
    for doc_id, error in sorted(report.failures.items()):
        logger.error("%s/%s: %s", report.name, doc_id, error)


def log_summary(reports: List[CollectionReport]) -> None:
    stopped = ", ".join(f"{r.name} ({r.stopped})" for r in reports if r.stopped)
    failed = ", ".join(r.name for r in reports if r.failed)
    logger.info(
        "Warm-up done: %d collections; stopped: %s; failed: %s; skipped: %d",
        len(reports),
        stopped or "none",
        failed or "none",
        sum(1 for r in reports if r.skip_reason),
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Write a target embedding model's vectors for knowledge-base collections.\n\n"
            "For each LanceDB collection in scope that stays bound to its model, the\n"
            "usable source rows a core chunk row backs are embedded with the target\n"
            "model and written to its table, one document at a time under a write\n"
            "lease. No binding or source row changes; a rerun skips what the target\n"
            "already holds.\n\n"
            "Not a completeness claim: chunks ingested after a collection was read\n"
            "are not written, and switching a collection is left to its migration.\n\n"
            "A collection can be skipped (unbound, target-is-source, other-engine,\n"
            "name-not-stripped, no-metadata, ...); rows are counted per skip reason\n"
            "(no-core-chunk, no-core-document, no-owner, ...). A collection stops\n"
            "when it is frozen for migration ('frozen'), or its binding changed or\n"
            "its metadata row is gone ('binding-changed'); a stop is not a failure.\n"
            "Five failed documents in a row in one collection abort the run.\n\n"
            "Exit codes: 0 every collection in scope was processed, skipped or\n"
            "stopped (0 does not mean every collection was warmed up); 1 a\n"
            "collection, a document or an index build failed, nothing is in\n"
            "scope, or the target's table was gone before a write, which aborts\n"
            "the run; 2 the run never started, including when the target's table\n"
            "does not exist or its user_id column is of a type the write path\n"
            "converts. That table is looked up before the run and before every\n"
            "write; ingesting into a collection bound to the target model creates\n"
            "it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, help="Target embedding model id from the model hub"
    )
    parser.add_argument(
        "--collection",
        action="append",
        metavar="NAME",
        help="Restrict to this collection; repeatable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count what would be written; calls no provider and writes nothing "
        "to LanceDB (opening the model hub may create its tables)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        metavar="N",
        help="Texts sent per second at most, on average, each batch counted once; "
        "0 = unlimited, with a warning. A failed provider call is not retried.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.rate_limit) or args.rate_limit < 0:
        logger.error("--rate-limit must be a finite, non-negative number")
        return 2
    try:
        reports = warm_up(
            args.model,
            collections=args.collection,
            dry_run=args.dry_run,
            rate_limit=args.rate_limit,
        )
    except Exception as exc:  # noqa: BLE001 - the run never started
        logger.error("Warm-up did not start: %s", exc)
        return 2
    if not reports:
        logger.warning("No collections in scope")
        return 1
    log_summary(reports)
    return 1 if any(report.failed for report in reports) else 0


if __name__ == "__main__":
    sys.exit(main())

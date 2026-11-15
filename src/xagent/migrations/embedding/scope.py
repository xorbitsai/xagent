"""Read-only scope of an embedding migration, per knowledge-base collection.

    python -m xagent.migrations.embedding.scope [--model TARGET] \
        [--collection NAME ...]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import lancedb

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_candidates,
    written_embeddings_table_name,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import list_table_names
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string
from xagent.providers.vector_store.lancedb import LanceDBConnectionManager

logger = logging.getLogger(__name__)

METADATA_TABLE = "collection_metadata"
CORE_TABLE = "chunks"

# Why a collection is not counted.
SKIP_OTHER_ENGINE = "other-engine"
SKIP_UNREADABLE_BINDING = "unreadable-binding"
SKIP_UNBOUND = "unbound"
SKIP_BOUND_WITHOUT_DIMENSION = "bound-without-dimension"
SKIP_ROWS_IN_OTHER_TABLE = "rows-in-another-table"
SKIP_ROWS_IN_MULTIPLE_TABLES = "rows-in-multiple-tables"
SKIP_MISSING_COLUMN = "missing-required-column"
SKIP_SHARED_TABLE = "shared-source-target-table"
SKIP_TARGET_ROWS_IN_OTHER_TABLE = "target-rows-in-another-table"
SKIP_TARGET_IS_SOURCE = "target-is-source"
SKIP_NO_METADATA = "no-metadata"
# Why a source row is not counted as missing.
SKIP_NO_DOC_ID = "no-doc-id"
SKIP_NO_CHUNK_ID = "no-chunk-id"
SKIP_NO_OWNER = "no-owner"
SKIP_NO_TEXT = "no-text"
SKIP_NO_CORE_CHUNK = "no-core-chunk"

REQUIRED_COLUMNS = ("collection", "doc_id", "chunk_id", "user_id", "text")

READ_BATCH_SIZE = 1000
EMBEDDINGS_PREFIX = "embeddings_"


class MigrationCliError(RuntimeError):
    """A problem the command reports: exit 2 before any collection is read,
    otherwise an error on the collection it concerns."""


class MissingCollectionError(MigrationCliError):
    """The collection has no metadata row (never existed, or was just deleted)."""


@dataclass(frozen=True)
class SearchableChunk:
    """One row of the source model's embeddings table."""

    doc_id: str
    chunk_id: str
    text: str
    user_id: Optional[int]


@dataclass(frozen=True)
class Eligibility:
    """Whether one collection can be counted: ``reason`` is ``None`` only then."""

    model_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class CollectionScope:
    """One collection's report line."""

    name: str
    source_model_id: Optional[str] = None
    skip_reason: Optional[str] = None
    usable_documents: int = 0
    chunks: int = 0
    # ``None`` unless the collection was counted against a target.
    missing_chunks: Optional[int] = None
    skipped: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.skip_reason is None and self.error is None


def connection() -> Any:
    """The application's LanceDB directory, opened only if it already exists."""
    db_dir = LanceDBConnectionManager().resolve_dir_from_env(create=False)
    if not Path(db_dir).is_dir():
        raise MigrationCliError(f"LanceDB directory {db_dir} is not a directory")
    return lancedb.connect(db_dir)


def table_names(conn: Any) -> Set[str]:
    return set(list_table_names(conn))


def source_table(model_id: str) -> str:
    return written_embeddings_table_name(model_id)


def iter_rows(
    conn: Any,
    table: str,
    where: str,
    columns: Sequence[str],
) -> Iterator[Dict[str, Any]]:
    """Stream rows of ``table``; a failed read or a missing column raises."""
    opened = conn.open_table(table)
    try:
        missing = [name for name in columns if name not in opened.schema.names]
        if missing:
            raise MigrationCliError(f"table '{table}' has no {', '.join(missing)}")
        reader = (
            opened.search()
            .where(where)
            .select(list(columns))
            .limit(None)
            .to_batches(READ_BATCH_SIZE)
        )
        for batch in reader:
            yield from batch.to_pylist()
    finally:
        _safe_close_table(opened)


def schema_names(conn: Any, table: str) -> List[str]:
    opened = conn.open_table(table)
    try:
        return list(opened.schema.names)
    finally:
        _safe_close_table(opened)


def collection_filter(collection: str) -> str:
    return f"collection = '{escape_lancedb_string(collection)}'"


def chunk_key(row: Dict[str, Any]) -> Tuple[str, str]:
    """The ``(doc_id, chunk_id)`` every reader compares, exactly as stored."""
    return (str(row.get("doc_id") or ""), str(row.get("chunk_id") or ""))


def collection_info(collection: str, conn: Optional[Any] = None) -> Any:
    """The metadata row; only an absent row is :class:`MissingCollectionError`."""
    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo

    conn = conn if conn is not None else connection()
    if METADATA_TABLE not in table_names(conn):
        raise MissingCollectionError(f"collection '{collection}' does not exist")
    table = conn.open_table(METADATA_TABLE)
    try:
        rows = (
            table.search()
            .where(f"name = '{escape_lancedb_string(collection)}'")
            .to_arrow()
            .to_pylist()
        )
    finally:
        _safe_close_table(table)
    if not rows:
        raise MissingCollectionError(f"collection '{collection}' does not exist")
    try:
        return CollectionInfo.from_storage(rows[0])
    except Exception as exc:  # noqa: BLE001 - an error on the collection
        raise MigrationCliError(
            f"collection '{collection}' metadata does not decode: {exc}"
        ) from exc


def collection_engine(info: Any) -> str:
    """The engine a collection is bound to; ``""`` for a malformed binding."""
    from xagent.core.tools.core.RAG_tools.kb.coordinator import (
        get_kb_coordinator,
        resolve_collection_backend,
    )

    # Built outside the try: only a malformed binding may become a reason.
    get_kb_coordinator()
    try:
        return str(resolve_collection_backend(info).value)
    except ValueError:
        return ""


def own_tables(model_id: str) -> List[str]:
    """The tables search tries for ``model_id`` before asking the model hub."""
    return embeddings_table_candidates(model_id)


def written_by(row: Dict[str, Any], model_id: str) -> bool:
    """Whether the row's ``model`` is ``model_id``: distinct ids can share a writer
    table (``old-model`` and ``old_model``)."""
    return row.get("model") == model_id


def _count_rows(conn: Any, table: str, where: Optional[str] = None) -> int:
    opened = conn.open_table(table)
    try:
        return int(opened.count_rows(where) if where else opened.count_rows())
    finally:
        _safe_close_table(opened)


def holds_rows(conn: Any, table: str, collection: str) -> bool:
    if "collection" not in schema_names(conn, table):
        # Cannot attribute a row to a collection: any row counts.
        return _count_rows(conn, table) > 0
    return _count_rows(conn, table, collection_filter(collection)) > 0


def _holds_rows_written_elsewhere(
    conn: Any, table: str, collection: str, model: str
) -> bool:
    """Whether ``table`` holds rows of this collection that ``model`` did not
    write; without a ``collection`` column any row counts."""
    if "collection" not in schema_names(conn, table):
        return _count_rows(conn, table) > 0
    if "model" not in schema_names(conn, table):
        return _count_rows(conn, table, collection_filter(collection)) > 0
    foreign = (
        f"{collection_filter(collection)} AND "
        f"(model IS NULL OR model != '{escape_lancedb_string(model)}')"
    )
    return _count_rows(conn, table, foreign) > 0


def _embeddings_tables(names: Set[str]) -> List[str]:
    return sorted(name for name in names if name.startswith(EMBEDDINGS_PREFIX))


def target_tables_hold_foreign_rows(
    conn: Any, collection: str, model_id: str, names: Set[str], source: List[str]
) -> bool:
    """Whether search for ``model_id`` could serve rows of this collection other
    than those ``model_id`` wrote in its writer table."""
    present = [table for table in own_tables(model_id) if table in names]
    if not present:
        # Search would ask the model hub for a table this command cannot name.
        return any(
            holds_rows(conn, table, collection)
            for table in _embeddings_tables(names)
            if table not in source
        )
    writer = source_table(model_id)
    for table in present:
        if table != writer:
            if holds_rows(conn, table, collection):
                return True
        elif _holds_rows_written_elsewhere(conn, table, collection, model_id):
            return True
    return False


def missing_columns(conn: Any, table: str) -> List[str]:
    present = set(schema_names(conn, table))
    return [name for name in REQUIRED_COLUMNS if name not in present]


def precompute_eligibility(
    collection: str,
    *,
    conn: Optional[Any] = None,
    target_model_id: Optional[str] = None,
) -> Eligibility:
    """:func:`_source_eligibility`, also refusing a target that is the bound model."""
    result = _source_eligibility(collection, conn=conn, target_model_id=target_model_id)
    if result.reason is None and target_model_id and target_model_id == result.model_id:
        return Eligibility(reason=SKIP_TARGET_IS_SOURCE)
    return result


def _source_eligibility(
    collection: str,
    *,
    conn: Optional[Any] = None,
    target_model_id: Optional[str] = None,
) -> Eligibility:
    """Eligible only when bound with a dimension, with no row search could serve
    from a table but the writer's, and (with a target) no shared or foreign table."""
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        _normalize_model_id,
    )

    conn = conn if conn is not None else connection()
    info = collection_info(collection, conn)
    engine = collection_engine(info)
    if engine != "lancedb":
        return Eligibility(
            reason=SKIP_OTHER_ENGINE if engine else SKIP_UNREADABLE_BINDING
        )
    model = _normalize_model_id(info.embedding_model_id)
    if not model:
        return Eligibility(reason=SKIP_UNBOUND)
    dimension = info.embedding_dimension
    if not isinstance(dimension, int) or dimension <= 0:
        return Eligibility(reason=SKIP_BOUND_WITHOUT_DIMENSION)
    table = source_table(model)
    own = own_tables(model)
    names = table_names(conn)
    if table in names and missing_columns(conn, table):
        return Eligibility(reason=SKIP_MISSING_COLUMN)
    if target_model_id and target_model_id != model:
        if set(own) & set(own_tables(target_model_id)):
            return Eligibility(reason=SKIP_SHARED_TABLE)
        if target_tables_hold_foreign_rows(
            conn, collection, target_model_id, names, own
        ):
            return Eligibility(reason=SKIP_TARGET_ROWS_IN_OTHER_TABLE)
    if not any(name in names for name in own):
        # Search would ask the model hub for a table this command cannot name.
        candidates = _embeddings_tables(names)
        if any(holds_rows(conn, name, collection) for name in candidates):
            return Eligibility(reason=SKIP_ROWS_IN_OTHER_TABLE)
        return Eligibility(model_id=model)
    holders = [
        name for name in own if name in names and holds_rows(conn, name, collection)
    ]
    if table not in holders:
        if holders:
            return Eligibility(reason=SKIP_ROWS_IN_OTHER_TABLE)
        return Eligibility(model_id=model)
    if len(holders) > 1:
        return Eligibility(reason=SKIP_ROWS_IN_MULTIPLE_TABLES)
    return Eligibility(model_id=model)


@dataclass
class SourceScan:
    """One pass over a collection's source rows; ``pending`` counts the usable rows
    the target does not hold."""

    usable_documents: Set[str] = field(default_factory=set)
    pending: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    rows: int = 0


def scan_source(
    collection: str,
    model_id: str,
    *,
    have: Optional[Set[Tuple[str, str]]] = None,
    core: Optional[Set[Tuple[str, str]]] = None,
    conn: Optional[Any] = None,
) -> SourceScan:
    """Classify every source row once; without ``core`` (:func:`core_chunk_pairs`)
    no row is checked against the core table."""
    conn = conn if conn is not None else connection()
    scan = SourceScan()
    core_keys = core if core is not None else set()
    core_docs = {doc_id for doc_id, _ in core_keys}
    table = source_table(model_id)
    if table not in table_names(conn):
        return scan
    for row in iter_rows(
        conn,
        table,
        collection_filter(collection),
        ["doc_id", "chunk_id", "user_id", "text"],
    ):
        scan.rows += 1
        key = chunk_key(row)
        chunk = SearchableChunk(
            doc_id=key[0],
            chunk_id=key[1],
            text=str(row.get("text") or ""),
            user_id=_owner(row.get("user_id")),
        )
        reason = skip_reason(chunk)
        if reason is None and chunk.doc_id in core_docs and key not in core_keys:
            reason = SKIP_NO_CORE_CHUNK
        if reason:
            scan.skipped[reason] = scan.skipped.get(reason, 0) + 1
            continue
        scan.usable_documents.add(chunk.doc_id)
        if have is not None and key not in have:
            scan.pending += 1
    return scan


def core_chunk_pairs(
    collection: str, *, conn: Optional[Any] = None
) -> Optional[Set[Tuple[str, str]]]:
    """Every ``(doc_id, chunk_id)`` the core chunks table holds, or ``None`` without one."""
    conn = conn if conn is not None else connection()
    if CORE_TABLE not in table_names(conn):
        return None
    return {
        chunk_key(row)
        for row in iter_rows(
            conn,
            CORE_TABLE,
            collection_filter(collection),
            ["doc_id", "chunk_id"],
        )
    }


def skip_reason(chunk: SearchableChunk) -> Optional[str]:
    """Why this source row is excluded, or ``None`` when it is usable."""
    if not chunk.doc_id.strip():
        return SKIP_NO_DOC_ID
    if not chunk.chunk_id.strip():
        return SKIP_NO_CHUNK_ID
    if chunk.user_id is None:
        return SKIP_NO_OWNER
    if not chunk.text:
        return SKIP_NO_TEXT
    return None


def _owner(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def target_chunk_ids(
    collection: str,
    model_id: str,
    *,
    conn: Optional[Any] = None,
    dimension: Optional[int] = None,
) -> Set[Tuple[str, str]]:
    """The ``(doc_id, chunk_id)`` of each row in the target's writer table that
    ``model_id`` wrote with a finite, non-zero vector (of ``dimension``, if given)."""
    conn = conn if conn is not None else connection()
    table = source_table(model_id)
    if table not in table_names(conn) or "collection" not in schema_names(conn, table):
        return set()
    held: Set[Tuple[str, str]] = set()
    columns = ["doc_id", "chunk_id", "vector", "model"]
    for row in iter_rows(conn, table, collection_filter(collection), columns):
        if not written_by(row, model_id):
            continue
        if not _usable_vector(row.get("vector"), dimension):
            continue
        held.add(chunk_key(row))
    return held


def _usable_vector(vector: Any, dimension: Optional[int]) -> bool:
    if not isinstance(vector, list) or not vector:
        return False
    if dimension is not None and len(vector) != dimension:
        return False
    if not all(isinstance(value, (int, float)) for value in vector):
        return False
    return all(math.isfinite(value) for value in vector) and any(vector)


def resolve_collection_names(
    *, collections: Optional[Sequence[str]] = None, conn: Optional[Any] = None
) -> List[str]:
    """The collections this run covers; a named one that does not exist raises."""
    conn = conn if conn is not None else connection()
    existing = _all_collection_names(conn)
    if not collections:
        return existing
    names: List[str] = []
    for name in collections:
        if not name.strip():
            raise MigrationCliError("--collection must not be empty")
        if name not in existing:
            raise MissingCollectionError(f"collection '{name}' does not exist")
        if name not in names:
            names.append(name)
    return names


def _all_collection_names(conn: Any) -> List[str]:
    """Every collection name; ``list_collections`` would create missing tables."""
    if METADATA_TABLE not in table_names(conn):
        return []
    return sorted(
        {
            str(row.get("name") or "")
            for row in iter_rows(conn, METADATA_TABLE, "true", ["name"])
            if row.get("name")
        }
    )


def resolve_target(model_id: str) -> str:
    """The target id, normalised like a binding; a blank id raises."""
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        _normalize_model_id,
    )

    target = _normalize_model_id(model_id)
    if not target:
        raise MigrationCliError("--model must name a model")
    return target


def scope_collection(
    name: str,
    *,
    conn: Optional[Any] = None,
    target: Optional[str] = None,
) -> CollectionScope:
    """Count one collection's source rows and, with a target, those it lacks."""
    target_model_id = target
    conn = conn if conn is not None else connection()
    eligibility = precompute_eligibility(
        name, conn=conn, target_model_id=target_model_id
    )
    item = CollectionScope(name=name)
    if eligibility.reason:
        item.skip_reason = eligibility.reason
        return item
    source_model_id = eligibility.model_id
    assert source_model_id is not None
    item.source_model_id = source_model_id
    have: Optional[Set[Tuple[str, str]]] = None
    if target_model_id:
        have = target_chunk_ids(name, target_model_id, conn=conn)
    scan = scan_source(
        name,
        source_model_id,
        have=have,
        core=core_chunk_pairs(name, conn=conn),
        conn=conn,
    )
    item.chunks = scan.rows
    item.skipped = dict(scan.skipped)
    item.usable_documents = len(scan.usable_documents)
    if have is not None:
        item.missing_chunks = scan.pending
    return item


def scope_report(
    *,
    collections: Optional[Sequence[str]] = None,
    target_model_id: Optional[str] = None,
) -> List[CollectionScope]:
    """Count every collection in scope, keeping a failed read on its collection."""
    target = resolve_target(target_model_id) if target_model_id is not None else None
    conn = connection()
    names = resolve_collection_names(collections=collections, conn=conn)
    report: List[CollectionScope] = []
    for name in names:
        try:
            report.append(scope_collection(name, conn=conn, target=target))
        except MissingCollectionError:
            # Listed, then gone before it was read: nothing to report on.
            report.append(CollectionScope(name=name, skip_reason=SKIP_NO_METADATA))
        except Exception as exc:  # noqa: BLE001 - kept on the collection, exit 1
            report.append(
                CollectionScope(name=name, error=f"{type(exc).__name__}: {exc}")
            )
    return report


def describe(item: CollectionScope) -> str:
    parts = [f"{item.usable_documents} usable documents", f"{item.chunks} chunks"]
    if item.source_model_id:
        parts.insert(0, f"source model {item.source_model_id}")
    if item.missing_chunks is not None:
        parts.append(f"{item.missing_chunks} missing for the target model")
    if item.skipped:
        parts.append(
            "skipped " + ", ".join(f"{k}={v}" for k, v in sorted(item.skipped.items()))
        )
    return f"{item.name}: " + "; ".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Report the scope of an embedding migration.\n\n"
            "Read-only: lists the collections in scope, the model each is bound\n"
            "to, the chunks searchable under it, and with --model how many of\n"
            "them the target model still lacks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Target embedding model id; adds the missing-chunk count",
    )
    parser.add_argument(
        "--collection",
        action="append",
        metavar="NAME",
        help="Restrict to this collection; repeatable",
    )
    args = parser.parse_args(argv)

    try:
        report = scope_report(collections=args.collection, target_model_id=args.model)
    except MigrationCliError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        logger.error("Scope report failed: %s", exc, exc_info=True)
        return 2

    if not report:
        logger.warning("No collections in scope")
        return 1
    for item in report:
        if item.error:
            logger.error("%s: failed: %s", item.name, item.error)
        elif item.usable:
            logger.info("%s", describe(item))
        else:
            logger.info("%s: skipped (%s)", item.name, item.skip_reason)
    return 1 if any(item.error for item in report) else 0


if __name__ == "__main__":
    sys.exit(main())

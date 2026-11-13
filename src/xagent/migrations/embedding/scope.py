"""Read-only scope of an embedding migration.

The unit is what is searchable right now: the rows of the collection's bound
model's embeddings table. Each row carries its own chunk text, parse hash and
owner, so a migration copies exactly those chunks and never re-parses anything.

Runs on its own to report coverage, and both commands build on it:

    python -m xagent.migrations.embedding.scope [--model TARGET] \
        [--collection NAME ... | --team ID]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
    embeddings_table_candidates,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import list_table_names
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string

logger = logging.getLogger(__name__)

METADATA_TABLE = "collection_metadata"

SKIP_OTHER_ENGINE = "other-engine"
SKIP_UNREADABLE_BINDING = "unreadable-binding"
SKIP_NO_SOURCE_MODEL = "no-source-model"
SKIP_NO_OWNER = "no-owner"
SKIP_NO_DOC_ID = "no-doc-id"
SKIP_NO_TEXT = "no-text"
SKIP_LEGACY_SHADOWED = "legacy-table-shadowed"
SKIP_NO_METADATA = "no-metadata"
SKIP_SOURCE_UNPINNED = "source-binding-unpinned"
SOURCE_BINDING_PINNED = "source-binding-pinned"

# A table without one of these turns rows into skips that acceptance then
# "confirms" on both sides; that is a broken table, not an empty collection.
REQUIRED_COLUMNS = ("doc_id", "chunk_id", "user_id", "text")

READ_BATCH_SIZE = 1000
CHUNK_COLUMNS = [
    "doc_id",
    "chunk_id",
    "parse_hash",
    "text",
    "chunk_hash",
    "metadata",
    "user_id",
]


class MigrationCliError(RuntimeError):
    """A scope or configuration problem that must stop the command."""


class MissingCollectionError(MigrationCliError):
    """The collection has no metadata row (never existed, or was just deleted)."""


@dataclass(frozen=True)
class SearchableChunk:
    """One row of the source model's embeddings table."""

    doc_id: str
    chunk_id: str
    parse_hash: str
    text: str
    chunk_hash: str
    metadata: Optional[str]
    user_id: Optional[int]


@dataclass
class CollectionScope:
    """What a migration would do with one collection, without doing it."""

    name: str
    engine: str = ""
    source_model_id: Optional[str] = None
    skip_reason: Optional[str] = None
    documents: int = 0
    chunks: int = 0
    missing_chunks: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.skip_reason is None and self.error is None


def connection() -> Any:
    """The metadata store's LanceDB connection, opened without creating tables."""
    return get_metadata_store().get_raw_connection()


def table_names(conn: Any) -> Set[str]:
    return set(list_table_names(conn))


def embeddings_table(model_id: str, conn: Optional[Any] = None) -> Optional[str]:
    """The existing embeddings table for ``model_id``, if any."""
    names = table_names(conn if conn is not None else connection())
    for candidate in embeddings_table_candidates(model_id):
        if candidate in names:
            return candidate
    return None


def _searchable_tables(model_id: str) -> List[str]:
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        searchable_embeddings_table_names,
    )

    return searchable_embeddings_table_names(model_id)


def source_embeddings_table(model_id: str, conn: Optional[Any] = None) -> Optional[str]:
    """The table search actually reads for ``model_id``: the first that exists
    in ``open_embeddings_table``'s own list."""
    names = table_names(conn if conn is not None else connection())
    for candidate in _searchable_tables(model_id):
        if candidate in names:
            return candidate
    return None


def writable_embeddings_table(model_id: str, conn: Optional[Any] = None) -> str:
    """The table the writer will use, refusing to shadow a populated one.

    The writer always creates the first name search tries, so while it is
    missing, any other table search can open for the model that holds rows
    would be hidden from every collection on this model.
    """
    conn = conn if conn is not None else connection()
    tables = _searchable_tables(model_id)
    canonical = tables[0]
    names = table_names(conn)
    if canonical in names:
        return canonical
    for other in tables[1:]:
        if other in names and _row_count(conn, other) > 0:
            raise MigrationCliError(
                f"embeddings table '{other}' holds rows and the canonical "
                f"'{canonical}' does not exist; writing the canonical table "
                f"would hide those rows from search. Migrate {other} to "
                f"{canonical} first."
            )
    return canonical


def _row_count(conn: Any, table: str) -> int:
    opened = conn.open_table(table)
    try:
        return int(opened.count_rows())
    finally:
        _safe_close_table(opened)


def iter_rows(
    conn: Any,
    table: str,
    where: str,
    columns: Sequence[str],
    *,
    batch_size: int = READ_BATCH_SIZE,
) -> Iterator[Dict[str, Any]]:
    """Stream rows of ``table``, raising instead of reporting a failed read as empty."""
    opened = conn.open_table(table)
    try:
        present = [name for name in columns if name in opened.schema.names]
        reader = (
            opened.search()
            .where(where)
            .select(present)
            .limit(None)
            .to_batches(batch_size)
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


def source_model_id(collection: str, info: Any) -> Optional[str]:
    """The model search uses for this collection, resolved the same way.

    Not just the bound id: a legacy collection may only carry
    ``ingestion_config.embedding_model_id``, or only have tables to infer from.
    The read-only resolver: a report must not mark the collection accessed or
    persist an inferred binding.
    """
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        _normalize_model_id,
        resolve_effective_embedding_model_read_only,
    )

    try:
        resolved = resolve_effective_embedding_model_read_only(
            collection,
            config_model_id=getattr(info, "embedding_model_id", None),
        )
    except ValueError as exc:
        logger.warning("%s: no effective embedding model: %s", collection, exc)
        return None
    return _normalize_model_id(resolved) or None


def collection_info(collection: str) -> Any:
    """The collection's metadata row; a missing collection is a hard error."""
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        get_collection_sync,
    )

    try:
        return get_collection_sync(collection)
    except ValueError as exc:
        raise MissingCollectionError(
            f"collection '{collection}' does not exist"
        ) from exc


def collection_engine(info: Any) -> str:
    """The engine a collection is bound to; an unreadable binding is reported."""
    from xagent.core.tools.core.RAG_tools.kb.coordinator import (
        resolve_collection_backend,
    )

    try:
        return str(resolve_collection_backend(info).value)
    except Exception as exc:  # noqa: BLE001 - reported, the run continues
        logger.warning("Unreadable engine binding: %s", exc)
        return ""


def iter_source_doc_ids(
    collection: str, model_id: str, *, conn: Optional[Any] = None
) -> Tuple[List[str], int]:
    """The collection's document ids and the count of rows without one.

    Chunk rows are not ordered by document, so a document is only complete
    when read on its own; this lists what to read.
    """
    conn = conn if conn is not None else connection()
    table = source_embeddings_table(model_id, conn)
    if table is None:
        return [], 0
    require_columns(conn, table, collection)
    ids: Set[str] = set()
    unaddressable = 0
    for row in iter_rows(
        conn,
        table,
        collection_filter(collection),
        ["doc_id", "chunk_id"],
    ):
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id or not str(row.get("chunk_id") or "").strip():
            unaddressable += 1
            continue
        ids.add(doc_id)
    return sorted(ids), unaddressable


def iter_source_chunks(
    collection: str,
    model_id: str,
    *,
    conn: Optional[Any] = None,
    doc_id: Optional[str] = None,
) -> Iterator[SearchableChunk]:
    """Every searchable chunk of ``collection`` under ``model_id``.

    A missing table is an empty collection: the rows are what search can reach,
    so there is nothing to copy. A failed read raises.
    """
    conn = conn if conn is not None else connection()
    table = source_embeddings_table(model_id, conn)
    if table is None:
        return
    require_columns(conn, table, collection)
    where = collection_filter(collection)
    if doc_id is not None:
        where += f" AND doc_id = '{escape_lancedb_string(doc_id)}'"
    for row in iter_rows(conn, table, where, CHUNK_COLUMNS):
        doc_id = str(row.get("doc_id") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not doc_id or not chunk_id:
            # Not addressable by search either; counted by the caller's skip
            # reasons through the empty fields below.
            yield SearchableChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                parse_hash=str(row.get("parse_hash") or ""),
                text="",
                chunk_hash="",
                metadata=None,
                user_id=None,
            )
            continue
        yield SearchableChunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            parse_hash=str(row.get("parse_hash") or ""),
            text=str(row.get("text") or ""),
            chunk_hash=str(row.get("chunk_hash") or ""),
            metadata=row.get("metadata"),
            user_id=_owner(row.get("user_id")),
        )


def require_columns(conn: Any, table: str, collection: str) -> None:
    """Raise when ``table`` lacks a column the migration's accounting needs."""
    present = set(schema_names(conn, table))
    missing = [name for name in REQUIRED_COLUMNS if name not in present]
    if missing:
        # Without them every row would look textless or ownerless and the
        # switch would install an empty target over the collection.
        raise MigrationCliError(
            f"embeddings table '{table}' has no {', '.join(missing)} column; "
            f"cannot migrate collection '{collection}'"
        )


def skip_reason(chunk: SearchableChunk) -> Optional[str]:
    """Why this source row cannot be migrated, or ``None`` when it can."""
    if not chunk.doc_id or not chunk.chunk_id:
        return SKIP_NO_DOC_ID
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
) -> Set[str]:
    """The chunk_ids the target model already holds *with a usable vector*.

    A missing table means no rows yet (nothing was ever embedded for it), not a
    read failure: the migration creates the table on its first write. A row
    whose vector is null, non-finite, all-zero or the wrong dimension is not
    held: pending must include it so a rerun repairs it.
    """
    from xagent.migrations.embedding.embed import _validate_vector

    # The table search selects: a legacy-named one is what the run must compare
    # against, or a held row would be copied into a table nobody reads.
    conn = conn if conn is not None else connection()
    table = source_embeddings_table(model_id, conn)
    if table is None:
        return set()
    held: Set[str] = set()
    for row in iter_rows(
        conn, table, collection_filter(collection), ["chunk_id", "vector"]
    ):
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            continue
        try:
            _validate_vector(row.get("vector"), dimension)
            dimension = len(row["vector"])
        except ValueError:
            continue
        held.add(chunk_id)
    return held


def resolve_collection_names(
    *, collections: Optional[Sequence[str]] = None, team: Optional[int] = None
) -> List[str]:
    """The collections this run covers, in the order they are processed."""
    if collections and team is not None:
        raise MigrationCliError("pass --collection or --team, not both")
    if collections:
        names: List[str] = []
        for raw in collections:
            name = raw.strip()
            if not name:
                raise MigrationCliError("--collection must not be empty")
            if name not in names:
                names.append(name)
        return names
    if team is not None:
        return _team_collection_names(team)
    return _all_collection_names()


def _all_collection_names() -> List[str]:
    """Every collection, read from the metadata table without creating it.

    ``list_collections`` aggregates statistics and creates missing tables on an
    empty or wrong data dir; the migration only needs the names.
    """
    conn = connection()
    if METADATA_TABLE not in table_names(conn):
        return []
    return sorted(
        {
            str(row.get("name") or "")
            for row in iter_rows(conn, METADATA_TABLE, "true", ["name"])
            if row.get("name")
        }
    )


def _team_collection_names(team_id: int) -> List[str]:
    # Teams live outside this package: the application installs the hook that
    # maps a team to its knowledge bases, and the command cannot invent one.
    from xagent.web.models.database import configure_db, get_session_local
    from xagent.web.services.knowledge_base_team_scope import (
        team_knowledge_base_hook_installed,
        team_knowledge_bases,
    )

    if not team_knowledge_base_hook_installed():
        raise MigrationCliError(
            "--team needs an application-installed team visibility hook; "
            "pass --collection NAME instead"
        )
    configure_db(read_only=True)
    session = get_session_local()()
    try:
        access = team_knowledge_bases(session, team_id=int(team_id))
    finally:
        session.close()
    return sorted({item.name for item in access if item.name})


def resolve_target(model_id: str) -> Tuple[str, Optional[int]]:
    """The target as embed will use it: canonical id and configured dimension.

    An alias must not reach the held-row scan as a table-name candidate, and a
    row can only be counted as held if it has the configured dimension.
    """
    from xagent.core.tools.core.RAG_tools.management.embedding_switch import (
        resolve_configured_model,
    )

    canonical, dimension = resolve_configured_model(model_id)
    return canonical, dimension


def scope_collection(
    name: str, target_model_id: Optional[str] = None, *, conn: Optional[Any] = None
) -> CollectionScope:
    """Count what a migration would copy for one collection."""
    target_dimension: Optional[int] = None
    if target_model_id:
        target_model_id, target_dimension = resolve_target(target_model_id)
    info = collection_info(name)
    item = CollectionScope(name=name, engine=collection_engine(info))
    if not item.engine:
        item.skip_reason = SKIP_UNREADABLE_BINDING
        return item
    if item.engine != "lancedb":
        item.skip_reason = SKIP_OTHER_ENGINE
        return item
    item.source_model_id = source_model_id(name, info)
    if item.source_model_id is None:
        item.skip_reason = SKIP_NO_SOURCE_MODEL
        return item

    conn = conn if conn is not None else connection()
    have = (
        target_chunk_ids(name, target_model_id, conn=conn, dimension=target_dimension)
        if target_model_id
        else set()
    )
    documents: Set[str] = set()
    for chunk in iter_source_chunks(name, item.source_model_id, conn=conn):
        reason = skip_reason(chunk)
        item.chunks += 1
        if reason:
            item.skipped[reason] = item.skipped.get(reason, 0) + 1
            continue
        documents.add(chunk.doc_id)
        if target_model_id and chunk.chunk_id not in have:
            item.missing_chunks += 1
    item.documents = len(documents)
    return item


def scope_report(
    *,
    collections: Optional[Sequence[str]] = None,
    team: Optional[int] = None,
    target_model_id: Optional[str] = None,
) -> List[CollectionScope]:
    """Count what a migration would copy, per collection, changing nothing."""
    names = resolve_collection_names(collections=collections, team=team)
    explicit = bool(collections)
    conn = connection()
    report: List[CollectionScope] = []
    for name in names:
        try:
            item = scope_collection(name, target_model_id, conn=conn)
        except MissingCollectionError as exc:
            if explicit:
                raise
            # Discovered, then gone before it was read: nothing to report on.
            logger.warning("%s: %s", name, exc)
            report.append(CollectionScope(name=name, skip_reason=SKIP_NO_METADATA))
            continue
        except MigrationCliError as exc:
            # A collection that exists but cannot be read (a schema error) is a
            # failure of the report, kept per collection so the rest still print.
            logger.error("%s: %s", name, exc)
            report.append(CollectionScope(name=name, error=str(exc)))
            continue
        if item.skip_reason:
            logger.warning("%s: skipped (%s)", name, item.skip_reason)
        report.append(item)
    return report


def add_scope_args(parser: argparse.ArgumentParser) -> None:
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--collection",
        action="append",
        metavar="NAME",
        help="Restrict to this collection; repeatable",
    )
    scope.add_argument(
        "--team",
        type=int,
        metavar="ID",
        help="Restrict to the knowledge bases of this team",
    )


def describe(item: CollectionScope) -> str:
    parts = [f"{item.documents} documents", f"{item.chunks} chunks"]
    if item.source_model_id:
        parts.insert(0, f"source model {item.source_model_id}")
    if item.missing_chunks:
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
            "Report what an embedding migration would copy.\n\n"
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
    add_scope_args(parser)
    args = parser.parse_args(argv)

    try:
        report = scope_report(
            collections=args.collection,
            team=args.team,
            target_model_id=args.model,
        )
    except MigrationCliError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        logger.error("Scope report failed: %s", exc, exc_info=True)
        return 2

    if not report:
        logger.warning("No collections in scope; check --collection/--team")
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

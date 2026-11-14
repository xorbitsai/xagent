"""Read-only scope of an embedding migration.

The unit is what is searchable right now: the rows of the collection's bound
model's embeddings table. Each row carries its own chunk text, parse hash and
owner, so a migration copies exactly those chunks and never re-parses anything.

A collection is eligible only when its shape is one the migration can copy
faithfully; anything else is reported with a reason and left untouched. Runs on
its own to report coverage, and both commands build on it:

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
    written_embeddings_table_name,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from xagent.core.tools.core.RAG_tools.storage.factory import get_metadata_store
from xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils import list_table_names
from xagent.core.tools.core.RAG_tools.utils.string_utils import escape_lancedb_string

logger = logging.getLogger(__name__)

METADATA_TABLE = "collection_metadata"

# Why a collection is not migrated. Each is reported with its count and the
# collection is left untouched.
SKIP_OTHER_ENGINE = "other-engine"
SKIP_UNREADABLE_BINDING = "unreadable-binding"
SKIP_UNBOUND = "unbound"
SKIP_BOUND_WITHOUT_DIMENSION = "bound-without-dimension"
SKIP_ROWS_IN_OTHER_TABLE = "rows-in-another-table"
SKIP_ROWS_IN_MULTIPLE_TABLES = "rows-in-multiple-tables"
SKIP_MISSING_COLUMN = "missing-required-column"
SKIP_NO_METADATA = "no-metadata"
# Why a source row is not copied.
SKIP_NO_DOC_ID = "no-doc-id"
SKIP_NO_OWNER = "no-owner"
SKIP_NO_TEXT = "no-text"

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


@dataclass(frozen=True)
class Eligibility:
    """Whether one collection can be migrated, and on what.

    ``reason`` is ``None`` only for an eligible collection; then ``model_id``,
    ``dimension`` and ``table`` say what the copy reads and what it writes.
    """

    model_id: Optional[str] = None
    dimension: Optional[int] = None
    table: Optional[str] = None
    engine: str = ""
    reason: Optional[str] = None


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


def source_table(model_id: str) -> str:
    """The one table the migration reads and writes for ``model_id``.

    The name the write path itself creates (``to_model_tag`` applied twice), so
    a row written here is the row search reads back.
    """
    return written_embeddings_table_name(model_id)


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


def collection_info(collection: str, conn: Optional[Any] = None) -> Any:
    """The collection's metadata row, read strictly.

    ``get_collection`` reports every store or decoding failure as "not found";
    here only an absent row is :class:`MissingCollectionError`. A failed read
    raises as itself, and a row that does not decode is a ``MigrationCliError``.
    """
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
    except Exception as exc:  # noqa: BLE001 - reported on the collection
        raise MigrationCliError(
            f"collection '{collection}' metadata does not decode: {exc}"
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


def _legacy_spelling(model_id: str) -> Optional[str]:
    """The provider-name table search falls back to, when the model resolves.

    Mirrors ``open_embeddings_table``'s last fallback: the table is named after
    the model's own ``model_name``, which only the hub knows.
    """
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import to_model_tag
    from xagent.core.tools.core.RAG_tools.utils.model_resolver import (
        resolve_embedding_adapter,
    )
    from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context

    try:
        with user_scope_context(None, True):
            config, _ = resolve_embedding_adapter(model_id)
    except Exception:  # noqa: BLE001 - an unconfigured model has no legacy table
        return None
    name = getattr(config, "model_name", None)
    return f"embeddings_{to_model_tag(name)}" if name else None


def source_spellings(model_id: str) -> List[str]:
    """Every table name search may serve this binding from, writer's first.

    The writer's own spelling (``to_model_tag`` applied twice, what every write
    creates), its once-applied form, and the provider-name table search falls
    back to when the id and the model's name differ. Another model's table is
    never here, so a precomputed target does not disqualify its source.
    """
    from xagent.core.tools.core.RAG_tools.LanceDB.model_tag_utils import (
        embeddings_table_candidates,
        written_embeddings_table_name,
    )

    names = [written_embeddings_table_name(model_id)]
    names += embeddings_table_candidates(model_id)
    legacy = _legacy_spelling(model_id)
    if legacy:
        names.append(legacy)
    return list(dict.fromkeys(names))


def _holds_rows(conn: Any, table: str, collection: str) -> bool:
    opened = conn.open_table(table)
    try:
        if "collection" not in opened.schema.names:
            # Cannot attribute a row to a collection: any row counts.
            return int(opened.count_rows()) > 0
        return int(opened.count_rows(collection_filter(collection))) > 0
    finally:
        _safe_close_table(opened)


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


def source_eligibility(collection: str, *, conn: Optional[Any] = None) -> Eligibility:
    """Whether ``collection`` can be migrated, and the one table it lives in.

    The only eligible shape is the one the write path produces: an explicit
    binding with a dimension, and the collection's rows in that binding's
    writer-spelled table and in none of the other spellings search may read. Anything else is refused with a
    reason: copying it would either lose rows or write a table search does not
    read. Read failures raise (see :func:`collection_info`) instead of being
    reported as skips.
    """
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        _normalize_model_id,
    )

    conn = conn if conn is not None else connection()
    info = collection_info(collection, conn)
    engine = collection_engine(info)
    if engine != "lancedb":
        return Eligibility(
            engine=engine,
            reason=SKIP_OTHER_ENGINE if engine else SKIP_UNREADABLE_BINDING,
        )
    model = _normalize_model_id(info.embedding_model_id)
    if not model:
        return Eligibility(engine=engine, reason=SKIP_UNBOUND)
    dimension = info.embedding_dimension
    if not isinstance(dimension, int) or dimension <= 0:
        return Eligibility(
            model_id=model, engine=engine, reason=SKIP_BOUND_WITHOUT_DIMENSION
        )
    table = source_table(model)
    names = table_names(conn)
    holders = [
        name
        for name in source_spellings(model)
        if name in names and _holds_rows(conn, name, collection)
    ]
    eligible = Eligibility(
        model_id=model, dimension=dimension, table=table, engine=engine
    )
    if table not in holders:
        # No rows yet, or rows search reaches only through another table.
        if holders:
            return Eligibility(engine=engine, reason=SKIP_ROWS_IN_OTHER_TABLE)
        return eligible
    if len(holders) > 1:
        return Eligibility(engine=engine, reason=SKIP_ROWS_IN_MULTIPLE_TABLES)
    try:
        require_columns(conn, table, collection)
    except MigrationCliError as exc:
        logger.warning("%s", exc)
        return Eligibility(engine=engine, reason=SKIP_MISSING_COLUMN)
    return eligible


def iter_source_doc_ids(
    collection: str, model_id: str, *, conn: Optional[Any] = None
) -> Tuple[List[str], int]:
    """The collection's document ids and the count of rows without one.

    Chunk rows are not ordered by document, so a document is only complete
    when read on its own; this lists what to read.
    """
    conn = conn if conn is not None else connection()
    table = source_table(model_id)
    if table not in table_names(conn):
        return [], 0
    ids: Set[str] = set()
    unaddressable = 0
    for row in iter_rows(
        conn,
        table,
        collection_filter(collection),
        ["doc_id", "chunk_id"],
    ):
        # A chunkless row is counted when its document is read; counting it
        # here too would count it twice.
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
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
    table = source_table(model_id)
    if table not in table_names(conn):
        return
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

    conn = conn if conn is not None else connection()
    table = source_table(model_id)
    if table not in table_names(conn):
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


def resolve_target(model_id: str) -> Tuple[str, int]:
    """The target as embed will use it: canonical id and configured dimension.

    Binding a model the hub does not know would silently switch the collection
    to whatever the environment default is.
    """
    from xagent.core.tools.core.RAG_tools.management.embedding_switch import (
        resolve_configured_model,
    )

    return resolve_configured_model(model_id)


def scope_collection(
    name: str,
    target_model_id: Optional[str] = None,
    *,
    conn: Optional[Any] = None,
    target: Optional[Tuple[str, int]] = None,
) -> CollectionScope:
    """Count what a migration would copy for one collection."""
    if target is None and target_model_id:
        target = resolve_target(target_model_id)
    target_model_id, target_dimension = target if target else (None, None)
    conn = conn if conn is not None else connection()
    eligibility = source_eligibility(name, conn=conn)
    item = CollectionScope(name=name, engine=eligibility.engine)
    if eligibility.reason:
        item.skip_reason = eligibility.reason
        return item
    item.source_model_id = eligibility.model_id
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
    # Once, before the loop: an unresolvable target stops the run (exit 2).
    target = resolve_target(target_model_id) if target_model_id else None
    conn = connection()
    report: List[CollectionScope] = []
    for name in names:
        try:
            item = scope_collection(name, conn=conn, target=target)
        except MissingCollectionError as exc:
            if explicit:
                raise
            # Discovered, then gone before it was read: nothing to report on.
            logger.warning("%s: %s", name, exc)
            report.append(CollectionScope(name=name, skip_reason=SKIP_NO_METADATA))
            continue
        except Exception as exc:  # noqa: BLE001 - kept on the collection, exit 1
            # A collection that exists but cannot be read is a failure of the
            # report, kept per collection so the rest are still checked.
            logger.error("%s: %s", name, exc)
            report.append(
                CollectionScope(name=name, error=f"{type(exc).__name__}: {exc}")
            )
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

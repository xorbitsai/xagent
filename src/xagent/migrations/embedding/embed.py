"""Copy a collection's searchable chunks into another embedding model.

Online and non-destructive: every collection keeps its bound model, so search
and ingestion behave exactly as before. The unit is what the source model's
embeddings table holds right now, so a rerun copies only what the target still
lacks and the switch phase can re-run it to pick up writes since.

    python -m xagent.migrations.embedding.embed --model <model-id> --dry-run
    python -m xagent.migrations.embedding.embed --model <model-id> --rate-limit 20
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    ChunkEmbeddingData,
    IngestionConfig,
)
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    write_vectors_to_db,
)
from xagent.migrations.embedding.scope import (
    SKIP_NO_DOC_ID,
    MigrationCliError,
    SearchableChunk,
    add_scope_args,
    connection,
    iter_source_chunks,
    iter_source_doc_ids,
    resolve_collection_names,
    skip_reason,
    source_eligibility,
    source_table,
    table_names,
    target_chunk_ids,
)

logger = logging.getLogger(__name__)

# A broken provider or credential fails every document alike; stop instead of
# walking the rest of the deployment.
MAX_CONSECUTIVE_DOCUMENT_FAILURES = 5


class RateLimiter:
    """Paces encode calls at ``rate`` texts per second on average."""

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
        wait = start - now
        if wait > 0:
            time.sleep(wait)


class PacedEmbedding(BaseEmbedding):
    """Spends rate-limit budget for every encode call the run makes."""

    def __init__(self, inner: BaseEmbedding, limiter: RateLimiter) -> None:
        self._inner = inner
        self._limiter = limiter

    def encode(
        self,
        text: Union[str, List[str]],
        dimension: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        self._limiter.acquire(1 if isinstance(text, str) else len(text))
        return self._inner.encode(text, dimension=dimension, instruct=instruct)

    def get_dimension(self) -> Optional[int]:
        return self._inner.get_dimension()

    @property
    def abilities(self) -> List[str]:
        return self._inner.abilities


@dataclass
class DocumentFailure:
    doc_id: str
    error: str


@dataclass
class CollectionReport:
    name: str
    engine: str = ""
    source_model_id: Optional[str] = None
    collection_skip: Optional[str] = None
    documents_total: int = 0
    documents_embedded: int = 0
    documents_unchanged: int = 0
    chunks_pending: int = 0
    chunks_embedded: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    failures: List[DocumentFailure] = field(default_factory=list)
    aborted: bool = False


@dataclass
class EmbedReport:
    model_id: str
    dry_run: bool
    collections: List[CollectionReport] = field(default_factory=list)
    aborted: bool = False

    @property
    def failed(self) -> bool:
        return self.aborted or any(item.failures for item in self.collections)


def embed_collections(
    model_id: str,
    *,
    collections: Optional[Sequence[str]] = None,
    team: Optional[int] = None,
    dry_run: bool = False,
    rate_limit: float = 0.0,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    use_async: bool = False,
) -> EmbedReport:
    """Copy every collection's searchable chunks into ``model_id``."""
    names = resolve_collection_names(collections=collections, team=team)
    report = EmbedReport(model_id=model_id, dry_run=dry_run)
    if not names:
        # Silence here is indistinguishable from success, and the usual cause
        # is a LANCEDB_DIR that does not match the backend's.
        logger.warning("No collections in scope; check --collection/--team")
        return report

    embedding_config, adapter = resolve_target_model(model_id)
    cfg = IngestionConfig(
        embedding_model_id=embedding_config.id,
        embedding_batch_size=batch_size,
        embedding_use_async=use_async,
    )
    paced = PacedEmbedding(adapter, RateLimiter(rate_limit))
    if rate_limit <= 0:
        logger.warning(
            "No --rate-limit set; the run will use the provider's whole quota"
        )
    logger.info(
        "Target model: %s (dimension %s)",
        embedding_config.id,
        embedding_config.dimension,
    )
    logger.info("Collections in scope (%d): %s", len(names), names)
    conn = connection()

    for name in names:
        logger.info("Collection %s: starting", name)
        try:
            item = embed_collection(
                name,
                model_id=embedding_config.id,
                embedding_config=embedding_config,
                embedding_adapter=paced,
                cfg=cfg,
                dry_run=dry_run,
                conn=conn,
            )
        except Exception as exc:  # noqa: BLE001 - one collection must not stop the run
            item = CollectionReport(name=name)
            item.failures.append(DocumentFailure("*", f"{type(exc).__name__}: {exc}"))
            logger.error("Collection %s: failed: %s", name, exc, exc_info=True)
        report.collections.append(item)
        logger.info("%s", _describe(item, dry_run=dry_run))
        if item.aborted:
            report.aborted = True
            break
    return report


def resolve_target_model(model_id: str) -> Tuple[EmbeddingModelConfig, BaseEmbedding]:
    """The configured model, never the environment's fallback."""
    from xagent.core.tools.core.RAG_tools.management.embedding_switch import (
        resolve_configured_model,
    )

    try:
        canonical, _dimension = resolve_configured_model(model_id)
    except ValueError as exc:
        raise MigrationCliError(str(exc)) from exc
    from xagent.core.tools.core.RAG_tools.utils.model_resolver import (
        resolve_embedding_adapter,
    )
    from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context

    # The hub resolves a bare name only among the models the scope may use;
    # the command acts for the deployment, so it resolves as an admin.
    with user_scope_context(None, True):
        config, adapter = resolve_embedding_adapter(canonical)
    if config.id != canonical:
        raise MigrationCliError(
            f"'{canonical}' resolves to '{config.id}'; pass the configured id"
        )
    return config, adapter


def embed_collection(
    name: str,
    *,
    model_id: str,
    embedding_config: EmbeddingModelConfig,
    embedding_adapter: BaseEmbedding,
    cfg: IngestionConfig,
    dry_run: bool,
    conn: Optional[Any] = None,
) -> CollectionReport:
    """Copy one collection's searchable chunks the target model lacks."""
    conn = conn if conn is not None else connection()
    report = CollectionReport(name=name)
    eligibility = source_eligibility(name, conn=conn)
    report.engine = eligibility.engine
    if eligibility.reason:
        report.collection_skip = eligibility.reason
        logger.warning("%s: skipped (%s)", name, eligibility.reason)
        return report
    report.source_model_id = eligibility.model_id

    have = target_chunk_ids(
        name, model_id, conn=conn, dimension=embedding_config.dimension
    )
    doc_ids, unaddressable = iter_source_doc_ids(
        name, report.source_model_id, conn=conn
    )
    if unaddressable:
        report.skipped[SKIP_NO_DOC_ID] = (
            report.skipped.get(SKIP_NO_DOC_ID, 0) + unaddressable
        )
    consecutive_failures = 0
    # (doc_id, chunk_id) only: the text is not kept after its write.
    written: List[Tuple[str, str]] = []
    # One document at a time: its rows are not contiguous in the table, so a
    # window of documents would split its counts and failures across windows.
    for doc_id in doc_ids:
        items = [
            chunk
            for chunk in iter_source_chunks(
                name, report.source_model_id, conn=conn, doc_id=doc_id
            )
            if _count_skip(chunk, report.skipped)
        ]
        report.documents_total += 1
        if not items:
            continue
        missing = [chunk for chunk in items if chunk.chunk_id not in have]
        report.chunks_pending += len(missing)
        if dry_run:
            if missing:
                report.documents_embedded += 1
            else:
                report.documents_unchanged += 1
            continue
        if not missing:
            report.documents_unchanged += 1
            continue
        try:
            _write_chunks(
                name,
                missing,
                model_id=model_id,
                embedding_config=embedding_config,
                embedding_adapter=embedding_adapter,
                cfg=cfg,
            )
        except Exception as exc:  # noqa: BLE001 - continue with the next document
            consecutive_failures += 1
            report.failures.append(
                DocumentFailure(doc_id, f"{type(exc).__name__}: {exc}")
            )
            logger.error(
                "%s/%s: embedding failed: %s", name, doc_id, exc, exc_info=True
            )
            if consecutive_failures >= MAX_CONSECUTIVE_DOCUMENT_FAILURES:
                report.aborted = True
                logger.error(
                    "%s: aborting after %d consecutive document failures",
                    name,
                    consecutive_failures,
                )
                return report
            continue
        consecutive_failures = 0
        written.extend((chunk.doc_id, chunk.chunk_id) for chunk in missing)
        report.documents_embedded += 1
        report.chunks_embedded += len(missing)

    if dry_run:
        return report
    _verify_written(
        name,
        model_id,
        written,
        report,
        conn=conn,
        dimension=embedding_config.dimension,
    )
    _ensure_indexes(name, model_id, report, conn=conn)
    return report


def _count_skip(chunk: SearchableChunk, skipped: Dict[str, int]) -> bool:
    """Count this row as skipped when it must be, and say whether it is usable."""
    reason = skip_reason(chunk)
    if reason:
        skipped[reason] = skipped.get(reason, 0) + 1
        return False
    return True


def _index_store() -> Any:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    return get_vector_index_store()


def _ensure_indexes(
    name: str, model_id: str, report: CollectionReport, *, conn: Optional[Any] = None
) -> None:
    """Build the target table's indexes so search works right after the flip."""
    conn = conn if conn is not None else connection()
    if source_table(model_id) not in table_names(conn):
        # Nothing was ever written for this model: no table, no index needed.
        return
    result = _index_store().create_index(model_id)
    logger.info(
        "%s: target index status %s (fts=%s)", name, result.status, result.fts_enabled
    )
    if result.status in ("failed", "index_corrupted"):
        # Search would be degraded after the flip, or fail outright.
        report.failures.append(
            DocumentFailure(
                "*", f"target table index build ended {result.status}: {result.advice}"
            )
        )
        return
    if not result.fts_enabled:
        # A populated target needs FTS even when this invocation wrote nothing:
        # an earlier run may have written the rows and failed the index.
        report.failures.append(
            DocumentFailure("*", "target table has no full-text index after the copy")
        )


def _write_chunks(
    name: str,
    chunks: List[SearchableChunk],
    *,
    model_id: str,
    embedding_config: EmbeddingModelConfig,
    embedding_adapter: BaseEmbedding,
    cfg: IngestionConfig,
) -> None:
    """Encode and write ``chunks``, one owner at a time, in configured batches."""
    by_owner: Dict[Optional[int], List[SearchableChunk]] = {}
    for chunk in chunks:
        by_owner.setdefault(chunk.user_id, []).append(chunk)
    for owner, items in by_owner.items():
        if owner is None:
            continue
        for start in range(0, len(items), cfg.embedding_batch_size):
            batch = items[start : start + cfg.embedding_batch_size]
            texts = [chunk.text for chunk in batch]
            vectors = _encode(embedding_adapter, texts, cfg)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"provider returned {len(vectors)} vectors for {len(batch)} texts"
                )
            for vector in vectors:
                _validate_vector(vector, embedding_config.dimension)
            write_vectors_to_db(
                collection=name,
                embeddings=[
                    ChunkEmbeddingData(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        parse_hash=chunk.parse_hash,
                        model=embedding_config.id,
                        vector=vector,
                        text=chunk.text,
                        chunk_hash=chunk.chunk_hash,
                        metadata=_metadata(chunk.metadata),
                    )
                    for chunk, vector in zip(batch, vectors)
                ],
                create_index=False,
                user_id=owner,
            )


def _encode(
    adapter: BaseEmbedding, texts: List[str], cfg: IngestionConfig
) -> List[List[float]]:
    if not cfg.embedding_use_async:
        raw = adapter.encode(texts)
        if (
            isinstance(raw, list)
            and len(raw) == len(texts)
            and raw
            and isinstance(raw[0], list)
        ):
            return raw
        return [raw]
    results: List[Optional[List[float]]] = [None] * len(texts)
    lock = threading.Lock()
    errors: List[BaseException] = []
    workers = max(1, min(cfg.embedding_concurrent or 1, len(texts)))

    def _one(index: int) -> None:
        try:
            vector = adapter.encode(texts[index])
        except BaseException as exc:  # noqa: BLE001 - reported after the pool
            with lock:
                errors.append(exc)
            return
        with lock:
            results[index] = vector if isinstance(vector, list) else [vector]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_one, range(len(texts))))
    if errors or any(item is None for item in results):
        raise errors[0] if errors else RuntimeError("a chunk produced no vector")
    return [item for item in results if item is not None]


def _validate_vector(vector: Any, dimension: Optional[int]) -> None:
    import math

    if not isinstance(vector, list) or not vector:
        raise ValueError("provider returned a non-list vector")
    if dimension is not None and len(vector) != dimension:
        raise ValueError(
            f"provider returned a {len(vector)}-dimension vector for a "
            f"{dimension}-dimension model"
        )
    if not all(isinstance(value, float) or isinstance(value, int) for value in vector):
        raise ValueError("provider returned a vector with non-numeric values")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("provider returned a non-finite vector")
    if all(value == 0 for value in vector):
        raise ValueError("provider returned an all-zero vector")


def _metadata(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None or isinstance(raw, dict):
        return raw
    import json

    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _verify_written(
    name: str,
    model_id: str,
    written: List[Tuple[str, str]],
    report: CollectionReport,
    *,
    conn: Any,
    dimension: Optional[int],
) -> None:
    """A dropped chunk is a failure, not a success: check the target holds it."""
    if not written:
        return
    have = target_chunk_ids(name, model_id, conn=conn, dimension=dimension)
    shortfall: Dict[str, int] = {}
    for doc_id, chunk_id in written:
        if chunk_id not in have:
            shortfall[doc_id] = shortfall.get(doc_id, 0) + 1
    for doc_id, missing in sorted(shortfall.items()):
        report.failures.append(
            DocumentFailure(doc_id, f"{missing} chunks missing from the target model")
        )
        report.documents_embedded = max(0, report.documents_embedded - 1)
        report.chunks_embedded = max(0, report.chunks_embedded - missing)
    if shortfall:
        logger.error("%s: %d documents lost chunks while writing", name, len(shortfall))


def _describe(report: CollectionReport, *, dry_run: bool) -> str:
    if report.collection_skip:
        return f"{report.name}: skipped ({report.collection_skip})"
    verb = "would copy" if dry_run else "copied"
    parts = [
        f"{report.documents_total} documents",
        f"{verb} {report.documents_embedded}",
        f"unchanged {report.documents_unchanged}",
        f"{report.chunks_pending} pending chunks",
    ]
    if not dry_run:
        parts.append(f"{report.chunks_embedded} chunks")
    if report.skipped:
        parts.append(
            "skipped "
            + ", ".join(f"{k}={v}" for k, v in sorted(report.skipped.items()))
        )
    if report.failures:
        parts.append(f"failed {len(report.failures)}")
    return f"{report.name}: " + "; ".join(parts)


def _log_report(report: EmbedReport) -> None:
    logger.info("=" * 60)
    logger.info(
        "%s report (model %s)",
        "Dry-run" if report.dry_run else "Migration",
        report.model_id,
    )
    for item in report.collections:
        logger.info("%s", _describe(item, dry_run=report.dry_run))
        for failure in item.failures:
            logger.error("%s/%s: %s", item.name, failure.doc_id, failure.error)
    if report.aborted:
        logger.error("Run aborted before every collection was processed")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a collection's searchable chunks into another embedding model.\n\n"
            "Collections keep their bound model; retrieval is unaffected. Chunks\n"
            "the target already holds are skipped, so the command is safe to\n"
            "re-run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Target embedding model id from the model hub",
    )
    add_scope_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be copied without calling the model or writing",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "Average maximum of texts embedded per second across the run (one "
            "per chunk; a batch request counts its items); 0 = unlimited"
        ),
    )
    parser.add_argument(
        "--use-async",
        action="store_true",
        help="Encode chunks concurrently; needed by models without batch support",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        metavar="N",
        help=f"Chunks per embedding request (default {DEFAULT_EMBEDDING_BATCH_SIZE})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.batch_size <= 0:
        logger.error("--batch-size must be positive")
        return 2
    if not math.isfinite(args.rate_limit) or args.rate_limit < 0:
        logger.error("--rate-limit must be a finite, non-negative number")
        return 2

    try:
        report = embed_collections(
            args.model,
            collections=args.collection,
            team=args.team,
            dry_run=args.dry_run,
            rate_limit=args.rate_limit,
            batch_size=args.batch_size,
            use_async=args.use_async,
        )
    except MigrationCliError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        logger.error("Embedding migration failed: %s", exc, exc_info=True)
        return 2

    _log_report(report)
    if not report.collections:
        return 1
    if all(item.collection_skip for item in report.collections):
        logger.error("Nothing was migrated: every collection was skipped")
        return 1
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, Mapping, Sequence

from .schemas import IndexMetric

logger = logging.getLogger(__name__)

# ------------------------- Paths -------------------------

DATA_DIR: Final[Path] = Path("data")
"""Base data directory (relative to CWD)."""

ARTIFACTS_DIR: Final[Path] = DATA_DIR / "artifacts"
"""Base artifacts directory."""

# ------------------------- Defaults -------------------------

DEFAULT_INDEX_ROW_THRESHOLD: Final[int] = 50_000
"""Row count threshold to consider building ANN index on embeddings tables."""

DEFAULT_IVFPQ_ROW_THRESHOLD: Final[int] = 10_000_000
"""Row count threshold to recommend IVFPQ index over HNSW for large datasets."""

DEFAULT_INDEX_TYPE: Final[str] = "HNSW"
"""Default ANN index type for medium-size datasets."""

DEFAULT_HNSW_PARAMS: Final[Dict[str, Any]] = {}
"""Default HNSW index parameters. Uses LanceDB defaults if empty."""

DEFAULT_IVFPQ_PARAMS: Final[Dict[str, Any]] = {}
"""Default IVFPQ index parameters. Uses LanceDB defaults if empty."""

DEFAULT_FTS_PARAMS: Final[Dict[str, Any]] = {
    "base_tokenizer": "ngram",
    "ngram_min_length": 2,
    "prefix_only": True,
    "with_position": True,
}
"""
Default FTS index parameters tuned for East Asian text.

We use n-gram tokenization (min length 2, prefix-only) so that contiguous
Chinese/Japanese/Korean strings can still be matched
efficiently without requiring word-segmentation libraries. LanceDB falls back
to its defaults for any unspecified options (e.g. stemming, stop-word removal).
"""

DEFAULT_LANCEDB_SCAN_BATCH_SIZE: Final[int] = 2048
"""Default batch size when streaming LanceDB tables for statistics aggregation."""

DEFAULT_LANCEDB_BATCH_DELAY_MS: Final[int] = 0
"""Default delay (ms) between embedding write batches to LanceDB.

Set to 0 to disable any artificial throttling.
"""

DEFAULT_LANCEDB_BATCH_SIZE: Final[int] = 1000
"""Default batch size for embedding writes to LanceDB (env: LANCEDB_BATCH_SIZE)."""

DEFAULT_VECTOR_STORE_SCAN_LIMIT: Final[int] = 10_000
"""Default max rows scanned in vector-store document listing operations."""

DEFAULT_VECTOR_STORE_EXTENDED_SCAN_LIMIT: Final[int] = 1_000_000
"""Higher limit for operations like listing all documents in a collection or deleting a collection."""

DEFAULT_VECTOR_STORE_DELETE_BATCH_SIZE: Final[int] = 100
"""Default doc_id batch size for vector-store delete predicates."""

DEFAULT_BACKFILL_BATCH_SIZE: Final[int] = 1000
"""Default batch size for backfill operations (rows per iteration).

Used when backfilling legacy data (e.g., NULL user_id recovery from source_path).
"""

DEFAULT_BACKFILL_MAX_ITERATIONS: Final[int] = 100
"""Maximum iterations for backfill loops to prevent infinite loops.

Safety limit for backfill operations that process data in batches.
"""

# Reserved int64 lower bound for internal system sentinel values.
MIN_INT64: Final[int] = -(2**63)
"""Minimum 64-bit integer, used as internal sentinel value."""

# Stable expression that always matches no rows for unauthenticated reads.
UNAUTHENTICATED_NO_ACCESS_FILTER: Final[str] = (
    "(user_id IS NULL and user_id IS NOT NULL)"
)
"""A stable LanceDB filter expression that always matches no rows."""

ENABLE_AUTO_EMBEDDINGS_MIGRATION: Final[bool] = (
    os.getenv("ENABLE_AUTO_EMBEDDINGS_MIGRATION", "false").lower() == "true"
)
"""
Enable automatic forward migration of legacy embeddings tables.

When disabled (default), the system will not automatically migrate data from
legacy table names (embeddings_{model_name}) to new Hub ID-based names
(embeddings_{hub_id}). This prevents unexpected data movement and performance
impact during normal operations.

To enable automatic migration, set the environment variable:
    ENABLE_AUTO_EMBEDDINGS_MIGRATION=true

Automatic migration should only be enabled during controlled maintenance windows
or when explicitly executing migration tools.
"""

# Parameters that affect parse hash
PARSE_PARAM_WHITELIST: Final[Sequence[str]] = (
    "extract_tables",
    "extract_images",
    "model",
    "prompt_template_id",
    "ocr_enabled",
)

# Default tiktoken encoding for chunk token counting (OpenAI cl100k_base: GPT-4, GPT-3.5-turbo)
DEFAULT_TIKTOKEN_ENCODING: Final[str] = "cl100k_base"

# P1: Default regex patterns for protected content (not split inside these)
# Order matters: longer/more specific patterns can be listed first
DEFAULT_PROTECTED_PATTERNS: Final[Sequence[str]] = (
    r"```[\s\S]*?```",  # Markdown fenced code block
    r"\$\$[\s\S]*?\$\$",  # LaTeX display math
    r"\$[^$\n]+\$",  # LaTeX inline math (single line)
    r"!\[.*?\]\(.*?\)",  # Markdown image
    r"\[.*?\]\(.*?\)",  # Markdown link
    r"(?:^|\n)[ \]*(?:\|[^|\n]*)+\|[\r\n]+",  # Markdown table row (simplified)
)

# P2: Default context size (chars) to attach before/after table or image chunks; 0 = disabled
DEFAULT_TABLE_CONTEXT_SIZE: Final[int] = 0
DEFAULT_IMAGE_CONTEXT_SIZE: Final[int] = 0

# Parameters that affect chunk hash
CHUNK_PARAM_WHITELIST: Final[Sequence[str]] = (
    "chunk_strategy",
    "chunk_size",
    "chunk_overlap",
    "headers_to_split_on",
    "separators",
    "use_token_count",
    "tiktoken_encoding",
    "enable_protected_content",
    "protected_patterns",
    "table_context_size",
    "image_context_size",
)

# Common model synonyms to canonical names (vendor/name or simple name)
MODEL_SYNONYMS: Final[Mapping[str, str]] = {
    # QWEN
    "text-embedding-v4": "QWEN/text-embedding-v4",
    "text-embedding-v3": "QWEN/text-embedding-v3",
    "multimodal-embedding-v1": "QWEN/multimodal-embedding-v1",
    # BAAI BGE
    "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
    "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
}


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on anything else."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r; must be positive, using %s", name, raw, default)
        return default
    return value


@dataclass(frozen=True)
class IndexPolicy:
    """Index policy configuration for embeddings tables.

    Attributes:
        enable_threshold_rows: Row-count threshold to consider building ANN index.
        ivfpq_threshold_rows: Row-count threshold to recommend IVFPQ over HNSW.
        hnsw_params: HNSW index parameters (overrides defaults if provided).
        ivfpq_params: IVFPQ index parameters (overrides defaults if provided).
        metric: Distance metric to use for the vector index (e.g., L2, COSINE, DOT).
        fts_enabled: Whether to enable Full-Text Search indexing.
        fts_params: FTS index parameters.
        reindex_batch_size: Batch size threshold for triggering reindex.
        reindex_unindexed_ratio_threshold: Ratio threshold for triggering reindex.
        enable_immediate_reindex: Whether to reindex immediately after writes.
        enable_smart_reindex: Whether to use smart reindex based on unindexed ratio.
        compact_fragment_threshold: Fragment count at or above which a table is
            compacted. Catches append-heavy tables, where reads slow down with
            the file count. Must be positive.
        compact_stale_version_threshold: Number of versions already older than
            ``version_retention_days`` -- that is, versions the next compaction
            can actually delete -- at or above which a table is compacted.
            Catches update-heavy tables such as ``collection_metadata``, whose
            ``merge_insert`` upserts rewrite rows instead of appending: their
            fragment count plateaus around the row count while version history,
            and the disk it holds, grows without bound.
            Counting only reclaimable versions is what keeps the trigger from
            ratcheting: counting *all* versions would stay above any threshold
            forever once crossed, since compaction cannot remove versions still
            inside the retention window and each run commits another one.
            Must be positive.
        version_retention_days: Age below which old table versions are kept; the
            safety margin that protects readers holding an older version. Must be
            positive: zero drops every version but the latest.
    """

    enable_threshold_rows: int = DEFAULT_INDEX_ROW_THRESHOLD
    ivfpq_threshold_rows: int = DEFAULT_IVFPQ_ROW_THRESHOLD
    hnsw_params: Dict[str, Any] = None  # type: ignore
    ivfpq_params: Dict[str, Any] = None  # type: ignore
    metric: IndexMetric = IndexMetric.L2
    fts_enabled: bool = True
    fts_params: Dict[str, Any] = None  # type: ignore

    # Reindex configuration
    reindex_batch_size: int = 1000
    reindex_unindexed_ratio_threshold: float = 0.05
    enable_immediate_reindex: bool = False
    enable_smart_reindex: bool = True

    # Compaction configuration. Env overrides exist because this policy governs
    # an always-on inline path: raising a threshold is the way to stand it down
    # without a redeploy.
    compact_fragment_threshold: int = field(
        default_factory=lambda: _positive_int_env(
            "XAGENT_KB_COMPACT_FRAGMENT_THRESHOLD", 100
        )
    )
    compact_stale_version_threshold: int = field(
        default_factory=lambda: _positive_int_env(
            "XAGENT_KB_COMPACT_VERSION_THRESHOLD", 100
        )
    )
    version_retention_days: int = field(
        default_factory=lambda: _positive_int_env("XAGENT_KB_VERSION_RETENTION_DAYS", 7)
    )

    def __post_init__(self) -> None:
        """Validate the compaction thresholds, then fill in default param dicts.

        This is the only enforcement point for these bounds, so every
        construction path goes through it.

        Raises:
            ValueError: If ``version_retention_days`` is not positive -- a zero
                or negative window makes the backend delete every table version
                but the latest, invalidating readers holding an older one -- or
                if either compaction threshold is not positive, which would make
                every table qualify on every ingestion.
        """
        if self.version_retention_days <= 0:
            raise ValueError(
                "version_retention_days must be positive; a zero window deletes "
                "every version but the latest and invalidates concurrent readers"
            )
        for field_name in (
            "compact_fragment_threshold",
            "compact_stale_version_threshold",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(
                    f"{field_name} must be positive; a non-positive threshold "
                    "makes every table qualify for compaction on every write"
                )
        if self.hnsw_params is None:
            object.__setattr__(self, "hnsw_params", DEFAULT_HNSW_PARAMS.copy())
        if self.ivfpq_params is None:
            object.__setattr__(self, "ivfpq_params", DEFAULT_IVFPQ_PARAMS.copy())
        if self.fts_params is None:
            object.__setattr__(self, "fts_params", DEFAULT_FTS_PARAMS.copy())


DEFAULT_INDEX_POLICY: Final[IndexPolicy] = IndexPolicy()
"""Default index policy instance."""

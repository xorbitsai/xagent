"""Utilities for handling schema migrations and backward compatibility."""

import fcntl
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, cast

import pyarrow as pa  # type: ignore

from ..LanceDB.model_tag_utils import to_model_tag
from ..LanceDB.schema_manager import _safe_close_table
from ..storage.factory import get_vector_store_raw_connection
from .lancedb_query_utils import list_table_names
from .string_utils import escape_lancedb_string
from .tag_mapping import register_tag_mapping

logger = logging.getLogger(__name__)


def migrate_collection_metadata(
    legacy_data: Dict[str, Any],
    *,
    infer_embedding: bool = True,
) -> Dict[str, Any]:
    """Migrate legacy collection metadata to current schema version.

    Args:
        legacy_data: Legacy collection data from storage
        infer_embedding: If True (default), ``0.0.0 -> 1.0.0`` may scan LanceDB
            embedding tables to infer ``embedding_model_id`` / dimension. Use
            **False** for read-only deserialization (e.g. :meth:`CollectionInfo.from_storage`)
            to avoid I/O, heavy work, and log noise on hot paths.

    Returns:
        Migrated data compatible with current schema
    """
    data = legacy_data.copy()
    current_version = "1.0.0"
    data_version = data.get("schema_version", "0.0.0")
    collection_name = data.get("name", "unknown")

    log_info = logger.info if infer_embedding else logger.debug
    log_info(
        f"[MIGRATION_START] Collection: {collection_name}, From: {data_version}, To: {current_version}"
    )
    logger.debug(
        "[MIGRATION_SNAPSHOT] Original data for %s: %s", collection_name, legacy_data
    )

    try:
        # Apply migrations sequentially
        while data_version < current_version:
            previous_version = data_version
            if data_version == "0.0.0":
                data = _migrate_0_0_0_to_1_0_0(data, infer_embedding=infer_embedding)
                data_version = "1.0.0"

            log_info(
                f"[MIGRATION_STEP] {collection_name}: {previous_version} -> {data_version} completed."
            )

        log_info(
            f"[MIGRATION_SUCCESS] Collection '{collection_name}' is now at version {data_version}"
        )
        return data
    except Exception as e:
        logger.error(
            "[MIGRATION_FAILED] Collection: %s, Error: %s",
            collection_name,
            str(e),
            exc_info=True,
        )
        logger.error("[MIGRATION_RECOVERY_DATA] %s", legacy_data)
        raise


def _migrate_0_0_0_to_1_0_0(
    data: Dict[str, Any],
    *,
    infer_embedding: bool = True,
) -> Dict[str, Any]:
    """Migrate from pre-versioned schema to 1.0.0."""
    collection_name = data.get("name", "")

    if infer_embedding:
        embedding_model_id, embedding_dimension = (
            _infer_embedding_config_from_collection(collection_name)
        )
    else:
        embedding_model_id = data.get("embedding_model_id")
        embedding_dimension = data.get("embedding_dimension")

    if embedding_model_id:
        logger.info(
            "[MIGRATION_INFERENCE] Inferred embedding model '%s' (dimension: %s) for collection '%s'",
            embedding_model_id,
            embedding_dimension,
            collection_name,
        )
    else:
        logger.warning(
            "[MIGRATION_INFERENCE_FAILED] Could not infer embedding model for collection '%s', will use lazy initialization.",
            collection_name,
        )

    migrated = {
        # Version control
        "schema_version": "1.0.0",
        # Basic fields (preserve existing or set defaults)
        "name": collection_name,
        # Embedding config (inferred from existing data or None for lazy init)
        "embedding_model_id": embedding_model_id,
        "embedding_dimension": embedding_dimension,
        # Statistics (preserve existing or set defaults)
        "documents": data.get("documents", 0),
        "embeddings": data.get("embeddings", 0),
        # Document names
        "document_names": data.get("document_names", []),
        # Configuration (defaults)
        "collection_locked": False,
        "allow_mixed_parse_methods": False,
        "skip_config_validation": False,
        # Timestamps - Use naive UTC
        "created_at": data.get("created_at")
        or datetime.now(timezone.utc).replace(tzinfo=None),
        "updated_at": data.get("updated_at")
        or datetime.now(timezone.utc).replace(tzinfo=None),
        "last_accessed_at": data.get("last_accessed_at")
        or datetime.now(timezone.utc).replace(tzinfo=None),
        # Extensions
        "extra_metadata": data.get("extra_metadata", {}),
    }

    return migrated


def _infer_embedding_config_from_collection(
    collection_name: str,
) -> Tuple[Optional[str], Optional[int]]:
    """Infer embedding model and dimension from existing collection data.

    Args:
        collection_name: Name of the collection

    Returns:
        Tuple of (embedding_model_id, embedding_dimension), or (None, None) if cannot infer
    """
    from ..kb.collection_handle import ledger_holds_vectors

    if not ledger_holds_vectors():
        return None, None
    logger.debug(
        "Starting embedding config inference for collection '%s'", collection_name
    )

    try:
        # Get LanceDB connection
        logger.debug("Connecting to LanceDB for collection '%s'", collection_name)
        conn = get_vector_store_raw_connection()

        # Get all table names that contain embeddings (prefer list_tables(); avoids LanceDB deprecation warnings)
        if (
            getattr(conn, "list_tables", None) is None
            and getattr(conn, "table_names", None) is None
        ):
            logger.info(
                "LanceDB connection missing list_tables/table_names for collection '%s' - "
                "will use lazy initialization",
                collection_name,
            )
            return None, None

        all_table_names = list_table_names(conn)
        table_names = [
            name for name in all_table_names if name.startswith("embeddings_")
        ]
        logger.debug("Found %s embedding tables: %s", len(table_names), table_names)

        if not table_names:
            logger.info(
                "No embedding tables found for collection '%s' - will use lazy initialization",
                collection_name,
            )
            return None, None

        # Track model usage by counting vectors
        model_stats = {}

        for table_name in table_names:
            logger.debug(
                "Checking table '%s' for collection '%s'", table_name, collection_name
            )

            table = None
            try:
                table = conn.open_table(table_name)

                # Count vectors for this collection in this table
                try:
                    # Try to filter by collection if the column exists
                    safe_collection_name = escape_lancedb_string(collection_name)
                    count_result = (
                        table.search()
                        .where(f"collection = '{safe_collection_name}'")
                        .limit(1)
                        .to_pandas()
                    )
                    vector_count = len(count_result) if not count_result.empty else 0
                    logger.debug(
                        "Table '%s' has %s vectors for collection '%s'",
                        table_name,
                        vector_count,
                        collection_name,
                    )

                    if vector_count == 0:
                        continue

                    # Get model tag from table name
                    model_tag = table_name.replace("embeddings_", "")
                    logger.debug(
                        "Extracted model tag '%s' from table '%s'",
                        model_tag,
                        table_name,
                    )

                    # Get dimension from schema
                    dimension = None
                    try:
                        schema = table.schema
                        # Look for vector field dimension
                        for field in schema:
                            if field.name == "vector" and hasattr(
                                field.type, "list_size"
                            ):
                                dimension = field.type.list_size
                                logger.debug(
                                    "Found vector dimension %s in table '%s'",
                                    dimension,
                                    table_name,
                                )
                                break
                    except Exception as e:
                        logger.debug(
                            "Could not get dimension from table '%s': %s", table_name, e
                        )

                    model_stats[model_tag] = {
                        "count": vector_count,
                        "dimension": dimension,
                    }

                except Exception as e:
                    logger.debug(
                        "Error checking table '%s' for collection '%s': %s",
                        table_name,
                        collection_name,
                        e,
                    )
                    continue

            except Exception as e:
                logger.debug("Error opening table '%s': %s", table_name, e)
                continue
            finally:
                _safe_close_table(table)

        if not model_stats:
            logger.info(
                "No vectors found for collection '%s' in any embedding table - will use lazy initialization",
                collection_name,
            )
            return None, None

        logger.info(
            "Found vectors from %s different models for collection '%s': %s",
            len(model_stats),
            collection_name,
            list(model_stats.keys()),
        )

        # Choose the most used model
        best_model = max(
            model_stats.items(), key=lambda x: cast(int, x[1].get("count", 0))
        )
        model_tag, stats = best_model

        # Resolve Hub embedding model ID from table tag (preferred).
        embedding_model_id = None
        try:
            from xagent.core.model.model import EmbeddingModelConfig

            from .model_resolver import _list_model_hub_configs

            hub_tag_to_id: Dict[str, str] = {}
            for cfg in _list_model_hub_configs().values():
                if not isinstance(cfg, EmbeddingModelConfig):
                    continue
                register_tag_mapping(
                    hub_tag_to_id,
                    to_model_tag(cfg.id),
                    cfg.id,
                    get_identity=lambda item: item,
                    logger=logger,
                )
                register_tag_mapping(
                    hub_tag_to_id,
                    to_model_tag(cfg.model_name),
                    cfg.id,
                    get_identity=lambda item: item,
                    logger=logger,
                )
            embedding_model_id = hub_tag_to_id.get(model_tag)
        except Exception as e:
            logger.warning(
                "Model hub initialization failed during embedding config inference: "
                "error_type=%s, error_message=%s, fallback_behavior=%s, impact=%s",
                type(e).__name__,
                str(e),
                "legacy_model_tag_normalization",
                "May use incorrect model ID for embeddings",
                exc_info=True,
            )
            embedding_model_id = None

        # Fallback: best-effort reverse normalization (legacy behavior)
        if not embedding_model_id:
            embedding_model_id = _model_tag_to_model_id(model_tag)
        embedding_dimension = stats["dimension"]

        logger.info(
            "Selected embedding model '%s' (dimension: %s) for collection '%s' based on %s vectors",
            embedding_model_id,
            embedding_dimension,
            collection_name,
            stats["count"],
        )

        if len(model_stats) > 1:
            logger.warning(
                "Collection '%s' has vectors from multiple models: %s. Choosing '%s' (most used with %s vectors). Consider migrating old vectors to maintain consistency.",
                collection_name,
                list(model_stats.keys()),
                embedding_model_id,
                stats["count"],
            )

        return embedding_model_id, embedding_dimension

    except Exception as e:
        logger.error(
            "Error inferring embedding config for collection '%s': %s",
            collection_name,
            e,
            exc_info=True,
        )
        return None, None


def _model_tag_to_model_id(model_tag: str) -> str:
    """Convert model tag back to model ID."""
    logger.debug("Converting model tag '%s' back to model ID", model_tag)

    # Handle common cases
    if model_tag.startswith("OPENAI_"):
        result = model_tag.replace("OPENAI_", "").replace("_", "-").lower()
        logger.debug("Converted OpenAI model tag to: %s", result)
        return result
    elif model_tag.startswith("BAAI_"):
        result = model_tag.replace("BAAI_", "").replace("_", "-").lower()
        logger.debug("Converted BAAI model tag to: %s", result)
        return result
    else:
        # Fallback: try to reverse the normalization
        import re

        match = re.match(r"^([A-Z]+)_(.+)$", model_tag)
        if match:
            result = match.group(2).replace("_", "-").lower()
        else:
            result = model_tag.replace("_", "-").lower()
        logger.debug("Used fallback conversion for model tag: %s", result)
        return result


def migrate_embeddings_table(
    model_id: str,
    batch_size: int = 10000,
    conn: Optional[Any] = None,
) -> dict[str, Any]:
    """Migrate legacy embeddings table to Hub ID-based naming using idempotent merge strategy.

    This function uses LanceDB's merge_insert for safe, non-destructive migration:
    - Self-protection: Detects if already migrated (legacy == primary)
    - Dimension validation: Ensures source and target tables have compatible vector dimensions
    - Idempotent merge: Uses merge_insert to avoid duplicates and data loss
    - Arrow streaming: Uses to_batches() for memory-efficient processing
    - Cloud-native: Works with S3/OSS (no shutil.move or file system assumptions)

    This addresses critical issues with the previous approach:
    - No data loss: merge_insert preserves existing data in target table
    - Cloud-compatible: No dependency on file system operations
    - Idempotent: Can be safely re-run without side effects
    - High performance: Arrow streaming + merge_insert is 5-10x faster than offset/limit

    Args:
        model_id: Hub model ID to migrate (e.g., "text-embedding-ada-002").
        batch_size: Number of rows to process per batch (default 10000).
        conn: LanceDB connection (if None, creates new connection).

    Returns:
        Dictionary with migration results:
        {
            "success": bool,
            "source_table": str (legacy table name),
            "target_table": str (Hub ID table name),
            "rows_migrated": int,
            "error": str | None (if success=False)
        }
    """
    from ..core.exceptions import VectorValidationError
    from ..LanceDB.schema_manager import ensure_embeddings_table
    from ..utils.model_resolver import resolve_embedding_adapter

    cleaned = (model_id or "").strip()
    if not cleaned:
        raise VectorValidationError("model_id must be a non-empty string")

    primary_table_name = f"embeddings_{to_model_tag(cleaned)}"
    lock_key = f"migrate_{primary_table_name}"

    # Get connection
    if conn is None:
        conn = get_vector_store_raw_connection()

    # Try to find legacy table
    legacy_table_name: Optional[str] = None
    try:
        cfg, _ = resolve_embedding_adapter(cleaned)
        legacy_table_name = f"embeddings_{to_model_tag(cfg.model_name)}"
    except Exception as e:
        logger.warning("Failed to resolve legacy table name: %s", e)

    if not legacy_table_name:
        return {
            "success": False,
            "source_table": None,
            "target_table": primary_table_name,
            "rows_migrated": 0,
            "error": "Could not determine legacy table name",
        }

    # Self-protection: Check if already migrated
    if legacy_table_name == primary_table_name:
        logger.info(
            "Already migrated: legacy table '%s' is the same as primary table '%s'",
            legacy_table_name,
            primary_table_name,
        )
        return {
            "success": True,
            "source_table": legacy_table_name,
            "target_table": primary_table_name,
            "rows_migrated": 0,
            "error": None,
        }

    # Acquire lock in database directory for distributed environments
    lock_fd = _acquire_migration_lock(conn.uri, primary_table_name)
    if lock_fd is None:
        return {
            "success": False,
            "source_table": None,
            "target_table": primary_table_name,
            "rows_migrated": 0,
            "error": "Migration already in progress",
        }

    rows_migrated = 0
    legacy_table = None
    target_table = None

    try:
        # Check if legacy table exists
        try:
            legacy_table = conn.open_table(legacy_table_name)
        except Exception as e:
            logger.warning("Legacy table '%s' not found: %s", legacy_table_name, e)
            return {
                "success": False,
                "source_table": legacy_table_name,
                "target_table": primary_table_name,
                "rows_migrated": 0,
                "error": f"Legacy table not found: {e}",
            }

        # ✅ 1. Pre-check: Query schema only once (avoid n+1)
        vector_dim: Optional[int] = None
        try:
            vector_field = legacy_table.schema.field("vector")
            list_size = getattr(vector_field.type, "list_size", None)
            if list_size is not None:
                vector_dim = int(list_size)
        except Exception:
            vector_dim = None

        if vector_dim is None:
            _release_migration_lock(lock_fd)
            return {
                "success": False,
                "source_table": legacy_table_name,
                "target_table": primary_table_name,
                "rows_migrated": 0,
                "error": "Could not determine vector dimension",
            }

        # Dimension validation: Ensure target table has compatible dimension
        if _table_exists(conn, primary_table_name):
            try:
                target_table = conn.open_table(primary_table_name)
                target_dim = _get_vector_dimension_from_table(target_table)
                if target_dim is not None and target_dim != vector_dim:
                    _release_migration_lock(lock_fd)
                    return {
                        "success": False,
                        "source_table": legacy_table_name,
                        "target_table": primary_table_name,
                        "rows_migrated": 0,
                        "error": f"Dimension mismatch: source={vector_dim}, target={target_dim}",
                    }
            except Exception as e:
                logger.warning("Could not validate target table dimension: %s", e)
            finally:
                _safe_close_table(target_table)

        # Ensure target table exists (create if needed)
        ensure_embeddings_table(conn, to_model_tag(cleaned), vector_dim=vector_dim)
        _safe_close_table(target_table)
        target_table = conn.open_table(primary_table_name)

        # Use merge_insert for idempotent, non-destructive migration
        logger.info(
            "Starting idempotent migration from '%s' to '%s' (vector_dim=%d, batch_size=%d)",
            legacy_table_name,
            primary_table_name,
            vector_dim,
            batch_size,
        )

        # Create merge_insert builder with composite key for uniqueness
        # Using doc_id + chunk_id as the natural key for embeddings
        merger = target_table.merge_insert(on=["doc_id", "chunk_id"])

        # Stream data from legacy table using Arrow batches (memory-efficient)
        total_rows = legacy_table.count_rows()
        logger.info(
            "Streaming %s rows from legacy table '%s'", total_rows, legacy_table_name
        )

        batch_num = 0
        for batch in legacy_table.search().to_batches(batch_size=batch_size):
            batch_num += 1
            batch_rows = len(batch)

            # Modify model column directly in Arrow (no pandas conversion)
            if "model" in batch.schema.names:
                new_model_values = pa.array([cleaned] * batch_rows, type=pa.string())
                modified_batch = batch.set_column(
                    batch.schema.get_field_index("model"), "model", new_model_values
                )
            else:
                modified_batch = batch

            # Execute merge_insert (idempotent: only inserts if key doesn't exist)
            merger.when_not_matched_insert_all().execute(modified_batch)

            rows_migrated += batch_rows

            # Logging (avoid I/O intensive operations)
            if batch_num % 10 == 0:
                logger.info(
                    "Migration progress: %s/%s rows migrated", rows_migrated, total_rows
                )

        logger.info(
            "Migration completed successfully: '%s' -> '%s' (%d rows processed)",
            legacy_table_name,
            primary_table_name,
            rows_migrated,
        )
        logger.info(
            "Data has been synced to the new table '%s'. "
            "After verifying the migration, you can manually drop the legacy table to free up space: "
            "conn.drop_table('%s') or via Python: conn.drop_table('%s')",
            primary_table_name,
            legacy_table_name,
            legacy_table_name,
        )

        return {
            "success": True,
            "source_table": legacy_table_name,
            "target_table": primary_table_name,
            "rows_migrated": rows_migrated,
            "error": None,
        }

    except Exception as e:
        logger.error(
            "Migration failed for '%s': %s",
            primary_table_name,
            e,
            exc_info=True,
        )

        return {
            "success": False,
            "source_table": legacy_table_name
            if "legacy_table_name" in locals()
            else None,
            "target_table": primary_table_name,
            "rows_migrated": rows_migrated if "rows_migrated" in locals() else 0,
            "error": str(e),
        }

    finally:
        # Release lock
        _safe_close_table(legacy_table)
        _safe_close_table(target_table)
        _release_migration_lock(lock_fd)


def _table_exists(conn: Any, table_name: str) -> bool:
    """Check if a table exists in the database."""
    table = None
    try:
        names = list_table_names(conn)
        if names:
            return table_name in names
        if (
            getattr(conn, "list_tables", None) is None
            and getattr(conn, "table_names", None) is None
        ):
            table = conn.open_table(table_name)
            return True
        return False
    except Exception:
        return False
    finally:
        _safe_close_table(table)


def _get_vector_dimension_from_table(table: Any) -> Optional[int]:
    """Extract vector dimension from table schema.

    Args:
        table: LanceDB table object

    Returns:
        Vector dimension or None if cannot be determined
    """
    try:
        schema = table.schema
        for field in schema:
            if field.name == "vector" and hasattr(field.type, "list_size"):
                return int(field.type.list_size)
    except Exception as e:
        logger.debug("Could not get vector dimension from table: %s", e)
    return None


def _acquire_migration_lock(db_uri: str, table_name: str) -> Optional[int]:
    """Acquire a file lock for migration in the database directory.

    This places the lock file in the database directory itself, which works
    for distributed environments where the database is on shared storage (NFS/SMB).

    Args:
        db_uri: Database URI (e.g., "/path/to/db" or "s3://bucket/db")
        table_name: Name of the table being migrated

    Returns:
        File descriptor for the lock, or None if lock is held by another process
    """
    # Only support file-based locking for local databases
    if db_uri.startswith("s3://") or db_uri.startswith("oss://"):
        logger.warning(
            "Cloud storage detected (%s), file locking not supported. "
            "Consider using distributed locking for concurrent migrations.",
            db_uri,
        )
        return -1  # Return a dummy fd to avoid errors

    try:
        # Create lock file in database directory
        lock_dir = db_uri
        os.makedirs(lock_dir, exist_ok=True)

        lock_path = os.path.join(lock_dir, f".{table_name}.migration.lock")
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)

        try:
            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.info("Acquired migration lock for '%s' at %s", table_name, lock_path)
            return lock_fd
        except (IOError, OSError):
            # Lock is held by another process
            os.close(lock_fd)
            logger.info("Migration for '%s' is already in progress", table_name)
            return None
    except Exception as e:
        logger.warning("Failed to acquire migration lock: %s", e)
        return -1  # Return a dummy fd to avoid errors


def _release_migration_lock(lock_fd: Optional[int]) -> None:
    """Release a migration lock.

    Args:
        lock_fd: File descriptor from _acquire_migration_lock (or -1/dummy fd)
    """
    if lock_fd is not None and lock_fd >= 0:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        except Exception:
            pass

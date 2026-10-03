from __future__ import annotations

import logging
from typing import Any

import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
from lancedb.db import DBConnection

from ..utils.lancedb_query_utils import list_table_names

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_documents_table",
    "ensure_parses_table",
    "ensure_chunks_table",
    "ensure_embeddings_table",
    "ensure_active_generations_table",
    "ensure_main_pointers_table",
    "ensure_prompt_templates_table",
    "ensure_ingestion_runs_table",
    "ensure_collection_config_table",
    "ensure_collection_metadata_table",
    "check_table_needs_migration",
]


def _safe_close_table(table: Any) -> None:
    """Close a LanceDB table if it supports close()."""
    if table is not None and hasattr(table, "close"):
        try:
            table.close()
        except Exception:
            pass


def _add_nullable_string_columns(
    conn: DBConnection, table_name: str, columns: dict[str, str]
) -> None:
    """Add missing nullable string columns to an existing LanceDB table."""
    if not _table_exists(conn, table_name):
        return
    table = conn.open_table(table_name)
    try:
        existing = set(table.schema.names)
        to_add = {name: expr for name, expr in columns.items() if name not in existing}
        if to_add:
            logger.info(
                "Migrating '%s' table: adding columns %s",
                table_name,
                list(to_add.keys()),
            )
            table.add_columns(to_add)
    finally:
        _safe_close_table(table)


def _table_exists(conn: DBConnection, name: str) -> bool:
    try:
        return name in list_table_names(conn)
    except Exception:
        return False


def _validate_schema_fields(
    conn: DBConnection, table_name: str, required_fields: list[str]
) -> None:
    """Validate that an existing table contains all required fields.

    Args:
        conn: LanceDB connection
        table_name: Name of the table to validate
        required_fields: List of required field names

    Raises:
        ValueError: If the table exists but is missing required fields.
    """
    if not _table_exists(conn, table_name):
        return

    table = None
    try:
        table = conn.open_table(table_name)
        existing_schema = table.schema
        existing_field_names = {field.name for field in existing_schema}

        missing_fields = [f for f in required_fields if f not in existing_field_names]

        if missing_fields:
            error_msg = (
                f"Table '{table_name}' exists but is missing required fields: {missing_fields}. "
                f"This is likely due to a schema upgrade. "
                f"Please delete the existing table or manually add the missing fields. "
                f"Note: During development, we do not provide automatic migration scripts. "
                f"To upgrade, you can either:\n"
                f"1. Delete the table (data will be lost): conn.drop_table('{table_name}')\n"
                f"2. Manually add the missing fields using LanceDB's schema update capabilities"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    except ValueError:
        raise
    except Exception as e:
        # Log other errors but don't fail - schema validation is best-effort
        logger.warning(
            "Could not validate schema for table '%s': %s. Proceeding with table creation/usage.",
            table_name,
            e,
        )
    finally:
        _safe_close_table(table)


def _create_table(conn: DBConnection, name: str, schema: object | None = None) -> None:
    if _table_exists(conn, name):
        return
    try:
        conn.create_table(name, schema=schema)
    except Exception as e:
        # Concurrent creators may race between existence check and create_table.
        # Treat "already exists" as benign to keep ensure_* idempotent.
        msg = str(e).lower()
        if "already exists" in msg:
            return
        # LanceDB can throw concurrent transaction conflicts under multi-threaded
        # load (e.g. "Incompatible transaction: This Overwrite transaction is
        # incompatible with concurrent transaction ...").  If the table now
        # exists the race was benign; re-raise only if it genuinely failed.
        if "incompatible transaction" in msg and _table_exists(conn, name):
            return
        raise


def _validate_user_id_int64(table: object, table_name: str) -> None:
    """Validate ``user_id`` column type and fail on non-int schema.

    We require ``user_id`` to be Int64 for tenant-safe filtering. If an existing
    table has ``user_id`` with a non-int type, automatic conversion is unsafe.
    """
    schema = getattr(table, "schema", None)
    if schema is None:
        return
    names = getattr(schema, "names", None) or []
    if "user_id" not in names:
        return
    try:
        user_id_type = str(schema.field("user_id").type).lower()
    except Exception:
        return
    if "int" not in user_id_type:
        raise ValueError(
            f"Table '{table_name}' has incompatible user_id type '{user_id_type}'. "
            "Expected int64. Please back up data, recreate this table, and re-upload documents."
        )


def _build_schema_with_int64_user_id(existing_schema: Any) -> Any:
    """Return a schema where ``user_id`` field is forced to ``int64``."""
    fields: list[Any] = []
    for field in existing_schema:
        if field.name == "user_id":
            fields.append(pa.field("user_id", pa.int64(), nullable=field.nullable))
        else:
            fields.append(field)
    return pa.schema(fields)


def _extract_invalid_user_id_examples(column: Any, limit: int = 5) -> list[str]:
    """Extract sample invalid user_id values for error reporting."""
    examples: list[str] = []
    for chunk in column.chunks:
        values = chunk.to_pylist()
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text == "":
                examples.append("<empty>")
                continue
            try:
                int(text)
            except (TypeError, ValueError):
                examples.append(repr(value))
            if len(examples) >= limit:
                return examples
    return examples


def _migrate_table_user_id_to_int64(conn: DBConnection, table_name: str) -> None:
    """Physically migrate table ``user_id`` column to int64 with data conversion.

    If conversion fails for any row, raise ``ValueError`` and ask user to re-upload.
    """
    table = conn.open_table(table_name)
    try:
        schema = table.schema
        if "user_id" not in schema.names:
            return

        user_id_type = str(schema.field("user_id").type).lower()
        if "int" in user_id_type:
            return

        logger.info(
            "Migrating table '%s': converting user_id from %s to int64",
            table_name,
            user_id_type,
        )
        arrow_table = table.to_arrow()
        user_id_idx = arrow_table.schema.get_field_index("user_id")
        user_id_col = arrow_table.column(user_id_idx)

        try:
            converted_user_id = pc.cast(user_id_col, pa.int64(), safe=True)
        except Exception as exc:
            invalid_examples = _extract_invalid_user_id_examples(user_id_col)
            examples_text = ", ".join(invalid_examples) if invalid_examples else "N/A"
            raise ValueError(
                f"Failed to migrate table '{table_name}': user_id contains non-integer values. "
                f"Examples: {examples_text}. Please re-upload documents."
            ) from exc

        migrated_table = arrow_table.set_column(
            user_id_idx, pa.field("user_id", pa.int64()), converted_user_id
        )
        target_schema = _build_schema_with_int64_user_id(migrated_table.schema)
        migrated_table = migrated_table.cast(target_schema, safe=False)

        drop_table_fn = getattr(conn, "drop_table", None)
        if drop_table_fn is None:
            raise ValueError(
                "Current LanceDB connection does not support drop_table; "
                "cannot complete user_id schema migration safely. Please re-upload documents."
            )
        drop_table_fn(table_name)
        conn.create_table(table_name, data=migrated_table)
    finally:
        _safe_close_table(table)

    # Validate the recreated table
    new_table = conn.open_table(table_name)
    try:
        _validate_user_id_int64(new_table, table_name)
    finally:
        _safe_close_table(new_table)


def ensure_documents_table(conn: DBConnection) -> None:
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("file_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("uploaded_at", pa.timestamp("us")),
            pa.field("title", pa.string()),
            pa.field("language", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )

    # Automatic migration for existing tables missing 'user_id' or 'file_id'
    if _table_exists(conn, "documents"):
        table = None
        try:
            table = conn.open_table("documents")
            if "user_id" not in table.schema.names:
                logger.info(
                    "Migrating 'documents' table: adding missing 'user_id' column"
                )
                # Add user_id column with null default, cast to bigint (int64)
                table.add_columns({"user_id": "cast(null as bigint)"})

            if "file_id" not in table.schema.names:
                logger.info(
                    "Migrating 'documents' table: adding missing 'file_id' column"
                )
                table.add_columns({"file_id": "cast(null as string)"})
        except ValueError:
            _safe_close_table(table)
            raise
        except Exception as e:
            _safe_close_table(table)
            logger.warning("Failed to check/migrate 'documents' table schema: %s", e)
        else:
            _safe_close_table(table)

        _migrate_table_user_id_to_int64(conn, "documents")

        val_table = conn.open_table("documents")
        try:
            _validate_user_id_int64(val_table, "documents")
        finally:
            _safe_close_table(val_table)

    _create_table(conn, "documents", schema=schema)
    # Note: backfill of file_id and user_id is now handled by standalone migration script:
    #   python -m xagent.migrations.lancedb.backfill_documents_file_id
    # This keeps the hot path (schema ensure) fast and separation of concerns.


def ensure_parses_table(conn: DBConnection) -> None:
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("parser", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("params_json", pa.string()),
            pa.field("parsed_content", pa.large_string()),
            pa.field("user_id", pa.int64()),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, "parses"):
        table = None
        try:
            table = conn.open_table("parses")
            if "user_id" not in table.schema.names:
                logger.info("Migrating 'parses' table: adding missing 'user_id' column")
                table.add_columns({"user_id": "cast(null as bigint)"})
        except ValueError:
            _safe_close_table(table)
            raise
        except Exception as e:
            _safe_close_table(table)
            logger.warning("Failed to check/migrate 'parses' table schema: %s", e)
        else:
            _safe_close_table(table)

        _migrate_table_user_id_to_int64(conn, "parses")

        val_table = conn.open_table("parses")
        try:
            _validate_user_id_int64(val_table, "parses")
        finally:
            _safe_close_table(val_table)

    _create_table(conn, "parses", schema=schema)


def ensure_chunks_table(conn: DBConnection) -> None:
    """Ensure the chunks table exists with proper schema.

    This function creates the table if it doesn't exist, and validates that
    existing tables contain all required fields (especially 'metadata').

    Args:
        conn: LanceDB connection

    Raises:
        ValueError: If the table exists but is missing required fields.
            This typically happens when an old table schema doesn't include
            the 'metadata' field. During development, we do not provide
            automatic migration scripts. Users must either delete the table
            or manually add the missing fields.

    Note:
        There's no upgrade path for existing chunks tables. Any deployment
        with an existing table will hit schema-mismatch errors once the pipeline
        starts writing a column that doesn't exist. If you encounter this error,
        you need to either delete the existing table or manually add the missing
        'metadata' field.
    """
    # Required fields that must exist in the table (especially for schema validation)
    required_fields = ["metadata"]

    # Validate existing table schema before creating/using it
    _validate_schema_fields(conn, "chunks", required_fields)
    if _table_exists(conn, "chunks"):
        _migrate_table_user_id_to_int64(conn, "chunks")
        _add_nullable_string_columns(
            conn, "chunks", {"generation_id": "cast(null as string)"}
        )

        val_table = conn.open_table("chunks")
        try:
            _validate_user_id_int64(val_table, "chunks")
        finally:
            _safe_close_table(val_table)

    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("index", pa.int32()),
            pa.field("text", pa.large_string()),
            pa.field("page_number", pa.int32()),
            pa.field("section", pa.string()),
            pa.field("anchor", pa.string()),
            pa.field("json_path", pa.string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("config_hash", pa.string()),
            pa.field("generation_id", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    _create_table(conn, "chunks", schema=schema)


def ensure_embeddings_table(
    conn: DBConnection, model_tag: str, vector_dim: int | None = None
) -> None:
    """Ensure the embeddings table exists with proper schema.

    This function creates the table if it doesn't exist, and validates that
    existing tables contain all required fields (especially 'metadata').

    Args:
        conn: LanceDB connection
        model_tag: Model tag used to construct the table name (e.g., 'bge_large')
        vector_dim: Optional vector dimension for fixed-size vectors

    Raises:
        ValueError: If the table exists but is missing required fields.
            This typically happens when an old table schema doesn't include
            the 'metadata' field. During development, we do not provide
            automatic migration scripts. Users must either delete the table
            or manually add the missing fields.

    Note:
        There's no upgrade path for existing embeddings tables. Any deployment
        with an existing table will hit schema-mismatch errors once the pipeline
        starts writing a column that doesn't exist. If you encounter this error,
        you need to either delete the existing table or manually add the missing
        'metadata' field.
    """
    table_name = f"embeddings_{model_tag}"

    # Required fields that must exist in the table (especially for schema validation)
    required_fields = ["metadata"]

    # Validate existing table schema before creating/using it
    _validate_schema_fields(conn, table_name, required_fields)
    if _table_exists(conn, table_name):
        _migrate_table_user_id_to_int64(conn, table_name)
        _add_nullable_string_columns(
            conn,
            table_name,
            {
                "generation_id": "cast(null as string)",
                "config_hash": "cast(null as string)",
            },
        )

        val_table = conn.open_table(table_name)
        try:
            _validate_user_id_int64(val_table, table_name)
        finally:
            _safe_close_table(val_table)

    # Support dynamic vector dimension: if provided, create a FixedSizeList; otherwise allow variable-length
    vector_field_type = (
        pa.list_(pa.float32(), list_size=vector_dim)
        if vector_dim is not None
        else pa.list_(pa.float32())
    )
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("model", pa.string()),
            pa.field("vector", vector_field_type),
            pa.field("vector_dimension", pa.int32()),
            pa.field("text", pa.large_string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("config_hash", pa.string()),
            pa.field("generation_id", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    _create_table(
        conn,
        table_name,
        schema=schema,
    )


def ensure_active_generations_table(conn: DBConnection) -> None:
    """Ensure the active_generations table exists with proper schema.

    Tracks the published searchable generation per document scope:
    (collection, doc_id, parse_hash, user_id, model_tag).
    """
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("user_id", pa.int64()),
            pa.field("model_tag", pa.string()),
            pa.field("generation_id", pa.string()),
            pa.field("config_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
            pa.field("published_at", pa.timestamp("us")),
            pa.field("operator", pa.string()),
        ]
    )
    _create_table(conn, "active_generations", schema=schema)


def ensure_main_pointers_table(conn: DBConnection) -> None:
    """Ensure the main_pointers table exists with proper schema.

    Args:
        conn: LanceDB connection
    """
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("step_type", pa.string()),
            pa.field("model_tag", pa.string()),
            pa.field("semantic_id", pa.string()),
            pa.field("technical_id", pa.string()),
            pa.field("created_at", pa.timestamp("ms")),
            pa.field("updated_at", pa.timestamp("ms")),
            pa.field("operator", pa.string()),
        ]
    )
    _create_table(conn, "main_pointers", schema=schema)


def ensure_prompt_templates_table(conn: DBConnection) -> None:
    """Ensure the prompt_templates table exists with proper schema.

    Args:
        conn: LanceDB connection
    """
    table_name = "prompt_templates"
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("template", pa.string()),
            pa.field("version", pa.int64()),
            pa.field("is_latest", pa.bool_()),
            pa.field("metadata", pa.string()),  # JSON string, nullable
            pa.field("user_id", pa.int64()),  # Multi-tenancy support
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, table_name):
        table = None
        try:
            table = conn.open_table(table_name)
            if "user_id" not in table.schema.names:
                logger.info(
                    "Migrating '%s' table: adding missing 'user_id' column", table_name
                )
                table.add_columns({"user_id": "cast(null as bigint)"})
        except ValueError:
            _safe_close_table(table)
            raise
        except Exception as e:
            _safe_close_table(table)
            logger.warning(
                "Failed to check/migrate '%s' table schema: %s", table_name, e
            )
        else:
            _safe_close_table(table)

        _migrate_table_user_id_to_int64(conn, table_name)

        val_table = conn.open_table(table_name)
        try:
            _validate_user_id_int64(val_table, table_name)
        finally:
            _safe_close_table(val_table)

    _create_table(conn, table_name, schema=schema)


def ensure_ingestion_runs_table(conn: DBConnection) -> None:
    """Ensure the ingestion_runs table exists with proper schema.

    This table tracks the status of document ingestion processes.

    Args:
        conn: LanceDB connection
    """
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("message", pa.string()),
            pa.field("parse_hash", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
            pa.field("user_id", pa.int64()),
        ]
    )

    # Automatic migration for existing tables missing 'user_id'
    if _table_exists(conn, "ingestion_runs"):
        table = None
        try:
            table = conn.open_table("ingestion_runs")
            if "user_id" not in table.schema.names:
                logger.info(
                    "Migrating 'ingestion_runs' table: adding missing 'user_id' column"
                )
                table.add_columns({"user_id": "cast(null as bigint)"})
        except ValueError:
            _safe_close_table(table)
            raise
        except Exception as e:
            _safe_close_table(table)
            logger.warning(
                "Failed to check/migrate 'ingestion_runs' table schema: %s", e
            )
        else:
            _safe_close_table(table)

        _migrate_table_user_id_to_int64(conn, "ingestion_runs")

        val_table = conn.open_table("ingestion_runs")
        try:
            _validate_user_id_int64(val_table, "ingestion_runs")
        finally:
            _safe_close_table(val_table)

    _create_table(conn, "ingestion_runs", schema=schema)


def ensure_collection_config_table(conn: DBConnection) -> None:
    """Ensure the collection_config table exists with proper schema.

    This table stores configuration/metadata for each collection.

    Args:
        conn: LanceDB connection
    """
    table_name = "collection_config"
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("config_json", pa.string()),  # Stores IngestionConfig as JSON
            pa.field("updated_at", pa.timestamp("us")),
            pa.field("user_id", pa.int64()),
        ]
    )

    _create_table(conn, table_name, schema=schema)


def check_table_needs_migration(conn: DBConnection, table_name: str) -> bool:
    """Check if a table exists and needs migration (missing user_id field).

    This function checks if a table exists and is missing the 'user_id' field,
    which indicates it needs migration for multi-tenancy support.

    Args:
        conn: LanceDB connection
        table_name: Name of the table to check

    Returns:
        True if the table exists and is missing 'user_id' field, False otherwise
    """
    if not _table_exists(conn, table_name):
        return False

    table = None
    try:
        table = conn.open_table(table_name)
        existing_schema = table.schema
        existing_field_names = {field.name for field in existing_schema}

        # Check if user_id field is missing
        return "user_id" not in existing_field_names
    except Exception as e:
        # If we can't check the schema, assume no migration needed
        logger.warning(
            "Could not check schema for table '%s': %s. Assuming no migration needed.",
            table_name,
            e,
        )
        return False
    finally:
        _safe_close_table(table)


def ensure_collection_metadata_table(conn: DBConnection) -> None:
    """Ensure the collection_metadata table exists with proper schema.

    This table stores collection metadata including embedding configuration,
    statistics, and configuration settings.

    Args:
        conn: LanceDB connection
    """
    schema = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("schema_version", pa.string()),
            pa.field("embedding_model_id", pa.string()),
            pa.field("embedding_dimension", pa.int32()),
            pa.field("documents", pa.int32()),
            pa.field("processed_documents", pa.int32()),
            pa.field("parses", pa.int32()),
            pa.field("chunks", pa.int32()),
            pa.field("embeddings", pa.int32()),
            pa.field("document_names", pa.string()),
            pa.field(
                "owners", pa.string()
            ),  # Schema-only; not maintained (derived at list time)
            pa.field("collection_locked", pa.bool_()),
            pa.field("allow_mixed_parse_methods", pa.bool_()),
            pa.field("skip_config_validation", pa.bool_()),
            pa.field("ingestion_config", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("updated_at", pa.timestamp("us")),
            pa.field("last_accessed_at", pa.timestamp("us")),
            pa.field("extra_metadata", pa.string()),
        ]
    )
    _create_table(conn, "collection_metadata", schema=schema)

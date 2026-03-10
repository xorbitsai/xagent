from __future__ import annotations

import logging
import time

import pyarrow as pa  # type: ignore
from lancedb.db import DBConnection

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_documents_table",
    "ensure_parses_table",
    "ensure_chunks_table",
    "ensure_embeddings_table",
    "ensure_main_pointers_table",
    "ensure_prompt_templates_table",
    "ensure_ingestion_runs_table",
]


def _table_exists(conn: DBConnection, name: str) -> bool:
    try:
        conn.open_table(name)
        return True
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
        # Re-raise ValueError (our validation error)
        raise
    except Exception as e:
        # Log other errors but don't fail - schema validation is best-effort
        logger.warning(
            f"Could not validate schema for table '{table_name}': {e}. "
            f"Proceeding with table creation/usage."
        )


def _create_table(conn: DBConnection, name: str, schema: object | None = None) -> None:
    """Create a table if it doesn't exist.

    Args:
        conn: LanceDB connection
        name: Table name
        schema: Table schema (PyArrow schema)

    Raises:
        Exception: If table creation fails
    """
    if _table_exists(conn, name):
        return

    try:
        conn.create_table(name, schema=schema)
        if not _table_exists(conn, name):
            raise RuntimeError(
                f"Table '{name}' creation reported success but table does not exist"
            )
        logger.info("Successfully created table '%s'", name)
    except Exception as e:
        logger.error("Failed to create table '%s': %s", name, e)
        raise


def _ensure_table_fields(
    conn: DBConnection,
    table_name: str,
    required_fields: list[str],
    auto_addable_sql: dict[str, str],
    validate_if_not_exists: bool = False,
) -> None:
    """Ensure table has required fields; add missing ones via SQL expression when possible.

    Shared logic for schema migration: if table exists, check for missing required fields,
    add them using auto_addable_sql (field name -> SQL expression, e.g. {"user_id": "cast(null as bigint)"}),
    then re-check. If still missing or add fails, validate and may raise.

    Args:
        conn: LanceDB connection
        table_name: Table to check/update
        required_fields: Required field names (e.g. ["user_id"] or ["metadata", "user_id"])
        auto_addable_sql: Map field name -> SQL expression for adding column (only these are auto-added)
        validate_if_not_exists: If True, run validation even when table does not exist (e.g. chunks)

    Raises:
        ValueError: If table exists but is missing required fields that could not be added.
    """
    table_exists = _table_exists(conn, table_name)
    if table_exists:
        try:
            table = conn.open_table(table_name)
            existing_names = getattr(
                table.schema, "names", [f.name for f in table.schema]
            )
            missing = [f for f in required_fields if f not in existing_names]
            if missing:
                logger.info(
                    "Table '%s' missing fields %s; attempting to add automatically",
                    table_name,
                    missing,
                )
                to_add = {k: v for k, v in auto_addable_sql.items() if k in missing}
                if to_add:
                    try:
                        table.add_columns(to_add)
                        logger.info(
                            "Added fields %s to table '%s'",
                            list(to_add.keys()),
                            table_name,
                        )
                        time.sleep(0.1)
                        table = conn.open_table(table_name)
                        existing_names = getattr(
                            table.schema, "names", [f.name for f in table.schema]
                        )
                        missing = [
                            f for f in required_fields if f not in existing_names
                        ]
                        if not missing:
                            logger.info(
                                "All required fields present in table '%s'",
                                table_name,
                            )
                        else:
                            logger.warning(
                                "After add, table '%s' still missing: %s",
                                table_name,
                                missing,
                            )
                    except Exception as e:
                        logger.error(
                            "Failed to add fields to table '%s': %s",
                            table_name,
                            e,
                            exc_info=True,
                        )
                if missing:
                    logger.error(
                        "Table '%s' still missing fields after auto-add: %s",
                        table_name,
                        missing,
                    )
                    _validate_schema_fields(conn, table_name, required_fields)
        except ValueError:
            raise
        except Exception as e:
            logger.warning(
                "Error checking/updating table '%s' schema: %s",
                table_name,
                e,
            )
            _validate_schema_fields(conn, table_name, required_fields)
    else:
        if validate_if_not_exists:
            _validate_schema_fields(conn, table_name, required_fields)


def ensure_documents_table(conn: DBConnection) -> None:
    """Ensure the documents table exists with proper schema and user_id."""
    _ensure_table_fields(
        conn,
        "documents",
        required_fields=["user_id"],
        auto_addable_sql={"user_id": "cast(null as bigint)"},
    )
    if _table_exists(conn, "documents"):
        logger.debug("Table 'documents' already exists with correct schema")
        return
    schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("file_type", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("uploaded_at", pa.timestamp("us")),
            pa.field("title", pa.string()),
            pa.field("language", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    _create_table(conn, "documents", schema=schema)


def ensure_parses_table(conn: DBConnection) -> None:
    """Ensure the parses table exists with proper schema and user_id."""
    _ensure_table_fields(
        conn,
        "parses",
        required_fields=["user_id"],
        auto_addable_sql={"user_id": "cast(null as bigint)"},
    )
    if _table_exists(conn, "parses"):
        logger.debug("Table 'parses' already exists with correct schema")
        return
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
    _ensure_table_fields(
        conn,
        "chunks",
        required_fields=["metadata", "user_id"],
        auto_addable_sql={"user_id": "cast(null as bigint)"},
        validate_if_not_exists=True,
    )
    if _table_exists(conn, "chunks"):
        logger.debug("Table 'chunks' already exists with correct schema")
        return
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
    _ensure_table_fields(
        conn,
        table_name,
        required_fields=["metadata", "user_id"],
        auto_addable_sql={"user_id": "cast(null as bigint)"},
    )
    if _table_exists(conn, table_name):
        logger.debug("Table '%s' already exists with correct schema", table_name)
        return
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
            pa.field("created_at", pa.timestamp("us")),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int64()),
        ]
    )
    try:
        _create_table(
            conn,
            table_name,
            schema=schema,
        )
        if not _table_exists(conn, table_name):
            raise RuntimeError(
                f"Table '{table_name}' creation failed: table does not exist after creation"
            )
        logger.info("Successfully created embeddings table '%s'", table_name)
    except Exception as e:
        logger.error(
            "Failed to create embeddings table '%s' for model '%s': %s",
            table_name,
            model_tag,
            e,
        )
        raise ValueError(
            f"Failed to create embeddings table '{table_name}': {str(e)}"
        ) from e


def ensure_main_pointers_table(conn: DBConnection) -> None:
    """Ensure the main_pointers table exists with proper schema and user_id.

    Args:
        conn: LanceDB connection
    """
    _ensure_table_fields(
        conn,
        "main_pointers",
        required_fields=["user_id"],
        auto_addable_sql={"user_id": "cast(null as bigint)"},
    )
    if _table_exists(conn, "main_pointers"):
        logger.debug("Table 'main_pointers' already exists with correct schema")
        return
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
            pa.field("user_id", pa.int64()),
        ]
    )
    _create_table(conn, "main_pointers", schema=schema)


def ensure_prompt_templates_table(conn: DBConnection) -> None:
    """Ensure the prompt_templates table exists with proper schema and user_id.

    Args:
        conn: LanceDB connection
    """
    table_name = "prompt_templates"
    _ensure_table_fields(
        conn,
        table_name,
        required_fields=["user_id"],
        auto_addable_sql={"user_id": "cast(null as bigint)"},
    )
    if _table_exists(conn, table_name):
        logger.debug("Table '%s' already exists with correct schema", table_name)
        return
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
    _create_table(conn, table_name, schema=schema)


def ensure_ingestion_runs_table(conn: DBConnection) -> None:
    """Ensure the ingestion_runs table exists with proper schema and user_id.

    This table tracks the status of document ingestion processes.

    Args:
        conn: LanceDB connection
    """
    _ensure_table_fields(
        conn,
        "ingestion_runs",
        required_fields=["user_id"],
        auto_addable_sql={"user_id": "cast(null as bigint)"},
    )
    if _table_exists(conn, "ingestion_runs"):
        logger.debug("Table 'ingestion_runs' already exists with correct schema")
        return
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
    _create_table(conn, "ingestion_runs", schema=schema)

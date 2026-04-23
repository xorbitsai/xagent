"""Cascade cleanup functions for version management.

Provide cascade cleanup utilities when promoting main versions,
ensuring data consistency across processing stages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from typing_extensions import Literal

from ..core.exceptions import CascadeCleanupError
from ..LanceDB.schema_manager import (
    ensure_chunks_table,
    ensure_documents_table,
    ensure_ingestion_runs_table,
    ensure_main_pointers_table,
    ensure_parses_table,
)
from ..storage.factory import get_vector_store_raw_connection
from ..utils.lancedb_query_utils import _safe_count_rows
from ..utils.string_utils import (
    build_lancedb_filter_expression,
    build_user_id_filter_for_table,
    escape_lancedb_string,
)
from .main_pointer_manager import get_main_pointer

logger = logging.getLogger(__name__)


def _table_has_column(table: Any, column: str) -> bool:
    try:
        schema = getattr(table, "schema", None)
        if schema is None:
            return False
        names = getattr(schema, "names", None)
        if not isinstance(names, list):
            return False
        return column in names
    except Exception:
        return False


def _build_collection_filter(
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
    base: Dict[str, str] = {"collection": collection}
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
            return build_lancedb_filter_expression(base)
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    except Exception:
        # If table introspection fails, keep tenant-safe fallback.
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)


def _build_document_filter(
    *,
    conn: Any,
    table_name: str,
    collection: str,
    doc_id: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Build a safe filter for document-scoped deletion."""
    base: Dict[str, str] = {"collection": collection, "doc_id": doc_id}
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
            return build_lancedb_filter_expression(base)
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    except Exception:
        # If table introspection fails, keep tenant-safe fallback.
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)


def _append_user_filter_if_needed(
    *,
    conn: Any,
    table_name: str,
    base_expr: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Append user_id filter when non-admin and table contains user_id."""
    if is_admin or user_id is None:
        return base_expr
    try:
        table = conn.open_table(table_name)
        if _table_has_column(table, "user_id"):
            return (
                f"{base_expr} AND {build_user_id_filter_for_table(table, int(user_id))}"
            )
    except Exception:
        return base_expr
    return base_expr


def _append_user_filter_for_embeddings_if_needed(
    *,
    conn: Any,
    base_expr: str,
    user_id: Optional[int],
    is_admin: bool,
    model_tag: Optional[str] = None,
) -> str:
    """Append user filter for embeddings by inspecting embeddings table schema."""
    if is_admin or user_id is None:
        return base_expr
    table_names = _get_table_names(conn)
    target_tables = [t for t in table_names if t.startswith("embeddings_")]
    if model_tag is not None:
        target_tables = [t for t in target_tables if t == f"embeddings_{model_tag}"]
    if not target_tables:
        return base_expr
    try:
        table = conn.open_table(target_tables[0])
        if not _table_has_column(table, "user_id"):
            return base_expr
        return f"{base_expr} AND {build_user_id_filter_for_table(table, int(user_id))}"
    except Exception:
        return base_expr


def _get_table_names(conn: Any) -> list[str]:
    """Get table names from LanceDB connection with mypy-safe access."""
    table_names_fn = getattr(conn, "table_names", None)
    if table_names_fn is None:
        return []
    try:
        names = table_names_fn()
    except Exception:
        return []
    if not names:
        return []
    return [str(name) for name in names]


def _plan_by_predicates(
    conn: Any, table_to_filter: Dict[str, str], model_tag: Optional[str] = None
) -> Dict[str, int]:
    """Count rows that match each table predicate without deleting.

    Args:
        conn: LanceDB connection
        table_to_filter: Mapping of table name -> filter expression
        model_tag: Optional model tag to filter embeddings tables. If specified,
                   only the embeddings table matching this model will be counted.

    Returns:
        Mapping of table name -> matched row count
    """
    counts: Dict[str, int] = {}
    table_names = _get_table_names(conn)

    # If predicates explicitly include embeddings tables, plan them first.
    for t in table_names:
        if t.startswith("embeddings_") and t in table_to_filter:
            table = conn.open_table(t)
            counts[t] = _safe_count_rows(table, table_to_filter[t])

    for table_name, filt in table_to_filter.items():
        # Special fan-out handling for embeddings preview like deleter
        if table_name == "__embeddings__":
            total = 0
            all_embed_tables = [t for t in table_names if t.startswith("embeddings_")]
            # Apply model_tag filter if specified (must match _delete_by_predicates logic)
            if model_tag:
                all_embed_tables = [
                    t for t in all_embed_tables if t == f"embeddings_{model_tag}"
                ]
            for t in all_embed_tables:
                table = conn.open_table(t)
                count = _safe_count_rows(table, filt)
                total += count
            counts[table_name] = total
            continue

        if table_name not in table_names:
            counts[table_name] = 0
            continue
        table = conn.open_table(table_name)
        count = _safe_count_rows(table, filt)
        counts[table_name] = count
    return counts


def _delete_by_predicates(
    conn: Any, table_to_filter: Dict[str, str], model_tag: Optional[str] = None
) -> Dict[str, int]:
    """Delete rows by table predicates in a fixed, safe order.

    Order: embeddings_* -> chunks -> parses -> main_pointers -> documents
    Unknown tables are executed after the known order, in given insertion order.

    Args:
        conn: LanceDB connection
        table_to_filter: Dictionary mapping table names to filter expressions
        model_tag: Optional model tag to filter embeddings tables. If specified,
                   only the embeddings table matching this model will be processed.
    """
    deleted: Dict[str, int] = {}
    table_names = _get_table_names(conn)

    # If predicates explicitly include embeddings tables, delete them first.
    for t in table_names:
        if not t.startswith("embeddings_") or t not in table_to_filter:
            continue
        filt = table_to_filter[t]
        table = conn.open_table(t)
        cnt = _safe_count_rows(table, filt)
        if cnt > 0:
            table.delete(filt)
            logger.info(f"Cascade cleanup: deleted {cnt} rows from {t}")
        deleted[t] = cnt

    order = [
        # embeddings handled specially below (fan-out across many tables)
        "__embeddings__",
        "chunks",
        "parses",
        "main_pointers",
        "ingestion_runs",
        "documents",
    ]

    # First handle embeddings fan-out
    if "__embeddings__" in table_to_filter:
        filt = table_to_filter["__embeddings__"]
        total = 0

        # Filter embeddings tables based on model_tag if specified
        all_embed_tables = [t for t in table_names if t.startswith("embeddings_")]
        if model_tag is not None:
            target_tables = [
                t for t in all_embed_tables if t == f"embeddings_{model_tag}"
            ]
        else:
            target_tables = all_embed_tables

        for t in target_tables:
            table = conn.open_table(t)
            cnt = _safe_count_rows(table, filt)
            if cnt > 0:
                table.delete(filt)
            total += cnt
        deleted["embeddings"] = total
        if total > 0:
            logger.info(f"Cascade cleanup: deleted {total} rows from embeddings tables")

    # Then handle known tables
    for name in order[1:]:
        if name in table_to_filter and name in table_names:
            filt = table_to_filter[name]
            table = conn.open_table(name)
            cnt = _safe_count_rows(table, filt)
            if cnt > 0:
                table.delete(filt)
                logger.info(f"Cascade cleanup: deleted {cnt} rows from {name}")
            deleted[name] = cnt

    # Finally, handle any remaining custom tables once
    for name, filt in table_to_filter.items():
        if name in (
            "__embeddings__",
            "chunks",
            "parses",
            "main_pointers",
            "ingestion_runs",
            "documents",
        ):
            continue
        if name.startswith("embeddings_"):
            # Already handled in the explicit embeddings pass above.
            continue
        if name not in table_names:
            deleted[name] = 0
            continue
        table = conn.open_table(name)
        cnt = _safe_count_rows(table, filt)
        if cnt > 0:
            table.delete(filt)
            logger.info(f"Cascade cleanup: deleted {cnt} rows from {name}")
        deleted[name] = cnt

    return deleted


def cascade_delete(
    *,
    target: Literal["collection", "document"],
    collection: str,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    model_tag: Optional[str] = None,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Unified cascade delete for collection or document targets.

    This is intended for user-facing destructive operations (e.g. KB delete)
    and is separate from version promotion cleanup scopes.

    Args:
        target: "collection" or "document".
        collection: Collection name.
        doc_id: Required when target == "document".
        user_id: Optional user ID for multi-tenancy filtering.
        is_admin: Whether the caller is an admin (no user_id filtering).
        model_tag: Optional embeddings model tag limiter.
        preview_only: If True, only plan counts.
        confirm: If True, execute deletions.

    Returns:
        Mapping of table name -> deleted (or planned) row count.
    """
    if target == "document" and not doc_id:
        raise CascadeCleanupError("doc_id is required for document cascade delete")

    conn = get_vector_store_raw_connection()
    ensure_documents_table(conn)
    ensure_parses_table(conn)
    ensure_chunks_table(conn)
    ensure_main_pointers_table(conn)
    ensure_ingestion_runs_table(conn)

    table_names = _get_table_names(conn)
    predicates: Dict[str, str] = {}

    # Core known tables
    core_tables = ["documents", "parses", "chunks", "main_pointers", "ingestion_runs"]
    for t in core_tables:
        if t not in table_names:
            continue
        if target == "collection":
            predicates[t] = _build_collection_filter(
                conn=conn,
                table_name=t,
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )
        else:
            predicates[t] = _build_document_filter(
                conn=conn,
                table_name=t,
                collection=collection,
                doc_id=str(doc_id),
                user_id=user_id,
                is_admin=is_admin,
            )

    # Embeddings tables: expand explicitly so we can safely include user_id filter
    for t in table_names:
        if not t.startswith("embeddings_"):
            continue
        if model_tag is not None and t != f"embeddings_{model_tag}":
            continue
        if target == "collection":
            predicates[t] = _build_collection_filter(
                conn=conn,
                table_name=t,
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )
        else:
            predicates[t] = _build_document_filter(
                conn=conn,
                table_name=t,
                collection=collection,
                doc_id=str(doc_id),
                user_id=user_id,
                is_admin=is_admin,
            )

    if preview_only and not confirm:
        return _plan_by_predicates(conn, predicates, model_tag=None)

    return _delete_by_predicates(conn, predicates, model_tag=None)


def cleanup_cascade(
    collection: str,
    doc_id: str,
    scope: str,
    new_parse_hash: Optional[str] = None,
    old_parse_hash: Optional[str] = None,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Unified cascade cleanup by scope with preview/confirm semantics.

    Args:
        collection: Collection name
        doc_id: Document ID
        scope: "document" | "parse" | "chunk" | "embeddings" | "pointers"
        new_parse_hash: New main parse hash for parse/chunk scopes
        old_parse_hash: Optional old main parse hash (auto-filled from pointers if None)
        model_tag: Optional embed model tag limiter
        user_id: Optional user ID for tenant scoping
        is_admin: Whether caller is admin (no user_id filter)
        preview_only: If True, only plan counts
        confirm: If True, execute deletions

    Returns:
        Deleted (or planned) counts per table scope
    """
    conn = get_vector_store_raw_connection()
    ensure_documents_table(conn)
    ensure_parses_table(conn)
    ensure_chunks_table(conn)
    ensure_main_pointers_table(conn)

    if scope == "document":
        raw = cascade_delete(
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

    predicates: Dict[str, str] = {}

    if scope == "parse":
        # Fill old from pointer if needed
        if old_parse_hash is None:
            pointer = get_main_pointer(collection, doc_id, "parse")
            old_parse_hash = pointer["technical_id"] if pointer else None

        if old_parse_hash:
            # Build safe filter expression for old parse hash
            base_filters = {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": old_parse_hash,
            }
            base = build_lancedb_filter_expression(
                base_filters, user_id=user_id, is_admin=is_admin, skip_user_filter=True
            )
            predicates["__embeddings__"] = _append_user_filter_for_embeddings_if_needed(
                conn=conn,
                base_expr=base,
                user_id=user_id,
                is_admin=is_admin,
                model_tag=model_tag,
            )
            predicates["chunks"] = _append_user_filter_if_needed(
                conn=conn,
                table_name="chunks",
                base_expr=base,
                user_id=user_id,
                is_admin=is_admin,
            )
        if new_parse_hash:
            # Build safe filter expression for new parse hash (using != operator)
            escaped_collection = escape_lancedb_string(collection)
            escaped_doc_id = escape_lancedb_string(doc_id)
            escaped_new_parse_hash = escape_lancedb_string(new_parse_hash)
            other = f"collection == '{escaped_collection}' AND doc_id == '{escaped_doc_id}' AND parse_hash != '{escaped_new_parse_hash}'"
            predicates["__embeddings__"] = _append_user_filter_for_embeddings_if_needed(
                conn=conn,
                base_expr=other,
                user_id=user_id,
                is_admin=is_admin,
                model_tag=model_tag,
            )
            predicates["chunks"] = _append_user_filter_if_needed(
                conn=conn,
                table_name="chunks",
                base_expr=other,
                user_id=user_id,
                is_admin=is_admin,
            )
            predicates["parses"] = _append_user_filter_if_needed(
                conn=conn,
                table_name="parses",
                base_expr=other,
                user_id=user_id,
                is_admin=is_admin,
            )
    elif scope == "chunk":
        if old_parse_hash is None:
            pointer = get_main_pointer(collection, doc_id, "chunk")
            old_parse_hash = pointer["technical_id"] if pointer else None
        if old_parse_hash:
            # Build safe filter expression for old parse hash
            base_filters = {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": old_parse_hash,
            }
            base = build_lancedb_filter_expression(
                base_filters, user_id=user_id, is_admin=is_admin, skip_user_filter=True
            )
            predicates["__embeddings__"] = _append_user_filter_for_embeddings_if_needed(
                conn=conn,
                base_expr=base,
                user_id=user_id,
                is_admin=is_admin,
                model_tag=model_tag,
            )
        if new_parse_hash:
            # Note: For != operator, we need to manually construct the filter
            # as build_lancedb_filter_expression only supports == operator
            escaped_collection = escape_lancedb_string(collection)
            escaped_doc_id = escape_lancedb_string(doc_id)
            escaped_parse_hash = escape_lancedb_string(new_parse_hash)
            other = f"collection == '{escaped_collection}' AND doc_id == '{escaped_doc_id}' AND parse_hash != '{escaped_parse_hash}'"
            predicates["__embeddings__"] = _append_user_filter_for_embeddings_if_needed(
                conn=conn,
                base_expr=other,
                user_id=user_id,
                is_admin=is_admin,
                model_tag=model_tag,
            )
            predicates["chunks"] = _append_user_filter_if_needed(
                conn=conn,
                table_name="chunks",
                base_expr=other,
                user_id=user_id,
                is_admin=is_admin,
            )
    elif scope == "embeddings":
        # Build per-embeddings-table predicates so user_id filtering is based on
        # each embeddings table's own schema (forward/backward compatible).
        table_names = _get_table_names(conn)
        for t in table_names:
            if not t.startswith("embeddings_"):
                continue
            if model_tag is not None and t != f"embeddings_{model_tag}":
                continue
            predicates[t] = _build_document_filter(
                conn=conn,
                table_name=t,
                collection=collection,
                doc_id=doc_id,
                user_id=user_id,
                is_admin=is_admin,
            )
    elif scope == "pointers":
        filt = _build_document_filter(
            conn=conn,
            table_name="main_pointers",
            collection=collection,
            doc_id=doc_id,
            user_id=user_id,
            is_admin=is_admin,
        )
        predicates["main_pointers"] = filt
    else:
        raise CascadeCleanupError(f"Unsupported scope: {scope}")

    if preview_only and not confirm:
        planned = _plan_by_predicates(conn, predicates, model_tag=model_tag)
        if "__embeddings__" in planned:
            planned["embeddings"] = planned.pop("__embeddings__")
        elif scope == "embeddings":
            planned["embeddings"] = sum(
                int(v) for k, v in planned.items() if str(k).startswith("embeddings_")
            )
        return planned

    deleted = _delete_by_predicates(conn, predicates, model_tag=model_tag)
    if scope == "embeddings" and "__embeddings__" not in predicates:
        deleted["embeddings"] = sum(
            int(v) for k, v in deleted.items() if str(k).startswith("embeddings_")
        )
    return deleted


def cleanup_document_cascade(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Cascade delete all data for a document across all stages.

    Order: embeddings_* -> chunks -> parses -> main_pointers -> documents

    Args:
        collection: Collection name
        doc_id: Document ID
        model_tag: Optional model tag to limit embeddings deletion

    Returns:
        Deleted counts per scope
    """
    try:
        # Delegate to unified entry
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="document",
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup document cascade: {e}")


def cleanup_parse_cascade(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Clean up cascade when promoting a new parse version.

    This method:
    1. Deletes old parse's chunks and embeddings
    2. Deletes other parse candidates and their downstream data

    Args:
        collection: Collection name
        doc_id: Document ID
        old_parse_hash: Old main parse hash (optional)
        new_parse_hash: New main parse hash (optional)

    Returns:
        Dictionary with deletion counts

    Raises:
        CascadeCleanupError: If cleanup fails
    """
    try:
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="parse",
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup parse cascade: {e}")


def cleanup_chunk_cascade(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Clean up cascade when promoting a new chunk version.

    This method:
    1. Deletes old chunk's embeddings
    2. Deletes other chunk candidates

    Args:
        collection: Collection name
        doc_id: Document ID
        old_parse_hash: Old main parse hash (optional)
        new_parse_hash: New main parse hash (optional)

    Returns:
        Dictionary with deletion counts

    Raises:
        CascadeCleanupError: If cleanup fails
    """
    try:
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="chunk",
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup chunk cascade: {e}")


def cleanup_embed_cascade(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    old_technical_id: Optional[str] = None,
    new_technical_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Clean up cascade when promoting a new embeddings version.

    This method:
    1. Deletes other embeddings candidates (optionally filtered by model_tag)

    Args:
        collection: Collection name
        doc_id: Document ID
        model_tag: Model tag filter (optional)
        old_technical_id: Old main technical ID (optional)
        new_technical_id: New main technical ID (optional)

    Returns:
        Dictionary with deletion counts

    Raises:
        CascadeCleanupError: If cleanup fails
    """
    try:
        # Delegate to unified entry; old/new technical ids are not used in current schema
        return cleanup_cascade(
            collection=collection,
            doc_id=doc_id,
            scope="embeddings",
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup embed cascade: {e}")

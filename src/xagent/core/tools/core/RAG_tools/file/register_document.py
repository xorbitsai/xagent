"""Document registration functionality for RAG tools.

This module provides the core functionality for registering uploaded documents
into the LanceDB system. It handles document metadata extraction, validation,
and database insertion with proper error handling and idempotency.

The register_document function serves as the main entry point for document
registration in the RAG pipeline.
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ......providers.vector_store.lancedb import get_connection_from_env
from ..core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
    DocumentValidationError,
    HashComputationError,
)
from ..core.schemas import RegisterDocumentRequest, RegisterDocumentResponse
from ..LanceDB.schema_manager import ensure_documents_table
from ..utils import check_file_type, compute_file_hash
from ..utils.string_utils import (
    build_lancedb_filter_expression,
    generate_doc_id_from_filename,
)

logger = logging.getLogger(__name__)

# Public entry with explicit arguments (for LG/CLI/FastAPI). Returns plain dict.
# Internally constructs Pydantic request and delegates to _register_document.


def register_document(
    collection: str,
    source_path: str,
    file_type: Optional[str] = None,
    doc_id: Optional[str] = None,
    uploaded_at: Optional[str] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Register a document into the LanceDB system.

    Args:
        collection: LanceDB collection name (data isolation)
        source_path: Absolute path to the uploaded file
        file_type: Optional file type; auto-detected from extension if not provided
        doc_id: Optional document ID; UUIDv4 generated if not provided
        uploaded_at: Optional ISO8601 timestamp string (supports trailing 'Z');
            defaults to now if not provided or parse fails
        user_id: Optional user ID for multi-tenancy ownership

    Returns:
        A plain dict converted from RegisterDocumentResponse
    """
    uploaded_at_dt: Optional[datetime] = None
    if uploaded_at:
        try:
            if uploaded_at.endswith("Z"):
                uploaded_at_dt = datetime.fromisoformat(
                    uploaded_at.replace("Z", "+00:00")
                )
            else:
                uploaded_at_dt = datetime.fromisoformat(uploaded_at)
            if uploaded_at_dt.tzinfo is None:
                uploaded_at_dt = uploaded_at_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            uploaded_at_dt = None

    request = RegisterDocumentRequest(
        collection=collection,
        source_path=source_path,
        file_type=file_type,
        doc_id=doc_id,
        uploaded_at=uploaded_at_dt,
        user_id=user_id,
    )
    response = _register_document(request)
    result: Dict[str, Any] = response.model_dump()
    return result


# Private entry that accepts the Pydantic model and returns the Pydantic response.
def _register_document(request: RegisterDocumentRequest) -> RegisterDocumentResponse:
    """Register a document into the LanceDB system using a typed request object.

    This function takes an uploaded document and registers it into the specified
    LanceDB collection. It performs validation, computes content hash, and
    ensures idempotent registration based on document ID.

    Returns:
        RegisterDocumentResponse containing the document ID, creation status,
        and content hash.


    Raises:
        DocumentValidationError: If the document path is invalid or file is corrupted.
        HashComputationError: If content hash computation fails.
        DocumentRegistrationError: If database operation fails.
        DatabaseOperationError: If LanceDB connection or table operations fail.
    """
    collection = request.collection
    source_path = request.source_path
    file_type = request.file_type
    doc_id = request.doc_id
    uploaded_at = request.uploaded_at

    # Input validation
    if not collection:
        raise DocumentValidationError("Collection name cannot be empty")

    if not source_path or not Path(source_path).exists():
        raise DocumentValidationError(f"Source path does not exist: {source_path}")

    # Auto-detect file type if not provided
    if not file_type:
        try:
            file_type = check_file_type(source_path)
        except DocumentValidationError as e:
            raise DocumentValidationError(f"File type detection failed: {e}") from e

    # Generate document ID if not provided
    # Use filename-based ID for better user experience, with UUID fallback
    if not doc_id:
        try:
            doc_id = generate_doc_id_from_filename(source_path, include_extension=False)
        except Exception as e:
            # Fallback to UUID if filename-based generation fails
            logger.debug(
                "Filename-based ID generation failed (%s), falling back to UUID", e
            )
            doc_id = str(uuid.uuid4())

    # Set upload timestamp
    if not uploaded_at:
        uploaded_at = pd.Timestamp.now(tz="UTC")
    else:
        if uploaded_at.tzinfo is None:
            uploaded_at = uploaded_at.replace(tzinfo=timezone.utc)

    # Compute content hash using utility function
    try:
        content_hash = compute_file_hash(source_path)
    except Exception as e:
        raise HashComputationError(f"Failed to compute content hash: {e}") from e

    # LanceDB operations
    try:
        # Get LanceDB connection
        db = get_connection_from_env()

        # Ensure documents table exists
        ensure_documents_table(db)

        # Open the documents table
        table = db.open_table("documents")

        # Check if document already exists (for idempotency)
        query_filters = {
            "collection": collection,
            "doc_id": doc_id,
        }
        filter_expr = build_lancedb_filter_expression(query_filters)

        exists = table.count_rows(filter_expr) > 0

        # Prepare document record
        doc_record = {
            "collection": collection,
            "doc_id": doc_id,
            "source_path": source_path,
            "file_type": file_type,
            "content_hash": content_hash,
            # Store timestamp object directly, let Arrow handle precision conversion
            "uploaded_at": uploaded_at,
            "title": None,  # Optional field, can be filled later
            "language": None,  # Optional field, can be filled later
            "user_id": request.user_id,  # Add user_id for multi-tenancy
        }

        # Use merge_insert for efficient upsert operation
        table.merge_insert(
            ["collection", "doc_id"]
        ).when_matched_update_all().when_not_matched_insert_all().execute([doc_record])

        created = not exists

    except ConfigurationError:
        raise
    except Exception as e:
        raise DatabaseOperationError(
            f"Failed to register document in database: {e}"
        ) from e

    return RegisterDocumentResponse(
        doc_id=doc_id,
        created=created,
        content_hash=content_hash,
    )


def get_document(db_dir: str, collection: str, doc_id: str) -> Optional[Any]:
    """Retrieve a document record from LanceDB.


    Args:
        db_dir: LanceDB directory path
        collection: Collection name to filter by (only returns documents from this collection)
        doc_id: Document ID to retrieve

    Returns:
        Document record dict if found, None otherwise

    Raises:
        DatabaseOperationError: If database operation fails
    """
    try:
        db = get_connection_from_env()
        ensure_documents_table(db)
        table = db.open_table("documents")

        filter_expr = build_lancedb_filter_expression(
            {"collection": collection, "doc_id": doc_id}
        )
        if table.count_rows(filter_expr) == 0:
            return None

        # Convert to dict and handle datetime
        record = table.search().where(filter_expr).to_pandas().iloc[0].to_dict()
        return record

    except Exception as e:
        raise DatabaseOperationError(f"Failed to retrieve document: {e}") from e


def list_documents(
    db_dir: str, collection: str, limit: int = 100
) -> list[Dict[str, Any]]:
    """List documents in the collection.

    Args:
        db_dir: LanceDB directory path
        collection: Collection name to filter by (only documents in this KB are returned)
        limit: Maximum number of documents to return

    Returns:
        List of document records for the given collection

    Raises:
        DatabaseOperationError: If database operation fails
    """
    try:
        db = get_connection_from_env()
        ensure_documents_table(db)
        table = db.open_table("documents")

        filter_expr = build_lancedb_filter_expression({"collection": collection})
        results = table.search().where(filter_expr).limit(limit).to_pandas()
        return list(results.to_dict("records"))

    except Exception as e:
        raise DatabaseOperationError(f"Failed to list documents: {e}") from e

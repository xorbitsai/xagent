"""Utilities package for RAG tools.

This package contains common utility functions used across RAG tool modules.
"""

from .embedding_utils import (
    normalize_raw_embedding_to_vectors,
    normalize_single_embedding,
)
from .file_utils import check_file_type, validate_file_path
from .hash_utils import compute_content_hash, compute_file_hash, validate_hash_format
from .lancedb_query_utils import query_to_list
from .metadata_utils import deserialize_metadata, serialize_metadata
from .string_utils import (
    build_lancedb_filter_expression,
    escape_lancedb_string,
    generate_doc_id_from_filename,
    sanitize_for_doc_id,
)
from .validation_utils import validate_and_convert_user_id

__all__ = [
    # Embedding utilities
    "normalize_raw_embedding_to_vectors",
    "normalize_single_embedding",
    # File utilities
    "check_file_type",
    "validate_file_path",
    # Hash utilities
    "compute_content_hash",
    "compute_file_hash",
    "validate_hash_format",
    # LanceDB query utilities
    "query_to_list",
    # Metadata utilities
    "serialize_metadata",
    "deserialize_metadata",
    # String utilities
    "escape_lancedb_string",
    "build_lancedb_filter_expression",
    "sanitize_for_doc_id",
    "generate_doc_id_from_filename",
    # Validation utilities
    "validate_and_convert_user_id",
]

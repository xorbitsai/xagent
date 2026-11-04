"""Version management module for RAG tools.

Provide functions for listing candidates, promoting versions, and
maintaining main pointers with cascade cleanup.
"""

from .cascade_cleaner import (
    cascade_delete,
    cleanup_cascade,
    cleanup_chunk_cascade,
    cleanup_document_cascade,
    cleanup_embed_cascade,
    cleanup_parse_cascade,
)
from .list_candidates import list_candidates
from .main_pointer_manager import (
    delete_main_pointer,
    get_main_pointer,
    list_main_pointers,
    set_main_pointer,
)
from .promote_version_main import promote_version_main

__all__ = [
    "list_candidates",
    "promote_version_main",
    "get_main_pointer",
    "set_main_pointer",
    "list_main_pointers",
    "delete_main_pointer",
    "cleanup_parse_cascade",
    "cascade_delete",
    "cleanup_chunk_cascade",
    "cleanup_embed_cascade",
    "cleanup_document_cascade",
    "cleanup_cascade",
]

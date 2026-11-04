"""
Retrieval module for RAG tools.

This module provides search functionality for retrieving relevant documents
and chunks from the vector database, including:
- Dense vector search (ANN)
- Sparse full-text search (FTS)
- Hybrid search with fusion strategies
"""

from .search_dense import search_dense
from .search_hybrid import search_hybrid
from .search_sparse import search_sparse

__all__ = [
    "search_dense",
    "search_sparse",
    "search_hybrid",
]

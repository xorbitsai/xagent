"""
Vector store providers module.

This module provides various vector storage backends that implement
the standard VectorStore interface.
"""

import importlib.util

from .base import VectorStore
from .lancedb import (
    LanceDBConnectionManager,
    LanceDBVectorStore,
)

# ChromaVectorStore is optional (requires chromadb)
_chroma_available = importlib.util.find_spec("chromadb") is not None

if _chroma_available:
    from .chroma import ChromaVectorStore

    __all__ = [
        "VectorStore",
        "LanceDBVectorStore",
        "LanceDBConnectionManager",
        "ChromaVectorStore",
    ]
else:
    __all__ = [
        "VectorStore",
        "LanceDBVectorStore",
        "LanceDBConnectionManager",
    ]

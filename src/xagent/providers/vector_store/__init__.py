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
_milvus_available = importlib.util.find_spec("pymilvus") is not None

if _chroma_available:
    from .chroma import ChromaVectorStore as ChromaVectorStore

if _milvus_available:
    from .milvus import MilvusConnectionManager as MilvusConnectionManager
    from .milvus import MilvusVectorStore as MilvusVectorStore

__all__ = [
    "VectorStore",
    "LanceDBVectorStore",
    "LanceDBConnectionManager",
]

if _chroma_available:
    __all__.append("ChromaVectorStore")

if _milvus_available:
    __all__.extend(["MilvusVectorStore", "MilvusConnectionManager"])

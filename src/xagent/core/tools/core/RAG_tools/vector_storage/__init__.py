"""Vector storage module for RAG tools.

This module provides pure vector data management functions:
- Reading chunks from database for embedding computation
- Writing embedding vectors to database with idempotency
- Vector validation and consistency checking

## Architecture

This module handles only data management and does not perform any text-to-vector
conversion. The actual embedding is handled by AgentOS embedding nodes in workflows.

Index management is now handled by the storage abstraction layer in
`storage.contracts.VectorIndexStore` and implemented in `storage.lancedb_stores`.

```
AgentOS Workflow:
1. read_chunks_for_embedding() → Get chunks needing vectors
2. AgentOS embedding node → Convert text to vectors
3. write_vectors_to_db() → Store vectors with idempotency
```

## Core Functions

- `read_chunks_for_embedding()`: Read chunks needing embedding from database
- `write_vectors_to_db()`: Write vectors with staleness cleanup and indexing
- `validate_query_vector()`: Validate vector format for search operations
- `commit_vectors_to_db()`: Make written vectors searchable (LanceDB: no-op)

## Vector Management

- Automatic dimension consistency checking
- Stale data cleanup when chunk_hash changes
- Index creation handled by storage abstraction layer
- Multi-model support with separate tables per model
"""

from .vector_manager import (
    commit_vectors_to_db,
    read_chunks_for_embedding,
    validate_query_vector,
    write_vectors_to_db,
)

__all__ = [
    "commit_vectors_to_db",
    "read_chunks_for_embedding",
    "write_vectors_to_db",
    "validate_query_vector",
]

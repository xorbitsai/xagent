from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Union
from uuid import uuid4

import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore

from ...providers.vector_store.lancedb import (
    LanceDBConnectionManager,
    LanceDBVectorStore,
)
from ..model.embedding import BaseEmbedding, DashScopeEmbedding
from ..model.embedding.adapter import create_embedding_adapter
from ..model.model import EmbeddingModelConfig
from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from .base import (
    MEMORY_BACKEND_UNAVAILABLE_REASON,
    MemoryBackendUnavailableError,
    MemoryStore,
)
from .core import MemoryNote, MemoryResponse
from .schema_migration import (
    MemoryMismatchKind,
    classify_memory_schema_mismatch,
    migrate_table_swap,
)
from .scope_columns import (
    SCOPE_DIMS_COLUMN,
    USER_ID_COLUMN,
    build_scope_where,
    coerce_user_id,
    derive_scope_columns,
    encode_scope_dims,
    scope_dim_where_term,
)

logger = logging.getLogger(__name__)

_VECTOR_SPACE_METADATA_KEY = b"xagent.memory.vector_space"


class LanceDBMemoryStore(MemoryStore):
    """LanceDB-based memory store implementation with vector search capabilities."""

    _embedding_model: Optional[BaseEmbedding]

    def __init__(
        self,
        db_dir: str,
        collection_name: str = "memories",
        embedding_model: Optional[Union[BaseEmbedding, EmbeddingModelConfig]] = None,
        similarity_threshold: float = 1.0,
        vector_space_identity: Optional[dict[str, Any]] = None,
        allow_schema_migration: bool = True,
        **embedding_kwargs: Any,
    ):
        """
        Initialize LanceDB memory store.

        Args:
            db_dir: Database directory path
            collection_name: Collection name for storing memories
            embedding_model: Optional BaseEmbedding instance or EmbeddingModel config
            similarity_threshold: Cosine distance threshold for vector search (lower = more strict)
            **embedding_kwargs: Additional arguments for embedding model
        """
        self._collection_name = collection_name
        self._vector_space_identity = vector_space_identity
        self._allow_schema_migration = allow_schema_migration

        # Handle different types of embedding_model input
        if embedding_model is None:
            # Try to create a default embedding model only if embedding_kwargs are provided
            if embedding_kwargs:
                try:
                    self._embedding_model = DashScopeEmbedding(**embedding_kwargs)
                except Exception:
                    # If embedding model creation fails, set to None (will use fallback)
                    self._embedding_model = None
                    logger.warning(
                        "Failed to create embedding model, will use fallback text search"
                    )
            else:
                self._embedding_model = None
                logger.info(
                    "No embedding model provided, will use fallback text search"
                )
        elif isinstance(embedding_model, BaseEmbedding):
            self._embedding_model = embedding_model
        elif isinstance(embedding_model, EmbeddingModelConfig):
            self._embedding_model = create_embedding_adapter(embedding_model)
        else:
            raise ValueError(
                f"Unsupported embedding model type: {type(embedding_model)}"
            )
        self._similarity_threshold = similarity_threshold
        self._conn_manager = LanceDBConnectionManager()
        self._vector_store = LanceDBVectorStore(
            db_dir,
            collection_name,
            connection_manager=self._conn_manager,
            initial_data=self._initial_table_data(),
        )
        self._ensure_table_schema()

    def _configured_embedding_dimension(self) -> Optional[int]:
        if self._vector_space_identity is not None:
            dimension = self._vector_space_identity.get("dimension")
            return int(dimension) if dimension else None
        if not self._embedding_model:
            return None
        try:
            dimension = self._embedding_model.get_dimension()
        except Exception:
            return None
        return int(dimension) if dimension else None

    def _initial_table_data(self) -> Any:
        base: dict[str, Any] = {
            "id": ["sample"],
            "text": ["sample"],
            "metadata": ["{}"],
            USER_ID_COLUMN: pa.array([0], pa.int64()),
            SCOPE_DIMS_COLUMN: pa.array([["sample"]], pa.list_(pa.string())),
        }
        dimension = self._configured_embedding_dimension()
        if dimension is not None:
            base["vector"] = pa.array(
                [[0.0] * dimension], pa.list_(pa.float32(), dimension)
            )
        data = pa.table(base)
        if self._vector_space_identity is not None:
            metadata = dict(data.schema.metadata or {})
            metadata[_VECTOR_SPACE_METADATA_KEY] = json.dumps(
                self._vector_space_identity, sort_keys=True, separators=(",", ":")
            ).encode()
            data = data.replace_schema_metadata(metadata)
        return data

    def _ensure_table_schema(self) -> None:
        """Ensure the table has the correct schema for memory storage.

        If the table is missing a required column, migrate it in place
        (preserving all rows) instead of dropping and recreating it. This path
        runs on every store construction, so a wipe here would destroy data with
        no write in flight. On any migration failure the original table is left
        intact and the error propagates (out of ``__init__``); we never fall back
        to a wipe. Note the migration branch may perform a batched re-embed when
        a table is both missing a base column and vector-mismatched.
        """
        conn = self._vector_store.get_raw_connection()

        if not self._allow_schema_migration:
            # Request-time manager stores use read-only admission. Existing
            # shared tables are never scanned, rebuilt, or re-embedded here.
            return

        # Determine whether the table already exists and read its columns.
        table = None
        try:
            table = conn.open_table(self._collection_name)
            column_names = set(table.schema.names)
        except Exception:
            # Table doesn't exist yet, create it with the basic schema.
            logger.info(f"Creating table {self._collection_name} with basic schema")
            self._create_empty_table()
            return
        finally:
            _safe_close_table(table)

        # Table exists. Init's trigger is a missing required non-vector column;
        # a vector-dimension mismatch is detected and migrated lazily on the
        # add() path instead. Route the resolution through the shared classifier
        # and transform-then-swap primitive so we migrate rather than wipe.
        if not {"id", "text", "metadata"} <= column_names:
            logger.warning(
                f"Table {self._collection_name} has incompatible schema, "
                "migrating in place"
            )
            self._resolve_schema_mismatch(
                conn, self._current_embedding_dim(), raise_when_compatible=False
            )

        # #822: promote user_id + scope_dims to real columns so scope filters
        # can be pushed into a `where` prefilter (slice 001). Idempotent and
        # data-preserving; runs after the base-schema resolution above so it sees
        # a table that already has id/text/metadata.
        self._ensure_scope_columns(conn)

    def _create_empty_table(self) -> None:
        """Create an empty table with the correct schema."""
        conn = self._vector_store.get_raw_connection()

        # Base row carries the derived scope columns (#822) so their types are
        # fixed at creation: user_id -> int64, scope_dims -> list<string>.
        # Concrete values are only for schema inference; the sample row is
        # deleted below.
        base_sample = {
            "id": "sample",
            "text": "sample",
            "metadata": "{}",
            USER_ID_COLUMN: 0,
            SCOPE_DIMS_COLUMN: ["sample"],
        }

        # Check if we have an embedding model
        if self._embedding_model:
            # Create table with vector support
            try:
                # Generate a sample embedding to get dimension
                sample_embedding = self._get_embedding("sample")
                if sample_embedding:
                    # Create sample data with vector
                    sample_data = [{**base_sample, "vector": sample_embedding}]
                else:
                    # Fallback to non-vector schema
                    sample_data = [base_sample]
            except Exception:
                # If embedding fails, use non-vector schema
                sample_data = [base_sample]
        else:
            # No embedding model, create without vector column
            sample_data = [base_sample]

        # Create table with appropriate schema
        table = conn.create_table(self._collection_name, data=sample_data)
        # Remove sample data
        table.delete("id = 'sample'")

    def _table_schema(self) -> Any:
        table = self._vector_store.get_raw_connection().open_table(
            self._collection_name
        )
        try:
            return table.schema
        finally:
            _safe_close_table(table)

    @staticmethod
    def _vector_dimension(schema: Any) -> Optional[int]:
        if "vector" not in schema.names:
            return None
        vector_type = schema.field("vector").type
        return (
            int(vector_type.list_size)
            if pa.types.is_fixed_size_list(vector_type)
            else None
        )

    def _stored_vector_space_identity(self, schema: Any) -> Optional[dict[str, Any]]:
        encoded = (schema.metadata or {}).get(_VECTOR_SPACE_METADATA_KEY)
        if encoded is None:
            return None
        try:
            value = json.loads(encoded.decode())
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def _has_compatible_vector_space(self) -> bool:
        try:
            schema = self._table_schema()
        except Exception:
            return False
        dimension = self._configured_embedding_dimension()
        if dimension is None or self._vector_dimension(schema) != dimension:
            return False
        if self._vector_space_identity is None:
            # Preserve direct/legacy callers that predate persisted identity.
            return self._embedding_model is not None
        return self._stored_vector_space_identity(schema) == self._vector_space_identity

    def ensure_persistence(self) -> None:
        """Verify the persistent table is readable without mutating it."""

        try:
            schema = self._table_schema()
            if not {
                "id",
                "text",
                "metadata",
                USER_ID_COLUMN,
                SCOPE_DIMS_COLUMN,
            } <= set(schema.names):
                raise RuntimeError("memory table requires an offline schema migration")
        except Exception as exc:
            raise MemoryBackendUnavailableError(
                MEMORY_BACKEND_UNAVAILABLE_REASON
            ) from exc

    def ensure_required_vector_search(self) -> None:
        """Read-only strict admission; it performs no embedding call."""

        self.ensure_persistence()
        if (
            self._embedding_model is None
            or self._vector_space_identity is None
            or not self._has_compatible_vector_space()
        ):
            raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text using the configured embedding model."""
        if not self._embedding_model or not text.strip():
            return None
        if (
            self._vector_space_identity is not None
            and not self._has_compatible_vector_space()
        ):
            logger.warning(
                "Configured embedding identity does not match the persisted "
                "vector space; using text-only fallback"
            )
            return None

        try:
            result = self._embedding_model.encode(text)
            # encode should return list[float] for single text input
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], list):
                    # Got list[list[float]], return the first embedding
                    return result[0]
                elif len(result) > 0 and isinstance(result[0], (int, float)):
                    # Got list[float], return as is
                    return result  # type: ignore[return-value]
            logger.warning(f"Unexpected embedding result format: {type(result)}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate embedding for text '{text[:50]}...': {e}")
            return None

    def _get_required_embedding(self, text: str) -> list[float]:
        self.ensure_required_vector_search()
        embedding = self._get_embedding(text)
        expected_dimension = self._configured_embedding_dimension()
        if (
            embedding is None
            or expected_dimension is None
            or len(embedding) != expected_dimension
        ):
            raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)
        return [float(value) for value in embedding]

    def _current_embedding_dim(self) -> Optional[int]:
        """Return the vector dimension the store currently produces, or None.

        None means no embedding model is available and the store operates in
        vector-less (text-search) mode.
        """
        configured = self._configured_embedding_dimension()
        if configured is not None:
            return configured
        if not self._embedding_model:
            return None
        try:
            dim = self._embedding_model.get_dimension()
            if dim:
                return int(dim)
        except Exception:
            pass
        sample = self._get_embedding("sample")
        return len(sample) if sample else None

    def _embed_texts_batch(
        self, texts: list[str], target_dim: int
    ) -> list[list[float]]:
        """Re-embed all texts in a single batched encode call (all-or-nothing).

        Raises if no model is available, the batch shape is unexpected, or any
        row's embedding is missing or has the wrong dimension. The caller relies
        on this raising so the migration aborts with the original table intact
        (never a partially-vectorized table).
        """
        if not self._embedding_model:
            raise RuntimeError(
                "Cannot rebuild vector column without an embedding model"
            )
        if not texts:
            return []
        result = self._embedding_model.encode(texts)
        if not isinstance(result, list) or len(result) != len(texts):
            raise RuntimeError(
                f"Embedding batch returned unexpected shape for {len(texts)} rows"
            )
        vectors: list[list[float]] = []
        for index, vector in enumerate(result):
            if not isinstance(vector, list) or len(vector) != target_dim:
                raise RuntimeError(
                    f"Re-embedding row {index} failed or produced the wrong "
                    f"dimension (expected {target_dim})"
                )
            vectors.append([float(value) for value in vector])
        return vectors

    def _build_migrated_table(self, existing: Any, target_dim: Optional[int]) -> Any:
        """Transform for the migration primitive: rebuild rows at the target schema.

        Preserves every existing row's id/text/metadata. When ``target_dim`` is
        set, re-embeds all rows into a fresh fixed-width vector column; when it
        is None, produces a vector-less table (text-only search).
        """
        row_count = existing.num_rows
        names = set(existing.schema.names)

        def _string_column(name: str) -> Any:
            # Stay in the Arrow format and let PyArrow cast natively instead of
            # materializing Python lists per element.
            if name in names:
                return existing.column(name).cast(pa.string())
            return pa.array([None] * row_count, pa.string())

        columns: dict[str, Any] = {
            "id": _string_column("id"),
            "text": _string_column("text"),
            "metadata": _string_column("metadata"),
        }

        if target_dim is not None:
            # The batched embedding interface needs Python strings; reuse the
            # already-cast text column so it is only materialized once.
            texts = [text or "" for text in columns["text"].to_pylist()]
            vectors = self._embed_texts_batch(texts, target_dim)
            columns["vector"] = pa.array(vectors, pa.list_(pa.float32(), target_dim))

        # #822: keep the derived scope columns present through a vector rebuild so
        # every migration path produces the full schema.
        user_ids, scope_dims = self._derive_scope_arrays(columns["metadata"])
        columns[USER_ID_COLUMN] = user_ids
        columns[SCOPE_DIMS_COLUMN] = scope_dims

        return pa.table(columns)

    def _derive_scope_arrays(self, metadata_column: Any) -> tuple[Any, Any]:
        """Derive the (user_id, scope_dims) Arrow columns from a metadata column.

        Shared by every migration transform so the write path and all rebuild
        paths encode the columns identically.
        """
        user_ids: list[Optional[int]] = []
        scope_dims: list[list[str]] = []
        for metadata_json in metadata_column.to_pylist():
            user_id, dims = derive_scope_columns(metadata_json)
            user_ids.append(user_id)
            scope_dims.append(dims)
        return (
            pa.array(user_ids, pa.int64()),
            pa.array(scope_dims, pa.list_(pa.string())),
        )

    def _add_scope_columns(self, existing: Any) -> Any:
        """Migration transform: add derived scope columns, preserve everything else.

        Unlike the vector rebuild, this preserves the existing ``vector`` column
        as-is (no re-embedding) — it only projects ``user_id`` / ``scope_dims``
        out of each row's metadata JSON.
        """
        columns: dict[str, Any] = {
            name: existing.column(name) for name in existing.schema.names
        }
        if "metadata" in columns:
            metadata_column = columns["metadata"].cast(pa.string())
        else:
            metadata_column = pa.array([None] * existing.num_rows, pa.string())
        user_ids, scope_dims = self._derive_scope_arrays(metadata_column)
        columns[USER_ID_COLUMN] = user_ids
        columns[SCOPE_DIMS_COLUMN] = scope_dims
        return pa.table(columns)

    def _ensure_scope_columns(self, conn: Any) -> None:
        """Promote user_id + scope_dims to real columns on an existing table (#822).

        Idempotent: does nothing when both columns already exist (fresh tables are
        created with them). Otherwise rebuilds the table via transform-then-swap,
        back-filling both columns from each row's metadata JSON and preserving all
        other columns (including ``vector``). On failure the original table is left
        intact and the error propagates — this never drops data.
        """
        table = conn.open_table(self._collection_name)
        try:
            names = set(table.schema.names)
        finally:
            _safe_close_table(table)

        if {USER_ID_COLUMN, SCOPE_DIMS_COLUMN} <= names:
            return

        logger.info(
            "Promoting user_id/scope_dims to real columns on table '%s'",
            self._collection_name,
        )
        migrate_table_swap(conn, self._collection_name, self._add_scope_columns)

    def _backfill_missing_columns(self, conn: Any, columns: tuple[str, ...]) -> None:
        """Add missing non-vector columns in place (no data loss, no rebuild)."""
        table = conn.open_table(self._collection_name)
        try:
            for column in columns:
                # All non-vector memory columns are strings.
                table.add_columns({column: "cast(null as string)"})
        finally:
            _safe_close_table(table)

    def _resolve_schema_mismatch(
        self, conn: Any, expected_dim: Optional[int], *, raise_when_compatible: bool
    ) -> None:
        """Classify and safely resolve a schema mismatch (shared by add/init).

        Missing non-vector columns are backfilled in place; a vector
        dimension/presence change rebuilds the table via transform-then-swap.
        On any failure the original table is left intact and the error
        propagates; no path drops or empties the table.

        When the schema is classified compatible, ``raise_when_compatible``
        controls behavior: the ``add()`` path passes ``True`` (its insert failed,
        so a compatible schema means an unexpected error to surface rather than
        silently drop); the init path passes ``False`` (nothing to migrate).
        """
        table = conn.open_table(self._collection_name)
        try:
            schema = table.schema
        finally:
            _safe_close_table(table)

        mismatch = classify_memory_schema_mismatch(schema, expected_dim)

        if mismatch.kind is MemoryMismatchKind.MISSING_NON_VECTOR_COLUMN:
            self._backfill_missing_columns(conn, mismatch.missing_columns)
        elif mismatch.kind is MemoryMismatchKind.VECTOR_REBUILD:
            target_dim = expected_dim
            migrate_table_swap(
                conn,
                self._collection_name,
                lambda existing: self._build_migrated_table(existing, target_dim),
            )
        elif raise_when_compatible:
            # add() failed but the schema looks compatible: do NOT drop the
            # table. Surface the original failure to the caller.
            raise RuntimeError(
                "add() failed but no resolvable schema mismatch was detected"
            )

    def _migrate_schema_mismatch(self, conn: Any, record: dict[str, Any]) -> None:
        """Resolve the schema mismatch that made an ``add()`` insert fail."""
        # The dimension we are trying to store now determines the target schema.
        if record.get("vector"):
            expected_dim: Optional[int] = len(record["vector"])
        elif self._embedding_model:
            expected_dim = self._current_embedding_dim()
        else:
            expected_dim = None

        self._resolve_schema_mismatch(conn, expected_dim, raise_when_compatible=True)

    def _insert_record(self, table: Any, record: dict[str, Any]) -> None:
        """Insert a record, adapting it to the (possibly migrated) table schema."""
        schema_names = set(table.schema.names)
        if "vector" in record and "vector" not in schema_names:
            record = {k: v for k, v in record.items() if k != "vector"}
        table.add([record])

    def _memory_note_to_dict(
        self, note: MemoryNote, *, require_vector: bool = False
    ) -> dict[str, Any]:
        """Convert MemoryNote to dictionary for storage."""
        # Get embedding for the content
        content_text = (
            note.content.decode() if isinstance(note.content, bytes) else note.content
        )
        embedding = (
            self._get_required_embedding(content_text)
            if require_vector
            else self._get_embedding(content_text)
        )

        # Prepare metadata
        metadata = {
            "content": note.content,
            "keywords": note.keywords,
            "tags": note.tags,
            "category": note.category,
            "timestamp": note.timestamp.isoformat(),
            "mime_type": note.mime_type,
            **note.metadata,
        }

        return {
            "id": note.id,
            "vector": embedding,
            "text": note.content,
            "metadata": json.dumps(metadata, ensure_ascii=False),
            # #822: derived filter projections; the metadata JSON stays
            # authoritative. Computed from the scope stamps the isolation layer
            # writes onto note.metadata (user_id + execution_scope_* keys).
            USER_ID_COLUMN: coerce_user_id(note.metadata.get("user_id")),
            SCOPE_DIMS_COLUMN: encode_scope_dims(note.metadata),
        }

    def _dict_to_memory_note(self, data: dict[str, Any]) -> MemoryNote:
        """Convert dictionary from storage to MemoryNote."""
        try:
            metadata = json.loads(data.get("metadata", "{}"))
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        note_kwargs: dict[str, Any] = {
            "id": data.get("id"),
            "content": metadata.pop("content", data.get("text", "")),
            "keywords": metadata.pop("keywords", []),
            "tags": metadata.pop("tags", []),
            "category": metadata.pop("category", "general"),
            "mime_type": metadata.pop("mime_type", "text/plain"),
            "metadata": metadata,
        }
        # #847: legacy rows may lack `timestamp`. MemoryNote.timestamp is a
        # non-optional datetime with a default_factory, and pydantic only
        # applies the factory when the field is omitted — an explicit None
        # raises ValidationError.
        timestamp = metadata.pop("timestamp", None)
        if timestamp is not None:
            note_kwargs["timestamp"] = timestamp

        return MemoryNote(**note_kwargs)

    # Filter matching (_matches_filters and friends) is inherited from the
    # MemoryStore base so stores cannot drift apart on filter semantics (#916).
    #
    # On the vector path the dispatch receives the residual dict left by
    # ``build_scope_where``: ``SCOPE_EXCLUSIVE_FILTER_KEY`` and the
    # user_id/scope-dimension entries of the nested ``metadata`` filter were
    # already pushed into the ``where`` prefilter, so those checks simply do
    # not trigger there. The text fallback passes the full filters and relies
    # on all the base checks.

    def add(self, note: MemoryNote) -> MemoryResponse:
        """Add a memory note to the store."""
        return self._add(note, require_vector=False)

    def add_required_vector(self, note: MemoryNote) -> MemoryResponse:
        """Add only after a compatible embedding has been produced."""

        return self._add(note, require_vector=True)

    def _add(self, note: MemoryNote, *, require_vector: bool) -> MemoryResponse:
        try:
            # Generate ID if not provided
            if not note.id:
                note.id = str(uuid4())

            # Convert to storage format
            data = self._memory_note_to_dict(note, require_vector=require_vector)

            # Add to vector store - use a consistent approach
            conn = self._vector_store.get_raw_connection()
            table = None
            try:
                table = conn.open_table(self._collection_name)

                # Prepare record for insertion
                record = {
                    "id": data["id"],
                    "text": data["text"],
                    "metadata": data["metadata"],
                    USER_ID_COLUMN: data[USER_ID_COLUMN],
                    SCOPE_DIMS_COLUMN: data[SCOPE_DIMS_COLUMN],
                }

                # Add vector if available
                if data["vector"]:
                    record["vector"] = data["vector"]

                # Try to add the record. On a schema mismatch, migrate the
                # existing table in place (preserving all rows) instead of
                # dropping and recreating it.
                try:
                    table.add([record])
                except Exception as add_error:
                    if require_vector or not self._allow_schema_migration:
                        raise add_error
                    logger.warning(
                        f"add() failed on possible schema mismatch: {add_error}; "
                        "attempting safe in-place migration"
                    )
                    _safe_close_table(table)
                    table = None
                    # Migrate safely; on any failure the original table is left
                    # intact and we surface an error WITHOUT dropping data.
                    try:
                        self._migrate_schema_mismatch(conn, record)
                    except Exception as migrate_error:
                        logger.error(
                            "Safe schema migration failed; table left intact: %s",
                            migrate_error,
                        )
                        return MemoryResponse(
                            success=False,
                            error=f"Failed to add memory: {migrate_error}",
                            memory_id=data["id"],
                        )
                    # Retry the insert against the migrated schema.
                    table = conn.open_table(self._collection_name)
                    self._insert_record(table, record)
            finally:
                _safe_close_table(table)

            return MemoryResponse(success=True, memory_id=data["id"])

        except MemoryBackendUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Failed to add memory note {note.id}: {e}")
            if require_vector:
                raise MemoryBackendUnavailableError(
                    MEMORY_BACKEND_UNAVAILABLE_REASON
                ) from e
            return MemoryResponse(
                success=False,
                error=f"Failed to add memory: {str(e)}",
                memory_id=note.id,
            )

    def get(self, note_id: str) -> MemoryResponse:
        """Retrieve a memory note by its ID."""
        table = None
        try:
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )

            # Search by ID
            results = table.search().where(f"id = '{note_id}'").to_pandas()

            if len(results) == 0:
                return MemoryResponse(
                    success=False,
                    error="Memory not found",
                    memory_id=note_id,
                )

            # Convert to MemoryNote
            data = results.iloc[0].to_dict()
            note = self._dict_to_memory_note(data)

            return MemoryResponse(
                success=True,
                memory_id=note_id,
                content=note,
            )

        except Exception as e:
            logger.error(f"Failed to get memory note {note_id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to get memory: {str(e)}",
                memory_id=note_id,
            )
        finally:
            _safe_close_table(table)

    def update(self, note: MemoryNote) -> MemoryResponse:
        """Update an existing memory note."""
        try:
            # Check if memory exists
            get_response = self.get(note.id)
            if not get_response.success:
                return MemoryResponse(
                    success=False,
                    error="Memory not found",
                    memory_id=note.id,
                )

            # Delete old record
            self.delete(note.id)

            # Add updated record
            return self.add(note)

        except Exception as e:
            logger.error(f"Failed to update memory note {note.id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to update memory: {str(e)}",
                memory_id=note.id,
            )

    def update_required_vector(self, note: MemoryNote) -> MemoryResponse:
        """Atomically replace a row after obtaining a valid embedding."""

        get_response = self.get(note.id)
        if not get_response.success:
            if get_response.error != "Memory not found":
                raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)
            return MemoryResponse(
                success=False, error="Memory not found", memory_id=note.id
            )

        try:
            data = self._memory_note_to_dict(note, require_vector=True)
            record = {
                "id": data["id"],
                "text": data["text"],
                "metadata": data["metadata"],
                "vector": data["vector"],
                USER_ID_COLUMN: data[USER_ID_COLUMN],
                SCOPE_DIMS_COLUMN: data[SCOPE_DIMS_COLUMN],
            }
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )
            try:
                (table.merge_insert("id").when_matched_update_all().execute([record]))
            finally:
                _safe_close_table(table)
            return MemoryResponse(success=True, memory_id=note.id)
        except MemoryBackendUnavailableError:
            raise
        except Exception as exc:
            raise MemoryBackendUnavailableError(
                MEMORY_BACKEND_UNAVAILABLE_REASON
            ) from exc

    def delete(self, note_id: str) -> MemoryResponse:
        """Delete a memory note by its ID."""
        try:
            success = self._vector_store.delete_vectors([note_id])

            if success:
                return MemoryResponse(success=True, memory_id=note_id)
            else:
                return MemoryResponse(
                    success=False,
                    error="Failed to delete memory",
                    memory_id=note_id,
                )

        except Exception as e:
            logger.error(f"Failed to delete memory note {note_id}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to delete memory: {str(e)}",
                memory_id=note_id,
            )

    def delete_by_scope_dimension(self, dim_key: str, value: Any) -> MemoryResponse:
        """Bulk-delete every note stamped with dimension ``dim_key=value``.

        Pushes a single ``array_contains(scope_dims, 'key=value')`` predicate
        into LanceDB instead of the base class's list-and-delete walk, so the
        whole matching set goes in one table operation. Exact per-element
        equality (see scope_columns), so similar values never collide.

        ``deleted_count`` is best-effort under concurrent writers: LanceDB's
        ``delete()`` reports only the new table version, not a row count, so
        the count comes from a ``count_rows`` on the same predicate just
        before the delete. The operation itself stays idempotent either way.
        """
        where = scope_dim_where_term(dim_key, value)
        table = None
        try:
            # The table always exists after construction (the vector store
            # creates it in __init__), so open directly like get()/search().
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )
            count = table.count_rows(where)
            if count:
                table.delete(where)
            return MemoryResponse(success=True, metadata={"deleted_count": count})
        except Exception as e:
            logger.error(f"Failed to delete memories by scope dimension {dim_key}: {e}")
            return MemoryResponse(
                success=False,
                error=f"Failed to delete memories by scope dimension: {str(e)}",
                metadata={"deleted_count": 0},
            )
        finally:
            _safe_close_table(table)

    def list_scope_dimension_values(self, dim_key: str) -> set[str]:
        """Distinct stamped values for one scope dimension, store-wide.

        Projects only the ``scope_dims`` column (no vectors, no metadata JSON)
        so the scan stays cheap on large tables. Raises on backend failure —
        see the base class contract.
        """
        prefix = f"{dim_key}="
        table = None
        try:
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )
            arrow_table = (
                table.search().select([SCOPE_DIMS_COLUMN]).limit(None).to_arrow()
            )
            # Filter in Arrow: flatten the list<string> column (null rows drop
            # out), keep only this dimension's elements, then dedupe — only the
            # distinct values ever cross into Python.
            flat = pc.list_flatten(arrow_table[SCOPE_DIMS_COLUMN])
            matched = flat.filter(pc.starts_with(flat, pattern=prefix))
            return {element[len(prefix) :] for element in matched.unique().to_pylist()}
        finally:
            _safe_close_table(table)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[MemoryNote]:
        """Search memory notes by query text with optional filters.

        Known limitation (#916): on the vector path, residual filters —
        ``category``, ``tags``/``keywords``, ``date_from``/``date_to``, and
        arbitrary metadata keys — are applied as a Python post-filter *after*
        ANN top-k retrieval (``vector_query.limit(k)``). A note that matches
        the filters but falls outside the ANN's top-k window will therefore
        not surface, even though it exists in the store. Only user_id, the
        scope dimensions, and the scope-exclusive directive
        (``SCOPE_EXCLUSIVE_FILTER_KEY``, compiled to an empty-``scope_dims``
        clause) are pushed into the ``where`` prefilter, because
        cross-principal crowd-out is an isolation-recall problem (#822);
        pushing more filter keys into the prefilter was considered and
        deliberately deferred as a possible future recall improvement (#916).
        Callers that need filter-complete results should use ``list_all()``
        (full scan) or pass a large ``k`` (the web route uses k=1000).
        """
        table = None
        try:
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )
            results = []

            # #822: push user_id + scope-dimension filters into a `where`
            # prefilter so the ANN returns k already-scoped neighbours; the rest
            # (category, tags/keywords/dates, arbitrary metadata keys) stays a
            # Python post-filter.
            where_sql, residual_filters = build_scope_where(filters)
            residual_other_filters = self._flat_other_filters(residual_filters)

            # Try vector search first
            try:
                query_embedding = self._get_embedding(query)
                if query_embedding:
                    # Check if vector column exists and has the right dimension
                    sample_df = table.search().limit(1).to_pandas()
                    if not sample_df.empty and "vector" in sample_df.columns:
                        # Try vector search
                        try:
                            vector_query = table.search(
                                query_embedding, vector_column_name="vector"
                            )
                            if where_sql:
                                # Prefilter: scope BEFORE the top-k selection, so
                                # crowd-out from other principals cannot collapse
                                # recall.
                                vector_query = vector_query.where(
                                    where_sql, prefilter=True
                                )
                            vector_df = vector_query.limit(k).to_pandas()

                            for _, row in vector_df.iterrows():
                                # Check similarity threshold
                                threshold = (
                                    similarity_threshold
                                    if similarity_threshold is not None
                                    else self._similarity_threshold
                                )
                                distance = row.get("_distance", float("inf"))
                                if distance > threshold:
                                    logger.info(
                                        f"Skipping result with distance {distance} > threshold {threshold}"
                                    )
                                    continue

                                logger.info(
                                    f"Accepting result with distance {distance} <= threshold {threshold}"
                                )

                                note_data = {
                                    "id": row.get("id", ""),
                                    "text": row.get("text", ""),
                                    "metadata": row.get("metadata", "{}"),
                                }
                                # #847: a malformed row must not abort the whole
                                # vector branch — escaping to the outer except
                                # after earlier appends would skip the text
                                # fallback and silently truncate the results.
                                try:
                                    note = self._dict_to_memory_note(note_data)
                                except Exception as row_error:
                                    logger.warning(
                                        "Skipping malformed memory row %r in "
                                        "vector search results: %s",
                                        note_data["id"],
                                        row_error,
                                    )
                                    continue

                                # user_id + scope dimensions were already applied
                                # as a `where` prefilter; only residual filters
                                # (category, tags/keywords/dates, arbitrary
                                # metadata keys) remain.
                                if residual_filters and not self._matches_filters(
                                    note, residual_filters, residual_other_filters
                                ):
                                    continue

                                results.append(note)
                        except Exception as vector_error:
                            if where_sql:
                                # Distinguish a pushdown-specific failure: the
                                # text fallback re-applies the full filters in
                                # Python, so isolation is preserved, but the
                                # population-independent recall pushdown (#822)
                                # is silently bypassed until the query is fixed.
                                logger.warning(
                                    "Scoped vector search with where-prefilter "
                                    "%r failed; falling back to text search "
                                    "(isolation preserved via Python filtering, "
                                    "but the recall pushdown is bypassed): %s",
                                    where_sql,
                                    vector_error,
                                )
                            else:
                                logger.warning(
                                    "Vector search failed, falling back to text "
                                    "search: %s",
                                    vector_error,
                                )
            except Exception as embedding_error:
                logger.warning(
                    f"Embedding generation failed, using text search: {embedding_error}"
                )

            # Fallback to text search if no vector results or vector search failed
            if not results:
                # Text search
                df = table.search().to_pandas()
                other_filters = self._flat_other_filters(filters)

                # Filter by query text and apply filters
                for _, row in df.iterrows():
                    text = row.get("text", "")

                    # Simple text matching
                    if query and query.lower() not in text.lower():
                        continue

                    note_data = {
                        "id": row.get("id", ""),
                        "text": text,
                        "metadata": row.get("metadata", "{}"),
                    }
                    # #847: here a malformed row would escape to the outer
                    # except and turn the whole query into an empty result.
                    try:
                        note = self._dict_to_memory_note(note_data)
                    except Exception as row_error:
                        logger.warning(
                            "Skipping malformed memory row %r in text "
                            "search results: %s",
                            note_data["id"],
                            row_error,
                        )
                        continue

                    # Unlike the vector path, nothing was pushed into `where`
                    # here, so the full filters apply (including the
                    # scope-exclusive directive and nested metadata isolation).
                    if filters and not self._matches_filters(
                        note, filters, other_filters
                    ):
                        continue

                    results.append(note)

                    if len(results) >= k:
                        break

            return results[:k]

        except Exception as e:
            logger.error(f"Failed to search memories with query '{query[:50]}...': {e}")
            return []
        finally:
            _safe_close_table(table)

    def search_required_vector(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[MemoryNote]:
        """Search through the compatible vector column without text fallback."""

        table = None
        try:
            query_embedding = self._get_required_embedding(query)
            table = self._vector_store.get_raw_connection().open_table(
                self._collection_name
            )
            if "vector" not in table.schema.names:
                raise RuntimeError("memory table has no vector column")
            where_sql, residual_filters = build_scope_where(filters)
            residual_other_filters = self._flat_other_filters(residual_filters)
            vector_query = table.search(query_embedding, vector_column_name="vector")
            if where_sql:
                vector_query = vector_query.where(where_sql, prefilter=True)
            rows = vector_query.limit(k).to_pandas()
            threshold = (
                similarity_threshold
                if similarity_threshold is not None
                else self._similarity_threshold
            )
            results: list[MemoryNote] = []
            for _, row in rows.iterrows():
                if row.get("_distance", float("inf")) > threshold:
                    continue
                note = self._dict_to_memory_note(
                    {
                        "id": row.get("id", ""),
                        "text": row.get("text", ""),
                        "metadata": row.get("metadata", "{}"),
                    }
                )
                if residual_filters and not self._matches_filters(
                    note, residual_filters, residual_other_filters
                ):
                    continue
                results.append(note)
            return results
        except MemoryBackendUnavailableError:
            raise
        except Exception as exc:
            raise MemoryBackendUnavailableError(
                MEMORY_BACKEND_UNAVAILABLE_REASON
            ) from exc
        finally:
            _safe_close_table(table)

    def clear(self) -> None:
        """Clear all memory notes from the store."""
        try:
            self._vector_store.clear()
        except Exception as e:
            logger.error(f"Failed to clear memory store: {e}")

    def list_all(self, filters: Optional[dict[str, Any]] = None) -> List[MemoryNote]:
        """List all memory notes with optional filtering.

        Delegates all filtering to search(): its ``_matches_filters`` dispatch
        is the single source of truth (category, nested ``{"metadata": {...}}``
        isolation filters, tags/keywords/date ranges, flat metadata equality),
        so list_all and search cannot diverge (#909). Results are sorted
        newest-first to match InMemoryStore.list_all.
        """
        try:
            notes = self.search(query="", k=10000, filters=filters or None)
            # Mirror InMemoryStore: newest first.
            notes.sort(key=lambda n: n.timestamp, reverse=True)
            return notes
        except Exception as e:
            logger.error(f"Failed to list all memories: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the memory store."""
        try:
            # Get all memories to calculate stats
            all_memories = self.list_all()

            total_count = len(all_memories)
            category_counts: dict[str, int] = {}
            tag_counts: dict[str, int] = {}

            for note in all_memories:
                # Count by category
                category_counts[note.category] = (
                    category_counts.get(note.category, 0) + 1
                )

                # Count tags
                for tag in note.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            return {
                "total_count": total_count,
                "category_counts": category_counts,
                "tag_counts": tag_counts,
                "memory_store_type": "lancedb",
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "total_count": 0,
                "category_counts": {},
                "tag_counts": {},
                "memory_store_type": "lancedb",
                "error": str(e),
            }

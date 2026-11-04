"""Tests for the public vector-storage helper surface.

After #510 the storage mechanics (chunk-selection read, vector write/upsert,
model-tag routing, dimension checks, spill-retry, index status) live in the
coordinator-owned collection handle, and the merge-error classifier lives in the
storage layer. These module-level helpers are now thin wrappers that delegate to
the ``KBVectorStorageCompatibilityFacade``.

This file therefore covers only the surviving public contract:
- ``validate_query_vector`` rules (pure validation);
- ``read_chunks_for_embedding`` / ``write_vectors_to_db`` behavior end-to-end
  through the real coordinator + LanceDB store.

The moved mechanics are covered at their new seams:
- handle write/read mechanics: ``kb/test_collection_handle_embedding.py``
- merge fallback / classifier: ``storage/test_lancedb_stores.py``
- column/idempotency/cleanup oracle: ``test_embedding_lifecycle_characterization.py``

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
)
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    read_chunks_for_embedding,
    validate_query_vector,
    write_vectors_to_db,
)


class TestReadChunksForEmbedding:
    def test_read_chunks_no_data(self) -> None:
        result = read_chunks_for_embedding(
            collection="coll",
            doc_id="nonexistent_doc",
            parse_hash="nonexistent_hash",
            model="test_model",
            is_admin=True,
        )
        assert isinstance(result, EmbeddingReadResponse)
        assert result.chunks == []
        assert result.total_count == 0
        assert result.pending_count == 0


class TestWriteVectorsToDb:
    def test_concurrent_writes_same_collection_are_serialized(self) -> None:
        """Regression (PR #915): concurrent page ingestions write to the same
        LanceDB table on separate threads; write_vectors_to_db must serialize
        them per collection so overlapping commits can't raise CommitConflict.
        """
        import threading
        import time
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.management.collection_manager import (
            reset_locks_for_testing,
        )

        reset_locks_for_testing()

        state_lock = threading.Lock()
        active = 0
        max_active = 0

        def _fake_write(**kwargs: object) -> EmbeddingWriteResponse:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return EmbeddingWriteResponse(
                upsert_count=0, deleted_stale_count=0, index_status="skipped"
            )

        facade = MagicMock()
        facade.write_vectors_to_db.side_effect = _fake_write

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager"
            "._get_vector_storage_compatibility_facade",
            return_value=facade,
        ):

            def _writer() -> None:
                write_vectors_to_db(collection="same", embeddings=[])

            threads = [threading.Thread(target=_writer) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert facade.write_vectors_to_db.call_count == 4
        # Serialized: never two writers inside the facade call at once.
        assert max_active == 1

    def test_write_vectors_empty_list(self) -> None:
        result = write_vectors_to_db(
            collection="coll", embeddings=[], create_index=True
        )
        assert isinstance(result, EmbeddingWriteResponse)
        assert result.upsert_count == 0
        assert result.deleted_stale_count == 0
        assert result.index_status == "skipped"

    def test_write_then_read_round_trip(self) -> None:
        # Seed a chunk so read-for-embedding can find it, then embed it and
        # confirm it is excluded from the next pending read (end-to-end through
        # the real coordinator + store).
        from datetime import datetime, timezone

        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_index_store,
        )
        from xagent.core.tools.core.RAG_tools.utils.metadata_utils import (
            serialize_metadata,
        )

        get_vector_index_store().upsert_chunks(
            [
                {
                    "collection": "coll",
                    "doc_id": "d1",
                    "parse_hash": "h1",
                    "chunk_id": "c0",
                    "index": 0,
                    "text": "hello",
                    "page_number": None,
                    "section": None,
                    "anchor": None,
                    "json_path": None,
                    "chunk_hash": "ch-c0",
                    "config_hash": "cfg1",
                    "created_at": datetime.now(timezone.utc),
                    "metadata": serialize_metadata({"k": "v"}),
                    "user_id": None,
                }
            ]
        )

        write_result = write_vectors_to_db(
            collection="coll",
            embeddings=[
                ChunkEmbeddingData(
                    doc_id="d1",
                    chunk_id="c0",
                    parse_hash="h1",
                    model="test_model",
                    vector=[0.1, 0.2, 0.3],
                    text="hello",
                    chunk_hash="ch-c0",
                    metadata=None,
                )
            ],
            create_index=False,
        )
        assert write_result.upsert_count == 1

        read_result = read_chunks_for_embedding(
            collection="coll",
            doc_id="d1",
            parse_hash="h1",
            model="test_model",
            is_admin=True,
        )
        assert read_result.total_count == 1
        assert read_result.pending_count == 0


class TestVectorValidation:
    """Validation of the public ``validate_query_vector`` contract."""

    def test_validate_query_vector_valid(self) -> None:
        validate_query_vector([1.0, 2.0, 3.0])
        validate_query_vector([0.5, -0.5, 0.0])
        validate_query_vector([1, 2, 3])  # integers are valid

    def test_validate_query_vector_invalid_type(self) -> None:
        with pytest.raises(VectorValidationError, match="query_vector must be a list"):
            validate_query_vector("not a list")  # type: ignore[arg-type]
        with pytest.raises(VectorValidationError, match="query_vector must be a list"):
            validate_query_vector(None)  # type: ignore[arg-type]

    def test_validate_query_vector_empty(self) -> None:
        with pytest.raises(VectorValidationError, match="query_vector cannot be empty"):
            validate_query_vector([])

    def test_validate_query_vector_invalid_elements(self) -> None:
        with pytest.raises(
            VectorValidationError, match="query_vector must contain only numbers"
        ):
            validate_query_vector([1.0, "invalid", 3.0])  # type: ignore[list-item]
        with pytest.raises(
            VectorValidationError, match="query_vector must contain only numbers"
        ):
            validate_query_vector([1.0, None, 3.0])  # type: ignore[list-item]

    def test_validate_query_vector_nan_infinity(self) -> None:
        with pytest.raises(
            VectorValidationError, match="query_vector contains invalid values"
        ):
            validate_query_vector([1.0, float("nan"), 3.0])
        with pytest.raises(
            VectorValidationError, match="query_vector contains invalid values"
        ):
            validate_query_vector([1.0, float("inf"), 3.0])
        with pytest.raises(
            VectorValidationError, match="query_vector contains invalid values"
        ):
            validate_query_vector([1.0, -float("inf"), 3.0])

    def test_validate_query_vector_numpy_scalar_types(self) -> None:
        np = pytest.importorskip("numpy")
        validate_query_vector(
            [np.float64(1.0), np.float32(2.0), np.int32(3)]  # type: ignore[list-item]
        )
        validate_query_vector(
            [np.float64(0.5), np.float32(-0.5), np.int64(0)]  # type: ignore[list-item]
        )
        validate_query_vector([np.float64(1.0), 2.0, np.int32(3)])  # type: ignore[list-item]

    def test_validate_without_connection(self) -> None:
        # Backward-compatible signature: model_tag/conn accepted, ignored.
        validate_query_vector([1.0, 2.0, 3.0])
        validate_query_vector([1.0, 2.0, 3.0], model_tag="test_model")

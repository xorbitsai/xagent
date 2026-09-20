"""Tests for search_dense functionality.

This module tests the dense vector search public surface:
- search_dense / search_dense_async public functions
- input validation at the public boundary
- routing through the collection handle
"""

import os
import tempfile
import uuid

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DenseSearchResponse,
    IndexResult,
    IndexStatus,
)
from xagent.core.tools.core.RAG_tools.retrieval.search_dense import (
    search_dense,
    search_dense_async,
)


class TestSearchDense:
    """Test search_dense main function."""

    def _patch_search_dense_module(self):
        """Helper method to import and patch search_dense module.

        Resolves ambiguity when module name and function name are the same.
        """
        import importlib

        search_dense_module = importlib.import_module(
            "xagent.core.tools.core.RAG_tools.retrieval.search_dense"
        )
        return search_dense_module

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_search_dense_input_validation(self):
        """Test input validation in search_dense."""
        # Test invalid collection
        with pytest.raises(DocumentValidationError):
            search_dense("", "model", [1.0, 2.0, 3.0], user_id=None, is_admin=True)

        # Test invalid model_tag
        with pytest.raises(DocumentValidationError):
            search_dense("collection", "", [1.0, 2.0, 3.0], user_id=None, is_admin=True)

        # Test invalid top_k
        with pytest.raises(DocumentValidationError):
            search_dense(
                "collection",
                "model",
                [1.0, 2.0, 3.0],
                top_k=0,
                user_id=None,
                is_admin=True,
            )

        with pytest.raises(DocumentValidationError):
            search_dense(
                "collection",
                "model",
                [1.0, 2.0, 3.0],
                top_k=2000,
                user_id=None,
                is_admin=True,
            )

    async def test_search_dense_async_input_validation(self):
        """Async path validates the same inputs as the sync path (#670)."""
        with pytest.raises(DocumentValidationError):
            await search_dense_async(
                "", "model", [1.0, 2.0, 3.0], user_id=None, is_admin=True
            )
        with pytest.raises(DocumentValidationError):
            await search_dense_async(
                "collection", "", [1.0, 2.0, 3.0], user_id=None, is_admin=True
            )
        with pytest.raises(DocumentValidationError):
            await search_dense_async(
                "collection",
                "model",
                [1.0, 2.0, 3.0],
                top_k=0,
                user_id=None,
                is_admin=True,
            )
        with pytest.raises(DocumentValidationError):
            await search_dense_async(
                "collection",
                "model",
                [1.0, 2.0, 3.0],
                top_k=2000,
                user_id=None,
                is_admin=True,
            )

    def test_search_dense_success_path(self, make_handle, routed_coordinator):
        """Test successful search_dense execution through the routed handle.

        Re-pointed (#511): the public ``search_dense`` now runs the real handle
        logic against a mock ``vector_index_store`` instead of patching the
        ``search_dense_engine`` free function. The handle reads raw rows from
        ``store.search_vectors_by_model`` and converts ``_distance`` to a score.
        """
        search_dense_module = self._patch_search_dense_module()
        handle, store, _ = make_handle()
        store.create_index.return_value = IndexResult(
            status="index_ready", advice="Index is ready", fts_enabled=True
        )
        store.search_vectors_by_model.return_value = [
            {
                "doc_id": "doc1",
                "chunk_id": "chunk1",
                "text": "content",
                "parse_hash": "hash1",
                "created_at": "2026-01-01",
                "metadata": None,
                "_distance": 0.0,  # score = 1/(1+0) = 1.0
            }
        ]

        with routed_coordinator(search_dense_module, handle):
            response = search_dense(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

        # Verify response
        assert isinstance(response, DenseSearchResponse)
        assert response.status == "success"
        assert len(response.results) == 1
        assert response.total_count == 1
        assert response.index_status == IndexStatus.INDEX_READY
        assert response.results[0].doc_id == "doc1"

        # The store search reached with the right model_tag/top_k/scope.
        store.search_vectors_by_model.assert_called_once()
        kwargs = store.search_vectors_by_model.call_args.kwargs
        assert kwargs["model_tag"] == "test_model"
        assert kwargs["top_k"] == 5
        assert kwargs["is_admin"] is True

    def test_search_dense_validation_fallback(self, make_handle, routed_coordinator):
        """Test search_dense returns cleanly when the store yields no rows."""
        search_dense_module = self._patch_search_dense_module()
        handle, store, _ = make_handle()
        store.create_index.return_value = IndexResult(
            status="index_ready", advice="Index is ready", fts_enabled=True
        )
        store.search_vectors_by_model.return_value = []

        with routed_coordinator(search_dense_module, handle):
            response = search_dense(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

        assert response.status == "success"
        assert response.total_count == 0
        # An empty store result yields a clean success response with no rows.
        store.search_vectors_by_model.assert_called_once()
        assert (
            store.search_vectors_by_model.call_args.kwargs["model_tag"] == "test_model"
        )

    def test_search_dense_index_status_mapping(self, make_handle, routed_coordinator):
        """Test index status mapping in search_dense via the routed handle."""
        search_dense_module = self._patch_search_dense_module()

        test_cases = [
            ("index_ready", IndexStatus.INDEX_READY),
            ("index_building", IndexStatus.INDEX_BUILDING),
            ("no_index", IndexStatus.NO_INDEX),
            ("index_corrupted", IndexStatus.INDEX_CORRUPTED),
            ("readonly", IndexStatus.READONLY),
            ("below_threshold", IndexStatus.BELOW_THRESHOLD),
        ]

        for engine_status, expected_enum in test_cases:
            handle, store, _ = make_handle()
            store.create_index.return_value = IndexResult(
                status=engine_status, advice="test advice", fts_enabled=True
            )
            store.search_vectors_by_model.return_value = []

            with routed_coordinator(search_dense_module, handle):
                response = search_dense(
                    "col", "model", [1.0], top_k=1, user_id=None, is_admin=True
                )
            assert response.index_status == expected_enum


class TestSearchDenseIntegration:
    """Integration tests for search_dense with real LanceDB operations."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_full_search_workflow(self, temp_lancedb_dir, test_collection):
        """Test complete search workflow from data insertion to retrieval."""
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_store_raw_connection,
        )
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            write_vectors_to_db,
        )

        conn = get_vector_store_raw_connection()
        model_tag = "integration_test_model"

        # Step 1: Clean up any existing table and create fresh table
        table_name = f"embeddings_{model_tag}"
        try:
            conn.drop_table(table_name)
        except Exception:
            pass  # Table might not exist, that's fine

        ensure_embeddings_table(conn, model_tag, vector_dim=3)

        # Create embeddings with Python lists for LanceDB compatibility
        embeddings = [
            ChunkEmbeddingData(
                doc_id="doc1",
                chunk_id="chunk1",
                parse_hash="parse1",
                model=model_tag,
                vector=[1.0, 0.0, 0.0],  # Unit vector along x-axis
                text="This is about artificial intelligence",
                chunk_hash="hash1",
            ),
            ChunkEmbeddingData(
                doc_id="doc2",
                chunk_id="chunk2",
                parse_hash="parse2",
                model=model_tag,
                vector=[0.0, 1.0, 0.0],  # Unit vector along y-axis
                text="This is about machine learning",
                chunk_hash="hash2",
            ),
        ]

        # Insert data
        write_result = write_vectors_to_db(
            test_collection,
            embeddings,
            create_index=False,  # Skip index creation for now
        )
        assert write_result.upsert_count == 2

        # Step 2: Execute search
        query_vector = [1.0, 0.0, 0.0]  # Same as first embedding
        response = search_dense(
            collection=test_collection,
            model_tag=model_tag,
            query_vector=query_vector,
            top_k=2,
            user_id=None,
            is_admin=True,
        )

        # Step 3: Verify results
        assert response.status == "success"
        assert len(response.results) == 2
        assert response.total_count == 2

        # First result should be the most similar (exact match)
        assert response.results[0].doc_id == "doc1"
        assert response.results[0].chunk_id == "chunk1"
        assert abs(response.results[0].score - 1.0) < 0.1  # High similarity score

        # Second result should be less similar
        assert response.results[1].doc_id == "doc2"
        assert response.results[1].score < response.results[0].score

        # Verify index status (include BELOW_THRESHOLD for small datasets)
        assert response.index_status in [
            IndexStatus.INDEX_READY,
            IndexStatus.INDEX_BUILDING,
            IndexStatus.BELOW_THRESHOLD,
        ]

    def test_search_with_filters(self, temp_lancedb_dir, test_collection):
        """Test search functionality with filters."""
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_store_raw_connection,
        )
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            write_vectors_to_db,
        )

        conn = get_vector_store_raw_connection()
        model_tag = "filter_test_model"

        # Clean up any existing table and create fresh table
        table_name = f"embeddings_{model_tag}"
        try:
            conn.drop_table(table_name)
        except Exception:
            pass  # Table might not exist, that's fine

        ensure_embeddings_table(conn, model_tag, vector_dim=2)

        # Create embeddings with Python lists for LanceDB compatibility
        embeddings = [
            ChunkEmbeddingData(
                doc_id="doc1",
                chunk_id="chunk1",
                parse_hash="parse1",
                model=model_tag,
                vector=[1.0, 0.0],
                text="First document content",
                chunk_hash="hash1",
            ),
            ChunkEmbeddingData(
                doc_id="doc2",
                chunk_id="chunk2",
                parse_hash="parse1",
                model=model_tag,
                vector=[0.0, 1.0],
                text="Second document content",
                chunk_hash="hash2",
            ),
        ]

        write_vectors_to_db(test_collection, embeddings, create_index=False)

        # Search with doc_id filter
        response = search_dense(
            collection=test_collection,
            model_tag=model_tag,
            query_vector=[1.0, 0.0],
            top_k=5,
            filters={"doc_id": "doc1"},
            user_id=None,
            is_admin=True,
        )

        # Should only return results from doc1
        assert len(response.results) == 1
        assert response.results[0].doc_id == "doc1"

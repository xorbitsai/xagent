"""Tests for search_dense functionality.

This module tests the dense vector search implementation:
- search_dense main function
- search_engine core logic
- _build_safe_filter utility
- Integration with LanceDB and index management
"""

import os
import tempfile
import uuid
from unittest.mock import Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DenseSearchResponse,
    IndexStatus,
    SearchResult,
)
from xagent.core.tools.core.RAG_tools.retrieval.search_dense import search_dense
from xagent.core.tools.core.RAG_tools.retrieval.search_engine import search_dense_engine


class TestSearchDenseEngine:
    """Test search_dense_engine function."""

    @pytest.fixture
    def mock_search_chain(self):
        """Create a reusable mock search chain for table operations.

        Returns a function that sets up the mock chain and returns mock objects.
        The returned function accepts an optional results_df parameter.
        """

        def _create_mock_chain(mock_table: Mock, results_df=None):
            """Create and configure the mock search chain.

            Args:
                mock_table: The mock table to attach the chain to
                results_df: Optional DataFrame to return from to_pandas()
                           If None, defaults to empty DataFrame

            Returns:
                Tuple of (mock_search, mock_where, mock_limit)
            """
            import pandas as pd

            # Default to empty DataFrame if not provided
            if results_df is None:
                results_df = pd.DataFrame([])

            # Create mock chain for search -> where -> limit -> to_pandas
            mock_search = Mock()
            mock_where = Mock()
            mock_limit = Mock()

            mock_table.search.return_value = mock_search
            mock_search.where.return_value = mock_where
            mock_search.limit.return_value = (
                mock_limit  # For when no filters are applied
            )
            mock_where.limit.return_value = mock_limit  # For when filters are applied

            # UPDATED: Support to_arrow() -> to_list() -> to_pandas() three-tier fallback
            # Create mock Arrow table
            mock_arrow_table = Mock()
            mock_arrow_table.to_pylist.return_value = results_df.to_dict("records")
            mock_limit.to_arrow.return_value = mock_arrow_table

            mock_limit.to_list.return_value = results_df.to_dict("records")
            mock_limit.to_pandas.return_value = results_df

            return mock_search, mock_where, mock_limit

        return _create_mock_chain

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_connection_from_env"
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.build_lancedb_filter_expression"
    )
    def test_search_engine_basic(
        self, mock_build_filter: Mock, mock_get_conn: Mock, mock_search_chain
    ) -> None:
        """Test basic search engine functionality."""
        # Mock connection and table
        mock_conn = Mock()
        mock_table = Mock()
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.return_value = mock_table

        # Mock table operations - create proper chain of mocks
        import pandas as pd

        mock_results_df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "chunk_id": "chunk1",
                    "text": "test content",
                    "score": 0.8,
                    "parse_hash": "hash1",
                    "model_tag": "test_model",
                    "created_at": pd.Timestamp.now(),
                    "_distance": 0.5,  # Squared Euclidean distance
                }
            ]
        )

        # Use fixture to create mock search chain
        mock_search, mock_where, mock_limit = mock_search_chain(
            mock_table, mock_results_df
        )

        # Collection filter is always applied for KB isolation
        mock_build_filter.return_value = "collection == 'test_collection'"

        # Mock index manager
        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_index_manager"
        ) as mock_get_index_manager:
            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "index_ready",
                "Index ready",
            )
            mock_get_index_manager.return_value = mock_index_manager

            # Execute search
            results, index_status, index_advice = search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            # Verify results
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].doc_id == "doc1"
            assert results[0].chunk_id == "chunk1"
            assert results[0].text == "test content"
            assert (
                abs(results[0].score - (1.0 / (1.0 + 0.5))) < 0.001
            )  # Distance to similarity conversion

            # Verify table operations
            mock_get_conn.assert_called_once()
            mock_conn.open_table.assert_called_once_with("embeddings_test_model")
            mock_get_index_manager.assert_called_once()
            mock_index_manager.check_and_create_index.assert_called_once_with(
                mock_table, "embeddings_test_model", False
            )
            mock_table.search.assert_called_once_with(
                [0.1, 0.2, 0.3],
                vector_column_name="vector",
            )
            # Collection filter must be applied for KB isolation (Issue #72)
            mock_build_filter.assert_any_call({"collection": "test_collection"})

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_connection_from_env"
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.build_lancedb_filter_expression"
    )
    def test_search_engine_with_filters(
        self, mock_build_filter: Mock, mock_get_conn: Mock, mock_search_chain
    ) -> None:
        """Test search engine with filters."""
        mock_conn = Mock()
        mock_table = Mock()
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.return_value = mock_table

        # Mock search results - use fixture
        import pandas as pd

        mock_results_df = pd.DataFrame([])

        # Use fixture to create mock search chain
        mock_search_chain(mock_table, mock_results_df)

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_index_manager"
        ) as mock_get_index_manager:
            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "index_ready",
                "Index ready",
            )
            mock_get_index_manager.return_value = mock_index_manager

            # Execute search with filters (collection filter + custom filters)
            filters = {"doc_id": "test_doc", "file_type": "pdf"}
            expected_filter_clause = "doc_id = 'test_doc' AND file_type = 'pdf'"
            mock_build_filter.side_effect = [
                "collection == 'test_collection'",
                expected_filter_clause,
            ]

            search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                filters=filters,
                user_id=None,
                is_admin=True,
            )

            # Verify filter application (collection filter + custom filters)
            mock_get_conn.assert_called_once()
            mock_conn.open_table.assert_called_once_with("embeddings_test_model")
            mock_get_index_manager.assert_called_once()
            mock_index_manager.check_and_create_index.assert_called_once_with(
                mock_table, "embeddings_test_model", False
            )
            mock_build_filter.assert_any_call({"collection": "test_collection"})
            mock_build_filter.assert_any_call(filters)
            search_query = mock_table.search.return_value
            # Note: The filter is wrapped in parentheses by the filter application logic
            search_query.where.assert_called_once()
            where_arg = search_query.where.call_args[0][0]
            assert expected_filter_clause in where_arg
            search_query.where.return_value.limit.assert_called_once_with(5)

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_connection_from_env"
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.build_lancedb_filter_expression"
    )
    def test_search_dense_engine_applies_collection_filter(
        self, mock_build_filter: Mock, mock_get_conn: Mock, mock_search_chain
    ) -> None:
        """Test that search_dense_engine always applies collection filter for KB isolation (Issue #72)."""
        mock_conn = Mock()
        mock_table = Mock()
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.return_value = mock_table

        import pandas as pd

        mock_search_chain(mock_table, pd.DataFrame([]))
        mock_build_filter.return_value = "collection == 'my_kb'"

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_index_manager"
        ) as mock_get_index_manager:
            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "index_ready",
                None,
            )
            mock_get_index_manager.return_value = mock_index_manager

            search_dense_engine(
                collection="my_kb",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            mock_build_filter.assert_any_call({"collection": "my_kb"})
            search_query = mock_table.search.return_value
            search_query.where.assert_called_once()
            where_arg = search_query.where.call_args[0][0]
            assert "collection" in where_arg.lower() or "my_kb" in where_arg

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_connection_from_env"
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.build_lancedb_filter_expression"
    )
    def test_search_engine_readonly_mode(
        self, mock_build_filter: Mock, mock_get_conn: Mock, mock_search_chain
    ) -> None:
        """Test search engine in readonly mode."""
        mock_conn = Mock()
        mock_table = Mock()
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.return_value = mock_table

        # Mock search results - use fixture
        import pandas as pd

        mock_results_df = pd.DataFrame([])

        # Use fixture to create mock search chain
        mock_search_chain(mock_table, mock_results_df)

        # Collection filter is always applied for KB isolation
        mock_build_filter.return_value = "collection == 'test_collection'"

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_index_manager"
        ) as mock_get_index_manager:
            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "readonly",
                "Readonly mode - no index operations",
            )
            mock_get_index_manager.return_value = mock_index_manager

            # Execute search in readonly mode
            results, index_status, index_advice = search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                readonly=True,
                user_id=None,
                is_admin=True,
            )

            assert index_status == "readonly"
            assert index_advice == "Readonly mode - no index operations"

            # Verify readonly mode passed to index manager
            mock_get_conn.assert_called_once()
            mock_conn.open_table.assert_called_once_with("embeddings_test_model")
            mock_get_index_manager.assert_called_once()
            mock_index_manager.check_and_create_index.assert_called_once_with(
                mock_table, "embeddings_test_model", True
            )
            mock_table.search.assert_called_once_with(
                [0.1, 0.2, 0.3],
                vector_column_name="vector",
            )
            # Collection filter is always applied for KB isolation
            mock_build_filter.assert_any_call({"collection": "test_collection"})

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_connection_from_env"
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.build_lancedb_filter_expression"
    )
    def test_search_engine_error_handling(
        self, mock_build_filter: Mock, mock_get_conn: Mock
    ) -> None:
        """Test error handling in search engine."""
        mock_conn = Mock()
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.side_effect = Exception("Database connection failed")

        mock_build_filter.return_value = None

        # Mock index manager to avoid uncalled mock issues if exception occurs early
        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_index_manager"
        ) as mock_get_index_manager:
            mock_get_index_manager.return_value = Mock()

            with pytest.raises(Exception, match="Database connection failed"):
                search_dense_engine(
                    collection="test_collection",
                    model_tag="test_model",
                    query_vector=[0.1, 0.2, 0.3],
                    top_k=5,
                    user_id=None,
                    is_admin=True,
                )
            mock_get_conn.assert_called_once()
            mock_conn.open_table.assert_called_once_with("embeddings_test_model")
            mock_get_index_manager.assert_not_called()  # Should not be called if open_table fails


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

    def test_search_dense_success_path(self):
        """Test successful search_dense execution."""
        search_dense_module = self._patch_search_dense_module()

        with (
            patch.object(search_dense_module, "search_dense_engine") as mock_engine,
            patch.object(
                search_dense_module, "get_connection_from_env"
            ) as mock_get_conn,
            patch.object(search_dense_module, "validate_query_vector") as mock_validate,
        ):
            # Mock dependencies
            mock_conn = Mock()
            mock_get_conn.return_value = mock_conn

            mock_validate.return_value = None

            from datetime import datetime

            mock_results = [
                SearchResult(
                    doc_id="doc1",
                    chunk_id="chunk1",
                    text="content",
                    score=0.8,
                    parse_hash="hash1",
                    model_tag="test_model",
                    created_at=datetime.now(),
                )
            ]
            mock_engine.return_value = (mock_results, "index_ready", "Index is ready")

            # Execute search
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

            # Verify function calls
            mock_validate.assert_called_once_with(
                [0.1, 0.2, 0.3], "test_model", conn=mock_conn
            )
            mock_engine.assert_called_once()

    def test_search_dense_validation_fallback(self):
        """Test search_dense with validation fallback."""
        search_dense_module = self._patch_search_dense_module()

        with (
            patch.object(search_dense_module, "search_dense_engine") as mock_engine,
            patch.object(
                search_dense_module, "get_connection_from_env"
            ) as mock_get_conn,
            patch.object(search_dense_module, "validate_query_vector") as mock_validate,
        ):
            # Mock connection failure - get_connection_from_env fails before validation
            mock_get_conn.side_effect = Exception("Connection failed")

            # Mock validation: only fallback call (without conn) will happen
            def validate_side_effect(*args, **kwargs):
                if "conn" in kwargs and kwargs["conn"] is not None:
                    # This branch won't be reached because get_connection_from_env fails first
                    raise Exception("Validation failed")
                else:
                    # Call without conn parameter - should succeed (fallback validation)
                    return None

            mock_validate.side_effect = validate_side_effect

            mock_results = []
            mock_engine.return_value = (mock_results, "index_ready", "Index is ready")

            # Execute search (should not fail)
            search_dense(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            # Verify fallback behavior - since get_connection_from_env fails, only fallback call happens
            assert mock_validate.call_count == 1  # Only fallback call without conn
            # Verify the call was made without conn parameter
            mock_validate.assert_called_with([0.1, 0.2, 0.3])

    def test_search_dense_index_status_mapping(self):
        """Test index status mapping in search_dense."""
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
            with (
                patch.object(search_dense_module, "search_dense_engine") as mock_engine,
                patch.object(search_dense_module, "validate_query_vector"),
                patch("xagent.providers.vector_store.lancedb.get_connection_from_env"),
            ):
                mock_engine.return_value = ([], engine_status, "test advice")

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
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            write_vectors_to_db,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
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
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            write_vectors_to_db,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
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

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_connection_from_env"
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.build_lancedb_filter_expression"
    )
    def test_search_engine_arrow_fallback_to_list(
        self, mock_build_filter: Mock, mock_get_conn: Mock
    ) -> None:
        """Test search engine fallback from to_arrow() to to_list()."""
        mock_conn = Mock()
        mock_table = Mock()
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.return_value = mock_table

        import pandas as pd

        mock_results_df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "chunk_id": "chunk1",
                    "text": "test content",
                    "_distance": 0.5,
                    "parse_hash": "hash1",
                    "created_at": pd.Timestamp.now(),
                    "metadata": '{"key": "value"}',
                }
            ]
        )

        # Create mock search chain - use chainable mocks
        mock_search = Mock()
        mock_limit = Mock()

        mock_table.search.return_value = mock_search
        # Chain: search().where().limit() - each returns the next in chain
        mock_search.where.return_value = mock_search
        mock_search.limit.return_value = mock_limit

        # Simulate to_arrow() failing (AttributeError), fallback to to_list()
        mock_limit.to_arrow.side_effect = AttributeError("to_arrow not available")
        # to_list() should return a list, not a Mock
        mock_limit.to_list.return_value = mock_results_df.to_dict("records")

        mock_build_filter.return_value = None

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_index_manager"
        ) as mock_get_index_manager:
            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "index_ready",
                "Index ready",
            )
            mock_get_index_manager.return_value = mock_index_manager

            results, _, _ = search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            # Verify results
            assert len(results) == 1
            assert results[0].doc_id == "doc1"
            # Verify fallback was used
            mock_limit.to_arrow.assert_called_once()
            mock_limit.to_list.assert_called_once()

    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_connection_from_env"
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.retrieval.search_engine.build_lancedb_filter_expression"
    )
    def test_search_engine_arrow_fallback_to_pandas_with_nan(
        self, mock_build_filter: Mock, mock_get_conn: Mock
    ) -> None:
        """Test search engine fallback to to_pandas() and NaN normalization."""
        mock_conn = Mock()
        mock_table = Mock()
        mock_get_conn.return_value = mock_conn
        mock_conn.open_table.return_value = mock_table

        import numpy as np
        import pandas as pd

        # Create DataFrame with NaN values
        mock_results_df = pd.DataFrame(
            [
                {
                    "doc_id": "doc1",
                    "chunk_id": "chunk1",
                    "text": "test content",
                    "_distance": 0.5,
                    "parse_hash": "hash1",
                    "created_at": pd.Timestamp.now(),
                    "metadata": '{"key": "value"}',
                    "optional_field": np.nan,  # NaN value
                }
            ]
        )

        # Create mock search chain - use chainable mocks
        mock_search = Mock()
        mock_limit = Mock()

        mock_table.search.return_value = mock_search
        # Chain: search().where().limit() - each returns the next in chain
        mock_search.where.return_value = mock_search
        mock_search.limit.return_value = mock_limit

        # Simulate both to_arrow() and to_list() failing, fallback to to_pandas()
        mock_limit.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_limit.to_list.side_effect = AttributeError("to_list not available")
        mock_limit.to_pandas.return_value = mock_results_df

        mock_build_filter.return_value = None

        with patch(
            "xagent.core.tools.core.RAG_tools.retrieval.search_engine.get_index_manager"
        ) as mock_get_index_manager:
            mock_index_manager = Mock()
            mock_index_manager.check_and_create_index.return_value = (
                "index_ready",
                "Index ready",
            )
            mock_get_index_manager.return_value = mock_index_manager

            results, _, _ = search_dense_engine(
                collection="test_collection",
                model_tag="test_model",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                user_id=None,
                is_admin=True,
            )

            # Verify results
            assert len(results) == 1
            assert results[0].doc_id == "doc1"
            # Verify all fallbacks were attempted
            mock_limit.to_arrow.assert_called_once()
            mock_limit.to_list.assert_called_once()
            mock_limit.to_pandas.assert_called_once()

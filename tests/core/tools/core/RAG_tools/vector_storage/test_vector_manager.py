"""Tests for vector_manager functionality.

This module tests the vector storage data management functions:
- read_chunks_for_embedding: Reading chunks from database for embedding
- write_vectors_to_db: Writing embedding vectors with idempotency
- validate_query_vector: Vector validation functionality
- Vector consistency and error handling
"""

import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
)
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    _group_embeddings_by_model,
    _validate_and_prepare_table,
    read_chunks_for_embedding,
    validate_query_vector,
    write_vectors_to_db,
)


def _create_mock_table_with_schema() -> MagicMock:
    """Create a mock table with a schema that includes the metadata and user_id fields.

    This helper function ensures that schema validation passes in tests
    by providing a mock schema that includes all required fields, especially
    the 'metadata' and 'user_id' fields that are validated in ensure_chunks_table
    and ensure_embeddings_table.

    Returns:
        A MagicMock table object with a properly configured schema.
    """
    table = MagicMock()
    # Create mock schema fields - include 'metadata' and 'user_id' which are validated
    metadata_field = MagicMock()
    metadata_field.name = "metadata"
    user_id_field = MagicMock()
    user_id_field.name = "user_id"
    collection_field = MagicMock()
    collection_field.name = "collection"
    doc_id_field = MagicMock()
    doc_id_field.name = "doc_id"
    # Set schema as a list of field objects (mimicking PyArrow schema structure)
    table.schema = [collection_field, doc_id_field, metadata_field, user_id_field]
    return table


class TestReadChunksForEmbedding:
    """Test cases for read_chunks_for_embedding functionality."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set the environment variable for LanceDB directory
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            # Restore original environment
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_read_chunks_no_data(self, temp_lancedb_dir, test_collection):
        """Test reading chunks when no data exists."""
        result = read_chunks_for_embedding(
            collection=test_collection,
            doc_id="nonexistent_doc",
            parse_hash="nonexistent_hash",
            model="test_model",
        )

        assert isinstance(result, EmbeddingReadResponse)
        assert len(result.chunks) == 0
        assert result.total_count == 0
        assert result.pending_count == 0

    def test_read_chunks_for_embedding_sql_injection_protection(
        self, temp_lancedb_dir, test_collection
    ):
        """Test read_chunks_for_embedding protects against SQL injection."""
        from unittest.mock import MagicMock

        # Create mock connection and tables
        mock_db_connection = MagicMock()
        mock_chunks_table = _create_mock_table_with_schema()
        mock_embeddings_table = _create_mock_table_with_schema()

        # Configure open_table to return appropriate mock tables using side_effect
        def mock_open_table_func(table_name):
            if table_name == "chunks":
                return mock_chunks_table
            elif table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return MagicMock()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        # Mock create_table to do nothing (tables are "created" but we use our mocks)
        mock_db_connection.create_table.return_value = None

        # UPDATED: Mock both to_list() and to_pandas() for optimization support
        # Mock empty results for chunks
        mock_chunks_table.search.return_value.where.return_value.to_list.return_value = []
        mock_chunks_table.search.return_value.where.return_value.to_pandas.return_value = pd.DataFrame()
        mock_chunks_table.count_rows.return_value = (
            0  # Changed to 0 to match empty results
        )

        # Mock empty results for embeddings
        mock_embeddings_table.search.return_value.where.return_value.select.return_value.to_list.return_value = []
        mock_embeddings_table.search.return_value.where.return_value.select.return_value.to_pandas.return_value = pd.DataFrame()

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            malicious_input = "malicious' OR 1=1 --"
            safe_collection = test_collection
            safe_parse_hash = "safe_hash"
            safe_model = "test_model"

            result = read_chunks_for_embedding(
                collection=safe_collection,
                doc_id=malicious_input,
                parse_hash=safe_parse_hash,
                model=safe_model,
                user_id=None,
                is_admin=True,  # Use admin to avoid user_id filter
            )

            # Verify count_rows was called with escaped input
            # Single quotes should be doubled: ' becomes ''
            expected_chunks_where_clause = (
                f"collection == '{safe_collection}' AND "
                f"doc_id == 'malicious'' OR 1=1 --' AND "
                f"parse_hash == '{safe_parse_hash}'"
            )
            # When total_count==0 we may call count_rows again (base_filter_expr); allow at least one call with expected clause
            mock_chunks_table.count_rows.assert_any_call(expected_chunks_where_clause)

            # Since count_rows returns 0, search() should not be called
            mock_chunks_table.search.assert_not_called()

            # Since no chunks exist, embeddings table should not be queried
            mock_embeddings_table.search.assert_not_called()

            assert result.chunks == []
            assert result.total_count == 0  # Changed from 1 to 0
            assert result.pending_count == 0


class TestGroupEmbeddingsByModel:
    """Tests for _group_embeddings_by_model helper."""

    def test_group_embeddings_by_model_empty(self):
        """Test grouping empty list returns empty dict."""
        assert _group_embeddings_by_model([]) == {}

    def test_group_embeddings_by_model_single_model(self):
        """Test grouping single model returns one key with all items."""
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        embeddings = [
            ChunkEmbeddingData(
                collection="c",
                doc_id="d1",
                chunk_id="ch1",
                parse_hash="h",
                model="m1",
                vector=[0.1, 0.2],
                text="t1",
                chunk_hash="ch",
            ),
            ChunkEmbeddingData(
                collection="c",
                doc_id="d2",
                chunk_id="ch2",
                parse_hash="h",
                model="m1",
                vector=[0.2, 0.3],
                text="t2",
                chunk_hash="ch",
            ),
        ]
        result = _group_embeddings_by_model(embeddings)
        assert list(result.keys()) == ["m1"]
        assert len(result["m1"]) == 2

    def test_group_embeddings_by_model_multiple_models(self):
        """Test grouping multiple models returns separate lists."""
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        embeddings = [
            ChunkEmbeddingData(
                collection="c",
                doc_id="d1",
                chunk_id="ch1",
                parse_hash="h",
                model="m1",
                vector=[0.1, 0.2],
                text="t1",
                chunk_hash="ch",
            ),
            ChunkEmbeddingData(
                collection="c",
                doc_id="d2",
                chunk_id="ch2",
                parse_hash="h",
                model="m2",
                vector=[0.2, 0.3],
                text="t2",
                chunk_hash="ch",
            ),
        ]
        result = _group_embeddings_by_model(embeddings)
        assert set(result.keys()) == {"m1", "m2"}
        assert len(result["m1"]) == 1 and result["m1"][0].model == "m1"
        assert len(result["m2"]) == 1 and result["m2"][0].model == "m2"


class TestValidateAndPrepareTable:
    """Tests for _validate_and_prepare_table helper."""

    def test_validate_and_prepare_table_existing_same_dimension(self):
        """Test table exists with same vector dimension is not dropped."""
        from unittest.mock import MagicMock, patch

        conn = MagicMock()
        table_name = "embeddings_test_tag"
        conn.table_names.return_value = [table_name]
        existing_table = MagicMock()
        mock_vector_field = MagicMock()
        mock_vector_field.type.list_size = 2
        mock_schema = MagicMock()
        mock_schema.field.return_value = mock_vector_field
        existing_table.schema = mock_schema
        conn.open_table.return_value = existing_table
        conn.drop_table = MagicMock()

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_embeddings_table"
        ) as mock_ensure:
            result = _validate_and_prepare_table(
                conn, "test_tag", table_name, vector_dim=2
            )
        # Same dimension: should not drop; ensure_embeddings_table then open_table
        conn.drop_table.assert_not_called()
        mock_ensure.assert_called_once_with(conn, "test_tag", vector_dim=2)
        assert result is existing_table

    def test_validate_and_prepare_table_incompatible_vector_type_no_list_size(
        self,
    ):
        """Test table with vector field without list_size is dropped and recreated."""
        from unittest.mock import MagicMock, patch

        conn = MagicMock()
        table_name = "embeddings_test_tag"
        conn.table_names.return_value = [table_name]
        existing_table = MagicMock()
        # Use a type object without list_size so hasattr(..., "list_size") is False
        vector_type_no_list_size = type("VectorType", (), {})()
        mock_vector_field = MagicMock()
        mock_vector_field.type = vector_type_no_list_size
        mock_schema = MagicMock()
        mock_schema.field.return_value = mock_vector_field
        existing_table.schema = mock_schema
        conn.open_table.return_value = existing_table
        conn.drop_table = MagicMock()
        new_table = MagicMock()
        conn.open_table.side_effect = [existing_table, new_table]

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_embeddings_table"
        ):
            result = _validate_and_prepare_table(
                conn, "test_tag", table_name, vector_dim=2
            )
        conn.drop_table.assert_called_once_with(table_name)
        assert result is new_table

    def test_validate_and_prepare_table_schema_check_exception_then_recreate(
        self,
    ):
        """Test when schema check raises, drop is attempted and table is recreated."""
        from unittest.mock import MagicMock, patch

        conn = MagicMock()
        table_name = "embeddings_test_tag"
        conn.table_names.return_value = [table_name]
        conn.drop_table = MagicMock()
        new_table = MagicMock()
        # First open_table (in try) raises; after ensure_embeddings_table, second open_table returns new_table
        conn.open_table.side_effect = [RuntimeError("schema error"), new_table]

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_embeddings_table"
        ):
            result = _validate_and_prepare_table(
                conn, "test_tag", table_name, vector_dim=2
            )
        conn.drop_table.assert_called_once_with(table_name)
        assert result is new_table


class TestWriteVectorsToDb:
    """Test cases for write_vectors_to_db functionality."""

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

    def test_write_vectors_empty_list(self, temp_lancedb_dir, test_collection):
        """Test writing empty embedding list."""
        result = write_vectors_to_db(
            collection=test_collection,
            embeddings=[],
        )

        assert isinstance(result, EmbeddingWriteResponse)
        assert result.upsert_count == 0
        assert result.deleted_stale_count == 0
        assert result.index_status == "skipped"

    def test_write_vectors_to_db_sql_injection_protection(
        self, temp_lancedb_dir, test_collection
    ):
        """Test write_vectors_to_db protects against SQL injection."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock connection and table
        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        # Configure open_table to return the mock embeddings table using side_effect
        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        # Mock create_table to do nothing (tables are "created" but we use our mocks)
        mock_db_connection.create_table.return_value = None

        # Mock search to return empty DataFrame so no deletions happen initially
        mock_embeddings_table.search.return_value.where.return_value.to_pandas.return_value = pd.DataFrame()
        # Mock merge_insert method and its chain calls
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_execute = MagicMock()

        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        mock_when_not_matched.execute.return_value = mock_execute
        # Keep add method for fallback testing
        mock_embeddings_table.add.return_value = None  # Mock add method
        mock_embeddings_table.__len__.return_value = 0  # Mock len for index creation
        mock_embeddings_table.count_rows.return_value = (
            0  # Mock count_rows for index creation
        )
        mock_embeddings_table.create_index.return_value = (
            None  # Mock create_index method
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            malicious_doc_id = "malicious' OR 1=1 --"
            safe_collection = test_collection
            safe_parse_hash = "safe_hash"
            safe_model = "test_model"
            malicious_chunk_id = "chunk'id"
            safe_chunk_hash = "safe_hash"

            # Create an embedding with malicious doc_id
            malicious_embedding = ChunkEmbeddingData(
                collection=safe_collection,
                doc_id=malicious_doc_id,
                chunk_id=malicious_chunk_id,
                parse_hash=safe_parse_hash,
                model=safe_model,
                vector=[0.1, 0.2],
                text="malicious text",
                chunk_hash=safe_chunk_hash,
            )

            result = write_vectors_to_db(
                collection=safe_collection,
                embeddings=[malicious_embedding],
            )

            # With merge_insert, we no longer need to search for existing records
            # merge_insert handles upsert automatically based on primary keys
            # Verify that search was not called (merge_insert doesn't need it)
            mock_embeddings_table.search.assert_not_called()
            # Verify that delete was not called (merge_insert handles updates automatically)
            mock_embeddings_table.delete.assert_not_called()
            # Verify that merge_insert was called with the correct data
            mock_embeddings_table.merge_insert.assert_called_once()
            # Get the records argument from execute() method call
            call_args = mock_when_not_matched.execute.call_args[0][0]
            assert len(call_args) == 1
            assert call_args[0]["doc_id"] == malicious_doc_id
            assert call_args[0]["chunk_id"] == malicious_chunk_id

            # Verify the chain calls were made
            mock_merge_insert.when_matched_update_all.assert_called_once()
            mock_when_matched.when_not_matched_insert_all.assert_called_once()
            mock_when_not_matched.execute.assert_called_once()

            # Verify that add was not called (since merge_insert succeeded)
            mock_embeddings_table.add.assert_not_called()

            assert result.upsert_count == 1
            assert result.deleted_stale_count == 0
            assert result.index_status == "skipped_threshold"

    def test_write_vectors_merge_insert_fallback_to_add(
        self, temp_lancedb_dir, test_collection
    ):
        """Test merge_insert failure fallback to add method."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Mock merge_insert to fail, then add to succeed
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        # merge_insert fails
        mock_when_not_matched.execute.side_effect = Exception("merge_insert failed")
        # add succeeds
        mock_embeddings_table.add.return_value = None
        mock_embeddings_table.count_rows.return_value = 0

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
            )

            # Verify merge_insert was attempted
            mock_embeddings_table.merge_insert.assert_called_once()
            # Verify fallback to add was used
            mock_embeddings_table.add.assert_called_once()
            assert result.upsert_count == 1

    def test_write_vectors_merge_insert_non_recoverable_error_no_fallback(
        self, temp_lancedb_dir, test_collection
    ):
        """Test that non-recoverable errors (schema, type mismatch) do not fallback to add."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DatabaseOperationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Mock merge_insert to fail with schema error (non-recoverable)
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        # Schema error - should not fallback
        mock_when_not_matched.execute.side_effect = ValueError(
            "Schema mismatch: expected int, got string"
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            # ValueError is wrapped in DatabaseOperationError by outer exception handler
            with pytest.raises(DatabaseOperationError, match="Schema mismatch"):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=[embedding],
                )

            # Verify merge_insert was attempted
            mock_embeddings_table.merge_insert.assert_called_once()
            # Verify add was NOT called (no fallback for non-recoverable errors)
            mock_embeddings_table.add.assert_not_called()

    def test_write_vectors_merge_insert_type_mismatch_error_no_fallback(
        self, temp_lancedb_dir, test_collection
    ):
        """Test that type mismatch errors do not fallback to add."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DatabaseOperationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Mock merge_insert to fail with type error (non-recoverable)
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        # Type error - should not fallback
        mock_when_not_matched.execute.side_effect = TypeError(
            "Type mismatch: invalid type for field"
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            # TypeError is wrapped in DatabaseOperationError by outer exception handler
            with pytest.raises(DatabaseOperationError, match="Type mismatch"):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=[embedding],
                )

            # Verify add was NOT called
            mock_embeddings_table.add.assert_not_called()

    def test_write_vectors_merge_insert_dimension_error_no_fallback(
        self, temp_lancedb_dir, test_collection
    ):
        """Test that dimension mismatch errors do not fallback to add."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DatabaseOperationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Mock merge_insert to fail with dimension error (non-recoverable)
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        # Dimension error - should not fallback
        mock_when_not_matched.execute.side_effect = ValueError(
            "Vector dimension mismatch: expected 3, got 2"
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            # ValueError is wrapped in DatabaseOperationError by outer exception handler
            with pytest.raises(DatabaseOperationError, match="dimension mismatch"):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=[embedding],
                )

            # Verify add was NOT called
            mock_embeddings_table.add.assert_not_called()

    def test_write_vectors_merge_insert_recoverable_error_with_fallback(
        self, temp_lancedb_dir, test_collection
    ):
        """Test that recoverable errors (network, timeout) do fallback to add."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Mock merge_insert to fail with network error (recoverable)
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        # Network/timeout error - should fallback
        mock_when_not_matched.execute.side_effect = ConnectionError(
            "Network timeout: connection lost"
        )
        # add succeeds
        mock_embeddings_table.add.return_value = None
        mock_embeddings_table.count_rows.return_value = 0

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
            )

            # Verify merge_insert was attempted
            mock_embeddings_table.merge_insert.assert_called_once()
            # Verify fallback to add was used
            mock_embeddings_table.add.assert_called_once()
            assert result.upsert_count == 1

    def test_write_vectors_merge_insert_and_add_both_fail(
        self, temp_lancedb_dir, test_collection
    ):
        """Test when both merge_insert and add fail."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DatabaseOperationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Both merge_insert and add fail
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        mock_when_not_matched.execute.side_effect = Exception("merge_insert failed")
        mock_embeddings_table.add.side_effect = Exception("add also failed")

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            with pytest.raises(DatabaseOperationError):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=[embedding],
                )

    def test_write_vectors_spill_retry(self, temp_lancedb_dir, test_collection):
        """Test that spill error reduces batch size and retries without losing data."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # First execute() raises spill; subsequent succeed
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        mock_when_not_matched.execute.side_effect = [
            Exception("Spill has sent an error"),
            None,
            None,
            None,
            None,
            None,
        ]
        mock_embeddings_table.count_rows.return_value = 0

        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text=f"text_{i}",
                chunk_hash="test_hash",
            )
            for i in range(5)
        ]

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch.dict(os.environ, {"LANCEDB_BATCH_SIZE": "2"}, clear=False),
        ):
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
                create_index=False,
            )

        assert result.upsert_count == 5
        assert mock_embeddings_table.merge_insert.call_count >= 2

    def test_write_vectors_batch_partial_failure(
        self, temp_lancedb_dir, test_collection
    ):
        """Test batch processing with partial failures."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Create multiple embeddings to trigger batch processing
        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text=f"text_{i}",
                chunk_hash="test_hash",
            )
            for i in range(5)
        ]

        # Mock merge_insert to fail for first batch, succeed for others
        call_count = 0

        def mock_merge_insert_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_merge_insert = MagicMock()
            mock_when_matched = MagicMock()
            mock_when_not_matched = MagicMock()
            mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
            mock_when_matched.when_not_matched_insert_all.return_value = (
                mock_when_not_matched
            )
            if call_count == 1:
                # First batch fails
                mock_when_not_matched.execute.side_effect = Exception("Batch 1 failed")
            else:
                # Other batches succeed
                mock_when_not_matched.execute.return_value = None
            return mock_merge_insert

        mock_embeddings_table.merge_insert.side_effect = mock_merge_insert_side_effect
        # add succeeds for fallback
        mock_embeddings_table.add.return_value = None
        mock_embeddings_table.count_rows.return_value = 0

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch.dict(os.environ, {"LANCEDB_BATCH_SIZE": "2"}),
        ):  # Small batch size
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
            )

            # Some batches should have succeeded
            assert result.upsert_count > 0

    def test_write_vectors_spill_error_reduces_batch_size(
        self, temp_lancedb_dir, test_collection
    ):
        """Test LanceDB spill error triggers batch size reduction."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Create embeddings to trigger batch processing
        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text=f"text_{i}",
                chunk_hash="test_hash",
            )
            for i in range(5)
        ]

        # Mock merge_insert to raise spill error
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        # Raise spill error
        mock_when_not_matched.execute.side_effect = Exception(
            "Spill has sent an error: memory limit exceeded"
        )
        # add also fails initially
        mock_embeddings_table.add.side_effect = Exception(
            "Spill has sent an error: memory limit exceeded"
        )

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch.dict(os.environ, {"LANCEDB_BATCH_SIZE": "100"}),
        ):  # Large batch size
            # Should handle spill error gracefully
            with pytest.raises(Exception):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=embeddings,
                )

    def test_write_vectors_schema_mismatch_drops_table(
        self, temp_lancedb_dir, test_collection
    ):
        """Test schema compatibility check and table dropping."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        # Create a list to track table names, so drop_table can modify it
        table_names_list = ["embeddings_test_model"]

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        # Use a property or method that can be modified
        mock_db_connection.table_names = MagicMock(return_value=table_names_list)

        # Mock existing table with different vector dimension
        # Create a proper schema with all required fields including metadata
        mock_vector_field = MagicMock()
        mock_vector_field.name = "vector"
        mock_vector_field.type.list_size = 3  # Different dimension

        mock_metadata_field = MagicMock()
        mock_metadata_field.name = "metadata"
        mock_user_id_field = MagicMock()
        mock_user_id_field.name = "user_id"

        # Create a custom schema class that is both iterable and has field() method
        class MockSchema:
            def __init__(self, fields):
                self._fields = fields
                self._field_dict = {f.name: f for f in fields}

            def __iter__(self):
                return iter(self._fields)

            def field(self, name):
                return self._field_dict.get(name)

        mock_schema = MockSchema(
            [mock_vector_field, mock_metadata_field, mock_user_id_field]
        )
        mock_embeddings_table.schema = mock_schema

        # Mock drop_table to remove table from list
        def mock_drop_table(table_name):
            if table_name in table_names_list:
                table_names_list.remove(table_name)

        mock_db_connection.drop_table = MagicMock(side_effect=mock_drop_table)

        # Mock merge_insert chain
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        mock_when_not_matched.execute.return_value = None
        mock_embeddings_table.count_rows.return_value = 0

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],  # 2 dimensions, different from existing 3
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
            )

            # Verify table was dropped due to dimension mismatch
            mock_db_connection.drop_table.assert_called_once_with(
                "embeddings_test_model"
            )
            assert result.upsert_count == 1

    def test_write_vectors_inconsistent_dimensions(
        self, temp_lancedb_dir, test_collection
    ):
        """Test vector dimension inconsistency detection."""
        from unittest.mock import patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            VectorValidationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_1",
                chunk_id="chunk_1",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],  # 2 dimensions
                text="text_1",
                chunk_hash="test_hash",
            ),
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_2",
                chunk_id="chunk_2",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2, 0.3],  # 3 dimensions - inconsistent!
                text="text_2",
                chunk_hash="test_hash",
            ),
        ]

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env"
        ):
            with pytest.raises(
                VectorValidationError, match="Multiple vector dimensions found"
            ):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=embeddings,
                )

    def test_write_vectors_index_creation_failure(
        self, temp_lancedb_dir, test_collection
    ):
        """Test index creation failure handling."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Mock merge_insert chain
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        mock_when_not_matched.execute.return_value = None
        mock_embeddings_table.count_rows.return_value = 0

        # Mock index manager to fail
        mock_index_manager = MagicMock()
        mock_index_manager.check_and_create_index.side_effect = Exception(
            "Index creation failed"
        )

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_index_manager",
                return_value=mock_index_manager,
            ),
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
                create_index=True,
            )

            # Should still succeed but with failed index status
            assert result.upsert_count == 1
            assert result.index_status == "failed"

    def test_write_vectors_empty_collection_name(self, temp_lancedb_dir):
        """Test empty collection name validation."""
        from unittest.mock import patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DocumentValidationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        embedding = ChunkEmbeddingData(
            collection="",
            doc_id="test_doc",
            chunk_id="test_chunk",
            parse_hash="test_parse",
            model="test_model",
            vector=[0.1, 0.2],
            text="test text",
            chunk_hash="test_hash",
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env"
        ):
            with pytest.raises(
                DocumentValidationError, match="Collection name is required"
            ):
                write_vectors_to_db(
                    collection="",
                    embeddings=[embedding],
                )

    def test_write_vectors_multiple_models(self, temp_lancedb_dir, test_collection):
        """Test processing multiple models separately."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table_1 = _create_mock_table_with_schema()
        mock_embeddings_table_2 = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name == "embeddings_model_1":
                return mock_embeddings_table_1
            elif table_name == "embeddings_model_2":
                return mock_embeddings_table_2
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Mock merge_insert for both tables
        def create_mock_merge_insert_chain():
            mock_merge_insert = MagicMock()
            mock_when_matched = MagicMock()
            mock_when_not_matched = MagicMock()
            mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
            mock_when_matched.when_not_matched_insert_all.return_value = (
                mock_when_not_matched
            )
            mock_when_not_matched.execute.return_value = None
            return mock_merge_insert

        mock_embeddings_table_1.merge_insert.return_value = (
            create_mock_merge_insert_chain()
        )
        mock_embeddings_table_2.merge_insert.return_value = (
            create_mock_merge_insert_chain()
        )
        mock_embeddings_table_1.count_rows.return_value = 0
        mock_embeddings_table_2.count_rows.return_value = 0

        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_1",
                chunk_id="chunk_1",
                parse_hash="test_parse",
                model="model_1",
                vector=[0.1, 0.2],
                text="text_1",
                chunk_hash="test_hash",
            ),
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_2",
                chunk_id="chunk_2",
                parse_hash="test_parse",
                model="model_2",
                vector=[0.3, 0.4],
                text="text_2",
                chunk_hash="test_hash",
            ),
        ]

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
            return_value=mock_db_connection,
        ):
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
            )

            # Both models should be processed
            assert result.upsert_count == 2
            # Verify both tables were accessed
            mock_embeddings_table_1.merge_insert.assert_called_once()
            mock_embeddings_table_2.merge_insert.assert_called_once()

    def test_write_vectors_batch_size_from_env(self, temp_lancedb_dir, test_collection):
        """Test batch size configuration from environment variable."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Create enough embeddings to trigger multiple batches
        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text=f"text_{i}",
                chunk_hash="test_hash",
            )
            for i in range(5)
        ]

        # Mock merge_insert chain
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        mock_when_not_matched.execute.return_value = None
        mock_embeddings_table.count_rows.return_value = 0

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch.dict(os.environ, {"LANCEDB_BATCH_SIZE": "2"}),
        ):  # Custom batch size
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
            )

            # Should process all embeddings
            assert result.upsert_count == 5
            # With batch size 2, should have multiple merge_insert calls
            assert mock_embeddings_table.merge_insert.call_count >= 2

    def test_write_vectors_index_status_aggregation(
        self, temp_lancedb_dir, test_collection
    ):
        """Test index status aggregation for multiple models."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock()
        mock_embeddings_table_1 = _create_mock_table_with_schema()
        mock_embeddings_table_2 = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name == "embeddings_model_1":
                return mock_embeddings_table_1
            elif table_name == "embeddings_model_2":
                return mock_embeddings_table_2
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Mock merge_insert chains
        def create_mock_merge_insert_chain():
            mock_merge_insert = MagicMock()
            mock_when_matched = MagicMock()
            mock_when_not_matched = MagicMock()
            mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
            mock_when_matched.when_not_matched_insert_all.return_value = (
                mock_when_not_matched
            )
            mock_when_not_matched.execute.return_value = None
            return mock_merge_insert

        mock_embeddings_table_1.merge_insert.return_value = (
            create_mock_merge_insert_chain()
        )
        mock_embeddings_table_2.merge_insert.return_value = (
            create_mock_merge_insert_chain()
        )
        mock_embeddings_table_1.count_rows.return_value = 0
        mock_embeddings_table_2.count_rows.return_value = 0

        # Mock index manager with different statuses
        mock_index_manager = MagicMock()
        # First model: index_building, second model: failed
        mock_index_manager.check_and_create_index.side_effect = [
            ("index_building", "Building"),
            ("failed", "Failed"),
        ]

        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_1",
                chunk_id="chunk_1",
                parse_hash="test_parse",
                model="model_1",
                vector=[0.1, 0.2],
                text="text_1",
                chunk_hash="test_hash",
            ),
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_2",
                chunk_id="chunk_2",
                parse_hash="test_parse",
                model="model_2",
                vector=[0.3, 0.4],
                text="text_2",
                chunk_hash="test_hash",
            ),
        ]

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_index_manager",
                return_value=mock_index_manager,
            ),
        ):
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
                create_index=True,
            )

            # index_building should take priority over failed
            assert result.index_status == "created"


class TestVectorValidation:
    """Test cases for vector validation functionality."""

    def test_validate_query_vector_valid(self):
        """Test validation of valid query vectors."""
        # Test valid vectors
        validate_query_vector([1.0, 2.0, 3.0])
        validate_query_vector([0.5, -0.5, 0.0])
        validate_query_vector([1, 2, 3])  # integers are valid

    def test_validate_query_vector_invalid_type(self):
        """Test validation with invalid types."""
        with pytest.raises(VectorValidationError, match="query_vector must be a list"):
            validate_query_vector("not a list")

        with pytest.raises(VectorValidationError, match="query_vector must be a list"):
            validate_query_vector(None)

    def test_validate_query_vector_empty(self):
        """Test validation of empty vector."""
        with pytest.raises(VectorValidationError, match="query_vector cannot be empty"):
            validate_query_vector([])

    def test_validate_query_vector_invalid_elements(self):
        """Test validation with invalid vector elements."""
        with pytest.raises(
            VectorValidationError, match="query_vector must contain only numbers"
        ):
            validate_query_vector([1.0, "invalid", 3.0])

        with pytest.raises(
            VectorValidationError, match="query_vector must contain only numbers"
        ):
            validate_query_vector([1.0, None, 3.0])

    def test_validate_query_vector_nan_infinity(self):
        """Test validation with NaN and infinity values."""

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

    def test_validate_query_vector_numpy_scalar_types(self):
        """Test validation with numpy scalar types (np.int32, np.float64, etc.)."""
        try:
            import numpy as np

            # Test with numpy scalar types - should pass validation
            validate_query_vector([np.float64(1.0), np.float32(2.0), np.int32(3)])
            validate_query_vector([np.float64(0.5), np.float32(-0.5), np.int64(0)])
            validate_query_vector([np.float64(1.0), 2.0, np.int32(3)])  # Mixed types

            # Test with numpy array elements (should also work)
            validate_query_vector([np.float64(1.0), np.float64(2.0), np.float64(3.0)])

        except ImportError:
            pytest.skip("numpy not available")


class TestValidateQueryVectorExtended:
    """Test extended validate_query_vector functionality with model and dimension validation."""

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

    def test_validate_without_connection(self):
        """Test validation without database connection (backward compatibility)."""
        # Should work without model_tag and conn parameters
        validate_query_vector([1.0, 2.0, 3.0])

        # Should work with model_tag but no conn
        validate_query_vector([1.0, 2.0, 3.0], model_tag="test_model")

    def test_model_validation_invalid_format(self, temp_lancedb_dir):
        """Test model validation with invalid model_tag format."""
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            VectorValidationError,
        )
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            validate_embed_model,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()

        # Invalid characters in model_tag
        with pytest.raises(VectorValidationError, match="Invalid model_tag format"):
            validate_embed_model(conn, "invalid@model")

        with pytest.raises(VectorValidationError, match="Invalid model_tag format"):
            validate_embed_model(conn, "model with spaces")

        # Valid format with hyphen should not raise exception
        # (This will fail because table doesn't exist, but not due to format)
        with pytest.raises(VectorValidationError, match="does not exist"):
            validate_embed_model(conn, "model-with-dash")

    def test_model_validation_table_not_exists(self, temp_lancedb_dir):
        """Test model validation when table doesn't exist."""
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            VectorValidationError,
        )
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            validate_embed_model,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()

        # Table doesn't exist
        try:
            validate_embed_model(conn, "nonexistent_model")
            assert False, "Expected VectorValidationError to be raised"
        except VectorValidationError:
            pass  # Expected

    def test_dimension_validation_mismatch(self, temp_lancedb_dir, test_collection):
        """Test dimension validation when query vector dimension doesn't match stored."""
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            get_stored_vector_dimension,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        model_tag = "test_model"

        # Create embeddings table
        ensure_embeddings_table(conn, model_tag)

        # Manually insert a record with known dimension
        table = conn.open_table(f"embeddings_{model_tag}")
        import pandas as pd

        test_record = {
            "collection": test_collection,
            "doc_id": "test_doc",
            "chunk_id": "test_chunk",
            "parse_hash": "test_parse",
            "model": model_tag,
            "vector": [1.0, 2.0, 3.0, 4.0],  # 4 dimensions
            "vector_dimension": 4,
            "text": "test text",
            "chunk_hash": "test_hash",
            "created_at": pd.Timestamp.now(tz="UTC"),
            "metadata": "{}",
            "user_id": None,
        }
        table.add([test_record])

        # Test dimension retrieval
        stored_dim = get_stored_vector_dimension(
            conn, model_tag, user_id=None, is_admin=True
        )
        assert stored_dim == 4

        # Test dimension validation - should pass
        validate_query_vector(
            [0.1, 0.2, 0.3, 0.4], model_tag, conn=conn, user_id=None, is_admin=True
        )

        # Test dimension validation - should fail
        with pytest.raises(
            VectorValidationError,
            match="Query vector dimension 3 does not match stored dimension 4",
        ):
            validate_query_vector(
                [0.1, 0.2, 0.3], model_tag, conn=conn, user_id=None, is_admin=True
            )

    def test_dimension_validation_no_data(self, temp_lancedb_dir):
        """Test dimension validation when table exists but has no data."""
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            get_stored_vector_dimension,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        model_tag = "empty_model"

        # Create empty embeddings table
        ensure_embeddings_table(conn, model_tag)

        # Should return None when no data
        stored_dim = get_stored_vector_dimension(conn, model_tag)
        assert stored_dim is None

        # Validation should pass when no stored dimension
        validate_query_vector([0.1, 0.2, 0.3], model_tag, conn=conn)

    def test_full_validation_integration(self, temp_lancedb_dir, test_collection):
        """Test full validation integration with model and dimension checks."""
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        model_tag = "integration_test_model"

        # Create table and add test data
        ensure_embeddings_table(conn, model_tag)
        table = conn.open_table(f"embeddings_{model_tag}")

        import pandas as pd

        test_record = {
            "collection": test_collection,
            "doc_id": "test_doc",
            "chunk_id": "test_chunk",
            "parse_hash": "test_parse",
            "model": model_tag,
            "vector": [1.0, 2.0],  # 2 dimensions
            "vector_dimension": 2,
            "text": "test text",
            "chunk_hash": "test_hash",
            "created_at": pd.Timestamp.now(tz="UTC"),
            "metadata": "{}",
            "user_id": None,
        }
        table.add([test_record])

        # Test successful validation
        validate_query_vector(
            [0.5, 0.7], model_tag, conn=conn, user_id=None, is_admin=True
        )

        # Test model validation failure - model_tag is normalized by to_model_tag(),
        # so "invalid@model" becomes "invalid_model", then fails because table doesn't exist
        with pytest.raises(
            VectorValidationError, match="does not exist or is inaccessible"
        ):
            validate_query_vector(
                [0.5, 0.7], "invalid@model", conn=conn, user_id=None, is_admin=True
            )

        # Test dimension mismatch failure
        with pytest.raises(VectorValidationError, match="dimension 3 does not match"):
            validate_query_vector(
                [0.5, 0.7, 0.9], model_tag, conn=conn, user_id=None, is_admin=True
            )


class TestReindexingFunctionality:
    """Test cases for reindexing functionality."""

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

    def test_should_reindex_batch_threshold(self):
        """Test reindex decision based on batch size threshold."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _should_reindex,
        )

        mock_table = MagicMock()
        policy = IndexPolicy(reindex_batch_size=100)

        # Test batch threshold
        assert _should_reindex(mock_table, "test_table", 150, policy) is True
        assert _should_reindex(mock_table, "test_table", 50, policy) is False

    def test_should_reindex_immediate_mode(self):
        """Test immediate reindex mode."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _should_reindex,
        )

        mock_table = MagicMock()
        policy = IndexPolicy(enable_immediate_reindex=True, reindex_batch_size=1000)

        # Test immediate reindex
        assert _should_reindex(mock_table, "test_table", 1, policy) is True
        assert _should_reindex(mock_table, "test_table", 0, policy) is False

    def test_should_reindex_smart_mode(self):
        """Test smart reindex mode based on unindexed ratio."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _should_reindex,
        )

        mock_table = MagicMock()
        policy = IndexPolicy(
            enable_smart_reindex=True, reindex_unindexed_ratio_threshold=0.05
        )

        # Mock index stats
        mock_stats = MagicMock()
        mock_stats.num_indexed_rows = 1000
        mock_stats.num_unindexed_rows = 60  # 6% > 5% threshold
        mock_table.index_stats.return_value = mock_stats

        assert _should_reindex(mock_table, "test_table", 10, policy) is True

        # Test below threshold
        mock_stats.num_unindexed_rows = 30  # 3% < 5% threshold
        assert _should_reindex(mock_table, "test_table", 10, policy) is False

    def test_trigger_reindex_success(self):
        """Test successful reindex trigger."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _trigger_reindex,
        )

        mock_table = MagicMock()
        mock_table.optimize.return_value = None

        result = _trigger_reindex(mock_table, "test_table")

        assert result is True
        mock_table.optimize.assert_called_once()

    def test_trigger_reindex_failure(self):
        """Test reindex trigger failure."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            _trigger_reindex,
        )

        mock_table = MagicMock()
        mock_table.optimize.side_effect = Exception("Optimize failed")

        result = _trigger_reindex(mock_table, "test_table")

        assert result is False
        mock_table.optimize.assert_called_once()

    def test_write_vectors_with_reindex_integration(
        self, temp_lancedb_dir, test_collection
    ):
        """Test write_vectors_to_db with reindex integration."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock connection and table
        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None

        # Mock merge_insert chain
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_execute = MagicMock()

        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        mock_when_not_matched.execute.return_value = mock_execute

        # Mock index manager
        mock_index_manager = MagicMock()
        mock_index_manager.check_and_create_index.return_value = (
            "index_building",
            "Index created",
        )

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_index_manager",
                return_value=mock_index_manager,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager._should_reindex",
                return_value=True,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager._trigger_reindex",
                return_value=True,
            ),
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
                create_index=True,
            )

            # Verify index manager was called
            mock_index_manager.check_and_create_index.assert_called_once()

            # Verify reindex was triggered
            from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
                _trigger_reindex,
            )

            _trigger_reindex.assert_called_once()

            assert result.upsert_count == 1
            assert result.index_status == "created"

    def test_write_vectors_reindex_policy_configuration(
        self, temp_lancedb_dir, test_collection
    ):
        """Test write_vectors_to_db with different reindex policy configurations."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Test with custom policy
        custom_policy = IndexPolicy(
            reindex_batch_size=500,
            enable_immediate_reindex=True,
            enable_smart_reindex=False,
        )

        mock_db_connection = MagicMock()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None

        # Mock merge_insert chain
        mock_merge_insert = MagicMock()
        mock_when_matched = MagicMock()
        mock_when_not_matched = MagicMock()
        mock_execute = MagicMock()

        mock_embeddings_table.merge_insert.return_value = mock_merge_insert
        mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
        mock_when_matched.when_not_matched_insert_all.return_value = (
            mock_when_not_matched
        )
        mock_when_not_matched.execute.return_value = mock_execute

        # Mock index manager
        mock_index_manager = MagicMock()
        mock_index_manager.check_and_create_index.return_value = (
            "index_building",
            "Index created",
        )

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_index_manager",
                return_value=mock_index_manager,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.IndexPolicy",
                return_value=custom_policy,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager._should_reindex",
                return_value=True,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager._trigger_reindex",
                return_value=True,
            ),
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
                create_index=True,
            )

            assert result.upsert_count == 1
            assert result.index_status == "created"

    def test_read_chunks_arrow_fallback_chain(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test read_chunks_for_embedding three-tier fallback: to_arrow() -> to_list() -> to_pandas()."""
        from unittest.mock import MagicMock, patch

        mock_db_connection = MagicMock()
        mock_chunks_table = _create_mock_table_with_schema()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name == "chunks":
                return mock_chunks_table
            elif table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return MagicMock()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None

        # Test case 1: to_arrow() works
        chunks_data = [
            {
                "chunk_id": "chunk1",
                "text": "test content",
                "collection": test_collection,
                "doc_id": "doc1",
                "parse_hash": "hash1",
                "index": 0,
                "chunk_hash": "test_hash",
                "metadata": '{"key": "value"}',
            }
        ]
        mock_arrow_table = MagicMock()
        mock_arrow_table.to_pylist.return_value = chunks_data

        mock_chunks_search = MagicMock()
        mock_chunks_where = MagicMock()
        mock_chunks_table.search.return_value = mock_chunks_search
        mock_chunks_search.where.return_value = mock_chunks_where
        mock_chunks_where.to_arrow.return_value = mock_arrow_table
        mock_chunks_table.count_rows.return_value = 1

        # Mock embeddings table (empty)
        mock_embeddings_search = MagicMock()
        mock_embeddings_where = MagicMock()
        mock_embeddings_select = MagicMock()
        mock_embeddings_table.search.return_value = mock_embeddings_search
        mock_embeddings_search.where.return_value = mock_embeddings_where
        mock_embeddings_where.select.return_value = mock_embeddings_select
        mock_embeddings_arrow_table = MagicMock()
        mock_embeddings_arrow_table.to_pylist.return_value = []
        mock_embeddings_select.to_arrow.return_value = mock_embeddings_arrow_table
        mock_embeddings_table.count_rows.return_value = 0

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_chunks_table"
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_embeddings_table"
            ),
        ):
            result = read_chunks_for_embedding(
                collection=test_collection,
                doc_id="doc1",
                parse_hash="hash1",
                model="test_model",
            )

            assert result.total_count == 1
            assert len(result.chunks) == 1
            # Verify to_arrow() was called
            mock_chunks_where.to_arrow.assert_called_once()

    def test_read_chunks_fallback_to_list(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test read_chunks_for_embedding fallback from to_arrow() to to_list()."""
        from unittest.mock import MagicMock, patch

        mock_db_connection = MagicMock()
        mock_chunks_table = _create_mock_table_with_schema()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name == "chunks":
                return mock_chunks_table
            elif table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return MagicMock()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None

        chunks_data = [
            {
                "chunk_id": "chunk1",
                "text": "test content",
                "collection": test_collection,
                "doc_id": "doc1",
                "parse_hash": "hash1",
                "index": 0,
                "chunk_hash": "test_hash",
                "metadata": '{"key": "value"}',
            }
        ]

        mock_chunks_search = MagicMock()
        mock_chunks_where = MagicMock()
        mock_chunks_table.search.return_value = mock_chunks_search
        mock_chunks_search.where.return_value = mock_chunks_where
        # to_arrow() fails, fallback to to_list()
        mock_chunks_where.to_arrow.side_effect = AttributeError(
            "to_arrow not available"
        )
        mock_chunks_where.to_list.return_value = chunks_data
        mock_chunks_table.count_rows.return_value = 1

        # Mock embeddings table (empty)
        mock_embeddings_search = MagicMock()
        mock_embeddings_where = MagicMock()
        mock_embeddings_select = MagicMock()
        mock_embeddings_table.search.return_value = mock_embeddings_search
        mock_embeddings_search.where.return_value = mock_embeddings_where
        mock_embeddings_where.select.return_value = mock_embeddings_select
        mock_embeddings_select.to_arrow.side_effect = AttributeError(
            "to_arrow not available"
        )
        mock_embeddings_select.to_list.return_value = []
        mock_embeddings_table.count_rows.return_value = 0

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_chunks_table"
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_embeddings_table"
            ),
        ):
            result = read_chunks_for_embedding(
                collection=test_collection,
                doc_id="doc1",
                parse_hash="hash1",
                model="test_model",
            )

            assert result.total_count == 1
            assert len(result.chunks) == 1
            # Verify fallback was used
            mock_chunks_where.to_arrow.assert_called_once()
            mock_chunks_where.to_list.assert_called_once()

    def test_read_chunks_fallback_to_pandas_with_nan(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test read_chunks_for_embedding fallback to to_pandas() and NaN normalization."""
        from unittest.mock import MagicMock, patch

        import numpy as np

        mock_db_connection = MagicMock()
        mock_chunks_table = _create_mock_table_with_schema()
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name == "chunks":
                return mock_chunks_table
            elif table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return MagicMock()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None

        # Create DataFrame with NaN values
        chunks_df = pd.DataFrame(
            [
                {
                    "chunk_id": "chunk1",
                    "text": "test content",
                    "collection": test_collection,
                    "doc_id": "doc1",
                    "parse_hash": "hash1",
                    "index": 0,
                    "chunk_hash": "test_hash",
                    "metadata": '{"key": "value"}',
                    "page_number": np.nan,  # NaN value
                }
            ]
        )

        mock_chunks_search = MagicMock()
        mock_chunks_where = MagicMock()
        mock_chunks_table.search.return_value = mock_chunks_search
        mock_chunks_search.where.return_value = mock_chunks_where
        # Both to_arrow() and to_list() fail, fallback to to_pandas()
        mock_chunks_where.to_arrow.side_effect = AttributeError(
            "to_arrow not available"
        )
        mock_chunks_where.to_list.side_effect = AttributeError("to_list not available")
        mock_chunks_where.to_pandas.return_value = chunks_df
        mock_chunks_table.count_rows.return_value = 1

        # Mock embeddings table (empty)
        mock_embeddings_search = MagicMock()
        mock_embeddings_where = MagicMock()
        mock_embeddings_select = MagicMock()
        mock_embeddings_table.search.return_value = mock_embeddings_search
        mock_embeddings_search.where.return_value = mock_embeddings_where
        mock_embeddings_where.select.return_value = mock_embeddings_select
        mock_embeddings_select.to_arrow.side_effect = AttributeError(
            "to_arrow not available"
        )
        mock_embeddings_select.to_list.side_effect = AttributeError(
            "to_list not available"
        )
        mock_embeddings_select.to_pandas.return_value = pd.DataFrame()
        mock_embeddings_table.count_rows.return_value = 0

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_chunks_table"
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_embeddings_table"
            ),
        ):
            result = read_chunks_for_embedding(
                collection=test_collection,
                doc_id="doc1",
                parse_hash="hash1",
                model="test_model",
            )

            assert result.total_count == 1
            assert len(result.chunks) == 1
            # Verify all fallbacks were attempted
            mock_chunks_where.to_arrow.assert_called_once()
            mock_chunks_where.to_list.assert_called_once()
            mock_chunks_where.to_pandas.assert_called_once()
            # Verify NaN was normalized to None (page_number should be None, not NaN)
            assert result.chunks[0].page_number is None

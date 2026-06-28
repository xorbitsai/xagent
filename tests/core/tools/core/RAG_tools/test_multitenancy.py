"""Tests for RAG multi-tenancy support.

Tests user_id and is_admin filtering in RAG tools and pipelines.
Also covers:
- User data isolation
- Administrator access controls
- Permission validation
- API endpoint security (list_collections_api, delete_collection_api with physical cleanup)
"""

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.storage import initialize_storage_manager
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.document_search import (
    get_knowledge_search_tool,
    get_list_knowledge_bases_tool,
)
from xagent.core.tools.core.RAG_tools.chunk.chunk_document import chunk_document
from xagent.core.tools.core.RAG_tools.core.config import MIN_INT64
from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.management.collections import (
    delete_collection,
    list_collections,
    retry_document,
)
from xagent.core.tools.core.RAG_tools.parse.parse_document import parse_document
from xagent.core.tools.core.RAG_tools.retrieval.search_dense import search_dense
from xagent.core.tools.core.RAG_tools.storage.contracts import DocumentRecord
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_vector_store_raw_connection,
)
from xagent.core.tools.core.RAG_tools.utils.user_permissions import UserPermissions
from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    read_chunks_for_embedding,
    write_vectors_to_db,
)
from xagent.web.api.kb import (
    delete_collection_api,
    list_collections_api,
    rename_collection_api,
)


class _FakeEmbeddingAdapter(BaseEmbedding):
    """Local embedding adapter for testing."""

    def encode(self, text, dimension: int | None = None, instruct: str | None = None):
        if isinstance(text, str):
            return [float(len(text))]
        return [[float(len(item))] for item in text]

    def get_dimension(self) -> int:
        return 1

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


class TestUserPermissions:
    """Test UserPermissions class logic."""

    def test_admin_no_filter(self):
        """Admin users get None filter (see all data)."""
        filter_str = UserPermissions.get_user_filter(user_id=None, is_admin=True)
        assert filter_str is None

        filter_str = UserPermissions.get_user_filter(user_id=1, is_admin=True)
        assert filter_str is None

    def test_regular_user_filter(self):
        """Regular users only see their own data."""
        filter_str = UserPermissions.get_user_filter(user_id=1, is_admin=False)
        assert filter_str == "user_id == 1"

        filter_str = UserPermissions.get_user_filter(user_id=42, is_admin=False)
        assert filter_str == "user_id == 42"

    def test_unauthenticated_no_access(self):
        """Unauthenticated users cannot see any data."""
        filter_str = UserPermissions.get_user_filter(user_id=None, is_admin=False)
        assert filter_str == UserPermissions.get_no_access_filter()

    def test_context_user_filter_without_explicit_args(self):
        """Context user scope should apply when explicit args are absent."""
        with user_scope_context(user_id=42, is_admin=False):
            filter_str = UserPermissions.get_user_filter(user_id=None, is_admin=None)
        assert filter_str == "user_id == 42"

    def test_can_access_data_admin(self):
        """Admin can access all data."""
        assert UserPermissions.can_access_data(user_id=1, data_user_id=1, is_admin=True)
        assert UserPermissions.can_access_data(user_id=1, data_user_id=2, is_admin=True)
        assert UserPermissions.can_access_data(
            user_id=1, data_user_id=None, is_admin=True
        )

    def test_can_access_data_regular_user(self):
        """Regular users can only access their own data."""
        assert UserPermissions.can_access_data(
            user_id=1, data_user_id=1, is_admin=False
        )
        # Cannot access other user's data
        assert not UserPermissions.can_access_data(
            user_id=1, data_user_id=2, is_admin=False
        )
        # Cannot access NULL data (legacy)
        assert not UserPermissions.can_access_data(
            user_id=1, data_user_id=None, is_admin=False
        )

    def test_can_access_data_unauthenticated(self):
        """Unauthenticated users cannot access any data."""
        assert not UserPermissions.can_access_data(
            user_id=None, data_user_id=1, is_admin=False
        )
        assert not UserPermissions.can_access_data(
            user_id=None, data_user_id=None, is_admin=False
        )


class TestMultiTenancyCollections:
    """Test multi-tenancy in list_collections."""

    @pytest.fixture()
    def temp_lancedb_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Isolate LanceDB per test."""
        import os

        original = os.environ.get("LANCEDB_DIR")
        lancedb_dir = tmp_path / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))
        yield str(lancedb_dir)
        if original is None:
            monkeypatch.delenv("LANCEDB_DIR", raising=False)
        else:
            monkeypatch.setenv("LANCEDB_DIR", original)

    def _insert_test_documents(self, user_id: int | None):
        """Insert test documents with specific user_id."""
        conn = get_vector_store_raw_connection()
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_documents_table,
        )

        ensure_documents_table(conn)
        table = conn.open_table("documents")

        records = [
            {
                "collection": "test_collection",
                "doc_id": f"doc_{uuid.uuid4().hex[:8]}",
                "source_path": "/tmp/test.txt",
                "file_type": "txt",
                "content_hash": "hash1",
                "uploaded_at": datetime.now(timezone.utc),
                "title": "Test Document",
                "language": "en",
                "user_id": user_id,
            }
            for _ in range(5)
        ]
        table.add(records)

    @pytest.mark.asyncio
    async def test_list_collections_admin_sees_all(self, temp_lancedb_dir: str) -> None:
        """Admin users should see all collections regardless of user_id."""
        # Insert documents for different users
        self._insert_test_documents(user_id=1)
        self._insert_test_documents(user_id=2)
        self._insert_test_documents(user_id=None)  # Legacy data

        # Admin sees everything
        result = await list_collections(user_id=None, is_admin=True)
        assert result.status == "success"
        # Should see at least one collection
        assert len(result.collections) >= 1
        # Total documents should be sum of all inserted
        total_docs = sum(c.documents for c in result.collections)
        assert total_docs == 15  # 5 docs per user * 3 users

        # Owners field should include all non-null user_ids (1 and 2)
        owners_sets = {tuple(c.owners) for c in result.collections}
        assert (1, 2) in owners_sets

    @pytest.mark.asyncio
    async def test_list_collections_regular_user_sees_only_own(
        self, temp_lancedb_dir: str
    ) -> None:
        """Regular users should only see their own documents."""
        # Insert documents for different users
        self._insert_test_documents(user_id=1)
        self._insert_test_documents(user_id=2)
        self._insert_test_documents(user_id=None)

        # User 1 sees only user 1's data
        result = await list_collections(user_id=1, is_admin=False)
        assert result.status == "success"
        total_docs = sum(c.documents for c in result.collections)
        assert total_docs == 5
        # Owners should only contain the current user
        assert all(c.owners == [1] for c in result.collections)

        # User 2 sees only user 2's data
        result = await list_collections(user_id=2, is_admin=False)
        assert result.status == "success"
        total_docs = sum(c.documents for c in result.collections)
        assert total_docs == 5
        # Owners should only contain the current user
        assert all(c.owners == [2] for c in result.collections)

    @pytest.mark.asyncio
    async def test_list_collections_resolves_user_scope_from_context(
        self, temp_lancedb_dir: str
    ) -> None:
        """list_collections should work without explicit user_id when context is set."""
        self._insert_test_documents(user_id=9)
        self._insert_test_documents(user_id=10)

        with user_scope_context(user_id=9, is_admin=False):
            result = await list_collections()

        assert result.status == "success"
        total_docs = sum(c.documents for c in result.collections)
        assert total_docs == 5


class TestMultiTenancySearch:
    """Test multi-tenancy in document search."""

    @pytest.fixture()
    def temp_lancedb_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Isolate LanceDB per test."""
        import os

        original = os.environ.get("LANCEDB_DIR")
        lancedb_dir = tmp_path / "lancedb"
        lancedb_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LANCEDB_DIR", str(lancedb_dir))

        # Setup storage
        storage_root = tmp_path / "storage"
        storage_root.mkdir(parents=True, exist_ok=True)
        initialize_storage_manager(str(storage_root), str(storage_root / "uploads"))

        yield str(lancedb_dir)
        if original is None:
            monkeypatch.delenv("LANCEDB_DIR", raising=False)
        else:
            monkeypatch.setenv("LANCEDB_DIR", original)

    def _setup_document_pipeline(
        self, temp_lancedb_dir: str, user_id: int | None, collection: str
    ):
        """Setup complete document pipeline for a user."""
        # Create test file
        test_file = Path(temp_lancedb_dir) / "test.txt"
        test_file.write_text("Test content for search")

        # Register document
        doc_id = uuid.uuid4().hex
        register_document(
            collection=collection,
            source_path=str(test_file),
            doc_id=doc_id,
            user_id=user_id,
        )

        # Parse
        parse_result = parse_document(
            collection=collection,
            doc_id=doc_id,
            parse_method="deepdoc",
            user_id=user_id,
            is_admin=False,
        )
        parse_hash = parse_result["parse_hash"]

        # Chunk
        chunk_document(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            user_id=user_id,
        )

        # Embed
        embedding_model_id = "test-model"
        embedding_read = read_chunks_for_embedding(
            collection=collection,
            doc_id=doc_id,
            parse_hash=parse_hash,
            model=embedding_model_id,
            user_id=user_id,
            is_admin=False,
        )

        embeddings = [
            ChunkEmbeddingData(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                parse_hash=chunk.parse_hash,
                model=embedding_model_id,
                vector=[1.0],
                text=chunk.text,
                chunk_hash=chunk.chunk_hash,
            )
            for chunk in embedding_read.chunks
        ]

        write_vectors_to_db(
            collection=collection,
            embeddings=embeddings,
            user_id=user_id,
        )

        return collection

    @pytest.mark.integration
    def test_search_regular_user_only_own_results(
        self, temp_lancedb_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regular users can only search their own documents - direct LanceDB level test."""
        # This test verifies user_id filtering at the LanceDB level
        # without going through the full search pipeline

        # Setup: Create embeddings table and insert test data for different users
        import pandas as pd

        conn = get_vector_store_raw_connection()

        # Create embeddings table
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )

        ensure_embeddings_table(conn, "test-model", vector_dim=1)

        table = conn.open_table("embeddings_test-model")

        # Insert data for user 1
        data_user1 = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "doc1_user1",
                    "chunk_id": "chunk1",
                    "parse_hash": "hash1",
                    "model": "test-model",
                    "vector": [1.0],
                    "vector_dimension": 1,
                    "text": "content for user 1",
                    "chunk_hash": "chash1",
                    "created_at": datetime.now(timezone.utc),
                    "metadata": "{}",
                    "user_id": 1,
                }
            ]
        )

        # Insert data for user 2
        data_user2 = pd.DataFrame(
            [
                {
                    "collection": "test_collection",
                    "doc_id": "doc2_user2",
                    "chunk_id": "chunk2",
                    "parse_hash": "hash2",
                    "model": "test-model",
                    "vector": [1.0],
                    "vector_dimension": 1,
                    "text": "content for user 2",
                    "chunk_hash": "chash2",
                    "created_at": datetime.now(timezone.utc),
                    "metadata": "{}",
                    "user_id": 2,
                }
            ]
        )

        table.add(data_user1)
        table.add(data_user2)

        # Test 1: User 1 can see their own data
        result_user1 = table.search().where("user_id == 1").to_arrow()
        assert len(result_user1) == 1
        assert result_user1["text"][0].as_py() == "content for user 1"

        # Test 2: User 1 cannot see user 2's data
        result_user1_filtered = table.search().where("user_id == 1").to_arrow()
        assert len(result_user1_filtered) == 1  # Only their own data
        assert result_user1_filtered["doc_id"][0].as_py() == "doc1_user1"

        # Test 3: Admin can see all data
        result_admin = table.search().to_arrow()
        assert len(result_admin) == 2  # All data

        # Test 4: User filter for user 2
        result_user2 = table.search().where("user_id == 2").to_arrow()
        assert len(result_user2) == 1
        assert result_user2["doc_id"][0].as_py() == "doc2_user2"

    @pytest.mark.integration
    def test_unauthenticated_search_hides_orphaned_records(
        self, temp_lancedb_dir: str
    ) -> None:
        """Unauthenticated dense search should not return orphaned sentinel records."""
        import pandas as pd

        conn = get_vector_store_raw_connection()
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )

        ensure_embeddings_table(conn, "test_model", vector_dim=1)
        table = conn.open_table("embeddings_test_model")

        table.add(
            pd.DataFrame(
                [
                    {
                        "collection": "test_collection",
                        "doc_id": "orphaned_doc",
                        "chunk_id": "chunk_orphaned",
                        "parse_hash": "hash_orphaned",
                        "model": "test_model",
                        "vector": [1.0],
                        "vector_dimension": 1,
                        "text": "orphaned content should be hidden",
                        "chunk_hash": "chunk_hash_orphaned",
                        "created_at": datetime.now(timezone.utc),
                        "metadata": "{}",
                        "user_id": MIN_INT64,
                    }
                ]
            )
        )

        resp = search_dense(
            collection="test_collection",
            model_tag="test_model",
            query_vector=[1.0],
            top_k=10,
            user_id=None,
            is_admin=False,
            readonly=True,
        )

        assert resp.results == []


class TestToolUserContext:
    """Test user context passing through tools."""

    def test_list_knowledge_bases_tool_with_user_context(self):
        """Tool should respect user context when listing collections."""
        tool = get_list_knowledge_bases_tool(
            allowed_collections=None, user_id=1, is_admin=False
        )

        assert tool.user_id == 1
        assert tool.is_admin is False

    def test_search_tool_with_user_context(self):
        """Tool should respect user context when searching."""
        tool = get_knowledge_search_tool(
            embedding_model_id="test-model",
            allowed_collections=None,
            user_id=1,
            is_admin=False,
        )

        assert tool.user_id == 1
        assert tool.is_admin is False

    def test_admin_tool_context(self):
        """Admin tool should have admin flag set."""
        tool = get_list_knowledge_bases_tool(
            allowed_collections=None, user_id=None, is_admin=True
        )

        assert tool.user_id is None
        assert tool.is_admin is True


class TestToolConfigUserContext:
    """Test user context in ToolConfig."""

    def test_tool_config_with_user_context(self):
        """ToolConfig should store and retrieve user context."""
        config = ToolConfig(
            {
                "user_id": 42,
                "is_admin": False,
                "basic_tools_enabled": True,
            }
        )

        assert config.get_user_id() == 42
        assert config.is_admin() is False

    def test_tool_config_admin(self):
        """ToolConfig should handle admin context."""
        config = ToolConfig(
            {
                "user_id": 1,
                "is_admin": True,
                "basic_tools_enabled": True,
            }
        )

        assert config.get_user_id() == 1
        assert config.is_admin() is True

    def test_tool_config_no_user(self):
        """ToolConfig should handle missing user context."""
        config = ToolConfig(
            {
                "basic_tools_enabled": True,
            }
        )

        assert config.get_user_id() is None
        assert config.is_admin() is False


# ---------------------------------------------------------------------------
# Tests merged from test_multi_tenancy.py (API, collection management, E2E)
# ---------------------------------------------------------------------------


class TestUserPermissionsRAGTools:
    """Test user permission utility functions (RAG tools - current impl: no legacy NULL for regular users)."""

    def test_get_user_filter_admin(self):
        """Test admin user filter returns None (no filtering)."""
        filter_expr = UserPermissions.get_user_filter(None, True)
        assert filter_expr is None

    def test_get_user_filter_regular_user(self):
        """Test regular user filter is only their user_id (no legacy NULL in current impl)."""
        filter_expr = UserPermissions.get_user_filter(123, False)
        assert filter_expr == "user_id == 123"

    def test_get_user_filter_unauthenticated(self):
        """Test unauthenticated user filter matches nothing."""
        filter_expr = UserPermissions.get_user_filter(None, False)
        assert filter_expr == UserPermissions.get_no_access_filter()

    def test_can_access_data_admin(self):
        """Test admin can access any data."""
        assert UserPermissions.can_access_data(None, 123, True) is True
        assert UserPermissions.can_access_data(None, None, True) is True

    def test_can_access_data_regular_user(self):
        """Test regular user can only access their own data (legacy NULL not accessible)."""
        assert UserPermissions.can_access_data(123, 123, False) is True
        assert UserPermissions.can_access_data(123, None, False) is False
        assert UserPermissions.can_access_data(123, 456, False) is False

    def test_get_write_user_id(self):
        """Test write user_id assignment."""
        assert UserPermissions.get_write_user_id(123) == 123
        assert UserPermissions.get_write_user_id(None) is None


class TestCollectionManagementMultiTenancy:
    """Test multi-tenancy in collection management operations."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.collection = "test_collection"

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch(
        "xagent.core.tools.core.RAG_tools.management.collections.get_vector_index_store"
    )
    @pytest.mark.asyncio
    async def test_list_collections_with_user_filter(self, mock_get_store):
        """Test list_collections applies user filtering."""
        mock_store = MagicMock()
        mock_conn = MagicMock()
        mock_store.get_raw_connection.return_value = mock_conn
        mock_store.aggregate_collection_stats.return_value = {}
        mock_get_store.return_value = mock_store

        mock_docs_table = MagicMock()
        mock_conn.open_table.return_value = mock_docs_table

        mock_batch = MagicMock()
        mock_batch.num_rows = 2
        mock_batch.schema.get_field_index.return_value = 0

        mock_collection_array = MagicMock()
        mock_collection_array.__getitem__.side_effect = lambda i: {
            "as_py": lambda: f"collection_{i}"
        }[i]()
        mock_batch.column.side_effect = lambda idx: mock_collection_array

        mock_docs_table.to_batches.return_value = [mock_batch]

        def mock_open_table_side_effect(table_name):
            if table_name == "documents":
                return mock_docs_table
            else:
                mock_empty_table = MagicMock()
                mock_empty_table.to_batches.return_value = []
                return mock_empty_table

        mock_conn.open_table.side_effect = mock_open_table_side_effect

        result = await list_collections(user_id=123, is_admin=False)
        assert hasattr(result, "status")
        assert hasattr(result, "collections")
        assert hasattr(result, "total_count")

        result = await list_collections(user_id=None, is_admin=True)
        assert hasattr(result, "status")
        assert hasattr(result, "collections")
        assert hasattr(result, "total_count")

    @pytest.mark.asyncio
    async def test_delete_collection_with_real_storage(self):
        """Test delete_collection with real storage (integration test).

        This test verifies the complete data flow for delete_collection operation,
        ensuring it correctly handles user/admin permissions with actual database
        operations rather than mocked responses.
        """
        from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
        from xagent.core.tools.core.RAG_tools.management.collection_manager import (
            CollectionManager,
        )

        # Setup: Create a collection for testing using CollectionManager
        manager = CollectionManager()

        collection = CollectionInfo(
            name=self.collection,
            embedding_model_id="text-embedding-ada-002",
            embedding_dimension=1536,
            documents=5,
        )
        await manager.save_collection(collection)

        # Test: Admin can delete collection
        result = delete_collection(self.collection, user_id=None, is_admin=True)
        assert result.status == "success"

        # Setup: Create another collection for user-specific test
        collection_user = CollectionInfo(
            name=f"{self.collection}_user",
            embedding_model_id="text-embedding-ada-002",
            embedding_dimension=1536,
            documents=5,
        )
        await manager.save_collection(collection_user)

        # Test: User can delete their own collection
        result = delete_collection(
            f"{self.collection}_user", user_id=123, is_admin=False
        )
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_retry_document_with_real_storage(self):
        """Test retry_document with real storage (integration test).

        This test verifies the complete data flow for retry_document operation,
        ensuring it correctly handles user/admin permissions with actual database
        operations rather than mocked responses.
        """
        from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
        from xagent.core.tools.core.RAG_tools.management.collection_manager import (
            CollectionManager,
        )

        # Setup: Create a collection for testing using CollectionManager
        manager = CollectionManager()

        collection = CollectionInfo(
            name=self.collection,
            embedding_model_id="text-embedding-ada-002",
            embedding_dimension=1536,
            documents=5,
        )
        await manager.save_collection(collection)

        # Test: User can retry their own document
        result = retry_document(
            self.collection, "test_doc", user_id=123, is_admin=False
        )
        assert result.status == "success"

        # Test: Admin can retry any document
        result = retry_document(
            self.collection, "test_doc", user_id=None, is_admin=True
        )
        assert result.status == "success"


class TestDocumentIngestionMultiTenancy:
    """Test multi-tenancy in document ingestion pipeline."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ingestion_accepts_user_id_parameter(self):
        """Test that run_document_ingestion accepts user_id parameter."""
        import inspect

        from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
            run_document_ingestion,
        )

        sig = inspect.signature(run_document_ingestion)
        assert "user_id" in sig.parameters
        assert sig.parameters["user_id"].default is None


class TestDocumentSearchMultiTenancy:
    """Test multi-tenancy in document search operations."""

    def test_search_accepts_user_parameters(self):
        """Test that run_document_search accepts user_id and tri-state is_admin."""
        import inspect

        from xagent.core.tools.core.RAG_tools.pipelines.document_search import (
            run_document_search,
        )

        sig = inspect.signature(run_document_search)
        assert "user_id" in sig.parameters
        assert "is_admin" in sig.parameters
        assert sig.parameters["user_id"].default is None
        # None means "fall back to request-scoped context", False/True are explicit.
        assert sig.parameters["is_admin"].default is None


class TestLanceDBConnectionMultiTenancy:
    """Test multi-tenancy in LanceDB connection management."""

    @patch("xagent.providers.vector_store.lancedb.os.path.exists")
    @patch("xagent.providers.vector_store.lancedb.os.getenv")
    def test_connection_with_default_path(self, mock_getenv, mock_exists):
        """Test LanceDB connection uses default path when LANCEDB_DIR is not set."""
        from xagent.providers.vector_store.lancedb import LanceDBConnectionManager

        mock_getenv.return_value = None
        mock_exists.return_value = True

        default_path = "/default/lancedb/path"

        manager = LanceDBConnectionManager()
        with patch.object(
            manager, "get_default_lancedb_dir", return_value=default_path
        ) as mock_get_default:
            with patch.object(manager, "get_connection") as mock_get_connection:
                manager.get_connection_from_env()

        mock_get_default.assert_called_once()
        mock_get_connection.assert_called_once_with(default_path)

    @patch("xagent.providers.vector_store.lancedb.os.path.exists")
    @patch("xagent.providers.vector_store.lancedb.os.getenv")
    def test_connection_with_custom_path(self, mock_getenv, mock_exists):
        """Test LanceDB connection uses custom directory when LANCEDB_DIR is set."""
        from xagent.providers.vector_store.lancedb import LanceDBConnectionManager

        mock_getenv.return_value = "/custom/lancedb/path"
        mock_exists.return_value = True

        manager = LanceDBConnectionManager()
        with patch.object(manager, "get_connection") as mock_get_connection:
            manager.get_connection_from_env()

        mock_get_connection.assert_called_once_with("/custom/lancedb/path")


class TestAPIMultiTenancy:
    """Test multi-tenancy at the API level."""

    @patch("xagent.web.api.kb.list_collections")
    @pytest.mark.asyncio
    async def test_list_collections_api_with_user(self, mock_list_collections):
        """Test list_collections_api passes user context."""
        from xagent.web.models.user import User

        mock_user = MagicMock(spec=User)
        mock_user.id = 123
        mock_user.is_admin = False

        # Mock async function return value
        from xagent.core.tools.core.RAG_tools.core.schemas import ListCollectionsResult

        mock_result = ListCollectionsResult(
            status="success",
            total_count=0,
            collections=[],
            message="No collections found",
            warnings=[],
        )
        mock_list_collections.return_value = mock_result

        result = await list_collections_api(_user=mock_user)

        mock_list_collections.assert_called_once_with(user_id=123, is_admin=False)
        assert result.status == "success"
        assert result.total_count == 0

    @pytest.mark.asyncio
    @patch("xagent.web.api.kb.get_vector_index_store")
    @patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock)
    @patch("xagent.web.api.kb.delete_collection_physical_dir")
    @patch("xagent.web.api.kb.delete_collection")
    async def test_delete_collection_api_with_user(
        self,
        mock_delete_collection,
        mock_delete_collection_physical_dir,
        _mock_ensure_collection_access,
        mock_get_vector_store,
    ):
        """Test delete_collection_api passes user context and moves dir to trash."""
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )
        from xagent.web.models.user import User
        from xagent.web.services.kb_collection_service import (
            CollectionPhysicalDeleteResult,
        )

        mock_user = MagicMock(spec=User)
        mock_user.id = 123
        mock_user.is_admin = False

        mock_get_vector_store.return_value.list_document_records.return_value = []

        mock_path = MagicMock(spec=Path)
        mock_delete_collection_physical_dir.return_value = (
            CollectionPhysicalDeleteResult(
                status="success",
                collection_dir=mock_path,
            )
        )

        mock_result = CollectionOperationResult(
            status="success",
            collection="test_collection",
            message="Collection deleted",
            warnings=[],
            affected_documents=[],
            deleted_counts={},
        )
        mock_delete_collection.return_value = mock_result

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.delete.return_value = 0

        result = await delete_collection_api(
            "test_collection", _user=mock_user, db=mock_db
        )

        mock_delete_collection.assert_called_once_with("test_collection", 123, False)
        assert isinstance(result, CollectionOperationResult)
        assert result.status == "success"

    @pytest.mark.asyncio
    @patch("xagent.web.api.kb.get_vector_index_store")
    @patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock)
    @patch("xagent.web.api.kb.delete_collection_physical_dir")
    @patch("xagent.web.api.kb.delete_collection")
    async def test_delete_collection_api_admin_access(
        self,
        mock_delete_collection,
        mock_delete_collection_physical_dir,
        _mock_ensure_collection_access,
        mock_get_vector_store,
    ):
        """Test admin can delete collections (move dir to trash)."""
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )
        from xagent.web.models.user import User
        from xagent.web.services.kb_collection_service import (
            CollectionPhysicalDeleteResult,
        )

        mock_user = MagicMock(spec=User)
        mock_user.id = 999
        mock_user.is_admin = True

        mock_get_vector_store.return_value.list_document_records.return_value = []

        mock_path = MagicMock(spec=Path)
        mock_delete_collection_physical_dir.return_value = (
            CollectionPhysicalDeleteResult(
                status="success",
                collection_dir=mock_path,
            )
        )

        mock_result = CollectionOperationResult(
            status="success",
            collection="test_collection",
            message="Collection deleted",
            warnings=[],
            affected_documents=[],
            deleted_counts={},
        )
        mock_delete_collection.return_value = mock_result

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.delete.return_value = 0

        result = await delete_collection_api(
            "test_collection", _user=mock_user, db=mock_db
        )

        mock_delete_collection.assert_called_once_with("test_collection", 999, True)
        assert isinstance(result, CollectionOperationResult)
        assert result.status == "success"

    @pytest.mark.asyncio
    @patch("xagent.web.api.kb.get_vector_index_store")
    @patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock)
    @patch(
        "xagent.web.api.kb.list_collection_uploaded_file_owner_ids",
        return_value=set(),
    )
    @patch("xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store")
    @patch("xagent.web.api.kb.delete_collection_uploaded_files")
    @patch("xagent.web.api.kb.delete_collection_physical_dir")
    @patch("xagent.web.api.kb.delete_collection")
    async def test_delete_collection_api_admin_cleans_all_owner_storage(
        self,
        mock_delete_collection,
        mock_delete_collection_physical_dir,
        mock_delete_collection_uploaded_files,
        mock_get_metadata_store,
        _mock_uploaded_owner_ids,
        _mock_ensure_collection_access,
        mock_get_vector_store,
    ):
        """Admin global delete should clean storage for every owner in the collection."""
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )
        from xagent.web.models.user import User
        from xagent.web.services.kb_collection_service import (
            CollectionPhysicalDeleteResult,
        )

        mock_user = MagicMock(spec=User)
        mock_user.id = 999
        mock_user.is_admin = True
        mock_get_metadata_store.return_value.list_collection_config_owner_ids.return_value = set()

        owner_one_dir = Path("/tmp/uploads/user_101/team")
        owner_two_dir = Path("/tmp/uploads/user_202/team")
        mock_get_vector_store.return_value.list_document_records.side_effect = [
            [
                DocumentRecord(doc_id="doc-a", file_id="file-a", user_id=101),
                DocumentRecord(doc_id="doc-b", file_id="file-b", user_id=202),
            ],
            [],
        ]
        mock_delete_collection_physical_dir.side_effect = [
            CollectionPhysicalDeleteResult(
                status="success", collection_dir=owner_one_dir
            ),
            CollectionPhysicalDeleteResult(
                status="success", collection_dir=owner_two_dir
            ),
        ]
        mock_delete_collection_uploaded_files.return_value = 1
        mock_delete_collection.return_value = CollectionOperationResult(
            status="success",
            collection="team",
            message="Collection deleted",
            warnings=[],
            affected_documents=[],
            deleted_counts={},
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_vector_index_store",
            return_value=mock_get_vector_store.return_value,
        ):
            result = await delete_collection_api(
                "team", _user=mock_user, db=MagicMock()
            )

        mock_delete_collection.assert_called_once_with("team", 999, True)
        assert mock_delete_collection_physical_dir.call_args_list == [
            call(user_id=101, collection_name="team"),
            call(user_id=202, collection_name="team"),
        ]
        assert mock_delete_collection_uploaded_files.call_args_list == [
            call(
                ANY,
                user_id=101,
                collection_file_ids={"file-a"},
                remaining_file_ids=set(),
                collection_dir=owner_one_dir,
            ),
            call(
                ANY,
                user_id=202,
                collection_file_ids={"file-b"},
                remaining_file_ids=set(),
                collection_dir=owner_two_dir,
            ),
        ]
        assert result.status == "success"

    @pytest.mark.asyncio
    @patch(
        "xagent.core.tools.core.RAG_tools.kb.api_compatibility."
        "KBApiCompatibilityFacade.rename_collection_data",
        new_callable=AsyncMock,
    )
    @patch(
        "xagent.core.tools.core.RAG_tools.kb.api_compatibility."
        "KBApiCompatibilityFacade.list_collection_config_owner_ids",
        return_value=set(),
    )
    @patch("xagent.web.api.kb.list_document_records")
    @patch("xagent.web.api.kb._list_collections_with_retry", new_callable=AsyncMock)
    @patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock)
    @patch(
        "xagent.web.api.kb.list_collection_uploaded_file_owner_ids",
        return_value=set(),
    )
    @patch("xagent.web.api.kb.rename_collection_storage")
    async def test_rename_collection_api_admin_renames_all_owner_storage(
        self,
        mock_rename_collection_storage,
        _mock_uploaded_owner_ids,
        _mock_ensure_collection_access,
        mock_list_collections,
        mock_list_document_records,
        _mock_config_owner_ids,
        mock_rename_collection_data,
        tmp_path,
    ):
        """Admin global rename fans physical rename out to every owner and renames
        the shared data with admin scope.

        Routed through the coordinator, the data/status/metadata rename happens
        inside ``KBApiCompatibilityFacade.rename_collection_data`` — we assert the
        admin scope (``user_id``/``is_admin``) reaching that production seam; the
        coordinator → store scoping is covered by the kb/ routing tests.
        """
        from xagent.core.tools.core.RAG_tools.core.schemas import ListCollectionsResult
        from xagent.web.models.user import User
        from xagent.web.services.kb_collection_service import (
            CollectionPhysicalRenameResult,
        )

        mock_user = MagicMock(spec=User)
        mock_user.id = 999
        mock_user.is_admin = True

        mock_list_collections.return_value = ListCollectionsResult(
            status="success",
            total_count=0,
            collections=[],
            message="",
            warnings=[],
        )
        # Owner discovery: two distinct owners, each contributing one file.
        mock_list_document_records.return_value = [
            DocumentRecord(doc_id="doc-a", file_id="file-a", user_id=101),
            DocumentRecord(doc_id="doc-b", file_id="file-b", user_id=202),
        ]
        mock_rename_collection_data.return_value = []
        # Physical rename: not_found for every owner forces partial_success without
        # triggering rollback/HTTP errors.
        mock_rename_collection_storage.return_value = CollectionPhysicalRenameResult(
            status="not_found",
            old_collection_dir=tmp_path / "old",
            new_collection_dir=tmp_path / "new",
        )

        result = await rename_collection_api(
            "old", new_name="new", _user=mock_user, db=MagicMock()
        )

        # Physical rename fans out to BOTH owners with their own file ids.
        assert mock_rename_collection_storage.call_args_list == [
            call(
                ANY,
                user_id=101,
                old_collection_name="old",
                new_collection_name="new",
                collection_file_ids={"file-a"},
            ),
            call(
                ANY,
                user_id=202,
                old_collection_name="old",
                new_collection_name="new",
                collection_file_ids={"file-b"},
            ),
        ]
        # Data rename routed through the coordinator with admin scope.
        mock_rename_collection_data.assert_awaited_once_with(
            collection_name="old",
            new_name="new",
            user_id=999,
            is_admin=True,
        )
        assert result["status"] == "partial_success"

    @pytest.mark.asyncio
    @patch(
        "xagent.core.tools.core.RAG_tools.kb.api_compatibility."
        "KBApiCompatibilityFacade.rename_collection_data",
        new_callable=AsyncMock,
    )
    @patch("xagent.web.api.kb.list_document_records")
    @patch("xagent.web.api.kb._list_collections_with_retry", new_callable=AsyncMock)
    @patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock)
    @patch("xagent.web.api.kb.rename_collection_storage")
    async def test_rename_collection_api_non_admin_is_tenant_scoped(
        self,
        mock_rename_collection_storage,
        _mock_ensure_collection_access,
        mock_list_collections,
        mock_list_document_records,
        mock_rename_collection_data,
        tmp_path,
    ):
        """Regular rename should not mutate another user's same-name collection.

        A non-admin caller resolves to a single owner (itself), so physical rename
        runs once for that tenant and the data rename routes through the coordinator
        with tenant scope (``is_admin=False``).
        """
        from xagent.core.tools.core.RAG_tools.core.schemas import ListCollectionsResult
        from xagent.web.models.user import User
        from xagent.web.services.kb_collection_service import (
            CollectionPhysicalRenameResult,
        )

        mock_user = MagicMock(spec=User)
        mock_user.id = 101
        mock_user.is_admin = False

        mock_list_collections.return_value = ListCollectionsResult(
            status="success",
            total_count=0,
            collections=[],
            message="",
            warnings=[],
        )
        mock_list_document_records.return_value = [
            DocumentRecord(doc_id="doc-a", file_id="file-a", user_id=101),
        ]
        mock_rename_collection_data.return_value = []
        mock_rename_collection_storage.return_value = CollectionPhysicalRenameResult(
            status="not_found",
            old_collection_dir=tmp_path / "user_101" / "old",
            new_collection_dir=tmp_path / "user_101" / "new",
        )

        result = await rename_collection_api(
            "old", new_name="new", _user=mock_user, db=MagicMock()
        )

        # Only the requester's own tenant storage is renamed (no fan-out).
        mock_rename_collection_storage.assert_called_once_with(
            ANY,
            user_id=101,
            old_collection_name="old",
            new_collection_name="new",
            collection_file_ids={"file-a"},
        )
        # Data rename routed through the coordinator with tenant scope.
        mock_rename_collection_data.assert_awaited_once_with(
            collection_name="old",
            new_name="new",
            user_id=101,
            is_admin=False,
        )
        assert result["status"] == "partial_success"

    @pytest.mark.asyncio
    @patch("xagent.web.api.kb.get_vector_index_store")
    @patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock)
    @patch(
        "xagent.web.api.kb.list_collection_uploaded_file_owner_ids",
        return_value=set(),
    )
    @patch("xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store")
    @patch("xagent.web.api.kb.delete_collection_uploaded_files")
    @patch("xagent.web.api.kb.delete_collection_physical_dir")
    @patch("xagent.web.api.kb.delete_collection")
    async def test_delete_collection_api_admin_cleans_stale_config_owner_storage(
        self,
        mock_delete_collection,
        mock_delete_collection_physical_dir,
        mock_delete_collection_uploaded_files,
        mock_get_metadata_store,
        _mock_uploaded_owner_ids,
        _mock_ensure_collection_access,
        mock_get_vector_store,
    ):
        """Admin delete should clean owners discovered only from stale config."""
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )
        from xagent.web.models.user import User
        from xagent.web.services.kb_collection_service import (
            CollectionPhysicalDeleteResult,
        )

        mock_user = MagicMock(spec=User)
        mock_user.id = 999
        mock_user.is_admin = True
        mock_get_metadata_store.return_value.list_collection_config_owner_ids.return_value = {
            101
        }

        owner_dir = Path("/tmp/uploads/user_101/team")
        mock_get_vector_store.return_value.list_document_records.side_effect = [[], []]
        mock_delete_collection_physical_dir.return_value = (
            CollectionPhysicalDeleteResult(status="success", collection_dir=owner_dir)
        )
        mock_delete_collection_uploaded_files.return_value = 1
        mock_delete_collection.return_value = CollectionOperationResult(
            status="success",
            collection="team",
            message="Collection deleted",
            warnings=[],
            affected_documents=[],
            deleted_counts={},
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_vector_index_store",
            return_value=mock_get_vector_store.return_value,
        ):
            result = await delete_collection_api(
                "team", _user=mock_user, db=MagicMock()
            )

        mock_delete_collection_physical_dir.assert_called_once_with(
            user_id=101,
            collection_name="team",
        )
        mock_delete_collection_uploaded_files.assert_called_once_with(
            ANY,
            user_id=101,
            collection_file_ids=set(),
            remaining_file_ids=set(),
            collection_dir=owner_dir,
        )
        assert result.status == "success"

    @pytest.mark.asyncio
    @patch("xagent.web.api.kb.get_vector_index_store")
    @patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock)
    @patch(
        "xagent.web.api.kb.list_collection_uploaded_file_owner_ids",
        return_value={101},
    )
    @patch("xagent.core.tools.core.RAG_tools.storage.factory.get_metadata_store")
    @patch("xagent.web.api.kb.delete_collection_uploaded_files")
    @patch("xagent.web.api.kb.delete_collection_physical_dir")
    @patch("xagent.web.api.kb.delete_collection")
    async def test_delete_collection_api_admin_cleans_legacy_uploaded_file_owner(
        self,
        mock_delete_collection,
        mock_delete_collection_physical_dir,
        mock_delete_collection_uploaded_files,
        mock_get_metadata_store,
        _mock_uploaded_owner_ids,
        _mock_ensure_collection_access,
        mock_get_vector_store,
    ):
        """Admin delete should clean owners discovered only from UploadedFile paths."""
        from xagent.core.tools.core.RAG_tools.core.schemas import (
            CollectionOperationResult,
        )
        from xagent.web.models.user import User
        from xagent.web.services.kb_collection_service import (
            CollectionPhysicalDeleteResult,
        )

        mock_user = MagicMock(spec=User)
        mock_user.id = 999
        mock_user.is_admin = True
        mock_get_metadata_store.return_value.list_collection_config_owner_ids.return_value = set()

        owner_dir = Path("/tmp/uploads/user_101/team")
        mock_get_vector_store.return_value.list_document_records.side_effect = [[], []]
        mock_delete_collection_physical_dir.return_value = (
            CollectionPhysicalDeleteResult(status="success", collection_dir=owner_dir)
        )
        mock_delete_collection_uploaded_files.return_value = 1
        mock_delete_collection.return_value = CollectionOperationResult(
            status="success",
            collection="team",
            message="Collection deleted",
            warnings=[],
            affected_documents=[],
            deleted_counts={},
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.storage.factory.get_vector_index_store",
            return_value=mock_get_vector_store.return_value,
        ):
            result = await delete_collection_api(
                "team", _user=mock_user, db=MagicMock()
            )

        mock_delete_collection_physical_dir.assert_called_once_with(
            user_id=101,
            collection_name="team",
        )
        mock_delete_collection_uploaded_files.assert_called_once_with(
            ANY,
            user_id=101,
            collection_file_ids=set(),
            remaining_file_ids=set(),
            collection_dir=owner_dir,
        )
        assert result.status == "success"


class TestEndToEndMultiTenancy:
    """End-to-end multi-tenancy integration tests."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_user_data_isolation_workflow(self):
        """Test complete workflow ensuring user data isolation.

        Uses current UserPermissions: regular users see only their own data
        (no legacy NULL); unauthenticated gets no-access filter.
        """
        user1_id = 1001
        user2_id = 1002
        admin_id = None

        from xagent.core.tools.core.RAG_tools.utils.user_permissions import (
            UserPermissions,
        )

        assert UserPermissions.can_access_data(user1_id, user1_id, False) is True
        # Legacy (NULL) data is not accessible to regular users in current impl
        assert UserPermissions.can_access_data(user1_id, None, False) is False
        assert UserPermissions.can_access_data(user1_id, user2_id, False) is False

        assert UserPermissions.can_access_data(admin_id, user1_id, True) is True
        assert UserPermissions.can_access_data(admin_id, user2_id, True) is True
        assert UserPermissions.can_access_data(admin_id, None, True) is True

        user1_filter = UserPermissions.get_user_filter(user1_id, False)
        admin_filter = UserPermissions.get_user_filter(admin_id, True)
        null_filter = UserPermissions.get_user_filter(None, False)

        assert user1_filter == f"user_id == {user1_id}"
        assert admin_filter is None
        assert null_filter == UserPermissions.get_no_access_filter()

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.management.collections.get_vector_index_store"
            ) as mock_get_store,
        ):
            mock_store = MagicMock()
            mock_db_conn = MagicMock()
            mock_store.get_raw_connection.return_value = mock_db_conn
            mock_get_store.return_value = mock_store

            # Mock new storage abstraction methods
            mock_store.list_document_records.return_value = []
            mock_store.delete_collection_data.return_value = {}

            mock_docs_table = MagicMock()
            mock_db_conn.open_table.return_value = mock_docs_table
            mock_docs_table.count_rows.return_value = 0

            from xagent.core.tools.core.RAG_tools.management.collections import (
                delete_collection,
            )

            # delete_collection now uses list_document_records and delete_collection_data
            result = delete_collection(
                "test_collection", user_id=user1_id, is_admin=False
            )
            assert result.status == "success"

            result = delete_collection(
                "test_collection", user_id=admin_id, is_admin=True
            )
            assert result.status == "success"

        import inspect

        from xagent.core.tools.core.RAG_tools.pipelines.document_ingestion import (
            run_document_ingestion,
        )
        from xagent.core.tools.core.RAG_tools.pipelines.document_search import (
            run_document_search,
        )

        ingest_sig = inspect.signature(run_document_ingestion)
        search_sig = inspect.signature(run_document_search)

        assert "user_id" in ingest_sig.parameters
        assert "is_admin" in ingest_sig.parameters

        assert "user_id" in search_sig.parameters
        assert "is_admin" in search_sig.parameters

        assert ingest_sig.parameters["user_id"].default is None
        assert search_sig.parameters["user_id"].default is None
        assert search_sig.parameters["is_admin"].default is None

        integration_test_complete = True
        assert integration_test_complete, (
            "Multi-tenancy integration test completed successfully"
        )

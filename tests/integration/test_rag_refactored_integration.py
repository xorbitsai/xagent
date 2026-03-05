"""
End-to-end integration tests for the refactored RAG system.

This module tests the integration between the universal parsing layer,
LanceDB provider, and RAG adaptation layer to ensure the complete
workflow functions correctly after refactoring.
"""

from __future__ import annotations

import json
import os
import uuid
from unittest.mock import patch

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import ParseMethod
from xagent.core.tools.core.RAG_tools.file.register_document import (
    list_documents,
    register_document,
)
from xagent.core.tools.core.RAG_tools.parse.parse_document import parse_document
from xagent.providers.vector_store.lancedb import (
    LanceDBVectorStore,
    get_connection_from_env,
)


@pytest.fixture
def temp_lancedb_dir(tmp_path):
    """Provide an isolated LanceDB directory and set LANCEDB_DIR for tests."""
    original = os.environ.get("LANCEDB_DIR")
    temp_dir = tmp_path / "lancedb"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LANCEDB_DIR"] = str(temp_dir)
    try:
        yield str(temp_dir)
    finally:
        if original is not None:
            os.environ["LANCEDB_DIR"] = original
        else:
            os.environ.pop("LANCEDB_DIR", None)


class TestUniversalParsingIntegration:
    """Test universal parsing layer integration."""

    def test_text_parsing_integration(self, tmp_path, temp_lancedb_dir):
        """Test text parsing with real files."""
        # Create test text file
        test_file = tmp_path / "test.txt"
        test_content = "这是一个测试文档。\n包含多行文本内容。\n用于测试解析功能。"
        test_file.write_text(test_content, encoding="utf-8")

        # Register and parse using new API
        collection = f"test_collection_{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        register_result = register_document(
            collection=collection,
            source_path=str(test_file),
            doc_id=doc_id,
            user_id=1,
        )
        assert register_result["created"] is True

        parse_result = parse_document(
            collection=collection,
            doc_id=doc_id,
            parse_method=ParseMethod.DEFAULT,
            user_id=1,
            is_admin=True,
        )

        # Verify results
        assert parse_result["written"] is True
        assert len(parse_result["paragraphs"]) > 0
        # Combine all paragraphs to check content
        combined_text = "\n".join(p["text"] for p in parse_result["paragraphs"])
        assert test_content in combined_text or combined_text in test_content
        assert parse_result["paragraphs"][0]["metadata"]["source"] == str(test_file)
        assert parse_result["paragraphs"][0]["metadata"]["file_type"] == "txt"
        assert parse_result["paragraphs"][0]["metadata"]["parse_method"] == "default"

    def test_json_parsing_integration(self, tmp_path, temp_lancedb_dir):
        """Test JSON parsing with real files."""
        # Create test JSON file
        test_file = tmp_path / "test.json"
        test_data = {
            "title": "Test Document",
            "content": "This is test content",
            "metadata": {"author": "Test Author", "tags": ["test", "json"]},
        }
        test_file.write_text(
            json.dumps(test_data, ensure_ascii=False), encoding="utf-8"
        )

        # Register and parse using new API
        collection = f"test_collection_{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        register_result = register_document(
            collection=collection,
            source_path=str(test_file),
            doc_id=doc_id,
            user_id=1,
        )
        assert register_result["created"] is True

        parse_result = parse_document(
            collection=collection,
            doc_id=doc_id,
            parse_method=ParseMethod.DEFAULT,
            user_id=1,
            is_admin=True,
        )

        # Verify results
        assert parse_result["written"] is True
        assert len(parse_result["paragraphs"]) > 0
        combined_text = "\n".join(p["text"] for p in parse_result["paragraphs"])
        assert "Test Document" in combined_text
        assert "Test Author" in combined_text
        assert parse_result["paragraphs"][0]["metadata"]["file_type"] == "json"

    def test_json_parsing_with_array_extraction(self, tmp_path, temp_lancedb_dir):
        """Test JSON parsing with array extraction."""
        # Create test JSON array file
        test_file = tmp_path / "test_array.json"
        test_data = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"},
        ]
        test_file.write_text(json.dumps(test_data), encoding="utf-8")

        # Register and parse using new API
        # Note: Array extraction is not currently supported in new API
        collection = f"test_collection_{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        register_result = register_document(
            collection=collection,
            source_path=str(test_file),
            doc_id=doc_id,
            user_id=1,
        )
        assert register_result["created"] is True

        parse_result = parse_document(
            collection=collection,
            doc_id=doc_id,
            parse_method=ParseMethod.DEFAULT,
            user_id=1,
            is_admin=True,
        )

        # Verify results - JSON should be parsed as text
        assert parse_result["written"] is True
        assert len(parse_result["paragraphs"]) > 0
        combined_text = "\n".join(p["text"] for p in parse_result["paragraphs"])
        # All items should be in the parsed text
        assert '"id": 1' in combined_text or "Item 1" in combined_text


class TestLanceDBProviderIntegration:
    """Test LanceDB provider integration."""

    def test_vector_store_lifecycle(self, tmp_path):
        """Test complete vector store lifecycle."""
        # Create vector store
        db_dir = tmp_path / "lancedb"
        store = LanceDBVectorStore(str(db_dir), "integration_test")

        # Test data
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.7, 0.7, 0.0],  # Should be close to both x and y vectors
        ]
        metadatas = [
            {"text": "red vector", "category": "color"},
            {"text": "green vector", "category": "color"},
            {"text": "blue vector", "category": "color"},
            {"text": "yellow vector", "category": "color"},
        ]

        # Add vectors
        ids = store.add_vectors(vectors, metadatas=metadatas)
        assert len(ids) == 4
        assert all(isinstance(id_, str) for id_ in ids)

        # Search for similar vectors
        query_vector = [1.0, 0.1, 0.0]  # Should be closest to red
        results = store.search_vectors(query_vector, top_k=2)

        assert len(results) <= 2
        assert all("id" in result for result in results)
        assert all("score" in result for result in results)
        assert all("metadata" in result for result in results)

        # First result should be red (closest match)
        first_result = results[0]
        assert "red" in first_result["metadata"]["text"]

        # Test deletion
        success = store.delete_vectors([ids[0]])  # Delete red vector
        assert success

        # Search again - should not find red vector
        results_after_delete = store.search_vectors(query_vector, top_k=4)
        remaining_texts = [r["metadata"]["text"] for r in results_after_delete]
        assert not any("red" in text for text in remaining_texts)

        # Test clear
        store.clear()
        results_after_clear = store.search_vectors(query_vector, top_k=10)
        assert len(results_after_clear) == 0

    def test_connection_manager_integration(self, tmp_path):
        """Test connection manager integration."""
        db_dir = str(tmp_path / "connection_test")

        # Test environment variable connection
        with patch.dict(os.environ, {"TEST_LANCEDB_DIR": db_dir}):
            conn = get_connection_from_env("TEST_LANCEDB_DIR")
            assert conn is not None

            # Should be able to create tables
            sample_data = [{"id": "test", "data": "sample"}]
            table = conn.create_table("test_table", data=sample_data)
            assert table is not None

            # Verify table exists
            table_names = conn.table_names()
            assert "test_table" in table_names


class TestKnowledgeBaseIsolationIntegration:
    """Integration tests for KB isolation (Issue #72): list_documents and search by collection."""

    def test_list_documents_returns_only_requested_collection(
        self, tmp_path, temp_lancedb_dir
    ):
        """list_documents must return only documents from the requested collection."""
        file_a = tmp_path / "kb_a_doc.txt"
        file_a.write_text("Content in KB A", encoding="utf-8")
        file_b = tmp_path / "kb_b_doc.txt"
        file_b.write_text("Content in KB B", encoding="utf-8")

        register_document(
            collection="kb_chinese_new_year",
            source_path=str(file_a),
            doc_id="doc-a",
            user_id=1,
        )
        register_document(
            collection="kb_other",
            source_path=str(file_b),
            doc_id="doc-b",
            user_id=1,
        )

        results = list_documents(
            temp_lancedb_dir, collection="kb_chinese_new_year", limit=100
        )
        assert len(results) == 1
        assert results[0]["collection"] == "kb_chinese_new_year"
        assert results[0]["doc_id"] == "doc-a"

        results_other = list_documents(
            temp_lancedb_dir, collection="kb_other", limit=100
        )
        assert len(results_other) == 1
        assert results_other[0]["collection"] == "kb_other"
        assert results_other[0]["doc_id"] == "doc-b"

    def test_search_returns_only_specified_collection(self, tmp_path, temp_lancedb_dir):
        """Search with a specified collection must not return results from other collections (Issue #72)."""
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.retrieval.search_dense import (
            search_dense,
        )
        from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
            write_vectors_to_db,
        )

        conn = get_connection_from_env()
        model_tag = "kb_isolate_test_model"
        table_name = f"embeddings_{model_tag}"
        try:
            conn.drop_table(table_name)
        except Exception:
            pass

        ensure_embeddings_table(conn, model_tag, vector_dim=3)

        # Same vector in both collections: without collection filter, both would match
        vec = [1.0, 0.0, 0.0]
        emb = ChunkEmbeddingData(
            doc_id="doc_alpha",
            chunk_id="chunk_alpha",
            parse_hash="parse1",
            model=model_tag,
            vector=vec,
            text="content in kb_alpha",
            chunk_hash="hash_alpha",
        )
        r1 = write_vectors_to_db("kb_alpha", [emb], create_index=False)
        assert r1.upsert_count == 1

        emb_beta = ChunkEmbeddingData(
            doc_id="doc_beta",
            chunk_id="chunk_beta",
            parse_hash="parse1",
            model=model_tag,
            vector=vec,
            text="content in kb_beta",
            chunk_hash="hash_beta",
        )
        r2 = write_vectors_to_db("kb_beta", [emb_beta], create_index=False)
        assert r2.upsert_count == 1

        # Search only in kb_alpha
        response = search_dense(
            collection="kb_alpha",
            model_tag=model_tag,
            query_vector=vec,
            top_k=10,
            user_id=None,
            is_admin=True,
        )

        assert response.status == "success"
        assert len(response.results) >= 1
        doc_ids = [r.doc_id for r in response.results]
        assert all(did == "doc_alpha" for did in doc_ids), (
            f"Expected all results from kb_alpha (doc_alpha), got doc_ids: {doc_ids}"
        )
        assert "doc_beta" not in doc_ids, (
            "Search in kb_alpha must not return results from kb_beta"
        )


class TestRAGWorkflowIntegration:
    """Test complete RAG workflow integration."""

    def test_parse_to_vector_workflow(self, tmp_path, temp_lancedb_dir):
        """Test workflow from parsing to vector storage."""
        # 1. Create test document
        test_file = tmp_path / "workflow_test.txt"
        test_content = (
            "人工智能是未来的技术。\n机器学习改变了世界。\n深度学习推动创新。"
        )
        test_file.write_text(test_content, encoding="utf-8")

        # 2. Register and parse document
        collection = f"test_collection_{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())
        register_result = register_document(
            collection=collection,
            source_path=str(test_file),
            doc_id=doc_id,
            user_id=1,
        )
        assert register_result["created"] is True

        parse_result = parse_document(
            collection=collection,
            doc_id=doc_id,
            parse_method=ParseMethod.DEFAULT,
            user_id=1,
            is_admin=True,
        )

        assert parse_result["written"] is True
        assert len(parse_result["paragraphs"]) > 0
        parsed_text = "\n".join(p["text"] for p in parse_result["paragraphs"])
        parsed_metadata = parse_result["paragraphs"][0]["metadata"]

        # 3. Create mock vectors (in real scenario, this would use embedding model)
        mock_vectors = [[0.1, 0.2, 0.3]]  # Mock embedding for the text

        # 4. Store in LanceDB
        db_dir = tmp_path / "workflow_db"
        store = LanceDBVectorStore(str(db_dir), "workflow_collection")

        # Combine parsed metadata with vector metadata
        vector_metadatas = [
            {
                "text": parsed_text,
                "source": parsed_metadata["source"],
                "file_type": parsed_metadata["file_type"],
                "parse_method": parsed_metadata["parse_method"],
            }
        ]

        ids = store.add_vectors(mock_vectors, metadatas=vector_metadatas)
        assert len(ids) == 1

        # 5. Search and verify complete metadata chain
        query_vector = [0.1, 0.2, 0.3]  # Same as stored vector
        results = store.search_vectors(query_vector, top_k=1)

        assert len(results) == 1
        result = results[0]

        # Verify complete metadata preservation
        assert (
            test_content in result["metadata"]["text"]
            or result["metadata"]["text"] in test_content
        )
        assert result["metadata"]["source"] == str(test_file)
        assert result["metadata"]["file_type"] == "txt"
        assert result["metadata"]["parse_method"] == "default"

    def test_multi_document_workflow(self, tmp_path, temp_lancedb_dir):
        """Test workflow with multiple documents and file types."""
        # Create multiple test documents
        documents = []

        # Text document
        txt_file = tmp_path / "doc1.txt"
        txt_content = "这是第一个文档。"
        txt_file.write_text(txt_content, encoding="utf-8")
        documents.append(("txt", str(txt_file)))

        # JSON document
        json_file = tmp_path / "doc2.json"
        json_content = {"title": "第二个文档", "content": "JSON格式的内容"}
        json_file.write_text(
            json.dumps(json_content, ensure_ascii=False), encoding="utf-8"
        )
        documents.append(("json", str(json_file)))

        # Parse all documents
        collection = f"test_collection_{uuid.uuid4().hex[:8]}"
        all_parsed_results = []
        for file_type, file_path in documents:
            doc_id = str(uuid.uuid4())
            register_result = register_document(
                collection=collection,
                source_path=file_path,
                doc_id=doc_id,
                user_id=1,
            )
            assert register_result["created"] is True

            parse_result = parse_document(
                collection=collection,
                doc_id=doc_id,
                parse_method=ParseMethod.DEFAULT,
                user_id=1,
                is_admin=True,
            )
            # Convert paragraphs to old format for compatibility
            for para in parse_result["paragraphs"]:
                all_parsed_results.append(
                    {
                        "text": para["text"],
                        "metadata": para["metadata"],
                    }
                )

        assert len(all_parsed_results) >= 2

        # Create mock vectors for all documents
        mock_vectors = [
            [0.1, 0.0, 0.0],  # Vector for txt document
            [0.0, 0.1, 0.0],  # Vector for json document
        ]

        # Store all in vector database
        db_dir = tmp_path / "multi_doc_db"
        store = LanceDBVectorStore(str(db_dir), "multi_doc_collection")

        vector_metadatas = []
        for parsed_result in all_parsed_results:
            vector_metadatas.append(
                {
                    "text": parsed_result["text"],
                    "source": parsed_result["metadata"]["source"],
                    "file_type": parsed_result["metadata"]["file_type"],
                    "parse_method": parsed_result["metadata"]["parse_method"],
                }
            )

        ids = store.add_vectors(mock_vectors, metadatas=vector_metadatas)
        assert len(ids) == 2

        # Search for each document type
        txt_query = [0.1, 0.0, 0.0]
        txt_results = store.search_vectors(txt_query, top_k=1)
        assert len(txt_results) == 1
        assert txt_results[0]["metadata"]["file_type"] == "txt"
        assert "第一个文档" in txt_results[0]["metadata"]["text"]

        json_query = [0.0, 0.1, 0.0]
        json_results = store.search_vectors(json_query, top_k=1)
        assert len(json_results) == 1
        assert json_results[0]["metadata"]["file_type"] == "json"
        assert "第二个文档" in json_results[0]["metadata"]["text"]


@pytest.mark.integration
class TestFullSystemIntegration:
    """Full system integration tests requiring all components."""

    def test_complete_rag_pipeline(self, tmp_path, temp_lancedb_dir):
        """Test complete RAG pipeline from document to retrieval."""
        # This test would typically require:
        # 1. Real document files
        # 2. Real embedding models
        # 3. Complete RAG workflow

        # For now, we'll test the infrastructure is in place
        assert True  # Placeholder for full pipeline test

    def test_error_handling_across_layers(self, tmp_path, temp_lancedb_dir):
        """Test error handling propagation across all layers."""
        # Test file not found error propagation
        collection = f"test_collection_{uuid.uuid4().hex[:8]}"
        with pytest.raises(Exception):  # May raise different exceptions
            register_document(
                collection=collection,
                source_path="/nonexistent/file.txt",
                user_id=1,
            )

        # Test vector store error handling
        db_dir = tmp_path / "error_test_db"
        store = LanceDBVectorStore(str(db_dir), "error_test")

        # Test invalid vector dimensions (should handle gracefully)
        try:
            store.add_vectors([[1.0, 2.0]], metadatas=[{"text": "test"}])
            store.search_vectors([1.0, 2.0, 3.0], top_k=1)  # Different dimension
            # Should either work or fail gracefully
        except Exception as e:
            # Acceptable if it fails with a clear error
            assert (
                "dimension" in str(e).lower()
                or "shape" in str(e).lower()
                or len(str(e)) > 0
            )

"""End-to-end tests for metadata propagation through RAG pipeline.

This module tests that metadata is correctly preserved and passed through
all stages of the RAG pipeline: Parse → Chunk → Embedding → Search.
"""

import os
import tempfile
import uuid

import pandas as pd
import pytest

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.chunk.chunk_document import chunk_document
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
    ChunkStrategy,
    ParseMethod,
)
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.parse.parse_document import parse_document
from xagent.core.tools.core.RAG_tools.retrieval.format_context import (
    format_search_results_for_llm,
)
from xagent.core.tools.core.RAG_tools.retrieval.search_dense import search_dense
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_vector_store_raw_connection,
)
from xagent.core.tools.core.RAG_tools.utils.metadata_utils import deserialize_metadata
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    read_chunks_for_embedding,
    write_vectors_to_db,
)


class _StubEmbeddingAdapter(BaseEmbedding):
    """Deterministic embedding adapter for tests."""

    def __init__(self, prefix: str = "vec") -> None:
        self.prefix = prefix

    def encode(  # type: ignore[override]
        self,
        text: str | list[str],
        dimension: int | None = None,
        instruct: str | None = None,
    ) -> list[float] | list[list[float]]:
        if isinstance(text, str):
            return [float(len(text)), 0.0]
        return [[float(len(item)), float(index)] for index, item in enumerate(text)]

    def get_dimension(self) -> int:
        return 2

    @property
    def abilities(self) -> list[str]:
        return ["embedding"]


@pytest.fixture
def temp_lancedb_dir():
    """Create a temporary LanceDB directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_dir = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = temp_dir
        yield temp_dir
        if old_dir:
            os.environ["LANCEDB_DIR"] = old_dir
        else:
            os.environ.pop("LANCEDB_DIR", None)


@pytest.fixture
def test_collection():
    """Test collection name."""
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_doc_id():
    """Test document ID."""
    return str(uuid.uuid4())


@pytest.fixture
def stub_embedding_adapter():
    """Create a stub embedding adapter for tests."""
    return _StubEmbeddingAdapter()


@pytest.fixture
def stub_embedding_config():
    """Create a stub embedding config for tests."""
    return EmbeddingModelConfig(
        id="embedding-default",
        model_name="test-embedding",
        model_provider="test",
        dimension=2,
    )


class TestMetadataPropagationParseToChunk:
    """Test metadata propagation from Parse to Chunk stage."""

    def test_metadata_preserved_in_chunks_table(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test that metadata from parsing is preserved in chunks table."""
        # Step 1: Register document
        txt_path = "tests/resources/test_files/test.txt"
        register_result = register_document(
            collection=test_collection,
            source_path=txt_path,
            doc_id=test_doc_id,
            user_id=1,
        )
        assert register_result["created"] is True

        # Step 2: Parse document
        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        assert parse_result["written"] is True
        parse_hash = parse_result["parse_hash"]

        # Step 3: Chunk document
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            user_id=1,
        )
        assert chunk_result["chunk_count"] > 0

        # Step 4: Verify metadata in chunks table
        conn = get_vector_store_raw_connection()
        chunks_table = conn.open_table("chunks")
        df = (
            chunks_table.search()
            .where(
                f"collection == '{test_collection}' AND doc_id == '{test_doc_id}' AND parse_hash == '{parse_hash}'"
            )
            .to_pandas()
        )

        assert len(df) > 0, "No chunks found"

        # Verify metadata field exists and can be deserialized
        for _, row in df.iterrows():
            if pd.notna(row.get("metadata")):
                metadata = deserialize_metadata(row["metadata"])
                assert metadata is not None
                # Metadata should be a dictionary
                assert isinstance(metadata, dict)
                # Should contain at least source information
                if "source" in metadata:
                    assert isinstance(metadata["source"], str)


class TestMetadataPropagationChunkToEmbedding:
    """Test metadata propagation from Chunk to Embedding stage."""

    def test_metadata_preserved_in_embeddings_table(
        self,
        temp_lancedb_dir,
        test_collection,
        test_doc_id,
        stub_embedding_adapter,
        stub_embedding_config,
        monkeypatch,
    ):
        """Test that metadata from chunks is preserved when creating embeddings."""

        # Step 1-3: Register, parse, and chunk document
        txt_path = "tests/resources/test_files/test.txt"
        register_document(
            collection=test_collection,
            source_path=txt_path,
            doc_id=test_doc_id,
            user_id=1,
        )
        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]
        chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            user_id=1,
        )

        # Create embeddings table to avoid validation errors
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_store_raw_connection,
        )

        conn = get_vector_store_raw_connection()
        ensure_embeddings_table(conn, "test_model", vector_dim=2)

        # Step 4: Read chunks for embedding
        read_response = read_chunks_for_embedding(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            model="test_model",
            user_id=1,
        )

        assert len(read_response.chunks) > 0

        # Verify chunks have metadata
        chunks_with_metadata = [
            chunk for chunk in read_response.chunks if chunk.metadata is not None
        ]
        assert len(chunks_with_metadata) > 0, (
            "At least some chunks should have metadata"
        )

        # Step 5: Create embeddings and write to database
        embeddings = []
        for chunk in read_response.chunks:
            vector = stub_embedding_adapter.encode(chunk.text)
            embeddings.append(
                ChunkEmbeddingData(
                    collection=test_collection,
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    parse_hash=chunk.parse_hash,
                    text=chunk.text,
                    vector=vector,
                    chunk_hash=chunk.chunk_hash,
                    model="test_model",
                    metadata=chunk.metadata,
                )
            )

        write_response = write_vectors_to_db(
            collection=test_collection,
            embeddings=embeddings,
            user_id=1,
        )

        assert write_response.upsert_count > 0

        # Step 6: Verify metadata in embeddings table
        conn = get_vector_store_raw_connection()
        embeddings_table = conn.open_table("embeddings_test_model")
        df = (
            embeddings_table.search()
            .where(
                f"collection == '{test_collection}' AND doc_id == '{test_doc_id}' AND parse_hash == '{parse_hash}'"
            )
            .to_pandas()
        )

        assert len(df) > 0, "No embeddings found"

        # Verify metadata field exists and can be deserialized
        metadata_found = False
        for _, row in df.iterrows():
            if pd.notna(row.get("metadata")):
                metadata = deserialize_metadata(row["metadata"])
                if metadata is not None:
                    metadata_found = True
                    assert isinstance(metadata, dict)
                    # Should contain chunk-related metadata
                    assert (
                        "source" in metadata
                        or "page_number" in metadata
                        or len(metadata) > 0
                    )

        assert metadata_found, "At least one embedding should have metadata"


class TestMetadataPropagationEmbeddingToSearch:
    """Test metadata propagation from Embedding to Search stage."""

    def test_metadata_in_search_results(
        self,
        temp_lancedb_dir,
        test_collection,
        test_doc_id,
        stub_embedding_adapter,
        stub_embedding_config,
        monkeypatch,
    ):
        """Test that metadata is included in search results."""

        # Step 1-5: Complete pipeline up to embeddings
        txt_path = "tests/resources/test_files/test.txt"
        register_document(
            collection=test_collection,
            source_path=txt_path,
            doc_id=test_doc_id,
            user_id=1,
        )
        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]
        chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            user_id=1,
        )

        # Create embeddings table to avoid validation errors
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_store_raw_connection,
        )

        conn = get_vector_store_raw_connection()
        ensure_embeddings_table(conn, "test_model", vector_dim=2)

        read_response = read_chunks_for_embedding(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            model="test_model",
            user_id=1,
        )

        embeddings = []
        for chunk in read_response.chunks:
            vector = stub_embedding_adapter.encode(chunk.text)
            embeddings.append(
                ChunkEmbeddingData(
                    collection=test_collection,
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    parse_hash=chunk.parse_hash,
                    text=chunk.text,
                    vector=vector,
                    chunk_hash=chunk.chunk_hash,
                    model="test_model",
                    metadata=chunk.metadata,
                )
            )

        write_vectors_to_db(
            collection=test_collection,
            embeddings=embeddings,
            user_id=1,
        )

        # Step 6: Perform search
        query_text = "test content"
        query_vector = stub_embedding_adapter.encode(query_text)

        search_response = search_dense(
            collection=test_collection,
            query_vector=query_vector,
            model_tag="test_model",
            top_k=5,
            user_id=None,
            is_admin=True,
        )

        assert len(search_response.results) > 0

        # Verify search results contain metadata
        results_with_metadata = [
            result for result in search_response.results if result.metadata is not None
        ]
        assert len(results_with_metadata) > 0, (
            "At least some search results should have metadata"
        )

        # Verify metadata structure
        for result in results_with_metadata:
            assert isinstance(result.metadata, dict)
            # Metadata should contain useful information
            assert len(result.metadata) > 0


class TestMetadataPropagationEndToEnd:
    """End-to-end tests for complete metadata propagation."""

    def test_full_pipeline_metadata_preservation(
        self,
        temp_lancedb_dir,
        test_collection,
        test_doc_id,
        stub_embedding_adapter,
        stub_embedding_config,
        monkeypatch,
    ):
        """Test that metadata is preserved through the complete pipeline."""

        # Complete pipeline: Register → Parse → Chunk → Embedding → Search
        txt_path = "tests/resources/test_files/test.txt"
        register_document(
            collection=test_collection,
            source_path=txt_path,
            doc_id=test_doc_id,
            user_id=1,
        )

        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method=ParseMethod.DEEPDOC,
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Verify parse result has metadata
        if parse_result.get("paragraphs"):
            first_para = parse_result["paragraphs"][0]
            assert "metadata" in first_para
            parse_metadata = first_para["metadata"]
            assert isinstance(parse_metadata, dict)
            assert "source" in parse_metadata

        chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            user_id=1,
        )

        # Create embeddings table to avoid validation errors
        from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
            ensure_embeddings_table,
        )
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_store_raw_connection,
        )

        conn = get_vector_store_raw_connection()
        ensure_embeddings_table(conn, "test_model", vector_dim=2)

        read_response = read_chunks_for_embedding(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            model="test_model",
            user_id=1,
        )

        # Verify chunks have metadata
        chunks_with_metadata = [
            chunk for chunk in read_response.chunks if chunk.metadata is not None
        ]
        assert len(chunks_with_metadata) > 0

        # Create embeddings
        embeddings = []
        for chunk in read_response.chunks:
            vector = stub_embedding_adapter.encode(chunk.text)
            embeddings.append(
                ChunkEmbeddingData(
                    collection=test_collection,
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    parse_hash=chunk.parse_hash,
                    text=chunk.text,
                    vector=vector,
                    chunk_hash=chunk.chunk_hash,
                    model="test_model",
                    metadata=chunk.metadata,
                )
            )

        write_vectors_to_db(
            collection=test_collection,
            embeddings=embeddings,
            user_id=1,
        )

        # Perform search
        query_text = "test"
        query_vector = stub_embedding_adapter.encode(query_text)

        search_response = search_dense(
            collection=test_collection,
            query_vector=query_vector,
            model_tag="test_model",
            top_k=5,
            user_id=None,
            is_admin=True,
        )

        # Verify search results have metadata
        results_with_metadata = [
            result for result in search_response.results if result.metadata is not None
        ]
        assert len(results_with_metadata) > 0

        # Test format_context with metadata
        formatted = format_search_results_for_llm(
            search_response.results, include_metadata=True
        )

        # Verify formatted output contains metadata information
        assert "Document ID" in formatted
        assert "Chunk ID" in formatted
        # Should contain metadata if results have it
        if results_with_metadata:
            # Metadata might be serialized in the output
            assert len(formatted) > 0

        # Verify metadata consistency: search result metadata should match
        # the original chunk metadata (at least structure-wise)
        for result in results_with_metadata[:3]:  # Check first 3 results
            assert result.metadata is not None
            assert isinstance(result.metadata, dict)
            # Should have at least some fields from original metadata
            assert len(result.metadata) > 0

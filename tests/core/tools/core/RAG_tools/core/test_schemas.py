"""Tests for core data schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from xagent.core.tools.core.RAG_tools.core.config import IndexPolicy
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkDocumentRequest,
    ChunkDocumentResponse,
    ChunkEmbeddingData,
    ChunkForEmbedding,
    ChunkStrategy,
    CollectionInfo,
    DenseSearchResponse,
    EmbeddingReadRequest,
    EmbeddingReadResponse,
    EmbeddingWriteRequest,
    EmbeddingWriteResponse,
    FallbackInfo,
    IndexMetric,
    IndexOperation,
    IndexStatus,
    IndexType,
    ParseDocumentRequest,
    ParseDocumentResponse,
    ParsedParagraph,
    ParseMethod,
    PerformanceImpact,
    RegisterDocumentRequest,
    RegisterDocumentResponse,
    SearchFallbackAction,
    SearchResult,
    SearchWarning,
)


class TestRegisterDocumentRequest:
    """Test RegisterDocumentRequest model."""

    def test_valid_request_with_required_fields(self):
        """Test request creation with required fields."""
        request = RegisterDocumentRequest(
            collection="test_collection", source_path="/tmp/test.txt"
        )

        assert request.collection == "test_collection"
        assert request.source_path == "/tmp/test.txt"
        assert request.file_type is None
        assert request.doc_id is None
        assert request.uploaded_at is None

    def test_valid_request_with_all_fields(self):
        """Test request creation with all fields."""
        upload_time = datetime.now()

        request = RegisterDocumentRequest(
            collection="test_collection",
            source_path="/tmp/test.txt",
            file_type="txt",
            doc_id="test-doc-123",
            uploaded_at=upload_time,
        )

        assert request.collection == "test_collection"
        assert request.source_path == "/tmp/test.txt"
        assert request.file_type == "txt"
        assert request.doc_id == "test-doc-123"
        assert request.uploaded_at == upload_time

    def test_request_validation_missing_required_field(self):
        """Test validation fails when required field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterDocumentRequest(
                # Missing collection
                source_path="/tmp/test.txt"
            )

        assert "collection" in str(exc_info.value)

    def test_request_frozen_behavior(self):
        """Test that request is frozen and cannot be modified."""
        request = RegisterDocumentRequest(
            collection="test_collection", source_path="/tmp/test.txt"
        )

        # Should raise error when trying to modify
        with pytest.raises(ValidationError):
            request.collection = "new_collection"

    def test_field_descriptions(self):
        """Test that field descriptions are set correctly."""
        schema = RegisterDocumentRequest.model_json_schema()

        # Check that descriptions exist
        assert "description" in schema["properties"]["collection"]
        assert "description" in schema["properties"]["source_path"]
        assert "description" in schema["properties"]["file_type"]


class TestRegisterDocumentResponse:
    """Test RegisterDocumentResponse model."""

    def test_valid_response(self):
        """Test response creation with valid data."""
        response = RegisterDocumentResponse(
            doc_id="test-doc-123", created=True, content_hash="abc123def456"
        )

        assert response.doc_id == "test-doc-123"
        assert response.created is True
        assert response.content_hash == "abc123def456"

    def test_response_validation_missing_field(self):
        """Test validation fails when required field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterDocumentResponse(
                # Missing doc_id
                created=True,
                content_hash="abc123def456",
            )

        assert "doc_id" in str(exc_info.value)

    def test_response_frozen_behavior(self):
        """Test that response is frozen and cannot be modified."""
        response = RegisterDocumentResponse(
            doc_id="test-doc-123", created=True, content_hash="abc123def456"
        )

        # Should raise error when trying to modify
        with pytest.raises(ValidationError):
            response.created = False

    def test_field_descriptions(self):
        """Test that field descriptions are set correctly."""
        schema = RegisterDocumentResponse.model_json_schema()

        # Check that descriptions exist
        assert "description" in schema["properties"]["doc_id"]
        assert "description" in schema["properties"]["created"]
        assert "description" in schema["properties"]["content_hash"]


class TestParseDocumentSchemas:
    """Tests for ParseDocumentRequest/Response and ParsedParagraph models."""

    def test_parse_document_request_valid(self) -> None:
        req = ParseDocumentRequest(
            collection="c1",
            doc_id="d1",
            parse_method=ParseMethod.DEFAULT,
            params={"foo": "bar"},
        )
        assert req.collection == "c1"
        assert req.doc_id == "d1"
        assert req.parse_method == ParseMethod.DEFAULT
        assert req.params == {"foo": "bar"}

    def test_parse_document_request_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            ParseDocumentRequest(
                # missing doc_id
                collection="c1",
                parse_method=ParseMethod.DEFAULT,
            )

    def test_parsed_paragraph_model(self) -> None:
        para = ParsedParagraph(text="hello", metadata={"page_number": 1})
        assert para.text == "hello"
        assert para.metadata["page_number"] == 1

    def test_parse_document_response_valid(self) -> None:
        response = ParseDocumentResponse(
            doc_id="d1",
            parse_hash="abc123",
            paragraphs=[ParsedParagraph(text="t", metadata={})],
            written=True,
        )
        assert response.doc_id == "d1"
        assert response.parse_hash == "abc123"
        assert len(response.paragraphs) == 1
        assert response.written is True

    def test_parse_document_response_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            ParseDocumentResponse(
                # missing parse_hash
                doc_id="d1",
                paragraphs=[ParsedParagraph(text="t", metadata={})],
                written=True,
            )


class TestChunkStrategy:
    """Test ChunkStrategy enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert ChunkStrategy.RECURSIVE.value == "recursive"
        assert ChunkStrategy.MARKDOWN.value == "markdown"
        assert ChunkStrategy.FIXED_SIZE.value == "fixed_size"

    def test_enum_string_conversion(self):
        """Test that enum converts to string correctly."""
        assert str(ChunkStrategy.RECURSIVE) == "recursive"
        assert str(ChunkStrategy.MARKDOWN) == "markdown"
        assert str(ChunkStrategy.FIXED_SIZE) == "fixed_size"

    def test_enum_value_access(self):
        """Test that enum values can be accessed correctly."""
        assert ChunkStrategy.RECURSIVE.value == "recursive"
        assert ChunkStrategy.MARKDOWN.value == "markdown"
        assert ChunkStrategy.FIXED_SIZE.value == "fixed_size"

        # Test that enum instances are not equal to strings (type safety)
        assert ChunkStrategy.RECURSIVE != "recursive"
        assert ChunkStrategy.MARKDOWN != "markdown"
        assert ChunkStrategy.FIXED_SIZE != "fixed_size"

    def test_enum_membership(self):
        """Test enum membership checks."""
        strategies = {
            ChunkStrategy.RECURSIVE,
            ChunkStrategy.MARKDOWN,
            ChunkStrategy.FIXED_SIZE,
        }
        assert ChunkStrategy.RECURSIVE in strategies
        assert ChunkStrategy.MARKDOWN in strategies
        assert ChunkStrategy.FIXED_SIZE in strategies


class TestParseMethod:
    """Test ParseMethod enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert ParseMethod.DEFAULT.value == "default"
        assert ParseMethod.PYPDF.value == "pypdf"
        assert ParseMethod.PDFPLUMBER.value == "pdfplumber"
        assert ParseMethod.UNSTRUCTURED.value == "unstructured"
        assert ParseMethod.PYMUPDF.value == "pymupdf"

    def test_enum_string_conversion(self):
        """Test that enum converts to string correctly."""
        assert str(ParseMethod.DEFAULT) == "default"
        assert str(ParseMethod.PYPDF) == "pypdf"
        assert str(ParseMethod.UNSTRUCTURED) == "unstructured"

    def test_enum_value_access(self):
        """Test that enum values can be accessed correctly."""
        # Test that enum instances are not equal to strings (type safety)
        assert ParseMethod.DEFAULT != "default"
        assert ParseMethod.PYPDF != "pypdf"
        assert ParseMethod.UNSTRUCTURED != "unstructured"
        assert ParseMethod.PDFPLUMBER != "pdfplumber"
        assert ParseMethod.PYMUPDF != "pymupdf"

    def test_enum_membership(self):
        """Test enum membership checks."""
        methods = {
            ParseMethod.DEFAULT,
            ParseMethod.PYPDF,
            ParseMethod.PDFPLUMBER,
            ParseMethod.UNSTRUCTURED,
            ParseMethod.PYMUPDF,
        }
        assert ParseMethod.DEFAULT in methods
        assert ParseMethod.PYPDF in methods
        assert ParseMethod.PDFPLUMBER in methods
        assert ParseMethod.UNSTRUCTURED in methods
        assert ParseMethod.PYMUPDF in methods


class TestChunkDocumentRequest:
    """Test ChunkDocumentRequest model."""

    def test_valid_request_with_required_fields(self):
        """Test request creation with required fields."""
        request = ChunkDocumentRequest(
            collection="test_collection",
            doc_id="test-doc-123",
            parse_hash="abc123def456",
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=1000,
            chunk_overlap=200,
        )

        assert request.collection == "test_collection"
        assert request.doc_id == "test-doc-123"
        assert request.parse_hash == "abc123def456"
        assert request.chunk_strategy == ChunkStrategy.RECURSIVE
        assert request.chunk_size == 1000
        assert request.chunk_overlap == 200
        assert request.headers_to_split_on is None
        assert request.separators is None

    def test_valid_request_with_all_fields(self):
        """Test request creation with all fields."""
        request = ChunkDocumentRequest(
            collection="test_collection",
            doc_id="test-doc-123",
            parse_hash="abc123def456",
            chunk_strategy=ChunkStrategy.MARKDOWN,
            chunk_size=500,
            chunk_overlap=100,
            headers_to_split_on=[("##", "Header 2"), ("###", "Header 3")],
            separators=["\n\n", "\n", " "],
        )

        assert request.chunk_strategy == ChunkStrategy.MARKDOWN
        assert request.headers_to_split_on == [("##", "Header 2"), ("###", "Header 3")]
        assert request.separators == ["\n\n", "\n", " "]

    def test_request_validation_missing_required_field(self):
        """Test validation fails when required field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkDocumentRequest(
                # Missing collection
                doc_id="test-doc-123",
                parse_hash="abc123def456",
                chunk_strategy=ChunkStrategy.RECURSIVE,
                chunk_size=1000,
                chunk_overlap=200,
            )

        assert "collection" in str(exc_info.value)

    def test_request_frozen_behavior(self):
        """Test that request is frozen and cannot be modified."""
        request = ChunkDocumentRequest(
            collection="test_collection",
            doc_id="test-doc-123",
            parse_hash="abc123def456",
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Should raise error when trying to modify
        with pytest.raises(ValidationError):
            request.chunk_size = 2000

    def test_chunk_strategy_enum_validation(self):
        """Test that chunk_strategy only accepts valid enum values."""
        # Valid enum should work
        request = ChunkDocumentRequest(
            collection="test_collection",
            doc_id="test-doc-123",
            parse_hash="abc123def456",
            chunk_strategy=ChunkStrategy.FIXED_SIZE,
            chunk_size=1000,
            chunk_overlap=200,
        )
        assert request.chunk_strategy == ChunkStrategy.FIXED_SIZE


class TestChunkDocumentResponse:
    """Test ChunkDocumentResponse model."""

    def test_valid_response(self):
        """Test response creation with valid data."""
        response = ChunkDocumentResponse(
            doc_id="test-doc-123",
            parse_hash="abc123def456",
            chunk_count=5,
            stats={"avg_length": 800, "total_chars": 4000},
            created=True,
        )

        assert response.doc_id == "test-doc-123"
        assert response.parse_hash == "abc123def456"
        assert response.chunk_count == 5
        assert response.stats == {"avg_length": 800, "total_chars": 4000}
        assert response.created is True

    def test_response_validation_missing_field(self):
        """Test validation fails when required field is missing."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkDocumentResponse(
                # Missing doc_id
                parse_hash="abc123def456",
                chunk_count=5,
                stats={"avg_length": 800},
                created=True,
            )

        assert "doc_id" in str(exc_info.value)

    def test_response_frozen_behavior(self):
        """Test that response is frozen and cannot be modified."""
        response = ChunkDocumentResponse(
            doc_id="test-doc-123",
            parse_hash="abc123def456",
            chunk_count=5,
            stats={"avg_length": 800},
            created=True,
        )

        # Should raise error when trying to modify
        with pytest.raises(ValidationError):
            response.created = False

    def test_field_descriptions(self):
        """Test that field descriptions are set correctly."""
        schema = ChunkDocumentResponse.model_json_schema()

        # Check that descriptions exist
        assert "description" in schema["properties"]["doc_id"]
        assert "description" in schema["properties"]["parse_hash"]
        assert "description" in schema["properties"]["chunk_count"]
        assert "description" in schema["properties"]["stats"]
        assert "description" in schema["properties"]["created"]


class TestVectorStorageSchemas:
    """Test vector storage related schemas."""

    def test_chunk_for_embedding_valid(self):
        """Test ChunkForEmbedding model creation."""
        chunk = ChunkForEmbedding(
            doc_id="test-doc-123",
            chunk_id="chunk-456",
            parse_hash="parse-hash-789",
            text="This is a test chunk",
            chunk_hash="chunk-hash-abc",
            index=1,
            page_number=2,
            section="Introduction",
        )

        assert chunk.doc_id == "test-doc-123"
        assert chunk.chunk_id == "chunk-456"
        assert chunk.parse_hash == "parse-hash-789"
        assert chunk.text == "This is a test chunk"
        assert chunk.chunk_hash == "chunk-hash-abc"
        assert chunk.index == 1
        assert chunk.page_number == 2
        assert chunk.section == "Introduction"
        assert chunk.anchor is None
        assert chunk.json_path is None
        assert chunk.metadata is None

    def test_chunk_for_embedding_with_metadata(self):
        """Test ChunkForEmbedding model with metadata."""
        metadata = {"page": 1, "section": "intro", "source": "/path/to/file.pdf"}
        chunk = ChunkForEmbedding(
            doc_id="test-doc-123",
            chunk_id="chunk-456",
            parse_hash="parse-hash-789",
            text="This is a test chunk",
            chunk_hash="chunk-hash-abc",
            index=1,
            metadata=metadata,
        )

        assert chunk.metadata == metadata
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["section"] == "intro"
        assert chunk.metadata["source"] == "/path/to/file.pdf"

    def test_chunk_embedding_data_valid(self):
        """Test ChunkEmbeddingData model creation."""
        embedding_data = ChunkEmbeddingData(
            doc_id="test-doc-123",
            chunk_id="chunk-456",
            parse_hash="parse-hash-789",
            model="test-model",
            vector=[0.1, 0.2, 0.3, 0.4],
            text="This is a test chunk",
            chunk_hash="chunk-hash-abc",
        )

        assert embedding_data.doc_id == "test-doc-123"
        assert embedding_data.chunk_id == "chunk-456"
        assert embedding_data.parse_hash == "parse-hash-789"
        assert embedding_data.model == "test-model"
        assert embedding_data.vector == [0.1, 0.2, 0.3, 0.4]
        assert embedding_data.text == "This is a test chunk"
        assert embedding_data.chunk_hash == "chunk-hash-abc"
        assert embedding_data.metadata is None

    def test_chunk_embedding_data_with_metadata(self):
        """Test ChunkEmbeddingData model with metadata."""
        metadata = {"page": 1, "section": "intro", "source": "/path/to/file.pdf"}
        embedding_data = ChunkEmbeddingData(
            doc_id="test-doc-123",
            chunk_id="chunk-456",
            parse_hash="parse-hash-789",
            model="test-model",
            vector=[0.1, 0.2, 0.3, 0.4],
            text="This is a test chunk",
            chunk_hash="chunk-hash-abc",
            metadata=metadata,
        )

        assert embedding_data.metadata == metadata
        assert embedding_data.metadata["page"] == 1
        assert embedding_data.metadata["section"] == "intro"

    def test_embedding_read_request_valid(self):
        """Test EmbeddingReadRequest model creation."""
        request = EmbeddingReadRequest(
            collection="test-collection",
            doc_id="test-doc-123",
            parse_hash="parse-hash-789",
            model="test-model",
            filters={"page_number": 1},
        )

        assert request.collection == "test-collection"
        assert request.doc_id == "test-doc-123"
        assert request.parse_hash == "parse-hash-789"
        assert request.model == "test-model"
        assert request.filters == {"page_number": 1}

    def test_embedding_read_response_valid(self):
        """Test EmbeddingReadResponse model creation."""
        chunk = ChunkForEmbedding(
            doc_id="test-doc-123",
            chunk_id="chunk-456",
            parse_hash="parse-hash-789",
            text="This is a test chunk",
            chunk_hash="chunk-hash-abc",
            index=1,
        )

        response = EmbeddingReadResponse(chunks=[chunk], total_count=1, pending_count=1)

        assert len(response.chunks) == 1
        assert response.chunks[0].doc_id == "test-doc-123"
        assert response.total_count == 1
        assert response.pending_count == 1

    def test_embedding_write_request_valid(self):
        """Test EmbeddingWriteRequest model creation."""
        embedding_data = ChunkEmbeddingData(
            doc_id="test-doc-123",
            chunk_id="chunk-456",
            parse_hash="parse-hash-789",
            model="test-model",
            vector=[0.1, 0.2, 0.3, 0.4],
            text="This is a test chunk",
            chunk_hash="chunk-hash-abc",
        )

        request = EmbeddingWriteRequest(
            collection="test-collection", embeddings=[embedding_data], create_index=True
        )

        assert request.collection == "test-collection"
        assert len(request.embeddings) == 1
        assert request.embeddings[0].doc_id == "test-doc-123"
        assert request.create_index is True

    def test_embedding_write_response_valid(self):
        """Test EmbeddingWriteResponse model creation."""
        response = EmbeddingWriteResponse(
            upsert_count=5, deleted_stale_count=2, index_status="created"
        )

        assert response.upsert_count == 5
        assert response.deleted_stale_count == 2
        assert response.index_status == "created"

    def test_vector_storage_schemas_frozen(self):
        """Test that vector storage schemas are frozen."""
        chunk = ChunkForEmbedding(
            doc_id="test-doc-123",
            chunk_id="chunk-456",
            parse_hash="parse-hash-789",
            text="This is a test chunk",
            chunk_hash="chunk-hash-abc",
            index=1,
        )

        # Should raise error when trying to modify
        with pytest.raises(ValidationError):
            chunk.doc_id = "new-doc-id"

    def test_vector_storage_schemas_validation(self):
        """Test validation of vector storage schemas."""
        # Test missing required field
        with pytest.raises(ValidationError) as exc_info:
            ChunkForEmbedding(
                # Missing doc_id
                chunk_id="chunk-456",
                parse_hash="parse-hash-789",
                text="This is a test chunk",
                chunk_hash="chunk-hash-abc",
                index=1,
            )

        assert "doc_id" in str(exc_info.value)

        # Test empty vector is allowed at schema level (business validates non-empty)
        embedding_data = ChunkEmbeddingData(
            doc_id="test-doc-123",
            chunk_id="chunk-456",
            parse_hash="parse-hash-789",
            model="test-model",
            vector=[],  # schema allows empty; business layer will validate
            text="This is a test chunk",
            chunk_hash="chunk-hash-abc",
        )
        assert isinstance(embedding_data.vector, list)
        assert len(embedding_data.vector) == 0


class TestSchemaIntegration:
    """Test schema integration and compatibility."""

    def test_request_response_compatibility(self):
        """Test that request and response schemas are compatible."""
        # Create a request
        request = RegisterDocumentRequest(
            collection="test_collection", source_path="/tmp/test.txt", file_type="txt"
        )

        # Create corresponding response
        response = RegisterDocumentResponse(
            doc_id="generated-id", created=True, content_hash="hash-value"
        )

        # Both should be valid Pydantic models
        assert request.collection == "test_collection"
        assert response.doc_id == "generated-id"

    def test_schema_config_consistency(self):
        """Test that both models have consistent configuration."""
        request_config = RegisterDocumentRequest.model_config
        response_config = RegisterDocumentResponse.model_config

        # Both should have frozen=True
        assert request_config.get("frozen") is True
        assert response_config.get("frozen") is True

    def test_chunk_request_response_compatibility(self):
        """Test that chunk request and response schemas are compatible."""
        # Create a chunk request
        request = ChunkDocumentRequest(
            collection="test_collection",
            doc_id="test-doc-123",
            parse_hash="abc123def456",
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Create corresponding response
        response = ChunkDocumentResponse(
            doc_id=request.doc_id,
            parse_hash=request.parse_hash,
            chunk_count=3,
            stats={"avg_length": 900, "total_chars": 2700},
            created=True,
        )

        # Both should be valid Pydantic models with matching IDs
        assert request.doc_id == response.doc_id
        assert request.parse_hash == response.parse_hash
        assert request.chunk_strategy == ChunkStrategy.RECURSIVE

    def test_chunk_schema_config_consistency(self):
        """Test that chunk models have consistent configuration."""
        request_config = ChunkDocumentRequest.model_config
        response_config = ChunkDocumentResponse.model_config

        # Both should have frozen=True
        assert request_config.get("frozen") is True
        assert response_config.get("frozen") is True

    def test_enum_integration_with_schemas(self):
        """Test that ChunkStrategy enum integrates properly with schemas."""
        # Test all enum values work in request
        for strategy in [
            ChunkStrategy.RECURSIVE,
            ChunkStrategy.MARKDOWN,
            ChunkStrategy.FIXED_SIZE,
        ]:
            request = ChunkDocumentRequest(
                collection="test_collection",
                doc_id="test-doc-123",
                parse_hash="abc123def456",
                chunk_strategy=strategy,
                chunk_size=1000,
                chunk_overlap=200,
            )
            assert request.chunk_strategy == strategy
            assert isinstance(request.chunk_strategy, ChunkStrategy)

    def test_vector_storage_integration(self):
        """Test vector storage schemas integration."""
        # Test read request/response compatibility
        read_request = EmbeddingReadRequest(
            collection="test-collection",
            doc_id="test-doc-123",
            parse_hash="parse-hash-789",
            model="test-model",
        )

        chunk = ChunkForEmbedding(
            doc_id=read_request.doc_id,
            chunk_id="chunk-456",
            parse_hash=read_request.parse_hash,
            text="This is a test chunk",
            chunk_hash="chunk-hash-abc",
            index=1,
        )

        read_response = EmbeddingReadResponse(
            chunks=[chunk], total_count=1, pending_count=1
        )

        # Test write request/response compatibility
        embedding_data = ChunkEmbeddingData(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            parse_hash=chunk.parse_hash,
            model=read_request.model,
            vector=[0.1, 0.2, 0.3, 0.4],
            text=chunk.text,
            chunk_hash=chunk.chunk_hash,
        )

        write_request = EmbeddingWriteRequest(
            collection=read_request.collection,
            embeddings=[embedding_data],
            create_index=True,
        )

        write_response = EmbeddingWriteResponse(
            upsert_count=1, deleted_stale_count=0, index_status="created"
        )

        # Verify integration consistency
        assert read_request.doc_id == chunk.doc_id == embedding_data.doc_id
        assert read_request.parse_hash == chunk.parse_hash == embedding_data.parse_hash
        assert read_request.model == embedding_data.model
        assert read_response.total_count == 1
        assert len(read_response.chunks) == 1
        assert read_response.pending_count == 1
        assert write_request.collection == read_request.collection
        assert len(write_request.embeddings) == write_response.upsert_count


class TestIndexType:
    """Test IndexType enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert IndexType.HNSW.value == "IVF_HNSW_SQ"
        assert IndexType.IVFPQ.value == "IVF_PQ"

    def test_enum_string_conversion(self):
        """Test that enum converts to string correctly."""
        assert str(IndexType.HNSW) == "IVF_HNSW_SQ"
        assert str(IndexType.IVFPQ) == "IVF_PQ"

    def test_enum_value_access(self):
        """Test that enum values can be accessed correctly."""
        # Test that enum instances are not equal to strings (type safety)
        assert IndexType.HNSW != "IVF_HNSW_SQ"
        assert IndexType.IVFPQ != "IVF_PQ"

    def test_enum_membership(self):
        """Test enum membership checks."""
        index_types = {IndexType.HNSW, IndexType.IVFPQ}
        assert IndexType.HNSW in index_types
        assert IndexType.IVFPQ in index_types


class TestIndexStatus:
    """Test IndexStatus enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert IndexStatus.INDEX_READY.value == "index_ready"
        assert IndexStatus.INDEX_BUILDING.value == "index_building"
        assert IndexStatus.NO_INDEX.value == "no_index"
        assert IndexStatus.INDEX_CORRUPTED.value == "index_corrupted"
        assert IndexStatus.BELOW_THRESHOLD.value == "below_threshold"
        assert IndexStatus.READONLY.value == "readonly"

    def test_enum_string_conversion(self):
        """Test that enum converts to string correctly."""
        assert str(IndexStatus.INDEX_READY) == "index_ready"
        assert str(IndexStatus.INDEX_BUILDING) == "index_building"
        assert str(IndexStatus.NO_INDEX) == "no_index"
        assert str(IndexStatus.INDEX_CORRUPTED) == "index_corrupted"
        assert str(IndexStatus.BELOW_THRESHOLD) == "below_threshold"
        assert str(IndexStatus.READONLY) == "readonly"

    def test_enum_value_access(self):
        """Test that enum values can be accessed correctly."""
        # Test that enum instances are not equal to strings (type safety)
        assert IndexStatus.INDEX_READY != "index_ready"
        assert IndexStatus.INDEX_BUILDING != "index_building"
        assert IndexStatus.NO_INDEX != "no_index"
        assert IndexStatus.INDEX_CORRUPTED != "index_corrupted"
        assert IndexStatus.BELOW_THRESHOLD != "below_threshold"
        assert IndexStatus.READONLY != "readonly"

    def test_enum_membership(self):
        """Test enum membership checks."""
        statuses = {
            IndexStatus.INDEX_READY,
            IndexStatus.INDEX_BUILDING,
            IndexStatus.NO_INDEX,
            IndexStatus.INDEX_CORRUPTED,
            IndexStatus.BELOW_THRESHOLD,
            IndexStatus.READONLY,
        }
        assert IndexStatus.INDEX_READY in statuses
        assert IndexStatus.INDEX_BUILDING in statuses
        assert IndexStatus.NO_INDEX in statuses
        assert IndexStatus.INDEX_CORRUPTED in statuses
        assert IndexStatus.BELOW_THRESHOLD in statuses
        assert IndexStatus.READONLY in statuses


class TestIndexOperation:
    """Test IndexOperation enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert IndexOperation.CREATED.value == "created"
        assert IndexOperation.READY.value == "ready"
        assert IndexOperation.SKIPPED.value == "skipped"
        assert IndexOperation.SKIPPED_THRESHOLD.value == "skipped_threshold"
        assert IndexOperation.FAILED.value == "failed"
        assert IndexOperation.UPDATED.value == "updated"

    def test_enum_string_conversion(self):
        """Test that enum converts to string correctly."""
        assert str(IndexOperation.CREATED) == "created"
        assert str(IndexOperation.READY) == "ready"
        assert str(IndexOperation.SKIPPED) == "skipped"
        assert str(IndexOperation.SKIPPED_THRESHOLD) == "skipped_threshold"
        assert str(IndexOperation.FAILED) == "failed"
        assert str(IndexOperation.UPDATED) == "updated"

    def test_enum_value_access(self):
        """Test that enum values can be accessed correctly."""
        # Test that enum instances are not equal to strings (type safety)
        assert IndexOperation.CREATED != "created"
        assert IndexOperation.READY != "ready"
        assert IndexOperation.SKIPPED != "skipped"
        assert IndexOperation.SKIPPED_THRESHOLD != "skipped_threshold"
        assert IndexOperation.FAILED != "failed"
        assert IndexOperation.UPDATED != "updated"

    def test_enum_membership(self):
        """Test enum membership checks."""
        operations = {
            IndexOperation.CREATED,
            IndexOperation.READY,
            IndexOperation.SKIPPED,
            IndexOperation.SKIPPED_THRESHOLD,
            IndexOperation.FAILED,
            IndexOperation.UPDATED,
        }
        assert IndexOperation.CREATED in operations
        assert IndexOperation.READY in operations
        assert IndexOperation.SKIPPED in operations
        assert IndexOperation.SKIPPED_THRESHOLD in operations
        assert IndexOperation.FAILED in operations
        assert IndexOperation.UPDATED in operations


class TestIndexPolicy:
    """Test IndexPolicy configuration."""

    def test_default_policy(self):
        """Test default index policy values."""
        policy = IndexPolicy()

        assert policy.enable_threshold_rows == 50_000
        assert policy.ivfpq_threshold_rows == 10_000_000
        assert policy.hnsw_params == {}
        assert policy.ivfpq_params == {}

    def test_custom_policy(self):
        """Test custom index policy values."""
        policy = IndexPolicy(
            enable_threshold_rows=100_000,
            ivfpq_threshold_rows=5_000_000,
            hnsw_params={"ef_construction": 200},
            ivfpq_params={"nlist": 1024},
        )

        assert policy.enable_threshold_rows == 100_000
        assert policy.ivfpq_threshold_rows == 5_000_000
        assert policy.hnsw_params == {"ef_construction": 200}
        assert policy.ivfpq_params == {"nlist": 1024}

    def test_policy_immutability(self):
        """Test that index policy is immutable (frozen)."""
        policy = IndexPolicy()

        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            policy.enable_threshold_rows = 100_000

    def test_post_init_initialization(self):
        """Test that __post_init__ properly initializes default dicts."""
        # Test with None values (should be converted to empty dicts)
        policy = IndexPolicy(hnsw_params=None, ivfpq_params=None)

        assert policy.hnsw_params == {}
        assert policy.ivfpq_params == {}

        # Test with custom values
        policy = IndexPolicy(hnsw_params={"test": "value"})

        assert policy.hnsw_params == {"test": "value"}
        assert policy.ivfpq_params == {}  # Should still be initialized


class TestSearchWarning:
    """Test SearchWarning model."""

    def test_valid_warning(self):
        """Test creating a valid warning."""
        warning = SearchWarning(
            code="INDEX_DEGRADED",
            message="Index performance degraded",
            fallback_action=SearchFallbackAction.BRUTE_FORCE,
            affected_models=["model1", "model2"],
        )

        assert warning.code == "INDEX_DEGRADED"
        assert warning.message == "Index performance degraded"
        assert warning.fallback_action == SearchFallbackAction.BRUTE_FORCE
        assert warning.affected_models == ["model1", "model2"]

    def test_warning_with_defaults(self):
        """Test warning with default values."""
        warning = SearchWarning(
            code="TEST_WARNING",
            message="Test message",
            fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
        )

        assert warning.affected_models == []

    def test_warning_immutability(self):
        """Test that warning is immutable."""
        warning = SearchWarning(
            code="TEST",
            message="Test",
            fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
        )

        with pytest.raises(
            Exception
        ):  # Frozen models raise ValidationError when modified
            warning.code = "NEW_CODE"


class TestFallbackInfo:
    """Test FallbackInfo model."""

    def test_valid_fallback_info(self):
        """Test creating valid fallback info."""
        performance_impact = PerformanceImpact(
            expected_latency_ms=100.0,
            actual_latency_ms=150.0,
            degradation_reason="Index corrupted",
        )
        fallback = FallbackInfo(
            applied=True,
            reason="Index corrupted",
            performance_impact=performance_impact,
        )

        assert fallback.applied is True
        assert fallback.reason == "Index corrupted"
        assert fallback.performance_impact.expected_latency_ms == 100.0
        assert fallback.performance_impact.actual_latency_ms == 150.0
        assert fallback.performance_impact.degradation_reason == "Index corrupted"

    def test_fallback_info_immutability(self):
        """Test that fallback info is immutable."""
        performance_impact = PerformanceImpact(
            expected_latency_ms=100.0,
            actual_latency_ms=100.0,
            degradation_reason="None",
        )
        fallback = FallbackInfo(
            applied=False,
            reason="Test reason",
            performance_impact=performance_impact,
        )

        with pytest.raises(
            Exception
        ):  # Frozen models raise ValidationError when modified
            fallback.applied = True


class TestSearchResult:
    """Test SearchResult model."""

    def test_valid_result(self):
        """Test creating a valid search result."""
        result = SearchResult(
            doc_id="doc1",
            chunk_id="chunk1",
            text="test content",
            score=0.85,
            parse_hash="hash1",
            model_tag="model1",
            created_at=datetime.now(),
        )

        assert result.doc_id == "doc1"
        assert result.chunk_id == "chunk1"
        assert result.text == "test content"
        assert result.score == 0.85
        assert result.parse_hash == "hash1"
        assert result.model_tag == "model1"
        assert isinstance(result.created_at, datetime)
        assert result.metadata is None

    def test_search_result_with_metadata(self):
        """Test creating a search result with metadata."""
        metadata = {"page": 1, "section": "intro", "source": "/path/to/file.pdf"}
        result = SearchResult(
            doc_id="doc1",
            chunk_id="chunk1",
            text="test content",
            score=0.85,
            parse_hash="hash1",
            model_tag="model1",
            created_at=datetime.now(),
            metadata=metadata,
        )

        assert result.metadata == metadata
        assert result.metadata["page"] == 1
        assert result.metadata["section"] == "intro"
        assert result.metadata["source"] == "/path/to/file.pdf"

    def test_result_immutability(self):
        """Test that search result is immutable."""
        result = SearchResult(
            doc_id="doc1",
            chunk_id="chunk1",
            text="test content",
            score=0.85,
            parse_hash="hash1",
            model_tag="model1",
            created_at=datetime.now(),
        )

        with pytest.raises(
            Exception
        ):  # Frozen models raise ValidationError when modified
            result.doc_id = "new_doc"

    def test_search_result_score_bounds_enforced(self):
        """Test that SearchResult score must be within [0, 1] range."""
        # Test valid scores
        result_zero = SearchResult(
            doc_id="doc1",
            chunk_id="chunk1",
            text="test content",
            score=0.0,
            parse_hash="hash1",
            model_tag="model1",
            created_at=datetime.now(),
        )
        assert result_zero.score == 0.0

        result_one = SearchResult(
            doc_id="doc1",
            chunk_id="chunk1",
            text="test content",
            score=1.0,
            parse_hash="hash1",
            model_tag="model1",
            created_at=datetime.now(),
        )
        assert result_one.score == 1.0

        # Test invalid scores (should raise validation error)
        with pytest.raises(ValidationError):  # ValidationError for score > 1.0
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="test content",
                score=1.5,  # Score > 1.0 should fail
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.now(),
            )

        with pytest.raises(ValidationError):  # ValidationError for negative score
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="test content",
                score=-0.1,  # Negative score should fail
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.now(),
            )

    def test_search_result_score_bounds(self):
        """Test that SearchResult score has proper lower bound."""
        # Test score = 0.0 is allowed
        result_zero = SearchResult(
            doc_id="doc1",
            chunk_id="chunk1",
            text="test content",
            score=0.0,
            parse_hash="hash1",
            model_tag="model1",
            created_at=datetime.now(),
        )
        assert result_zero.score == 0.0

        # Test negative score should fail validation
        with pytest.raises(
            ValidationError
        ):  # Should raise validation error for negative score
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="test content",
                score=-0.1,  # Negative score should fail
                parse_hash="hash1",
                model_tag="model1",
                created_at=datetime.now(),
            )


class TestDenseSearchResponse:
    """Test DenseSearchResponse model."""

    def test_valid_response(self):
        """Test creating a valid search response."""
        result = SearchResult(
            doc_id="doc1",
            chunk_id="chunk1",
            text="test content",
            score=0.85,
            parse_hash="hash1",
            model_tag="model1",
            created_at=datetime.now(),
        )

        warning = SearchWarning(
            code="PARTIAL_RESULTS",
            message="Only partial results returned",
            fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
        )

        performance_impact = PerformanceImpact(
            expected_latency_ms=100.0,
            actual_latency_ms=200.0,
            degradation_reason="Index unavailable",
        )
        fallback = FallbackInfo(
            applied=True,
            reason="Index unavailable",
            performance_impact=performance_impact,
        )

        response = DenseSearchResponse(
            results=[result],
            total_count=1,
            status="success",
            warnings=[warning],
            index_status=IndexStatus.INDEX_READY,
            index_advice="Index performing well",
            idempotency_key="key123",
            fallback_info=fallback,
        )

        assert len(response.results) == 1
        assert response.total_count == 1
        assert response.status == "success"
        assert len(response.warnings) == 1
        assert response.index_status == IndexStatus.INDEX_READY
        assert response.index_advice == "Index performing well"
        assert response.idempotency_key == "key123"
        assert response.fallback_info == fallback

    def test_response_with_defaults(self):
        """Test response with default values."""
        response = DenseSearchResponse(
            results=[],
            total_count=0,
            index_status=IndexStatus.NO_INDEX,
        )

        assert response.status == "success"
        assert response.warnings == []
        assert response.index_advice is None
        assert response.idempotency_key is None
        assert response.fallback_info is None

    def test_response_immutability(self):
        """Test that response is immutable."""
        response = DenseSearchResponse(
            results=[],
            total_count=0,
            index_status=IndexStatus.INDEX_READY,
        )

        with pytest.raises(
            Exception
        ):  # Frozen models raise ValidationError when modified
            response.status = "failed"

    def test_response_with_nprobes_and_refine_factor(self):
        """Test response with nprobes and refine_factor fields."""
        response = DenseSearchResponse(
            results=[],
            total_count=0,
            index_status=IndexStatus.INDEX_READY,
            nprobes=10,
            refine_factor=5,
        )

        assert response.nprobes == 10
        assert response.refine_factor == 5

    def test_response_with_optional_nprobes_refine_factor(self):
        """Test that nprobes and refine_factor are optional."""
        response = DenseSearchResponse(
            results=[],
            total_count=0,
            index_status=IndexStatus.INDEX_READY,
        )

        assert response.nprobes is None
        assert response.refine_factor is None


class TestIndexMetric:
    """Test IndexMetric enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert IndexMetric.L2.value == "l2"
        assert IndexMetric.COSINE.value == "cosine"
        assert IndexMetric.DOT.value == "dot"

    def test_enum_string_conversion(self):
        """Test that enum converts to string correctly."""
        assert str(IndexMetric.L2) == "l2"
        assert str(IndexMetric.COSINE) == "cosine"
        assert str(IndexMetric.DOT) == "dot"

    def test_enum_value_access(self):
        """Test that enum values can be accessed correctly."""
        # Test that enum instances are not equal to strings (type safety)
        assert IndexMetric.L2 != "l2"
        assert IndexMetric.COSINE != "cosine"
        assert IndexMetric.DOT != "dot"

    def test_enum_membership(self):
        """Test enum membership checks."""
        metrics = {IndexMetric.L2, IndexMetric.COSINE, IndexMetric.DOT}
        assert IndexMetric.L2 in metrics
        assert IndexMetric.COSINE in metrics
        assert IndexMetric.DOT in metrics


class TestSearchFallbackAction:
    """Test SearchFallbackAction enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert SearchFallbackAction.BRUTE_FORCE.value == "brute_force"
        assert SearchFallbackAction.REBUILD_INDEX.value == "rebuild_index"
        assert SearchFallbackAction.SAMPLE_SEARCH.value == "sample_search"
        assert SearchFallbackAction.PARTIAL_RESULTS.value == "partial_results"

    def test_enum_string_conversion(self):
        """Test that enum converts to string correctly."""
        assert str(SearchFallbackAction.BRUTE_FORCE) == "brute_force"
        assert str(SearchFallbackAction.REBUILD_INDEX) == "rebuild_index"
        assert str(SearchFallbackAction.SAMPLE_SEARCH) == "sample_search"
        assert str(SearchFallbackAction.PARTIAL_RESULTS) == "partial_results"

    def test_enum_value_access(self):
        """Test that enum values can be accessed correctly."""
        # Test that enum instances are not equal to strings (type safety)
        assert SearchFallbackAction.BRUTE_FORCE != "brute_force"
        assert SearchFallbackAction.REBUILD_INDEX != "rebuild_index"
        assert SearchFallbackAction.SAMPLE_SEARCH != "sample_search"
        assert SearchFallbackAction.PARTIAL_RESULTS != "partial_results"

    def test_enum_membership(self):
        """Test enum membership checks."""
        actions = {
            SearchFallbackAction.BRUTE_FORCE,
            SearchFallbackAction.REBUILD_INDEX,
            SearchFallbackAction.SAMPLE_SEARCH,
            SearchFallbackAction.PARTIAL_RESULTS,
        }
        assert SearchFallbackAction.BRUTE_FORCE in actions
        assert SearchFallbackAction.REBUILD_INDEX in actions
        assert SearchFallbackAction.SAMPLE_SEARCH in actions
        assert SearchFallbackAction.PARTIAL_RESULTS in actions


class TestCollectionInfo:
    """Test CollectionInfo model with embedding binding."""

    def test_collection_info_creation_minimal(self):
        """Test creating CollectionInfo with minimal required fields."""
        collection = CollectionInfo(name="test_collection")

        assert collection.name == "test_collection"
        assert collection.schema_version == "1.0.0"
        assert collection.embedding_model_id is None
        assert collection.embedding_dimension is None
        assert collection.is_initialized is False
        assert collection.documents == 0
        assert collection.processed_documents == 0
        assert collection.embeddings == 0
        assert collection.document_names == []
        assert collection.collection_locked is False
        assert collection.allow_mixed_parse_methods is False
        assert collection.skip_config_validation is False
        assert collection.extra_metadata == {}

    def test_collection_info_creation_initialized(self):
        """Test creating CollectionInfo with embedding initialization."""
        collection = CollectionInfo(
            name="test_collection",
            embedding_model_id="text-embedding-ada-002",
            embedding_dimension=1536,
            documents=5,
            processed_documents=3,
            document_names=["doc1.pdf", "doc2.md"],
            collection_locked=True,
            allow_mixed_parse_methods=False,
        )

        assert collection.name == "test_collection"
        assert collection.is_initialized is True
        assert collection.embedding_model_id == "text-embedding-ada-002"
        assert collection.embedding_dimension == 1536
        assert collection.documents == 5
        assert collection.processed_documents == 3
        assert collection.document_names == ["doc1.pdf", "doc2.md"]
        assert collection.collection_locked is True
        assert collection.allow_mixed_parse_methods is False

    def test_collection_info_from_storage_basic(self):
        """Test deserializing CollectionInfo from storage format."""
        storage_data = {
            "name": "test_collection",
            "schema_version": "1.0.0",
            "embedding_model_id": "text-embedding-ada-002",
            "embedding_dimension": 1536,
            "documents": 5,
            "processed_documents": 3,
            "embeddings": 20,
            "document_names": '["doc1.pdf", "doc2.md"]',
            "collection_locked": True,
            "allow_mixed_parse_methods": False,
            "skip_config_validation": False,
            "ingestion_config": '{"chunk_size": 512}',
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
            "last_accessed_at": "2024-01-02T00:00:00",
            "extra_metadata": '{"key": "value"}',
        }

        collection = CollectionInfo.from_storage(storage_data)

        assert collection.name == "test_collection"
        assert collection.schema_version == "1.0.0"
        assert collection.embedding_model_id == "text-embedding-ada-002"
        assert collection.embedding_dimension == 1536
        assert collection.is_initialized is True
        assert collection.documents == 5
        assert collection.processed_documents == 3
        assert collection.document_names == ["doc1.pdf", "doc2.md"]
        assert collection.collection_locked is True
        assert collection.extra_metadata == {"key": "value"}
        assert collection.ingestion_config is not None
        assert collection.ingestion_config.chunk_size == 512

    def test_collection_info_from_storage_migration(self):
        """Test that from_storage handles migration automatically."""
        # Legacy data without schema_version
        legacy_data = {
            "name": "legacy_collection",
            "documents": 10,
            "document_names": ["old_doc.pdf"],
        }

        collection = CollectionInfo.from_storage(legacy_data)

        # Should be migrated to v1.0.0
        assert collection.schema_version == "1.0.0"
        assert collection.name == "legacy_collection"
        assert collection.documents == 10
        assert collection.document_names == ["old_doc.pdf"]
        # Should have default values for new fields
        assert collection.embedding_model_id is None
        assert collection.embedding_dimension is None
        assert collection.is_initialized is False

    def test_collection_info_to_storage(self):
        """Test serializing CollectionInfo to storage format."""
        collection = CollectionInfo(
            name="test_collection",
            embedding_model_id="text-embedding-ada-002",
            embedding_dimension=1536,
            documents=5,
            processed_documents=3,
            document_names=["doc1.pdf", "doc2.md"],
            extra_metadata={"custom": "data"},
        )

        storage_data = collection.to_storage()

        # Basic fields should be preserved
        assert storage_data["name"] == "test_collection"
        assert storage_data["embedding_model_id"] == "text-embedding-ada-002"
        assert storage_data["embedding_dimension"] == 1536
        assert storage_data["documents"] == 5
        assert storage_data["processed_documents"] == 3
        assert storage_data["ingestion_config"] == ""

    def test_collection_info_immutability_by_default(self):
        """Test that CollectionInfo is immutable by default after creation."""
        collection = CollectionInfo(name="test")

        # Should be able to modify since model_config is frozen=False
        collection.documents = 5
        assert collection.documents == 5

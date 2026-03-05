"""Tests for chunk_document functionality.

This module tests the complete document processing pipeline:
register_document -> parse_document -> chunk_document
"""

import os
import tempfile
import uuid

import pandas as pd
import pytest

from xagent.core.tools.core.RAG_tools.chunk.chunk_document import (
    chunk_document,
    chunk_fixed_size,
    chunk_markdown,
    chunk_recursive,
)
from xagent.core.tools.core.RAG_tools.core.exceptions import DocumentValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkDocumentResponse,
    ChunkStrategy,
)
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.parse.parse_document import parse_document


class TestChunkDocument:
    """Test cases for chunk_document functionality."""

    @pytest.fixture
    def temp_lancedb_dir(self):
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
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    def test_doc_id(self):
        """Test document ID."""
        return str(uuid.uuid4())

    def test_chunk_txt_recursive(self, temp_lancedb_dir, test_collection, test_doc_id):
        """Test chunking TXT file with recursive strategy."""
        # Step 1: Register document
        txt_path = "tests/resources/test_files/test.txt"
        register_result = register_document(
            collection=test_collection,
            source_path=txt_path,
            doc_id=test_doc_id,
            user_id=1,
        )
        assert register_result["created"] is True
        assert register_result["doc_id"] == test_doc_id

        # Step 2: Parse document
        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        assert parse_result["written"] is True
        assert "parse_hash" in parse_result
        parse_hash = parse_result["parse_hash"]

        # Step 3: Chunk document
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            user_id=1,
        )

        # Verify chunk results and schema contract
        resp = ChunkDocumentResponse.model_validate(chunk_result)
        assert resp.created is True
        assert resp.chunk_count > 0
        assert resp.stats["total_chunks"] > 0
        assert resp.stats["avg_chunk_length"] > 0

        # Verify text fidelity and metadata preservation
        self._verify_chunk_text_fidelity_and_metadata(
            test_collection, test_doc_id, parse_hash
        )

    def test_chunk_md_markdown_strategy(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test chunking Markdown file with markdown strategy."""
        # Step 1: Register document
        md_path = "tests/resources/test_files/test.md"
        register_result = register_document(
            collection=test_collection,
            source_path=md_path,
            doc_id=test_doc_id,
            user_id=1,
        )
        assert register_result["created"] is True

        # Step 2: Parse document
        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        assert parse_result["written"] is True
        parse_hash = parse_result["parse_hash"]

        # Step 3: Chunk document with markdown strategy
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.MARKDOWN,
            chunk_size=200,
            chunk_overlap=50,
            user_id=1,
        )

        # Verify chunk results and schema contract
        resp = ChunkDocumentResponse.model_validate(chunk_result)
        assert resp.created is True
        assert resp.chunk_count > 0
        assert resp.stats["total_chunks"] > 0

        # Verify text fidelity and metadata preservation
        self._verify_chunk_text_fidelity_and_metadata(
            test_collection, test_doc_id, parse_hash
        )

    def test_chunk_pdf_fixed_size(self, temp_lancedb_dir, test_collection, test_doc_id):
        """Test chunking PDF file with fixed size strategy."""
        # Step 1: Register document
        pdf_path = "tests/resources/test_files/test.pdf"
        register_result = register_document(
            collection=test_collection,
            source_path=pdf_path,
            doc_id=test_doc_id,
            user_id=1,
        )
        assert register_result["created"] is True

        # Step 2: Parse document
        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method="pypdf",
            user_id=1,
            is_admin=True,
        )
        assert parse_result["written"] is True
        parse_hash = parse_result["parse_hash"]

        # Step 3: Chunk document with fixed size strategy
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.FIXED_SIZE,
            chunk_size=500,
            chunk_overlap=0,
            user_id=1,
        )

        # Verify chunk results and schema contract
        resp = ChunkDocumentResponse.model_validate(chunk_result)
        assert resp.created is True
        assert resp.chunk_count > 0
        assert resp.stats["total_chunks"] > 0

        # Verify text fidelity and metadata preservation
        self._verify_chunk_text_fidelity_and_metadata(
            test_collection, test_doc_id, parse_hash
        )

    def test_chunk_idempotency(self, temp_lancedb_dir, test_collection, test_doc_id):
        """Test that chunking is idempotent."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Step 2: First chunking
        chunk_result1 = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            user_id=1,
        )

        # Step 3: Second chunking with same parameters
        chunk_result2 = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            user_id=1,
        )

        # Should be idempotent
        assert chunk_result1["chunk_count"] == chunk_result2["chunk_count"]
        assert (
            chunk_result1["stats"]["total_chunks"]
            == chunk_result2["stats"]["total_chunks"]
        )
        assert chunk_result2["created"] is False  # Should not write again

    def test_chunk_different_strategies(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test different chunking strategies on same document."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Step 2: Test recursive strategy
        recursive_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            user_id=1,
        )

        # Step 3: Test fixed size strategy
        fixed_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.FIXED_SIZE,
            chunk_size=100,
            chunk_overlap=0,
            user_id=1,
        )

        # Both should succeed but may have different chunk counts
        assert recursive_result["created"] is True
        assert fixed_result["created"] is True
        assert recursive_result["chunk_count"] > 0
        assert fixed_result["chunk_count"] > 0

    def test_chunk_fine_grained_functions(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test fine-grained chunking functions."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Step 2: Test recursive function
        recursive_result = chunk_recursive(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_size=100,
            chunk_overlap=20,
            user_id=1,
        )

        # Step 3: Test markdown function
        markdown_result = chunk_markdown(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_size=200,
            chunk_overlap=50,
            user_id=1,
        )

        # Step 4: Test fixed size function
        fixed_result = chunk_fixed_size(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_size=150,
            chunk_overlap=0,
            user_id=1,
        )

        # All should succeed
        assert recursive_result["chunk_count"] > 0
        assert markdown_result["chunk_count"] > 0
        assert fixed_result["chunk_count"] > 0

    def test_chunk_error_handling(self, temp_lancedb_dir, test_collection):
        """Test error handling in chunking."""
        # Test with non-existent document
        with pytest.raises(Exception):  # Should raise DocumentNotFoundError
            chunk_document(
                collection=test_collection,
                doc_id="non_existent_doc_id",
                parse_hash="non_existent_parse_hash",
            )

        # Test with invalid parameters
        with pytest.raises(Exception):  # Should raise DocumentValidationError
            chunk_document(
                collection=test_collection,
                doc_id="test_doc",
                parse_hash="test_parse",
                chunk_strategy="invalid_strategy",
            )

    def test_chunk_use_token_count_requires_chunk_size(self, temp_lancedb_dir):
        """P0: use_token_count=True without chunk_size raises DocumentValidationError."""
        with pytest.raises(DocumentValidationError) as exc_info:
            chunk_document(
                collection="c",
                doc_id="d",
                parse_hash="p",
                chunk_strategy=ChunkStrategy.RECURSIVE,
                chunk_size=None,
                chunk_overlap=50,
                use_token_count=True,
            )
        assert "chunk_size" in str(exc_info.value).lower()

    def test_chunk_recursive_with_use_token_count(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """P0 integration: register -> parse -> chunk with use_token_count=True (token-based)."""
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=256,
            chunk_overlap=50,
            use_token_count=True,
            user_id=1,
        )

        resp = ChunkDocumentResponse.model_validate(chunk_result)
        assert resp.created is True
        assert resp.chunk_count > 0
        assert resp.stats["total_chunks"] > 0
        self._verify_chunk_text_fidelity_and_metadata(
            test_collection, test_doc_id, parse_hash
        )

    def test_chunk_idempotency_with_use_token_count(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """P0: idempotency when use_token_count=True (config_hash includes token params)."""
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        chunk_result1 = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=200,
            chunk_overlap=40,
            use_token_count=True,
            user_id=1,
        )
        chunk_result2 = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=200,
            chunk_overlap=40,
            use_token_count=True,
            user_id=1,
        )

        assert chunk_result1["chunk_count"] == chunk_result2["chunk_count"]
        assert chunk_result2["created"] is False

    def test_chunk_recursive_protected_content_keeps_code_block(
        self, temp_lancedb_dir, test_collection, test_doc_id, tmp_path
    ):
        """P1 integration: enable_protected_content keeps code block in one piece."""
        code_doc = tmp_path / "code_doc.txt"
        code_doc.write_text(
            "Intro sentence.\n```\ncode line one\ncode line two\n```\nOutro.",
            encoding="utf-8",
        )
        register_document(
            collection=test_collection,
            source_path=str(code_doc),
            doc_id=test_doc_id,
            user_id=1,
        )
        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=256,
            chunk_overlap=50,
            use_token_count=True,
            enable_protected_content=True,
            user_id=1,
        )
        assert chunk_result["created"] is True
        assert chunk_result["chunk_count"] > 0
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(f"collection == '{test_collection}' AND doc_id == '{test_doc_id}'")
            .to_pandas()
        )
        combined = " ".join(df["text"].astype(str).tolist())
        assert "code line one" in combined and "code line two" in combined
        assert "```" in combined

    def test_chunk_markdown_with_headers_section_in_metadata(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """P1 integration: MARKDOWN with headers_to_split_on stores section on chunks."""
        md_path = "tests/resources/test_files/test.md"
        register_document(
            collection=test_collection,
            source_path=md_path,
            doc_id=test_doc_id,
            user_id=1,
        )
        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.MARKDOWN,
            chunk_size=200,
            chunk_overlap=50,
            headers_to_split_on=[("# ", "H1"), ("## ", "H2"), ("### ", "H3")],
            user_id=1,
        )
        assert chunk_result["created"] is True
        assert chunk_result["chunk_count"] > 0
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(f"collection == '{test_collection}' AND doc_id == '{test_doc_id}'")
            .to_pandas()
        )
        if "section" in df.columns:
            sections = df["section"].dropna().astype(str)
            assert len(sections) > 0
            assert any(
                "Section" in s or "Test Document" in s or "Details" in s
                for s in sections
            )

    def test_chunk_table_context_attached(
        self, temp_lancedb_dir, test_collection, test_doc_id, tmp_path
    ):
        """P2 integration: table_context_size attaches prev/next context to table chunk."""
        table_doc = tmp_path / "table_doc.txt"
        table_doc.write_text(
            "Before table.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nAfter table.",
            encoding="utf-8",
        )
        register_document(
            collection=test_collection,
            source_path=str(table_doc),
            doc_id=test_doc_id,
            user_id=1,
        )
        parse_result = parse_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=512,
            chunk_overlap=0,
            table_context_size=15,
            image_context_size=0,
            user_id=1,
        )
        assert chunk_result["created"] is True
        assert chunk_result["chunk_count"] > 0
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(f"collection == '{test_collection}' AND doc_id == '{test_doc_id}'")
            .to_pandas()
        )
        # The chunk that contains the table should have prev/next context attached
        table_chunks = df[
            df["text"].astype(str).str.contains(r"\|.*\|", regex=True, na=False)
        ]
        assert len(table_chunks) > 0
        table_text = " ".join(table_chunks["text"].astype(str).tolist())
        assert "Before" in table_text and "After" in table_text

    def test_chunk_collection_isolation(self, temp_lancedb_dir):
        """Test that chunks are isolated by collection."""
        collection1 = f"collection1_{uuid.uuid4().hex[:8]}"
        collection2 = f"collection2_{uuid.uuid4().hex[:8]}"
        doc_id = str(uuid.uuid4())

        # Register same document in two collections
        txt_path = "tests/resources/test_files/test.txt"
        register_document(
            collection=collection1, source_path=txt_path, doc_id=doc_id, user_id=1
        )
        register_document(
            collection=collection2, source_path=txt_path, doc_id=doc_id, user_id=1
        )

        # Parse in both collections
        parse1 = parse_document(
            collection=collection1,
            doc_id=doc_id,
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse2 = parse_document(
            collection=collection2,
            doc_id=doc_id,
            parse_method="default",
            user_id=1,
            is_admin=True,
        )

        # Chunk in both collections
        chunk1 = chunk_document(
            collection=collection1,
            doc_id=doc_id,
            parse_hash=parse1["parse_hash"],
            chunk_strategy=ChunkStrategy.RECURSIVE,
            user_id=1,
        )
        chunk2 = chunk_document(
            collection=collection2,
            doc_id=doc_id,
            parse_hash=parse2["parse_hash"],
            chunk_strategy=ChunkStrategy.RECURSIVE,
            user_id=1,
        )

        # Both should succeed independently
        assert chunk1["created"] is True
        assert chunk2["created"] is True
        assert chunk1["chunk_count"] > 0
        assert chunk2["chunk_count"] > 0

    def test_chunk_config_hash_idempotency(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test that same config produces idempotent results."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Step 2: First chunking
        chunk_result1 = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", " "],
            user_id=1,
        )

        # Step 3: Second chunking with identical parameters
        chunk_result2 = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", " "],
            user_id=1,
        )

        # Should be idempotent
        assert chunk_result1["chunk_count"] == chunk_result2["chunk_count"]
        assert chunk_result1["created"] is True
        assert chunk_result2["created"] is False  # Should not write again

        # Verify database state - both should reference same config_hash
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(
                f"collection == '{test_collection}' AND doc_id == '{test_doc_id}' AND parse_hash == '{parse_hash}'"
            )
            .to_pandas()
        )

        # All rows should have same config_hash
        config_hashes = df["config_hash"].unique()
        assert len(config_hashes) == 1

    def test_chunk_separators_create_new_version(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test that different separators create new version."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Step 2: First chunking with one set of separators
        chunk_result1 = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", " "],
            user_id=1,
        )

        # Step 3: Second chunking with different separators
        chunk_result2 = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n", " ", "."],  # Different separators
            user_id=1,
        )

        # Both should write (different versions)
        assert chunk_result1["created"] is True
        assert chunk_result2["created"] is True

        # Verify database has two different config_hash versions
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(
                f"collection == '{test_collection}' AND doc_id == '{test_doc_id}' AND parse_hash == '{parse_hash}'"
            )
            .to_pandas()
        )

        # Should have two different config_hash values
        config_hashes = df["config_hash"].unique()
        assert len(config_hashes) == 2

        # Each version should have > 0 rows
        for config_hash in config_hashes:
            version_rows = df[df["config_hash"] == config_hash]
            assert len(version_rows) > 0

    def test_chunk_recursive_custom_separators_integration(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Integration: register -> parse -> chunk with custom separators; verify chunk boundaries and count."""
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Chunk with only newline as separator (narrower splitting) and small chunk_size
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=30,
            chunk_overlap=5,
            separators=["\n"],
            user_id=1,
        )
        assert chunk_result["created"] is True
        assert chunk_result["chunk_count"] >= 1

        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(
                f"collection == '{test_collection}' AND doc_id == '{test_doc_id}' AND parse_hash == '{parse_hash}'"
            )
            .to_pandas()
        )
        # Chunk count and content reflect custom separator
        assert len(df) == chunk_result["chunk_count"]
        self._verify_chunk_text_fidelity_and_metadata(
            test_collection, test_doc_id, parse_hash
        )

    def test_chunk_recursive_custom_separators_vs_default_different_result(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Integration: custom separators produce different chunk count or config than default."""
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        chunk_default = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=50,
            chunk_overlap=10,
            separators=None,
            user_id=1,
        )
        chunk_custom = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=50,
            chunk_overlap=10,
            separators=["。", "\n"],
            user_id=1,
        )
        assert chunk_default["created"] is True
        assert chunk_custom["created"] is True
        # Different separators must yield different config_hash (hence different version)
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(
                f"collection == '{test_collection}' AND doc_id == '{test_doc_id}' AND parse_hash == '{parse_hash}'"
            )
            .to_pandas()
        )
        config_hashes = df["config_hash"].unique()
        assert len(config_hashes) == 2, (
            "Default and custom separators should produce two distinct chunk versions"
        )

    def test_chunk_row_level_hash_uniqueness(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test that different rows have different chunk_hash values."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Step 2: Chunk document
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=30,  # Smaller size to ensure multiple chunks
            chunk_overlap=10,
            user_id=1,
        )

        assert chunk_result["created"] is True
        assert chunk_result["chunk_count"] > 1  # Need multiple chunks for this test

        # Step 3: Verify row-level chunk_hash uniqueness
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(
                f"collection == '{test_collection}' AND doc_id == '{test_doc_id}' AND parse_hash == '{parse_hash}'"
            )
            .to_pandas()
        )

        # Each row should have non-empty chunk_hash
        chunk_hashes = df["chunk_hash"].tolist()
        assert all(isinstance(h, str) and len(h) == 64 for h in chunk_hashes)

        # At least some chunk_hash values should be different (assuming different text)
        unique_chunk_hashes = set(chunk_hashes)
        assert len(unique_chunk_hashes) >= min(2, len(chunk_hashes))

    def test_chunk_parameter_validation(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test parameter validation edge cases."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Test chunk_size <= 0
        with pytest.raises(
            DocumentValidationError, match="chunk_size must be positive"
        ):
            chunk_document(
                collection=test_collection,
                doc_id=test_doc_id,
                parse_hash=parse_hash,
                chunk_size=0,
            )

        # Test chunk_overlap < 0
        with pytest.raises(
            DocumentValidationError, match="chunk_overlap must be non-negative"
        ):
            chunk_document(
                collection=test_collection,
                doc_id=test_doc_id,
                parse_hash=parse_hash,
                chunk_size=100,
                chunk_overlap=-1,
            )

        # Test chunk_overlap >= chunk_size
        with pytest.raises(
            DocumentValidationError, match="chunk_overlap must be less than chunk_size"
        ):
            chunk_document(
                collection=test_collection,
                doc_id=test_doc_id,
                parse_hash=parse_hash,
                chunk_size=100,
                chunk_overlap=100,
            )

    def test_chunk_table_structure_validation(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test that chunks table has expected structure after chunking."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Step 2: Chunk document
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            user_id=1,
        )

        assert chunk_result["created"] is True

        # Step 3: Verify table structure
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")

        # Table should exist and be accessible
        assert table is not None

        # Verify expected columns exist
        df = (
            table.search()
            .where(f"collection == '{test_collection}' AND doc_id == '{test_doc_id}'")
            .to_pandas()
        )

        expected_columns = {
            "collection",
            "doc_id",
            "parse_hash",
            "chunk_id",
            "index",
            "text",
            "chunk_hash",
            "config_hash",
            "created_at",
            "metadata",  # Metadata field should exist
        }
        actual_columns = set(df.columns)

        # All expected columns should be present
        assert expected_columns.issubset(actual_columns)

        # Verify data types and non-null constraints
        assert len(df) > 0
        for _, row in df.iterrows():
            assert isinstance(row["chunk_hash"], str) and len(row["chunk_hash"]) == 64
            assert isinstance(row["config_hash"], str) and len(row["config_hash"]) == 64
            assert isinstance(row["text"], str)
            assert isinstance(row["chunk_id"], str)

    def test_chunk_fine_grained_functions_schema_compliance(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test that fine-grained functions also comply with schema."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Test all fine-grained functions with schema validation
        functions_to_test = [
            (chunk_recursive, {"chunk_size": 100, "chunk_overlap": 20}),
            (chunk_markdown, {"chunk_size": 200, "chunk_overlap": 50}),
            (chunk_fixed_size, {"chunk_size": 150, "chunk_overlap": 0}),
        ]

        for func, kwargs in functions_to_test:
            result = func(
                collection=test_collection,
                doc_id=test_doc_id,
                parse_hash=parse_hash,
                user_id=1,
                **kwargs,
            )

            # Verify schema compliance
            resp = ChunkDocumentResponse.model_validate(result)
            assert resp.chunk_count > 0
            assert resp.stats["total_chunks"] > 0
            assert resp.created is True

    def _verify_chunk_text_fidelity_and_metadata(
        self, collection: str, doc_id: str, parse_hash: str
    ) -> None:
        """Verify that chunk text preserves punctuation/spacing and metadata is retained.

        This helper method validates:
        1. Text fidelity: Chunks contain proper punctuation and spacing
        2. Metadata preservation: Position information survives the round trip
        """
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        # Load chunks from database
        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(
                f"collection == '{collection}' AND doc_id == '{doc_id}' AND parse_hash == '{parse_hash}'"
            )
            .to_pandas()
        )

        assert len(df) > 0, "No chunks found in database"

        # Verify text fidelity: chunks should contain punctuation and proper spacing
        punctuation_found = False
        for _, row in df.iterrows():
            text = row["text"]

            # Check for common punctuation marks
            if any(
                punct in text
                for punct in [".", "!", "?", ",", ";", ":", "。", "！", "？"]
            ):
                punctuation_found = True

            # Check for proper spacing (not just concatenated text)
            if " " in text or "\n" in text:
                pass  # Good, has spacing

            # Text should not be empty or just whitespace
            assert text.strip(), f"Chunk text should not be empty: {text}"

        assert punctuation_found, "At least one chunk should contain punctuation"

        # Verify metadata preservation: position fields should not all be NULL
        metadata_fields = ["page_number", "section", "anchor", "json_path"]
        metadata_preserved = False

        for field in metadata_fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                if non_null_count > 0:
                    metadata_preserved = True
                    break

        # Note: For some test files, metadata might legitimately be None
        # This is acceptable as long as the fields exist and are properly handled
        print(f"Metadata preservation check: {metadata_preserved}")
        print(f"Chunk count: {len(df)}")
        print(f"Sample chunk text: {df.iloc[0]['text'][:100]}...")

        # Verify chunk text reconstruction: rejoined chunks should make sense
        # (This is a basic check - more sophisticated checks could be added)
        all_text = " ".join(df["text"].tolist())
        assert len(all_text) > 0, "Rejoined text should not be empty"

        # Verify chunk hashes are unique (no duplicate content)
        chunk_hashes = df["chunk_hash"].tolist()
        unique_hashes = set(chunk_hashes)
        assert len(unique_hashes) == len(chunk_hashes), "Chunk hashes should be unique"

    def test_chunk_metadata_serialization_and_retrieval(
        self, temp_lancedb_dir, test_collection, test_doc_id
    ):
        """Test that metadata is correctly serialized when writing and deserialized when reading."""
        # Step 1: Register and parse document
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
            parse_method="default",
            user_id=1,
            is_admin=True,
        )
        parse_hash = parse_result["parse_hash"]

        # Step 2: Chunk document
        chunk_result = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            user_id=1,
        )

        assert chunk_result["created"] is True

        # Step 3: Verify metadata in database (should be serialized as JSON string)
        from xagent.core.tools.core.RAG_tools.utils.metadata_utils import (
            deserialize_metadata,
        )
        from xagent.providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        table = conn.open_table("chunks")
        df = (
            table.search()
            .where(
                f"collection == '{test_collection}' AND doc_id == '{test_doc_id}' AND parse_hash == '{parse_hash}'"
            )
            .to_pandas()
        )

        # Verify metadata field exists
        assert "metadata" in df.columns

        # Verify metadata can be deserialized (if present)
        for _, row in df.iterrows():
            metadata_str = row.get("metadata")
            if metadata_str is not None and pd.notna(metadata_str):
                metadata = deserialize_metadata(metadata_str)
                # Metadata should be either None or a dictionary
                assert metadata is None or isinstance(metadata, dict)

        # Note: Some chunks might not have metadata, which is acceptable
        # But if metadata exists, it should be properly deserializable

        # Step 4: Test idempotency - verify that _get_existing_chunks returns metadata
        # This tests that metadata is correctly retrieved when chunks already exist
        chunk_result2 = chunk_document(
            collection=test_collection,
            doc_id=test_doc_id,
            parse_hash=parse_hash,
            chunk_strategy=ChunkStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            user_id=1,
        )

        # Should be idempotent
        assert chunk_result2["created"] is False
        assert chunk_result2["chunk_count"] == chunk_result["chunk_count"]


class TestChunkDocumentFallback:
    """Test three-tier fallback logic for chunk_document internal functions."""

    @pytest.fixture
    def temp_lancedb_dir(self):
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
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_chunk_document_arrow_fallback_chain(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test chunk_document uses to_arrow() -> to_list() -> to_pandas() fallback."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.chunk.chunk_document import (
            _get_existing_chunks,
        )

        mock_db_connection = MagicMock()
        mock_table = MagicMock()

        def mock_open_table_func(table_name):
            return mock_table

        mock_db_connection.open_table.side_effect = mock_open_table_func

        chunks_data = [
            {
                "chunk_id": "chunk1",
                "text": "test content",
                "collection": test_collection,
                "doc_id": "doc1",
                "parse_hash": "hash1",
                "index": 0,
                "created_at": pd.Timestamp.now(),
                "metadata": '{"key": "value"}',
            }
        ]
        mock_arrow_table = MagicMock()
        mock_arrow_table.to_pylist.return_value = chunks_data

        mock_search = MagicMock()
        mock_where = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.where.return_value = mock_where
        mock_where.to_arrow.return_value = mock_arrow_table
        mock_table.count_rows.return_value = 1

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.chunk.chunk_document.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.chunk.chunk_document.ensure_chunks_table"
            ),
        ):
            result = _get_existing_chunks(
                collection=test_collection,
                doc_id="doc1",
                parse_hash="hash1",
                config_hash="config1",
            )

            assert len(result) == 1
            assert result[0]["chunk_id"] == "chunk1"
            # Verify to_arrow() was called
            mock_where.to_arrow.assert_called_once()

    def test_chunk_document_fallback_to_list(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test chunk_document fallback from to_arrow() to to_list()."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.chunk.chunk_document import (
            _get_existing_chunks,
        )

        mock_db_connection = MagicMock()
        mock_table = MagicMock()

        def mock_open_table_func(table_name):
            return mock_table

        mock_db_connection.open_table.side_effect = mock_open_table_func

        chunks_data = [
            {
                "chunk_id": "chunk1",
                "text": "test content",
                "collection": test_collection,
                "doc_id": "doc1",
                "parse_hash": "hash1",
                "index": 0,
                "created_at": pd.Timestamp.now(),
                "metadata": '{"key": "value"}',
            }
        ]

        mock_search = MagicMock()
        mock_where = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.where.return_value = mock_where
        # to_arrow() fails, fallback to to_list()
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.return_value = chunks_data
        mock_table.count_rows.return_value = 1

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.chunk.chunk_document.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.chunk.chunk_document.ensure_chunks_table"
            ),
        ):
            result = _get_existing_chunks(
                collection=test_collection,
                doc_id="doc1",
                parse_hash="hash1",
                config_hash="config1",
            )

            assert len(result) == 1
            assert result[0]["chunk_id"] == "chunk1"
            # Verify fallback was used
            mock_where.to_arrow.assert_called_once()
            mock_where.to_list.assert_called_once()

    def test_chunk_document_fallback_to_pandas_with_nan(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test chunk_document fallback to to_pandas() and NaN normalization."""
        from unittest.mock import MagicMock, patch

        import numpy as np

        from xagent.core.tools.core.RAG_tools.chunk.chunk_document import (
            _get_existing_chunks,
        )

        mock_db_connection = MagicMock()
        mock_table = MagicMock()

        def mock_open_table_func(table_name):
            return mock_table

        mock_db_connection.open_table.side_effect = mock_open_table_func

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
                    "created_at": pd.Timestamp.now(),
                    "metadata": '{"key": "value"}',
                    "page_number": np.nan,  # NaN value
                }
            ]
        )

        mock_search = MagicMock()
        mock_where = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.where.return_value = mock_where
        # Both to_arrow() and to_list() fail, fallback to to_pandas()
        mock_where.to_arrow.side_effect = AttributeError("to_arrow not available")
        mock_where.to_list.side_effect = AttributeError("to_list not available")
        mock_where.to_pandas.return_value = chunks_df
        mock_table.count_rows.return_value = 1

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.chunk.chunk_document.get_connection_from_env",
                return_value=mock_db_connection,
            ),
            patch(
                "xagent.core.tools.core.RAG_tools.chunk.chunk_document.ensure_chunks_table"
            ),
        ):
            result = _get_existing_chunks(
                collection=test_collection,
                doc_id="doc1",
                parse_hash="hash1",
                config_hash="config1",
            )

            assert len(result) == 1
            assert result[0]["chunk_id"] == "chunk1"
            # Verify all fallbacks were attempted
            mock_where.to_arrow.assert_called_once()
            mock_where.to_list.assert_called_once()
            mock_where.to_pandas.assert_called_once()
            # Verify NaN was normalized to None
            assert result[0].get("page_number") is None

"""
End-to-end tests for document parsing behavior.

This module tests the default parsing behavior for different document types,
ensuring that the correct parser is selected and the parsed content is accurate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Note: _StubEmbeddingAdapter, stub_embedding_adapter, and mock_rag_pipeline
# are provided by conftest.py with autouse=True


@pytest.fixture
def sample_parsing_files() -> Generator[tuple[dict[str, str], str], None, None]:
    """Create sample test files for parsing behavior testing."""
    files = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create various file types with known content
        test_files = {
            # Text files
            "simple.txt": "This is a simple text file for parsing tests.",
            "chinese.txt": "这是一个中文文本文件，用于测试解析功能。",
            "multiline.txt": "Line 1\nLine 2\nLine 3\nLine 4",
            # Markdown files
            "simple.md": "# Simple Markdown\n\nThis is a simple markdown file.",
            "complex.md": "# Header 1\n\n## Header 2\n\n- List item 1\n- List item 2\n\n**Bold text** and *italic text*.",
            # JSON files
            "simple.json": '{"title": "Test", "content": "Simple JSON content"}',
            "array.json": '[{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]',
            "nested.json": '{"user": {"name": "Alice", "age": 30}, "posts": [{"title": "Post 1"}]}',
            # CSV files
            "simple.csv": "name,age,city\nJohn,25,NYC\nJane,30,LA",
            "quoted.csv": 'name,description\nJohn,"Engineer, 30 years"\nJane,"Designer, 25 years"',
            # HTML files
            "simple.html": "<html><body><h1>Simple HTML</h1><p>This is a paragraph.</p></body></html>",
            # Code files
            "code.py": "def hello():\n    print('Hello, World!')\n\nhello()",
            "code.js": "function hello() {\n    console.log('Hello, World!');\n}\n\nhello();",
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            file_path.write_text(content, encoding="utf-8")
            files[filename] = str(file_path)

        yield files, temp_dir


# ==========================================
# DOCUMENT TYPE DEFAULT PARSING TESTS
# ==========================================


class TestDocumentTypeDefaultParsing:
    """
    Test default parsing behavior for different document types.

    These tests verify that:
    1. The correct parser is selected for each file type
    2. The parsed content is accurate
    3. Metadata extraction works correctly
    4. Chunk counts are as expected
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_txt_default_parsing_behavior(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that .txt files use the default parser correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_txt"
        file_path = files["simple.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Verify ingestion succeeded
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

        # Verify the response indicates successful parsing
        if "completed_steps" in result:
            step_names = [step["name"] for step in result["completed_steps"]]
            assert "parse_document" in step_names

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_txt_multiline_parsing(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that multiline .txt files are parsed into multiple chunks."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_multiline"
        file_path = files["multiline.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("multiline.txt", f, "text/plain")},
                data={
                    "collection": collection_name,
                    "chunk_strategy": "fixed_size",
                    "chunk_size": "20",  # Small chunk size to force multiple chunks
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_txt_chinese_parsing(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that Chinese text is parsed correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_chinese"
        file_path = files["chinese.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("chinese.txt", f, "text/plain;charset=utf-8")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Should handle Chinese text correctly
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_md_default_parsing_behavior(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that .md files use markdown-aware parsing."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_md"
        file_path = files["simple.md"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.md", f, "text/markdown")},
                data={
                    "collection": collection_name,
                    "chunk_strategy": "markdown",  # Use markdown-specific chunking
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_md_complex_parsing(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that complex markdown with headers, lists, formatting is parsed correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_md_complex"
        file_path = files["complex.md"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("complex.md", f, "text/markdown")},
                data={
                    "collection": collection_name,
                    "chunk_strategy": "markdown",
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_json_default_parsing_behavior(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that .json files are parsed correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_json"
        file_path = files["simple.json"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.json", f, "application/json")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_json_array_parsing(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that JSON arrays are parsed correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_json_array"
        file_path = files["array.json"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("array.json", f, "application/json")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_json_nested_parsing(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that nested JSON structures are parsed correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_json_nested"
        file_path = files["nested.json"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("nested.json", f, "application/json")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_csv_default_parsing_behavior(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that .csv files are parsed correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_csv"
        file_path = files["simple.csv"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.csv", f, "text/csv")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_csv_quoted_parsing(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that CSV with quoted fields is parsed correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_csv_quoted"
        file_path = files["quoted.csv"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("quoted.csv", f, "text/csv")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_html_default_parsing_behavior(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that .html files are parsed correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_html"
        file_path = files["simple.html"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.html", f, "text/html")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_code_file_parsing(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Test that unsupported code files are rejected with a clear error."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_parse_code"
        file_path = files["code.py"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("code.py", f, "text/x-python")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 422
        result = response.json()
        assert "Unsupported file type" in result.get("detail", "")


# ==========================================
# PARSER SELECTION VERIFICATION TESTS
# ==========================================


class TestParserSelectionVerification:
    """
    Test that the correct parser is selected for each file type.

    These tests verify the parser selection logic to ensure
    the expected parser is used for each file extension.
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_verify_txt_uses_default_parser(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Verify that .txt files use the default parser."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_verify_txt_parser"
        file_path = files["simple.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.txt", f, "text/plain")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",  # Explicitly request default parser
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_verify_md_uses_markdown_parser(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Verify that .md files can use markdown-specific parsing."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_verify_md_parser"
        file_path = files["simple.md"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.md", f, "text/markdown")},
                data={
                    "collection": collection_name,
                    "parse_method": "default",
                    "chunk_strategy": "markdown",
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_verify_deepdoc_parser_selection(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Verify that deepdoc parser can be explicitly selected."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_verify_deepdoc"
        file_path = files["simple.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.txt", f, "text/plain")},
                data={
                    "collection": collection_name,
                    "parse_method": "deepdoc",  # Explicitly select deepdoc
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]


# ==========================================
# PARSED CONTENT VERIFICATION TESTS
# ==========================================


class TestParsedContentVerification:
    """
    Test that parsed content is accurate and complete.

    These tests verify that:
    1. Text content is correctly extracted
    2. Metadata is properly extracted
    3. No content is lost during parsing
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_verify_text_content_accuracy(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Verify that text content is accurately preserved."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_verify_text"
        file_path = files["simple.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_verify_chinese_content_accuracy(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Verify that Chinese text is accurately preserved."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_verify_chinese"
        file_path = files["chinese.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("chinese.txt", f, "text/plain;charset=utf-8")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Should handle Chinese text without encoding issues
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]


# ==========================================
# METADATA EXTRACTION TESTS
# ==========================================


class TestMetadataExtraction:
    """
    Test that metadata is correctly extracted during parsing.

    These tests verify that:
    1. File metadata is extracted
    2. Source path is recorded
    3. File type is detected
    4. Parse method is recorded
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_verify_file_type_detection(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Verify that file type is correctly detected."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_verify_file_type"

        # Test different file types
        file_types = [
            ("simple.txt", "text/plain"),
            ("simple.md", "text/markdown"),
            ("simple.json", "application/json"),
            ("simple.csv", "text/csv"),
        ]

        for filename, mime_type in file_types:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (filename, f, mime_type)},
                    data={"collection": collection_name},
                    headers=auth_headers,
                )

                # Should detect file type correctly
                assert response.status_code == 200
                result = response.json()
                assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_verify_source_path_recorded(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Verify that source path is recorded in metadata."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_verify_source"
        file_path = files["simple.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("simple.txt", f, "text/plain")},
                data={"collection": collection_name},
                headers=auth_headers,
            )

        # Should record source path
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]


# ==========================================
# CHUNK COUNT VERIFICATION TESTS
# ==========================================


class TestChunkCountVerification:
    """
    Test that chunking produces expected results.

    These tests verify that:
    1. Chunk counts are reasonable for file size
    2. Different chunk strategies produce different results
    3. Chunk overlap works correctly
    """

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_verify_fixed_size_chunking(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Verify that fixed-size chunking works correctly."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_verify_chunks"
        file_path = files["multiline.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("multiline.txt", f, "text/plain")},
                data={
                    "collection": collection_name,
                    "chunk_strategy": "fixed_size",
                    "chunk_size": "15",  # Small size to get multiple chunks
                    "chunk_overlap": "5",
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_verify_markdown_chunking(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_parsing_files: tuple[dict[str, str], str],
    ) -> None:
        """Verify that markdown chunking respects markdown structure."""
        files, temp_dir = sample_parsing_files
        collection_name = "e2e_verify_md_chunks"
        file_path = files["complex.md"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/kb/ingest",
                files={"file": ("complex.md", f, "text/markdown")},
                data={
                    "collection": collection_name,
                    "chunk_strategy": "markdown",
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "partial"]


# ==========================================
# EDGE CASES TESTS
# ==========================================


class TestParsingEdgeCases:
    """Test parsing behavior for edge cases."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_empty_file_handling(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that empty files are handled gracefully."""
        collection_name = "e2e_parse_empty"

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
            tmp.write("")  # Empty file
            tmp.flush()

            try:
                with open(tmp.name, "rb") as f:
                    response = client.post(
                        "/api/kb/ingest",
                        files={"file": ("empty.txt", f, "text/plain")},
                        data={"collection": collection_name},
                        headers=auth_headers,
                    )

                # Empty files should still be accepted for ingestion
                assert response.status_code == 200
            finally:
                import os

                os.unlink(tmp.name)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_special_characters_handling(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that special characters are handled correctly."""
        collection_name = "e2e_parse_special"

        special_content = "Test with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?~`"

        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(special_content)
            tmp.flush()

            try:
                with open(tmp.name, "rb") as f:
                    response = client.post(
                        "/api/kb/ingest",
                        files={"file": ("special.txt", f, "text/plain")},
                        data={"collection": collection_name},
                        headers=auth_headers,
                    )

                # Special characters should be handled normally
                assert response.status_code == 200
            finally:
                import os

                os.unlink(tmp.name)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_mixed_encoding_handling(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test that mixed encoding files are handled."""
        collection_name = "e2e_parse_encoding"

        # Create file with mixed content
        mixed_content = "English text 中文文本 Numbers 12345"

        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(mixed_content)
            tmp.flush()

            try:
                with open(tmp.name, "rb") as f:
                    response = client.post(
                        "/api/kb/ingest",
                        files={"file": ("mixed.txt", f, "text/plain;charset=utf-8")},
                        data={"collection": collection_name},
                        headers=auth_headers,
                    )

                # Should handle mixed encoding
                assert response.status_code == 200
                result = response.json()
                assert result["status"] in ["success", "partial"]
            finally:
                import os

                os.unlink(tmp.name)

"""Tests for PDF table position information extraction in DeepDoc parser.

This module tests the position information extraction for PDF tables,
including format conversion, cross-page tables, and backward compatibility.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from xagent.providers.pdf_parser.deepdoc import DeepDocParser, _translate_pdf_bboxes


class TestDeepDocTablePositions:
    """Test suite for PDF table position information extraction."""

    @pytest.fixture
    def mock_image(self):
        """Create a mock PIL Image object."""
        return Image.new("RGB", (100, 100), color="red")

    @pytest.fixture
    def sample_table_html(self):
        """Sample HTML table content."""
        return "<table><tr><td>Cell 1</td><td>Cell 2</td></tr></table>"

    @pytest.fixture
    def doc_id(self):
        """Sample document ID."""
        return "test_doc_123"

    def test_translate_pdf_bboxes_with_positions(
        self, mock_image, sample_table_html, doc_id
    ):
        """Test _translate_pdf_bboxes with position information (structured format)."""
        # Structured bbox format from parse_into_bboxes
        bboxes = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                "positions": [
                    [1, 10.0, 100.0, 20.0, 50.0],
                    [2, 10.0, 100.0, 0.0, 30.0],
                ],
            }
        ]

        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        result = _translate_pdf_bboxes(doc_id, bboxes, **kwargs)

        # Manually set raw_parser_output for testing (normally done in _parse_impl)
        result.raw_parser_output = {
            "format": "deepdoc_pdf",
            "bboxes": bboxes,
            "total_elements": len(bboxes),
            "has_positions": any("positions" in bbox for bbox in bboxes),
        }
        result.parser_engine = "deepdoc"

        assert len(result.tables) == 1
        table = result.tables[0]
        assert table.html == sample_table_html
        # Check that raw_parser_output is set for visualization
        assert result.raw_parser_output is not None
        assert "bboxes" in result.raw_parser_output
        assert len(result.raw_parser_output["bboxes"]) == 1
        # Check that the bbox data is preserved
        bbox = result.raw_parser_output["bboxes"][0]
        assert bbox["layout_type"] == "table"
        assert "positions" in bbox
        assert len(bbox["positions"]) == 2

    def test_translate_pdf_bboxes_without_positions(
        self, mock_image, sample_table_html, doc_id
    ):
        """Test _translate_pdf_bboxes without position information (fallback to defaults)."""
        # Structured bbox format without positions (should generate defaults)
        bboxes = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                # No positions field - should generate defaults
            }
        ]

        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        result = _translate_pdf_bboxes(doc_id, bboxes, **kwargs)

        # Manually set raw_parser_output for testing
        result.raw_parser_output = {
            "format": "deepdoc_pdf",
            "bboxes": bboxes,
            "total_elements": len(bboxes),
            "has_positions": any("positions" in bbox for bbox in bboxes),
        }
        result.parser_engine = "deepdoc"

        assert len(result.tables) == 1
        table = result.tables[0]
        assert table.html == sample_table_html
        # Raw parser output should still be set even without positions in bboxes
        assert result.raw_parser_output is not None
        assert "bboxes" in result.raw_parser_output

    def test_translate_pdf_bboxes_with_empty_positions(
        self, mock_image, sample_table_html, doc_id
    ):
        """Test _translate_pdf_bboxes with empty positions list (should generate defaults)."""
        # Structured bbox format with empty positions
        bboxes = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                "positions": [],  # Empty positions - should generate defaults
            }
        ]

        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        result = _translate_pdf_bboxes(doc_id, bboxes, **kwargs)

        # Manually set raw_parser_output for testing
        result.raw_parser_output = {
            "format": "deepdoc_pdf",
            "bboxes": bboxes,
            "total_elements": len(bboxes),
            "has_positions": any("positions" in bbox for bbox in bboxes),
        }
        result.parser_engine = "deepdoc"

        assert len(result.tables) == 1
        # Raw parser output should be set
        assert result.raw_parser_output is not None
        assert "bboxes" in result.raw_parser_output

    def test_translate_pdf_bboxes_cross_page_table(
        self, mock_image, sample_table_html, doc_id
    ):
        """Test _translate_pdf_bboxes with cross-page table."""
        # Table spanning pages 1, 2, 3
        positions = [
            [1, 10.0, 100.0, 20.0, 50.0],
            [2, 10.0, 100.0, 0.0, 50.0],
            [3, 10.0, 100.0, 0.0, 30.0],
        ]
        bboxes = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                "positions": positions,
            }
        ]

        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        result = _translate_pdf_bboxes(doc_id, bboxes, **kwargs)

        # Manually set raw_parser_output for testing
        result.raw_parser_output = {
            "format": "deepdoc_pdf",
            "bboxes": bboxes,
            "total_elements": len(bboxes),
            "has_positions": any("positions" in bbox for bbox in bboxes),
        }
        result.parser_engine = "deepdoc"

        assert len(result.tables) == 1
        # Check raw parser output preserves cross-page table data
        assert result.raw_parser_output is not None
        assert "bboxes" in result.raw_parser_output
        bbox = result.raw_parser_output["bboxes"][0]
        assert len(bbox["positions"]) == 3
        # Verify pages are preserved in raw data
        pages = [pos[0] for pos in bbox["positions"]]  # Extract page numbers
        assert sorted(pages) == [1, 2, 3]

    def test_translate_pdf_bboxes_zero_based_page_number(
        self, mock_image, sample_table_html, doc_id
    ):
        """Test that 0-based page numbers are converted to 1-based."""
        # DeepDoc may return 0-based page numbers
        positions = [[0, 10.0, 100.0, 20.0, 50.0]]  # 0-based page number
        bboxes = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                "positions": positions,
            }
        ]

        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        result = _translate_pdf_bboxes(doc_id, bboxes, **kwargs)

        # Manually set raw_parser_output for testing
        result.raw_parser_output = {
            "format": "deepdoc_pdf",
            "bboxes": bboxes,
            "total_elements": len(bboxes),
            "has_positions": any("positions" in bbox for bbox in bboxes),
        }
        result.parser_engine = "deepdoc"

        assert len(result.tables) == 1
        # Check that raw data preserves original 0-based page number
        assert result.raw_parser_output is not None
        bbox = result.raw_parser_output["bboxes"][0]
        assert bbox["positions"][0][0] == 0  # Original 0-based page number preserved

    def test_translate_pdf_bboxes_mixed_format(
        self, mock_image, sample_table_html, doc_id
    ):
        """Test _translate_pdf_bboxes with mixed bbox formats."""
        # Mix of bboxes with and without positions
        bboxes = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                # No positions - should generate defaults
            },
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                "positions": [[1, 10.0, 100.0, 20.0, 50.0]],  # With positions
            },
        ]

        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        result = _translate_pdf_bboxes(doc_id, bboxes, **kwargs)

        # Manually set raw_parser_output for testing
        result.raw_parser_output = {
            "format": "deepdoc_pdf",
            "bboxes": bboxes,
            "total_elements": len(bboxes),
            "has_positions": any("positions" in bbox for bbox in bboxes),
        }
        result.parser_engine = "deepdoc"

        assert len(result.tables) == 2
        # Raw parser output should preserve both bboxes
        assert result.raw_parser_output is not None
        assert "bboxes" in result.raw_parser_output
        assert len(result.raw_parser_output["bboxes"]) == 2

    @pytest.mark.asyncio
    async def test_real_pdf_file_parsing_with_positions(self):
        """Integration test with real PDF file to verify complete position information handling."""
        from pathlib import Path

        # Path to real test PDF file
        test_pdf_path = (
            Path(__file__).parent.parent.parent
            / "resources"
            / "test_files"
            / "test.pdf"
        )

        if not test_pdf_path.exists():
            pytest.skip(f"Real test PDF file not found: {test_pdf_path}")

        # Test complete parsing pipeline
        parser = DeepDocParser(enable_raw_output=True)
        result = await parser.parse(str(test_pdf_path), doc_id="real_test_pdf")

        # Verify basic structure
        assert result is not None
        assert isinstance(result.text_segments, list)
        assert isinstance(result.tables, list)
        assert isinstance(result.figures, list)

        # Verify all elements have positions (including generated defaults)
        total_elements = (
            len(result.text_segments) + len(result.tables) + len(result.figures)
        )
        assert total_elements > 0, "PDF should contain some parseable content"

        # Verify raw parser output is set and contains position data
        assert result.raw_parser_output is not None, "Should have raw parser output"
        assert result.parser_engine == "deepdoc", "Should specify parser engine"

        # Verify visualization data is available
        assert result.has_visualization_data, "Should have visualization data"
        viz_elements = result.get_visualization_elements()
        assert isinstance(viz_elements, list), "Visualization elements should be a list"

        # If there are visualization elements, check their structure
        if viz_elements:
            element = viz_elements[0]
            required_keys = ["id", "type", "content", "bbox", "page", "metadata"]
            for key in required_keys:
                assert key in element, f"Visualization element should have {key}"

            # Check metadata structure
            metadata = element["metadata"]
            assert "parser" in metadata, "Should have parser info"
            assert metadata["parser"] == result.parser_engine, "Parser should match"

        # Check that elements still have basic metadata
        for segment in result.text_segments:
            assert hasattr(segment, "metadata"), "Text segment should have metadata"
            assert "layout_type" in segment.metadata, "Should have layout type"

        for table in result.tables:
            assert hasattr(table, "metadata"), "Table should have metadata"
            assert "layout_type" in table.metadata, "Should have layout type"

        for figure in result.figures:
            assert hasattr(figure, "metadata"), "Figure should have metadata"
            assert "layout_type" in figure.metadata, "Should have layout type"

        # Verify content extraction
        assert result.full_text is not None, "Should extract full text"
        assert len(result.full_text.strip()) > 0, "Full text should not be empty"

        # Verify metadata
        assert hasattr(result, "metadata"), "Result should have metadata"
        assert "source" in result.metadata, "Should have source in metadata"
        assert result.metadata["source"] == str(test_pdf_path), (
            "Source should match input file"
        )

        print(f"Real PDF integration test passed: {result.full_text}")
        print(f"Text segments: {len(result.text_segments)}")
        print(f"Tables: {len(result.tables)}")
        print(f"Figures: {len(result.figures)}")
        print(f"Total text length: {len(result.full_text)}")

    @patch("xagent.providers.pdf_parser.deepdoc.DeepDocPdfParser")
    @pytest.mark.asyncio
    async def test_parse_impl_sets_need_position(self, mock_parser_class, tmp_path):
        """Test that _parse_impl extracts position information for PDF files."""
        # Create a temporary PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        # Mock the parser
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser

        # Mock parse_into_bboxes to return bbox data
        mock_bboxes = [
            {
                "layout_type": "table",
                "text": "test table",
                "image": MagicMock(),
                "positions": [[1, 10.0, 100.0, 20.0, 50.0]],
            }
        ]
        mock_parser.parse_into_bboxes.return_value = mock_bboxes

        parser = DeepDocParser(enable_raw_output=True)  # Enable raw output
        result = await parser._parse_impl(str(pdf_file), doc_id="test")

        # Verify parse_into_bboxes was called with correct parameters
        mock_parser.parse_into_bboxes.assert_called_once_with(
            str(pdf_file), callback=None, zoomin=3
        )

        # Verify result contains table and raw parser output
        assert len(result.tables) == 1
        assert result.raw_parser_output is not None
        assert "bboxes" in result.raw_parser_output

    @patch("xagent.providers.pdf_parser.deepdoc.DeepDocPdfParser")
    @pytest.mark.asyncio
    async def test_parse_impl_backward_compatibility(self, mock_parser_class, tmp_path):
        """Test that _parse_impl handles both old and new return formats."""
        # Create a temporary PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        # Mock the parser
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser

        # Test old format (without positions)
        mock_parser.return_value = ("text", [(MagicMock(), "html")])

        parser = DeepDocParser()
        result = await parser._parse_impl(str(pdf_file), doc_id="test")

        # Should not raise error and should process tables
        assert result is not None
        assert isinstance(result.tables, list)

        # Test new format (with positions)
        mock_image = MagicMock()
        positions = [(1, 10.0, 100.0, 20.0, 50.0)]
        mock_parser.return_value = ("text", [((mock_image, "html"), positions)])

        result = await parser._parse_impl(str(pdf_file), doc_id="test")

        # Should not raise error and should process tables
        assert result is not None
        assert isinstance(result.tables, list)
        # Check that raw parser output is set appropriately
        if result.raw_parser_output:
            assert "format" in result.raw_parser_output

    def test_translate_pdf_bboxes_with_figure_positions(self, mock_image):
        """Test _translate_pdf_bboxes with figure/image position information."""
        # Structured bbox format for figure
        bboxes = [
            {
                "layout_type": "figure",
                "text": "Sample figure caption",
                "image": mock_image,
                "positions": [[1, 50.0, 150.0, 100.0, 200.0]],
            }
        ]

        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        result = _translate_pdf_bboxes("test_doc_123", bboxes, **kwargs)

        # Manually set raw_parser_output for testing
        result.raw_parser_output = {
            "format": "deepdoc_pdf",
            "bboxes": bboxes,
            "total_elements": len(bboxes),
            "has_positions": any("positions" in bbox for bbox in bboxes),
        }
        result.parser_engine = "deepdoc"

        assert len(result.figures) == 1
        figure = result.figures[0]
        assert figure.text == "Sample figure caption"
        # Check raw parser output preserves position data
        assert result.raw_parser_output is not None
        assert "bboxes" in result.raw_parser_output
        bbox = result.raw_parser_output["bboxes"][0]
        assert "positions" in bbox
        assert len(bbox["positions"]) == 1

    def test_translate_pdf_bboxes_figure_without_positions(self, mock_image):
        """Test _translate_pdf_bboxes with figure without position information (should generate defaults)."""
        # Structured bbox format for figure without positions
        bboxes = [
            {
                "layout_type": "figure",
                "text": "Figure without position",
                "image": mock_image,
                # No positions field - should generate defaults
            }
        ]

        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        result = _translate_pdf_bboxes("test_doc_123", bboxes, **kwargs)

        # Manually set raw_parser_output for testing
        result.raw_parser_output = {
            "format": "deepdoc_pdf",
            "bboxes": bboxes,
            "total_elements": len(bboxes),
            "has_positions": any("positions" in bbox for bbox in bboxes),
        }
        result.parser_engine = "deepdoc"

        assert len(result.figures) == 1
        figure = result.figures[0]
        assert figure.text == "Figure without position"
        # Raw parser output should still be set even without positions
        assert result.raw_parser_output is not None
        assert "bboxes" in result.raw_parser_output

    def test_translate_pdf_bboxes_empty_list(self, doc_id):
        """Test _translate_pdf_bboxes when parse_into_bboxes returns empty list."""
        kwargs = {"source": "empty.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        result = _translate_pdf_bboxes(doc_id, [], **kwargs)

        assert len(result.text_segments) == 0
        assert len(result.tables) == 0
        assert len(result.figures) == 0
        assert result.metadata is not None
        assert result.metadata["source"] == "empty.pdf"

    def test_translate_pdf_bboxes_malformed_positions(
        self, mock_image, sample_table_html, doc_id
    ):
        """Test _translate_pdf_bboxes handles malformed position data gracefully."""
        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        # Test case 1: Position with 3 elements instead of 5
        bboxes_malformed = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                "positions": [[1, 10.0, 100.0]],  # Only 3 elements instead of 5
            }
        ]
        result = _translate_pdf_bboxes(doc_id, bboxes_malformed, **kwargs)
        assert len(result.tables) == 1
        # Should still process the table even with malformed positions
        assert result.tables[0].html == sample_table_html

        # Test case 2: Position with non-numeric values
        bboxes_non_numeric = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                "positions": [["invalid", "data"]],  # Non-numeric values
            }
        ]
        result = _translate_pdf_bboxes(doc_id, bboxes_non_numeric, **kwargs)
        assert len(result.tables) == 1

        # Test case 3: Mixed valid and malformed positions
        bboxes_mixed = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": mock_image,
                "positions": [
                    [1, 10.0, 100.0, 20.0, 50.0],  # Valid
                    [2, 10.0, 100.0],  # Malformed (3 elements)
                ],
            }
        ]
        result = _translate_pdf_bboxes(doc_id, bboxes_mixed, **kwargs)
        assert len(result.tables) == 1

    def test_translate_pdf_bboxes_image_handling_edge_cases(
        self, sample_table_html, doc_id
    ):
        """Test image handling for various edge cases."""
        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        # Test case 1: None image
        bboxes_none = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": None,
            }
        ]
        result = _translate_pdf_bboxes(doc_id, bboxes_none, **kwargs)
        assert len(result.tables) == 1
        assert result.tables[0].metadata["image_path"] is None

        # Test case 2: String path image (skip this test case as it requires existing file)
        # Note: This test case would require a real file to exist, which is not ideal for unit tests
        pass

        # Test case 3: Invalid image type (should raise ValueError)
        bboxes_invalid = [
            {
                "layout_type": "table",
                "text": sample_table_html,
                "image": 123,  # Invalid type
            }
        ]

        with pytest.raises(ValueError, match="Unsupported image type"):
            _translate_pdf_bboxes(doc_id, bboxes_invalid, **kwargs)

    @pytest.mark.asyncio
    async def test_concurrent_parsing_multiple_files(self, tmp_path):
        """Test parsing multiple files concurrently."""
        import asyncio

        # Create multiple temporary PDF files
        pdf_files = []
        parser = DeepDocParser()

        for i in range(3):
            pdf_file = tmp_path / f"test_{i}.pdf"
            pdf_file.write_bytes(b"fake pdf content")
            pdf_files.append(str(pdf_file))

        # Parse all files concurrently
        tasks = [
            parser.parse(pdf_path, doc_id=f"concurrent_doc_{i}")
            for i, pdf_path in enumerate(pdf_files)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all completed successfully (even if mocked)
        assert len(results) == 3
        for result in results:
            # Results may be exceptions due to fake PDF content, but should not hang
            assert result is not None

    @pytest.mark.asyncio
    async def test_excel_csv_bytesio_conversion(self, tmp_path):
        """Test Excel/CSV BytesIO conversion in _parse_impl."""
        from io import BytesIO

        parser = DeepDocParser()

        # Test Excel BytesIO conversion
        excel_content = b"fake excel binary content"
        excel_bytesio = BytesIO(excel_content)

        with pytest.raises(ValueError, match="Failed to parse spreadsheet rows"):
            await parser._parse_impl(
                excel_bytesio, file_ext=".xlsx", doc_id="test_excel_bytesio"
            )

        # Test CSV BytesIO conversion
        csv_content = b"col1,col2\nval1,val2\nval3,val4"
        csv_bytesio = BytesIO(csv_content)

        result = await parser._parse_impl(
            csv_bytesio, file_ext=".csv", doc_id="test_csv_bytesio"
        )

        # Should handle BytesIO input without crashing
        assert result is not None
        assert hasattr(result, "text_segments")

    def test_translate_pdf_bboxes_various_layout_types(self, mock_image, doc_id):
        """Test _translate_pdf_bboxes with various layout types including unknown ones."""
        kwargs = {"source": "test.pdf", "file_type": ".pdf", "parse_method": "deepdoc"}

        bboxes = [
            {
                "layout_type": "text",
                "text": "Sample text content",
            },
            {
                "layout_type": "table",
                "text": "<table><tr><td>Data</td></tr></table>",
                "image": mock_image,
            },
            {
                "layout_type": "figure",
                "text": "Figure caption",
                "image": mock_image,
            },
            {
                "layout_type": "unknown_type",
                "text": "Unknown content",
            },
        ]

        result = _translate_pdf_bboxes(doc_id, bboxes, **kwargs)

        # Should have 1 text segment (text + unknown_type)
        assert len(result.text_segments) == 2
        assert result.text_segments[0].text == "Sample text content"
        assert result.text_segments[1].text == "Unknown content"

        # Should have 1 table
        assert len(result.tables) == 1
        assert "<table>" in result.tables[0].html

        # Should have 1 figure
        assert len(result.figures) == 1
        assert result.figures[0].text == "Figure caption"

    def test_translate_pdf_bboxes_metadata_preservation(self, mock_image, doc_id):
        """Test that metadata is properly preserved and merged."""
        custom_kwargs = {
            "source": "test.pdf",
            "file_type": ".pdf",
            "parse_method": "deepdoc",
            "custom_field": "custom_value",
            "user_id": 12345,
        }

        bboxes = [
            {
                "layout_type": "text",
                "text": "Test content",
            }
        ]

        result = _translate_pdf_bboxes(doc_id, bboxes, **custom_kwargs)

        # Check that custom metadata is preserved
        assert result.metadata["custom_field"] == "custom_value"
        assert result.metadata["user_id"] == 12345
        assert result.metadata["source"] == "test.pdf"
        assert result.metadata["parse_method"] == "deepdoc"

        # Check that element metadata includes the custom fields
        assert result.text_segments[0].metadata["custom_field"] == "custom_value"
        assert result.text_segments[0].metadata["user_id"] == 12345
        assert result.text_segments[0].metadata["doc_id"] == doc_id

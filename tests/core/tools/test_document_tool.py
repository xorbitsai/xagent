from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.tools.adapters.vibe.document_parser import (
    DocumentParseTool,
    DocumentParseWithOutputTool,
)
from xagent.core.tools.core.document_parser import (
    DocumentCapabilities,
    DocumentParseArgs,
    DocumentParseWithOutputArgs,
    DocumentParseWithOutputResult,
    document_parser_registry,
    filter_parsers_by_capabilities,
)
from xagent.providers.pdf_parser.base import (
    DocumentParser,
    FigureParsing,
    FullTextResult,
    LocalParsing,
    ParsedFigures,
    ParsedTable,
    ParsedTextSegment,
    ParseResult,
    RemoteParsing,
    SegmentedTextResult,
    TextParsing,
)


class MockWorkspace:
    """Mock TaskWorkspace for resolving paths."""

    def resolve_path_with_search(self, path: str):
        if path == "input.pdf":
            return "/mock/workspace/input.pdf"
        raise FileNotFoundError(f"Path not found: {path}")

    def resolve_path(self, path: str):
        if path == "output.txt":
            return "/mock/workspace/output.txt"
        return path


# Create mock parser class instances with specific capabilities
def create_mock_parser(name, capabilities):
    mock_parser_class = MagicMock(spec=DocumentParser)
    mock_parser_class.__name__ = name
    mock_parser_class.get_capabilities.return_value = set(capabilities)

    mock_instance = AsyncMock(spec=DocumentParser)
    mock_parser_class.return_value = mock_instance
    return mock_parser_class, mock_instance


# Global registry
REGISTRY_MOCK_CONTENT = {
    "pypdf": create_mock_parser(
        "PyPdfParser", [TextParsing, SegmentedTextResult, LocalParsing]
    )[0],
    "deepdoc": create_mock_parser(
        "DeepDocParser",
        [TextParsing, FigureParsing, SegmentedTextResult, LocalParsing],
    )[0],
    "mineru_basic": create_mock_parser(
        "MinerUBasicDocumentParser", [TextParsing, FullTextResult, RemoteParsing]
    )[0],
    "mineru_enhanced": create_mock_parser(
        "MinerUEnhancedDocumentParser", [TextParsing, FigureParsing, RemoteParsing]
    )[0],
    "full_local_combo": create_mock_parser(
        "FullLocalCombo",
        [TextParsing, FigureParsing, FullTextResult, SegmentedTextResult, LocalParsing],
    )[0],
}


@pytest.fixture
def mock_parser_registry():
    """Fixture to mock the registry and its parsers."""
    # Create fresh instances for each test
    registry_instances = {}
    for name in REGISTRY_MOCK_CONTENT:
        parser_class, mock_instance = create_mock_parser(
            name, REGISTRY_MOCK_CONTENT[name].get_capabilities.return_value
        )
        registry_instances[name] = mock_instance

    with (
        patch.object(
            document_parser_registry, "parsers", return_value=REGISTRY_MOCK_CONTENT
        ) as mock_parsers,
        patch.object(
            document_parser_registry,
            "get_parser",
            side_effect=lambda name: registry_instances[name],
        ) as mock_get,
    ):
        # Reset all mocks before yielding
        for instance in registry_instances.values():
            instance.reset_mock()

        yield {
            "parsers": mock_parsers,
            "get_parser": mock_get,
            "instances": registry_instances,
            "parser_classes": REGISTRY_MOCK_CONTENT,
        }

        # Reset all mocks after test
        for instance in registry_instances.values():
            instance.reset_mock()


@pytest.fixture
def mock_workspace():
    return MockWorkspace()


@pytest.fixture
def parse_result_full():
    """A ParseResult object containing all fields."""
    return ParseResult(
        full_text="Complete document text.",
        text_segments=[
            ParsedTextSegment(text="Segment 1 text.", metadata={"page": 1}),
            ParsedTextSegment(text="Segment 2 text.", metadata={"page": 2}),
        ],
        figures=[
            ParsedFigures(text="Image Description.", metadata={"type": "Figure"}),
        ],
        tables=[
            ParsedTable(
                html="<table><tr><td>Table Content</td></tr></table>",
                metadata={"type": "Table"},
            ),
        ],
        metadata={"doc_title": "Test Doc", "parser_version": "v1.0"},
    )


def test_document_parse_tool_metadata():
    tool = DocumentParseTool()
    assert tool.name == "document_parse"
    assert tool.args_type() == DocumentParseArgs
    assert tool.return_type() == ParseResult


def test_document_parse_with_output_tool_metadata(mock_workspace):
    tool = DocumentParseWithOutputTool(mock_workspace)
    assert tool.name == "document_parse"
    assert tool.args_type() == DocumentParseWithOutputArgs
    assert tool.return_type() == DocumentParseWithOutputResult


def test_full_text_local_with_figure(mock_parser_registry):
    caps = DocumentCapabilities(
        capability_text=True,
        capability_figure=True,
        requires_full_text_result=True,
        requires_segmented_result=False,
        use_local_parser=True,
    )
    # Expected matches: "full_local_combo"
    result = filter_parsers_by_capabilities(
        mock_parser_registry["parser_classes"], caps
    )
    assert result == ["full_local_combo"]


def test_segmented_local_text_only(mock_parser_registry):
    caps = DocumentCapabilities(
        capability_text=True,
        capability_figure=False,
        requires_full_text_result=False,
        requires_segmented_result=True,
        use_local_parser=True,
    )
    # Expected matches: "pypdf", "deepdoc", "full_local_combo"
    result = filter_parsers_by_capabilities(
        mock_parser_registry["parser_classes"], caps
    )
    assert sorted(result) == sorted(["pypdf", "deepdoc", "full_local_combo"])


def test_full_text_remote_text_only(mock_parser_registry):
    caps = DocumentCapabilities(
        capability_text=True,
        capability_figure=False,
        requires_full_text_result=True,
        requires_segmented_result=False,
        use_local_parser=False,
    )
    # Expected matches: "mineru_basic"
    result = filter_parsers_by_capabilities(
        mock_parser_registry["parser_classes"], caps
    )
    assert result == ["mineru_basic"]


def test_no_parsers_match(mock_parser_registry):
    caps = DocumentCapabilities(
        capability_text=True,
        capability_figure=False,
        requires_full_text_result=True,
        requires_segmented_result=True,
        use_local_parser=True,
    )
    result = filter_parsers_by_capabilities(
        mock_parser_registry["parser_classes"], caps
    )
    assert result == ["full_local_combo"]

    caps_fail = DocumentCapabilities(
        capability_text=True,
        capability_figure=True,
        requires_full_text_result=True,
        requires_segmented_result=False,
        use_local_parser=False,
    )
    result_fail = filter_parsers_by_capabilities(
        mock_parser_registry["parser_classes"], caps_fail
    )
    assert result_fail == []


@pytest.mark.asyncio
async def test_document_parse_tool_successful_run(
    mock_parser_registry, mock_workspace, parse_result_full
):
    parser_name = "full_local_combo"
    mock_parser_registry["instances"][
        parser_name
    ].parse.return_value = parse_result_full

    tool = DocumentParseTool(workspace=mock_workspace)
    args = {
        "file_path": "input.pdf",
        "parser_name": parser_name,
        "capabilities": {
            "capability_text": True,
            "capability_figure": True,
            "requires_full_text_result": True,
            "requires_segmented_result": True,
            "use_local_parser": True,
        },
    }

    result = await tool.run_json_async(args)

    assert result == parse_result_full
    mock_parser_registry["instances"][parser_name].parse.assert_called_once_with(
        "/mock/workspace/input.pdf", progress_callback=None
    )


@pytest.mark.asyncio
async def test_document_parse_tool_default_parser_selection(mock_workspace):
    """Test that document parse tool correctly uses deepdoc parser for PDF files."""
    # Use a real PDF file for testing
    from pathlib import Path

    # Use fixed path for test PDF file
    test_pdf_path = (
        Path(__file__).parent.parent.parent / "resources" / "test_files" / "test.pdf"
    )

    if not test_pdf_path.exists():
        pytest.skip(f"Test PDF not found: {test_pdf_path}")

    tool = DocumentParseTool(workspace=mock_workspace)
    args = {
        "file_path": str(test_pdf_path),
        "parser_name": "",  # Empty parser_name triggers auto-selection for PDF -> deepdoc
        "capabilities": {
            "capability_text": True,
            "capability_figure": True,  # Allow figure parsing for deepdoc
            "requires_full_text_result": False,
            "requires_segmented_result": True,
            "use_local_parser": True,
        },
    }

    result = await tool.run_json_async(args)

    # Verify we got a valid ParseResult
    assert result is not None
    from xagent.core.tools.core.document_parser import ParseResult

    assert isinstance(result, ParseResult)
    assert hasattr(result, "text_segments")
    assert hasattr(result, "tables")
    assert hasattr(result, "figures")
    assert hasattr(result, "full_text")

    # Verify content was extracted
    assert len(result.text_segments) > 0
    assert len(result.full_text) > 0

    # For PDF files, deepdoc should be able to extract figures
    # (This depends on the actual PDF content)


@pytest.mark.asyncio
async def test_document_parse_tool_error_no_compatible_parsers(
    mock_parser_registry, mock_workspace
):
    tool = DocumentParseTool(workspace=mock_workspace)
    args = {
        "file_path": "input.pdf",
        "parser_name": "",
        "capabilities": {
            "capability_text": True,
            "capability_figure": True,
            "requires_full_text_result": True,
            "requires_segmented_result": False,
            "use_local_parser": False,  # remote
        },
    }
    with pytest.raises(ValueError, match="No parsers found matching requirements"):
        await tool.run_json_async(args)


@pytest.mark.asyncio
async def test_document_parse_tool_error_parser_failure(
    mock_parser_registry, mock_workspace
):
    parser_name = "pypdf"
    mock_parser_registry["instances"][parser_name].parse.side_effect = Exception(
        "Parsing failed internally"
    )

    tool = DocumentParseTool(workspace=mock_workspace)
    args = {
        "file_path": "input.pdf",
        "parser_name": parser_name,
        "capabilities": {
            "capability_text": True,
            "capability_figure": False,
            "requires_full_text_result": False,
            "requires_segmented_result": True,
            "use_local_parser": True,
        },
    }

    with pytest.raises(
        RuntimeError, match=f"Document parsing failed with {parser_name}"
    ):
        await tool.run_json_async(args)


@pytest.mark.asyncio
async def test_document_parse_tool_error_incompatible_requested_parser(
    mock_parser_registry, mock_workspace
):
    # Requesting pypdf (Segmented/Local) but demanding FullTextResult
    tool = DocumentParseTool(workspace=mock_workspace)
    args = {
        "file_path": "input.pdf",
        "parser_name": "pypdf",
        "capabilities": {
            "capability_text": True,
            "capability_figure": False,
            "requires_full_text_result": True,  # Incompatible with pypdf
            "requires_segmented_result": False,
            "use_local_parser": True,
        },
    }

    with pytest.raises(
        ValueError, match="Requested parser 'pypdf' doesn't meet requirements"
    ):
        await tool.run_json_async(args)


@pytest.mark.asyncio
@patch("builtins.open")
async def test_document_parse_with_output_tool_successful_execution(
    mock_open, mock_parser_registry, mock_workspace, parse_result_full
):
    parser_name = "full_local_combo"
    mock_parser_registry["instances"][
        parser_name
    ].parse.return_value = parse_result_full

    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    tool = DocumentParseWithOutputTool(workspace=mock_workspace)
    args = {
        "file_path": "input.pdf",
        "parser_name": parser_name,
        "output_path": "output.txt",
        "output_format": "txt",  # Explicitly specify txt format for this test
        "capabilities": {
            "capability_text": True,
            "capability_figure": False,
            "requires_full_text_result": False,
            "requires_segmented_result": True,
            "use_local_parser": True,
        },
    }

    result = await tool.run_json_async(args)

    assert result == DocumentParseWithOutputResult()
    mock_parser_registry["instances"][parser_name].parse.assert_called_once_with(
        "/mock/workspace/input.pdf", progress_callback=None
    )
    mock_open.assert_called_once_with(
        "/mock/workspace/output.txt", "w", encoding="utf-8"
    )
    mock_file.write.assert_any_call("=== FULL TEXT ===\n")
    mock_file.write.assert_any_call("Complete document text.")
    mock_file.write.assert_any_call("\n\n")
    mock_file.write.assert_any_call("=== TEXT SEGMENTS ===\n")
    mock_file.write.assert_any_call("--- Segment 1 ---\n")
    mock_file.write.assert_any_call("Text: Segment 1 text.\n")
    mock_file.write.assert_any_call("Metadata: {'page': 1}\n")
    mock_file.write.assert_any_call("=== FIGURES ===\n")
    mock_file.write.assert_any_call("--- Figure 1 ---\n")
    mock_file.write.assert_any_call("Text: Image Description.\n")
    mock_file.write.assert_any_call("Metadata: {'type': 'Figure'}\n")
    mock_file.write.assert_any_call("=== TABLES ===\n")
    mock_file.write.assert_any_call("--- Table 1 ---\n")
    mock_file.write.assert_any_call(
        "HTML: <table><tr><td>Table Content</td></tr></table>\n"
    )
    mock_file.write.assert_any_call("Metadata: {'type': 'Table'}\n")
    mock_file.write.assert_any_call("=== METADATA ===\n")
    mock_file.write.assert_any_call("doc_title: Test Doc\n")
    mock_file.write.assert_any_call("parser_version: v1.0\n")


@pytest.mark.asyncio
@patch("builtins.open")
async def test_document_parse_with_output_tool_error_on_file_writing(
    mock_open, mock_parser_registry, mock_workspace, parse_result_full
):
    parser_name = "full_local_combo"
    mock_parser_registry["instances"][
        parser_name
    ].parse.return_value = parse_result_full

    mock_open.side_effect = PermissionError("Cannot access file")

    tool = DocumentParseWithOutputTool(workspace=mock_workspace)

    args = {
        "file_path": "input.pdf",
        "parser_name": parser_name,
        "output_path": "output.txt",
        "capabilities": {
            "capability_text": True,
            "capability_figure": False,
            "requires_full_text_result": False,
            "requires_segmented_result": True,
            "use_local_parser": True,
        },
    }

    with pytest.raises(
        RuntimeError, match="Failed to write output to /mock/workspace/output.txt"
    ):
        await tool.run_json_async(args)

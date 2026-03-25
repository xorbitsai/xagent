import logging
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from ....providers.pdf_parser import (
    DeepDocParser,
    DocumentParser,
    ParseResult,
    PdfPlumberParser,
    PyMuPdfParser,
    PyPdfParser,
    UnstructuredParser,
)

logger = logging.getLogger(__name__)


class DocumentCapabilities(BaseModel):
    capability_text: bool = Field(
        True,
        description="Whether the parser should extract text content from the document",
    )
    capability_figure: bool = Field(
        False,
        description="Whether the parser should extract figures, images, and tables from the document",
    )
    requires_full_text_result: bool = Field(
        True,
        description="Whether the parsing result should be returned as a single concatenated text string",
    )
    requires_segmented_result: bool = Field(
        False,
        description="Whether the parsing result should be returned as separate text segments (e.g., per page or logical blocks)",
    )
    use_local_parser: bool = Field(
        True,
        description="Whether to use a local parsing implementation (True) or remote API-based parsing (False)",
    )


class DocumentParseArgs(BaseModel):
    file_path: str = Field(
        description="Path to the document file to be parsed. Can be absolute path or relative to workspace"
    )
    parser_name: str | None = Field(
        None,
        description="Name of the specific parser to use (e.g., 'pypdf', 'deepdoc'). If not specified, the first compatible parser will be used",
    )
    capabilities: DocumentCapabilities = Field(
        description="Configuration object specifying the parsing capabilities and output format requirements"
    )
    parser_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments forwarded to the underlying parser implementation",
    )


class DocumentParseWithOutputArgs(DocumentParseArgs):
    output_path: str = Field(
        description="Path to the parsed text written. Can be absolute path or relative to workspace"
    )
    output_format: Literal["txt", "md", "json"] = Field(
        default="md",
        description="Output format for the parsed result. 'txt' for plain text, 'md' for markdown, 'json' for structured JSON",
    )


class DocumentParseWithOutputResult(BaseModel):
    pass


async def parse_document(
    tool_args: DocumentParseArgs,
    progress_callback: Any = None,
) -> ParseResult:
    """Parse a document using the specified parser and capabilities.

    Args:
        tool_args: Configuration object containing file path, parser name (optional),
                   and capabilities requirements.
        progress_callback: Optional callback for progress updates during parsing.

    Returns:
        ParseResult containing extracted text, segments, figures, tables, and metadata.

    Raises:
        ValueError: If no parsers match the required capabilities or if the requested
                   parser doesn't meet requirements.
        RuntimeError: If document parsing fails.
    """
    resolved_file_path = tool_args.file_path

    # Filter parsers based on capabilities
    parsers = document_parser_registry.parsers()
    available_parsers = filter_parsers_by_capabilities(parsers, tool_args.capabilities)

    if not available_parsers:
        raise ValueError(
            f"No parsers found matching requirements: "
            f"text={tool_args.capabilities.capability_text}, "
            f"figure={tool_args.capabilities.capability_figure}, "
            f"full_text={tool_args.capabilities.requires_full_text_result}, "
            f"segmented={tool_args.capabilities.requires_segmented_result}, "
            f"local={tool_args.capabilities.use_local_parser}"
        )

    # If a specific parser is requested, validate it's in the available list
    if tool_args.parser_name:
        if tool_args.parser_name not in available_parsers:
            raise ValueError(
                f"Requested parser '{tool_args.parser_name}' doesn't meet requirements. Available parsers: {', '.join(available_parsers)}"
            )
        selected_parser = tool_args.parser_name
    else:
        # Auto-route based on file extension
        ext = Path(resolved_file_path).suffix.lower()

        if ext == ".pdf" and "deepdoc" in available_parsers:
            selected_parser = "deepdoc"
            logger.info(
                f"Auto-selected 'deepdoc' parser for .pdf file: {resolved_file_path}"
            )
        elif (
            ext
            in (
                ".ppt",
                ".pptx",
                ".doc",
                ".docx",
                ".xlsx",
                ".xls",
                ".txt",
                ".md",
                ".json",
            )
            and "unstructured" in available_parsers
        ):
            selected_parser = "unstructured"
            logger.info(
                f"Auto-selected 'unstructured' parser for {ext} file: {resolved_file_path}"
            )
        else:
            # Use the first available parser if none specified
            selected_parser = available_parsers[0]
            logger.info(f"No specific parser routed, using default: {selected_parser}")

    # Get parser instance and parse the file
    try:
        parser = document_parser_registry.get_parser(selected_parser)
        result = await parser.parse(
            resolved_file_path,
            progress_callback=progress_callback,
            **tool_args.parser_kwargs,
        )

        logger.info(f"Successfully parsed document with {selected_parser}")
        return result

    except Exception as e:
        logger.error(f"Failed to parse document with {selected_parser}: {e}")
        raise RuntimeError(
            f"Document parsing failed with {selected_parser}: {e}"
        ) from e


async def parse_document_with_output(
    tool_args: DocumentParseWithOutputArgs,
) -> DocumentParseWithOutputResult:
    """Parse a document and write the result to a file.

    This function parses a document using the specified parser and capabilities,
    then writes the structured result (full text, segments, figures, tables, metadata)
    to the specified output file path.

    Args:
        tool_args: Configuration object containing file path, parser name (optional),
                   capabilities requirements, and output file path.

    Returns:
        DocumentParseWithOutputResult indicating successful completion.

    Raises:
        ValueError: If no parsers match the required capabilities or if the requested
                   parser doesn't meet requirements.
        RuntimeError: If document parsing fails or file writing fails.
    """
    # Get the parsing result from the parent parser
    result: ParseResult = await parse_document(tool_args)

    # Resolve the output path
    resolved_output_path = tool_args.output_path

    try:
        output_ext = Path(resolved_output_path).suffix.lower()
        if tool_args.output_format == "json" or output_ext == ".json":
            write_json_output(result, resolved_output_path)
        elif tool_args.output_format == "txt" or output_ext == ".txt":
            write_text_output(result, resolved_output_path)
        else:  # Default to markdown
            write_markdown_output(result, resolved_output_path)

        logger.info(f"Successfully wrote parsing result to: {resolved_output_path}")
        return DocumentParseWithOutputResult()

    except Exception as e:
        logger.error(f"Failed to write parsing result to file: {e}")
        raise RuntimeError(
            f"Failed to write output to {resolved_output_path}: {e}"
        ) from e


def filter_parsers_by_capabilities(
    parsers: dict[str, type[Any]], capabilities: DocumentCapabilities
) -> list[str]:
    """Filter parsers based on required capabilities."""
    compatible_parsers = []

    for parser_name, parser_class in parsers.items():
        parser_capabilities = parser_class.get_capabilities()
        capability_names = {cap.__name__ for cap in parser_capabilities}

        # Check text parsing capability
        if capabilities.capability_text and "TextParsing" not in capability_names:
            continue

        # Check figure parsing capability
        if capabilities.capability_figure and "FigureParsing" not in capability_names:
            continue

        # Check result format requirements
        if (
            capabilities.requires_full_text_result
            and "FullTextResult" not in capability_names
        ):
            continue

        if (
            capabilities.requires_segmented_result
            and "SegmentedTextResult" not in capability_names
        ):
            continue

        # Check local/remote preference
        if capabilities.use_local_parser and "LocalParsing" not in capability_names:
            continue
        elif (
            not capabilities.use_local_parser
            and "RemoteParsing" not in capability_names
        ):
            continue

        compatible_parsers.append(parser_name)

    return compatible_parsers


class DocumentParserRegistry:
    """Simple registry for document parser implementations."""

    def __init__(self) -> None:
        self._parsers: dict[str, type[DocumentParser]] = {}

    def register_parser(self, name: str, parser_class: type[Any]) -> None:
        """Register a document parser implementation."""
        if name in self._parsers:
            raise ValueError(f"Parser '{name}' already registered")
        self._parsers[name] = parser_class

    def get_parser(self, name: str, **kwargs: Any) -> DocumentParser:
        """Get a parser instance by name."""
        if name not in self._parsers:
            available = ", ".join(self._parsers.keys())
            raise ValueError(f"Parser '{name}' not found. Available: {available}")
        parser_class = self._parsers[name]
        return parser_class(**kwargs)

    def parsers(self) -> dict[str, type[Any]]:
        """Get all registered parsers"""
        return self._parsers


document_parser_registry = DocumentParserRegistry()

document_parser_registry.register_parser("pypdf", PyPdfParser)
document_parser_registry.register_parser("pdfplumber", PdfPlumberParser)
document_parser_registry.register_parser("unstructured", UnstructuredParser)
document_parser_registry.register_parser("pymupdf", PyMuPdfParser)
# Only register deepdoc parser if it's available
if DeepDocParser is not None:
    document_parser_registry.register_parser("deepdoc", DeepDocParser)


class BaseFormatter:
    """Base class for ParseResult formatters."""

    def write_full_text(self, content: str | None) -> None:
        """Write full text content."""
        pass

    def write_text_segment(self, index: int, segment: Any) -> None:
        """Write a text segment."""
        pass

    def write_figure(self, index: int, figure: Any) -> None:
        """Write a figure."""
        pass

    def write_table(self, index: int, table: Any) -> None:
        """Write a table."""
        pass

    def write_metadata(self, metadata: dict[str, Any] | None) -> None:
        """Write metadata."""
        pass


def _iterate_parse_result(result: ParseResult, formatter: BaseFormatter) -> None:
    """Iterate over ParseResult and call formatter for each element.

    Args:
        result: The ParseResult to iterate over.
        formatter: The formatter to call for each element.
    """
    for i, segment in enumerate(result.text_segments or [], 1):
        formatter.write_text_segment(i, segment)
    for i, figure in enumerate(result.figures or [], 1):
        formatter.write_figure(i, figure)
    for i, table in enumerate(result.tables or [], 1):
        formatter.write_table(i, table)
    formatter.write_metadata(result.metadata)


class TextFormatter(BaseFormatter):
    """Formatter for plain text output."""

    def __init__(self, file_handle: Any) -> None:
        self.f = file_handle

    def write_full_text(self, content: str | None) -> None:
        if content is not None:
            self.f.write("=== FULL TEXT ===\n")
            self.f.write(content)
            self.f.write("\n\n")

    def write_text_segment(self, index: int, segment: Any) -> None:
        self.f.write(f"--- Segment {index} ---\n")
        self.f.write(f"Text: {segment.text}\n")
        if segment.metadata:
            self.f.write(f"Metadata: {segment.metadata}\n")
        self.f.write("\n")

    def write_figure(self, index: int, figure: Any) -> None:
        self.f.write(f"--- Figure {index} ---\n")
        self.f.write(f"Text: {figure.text}\n")
        if figure.metadata:
            self.f.write(f"Metadata: {figure.metadata}\n")
        self.f.write("\n")

    def write_table(self, index: int, table: Any) -> None:
        self.f.write(f"--- Table {index} ---\n")
        if table.html:
            self.f.write(f"HTML: {table.html}\n")
        if table.metadata:
            self.f.write(f"Metadata: {table.metadata}\n")
        self.f.write("\n")

    def write_metadata(self, metadata: dict[str, Any] | None) -> None:
        if metadata is not None:
            self.f.write("=== METADATA ===\n")
            for key, value in metadata.items():
                self.f.write(f"{key}: {value}\n")
            self.f.write("\n")


def write_text_output(result: ParseResult, output_path: str) -> None:
    """Write ParseResult to file in plain text format."""
    with open(output_path, "w", encoding="utf-8") as f:
        formatter = TextFormatter(f)

        if result.full_text is not None:
            formatter.write_full_text(result.full_text)

        if result.text_segments is not None:
            f.write("=== TEXT SEGMENTS ===\n")
            for i, segment in enumerate(result.text_segments, 1):
                formatter.write_text_segment(i, segment)
            f.write("\n")

        if result.figures is not None:
            f.write("=== FIGURES ===\n")
            for i, figure in enumerate(result.figures, 1):
                formatter.write_figure(i, figure)
            f.write("\n")

        if result.tables is not None:
            f.write("=== TABLES ===\n")
            for i, table in enumerate(result.tables, 1):
                formatter.write_table(i, table)
            f.write("\n")

        formatter.write_metadata(result.metadata)


class JsonFormatter(BaseFormatter):
    """Formatter for JSON output."""

    def __init__(self) -> None:
        self.result_dict: Dict[str, Any] = {
            "full_text": None,
            "text_segments": [],
            "figures": [],
            "tables": [],
            "metadata": {},
        }

    def write_full_text(self, content: str | None) -> None:
        self.result_dict["full_text"] = content

    def write_text_segment(self, index: int, segment: Any) -> None:
        self.result_dict["text_segments"].append(
            {"text": segment.text, "metadata": segment.metadata}
        )

    def write_figure(self, index: int, figure: Any) -> None:
        self.result_dict["figures"].append(
            {"text": figure.text, "metadata": figure.metadata}
        )

    def write_table(self, index: int, table: Any) -> None:
        self.result_dict["tables"].append(
            {"html": table.html, "metadata": table.metadata}
        )

    def write_metadata(self, metadata: dict[str, Any] | None) -> None:
        self.result_dict["metadata"] = metadata or {}

    def get_result(self) -> dict[str, Any]:
        return self.result_dict


def write_json_output(result: ParseResult, output_path: str) -> None:
    """Write ParseResult to file in JSON format."""
    import json

    formatter = JsonFormatter()

    formatter.write_full_text(result.full_text)
    _iterate_parse_result(result, formatter)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatter.get_result(), f, ensure_ascii=False, indent=2)


class MarkdownFormatter(BaseFormatter):
    """Formatter for Markdown output."""

    def __init__(self, file_handle: Any) -> None:
        self.f = file_handle

    def write_full_text(self, content: str | None) -> None:
        if content is not None:
            self.f.write("# Full Text\n\n")
            self.f.write(content)
            self.f.write("\n\n")

    def write_text_segment(self, index: int, segment: Any) -> None:
        self.f.write(f"## Segment {index}\n\n")
        self.f.write(f"{segment.text}\n\n")
        if segment.metadata:
            self.f.write("**Metadata:**\n")
            for key, value in segment.metadata.items():
                self.f.write(f"- {key}: {value}\n")
            self.f.write("\n")

    def write_figure(self, index: int, figure: Any) -> None:
        self.f.write(f"## Figure {index}\n\n")
        if figure.text:
            self.f.write(f"{figure.text}\n\n")
        if figure.metadata:
            self.f.write("**Metadata:**\n")
            for key, value in figure.metadata.items():
                self.f.write(f"- {key}: {value}\n")
            self.f.write("\n")

    def write_table(self, index: int, table: Any) -> None:
        self.f.write(f"## Table {index}\n\n")
        if table.html:
            self.f.write(f"{html_table_to_markdown(table.html)}\n\n")
        if table.metadata:
            self.f.write("**Metadata:**\n")
            for key, value in table.metadata.items():
                self.f.write(f"- {key}: {value}\n")
            self.f.write("\n")

    def write_metadata(self, metadata: dict[str, Any] | None) -> None:
        if metadata is not None:
            self.f.write("# Metadata\n\n")
            for key, value in metadata.items():
                self.f.write(f"- {key}: {value}\n")
            self.f.write("\n")


def write_markdown_output(result: ParseResult, output_path: str) -> None:
    """Write ParseResult to file in Markdown format."""
    with open(output_path, "w", encoding="utf-8") as f:
        formatter = MarkdownFormatter(f)

        formatter.write_full_text(result.full_text)

        if result.text_segments is not None:
            f.write("# Text Segments\n\n")
            for i, segment in enumerate(result.text_segments, 1):
                formatter.write_text_segment(i, segment)
            f.write("\n")

        if result.figures is not None:
            f.write("# Figures\n\n")
            for i, figure in enumerate(result.figures, 1):
                formatter.write_figure(i, figure)
            f.write("\n")

        if result.tables is not None:
            f.write("# Tables\n\n")
            for i, table in enumerate(result.tables, 1):
                formatter.write_table(i, table)
            f.write("\n")

        formatter.write_metadata(result.metadata)


def html_table_to_markdown(html_content: str) -> str:
    """Convert HTML table to Markdown table format."""
    try:
        from bs4 import BeautifulSoup, Tag

        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table")
        if not table or not isinstance(table, Tag):
            return html_content

        markdown_lines = []

        # Extract headers
        headers = table.find_all("th")
        if headers:
            header_row = [
                th.get_text(strip=True) for th in headers if isinstance(th, Tag)
            ]
            markdown_lines.append("| " + " | ".join(header_row) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")

        # Extract data rows
        rows = table.find_all("tr")
        for row in rows:
            if isinstance(row, Tag):
                cells = row.find_all(["td", "th"])
                if cells:
                    row_data = [
                        cell.get_text(strip=True)
                        for cell in cells
                        if isinstance(cell, Tag)
                    ]
                    markdown_lines.append("| " + " | ".join(row_data) + " |")

        return "\n".join(markdown_lines)
    except ImportError:
        # If BeautifulSoup is not available, return original HTML
        logger.warning("BeautifulSoup not available, keeping HTML table format")
        return html_content
    except Exception as e:
        logger.warning(f"Failed to convert HTML table to Markdown: {e}")
        return html_content

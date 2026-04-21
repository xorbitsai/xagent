import asyncio
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zipfile import BadZipFile

from deepdoc import ExcelParser as DeepDocExcelParser
from deepdoc import MarkdownParser as DeepDocMarkdownParser
from deepdoc import PdfParser as DeepDocPdfParser
from deepdoc import TxtParser as DeepDocTxtParser
from deepdoc.parser import DoclingParser as DeepDocDoclingParser
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from ...core.tools.core.RAG_tools.core.config import ARTIFACTS_DIR
from ...core.tools.core.RAG_tools.utils.string_utils import sanitize_for_doc_id
from .base import (
    DocumentParser,
    FigureParsing,
    LocalParsing,
    ParsedFigures,
    ParsedTable,
    ParsedTextSegment,
    ParseResult,
    SegmentedTextResult,
    TextParsing,
    validate_office_file_format,
)

logger = logging.getLogger(__name__)


def _handle_image(image_obj: Any, doc_id: str) -> str:
    """Handle image object - return path if already a file path, otherwise save and return path.

    Args:
        image_obj: Either a file path (str) or PIL Image object
        doc_id: Document ID for organizing saved images

    Returns:
        Path to the image file

    Raises:
        ValueError: If path doesn't exist or unsupported image type
    """
    # If it's already a string path, validate and return
    if isinstance(image_obj, str):
        image_path = Path(image_obj)
        if image_path.exists() and image_path.is_file():
            return str(image_path)
        else:
            raise ValueError(f"Image path does not exist or is not a file: {image_obj}")

    # If it's a PIL Image object, save it
    elif hasattr(image_obj, "save"):
        return _save_image_to_disk(doc_id, image_obj)

    else:
        raise ValueError(f"Unsupported image type: {type(image_obj)}")


def _save_image_to_disk(doc_id: str, image_obj: Any) -> str:
    """Saves a PIL image object to disk and returns the path."""
    # Convert PIL Image to bytes
    img_byte_arr = BytesIO()
    image_obj.save(img_byte_arr, format="PNG")
    image_binary = img_byte_arr.getvalue()

    return _save_bytes_to_disk(doc_id, image_binary, ".png")


def _save_bytes_to_disk(doc_id: str, image_bytes: bytes, suffix: str = ".png") -> str:
    """Saves raw image bytes to disk and returns the path."""
    safe_doc_id = sanitize_for_doc_id(doc_id, max_length=64)
    base_dir = ARTIFACTS_DIR / "providers" / "deepdoc"
    image_dir = base_dir / safe_doc_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_filename = f"{uuid.uuid4()}{suffix}"
    image_path = image_dir / image_filename

    # Write binary image data directly to file
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    return str(image_path)


def _build_element_metadata(
    bbox: Dict[str, Any], doc_id: str, **kwargs: Any
) -> Dict[str, Any]:
    """Build metadata dict for an element."""
    layout_type = bbox.get("layout_type", "text")
    metadata = {
        "layout_type": layout_type,
        "doc_id": doc_id,
        "page_number": bbox.get("page_number", 1),
        **kwargs,
    }

    # Extract col_id for two-column layout support
    col_id = bbox.get("col_id", 0)
    metadata["col_id"] = col_id

    # Extract positions for PDF visualization (format: [[page_num, left, right, top, bottom], ...])
    # We'll enrich it with col_id to format: [[page_num, col_id, left, right, top, bottom], ...]
    positions = bbox.get("positions", [])
    if positions:
        enriched_positions = []
        for pos in positions:
            if isinstance(pos, (list, tuple)) and len(pos) >= 5:
                # pos format: [page_num, left, right, top, bottom]; len already >= 5
                enriched_positions.append(
                    [
                        int(pos[0]),
                        col_id,
                        float(pos[1]),
                        float(pos[2]),
                        float(pos[3]),
                        float(pos[4]),
                    ]
                )
        if enriched_positions:
            metadata["positions"] = enriched_positions

    return metadata


def _process_table_element(
    bbox: Dict[str, Any], base_metadata: Dict[str, Any], doc_id: str
) -> ParsedTable:
    """Process a table element into ParsedTable."""
    # Handle table image
    image_path = None
    if "image" in bbox and bbox["image"]:
        image_path = _handle_image(bbox["image"], doc_id)

    table_metadata = base_metadata.copy()
    table_metadata["image_path"] = image_path
    table_metadata["type"] = "table"
    # positions and col_id are already in base_metadata from _build_element_metadata

    return ParsedTable(html=bbox.get("text", ""), image=None, metadata=table_metadata)


def _process_figure_element(
    bbox: Dict[str, Any], base_metadata: Dict[str, Any], doc_id: str
) -> ParsedFigures:
    """Process a figure element into ParsedFigures."""
    # Handle figure image
    image_path = None
    logger.debug(f"Processing figure bbox: {bbox.keys()}")
    if "image" in bbox:
        logger.debug(
            f"Figure has image key, value type: {type(bbox['image'])}, value: {bbox['image']}"
        )
        if bbox["image"]:
            try:
                image_path = _handle_image(bbox["image"], doc_id)
                logger.debug(f"Successfully saved figure image to: {image_path}")
            except Exception as e:
                logger.error(f"Failed to handle figure image: {e}")
                image_path = None
        else:
            logger.debug("Figure image value is empty/falsy")
    else:
        logger.debug("Figure bbox does not have 'image' key")

    figure_metadata = base_metadata.copy()
    figure_metadata["image_path"] = image_path
    figure_metadata["type"] = "figure"
    # positions and col_id are already in base_metadata from _build_element_metadata

    # Ensure figure has text for proper processing
    figure_text = bbox.get("text", "").strip()
    if not figure_text:
        figure_text = "Figure"

    logger.debug(
        f"Processed figure: text='{bbox.get('text', '')[:50]}...', image_path={image_path}"
    )
    return ParsedFigures(text=figure_text, image=None, metadata=figure_metadata)


def _translate_pdf_bboxes(
    doc_id: str, bboxes: List[Dict[str, Any]], **kwargs: Any
) -> ParseResult:
    """Translate ragflow-style bboxes into our ParseResult format.

    Args:
        doc_id: Document ID
        bboxes: List of unified document elements from parse_into_bboxes
        **kwargs: Additional metadata

    Returns:
        ParseResult with text_segments, tables, and figures
    """
    text_segments, figures, tables = [], [], []

    for bbox in bboxes:
        layout_type = bbox.get("layout_type", "text")
        element_metadata = _build_element_metadata(bbox, doc_id, **kwargs)

        # Process different layout types
        if layout_type == "text":
            text_segments.append(
                ParsedTextSegment(text=bbox.get("text", ""), metadata=element_metadata)
            )
        elif layout_type == "table":
            tables.append(_process_table_element(bbox, element_metadata, doc_id))
        elif layout_type == "figure":
            figures.append(_process_figure_element(bbox, element_metadata, doc_id))
        else:
            # Handle other layout types as text for now
            logger.debug(f"Unhandled layout_type '{layout_type}', treating as text")
            text_segments.append(
                ParsedTextSegment(text=bbox.get("text", ""), metadata=element_metadata)
            )

    logger.info(
        f"Translated {len(bboxes)} bboxes into {len(text_segments)} text segments, {len(tables)} tables, {len(figures)} figures"
    )
    return ParseResult(
        text_segments=text_segments, figures=figures, tables=tables, metadata=kwargs
    )


def _translate_docx_output(
    doc_id: str, raw_output: Tuple[Any, Any], **kwargs: Any
) -> ParseResult:
    """
    Translate DOCX parser output to ParseResult.

    Supports two formats:
    1. Old format (from RAGFlowDocxParser):
       - raw_sections: List[tuple[str, str]] - (text, style_name)
       - raw_tables: List[List[str]] - table rows

    2. New format (from DoclingParser.parse_docx()):
       - raw_sections: List[tuple[str, str]] - (text, style_name) tuples
         - style_name is mapped from Docling label (e.g., "Heading", "Normal", "List Item")
       - raw_tables: List[tuple[tuple, str]] - ((image, html_or_captions), positions) tuples
         - For tables: ((None, html), "")
         - For pictures: ((PIL_Image, [captions]), "")
    """
    raw_sections, raw_tables = raw_output
    text_segments = []
    figures = []
    tables = []

    # Process text sections (same format for both old and new)
    for section in raw_sections:
        if isinstance(section, tuple):
            if len(section) == 2:
                text, style_or_tag = section
            elif len(section) == 3:
                text, typ, tag = section
                style_or_tag = tag
            else:
                text = section[0] if section else ""
                style_or_tag = ""
        else:
            text = str(section)
            style_or_tag = ""

        if text.strip():
            segment_metadata = {"doc_id": doc_id, **kwargs}
            # Preserve style if it's from old parser
            if style_or_tag and not style_or_tag.startswith("@@"):
                segment_metadata["style"] = style_or_tag
            text_segments.append(
                ParsedTextSegment(text=text.strip(), metadata=segment_metadata)
            )

    # Process tables and pictures
    # Check if it's new format (DoclingParser) or old format (RAGFlowDocxParser)
    # New format: tuple of ((image, html_or_captions), positions)
    def _is_new_format(table_item: Any) -> bool:
        return (
            isinstance(table_item, tuple)
            and len(table_item) == 2
            and isinstance(table_item[0], tuple)
            and len(table_item[0]) == 2
        )

    is_new_format = raw_tables and _is_new_format(raw_tables[0])

    if is_new_format:
        # New format: DoclingParser output
        for table_item in raw_tables:
            if not isinstance(table_item, tuple) or len(table_item) != 2:
                continue

            (img_or_none, html_or_captions), positions = table_item

            # Determine if it's a table or a picture
            if img_or_none is not None:
                # It's a picture
                if isinstance(html_or_captions, list):
                    captions = html_or_captions
                else:
                    captions = [html_or_captions] if html_or_captions else []

                caption_text = "\n".join(captions) if captions else ""

                # Save the image
                image_path = _save_image_to_disk(doc_id, img_or_none)

                figure_metadata = kwargs.copy()
                figure_metadata.update(
                    {
                        "type": "figure",
                        "image_path": image_path,
                        "doc_id": doc_id,
                        "parser": "deepdoc",
                    }
                )
                figures.append(
                    ParsedFigures(
                        text=caption_text, image=None, metadata=figure_metadata
                    )
                )
            else:
                # It's a table
                table_html = ""
                table_caption = ""

                if isinstance(html_or_captions, str):
                    # Simple HTML string (no caption)
                    table_html = html_or_captions
                elif isinstance(html_or_captions, dict):
                    # Dict format with caption and html
                    table_caption = html_or_captions.get("caption", "")
                    table_html = html_or_captions.get("html", "")
                elif isinstance(html_or_captions, list):
                    # Fallback: if it's a list, join it
                    table_html = "\n".join(html_or_captions)

                table_metadata = kwargs.copy()
                table_metadata.update(
                    {
                        "type": "table",
                        "doc_id": doc_id,
                        "parser": "deepdoc",
                    }
                )
                # Add caption to metadata if present
                if table_caption:
                    table_metadata["caption"] = table_caption

                tables.append(
                    ParsedTable(html=table_html, image=None, metadata=table_metadata)
                )
    else:
        # Old format: RAGFlowDocxParser output (List[List[str]])
        for table_rows in raw_tables:
            if isinstance(table_rows, list):
                # Join the list of row strings into a single text block for the table
                table_html = "\n".join(table_rows) if table_rows else ""
                table_metadata = kwargs.copy()
                table_metadata.update(
                    {
                        "type": "table",
                        "doc_id": doc_id,
                        "parser": "deepdoc",
                    }
                )
                tables.append(
                    ParsedTable(html=table_html, image=None, metadata=table_metadata)
                )

    return ParseResult(text_segments=text_segments, figures=figures, tables=tables)


def _translate_excel_output(raw_output: Any, **kwargs: Any) -> ParseResult:
    text_segments = [ParsedTextSegment(text=row, metadata=kwargs) for row in raw_output]
    return ParseResult(text_segments=text_segments)


def _translate_text_output(raw_output: Any, **kwargs: Any) -> ParseResult:
    """Translates simple text parser output (list of lists) into a ParseResult."""
    text_segments = []
    if isinstance(raw_output, list) and all(
        isinstance(item, list) for item in raw_output
    ):
        for item in raw_output:
            if item and isinstance(item[0], str):
                text_segments.append(ParsedTextSegment(text=item[0], metadata=kwargs))
    elif isinstance(raw_output, str):
        text_segments.append(ParsedTextSegment(text=raw_output, metadata=kwargs))
    return ParseResult(text_segments=text_segments)


def _translate_markdown_output(raw_output: Any, **kwargs: Any) -> ParseResult:
    """Translates markdown parser output (tuple of text and tables) into a ParseResult."""
    text_segments = []
    if isinstance(raw_output, tuple) and len(raw_output) == 2:
        remainder_text, tables = raw_output
        if remainder_text:
            text_segments.append(
                ParsedTextSegment(text=remainder_text, metadata=kwargs)
            )
        for tbl in tables:
            table_meta = kwargs.copy()
            table_meta["type"] = "table"
            text_segments.append(ParsedTextSegment(text=tbl, metadata=table_meta))
    return ParseResult(text_segments=text_segments)


def _parse_docx_with_python_docx(
    file_path: str | BytesIO,
) -> Tuple[List[Tuple[str, str]], List[List[str]]]:
    """Fallback DOCX parser when Docling is unavailable."""
    from docx import Document as DocxDocument

    if isinstance(file_path, BytesIO):
        file_path.seek(0)
        document = DocxDocument(file_path)
        file_path.seek(0)
    else:
        document = DocxDocument(file_path)

    sections: List[Tuple[str, str]] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style is not None else ""
        sections.append((text, style_name))

    tables: List[List[str]] = []
    for table in document.tables:
        rows: List[str] = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            tables.append(rows)

    return sections, tables


class DeepDocParser(
    DocumentParser, TextParsing, FigureParsing, SegmentedTextResult, LocalParsing
):
    """
    A universal parser powered by DeepDoc, supporting multiple file formats.

    Supports both standard parsing (converted to unified format) and raw output passthrough
    for advanced visualization features.
    """

    # DeepDoc supports multiple structured document formats
    supported_extensions = [
        ".pdf",
        ".docx",
        ".xlsx",
        ".xls",
        ".csv",
        ".md",
        ".txt",
        ".json",
        ".html",
    ]

    def __init__(self, enable_raw_output: bool = False) -> None:
        """
        Initialize the DeepDoc parser.

        Args:
            enable_raw_output: If True, include raw parser output in ParseResult for visualization
        """
        self._parsers: dict[str, Any] = {}
        self.enable_raw_output = enable_raw_output

    def _get_parser(self, file_path: str) -> Tuple[Any, str]:
        """Get parser and extension for a file path."""
        ext = Path(file_path).suffix.lower()
        return self._get_parser_for_ext(ext), ext

    def _get_parser_for_ext(self, ext: str) -> Any:
        """Get parser for a specific file extension (used for file paths and BytesIO objects)."""
        if ext not in self._parsers:
            if ext == ".pdf":
                self._parsers[ext] = DeepDocPdfParser()
            elif ext == ".docx":
                # Use DoclingParser for DOCX to support images and captions
                self._parsers[ext] = DeepDocDoclingParser()
            elif ext in [".xlsx", ".xls", ".csv"]:
                self._parsers[ext] = DeepDocExcelParser()
            elif ext == ".md":
                self._parsers[ext] = DeepDocMarkdownParser()
            elif ext in [".txt", ".json", ".html"]:
                self._parsers[ext] = DeepDocTxtParser()
            else:
                raise ValueError(f"DeepDoc does not support file type: {ext}")
        return self._parsers[ext]

    async def _parse_impl(
        self, file_path: str | BytesIO, progress_callback: Any = None, **kwargs: Any
    ) -> ParseResult:
        # Handle Excel/CSV file compatibility - convert to BytesIO if needed
        if isinstance(file_path, str):
            path_obj = Path(file_path)
            # Check if this is an Excel/CSV file that might need BytesIO conversion
            excel_extensions = [".xlsx", ".xls", ".csv"]
            if path_obj.suffix.lower() in excel_extensions:
                logger.debug(
                    f"Detected Excel/CSV file: {path_obj.name}, attempting BytesIO conversion"
                )
                try:
                    with open(file_path, "rb") as f:
                        file_content = f.read()
                    file_path = BytesIO(file_content)
                    kwargs["file_ext"] = path_obj.suffix.lower()
                    logger.debug(
                        f"Successfully converted {path_obj.name} to BytesIO object"
                    )
                except Exception as e:
                    logger.warning(f"Failed to convert Excel file to BytesIO: {e}")
                    # Continue with original file path

        def _sync_parse() -> ParseResult:
            # Handle different file path types (string path or BytesIO object)
            nonlocal file_path  # Access the potentially modified file_path

            if isinstance(file_path, BytesIO):
                # For BytesIO objects, we need to determine file type from kwargs or use a default
                ext = kwargs.get("file_ext", ".xlsx")  # Default to Excel for BytesIO
                parser = self._get_parser_for_ext(ext)
            else:
                parser, ext = self._get_parser(file_path)

            # Extract doc_id from kwargs if available, otherwise use file_path as fallback
            if isinstance(file_path, BytesIO):
                doc_id = kwargs.get("doc_id", "bytesio_document")
            else:
                doc_id = kwargs.get("doc_id", str(Path(file_path).stem))

            base_kwargs = {
                key: value for key, value in kwargs.items() if key != "doc_id"
            }
            metadata = {
                "source": str(file_path)
                if not isinstance(file_path, BytesIO)
                else "memory_buffer",
                "file_type": ext,
                "parse_method": "deepdoc",
                **base_kwargs,
            }

            if ext == ".xlsx":
                return _parse_xlsx_rows(file_path, **metadata)

            # Dispatch to correct parser method and translator
            if ext == ".md":
                if isinstance(file_path, BytesIO):
                    markdown_text = file_path.getvalue().decode("utf-8")
                    file_path.seek(0)  # Reset position for parser
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        markdown_text = f.read()
                markdown_output = parser.extract_tables_and_remainder(markdown_text)
                return _translate_markdown_output(markdown_output, **metadata)

            # For TXT files, read directly to preserve original format and punctuation
            if ext == ".txt":
                if isinstance(file_path, BytesIO):
                    text_content = file_path.getvalue().decode("utf-8")
                    file_path.seek(0)  # Reset position for parser
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                return _translate_text_output(text_content, **metadata)

            # Validate Office document format (.docx, .xlsx, .pptx)
            # Skip validation for BytesIO objects as they're already validated during conversion
            if ext in [".docx", ".xlsx", ".pptx"] and not isinstance(
                file_path, BytesIO
            ):
                validate_office_file_format(
                    file_path, ext, strict=True, parser_name="deepdoc"
                )

            # Most other parsers are callable
            parser_call_kwargs: dict[str, Any] = {}
            bboxes: List[Dict[str, Any]] = []
            raw_output: Any = None

            if ext == ".pdf":
                parser_call_kwargs = base_kwargs.copy()
                parser_call_kwargs.setdefault("return_html", True)
                parser_call_kwargs.setdefault("need_image", True)
                # Note: __call__ doesn't accept need_position, we'll extract positions separately

            try:
                # For PDF, use parse_into_bboxes for unified document structure with positions
                if ext == ".pdf":
                    # Use standard DeepDoc parser
                    zoomin = parser_call_kwargs.get("zoomin", 3)

                    # Set up progress callback for DeepDoc if provided
                    callback = None
                    if progress_callback is not None:
                        from ...core.tools.core.RAG_tools.progress.adapters import (
                            DeepDocProgressAdapter,
                        )

                        adapter = DeepDocProgressAdapter(progress_callback)
                        callback = adapter.get_callback()

                    bboxes = parser.parse_into_bboxes(
                        file_path, callback=callback, zoomin=zoomin
                    )
                    logger.info(
                        f"Parsed PDF into {len(bboxes)} unified elements with position information"
                    )
                else:
                    # Handle DOCX with DoclingParser
                    if ext == ".docx" and isinstance(parser, DeepDocDoclingParser):
                        if isinstance(file_path, BytesIO):
                            # DoclingParser needs a file path, not BytesIO
                            # For BytesIO, fall back to old parser behavior
                            file_bytes: bytes = file_path.getvalue()
                            raw_output = parser(file_bytes, **parser_call_kwargs)
                        else:
                            # Use DoclingParser.parse_docx() for DOCX files when available,
                            # otherwise fall back to python-docx parsing.
                            if (
                                getattr(parser, "check_installation", None)
                                and not parser.check_installation()
                            ):
                                raw_output = _parse_docx_with_python_docx(file_path)
                            else:
                                raw_output = parser.parse_docx(
                                    file_path, **parser_call_kwargs
                                )
                    else:
                        # DeepDoc ExcelParser expects bytes if not path, not BytesIO
                        input_arg: str | BytesIO | bytes = file_path
                        if isinstance(file_path, BytesIO) and ext in [
                            ".xlsx",
                            ".xls",
                            ".csv",
                        ]:
                            input_arg = file_path.getvalue()

                        raw_output = parser(input_arg, **parser_call_kwargs)

            except KeyError as e:
                # Catch KeyError from python-docx, usually due to format mismatch
                if ext == ".docx" and "relationship" in str(e).lower():
                    raise ValueError(
                        f"File '{file_path}' has extension .docx but the format may be incompatible.\n"
                        f"DeepDoc DocxParser only supports Open XML .docx (Office 2007+).\n"
                        f"Ensure the file is valid .docx or use another parser."
                    ) from e
                raise

            if ext == ".pdf":
                parse_result = _translate_pdf_bboxes(doc_id, bboxes, **metadata)

                # Add raw output passthrough if enabled
                if self.enable_raw_output:
                    parse_result.raw_parser_output = {
                        "format": "deepdoc_pdf",
                        "bboxes": bboxes,
                        "total_elements": len(bboxes),
                        "has_positions": any("positions" in bbox for bbox in bboxes),
                    }
                    parse_result.parser_engine = "deepdoc"

                return parse_result
            elif ext == ".docx":
                # raw_output is already set in the try block above
                return _translate_docx_output(doc_id, raw_output, **metadata)
            elif ext in [".xlsx", ".xls", ".csv"]:
                return _translate_excel_output(raw_output, **metadata)
            elif ext in [".json", ".html"]:
                return _translate_text_output(raw_output, **metadata)

            return ParseResult(metadata=metadata)

        # Execute the synchronous parsing logic in a thread executor
        # This isolates it from the current event loop, allowing deepdoc's internal
        # asyncio.run() calls to work correctly without "running loop" conflicts.
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _sync_parse)
        except RuntimeError:
            # Fallback if no loop is running (unlikely in async method, but safe)
            return _sync_parse()


def _normalize_spreadsheet_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_title_row(cells: List[str]) -> bool:
    return len(cells) == 1 and len(cells[0]) > 4


def _format_spreadsheet_row(headers: List[str] | None, cells: List[str]) -> str:
    if headers:
        pairs = []
        for index, value in enumerate(cells):
            if not value:
                continue
            header = (
                headers[index]
                if index < len(headers) and headers[index]
                else f"col_{index + 1}"
            )
            pairs.append(f"{header}: {value}")
        if pairs:
            return " | ".join(pairs)
    return " | ".join(value for value in cells if value)


def _parse_xlsx_rows(file_path: str | BytesIO, **kwargs: Any) -> ParseResult:
    workbook = None
    source = file_path if isinstance(file_path, str) else "in-memory spreadsheet"
    try:
        if isinstance(file_path, BytesIO):
            file_path.seek(0)
            workbook = load_workbook(filename=file_path, read_only=True, data_only=True)
        else:
            workbook = load_workbook(filename=file_path, read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException, OSError, ValueError) as exc:
        logger.warning("Failed to parse spreadsheet rows from %s: %s", source, exc)
        raise ValueError(
            f"Failed to parse spreadsheet rows from {source}: {exc}"
        ) from exc

    try:
        text_segments: List[ParsedTextSegment] = []
        multiple_sheets = len(workbook.sheetnames) > 1

        for sheet in workbook.worksheets:
            headers: List[str] | None = None
            title_seen = False
            for row_number, row in enumerate(
                sheet.iter_rows(values_only=True), start=1
            ):
                cells = [_normalize_spreadsheet_cell(cell) for cell in row]
                non_empty = [cell for cell in cells if cell]
                if not non_empty:
                    continue

                metadata = {
                    **kwargs,
                    "sheet_name": sheet.title,
                    "row_number": row_number,
                }

                if not title_seen and _is_title_row(non_empty):
                    title_seen = True
                    text = non_empty[0]
                    if multiple_sheets:
                        text = f"[{sheet.title}] {text}"
                    metadata["row_type"] = "title"
                    text_segments.append(
                        ParsedTextSegment(text=text, metadata=metadata)
                    )
                    continue

                if headers is None and len(non_empty) > 1:
                    headers = cells
                    header_text = " | ".join(value for value in headers if value)
                    if header_text:
                        if multiple_sheets:
                            header_text = f"[{sheet.title}] {header_text}"
                        metadata["row_type"] = "header"
                        text_segments.append(
                            ParsedTextSegment(text=header_text, metadata=metadata)
                        )
                    continue

                row_text = _format_spreadsheet_row(headers, cells)
                if not row_text:
                    continue
                metadata["row_type"] = "data"
                text_segments.append(
                    ParsedTextSegment(text=row_text, metadata=metadata)
                )

        return ParseResult(text_segments=text_segments)
    finally:
        workbook.close()

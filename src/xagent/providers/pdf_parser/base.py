import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Capability markers
class TextParsing:
    """Capability of parsing text"""

    pass


class FigureParsing:
    """Capability of parsing images and tables"""

    pass


# Parsing format markers
class FullTextResult:
    """Returns parsing result in a single string"""

    pass


class SegmentedTextResult(FullTextResult):
    """Returns parsing result in segments"""

    pass


# Parsing provider markers
class LocalParsing:
    """Parser that uses local tools for parsing operations."""

    pass


class RemoteParsing:
    """Parser that uses remote API for parsing operations."""

    pass


class ParsedTextSegment(BaseModel):
    text: str = Field(
        description="The extracted text content from a specific segment of the document"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Metadata about the text segment.
        For deepdoc(pdf): contains 'positions' with coordinates.
        For deepdoc(docx): contains 'style' information.
        For deepdoc(excel): contains 'sheet_name' and 'row_number'.
        """,
    )


class ParsedTable(BaseModel):
    html: Optional[str] = Field(
        default=None, description="HTML representation of the table"
    )
    image: Optional[Any] = Field(
        default=None, description="PIL.Image object of the table, if available"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedFigures(BaseModel):
    text: str = Field(description="Caption or text associated with the figure")
    image: Optional[Any] = Field(
        default=None, description="PIL.Image object of the figure"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParseResult(BaseModel):
    full_text: str | None = Field(
        default=None,
        description="Complete text content of the document as a single concatenated string. Populated when full text result is requested",
    )
    text_segments: List[ParsedTextSegment] = Field(
        default_factory=list,
        description="List of text segments extracted from the document, typically organized by page or logical blocks.",
    )
    figures: List[ParsedFigures] = Field(
        default_factory=list,
        description="List of figures and images extracted from the document.",
    )
    tables: List[ParsedTable] = Field(
        default_factory=list,
        description="List of tables extracted from the document.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the parsing operation, such as parser used, processing time, or document properties",
    )

    # Enhanced fields for raw parser output passthrough
    raw_parser_output: Dict[str, Any] | None = Field(
        default=None,
        description="Raw output from the underlying parser engine (e.g., DeepDoc bboxes, Docling sections). "
        "Used for advanced visualization and analysis features.",
    )
    parser_engine: str | None = Field(
        default=None,
        description="Identifier for the parser engine used. Examples: 'deepdoc', 'docling', 'standard'. "
        "Helps consumers understand the format of raw_parser_output.",
    )

    @property
    def has_visualization_data(self) -> bool:
        """Check if this result contains data suitable for visualization."""
        return (
            self.raw_parser_output is not None
            and self.parser_engine is not None
            and self.parser_engine == "deepdoc"
        )

    @property
    def visualization_engine(self) -> str | None:
        """Get the visualization-compatible engine identifier."""
        if self.parser_engine == "deepdoc":
            return self.parser_engine
        return None

    def get_visualization_elements(self) -> List[Dict[str, Any]]:
        """
        Extract standardized visualization elements from raw parser output.

        Returns:
            List of elements with consistent structure for frontend visualization:
            [
                {
                    "id": "unique_element_id",
                    "type": "text|table|figure|equation",
                    "content": "text content, html, or latex",
                    "bbox": {"x0": float, "y0": float, "x1": float, "y1": float},
                    "page": int,
                    "metadata": {
                        "parser": "deepdoc",        # parser source
                        "content_type": "text|html|latex",   # content format
                        "has_image": bool,                   # whether has associated image
                        "image_type": str,                   # PIL.Image, etc.
                        "layout_type": str,                  # original layout classification
                        "confidence": float,                 # optional confidence score
                        "positions": [...],                  # detailed position data
                        "data_source": str,                  # "bboxes" or "sections/tables"
                        # ... other parser-specific metadata
                    }
                },
                ...
            ]
        """
        if not self.has_visualization_data:
            return []

        try:
            if self.parser_engine == "deepdoc":
                return self._extract_deepdoc_visualization_elements()
            else:
                return []
        except Exception:
            # Gracefully degrade if extraction fails
            return []

    def _extract_deepdoc_visualization_elements(self) -> List[Dict[str, Any]]:
        """Extract visualization elements from DeepDoc bbox format."""
        elements: List[Dict[str, Any]] = []
        if self.raw_parser_output is None:
            return elements
        bboxes = self.raw_parser_output.get("bboxes", [])

        for idx, bbox in enumerate(bboxes):
            layout_type = bbox.get("layout_type", "text")

            # Determine content and image handling based on layout type
            if layout_type == "table":
                # Table: content is HTML, may have image
                content = bbox.get("text", "")
                has_image = "image" in bbox
                content_type = "html"
            elif layout_type == "figure":
                # Figure: content is caption/title, has image
                content = bbox.get("text", "")
                has_image = "image" in bbox
                content_type = "text"
            elif layout_type == "equation":
                # Equation: content is math expression, may have rendered image
                content = bbox.get("text", "")
                has_image = "image" in bbox
                content_type = "latex"
            else:
                # Text and other types: plain text content
                content = bbox.get("text", "")
                has_image = False
                content_type = "text"

            element = {
                "id": f"deepdoc_{idx}",
                "type": layout_type,
                "content": content,
                "bbox": {
                    "x0": bbox.get("x0", 0),
                    "y0": bbox.get("y0", 0),
                    "x1": bbox.get("x1", 0),
                    "y1": bbox.get("y1", 0),
                },
                "page": bbox.get("page_number", 1),
                "metadata": {
                    "parser": "deepdoc",
                    "layout_type": layout_type,
                    "content_type": content_type,
                    "has_image": has_image,
                    "image_type": type(bbox.get("image")).__name__
                    if has_image
                    else None,
                    "positions": bbox.get("positions", []),
                    "confidence": bbox.get("score", None),  # If available
                },
            }
            elements.append(element)

        return elements

    def get_visualization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of visualization data for quick frontend overview.

        Returns:
            Summary statistics and metadata about the visualization elements.
        """
        if not self.has_visualization_data:
            return {
                "available": False,
                "parser_engine": None,
                "total_elements": 0,
            }

        elements = self.get_visualization_elements()

        # Count elements by type
        type_counts: Dict[str, int] = {}
        page_counts: Dict[int, int] = {}
        elements_with_images = 0

        for elem in elements:
            # Count by type
            elem_type = elem.get("type", "unknown")
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

            # Count by page
            page = elem.get("page", 1)
            page_counts[page] = page_counts.get(page, 0) + 1

            # Count elements with images
            if elem.get("metadata", {}).get("has_image", False):
                elements_with_images += 1

        return {
            "available": True,
            "parser_engine": self.parser_engine,
            "total_elements": len(elements),
            "elements_by_type": type_counts,
            "elements_by_page": page_counts,
            "elements_with_images": elements_with_images,
            "pages_with_content": sorted(page_counts.keys()),
            "content_types": list(
                set(
                    elem.get("metadata", {}).get("content_type", "unknown")
                    for elem in elements
                )
            ),
        }

    def filter_visualization_elements(
        self,
        element_types: List[str] | None = None,
        pages: List[int] | None = None,
        has_image: bool | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter visualization elements by various criteria.

        Args:
            element_types: List of element types to include (e.g., ["text", "table"])
            pages: List of page numbers to include
            has_image: If True, only elements with images; if False, only without images

        Returns:
            Filtered list of visualization elements
        """
        elements = self.get_visualization_elements()
        if not elements:
            return []

        filtered = elements

        # Filter by element type
        if element_types:
            filtered = [e for e in filtered if e.get("type") in element_types]

        # Filter by page
        if pages:
            filtered = [e for e in filtered if e.get("page") in pages]

        # Filter by image presence
        if has_image is not None:
            filtered = [
                e
                for e in filtered
                if e.get("metadata", {}).get("has_image", False) == has_image
            ]

        return filtered


# Document Parser Provider Interface
class DocumentParser(Protocol):
    async def parse(
        self,
        file_path: str | Any,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> ParseResult:
        """
        Extract content from a document file (PDF, DOCX, etc.).
        Different implementations (vendors) will utilize distinct libraries or services to achieve this.
        Returns a ParseResult object containing structured content.

        Args:
            file_path: Path to the document file or file-like object.
            progress_callback: Optional callback for progress updates during parsing.
            **kwargs: Parser-specific options.
        """
        if isinstance(file_path, str):
            validate_file_exists(file_path)
        result = await self._parse_impl(
            file_path, progress_callback=progress_callback, **kwargs
        )

        if isinstance(self, SegmentedTextResult) and result.text_segments is not None:
            full_text = "\n\n".join([segment.text for segment in result.text_segments])
            result.full_text = full_text

        return result

    @classmethod
    def get_capabilities(cls) -> set[type[Any]]:
        capability_names = {
            "TextParsing",
            "FigureParsing",
            "FullTextResult",
            "SegmentedTextResult",
            "LocalParsing",
            "RemoteParsing",
        }
        return {base for base in cls.__mro__ if base.__name__ in capability_names}

    async def _parse_impl(self, file_path: str, **kwargs: Any) -> ParseResult: ...


def validate_file_exists(file_path: str) -> None:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file to validate

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")


# Office document format mapping: Open XML extension -> (new format, legacy format, document type)
OFFICE_FORMAT_MAP: Dict[str, tuple[str, str, str]] = {
    ".docx": ("docx", "doc", "Word"),
    ".xlsx": ("xlsx", "xls", "Excel"),
    ".pptx": ("pptx", "ppt", "PowerPoint"),
}

# Office document error message templates (for legacy vs Open XML mismatch)
OFFICE_ERROR_MESSAGES: Dict[str, Dict[str, str]] = {
    "docx": {
        "old_format": "legacy .doc (OLE2) format",
        "new_format": "Open XML .docx file (Office 2007+)",
        "conversion_hint": "Open the file in Microsoft Word and save it as a .docx file",
    },
    "xlsx": {
        "old_format": "legacy .xls (OLE2) format",
        "new_format": "Open XML .xlsx file (Office 2007+)",
        "conversion_hint": "Open the file in Microsoft Excel and save it as an .xlsx file",
    },
    "pptx": {
        "old_format": "legacy .ppt (OLE2) format",
        "new_format": "Open XML .pptx file (Office 2007+)",
        "conversion_hint": "Open the file in Microsoft PowerPoint and save it as a .pptx file",
    },
}


def detect_file_format(file_path: str) -> str:
    """
    Detect the actual file format by reading file header (magic bytes).

    Args:
        file_path: Path to the file to detect

    Returns:
        Detected file format, such as 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt', 'pdf', etc.

    Raises:
        ValueError: If file format cannot be detected
    """
    with open(file_path, "rb") as f:
        header = f.read(8)

    # Open XML formats (.docx, .xlsx, .pptx) are ZIP containers.
    # ZIP signatures: PK\x03\x04, PK\x05\x06 or PK\x07\x08
    if header[:2] == b"PK":
        try:
            # Try opening as ZIP and inspect internal structure
            with zipfile.ZipFile(file_path, "r") as zip_file:
                file_list = zip_file.namelist()
                # Open XML Word document contains [Content_Types].xml and word/document.xml
                if "[Content_Types].xml" in file_list:
                    if "word/document.xml" in file_list:
                        return "docx"
                    elif "xl/workbook.xml" in file_list:
                        return "xlsx"
                    elif "ppt/presentation.xml" in file_list:
                        return "pptx"
        except zipfile.BadZipFile:
            pass

    # Legacy Microsoft Office formats (OLE2)
    # OLE2 magic bytes: D0 CF 11 E0 A1 B1 1A E1
    if header[:8] == bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]):
        # Need further inspection to distinguish .doc, .xls or .ppt.
        # We simplify by using the extension as a hint.
        ext = Path(file_path).suffix.lower()
        if ext == ".docx":
            return "doc"  # Actually legacy .doc format, but extension is .docx
        elif ext in [".xlsx", ".xls"]:
            return "xls"
        elif ext in [".pptx", ".ppt"]:
            return "ppt"
        return "doc"  # Default to Word document

    # PDF magic bytes: %PDF
    if header[:4] == b"%PDF":
        return "pdf"

    # If detection fails, fall back to extension-based format
    ext = Path(file_path).suffix.lower()
    return ext.lstrip(".") if ext else "unknown"


def validate_office_file_format(
    file_path: str,
    expected_ext: str,
    strict: bool = True,
    parser_name: str = "parser",
) -> None:
    """
    Validate that Office document file extension matches the actual file format.

    This function detects mismatches between file extension and actual format,
    particularly when the extension indicates Open XML format (.docx, .xlsx, .pptx)
    but the actual file is in legacy OLE2 format (.doc, .xls, .ppt).

    Args:
        file_path: Path to the file to validate
        expected_ext: Expected file extension (e.g., ".docx", ".xlsx", ".pptx")
        strict: Whether to strictly validate. If True, raises exception on mismatch;
                if False, only logs a warning
        parser_name: Parser name for error messages

    Raises:
        ValueError: If format mismatch and strict=True

    Example:
        >>> validate_office_file_format("file.docx", ".docx")
        >>> # If file.docx is actually .doc format, raises ValueError
    """
    expected_ext = expected_ext.lower()

    # Only validate Office documents
    if expected_ext not in OFFICE_FORMAT_MAP:
        return  # Not an Office document, skip validation

    new_format, old_format, doc_type = OFFICE_FORMAT_MAP[expected_ext]
    actual_format = detect_file_format(file_path)

    # If we detect a legacy OLE2 format while the extension claims Open XML
    if actual_format == old_format:
        error_info = OFFICE_ERROR_MESSAGES[new_format]
        message = (
            f"File '{file_path}' has extension {expected_ext}, "
            f"but the actual format is {error_info['old_format']}.\n"
            f"The {parser_name} parser only supports {error_info['new_format']}.\n"
            f"Please convert the file to a real {expected_ext} file, "
            f"or use a different parser for legacy {old_format} files.\n"
            f"Hint: {error_info['conversion_hint']}."
        )

        if strict:
            raise ValueError(message)
        else:
            logger.warning(message)

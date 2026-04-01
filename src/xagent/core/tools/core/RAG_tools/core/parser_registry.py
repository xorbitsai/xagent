"""Parser registry for managing file type to parser mappings."""

from typing import Dict, List, Set

from ....core.document_parser import document_parser_registry

# File extension to supported parser methods mapping.
# Only parsers actually registered in document_parser_registry are listed.
# This ensures type-based parse method consistency when allow_mixed_parse_methods=False.
# Registered parsers: pypdf, pdfplumber, unstructured, pymupdf, deepdoc.
PARSER_COMPATIBILITY: Dict[str, List[str]] = {
    # Documents (only registered parsers)
    ".pdf": ["deepdoc", "pymupdf", "pdfplumber", "unstructured", "pypdf"],
    ".docx": ["deepdoc", "unstructured"],
    ".doc": ["unstructured"],
    ".pptx": ["unstructured"],
    ".ppt": ["unstructured"],
    # Text/Markdown (unstructured supports .txt, .md, .json via basic.py)
    ".txt": ["unstructured"],
    ".md": ["unstructured"],
    ".json": ["unstructured"],
    # Unstructured can handle other types via partition auto; list only where explicitly used
    ".xlsx": ["unstructured"],
    ".xls": ["unstructured"],
    ".rst": [],
    ".py": [],
    ".js": [],
    ".ts": [],
    ".java": [],
    ".cpp": [],
    ".c": [],
    ".go": [],
    ".rs": [],
    ".php": [],
    ".rb": [],
    ".sh": [],
    ".sql": [],
    ".html": ["deepdoc", "unstructured"],
    ".xml": [],
    ".yaml": [],
    ".yml": [],
    ".csv": [],
    ".jpg": [],
    ".jpeg": [],
    ".png": [],
    ".gif": [],
    ".bmp": [],
    ".tiff": [],
    ".webp": [],
}


def _normalize_extension(file_extension: str) -> str:
    """Normalize file extension to canonical form with leading dot and lowercase."""
    if not file_extension.startswith("."):
        file_extension = "." + file_extension
    return file_extension.lower()


def _build_dynamic_compatibility() -> Dict[str, List[str]]:
    """Build dynamic extension → parser mapping from registered parsers."""
    mapping: Dict[str, List[str]] = {}

    for parser_name, parser_class in document_parser_registry.parsers().items():
        supported = getattr(parser_class, "supported_extensions", None)
        if not supported:
            continue
        for ext in supported:
            norm_ext = _normalize_extension(ext)
            if norm_ext not in mapping:
                mapping[norm_ext] = []
            if parser_name not in mapping[norm_ext]:
                mapping[norm_ext].append(parser_name)

    return mapping


# Built at import time so no lock is needed; parser registry is populated at import.
_DYNAMIC_COMPATIBILITY: Dict[str, List[str]] = _build_dynamic_compatibility()


def get_supported_parsers(file_extension: str) -> List[str]:
    """Get supported parser methods for a file extension.

    Args:
        file_extension: File extension (with or without leading dot)

    Returns:
        List of supported parser method names
    """
    norm_ext = _normalize_extension(file_extension)

    # Merge dynamic (from registry supported_extensions) and static mapping so that
    # register_parser_support() remains effective for all extensions.
    dynamic_parsers = _DYNAMIC_COMPATIBILITY.get(norm_ext, [])
    static_parsers = PARSER_COMPATIBILITY.get(norm_ext, [])
    merged = list(dict.fromkeys(dynamic_parsers + static_parsers))
    return merged if merged else []


def validate_parser_compatibility(
    file_extension: str, parser_method: str, allow_mixed: bool = False
) -> bool:
    """Validate if a parser method is compatible with a file type.

    Args:
        file_extension: File extension to check
        parser_method: Parser method to validate
        allow_mixed: If True, allow any parser method

    Returns:
        True if compatible, False otherwise
    """
    if allow_mixed:
        return True

    # Only "default" is allowed without being in the registry; others must exist
    if (
        parser_method != "default"
        and parser_method not in document_parser_registry.parsers()
    ):
        return False

    supported_parsers = get_supported_parsers(file_extension)
    return parser_method in supported_parsers


def get_all_supported_extensions() -> Set[str]:
    """Get all supported file extensions (static and dynamic)."""
    return set(PARSER_COMPATIBILITY.keys()) | set(_DYNAMIC_COMPATIBILITY.keys())


def register_parser_support(file_extension: str, parser_method: str) -> None:
    """Register a new parser method for a file extension.

    This is used when adding new parsers to the system.

    Args:
        file_extension: File extension (with leading dot)
        parser_method: Parser method name to add
    """
    if not file_extension.startswith("."):
        file_extension = "." + file_extension

    file_extension = file_extension.lower()

    if file_extension not in PARSER_COMPATIBILITY:
        PARSER_COMPATIBILITY[file_extension] = []

    if parser_method not in PARSER_COMPATIBILITY[file_extension]:
        PARSER_COMPATIBILITY[file_extension].append(parser_method)

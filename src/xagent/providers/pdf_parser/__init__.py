"""
Pdf parser providers module.

This module provides various pdf parser backends
"""

from typing import Type

from xagent.providers.pdf_parser.base import (
    DocumentParser,
    FigureParsing,
    FullTextResult,
    LocalParsing,
    ParsedFigures,
    ParsedTextSegment,
    ParseResult,
    RemoteParsing,
    SegmentedTextResult,
    TextParsing,
)

from .basic import PdfPlumberParser, PyMuPdfParser, PyPdfParser, UnstructuredParser

DeepDocParser: Type | None
try:
    from .deepdoc import DeepDocParser
except ImportError:
    DeepDocParser = None

__all__ = [
    "ParseResult",
    "FigureParsing",
    "DeepDocParser",  # Will be None if deepdoc is not installed
    "PyPdfParser",
    "PdfPlumberParser",
    "UnstructuredParser",
    "PyMuPdfParser",
    "DocumentParser",
    "TextParsing",
    "FullTextResult",
    "SegmentedTextResult",
    "LocalParsing",
    "RemoteParsing",
    "ParsedTextSegment",
    "ParsedFigures",
]

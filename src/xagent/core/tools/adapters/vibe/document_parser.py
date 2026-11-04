import asyncio
import logging
from typing import Any, Mapping

from pydantic import BaseModel

from .....core.workspace import TaskWorkspace
from .....providers.pdf_parser import ParseResult
from ...core.document_parser import (
    DocumentParseArgs,
    DocumentParseWithOutputArgs,
    DocumentParseWithOutputResult,
    parse_document,
    parse_document_with_output,
)
from .base import AbstractBaseTool, ToolVisibility

logger = logging.getLogger(__name__)


class DocumentParseTool(AbstractBaseTool):
    def __init__(self, workspace: TaskWorkspace | None = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "document_parse"

    @property
    def description(self) -> str:
        return "Parse document files to extract text content, figures, images, and tables using various parsing engines. Supports both local and remote parsing methods with configurable output formats including full text or segmented results."

    @property
    def tags(self) -> list[str]:
        return ["document", "parse", "extraction"]

    def args_type(self) -> type[BaseModel]:
        return DocumentParseArgs

    def return_type(self) -> type[BaseModel]:
        return ParseResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return asyncio.run(self.run_json_async(args))

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        tool_args = DocumentParseArgs.model_validate(args)
        if self.workspace:
            try:
                tool_args.file_path = str(
                    self.workspace.resolve_path_with_search(tool_args.file_path)
                )
            except Exception as e:
                logger.warning(
                    f"Failed to resolve output path through workspace: {e}, using non workspace path"
                )
        return await parse_document(tool_args)


# This alternative tool will write the result to a file
class DocumentParseWithOutputTool(AbstractBaseTool):
    def __init__(self, workspace: TaskWorkspace | None = None) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "document_parse"

    @property
    def description(self) -> str:
        return (
            "Parse document files to extract text content, figures, images, and tables "
            "using various parsing engines. Supports both local and remote parsing methods "
            "with configurable output formats: plain text (.txt), markdown (.md), "
            "or structured JSON (.json)."
        )

    @property
    def tags(self) -> list[str]:
        return ["document", "parse", "extraction"]

    def args_type(self) -> type[BaseModel]:
        return DocumentParseWithOutputArgs

    def return_type(self) -> type[BaseModel]:
        return DocumentParseWithOutputResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return asyncio.run(self.run_json_async(args))

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        tool_args = DocumentParseWithOutputArgs.model_validate(args)
        if self.workspace:
            try:
                tool_args.file_path = str(
                    self.workspace.resolve_path_with_search(tool_args.file_path)
                )
                tool_args.output_path = str(
                    self.workspace.resolve_path(tool_args.output_path)
                )
            except Exception as e:
                logger.warning(
                    f"Failed to resolve output path through workspace: {e}, using non workspace path"
                )
        return await parse_document_with_output(tool_args)

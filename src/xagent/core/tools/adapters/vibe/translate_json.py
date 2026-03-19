"""
JSON Translation Tool for xagent
Framework adapter for translate_json functionality
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from ....workspace import TaskWorkspace
from ...core.translate_json_tool import TranslateJSONToolCore
from .base import AbstractBaseTool, ToolCategory
from .function import FunctionTool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TranslateJsonTool(AbstractBaseTool):
    """Framework wrapper for JSON translation tool"""

    category = ToolCategory.BASIC

    def __init__(
        self, workspace: Optional[TaskWorkspace] = None, llm: Optional[Any] = None
    ):
        self._workspace = workspace
        self._llm = llm
        self._core = TranslateJSONToolCore(llm=llm, workspace=workspace)

    @property
    def name(self) -> str:
        return "translate_json"

    @property
    def description(self) -> str:
        return """Translate specific fields in JSON structure using LLM.

Supports nested structures and batch translation for efficiency.

Language codes:
- 'zh': Chinese (Mandarin), 'en': English, 'yue': Cantonese, 'ja': Japanese, 'ko': Korean, etc.

Parameters:
- json_data (optional): JSON string or dict to translate. Either json_data or file_id must be provided.
- file_id (optional): File ID to read JSON data from. Either json_data or file_id must be provided.
- target_fields (required): List of field paths to translate, e.g., ["segments.text", "title"]
- output_field (optional): Field name for translated text. Default: "translated_text"
- target_lang (optional): Target language code. Default: "en"
- source_lang (optional): Source language code. Auto-detect if not specified
- batch_size (optional): Number of texts to translate per batch (1-50, default: 10). Larger batches maintain better context but may be slower for long texts
- instructions (optional): Additional translation instructions for style, terminology, or context (e.g., "Use formal tone", "Keep technical terms in English", "Preserve formatting")

Returns:
Dictionary with translation result containing:
- success (bool): Whether translation succeeded
- result (str): Translated JSON string with translations added to specified fields
- error (str): Error message if failed
- fields_translated (int): Number of fields translated
- target_lang (str): Target language used
- file_id (str): File ID for accessing the translation JSON file
- translation_path (str): Path to saved translation JSON file
- saved_to_workspace (bool): Whether the translation was saved to workspace

Examples:
1. Direct JSON translation:
   translate_json(json_data='{"text": "你好"}', target_fields=["text"], target_lang="en")
   Returns: {"text": "你好", "translated_text": "Hello"}

2. Translation from file (recommended):
   translate_json(file_id='40f4ec64-0367-44e7-9161-b8fdb3f4eabb', target_fields=["text"], target_lang="en")
   Automatically reads JSON from file and translates it

3. Nested structure from file:
   translate_json(file_id='abc123', target_fields=["segments.text"], target_lang="en")
   Returns: {"segments": [{"text": "测试", "translated_text": "Test"}]}

4. Multiple fields:
   translate_json(file_id='file123', target_fields=["title", "content"], target_lang="en")
   Returns: {"title": "标题", "title_translated_text": "Title", ...}

Note: Translation is done in batches for efficiency. All texts are sent to LLM together.
Translation results are automatically saved to workspace when available.
Using file_id parameter is recommended for workflows with file chaining.
This tool automatically handles file reading when file_id is provided.
"""

    @property
    def tags(self) -> list[str]:
        return ["json", "translate", "llm"]

    def args_type(self) -> type[Any]:
        from pydantic import BaseModel, Field

        class TranslateJsonArgs(BaseModel):
            json_data: Optional[str] = Field(
                default=None,
                description="JSON string to translate. Either json_data or file_id must be provided.",
            )
            file_id: Optional[str] = Field(
                default=None,
                description="File ID to read JSON data from. Either json_data or file_id must be provided.",
            )
            target_fields: List[str] = Field(
                description="List of field paths to translate (e.g., ['segments.text', 'title'])"
            )
            output_field: str = Field(
                default="translated_text",
                description="Field name for translated text",
            )
            target_lang: str = Field(default="en", description="Target language code")
            source_lang: Optional[str] = Field(
                default=None, description="Source language code (auto-detect if None)"
            )
            batch_size: int = Field(
                default=10,
                ge=1,
                le=50,
                description="Number of texts to translate per batch (1-50, default: 10). Larger batches may be slower but can maintain better context.",
            )
            instructions: Optional[str] = Field(
                default=None,
                description="Additional translation instructions (e.g., 'Use formal tone', 'Keep technical terms in English', 'Preserve formatting')",
            )

        return TranslateJsonArgs

    def return_type(self) -> type[Any]:
        from pydantic import BaseModel, Field

        class TranslateJsonResult(BaseModel):
            success: bool = Field(description="Whether translation succeeded")
            result: str = Field(description="Translated JSON string")
            error: Optional[str] = Field(
                default=None, description="Error message if failed"
            )
            fields_translated: int = Field(description="Number of fields translated")
            target_lang: str = Field(description="Target language used")

        return TranslateJsonResult

    def _parse_and_execute(
        self, args: Mapping[str, Any]
    ) -> tuple[
        str | Dict[str, Any], List[str], str, str, Optional[str], int, Optional[str]
    ]:
        """
        Parse and validate arguments for JSON translation.

        Common logic extracted from sync and async methods.

        Args:
            args: Input arguments mapping

        Returns:
            Tuple of (json_data, target_fields, output_field, target_lang, source_lang, batch_size, instructions)

        Raises:
            ValueError: If validation fails
            AssertionError: If LLM is not available
        """
        # Assert LLM is available for execution
        assert self._llm is not None, "translate_json tool requires an LLM to function"

        # Parse arguments
        json_data = args.get("json_data")
        file_id = args.get("file_id")
        target_fields = args.get("target_fields")
        output_field = args.get("output_field", "translated_text")
        target_lang = args.get("target_lang", "en")
        source_lang = args.get("source_lang")
        batch_size = args.get("batch_size", 10)
        instructions = args.get("instructions")

        # Validate required arguments - either json_data or file_id must be provided
        if json_data is None and file_id is None:
            raise ValueError("Either json_data or file_id must be provided")
        if target_fields is None:
            raise ValueError("target_fields is required")

        # If file_id is provided, read JSON data from workspace
        if file_id is not None:
            if not self._workspace:
                raise ValueError("Workspace is required when using file_id parameter")
            try:
                # Get file path from file_id
                file_path = self._workspace.resolve_file_id(file_id)
                if not file_path or not file_path.exists():
                    raise ValueError(f"File not found: {file_path}")

                # Read JSON content from file
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data = f.read()

                logger.info(f"Read JSON data from file: {file_path}")

            except Exception as e:
                raise ValueError(f"Failed to read file {file_id}: {e}")

        # Type narrowing for mypy
        json_data_typed: str | Dict[str, Any] = (
            json_data if isinstance(json_data, (str, dict)) else str(json_data)
        )
        target_fields_typed: List[str] = (
            target_fields if isinstance(target_fields, list) else [target_fields]
        )

        return (
            json_data_typed,
            target_fields_typed,
            output_field,
            target_lang,
            source_lang,
            batch_size,
            instructions,
        )

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Execute translation synchronously"""
        import asyncio

        try:
            # Parse and validate arguments
            (
                json_data_typed,
                target_fields_typed,
                output_field,
                target_lang,
                source_lang,
                batch_size,
                instructions,
            ) = self._parse_and_execute(args)

            # Run async translation
            result = asyncio.run(
                self._core.translate_json(
                    json_data=json_data_typed,
                    target_fields=target_fields_typed,
                    output_field=output_field,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    batch_size=batch_size,
                    instructions=instructions,
                )
            )

            return result

        except AssertionError:
            # Re-raise assertion errors (e.g., LLM not available)
            raise
        except Exception as e:
            logger.error(f"JSON translation failed: {e}")
            return {
                "success": False,
                "result": "",
                "error": str(e),
                "fields_translated": 0,
                "target_lang": args.get("target_lang", "en"),
                "file_id": None,
                "translation_path": None,
                "saved_to_workspace": False,
            }

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute translation asynchronously"""
        try:
            # Parse and validate arguments
            (
                json_data_typed,
                target_fields_typed,
                output_field,
                target_lang,
                source_lang,
                batch_size,
                instructions,
            ) = self._parse_and_execute(args)

            # Direct async call
            result = await self._core.translate_json(
                json_data=json_data_typed,
                target_fields=target_fields_typed,
                output_field=output_field,
                target_lang=target_lang,
                source_lang=source_lang,
                batch_size=batch_size,
                instructions=instructions,
            )

            return result

        except AssertionError:
            # Re-raise assertion errors (e.g., LLM not available)
            raise
        except Exception as e:
            logger.error(f"JSON translation failed: {e}")
            return {
                "success": False,
                "result": "",
                "error": str(e),
                "fields_translated": 0,
                "target_lang": args.get("target_lang", "en"),
                "file_id": None,
                "translation_path": None,
                "saved_to_workspace": False,
            }


def get_translate_json_tool(info: Optional[Dict[str, Any]] = None) -> FunctionTool:
    """
    Create a translate_json tool with workspace and LLM binding.

    Args:
        info: Dictionary containing workspace and LLM instances

    Returns:
        A translate_json tool bound to the specified workspace and LLM
    """
    # Extract workspace from info if provided
    workspace = None
    if info and "workspace" in info:
        workspace = (
            info["workspace"] if isinstance(info["workspace"], TaskWorkspace) else None
        )

    # Extract llm from info if provided
    llm = None
    if info and "llm" in info:
        llm = info["llm"]

    # Create tool with LLM
    tool = TranslateJsonTool(workspace=workspace, llm=llm)

    # Wrap as FunctionTool
    def translate_json_sync(
        json_data: str,
        target_fields: List[str],
        output_field: str = "translated_text",
        target_lang: str = "en",
        source_lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate JSON fields using LLM"""
        result: Any = tool.run_json_sync(
            {
                "json_data": json_data,
                "target_fields": target_fields,
                "output_field": output_field,
                "target_lang": target_lang,
                "source_lang": source_lang,
            }
        )
        # Ensure we return a dict
        if isinstance(result, dict):
            return result
        return {
            "success": False,
            "result": str(result),
            "error": None,
            "fields_translated": 0,
            "target_lang": target_lang,
        }

    return FunctionTool(translate_json_sync, description=tool.description)


# Register tool creator for auto-discovery
from .factory import register_tool  # noqa: E402

if TYPE_CHECKING:
    from .config import BaseToolConfig


@register_tool
async def create_translate_json_tool(config: "BaseToolConfig") -> List[Any]:
    """Create translate_json tool with LLM and workspace from configuration."""
    llm = config.get_llm()

    # Get workspace from config for file_id support
    workspace = None
    workspace_config = config.get_workspace_config()
    if workspace_config:
        from .factory import ToolFactory

        workspace = ToolFactory._create_workspace(workspace_config)

    try:
        # Create tool with LLM and workspace
        # Tool will still appear in tool list even without LLM
        tool_instance = TranslateJsonTool(llm=llm, workspace=workspace)
        return [tool_instance]

    except Exception as e:
        logger.warning(f"Failed to create translate_json tool: {e}")
        return []

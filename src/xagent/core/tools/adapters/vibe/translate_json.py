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
- json_data (required): JSON string or dict to translate
- target_fields (required): List of field paths to translate, e.g., ["segments.text", "title"]
- output_field (optional): Field name for translated text. Default: "translated_text"
- target_lang (optional): Target language code. Default: "en"
- source_lang (optional): Source language code. Auto-detect if not specified

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
1. Simple translation:
   translate_json('{"text": "你好"}', ["text"], target_lang="en")
   Returns: {"text": "你好", "translated_text": "Hello"}

2. Nested structure:
   translate_json('{"segments": [{"text": "测试"}]}', ["segments.text"], target_lang="en")
   Returns: {"segments": [{"text": "测试", "translated_text": "Test"}]}

3. Multiple fields:
   translate_json('{"title": "标题", "content": "内容"}', ["title", "content"], target_lang="en")
   Returns: {"title": "标题", "translated_title": "Title", "content": "内容", "translated_content": "Content"}

Note: Translation is done in batches for efficiency. All texts are sent to LLM together.
Translation results are automatically saved to workspace when available.
"""

    @property
    def tags(self) -> list[str]:
        return ["json", "translate", "llm"]

    def args_type(self) -> type[Any]:
        from pydantic import BaseModel, Field

        class TranslateJsonArgs(BaseModel):
            json_data: str = Field(description="JSON string or dict to translate")
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

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Execute translation synchronously"""
        import asyncio

        # Assert LLM is available for execution
        assert self._llm is not None, "translate_json tool requires an LLM to function"

        try:
            # Parse arguments
            json_data = args.get("json_data")
            target_fields = args.get("target_fields")
            output_field = args.get("output_field", "translated_text")
            target_lang = args.get("target_lang", "en")
            source_lang = args.get("source_lang")

            # Validate required arguments
            if json_data is None:
                raise ValueError("json_data is required")
            if target_fields is None:
                raise ValueError("target_fields is required")

            # Type narrowing for mypy
            json_data_typed: str | Dict[str, Any] = (
                json_data if isinstance(json_data, (str, dict)) else str(json_data)
            )
            target_fields_typed: List[str] = (
                target_fields if isinstance(target_fields, list) else [target_fields]
            )

            # Run async translation
            result = asyncio.run(
                self._core.translate_json(
                    json_data=json_data_typed,
                    target_fields=target_fields_typed,
                    output_field=output_field,
                    target_lang=target_lang,
                    source_lang=source_lang,
                )
            )

            return result

        except Exception as e:
            logger.error(f"JSON translation failed: {e}")
            return {
                "success": False,
                "result": "",
                "error": str(e),
                "fields_translated": 0,
                "target_lang": target_lang if "target_lang" in args else "en",
                "file_id": None,
                "translation_path": None,
                "saved_to_workspace": False,
            }

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute translation asynchronously"""
        # Assert LLM is available for execution
        assert self._llm is not None, "translate_json tool requires an LLM to function"

        try:
            # Parse arguments
            json_data = args.get("json_data")
            target_fields = args.get("target_fields")
            output_field = args.get("output_field", "translated_text")
            target_lang = args.get("target_lang", "en")
            source_lang = args.get("source_lang")

            # Validate required arguments
            if json_data is None:
                raise ValueError("json_data is required")
            if target_fields is None:
                raise ValueError("target_fields is required")

            # Type narrowing for mypy
            json_data_typed: str | Dict[str, Any] = (
                json_data if isinstance(json_data, (str, dict)) else str(json_data)
            )
            target_fields_typed: List[str] = (
                target_fields if isinstance(target_fields, list) else [target_fields]
            )

            # Direct async call
            result = await self._core.translate_json(
                json_data=json_data_typed,
                target_fields=target_fields_typed,
                output_field=output_field,
                target_lang=target_lang,
                source_lang=source_lang,
            )

            return result

        except Exception as e:
            logger.error(f"JSON translation failed: {e}")
            return {
                "success": False,
                "result": "",
                "error": str(e),
                "fields_translated": 0,
                "target_lang": target_lang if "target_lang" in args else "en",
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
    """Create translate_json tool with LLM from configuration."""
    llm = config.get_llm()

    try:
        # Create tool with or without LLM
        # Tool will still appear in tool list even without LLM
        tool_instance = TranslateJsonTool(llm=llm)
        return [tool_instance]

    except Exception as e:
        logger.warning(f"Failed to create translate_json tool: {e}")
        return []

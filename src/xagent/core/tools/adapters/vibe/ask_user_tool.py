import json
import logging
from typing import Any, Mapping, Optional, Type, Union

from pydantic import BaseModel, Field

from .....web.tools.config import WebToolConfig
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool

logger = logging.getLogger(__name__)


class InteractionOption(BaseModel):
    label: str = Field(description="Display label for the option")
    value: str = Field(description="Value for the option")
    description: Optional[str] = Field(
        default=None,
        description="Optional subtitle/description for the option (used in action_cards)",
    )
    action_type: Optional[str] = Field(
        default=None,
        description="Specific action trigger for this option. Can be 'upload' (for file uploads), 'input_url' (for website URLs), or 'none'",
    )


class InteractionArg(BaseModel):
    type: str = Field(
        description="Type of interaction: select_one, select_multiple, text_input, file_upload, confirm, number_input, action_cards"
    )
    field: str = Field(description="Field name for the data")
    label: str = Field(description="Display label for the field")
    options: Optional[list[InteractionOption]] = Field(
        default=None, description="Options for select types"
    )
    placeholder: Optional[str] = Field(default=None, description="Placeholder text")
    multiline: Optional[bool] = Field(
        default=False, description="For text_input, whether it should be multiline"
    )
    min: Optional[int] = Field(default=None, description="Min value for number_input")
    max: Optional[int] = Field(default=None, description="Max value for number_input")
    default_value: Union[str, bool, int, float, None] = Field(
        default=None, description="Default value"
    )
    accept: Optional[list[str]] = Field(
        default=None,
        description="Allowed file extensions for file_upload (e.g. ['.csv', '.pdf'])",
    )
    multiple: Optional[bool] = Field(
        default=False, description="Allow multiple files for file_upload"
    )


class AskUserQuestionArgs(BaseModel):
    message: str = Field(description="The message or question to display to the user.")
    interactions: list[InteractionArg] = Field(
        description="A list of input fields/forms you want the user to fill out."
    )


class AskUserQuestionResult(BaseModel):
    status: str = Field(description="Status of the interaction request")
    message: str = Field(description="Response message to the agent")
    response_data: Optional[dict[str, Any]] = Field(
        default=None, description="Data provided by the user (if any)"
    )


class AskUserQuestionTool(AbstractBaseTool):
    """
    Tool for asking the user a question and gathering structured input via a form.
    This works by emitting a specific trace event or message format that the frontend
    recognizes and renders as a ClarificationForm.
    """

    category: ToolCategory = ToolCategory.AGENT

    def __init__(
        self,
    ) -> None:
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "ask_user_question"

    @property
    def description(self) -> str:
        return (
            "Ask the user a question and provide a structured form for them to fill out. "
            "Use this when you need clarification, specific information, or a decision from the user "
            "before proceeding with a task. "
            "For example, you can ask if they want to create a knowledge base (use 'action_cards' for Import/Upload), or select from a list (use 'select_one')."
        )

    def args_type(self) -> Type[BaseModel]:
        return AskUserQuestionArgs

    def return_type(self) -> Type[BaseModel]:
        return AskUserQuestionResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        # In the context of the Agent Builder chat (ReAct), we can just return a structured response
        # that the websocket handler or frontend will intercept.
        # However, since the LLM executes this tool and expects an immediate response,
        # but the user response is asynchronous, this is tricky in a standard tool call without pausing execution.

        # A simpler approach for the ReAct builder chat is to return a special instruction that causes the
        # agent to format its final response as a clarification form.

        message = args.get("message", "")
        interactions = args.get("interactions", [])

        # Return instructions to the LLM to output the specific JSON structure required by the frontend
        return AskUserQuestionResult(
            status="paused",
            message=(
                "To ask the user this question, you MUST end your turn and output EXACTLY the following JSON block "
                "as your final response (with no other text around it):\n\n"
                "```json\n"
                "{\n"
                f'  "type": "chat",\n'
                f'  "chat": {{\n'
                f'    "message": {json.dumps(message)},\n'
                f'    "interactions": {json.dumps(interactions)}\n'
                f"  }}\n"
                "}\n"
                "```\n\n"
                "Output ONLY the JSON block."
            ),
        ).model_dump()


@register_tool
async def create_ask_user_tool(config: WebToolConfig) -> list[AbstractBaseTool]:
    """Create ask user tool."""
    return [AskUserQuestionTool()]

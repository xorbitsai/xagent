"""Fixed client-visible fallbacks for incidental server failures."""

from ...core.tools.adapters.vibe.config import RequiredMCPUnavailableError

CLIENT_SAFE_VALIDATION_ERROR = "The message could not be processed. Please try again."

# Task audiences did not necessarily initiate the failing operation, so a
# task-level failure uses neutral wording instead of the validation fallback.
CLIENT_SAFE_TASK_FAILURE = "Task execution failed."


def required_mcp_unavailable_client_message(
    error: BaseException,
    *,
    fallback: str = CLIENT_SAFE_VALIDATION_ERROR,
) -> str:
    """Adapt the curated required-MCP failure without opening a generic escape.

    The runtime check keeps this boundary fail-closed even if a future caller
    passes an incidental exception despite the function's specific name.
    """

    if not isinstance(error, RequiredMCPUnavailableError):
        return fallback
    message = str(error)
    if message.strip():
        return message
    return fallback

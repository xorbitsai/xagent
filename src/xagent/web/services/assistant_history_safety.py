"""Client-safety policy for persisted assistant history."""

from .client_error_messages import CLIENT_SAFE_TASK_FAILURE

LEGACY_ASSISTANT_RESPONSE_MESSAGE_TYPE = "chat_response"
LEGACY_UNTRUSTED_ASSISTANT_MESSAGE_TYPE = "assistant_message"
ASSISTANT_RESPONSE_MESSAGE_TYPE = "assistant_response"
TASK_FAILURE_MESSAGE_TYPE = "task_failure"
CLIENT_SAFE_FAILURE_MESSAGE_TYPE = "client_safe_failure"
KNOWN_SAFE_ASSISTANT_MESSAGE_TYPES = frozenset(
    {
        ASSISTANT_RESPONSE_MESSAGE_TYPE,
        CLIENT_SAFE_FAILURE_MESSAGE_TYPE,
        "assistant",
        "question",
        "question_superseded",
    }
)


def safe_str(value: object | None) -> str:
    """Convert a nullable value to text without producing literal ``None``."""

    return "" if value is None else str(value)


def assistant_history_values_for_persistence(
    *,
    content: str,
    message_type: str,
    is_failure: bool,
) -> tuple[str, str]:
    """Choose durable client content and provenance for one assistant result."""

    if is_failure:
        return CLIENT_SAFE_TASK_FAILURE, TASK_FAILURE_MESSAGE_TYPE
    return content, message_type


def client_safe_assistant_history_content(
    *,
    content: str,
    message_type: str,
) -> str:
    """Return client- and model-safe content for one persisted assistant row.

    Before assistant history carried explicit provenance, terminal failures and
    ordinary responses both used ``chat_response``. No durable metadata proves
    which legacy rows are safe, so every pre-cutover ``chat_response`` fails
    closed, including legitimate plain responses. Unknown future message types
    also fail closed until they are explicitly classified as client-visible.
    """

    if message_type not in KNOWN_SAFE_ASSISTANT_MESSAGE_TYPES:
        return CLIENT_SAFE_TASK_FAILURE
    return content


def assistant_history_has_safe_ancillary_payload(message_type: str) -> bool:
    """Return whether an assistant row may expose stored client metadata."""

    return message_type in KNOWN_SAFE_ASSISTANT_MESSAGE_TYPES

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

TOOL_PROTOCOL_ERROR_KEY = "_xagent_tool_protocol_error"


@dataclass(frozen=True)
class ToolProtocolViolation:
    provider: str
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def tool_protocol_error_response(
    violation: ToolProtocolViolation,
    *,
    raw: Any = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "type": "tool_protocol_error",
        "content": "",
        "tool_calls": [],
        TOOL_PROTOCOL_ERROR_KEY: violation.to_dict(),
    }
    if raw is not None:
        response["raw"] = raw
    return response


def get_tool_protocol_error(response: Any) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None
    payload = response.get(TOOL_PROTOCOL_ERROR_KEY)
    return dict(payload) if isinstance(payload, dict) else None

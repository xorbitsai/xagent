"""Streaming data types for LLM chat responses."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ChunkType(Enum):
    """Streaming response chunk types"""

    TOKEN = "token"  # Regular token
    TOOL_CALL = "tool_call"  # Tool call
    USAGE = "usage"  # Token usage statistics
    ERROR = "error"  # Error
    PROTOCOL_ERROR = "protocol_error"  # Provider tool-protocol violation
    END = "end"  # End


@dataclass
class StreamChunk:
    """Streaming response chunk

    Attributes:
        type: Chunk type
        content: Accumulated complete content
        delta: Incremental content (new content added by current chunk)
        tool_calls: Tool call list (only for TOOL_CALL type)
        usage: Token usage statistics (only for USAGE type)
        finish_reason: Finish reason (only for END type)
        raw: Original response object
    """

    type: ChunkType
    content: str = ""
    delta: str = ""
    tool_calls: List[Dict] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    raw: Any = None
    protocol_error: Dict[str, Any] = field(default_factory=dict)

    def is_token(self) -> bool:
        """Check if this is a token type"""
        return self.type == ChunkType.TOKEN

    def is_tool_call(self) -> bool:
        """Check if this is a tool call type"""
        return self.type == ChunkType.TOOL_CALL

    def is_usage(self) -> bool:
        """Check if this is a usage type"""
        return self.type == ChunkType.USAGE

    def is_error(self) -> bool:
        """Check if this is an error type"""
        return self.type == ChunkType.ERROR

    def is_protocol_error(self) -> bool:
        """Check if this is a provider tool-protocol error."""
        return self.type == ChunkType.PROTOCOL_ERROR

    def is_end(self) -> bool:
        """Check if this is an end type"""
        return self.type == ChunkType.END

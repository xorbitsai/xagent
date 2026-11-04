"""
Compact utilities for conversation compaction.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class CompactConfig:
    """Configuration for conversation compaction."""

    enabled: bool = True
    threshold: int = 32000
    fallback_truncate_count: int = 20


@dataclass
class CompactResult:
    """Result of conversation compaction."""

    success: bool
    messages: List[Dict[str, str]]
    original_tokens: int
    final_tokens: int
    tokens_saved: int
    method_used: str


class CompactUtils:
    """Utility functions for conversation compaction."""

    @staticmethod
    def estimate_tokens(messages: List[Dict[str, str]]) -> int:
        """Rough estimation of token count for messages."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4  # Very rough approximation

    @staticmethod
    def truncate_messages(
        messages: List[Dict[str, str]], max_messages: int
    ) -> List[Dict[str, str]]:
        """Truncate messages to keep only the most recent ones."""
        if len(messages) <= max_messages:
            return messages

        # Always keep system message if it exists
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]

        # Keep the most recent non-system messages
        recent_messages = other_messages[-max_messages + len(system_messages) :]

        return system_messages + recent_messages

    @staticmethod
    def format_messages_for_compact(messages: List[Dict[str, str]]) -> str:
        """Format messages as text for compaction."""
        formatted_lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines)

    @staticmethod
    def parse_compact_response(response: str) -> List[Dict[str, str]]:
        """Parse compacted response back to messages format."""
        messages = []
        lines = response.strip().split("\n")

        current_role = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a role line (case insensitive)
            if line.upper().startswith(("USER:", "ASSISTANT:", "SYSTEM:")):
                # Save previous message
                if current_role and current_content:
                    messages.append(
                        {
                            "role": current_role.lower(),
                            "content": "\n".join(current_content),
                        }
                    )

                # Start new message
                parts = line.split(":", 1)
                current_role = parts[0].strip().lower()  # Normalize to lowercase
                current_content = (
                    [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
                )
            else:
                # Continue current message content
                if current_role:
                    current_content.append(line)

        # Save last message
        if current_role and current_content:
            messages.append(
                {"role": current_role.lower(), "content": "\n".join(current_content)}
            )

        # Fallback: if no structured messages found, treat entire response as assistant message
        if not messages and response.strip():
            logger.warning(
                "No properly formatted messages found, treating entire response as assistant message"
            )
            messages.append({"role": "assistant", "content": response.strip()})

        return messages

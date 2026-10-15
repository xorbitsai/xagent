"""Shared model-name rules for tool-choice capabilities."""

from __future__ import annotations

import re

AUTO_ONLY_TOOL_CHOICE_ABILITY = "auto_tool_choice_only"

_AUTO_ONLY_CLAUDE_MODEL = re.compile(
    r"^claude-(?:opus-5-5|sonnet-5-5|fable-5-1|mythos-5-1)(?:[-.:]|$)",
    re.IGNORECASE,
)
_BEDROCK_ANTHROPIC_PREFIX = re.compile(r"^(?:[a-z0-9-]+\.)?anthropic\.", re.IGNORECASE)


def is_auto_only_claude_model(model_name: str) -> bool:
    """Whether a direct or Bedrock Claude identifier rejects forced tools."""

    candidate = model_name.strip().rsplit("/", 1)[-1]
    candidate = _BEDROCK_ANTHROPIC_PREFIX.sub("", candidate, count=1)
    return _AUTO_ONLY_CLAUDE_MODEL.match(candidate) is not None


def supports_forced_tool_choice(abilities: list[str], model_name: str) -> bool:
    """Whether a Claude model/configuration accepts a forced tool choice."""

    return not (
        AUTO_ONLY_TOOL_CHOICE_ABILITY in abilities
        or is_auto_only_claude_model(model_name)
    )

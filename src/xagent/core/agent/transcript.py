"""Helpers for building and normalizing persisted chat transcripts."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..context_ref import CONTEXT_REFS_KEY, normalize_context_references

logger = logging.getLogger(__name__)


def build_assistant_transcript_content(
    content: str | None, interactions: Optional[List[Any]] = None
) -> str:
    """Build assistant transcript text that preserves interactive prompts."""
    content_parts = [str(content)] if content is not None else []

    if interactions:
        interaction_lines: List[str] = []
        for interaction in interactions:
            interaction_type = _get_interaction_attr(interaction, "type")
            options = _get_interaction_attr(interaction, "options") or []
            label = _get_interaction_attr(interaction, "label")
            placeholder = _get_interaction_attr(interaction, "placeholder")
            accept = _get_interaction_attr(interaction, "accept") or []
            multiple = bool(_get_interaction_attr(interaction, "multiple"))
            default = _get_interaction_attr(interaction, "default")
            minimum = _get_interaction_attr(interaction, "min")
            maximum = _get_interaction_attr(interaction, "max")

            if interaction_type == "select_one":
                options_desc = ", ".join(
                    [
                        f"{_safe_option_value(option, 'value')}: {_safe_option_value(option, 'label')}"
                        for option in options
                    ]
                )
                interaction_lines.append(f"- {label or 'Select'}: {options_desc}")
            elif interaction_type == "select_multiple":
                options_desc = ", ".join(
                    [
                        f"{_safe_option_value(option, 'value')}: {_safe_option_value(option, 'label')}"
                        for option in options
                    ]
                )
                interaction_lines.append(
                    f"- {label or 'Select multiple options'}: {options_desc}"
                )
            elif interaction_type == "text_input":
                interaction_lines.append(
                    f"- {label or 'Enter text'}: {placeholder or 'text input'}"
                )
            elif interaction_type == "file_upload":
                accept_desc = (
                    ", ".join(str(item) for item in accept) if accept else "any file"
                )
                multiple_desc = "multiple files allowed" if multiple else "single file"
                interaction_lines.append(
                    f"- {label or 'Upload file'}: {accept_desc} ({multiple_desc})"
                )
            elif interaction_type == "confirm":
                default_desc = "Default: yes" if default else "Default: no"
                interaction_lines.append(f"- {label or 'Confirm'} ({default_desc})")
            elif interaction_type == "number_input":
                range_desc = ""
                if minimum is not None and maximum is not None:
                    range_desc = f" (range: {minimum}-{maximum})"
                interaction_lines.append(f"- {label or 'Enter number'}{range_desc}")

        if interaction_lines:
            content_parts.append("\n\nPlease answer the following questions:")
            content_parts.extend(interaction_lines)

    return "\n".join(content_parts)


def normalize_transcript_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize transcript messages for use by LLM-backed chat patterns."""
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()

        try:
            context_refs = normalize_context_references(
                message.get(CONTEXT_REFS_KEY, message.get("context_refs"))
            )
        except (TypeError, ValueError):
            logger.debug("Filtered invalid transcript context refs", exc_info=True)
            context_refs = ()

        if role not in {"user", "assistant", "system"} or not (content or context_refs):
            logger.debug(
                f"Filtered invalid message: role={role}, content_length={len(content)}"
            )
            continue
        normalized_message: Dict[str, Any] = {"role": role, "content": content}
        if context_refs:
            normalized_message[CONTEXT_REFS_KEY] = [
                reference.durable_dict() for reference in context_refs
            ]
        normalized.append(normalized_message)
    return normalized


def _get_interaction_attr(interaction: Any, key: str) -> Any:
    if isinstance(interaction, dict):
        return interaction.get(key)
    return getattr(interaction, key, None)


def _safe_option_value(option: Any, key: str) -> str:
    if isinstance(option, dict):
        return str(option.get(key, ""))
    return str(getattr(option, key, ""))

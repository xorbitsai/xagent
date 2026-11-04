"""Display-only projection of the interaction appendix in legacy transcripts."""

from typing import Any

from ...core.agent.transcript import build_assistant_transcript_content


def question_content_for_display(content: str, interactions: Any) -> str:
    """Remove only the exact generated appendix when the form carries it.

    Call only for persisted question transcripts, never raw agent messages or
    model context. The canonical text stays intact for replay pairing, drafts
    and continuation. Unknown or malformed controls keep the text fallback.
    """
    if not isinstance(interactions, list) or not interactions:
        return content
    for item in interactions:
        if not isinstance(item, dict) or item.get("type") not in (
            "select_one",
            "select_multiple",
            "text_input",
            "file_upload",
            "confirm",
            "number_input",
        ):
            return content
        if item.get("label") is not None and not isinstance(item["label"], str):
            return content
        if item["type"] in ("select_one", "select_multiple"):
            options = item.get("options")
            if not isinstance(options, list) or not options:
                return content
            if any(
                not isinstance(option, dict)
                or not isinstance(option.get("value"), str)
                or not isinstance(option.get("label"), str)
                for option in options
            ):
                return content
    try:
        appendix = build_assistant_transcript_content("", interactions).rstrip()
    except (TypeError, ValueError, AttributeError):
        return content
    # Persistence strips the complete transcript, including whitespace on the
    # last control line. Match that normalization without changing fallbacks.
    matching_content = content.rstrip()
    if appendix and matching_content.endswith(appendix):
        introduction = matching_content[: -len(appendix)]
        # Empty introductions still need text on clients that use non-empty
        # content to admit a historical message. Preserve that fallback too.
        if introduction.strip():
            return introduction
    return content

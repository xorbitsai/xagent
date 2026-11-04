from __future__ import annotations


def merge_streamed_tool_call_arguments(
    existing: str,
    fragment: str,
    *,
    mode: str | None = None,
) -> str:
    """Merge streamed tool-call arguments without guessing from JSON content.

    Providers use two shapes for streamed tool-call arguments:
    - delta: each chunk contains only the newly-added bytes
    - snapshot: each chunk contains the full accumulated argument string so far

    When the mode is unknown, treat a fragment as a snapshot only if it is
    demonstrably an accumulated prefix extension of the existing value. This
    keeps valid delta fragments such as ``"{hi"`` inside a JSON string from
    replacing the already-accumulated prefix.
    """

    if not existing:
        return fragment
    normalized_mode = (mode or "").strip().lower()
    if normalized_mode == "snapshot":
        return fragment
    if normalized_mode == "delta":
        return existing + fragment
    if _looks_like_argument_snapshot(existing, fragment):
        return fragment
    return existing + fragment


def _looks_like_argument_snapshot(existing: str, fragment: str) -> bool:
    if fragment == existing or fragment.startswith(existing):
        return True

    existing_lstrip = existing.lstrip()
    fragment_lstrip = fragment.lstrip()
    return bool(existing_lstrip and fragment_lstrip.startswith(existing_lstrip))

"""Turn a Linear GraphQL API error response into a human-readable message.

Shared by `tools/mcp/linear.py`'s `_graphql()` (used by every Linear MCP tool
call) and `api/auth.py`'s `_fetch_linear_viewer_identity()` (used once, at
OAuth-connect time) -- both parse the same GraphQL endpoint's error shape,
and had drifted into two independently-maintained copies (the `auth.py` copy
lost the "message is present but empty" fallback) before this module existed.
"""

from __future__ import annotations

from typing import Any


def graphql_errors_message(errors: list[Any]) -> str:
    """Join a GraphQL response's top-level `errors` array into one message.

    Callers only invoke this once `errors` is known non-empty; the "Unknown"
    fallback exists only for the un-narrowed call shape.
    """
    messages = []
    for entry in errors:
        if isinstance(entry, dict) and entry.get("message"):
            messages.append(str(entry["message"]))
        else:
            messages.append(str(entry))
    return "; ".join(messages) if messages else "Unknown Linear API error"


def truncate_error_text(text: str, limit: int = 1000) -> str:
    """Cap an arbitrary (e.g. HTML gateway/WAF) error body before it reaches
    logs, an LLM, or a rendered error page. `limit` is a parameter because
    callers differ: linear.py's tool responses are read by an LLM (1000
    chars), while auth.py's OAuth-callback error page is rendered directly
    in a browser (500 chars)."""
    if len(text) > limit:
        return text[:limit] + "... [truncated]"
    return text

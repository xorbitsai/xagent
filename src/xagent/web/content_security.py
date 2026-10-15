"""Shared classification for browser-active user content."""

ACTIVE_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "image/svg+xml",
        "application/xml",
    }
)


def is_active_content_type(media_type: str | None) -> bool:
    """Return whether a response can execute or interpret user markup."""
    if not media_type:
        return False
    normalized = media_type.lower().split(";", 1)[0].strip()
    return normalized in ACTIVE_CONTENT_TYPES or normalized.endswith("+xml")

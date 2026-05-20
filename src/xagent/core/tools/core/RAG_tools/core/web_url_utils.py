"""Shared URL normalization helpers for web crawling."""

from __future__ import annotations

from typing import Optional
from urllib.parse import ParseResult, urljoin, urlparse, urlunparse


def _build_normalized_netloc(parsed: ParseResult) -> str:
    """Rebuild netloc with a normalized hostname while preserving userinfo/port."""
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("Invalid start_url: URL must include a hostname")

    normalized_host = hostname.lower()
    if ":" in normalized_host:
        normalized_host = f"[{normalized_host}]"

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo = f"{userinfo}:{parsed.password}"
        userinfo = f"{userinfo}@"

    port = ""
    if parsed.port is not None:
        port = f":{parsed.port}"

    return f"{userinfo}{normalized_host}{port}"


def validate_and_normalize_web_url(url: str, *, base_url: Optional[str] = None) -> str:
    """Validate and normalize an HTTP(S) URL for web ingestion/crawling."""
    if not isinstance(url, str):
        raise ValueError("Invalid start_url: URL must be a string")

    normalized_input = url.strip()
    if not normalized_input:
        raise ValueError("Invalid start_url: URL must not be empty")

    candidate = urljoin(base_url, normalized_input) if base_url else normalized_input
    parsed = urlparse(candidate)
    scheme = parsed.scheme.lower()

    if scheme not in {"http", "https"}:
        raise ValueError("Invalid start_url: URL must start with http:// or https://")

    normalized_netloc = _build_normalized_netloc(parsed)

    return urlunparse(
        (
            scheme,
            normalized_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )


def normalize_web_url(url: str, *, base_url: Optional[str] = None) -> Optional[str]:
    """Normalize a web URL, returning ``None`` when the URL is invalid."""
    try:
        return validate_and_normalize_web_url(url, base_url=base_url)
    except ValueError:
        return None

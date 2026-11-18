"""Recognise search-engine result-page URLs so page-fetch tools can refuse them.

Page-fetch tools exist to read a page the agent already has a URL for. Loading a
search engine's result page through them is a search done outside the
``web_search`` tool, so it bypasses whatever that tool applies to queries
(provider choice, redaction, auditing) and writes the query text into any log
that records the URL. Matching is by exact host (or a documented subdomain
pattern) plus path, never by substring, so ordinary pages on the same
companies' sites keep loading.
"""

from __future__ import annotations

import logging
import posixpath
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from ....config import get_refuse_search_result_fetch_enabled

logger = logging.getLogger(__name__)

SEARCH_RESULT_PAGE_REFUSAL = (
    "Search-engine result pages are not fetched with this tool. Use the "
    "web_search tool to search; if it found nothing, tell the user that no "
    "results were found instead of loading a search engine's page directly."
)

# Country-code TLD suffixes such as ``de``, ``co.uk`` or ``com.au``. A label
# chain that merely starts with ``com`` (``com.evil.example``) does not match.
_CC_TLD = r"(?:com|[a-z]{2,3}|(?:co|com|org|net|ac)\.[a-z]{2})"


@dataclass(frozen=True)
class _Rule:
    host: re.Pattern[str]
    # Normalised paths that are a result page. ``None`` means every path on the
    # host (a host that serves nothing but search).
    paths: frozenset[str] | None
    # When set, the page only counts if one of these query keys is present.
    query_keys: frozenset[str] = frozenset()


def _host(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


_RULES: tuple[_Rule, ...] = (
    # Google: google.<tld>/search (images, news and shopping use tbm= on it).
    _Rule(_host(rf"(?:www\.|encrypted\.)?google\.{_CC_TLD}"), frozenset({"/search"})),
    # Bing, including regional and global.bing.com subdomains.
    _Rule(
        _host(r"(?:[a-z0-9-]+\.)?bing\.com"),
        frozenset({"/search", "/images/search", "/videos/search", "/news/search"}),
    ),
    # DuckDuckGo: html./lite. endpoints and the root with a query.
    _Rule(
        _host(r"(?:(?:www|html|lite)\.)?duckduckgo\.com"),
        frozenset({"/html", "/lite"}),
    ),
    _Rule(
        _host(r"(?:www\.)?duckduckgo\.com"),
        frozenset({"/"}),
        frozenset({"q"}),
    ),
    # Yahoo, including country subdomains such as uk.search.yahoo.com.
    _Rule(
        _host(r"(?:[a-z]{2}\.)?search\.yahoo\.(?:com|co\.jp)"), frozenset({"/search"})
    ),
    _Rule(_host(rf"(?:www\.)?(?:yandex\.{_CC_TLD}|ya\.ru)"), frozenset({"/search"})),
    _Rule(_host(r"(?:www\.|m\.)?baidu\.com"), frozenset({"/s", "/baidu"})),
    _Rule(
        _host(r"search\.brave\.com"),
        frozenset({"/search", "/images", "/news", "/videos"}),
    ),
    _Rule(
        _host(r"(?:www\.)?startpage\.com"),
        frozenset({"/do/search", "/do/dsearch", "/sp/search", "/search"}),
    ),
    _Rule(_host(r"(?:www\.)?ecosia\.org"), frozenset({"/search"})),
    _Rule(
        _host(r"(?:www\.|m\.|weixin\.)?sogou\.com"),
        frozenset({"/web", "/sogou", "/weixin"}),
    ),
    _Rule(_host(r"(?:www\.|m\.)?so\.com"), frozenset({"/s"})),
    _Rule(_host(r"(?:m\.)?search\.naver\.com"), None),
)


def _normalise_path(path: str) -> str:
    """Collapse the spellings a server treats as the same path."""

    segments = [segment.split(";", 1)[0] for segment in unquote(path).split("/")]
    cleaned = posixpath.normpath("/" + "/".join(segments))
    return "/" if cleaned in {"", "."} else cleaned.replace("//", "/")


def _request_host(url: str, fallback: str) -> str:
    """The host an HTTP client would actually connect to for ``url``.

    ``httpx`` canonicalises the hostname when it builds a request (Unicode
    full stops such as U+3002 become ``.``, IDNA mapping is applied), so
    matching the raw spelling would miss a URL the client sends to a search
    engine. A URL httpx cannot parse is never sent by it; keep the raw host
    for those so the check stays at least as strict as before.
    """

    try:
        return httpx.URL(url).host or fallback
    except Exception:
        return fallback


def search_result_page_host(url: str) -> str | None:
    """Return the host when ``url`` is a search-engine result page, else None."""

    try:
        parts = urlsplit(url.strip())
        hostname = parts.hostname
    except ValueError:
        return None
    if not hostname or parts.scheme.lower() not in {"http", "https"}:
        return None
    hostname = _request_host(url.strip(), hostname).rstrip(".").lower()
    # Every rule path is lowercase; servers treat /Search like /search.
    path = _normalise_path(parts.path).lower()
    for rule in _RULES:
        if not rule.host.fullmatch(hostname):
            continue
        if rule.paths is not None and path not in rule.paths:
            continue
        if rule.query_keys and not (
            rule.query_keys & set(parse_qs(parts.query, keep_blank_values=True))
        ):
            continue
        return hostname
    return None


def is_search_result_page_url(url: str) -> bool:
    """Whether ``url`` points at a search-engine result page."""

    return search_result_page_host(url) is not None


def search_result_page_refusal(url: str) -> str | None:
    """Return the refusal message when ``url`` must not be fetched, else None.

    Only the host is logged: the path and query carry the search text.
    """

    if not get_refuse_search_result_fetch_enabled():
        return None
    host = search_result_page_host(url)
    if host is None:
        return None
    logger.warning("Refused to fetch a search-engine result page (host=%s)", host)
    return SEARCH_RESULT_PAGE_REFUSAL

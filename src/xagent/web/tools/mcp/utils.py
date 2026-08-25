import json
import os
import re
import urllib.request
from typing import Any
from urllib.parse import quote

from ....config import get_tool_max_output_length


def require_clean_identifier(value: str, field_name: str) -> str:
    """Reject an empty or whitespace-padded id rather than silently fixing it.

    An id copy-pasted or concatenated by a caller with accidental whitespace
    is more likely a bug worth surfacing than a value to repair - repairing
    it would mask the bug and could send a query for a different object.
    Use this for ids that go into a JSON request body; for ids interpolated
    into a URL path, use url_path_id instead - encoding (not just rejecting
    whitespace) is what actually closes path/query injection.
    """
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(
            f"{field_name} must be a non-empty id with no surrounding whitespace"
        )
    return value


def url_path_id(value: str, field_name: str) -> str:
    """Validate then percent-encode an id for safe interpolation into a URL
    path segment.

    Percent-encoding - not a blocklist of "/", "?", "#" - is what actually
    prevents a value like "x?limit=1&foo=/reports/metrics" from escaping
    its intended path segment: any character that could do that gets
    encoded regardless of which one it is, rather than relying on an
    enumeration that could miss one. "." and ".." are the one exception
    that survives encoding unchanged (they're always-unreserved characters
    per RFC 3986, so quote() never touches them), and requests/urllib3
    normalize dot-segments out of the final URL before sending it --
    verified directly: requests.Request("GET",
    ".../sobjects/Account/..").prepare().url collapses to ".../sobjects/",
    a completely different (still valid) endpoint. Rejected explicitly
    since encoding can't close this one off.
    """
    require_clean_identifier(value, field_name)
    if value in (".", ".."):
        raise ValueError(f"{field_name} must not be '.' or '..'")
    return quote(value, safe="")


def success_with_capped_dict(field_name: str, data: Any) -> str:
    """Build a ``{"status": "success", ...}`` payload, trimming a dict
    until it fits the platform's output limit.

    A record/report/metrics response can be a dict keyed by date,
    dimension, or id depending on the endpoint, where most of the
    payload's size typically lives in one or two large nested list/dict
    values while the rest are small scalars (an "offset" or "total" field
    alongside a big "breakdowns" list, or a handful of small standard
    fields alongside one huge Long Text Area value). Dropping whole
    top-level keys to shrink such a dict can discard the entire useful
    payload on the very first step while leaving small, mostly empty
    scalar fields behind -- and there's no cursor to retry with, so that
    data is gone for this call. Phase 1 instead repeatedly finds the
    largest list/dict-valued key and halves *its* contents (recursing one
    level, not further), so small scalar keys survive untouched as long as
    there is a bigger key left to shrink first. Phase 2 is a fallback for
    the residual case -- a dict with no list/dict-valued keys at all (e.g.
    a handful of scalar keys with huge string values) -- and drops whole
    keys, exactly as phase 1 replaces; it's guaranteed to terminate at {}.
    """
    max_output_length = get_tool_max_output_length()
    response = json.dumps(
        {"status": "success", field_name: data, "truncated": False},
        ensure_ascii=False,
    )
    if not isinstance(data, dict) or len(response) <= max_output_length:
        return response

    def _build(payload: dict[str, Any], truncated: bool) -> str:
        return json.dumps(
            {"status": "success", field_name: payload, "truncated": truncated},
            ensure_ascii=False,
        )

    working = dict(data)
    truncated = False
    while len(response) > max_output_length:
        collection_keys = [
            key
            for key, value in working.items()
            if isinstance(value, (list, dict)) and len(value) > 0
        ]
        if not collection_keys:
            break
        target_key = max(
            collection_keys,
            key=lambda key: len(json.dumps(working[key], ensure_ascii=False)),
        )
        target_value = working[target_key]
        if isinstance(target_value, list):
            working[target_key] = target_value[: len(target_value) // 2]
        else:
            sub_keys = list(target_value.keys())
            working[target_key] = {
                sub_key: target_value[sub_key]
                for sub_key in sub_keys[: len(sub_keys) // 2]
            }
        truncated = True
        response = _build(working, truncated)

    keys = list(working.keys())
    while len(response) > max_output_length and keys:
        keys = keys[: len(keys) // 2]
        working = {key: working[key] for key in keys}
        truncated = True
        response = _build(working, truncated)
    return response


def clamp_limit(limit: int, *, max_limit: int) -> int:
    """Clamp a caller-supplied pagination page size to ``[1, max_limit]``.

    An LLM caller can pass 0, a negative number, or an absurdly large value
    for a tool's ``limit`` parameter. Silently clamping (rather than
    raising) keeps a malformed value from producing a permanently-stuck,
    zero-progress page -- 0 or a negative limit always slices to an empty
    page regardless of offset, so a caller mechanically following a
    pagination contract's own has_more/next_offset would retry forever
    with no error to signal why.
    """
    return max(1, min(int(limit), max_limit))


def clamp_offset(offset: int) -> int:
    """Clamp a caller-supplied pagination offset to ``>= 0``.

    Python slicing treats a negative start index as "count from the end",
    so an unclamped negative offset would silently return items from the
    tail of the list instead of erroring or being treated as the first
    page.
    """
    return max(0, int(offset))


def resolve_id_from_url(value: str, pattern: re.Pattern[str]) -> str:
    """Return the id captured by ``pattern`` when ``value`` is a matching URL,
    otherwise the stripped value itself."""
    match = pattern.search(value)
    if match:
        return match.group(1)
    return value.strip()


def setup_proxy_env() -> None:
    """Setup proxy environment variables from system proxies if missing."""
    # Filter out empty proxy vars to prevent httplib2 hangs
    for var in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]:
        if var in os.environ and not os.environ[var]:
            del os.environ[var]

    system_proxies = urllib.request.getproxies()
    if (
        "https" in system_proxies
        and "HTTPS_PROXY" not in os.environ
        and "https_proxy" not in os.environ
    ):
        os.environ["HTTPS_PROXY"] = system_proxies["https"]
    if (
        "http" in system_proxies
        and "HTTP_PROXY" not in os.environ
        and "http_proxy" not in os.environ
    ):
        os.environ["HTTP_PROXY"] = system_proxies["http"]

    # If ALL_PROXY is set, ensure HTTPS_PROXY is also set
    if "ALL_PROXY" in os.environ and "HTTPS_PROXY" not in os.environ:
        os.environ["HTTPS_PROXY"] = os.environ["ALL_PROXY"]

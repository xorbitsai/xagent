"""Trusted ingress and transaction-bound browser entry helpers.

Forwarding headers must be overwritten by the authenticated ingress proxy.
These helpers do not make an untrusted direct-to-origin deployment safe.
"""

from typing import Any
from urllib.parse import urlencode, urlsplit

from fastapi import HTTPException, Request

from ..config import (
    PublicEntry,
    get_frontend_url,
    get_legacy_public_origin,
    get_public_entries,
    public_entries_configured,
)


def request_public_entry(request: Request) -> PublicEntry:
    entries = get_public_entries()
    if not public_entries_configured():
        return entries[0]
    values = request.headers.getlist("x-forwarded-host")
    if not values:
        values = request.headers.getlist("host")
    if len(values) != 1 or any(c in values[0] for c in ",/\\@?#"):
        raise HTTPException(400, "Invalid public entry")
    host = values[0].lower()
    matches = [
        e
        for e in entries
        if urlsplit(e.origin).netloc == host
        or (
            urlsplit(e.origin).scheme == "https"
            and host == urlsplit(e.origin).netloc + ":443"
        )
    ]
    if len(matches) != 1:
        raise HTTPException(400, "Unknown public entry")
    return matches[0]


def entry_url(
    entry: PublicEntry, path: str, params: dict[str, str] | None = None
) -> str:
    if not path.startswith("/") or path.startswith("//") or "\\" in path:
        raise ValueError("Expected a local path")
    origin = entry.origin if public_entries_configured() else get_frontend_url()
    return origin + path + ("?" + urlencode(params) if params else "")


async def google_state_entry(request: Request, google: Any) -> PublicEntry:
    """Authlib framework adapter: read before normal authorize_access_token consumption."""
    states = request.query_params.getlist("state")
    if len(states) != 1 or not states[0]:
        raise ValueError("invalid_callback")
    data = await google.framework.get_state_data(request.session, states[0])
    if not isinstance(data, dict):
        raise ValueError("invalid_state")
    uri = data.get("redirect_uri")
    entries = [e for e in get_public_entries() if e.google_oidc_redirect_uri == uri]
    if len(entries) != 1 or request_public_entry(request) != entries[0]:
        raise ValueError("invalid_entry")
    return entries[0]


def validate_exchange_entry(request: Request, origin: str | None) -> bool:
    """Bind an exchange code to the entry that completed the Google callback.

    Codes without an origin predate explicit entries; they are accepted only at
    the legacy entry (always, when entries are not configured).
    """
    entry = request_public_entry(request)
    if origin is None:
        return (
            not public_entries_configured()
            or entry.origin == get_legacy_public_origin()
        )
    return origin == entry.origin

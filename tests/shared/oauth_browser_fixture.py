"""Valid browser-bound flows for provider contract tests.

Security/admission tests deliberately call production functions directly.
This factory adds a fresh nonce to ordinary fixture states and supplies only
that factory's matching cookie when invoking a provider callback.
"""

import hashlib
import secrets
from typing import Any

from xagent.web.api.auth import (
    _ordinary_oauth_cookie_name,
)
from xagent.web.api.auth import create_access_token as _create_access_token
from xagent.web.api.auth import generic_oauth_callback as _generic_oauth_callback
from xagent.web.api.auth import (
    verify_token,
)

_fixture_nonces: dict[str, str] = {}


def create_access_token(data: dict[str, Any], *args: Any, **kwargs: Any) -> str:
    data = dict(data)
    if (
        data.get("type") == "oauth_state"
        and "browser_nonce_hash" not in data
        and data.get("actor_flow_nonce") is None
        and data.get("resource_owner_key") is None
    ):
        nonce = secrets.token_urlsafe(32)
        digest = hashlib.sha256(nonce.encode()).hexdigest()
        data["browser_nonce_hash"] = digest
        _fixture_nonces[digest] = nonce
    return _create_access_token(data, *args, **kwargs)


def generic_oauth_callback(
    provider: str, request: Any, *args: Any, **kwargs: Any
) -> Any:
    payload = verify_token(request.query_params.get("state", "")) or {}
    digest = payload.get("browser_nonce_hash")
    nonce = _fixture_nonces.get(digest) if isinstance(digest, str) else None
    if nonce:
        if not hasattr(request, "cookies"):
            request.cookies = {}
        request.cookies[_ordinary_oauth_cookie_name(digest)] = nonce
    return _generic_oauth_callback(provider, request, *args, **kwargs)

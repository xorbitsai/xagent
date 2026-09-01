"""Per-provider quirks in how OAuth token endpoints respond to a token
request, shared between the initial code exchange (api/auth.py) and token
refresh (tools/config.py) so a quirk fixed in one lifecycle path can't
silently regress in the other -- see the two call sites' own comments for
the underlying provider behavior each policy works around.
"""

from typing import Mapping

# Providers whose token endpoint answers form-urlencoded
# (access_token=...&scope=...&token_type=...) unless the request explicitly
# asks for JSON via an Accept header -- without it, response.json() raises
# instead of parsing a genuinely successful response.
_PROVIDERS_REQUIRING_JSON_ACCEPT_HEADER = frozenset({"github"})

# Meta's two distinct top-level OAuthException codes for a dead
# access/session token within its nested error shape (see
# meta_invalid_token_error_code): 190 access token is invalid/expired, 102
# session key invalid or no longer valid. Meta nests further detail for
# code 190 (password changed, expired session, user logged out, ...) as a
# separate `error_subcode` field rather than a different top-level `code`
# -- checking `code == 190` alone already covers every one of those
# subcodes without needing to enumerate them.
# Not exhaustive -- Meta has other, rarer session-invalidation codes this
# doesn't cover; those fail open (classified transient, retried) rather
# than incorrectly deleting a connection that's still alive.
_META_INVALID_TOKEN_ERROR_CODES = frozenset({190, 102})


def requires_json_accept_header(provider: str) -> bool:
    return provider.lower() in _PROVIDERS_REQUIRING_JSON_ACCEPT_HEADER


def meta_invalid_token_error_code(error: object) -> str | None:
    """Normalize Meta's nested refresh-error shape to a standard OAuth2
    error code, or None if `error` doesn't match it.

    Meta nests its error as an object instead of the standard top-level
    string `error` field other providers use (``{"error": {"type":
    "OAuthException", "code": 190, ...}}``); see
    _META_INVALID_TOKEN_ERROR_CODES for which codes mean the token/session
    is dead, normalized here to the standard `invalid_grant` code callers
    already recognize.
    """
    if (
        isinstance(error, Mapping)
        and error.get("type") == "OAuthException"
        and error.get("code") in _META_INVALID_TOKEN_ERROR_CODES
    ):
        return "invalid_grant"
    return None

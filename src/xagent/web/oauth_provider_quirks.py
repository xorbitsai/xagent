"""Per-provider quirks in how OAuth token endpoints respond to a token
request. Some of these are shared between the initial code exchange
(api/auth.py) and token refresh (tools/config.py) so a quirk fixed in one
lifecycle path can't silently regress in the other (requires_json_accept_
header is); others (meta_invalid_token_error_code) are refresh-specific --
see each function's own docstring, and the call sites' own comments, for
which lifecycle path(s) it applies to and the underlying provider behavior
it works around.
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


def matches_provider_family(provider: str, base_name: str) -> bool:
    """Match `provider` to a family rooted at `base_name`: exact equality, or
    a "-"-anchored prefix (`base_name` + "-"), case-insensitively.

    Anchored to a "-" separator, not a bare prefix: `oauth_providers.name` is
    admin-settable via POST/PUT /admin/mcp/providers, so a bare
    `.startswith(base_name)` would also match an unrelated custom provider an
    admin happened to name e.g. "salesforcelite" -- silently pulling it into
    a family (and whatever safeguard/quirk that family requires) it has no
    reason to be part of. The "-" anchor is what an admin-created variant row
    (e.g. "salesforce-sandbox", "employment-hero-sandbox") is expected to use,
    per example.env's documented workaround for providers with no per-user
    environment toggle.

    Shared by every family-matching predicate in this codebase
    (auth.py's _is_salesforce_provider, requires_pkce below) so the matching
    algorithm itself has exactly one implementation -- a caller composing a
    new family check should use this rather than re-deriving the same
    equality/prefix logic.
    """
    lowered = provider.lower()
    return lowered == base_name or lowered.startswith(f"{base_name}-")


# Providers whose authorization-code grant requires PKCE (a code_challenge on
# the authorize redirect, a code_verifier on the token exchange), with no
# per-app way to disable it. Every PKCE-only code path must use this same
# predicate, or a variant row would silently skip the safeguard its family
# requires.
_PKCE_PROVIDER_PREFIXES = (
    "salesforce",
    # Employment Hero rolled out a PKCE mandate for this grant effective
    # 2026-09-14, with no per-app opt-out. See
    # https://developer.employmenthero.com/partner-guides for the notice.
    "employment-hero",
)


def requires_pkce(provider: str) -> bool:
    return any(
        matches_provider_family(provider, prefix) for prefix in _PKCE_PROVIDER_PREFIXES
    )

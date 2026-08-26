"""Per-provider quirks in how OAuth token endpoints respond to a token
request, shared between the initial code exchange (api/auth.py) and token
refresh (tools/config.py) so a quirk fixed in one lifecycle path can't
silently regress in the other -- see the two call sites' own comments for
the underlying provider behavior each policy works around.
"""

# Providers whose token endpoint answers form-urlencoded
# (access_token=...&scope=...&token_type=...) unless the request explicitly
# asks for JSON via an Accept header -- without it, response.json() raises
# instead of parsing a genuinely successful response.
_PROVIDERS_REQUIRING_JSON_ACCEPT_HEADER = frozenset({"github"})


def requires_json_accept_header(provider: str) -> bool:
    return provider.lower() in _PROVIDERS_REQUIRING_JSON_ACCEPT_HEADER


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

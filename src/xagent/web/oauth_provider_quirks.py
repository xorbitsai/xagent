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

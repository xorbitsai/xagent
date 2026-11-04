"""Builtin OAuth providers whose endpoints live on the customer's own account
host rather than one fixed domain -- e.g. Zendesk's
``https://{subdomain}.zendesk.com/oauth/tokens``.

Every other builtin provider reads one static ``auth_url``/``token_url`` off
its ``oauth_providers`` row. A provider registered here instead needs the
user's account identifier (a single DNS label) before the authorize
redirect, and must keep using that same account for the token exchange,
refresh, and API calls. The flow binds it like this:

- Bearer-authenticated ``POST /api/auth/{provider}/login`` accepts the account
  option and issues a one-use ticket. Its navigation GET validates the label, resolves
  the composed host and rejects private-network addresses, then carries the
  label inside the signed OAuth state.
- The callback reads the label only from that state -- never from the
  callback query string -- and exchanges the code at the account's token
  endpoint without following redirects.
- The label is persisted to ``UserOAuth.instance_url`` alongside the tokens,
  and refresh derives its endpoint from that stored value.

Endpoint shapes are code-owned on purpose: a URL template in the
admin-editable ``oauth_providers`` columns could name any host, while this
registry can only ever produce ``https://<label>.<host_suffix>``.
"""

import re
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ..config import get_zendesk_marketplace_headers
from ..core.utils.security import PrivateNetworkHostError, reject_private_network_host
from .oauth_provider_quirks import matches_provider_family

# Fields the code exchange and refresh set themselves. token_params is merged
# into those requests at different points, so it must not name any of them.
_RESERVED_TOKEN_PARAMS = frozenset(
    {
        "grant_type",
        "code",
        "redirect_uri",
        "code_verifier",
        "refresh_token",
        "client_id",
        "client_secret",
        "scope",
    }
)

# One DNS label: letters, digits and inner hyphens, at most 63 characters.
# Matched with fullmatch, so a trailing newline cannot slip past "$".
_ACCOUNT_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def _no_request_headers() -> Mapping[str, str]:
    return {}


@dataclass(frozen=True)
class OAuthAccountHost:
    """How one provider's endpoints are derived from an account label."""

    host_suffix: str
    authorize_path: str
    token_path: str
    # None only for a provider with its own identity branch in
    # generic_oauth_callback; otherwise the callback fails closed.
    userinfo_path: str | None
    input_label: str
    input_placeholder: str
    input_help: str
    # Key the userinfo response nests the identity under (e.g. Zendesk's
    # {"user": {"id": ..., "email": ...}}); None reads top-level keys.
    userinfo_envelope: str | None = None
    # Extra form fields sent on both the code exchange and every refresh.
    token_params: Mapping[str, str] = field(default_factory=dict)
    # Extra headers for the code exchange, the identity lookup, and every
    # refresh. Called per request, so it reads the current settings.
    request_headers: Callable[[], Mapping[str, str]] = _no_request_headers

    def __post_init__(self) -> None:
        reserved = sorted(_RESERVED_TOKEN_PARAMS.intersection(self.token_params))
        if reserved:
            raise ValueError(
                f"token_params must not set standard token request fields: "
                f"{', '.join(reserved)}"
            )

    def input_metadata(self) -> dict[str, str]:
        """What the connect UI shows when asking for the account label."""
        return {
            "label": self.input_label,
            "placeholder": self.input_placeholder,
            "help": self.input_help,
            "suffix": f".{self.host_suffix}",
        }


@dataclass(frozen=True)
class OAuthAccountEndpoints:
    """The endpoints of one validated account."""

    account: str
    authorize_url: str
    token_url: str
    userinfo_url: str | None


class OAuthAccountError(ValueError):
    """The account identifier is missing, malformed, or not allowed."""


# provider_name (lowercase) -> host spec. "-"-anchored variant rows (e.g.
# "zendesk-sandbox") share the base provider's account host.
OAUTH_ACCOUNT_HOSTS: dict[str, OAuthAccountHost] = {
    "zendesk": OAuthAccountHost(
        host_suffix="zendesk.com",
        authorize_path="/oauth/authorizations/new",
        token_path="/oauth/tokens",
        userinfo_path="/api/v2/users/me.json",
        input_label="Zendesk subdomain",
        input_placeholder="your-company",
        input_help="The part before .zendesk.com in your Zendesk URL.",
        userinfo_envelope="user",
        # Zendesk rotates the refresh token on every refresh and applies its
        # defaults (for clients created since 2026-04-30, a 30-minute access
        # token; a 30-day refresh token) to any request that omits these,
        # including each refresh. Request the
        # documented maximums on every token request: two days per access
        # token keeps rotations rare, and 90 days lets an idle user's
        # connection survive a quarter without reconnecting.
        token_params={
            "expires_in": "172800",
            "refresh_token_expires_in": "7776000",
        },
        # Marketplace integration apps must identify themselves on every
        # request to the account (#3222); unconfigured, this adds nothing.
        request_headers=get_zendesk_marketplace_headers,
    ),
}


def get_oauth_account_host(provider: str | None) -> OAuthAccountHost | None:
    """Return the account host spec for ``provider``, if it has one.

    "-"-anchored variant rows (e.g. "zendesk-sandbox") share the base
    provider's account host.
    """
    if not provider:
        return None
    host = OAUTH_ACCOUNT_HOSTS.get(provider.lower())
    if host is not None:
        return host
    # Longest name first, so "foo-bar-sandbox" prefers "foo-bar" over "foo".
    for name in sorted(OAUTH_ACCOUNT_HOSTS, key=len, reverse=True):
        if matches_provider_family(provider, name):
            return OAUTH_ACCOUNT_HOSTS[name]
    return None


def normalize_oauth_account(account: object) -> str:
    """Return ``account`` as a lowercase DNS label, or raise OAuthAccountError."""
    if not isinstance(account, str) or not account.strip():
        raise OAuthAccountError("An account identifier is required.")
    label = account.strip().lower()
    if not _ACCOUNT_LABEL_PATTERN.fullmatch(label):
        raise OAuthAccountError(
            "The account identifier must be a single DNS label (letters, "
            "digits, and inner hyphens only) -- for example 'acme', not a "
            "full URL or hostname."
        )
    return label


def resolve_oauth_account_endpoints(
    host: OAuthAccountHost, account: object
) -> OAuthAccountEndpoints:
    """Validate ``account`` and build its endpoints on the account host.

    The label alone already pins the host under ``host_suffix``, but a
    legitimate name can still be pointed at an internal address by DNS, so
    every resolved address is checked too -- the same resolve-and-reject
    posture as zendesk.py's ``_base_url()``. Like that check, this narrows
    rather than closes the rebinding window: the HTTP client resolves the
    name again when it connects.
    """
    label = normalize_oauth_account(account)
    hostname = f"{label}.{host.host_suffix}"
    try:
        resolved = socket.getaddrinfo(
            hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        if not resolved:
            raise OSError("no addresses")
        for *_, sockaddr in resolved:
            reject_private_network_host(str(sockaddr[0]))
    except PrivateNetworkHostError as exc:
        raise OAuthAccountError(f"{hostname} is not allowed: {exc}") from exc
    except OSError as exc:
        raise OAuthAccountError(f"{hostname} could not be resolved.") from exc
    base_url = f"https://{hostname}"
    return OAuthAccountEndpoints(
        account=label,
        authorize_url=f"{base_url}{host.authorize_path}",
        token_url=f"{base_url}{host.token_path}",
        userinfo_url=(
            f"{base_url}{host.userinfo_path}" if host.userinfo_path else None
        ),
    )

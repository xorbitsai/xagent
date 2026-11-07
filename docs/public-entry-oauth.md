# Optional public browser entries and regional OAuth

This feature does not change API/app bases, S2S/widget audiences, provider callback
registration, MCP client registration keys, or the single-refresh-session model.
All regional OAuth behavior requires both `XAGENT_DUAL_ENTRY_OAUTH_ENABLED=true`
and a caller header `X-XAgent-OAuth-Launch-Version: 2`. Missing negotiation retains
legacy responses. No production entries or region routes are enabled by this code.

## Configuration API (`xagent.config`)

- `get_public_entries() -> tuple[PublicEntry, ...]`
- `PublicEntry(origin: str, region_cookie_domain: str | None, google_oidc_redirect_uri: str | None)`
- `public_entries_configured() -> bool`
- `get_legacy_public_origin() -> str`
- `normalize_public_origin(value: str) -> str`
- `get_primary_public_site_url() -> str | None`
- `get_dual_entry_oauth_enabled() -> bool`
- `get_oauth_authorization_entry_origin() -> str`
- `get_oauth_callback_origin_map() -> dict[str, str]`

The environment variables are `XAGENT_PUBLIC_ENTRIES_JSON`,
`XAGENT_PRIMARY_PUBLIC_SITE_URL`, `XAGENT_DUAL_ENTRY_OAUTH_ENABLED`,
`XAGENT_OAUTH_AUTHORIZATION_ENTRY_BASE_URL`, and `XAGENT_OAUTH_CALLBACK_ORIGIN_MAP`.
Entry JSON uses exactly `origin`, `region_cookie_domain`, and
`google_oidc_redirect_uri`. Explicit null domain means host-only, never fallback.
The configured legacy origin must remain present. Primary links are separate from
current-interaction redirects. Callback aliases are explicit; no redirect discovery.

## Web helpers (`xagent.web.public_entries`)

- `request_public_entry(request: Request) -> PublicEntry`
- `entry_url(entry: PublicEntry, path: str, params: dict[str, str] | None = None) -> str`
- `async google_state_entry(request: Request, google: Any) -> PublicEntry`
- `validate_exchange_entry(request: Request, origin: str | None) -> bool`

Ingress MUST overwrite forwarding headers after authenticating the upstream;
applications must not expose trusted forwarded-host parsing directly to attackers.
Explicit entry configuration rejects duplicate/multi-value and unknown hosts.
Legacy unconfigured deployments retain their previous configured frontend behavior.
`google_state_entry` reads the Authlib framework's `get_state_data(session, state)`
before ordinary Authlib token exchange. It does not inspect `_state_*` keys.
The callback URL in saved state is authoritative, including its exact spelling.

SaaS OIDC overrides should resolve the entry at start and before callback errors or
completion, pass its callback URI to `complete_google_oidc_authorization`, and pass
`entry.origin` as the third argument to `_create_exchange_code(user_id, epoch, entry_origin=None)`.
Before `_consume_exchange_code`, decode and call `validate_exchange_entry` on
`OidcExchangeTransaction.entry_origin`. Missing origin is accepted only at the
legacy entry. Preserve the existing DB replay/epoch checks.

## Hosting adapter (`xagent.web.oauth_handoff`)

Install `request.app.state.oauth_handoff_adapter` implementing:

```python
@dataclass(frozen=True)
class OAuthHandoffContext:
    product: str
    environment: str
    region: str
    authorization_entry_origin: str

class OAuthHandoffAdapter(Protocol):
    def context(self, request: Request) -> OAuthHandoffContext: ...
    def set_route_cookie(
        self, response: Response, *, state: str,
        kind: Literal["builtin", "mcp"], provider: str | None,
    ) -> None: ...
```

Context MUST come from authoritative deployment configuration, not request region
parameters. Cookie signing MUST use the adapter's deployment region and existing
route-token codec/secret. Never set `selected_region` in this hook. Register
`xagent.web.oauth_handoff.router` if the hosting app does not use the upstream app.
It exports GET `/api/oauth/launch/{region}`; no adapter or disabled feature fails
closed. `context` is called both when signing and redeeming. The handler checks
signed product/environment/target origin/region, path and actual host, purpose,
expiry, user/session epoch, and resource bindings before releasing browser proof.
Tickets expire after 120 seconds and use distinct `oauth-launch:builtin:` and
`oauth-launch:mcp:` consumption namespaces in the existing store.

Redemption takes two GETs. The authorization entry has no session for a user
signed in only on another entry, so a ticket is a bearer credential that is not
bound to the initiating browser. The first GET validates everything, consumes
nothing and renders a script-free, unframeable confirmation page naming the
target account and provider/server, while setting a per-ticket
`SameSite=Strict`, `HttpOnly` nonce cookie scoped to the launch path. Only the
second GET (`?ticket=...&confirm=<nonce>`) with the matching cookie consumes the
ticket and releases browser proof. A ticket URL received from someone else stops
at a page naming *their* account (account-linking CSRF), and a cross-site
navigation straight to the second step lacks the cookie. The legacy
`oauth-launch:` namespace and old launch route are unchanged.

For builtin consent retries, the hosting callback middleware must issue a new
scoped route cookie for the replacement authorization state; the bootstrap hook
alone cannot handle a later callback-generated redirect. MCP callback 303 does
not necessarily mean completion: do not indiscriminately clear routing cookies.

## Browser response contract

Negotiated responses contain:

```json
{"authorization_url":"https://configured-old-origin/api/oauth/launch/au?ticket=...","launch_mode":"regional_bootstrap"}
```

Builtin also preserves `callback_origin`, after the configured historical alias
mapping. The frontend validates new URLs against **same-origin**
`/api/deployment-config.oauth_authorization_origins`, not an allowlist echoed by a
launch response. SaaS's replacement deployment-config route must expose that
optional list when enabled. Unknown modes and unavailable config fail closed.
Builtin retains source/origin/provider/type message checks. MCP retains null
opener and status polling; closing the popup is never proof of success.

MCP keeps its existing flow/client/callback/registration hash; only proof-cookie
write moves to the old-host bootstrap. The flow uses relative
`/auth/mcp-complete`, a credential-free public frontend page. The hosting edge must
allow that exact path without region cookies. Legacy `redirect_after` behavior is
unchanged. New error redirects use only a controlled error code.

Deployment prerequisites (not performed here): trusted dual ingress, matching
fixed-region edge bootstrap routes, adapter and callback middleware integration,
Google callback registrations, and browser verification of actual historical 301
callback hosts. Promote the primary site separately from enabling entries/routes.

Additional prerequisites:

- The browser calls `/api` on the entry's own host (empty/same-origin
  `NEXT_PUBLIC_API_URL`). Entry resolution and the Google exchange binding use
  the request Host, so an absolute API base on another host is rejected.
- The authorization entry must equal the provider callback host or one of its
  configured `XAGENT_OAUTH_CALLBACK_ORIGIN_MAP` aliases; nonce/state cookies are
  host-only, so a mismatch surfaces only as `invalid_state` after consent.
- Ingress overwrites Host/X-Forwarded-Host with the validated public entry.
- Unconfigured deployments (`XAGENT_PUBLIC_ENTRIES_JSON` unset) keep the exact
  legacy frontend origin and Google error mapping; the strict origin validation
  applies only to explicit entries.

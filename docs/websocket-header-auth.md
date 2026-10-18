# Private WebSocket login credentials

Private chat, builder chat, builder preview and progress WebSockets no longer
accept `token` or `access_token` in the query string, including empty, repeated
and percent-encoded names. There is no query-credential fallback.

Browser clients offer two subprotocol values:

```javascript
new WebSocket(urlWithoutCredentials, [
  "xagent-auth-v1",
  `xagent-auth-token.${accessToken}`,
]);
```

Non-browser clients may instead send one `Authorization: Bearer ...` header.
Other Authorization schemes (such as gateway Basic authentication) and empty
headers are not application login credentials. They do not authenticate a user
or conflict with the browser carrier; malformed Bearer credentials are rejected.
Combining the two carriers or supplying duplicate/malformed login credentials
is rejected. The server does not select or echo the credential subprotocol.
This is a header carrier for the existing access JWT, **not** a newly minted,
short-lived or single-use ticket. Use WSS in production and treat the entire
`Sec-WebSocket-Protocol` request header as sensitive in proxy/APM/debug logs.
The common application text redactor recognizes the new credential prefix,
WebSocket URLs and navigation tickets; it cannot sanitize third-party logs or
delete credentials already retained in historical logs.

Existing user/purpose/signature/session-generation checks, task ownership,
five-second idle revocation polling and per-message revalidation still apply.
Changing an explicit `token` prop changes the hook's in-memory login identity,
so uploads and retries are not reused across explicit login identities.
An auth-context refresh for the same user/session intentionally retains that
identity while using the refreshed credential for subsequent connections.

Public widget/share capabilities and the SaaS external client/session protocols
retain their separate, purpose-scoped admission contracts. They are not ordinary
administrator/user login credentials and are not granted access by this helper.

## Deployment

Deploy matching frontend and every API replica together. Old private query-JWT
clients must migrate; old API replicas do not understand browser header login.
Legacy URI credentials are rejected before the upgrade. In production this is
an HTTP 403 failed handshake (typically browser close 1006), not a delivered
WebSocket close reason. There is no authenticated legacy fallback.
Proxies must forward `Sec-WebSocket-Protocol` unchanged on upgrade requests.

Integrations with additional authentication middleware must inspect the offered
login credential for initial
MFA admission, per-frame checks and idle revocation. The core upgrade, SaaS
authentication adapters and MFA changes require combined release acceptance.

This addresses only the private WebSocket URL carrier, not public capability
protocols, historical-log cleanup, one-use capabilities or OAuth navigation.

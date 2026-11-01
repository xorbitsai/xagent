# Private WebSocket login credentials

Private chat, builder chat, builder preview, progress and streaming speech
WebSockets no longer
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
five-second idle revocation polling and chat-message revalidation still apply.
Streaming speech revalidates before starting the provider session and uses the
same revocation monitor. Individual PCM fragments and `commit`/`close` controls
do not perform database reads; revocation during an active stream is detected
by that monitor.
Changing an explicit `token` prop changes the hook's in-memory login identity,
so uploads and retries are not reused across explicit login identities.
An auth-context refresh for the same user/session intentionally retains that
identity while using the refreshed credential for subsequent connections.

The SaaS external client/session protocols retain their separate,
purpose-scoped admission contracts and are not granted access by this helper.

## Public widget/share guest tokens

Public widget and share chat sockets (`/api/widget/chat/ws/{task_id}`,
`/api/share/chat/ws/{task_id}`) use the same header carriers for the 30-day
guest token: the `xagent-auth-v1` + `xagent-auth-token.<token>` subprotocol
pair from browsers, or `Authorization: Bearer` from other clients. The old
`?token=` query parameter is retired: a handshake that still carries a URI
credential is refused even next to a valid header (accepted, then closed with
4003 "Authentication required"). The guest token remains a guest capability,
not a user login: admission, per-guest task isolation and per-message
revalidation are unchanged, and the server negotiates only `xagent-auth-v1`.

Public file routes (`/api/files/public/preview|download/{file_id}`) likewise
read the guest token only from the Authorization header and ignore `?token=`.
URLs the browser loads itself (media elements, "Open" links, HTML-preview
assets) cannot send that header, so the frontend exchanges it for a
short-lived ticket bound to one file: `POST /api/files/public/tickets/{file_id}`
returns `{ticket, expires_in}`, used as `?ticket=` on those two routes
(including `relative_path` assets under that file). The ticket lives
`XAGENT_FILE_STREAM_TICKET_TTL_SECONDS` and is signed but readable, so it
carries no reusable identity: only a digest of the share token / widget key
and a keyed, non-reversible binding of the guest id (never the id itself,
which `/api/widget/auth` would re-sign into a fresh guest token). Redemption
restores the guest from the file's own task and revalidates against the live
deployment on every use, so disabling or rotating the link revokes it. One
exception: a ticket minted from a legacy widget token that carries no
`widget_key` has no key digest to compare, so rotating the widget key does not
revoke it before it expires; disabling or unpublishing the widget still does,
as does key rotation for share-link and keyed widget tickets. A ticket is not accepted as
a Bearer guest token, and a guest token or private stream ticket is not
accepted as `?ticket=`. Server access and handshake logs redact credential-like
query values (`token`, `ticket`, `*_key`, ...), and the per-header lines the
WebSocket stack logs at debug level (`--debug`) mask the
`xagent-auth-token.` subprotocol item, `Authorization`/`Proxy-Authorization`
credentials (the scheme is kept), cookies and `*-key`/`*-token` headers.

## Deployment

Deploy matching frontend and every API replica together. Old private query-JWT
clients must migrate; old API replicas do not understand browser header login.
For the public guest contract, roll out the API first (it accepts the header
carriers and serves the ticket endpoint), then the frontend and any embedded
widget bundle. A new frontend against an old API replica cannot mint file
tickets (404, so media falls back to a blob fetch and guest-task "Open" links
and HTML-preview assets fail) and its header-only guest sockets are refused;
an old frontend against the new API loses guest-task file and socket access
because its `?token=` URLs are ignored.
Legacy URI credentials are rejected before the upgrade. In production this is
an HTTP 403 failed handshake (typically browser close 1006), not a delivered
WebSocket close reason. There is no authenticated legacy fallback.
Proxies must forward `Sec-WebSocket-Protocol` unchanged on upgrade requests.

Integrations with additional authentication middleware must inspect the offered
login credential for initial
MFA admission, per-frame checks and idle revocation. The core upgrade, SaaS
authentication adapters and MFA changes require combined release acceptance.

This addresses the private and public guest WebSocket URL carriers and the
public file-route guest token, not historical-log cleanup, one-use
capabilities or OAuth navigation.

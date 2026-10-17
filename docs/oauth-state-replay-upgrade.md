# Ordinary OAuth state replay protection

Ordinary callbacks require a validated browser nonce and a durable database
claim before any provider exchange. Both the public route and direct callback
helper enforce this invariant; actor flows retain their separate durable flow
state. Legacy unbound ordinary states must restart from the connect screen.

Claims use the verified JWT signing input, not the malleable base64url spelling
of its signature. Canonically encoded claims from the previous raw-token
implementation are recognized during upgrade. OAuth and OIDC exchange cleanup
are restricted to their own token namespaces so a fast OIDC worker cannot
delete a still-valid OAuth replay claim.

Deploy all API replicas together. For a rollout from the previous raw-token
claim scheme, drain callbacks for the ten-minute state lifetime and restart
connections after every replica is upgraded. Mixed fleets still include the
old vulnerable verifier and are not a closed replay-protection boundary.

SQLite concurrency and expiry tests run locally; the PostgreSQL concurrency
release gate uses `XAGENT_TEST_POSTGRES_URL` and disposable test databases. A
skipped PostgreSQL test is not proof of PostgreSQL concurrency behavior.

## Connector launch without login JWT URLs

Start ordinary connectors using a Bearer-authenticated
`POST /api/auth/{provider}/login` with a JSON body containing `app_id`,
`redirect` and optional `account`. An empty body is valid for connector
manifests passing these non-secret options as query parameters; conflicting
body/query options are rejected. Never put access or refresh credentials in
the URL. The old `GET ...?token=` contract now returns 410 without resolving it.
The Python `oauth_login` adapter is retained for trusted in-process callers;
its token argument is not exposed as an HTTP query parameter.

The response contains a relative `authorization_url` under the trusted API
base, carrying a signed, purpose/provider/session-bound 30-second launch
ticket. Open a blank popup synchronously on the click stack, then navigate it
after the POST succeeds. Only this top-level navigation sets the state cookie,
so a frontend on another site does not require third-party fetch cookies.
Ticket consumption is durable and single-use across workers/restarts, using
the existing `oidc_consumed_tokens` table in an isolated `oauth-launch:`
namespace. Its cleanup uses the database clock. A provider setup failure also
consumes the ticket; restart from the POST rather than retrying the same URL.
The resulting provider state retains its existing browser cookie, single-use
callback claim, PKCE and password-generation checks.

The POST also returns `callback_origin`, derived from the configured provider
redirect URI, not from the incoming Host header or the frontend/API base URL.
The frontend validates the launch destination and requires that callback
origin, the exact source popup and app identity on completion messages (also
for knowledge-base connectors). It sends only the opener origin, not its
current path/query/fragment, as the completion target. Launch failures and
blocked popups show safe user feedback without logging tickets or responses.
Launch responses and redirects use no-store and
no-referrer policies. No new database migration or deployment-owned secret is
required beyond the existing OIDC schema and signing configuration.

Deploy matching clients and all API replicas together. Update external
connector manifests (including SaaS/Toby) from query-JWT GET to the POST
contract before publishing the core upgrade. Existing query-token clients
intentionally stop working rather than silently keeping the unsafe route.
SaaS must advance its core gitlink before advertising this new POST contract.
Coordinate that upgrade with its authentication/MFA adapters; this branch does
not change those open PRs, deploy production or claim full CASA item 7 closure.
Authenticated WebSocket transports and historical token-bearing logs are
separate remaining work. Real-browser/cross-site and PostgreSQL acceptance
remain release checks; the SQLite/jsdom suites are not substitutes for them.

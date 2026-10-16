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

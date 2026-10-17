# Authentication rate limits and deployment

Before starting a production/staging API, configure a reachable
`XAGENT_REDIS_URL` on every replica. Startup pings Redis and refuses admission
if it is absent, malformed or unreachable. Runtime failures fail closed with a
sanitized HTTP 503. A Redis command has a two-second connection/read timeout
and no automatic retry/backoff. Only development without secure-secret
enforcement can use bounded in-process budgets, including during an outage.
Local fallback does not provide cross-process protection.

Budgets use a fifteen-minute sliding window:

| Endpoint | Source IP | Resolved account/email | IP + account/email |
| --- | --- | --- | --- |
| Login | 200 | 50 | 5 |
| Forgot password | 200, shared with reset | 50 | 5 |
| Reset password | same shared recovery budget | — | — |

Username and email login aliases share the resolved user-ID budget. Recovery
email budgets are deliberately higher than the browser/email-pair budget so
five unauthenticated requests cannot globally lock an email. Reset guesses all
share the source-IP budget; arbitrary token strings never allocate buckets.
Short reset passwords are rejected before consuming rate-limit capacity.

Login first reserves source-IP capacity before opening an auth database session.
Only admitted requests resolve the account and atomically reserve account/pair
budgets. The pre-admission budget counts failures and in-flight work, including
requests subsequently rejected by the identity budgets. On successful login,
only that request's IP reservation is removed; other users' failures remain.
This prevents many successful logins behind corporate NAT or an unknown peer
from cumulatively exhausting the IP budget. Refund failure never fails an
already-committed successful login.

Recovery reserves its shared IP/email/pair budgets atomically and counts all
requests, including generic successful forgot-password responses: refunding those
would disable email-abuse protection and reveal account existence. Forgot/reset
intentionally share one IP budget. Rejection returns HTTP 429 with `Retry-After`
equal to the maximum remaining wait across exhausted buckets in that reservation.

Configure positive integer limits in `config.py` through:

- `XAGENT_AUTH_RATE_LIMIT_WINDOW_SECONDS` (900)
- `XAGENT_AUTH_LOGIN_IP_LIMIT` (200), `XAGENT_AUTH_LOGIN_ACCOUNT_LIMIT` (50),
  `XAGENT_AUTH_LOGIN_PAIR_LIMIT` (5)
- `XAGENT_AUTH_RECOVERY_IP_LIMIT` (200), `XAGENT_AUTH_RECOVERY_EMAIL_LIMIT` (50),
  `XAGENT_AUTH_RECOVERY_PAIR_LIMIT` (5)
- `XAGENT_AUTH_RATE_LIMIT_MAX_LOCAL_KEYS` (10000, development only)

Use consistent configuration across replicas. Size limits for actual NAT/proxy
traffic and configure `XAGENT_TRUSTED_PROXY_HOPS` only for trusted proxies; do not
blindly trust forwarded addresses to work around an incorrectly identified peer.
Invalid/non-positive values fall back to defaults, never disabling protection.

Accepted trade-off: resolved user-ID account budgets prevent username/email alias
bypass, but saturating a known account can reveal an alias relationship through
429 versus 401. These generic response bodies are not a claim of complete account
enumeration resistance; monitor distributed abuse and assess this privacy trade-off
for the deployment. Switching to identifier-only keys would reopen alias bypass.

Redis keys expire independently; there is no global active-identity cap that
one source can exhaust to reject unrelated users. Plan Redis memory/availability
and use an appropriate dedicated namespace/service; do not use an eviction
policy that silently drops active security budgets. The plain Redis client
targets standalone Redis or a managed proxy, not RedisCluster discovery.
Local development evicts expired or unlocked buckets at capacity, never active
lockouts. If every bounded slot is locked it returns a distinct 503 capacity
error rather than unlocking an attacker or claiming an identity is rate-limited.

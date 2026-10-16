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
| Forgot password | 200 | 50 | 5 |
| Reset password | 200 | — | — |

Username and email login aliases share the resolved user-ID budget. Recovery
email budgets are deliberately higher than the browser/email-pair budget so
five unauthenticated requests cannot globally lock an email. Reset guesses all
share the source-IP budget; arbitrary token strings never allocate buckets.
Short reset passwords are rejected before consuming rate-limit capacity.

All budgets for one operation are checked/reserved atomically in one Redis
script (or one local lock). Rejection returns HTTP 429 with remaining-window
`Retry-After` seconds, without partially charging other buckets. Successful
login clears its identity/pair budgets, not the IP-wide anti-churn budget.

Redis keys expire independently; there is no global active-identity cap that
one source can exhaust to reject unrelated users. Plan Redis memory/availability
and use an appropriate dedicated namespace/service; do not use an eviction
policy that silently drops active security budgets. The plain Redis client
targets standalone Redis or a managed proxy, not RedisCluster discovery.
Local development evicts expired or unlocked buckets at capacity, never active
lockouts. If every bounded slot is locked it returns a distinct 503 capacity
error rather than unlocking an attacker or claiming an identity is rate-limited.

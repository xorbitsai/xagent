# Browser idle timeout

Interactive browser logins can be expired by the server after a period without
user activity. This is separate from token lifetimes: the browser refreshes its
access token automatically, so a 120-minute access token and a 7-day refresh
token (which slides on every refresh) never end an unattended login on their
own. Xagent has no absolute login limit of its own; a deployment that adds one
keeps it, and activity never extends it.

The control is off by default. Enabling it requires a browser that reports
activity (see below); otherwise every login expires after the threshold.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `XAGENT_BROWSER_IDLE_TIMEOUT_ENABLED` | `false` | Enforce idle expiry |
| `XAGENT_BROWSER_IDLE_TIMEOUT_MINUTES` | `30` | Threshold for ordinary users |
| `XAGENT_BROWSER_IDLE_TIMEOUT_ADMIN_MINUTES` | `15` | Threshold for platform administrators (`users.is_admin`) |

An unrecognised switch value, or a non-positive or non-numeric threshold while
the switch is on, stops startup instead of falling back to a weaker value. The
role is read from the database on every check; claims and organisation or team
roles never select the threshold. Use the same values on every replica.

## Logins

Every login (password, Google OIDC, or an adapter using
`begin_browser_session`) creates a random session id `sid` and a row in
`browser_sessions` (`sid`, `last_activity_at`, `last_used_at`, `revoked_at`).
The access token, refresh token and every ticket minted for the login carry the
`sid`. The refresh token itself is still stored once per user;
`users.browser_session_id` records which login holds it. Signing in on another
device does not end the first device's login: as before, the first device
fails at its next refresh, and until then it follows its own idle deadline.

Rows live in the database, so every replica sees the same state and it survives
restarts. Activity, refresh and logout use conditional UPDATEs on the login's
row (taken before the `users` row) that repeat the deadline and revocation
tests, so a request racing expiry cannot revive a login.

## Activity

Only two things move the deadline: the login itself and
`POST /api/auth/activity` (Bearer access token, empty body). Migrating a legacy
refresh token (below) starts a new clock. Refresh, polling,
`GET /api/auth/verify`, WebSocket heartbeats and reconnects, background tasks
and all other requests, including writes, do not.

The browser decides which interactions to report. The server writes at most
once per login per 60 seconds; extra reports return 200 without a write, so the
deadline can be up to 60 seconds earlier than the last interaction plus the
threshold. A report for an expired login is rejected and writes nothing.

## Responses

Login, refresh, Google OIDC exchange, `GET /api/auth/verify` and
`POST /api/auth/activity` include:

```json
"idle_session": {
  "enforced": true,
  "sid": "...",
  "idle_timeout_seconds": 1800,
  "idle_expires_in": 1734,
  "server_time": "2026-11-13T10:00:00Z",
  "absolute_expires_in": null,
  "activity_min_interval_seconds": 60
}
```

- No `idle_session` object: the server predates this feature.
- `enforced: false`: the switch is off. `idle_expires_in` is informational;
  do not log the user out because of it. Keep reporting activity so the data is
  accurate when the switch is turned on.
- Use `idle_expires_in` (server clock), not a deadline derived from the client
  clock.
- `absolute_expires_in` is always `null` in Xagent; a deployment with an
  absolute limit fills it.

## Revocation and rejections

A login is revoked (`revoked_at`) while the switch is on when a request
carrying its `sid` arrives after its deadline, when the sweep finds it past its
deadline, or when it logs out. Every replica runs the sweep every 60 seconds;
it also deletes rows whose last use is older than the refresh-token lifetime
plus the access-token lifetime plus one day, when no token for them can still
be valid. **A revoked login stays rejected after the switch is turned off.**
Revocation is keyed by `sid`, so it never affects another login of the user.

| Case | Switch on | Switch off |
| --- | --- | --- |
| Idle deadline passed | 401 `SessionIdleExpired` | accepted |
| Revoked login | 401 `SessionIdleExpired` | 401 `SessionIdleExpired` |
| Token without `sid`, or `sid` without a row | 401 `SessionIdleExpired` | accepted |
| Database unavailable | 5xx, never 401 | 5xx |

Once a login is rejected, access-token requests carrying its `sid` and
refreshes with the user's current refresh token keep returning
`SessionIdleExpired`. An access token past its own expiry still gets
`TokenExpired`, and a refresh token already replaced by a newer login gets the
existing invalid-refresh-token error.

401 responses carry `WWW-Authenticate: Bearer` and apply to every route using
`get_current_user`, refresh, verify, activity, file stream tickets and OAuth
launch/handoff tickets; optional authentication treats the request as
anonymous. An open WebSocket is closed with code 4010. A rejected WebSocket
handshake keeps the endpoint's existing close code (4001 or 1008); on 4001 the
browser refreshes once and then receives `SessionIdleExpired`. Do not refresh
on `SessionIdleExpired` or 4010; `TokenExpired` and 4001 keep their meaning.

Logout with the switch on also revokes the presented access token's login.
Its refresh-token handling is unchanged: the stored refresh token is cleared
when the presented one matches, or when none is presented.

Remaining gap: a login whose deadline passed after the last sweep and that saw
no request before the last switch-on replica stopped can still be used after
the switch is turned off. With the sweep running, this window is at most about
60 seconds.

## Streaming and derived access

With the switch on and the database reachable, access stops after expiry:

| Path | Bound |
| --- | --- |
| WebSockets authenticated with access tokens (chat, build chat, build preview, progress, ASR) | Rechecked about every 5 s (±20%) and, except ASR, on each inbound message; closed with 4010 within about 6 s. Three failed rechecks close with 1011. |
| File stream tickets | Checked on every redemption, including each Range request |
| Whole-file transfers already started (FileResponse, X-Accel) | Run to completion |
| Object-storage signed URLs (`XAGENT_FILE_DELIVERY_REDIRECT_ENABLED`, off by default) | Valid for `XAGENT_FILE_DELIVERY_SIGNED_URL_TTL_SECONDS` (default 300 s) after issue; a download started within it completes |
| TTS stream | Checked at start; at most 600 s |
| OAuth launch (30 s) and regional handoff (120 s) tickets | Checked at redemption |

Not covered: `/api/files/public/preview` and `/api/files/public/download`
accept the file ID alone for ordinary tasks, independent of any login.

## Scope

API keys, widget and share guest tokens, public file and external media
tickets, OAuth state tokens, trigger webhooks and the SQL worker do not go
through browser-token validation. Browser expiry does not cancel accepted
tasks or schedules; they run on the server without browser tokens.

Scripts that sign in with a password and reuse the access token are browser
logins and follow the same thresholds. API keys are an alternative only for
part of the API: no `/api/*` route accepts an API key except A2A
(`/api/a2a`, agent-bound keys). Personal keys work on `/v1` me, features,
templates, agents, skills, MCP servers and chat tasks and files; workforce runs
accept workforce-bound keys. Knowledge bases, file listing, models, triggers
and similar endpoints have no API-key route.

## Adapters that sign their own tokens

Logins through `_prepare_login_response` + `commit_refresh_credentials` need no
change: the refresh token's `sid` is recorded in the same transaction. Paths
that issue browser tokens themselves use `xagent.web.browser_idle`:

- `begin_browser_session(db, user_id)` starts a login (caller commits) and
  returns its `sid`; put `browser_session_claims(sid)` in every token for it.
  Any path that writes `users.refresh_token` stores `hash_refresh_token(...)`
  and records the login's `sid` in `users.browser_session_id`.
- Before a refresh, `check_browser_session(db, user, presented_sid)` returns
  `(sid, last_activity_at)` or raises `BrowserSessionRejected` (map it with
  `rejection_http_exception()`). With the switch off, a refresh token without
  a known `sid` gets a new `sid` and `last_activity_at=None`: call
  `begin_browser_session(db, user_id, sid=sid)` to migrate it. Otherwise call
  `fence_browser_session_for_refresh(db, user, sid)` inside the refresh
  transaction, before updating `users`, and reject when it returns `False`.
  A refresh never moves the deadline.
- `idle_session_fields(user, sid, last_activity_at=..., absolute_expires_at=...)`
  builds the response object. Re-signed tickets keep the `sid` claim.
- Clearing `users.refresh_token` needs no change to `browser_session_id`;
  revoke a login only when the action is meant to end it.

Adapters without a core user (for example Toby) use
`idle_timeout_seconds(is_platform_admin)`, `build_idle_session(...)` and
`rejection_http_exception()`, and keep their own session state with the same
guarantees: shared across replicas and restarts, conditional writes that
cannot revive an expired login, per-login revocation that survives turning the
switch off. An adapter may expose its own activity endpoint (keeping its
CSRF/Origin checks) with the same semantics: check expiry before writing, write
at most once per login per 60 seconds, return 200 with `idle_session`, no 429.
Only a real authentication (password, provider OAuth callback, MFA
verification) starts a new login; re-signing session credentials keeps the
same `sid` and is not activity.

## Rollout and rollback

1. Deploy with the switch off. Logins and refreshes now carry `sid`; refreshing
   a login issued before this release moves it onto a new row.
2. Deploy browser activity reporting, still with the switch off. Without it,
   last activity is only the login time and nearly every login older than the
   threshold must sign in again at activation.
3. After the rollout has completed on every replica, wait at least the
   configured access-token lifetime, then check how many valid refresh tokens
   are not yet attached to a login:

   ```sql
   SELECT count(*) FROM users
   WHERE refresh_token IS NOT NULL
     AND refresh_token_expires_at > now()
     AND browser_session_id IS NULL;
   ```

   On SQLite use `CURRENT_TIMESTAMP` instead of `now()`. What remains are
   logins that have not been used since the deployment. The count can miss a
   login written by a replica still running the previous release during the
   rollout (its refresh token has no `sid`, but an older `sid` is still
   recorded); such a login migrates on its next refresh, or must sign in again
   at activation if it is never refreshed.
4. Turn the switch on in staging, verify, then in production. Logins without
   `sid`, and logins already idle beyond the threshold, must sign in again.

Turning the switch off stops new expiries immediately; revoked logins stay
revoked (see the remaining gap above).

// Shared by the link session Worker (link-session-worker-core.ts) and the
// main-thread host (link-session-host.ts). This module holds only constants,
// types and pure helpers: it carries no state and performs no I/O, so both
// sides can import it without sharing anything at run time.

// ---------------------------------------------------------------------------
// Retry and timing constants. Each one mirrors a value in public/widget.js; the
// comment names the line it is copied from. link-session-constants.test.ts
// compares each of them, the error-code set and the two functions below with
// the widget.js source text.
// ---------------------------------------------------------------------------

// widget.js:39 SESSION_TIMEOUT_MS.
export const RENEW_TIMEOUT_MS = 5000
// widget.js:57 SESSION_RETRY_POLICY.baseDelayMs.
export const RETRY_BASE_DELAY_MS = 1000
// widget.js:58 SESSION_RETRY_POLICY.backoffFactor.
export const RETRY_BACKOFF_FACTOR = 1.5
// widget.js:59 SESSION_RETRY_POLICY.maxDelayMs.
export const RETRY_MAX_DELAY_MS = 300000
// widget.js:60 SESSION_RETRY_POLICY.maxDelayAfterTimeoutMs.
export const RETRY_MAX_DELAY_AFTER_TIMEOUT_MS = 10000
// widget.js:61 SESSION_RETRY_POLICY.jitter.
export const RETRY_JITTER = 0.2
// widget.js:55 SESSION_RETRY_POLICY.maxRateLimitRetries.
export const MAX_RATE_LIMIT_RETRIES = 2
// widget.js:56 SESSION_RETRY_POLICY.maxRateLimitWaitMs.
export const MAX_RATE_LIMIT_WAIT_MS = 30000
// widget.js:70 SESSION_RETRY_AFTER_FALLBACK_MS.
export const RETRY_AFTER_FALLBACK_MS = 1000
// widget.js:71 SESSION_REFRESH_THRESHOLD_MS.
export const SESSION_REFRESH_THRESHOLD_MS = 60000
// widget.js:80 SESSION_STABLE_CONNECTION_MS.
export const STABLE_CONNECTION_MS = 60000
// widget.js:1618 `state.rapidRecoveries >= 3` in runRecoveryFlow: the fourth
// rapid recovery inside one stability window trips the breaker.
export const MAX_RAPID_RECOVERIES = 3

// The redeem endpoint has no replay window (a redeemed code is spent), so these
// two intentionally differ from widget.js: a slower per-attempt timeout that
// makes a spent code less likely on slow networks, and a bounded attempt count.
export const REDEEM_TIMEOUT_MS = 15000
export const REDEEM_MAX_ATTEMPTS = 5

// Host watchdogs for a Worker that cannot run (fail closed, never fall back to
// holding the reconnect token on the main thread).
export const WORKER_HELLO_TIMEOUT_MS = 30000
export const WORKER_REDEEMING_TIMEOUT_MS = 10000

// ---------------------------------------------------------------------------
// Link code and end reasons.
// ---------------------------------------------------------------------------

// "lk_" followed by 43 base64url characters (32 random bytes, unpadded).
export const LINK_CODE_PATTERN = /^lk_[A-Za-z0-9_-]{43}$/

export const ENDED_PATH = "/widget/chat/link/ended"
export const SESSION_IFRAME_PATH = "/widget/chat/session"

// The end page URL is the signal to the embedding app; this set is a stable
// contract with embedders.
export const END_REASONS = [
  "link_invalid",
  "session_ended",
  "unavailable",
  "network",
  "closed",
] as const
export type EndReason = (typeof END_REASONS)[number]

// The one runtime check for an end reason, derived from END_REASONS.
export function isEndReason(value: unknown): value is EndReason {
  return typeof value === "string" && (END_REASONS as readonly string[]).includes(value)
}

// Reasons a Worker may report. `closed` only comes from the user.
export type WorkerTerminalReason = Exclude<EndReason, "closed">

export function isWorkerTerminalReason(value: unknown): value is WorkerTerminalReason {
  return isEndReason(value) && value !== "closed"
}

export function endedUrl(reason: EndReason): string {
  return `${ENDED_PATH}?reason=${reason}`
}

// ---------------------------------------------------------------------------
// Server error codes.
// ---------------------------------------------------------------------------

// Known error codes of the frozen session error table (widget.js
// SESSION_ERROR_CODES) plus link_invalid.
export const KNOWN_ERROR_CODES: ReadonlySet<string> = new Set([
  "grant_malformed",
  "signature_invalid",
  "grant_expired",
  "grant_already_used",
  "agent_not_granted",
  "agent_not_available",
  "widget_disabled",
  "invalid_input",
  "invalid_runtime_context",
  "encryption_required",
  "reconnect_invalid",
  "session_expired",
  "identity_mismatch",
  "rate_limited",
  "link_invalid",
])

// Known code -> end reason for every code except rate_limited, which classify
// (link-session-worker-core.ts) handles before this table: retried within the
// rate-limit budget, otherwise network.
export function reasonForKnownCode(code: string): WorkerTerminalReason {
  switch (code) {
    case "link_invalid":
      return "link_invalid"
    case "session_expired":
    case "reconnect_invalid":
    case "identity_mismatch":
      return "session_ended"
    default:
      return "unavailable"
  }
}

// ---------------------------------------------------------------------------
// Worker <-> host messages.
// ---------------------------------------------------------------------------

export interface LinkSessionFields {
  session_token: string
  session_token_expires_at: string
  absolute_expires_at: string
  agent: Record<string, unknown>
  capabilities?: Record<string, unknown>
}

// Codes the Worker may report while it waits to retry; the chat page treats
// both as recoverable (use-widget-session.ts isParentRecoverableCode).
export type WorkerDegradedCode = "network_unavailable" | "rate_limited"

// The reconnect token is never a field of any message the Worker sends.
export type WorkerOutboundMessage =
  | { type: "hello" }
  | { type: "redeeming" }
  | { type: "session"; session: LinkSessionFields }
  | { type: "degraded"; code: WorkerDegradedCode }
  | { type: "terminal"; reason: WorkerTerminalReason }

// Messages the host posts to the Worker (link-session-host.ts
// LinkWorkerLike.postMessage takes only these).
export type WorkerInboundMessage =
  | { type: "start"; code: string }
  | { type: "renew" }
  | { type: "wake" }

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

export function isNonBlankString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0
}

export function isTimestamp(value: unknown): value is string {
  return isNonBlankString(value) && !Number.isNaN(Date.parse(value))
}

// `attempt` is 1-based and `random` is in [0, 1). Same formula as widget.js
// sessionRetryDelayMs.
export function retryDelayMs(attempt: number, random: number): number {
  const nominal = Math.min(
    RETRY_MAX_DELAY_MS,
    RETRY_BASE_DELAY_MS * Math.pow(RETRY_BACKOFF_FACTOR, Math.max(0, attempt - 1)),
  )
  return Math.round(nominal * (1 - RETRY_JITTER + 2 * RETRY_JITTER * random))
}

// The wait a `Retry-After` header asks for, in milliseconds. Same rules as
// widget.js retryAfterMs: whole seconds (a negative value or a missing or
// unreadable header falls back to RETRY_AFTER_FALLBACK_MS), or an HTTP date
// measured from `now` and never negative.
export function retryAfterMs(header: string | null, now: number): number {
  const value = header?.trim()
  if (value && /^-?\d+$/.test(value)) {
    const seconds = Number(value)
    return seconds >= 0 ? seconds * 1000 : RETRY_AFTER_FALLBACK_MS
  }
  const retryAt = value ? Date.parse(value) : NaN
  return Number.isNaN(retryAt) ? RETRY_AFTER_FALLBACK_MS : Math.max(0, retryAt - now)
}

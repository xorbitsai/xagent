import {
  KNOWN_ERROR_CODES,
  LINK_CODE_PATTERN,
  MAX_RATE_LIMIT_RETRIES,
  MAX_RATE_LIMIT_WAIT_MS,
  REDEEM_MAX_ATTEMPTS,
  REDEEM_TIMEOUT_MS,
  RENEW_TIMEOUT_MS,
  RETRY_MAX_DELAY_AFTER_TIMEOUT_MS,
  SESSION_REFRESH_THRESHOLD_MS,
  isNonBlankString,
  isRecord,
  isTimestamp,
  reasonForKnownCode,
  retryAfterMs,
  retryDelayMs,
  type LinkSessionFields,
  type WorkerOutboundMessage,
  type WorkerTerminalReason,
} from "./link-session-constants"

// Pure logic of the link session Worker: the redeem and renew requests, their
// retry rules, and the one place that holds the reconnect token.
//
// The reconnect token lives only in a closure variable of
// createLinkSessionWorkerCore. It is never part of any posted message, never
// logged, and never interpolated into an error text: every failure is reported
// as a fixed `terminal` reason. The entry file (link-session-worker.ts) only
// wires this module to the Worker global scope.

const REDEEM_URL = "/v1/external/chat/links/redeem"
const RENEW_URL = "/v1/external/chat/links/reconnect"

export interface LinkSessionWorkerDeps {
  fetch: (input: string, init: RequestInit) => Promise<Response>
  post: (message: WorkerOutboundMessage) => void
  now: () => number
  random: () => number
  setTimer: (callback: () => void, ms: number) => unknown
  clearTimer: (handle: unknown) => void
}

export interface LinkSessionWorkerCore {
  // Posts `hello`. Called once, when the Worker script starts running.
  announce: () => void
  handleMessage: (data: unknown) => void
}

type Phase = "redeem" | "renew"

type RequestOutcome =
  | { kind: "response"; status: number; data: unknown; retryAfter: string | null }
  // The request failed before an answer arrived. `mayHaveReachedServer` is set
  // when the server may have processed it anyway (a timeout).
  | { kind: "transport"; mayHaveReachedServer: boolean }

type Verdict =
  | { kind: "ok"; data: unknown }
  | { kind: "final"; reason: WorkerTerminalReason }
  | { kind: "rate_limited"; waitMs: number }
  | { kind: "retry"; mayHaveReachedServer: boolean }

interface ParsedSession {
  fields: LinkSessionFields
  reconnectToken: string
}

function errorCode(data: unknown): string | null {
  if (!isRecord(data) || !Object.prototype.hasOwnProperty.call(data, "error")) return null
  const error = data.error
  if (!isRecord(error) || !Object.prototype.hasOwnProperty.call(error, "code")) return null
  const code = error.code
  return typeof code === "string" && KNOWN_ERROR_CODES.has(code) ? code : null
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError"
}

// Same acceptance rule as widget.js parseSessionPayload, but the result keeps
// the reconnect token apart from the fields that are allowed to leave the
// Worker. `fields` is assembled key by key, never as "the response minus one
// key".
function parseSessionResponse(data: unknown): ParsedSession | null {
  if (!isRecord(data)) return null
  const session = data.session
  if (
    !isNonBlankString(data.session_token)
    || !isTimestamp(data.session_token_expires_at)
    || !isNonBlankString(data.reconnect_token)
    || !isRecord(session)
    || !isTimestamp(session.absolute_expires_at)
    || !isRecord(session.agent)
  ) {
    return null
  }
  const fields: LinkSessionFields = {
    session_token: data.session_token,
    session_token_expires_at: data.session_token_expires_at,
    absolute_expires_at: session.absolute_expires_at,
    agent: session.agent,
  }
  if (isRecord(session.capabilities)) fields.capabilities = session.capabilities
  return { fields, reconnectToken: data.reconnect_token }
}

function classify(
  phase: Phase,
  outcome: RequestOutcome,
  rateLimitedSoFar: number,
  redeemRetried: boolean,
  now: number,
): Verdict {
  if (outcome.kind === "transport") {
    return { kind: "retry", mayHaveReachedServer: outcome.mayHaveReachedServer }
  }
  const { status, data } = outcome
  const code = errorCode(data)
  // A coded rate_limited follows widget.js's rate-limit rule (widget.js:318-334):
  // the code decides, whatever the status; the request is retried after
  // Retry-After while fewer than MAX_RATE_LIMIT_RETRIES such retries were used
  // and the wait is under MAX_RATE_LIMIT_WAIT_MS, and is final otherwise. A 429
  // without that code is always final. widget.js also keeps that wait's
  // deadline across a bfcache restore; this page ends the session on a restore
  // instead, so it has nothing to keep.
  if (code === "rate_limited") {
    const waitMs = retryAfterMs(outcome.retryAfter, now)
    return rateLimitedSoFar >= MAX_RATE_LIMIT_RETRIES || waitMs >= MAX_RATE_LIMIT_WAIT_MS
      ? { kind: "final", reason: "network" }
      : { kind: "rate_limited", waitMs }
  }
  if (status === 429) return { kind: "final", reason: "network" }
  if (status >= 200 && status < 300) return { kind: "ok", data }
  if (phase === "redeem" && status === 422) return { kind: "final", reason: "link_invalid" }
  if (code !== null) {
    const reason = reasonForKnownCode(code)
    // An earlier attempt of this redeem was retried (a timeout, a transport
    // failure, a 504 or a 5xx without a known code). Such an attempt may have
    // reached the server and spent the single-use code, so a later link_invalid
    // reports that spend: the failure was the network. A 422 is answered above
    // and stays link_invalid, because it is about the request itself.
    if (redeemRetried && reason === "link_invalid") return { kind: "final", reason: "network" }
    return { kind: "final", reason }
  }
  if (status >= 500) return { kind: "retry", mayHaveReachedServer: status === 504 }
  // A 4xx without a known code (for example a 404 `{"detail": ...}` while the
  // endpoint is switched off) never goes away by retrying.
  return { kind: "final", reason: "unavailable" }
}

export function createLinkSessionWorkerCore(deps: LinkSessionWorkerDeps): LinkSessionWorkerCore {
  let started = false
  let terminated = false
  let renewing = false
  let reconnectToken: string | null = null
  // Set while a retry is waiting in backoff; calling it ends the wait now.
  let wakeWait: (() => void) | null = null

  function terminate(reason: WorkerTerminalReason): void {
    if (terminated) return
    terminated = true
    reconnectToken = null
    wakeWait = null
    try {
      deps.post({ type: "terminal", reason })
    } catch {
      // Nothing more can be done from inside the Worker.
    }
  }

  // Any exception that reaches this point is a defect, not a server answer. The
  // error text is dropped: only the fixed reason leaves the Worker.
  function fail(): void {
    terminate("unavailable")
  }

  // A rate-limit wait is not wakeable: the server asked for it (widget.js
  // passes no wake for it).
  function wait(ms: number, wakeable = true): Promise<void> {
    return new Promise((resolve) => {
      const done = () => {
        deps.clearTimer(timer)
        if (wakeable && wakeWait === done) wakeWait = null
        resolve()
      }
      const timer = deps.setTimer(done, ms)
      if (wakeable) wakeWait = done
    })
  }

  // One request with one deadline that covers the fetch and the body read.
  async function send(url: string, body: Record<string, string>, timeoutMs: number): Promise<RequestOutcome> {
    const controller = new AbortController()
    let timer: unknown
    const timeout = new Promise<"timeout">((resolve) => {
      timer = deps.setTimer(() => {
        controller.abort()
        resolve("timeout")
      }, timeoutMs)
    })
    const exchange = (async (): Promise<RequestOutcome> => {
      let response: Response
      try {
        response = await deps.fetch(url, {
          method: "POST",
          credentials: "omit",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        })
      } catch (error) {
        return { kind: "transport", mayHaveReachedServer: isAbortError(error) }
      }
      const status = response.status
      const retryAfter = response.headers.get("Retry-After")
      let text: string
      try {
        text = await response.text()
      } catch (error) {
        return { kind: "transport", mayHaveReachedServer: isAbortError(error) }
      }
      let data: unknown = null
      try {
        data = JSON.parse(text)
      } catch {
        // A complete non-JSON body is judged by its status alone.
      }
      return { kind: "response", status, data, retryAfter }
    })()
    // If the deadline wins, the late result of `exchange` is ignored; keep it
    // from surfacing as an unhandled rejection.
    exchange.catch(() => undefined)
    try {
      const winner = await Promise.race([exchange, timeout])
      return winner === "timeout" ? { kind: "transport", mayHaveReachedServer: true } : winner
    } finally {
      deps.clearTimer(timer)
    }
  }

  function accept(parsed: ParsedSession): void {
    reconnectToken = parsed.reconnectToken
    deps.post({ type: "session", session: parsed.fields })
  }

  async function runRedeem(code: string): Promise<void> {
    let rateLimited = 0
    for (let attempt = 1; ; ) {
      deps.post({ type: "redeeming" })
      const outcome = await send(REDEEM_URL, { code }, REDEEM_TIMEOUT_MS)
      if (terminated) return
      // `attempt` grows only after a retried attempt (rate-limit waits do not
      // count), so `attempt > 1` means an earlier attempt of this redeem was retried.
      const verdict = classify("redeem", outcome, rateLimited, attempt > 1, deps.now())
      if (verdict.kind === "ok") {
        const parsed = parseSessionResponse(verdict.data)
        if (parsed) accept(parsed)
        else terminate("unavailable")
        return
      }
      if (verdict.kind === "final") {
        terminate(verdict.reason)
        return
      }
      if (verdict.kind === "rate_limited") {
        // Counted on its own budget, apart from REDEEM_MAX_ATTEMPTS, like
        // widget.js keeps rateLimited apart from transportRetries.
        rateLimited += 1
        await wait(verdict.waitMs, false)
        if (terminated) return
        continue
      }
      if (attempt >= REDEEM_MAX_ATTEMPTS) {
        terminate("network")
        return
      }
      // The redeem endpoint has no replay window, so there is no shorter cap
      // after a timeout here. If this attempt did spend the code, the next one
      // is answered link_invalid, which classify reports as network.
      await wait(retryDelayMs(attempt, deps.random()))
      if (terminated) return
      attempt += 1
    }
  }

  async function runRenew(): Promise<void> {
    renewing = true
    try {
      let rateLimited = 0
      for (let retries = 0; ; ) {
        const token = reconnectToken
        if (token === null) return
        const outcome = await send(RENEW_URL, { reconnect_token: token }, RENEW_TIMEOUT_MS)
        if (terminated) return
        const verdict = classify("renew", outcome, rateLimited, false, deps.now())
        if (verdict.kind === "ok") {
          const parsed = parseSessionResponse(verdict.data)
          if (
            parsed === null
            || Date.parse(parsed.fields.session_token_expires_at) - deps.now() < SESSION_REFRESH_THRESHOLD_MS
          ) {
            terminate("unavailable")
            return
          }
          accept(parsed)
          return
        }
        if (verdict.kind === "final") {
          terminate(verdict.reason)
          return
        }
        if (verdict.kind === "rate_limited") {
          rateLimited += 1
          deps.post({ type: "degraded", code: "rate_limited" })
          await wait(verdict.waitMs, false)
          if (terminated) return
          continue
        }
        retries += 1
        deps.post({ type: "degraded", code: "network_unavailable" })
        let delay = retryDelayMs(retries, deps.random())
        // The server keeps the previous reconnect token usable for a grace
        // window of at least 30 s after a rotation. A request that timed out (or
        // answered 504) may already have rotated it, so the attempt right after
        // it must still land inside that grace: 5 s timeout + at most 10 s wait +
        // 5 s request = 20 s. Only that attempt is bounded. If it fails again or
        // is rate limited, a later attempt may arrive after the grace; the server
        // then answers reconnect_invalid and the session ends as session_ended,
        // as it does in widget.js.
        if (verdict.mayHaveReachedServer) delay = Math.min(delay, RETRY_MAX_DELAY_AFTER_TIMEOUT_MS)
        await wait(delay)
        if (terminated) return
      }
    } finally {
      renewing = false
    }
  }

  function handleMessage(data: unknown): void {
    try {
      if (terminated || !isRecord(data)) return
      switch (data.type) {
        case "start": {
          if (started) return
          started = true
          const code = data.code
          if (typeof code !== "string" || !LINK_CODE_PATTERN.test(code)) {
            terminate("link_invalid")
            return
          }
          runRedeem(code).catch(fail)
          return
        }
        case "renew":
          if (reconnectToken === null || renewing) return
          runRenew().catch(fail)
          return
        case "wake":
          wakeWait?.()
          return
        default:
          return
      }
    } catch {
      fail()
    }
  }

  return {
    announce: () => deps.post({ type: "hello" }),
    handleMessage,
  }
}

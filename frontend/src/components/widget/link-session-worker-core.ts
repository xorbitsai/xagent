import {
  KNOWN_ERROR_CODES,
  LINK_CODE_PATTERN,
  REDEEM_MAX_ATTEMPTS,
  REDEEM_TIMEOUT_MS,
  RENEW_TIMEOUT_MS,
  RETRY_MAX_DELAY_AFTER_TIMEOUT_MS,
  SESSION_REFRESH_THRESHOLD_MS,
  isNonBlankString,
  isRecord,
  isTimestamp,
  reasonForKnownCode,
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
  | { kind: "response"; status: number; data: unknown }
  // The request failed before an answer arrived. `mayHaveReachedServer` is set
  // when the server may have processed it anyway (a timeout).
  | { kind: "transport"; mayHaveReachedServer: boolean }

type Verdict =
  | { kind: "ok"; data: unknown }
  | { kind: "final"; reason: WorkerTerminalReason }
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

function classify(phase: Phase, outcome: RequestOutcome): Verdict {
  if (outcome.kind === "transport") {
    return { kind: "retry", mayHaveReachedServer: outcome.mayHaveReachedServer }
  }
  const { status, data } = outcome
  const code = errorCode(data)
  // The edge proxy's own 429 carries no error body, so the status decides.
  if (status === 429 || code === "rate_limited") return { kind: "final", reason: "network" }
  if (status >= 200 && status < 300) return { kind: "ok", data }
  if (phase === "redeem" && status === 422) return { kind: "final", reason: "link_invalid" }
  if (code !== null) return { kind: "final", reason: reasonForKnownCode(code) }
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

  function wait(ms: number): Promise<void> {
    return new Promise((resolve) => {
      const done = () => {
        deps.clearTimer(timer)
        if (wakeWait === done) wakeWait = null
        resolve()
      }
      const timer = deps.setTimer(done, ms)
      wakeWait = done
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
      return { kind: "response", status, data }
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
    for (let attempt = 1; ; attempt += 1) {
      deps.post({ type: "redeeming" })
      const outcome = await send(REDEEM_URL, { code }, REDEEM_TIMEOUT_MS)
      if (terminated) return
      const verdict = classify("redeem", outcome)
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
      if (attempt >= REDEEM_MAX_ATTEMPTS) {
        terminate("network")
        return
      }
      // The redeem endpoint has no replay window, so there is no shorter cap
      // after a timeout here.
      await wait(retryDelayMs(attempt, deps.random()))
      if (terminated) return
    }
  }

  async function runRenew(): Promise<void> {
    renewing = true
    try {
      for (let retries = 0; ; ) {
        const token = reconnectToken
        if (token === null) return
        const outcome = await send(RENEW_URL, { reconnect_token: token }, RENEW_TIMEOUT_MS)
        if (terminated) return
        const verdict = classify("renew", outcome)
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
        retries += 1
        deps.post({ type: "degraded", code: "network_unavailable" })
        let delay = retryDelayMs(retries, deps.random())
        // The server keeps the previous reconnect token usable for 30 s after a
        // rotation. A request that timed out (or answered 504) may already have
        // rotated it, so the next attempt must still land inside that grace:
        // 5 s timeout + at most 10 s wait + 5 s request = 20 s.
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

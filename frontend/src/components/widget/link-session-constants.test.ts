import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

import {
  END_REASONS,
  MAX_RAPID_RECOVERIES,
  REDEEM_MAX_ATTEMPTS,
  REDEEM_TIMEOUT_MS,
  RENEW_TIMEOUT_MS,
  RETRY_BACKOFF_FACTOR,
  RETRY_BASE_DELAY_MS,
  RETRY_JITTER,
  RETRY_MAX_DELAY_AFTER_TIMEOUT_MS,
  RETRY_MAX_DELAY_MS,
  SESSION_REFRESH_THRESHOLD_MS,
  STABLE_CONNECTION_MS,
  WORKER_HELLO_TIMEOUT_MS,
  WORKER_REDEEMING_TIMEOUT_MS,
  endedUrl,
  reasonForKnownCode,
  retryDelayMs,
} from "./link-session-constants"

const widgetSource = readFileSync(path.resolve(__dirname, "../../../public/widget.js"), "utf8")

function widgetNumber(pattern: RegExp): number {
  const match = widgetSource.match(pattern)
  if (!match) throw new Error(`widget.js no longer matches ${String(pattern)}`)
  return Number(match[1])
}

// The link page re-implements widget.js's retry rules in TypeScript. These
// tests read the constants out of the widget.js source text, so changing either
// side alone fails here.
describe("link session retry constants equal widget.js", () => {
  it.each([
    ["request timeout", RENEW_TIMEOUT_MS, /var SESSION_TIMEOUT_MS = (\d+)/],
    ["backoff start", RETRY_BASE_DELAY_MS, /baseDelayMs: (\d+)/],
    ["backoff factor", RETRY_BACKOFF_FACTOR, /backoffFactor: ([\d.]+)/],
    ["backoff cap", RETRY_MAX_DELAY_MS, /maxDelayMs: (\d+)/],
    ["wait cap after a timeout", RETRY_MAX_DELAY_AFTER_TIMEOUT_MS, /maxDelayAfterTimeoutMs: (\d+)/],
    ["jitter", RETRY_JITTER, /jitter: ([\d.]+)/],
    ["refresh threshold", SESSION_REFRESH_THRESHOLD_MS, /var SESSION_REFRESH_THRESHOLD_MS = (\d+)/],
    ["stable connection span", STABLE_CONNECTION_MS, /var SESSION_STABLE_CONNECTION_MS = (\d+)/],
    ["rapid recoveries before the breaker trips", MAX_RAPID_RECOVERIES, /state\.rapidRecoveries >= (\d+)/],
  ])("%s", (_label, value, pattern) => {
    expect(value).toBe(widgetNumber(pattern))
  })

  it("computes the same backoff curve as widget.js sessionRetryDelayMs", () => {
    // widget.js: nominal = min(max, base * factor^(attempt-1)); scaled by
    // (1 - jitter + 2 * jitter * random).
    expect(retryDelayMs(1, 0.5)).toBe(1000)
    expect(retryDelayMs(2, 0.5)).toBe(1500)
    expect(retryDelayMs(3, 0.5)).toBe(2250)
    expect(retryDelayMs(1, 0)).toBe(800)
    expect(retryDelayMs(1, 1)).toBe(1200)
    expect(retryDelayMs(40, 0.5)).toBe(300000)
  })
})

// These rules differ from widget.js on purpose.
describe("link session rules that intentionally differ from widget.js", () => {
  it("redeem waits 15 s per attempt where widget.js waits 5 s", () => {
    // The redeem endpoint keeps no replay window, so a request that reached the
    // server but whose answer was slow spends the code; a longer deadline makes
    // that rarer on slow networks.
    expect(REDEEM_TIMEOUT_MS).toBe(15000)
    expect(REDEEM_TIMEOUT_MS).not.toBe(RENEW_TIMEOUT_MS)
  })

  it("redeem gives up after 5 attempts where widget.js retries without limit", () => {
    expect(REDEEM_MAX_ATTEMPTS).toBe(5)
  })

  it("the Worker watchdogs are 30 s for the script and 10 s for the first redeem", () => {
    expect(WORKER_HELLO_TIMEOUT_MS).toBe(30000)
    expect(WORKER_REDEEMING_TIMEOUT_MS).toBe(10000)
  })
})

describe("link session end reasons", () => {
  it("are exactly the five values that make up the contract with embedders", () => {
    expect([...END_REASONS]).toEqual(["link_invalid", "session_ended", "unavailable", "network", "closed"])
  })

  it.each([...END_REASONS])("builds the end page URL for %s", (reason) => {
    expect(endedUrl(reason)).toBe(`/widget/chat/link/ended?reason=${reason}`)
  })

  it.each([
    ["link_invalid", "link_invalid"],
    ["session_expired", "session_ended"],
    ["reconnect_invalid", "session_ended"],
    ["identity_mismatch", "session_ended"],
    ["rate_limited", "network"],
    ["agent_not_available", "unavailable"],
    ["widget_disabled", "unavailable"],
    ["invalid_input", "unavailable"],
  ])("maps the server code %s to %s", (code, reason) => {
    expect(reasonForKnownCode(code)).toBe(reason)
  })
})

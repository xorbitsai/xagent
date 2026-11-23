import { readFileSync } from "node:fs"
import path from "node:path"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  END_REASONS,
  isEndReason,
  isWorkerTerminalReason,
  KNOWN_ERROR_CODES,
  MAX_RAPID_RECOVERIES,
  MAX_RATE_LIMIT_RETRIES,
  MAX_RATE_LIMIT_WAIT_MS,
  REDEEM_MAX_ATTEMPTS,
  REDEEM_TIMEOUT_MS,
  RENEW_TIMEOUT_MS,
  RETRY_AFTER_FALLBACK_MS,
  RETRY_BACKOFF_FACTOR,
  RETRY_BASE_DELAY_MS,
  RETRY_JITTER,
  RETRY_MAX_DELAY_AFTER_TIMEOUT_MS,
  RETRY_MAX_DELAY_MS,
  SESSION_REFRESH_THRESHOLD_MS,
  STABLE_CONNECTION_MS,
  retryAfterMs,
  retryDelayMs,
} from "./link-session-constants"

const widgetSource = readFileSync(path.resolve(__dirname, "../../../public/widget.js"), "utf8")

function widgetSnippet(pattern: RegExp): string {
  const match = widgetSource.match(pattern)
  if (!match) throw new Error(`widget.js no longer matches ${String(pattern)}`)
  return match[0]
}

function widgetNumber(pattern: RegExp): number {
  const match = widgetSource.match(pattern)
  if (!match) throw new Error(`widget.js no longer matches ${String(pattern)}`)
  return Number(match[1])
}

// The link page re-implements parts of widget.js in TypeScript. These tests
// compare each copied value, the error-code set, and the sessionRetryDelayMs
// and retryAfterMs functions with the widget.js source text, so changing one
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
    ["rate-limit retries", MAX_RATE_LIMIT_RETRIES, /maxRateLimitRetries: (\d+)/],
    ["rate-limit wait cap", MAX_RATE_LIMIT_WAIT_MS, /maxRateLimitWaitMs: (\d+)/],
    ["Retry-After fallback", RETRY_AFTER_FALLBACK_MS, /var SESSION_RETRY_AFTER_FALLBACK_MS = (\d+)/],
  ])("%s", (_label, value, pattern) => {
    expect(value).toBe(widgetNumber(pattern))
  })

  it("computes the same backoff curve as widget.js sessionRetryDelayMs", () => {
    const policy = widgetSnippet(/var SESSION_RETRY_POLICY = \{[\s\S]*?\n  \};/)
    const fn = widgetSnippet(/function sessionRetryDelayMs\(policy, attempt, random\) \{[\s\S]*?\n  \}/)
    const widgetDelay = new Function(
      `${policy}\n${fn}\nreturn function (a, r) { return sessionRetryDelayMs(SESSION_RETRY_POLICY, a, r) }`,
    )() as (attempt: number, random: number) => number
    for (const attempt of [1, 2, 3, 5, 10, 20, 40]) {
      for (const random of [0, 0.25, 0.5, 0.75, 0.999999]) {
        expect(retryDelayMs(attempt, random), `attempt ${attempt}, random ${random}`)
          .toBe(widgetDelay(attempt, random))
      }
    }
  })

  describe("Retry-After parsing", () => {
    afterEach(() => vi.useRealTimers())

    it("reads every header value the way widget.js retryAfterMs does", () => {
      vi.useFakeTimers()
      vi.setSystemTime(new Date("2026-01-01T00:00:00Z"))
      const fn = widgetSnippet(/function retryAfterMs\(result\) \{[\s\S]*?\n  \}/)
      const widgetRetryAfter = new Function(
        "SESSION_RETRY_AFTER_FALLBACK_MS",
        `${fn}\nreturn retryAfterMs`,
      )(widgetNumber(/var SESSION_RETRY_AFTER_FALLBACK_MS = (\d+)/)) as (result: { retryAfter: string | null }) => number
      const now = Date.now()
      const headers = [
        null, "", "  ", "0", "5", " 12 ", "-3", "29", "30", "1.5", "abc",
        new Date(now + 10_000).toUTCString(),
        new Date(now - 10_000).toUTCString(),
      ]
      for (const header of headers) {
        expect(retryAfterMs(header, Date.now()), `header ${JSON.stringify(header)}`)
          .toBe(widgetRetryAfter({ retryAfter: header }))
      }
    })
  })

  it("knows exactly the error codes of widget.js plus link_invalid", () => {
    const table = widgetSnippet(/var SESSION_ERROR_CODES = \{[\s\S]*?\};/)
    const widgetCodes = [...table.matchAll(/(\w+): true/g)].map((match) => match[1])
    expect(widgetCodes).not.toContain("link_invalid")
    expect([...KNOWN_ERROR_CODES].filter((code) => code !== "link_invalid").sort())
      .toEqual([...widgetCodes].sort())
    expect(KNOWN_ERROR_CODES.has("link_invalid")).toBe(true)
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
})

describe("link session end reasons", () => {
  it("are exactly the five values that make up the contract with embedders", () => {
    expect([...END_REASONS]).toEqual(["link_invalid", "session_ended", "unavailable", "network", "closed"])
  })

  it.each([...END_REASONS])("isEndReason accepts %s", (reason) => {
    expect(isEndReason(reason)).toBe(true)
  })

  it.each([null, undefined, 1, "", "CLOSED", "unknown_x"])("isEndReason rejects %s", (value) => {
    expect(isEndReason(value)).toBe(false)
  })

  it("isWorkerTerminalReason accepts every end reason except closed", () => {
    for (const reason of END_REASONS) expect(isWorkerTerminalReason(reason)).toBe(reason !== "closed")
    expect(isWorkerTerminalReason("unknown_x")).toBe(false)
    expect(isWorkerTerminalReason(null)).toBe(false)
  })
})

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AUTH_CACHE_KEY, markIdleActivityReportedIfCurrent, readAuthCache, readAuthSessionSnapshot, readCurrentIdleSession } from "@/lib/auth-cache"
import { startIdleActivityReporter } from "@/lib/idle-activity"
import { IDLE_SESSION_STORAGE_KEY } from "@/lib/idle-session"
import { reportIdleActivity } from "@/lib/idle-session-client"

/*
 * The real reporter, client and api-wrapper together; only fetch is faked.
 * The activity endpoint carries no input time and credits arrival time, so a
 * request that leaves the browser late must not be sent for old input.
 */

const SESSION_ID = "browser-login"
const isActivity = (input: unknown) => String(input).endsWith("/api/auth/activity")

function login() {
  const now = Date.now()
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
    schemaVersion: 2, sessionId: SESSION_ID, credentialRevision: 0, profileRevision: 0,
    user: { id: "1", username: "alice" }, token: "access", refreshToken: "refresh", timestamp: now,
    expiresAt: now + 3_600_000, refreshExpiresAt: now + 7_200_000,
  }))
  localStorage.setItem(IDLE_SESSION_STORAGE_KEY, JSON.stringify({
    schemaVersion: 1, sessionId: SESSION_ID, enforced: true, sid: "sid-a", idleTimeoutSeconds: 1800, idleExpiresIn: 1700,
    absoluteExpiresIn: null, activityMinIntervalSeconds: 60, observedAt: now, lastReportAt: null,
  }))
}

const started: Array<{ stop: () => void }> = []

function startReporter() {
  const reporter = startIdleActivityReporter({
    target: window,
    isUserEvent: () => true,
    getRecord: () => readCurrentIdleSession(),
    markReported: () => markIdleActivityReportedIfCurrent(readAuthSessionSnapshot()),
    report: isStillFresh => reportIdleActivity(SESSION_ID, isStillFresh),
  })
  started.push(reporter)
  return reporter
}

describe("activity report transport", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-11-13T10:00:00Z"))
    localStorage.clear()
    vi.restoreAllMocks()
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { request: vi.fn(async (_: string, action: () => Promise<unknown>) => action()) },
    })
    login()
  })
  afterEach(() => {
    for (const reporter of started.splice(0)) reporter.stop()
    vi.useRealTimers()
  })

  it("does not send a transport retry once the input it stands for is stale", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async input => {
      if (isActivity(input)) throw new TypeError("Failed to fetch")
      return new Response(null, { status: 404 })
    })
    startReporter()

    window.dispatchEvent(new Event("keydown"))
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock.mock.calls.filter(([input]) => isActivity(input))).toHaveLength(1)

    // The browser suspends before the retry timer fires and resumes minutes later.
    vi.setSystemTime(Date.now() + 5 * 60_000)
    await vi.advanceTimersByTimeAsync(10_000)

    expect(fetchMock.mock.calls.filter(([input]) => isActivity(input))).toHaveLength(1)
    // Not sending is not a sign-out: the login stays.
    expect(readAuthCache()?.token).toBe("access")
    // The dropped report is finished, not queued for a later retry.
    await vi.advanceTimersByTimeAsync(2 * 60_000)
    expect(fetchMock.mock.calls.filter(([input]) => isActivity(input))).toHaveLength(1)
  })

  it("still retries a lost connection while the input is recent", async () => {
    let attempts = 0
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async input => {
      if (!isActivity(input)) return new Response(null, { status: 404 })
      attempts += 1
      if (attempts === 1) throw new TypeError("Failed to fetch")
      return new Response(JSON.stringify({}), { headers: { "Content-Type": "application/json" } })
    })
    startReporter()

    window.dispatchEvent(new Event("keydown"))
    await vi.advanceTimersByTimeAsync(1_000)

    expect(fetchMock.mock.calls.filter(([input]) => isActivity(input))).toHaveLength(2)
  })
})

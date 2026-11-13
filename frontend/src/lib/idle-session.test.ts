import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  AUTH_CACHE_KEY, claimAuthLoginIntent, clearAuthSessionIfCurrent, clearStoredAuth, commitAuthSessionRefresh,
  createAuthSession, endExpiredAuthSession, markIdleActivityReportedIfCurrent, readCurrentIdleSession,
  recordIdleSessionIfCurrent, type AuthSessionSnapshot,
} from "@/lib/auth-cache"
import {
  IDLE_SESSION_STORAGE_KEY, SESSION_END_NOTICE_KEY, SESSION_END_NOTICE_TTL_MS, idleRemainingMs,
  isIdleSessionExpiredResponse, parseIdleSession, readSessionEndNotice, writeSessionEndNotice,
} from "@/lib/idle-session"

const user = { id: "1", username: "alice", email: null, is_admin: false }
const idle = (overrides: Record<string, unknown> = {}) => ({
  enforced: true, sid: "sid-a", idle_timeout_seconds: 1800, idle_expires_in: 1734,
  server_time: "2026-11-13T10:00:00Z", absolute_expires_in: null, activity_min_interval_seconds: 60,
  ...overrides,
})
function installLock() {
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: { request: vi.fn(async (_: string, action: () => Promise<unknown>) => action()) },
  })
}
async function login(body: Record<string, unknown> = {}): Promise<AuthSessionSnapshot> {
  const claim = await claimAuthLoginIntent()
  if (claim.status !== "claimed") throw new Error("expected login intent")
  const result = await createAuthSession(
    { user, access_token: "access", refresh_token: "refresh", expires_in: 7200, refresh_expires_in: 604800, ...body },
    claim.intent,
  )
  if (result.status !== "created") throw new Error("expected creation")
  return result.projection.snapshot
}
const refreshBody = (body: Record<string, unknown> = {}) => ({
  success: true, access_token: "access-2", refresh_token: "refresh-2", expires_in: 7200, refresh_expires_in: 604800, ...body,
})

describe("idle session policy", () => {
  beforeEach(() => { localStorage.clear(); vi.useRealTimers(); installLock() })

  it("parses the top-level idle_session object and treats an absent or malformed one as unsupported", () => {
    expect(parseIdleSession({ idle_session: idle() })).toEqual({
      enforced: true, sid: "sid-a", idleTimeoutSeconds: 1800, idleExpiresIn: 1734,
      absoluteExpiresIn: null, activityMinIntervalSeconds: 60,
    })
    // A legacy token while the switch is off has no server login yet.
    expect(parseIdleSession({ idle_session: idle({ enforced: false, sid: null }) })?.sid).toBeNull()
    expect(parseIdleSession({ access_token: "x" })).toBeNull()
    expect(parseIdleSession({ data: { idle_session: idle() } })).toBeNull()
    for (const bad of [
      { enforced: "true" }, { idle_timeout_seconds: 0 }, { idle_expires_in: -1 },
      { activity_min_interval_seconds: 0 }, { absolute_expires_in: "soon" }, { sid: "" },
    ]) expect(parseIdleSession({ idle_session: idle(bad) })).toBeNull()
  })

  it("measures the remaining time from the server's seconds, not from a client deadline", () => {
    const record = { ...parseIdleSession({ idle_session: idle({ idle_expires_in: 100 }) })!, observedAt: 1_000_000 }
    expect(idleRemainingMs(record, 1_000_000)).toBe(100_000)
    expect(idleRemainingMs(record, 1_040_000)).toBe(60_000)
    // A clock that moved backwards never adds time.
    expect(idleRemainingMs(record, 900_000)).toBe(100_000)
  })

  it("recognizes only the idle-expiry 401 as a sign-out signal", () => {
    const response = (status: number, errorType?: string) =>
      new Response(null, { status, headers: errorType ? { "Error-Type": errorType } : {} })
    expect(isIdleSessionExpiredResponse(response(401, "SessionIdleExpired"))).toBe(true)
    expect(isIdleSessionExpiredResponse(response(401, "TokenExpired"))).toBe(false)
    expect(isIdleSessionExpiredResponse(response(503, "SessionIdleExpired"))).toBe(false)
  })

  it("keeps a sign-in-again notice for a limited time", () => {
    writeSessionEndNotice(localStorage, "session_expired", 5_000)
    expect(readSessionEndNotice(localStorage, 5_000 + SESSION_END_NOTICE_TTL_MS)).toBe("session_expired")
    expect(readSessionEndNotice(localStorage, 5_001 + SESSION_END_NOTICE_TTL_MS)).toBeNull()
    localStorage.setItem(SESSION_END_NOTICE_KEY, "{bad")
    expect(readSessionEndNotice(localStorage)).toBeNull()
  })
})

describe("idle session state bound to one login", () => {
  beforeEach(() => { localStorage.clear(); installLock() })

  it("stores the login response policy for the new login and drops an older notice", async () => {
    writeSessionEndNotice(localStorage, "session_expired")
    const snapshot = await login({ idle_session: idle() })
    expect(readCurrentIdleSession()).toMatchObject({ sessionId: snapshot.sessionId, sid: "sid-a", enforced: true, lastReportAt: null })
    expect(localStorage.getItem(SESSION_END_NOTICE_KEY)).toBeNull()
  })

  it("keeps the existing behavior for a server without idle_session", async () => {
    await login()
    expect(readCurrentIdleSession()).toBeNull()
  })

  it("never lets a new login read the idle state of the previous one", async () => {
    const first = await login({ idle_session: idle() })
    markIdleActivityReportedIfCurrent(first)
    expect(readCurrentIdleSession()?.lastReportAt).not.toBeNull()
    await login()
    expect(readCurrentIdleSession()).toBeNull()
    expect(localStorage.getItem(IDLE_SESSION_STORAGE_KEY)).toBeNull()
  })

  it("does not hand the stored idle state to a different browser login", async () => {
    const snapshot = await login({ idle_session: idle() })
    const stored = JSON.parse(localStorage.getItem(IDLE_SESSION_STORAGE_KEY)!)
    localStorage.setItem(IDLE_SESSION_STORAGE_KEY, JSON.stringify({ ...stored, sessionId: "another-login" }))
    expect(readCurrentIdleSession()).toBeNull()
    expect(recordIdleSessionIfCurrent(snapshot, { idle_session: idle() })?.sessionId).toBe(snapshot.sessionId)
  })

  it("updates the policy from a refresh and keeps another tab's report time for the same server login", async () => {
    const snapshot = await login({ idle_session: idle() })
    markIdleActivityReportedIfCurrent(snapshot)
    const reportedAt = readCurrentIdleSession()!.lastReportAt
    const result = await commitAuthSessionRefresh(snapshot, refreshBody({ idle_session: idle({ idle_expires_in: 900 }) }))
    expect(result.status).toBe("updated")
    expect(readCurrentIdleSession()).toMatchObject({ idleExpiresIn: 900, lastReportAt: reportedAt })
  })

  it("starts a fresh report history when a legacy login receives its first server login id", async () => {
    const snapshot = await login({ idle_session: idle({ enforced: false, sid: null }) })
    markIdleActivityReportedIfCurrent(snapshot)
    await commitAuthSessionRefresh(snapshot, refreshBody({ idle_session: idle({ enforced: false, sid: "sid-new" }) }))
    expect(readCurrentIdleSession()).toMatchObject({ sid: "sid-new", lastReportAt: null })
  })

  it("ignores a verify or activity response that arrives after another login replaced this one", async () => {
    const first = await login({ idle_session: idle() })
    const second = await login({ idle_session: idle({ sid: "sid-b" }) })
    expect(recordIdleSessionIfCurrent(first, { idle_session: idle({ idle_expires_in: 5 }) })).toBeNull()
    markIdleActivityReportedIfCurrent(first)
    expect(readCurrentIdleSession()).toMatchObject({ sessionId: second.sessionId, sid: "sid-b", lastReportAt: null })
    expect(recordIdleSessionIfCurrent(second, { idle_session: idle({ sid: "sid-b", idle_expires_in: 5 }) })?.idleExpiresIn).toBe(5)
  })

  it.each([
    ["conditional clear", (snapshot: AuthSessionSnapshot) => clearAuthSessionIfCurrent(snapshot)],
    ["logout", () => clearStoredAuth()],
    ["idle expiry", (snapshot: AuthSessionSnapshot) => endExpiredAuthSession(snapshot)],
  ])("removes the idle state together with the credentials on %s", async (_, clear) => {
    const snapshot = await login({ idle_session: idle() })
    await clear(snapshot)
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
    expect(localStorage.getItem(IDLE_SESSION_STORAGE_KEY)).toBeNull()
  })

  it("leaves a sign-in-again notice only when idle expiry cleared the exact login", async () => {
    const first = await login({ idle_session: idle() })
    const second = await login({ idle_session: idle({ sid: "sid-b" }) })
    expect(await endExpiredAuthSession(first)).toEqual({ status: "not_current" })
    expect(readCurrentIdleSession()?.sessionId).toBe(second.sessionId)
    expect(readSessionEndNotice(localStorage)).toBeNull()
    expect(await endExpiredAuthSession(second)).toEqual({ status: "cleared" })
    expect(readSessionEndNotice(localStorage)).toBe("session_expired")
  })
})

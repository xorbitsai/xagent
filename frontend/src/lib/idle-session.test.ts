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
const emptySnapshot: AuthSessionSnapshot = {
  sessionId: null, credentialRevision: null, profileRevision: null, userId: null,
  accessToken: null, refreshToken: null, profileFingerprint: null,
}
const refreshBody = (body: Record<string, unknown> = {}) => ({
  success: true, access_token: "access-2", refresh_token: "refresh-2", expires_in: 7200, refresh_expires_in: 604800, ...body,
})

describe("idle session policy", () => {
  beforeEach(() => { localStorage.clear(); vi.useRealTimers(); installLock() })

  it("parses the top-level idle_session object and treats an absent or malformed one as unsupported", () => {
    expect(parseIdleSession({ idle_session: idle() })).toEqual({
      enforced: true, sid: "sid-a", idleTimeoutSeconds: 1800, idleExpiresIn: 1734,
      absoluteExpiresIn: null, activityMinIntervalSeconds: 60, serverTime: Date.parse("2026-11-13T10:00:00Z"),
    })
    // A legacy token while the switch is off has no server login yet.
    expect(parseIdleSession({ idle_session: idle({ enforced: false, sid: null }) })?.sid).toBeNull()
    expect(parseIdleSession({ access_token: "x" })).toBeNull()
    expect(parseIdleSession(null)).toBeNull()
    // The server time only orders responses; an unreadable one is not a reason to drop the policy.
    expect(parseIdleSession({ idle_session: idle({ server_time: "not a time" }) })?.serverTime).toBeNull()
    expect(parseIdleSession({ idle_session: idle({ server_time: 1_700_000_000_000 }) })?.serverTime).toBe(1_700_000_000_000)
    expect(parseIdleSession({ idle_session: "on" })).toBeNull()
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
    localStorage.setItem(SESSION_END_NOTICE_KEY, JSON.stringify({ reason: "something_else", at: Date.now() }))
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
    await markIdleActivityReportedIfCurrent(first)
    expect(readCurrentIdleSession()?.lastReportAt).not.toBeNull()
    await login()
    expect(readCurrentIdleSession()).toBeNull()
    expect(localStorage.getItem(IDLE_SESSION_STORAGE_KEY)).toBeNull()
  })

  it("has no idle state without a current login", async () => {
    expect(readCurrentIdleSession()).toBeNull()
    await markIdleActivityReportedIfCurrent({ ...emptySnapshot })
    expect(localStorage.getItem(IDLE_SESSION_STORAGE_KEY)).toBeNull()
  })

  it.each([
    ["not JSON", "{bad"],
    ["not an object", "5"],
    ["an unknown schema", { schemaVersion: 2 }],
    ["an invalid observation time", { observedAt: -1 }],
    ["an invalid report time", { lastReportAt: "now" }],
    ["an invalid policy", { idleTimeoutSeconds: 0 }],
  ])("treats a stored record with %s as absent", async (_, value) => {
    const snapshot = await login({ idle_session: idle() })
    const stored = JSON.parse(localStorage.getItem(IDLE_SESSION_STORAGE_KEY)!)
    localStorage.setItem(IDLE_SESSION_STORAGE_KEY, typeof value === "string" ? value : JSON.stringify({ ...stored, ...value }))
    expect(readCurrentIdleSession()).toBeNull()
    // Reporting against an unreadable record writes nothing.
    await markIdleActivityReportedIfCurrent(snapshot)
    expect(readCurrentIdleSession()).toBeNull()
  })

  it("ignores an unreadable response body, and records best effort when storage refuses", async () => {
    const snapshot = await login({ idle_session: idle() })
    expect(await recordIdleSessionIfCurrent(snapshot, null)).toBeNull()
    const setItem = vi.spyOn(localStorage, "setItem").mockImplementation(() => { throw new Error("quota") })
    try {
      expect(await recordIdleSessionIfCurrent(snapshot, { idle_session: idle({ idle_expires_in: 1800 }) })).toBeNull()
      await expect(markIdleActivityReportedIfCurrent(snapshot)).resolves.toBeUndefined()
    } finally {
      setItem.mockRestore()
    }
    expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1734)
  })

  it("forgets the idle state when a successful response says the server no longer supports it", async () => {
    const snapshot = await login({ idle_session: idle() })
    expect(await recordIdleSessionIfCurrent(snapshot, { success: true })).toBeNull()
    expect(readCurrentIdleSession()).toBeNull()
    const again = await login({ idle_session: idle() })
    await commitAuthSessionRefresh(again, refreshBody())
    expect(readCurrentIdleSession()).toBeNull()
  })

  it("keeps a newer stricter observation when an older, looser response arrives late", async () => {
    // A rolling change tightened the threshold; a verify produced before it arrives afterwards.
    const snapshot = await login({ idle_session: idle({ idle_timeout_seconds: 900, idle_expires_in: 890, server_time: "2026-11-13T10:00:10Z" }) })
    const older = idle({ idle_timeout_seconds: 1800, idle_expires_in: 1795, server_time: "2026-11-13T10:00:05Z" })
    expect((await recordIdleSessionIfCurrent(snapshot, { idle_session: older }))?.idleTimeoutSeconds).toBe(900)
  })

  it("orders responses by the full microsecond server time", async () => {
    // Same millisecond, different microseconds; both describe a 10:02:00.123456 deadline.
    const snapshot = await login({ idle_session: idle({ idle_expires_in: 109, server_time: "2026-11-13T10:00:10.123900Z" }) })
    const older = idle({ idle_expires_in: 110, server_time: "2026-11-13T10:00:10.123100Z" })
    expect((await recordIdleSessionIfCurrent(snapshot, { idle_session: older }))?.idleExpiresIn).toBe(109)
  })

  it("keeps the newer observation when an older response rounds to the same deadline", async () => {
    // Both describe a 10:02:00 deadline: idle_expires_in is truncated, server_time is not.
    const snapshot = await login({ idle_session: idle({ idle_expires_in: 109, server_time: "2026-11-13T10:00:10.100Z" }) })
    const older = idle({ idle_expires_in: 119, server_time: "2026-11-13T10:00:00.900Z" })
    expect((await recordIdleSessionIfCurrent(snapshot, { idle_session: older }))?.serverTime).toBe(Date.parse("2026-11-13T10:00:10.100Z"))
    expect(readCurrentIdleSession()?.idleExpiresIn).toBe(109)
  })

  it("waits for the auth lock, and does not touch a login another tab committed meanwhile", async () => {
    const first = await login({ idle_session: idle() })
    const before = localStorage.getItem(IDLE_SESSION_STORAGE_KEY)
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { request: vi.fn((_: string, action: () => Promise<unknown>) => held.then(action)) },
    })
    const recorded = recordIdleSessionIfCurrent(first, { idle_session: idle({ server_time: "2026-11-13T10:05:00Z" }) })
    const marked = markIdleActivityReportedIfCurrent(first)
    // Nothing is written while another holder has the lock.
    expect(localStorage.getItem(IDLE_SESSION_STORAGE_KEY)).toBe(before)
    // Another tab commits login B under that lock.
    const cache = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY)!)
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({ ...cache, sessionId: "login-b", token: "access-b" }))
    const recordB = { ...JSON.parse(before!), sessionId: "login-b", sid: "sid-b", lastReportAt: null }
    localStorage.setItem(IDLE_SESSION_STORAGE_KEY, JSON.stringify(recordB))
    release()
    expect(await recorded).toBeNull()
    await marked
    expect(JSON.parse(localStorage.getItem(IDLE_SESSION_STORAGE_KEY)!)).toEqual(recordB)
  })

  it("ignores a response the server produced before the stored one", async () => {
    const snapshot = await login({ idle_session: idle({ idle_expires_in: 1800, server_time: "2026-11-13T10:00:30Z" }) })
    // A verify sent before the activity report arrives after it.
    const late = idle({ idle_expires_in: 1770, server_time: "2026-11-13T10:00:00Z" })
    expect((await recordIdleSessionIfCurrent(snapshot, { idle_session: late }))?.idleExpiresIn).toBe(1800)
    expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1800)
  })

  it.each([
    // Stored at 10:00:30 (deadline 10:30:30). Each response arrives within a minute.
    ["a newer response with a shorter threshold", "2026-11-13T10:01:00Z",
      { idle_timeout_seconds: 900, idle_expires_in: 900, server_time: "2026-11-13T10:01:00Z" }],
    // The client clock moved back 30 s after 40 s; the server deadline is unchanged.
    ["a newer response after the client clock moved backwards", "2026-11-13T10:00:40Z",
      { idle_expires_in: 1760, server_time: "2026-11-13T10:01:10Z" }],
    ["a response without a server time", "2026-11-13T10:00:40Z",
      { idle_expires_in: 1700, server_time: undefined }],
    ["a change of switch state", "2026-11-13T10:00:40Z",
      { enforced: false, idle_expires_in: 1700, server_time: "2026-11-13T10:00:00Z" }],
  ])("takes %s as is", async (_, receivedAt, overrides) => {
    vi.useFakeTimers({ now: new Date("2026-11-13T10:00:30Z") })
    try {
      const snapshot = await login({ idle_session: idle({ idle_expires_in: 1800, server_time: "2026-11-13T10:00:30Z" }) })
      vi.setSystemTime(new Date(receivedAt))
      const next = idle(overrides)
      expect((await recordIdleSessionIfCurrent(snapshot, { idle_session: next }))?.idleExpiresIn).toBe(next.idle_expires_in)
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not hand the stored idle state to a different browser login", async () => {
    const snapshot = await login({ idle_session: idle() })
    const stored = JSON.parse(localStorage.getItem(IDLE_SESSION_STORAGE_KEY)!)
    localStorage.setItem(IDLE_SESSION_STORAGE_KEY, JSON.stringify({ ...stored, sessionId: "another-login" }))
    expect(readCurrentIdleSession()).toBeNull()
    expect((await recordIdleSessionIfCurrent(snapshot, { idle_session: idle() }))?.sessionId).toBe(snapshot.sessionId)
  })

  it("updates the policy from a refresh and keeps another tab's report time for the same server login", async () => {
    const snapshot = await login({ idle_session: idle() })
    await markIdleActivityReportedIfCurrent(snapshot)
    const reportedAt = readCurrentIdleSession()!.lastReportAt
    const result = await commitAuthSessionRefresh(snapshot, refreshBody({ idle_session: idle({ idle_expires_in: 1800 }) }))
    expect(result.status).toBe("updated")
    expect(readCurrentIdleSession()).toMatchObject({ idleExpiresIn: 1800, lastReportAt: reportedAt })
  })

  it("starts a fresh report history when a legacy login receives its first server login id", async () => {
    const snapshot = await login({ idle_session: idle({ enforced: false, sid: null }) })
    await markIdleActivityReportedIfCurrent(snapshot)
    await commitAuthSessionRefresh(snapshot, refreshBody({ idle_session: idle({ enforced: false, sid: "sid-new" }) }))
    expect(readCurrentIdleSession()).toMatchObject({ sid: "sid-new", lastReportAt: null })
  })

  it("ignores a verify or activity response that arrives after another login replaced this one", async () => {
    const first = await login({ idle_session: idle() })
    const second = await login({ idle_session: idle({ sid: "sid-b" }) })
    expect(await recordIdleSessionIfCurrent(first, { idle_session: idle({ idle_expires_in: 5 }) })).toBeNull()
    await markIdleActivityReportedIfCurrent(first)
    expect(readCurrentIdleSession()).toMatchObject({ sessionId: second.sessionId, sid: "sid-b", lastReportAt: null })
    expect((await recordIdleSessionIfCurrent(second, { idle_session: idle({ sid: "sid-b", idle_expires_in: 1800 }) }))?.idleExpiresIn).toBe(1800)
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

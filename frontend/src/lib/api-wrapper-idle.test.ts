import { beforeEach, describe, expect, it, vi } from "vitest"

import { apiRequest, refreshStoredAccessToken, RequestNotSentError } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY, readAuthCache, readAuthSessionSnapshot, readCurrentIdleSession } from "@/lib/auth-cache"
import { readSessionEndNotice } from "@/lib/idle-session"

const user = { id: "1", username: "alice", email: null, is_admin: false }

function writeAuthCache(token: string, refreshToken: string, sessionId = `test-${Math.random()}`) {
  const now = Date.now()
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
    schemaVersion: 2, sessionId, credentialRevision: 0, profileRevision: 0,
    user, token, refreshToken, timestamp: now, expiresAt: now + 120_000, refreshExpiresAt: now + 240_000,
  }))
}
function installLocks() {
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: { request: vi.fn(async (_: string, callback: () => Promise<unknown>) => callback()) },
  })
}
const idleExpired = () => new Response(null, { status: 401, headers: { "Error-Type": "SessionIdleExpired" } })
const isRefresh = (input: unknown) => String(input).endsWith("/api/auth/refresh")

describe("api-wrapper idle expiry", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
    installLocks()
  })

  it("signs out on an idle-expired response without refreshing or replaying", async () => {
    writeAuthCache("access", "refresh")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => idleExpired())

    const response = await apiRequest("http://api.local/protected", { method: "POST" })

    expect(response.status).toBe(401)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls.some(([input]) => isRefresh(input))).toBe(false)
    expect(readAuthCache()).toBeNull()
    expect(readSessionEndNotice(localStorage)).toBe("session_expired")
  })

  it("signs out when the refresh behind an expired access token reports idle expiry", async () => {
    writeAuthCache("old-access", "old-refresh")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async input =>
      isRefresh(input) ? idleExpired() : new Response(null, { status: 401, headers: { "Error-Type": "TokenExpired" } }))

    const response = await apiRequest("http://api.local/protected", { method: "POST" })

    expect(response.status).toBe(401)
    // The protected request is sent once and never replayed.
    expect(fetchMock.mock.calls.filter(([input]) => !isRefresh(input))).toHaveLength(1)
    expect(readAuthCache()).toBeNull()
    expect(readSessionEndNotice(localStorage)).toBe("session_expired")
  })

  it("signs out when the replay after a refresh reports idle expiry", async () => {
    writeAuthCache("old-access", "old-refresh")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
      if (isRefresh(input)) {
        return new Response(JSON.stringify({ success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: 120, refresh_expires_in: 240 }), { headers: { "Content-Type": "application/json" } })
      }
      return new Headers(options?.headers).get("Authorization") === "Bearer new-access"
        ? idleExpired()
        : new Response(null, { status: 401, headers: { "Error-Type": "TokenExpired" } })
    })

    const response = await apiRequest("http://api.local/protected")

    expect(response.status).toBe(401)
    // One original request and one replay; the replay is not repeated.
    expect(fetchMock.mock.calls.filter(([input]) => !isRefresh(input))).toHaveLength(2)
    expect(readAuthCache()).toBeNull()
    expect(readSessionEndNotice(localStorage)).toBe("session_expired")
  })

  it("leaves a login that replaced the replayed one in place", async () => {
    writeAuthCache("old-access", "old-refresh", "old-login")
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
      if (isRefresh(input)) {
        return new Response(JSON.stringify({ success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: 120, refresh_expires_in: 240 }), { headers: { "Content-Type": "application/json" } })
      }
      if (new Headers(options?.headers).get("Authorization") === "Bearer new-access") {
        // Another user signs in while the replay is in flight.
        writeAuthCache("other-access", "other-refresh", "other-login")
        return idleExpired()
      }
      return new Response(null, { status: 401, headers: { "Error-Type": "TokenExpired" } })
    })

    await apiRequest("http://api.local/protected")

    expect(readAuthCache()?.token).toBe("other-access")
    expect(readSessionEndNotice(localStorage)).toBeNull()
  })

  it("signs out when the replay with another tab's rotated token reports idle expiry", async () => {
    writeAuthCache("old-access", "old-refresh", "same-login")
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, options) => {
      if (new Headers(options?.headers).get("Authorization") === "Bearer rotated-access") return idleExpired()
      // Another tab rotated the same login's credentials while this request was in flight.
      const cache = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY)!)
      localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({ ...cache, credentialRevision: 1, token: "rotated-access", refreshToken: "rotated-refresh" }))
      return new Response(null, { status: 401, headers: { "Error-Type": "TokenExpired" } })
    })

    const response = await apiRequest("http://api.local/protected")

    expect(response.status).toBe(401)
    expect(readAuthCache()).toBeNull()
    expect(readSessionEndNotice(localStorage)).toBe("session_expired")
  })

  it.each([
    ["after a refresh", "old-access"],
    ["with another tab's rotated token", "rotated"],
  ])("skips the replay %s when the caller says the request is no longer wanted", async (_, scenario) => {
    writeAuthCache("old-access", "old-refresh", "same-login")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async input => {
      if (isRefresh(input)) {
        return new Response(JSON.stringify({ success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: 120, refresh_expires_in: 240 }), { headers: { "Content-Type": "application/json" } })
      }
      if (scenario === "rotated") {
        const cache = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY)!)
        localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({ ...cache, credentialRevision: 1, token: "rotated-access", refreshToken: "rotated-refresh" }))
      }
      return new Response(null, { status: 401, headers: { "Error-Type": "TokenExpired" } })
    })
    // Wanted for the first attempt, no longer wanted by the time of the replay.
    const shouldSend = vi.fn().mockReturnValueOnce(true).mockReturnValue(false)

    const response = await apiRequest("http://api.local/activity", { method: "POST" }, undefined, { shouldSend })

    expect(response.status).toBe(401)
    expect(shouldSend).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls.filter(([input]) => !isRefresh(input))).toHaveLength(1)
    // The login itself is kept with its newer credentials; only this request is dropped.
    expect(readAuthCache()?.token).toBe(scenario === "rotated" ? "rotated-access" : "new-access")
  })

  it("asks before every transport attempt and stops retrying once the request is no longer wanted", async () => {
    writeAuthCache("access", "refresh")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"))
    const shouldSend = vi.fn().mockReturnValueOnce(true).mockReturnValue(false)

    await expect(apiRequest("http://api.local/activity", { method: "POST" }, undefined, { shouldSend }))
      .rejects.toBeInstanceOf(RequestNotSentError)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(shouldSend).toHaveBeenCalledTimes(2)
    expect(readAuthCache()?.token).toBe("access")
  })

  it("keeps the ordinary transport retries for a request without the hook", async () => {
    writeAuthCache("access", "refresh")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"))

    await expect(apiRequest("http://api.local/protected")).rejects.toThrow("Failed to fetch")

    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it("sends nothing when the request is no longer wanted before the first attempt", async () => {
    writeAuthCache("access", "refresh")
    const fetchMock = vi.spyOn(globalThis, "fetch")

    await expect(apiRequest("http://api.local/activity", { method: "POST" }, undefined, { shouldSend: () => false }))
      .rejects.toBeInstanceOf(RequestNotSentError)

    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("ends the same login when its rotated credentials are not replayed but the server reported idle expiry", async () => {
    writeAuthCache("old-access", "old-refresh", "same-login")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      // Another tab rotated the same login's credentials while this request was in flight.
      const cache = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY)!)
      localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({ ...cache, credentialRevision: 1, token: "rotated-access", refreshToken: "rotated-refresh" }))
      return idleExpired()
    })
    const shouldSend = vi.fn().mockReturnValueOnce(true).mockReturnValue(false)

    const response = await apiRequest("http://api.local/activity", { method: "POST" }, undefined, { shouldSend })

    expect(response.status).toBe(401)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(readAuthCache()).toBeNull()
    expect(readSessionEndNotice(localStorage)).toBe("session_expired")
  })

  it("keeps a replacement login when a skipped replay's original response reported idle expiry", async () => {
    writeAuthCache("old-access", "old-refresh", "same-login")
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      const cache = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY)!)
      localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({ ...cache, credentialRevision: 1, token: "rotated-access", refreshToken: "rotated-refresh" }))
      return idleExpired()
    })
    let attempts = 0
    const shouldSend = () => {
      attempts += 1
      if (attempts === 1) return true
      // Another user signs in just before the replay decision.
      writeAuthCache("other-access", "other-refresh", "other-login")
      return false
    }

    await apiRequest("http://api.local/activity", { method: "POST" }, undefined, { shouldSend })

    expect(readAuthCache()?.token).toBe("other-access")
    expect(readSessionEndNotice(localStorage)).toBeNull()
  })

  it("tells the caller that a refresh was refused because the login expired", async () => {
    writeAuthCache("old-access", "old-refresh")
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => idleExpired())
    await expect(refreshStoredAccessToken(readAuthSessionSnapshot())).resolves.toEqual({
      status: "rejected", accessToken: null, reason: "session_expired",
    })
  })

  it("keeps an ordinary refresh rejection free of the idle notice", async () => {
    writeAuthCache("old-access", "old-refresh")
    vi.spyOn(globalThis, "fetch").mockImplementation(async input =>
      isRefresh(input) ? new Response(null, { status: 401 }) : new Response(null, { status: 401, headers: { "Error-Type": "TokenExpired" } }))
    await apiRequest("http://api.local/protected")
    expect(readAuthCache()).toBeNull()
    expect(readSessionEndNotice(localStorage)).toBeNull()
  })

  it("keeps the session on a server error", async () => {
    writeAuthCache("access", "refresh")
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(null, { status: 503 }))
    await apiRequest("http://api.local/protected")
    expect(readAuthCache()?.token).toBe("access")
    expect(readSessionEndNotice(localStorage)).toBeNull()
  })

  it("leaves a replacement login intact when an old login's request reports idle expiry", async () => {
    writeAuthCache("old-access", "old-refresh", "old-login")
    const old = readAuthSessionSnapshot()
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      writeAuthCache("new-access", "new-refresh", "new-login")
      return idleExpired()
    })

    await apiRequest("http://api.local/protected", {}, old)

    expect(readAuthCache()?.token).toBe("new-access")
    expect(readSessionEndNotice(localStorage)).toBeNull()
  })

  it("stores the idle_session returned with a refreshed credential", async () => {
    writeAuthCache("old-access", "old-refresh")
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({
      success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: 120, refresh_expires_in: 240,
      idle_session: {
        enforced: true, sid: "sid-a", idle_timeout_seconds: 900, idle_expires_in: 400,
        server_time: "2026-11-13T10:00:00Z", absolute_expires_in: null, activity_min_interval_seconds: 60,
      },
    }), { headers: { "Content-Type": "application/json" } }))

    await expect(refreshStoredAccessToken(readAuthSessionSnapshot())).resolves.toMatchObject({ status: "refreshed" })
    expect(readCurrentIdleSession()).toMatchObject({ sid: "sid-a", idleTimeoutSeconds: 900, idleExpiresIn: 400 })
  })
})

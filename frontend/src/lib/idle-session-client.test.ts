import { beforeEach, describe, expect, it, vi } from "vitest"

import { apiRequest } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY, readAuthSessionSnapshot, readCurrentIdleSession } from "@/lib/auth-cache"
import { IDLE_SESSION_STORAGE_KEY } from "@/lib/idle-session"
import { reportIdleActivity, revalidateIdleSession } from "@/lib/idle-session-client"

vi.mock("@/lib/api-wrapper", () => ({ apiRequest: vi.fn() }))

const idleBody = (idleExpiresIn: number) => JSON.stringify({
  idle_session: {
    enforced: true, sid: "sid-a", idle_timeout_seconds: 1800, idle_expires_in: idleExpiresIn,
    server_time: "2026-11-13T10:00:00Z", absolute_expires_in: null, activity_min_interval_seconds: 60,
  },
})

function login(sessionId = "browser-login") {
  const now = Date.now()
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
    schemaVersion: 2, sessionId, credentialRevision: 0, profileRevision: 0,
    user: { id: "1", username: "alice" }, token: "access", refreshToken: "refresh", timestamp: now,
    expiresAt: now + 3_600_000, refreshExpiresAt: now + 7_200_000,
  }))
  localStorage.setItem(IDLE_SESSION_STORAGE_KEY, JSON.stringify({
    schemaVersion: 1, sessionId, enforced: true, sid: "sid-a", idleTimeoutSeconds: 1800, idleExpiresIn: 1000,
    absoluteExpiresIn: null, activityMinIntervalSeconds: 60, observedAt: now, lastReportAt: null,
  }))
}

describe("idle session client", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.mocked(apiRequest).mockReset()
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { request: vi.fn(async (_: string, action: () => Promise<unknown>) => action()) },
    })
  })

  it("reports activity with the current credential and stores the new deadline", async () => {
    login()
    vi.mocked(apiRequest).mockResolvedValue(new Response(idleBody(1800)))
    await expect(reportIdleActivity("browser-login")).resolves.toBe(true)
    expect(apiRequest).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/auth\/activity$/), { method: "POST" }, readAuthSessionSnapshot(),
    )
    expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1800)
  })

  it.each([
    ["a server error is retried", () => Promise.resolve(new Response(null, { status: 503 })), false],
    ["a lost connection is retried", () => Promise.reject(new Error("offline")), false],
    ["a rejected login is not retried", () => Promise.resolve(new Response(null, { status: 401 })), true],
    ["an unreadable success is not retried", () => Promise.resolve(new Response("not json")), true],
  ])("%s", async (_, respond, expected) => {
    login()
    vi.mocked(apiRequest).mockImplementation(respond)
    await expect(reportIdleActivity("browser-login")).resolves.toBe(expected)
    expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1000)
  })

  it("sends nothing on behalf of a login that is no longer current", async () => {
    login("newer-login")
    await expect(reportIdleActivity("browser-login")).resolves.toBe(true)
    await revalidateIdleSession("browser-login")
    expect(apiRequest).not.toHaveBeenCalled()
  })

  it("revalidates through verify and stores the server's answer", async () => {
    login()
    vi.mocked(apiRequest).mockResolvedValue(new Response(idleBody(1500)))
    await revalidateIdleSession("browser-login")
    expect(apiRequest).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/auth\/verify$/), { headers: { "X-Username": "alice" } }, readAuthSessionSnapshot(),
    )
    expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1500)
  })

  it("keeps the stored deadline when a successful check has an unreadable body", async () => {
    login()
    vi.mocked(apiRequest).mockResolvedValue(new Response("not json"))
    await revalidateIdleSession("browser-login")
    expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1000)
  })

  it("treats an unreachable server as unknown, not expired", async () => {
    login()
    vi.mocked(apiRequest).mockRejectedValue(new Error("offline"))
    await expect(revalidateIdleSession("browser-login")).resolves.toBeUndefined()
    expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1000)
  })
})

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AuthProvider, useAuth } from "@/contexts/auth-context"
import { apiRequest, refreshStoredAccessToken } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY, readCurrentIdleSession } from "@/lib/auth-cache"

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: vi.fn(async () => new Response(null, { status: 404 })),
  refreshStoredAccessToken: vi.fn(),
}))
vi.mock("@/components/ui/sonner", () => ({ toast: { error: vi.fn() } }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))

const idlePolicy = (overrides: Record<string, unknown> = {}) => ({
  enforced: true, sid: "sid-a", idle_timeout_seconds: 1800, idle_expires_in: 1700,
  server_time: "2026-11-13T10:00:00Z", absolute_expires_in: null, activity_min_interval_seconds: 60,
  ...overrides,
})

function writeAuthCache(token = "token", expiresIn = 3600, refreshExpiresIn = 7200) {
  const now = Date.now()
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
    schemaVersion: 2, sessionId: `test-${Math.random()}`, credentialRevision: 0, profileRevision: 0,
    user: { id: "1", username: "alice" }, token, refreshToken: "refresh-token", timestamp: now,
    expiresAt: now + expiresIn * 1000, refreshExpiresAt: now + refreshExpiresIn * 1000,
  }))
}

function CheckProbe() {
  const { checkAuth, token } = useAuth()
  const [result, setResult] = React.useState("pending")
  return <>
    <span data-testid="token">{token || "none"}</span>
    <span data-testid="check">{result}</span>
    <button onClick={() => { void checkAuth().then(value => setResult(String(value))) }}>Check</button>
  </>
}

const json = (body: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" }, ...init })

function installLocks() {
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: { request: vi.fn(async (_: string, callback: () => Promise<unknown>) => callback()) },
  })
}

beforeEach(() => {
  localStorage.clear()
  installLocks()
  vi.mocked(apiRequest).mockReset()
  vi.mocked(apiRequest).mockImplementation(async () => new Response(null, { status: 404 }))
  vi.mocked(refreshStoredAccessToken).mockReset()
  vi.mocked(refreshStoredAccessToken).mockResolvedValue({ status: "not_current", accessToken: null })
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  Reflect.deleteProperty(document, "visibilityState")
})

describe("AuthProvider idle policy from verify", () => {
  it("records the idle_session returned by a session check", async () => {
    writeAuthCache()
    vi.mocked(apiRequest).mockImplementation(async url =>
      String(url).endsWith("/api/auth/verify") ? json({ success: true, idle_session: idlePolicy({ idle_expires_in: 1200 }) }) : new Response(null, { status: 404 }))
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    fireEvent.click(screen.getByRole("button", { name: "Check" }))
    await waitFor(() => expect(screen.getByTestId("check")).toHaveTextContent("true"))
    expect(readCurrentIdleSession()).toMatchObject({ enforced: true, sid: "sid-a", idleExpiresIn: 1200 })
  })
})


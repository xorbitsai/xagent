import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AuthProvider, useAuth } from "@/contexts/auth-context"
import { AUTH_REVALIDATE_EVENT } from "@/lib/idle-session"
import { apiRequest, refreshStoredAccessToken } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY, readAuthSessionSnapshot, readCurrentIdleSession } from "@/lib/auth-cache"
import { IDLE_SESSION_STORAGE_KEY, readSessionEndNotice } from "@/lib/idle-session"

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: vi.fn(async () => new Response(null, { status: 404 })),
  refreshStoredAccessToken: vi.fn(),
}))
vi.mock("@/components/ui/sonner", () => ({ toast: { error: vi.fn() } }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))
// jsdom marks dispatched events as untrusted; treat test input as real user input.
vi.mock("@/lib/idle-activity", async importOriginal => ({
  ...await importOriginal<typeof import("@/lib/idle-activity")>(),
  isUserActivityEvent: () => true,
}))

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

function writeIdleRecord(overrides: Record<string, unknown> = {}) {
  const snapshot = readAuthSessionSnapshot()
  localStorage.setItem(IDLE_SESSION_STORAGE_KEY, JSON.stringify({
    schemaVersion: 1, sessionId: snapshot.sessionId, enforced: true, sid: "sid-a",
    idleTimeoutSeconds: 1800, idleExpiresIn: 1700, absoluteExpiresIn: null, activityMinIntervalSeconds: 60,
    observedAt: Date.now(), lastReportAt: null, ...overrides,
  }))
}

const verifyCalls = () => vi.mocked(apiRequest).mock.calls.filter(([url]) => String(url).endsWith("/api/auth/verify"))
const activityCalls = () => vi.mocked(apiRequest).mock.calls.filter(([url]) => String(url).endsWith("/api/auth/activity"))

function RefreshProbe() {
  const { refreshAccessToken, token } = useAuth()
  return <>
    <span data-testid="refresh-token">{token || "none"}</span>
    <button onClick={() => { void refreshAccessToken() }}>Refresh</button>
  </>
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
  it("checks again after the clock moves backwards, instead of trusting a recent check", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    writeAuthCache()
    vi.mocked(apiRequest).mockImplementation(async url =>
      String(url).endsWith("/api/auth/verify") ? json({ success: true }) : new Response(null, { status: 404 }))
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    fireEvent.click(screen.getByRole("button", { name: "Check" }))
    await waitFor(() => expect(verifyCalls()).toHaveLength(1))
    vi.setSystemTime(Date.now() - 10 * 60_000)
    fireEvent.click(screen.getByRole("button", { name: "Check" }))
    await waitFor(() => expect(verifyCalls()).toHaveLength(2))
  })

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


describe("AuthProvider activity reporting", () => {
  it("reports user input for the current login and stores the returned deadline", async () => {
    writeAuthCache()
    writeIdleRecord()
    vi.mocked(apiRequest).mockImplementation(async url =>
      String(url).endsWith("/api/auth/activity") ? json({ idle_session: idlePolicy({ idle_expires_in: 1800 }) }) : new Response(null, { status: 404 }))
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    fireEvent.keyDown(window, { key: "a" })
    await waitFor(() => expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1800))
    expect(activityCalls()).toHaveLength(1)
    const [, init, captured] = activityCalls()[0]
    expect(init).toMatchObject({ method: "POST" })
    expect(captured).toMatchObject({ accessToken: "token" })
    expect(readCurrentIdleSession()?.lastReportAt).not.toBeNull()
  })

  it("reports nothing for a server without idle support", async () => {
    writeAuthCache()
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    fireEvent.keyDown(window, { key: "a" })
    fireEvent.pointerDown(window)
    expect(activityCalls()).toHaveLength(0)
  })

  it("does not report on behalf of a login that has been replaced", async () => {
    writeAuthCache()
    writeIdleRecord({ sessionId: "an-older-login" })
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    fireEvent.keyDown(window, { key: "a" })
    expect(activityCalls()).toHaveLength(0)
  })

  it("stops reporting once the provider unmounts", async () => {
    writeAuthCache()
    writeIdleRecord()
    const { unmount } = render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    unmount()
    fireEvent.keyDown(window, { key: "a" })
    expect(activityCalls()).toHaveLength(0)
  })
})

describe("AuthProvider never refreshes past the idle deadline", () => {
  it("asks the server instead of refreshing when the deadline has passed while the switch is on", async () => {
    writeAuthCache("expiring-token", 60, 3600)
    writeIdleRecord({ idleExpiresIn: 0 })
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("expiring-token")
    act(() => { window.dispatchEvent(new Event("focus")) })
    await waitFor(() => expect(verifyCalls().length).toBeGreaterThan(0))
    expect(refreshStoredAccessToken).not.toHaveBeenCalled()
  })

  it("keeps refreshing a token near expiry while the switch is off", async () => {
    writeAuthCache("expiring-token", 60, 3600)
    writeIdleRecord({ enforced: false, idleExpiresIn: 0 })
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("expiring-token")
    act(() => { window.dispatchEvent(new Event("focus")) })
    await waitFor(() => expect(refreshStoredAccessToken).toHaveBeenCalled())
    expect(verifyCalls()).toHaveLength(0)
  })

  it("ends the login with a sign-in-again notice when the refresh is refused for inactivity", async () => {
    writeAuthCache()
    vi.mocked(refreshStoredAccessToken).mockResolvedValue({ status: "rejected", accessToken: null, reason: "session_expired" })
    render(<AuthProvider><RefreshProbe /></AuthProvider>)
    await screen.findByText("token")
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }))
    await waitFor(() => expect(screen.getByTestId("refresh-token")).toHaveTextContent("none"))
    expect(readSessionEndNotice(localStorage)).toBe("session_expired")
  })
})

describe("IdleSessionManager revalidation", () => {
  it("asks the server at the deadline and keeps a login another tab kept active", async () => {
    writeAuthCache()
    writeIdleRecord({ idleExpiresIn: 0, observedAt: Date.now() - 5_000 })
    vi.mocked(apiRequest).mockImplementation(async url =>
      String(url).endsWith("/api/auth/verify") ? json({ idle_session: idlePolicy({ idle_expires_in: 1500 }) }) : new Response(null, { status: 404 }))
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await waitFor(() => expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1500))
    expect(screen.getByTestId("token")).toHaveTextContent("token")
    expect(activityCalls()).toHaveLength(0)
  })

  it("waits until the server's own deadline has passed before asking", async () => {
    // idle_expires_in is rounded down, so the local deadline can be up to a second early.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    writeAuthCache()
    writeIdleRecord({ idleExpiresIn: 0 })
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    await act(async () => { await vi.advanceTimersByTimeAsync(1_000) })
    expect(verifyCalls()).toHaveLength(0)
    await act(async () => { await vi.advanceTimersByTimeAsync(1_000) })
    expect(verifyCalls()).toHaveLength(1)
  })

  it("does not sign out when the server cannot confirm the session", async () => {
    writeAuthCache()
    writeIdleRecord({ idleExpiresIn: 0, observedAt: Date.now() - 5_000 })
    vi.mocked(apiRequest).mockImplementation(async () => new Response(null, { status: 503 }))
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await waitFor(() => expect(verifyCalls().length).toBeGreaterThan(0))
    expect(screen.getByTestId("token")).toHaveTextContent("token")
    expect(readCurrentIdleSession()).not.toBeNull()
  })

  it("does not turn a server outage at the deadline into a request loop", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    writeAuthCache()
    writeIdleRecord({ idleExpiresIn: 0, observedAt: Date.now() - 5_000 })
    vi.mocked(apiRequest).mockImplementation(async () => new Response(null, { status: 503 }))
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await waitFor(() => expect(verifyCalls()).toHaveLength(1))
    await act(async () => { await vi.advanceTimersByTimeAsync(12_000) })
    expect(verifyCalls()).toHaveLength(1)
    await act(async () => { await vi.advanceTimersByTimeAsync(5_000) })
    expect(verifyCalls()).toHaveLength(2)
  })

  it("does not act on the deadline while the switch is off", async () => {
    writeAuthCache()
    writeIdleRecord({ enforced: false, idleExpiresIn: 0 })
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    act(() => { window.dispatchEvent(new Event("focus")) })
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(verifyCalls()).toHaveLength(0)
  })

  it.each([
    ["the page becomes visible", () => document.dispatchEvent(new Event("visibilitychange"))],
    ["the window regains focus", () => window.dispatchEvent(new Event("focus"))],
    ["the browser comes back online", () => window.dispatchEvent(new Event("online"))],
  ])("revalidates before continuing when %s", async (_, wake) => {
    writeAuthCache()
    writeIdleRecord()
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    expect(verifyCalls()).toHaveLength(0)
    act(() => wake())
    await waitFor(() => expect(verifyCalls()).toHaveLength(1))
  })

  it("sends one check for a burst of wake events", async () => {
    writeAuthCache()
    writeIdleRecord()
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"))
      window.dispatchEvent(new Event("focus"))
      window.dispatchEvent(new Event("online"))
    })
    await waitFor(() => expect(verifyCalls()).toHaveLength(1))
  })

  it("revalidates when a socket reports that the session ended", async () => {
    writeAuthCache()
    writeIdleRecord()
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    act(() => { window.dispatchEvent(new Event(AUTH_REVALIDATE_EVENT)) })
    await waitFor(() => expect(verifyCalls()).toHaveLength(1))
  })

  it("does not mistake background-tab timer throttling for a clock jump", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    writeAuthCache()
    writeIdleRecord()
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" })
    // A hidden tab's timers run about once a minute.
    vi.setSystemTime(Date.now() + 60_000)
    await act(async () => { await vi.advanceTimersByTimeAsync(1_000) })
    expect(verifyCalls()).toHaveLength(0)
  })

  it("asks the server after the device clock moves backwards, even right after a check", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    writeAuthCache()
    writeIdleRecord()
    vi.mocked(apiRequest).mockImplementation(async url =>
      String(url).endsWith("/api/auth/verify") ? json({ idle_session: idlePolicy({ idle_expires_in: 1500 }) }) : new Response(null, { status: 404 }))
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    act(() => { window.dispatchEvent(new Event("focus")) })
    await waitFor(() => expect(verifyCalls()).toHaveLength(1))
    vi.setSystemTime(Date.now() - 10 * 60_000)
    await act(async () => { await vi.advanceTimersByTimeAsync(2_000) })
    await waitFor(() => expect(verifyCalls()).toHaveLength(2))
  })

  it("leaves servers without idle support alone", async () => {
    writeAuthCache()
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    act(() => {
      window.dispatchEvent(new Event("focus"))
      window.dispatchEvent(new Event(AUTH_REVALIDATE_EVENT))
    })
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(verifyCalls()).toHaveLength(0)
  })

  it("revalidates after the device clock jumps, as after sleep", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    writeAuthCache()
    writeIdleRecord()
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    expect(verifyCalls()).toHaveLength(0)
    // Wall clock moves 10 minutes between two ticks.
    vi.setSystemTime(Date.now() + 10 * 60_000)
    await act(async () => { await vi.advanceTimersByTimeAsync(5_000) })
    await waitFor(() => expect(verifyCalls()).toHaveLength(1))
  })
})

describe("IdleSessionManager expiry warning", () => {
  it("warns before the deadline and keeps the session when the user continues", async () => {
    writeAuthCache()
    writeIdleRecord({ idleExpiresIn: 90 })
    vi.mocked(apiRequest).mockImplementation(async url =>
      String(url).endsWith("/api/auth/activity") ? json({ idle_session: idlePolicy({ idle_expires_in: 1800 }) }) : new Response(null, { status: 404 }))
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    const dialog = await screen.findByRole("alertdialog")
    expect(dialog).toHaveTextContent("session_idle.warning_title")
    expect(dialog).toHaveTextContent("session_idle.warning_description")
    // The safe choice has focus, so Enter keeps the session.
    expect(document.activeElement).toHaveTextContent("session_idle.continue")
    fireEvent.click(screen.getByRole("button", { name: "session_idle.continue" }))
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull())
    expect(activityCalls().length).toBeGreaterThan(0)
    expect(readCurrentIdleSession()?.idleExpiresIn).toBe(1800)
  })

  it("lets the user sign out from the warning", async () => {
    writeAuthCache()
    writeIdleRecord({ idleExpiresIn: 90 })
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 200 })))
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByRole("alertdialog")
    fireEvent.click(screen.getByRole("button", { name: "session_idle.sign_out" }))
    await waitFor(() => expect(screen.getByTestId("token")).toHaveTextContent("none"))
    vi.unstubAllGlobals()
  })

  it.each([
    ["the deadline is still far away", { idleExpiresIn: 600 }],
    ["the switch is off", { enforced: false, idleExpiresIn: 90 }],
  ])("does not warn when %s", async (_, overrides) => {
    writeAuthCache()
    writeIdleRecord(overrides)
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(screen.queryByRole("alertdialog")).toBeNull()
  })

  it("shortens the warning for very short thresholds so the dialog is not always open", async () => {
    writeAuthCache()
    writeIdleRecord({ idleTimeoutSeconds: 60, idleExpiresIn: 50 })
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    await screen.findByText("token")
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(screen.queryByRole("alertdialog")).toBeNull()
    cleanup()
    localStorage.clear()
    writeAuthCache()
    writeIdleRecord({ idleTimeoutSeconds: 60, idleExpiresIn: 25 })
    render(<AuthProvider><CheckProbe /></AuthProvider>)
    expect(await screen.findByRole("alertdialog")).toBeInTheDocument()
  })
})

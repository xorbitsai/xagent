import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react"
import React from "react"
import { renderToString } from "react-dom/server"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AnonymousAuthProvider, AuthProvider, useAuth } from "@/contexts/auth-context"
import { apiRequest, refreshStoredAccessToken } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY, AUTH_TOKEN_UPDATED_EVENT, clearStoredAuth } from "@/lib/auth-cache"

const toastError = vi.hoisted(() => vi.fn())
const translate = vi.hoisted(() => vi.fn((key: string) => `translated:${key}`))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: vi.fn(async () => new Response(null, { status: 404 })),
  refreshStoredAccessToken: vi.fn(),
}))
vi.mock("@/components/ui/sonner", () => ({ toast: { error: toastError } }))
vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: translate }) }))

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function writeAuthCache(
  user: { id: string; username: string; email?: string | null; is_admin?: boolean },
  token: string,
  refreshToken: string | null = null,
  expiresIn?: number,
  refreshExpiresIn?: number,
) {
  const now = Date.now()
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
    schemaVersion: 2, sessionId: `test-${Math.random()}`, credentialRevision: 0, profileRevision: 0,
    user, token, refreshToken, timestamp: now,
    expiresAt: expiresIn ? now + expiresIn * 1000 : undefined,
    refreshExpiresAt: refreshExpiresIn ? now + refreshExpiresIn * 1000 : undefined,
  }))
}

function AuthProbe() {
  const { checkAuth, token } = useAuth()
  const [checkResult, setCheckResult] = React.useState("pending")
  return (
    <>
      <span data-testid="access-token">{token || "none"}</span>
      <span data-testid="check-result">{checkResult}</span>
      <button onClick={() => {
        void checkAuth().then(result => setCheckResult(String(result)))
      }}>
        Check auth
      </button>
    </>
  )
}

function AuthRefreshProbe() {
  const { refreshAccessToken, token } = useAuth()
  const [refreshResult, setRefreshResult] = React.useState("pending")
  return (
    <>
      <span data-testid="refresh-access-token">{token || "none"}</span>
      <span data-testid="refresh-result">{refreshResult}</span>
      <button onClick={() => {
        void refreshAccessToken().then(result => setRefreshResult(String(result)))
      }}>
        Refresh access token
      </button>
    </>
  )
}

function AuthLogoutProbe() {
  const { logout, token } = useAuth()
  const [result, setResult] = React.useState("pending")
  return <>
    <span data-testid="logout-token">{token || "none"}</span>
    <span data-testid="logout-result">{result}</span>
    <button onClick={() => { void logout().then(value => setResult(String(value))) }}>Logout</button>
  </>
}

function AuthLoginLogoutProbe() {
  const { login, logout, token } = useAuth()
  const [loginResult, setLoginResult] = React.useState("pending")
  const [logoutResult, setLogoutResult] = React.useState("pending")
  return <>
    <span data-testid="login-logout-token">{token || "none"}</span>
    <span data-testid="login-result">{loginResult}</span>
    <span data-testid="login-logout-result">{logoutResult}</span>
    <button onClick={() => { void login("alice", "password").then(value => setLoginResult(String(value))) }}>Login</button>
    <button onClick={() => { void logout().then(value => setLogoutResult(String(value))) }}>Logout</button>
  </>
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(resolvePromise => { resolve = resolvePromise })
  return { promise, resolve }
}

function AuthSsrProbe() {
  const { isLoading, session } = useAuth()
  return <span>{`${isLoading}:${session.accessToken || "none"}`}</span>
}

function AnonymousAuthProbe() {
  const auth = useAuth()
  const [results, setResults] = React.useState<boolean[] | null>(null)
  const projection = {
    user: auth.user,
    isAuthenticated: auth.isAuthenticated,
    token: auth.token,
    refreshToken: auth.refreshToken,
    session: auth.session,
    isLoading: auth.isLoading,
    inTeam: auth.inTeam,
    teamRole: auth.teamRole,
  }
  return <>
    <span data-testid="anonymous-auth-projection">{JSON.stringify(projection)}</span>
    <span data-testid="anonymous-auth-results">{JSON.stringify(results)}</span>
    <button onClick={() => {
      void Promise.all([
        auth.login("ignored", "ignored"),
        auth.logout(),
        auth.checkAuth(),
        auth.refreshAccessToken(),
      ]).then(setResults)
    }}>
      Exercise anonymous auth
    </button>
  </>
}

describe("AuthProvider storage synchronization", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
    toastError.mockReset()
    translate.mockClear()
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { request: vi.fn(async (_name: string, callback: () => Promise<unknown>) => callback()) },
    })
    vi.mocked(apiRequest).mockImplementation(
      async () => new Response(null, { status: 404 })
    )
  })

  it("server-renders an empty loading projection when browser storage is unavailable", () => {
    vi.stubGlobal("localStorage", {})
    expect(() => renderToString(<AuthProvider><AuthSsrProbe /></AuthProvider>)).not.toThrow()
    expect(renderToString(<AuthProvider><AuthSsrProbe /></AuthProvider>)).toContain("true:none")
  })

  it("keeps anonymous routes neutral without auth storage or API side effects", async () => {
    const cacheBefore = JSON.stringify({
      schemaVersion: 2, sessionId: "personal-session", credentialRevision: 0, profileRevision: 0,
      user: { id: "1", username: "alice" }, token: "personal-access", refreshToken: "personal-refresh",
      timestamp: Date.now(),
    })
    localStorage.setItem(AUTH_CACHE_KEY, cacheBefore)
    vi.mocked(apiRequest).mockClear()
    vi.mocked(refreshStoredAccessToken).mockClear()
    const lockRequest = vi.mocked(navigator.locks.request)
    lockRequest.mockClear()

    render(<AnonymousAuthProvider><AnonymousAuthProbe /></AnonymousAuthProvider>)

    expect(JSON.parse(screen.getByTestId("anonymous-auth-projection").textContent || "null")).toEqual({
      user: null,
      isAuthenticated: false,
      token: null,
      refreshToken: null,
      session: {
        sessionId: null, credentialRevision: null, profileRevision: null,
        userId: null, accessToken: null, refreshToken: null, profileFingerprint: null,
      },
      isLoading: false,
      inTeam: false,
      teamRole: null,
    })

    fireEvent.click(screen.getByRole("button", { name: "Exercise anonymous auth" }))

    await waitFor(() => {
      expect(screen.getByTestId("anonymous-auth-results")).toHaveTextContent("[false,false,false,false]")
    })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(cacheBefore)
    expect(apiRequest).not.toHaveBeenCalled()
    expect(refreshStoredAccessToken).not.toHaveBeenCalled()
    expect(lockRequest).not.toHaveBeenCalled()
  })

  it("ignores non-object auth cache payloads without a runtime error", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "access-token",
      "refresh-token",
      120,
      240
    )
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})

    render(
      <AuthProvider>
        <AuthProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("access-token")).toHaveTextContent("access-token")
    })

    act(() => {
      window.dispatchEvent(new StorageEvent("storage", {
        key: AUTH_CACHE_KEY,
        newValue: "null",
      }))
    })

    expect(screen.getByTestId("access-token")).toHaveTextContent("access-token")
    expect(consoleError).not.toHaveBeenCalled()
  })

  it("updates auth state from a valid cross-tab cache payload", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )

    render(
      <AuthProvider>
        <AuthProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("access-token")).toHaveTextContent("alice-access")
    })

    writeAuthCache(
      { id: "2", username: "bob", email: null, is_admin: false },
      "bob-access",
      "bob-refresh",
      120,
      240
    )
    act(() => {
      window.dispatchEvent(new StorageEvent("storage", {
        key: AUTH_CACHE_KEY,
        newValue: localStorage.getItem(AUTH_CACHE_KEY),
      }))
    })

    expect(screen.getByTestId("access-token")).toHaveTextContent("bob-access")
  })

  it("recomputes the projection from a value-free same-tab invalidation", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240,
    )
    render(<AuthProvider><AuthProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("access-token")).toHaveTextContent("alice-access"))

    localStorage.removeItem(AUTH_CACHE_KEY)
    act(() => window.dispatchEvent(new Event(AUTH_TOKEN_UPDATED_EVENT)))

    expect(screen.getByTestId("access-token")).toHaveTextContent("none")
  })

  it("keeps auth state when a 401 leaves the refresh cache intact", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )
    vi.mocked(apiRequest).mockImplementation(async (url) =>
      new Response(null, {
        status: String(url).endsWith("/api/auth/verify") ? 401 : 404,
        headers: { "Error-Type": "TokenExpired" },
      })
    )

    render(
      <AuthProvider>
        <AuthProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("access-token")).toHaveTextContent("alice-access")
    })

    fireEvent.click(screen.getByRole("button", { name: "Check auth" }))

    await waitFor(() => {
      expect(screen.getByTestId("check-result")).toHaveTextContent("true")
    })
    expect(screen.getByTestId("access-token")).toHaveTextContent("alice-access")
  })

  it("clears auth state when a 401 follows a rejected refresh", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )
    vi.mocked(apiRequest).mockImplementation(async (url) => {
      if (String(url).endsWith("/api/auth/verify")) {
        clearStoredAuth()
        return new Response(null, {
          status: 401,
          headers: { "Error-Type": "TokenExpired" },
        })
      }
      return new Response(null, { status: 404 })
    })

    render(
      <AuthProvider>
        <AuthProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("access-token")).toHaveTextContent("alice-access")
    })

    fireEvent.click(screen.getByRole("button", { name: "Check auth" }))

    await waitFor(() => {
      expect(screen.getByTestId("check-result")).toHaveTextContent("false")
      expect(screen.getByTestId("access-token")).toHaveTextContent("none")
    })
  })

  it("does not log out a replacement session after an older refresh is rejected", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )
    let resolveRefresh!: (value: { status: "rejected"; accessToken: null }) => void
    const refreshPromise = new Promise<{ status: "rejected"; accessToken: null }>((resolve) => {
      resolveRefresh = resolve
    })
    vi.mocked(refreshStoredAccessToken).mockReturnValue(refreshPromise)

    render(
      <AuthProvider>
        <AuthRefreshProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("alice-access")
    })

    fireEvent.click(screen.getByRole("button", { name: "Refresh access token" }))
    await waitFor(() => {
      expect(refreshStoredAccessToken).toHaveBeenCalledWith(expect.objectContaining({
        accessToken: "alice-access",
        userId: "1",
      }))
    })

    writeAuthCache(
      { id: "2", username: "bob", email: null, is_admin: false },
      "bob-access",
      "bob-refresh",
      120,
      240
    )
    act(() => {
      window.dispatchEvent(new StorageEvent("storage", {
        key: AUTH_CACHE_KEY,
        newValue: localStorage.getItem(AUTH_CACHE_KEY),
      }))
    })
    expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("bob-access")

    await act(async () => {
      resolveRefresh({ status: "rejected", accessToken: null })
      await refreshPromise
    })

    await waitFor(() => {
      expect(screen.getByTestId("refresh-result")).toHaveTextContent("false")
    })
    expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("bob-access")
    expect(JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "null")).toMatchObject({
      token: "bob-access",
      refreshToken: "bob-refresh",
      user: { id: "2" },
    })
  })

  it("logs out the current session after its own refresh is rejected", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "alice-access",
      "alice-refresh",
      120,
      240
    )
    vi.mocked(refreshStoredAccessToken).mockResolvedValue({ status: "rejected", accessToken: null })
    vi.spyOn(console, "error").mockImplementation(() => {})

    render(
      <AuthProvider>
        <AuthRefreshProbe />
      </AuthProvider>
    )
    await waitFor(() => {
      expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("alice-access")
    })

    fireEvent.click(screen.getByRole("button", { name: "Refresh access token" }))

    await waitFor(() => {
      expect(screen.getByTestId("refresh-result")).toHaveTextContent("false")
      expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("none")
    })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
  })

  it("does not reuse the old check debounce after credentials advance in storage", async () => {
    writeAuthCache(
      { id: "1", username: "alice", email: null, is_admin: false },
      "old-access", "old-refresh", 120, 240,
    )
    render(<AuthProvider><AuthProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("access-token")).toHaveTextContent("old-access"))
    const raw = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "{}")
    raw.token = "new-access"
    raw.credentialRevision += 1
    localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(raw))

    fireEvent.click(screen.getByRole("button", { name: "Check auth" }))

    await waitFor(() => {
      expect(screen.getByTestId("check-result")).toHaveTextContent("true")
      expect(screen.getByTestId("access-token")).toHaveTextContent("new-access")
    })
    expect(vi.mocked(apiRequest).mock.calls.filter(([url]) => String(url).endsWith("/api/auth/verify"))).toHaveLength(0)
  })

  it("keeps the Provider projection when logout cannot acquire the storage lock", async () => {
    writeAuthCache({ id: "1", username: "alice", email: null, is_admin: false }, "access", "refresh", 120, 240)
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { request: vi.fn(async () => { throw new Error("lock unavailable") }) },
    })
    render(<AuthProvider><AuthLogoutProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("logout-token")).toHaveTextContent("access"))
    fireEvent.click(screen.getByRole("button", { name: "Logout" }))
    await waitFor(() => expect(screen.getByTestId("logout-result")).toHaveTextContent("false"))
    expect(screen.getByTestId("logout-token")).toHaveTextContent("access")
    expect(JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "null")).toMatchObject({ token: "access" })
  })

  it("clears the Provider projection when logout falls back from an intent barrier write", async () => {
    writeAuthCache({ id: "1", username: "alice", email: null, is_admin: false }, "access", "refresh", 120, 240)
    const originalSetItem = localStorage.setItem.bind(localStorage)
    vi.spyOn(localStorage, "setItem").mockImplementation((key, value) => {
      if (key === "auth_login_intent") throw new Error("intent write failed")
      originalSetItem(key, value)
    })
    vi.spyOn(console, "error").mockImplementation(() => {})
    render(<AuthProvider><AuthLogoutProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("logout-token")).toHaveTextContent("access"))

    fireEvent.click(screen.getByRole("button", { name: "Logout" }))

    await waitFor(() => {
      expect(screen.getByTestId("logout-result")).toHaveTextContent("true")
      expect(screen.getByTestId("logout-token")).toHaveTextContent("none")
    })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
  })

  it("clears the Provider projection even when the login-intent barrier cannot be replaced or removed", async () => {
    writeAuthCache({ id: "1", username: "alice", email: null, is_admin: false }, "access", "refresh", 120, 240)
    const originalSetItem = localStorage.setItem.bind(localStorage)
    const originalRemoveItem = localStorage.removeItem.bind(localStorage)
    vi.spyOn(localStorage, "setItem").mockImplementation((key, value) => {
      if (key === "auth_login_intent") throw new Error("intent write failed")
      originalSetItem(key, value)
    })
    vi.spyOn(localStorage, "removeItem").mockImplementation(key => {
      if (key === "auth_login_intent") throw new Error("intent removal failed")
      originalRemoveItem(key)
    })
    render(<AuthProvider><AuthLogoutProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("logout-token")).toHaveTextContent("access"))

    fireEvent.click(screen.getByRole("button", { name: "Logout" }))

    await waitFor(() => {
      expect(screen.getByTestId("logout-result")).toHaveTextContent("false")
      expect(screen.getByTestId("logout-token")).toHaveTextContent("none")
    })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
  })

  it("does not persist a same-tab login response after logout invalidates the operation", async () => {
    writeAuthCache({ id: "1", username: "alice", email: null, is_admin: false }, "existing-access", "existing-refresh", 120, 240)
    const loginResponse = deferred<Response>()
    vi.mocked(apiRequest).mockImplementation(async url => {
      if (String(url).endsWith("/api/auth/login")) return loginResponse.promise
      return new Response(null, { status: 404 })
    })
    render(<AuthProvider><AuthLoginLogoutProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("login-logout-token")).toHaveTextContent("existing-access"))

    fireEvent.click(screen.getByRole("button", { name: "Login" }))
    await waitFor(() => expect(apiRequest).toHaveBeenCalledWith(expect.stringContaining("/api/auth/login"), expect.anything()))
    const originalSetItem = localStorage.setItem.bind(localStorage)
    const originalRemoveItem = localStorage.removeItem.bind(localStorage)
    const setItem = vi.spyOn(localStorage, "setItem").mockImplementation((key, value) => {
      if (key === "auth_login_intent" || key === "auth_revoked_login_intent") throw new Error("barrier persistence failed")
      originalSetItem(key, value)
    })
    const removeItem = vi.spyOn(localStorage, "removeItem").mockImplementation(key => {
      if (key === "auth_login_intent") throw new Error("intent removal failed")
      originalRemoveItem(key)
    })
    fireEvent.click(screen.getByRole("button", { name: "Logout" }))
    await waitFor(() => expect(screen.getByTestId("login-logout-token")).toHaveTextContent("none"))
    setItem.mockRestore()
    removeItem.mockRestore()

    await act(async () => {
      loginResponse.resolve(new Response(JSON.stringify({
        user: { id: "1", username: "alice" }, access_token: "late-access", refresh_token: "late-refresh",
      }), { status: 200, headers: { "Content-Type": "application/json" } }))
      await loginResponse.promise
    })

    await waitFor(() => {
      expect(screen.getByTestId("login-result")).toHaveTextContent("false")
      expect(screen.getByTestId("login-logout-token")).toHaveTextContent("none")
    })
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
  })

  it("shows the localized unavailable refresh toast with its stable deduplication ID", async () => {
    writeAuthCache({ id: "1", username: "alice", email: null, is_admin: false }, "access", "refresh", 120, 240)
    vi.mocked(refreshStoredAccessToken).mockResolvedValue({
      status: "unavailable", accessToken: null, reason: "coordination_unavailable",
    })
    render(<AuthProvider><AuthRefreshProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("access"))

    fireEvent.click(screen.getByRole("button", { name: "Refresh access token" }))

    await waitFor(() => expect(toastError).toHaveBeenCalledWith(
      "translated:login.alerts.coordination_unavailable",
      { id: "auth-refresh-unavailable" },
    ))
  })

  it("does not show a refresh toast after the provider unmounts", async () => {
    writeAuthCache({ id: "1", username: "alice", email: null, is_admin: false }, "access", "refresh", 120, 240)
    const refresh = deferred<{ status: "unavailable"; accessToken: null; reason: "operation_failed" }>()
    vi.mocked(refreshStoredAccessToken).mockReturnValue(refresh.promise)
    const rendered = render(<AuthProvider><AuthRefreshProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("access"))

    fireEvent.click(screen.getByRole("button", { name: "Refresh access token" }))
    rendered.unmount()
    await act(async () => {
      refresh.resolve({ status: "unavailable", accessToken: null, reason: "operation_failed" })
      await refresh.promise
    })

    expect(toastError).not.toHaveBeenCalled()
  })

  it("does not show a refresh toast after the captured session is replaced", async () => {
    writeAuthCache({ id: "1", username: "alice", email: null, is_admin: false }, "alice-access", "alice-refresh", 120, 240)
    const refresh = deferred<{ status: "unavailable"; accessToken: null; reason: "operation_failed" }>()
    vi.mocked(refreshStoredAccessToken).mockReturnValue(refresh.promise)
    render(<AuthProvider><AuthRefreshProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("alice-access"))

    fireEvent.click(screen.getByRole("button", { name: "Refresh access token" }))
    writeAuthCache({ id: "2", username: "bob", email: null, is_admin: false }, "bob-access", "bob-refresh", 120, 240)
    act(() => window.dispatchEvent(new StorageEvent("storage", { key: AUTH_CACHE_KEY })))
    await waitFor(() => expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("bob-access"))
    await act(async () => {
      refresh.resolve({ status: "unavailable", accessToken: null, reason: "operation_failed" })
      await refresh.promise
    })

    expect(toastError).not.toHaveBeenCalled()
  })

  it("suppresses the stale refresh operation and emits one stable-ID toast for the current operation", async () => {
    writeAuthCache({ id: "1", username: "alice", email: null, is_admin: false }, "access", "refresh", 120, 240)
    const first = deferred<{ status: "unavailable"; accessToken: null; reason: "operation_failed" }>()
    const second = deferred<{ status: "unavailable"; accessToken: null; reason: "operation_failed" }>()
    vi.mocked(refreshStoredAccessToken)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)
    render(<AuthProvider><AuthRefreshProbe /></AuthProvider>)
    await waitFor(() => expect(screen.getByTestId("refresh-access-token")).toHaveTextContent("access"))

    fireEvent.click(screen.getByRole("button", { name: "Refresh access token" }))
    fireEvent.click(screen.getByRole("button", { name: "Refresh access token" }))
    await act(async () => {
      first.resolve({ status: "unavailable", accessToken: null, reason: "operation_failed" })
      await first.promise
    })
    expect(toastError).not.toHaveBeenCalled()
    await act(async () => {
      second.resolve({ status: "unavailable", accessToken: null, reason: "operation_failed" })
      await second.promise
    })

    expect(toastError).toHaveBeenCalledTimes(1)
    expect(toastError).toHaveBeenCalledWith(
      "translated:login.alerts.operation_failed",
      { id: "auth-refresh-unavailable" },
    )
  })
})

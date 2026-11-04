import { cleanup, render, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { OidcCallbackPage } from "./oidc-callback"
import { claimAuthLoginIntent, claimOidcAuthLoginIntent, createAuthSession, readAuthSessionSnapshot } from "@/lib/auth-cache"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

afterEach(cleanup)

function captureNavigation() {
  const implementation = (window.location as unknown as Record<symbol, unknown>)[
    Object.getOwnPropertySymbols(window.location).find(symbol => String(symbol) === "Symbol(impl)")!
  ] as { _locationObjectSetterNavigate: (url: { path: string[] }) => void }
  const original = implementation._locationObjectSetterNavigate
  const targets: string[] = []
  implementation._locationObjectSetterNavigate = url => { targets.push(`/${url.path.join("/")}`) }
  return { targets, restore: () => { implementation._locationObjectSetterNavigate = original } }
}

describe("OidcCallbackPage", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { request: vi.fn(async (_name: string, action: () => Promise<unknown>) => action()) },
    })
    window.history.replaceState({}, "", "/auth/oidc/callback?provider=google&code=bad")
  })

  it("preserves an existing lineage when the OIDC exchange fails", async () => {
    const passwordIntent = await claimAuthLoginIntent()
    expect(passwordIntent.status).toBe("claimed")
    if (passwordIntent.status !== "claimed") throw new Error("expected password intent")
    const created = await createAuthSession({
      user: { id: "1", username: "alice" },
      access_token: "existing-access",
      refresh_token: "existing-refresh",
    }, passwordIntent.intent)
    expect(created.status).toBe("created")
    if (created.status !== "created") throw new Error("expected session")
    const existing = created.projection.snapshot
    expect((await claimOidcAuthLoginIntent()).status).toBe("claimed")
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 401 })))
    vi.spyOn(console, "error").mockImplementation(() => {})

    render(<OidcCallbackPage />)

    await waitFor(() => {
      expect(vi.mocked(fetch)).toHaveBeenCalledOnce()
    })
    expect(readAuthSessionSnapshot()).toMatchObject({
      sessionId: existing.sessionId,
      accessToken: "existing-access",
    })
  })
  it("does not let a callback use a newer password intent after its OIDC intent was superseded", async () => {
    expect((await claimOidcAuthLoginIntent()).status).toBe("claimed")
    const passwordIntent = await claimAuthLoginIntent()
    expect(passwordIntent.status).toBe("claimed")
    if (passwordIntent.status !== "claimed") throw new Error("expected password intent")
    await expect(createAuthSession({
      user: { id: "2", username: "bob" }, access_token: "password-access", refresh_token: "password-refresh",
    }, passwordIntent.intent)).resolves.toMatchObject({ status: "created" })
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      success: true, user: { id: "1", username: "alice" }, access_token: "oidc-access", refresh_token: "oidc-refresh",
    }), { headers: { "Content-Type": "application/json" } })))
    vi.spyOn(console, "error").mockImplementation(() => {})

    render(<OidcCallbackPage />)

    await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalledOnce())
    expect(readAuthSessionSnapshot()).toMatchObject({ accessToken: "password-access", userId: "2" })
  })
  it("routes unavailable OIDC intent storage to login without attempting an exchange", async () => {
    const navigation = captureNavigation()
    vi.stubGlobal("sessionStorage", undefined)
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)

    render(<OidcCallbackPage />)

    await waitFor(() => expect(navigation.targets).toEqual(["/login"]))
    expect(fetchMock).not.toHaveBeenCalled()
    navigation.restore()
  })
})

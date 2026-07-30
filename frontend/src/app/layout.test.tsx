import React from "react"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import RootLayout from "@/app/layout"
import { useAuth } from "@/contexts/auth-context"
import { apiRequest, refreshStoredAccessToken } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY } from "@/lib/auth-cache"

const route = vi.hoisted(() => ({ pathname: "/widget/chat/session" as string | null }))

vi.mock("next/navigation", () => ({
  usePathname: () => route.pathname,
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: vi.fn(async () => new Response(JSON.stringify({ team_role: "member" }), { status: 200 })),
  refreshStoredAccessToken: vi.fn(),
}))

vi.mock("@/components/auth/auth-guard", () => ({
  AuthGuard: ({ children }: { children: React.ReactNode }) => <div data-testid="auth-guard">{children}</div>,
}))

vi.mock("@/components/layout/layout-content", () => ({
  LayoutContent: ({ children }: { children: React.ReactNode }) => <div data-testid="layout-content">{children}</div>,
}))

vi.mock("@/components/voice-input-controller", () => ({
  VoiceInputController: () => <div data-testid="voice-controller" />,
}))

vi.mock("@/components/task-error-controller", () => ({
  TaskErrorController: () => <div data-testid="task-error-controller" />,
}))

function AuthProbe() {
  const { user, token, isLoading } = useAuth()
  return <span data-testid="auth-probe">{`${user?.username ?? "anonymous"}:${token ?? "none"}:${isLoading}`}</span>
}

function seedPersonalAuthCache() {
  const now = Date.now()
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
    schemaVersion: 2,
    sessionId: "personal-session",
    credentialRevision: 0,
    profileRevision: 0,
    user: { id: "owner", username: "owner" },
    token: "personal-access-token",
    refreshToken: "personal-refresh-token",
    timestamp: now,
    refreshExpiresAt: now + 60 * 60 * 1000,
  }))
}

describe("RootLayout provider boundary", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    vi.stubGlobal("React", React)
    route.pathname = "/widget/chat/session"
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { request: vi.fn(async (_name: string, callback: () => Promise<unknown>) => callback()) },
    })
  })

  afterEach(cleanup)

  it("keeps an external widget route anonymous without personal-auth side effects", async () => {
    seedPersonalAuthCache()
    const getItem = vi.spyOn(localStorage, "getItem")

    render(<RootLayout><AuthProbe /></RootLayout>)

    await waitFor(() => {
      expect(screen.getByTestId("auth-probe")).toHaveTextContent("anonymous:none:false")
    })
    expect(getItem).not.toHaveBeenCalledWith(AUTH_CACHE_KEY)
    expect(apiRequest).not.toHaveBeenCalled()
    expect(refreshStoredAccessToken).not.toHaveBeenCalled()
    expect(screen.queryByTestId("auth-guard")).not.toBeInTheDocument()
    expect(screen.queryByTestId("layout-content")).not.toBeInTheDocument()
    expect(screen.queryByTestId("voice-controller")).not.toBeInTheDocument()
    expect(screen.queryByTestId("task-error-controller")).not.toBeInTheDocument()
  })

  it("withholds route content while the pathname is unresolved", () => {
    route.pathname = null

    render(<RootLayout><AuthProbe /></RootLayout>)

    expect(screen.queryByTestId("auth-probe")).not.toBeInTheDocument()
    expect(screen.queryByTestId("auth-guard")).not.toBeInTheDocument()
    expect(screen.queryByTestId("layout-content")).not.toBeInTheDocument()
  })

  it("keeps the authenticated provider shell on general routes", async () => {
    route.pathname = "/settings"
    seedPersonalAuthCache()

    render(<RootLayout><AuthProbe /></RootLayout>)

    await waitFor(() => {
      expect(apiRequest).toHaveBeenCalledWith(expect.stringContaining("/api/teams/my-team"))
      expect(apiRequest).toHaveBeenCalledWith(expect.stringContaining("/api/mcp/apps"))
    })
    expect(screen.getByTestId("auth-guard")).toBeInTheDocument()
    expect(screen.getByTestId("layout-content")).toBeInTheDocument()
    expect(screen.getByTestId("voice-controller")).toBeInTheDocument()
    expect(screen.getByTestId("task-error-controller")).toBeInTheDocument()
  })
})

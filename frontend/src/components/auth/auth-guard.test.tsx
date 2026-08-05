import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { AuthGuard } from "./auth-guard"

const route = vi.hoisted(() => ({ pathname: "/" as string | null }))
const authState = vi.hoisted(() => ({ isAuthenticated: false, isLoading: true }))

vi.mock("next/navigation", () => ({
  usePathname: () => route.pathname,
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({
    isAuthenticated: authState.isAuthenticated,
    isLoading: authState.isLoading,
    checkAuth: vi.fn(async () => true),
  }),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({ appName: "Xagent", whiteLogoPath: "/logo-white.png" }),
}))

// Pins the exact regression this PR fixes: static export (next build)
// freezes this loading branch into out/index.html for "/", since auth only
// resolves client-side, so its content is what anonymous/non-JS crawlers see.
describe("AuthGuard loading state", () => {
  beforeEach(() => {
    route.pathname = "/"
    authState.isAuthenticated = false
    authState.isLoading = true
  })

  afterEach(cleanup)

  it("renders the home hero copy while auth resolves on the home route", () => {
    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(screen.getByText("home.hero.title")).toBeInTheDocument()
    expect(screen.getByText("home.hero.subtitle")).toBeInTheDocument()
    expect(screen.queryByTestId("children")).not.toBeInTheDocument()
  })

  it("keeps the generic spinner copy on other routes", () => {
    route.pathname = "/settings"

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(screen.getByText("common.loading")).toBeInTheDocument()
    expect(screen.queryByText("home.hero.title")).not.toBeInTheDocument()
  })
})

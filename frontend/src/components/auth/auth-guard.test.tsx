import React from "react"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { AuthGuard } from "./auth-guard"
import { resolveTranslation, type TranslationKey } from "@/i18n/translations"

const route = vi.hoisted(() => ({ pathname: "/" as string | null }))
const authState = vi.hoisted(() => ({ isAuthenticated: false, isLoading: true }))
const routerPush = vi.hoisted(() => vi.fn())
const fetchUserPreferencesMock = vi.hoisted(() => vi.fn())

vi.mock("next/navigation", () => ({
  usePathname: () => route.pathname,
  useRouter: () => ({ push: routerPush }),
}))

vi.mock("@/lib/user-preferences", () => ({
  fetchUserPreferences: fetchUserPreferencesMock,
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({
    isAuthenticated: authState.isAuthenticated,
    isLoading: authState.isLoading,
    checkAuth: vi.fn(async () => true),
  }),
}))

// Resolves against the real English translation table (rather than an
// identity passthrough) so this pins the actual copy shipped in out/index.html.
vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      resolveTranslation("en", key as TranslationKey, vars),
  }),
}))

// Reads whiteLogoPath/appName off the real defaultBranding so this mock can't
// drift from the actual default asset path.
vi.mock("@/lib/branding", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/branding")>()
  return {
    ...actual,
    getBrandingFromEnv: () => ({
      appName: actual.defaultBranding.appName,
      whiteLogoPath: actual.defaultBranding.whiteLogoPath,
    }),
  }
})

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

  // Literal strings on purpose: asserting via resolveTranslation would pass
  // even if the key were removed from the table (it falls back to the key on
  // both sides), which is exactly the regression this test exists to catch.
  it("renders the home hero copy while auth resolves on the home route", () => {
    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(screen.getByText("Welcome to Xagent")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Build, deploy, and scale intelligent agents that work for you — no code required.",
      ),
    ).toBeInTheDocument()
    expect(screen.getByRole("img", { name: "Xagent" })).toHaveAttribute(
      "src",
      "/xagent_white_logo.png",
    )
    expect(screen.queryByTestId("children")).not.toBeInTheDocument()
  })

  it("keeps the generic spinner copy on other routes", () => {
    route.pathname = "/settings"

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(screen.getByText("Loading...")).toBeInTheDocument()
    expect(screen.queryByText("Welcome to Xagent")).not.toBeInTheDocument()
  })
})

// Pins the exact regression found during live onboarding verification: GET
// /api/auth/me nests preferences under `user.preferences`, not top-level -
// fetchUserPreferences must unwrap that shape correctly, and this hook must
// act on the real result, not a shape that silently always reads as "not
// onboarded".
describe("AuthGuard onboarding redirect", () => {
  beforeEach(() => {
    route.pathname = "/settings"
    authState.isAuthenticated = true
    authState.isLoading = false
    routerPush.mockClear()
    fetchUserPreferencesMock.mockReset()
  })

  afterEach(cleanup)

  it("redirects an authenticated but unonboarded user to /onboarding", async () => {
    fetchUserPreferencesMock.mockResolvedValue({})

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    await waitFor(() => expect(routerPush).toHaveBeenCalledWith("/onboarding"))
  })

  it("does not redirect a user whose preferences already have onboarded: true", async () => {
    fetchUserPreferencesMock.mockResolvedValue({ onboarded: true })

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalled())
    expect(routerPush).not.toHaveBeenCalledWith("/onboarding")
  })

  it("does not check preferences at all while already on the onboarding page", () => {
    route.pathname = "/onboarding"

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(fetchUserPreferencesMock).not.toHaveBeenCalled()
  })

  it("does not check preferences on public auth pages", () => {
    route.pathname = "/login"
    authState.isAuthenticated = false

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(fetchUserPreferencesMock).not.toHaveBeenCalled()
  })

  // Pins a self-review finding: the "checked once" ref used to latch true
  // BEFORE the async check resolved. If a navigation cancelled that check
  // in flight, the ref stayed latched forever and no later route ever got
  // checked again in that tab. The ref must only latch once a check
  // actually completes, so a cancelled check can be retried on the next one.
  it("retries the check on the next route if navigation cancels it before the fetch resolves", async () => {
    fetchUserPreferencesMock
      .mockImplementationOnce(() => new Promise(() => {})) // never resolves - simulates an in-flight, then-cancelled check
      .mockResolvedValueOnce({ onboarded: true })

    const { rerender } = render(<AuthGuard><div data-testid="children" /></AuthGuard>)
    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalledTimes(1))

    // Navigate to a different authenticated route before the first check resolves.
    route.pathname = "/dashboard"
    rerender(<AuthGuard><div data-testid="children" /></AuthGuard>)

    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalledTimes(2))
  })
})

import React from "react"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { AuthGuard } from "./auth-guard"
import { resolveTranslation, type TranslationKey } from "@/i18n/translations"

const route = vi.hoisted(() => ({ pathname: "/" as string | null }))
const authState = vi.hoisted(() => ({
  isAuthenticated: false,
  isLoading: true,
  user: { id: "user-a" } as { id: string } | null,
}))
const routerPush = vi.hoisted(() => vi.fn())
const routerReplace = vi.hoisted(() => vi.fn())
const fetchUserPreferencesMock = vi.hoisted(() => vi.fn())
const consumeOnboardingSaveEscapeFlagMock = vi.hoisted(() => vi.fn(() => false))

vi.mock("next/navigation", () => ({
  usePathname: () => route.pathname,
  useRouter: () => ({ push: routerPush, replace: routerReplace }),
}))

vi.mock("@/lib/user-preferences", () => ({
  fetchUserPreferences: fetchUserPreferencesMock,
  consumeOnboardingSaveEscapeFlag: consumeOnboardingSaveEscapeFlagMock,
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({
    isAuthenticated: authState.isAuthenticated,
    isLoading: authState.isLoading,
    user: authState.user,
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
    authState.user = { id: "user-a" }
    routerPush.mockClear()
    routerReplace.mockClear()
    fetchUserPreferencesMock.mockReset()
    consumeOnboardingSaveEscapeFlagMock.mockReset()
    consumeOnboardingSaveEscapeFlagMock.mockReturnValue(false)
  })

  afterEach(cleanup)

  // replace, not push: a `push` here leaves the pre-redirect page in
  // history, so a single Back press would return the user there with the
  // ref already latched - permanently bypassing onboarding with no error
  // condition required (a PR review finding).
  it("redirects an authenticated but unonboarded user to /onboarding via replace", async () => {
    fetchUserPreferencesMock.mockResolvedValue({})

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/onboarding"))
    expect(routerPush).not.toHaveBeenCalledWith("/onboarding")
  })

  it("does not redirect a user whose preferences already have onboarded: true", async () => {
    fetchUserPreferencesMock.mockResolvedValue({ onboarded: true })

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalled())
    expect(routerReplace).not.toHaveBeenCalledWith("/onboarding")
  })

  // Pins a finding verified during PR review: the GET boundary doesn't
  // validate this field's type, so a malformed stored value (a truthy
  // non-boolean, e.g. the string "false") must still redirect rather than
  // being read as "already onboarded" via a loose truthiness check.
  it("redirects when the stored onboarded value is a truthy non-boolean (e.g. the string \"false\")", async () => {
    fetchUserPreferencesMock.mockResolvedValue({ onboarded: "false" as unknown as boolean })

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/onboarding"))
  })

  // Pins the core behavior checkedOnboardingRef exists to implement, found
  // untested in full-feature self-review: every other latch scenario (the
  // escape flag, a cancelled in-flight check) is pinned, but not the plain
  // "checked once per app load" happy path itself. A regression that never
  // latches on a successful check would silently re-fetch preferences on
  // every single route change for every user, forever.
  it("does not re-check preferences on a later route once a check has successfully completed", async () => {
    fetchUserPreferencesMock.mockResolvedValue({ onboarded: true })

    const { rerender } = render(<AuthGuard><div data-testid="children" /></AuthGuard>)
    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalledTimes(1))

    route.pathname = "/dashboard"
    rerender(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(fetchUserPreferencesMock).toHaveBeenCalledTimes(1)
  })

  // Pins a finding verified during PR review: AuthGuard doesn't remount
  // across a client-side auth swap - AuthProvider reacts to a same-origin
  // `storage` event (a DIFFERENT user logging in from another tab) by
  // updating `isAuthenticated`/`user` in place, in this same mounted
  // instance. A bare "have we ever checked" boolean would stay latched from
  // user A's completed check and let user B through with no check of their
  // own at all - the ref must be keyed by user id so a genuine identity
  // change always forces a fresh check.
  it("re-checks preferences when the authenticated user changes, even though a check already completed for the previous user", async () => {
    fetchUserPreferencesMock.mockResolvedValue({ onboarded: true })

    const { rerender } = render(<AuthGuard><div data-testid="children" /></AuthGuard>)
    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalledTimes(1))

    fetchUserPreferencesMock.mockResolvedValue({})
    authState.user = { id: "user-b" }
    rerender(<AuthGuard><div data-testid="children" /></AuthGuard>)

    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/onboarding"))
  })

  // Pins a PR review finding (regression in this mechanism's first version):
  // the flag used to be checked only inside the async preferences check,
  // which the "already checked" ref guard skips entirely once a check has
  // already run this app-load - the common case, since the user usually
  // reached /onboarding via an earlier check on another page that already
  // latched the ref. That left the flag unconsumed on the very escape it
  // was meant for, letting it linger to wrongly suppress some unrelated
  // LATER onboarding check instead. It's now consumed synchronously, ahead
  // of that guard, so this must not even call fetchUserPreferences at all -
  // there is nothing left to check once the escape is honored.
  it("does not redirect (and skips the preferences check entirely) when the onboarding page's save-escape flag is set, and still latches as checked", async () => {
    consumeOnboardingSaveEscapeFlagMock.mockReturnValue(true)

    const { rerender } = render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(consumeOnboardingSaveEscapeFlagMock).toHaveBeenCalledTimes(1)
    expect(fetchUserPreferencesMock).not.toHaveBeenCalled()
    expect(routerReplace).not.toHaveBeenCalledWith("/onboarding")

    // The ref must have latched (as "checked") even though the check was
    // skipped - otherwise the very next navigation re-triggers a real check
    // and, since the flag is one-shot and already consumed, redirects
    // anyway, defeating the whole point of the escape hatch.
    consumeOnboardingSaveEscapeFlagMock.mockReturnValue(false)
    route.pathname = "/dashboard"
    rerender(<AuthGuard><div data-testid="children" /></AuthGuard>)
    expect(fetchUserPreferencesMock).not.toHaveBeenCalled()
  })

  // Pins a PR review finding: the escape flag itself is identity-agnostic
  // (a bare timestamp) unless the currently authenticated user's id is
  // passed through so consumeOnboardingSaveEscapeFlag can bind/verify
  // against it - without this, a cross-tab identity swap could let a
  // different user consume a leftover flag meant for someone else.
  it("passes the current user's id to consumeOnboardingSaveEscapeFlag", async () => {
    authState.user = { id: "user-a" }
    consumeOnboardingSaveEscapeFlagMock.mockReturnValue(false)
    fetchUserPreferencesMock.mockResolvedValue(null)

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(consumeOnboardingSaveEscapeFlagMock).toHaveBeenCalledWith("user-a")
    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalled())
  })

  it("passes null to consumeOnboardingSaveEscapeFlag when no user has resolved yet", () => {
    authState.user = null
    consumeOnboardingSaveEscapeFlagMock.mockReturnValue(false)

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    expect(consumeOnboardingSaveEscapeFlagMock).toHaveBeenCalledWith(null)
    // With no user, the effect returns right after this check - the
    // preferences fetch never fires.
    expect(fetchUserPreferencesMock).not.toHaveBeenCalled()
  })

  // Pins a PR review finding: fetchUserPreferences returns null (not {}) on
  // a failed/unreachable GET - that's "unknown," not "confirmed not
  // onboarded," so it must not redirect an already-onboarded user into the
  // wizard just because a transient error hit this one GET.
  it("does not redirect when the preferences fetch fails (returns null)", async () => {
    fetchUserPreferencesMock.mockResolvedValue(null)

    render(<AuthGuard><div data-testid="children" /></AuthGuard>)

    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalledTimes(1))
    expect(routerReplace).not.toHaveBeenCalledWith("/onboarding")
  })

  it("retries on the next route after a failed (null) check, since a failure must not latch the ref", async () => {
    fetchUserPreferencesMock.mockResolvedValue(null)

    const { rerender } = render(<AuthGuard><div data-testid="children" /></AuthGuard>)
    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalledTimes(1))

    route.pathname = "/dashboard"
    rerender(<AuthGuard><div data-testid="children" /></AuthGuard>)

    await waitFor(() => expect(fetchUserPreferencesMock).toHaveBeenCalledTimes(2))
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

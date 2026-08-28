import { describe, expect, it, vi, beforeEach } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

import {
  consumeOnboardingSaveEscapeFlag,
  fetchUserPreferences,
  markOnboardingSaveEscaped,
  updateUserPreferences,
} from "./user-preferences"

describe("fetchUserPreferences", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  // Pins the exact shape returned by GET /api/auth/me (UserProfileResponse):
  // preferences live under `user.preferences`, not top-level - a live
  // onboarding-flow verification caught this reading the wrong path, which
  // silently made every call return {} and re-trigger onboarding forever.
  it("reads preferences from the nested user.preferences field", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        message: "ok",
        user: { id: 1, username: "a", email: "a@b.com", is_admin: false, preferences: { onboarded: true, voice: "warm" } },
      }),
    })

    const result = await fetchUserPreferences()

    expect(result).toEqual({ onboarded: true, voice: "warm" })
  })

  // Pins a PR review finding: a failed/unreachable GET is "unknown," not
  // "confirmed not onboarded" - returning {} for both used to make a
  // transient error redirect an already-onboarded user into the wizard.
  it("returns null (not {}) when the response is not ok", async () => {
    apiRequestMock.mockResolvedValue({ ok: false })

    expect(await fetchUserPreferences()).toBeNull()
  })

  it("returns {} (a real, successful answer) when user.preferences is missing or not an object", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, message: "ok", user: { id: 1, username: "a", email: "a@b.com", is_admin: false } }),
    })

    expect(await fetchUserPreferences()).toEqual({})
  })

  it("returns null (not {}) when the request throws", async () => {
    apiRequestMock.mockRejectedValue(new Error("network down"))

    expect(await fetchUserPreferences()).toBeNull()
  })
})

describe("updateUserPreferences", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it("PATCHes the given fields and reports ok: true on success", async () => {
    apiRequestMock.mockResolvedValue({ ok: true })

    const result = await updateUserPreferences({ onboarded: true, department: "sales" })

    expect(result).toEqual({ ok: true, retryable: true })
    const [url, options] = apiRequestMock.mock.calls[0]
    expect(url).toContain("/api/auth/me/preferences")
    expect(options.method).toBe("PATCH")
    expect(JSON.parse(options.body)).toEqual({ onboarded: true, department: "sales" })
  })

  it("reports ok: false, retryable: true without throwing when the request itself throws (network failure)", async () => {
    apiRequestMock.mockRejectedValue(new Error("network down"))

    expect(await updateUserPreferences({ onboarded: true })).toEqual({ ok: false, retryable: true })
  })

  // Pins a PR review finding: a caller (handleLaunch in page.tsx) that gives
  // up and proceeds anyway after repeated failures must not treat a
  // transient error the same as a permanent one - retrying an identical
  // rejected payload will only ever 4xx again, so proceeding to an
  // irreversible action afterward would be unsafe.
  it("reports retryable: true for a 5xx server error", async () => {
    apiRequestMock.mockResolvedValue({ ok: false, status: 503 })

    expect(await updateUserPreferences({ onboarded: true })).toEqual({ ok: false, retryable: true })
  })

  it("reports retryable: false for a 4xx client error (e.g. a rejected payload)", async () => {
    apiRequestMock.mockResolvedValue({ ok: false, status: 422 })

    expect(await updateUserPreferences({ onboarded: true })).toEqual({ ok: false, retryable: false })
  })

  // Pins the exact 4xx/5xx boundary: an off-by-one (>. instead of >=, or a
  // hardcoded wrong threshold) would silently change whether a failing
  // save is allowed to escalate to an irreversible action in handleLaunch.
  it("reports retryable: true at exactly the 500 boundary", async () => {
    apiRequestMock.mockResolvedValue({ ok: false, status: 500 })

    expect(await updateUserPreferences({ onboarded: true })).toEqual({ ok: false, retryable: true })
  })

  it("reports retryable: false at exactly 499, just below the boundary", async () => {
    apiRequestMock.mockResolvedValue({ ok: false, status: 499 })

    expect(await updateUserPreferences({ onboarded: true })).toEqual({ ok: false, retryable: false })
  })
})

// Coordinates persistAndLeave's give-up-after-repeated-failures escape hatch
// with AuthGuard's independent onboarding check - without this, escaping a
// failed save just gets immediately bounced back (see full-feature
// self-review finding on the onboarding-flow PR).
describe("markOnboardingSaveEscaped / consumeOnboardingSaveEscapeFlag", () => {
  beforeEach(() => {
    window.sessionStorage.clear()
  })

  it("consume returns false when nothing was marked", () => {
    expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(false)
  })

  it("consume returns true once after marking, for the SAME user, then false again (one-shot)", () => {
    markOnboardingSaveEscaped("user-a")

    expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(true)
    expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(false)
  })

  // Pins a PR review finding: a bare timestamp is identity-agnostic - the
  // same cross-tab-identity-swap vulnerability class the
  // checkedOnboardingUserIdRef fix in auth-guard.tsx closed for the normal
  // onboarding check also applied here, letting a DIFFERENT user who logs
  // in in this tab within the TTL window consume user A's leftover escape
  // and skip their own mandatory onboarding check.
  it("consume returns false (and still discards the flag) for a DIFFERENT user than the one who set it", () => {
    markOnboardingSaveEscaped("user-a")

    expect(consumeOnboardingSaveEscapeFlag("user-b")).toBe(false)
    // One-shot even on a mismatch - a second read (even by the correct
    // user) must not find a flag that was already consumed-and-discarded.
    expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(false)
  })

  it("consume returns false when no identity has resolved yet (null)", () => {
    markOnboardingSaveEscaped("user-a")

    expect(consumeOnboardingSaveEscapeFlag(null)).toBe(false)
  })

  it("mark no-ops (and consume finds nothing) when no user id is available to bind to", () => {
    markOnboardingSaveEscaped(undefined)

    expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(false)
  })

  it("does not throw when sessionStorage is unavailable", () => {
    const original = window.sessionStorage
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      get() {
        throw new Error("storage disabled")
      },
    })

    try {
      expect(() => markOnboardingSaveEscaped("user-a")).not.toThrow()
      expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(false)
    } finally {
      Object.defineProperty(window, "sessionStorage", { configurable: true, value: original })
    }
  })

  // Pins the 30s TTL added as defense-in-depth on top of the ordering fix:
  // a flag that's somehow still unconsumed after this long (a future caller
  // that stops consuming it promptly, a tab left open mid-navigation) must
  // expire rather than silently suppress an unrelated, much later redirect.
  it("consume returns false once the flag is older than the TTL, and still clears it", () => {
    vi.useFakeTimers()
    try {
      markOnboardingSaveEscaped("user-a")
      vi.advanceTimersByTime(30_001)

      expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(false)
      expect(window.sessionStorage.getItem("xagent-onboarding-save-escape")).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it("consume still returns true just under the TTL boundary", () => {
    vi.useFakeTimers()
    try {
      markOnboardingSaveEscaped("user-a")
      vi.advanceTimersByTime(29_999)

      expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(true)
    } finally {
      vi.useRealTimers()
    }
  })

  it("consume returns false and clears a malformed (non-JSON) stored value", () => {
    window.sessionStorage.setItem("xagent-onboarding-save-escape", "not-json")

    expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(false)
    expect(window.sessionStorage.getItem("xagent-onboarding-save-escape")).toBeNull()
  })

  it("consume returns false and clears a well-formed JSON value missing the expected shape", () => {
    window.sessionStorage.setItem("xagent-onboarding-save-escape", JSON.stringify({ foo: "bar" }))

    expect(consumeOnboardingSaveEscapeFlag("user-a")).toBe(false)
    expect(window.sessionStorage.getItem("xagent-onboarding-save-escape")).toBeNull()
  })
})

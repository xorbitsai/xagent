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

    expect(result).toEqual({ ok: true })
    const [url, options] = apiRequestMock.mock.calls[0]
    expect(url).toContain("/api/auth/me/preferences")
    expect(options.method).toBe("PATCH")
    expect(JSON.parse(options.body)).toEqual({ onboarded: true, department: "sales" })
  })

  it("reports ok: false without throwing when the request fails", async () => {
    apiRequestMock.mockRejectedValue(new Error("network down"))

    expect(await updateUserPreferences({ onboarded: true })).toEqual({ ok: false })
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
    expect(consumeOnboardingSaveEscapeFlag()).toBe(false)
  })

  it("consume returns true once after marking, then false again (one-shot)", () => {
    markOnboardingSaveEscaped()

    expect(consumeOnboardingSaveEscapeFlag()).toBe(true)
    expect(consumeOnboardingSaveEscapeFlag()).toBe(false)
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
      expect(() => markOnboardingSaveEscaped()).not.toThrow()
      expect(consumeOnboardingSaveEscapeFlag()).toBe(false)
    } finally {
      Object.defineProperty(window, "sessionStorage", { configurable: true, value: original })
    }
  })
})

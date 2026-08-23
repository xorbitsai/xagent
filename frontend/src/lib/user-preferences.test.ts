import { describe, expect, it, vi, beforeEach } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

import { fetchUserPreferences, updateUserPreferences } from "./user-preferences"

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

  it("returns {} when the response is not ok", async () => {
    apiRequestMock.mockResolvedValue({ ok: false })

    expect(await fetchUserPreferences()).toEqual({})
  })

  it("returns {} when user.preferences is missing or not an object", async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true, message: "ok", user: { id: 1, username: "a", email: "a@b.com", is_admin: false } }),
    })

    expect(await fetchUserPreferences()).toEqual({})
  })

  it("returns {} when the request throws", async () => {
    apiRequestMock.mockRejectedValue(new Error("network down"))

    expect(await fetchUserPreferences()).toEqual({})
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

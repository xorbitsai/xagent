import { afterEach, describe, expect, it, vi } from "vitest"

import {
  getBrowserLocationHash,
  getBrowserLocationHostname,
  getBrowserLocationOrigin,
  getBrowserLocationPathname,
  getBrowserLocationSearch,
  replaceBrowserLocation,
} from "./browser-location"

describe("browser location helpers", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("reads browser origin and hostname when available", () => {
    expect(getBrowserLocationOrigin()).toBe(window.location.origin)
    expect(getBrowserLocationHostname()).toBe(window.location.hostname)
  })

  it("returns empty strings when window is unavailable", () => {
    vi.stubGlobal("window", undefined)

    expect(getBrowserLocationOrigin()).toBe("")
    expect(getBrowserLocationHostname()).toBe("")
    expect(getBrowserLocationHash()).toBe("")
    expect(getBrowserLocationPathname()).toBe("")
    expect(getBrowserLocationSearch()).toBe("")
  })

  it("returns empty strings when location access throws", () => {
    vi.stubGlobal("window", {
      get location() {
        throw new DOMException("Blocked", "SecurityError")
      },
    })

    expect(getBrowserLocationOrigin()).toBe("")
    expect(getBrowserLocationHostname()).toBe("")
    expect(getBrowserLocationHash()).toBe("")
    expect(getBrowserLocationPathname()).toBe("")
    expect(getBrowserLocationSearch()).toBe("")
  })

  it("replaces the current history entry instead of assigning", () => {
    const replace = vi.fn()
    const assign = vi.fn()
    vi.stubGlobal("window", { location: { replace, assign } })

    replaceBrowserLocation("/next")

    expect(replace).toHaveBeenCalledTimes(1)
    expect(replace).toHaveBeenCalledWith("/next")
    expect(assign).not.toHaveBeenCalled()
  })

  it("does nothing when window is unavailable while replacing", () => {
    vi.stubGlobal("window", undefined)

    expect(() => replaceBrowserLocation("/next")).not.toThrow()
  })
})

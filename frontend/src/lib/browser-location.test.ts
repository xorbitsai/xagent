import { afterEach, describe, expect, it, vi } from "vitest"

import {
  getBrowserLocationHostname,
  getBrowserLocationOrigin,
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
  })

  it("returns empty strings when location access throws", () => {
    vi.stubGlobal("window", {
      get location() {
        throw new DOMException("Blocked", "SecurityError")
      },
    })

    expect(getBrowserLocationOrigin()).toBe("")
    expect(getBrowserLocationHostname()).toBe("")
  })
})

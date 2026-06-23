import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const getApiUrlMock = vi.hoisted(() => vi.fn())
const browserOriginMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/utils", () => ({
  getApiUrl: getApiUrlMock,
}))

vi.mock("@/lib/browser-location", () => ({
  getBrowserLocationOrigin: browserOriginMock,
}))

import { getApiSnippetTarget } from "./api-snippet-base-url"

describe("getApiSnippetTarget", () => {
  beforeEach(() => {
    getApiUrlMock.mockReset()
    browserOriginMock.mockReset()
    browserOriginMock.mockReturnValue(window.location.origin)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("uses the configured API URL when present", () => {
    getApiUrlMock.mockReturnValue(" https://api.example.test/ ")

    expect(getApiSnippetTarget()).toEqual({
      baseUrl: "https://api.example.test",
    })
  })

  it("resolves relative API URLs against the browser origin", () => {
    getApiUrlMock.mockReturnValue("/api")

    expect(getApiSnippetTarget()).toEqual({
      baseUrl: `${window.location.origin}/api`,
    })
  })

  it("falls back to the browser origin for same-origin deployments", () => {
    getApiUrlMock.mockReturnValue("")

    expect(getApiSnippetTarget()).toEqual({
      baseUrl: window.location.origin,
    })
  })

  it("returns an empty string when no base URL is available", () => {
    getApiUrlMock.mockReturnValue("")
    browserOriginMock.mockReturnValue("")

    expect(getApiSnippetTarget()).toEqual({
      baseUrl: "",
    })
  })

  it("returns an empty string when a relative API URL has no browser origin", () => {
    getApiUrlMock.mockReturnValue("/api")
    browserOriginMock.mockReturnValue("")

    expect(getApiSnippetTarget()).toEqual({
      baseUrl: "",
    })
  })
})

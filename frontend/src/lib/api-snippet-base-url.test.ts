import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const getApiUrlMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/utils", () => ({
  getApiUrl: getApiUrlMock,
}))

import { getApiSnippetTarget } from "./api-snippet-base-url"

describe("getApiSnippetTarget", () => {
  beforeEach(() => {
    getApiUrlMock.mockReset()
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

  it("falls back to the browser origin for same-origin deployments", () => {
    getApiUrlMock.mockReturnValue("")

    expect(getApiSnippetTarget()).toEqual({
      baseUrl: window.location.origin,
    })
  })

  it("returns an empty string when no base URL is available", () => {
    getApiUrlMock.mockReturnValue("")
    vi.stubGlobal("window", undefined)

    expect(getApiSnippetTarget()).toEqual({
      baseUrl: "",
    })
  })
})

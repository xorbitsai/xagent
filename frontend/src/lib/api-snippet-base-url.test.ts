import { beforeEach, describe, expect, it, vi } from "vitest"

const getApiUrlMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/utils", () => ({
  getApiUrl: getApiUrlMock,
}))

import { getApiSnippetTarget } from "./api-snippet-base-url"

describe("getApiSnippetTarget", () => {
  beforeEach(() => {
    getApiUrlMock.mockReset()
  })

  it("uses the configured API URL when present", () => {
    getApiUrlMock.mockReturnValue("https://api.example.test")

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
})

import { describe, expect, it } from "vitest"

import { normalizeApiSnippetBaseUrl } from "./api-snippet-target"

describe("normalizeApiSnippetBaseUrl", () => {
  it("trims whitespace and trailing slashes", () => {
    expect(normalizeApiSnippetBaseUrl(" https://api.example.test/// ")).toBe(
      "https://api.example.test"
    )
  })

  it("rejects empty base URLs", () => {
    expect(() => normalizeApiSnippetBaseUrl(" / ")).toThrow(
      "Unable to determine API snippet base URL"
    )
  })
})

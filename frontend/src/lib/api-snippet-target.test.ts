import { describe, expect, it } from "vitest"

import { normalizeApiSnippetBaseUrl } from "./api-snippet-target"

describe("normalizeApiSnippetBaseUrl", () => {
  it("trims whitespace and trailing slashes", () => {
    expect(normalizeApiSnippetBaseUrl(" https://api.example.test/// ")).toBe(
      "https://api.example.test"
    )
  })

  it("handles empty base URLs gracefully", () => {
    expect(normalizeApiSnippetBaseUrl(" / ")).toBe("")
  })
})

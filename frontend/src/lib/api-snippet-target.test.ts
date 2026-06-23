import { describe, expect, it } from "vitest"

import {
  normalizeApiSnippetBaseUrl,
  resolveApiSnippetBaseUrl,
} from "./api-snippet-target"

describe("normalizeApiSnippetBaseUrl", () => {
  it("trims whitespace and trailing slashes", () => {
    expect(normalizeApiSnippetBaseUrl(" https://api.example.test/// ")).toBe(
      "https://api.example.test"
    )
  })

  it("handles empty base URLs gracefully", () => {
    expect(normalizeApiSnippetBaseUrl("   ")).toBe("")
  })

  it("preserves a single slash representing the root path", () => {
    expect(normalizeApiSnippetBaseUrl(" / ")).toBe("/")
  })
})

describe("resolveApiSnippetBaseUrl", () => {
  it("keeps absolute HTTP URLs", () => {
    expect(resolveApiSnippetBaseUrl(" https://api.example.test/// ")).toBe(
      "https://api.example.test"
    )
  })

  it("resolves relative URLs against an absolute browser origin", () => {
    expect(resolveApiSnippetBaseUrl("/api", "https://app.example.test")).toBe(
      "https://app.example.test/api"
    )
  })

  it("resolves the root path against an absolute browser origin", () => {
    expect(resolveApiSnippetBaseUrl("/", "https://app.example.test")).toBe(
      "https://app.example.test"
    )
  })

  it("returns an empty string when a relative URL has no usable origin", () => {
    expect(resolveApiSnippetBaseUrl("/api", "")).toBe("")
  })

  it("rejects non-HTTP absolute URLs after URL resolution", () => {
    expect(resolveApiSnippetBaseUrl("javascript:alert(1)", "https://app.example.test")).toBe("")
    expect(resolveApiSnippetBaseUrl("ftp://api.example.test", "https://app.example.test")).toBe("")
    expect(resolveApiSnippetBaseUrl("ws://api.example.test", "https://app.example.test")).toBe("")
  })
})

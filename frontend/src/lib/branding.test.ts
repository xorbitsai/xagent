import { describe, expect, it, vi } from "vitest"
import { resolveMetadataBase } from "./branding"

describe("resolveMetadataBase", () => {
  it("resolves a well-formed site URL", () => {
    expect(resolveMetadataBase("https://example.com")).toEqual(new URL("https://example.com"))
  })

  it("falls back to the default site URL when given a malformed one", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    expect(resolveMetadataBase("not-a-valid-url")).toEqual(new URL("https://cloud.xagent.co"))
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("not-a-valid-url"))

    errorSpy.mockRestore()
  })
})

import { describe, expect, it } from "vitest"

import { getUploadErrorMessage, parseApiResponse } from "@/lib/api-wrapper"

const MESSAGES = {
  generic: "Upload failed",
  tooLarge: "File too large",
  proxy: "Proxy rejected upload",
}

describe("api-wrapper upload helpers", () => {
  it("parses json error payloads", async () => {
    const response = new Response(JSON.stringify({ detail: "too large" }), {
      status: 413,
      headers: { "Content-Type": "application/json" },
    })

    const parsed = await parseApiResponse(response)

    expect(parsed.data).toEqual({ detail: "too large" })
    expect(parsed.isHtml).toBe(false)
  })

  it("returns empty parsed payload for empty body", async () => {
    const response = new Response(null, {
      status: 500,
      headers: { "Content-Type": "text/plain" },
    })

    const parsed = await parseApiResponse(response)

    expect(parsed.data).toBeNull()
    expect(parsed.text).toBeNull()
    expect(parsed.isHtml).toBe(false)
  })

  it("treats malformed non-json bodies as raw text", async () => {
    const response = new Response("{not-json", {
      status: 500,
      headers: { "Content-Type": "text/plain" },
    })

    const parsed = await parseApiResponse(response)

    expect(parsed.data).toBeNull()
    expect(parsed.text).toBe("{not-json")
    expect(parsed.isHtml).toBe(false)
  })

  it("preserves html proxy bodies even when content type claims json", async () => {
    const response = new Response("<html><body>502 Bad Gateway</body></html>", {
      status: 502,
      headers: { "Content-Type": "application/json" },
    })

    const parsed = await parseApiResponse(response)
    const message = getUploadErrorMessage(response, parsed, MESSAGES)

    expect(parsed.data).toBeNull()
    expect(parsed.text).toContain("502 Bad Gateway")
    expect(parsed.isHtml).toBe(true)
    expect(message).toBe("Proxy rejected upload")
  })

  it("falls back to friendly proxy error for html responses", async () => {
    const response = new Response("<html><body>413 Request Entity Too Large</body></html>", {
      status: 413,
      headers: { "Content-Type": "text/html" },
    })

    const parsed = await parseApiResponse(response)
    const message = getUploadErrorMessage(response, parsed, MESSAGES)

    expect(parsed.isHtml).toBe(true)
    expect(message).toBe("File too large")
  })

  it("prefers detail messages from parsed json", () => {
    const response = new Response(null, { status: 400 })
    const message = getUploadErrorMessage(response, {
      data: { detail: "explicit detail" },
      text: null,
      isHtml: false,
    }, MESSAGES)

    expect(message).toBe("explicit detail")
  })

  it("returns truncated raw text for non-413 non-html responses", () => {
    const response = new Response(null, { status: 500 })
    const rawText = "x".repeat(240)
    const message = getUploadErrorMessage(response, {
      data: null,
      text: rawText,
      isHtml: false,
    }, MESSAGES)

    expect(message).toHaveLength(203)
    expect(message.endsWith("...")).toBe(true)
  })

  it("falls back to generic when nothing else is available", () => {
    const response = new Response(null, { status: 500 })
    const message = getUploadErrorMessage(response, {
      data: null,
      text: null,
      isHtml: false,
    }, MESSAGES)

    expect(message).toBe("Upload failed")
  })
})

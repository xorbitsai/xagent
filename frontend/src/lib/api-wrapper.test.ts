import { describe, expect, it } from "vitest"

import { getUploadErrorMessage, parseApiResponse } from "@/lib/api-wrapper"

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

  it("falls back to friendly proxy error for html responses", async () => {
    const response = new Response("<html><body>413 Request Entity Too Large</body></html>", {
      status: 413,
      headers: { "Content-Type": "text/html" },
    })

    const parsed = await parseApiResponse(response)
    const message = getUploadErrorMessage(response, parsed, {
      generic: "Upload failed",
      tooLarge: "File too large",
      proxy: "Proxy rejected upload",
    })

    expect(parsed.isHtml).toBe(true)
    expect(message).toBe("File too large")
  })
})

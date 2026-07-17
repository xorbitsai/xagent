import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  apiRequest,
  getApiErrorMessage,
  getUploadErrorMessage,
  parseApiResponse,
  refreshStoredAccessToken,
} from "@/lib/api-wrapper"
import { clearStoredAuth, readAuthCache, writeAuthCache } from "@/lib/auth-cache"

function mockNavigatorLocks(
  beforeCallback: () => void | Promise<void> = () => {}
) {
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: {
      request: vi.fn(async (
        _name: string,
        callback: () => Promise<unknown>
      ) => {
        await beforeCallback()
        return callback()
      }),
    },
  })
}

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

describe("api-wrapper API error helpers", () => {
  it("prefers detail messages from parsed json", () => {
    const response = new Response(null, { status: 503 })
    const message = getApiErrorMessage(response, {
      data: { detail: "Startup file storage sync failed" },
      text: null,
      isHtml: false,
    }, "Request failed")

    expect(message).toBe("Startup file storage sync failed")
  })
})

describe("api-wrapper auth refresh", () => {
  const user = { id: "1", username: "alice", email: null, is_admin: false }

  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
    mockNavigatorLocks()
  })

  it("coalesces concurrent refreshes and retries every waiting request", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)

    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input, options) => {
        const url = String(input)
        const authorization = new Headers(options?.headers).get("Authorization")

        if (url.endsWith("/api/auth/refresh")) {
          return new Response(JSON.stringify({
            success: true,
            access_token: "new-access",
            refresh_token: "new-refresh",
            expires_in: 120,
            refresh_expires_in: 240,
          }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        }

        if (authorization === "Bearer old-access") {
          return new Response(null, {
            status: 401,
            headers: { "Error-Type": "TokenExpired" },
          })
        }

        return new Response(null, { status: 200 })
      }
    )

    const [first, second] = await Promise.all([
      apiRequest("http://api.local/protected"),
      apiRequest("http://api.local/protected"),
    ])

    expect(first.status).toBe(200)
    expect(second.status).toBe(200)
    expect(fetchMock.mock.calls.filter(([input]) =>
      String(input).endsWith("/api/auth/refresh")
    )).toHaveLength(1)
    expect(readAuthCache()?.refreshToken).toBe("new-refresh")
  })

  it("reuses a token refreshed by another tab while waiting for the lock", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    const fetchMock = vi.spyOn(globalThis, "fetch")

    mockNavigatorLocks(() => {
      writeAuthCache(user, "other-tab-access", "other-tab-refresh", 120, 240)
    })

    const result = await refreshStoredAccessToken("old-access")

    expect(result).toEqual({ accessToken: "other-tab-access" })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("reuses a token from another tab when the caller started without one", async () => {
    writeAuthCache(user, null, "old-refresh", 120, 240)
    const fetchMock = vi.spyOn(globalThis, "fetch")

    mockNavigatorLocks(() => {
      writeAuthCache(user, "other-tab-access", "other-tab-refresh", 120, 240)
    })

    const result = await refreshStoredAccessToken(null)

    expect(result).toEqual({ accessToken: "other-tab-access" })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("normalizes numeric user IDs at the refresh boundary", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
      success: true,
      access_token: "new-access",
      refresh_token: "new-refresh",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))

    const result = await refreshStoredAccessToken("old-access", 1)

    expect(result).toEqual({ accessToken: "new-access" })
  })

  it("does not restore a session cleared while refresh was in flight", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)

    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      clearStoredAuth()
      return new Response(JSON.stringify({
        success: true,
        access_token: "late-access",
        refresh_token: "late-refresh",
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    })

    const result = await refreshStoredAccessToken("old-access")

    expect(result).toEqual({ accessToken: null, rejected: true })
    expect(readAuthCache()).toBeNull()
  })

  it("keeps the session when refresh is temporarily unavailable", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/api/auth/refresh")) {
        return new Response(null, { status: 503 })
      }
      return new Response(null, {
        status: 401,
        headers: { "Error-Type": "TokenExpired" },
      })
    })

    const response = await apiRequest("http://api.local/protected")

    expect(response.status).toBe(401)
    expect(readAuthCache()?.refreshToken).toBe("old-refresh")
  })

  it.each([401, 403])(
    "clears the session when refresh returns %i",
    async (refreshStatus) => {
      writeAuthCache(user, "old-access", "old-refresh", 120, 240)
      vi.spyOn(console, "error").mockImplementation(() => {})

      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        if (String(input).endsWith("/api/auth/refresh")) {
          return new Response(null, { status: refreshStatus })
        }
        return new Response(null, {
          status: 401,
          headers: { "Error-Type": "TokenExpired" },
        })
      })

      const response = await apiRequest("http://api.local/protected")

      expect(response.status).toBe(401)
      expect(readAuthCache()).toBeNull()
    }
  )

  it("does not replay an old request under a replacement user", async () => {
    const replacementUser = {
      id: "2",
      username: "bob",
      email: null,
      is_admin: false,
    }
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)

    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input, options) => {
        if (String(input).endsWith("/api/auth/refresh")) {
          writeAuthCache(
            replacementUser,
            "replacement-access",
            "replacement-refresh",
            120,
            240
          )
          return new Response(JSON.stringify({
            success: true,
            access_token: "late-access",
            refresh_token: "late-refresh",
          }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        }

        const authorization = new Headers(options?.headers).get("Authorization")
        return new Response(null, {
          status: authorization === "Bearer old-access" ? 401 : 200,
          headers: { "Error-Type": "TokenExpired" },
        })
      }
    )

    const response = await apiRequest("http://api.local/protected")

    expect(response.status).toBe(401)
    expect(readAuthCache()?.user?.id).toBe("2")
    expect(readAuthCache()?.token).toBe("replacement-access")
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("releases the refresh lock when the request times out", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    vi.spyOn(console, "error").mockImplementation(() => {})
    vi.useFakeTimers()

    try {
      vi.spyOn(globalThis, "fetch").mockImplementation((_input, options) =>
        new Promise((_resolve, reject) => {
          const signal = options?.signal
          signal?.addEventListener("abort", () => {
            reject(new DOMException("Aborted", "AbortError"))
          })
        })
      )

      const resultPromise = refreshStoredAccessToken("old-access")
      await vi.advanceTimersByTimeAsync(15_000)

      await expect(resultPromise).resolves.toEqual({
        accessToken: null,
        rejected: false,
      })
    } finally {
      vi.useRealTimers()
    }
  })
})

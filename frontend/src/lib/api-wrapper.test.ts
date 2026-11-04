import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  apiRequest,
  classifyUploadError,
  getApiErrorMessage,
  getUploadErrorMessage,
  parseApiResponse,
  refreshStoredAccessToken,
} from "@/lib/api-wrapper"
import {
  AUTH_CACHE_KEY,
  AUTH_TOKEN_UPDATED_EVENT,
  clearStoredAuth,
  readAuthCache,
  readAuthSessionSnapshot,
} from "@/lib/auth-cache"

function writeAuthCache(
  user: { id: string; username: string; email?: string | null; is_admin?: boolean } | null,
  token: string | null,
  refreshToken: string | null = null,
  expiresIn?: number,
  refreshExpiresIn?: number,
) {
  if (!user || !token) {
    localStorage.removeItem(AUTH_CACHE_KEY)
    return
  }
  const now = Date.now()
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
    schemaVersion: 2, sessionId: `test-${Math.random()}`, credentialRevision: 0, profileRevision: 0,
    user, token, refreshToken, timestamp: now,
    expiresAt: expiresIn ? now + expiresIn * 1000 : undefined,
    refreshExpiresAt: refreshExpiresIn ? now + refreshExpiresIn * 1000 : undefined,
  }))
}

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

function mockQueuedNavigatorLocks() {
  const tails = new Map<string, Promise<void>>()
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: {
      request: vi.fn((name: string, callback: () => Promise<unknown>) => {
        const previous = tails.get(name) ?? Promise.resolve()
        const next = previous.then(callback)
        tails.set(name, next.then(() => undefined, () => undefined))
        return next
      }),
    },
  })
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
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

  it("renders FastAPI validation detail arrays as readable messages", () => {
    const response = new Response(null, { status: 422 })
    const message = getUploadErrorMessage(response, {
      data: {
        detail: [
          { type: "too_long", loc: ["body", "files"], msg: "List should have at most 5 items", input: [] },
          { type: "string_pattern_mismatch", loc: ["body", "files", 0, "fileId"], msg: "String should match pattern" },
        ],
      },
      text: null,
      isHtml: false,
    }, MESSAGES)

    expect(message).toBe("List should have at most 5 items; String should match pattern")
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

  it("classifies public upload failures without exposing raw bodies", () => {
    const response = new Response(null, { status: 500 })
    const classified = classifyUploadError(response, {
      data: { detail: "storage path /srv/private/uploads" },
      text: "upstream token=secret",
      isHtml: false,
    })

    expect(classified).toEqual({
      errorCode: "upload_failed",
      message: "Upload failed. Please try again.",
    })
    expect(JSON.stringify(classified)).not.toContain("/srv/private")
    expect(JSON.stringify(classified)).not.toContain("token=secret")
  })

  it.each([
    [413, false, "upload_too_large"],
    [502, true, "upload_proxy_error"],
  ] as const)("classifies status %s safely", (status, isHtml, errorCode) => {
    const classified = classifyUploadError(new Response(null, { status }), {
      data: null,
      text: "raw proxy response",
      isHtml,
    })

    expect(classified.errorCode).toBe(errorCode)
    expect(classified.message).not.toContain("raw proxy response")
  })

  it("honors a recognized backend error code but rejects arbitrary codes", () => {
    const response = new Response(null, { status: 400 })
    expect(classifyUploadError(response, {
      data: { error_code: "upload_too_large", detail: "private detail" },
      text: null,
      isHtml: false,
    }).errorCode).toBe("upload_too_large")
    expect(classifyUploadError(response, {
      data: { error_code: "private_backend_fault" },
      text: null,
      isHtml: false,
    }).errorCode).toBe("upload_failed")
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

  it("replays once when the profile advances after the refreshed credential is committed", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    const updateProfileAfterRefresh = () => {
      const raw = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "{}")
      if (raw.token !== "new-access") return
      raw.profileRevision += 1
      raw.user.email = "fresh-profile@example.com"
      localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(raw))
    }
    window.addEventListener(AUTH_TOKEN_UPDATED_EVENT, updateProfileAfterRefresh)
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
      if (String(input).endsWith("/api/auth/refresh")) {
        return new Response(JSON.stringify({ success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: 120, refresh_expires_in: 240 }), {
          headers: { "Content-Type": "application/json" },
        })
      }
      return new Response(null, {
        status: new Headers(options?.headers).get("Authorization") === "Bearer old-access" ? 401 : 200,
        headers: { "Error-Type": "TokenExpired" },
      })
    })

    await expect(apiRequest("http://api.local/protected")).resolves.toMatchObject({ status: 200 })
    expect(fetchMock.mock.calls.map(([, options]) => new Headers(options?.headers).get("Authorization"))).toContain("Bearer new-access")
    window.removeEventListener(AUTH_TOKEN_UPDATED_EVENT, updateProfileAfterRefresh)
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

  it("uses one refresh request across isolated module realms that share browser storage and locks", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    mockQueuedNavigatorLocks()
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
      success: true,
      access_token: "new-access",
      refresh_token: "new-refresh",
      expires_in: 120,
      refresh_expires_in: 240,
    }), { status: 200, headers: { "Content-Type": "application/json" } }))

    vi.resetModules()
    const firstCache = await import("@/lib/auth-cache")
    const firstApi = await import("@/lib/api-wrapper")
    const snapshot = firstCache.readAuthSessionSnapshot()
    vi.resetModules()
    const secondApi = await import("@/lib/api-wrapper")

    const [first, second] = await Promise.all([
      firstApi.refreshStoredAccessToken(snapshot),
      secondApi.refreshStoredAccessToken(snapshot),
    ])

    expect(first).toMatchObject({ status: "refreshed", accessToken: "new-access" })
    expect(second).toMatchObject({ status: "advanced", accessToken: "new-access" })
    expect(vi.mocked(globalThis.fetch).mock.calls.filter(([input]) => String(input).endsWith("/api/auth/refresh"))).toHaveLength(1)
    expect(readAuthCache()).toMatchObject({ token: "new-access", refreshToken: "new-refresh" })
  })

  it("allows logout to acquire the mutation lock while refresh HTTP is pending", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    mockQueuedNavigatorLocks()
    const refreshStarted = deferred<void>()
    const refreshResponse = deferred<Response>()
    vi.spyOn(globalThis, "fetch").mockImplementation(async input => {
      if (!String(input).endsWith("/api/auth/refresh")) throw new Error("expected refresh request")
      refreshStarted.resolve()
      return refreshResponse.promise
    })
    const refresh = refreshStoredAccessToken(readAuthSessionSnapshot())
    await refreshStarted.promise

    const logout = clearStoredAuth()

    await expect(logout).resolves.toMatchObject({ status: "cleared", credentialsCleared: true })
    expect(readAuthCache()).toBeNull()
    refreshResponse.resolve(new Response(JSON.stringify({ success: true, access_token: "late-access", refresh_token: "late-refresh" }), {
      status: 200, headers: { "Content-Type": "application/json" },
    }))
    await expect(refresh).resolves.toEqual({ status: "invalid_response", accessToken: null })
    expect(readAuthCache()).toBeNull()
  })

  it("does not issue a conditional refresh without Web Locks while explicit logout remains available", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    Object.defineProperty(navigator, "locks", { configurable: true, value: undefined })
    const fetchMock = vi.spyOn(globalThis, "fetch")

    await expect(refreshStoredAccessToken(readAuthSessionSnapshot())).resolves.toEqual({ status: "unavailable", accessToken: null, reason: "coordination_unavailable" })
    await expect(clearStoredAuth()).resolves.toMatchObject({ status: "cleared", credentialsCleared: true })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("returns rejected without making a refresh request when the captured session has no refresh credential", async () => {
    writeAuthCache(user, "old-access")
    const fetchMock = vi.spyOn(globalThis, "fetch")

    await expect(refreshStoredAccessToken(readAuthSessionSnapshot())).resolves.toEqual({ status: "rejected", accessToken: null })

    expect(fetchMock).not.toHaveBeenCalled()
  })

  it.each([
    {},
    { success: false, access_token: "new-access", refresh_token: "new-refresh", expires_in: 60, refresh_expires_in: 120 },
    { success: true, access_token: " ", refresh_token: "new-refresh", expires_in: 60, refresh_expires_in: 120 },
    { success: true, access_token: "new-access", refresh_token: " ", expires_in: 60, refresh_expires_in: 120 },
    { success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: 1.5, refresh_expires_in: 120 },
    { success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: true, refresh_expires_in: 120 },
    { success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: "60", refresh_expires_in: 120 },
    { success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: 0, refresh_expires_in: 120 },
    { success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: 60, refresh_expires_in: -1 },
    { success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: Number.MAX_SAFE_INTEGER, refresh_expires_in: 120 },
  ])("rejects an incomplete or malformed refresh response without advancing cache lineage", async payload => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    const before = localStorage.getItem(AUTH_CACHE_KEY)
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(payload), {
      status: 200, headers: { "Content-Type": "application/json" },
    }))

    await expect(refreshStoredAccessToken(readAuthSessionSnapshot())).resolves.toEqual({ status: "invalid_response", accessToken: null })

    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(before)
  })

  it("classifies an expiry that becomes unsafe before commit as an invalid response", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    const before = localStorage.getItem(AUTH_CACHE_KEY)
    const now = 1_000
    const boundaryExpiry = Math.floor((Number.MAX_SAFE_INTEGER - now) / 1_000)
    vi.spyOn(Date, "now")
      .mockReturnValueOnce(now)
      .mockReturnValueOnce(now)
      .mockReturnValueOnce(now)
      .mockReturnValueOnce(now)
      .mockReturnValueOnce(now)
      .mockReturnValue(now + 1_000)
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
      success: true, access_token: "new-access", refresh_token: "new-refresh",
      expires_in: boundaryExpiry, refresh_expires_in: 120,
    }), { status: 200, headers: { "Content-Type": "application/json" } }))

    await expect(refreshStoredAccessToken(readAuthSessionSnapshot())).resolves.toEqual({ status: "invalid_response", accessToken: null })

    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(before)
  })
  it.each(["success", "access_token", "refresh_token", "expires_in", "refresh_expires_in"] as const)("rejects a refresh response missing required field %s", async field => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    const payload: Record<string, unknown> = {
      success: true, access_token: "new-access", refresh_token: "new-refresh", expires_in: 60, refresh_expires_in: 120,
    }
    delete payload[field]
    const before = localStorage.getItem(AUTH_CACHE_KEY)
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(payload), {
      status: 200, headers: { "Content-Type": "application/json" },
    }))

    await expect(refreshStoredAccessToken(readAuthSessionSnapshot())).resolves.toEqual({ status: "invalid_response", accessToken: null })

    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(before)
  })

  it("persists opaque rotated refresh credentials exactly after a complete refresh response", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-01-01T00:00:00.000Z"))
    try {
      writeAuthCache(user, "old-access", "old-refresh", 120, 240)
      const before = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "{}")
      vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
        success: true, access_token: " new-access ", refresh_token: " new-refresh ", expires_in: 60, refresh_expires_in: 120,
      }), { status: 200, headers: { "Content-Type": "application/json" } }))

      const result = await refreshStoredAccessToken(readAuthSessionSnapshot())

      const expectedSnapshot = {
        sessionId: before.sessionId, credentialRevision: 1, profileRevision: 0,
        userId: "1", accessToken: " new-access ", refreshToken: " new-refresh ",
        profileFingerprint: JSON.stringify(["alice", null, false]),
      }
      const expectedCache = {
        ...before, credentialRevision: 1, token: " new-access ", refreshToken: " new-refresh ",
        timestamp: Date.now(), expiresAt: Date.now() + 60_000, refreshExpiresAt: Date.now() + 120_000,
      }
      expect(result).toEqual({ status: "refreshed", accessToken: " new-access ", session: expectedSnapshot })
      expect(localStorage.getItem(AUTH_CACHE_KEY)).toBe(JSON.stringify(expectedCache))
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not reuse a changed session while waiting for the lock", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    const fetchMock = vi.spyOn(globalThis, "fetch")

    mockNavigatorLocks(() => {
      writeAuthCache(user, "other-tab-access", "other-tab-refresh", 120, 240)
    })

    const result = await refreshStoredAccessToken(readAuthSessionSnapshot())

    expect(result).toEqual({ status: "not_current", accessToken: null })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("does not reuse a changed session when the caller started without a token", async () => {
    writeAuthCache(user, null, "old-refresh", 120, 240)
    const fetchMock = vi.spyOn(globalThis, "fetch")

    mockNavigatorLocks(() => {
      writeAuthCache(user, "other-tab-access", "other-tab-refresh", 120, 240)
    })

    const result = await refreshStoredAccessToken(readAuthSessionSnapshot())

    expect(result).toEqual({ status: "not_current", accessToken: null })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("normalizes numeric user IDs at the refresh boundary", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
      success: true,
      access_token: "new-access",
      refresh_token: "new-refresh",
      expires_in: 120,
      refresh_expires_in: 240,
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))

    const result = await refreshStoredAccessToken(readAuthSessionSnapshot())

    expect(result).toMatchObject({
      status: "refreshed",
      accessToken: "new-access",
      session: {
        accessToken: "new-access",
        refreshToken: "new-refresh",
        userId: "1",
      },
    })
  })

  it("does not restore a session cleared while refresh was in flight", async () => {
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)

    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      clearStoredAuth()
      return new Response(JSON.stringify({
        success: true,
        access_token: "late-access",
        refresh_token: "late-refresh",
        expires_in: 120,
        refresh_expires_in: 240,
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    })

    const result = await refreshStoredAccessToken(readAuthSessionSnapshot())

    expect(result).toEqual({ status: "not_current", accessToken: null })
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

  it.each([401, 403])(
    "preserves a replacement session when an old refresh returns %i",
    async (refreshStatus) => {
      const replacementUser = {
        id: "2",
        username: "bob",
        email: null,
        is_admin: false,
      }
      writeAuthCache(user, "old-access", "old-refresh", 120, 240)
      vi.spyOn(console, "error").mockImplementation(() => {})
      const refreshStarted = deferred<void>()
      const refreshResponse = deferred<Response>()
      const authorizationHeaders: Array<string | null> = []
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
        if (String(input).endsWith("/api/auth/refresh")) {
          refreshStarted.resolve()
          return refreshResponse.promise
        }
        authorizationHeaders.push(
          new Headers(options?.headers).get("Authorization"),
        )
        return new Response(null, {
          status: 401,
          headers: { "Error-Type": "TokenExpired" },
        })
      })

      const request = apiRequest("http://api.local/protected")
      await refreshStarted.promise
      writeAuthCache(
        replacementUser,
        "replacement-access",
        "replacement-refresh",
        120,
        240,
      )
      refreshResponse.resolve(new Response(null, { status: refreshStatus }))
      const response = await request

      expect(response.status).toBe(401)
      expect(readAuthCache()).toMatchObject({
        token: "replacement-access",
        refreshToken: "replacement-refresh",
        user: { id: "2" },
      })
      expect(authorizationHeaders).toEqual(["Bearer old-access"])
    },
  )

  it("preserves a replacement session when an old refresh returns malformed success", async () => {
    const replacementUser = {
      id: "2",
      username: "bob",
      email: null,
      is_admin: false,
    }
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    vi.spyOn(console, "error").mockImplementation(() => {})
    const refreshStarted = deferred<void>()
    const refreshResponse = deferred<Response>()
    const authorizationHeaders: Array<string | null> = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
      if (String(input).endsWith("/api/auth/refresh")) {
        refreshStarted.resolve()
        return refreshResponse.promise
      }
      authorizationHeaders.push(new Headers(options?.headers).get("Authorization"))
      return new Response(null, {
        status: 401,
        headers: { "Error-Type": "TokenExpired" },
      })
    })

    const request = apiRequest("http://api.local/protected")
    await refreshStarted.promise
    writeAuthCache(
      replacementUser,
      "replacement-access",
      "replacement-refresh",
      120,
      240,
    )
    refreshResponse.resolve(new Response(JSON.stringify({ success: false }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))
    const response = await request

    expect(response.status).toBe(401)
    expect(readAuthCache()).toMatchObject({
      token: "replacement-access",
      refreshToken: "replacement-refresh",
      user: { id: "2" },
    })
    expect(authorizationHeaders).toEqual(["Bearer old-access"])
  })

  it("preserves a same-user replacement while an old response body is pending", async () => {
    const replacementSessionUser = {
      ...user,
      username: "alice-relogin",
    }
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    vi.spyOn(console, "error").mockImplementation(() => {})
    const responseBodyStarted = deferred<void>()
    const responseBody = deferred<{ success: boolean }>()
    const authorizationHeaders: Array<string | null> = []
    const pendingBodyResponse = {
      ok: true,
      status: 200,
      json: () => {
        responseBodyStarted.resolve()
        return responseBody.promise
      },
    } as Response
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
      if (String(input).endsWith("/api/auth/refresh")) {
        return pendingBodyResponse
      }
      authorizationHeaders.push(new Headers(options?.headers).get("Authorization"))
      return new Response(null, {
        status: 401,
        headers: { "Error-Type": "TokenExpired" },
      })
    })

    const request = apiRequest("http://api.local/protected")
    await responseBodyStarted.promise
    writeAuthCache(
      replacementSessionUser,
      "replacement-access",
      "replacement-refresh",
      120,
      240,
    )
    responseBody.resolve({ success: false })
    const response = await request

    expect(response.status).toBe(401)
    expect(readAuthCache()).toMatchObject({
      token: "replacement-access",
      refreshToken: "replacement-refresh",
      user: { id: "1" },
    })
    expect(authorizationHeaders).toEqual(["Bearer old-access"])
  })

  it("does not retry under a replacement with the same access token", async () => {
    const replacementUser = {
      id: "2",
      username: "bob",
      email: null,
      is_admin: false,
    }
    writeAuthCache(user, "old-access", "old-refresh", 120, 240)
    const authorizationHeaders: Array<string | null> = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, options) => {
      if (String(input).endsWith("/api/auth/refresh")) {
        return new Response(JSON.stringify({
          success: true,
          access_token: "shared-access",
          refresh_token: "refreshed-refresh",
          expires_in: 120,
          refresh_expires_in: 240,
        }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })
      }
      authorizationHeaders.push(new Headers(options?.headers).get("Authorization"))
      return new Response(null, {
        status: 401,
        headers: { "Error-Type": "TokenExpired" },
      })
    })
    const replaceAfterRefresh = () => {
      window.removeEventListener(AUTH_TOKEN_UPDATED_EVENT, replaceAfterRefresh)
      writeAuthCache(
        replacementUser,
        "shared-access",
        "replacement-refresh",
        120,
        240,
      )
    }
    window.addEventListener(AUTH_TOKEN_UPDATED_EVENT, replaceAfterRefresh)

    try {
      const response = await apiRequest("http://api.local/protected")

      expect(response.status).toBe(401)
      expect(authorizationHeaders).toEqual(["Bearer old-access"])
      expect(readAuthCache()).toMatchObject({
        token: "shared-access",
        refreshToken: "replacement-refresh",
        user: { id: "2" },
      })
    } finally {
      window.removeEventListener(AUTH_TOKEN_UPDATED_EVENT, replaceAfterRefresh)
    }
  })

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
            expires_in: 120,
            refresh_expires_in: 240,
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

      const resultPromise = refreshStoredAccessToken(readAuthSessionSnapshot())
      await vi.advanceTimersByTimeAsync(15_000)

      await expect(resultPromise).resolves.toEqual({ status: "transport_failed", accessToken: null })
    } finally {
      vi.useRealTimers()
    }
  })
})

import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  AUTH_CACHE_KEY,
  clearAuthTokenPayload,
  clearStoredAuth,
  LEGACY_AUTH_TOKEN_KEY,
  LEGACY_AUTH_USER_KEY,
  readAuthCache,
  storeAuthTokenPayload,
  syncLegacyAuthStorage,
  writeAuthCache,
} from "@/lib/auth-cache"

const user = { id: "1", username: "alice", email: null, is_admin: false }

describe("auth cache helpers", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it("stores the same token payload shape used by password login", () => {
    const dispatch = vi.spyOn(window, "dispatchEvent")

    storeAuthTokenPayload({
      user: { id: 42, username: "person@example.com", is_admin: false },
      access_token: "access-token",
      refresh_token: "refresh-token",
      expires_in: 120,
      refresh_expires_in: 240,
    })

    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBe("access-token")
    expect(JSON.parse(localStorage.getItem(LEGACY_AUTH_USER_KEY) || "{}")).toEqual({
      id: "42",
      username: "person@example.com",
      is_admin: false,
    })

    const cache = JSON.parse(localStorage.getItem(AUTH_CACHE_KEY) || "{}")
    expect(cache.user.username).toBe("person@example.com")
    expect(cache.token).toBe("access-token")
    expect(cache.refreshToken).toBe("refresh-token")
    expect(dispatch).toHaveBeenCalled()
  })

  it("clears legacy and current auth cache keys", () => {
    localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, "access-token")
    localStorage.setItem(LEGACY_AUTH_USER_KEY, "{}")
    localStorage.setItem(AUTH_CACHE_KEY, "{}")

    clearAuthTokenPayload()

    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBeNull()
    expect(localStorage.getItem(LEGACY_AUTH_USER_KEY)).toBeNull()
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
  })

  it("keeps an old cache when the refresh token is still valid", () => {
    const now = Date.now()
    localStorage.setItem(
      AUTH_CACHE_KEY,
      JSON.stringify({
        user,
        token: "expired-access",
        refreshToken: "valid-refresh",
        timestamp: now - 130 * 60 * 1000,
        expiresAt: now - 10 * 1000,
        refreshExpiresAt: now + 6 * 24 * 60 * 60 * 1000,
      })
    )

    const cache = readAuthCache(now)

    expect(cache?.token).toBe("expired-access")
    expect(cache?.refreshToken).toBe("valid-refresh")
  })

  it("removes the cache after the refresh token expires", () => {
    const now = Date.now()
    localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, "expired-access")
    localStorage.setItem(LEGACY_AUTH_USER_KEY, JSON.stringify(user))
    localStorage.setItem(
      AUTH_CACHE_KEY,
      JSON.stringify({
        user,
        token: "expired-access",
        refreshToken: "expired-refresh",
        timestamp: now - 130 * 60 * 1000,
        expiresAt: now - 10 * 1000,
        refreshExpiresAt: now - 10 * 1000,
      })
    )

    expect(readAuthCache(now)).toBeNull()
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBeNull()
    expect(localStorage.getItem(LEGACY_AUTH_USER_KEY)).toBeNull()
  })

  it("expires legacy caches that do not have refresh tokens", () => {
    const now = Date.now()
    localStorage.setItem(
      AUTH_CACHE_KEY,
      JSON.stringify({
        user,
        token: "old-access",
        refreshToken: null,
        timestamp: now - 130 * 60 * 1000,
      })
    )

    expect(readAuthCache(now)).toBeNull()
  })

  it("removes corrupted auth cache while preserving legacy fallback data", () => {
    localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, "legacy-access")
    localStorage.setItem(LEGACY_AUTH_USER_KEY, JSON.stringify(user))
    localStorage.setItem(AUTH_CACHE_KEY, "{not-json")

    expect(readAuthCache()).toBeNull()
    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBe("legacy-access")
    expect(JSON.parse(localStorage.getItem(LEGACY_AUTH_USER_KEY) || "{}")).toEqual(user)
  })

  it("can update only the legacy access token during refresh", () => {
    localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, "old-access")
    localStorage.setItem(LEGACY_AUTH_USER_KEY, JSON.stringify(user))

    syncLegacyAuthStorage(undefined, "new-access")

    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBe("new-access")
    expect(JSON.parse(localStorage.getItem(LEGACY_AUTH_USER_KEY) || "{}")).toEqual(user)
  })

  it("keeps legacy token storage in sync when writing the cache", () => {
    writeAuthCache(user, "access", "refresh", 120, 7 * 24 * 60 * 60)

    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBe("access")
    expect(JSON.parse(localStorage.getItem(LEGACY_AUTH_USER_KEY) || "{}")).toEqual(user)

    clearStoredAuth()

    expect(localStorage.getItem(AUTH_CACHE_KEY)).toBeNull()
    expect(localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBeNull()
    expect(localStorage.getItem(LEGACY_AUTH_USER_KEY)).toBeNull()
  })
})

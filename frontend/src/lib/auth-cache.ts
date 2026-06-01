export const AUTH_CACHE_KEY = "auth_cache"
export const AUTH_TOKEN_UPDATED_EVENT = "auth-token-updated"
export const LEGACY_AUTH_TOKEN_KEY = "auth_token"
export const LEGACY_AUTH_USER_KEY = "auth_user"

export const AUTH_CACHE_DURATION_MS = 120 * 60 * 1000

export interface AuthUser {
  id: string | number
  username: string
  email?: string | null
  is_admin?: boolean
}

export interface AuthCacheUser {
  id: string
  username: string
  email?: string | null
  is_admin?: boolean
}

export interface AuthTokenPayload {
  user: AuthUser
  access_token: string
  refresh_token?: string
  expires_in?: number
  refresh_expires_in?: number
}

export interface AuthCache {
  user: AuthCacheUser | null
  token: string | null
  refreshToken: string | null
  timestamp: number
  expiresAt?: number
  refreshExpiresAt?: number
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function dispatchAuthTokenUpdated() {
  window.dispatchEvent(new StorageEvent(AUTH_TOKEN_UPDATED_EVENT, {
    key: AUTH_CACHE_KEY,
    newValue: localStorage.getItem(AUTH_CACHE_KEY),
  }))
}

export function isAuthCacheUsable(cache: AuthCache, now: number = Date.now()): boolean {
  if (cache.refreshToken && isNumber(cache.refreshExpiresAt)) {
    return cache.refreshExpiresAt > now
  }

  return now - cache.timestamp <= AUTH_CACHE_DURATION_MS
}

export function readAuthCache(now: number = Date.now()): AuthCache | null {
  try {
    const cached = localStorage.getItem(AUTH_CACHE_KEY)
    if (!cached) return null

    const cache = JSON.parse(cached) as AuthCache
    if (!isAuthCacheUsable(cache, now)) {
      clearStoredAuth()
      return null
    }

    return cache
  } catch {
    localStorage.removeItem(AUTH_CACHE_KEY)
    return null
  }
}

export function syncLegacyAuthStorage(
  user: AuthCacheUser | null | undefined,
  token: string | null | undefined
) {
  if (token !== undefined) {
    if (token) {
      localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, token)
    } else {
      localStorage.removeItem(LEGACY_AUTH_TOKEN_KEY)
    }
  }

  if (user !== undefined) {
    if (user) {
      localStorage.setItem(LEGACY_AUTH_USER_KEY, JSON.stringify(user))
    } else {
      localStorage.removeItem(LEGACY_AUTH_USER_KEY)
    }
  }
}

export function writeAuthCache(
  user: AuthCacheUser | null,
  token: string | null,
  refreshToken: string | null = null,
  expiresIn?: number,
  refreshExpiresIn?: number
) {
  const now = Date.now()
  const cache: AuthCache = {
    user,
    token,
    refreshToken,
    timestamp: now,
    expiresAt: expiresIn ? now + expiresIn * 1000 : undefined,
    refreshExpiresAt: refreshExpiresIn
      ? now + refreshExpiresIn * 1000
      : undefined,
  }
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(cache))
  syncLegacyAuthStorage(user, token)
}

export function storeAuthTokenPayload(data: AuthTokenPayload) {
  const userData = {
    id: String(data.user.id),
    username: data.user.username,
    email: data.user.email,
    is_admin: data.user.is_admin,
  }

  writeAuthCache(
    userData,
    data.access_token,
    data.refresh_token || null,
    data.expires_in,
    data.refresh_expires_in
  )
  dispatchAuthTokenUpdated()
}

export function clearStoredAuth() {
  localStorage.removeItem(AUTH_CACHE_KEY)
  localStorage.removeItem(LEGACY_AUTH_TOKEN_KEY)
  localStorage.removeItem(LEGACY_AUTH_USER_KEY)
}

export function clearAuthTokenPayload() {
  clearStoredAuth()
}

"lib/api-wrapper"

import { getApiUrl } from "@/lib/utils"
import {
  type AuthSessionSnapshot,
  type AuthMutationUnavailableReason,
  clearAuthSessionIfCurrent,
  compareAuthSession, compareCredentialSession,
  commitAuthSessionRefresh,
  getAuthStorageAvailability,
  isSafeAuthExpirySeconds,
  readAuthSessionSnapshot,
} from "@/lib/auth-cache"

const AUTH_REFRESH_TIMEOUT_MS = 15_000
const AUTH_REFRESH_LOCK_PREFIX = "xagent-auth-refresh:"
export type AuthRefreshResult =
  | { status: "refreshed" | "advanced"; accessToken: string; session: AuthSessionSnapshot }
  | { status: "rejected"; accessToken: null }
  | { status: "not_current"; accessToken: null }
  | { status: "invalid_response"; accessToken: null }
  | { status: "transport_failed"; accessToken: null }
  | { status: "unavailable"; accessToken: null; reason: AuthMutationUnavailableReason }
const refreshPromises = new Map<string, Promise<AuthRefreshResult>>()
const REFRESH_EXCLUDED_AUTH_ENDPOINTS = ["/api/auth/login", "/api/auth/register", "/api/auth/setup-admin", "/api/auth/forgot-password", "/api/auth/reset-password"]

function shouldSkipRefresh(url: string): boolean {
  if (url.includes("/api/auth/refresh")) return true
  try {
    const parsed = new URL(url, window.location.origin)
    return REFRESH_EXCLUDED_AUTH_ENDPOINTS.some(endpoint => parsed.pathname.endsWith(endpoint))
  } catch { return REFRESH_EXCLUDED_AUTH_ENDPOINTS.some(endpoint => url.includes(endpoint)) }
}
async function fetchWithRetry(url: string, options: RequestInit, maxRetries = 2): Promise<Response> {
  let lastError: Error | null = null
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options)
      if (response.status !== 0 && !response.url.includes("net::ERR_")) return response
      lastError = new Error(`Network error on attempt ${attempt + 1}`)
    } catch (error) {
      lastError = error as Error
      if (attempt < maxRetries) await new Promise(resolve => setTimeout(resolve, Math.min(1000, 100 * 2 ** attempt)))
    }
  }
  throw lastError || new Error("All retry attempts failed")
}
function refreshLockName(session: AuthSessionSnapshot): string | null {
  return session.sessionId ? `${AUTH_REFRESH_LOCK_PREFIX}${session.sessionId}` : null
}
type RefreshLockAttempt<T> =
  | { status: "completed"; value: T }
  | { status: "unavailable"; reason: AuthMutationUnavailableReason }
async function withRefreshLock<T>(session: AuthSessionSnapshot, action: () => Promise<T>): Promise<RefreshLockAttempt<T>> {
  const lockName = refreshLockName(session)
  const locks = typeof navigator === "undefined" ? undefined : navigator.locks
  if (!lockName) return { status: "unavailable", reason: "operation_failed" }
  if (!locks) return { status: "unavailable", reason: "coordination_unavailable" }
  try {
    return { status: "completed", value: await locks.request(lockName, action) }
  } catch {
    return { status: "unavailable", reason: "operation_failed" }
  }
}
type StrictRefreshResponse = {
  success: true
  access_token: string
  refresh_token: string
  expires_in: number
  refresh_expires_in: number
}
function parseStrictRefreshResponse(value: unknown): StrictRefreshResponse | null {
  if (typeof value !== "object" || value === null) return null
  const payload = value as Partial<StrictRefreshResponse>
  const nonblank = (token: unknown): token is string => typeof token === "string" && token.trim().length > 0
  const expiry = isSafeAuthExpirySeconds
  return payload.success === true && nonblank(payload.access_token) && nonblank(payload.refresh_token)
    && expiry(payload.expires_in) && expiry(payload.refresh_expires_in)
    ? payload as StrictRefreshResponse
    : null
}
async function performTokenRefresh(session: AuthSessionSnapshot): Promise<AuthRefreshResult> {
  const result = await withRefreshLock(session, async () => {
    const before = compareCredentialSession(session)
    if (before.status === "credentials_advanced") {
      return { status: "advanced", accessToken: before.projection.snapshot.accessToken!, session: before.projection.snapshot } satisfies AuthRefreshResult
    }
    if (before.status !== "exact_credentials") {
      return { status: "not_current", accessToken: null } satisfies AuthRefreshResult
    }
    if (!before.projection.cache.refreshToken) return { status: "rejected", accessToken: null } satisfies AuthRefreshResult

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), AUTH_REFRESH_TIMEOUT_MS)
    try {
      const response = await fetch(`${getApiUrl()}/api/auth/refresh`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: before.projection.cache.refreshToken }), signal: controller.signal,
      })
      if (!response.ok) return response.status === 401 || response.status === 403
        ? { status: "rejected", accessToken: null } satisfies AuthRefreshResult
        : { status: "transport_failed", accessToken: null } satisfies AuthRefreshResult
      const payload = parseStrictRefreshResponse(await response.json())
      if (!payload) return { status: "invalid_response", accessToken: null } satisfies AuthRefreshResult
      const committed = await commitAuthSessionRefresh(session, payload)
      if (committed.status === "updated" || committed.status === "advanced") {
        return {
          status: committed.status === "updated" ? "refreshed" : "advanced",
          accessToken: committed.projection.snapshot.accessToken!,
          session: committed.projection.snapshot,
        } satisfies AuthRefreshResult
      }
      if (committed.status === "unavailable") return { status: "unavailable", accessToken: null, reason: committed.reason } satisfies AuthRefreshResult
      if (committed.status === "invalid") return { status: "invalid_response", accessToken: null } satisfies AuthRefreshResult
      return { status: "not_current", accessToken: null } satisfies AuthRefreshResult
    } catch (error) {
      console.error("Token refresh failed:", error)
      return { status: "transport_failed", accessToken: null } satisfies AuthRefreshResult
    } finally {
      clearTimeout(timeout)
    }
  })
  return result.status === "completed" ? result.value : { status: "unavailable", accessToken: null, reason: result.reason }
}
/** Refreshes only the immutable snapshot captured by the caller. */
export function refreshStoredAccessToken(expectedSession: AuthSessionSnapshot): Promise<AuthRefreshResult> {
  const availability = getAuthStorageAvailability()
  if (availability.status === "unavailable") return Promise.resolve({ status: "unavailable", accessToken: null, reason: availability.reason })
  const comparison = compareAuthSession(expectedSession)
  if (comparison.status === "credentials_advanced" || comparison.status === "credentials_and_profile_advanced") {
    return Promise.resolve({ status: "advanced", accessToken: comparison.projection.snapshot.accessToken!, session: comparison.projection.snapshot })
  }
  if (comparison.status !== "exact" && comparison.status !== "profile_advanced") return Promise.resolve({ status: "not_current", accessToken: null })
  const key = `${expectedSession.sessionId}::${expectedSession.credentialRevision}::${expectedSession.accessToken}`
  const pending = refreshPromises.get(key)
  if (pending) return pending
  const promise = performTokenRefresh(expectedSession).finally(() => refreshPromises.delete(key))
  refreshPromises.set(key, promise)
  return promise
}

function withBearer(options: RequestInit, token: string): RequestInit {
  return { ...options, headers: { ...options.headers, Authorization: `Bearer ${token}` } }
}
/** A request has at most one post-401 replay, bound to an exact immutable credential snapshot. */
export async function apiRequest(url: string, options: RequestInit = {}): Promise<Response> {
  const session = readAuthSessionSnapshot()
  if (!session.accessToken) return fetch(url, options)
  const response = await fetchWithRetry(url, withBearer(options, session.accessToken))
  if (response.status !== 401 || shouldSkipRefresh(url)) return response
  const afterResponse = compareAuthSession(session)
  if (afterResponse.status === "credentials_advanced" || afterResponse.status === "credentials_and_profile_advanced") {
    const advanced = afterResponse.projection.snapshot
    if (compareCredentialSession(advanced).status === "exact_credentials") return fetch(url, withBearer(options, advanced.accessToken!))
    return response
  }
  if (afterResponse.status !== "exact" && afterResponse.status !== "profile_advanced") return response
  const errorType = response.headers.get("Error-Type")
  if (errorType && errorType !== "TokenExpired") {
    if ((await clearAuthSessionIfCurrent(session)).status === "cleared") window.location.href = "/login"
    return response
  }
  const refreshed = await refreshStoredAccessToken(session)
  switch (refreshed.status) {
    case "refreshed":
    case "advanced":
      if (compareCredentialSession(refreshed.session).status === "exact_credentials") {
        return fetch(url, withBearer(options, refreshed.accessToken))
      }
      return response
    case "rejected":
      if ((await clearAuthSessionIfCurrent(session)).status === "cleared") {
        console.error("Refresh token was rejected, redirecting to login")
        window.location.href = "/login"
      }
      return response
    case "not_current":
    case "invalid_response":
    case "transport_failed":
    case "unavailable":
      return response
  }
}

const MAX_RAW_UPLOAD_MESSAGE_LENGTH = 200
function truncateUploadMessage(text: string): string { const trimmed = text.trim(); return trimmed.length <= MAX_RAW_UPLOAD_MESSAGE_LENGTH ? trimmed : `${trimmed.slice(0, MAX_RAW_UPLOAD_MESSAGE_LENGTH)}...` }
type JsonRecord = Record<string, unknown>
export interface ParsedApiResponse { data: JsonRecord | JsonRecord[] | null; text: string | null; isHtml: boolean }
export function isJsonRecord(value: unknown): value is JsonRecord { return typeof value === "object" && value !== null && !Array.isArray(value) }
export async function parseApiResponse(response: Response): Promise<ParsedApiResponse> {
  const contentType = response.headers.get("content-type")?.toLowerCase() || ""
  const text = await response.text().catch(() => "")
  if (!text) return { data: null, text: null, isHtml: contentType.includes("text/html") }
  try { return { data: JSON.parse(text), text, isHtml: /^\s*</.test(text) } }
  catch { return { data: null, text, isHtml: contentType.includes("text/html") || /^\s*</.test(text) } }
}
export const UPLOAD_ERROR_MESSAGES = { tooLarge: "File is too large. Please reduce the upload size and try again.", proxy: "Upload failed before reaching the application. Please check the server upload limit." }
export function getUploadErrorMessage(response: Response, parsed: ParsedApiResponse, messages: { generic: string; tooLarge: string; proxy: string }): string {
  if (isJsonRecord(parsed.data) && typeof parsed.data.detail === "string" && parsed.data.detail.trim()) return parsed.data.detail
  if (isJsonRecord(parsed.data) && Array.isArray(parsed.data.detail)) {
    const validationMessages = parsed.data.detail
      .map(item => isJsonRecord(item) && typeof item.msg === "string" ? item.msg.trim() : "")
      .filter(Boolean)
    if (validationMessages.length) return validationMessages.join("; ")
  }
  if (isJsonRecord(parsed.data) && typeof parsed.data.message === "string" && parsed.data.message.trim()) return parsed.data.message
  if (response.status === 413) return messages.tooLarge
  if (parsed.isHtml) return messages.proxy
  return parsed.text?.trim() ? truncateUploadMessage(parsed.text) : messages.generic
}
export function getApiErrorMessage(response: Response, parsed: ParsedApiResponse, generic: string): string {
  if (isJsonRecord(parsed.data) && typeof parsed.data.detail === "string" && parsed.data.detail.trim()) return parsed.data.detail
  if (isJsonRecord(parsed.data) && typeof parsed.data.message === "string" && parsed.data.message.trim()) return parsed.data.message
  if (parsed.text?.trim() && !parsed.isHtml) return truncateUploadMessage(parsed.text)
  return response.statusText?.trim() || generic
}
export const api = {
  get: (url: string, options?: RequestInit) => apiRequest(url, { ...options, method: "GET" }),
  post: (url: string, data?: unknown, options?: RequestInit) => apiRequest(url, { ...options, method: "POST", headers: { "Content-Type": "application/json", ...options?.headers }, body: data ? JSON.stringify(data) : undefined }),
  put: (url: string, data?: unknown, options?: RequestInit) => apiRequest(url, { ...options, method: "PUT", headers: { "Content-Type": "application/json", ...options?.headers }, body: data ? JSON.stringify(data) : undefined }),
  delete: (url: string, options?: RequestInit) => apiRequest(url, { ...options, method: "DELETE" }),
}
export async function handleAuthError(response: Response): Promise<boolean> {
  if (response.status !== 401) return false
  const cleared = await clearAuthSessionIfCurrent(readAuthSessionSnapshot())
  if (cleared.status === "cleared") window.location.href = "/login"
  return cleared.status === "cleared"
}

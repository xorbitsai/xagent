"lib/api-wrapper"

import { getApiUrl } from "@/lib/utils"
import { AUTH_CACHE_KEY, AUTH_TOKEN_UPDATED_EVENT } from "@/lib/auth-cache"

let isRefreshing = false
let refreshSubscribers: ((token: string) => void)[] = []
const REFRESH_EXCLUDED_AUTH_ENDPOINTS = [
  "/api/auth/login",
  "/api/auth/register",
  "/api/auth/setup-admin",
  "/api/auth/forgot-password",
  "/api/auth/reset-password",
]

function shouldSkipRefresh(url: string): boolean {
  if (url.includes("/api/auth/refresh")) {
    return true
  }

  try {
    const parsedUrl = new URL(url, window.location.origin)
    return REFRESH_EXCLUDED_AUTH_ENDPOINTS.some(endpoint =>
      parsedUrl.pathname.endsWith(endpoint)
    )
  } catch {
    return REFRESH_EXCLUDED_AUTH_ENDPOINTS.some(endpoint => url.includes(endpoint))
  }
}

// Fetch function with retry mechanism
async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries: number = 2
): Promise<Response> {
  let lastError: Error | null = null

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options)

      // If not a network error, return directly
      if (response.status !== 0 && !response.url.includes('net::ERR_')) {
        return response
      }

      // Network error, retry
      lastError = new Error(`Network error on attempt ${attempt + 1}`)

    } catch (error) {
      lastError = error as Error
      console.warn(`Network request failed (attempt ${attempt + 1}/${maxRetries + 1}):`, error)

      // Last attempt, no wait
      if (attempt < maxRetries) {
        // Exponential backoff, max wait 1 second
        await new Promise(resolve => setTimeout(resolve, Math.min(1000, 100 * Math.pow(2, attempt))))
      }
    }
  }

  // All retries failed, throw last error
  throw lastError || new Error('All retry attempts failed')
}

// Add refresh subscriber
function addRefreshSubscriber(callback: (token: string) => void) {
  refreshSubscribers.push(callback)
}

// Notify all subscribers that refresh is complete
function notifyRefreshSubscribers(token: string) {
  refreshSubscribers.forEach(callback => callback(token))
  refreshSubscribers = []
}

// Get current tokens
function getCurrentTokens(): { accessToken: string | null; refreshToken: string | null } {
  // Try new cache format first
  const cache = localStorage.getItem(AUTH_CACHE_KEY)
  if (cache) {
    try {
      const authCache = JSON.parse(cache)
      return {
        accessToken: authCache.token || null,
        refreshToken: authCache.refreshToken || null,
      }
    } catch {
      return {
        accessToken: localStorage.getItem("auth_token"),
        refreshToken: null,
      }
    }
  }

  // Fall back to old format
  return {
    accessToken: localStorage.getItem("auth_token"),
    refreshToken: null,
  }
}

// Refresh token
async function refreshToken(): Promise<string | null> {
  const { refreshToken: refresh } = getCurrentTokens()
  if (!refresh) return null

  try {
    const response = await fetch(`${getApiUrl()}/api/auth/refresh`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ refresh_token: refresh }),
    })

    if (response.ok) {
      const data = await response.json()
      if (data.success && data.access_token) {
        // Update tokens in cache
        const cache = localStorage.getItem(AUTH_CACHE_KEY)
        if (cache) {
          try {
            const authCache = JSON.parse(cache)
            authCache.token = data.access_token
            if (data.expires_in) {
              authCache.expiresAt = Date.now() + data.expires_in * 1000
            }
            if (data.refresh_token) {
              authCache.refreshToken = data.refresh_token
            }
            if (data.refresh_expires_in) {
              authCache.refreshExpiresAt = Date.now() + data.refresh_expires_in * 1000
            }
            authCache.timestamp = Date.now()  // Update timestamp
            localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(authCache))
          } catch {
            // If parsing fails, use old format
            localStorage.setItem("auth_token", data.access_token)
          }
        } else {
          // Use old format
          localStorage.setItem("auth_token", data.access_token)
        }

        // Trigger a storage event to notify AuthContext to update state
        window.dispatchEvent(new StorageEvent(AUTH_TOKEN_UPDATED_EVENT, {
          key: AUTH_CACHE_KEY,
          newValue: localStorage.getItem(AUTH_CACHE_KEY)
        }))

        return data.access_token
      }
    }
  } catch (error) {
    console.error("Token refresh failed:", error)
  }

  return null
}

// API request wrapper
export async function apiRequest(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const { accessToken } = getCurrentTokens()

  // If no token, request directly
  if (!accessToken) {
    return fetch(url, options)
  }

  // Add authorization header
  const headers = {
    ...options.headers,
    "Authorization": `Bearer ${accessToken}`,
  }

  // Fetch request with retry mechanism
  let response = await fetchWithRetry(url, { ...options, headers })

  // If 401 error and not a refresh request, try to refresh token
  if (response.status === 401 && !shouldSkipRefresh(url)) {
    // Check if token is expired or invalid
    const errorType = response.headers.get("Error-Type")
    const isExpired = errorType === "TokenExpired" || !errorType // Default to expired, try to refresh

    if (!isExpired) {
      // Explicitly invalid token, redirect to login page directly
      localStorage.removeItem("auth_token")
      localStorage.removeItem("auth_user")
      localStorage.removeItem(AUTH_CACHE_KEY)
      window.location.href = "/login"
      return response
    }
    if (isRefreshing) {
      // If refreshing, wait for refresh to complete
      return new Promise((resolve, reject) => {
        addRefreshSubscriber((newToken: string) => {
          const retryHeaders = {
            ...options.headers,
            "Authorization": `Bearer ${newToken}`,
          }
          fetch(url, { ...options, headers: retryHeaders })
            .then(resolve)
            .catch(reject)
        })
      })
    }

    isRefreshing = true

    try {
      const newToken = await refreshToken()

      if (newToken) {
        // Notify all waiting subscribers
        notifyRefreshSubscribers(newToken)

        // Retry request with new token
        const retryHeaders = {
          ...options.headers,
          "Authorization": `Bearer ${newToken}`,
        }
        response = await fetch(url, { ...options, headers: retryHeaders })
      } else {
        // Refresh failed, clear auth data and redirect to login page
        console.error("Token refresh failed, redirecting to login")
        localStorage.removeItem("auth_token")
        localStorage.removeItem("auth_user")
        localStorage.removeItem(AUTH_CACHE_KEY)
        window.location.href = "/login"
      }
    } finally {
      isRefreshing = false
    }
  }

  return response
}

const MAX_RAW_UPLOAD_MESSAGE_LENGTH = 200

function truncateUploadMessage(text: string): string {
  const trimmed = text.trim()
  if (trimmed.length <= MAX_RAW_UPLOAD_MESSAGE_LENGTH) {
    return trimmed
  }
  return `${trimmed.slice(0, MAX_RAW_UPLOAD_MESSAGE_LENGTH)}...`
}

type JsonRecord = Record<string, unknown>

export interface ParsedApiResponse {
  data: JsonRecord | JsonRecord[] | null
  text: string | null
  isHtml: boolean
}

export function isJsonRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export async function parseApiResponse(response: Response): Promise<ParsedApiResponse> {
  const contentType = response.headers.get("content-type")?.toLowerCase() || ""
  const text = await response.text().catch(() => "")

  if (!text) {
    return {
      data: null,
      text: null,
      isHtml: contentType.includes("text/html"),
    }
  }

  try {
    return {
      data: JSON.parse(text),
      text,
      isHtml: /^\s*</.test(text),
    }
  } catch {
    return {
      data: null,
      text,
      isHtml: contentType.includes("text/html") || /^\s*</.test(text),
    }
  }
}

export const UPLOAD_ERROR_MESSAGES = {
  tooLarge: "File is too large. Please reduce the upload size and try again.",
  proxy: "Upload failed before reaching the application. Please check the server upload limit.",
}

export function getUploadErrorMessage(
  response: Response,
  parsed: ParsedApiResponse,
  messages: {
    generic: string
    tooLarge: string
    proxy: string
  }
): string {
  if (isJsonRecord(parsed.data) && typeof parsed.data.detail === "string" && parsed.data.detail.trim()) {
    return parsed.data.detail
  }

  if (isJsonRecord(parsed.data) && typeof parsed.data.message === "string" && parsed.data.message.trim()) {
    return parsed.data.message
  }

  if (response.status === 413) {
    return messages.tooLarge
  }

  if (parsed.isHtml) {
    return messages.proxy
  }

  if (parsed.text?.trim()) {
    return truncateUploadMessage(parsed.text)
  }

  return messages.generic
}

export function getApiErrorMessage(
  response: Response,
  parsed: ParsedApiResponse,
  generic: string
): string {
  if (isJsonRecord(parsed.data) && typeof parsed.data.detail === "string" && parsed.data.detail.trim()) {
    return parsed.data.detail
  }

  if (isJsonRecord(parsed.data) && typeof parsed.data.message === "string" && parsed.data.message.trim()) {
    return parsed.data.message
  }

  if (parsed.text?.trim() && !parsed.isHtml) {
    return truncateUploadMessage(parsed.text)
  }

  if (response.statusText?.trim()) {
    return response.statusText
  }

  return generic
}

// Convenience methods
export const api = {
  get: (url: string, options?: RequestInit) =>
    apiRequest(url, { ...options, method: "GET" }),

  post: (url: string, data?: unknown, options?: RequestInit) =>
    apiRequest(url, {
      ...options,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      body: data ? JSON.stringify(data) : undefined,
    }),

  put: (url: string, data?: unknown, options?: RequestInit) =>
    apiRequest(url, {
      ...options,
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      body: data ? JSON.stringify(data) : undefined,
    }),

  delete: (url: string, options?: RequestInit) =>
    apiRequest(url, { ...options, method: "DELETE" }),
}

// Check response status, if auth error redirect to login
export function handleAuthError(response: Response) {
  if (response.status === 401) {
    // Clear auth data and redirect to login page
    localStorage.removeItem("auth_token")
    localStorage.removeItem("auth_user")
    localStorage.removeItem(AUTH_CACHE_KEY)
    window.location.href = "/login"
    return true
  }
  return false
}

"lib/api-wrapper"

import { getApiUrl } from "@/lib/utils"
import { AUTH_CACHE_KEY, AUTH_TOKEN_UPDATED_EVENT } from "@/lib/auth-cache"

let isRefreshing = false
let refreshSubscribers: ((token: string) => void)[] = []
const REFRESH_EXCLUDED_AUTH_ENDPOINTS = [
  "/api/auth/login",
  "/api/auth/register",
  "/api/auth/setup-admin",
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

// 带重试机制的fetch函数
async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries: number = 2
): Promise<Response> {
  let lastError: Error | null = null

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options)

      // 如果不是网络错误，直接返回
      if (response.status !== 0 && !response.url.includes('net::ERR_')) {
        return response
      }

      // 网络错误，重试
      lastError = new Error(`Network error on attempt ${attempt + 1}`)

    } catch (error) {
      lastError = error as Error
      console.warn(`Network request failed (attempt ${attempt + 1}/${maxRetries + 1}):`, error)

      // 最后一次尝试，不再等待
      if (attempt < maxRetries) {
        // 指数退避，最多等待1秒
        await new Promise(resolve => setTimeout(resolve, Math.min(1000, 100 * Math.pow(2, attempt))))
      }
    }
  }

  // 所有重试都失败了，抛出最后一个错误
  throw lastError || new Error('All retry attempts failed')
}

// 添加等待刷新的订阅者
function addRefreshSubscriber(callback: (token: string) => void) {
  refreshSubscribers.push(callback)
}

// 通知所有订阅者刷新完成
function notifyRefreshSubscribers(token: string) {
  refreshSubscribers.forEach(callback => callback(token))
  refreshSubscribers = []
}

// 获取当前tokens
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

// 刷新token
async function refreshToken(): Promise<string | null> {
  const { accessToken, refreshToken: refresh } = getCurrentTokens()
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
        // 更新缓存中的tokens
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
            authCache.timestamp = Date.now()  // 更新时间戳
            localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(authCache))
          } catch {
            // 如果解析失败，使用旧格式
            localStorage.setItem("auth_token", data.access_token)
          }
        } else {
          // 使用旧格式
          localStorage.setItem("auth_token", data.access_token)
        }

        // 触发一个存储事件，通知 AuthContext 更新状态
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

// API请求包装器
export async function apiRequest(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const { accessToken, refreshToken: refresh } = getCurrentTokens()

  // 如果没有token，直接请求
  if (!accessToken) {
    return fetch(url, options)
  }

  // 添加认证头
  const headers = {
    ...options.headers,
    "Authorization": `Bearer ${accessToken}`,
  }

  // 带重试机制的fetch请求
  let response = await fetchWithRetry(url, { ...options, headers })

  // 如果401错误且不是refresh请求，尝试刷新token
  if (response.status === 401 && !shouldSkipRefresh(url)) {
    // 检查是否是token过期还是无效token
    const errorType = response.headers.get("Error-Type")
    const isExpired = errorType === "TokenExpired" || !errorType // 默认认为是过期，尝试刷新

    if (!isExpired) {
      // 明确的无效token，直接重定向到登录页
      localStorage.removeItem("auth_token")
      localStorage.removeItem("auth_user")
      localStorage.removeItem(AUTH_CACHE_KEY)
      window.location.href = "/login"
      return response
    }
    if (isRefreshing) {
      // 如果正在刷新，等待刷新完成
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
        // 通知所有等待的订阅者
        notifyRefreshSubscribers(newToken)

        // 使用新token重试请求
        const retryHeaders = {
          ...options.headers,
          "Authorization": `Bearer ${newToken}`,
        }
        response = await fetch(url, { ...options, headers: retryHeaders })
      } else {
        // 刷新失败，清除认证数据并重定向到登录页
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

// 便捷方法
export const api = {
  get: (url: string, options?: RequestInit) =>
    apiRequest(url, { ...options, method: "GET" }),

  post: (url: string, data?: any, options?: RequestInit) =>
    apiRequest(url, {
      ...options,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      body: data ? JSON.stringify(data) : undefined,
    }),

  put: (url: string, data?: any, options?: RequestInit) =>
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

// 检查响应状态，如果是认证错误则重定向到登录
export function handleAuthError(response: Response) {
  if (response.status === 401) {
    // 清除认证数据并重定向到登录页
    localStorage.removeItem("auth_token")
    localStorage.removeItem("auth_user")
    localStorage.removeItem(AUTH_CACHE_KEY)
    window.location.href = "/login"
    return true
  }
  return false
}

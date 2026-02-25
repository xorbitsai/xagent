"use client"

import { createContext, useContext, useEffect, useState, ReactNode } from "react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY, AUTH_TOKEN_UPDATED_EVENT } from "@/lib/auth-cache"

interface User {
  id: string
  username: string
  is_admin?: boolean
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  token: string | null
  refreshToken: string | null
  isLoading: boolean
  login: (username: string, password: string) => Promise<boolean>
  logout: () => void
  checkAuth: () => Promise<boolean>
  refreshAccessToken: () => Promise<boolean>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// 缓存配置
const CACHE_DURATION = 120 * 60 * 1000 // 120分钟

interface AuthCache {
  user: User | null
  token: string | null
  refreshToken: string | null
  timestamp: number
  expiresAt?: number  // access token过期时间
  refreshExpiresAt?: number  // refresh token过期时间
}

function getAuthCache(): AuthCache | null {
  try {
    const cached = localStorage.getItem(AUTH_CACHE_KEY)
    if (!cached) return null

    const cache: AuthCache = JSON.parse(cached)
    // 检查缓存是否过期
    if (Date.now() - cache.timestamp > CACHE_DURATION) {
      localStorage.removeItem(AUTH_CACHE_KEY)
      return null
    }

    return cache
  } catch {
    return null
  }
}

function setAuthCache(
  user: User | null,
  token: string | null,
  refreshToken: string | null = null,
  expiresIn?: number,  // access token过期时间（秒）
  refreshExpiresIn?: number  // refresh token过期时间（秒）
) {
  const cache: AuthCache = {
    user,
    token,
    refreshToken,
    timestamp: Date.now(),
    expiresAt: expiresIn ? Date.now() + expiresIn * 1000 : undefined,
    refreshExpiresAt: refreshExpiresIn ? Date.now() + refreshExpiresIn * 1000 : undefined,
  }
  localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify(cache))
}

function clearAuthCache() {
  localStorage.removeItem(AUTH_CACHE_KEY)
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [refreshToken, setRefreshToken] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastCheckTime, setLastCheckTime] = useState(0)

  // 主动刷新token的定时器
  useEffect(() => {
    if (!token || !refreshToken) return

    const refreshInterval = setInterval(async () => {
      const cache = getAuthCache()
      if (!cache) return

      if (cache.expiresAt) {
        const now = Date.now()
        const timeUntilExpiry = cache.expiresAt - now
        const shouldRefresh = timeUntilExpiry < 5 * 60 * 1000 // 提前5分钟刷新

        if (shouldRefresh) {
          console.log("Token即将过期，主动刷新...")
          const success = await refreshAccessToken()
          if (!success) {
            // 刷新失败，停止定时器（因为会自动跳转登录页）
            clearInterval(refreshInterval)
          }
        }
      } else {
        // 没有过期时间信息，使用旧的逻辑
        const timeSinceCreation = Date.now() - cache.timestamp
        const shouldRefresh = timeSinceCreation > (CACHE_DURATION - 5 * 60 * 1000) // 提前5分钟

        if (shouldRefresh) {
          console.log("Token即将过期（基于创建时间），主动刷新...")
          const success = await refreshAccessToken()
          if (!success) {
            clearInterval(refreshInterval)
          }
        }
      }
    }, 60000) // 每分钟检查一次

    return () => clearInterval(refreshInterval)
  }, [token, refreshToken, user])

  useEffect(() => {
    // 初始化时检查缓存
    const timer = setTimeout(() => {
      // Try new cache format first
      const cache = getAuthCache()
      if (cache && cache.user && cache.token) {
        setUser(cache.user)
        setToken(cache.token)
        setRefreshToken(cache.refreshToken)
      } else {
        // Fall back to old format for backward compatibility
        const savedToken = localStorage.getItem("auth_token")
        const savedUser = localStorage.getItem("auth_user")

        if (savedToken && savedUser) {
          try {
            const userData = JSON.parse(savedUser)
            setToken(savedToken)
            setUser(userData)

            // Migrate to new cache format
            setAuthCache(userData, savedToken)
          } catch (error) {
            console.error("Failed to parse saved user data:", error)
            // Clear invalid data
            localStorage.removeItem("auth_token")
            localStorage.removeItem("auth_user")
          }
        }
      }
      setIsLoading(false)
    }, 100)

    return () => clearTimeout(timer)
  }, [])

  // 监听 token 更新事件
  useEffect(() => {
    const handleTokenUpdate = (event: Event) => {
      const storageEvent = event as StorageEvent
      if (storageEvent.key === AUTH_CACHE_KEY && storageEvent.newValue) {
        try {
          const cache = JSON.parse(storageEvent.newValue)
          if (cache.user && cache.token) {
            setUser(cache.user)
            setToken(cache.token)
            setRefreshToken(cache.refreshToken)
          }
        } catch (error) {
          console.error("Failed to parse updated auth cache:", error)
        }
      }
    }

    window.addEventListener(AUTH_TOKEN_UPDATED_EVENT, handleTokenUpdate)
    return () => window.removeEventListener(AUTH_TOKEN_UPDATED_EVENT, handleTokenUpdate)
  }, [])

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, password }),
      })

      if (response.ok) {
        const data = await response.json()
        const userData = {
          id: data.user.id,
          username: data.user.username,
          is_admin: data.user.is_admin
        }

        setToken(data.access_token)
        setRefreshToken(data.refresh_token)
        setUser(userData)

        // 更新缓存
        setAuthCache(
          userData,
          data.access_token,
          data.refresh_token,
          data.expires_in ? data.expires_in : undefined,
          data.refresh_expires_in ? data.refresh_expires_in : undefined
        )

        return true
      }
      return false
    } catch (error) {
      console.error("Login error:", error)
      return false
    }
  }

  const logout = () => {
    setUser(null)
    setToken(null)
    setRefreshToken(null)
    // Clear all auth-related data
    localStorage.removeItem("auth_token")
    localStorage.removeItem("auth_user")
    clearAuthCache()
    window.location.href = "/login"
  }

  const checkAuth = async (): Promise<boolean> => {
    if (!token || !user) return false

    // 防抖：如果距离上次检查时间太短，直接返回true
    const now = Date.now()
    if (now - lastCheckTime < 15000) { // 15秒内不重复检查，减少服务器压力
      return true
    }

    try {
      // 使用新的verify endpoint来检查token有效性
      const response = await apiRequest(`${getApiUrl()}/api/auth/verify`, {
        headers: {
          "X-Username": user.username,
        },
      })

      setLastCheckTime(now)

      if (!response.ok) {
        // apiRequest 已经自动处理了 token 刷新，如果还是失败则说明认证有问题
        if (response.status === 401) {
          // 检查是否是明确的无效token（而不是过期）
          const errorType = response.headers.get("Error-Type")
          const isInvalid = errorType === "InvalidToken"

          if (isInvalid) {
            // 明确的无效token，清除状态
            logout()
            return false
          } else {
            // Token过期但刷新失败，此时 apiRequest 已经处理了重定向
            // 我们只需要清除本地状态
            setUser(null)
            setToken(null)
            setRefreshToken(null)
            clearAuthCache()
            return false
          }
        }
        // 网络错误或服务器错误，保持当前状态
        return false  // 改为false，因为认证检查失败
      }

      const data = await response.json()
      if (data.success === true) {
        // 认证成功，同步更新状态（因为 apiRequest 可能已经更新了缓存）
        const updatedCache = getAuthCache()
        if (updatedCache && updatedCache.token && updatedCache.user) {
          setToken(updatedCache.token)
          setUser(updatedCache.user)
          setRefreshToken(updatedCache.refreshToken)
        }
        return true
      }

      return false
    } catch (error) {
      console.error("Auth check error:", error)
      // 网络错误，保持当前状态
      return true
    }
  }

  const refreshAccessToken = async (): Promise<boolean> => {
    if (!refreshToken) return false

    try {
      const response = await apiRequest(`${getApiUrl()}/api/auth/refresh`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })

      if (response.ok) {
        const data = await response.json()
        if (data.success && data.access_token) {
          setToken(data.access_token)
          if (data.refresh_token) {
            setRefreshToken(data.refresh_token)
          }

          // Update cache with new tokens and expiration times
          setAuthCache(
            user,
            data.access_token,
            data.refresh_token || refreshToken,
            data.expires_in ? data.expires_in : undefined,
            data.refresh_expires_in ? data.refresh_expires_in : undefined
          )
          return true
        }
      }
    } catch (error) {
      console.error("Token refresh failed:", error)
    }

    // If refresh fails, logout and redirect to login
    logout()
    return false
  }

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user && !!token,
    token,
    refreshToken,
    isLoading,
    login,
    logout,
    checkAuth,
    refreshAccessToken,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}

"use client"

import { createContext, useContext, useEffect, useState, ReactNode } from "react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { AUTH_CACHE_KEY, AUTH_TOKEN_UPDATED_EVENT } from "@/lib/auth-cache"

interface User {
  id: string
  username: string
  email?: string | null
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

// Cache configuration
const CACHE_DURATION = 120 * 60 * 1000 // 120 minutes

interface AuthCache {
  user: User | null
  token: string | null
  refreshToken: string | null
  timestamp: number
  expiresAt?: number  // access token expiration time
  refreshExpiresAt?: number  // refresh token expiration time
}

function getAuthCache(): AuthCache | null {
  try {
    const cached = localStorage.getItem(AUTH_CACHE_KEY)
    if (!cached) return null

    const cache: AuthCache = JSON.parse(cached)
    // Check if cache is expired
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
  expiresIn?: number,  // access token expiration time (seconds)
  refreshExpiresIn?: number  // refresh token expiration time (seconds)
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

  // Timer for active token refresh
  useEffect(() => {
    if (!token || !refreshToken) return

    const refreshInterval = setInterval(async () => {
      const cache = getAuthCache()
      if (!cache) return

      if (cache.expiresAt) {
        const now = Date.now()
        const timeUntilExpiry = cache.expiresAt - now
        const shouldRefresh = timeUntilExpiry < 5 * 60 * 1000 // Refresh 5 minutes in advance

        if (shouldRefresh) {
          console.log("Token is about to expire, refreshing actively...")
          const success = await refreshAccessToken()
          if (!success) {
            // Refresh failed, stop timer (will automatically redirect to login page)
            clearInterval(refreshInterval)
          }
        }
      } else {
        // No expiration time info, use old logic
        const timeSinceCreation = Date.now() - cache.timestamp
        const shouldRefresh = timeSinceCreation > (CACHE_DURATION - 5 * 60 * 1000) // 5 minutes in advance

        if (shouldRefresh) {
          console.log("Token is about to expire (based on creation time), refreshing actively...")
          const success = await refreshAccessToken()
          if (!success) {
            clearInterval(refreshInterval)
          }
        }
      }
    }, 60000) // Check every minute

    return () => clearInterval(refreshInterval)
  }, [token, refreshToken, user])

  // Check cache on initialization
  useEffect(() => {
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

  // Listen for token update events
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
          email: data.user.email,
          is_admin: data.user.is_admin
        }

        setToken(data.access_token)
        setRefreshToken(data.refresh_token)
        setUser(userData)

        // Update cache
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

    // Debounce: if interval since last check is too short, return true directly
    const now = Date.now()
    if (now - lastCheckTime < 15000) { // Do not check repeatedly within 15 seconds to reduce server load
      return true
    }

    try {
      // Use new verify endpoint to check token validity
      const response = await apiRequest(`${getApiUrl()}/api/auth/verify`, {
        headers: {
          "X-Username": user.username,
        },
      })

      setLastCheckTime(now)

      if (!response.ok) {
        // apiRequest has automatically handled token refresh, if it still fails, it means there is an authentication problem
        if (response.status === 401) {
          // Check if it is explicitly an invalid token (not expired)
          const errorType = response.headers.get("Error-Type")
          const isInvalid = errorType === "InvalidToken"

          if (isInvalid) {
            // Explicitly invalid token, clear state
            logout()
            return false
          } else {
            // Token expired but refresh failed, apiRequest has handled redirection
            // We just need to clear local state
            setUser(null)
            setToken(null)
            setRefreshToken(null)
            clearAuthCache()
            return false
          }
        }
        // Network error or server error, keep current state
        return false  // Changed to false because auth check failed
      }

      const data = await response.json()
      if (data.success === true) {
        // Auth success, sync update state (because apiRequest may have updated cache)
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
      // Network error, keep current state
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

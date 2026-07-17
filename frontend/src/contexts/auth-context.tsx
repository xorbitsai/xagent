"use client"

import React, { createContext, useContext, useEffect, useRef, useState, ReactNode } from "react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest, refreshStoredAccessToken } from "@/lib/api-wrapper"
import {
  AUTH_CACHE_DURATION_MS,
  AUTH_CACHE_KEY,
  AUTH_TOKEN_UPDATED_EVENT,
  AuthCache,
  AuthCacheUser,
  clearStoredAuth,
  LEGACY_AUTH_TOKEN_KEY,
  LEGACY_AUTH_USER_KEY,
  readAuthCache,
  writeAuthCache,
} from "@/lib/auth-cache"

type User = AuthCacheUser

type TeamRole = "admin" | "member" | null

function isAuthCacheEventPayload(value: unknown): value is AuthCache {
  if (typeof value !== "object" || value === null) return false

  const candidate = value as Partial<AuthCache>
  return (
    typeof candidate.user === "object" &&
    candidate.user !== null &&
    typeof candidate.token === "string"
  )
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  token: string | null
  refreshToken: string | null
  isLoading: boolean
  // SaaS team context; on standard xagent (no teams / my-team 404) inTeam=false, teamRole=null.
  inTeam: boolean
  teamRole: TeamRole
  login: (username: string, password: string) => Promise<boolean>
  logout: () => void
  checkAuth: () => Promise<boolean>
  refreshAccessToken: () => Promise<boolean>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [refreshToken, setRefreshToken] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastCheckTime, setLastCheckTime] = useState(0)
  const [inTeam, setInTeam] = useState(false)
  const [teamRole, setTeamRole] = useState<TeamRole>(null)
  const refreshAccessTokenRef = useRef<(
    expectedAccessToken?: string | null,
    expectedUserId?: string | null
  ) => Promise<boolean>>(
    async () => false
  )

  // Timer for active token refresh
  useEffect(() => {
    if (!token || !refreshToken) return

    const refreshInterval = setInterval(async () => {
      const cache = readAuthCache()
      if (!cache) return

      if (cache.expiresAt) {
        const now = Date.now()
        const timeUntilExpiry = cache.expiresAt - now
        const shouldRefresh = timeUntilExpiry < 5 * 60 * 1000 // Refresh 5 minutes in advance

        if (shouldRefresh) {
          console.log("Token is about to expire, refreshing actively...")
          await refreshAccessTokenRef.current(cache.token, cache.user?.id || null)
        }
      } else {
        const timeSinceCreation = Date.now() - cache.timestamp
        const shouldRefreshAccess = timeSinceCreation > (AUTH_CACHE_DURATION_MS - 5 * 60 * 1000)
        const timeUntilRefreshExpiry = cache.refreshExpiresAt
          ? cache.refreshExpiresAt - Date.now()
          : Number.POSITIVE_INFINITY
        const shouldRefreshToken = timeUntilRefreshExpiry < 5 * 60 * 1000
        const shouldRefresh = shouldRefreshAccess || shouldRefreshToken

        if (shouldRefresh) {
          console.log("Token cache is missing access expiry info, refreshing actively...")
          await refreshAccessTokenRef.current(cache.token, cache.user?.id || null)
        }
      }
    }, 60000) // Check every minute

    return () => clearInterval(refreshInterval)
  }, [token, refreshToken])

  // Check cache on initialization
  useEffect(() => {
    const timer = setTimeout(() => {
      // Try new cache format first
      const cache = readAuthCache()
      if (cache && cache.user && cache.token) {
        setUser(cache.user)
        setToken(cache.token)
        setRefreshToken(cache.refreshToken)
      } else {
        // Fall back to old format for backward compatibility
        const savedToken = localStorage.getItem(LEGACY_AUTH_TOKEN_KEY)
        const savedUser = localStorage.getItem(LEGACY_AUTH_USER_KEY)

        if (savedToken && savedUser) {
          try {
            const userData = JSON.parse(savedUser)
            setToken(savedToken)
            setUser(userData)

            // Migrate to new cache format
            writeAuthCache(userData, savedToken)
          } catch (error) {
            console.error("Failed to parse saved user data:", error)
            clearStoredAuth()
          }
        }
      }
      setIsLoading(false)
    }, 100)

    return () => clearTimeout(timer)
  }, [])

  // Listen for same-tab refresh events and native cross-tab storage events.
  useEffect(() => {
    const handleTokenUpdate = (event: Event) => {
      const storageEvent = event as StorageEvent
      if (storageEvent.key !== AUTH_CACHE_KEY) return

      if (!storageEvent.newValue) {
        setUser(null)
        setToken(null)
        setRefreshToken(null)
        return
      }

      if (storageEvent.newValue) {
        try {
          const cache: unknown = JSON.parse(storageEvent.newValue)
          if (isAuthCacheEventPayload(cache)) {
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
    window.addEventListener("storage", handleTokenUpdate)
    return () => {
      window.removeEventListener(AUTH_TOKEN_UPDATED_EVENT, handleTokenUpdate)
      window.removeEventListener("storage", handleTokenUpdate)
    }
  }, [])

  // Resolve SaaS team role once we have a session. my-team 404 / any failure =>
  // standard xagent with no team concept; keep inTeam=false, teamRole=null.
  useEffect(() => {
    // Reset first so a token change / failed or non-team response never leaves
    // a previous user's team context behind.
    setInTeam(false)
    setTeamRole(null)
    if (!token) {
      return
    }
    let active = true
    ;(async () => {
      try {
        const res = await apiRequest(`${getApiUrl()}/api/teams/my-team`)
        if (!active || !res.ok) return
        const team = await res.json()
        if (!active) return
        setInTeam(true)
        setTeamRole(team?.team_role === "admin" ? "admin" : "member")
      } catch {
        // no team context; keep defaults
      }
    })()
    return () => {
      active = false
    }
  }, [token])

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
        writeAuthCache(
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
    clearStoredAuth()
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
          }

          // A rejected refresh clears the shared cache in apiRequest. A
          // temporary refresh outage leaves it intact so the next check can
          // retry without ejecting the user from the app.
          if (!readAuthCache()) {
            setUser(null)
            setToken(null)
            setRefreshToken(null)
            return false
          }
          return true
        }
        // Network or server errors must not turn a temporary outage into logout.
        return true
      }

      const data = await response.json()
      if (data.success === true) {
        // Auth success, sync update state (because apiRequest may have updated cache)
        const updatedCache = readAuthCache()
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

  const refreshAccessToken = async (
    expectedAccessToken?: string | null,
    expectedUserId?: string | null
  ): Promise<boolean> => {
    const cache = readAuthCache()
    const result = await refreshStoredAccessToken(
      expectedAccessToken === undefined ? cache?.token || null : expectedAccessToken,
      expectedUserId === undefined ? cache?.user?.id || null : expectedUserId
    )
    if (result.accessToken !== null) {
      const updatedCache = readAuthCache()
      if (updatedCache?.user && updatedCache.token) {
        setUser(updatedCache.user)
        setToken(updatedCache.token)
        setRefreshToken(updatedCache.refreshToken)
      }
      return true
    }

    if (result.rejected) {
      logout()
    }
    return false
  }

  refreshAccessTokenRef.current = refreshAccessToken

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user && !!token,
    token,
    refreshToken,
    isLoading,
    inTeam,
    teamRole,
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

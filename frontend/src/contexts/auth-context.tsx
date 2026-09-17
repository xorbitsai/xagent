"use client"

import React, { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react"
import { getApiUrl } from "@/lib/utils"
import { apiRequest, refreshStoredAccessToken } from "@/lib/api-wrapper"
import { toast } from "@/components/ui/sonner"
import { useI18n } from "@/contexts/i18n-context"
import { authMutationUnavailableTranslationKey } from "@/lib/auth-pages"
import {
  AUTH_CACHE_DURATION_MS, AUTH_CACHE_KEY, AUTH_TOKEN_UPDATED_EVENT,
  type AuthCacheUser, type AuthSessionProjection, type AuthSessionSnapshot,
  clearAuthSessionIfCurrent, clearStoredAuth, compareAuthSession,
  claimAuthLoginIntent, createAuthSession, inspectAuthSession, migrateLegacyAuthSession,
} from "@/lib/auth-cache"

type TeamRole = "admin" | "member" | null
interface AuthContextType {
  user: AuthCacheUser | null; isAuthenticated: boolean; token: string | null; refreshToken: string | null
  session: AuthSessionSnapshot; isLoading: boolean; inTeam: boolean; teamRole: TeamRole
  login: (username: string, password: string) => Promise<boolean>
  logout: () => Promise<boolean>
  checkAuth: () => Promise<boolean>
  refreshAccessToken: (expectedSession?: AuthSessionSnapshot) => Promise<boolean>
}
const EMPTY_SESSION: AuthSessionSnapshot = { sessionId: null, credentialRevision: null, profileRevision: null, userId: null, accessToken: null, refreshToken: null, profileFingerprint: null }
const AuthContext = createContext<AuthContextType | undefined>(undefined)
const ANONYMOUS_AUTH_CONTEXT: AuthContextType = {
  user: null, isAuthenticated: false, token: null, refreshToken: null, session: EMPTY_SESSION,
  isLoading: false, inTeam: false, teamRole: null,
  login: async () => false, logout: async () => false, checkAuth: async () => false, refreshAccessToken: async () => false,
}
const AUTH_REFRESH_UNAVAILABLE_TOAST_ID = "auth-refresh-unavailable"
function currentProjection(): AuthSessionProjection | null {
  const inspection = inspectAuthSession()
  return inspection.status === "valid" ? inspection.projection : null
}
function isCurrentCredentialLineage(current: AuthSessionProjection | null, captured: AuthSessionSnapshot): boolean {
  return current?.snapshot.sessionId === captured.sessionId
    && current.snapshot.credentialRevision === captured.credentialRevision
    && current.snapshot.accessToken === captured.accessToken
    && current.snapshot.refreshToken === captured.refreshToken
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const { t } = useI18n()
  const [projection, setProjection] = useState<AuthSessionProjection | null>(() => currentProjection())
  const [isLoading, setIsLoading] = useState(true)
  const [lastCheck, setLastCheck] = useState<{ key: string | null; at: number }>({ key: null, at: 0 })
  const [inTeam, setInTeam] = useState(false)
  const [teamRole, setTeamRole] = useState<TeamRole>(null)
  const mountedRef = useRef(true)
  const initializationRef = useRef(0)
  const operationRef = useRef(0)
  const refreshRef = useRef<(session?: AuthSessionSnapshot) => Promise<boolean>>(async () => false)
  const sync = useCallback(() => { if (mountedRef.current) setProjection(currentProjection()) }, [])
  const session = projection?.snapshot ?? EMPTY_SESSION
  const user = projection?.cache.user ?? null
  const token = session.accessToken
  const refreshToken = session.refreshToken

  useEffect(() => {
    mountedRef.current = true
    const initialization = ++initializationRef.current
    void (async () => {
      await migrateLegacyAuthSession()
      if (mountedRef.current && initialization === initializationRef.current) {
        sync(); setIsLoading(false)
      }
    })()
    return () => { mountedRef.current = false; initializationRef.current += 1 }
  }, [sync])
  useEffect(() => {
    const handler = (event: Event) => {
      if (event.type === AUTH_TOKEN_UPDATED_EVENT || (event as StorageEvent).key === AUTH_CACHE_KEY) sync()
    }
    window.addEventListener(AUTH_TOKEN_UPDATED_EVENT, handler); window.addEventListener("storage", handler)
    return () => { window.removeEventListener(AUTH_TOKEN_UPDATED_EVENT, handler); window.removeEventListener("storage", handler) }
  }, [sync])
  useEffect(() => {
    setInTeam(false); setTeamRole(null)
    if (!token) return
    let active = true
    void (async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/teams/my-team`)
        if (!active || !response.ok) return
        const team = await response.json()
        if (active) { setInTeam(true); setTeamRole(team?.team_role === "admin" ? "admin" : "member") }
      } catch { /* no team context */ }
    })()
    return () => { active = false }
  }, [token])
  useEffect(() => {
    if (!projection || !session.refreshToken) return
    const interval = setInterval(() => {
      const current = currentProjection()
      if (!current || current.snapshot.sessionId !== session.sessionId) return
      const cache = current.cache
      const deadline = Math.min(cache.expiresAt ?? cache.timestamp + AUTH_CACHE_DURATION_MS, cache.refreshExpiresAt ?? Number.POSITIVE_INFINITY)
      if (deadline - Date.now() < 5 * 60 * 1000) void refreshRef.current(current.snapshot)
    }, 60_000)
    return () => clearInterval(interval)
  }, [projection, session.sessionId, session.refreshToken])

  const login = useCallback(async (username: string, password: string) => {
    const operation = ++operationRef.current
    try {
      const claimed = await claimAuthLoginIntent()
      if (claimed.status !== "claimed") return false
      const response = await apiRequest(`${getApiUrl()}/api/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }) })
      if (!response.ok) return false
      const payload = await response.json()
      if (!mountedRef.current || operation !== operationRef.current) return false
      const created = await createAuthSession(payload, claimed.intent)
      if (created.status !== "created") return false
      if (mountedRef.current && operation === operationRef.current) setProjection(created.projection)
      return true
    } catch (error) { console.error("Login error:", error); return false }
  }, [])
  const logout = useCallback(async () => {
    ++operationRef.current
    const result = await clearStoredAuth()
    if (result.credentialsCleared && mountedRef.current) setProjection(null)
    if (result.status !== "cleared") return false
    window.location.href = "/login"
    return true
  }, [])
  const checkAuth = useCallback(async () => {
    const captured = projection?.snapshot
    const beforeInspection = inspectAuthSession()
    const before = beforeInspection.status === "valid" ? beforeInspection.projection : null
    sync()
    const comparison = captured ? compareAuthSession(captured, beforeInspection) : null
    if (!captured || !before || !comparison || (comparison.status !== "exact" && comparison.status !== "profile_advanced")) {
      return Boolean(before?.snapshot.accessToken)
    }
    const key = `${before.snapshot.sessionId}:${before.snapshot.credentialRevision}`
    const now = Date.now()
    if (lastCheck.key === key && now - lastCheck.at < 15_000) return true
    const operation = ++operationRef.current
    try {
      const response = await apiRequest(`${getApiUrl()}/api/auth/verify`, { headers: { "X-Username": before.cache.user.username } })
      const current = currentProjection()
      if (mountedRef.current && operation === operationRef.current) { setProjection(current); setLastCheck({ key, at: now }) }
      if (!current) return false
      if (current.snapshot.sessionId !== captured.sessionId) return Boolean(current.snapshot.accessToken)
      if (!response.ok && response.status === 401 && response.headers.get("Error-Type") === "InvalidToken") {
        const cleared = await clearAuthSessionIfCurrent(captured)
        const after = currentProjection()
        if (mountedRef.current && operation === operationRef.current) setProjection(after)
        return cleared.status !== "cleared" && Boolean(after?.snapshot.accessToken)
      }
      return Boolean(current.snapshot.accessToken)
    } catch (error) {
      console.error("Auth check error:", error)
      const current = currentProjection()
      if (mountedRef.current && operation === operationRef.current) setProjection(current)
      return Boolean(current?.snapshot.accessToken)
    }
  }, [lastCheck, projection, sync])
  const refreshAccessToken = useCallback(async (expected = projection?.snapshot ?? EMPTY_SESSION) => {
    const operation = ++operationRef.current
    const result = await refreshStoredAccessToken(expected)
    const current = currentProjection()
    if (mountedRef.current && operation === operationRef.current) setProjection(current)
    switch (result.status) {
      case "refreshed":
      case "advanced":
        return Boolean(current && current.snapshot.sessionId === expected.sessionId && current.snapshot.accessToken === result.accessToken)
      case "rejected":
        await clearAuthSessionIfCurrent(expected)
        break
      case "unavailable":
        if (mountedRef.current && operation === operationRef.current && isCurrentCredentialLineage(current, expected)) {
          toast.error(t(authMutationUnavailableTranslationKey(result.reason)), { id: AUTH_REFRESH_UNAVAILABLE_TOAST_ID })
        }
        break
      case "not_current":
      case "invalid_response":
      case "transport_failed":
        break
    }
    const after = currentProjection()
    if (mountedRef.current && operation === operationRef.current) setProjection(after)
    return false
  }, [projection, t])
  refreshRef.current = refreshAccessToken
  return <AuthContext.Provider value={{ user, isAuthenticated: Boolean(user && token), token, refreshToken, session, isLoading, inTeam, teamRole, login, logout, checkAuth, refreshAccessToken }}>{children}</AuthContext.Provider>
}
/** Provides a stable anonymous projection for public routes without reading browser auth state. */
export function AnonymousAuthProvider({ children }: { children: ReactNode }) {
  return <AuthContext.Provider value={ANONYMOUS_AUTH_CONTEXT}>{children}</AuthContext.Provider>
}
export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error("useAuth must be used within an AuthProvider")
  return context
}

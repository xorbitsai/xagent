import type { AuthMutationUnavailableReason } from "./auth-cache"
import type { TranslationKey } from "@/i18n/translations"

export const AUTH_PUBLIC_PATHS = [
  "/login",
  "/register",
  "/setup",
  "/forgot-password",
  "/reset-password",
  "/auth/oidc/callback",
] as const

const AUTH_MUTATION_UNAVAILABLE_TRANSLATION_KEYS: Record<AuthMutationUnavailableReason, Extract<TranslationKey,
  "login.alerts.storage_unavailable" | "login.alerts.coordination_unavailable" | "login.alerts.operation_failed"
>> = {
  storage_unavailable: "login.alerts.storage_unavailable",
  coordination_unavailable: "login.alerts.coordination_unavailable",
  operation_failed: "login.alerts.operation_failed",
}

/** Maps auth-domain availability reasons to the localized presentation contract. */
export function authMutationUnavailableTranslationKey(reason: AuthMutationUnavailableReason) {
  return AUTH_MUTATION_UNAVAILABLE_TRANSLATION_KEYS[reason]
}

export function isExternalRoutePath(pathname: string | null): boolean {
  return pathname === "/widget"
    || pathname?.startsWith("/widget/") === true
    || pathname === "/share"
    || pathname?.startsWith("/share/") === true
}

export function isAuthPublicPath(pathname: string | null): boolean {
  if (!pathname) {
    return false
  }
  if (isExternalRoutePath(pathname)) {
    return true
  }
  return AUTH_PUBLIC_PATHS.includes(pathname as (typeof AUTH_PUBLIC_PATHS)[number])
}

/** The single source of truth for the onboarding route path - both
 * auth-guard.tsx's redirect target and CHROMELESS_AUTHENTICATED_PATHS below
 * derive from this, instead of each independently hardcoding "/onboarding"
 * and risking a silent divergence if the route is ever renamed. */
export const ONBOARDING_PATH = "/onboarding"

/** Paths that render full-screen with no sidebar/app chrome, but - unlike
 * AUTH_PUBLIC_PATHS - still require a real authenticated session (AuthGuard
 * still runs its login redirect for them). Kept separate from
 * isAuthPublicPath so a page here is never accidentally treated as not
 * requiring login. */
export const CHROMELESS_AUTHENTICATED_PATHS = [ONBOARDING_PATH] as const

export function isChromelessAuthenticatedPath(pathname: string | null): boolean {
  if (!pathname) {
    return false
  }
  return CHROMELESS_AUTHENTICATED_PATHS.includes(
    pathname as (typeof CHROMELESS_AUTHENTICATED_PATHS)[number]
  )
}

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

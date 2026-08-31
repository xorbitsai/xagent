"use client"

import React, { useEffect, useRef, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { isAuthPublicPath, ONBOARDING_PATH } from "@/lib/auth-pages"
import { useRouter, usePathname } from "next/navigation"
import { useI18n } from "@/contexts/i18n-context"
import { getBrandingFromEnv } from "@/lib/branding"
import { consumeOnboardingSaveEscapeFlag, fetchUserPreferences } from "@/lib/user-preferences"

const branding = getBrandingFromEnv()

interface AuthGuardProps {
  children: React.ReactNode
}

export function AuthGuard({ children }: AuthGuardProps) {
  const { isAuthenticated, isLoading, checkAuth, user } = useAuth()
  const router = useRouter()
  const pathname = usePathname()
  const [mounted, setMounted] = useState(false)

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  // Don't protect login and register pages
  const isAuthPage = isAuthPublicPath(pathname)

  useEffect(() => {
    if (!mounted || isAuthPage) return

    if (!isLoading && !isAuthenticated) {
      router.push("/login")
    }
  }, [isAuthenticated, isLoading, router, mounted, isAuthPage])

  // On the normal (non-escape) path below, only latches once onboarding is
  // CONFIRMED complete for this user, not merely once a check has run - a
  // PR review finding caught that latching on any successfully-resolved
  // check (including a "not onboarded" one) let a user who never actually
  // finishes the wizard bypass every future check for the rest of the
  // session: redirected once to /onboarding, then instead of completing
  // it, a browser Back (or any other client-side nav to a DIFFERENT
  // already-visited protected route - this component doesn't remount on
  // client-side navigation) lands on a page whose effect re-runs, sees the
  // ref already latched to this user id, and skips the check entirely,
  // rendering protected children with onboarding never actually done.
  // Re-checking on every route change until it's actually confirmed true
  // is the correct tradeoff here (an extra GET per route while genuinely
  // not onboarded, which is expected to be a short-lived state) over ever
  // latching on an unconfirmed outcome. The escape-flag branch above is a
  // separate, deliberate exception to this - it latches unconditionally on
  // its own one-time bypass signal, not on an ordinary "not onboarded"
  // resolution, so it doesn't reintroduce the same problem; see its own
  // comment for why.
  //
  // The ref only latches once the check actually finishes (not before the
  // await) - if a dependency changes and cancels this run first (e.g. the
  // user navigates again while the GET is still in flight), `active` goes
  // false and the ref is left untouched, so the next run (with the new
  // deps) retries instead of the check being silently disarmed forever.
  //
  // Keyed by user id, not a bare boolean: AuthGuard doesn't remount across
  // a client-side auth swap - AuthProvider's own `storage`-event listener
  // replaces `isAuthenticated`/`user` (a React state update, not an
  // in-place mutation) when a DIFFERENT user logs in from another tab
  // (same-origin localStorage change), so a bare "have we
  // ever checked" boolean would stay latched from the PREVIOUS user's check
  // and let the new one through with no check of their own at all. Storing
  // whose check last completed, and comparing against the CURRENT user's id
  // instead, forces a fresh check whenever the identity actually changes.
  const checkedOnboardingUserIdRef = useRef<string | null>(null)
  useEffect(() => {
    if (!mounted || isAuthPage || pathname === ONBOARDING_PATH) return

    // Consumed unconditionally, AHEAD of the "already checked" guard below -
    // a previous version of this fix checked the flag only inside the async
    // check itself, which the ref guard skips entirely once a check has
    // already run this app-load (the common case: the user usually reached
    // /onboarding via an earlier check on some OTHER page that already
    // latched this ref). That left the flag unconsumed on the escape it was
    // meant for, and let it linger to wrongly suppress some unrelated LATER
    // onboarding check instead - a real regression an earlier round shipped.
    // Reading it first, before any early return, guarantees it's consumed
    // the very next time this effect runs after being set, regardless of
    // which branch would otherwise apply.
    if (consumeOnboardingSaveEscapeFlag(user?.id ?? null)) {
      checkedOnboardingUserIdRef.current = user?.id ?? null
      return
    }

    if (isLoading || !isAuthenticated || !user || checkedOnboardingUserIdRef.current === user.id) return

    let active = true
    void (async () => {
      const preferences = await fetchUserPreferences()
      if (!active) return
      // null means the fetch itself failed - "unknown," not "confirmed not
      // onboarded." Leave the ref unlatched so the next navigation retries
      // instead of redirecting an already-onboarded user on a transient error.
      if (preferences === null) return
      // Strict `!== true`, not `!preferences.onboarded`: the GET boundary
      // doesn't validate this field's type, so a malformed stored value
      // (e.g. a string "false", which is truthy) must not read as "already
      // onboarded" - only an explicit boolean true should ever skip the redirect.
      if (preferences.onboarded !== true) {
        // Deliberately NOT latching the ref here (see the comment on the
        // ref's declaration) - `replace` alone only prevents a Back press
        // from returning to THIS specific page (its history entry is gone),
        // it does nothing about navigating to some OTHER already-visited
        // protected route, which this effect will still see on its own
        // pathname change and must actually re-check.
        router.replace(ONBOARDING_PATH)
        return
      }
      checkedOnboardingUserIdRef.current = user.id
    })()
    return () => {
      active = false
    }
  }, [mounted, isAuthPage, pathname, isLoading, isAuthenticated, user, router])

  useEffect(() => {
    // Reduce check frequency, only check when user is active
    let checkTimeout: NodeJS.Timeout
    let retryCount = 0
    const maxRetries = 2 // Reduce retry count
    const checkInterval = 15 * 60 * 1000 // Check every 15 minutes instead of 5 minutes

    const scheduleNextCheck = () => {
      checkTimeout = setTimeout(async () => {
        if (isAuthenticated && !isAuthPage) {
          try {
            const isValid = await checkAuth()
            if (isValid) {
              retryCount = 0 // Reset retry count
            } else {
              retryCount++
              if (retryCount >= maxRetries) {
                console.warn(`Authentication failed after ${maxRetries} retries, logging out...`)
                router.push("/login")
              } else {
                console.warn(`Authentication check failed, retry ${retryCount}/${maxRetries}`)
              }
            }
          } catch (error) {
            console.error('Authentication check error:', error)
            retryCount++
            if (retryCount >= maxRetries) {
              console.warn(`Authentication error after ${maxRetries} retries, logging out...`)
              router.push("/login")
            }
          }
        }
        scheduleNextCheck() // Schedule next check
      }, checkInterval)
    }

    // Only start checking when user is active
    if (isAuthenticated && !isAuthPage) {
      scheduleNextCheck()
    }

    return () => {
      if (checkTimeout) {
        clearTimeout(checkTimeout)
      }
    }
  }, [isAuthenticated, checkAuth, router, isAuthPage])

  // For auth pages (login/register), just render children without protection
  if (isAuthPage) {
    return <>{children}</>
  }

  const { t } = useI18n()
  if (isLoading) {
    // Auth only resolves client-side, so static export (next build) freezes
    // this branch's JSX into out/index.html for the home route. Non-JS
    // fetchers of that frozen HTML — including automated brand-verification
    // crawlers — see only this markup, so it carries real copy about what
    // the app does instead of a bare spinner. A JS-executing crawler that
    // runs the redirect effect above will move past it once auth resolves.
    //
    // Logo and title/subtitle typography intentionally match the real hero in
    // page.tsx (logo size, font size/weight, line-height) so the swap to the
    // authenticated page is visually seamless. The decorative gradient
    // background is deliberately not replicated here.
    if (pathname === "/") {
      return (
        <div className="min-h-screen bg-[#0D1117] flex items-center justify-center px-6">
          <div className="text-center max-w-xl">
            <img
              src={branding.whiteLogoPath}
              alt={branding.appName}
              className="w-14 h-14 mb-6 object-contain rounded-[16px] shadow-2xl mx-auto"
            />
            <h1 className="text-white text-[34px] font-extrabold mb-3 tracking-tight leading-[1.15]">
              {t('home.hero.title', { appName: branding.appName })}
            </h1>
            <p className="text-gray-400 text-[13.5px] font-medium mb-8 leading-[1.7]">
              {t('home.hero.subtitle')}
            </p>
            <div className="w-8 h-8 border-2 border-[#8B949E] border-t-transparent rounded-full animate-spin mx-auto"></div>
          </div>
        </div>
      )
    }

    return (
      <div className="min-h-screen bg-[#0D1117] flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-[#8B949E] border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-[#8B949E]">{t('common.loading')}</p>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return null // Will redirect to login
  }

  return <>{children}</>
}

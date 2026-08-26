"use client"

import React, { useEffect, useRef, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { isAuthPublicPath } from "@/lib/auth-pages"
import { useRouter, usePathname } from "next/navigation"
import { useI18n } from "@/contexts/i18n-context"
import { getBrandingFromEnv } from "@/lib/branding"
import { fetchUserPreferences } from "@/lib/user-preferences"

const ONBOARDING_PATH = "/onboarding"

const branding = getBrandingFromEnv()

interface AuthGuardProps {
  children: React.ReactNode
}

export function AuthGuard({ children }: AuthGuardProps) {
  const { isAuthenticated, isLoading, checkAuth } = useAuth()
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

  // Checked once per app load (this component doesn't remount on client-side
  // navigation), not on every route change - a stale read only matters until
  // the user next completes or skips onboarding, at which point every exit
  // path there PATCHes onboarded:true before leaving, so this won't loop.
  //
  // The ref only latches once the check actually finishes (not before the
  // await) - if a dependency changes and cancels this run first (e.g. the
  // user navigates again while the GET is still in flight), `active` goes
  // false and the ref is left untouched, so the next run (with the new
  // deps) retries instead of the check being silently disarmed forever.
  const checkedOnboardingRef = useRef(false)
  useEffect(() => {
    if (!mounted || isAuthPage || pathname === ONBOARDING_PATH) return
    if (isLoading || !isAuthenticated || checkedOnboardingRef.current) return

    let active = true
    void (async () => {
      const preferences = await fetchUserPreferences()
      if (!active) return
      checkedOnboardingRef.current = true
      if (!preferences.onboarded) {
        router.push(ONBOARDING_PATH)
      }
    })()
    return () => {
      active = false
    }
  }, [mounted, isAuthPage, pathname, isLoading, isAuthenticated, router])

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

"use client"

import { useEffect, useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { isAuthPublicPath } from "@/lib/auth-pages"
import { useRouter, usePathname } from "next/navigation"
import { useI18n } from "@/contexts/i18n-context"

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

  useEffect(() => {
    // 降低检查频率，只在用户活跃时检查
    let checkTimeout: NodeJS.Timeout
    let retryCount = 0
    const maxRetries = 2 // 减少重试次数
    const checkInterval = 15 * 60 * 1000 // 15分钟检查一次，而不是5分钟

    const scheduleNextCheck = () => {
      checkTimeout = setTimeout(async () => {
        if (isAuthenticated && !isAuthPage) {
          try {
            const isValid = await checkAuth()
            if (isValid) {
              retryCount = 0 // 重置重试计数
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
        scheduleNextCheck() // 安排下一次检查
      }, checkInterval)
    }

    // 只在用户活跃时开始检查
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

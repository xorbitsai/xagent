"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { getApiUrl } from "@/lib/utils"
import { getBrandingFromEnv } from "@/lib/branding"
import { apiRequest } from "@/lib/api-wrapper"
import {
  ArrowRight,
  Loader2,
  Eye,
  EyeOff,
  Workflow,
  Database,
  UserCheck,
  User,
  Lock
} from "lucide-react"
import Link from "next/link"
import { useI18n } from "@/contexts/i18n-context"
import { useSetupStatus } from "@/hooks/use-setup-status"
import { AuthPageShell } from "@/components/auth/auth-page-shell"
import { AuthFormCard } from "@/components/auth/auth-form-card"
import { AUTH_CACHE_KEY } from "@/lib/auth-cache"

export function LoginPage() {
  const branding = getBrandingFromEnv()
  const { t } = useI18n()
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [formData, setFormData] = useState({
    identifier: "",
    password: ""
  })

  const { isLoading: isStatusLoading, registrationEnabled } = useSetupStatus({
    redirectToSetupIfNeeded: true,
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setIsLoading(true)

    try {
      const response = await apiRequest(`${getApiUrl()}/api/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username: formData.identifier, password: formData.password }),
      })

      if (response.ok) {
        const data = await response.json()
        const userData = {
          id: data.user.id,
          username: data.user.username,
          email: data.user.email,
          is_admin: data.user.is_admin,
        }

        // Store token in localStorage using the same keys as AuthContext
        localStorage.setItem("auth_token", data.access_token)
        localStorage.setItem("auth_user", JSON.stringify(userData))

        // Also update the new cache format
        localStorage.setItem(AUTH_CACHE_KEY, JSON.stringify({
          user: userData,
          token: data.access_token,
          refreshToken: data.refresh_token,
          expiresAt: Date.now() + (data.expires_in || 1800) * 1000, // 30 minutes default
          refreshExpiresAt: Date.now() + (data.refresh_expires_in || 604800) * 1000, // 7 days default
          timestamp: Date.now()
        }))

        // Redirect to home on success
        window.location.href = "/"
      } else {
        setError(t("login.alerts.auth_failed"))
      }
    } catch (error) {
      console.error("Login failed:", error)
      setError(t("login.alerts.network_failed"))
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }))
    // Clear error when user starts typing
    if (error) setError("")
  }

  const features = [
    {
      icon: Workflow,
      title: t("login.features.version_control.title"),
      description: t("login.features.version_control.description")
    },
    {
      icon: Database,
      title: t("login.features.team.title"),
      description: t("login.features.team.description")
    },
    {
      icon: UserCheck,
      title: t("login.features.automation.title"),
      description: t("login.features.automation.description")
    }
  ]

  return (
    <AuthPageShell
      appName={branding.appName}
      logoPath={branding.logoPath}
      logoAlt={branding.logoAlt}
      heroTitle={process.env.NEXT_PUBLIC_APP_TAGLINE ? branding.tagline.replace(". ", ".\n") : t("branding.tagline")}
      leftDescription={t("branding.hero_description")}
      mobileSubtitle={t("login.mobile_title")}
      features={features}
    >
      <AuthFormCard
        appName={branding.appName}
        logoPath={branding.logoPath}
        logoAlt={branding.logoAlt}
        modeLabel={t("nav.login")}
        showSocialLogin={false}
        title={t("login.title", { appName: branding.appName })}
        description={t("login.description")}
        footer={
          isStatusLoading ? null : registrationEnabled ? (
            <>
              {t("login.register_prompt")}{" "}
              <Link href="/register" className="font-semibold text-[#3155F6] hover:text-[#2447D8]">
                {t("login.register_link")}
              </Link>
            </>
          ) : (
            <span>{t("login.register_closed")}</span>
          )
        }
      >
        <form onSubmit={handleSubmit} className="space-y-5">
          {error ? (
            <div className="rounded-[16px] border border-[#FFD5D9] bg-[#FFF5F6] px-4 py-3">
              <p className="text-sm text-[#C53030]">{error}</p>
            </div>
          ) : null}

          <div className="space-y-2">
            <label className="block text-sm font-semibold text-[#4A5365]">
              {t("login.form.username")}
            </label>
            <div className="relative">
              <User className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
              <Input
                type="text"
                name="identifier"
                value={formData.identifier}
                onChange={handleInputChange}
                placeholder={t("login.form.username_placeholder")}
                className="h-12 rounded-[14px] border-[#E2E8F3] bg-white pl-11 pr-4 text-[#171A2F] placeholder:text-[#A0A9B8] shadow-[0_1px_2px_rgba(16,24,40,0.04)] focus-visible:border-[#5B7CFF] focus-visible:ring-[#5B7CFF]/20"
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-semibold text-[#4A5365]">
              {t("login.form.password")}
            </label>
            <div className="relative">
              <Lock className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
              <Input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder={t("login.form.password_placeholder")}
                className="h-12 rounded-[14px] border-[#E2E8F3] bg-white pl-11 pr-11 text-[#171A2F] placeholder:text-[#A0A9B8] shadow-[0_1px_2px_rgba(16,24,40,0.04)] focus-visible:border-[#5B7CFF] focus-visible:ring-[#5B7CFF]/20"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-[#A0A9B8] transition-colors hover:text-[#4A5365]"
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between gap-4 text-sm">
            <label className="flex items-center gap-2 text-[#7B8496]">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border border-[#D7DDE8] text-[#3155F6] focus:ring-[#3155F6]/20"
              />
              <span>{t("login.options.remember_me")}</span>
            </label>
            <Link href="/forgot-password" className="font-semibold text-[#4E63C9] transition-colors hover:text-[#3155F6]">
              {t("login.options.forgot_password")}
            </Link>
          </div>

          <Button
            type="submit"
            disabled={!formData.identifier || !formData.password || isLoading}
            className="h-12 w-full rounded-[14px] bg-[linear-gradient(180deg,#4B6BFF_0%,#2F54EB_100%)] text-base font-semibold text-white shadow-[0_14px_30px_rgba(47,84,235,0.32)] transition-all hover:translate-y-[-1px] hover:opacity-95 disabled:translate-y-0 disabled:opacity-60"
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                {t("login.form.submitting")}
              </span>
            ) : (
              <span className="flex items-center gap-2">
                {t("login.form.submit")}
                <ArrowRight className="h-4 w-4" />
              </span>
            )}
          </Button>
        </form>
      </AuthFormCard>
    </AuthPageShell>
  )
}

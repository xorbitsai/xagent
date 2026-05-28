"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { getApiUrl } from "@/lib/utils"
import { getBrandingFromEnv } from "@/lib/branding"
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
import { apiRequest } from "@/lib/api-wrapper"
import { useSetupStatus } from "@/hooks/use-setup-status"
import { AuthPageShell } from "@/components/auth/auth-page-shell"
import { AuthFormCard } from "@/components/auth/auth-form-card"

export function RegisterPage() {
  const branding = getBrandingFromEnv()
  const { t } = useI18n()
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")
  const [formData, setFormData] = useState({
    username: "",
    password: "",
    confirmPassword: ""
  })

  const { isLoading: isStatusLoading } = useSetupStatus({
    redirectToSetupIfNeeded: true,
    redirectToLoginIfRegistrationClosed: true,
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setSuccess("")

    // Verify password match
    if (formData.password !== formData.confirmPassword) {
      setError(t("register.alerts.password_mismatch"))
      return
    }

    // Verify password length
    if (formData.password.length < 6) {
      setError(t("register.alerts.password_too_short"))
      return
    }

    setIsLoading(true)

    try {
      const response = await apiRequest(`${getApiUrl()}/api/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username: formData.username, password: formData.password }),
      })

      const data = await response.json()

      if (response.ok && data.success) {
        setSuccess(t("register.alerts.success"))
        setTimeout(() => {
          window.location.href = "/login"
        }, 2000)
      } else {
        setError(data.message || t("register.alerts.failed"))
      }
    } catch (error) {
      console.error("Registration failed:", error)
      setError(t("register.alerts.failed_retry"))
    } finally {
      setIsLoading(false)
    }
  }

  if (isStatusLoading) {
    return null
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }))
    // Clear error message
    if (error) setError("")
    if (success) setSuccess("")
  }

  const features = [
    {
      icon: Workflow,
      title: t("register.features.vbd.title"),
      description: t("register.features.vbd.description"),
    },
    {
      icon: Database,
      title: t("register.features.hitl.title"),
      description: t("register.features.hitl.description"),
    },
    {
      icon: UserCheck,
      title: t("register.features.timetravel.title"),
      description: t("register.features.timetravel.description"),
    },
  ]

  return (
    <AuthPageShell
      appName={branding.appName}
      logoPath={branding.logoPath}
      logoAlt={branding.logoAlt}
      heroTitle={process.env.NEXT_PUBLIC_APP_TAGLINE ? branding.tagline.replace(". ", ".\n") : t("branding.tagline")}
      leftDescription={t("branding.hero_description")}
      mobileSubtitle={t("register.mobile_title")}
      features={features}
    >
      <AuthFormCard
        appName={branding.appName}
        logoPath={branding.logoPath}
        logoAlt={branding.logoAlt}
        modeLabel="Register"
        showSocialLogin={false}
        title={t("register.title", { appName: branding.appName })}
        description={t("register.description")}
        footer={
          <>
            {t("register.login_hint.has_account")}{" "}
            <Link href="/login" className="font-semibold text-[#3155F6] hover:text-[#2447D8]">
              {t("register.login_hint.login_now")}
            </Link>
          </>
        }
      >
        <form onSubmit={handleSubmit} className="space-y-5">
          {error ? (
            <div className="rounded-[16px] border border-[#FFD5D9] bg-[#FFF5F6] px-4 py-3">
              <p className="text-sm text-[#C53030]">{error}</p>
            </div>
          ) : null}

          {success ? (
            <div className="rounded-[16px] border border-[#C8F1DA] bg-[#F1FFF7] px-4 py-3">
              <p className="text-sm text-[#137A47]">{success}</p>
            </div>
          ) : null}

          <div className="space-y-2">
            <label className="block text-sm font-semibold text-[#4A5365]">
              {t("register.form.username")}
            </label>
            <div className="relative">
              <User className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
              <Input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleInputChange}
                placeholder={t("register.form.username_placeholder")}
                className="h-12 rounded-[14px] border-[#E2E8F3] bg-white pl-11 pr-4 text-[#171A2F] placeholder:text-[#A0A9B8] shadow-[0_1px_2px_rgba(16,24,40,0.04)] focus-visible:border-[#5B7CFF] focus-visible:ring-[#5B7CFF]/20"
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-semibold text-[#4A5365]">
              {t("register.form.password")}
            </label>
            <div className="relative">
              <Lock className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
              <Input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder={t("register.form.password_placeholder")}
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

          <div className="space-y-2">
            <label className="block text-sm font-semibold text-[#4A5365]">
              {t("register.form.confirm_password")}
            </label>
            <div className="relative">
              <Lock className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
              <Input
                type={showConfirmPassword ? "text" : "password"}
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleInputChange}
                placeholder={t("register.form.confirm_password_placeholder")}
                className="h-12 rounded-[14px] border-[#E2E8F3] bg-white pl-11 pr-11 text-[#171A2F] placeholder:text-[#A0A9B8] shadow-[0_1px_2px_rgba(16,24,40,0.04)] focus-visible:border-[#5B7CFF] focus-visible:ring-[#5B7CFF]/20"
                required
              />
              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-[#A0A9B8] transition-colors hover:text-[#4A5365]"
                aria-label={showConfirmPassword ? "Hide password" : "Show password"}
              >
                {showConfirmPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>

          <Button
            type="submit"
            disabled={!formData.username || !formData.password || !formData.confirmPassword || isLoading}
            className="h-12 w-full rounded-[14px] bg-[linear-gradient(180deg,#4B6BFF_0%,#2F54EB_100%)] text-base font-semibold text-white shadow-[0_14px_30px_rgba(47,84,235,0.32)] transition-all hover:translate-y-[-1px] hover:opacity-95 disabled:translate-y-0 disabled:opacity-60"
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                {t("register.form.submitting")}
              </span>
            ) : (
              <span className="flex items-center gap-2">
                {t("register.form.submit")}
                <ArrowRight className="h-4 w-4" />
              </span>
            )}
          </Button>
        </form>
      </AuthFormCard>
    </AuthPageShell>
  )
}

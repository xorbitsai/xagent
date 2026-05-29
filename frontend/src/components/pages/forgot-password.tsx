"use client"

import { useState } from "react"
import Link from "next/link"
import { ArrowRight, Loader2, Mail, Workflow, Database, UserCheck } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { getApiUrl } from "@/lib/utils"
import { getBrandingFromEnv } from "@/lib/branding"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { AuthPageShell } from "@/components/auth/auth-page-shell"
import { AuthFormCard } from "@/components/auth/auth-form-card"

export function ForgotPasswordPage() {
  const branding = getBrandingFromEnv()
  const { t } = useI18n()
  const [email, setEmail] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setSuccess("")

    if (!/\S+@\S+\.\S+/.test(email)) {
      setError(t("forgotPassword.alerts.invalid_email"))
      return
    }

    setIsLoading(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/auth/forgot-password`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email }),
      })
      const data = await response.json()

      if (response.ok && data.success) {
        setSuccess(data.message || t("forgotPassword.alerts.success"))
      } else {
        setError(data.message || data.detail || t("forgotPassword.alerts.failed"))
      }
    } catch (submitError) {
      console.error("Forgot password failed:", submitError)
      setError(t("forgotPassword.alerts.failed_retry"))
    } finally {
      setIsLoading(false)
    }
  }

  const features = [
    {
      icon: Workflow,
      title: t("login.features.version_control.title"),
      description: t("login.features.version_control.description"),
    },
    {
      icon: Database,
      title: t("login.features.team.title"),
      description: t("login.features.team.description"),
    },
    {
      icon: UserCheck,
      title: t("login.features.automation.title"),
      description: t("login.features.automation.description"),
    }
  ]

  return (
    <AuthPageShell
      appName={branding.appName}
      logoPath={branding.logoPath}
      logoAlt={branding.logoAlt}
      heroTitle={process.env.NEXT_PUBLIC_APP_TAGLINE ? branding.tagline.replace(". ", ".\n") : t("branding.tagline")}
      leftDescription={t("branding.hero_description")}
      mobileSubtitle={t("forgotPassword.mobile_title")}
      features={features}
    >
      <AuthFormCard
        appName={branding.appName}
        logoPath={branding.logoPath}
        logoAlt={branding.logoAlt}
        modeLabel={t("forgotPassword.mode_label")}
        showSocialLogin={false}
        title={t("forgotPassword.title", { appName: branding.appName })}
        description={t("forgotPassword.description")}
        footer={
          <>
            {t("forgotPassword.back_to_login")}{" "}
            <Link href="/login" className="font-semibold text-[#3155F6] hover:text-[#2447D8]">
              {t("nav.login")}
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
              {t("forgotPassword.form.email")}
            </label>
            <div className="relative">
              <Mail className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
              <Input
                type="email"
                name="email"
                value={email}
                onChange={(event) => {
                  setEmail(event.target.value)
                  if (error) setError("")
                  if (success) setSuccess("")
                }}
                placeholder={t("forgotPassword.form.email_placeholder")}
                className="h-12 rounded-[14px] border-[#E2E8F3] bg-white pl-11 pr-4 text-[#171A2F] placeholder:text-[#A0A9B8] shadow-[0_1px_2px_rgba(16,24,40,0.04)] focus-visible:border-[#5B7CFF] focus-visible:ring-[#5B7CFF]/20"
                required
              />
            </div>
          </div>

          <Button
            type="submit"
            disabled={!email || isLoading}
            className="h-12 w-full rounded-[14px] bg-[linear-gradient(180deg,#4B6BFF_0%,#2F54EB_100%)] text-base font-semibold text-white shadow-[0_14px_30px_rgba(47,84,235,0.32)] transition-all hover:translate-y-[-1px] hover:opacity-95 disabled:translate-y-0 disabled:opacity-60"
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                {t("forgotPassword.form.submitting")}
              </span>
            ) : (
              <span className="flex items-center gap-2">
                {t("forgotPassword.form.submit")}
                <ArrowRight className="h-4 w-4" />
              </span>
            )}
          </Button>
        </form>
      </AuthFormCard>
    </AuthPageShell>
  )
}

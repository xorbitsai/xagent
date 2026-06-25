"use client"

import { useEffect, useState } from "react"
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
  Lock,
  Mail,
  Building2,
  Phone,
  ChevronDown,
} from "lucide-react"
import Link from "next/link"
import { useI18n } from "@/contexts/i18n-context"
import { apiRequest } from "@/lib/api-wrapper"
import { useSetupStatus } from "@/hooks/use-setup-status"
import { AuthPageShell } from "@/components/auth/auth-page-shell"
import { AuthFormCard } from "@/components/auth/auth-form-card"
import { COUNTRIES, DEFAULT_COUNTRY } from "@/lib/countries"

export function RegisterPage() {
  const branding = getBrandingFromEnv()
  const { t } = useI18n()
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")
  const [showCountryDropdown, setShowCountryDropdown] = useState(false)
  const [selectedCountry, setSelectedCountry] = useState(DEFAULT_COUNTRY)
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
    firstName: "",
    lastName: "",
    organization: "",
    phone: "",
  })

  const { isLoading: isStatusLoading } = useSetupStatus({
    redirectToSetupIfNeeded: true,
    redirectToLoginIfRegistrationClosed: true,
  })

  // Auto-detect country from IP
  useEffect(() => {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 3000)

    const detectCountry = async () => {
      try {
        const response = await fetch("https://ipapi.co/json/", { signal: controller.signal })
        if (!response.ok) return
        const data = await response.json()
        const countryCode = data.country_code as string
        const found = COUNTRIES.find((c) => c.code === countryCode)
        if (found) setSelectedCountry(found)
      } catch {
        // silently fall back to default
      } finally {
        clearTimeout(timeoutId)
      }
    }
    void detectCountry()

    return () => {
      controller.abort()
      clearTimeout(timeoutId)
    }
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setSuccess("")

    if (formData.password !== formData.confirmPassword) {
      setError(t("register.alerts.password_mismatch"))
      return
    }

    if (!/\S+@\S+\.\S+/.test(formData.email)) {
      setError(t("register.alerts.invalid_email"))
      return
    }

    if (formData.password.length < 6) {
      setError(t("register.alerts.password_too_short"))
      return
    }

    setIsLoading(true)

    try {
      const trimmedPhone = formData.phone.trim()
      const phone = trimmedPhone
        ? `${selectedCountry.dialCode} ${trimmedPhone}`
        : undefined

      const response = await apiRequest(`${getApiUrl()}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: formData.username,
          email: formData.email,
          password: formData.password,
          first_name: formData.firstName || undefined,
          last_name: formData.lastName || undefined,
          organization: formData.organization || undefined,
          country: selectedCountry.name,
          phone,
        }),
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
    } catch (err) {
      console.error("Registration failed:", err)
      setError(t("register.alerts.failed_retry"))
    } finally {
      setIsLoading(false)
    }
  }

  if (isStatusLoading) {
    return null
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }))
    if (error) setError("")
    if (success) setSuccess("")
  }

  const isSubmitDisabled =
    !formData.username || !formData.email || !formData.password || !formData.confirmPassword || isLoading

  const inputClass =
    "h-12 rounded-[14px] border-[#E2E8F3] bg-white pl-11 pr-4 text-[#171A2F] placeholder:text-[#A0A9B8] shadow-[0_1px_2px_rgba(16,24,40,0.04)] focus-visible:border-[#5B7CFF] focus-visible:ring-[#5B7CFF]/20"

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
        modeLabel={t("nav.register")}
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
        <form onSubmit={handleSubmit} className="space-y-4">
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

          {/* Username */}
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
                className={inputClass}
                required
              />
            </div>
          </div>

          {/* Email */}
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-[#4A5365]">
              {t("register.form.email")}
            </label>
            <div className="relative">
              <Mail className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
              <Input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                placeholder={t("register.form.email_placeholder")}
                className={inputClass}
                required
              />
            </div>
          </div>

          {/* First name + Last name */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <label className="block text-sm font-semibold text-[#4A5365]">
                {t("register.form.first_name")}
              </label>
              <div className="relative">
                <User className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
                <Input
                  type="text"
                  name="firstName"
                  value={formData.firstName}
                  onChange={handleInputChange}
                  placeholder={t("register.form.first_name_placeholder")}
                  className={inputClass}
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-semibold text-[#4A5365]">
                {t("register.form.last_name")}
              </label>
              <div className="relative">
                <User className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
                <Input
                  type="text"
                  name="lastName"
                  value={formData.lastName}
                  onChange={handleInputChange}
                  placeholder={t("register.form.last_name_placeholder")}
                  className={inputClass}
                />
              </div>
            </div>
          </div>

          {/* Organization */}
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-[#4A5365]">
              {t("register.form.organization")}
            </label>
            <div className="relative">
              <Building2 className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-[#A0A9B8]" />
              <Input
                type="text"
                name="organization"
                value={formData.organization}
                onChange={handleInputChange}
                placeholder={t("register.form.organization_placeholder")}
                className={inputClass}
              />
            </div>
          </div>

          {/* Password */}
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
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Confirm password */}
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
                {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Country */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-semibold text-[#4A5365]">
                {t("register.form.country")}
              </label>
              <span className="text-xs text-[#A0A9B8]">{t("register.form.country_auto")}</span>
            </div>
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowCountryDropdown((v) => !v)}
                className="flex h-12 w-full items-center gap-3 rounded-[14px] border border-[#E2E8F3] bg-white px-4 text-sm text-[#171A2F] shadow-[0_1px_2px_rgba(16,24,40,0.04)] transition-colors hover:border-[#5B7CFF] focus:outline-none focus:border-[#5B7CFF]"
              >
                <span className="text-base">{selectedCountry.flag}</span>
                <span className="flex-1 text-left">{selectedCountry.name}</span>
                <ChevronDown className="h-4 w-4 text-[#A0A9B8]" />
              </button>

              {showCountryDropdown ? (
                <div className="absolute top-[calc(100%+4px)] left-0 z-50 w-full max-h-56 overflow-y-auto rounded-[14px] border border-[#E2E8F3] bg-white shadow-[0_8px_24px_rgba(16,24,40,0.12)]">
                  {COUNTRIES.map((country) => (
                    <button
                      key={country.code}
                      type="button"
                      onClick={() => {
                        setSelectedCountry(country)
                        setShowCountryDropdown(false)
                      }}
                      className="flex w-full items-center gap-3 px-4 py-2.5 text-sm text-[#171A2F] hover:bg-[#F7F9FC] text-left"
                    >
                      <span className="text-base">{country.flag}</span>
                      <span>{country.name}</span>
                      <span className="ml-auto text-xs text-[#A0A9B8]">{country.dialCode}</span>
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          </div>

          {/* Contact number (optional) */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-semibold text-[#4A5365]">
                {t("register.form.phone")}
              </label>
              <span className="text-xs text-[#A0A9B8]">{t("register.form.phone_optional")}</span>
            </div>
            <div className="flex h-12 overflow-hidden rounded-[14px] border border-[#E2E8F3] bg-white shadow-[0_1px_2px_rgba(16,24,40,0.04)] focus-within:border-[#5B7CFF]">
              <div className="flex shrink-0 items-center gap-1.5 border-r border-[#E2E8F3] px-3 text-sm text-[#4A5365]">
                <Phone className="h-4 w-4 text-[#A0A9B8]" />
                <span>{selectedCountry.dialCode}</span>
              </div>
              <input
                type="tel"
                name="phone"
                value={formData.phone}
                onChange={handleInputChange}
                placeholder="555 000 0000"
                className="flex-1 bg-transparent px-3 text-sm text-[#171A2F] placeholder:text-[#A0A9B8] outline-none"
              />
            </div>
          </div>

          <Button
            type="submit"
            disabled={isSubmitDisabled}
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

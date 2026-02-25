"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { getApiUrl } from "@/lib/utils"
import { getBrandingFromEnv } from "@/lib/branding"
import {
  Eye,
  EyeOff,
  UserPlus,
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

    // 验证密码匹配
    if (formData.password !== formData.confirmPassword) {
      setError(t("register.alerts.password_mismatch"))
      return
    }

    // 验证密码长度
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
    // 清除错误信息
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
      leftDescription={process.env.NEXT_PUBLIC_APP_TAGLINE ? branding.tagline : t("branding.tagline")}
      mobileSubtitle={t("register.mobile_title")}
      features={features}
    >
      <Card className="p-8 bg-background/10 backdrop-blur-lg border-border shadow-2xl">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-foreground mb-2">{t("register.title", { appName: branding.appName })}</h2>
                <p className="text-muted-foreground">{t("register.description")}</p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                {error && (
                  <Alert className="border-red-200 bg-red-50">
                    <AlertDescription className="text-red-800">
                      {error}
                    </AlertDescription>
                  </Alert>
                )}

                {success && (
                  <Alert className="border-green-200 bg-green-50">
                    <AlertDescription className="text-green-800">
                      {success}
                    </AlertDescription>
                  </Alert>
                )}

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    {t("register.form.username")}
                  </label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="text"
                      name="username"
                      value={formData.username}
                      onChange={handleInputChange}
                      placeholder={t("register.form.username_placeholder")}
                      className="pl-10 bg-background/10 border-border text-foreground placeholder:text-muted-foreground focus:border-primary"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    {t("register.form.password")}
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type={showPassword ? "text" : "password"}
                      name="password"
                      value={formData.password}
                      onChange={handleInputChange}
                      placeholder={t("register.form.password_placeholder")}
                      className="pl-10 pr-10 bg-background/10 border-border text-foreground placeholder:text-muted-foreground focus:border-primary"
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showPassword ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    {t("register.form.confirm_password")}
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type={showConfirmPassword ? "text" : "password"}
                      name="confirmPassword"
                      value={formData.confirmPassword}
                      onChange={handleInputChange}
                      placeholder={t("register.form.confirm_password_placeholder")}
                      className="pl-10 pr-10 bg-background/10 border-border text-foreground placeholder:text-muted-foreground focus:border-primary"
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
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
                  className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-medium py-3 transition-all duration-200 transform hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin"></div>
                      {t("register.form.submitting")}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <UserPlus className="h-4 w-4" />
                      {t("register.form.submit")}
                    </div>
                  )}
                </Button>
              </form>

              <div className="mt-8 text-center">
                <p className="text-muted-foreground">
                  {t("register.login_hint.has_account")} {" "}
                  <Link href="/login" className="text-muted-foreground hover:text-foreground font-medium">
                    {t("register.login_hint.login_now")}
                  </Link>
                </p>
              </div>
      </Card>

      <div className="mt-6 flex items-center justify-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-pulse"></div>
          <span className="text-muted-foreground">{t("register.status.agent_running")}</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-pulse"></div>
          <span className="text-muted-foreground">{t("register.status.open_register")}</span>
        </div>
      </div>
    </AuthPageShell>
  )
}

"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { getApiUrl } from "@/lib/utils"
import { getBrandingFromEnv } from "@/lib/branding"
import { apiRequest } from "@/lib/api-wrapper"
import {
  Eye,
  EyeOff,
  LogIn,
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
import { AUTH_CACHE_KEY } from "@/lib/auth-cache"

export function LoginPage() {
  const branding = getBrandingFromEnv()
  const { t } = useI18n()
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [formData, setFormData] = useState({
    username: "",
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
        body: JSON.stringify({ username: formData.username, password: formData.password }),
      })

      if (response.ok) {
        const data = await response.json()
        const userData = { id: data.user.id, username: data.user.username, is_admin: data.user.is_admin }

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

        // Redirect to dashboard on success
        window.location.href = "/task"
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
      leftDescription={process.env.NEXT_PUBLIC_APP_TAGLINE ? branding.tagline : t("branding.tagline")}
      mobileSubtitle={t("login.mobile_title")}
      features={features}
    >
      <Card className="p-8 bg-background/10 backdrop-blur-lg border-border shadow-2xl">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-foreground mb-2">{t("login.title", { appName: branding.appName })}</h2>
                <p className="text-muted-foreground">{t("login.description")}</p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                {error && (
                  <div className="p-3 rounded-lg bg-destructive/20 border border-destructive/50">
                    <p className="text-sm text-destructive-foreground">{error}</p>
                  </div>
                )}

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    {t("login.form.username")}
                  </label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="text"
                      name="username"
                      value={formData.username}
                      onChange={handleInputChange}
                      placeholder={t("login.form.username_placeholder")}
                      className="pl-10 bg-background/10 border-border text-foreground placeholder:text-muted-foreground focus:border-primary"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    {t("login.form.password")}
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type={showPassword ? "text" : "password"}
                      name="password"
                      value={formData.password}
                      onChange={handleInputChange}
                      placeholder={t("login.form.password_placeholder")}
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

                <div className="flex items-center justify-between text-sm">
                  <label className="flex items-center text-muted-foreground">
                    <input type="checkbox" className="rounded mr-2" />
                    {t("login.options.remember_me")}
                  </label>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    {t("login.options.forgot_password")}
                  </a>
                </div>

                <Button
                  type="submit"
                  disabled={!formData.username || !formData.password || isLoading}
                  className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-medium py-3 transition-all duration-200 transform hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin"></div>
                      {t("login.form.submitting")}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <LogIn className="h-4 w-4" />
                      {t("login.form.submit")}
                    </div>
                  )}
                </Button>
              </form>

              <div className="mt-8 text-center">
                <p className="text-muted-foreground">
                  {isStatusLoading ? null : registrationEnabled ? (
                    <>
                      {t("login.register_prompt")} {" "}
                      <Link href="/register" className="text-muted-foreground hover:text-foreground font-medium">
                        {t("login.register_link")}
                      </Link>
                    </>
                  ) : (
                    <>{t("login.register_closed")}</>
                  )}
                </p>
              </div>
      </Card>
    </AuthPageShell>
  )
}
